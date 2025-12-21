# file: train_style_disentangle.py

import json
from collections import deque, defaultdict
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import build_dataloader
from models.generator import UNetGenerator
from models.discriminator import PatchDiscriminator
from models.semantic_moco_jepa import SemanticMoCoJEPA, SemAugConfig
from models.losses_nce import (
    PatchNCELoss,
    fft_texture_loss,
    swd_loss_images,
    spectral_noise,
    highpass,
)
from training.scheduler import CycleScheduler
from training.checkpoint import (
    save_checkpoint,
    save_state_json,
    last_epoch,
    load_checkpoint,
)
from training.train_detection_transformer import train_detection_transformer
from helpers import (
    run_hybrid_supervised_epoch,
    _triplet_grid,
    _denorm,
    print_epoch_summary,
    train_step_phase_A,
    train_step_phase_B,
    run_sup_freeze_mode,
    new_epoch_meters,
    get_style_lambda,  # scheduler de λ_style_A
)


# =========================================================================================
#   Helper : résolution du gel de backbone (global + overrides spécifiques)
# =========================================================================================

def resolve_backbone_freeze(opt, mode: str) -> bool:
    """
    Résout le flag 'freeze_backbone' effectif en combinant :
      - opt.freeze_backbone (global)
      - opt.det_freeze_backbone (override pour detect_transformer)

    mode ∈ {"detect_transformer", ...}
    """
    global_flag = bool(getattr(opt, "freeze_backbone", 0))

    if mode == "detect_transformer":
        specific = getattr(opt, "det_freeze_backbone", None)
    else:
        specific = None

    if specific is None:
        return global_flag
    return bool(specific)


# =========================================================================================
#   Helper : parsing générique de save_freq (aligné avec la détection)
# =========================================================================================

def _parse_save_freq(save_freq_str):
    """
    Interprète une chaîne save_freq :
      - 'none'       -> ('none', None)
      - 'epoch'      -> ('epoch', 1)
      - 'epoch:5'    -> ('epoch', 5)
      - 'step'       -> ('step', 1)
      - 'step:1000'  -> ('step', 1000)
      - '1000'       -> ('step', 1000)
    """
    if save_freq_str is None:
        return "none", None
    sf = str(save_freq_str).strip().lower()
    if sf == "" or sf == "none":
        return "none", None

    if sf.startswith("epoch"):
        parts = sf.split(":", 1)
        if len(parts) == 2:
            try:
                n = int(parts[1])
                return "epoch", max(1, n)
            except ValueError:
                pass
        return "epoch", 1

    if sf.startswith("step"):
        parts = sf.split(":", 1)
        if len(parts) == 2:
            try:
                n = int(parts[1])
                return "step", max(1, n)
            except ValueError:
                pass
        return "step", 1

    # Si c'est juste un entier → interprété comme step:N
    try:
        n = int(sf)
        return "step", max(1, n)
    except ValueError:
        # Valeur non reconnue → désactive
        return "none", None


# =========================================================================================
#           Entraînement alterné (style + JEPA + supervision + détection)
# =========================================================================================

def train_alternating(opt):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Mode d'entraînement ---
    mode = getattr(opt, "mode", "auto")
    mode_help = {
        "auto": "A+B (self-supervised: style + JEPA)",
        "sup_freeze": "C seul (supervisé), G&D gelés",
        "hybrid": "A+B puis C (supervisé)",
        "detect_transformer": "Entraînement détection (tête choisie par det_head_type)",
    }

    print("\n" + "=" * 88)
    print(f"🎛️  MODE D'ENTRAÎNEMENT : {mode.upper()}  →  {mode_help.get(mode, '?')}")
    print("=" * 88)

    # =====================================================================================
    #  MODE: détection transformer (DETR-like / simple_unet) → on sort tout de suite
    # =====================================================================================
    if mode == "detect_transformer":
        # Résoudre le gel de backbone pour la détection
        freeze_det = resolve_backbone_freeze(opt, mode="detect_transformer")
        # On force l'option (utile si non passée en CLI)
        opt.det_freeze_backbone = int(freeze_det)
        print(f"[DET] det_freeze_backbone (effectif) = {freeze_det}")

        # Appel direct au script de détection (gère det_head_type, save_freq, etc.)
        return train_detection_transformer(opt, dev)

    # =====================================================================================
    #  MODES: auto / hybrid / sup_freeze  → pipeline style + JEPA
    # =====================================================================================

    # --- Dataloaders principaux A/B (style / auto / hybrid / sup_freeze)
    loaders = build_dataloader(opt)
    k_folds = len(loaders)
    if k_folds == 0:
        raise RuntimeError(
            "[data] Aucun dataloader construit pour les modes auto/hybrid/sup_freeze. "
            "Vérifie les options de dataset."
        )

    k_folds = max(1, k_folds)
    nbatchs = min(len(dl) for dl in loaders) if k_folds > 0 else 0

    # --- Générateurs / Discriminateurs (style)
    G_A, D_A = UNetGenerator(token_dim=256).to(dev), PatchDiscriminator().to(dev)
    G_B, D_B = UNetGenerator(token_dim=256).to(dev), PatchDiscriminator().to(dev)

    base_lr = float(getattr(opt, "lr", 2e-4))
    adv_lrD_mult = float(getattr(opt, "adv_lrD_mult", 0.5))
    opt_GA = torch.optim.Adam(G_A.parameters(), lr=base_lr, betas=(0.5, 0.999))
    opt_GB = torch.optim.Adam(G_B.parameters(), lr=base_lr, betas=(0.5, 0.999))
    opt_DA = torch.optim.Adam(
        D_A.parameters(), lr=base_lr * adv_lrD_mult, betas=(0.5, 0.999)
    )
    opt_DB = torch.optim.Adam(
        D_B.parameters(), lr=base_lr * adv_lrD_mult, betas=(0.5, 0.999)
    )

    writer = SummaryWriter(opt.save_dir) if getattr(opt, "tb", False) else None

    resume_epoch = None

    # --- Reprise éventuelle ---
    if getattr(opt, "resume_dir", None):
        e = last_epoch(opt.resume_dir, "G_B")
        if e is not None:
            load_checkpoint(
                opt.resume_dir,
                e,
                G_A,
                D_A,
                G_B,
                D_B,
                opt_GA,
                opt_DA,
                opt_GB,
                opt_DB,
                device=str(dev),
                strict=False,
            )
            resume_epoch = int(e)
            print(f"✓ reprise depuis epoch {e}")

    out_dir = Path(opt.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_cfg.json").write_text(
        json.dumps({"static_hparams": vars(opt)}, indent=2)
    )

    if writer:
        writer.add_text("run/mode", f"{mode} – {mode_help.get(mode, '')}", 0)
        writer.add_scalar("hparams/batch_size", float(opt.batch_size), 0)
        writer.add_scalar("hparams/lr", float(base_lr), 0)

    tb_freq = int(getattr(opt, "tb_freq", 100))
    tb_freq_C = int(getattr(opt, "tb_freq_C", tb_freq))

    # ------------------------- Gestion globale de save_freq -------------------------
    save_freq_mode, save_freq_interval = _parse_save_freq(
        getattr(opt, "save_freq", "epoch")
    )
    epoch_ckpt_interval = getattr(opt, "epoch_ckpt_interval", None)
    if save_freq_mode == "epoch" and epoch_ckpt_interval is not None:
        try:
            save_freq_interval = max(1, int(epoch_ckpt_interval))
        except Exception:
            pass

    print(f"[CKPT] save_freq = {getattr(opt, 'save_freq', 'epoch')} "
          f"→ mode={save_freq_mode}, interval={save_freq_interval}")

    # --- État global de l'entraînement ---
    state = {
        "G_A": G_A,
        "G_B": G_B,
        "D_A": D_A,
        "D_B": D_B,
        "opt_GA": opt_GA,
        "opt_GB": opt_GB,
        "opt_DA": opt_DA,
        "opt_DB": opt_DB,
        "epoch": 0,
        "global_step": 0,
        "replay": deque(maxlen=getattr(opt, "replay_size", 2000)),
        "style_bank": deque(maxlen=2000),
        # JEPA tokens (style / contenu)
        "tok_jepa_A_style": None,
        "tok_jepa_A_content": None,
        "tok_jepa_B_style": None,
        "tok_jepa_B_content": None,
        "λ_style_B_dyn": float(getattr(opt, "lambda_style_b", 0.5)),
    }

    import copy as _copy
    state["T_A"] = _copy.deepcopy(G_A).eval()
    for p in state["T_A"].parameters():
        p.requires_grad_(False)

    state["T_B"] = _copy.deepcopy(G_B).eval()
    for p in state["T_B"].parameters():
        p.requires_grad_(False)

    # =====================================================================================
    #  SEMANTIC CONTENT BRANCH (ResNet50 + MoCo + optional JEPA-content)
    #  + NEW: load external pretrained weights path (e.g., MoCo v3 resnet50 checkpoint)
    # =====================================================================================
    sem_enable = bool(getattr(opt, "sem_content", False) and mode == "auto")
    state["SEM"] = None
    state["opt_SEM"] = None
    if sem_enable:
        aug_cfg = SemAugConfig(
            use_aug=bool(getattr(opt, "sem_use_aug", False)),
            crop=int(getattr(opt, "sem_crop", 224)),
            min_scale=float(getattr(opt, "sem_min_scale", 0.5)),
            color_jitter=float(getattr(opt, "sem_color_jitter", 0.4)),
            gray_p=float(getattr(opt, "sem_gray", 0.2)),
            blur_p=float(getattr(opt, "sem_blur", 0.1)),
        )

        # Quand sem_content est actif, on redirige JEPA-content vers la branche sémantique
        jepa_on_content_raw = bool(
            getattr(opt, "jepa_on_content", 0) and getattr(opt, "jepa_tokens", False)
        )
        sem_jepa_on = bool(jepa_on_content_raw)

        # >>> NEW OPTIONS (semantic_moco_jepa.py updated):
        #   opt.sem_pretrained_path: path to external checkpoint (MoCo v3, etc.)
        #   opt.sem_pretrained_strict: strict load_state_dict for backbone (default 0)
        #   opt.sem_pretrained_verbose: print logs (default 1)
        sem_pretrained_path = getattr(opt, "sem_pretrained_path", None)
        sem_pretrained_strict = bool(int(getattr(opt, "sem_pretrained_strict", 0)) != 0)
        sem_pretrained_verbose = bool(int(getattr(opt, "sem_pretrained_verbose", 1)) != 0)

        sem_model = SemanticMoCoJEPA(
            dim=int(getattr(opt, "sem_dim", 256)),
            tok_dim=int(getattr(opt, "sem_tok_dim", 256)),
            queue_size=int(getattr(opt, "sem_queue", 65536)),
            m=float(getattr(opt, "sem_m", 0.999)),
            T=float(getattr(opt, "sem_t", 0.2)),
            pretrained=bool(int(getattr(opt, "sem_pretrained", 1)) != 0),
            pretrained_path=sem_pretrained_path,
            pretrained_strict=sem_pretrained_strict,
            pretrained_verbose=sem_pretrained_verbose,
            aug_cfg=aug_cfg,
            img_size=int(getattr(opt, "crop_size", 256)),
            jepa_use=sem_jepa_on,
            jepa_hidden_mult=int(getattr(opt, "jepa_hidden_mult", 2)),
            jepa_heads=int(getattr(opt, "jepa_heads", 4)),
            jepa_norm=int(getattr(opt, "jepa_norm", 1)),
            jepa_var=float(getattr(opt, "lambda_jepa_var", 0.05)),
            jepa_cov=float(getattr(opt, "lambda_jepa_cov", 0.05)),
        ).to(dev)

        # Optimiseur (query encoder + projections + JEPA predictor si actif)
        sem_params = [p for p in sem_model.parameters() if p.requires_grad]
        _lr_sem = getattr(opt, "lr_sem", None)
        sem_lr = float(_lr_sem) if _lr_sem is not None else float(getattr(opt, "lr", 2e-4))
        opt_sem = torch.optim.AdamW(
            sem_params, lr=sem_lr, betas=(0.9, 0.999), weight_decay=1e-4
        )

        state["SEM"] = sem_model
        state["opt_SEM"] = opt_sem

        print(
            f"[SEM] sem_content enabled | pretrained={getattr(opt, 'sem_pretrained', 1)} "
            f"| sem_pretrained_path={sem_pretrained_path} | strict={int(sem_pretrained_strict)}"
        )

    # --- Si reprise et SEM activé : charger aussi la branche sémantique ---
    if sem_enable and (resume_epoch is not None) and state.get("SEM", None) is not None:
        try:
            load_checkpoint(
                opt.resume_dir,
                resume_epoch,
                G_A, D_A, G_B, D_B,
                opt_GA, opt_DA, opt_GB, opt_DB,
                device=str(dev),
                strict=False,
                sem_model=state.get("SEM", None),
                opt_sem=state.get("opt_SEM", None),
            )
            print("✓ reprise SEM depuis checkpoint")
        except Exception as e:
            print(f"[WARN] reprise SEM impossible: {e}")

    # --- Config partagée pour helpers / phases ---
    cfg = {}
    cfg["device"] = dev
    cfg["opt"] = opt
    cfg["writer"] = writer
    cfg["tb_freq"] = tb_freq
    cfg["tb_freq_C"] = tb_freq_C

    cfg["l1_loss"] = torch.nn.L1Loss().to(dev)

    # --- Perte NCE patch ---
    cfg["nce_loss"] = PatchNCELoss(
        temperature=getattr(opt, "nce_t", 0.07),
        use_intra_neg=getattr(opt, "nce_intra", True),
        use_inter_neg=getattr(opt, "nce_inter", True),
        max_patches=getattr(opt, "nce_max_patches", None),
    )

    nce_layers = [
        l.strip()
        for l in str(getattr(opt, "nce_layers", "bot,skip64,skip32")).split(",")
        if l.strip()
    ]
    layer_w = [
        float(w)
        for w in (
            getattr(opt, "nce_layer_weights", None)
            or ",".join(["1"] * len(nce_layers))
        ).split(",")
    ]
    if len(layer_w) < len(nce_layers):
        layer_w = layer_w + [layer_w[-1]] * (len(nce_layers) - len(layer_w))
    layer_w = layer_w[: len(nce_layers)]
    sw = sum(layer_w) or 1.0
    layer_w = [w / sw for w in layer_w]

    cfg["nce_layers"] = nce_layers
    cfg["nce_layer_w"] = layer_w

    # --- Adversarial / R1 / highpass ---
    cfg["adv_enable_A"] = bool(getattr(opt, "adv_enable_A", True))
    cfg["adv_enable_B"] = bool(getattr(opt, "adv_enable_B", False))
    cfg["adv_type"] = str(getattr(opt, "adv_type", "hinge")).lower().strip()
    cfg["adv_r1_gamma"] = float(getattr(opt, "adv_r1_gamma", 10.0))
    cfg["adv_r1_every"] = int(getattr(opt, "adv_r1_every", 16))
    cfg["adv_highpass"] = bool(getattr(opt, "adv_highpass", False))
    cfg["highpass_fn"] = highpass if "highpass" in globals() else None

    # --- Texture (FFT / SWD) ---
    cfg["tex_enable"] = bool(getattr(opt, "tex_enable", 0))
    cfg["tex_sigma"] = float(getattr(opt, "tex_sigma", 0.0))
    cfg["tex_gamma"] = float(getattr(opt, "tex_gamma", 1.0))
    cfg["tex_use_fft"] = bool(getattr(opt, "tex_use_fft", 0))
    cfg["tex_use_swd"] = bool(getattr(opt, "tex_use_swd", 0))
    cfg["lambda_fft"] = float(getattr(opt, "lambda_fft", 0.0))
    cfg["lambda_swd"] = float(getattr(opt, "lambda_swd", 0.0))
    cfg["swd_levels"] = getattr(opt, "swd_levels", "64")
    cfg["swd_patch"] = int(getattr(opt, "swd_patch", 64))
    cfg["swd_proj"] = int(getattr(opt, "swd_proj", 128))
    cfg["swd_max_patches"] = int(getattr(opt, "swd_max_patches", 64))
    cfg["tex_apply_A"] = bool(getattr(opt, "tex_apply_A", 0))

    # --- Lambdas NCE / régul par phase ---
    λ_nce_AADV = getattr(opt, "lambda_nce_a_adv", 0.0)
    λ_reg_AADV = getattr(opt, "lambda_reg_a_adv", 0.0)
    λ_nce_AMIX = getattr(opt, "lambda_nce_a_mix", 1.0)
    λ_reg_AMIX = getattr(opt, "lambda_reg_a_mix", 0.5)
    λ_nce_B = getattr(opt, "lambda_nce_b", 1.0)
    λ_idt_B = getattr(opt, "lambda_idt_b", getattr(opt, "lambda_reg_b", 10.0))
    # NCE supplémentaire sur contenu (invariance au style)
    cfg["content_nce_enable"] = bool(getattr(opt, "content_nce_enable", 0))
    cfg["lambda_content_nce"] = float(getattr(opt, "lambda_content_nce", 0.0))

    cfg["fft_texture_loss"] = fft_texture_loss
    cfg["swd_loss_images"] = swd_loss_images
    cfg["spectral_noise"] = spectral_noise

    # --- Style A / B + gains (avec scheduler pour λ_style_A) ---
    λ_style_max = float(getattr(opt, "style_lambda", 10.0))
    λ_style_min = float(getattr(opt, "style_lambda_min", λ_style_max))
    style_sched_type = str(getattr(opt, "style_lambda_sched", "none")).lower().strip()
    style_warmup_ep = int(getattr(opt, "style_lambda_warmup", 0))

    cfg["style_sched"] = {
        "lambda_max": λ_style_max,
        "lambda_min": λ_style_min,
        "type": style_sched_type,
        "warmup_epochs": style_warmup_ep,
        "T_total": int(getattr(opt, "epochs", 1)),
    }

    cfg["λ_style_A"] = λ_style_max
    cfg["style_gain_A"] = float(getattr(opt, "style_gain_A", cfg["λ_style_A"]))

    cfg["λ_spade"] = float(getattr(opt, "lambda_spade_gate", 0.05))
    cfg["spade_margin"] = float(getattr(opt, "spade_gate_margin", 0.75))

    cfg["λ_style_B_min"] = float(getattr(opt, "lambda_style_b_min", 0.0001))
    cfg["λ_style_B_max"] = float(getattr(opt, "lambda_style_b_max", 2.0))

    cfg["style_gain_B"] = float(getattr(opt, "style_gain_B", 1.0))

    cfg["style_B_warmup_ep"] = int(getattr(opt, "style_b_warmup_epochs", 1))
    cfg["style_balance_target"] = float(getattr(opt, "style_balance_target", 0.06))
    cfg["style_balance_alpha"] = float(getattr(opt, "style_balance_alpha", 0.10))
    cfg["lambda_nce_b"] = float(getattr(opt, "lambda_nce_b", 1.0))
    cfg["lambda_idt_b"] = float(
        getattr(opt, "lambda_idt_b", getattr(opt, "lambda_reg_b", 10.0))
    )

    # --- MixSwap (tokens / FFT) ---
    cfg["mixswap_enable"] = bool(getattr(opt, "mixswap_enable", 0))
    cfg["mixswap_token_p"] = float(getattr(opt, "mixswap_token_p", 1.0))
    cfg["mixswap_fft_p"] = float(getattr(opt, "mixswap_fft_p", 0.0))

    def _parse_range(s, dflt=(0.3, 0.7)):
        try:
            a, b = [float(x) for x in str(s).split(",")[:2]]
            if a > b:
                a, b = b, a
            return max(0.0, a), min(1.0, b)
        except Exception:
            return dflt

    mix_lo, mix_hi = _parse_range(getattr(opt, "mixswap_alpha", "0.3,0.7"))
    cfg["mixswap_alpha_lo"] = mix_lo
    cfg["mixswap_alpha_hi"] = mix_hi

    from training.texture_fft_swd import fft_amp_mix
    cfg["fft_amp_mix"] = fft_amp_mix

    # --- JEPA : config partagée (style + contenu) ---
    jepa_on_style = bool(getattr(opt, "jepa_on_style", 1) and mode == "auto")
    jepa_on_content = bool(getattr(opt, "jepa_on_content", 0) and mode == "auto" and (not sem_enable))
    cfg["jepa_on_style"] = jepa_on_style
    cfg["jepa_on_content"] = jepa_on_content

    # --- Config branche sémantique ---
    cfg["sem_enable"] = sem_enable
    cfg["lambda_sem"] = float(getattr(opt, "lambda_sem", 0.0))
    cfg["sem_every"] = int(getattr(opt, "sem_every", 1))
    cfg["sem_sym"] = bool(getattr(opt, "sem_sym", False))
    cfg["sem_two_styles"] = bool(getattr(opt, "sem_two_styles", False))
    cfg["sem_detach_far"] = int(getattr(opt, "sem_detach_far", 1))
    cfg["sem_use_aug"] = bool(getattr(opt, "sem_use_aug", False))
    cfg["sem_jepa_on"] = bool(
        sem_enable and getattr(opt, "jepa_on_content", 0) and getattr(opt, "jepa_tokens", False)
    )

    cfg["jepa_on"] = jepa_on_style or jepa_on_content

    cfg["jepa_mask_ratio"] = float(getattr(opt, "jepa_mask_ratio", 0.6))
    cfg["lambda_jepa_style"] = float(
        getattr(opt, "lambda_jepa_style", getattr(opt, "lambda_jepa", 0.15))
    )
    cfg["lambda_jepa_content"] = float(getattr(opt, "lambda_jepa_content", 0.15))
    cfg["lambda_jepa_var"] = float(getattr(opt, "lambda_jepa_var", 0.05))
    cfg["lambda_jepa_cov"] = float(getattr(opt, "lambda_jepa_cov", 0.05))
    cfg["lambda_jepa_kd"] = float(getattr(opt, "lambda_jepa_kd", 0.05))
    cfg["jepa_hidden_mult"] = int(getattr(opt, "jepa_hidden_mult", 2))
    cfg["jepa_use_teacher"] = bool(getattr(opt, "jepa_use_teacher", 1))
    cfg["jepa_every"] = max(1, int(getattr(opt, "jepa_every", 2)))
    cfg["jepa_heads"] = int(getattr(opt, "jepa_heads", 4))
    cfg["jepa_norm"] = bool(getattr(opt, "jepa_norm", 1))
    cfg["jepa_bias_high"] = float(getattr(opt, "jepa_mask_bias_high", 2.0))

    try:
        _usr = [
            float(x)
            for x in str(
                getattr(opt, "jepa_scale_weights", "2,2,1.5,1,0.75,0.5")
            ).split(",")
            if x.strip()
        ]
        if _usr:
            jepa_scale_w = torch.tensor(_usr, dtype=torch.float32, device=dev)
        else:
            jepa_scale_w = torch.tensor(
                [2.0, 2.0, 1.5, 1.0, 0.75, 0.5],
                dtype=torch.float32,
                device=dev,
            )
    except Exception:
        jepa_scale_w = torch.tensor(
            [2.0, 2.0, 1.5, 1.0, 0.75, 0.5],
            dtype=torch.float32,
            device=dev,
        )

    cfg["jepa_scale_w"] = jepa_scale_w

    cfg["feat_switch_epoch"] = int(
        getattr(opt, "feat_switch_epoch", getattr(opt, "recon_epochs", 2))
    )
    cfg["ema_every"] = int(getattr(opt, "ema_update_every", 1))
    cfg["nce_m"] = float(getattr(opt, "nce_m", 0.999))

    cfg["_denorm"] = _denorm
    cfg["_triplet_grid"] = _triplet_grid

    # --- Scheduler (cycle A / mix / recon) ---
    base_adv = getattr(opt, "adv_only_epochs", 2)
    base_mix = getattr(opt, "adv_mix_epochs", 0)
    base_rec = getattr(opt, "recon_epochs", 2)
    cycle_sched = CycleScheduler(
        base_adv=base_adv,
        base_mix=base_mix,
        base_rec=base_rec,
        adv_boost=max(0, int(getattr(opt, "adv_boost", 0))),
        b_boost=max(0, int(getattr(opt, "b_boost", 0))),
        skip_amix=bool(getattr(opt, "skip_amix", False)),
    )
    cfg["cycle_sched"] = cycle_sched

    # --- Fonctions pour reconstruire les conditionnements de style ---
    def _build_style_cond(G, style_img):
        if hasattr(G, "build_style_cond"):
            return G.build_style_cond(style_img)
        return style_img

    def _build_style_cond_from_tokens(toks, tokG, for_G="A"):
        if for_G == "A":
            G = state["G_A"]
        else:
            G = state["G_B"]
        if hasattr(G, "build_style_cond_from_tokens"):
            return G.build_style_cond_from_tokens(toks, tokG)
        return (toks, tokG)

    cfg["build_style_cond"] = _build_style_cond
    cfg["build_style_cond_from_tokens"] = _build_style_cond_from_tokens

    # --- Folds / loaders A/B ---
    current_fold = 0
    src_loader = loaders[current_fold]
    tgt_loader = loaders[(current_fold + 1) % k_folds]
    rounds_on_this_fold = 0
    fold_switch_every_rounds = int(getattr(opt, "fold_epochs", 4))

    # --- Runtime pour la supervision (hybrid / sup_freeze) ---
    sup_runtime = {
        "inited": False,
        "G_sup": None,
        "opt_Sup": None,
        "feat_type": None,
        "delta_w_str": None,
        "tasks": None,
        "class_names": None,
        "token_mode": None,
        "in_dim": None,
        "task_map": None,
        "task_map_printed": False,
        "class_map": None,
        "class_map_built": False,
    }

    # =====================================================================================
    #  MODE: sup_freeze (C seul, style gelé)
    # =====================================================================================
    if mode == "sup_freeze":
        run_sup_freeze_mode(
            opt=opt,
            loaders=loaders,
            G_A=G_A,
            G_B=G_B,
            D_A=D_A,
            D_B=D_B,
            opt_GA=opt_GA,
            opt_DA=opt_DA,
            opt_GB=opt_GB,
            opt_DB=opt_DB,
            dev=dev,
            writer=writer,
            tb_freq_C=tb_freq_C,
            global_step_start=state["global_step"],
        )
        if writer:
            writer.close()
        return

    # =====================================================================================
    #  MODES: auto / hybrid  (style + JEPA + sup hybride)
    # =====================================================================================
    ema_tau = float(getattr(opt, "ema_tau", 0.0))
    cycle_sched = cfg["cycle_sched"]

    while state["epoch"] < opt.epochs:
        epoch = state["epoch"]

        epoch_meters = new_epoch_meters()

        src_iter, tgt_iter = iter(src_loader), iter(tgt_loader)

        phase = cycle_sched.phase_now()
        λN, λR = cycle_sched.current_lambdas(
            λ_nce_AADV, λ_reg_AADV, λ_nce_AMIX, λ_reg_AMIX, λ_nce_B, λ_idt_B
        )
        cfg["λN_current"] = λN
        cfg["λR_current"] = λR
        cfg["phase_current"] = phase
        cfg["epoch"] = epoch

        # --- Mise à jour de λ_style_A / style_gain_A via le scheduler ---
        try:
            style_sched_cfg = cfg.get("style_sched", None)
            if style_sched_cfg is not None:
                λ_style_now = get_style_lambda(style_sched_cfg, epoch)
            else:
                λ_style_now = cfg.get("λ_style_A", λ_style_max)
        except Exception:
            λ_style_now = cfg.get("λ_style_A", λ_style_max)

        cfg["λ_style_A"] = float(λ_style_now)
        cfg["style_gain_A"] = float(getattr(opt, "style_gain_A", cfg["λ_style_A"]))

        budgets = cycle_sched.budgets()
        print(
            f"\n📅 Epoch {epoch + 1:03d}/{opt.epochs}"
            f" | ROUND={budgets['round']}  A={budgets['A_done']}/{budgets['adv'] + budgets['mix']}"
            f"  R={budgets['R_done']}/{budgets['rec']}  | phase={phase:<6}"
            f" | λN={λN:.3f} λL1/ID={λR:.3f} λ_style_A={cfg['λ_style_A']:.3f}"
        )

        if writer:
            writer.add_scalar("phase/epoch", epoch, state["global_step"])
            writer.add_scalar("A/style_lambda_epoch", float(cfg["λ_style_A"]), epoch)

        # =================================================================================
        #  Boucle A/B : style + JEPA
        # =================================================================================
        if mode in ["auto", "hybrid"]:
            pbar = tqdm(range(nbatchs), ncols=180, leave=False)
            for _ in pbar:
                try:
                    x, _ = next(src_iter)
                except StopIteration:
                    src_iter = iter(src_loader)
                    x, _ = next(src_iter)

                try:
                    y, _ = next(tgt_iter)
                except StopIteration:
                    tgt_iter = iter(tgt_loader)
                    y, _ = next(tgt_iter)

                x, y = x.to(dev), y.to(dev)

                try:
                    if phase.startswith("A"):
                        train_step_phase_A(
                            x=x,
                            y=y,
                            state=state,
                            cfg=cfg,
                            epoch_meters=epoch_meters,
                            writer=writer,
                        )
                    else:
                        train_step_phase_B(
                            x=x,
                            state=state,
                            cfg=cfg,
                            epoch_meters=epoch_meters,
                            writer=writer,
                        )

                    state["global_step"] += 1

                    # --- Sauvegarde en mode "step" ---
                    if (
                        save_freq_mode == "step"
                        and save_freq_interval is not None
                        and state["global_step"] % save_freq_interval == 0
                    ):
                        save_checkpoint(
                            epoch,
                            G_A,
                            D_A,
                            G_B,
                            D_B,
                            opt_GA,
                            opt_DA,
                            opt_GB,
                            opt_DB,
                            state["global_step"],
                            Path(opt.save_dir),
                            sem_model=state.get("SEM", None),
                            opt_sem=state.get("opt_SEM", None),
                        )
                        save_state_json(epoch, state["global_step"], opt, Path(opt.save_dir))

                except Exception as ex:
                    msg = (
                        f"[ERROR] step={state['global_step']} epoch={epoch} "
                        f"{type(ex).__name__}: {ex}"
                    )
                    tqdm.write(msg)
                    if getattr(opt, "print_trace_on_error", False):
                        import traceback as _tb
                        tqdm.write(_tb.format_exc())
                    if writer:
                        import traceback as _tb
                        writer.add_text(
                            "errors/exception",
                            f"{msg}\n{_tb.format_exc()}",
                            state["global_step"],
                        )
                    state["global_step"] += 1
                    continue

        # --- Phase de supervision hybride après warmup ---
        if mode == "hybrid" and (epoch + 1) > getattr(opt, "warmup_epochs", 0):
            state["global_step"] = run_hybrid_supervised_epoch(
                opt=opt,
                epoch=epoch,
                epoch_meters=epoch_meters,
                sup_runtime=sup_runtime,
                src_loader=src_loader,
                G_A=G_A,
                G_B=G_B,
                D_A=D_A,
                D_B=D_B,
                opt_GA=opt_GA,
                opt_GB=opt_GB,
                dev=dev,
                nbatchs=nbatchs,
                writer=writer,
                tb_freq_C=tb_freq_C,
                global_step=state["global_step"],
            )

            if (
                save_freq_mode == "step"
                and save_freq_interval is not None
                and state["global_step"] % save_freq_interval == 0
            ):
                save_checkpoint(
                    epoch,
                    G_A,
                    D_A,
                    G_B,
                    D_B,
                    opt_GA,
                    opt_DA,
                    opt_GB,
                    opt_DB,
                    state["global_step"],
                    Path(opt.save_dir),
                    sem_model=state.get("SEM", None),
                    opt_sem=state.get("opt_SEM", None),
                )
                save_state_json(epoch, state["global_step"], opt, Path(opt.save_dir))

        # --- EMA pour D_B éventuel ---
        if ema_tau > 0:
            with torch.no_grad():
                for pA, pB in zip(D_A.parameters(), D_B.parameters()):
                    pB.lerp_(pA, ema_tau)
            opt_DB.state = defaultdict(dict)
        if writer:
            writer.add_scalar("EMA/tau", float(ema_tau), epoch)

        print_epoch_summary(epoch, epoch_meters)

        # --- Step scheduler / folds ---
        cycle_sched.step_epoch()
        if cycle_sched.round_done():
            rounds_on_this_fold += 1
            if rounds_on_this_fold >= fold_switch_every_rounds and k_folds > 1:
                current_fold = (current_fold + 1) % k_folds
                src_loader = loaders[current_fold]
                tgt_loader = loaders[(current_fold + 1) % k_folds]
                rounds_on_this_fold = 0
                print(f"🔁 SWITCH FOLD → {current_fold}")
            cycle_sched.next_round()

        # --- Sauvegarde périodique en mode "epoch" / "none" ---
        do_save_epoch = False
        if save_freq_mode == "epoch" and save_freq_interval is not None:
            if (epoch + 1) % save_freq_interval == 0:
                do_save_epoch = True
        elif save_freq_mode in ("none", "step"):
            if (epoch + 1) == opt.epochs:
                do_save_epoch = True

        if do_save_epoch:
            save_checkpoint(
                epoch,
                G_A,
                D_A,
                G_B,
                D_B,
                opt_GA,
                opt_DA,
                opt_GB,
                opt_DB,
                state["global_step"],
                Path(opt.save_dir),
                sem_model=state.get("SEM", None),
                opt_sem=state.get("opt_SEM", None),
            )
            save_state_json(epoch, state["global_step"], opt, Path(opt.save_dir))

        state["epoch"] += 1

    if writer:
        writer.close()
