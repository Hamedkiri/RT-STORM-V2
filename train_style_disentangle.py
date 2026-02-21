# file: train_style_disentangle.py
# -*- coding: utf-8 -*-

"""
Entraînement alterné ST-STORM / SC-STORM : Double-GAN (G_A, D_A, G_B, D_B)
+ JEPA tokens (style / contenu)
+ MixSwap, FFT/SWD textures
+ mode supervision hybride (C)
+ mode sup_freeze (C seul)
+ mode détection (detect_transformer) : route vers training/train_detection_transformer.py

Ce fichier est conçu pour :
- un logging robuste terminal/tqdm + TensorBoard
- une gestion robuste des checkpoints (epoch/step/none) + sauvegarde finale
- une compatibilité avec branche sémantique (ResNet50 + MoCo + JEPA-content) en mode auto
- une intégration claire des options de détection (head: fasterrcnn/detr/vitdet/fastrnn) via opt.det_head

⚠️ NOTES D’INTÉGRATION
- Ce fichier suppose l’existence des modules importés (data/models/training/helpers…).
- Si certains noms diffèrent dans ton repo (ex: SemanticMoCoJEPA / SemAugConfig), adapte les imports.
- Les fonctions train_step_phase_A / train_step_phase_B sont importées depuis helpers.py et doivent
  inclure les ajouts SEM (courbes raw/rel/ema, q/k diagnostics, etc.).
"""

import json
import time
import math
from collections import deque, defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# -----------------------------------------------------------------------------------------
# Local imports (projet)
# -----------------------------------------------------------------------------------------
from data import build_dataloader
from models.generator import UNetGenerator
from models.discriminator import PatchDiscriminator
from models.semantic_moco_jepa import SemanticMoCoJEPA, SemAugConfig  # adapte si nom différent
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
    should_save_ckpt,  # ✅ décision centralisée (epoch/step/none + final)
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
    run_sup_freeze_sem_only,
    new_epoch_meters,
    get_style_lambda,  # scheduler de λ_style_A
)

# =========================================================================================
#   Terminal + TensorBoard logging helpers
# =========================================================================================

def _is_finite_number(x: float) -> bool:
    try:
        return torch.isfinite(torch.tensor(float(x))).item()
    except Exception:
        return False


def _to_float(v: Any) -> Optional[float]:
    """Convertit proprement vers float (tensor/np/number), sinon None."""
    if v is None:
        return None
    try:
        if torch.is_tensor(v):
            if v.numel() == 1:
                return float(v.detach().float().cpu().item())
            return float(v.detach().float().mean().cpu().item())
        return float(v)
    except Exception:
        return None


def _meter_value(m: Any) -> Optional[float]:
    """
    Récupère une valeur (avg de préférence) depuis des formats courants :
      - objet avec .avg / .value / .val / .mean
      - dict {'avg':..} / {'value':..} / {'val':..} / {'mean':..}
      - nombre / tensor
    """
    if m is None:
        return None

    for attr in ("avg", "value", "val", "mean"):
        if hasattr(m, attr):
            return _to_float(getattr(m, attr))

    if isinstance(m, dict):
        for k in ("avg", "value", "val", "mean"):
            if k in m:
                return _to_float(m[k])
        for _, vv in m.items():
            x = _to_float(vv)
            if x is not None:
                return x

    return _to_float(m)


def _meters_to_scalars(epoch_meters: Any) -> Dict[str, float]:
    """Convertit epoch_meters (souvent dict) en {name: float} filtré."""
    out: Dict[str, float] = {}
    if isinstance(epoch_meters, dict):
        for k, v in epoch_meters.items():
            x = _meter_value(v)
            if x is None:
                continue
            if _is_finite_number(x):
                out[str(k)] = float(x)
    return out


def _get_lrs(optimizers: Dict[str, Optional[torch.optim.Optimizer]]) -> Dict[str, float]:
    lrs: Dict[str, float] = {}
    for name, opt in optimizers.items():
        if opt is None:
            continue
        try:
            lr = opt.param_groups[0].get("lr", None)
            lr = _to_float(lr)
            if lr is not None:
                lrs[f"lr/{name}"] = float(lr)
        except Exception:
            pass
    return lrs


def _tb_add_scalars(writer: Optional[SummaryWriter], scalars: Dict[str, float], step: int, prefix: str = ""):
    if writer is None:
        return
    for k, v in scalars.items():
        tag = f"{prefix}{k}" if prefix else k
        try:
            writer.add_scalar(tag, float(v), step)
        except Exception:
            pass


def _tb_add_text(writer: Optional[SummaryWriter], tag: str, text: str, step: int):
    if writer is None:
        return
    try:
        writer.add_text(tag, text, step)
    except Exception:
        pass


def _tb_maybe_flush(writer: Optional[SummaryWriter], every_steps: int, step: int):
    if writer is None or every_steps <= 0:
        return
    if step % every_steps == 0:
        try:
            writer.flush()
        except Exception:
            pass


def _format_postfix(values: Dict[str, float], keys: Tuple[str, ...]) -> str:
    parts = []
    for k in keys:
        if k in values:
            parts.append(f"{k}={values[k]:.3f}")
    return " ".join(parts)


def _print_kv_line(title: str, d: Dict[str, float], keys: Tuple[str, ...]):
    items = []
    for k in keys:
        if k in d:
            items.append(f"{k}={d[k]:.4f}")
    if items:
        print(f"{title} " + " | ".join(items))


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

    specific = None
    if mode == "detect_transformer":
        specific = getattr(opt, "det_freeze_backbone", None)

    if specific is None:
        return global_flag
    return bool(int(specific) != 0)


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
      - '1000'       -> ('epoch', 1000)  (use 'step:1000' for step-based)
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

    # Fallback: numeric means "every N epochs" (more intuitive).
    # Use explicit 'step:N' for step-based checkpointing.
    try:
        n = int(sf)
        return "epoch", max(1, n)
    except ValueError:
        return "none", None


# =========================================================================================
#   SEM Warmup + Cosine scheduler helper (LambdaLR)
# =========================================================================================

def build_warmup_cosine_lambda(
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 0.0,
    warmup_init_lr: float = 0.0,
):
    """
    Retourne f(step)->multiplier pour LambdaLR (lr = base_lr * multiplier),
    avec warmup linéaire (warmup_init_lr -> base_lr) puis cosine decay (base_lr -> min_lr).
    """
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))
    min_lr = float(min_lr)
    warmup_init_lr = float(warmup_init_lr)

    # Clamp valeurs
    min_lr = max(0.0, min(min_lr, base_lr))
    warmup_init_lr = max(0.0, min(warmup_init_lr, base_lr))

    def _lr_mult(step_idx: int) -> float:
        s = int(step_idx)
        if warmup_steps > 0 and s < warmup_steps:
            lr = warmup_init_lr + (base_lr - warmup_init_lr) * (float(s + 1) / float(warmup_steps))
        else:
            if total_steps <= warmup_steps:
                lr = min_lr
            else:
                t = float(s - warmup_steps)
                T = float(max(1, total_steps - warmup_steps))
                cos_v = 0.5 * (1.0 + math.cos(math.pi * min(1.0, t / T)))
                lr = min_lr + (base_lr - min_lr) * cos_v

        return float(lr / base_lr) if base_lr > 0 else 1.0

    return _lr_mult


# =========================================================================================
#   Helper : validation minimale des options de détection
# =========================================================================================

def _validate_detection_opts(opt):
    required = ["det_train_img_root", "det_train_ann", "det_val_img_root", "det_val_ann"]
    missing = [k for k in required if not getattr(opt, k, None)]
    if missing:
        raise ValueError(
            "[DET] Options manquantes pour la détection: "
            + ", ".join(missing)
            + " (attendu: COCO-like train/val images + annotations)."
        )

    head = str(getattr(opt, "det_head", "fasterrcnn")).lower().strip()
    if head not in {"fasterrcnn", "detr", "vitdet", "fastrnn"}:
        raise ValueError(f"[DET] det_head invalide: {head}")

    det_num_classes = int(getattr(opt, "det_num_classes", 91))
    if det_num_classes <= 1:
        raise ValueError("[DET] det_num_classes doit être >= 2 (incluant background).")

    h = int(getattr(opt, "det_img_h", 256))
    w = int(getattr(opt, "det_img_w", 256))
    if h <= 0 or w <= 0:
        raise ValueError("[DET] det_img_h/det_img_w doivent être > 0.")


# =========================================================================================
#           Entraînement alterné (style + JEPA + supervision + détection)
# =========================================================================================

def train_alternating(opt):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode = getattr(opt, "mode", "auto")
    mode_help = {
        "auto": "A+B (self-supervised: style + JEPA)",
        "sup_freeze": "C seul (supervisé), G&D gelés",
        "hybrid": "A+B puis C (supervisé)",
        "cls_tokens": "Classification via tokens multi-échelles",
        "detect_transformer": "Entraînement détection (head: fasterrcnn/detr/vitdet/fastrnn)",
    }

    print("\n" + "=" * 88)
    print(f"🎛️  MODE D'ENTRAÎNEMENT : {str(mode).upper()}  →  {mode_help.get(mode, '?')}")
    print("=" * 88)

    # =====================================================================================
    #  MODE: détection → route directe
    # =====================================================================================
    if mode == "detect_transformer":
        _validate_detection_opts(opt)
        freeze_det = resolve_backbone_freeze(opt, mode="detect_transformer")
        opt.det_freeze_backbone = int(freeze_det)
        print(f"[DET] det_head={getattr(opt, 'det_head', 'fasterrcnn')} | det_freeze_backbone={freeze_det}")
        return train_detection_transformer(opt, dev)

    # =====================================================================================
    #  MODES: auto / hybrid / sup_freeze / cls_tokens  → pipeline style + JEPA
    # =====================================================================================

    loaders = build_dataloader(opt)
    k_folds = len(loaders)
    if k_folds == 0:
        raise RuntimeError("[data] Aucun dataloader construit. Vérifie --data/--data_json/--search_folder...")

    nbatchs = min(len(dl) for dl in loaders) if k_folds > 0 else 0
    if nbatchs <= 0:
        raise RuntimeError("[data] Dataloaders vides (nbatchs=0).")

    # ==================================================================================
    #  MODE: sup_freeze + backbone sémantique uniquement
    #   -> ne charge pas G_A/G_B/D_A/D_B. On entraîne SupHeads multi-tâches sur SemBackbone.
    # ==================================================================================
    sup_feat_source = str(getattr(opt, "sup_feat_source", "generator")).lower().strip()
    sup_sem_only = int(getattr(opt, "sup_sem_only", 1)) == 1
    if mode == "sup_freeze" and sup_feat_source == "sem_resnet50" and sup_sem_only:
        writer = SummaryWriter(opt.save_dir) if getattr(opt, "tb", False) else None
        run_sup_freeze_sem_only(
            opt=opt,
            loaders=loaders,
            dev=dev,
            writer=writer,
            tb_freq_C=int(getattr(opt, "tb_freq_C", getattr(opt, "tb_freq", 50))),
            global_step_start=0,
        )
        if writer:
            writer.flush()
            writer.close()
        return

    # ==================================================================================
    #  Reprise: aligner automatiquement l'architecture du générateur sur le run résumé
    #  (sinon: size mismatch même en strict=False).
    # ==================================================================================
    def _autoload_resume_arch(opt):
        rdir = getattr(opt, "resume_dir", None)
        if not rdir:
            return
        try:
            rdir = Path(rdir)
        except Exception:
            return
        cfg_path = rdir / "train_cfg.json"
        if not cfg_path.exists():
            return
        try:
            j = json.loads(cfg_path.read_text())
        except Exception:
            return
        hp = j.get("static_hparams", j)

        # These keys affect UNetGenerator state_dict shapes.
        keys = [
            ("arch_depth_delta", 0),
            ("style_token_levels", -1),
            ("unet_min_spatial", 2),
            ("crop_size", 256),
        ]
        changed = []
        for k, default in keys:
            if k in hp:
                old = getattr(opt, k, default)
                new = hp.get(k, default)
                try:
                    old_i = int(old)
                    new_i = int(new)
                    if old_i != new_i:
                        setattr(opt, k, new_i)
                        changed.append((k, old_i, new_i))
                except Exception:
                    if old != new:
                        setattr(opt, k, new)
                        changed.append((k, old, new))

        # token_dim may impact conditioning shapes; follow checkpoint if present.
        if "token_dim" in hp:
            try:
                opt.token_dim = int(hp["token_dim"])
            except Exception:
                opt.token_dim = 256
        else:
            opt.token_dim = int(getattr(opt, "token_dim", 256) or 256)

        if changed:
            msg = ", ".join([f"{k}:{a}->{b}" for k, a, b in changed])
            print(f"[RESUME][ARCH] Override depuis {cfg_path}: {msg}", flush=True)

    _autoload_resume_arch(opt)

    # ---------------------------------------------------------------------
    # Norm variant (legacy vs safe) + optional extra bottleneck resblocks
    # ---------------------------------------------------------------------
    # Default: legacy (backward compatible). If --safe_norm: force safe.
    if bool(getattr(opt, "safe_norm", False)):
        opt.norm_variant = "safe"
    else:
        opt.norm_variant = str(getattr(opt, "norm_variant", "legacy") or "legacy")
    opt.extra_bot_resblocks = int(getattr(opt, "extra_bot_resblocks", 0) or 0)
    opt.use_res_skip_bot = bool(getattr(opt, "use_res_skip_bot", False))
    opt.style_tokg_head_variant = str(getattr(opt, "style_tokg_head_variant", "tokG_head") or "tokG_head")

    def _infer_ckpt_compat_from_state(sd: dict) -> tuple[str, int, bool, str]:
        keys = list(sd.keys())
        is_safe = any(".inorm." in k or ".gnorm." in k for k in keys)
        norm_v = "safe" if is_safe else "legacy"
        extra = 0
        if any(k.startswith("res5.") for k in keys):
            extra = max(extra, 1)
        if any(k.startswith("res6.") for k in keys):
            extra = max(extra, 2)

        use_res_skip_bot = any(k.startswith("res_skip.") or k.startswith("res_bot.") for k in keys)

        if any(k.startswith("style_enc.tbot.") for k in keys):
            tokg_head_variant = "tbot"
        else:
            tokg_head_variant = "tokG_head"
        return norm_v, extra, use_res_skip_bot, tokg_head_variant

    # If resuming and user did not force safe_norm, try to infer from checkpoint if not stored.
    if (not bool(getattr(opt, "safe_norm", False))) and getattr(opt, "resume_dir", None):
        rdir = Path(getattr(opt, "resume_dir"))
        cfg_path = rdir / "train_cfg.json"
        hp = {}
        try:
            if cfg_path.exists():
                hp = json.loads(cfg_path.read_text()).get("static_hparams", {})
        except Exception:
            hp = {}
        if "norm_variant" in hp:
            opt.norm_variant = str(hp.get("norm_variant") or opt.norm_variant)
        if "extra_bot_resblocks" in hp:
            try:
                opt.extra_bot_resblocks = int(hp.get("extra_bot_resblocks") or 0)
            except Exception:
                pass
        if "use_res_skip_bot" in hp:
            opt.use_res_skip_bot = bool(hp.get("use_res_skip_bot"))
        if "style_tokg_head_variant" in hp:
            opt.style_tokg_head_variant = str(hp.get("style_tokg_head_variant") or opt.style_tokg_head_variant)

        # Always infer from last generator checkpoint (source of truth) unless user forced safe_norm.
        import glob
        cand = sorted(glob.glob(str(rdir / "**" / "G_A_epoch*.pt"), recursive=True))
        if cand:
            try:
                ck = torch.load(cand[-1], map_location="cpu")
                gen_sd = ck.get("model", ck) if isinstance(ck, dict) else ck
                nv, extra, use_rs, thv = _infer_ckpt_compat_from_state(gen_sd)
                if nv != opt.norm_variant:
                    print(f"[RESUME][NORM] Override norm_variant {opt.norm_variant} -> {nv} (from checkpoint)", flush=True)
                opt.norm_variant = nv
                opt.extra_bot_resblocks = max(opt.extra_bot_resblocks, extra)
                opt.use_res_skip_bot = bool(use_rs)
                opt.style_tokg_head_variant = str(thv)
            except Exception:
                pass

    # --- Générateurs / Discriminateurs (style)
    _arch_dd = int(getattr(opt, "arch_depth_delta", 0))
    _sty_lv = int(getattr(opt, "style_token_levels", -1))
    _img_sz = int(getattr(opt, "crop_size", 256))
    _min_sp = int(getattr(opt, "unet_min_spatial", 2))
    _tokdim = int(getattr(opt, "token_dim", 256))

    G_A, D_A = UNetGenerator(
        token_dim=_tokdim,
        arch_depth_delta=_arch_dd,
        style_token_levels=_sty_lv,
        img_size=_img_sz,
        unet_min_spatial=_min_sp,
        norm_variant=str(getattr(opt, "norm_variant", "legacy") or "legacy"),
        extra_bot_resblocks=int(getattr(opt, "extra_bot_resblocks", 0) or 0),
        use_res_skip_bot=bool(getattr(opt, "use_res_skip_bot", False)),
        style_tokg_head_variant=str(getattr(opt, "style_tokg_head_variant", "tokG_head") or "tokG_head"),
    ).to(dev), PatchDiscriminator(norm_variant=str(getattr(opt, "norm_variant", "legacy") or "legacy")).to(dev)
    G_B, D_B = UNetGenerator(
        token_dim=_tokdim,
        arch_depth_delta=_arch_dd,
        style_token_levels=_sty_lv,
        img_size=_img_sz,
        unet_min_spatial=_min_sp,
        norm_variant=str(getattr(opt, "norm_variant", "legacy") or "legacy"),
        extra_bot_resblocks=int(getattr(opt, "extra_bot_resblocks", 0) or 0),
        use_res_skip_bot=bool(getattr(opt, "use_res_skip_bot", False)),
        style_tokg_head_variant=str(getattr(opt, "style_tokg_head_variant", "tokG_head") or "tokG_head"),
    ).to(dev), PatchDiscriminator(norm_variant=str(getattr(opt, "norm_variant", "legacy") or "legacy")).to(dev)

    if bool(getattr(opt, 'debug_shapes', False)):
        for _g in (G_A, G_B):
            try:
                _g.debug_shapes = True
                _g.style_enc.debug_shapes = True
            except Exception:
                pass
        print('[debug_shapes] enabled (will trace shapes once).', flush=True)

    base_lr = float(getattr(opt, "lr", 2e-4))
    adv_lrD_mult = float(getattr(opt, "adv_lrD_mult", 0.5))

    opt_GA = torch.optim.Adam(G_A.parameters(), lr=base_lr, betas=(0.5, 0.999))
    opt_GB = torch.optim.Adam(G_B.parameters(), lr=base_lr, betas=(0.5, 0.999))
    opt_DA = torch.optim.Adam(D_A.parameters(), lr=base_lr * adv_lrD_mult, betas=(0.5, 0.999))
    opt_DB = torch.optim.Adam(D_B.parameters(), lr=base_lr * adv_lrD_mult, betas=(0.5, 0.999))

    # --- TensorBoard
    writer = SummaryWriter(opt.save_dir) if getattr(opt, "tb", False) else None

    resume_epoch = None

    # --- Checkpoint I/O options (propagés à save_checkpoint)
    ckpt_use_safetensors = bool(int(getattr(opt, "use_safetensors", 0)) != 0)
    ckpt_safe_write = bool(int(getattr(opt, "safe_write", 1)) != 0)

    # --- Reprise éventuelle ---
    if getattr(opt, "resume_dir", None):
        e = last_epoch(opt.resume_dir, "G_B")
        if e is not None:
            load_checkpoint(
                opt.resume_dir,
                e,
                G_A, D_A, G_B, D_B,
                opt_GA, opt_DA, opt_GB, opt_DB,
                device=str(dev),
                strict=True,
            )
            resume_epoch = int(e)
            print(f"✓ reprise depuis epoch {e}")

    out_dir = Path(opt.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_cfg.json").write_text(json.dumps({"static_hparams": vars(opt)}, indent=2))

    # --- TB: hparams
    if writer:
        _tb_add_text(writer, "run/mode", f"{mode} – {mode_help.get(mode, '')}", 0)
        _tb_add_text(writer, "run/device", str(dev), 0)
        try:
            _tb_add_text(writer, "run/opt_json", json.dumps(vars(opt), indent=2), 0)
        except Exception:
            pass
        writer.add_scalar("hparams/batch_size", float(getattr(opt, "batch_size", 0)), 0)
        writer.add_scalar("hparams/lr", float(base_lr), 0)

    # --- Fréquences d'affichage
    tb_freq = int(getattr(opt, "tb_freq", 100))
    tb_freq_C = int(getattr(opt, "tb_freq_C", tb_freq))
    print_freq = int(getattr(opt, "print_freq", 50))
    postfix_keys = tuple(
        k.strip()
        for k in str(getattr(opt, "postfix_keys", "loss_G,loss_D,loss_nce,loss_jepa,loss_idt,loss_reg")).split(",")
        if k.strip()
    )

    # --- Gestion globale de save_freq (string -> (mode, interval))
    save_freq_mode, save_freq_interval = _parse_save_freq(getattr(opt, "save_freq", "epoch"))
    epoch_ckpt_interval = getattr(opt, "epoch_ckpt_interval", None)
    if save_freq_mode == "epoch" and epoch_ckpt_interval is not None:
        try:
            save_freq_interval = max(1, int(epoch_ckpt_interval))
        except Exception:
            pass

    print(f"[CKPT] save_freq = {getattr(opt, 'save_freq', 'epoch')} → mode={save_freq_mode}, interval={save_freq_interval}")
    print(f"[CKPT] use_safetensors={ckpt_use_safetensors} | safe_write={ckpt_safe_write}")

    # --- État global de l'entraînement ---
    state: Dict[str, Any] = {
        "G_A": G_A,
        "G_B": G_B,
        "D_A": D_A,
        "D_B": D_B,
        "opt_GA": opt_GA,
        "opt_GB": opt_GB,
        "opt_DA": opt_DA,
        "opt_DB": opt_DB,
        "epoch": 0 if resume_epoch is None else int(resume_epoch) + 1,
        "global_step": 0,
        "replay": deque(maxlen=int(getattr(opt, "replay_size", 50))),
        "style_bank": deque(maxlen=2000),

        # JEPA caches
        "tokJEPA_A": None,
        "tokJEPA_B": None,

        "λ_style_B_dyn": float(getattr(opt, "lambda_style_b", 0.005)),

        # SEM tracking + errors
        "sem_track": {},
        "sem_exceptions": 0,

        # SEM sched
        "sch_SEM": None,
        "sem_lr_step": 0,
    }

    # --- Teachers EMA pour JEPA/NCE
    import copy as _copy
    state["T_A"] = _copy.deepcopy(G_A).eval()
    for p in state["T_A"].parameters():
        p.requires_grad_(False)
    state["T_B"] = _copy.deepcopy(G_B).eval()
    for p in state["T_B"].parameters():
        p.requires_grad_(False)

    # =====================================================================================
    #  SEMANTIC CONTENT BRANCH (ResNet50 + MoCo + optional JEPA-content)
    # =====================================================================================
    sem_enable = bool(getattr(opt, "sem_content", False) and mode == "auto")
    cfg_sem_enable_flag = sem_enable

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

        sem_jepa_on = bool(getattr(opt, "jepa_on_content", 0) and getattr(opt, "jepa_tokens", False))

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
            backbone_arch=str(getattr(opt, "sem_backbone", "resnet50")),
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

        sem_params = [p for p in sem_model.parameters() if p.requires_grad]
        _lr_sem = getattr(opt, "lr_sem", None)
        sem_lr = float(_lr_sem) if _lr_sem is not None else float(getattr(opt, "lr", 2e-4))
        opt_sem = torch.optim.AdamW(sem_params, lr=sem_lr, betas=(0.9, 0.999), weight_decay=1e-4)

        state["SEM"] = sem_model
        state["opt_SEM"] = opt_sem

        # -----------------------------
        # SEM LR Scheduler (Warmup + Cosine)
        # -----------------------------
        sem_sched = str(getattr(opt, "sem_lr_sched", "none")).lower().strip()
        sem_sched_by = str(getattr(opt, "sem_sched_by", "epoch")).lower().strip()

        if sem_sched == "warmup_cosine":
            warmup_epochs = int(getattr(opt, "sem_warmup_epochs", 0))
            min_lr = float(getattr(opt, "sem_min_lr", 0.0))
            warmup_init_lr = float(getattr(opt, "sem_warmup_init_lr", 0.0))

            base_sem_lr = float(opt_sem.param_groups[0].get("lr", sem_lr))
            sem_every = max(1, int(getattr(opt, "sem_every", 1)))

            if sem_sched_by == "step":
                updates_per_epoch = int(math.ceil(float(nbatchs) / float(sem_every)))
                total_steps = max(1, updates_per_epoch * int(getattr(opt, "epochs", 1)))
                warmup_steps = max(0, updates_per_epoch * warmup_epochs)
            else:
                total_steps = max(1, int(getattr(opt, "epochs", 1)))
                warmup_steps = max(0, warmup_epochs)

            lr_lambda = build_warmup_cosine_lambda(
                base_lr=base_sem_lr,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                min_lr=min_lr,
                warmup_init_lr=warmup_init_lr,
            )

            state["sch_SEM"] = torch.optim.lr_scheduler.LambdaLR(opt_sem, lr_lambda=lr_lambda, last_epoch=-1)

            print(
                f"[SEM][sched] warmup_cosine | by={sem_sched_by} | base_lr={base_sem_lr:.3e} "
                f"| warmup={warmup_steps} | total={total_steps} | min_lr={min_lr:.3e} | init_lr={warmup_init_lr:.3e}"
            )

        print(
            f"[SEM] enabled | pretrained={getattr(opt, 'sem_pretrained', 1)} "
            f"| sem_pretrained_path={sem_pretrained_path} | strict={int(sem_pretrained_strict)}"
        )
        if writer:
            writer.add_scalar("SEM/enabled", 1.0, 0)
            if sem_pretrained_path:
                _tb_add_text(writer, "SEM/pretrained_path", str(sem_pretrained_path), 0)

    # --- Reprise SEM si nécessaire
    if sem_enable and getattr(opt, "resume_dir", None) and resume_epoch is not None and state.get("SEM", None) is not None:
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
                T_A=state.get("T_A", None),
                T_B=state.get("T_B", None),
                tokJEPA_A=state.get("tokJEPA_A", None),
                tokJEPA_B=state.get("tokJEPA_B", None),
            )
            print("✓ reprise SEM/Teachers/JEPA depuis checkpoint")
        except Exception as e:
            print(f"[WARN] reprise SEM/Teachers/JEPA impossible: {e}")

    # =====================================================================================
    #  Config partagée (helpers / phases)
    # =====================================================================================
    cfg: Dict[str, Any] = {}
    cfg["device"] = dev
    cfg["opt"] = opt
    cfg["writer"] = writer
    cfg["tb_freq"] = tb_freq
    cfg["tb_freq_C"] = tb_freq_C

    # --- Ajouts SEM : fréquence / affichage / courbes (utilisés dans train_step_phase_A/B)
    cfg["tb_freq_sem"] = int(getattr(opt, "tb_freq_sem", tb_freq))
    cfg["sem_print_every"] = int(getattr(opt, "sem_print_every", cfg["tb_freq_sem"]))
    cfg["sem_ema_beta"] = float(getattr(opt, "sem_ema_beta", 0.98))
    cfg["sem_rel_eps"] = float(getattr(opt, "sem_rel_eps", 1e-8))
    cfg["sem_on_B"] = bool(int(getattr(opt, "sem_on_B", 1)) != 0)

    cfg["l1_loss"] = torch.nn.L1Loss().to(dev)

    # --- Perte NCE patch
    cfg["nce_loss"] = PatchNCELoss(
        temperature=float(getattr(opt, "nce_t", 0.07)),
        use_intra_neg=bool(getattr(opt, "nce_intra", True)),
        use_inter_neg=bool(getattr(opt, "nce_inter", True)),
        max_patches=getattr(opt, "nce_max_patches", None),
    )

    nce_layers = [l.strip() for l in str(getattr(opt, "nce_layers", "bot,skip64,skip32")).split(",") if l.strip()]
    w_str = getattr(opt, "nce_layer_weights", None) or ",".join(["1"] * len(nce_layers))
    try:
        layer_w = [float(x) for x in str(w_str).split(",") if str(x).strip() != ""]
    except Exception:
        layer_w = [1.0] * len(nce_layers)
    if len(layer_w) < len(nce_layers):
        layer_w = layer_w + [layer_w[-1]] * (len(nce_layers) - len(layer_w))
    layer_w = layer_w[: len(nce_layers)]
    sw = sum(layer_w) or 1.0
    layer_w = [w / sw for w in layer_w]

    cfg["nce_layers"] = nce_layers
    cfg["nce_layer_w"] = layer_w

    # --- Adversarial / R1 / highpass
    cfg["adv_enable_A"] = bool(getattr(opt, "adv_enable_A", True))
    cfg["adv_enable_B"] = bool(getattr(opt, "adv_enable_B", False))
    cfg["adv_type"] = str(getattr(opt, "adv_type", "hinge")).lower().strip()
    cfg["adv_r1_gamma"] = float(getattr(opt, "adv_r1_gamma", 10.0))
    cfg["adv_r1_every"] = int(getattr(opt, "adv_r1_every", 16))
    cfg["adv_highpass"] = bool(getattr(opt, "adv_highpass", False))
    cfg["highpass_fn"] = highpass

    # --- Texture (FFT / SWD)
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

    cfg["fft_texture_loss"] = fft_texture_loss
    cfg["swd_loss_images"] = swd_loss_images
    cfg["spectral_noise"] = spectral_noise

    # --- Lambdas NCE / régul par phase
    λ_nce_AADV = float(getattr(opt, "lambda_nce_a_adv", 1.0))
    λ_reg_AADV = float(getattr(opt, "lambda_reg_a_adv", 1.0))
    λ_nce_AMIX = float(getattr(opt, "lambda_nce_a_mix", 1.0))
    λ_reg_AMIX = float(getattr(opt, "lambda_reg_a_mix", 0.5))
    λ_nce_B = float(getattr(opt, "lambda_nce_b", 1.0))
    λ_idt_B = float(getattr(opt, "lambda_idt_b", getattr(opt, "lambda_reg_b", 10.0)))

    cfg["content_nce_enable"] = bool(getattr(opt, "content_nce_enable", 0))
    cfg["lambda_content_nce"] = float(getattr(opt, "lambda_content_nce", 0.0))

    # --- Style scheduler pour λ_style_A
    λ_style_max = float(getattr(opt, "style_lambda", 10.0))
    λ_style_min = float(getattr(opt, "style_lambda_min", 0.0))
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
    cfg["lambda_idt_b"] = float(getattr(opt, "lambda_idt_b", getattr(opt, "lambda_reg_b", 10.0)))

    # --- MixSwap
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

    # --- JEPA config (style + contenu)
    jepa_on_style = bool(getattr(opt, "jepa_on_style", 1) and mode == "auto")
    jepa_on_content = bool(getattr(opt, "jepa_on_content", 0) and mode == "auto" and (not sem_enable))
    cfg["jepa_on_style"] = jepa_on_style
    cfg["jepa_on_content"] = jepa_on_content

    # --- SEM config (branch moCo+JEPA-content)
    cfg["sem_enable"] = cfg_sem_enable_flag
    cfg["lambda_sem"] = float(getattr(opt, "lambda_sem", 0.0))
    cfg["sem_every"] = int(getattr(opt, "sem_every", 1))
    cfg["sem_sym"] = bool(getattr(opt, "sem_sym", False))
    cfg["sem_two_styles"] = bool(getattr(opt, "sem_two_styles", False))
    cfg["sem_detach_far"] = int(getattr(opt, "sem_detach_far", 1))
    cfg["sem_use_aug"] = bool(getattr(opt, "sem_use_aug", False))
    cfg["sem_jepa_on"] = bool(sem_enable and getattr(opt, "jepa_on_content", 0) and getattr(opt, "jepa_tokens", False))

    cfg["jepa_on"] = jepa_on_style or jepa_on_content
    cfg["jepa_mask_ratio"] = float(getattr(opt, "jepa_mask_ratio", 0.6))
    cfg["lambda_jepa_style"] = float(getattr(opt, "lambda_jepa_style", getattr(opt, "lambda_jepa", 0.15)))
    cfg["lambda_jepa_content"] = float(getattr(opt, "lambda_jepa_content", 0.15))
    cfg["lambda_jepa_var"] = float(getattr(opt, "lambda_jepa_var", 0.05))
    cfg["lambda_jepa_cov"] = float(getattr(opt, "lambda_jepa_cov", 0.05))
    cfg["lambda_jepa_kd"] = float(getattr(opt, "lambda_jepa_kd", 0.05))
    cfg["jepa_hidden_mult"] = int(getattr(opt, "jepa_hidden_mult", 2))
    cfg["jepa_use_teacher"] = bool(int(getattr(opt, "jepa_use_teacher", 1)) != 0)
    cfg["jepa_every"] = max(1, int(getattr(opt, "jepa_every", 2)))
    cfg["jepa_heads"] = int(getattr(opt, "jepa_heads", 4))
    cfg["jepa_norm"] = bool(int(getattr(opt, "jepa_norm", 1)) != 0)
    cfg["jepa_bias_high"] = float(getattr(opt, "jepa_mask_bias_high", 2.0))

    try:
        _usr = [float(x) for x in str(getattr(opt, "jepa_scale_weights", "2,2,1.5,1,0.75,0.5")).split(",") if x.strip()]
        if _usr:
            jepa_scale_w = torch.tensor(_usr, dtype=torch.float32, device=dev)
        else:
            jepa_scale_w = torch.tensor([2.0, 2.0, 1.5, 1.0, 0.75, 0.5], dtype=torch.float32, device=dev)
    except Exception:
        jepa_scale_w = torch.tensor([2.0, 2.0, 1.5, 1.0, 0.75, 0.5], dtype=torch.float32, device=dev)
    cfg["jepa_scale_w"] = jepa_scale_w

    cfg["feat_switch_epoch"] = int(getattr(opt, "feat_switch_epoch", getattr(opt, "recon_epochs", 2)))
    cfg["ema_every"] = int(getattr(opt, "ema_update_every", 1))
    cfg["nce_m"] = float(getattr(opt, "nce_m", 0.999))

    cfg["_denorm"] = _denorm
    cfg["_triplet_grid"] = _triplet_grid

    # --- Cycle scheduler A / mix / recon
    cycle_sched = CycleScheduler(
        base_adv=int(getattr(opt, "adv_only_epochs", 2)),
        base_mix=int(getattr(opt, "adv_mix_epochs", 0)),
        base_rec=int(getattr(opt, "recon_epochs", 2)),
        adv_boost=max(0, int(getattr(opt, "adv_boost", 0))),
        b_boost=max(0, int(getattr(opt, "b_boost", 0))),
        skip_amix=bool(getattr(opt, "skip_amix", False)),
    )
    cfg["cycle_sched"] = cycle_sched

    # --- Fonctions de conditionnement style (compat avec UNetGenerator)
    def _build_style_cond(G, style_img):
        if hasattr(G, "build_style_cond"):
            return G.build_style_cond(style_img)
        return style_img

    def _build_style_cond_from_tokens(toks, tokG, for_G="A"):
        G = state["G_A"] if for_G == "A" else state["G_B"]
        if hasattr(G, "build_style_cond_from_tokens"):
            return G.build_style_cond_from_tokens(toks, tokG)
        return (toks, tokG)

    cfg["build_style_cond"] = _build_style_cond
    cfg["build_style_cond_from_tokens"] = _build_style_cond_from_tokens

    # --- Folds / loaders A/B
    current_fold = 0
    src_loader = loaders[current_fold]
    tgt_loader = loaders[(current_fold + 1) % k_folds]
    rounds_on_this_fold = 0
    fold_switch_every_rounds = int(getattr(opt, "fold_epochs", 4))

    # --- Runtime supervision (hybrid / sup_freeze)
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
    #  MODE: sup_freeze (C seul)
    # =====================================================================================
    if mode == "sup_freeze":
        run_sup_freeze_mode(
            opt=opt,
            loaders=loaders,
            G_A=G_A, G_B=G_B,
            D_A=D_A, D_B=D_B,
            opt_GA=opt_GA, opt_DA=opt_DA,
            opt_GB=opt_GB, opt_DB=opt_DB,
            dev=dev,
            writer=writer,
            tb_freq_C=tb_freq_C,
            global_step_start=state["global_step"],
        )
        if writer:
            writer.flush()
            writer.close()
        return

    # =====================================================================================
    #  MODES: auto / hybrid
    # =====================================================================================
    ema_tau = float(getattr(opt, "ema_tau", 0.0))

    epoch_wall_start = time.time()
    last_step_wall = time.time()

    optimizers = {
        "GA": opt_GA,
        "GB": opt_GB,
        "DA": opt_DA,
        "DB": opt_DB,
        "SEM": state.get("opt_SEM", None),
    }

    epochs_total = int(getattr(opt, "epochs", 1))

    # --- boucle epochs
    while state["epoch"] < epochs_total:
        epoch = int(state["epoch"])
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

        # --- λ_style_A scheduler
        try:
            style_sched_cfg = cfg.get("style_sched", None)
            λ_style_now = get_style_lambda(style_sched_cfg, epoch) if style_sched_cfg is not None else cfg.get("λ_style_A", λ_style_max)
        except Exception:
            λ_style_now = cfg.get("λ_style_A", λ_style_max)

        cfg["λ_style_A"] = float(λ_style_now)
        cfg["style_gain_A"] = float(getattr(opt, "style_gain_A", cfg["λ_style_A"]))

        budgets = cycle_sched.budgets()
        print(
            f"\n📅 Epoch {epoch + 1:03d}/{epochs_total}"
            f" | ROUND={budgets['round']}  A={budgets['A_done']}/{budgets['adv'] + budgets['mix']}"
            f"  R={budgets['R_done']}/{budgets['rec']}  | phase={phase:<6}"
            f" | λN={λN:.3f} λL1/ID={λR:.3f} λ_style_A={cfg['λ_style_A']:.3f}"
        )

        if writer:
            writer.add_scalar("phase/epoch", float(epoch), state["global_step"])
            writer.add_scalar("A/style_lambda_epoch", float(cfg["λ_style_A"]), epoch)
            _tb_add_scalars(writer, _get_lrs(optimizers), epoch, prefix="epoch_")
            if state.get("opt_SEM", None) is not None:
                try:
                    writer.add_scalar("lr/SEM_epoch_start", float(state["opt_SEM"].param_groups[0]["lr"]), epoch)
                except Exception:
                    pass

        # =================================================================================
        #  Boucle batch : phases A/B
        # =================================================================================
        if mode in ["auto", "hybrid"]:
            pbar = tqdm(range(nbatchs), ncols=180, leave=False, mininterval=0.5, dynamic_ncols=True)
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

                x = x.to(dev, non_blocking=True)
                y = y.to(dev, non_blocking=True)

                try:
                    if str(phase).startswith("A"):
                        train_step_phase_A(
                            x=x, y=y,
                            state=state, cfg=cfg,
                            epoch_meters=epoch_meters,
                            writer=writer,
                        )
                    else:
                        train_step_phase_B(
                            x=x,
                            state=state, cfg=cfg,
                            epoch_meters=epoch_meters,
                            writer=writer,
                        )

                    # IMPORTANT : train_step_phase_A/B incrémentent déjà global_step.
                    step = int(state["global_step"])

                    # -----------------------------
                    # SEM Scheduler step (by step)
                    # -----------------------------
                    sch_sem = state.get("sch_SEM", None)
                    if sch_sem is not None:
                        sem_sched = str(getattr(opt, "sem_lr_sched", "none")).lower().strip()
                        sem_sched_by = str(getattr(opt, "sem_sched_by", "epoch")).lower().strip()
                        if sem_sched == "warmup_cosine" and sem_sched_by == "step":
                            sem_every = max(1, int(getattr(opt, "sem_every", 1)))
                            sem_on_B = bool(int(getattr(opt, "sem_on_B", 1)) != 0)
                            sem_update_now = (step % sem_every == 0) and (sem_on_B or str(phase).startswith("A"))
                            if sem_update_now and state.get("opt_SEM", None) is not None:
                                state["sem_lr_step"] = int(state.get("sem_lr_step", 0)) + 1
                                sch_sem.step()
                                if writer and tb_freq > 0 and (step % tb_freq == 0):
                                    try:
                                        lr_now = float(state["opt_SEM"].param_groups[0]["lr"])
                                        writer.add_scalar("lr/SEM_sched", lr_now, step)
                                    except Exception:
                                        pass

                    now = time.time()
                    dt = max(1e-6, now - last_step_wall)
                    last_step_wall = now

                    scalars = _meters_to_scalars(epoch_meters)
                    scalars["time/batch_sec"] = float(dt)
                    bs = int(getattr(opt, "batch_size", 0) or 0)
                    if bs > 0:
                        scalars["time/img_sec"] = float(bs / dt)

                    postfix = _format_postfix(scalars, postfix_keys)
                    if postfix:
                        pbar.set_postfix_str(postfix)

                    if print_freq > 0 and (step % print_freq == 0):
                        _print_kv_line(
                            f"[step {step}]",
                            scalars,
                            tuple(list(postfix_keys) + ["time/batch_sec", "time/img_sec"])
                        )

                    if writer and tb_freq > 0 and (step % tb_freq == 0):
                        _tb_add_scalars(writer, _get_lrs(optimizers), step)
                        _tb_add_scalars(writer, scalars, step, prefix="train/")
                        _tb_maybe_flush(writer, every_steps=int(getattr(opt, "tb_flush", tb_freq)), step=step)

                    # ✅ Checkpoint: en mode "step", on sauvegarde au fil de l'entraînement.
                    # En mode "epoch", la sauvegarde est faite une seule fois en fin d'époque (voir plus bas).
                    if save_freq_mode == "step":
                        do_save, reason = should_save_ckpt(
                            save_mode=save_freq_mode,
                            interval=save_freq_interval,
                            step=step,
                            epoch=epoch,
                            epochs_total=epochs_total,
                            final_save=False,
                        )
                    else:
                        do_save, reason = (False, "")

                    if do_save:
                        save_checkpoint(
                            epoch,
                            G_A, D_A, G_B, D_B,
                            opt_GA, opt_DA, opt_GB, opt_DB,
                            step,
                            out_dir,
                            use_safetensors=ckpt_use_safetensors,
                            safe_write=ckpt_safe_write,
                            sem_model=state.get("SEM", None),
                            opt_sem=state.get("opt_SEM", None),
                            T_A=state.get("T_A", None),
                            T_B=state.get("T_B", None),
                            tokJEPA_A=state.get("tokJEPA_A", None),
                            tokJEPA_B=state.get("tokJEPA_B", None),
                        )
                        save_state_json(epoch, step, opt, out_dir)
                        tqdm.write(f"[CKPT] saved ({reason}) @epoch={epoch} step={step}")

                except Exception as ex:
                    msg = f"[ERROR] step={state.get('global_step', -1)} epoch={epoch} {type(ex).__name__}: {ex}"
                    tqdm.write(msg)
                    if getattr(opt, "print_trace_on_error", False):
                        import traceback as _tb
                        tqdm.write(_tb.format_exc())
                    if writer:
                        import traceback as _tb
                        _tb_add_text(writer, "errors/exception", f"{msg}\n{_tb.format_exc()}", int(state.get("global_step", 0)))
                    state["global_step"] = int(state.get("global_step", 0)) + 1
                    continue

        # --- Supervision hybride (phase C) après warmup
        if mode == "hybrid" and (epoch + 1) > int(getattr(opt, "warmup_epochs", 0)):
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
                global_step=int(state["global_step"]),
            )

            step = int(state["global_step"])
            do_save, reason = should_save_ckpt(
                save_mode=save_freq_mode,
                interval=save_freq_interval,
                step=step,
                epoch=epoch,
                epochs_total=epochs_total,
                final_save=False,
            )
            if do_save:
                save_checkpoint(
                    epoch,
                    G_A, D_A, G_B, D_B,
                    opt_GA, opt_DA, opt_GB, opt_DB,
                    step,
                    out_dir,
                    use_safetensors=ckpt_use_safetensors,
                    safe_write=ckpt_safe_write,
                    sem_model=state.get("SEM", None),
                    opt_sem=state.get("opt_SEM", None),
                    T_A=state.get("T_A", None),
                    T_B=state.get("T_B", None),
                    tokJEPA_A=state.get("tokJEPA_A", None),
                    tokJEPA_B=state.get("tokJEPA_B", None),
                )
                save_state_json(epoch, step, opt, out_dir)
                print(f"[CKPT] saved ({reason}) @epoch={epoch} step={step}")

        # --- EMA D_B <- D_A (optionnel)
        if ema_tau > 0:
            with torch.no_grad():
                for pA, pB in zip(D_A.parameters(), D_B.parameters()):
                    pB.lerp_(pA, ema_tau)
            opt_DB.state = defaultdict(dict)
        if writer:
            writer.add_scalar("EMA/tau", float(ema_tau), epoch)

        # --- Résumé epoch
        print_epoch_summary(epoch, epoch_meters)

        if writer:
            epoch_scalars = _meters_to_scalars(epoch_meters)
            _tb_add_scalars(writer, epoch_scalars, epoch, prefix="epoch/")
            wall = time.time() - epoch_wall_start
            writer.add_scalar("time/wall_sec", float(wall), epoch)
            writer.flush()

        # --- step scheduler / folds
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

        # -----------------------------
        # SEM Scheduler step (by epoch)
        # -----------------------------
        sch_sem = state.get("sch_SEM", None)
        if sch_sem is not None:
            sem_sched = str(getattr(opt, "sem_lr_sched", "none")).lower().strip()
            sem_sched_by = str(getattr(opt, "sem_sched_by", "epoch")).lower().strip()
            if sem_sched == "warmup_cosine" and sem_sched_by == "epoch":
                sch_sem.step()
                if writer and state.get("opt_SEM", None) is not None:
                    try:
                        writer.add_scalar("lr/SEM_epoch", float(state["opt_SEM"].param_groups[0]["lr"]), epoch)
                    except Exception:
                        pass

        # ✅ Checkpoint "epoch" + "final" (une seule fois à la fin de l'epoch)
        step_now = int(state["global_step"])
        do_save_epoch, reason_epoch = should_save_ckpt(
            save_mode=save_freq_mode,
            interval=save_freq_interval,
            step=step_now,
            epoch=epoch,
            epochs_total=epochs_total,
            final_save=True,
        )
        if do_save_epoch:
            save_checkpoint(
                epoch,
                G_A, D_A, G_B, D_B,
                opt_GA, opt_DA, opt_GB, opt_DB,
                step_now,
                out_dir,
                use_safetensors=ckpt_use_safetensors,
                safe_write=ckpt_safe_write,
                sem_model=state.get("SEM", None),
                opt_sem=state.get("opt_SEM", None),
                T_A=state.get("T_A", None),
                T_B=state.get("T_B", None),
                tokJEPA_A=state.get("tokJEPA_A", None),
                tokJEPA_B=state.get("tokJEPA_B", None),
            )
            save_state_json(epoch, step_now, opt, out_dir)
            print(f"[CKPT] saved ({reason_epoch}) @epoch={epoch} step={step_now}")

        state["epoch"] += 1

    if writer:
        writer.flush()
        writer.close()


# =========================================================================================
# Optional: entrypoint direct (si tu lances ce fichier)
# =========================================================================================
if __name__ == "__main__":
    try:
        from config import get_opts
    except Exception as e:
        raise RuntimeError(f"Impossible d'importer config.get_opts(): {e}")
    opts = get_opts()
    train_alternating(opts)
