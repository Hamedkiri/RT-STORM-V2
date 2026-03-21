import random
from torch import autograd
from collections import defaultdict
from torchvision import utils as vutils
import collections
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Subset
from tqdm import tqdm

from models.sup_heads import SupHeads
from models.fusion_head import VectorGatedFusionHead
from training.checkpoint import save_supheads_rich, save_sem_backbone_rich, save_checkpoint, save_state_json

# Import des vraies perturbations / pertes texture (FFT + SWD)
from training.texture_fft_swd import (
    spectral_noise,  # bruit spectral (texture)
)
from training.scheduler import ensure_sem_scheduler_in_state, step_sem_scheduler, sem_scheduler_get_lr

# =========================================================================================
#                       Meters / résumé d'époque (A/B/C)
# =========================================================================================

class AvgMeter:
    """Moyenne robuste (ignore NaN/inf, permet d'ajouter avec un poids c)."""
    __slots__ = ("s", "n")

    def __init__(self):
        self.s = 0.0
        self.n = 0

    def add(self, v, c: int = 1):
        try:
            v = float(v)
        except Exception:
            return
        if not math.isfinite(v):
            return
        self.s += v * c
        self.n += c

    @property
    def avg(self):
        return (self.s / max(self.n, 1)) if self.n else float("nan")


def new_epoch_meters():
    """
    Crée les compteurs par phase :
      - meters["A"] : dict de AvgMeter pour la phase A (x→y, adv/style/NCE/L1, etc.)
      - meters["B"] : idem pour la phase B (reconstruction, λ_style_B_dyn, etc.)
      - meters["C"] : phase supervisée (sup_freeze / hybrid)
    """
    return {
        "A": collections.defaultdict(AvgMeter),
        "B": collections.defaultdict(AvgMeter),
        "C": collections.defaultdict(AvgMeter),
        "SEM": collections.defaultdict(AvgMeter),
    }


def _format_phase(phase_name: str, meters_dict: dict) -> str:
    """Formate proprement une ligne de résumé pour une phase A/B/C."""
    if not meters_dict:
        return f"{phase_name}: (no data)"

    keys_order = [
        "D_loss", "G_adv", "NCE", "NCE_content2",
        "λ_style_A", "styleTok", "style_gain_A",
        "styleTok_B", "style_gain_B",
        "L1", "L1_idt",
        "CE", "total",
        "λ_style_B_dyn",
        "JEPA_style", "JEPA_content",
    ]
    seen = set()
    parts = []

    for k in keys_order + [k for k in meters_dict.keys() if k not in keys_order]:
        if k in meters_dict and k not in seen:
            parts.append(f"{k}={meters_dict[k].avg:.4f}")
            seen.add(k)

    return f"{phase_name}: " + (" | ".join(parts) if parts else "(no data)")


def print_epoch_summary(epoch_idx: int, meters: dict):
    bar = "─" * 88
    msg = []
    msg.append("\n" + bar)
    msg.append(f"📊  RÉSUMÉ ÉPOQUE {epoch_idx + 1:03d}")
    msg.append(_format_phase("A", meters.get("A", {})))
    msg.append(_format_phase("B", meters.get("B", {})))
    msg.append(_format_phase("C", meters.get("C", {})))
    msg.append(_format_phase("SEM", meters.get("SEM", {})))
    msg.append(bar + "\n")
    tqdm.write("\n".join(msg))


def fmt_phase(ph_name, meters_dict):
    """Version simple si tu veux juste formatter sans le header complet."""
    if not meters_dict:
        return f"{ph_name}: (no data)"
    keys = [
        "D_loss", "G_adv", "NCE", "NCE_content2",
        "λ_style_A", "styleTok", "style_gain_A",
        "L1", "L1_idt",
        "styleTok_B", "style_gain_B",
        "total", "λ_style_B_dyn", "CE",
        "JEPA_style", "JEPA_content",
    ]
    seen = set()
    parts = []
    for k in keys + [k for k in meters_dict.keys() if k not in keys]:
        if k in meters_dict and k not in seen:
            parts.append(f"{k}={meters_dict[k].avg:.4f}")
            seen.add(k)
    return f"{ph_name}: " + (" | ".join(parts) if parts else "(no data)")


# =========================================================================================
#                       Scheduler pour style_lambda (phase A)
# =========================================================================================

def get_style_lambda(epoch: int, cfg: dict, base_lambda_key: str = "λ_style_A") -> float:
    """
    Renvoie le lambda_style effectif pour la phase A.

    On suppose que le parser fournit notamment :
      --style_lambda        (valeur max / "cible" = λ_style_A dans cfg)
      --style_lambda_min    (valeur de départ)
      --style_lambda_sched  none | linear | cosine | exp | piecewise
      --style_lambda_warmup nb d'epochs pour passer de min à max

    Mapping dans cfg (fait dans train_style_disentangle.py) :
      cfg["λ_style_A"]          (base_lambda_key, vient de style_lambda)
      cfg["style_lambda_min"]
      cfg["style_lambda_max"]   (par défaut = cfg["λ_style_A"])
      cfg["style_lambda_sched"]
      cfg["style_lambda_warmup"]
      cfg["epochs"] / ["total_epochs"] / ["max_epochs"] (optionnel)

    - "none"      → valeur fixe cfg[base_lambda_key]
    - "linear"    → interpolation linéaire min → max pendant warmup
    - "cosine"    → rampe cosinus min → max pendant warmup
    - "exp"       → interpolation exponentielle min → max pendant warmup
    - "piecewise" → schéma en 3 pièces (min / mid / max) sur la phase de warmup
    """
    lam_base = float(cfg.get(base_lambda_key, 1.0))

    sched = str(cfg.get("style_lambda_sched", "none")).lower()
    if sched == "none":
        return lam_base

    lam_min = float(cfg.get("style_lambda_min", lam_base))
    lam_max = float(cfg.get("style_lambda_max", lam_base))

    total_epochs = int(
        cfg.get(
            "epochs",
            cfg.get("total_epochs", cfg.get("max_epochs", max(epoch + 1, 1))),
        )
    )
    default_warmup = max(1, int(0.2 * total_epochs))
    warmup = int(cfg.get("style_lambda_warmup", default_warmup))

    e = max(0, epoch)
    if warmup <= 0:
        t = 1.0
    else:
        t = min(1.0, e / float(warmup))  # t ∈ [0,1] sur la phase de warmup

    if sched == "linear":
        lam = lam_min + t * (lam_max - lam_min)

    elif sched == "cosine":
        lam = lam_min + 0.5 * (1.0 - math.cos(math.pi * t)) * (lam_max - lam_min)

    elif sched == "exp":
        # interpolation exponentielle entre min et max
        if lam_min <= 0:
            lam_min = 1e-6
        ratio = lam_max / lam_min
        lam = lam_min * (ratio ** t)

    elif sched == "piecewise":
        # Schéma simple en 3 morceaux :
        #   - t ∈ [0, 1/3)   → lam_min
        #   - t ∈ [1/3, 2/3) → (lam_min + lam_max)/2
        #   - t ∈ [2/3, 1]   → lam_max
        if t < (1.0 / 3.0):
            lam = lam_min
        elif t < (2.0 / 3.0):
            lam = 0.5 * (lam_min + lam_max)
        else:
            lam = lam_max

    else:
        # mode inconnu → fallback valeur de base
        lam = lam_base

    return float(lam)


# =========================================================================================
#                       Mode sup_freeze (C seul, G/D gelés)
# =========================================================================================

def run_sup_freeze_mode(
        opt,
        loaders,
        G_A, G_B, D_A, D_B,
        opt_GA, opt_DA, opt_GB, opt_DB,
        dev,
        writer=None,
        tb_freq_C: int = 50,
        global_step_start: int = 0,
):
    """
    Mode 'sup_freeze' :
      - G_A, G_B, D_A, D_B sont gelés (no grad).
      - On entraîne uniquement des SupHeads par fold.
      - Pour chaque fold "train", on évalue sur tous les folds, et on sauvegarde
        des SupHeads 'best' et 'last' dans save_dir/sup_freeze/fold_xx/.

    Gestion des sup_feat_type :
      - 'tok6', 'tok6_w' : on utilise des delta_weights de longueur 6
      - 'tokG', 'style_tok', 'tok6_mean' : delta_weights de longueur 5
      - 'cont_tok', 'cont_tok_vit' (et variantes) : pas de delta_weights → None
    """
    from copy import deepcopy

    # --- Optional: use a semantic ResNet backbone as feature source for SupHeads ---
    sup_feat_source = str(getattr(opt, "sup_feat_source", "generator")).lower().strip()
    sup_sem_imagenet_norm = int(getattr(opt, "sup_sem_imagenet_norm", 1)) == 1

    sem_backbone = None
    sem_out_channels = None

    if sup_feat_source in ("sem_resnet50", "fusion"):
        # Reuse the same robust builder as detection training
        from training.train_detection_transformer import _build_sem_resnet_backbone

        sem_backbone, sem_out_channels = _build_sem_resnet_backbone(
            pretrained=bool(int(getattr(opt, "sem_pretrained", 1))),
            arch=str(getattr(opt, "det_sem_backbone", "resnet50")),
            return_layer=str(getattr(opt, "det_sem_return_layer", "layer4")),
            pretrained_path=str(getattr(opt, "sem_pretrained_path", "") or ""),
            strict=bool(int(getattr(opt, "sem_pretrained_strict", 0))),
            verbose=bool(int(getattr(opt, "sem_pretrained_verbose", 1))),
        )
        sem_backbone = sem_backbone.to(dev)
        sem_backbone.eval()
        for p in sem_backbone.parameters():
            p.requires_grad_(False)

        # ImageNet normalization constants
        _im_mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        _im_std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

        def _sem_feats(imgs: torch.Tensor) -> torch.Tensor:
            # imgs expected shape: (B,C,H,W)
            x = imgs
            if x.dim() != 4:
                raise ValueError(f"sup_freeze: expected imgs BCHW, got {tuple(x.shape)}")
            if x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1)
            if sup_sem_imagenet_norm:
                # many GAN pipelines use [-1,1]; convert to [0,1] then normalize
                x = (x + 1.0) * 0.5
                x = x.clamp(0.0, 1.0)
                x = (x - _im_mean) / _im_std
            out = sem_backbone(x)
            if isinstance(out, dict):
                feat = out.get("0", None)
                if feat is None:
                    feat = next(iter(out.values()))
            else:
                feat = out
            # Global average pooling
            feat = feat.mean(dim=(2, 3))
            return feat

    def _flatten_for_fusion(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            return x.mean(dim=(2, 3))
        if x.dim() == 3:
            return x.mean(dim=1)
        return x

    def _style_feats(G_sup, imgs: torch.Tensor, feat_type: str, δw_str: str | None) -> torch.Tensor:
        return G_sup.sup_features(imgs, feat_type, delta_weights=δw_str)

    def _get_sup_feats(G_sup, imgs: torch.Tensor, feat_type: str, δw_str: str | None) -> torch.Tensor:
        if sup_feat_source == "sem_resnet50":
            return _sem_feats(imgs)
        if sup_feat_source == "fusion":
            style_feat = _flatten_for_fusion(_style_feats(G_sup, imgs, feat_type, δw_str))
            sem_feat = _flatten_for_fusion(_sem_feats(imgs))
            if not hasattr(G_sup, "sup_fusion") or G_sup.sup_fusion is None:
                raise RuntimeError("sup_feat_source=fusion mais G_sup.sup_fusion n'est pas initialisé.")
            return G_sup.sup_fusion(style_feat, sem_feat)
        # NOTE: for cont_tok/cont_tok_vit, δw_str should be None
        return _style_feats(G_sup, imgs, feat_type, δw_str)


    def _parse_dw(dw: str, need: int) -> str:
        vals = [float(t) for t in str(dw).split(",") if t.strip()]
        if len(vals) == need:
            return ",".join(str(v) for v in vals)
        if len(vals) == 1:
            vals = vals * need
        elif len(vals) < need:
            vals = vals + [vals[-1]] * (need - len(vals))
        else:
            vals = vals[:need]
        return ",".join(str(v) for v in vals)

    @torch.no_grad()
    def _sup_metrics(out, y_true, num_classes):
        pred = out.argmax(1)
        if y_true.numel() == 0:
            return 0.0, 0.0, 0.0
        acc = float((pred == y_true).float().mean().item())
        Pm, Rm = [], []
        for c in range(num_classes):
            tp = ((pred == c) & (y_true == c)).sum().item()
            fp = ((pred == c) & (y_true != c)).sum().item()
            fn = ((pred != c) & (y_true == c)).sum().item()
            if (tp + fp) > 0:
                Pm.append(tp / (tp + fp + 1e-9))
            if (tp + fn) > 0:
                Pm_val = tp / (tp + fn + 1e-9)
                Rm.append(Pm_val)
        return acc, (float(np.mean(Pm)) if Pm else 0.0), (float(np.mean(Rm)) if Rm else 0.0)

    @torch.no_grad()
    def _eval_on_fold(G_sup, loader, tasks, feat_type, δw_str, writer, global_step, tag_prefix):
        G_sup.eval()
        if hasattr(G_sup, "sup_heads") and G_sup.sup_heads is not None:
            G_sup.sup_heads.eval()
        if hasattr(G_sup, "sup_fusion") and G_sup.sup_fusion is not None:
            G_sup.sup_fusion.eval()

        agg = {t: {"ce": 0.0, "n": 0, "acc": 0.0, "P": 0.0, "R": 0.0, "H": 0.0} for t in tasks}

        for batch in loader:
            if len(batch) >= 2:
                imgs, raw = batch[0], batch[1]
            else:
                imgs, raw = batch
            imgs = imgs.to(next(G_sup.parameters()).device)

            # NOTE : pour cont_tok / cont_tok_vit, δw_str sera None
            feats = _get_sup_feats(G_sup, imgs, feat_type, δw_str)
            logits, attn = G_sup.sup_heads(feats, return_attn=True)

            if not isinstance(logits, dict):
                logits = {"default": logits}
            if not isinstance(raw, dict):
                raw = {"default": raw}

            B = imgs.size(0)
            for t, out in logits.items():
                if t not in raw:
                    continue
                y = torch.as_tensor(raw[t], device=imgs.device, dtype=torch.long)
                mask = (y >= 0) & (y < out.size(1))
                if mask.any():
                    ce = F.cross_entropy(out[mask], y[mask]).item()
                    acc, Pm, Rm = _sup_metrics(out[mask], y[mask], out.size(1))
                    n = int(mask.sum().item())
                    agg[t]["ce"] += ce * n
                    agg[t]["acc"] += acc * n
                    agg[t]["P"] += Pm * n
                    agg[t]["R"] += Rm * n
                    agg[t]["n"] += n
                    if isinstance(attn, dict) and t in attn and "entropy" in attn[t]:
                        agg[t]["H"] += float(attn[t]["entropy"]) * B

        if writer:
            for t, d in agg.items():
                n = max(1, d["n"])
                writer.add_scalars(
                    f"{tag_prefix}/{t}",
                    {
                        "CE": d["ce"] / n,
                        "acc": d["acc"] / n,
                        "P_macro": d["P"] / n,
                        "R_macro": d["R"] / n,
                        "attn_H": d["H"] / max(1, n),
                    },
                    global_step,
                )
        return agg

    global_step = global_step_start

    # G/D gelés
    freeze(G_A)
    freeze(G_B)
    freeze(D_A)
    freeze(D_B)
    G_A.eval()
    G_B.eval()
    D_A.eval()
    D_B.eval()

    # Quel générateur sert de backbone de features sup ?
    sup_from = getattr(opt, "sup_from", "GB").upper()
    G_sup = G_B if sup_from == "GB" else G_A

    # ------------------------------------------------------------------
    #  Gestion explicite de sup_feat_type + delta_weights
    # ------------------------------------------------------------------
    feat_type = str(getattr(opt, "sup_feat_type", "tok6"))
    δw_raw = str(getattr(opt, "sup_delta_weights", getattr(opt, "delta_weights", "1,1,1,1,1")))

    # Cas style-tokens (hiérarchie multi-échelles)
    if feat_type in ("tok6", "tok6_w"):
        δw_str = _parse_dw(δw_raw, 6)
    elif feat_type in ("tokG", "style_tok", "tok6_mean"):
        δw_str = _parse_dw(δw_raw, 5)
    # Cas contenu tokens (ViT / JEPA…) : PAS de delta_weights
    elif feat_type.startswith("cont_tok"):
        δw_str = None
    # Autre type exotique : on ne passe pas de delta_weights
    else:
        δw_str = None

    λ_sup = float(getattr(opt, "lambda_sup", 1.0))

    k_folds = len(loaders) if len(loaders) > 0 else 1
    eval_every = int(getattr(opt, "sup_eval_every", 1))
    reset_between = bool(getattr(opt, "sup_reset_between_folds", True))
    folds_order = list(range(k_folds)) if k_folds >= 1 else [0]

    def _make_heads_and_opt(train_fold_idx: int):
        ds = loaders[train_fold_idx].dataset
        ds = ds.dataset if isinstance(ds, Subset) else ds

        # Détermination du dict tasks : {task_name: num_classes}
        if hasattr(ds, "task_classes") and isinstance(ds.task_classes, dict) and ds.task_classes:
            tasks = {task: len(lst) for task, lst in ds.task_classes.items()}
        elif hasattr(ds, "classes") and isinstance(ds.classes, (list, tuple)) and ds.classes:
            tasks = {"default": len(ds.classes)}
        else:
            tasks = {"default": int(getattr(opt, "sup_num_classes", 2))}

        style_in_dim = int(G_sup.sup_in_dim_for(feat_type))
        real_num_scales = int(1 + getattr(G_sup, "style_levels", 5))

        if sup_feat_source == "sem_resnet50":
            in_dim = int(sem_out_channels)
            token_mode = "flat"
            G_sup.sup_fusion = None
        elif sup_feat_source == "fusion":
            fusion_dim = int(getattr(opt, "fusion_dim", 1024))
            in_dim = fusion_dim
            token_mode = "flat"
            G_sup.sup_fusion = VectorGatedFusionHead(
                style_in_dim=style_in_dim,
                sem_in_dim=int(sem_out_channels),
                fusion_dim=fusion_dim,
                dropout=float(getattr(opt, "fusion_dropout", 0.1)),
            ).to(dev)
        else:
            in_dim = style_in_dim
            token_mode = (
                "multi6"
                if feat_type in ("tok6",)
                else "single"
                if feat_type in ("tokG", "style_tok", "tok6_mean", "tok6_w")
                else "flat"
            )
            G_sup.sup_fusion = None

        G_sup.sup_heads = SupHeads(
            tasks,
            in_dim,
            num_scales=real_num_scales,
            token_mode=token_mode,
            heads=int(getattr(opt, "sup_heads_nheads", 4)),
            dropout=float(getattr(opt, "sup_heads_dropout", 0.1)),
            mlp_mult=int(getattr(opt, "sup_heads_mlp_mult", 2)),
        ).to(dev)

        for p in G_sup.parameters():
            p.requires_grad_(False)
        if sem_backbone is not None:
            for p in sem_backbone.parameters():
                p.requires_grad_(False)
        for p in G_sup.sup_heads.parameters():
            p.requires_grad_(True)

        trainable_params = list(G_sup.sup_heads.parameters())
        if hasattr(G_sup, "sup_fusion") and G_sup.sup_fusion is not None:
            for p in G_sup.sup_fusion.parameters():
                p.requires_grad_(True)
            trainable_params += list(G_sup.sup_fusion.parameters())

        opt_Sup = torch.optim.Adam(
            trainable_params,
            lr=float(getattr(opt, "sup_lr", 1e-4)),
            betas=(0.9, 0.999),
        )
        return tasks, opt_Sup

    for stage, tr_fold in enumerate(folds_order):
        if stage == 0 or reset_between:
            tasks, opt_Sup = _make_heads_and_opt(tr_fold)

        train_loader = loaders[tr_fold]
        val_loaders = loaders

        sup_dir = Path(opt.save_dir) / "sup_freeze" / f"fold_{tr_fold:02d}"
        sup_dir.mkdir(parents=True, exist_ok=True)

        best_val = float("inf")
        best_state = None
        best_epoch = -1

        print(
            f"✓ SupHeads (sup_freeze) prêtes | source={sup_feat_source} | feat_type={feat_type} | "
            f"in_dim={getattr(G_sup.sup_heads, 'in_dim', None)} | token_mode={getattr(G_sup.sup_heads, 'token_mode', None)} | "
            f"fusion_dim={getattr(getattr(G_sup, 'sup_fusion', None), 'fusion_dim', 'NA')}"
        )
        if writer:
            writer.add_text(
                "C/run",
                f"sup_freeze — stage {stage} — train on fold {tr_fold} "
                f"(sup_feat_source={sup_feat_source}, feat_type={feat_type}, delta_w_str={δw_str}, fusion_dim={getattr(getattr(G_sup, 'sup_fusion', None), 'fusion_dim', 'NA')})",
                global_step,
            )

        for epoch in range(opt.epochs):
            epoch_meters = new_epoch_meters()
            G_sup.sup_heads.train(True)
            if hasattr(G_sup, "sup_fusion") and G_sup.sup_fusion is not None:
                G_sup.sup_fusion.train(True)

            pbar = tqdm(
                train_loader,
                desc=f"C[sup_freeze]-fold{tr_fold} ep{epoch + 1}/{opt.epochs}",
                ncols=160,
                leave=False,
            )

            for batch in pbar:
                if len(batch) == 3:
                    imgs, raw, _paths = batch
                elif len(batch) >= 2:
                    imgs, raw = batch[0], batch[1]
                    _paths = None
                else:
                    imgs, raw = batch
                    _paths = None

                imgs = imgs.to(dev)

                # NOTE : pour cont_tok / cont_tok_vit, δw_str = None
                feats = _get_sup_feats(G_sup, imgs, feat_type, δw_str)
                logits, attn = G_sup.sup_heads(feats, return_attn=True)

                if not isinstance(logits, dict):
                    logits = {"default": logits}
                if not isinstance(raw, dict):
                    raw = {"default": raw}

                terms = []
                log_ce = {}

                for t, out in logits.items():
                    y = torch.as_tensor(raw[t], device=imgs.device, dtype=torch.long)
                    mask = (y >= 0) & (y < out.size(1))
                    if mask.any():
                        ce = F.cross_entropy(out[mask], y[mask])
                        terms.append(ce)
                        log_ce[t] = float(ce.item())

                if terms:
                    loss = torch.stack(terms).sum()
                    opt_Sup.zero_grad(set_to_none=True)
                    (λ_sup * loss).backward()
                    opt_Sup.step()

                    epoch_meters["C"]["CE"].add(float(loss.item()))
                    for t, v in log_ce.items():
                        epoch_meters["C"][f"CE/{t}"].add(v)

                if writer and (global_step % tb_freq_C == 0):
                    for t, v in log_ce.items():
                        writer.add_scalar(
                            f"C/train/CE_fold{tr_fold}/{t}", v, global_step
                        )

                global_step += 1

            # Évaluation inter-folds
            if ((epoch + 1) % eval_every) == 0 or ((epoch + 1) == opt.epochs):
                val_sum = 0.0
                val_count = 0
                for vf, vloader in enumerate(val_loaders):
                    metrics = _eval_on_fold(
                        G_sup,
                        vloader,
                        tasks,
                        feat_type,
                        δw_str,
                        writer,
                        global_step,
                        tag_prefix=f"C/eval/fold{vf}_when_tr{tr_fold}",
                    )
                    for t, d in metrics.items():
                        if d["n"] > 0:
                            val_sum += d["ce"]
                            val_count += d["n"]
                mean_ce = (val_sum / max(1, val_count)) if val_count > 0 else float("inf")

                if mean_ce < best_val:
                    best_val = mean_ce
                    best_epoch = epoch + 1
                    best_state = deepcopy(G_sup.sup_heads.state_dict())
                    save_supheads_rich(
                        G_sup.sup_heads,
                        sup_dir / f"SupHeads_best_fold{tr_fold}.pth",
                        safe_write=True,
                    )
                    if hasattr(G_sup, "sup_fusion") and G_sup.sup_fusion is not None:
                        torch.save({
                            "state_dict": G_sup.sup_fusion.state_dict(),
                            "style_in_dim": getattr(G_sup.sup_fusion, "style_in_dim", None),
                            "sem_in_dim": getattr(G_sup.sup_fusion, "sem_in_dim", None),
                            "fusion_dim": getattr(G_sup.sup_fusion, "fusion_dim", None),
                        }, sup_dir / f"FusionHead_best_fold{tr_fold}.pth")
                    (sup_dir / "best_meta.json").write_text(
                        json.dumps(
                            {
                                "best_epoch": best_epoch,
                                "best_mean_CE": float(best_val),
                                "feat_type": feat_type,
                                "sup_feat_source": sup_feat_source,
                                "delta_weights": δw_str,
                                "token_mode": getattr(
                                    G_sup.sup_heads, "token_mode", "multi6"
                                ),
                                "in_dim": getattr(G_sup.sup_heads, "in_dim", None),
                                "fusion_dim": getattr(getattr(G_sup, "sup_fusion", None), "fusion_dim", None),
                                "tasks": getattr(G_sup.sup_heads, "tasks", {}),
                            },
                            indent=2,
                        )
                    )
                    print(
                        f"✓ [fold {tr_fold}] nouveau BEST (epoch {best_epoch}) — mean CE={best_val:.4f}"
                    )

            print_epoch_summary(epoch, epoch_meters)

            save_freq = getattr(opt, "save_freq", 1)
            try:
                save_freq_int = int(save_freq)
            except Exception:
                save_freq_int = int(getattr(opt, "epochs", 1))

            # Extra periodic ckpt control (epoch_ckpt_interval) — useful to snapshot earlier than end.
            try:
                epoch_ckpt_interval = int(getattr(opt, "epoch_ckpt_interval", 0) or 0)
            except Exception:
                epoch_ckpt_interval = 0

            do_save_epoch = ((epoch + 1) % max(1, save_freq_int) == 0) or ((epoch + 1) == int(getattr(opt, "epochs", 1)))
            do_save_extra = (epoch_ckpt_interval > 0) and ((epoch + 1) % epoch_ckpt_interval == 0)

            if do_save_epoch or do_save_extra:
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
                    global_step,
                    Path(opt.save_dir),
                    sem_model=(sem_backbone if sup_feat_source in ("sem_resnet50", "fusion") else None),
                    sem_filename="SemBackbone",
                )
                if hasattr(G_sup, "sup_fusion") and G_sup.sup_fusion is not None:
                    torch.save({
                        "state_dict": G_sup.sup_fusion.state_dict(),
                        "style_in_dim": getattr(G_sup.sup_fusion, "style_in_dim", None),
                        "sem_in_dim": getattr(G_sup.sup_fusion, "sem_in_dim", None),
                        "fusion_dim": getattr(G_sup.sup_fusion, "fusion_dim", None),
                    }, sup_dir / f"FusionHead_last_fold{tr_fold}.pth")
                save_state_json(epoch, global_step, opt, Path(opt.save_dir))
                save_supheads_rich(
                    G_sup.sup_heads,
                    sup_dir / f"SupHeads_last_fold{tr_fold}.pth",
                    safe_write=True,
                )

        if best_state is not None:
            G_sup.sup_heads.load_state_dict(best_state, strict=False)
            save_supheads_rich(
                G_sup.sup_heads,
                sup_dir / f"SupHeads_best_fold{tr_fold}.pth",
                safe_write=True,
            )
        save_supheads_rich(
            G_sup.sup_heads,
            sup_dir / f"SupHeads_last_fold{tr_fold}.pth",
            safe_write=True,
        )

        if hasattr(G_sup, "sup_fusion") and G_sup.sup_fusion is not None:
            torch.save({
                "state_dict": G_sup.sup_fusion.state_dict(),
                "style_in_dim": getattr(G_sup.sup_fusion, "style_in_dim", None),
                "sem_in_dim": getattr(G_sup.sup_fusion, "sem_in_dim", None),
                "fusion_dim": getattr(G_sup.sup_fusion, "fusion_dim", None),
            }, sup_dir / f"FusionHead_last_fold{tr_fold}.pth")
        (sup_dir / "fold_summary.json").write_text(
            json.dumps(
                {
                    "fold_index": tr_fold,
                    "best_epoch": best_epoch,
                    "best_mean_CE": float(best_val),
                    "epochs": int(opt.epochs),
                    "feat_type": feat_type,
                    "sup_feat_source": sup_feat_source,
                    "delta_weights": δw_str,
                    "token_mode": getattr(G_sup.sup_heads, "token_mode", "multi6"),
                    "in_dim": getattr(G_sup.sup_heads, "in_dim", None),
                    "fusion_dim": getattr(getattr(G_sup, "sup_fusion", None), "fusion_dim", None),
                    "tasks": getattr(G_sup.sup_heads, "tasks", {}),
                },
                indent=2,
            )
        )

    if writer:
        writer.close()

    return global_step


# =========================================================================================
#           Phase C (supervisée) en HYBRID : run_hybrid_supervised_epoch
# =========================================================================================

def run_hybrid_supervised_epoch(
        opt,
        epoch: int,
        epoch_meters: dict,
        sup_runtime: dict,
        src_loader,
        G_A, G_B, D_A, D_B,
        opt_GA, opt_GB,
        dev,
        nbatchs: int,
        writer=None,
        tb_freq_C: int = 50,
        global_step: int = 0,
):
    def _macro_prec_recall(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int):
        eps = 1e-9
        precs, recs = [], []
        for c in range(num_classes):
            tp = ((y_pred == c) & (y_true == c)).sum().item()
            fp = ((y_pred == c) & (y_true != c)).sum().item()
            fn = ((y_pred != c) & (y_true == c)).sum().item()
            if (tp + fp) > 0:
                precs.append(tp / (tp + fp + eps))
            if (tp + fn) > 0:
                recs.append(tp / (tp + fn + eps))
        return (float(np.mean(precs)) if precs else 0.0,
                float(np.mean(recs)) if recs else 0.0)

    if not sup_runtime.get("inited", False):
        sup_from = getattr(opt, "sup_from", "GB").upper()
        G_sup = G_B if sup_from == "GB" else G_A
        sup_runtime["G_sup"] = G_sup

        feat_type = str(getattr(opt, "sup_feat_type", "tok6"))
        δw_raw = str(getattr(opt, "sup_delta_weights", getattr(opt, "delta_weights", "1,1,1,1,1")))

        def _parse_dw(dw: str, need: int) -> str:
            vals = [float(t) for t in str(dw).split(",") if t.strip()]
            if len(vals) == need:
                return ",".join(str(v) for v in vals)
            if len(vals) == 1:
                vals = vals * need
            elif len(vals) < need:
                vals = vals + [vals[-1]] * (need - len(vals))
            else:
                vals = vals[:need]
            return ",".join(str(v) for v in vals)

        δw_str = _parse_dw(δw_raw, 6 if feat_type == "tok6_w" else 5)

        sup_names = {}
        if getattr(opt, "sup_tasks_json", None):
            with open(opt.sup_tasks_json) as f:
                raw_tasks = json.load(f)
            tasks = {}
            for t, v in raw_tasks.items():
                if isinstance(v, (list, tuple)):
                    tasks[t] = len(v)
                    sup_names[t] = list(v)
                else:
                    tasks[t] = int(v)
        else:
            ds = src_loader.dataset.dataset if isinstance(src_loader.dataset, Subset) else src_loader.dataset
            if hasattr(ds, "task_classes") and isinstance(ds.task_classes, dict) and ds.task_classes:
                tasks = {task: len(lst) for task, lst in ds.task_classes.items()}
            elif hasattr(ds, "classes") and isinstance(ds.classes, (list, tuple)) and ds.classes:
                tasks = {"default": len(ds.classes)}
            else:
                tasks = {"default": int(getattr(opt, "sup_num_classes", 2))}

        in_dim = G_sup.sup_in_dim_for(feat_type)
        token_mode = (
            "multi6"
            if feat_type in ("tok6",)
            else "single"
            if feat_type in ("tokG", "style_tok", "tok6_mean", "tok6_w")
            else "flat"
        )

        recreate = True
        if hasattr(G_sup, "sup_heads") and isinstance(G_sup.sup_heads, nn.Module):
            try:
                same_in = (getattr(G_sup.sup_heads, "in_dim", None) == in_dim)
                same_tasks = (getattr(G_sup.sup_heads, "tasks", {}) == tasks)
                same_tm = (getattr(G_sup.sup_heads, "token_mode", None) == token_mode)
                recreate = not (same_in and same_tasks and same_tm)
            except Exception:
                recreate = True

        if recreate:
            G_sup.sup_heads = SupHeads(
                tasks,
                in_dim,
                num_scales=6,
                token_mode=token_mode,
                heads=int(getattr(opt, "sup_heads_nheads", 4)),
                dropout=float(getattr(opt, "sup_heads_dropout", 0.1)),
                mlp_mult=int(getattr(opt, "sup_heads_mlp_mult", 2)),
            ).to(dev)

        setattr(G_sup.sup_heads, "class_names", sup_names)

        sup_opt_lr = float(getattr(opt, "sup_lr", 1e-4))
        opt_Sup = torch.optim.Adam(
            G_sup.sup_heads.parameters(),
            lr=sup_opt_lr,
            betas=(0.9, 0.999),
        )

        sup_runtime.update(
            dict(
                inited=True,
                G_sup=G_sup,
                feat_type=feat_type,
                delta_w_str=δw_str,
                tasks=tasks,
                class_names=sup_names,
                token_mode=token_mode,
                in_dim=in_dim,
                opt_Sup=opt_Sup,
                task_map=None,
                task_map_printed=False,
                class_map={},
                class_map_built=False,
            )
        )
        print(
            f"✓ SupHeads (hybrid) prêtes (feat_type={feat_type} | "
            f"token_mode={token_mode} | in_dim={in_dim} | tasks={tasks})"
        )

    G_sup = sup_runtime["G_sup"]
    opt_Sup = sup_runtime["opt_Sup"]
    feat_type = sup_runtime["feat_type"]
    delta_w_str = sup_runtime["delta_w_str"]

    λ_sup = float(getattr(opt, "lambda_sup", 1.0))
    sup_ratio = float(getattr(opt, "sup_ratio", 0.25))

    freeze(D_A)
    freeze(D_B)
    unfreeze(G_sup)
    G_sup.train(True)

    sup_batches = max(1, int(nbatchs * sup_ratio))
    sup_iter = iter(src_loader)

    pbarC = tqdm(
        range(sup_batches),
        total=sup_batches,
        desc="C (hybrid sup+attn)",
        ncols=160,
        leave=False,
    )

    for _ in pbarC:
        try:
            batch = next(sup_iter)
        except StopIteration:
            sup_iter = iter(src_loader)
            batch = next(sup_iter)

        if len(batch) == 3:
            imgs, raw, _paths = batch
        else:
            imgs, raw = batch
            _paths = None

        imgs = imgs.to(dev)

        # mapping de tâches dataset -> sup_heads
        if sup_runtime["task_map"] is None:
            if isinstance(raw, dict):
                nk = {
                    str(k)
                    .lower()
                    .replace(" ", "")
                    .replace("_", "")
                    .replace("-", ""): k
                    for k in raw.keys()
                }
                sup_runtime["task_map"] = {
                    t: nk.get(
                        str(t)
                        .lower()
                        .replace(" ", "")
                        .replace("_", "")
                        .replace("-", ""),
                        None,
                    )
                    for t in G_sup.sup_heads.tasks
                }
            else:
                t0 = next(iter(G_sup.sup_heads.tasks))
                sup_runtime["task_map"] = {t0: "__DEFAULT__"}

            if not sup_runtime["task_map_printed"]:
                print("   [C] task mapping →", sup_runtime["task_map"])
                sup_runtime["task_map_printed"] = True

        # mapping classes dataset -> classes sup_heads
        if not sup_runtime["class_map_built"]:
            _ds_for_names = (
                src_loader.dataset.dataset
                if isinstance(src_loader.dataset, Subset)
                else src_loader.dataset
            )
            sup_class_names = getattr(G_sup.sup_heads, "class_names", {}) or {}
            ds_task_classes_attr = getattr(_ds_for_names, "task_classes", None)

            if isinstance(ds_task_classes_attr, dict) and len(ds_task_classes_attr) > 0:
                ds_task_classes = ds_task_classes_attr
            elif hasattr(_ds_for_names, "classes"):
                ds_task_classes = {"__DEFAULT__": list(_ds_for_names.classes)}
            else:
                ds_task_classes = {}

            def _build_class_map(ds_names, sup_names):
                if sup_names is None:
                    return (
                        np.arange(len(ds_names), dtype=np.int64)
                        if len(ds_names) > 0
                        else np.zeros((0,), np.int64)
                    )
                lut = {str(n).lower(): i for i, n in enumerate(sup_names)}
                out = np.full((len(ds_names),), -1, dtype=np.int64)
                for i, n in enumerate(ds_names):
                    out[i] = lut.get(str(n).lower(), -1)
                return out

            cm = {}
            for t, ds_key in sup_runtime["task_map"].items():
                if ds_key is None:
                    continue
                ds_names = list(ds_task_classes.get(ds_key, []))
                sup_names_t = sup_class_names.get(t, None)
                cm[t] = _build_class_map(ds_names, sup_names_t)

            sup_runtime["class_map"] = cm
            sup_runtime["class_map_built"] = True

        feats = G_sup.sup_features(imgs, feat_type, delta_weights=delta_w_str)

        opt_Sup.zero_grad(set_to_none=True)
        if G_sup is G_B:
            opt_GB.zero_grad(set_to_none=True)
        else:
            opt_GA.zero_grad(set_to_none=True)

        try:
            logits, extra_attn = G_sup.sup_heads(feats, return_attn=True)
        except TypeError:
            logits = G_sup.sup_heads(feats)
            extra_attn = None

        if not isinstance(logits, dict):
            logits = {"default": logits}

        task_dict = G_sup.sup_heads.tasks

        lbls_list = []
        if isinstance(raw, dict):
            Bbatch = len(next(iter(raw.values())))
            for i in range(Bbatch):
                d = {}
                for t in task_dict:
                    ds_key = sup_runtime["task_map"].get(t)
                    if ds_key is None:
                        d[t] = -1
                    else:
                        v = raw.get(ds_key, None)
                        d[t] = -1 if (v is None or v[i] is None) else int(v[i])
                lbls_list.append(d)
        else:
            t0 = next(iter(task_dict))
            Bbatch = raw.shape[0] if hasattr(raw, "shape") else len(raw)
            for i in range(Bbatch):
                lbls_list.append(
                    {
                        t0: int(
                            raw[i].item() if hasattr(raw[i], "item") else raw[i]
                        )
                    }
                )

        terms = []
        losses_step, accs_step = {}, {}
        precs_step, recalls_step = {}, {}
        attn_H_step = {}

        for t, out in logits.items():
            y_ds = torch.as_tensor(
                [d.get(t, -1) for d in lbls_list], device=dev, dtype=torch.long
            )

            if t in sup_runtime["class_map"] and hasattr(
                    sup_runtime["class_map"][t], "__len__"
            ):
                lut = sup_runtime["class_map"][t]
                y_cpu = y_ds.detach().cpu().numpy()
                y_cpu = np.where(
                    (y_cpu >= 0) & (y_cpu < len(lut)), lut[y_cpu], -1
                )
                y_t = torch.from_numpy(y_cpu).to(dev)
            else:
                y_t = y_ds

            Cc = out.size(1)
            y_t = torch.where(
                (y_t >= 0) & (y_t < Cc),
                y_t,
                torch.full_like(y_t, -1),
            )
            maskv = (y_t >= 0)
            n_val = int(maskv.sum().item())

            if n_val > 0:
                loss_t = F.cross_entropy(out[maskv], y_t[maskv])
                terms.append(loss_t)

                with torch.no_grad():
                    pred = out[maskv].argmax(1)
                    corr = int((pred == y_t[maskv]).sum().item())
                    accs_step[t] = corr / max(1, n_val)
                    losses_step[t] = float(loss_t.item())
                    Pm, Rm = _macro_prec_recall(y_t[maskv], pred, Cc)
                    precs_step[t] = float(Pm)
                    recalls_step[t] = float(Rm)

            if extra_attn and isinstance(extra_attn, dict) and t in extra_attn:
                if extra_attn[t].get("entropy", None) is not None:
                    attn_H_step[t] = float(extra_attn[t]["entropy"])

        if terms:
            loss_sup = torch.stack(terms).sum()
            (λ_sup * loss_sup).backward()
            if G_sup is G_B:
                opt_GB.step()
            else:
                opt_GA.step()
            opt_Sup.step()

            epoch_meters["C"]["CE"].add(float(loss_sup.item()))
            for k, v in losses_step.items():
                epoch_meters["C"][f"CE/{k}"].add(float(v))
            for k, v in attn_H_step.items():
                epoch_meters["C"][f"attn_H/{k}"].add(float(v))
        else:
            if writer and (global_step % tb_freq_C == 0):
                writer.add_scalar("C/loss_total", 0.0, global_step)

        global_step += 1

    unfreeze(D_A)
    unfreeze(D_B)

    return global_step


# =========================================================================================
#           Phase A  (x -> y style)   + SEM (MoCo + JEPA-content) TensorBoard logging
# =========================================================================================

def train_step_phase_A(
        x,
        y,
        state,
        cfg,
        epoch_meters,
        writer,
):
    import math
    import random
    import numpy as np
    import torch
    import torch.nn.functional as F

    # ----------------------------
    # Small safe helpers (no hard deps)
    # ----------------------------
    def _to_float(v, default=None):
        try:
            if v is None:
                return default
            if torch.is_tensor(v):
                return float(v.detach().float().mean().cpu().item())
            return float(v)
        except Exception:
            return default

    def _is_finite(v):
        try:
            return (v is not None) and (not math.isnan(float(v))) and (not math.isinf(float(v)))
        except Exception:
            return False

    def _tqdm_write(msg: str):
        try:
            from tqdm import tqdm
            tqdm.write(msg)
        except Exception:
            print(msg)

    def _get_stat(stats: dict, *keys, default=0.0):
        if not isinstance(stats, dict):
            return default
        for k in keys:
            if k in stats:
                fv = _to_float(stats.get(k), None)
                if fv is not None:
                    return fv
        return default

    def _ema_update(state_dict, key: str, value: float, beta: float):
        """EMA stored in state_dict[key]."""
        if not _is_finite(value):
            return None
        if key not in state_dict or state_dict[key] is None:
            state_dict[key] = float(value)
        else:
            state_dict[key] = float(beta) * float(state_dict[key]) + (1.0 - float(beta)) * float(value)
        return state_dict[key]

    def _sem_track_update(state, phase_tag: str, moco_v: float, jepa_v: float, total_v: float):
        """
        Maintains:
          - SEM refs (first observed values) to compute relative curves
          - EMA curves
        Stored in state["sem_track"].
        """
        eps = float(cfg.get("sem_rel_eps", 1e-8))
        beta = float(cfg.get("sem_ema_beta", 0.98))

        tr = state.get("sem_track", None)
        if tr is None:
            tr = {}
            state["sem_track"] = tr

        # reference values per phase (A/B)
        ref = tr.get(f"ref_{phase_tag}", None)
        if ref is None:
            ref = {"moco": None, "jepa": None, "total": None}
            tr[f"ref_{phase_tag}"] = ref

        # init refs once
        if ref["moco"] is None and _is_finite(moco_v) and moco_v > 0:
            ref["moco"] = float(moco_v)
        if ref["jepa"] is None and _is_finite(jepa_v) and jepa_v > 0:
            ref["jepa"] = float(jepa_v)
        if ref["total"] is None and _is_finite(total_v) and total_v > 0:
            ref["total"] = float(total_v)

        # EMA values per phase
        ema = tr.get(f"ema_{phase_tag}", None)
        if ema is None:
            ema = {"moco": None, "jepa": None, "total": None}
            tr[f"ema_{phase_tag}"] = ema

        ema_m = _ema_update(ema, "moco", moco_v, beta)
        ema_j = _ema_update(ema, "jepa", jepa_v, beta)
        ema_t = _ema_update(ema, "total", total_v, beta)

        # relative ratios (vs first ref)
        rel_m = None
        rel_j = None
        rel_t = None
        if ref["moco"] is not None and _is_finite(moco_v):
            rel_m = float(moco_v) / (float(ref["moco"]) + eps)
        if ref["jepa"] is not None and _is_finite(jepa_v):
            rel_j = float(jepa_v) / (float(ref["jepa"]) + eps)
        if ref["total"] is not None and _is_finite(total_v):
            rel_t = float(total_v) / (float(ref["total"]) + eps)

        return {
            "ref_moco": ref["moco"], "ref_jepa": ref["jepa"], "ref_total": ref["total"],
            "ema_moco": ema_m, "ema_jepa": ema_j, "ema_total": ema_t,
            "rel_moco": rel_m, "rel_jepa": rel_j, "rel_total": rel_t,
        }

    def _log_sem_curves(writer, phase: str, sem_loss_val, sem_stats: dict, step: int, q_vec=None, k_vec=None):
        """
        Logs:
          - SEM/{phase}/raw : moco, jepa, total, lambdas
          - SEM/{phase}/rel : rel_moco, rel_jepa, rel_total
          - SEM/{phase}/ema : ema_moco, ema_jepa, ema_total
          - SEM/{phase}/qk  : cos_sim, norm_q, norm_k (if q/k available)
        """
        if writer is None:
            return

        # Accept both key styles
        moco_v = _get_stat(sem_stats, "moco", "MoCo", default=0.0)
        jepa_v = _get_stat(sem_stats, "jepa", "JEPA_content", "JEPA", default=0.0)
        total_v = _get_stat(sem_stats, "loss_total", "total", default=_to_float(sem_loss_val, 0.0))
        lam_sem = _get_stat(sem_stats, "lam_sem", "λ_sem", default=float(cfg.get("lambda_sem", 0.0)))
        lam_jepa = _get_stat(sem_stats, "lam_jepa", "λ_sem_jepa", "λ_jepa", default=float(cfg.get("lambda_jepa_content", 0.0)))

        # Update state tracking (refs, EMA, rel)
        tr = _sem_track_update(state, phase, moco_v, jepa_v, total_v)

        # RAW
        writer.add_scalars(
            f"SEM/{phase}/raw",
            {
                "MoCo": float(moco_v),
                "JEPA_content": float(jepa_v),
                "loss_total": float(total_v),
                "λ_sem": float(lam_sem),
                "λ_jepa": float(lam_jepa),
            },
            step
        )

        # REL (relative to first seen)
        rel_dict = {}
        if tr["rel_moco"] is not None:
            rel_dict["MoCo_rel"] = float(tr["rel_moco"])
            rel_dict["MoCo_%"] = 100.0 * float(tr["rel_moco"])
        if tr["rel_jepa"] is not None:
            rel_dict["JEPA_rel"] = float(tr["rel_jepa"])
            rel_dict["JEPA_%"] = 100.0 * float(tr["rel_jepa"])
        if tr["rel_total"] is not None:
            rel_dict["total_rel"] = float(tr["rel_total"])
            rel_dict["total_%"] = 100.0 * float(tr["rel_total"])
        if rel_dict:
            writer.add_scalars(f"SEM/{phase}/rel", rel_dict, step)

        # EMA
        ema_dict = {}
        if tr["ema_moco"] is not None:
            ema_dict["MoCo_ema"] = float(tr["ema_moco"])
        if tr["ema_jepa"] is not None:
            ema_dict["JEPA_ema"] = float(tr["ema_jepa"])
        if tr["ema_total"] is not None:
            ema_dict["total_ema"] = float(tr["ema_total"])
        if ema_dict:
            writer.add_scalars(f"SEM/{phase}/ema", ema_dict, step)

        # q/k diagnostics (optional)
        if (q_vec is not None) and (k_vec is not None) and torch.is_tensor(q_vec) and torch.is_tensor(k_vec):
            try:
                q = q_vec.detach()
                k = k_vec.detach()
                # normalize
                qn = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
                kn = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
                cos = (qn * kn).sum(dim=-1)  # (B,)
                writer.add_scalars(
                    f"SEM/{phase}/qk",
                    {
                        "cos_mean": float(cos.mean().cpu().item()),
                        "cos_std": float(cos.std(unbiased=False).cpu().item()) if cos.numel() > 1 else 0.0,
                        "norm_q": float(q.norm(dim=-1).mean().cpu().item()),
                        "norm_k": float(k.norm(dim=-1).mean().cpu().item()),
                    },
                    step
                )
            except Exception:
                pass

    # ----------------------------
    # Setup / unpack
    # ----------------------------
    dev = cfg["device"]

    G_A = state["G_A"]
    D_A = state["D_A"]
    T_A = state["T_A"]
    T_B = state["T_B"]

    opt_GA = state["opt_GA"]
    opt_DA = state["opt_DA"]

    global_step = int(state.get("global_step", 0))

    l1_loss = cfg["l1_loss"]
    nce_loss = cfg["nce_loss"]
    nce_layers = cfg["nce_layers"]
    layer_w = cfg["nce_layer_w"]
    adv_type = cfg["adv_type"]
    adv_r1_gamma = cfg["adv_r1_gamma"]
    adv_r1_every = cfg["adv_r1_every"]
    adv_highpass = cfg["adv_highpass"]
    highpass_fn = cfg["highpass_fn"]

    tex_enable = cfg["tex_enable"]
    tex_apply_A = cfg["tex_apply_A"]
    tex_use_fft = cfg["tex_use_fft"]
    tex_use_swd = cfg["tex_use_swd"]
    tex_sigma = cfg["tex_sigma"]
    tex_gamma = cfg["tex_gamma"]
    λ_fft = cfg["lambda_fft"]
    λ_swd = cfg["lambda_swd"]
    swd_levels = cfg["swd_levels"]
    swd_patch = cfg["swd_patch"]
    swd_proj = cfg["swd_proj"]
    swd_max_patches = cfg["swd_max_patches"]

    fft_texture_loss_fn = cfg["fft_texture_loss"]
    swd_loss_images_fn = cfg["swd_loss_images"]

    λN = cfg["λN_current"]
    λR = cfg["λR_current"]

    λ_style_A = get_style_lambda(epoch=state.get("epoch", 0), cfg=cfg, base_lambda_key="λ_style_A")
    λ_spade = cfg["λ_spade"]
    spade_marg = cfg["spade_margin"]

    λ_content_nce2 = float(cfg.get("lambda_content_nce_two_styles", 0.0))

    mixswap_enable = cfg["mixswap_enable"]
    mixswap_token_p = cfg["mixswap_token_p"]
    mixswap_fft_p = cfg["mixswap_fft_p"]
    mixswap_alpha_lo = cfg["mixswap_alpha_lo"]
    mixswap_alpha_hi = cfg["mixswap_alpha_hi"]
    fft_amp_mix_fn = cfg["fft_amp_mix"]

    # --- JEPA config ---
    jepa_on_style = bool(cfg.get("jepa_on_style", False))
    jepa_on_content = bool(cfg.get("jepa_on_content", False))
    jepa_every = int(cfg["jepa_every"])
    jepa_scale_base = cfg["jepa_scale_w"]
    jepa_mask_ratio = float(cfg["jepa_mask_ratio"])
    jepa_bias_high = float(cfg["jepa_bias_high"])
    jepa_use_teacher = bool(cfg["jepa_use_teacher"])
    λ_jepa_style = float(cfg.get("lambda_jepa_style", cfg.get("lambda_jepa", 0.0)))
    λ_jepa_content = float(cfg.get("lambda_jepa_content", 0.0))
    λ_jepa_kd = float(cfg.get("lambda_jepa_kd", 0.0))

    tok_jepa_A_style = state.get("tok_jepa_A_style", None)
    tok_jepa_A_content = state.get("tok_jepa_A_content", None)

    use_B_feats = (state.get("epoch", 0) >= cfg["feat_switch_epoch"])

    def _make_jepa_w_for_seq(S, device):
        base = jepa_scale_base
        if isinstance(base, torch.Tensor):
            w_vec = base.to(device).flatten()
            if w_vec.numel() == S:
                return w_vec
            return w_vec.mean().repeat(S)
        val = float(base)
        return torch.full((S,), val, device=device)

    def stack_content_tokens_multiscale_for_jepa(feat_dict, ref_key="bot"):
        if not feat_dict:
            return None
        if ref_key in feat_dict:
            ref = feat_dict[ref_key]
        else:
            ref = max(feat_dict.values(), key=lambda t: t.shape[2] * t.shape[3])
        _, _, H_ref, W_ref = ref.shape
        maps = []
        for _, f in feat_dict.items():
            if not (torch.is_tensor(f) and f.dim() == 4):
                continue
            if (f.shape[2] != H_ref) or (f.shape[3] != W_ref):
                f = F.interpolate(f, size=(H_ref, W_ref), mode="bilinear", align_corners=False)
            maps.append(f)
        if not maps:
            return None
        f_cat = torch.cat(maps, dim=1)
        Bc, Ctot, Hr, Wr = f_cat.shape
        Ts = f_cat.view(Bc, Ctot, Hr * Wr).transpose(1, 2)
        return Ts

    # ----------------------------
    # STYLE TARGET: Mix-Swap (optional)
    # ----------------------------
    if mixswap_enable and (random.random() < mixswap_token_p):
        perm = torch.randperm(x.size(0), device=x.device)
        y_alt = y[perm]

        if mixswap_fft_p > 0.0 and (random.random() < mixswap_fft_p):
            alpha_fft = random.uniform(mixswap_alpha_lo, mixswap_alpha_hi)
            y = fft_amp_mix_fn(y, y_alt, alpha_fft).clamp(-1, 1)

        with torch.no_grad():
            _, toks_y, tokG_y = T_A.style_enc(y)
            _, toks_z, tokG_z = T_A.style_enc(y_alt)

        alpha = torch.rand(x.size(0), 1, device=x.device) * (mixswap_alpha_hi - mixswap_alpha_lo) + mixswap_alpha_lo

        def _blend(a, b):
            return a * alpha + b * (1 - alpha)

        toks_target = tuple(_blend(ty, tz) for ty, tz in zip(toks_y, toks_z))
        tokG_target = _blend(tokG_y, tokG_z)

        style_cond_A = cfg["build_style_cond_from_tokens"](toks_target, tokG_target, for_G="A")
        style_target_for_loss = ("tokens", toks_target, tokG_target)
    else:
        style_cond_A = cfg["build_style_cond"](G_A, y)
        style_target_for_loss = ("image", y)

    style_gain_A = float(cfg.get("style_gain_A", 1.0))
    style_cond_A = apply_style_gain(style_cond_A, style_gain_A)

    # ----------------------------
    # D_A
    # ----------------------------
    adv_enable_A = bool(cfg["adv_enable_A"])
    if adv_enable_A:
        with torch.no_grad():
            far_det = G_A(x, style=style_cond_A)

        opt_DA.zero_grad(set_to_none=True)
        use_hinge = (adv_type == "hinge")
        do_r1 = ((global_step % adv_r1_every) == 0)

        loss_DA, dstatsA = d_loss_and_stats(
            D_A, y, far_det,
            use_hinge=use_hinge,
            r1_gamma=adv_r1_gamma,
            do_r1=do_r1,
            adv_highpass=adv_highpass,
            highpass_fn=highpass_fn
        )
        loss_DA.backward()
        opt_DA.step()
    else:
        dstatsA = {"real_mean": 0, "fake_mean": 0, "r1": 0.0}
        loss_DA = torch.tensor(0.0, device=dev)

    # ----------------------------
    # G_A
    # ----------------------------
    opt_GA.zero_grad(set_to_none=True)
    far = G_A(x, style=style_cond_A)

    advA = torch.tensor(0.0, device=dev)
    if adv_enable_A:
        advA = g_adv_loss(
            D_A, far,
            use_hinge=(adv_type == "hinge"),
            adv_highpass=adv_highpass,
            highpass_fn=highpass_fn
        )

    if use_B_feats:
        feats_x = nce_feats_dict(T_B, x, normalize=True, no_grad=False)
        feats_f = nce_feats_dict(T_B, far, normalize=True, no_grad=False)
    else:
        feats_x = nce_feats_dict(G_A, x, normalize=True, no_grad=False)
        feats_x = {k: v.detach() for k, v in feats_x.items()}
        feats_f = nce_feats_dict(G_A, far, normalize=True, no_grad=False)

    nce_total = 0.0
    for l, w in zip(nce_layers, layer_w):
        nce_total = nce_total + (w * nce_loss(feats_x[l], feats_f[l]))

    content_nce_2 = torch.tensor(0.0, device=dev)
    if λ_content_nce2 > 0.0 and x.size(0) > 1:
        perm2 = torch.randperm(x.size(0), device=x.device)
        y2 = y[perm2]
        style_cond_A2 = cfg["build_style_cond"](G_A, y2)
        style_cond_A2 = apply_style_gain(style_cond_A2, style_gain_A)
        far2 = G_A(x, style=style_cond_A2)

        if use_B_feats:
            feats_f2 = nce_feats_dict(T_B, far2, normalize=True, no_grad=False)
        else:
            feats_f2 = nce_feats_dict(G_A, far2, normalize=True, no_grad=False)

        content_nce_2 = content_nce_two_styles(
            nce_loss=nce_loss,
            feats_x=feats_x,
            feats_f1=feats_f,
            feats_f2=feats_f2,
            nce_layers=nce_layers,
            layer_w=layer_w,
        )

    if style_target_for_loss[0] == "image":
        styA = style_loss_tokens(far, style_target_for_loss[1], teacher_G=T_A)
    else:
        _, toks_t, tokG_t = style_target_for_loss
        styA = style_loss_tokens_to_target(far, toks_t, tokG_t, teacher_G=T_A)

    tex_A_total = torch.tensor(0.0, device=dev)
    if tex_enable and tex_apply_A:
        if tex_use_fft and λ_fft > 0:
            tex_A_total = tex_A_total + (λ_fft * fft_texture_loss_fn(far, y, log_mag=True, per_channel=True))
        if tex_use_swd and λ_swd > 0:
            tex_A_total = tex_A_total + (λ_swd * swd_loss_images_fn(
                far, y, levels=swd_levels, patch=swd_patch, proj=swd_proj, max_patches=swd_max_patches
            ))

    regA = l1_loss(far, x)
    gateA, ratioA = spade_gate_reg(G_A, margin=spade_marg)

    # ----------------------------
    # JEPA (style + content)
    # ----------------------------
    jepa_styleA = torch.tensor(0.0, device=dev)
    jepa_styleA_kd = torch.tensor(0.0, device=dev)
    jepa_contentA = torch.tensor(0.0, device=dev)
    jepa_contentA_kd = torch.tensor(0.0, device=dev)

    jepa_attn_styleA_H = None
    jepa_attn_contentA_H = None

    do_jepa = (jepa_on_style or jepa_on_content) and ((global_step % jepa_every) == 0)
    if do_jepa:
        from models.jepa import TokenJEPA

        if jepa_on_style:
            style_sigma = float(tex_sigma)
            style_gamma = float(tex_gamma)
            y_v1, y_v2 = two_style_views(y, sigma=style_sigma, gamma=style_gamma, flip_p=0.5)

            with torch.no_grad():
                if jepa_use_teacher:
                    _, toks_tA, tokG_tA = T_A.style_enc(y_v2)
                else:
                    _, toks_tA, tokG_tA = G_A.style_enc(y_v2)
                TtA = stack_tokens_for_jepa(toks_tA, tokG_tA)

            _, toks_sA, tokG_sA = G_A.style_enc(y_v1)
            TsA = stack_tokens_for_jepa(toks_sA, tokG_sA)

            if (TtA is not None) and (TsA is not None):
                Bsz, Ssz, Dsz = TsA.shape
                if tok_jepa_A_style is None:
                    tok_jepa_A_style = TokenJEPA(
                        Ssz, Dsz,
                        hidden_mult=cfg["jepa_hidden_mult"],
                        heads=cfg["jepa_heads"],
                        use_norm=cfg["jepa_norm"],
                        var_lambda=cfg["lambda_jepa_var"],
                        cov_lambda=cfg["lambda_jepa_cov"]
                    ).to(dev)
                    state["tok_jepa_A_style"] = tok_jepa_A_style

                maskA = bias_mask(Bsz, Ssz, jepa_mask_ratio, jepa_bias_high, device=dev)
                w_styleA = _make_jepa_w_for_seq(Ssz, dev)
                LjA, infoA = tok_jepa_A_style(TsA, TtA.detach(), maskA, w=w_styleA)
                jepa_styleA = LjA

                if λ_jepa_kd > 0:
                    jepa_styleA_kd = kd_logits_supheads_if_available(G_A, infoA.get("pred_raw", TsA), TtA.detach())

                if isinstance(infoA, dict):
                    H = infoA.get("attn_entropy", None)
                    if H is None and "attn" in infoA:
                        att = infoA["attn"]
                        if torch.is_tensor(att):
                            p = att.clamp_min(1e-8)
                            H = -(p * p.log()).sum(-1).mean()
                    if H is not None:
                        jepa_attn_styleA_H = _to_float(H, None)

        if jepa_on_content:
            x_v1c, x_v2c = two_style_views(x, sigma=0.0, gamma=1.0, flip_p=0.5)

            with torch.no_grad():
                if jepa_use_teacher:
                    feats_t = nce_feats_dict(T_A, x_v2c, normalize=False, no_grad=True)
                else:
                    feats_t = nce_feats_dict(G_A, x_v2c, normalize=False, no_grad=True)

            feats_s = nce_feats_dict(G_A, x_v1c, normalize=False, no_grad=False)

            TtA_c = stack_content_tokens_multiscale_for_jepa(feats_t, ref_key="bot")
            TsA_c = stack_content_tokens_multiscale_for_jepa(feats_s, ref_key="bot")

            if (TtA_c is not None) and (TsA_c is not None):
                Bsz, Ssz, Dsz = TsA_c.shape
                if tok_jepa_A_content is None:
                    tok_jepa_A_content = TokenJEPA(
                        Ssz, Dsz,
                        hidden_mult=cfg["jepa_hidden_mult"],
                        heads=cfg["jepa_heads"],
                        use_norm=cfg["jepa_norm"],
                        var_lambda=cfg["lambda_jepa_var"],
                        cov_lambda=cfg["lambda_jepa_cov"]
                    ).to(dev)
                    state["tok_jepa_A_content"] = tok_jepa_A_content

                maskA_c = bias_mask(Bsz, Ssz, jepa_mask_ratio, jepa_bias_high, device=dev)
                w_contA = _make_jepa_w_for_seq(Ssz, dev)
                LjA_c, infoA_c = tok_jepa_A_content(TsA_c, TtA_c.detach(), maskA_c, w=w_contA)
                jepa_contentA = LjA_c

                if λ_jepa_kd > 0:
                    jepa_contentA_kd = kd_logits_supheads_if_available(G_A, infoA_c.get("pred_raw", TsA_c), TtA_c.detach())

                if isinstance(infoA_c, dict):
                    Hc = infoA_c.get("attn_entropy", None)
                    if Hc is None and "attn" in infoA_c:
                        attc = infoA_c["attn"]
                        if torch.is_tensor(attc):
                            p = attc.clamp_min(1e-8)
                            Hc = -(p * p.log()).sum(-1).mean()
                    if Hc is not None:
                        jepa_attn_contentA_H = _to_float(Hc, None)

    jepa_totalA = (
        λ_jepa_style * jepa_styleA
        + λ_jepa_content * jepa_contentA
        + λ_jepa_kd * (jepa_styleA_kd + jepa_contentA_kd)
    )

    totalA = (
        advA
        + λN * nce_total
        + λ_content_nce2 * content_nce_2
        + λ_style_A * styA
        + λR * regA
        + λ_spade * gateA
        + tex_A_total
        + jepa_totalA
    )

    if torch.is_tensor(totalA):
        totalA.backward()
        opt_GA.step()

    # ----------------------------
    # EMA teachers
    # ----------------------------
    if (global_step % int(cfg["ema_every"])) == 0:
        nce_m = float(cfg["nce_m"])
        for src, tgt in [(state["G_B"], T_B), (G_A, T_A)]:
            for po, pt in zip(src.parameters(), tgt.parameters()):
                pt.data.mul_(nce_m).add_(po.data, alpha=1.0 - nce_m)

    state["replay"].extend(far.detach().cpu())
    state["style_bank"].extend(y.detach().cpu())

    # ----------------------------
    # meters epoch (A)
    # ----------------------------
    if adv_enable_A:
        epoch_meters["A"]["D_loss"].add(float(loss_DA.item()))
    epoch_meters["A"]["G_adv"].add(float(advA.item()))
    epoch_meters["A"]["NCE"].add(float(nce_total.item()))
    epoch_meters["A"]["NCE_content2"].add(float(content_nce_2.item()))
    epoch_meters["A"]["styleTok"].add(float(styA.item()))
    epoch_meters["A"]["L1"].add(float(regA.item()))
    epoch_meters["A"]["style_gain_A"].add(float(style_gain_A))
    epoch_meters["A"]["λ_style_A"].add(float(λ_style_A))
    epoch_meters["A"]["JEPA_style"].add(float(jepa_styleA.item()))
    epoch_meters["A"]["JEPA_content"].add(float(jepa_contentA.item()))
    if torch.is_tensor(totalA):
        epoch_meters["A"]["total"].add(float(totalA.item()))

    # =============================================================================
    # ✅ SEM branch (MoCo + JEPA-content) — ALWAYS LOG (even if SEM fails)
    # =============================================================================
    sem_enabled = bool(cfg.get("sem_enable", False)) and ("SEM" in state)

    moco_v = 0.0
    jepa_v = 0.0
    sem_total_v = 0.0
    lam_sem = float(cfg.get("lambda_sem", 0.0))
    lam_jepa = float(cfg.get("lambda_jepa_content", 0.0))
    q_vec = None
    k_vec = None
    sem_exc = 0
    sem_stats = {"moco": 0.0, "jepa": 0.0, "loss_total": 0.0, "lam_sem": lam_sem, "lam_jepa": lam_jepa}

    if sem_enabled:
        try:
            sem_out = semantic_content_step_moco_jepa(x, far, y, state, cfg)
            if sem_out is not None:
                sem_loss_val, sem_stats_out, q_vec, k_vec = sem_out
                # read stats robustly
                moco_v = _get_stat(sem_stats_out, "moco", "MoCo", default=0.0)
                jepa_v = _get_stat(sem_stats_out, "jepa", "JEPA_content", default=0.0)
                sem_total_v = _get_stat(sem_stats_out, "loss_total", "total", default=_to_float(sem_loss_val, 0.0))
                lam_sem = _get_stat(sem_stats_out, "lam_sem", "λ_sem", default=lam_sem)
                lam_jepa = _get_stat(sem_stats_out, "lam_jepa", "λ_sem_jepa", "λ_jepa", default=lam_jepa)
        except Exception:
            sem_exc = 1
            state["sem_exceptions"] = int(state.get("sem_exceptions", 0)) + 1

    # epoch meters SEM (always)
    if "SEM" in epoch_meters:
        try:
            epoch_meters["SEM"]["MoCo"].add(float(moco_v))
            epoch_meters["SEM"]["JEPA_content"].add(float(jepa_v))
            if "λ_sem" in epoch_meters["SEM"]:
                epoch_meters["SEM"]["λ_sem"].add(float(lam_sem))
            if "λ_sem_jepa" in epoch_meters["SEM"]:
                epoch_meters["SEM"]["λ_sem_jepa"].add(float(lam_jepa))
            if "total" in epoch_meters["SEM"]:
                epoch_meters["SEM"]["total"].add(float(sem_total_v))
        except Exception:
            pass

    # ----------------------------
    # TensorBoard logs (A + JEPA + images)
    # ----------------------------
    tb_freq = int(cfg["tb_freq"])
    tb_freq_sem = int(cfg.get("tb_freq_sem", tb_freq))
    sem_print_every = int(cfg.get("sem_print_every", tb_freq_sem))

    if writer and (global_step % tb_freq == 0):
        if adv_enable_A:
            writer.add_scalars(
                "A/D",
                {
                    "loss": float(loss_DA.item()),
                    "real_mean": float(dstatsA.get("real_mean", 0.0)),
                    "fake_mean": float(dstatsA.get("fake_mean", 0.0)),
                    "r1": float(dstatsA.get("r1", 0.0)),
                },
                global_step
            )

        writer.add_scalars(
            "A/G",
            {
                "adv": float(advA.item()),
                "NCE": float(nce_total.item()),
                "NCE_content2": float(content_nce_2.item()),
                "styleTok": float(styA.item()),
                "style_gain": float(style_gain_A),
                "lambda_style_A": float(λ_style_A),
                "L1": float(regA.item()),
                "spade_gate": float(gateA.item()),
                "tex": float(_to_float(tex_A_total, 0.0)),
                "total": float(totalA.item()) if torch.is_tensor(totalA) else float(totalA),
            },
            global_step
        )

        writer.add_scalars(
            "A/JEPA",
            {
                "style": float(jepa_styleA.item()),
                "content": float(jepa_contentA.item()),
                "style_kd": float(jepa_styleA_kd.item()),
                "content_kd": float(jepa_contentA_kd.item()),
                "total": float(jepa_totalA.item()),
            },
            global_step
        )

        jepa_attn_scalars = {}
        if jepa_attn_styleA_H is not None:
            jepa_attn_scalars["style_attn_H"] = float(jepa_attn_styleA_H)
        if jepa_attn_contentA_H is not None:
            jepa_attn_scalars["content_attn_H"] = float(jepa_attn_contentA_H)
        if jepa_attn_scalars:
            writer.add_scalars("A/JEPA_attn", jepa_attn_scalars, global_step)

        xa = cfg["_denorm"](x)
        ya = cfg["_denorm"](y)
        fa = cfg["_denorm"](far)
        writer.add_images("A/imgs/x_content", xa[:4], global_step)
        writer.add_images("A/imgs/y_style", ya[:4], global_step)
        writer.add_images("A/imgs/far(x→y)", fa[:4], global_step)
        try:
            trip_A = cfg["_triplet_grid"](xa, ya, fa, max_k=4, nrow=3)
            writer.add_image("A/triplet_[x | y | far]", trip_A, global_step)
        except Exception:
            pass

        try:
            writer.flush()
        except Exception:
            pass

    # ----------------------------
    # ✅ SEM TensorBoard + terminal (A)  + evolution curves (ALWAYS)
    # ----------------------------
    if writer and sem_enabled and (global_step % tb_freq_sem == 0):
        # raw curves always visible
        writer.add_scalars(
            "SEM/A/raw_direct",
            {"MoCo": float(moco_v), "JEPA_content": float(jepa_v), "loss_total": float(sem_total_v),
             "λ_sem": float(lam_sem), "λ_jepa": float(lam_jepa)},
            global_step
        )
        # evolution curves (raw/rel/ema/qk)
        sem_stats = {"moco": moco_v, "jepa": jepa_v, "loss_total": sem_total_v,
                     "lam_sem": lam_sem, "lam_jepa": lam_jepa,
                      "lr_sem": float(state.get("opt_SEM", None).param_groups[0]["lr"]) if state.get("opt_SEM", None) else 0.0}

        _log_sem_curves(writer, "A", sem_total_v, sem_stats, global_step, q_vec=q_vec, k_vec=k_vec)

        writer.add_scalar("SEM/exceptions_total", float(state.get("sem_exceptions", 0)), global_step)
        writer.add_scalar("SEM/A/exception_step", float(sem_exc), global_step)

        try:
            writer.flush()
        except Exception:
            pass

    if sem_enabled and ((global_step % sem_print_every) == 0):
        _tqdm_write(
            f"[SEM/A] step={global_step}  "
            f"MoCo={moco_v:.4f}  JEPAc={jepa_v:.4f}  total={sem_total_v:.4f}  exc={sem_exc}"
        )

    # ----------------------------
    # update global step
    # ----------------------------
    state["global_step"] = global_step + 1


# =========================================================================================
#           Phase B  (far -> recon to x)   + SEM (optional) TensorBoard logging
# =========================================================================================

def train_step_phase_B(
        x,
        state,
        cfg,
        epoch_meters,
        writer,
):
    import math
    import random
    import numpy as np
    import torch
    import torch.nn.functional as F

    # ----------------------------
    # Helpers (same logic as phase A)
    # ----------------------------
    def _to_float(v, default=None):
        try:
            if v is None:
                return default
            if torch.is_tensor(v):
                return float(v.detach().float().mean().cpu().item())
            return float(v)
        except Exception:
            return default

    def _is_finite(v):
        try:
            return (v is not None) and (not math.isnan(float(v))) and (not math.isinf(float(v)))
        except Exception:
            return False

    def _tqdm_write(msg: str):
        try:
            from tqdm import tqdm
            tqdm.write(msg)
        except Exception:
            print(msg)

    def _get_stat(stats: dict, *keys, default=0.0):
        if not isinstance(stats, dict):
            return default
        for k in keys:
            if k in stats:
                fv = _to_float(stats.get(k), None)
                if fv is not None:
                    return fv
        return default

    def _ema_update(state_dict, key: str, value: float, beta: float):
        if not _is_finite(value):
            return None
        if key not in state_dict or state_dict[key] is None:
            state_dict[key] = float(value)
        else:
            state_dict[key] = float(beta) * float(state_dict[key]) + (1.0 - float(beta)) * float(value)
        return state_dict[key]

    def _sem_track_update(state, phase_tag: str, moco_v: float, jepa_v: float, total_v: float):
        eps = float(cfg.get("sem_rel_eps", 1e-8))
        beta = float(cfg.get("sem_ema_beta", 0.98))

        tr = state.get("sem_track", None)
        if tr is None:
            tr = {}
            state["sem_track"] = tr

        ref = tr.get(f"ref_{phase_tag}", None)
        if ref is None:
            ref = {"moco": None, "jepa": None, "total": None}
            tr[f"ref_{phase_tag}"] = ref

        if ref["moco"] is None and _is_finite(moco_v) and moco_v > 0:
            ref["moco"] = float(moco_v)
        if ref["jepa"] is None and _is_finite(jepa_v) and jepa_v > 0:
            ref["jepa"] = float(jepa_v)
        if ref["total"] is None and _is_finite(total_v) and total_v > 0:
            ref["total"] = float(total_v)

        ema = tr.get(f"ema_{phase_tag}", None)
        if ema is None:
            ema = {"moco": None, "jepa": None, "total": None}
            tr[f"ema_{phase_tag}"] = ema

        ema_m = _ema_update(ema, "moco", moco_v, beta)
        ema_j = _ema_update(ema, "jepa", jepa_v, beta)
        ema_t = _ema_update(ema, "total", total_v, beta)

        rel_m = None
        rel_j = None
        rel_t = None
        if ref["moco"] is not None and _is_finite(moco_v):
            rel_m = float(moco_v) / (float(ref["moco"]) + eps)
        if ref["jepa"] is not None and _is_finite(jepa_v):
            rel_j = float(jepa_v) / (float(ref["jepa"]) + eps)
        if ref["total"] is not None and _is_finite(total_v):
            rel_t = float(total_v) / (float(ref["total"]) + eps)

        return {
            "ema_moco": ema_m, "ema_jepa": ema_j, "ema_total": ema_t,
            "rel_moco": rel_m, "rel_jepa": rel_j, "rel_total": rel_t,
        }

    def _log_sem_curves(writer, phase: str, sem_loss_val, sem_stats: dict, step: int, q_vec=None, k_vec=None):
        if writer is None:
            return

        moco_v = _get_stat(sem_stats, "moco", "MoCo", default=0.0)
        jepa_v = _get_stat(sem_stats, "jepa", "JEPA_content", "JEPA", default=0.0)
        total_v = _get_stat(sem_stats, "loss_total", "total", default=_to_float(sem_loss_val, 0.0))
        lam_sem = _get_stat(sem_stats, "lam_sem", "λ_sem", default=float(cfg.get("lambda_sem", 0.0)))
        lam_jepa = _get_stat(sem_stats, "lam_jepa", "λ_sem_jepa", "λ_jepa", default=float(cfg.get("lambda_jepa_content", 0.0)))

        tr = _sem_track_update(state, phase, moco_v, jepa_v, total_v)

        writer.add_scalars(
            f"SEM/{phase}/raw",
            {
                "MoCo": float(moco_v),
                "JEPA_content": float(jepa_v),
                "loss_total": float(total_v),
                "λ_sem": float(lam_sem),
                "λ_jepa": float(lam_jepa),
            },
            step
        )

        rel_dict = {}
        if tr["rel_moco"] is not None:
            rel_dict["MoCo_rel"] = float(tr["rel_moco"])
            rel_dict["MoCo_%"] = 100.0 * float(tr["rel_moco"])
        if tr["rel_jepa"] is not None:
            rel_dict["JEPA_rel"] = float(tr["rel_jepa"])
            rel_dict["JEPA_%"] = 100.0 * float(tr["rel_jepa"])
        if tr["rel_total"] is not None:
            rel_dict["total_rel"] = float(tr["rel_total"])
            rel_dict["total_%"] = 100.0 * float(tr["rel_total"])
        if rel_dict:
            writer.add_scalars(f"SEM/{phase}/rel", rel_dict, step)

        ema_dict = {}
        if tr["ema_moco"] is not None:
            ema_dict["MoCo_ema"] = float(tr["ema_moco"])
        if tr["ema_jepa"] is not None:
            ema_dict["JEPA_ema"] = float(tr["ema_jepa"])
        if tr["ema_total"] is not None:
            ema_dict["total_ema"] = float(tr["ema_total"])
        if ema_dict:
            writer.add_scalars(f"SEM/{phase}/ema", ema_dict, step)

        if (q_vec is not None) and (k_vec is not None) and torch.is_tensor(q_vec) and torch.is_tensor(k_vec):
            try:
                q = q_vec.detach()
                k = k_vec.detach()
                qn = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
                kn = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
                cos = (qn * kn).sum(dim=-1)
                writer.add_scalars(
                    f"SEM/{phase}/qk",
                    {
                        "cos_mean": float(cos.mean().cpu().item()),
                        "cos_std": float(cos.std(unbiased=False).cpu().item()) if cos.numel() > 1 else 0.0,
                        "norm_q": float(q.norm(dim=-1).mean().cpu().item()),
                        "norm_k": float(k.norm(dim=-1).mean().cpu().item()),
                    },
                    step
                )
            except Exception:
                pass

    # ----------------------------
    # Setup / unpack
    # ----------------------------
    dev = cfg["device"]

    G_A = state["G_A"]
    G_B = state["G_B"]
    D_B = state["D_B"]
    T_B = state["T_B"]

    opt_GB = state["opt_GB"]
    opt_DB = state["opt_DB"]

    global_step = int(state.get("global_step", 0))

    replay = state["replay"]
    style_bank = state["style_bank"]

    l1_loss = cfg["l1_loss"]
    nce_loss = cfg["nce_loss"]
    nce_layers = cfg["nce_layers"]
    layer_w = cfg["nce_layer_w"]

    adv_type = cfg["adv_type"]
    adv_r1_gamma = cfg["adv_r1_gamma"]
    adv_r1_every = cfg["adv_r1_every"]
    adv_highpass = cfg["adv_highpass"]
    highpass_fn = cfg["highpass_fn"]

    tex_enable = cfg["tex_enable"]
    tex_use_fft = cfg["tex_use_fft"]
    tex_use_swd = cfg["tex_use_swd"]
    tex_sigma = cfg["tex_sigma"]
    tex_gamma = cfg["tex_gamma"]
    λ_fft = cfg["lambda_fft"]
    λ_swd = cfg["lambda_swd"]
    swd_levels = cfg["swd_levels"]
    swd_patch = cfg["swd_patch"]
    swd_proj = cfg["swd_proj"]
    swd_max_patches = cfg["swd_max_patches"]

    fft_texture_loss_fn = cfg["fft_texture_loss"]
    swd_loss_images_fn = cfg["swd_loss_images"]

    λ_nce_B = cfg["lambda_nce_b"]
    λ_idt_B = cfg["lambda_idt_b"]

    λ_style_B_dyn = float(state["λ_style_B_dyn"])
    λ_style_B_min = float(cfg["λ_style_B_min"])
    λ_style_B_max = float(cfg["λ_style_B_max"])
    style_B_warmup_ep = int(cfg["style_B_warmup_ep"])
    style_target = float(cfg["style_balance_target"])
    style_alpha = float(cfg["style_balance_alpha"])

    # effective style weight for B (warmup + dyn)
    if state.get("epoch", 0) < style_B_warmup_ep:
        λ_style_B_eff = 0.0
    else:
        λ_style_B_eff = λ_style_B_dyn

    if len(style_bank) < x.size(0):
        if writer and (global_step % int(cfg.get("tb_freq_sem", cfg["tb_freq"])) == 0):
            writer.add_scalar("B/skip_not_enough_style_bank", 1.0, global_step)
        state["global_step"] = global_step + 1
        return

    # ----------------------------
    # Build far_mix (from G_A) + replay
    # ----------------------------
    with torch.no_grad():
        y_s = torch.stack(random.sample(style_bank, x.size(0))).to(dev)
        far_x = G_A(x, style=cfg["build_style_cond"](G_A, y_s))

    nb_r = int(getattr(cfg["opt"], "replay_ratio", 0.0) * x.size(0))
    if nb_r and len(replay) >= nb_r:
        far_rep = torch.stack(random.sample(replay, nb_r)).to(dev)
    else:
        far_rep = torch.empty(0, *x.shape[1:], device=dev)

    if nb_r:
        far_mix = torch.cat([far_x, far_rep], 0)
        x_mix = torch.cat([x, x[:nb_r]], 0)
    else:
        far_mix = far_x
        x_mix = x

    spectral_noise_fn = cfg["spectral_noise"]
    if tex_enable and tex_sigma > 0.0:
        far_mix_noisy = spectral_noise_fn(far_mix, tex_sigma, tex_gamma)
    else:
        far_mix_noisy = far_mix

    style_cond_B = cfg["build_style_cond"](G_B, x_mix)
    style_gain_B = float(cfg.get("style_gain_B", λ_style_B_eff))
    style_cond_B = apply_style_gain(style_cond_B, style_gain_B)

    # ----------------------------
    # D_B
    # ----------------------------
    adv_enable_B = bool(cfg["adv_enable_B"])
    if adv_enable_B:
        with torch.no_grad():
            recon_det = G_B(far_mix_noisy, style=style_cond_B)

        opt_DB.zero_grad(set_to_none=True)
        use_hinge = (adv_type == "hinge")
        do_r1 = ((global_step % adv_r1_every) == 0)

        loss_DB, dstatsB = d_loss_and_stats(
            D_B, x_mix, recon_det,
            use_hinge=use_hinge,
            r1_gamma=cfg["adv_r1_gamma"],
            do_r1=do_r1,
            adv_highpass=adv_highpass,
            highpass_fn=highpass_fn
        )
        loss_DB.backward()
        opt_DB.step()
    else:
        dstatsB = {"real_mean": 0, "fake_mean": 0, "r1": 0.0}
        loss_DB = torch.tensor(0.0, device=dev)

    # ----------------------------
    # G_B
    # ----------------------------
    opt_GB.zero_grad(set_to_none=True)
    recon = G_B(far_mix_noisy, style=style_cond_B)

    advB = torch.tensor(0.0, device=dev)
    if adv_enable_B:
        advB = g_adv_loss(
            D_B, recon,
            use_hinge=(adv_type == "hinge"),
            adv_highpass=adv_highpass,
            highpass_fn=highpass_fn
        )

    with torch.no_grad():
        feats_xm = nce_feats_dict(T_B, x_mix, normalize=True, no_grad=True)
    feats_rec = nce_feats_dict(G_B, recon, normalize=True, no_grad=False)

    nceB = 0.0
    for l, w in zip(nce_layers, layer_w):
        nceB = nceB + (w * nce_loss(feats_xm[l], feats_rec[l]))

    idtB = l1_loss(recon, x_mix)
    styB = style_loss_tokens(recon, x_mix, teacher_G=T_B)

    tex_B_total = torch.tensor(0.0, device=dev)
    if tex_enable:
        if cfg["tex_use_fft"] and λ_fft > 0:
            tex_B_total = tex_B_total + (λ_fft * fft_texture_loss_fn(recon, x_mix, log_mag=True, per_channel=True))
        if cfg["tex_use_swd"] and λ_swd > 0:
            tex_B_total = tex_B_total + (λ_swd * swd_loss_images_fn(
                recon, x_mix, levels=swd_levels, patch=swd_patch, proj=swd_proj, max_patches=swd_max_patches
            ))

    gateB, ratioB = spade_gate_reg(G_B, margin=cfg["spade_margin"])

    totalB = (
        advB
        + λ_nce_B * nceB
        + λ_idt_B * idtB
        + λ_style_B_eff * styB
        + cfg["λ_spade"] * gateB
        + tex_B_total
    )

    if torch.is_tensor(totalB):
        totalB.backward()
        opt_GB.step()

    if (global_step % int(cfg["ema_every"])) == 0:
        nce_m = float(cfg["nce_m"])
        for po, pt in zip(G_B.parameters(), T_B.parameters()):
            pt.data.mul_(nce_m).add_(po.data, alpha=1.0 - nce_m)

    # =============================================================================
    # ✅ SEM branch (B) — ALWAYS LOG (even if SEM fails)
    # =============================================================================
    sem_on_B = bool(cfg.get("sem_on_B", True))
    sem_enabled = sem_on_B and bool(cfg.get("sem_enable", False)) and ("SEM" in state)

    moco_v = 0.0
    jepa_v = 0.0
    sem_total_v = 0.0
    lam_sem = float(cfg.get("lambda_sem", 0.0))
    lam_jepa = float(cfg.get("lambda_jepa_content", 0.0))
    q_vec = None
    k_vec = None
    sem_exc_B = 0

    if sem_enabled:
        try:
            cfg_semB = dict(cfg)
            cfg_semB["sem_two_styles"] = False
            sem_outB = semantic_content_step_moco_jepa(x_mix, recon, y=None, state=state, cfg=cfg_semB)
            if sem_outB is not None:
                sem_loss_val_B, sem_stats_B, q_vec, k_vec = sem_outB
                moco_v = _get_stat(sem_stats_B, "moco", "MoCo", default=0.0)
                jepa_v = _get_stat(sem_stats_B, "jepa", "JEPA_content", default=0.0)
                sem_total_v = _get_stat(sem_stats_B, "loss_total", "total", default=_to_float(sem_loss_val_B, 0.0))
        except Exception:
            sem_exc_B = 1
            state["sem_exceptions"] = int(state.get("sem_exceptions", 0)) + 1

    # epoch meters SEM B (always)
    if "SEM" in epoch_meters:
        try:
            if "MoCo_B" in epoch_meters["SEM"]:
                epoch_meters["SEM"]["MoCo_B"].add(float(moco_v))
            if "JEPA_content_B" in epoch_meters["SEM"]:
                epoch_meters["SEM"]["JEPA_content_B"].add(float(jepa_v))
            if "total_B" in epoch_meters["SEM"]:
                epoch_meters["SEM"]["total_B"].add(float(sem_total_v))
        except Exception:
            pass

    # ----------------------------
    # TensorBoard logs (B) + SEM curves
    # ----------------------------
    tb_freq = int(cfg["tb_freq"])
    tb_freq_sem = int(cfg.get("tb_freq_sem", tb_freq))
    sem_print_every = int(cfg.get("sem_print_every", tb_freq_sem))

    if writer and (global_step % tb_freq == 0):
        if state.get("epoch", 0) >= style_B_warmup_ep:
            with torch.no_grad():
                d_rx = style_dist_tokens(T_B, recon, x_mix)
                err = float(d_rx.item()) - style_target
                scale = 1.0 + style_alpha * (err / max(style_target, 1e-6))
                λ_style_B_dyn = float(np.clip(λ_style_B_dyn * scale, λ_style_B_min, λ_style_B_max))
                state["λ_style_B_dyn"] = λ_style_B_dyn

        writer.add_scalars(
            "B/G",
            {
                "adv": float(advB.item()),
                "NCE": float(nceB.item()),
                "L1_idt": float(idtB.item()),
                "styleTok_B": float(styB.item()),
                "style_gain": float(λ_style_B_eff),
                "tex": float(_to_float(tex_B_total, 0.0)),
                "λ_style_B_dyn": float(λ_style_B_dyn),
                "total": float(totalB.item()) if torch.is_tensor(totalB) else float(totalB),
            },
            global_step,
        )

        fno = cfg["_denorm"](far_mix_noisy)
        xm = cfg["_denorm"](x_mix)
        rc = cfg["_denorm"](recon)
        writer.add_images("B/imgs/x_mix(style target)", xm[:4], global_step)
        writer.add_images("B/imgs/far_mix_noisy(input)", fno[:4], global_step)
        writer.add_images("B/imgs/recon→x_mix", rc[:4], global_step)

    # ✅ SEM TB always
    if writer and sem_enabled and (global_step % tb_freq_sem == 0):
        writer.add_scalars(
            "SEM/B/raw_direct",
            {"MoCo": float(moco_v), "JEPA_content": float(jepa_v), "loss_total": float(sem_total_v),
             "λ_sem": float(lam_sem), "λ_jepa": float(lam_jepa)},
            global_step
        )
        sem_stats = {"moco": moco_v, "jepa": jepa_v, "loss_total": sem_total_v,
                     "lam_sem": lam_sem, "lam_jepa": lam_jepa,
                      "lr_sem": float(state.get("opt_SEM", None).param_groups[0]["lr"]) if state.get("opt_SEM", None) else 0.0}

        _log_sem_curves(writer, "B", sem_total_v, sem_stats, global_step, q_vec=q_vec, k_vec=k_vec)

        writer.add_scalar("SEM/exceptions_total", float(state.get("sem_exceptions", 0)), global_step)
        writer.add_scalar("SEM/B/exception_step", float(sem_exc_B), global_step)

    if sem_enabled and ((global_step % sem_print_every) == 0):
        _tqdm_write(
            f"[SEM/B] step={global_step}  "
            f"MoCo={moco_v:.4f}  JEPAc={jepa_v:.4f}  total={sem_total_v:.4f}  exc={sem_exc_B}"
        )

    # ----------------------------
    # meters epoch (B)
    # ----------------------------
    if adv_enable_B:
        epoch_meters["B"]["D_loss"].add(float(loss_DB.item()))
    epoch_meters["B"]["G_adv"].add(float(advB.item()))
    epoch_meters["B"]["NCE"].add(float(nceB.item()))
    epoch_meters["B"]["L1_idt"].add(float(idtB.item()))
    epoch_meters["B"]["styleTok_B"].add(float(styB.item()))
    epoch_meters["B"]["λ_style_B_dyn"].add(float(state.get("λ_style_B_dyn", λ_style_B_dyn)))
    epoch_meters["B"]["style_gain_B"].add(float(λ_style_B_eff))
    if torch.is_tensor(totalB):
        epoch_meters["B"]["total"].add(float(totalB.item()))

    state["global_step"] = global_step + 1


# =========================================================================================
#           Utilitaires GAN / NCE / style / JEPA
# =========================================================================================

def nce_feats_dict(net, X, *, normalize=True, no_grad=False):
    """
    Extrait un dictionnaire de features pour la perte NCE / contenu.

    Suppose que net.encode_content(X) → (z, (s1, s2, s3, s4, s5))
      - z   : "bot" (carte de contenu la plus profonde)
      - s2  : skip 64×64
      - s3  : skip 32×32
      - s4  : skip 16×16

    On renvoie un dict :
      {
        "bot":   z,
        "skip64": s2,
        "skip32": s3,
        "skip16": s4,
      }
    """
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        z, skips = net.encode_content(X)
        # skips is (s1..sL). Keep backward-compatible naming for first 5 levels when available.
        s1 = skips[0] if len(skips) > 0 else None
        s2 = skips[1] if len(skips) > 1 else s1
        s3 = skips[2] if len(skips) > 2 else s2
        s4 = skips[3] if len(skips) > 3 else s3
        s5 = skips[4] if len(skips) > 4 else s4
        feats = {
            "bot": z,
            "skip64": s2,
            "skip32": s3,
            "skip16": s4,
        }
        if normalize:
            feats = {k: v / (v.norm(dim=1, keepdim=True) + 1e-8) for k, v in feats.items()}
        return feats


def content_nce_two_styles(
        nce_loss,
        feats_x: dict,
        feats_f1: dict,
        feats_f2: dict,
        nce_layers,
        layer_w,
        include_pair12: bool = True,
):
    """
    Perte NCE pour forcer l'invariance du contenu à travers deux styles.

    - feats_x  : features du contenu "pur" x
    - feats_f1 : features de far(x, style_1)
    - feats_f2 : features de far(x, style_2)

    On combine :
      - NCE(x, far_1)
      - NCE(x, far_2)
      - éventuellement NCE(far_1, far_2) pour rapprocher les deux vues stylisées.

    Le tout est moyenné par couche et pondéré par layer_w.
    """
    total = 0.0
    for l, w in zip(nce_layers, layer_w):
        if w == 0:
            continue
        fx = feats_x[l]
        f1 = feats_f1[l]
        f2 = feats_f2[l]

        loss_x1 = nce_loss(fx, f1)
        loss_x2 = nce_loss(fx, f2)
        acc = loss_x1 + loss_x2

        if include_pair12:
            loss_12 = nce_loss(f1, f2)
            acc = (acc + loss_12) / 3.0
        else:
            acc = acc * 0.5

        total = total + w * acc

    return total


# =========================================================================================
#       Contenu sémantique : MoCo + JEPA-content (ResNet50)
# =========================================================================================

import torch
import torch.nn.functional as F
from typing import Dict, Optional

@torch.no_grad()
def _sem_batch_contrastive_metrics(q: torch.Tensor, k: torch.Tensor) -> Dict[str, float]:
    """
    q, k : (B, D) déjà normalisés (L2).
    Calcule :
      - pos_sim : cos(q_i, k_i)
      - neg_sim : moyenne des cos(q_i, k_j) pour j != i (batch negatives)
      - acc1    : top-1 retrieval (argmax(q_i·k_j) == i)
      - margin  : pos_sim - neg_sim
      - err_sem : 1 - pos_sim
    """
    if q.ndim != 2 or k.ndim != 2 or q.size(0) != k.size(0):
        return {}

    B = q.size(0)
    sim = q @ k.t()  # (B,B) cos si normés

    pos = sim.diag()
    # neg = moyenne hors diag
    if B > 1:
        mask = ~torch.eye(B, device=sim.device, dtype=torch.bool)
        neg = sim[mask].view(B, B - 1).mean(dim=1)
        acc1 = (sim.argmax(dim=1) == torch.arange(B, device=sim.device)).float().mean()
        neg_sim = float(neg.mean().item())
        acc1v = float(acc1.item())
    else:
        neg_sim = 0.0
        acc1v = 1.0

    pos_sim = float(pos.mean().item())
    margin = float(pos_sim - neg_sim)
    err_sem = float(1.0 - pos_sim)

    return {
        "pos_sim": pos_sim,
        "neg_sim": neg_sim,
        "acc1": acc1v,
        "margin": margin,
        "err_sem": err_sem,
    }


def log_sem_tensorboard(
    writer,
    global_step: int,
    tag: str,
    stats: Dict[str, float],
    q: Optional[torch.Tensor] = None,
    k: Optional[torch.Tensor] = None,
    cfg: Optional[dict] = None,
):
    """
    Log scalars + (optionnel) histogrammes + embeddings TB.
    - tag ex: "SEM/A" ou "SEM/B"
    """
    if writer is None:
        return

    # scalars principaux (déjà chez toi : MoCo/JEPA/total/λ_*)
    writer.add_scalars(tag, stats, global_step)

    # Diagnostics contrastifs (si q/k fournis)
    if q is not None and k is not None:
        qn = F.normalize(q.detach(), dim=1)
        kn = F.normalize(k.detach(), dim=1)
        diag = _sem_batch_contrastive_metrics(qn, kn)
        if diag:
            writer.add_scalars(f"{tag}_diag", diag, global_step)

        # optionnel : histogrammes
        if cfg is not None and bool(cfg.get("sem_tb_hist", True)):
            try:
                sim = (qn @ kn.t()).detach().float().cpu()
                writer.add_histogram(f"{tag}_hist/sim_matrix", sim, global_step)
                writer.add_histogram(f"{tag}_hist/pos_sim", sim.diag(), global_step)
                if sim.numel() > sim.size(0):  # hors diag
                    mask = ~torch.eye(sim.size(0), dtype=torch.bool)
                    writer.add_histogram(f"{tag}_hist/neg_sim", sim[mask], global_step)
                writer.add_histogram(f"{tag}_hist/q_norm", q.detach().norm(dim=1).float().cpu(), global_step)
                writer.add_histogram(f"{tag}_hist/k_norm", k.detach().norm(dim=1).float().cpu(), global_step)
            except Exception:
                pass

        # optionnel : projector (embedding)
        if cfg is not None:
            every = int(cfg.get("sem_tb_embed_every", 0))
            if every > 0 and (global_step % every == 0):
                max_n = int(cfg.get("sem_tb_embed_max", 256))
                qq = qn[:max_n].detach().float().cpu()
                kk = kn[:max_n].detach().float().cpu()
                try:
                    writer.add_embedding(qq, global_step=global_step, tag=f"{tag}_emb/q")
                    writer.add_embedding(kk, global_step=global_step, tag=f"{tag}_emb/k")
                except Exception:
                    pass


# =========================================================================================
#  ✅ MODIF: semantic_content_step_moco_jepa avec Warmup + Cosine decay (SEM LR)
#  -> Remplace complètement ta fonction actuelle par celle-ci.
# =========================================================================================

import traceback
import torch
import torch.nn.functional as F

def semantic_content_step_moco_jepa(x, far, y, state, cfg):
    dev = cfg["device"]
    sem_every = int(cfg.get("sem_every", 1))
    lam_sem = float(cfg.get("lambda_sem", 0.0))
    if lam_sem <= 0.0:
        return None
    if sem_every > 1 and (state["global_step"] % sem_every != 0):
        return None

    sem = state.get("SEM", None)
    opt_sem = state.get("opt_SEM", None)
    if sem is None or opt_sem is None:
        return None

    # ✅ create scheduler (once)
    ensure_sem_scheduler_in_state(state, cfg)

    sem_use_aug = bool(cfg.get("sem_use_aug", False))
    sem_sym = bool(cfg.get("sem_sym", False))
    sem_two_styles = bool(cfg.get("sem_two_styles", False))
    sem_detach_far = bool(int(cfg.get("sem_detach_far", 1)) != 0)

    writer = state.get("writer", None) or cfg.get("writer", None)
    step_global = int(state.get("global_step", -1))

    x_in = x.to(dev, non_blocking=True)
    far_in = (far.detach() if sem_detach_far else far).to(dev, non_blocking=True)

    far2_in = None
    if sem_two_styles and (y is not None):
        try:
            y2 = y[torch.randperm(y.size(0), device=y.device)]
            with torch.no_grad():
                far2 = state["G_A"](x, style=y2)
            far2_in = (far2.detach() if sem_detach_far else far2).to(dev, non_blocking=True)
        except Exception:
            far2_in = None

    try:
        opt_sem.zero_grad(set_to_none=True)

        # ---------------- MoCo ----------------
        keys_to_enqueue = []

        out = sem.loss_moco(x_in, far_in, apply_aug=sem_use_aug)

        st1 = {}
        if isinstance(out, (tuple, list)) and len(out) == 3:
            loss_moco_1, st1, k1 = out
            if k1 is not None:
                keys_to_enqueue.append(k1)
        else:
            loss_moco_1, st1 = out
            k1 = None

        loss_moco = loss_moco_1
        moco_val = float(st1.get("loss", loss_moco_1.item())) if isinstance(st1, dict) else float(loss_moco_1.item())

        logits_pos = float(st1.get("logits_pos", 0.0)) if isinstance(st1, dict) else 0.0
        logits_neg = float(st1.get("logits_neg", 0.0)) if isinstance(st1, dict) else 0.0

        if far2_in is not None:
            out = sem.loss_moco(x_in, far2_in, apply_aug=sem_use_aug)
            st2 = {}
            if isinstance(out, (tuple, list)) and len(out) == 3:
                loss_moco_2, st2, k2 = out
                if k2 is not None:
                    keys_to_enqueue.append(k2)
            else:
                loss_moco_2, st2 = out

            loss_moco = 0.5 * (loss_moco + loss_moco_2)
            v2 = float(st2.get("loss", loss_moco_2.item())) if isinstance(st2, dict) else float(loss_moco_2.item())
            moco_val = 0.5 * (moco_val + v2)

        if sem_sym:
            out = sem.loss_moco(far_in, x_in, apply_aug=sem_use_aug)
            st_sym = {}
            if isinstance(out, (tuple, list)) and len(out) == 3:
                loss_sym, st_sym, ksym = out
                if ksym is not None:
                    keys_to_enqueue.append(ksym)
            else:
                loss_sym, st_sym = out

            loss_moco = 0.5 * (loss_moco + loss_sym)
            vs = float(st_sym.get("loss", loss_sym.item())) if isinstance(st_sym, dict) else float(loss_sym.item())
            moco_val = 0.5 * (moco_val + vs)

        # ---------------- JEPA semantic (optionnel) ----------------
        jepa_on = bool(cfg.get("sem_jepa_on", False))
        jepa_every = int(cfg.get("jepa_every", 2))
        jepa_mask_ratio = float(cfg.get("jepa_mask_ratio", 0.6))
        loss_jepa = x_in.new_tensor(0.0)
        jepa_val = 0.0

        if jepa_on and getattr(sem, "jepa_use", False) and (step_global % max(1, jepa_every) == 0):
            loss_jepa, info = sem.loss_jepa(x_in, far_in, mask_ratio=jepa_mask_ratio, apply_aug=sem_use_aug)
            try:
                jepa_val = float(info.get("loss_total", info.get("loss", loss_jepa.item())))
            except Exception:
                jepa_val = float(loss_jepa.item())

        lam_jepa = float(cfg.get("lambda_jepa_content", 0.0)) if jepa_on else 0.0

        total = lam_sem * loss_moco + lam_jepa * loss_jepa

        total.backward()
        opt_sem.step()

        # ✅ Step scheduler SEM (Warmup + Cosine) : 1 step par update SEM
        step_sem_scheduler(state)

        # enqueue AFTER step (safe)
        if len(keys_to_enqueue) > 0 and hasattr(sem, "_dequeue_and_enqueue"):
            with torch.no_grad():
                for kk in keys_to_enqueue:
                    sem._dequeue_and_enqueue(kk)

        # EMA key encoder
        with torch.no_grad():
            if hasattr(sem, "momentum_update"):
                sem.momentum_update()

        total_val = float(total.item())
        lr_sem = sem_scheduler_get_lr(opt_sem)

        stats = {
            "MoCo": float(moco_val),
            "JEPA_content": float(jepa_val) if jepa_on else 0.0,
            "loss_total": float(total_val),
            "λ_sem": float(lam_sem),
            "λ_sem_jepa": float(lam_jepa),
            "lr_sem": float(lr_sem),  # ✅ nouveau
            "sem_lr_step": float(state.get("sem_lr_step", 0)),  # ✅ utile pour debug
        }

        # TB scalars
        if writer is not None and step_global >= 0:
            try:
                writer.add_scalar("SEM/MoCo", stats["MoCo"], step_global)
                writer.add_scalar("SEM/JEPA_content", stats["JEPA_content"], step_global)
                writer.add_scalar("SEM/total", stats["loss_total"], step_global)
                writer.add_scalar("SEM/lambda_sem", stats["λ_sem"], step_global)
                writer.add_scalar("SEM/lambda_sem_jepa", stats["λ_sem_jepa"], step_global)
                writer.add_scalar("SEM/lr_sem", stats["lr_sem"], step_global)  # ✅ nouveau
                writer.add_scalar("SEM/sem_lr_step", stats["sem_lr_step"], step_global)  # ✅ nouveau
                if logits_pos != 0.0 or logits_neg != 0.0:
                    writer.add_scalar("SEM/logits_pos", logits_pos, step_global)
                    writer.add_scalar("SEM/logits_neg", logits_neg, step_global)
            except Exception:
                pass

        # (q_vec, k_vec) optionnels: ici on ne les renvoie pas (comme ton code actuel)
        return total_val, stats, None, None

    except Exception as e:
        print(f"[SEM][ERROR] sem step failed @step={step_global}: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return None



def d_forward_logits(D, x):
    out = D(x)
    return out if isinstance(out, torch.Tensor) else out[0]


def d_loss_and_stats(
        D,
        real,
        fake,
        *,
        use_hinge=True,
        r1_gamma=0.0,
        do_r1=False,
        adv_highpass=True,
        highpass_fn=None
):
    """
    Perte du discriminateur (hinge ou LSGAN) + R1 optionnel.
    highpass_fn: fonction highpass(x) si adv_highpass=True.
    """
    if adv_highpass and (highpass_fn is not None):
        real_in = highpass_fn(real)
        fake_in = highpass_fn(fake.detach())
    else:
        real_in = real
        fake_in = fake.detach()

    real_logits = d_forward_logits(D, real_in)
    fake_logits = d_forward_logits(D, fake_in)

    if use_hinge:
        loss_real = F.relu(1. - real_logits).mean()
        loss_fake = F.relu(1. + fake_logits).mean()
    else:
        loss_real = F.mse_loss(real_logits, torch.ones_like(real_logits))
        loss_fake = F.mse_loss(fake_logits, torch.zeros_like(fake_logits))

    loss = loss_real + loss_fake
    r1_pen = real_logits.new_tensor(0.0)

    if do_r1 and r1_gamma > 0:
        real_in_r1 = real_in.detach().requires_grad_(True)
        r_out = d_forward_logits(D, real_in_r1)
        grad = autograd.grad(
            outputs=r_out.sum(),
            inputs=real_in_r1,
            create_graph=True,
            only_inputs=True
        )[0]
        r1_pen = (grad.pow(2).reshape(grad.size(0), -1).sum(1)).mean()
        loss = loss + 0.5 * r1_gamma * r1_pen

    stats = {
        "real_mean": float(real_logits.mean().detach().cpu()),
        "fake_mean": float(fake_logits.mean().detach().cpu()),
        "r1": float(r1_pen.detach().cpu()) if do_r1 and r1_gamma > 0 else 0.0,
    }
    return loss, stats


def g_adv_loss(D, fake, *, use_hinge=True, adv_highpass=True, highpass_fn=None):
    """Perte adversariale générateur."""
    if adv_highpass and (highpass_fn is not None):
        fake_in = highpass_fn(fake)
    else:
        fake_in = fake
    fake_logits = d_forward_logits(D, fake_in)
    if use_hinge:
        return -fake_logits.mean()
    return F.mse_loss(fake_logits, torch.ones_like(fake_logits))


def spade_gate_reg(model: torch.nn.Module, margin: float):
    """Régularisation sur les portes SPADE."""
    sp = torch.nn.Softplus()
    loss = 0.0
    ratios = []
    n = 0
    for m in model.modules():
        if m.__class__.__name__ == "SPADELayer":
            ws = sp(m._p_ws_gamma)
            wg = sp(m._p_wg_gamma)
            loss = loss + F.relu(margin * ws - wg).mean()
            ratios.append(float((wg / (ws + 1e-6)).mean().detach().cpu()))
            n += 1
    avg_ratio = (sum(ratios) / max(1, len(ratios))) if ratios else float("nan")
    return loss / max(1, n), avg_ratio


def style_loss_tokens(fake_img, style_img, teacher_G):
    """Style loss entre deux images via style_enc du teacher."""
    with torch.no_grad():
        _, toks_s, tokG_s = teacher_G.style_enc(style_img)
    _, toks_f, tokG_f = teacher_G.style_enc(fake_img)
    loss_g = F.l1_loss(tokG_f, tokG_s)
    loss_ms = sum(F.l1_loss(tf, ts) for tf, ts in zip(toks_f, toks_s)) / float(len(toks_f))
    return loss_g + loss_ms


def style_loss_tokens_to_target(fake_img, target_tokens, target_tokG, teacher_G):
    """Style loss entre fake et tokens cibles (déjà extraits)."""
    _, toks_f, tokG_f = teacher_G.style_enc(fake_img)
    loss_g = F.l1_loss(tokG_f, target_tokG)
    loss_ms = sum(F.l1_loss(tf, tt) for tf, tt in zip(toks_f, target_tokens)) / float(len(toks_f))
    return loss_g + loss_ms


@torch.no_grad()
def style_dist_tokens(teacher_G, a, b):
    """Distance style (modulable pour le warmup / équilibre de λ_style_B_dyn)."""
    _, ta, ga = teacher_G.style_enc(a)
    _, tb, gb = teacher_G.style_enc(b)
    d_g = F.l1_loss(ga, gb)
    d_ms = sum(F.l1_loss(x, y) for x, y in zip(ta, tb)) / float(len(ta))
    return d_g + d_ms


def stack_tokens_for_jepa(tokens_list, tokG):
    """Empile tokens multi-échelles + tokG → (B,S,D)."""
    seq = [tokG] + list(tokens_list)  # ordre: tokG, t5, t4, t3, t2, t1
    Dset = {t.shape[1] for t in seq}
    if len(Dset) != 1:
        return None
    return torch.stack(seq, dim=1)  # (B,S,D)


def bias_mask(B: int, S: int, base_ratio: float, bias_high: float, device) -> torch.Tensor:
    """Masque JEPA biaisé vers les échelles hautes."""
    imp = torch.linspace(bias_high, 1.0, steps=S, device=device)
    imp = imp / imp.mean()
    p = torch.clamp(base_ratio * imp, 0.0, 1.0)  # (S,)
    u = torch.rand(B, S, device=device)
    return (u < p.view(1, S)).bool()


@torch.no_grad()
def two_style_views(imgs, sigma: float = 0.02, gamma: float = 1.0, flip_p: float = 0.5):
    """
    Deux vues style légèrement différentes.

    On utilise le bruit spectral (training.texture_fft_swd.spectral_noise)
    pour créer de vraies perturbations de texture en fréquence. Cela aligne
    JEPA sur les perturbations de texture que tu utilises déjà pour la phase B.
    """
    v1 = imgs.clone()
    v2 = imgs.clone()

    if sigma > 0:
        v1 = spectral_noise(v1, sigma=sigma, gamma=gamma)
        v2 = spectral_noise(v2, sigma=sigma, gamma=gamma)

    if random.random() < flip_p:
        v1 = torch.flip(v1, dims=[-1])
    if random.random() < flip_p:
        v2 = torch.flip(v2, dims=[-1])

    return v1.clamp(-1, 1), v2.clamp(-1, 1)


def kd_logits_supheads_if_available(G, T_pred, T_teacher):
    """
    Knowledge distillation SupHeads sur les logits si disponible.
    On projette les tokens du teacher et du student via les mêmes sup_heads,
    puis on minimise la divergence symétrique entre leurs distributions.
    """
    if (not hasattr(G, "sup_heads")) or (G.sup_heads is None):
        return T_pred.new_tensor(0.0)

    try:
        # Cas 1 : sup_heads sait déjà gérer des tokens (B,S,D)
        if hasattr(G.sup_heads, "forward_tokens"):
            logits_t = G.sup_heads.forward_tokens(T_teacher.detach())
            logits_p = G.sup_heads.forward_tokens(T_pred)
        else:
            # Cas 2 : on a un MLP classique qui attend du (B, D_flat)
            B, S, D = T_pred.shape
            logits_t = G.sup_heads(T_teacher.detach().reshape(B, S * D))
            logits_p = G.sup_heads(T_pred.reshape(B, S * D))

        def as_list(x):
            return list(x.values()) if isinstance(x, dict) else [x]

        Lt, Lp = as_list(logits_t), as_list(logits_p)
        loss = 0.0
        for a, b in zip(Lt, Lp):
            pt = a.softmax(1).clamp_min(1e-8)
            logps = b.log_softmax(1)
            ps = b.softmax(1).clamp_min(1e-8)
            logpt = a.log_softmax(1)
            # symmetrised KL
            loss += F.kl_div(logps, pt, reduction="batchmean") + F.kl_div(logpt, ps, reduction="batchmean")
        return loss / max(1, len(Lt))
    except Exception:
        # En cas de shape bizarre ou autre, on coupe proprement la branche KD
        return T_pred.new_tensor(0.0)


# =========================================================================================
#           Helpers génériques (freeze, stats, visualisation, FFT mix, CLS, style_gain)
# =========================================================================================

def freeze(m):
    if m is None:
        return
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)


def unfreeze(m):
    if m is None:
        return
    m.train()
    for p in m.parameters():
        p.requires_grad_(True)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def grad_norm(m):
    s = 0.0
    for p in m.parameters():
        if p.grad is not None:
            g = p.grad.norm().item()
            s += g * g
    return math.sqrt(s) if s > 0 else 0.0


def _triplet_grid(a, b, c, max_k=4, nrow=3):
    K = min(max_k, a.size(0), b.size(0), c.size(0))
    rows = []
    for i in range(K):
        rows += [a[i], b[i], c[i]]
    return vutils.make_grid(torch.stack(rows, 0), nrow=nrow)


def _gap(x: torch.Tensor) -> torch.Tensor:
    """Global Average Pooling (+ flatten) si 4D, no-op si déjà 2D."""
    if x.dim() == 4:
        return F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
    return x


def _denorm(t):  # suppose images dans [-1,1] → remet en [0,1]
    return ((t.detach().float().cpu().clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)


def apply_style_gain(style, gain: float):
    """
    Applique un gain multiplicatif sur les tokens de style.

    Cas gérés :
      - style est un dict {"tokens": (t5,...,t1), "token": tokG}
      - style est un dict déjà au format {"tokens": [(t, gain0), ...], "token": (tokG, gain0)}
      - style est un tuple/list (toks, tokG) (fallback build_style_cond_from_tokens)
      - style est un tensor (image) → on le renvoie * gain (au besoin)
      - sinon: on renvoie tel quel.
    """
    if style is None or gain == 1.0:
        return style

    # 1) Cas dict (format "officiel")
    if isinstance(style, dict):
        out = style.copy()
        toks = out.get("tokens", None)
        tokG = out.get("token", None)

        if toks is not None:
            if isinstance(toks, (list, tuple)) and len(toks) > 0 and isinstance(toks[0], tuple):
                out["tokens"] = [(t, gain) for (t, _old_g) in toks]
            else:
                out["tokens"] = [(t, gain) for t in toks]

        if tokG is not None:
            if isinstance(tokG, tuple) and len(tokG) == 2:
                tokG_tensor, _old_g = tokG
                out["token"] = (tokG_tensor, gain)
            else:
                out["token"] = (tokG, gain)

        return out

    # 2) Cas tuple/list (toks, tokG)
    if isinstance(style, (tuple, list)) and len(style) == 2:
        toks, tokG = style
        return {
            "tokens": [(t, gain) for t in toks],
            "token": (tokG, gain),
        }

    # 3) Cas tensor image
    if torch.is_tensor(style):
        return style * gain

    # 4) Type inconnu → on ne casse surtout pas le train
    return style


def run_cls_epoch(
        epoch: int,
        dataloader,
        backbone,  # ex: G_A (pré-entraîné en auto-supervisé)
        token_encoder,  # MultiScaleTokenEncoder
        cls_head,  # TokenClassifier
        opt_backbone,  # optimiser backbone (ou None si on gèle)
        opt_head,  # optimiser tête
        dev,
        freeze_backbone: bool = True,
        writer=None,
        global_step: int = 0,
        tag_prefix: str = "CLS",
):
    """
    Entraîne une époque de classification supervisée.

    - Si freeze_backbone=True : "linear probe" → on ne met à jour que la tête.
    - Si freeze_backbone=False : on fine-tune aussi le backbone (style MoCo finetune).
    """
    backbone.to(dev)
    token_encoder.to(dev)
    cls_head.to(dev)

    if freeze_backbone:
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
    else:
        backbone.train(True)
        for p in backbone.parameters():
            p.requires_grad_(True)

    token_encoder.train(True)
    cls_head.train(True)

    ce_meter = AvgMeter()
    acc_meter = AvgMeter()

    pbar = tqdm(dataloader, desc=f"{tag_prefix} epoch {epoch + 1}", ncols=160, leave=False)

    for batch in pbar:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            imgs, labels = batch[0], batch[1]
        else:
            imgs, labels = batch

        imgs = imgs.to(dev)
        labels = torch.as_tensor(labels, device=dev, dtype=torch.long)

        if opt_backbone is not None:
            opt_backbone.zero_grad(set_to_none=True)
        opt_head.zero_grad(set_to_none=True)

        tokens = token_encoder(imgs)  # (B, S, D)
        logits = cls_head(tokens)  # (B, num_classes)

        loss = F.cross_entropy(logits, labels)

        loss.backward()
        if opt_backbone is not None and (not freeze_backbone):
            opt_backbone.step()
        opt_head.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == labels).float().mean().item()
            ce_meter.add(float(loss.item()), c=imgs.size(0))
            acc_meter.add(acc, c=imgs.size(0))

        pbar.set_postfix(
            CE=f"{ce_meter.avg:.4f}",
            acc=f"{acc_meter.avg * 100:.2f}%",
        )

        if writer is not None:
            writer.add_scalar(f"{tag_prefix}/train_CE", float(loss.item()), global_step)
            writer.add_scalar(f"{tag_prefix}/train_acc", float(acc), global_step)

        global_step += 1

    print(
        f"[{tag_prefix}] epoch {epoch + 1:03d} — "
        f"CE={ce_meter.avg:.4f}  acc={acc_meter.avg * 100:.2f}%"
    )

    return global_step, ce_meter.avg, acc_meter.avg


# =========================================================================================
#                       Mode sup_freeze — SEM ONLY (pas de G_A/G_B)
# =========================================================================================

def run_sup_freeze_sem_only(
        opt,
        loaders,
        dev,
        writer=None,
        tb_freq_C: int = 50,
        global_step_start: int = 0,
):
    """Entraîne uniquement des SupHeads multi-tâches sur un backbone sémantique gelé.

    Objectif : reproduire le comportement de sup_freeze côté 'style', mais sans nécessiter G_A/G_B/D_A/D_B.
    - Les labels proviennent de MultiTaskDataset (via --data_json + --classes_json).
    - Le backbone sémantique est construit comme en détection (resnet{50,101,152} + return_layer).
    - On peut reprendre le backbone depuis --resume_dir s'il contient SemBackbone_epoch*.pt.
    - On sauvegarde SupHeads + SemBackbone dans save_dir/sup_freeze/fold_xx/.
    """

    import shutil

    from training.train_detection_transformer import _build_sem_resnet_backbone
    from training.checkpoint import should_save_ckpt

    sup_sem_imagenet_norm = int(getattr(opt, "sup_sem_imagenet_norm", 1)) == 1
    ckpt_safe_write = bool(int(getattr(opt, "safe_write", 1)) != 0)

    # --- checkpoint frequency (align with train loop behaviour)
    def _parse_save_freq_local(save_freq_value):
        """Parse save_freq in a user-friendly way.

        Accepted:
          - None / 'none'
          - 'epoch' / 'epoch:N'
          - 'step'  / 'step:N'
          - integer or digit-string -> epoch:N (more intuitive for training runs)
        """
        if save_freq_value is None:
            return "none", None
        # allow int
        if isinstance(save_freq_value, (int, float)) and not isinstance(save_freq_value, bool):
            n = int(save_freq_value)
            return ("epoch", max(1, n)) if n > 0 else ("none", None)
        sf = str(save_freq_value).strip().lower()
        if sf in {"", "none"}:
            return "none", None
        if sf.startswith("epoch"):
            parts = sf.split(":", 1)
            if len(parts) == 2 and parts[1].strip().isdigit():
                return "epoch", max(1, int(parts[1].strip()))
            return "epoch", 1
        if sf.startswith("step"):
            parts = sf.split(":", 1)
            if len(parts) == 2 and parts[1].strip().isdigit():
                return "step", max(1, int(parts[1].strip()))
            return "step", 1
        if sf.isdigit():
            # by default, numeric means "every N epochs" (safer than step)
            return "epoch", max(1, int(sf))
        return "epoch", 1

    save_freq_mode, save_freq_interval = _parse_save_freq_local(getattr(opt, "save_freq", None))
    epoch_ckpt_interval = getattr(opt, "epoch_ckpt_interval", None)
    try:
        epoch_ckpt_interval = int(epoch_ckpt_interval) if epoch_ckpt_interval is not None else None
        if epoch_ckpt_interval is not None and epoch_ckpt_interval <= 0:
            epoch_ckpt_interval = None
    except Exception:
        epoch_ckpt_interval = None

    # Build semantic backbone
    sem_backbone, sem_out_channels = _build_sem_resnet_backbone(
        pretrained=bool(int(getattr(opt, "sem_pretrained", 1))),
        arch=str(getattr(opt, "det_sem_backbone", "resnet50")),
        return_layer=str(getattr(opt, "det_sem_return_layer", "layer4")),
        pretrained_path=str(getattr(opt, "sem_pretrained_path", "") or ""),
        strict=bool(int(getattr(opt, "sem_pretrained_strict", 0))),
        verbose=bool(int(getattr(opt, "sem_pretrained_verbose", 1))),
    )
    sem_backbone = sem_backbone.to(dev)

    # Optional resume: load latest SemBackbone from resume_dir
    resume_dir = getattr(opt, "resume_dir", None)
    if resume_dir:
        rdir = Path(resume_dir)

        # Copy frozen GAN weights (and other useful run files) to current save_dir so that
        # the run directory remains self-contained for later testing.
        # This is important because sup_freeze_sem_only trains ONLY SupHeads, but users
        # often expect the save_dir to also contain G/D weights used for feature extraction.
        try:
            out_root = Path(opt.save_dir)
            out_root.mkdir(parents=True, exist_ok=True)
            patterns = [
                "G_A_epoch*.pt", "G_B_epoch*.pt", "D_A_epoch*.pt", "D_B_epoch*.pt",
                "trainer_epoch*.pth", "state.json", "train_cfg.json"
            ]
            for pat in patterns:
                for src in sorted(rdir.glob(pat), key=lambda p: p.stat().st_mtime):
                    dst = out_root / src.name
                    if not dst.exists():
                        shutil.copy2(src, dst)
        except Exception as e:
            print(f"[WARN] sup_freeze_sem_only: copy frozen weights from resume_dir failed: {e}")

        cand = sorted(list(rdir.glob("SemBackbone_epoch*.pt")), key=lambda p: p.stat().st_mtime)
        if cand:
            try:
                bundle = torch.load(cand[-1], map_location=str(dev))
                sd = bundle.get("state_dict", bundle)
                sem_backbone.load_state_dict(sd, strict=False)
                print(f"✓ [sup_freeze_sem_only] reprise SemBackbone depuis {cand[-1].name}")
            except Exception as e:
                print(f"[WARN] reprise SemBackbone impossible ({cand[-1]}): {e}")

    sem_backbone.eval()
    for p in sem_backbone.parameters():
        p.requires_grad_(False)

    # ImageNet normalization
    _im_mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    _im_std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    def _sem_feats(imgs: torch.Tensor) -> torch.Tensor:
        x = imgs
        if x.dim() != 4:
            raise ValueError(f"sup_freeze_sem_only: expected BCHW, got {tuple(x.shape)}")
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        if sup_sem_imagenet_norm:
            x = (x + 1.0) * 0.5
            x = x.clamp(0.0, 1.0)
            x = (x - _im_mean) / _im_std
        out = sem_backbone(x)
        if isinstance(out, dict):
            feat = out.get("0", None)
            if feat is None:
                feat = next(iter(out.values()))
        else:
            feat = out
        return feat.mean(dim=(2, 3))

    # --- training hyperparams
    λ_sup = float(getattr(opt, "lambda_sup", 1.0))
    eval_every = int(getattr(opt, "sup_eval_every", 1))
    reset_between = bool(getattr(opt, "sup_reset_between_folds", True))

    k_folds = len(loaders) if len(loaders) > 0 else 1
    folds_order = list(range(k_folds)) if k_folds >= 1 else [0]
    global_step = int(global_step_start)

    def _infer_tasks(train_loader):
        ds = train_loader.dataset
        ds = ds.dataset if isinstance(ds, Subset) else ds
        if hasattr(ds, "task_classes") and isinstance(ds.task_classes, dict) and ds.task_classes:
            return {task: len(lst) for task, lst in ds.task_classes.items()}
        if hasattr(ds, "classes") and isinstance(ds.classes, (list, tuple)) and ds.classes:
            return {"default": len(ds.classes)}
        return {"default": int(getattr(opt, "sup_num_classes", 2))}

    def _make_heads_and_opt(train_loader):
        tasks = _infer_tasks(train_loader)
        sup_heads = SupHeads(
            tasks,
            int(sem_out_channels),
            num_scales=6,
            token_mode="flat",
            heads=int(getattr(opt, "sup_heads_nheads", 4)),
            dropout=float(getattr(opt, "sup_heads_dropout", 0.1)),
            mlp_mult=int(getattr(opt, "sup_heads_mlp_mult", 2)),
        ).to(dev)
        for p in sup_heads.parameters():
            p.requires_grad_(True)
        opt_Sup = torch.optim.Adam(
            sup_heads.parameters(),
            lr=float(getattr(opt, "sup_lr", 1e-4)),
            betas=(0.9, 0.999),
        )
        return tasks, sup_heads, opt_Sup

    for stage, tr_fold in enumerate(folds_order):
        train_loader = loaders[tr_fold]
        val_loaders = loaders

        if stage == 0 or reset_between:
            tasks, sup_heads, opt_Sup = _make_heads_and_opt(train_loader)
        else:
            sup_heads.train(True)

        sup_dir = Path(opt.save_dir) / "sup_freeze" / f"fold_{tr_fold:02d}"
        sup_dir.mkdir(parents=True, exist_ok=True)

        best_val = float("inf")
        best_state = None
        best_epoch = -1

        if writer:
            writer.add_text(
                "C/run",
                f"sup_freeze_sem_only — stage {stage} — train on fold {tr_fold} (in_dim={int(sem_out_channels)})",
                global_step,
            )

        for epoch in range(int(getattr(opt, "epochs", 1))):
            sup_heads.train(True)

            ce_meter, acc_meter = AvgMeter(), AvgMeter()
            pbar = tqdm(train_loader, desc=f"C[sup_freeze_sem]-fold{tr_fold} ep{epoch+1}/{opt.epochs}", ncols=160, leave=False)
            for batch in pbar:
                if len(batch) == 3:
                    imgs, raw, _paths = batch
                elif len(batch) >= 2:
                    imgs, raw = batch[0], batch[1]
                else:
                    imgs, raw = batch
                imgs = imgs.to(dev)

                feats = _sem_feats(imgs)
                logits, attn = sup_heads(feats, return_attn=True)

                if not isinstance(logits, dict):
                    logits = {"default": logits}
                if not isinstance(raw, dict):
                    raw = {"default": raw}

                loss = 0.0
                used = 0
                for t, out in logits.items():
                    if t not in raw:
                        continue
                    y = torch.as_tensor(raw[t], device=dev, dtype=torch.long)
                    mask = (y >= 0) & (y < out.size(1))
                    if mask.any():
                        loss_t = F.cross_entropy(out[mask], y[mask])
                        loss = loss + loss_t
                        used += 1

                if used == 0:
                    continue

                loss = loss / float(used)
                loss = λ_sup * loss
                opt_Sup.zero_grad(set_to_none=True)
                loss.backward()
                opt_Sup.step()

                # simple accuracy on first task present
                with torch.no_grad():
                    t0 = next(iter(logits.keys()))
                    out0 = logits[t0]
                    y0 = torch.as_tensor(raw.get(t0, []), device=dev, dtype=torch.long)
                    if y0.numel() == out0.size(0):
                        acc = float((out0.argmax(1) == y0).float().mean().item())
                    else:
                        acc = 0.0

                ce_meter.add(float(loss.item()), c=imgs.size(0))
                acc_meter.add(float(acc), c=imgs.size(0))

                if writer and (global_step % max(1, tb_freq_C) == 0):
                    writer.add_scalar("C_sem/train_CE", float(loss.item()), global_step)
                    writer.add_scalar("C_sem/train_acc", float(acc), global_step)

                pbar.set_postfix(CE=f"{ce_meter.avg:.4f}", acc=f"{acc_meter.avg*100:.2f}%")

                # ---- periodic ckpt during sup_freeze_sem_only
                if save_freq_mode == "step":
                    do_save, reason = should_save_ckpt(
                        save_mode=save_freq_mode,
                        interval=save_freq_interval,
                        step=global_step,
                        epoch=epoch,
                        epochs_total=int(getattr(opt, "epochs", 1)),
                        final_save=False,
                    )
                    if do_save:
                        save_supheads_rich(sup_heads, sup_dir / f"SupHeads_step{global_step}.pt", safe_write=ckpt_safe_write)
                        save_sem_backbone_rich(sem_backbone, sup_dir / f"SemBackbone_step{global_step}.pt", safe_write=ckpt_safe_write)
                        if writer:
                            writer.add_text("C_sem/ckpt", f"saved({reason}) step={global_step} epoch={epoch}", global_step)

                global_step += 1

            # eval
            if (epoch + 1) % max(1, eval_every) == 0:
                sup_heads.eval()
                val_ce = 0.0
                val_n = 0
                with torch.no_grad():
                    for vloader in val_loaders:
                        for batch in vloader:
                            if len(batch) >= 2:
                                imgs, raw = batch[0], batch[1]
                            else:
                                imgs, raw = batch
                            imgs = imgs.to(dev)
                            feats = _sem_feats(imgs)
                            _out = sup_heads(feats, return_attn=False)
                            logits = _out[0] if isinstance(_out, tuple) else _out
                            if not isinstance(logits, dict):
                                logits = {"default": logits}
                            if not isinstance(raw, dict):
                                raw = {"default": raw}
                            for t, out in logits.items():
                                if t not in raw:
                                    continue
                                y = torch.as_tensor(raw[t], device=dev, dtype=torch.long)
                                mask = (y >= 0) & (y < out.size(1))
                                if mask.any():
                                    ce = F.cross_entropy(out[mask], y[mask]).item()
                                    n = int(mask.sum().item())
                                    val_ce += ce * n
                                    val_n += n
                val_ce = val_ce / max(1, val_n)
                if writer:
                    writer.add_scalar("C_sem/val_CE", float(val_ce), global_step)

                if val_ce < best_val:
                    best_val = float(val_ce)
                    best_state = {k: v.detach().cpu() for k, v in sup_heads.state_dict().items()}
                    best_epoch = int(epoch)

            # ---- epoch-based ckpt frequency
            if save_freq_mode != "step":
                do_save, reason = should_save_ckpt(
                    save_mode=save_freq_mode,
                    interval=save_freq_interval,
                    step=global_step,
                    epoch=epoch,
                    epochs_total=int(getattr(opt, "epochs", 1)),
                    final_save=False,
                )
                if do_save:
                    save_supheads_rich(sup_heads, sup_dir / f"SupHeads_last_epoch{epoch}.pt", safe_write=ckpt_safe_write)
                    save_sem_backbone_rich(sem_backbone, sup_dir / f"SemBackbone_epoch{epoch}.pt", safe_write=ckpt_safe_write)

            # ---- optional richer bundle ckpt (epoch_ckpt_interval)
            if epoch_ckpt_interval is not None and (epoch + 1) % int(epoch_ckpt_interval) == 0:
                try:
                    torch.save(
                        {
                            "meta": {"epoch": int(epoch), "global_step": int(global_step)},
                            "sup_heads": {k: v.detach().cpu() for k, v in sup_heads.state_dict().items()},
                            "sem_backbone": {k: v.detach().cpu() for k, v in sem_backbone.state_dict().items()},
                            "opt_sup": opt_Sup.state_dict(),
                        },
                        sup_dir / f"Ckpt_semSup_epoch{epoch}.pt",
                    )
                except Exception:
                    pass

        # save best
        if best_state is not None:
            torch.save({"meta": {"best_epoch": best_epoch, "best_val": best_val}, "state_dict": best_state},
                       sup_dir / "SupHeads_best.pt")
            save_sem_backbone_rich(sem_backbone, sup_dir / f"SemBackbone_best_epoch{best_epoch}.pt", safe_write=ckpt_safe_write)

    if writer:
        writer.flush()

    return