# training/checkpoint.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Optional, Union, Tuple, Dict, List
import json
import os
import tempfile
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


# ======================================================================
# 0) Logging : remplace les print par tqdm.write (stable avec barres)
# ======================================================================

def _log(msg: str) -> None:
    """Log compatible tqdm (n'écrase pas la barre)."""
    try:
        tqdm.write(str(msg))
    except Exception:
        print(msg)


# ======================================================================
# 1) Détection du dernier epoch
# ======================================================================

def last_epoch(
    dir_path: Union[str, Path],
    stem: str,
    exts: Tuple[str, ...] = (".pt", ".pth", ".safetensors"),
) -> Optional[int]:
    """
    Renvoie l’index d’époque le plus élevé pour des fichiers :
        <stem>_epoch{n}{ext}
    Exemple :  G_A_epoch17.pt  ou  trainer_epoch42.pth
    """
    dir_path = Path(dir_path)
    if not dir_path.exists():
        return None

    pattern = re.compile(rf"{re.escape(stem)}_epoch(\d+)")
    epochs: List[int] = []

    for p in dir_path.glob(f"{stem}_epoch*"):
        if (p.suffix.lower() not in exts) and (not any(p.name.endswith(ext) for ext in exts)):
            continue
        m = pattern.search(p.stem)
        if m:
            try:
                epochs.append(int(m.group(1)))
            except Exception:
                pass

    return max(epochs) if epochs else None


# ======================================================================
# 1b) Décision centralisée de sauvegarde
# ======================================================================

def should_save_ckpt(
    *,
    save_mode: str,
    interval: Optional[int],
    step: int,
    epoch: int,
    epochs_total: int,
    final_save: bool = True,
) -> Tuple[bool, str]:
    """
    Décide si on doit sauvegarder, et renvoie (do_save, reason).

    - epoch est 0-based
    - step est l'itération globale
    - save_mode ∈ {"none","step","epoch"}
    - interval: N (pour step:N ou epoch:N)
    - final_save: si True, on sauvegarde aussi au dernier epoch.
    """
    mode = (save_mode or "none").strip().lower()
    if mode not in {"none", "step", "epoch"}:
        mode = "none"

    if interval is not None:
        try:
            interval = max(1, int(interval))
        except Exception:
            interval = None

    is_final_epoch = (epoch + 1) >= int(epochs_total)

    if mode == "none":
        if final_save and is_final_epoch:
            return True, "final"
        return False, ""

    if mode == "step":
        if interval is None:
            interval = 1
        if interval > 0 and (int(step) % int(interval) == 0):
            return True, f"step:{int(interval)}"
        if final_save and is_final_epoch:
            return True, "final"
        return False, ""

    # mode == "epoch"
    if interval is None:
        interval = 1
    if (int(epoch) + 1) % int(interval) == 0:
        return True, f"epoch:{int(interval)}"
    if final_save and is_final_epoch:
        return True, "final"
    return False, ""


# ────────────────────────────────────────────────────────────────
# 2) Remap rétro-compat + filtrage legacy attn-style
# ────────────────────────────────────────────────────────────────

def _remap_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Mappings rétro-compatibles :
      • 'attn.'  → 'attentions.'        (anciens SupHeads)
      • 'clf.'   → 'classifiers.'       (anciens SupHeads)

    NOTE (SPADE) :
      Les anciennes clés du style-encoder attentionnel n'ont **pas d'équivalent**
      dans SPADE/SEAN avec tokens multi-échelles. On les ignore volontairement.
    """
    new_sd: Dict[str, torch.Tensor] = {}

    legacy_style_prefixes = (
        "style_enc.bot_pre", "style_enc.bot_attn", "style_enc.bot_norm",
        "style_enc.to_mu", "style_enc.to_std", "style_enc.cross_blocks",
        "style_enc.style_tok",
        "style_enc.q_proj", "style_enc.k_proj", "style_enc.v_proj",
        "style_enc.self_attn", "style_enc.cross_attn",
    )

    def _remap_one(k: str) -> Optional[Tuple[str, torch.Tensor]]:
        if any(k.startswith(pref) for pref in legacy_style_prefixes):
            return None

        v = sd[k]

        if k.startswith("attn."):
            return k.replace("attn.", "attentions.", 1), v
        if k.startswith("clf."):
            return k.replace("clf.", "classifiers.", 1), v

        if k.startswith("sup_heads.attn."):
            return k.replace("sup_heads.attn.", "sup_heads.attentions.", 1), v
        if k.startswith("sup_heads.clf."):
            return k.replace("sup_heads.clf.", "sup_heads.classifiers.", 1), v

        return k, v

    for k in list(sd.keys()):
        out = _remap_one(k)
        if out is not None:
            new_sd[out[0]] = out[1]

    return new_sd


# ────────────────────────────────────────────────────────────────
# 3) Chargement générique (DP + remap + safetensors)
# ────────────────────────────────────────────────────────────────

def _load_weights(
    model: nn.Module,
    path: Path,
    device: torch.device,
    strict: bool = True,
) -> None:
    """
    Charge un state_dict sur 'model' depuis 'path' (.pt/.pth/.safetensors).
    Remappe/filtre les anciennes clés, gère le préfixe DataParallel 'module.'
    et bascule en strict=False si nécessaire, en loggant les différences.
    """
    if not path.exists():
        _log(f"[WARN] Poids manquant : {path}")
        return

    # --- lecture brute
    if path.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file
        except Exception as e:
            _log(f"[WARN] safetensors n'est pas installé ({e}). Impossible de lire {path.name}.")
            return
        raw_sd = load_file(str(path), device=str(device))
    else:
        obj = torch.load(path, map_location=device)
        raw_sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj

    # --- info legacy attn-style
    legacy_prefixes = (
        "style_enc.bot_pre", "style_enc.bot_attn", "style_enc.bot_norm",
        "style_enc.to_mu", "style_enc.to_std", "style_enc.cross_blocks",
        "style_enc.style_tok", "style_enc.q_proj", "style_enc.k_proj",
        "style_enc.v_proj", "style_enc.self_attn", "style_enc.cross_attn"
    )
    is_legacy_attn_style = any(any(k.startswith(pref) for pref in legacy_prefixes) for k in raw_sd.keys())
    if is_legacy_attn_style:
        _log(
            "[INFO] Checkpoint *legacy* détecté (AttnStyleEncoder). "
            "Les clés de style obsolètes seront ignorées (SPADE/SEAN multi-échelles)."
        )

    # --- remap + filtrage legacy
    sd = _remap_keys(raw_sd)

    # --- DP prefix handling ('module.')
    model_keys = list(model.state_dict().keys())
    model_has_module = any(k.startswith("module.") for k in model_keys)
    ckpt_has_module = any(k.startswith("module.") for k in sd.keys())

    if ckpt_has_module and not model_has_module:
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    elif (not ckpt_has_module) and model_has_module:
        sd = {(f"module.{k}" if not k.startswith("module.") else k): v for k, v in sd.items()}

    # --- tentative de chargement
    try:
        model.load_state_dict(sd, strict=strict)
    except RuntimeError as e:
        if strict:
            _log(f"[INFO] strict=True a échoué sur {path.name} — retry strict=False.\n      ({e})")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            preview = ", ".join(missing[:6]) + ("…" if len(missing) > 6 else "")
            _log(f"   → Clés manquantes ({len(missing)}) : {preview}")
        if unexpected:
            preview = ", ".join(unexpected[:6]) + ("…" if len(unexpected) > 6 else "")
            _log(f"   → Clés inattendues ({len(unexpected)}) : {preview}")


# ────────────────────────────────────────────────────────────────
# 4) Sauvegarde “riche” SupHeads (meta + state_dict)
# ────────────────────────────────────────────────────────────────

def save_supheads_rich(sup_heads: nn.Module, out_path: Path, *, safe_write: bool = True) -> None:
    """
    Sauvegarde un bundle 'rich' pour SupHeads :
      {"meta": {...}, "state_dict": ...}
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "tasks": getattr(sup_heads, "tasks", {}),
        "in_dim": getattr(sup_heads, "in_dim", None),
        "token_mode": getattr(sup_heads, "token_mode", "multi6"),
        "num_scales": getattr(sup_heads, "num_scales", 6),
        "heads": getattr(sup_heads, "heads", getattr(sup_heads, "nheads", 4)),
        "mlp_mult": getattr(sup_heads, "mlp_mult", 2),
        "dropout": float(getattr(sup_heads, "dropout", 0.1)),
        "mlp_depth": getattr(sup_heads, "mlp_depth", None),
    }
    if meta["in_dim"] is None and hasattr(sup_heads, "classifiers") and sup_heads.classifiers:
        first = next(iter(sup_heads.classifiers.values()))
        meta["in_dim"] = int(first.weight.shape[1])

    bundle = {"meta": meta, "state_dict": sup_heads.state_dict()}

    if not safe_write:
        torch.save(bundle, out_path)
    else:
        with tempfile.NamedTemporaryFile(delete=False, dir=str(out_path.parent)) as tmp:
            tmp_name = tmp.name
        try:
            torch.save(bundle, tmp_name)
            os.replace(tmp_name, out_path)
        finally:
            try:
                os.remove(tmp_name)
            except Exception:
                pass

    _log(f"✓ SupHeads (rich) sauvegardé → {out_path}")


def save_sem_backbone_rich(sem_backbone: nn.Module, out_path: Path, *, meta: Optional[Dict[str, Any]] = None,
                           safe_write: bool = True) -> None:
    """Sauvegarde un backbone sémantique sous forme {meta, state_dict}.

    On garde un format similaire à la branche 'sem_model' de save_checkpoint, afin que le chargement
    automatique depuis --weights_dir puisse fonctionner avec un simple torch.load.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if meta is None:
        meta = {}
        try:
            meta["arch"] = str(getattr(sem_backbone, "arch", ""))
            meta["return_layer"] = str(getattr(sem_backbone, "return_layer", ""))
        except Exception:
            pass

    bundle = {"meta": meta, "state_dict": sem_backbone.state_dict()}
    _atomic_save_torch(bundle, out_path, safe_write=safe_write)
    _log(f"✓ SemBackbone (rich) sauvegardé → {out_path}")


# ────────────────────────────────────────────────────────────────
# 5) Helpers d’écriture atomique (torch / safetensors)
# ────────────────────────────────────────────────────────────────

def _atomic_save_torch(obj: Any, fpath: Path, *, safe_write: bool = True) -> None:
    fpath = Path(fpath)
    fpath.parent.mkdir(parents=True, exist_ok=True)

    if not safe_write:
        torch.save(obj, fpath)
        return

    with tempfile.NamedTemporaryFile(delete=False, dir=str(fpath.parent)) as tmp:
        tmp_name = tmp.name

    try:
        torch.save(obj, tmp_name)
        os.replace(tmp_name, fpath)
    finally:
        try:
            os.remove(tmp_name)
        except Exception:
            pass


def _atomic_save_safetensors(sd: Dict[str, torch.Tensor], fpath: Path, *, safe_write: bool = True) -> None:
    try:
        from safetensors.torch import save_file
    except Exception:
        # fallback (selon env)
        from torch.safetensors import save_file  # type: ignore

    fpath = Path(fpath)
    fpath.parent.mkdir(parents=True, exist_ok=True)

    if not safe_write:
        save_file(sd, str(fpath))
        return

    with tempfile.NamedTemporaryFile(delete=False, dir=str(fpath.parent)) as tmp:
        tmp_name = tmp.name

    try:
        save_file(sd, tmp_name)
        os.replace(tmp_name, fpath)
    finally:
        try:
            os.remove(tmp_name)
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────
# 6) Bundles JEPA “rich” (meta + state) + helpers
# ────────────────────────────────────────────────────────────────

def _infer_jepa_meta(jepa_mod: nn.Module) -> Dict[str, Any]:
    meta = {
        "D": None,
        "hidden": None,
        "heads": getattr(jepa_mod, "heads", None),
        "norm": bool(getattr(jepa_mod, "norm", False)),
        "scale_weights": getattr(jepa_mod, "scale_weights", None),
        "mask_ratio": getattr(jepa_mod, "mask_ratio", None),
    }
    try:
        if hasattr(jepa_mod, "net") and isinstance(jepa_mod.net, nn.Sequential):
            linears = [m for m in jepa_mod.net if isinstance(m, nn.Linear)]
            if len(linears) >= 2:
                meta["hidden"] = int(linears[0].out_features)
                meta["D"] = int(linears[-1].out_features)
            elif len(linears) == 1:
                meta["D"] = int(linears[0].out_features)
    except Exception:
        pass
    return meta


def save_jepa_rich(jepa_mod: nn.Module, out_path: Path, *, safe_write: bool = True) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"meta": _infer_jepa_meta(jepa_mod), "state_dict": jepa_mod.state_dict()}

    if not safe_write:
        torch.save(bundle, out_path)
    else:
        with tempfile.NamedTemporaryFile(delete=False, dir=str(out_path.parent)) as tmp:
            tmp_name = tmp.name
        try:
            torch.save(bundle, tmp_name)
            os.replace(tmp_name, out_path)
        finally:
            try:
                os.remove(tmp_name)
            except Exception:
                pass

    _log(f"✓ JEPA (rich) sauvegardé → {out_path}")


def _load_jepa_weights(
    jepa_mod: nn.Module,
    path: Path,
    device: torch.device,
    strict: bool = True,
) -> Dict[str, Any]:
    if not path.exists():
        _log(f"[WARN] JEPA manquant : {path}")
        return {}
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
        meta = obj.get("meta", {})
    else:
        sd = obj
        meta = {}

    try:
        jepa_mod.load_state_dict(sd, strict=strict)
    except RuntimeError as e:
        if strict:
            _log(f"[INFO] JEPA strict=True a échoué sur {path.name} — retry strict=False.\n      ({e})")
        missing, unexpected = jepa_mod.load_state_dict(sd, strict=False)
        if missing:
            _log(f"   → JEPA: clés manquantes ({len(missing)})")
        if unexpected:
            _log(f"   → JEPA: clés inattendues ({len(unexpected)})")
    return meta


# ────────────────────────────────────────────────────────────────
# 7) Sauvegarde complète d’un checkpoint (+ Teachers + JEPA + SEM)
# ────────────────────────────────────────────────────────────────

def save_checkpoint(
    epoch: int,
    G_A,
    D_A,
    G_B,
    D_B,
    opt_GA,
    opt_DA,
    opt_GB,
    opt_DB,
    global_step: int,
    out_dir: Path,
    *,
    use_safetensors: bool = False,
    safe_write: bool = True,
    amp_scaler: Optional[Any] = None,
    sched_GA: Optional[Any] = None,
    sched_GB: Optional[Any] = None,
    sem_model: Optional[nn.Module] = None,
    opt_sem: Optional[Any] = None,
    sem_filename: str = "SemMoCo",
    sup_heads: Optional[nn.Module] = None,
    sup_filename: str = "SupHeads",
    save_supheads_every_epoch: bool = True,
    T_A: Optional[nn.Module] = None,
    T_B: Optional[nn.Module] = None,
    save_teachers: bool = True,
    tokJEPA_A: Optional[nn.Module] = None,
    tokJEPA_B: Optional[nn.Module] = None,
    save_jepa_every_epoch: bool = True,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # safetensors dispo ?
    if use_safetensors:
        try:
            import safetensors.torch as _st  # noqa: F401
        except Exception as e:
            _log(f"[WARN] safetensors indisponible ({e}) → fallback .pt")
            use_safetensors = False

    # 1) Poids réseaux
    if use_safetensors:
        _atomic_save_safetensors(G_A.state_dict(), out_dir / f"G_A_epoch{epoch}.safetensors", safe_write=safe_write)
        _atomic_save_safetensors(D_A.state_dict(), out_dir / f"D_A_epoch{epoch}.safetensors", safe_write=safe_write)
        _atomic_save_safetensors(G_B.state_dict(), out_dir / f"G_B_epoch{epoch}.safetensors", safe_write=safe_write)
        _atomic_save_safetensors(D_B.state_dict(), out_dir / f"D_B_epoch{epoch}.safetensors", safe_write=safe_write)
    else:
        _atomic_save_torch(G_A.state_dict(), out_dir / f"G_A_epoch{epoch}.pt", safe_write=safe_write)
        _atomic_save_torch(D_A.state_dict(), out_dir / f"D_A_epoch{epoch}.pt", safe_write=safe_write)
        _atomic_save_torch(G_B.state_dict(), out_dir / f"G_B_epoch{epoch}.pt", safe_write=safe_write)
        _atomic_save_torch(D_B.state_dict(), out_dir / f"D_B_epoch{epoch}.pt", safe_write=safe_write)

    # 1b) Branche sémantique
    if sem_model is not None:
        try:
            meta: Dict[str, Any] = {}
            try:
                bb = getattr(sem_model, "backbone", None)
                if bb is not None:
                    meta["backbone_arch"] = str(getattr(bb, "arch", ""))
                    meta["return_layer"] = str(getattr(bb, "return_layer", ""))
                meta["embed_dim"] = int(getattr(sem_model, "embed_dim", 0) or 0)
            except Exception:
                pass

            if use_safetensors:
                _atomic_save_safetensors(
                    sem_model.state_dict(),
                    out_dir / f"{sem_filename}_epoch{epoch}.safetensors",
                    safe_write=safe_write,
                )
                try:
                    (out_dir / f"{sem_filename}_epoch{epoch}.meta.json").write_text(
                        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
                    )
                except Exception:
                    pass
            else:
                _atomic_save_torch(
                    {"state_dict": sem_model.state_dict(), "meta": meta},
                    out_dir / f"{sem_filename}_epoch{epoch}.pt",
                    safe_write=safe_write,
                )
        except Exception as e:
            _log(f"[WARN] sauvegarde {sem_filename} impossible: {e}")

    # 2) État entraîneur (optimiseurs + RNG/AMP/Schedulers)  ✅ FIX: try/except complets
    trainer: Dict[str, Any] = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "opt_GA": opt_GA.state_dict(),
        "opt_DA": opt_DA.state_dict(),
        "opt_GB": opt_GB.state_dict(),
        "opt_DB": opt_DB.state_dict(),
        "opt_SEM": (opt_sem.state_dict() if opt_sem is not None else None),
        "rng_state": torch.get_rng_state(),
        "py_rng_state": random.getstate(),
        "np_rng_state": np.random.get_state(),
    }

    if torch.cuda.is_available():
        try:
            trainer["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass

    if amp_scaler is not None:
        try:
            trainer["amp_scaler"] = amp_scaler.state_dict()
        except Exception:
            pass

    if sched_GA is not None:
        try:
            trainer["sched_GA"] = sched_GA.state_dict()
        except Exception:
            pass

    if sched_GB is not None:
        try:
            trainer["sched_GB"] = sched_GB.state_dict()
        except Exception:
            pass

    _atomic_save_torch(trainer, out_dir / f"trainer_epoch{epoch}.pth", safe_write=safe_write)

    # 3) Export SupHeads “riche”
    if save_supheads_every_epoch:
        sup = (
            sup_heads
            or getattr(G_A, "sup_heads", None)
            or getattr(G_B, "sup_heads", None)
            or getattr(G_A, "Sup", None)
            or getattr(G_B, "Sup", None)
        )
        if isinstance(sup, nn.Module):
            save_supheads_rich(sup, out_dir / f"{sup_filename}_epoch{epoch}.pth", safe_write=safe_write)

    # 4) Teachers EMA
    if save_teachers:
        if T_A is not None:
            if use_safetensors:
                _atomic_save_safetensors(T_A.state_dict(), out_dir / f"T_A_epoch{epoch}.safetensors", safe_write=safe_write)
            else:
                _atomic_save_torch(T_A.state_dict(), out_dir / f"T_A_epoch{epoch}.pt", safe_write=safe_write)
        if T_B is not None:
            if use_safetensors:
                _atomic_save_safetensors(T_B.state_dict(), out_dir / f"T_B_epoch{epoch}.safetensors", safe_write=safe_write)
            else:
                _atomic_save_torch(T_B.state_dict(), out_dir / f"T_B_epoch{epoch}.pt", safe_write=safe_write)

    # 5) JEPA
    if save_jepa_every_epoch:
        if tokJEPA_A is not None:
            save_jepa_rich(tokJEPA_A, out_dir / f"TokenJEPA_A_epoch{epoch}.pth", safe_write=safe_write)
        if tokJEPA_B is not None:
            save_jepa_rich(tokJEPA_B, out_dir / f"TokenJEPA_B_epoch{epoch}.pth", safe_write=safe_write)

    _log(f"✓ checkpoint @epoch {epoch} sauvegardé → {out_dir}")


# ────────────────────────────────────────────────────────────────
# 8) Chargement checkpoints (+ Teachers + JEPA + SEM)
# ────────────────────────────────────────────────────────────────

def load_checkpoint(
    run_dir: Union[str, Path],
    epoch: int,
    G_A,
    D_A,
    G_B,
    D_B,
    opt_GA=None,
    opt_DA=None,
    opt_GB=None,
    opt_DB=None,
    *,
    sem_model: Optional[nn.Module] = None,
    opt_sem: Optional[Any] = None,
    sem_filename: str = "SemMoCo",
    device: str = "cpu",
    strict_GA: bool = False,
    strict_GB: bool = False,
    strict: Optional[bool] = None,
    weights_only: bool = False,
    prefer_exts: Tuple[str, ...] = (".pt", ".safetensors"),
    T_A: Optional[nn.Module] = None,
    T_B: Optional[nn.Module] = None,
    strict_TA: bool = False,
    strict_TB: bool = False,
    tokJEPA_A: Optional[nn.Module] = None,
    tokJEPA_B: Optional[nn.Module] = None,
    strict_tokJEPA: bool = True,
) -> dict:
    run_dir = Path(run_dir)
    dev = torch.device(device)

    if strict is not None:
        strict_GA = bool(strict_GA or strict)
        strict_GB = bool(strict_GB or strict)

    def _find_weight_file(stem: str) -> Optional[Path]:
        for ext in prefer_exts:
            p = run_dir / f"{stem}_epoch{epoch}{ext}"
            if p.exists():
                return p
        for ext in (".pt", ".safetensors", ".pth"):
            p = run_dir / f"{stem}_epoch{epoch}{ext}"
            if p.exists():
                return p
        return None

    # --- poids réseaux ---
    pa = _find_weight_file("G_A")
    da = _find_weight_file("D_A")
    pb = _find_weight_file("G_B")
    db = _find_weight_file("D_B")

    if pa is not None:
        _load_weights(G_A, pa, dev, strict=strict_GA)
    else:
        _log(f"[WARN] Fichier poids manquant: G_A_epoch{epoch}(.pt|.safetensors)")
    if da is not None:
        _load_weights(D_A, da, dev, strict=False)
    else:
        _log(f"[WARN] Fichier poids manquant: D_A_epoch{epoch}(.pt|.safetensors)")
    if pb is not None:
        _load_weights(G_B, pb, dev, strict=strict_GB)
    else:
        _log(f"[WARN] Fichier poids manquant: G_B_epoch{epoch}(.pt|.safetensors)")
    if db is not None:
        _load_weights(D_B, db, dev, strict=False)
    else:
        _log(f"[WARN] Fichier poids manquant: D_B_epoch{epoch}(.pt|.safetensors)")

    # --- SEM ---
    if sem_model is not None:
        ps = _find_weight_file(sem_filename)
        if ps is not None:
            _load_weights(sem_model, ps, dev, strict=False)
        else:
            _log(f"[INFO] Pas de poids trouvés pour {sem_filename} @epoch {epoch}")

    # --- Teachers ---
    if T_A is not None:
        ta = _find_weight_file("T_A")
        if ta is not None:
            _load_weights(T_A, ta, dev, strict=strict_TA)
        else:
            _log(f"[INFO] Pas de poids teacher trouvés pour T_A @epoch {epoch}")
    if T_B is not None:
        tb = _find_weight_file("T_B")
        if tb is not None:
            _load_weights(T_B, tb, dev, strict=strict_TB)
        else:
            _log(f"[INFO] Pas de poids teacher trouvés pour T_B @epoch {epoch}")

    # --- JEPA ---
    jepa_meta: Dict[str, Any] = {}
    if tokJEPA_A is not None:
        jp = run_dir / f"TokenJEPA_A_epoch{epoch}.pth"
        if jp.exists():
            jepa_meta["A"] = _load_jepa_weights(tokJEPA_A, jp, dev, strict=strict_tokJEPA)
        else:
            alt = _find_weight_file("TokenJEPA_A")
            if alt is not None:
                jepa_meta["A"] = _load_jepa_weights(tokJEPA_A, alt, dev, strict=strict_tokJEPA)
            else:
                _log(f"[INFO] Pas de bundle JEPA A trouvé @epoch {epoch}")

    if tokJEPA_B is not None:
        jp = run_dir / f"TokenJEPA_B_epoch{epoch}.pth"
        if jp.exists():
            jepa_meta["B"] = _load_jepa_weights(tokJEPA_B, jp, dev, strict=strict_tokJEPA)
        else:
            alt = _find_weight_file("TokenJEPA_B")
            if alt is not None:
                jepa_meta["B"] = _load_jepa_weights(tokJEPA_B, alt, dev, strict=strict_tokJEPA)
            else:
                _log(f"[INFO] Pas de bundle JEPA B trouvé @epoch {epoch}")

    if weights_only:
        return {"epoch": epoch, "global_step": 0, "jepa_meta": jepa_meta}

    # --- état entraîneur ---
    tfile = run_dir / f"trainer_epoch{epoch}.pth"
    if not tfile.exists():
        return {"epoch": epoch, "global_step": 0, "jepa_meta": jepa_meta}

    def _torch_load_full(path: Path, map_location: torch.device):
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)

    try:
        trainer = _torch_load_full(tfile, dev)
    except Exception as e:
        _log(f"[WARN] trainer not loaded (will re-init trainer): {e}")
        trainer = {}

    if not isinstance(trainer, dict):
        trainer = {"trainer_obj": trainer}

    # Load optimizers (existant dans ton code initial : opt_DB & opt_sem)
    if opt_DB is not None and isinstance(trainer.get("opt_DB", None), dict):
        try:
            opt_DB.load_state_dict(trainer["opt_DB"])
        except Exception as e:
            _log(f"[WARN] opt_DB state not loaded: {e}")

    if opt_sem is not None and trainer.get("opt_SEM", None) is not None:
        try:
            opt_sem.load_state_dict(trainer["opt_SEM"])
        except Exception as e:
            _log(f"[WARN] opt_SEM state not loaded: {e}")

    # RNG states
    if trainer.get("rng_state", None) is not None:
        try:
            rs = trainer["rng_state"]
            torch.set_rng_state(rs.cpu() if hasattr(rs, "cpu") else rs)
        except Exception as e:
            _log(f"[WARN] torch RNG state not restored: {e}")

    if trainer.get("py_rng_state", None) is not None:
        try:
            random.setstate(trainer["py_rng_state"])
        except Exception as e:
            _log(f"[WARN] python RNG state not restored: {e}")

    if trainer.get("np_rng_state", None) is not None:
        try:
            np.random.set_state(trainer["np_rng_state"])
        except Exception as e:
            _log(f"[WARN] numpy RNG state not restored: {e}")

    if torch.cuda.is_available() and trainer.get("cuda_rng_state_all", None) is not None:
        try:
            torch.cuda.set_rng_state_all(trainer["cuda_rng_state_all"])
        except Exception as e:
            _log(f"[WARN] cuda RNG state not restored: {e}")

    return {
        "epoch": trainer.get("epoch", epoch),
        "global_step": trainer.get("global_step", 0),
        "amp_scaler": trainer.get("amp_scaler", None),
        "sched_GA": trainer.get("sched_GA", None),
        "sched_GB": trainer.get("sched_GB", None),
        "jepa_meta": jepa_meta,
    }


# ────────────────────────────────────────────────────────────────
# 9) Snapshot JSON lisible – étendu avec + d’options JEPA
# ────────────────────────────────────────────────────────────────

def save_state_json(epoch: int, global_step: int, opt, out_dir: Path) -> None:
    """
    Écrit un snapshot lisible (<out_dir>/train_state.json).
    """
    snapshot = {
        "schema_version": 7,
        "epoch": int(epoch),
        "global_step": int(global_step),

        "k_folds": getattr(opt, "k_folds", None),
        "fold_epochs": getattr(opt, "fold_epochs", None),
        "feat_switch_epoch": getattr(opt, "feat_switch_epoch", None),

        "phases": {
            "adv_only_epochs": getattr(opt, "adv_only_epochs", None),
            "adv_mix_epochs": getattr(opt, "adv_mix_epochs", None),
            "recon_epochs": getattr(opt, "recon_epochs", None),
            "adv_boost": getattr(opt, "adv_boost", None),
            "b_boost": getattr(opt, "b_boost", None),
            "skip_amix": getattr(opt, "skip_amix", None),
            "amix_ramp": getattr(opt, "amix_ramp", None),
        },

        "lambdas": {
            "style_lambda": getattr(opt, "style_lambda", None),
            "A_adv": {"lambda_nce": getattr(opt, "lambda_nce_a_adv", None),
                      "lambda_reg": getattr(opt, "lambda_reg_a_adv", None)},
            "A_mix": {"lambda_nce": getattr(opt, "lambda_nce_a_mix", None),
                      "lambda_reg": getattr(opt, "lambda_reg_a_mix", None)},
            "B": {"lambda_nce": getattr(opt, "lambda_nce_b", None),
                  "lambda_idt": getattr(opt, "lambda_idt_b", getattr(opt, "lambda_reg_b", None))},
            "C": {"lambda_sup": getattr(opt, "lambda_sup", None)},
            "legacy": {"lambda_nce": getattr(opt, "lambda_nce", None),
                       "lambda_reg": getattr(opt, "lambda_reg", None)},
        },

        "nce": {
            "nce_t": getattr(opt, "nce_t", None),
            "layers": getattr(opt, "nce_layers", None),
            "layer_weights": getattr(opt, "nce_layer_weights", None),
            "intra": getattr(opt, "nce_intra", None),
            "inter": getattr(opt, "nce_inter", None),
            "max_patches": getattr(opt, "nce_max_patches", None),
            "gate": getattr(opt, "nce_gate", None),
            "momentum": getattr(opt, "nce_m", None),
            "ema_update_every": getattr(opt, "ema_update_every", None),
        },

        "texture": {
            "tex_enable": getattr(opt, "tex_enable", None),
            "tex_apply_A": getattr(opt, "tex_apply_A", None),
            "tex_sigma": getattr(opt, "tex_sigma", None),
            "tex_gamma": getattr(opt, "tex_gamma", None),
            "tex_use_fft": getattr(opt, "tex_use_fft", None),
            "tex_use_swd": getattr(opt, "tex_use_swd", None),
            "lambda_fft": getattr(opt, "lambda_fft", None),
            "lambda_swd": getattr(opt, "lambda_swd", None),
            "swd_levels": getattr(opt, "swd_levels", None),
            "swd_patch": getattr(opt, "swd_patch", None),
            "swd_proj": getattr(opt, "swd_proj", None),
            "swd_max_patches": getattr(opt, "swd_max_patches", None),
        },

        "optim": {
            "lr": getattr(opt, "lr", None),
            "batch_size": getattr(opt, "batch_size", None),
            "ema_tau": getattr(opt, "ema_tau", None),
        },

        "replay": {
            "replay_ratio": getattr(opt, "replay_ratio", None),
            "replay_size": getattr(opt, "replay_size", None),
        },

        "supervised": {
            "mode": getattr(opt, "mode", None),
            "warmup_epochs": getattr(opt, "warmup_epochs", None),
            "sup_ratio": getattr(opt, "sup_ratio", None),
            "sup_from": getattr(opt, "sup_from", None),
            "sup_feats": getattr(opt, "sup_feats", None),
            "sup_feat_type": getattr(opt, "sup_feat_type", None),
            "sup_tasks_json": getattr(opt, "sup_tasks_json", None),
        },

        "jepa": {
            "enabled": bool(getattr(opt, "jepa_tokens", False)),
            "lambda_jepa": getattr(opt, "lambda_jepa", None),
            "every": getattr(opt, "jepa_every", None),
            "mask_ratio": getattr(opt, "jepa_mask_ratio", None),
            "mask_bias_high": getattr(opt, "jepa_mask_bias_high", None),
            "scale_weights": getattr(opt, "jepa_scale_weights", None),
            "hidden_mult": getattr(opt, "jepa_hidden_mult", None),
            "heads": getattr(opt, "jepa_heads", None),
            "norm": getattr(opt, "jepa_norm", None),
            "lambda_var": getattr(opt, "lambda_jepa_var", None),
            "lambda_cov": getattr(opt, "lambda_jepa_cov", None),
            "lambda_kd": getattr(opt, "lambda_jepa_kd", None),
            "use_teacher": getattr(opt, "jepa_use_teacher", None),
        },

        "style_injection": {
            "type": "SPADE_SEAN",
            "option": "Option2_maps_from_content_tokens_from_style",
            "multi_scale_tokens": True,
            "style_nc": getattr(opt, "style_nc", None),
            "spade_ch": getattr(opt, "spade_ch", None),
            "token_dim": getattr(opt, "token_dim", None),
        },

        "logging": {
            "tb": getattr(opt, "tb", None),
            "tb_freq": getattr(opt, "tb_freq", None),
            "save_freq": getattr(opt, "save_freq", None),
            "save_dir": getattr(opt, "save_dir", None),
        },

        "data": {
            "data": getattr(opt, "data", None),
            "data_json": getattr(opt, "data_json", None),
            "classes_json": getattr(opt, "classes_json", None),
            "search_folder": getattr(opt, "search_folder", None),
            "find_images_by_sub_folder": getattr(opt, "find_images_by_sub_folder", None),
        },
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_state.json").write_text(json.dumps(snapshot, indent=4), encoding="utf-8")
    _log(f"✓ train_state.json mis à jour ({out_dir})")
