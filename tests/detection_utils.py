#!/usr/bin/env python3
# testsFile/detectionUtils.py
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from tqdm import tqdm


# ---------------------------------------------------------------------
# Optional deps
# ---------------------------------------------------------------------
def _try_import_cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except Exception:
        return None


def _try_import_pycocotools() -> bool:
    try:
        from pycocotools.cocoeval import COCOeval  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------
# Simple image utils
# ---------------------------------------------------------------------
def _to_device_images(
    images: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]],
    dev: torch.device,
) -> List[torch.Tensor]:
    """Normalise en list[Tensor(C,H,W)] sur device."""
    if torch.is_tensor(images):
        return [img.to(dev) for img in images]
    return [img.to(dev) for img in list(images)]


def _ensure_chw_float01(img: torch.Tensor) -> torch.Tensor:
    """Best-effort: assure float32, [0,1], CHW."""
    if img.dtype != torch.float32:
        img = img.float()
    if img.numel() > 0 and img.max().item() > 1.5:
        img = img / 255.0
    return img


# ---------------------------------------------------------------------
# Checkpoint key helpers
# ---------------------------------------------------------------------
def _strip_module_prefix_if_needed(state: Dict[str, Any]) -> Dict[str, Any]:
    """Supprime un éventuel préfixe 'module.' (DDP/DataParallel)."""
    if not isinstance(state, dict):
        return state
    if not any(k.startswith("module.") for k in state.keys()):
        return state
    return {k[len("module.") :]: v for k, v in state.items()}


def _guess_head_type_from_state(state: Dict[str, Any]) -> Optional[str]:
    """
    Heuristique: détecte si c'est un FasterRCNN ou un FastRNN checkpoint.
    """
    if not isinstance(state, dict) or not state:
        return None

    keys = list(state.keys())

    # typique torchvision FasterRCNN
    if any(k.startswith("rpn.") for k in keys) or any(k.startswith("roi_heads.") for k in keys):
        return "fasterrcnn"

    # ton detector custom
    if any(k.startswith("head.") for k in keys):
        return "fastrnn"

    return None


def _infer_num_classes_from_state(state: Dict[str, Any], head_type: str) -> Optional[int]:
    """
    Infère num_classes depuis le checkpoint (utile pour éviter les size-mismatch).
    - FasterRCNN: roi_heads.box_predictor.cls_score.weight -> [num_classes, in_features]
    - FastRNN: tente quelques clés probables (fallback).
    """
    head_type = str(head_type).lower().strip()

    if head_type == "fasterrcnn":
        w = state.get("roi_heads.box_predictor.cls_score.weight", None)
        if torch.is_tensor(w) and w.ndim == 2:
            return int(w.shape[0])
        b = state.get("roi_heads.box_predictor.cls_score.bias", None)
        if torch.is_tensor(b) and b.ndim == 1:
            return int(b.shape[0])
        return None

    if head_type == "fastrnn":
        # clés typiques possibles (à adapter si ton head a un naming différent)
        candidates = [
            "head.cls.weight",
            "head.classifier.weight",
            "head.fc_cls.weight",
            "head.cls_score.weight",
        ]
        for k in candidates:
            w = state.get(k, None)
            if torch.is_tensor(w) and w.ndim == 2:
                return int(w.shape[0])
        candidates_b = [
            "head.cls.bias",
            "head.classifier.bias",
            "head.fc_cls.bias",
            "head.cls_score.bias",
        ]
        for k in candidates_b:
            b = state.get(k, None)
            if torch.is_tensor(b) and b.ndim == 1:
                return int(b.shape[0])
        return None

    return None


def _remap_backbone_keys_to_match_model(
    state: Dict[str, Any],
    model_state_keys: List[str],
) -> Dict[str, Any]:
    """
    Gère le cas backbone.body.* <-> backbone.*
    selon ce que le modèle ATTEND réellement.

    - Si le checkpoint a backbone.body.* mais le modèle attend backbone.* -> on enlève ".body"
    - Si le checkpoint a backbone.* mais le modèle attend backbone.body.* -> on ajoute ".body"
    """
    if not isinstance(state, dict) or not state:
        return state

    has_ckpt_body = any(k.startswith("backbone.body.") for k in state.keys())
    has_ckpt_plain = any(k.startswith("backbone.") for k in state.keys()) and not has_ckpt_body

    model_has_body = any(k.startswith("backbone.body.") for k in model_state_keys)
    model_has_plain = any(k.startswith("backbone.") for k in model_state_keys) and not model_has_body

    # ckpt: backbone.body.*, model: backbone.*
    if has_ckpt_body and model_has_plain:
        new_state: Dict[str, Any] = {}
        for k, v in state.items():
            if k.startswith("backbone.body."):
                new_state[k.replace("backbone.body.", "backbone.", 1)] = v
            else:
                new_state[k] = v
        return new_state

    # ckpt: backbone.*, model: backbone.body.*
    if has_ckpt_plain and model_has_body:
        new_state: Dict[str, Any] = {}
        for k, v in state.items():
            if k.startswith("backbone.") and not k.startswith("backbone.body."):
                new_state[k.replace("backbone.", "backbone.body.", 1)] = v
            else:
                new_state[k] = v
        return new_state

    return state


def _drop_fasterrcnn_predictor_keys(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    En cas de mismatch voulu (changement de nb de classes), on peut ignorer la tête :
    - roi_heads.box_predictor.*
    """
    if not isinstance(state, dict) or not state:
        return state
    bad_prefix = "roi_heads.box_predictor."
    return {k: v for k, v in state.items() if not k.startswith(bad_prefix)}


# ---------------------------------------------------------------------
# Checkpoint + hyperparams loading
# ---------------------------------------------------------------------
def _load_detector_ckpt(
    ckpt_path: Union[str, Path],
    map_location: str = "cpu",
) -> Dict[str, Any]:
    ckpt_path = str(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=map_location)

    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Invalid detector checkpoint (not a dict): {ckpt_path}")

    # accept legacy key
    if "model" not in ckpt and "state_dict" in ckpt:
        ckpt["model"] = ckpt["state_dict"]

    if "model" not in ckpt:
        raise RuntimeError(
            f"Invalid detector checkpoint (missing 'model'/'state_dict'): {ckpt_path}"
        )

    if "hparams" in ckpt and ckpt["hparams"] is not None and not isinstance(ckpt["hparams"], dict):
        raise RuntimeError("ckpt['hparams'] must be a dict when present")

    return ckpt


def _load_hparams_json(hparams_json: Union[str, Path]) -> Dict[str, Any]:
    p = Path(hparams_json)
    if not p.is_file():
        raise FileNotFoundError(f"hparams_json not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise RuntimeError(f"hparams_json must be a dict, got {type(obj)}")
    return obj


def _infer_hparams_json_from_ckpt(ckpt_path: Union[str, Path]) -> Optional[Path]:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        return None
    ckpt_dir = ckpt_path.parent
    for name in ("hyperparameters.json", "hparams.json", "hparams_detection.json"):
        p = ckpt_dir / name
        if p.is_file():
            return p
    return None


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out.update({k: v for k, v in override.items() if v is not None})
    return out


# ---------------------------------------------------------------------
# HParams dataclass
# ---------------------------------------------------------------------
@dataclass
class DetectorHParams:
    # Core
    head_type: str = "fastrnn"         # "fastrnn" | "fasterrcnn"
    num_classes: int = 81              # inclut background = 0
    # Older ckpts may store feat_source="sem_resnet50".
    # Newer code uses feat_source="sem_resnet" and selects depth via det_sem_backbone.
    feat_source: str = "sem_resnet"
    freeze_backbone: bool = True
    img_h: int = 256
    img_w: int = 256
    sem_pretrained: bool = True

    # ✅ ResNet semantic backbone depth (resnet50/resnet101/resnet152)
    det_sem_backbone: str = "resnet50"

    det_sem_return_layer: str = "layer4"
    # Optional SSL-pretrained semantic backbone checkpoint (MoCo/JEPA/etc.)
    sem_pretrained_path: str = ""
    sem_pretrained_strict: bool = False
    sem_pretrained_verbose: bool = True


    # FastRNN params
    fastrnn_hidden: int = 256
    fastrnn_bidir: bool = True
    fastrnn_dropout: float = 0.0
    fastrnn_focal_alpha: float = 0.25
    fastrnn_focal_gamma: float = 2.0
    fastrnn_score_thresh: float = 0.05
    fastrnn_nms_thresh: float = 0.5
    fastrnn_topk: int = 1000
    fastrnn_size_divisible: int = 32

    # FasterRCNN params
    frcnn_rpn_pre_nms_top_n_train: int = 2000
    frcnn_rpn_pre_nms_top_n_test: int = 1000
    frcnn_rpn_post_nms_top_n_train: int = 2000
    frcnn_rpn_post_nms_top_n_test: int = 1000
    frcnn_rpn_nms_thresh: float = 0.7
    frcnn_rpn_fg_iou_thresh: float = 0.7
    frcnn_rpn_bg_iou_thresh: float = 0.3
    frcnn_rpn_batch_size_per_image: int = 256
    frcnn_rpn_positive_fraction: float = 0.5

    frcnn_box_score_thresh: float = 0.05
    frcnn_box_nms_thresh: float = 0.5
    frcnn_box_detections_per_img: int = 100

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DetectorHParams":
        allowed = {f.name for f in fields(DetectorHParams)}
        filtered = {k: v for k, v in d.items() if k in allowed}
        return DetectorHParams(**filtered)


# ---------------------------------------------------------------------
# Backbone builders
# ---------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# Robust SSL backbone loader (MoCo/JEPA-style) for torchvision ResNet{50,101,152}
# --------------------------------------------------------------------------------------
def _load_resnet_backbone_weights(
    *,
    resnet: torch.nn.Module,
    ckpt_path: str,
    device: torch.device = torch.device("cpu"),
    strict: bool = True,
    verbose: bool = True,
) -> None:
    """
    Load SSL/MoCo-style backbone weights into a torchvision ResNet (50/101/152).

    Supports checkpoints where backbone weights are under keys like:
      - state_dict / model / backbone / encoder / net (auto-detected)
    And parameter prefixes like:
      - module.
      - base_encoder. / backbone_q. / encoder_q. / backbone. / encoder.

    Also remaps timm-like 'stem.0.*' -> 'conv1.*' and 'stem.1.*' -> 'bn1.*'.
    Drops classification head keys (fc.*, head.*, classifier.*).

    To keep strict=True possible:
      - start from the model's full state_dict and overwrite only matching keys.
    """
    import os

    if not ckpt_path or str(ckpt_path).strip() == "":
        raise ValueError("ckpt_path is empty")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    # 1) Extract raw state dict
    state = None
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "model_state", "net", "encoder", "backbone"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]
                break
        if state is None:
            state = ckpt
    else:
        raise RuntimeError(f"Unsupported checkpoint type: {type(ckpt)}")

    # 2) Strip DDP "module." repeatedly
    def strip_prefix(sd: dict, pref: str) -> dict:
        fixed = {}
        for k, v in sd.items():
            if k.startswith(pref):
                fixed[k[len(pref):]] = v
            else:
                fixed[k] = v
        return fixed

    for _ in range(3):
        if any(k.startswith("module.") for k in state.keys()):
            state = strip_prefix(state, "module.")

    # 3) Remap keys
    def remap_key(k: str) -> str:
        prefixes = [
            "base_encoder.", "backbone_q.", "encoder_q.", "encoder.", "backbone.",
            "model.", "net.",
        ]
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
                break

        if k.startswith("stem.0."):
            k = "conv1." + k[len("stem.0."):]
        elif k.startswith("stem.1."):
            k = "bn1." + k[len("stem.1."):]
        # drop heads
        if k.startswith("fc.") or k.startswith("head.") or k.startswith("classifier."):
            k = "__DROP__." + k
        return k

    remapped: Dict[str, Any] = {}
    for k, v in state.items():
        rk = remap_key(k)
        if rk.startswith("__DROP__."):
            continue
        remapped[rk] = v

    # 4) Full target state dict then overwrite matches
    target_sd = resnet.state_dict()
    target_keys = set(target_sd.keys())

    loaded_keys: List[str] = []
    shape_mismatch: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []

    for k, v in remapped.items():
        if k not in target_keys:
            continue
        if not torch.is_tensor(v):
            continue
        if target_sd[k].shape != v.shape:
            shape_mismatch.append((k, tuple(v.shape), tuple(target_sd[k].shape)))
            continue
        target_sd[k] = v
        loaded_keys.append(k)

    resnet.load_state_dict(target_sd, strict=bool(strict))

    if verbose:
        print(f"[DET][SEM] Loaded backbone ckpt: {ckpt_path}")
        print(f"[DET][SEM] strict={strict} | loaded_backbone_keys={len(loaded_keys)}")
        if loaded_keys:
            print(f"[DET][SEM] example loaded keys: {loaded_keys[:8]}")
        if shape_mismatch:
            print(f"[DET][SEM] ⚠️ shape mismatches skipped: {len(shape_mismatch)}")
            for k, s_src, s_tgt in shape_mismatch[:10]:
                print(f"  - {k}: ckpt{s_src} != model{s_tgt}")

def _build_sem_resnet_backbone(
    *,
    pretrained: bool,
    arch: str,
    return_layer: str,
    pretrained_path: str = "",
    strict: bool = False,
    verbose: bool = True,
) -> Tuple[torch.nn.Module, int]:
    """
    Build semantic ResNet backbone (50/101/152) and return an IntermediateLayerGetter
    exposing {return_layer: "0"} with attribute .out_channels.

    If `pretrained_path` is provided, loads SSL weights (MoCo/JEPA-style) with
    `_load_resnet_backbone_weights`, optionally with strict=True.

    Otherwise, uses torchvision supervised weights when `pretrained=True`.
    """
    import torchvision
    from torchvision.models._utils import IntermediateLayerGetter

    arch = str(arch).lower().strip()
    if arch not in {"resnet50", "resnet101", "resnet152"}:
        raise ValueError(f"det_sem_backbone must be one of {{resnet50,resnet101,resnet152}}, got {arch}")

    # torchvision supervised weights (optional)
    weights = None
    if pretrained and not str(pretrained_path).strip():
        try:
            if arch == "resnet50":
                weights = torchvision.models.ResNet50_Weights.DEFAULT
            elif arch == "resnet101":
                weights = torchvision.models.ResNet101_Weights.DEFAULT
            else:
                weights = torchvision.models.ResNet152_Weights.DEFAULT
        except Exception:
            weights = None

    if arch == "resnet50":
        m = torchvision.models.resnet50(weights=weights)
    elif arch == "resnet101":
        m = torchvision.models.resnet101(weights=weights)
    else:
        m = torchvision.models.resnet152(weights=weights)

    # SSL backbone override
    if str(pretrained_path).strip():
        _load_resnet_backbone_weights(
            resnet=m,
            ckpt_path=str(pretrained_path),
            device=torch.device("cpu"),
            strict=bool(strict),
            verbose=bool(verbose),
        )

    layer_to_c = {"layer2": 512, "layer3": 1024, "layer4": 2048}
    if return_layer not in layer_to_c:
        raise ValueError(f"det_sem_return_layer must be one of {list(layer_to_c.keys())}, got {return_layer}")

    out_channels = layer_to_c[return_layer]
    backbone = IntermediateLayerGetter(m, return_layers={return_layer: "0"})
    setattr(backbone, "out_channels", int(out_channels))
    return backbone, int(out_channels)


# ---------------------------------------------------------------------
# FasterRCNN builder
# ---------------------------------------------------------------------
def _build_fasterrcnn_detector(
    backbone: torch.nn.Module,
    num_classes: int,
    hp: DetectorHParams,
) -> torch.nn.Module:
    """
    FasterRCNN torchvision. Compatible avec le checkpoint FasterRCNN
    (rpn.*, roi_heads.* etc.).
    """
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.rpn import AnchorGenerator

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),      # 1 feature map -> tuple len 1
        aspect_ratios=((0.5, 1.0, 2.0),),
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=int(num_classes),
        rpn_anchor_generator=anchor_generator,

        rpn_pre_nms_top_n_train=int(hp.frcnn_rpn_pre_nms_top_n_train),
        rpn_pre_nms_top_n_test=int(hp.frcnn_rpn_pre_nms_top_n_test),
        rpn_post_nms_top_n_train=int(hp.frcnn_rpn_post_nms_top_n_train),
        rpn_post_nms_top_n_test=int(hp.frcnn_rpn_post_nms_top_n_test),
        rpn_nms_thresh=float(hp.frcnn_rpn_nms_thresh),
        rpn_fg_iou_thresh=float(hp.frcnn_rpn_fg_iou_thresh),
        rpn_bg_iou_thresh=float(hp.frcnn_rpn_bg_iou_thresh),
        rpn_batch_size_per_image=int(hp.frcnn_rpn_batch_size_per_image),
        rpn_positive_fraction=float(hp.frcnn_rpn_positive_fraction),

        box_score_thresh=float(hp.frcnn_box_score_thresh),
        box_nms_thresh=float(hp.frcnn_box_nms_thresh),
        box_detections_per_img=int(hp.frcnn_box_detections_per_img),
    )
    return model


# ---------------------------------------------------------------------
# Build detector from checkpoint
# ---------------------------------------------------------------------
def build_detector_from_checkpoint(
    ckpt_path: Union[str, Path],
    dev: torch.device,
    *,
    hparams_json: Optional[Union[str, Path]] = None,
    auto_hparams_json: bool = True,
    prefer_json_over_ckpt: bool = True,
    strict: bool = False,   # IMPORTANT: par défaut False (diagnostic)
    auto_fix_num_classes: bool = True,
    drop_predictor_if_mismatch: bool = False,
) -> Tuple[torch.nn.Module, DetectorHParams]:
    """
    Reconstruit le détecteur du test.

    FIX PRINCIPAL:
      - auto-détection du head_type à partir des clés du state_dict du ckpt.
        (ex: rpn./roi_heads. => FasterRCNN)
      - remap backbone.body <-> backbone si nécessaire.
      - auto-fix num_classes depuis le checkpoint (évite les size mismatch),
        et option pour ignorer la tête si mismatch volontaire.
    """
    ckpt = _load_detector_ckpt(ckpt_path, map_location=str(dev))

    state = ckpt["model"]
    if not isinstance(state, dict):
        raise RuntimeError("[detection_utils] ckpt['model'] must be a state_dict dict.")
    state = _strip_module_prefix_if_needed(state)

    ckpt_hp = ckpt.get("hparams") or {}
    if not isinstance(ckpt_hp, dict):
        raise RuntimeError("ckpt['hparams'] must be a dict when present")

    json_hp: Dict[str, Any] = {}
    inferred_json_path: Optional[Path] = None

    if hparams_json is not None:
        json_hp = _load_hparams_json(hparams_json)
    elif auto_hparams_json:
        inferred_json_path = _infer_hparams_json_from_ckpt(ckpt_path)
        if inferred_json_path is not None:
            json_hp = _load_hparams_json(inferred_json_path)

    if json_hp and ckpt_hp:
        merged = _merge_dicts(ckpt_hp, json_hp) if prefer_json_over_ckpt else _merge_dicts(json_hp, ckpt_hp)
    elif json_hp:
        merged = json_hp
    elif ckpt_hp:
        merged = ckpt_hp
    else:
        merged = {}

    hp = DetectorHParams.from_dict(merged)

    # ---- AUTO head_type override if mismatch with checkpoint ----
    guessed = _guess_head_type_from_state(state)
    if guessed is not None and str(hp.head_type).lower().strip() != guessed:
        print(
            f"[detection_utils] ⚠️ head_type from hparams='{hp.head_type}' "
            f"but checkpoint looks like '{guessed}'. Overriding head_type -> '{guessed}'."
        )
        hp.head_type = guessed

    # ---- AUTO num_classes override to match checkpoint (avoids size mismatch) ----
    if auto_fix_num_classes:
        inferred_nc = _infer_num_classes_from_state(state, head_type=str(hp.head_type))
        if inferred_nc is not None and int(inferred_nc) != int(hp.num_classes):
            msg = (
                f"[detection_utils] ⚠️ num_classes from hparams={hp.num_classes} "
                f"but checkpoint has {inferred_nc}."
            )
            if drop_predictor_if_mismatch and str(hp.head_type).lower().strip() == "fasterrcnn":
                print(msg + " Will DROP predictor weights (roi_heads.box_predictor.*) and keep hp.num_classes.")
            else:
                print(msg + f" Overriding hp.num_classes -> {inferred_nc}")
                hp.num_classes = int(inferred_nc)

    if inferred_json_path is not None:
        print(f"[detection_utils] auto hparams_json loaded: {inferred_json_path}")
    if hparams_json is not None:
        print(f"[detection_utils] hparams_json used: {Path(hparams_json)}")
    print(f"[detection_utils] head_type={hp.head_type}, feat_source={hp.feat_source}, num_classes={hp.num_classes}")

    # ---- build backbone ----
    feat_source = str(hp.feat_source).lower().strip()
    if feat_source in ("sem_resnet", "sem_resnet50"):
        backbone, out_ch = _build_sem_resnet_backbone(
            arch=str(getattr(hp, "det_sem_backbone", "resnet50")),
            pretrained=bool(hp.sem_pretrained),
            return_layer=str(hp.det_sem_return_layer),
        )
    else:
        raise ValueError(
            f"[detection_utils] Unsupported feat_source='{hp.feat_source}'. "
            "Expected 'sem_resnet' (new) or 'sem_resnet50' (legacy)."
        )

    if hp.freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False

    head = str(hp.head_type).lower().strip()

    if head == "fastrnn":
        from models.detection.fastrnn_detector import FastRNNDetector

        model = FastRNNDetector(
            backbone=backbone,
            out_channels=int(out_ch),
            num_classes=int(hp.num_classes),
            head_hidden=int(hp.fastrnn_hidden),
            head_bidir=bool(hp.fastrnn_bidir),
            head_dropout=float(hp.fastrnn_dropout),
            size_divisible=int(hp.fastrnn_size_divisible),
            focal_alpha=float(hp.fastrnn_focal_alpha),
            focal_gamma=float(hp.fastrnn_focal_gamma),
            score_thresh=float(hp.fastrnn_score_thresh),
            nms_thresh=float(hp.fastrnn_nms_thresh),
            topk=int(hp.fastrnn_topk),
        ).to(dev)

    elif head == "fasterrcnn":
        model = _build_fasterrcnn_detector(
            backbone=backbone,
            num_classes=int(hp.num_classes),
            hp=hp,
        ).to(dev)

        # Si on a gardé hp.num_classes différent du ckpt (cas drop_predictor_if_mismatch),
        # on retire les poids de la tête du state pour éviter les size-mismatch.
        if auto_fix_num_classes and drop_predictor_if_mismatch:
            inferred_nc = _infer_num_classes_from_state(state, head_type="fasterrcnn")
            if inferred_nc is not None and int(inferred_nc) != int(hp.num_classes):
                state = _drop_fasterrcnn_predictor_keys(state)

    else:
        raise ValueError(f"[detection_utils] Unsupported head_type='{hp.head_type}'.")

    # ---- remap backbone keys to match model expected keys ----
    model_keys = list(model.state_dict().keys())
    state = _remap_backbone_keys_to_match_model(state, model_keys)

    # ---- load ----
    # NOTE: strict=False n'évite PAS les size mismatch. D'où l'auto-fix num_classes / drop predictor.
    missing, unexpected = model.load_state_dict(state, strict=bool(strict))

    if not strict:
        if missing:
            print(f"[detection_utils] missing keys: {len(missing)} (showing 20)")
            for k in missing[:20]:
                print("  -", k)
        if unexpected:
            print(f"[detection_utils] unexpected keys: {len(unexpected)} (showing 20)")
            for k in unexpected[:20]:
                print("  -", k)

    model.eval()
    return model, hp


# ---------------------------------------------------------------------
# Inference unified
# ---------------------------------------------------------------------
@torch.no_grad()
def infer_batch(
    model: torch.nn.Module,
    images: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]],
    dev: torch.device,
    *,
    score_thresh: Optional[float] = None,
    max_dets: Optional[int] = None,
) -> List[Dict[str, torch.Tensor]]:
    imgs = _to_device_images(images, dev)
    imgs = [_ensure_chw_float01(x) for x in imgs]

    outs = model(imgs)  # torchvision style list[dict]
    if not isinstance(outs, (list, tuple)) or (len(outs) > 0 and not isinstance(outs[0], dict)):
        raise RuntimeError(
            "[infer_batch] Expected list[dict(boxes, labels, scores)]. "
            f"Got type={type(outs)}."
        )

    out_cpu: List[Dict[str, torch.Tensor]] = []
    for o in outs:
        boxes = o.get("boxes", torch.empty((0, 4), device=dev)).detach()
        scores = o.get("scores", torch.empty((0,), device=dev)).detach()
        labels = o.get("labels", torch.empty((0,), device=dev, dtype=torch.long)).detach()

        if score_thresh is not None and scores.numel() > 0:
            keep = scores >= float(score_thresh)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

        if max_dets is not None and scores.numel() > int(max_dets):
            idx = torch.argsort(scores, descending=True)[: int(max_dets)]
            boxes = boxes[idx]
            scores = scores[idx]
            labels = labels[idx]

        out_cpu.append({"boxes": boxes.cpu(), "scores": scores.cpu(), "labels": labels.cpu()})

    return out_cpu


# ---------------------------------------------------------------------
# COCO mAP evaluation (COCOeval)
# ---------------------------------------------------------------------
def _xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([x1, y1, (x2 - x1).clamp(min=0), (y2 - y1).clamp(min=0)], dim=-1)


@torch.no_grad()
def evaluate_coco_map(
    model: torch.nn.Module,
    data_loader,
    dev: torch.device,
    *,
    score_thresh: float = 0.05,
    max_images: Optional[int] = None,
) -> Dict[str, float]:
    if not _try_import_pycocotools():
        raise ImportError("pycocotools required for COCOeval")

    from pycocotools.cocoeval import COCOeval

    ds = getattr(data_loader, "dataset", None)
    coco_gt = getattr(ds, "coco", None)
    if coco_gt is None:
        raise RuntimeError("data_loader.dataset has no .coco (expected CocoDetection-like dataset)")

    model.eval()
    results: List[Dict[str, Any]] = []
    seen = 0

    pbar = tqdm(data_loader, desc="[TEST] COCOeval", ncols=140, leave=False)
    for images, targets in pbar:
        outputs = infer_batch(model, images, dev, score_thresh=score_thresh)

        for out, tgt in zip(outputs, targets):
            if not (isinstance(tgt, dict) and "image_id" in tgt):
                raise RuntimeError("Target must contain 'image_id'.")

            v = tgt["image_id"]
            img_id = int(v.item() if torch.is_tensor(v) else v)

            boxes = out["boxes"]
            scores = out["scores"]
            labels = out["labels"]
            if boxes.numel() == 0:
                continue

            boxes_xywh = _xyxy_to_xywh(boxes)
            for i in range(boxes_xywh.shape[0]):
                results.append(
                    {
                        "image_id": img_id,
                        "category_id": int(labels[i].item()),
                        "bbox": [float(x) for x in boxes_xywh[i].tolist()],
                        "score": float(scores[i].item()),
                    }
                )

        seen += len(outputs)
        if max_images is not None and seen >= int(max_images):
            break

    if not results:
        return {k: 0.0 for k in ["AP","AP50","AP75","APs","APm","APl","AR1","AR10","AR100"]}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    s = coco_eval.stats
    return {
        "AP": float(s[0]),
        "AP50": float(s[1]),
        "AP75": float(s[2]),
        "APs": float(s[3]),
        "APm": float(s[4]),
        "APl": float(s[5]),
        "AR1": float(s[6]),
        "AR10": float(s[7]),
        "AR100": float(s[8]),
    }


# ---------------------------------------------------------------------
# Camera demo (optional)
# ---------------------------------------------------------------------
@torch.no_grad()
def run_camera_demo(
    model: torch.nn.Module,
    dev: torch.device,
    *,
    class_names: Optional[Dict[int, str]] = None,
    cam_id: int = 0,
    score_thresh: float = 0.3,
    resize_hw: Optional[Tuple[int, int]] = None,
    max_dets: int = 200,
) -> None:
    cv2 = _try_import_cv2()
    if cv2 is None:
        raise ImportError("OpenCV required (pip install opencv-python)")

    cap = cv2.VideoCapture(int(cam_id))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera id={cam_id}")

    print("🎥 Camera demo: press 'q' to quit")
    model.eval()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize_hw is not None:
            h, w = resize_hw
            rgb = cv2.resize(rgb, (int(w), int(h)))

        img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        out = infer_batch(
            model,
            images=[img],
            dev=dev,
            score_thresh=float(score_thresh),
            max_dets=int(max_dets),
        )[0]

        disp = frame.copy()
        boxes = out["boxes"].numpy()
        scores = out["scores"].numpy()
        labels = out["labels"].numpy()

        for (x1, y1, x2, y2), sc, lab in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            name = class_names.get(int(lab), str(int(lab))) if class_names else str(int(lab))
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                disp,
                f"{name}:{sc:.2f}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow("Detection", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()