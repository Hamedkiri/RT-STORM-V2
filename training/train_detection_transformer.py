#!/usr/bin/env python3
# file: training/train_detection_transformer.py
# -*- coding: utf-8 -*-

"""
Detection TRAINING only (no eval/camera here).

Called from train_style_disentangle.py when --mode detect_transformer.

Supported heads via --det_head:
  - fasterrcnn : torchvision Faster R-CNN (custom backbone = ResNet50 layer{2,3,4} via IntermediateLayerGetter)
  - fastrnn    : custom FastRNNDetector (models/detection/fastrnn_detector.py) using same backbone output dict {"0": feat}

✅ Robust handling of empty targets:
  - If your data.py filters empty images at dataset init (recommended), FasterRCNN should never see empty boxes.
  - We still keep an additional safety filter per-batch (optional) to avoid crashes if something slips through
    (missing file, corrupted ann, broken box, etc).

Checkpoints:
  - save_dir/detector_last.pth
  - save_dir/detector_epoch_XXXX.pth (periodic)
  - save_dir/hparams_detection.json (exact hparams used)

These artifacts are designed to be reloaded by testsFile/detectionUtils.py:
  - build_detector_from_checkpoint(...), or
  - build_detector_from_checkpoint(ckpt_path, hparams_json=..., auto_hparams_json=True, ...)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from data import build_detection_dataloader


# --------------------------------------------------------------------------------------
# Utils
# --------------------------------------------------------------------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _parse_save_freq_epoch_only(save_freq_str: Optional[str]) -> Tuple[str, Optional[int]]:
    """
    - 'none'       -> ('none', None)
    - 'epoch'      -> ('epoch', 1)
    - 'epoch:5'    -> ('epoch', 5)
    - '5'          -> ('epoch', 5)
    """
    if save_freq_str is None:
        return "none", None
    sf = str(save_freq_str).strip().lower()
    if sf in ("", "none"):
        return "none", None

    if sf.startswith("epoch"):
        parts = sf.split(":", 1)
        if len(parts) == 2:
            try:
                n = int(parts[1])
                return "epoch", max(1, n)
            except ValueError:
                return "epoch", 1
        return "epoch", 1

    try:
        n = int(sf)
        return "epoch", max(1, n)
    except ValueError:
        return "none", None


def _strip_module_prefix_if_needed(state: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return state
    if not any(k.startswith("module.") for k in state.keys()):
        return state
    return {k[len("module.") :]: v for k, v in state.items()}


def _is_valid_boxes_tensor(boxes: Any) -> bool:
    """
    FasterRCNN expects boxes: FloatTensor [N,4], N>=1 during training.
    """
    if not torch.is_tensor(boxes):
        return False
    if boxes.ndim != 2:
        return False
    if boxes.shape[-1] != 4:
        return False
    if boxes.numel() == 0:
        return False
    return True


def _filter_empty_targets_batch(
    images: torch.Tensor,
    targets: List[Dict[str, Any]],
    dev: torch.device,
    debug: bool = False,
) -> Tuple[List[torch.Tensor], List[Dict[str, Any]], Dict[str, int]]:
    """
    images: (B,C,H,W)
    targets: list of dicts with at least "boxes" and "labels"
    Return: list_images(CHW), list_targets, stats dict
    """
    B = images.shape[0]
    imgs_list: List[torch.Tensor] = []
    tgts_list: List[Dict[str, Any]] = []

    total = int(B)
    used = 0
    ignored = 0

    for i in range(B):
        t = targets[i]
        boxes = t.get("boxes", None)
        labels = t.get("labels", None)

        ok = _is_valid_boxes_tensor(boxes) and (torch.is_tensor(labels) and labels.numel() > 0)

        if not ok:
            ignored += 1
            if debug:
                shape = getattr(boxes, "shape", None)
                print(f"[DEBUG DET] ignore batch img[{i}] invalid target (boxes_shape={shape})")
            continue

        imgs_list.append(images[i].to(dev))
        tgts_list.append({k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in t.items()})
        used += 1

    return imgs_list, tgts_list, {"total": total, "used": used, "ignored": ignored}


# --------------------------------------------------------------------------------------
# ✅ NEW: robust loader for SSL ResNet50 backbone checkpoints
# --------------------------------------------------------------------------------------
def _load_resnet50_backbone_weights(
    *,
    resnet: torch.nn.Module,
    ckpt_path: str,
    device: torch.device = torch.device("cpu"),
    strict: bool = True,
    verbose: bool = True,
) -> None:
    """
    Load SSL/MoCo-style ResNet50 backbone weights into a torchvision ResNet50
    WHILE KEEPING strict=True.

    ✅ Key idea to keep strict=True:
      - We build a *full* state_dict with ALL expected keys from `resnet.state_dict()`,
        then we overwrite only the matching backbone keys coming from the checkpoint.
      - This means `fc.weight` / `fc.bias` remain present (kept from the model init),
        so strict=True will not complain about missing classifier head.

    Supports common prefix patterns:
      - 'module.' / 'model.' / 'state_dict' nesting
      - 'backbone_q.' (MoCo queue encoder)
      - 'encoder_q.' / 'encoder.' / 'backbone.' etc.
      - ResNet stem mapping:
          checkpoint: stem.0/1/...  -> torchvision: conv1/bn1/...
    """

    import os
    import torch

    if ckpt_path is None or str(ckpt_path).strip() == "":
        raise ValueError("ckpt_path is empty")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    # -------------------------
    # 1) Extract raw state dict
    # -------------------------
    state = None
    if isinstance(ckpt, dict):
        # Common patterns
        for k in ["state_dict", "model", "model_state", "net", "encoder", "backbone"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]
                break
        if state is None:
            # maybe it's already a state_dict-like dict
            state = ckpt
    else:
        raise RuntimeError(f"Unsupported checkpoint type: {type(ckpt)}")

    # -----------------------------------------
    # 2) Strip DDP prefixes like 'module.'
    # -----------------------------------------
    def strip_prefix(sd: dict, pref: str) -> dict:
        out = {}
        for k, v in sd.items():
            if k.startswith(pref):
                out[k[len(pref):]] = v
            else:
                out[k] = v
        return out

    # strip "module." repeatedly (sometimes nested "module.module.")
    for _ in range(3):
        if any(k.startswith("module.") for k in state.keys()):
            state = strip_prefix(state, "module.")

    # ----------------------------------------------------------
    # 3) Remap checkpoint keys -> torchvision ResNet50 keys
    # ----------------------------------------------------------
    def remap_key(k: str) -> str:
        """
        Convert common SSL/MoCo naming to torchvision resnet naming.
        """
        # Prefer query encoder if both exist
        # (We don't attempt to merge q/k; we just take what is in sd)
        prefixes = [
            "backbone_q.",
            "encoder_q.",
            "encoder.",
            "backbone.",
            "model.",
            "net.",
        ]
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
                break

        # MoCo-v3 / custom stem naming
        # checkpoint: stem.0.weight -> torchvision: conv1.weight
        # checkpoint: stem.1.*      -> torchvision: bn1.*
        # Some codebases also have stem.2=relu, stem.3=maxpool (no params)
        if k.startswith("stem.0."):
            k = "conv1." + k[len("stem.0."):]
        elif k.startswith("stem.1."):
            k = "bn1." + k[len("stem.1."):]
        # stem.2 / stem.3 have no params; nothing to map

        # Some SSL checkpoints store fc as "head" or "classifier"
        # We intentionally DO NOT map those to resnet.fc to avoid overwriting;
        # we keep resnet.fc init weights to satisfy strict=True.
        if k.startswith("head.") or k.startswith("classifier."):
            # map to something that won't match; it will be dropped later
            k = "__DROP__." + k

        return k

    remapped = {}
    for k, v in state.items():
        rk = remap_key(k)
        if rk.startswith("__DROP__."):
            continue
        remapped[rk] = v

    # -------------------------------------------------------------------
    # 4) Build a FULL state dict (keeps fc.* keys) then overwrite matches
    # -------------------------------------------------------------------
    target_sd = resnet.state_dict()
    target_keys = set(target_sd.keys())

    # Only copy tensors that exist in target AND have same shape
    loaded_keys = []
    shape_mismatch = []
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

    # Optionally: you can also explicitly keep fc from init (already kept).
    # target_sd["fc.weight"] and target_sd["fc.bias"] remain untouched.

    # -------------------------------------------------------------------
    # 5) strict=True load (will pass because ALL keys exist in target_sd)
    # -------------------------------------------------------------------
    # With strict=True:
    #  - missing keys: none (we pass full dict)
    #  - unexpected keys: none (we pass only target keys)
    resnet.load_state_dict(target_sd, strict=strict)

    if verbose:
        print(f"[DET][SEM] Loaded ckpt: {ckpt_path}")
        print(f"[DET][SEM] strict={strict} | loaded_backbone_keys={len(loaded_keys)}")
        if len(loaded_keys) > 0:
            print(f"[DET][SEM] example loaded keys: {loaded_keys[:8]}")
        if shape_mismatch:
            print(f"[DET][SEM] ⚠️ shape mismatches skipped: {len(shape_mismatch)}")
            for k, s_src, s_tgt in shape_mismatch[:10]:
                print(f"  - {k}: ckpt{s_src} != model{s_tgt}")




# --------------------------------------------------------------------------------------
# HParams (align with testsFile/detectionUtils.py fields)
# --------------------------------------------------------------------------------------
@dataclass
class DetectorHParams:
    # Core
    head_type: str = "fastrnn"         # "fastrnn" | "fasterrcnn"
    num_classes: int = 81              # inclut bg=0
    feat_source: str = "sem_resnet50"
    freeze_backbone: bool = True
    img_h: int = 256
    img_w: int = 256
    sem_pretrained: bool = True
    det_sem_return_layer: str = "layer4"

    # ✅ NEW: path to SSL-pretrained backbone checkpoint
    sem_pretrained_path: str = ""      # if non-empty, overrides ImageNet init for ResNet50 backbone
    sem_pretrained_strict: bool = False
    sem_pretrained_verbose: bool = True

    # Data / robustness
    det_drop_empty: bool = True        # should match data.py dataset filtering (recommended)
    det_filter_batch_safety: bool = True  # extra safety filter inside training loop

    # Optim
    det_epochs: int = 20
    det_lr_head: float = 1e-4
    det_lr_backbone: float = 1e-5
    det_weight_decay: float = 1e-4
    det_grad_clip: float = 0.0

    # FastRNN
    fastrnn_hidden: int = 256
    fastrnn_bidir: bool = True
    fastrnn_dropout: float = 0.0
    fastrnn_focal_alpha: float = 0.25
    fastrnn_focal_gamma: float = 2.0
    fastrnn_score_thresh: float = 0.05
    fastrnn_nms_thresh: float = 0.5
    fastrnn_topk: int = 1000
    fastrnn_size_divisible: int = 32

    # FasterRCNN (RPN/box postprocess)
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
    def from_opt(opt, num_classes: int) -> "DetectorHParams":
        hp = DetectorHParams()
        hp.head_type = str(getattr(opt, "det_head", "fasterrcnn")).lower().strip()
        hp.num_classes = int(getattr(opt, "det_num_classes", num_classes))
        hp.feat_source = str(getattr(opt, "det_feat_source", "sem_resnet50")).lower().strip()
        hp.freeze_backbone = bool(int(getattr(opt, "det_freeze_backbone", 0)))
        hp.img_h = int(getattr(opt, "det_img_h", getattr(opt, "crop_size", 256)))
        hp.img_w = int(getattr(opt, "det_img_w", getattr(opt, "crop_size", 256)))
        hp.sem_pretrained = bool(int(getattr(opt, "sem_pretrained", 1)))
        hp.det_sem_return_layer = str(getattr(opt, "det_sem_return_layer", "layer4")).lower().strip()

        # ✅ NEW
        hp.sem_pretrained_path = str(getattr(opt, "sem_pretrained_path", "") or "")
        hp.sem_pretrained_strict = bool(int(getattr(opt, "sem_pretrained_strict", 0)))
        hp.sem_pretrained_verbose = bool(int(getattr(opt, "sem_pretrained_verbose", 1)))

        # robustness flags (consistent with data.py)
        hp.det_drop_empty = bool(int(getattr(opt, "det_drop_empty", 1)))
        hp.det_filter_batch_safety = bool(int(getattr(opt, "det_filter_batch_safety", 1)))

        hp.det_epochs = int(getattr(opt, "det_epochs", 20))
        hp.det_lr_head = float(getattr(opt, "det_lr_head", 1e-4))
        hp.det_lr_backbone = float(getattr(opt, "det_lr_backbone", 1e-5))
        hp.det_weight_decay = float(getattr(opt, "det_weight_decay", 1e-4))
        hp.det_grad_clip = float(getattr(opt, "det_grad_clip", 0.0))

        hp.fastrnn_hidden = int(getattr(opt, "fastrnn_hidden", 256))
        hp.fastrnn_dropout = float(getattr(opt, "fastrnn_dropout", 0.0))
        hp.fastrnn_bidir = bool(getattr(opt, "fastrnn_bidir", True))
        hp.fastrnn_focal_alpha = float(getattr(opt, "fastrnn_focal_alpha", 0.25))
        hp.fastrnn_focal_gamma = float(getattr(opt, "fastrnn_focal_gamma", 2.0))
        hp.fastrnn_score_thresh = float(getattr(opt, "fastrnn_score_thresh", 0.05))
        hp.fastrnn_nms_thresh = float(getattr(opt, "fastrnn_nms_thresh", 0.5))
        hp.fastrnn_topk = int(getattr(opt, "fastrnn_topk", 1000))
        hp.fastrnn_size_divisible = int(getattr(opt, "fastrnn_size_divisible", 32))

        return hp


def _write_hparams_json(save_dir: Path, hp: DetectorHParams) -> Path:
    p = save_dir / "hparams_detection.json"
    p.write_text(json.dumps(asdict(hp), indent=2, sort_keys=True))
    return p


# --------------------------------------------------------------------------------------
# Backbone builders (same logic as testsFile/detectionUtils.py)
# --------------------------------------------------------------------------------------
def _build_sem_resnet50_backbone(
    *,
    pretrained: bool,
    return_layer: str = "layer4",
    pretrained_path: str = "",
    strict: bool = False,
    verbose: bool = True,
) -> Tuple[torch.nn.Module, int]:
    import torchvision
    from torchvision.models._utils import IntermediateLayerGetter

    weights = None
    # If a custom pretrained_path is provided, we don't need ImageNet weights.
    if pretrained and not str(pretrained_path).strip():
        try:
            weights = torchvision.models.ResNet50_Weights.DEFAULT
        except Exception:
            weights = None

    m = torchvision.models.resnet50(weights=weights)

    # ✅ NEW: override/initialize from SSL checkpoint if provided
    if str(pretrained_path).strip():
        _load_resnet50_backbone_weights(
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


def _build_fasterrcnn_detector(backbone: torch.nn.Module, num_classes: int, hp: DetectorHParams) -> torch.nn.Module:
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.rpn import AnchorGenerator

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
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


def _build_detector(opt, dev: torch.device, hp: DetectorHParams) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Returns (det_model, backbone_module_for_freeze)
    """
    if hp.feat_source != "sem_resnet50":
        raise ValueError(
            f"[train_detection] Unsupported det_feat_source='{hp.feat_source}' in this simplified script. "
            "Use sem_resnet50 for now."
        )

    backbone, out_ch = _build_sem_resnet50_backbone(
        pretrained=bool(hp.sem_pretrained),
        return_layer=str(hp.det_sem_return_layer),
        pretrained_path=str(hp.sem_pretrained_path),
        strict=bool(hp.sem_pretrained_strict),
        verbose=bool(hp.sem_pretrained_verbose),
    )
    backbone = backbone.to(dev)

    if hp.head_type == "fasterrcnn":
        det_model = _build_fasterrcnn_detector(backbone=backbone, num_classes=hp.num_classes, hp=hp).to(dev)
        backbone_module = det_model.backbone  # for freeze toggles

    elif hp.head_type == "fastrnn":
        from models.detection.fastrnn_detector import FastRNNDetector

        det_model = FastRNNDetector(
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
        backbone_module = backbone  # FastRNNDetector uses our backbone directly

    else:
        raise ValueError(f"[train_detection] Unknown det_head='{hp.head_type}'. Use fasterrcnn or fastrnn.")

    # Freeze backbone if requested
    if hp.freeze_backbone:
        for p in backbone_module.parameters():
            p.requires_grad = False

    return det_model, backbone_module


# --------------------------------------------------------------------------------------
# Optim groups (head vs backbone)
# --------------------------------------------------------------------------------------
def _build_optimizer(det_model: torch.nn.Module, hp: DetectorHParams) -> torch.optim.Optimizer:
    head_params: List[torch.nn.Parameter] = []
    backbone_params: List[torch.nn.Parameter] = []

    for name, p in det_model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(p)
        else:
            head_params.append(p)

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": float(hp.det_lr_head)})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": float(hp.det_lr_backbone)})

    if not param_groups:
        raise RuntimeError("No trainable parameters found (everything frozen?).")

    return torch.optim.AdamW(param_groups, weight_decay=float(hp.det_weight_decay))


# --------------------------------------------------------------------------------------
# Checkpoint IO (compatible with testsFile/detectionUtils.py)
# --------------------------------------------------------------------------------------
def _save_ckpt(
    path: Path,
    *,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    hp: DetectorHParams,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "head_type": str(hp.head_type),
        "hparams": asdict(hp),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "time": time.time(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def _load_ckpt(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise RuntimeError(f"Invalid detector checkpoint: {path}")
    return ckpt


# --------------------------------------------------------------------------------------
# Training entry
# --------------------------------------------------------------------------------------
def train_detection_transformer(opt, dev: torch.device):
    """
    Training only.
    Evaluation/camera are handled elsewhere in your codebase (testsFile/*).
    """
    save_dir = Path(opt.save_dir)
    _ensure_dir(save_dir)

    # Data (data.py prints counts filtered in dataset init)
    train_loader, _val_loader, num_classes_dl = build_detection_dataloader(opt)

    # HParams
    hp = DetectorHParams.from_opt(opt, num_classes=num_classes_dl)
    _write_hparams_json(save_dir, hp)
    print(f"[DET] hparams -> {save_dir / 'hparams_detection.json'}")
    print(
        f"[DET] head={hp.head_type} | num_classes={hp.num_classes} | "
        f"freeze_backbone={hp.freeze_backbone} | drop_empty={hp.det_drop_empty} | "
        f"batch_safety={hp.det_filter_batch_safety}"
    )
    if str(hp.sem_pretrained_path).strip():
        print(
            f"[DET] sem_pretrained_path='{hp.sem_pretrained_path}' "
            f"(strict={hp.sem_pretrained_strict}, verbose={hp.sem_pretrained_verbose})"
        )

    # Model
    det_model, _backbone_module = _build_detector(opt, dev=dev, hp=hp)

    # Optim
    optimizer = _build_optimizer(det_model, hp)

    # Resume (optional)
    start_epoch = 0
    global_step = 0
    det_resume = getattr(opt, "det_resume", "")
    if isinstance(det_resume, str) and det_resume.strip():
        ckpt = _load_ckpt(det_resume, map_location="cpu")
        sd = _strip_module_prefix_if_needed(ckpt["model"])
        det_model.load_state_dict(sd, strict=True)
        if ckpt.get("optimizer", None) is not None:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"[DET] ⚠️ optimizer state not loaded: {e}")
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        print(f"[DET] resumed from {det_resume} (start_epoch={start_epoch}, global_step={global_step})")

    # Save frequency
    save_freq_mode, save_freq_interval = _parse_save_freq_epoch_only(getattr(opt, "save_freq", "epoch"))
    epoch_ckpt_interval = getattr(opt, "epoch_ckpt_interval", None)
    if save_freq_mode == "epoch" and epoch_ckpt_interval is not None:
        try:
            save_freq_interval = max(1, int(epoch_ckpt_interval))
        except Exception:
            pass

    # Training loop
    total_ignored_batches = 0          # batches entirely skipped due to empty targets after filtering
    total_ignored_samples = 0          # samples removed by safety filter during training
    total_seen_samples = 0

    det_model.train()
    for epoch in range(start_epoch, int(hp.det_epochs)):
        det_model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"[DET-{hp.head_type}] epoch {epoch+1}/{hp.det_epochs}",
            ncols=140,
            leave=False,
        )

        for imgs, targets in pbar:
            total_seen_samples += int(imgs.shape[0])

            optimizer.zero_grad(set_to_none=True)

            if hp.head_type == "fasterrcnn":
                # FasterRCNN expects list[Tensor] and list[Dict]
                if hp.det_filter_batch_safety:
                    valid_imgs, valid_tgts, st = _filter_empty_targets_batch(
                        images=imgs,
                        targets=targets,
                        dev=dev,
                        debug=bool(getattr(opt, "debug_detection", False)),
                    )
                    total_ignored_samples += int(st["ignored"])
                    if st["used"] == 0:
                        total_ignored_batches += 1
                        continue
                    imgs_list = valid_imgs
                    targets_list = valid_tgts
                else:
                    imgs = imgs.to(dev)
                    imgs_list = [im for im in imgs]
                    targets_list = [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]

                loss_dict = det_model(imgs_list, targets_list)
                loss = sum(loss_dict.values())

            elif hp.head_type == "fastrnn":
                if hp.det_filter_batch_safety:
                    valid_imgs, valid_tgts, st = _filter_empty_targets_batch(
                        images=imgs,
                        targets=targets,
                        dev=dev,
                        debug=bool(getattr(opt, "debug_detection", False)),
                    )
                    total_ignored_samples += int(st["ignored"])
                    if st["used"] == 0:
                        total_ignored_batches += 1
                        continue
                    loss_dict = det_model(valid_imgs, valid_tgts)
                else:
                    imgs = imgs.to(dev)
                    targets_list = [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
                    loss_dict = det_model([im for im in imgs], targets_list)

                loss = sum(loss_dict.values())

            else:
                raise ValueError(f"Unexpected head_type={hp.head_type}")

            loss.backward()
            if float(hp.det_grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in det_model.parameters() if p.requires_grad],
                    max_norm=float(hp.det_grad_clip),
                )
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1
            global_step += 1
            pbar.set_postfix(loss=f"{total_loss / max(1, n_batches):.4f}")

        train_loss = total_loss / max(1, n_batches)
        print(
            f"[DET] epoch {epoch+1}/{hp.det_epochs} | train_loss={train_loss:.4f} | "
            f"ignored_samples_total={total_ignored_samples} | ignored_batches_total={total_ignored_batches}"
        )

        # Save last
        _save_ckpt(
            save_dir / "detector_last.pth",
            epoch=epoch,
            global_step=global_step,
            model=det_model,
            optimizer=optimizer,
            hp=hp,
            extra={
                "train_loss": float(train_loss),
                "ignored_samples_total": int(total_ignored_samples),
                "ignored_batches_total": int(total_ignored_batches),
                "seen_samples_total": int(total_seen_samples),
            },
        )

        # Save periodic
        if save_freq_mode == "epoch" and save_freq_interval is not None and ((epoch + 1) % int(save_freq_interval) == 0):
            _save_ckpt(
                save_dir / f"detector_epoch_{epoch+1:04d}.pth",
                epoch=epoch,
                global_step=global_step,
                model=det_model,
                optimizer=optimizer,
                hp=hp,
                extra={
                    "train_loss": float(train_loss),
                    "ignored_samples_total": int(total_ignored_samples),
                    "ignored_batches_total": int(total_ignored_batches),
                    "seen_samples_total": int(total_seen_samples),
                },
            )

        # Tiny json log (optional)
        (save_dir / "det_train_log.json").write_text(
            json.dumps(
                {
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "train_loss": float(train_loss),
                    "hparams": asdict(hp),
                    "ignored_samples_total": int(total_ignored_samples),
                    "ignored_batches_total": int(total_ignored_batches),
                    "seen_samples_total": int(total_seen_samples),
                },
                indent=2,
                sort_keys=True,
            )
        )

    print(
        "[DET] ✅ training finished. "
        f"seen_samples_total={total_seen_samples} | ignored_samples_total={total_ignored_samples} | "
        f"ignored_batches_total={total_ignored_batches}"
    )
    return None
