#!/usr/bin/env python3
# file: training/train_detection_transformer.py
# -*- coding: utf-8 -*-

"""
Detection TRAINING only (no eval/camera here).

Called from train_style_disentangle.py when --mode detect_transformer.

Supported heads via --det_head:
  - fastrnn    : custom FastRNNDetector (models/detection/fastrnn_detector.py)  ✅ DEFAULT (comme main_moco_modified_classifier.py)
  - fasterrcnn : torchvision Faster R-CNN

✅ Default backbone return layer:
  - --det_sem_return_layer = layer4 ✅ DEFAULT (mêmes couches par défaut que ton script main_moco)
    (tu peux toujours passer layer2/layer3/layer4 via l’option)

✅ Main-moco-like training:
  - data.py returns (imgs, targets) with imgs = list[Tensor] by default (variable sizes)
  - forward: loss_dict = model(imgs, targets)
  - move imgs/targets to device (list-wise)
  - optional warmup + StepLR
  - dataset can drop empty images; optional per-batch safety filter remains

Checkpoints:
  - save_dir/detector_last.pth
  - save_dir/detector_epoch_XXXX.pth (periodic)
  - save_dir/hparams_detection.json
  - save_dir/det_train_log.json
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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


def _to_device_target(t: Dict[str, Any], dev: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in t.items():
        out[k] = v.to(dev) if torch.is_tensor(v) else v
    return out


def _to_device_images_targets(
    imgs: Union[List[torch.Tensor], torch.Tensor],
    targets: List[Dict[str, Any]],
    dev: torch.device,
) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
    """
    Accepts:
      - imgs as list[Tensor] (main_moco-like default)
      - imgs as Tensor (B,C,H,W) if user enabled det_stack_batch

    Returns:
      - imgs_list: list[Tensor] on device
      - targets_list: list[dict] on device
    """
    if torch.is_tensor(imgs):
        imgs_list = [im.to(dev) for im in imgs]  # split along batch
    else:
        imgs_list = [im.to(dev) for im in imgs]

    targets_list = [_to_device_target(t, dev) for t in targets]
    return imgs_list, targets_list


def _is_valid_boxes_tensor(boxes: Any) -> bool:
    """
    FasterRCNN training expects boxes: FloatTensor [N,4], N>=1 for that image.
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


def _filter_empty_targets(
    imgs_list: List[torch.Tensor],
    targets_list: List[Dict[str, Any]],
    *,
    num_classes: Optional[int] = None,
    debug: bool = False,
) -> Tuple[List[torch.Tensor], List[Dict[str, Any]], Dict[str, int]]:
    """
    Per-sample safety filter. Keeps only samples with valid boxes & labels.
    """
    total = len(imgs_list)
    used = 0
    ignored = 0
    out_imgs: List[torch.Tensor] = []
    out_tgts: List[Dict[str, Any]] = []

    for i in range(total):
        t = targets_list[i]
        boxes = t.get("boxes", None)
        labels = t.get("labels", None)

        ok = _is_valid_boxes_tensor(boxes) and (torch.is_tensor(labels) and labels.numel() > 0)
        if ok and (num_classes is not None) and torch.is_tensor(labels) and labels.numel() > 0:
            # torchvision FasterRCNN expects labels in [1, num_classes-1]
            # (0 reserved for background, not used in targets)
            lab_min = int(labels.min().item())
            lab_max = int(labels.max().item())
            if lab_min < 1 or lab_max >= int(num_classes):
                ok = False
                if debug:
                    print(
                        f"[DEBUG DET] ignore sample[{i}] labels out of range: "
                        f"min={lab_min} max={lab_max} (num_classes={int(num_classes)})"
                    )

        if not ok:
            ignored += 1
            if debug:
                shape = getattr(boxes, "shape", None)
                print(f"[DEBUG DET] ignore sample[{i}] invalid target (boxes_shape={shape})")
            continue

        out_imgs.append(imgs_list[i])
        out_tgts.append(t)
        used += 1

    return out_imgs, out_tgts, {"total": int(total), "used": int(used), "ignored": int(ignored)}


# --------------------------------------------------------------------------------------
# ✅ Robust loader for SSL ResNet50 backbone checkpoints (keeps strict=True possible)
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
    Load SSL/MoCo-style ResNet50 backbone weights into a torchvision ResNet50.

    ✅ Key idea to keep strict=True:
      - Build a *full* target state_dict from `resnet.state_dict()` (includes fc.*),
        overwrite only matching keys from ckpt, then load with strict=True.
    """
    import os

    if ckpt_path is None or str(ckpt_path).strip() == "":
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
        prefixes = ["backbone_q.", "encoder_q.", "encoder.", "backbone.", "model.", "net."]
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
                break

        if k.startswith("stem.0."):
            k = "conv1." + k[len("stem.0."):]
        elif k.startswith("stem.1."):
            k = "bn1." + k[len("stem.1."):]
        if k.startswith("head.") or k.startswith("classifier."):
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
        print(f"[DET][SEM] Loaded ckpt: {ckpt_path}")
        print(f"[DET][SEM] strict={strict} | loaded_backbone_keys={len(loaded_keys)}")
        if loaded_keys:
            print(f"[DET][SEM] example loaded keys: {loaded_keys[:8]}")
        if shape_mismatch:
            print(f"[DET][SEM] ⚠️ shape mismatches skipped: {len(shape_mismatch)}")
            for k, s_src, s_tgt in shape_mismatch[:10]:
                print(f"  - {k}: ckpt{s_src} != model{s_tgt}")


# Backward-compat alias
def _load_resnet50_backbone_weights(*args, **kwargs):
    return _load_resnet_backbone_weights(*args, **kwargs)


# --------------------------------------------------------------------------------------
# HParams
# --------------------------------------------------------------------------------------
@dataclass
class DetectorHParams:
    # Core
    head_type: str = "fastrnn"         # ✅ DEFAULT comme main_moco_modified_classifier.py
    num_classes: int = 81              # incl bg=0
    feat_source: str = "sem_resnet50"
    freeze_backbone: bool = False
    sem_pretrained: bool = True
    det_sem_return_layer: str = "layer4"  # ✅ DEFAULT (mêmes couches par défaut)
    det_sem_backbone: str = "resnet50"    # ✅ resnet50|resnet101|resnet152

    # SSL-pretrained backbone checkpoint (optional)
    sem_pretrained_path: str = ""
    sem_pretrained_strict: bool = False
    sem_pretrained_verbose: bool = True

    # Data / robustness
    det_drop_empty: bool = True
    det_filter_batch_safety: bool = True

    # Optim / schedule
    det_epochs: int = 20
    det_optimizer: str = "sgd"         # "sgd" | "adamw"
    det_lr_head: float = 5e-3
    det_lr_backbone: float = 5e-4
    det_weight_decay: float = 1e-4
    det_sgd_momentum: float = 0.9
    det_grad_clip: float = 0.0

    # LR scheduler (StepLR)
    det_lr_step_size: int = 8
    det_lr_gamma: float = 0.1

    # Warmup (iterations)
    det_warmup_iters: int = 500
    det_warmup_factor: float = 1.0 / 1000.0

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

    # FasterRCNN extras
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

        # ✅ IMPORTANT: default det_head = "fastrnn" (pas fasterrcnn)
        hp.head_type = str(getattr(opt, "det_head", "fastrnn")).lower().strip()

        hp.num_classes = int(getattr(opt, "det_num_classes", num_classes))
        hp.feat_source = str(getattr(opt, "det_feat_source", "sem_resnet50")).lower().strip()
        hp.freeze_backbone = bool(int(getattr(opt, "det_freeze_backbone", 0)))
        hp.sem_pretrained = bool(int(getattr(opt, "sem_pretrained", 1)))

        # ✅ IMPORTANT: default return layer = layer4
        hp.det_sem_return_layer = str(getattr(opt, "det_sem_return_layer", "layer4")).lower().strip()
        hp.det_sem_backbone = str(getattr(opt, "det_sem_backbone", "resnet50")).lower().strip()

        hp.sem_pretrained_path = str(getattr(opt, "sem_pretrained_path", "") or "")
        hp.sem_pretrained_strict = bool(int(getattr(opt, "sem_pretrained_strict", 0)))
        hp.sem_pretrained_verbose = bool(int(getattr(opt, "sem_pretrained_verbose", 1)))

        hp.det_drop_empty = bool(int(getattr(opt, "det_drop_empty", 1)))
        hp.det_filter_batch_safety = bool(int(getattr(opt, "det_filter_batch_safety", 1)))

        hp.det_epochs = int(getattr(opt, "det_epochs", 20))
        hp.det_optimizer = str(getattr(opt, "det_optimizer", "sgd")).lower().strip()

        hp.det_lr_head = float(getattr(opt, "det_lr_head", 5e-3))
        hp.det_lr_backbone = float(getattr(opt, "det_lr_backbone", 5e-4))
        hp.det_weight_decay = float(getattr(opt, "det_weight_decay", 1e-4))
        hp.det_sgd_momentum = float(getattr(opt, "det_sgd_momentum", 0.9))
        hp.det_grad_clip = float(getattr(opt, "det_grad_clip", 0.0))

        hp.det_lr_step_size = int(getattr(opt, "det_lr_step_size", 8))
        hp.det_lr_gamma = float(getattr(opt, "det_lr_gamma", 0.1))

        hp.det_warmup_iters = int(getattr(opt, "det_warmup_iters", 500))
        hp.det_warmup_factor = float(getattr(opt, "det_warmup_factor", 1.0 / 1000.0))

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
# Backbone / detector builders
# --------------------------------------------------------------------------------------
def _build_sem_resnet_backbone(
    *,
    pretrained: bool,
    arch: str = "resnet50",
    return_layer: str = "layer4",
    pretrained_path: str = "",
    strict: bool = False,
    verbose: bool = True,
) -> Tuple[torch.nn.Module, int]:
    import torchvision
    from torchvision.models._utils import IntermediateLayerGetter
    arch = str(arch).lower().strip()
    if arch not in ("resnet50", "resnet101", "resnet152"):
        raise ValueError(f"det_sem_backbone must be one of resnet50/resnet101/resnet152, got {arch}")

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

    fn = getattr(torchvision.models, arch)
    m = fn(weights=weights)

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
        raise ValueError(f"[train_detection] Unsupported det_feat_source='{hp.feat_source}'. Use sem_resnet50.")

    backbone, out_ch = _build_sem_resnet_backbone(
        pretrained=bool(hp.sem_pretrained),
        arch=str(hp.det_sem_backbone),
        return_layer=str(hp.det_sem_return_layer),
        pretrained_path=str(hp.sem_pretrained_path),
        strict=bool(hp.sem_pretrained_strict),
        verbose=bool(hp.sem_pretrained_verbose),
    )
    backbone = backbone.to(dev)

    if hp.head_type == "fastrnn":
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
        backbone_module = backbone

    elif hp.head_type == "fasterrcnn":
        det_model = _build_fasterrcnn_detector(backbone=backbone, num_classes=hp.num_classes, hp=hp).to(dev)
        backbone_module = det_model.backbone

    else:
        raise ValueError(f"[train_detection] Unknown det_head='{hp.head_type}'. Use fastrnn or fasterrcnn.")

    if hp.freeze_backbone:
        for p in backbone_module.parameters():
            p.requires_grad = False

    return det_model, backbone_module


# --------------------------------------------------------------------------------------
# Optim + sched
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

    if hp.det_optimizer == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=float(hp.det_weight_decay))

    return torch.optim.SGD(
        param_groups,
        lr=float(hp.det_lr_head),
        momentum=float(hp.det_sgd_momentum),
        weight_decay=float(hp.det_weight_decay),
        nesterov=True,
    )


class _WarmupLinearLR:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_iters: int, warmup_factor: float):
        self.optimizer = optimizer
        self.warmup_iters = max(0, int(warmup_iters))
        self.warmup_factor = float(warmup_factor)
        self._it = 0
        self.base_lrs = [pg.get("lr", 0.0) for pg in optimizer.param_groups]

    def step(self) -> None:
        if self.warmup_iters <= 0:
            return
        self._it += 1
        if self._it > self.warmup_iters:
            return
        alpha = float(self._it) / float(self.warmup_iters)
        factor = self.warmup_factor + (1.0 - self.warmup_factor) * alpha
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = float(base_lr) * float(factor)

    def state_dict(self) -> Dict[str, Any]:
        return {"it": int(self._it), "warmup_iters": int(self.warmup_iters), "warmup_factor": float(self.warmup_factor)}

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self._it = int(d.get("it", 0))
        self.warmup_iters = int(d.get("warmup_iters", self.warmup_iters))
        self.warmup_factor = float(d.get("warmup_factor", self.warmup_factor))
        if self.warmup_iters > 0 and self._it > 0:
            alpha = min(1.0, float(self._it) / float(self.warmup_iters))
            factor = self.warmup_factor + (1.0 - self.warmup_factor) * alpha
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg["lr"] = float(base_lr) * float(factor)


def _build_epoch_scheduler(optimizer: torch.optim.Optimizer, hp: DetectorHParams):
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, int(hp.det_lr_step_size)),
        gamma=float(hp.det_lr_gamma),
    )


# --------------------------------------------------------------------------------------
# Checkpoint IO
# --------------------------------------------------------------------------------------
def _save_ckpt(
    path: Path,
    *,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    hp: DetectorHParams,
    scheduler: Optional[Any] = None,
    warmup: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "head_type": str(hp.head_type),
        "hparams": asdict(hp),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "warmup": warmup.state_dict() if warmup is not None else None,
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
    save_dir = Path(opt.save_dir)
    _ensure_dir(save_dir)

    # Data
    train_loader, _val_loader, num_classes_dl = build_detection_dataloader(opt)

    # HParams (possibly overridden by checkpoint)
    hp = DetectorHParams.from_opt(opt, num_classes=num_classes_dl)

    # Resume (⚠️ we must read ckpt *before* building the model to rebuild the correct ResNet depth)
    start_epoch = 0
    global_step = 0
    det_resume = getattr(opt, "det_resume", "")
    ckpt: Optional[Dict[str, Any]] = None
    if isinstance(det_resume, str) and det_resume.strip():
        ckpt = _load_ckpt(det_resume, map_location="cpu")
        # If checkpoint contains hparams, trust them to rebuild the same architecture
        hpd = ckpt.get("hparams", None)
        if isinstance(hpd, dict):
            for k, v in hpd.items():
                if hasattr(hp, k):
                    try:
                        setattr(hp, k, v)
                    except Exception:
                        pass

        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))

    # ------------------------------------------------------------------
    # SAFETY (important for torchvision FasterRCNN):
    # If hp.num_classes is smaller than what the dataloader produces,
    # classification targets can contain labels >= num_classes, which
    # triggers a CUDA "device-side assert" inside CrossEntropy.
    # We auto-bump hp.num_classes to match the dataloader.
    # ------------------------------------------------------------------
    try:
        nc_dl = int(num_classes_dl)
        if int(hp.num_classes) < nc_dl:
            print(
                f"[DET] ⚠️ hp.num_classes={hp.num_classes} < dataloader_num_classes={nc_dl}. "
                f"Overriding hp.num_classes -> {nc_dl} to avoid CUDA asserts."
            )
            hp.num_classes = nc_dl
    except Exception:
        pass

    # Save hparams (final)
    _write_hparams_json(save_dir, hp)

    print(f"[DET] hparams -> {save_dir / 'hparams_detection.json'}")
    print(
        f"[DET] head={hp.head_type} | num_classes={hp.num_classes} | "
        f"sem_backbone={getattr(hp, 'det_sem_backbone', 'resnet50')} | "
        f"return_layer={hp.det_sem_return_layer} | "
        f"freeze_backbone={hp.freeze_backbone} | drop_empty={hp.det_drop_empty} | "
        f"batch_safety={hp.det_filter_batch_safety} | optim={hp.det_optimizer}"
    )
    if str(hp.sem_pretrained_path).strip():
        print(
            f"[DET] sem_pretrained_path='{hp.sem_pretrained_path}' "
            f"(strict={hp.sem_pretrained_strict}, verbose={hp.sem_pretrained_verbose})"
        )

    # Model
    det_model, _backbone_module = _build_detector(opt, dev=dev, hp=hp)

    # Optim + sched
    optimizer = _build_optimizer(det_model, hp)
    epoch_scheduler = _build_epoch_scheduler(optimizer, hp)
    warmup = _WarmupLinearLR(
        optimizer,
        warmup_iters=int(hp.det_warmup_iters),
        warmup_factor=float(hp.det_warmup_factor),
    )

    # Load states if resuming
    if ckpt is not None:
        sd = _strip_module_prefix_if_needed(ckpt["model"])
        det_model.load_state_dict(sd, strict=True)

        if ckpt.get("optimizer", None) is not None:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"[DET] ⚠️ optimizer state not loaded: {e}")

        if ckpt.get("scheduler", None) is not None:
            try:
                epoch_scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e:
                print(f"[DET] ⚠️ scheduler state not loaded: {e}")

        if ckpt.get("warmup", None) is not None:
            try:
                warmup.load_state_dict(ckpt["warmup"])
            except Exception as e:
                print(f"[DET] ⚠️ warmup state not loaded: {e}")

        print(f"[DET] resumed from {det_resume} (start_epoch={start_epoch}, global_step={global_step})")

    # Save frequency
    save_freq_mode, save_freq_interval = _parse_save_freq_epoch_only(getattr(opt, "save_freq", "epoch"))
    epoch_ckpt_interval = getattr(opt, "epoch_ckpt_interval", None)
    if save_freq_mode == "epoch" and epoch_ckpt_interval is not None:
        try:
            save_freq_interval = max(1, int(epoch_ckpt_interval))
        except Exception:
            pass

    # Stats
    total_ignored_batches = 0
    total_ignored_samples = 0
    total_seen_samples = 0

    det_model.train()
    for epoch in range(start_epoch, int(hp.det_epochs)):
        det_model.train()
        total_loss = 0.0
        n_steps = 0

        pbar = tqdm(
            train_loader,
            desc=f"[DET-{hp.head_type}] epoch {epoch+1}/{hp.det_epochs}",
            ncols=140,
            leave=False,
        )

        for imgs, targets in pbar:
            B = int(len(imgs)) if isinstance(imgs, list) else int(imgs.shape[0])
            total_seen_samples += B

            imgs_list, targets_list = _to_device_images_targets(imgs, targets, dev)

            if hp.det_filter_batch_safety:
                imgs_list, targets_list, st = _filter_empty_targets(
                    imgs_list,
                    targets_list,
                    num_classes=int(hp.num_classes) if hp.num_classes is not None else None,
                    debug=bool(getattr(opt, "debug_detection", False)),
                )
                total_ignored_samples += int(st["ignored"])
                if st["used"] == 0:
                    total_ignored_batches += 1
                    continue

            optimizer.zero_grad(set_to_none=True)

            # main_moco-like: direct call with list inputs
            loss_dict = det_model(imgs_list, targets_list)
            loss = sum(loss_dict.values())

            loss.backward()

            if float(hp.det_grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in det_model.parameters() if p.requires_grad],
                    max_norm=float(hp.det_grad_clip),
                )

            optimizer.step()
            warmup.step()

            total_loss += float(loss.item())
            n_steps += 1
            global_step += 1

            cur_lrs = [pg.get("lr", 0.0) for pg in optimizer.param_groups]
            mean_lr = float(sum(cur_lrs) / max(1, len(cur_lrs)))
            pbar.set_postfix(loss=f"{total_loss / max(1, n_steps):.4f}", lr=f"{mean_lr:.2e}")

        train_loss = total_loss / max(1, n_steps)

        try:
            epoch_scheduler.step()
        except Exception as e:
            print(f"[DET] ⚠️ scheduler.step failed: {e}")

        print(
            f"[DET] epoch {epoch+1}/{hp.det_epochs} | train_loss={train_loss:.4f} | "
            f"ignored_samples_total={total_ignored_samples} | ignored_batches_total={total_ignored_batches}"
        )

        _save_ckpt(
            save_dir / "detector_last.pth",
            epoch=epoch,
            global_step=global_step,
            model=det_model,
            optimizer=optimizer,
            hp=hp,
            scheduler=epoch_scheduler,
            warmup=warmup,
            extra={
                "train_loss": float(train_loss),
                "ignored_samples_total": int(total_ignored_samples),
                "ignored_batches_total": int(total_ignored_batches),
                "seen_samples_total": int(total_seen_samples),
            },
        )

        if save_freq_mode == "epoch" and save_freq_interval is not None and ((epoch + 1) % int(save_freq_interval) == 0):
            _save_ckpt(
                save_dir / f"detector_epoch_{epoch+1:04d}.pth",
                epoch=epoch,
                global_step=global_step,
                model=det_model,
                optimizer=optimizer,
                hp=hp,
                scheduler=epoch_scheduler,
                warmup=warmup,
                extra={
                    "train_loss": float(train_loss),
                    "ignored_samples_total": int(total_ignored_samples),
                    "ignored_batches_total": int(total_ignored_batches),
                    "seen_samples_total": int(total_seen_samples),
                },
            )

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
