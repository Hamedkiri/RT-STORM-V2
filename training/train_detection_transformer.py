# file: training/train_detection_transformer.py
# -*- coding: utf-8 -*-

"""Detection training / evaluation / camera demo.

This module is called from `train_style_disentangle.py` when `--mode detect_transformer`.

It supports multiple detection heads via `--det_head`:
  - "detr"       : UNetGenerator backbone + SimpleDETRHead (DETR-like)
  - "fasterrcnn" : torchvision Faster R-CNN ResNet50-FPN (baseline)
  - "fastrnn"    : torchvision Faster R-CNN using a custom RNN RoI head (models/detection/fastrnn_detector.py)
  - "vitdet"     : placeholder (Not implemented here)

Key requirements implemented:
  1) Coherent FastRNN integration (training + inference) with the already-defined FastRNNDetector.
  2) When fine-tuning detection heads, we save weights **with the exact hparams used**.
     Checkpoints contain `hparams` and we also save `det_hparams.json` next to them.
     For tests/inference, you can build the detector directly from checkpoint hparams.
  3) COCO-style evaluation (mAP) with pycocotools when a COCO dataset/ann file is used.
  4) Camera demo (OpenCV) for real-time inference.

Notes
-----
* `build_detection_dataloader(opt)` is expected to return (train_loader, val_loader, num_classes).
  In this codebase it builds torchvision CocoDetection datasets.
* For COCO metrics we expect `val_loader.dataset.coco` and `val_loader.dataset.ids` to exist.
* For DETR-like head, boxes are predicted as normalized cxcywh and converted to xyxy for eval/demo.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from tqdm import tqdm

from data import build_detection_dataloader
from helpers import freeze, unfreeze
from models.generator import UNetGenerator
from models.det_transformer import SimpleDETRHead
from training.detr_criterion import SetCriterionDETR


# --------------------------------------------------------------------------------------
# Save-freq parsing (epoch-only for detector training)
# --------------------------------------------------------------------------------------

def _parse_save_freq_epoch_only(save_freq_str: Optional[str]) -> Tuple[str, Optional[int]]:
    """Parse save_freq for detector training.

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


# --------------------------------------------------------------------------------------
# Small utils
# --------------------------------------------------------------------------------------

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if torch.is_tensor(x):
            return float(x.detach().float().mean().cpu().item())
        return float(x)
    except Exception:
        return float(default)


def _box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """boxes: (..., 4) in cxcywh -> xyxy (same units)."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([x1, y1, (x2 - x1).clamp(min=0), (y2 - y1).clamp(min=0)], dim=-1)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _try_import_cv2():
    try:
        import cv2  # noqa

        return cv2
    except Exception:
        return None


def _try_import_pycocotools():
    try:
        from pycocotools.cocoeval import COCOeval  # noqa

        return True
    except Exception:
        return False


def _get_coco_cat_mapping(coco) -> Dict[int, str]:
    """Return {category_id: category_name} if available."""
    try:
        return {int(k): str(v.get("name", k)) for k, v in coco.cats.items()}
    except Exception:
        return {}

def _filter_empty_targets(images: torch.Tensor, targets: List[Dict[str, torch.Tensor]], dev, debug=False):
    """
    images: (B,C,H,W)
    targets: list of dicts with at least "boxes" and "labels"
    Return: list_images, list_targets, stats dict
    """
    B = images.shape[0]
    imgs_list = []
    tgts_list = []

    total = B
    used = 0
    ignored = 0

    for i in range(B):
        t = targets[i]
        boxes = t.get("boxes", None)
        labels = t.get("labels", None)

        ok = True
        if boxes is None or labels is None:
            ok = False
        else:
            if torch.is_tensor(boxes):
                ok = boxes.numel() > 0 and boxes.shape[-1] == 4
            else:
                ok = False

        if not ok:
            ignored += 1
            if debug:
                print(f"[DEBUG DET] ignore img[{i}] (no valid boxes/labels)")
            continue

        imgs_list.append(images[i].to(dev))
        tgts_list.append({k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in t.items()})
        used += 1

    stats = {"total": total, "used": used, "ignored": ignored}
    return imgs_list, tgts_list, stats

# --------------------------------------------------------------------------------------
# HParams captured in ckpts
# --------------------------------------------------------------------------------------

@dataclass
class DetectorHParams:
    head_type: str
    num_classes: int
    feat_source: str
    freeze_backbone: bool
    img_h: int
    img_w: int
    lr: float
    weight_decay: float
    epochs: int
    # DETR-like
    det_num_queries: int = 300
    det_nheads: int = 8
    det_dec_layers: int = 6
    det_token_dim: int = 256
    det_d_model: int = 256
    det_eos_coef: float = 0.1
    det_score_thresh: float = 0.5
    # FastRNN
    fastrnn_hidden: int = 256
    fastrnn_bidir: bool = True
    fastrnn_dropout: float = 0.0
    fastrnn_focal_alpha: float = 0.25
    fastrnn_focal_gamma: float = 2.0
    fastrnn_score_thresh: float = 0.05
    fastrnn_nms_thresh: float = 0.5
    fastrnn_topk: int = 1000


def _collect_hparams(opt, head_type: str, num_classes: int) -> DetectorHParams:
    head_type = str(head_type).lower().strip()
    feat_source = str(getattr(opt, "det_feat_source", "content")).lower().strip()
    freeze_backbone = bool(getattr(opt, "det_freeze_backbone", False))
    img_h = int(getattr(opt, "det_img_h", getattr(opt, "crop_size", 256)))
    img_w = int(getattr(opt, "det_img_w", getattr(opt, "crop_size", 256)))
    lr = float(getattr(opt, "det_lr", getattr(opt, "lr", 1e-4)))
    wd = float(getattr(opt, "det_weight_decay", 1e-4))
    epochs = int(getattr(opt, "det_epochs", getattr(opt, "epochs", 20)))

    hp = DetectorHParams(
        head_type=head_type,
        num_classes=int(num_classes),
        feat_source=feat_source,
        freeze_backbone=freeze_backbone,
        img_h=img_h,
        img_w=img_w,
        lr=lr,
        weight_decay=wd,
        epochs=epochs,
        det_num_queries=int(getattr(opt, "det_num_queries", 300)),
        det_nheads=int(getattr(opt, "det_nheads", 8)),
        det_dec_layers=int(getattr(opt, "det_dec_layers", 6)),
        det_token_dim=int(getattr(opt, "det_token_dim", 256)),
        det_d_model=int(getattr(opt, "det_d_model", int(getattr(opt, "det_token_dim", 256)))),
        det_eos_coef=float(getattr(opt, "det_eos_coef", 0.1)),
        det_score_thresh=float(getattr(opt, "det_score_thresh", 0.5)),
        fastrnn_hidden=int(getattr(opt, "fastrnn_hidden", 256)),
        fastrnn_bidir=bool(getattr(opt, "fastrnn_bidir", True)),
        fastrnn_dropout=float(getattr(opt, "fastrnn_dropout", 0.0)),
        fastrnn_focal_alpha=float(getattr(opt, "fastrnn_focal_alpha", 0.25)),
        fastrnn_focal_gamma=float(getattr(opt, "fastrnn_focal_gamma", 2.0)),
        fastrnn_score_thresh=float(getattr(opt, "fastrnn_score_thresh", 0.05)),
        fastrnn_nms_thresh=float(getattr(opt, "fastrnn_nms_thresh", 0.5)),
        fastrnn_topk=int(getattr(opt, "fastrnn_topk", 1000)),
    )
    return hp


def _write_hparams_json(save_dir: Path, hp: DetectorHParams) -> Path:
    p = save_dir / "det_hparams.json"
    p.write_text(json.dumps(asdict(hp), indent=2, sort_keys=True))
    return p


# --------------------------------------------------------------------------------------
# Backbones for FastRNN
# --------------------------------------------------------------------------------------

class _UNetFeatBackbone(torch.nn.Module):
    """Extract a single feature map from UNetGenerator.

    feat_source:
      - "content": use UNetGenerator.encode_content(x) -> bottleneck feature map (z)
      - "style"  : use UNetGenerator.style_enc(x) -> choose a pyramid map (default s5)
    """

    def __init__(self, G: UNetGenerator, feat_source: str = "content", style_level: str = "s5"):
        super().__init__()
        self.G = G
        self.feat_source = str(feat_source).lower().strip()
        self.style_level = str(style_level)

    @torch.no_grad()
    def infer_out_channels(self, dev: torch.device, img_h: int, img_w: int) -> int:
        x = torch.zeros(1, 3, img_h, img_w, device=dev)
        f = self.forward(x)
        return int(f.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.feat_source == "style":
            maps, _toks, _tokG = self.G.style_enc(x)
            if isinstance(maps, dict) and self.style_level in maps:
                return maps[self.style_level]
            # fallback: try last
            if isinstance(maps, dict) and len(maps) > 0:
                return list(maps.values())[0]
            raise RuntimeError("UNet style encoder returned empty maps")

        # content (default)
        z, _skips = self.G.encode_content(x)
        return z


class _SemanticResNetBackbone(torch.nn.Module):
    """Wrapper around ResNet50Backbone from semantic_moco_jepa.

    It outputs a single feature map (B, 2048, H/32, W/32).
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from models.semantic_moco_jepa import ResNet50Backbone

        self.backbone = ResNet50Backbone(pretrained=pretrained)
        self.out_channels = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def _load_backbone_weights_from_ckpt(backbone: torch.nn.Module, ckpt_path: str) -> None:
    """Best-effort loading for various ckpt formats in this repo."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"⚠️ Could not load ckpt '{ckpt_path}': {e}")
        return

    if not isinstance(ckpt, dict):
        try:
            backbone.load_state_dict(ckpt, strict=False)
            print(f"✓ Backbone loaded from '{ckpt_path}' (raw state_dict)")
        except Exception as e:
            print(f"⚠️ Backbone load failed from '{ckpt_path}': {e}")
        return

    # Heuristics (most common)
    candidates = [
        "backbone",
        "G_B",
        "G_A",
        "SEM",
        "semantic",
        "model",
        "state_dict",
    ]
    state_dict = None
    src_key = None
    for k in candidates:
        if k in ckpt and isinstance(ckpt[k], dict):
            state_dict = ckpt[k]
            src_key = k
            break
    if state_dict is None:
        # maybe already a state_dict-like ckpt
        state_dict = ckpt
        src_key = "<full_dict>"

    try:
        backbone.load_state_dict(state_dict, strict=False)
        print(f"✓ Backbone loaded from '{ckpt_path}' (key={src_key})")
    except Exception as e:
        print(f"⚠️ Backbone load failed from '{ckpt_path}' (key={src_key}): {e}")


# --------------------------------------------------------------------------------------
# Detector builders
# --------------------------------------------------------------------------------------

def _build_detector(
    opt,
    dev: torch.device,
    num_classes: int,
    hp: Optional[DetectorHParams] = None,
) -> Tuple[torch.nn.Module, Optional[torch.nn.Module], Optional[Any]]:
    """Build detector model.

    Returns: (det_model, backbone_module, criterion)
      - backbone_module is used for freeze/unfreeze toggles (can be None)
      - criterion is for DETR-like head only (can be None)
    """

    head = str(getattr(opt, "det_head", "detr")).lower().strip()
    if hp is not None:
        head = str(hp.head_type).lower().strip()

    freeze_backbone = bool(getattr(opt, "det_freeze_backbone", False))
    feat_source = str(getattr(opt, "det_feat_source", "content")).lower().strip()
    det_backbone_ckpt = getattr(opt, "det_backbone_ckpt", None)

    # ------------------------------ DETR-like (UNet + transformer head) ------------------------------
    if head == "detr":
        token_dim = int(getattr(opt, "det_token_dim", 256))
        d_model = int(getattr(opt, "det_d_model", token_dim))
        num_queries = int(getattr(opt, "det_num_queries", 300))
        nheads = int(getattr(opt, "det_nheads", 8))
        dec_layers = int(getattr(opt, "det_dec_layers", 6))
        feat_branch = str(getattr(opt, "det_feat_branch", "content")).lower().strip()

        G_det = UNetGenerator(token_dim=token_dim).to(dev)
        if det_backbone_ckpt:
            _load_backbone_weights_from_ckpt(G_det, det_backbone_ckpt)

        det_model = SimpleDETRHead(
            generator=G_det,
            num_classes=num_classes,
            num_queries=num_queries,
            d_model=d_model,
            nheads=nheads,
            num_decoder_layers=dec_layers,
            feat_branch=feat_branch,
        ).to(dev)

        backbone_module = G_det
        if freeze_backbone:
            freeze(G_det)
            print("🧊 Backbone UNet frozen (DETR head trains)")
        else:
            unfreeze(G_det)
            print("🔥 Backbone UNet trainable")

        criterion = SetCriterionDETR(
            num_classes=num_classes,
            matcher=None,
            eos_coef=float(getattr(opt, "det_eos_coef", 0.1)),
        )

        return det_model, backbone_module, criterion

    # ------------------------------ torchvision Faster R-CNN baseline ------------------------------
    if head == "fasterrcnn":
        try:
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
        except Exception as e:
            raise ImportError("torchvision detection models not available") from e

        det_model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
        det_model = det_model.to(dev)

        if det_backbone_ckpt:
            # best-effort: assume ckpt contains whole model
            try:
                ckpt = torch.load(det_backbone_ckpt, map_location="cpu")
                sd = ckpt.get("model", ckpt)
                det_model.load_state_dict(sd, strict=False)
                print(f"✓ FasterRCNN loaded from '{det_backbone_ckpt}'")
            except Exception as e:
                print(f"⚠️ FasterRCNN load failed from '{det_backbone_ckpt}': {e}")

        backbone_module = getattr(det_model, "backbone", None)
        if freeze_backbone:
            if backbone_module is not None:
                freeze(backbone_module)
                print("🧊 FasterRCNN backbone frozen")
            else:
                freeze(det_model)
                print("🧊 FasterRCNN frozen (no explicit backbone)")
        else:
            unfreeze(det_model)
            print("🔥 FasterRCNN trainable")

        return det_model, backbone_module, None

    # ------------------------------ FastRNN (custom RNN RoI head) ------------------------------
    # ------------------------------ FastRNN (FCOS-like head with small RNN head) ------------------------------
    if head == "fastrnn":
        from models.detection.fastrnn_detector import FastRNNDetector
        import torchvision.models as tvm
        from torchvision.models._utils import IntermediateLayerGetter
        import torch.nn as nn

        # ---------------- semantic ResNet50 backbone (traditional detection style) ----------------
        class _SemanticResNetBackbone(nn.Module):
            """
            ResNet50 backbone returning ONE feature map from layer2/layer3/layer4 using IntermediateLayerGetter.
            Output is a dict {"0": feat} (as expected by FastRNNDetector._extract_feat).
            """

            def __init__(self, return_layer: str = "layer4", pretrained: bool = True):
                super().__init__()
                return_layer = str(return_layer).lower().strip()
                assert return_layer in ("layer2", "layer3", "layer4")

                # weights API may differ depending torchvision version; keep it robust:
                weights = None
                try:
                    if pretrained:
                        weights = tvm.ResNet50_Weights.DEFAULT
                except Exception:
                    weights = "IMAGENET1K_V1" if pretrained else None

                resnet = tvm.resnet50(weights=weights)
                # remove classification head
                resnet.avgpool = nn.Identity()
                resnet.fc = nn.Identity()

                self.body = IntermediateLayerGetter(resnet, {return_layer: "0"})
                self.return_layer = return_layer
                self.out_channels = {"layer2": 512, "layer3": 1024, "layer4": 2048}[return_layer]

            def forward(self, x: torch.Tensor):
                return self.body(x)  # dict {"0": feat}

        # ---------------- choose feat source ----------------
        feat_source = str(getattr(opt, "det_feat_source", "sem_resnet50")).lower().strip()
        det_backbone_ckpt = getattr(opt, "det_backbone_ckpt", None)

        if feat_source == "sem_resnet50":
            return_layer = str(getattr(opt, "det_sem_return_layer", "layer4")).lower().strip()
            sem_pretrained = bool(getattr(opt, "sem_pretrained", 1))

            backbone = _SemanticResNetBackbone(return_layer=return_layer, pretrained=sem_pretrained).to(dev)

            # optional load ckpt into the resnet body (best-effort)
            if det_backbone_ckpt:
                try:
                    ckpt = torch.load(det_backbone_ckpt, map_location="cpu")
                    sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
                    # try to load directly into resnet inside IntermediateLayerGetter
                    backbone.body.load_state_dict(sd, strict=False)
                    print(f"✓ Loaded semantic ResNet weights from {det_backbone_ckpt} (strict=False)")
                except Exception as e:
                    print(f"⚠️ Could not load semantic ResNet weights from {det_backbone_ckpt}: {e}")

            backbone_module = backbone  # for freeze/unfreeze convenience
            out_channels = int(backbone.out_channels)

        else:
            # fallback: UNet backbone path (your existing one)
            token_dim = int(getattr(opt, "det_token_dim", 256))
            G_det = UNetGenerator(token_dim=token_dim).to(dev)
            if det_backbone_ckpt:
                _load_backbone_weights_from_ckpt(G_det, det_backbone_ckpt)
            backbone = _UNetFeatBackbone(
                G_det,
                feat_source=feat_source,  # "content" or "style" etc.
                style_level=str(getattr(opt, "det_style_level", "s5")),
            ).to(dev)
            backbone_module = G_det
            img_h = int(getattr(opt, "det_img_h", getattr(opt, "crop_size", 256)))
            img_w = int(getattr(opt, "det_img_w", getattr(opt, "crop_size", 256)))
            out_channels = int(backbone.infer_out_channels(dev, img_h=img_h, img_w=img_w))

        # ✅ IMPORTANT: match FastRNNDetector signature EXACTLY
        det_model = FastRNNDetector(
            backbone=backbone,
            out_channels=int(out_channels),
            num_classes=int(num_classes),
            head_hidden=int(getattr(opt, "fastrnn_hidden", 256)),
            head_bidir=bool(getattr(opt, "fastrnn_bidir", True)),
            head_dropout=float(getattr(opt, "fastrnn_dropout", 0.0)),
            size_divisible=int(getattr(opt, "fastrnn_size_divisible", 32)),
            focal_alpha=float(getattr(opt, "fastrnn_focal_alpha", 0.25)),
            focal_gamma=float(getattr(opt, "fastrnn_focal_gamma", 2.0)),
            score_thresh=float(getattr(opt, "fastrnn_score_thresh", 0.05)),
            nms_thresh=float(getattr(opt, "fastrnn_nms_thresh", 0.5)),
            topk=int(getattr(opt, "fastrnn_topk", 1000)),
        ).to(dev)

        # Freeze/unfreeze backbone module
        freeze_backbone = bool(getattr(opt, "det_freeze_backbone", False))
        if freeze_backbone:
            freeze(backbone_module)
            print(f"🧊 FastRNN backbone frozen (feat_source={feat_source}, C={out_channels})")
        else:
            unfreeze(backbone_module)
            print(f"🔥 FastRNN backbone trainable (feat_source={feat_source}, C={out_channels})")

        return det_model, backbone_module, None

    if head == "vitdet":
        raise NotImplementedError(
            "det_head='vitdet' is declared in config but not implemented in this training script. "
            "Use det_head in {detr, fasterrcnn, fastrnn}."
        )

    raise ValueError(f"Unknown det_head='{head}'.")


# --------------------------------------------------------------------------------------
# Checkpoint IO (head weights + backbone weights + exact hparams)
# --------------------------------------------------------------------------------------

def _save_detector_ckpt(
    path: Path,
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


def _load_detector_ckpt(ckpt_path: str, map_location: str = "cpu") -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise RuntimeError(f"Invalid detector checkpoint: {ckpt_path}")
    return ckpt


def build_detector_from_checkpoint(
    ckpt_path: str,
    dev: torch.device,
    opt_fallback=None,
) -> Tuple[torch.nn.Module, DetectorHParams]:
    """Build a detector solely from a detector checkpoint (weights + hparams).

    This is the recommended path for tests/inference to guarantee consistent hparams.
    """
    ckpt = _load_detector_ckpt(ckpt_path)
    hp_dict = ckpt.get("hparams", {})
    hp = DetectorHParams(**hp_dict)

    # Build model using an opt-like object: prefer user opt (for paths), but override core hparams.
    class _OptProxy:
        pass

    opt = _OptProxy()
    if opt_fallback is not None:
        for k, v in vars(opt_fallback).items():
            setattr(opt, k, v)
    # overwrite with hp fields
    for k, v in hp_dict.items():
        setattr(opt, k, v)
    setattr(opt, "det_head", hp.head_type)
    setattr(opt, "det_feat_source", hp.feat_source)
    setattr(opt, "det_freeze_backbone", hp.freeze_backbone)
    setattr(opt, "det_img_h", hp.img_h)
    setattr(opt, "det_img_w", hp.img_w)
    setattr(opt, "det_lr", hp.lr)
    setattr(opt, "det_weight_decay", hp.weight_decay)
    setattr(opt, "det_epochs", hp.epochs)
    # Ensure these exist for builders
    setattr(opt, "det_num_queries", hp.det_num_queries)
    setattr(opt, "det_nheads", hp.det_nheads)
    setattr(opt, "det_dec_layers", hp.det_dec_layers)
    setattr(opt, "det_token_dim", hp.det_token_dim)
    setattr(opt, "det_d_model", hp.det_d_model)
    setattr(opt, "det_eos_coef", hp.det_eos_coef)
    setattr(opt, "det_score_thresh", hp.det_score_thresh)
    setattr(opt, "fastrnn_hidden", hp.fastrnn_hidden)
    setattr(opt, "fastrnn_bidir", hp.fastrnn_bidir)
    setattr(opt, "fastrnn_dropout", hp.fastrnn_dropout)
    setattr(opt, "fastrnn_focal_alpha", hp.fastrnn_focal_alpha)
    setattr(opt, "fastrnn_focal_gamma", hp.fastrnn_focal_gamma)
    setattr(opt, "fastrnn_score_thresh", hp.fastrnn_score_thresh)
    setattr(opt, "fastrnn_nms_thresh", hp.fastrnn_nms_thresh)
    setattr(opt, "fastrnn_topk", hp.fastrnn_topk)

    model, _bb, _crit = _build_detector(opt, dev=dev, num_classes=hp.num_classes, hp=hp)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, hp


# --------------------------------------------------------------------------------------
# Inference helpers (common format)
# --------------------------------------------------------------------------------------

@torch.no_grad()
def _infer_batch(
    model: torch.nn.Module,
    head_type: str,
    images: torch.Tensor,
    score_thresh: float,
) -> List[Dict[str, torch.Tensor]]:
    """Return detections list[{boxes, scores, labels}] in xyxy absolute pixels."""
    head_type = str(head_type).lower().strip()

    if head_type in ("fasterrcnn", "fastrnn"):
        imgs_list = [img for img in images]
        outputs = model(imgs_list)
        return outputs

    if head_type == "detr":
        pred_logits, pred_boxes = model(images)
        # pred_boxes is normalized cxcywh
        B, _, H, W = images.shape
        probs = torch.softmax(pred_logits, dim=-1)
        # best class excluding background 0
        scores, labels = probs[..., 1:].max(-1)
        labels = labels + 1
        boxes_xyxy = _box_cxcywh_to_xyxy(pred_boxes)
        boxes_xyxy[..., 0::2] *= float(W)
        boxes_xyxy[..., 1::2] *= float(H)

        outs: List[Dict[str, torch.Tensor]] = []
        for b in range(B):
            keep = scores[b] >= float(score_thresh)
            outs.append(
                {
                    "boxes": boxes_xyxy[b][keep].detach().cpu(),
                    "scores": scores[b][keep].detach().cpu(),
                    "labels": labels[b][keep].detach().cpu(),
                }
            )
        return outs

    raise ValueError(f"Unknown head_type='{head_type}'")


# --------------------------------------------------------------------------------------
# COCO evaluation
# --------------------------------------------------------------------------------------

@torch.no_grad()
def evaluate_coco_map(
    model: torch.nn.Module,
    head_type: str,
    data_loader,
    dev: torch.device,
    score_thresh: float = 0.05,
    max_images: Optional[int] = None,
) -> Dict[str, float]:
    """Compute COCO mAP metrics (bbox) with pycocotools.

    Returns a dict of metrics (AP, AP50, AP75, etc.) when possible.
    """
    if not _try_import_pycocotools():
        raise ImportError(
            "pycocotools is required for COCO evaluation. Install it or disable det_run=eval."
        )

    from pycocotools.cocoeval import COCOeval

    ds = getattr(data_loader, "dataset", None)
    coco_gt = getattr(ds, "coco", None)
    if coco_gt is None:
        raise RuntimeError("Validation dataset does not expose .coco; cannot run COCOeval.")

    cat_map = _get_coco_cat_mapping(coco_gt)
    if cat_map:
        print(f"[COCO] categories: {len(cat_map)}")

    model.eval()
    results: List[Dict[str, Any]] = []
    seen = 0

    pbar = tqdm(data_loader, desc="[EVAL] COCO", ncols=140, leave=False)
    for images, targets in pbar:
        images = images.to(dev)
        targets_list = targets  # keep on CPU for IDs
        outputs = _infer_batch(model, head_type=head_type, images=images, score_thresh=score_thresh)

        for out, tgt in zip(outputs, targets_list):
            # COCO image_id
            img_id = int(tgt.get("image_id").item() if torch.is_tensor(tgt.get("image_id")) else tgt.get("image_id"))
            boxes = out["boxes"]
            scores = out["scores"]
            labels = out["labels"]

            if boxes.numel() == 0:
                continue

            boxes_xywh = _xyxy_to_xywh(boxes)
            for b in range(boxes_xywh.shape[0]):
                results.append(
                    {
                        "image_id": img_id,
                        "category_id": int(labels[b].item()),
                        "bbox": [float(x) for x in boxes_xywh[b].tolist()],
                        "score": float(scores[b].item()),
                    }
                )

        seen += len(outputs)
        if max_images is not None and seen >= int(max_images):
            break

    if len(results) == 0:
        print("[EVAL] No detections produced — mAP will be 0.")
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # stats indices (COCOeval):
    # 0: AP, 1: AP50, 2: AP75, 3: AP_small, 4: AP_medium, 5: AP_large,
    # 6: AR@1, 7: AR@10, 8: AR@100, 9: AR_small, 10: AR_medium, 11: AR_large
    s = coco_eval.stats
    metrics = {
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
    return metrics


# --------------------------------------------------------------------------------------
# Camera demo
# --------------------------------------------------------------------------------------

@torch.no_grad()
def run_camera_demo(
    model: torch.nn.Module,
    head_type: str,
    dev: torch.device,
    class_names: Optional[Dict[int, str]] = None,
    cam_id: int = 0,
    score_thresh: float = 0.3,
    resize_hw: Optional[Tuple[int, int]] = None,
) -> None:
    cv2 = _try_import_cv2()
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for camera demo.")

    cap = cv2.VideoCapture(int(cam_id))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera id={cam_id}")

    print("🎥 Camera demo: press 'q' to quit")
    model.eval()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize_hw is not None:
            h, w = resize_hw
            rgb = cv2.resize(rgb, (int(w), int(h)))

        img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0).to(dev)

        outputs = _infer_batch(model, head_type=head_type, images=img, score_thresh=score_thresh)
        out = outputs[0]

        # draw on original frame (use frame dims)
        disp = frame.copy()
        H, W = disp.shape[:2]
        boxes = out["boxes"].numpy() if torch.is_tensor(out["boxes"]) else out["boxes"]
        scores = out["scores"].numpy() if torch.is_tensor(out["scores"]) else out["scores"]
        labels = out["labels"].numpy() if torch.is_tensor(out["labels"]) else out["labels"]

        for (x1, y1, x2, y2), sc, lab in zip(boxes, scores, labels):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if sc < score_thresh:
                continue
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


# --------------------------------------------------------------------------------------
# Training loop (supports detr/fasterrcnn/fastrnn)
# --------------------------------------------------------------------------------------

def train_detection_transformer(opt, dev: torch.device):
    """Entry point for detection mode.

    Controlled by `--det_run`:
      - train  : run training
      - eval   : run COCO evaluation on val split (requires --det_ckpt)
      - camera : run camera demo (requires --det_ckpt)
    """

    save_dir = Path(opt.save_dir)
    _ensure_dir(save_dir)

    det_run = str(getattr(opt, "det_run", "train")).lower().strip()
    det_ckpt = getattr(opt, "det_ckpt", None)
    if det_run in ("eval", "camera") and not det_ckpt:
        raise ValueError("det_run requires --det_ckpt (path to detector_*.pth)")

    # ------------------------------ eval / camera: build from checkpoint hparams ------------------------------
    if det_run in ("eval", "camera"):
        model, hp = build_detector_from_checkpoint(det_ckpt, dev=dev, opt_fallback=opt)
        head_type = hp.head_type

        # Build dataloader only for: (a) class names, (b) eval dataset
        _train_loader, val_loader, _num_classes = build_detection_dataloader(opt)
        coco = getattr(getattr(val_loader, "dataset", None), "coco", None)
        class_names = _get_coco_cat_mapping(coco) if coco is not None else {}

        if det_run == "eval":
            score_th = float(getattr(opt, "det_score_thresh", hp.det_score_thresh))
            max_imgs = getattr(opt, "det_eval_max_images", None)
            metrics = evaluate_coco_map(model, head_type=head_type, data_loader=val_loader, dev=dev, score_thresh=score_th, max_images=max_imgs)
            (save_dir / "det_eval_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
            print(f"🧾 Metrics saved -> {save_dir / 'det_eval_metrics.json'}")
            return metrics

        # camera
        cam_id = int(getattr(opt, "det_cam_id", 0))
        score_th = float(getattr(opt, "det_score_thresh", hp.det_score_thresh))
        resize_hw = None
        if bool(getattr(opt, "det_cam_resize", False)):
            resize_hw = (int(hp.img_h), int(hp.img_w))
        run_camera_demo(model, head_type=head_type, dev=dev, class_names=class_names, cam_id=cam_id, score_thresh=score_th, resize_hw=resize_hw)
        return None

    # ------------------------------ train: build loaders + model from opt ------------------------------
    train_loader, val_loader, num_classes = build_detection_dataloader(opt)
    head_type = str(getattr(opt, "det_head", "detr")).lower().strip()

    # Determine save frequency (epoch-based)
    save_freq_mode, save_freq_interval = _parse_save_freq_epoch_only(getattr(opt, "save_freq", "epoch"))
    epoch_ckpt_interval = getattr(opt, "epoch_ckpt_interval", None)
    if save_freq_mode == "epoch" and epoch_ckpt_interval is not None:
        try:
            save_freq_interval = max(1, int(epoch_ckpt_interval))
        except Exception:
            pass

    hp = _collect_hparams(opt, head_type=head_type, num_classes=num_classes)
    _write_hparams_json(save_dir, hp)
    print(f"📝 Detection hparams saved -> {save_dir / 'det_hparams.json'}")

    det_model, backbone_module, criterion = _build_detector(opt, dev=dev, num_classes=num_classes, hp=hp)

    # Optimizer (respecting freeze flags via requires_grad)
    params = [p for p in det_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=hp.lr, weight_decay=hp.weight_decay)

    # Resume detection finetune (optional)
    det_resume = getattr(opt, "det_resume", None)
    start_epoch = 0
    global_step = 0
    best_ap = -1.0
    if det_resume:
        ckpt = _load_detector_ckpt(det_resume)
        det_model.load_state_dict(ckpt["model"], strict=True)
        if ckpt.get("optimizer", None) is not None:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"⚠️ Could not load optimizer state: {e}")
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        print(f"✓ Resumed detector from {det_resume} (start_epoch={start_epoch}, global_step={global_step})")

    # For COCO eval, keep a mapping
    coco_val = getattr(getattr(val_loader, "dataset", None), "coco", None)
    class_names = _get_coco_cat_mapping(coco_val) if coco_val is not None else {}

    # Training loop
    for epoch in range(start_epoch, hp.epochs):
        det_model.train()
        if backbone_module is not None:
            backbone_module.train(not hp.freeze_backbone)

        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"[DET-{head_type}] epoch {epoch+1}/{hp.epochs}", ncols=160, leave=False)
        for imgs, targets in pbar:
            imgs = imgs.to(dev)
            targets_list = [{k: v.to(dev) for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)

            if head_type == "fasterrcnn":
                imgs_list = [im for im in imgs]
                loss_dict = det_model(imgs_list, targets_list)
                loss = sum(loss_dict.values())

            elif head_type == "fastrnn":
                debug_det = bool(getattr(opt, "debug_detection", False))
                valid_imgs, valid_tgts, st = _filter_empty_targets(imgs, targets_list, dev=dev, debug=debug_det)
                if st["used"] == 0:
                    if debug_det:
                        print("[DEBUG DET] batch skipped (0 valid images)")
                    continue
                loss_dict = det_model(valid_imgs, valid_tgts)
                loss = sum(loss_dict.values())

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1
            global_step += 1
            pbar.set_postfix(loss=f"{total_loss / max(1, n_batches):.4f}")

        train_loss = total_loss / max(1, n_batches)
        print(f"Epoch {epoch+1}/{hp.epochs} - train loss = {train_loss:.4f}")

        # ------------------------------ validation loss (loss-only) ------------------------------
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            if head_type in ("fasterrcnn", "fastrnn"):
                # to get losses in torchvision detectors, keep train() on
                det_model.train()
            else:
                det_model.eval()
                if backbone_module is not None:
                    backbone_module.eval()

            for imgs, targets in val_loader:
                imgs = imgs.to(dev)
                targets_list = [{k: v.to(dev) for k, v in t.items()} for t in targets]

                if head_type in ("fasterrcnn", "fastrnn"):
                    imgs_list = [im for im in imgs]
                    loss_dict = det_model(imgs_list, targets_list)
                    loss = sum(loss_dict.values())
                else:
                    pred_logits, pred_boxes = det_model(imgs)
                    outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
                    loss_dict = criterion(outputs, targets_list)
                    loss = sum(loss_dict[k] * criterion.weight_dict.get(k, 1.0) for k in loss_dict.keys())

                val_loss += float(loss.item())
                val_batches += 1

        val_loss = val_loss / max(1, val_batches)
        print(f"Epoch {epoch+1}/{hp.epochs} - val loss = {val_loss:.4f}")

        # ------------------------------ COCO mAP (bbox) ------------------------------
        ap_metrics: Dict[str, float] = {}
        try:
            if coco_val is not None:
                det_model.eval()
                score_th = float(getattr(opt, "det_score_thresh", hp.det_score_thresh))
                ap_metrics = evaluate_coco_map(det_model, head_type=head_type, data_loader=val_loader, dev=dev, score_thresh=score_th)
                ap = float(ap_metrics.get("AP", 0.0))
                print(f"Epoch {epoch+1} - COCO AP={ap:.4f} AP50={ap_metrics.get('AP50', 0.0):.4f}")
                if ap > best_ap:
                    best_ap = ap
        except Exception as e:
            print(f"⚠️ COCO evaluation skipped/failed: {e}")

        # ------------------------------ save checkpoints (last, best, periodic) ------------------------------
        extra = {
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "ap_metrics": ap_metrics,
            "class_names": class_names,
        }

        _save_detector_ckpt(
            save_dir / "detector_last.pth",
            epoch=epoch,
            global_step=global_step,
            model=det_model,
            optimizer=optimizer,
            hp=hp,
            extra=extra,
        )

        if ap_metrics and ap_metrics.get("AP", -1.0) >= best_ap - 1e-12:
            _save_detector_ckpt(
                save_dir / "detector_best.pth",
                epoch=epoch,
                global_step=global_step,
                model=det_model,
                optimizer=optimizer,
                hp=hp,
                extra=extra,
            )

        if save_freq_mode == "epoch" and save_freq_interval is not None and ((epoch + 1) % save_freq_interval == 0):
            _save_detector_ckpt(
                save_dir / f"detector_epoch_{epoch+1:04d}.pth",
                epoch=epoch,
                global_step=global_step,
                model=det_model,
                optimizer=optimizer,
                hp=hp,
                extra=extra,
            )

        # Save a small json log for quick checks
        (save_dir / "det_train_log.json").write_text(
            json.dumps(
                {
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "ap_metrics": ap_metrics,
                    "best_ap": float(best_ap),
                    "hparams": asdict(hp),
                },
                indent=2,
                sort_keys=True,
            )
        )

    print("✅ Detection training finished.")
    return None