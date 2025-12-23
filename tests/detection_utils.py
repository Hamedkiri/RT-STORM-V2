# tests/detection_utils.py
from __future__ import annotations

import json
import math
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
    """
    Normalise en list[Tensor(C,H,W)] sur device.
    """
    if torch.is_tensor(images):
        # images: (B,C,H,W) -> list
        return [img.to(dev) for img in images]
    return [img.to(dev) for img in list(images)]


def _ensure_chw_float01(img: torch.Tensor) -> torch.Tensor:
    """
    Best-effort: assure float32, [0,1], CHW.
    """
    if img.dtype != torch.float32:
        img = img.float()
    # si vraisemblablement en 0..255
    if img.max().item() > 1.5:
        img = img / 255.0
    return img


# ---------------------------------------------------------------------
# Checkpoint + hyperparams loading
# ---------------------------------------------------------------------
def _load_detector_ckpt(ckpt_path: Union[str, Path], map_location: str = "cpu") -> Dict[str, Any]:
    ckpt_path = str(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Invalid detector checkpoint (not a dict): {ckpt_path}")

    # accept legacy key
    if "model" not in ckpt and "state_dict" in ckpt:
        ckpt["model"] = ckpt["state_dict"]
    if "model" not in ckpt:
        raise RuntimeError(f"Invalid detector checkpoint (missing 'model'/'state_dict'): {ckpt_path}")

    # hparams optional
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
# HParams dataclass (centrée sur FastRNN pour l'instant)
# ---------------------------------------------------------------------
@dataclass
class DetectorHParams:
    # Core
    head_type: str = "fastrnn"        # <-- pour l'instant on supporte seulement fastrnn
    num_classes: int = 81             # inclut background = 0
    feat_source: str = "sem_resnet50" # sem_resnet50 (backbone torchvision) ou autre plus tard
    freeze_backbone: bool = True
    img_h: int = 256
    img_w: int = 256
    sem_pretrained: bool = True       # backbone torchvision pretrained

    # (optionnel) layer à retourner si tu passes par IntermediateLayerGetter
    det_sem_return_layer: str = "layer4"

    # FastRNN detector params
    fastrnn_hidden: int = 256
    fastrnn_bidir: bool = True
    fastrnn_dropout: float = 0.0
    fastrnn_focal_alpha: float = 0.25
    fastrnn_focal_gamma: float = 2.0
    fastrnn_score_thresh: float = 0.05
    fastrnn_nms_thresh: float = 0.5
    fastrnn_topk: int = 1000
    fastrnn_size_divisible: int = 32

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DetectorHParams":
        allowed = {f.name for f in fields(DetectorHParams)}
        filtered = {k: v for k, v in d.items() if k in allowed}
        return DetectorHParams(**filtered)


# ---------------------------------------------------------------------
# Backbone builders (cohérent + minimal)
# ---------------------------------------------------------------------
def _build_sem_resnet50_backbone(
    *,
    pretrained: bool,
    return_layer: str = "layer4",
) -> Tuple[torch.nn.Module, int]:
    """
    Backbone torchvision ResNet50 qui renvoie un dict {"0": feat}.
    out_channels dépend du layer:
      - layer4 -> 2048
      - layer3 -> 1024
      - layer2 -> 512
      - layer1 -> 256
    """
    import torchvision
    from torchvision.models._utils import IntermediateLayerGetter

    # weights API change selon torchvision; on fait best-effort
    weights = None
    if pretrained:
        try:
            weights = torchvision.models.ResNet50_Weights.DEFAULT
        except Exception:
            weights = None

    m = torchvision.models.resnet50(weights=weights)

    # channels per layer
    layer_to_c = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}
    if return_layer not in layer_to_c:
        raise ValueError(f"return_layer must be one of {list(layer_to_c.keys())}, got {return_layer}")

    out_channels = layer_to_c[return_layer]

    backbone = IntermediateLayerGetter(m, return_layers={return_layer: "0"})
    return backbone, out_channels


# ---------------------------------------------------------------------
# Build detector from checkpoint (+ hparams JSON)
# ---------------------------------------------------------------------
def build_detector_from_checkpoint(
    ckpt_path: Union[str, Path],
    dev: torch.device,
    *,
    hparams_json: Optional[Union[str, Path]] = None,
    auto_hparams_json: bool = True,
    prefer_json_over_ckpt: bool = True,
    strict: bool = True,
) -> Tuple[torch.nn.Module, DetectorHParams]:
    """
    Un seul point d'entrée pour reconstruire le détecteur en test.

    Résolution des hparams :
      - hparams_json fourni -> utilisé
      - sinon auto_hparams_json -> cherche hyperparameters.json à côté du ckpt
      - sinon ckpt["hparams"] si dispo
      - sinon defaults dataclass

    Support actuel: head_type == "fastrnn" uniquement.
    """
    from models.detection.fastrnn_detector import FastRNNDetector

    ckpt = _load_detector_ckpt(ckpt_path, map_location=str(dev))
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

    # merge
    if json_hp and ckpt_hp:
        merged = _merge_dicts(ckpt_hp, json_hp) if prefer_json_over_ckpt else _merge_dicts(json_hp, ckpt_hp)
    elif json_hp:
        merged = json_hp
    elif ckpt_hp:
        merged = ckpt_hp
    else:
        merged = {}

    hp = DetectorHParams.from_dict(merged)

    if inferred_json_path is not None:
        print(f"[detection_utils] auto hparams_json loaded: {inferred_json_path}")
    if hparams_json is not None:
        print(f"[detection_utils] hparams_json used: {Path(hparams_json)}")
    print(f"[detection_utils] head_type={hp.head_type}, feat_source={hp.feat_source}, num_classes={hp.num_classes}")

    head = str(hp.head_type).lower().strip()
    if head != "fastrnn":
        raise ValueError(
            f"[detection_utils] Only head_type='fastrnn' is supported for now, got '{hp.head_type}'."
        )

    # build backbone
    feat_source = str(hp.feat_source).lower().strip()
    if feat_source == "sem_resnet50":
        backbone, out_ch = _build_sem_resnet50_backbone(
            pretrained=bool(hp.sem_pretrained),
            return_layer=str(hp.det_sem_return_layer),
        )
    else:
        raise ValueError(
            f"[detection_utils] Unsupported feat_source='{hp.feat_source}' for now. "
            f"Supported: ['sem_resnet50']"
        )

    # freeze backbone
    if hp.freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False

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

    # load weights
    state = ckpt["model"]
    if not isinstance(state, dict):
        raise RuntimeError("[detection_utils] ckpt['model'] must be a state_dict dict.")
    model.load_state_dict(state, strict=bool(strict))
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
    """
    API UNIQUE de sortie:
      returns list[{"boxes","scores","labels"}] sur CPU.

    Pour FastRNNDetector:
      - le filtrage score_thresh est déjà dans le modèle (self.score_thresh),
        mais on permet un override optionnel ici.
    """
    imgs = _to_device_images(images, dev)
    imgs = [_ensure_chw_float01(x) for x in imgs]

    outs = model(imgs)  # attendu: list[dict]
    if not isinstance(outs, (list, tuple)) or (len(outs) > 0 and not isinstance(outs[0], dict)):
        raise RuntimeError(
            "[infer_batch] Expected torchvision-style outputs: list[dict(boxes, labels, scores)]. "
            f"Got type={type(outs)}."
        )

    out_cpu: List[Dict[str, torch.Tensor]] = []
    for o in outs:
        boxes = o.get("boxes", torch.empty((0, 4), device=dev)).detach()
        scores = o.get("scores", torch.empty((0,), device=dev)).detach()
        labels = o.get("labels", torch.empty((0,), device=dev, dtype=torch.long)).detach()

        # optional override filter
        if score_thresh is not None and scores.numel() > 0:
            keep = scores >= float(score_thresh)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

        # optional top-k
        if max_dets is not None and scores.numel() > int(max_dets):
            idx = torch.argsort(scores, descending=True)[: int(max_dets)]
            boxes = boxes[idx]
            scores = scores[idx]
            labels = labels[idx]

        out_cpu.append(
            {"boxes": boxes.cpu(), "scores": scores.cpu(), "labels": labels.cpu()}
        )
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
    """
    Suppose un dataset CocoDetection-like (data_loader.dataset.coco exist)
    et targets contenant image_id (dict avec "image_id").
    """
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
                raise RuntimeError(
                    "Target does not contain 'image_id'. "
                    "Ensure your dataset/collate returns a dict with image_id per image."
                )
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
        return {
            "AP": 0.0, "AP50": 0.0, "AP75": 0.0, "APs": 0.0, "APm": 0.0, "APl": 0.0,
            "AR1": 0.0, "AR10": 0.0, "AR100": 0.0,
        }

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
# Camera demo (unique)
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

        img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # CHW float01

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
