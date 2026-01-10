#!/usr/bin/env python3
# test_fastrnn_detection.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CocoDetection

from testsFile.detectionUtils import (
    build_detector_from_checkpoint,
    evaluate_coco_map,
    infer_batch,
)

# -------------------------------------------------------
# COCO dataset wrapper -> targets dict compatibles
# -------------------------------------------------------
class CocoDetectionWrapped(CocoDetection):
    def __init__(self, img_folder: str, ann_file: str, transforms=None):
        try:
            super().__init__(img_folder=img_folder, ann_file=ann_file, transforms=None)
        except TypeError:
            super().__init__(img_folder, ann_file, transforms=None)
        self._tfm = transforms

    def __getitem__(self, idx: int):
        img, anns = super().__getitem__(idx)
        img_id = int(self.ids[idx])

        if self._tfm is not None:
            img = self._tfm(img)

        boxes: List[List[float]] = []
        labels: List[int] = []
        for a in anns:
            x, y, w, h = a["bbox"]
            x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)
            boxes.append([x1, y1, x2, y2])
            labels.append(int(a["category_id"]))

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.long)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.long)

        target = {
            "image_id": torch.tensor(img_id, dtype=torch.int64),
            "boxes": boxes_t,
            "labels": labels_t,
        }
        return img, target


def collate_detection(batch: List[Tuple[torch.Tensor, Dict[str, Any]]]):
    images, targets = list(zip(*batch))
    return list(images), list(targets)


def build_coco_label_map(dataset: CocoDetectionWrapped) -> Dict[int, str]:
    coco = dataset.coco
    cats = coco.loadCats(coco.getCatIds())
    return {int(c["id"]): str(c["name"]) for c in cats}


# -------------------------------------------------------
# Post-process outputs for visualization (reduce clutter)
# -------------------------------------------------------
def postprocess_for_vis(
    out: Dict[str, torch.Tensor],
    *,
    topk: int = 20,
    topk_per_class: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    boxes = out["boxes"]
    scores = out["scores"]
    labels = out["labels"]

    if scores.numel() == 0:
        return out

    idx = torch.argsort(scores, descending=True)
    boxes = boxes[idx]
    scores = scores[idx]
    labels = labels[idx]

    if topk is not None and scores.numel() > int(topk):
        boxes = boxes[: int(topk)]
        scores = scores[: int(topk)]
        labels = labels[: int(topk)]

    if topk_per_class is not None:
        keep_idx: List[int] = []
        seen: Dict[int, int] = {}
        for i, lab in enumerate(labels.tolist()):
            seen[lab] = seen.get(lab, 0) + 1
            if seen[lab] <= int(topk_per_class):
                keep_idx.append(i)

        if len(keep_idx) == 0:
            return {"boxes": boxes[:0], "scores": scores[:0], "labels": labels[:0]}

        k = torch.tensor(keep_idx, dtype=torch.long)
        boxes = boxes[k]
        scores = scores[k]
        labels = labels[k]

    return {"boxes": boxes, "scores": scores, "labels": labels}


def _try_import_cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except Exception:
        return None


def save_predicted_images(
    *,
    model: torch.nn.Module,
    dataset: CocoDetectionWrapped,
    dev: torch.device,
    save_dir: Path,
    class_names: Optional[Dict[int, str]] = None,
    score_thresh: float = 0.3,
    max_images: Optional[int] = None,
    max_dets: int = 200,
    vis_topk: int = 20,
    vis_topk_per_class: Optional[int] = None,
) -> int:
    cv2 = _try_import_cv2()
    if cv2 is None:
        raise ImportError("OpenCV required to save images (pip install opencv-python)")

    save_dir.mkdir(parents=True, exist_ok=True)

    n_saved = 0
    n_total = len(dataset) if max_images is None else min(int(max_images), len(dataset))

    for i in range(n_total):
        img_id = int(dataset.ids[i])
        pil = dataset._load_image(img_id)
        img_t = T.ToTensor()(pil)

        out = infer_batch(
            model,
            images=[img_t],
            dev=dev,
            score_thresh=float(score_thresh),
            max_dets=int(max_dets),
        )[0]

        out = postprocess_for_vis(out, topk=int(vis_topk), topk_per_class=vis_topk_per_class)

        rgb = (img_t.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype("uint8")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        boxes = out["boxes"].numpy()
        scores = out["scores"].numpy()
        labels = out["labels"].numpy()

        for (x1, y1, x2, y2), sc, lab in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            name = class_names.get(int(lab), str(int(lab))) if class_names else str(int(lab))
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                bgr,
                f"{name}:{sc:.2f}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        out_path = save_dir / f"pred_{img_id:012d}.jpg"
        cv2.imwrite(str(out_path), bgr)
        n_saved += 1

    return n_saved


@torch.no_grad()
def run_camera_demo_local(
    model: torch.nn.Module,
    dev: torch.device,
    *,
    class_names: Optional[Dict[int, str]] = None,
    cam_id: int = 0,
    score_thresh: float = 0.3,
    resize_hw: Optional[Tuple[int, int]] = None,
    max_dets: int = 200,
    vis_topk: int = 15,
    vis_topk_per_class: Optional[int] = None,
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

        out = postprocess_for_vis(out, topk=int(vis_topk), topk_per_class=vis_topk_per_class)

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


def write_json_summary(
    *,
    save_dir: Path,
    ckpt: str,
    hparams_json: Optional[str],
    head: str,
    device: str,
    coco_img_root: Optional[str],
    coco_ann: Optional[str],
    batch_size: int,
    num_workers: int,
    max_images: Optional[int],
    eval_score_thresh: float,
    metrics: Optional[Dict[str, float]],
    saved_images: int,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "ckpt": ckpt,
        "hparams_json": hparams_json,
        "head": head,
        "device": device,
        "coco_img_root": coco_img_root,
        "coco_ann": coco_ann,
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "max_images": None if max_images is None else int(max_images),
        "eval_score_thresh": float(eval_score_thresh),
        "metrics": metrics,
        "saved_images": int(saved_images),
    }
    out_path = save_dir / "eval_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return out_path


def _make_temp_hparams_with_head_override(
    *,
    original_hparams_json: Optional[str],
    head: str,
    save_dir: Path,
) -> Optional[str]:
    head = str(head).lower().strip()
    if head not in {"fastrnn", "fasterrcnn"}:
        raise ValueError(f"--head must be in {{fastrnn,fasterrcnn}}, got {head}")

    save_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = save_dir / f"_tmp_hparams_override_head_{head}.json"

    base: Dict[str, Any] = {}
    if original_hparams_json is not None:
        p = Path(original_hparams_json)
        if p.is_file():
            with p.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                base = obj

    base["head_type"] = head

    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(base, f, indent=2)

    return str(tmp_path)


def main():
    p = argparse.ArgumentParser("Detection test (COCOeval + camera)")

    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--hparams_json", type=str, default=None)
    p.add_argument("--head", type=str, default=None, choices=["fastrnn", "fasterrcnn"],
                   help="Force head (otherwise auto-detected from checkpoint).")
    p.add_argument("--strict", action="store_true", help="Load ckpt with strict=True (default False).")
    p.add_argument("--device", type=str, default="cuda", help="cuda|cpu")

    p.add_argument("--coco_img_root", type=str, default=None)
    p.add_argument("--coco_ann", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--eval_score_thresh", type=float, default=0.05)

    p.add_argument("--show_or_save_images", action="store_true")
    p.add_argument("--save_dir", type=str, default="./runs/test_detection")
    p.add_argument("--vis_score_thresh", type=float, default=0.3)
    p.add_argument("--vis_max_dets", type=int, default=200)
    p.add_argument("--vis_topk", type=int, default=15)
    p.add_argument("--vis_topk_per_class", type=int, default=None)

    p.add_argument("--camera", action="store_true")
    p.add_argument("--cam_id", type=int, default=0)
    p.add_argument("--cam_score_thresh", type=float, default=0.3)
    p.add_argument("--cam_resize_h", type=int, default=None)
    p.add_argument("--cam_resize_w", type=int, default=None)
    p.add_argument("--cam_max_dets", type=int, default=200)
    p.add_argument("--cam_topk", type=int, default=10)
    p.add_argument("--cam_topk_per_class", type=int, default=None)

    args = p.parse_args()

    dev = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    save_dir = Path(args.save_dir)

    hparams_json_for_load = args.hparams_json
    if args.head is not None:
        hparams_json_for_load = _make_temp_hparams_with_head_override(
            original_hparams_json=args.hparams_json,
            head=args.head,
            save_dir=save_dir,
        )
        print(f"[test] --head override -> using temp hparams: {hparams_json_for_load}")

    model, hp = build_detector_from_checkpoint(
        args.ckpt,
        dev,
        hparams_json=hparams_json_for_load,
        strict=bool(args.strict),
    )

    head_used = str(args.head if args.head is not None else hp.head_type)
    print(f"[test] Loaded detector head: {head_used}")

    metrics: Optional[Dict[str, float]] = None
    saved_images = 0

    if args.coco_img_root and args.coco_ann:
        tfm = T.Compose([T.ToTensor()])
        ds = CocoDetectionWrapped(args.coco_img_root, args.coco_ann, transforms=tfm)

        loader = DataLoader(
            ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            collate_fn=collate_detection,
        )

        metrics = evaluate_coco_map(
            model,
            loader,
            dev,
            score_thresh=float(args.eval_score_thresh),
            max_images=args.max_images,
        )

        print("\n==== COCOeval metrics ====")
        for k, v in metrics.items():
            print(f"{k:>6s}: {v:.6f}")

        if args.show_or_save_images:
            class_names = build_coco_label_map(ds)
            img_out_dir = save_dir / "images"
            saved_images = save_predicted_images(
                model=model,
                dataset=ds,
                dev=dev,
                save_dir=img_out_dir,
                class_names=class_names,
                score_thresh=float(args.vis_score_thresh),
                max_images=args.max_images,
                max_dets=int(args.vis_max_dets),
                vis_topk=int(args.vis_topk),
                vis_topk_per_class=args.vis_topk_per_class,
            )
            print(f"[VIS] saved {saved_images} images to: {img_out_dir}")

    summary_path = write_json_summary(
        save_dir=save_dir,
        ckpt=args.ckpt,
        hparams_json=args.hparams_json,
        head=head_used,
        device=str(dev),
        coco_img_root=args.coco_img_root,
        coco_ann=args.coco_ann,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        max_images=args.max_images,
        eval_score_thresh=float(args.eval_score_thresh),
        metrics=metrics,
        saved_images=saved_images,
    )
    print(f"[JSON] summary written to: {summary_path}")

    if args.camera:
        resize_hw: Optional[Tuple[int, int]] = None
        if args.cam_resize_h is not None and args.cam_resize_w is not None:
            resize_hw = (int(args.cam_resize_h), int(args.cam_resize_w))

        class_names = None
        if args.coco_img_root and args.coco_ann:
            ds_tmp = CocoDetectionWrapped(args.coco_img_root, args.coco_ann, transforms=T.ToTensor())
            class_names = build_coco_label_map(ds_tmp)

        run_camera_demo_local(
            model,
            dev,
            class_names=class_names,
            cam_id=int(args.cam_id),
            score_thresh=float(args.cam_score_thresh),
            resize_hw=resize_hw,
            max_dets=int(args.cam_max_dets),
            vis_topk=int(args.cam_topk),
            vis_topk_per_class=args.cam_topk_per_class,
        )


if __name__ == "__main__":
    main()
