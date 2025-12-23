#!/usr/bin/env python3
# tests/test_fastrnn_detection.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CocoDetection

from tests.detection_utils import (
    build_detector_from_checkpoint,
    evaluate_coco_map,
    run_camera_demo,
    infer_batch,
)

# -------------------------------------------------------
# COCO dataset wrapper -> targets dict compatibles
# -------------------------------------------------------
class CocoDetectionWrapped(CocoDetection):
    """
    Wrap CocoDetection pour renvoyer:
      img: Tensor(C,H,W) float [0,1]
      target: dict { image_id, boxes(xyxy), labels }
    """
    def __init__(self, img_folder: str, ann_file: str, transforms=None):
        super().__init__(img_folder=img_folder, annFile=ann_file, transforms=None)
        self._tfm = transforms

    def __getitem__(self, idx: int):
        img, anns = super().__getitem__(idx)  # anns: list[dict] COCO

        # image_id côté COCO
        img_id = int(self.ids[idx])

        # convert PIL -> tensor float [0,1]
        if self._tfm is not None:
            img = self._tfm(img)

        # build boxes/labels
        boxes = []
        labels = []
        for a in anns:
            # COCO bbox: [x,y,w,h]
            x, y, w, h = a["bbox"]
            x1, y1, x2, y2 = x, y, x + w, y + h
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
    """
    DataLoader collate standard pour detection:
      images: list[Tensor(C,H,W)]
      targets: list[dict]
    """
    images, targets = list(zip(*batch))
    return list(images), list(targets)


# -------------------------------------------------------
# Optional: class names mapping (id->name) from COCO
# -------------------------------------------------------
def build_coco_label_map(dataset: CocoDetectionWrapped) -> Dict[int, str]:
    coco = dataset.coco
    cats = coco.loadCats(coco.getCatIds())
    return {int(c["id"]): str(c["name"]) for c in cats}


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    p = argparse.ArgumentParser("FastRNN detection test (COCOeval + camera)")

    # model
    p.add_argument("--ckpt", type=str, required=True, help="Checkpoint .pth (containing key 'model').")
    p.add_argument("--hparams_json", type=str, default=None, help="hyperparameters.json (optional).")
    p.add_argument("--device", type=str, default="cuda", help="cuda|cpu")

    # coco eval
    p.add_argument("--coco_img_root", type=str, default=None, help="COCO images root (val/test).")
    p.add_argument("--coco_ann", type=str, default=None, help="COCO annotations json.")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--eval_score_thresh", type=float, default=0.05)

    # camera
    p.add_argument("--camera", action="store_true", help="Run webcam demo.")
    p.add_argument("--cam_id", type=int, default=0)
    p.add_argument("--cam_score_thresh", type=float, default=0.3)
    p.add_argument("--cam_resize_h", type=int, default=None)
    p.add_argument("--cam_resize_w", type=int, default=None)
    p.add_argument("--cam_max_dets", type=int, default=200)

    args = p.parse_args()

    dev = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # 1) load model (FastRNN only)
    model, hp = build_detector_from_checkpoint(
        args.ckpt,
        dev,
        hparams_json=args.hparams_json,
        strict=True,
    )

    # 2) COCOeval (si args coco fournis)
    if args.coco_img_root and args.coco_ann:
        tfm = T.Compose([
            T.ToTensor(),  # PIL -> float [0,1]
            # NOTE: tu peux ajouter un Resize ici, mais attention: il faut alors aussi
            # resize les boxes GT. Donc par défaut: pas de resize.
        ])

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

    # 3) Camera demo
    if args.camera:
        resize_hw = None
        if args.cam_resize_h is not None and args.cam_resize_w is not None:
            resize_hw = (int(args.cam_resize_h), int(args.cam_resize_w))

        # map labels -> names si tu veux (ex: COCO)
        class_names = None
        if args.coco_img_root and args.coco_ann:
            # rebuild dataset just to extract names (cheap)
            ds_tmp = CocoDetectionWrapped(args.coco_img_root, args.coco_ann, transforms=T.ToTensor())
            class_names = build_coco_label_map(ds_tmp)

        run_camera_demo(
            model,
            dev,
            class_names=class_names,
            cam_id=int(args.cam_id),
            score_thresh=float(args.cam_score_thresh),
            resize_hw=resize_hw,
            max_dets=int(args.cam_max_dets),
        )


if __name__ == "__main__":
    main()
