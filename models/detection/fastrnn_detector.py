# models/detection/fastrnn_detector.py
import math
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms

from .fastrnn_det_head import FastRNNDetHead


def _batch_images(images: List[torch.Tensor], size_divisible: int = 32) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """
    Pad à droite/bas pour batcher.
    images: list of (C,H,W) in [0,1]
    returns: batch (B,C,Hmax,Wmax) + tailles originales [(H,W),...]
    """
    assert isinstance(images, (list, tuple)) and len(images) > 0
    sizes = [(img.shape[-2], img.shape[-1]) for img in images]
    max_h = max(h for h, w in sizes)
    max_w = max(w for h, w in sizes)

    # align
    if size_divisible > 1:
        max_h = int(math.ceil(max_h / size_divisible) * size_divisible)
        max_w = int(math.ceil(max_w / size_divisible) * size_divisible)

    B = len(images)
    C = images[0].shape[0]
    batch = images[0].new_zeros((B, C, max_h, max_w))
    for i, img in enumerate(images):
        h, w = img.shape[-2], img.shape[-1]
        batch[i, :, :h, :w] = img
    return batch, sizes


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    logits: (N,K)
    targets: (N,K) in {0,1}
    """
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


def _grid_centers(H: int, W: int, stride: float, device):
    ys = (torch.arange(H, device=device) + 0.5) * stride
    xs = (torch.arange(W, device=device) + 0.5) * stride
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return xx, yy  # both (H,W)


def _assign_targets_fcos_like(
    boxes: torch.Tensor,   # (M,4) xyxy
    labels: torch.Tensor,  # (M,) in [1..K]
    H: int,
    W: int,
    stride: float,
    device,
):
    """
    Assignation simple:
      - un point (cx,cy) est positif s'il est dans au moins une bbox
      - s'il est dans plusieurs, on prend la bbox de plus petite aire.
    Returns:
      cls_t: (H,W) label int (0=bg, 1..K)
      ltrb_t: (H,W,4) distances / stride
      ctr_t: (H,W) centerness [0,1]
      pos_mask: (H,W) bool
    """
    cls_t = torch.zeros((H, W), device=device, dtype=torch.long)
    ltrb_t = torch.zeros((H, W, 4), device=device, dtype=torch.float32)
    ctr_t = torch.zeros((H, W), device=device, dtype=torch.float32)
    pos_mask = torch.zeros((H, W), device=device, dtype=torch.bool)

    if boxes.numel() == 0:
        return cls_t, ltrb_t, ctr_t, pos_mask

    xx, yy = _grid_centers(H, W, stride, device=device)  # (H,W)
    pts = torch.stack([xx, yy], dim=-1).view(-1, 2)      # (S,2)
    S = pts.shape[0]

    # boxes: (M,4)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)  # (M,)

    px = pts[:, 0:1]  # (S,1)
    py = pts[:, 1:2]  # (S,1)

    l = px - x1.view(1, -1)
    t = py - y1.view(1, -1)
    r = x2.view(1, -1) - px
    b = y2.view(1, -1) - py

    inside = (l > 0) & (t > 0) & (r > 0) & (b > 0)  # (S,M)
    if inside.sum() == 0:
        return cls_t, ltrb_t, ctr_t, pos_mask

    # pour chaque point, choisir bbox d'aire min parmi celles valides
    areas_mat = areas.view(1, -1).repeat(S, 1)
    areas_mat = torch.where(inside, areas_mat, torch.full_like(areas_mat, float("inf")))
    best = areas_mat.argmin(dim=1)  # (S,)
    best_inside = inside[torch.arange(S, device=device), best]  # (S,)

    # gather ltrb de la bbox choisie
    l_best = l[torch.arange(S, device=device), best]
    t_best = t[torch.arange(S, device=device), best]
    r_best = r[torch.arange(S, device=device), best]
    b_best = b[torch.arange(S, device=device), best]

    # centerness
    lr_min = torch.min(l_best, r_best)
    lr_max = torch.max(l_best, r_best).clamp_min(1e-6)
    tb_min = torch.min(t_best, b_best)
    tb_max = torch.max(t_best, b_best).clamp_min(1e-6)
    ctr = torch.sqrt((lr_min / lr_max) * (tb_min / tb_max)).clamp(0, 1)

    # labels
    lab_best = labels[best]  # (S,)
    lab_best = torch.where(best_inside, lab_best, torch.zeros_like(lab_best))

    # fill maps
    cls_t = lab_best.view(H, W)
    pos_mask = best_inside.view(H, W)
    ltrb = torch.stack([l_best, t_best, r_best, b_best], dim=-1) / float(stride)
    ltrb_t = ltrb.view(H, W, 4)
    ctr_t = ctr.view(H, W)

    return cls_t, ltrb_t, ctr_t, pos_mask


class FastRNNDetector(nn.Module):
    """
    Wrapper type torchvision:
      - train: forward(images, targets) -> dict(loss_cls, loss_reg, loss_ctr)
      - eval : forward(images) -> list[{boxes, labels, scores}]
    """
    def __init__(
        self,
        backbone: nn.Module,          # returns features dict {"0": (B,C,H,W)} ou un tensor (B,C,H,W)
        out_channels: int,
        num_classes: int,             # inclut background
        head_hidden: int = 256,
        head_bidir: bool = True,
        head_dropout: float = 0.0,
        size_divisible: int = 32,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        topk: int = 1000,
    ):
        super().__init__()
        self.backbone = backbone
        self.size_divisible = int(size_divisible)
        self.num_classes = int(num_classes)
        self.K = self.num_classes - 1

        self.head = FastRNNDetHead(
            in_channels=out_channels,
            num_classes=num_classes,
            hidden_dim=head_hidden,
            bidir=head_bidir,
            dropout=head_dropout,
        )

        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.score_thresh = float(score_thresh)
        self.nms_thresh = float(nms_thresh)
        self.topk = int(topk)

    def _extract_feat(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if isinstance(feats, dict):
            # convention: key "0"
            if "0" in feats:
                return feats["0"]
            # sinon prendre la première clé
            return feats[list(feats.keys())[0]]
        return feats

    @torch.no_grad()
    def _infer_stride(self, img_hw: Tuple[int, int], feat_hw: Tuple[int, int]) -> float:
        # approx stride basé sur H (et W similaire)
        H, W = img_hw
        Hf, Wf = feat_hw
        return float(H) / float(Hf)

    @staticmethod
    def _sanitize_target_boxes_labels(
        t: Dict[str, torch.Tensor],
        device: torch.device,
        *,
        strict: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalise les targets pour éviter les cas courants qui cassent l'indexation:
          - boxes sous forme (4,) -> reshape en (1,4)
          - boxes vides -> (0,4)
          - labels scalaire -> (1,)
          - labels multi-dim -> flatten
        Attend des boxes au format xyxy.
        """
        if "boxes" not in t or "labels" not in t:
            if strict:
                raise KeyError(f"[FastRNNDetector] target must contain 'boxes' and 'labels'. Got keys={list(t.keys())}")
            # fallback: empty
            empty_boxes = torch.zeros((0, 4), device=device, dtype=torch.float32)
            empty_labels = torch.zeros((0,), device=device, dtype=torch.long)
            return empty_boxes, empty_labels

        gt_boxes = t["boxes"]
        gt_labels = t["labels"]

        # tensors / device
        if not torch.is_tensor(gt_boxes):
            gt_boxes = torch.as_tensor(gt_boxes, device=device)
        else:
            gt_boxes = gt_boxes.to(device)

        if not torch.is_tensor(gt_labels):
            gt_labels = torch.as_tensor(gt_labels, device=device)
        else:
            gt_labels = gt_labels.to(device)

        # empty
        if gt_boxes.numel() == 0:
            gt_boxes = gt_boxes.reshape(0, 4).to(dtype=torch.float32)
            gt_labels = gt_labels.reshape(0).to(dtype=torch.long)
            return gt_boxes, gt_labels

        # single box (4,) -> (1,4)
        if gt_boxes.dim() == 1 and gt_boxes.numel() == 4:
            gt_boxes = gt_boxes.view(1, 4)

        # enforce 2D (M,4)
        if gt_boxes.dim() != 2 or gt_boxes.shape[-1] != 4:
            msg = f"[FastRNNDetector] Bad gt_boxes shape: {tuple(gt_boxes.shape)} (expected (M,4))."
            if strict:
                raise ValueError(msg)
            # best-effort fallback
            gt_boxes = gt_boxes.reshape(-1, 4)

        # labels shape to (M,)
        if gt_labels.dim() == 0:
            gt_labels = gt_labels.view(1)
        elif gt_labels.dim() > 1:
            gt_labels = gt_labels.view(-1)

        # dtypes
        gt_boxes = gt_boxes.to(dtype=torch.float32)
        gt_labels = gt_labels.to(dtype=torch.long)

        # length consistency
        if gt_boxes.shape[0] != gt_labels.shape[0]:
            msg = f"[FastRNNDetector] Mismatch boxes/labels: boxes={tuple(gt_boxes.shape)}, labels={tuple(gt_labels.shape)}"
            if strict:
                raise ValueError(msg)
            m = min(gt_boxes.shape[0], gt_labels.shape[0])
            gt_boxes = gt_boxes[:m]
            gt_labels = gt_labels[:m]

        return gt_boxes, gt_labels

    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict[str, torch.Tensor]]] = None):
        assert isinstance(images, (list, tuple)), "images doit être une liste de tensors (C,H,W)."

        x, sizes = _batch_images(images, size_divisible=self.size_divisible)  # (B,C,Hmax,Wmax)
        feat = self._extract_feat(x)  # (B,C,Hf,Wf)
        cls_logits, reg_ltrb, ctr_logits = self.head(feat)  # (B,K,Hf,Wf), (B,4,Hf,Wf), (B,1,Hf,Wf)

        if targets is not None:
            return self._forward_train(cls_logits, reg_ltrb, ctr_logits, sizes, targets)
        else:
            return self._forward_infer(cls_logits, reg_ltrb, ctr_logits, sizes)

    def _forward_train(self, cls_logits, reg_ltrb, ctr_logits, sizes, targets):
        device = cls_logits.device
        B, K, Hf, Wf = cls_logits.shape

        # flatten
        cls_flat = cls_logits.permute(0, 2, 3, 1).reshape(B, Hf * Wf, K)     # (B,S,K)
        reg_flat = reg_ltrb.permute(0, 2, 3, 1).reshape(B, Hf * Wf, 4)       # (B,S,4)
        ctr_flat = ctr_logits.permute(0, 2, 3, 1).reshape(B, Hf * Wf)        # (B,S)

        loss_cls_all = []
        loss_reg_all = []
        loss_ctr_all = []

        for i in range(B):
            img_h, img_w = sizes[i]
            stride = self._infer_stride((img_h, img_w), (Hf, Wf))

            t = targets[i]

            # --- SANITIZE to avoid (4,) boxes etc. ---
            gt_boxes, gt_labels = self._sanitize_target_boxes_labels(t, device, strict=True)

            # clamp boxes within image
            gt_boxes = gt_boxes.clone()
            gt_boxes[:, 0::2] = gt_boxes[:, 0::2].clamp(0, img_w - 1)
            gt_boxes[:, 1::2] = gt_boxes[:, 1::2].clamp(0, img_h - 1)

            cls_t, ltrb_t, ctr_t, pos_mask = _assign_targets_fcos_like(
                gt_boxes, gt_labels, Hf, Wf, stride, device=device
            )

            # build classification targets: (S,K) one-hot (bg = all zeros)
            cls_t_flat = cls_t.view(-1)  # (S,)
            tgt_onehot = torch.zeros((Hf * Wf, K), device=device, dtype=torch.float32)
            pos_idx = torch.nonzero(cls_t_flat > 0, as_tuple=False).view(-1)
            if pos_idx.numel() > 0:
                tgt_onehot[pos_idx, (cls_t_flat[pos_idx] - 1)] = 1.0

            # focal cls
            loss_cls = sigmoid_focal_loss(
                cls_flat[i], tgt_onehot,
                alpha=self.focal_alpha, gamma=self.focal_gamma, reduction="mean"
            )

            # centerness loss (BCE) only on positives
            ctr_t_flat = ctr_t.view(-1)
            ctr_pred = ctr_flat[i]
            if pos_idx.numel() > 0:
                loss_ctr = F.binary_cross_entropy_with_logits(
                    ctr_pred[pos_idx], ctr_t_flat[pos_idx], reduction="mean"
                )
            else:
                loss_ctr = torch.tensor(0.0, device=device)

            # reg loss (SmoothL1) only on positives
            ltrb_t_flat = ltrb_t.view(-1, 4)
            reg_pred = reg_flat[i]
            if pos_idx.numel() > 0:
                loss_reg = F.smooth_l1_loss(reg_pred[pos_idx], ltrb_t_flat[pos_idx], reduction="mean")
            else:
                loss_reg = torch.tensor(0.0, device=device)

            loss_cls_all.append(loss_cls)
            loss_reg_all.append(loss_reg)
            loss_ctr_all.append(loss_ctr)

        loss_cls = torch.stack(loss_cls_all).mean()
        loss_reg = torch.stack(loss_reg_all).mean()
        loss_ctr = torch.stack(loss_ctr_all).mean()

        return {
            "loss_classifier": loss_cls,
            "loss_box_reg": loss_reg,
            "loss_centerness": loss_ctr,
        }

    @torch.no_grad()
    def _forward_infer(self, cls_logits, reg_ltrb, ctr_logits, sizes):
        device = cls_logits.device
        B, K, Hf, Wf = cls_logits.shape
        results = []

        # per image
        for i in range(B):
            img_h, img_w = sizes[i]
            stride = self._infer_stride((img_h, img_w), (Hf, Wf))

            cls = torch.sigmoid(cls_logits[i])  # (K,Hf,Wf)
            ctr = torch.sigmoid(ctr_logits[i, 0])  # (Hf,Wf)
            reg = reg_ltrb[i]  # (4,Hf,Wf) distances in strides

            # score per class
            scores = cls * ctr.unsqueeze(0)  # (K,Hf,Wf)

            # topk
            scores_flat = scores.view(-1)  # K*Hf*Wf
            num = min(self.topk, scores_flat.numel())
            top_scores, top_idx = torch.topk(scores_flat, k=num, largest=True, sorted=True)

            keep = top_scores > self.score_thresh
            top_scores = top_scores[keep]
            top_idx = top_idx[keep]

            if top_scores.numel() == 0:
                results.append({
                    "boxes": torch.zeros((0, 4), device=device),
                    "labels": torch.zeros((0,), device=device, dtype=torch.long),
                    "scores": torch.zeros((0,), device=device),
                })
                continue

            # decode indices
            # index in flattened: idx = c*(Hf*Wf) + p
            HW = Hf * Wf
            cls_id = (top_idx // HW).to(torch.long)          # 0..K-1
            pos_id = (top_idx % HW).to(torch.long)           # 0..HW-1
            py = (pos_id // Wf).to(torch.long)
            px = (pos_id % Wf).to(torch.long)

            # centers
            cx = (px.float() + 0.5) * stride
            cy = (py.float() + 0.5) * stride

            # ltrb distances
            l = reg[0, py, px] * stride
            t = reg[1, py, px] * stride
            r = reg[2, py, px] * stride
            b = reg[3, py, px] * stride

            x1 = (cx - l).clamp(0, img_w - 1)
            y1 = (cy - t).clamp(0, img_h - 1)
            x2 = (cx + r).clamp(0, img_w - 1)
            y2 = (cy + b).clamp(0, img_h - 1)
            boxes = torch.stack([x1, y1, x2, y2], dim=-1)

            # labels: +1 car 0=bg
            labels = cls_id + 1

            # NMS class-agnostic (simple). Tu peux faire par-classe si tu veux.
            keep_nms = nms(boxes, top_scores, self.nms_thresh)
            boxes = boxes[keep_nms]
            labels = labels[keep_nms]
            scores_out = top_scores[keep_nms]

            results.append({"boxes": boxes, "labels": labels, "scores": scores_out})

        return results