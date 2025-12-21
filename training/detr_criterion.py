# training/detr_criterion.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou(boxes1, boxes2):
    """
    boxes1: [N,4], boxes2: [M,4] en xyxy
    retourne iou [N,M]
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    return iou


def generalized_iou(boxes1, boxes2):
    """
    GIoU entre 2 sets d'anchor [N,4] et [N,4] en xyxy
    """
    assert boxes1.shape == boxes2.shape
    x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.max(boxes1[:, 3], boxes2[:, 3])

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    union = area1 + area2 - inter + 1e-6
    iou = inter / union

    area_c = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0) + 1e-6
    giou = iou - (area_c - union) / area_c
    return giou


class GreedyMatcher:
    """
    Matcher très simple : pour chaque GT, prend la meilleure query dispo (IoU).
    Pas un vrai Hungarian, mais suffisant pour ton expérimentation style vs contenu.
    """

    def __call__(self, pred_boxes, targets):
        """
        pred_boxes : (B,Q,4) en cxcywh [0,1]
        targets    : liste de dicts avec 'boxes' en xyxy (pixels)

        Retourne list[ (idx_pred, idx_gt) ] pour chaque batch.
        """
        indices = []
        B, Q, _ = pred_boxes.shape
        for b in range(B):
            pb = box_cxcywh_to_xyxy(pred_boxes[b])  # (Q,4) en [0,1]
            tgt = targets[b]
            boxes_gt = tgt["boxes"]
            if boxes_gt.numel() == 0:
                indices.append((torch.empty(0, dtype=torch.int64),
                                torch.empty(0, dtype=torch.int64)))
                continue

            # normaliser les GT en [0,1] si tu veux ; ici on suppose que tu as
            # redimensionné les images à taille fixe (H,W) pour l'entraînement.
            # On peut approx : on met tout en [0,1] avec un bounding global.
            h = max(boxes_gt[:, 3].max().item(), 1.0)
            w = max(boxes_gt[:, 2].max().item(), 1.0)
            boxes_gt_norm = boxes_gt.clone()
            boxes_gt_norm[:, 0::2] /= w
            boxes_gt_norm[:, 1::2] /= h

            iou_mat = box_iou(pb, boxes_gt_norm)  # (Q,Ng)
            # greedy : on match GT par GT
            q_idx_list = []
            g_idx_list = []
            used_q = set()
            for g_idx in range(boxes_gt_norm.size(0)):
                ious = iou_mat[:, g_idx]
                best_q = int(torch.argmax(ious).item())
                if best_q in used_q:
                    continue
                used_q.add(best_q)
                q_idx_list.append(best_q)
                g_idx_list.append(g_idx)

            if len(q_idx_list) == 0:
                indices.append((torch.empty(0, dtype=torch.int64),
                                torch.empty(0, dtype=torch.int64)))
            else:
                indices.append((torch.tensor(q_idx_list, dtype=torch.int64),
                                torch.tensor(g_idx_list, dtype=torch.int64)))
        return indices


class SetCriterionDETR(nn.Module):
    """
    Pertes type DETR :
      - CE sur classes (avec classe fond)
      - L1 sur boxes matched
      - GIoU sur boxes matched
    """

    def __init__(
        self,
        num_classes: int,
        matcher=None,
        weight_dict=None,
        eos_coef: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher if matcher is not None else GreedyMatcher()
        self.weight_dict = weight_dict or {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
        self.eos_coef = eos_coef

        empty_weight = torch.ones(num_classes)
        empty_weight[0] = eos_coef  # classe 0 = fond
        self.register_buffer("empty_weight", empty_weight)

    def forward(self, outputs, targets):
        """
        outputs : dict avec
          - "pred_logits": (B,Q,num_classes)
          - "pred_boxes" : (B,Q,4) en cxcywh [0,1]
        targets : liste de dicts avec "labels" (Ng,), "boxes" (Ng,4)
        """
        logits = outputs["pred_logits"]  # (B,Q,C)
        boxes = outputs["pred_boxes"]    # (B,Q,4)

        B, Q, C = logits.shape

        indices = self.matcher(boxes, targets)  # list[(idx_q, idx_g)]

        # ====== Class loss ======
        target_classes = logits.new_zeros((B, Q), dtype=torch.long)
        target_classes.fill_(0)  # fond

        for b, (idx_q, idx_g) in enumerate(indices):
            if idx_q.numel() == 0:
                continue
            tgt_labels = targets[b]["labels"][idx_g]
            target_classes[b, idx_q] = tgt_labels

        loss_ce = F.cross_entropy(
            logits.transpose(1, 2),
            target_classes,
            weight=self.empty_weight.to(logits.device),
        )

        # ====== Box + GIoU loss ======
        loss_bbox = logits.new_tensor(0.0)
        loss_giou = logits.new_tensor(0.0)
        n_matched = 0

        for b, (idx_q, idx_g) in enumerate(indices):
            if idx_q.numel() == 0:
                continue
            pb = boxes[b, idx_q]  # (Nm,4) cxcywh
            pb_xyxy = box_cxcywh_to_xyxy(pb)

            tgt = targets[b]["boxes"][idx_g]  # (Nm,4) xyxy pixels
            # même normalisation que dans matcher
            h = max(tgt[:, 3].max().item(), 1.0)
            w = max(tgt[:, 2].max().item(), 1.0)
            tgt_norm = tgt.clone()
            tgt_norm[:, 0::2] /= w
            tgt_norm[:, 1::2] /= h

            loss_bbox = loss_bbox + F.l1_loss(pb_xyxy, tgt_norm, reduction="sum")
            giou = generalized_iou(pb_xyxy, tgt_norm)
            loss_giou = loss_giou + (1.0 - giou).sum()
            n_matched += pb.shape[0]

        if n_matched > 0:
            loss_bbox = loss_bbox / n_matched
            loss_giou = loss_giou / n_matched

        losses = {
            "loss_ce": loss_ce,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
        }
        return losses
