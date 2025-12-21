
# file: models/semantic_moco_jepa.py
"""
Semantic content branch (ResNet50 + MoCo + optional JEPA-content).

Goal
----
Learn a *semantic* content representation invariant to style using (x, far) as
positive views, while keeping the existing rendering branch (UNet+PatchNCE)
unchanged.

Key design choices
------------------
- Images given to this module are expected in the repo's default range [-1, 1]
  (after Normalize([0.5],[0.5])).
- This module handles:
    (1) optional semantic augmentations (on [0,1] tensors),
    (2) ImageNet normalization for ResNet50,
    (3) MoCo queue and momentum encoder,
    (4) optional JEPA-content loss on token grids (8x8 for 256px inputs).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from torchvision import transforms

from models.jepa import TokenJEPA


# -------------------------- utils: normalization --------------------------

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _to_01(x: torch.Tensor) -> torch.Tensor:
    # x in [-1,1] -> [0,1]
    return (x * 0.5 + 0.5).clamp(0.0, 1.0)


def _to_m11(x01: torch.Tensor) -> torch.Tensor:
    # x in [0,1] -> [-1,1]
    return (x01 * 2.0 - 1.0).clamp(-1.0, 1.0)


def _imagenet_norm(x01: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(_IMAGENET_MEAN, device=x01.device, dtype=x01.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=x01.device, dtype=x01.dtype).view(1, 3, 1, 1)
    return (x01 - mean) / std


def _l2n(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


# -------------------------- batch augmentations --------------------------

@dataclass
class SemAugConfig:
    use_aug: bool = False
    crop: int = 224
    min_scale: float = 0.5
    color_jitter: float = 0.4
    gray_p: float = 0.2
    blur_p: float = 0.1


class _BatchAugmenter(nn.Module):
    """
    Torchvision transforms operate on single image tensors (C,H,W).
    We apply them per-sample. This is slower than kornia, but keeps deps minimal.
    """
    def __init__(self, cfg: SemAugConfig):
        super().__init__()
        self.cfg = cfg
        if not cfg.use_aug:
            self.tf = None
            return

        cj = transforms.ColorJitter(
            brightness=cfg.color_jitter,
            contrast=cfg.color_jitter,
            saturation=cfg.color_jitter,
            hue=min(0.1, cfg.color_jitter / 4.0),
        )

        self.tf = transforms.Compose([
            transforms.RandomResizedCrop(cfg.crop, scale=(cfg.min_scale, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([cj], p=0.8),
            transforms.RandomGrayscale(p=cfg.gray_p),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=cfg.blur_p),
        ])

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        if self.tf is None:
            return x01
        # x01: (B,3,H,W) in [0,1]
        out = []
        for i in range(x01.size(0)):
            out.append(self.tf(x01[i]))
        return torch.stack(out, dim=0)


# -------------------------- backbone --------------------------

class ResNet50Backbone(nn.Module):
    """ResNet50 trunk up to layer4, outputs a spatial feature map."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = tvm.ResNet50_Weights.IMAGENET1K_V2
            except Exception:
                try:
                    weights = tvm.ResNet50_Weights.DEFAULT
                except Exception:
                    weights = None
        m = tvm.resnet50(weights=weights)
        # keep conv1..layer4
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4
        self.out_channels = 2048

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        # x_norm: imagenet normalized, (B,3,H,W)
        x = self.stem(x_norm)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # (B,2048,H/32,W/32)
        return x


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hid = int(hidden_dim or max(out_dim * 2, in_dim))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------- MoCo + JEPA module --------------------------

class SemanticMoCoJEPA(nn.Module):
    """
    Query encoder is trained with gradients; key encoder is EMA-updated.

    Outputs:
      - global embedding z (for MoCo) of dimension `dim`
      - token grid T (for JEPA) with shape (B, S, tok_dim), where S = 1 + Hs*Ws (tokG + spatial).
    """
    def __init__(
        self,
        *,
        dim: int = 256,
        tok_dim: int = 256,
        queue_size: int = 65536,
        m: float = 0.999,
        T: float = 0.2,
        pretrained: bool = True,
        aug_cfg: Optional[SemAugConfig] = None,
        img_size: int = 256,
        # JEPA predictor config
        jepa_use: bool = False,
        jepa_hidden_mult: int = 2,
        jepa_heads: int = 4,
        jepa_norm: int = 1,
        jepa_var: float = 0.05,
        jepa_cov: float = 0.05,
    ):
        super().__init__()
        self.dim = int(dim)
        self.tok_dim = int(tok_dim)
        self.K = int(queue_size)
        self.m = float(m)
        self.T = float(T)

        self.augmenter = _BatchAugmenter(aug_cfg or SemAugConfig(use_aug=False))

        # Encoders
        self.backbone_q = ResNet50Backbone(pretrained=pretrained)
        self.backbone_k = ResNet50Backbone(pretrained=pretrained)

        # Token projections (1x1 conv) + global projection for MoCo
        self.tok_proj_q = nn.Conv2d(self.backbone_q.out_channels, self.tok_dim, kernel_size=1, bias=False)
        self.tok_proj_k = nn.Conv2d(self.backbone_k.out_channels, self.tok_dim, kernel_size=1, bias=False)

        self.proj_q = MLP(self.tok_dim, self.dim, hidden_dim=self.tok_dim * 2)
        self.proj_k = MLP(self.tok_dim, self.dim, hidden_dim=self.tok_dim * 2)

        # Init key = query
        self._copy_q_to_k()

        # MoCo queue (K, dim)
        self.register_buffer("queue", F.normalize(torch.randn(self.K, self.dim), dim=1))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Optional JEPA
        self.jepa_use = bool(jepa_use)
        self.jepa = None
        if self.jepa_use:
            # Token grid size depends on input resolution (stride=32)
            hs = max(1, int(img_size) // 32)
            S = 1 + hs * hs
            self.jepa = TokenJEPA(
                S=S,
                D=self.tok_dim,
                hidden_mult=int(jepa_hidden_mult),
                heads=int(jepa_heads),
                use_norm=bool(jepa_norm),
                var_lambda=float(jepa_var),
                cov_lambda=float(jepa_cov),
            )

        # Freeze key params (EMA updated)
        for p in self.backbone_k.parameters():
            p.requires_grad_(False)
        for p in self.tok_proj_k.parameters():
            p.requires_grad_(False)
        for p in self.proj_k.parameters():
            p.requires_grad_(False)

    # ---------------- internals ----------------

    @torch.no_grad()
    def _copy_q_to_k(self):
        for mq, mk in zip(self.backbone_q.parameters(), self.backbone_k.parameters()):
            mk.data.copy_(mq.data)
        for mq, mk in zip(self.tok_proj_q.parameters(), self.tok_proj_k.parameters()):
            mk.data.copy_(mq.data)
        for mq, mk in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            mk.data.copy_(mq.data)

    @torch.no_grad()
    def momentum_update(self, m: Optional[float] = None):
        """EMA update of key encoder parameters."""
        mm = float(self.m if m is None else m)
        for mq, mk in zip(self.backbone_q.parameters(), self.backbone_k.parameters()):
            mk.data.mul_(mm).add_(mq.data, alpha=1.0 - mm)
        for mq, mk in zip(self.tok_proj_q.parameters(), self.tok_proj_k.parameters()):
            mk.data.mul_(mm).add_(mq.data, alpha=1.0 - mm)
        for mq, mk in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            mk.data.mul_(mm).add_(mq.data, alpha=1.0 - mm)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """
        keys: (B, dim) normalized
        """
        B = keys.size(0)
        K = self.K
        ptr = int(self.queue_ptr.item())
        if B >= K:
            self.queue.copy_(keys[-K:])
            ptr = 0
        else:
            end = ptr + B
            if end <= K:
                self.queue[ptr:end] = keys
            else:
                first = K - ptr
                self.queue[ptr:] = keys[:first]
                self.queue[:end - K] = keys[first:]
            ptr = (ptr + B) % K
        self.queue_ptr[0] = ptr

    def _prep(self, x_m11: torch.Tensor, *, apply_aug: bool) -> torch.Tensor:
        """
        x_m11: (B,3,H,W) in [-1,1]
        Returns imagenet normalized tensor for ResNet.
        """
        x01 = _to_01(x_m11)
        if apply_aug:
            x01 = self.augmenter(x01)
        return _imagenet_norm(x01)

    def _tokens_from_fmap(self, fmap: torch.Tensor, proj: nn.Conv2d) -> torch.Tensor:
        """
        fmap: (B,C,Hs,Ws)
        Returns tokens (B, 1+Hs*Ws, tok_dim) with tokG prepended.
        """
        t = proj(fmap)  # (B,tok_dim,Hs,Ws)
        B, D, Hs, Ws = t.shape
        spatial = t.flatten(2).transpose(1, 2)  # (B, S=Hs*Ws, D)
        tokG = t.mean(dim=(2, 3), keepdim=False)  # (B, D)
        tokG = tokG.unsqueeze(1)  # (B,1,D)
        return torch.cat([tokG, spatial], dim=1)  # (B,1+Hs*Ws,D)

    def encode_q(self, x_m11: torch.Tensor, *, apply_aug: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (z_q, tokens_q)."""
        x = self._prep(x_m11, apply_aug=apply_aug)
        fmap = self.backbone_q(x)
        tokens = self._tokens_from_fmap(fmap, self.tok_proj_q)
        # global embedding from tokG -> MLP
        z = self.proj_q(tokens[:, 0, :])
        z = _l2n(z)
        return z, tokens

    @torch.no_grad()
    def encode_k(self, x_m11: torch.Tensor, *, apply_aug: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (z_k, tokens_k) with no grad."""
        x = self._prep(x_m11, apply_aug=apply_aug)
        fmap = self.backbone_k(x)
        tokens = self._tokens_from_fmap(fmap, self.tok_proj_k)
        z = self.proj_k(tokens[:, 0, :])
        z = _l2n(z)
        return z, tokens

    # ---------------- losses ----------------

    def moco_logits(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q: (B,dim) normalized
        k: (B,dim) normalized (no grad)
        Returns logits (B, 1+K) and labels (B,) with positives at index 0.
        """
        # positive
        l_pos = torch.einsum("bd,bd->b", [q, k]).unsqueeze(1)  # (B,1)
        # negatives from queue
        l_neg = torch.einsum("bd,kd->bk", [q, self.queue.detach()])  # (B,K)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return logits, labels

    def loss_moco(self, im_q: torch.Tensor, im_k: torch.Tensor, *, apply_aug: bool = True) -> Tuple[torch.Tensor, Dict]:
        """
        One-direction MoCo loss: q from im_q (grad), k from im_k (no grad).
        Updates queue (enqueue keys) and momentum encoder (must be called externally).
        """
        q, _ = self.encode_q(im_q, apply_aug=apply_aug)
        with torch.no_grad():
            k, _ = self.encode_k(im_k, apply_aug=apply_aug)
        logits, labels = self.moco_logits(q, k)
        loss = F.cross_entropy(logits, labels)
        # enqueue
        with torch.no_grad():
            self._dequeue_and_enqueue(k)
        stats = {
            "loss": float(loss.item()),
            "logits_pos": float(logits[:, 0].mean().item()),
            "logits_neg": float(logits[:, 1:].mean().item()),
        }
        return loss, stats

    def loss_jepa(self,
                  im_student: torch.Tensor,
                  im_teacher: torch.Tensor,
                  mask_ratio: float = 0.6,
                  *,
                  apply_aug: bool = True) -> Tuple[torch.Tensor, Dict]:
        """
        JEPA-content on token grids: student predicts teacher tokens on masked positions.
        """
        if not self.jepa_use or self.jepa is None:
            z = im_student.new_tensor(0.0)
            return z, {"loss": 0.0}
        _, Ts = self.encode_q(im_student, apply_aug=apply_aug)
        with torch.no_grad():
            _, Tt = self.encode_k(im_teacher, apply_aug=apply_aug)
        B, S, D = Ts.shape
        # random mask (B,S), avoid masking tokG too much by keeping it always visible
        mask = (torch.rand(B, S, device=Ts.device) < float(mask_ratio))
        mask[:, 0] = False  # keep global token visible
        loss, info = self.jepa(Ts, Tt, mask, w=None)
        return loss, info
