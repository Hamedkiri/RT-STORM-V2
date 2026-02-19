#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Safe normalization layers.

PyTorch's InstanceNorm2d raises:
  "Expected more than 1 spatial element when training"
when the input has spatial size 1x1.

This can happen when UNet/style encoders are deepened (more downsampling) or
when training with small crops.

We keep InstanceNorm behaviour for normal feature maps, but automatically
fall back to GroupNorm when H*W == 1 during training.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _pick_gn_groups(c: int, max_groups: int = 32) -> int:
    """Pick a GroupNorm group count that divides channels."""
    g = min(max_groups, c)
    while g > 1 and (c % g) != 0:
        g -= 1
    return max(1, g)


class SafeInstanceNorm2d(nn.Module):
    """InstanceNorm2d that falls back to GroupNorm for 1x1 spatial tensors."""

    def __init__(
        self,
        num_features: int,
        *,
        affine: bool = True,
        eps: float = 1e-5,
        momentum: float = 0.1,
        track_running_stats: bool = False,
        gn_groups: int | None = None,
    ) -> None:
        super().__init__()
        self.inorm = nn.InstanceNorm2d(
            num_features,
            affine=affine,
            eps=eps,
            momentum=momentum,
            track_running_stats=track_running_stats,
        )
        groups = _pick_gn_groups(num_features) if gn_groups is None else max(1, int(gn_groups))
        # GroupNorm is stable for any spatial size (including 1x1)
        self.gnorm = nn.GroupNorm(groups, num_features, eps=eps, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.dim() == 4:
            h, w = int(x.shape[-2]), int(x.shape[-1])
            if h * w <= 1:
                return self.gnorm(x)
        return self.inorm(x)
