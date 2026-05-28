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


class LegacySafeInstanceNorm2d(nn.Module):
    """Backward-compatible safe norm for legacy checkpoints.

    Legacy checkpoints used nn.InstanceNorm2d directly in Sequential blocks and
    therefore store parameters as "...net.1.weight" / "...net.1.bias".

    Newer SafeInstanceNorm2d introduces submodules (inorm/gnorm), changing keys to
    "...net.1.inorm.weight", etc., which breaks strict loading.

    This layer keeps legacy keys (weight/bias on this module) while providing a
    1x1-safe fallback using GroupNorm.
    """

    def __init__(self, num_features: int, *, affine: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        # InstanceNorm without affine; we expose affine params ourselves.
        self._inorm = nn.InstanceNorm2d(int(num_features), affine=False, eps=self.eps)

        if affine:
            self.weight = nn.Parameter(torch.ones(int(num_features)))
            self.bias = nn.Parameter(torch.zeros(int(num_features)))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self._gn_groups = int(_pick_gn_groups(int(num_features)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.dim() == 4:
            h, w = int(x.shape[-2]), int(x.shape[-1])
            if h * w <= 1:
                return torch.nn.functional.group_norm(
                    x,
                    self._gn_groups,
                    weight=self.weight,
                    bias=self.bias,
                    eps=self.eps,
                )
        y = self._inorm(x)
        if self.weight is not None:
            y = y * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return y
