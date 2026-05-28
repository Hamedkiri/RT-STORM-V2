# models/sem_supheads.py
# -*- coding: utf-8 -*-
"""SemSupHeads: Attention + multi-head classifiers for SEM tokens.

Designed to be used with SemanticMoCoJEPA.encode_q(...)[1] which returns tokens:
    tokens: (B, S, D)

We apply a lightweight attention pooling over tokens and predict per-task logits
(similar spirit to SupHeads), while also exposing unified embedding extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _l2n(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class AttnPool(nn.Module):
    """Attention pooling: learnable query attends over tokens."""

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = int(dim)
        self.heads = int(heads)
        self.dropout = float(dropout)

        self.q = nn.Parameter(torch.zeros(1, 1, self.dim))
        nn.init.trunc_normal_(self.q, std=0.02)

        self.attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=self.heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(self.dim)
        self.proj = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dim, self.dim),
        )

    def forward(self, tokens: torch.Tensor, *, return_attn: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """tokens: (B,S,D) -> pooled (B,D)."""
        assert tokens.dim() == 3, f"Expected (B,S,D), got {tuple(tokens.shape)}"
        B, S, D = tokens.shape
        q = self.q.expand(B, 1, D)
        x = self.ln(tokens)
        out, w = self.attn(q, x, x, need_weights=return_attn, average_attn_weights=False)
        pooled = out[:, 0, :]
        pooled = pooled + self.proj(pooled)
        if return_attn:
            # w: (B, heads, 1, S) -> (B, heads, S)
            w = w[:, :, 0, :]
        return pooled, w


class SemSupHeads(nn.Module):
    """Multi-task classifier operating on SEM tokens.

    Args:
        tasks: dict task -> list of class names (or dict with 'classes')
        in_dim: token dim D
        heads: attention heads for pooling
        mlp_mult: hidden multiplier for per-task MLP
        dropout: dropout
        normalize: if True, L2-normalize pooled embeddings for get_embedding(...)
    """

    def __init__(
        self,
        tasks: Dict[str, Any],
        in_dim: int,
        *,
        heads: int = 4,
        mlp_mult: int = 2,
        dropout: float = 0.1,
        normalize: bool = True,
    ):
        super().__init__()
        self.tasks = tasks or {}
        self.in_dim = int(in_dim)
        self.heads = int(heads)
        self.mlp_mult = int(mlp_mult)
        self.dropout = float(dropout)
        self.normalize = bool(normalize)

        self.pool = AttnPool(self.in_dim, heads=self.heads, dropout=self.dropout)

        self.classifiers = nn.ModuleDict()
        for t, spec in self.tasks.items():
            if isinstance(spec, dict):
                classes = spec.get("classes", [])
            else:
                classes = spec
            n = int(len(classes))
            hid = max(1, self.in_dim * self.mlp_mult)
            self.classifiers[t] = nn.Sequential(
                nn.LayerNorm(self.in_dim),
                nn.Linear(self.in_dim, hid),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(hid, n),
            )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        return_attn: bool = False,
        return_task_embeddings: bool = False,
    ):
        pooled, attn = self.pool(tokens, return_attn=return_attn)
        if self.normalize:
            pooled_n = _l2n(pooled)
        else:
            pooled_n = pooled

        logits: Dict[str, torch.Tensor] = {}
        task_emb: Optional[Dict[str, torch.Tensor]] = {} if return_task_embeddings else None

        for t, head in self.classifiers.items():
            logits[t] = head(pooled_n)
            if task_emb is not None:
                task_emb[t] = pooled_n

        if return_attn and return_task_embeddings:
            return logits, attn, task_emb
        if return_attn:
            return logits, attn
        if return_task_embeddings:
            return logits, task_emb
        return logits

    @torch.no_grad()
    def get_embedding(
        self,
        tokens: torch.Tensor,
        *,
        kind: str = "attn",
        task: Optional[str] = None,
        token_pool: str = "mean",
    ) -> torch.Tensor:
        """Return an embedding tensor (B,D or B,S*D depending on kind).

        kind:
          - "attn"     : attention-pooled (B,D)
          - "mean"     : mean pool over tokens (B,D)
          - "tokG"     : first token (B,D)
          - "raw"      : flattened tokens (B, S*D)
        token_pool only used for kind="mean" ("mean" or "max").
        """
        if tokens.dim() != 3:
            raise ValueError(f"Expected tokens (B,S,D), got {tuple(tokens.shape)}")

        kind = (kind or "attn").lower()
        if kind == "attn":
            pooled, _ = self.pool(tokens, return_attn=False)
            emb = pooled
        elif kind == "mean":
            if token_pool == "max":
                emb = tokens.max(dim=1).values
            else:
                emb = tokens.mean(dim=1)
        elif kind in ("tokg", "style_tok"):
            emb = tokens[:, 0, :]
        elif kind == "raw":
            B, S, D = tokens.shape
            emb = tokens.reshape(B, S * D)
        else:
            raise ValueError(f"Unknown embedding kind: {kind}")

        if self.normalize and emb.dim() == 2:
            emb = _l2n(emb)
        return emb
