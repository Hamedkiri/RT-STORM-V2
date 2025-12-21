# models/discriminator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
# ──────────────────────────────────────────────────────────────
#  PatchGAN (optionnellement conditionnel sur un embedding de style)
# ──────────────────────────────────────────────────────────────

class PatchDiscriminator(nn.Module):
    """
    PatchGAN 70×70 avec SN et option de conditionnement par projection.
    - Si cond_dim == 0 ou e_y=None au forward → comportement identique à avant.
    - Sinon, on ajoute <φ(x), e_y> en projection (broadcast spatialement).
    """
    def __init__(self,
                 nc: int = 3,
                 ndf: int = 64,
                 n_layers: int = 4,
                 use_spectral_norm: bool = True,
                 cond_dim: int = 0):
        super().__init__()

        def sn(m): return nn.utils.spectral_norm(m) if use_spectral_norm else m

        # corps
        layers = []
        # 1ère couche (sans norm)
        layers += [sn(nn.Conv2d(nc, ndf, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True)]
        in_f = ndf
        for _ in range(1, n_layers):
            out_f = min(in_f * 2, ndf * 8)
            layers += [sn(nn.Conv2d(in_f, out_f, 4, 2, 1)),
                       nn.InstanceNorm2d(out_f, affine=True),
                       nn.LeakyReLU(0.2, inplace=True)]
            in_f = out_f
        self.body = nn.Sequential(*layers)
        self.head = sn(nn.Conv2d(in_f, 1, kernel_size=4, padding=1))

        # embedding pour projection conditionnelle
        self.cond_dim = int(cond_dim)
        self.embed = sn(nn.Conv2d(in_f, self.cond_dim, 1)) if self.cond_dim > 0 else None

    def forward(self, x: torch.Tensor, e_y: Optional[torch.Tensor] = None):
        """
        x  : (B,3,H,W)
        e_y: (B,cond_dim) ou None
        out: (B,1,h,w)
        """
        feat = self.body(x)               # (B,Cf,h,w)
        logits = self.head(feat)          # (B,1,h,w)
        if self.embed is not None and e_y is not None:
            B, _, h, w = feat.shape
            ey = e_y.view(B, self.cond_dim, 1, 1)            # (B,D,1,1)
            proj = (self.embed(feat) * ey).sum(1, keepdim=True)  # (B,1,h,w)
            logits = logits + proj
        return logits