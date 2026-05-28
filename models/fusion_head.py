import torch
import torch.nn as nn


class VectorGatedFusionHead(nn.Module):
    """Fusion légère style + sémantique au niveau embedding.

    Pipeline: LayerNorm -> projection vers fusion_dim -> gating vectoriel -> LayerNorm.
    Sorties: vecteur fusionné (B, fusion_dim) + debug optionnel.
    """

    def __init__(self, style_in_dim: int, sem_in_dim: int, fusion_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.style_in_dim = int(style_in_dim)
        self.sem_in_dim = int(sem_in_dim)
        self.fusion_dim = int(fusion_dim)
        self.dropout_p = float(dropout)

        self.style_norm = nn.LayerNorm(self.style_in_dim)
        self.sem_norm = nn.LayerNorm(self.sem_in_dim)

        self.style_proj = nn.Sequential(
            nn.Linear(self.style_in_dim, self.fusion_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
        )
        self.sem_proj = nn.Sequential(
            nn.Linear(self.sem_in_dim, self.fusion_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
        )

        self.gate = nn.Sequential(
            nn.LayerNorm(2 * self.fusion_dim),
            nn.Linear(2 * self.fusion_dim, self.fusion_dim),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(self.fusion_dim)

    def forward(self, style_feat: torch.Tensor, sem_feat: torch.Tensor, return_debug: bool = False):
        zs = self.style_proj(self.style_norm(style_feat))
        zm = self.sem_proj(self.sem_norm(sem_feat))
        g = self.gate(torch.cat([zs, zm], dim=1))
        fused = self.out_norm(g * zm + (1.0 - g) * zs)
        if return_debug:
            return fused, {
                'gate_mean': float(g.mean().detach().item()),
                'gate_std': float(g.std(unbiased=False).detach().item()),
            }
        return fused
