# models/jepa.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


# ============================== JEPA tokens (scale-aware) ==============================
class TokenJEPA(nn.Module):
    """
    Prédicteur JEPA pour tokens multi-échelles.
    Entrées :
      Ts : (B,S,D)  tokens student (vue v1)
      Tt : (B,S,D)  tokens teacher (vue v2)  [STOP-GRAD recommandé]
      mask : (B,S)  True si position MASQUÉE (on entraîne sur ces positions)
      w : (S,)      poids par échelle (optionnel ; ex. prioriser tokG/t5)
    Pertes :
      - MSE sur positions masquées (pondérée par w)
      - Variance / Covariance (type VICReg light) pour éviter le collapse
    Sortie :
      loss_total, {
        'loss_mse','loss_var','loss_cov',
        'pred'     : P.detach(),      # (B,S,D)
        'pred_raw' : P,               # (B,S,D) avec gradient (pour KD)
      }
    """
    def __init__(self, S: int, D: int, *, hidden_mult: int = 2, heads: int = 4,
                 use_norm: bool = True, var_lambda: float = 0.05, cov_lambda: float = 0.05):
        super().__init__()
        self.S, self.D = S, D
        self.use_norm = use_norm
        self.var_lambda = float(var_lambda)
        self.cov_lambda = float(cov_lambda)

        self.pre = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(embed_dim=D, num_heads=heads, batch_first=True)
        self.post = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, hidden_mult * D),
            nn.GELU(),
            nn.Linear(hidden_mult * D, D),
        )
        self.scale_pos = nn.Parameter(torch.randn(1, S, D) * 0.02)

    def _l2n(self, x: torch.Tensor, eps: float = 1e-6):
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def _vicreg_var(self, Z: torch.Tensor, eps: float = 1e-4):
        std = Z.std(dim=0) + eps
        return torch.mean(F.relu(1.0 - std))

    def _vicreg_cov(self, Z: torch.Tensor):
        Z = Z - Z.mean(dim=0, keepdim=True)
        C = (Z.T @ Z) / max(1, Z.size(0) - 1)
        off = C - torch.diag(torch.diag(C))
        return (off.pow(2).sum() / (self.D * (self.D - 1)))

    def forward(self,
                Ts: torch.Tensor,
                Tt: torch.Tensor,
                mask: torch.Tensor,
                w: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        assert Ts.shape == Tt.shape and Ts.dim() == 3
        B, S, D = Ts.shape
        assert S == self.S and D == self.D
        assert mask.shape == (B, S)

        X = Ts
        if self.use_norm: X = self._l2n(X)
        X = self.pre(X) + self.scale_pos
        H, _ = self.attn(X, X, X, need_weights=False)
        P = self.post(H) + Ts  # résiduel

        Y = Tt
        if self.use_norm: Y = self._l2n(Y)

        if mask.any():
            if w is not None:
                w_b = w.view(1, S, 1).to(Ts)
                loss_mse = (((P - Y).pow(2) * w_b).mean(dim=-1))[mask].mean()
            else:
                loss_mse = (P[mask] - Y[mask]).pow(2).mean()
        else:
            loss_mse = Ts.new_tensor(0.0)

        Z = P.reshape(B * S, D)
        if self.use_norm:
            Z = (Z - Z.mean(dim=0, keepdim=True)) / (Z.std(dim=0, keepdim=True) + 1e-6)
        loss_var = self._vicreg_var(Z)
        loss_cov = self._vicreg_cov(Z)
        loss = loss_mse + self.var_lambda * loss_var + self.cov_lambda * loss_cov

        return loss, {
            "loss_mse": float(loss_mse.item()),
            "loss_var": float(loss_var.item()),
            "loss_cov": float(loss_cov.item()),
            "pred": P.detach(),
            "pred_raw": P
        }


class MapJEPA(nn.Module):
    """
    JEPA spatial sur cartes de style m_i.
    Entrées :
      Ms : (B,C,H,W) carte student
      Mt : (B,C,H,W) carte teacher
      mask : (B,S) bool avec S=H*W, True si position masquée
    La carte est convertie en séquence spatiale (B,S,C), puis traitée comme des tokens.
    """
    def __init__(self, S: int, C: int, *, hidden_mult: int = 2, heads: int = 4,
                 use_norm: bool = True, var_lambda: float = 0.05, cov_lambda: float = 0.05):
        super().__init__()
        self.S = int(S)
        self.C = int(C)
        self.token_jepa = TokenJEPA(S=self.S, D=self.C, hidden_mult=hidden_mult, heads=heads,
                                    use_norm=use_norm, var_lambda=var_lambda, cov_lambda=cov_lambda)

    @staticmethod
    def _map_to_seq(M: torch.Tensor) -> torch.Tensor:
        assert M.dim() == 4
        B, C, H, W = M.shape
        return M.view(B, C, H * W).transpose(1, 2).contiguous()

    def forward(self, Ms: torch.Tensor, Mt: torch.Tensor, mask: torch.Tensor, w: torch.Tensor | None = None):
        Ts = self._map_to_seq(Ms)
        Tt = self._map_to_seq(Mt)
        loss, info = self.token_jepa(Ts, Tt, mask, w=w)
        info["pred_map"] = info["pred"].transpose(1, 2).contiguous().view_as(Ms).detach()
        info["pred_map_raw"] = info["pred_raw"].transpose(1, 2).contiguous().view_as(Ms)
        return loss, info
