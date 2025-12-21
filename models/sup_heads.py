
# models/sup_heads.py
import torch
import torch.nn as nn
import math
import torch, torch.nn as nn, torch.nn.functional as F

@staticmethod
def _l2_norm_rows(t: torch.Tensor) -> torch.Tensor:
    return t / (t.norm(dim=1, keepdim=True) + 1e-8)


@torch.no_grad()
def _gap(self, x: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool2d(x, 1).flatten(1)

# ======================= ATTENTION POUR LES SUP-HEADS =======================
class ScaleMixer(nn.Module):
    """
    Attention multi-têtes à travers S échelles.
    Entrée  : T ∈ (B, S, D)
    Sorties : out ∈ (B, out_dim), alpha_mean ∈ (H, S), Hbar ∈ ()
    """
    def __init__(self, num_scales: int, token_dim: int, *, heads: int = 4,
                 out_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        assert num_scales >= 2, "ScaleMixer requiert au moins 2 échelles."
        self.S = int(num_scales)
        self.D = int(token_dim)
        self.H = int(max(1, heads))
        self.dp = float(dropout)
        self.dh = max(8, self.D // self.H)

        self.W_k = nn.Linear(self.D, self.H * self.dh, bias=False)
        self.W_v = nn.Linear(self.D, self.H * self.dh, bias=False)
        self.q_heads = nn.Parameter(torch.randn(self.H, self.dh) / math.sqrt(self.dh))

        proj_out = out_dim if out_dim is not None else self.S * self.D
        self.proj = nn.Sequential(
            nn.Linear(self.H * self.dh, proj_out),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dp),
        )
        self.ln_in  = nn.LayerNorm(self.D)
        self.ln_out = nn.LayerNorm(proj_out)

    def forward(self, T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, D = T.shape
        assert S == self.S and D == self.D, f"ScaleMixer: attendu (B,{self.S},{self.D}), reçu {T.shape}"

        Tin = self.ln_in(T)                                  # (B,S,D)
        K = self.W_k(Tin).view(B, S, self.H, self.dh)        # (B,S,H,dh)
        V = self.W_v(Tin).view(B, S, self.H, self.dh)        # (B,S,H,dh)

        q = self.q_heads.view(1, 1, self.H, self.dh)         # (1,1,H,dh)
        scores = (K * q).sum(-1) / math.sqrt(self.dh)        # (B,S,H)
        scores = scores.permute(0, 2, 1)                     # (B,H,S)

        alpha = torch.softmax(scores, dim=-1)                # (B,H,S)
        Vh = V.permute(0, 2, 1, 3)                           # (B,H,S,dh)
        ctx = (alpha.unsqueeze(-1) * Vh).sum(dim=2)          # (B,H,dh)
        out = self.proj(ctx.reshape(B, self.H * self.dh))    # (B,proj_out)
        out = self.ln_out(out)

        alpha_mean = alpha.mean(dim=0).detach()              # (H,S)
        p = alpha.clamp_min(1e-9)
        H = -(p * p.log()).sum(dim=-1) / math.log(float(self.S))  # (B,H)
        Hbar = H.mean().detach()
        return out, alpha_mean, Hbar


class SupHeads(nn.Module):
    """
    Têtes supervisées multi-tâches, sensibles aux tokens multi-échelles.

    Modes d'entrée:
      • 'multi6' : concat des S tokens → in_dim = S*D  (ex: tok6)
      • 'single' : un seul token       → in_dim = D    (ex: tokG, style_tok, tok6_mean, tok6_w)
      • 'flat'   : vecteur arbitraire  → in_dim = K    (ex: bot, mgap, bot+tok, ...)
    """
    def __init__(self, tasks: dict[str, int], in_dim: int, *, num_scales: int = 6,
                 token_mode: str = "auto", heads: int = 4, dropout: float = 0.1, mlp_mult: int = 2):
        super().__init__()
        self.tasks     = dict(tasks)
        self.in_dim    = int(in_dim)
        self.S         = int(num_scales)
        self.heads     = int(heads)
        self.dropout   = float(dropout)
        self.mlp_mult  = int(mlp_mult)

        if token_mode not in {"auto", "multi6", "single", "flat"}:
            raise ValueError(f"token_mode invalide: {token_mode}")
        if token_mode == "auto":
            token_mode = "multi6" if (self.in_dim % self.S == 0 and (self.in_dim // self.S) > 0) else "single"
        self.token_mode = token_mode

        self.token_dim = (self.in_dim // self.S) if self.token_mode == "multi6" else self.in_dim

        self.mixers      = nn.ModuleDict()
        self.classifiers = nn.ModuleDict()
        for t, n_cls in self.tasks.items():
            if self.token_mode == "multi6":
                target_in = self.S * self.token_dim
                self.mixers[t] = ScaleMixer(self.S, self.token_dim, heads=self.heads,
                                            out_dim=target_in, dropout=self.dropout)
                clf_in = target_in
            else:
                self.mixers[t] = None
                clf_in = self.in_dim

            self.classifiers[t] = nn.Sequential(
                nn.LayerNorm(clf_in),
                nn.Linear(clf_in, self.mlp_mult * clf_in),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
                nn.Linear(self.mlp_mult * clf_in, n_cls),
            )

        self._warned_align = False

    def _ensure_flat(self, x: torch.Tensor) -> torch.Tensor:
        """Applatissement robuste:
           - (B,C,H,W) -> GAP -> (B,C)
           - (B,S,D)   -> (B,S*D) en 'multi6', sinon moyenne sur S
           - (B,K)     -> inchangé
        """
        if x.dim() == 4:                   # (B,C,H,W)
            return _gap(x)                 # supposé renvoyer (B,C)
        if x.dim() == 3:                   # (B,S,D)
            return x.reshape(x.size(0), -1) if self.token_mode == "multi6" else x.mean(dim=1)
        return x                            # (B,K)

    def _align_in_dim(self, x: torch.Tensor) -> torch.Tensor:
        B, Din = x.shape
        if self.token_mode == "multi6":
            target = self.S * self.token_dim
            if Din == target: return x
            if Din > target:
                if not self._warned_align:
                    print(f"[SupHeads] multi6: crop {Din}→{target} (S*D).")
                    self._warned_align = True
                return x[:, :target]
            if not self._warned_align:
                print(f"[SupHeads] multi6: pad {Din}→{target} (zéros).")
                self._warned_align = True
            return torch.cat([x, x.new_zeros(B, target - Din)], dim=1)

        # single / flat
        if Din == self.in_dim: return x
        if Din > self.in_dim:
            if not self._warned_align:
                print(f"[SupHeads] single/flat: crop {Din}→{self.in_dim}.")
                self._warned_align = True
            return x[:, :self.in_dim]
        if not self._warned_align:
            print(f"[SupHeads] single/flat: pad {Din}→{self.in_dim} (zéros).")
            self._warned_align = True
        return torch.cat([x, x.new_zeros(B, self.in_dim - Din)], dim=1)

    def forward(self, feats: torch.Tensor, *, return_attn: bool = False,
                return_task_embeddings: bool = False):
        x = self._ensure_flat(feats)
        x = self._align_in_dim(x)
        B, Din = x.shape

        logits, embs, attn_info = {}, {}, {}
        if self.token_mode == "multi6":
            T = x.view(B, self.S, self.token_dim)
            for t in self.tasks:
                out, alpha_mean, Hbar = self.mixers[t](T)    # (B, S*token_dim)
                embs[t]   = out
                logits[t] = self.classifiers[t](out)
                if return_attn:
                    attn_info[t] = {"alpha": alpha_mean.detach().cpu().numpy(),  # (H,S)
                                    "entropy": float(Hbar.item())}
        else:
            for t in self.tasks:
                embs[t]   = x
                logits[t] = self.classifiers[t](x)
                if return_attn:
                    attn_info[t] = {"alpha": None, "entropy": float("nan")}

        if return_attn and return_task_embeddings:
            return logits, attn_info, embs
        if return_attn:
            return logits, attn_info
        if return_task_embeddings:
            return logits, embs
        return logits
