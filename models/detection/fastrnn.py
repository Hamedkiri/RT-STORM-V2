# models/detection/fastrnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FastRNNCell(nn.Module):
    """
    FastRNN cell (Kusupati et al.) style:
        z_t = tanh(W x_t + U h_{t-1})
        h_t = alpha * z_t + beta * h_{t-1}

    Ici alpha/beta sont learnables (contraints via sigmoid) pour la stabilité.
    """
    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # scalaires learnables, contraints dans (0,1)
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))
        self.beta_logit = nn.Parameter(torch.tensor(2.0))  # beta ~ 0.88 init

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)
        nn.init.orthogonal_(self.U.weight)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        # x_t: (B, Din), h_prev: (B, Dh)
        z = torch.tanh(self.W(x_t) + self.U(h_prev))
        alpha = torch.sigmoid(self.alpha_logit)
        beta = torch.sigmoid(self.beta_logit)
        h = alpha * z + beta * h_prev
        return h


class FastRNNEncoder(nn.Module):
    """
    Encode une séquence X: (B,S,D) -> H: (B,S,H)
    Optionnel: bidirectionnel.
    """
    def __init__(self, input_dim: int, hidden_dim: int, bidir: bool = False, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidir = bidir
        self.dropout = float(dropout)

        self.cell_fwd = FastRNNCell(input_dim, hidden_dim)
        self.cell_bwd = FastRNNCell(input_dim, hidden_dim) if bidir else None
        self.out_dim = hidden_dim * (2 if bidir else 1)

        self.drop = nn.Dropout(p=self.dropout) if self.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,S,D)
        returns: (B,S,out_dim)
        """
        B, S, D = x.shape
        dev = x.device

        h_f = torch.zeros(B, self.hidden_dim, device=dev, dtype=x.dtype)
        outs_f = []
        for t in range(S):
            h_f = self.cell_fwd(x[:, t], h_f)
            outs_f.append(self.drop(h_f))
        out_f = torch.stack(outs_f, dim=1)  # (B,S,H)

        if not self.bidir:
            return out_f

        h_b = torch.zeros(B, self.hidden_dim, device=dev, dtype=x.dtype)
        outs_b = []
        for t in reversed(range(S)):
            h_b = self.cell_bwd(x[:, t], h_b)
            outs_b.append(self.drop(h_b))
        outs_b.reverse()
        out_b = torch.stack(outs_b, dim=1)  # (B,S,H)

        return torch.cat([out_f, out_b], dim=-1)