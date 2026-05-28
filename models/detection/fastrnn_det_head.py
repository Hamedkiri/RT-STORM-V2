# models/detection/fastrnn_det_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fastrnn import FastRNNEncoder


class FastRNNDetHead(nn.Module):
    """
    Head anchor-free (FCOS-like):
      - cls_logits: (B, K, H, W)   K = num_classes-1 (sans background)
      - reg_ltrb:   (B, 4, H, W)   distances (l,t,r,b) >= 0
      - ctr_logits: (B, 1, H, W)

    On applique FastRNN sur la séquence des tokens spatiaux S = H*W.
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int,            # inclut background comme dans ton code (0=bg)
        hidden_dim: int = 256,
        bidir: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert num_classes >= 2, "num_classes doit inclure background (>=2)."
        self.num_classes = num_classes
        self.K = num_classes - 1  # sans background

        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0)

        self.rnn = FastRNNEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            bidir=bidir,
            dropout=dropout,
        )
        rnn_out = self.rnn.out_dim

        # MLP par token
        self.cls_mlp = nn.Sequential(
            nn.Linear(rnn_out, rnn_out),
            nn.ReLU(inplace=True),
            nn.Linear(rnn_out, self.K),
        )
        self.reg_mlp = nn.Sequential(
            nn.Linear(rnn_out, rnn_out),
            nn.ReLU(inplace=True),
            nn.Linear(rnn_out, 4),
        )
        self.ctr_mlp = nn.Sequential(
            nn.Linear(rnn_out, rnn_out),
            nn.ReLU(inplace=True),
            nn.Linear(rnn_out, 1),
        )

        # init plus stable
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor):
        """
        feat: (B, C, H, W)
        returns cls_logits, reg_ltrb, ctr_logits
        """
        B, C, H, W = feat.shape
        x = self.proj(feat)  # (B,hidden,H,W)

        # flatten spatial -> seq
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, -1)  # (B,S,D)
        h = self.rnn(x)  # (B,S,Dr)

        cls = self.cls_mlp(h)      # (B,S,K)
        reg = self.reg_mlp(h)      # (B,S,4)
        ctr = self.ctr_mlp(h)      # (B,S,1)

        # reshape back
        cls = cls.view(B, H, W, self.K).permute(0, 3, 1, 2).contiguous()  # (B,K,H,W)
        reg = reg.view(B, H, W, 4).permute(0, 3, 1, 2).contiguous()       # (B,4,H,W)
        ctr = ctr.view(B, H, W, 1).permute(0, 3, 1, 2).contiguous()       # (B,1,H,W)

        # distances positives
        reg = F.relu(reg)
        return cls, reg, ctr
