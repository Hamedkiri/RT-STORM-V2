# file: models/cls_tokens.py
import weakref
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_2d_sincos_pos_embed(h: int, w: int, dim: int, device):
    """
    Embedding sin/cos 2D (façon ViT) pour une grille HxW → (1, H*W, dim).
    Pas de paramètres appris → fonctionne pour n'importe quelle résolution.
    """
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, steps=h, device=device),
        torch.linspace(-1.0, 1.0, steps=w, device=device),
        indexing="ij",
    )  # (H, W), (H, W)

    dim_half = dim // 4  # pour x_sin, x_cos, y_sin, y_cos
    if dim_half == 0:
        return torch.zeros(1, h * w, dim, device=device)

    omega = torch.arange(dim_half, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(dim_half - 1, 1)))  # (dim_half,)

    out_x = x.unsqueeze(-1) * omega
    out_y = y.unsqueeze(-1) * omega

    sin_x = torch.sin(out_x)
    cos_x = torch.cos(out_x)
    sin_y = torch.sin(out_y)
    cos_y = torch.cos(out_y)

    emb = torch.cat([sin_x, cos_x, sin_y, cos_y], dim=-1)  # (H, W, 4*dim_half)

    if emb.shape[-1] < dim:
        pad = dim - emb.shape[-1]
        emb = torch.cat([emb, torch.zeros(h, w, pad, device=device)], dim=-1)
    elif emb.shape[-1] > dim:
        emb = emb[..., :dim]

    emb = emb.view(1, h * w, dim)  # (1, H*W, dim)
    return emb


class MultiScaleTokenEncoder(nn.Module):
    """
    Encodeur multi-échelles → tokens pour UNetGenerator (branche contenu).

    IMPORTANT : on NE stocke PAS le backbone comme sous-module PyTorch.
    On garde juste une weakref vers lui, pour éviter les boucles de modules.

    Hypothèse adaptée à ton UNetGenerator :
      s1 : C = ngf
      s2 : C = 2*ngf
      s3 : C = 4*ngf
      s4 : C = 8*ngf
      s5 : C = 8*ngf
      bottleneck z : C = 8*ngf
    """

    def __init__(
        self,
        backbone: nn.Module,
        d_model: int = 256,
        levels=("s3", "s4", "s5"),
        use_global_token: bool = True,
        use_scale_embed: bool = True,
    ):
        super().__init__()
        # weakref vers le UNetGenerator (pas un sous-module)
        self._backbone_ref = weakref.ref(backbone)

        self.d_model = int(d_model)
        self.levels = list(levels)
        self.use_global_token = bool(use_global_token)
        self.use_scale_embed = bool(use_scale_embed)

        # On récupère ngf depuis le backbone (défini dans UNetGenerator)
        ngf = int(getattr(backbone, "ngf", 64))

        # Canaux attendus pour chaque niveau de skip de ton UNet
        self.level_channels = {
            "s1": ngf,
            "s2": 2 * ngf,
            "s3": 4 * ngf,
            "s4": 8 * ngf,
            "s5": 8 * ngf,
        }

        # Conv 1x1 par niveau pour projeter C_l → d_model (pas Lazy)
        self.proj_convs = nn.ModuleDict()
        for lvl in self.levels:
            in_ch = self.level_channels[lvl]
            self.proj_convs[lvl] = nn.Conv2d(in_ch, self.d_model, kernel_size=1)

        # Embedding de scale (un embedding par niveau)
        if self.use_scale_embed:
            self.scale_ids = {lvl: i for i, lvl in enumerate(self.levels)}
            self.scale_embed = nn.Embedding(len(self.levels), self.d_model)
        else:
            self.scale_ids = {}

        # Token global optionnel (à partir de z, C=8*ngf)
        if self.use_global_token:
            bottleneck_ch = 8 * ngf
            self.global_proj = nn.Linear(bottleneck_ch, self.d_model)

    def _split_levels(self, skips):
        """
        skips = (s1, s2, s3, s4, s5) comme dans encode_content.
        On renvoie un dict { "s1": s1, ..., "s5": s5 } pour convenance.
        """
        assert isinstance(skips, (list, tuple)) and len(skips) >= 5, \
            "encode_content doit renvoyer z, (s1..sL) avec L>=5"
        return {
            "s1": skips[0],
            "s2": skips[1],
            "s3": skips[2],
            "s4": skips[3],
            "s5": skips[4],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, H, W)
        Retourne :
            tokens : (B, S_total, d_model)
        """
        backbone = self._backbone_ref()
        if backbone is None:
            raise RuntimeError("Backbone has been garbage-collected.")

        # Branche contenu du UNet
        z, skips = backbone.encode_content(x)  # z: (B,8*ngf,H',W'), skips: (s1..s5)
        skips_dict = self._split_levels(skips)

        B = x.size(0)
        tokens_per_scale = []

        for lvl in self.levels:
            fmap = skips_dict[lvl]  # (B, C_l, H_l, W_l)
            B, C, H, W = fmap.shape

            proj = self.proj_convs[lvl](fmap)  # (B, d_model, H_l, W_l)
            proj = proj.view(B, self.d_model, H * W).permute(0, 2, 1)  # (B, H_l*W_l, d_model)

            # Positional embedding 2D sin/cos
            pos = build_2d_sincos_pos_embed(H, W, self.d_model, device=proj.device)
            proj = proj + pos  # (B, S_l, D)

            if self.use_scale_embed:
                sid = self.scale_ids[lvl]
                scale_emb = self.scale_embed.weight[sid].view(1, 1, self.d_model)
                proj = proj + scale_emb  # broadcast sur B et S_l

            tokens_per_scale.append(proj)  # (B, S_l, D)

        tokens = torch.cat(tokens_per_scale, dim=1)  # (B, S_total, D)

        if self.use_global_token:
            # Global Average Pooling sur z (contenu bottleneck)
            if z.dim() == 4:
                zg = F.adaptive_avg_pool2d(z, (1, 1)).view(B, -1)
            else:
                zg = z
            zg_tok = self.global_proj(zg).unsqueeze(1)  # (B, 1, D)
            tokens = torch.cat([zg_tok, tokens], dim=1)

        return tokens  # (B, S, D)


class TokenClassifier(nn.Module):
    """
    Tête de classification type ViT :
      - TransformerEncoder (quelques couches)
      - pooling via CLS token (par défaut) ou mean pooling
      - Linear → num_classes
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_cls_token: bool = True,
        pool_type: str = "cls",  # "cls" ou "mean"
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_classes = int(num_classes)
        self.use_cls_token = bool(use_cls_token)
        assert pool_type in ("cls", "mean")
        self.pool_type = pool_type

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens : (B, S, D) en sortie de MultiScaleTokenEncoder
        Retourne :
            logits : (B, num_classes)
        """
        B, S, D = tokens.shape
        assert D == self.d_model, f"d_model mismatch: got {D}, expected {self.d_model}"

        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)  # (B,1,D)
            x = torch.cat([cls, tokens], dim=1)     # (B, 1+S, D)
        else:
            x = tokens

        x = self.encoder(x)  # (B, S', D)

        if self.pool_type == "cls" and self.use_cls_token:
            pooled = x[:, 0]  # (B,D)
        else:
            pooled = x.mean(dim=1)  # (B,D)

        pooled = self.norm(pooled)
        logits = self.head(pooled)  # (B,num_classes)
        return logits


class TokenFeatureAggregator(nn.Module):
    """
    Agrège une séquence de tokens (B, S, D) en un embedding d'image (B, D),
    avec éventuellement un petit TransformerEncoder type ViT.

    - Si use_transformer=False → simple mean pooling + LayerNorm.
    - Si use_transformer=True  → TransformerEncoder + pooling (CLS ou mean).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_transformer: bool = True,
        use_cls_token: bool = True,
        pool_type: str = "cls",  # "cls" ou "mean"
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.use_transformer = bool(use_transformer)
        self.use_cls_token = bool(use_cls_token)
        assert pool_type in ("cls", "mean")
        self.pool_type = pool_type

        if self.use_transformer:
            if self.use_cls_token:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
                nn.init.trunc_normal_(self.cls_token, std=0.02)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens : (B, S, D)
        Retourne :
            feats : (B, D)
        """
        B, S, D = tokens.shape
        assert D == self.d_model, f"d_model mismatch: got {D}, expected {self.d_model}"

        if not self.use_transformer:
            pooled = tokens.mean(dim=1)  # (B,D)
            return self.norm(pooled)

        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)  # (B,1,D)
            x = torch.cat([cls, tokens], dim=1)     # (B, 1+S, D)
        else:
            x = tokens

        x = self.encoder(x)  # (B, S', D)

        if self.pool_type == "cls" and self.use_cls_token:
            pooled = x[:, 0]  # (B,D)
        else:
            pooled = x.mean(dim=1)

        pooled = self.norm(pooled)
        return pooled  # (B,D)
