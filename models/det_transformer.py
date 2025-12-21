# models/det_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.generator import UNetGenerator


class MultiScaleTokenEncoder(nn.Module):
    """
    Encodage multi-échelles à partir d'un UNetGenerator :
      - feat_branch="content" : encode_content(x) -> (z, (s1..s5))
      - feat_branch="style"   : style_enc(x)      -> (F_pyr, toks, tokG)
      - feat_branch="concat"  : concat des deux le long de C

    Retourne des tokens (B,S,D) + pos encodings (B,S,D).
    """

    def __init__(
        self,
        generator: UNetGenerator,
        d_model: int = 256,
        levels=(1, 2, 3, 4, 5),
        feat_branch: str = "content",  # "content" | "style" | "concat"
    ):
        super().__init__()
        self.generator = generator
        self.d_model = d_model
        self.levels = tuple(levels)
        self.feat_branch = feat_branch

        # On crée des proj/embeds pour chaque niveau demandé.
        self.proj = nn.ModuleDict()
        self.level_embed = nn.ParameterDict()

        for lvl in self.levels:
            key = f"s{lvl}"
            # in_channels est mis à 256 par défaut, mais on corrigera
            # dynamiquement au premier forward si C!=256.
            self.proj[key] = nn.Conv2d(
                in_channels=256,
                out_channels=d_model,
                kernel_size=1,
            )
            self.level_embed[key] = nn.Parameter(torch.randn(1, 1, d_model))

        # Token global appris (vue globale supplémentaire)
        self.use_global_token = True
        self.global_token = nn.Parameter(torch.randn(1, 1, d_model))

    @staticmethod
    def _build_2d_sine_pos_embed(h, w, d_model, device):
        """
        Encodeur 2D sin/cos de dimension finale = d_model.

        Idée :
          - d_model/2 dimensions pour x, d_model/2 pour y
          - sin/cos alternés
        """
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        grid_x = grid_x.float() / max(w - 1, 1)
        grid_y = grid_y.float() / max(h - 1, 1)

        dim_half = d_model // 2
        dim_t = torch.arange(dim_half, device=device, dtype=torch.float32)
        dim_t = 10000 ** (2 * (dim_t // 2) / dim_half)

        # x
        pos_x = grid_x[..., None] / dim_t
        pos_x = torch.stack(
            (pos_x.sin()[..., 0::2], pos_x.cos()[..., 1::2]), dim=-1
        ).flatten(-2)  # (H,W,dim_half)

        # y
        pos_y = grid_y[..., None] / dim_t
        pos_y = torch.stack(
            (pos_y.sin()[..., 0::2], pos_y.cos()[..., 1::2]), dim=-1
        ).flatten(-2)  # (H,W,dim_half)

        pos = torch.cat([pos_x, pos_y], dim=-1)  # (H,W,d_model)
        pos = pos.view(1, h * w, d_model)
        return pos

    def _get_content_feats(self, x):
        # encode_content(x) doit renvoyer z, (s1..s5)
        z, (s1, s2, s3, s4, s5) = self.generator.encode_content(x)
        feat_map = {1: s1, 2: s2, 3: s3, 4: s4, 5: s5}
        return z, feat_map

    def _get_style_feats(self, x):
        F_pyr, toks, tokG = self.generator.style_enc(x)
        feat_map = {i + 1: F_pyr[i] for i in range(len(F_pyr))}
        return tokG, feat_map

    def forward(self, x):
        B = x.size(0)
        device = x.device

        # Sélection des features selon feat_branch
        if self.feat_branch == "content":
            bottleneck, feats_c = self._get_content_feats(x)
            feats_s = None
        elif self.feat_branch == "style":
            bottleneck, feats_s = self._get_style_feats(x)
            feats_c = None
        elif self.feat_branch == "concat":
            bottleneck_c, feats_c = self._get_content_feats(x)
            bottleneck_s, feats_s = self._get_style_feats(x)
            # On concat les bottlenecks si besoin plus tard
            bottleneck = (bottleneck_c, bottleneck_s)
        else:
            raise ValueError(f"Unknown feat_branch={self.feat_branch}")

        all_tokens, all_pos = [], []

        for lvl in self.levels:
            key = f"s{lvl}"

            if feats_c is not None and feats_s is None:
                f_map = feats_c[lvl]
            elif feats_s is not None and feats_c is None:
                f_map = feats_s[lvl]
            else:
                fc = feats_c[lvl]
                fs = feats_s[lvl]
                f_map = torch.cat([fc, fs], dim=1)

            B, C, H, W = f_map.shape

            # On s'assure que la proj a le bon C en entrée
            proj = self.proj[key]
            if proj.in_channels != C:
                new_proj = nn.Conv2d(C, self.d_model, kernel_size=1).to(f_map.device)
                new_proj.weight.data.normal_(0, 0.01)
                new_proj.bias.data.zero_()
                self.proj[key] = new_proj
                proj = new_proj

            feat = proj(f_map)                       # (B,D,H,W)
            feat = feat.flatten(2).transpose(1, 2)   # (B,H*W,D)

            pos = self._build_2d_sine_pos_embed(H, W, self.d_model, device)
            lvl_emb = self.level_embed[key]          # (1,1,D)
            feat = feat + lvl_emb                    # shift de niveau

            all_tokens.append(feat)
            all_pos.append(pos.expand(B, -1, -1))

        # Concat multi-échelles
        tokens = torch.cat(all_tokens, dim=1)  # (B,S,D)
        pos = torch.cat(all_pos, dim=1)        # (B,S,D)

        # Token global
        if self.use_global_token:
            g = self.global_token.expand(B, 1, -1)
            tokens = torch.cat([g, tokens], dim=1)
            pos = torch.cat([torch.zeros_like(g), pos], dim=1)

        mask = torch.zeros(B, tokens.size(1), dtype=torch.bool, device=device)
        return tokens, pos, mask



class SimpleDETRHead(nn.Module):
    """
    Tête de détection type DETR :
      - MultiScaleTokenEncoder → tokens multi-échelles
      - queries appris
      - TransformerDecoder (self + cross-attn)
      - heads: classes + boxes
    """

    def __init__(
        self,
        generator: UNetGenerator,
        num_classes: int,
        num_queries: int = 300,
        d_model: int = 256,
        nheads: int = 8,
        num_decoder_layers: int = 6,
        feat_branch: str = "content",
        levels=(1, 2, 3, 4, 5),
    ):
        super().__init__()

        self.generator = generator
        self.encoder = MultiScaleTokenEncoder(
            generator=generator,
            d_model=d_model,
            levels=levels,
            feat_branch=feat_branch,
        )

        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nheads,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )

        self.class_embed = nn.Linear(d_model, num_classes)
        self.bbox_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
        )

    def forward(self, x):
        """
        x :
          - soit un tenseur (B,3,H,W)
          - soit une liste/tuple de tenseurs (3,H,W) comme pour torchvision DETR

        Retourne :
          - pred_logits : (B,Q,num_classes)
          - pred_boxes  : (B,Q,4) normalisées [0,1] (cx, cy, w, h)
        """
        # Permet d'appeler le modèle comme TorchVision DETR: model([img])
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                raise ValueError("SimpleDETRHead.forward: input list is empty")
            if not torch.is_tensor(x[0]):
                raise TypeError(
                    "SimpleDETRHead.forward: list elements must be torch.Tensors"
                )

            if x[0].dim() == 3:
                x = torch.stack(x, dim=0)    # (B,3,H,W)
            elif x[0].dim() == 4:
                x = torch.cat(x, dim=0)      # concat
            else:
                raise ValueError(
                    "SimpleDETRHead.forward: tensors must have dim 3 or 4"
                )

        elif isinstance(x, torch.Tensor):
            if x.dim() != 4:
                raise ValueError(
                    f"SimpleDETRHead.forward: expected (B,3,H,W), got {tuple(x.shape)}"
                )
        else:
            raise TypeError(
                "SimpleDETRHead.forward: expected Tensor or list/tuple of Tensors"
            )

        tokens, pos, mask = self.encoder(x)  # (B,S,D), (B,S,D), (B,S)
        B = x.size(0)

        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B,Q,D)
        memory = tokens + pos

        hs = self.decoder(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=mask,
        )  # (B,Q,D)

        pred_logits = self.class_embed(hs)      # (B,Q,num_classes)
        pred_boxes = self.bbox_embed(hs).sigmoid()  # (B,Q,4) en [0,1]
        return pred_logits, pred_boxes
