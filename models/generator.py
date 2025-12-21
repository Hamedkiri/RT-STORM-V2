# file: models/generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union, List

from models.backbone import Down, ResBlock, ConvBlock, StylePyramidEncoder, SPADEUp
from models.cls_tokens import MultiScaleTokenEncoder, TokenFeatureAggregator


# ──────────────────────────────────────────────────────────────
#  UNet + SPADE (maps depuis x, tokens multi-échelles depuis y)
#  + content-tokens (MultiScaleTokenEncoder) pour sup_feat_type
#    "cont_tok" / "cont_tok_vit"
# ──────────────────────────────────────────────────────────────


class UNetGenerator(nn.Module):
    """
    U-Net SPADE/SEAN avec tokens multi-échelles.

    - Cartes SPADE (m5..m1) : dérivées du **contenu x** via StylePyramidEncoder.h{5..1}(s{5..1})
    - Tokens (t5..t1) + token global tokG : dérivés de l’**image de style y**
      via StylePyramidEncoder(style_img) qui retourne (maps, toks, tokG)
      (API tolérante aux variantes anciennes).

    En plus, pour la supervision de type contenu :
      - MultiScaleTokenEncoder(backbone=self) produit des tokens de contenu
        (à partir de s3, s4, s5).
      - TokenFeatureAggregator agrège ces tokens en un embedding global.

    Paramètres:
      nc:        canaux image
      ngf:       base channels
      style_nc:  canaux image de style (souvent 3)
      spade_ch:  canaux des cartes SPADE (m5..m1)
      token_dim: dimension des tokens (t*, tokG, content_tokens)
    """

    def __init__(
        self,
        nc: int = 3,
        ngf: int = 64,
        style_nc: int = 3,
        spade_ch: int = 64,
        token_dim: int = 256,
    ):
        super().__init__()
        self.ngf = int(ngf)
        self.hid_dim = int(token_dim)  # dim tokens style + contenu
        self.nc = int(nc)

        # tailles des skips (s5..s1) côté encodeur
        self.skip_channels: List[int] = [ngf * 8, ngf * 8, ngf * 4, ngf * 2, ngf]

        # --------------------- encodeur contenu ---------------------
        self.d1 = Down(nc, ngf)        # → s1
        self.d2 = Down(ngf, ngf * 2)   # → s2
        self.d3 = Down(ngf * 2, ngf * 4)  # → s3
        self.d4 = Down(ngf * 4, ngf * 8)  # → s4
        self.d5 = Down(ngf * 8, ngf * 8)  # → s5
        self.d6 = Down(ngf * 8, ngf * 8)  # → s6 (plus compact)
        self.res5, self.res6 = ResBlock(ngf * 8), ResBlock(ngf * 8)
        self.bot = ConvBlock(ngf * 8, ngf * 8)

        # ------------------ encodeur style / SPADE ------------------
        # Doit implémenter:
        #   • __call__(img) → (maps, toks, tokG) (toléré: (toks,tokG), (maps,tokG), (tokG,))
        #   • h5..h1(skip)  → maps m5..m1 (taille spade_ch)
        self.style_enc = StylePyramidEncoder(style_nc, ngf, spade_ch, token_dim)

        # ------------------------ décodeur --------------------------
        self.u1 = SPADEUp(
            c_up=ngf * 8, c_skip=ngf * 8, c_out=ngf * 8,
            s_ch=spade_ch, token_dim=token_dim
        )  # s5
        self.u2 = SPADEUp(
            c_up=ngf * 8, c_skip=ngf * 8, c_out=ngf * 8,
            s_ch=spade_ch, token_dim=token_dim
        )  # s4
        self.u3 = SPADEUp(
            c_up=ngf * 8, c_skip=ngf * 4, c_out=ngf * 4,
            s_ch=spade_ch, token_dim=token_dim
        )  # s3
        self.u4 = SPADEUp(
            c_up=ngf * 4, c_skip=ngf * 2, c_out=ngf * 2,
            s_ch=spade_ch, token_dim=token_dim
        )  # s2
        self.u5 = SPADEUp(
            c_up=ngf * 2, c_skip=ngf, c_out=ngf,
            s_ch=spade_ch, token_dim=token_dim
        )  # s1

        self.final = nn.Sequential(
            nn.Conv2d(ngf, ngf, 3, 1, 1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ngf, nc, 3, 1, 1),
            nn.Tanh(),
        )

        # ----------------- content tokens (branche contenu) ---------
        # MultiScaleTokenEncoder utilise encode_content(x) pour s3..s5
        self.content_tok_encoder = MultiScaleTokenEncoder(
            backbone=self,
            d_model=self.hid_dim,
            levels=("s3", "s4", "s5"),
            use_global_token=True,
            use_scale_embed=True,
        )
        # Agrégateur façon ViT pour "cont_tok_vit"
        self.content_tok_agg = TokenFeatureAggregator(
            d_model=self.hid_dim,
            nhead=4,
            num_layers=2,
            dim_feedforward=1024,
            dropout=0.1,
            use_transformer=True,
            use_cls_token=True,
            pool_type="cls",
        )

        # Pour compat : mini-têtes MLP si besoin rapide (debug)
        self.sup_heads: Optional[nn.Module] = None
        # heads pour tok+delta (lazy init)
        self._delta_heads: Optional[nn.ModuleList] = None

    # --------------------------- helpers ---------------------------

    @staticmethod
    def _l2_norm_rows(t: torch.Tensor) -> torch.Tensor:
        """
        Normalise un tenseur 2D (B, D) le long de dim=1.
        """
        return t / (t.norm(dim=1, keepdim=True) + 1e-8)

    @staticmethod
    def _l2_norm_lastdim(t: torch.Tensor) -> torch.Tensor:
        """
        Normalise un tenseur (B, S, D) sur la dernière dimension.
        """
        return t / (t.norm(dim=-1, keepdim=True) + 1e-8)

    @torch.no_grad()
    def _gap(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def _parse_delta_weights(self, w, n: int) -> List[float]:
        if isinstance(w, (list, tuple)):
            vals = [float(x) for x in w]
        else:
            try:
                vals = [float(x) for x in str(w).split(",") if x.strip()]
            except Exception:
                vals = []
        if len(vals) == n:
            return vals
        if len(vals) == 1:
            return vals * n
        if len(vals) == 0:
            return [1.0] * n
        vals = (vals + [vals[-1]] * n)[:n]
        return vals

    # robuste aux anciennes signatures du style-encoder
    @torch.no_grad()
    def _style_tokens_maps(self, img: torch.Tensor):
        maps = toks = tokG = None
        try:
            se = self.style_enc(img)
            if isinstance(se, (list, tuple)):
                if len(se) == 3:
                    a, b, c = se
                    if (
                        isinstance(a, (list, tuple))
                        and len(a)
                        and hasattr(a[0], "dim")
                        and a[0].dim() == 4
                    ):
                        maps = a
                    if (
                        isinstance(b, (list, tuple))
                        and len(b)
                        and hasattr(b[0], "dim")
                        and b[0].dim() == 2
                    ):
                        toks = b
                    if hasattr(c, "dim") and c.dim() == 2:
                        tokG = c
                elif len(se) == 2:
                    a, b = se
                    if (
                        isinstance(a, (list, tuple))
                        and len(a)
                        and hasattr(a[0], "dim")
                        and a[0].dim() == 4
                        and hasattr(b, "dim")
                        and b.dim() == 2
                    ):
                        maps, tokG = a, b
                    elif (
                        isinstance(a, (list, tuple))
                        and len(a)
                        and hasattr(a[0], "dim")
                        and a[0].dim() == 2
                        and hasattr(b, "dim")
                        and b.dim() == 2
                    ):
                        toks, tokG = a, b
                elif len(se) == 1 and hasattr(se[0], "dim") and se[0].dim() == 2:
                    tokG = se[0]
            elif hasattr(se, "dim") and se.dim() == 2:
                tokG = se
        except Exception:
            pass
        return maps, toks, tokG

    # ------------------------ pipeline U-Net ------------------------

    def encode_content(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Retourne:
          - z:   bottleneck (B, 8*ngf, h, w)
          - skips: (s1..s5)
        """
        s1 = self.d1(x)
        s2 = self.d2(s1)
        s3 = self.d3(s2)
        s4 = self.d4(s3)
        s5 = self.res5(self.d5(s4))
        z = self.res6(self.d6(s5))
        z = self.bot(z)
        return z, (s1, s2, s3, s4, s5)

    @torch.no_grad()
    def build_style(self, style_imgs: torch.Tensor):
        maps, toks, tokG = self._style_tokens_maps(style_imgs)
        return {"maps": maps, "tokens": toks, "token": tokG}

    def forward(self, x: torch.Tensor, *, style: Optional[Union[torch.Tensor, dict]] = None):
        # 1) contenu
        z, (s1, s2, s3, s4, s5) = self.encode_content(x)

        # 2) tokens de style
        if style is None:
            _, toks, tokG = self._style_tokens_maps(x)  # identité
        elif isinstance(style, torch.Tensor):
            _, toks, tokG = self._style_tokens_maps(style)  # style explicite y
        elif isinstance(style, dict):
            toks = style.get("tokens", None)
            tokG = style.get("token", None)
            if toks is None and tokG is None:
                _, toks, tokG = self._style_tokens_maps(x)
            elif toks is None and tokG is not None:
                toks = (tokG, tokG, tokG, tokG, tokG)
            elif toks is not None and tokG is None:
                # approx tokG = moyenne des t*
                comps = []
                for t in toks:
                    comps.append(t[0] if (isinstance(t, (tuple, list)) and len(t) == 2) else t)
                tokG = torch.stack(comps, dim=0).mean(0)
        else:
            raise TypeError(
                "style doit être None, Tensor(y), ou dict{'tokens':(t5..t1), 'token':tokG}."
            )

        # 3) cartes SPADE (depuis contenu x via h5..h1)
        m5 = self.style_enc.h5(s5)
        m4 = self.style_enc.h4(s4)
        m3 = self.style_enc.h3(s3)
        m2 = self.style_enc.h2(s2)
        m1 = self.style_enc.h1(s1)
        t5, t4, t3, t2, t1 = toks if isinstance(toks, (list, tuple)) else (None,) * 5

        # 4) décodeur conditionné
        x = self.u1(z, s5, m5, t5)
        x = self.u2(x, s4, m4, t4)
        x = self.u3(x, s3, m3, t3)
        x = self.u4(x, s2, m2, t2)
        x = self.u5(x, s1, m1, t1)

        # 5) sortie
        return self.final(x)

    # ------------------- intégration SupHeads ----------------------

    def attach_sup_heads(self, tasks: dict, in_dim: int) -> None:
        """
        Petit MLP générique pour debug/compat.
        Dans l’entraînement final, tu peux le remplacer par ta vraie classe SupHeads.
        """
        class _SimpleHeads(nn.Module):
            def __init__(self, tasks, in_dim):
                super().__init__()
                self.tasks = dict(tasks)
                self.in_dim = int(in_dim)
                self.classifiers = nn.ModuleDict(
                    {
                        t: nn.Sequential(
                            nn.LayerNorm(self.in_dim),
                            nn.Linear(self.in_dim, 2 * self.in_dim),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.1),
                            nn.Linear(2 * self.in_dim, int(nc)),
                        )
                        for t, nc in self.tasks.items()
                    }
                )

            @staticmethod
            def _ensure_flat(feats: torch.Tensor) -> torch.Tensor:
                if feats.dim() == 2:
                    return feats
                if feats.dim() == 3:
                    # (B,S,D) → mean-pool sur S
                    return feats.mean(dim=1)
                # sinon, GAP sur carte 2D
                return F.adaptive_avg_pool2d(feats, 1).flatten(1)

            def forward(
                self,
                feats: torch.Tensor,
                *,
                return_attn: bool = False,
                return_task_embeddings: bool = False,
            ):
                x = self._ensure_flat(feats)
                logits = {t: clf(x) for t, clf in self.classifiers.items()}
                if return_task_embeddings:
                    embs = {t: x for t in self.classifiers.keys()}
                    return logits, embs
                if return_attn:
                    return logits, {}
                return logits

        self.sup_heads = _SimpleHeads(tasks, in_dim)

    def sup_in_dim_for(self, feat_type: str) -> int:
        """
        Donne la dimension `in_dim` attendue par SupHeads pour un feat_type donné.
        Utilisé par helpers.run_* pour initialiser SupHeads.
        """
        Cb = self.ngf * 8
        # --- STYLE-BIASED ---
        if feat_type in ("style_tok", "tokG"):
            return self.hid_dim
        if feat_type == "tok6":
            return 6 * self.hid_dim
        if feat_type in ("tok6_mean", "tok6_w"):
            return self.hid_dim
        if feat_type == "bot+tok":
            return Cb + self.hid_dim
        if feat_type == "tok+delta":
            return self.hid_dim + sum(self.skip_channels)

        # --- CONTENT-BIASED ---
        # cont_tok : séquence de tokens (B, S, D) mais D = hid_dim
        # cont_tok_vit : embedding global agrégé (B, D)
        if feat_type.startswith("cont_tok"):
            return self.hid_dim

        # défaut: bottleneck GAP
        return Cb

    # -------------------- features pour SupHeads -------------------

    def sup_features(
        self,
        imgs: torch.Tensor,
        feat_type: str,
        *,
        delta_weights: Union[str, List[float], Tuple[float, ...]] = "1,1,1,1,1",
    ) -> torch.Tensor:
        """
        Retourne des features pour SupHeads selon feat_type :

          --- STYLE-BIASED ---
          - "tokG" / "style_tok"     : token global                         → (B, D)
          - "tok6"                   : concat [tokG, t5, t4, t3, t2, t1]    → (B, 6*D)
          - "tok6_mean" / "tok6_w"  : moyenne (pondérée) des 6 tokens       → (B, D)
          - "bot+tok"               : GAP(bottleneck) ⊕ tokG                → (B, 8*ngf + D)
          - "tok+delta"             : tokG ⊕ concat_i [ w_i * GAP(|m_i|) ]  → (B, D + Σ skip_channels)

          --- CONTENT-BIASED ---
          - "cont_tok"              : tokens de contenu multi-échelles      → (B, S, D)
          - "cont_tok_vit"         : tokens → Transformer + pooling         → (B, D)

          --- FALLBACK ---
          - sinon                   : GAP(bottleneck)                       → (B, 8*ngf)
        """
        # --- cas CONTENT en premier pour éviter du travail inutile ---
        if feat_type == "cont_tok":
            # Séquence de tokens de contenu (B, S, D)
            tokens = self.content_tok_encoder(imgs)  # (B,S,D)
            tokens = self._l2_norm_lastdim(tokens)
            return tokens

        if feat_type == "cont_tok_vit":
            # Multi-scale tokens + petit Transformer façon ViT → (B,D)
            tokens = self.content_tok_encoder(imgs)  # (B,S,D)
            tokens = self._l2_norm_lastdim(tokens)
            feats = self.content_tok_agg(tokens)     # (B,D)
            feats = self._l2_norm_rows(feats)
            return feats

        # --- STYLE / MIX / FALLBACK ---
        # encodeur contenu
        z, _ = self.encode_content(imgs)  # bottleneck (B, 8*ngf, h, w)

        # tokens & maps côté style-encoder (API tolérante)
        maps, toks, tokG = self._style_tokens_maps(imgs)

        # --- alias faciles ---
        if feat_type in ("style_tok", "tokG"):
            if tokG is None:
                # fallback: bot si tokG indispo
                return self._gap(z)
            return self._l2_norm_rows(tokG)  # (B,D)

        # --- concat 6 tokens ---
        if feat_type == "tok6":
            # tolère l’absence des tokens locaux
            if tokG is None and (toks is None or len(toks) == 0):
                return self._gap(z)
            if tokG is None and toks is not None and len(toks) > 0:
                tokG = toks[0].new_zeros(toks[0].shape[0], toks[0].shape[1])
            seq = [self._l2_norm_rows(tokG)] + (
                [self._l2_norm_rows(t) for t in toks] if toks is not None else []
            )
            return torch.cat(seq, dim=1)  # (B, 6*D)

        # --- moyenne (uniforme/pondérée) des 6 tokens ---
        if feat_type in ("tok6_mean", "tok6_w"):
            if tokG is None and (toks is None or len(toks) == 0):
                return self._gap(z)
            if tokG is None and toks is not None and len(toks) > 0:
                tokG = toks[0].new_zeros(toks[0].shape[0], toks[0].shape[1])

            seq = [self._l2_norm_rows(tokG)] + (
                [self._l2_norm_rows(t) for t in toks] if toks is not None else []
            )  # 6 × (B, D)
            S = torch.stack(seq, dim=1)  # (B, 6, D)

            if feat_type == "tok6_mean":
                return S.mean(dim=1)  # (B, D)

            # tok6_w: 6 poids wG,w5,w4,w3,w2,w1
            w6 = self._parse_delta_weights(delta_weights, 6)
            W = torch.tensor(w6, device=imgs.device, dtype=S.dtype).view(1, -1, 1)
            W = W / (W.sum() + 1e-8)
            return (S * W).sum(1)  # (B, D)

        # --- mix bottleneck + token ---
        if feat_type == "bot+tok":
            if tokG is None:
                return self._gap(z)
            return torch.cat([self._gap(z), self._l2_norm_rows(tokG)], dim=1)

        # --- tok + deltas sur cartes SPADE (m5..m1) ---
        if feat_type == "tok+delta":
            # besoin de tokG et maps; sinon fallback
            if tokG is None or maps is None or len(maps) == 0:
                return (
                    self._gap(z)
                    if tokG is None
                    else torch.cat([self._l2_norm_rows(tokG), self._gap(z)], dim=1)
                )

            ws = self._parse_delta_weights(delta_weights, 5)

            # initialisation lazy des heads (proj m_i → c_skip[i])
            if self._delta_heads is None:
                in_ch = int(maps[0].size(1))
                outs = self.skip_channels  # [8ngf, 8ngf, 4ngf, 2ngf, ngf]
                self._delta_heads = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Conv2d(in_ch, in_ch, 3, 1, 1),
                            nn.ReLU(True),
                            nn.Conv2d(in_ch, c, 3, 1, 1),
                        )
                        for c in outs
                    ]
                ).to(imgs.device)

            dvecs = []
            for w, head, m in zip(ws, self._delta_heads, maps):
                g = head(m)  # (B, c_skip, h, w)
                dvecs.append(w * self._gap(g.abs()))  # (B, c_skip)
            dvec = torch.cat(dvecs, dim=1)  # (B, Σ c_skip)
            return torch.cat([self._l2_norm_rows(tokG), dvec], dim=1)

        # --- fallback: bottleneck GAP ---
        return self._gap(z)
