# file: models/generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union, List

from models.backbone import Down, ResBlock, ConvBlock, StylePyramidEncoder, SPADEUp, _downsample_plan
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
        *,
        arch_depth_delta: int = 0,
        style_token_levels: int = -1,
        img_size: int = 256,
        unet_min_spatial: int = 2,
    ):
        super().__init__()
        self.ngf = int(ngf)
        self.hid_dim = int(token_dim)  # dim tokens style + contenu
        self.nc = int(nc)

        # ---------------- architecture scaling ----------------
        base_levels = 5
        depth_delta = int(arch_depth_delta)
        L = int(style_token_levels) if int(style_token_levels) > 0 else (base_levels + depth_delta)
        L = max(1, L)
        self.style_levels = L               # nb tokens locaux (tL..t1)
        self.unet_levels = L                # nb niveaux de skip (s1..sL)

        # schedule strides to avoid 1x1 (InstanceNorm crash) when deepening
        # downs count = unet_levels + 1 (extra bottleneck down)
        # En profondeur >0, on stoppe le downsampling dès 4x4 (règle)
        _stop_at = int(unet_min_spatial)
        if depth_delta > 0:
            _stop_at = max(_stop_at, 4)
        else:
            # comportement historique
            _stop_at = 2
        down_plan = _downsample_plan(num_downs=self.unet_levels + 1, img_size=int(img_size), stop_down_at=_stop_at)
        self._down_plan = down_plan  # keep for decoder upsample schedule

        def ch_for_level(lvl: int) -> int:
            # lvl: 1..L (+ bottleneck L+1)
            if lvl == 1: return self.ngf
            if lvl == 2: return self.ngf * 2
            if lvl == 3: return self.ngf * 4
            return self.ngf * 8

        # tailles des skips (sL..s1) côté encodeur (deep -> shallow)
        self.skip_channels: List[int] = [ch_for_level(lvl) for lvl in range(self.unet_levels, 0, -1)]

        # --------------------- encodeur contenu (variable) ---------------------
        # On crée d1..dK comme attributs (compat state_dict), et on garde une liste `downs`.
        self.downs = []  # list[Down] pour itération
        in_ch = self.nc
        for lvl in range(1, self.unet_levels + 2):
            out_ch = ch_for_level(lvl)
            stride, dil = down_plan[lvl-1]
            m = Down(in_ch, out_ch, stride=stride, dilation=dil)
            setattr(self, f"d{lvl}", m)
            self.downs.append(m)
            in_ch = out_ch

        # resblocks near bottom + bottleneck conv
        self.res_skip = ResBlock(ch_for_level(self.unet_levels))
        self.res_bot  = ResBlock(ch_for_level(self.unet_levels + 1))
        self.bot = ConvBlock(ch_for_level(self.unet_levels + 1), ch_for_level(self.unet_levels + 1))

        # ------------------ encodeur style / SPADE (scales variables) ------------------
        self.style_enc = StylePyramidEncoder(
            style_nc,
            self.ngf,
            spade_ch,
            token_dim,
            num_levels=self.style_levels,
            img_size=int(img_size),
            min_spatial=_stop_at,
        )

        # ------------------------ décodeur (variable) --------------------------
        # On crée u1..uL comme attributs (compat state_dict), et on garde une liste `ups`.
        self.ups = []  # list[SPADEUp] pour itération
        c_up = ch_for_level(self.unet_levels + 1)  # bottleneck
        idx = 1
        # Upsample factors mirror encoder strides: first matches bottleneck extra down, then each down from level L..2
        up_factors = [down_plan[self.unet_levels][0]] + [down_plan[lvl-1][0] for lvl in range(self.unet_levels, 1, -1)]
        for lvl in range(self.unet_levels, 0, -1):
            c_skip = ch_for_level(lvl)
            c_out  = ch_for_level(lvl)
            sf = int(up_factors[idx-1])
            m = SPADEUp(c_up=c_up, c_skip=c_skip, c_out=c_out, s_ch=spade_ch, token_dim=token_dim, scale_factor=sf)
            setattr(self, f"u{idx}", m)
            self.ups.append(m)
            c_up = c_out
            idx += 1

        self.final = nn.Sequential(
            nn.Conv2d(ch_for_level(1), ch_for_level(1), 3, 1, 1),
            nn.ReLU(True),
            (nn.Upsample(scale_factor=int(down_plan[0][0]), mode="nearest") if int(down_plan[0][0]) == 2 else nn.Identity()),
            nn.Conv2d(ch_for_level(1), self.nc, 3, 1, 1),
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

        # debug shapes
        self.debug_shapes: bool = False
        self._dbg_shapes_done: bool = False

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

    @staticmethod
    def _trace_shape(enabled: bool, name: str, t: torch.Tensor) -> None:
        if not enabled:
            return
        try:
            print(f"[shapes] {name}: {tuple(t.shape)}", flush=True)
        except Exception:
            pass


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
          - z:   bottleneck (B, Cb, h, w)
          - skips: (s1..sL) où L=self.unet_levels
        """
        skips = []
        h = x
        # levels 1..L
        for lvl in range(1, self.unet_levels + 1):
            h = self.downs[lvl-1](h)
            if self.debug_shapes and not self._dbg_shapes_done:
                self._trace_shape(True, f'G/enc/s{lvl}', h)
            skips.append(h)
        # resblock on deepest skip
        skips[-1] = self.res_skip(skips[-1])
        # bottleneck extra down
        z = self.downs[self.unet_levels](skips[-1])
        z = self.res_bot(z)
        z = self.bot(z)
        if self.debug_shapes and not self._dbg_shapes_done:
            self._trace_shape(True, 'G/enc/z', z)
        return z, tuple(skips)

    @torch.no_grad()
    def build_style(self, style_imgs: torch.Tensor):
        maps, toks, tokG = self._style_tokens_maps(style_imgs)
        return {"maps": maps, "tokens": toks, "token": tokG}

    def forward(self, x: torch.Tensor, *, style: Optional[Union[torch.Tensor, dict]] = None):
        # 1) contenu
        z, skips = self.encode_content(x)  # skips: (s1..sL)

        # 2) tokens de style (issus de l'image style)
        if style is None:
            _, toks, tokG = self._style_tokens_maps(x)  # identité
        elif isinstance(style, torch.Tensor):
            _, toks, tokG = self._style_tokens_maps(style)
        elif isinstance(style, dict):
            toks = style.get("tokens", None)
            tokG = style.get("token", None)
            if toks is None and tokG is None:
                _, toks, tokG = self._style_tokens_maps(x)
            elif toks is None and tokG is not None:
                # répéter tokG sur L tokens locaux
                toks = tuple(tokG for _ in range(self.style_levels))
            elif toks is not None and tokG is None:
                comps = []
                for t in toks:
                    comps.append(t[0] if (isinstance(t, (tuple, list)) and len(t) == 2) else t)
                tokG = torch.stack(comps, dim=0).mean(0)
        else:
            raise TypeError("style doit être None, Tensor(y), ou dict{'tokens':(tL..t1), 'token':tokG}.")

        # 3) cartes SPADE depuis le contenu (hL..h1)
        # skips are (s1..sL). We need deep->shallow ordering.
        skips_ds = list(skips)[::-1]  # sL..s1
        maps_ds = [getattr(self.style_enc, f'h{lvl}')(skips[lvl-1]) for lvl in range(self.style_levels, 0, -1)]

        # tokens locaux deep->shallow
        toks_ds = list(toks) if isinstance(toks, (list, tuple)) else []
        # tolère ancien format (len=5). Si mismatch, on pad/trim.
        if len(toks_ds) == 0:
            toks_ds = [tokG for _ in range(self.style_levels)] if tokG is not None else [None] * self.style_levels
        if len(toks_ds) != self.style_levels:
            if len(toks_ds) < self.style_levels:
                pad = toks_ds[-1] if len(toks_ds) else (tokG if tokG is not None else None)
                toks_ds = (toks_ds + [pad] * self.style_levels)[: self.style_levels]
            else:
                toks_ds = toks_ds[: self.style_levels]

        if self.debug_shapes and not self._dbg_shapes_done:
            if tokG is not None:
                self._trace_shape(True, 'G/style/tokG', tokG)
            try:
                if isinstance(toks_ds, (list, tuple)) and len(toks_ds):
                    t0 = toks_ds[0][0] if (isinstance(toks_ds[0], (tuple, list)) and len(toks_ds[0])==2) else toks_ds[0]
                    if hasattr(t0, 'shape'):
                        self._trace_shape(True, 'G/style/tok_local0', t0)
            except Exception:
                pass

        # 4) décodeur conditionné
        h = z
        for up, s, m, t in zip(self.ups, skips_ds, maps_ds, toks_ds):
            h = up(h, s, m, t)

        # 5) sortie
        out = self.final(h)
        if self.debug_shapes and not self._dbg_shapes_done:
            self._trace_shape(True, "G/out", out)
            self._dbg_shapes_done = True
        return out


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
        if feat_type in ("tok6", "tokL"):
            return (1 + self.style_levels) * self.hid_dim
        if feat_type in ("tok6_mean", "tok6_w", "tokL_mean", "tokL_w"):
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
            return self._l2_norm_rows(tokG)  # (B,D)        # --- concat multi-tokens (tokG + tL..t1) ---
        if feat_type in ("tok6", "tokL"):
            # tolère l’absence des tokens locaux
            if tokG is None and (toks is None or len(toks) == 0):
                return self._gap(z)
            if tokG is None and toks is not None and len(toks) > 0:
                tokG = toks[0].new_zeros(toks[0].shape[0], toks[0].shape[1])

            toks_list = list(toks) if isinstance(toks, (list, tuple)) else []
            # ajuste à self.style_levels
            if len(toks_list) < self.style_levels:
                pad = toks_list[-1] if len(toks_list) else tokG
                toks_list = (toks_list + [pad] * self.style_levels)[: self.style_levels]
            elif len(toks_list) > self.style_levels:
                toks_list = toks_list[: self.style_levels]

            seq = [self._l2_norm_rows(tokG)] + [self._l2_norm_rows(t) for t in toks_list]
            return torch.cat(seq, dim=1)  # (B, (1+L)*D)

        # --- moyenne (uniforme/pondérée) des tokens ---
        if feat_type in ("tok6_mean", "tok6_w", "tokL_mean", "tokL_w"):
            if tokG is None and (toks is None or len(toks) == 0):
                return self._gap(z)
            if tokG is None and toks is not None and len(toks) > 0:
                tokG = toks[0].new_zeros(toks[0].shape[0], toks[0].shape[1])

            toks_list = list(toks) if isinstance(toks, (list, tuple)) else []
            if len(toks_list) < self.style_levels:
                pad = toks_list[-1] if len(toks_list) else tokG
                toks_list = (toks_list + [pad] * self.style_levels)[: self.style_levels]
            elif len(toks_list) > self.style_levels:
                toks_list = toks_list[: self.style_levels]

            seq = [self._l2_norm_rows(tokG)] + [self._l2_norm_rows(t) for t in toks_list]
            S = torch.stack(seq, dim=1)  # (B, 1+L, D)

            if feat_type in ("tok6_mean", "tokL_mean"):
                return S.mean(dim=1)

            # pondéré: delta_weights fournit (1+L) poids
            wN = self._parse_delta_weights(delta_weights, 1 + self.style_levels)
            W = torch.tensor(wN, device=imgs.device, dtype=S.dtype).view(1, -1, 1)
            W = W / (W.sum() + 1e-8)
            return (S * W).sum(1)

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

            ws = self._parse_delta_weights(delta_weights, self.style_levels)

            # initialisation lazy des heads (proj m_i → c_skip[i])
            if self._delta_heads is None:
                in_ch = int(maps[0].size(1))
                outs = self.skip_channels  # (deep->shallow), length = style_levels
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