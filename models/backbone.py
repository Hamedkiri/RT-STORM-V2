import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ──────────────────────────────────────────────────────────────
#  Blocs de base
# ──────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, ks=3, stride=1, pad=1, norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, ks, stride, pad)]
        if norm:
            layers.append(nn.InstanceNorm2d(out_c, affine=True))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c, ks=4, stride=2, pad=1)

    def forward(self, x):
        return self.conv(x)

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, 1, 1)
        self.n1 = nn.InstanceNorm2d(c, affine=True)
        self.c2 = nn.Conv2d(c, c, 3, 1, 1)
        self.n2 = nn.InstanceNorm2d(c, affine=True)

    def forward(self, x):
        y = F.relu(self.n1(self.c1(x)), inplace=True)
        y = self.n2(self.c2(y))
        return F.relu(x + y, inplace=True)

# ──────────────────────────────────────────────────────────────
#  SPADE (FiLM global + cartes spatiales) – version « dé-bridée »
# ──────────────────────────────────────────────────────────────

def _inv_softplus(y: float) -> float:

    # renvoie x tel que softplus(x)=y (y>0)
    y = float(y)
    return math.log(max(math.exp(y) - 1.0, 1e-8))

class SPADELayer(nn.Module):
    def __init__(self, c, s_ch, token_dim, hidden=128,
                 # gains plus élevés côté cartes (et tokens équilibrés)
                 init_wg_gamma=1.0, init_ws_gamma=1.0,
                 init_wg_beta=1.0,  init_ws_beta=1.0):
        super().__init__()
        self.norm = nn.InstanceNorm2d(c, affine=False, eps=1e-5)

        self.s_conv = nn.Sequential(
            nn.Conv2d(s_ch, hidden, 3, 1, 1), nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 2 * c, 3, 1, 1)
        )
        self.g_mlp = nn.Sequential(
            nn.Linear(token_dim, hidden), nn.SiLU(inplace=True),
            nn.Linear(hidden, 2 * c)
        )

        self._sp = nn.Softplus()
        def _inv_sp(y): return math.log(max(math.exp(float(y)) - 1.0, 1e-8))
        self._p_ws_gamma = nn.Parameter(torch.tensor(_inv_sp(init_ws_gamma)))
        self._p_wg_gamma = nn.Parameter(torch.tensor(_inv_sp(init_wg_gamma)))
        self._p_ws_beta  = nn.Parameter(torch.tensor(_inv_sp(init_ws_beta)))
        self._p_wg_beta  = nn.Parameter(torch.tensor(_inv_sp(init_wg_beta)))

        # init: ReLU/Kaiming pour convs, Xavier pour linears
        for m in self.s_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
        for m in self.g_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=2.0)
                if m.bias is not None: nn.init.zeros_(m.bias)
        # ⚠️ on NE bride plus la dernière conv des cartes (plus de *.mul_(0.3))

    def forward(self, x, style_map, style_token):
        B, C, H, W = x.shape
        x_n = self.norm(x)

        # cartes spatiales (depuis style_map)
        sb = self.s_conv(style_map)
        gam_s, bet_s = torch.chunk(sb, 2, dim=1)

        # token global/multi-échelle (éventuel gain post-MLP)
        token_gain = 1.0
        if isinstance(style_token, (tuple, list)) and len(style_token) == 2:
            style_token, token_gain = style_token
        gb = self.g_mlp(style_token) * token_gain
        gam_g, bet_g = torch.chunk(gb, 2, dim=1)
        gam_g = gam_g.view(B, C, 1, 1)
        bet_g = bet_g.view(B, C, 1, 1)

        # calibration: **adoucie** côté cartes (pas de 1/√C brutal)
        # ≈1.0 à 64ch ; ≈0.5 à 256ch ; ≈0.35 à 512ch
        sc_s = 1.0 / math.sqrt(max(C / 64.0, 1.0))
        gam_s, bet_s = gam_s * sc_s, bet_s * sc_s

        ws_gam = self._sp(self._p_ws_gamma); wg_gam = self._sp(self._p_wg_gamma)
        ws_bet = self._sp(self._p_ws_beta ); wg_bet = self._sp(self._p_wg_beta )

        gam = 1.0 + ws_gam * gam_s + wg_gam * gam_g
        bet =        ws_bet * bet_s + wg_bet * bet_g
        return x_n * gam + bet

class SPADEResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, s_ch: int, token_dim: int):
        super().__init__()
        self.spade1 = SPADELayer(c_in,  s_ch, token_dim)
        self.conv1  = nn.Conv2d(c_in,   c_out, 3, 1, 1)
        self.spade2 = SPADELayer(c_out, s_ch, token_dim)
        self.conv2  = nn.Conv2d(c_out,  c_out, 3, 1, 1)
        self.skip = None
        if c_in != c_out:
            self.skip = nn.Conv2d(c_in, c_out, 1, 1, 0)

    def forward(self, x, style_map, style_token):
        y = F.relu(self.spade1(x, style_map, style_token), inplace=True)
        y = self.conv1(y)
        y = F.relu(self.spade2(y, style_map, style_token), inplace=True)
        y = self.conv2(y)
        s = x if self.skip is None else self.skip(x)
        return F.relu(y + s, inplace=True)

class SPADEUp(nn.Module):
    """ Upsample ×2 → concat skip → SPADEResBlock """
    def __init__(self, c_up: int, c_skip: int, c_out: int, s_ch: int, token_dim: int):
        super().__init__()
        self.up  = nn.Upsample(scale_factor=2, mode="nearest")
        self.res = SPADEResBlock(c_up + c_skip, c_out, s_ch, token_dim)

    def forward(self, x, skip, style_map, style_token):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x, style_map, style_token)

# ──────────────────────────────────────────────────────────────
#  Encodeur de style pyramidal (maps + tokens multi-échelles + token global)
# ──────────────────────────────────────────────────────────────

class StylePyramidEncoder(nn.Module):
    """Encodeur pyramidal de style (scales variables).

    Retourne:
      - maps:  (mL, ..., m1)  [s_ch canaux]
      - toks:  (tL, ..., t1)  [token_dim]
      - tokG:  token global (bottleneck)

    Par défaut, L=5 (comportement historique).
    """

    def __init__(self, in_nc: int, base: int, s_ch: int, token_dim: int, *, num_levels: int = 5):
        super().__init__()
        ngf = int(base)
        self.num_levels = int(num_levels)
        assert self.num_levels >= 1

        def ch_for_level(lvl: int) -> int:
            if lvl == 1: return ngf
            if lvl == 2: return ngf * 2
            if lvl == 3: return ngf * 4
            return ngf * 8

        # d1..dK (K = L+1) : L niveaux de skip + 1 down bottleneck
        self.downs = []  # list[Down] pour itération
        in_ch = in_nc
        for lvl in range(1, self.num_levels + 2):
            out_ch = ch_for_level(lvl)
            m = Down(in_ch, out_ch)
            setattr(self, f"d{lvl}", m)
            self.downs.append(m)
            in_ch = out_ch

        self.res_skip = ResBlock(ch_for_level(self.num_levels))
        self.res_bot = ResBlock(ch_for_level(self.num_levels + 1))
        self.bot = ConvBlock(ch_for_level(self.num_levels + 1), ch_for_level(self.num_levels + 1))

        # heads per skip level (registered only once)
        self._h_list = []
        self._t_list = []
        for lvl in range(1, self.num_levels + 1):
            h = nn.Conv2d(ch_for_level(lvl), s_ch, 3, 1, 1)
            t = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch_for_level(lvl), token_dim, 1))
            setattr(self, f"h{lvl}", h)
            setattr(self, f"t{lvl}", t)
            self._h_list.append(h)
            self._t_list.append(t)

        self.tokG_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch_for_level(self.num_levels + 1), token_dim, 1))

    def forward(self, img: torch.Tensor):
        skips = []
        x = img
        for lvl in range(1, self.num_levels + 1):
            x = self.downs[lvl-1](x)
            skips.append(x)

        skips[-1] = self.res_skip(skips[-1])

        z = self.downs[self.num_levels](skips[-1])
        z = self.res_bot(z)
        z = self.bot(z)

        maps = []
        toks = []
        for lvl in range(self.num_levels, 0, -1):
            s = skips[lvl-1]
            h = getattr(self, f"h{lvl}")
            t = getattr(self, f"t{lvl}")
            maps.append(h(s))
            toks.append(t(s).flatten(1))

        tokG = self.tokG_head(z).flatten(1)
        return tuple(maps), tuple(toks), tokG




