# models/losses_nce.py
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────────── SWD/FFT textures ─────────────────────────────

def _laplacian_pyramid(x: torch.Tensor, levels: int = 3):
    pyr, cur = [], x
    L = max(1, int(levels))
    for _ in range(max(1, L - 1)):
        h, w = cur.shape[-2], cur.shape[-1]
        if h < 2 or w < 2:
            break
        down = F.avg_pool2d(cur, kernel_size=2, stride=2, ceil_mode=True)
        up = F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)
        pyr.append(cur - up)
        cur = down
    pyr.append(cur)
    return pyr


def _max_pyr_levels_from_hw(h: int, w: int) -> int:
    return 1 + int(math.floor(math.log2(max(1, min(h, w)))))


def _random_projections(dim: int, proj: int, device: torch.device):
    W = torch.randn(dim, proj, device=device)
    return W / (W.norm(dim=0, keepdim=True) + 1e-12)


def _sample_patches(x: torch.Tensor, patch: int, max_patches: int):
    B, C, H, W = x.shape
    k = min(patch, H, W)
    s = max(k // 2, 1)

    patches = (
        F.unfold(x, kernel_size=k, stride=s)
        .transpose(1, 2)
        .contiguous()
        .view(-1, C * k * k)
    )

    if patches.numel() == 0:
        pooled = F.adaptive_avg_pool2d(x, output_size=(k, k))
        patches = pooled.view(B, -1)

    N = patches.size(0)
    if max_patches and N > max_patches:
        idx = torch.randperm(N, device=patches.device)[:max_patches]
        patches = patches[idx]

    mu = patches.mean(dim=1, keepdim=True)
    sd = patches.std(dim=1, keepdim=True, unbiased=False)
    return (patches - mu) / (sd + 1e-6)


def _swd_single_level(a: torch.Tensor, b: torch.Tensor, *, patch=64, proj=128, max_patches=64):
    assert a.shape == b.shape
    A = _sample_patches(a, patch=patch, max_patches=max_patches)
    B = _sample_patches(b, patch=patch, max_patches=max_patches)
    N = min(A.size(0), B.size(0))
    if N == 0:
        return torch.zeros((), device=a.device)

    if A.size(0) != N:
        A = A[torch.randperm(A.size(0), device=a.device)[:N]]
    if B.size(0) != N:
        B = B[torch.randperm(B.size(0), device=b.device)[:N]]

    W = _random_projections(A.size(1), proj=proj, device=a.device)
    Ap, Bp = A @ W, B @ W
    Ap_sorted, _ = torch.sort(Ap, dim=0)
    Bp_sorted, _ = torch.sort(Bp, dim=0)
    return (Ap_sorted - Bp_sorted).abs().mean(dim=0).mean()


class PatchNCELoss(nn.Module):
    """
    InfoNCE sur des patches (feat_q ↔ feat_k) avec :
      • négatifs *intra-image*  (positions permutées dans la même image),
      • négatifs *inter-image*  (patches d’autres images du batch),
      • sous-échantillonnage aléatoire de N≤H×W patches pour limiter le coût.

    Parameters
    ----------
    temperature      : τ dans l’InfoNCE.
    use_intra_neg    : ajoute des négatifs issus de la même image.
    use_inter_neg    : ajoute des négatifs issus d’images différentes.
    max_patches      : si renseigné, on tire au hasard `max_patches`
                       positions par image au lieu d’utiliser toute la carte.
    """
    def __init__(
        self,
        temperature: float = 0.07,
        use_intra_neg: bool = True,
        use_inter_neg: bool = False,
        max_patches: Optional[int] = None,
    ):
        super().__init__()
        self.t = float(temperature)
        self.use_intra_neg = bool(use_intra_neg)
        self.use_inter_neg = bool(use_inter_neg)
        self.max_patches = max_patches

    # ------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------
    @staticmethod
    def _sample_patches(feat: torch.Tensor, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        feat : (B,C,H,W)  –>  (B,C,N), idx (B,N)
        On renvoie la même sélection pour toutes les cartes q/k afin
        que l’indice du *positif* reste bien aligné.
        """
        B, C, H, W = feat.shape
        HW = H * W
        idx = torch.randperm(HW, device=feat.device)[:N]  # (N,)
        feat_flat = feat.reshape(B, C, HW)                # (B,C,HW)
        patches = feat_flat[:, :, idx]                    # (B,C,N)
        return patches, idx

    @staticmethod
    def _permute_spatial(x: torch.Tensor) -> torch.Tensor:
        """Permute les positions spatiales indépendamment par image."""
        B, N, C = x.shape
        perm = torch.stack([torch.randperm(N, device=x.device) for _ in range(B)])
        return x[torch.arange(B, device=x.device).unsqueeze(1), perm]

    # ------------------------------------------------------------
    # forward
    # ------------------------------------------------------------
    def forward(self, feat_q: torch.Tensor, feat_k: torch.Tensor) -> torch.Tensor:
        """
        feat_q / feat_k : (B,C,H,W)
        Retourne : CE(mean) sur (B*N, 1+neg) avec alignement positionnel.
        """
        B, C, H, W = feat_q.shape
        HW = H * W

        # ---------- (facultatif) sous-échantillonnage ----------
        if self.max_patches and self.max_patches < HW:
            N = int(self.max_patches)
            q, idx = self._sample_patches(feat_q, N)          # (B,C,N)
            k = feat_k.reshape(B, C, HW)[:, :, idx]           # mêmes positions
        else:
            N = HW
            q = feat_q.reshape(B, C, N)
            k = feat_k.reshape(B, C, N)

        # ---------- réarrangement (B,N,C) ----------
        q = q.permute(0, 2, 1).contiguous()  # (B,N,C)
        k = k.permute(0, 2, 1).contiguous()

        # ---------- négatifs ----------
        k_all = [k]  # positifs en tête
        if self.use_intra_neg:
            k_all.append(self._permute_spatial(k))
        if self.use_inter_neg and B > 1:
            perm = torch.randperm(B, device=q.device)
            k_all.append(k[perm])
        k_all = torch.cat(k_all, dim=1)  # (B, N*(1+neg), C)

        # ---------- logits & labels ----------
        logits = torch.bmm(q, k_all.transpose(1, 2)) / self.t  # (B,N,M)
        logits = logits.reshape(-1, logits.size(-1))           # (B*N, M)
        labels = torch.arange(N, device=q.device).repeat(B)    # positifs = 0..N-1

        return F.cross_entropy(logits, labels, reduction="mean")


def fft_texture_loss(a: torch.Tensor, b: torch.Tensor, *, log_mag: bool = True, per_channel: bool = True):
    def _amp(x):
        X = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        A = torch.abs(X).clamp_min(1e-8)
        return torch.log(A) if log_mag else A

    Aa, Bb = _amp(a), _amp(b)
    if not per_channel:
        Aa = Aa.mean(dim=1, keepdim=True)
        Bb = Bb.mean(dim=1, keepdim=True)
    return F.l1_loss(Aa, Bb)


def swd_loss_images(a: torch.Tensor, b: torch.Tensor, levels=3, patch=64, proj=128, max_patches=64):
    assert a.shape == b.shape
    _, _, H, W = a.shape

    # levels can be: int | "1,2,3" | "3" | list/tuple
    if isinstance(levels, str):
        parts = [s.strip() for s in levels.split(",") if s.strip()]
        if len(parts) == 0:
            L_req = 3
        elif len(parts) == 1:
            # string "3" means 3 levels
            try:
                L_req = int(parts[0])
            except Exception:
                L_req = 3
        else:
            # "1,2,3" => number of bands requested = len(parts)
            L_req = len(parts)
    elif isinstance(levels, (list, tuple)):
        L_req = len(levels)
    else:
        L_req = int(levels)

    L_use = min(max(1, int(L_req)), _max_pyr_levels_from_hw(H, W))

    pa = _laplacian_pyramid(a, levels=L_use)
    pb = _laplacian_pyramid(b, levels=L_use)

    loss = 0.0
    for band_a, band_b in zip(pa, pb):
        loss = loss + _swd_single_level(band_a, band_b, patch=patch, proj=proj, max_patches=max_patches)

    return loss / float(len(pa))


def spectral_noise(x: torch.Tensor, sigma: float = 0.1, gamma: float = 1.0):
    if sigma <= 0:
        return x
    _, _, H, W = x.shape
    eps = 1e-8
    X = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
    amp = torch.abs(X).clamp_min(eps)

    fy = torch.fft.fftfreq(H, d=1.0, device=x.device).abs().view(1, 1, H, 1)
    fx = torch.fft.rfftfreq(W, d=1.0, device=x.device).abs().view(1, 1, 1, W // 2 + 1)
    R = torch.sqrt(fy ** 2 + fx ** 2)
    R = (R / (R.max().clamp_min(1e-6))) ** gamma

    log_amp = torch.log(amp)
    noise = torch.randn_like(log_amp) * sigma * R
    log_amp_n = log_amp + noise
    amp_n = torch.exp(log_amp_n)
    Xn = X * (amp_n / amp)

    return torch.fft.irfft2(Xn, s=(H, W), dim=(-2, -1), norm="ortho")


def highpass(x: torch.Tensor):
    """Passe-haut (Laplacien 3x3) pour focaliser D sur les détails."""
    k = torch.tensor([[0., -1., 0.],
                      [-1., 4., -1.],
                      [0., -1., 0.]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    k = k.repeat(x.size(1), 1, 1, 1)
    return F.conv2d(x, k, padding=1, groups=x.size(1))
