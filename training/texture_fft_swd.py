# training/texture_fft_swd.py
import torch
import torch.nn.functional as F
import math
from torchvision import utils as vutils

# ───────────────────────────── petits helpers ─────────────────────────────
def _gap(x: torch.Tensor) -> torch.Tensor:
    """Global Average Pooling (+ flatten) si 4D, no-op si déjà 2D."""
    if x.dim() == 4:
        return F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
    return x

def _denorm(t):  # suppose images dans [-1,1] → remet en [0,1]
    return ((t.detach().float().cpu().clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)

def _triplet_grid(a, b, c, max_k=4, nrow=3):
    K = min(max_k, a.size(0), b.size(0), c.size(0))
    rows = []
    for i in range(K):
        rows += [a[i], b[i], c[i]]
    return vutils.make_grid(torch.stack(rows, 0), nrow=nrow)

def freeze(m):
    if m is None: return
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)

def unfreeze(m):
    if m is None: return
    m.train()
    for p in m.parameters(): p.requires_grad_(True)

def count_params(m): return sum(p.numel() for p in m.parameters())

def grad_norm(m):
    s = 0.0
    for p in m.parameters():
        if p.grad is not None:
            g = p.grad.norm().item(); s += g * g
    return math.sqrt(s) if s > 0 else 0.0

# ─────────────── visualisation cartes (utilisée pour SPADE/debug) ───────────────
def _to_img_grid(maps: torch.Tensor, target_hw: tuple, k: int = 4):
    B, C, h, w = maps.shape
    k = min(k, C)
    sel = maps[:, :k].clone()
    sel = sel - sel.view(B, k, -1).min(-1)[0].view(B, k, 1, 1)
    denom = sel.view(B, k, -1).max(-1)[0].view(B, k, 1, 1).clamp_min(1e-6)
    sel = (sel / denom).clamp(0, 1)
    sel = F.interpolate(sel, size=target_hw, mode="bilinear", align_corners=False)
    return sel.view(B * k, 1, *target_hw)

# ───────────────────────────── bruit / FFT / high-pass ─────────────────────────────
def spectral_noise(x: torch.Tensor, sigma: float = 0.1, gamma: float = 1.0):
    if sigma <= 0: return x
    B, C, H, W = x.shape
    eps = 1e-8
    X = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
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
    return torch.fft.irfft2(Xn, s=(H, W), dim=(-2, -1), norm='ortho')

def fft_amp_mix(a: torch.Tensor, b: torch.Tensor, alpha: float):
    """Mix visuel (amplitude log) entre 2 images, garde phase de a."""
    assert a.shape == b.shape
    Xa = torch.fft.rfft2(a, dim=(-2, -1), norm='ortho')
    Xb = torch.fft.rfft2(b, dim=(-2, -1), norm='ortho')
    Aa = torch.log(torch.abs(Xa).clamp_min(1e-8))
    Ab = torch.log(torch.abs(Xb).clamp_min(1e-8))
    Am = torch.exp(alpha * Ab + (1 - alpha) * Aa)
    Xa_new = Am * (Xa / (torch.abs(Xa).clamp_min(1e-8)))
    return torch.fft.irfft2(Xa_new, s=a.shape[-2:], dim=(-2, -1), norm='ortho')

def highpass(x: torch.Tensor):
    """Passe-haut (Laplacien 3x3) pour focaliser D sur les détails."""
    k = torch.tensor([[0., -1., 0.],
                      [-1., 4., -1.],
                      [0., -1., 0.]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    k = k.repeat(x.size(1), 1, 1, 1)
    return F.conv2d(x, k, padding=1, groups=x.size(1))

# ───────────────────────────── SWD/FFT textures ─────────────────────────────
def _laplacian_pyramid(x: torch.Tensor, levels: int = 3):
    pyr, cur = [], x
    L = max(1, int(levels))
    for _ in range(max(1, L - 1)):
        h, w = cur.shape[-2], cur.shape[-1]
        if h < 2 or w < 2: break
        down = F.avg_pool2d(cur, kernel_size=2, stride=2, ceil_mode=True)
        up = F.interpolate(down, size=(h, w), mode='bilinear', align_corners=False)
        pyr.append(cur - up); cur = down
    pyr.append(cur); return pyr

def _max_pyr_levels_from_hw(h: int, w: int) -> int:
    return 1 + int(math.floor(math.log2(max(1, min(h, w)))))

def _random_projections(dim: int, proj: int, device: torch.device):
    W = torch.randn(dim, proj, device=device)
    return W / (W.norm(dim=0, keepdim=True) + 1e-12)

def _sample_patches(x: torch.Tensor, patch: int, max_patches: int):
    B, C, H, W = x.shape
    k = min(patch, H, W); s = max(k // 2, 1)
    patches = F.unfold(x, kernel_size=k, stride=s).transpose(1, 2).contiguous().view(-1, C * k * k)
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
    if N == 0: return torch.zeros((), device=a.device)
    if A.size(0) != N: A = A[torch.randperm(A.size(0), device=a.device)[:N]]
    if B.size(0) != N: B = B[torch.randperm(B.size(0), device=b.device)[:N]]
    W = _random_projections(A.size(1), proj=proj, device=a.device)
    Ap, Bp = A @ W, B @ W
    Ap_sorted, _ = torch.sort(Ap, dim=0)
    Bp_sorted, _ = torch.sort(Bp, dim=0)
    return (Ap_sorted - Bp_sorted).abs().mean(dim=0).mean()

def swd_loss_images(a: torch.Tensor, b: torch.Tensor, levels=3, patch=64, proj=128, max_patches=64):
    assert a.shape == b.shape
    B, C, H, W = a.shape
    L_req = (len([s for s in levels.split(",") if s.strip()]) if isinstance(levels, str)
             else (len(levels) if isinstance(levels, (list, tuple)) else int(levels)))
    L_use = min(max(1, L_req), _max_pyr_levels_from_hw(H, W))
    pa = _laplacian_pyramid(a, levels=L_use); pb = _laplacian_pyramid(b, levels=L_use)
    loss = 0.0
    for band_a, band_b in zip(pa, pb):
        loss = loss + _swd_single_level(band_a, band_b, patch=patch, proj=proj, max_patches=max_patches)
    return loss / float(len(pa))

def fft_texture_loss(a: torch.Tensor, b: torch.Tensor, *, log_mag=True, per_channel=True):
    def _amp(x):
        X = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
        A = torch.abs(X).clamp_min(1e-8)
        return torch.log(A) if log_mag else A
    Aa, Bb = _amp(a), _amp(b)
    if not per_channel:
        Aa = Aa.mean(dim=1, keepdim=True); Bb = Bb.mean(dim=1, keepdim=True)
    return F.l1_loss(Aa, Bb)