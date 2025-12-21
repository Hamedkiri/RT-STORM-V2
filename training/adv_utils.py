# training/adv_utils.py
import torch
import torch.nn.functional as F
from torch import autograd
from config import get_opts

import numpy as np

opt = get_opts()

adv_highpass = bool(getattr(opt, "adv_highpass", True))

def highpass(x: torch.Tensor):
    """Passe-haut (Laplacien 3x3) pour focaliser D sur les détails."""
    k = torch.tensor([[0., -1., 0.],
                      [-1., 4., -1.],
                      [0., -1., 0.]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    k = k.repeat(x.size(1), 1, 1, 1)
    return F.conv2d(x, k, padding=1, groups=x.size(1))

# ========== Adversarial losses (hinge / lsgan) + R1 ==========
def D_forward_logits(D, x):
    out = D(x);
    return out if isinstance(out, torch.Tensor) else out[0]


def D_loss_and_stats(D, real, fake, *, use_hinge=True, r1_gamma=0.0, do_r1=False):
    real_in = highpass(real) if adv_highpass else real
    fake_in = highpass(fake.detach()) if adv_highpass else fake.detach()
    real_logits = D_forward_logits(D, real_in)
    fake_logits = D_forward_logits(D, fake_in)
    if use_hinge:
        loss_real = F.relu(1. - real_logits).mean()
        loss_fake = F.relu(1. + fake_logits).mean()
    else:
        loss_real = F.mse_loss(real_logits, torch.ones_like(real_logits))
        loss_fake = F.mse_loss(fake_logits, torch.zeros_like(fake_logits))
    loss = loss_real + loss_fake
    r1_pen = real_logits.new_tensor(0.0)
    if do_r1 and r1_gamma > 0:
        real_in_r1 = real_in.detach().requires_grad_(True)
        r_out = D_forward_logits(D, real_in_r1)
        grad = autograd.grad(outputs=r_out.sum(), inputs=real_in_r1,
                             create_graph=True, only_inputs=True)[0]
        r1_pen = (grad.pow(2).reshape(grad.size(0), -1).sum(1)).mean()
        loss = loss + 0.5 * r1_gamma * r1_pen
    stats = {"real_mean": float(real_logits.mean().detach().cpu()),
             "fake_mean": float(fake_logits.mean().detach().cpu()),
             "r1": float(r1_pen.detach().cpu()) if do_r1 and r1_gamma > 0 else 0.0}
    return loss, stats


def G_adv_loss(D, fake, *, use_hinge=True):
    fake_in = highpass(fake) if adv_highpass else fake
    fake_logits = D_forward_logits(D, fake_in)
    return (-fake_logits.mean()) if use_hinge else F.mse_loss(fake_logits, torch.ones_like(fake_logits))


# ========================== helpers sup_freeze ==========================
def _parse_dw(dw: str, need: int) -> str:
    vals = [float(t) for t in str(dw).split(",") if t.strip()]
    if len(vals) == need: return ",".join(str(v) for v in vals)
    if len(vals) == 1:
        vals = vals * need
    elif len(vals) < need:
        vals = vals + [vals[-1]] * (need - len(vals))
    else:
        vals = vals[:need]
    return ",".join(str(v) for v in vals)


@torch.no_grad()
def _sup_metrics(out, y_true, num_classes):
    pred = out.argmax(1)
    acc = float((pred == y_true).float().mean().item()) if y_true.numel() > 0 else 0.0
    Pm = [];
    Rm = []
    for c in range(num_classes):
        tp = ((pred == c) & (y_true == c)).sum().item()
        fp = ((pred == c) & (y_true != c)).sum().item()
        fn = ((pred != c) & (y_true == c)).sum().item()
        if (tp + fp) > 0: Pm.append(tp / (tp + fp + 1e-9))
        if (tp + fn) > 0: Rm.append(tp / (tp + fn + 1e-9))
    return acc, (float(np.mean(Pm)) if Pm else 0.0), (float(np.mean(Rm)) if Rm else 0.0)


@torch.no_grad()
def _eval_on_fold(G_sup, loader, tasks, feat_type, δw_str, writer, global_step, tag_prefix):
    G_sup.eval()
    if hasattr(G_sup, "sup_heads"): G_sup.sup_heads.eval()
    agg = {t: {"ce": 0.0, "n": 0, "acc": 0.0, "P": 0.0, "R": 0.0, "H": 0.0} for t in tasks}
    for batch in loader:
        imgs, raw = (batch[0], batch[1]) if len(batch) >= 2 else batch
        imgs = imgs.to(next(G_sup.parameters()).device)
        feats = G_sup.sup_features(imgs, feat_type, delta_weights=δw_str)
        logits, attn = G_sup.sup_heads(feats, return_attn=True)
        if not isinstance(logits, dict): logits = {"default": logits}
        if not isinstance(raw, dict):    raw = {"default": raw}
        B = imgs.size(0)
        for t, out in logits.items():
            if t not in raw: continue
            y = torch.as_tensor(raw[t], device=imgs.device, dtype=torch.long)
            mask = (y >= 0) & (y < out.size(1))
            if mask.any():
                ce = F.cross_entropy(out[mask], y[mask]).item()
                acc, Pm, Rm = _sup_metrics(out[mask], y[mask], out.size(1))
                n = int(mask.sum().item())
                agg[t]["ce"] += ce * n;
                agg[t]["acc"] += acc * n
                agg[t]["P"] += Pm * n;
                agg[t]["R"] += Rm * n
                agg[t]["n"] += n
                if isinstance(attn, dict) and t in attn and "entropy" in attn[t]:
                    agg[t]["H"] += float(attn[t]["entropy"]) * B
    if writer:
        for t, d in agg.items():
            n = max(1, d["n"])
            writer.add_scalars(f"{tag_prefix}/{t}", {
                "CE": d["ce"] / n, "acc": d["acc"] / n, "P_macro": d["P"] / n,
                "R_macro": d["R"] / n, "attn_H": d["H"] / max(1, n)
            }, global_step)
    return agg






