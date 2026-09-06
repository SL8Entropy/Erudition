"""Shared components for spatial regression of a formation surface from coordinates.

Provides a Fourier-feature MLP over (X, Y), a Laplacian smoothness penalty, a single-fold
training loop, a horizontal-well CSV loader, and reconstruction of TVT from a predicted
surface plus a per-well anchor.

Imported by anchor_deploy. The winning submission does not use the surface models
themselves; the import is kept because anchor_deploy defines its older reference-surface
deployment path in terms of them.
"""

from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULTS = dict(n_freqs=64, hidden=256, n_layers=4, sigma=1.0,
                lr=1e-3, n_epochs=200, batch=65536, n_colloc=512, lam=1e-6)


class FourierMLP(nn.Module):
    def __init__(self, n_freqs=64, hidden=256, n_layers=4, sigma=1.0, out_dim=1):
        super().__init__()
        self.register_buffer("B", torch.randn(2, n_freqs) * sigma)
        layers = [nn.Linear(2 * n_freqs, hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, x):
        proj = 2 * torch.pi * x @ self.B
        ff = torch.cat([torch.sin(proj), torch.cos(proj)], -1)
        out = self.net(ff)
        return out.squeeze(-1) if self.out_dim == 1 else out


class GaussianNLL(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, y):
        mu = pred[:, 0]
        sigma = nn.functional.softplus(pred[:, 1]) + self.eps
        return (torch.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2).mean()


class LaplaceNLL(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, y):
        mu = pred[:, 0]
        b = nn.functional.softplus(pred[:, 1]) + self.eps
        return (torch.log(b) + (y - mu).abs() / b).mean()


def laplacian_penalty(model, xy):
    xy = xy.detach().float().requires_grad_(True)
    z = model(xy)
    if z.dim() > 1:
        z = z[:, 0]
    g = torch.autograd.grad(z.sum(), xy, create_graph=True, retain_graph=True)[0]
    d2x = torch.autograd.grad(g[:, 0].sum(), xy, create_graph=True, retain_graph=True)[0][:, 0]
    d2y = torch.autograd.grad(g[:, 1].sum(), xy, create_graph=True, retain_graph=True)[0][:, 1]
    return (d2x ** 2 + d2y ** 2).mean()


def train_zlayer(feat, tgt, colloc_range, *, loss_fn=None, n_epochs=200, lr=1e-3,
                 batch=65536, lam=1e-6, n_colloc=512, seed=42, model=None,
                 aux_fn=None, aux_weight=0.0, **mlp_kw):
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    torch.manual_seed(seed)
    if model is None:
        kw = {k: mlp_kw.get(k, DEFAULTS[k]) for k in ("n_freqs", "hidden", "n_layers", "sigma")}
        model = FourierMLP(**kw).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    lo, hi = colloc_range
    for _ in range(n_epochs):
        model.train()
        idx = torch.randperm(len(feat), device=DEVICE)
        for i in range(0, len(feat), batch):
            xb = feat[idx[i:i + batch]].float(); yb = tgt[idx[i:i + batch]].float()
            opt.zero_grad(set_to_none=True)
            total = loss_fn(model(xb), yb)
            if lam > 0:
                xc = torch.empty(n_colloc, 2, device=DEVICE).uniform_(lo, hi)
                total = total + lam * laplacian_penalty(model, xc)
            if aux_fn is not None and aux_weight > 0:
                total = total + aux_weight * aux_fn(model)
            total.backward(); opt.step()
        sch.step()
    model.eval()
    return model


def load_well(name, train_dir):
    df = pl.read_csv(Path(train_dir) / f"{name}__horizontal_well.csv",
                     null_values=["nan", "NaN", "NA", ""])
    fc = [c for c in df.columns if df[c].dtype in (pl.Float32, pl.Float64)]
    return df.with_columns([pl.col(c).fill_nan(None) for c in fc])


def predict_zlayer(model, x, y, norm, amp: bool = True):
    xm, xs, ym, ys, zm, zs = norm
    xn = (x - xm) / (xs + 1e-8); yn = (y - ym) / (ys + 1e-8)
    f = torch.tensor(np.stack([xn, yn], 1), dtype=torch.float32, device=DEVICE)
    with torch.no_grad(), torch.autocast("cuda", enabled=amp):
        out = model(f).float()
        zp = (out[:, 0] if out.dim() > 1 else out).cpu().numpy()
    return zp * zs + zm


def predict_sigma(model, x, y, norm, amp: bool = True):
    xm, xs, ym, ys, zm, zs = norm
    xn = (x - xm) / (xs + 1e-8); yn = (y - ym) / (ys + 1e-8)
    f = torch.tensor(np.stack([xn, yn], 1), dtype=torch.float32, device=DEVICE)
    with torch.no_grad(), torch.autocast("cuda", enabled=amp):
        out = model(f).float()
        s = (torch.nn.functional.softplus(out[:, 1]) + 1e-3).cpu().numpy()
    return s * zs


def reconstruct_anchored_tvt(model, df, norm, return_sigma=False):
    known = df.filter(pl.col("TVT_input").is_not_null())
    ev = df.filter(pl.col("TVT_input").is_null())
    if known.height < 5 or ev.height == 0:
        return None
    kx = known["X"].to_numpy().astype(float); ky = known["Y"].to_numpy().astype(float)
    kt = known["TVT_input"].to_numpy().astype(float); kz = known["Z"].to_numpy().astype(float)
    kmd = known["MD"].to_numpy().astype(float)
    kok = np.isfinite(kx) & np.isfinite(ky) & np.isfinite(kt) & np.isfinite(kz)
    if kok.sum() < 5:
        return None
    b = float(np.median(kt[kok] + kz[kok] - predict_zlayer(model, kx[kok], ky[kok], norm)))
    ex = ev["X"].to_numpy().astype(float); ey = ev["Y"].to_numpy().astype(float)
    ez = ev["Z"].to_numpy().astype(float); et = ev["TVT"].to_numpy().astype(float)
    emd = ev["MD"].to_numpy().astype(float)
    eok = np.isfinite(ex) & np.isfinite(ey) & np.isfinite(ez) & np.isfinite(et)
    if eok.sum() < 2:
        return None
    ex, ey, ez, et, emd = ex[eok], ey[eok], ez[eok], et[eok], emd[eok]
    o = np.argsort(emd); ex, ey, ez, et, emd = ex[o], ey[o], ez[o], et[o], emd[o]
    tvt_plain = predict_zlayer(model, ex, ey, norm) - ez + b
    last_tvt_input = float(kt[kok][np.argmax(kmd[kok])])
    tvt_anch = tvt_plain - (tvt_plain[0] - last_tvt_input)
    if return_sigma:
        return et, tvt_anch, emd, predict_sigma(model, ex, ey, norm)
    return et, tvt_anch, emd
