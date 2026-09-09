#!/usr/bin/env python3
"""Inference for the anchor CNN: sample construction, model loading and path decoding.

Used by the submission notebook. The pipeline per well is:

    build_sample()   raw well arrays -> the model's 2D input grid, anchored at the last
                     known TVT_input before the evaluation zone
    load_anchor_models()  reads checkpoints; the head configuration is inferred from the
                     state dict, so a checkpoint trained with the auxiliary dz_layer head
                     and one without it both load correctly
    predict_wells()  runs the models and decodes a TVT path from the predicted move
                     distributions, then maps grid columns back to MD

Grid geometry is taken from module-level constants (PS_COL, ANCHOR_KW) and from
gr2tvt_data. The submission notebook overrides them per model family, because the two
families in the final ensemble use different MD column widths (32 ft and 16 ft).

This module also contains an older deployment path built on stochastic rollouts scored
against a reference surface (RefSurface, load_ref_models, rollout selection constants).
The winning submission does not use it: it decodes the expected path by dynamic
programming instead.

Determinism note: the rollout generator is created once per model and reused across the
whole well loop, so predictions depend on the order wells are processed in. Callers fix
the order with sorted().
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import uniform_filter1d

import gr2tvt_data as gd
from gr2tvt_model import GR2TVTAnchorNet, items_to_x, rollout_paths
from zlayer_mlp import DEFAULTS, FourierMLP, predict_zlayer

PS_COL = 16
N_SAMPLE = 64
TEMP = 1.6
GATE = 0.125
ROLLOUT_SEED = 0
DIP_ALPHA = 1.0
DIP_STRIDE = 32
ANCHOR_KW = dict(stem_stride=1, fuse_div=4, ps_col=PS_COL, in_chans=6, anchor_m=gd.H + PS_COL)

PRE_PS_GR = False
Z_DIP = False
REF_SEEDS = (42, 43, 44)
SIGMA = 1.0


def gen_seed(model_idx: int, rollout_seed: int = ROLLOUT_SEED) -> int:
    return 1000 * model_idx + rollout_seed


class ResidualZ(nn.Module):

    def __init__(self, hard, resid, zs_hard, rm, rs):
        super().__init__()
        self.hard = hard
        self.resid = resid
        self.register_buffer("zs_hard", torch.tensor(float(zs_hard)))
        self.register_buffer("rm", torch.tensor(float(rm)))
        self.register_buffer("rs", torch.tensor(float(rs)))

    def forward(self, x):
        zh = self.hard(x)
        zh = zh[:, 0] if zh.dim() > 1 else zh
        rn = self.resid(x)
        rn = rn[:, 0] if rn.dim() > 1 else rn
        return zh + (rn * self.rs + self.rm) / self.zs_hard


def _load_mlp(path: Path, device) -> FourierMLP:
    m = FourierMLP(n_freqs=DEFAULTS["n_freqs"], hidden=DEFAULTS["hidden"],
                   n_layers=DEFAULTS["n_layers"], sigma=SIGMA).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


def load_ref_models(root: Path, folds=(0, 1, 2, 3, 4), seeds=REF_SEEDS, device="cpu"):
    root = Path(root)
    keep_dir = root / "model/plan069/keep_lam1e-06_sig1_ep100"
    excl_dir = root / "model/plan069/exclude_lam1e-06_sig1_ep100"
    resid_dirs = {0: root / "model/plan072/p1a_resid_pure_exclude_lam1e-07_sig1_ep100"}
    for k in (1, 2, 3, 4):
        resid_dirs[k] = root / "model/plan074/p1a_resid_pure_exclude_lam1e-07_sig1_ep100"

    hard_keep, resid = [], []
    for k in folds:
        norm_keep = tuple(json.loads((keep_dir / f"fold{k}_norm.json").read_text()))
        norm_excl = tuple(json.loads((excl_dir / f"fold{k}_norm.json").read_text()))
        for s in seeds:
            hard_keep.append((_load_mlp(keep_dir / f"seed{s}_fold{k}.pt", device), norm_keep))
            hard = _load_mlp(excl_dir / f"seed{s}_fold{k}.pt", device)
            rd = resid_dirs[k]
            rm, rs = json.loads((rd / f"seed{s}_fold{k}_rnorm.json").read_text())
            comp = ResidualZ(hard, _load_mlp(rd / f"seed{s}_fold{k}.pt", device),
                             norm_excl[5], rm, rs).to(device)
            comp.eval()
            resid.append((comp, norm_excl))
    return hard_keep, resid


def _recon(model, norm, kx, ky, kz, kt, kmd, ex, ey, ez) -> np.ndarray:
    zl_k = predict_zlayer(model, kx, ky, norm).astype(np.float64)
    zl_e = predict_zlayer(model, ex, ey, norm).astype(np.float64)
    b = float(np.median(kt + kz - zl_k))
    tvt_plain = zl_e - ez + b
    last_tvt_input = float(kt[np.argmax(kmd)])
    return tvt_plain - (tvt_plain[0] - last_tvt_input)


class RefSurface:

    def __init__(self, hard_keep, resid):
        self.systems = (hard_keep, resid)

    def predict(self, x, y, z, tvt_input, md, eval_mask) -> np.ndarray | None:
        for a in (x, y, z, tvt_input, md):
            assert a.dtype == np.float64, f"raw coordinates and TVT must be passed as float64 (got {a.dtype}）"
        known = np.isfinite(tvt_input) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if known.sum() < 5 or eval_mask.sum() < 2:
            return None
        kx, ky, kz, kt, kmd = x[known], y[known], z[known], tvt_input[known], md[known]
        o = np.argsort(md[eval_mask])
        emd = md[eval_mask][o]
        ex, ey, ez = x[eval_mask][o], y[eval_mask][o], z[eval_mask][o]
        for a in (ex, ey, ez):
            bad = ~np.isfinite(a)
            if bad.any():
                if bad.all():
                    return None
                a[bad] = np.interp(emd[bad], emd[~bad], a[~bad])
        sysp = [np.mean([_recon(m, n, kx, ky, kz, kt, kmd, ex, ey, ez) for m, n in models], axis=0)
                for models in self.systems]
        out = np.empty(int(eval_mask.sum()))
        out[o] = 0.5 * (sysp[0] + sysp[1])
        return out


def dzl_kwargs_from_sd(sd: dict) -> dict:
    if not any(k.startswith("dzl_") for k in sd):
        return {}
    arch = "mdconv" if "dzl_proj.0.weight" in sd else "linear"
    if "dzl_film.weight" not in sd:
        film = "off"
    else:
        film = "full" if "dzl_film.bias" in sd else "cond"
    return dict(dzl_head=True, dzl_arch=arch, dzl_film=film)


def load_anchor_models(ckpts, device):
    models = []
    for p in ckpts:
        sd = torch.load(p, map_location=device)
        sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
        kw = dict(ANCHOR_KW, in_chans=8, ps_gr=True) if PRE_PS_GR else dict(ANCHOR_KW)
        if Z_DIP:
            kw = dict(kw, in_chans=kw["in_chans"] + 1, z_dip=True)
        kw.update(dzl_kwargs_from_sd(sd))
        m = GR2TVTAnchorNet(pretrained=False, **kw).to(device)
        miss, unexp = m.load_state_dict(sd, strict=False)
        bad = [k for k in list(miss) + list(unexp) if k.startswith("dzl_")]
        if bad:
            raise RuntimeError(f"dz_layer head weights disagree across checkpoints: {bad[:4]}（ckpt={p}）")
        m.eval()
        models.append(m)
    return models


def prep_static(meta, md_query: np.ndarray, ref_vals: np.ndarray) -> dict:
    xc = gd.col_centers(meta)
    i0, i1, w0, w1 = gd.interp_weights(xc, md_query)
    return dict(i0=i0, i1=i1, w0=w0, w1=w1, nc=meta["ncol"], tvt_ps=meta["tvt_ps"],
                rv_c=ref_vals - ref_vals.mean(), npt=len(md_query))


def build_sample(md, gr, tvt_input, tw_tvt, tw_gr, ref_vals, eval_mask, z=None, md_phase=0.0):
    if tw_tvt is None or len(tw_tvt) < 3:
        return None
    if Z_DIP and z is None:
        raise ValueError("Z_DIP=True requires z (well trajectory TVD)")
    w = dict(md=md, gr=gr, tvt_input=tvt_input, tw_tvt=tw_tvt, tw_gr=tw_gr)
    if z is not None:
        w["z"] = z
    try:
        c = gd.build_compact(w, with_label=False, pre_ps_cols=PS_COL, pre_ps_gr=PRE_PS_GR,
                             z_dip=Z_DIP, md_phase=md_phase)
    except ValueError:
        return None
    item = (c["t_n"], c["h_n"], c["h_valid"], c["t_cover"], c["y"].astype(np.float32),
            c["meta"]["ncol_tot"], c["y_bnd"], c["h_cnt"], c["h_rows"], c["k_lv"], c["k_flag"],
            c.get("s_n"), c.get("s_cover"), c.get("d_n"))
    return dict(item=item, meta=c["meta"],
                static=prep_static(c["meta"], md[eval_mask], ref_vals))


def predict_wells(names, samples, models, device, batch=8, n_sample=N_SAMPLE,
                  temp=TEMP, gate=GATE, rollout_seed=ROLLOUT_SEED, start_col=PS_COL,
                  dip_alpha=0.0, smooth_w=0, decim_st=0, ens_shape=False):
    k = max(1, int(round(gate * n_sample)))
    acc = {nm: np.zeros(samples[nm]["static"]["npt"]) for nm in names}
    for si, m in enumerate(models):
        gen = torch.Generator(device=device).manual_seed(gen_seed(si, rollout_seed))
        for i in range(0, len(names), batch):
            chunk = names[i:i + batch]
            x, _ = items_to_x([samples[c]["item"] for c in chunk], device, m)
            with torch.no_grad():
                cl, b = m(x)
            for j, nm in enumerate(chunk):
                s = samples[nm]["static"]
                paths = rollout_paths(cl[j], b[j], s["nc"], n_sample, temp, gen,
                                      start_col=start_col).cpu().numpy()
                col = 0.5 * (paths[:, :-1] + paths[:, 1:])
                mds = s["tvt_ps"] + col[:, s["i0"]] * s["w0"] + col[:, s["i1"]] * s["w1"]
                assert mds.dtype == np.float64
                d_shape = np.sqrt(((mds - mds.mean(1, keepdims=True) - s["rv_c"]) ** 2).mean(1))
                if smooth_w:
                    ps = uniform_filter1d(mds, size=smooth_w, axis=-1, mode="nearest")
                    rs = uniform_filter1d(s["rv_c"], size=smooth_w, mode="nearest")
                    st = decim_st or smooth_w
                    n_pts = ps.shape[-1]
                    if n_pts <= st:
                        pad = 2 * st - n_pts
                        ps = np.pad(ps, ((0, 0), (0, pad)), mode="edge")
                        rs = np.pad(rs, (0, pad), mode="edge")
                    dq = np.diff(ps[:, ::st], axis=1) - np.diff(rs[::st])
                    d_sl = np.sqrt((dq ** 2).mean(1))
                    if ens_shape:
                        acc[nm] += 0.5 * (mds[np.argsort(d_shape)[:k]].mean(0)
                                          + mds[np.argsort(d_sl)[:k]].mean(0)) / len(models)
                        continue
                    if dip_alpha:
                        zs = (d_shape - d_shape.mean()) / (d_shape.std() + 1e-12)
                        zq = (d_sl - d_sl.mean()) / (d_sl.std() + 1e-12)
                        dv = zs + dip_alpha * zq
                    else:
                        dv = d_sl
                elif dip_alpha:
                    dp = np.diff(mds[:, ::DIP_STRIDE], axis=1)
                    dr = np.diff(s["rv_c"][::DIP_STRIDE])
                    d_slope = np.sqrt(((dp - dr) ** 2).mean(1))
                    zs = (d_shape - d_shape.mean()) / (d_shape.std() + 1e-12)
                    zsl = (d_slope - d_slope.mean()) / (d_slope.std() + 1e-12)
                    dv = zs + dip_alpha * zsl
                else:
                    dv = d_shape
                acc[nm] += mds[np.argsort(dv)[:k]].mean(0) / len(models)
            del cl, b, x
    return acc
