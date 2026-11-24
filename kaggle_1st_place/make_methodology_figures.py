#!/usr/bin/env python3
"""Methodology figures for the ROGII geosteering paper.

Run:   python make_methodology_figures.py              (PNGs go to the working directory)
       python make_methodology_figures.py --outdir figs --only 1 3

Needs only numpy + matplotlib.  No competition data, torch or GPU.

What this script is
-------------------
A toy well (one made-up well + one made-up reference "typewell") is pushed through
the SAME recipe, with the SAME constants, as the training pipeline:

    window            prefix_len=1024, target_len=10000, downsample=32  -> 345 columns
    depth axis        typewell_len=400, typewell_window=100 ft          -> 400 depths, 0.5 ft apart
    target map        alignment_target_mode="exp_smooth", sigma=1.25 ft
    target scaling    (value - 1.591691) / 15.853182
    GR penalty        ((typewell GR - column GR)/gr_diff_std)^2, clip 25
    simulator         0.85 replace prob, blocks 1024/2048/4096 (0.45/0.40/0.15), 16-row blend,
                      jumps >0.15 ft kept, z-shift +-40 ft, TVT_bias_v2 GR noise
                      (windows 1536/256/2, tvt std 0.95, GR std 5.6-17.5, offset std 2, clip 36)

The constants and formulas are taken from seq_NN_cfg.py / seq_NN_dataset.py (e.g.
_make_typewell_exp_smooth_target_probs, _make_gr_penalty_error, _make_typewell_rel_grid).
The real classes (SeqUNetDataset, Simulator) need the 1.3 GB training set, so they are
re-implemented here in a few lines each, for illustration only.  To plot a REAL well,
replace build_demo() with a function that fills the same dictionary keys from
SeqUNetDataset / make_unet_item(); every figure function only reads that dictionary.

Figures written
---------------
fig0_methodology_overview.png      whole pipeline on one page
fig1_synthesis_side_by_side.png    real training well  vs  synthetic well
fig2_well_to_fixed_window.png      well -> 11,024-row window -> 345 columns
fig3_target_probability_map.png    345 x 400 soft one-hot target
fig4_input_image_16_channels.png   the 16 input channels
fig5_gr_penalty_map.png            GR penalty map and what it does in the loss
"""

from __future__ import annotations

import argparse
import os
import textwrap
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# --------------------------------------------------------------------------- constants
PREFIX_LEN = 1024
TARGET_LEN = 10000
RAW_LEN = PREFIX_LEN + TARGET_LEN            # 11,024 rows
DOWNSAMPLE = 32
NUM_BINS = RAW_LEN // DOWNSAMPLE + 1         # 345 columns
N_PREFIX_COLS = PREFIX_LEN // DOWNSAMPLE     # 32 known columns
TW_WINDOW = 100.0
TW_LEN = 400
DEPTH = np.linspace(-TW_WINDOW + TW_WINDOW / TW_LEN, TW_WINDOW - TW_WINDOW / TW_LEN, TW_LEN)
TARGET_MEAN, TARGET_STD = 1.591691, 15.853182
SIGMA = 1.25
GR_SCALE = 22.0           # toy stand-in for cfg.gr_diff_std (= horizontal_stats["gr"][1])
GR_PENALTY_CLIP = 25.0

C_REAL, C_SYN, C_KNOWN, C_PRED, C_PAD = "#2b6cb0", "#dd6b20", "#4a90d9", "#38a169", "#b0b0b0"

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 100, "savefig.dpi": 160, "axes.titlesize": 11,
})


# --------------------------------------------------------------------------- toy data
def smooth_noise(rng, n, std, window):
    """Box-smoothed white noise rescaled to a given std (what the simulator uses)."""
    w = rng.standard_normal(n + window)
    s = np.convolve(w, np.ones(window) / window, mode="valid")[:n]
    s = s - s.mean()
    return s / (s.std() + 1e-12) * std


def make_typewell(rng, lo=0.0, hi=300.0, step=0.25):
    tvt = np.arange(lo, hi + step, step)
    n = tvt.size
    gr = 70 + 22 * smooth_noise(rng, n, 1.0, 40) + 8 * smooth_noise(rng, n, 1.0, 8)
    for c in rng.uniform(lo + 20, hi - 20, 14):                       # shale spikes
        gr += rng.uniform(12, 30) * np.exp(-0.5 * ((tvt - c) / rng.uniform(1.0, 3.0)) ** 2)
    return tvt, np.clip(gr, 15, 140)


def make_step_pool(rng, n_wells=12, n=12000):
    """Row-to-row TVT changes of 'real' wells: slow drift + small noise + a few faults."""
    pool = []
    for _ in range(n_wells):
        steps = smooth_noise(rng, n, 0.006, 1200) + rng.standard_normal(n) * 0.012
        for _ in range(rng.integers(0, 4)):
            steps[rng.integers(200, n - 200)] += rng.laplace(0, 1.75)
        pool.append(steps)
    return pool


def tvt_bias_v2_gr(rng, tvt, tw):
    """GR regenerated as typewell(TVT) + noise, following the TVT_bias_v2 recipe."""
    tw_tvt, tw_gr = tw
    at = lambda q: np.interp(q, tw_tvt, tw_gr, left=np.nan, right=np.nan)
    weights = np.array([0.68, 0.20, 0.12])
    noise = sum(np.sqrt(w) * smooth_noise(rng, tvt.size, 1.0, win)
                for w, win in zip(weights, (1536, 256, 2))) * 0.95   # tvt std 0.95 ft
    clean = at(tvt)
    diff = np.nan_to_num(at(tvt + noise) - clean)
    ranges = ((5.6, 8.4), (8.4, 12.2), (12.2, 17.5))
    target_std = rng.uniform(*ranges[rng.choice(3, p=[0.24, 0.51, 0.25])])
    diff = diff / (diff.std() + 1e-12) * target_std                  # rescale to 5.6-17.5 GR
    diff = np.clip(diff + rng.normal(0, 2.0), -36, 36)               # offset, clip +-36
    return clean + diff, clean


def stitch_path(rng, pool, orig_steps, n_out):
    """New row-to-row steps from 1024/2048/4096-row blocks of real wells, 16-row blends."""
    blend = 16
    out = np.zeros(0)
    bounds = []
    while out.size < n_out:
        b = int(rng.choice([1024, 2048, 4096], p=[0.45, 0.40, 0.15]))
        src = pool[rng.integers(len(pool))]
        s0 = rng.integers(0, src.size - b)
        blk = src[s0:s0 + b].copy()
        if out.size:
            w = np.linspace(0, 1, blend)
            out[-blend:] = out[-blend:] * (1 - w) + blk[:blend] * w
            blk = blk[blend:]
        bounds.append((out.size, out.size + blk.size))
        out = np.concatenate([out, blk])
    out = out[:n_out]
    keep = np.where(np.abs(orig_steps) > 0.15, orig_steps, 0.0)       # real jumps kept on top
    return out + keep, bounds


def build_demo(seed=3):
    rng = np.random.default_rng(seed)
    tw_full = make_typewell(rng)
    pool = make_step_pool(rng)

    # ---- one "real" training well
    n_rows, anchor, tvt0 = 9000, 1499, 150.0
    md = np.arange(n_rows, dtype=float)
    steps = smooth_noise(rng, n_rows, 0.006, 1200) + rng.standard_normal(n_rows) * 0.012
    steps[4200] += 2.1                                                # one fault
    tvt_real = tvt0 + np.cumsum(steps) - np.cumsum(steps)[anchor]
    surface = 0.004 * md + 6 * np.sin(md / 2500.0)                    # S = TVT + Z, near flat
    z_real = surface - tvt_real
    gr_real, _ = tvt_bias_v2_gr(rng, tvt_real, tw_full)
    gap = slice(3200, 3350)
    gr_real[gap] = np.nan

    # ---- its synthetic sibling: same rock (surface), new path, new GR
    n_suf = n_rows - anchor - 1
    new_steps, bounds = stitch_path(rng, pool, steps[anchor + 1:], n_suf)
    path = np.concatenate([[0.0], np.cumsum(new_steps)])
    tvt_syn = tvt_real.copy()
    tvt_syn[anchor:] = tvt0 + path
    delta = float(rng.uniform(-40, 40))                               # z_shift_range = 40 ft
    tvt_syn = tvt_syn + delta
    z_syn = surface - tvt_syn                                         # TVT + Z unchanged
    gr_syn, gr_syn_clean = tvt_bias_v2_gr(rng, tvt_syn, tw_full)
    shift = 2400                                                      # missing-GR pattern moves
    gr_syn[3200 + shift:3350 + shift] = np.nan

    # ---- window (rows) around the anchor, relative to the anchor
    ws = anchor - (PREFIX_LEN - 1)
    pad_n = NUM_BINS * DOWNSAMPLE

    def to_window(x):
        out = np.full(pad_n, np.nan)
        m = min(RAW_LEN, x.size - ws)
        out[:m] = x[ws:ws + m]
        return out

    tvt_rel = to_window(tvt_real) - tvt0
    z_rel = to_window(z_real) - z_real[anchor]
    s_rel = to_window(surface) - surface[anchor]
    gr_win = to_window(gr_real)

    def col_mean(x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(x.reshape(NUM_BINS, DOWNSAMPLE), axis=1)

    target_col = col_mean(tvt_rel)
    is_pred = np.arange(NUM_BINS) >= N_PREFIX_COLS
    target_mask = is_pred & np.isfinite(target_col)
    target_norm = (target_col - TARGET_MEAN) / TARGET_STD

    # ---- local typewell on the 400-depth axis (never extrapolated) -> some depths have no data
    tw_local = (tvt0 - 115.0, tvt0 + 85.0)
    tw_grid = np.interp(tvt0 + DEPTH, *tw_full, left=np.nan, right=np.nan)
    tw_grid[(tvt0 + DEPTH < tw_local[0]) | (tvt0 + DEPTH > tw_local[1])] = np.nan

    d = dict(
        rng=rng, tw_full=tw_full, md=md, anchor=anchor, tvt0=tvt0, ws=ws, surface=surface,
        tvt_real=tvt_real, z_real=z_real, gr_real=gr_real,
        tvt_syn=tvt_syn, z_syn=z_syn, gr_syn=gr_syn, gr_syn_clean=gr_syn_clean,
        blocks=bounds, delta=delta,
        tvt_rel=tvt_rel, z_rel=z_rel, s_rel=s_rel, gr_win=gr_win,
        target_col=target_col, target_mask=target_mask, target_norm=target_norm,
        is_pred=is_pred, tw_grid=tw_grid,
    )
    d["col"] = column_stats(gr_win, tvt_rel, z_rel, s_rel, d, rng)
    d["channels"], d["channel_names"] = build_channels(d)
    d["target_probs"] = target_probs(target_col, target_mask)
    d["penalty"] = gr_penalty(d)
    return d


# --------------------------------------------------------------------------- features
def column_stats(gr_win, tvt_rel, z_rel, s_rel, d, rng):
    g = gr_win.reshape(NUM_BINS, DOWNSAMPLE)
    x = np.linspace(-1, 1, DOWNSAMPLE)
    keys = ["mean", "nan_rate", "std", "first_last", "slope", "qa", "qb", "qc", "qrmse"]
    st = {k: np.zeros(NUM_BINS) for k in keys}
    for c in range(NUM_BINS):
        f = np.isfinite(g[c])
        st["nan_rate"][c] = 1.0 - f.mean()
        if f.sum() < 4:
            st["mean"][c] = np.nan
            continue
        y, xf = g[c][f], x[f]
        st["mean"][c] = y.mean()
        st["std"][c] = y.std()
        st["first_last"][c] = y[-1] - y[0]
        st["slope"][c] = np.polyfit(xf, y, 1)[0]
        a, b, cc = np.polyfit(xf, y, 2)
        st["qa"][c], st["qb"][c], st["qc"][c] = a, b, cc
        st["qrmse"][c] = np.sqrt(np.mean((np.polyval([a, b, cc], xf) - y) ** 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        tv = np.nanmean(tvt_rel.reshape(NUM_BINS, DOWNSAMPLE), axis=1)
        zc = np.nanmean(z_rel.reshape(NUM_BINS, DOWNSAMPLE), axis=1)
        sc = np.nanmean(s_rel.reshape(NUM_BINS, DOWNSAMPLE), axis=1)
    st["seen_tvt"] = np.where(~d["is_pred"], tv, 0.0)                 # prefix columns only
    # geo prior: surface change guessed from neighbour wells (truth + slowly growing error)
    err = np.cumsum(rng.standard_normal(NUM_BINS)) * 0.35
    geo_s = sc + err * np.clip((np.arange(NUM_BINS) - N_PREFIX_COLS) / 100.0, 0, 1)
    st["geo_tvt"] = np.where(d["is_pred"], geo_s - zc, 0.0)           # surface change - Z change
    return st


def build_channels(d):
    st, tw = d["col"], d["tw_grid"]
    tw_nan = np.isnan(tw)
    W, H = NUM_BINS, TW_LEN
    ones = np.ones((H, W))
    tw_n = np.nan_to_num((tw - np.nanmean(tw)) / np.nanstd(tw))
    gr_mean = st["mean"]
    gr_ok = np.isfinite(gr_mean) & (st["nan_rate"] < 1)
    gm = np.where(gr_ok, gr_mean, np.nan)
    ch = {}
    ch["tw_gr"] = tw_n[:, None] * ones
    ch["gr"] = np.nan_to_num((gm - 70) / 30.0)[None, :] * ones
    diff = np.abs(gm[None, :] - tw[:, None]) / GR_SCALE
    ch["gr_abs_diff"] = np.nan_to_num(diff, nan=0.0)
    for name, key, sc in [("gr_isnan_rate", "nan_rate", 1.0), ("gr_std", "std", 25.0),
                          ("gr_first_last_delta", "first_last", 25.0), ("gr_slope", "slope", 25.0),
                          ("gr_quadratic_a", "qa", 15.0), ("gr_quadratic_b", "qb", 25.0),
                          ("gr_quadratic_c", "qc", 30.0), ("gr_quadratic_rmse", "qrmse", 15.0)]:
        ch[name] = (np.nan_to_num(st[key]) / sc)[None, :] * ones
    ch["tw_gr_is_nan"] = tw_nan[:, None].astype(float) * ones
    ch["tw_tvt_rel"] = (DEPTH / TW_WINDOW)[:, None] * ones
    seen = st["seen_tvt"]
    ch["seen_tvt_rel"] = ((seen - TARGET_MEAN) / TARGET_STD)[None, :] * ones * (~d["is_pred"])[None, :]
    ch["tw_seen_tvt_abs_diff"] = np.abs(DEPTH[:, None] - seen[None, :]) / TW_WINDOW * (~d["is_pred"])[None, :]
    geo = np.nan_to_num((st["geo_tvt"] - TARGET_MEAN) / TARGET_STD)
    ch["geo_tvt_diff"] = geo[None, :] * ones * d["is_pred"][None, :]
    names = list(ch)
    return [ch[n] for n in names], names


def target_probs(target_col, mask):
    """Soft one-hot: Gaussian (sigma 1.25 ft) on the true depth, normalised per column."""
    probs = np.zeros((NUM_BINS, TW_LEN))
    t = np.clip(target_col[mask], DEPTH[0], DEPTH[-1])
    p = np.exp(-0.5 * ((DEPTH[None, :] - t[:, None]) / SIGMA) ** 2)
    probs[mask] = p / p.sum(axis=1, keepdims=True)
    return probs


def gr_penalty(d):
    """((typewell GR - column GR)/scale)^2, capped; no typewell -> cap; no GR -> 0."""
    st, tw = d["col"], d["tw_grid"]
    gr = st["mean"]
    row_valid = np.isfinite(gr) & (st["nan_rate"] < 1.0)
    pen = np.square((tw[None, :] - np.nan_to_num(gr)[:, None]) / GR_SCALE)
    pen = np.minimum(np.nan_to_num(pen, nan=GR_PENALTY_CLIP), GR_PENALTY_CLIP)
    pen[~row_valid, :] = 0.0
    pen[np.ix_(row_valid, np.isnan(tw))] = GR_PENALTY_CLIP
    return pen.astype(np.float32)


def save(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", path)


# --------------------------------------------------------------------------- figure 1
def fig1_synthesis(d, outdir):
    md, a = d["md"], d["anchor"]
    zmin = min(d["z_real"].min(), d["z_syn"].min()) - 15
    zmax = max(d["z_real"].max(), d["z_syn"].max()) + 15
    mdg = np.arange(0, md.size, 10)
    zg = np.linspace(zmin, zmax, 260)
    tvt_img = d["surface"][mdg][None, :] - zg[:, None]                # TVT = S - Z
    img = np.interp(tvt_img, *d["tw_full"], left=np.nan, right=np.nan)
    img = np.ma.masked_invalid(img)
    halo = [pe.withStroke(linewidth=3.2, foreground="white")]

    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.7, 1.0, 1.1, 0.95], hspace=0.42, wspace=0.12)
    titles = ["TRAINING SIDE: a real well from the data", "SYNTHESIS SIDE: a synthetic well made from it"]
    row_a = []
    for j, (key, color) in enumerate([("real", C_REAL), ("syn", C_SYN)]):
        tvt = d["tvt_real"] if key == "real" else d["tvt_syn"]
        z = d["z_real"] if key == "real" else d["z_syn"]
        gr = d["gr_real"] if key == "real" else d["gr_syn"]

        ax = fig.add_subplot(gs[0, j]); row_a.append(ax)
        im = ax.imshow(img, extent=[mdg[0], mdg[-1], zg[0], zg[-1]], origin="lower",
                       aspect="auto", cmap="YlOrBr", vmin=15, vmax=150)
        ax.plot(md, z, color=color, lw=2.4, path_effects=halo)
        ax.axvline(md[a], color="k", ls="--", lw=1)
        ax.axvspan(0, md[a], color="white", alpha=0.35)
        ax.text(md[a] / 2, zmax - 8, "known", ha="center", fontsize=9)
        ax.text((md[a] + md[-1]) / 2, zmax - 8, "unknown part (to predict)", ha="center", fontsize=9)
        ax.set_title(titles[j], color=color, fontweight="bold")
        ax.set_ylabel("Z, depth of the well (ft)")
        ax.set_xlabel("Measured depth along the well (rows)")
        if key == "syn":
            ax.text(0.99, 0.04, f"(2) same rock: TVT + Z unchanged\n     whole well moved {d['delta']:+.0f} ft",
                    transform=ax.transAxes, ha="right", fontsize=9, bbox=dict(fc="white", ec=color))
        else:
            ax.text(0.99, 0.04, "colours = rock GR the well drills through",
                    transform=ax.transAxes, ha="right", fontsize=9, bbox=dict(fc="white", ec=color))

        ax = fig.add_subplot(gs[1, j])
        if key == "syn":
            ax.plot(md, d["tvt_real"], color="0.7", lw=1.2, ls="--", label="original path")
            for b0, b1 in d["blocks"]:
                ax.axvline(md[a] + 1 + b0, color="0.5", lw=0.8, ls=":")
            ax.text(0.01, 0.06, "(1) new path = blocks of real wells' steps\n     (dotted lines = block joins)",
                    transform=ax.transAxes, fontsize=9, bbox=dict(fc="white", ec=color))
        else:
            ax.text(0.01, 0.06, "path actually drilled", transform=ax.transAxes, fontsize=9,
                    bbox=dict(fc="white", ec=color))
        ax.plot(md, tvt, color=color, lw=1.8, label="TVT (the label)")
        ax.axvline(md[a], color="k", ls="--", lw=1)
        ax.set_ylabel("TVT: depth in the rock (ft)")
        ax.set_xlabel("Measured depth along the well (rows)")
        ax.set_ylim(min(d["tvt_real"].min(), d["tvt_syn"].min()) - 8,
                    max(d["tvt_real"].max(), d["tvt_syn"].max()) + 8)
        if key == "syn":
            ax.legend(loc="upper right", fontsize=8, frameon=False)

        ax = fig.add_subplot(gs[2, j])
        if key == "syn":
            ax.plot(md, d["gr_syn_clean"], color="0.65", lw=0.9, label="typewell GR at the new TVT")
        ax.plot(md, gr, color=color, lw=0.9, label="well GR")
        ax.axvline(md[a], color="k", ls="--", lw=1)
        ax.set_ylabel("Gamma ray (GR)")
        ax.set_xlabel("Measured depth along the well (rows)")
        ax.set_ylim(-10, 170)
        if key == "syn":
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8, frameon=False, ncol=2)
            ax.text(0.01, 0.06, "(3) GR rebuilt = typewell GR + realistic noise;\n     missing-GR gap moved elsewhere",
                    transform=ax.transAxes, fontsize=9, bbox=dict(fc="white", ec=color))
        else:
            ax.text(0.01, 0.06, "recorded GR (gap = missing values)", transform=ax.transAxes, fontsize=9,
                    bbox=dict(fc="white", ec=color))
    fig.colorbar(im, ax=row_a, fraction=0.025, pad=0.01, label="GR of the rock")

    ax = fig.add_subplot(gs[3, :])
    ax.plot(*d["tw_full"], color="0.2", lw=1.0)
    ax.axvspan(d["tvt_real"].min(), d["tvt_real"].max(), color=C_REAL, alpha=0.25, label="depths read by the real well")
    ax.axvspan(d["tvt_syn"].min(), d["tvt_syn"].max(), color=C_SYN, alpha=0.25, label="depths read by the synthetic well")
    ax.set_xlabel("TVT (ft)"); ax.set_ylabel("Typewell GR"); ax.set_ylim(0, 200)
    ax.set_title("Both wells read the SAME typewell (reference GR log); only the depths they visit differ", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    fig.suptitle("How a synthetic training well is made (toy data, real recipe)", fontsize=14, fontweight="bold", y=0.94)
    save(fig, outdir, "fig1_synthesis_side_by_side.png")


# --------------------------------------------------------------------------- figure 2
def fig2_window(d, outdir):
    a, ws = d["anchor"], d["ws"]
    n_valid = d["tvt_real"].size - ws
    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1.1, 1, 1.1], hspace=0.95)

    ax = fig.add_subplot(gs[0])
    rows = np.arange(d["tvt_real"].size)
    ax.plot(rows, d["tvt_real"], color=C_REAL, lw=1.3)
    ax.axvspan(ws, ws + RAW_LEN, color="gold", alpha=0.25, label=f"window ({RAW_LEN:,} rows)")
    ax.axvline(a, color="k", ls="--"); ax.text(a, ax.get_ylim()[1], " anchor = last known row", va="top", fontsize=9)
    ax.set_title("(a) One well as stored: any length. The anchor is the last row whose TVT is known")
    ax.set_xlabel("Row in the original well"); ax.set_ylabel("TVT (ft)"); ax.legend(frameon=False, loc="lower right")

    ax = fig.add_subplot(gs[1])
    x = np.arange(NUM_BINS * DOWNSAMPLE)
    y = d["tvt_rel"]
    ax.plot(x, y, color=C_REAL, lw=1.3)
    ax.axvspan(0, PREFIX_LEN, color=C_KNOWN, alpha=0.18)
    ax.axvspan(PREFIX_LEN, n_valid, color=C_PRED, alpha=0.10)
    ax.axvspan(n_valid, RAW_LEN, color=C_PAD, alpha=0.30, hatch="//", ec="none")
    ax.axvline(PREFIX_LEN, color="k", ls="--")
    top = np.nanmax(y)
    ax.text(PREFIX_LEN / 2, top, "1,024 known\nrows", ha="center", va="top", fontsize=9)
    ax.text((PREFIX_LEN + n_valid) / 2, top, "unknown part (up to 10,000 rows)", ha="center", va="top", fontsize=9)
    ax.text((n_valid + RAW_LEN) / 2, top, "empty\n(NaN)", ha="center", va="top", fontsize=9)
    ax.set_xlim(0, RAW_LEN)
    ax.set_title("(b) Cut a fixed window of 11,024 rows around the anchor; everything is made relative to the anchor (TVT - TVT0)")
    ax.set_xlabel("Row in the window"); ax.set_ylabel("TVT - TVT0 (ft)")
    sec = ax.secondary_xaxis("top", functions=(lambda r: r / DOWNSAMPLE, lambda c: c * DOWNSAMPLE))
    sec.set_xlabel("Column (32 rows each)"); sec.spines["top"].set_visible(True)

    ax = fig.add_subplot(gs[2])
    c0, c1 = 150, 160
    r0, r1 = c0 * DOWNSAMPLE, c1 * DOWNSAMPLE
    ax.scatter(x[r0:r1], y[r0:r1], s=4, color=C_REAL, alpha=0.6, label="rows")
    means = d["target_col"][c0:c1]
    for i, m in enumerate(means):
        ax.hlines(m, r0 + i * DOWNSAMPLE, r0 + (i + 1) * DOWNSAMPLE, color="crimson", lw=2.6,
                  label="column target = mean of its 32 rows" if i == 0 else None)
    for r in range(r0, r1 + 1, DOWNSAMPLE):
        ax.axvline(r, color="0.7", lw=0.7)
    ax.set_title(f"(c) Zoom on columns {c0}-{c1 - 1}: 32 rows are grouped into one column; the target is their mean")
    ax.set_xlabel("Row in the window"); ax.set_ylabel("TVT - TVT0 (ft)"); ax.legend(frameon=False, loc="best")

    ax = fig.add_subplot(gs[3])
    cols = np.arange(NUM_BINS)
    state = np.where(~d["is_pred"], 0, np.where(d["target_mask"], 1, 2))
    ax.bar(cols, np.ones(NUM_BINS) * -3.2, bottom=-0.2, color=[[C_KNOWN, C_PRED, C_PAD][s] for s in state], width=1.0)
    yn = np.where(d["target_mask"], d["target_norm"], np.nan)
    ax.plot(cols, yn, color="crimson", lw=1.6)
    ax.set_xlim(0, NUM_BINS)
    ax.set_title("(d) Result: 345 columns. Columns 0-31 are known; 32-344 are predicted. Red = normalised target (value - 1.59) / 15.85")
    ax.set_xlabel("Column"); ax.set_ylabel("Normalised target")
    ax.text(16, -1.7, "known", ha="center", fontsize=9, color="white", fontweight="bold")
    ax.text((N_PREFIX_COLS + n_valid // DOWNSAMPLE) / 2 + 5, -1.7, "has target (counts in the loss)", ha="center", fontsize=9, color="white", fontweight="bold")
    ax.text(NUM_BINS - 12, -1.7, "no data", ha="center", fontsize=9)
    fig.suptitle("From a well to a fixed-size window", fontsize=14, fontweight="bold", y=0.93)
    save(fig, outdir, "fig2_well_to_fixed_window.png")


# --------------------------------------------------------------------------- figure 3
def fig3_target(d, outdir):
    P = d["target_probs"]
    fig = plt.figure(figsize=(15, 6.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], hspace=0.5, wspace=0.38)
    ax = fig.add_subplot(gs[:, 0])
    im = ax.imshow(P.T, extent=[0, NUM_BINS, DEPTH[0], DEPTH[-1]], origin="lower", aspect="auto",
                   cmap="magma", vmax=P.max())
    ax.plot(np.where(d["target_mask"])[0] + 0.5, d["target_col"][d["target_mask"]], color="cyan", lw=0.6, ls="--", alpha=0.7)
    ax.axvline(N_PREFIX_COLS, color="w", ls="--", lw=1)
    ax.text(N_PREFIX_COLS / 2, 85, "known\n(no target)", color="w", ha="center", va="top", fontsize=9)
    c = 200
    ax.axvline(c + 0.5, color="lime", lw=1.2)
    ax.set_xlabel("Column along the well (345)"); ax.set_ylabel("Candidate depth relative to the anchor, TVT - TVT0 (ft)")
    ax.set_title("Target probability map: 345 columns x 400 depths. Bright = true depth; each column sums to 1")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="probability")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(DEPTH, P[c], color="crimson", lw=1.6)
    ax.axvline(d["target_col"][c], color="k", ls=":", lw=1)
    ax.set_title(f"One column (green line): Gaussian, sigma = {SIGMA} ft, sums to {P[c].sum():.2f}", fontsize=10)
    ax.set_xlabel("Candidate depth (ft)"); ax.set_ylabel("probability")

    ax = fig.add_subplot(gs[1, 1])
    t = d["target_col"][c]
    m = np.abs(DEPTH - t) < 8
    ax.bar(DEPTH[m], P[c][m], width=0.5, color="crimson", alpha=0.8)
    ax.set_title("Zoom: the 0.5 ft depth bins around the truth", fontsize=10)
    ax.set_xlabel("Candidate depth (ft)"); ax.set_ylabel("probability")
    fig.suptitle("Step 9: the 'soft one-hot' answer the network is taught to output", fontsize=14, fontweight="bold", y=1.0)
    save(fig, outdir, "fig3_target_probability_map.png")


# --------------------------------------------------------------------------- figure 4
def fig4_channels(d, outdir):
    ch, names = d["channels"], d["channel_names"]
    notes = {
        "tw_gr": "typewell GR at each depth", "gr": "well GR in each column",
        "gr_abs_diff": "|well GR - typewell GR|: matching cost", "gr_isnan_rate": "share of missing GR",
        "gr_std": "GR spread in the column", "gr_first_last_delta": "GR last - first", "gr_slope": "GR slope",
        "gr_quadratic_a": "GR curve fit: a", "gr_quadratic_b": "GR curve fit: b", "gr_quadratic_c": "GR curve fit: c",
        "gr_quadratic_rmse": "GR curve fit: error", "tw_gr_is_nan": "1 = typewell has no data here",
        "tw_tvt_rel": "the candidate depth itself", "seen_tvt_rel": "known path (known part only)",
        "tw_seen_tvt_abs_diff": "distance from known path", "geo_tvt_diff": "neighbour-well guess of path change",
    }
    cmaps = {"gr_abs_diff": "viridis_r", "tw_gr_is_nan": "gray_r", "geo_tvt_diff": "coolwarm"}
    fig, axes = plt.subplots(4, 4, figsize=(17, 13))
    for i, (ax, name, arr) in enumerate(zip(axes.ravel(), names, ch)):
        kw = {}
        if name == "geo_tvt_diff":
            lim = max(np.nanmax(np.abs(arr)), 1e-6); kw = dict(vmin=-lim, vmax=lim)
        ax.imshow(arr, extent=[0, NUM_BINS, DEPTH[0], DEPTH[-1]], origin="lower", aspect="auto",
                  cmap=cmaps.get(name, "viridis"), **kw)
        ax.axvline(N_PREFIX_COLS, color="w", ls="--", lw=0.8)
        ax.set_title(f"{i + 1}. {name}", fontsize=10, fontweight="bold", pad=17)
        ax.text(0.5, 1.015, notes[name], transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color="0.25")
        ax.tick_params(labelsize=8)
        if i % 4: ax.set_yticklabels([])
        if i < 12: ax.set_xticklabels([])
    for ax in axes[:, 0]: ax.set_ylabel("depth rel. anchor (ft)")
    for ax in axes[-1]: ax.set_xlabel("column")
    fig.suptitle("The 16-channel input image: every channel is 345 columns x 400 depths "
                 "(dashed line = anchor; colour scale is per channel)", fontsize=14, fontweight="bold", y=0.97)
    fig.subplots_adjust(hspace=0.42, wspace=0.12, top=0.90)
    save(fig, outdir, "fig4_input_image_16_channels.png")


# --------------------------------------------------------------------------- figure 5
def fig5_penalty(d, outdir):
    pen = d["penalty"]
    P = d["target_probs"]
    c = 200
    t = d["target_col"][c]
    fig = plt.figure(figsize=(15, 6.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.1, 1], hspace=0.55, wspace=0.38)

    ax = fig.add_subplot(gs[:, 0])
    im = ax.imshow(pen.T, extent=[0, NUM_BINS, DEPTH[0], DEPTH[-1]], origin="lower", aspect="auto",
                   cmap="magma_r", vmin=0, vmax=GR_PENALTY_CLIP)
    ax.plot(np.where(d["target_mask"])[0] + 0.5, d["target_col"][d["target_mask"]], color="cyan", lw=0.8, ls="--",
            label="true depth (not an input)")
    ax.axvline(N_PREFIX_COLS, color="k", ls="--", lw=1)
    ax.axvline(c + 0.5, color="lime", lw=1.2)
    nanrows = np.isnan(d["tw_grid"])
    top = DEPTH[nanrows].min()
    ax.annotate("typewell has no data here\n-> penalty = cap (25)", xy=(NUM_BINS * 0.55, top + 3),
                xytext=(NUM_BINS * 0.30, 40), arrowprops=dict(arrowstyle="->"), fontsize=9,
                bbox=dict(fc="white", ec="0.5"))
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.set_xlabel("Column along the well (345)"); ax.set_ylabel("Candidate depth relative to the anchor (ft)")
    ax.set_title("GR penalty map: ((typewell GR - column GR) / scale)^2, capped at 25. Dark = depth fits the GR")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="penalty")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(DEPTH, pen[c], color="0.2", lw=1.2, label="penalty")
    ax.axvline(t, color="crimson", ls=":", label="true depth")
    ax.set_ylim(0, GR_PENALTY_CLIP + 2)
    ax.set_title("One column (green line)", fontsize=10)
    ax.set_xlabel("Candidate depth (ft)"); ax.set_ylabel("penalty"); ax.legend(frameon=False, fontsize=8)
    ax.set_xlim(-100, 100)

    ax = fig.add_subplot(gs[1, 1])
    sharp = np.exp(-0.5 * ((DEPTH - t) / 2.0) ** 2)
    off = np.exp(-0.5 * ((DEPTH - (t + 30)) / 2.0) ** 2)
    flat = np.ones_like(DEPTH)
    cases = [("sharp, at the\ntrue depth", sharp), ("sharp, 30 ft\naway", off), ("spread over\nall depths", flat)]
    vals = [float((p / p.sum() * pen[c]).sum()) for _, p in cases]
    ax.bar(range(3), vals, color=["#38a169", "#dd6b20", "#a0a0a0"])
    ax.set_xticks(range(3)); ax.set_xticklabels([n for n, _ in cases], fontsize=8.5)
    ax.set_ylabel("expected penalty")
    ax.set_title("Loss term = sum(probability x penalty) for three guesses", fontsize=10)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("The GR penalty map: it punishes probability placed on depths whose typewell GR disagrees with the well",
                 fontsize=13, fontweight="bold", y=1.0)
    save(fig, outdir, "fig5_gr_penalty_map.png")


# --------------------------------------------------------------------------- figure 0
def box(ax, x, y, w, h, num, title, body, color, wrap=46):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=color, ec="#333", lw=1.2))
    ax.text(x + 1.2, y + h - 1.0, str(num), fontsize=10, fontweight="bold", va="top", ha="left",
            bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec="#333"))
    ax.text(x + w / 2 + 1, y + h - 1.2, title, ha="center", va="top", fontsize=10.5, fontweight="bold")
    if body:
        ax.text(x + w / 2, y + h - 4.6, textwrap.fill(body, wrap), ha="center", va="top", fontsize=8.6)


def arrow(ax, p, q, ls="-", label=None):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, lw=1.8, color="#333", ls=ls))
    if label:
        ax.text((p[0] + q[0]) / 2 + (2.5 if p[0] == q[0] else 0), (p[1] + q[1]) / 2 + (0 if p[0] == q[0] else 1.1), label, fontsize=8, ha="left" if p[0] == q[0] else "center", color="#333", style="italic")


def fig0_overview(d, outdir):
    fig = plt.figure(figsize=(18, 15))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 94); ax.axis("off")
    w, h = 29, 10
    X = [2, 35.5, 69]
    cA, cB, cC, cD, cE = "#e6f0fa", "#fdebd8", "#e4f5e6", "#f1e6fa", "#fbe6e9"
    cap = dict(fontsize=10, style="italic", color="0.35")

    ax.text(50, 92.5, "The whole method on one page", ha="center", fontsize=17, fontweight="bold")
    ax.text(6, 86.2, "A.  Prepare the data (once)", **cap)
    box(ax, X[0], 75, w, h, 1, "Data", "773 wells: drilled path (X, Y, Z), gamma-ray (GR) log and a nearby reference well (typewell). 618 train, 155 holdout.", cA)
    box(ax, X[1], 75, w, h, 2, "Geo prior", "Neighbouring wells give a first guess of how the rock tilts along the unknown part (11.6 ft error alone).", cA)
    box(ax, X[2], 75, w, h, 3, "Calibrate the typewell", "Blend the typewell GR with the GR already seen in this well, so the two logs match.", cA)
    arrow(ax, (X[0] + w + 0.5, 80), (X[1] - 0.5, 80)); arrow(ax, (X[1] + w + 0.5, 80), (X[2] - 0.5, 80))
    arrow(ax, (X[2] + w / 2, 74.5), (X[2] + w / 2, 69.7))

    ax.text(6, 70.2, "B.  Make one training sample (every time a well is drawn)", **cap)
    box(ax, X[2], 59, w, h, 4, "Simulate a new well (85%)", "Same rock, new drilling path built from real path pieces; GR rebuilt from the typewell plus noise. Labels stay exact.", cB)
    box(ax, X[1], 59, w, h, 5, "Augment", "Random changes: reverse the path, shift/scale the typewell, stretch, cut the tail.", cB)
    box(ax, X[0], 59, w, h, 6, "Cut a fixed window", "1,024 known + up to 10,000 unknown rows = 11,024 rows, grouped into 345 columns of 32 rows.", cB)
    arrow(ax, (X[2] - 0.5, 64), (X[1] + w + 0.5, 64)); arrow(ax, (X[1] - 0.5, 64), (X[0] + w + 0.5, 64))
    arrow(ax, (X[0] + w / 2, 58.5), (X[0] + w / 2, 53.7))

    ax.text(6, 54.2, "C.  Three pictures built from each window (each 345 columns x 400 depths)", **cap)
    thumbs = [(X[0], 7, "16-channel input image", "what the network sees", d["channels"][2], "viridis_r"),
              (X[1], 8, "Target probability map", "the answer it should output", d["target_probs"].T, "magma"),
              (X[2], 9, "GR penalty map", "used inside the loss", d["penalty"].T, "magma_r")]
    for x, n, t, sub, arr, cm in thumbs:
        ax.add_patch(FancyBboxPatch((x, 34), w, 19, boxstyle="round,pad=0.3,rounding_size=1.2", fc=cC, ec="#333", lw=1.2))
        ax.text(x + 1.2, 52, str(n), fontsize=10, fontweight="bold", va="top", bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec="#333"))
        ax.text(x + w / 2 + 1, 52, t, ha="center", va="top", fontsize=10.5, fontweight="bold")
        ax.text(x + w / 2, 48.8, sub, ha="center", va="top", fontsize=8.6)
        ins = ax.inset_axes([(x + 2) / 100, 35.2 / 94, (w - 4) / 100, 11.6 / 94])
        ins.imshow(arr, origin="lower", aspect="auto", cmap=cm, extent=[0, NUM_BINS, -100, 100])
        ins.set_xticks([]); ins.set_yticks([])
    arrow(ax, (X[0] + w / 2, 33.5), (X[0] + w / 2, 28.7), label="input")

    ax.text(6, 29.5, "D.  Learn", **cap)
    box(ax, X[0], 18, w, h, 10, "U-Net (ConvNeXt-small)", "Reads the image and scores every depth at every column.", cD)
    box(ax, X[1], 18, w, h, 11, "Softmax + soft-argmin", "Scores become probabilities over 400 depths; prediction = probability-weighted depth.", cD)
    box(ax, X[2], 18, w, h, 12, "Loss: four parts", "Alignment cross-entropy, regression, GR penalty, and the new multi-scale slope loss.", cD)
    arrow(ax, (X[0] + w + 0.5, 23), (X[1] - 0.5, 23)); arrow(ax, (X[1] + w + 0.5, 23), (X[2] - 0.5, 23))
    arrow(ax, (X[1] + w / 2 + 6, 33.5), (X[2] + 4, 28.5), ls="--", label="answer")
    arrow(ax, (X[2] + w / 2, 33.5), (X[2] + w / 2, 28.5), ls="--")
    arrow(ax, (X[2] + w / 2 + 4, 17.5), (X[2] + w / 2 + 4, 12.7))

    ax.text(6, 13.4, "E.  Train, check and score", **cap)
    box(ax, X[2], 2, w, h, 13, "Optimise", "150 epochs, AdamW, cosine learning rate, 16 samples per step.", cE)
    box(ax, X[1], 2, w, h, 14, "Check on holdout", "155 real wells, never simulated. RMSE every 5 epochs.", cE)
    box(ax, X[0], 2, w, h, 15, "Smooth and score", "Smooth the predicted rock surface. Report last-checkpoint and epoch 100-150 average.", cE)
    arrow(ax, (X[2] - 0.5, 7), (X[1] + w + 0.5, 7)); arrow(ax, (X[1] - 0.5, 7), (X[0] + w + 0.5, 7))
    save(fig, outdir, "fig0_methodology_overview.png")


FIGS = {0: fig0_overview, 1: fig1_synthesis, 2: fig2_window, 3: fig3_target, 4: fig4_channels, 5: fig5_penalty}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=".", help="where to write PNGs (default: working directory)")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--only", type=int, nargs="*", choices=sorted(FIGS), help="figure numbers to draw")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    d = build_demo(args.seed)
    for k in (args.only if args.only else sorted(FIGS)):
        FIGS[k](d, args.outdir)


if __name__ == "__main__":
    main()
