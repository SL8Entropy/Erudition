"""A particle filter over the structural state, used as an input channel.

State per particle:

    L   relative TVT (level)
    r   structural slope dS/dMD, S = TVT + Z   -- so dTVT = r*dMD - dZ
    b   slowly drifting GR bias between this well and the reference

The four things 1st place calls out as the improvements over the public
baseline are all here: low-probability large jumps (fault recovery), a
calibrated typewell, backward smoothing through the particle genealogy, and
updates in coarse MD bins rather than at raw resolution. Several configuration
profiles are blended, which is the fifth.

Its own accuracy (~7.4 RMSE standalone in 1st place's hands) is not the point.
The point is a *distribution* -- the 2D heatmap -- handed to the CNN as a
channel, plus a diverse candidate path for the ensemble.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .canvas import calibrate_typewell
from .data import GR_MEAN, GR_STD, initial_structural_rate


@dataclass
class PFConfig:
    n_particles: int = 512
    bin_ft: float = 64.0        # update in bins, not at raw resolution
    rate_std: float = 8e-5      # random walk on the structural slope, per sqrt(ft)
    level_std: float = 0.15     # per-bin level jitter, ft
    jump_prob: float = 0.004    # rare large jump -> fault recovery
    jump_std: float = 12.0
    bias_std: float = 0.01      # GR bias random walk (normalised units)
    obs_scale: float = 0.45     # Student-t scale on normalised GR residual
    obs_df: float = 2.0         # heavy tails: robust to bad GR samples
    obs_temp: float = 1.0       # weight of the likelihood per bin
    ess_frac: float = 0.5
    seed: int = 0


PROFILES = (
    PFConfig(),
    PFConfig(rate_std=1.6e-4, jump_prob=0.010, obs_scale=0.60, obs_temp=0.7, seed=1),
    PFConfig(rate_std=4e-5, jump_prob=0.001, obs_scale=0.33, obs_temp=1.4, seed=2),
)


def _initial_rate(well, n_rows: int = 256):
    """Estimate the current structural slope from the tail of the known prefix.

    3rd place measured that widening this window from 30 to 256 rows was worth
    real private-LB points: the long window is far less sensitive to noise just
    before PS, and every downstream foot inherits this number.
    """
    vis = np.flatnonzero(np.isfinite(well.tvt_input))
    if len(vis) < 8:
        return 0.0, 3e-4
    idx = vis[-min(n_rows, len(vis)):]
    s = well.tvt_input[idx] + well.z[idx]
    md = well.md[idx]
    A = np.column_stack([md - md.mean(), np.ones(len(md))])
    coef = np.array([initial_structural_rate(well, n_rows),
                     float(s.mean())])
    resid = s - A @ coef
    dof = max(len(md) - 2, 1)
    var = float(resid @ resid) / dof
    denom = float(((md - md.mean()) ** 2).sum()) + 1e-9
    return float(coef[0]), float(np.sqrt(var / denom) + 1e-4)


def _systematic_resample(w, rng):
    n = len(w)
    pos = (rng.random() + np.arange(n)) / n
    return np.searchsorted(np.cumsum(w), pos).clip(0, n - 1)


def run_pf(well, cfg: PFConfig, tw_gr=None, md_grid=None):
    """Forward filter + genealogy smoothing. Returns per-bin level samples."""
    rng = np.random.default_rng(cfg.seed + abs(hash(well.well_id)) % 100000)
    tw_gr = calibrate_typewell(well) if tw_gr is None else tw_gr
    tw_gr_n = (tw_gr - GR_MEAN) / GR_STD
    a_tvt, a_md = well.anchor_tvt, well.anchor_md

    md_end = float(well.md[-1])
    if md_grid is None:
        n_bins = max(int(np.ceil((md_end - a_md) / cfg.bin_ft)), 1)
        md_grid = a_md + np.arange(n_bins + 1) * cfg.bin_ft
    n_bins = len(md_grid) - 1

    N = cfg.n_particles
    r0, r_sd = _initial_rate(well)
    L = np.zeros(N)
    r = r0 + rng.normal(0.0, max(r_sd, 5e-5) * 3.0, N)
    b = rng.normal(0.0, 0.05, N)

    z_at = np.interp(md_grid, well.md, well.z)
    gr_n = (well.gr - GR_MEAN) / GR_STD

    levels = np.zeros((n_bins, N), dtype=np.float32)
    ancestors = np.zeros((n_bins, N), dtype=np.int32)
    logw = np.zeros(N)

    for k in range(n_bins):
        r = r + rng.normal(0.0, cfg.rate_std * np.sqrt(cfg.bin_ft), N)
        jump = rng.random(N) < cfg.jump_prob
        L = L + np.where(jump, rng.normal(0.0, cfg.jump_std, N), 0.0)
        dz = z_at[k + 1] - z_at[k]
        L = L + r * cfg.bin_ft - dz + rng.normal(0.0, cfg.level_std, N)
        b = b + rng.normal(0.0, cfg.bias_std, N)

        lo = np.searchsorted(well.md, md_grid[k], "left")
        hi = np.searchsorted(well.md, md_grid[k + 1], "left")
        sl = slice(lo, max(hi, lo + 1))
        obs = gr_n[sl][::4]
        obs = obs[np.isfinite(obs)]
        if len(obs):
            md_s = well.md[sl][::4][:len(obs)]
            frac = (md_s - md_grid[k]) / max(md_grid[k + 1] - md_grid[k], 1e-9)
            # each particle reads the reference at its own hypothesised level
            lvl = L[:, None] + (r[:, None] * cfg.bin_ft) * (frac[None, :] - 1.0)
            ref = np.interp(a_tvt + lvl.ravel(), well.tw_tvt, tw_gr_n,
                            left=np.nan, right=np.nan).reshape(lvl.shape)
            res = (obs[None, :] - (ref + b[:, None])) / cfg.obs_scale
            ll = -0.5 * (cfg.obs_df + 1.0) * np.log1p(res ** 2 / cfg.obs_df)
            ll = np.where(np.isfinite(ll), ll, -3.0)          # off-typewell penalty
            logw = logw + cfg.obs_temp * ll.mean(axis=1)

        w = np.exp(logw - logw.max())
        w /= w.sum()
        ess = 1.0 / np.sum(w ** 2)
        if ess < cfg.ess_frac * N:
            a = _systematic_resample(w, rng)
            L, r, b = L[a], r[a], b[a]
            ancestors[k] = a
            logw = np.zeros(N)
        else:
            ancestors[k] = np.arange(N)
        levels[k] = L

    # backward pass: keep only the lineages that survived to the end, which is
    # a cheap stand-in for FFBSi and removes most of the forward-pass jitter
    w = np.exp(logw - logw.max())
    w /= w.sum()
    smoothed = np.zeros_like(levels)
    idx = np.arange(N)
    for k in range(n_bins - 1, -1, -1):
        smoothed[k] = levels[k][idx]
        idx = ancestors[k][idx]
    return dict(md_grid=md_grid, levels=smoothed, weights=w)

