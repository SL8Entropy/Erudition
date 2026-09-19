#!/usr/bin/env python3
"""The particle filter as *input channels* rather than as a blend member.

The whole filter campaign here ran into one wall: after taking the filter from 28.98 ft to
11.95 ft, it still earns a weight of 0.02 in a blend with the ConvNeXt, because making it
more accurate also made it more correlated (0.340 -> 0.367) and the break-even threshold
`RMSE < sigma_c / rho` moved faster than the accuracy did.

22nd place hit the identical wall and named the way round it: stop blending the filter's
*output* and feed its per-row estimate and per-row spread in as extra network channels
(their 6.997 -> 6.719).  The difference is not cosmetic.  A blend can only take a convex
combination of two answers, so a partner that is right on 10% of wells and badly wrong on
the rest is worth almost nothing.  As a channel the same partner is worth a great deal: the
network sees the filter's belief *and* its spread at every column and can learn where to
believe it, which is a per-column decision rather than one scalar weight per well.

Two channels are added, both on the (level x column) grid the model already works on:

    pf_bump   exp(-((L - mu_m) / sigma_m)^2 / 2)   the filter's belief, rendered on the
                                                   same state grid as the model's output
    pf_conf   1 / (1 + sigma_m / REF_SD)           how sharp the filter is at that column,
                                                   broadcast over levels

`pf_bump` is deliberately the same shape as the author's existing `k_map` channel, which
renders the *known* prefix path as a soft bump; this is the same object extended across the
eval zone by the filter, so the model already has machinery that knows what to do with it.

Cost and correctness.  The filter is pure numpy and costs ~24 ms per well per profile, so
it runs inside the DataLoader workers on the *augmented* well.  It is evaluated once per
well on a 4 ft MD grid and interpolated to the column centres, so the 8 MD phases of TTA
share one filter run instead of repeating it eight times.  That matters: augmentation
re-anchors the well at a random row and may reverse it along MD, and a filter cached from
the real anchor would be answering a different question than the sample asks.  Computing it
per sample keeps the channel honest at the cost of ~7 s per epoch per profile across four
workers.

Leakage: `run_pf` reads `tvt_input`, `gr`, `z`, `md` and the typewell, and never `tvt` --
verified by inspection of `particle_filter.py` and `canvas.calibrate_typewell`.  The channel
is therefore computable at inference for a holdout well exactly as it is in training.
"""

from __future__ import annotations

import numpy as np

REF_SD = 16.0          # feet; the spread at which pf_conf falls to 0.5
MIN_SD = 2.0           # feet; floor on the bump width, so a confident filter is not a spike
MAX_SD = 64.0          # feet; above this the filter is saying nothing


def pf_columns(w: dict, md_centres: np.ndarray, n_profiles: int = 2):
    """Run the filter on ``w`` and report (mean TVT, spread) at each column centre.

    Returns (mu, sd) as float32, or (None, None) if the filter cannot run on this well --
    a short or all-known prefix, for instance.  Callers treat that as "no opinion".
    """
    if n_profiles <= 0:
        return None, None
    try:
        from rogii_champion.src.particle_filter import PROFILES
        from anchor_pf import pf_predict, to_champion_well
        name = str(w.get("name", "aug"))
        mu, sd = pf_predict(to_champion_well(name, w), md_centres.astype(np.float64),
                            PROFILES[:n_profiles])
    except Exception:
        return None, None
    if not np.isfinite(mu).any():
        return None, None
    return mu.astype(np.float32), sd.astype(np.float32)


CURVE_STEP_FT = 1.0     # resolution of the cached mu(MD) / sigma(MD) curve; the cost is the
                        # filter run, not the grid, so this is made fine enough to be exact
CURVE_OVERHANG_FT = 128.0  # how far past the last row to extend the curve
CACHE_N = 256           # entries; covers a full TTA chunk many times over

_CURVE_CACHE = []       # [(key_arrays, n_profiles, grid, mu, sd)], most-recent first


def _pf_curve(w, n_profiles):
    """mu(MD) and sigma(MD) for this well, computed once and reused across MD phases.

    TTA evaluates every well at 8 MD phases, and re-phasing only moves the column
    boundaries -- the filter's answer is a function of MD and does not change.  Running it
    per phase therefore did the same work eight times.  Here it is run once on a 4 ft grid
    and the column centres are interpolated from it, which is what ``pf_predict`` already
    does internally from the filter's own coarser grid.

    The cache is keyed on the *identity* of the arrays the filter reads, and holds a
    reference to them, so an id cannot be recycled into a false hit while the entry lives.
    Augmentation builds fresh arrays per sample, so training never hits the cache and never
    sees a stale curve; inference reuses the same well object across phases and always does.
    """
    key = (w["md"], w["gr"], w["tvt_input"], w["tw_gr"], w["z"])
    for ent in _CURVE_CACHE:
        if ent[1] == n_profiles and all(a is b for a, b in zip(ent[0], key)):
            return ent[2], ent[3], ent[4]
    md = np.asarray(w["md"], dtype=np.float64)
    # The filter's own MD grid runs past the well's last row, and the final column's centre
    # can sit up to half a column beyond it, so the curve is extended to stay inside the
    # filter's range there instead of clamping early.
    grid = np.arange(md[0], md[-1] + CURVE_OVERHANG_FT, CURVE_STEP_FT)
    mu, sd = pf_columns(w, grid, n_profiles)
    _CURVE_CACHE.insert(0, (key, n_profiles, grid, mu, sd))
    del _CURVE_CACHE[CACHE_N:]
    return grid, mu, sd


def pf_level_columns(w: dict, meta: dict, n_cols: int, n_profiles: int = 2):
    """``_pf_curve`` mapped into the model's level units for one built sample.

    ``anchor0`` is the TVT datum the sample's levels are measured from; ``build_compact``
    derives it as ``tvt_ps - tvt_shift`` and stores both in ``meta``.
    """
    md0 = float(meta["md0"])
    cw = float(meta["colw_in"])
    anchor0 = float(meta["tvt_ps"]) - float(meta["tvt_shift"])
    centres = md0 + (np.arange(n_cols, dtype=np.float64) + 0.5) * cw
    grid, mu, sd = _pf_curve(w, n_profiles)
    if mu is None:
        return (np.zeros(n_cols, np.float32), np.zeros(n_cols, np.float32))
    # np.interp clamps to the end values outside [grid[0], grid[-1]], which is exactly what
    # running the filter at the column centres did.  The last active column's centre can sit
    # up to half a column past the well's final MD, so clamping (not masking) is what keeps
    # this a pure speedup rather than a behaviour change.
    lv = (np.interp(centres, grid, mu) - anchor0).astype(np.float32)
    sd = np.clip(np.interp(centres, grid, sd), MIN_SD, MAX_SD).astype(np.float32)
    bad = ~np.isfinite(lv)
    if bad.any():
        lv = np.where(bad, 0.0, lv).astype(np.float32)
        sd = np.where(bad, 0.0, sd).astype(np.float32)   # 0 = the "no opinion" sentinel
    return lv, sd


def make_channels(pf_lv, pf_sd, levels, act, n_cols):
    """The two channels, on the model's (level x column) grid.

    ``pf_lv``/``pf_sd`` are (B, M') as built for the input grid; if a stem has reduced the
    column count they are pooled to match, mean for the level and quadratic mean for the
    spread (averaging a belief widens it).
    """
    import torch
    B, M_in = pf_lv.shape
    if M_in != n_cols:
        st = max(1, M_in // n_cols)
        pf_lv = pf_lv[:, :n_cols * st].view(B, n_cols, st).mean(-1)
        pf_sd = pf_sd[:, :n_cols * st].view(B, n_cols, st).pow(2).mean(-1).sqrt()
    has = (pf_sd > 0).to(pf_lv.dtype)
    sd = pf_sd.clamp(min=MIN_SD)
    L = levels.to(pf_lv.dtype).to(pf_lv.device)[None, :, None]     # (1, T, 1)
    zscore = (L - pf_lv[:, None, :]) / sd[:, None, :]
    bump = torch.exp(-0.5 * zscore.clamp(-8.0, 8.0) ** 2) * has[:, None, :]
    conf = (1.0 / (1.0 + sd / REF_SD)) * has
    a = act[:, None, :]
    return torch.stack([bump * a, conf[:, None, :].expand_as(bump) * a], dim=1)


def append_channels(x, items, device):
    """Concatenate the two filter channels onto an assembled batch, if the items carry them.

    Items built with ``_PF_PROFILES == 0`` are 14-tuples and pass through unchanged, which
    is what keeps every earlier run and checkpoint reproducible.
    """
    import torch
    import gr2tvt_data as gd
    extra = []
    if len(items[0]) > 17 and items[0][17] is not None:
        # sibling disagreement: one value per LEVEL, broadcast along the column axis
        sg = torch.tensor(np.stack([it[17] for it in items]), device=device, dtype=x.dtype)
        extra.append(sg[:, :, None].expand(-1, -1, x.shape[-1]))
    if len(items[0]) <= 16 or items[0][16] is None:
        return torch.cat([x] + [e[:, None] for e in extra], 1) if extra else x
    pf_lv = torch.tensor(np.stack([it[15] for it in items]), device=device, dtype=x.dtype)
    pf_sd = torch.tensor(np.stack([it[16] for it in items]), device=device, dtype=x.dtype)
    ncol = torch.tensor(np.stack([it[5] for it in items]), device=device)
    M = x.shape[-1]
    act = (torch.arange(M, device=device)[None, :] < ncol[:, None]).to(x.dtype)
    ch = make_channels(pf_lv, pf_sd, torch.as_tensor(gd.LEVELS), act, M)
    return torch.cat([x, ch] + [(e * act[:, None, :])[:, None] for e in extra], 1)
