"""Physically consistent synthetic wells (1st place's Z-shift, extended).

The generative assumption every top team leaned on:

    GR_observed(MD) = f(TVT(MD)) + r(MD)

with f the typewell profile and r a residual. Because the right-hand side is a
function of TVT, *any* TVT path can be turned into a consistent well. So take a
real well, keep the geology S = TVT + Z fixed, draw a different trajectory
through it, and regenerate GR by reading the typewell along the new path.

Two refinements over the plain version:

* the residual r is pasted from a real well rather than drawn white (2nd
  place), so the synthetic GR keeps real autocorrelation and the well-specific
  systematic offset;
* rare fault jumps are injected as steps in S, which is the only way the model
  ever sees the failure mode that dominates the pooled metric.

1st place found that mixing this into real-well training beat a separate
pretraining stage, so it is used as an augmentation, not a curriculum.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PathBank:
    """Real dTVT/dMD sequences and real GR residuals, for bootstrapping."""

    deltas: list = field(default_factory=list)     # per-well dTVT arrays
    residuals: list = field(default_factory=list)  # per-well GR residual arrays

    def add(self, well):
        if well.tvt is None:
            return
        ok = np.isfinite(well.tvt)
        if ok.sum() < 512:
            return
        d = np.diff(well.tvt[ok]) / np.maximum(np.diff(well.md[ok]), 1e-6)
        self.deltas.append(d.astype(np.float32))
        ref = np.interp(well.tvt[ok], well.tw_tvt, well.tw_gr,
                        left=np.nan, right=np.nan)
        res = well.gr[ok] - ref
        if np.isfinite(res).mean() > 0.5:
            self.residuals.append(np.nan_to_num(res).astype(np.float32))

    def sample_path(self, n: int, rng, block: int = 400) -> np.ndarray:
        """Block bootstrap of dTVT/dMD -> a plausible new relative TVT path."""
        if not self.deltas:
            return np.zeros(n)
        out = np.empty(n, dtype=np.float64)
        i = 0
        while i < n:
            src = self.deltas[rng.integers(len(self.deltas))]
            b = min(int(rng.integers(block // 2, block * 2)), n - i)
            if len(src) <= b:
                seg = np.resize(src, b)
            else:
                s = int(rng.integers(0, len(src) - b))
                seg = src[s:s + b]
            if rng.random() < 0.5:
                seg = seg[::-1]
            out[i:i + b] = seg
            i += b
        return np.cumsum(out)

    def sample_residual(self, n: int, rng) -> np.ndarray:
        if not self.residuals:
            return np.zeros(n)
        src = self.residuals[rng.integers(len(self.residuals))]
        if len(src) <= n:
            return np.resize(src, n)
        s = int(rng.integers(0, len(src) - n))
        return src[s:s + n].copy()


def build_path_bank(wells) -> PathBank:
    bank = PathBank()
    for w in wells:
        bank.add(w)
    return bank


def _smooth_noise(n: int, rng, scale: float, corr_ft: float = 40.0):
    """Smoothed white noise: TVT jitter with a realistic correlation length."""
    if scale <= 0:
        return np.zeros(n)
    k = max(int(corr_ft), 1)
    ker = np.ones(k) / k
    raw = rng.normal(0.0, 1.0, n + 4 * k)
    sm = np.convolve(raw, ker, mode="same")[2 * k:2 * k + n]
    sd = sm.std() + 1e-9
    return sm / sd * scale


def synth_well(well, bank: PathBank, rng, *, path_scale: float = 1.0,
               fault_prob: float = 0.15, fault_std: float = 14.0,
               tvt_noise: float = 0.9, residual_mix: float = 0.7,
               keep_prefix: bool = True):
    """Return a new `Well` with a resampled trajectory and regenerated GR.

    The typewell, the MD grid and the prefix length are inherited from the real
    well, so the synthetic sample sits in the same regime as the data.
    """
    if well.tvt is None or not np.isfinite(well.tvt).all():
        return well
    w = well.copy()
    n = w.n
    ps = w.ps_idx
    a_tvt = w.anchor_tvt

    S = well.tvt + well.z                      # the geology we keep
    new_rel = bank.sample_path(n, rng) * path_scale
    new_rel = new_rel - new_rel[max(ps - 1, 0)]
    tvt_new = a_tvt + new_rel

    # keep the path inside the typewell's coverage, else GR cannot be generated
    lo, hi = float(w.tw_tvt[0]) + 2.0, float(w.tw_tvt[-1]) - 2.0
    if tvt_new.max() > hi or tvt_new.min() < lo:
        span = max(tvt_new.max() - tvt_new.min(), 1e-6)
        if span > (hi - lo):
            tvt_new = a_tvt + new_rel * ((hi - lo) / span) * 0.9
        tvt_new = np.clip(tvt_new, lo, hi)

    S_new = S.copy()
    if rng.random() < fault_prob:
        for _ in range(int(rng.integers(1, 3))):
            cut = int(rng.integers(ps, n)) if n > ps else n - 1
            S_new[cut:] += rng.normal(0.0, fault_std)
    if keep_prefix:
        tvt_new[:ps] = well.tvt[:ps]           # do not disturb the given prefix

    z_new = S_new - tvt_new

    ref = np.interp(tvt_new + _smooth_noise(n, rng, tvt_noise),
                    w.tw_tvt, w.tw_gr, left=np.nan, right=np.nan)
    res = bank.sample_residual(n, rng) * residual_mix
    own = well.gr - np.interp(well.tvt, w.tw_tvt, w.tw_gr,
                              left=np.nan, right=np.nan)
    own = np.nan_to_num(own) * (1.0 - residual_mix)
    gr_new = ref + res + own
    gr_new = np.where(np.isfinite(gr_new), gr_new, np.nan)
    gr_new[~np.isfinite(well.gr)] = np.nan     # inherit the real missingness

    w.tvt = tvt_new
    w.z = z_new
    w.gr = gr_new
    w.tvt_input = np.where(np.arange(n) < ps, tvt_new, np.nan)
    w.meta = dict(w.meta, synthetic=True)
    return w
