"""Reference-GR enrichment: sibling laterals and the well's own prefix.

3rd place's strongest private-LB finding was that *which GR curve you match
against* mattered more than any network change: use the laterals that share a
typewell, aggregate them in fine TVT bins, and mix in the target well's own
known prefix as a self-reference. 5th place independently used sibling laterals
to correct the typewell at inference time, noting that the typewell is smoothed
along TVT and so systematically disagrees with the sharper lateral logs.

Both are the same observation: the typewell is a blurred, differently
calibrated view of rocks that the laterals see clearly.

Fold safety: `allowed_full` gates which wells may contribute *labelled*
(post-PS) samples. Prefix samples are always legitimate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import GR_MEAN, GR_STD

BIN_FT = 0.25           # 3rd place's fine-bin sibling reference
BIN_ORIGIN = -20000.0   # fixed grid origin: a per-group origin taken from the
                        # first well seen silently drops samples from any later
                        # sibling that sits above it


@dataclass
class SiblingBank:
    """typewell key -> binned GR profile contributed by its laterals."""

    bin_ft: float = BIN_FT
    _sum: dict = field(default_factory=dict)
    _cnt: dict = field(default_factory=dict)
    _by_well: dict = field(default_factory=dict)

    def _index(self, tvt):
        return ((tvt - BIN_ORIGIN) / self.bin_ft).astype(np.int64)

    def _accumulate(self, key, wid, tvt, gr):
        ok = np.isfinite(tvt) & np.isfinite(gr)
        if ok.sum() < 16:
            return
        tvt, gr = tvt[ok], gr[ok]
        idx = self._index(tvt)
        keep = idx >= 0
        idx, tvt, gr = idx[keep], tvt[keep], gr[keep]
        if idx.size == 0:
            return
        n = int(idx.max()) + 1
        for store, val in ((self._sum, gr), (self._cnt, np.ones_like(gr))):
            arr = store.get(key)
            if arr is None or len(arr) < n:
                new = np.zeros(max(n + 4096, 4096))
                if arr is not None:
                    new[:len(arr)] = arr
                arr = new
                store[key] = arr
            np.add.at(arr, idx, val)
        # keep the per-well contribution as (bin index, GR) in compact dtypes:
        # this bank is pickled into every dataloader worker
        contrib = (idx.astype(np.int32), gr.astype(np.float32))
        prev = self._by_well.setdefault(key, {}).get(wid)
        if prev is None:
            self._by_well[key][wid] = contrib
        else:                       # prefix and labelled sections both counted
            self._by_well[key][wid] = (np.concatenate([prev[0], contrib[0]]),
                                       np.concatenate([prev[1], contrib[1]]))

    def add_well(self, well, use_labels: bool):
        vis = np.isfinite(well.tvt_input)
        self._accumulate(well.tw_key, well.well_id,
                         well.tvt_input[vis], well.gr[vis])
        if use_labels and well.tvt is not None:
            hid = ~vis & np.isfinite(well.tvt)
            self._accumulate(well.tw_key, well.well_id,
                             well.tvt[hid], well.gr[hid])

    def profile(self, key: str, exclude_well: str = None):
        """(tvt_centres, gr_mean, count) for one typewell group."""
        if key not in self._sum:
            return None
        s, c = self._sum[key].copy(), self._cnt[key].copy()
        if exclude_well is not None:
            own = self._by_well.get(key, {}).get(exclude_well)
            if own is not None:
                idx, gr = own
                keep = (idx >= 0) & (idx < len(s))
                np.add.at(s, idx[keep], -gr[keep])
                np.add.at(c, idx[keep], -1.0)
        have = c > 0.5
        if have.sum() < 8:
            return None
        centres = BIN_ORIGIN + (np.arange(len(s))[have] + 0.5) * self.bin_ft
        return centres, s[have] / c[have], c[have]

    def n_siblings(self, key: str, exclude_well: str = None) -> int:
        wells = set(self._by_well.get(key, {}))
        wells.discard(exclude_well)
        return len(wells)


def build_sibling_bank(wells, allowed_full=None, bin_ft: float = BIN_FT):
    allowed_full = set() if allowed_full is None else set(allowed_full)
    bank = SiblingBank(bin_ft=bin_ft)
    for w in wells:
        bank.add_well(w, use_labels=w.well_id in allowed_full)
    return bank


# ---------------------------------------------------------------------- #
def self_reference(well, bin_ft: float = 1.0):
    """Level -> GR built from this well's own visible prefix.

    Same tool, same well, same calibration, and higher resolution than the
    typewell (2nd place's x6/x7 channels, following the host's own hint).
    """
    vis = np.isfinite(well.tvt_input) & np.isfinite(well.gr)
    if vis.sum() < 32:
        return None
    tvt, gr = well.tvt_input[vis], well.gr[vis]
    lo = float(tvt.min())
    n = max(int(np.ceil((tvt.max() - lo) / bin_ft)), 1)
    idx = np.clip(((tvt - lo) / bin_ft).astype(int), 0, n - 1)
    s = np.zeros(n)
    c = np.zeros(n)
    np.add.at(s, idx, gr)
    np.add.at(c, idx, 1.0)
    have = c > 0
    if have.sum() < 4:
        return None
    return lo + (np.arange(n)[have] + 0.5) * bin_ft, s[have] / c[have]


def sample_profile_on_rows(profile, anchor_tvt: float, row_tvt: np.ndarray,
                           max_gap: float = 4.0):
    """Interpolate a (tvt, gr) profile onto the canvas rows.

    Returns normalised GR and a validity mask; the mask is zero where the
    nearest supporting sample is further than `max_gap`, because a channel
    without its coverage mask invites the network to trust extrapolation.
    """
    H = len(row_tvt)
    if profile is None:
        return np.zeros(H), np.zeros(H)
    t, g = profile[0], profile[1]
    q = anchor_tvt + row_tvt
    val = np.interp(q, t, g, left=np.nan, right=np.nan)
    j = np.clip(np.searchsorted(t, q), 1, len(t) - 1)
    gap = np.minimum(np.abs(q - t[j - 1]), np.abs(q - t[j]))
    ok = np.isfinite(val) & (gap <= max_gap)
    return np.nan_to_num((val - GR_MEAN) / GR_STD) * ok, ok.astype(float)


def sibling_canvas_features(well, bank: SiblingBank, anchor_tvt: float,
                            row_tvt: np.ndarray):
    prof = bank.profile(well.tw_key, exclude_well=well.well_id) if bank else None
    sib_gr, sib_valid = sample_profile_on_rows(
        None if prof is None else (prof[0], prof[1]), anchor_tvt, row_tvt)
    self_gr, self_valid = sample_profile_on_rows(
        self_reference(well), anchor_tvt, row_tvt, max_gap=3.0)
    return dict(sib_gr_row=sib_gr, sib_valid_row=sib_valid,
                self_gr_row=self_gr, self_valid_row=self_valid)


def corrected_typewell(well, bank: SiblingBank, base_tw_gr=None,
                       strength: float = 1.0, min_siblings: int = 2):
    """5th place: add the sibling median residual profile to the typewell.

    The typewell GR is the expected value at each TVT; where sibling laterals
    consistently disagree with it, they -- not the typewell -- are right.
    """
    tw_gr = well.tw_gr.copy() if base_tw_gr is None else np.asarray(base_tw_gr).copy()
    if bank is None or bank.n_siblings(well.tw_key, well.well_id) < min_siblings:
        return tw_gr
    prof = bank.profile(well.tw_key, exclude_well=well.well_id)
    if prof is None:
        return tw_gr
    t, g, c = prof
    obs = np.interp(well.tw_tvt, t, g, left=np.nan, right=np.nan)
    resid = obs - tw_gr
    ok = np.isfinite(resid)
    if ok.sum() < 8:
        return tw_gr
    # smooth the residual: we are correcting calibration and blur, not chasing
    # the sibling wells' own noise
    r = np.where(ok, resid, 0.0)
    k = np.ones(21) / 21.0
    r_s = np.convolve(np.pad(r, 10, mode="edge"), k, mode="valid")
    w_s = np.convolve(np.pad(ok.astype(float), 10, mode="edge"), k, mode="valid")
    corr = np.where(w_s > 0.2, r_s / np.maximum(w_s, 1e-6), 0.0)
    return tw_gr + strength * corr
