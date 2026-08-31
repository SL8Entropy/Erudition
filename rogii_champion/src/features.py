"""Per-well feature provider: XY-neighbour, particle filter, sibling reference.

These are expensive and view-independent, while the canvas is rebuilt for every
MD phase and every re-anchor. So each is computed once per well on a fixed
dense grid and resampled onto whatever canvas asks for it.

The `allowed_full` argument is the fold-safety switch: it names the wells whose
labels this provider may use. For out-of-fold features it must exclude the
validation wells, or the label leaks in through a channel and the CV becomes
fiction.
"""
from __future__ import annotations

import numpy as np

from . import particle_filter as pf
from . import sibling_ref as sib
from . import xy_neighbor as xyn
from .canvas import calibrate_typewell
from .data import link_typewells


class FeatureProvider:
    def __init__(self, wells, allowed_full=None, use_pf: bool = False,
                 use_xy: bool = False, use_sibling: bool = False,
                 correct_typewell: bool = False, xy_step: int = 20,
                 pf_profiles=pf.PROFILES, verbose: bool = False):
        self.use_pf = use_pf
        self.use_xy = use_xy
        self.use_sibling = use_sibling
        self.correct_typewell = correct_typewell
        self.pf_profiles = pf_profiles
        self.verbose = verbose
        self._cache: dict = {}
        self.stats: dict = {}

        if use_sibling or correct_typewell:
            # sibling lookups are meaningless until the typewells are clustered
            # into their master series
            link_typewells(wells, verbose=verbose)

        self.cloud = (xyn.build_point_cloud(wells, allowed_full, step=xy_step)
                      if use_xy else None)
        self.bank = (sib.build_sibling_bank(wells, allowed_full)
                     if (use_sibling or correct_typewell) else None)

    # ------------------------------------------------------------------ #
    # Augmented wells must not read cached features -- the augmentation has
    # changed the trajectory and the GR the features are derived from. The
    # dataset passes cache=False for those.
    def _memo(self, key, fn, cache: bool):
        if not cache:
            return fn()
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def typewell_gr(self, well, cache: bool = True):
        """Calibrated (and optionally sibling-corrected) reference curve."""
        def build():
            base = well.tw_gr
            if self.correct_typewell and self.bank is not None:
                base = sib.corrected_typewell(well, self.bank, base)
            tmp = well.copy()
            tmp.tw_gr = base
            return calibrate_typewell(tmp)
        return self._memo(("tw", well.well_id), build, cache)

    def _xy_curve(self, well, cache: bool = True):
        def build():
            md_q = np.arange(well.md[0], well.md[-1] + 1.0, 64.0)
            pred, st = xyn.predict_xy(well, self.cloud, md_q)
            if cache:
                self.stats.setdefault(well.well_id, {}).update(st)
            return md_q, pred
        return self._memo(("xy", well.well_id), build, cache)

    def _pf_runs(self, well, cache: bool = True):
        def build():
            tw = self.typewell_gr(well, cache=cache)
            runs = []
            for cfg in self.pf_profiles:
                try:
                    runs.append(pf.run_pf(well, cfg, tw_gr=tw))
                except Exception:
                    continue
            return runs
        return self._memo(("pf", well.well_id), build, cache)

    # ------------------------------------------------------------------ #
    def extras(self, well, col_md, row_tvt, anchor_tvt, cache: bool = True):
        """Channels for one canvas view."""
        out = {}
        H, W = len(row_tvt), len(col_md)

        if self.use_xy and self.cloud is not None:
            md_q, curve = self._xy_curve(well, cache=cache)
            # the cached curve is relative to the well anchor; re-reference it
            # if this view uses a different anchor (re-anchoring TTA)
            shift = anchor_tvt - well.anchor_tvt
            out["xy_pred"] = np.interp(col_md, md_q, curve) - shift

        if self.use_pf:
            runs = self._pf_runs(well, cache=cache)
            heat = np.zeros((H, W))
            mean = np.zeros(W)
            spread = np.zeros(W)
            shift = anchor_tvt - well.anchor_tvt
            row_lo, row_step = row_tvt[0] + shift, row_tvt[1] - row_tvt[0]
            for r in runs:
                mids = 0.5 * (r["md_grid"][:-1] + r["md_grid"][1:])
                lv, w = r["levels"], r["weights"]
                m = (lv * w[None, :]).sum(axis=1)
                sd = np.sqrt(np.maximum(((lv - m[:, None]) ** 2 * w[None, :])
                                        .sum(axis=1), 0.0))
                ridx = np.clip(((lv - row_lo) / row_step).round().astype(int),
                               0, H - 1)
                hb = np.zeros((len(mids), H))
                for k in range(len(mids)):
                    np.add.at(hb[k], ridx[k], w)
                heat += np.stack([np.interp(col_md, mids, hb[:, h])
                                  for h in range(H)])
                mean += np.interp(col_md, mids, m) - shift
                spread += np.interp(col_md, mids, sd)
            if runs:
                heat = heat / len(runs)
                heat = heat / max(heat.max(), 1e-9)
                mean /= len(runs)
                spread /= len(runs)
            out.update(pf_heat=heat, pf_mean=mean, pf_spread=spread)

        if self.use_sibling and self.bank is not None:
            out.update(sib.sibling_canvas_features(well, self.bank, anchor_tvt,
                                                   row_tvt))
        return out

    # ------------------------------------------------------------------ #
    def xy_stats(self, well):
        if self.use_xy and self.cloud is not None:
            self._xy_curve(well)
        return self.stats.get(well.well_id, {})

    def pf_candidate(self, well):
        """PF path on the raw MD grid, as an ensemble candidate."""
        runs = self._pf_runs(well)
        if not runs:
            return np.full(well.n, well.anchor_tvt)
        acc = np.zeros(well.n)
        for r in runs:
            mids = 0.5 * (r["md_grid"][:-1] + r["md_grid"][1:])
            m = (r["levels"] * r["weights"][None, :]).sum(axis=1)
            acc += np.interp(well.md, mids, m)
        pred = acc / len(runs) + well.anchor_tvt
        vis = np.isfinite(well.tvt_input)
        pred[vis] = well.tvt_input[vis]
        return pred

    def xy_candidate(self, well):
        if not (self.use_xy and self.cloud is not None):
            return None
        md_q, curve = self._xy_curve(well)
        pred = np.interp(well.md, md_q, curve) + well.anchor_tvt
        vis = np.isfinite(well.tvt_input)
        pred[vis] = well.tvt_input[vis]
        return pred

    def clear(self):
        self._cache.clear()
