#!/usr/bin/env python3
"""Per-well dip correction: predict each well's error line instead of the average one.

Decomposing every model's holdout error the way the 8th-place team did shows the same
thing in all of them -- the dominant error is one wrong line per well, not tail difficulty
and not ambiguity:

    model                     baseline   - per-well offset   - per-well linear trend
    ConvNeXt (1st place)        4.937          3.761                 3.072
    AnchorCNN                   6.024          4.148                 3.269
    particle filter (calib.)   15.845          9.491                 5.478
    8th place U-Net             6.44           4.58                  3.33

The particle filter is the extreme case: its *shape* is fine and almost all of its error
is a single wrong dip per well.  A global two-parameter calibration only removes the
population average of that; this module tries to predict each well's own line.

What makes it possible for the filter specifically is that the filter needs no training,
so it can be run over all 618 training wells and the corrector fitted there, with the
holdout never involved.  The same thing cannot be done for the CNNs without out-of-fold
predictions, which we do not have.

The feature that should carry the signal is geological rather than log-based: the
structural slope of *neighbouring training wells*.  Dip varies smoothly in space, so
nearby wells say what this well's slope ought to be, and the discrepancy between that and
the prefix estimate is what the filter gets wrong.  That is the 1st place's geo prior and
the 8th place's "surface context" / "spatial reliability" features, in miniature.

Neighbour pools always exclude the well itself, so a training well is never its own
neighbour.
"""

from __future__ import annotations

import numpy as np

FEATURES = ["r0", "r_nbr", "r_nbr_minus_r0", "log_nbr_dist", "pf_slope",
            "pf_slope_minus_nbr", "pf_slope_minus_r0", "eval_kft", "pre_kft",
            "gr_cov", "pf_spread", "dz_mean"]


def eval_slice(w):
    known = np.isfinite(w["tvt_input"])
    ev = np.flatnonzero(~known)
    return ev, known


def structural_slope(w, rows):
    """d(TVT + Z)/dMD fitted over ``rows``; needs ground truth, so training wells only."""
    md = w["md"][rows].astype(np.float64)
    s = (w["tvt"][rows] + w["z"][rows]).astype(np.float64)
    if len(rows) < 8:
        return 0.0
    return float(np.polyfit(md, s, 1)[0])


def prefix_slope(w, n_rows=256):
    """The same quantity from the known prefix -- what the filter starts from."""
    known = np.flatnonzero(np.isfinite(w["tvt_input"]))
    if len(known) < 8:
        return 0.0, 0.0
    idx = known[-min(n_rows, len(known)):]
    md = w["md"][idx].astype(np.float64)
    s = (w["tvt_input"][idx] + w["z"][idx]).astype(np.float64)
    A = np.column_stack([md - md.mean(), np.ones(len(md))])
    coef, *_ = np.linalg.lstsq(A, s, rcond=None)
    resid = s - A @ coef
    se = float(np.sqrt((resid @ resid) / max(len(md) - 2, 1) /
                       max(((md - md.mean()) ** 2).sum(), 1e-9)))
    return float(coef[0]), se


def build_pool(wells, names):
    """Neighbour pool: each training well's true structural slope and its XY position."""
    xy, slope, ids = [], [], []
    for nm in names:
        w = wells[nm]
        ev, _ = eval_slice(w)
        if len(ev) < 32:
            continue
        xy.append([float(np.nanmean(w["x"][ev])), float(np.nanmean(w["y"][ev]))])
        slope.append(structural_slope(w, ev))
        ids.append(nm)
    return dict(xy=np.asarray(xy, float), slope=np.asarray(slope, float), ids=np.asarray(ids))


def neighbour_slope(pool, x, y, exclude=None, k=8):
    """Inverse-distance weighted slope of the k nearest training wells."""
    if len(pool["ids"]) == 0:
        return 0.0, 1e6
    d = np.hypot(pool["xy"][:, 0] - x, pool["xy"][:, 1] - y)
    if exclude is not None:
        d = np.where(pool["ids"] == exclude, np.inf, d)
    order = np.argsort(d)[:k]
    dd = d[order]
    ok = np.isfinite(dd)
    if not ok.any():
        return 0.0, 1e6
    order, dd = order[ok], dd[ok]
    wgt = 1.0 / np.maximum(dd, 1.0)
    return float((pool["slope"][order] * wgt).sum() / wgt.sum()), float(dd.min())


def well_features(w, pool, pf_pred, pf_std, exclude=None):
    """Feature row for one well.  Uses nothing that is unavailable at inference time."""
    ev, known = eval_slice(w)
    md = w["md"][ev].astype(np.float64)
    r0, _ = prefix_slope(w)
    x, y = float(np.nanmean(w["x"][ev])), float(np.nanmean(w["y"][ev]))
    r_nbr, dist = neighbour_slope(pool, x, y, exclude=exclude)
    # what the filter itself implies for the structural slope
    s_pf = pf_pred + w["z"][ev].astype(np.float64)
    pf_slope = float(np.polyfit(md, s_pf, 1)[0]) if len(md) >= 8 else 0.0
    dz = float(np.polyfit(md, w["z"][ev].astype(np.float64), 1)[0]) if len(md) >= 8 else 0.0
    return np.array([
        r0, r_nbr, r_nbr - r0, np.log1p(dist), pf_slope,
        pf_slope - r_nbr, pf_slope - r0,
        (md[-1] - md[0]) / 1000.0, (md[0] - float(w["md"][0])) / 1000.0,
        float(np.isfinite(w["gr"][ev]).mean()), float(np.nanmean(pf_std)), dz,
    ], dtype=np.float64)


def error_line(pred, truth, md):
    """The well's error as a line: err ~ a + b * (MD - MD_PS)."""
    dm = md - md[0]
    A = np.column_stack([np.ones_like(dm), dm])
    a, b = np.linalg.lstsq(A, pred - truth, rcond=None)[0]
    return float(a), float(b)


class DipCorrector:
    """Two ridge regressions: the well's error offset, and its error slope."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.mu = self.sd = self.wa = self.wb = None

    def fit(self, X, a, b, log=print):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.column_stack([(X - self.mu) / self.sd, np.ones(len(X))])
        eye = np.eye(Z.shape[1])
        eye[-1, -1] = 0.0                                   # do not penalise the intercept
        G = Z.T @ Z + self.alpha * eye
        self.wa = np.linalg.solve(G, Z.T @ a)
        self.wb = np.linalg.solve(G, Z.T @ b)
        for name, tgt, w in (("offset a [ft]", a, self.wa), ("slope b [ft/ft]", b, self.wb)):
            pred = Z @ w
            ss = 1.0 - ((tgt - pred) ** 2).sum() / max(((tgt - tgt.mean()) ** 2).sum(), 1e-12)
            log(f"   {name:16s} in-sample R2 {ss:5.3f}   mean {tgt.mean():+.5f}")
        top = np.argsort(-np.abs(self.wb[:-1]))[:4]
        log("   strongest features for the slope: "
            + ", ".join(f"{FEATURES[i]} ({self.wb[i]:+.4f})" for i in top))
        return self

    def predict(self, X):
        Z = np.column_stack([(X - self.mu) / self.sd, np.ones(len(X))])
        return Z @ self.wa, Z @ self.wb
