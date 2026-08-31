"""XY-neighbour structural prior (1st place) plus the five neighbourhood-quality
statistics used to decide when to trust it.

The geology: assume the local formation surface

    S = TVT + Z + C

is locally a plane. Then moving by (dX, dY) changes S by a*dX + b*dY, so

    dTVT = a*dX + b*dY - dZ

with (a, b) fitted by weighted least squares on nearby wells whose TVT is
known. Standalone this scores ~11.4 RMSE -- far worse than the network, but it
is built from entirely different information, which is what an ensemble wants.

Fold safety matters here and is explicit: `allowed_full` is the set of wells
whose *labels* may be used. For an out-of-fold prediction that set must exclude
the validation wells, otherwise the label leaks in through this channel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - scipy is in requirements
    cKDTree = None


@dataclass
class PointCloud:
    """Known (x, y, z, S) samples that a plane can be fitted to."""

    xy: np.ndarray          # (P, 2)
    z: np.ndarray           # (P,)
    s: np.ndarray           # (P,) = TVT + Z
    well: np.ndarray        # (P,) well id per point
    is_prefix: np.ndarray   # (P,) came from a visible prefix rather than a label
    tree: object = None

    def build(self):
        if cKDTree is not None and len(self.xy):
            self.tree = cKDTree(self.xy)
        return self


def build_point_cloud(wells, allowed_full=None, step: int = 20) -> PointCloud:
    """Collect known structural-surface samples from a set of wells.

    * every well contributes its visible prefix (always legitimate);
    * wells in `allowed_full` also contribute their labelled section.
    """
    allowed_full = set() if allowed_full is None else set(allowed_full)
    xs, zs, ss, wid, pref = [], [], [], [], []
    for w in wells:
        vis = np.isfinite(w.tvt_input)
        idx = np.flatnonzero(vis)[::step]
        if len(idx):
            xs.append(np.stack([w.x[idx], w.y[idx]], axis=1))
            zs.append(w.z[idx])
            ss.append(w.tvt_input[idx] + w.z[idx])
            wid.append(np.full(len(idx), w.well_id))
            pref.append(np.ones(len(idx), dtype=bool))
        if w.well_id in allowed_full and w.tvt is not None:
            hid = np.flatnonzero(~vis & np.isfinite(w.tvt))[::step]
            if len(hid):
                xs.append(np.stack([w.x[hid], w.y[hid]], axis=1))
                zs.append(w.z[hid])
                ss.append(w.tvt[hid] + w.z[hid])
                wid.append(np.full(len(hid), w.well_id))
                pref.append(np.zeros(len(hid), dtype=bool))
    if not xs:
        return PointCloud(np.zeros((0, 2)), np.zeros(0), np.zeros(0),
                          np.zeros(0, dtype=object), np.zeros(0, dtype=bool)).build()
    return PointCloud(np.concatenate(xs), np.concatenate(zs), np.concatenate(ss),
                      np.concatenate(wid), np.concatenate(pref)).build()


def predict_xy(well, cloud: PointCloud, query_md: np.ndarray,
               k: int = 256, radius: float = 6000.0,
               length_scale: float = 2000.0, ridge: float = 1e-4,
               max_rel: float = 150.0):
    """Relative-TVT prediction at each `query_md`, plus per-well quality stats.

    Returns `(pred_rel_tvt, stats)`. `pred_rel_tvt` is relative to the well
    anchor so it drops straight into a canvas channel.
    """
    n_q = len(query_md)
    out = np.zeros(n_q)
    stats_acc = {k_: [] for k_ in ("nbr_distance", "nbr_distance_q10",
                                   "prefix_weight_ratio", "radial_extrap_score",
                                   "path_alignment")}
    if cloud.tree is None or len(cloud.xy) < 8:
        return out, {k_: np.nan for k_ in stats_acc}

    qx = np.interp(query_md, well.md, well.x)
    qy = np.interp(query_md, well.md, well.y)
    qz = np.interp(query_md, well.md, well.z)
    # direction of travel at the query point, for statistic 5
    dqx = np.gradient(qx)
    dqy = np.gradient(qy)

    kk = min(k, len(cloud.xy))
    dist, idx = cloud.tree.query(np.stack([qx, qy], axis=1), k=kk,
                                 distance_upper_bound=radius)
    dist = np.atleast_2d(dist)
    idx = np.atleast_2d(idx)

    a_tvt = well.anchor_tvt
    own = cloud.well == well.well_id

    for i in range(n_q):
        m = np.isfinite(dist[i]) & (idx[i] < len(cloud.xy))
        if m.sum() < 4:
            out[i] = -(qz[i] - well.anchor_z)      # fall back to flat layers
            continue
        j = idx[i][m]
        d = dist[i][m]
        w = np.exp(-((d / length_scale) ** 2))
        # never let the query well's own hidden section in: it is the label
        keep = ~(own[j] & ~cloud.is_prefix[j])
        j, d, w = j[keep], d[keep], w[keep]
        if len(j) < 4 or w.sum() <= 1e-9:
            out[i] = -(qz[i] - well.anchor_z)
            continue

        P = cloud.xy[j]
        S = cloud.s[j]
        A = np.column_stack([P[:, 0] - qx[i], P[:, 1] - qy[i], np.ones(len(j))])
        Aw = A * w[:, None]
        M = Aw.T @ A + ridge * np.trace(A.T @ A + 1e-9) / 3.0 * np.eye(3)
        try:
            coef = np.linalg.solve(M, Aw.T @ S)
        except np.linalg.LinAlgError:
            coef = np.array([0.0, 0.0, float(np.average(S, weights=w))])
        s_q = coef[2]                       # A is centred on the query point
        out[i] = (s_q - qz[i]) - a_tvt

        centre = np.average(P, axis=0, weights=w)
        r_q = np.hypot(qx[i] - centre[0], qy[i] - centre[1])
        r_n = np.average(np.hypot(P[:, 0] - centre[0], P[:, 1] - centre[1]),
                         weights=w)
        stats_acc["nbr_distance"].append(float(np.average(d, weights=w)))
        stats_acc["nbr_distance_q10"].append(float(np.quantile(d, 0.10)))
        stats_acc["prefix_weight_ratio"].append(
            float(w[cloud.is_prefix[j] & own[j]].sum() / max(w.sum(), 1e-9)))
        stats_acc["radial_extrap_score"].append(float(r_q / max(r_n, 1e-6)))
        v = np.array([dqx[i], dqy[i]])
        nv = P - np.array([qx[i], qy[i]])
        nn = np.linalg.norm(nv, axis=1) * (np.linalg.norm(v) + 1e-9)
        cos = np.abs(nv @ v) / np.maximum(nn, 1e-9)
        stats_acc["path_alignment"].append(float(np.average(cos, weights=w)))

    # a plane fit is only locally valid; smooth the result along MD so that
    # neighbour-set changes do not inject steps into the channel
    if n_q >= 9:
        ker = np.ones(9) / 9.0
        out = np.convolve(np.pad(out, 4, mode="edge"), ker, mode="valid")
    # a prediction outside the canvas window cannot be represented anyway, and
    # an unclipped plane fit in a sparse neighbourhood can run to hundreds of ft
    out = np.clip(out, -max_rel, max_rel)

    stats = {k_: (float(np.mean(v)) if v else np.nan)
             for k_, v in stats_acc.items()}
    return out, stats


# ---------------------------------------------------------------------- #
def xy_safety_mask(stats_by_well: dict, thresholds: dict) -> dict:
    """Well -> True when the XY neighbourhood looks like the training regime.

    1st place routes the ~10% of wells that fail this to models without XY
    channels. Directions follow the semantics of each statistic: distances and
    extrapolation scores must be *small*, alignment must be *large*.
    """
    out = {}
    for wid, st in stats_by_well.items():
        ok = True
        for key, thr in thresholds.items():
            v = st.get(key, np.nan)
            if not np.isfinite(v):
                ok = False
                break
            ok &= (v > thr) if key == "path_alignment" else (v < thr)
        out[wid] = bool(ok)
    return out


def fit_safety_thresholds(stats_by_well: dict, q: float = 0.95) -> dict:
    """Quantiles of the training distribution, as 1st place used (95th pct)."""
    keys = ("nbr_distance", "nbr_distance_q10", "prefix_weight_ratio",
            "radial_extrap_score", "path_alignment")
    thr = {}
    for k_ in keys:
        v = np.array([s.get(k_, np.nan) for s in stats_by_well.values()],
                     dtype=float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        thr[k_] = float(np.quantile(v, 1.0 - q if k_ == "path_alignment" else q))
    return thr
