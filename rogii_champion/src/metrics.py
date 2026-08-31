"""Scoring and the model-selection protocol.

The competition metric is *pooled* RMSE: every scored row of every well goes
into one pool. Squaring means a well that is off by 40 ft contributes 400x more
per row than one off by 2 ft, so the score is dominated by a handful of
catastrophic wells -- and so is any *difference* between two models.

Hence the acceptance test, taken from 2nd place: an improvement that evaporates
when you drop the few wells contributing most of it was luck, not signal.
"""
from __future__ import annotations

import numpy as np


def eval_rows(well):
    m = well.eval_mask & np.isfinite(well.tvt)
    return m


def pooled_rmse(preds: dict, wells: list) -> float:
    se, n = 0.0, 0
    for w in wells:
        p = preds.get(w.well_id)
        if p is None:
            continue
        m = eval_rows(w)
        d = p[m] - w.tvt[m]
        se += float(d @ d)
        n += int(m.sum())
    return float(np.sqrt(se / max(n, 1)))


def well_mean_rmse(preds: dict, wells: list) -> float:
    vals = []
    for w in wells:
        p = preds.get(w.well_id)
        if p is None:
            continue
        m = eval_rows(w)
        if m.sum():
            vals.append(float(np.sqrt(np.mean((p[m] - w.tvt[m]) ** 2))))
    return float(np.mean(vals)) if vals else float("nan")


def per_well_sse(preds: dict, wells: list):
    """well -> (sum of squared error, row count)."""
    out = {}
    for w in wells:
        p = preds.get(w.well_id)
        if p is None:
            continue
        m = eval_rows(w)
        d = p[m] - w.tvt[m]
        out[w.well_id] = (float(d @ d), int(m.sum()))
    return out


def score_report(preds: dict, wells: list) -> dict:
    sse = per_well_sse(preds, wells)
    tot = sum(v[0] for v in sse.values())
    n = sum(v[1] for v in sse.values())
    worst = sorted(sse.items(), key=lambda kv: -kv[1][0])[:5]
    return dict(
        pooled_rmse=float(np.sqrt(tot / max(n, 1))),
        well_mean_rmse=well_mean_rmse(preds, wells),
        n_wells=len(sse), n_rows=n,
        worst_wells=[(k, float(np.sqrt(v[0] / max(v[1], 1)))) for k, v in worst],
        top5_share=float(sum(v[1][0] for v in worst) / max(tot, 1e-9)),
    )


# ---------------------------------------------------------------------- #
def llco_curve(base_preds: dict, treat_preds: dict, wells: list):
    """Leave-largest-contribution-out curve (2nd place).

    Remove wells in descending order of |SSE_treat - SSE_base| and recompute
    the pooled-RMSE difference. `k_star` is the number of removals at which the
    improvement first disappears; a candidate whose gain dies after a handful
    of wells was lucky.
    """
    b = per_well_sse(base_preds, wells)
    t = per_well_sse(treat_preds, wells)
    ids = [w for w in b if w in t]
    g = np.array([t[w][0] - b[w][0] for w in ids])
    n = np.array([b[w][1] for w in ids], dtype=float)
    order = np.argsort(-np.abs(g))

    sse_b = np.array([b[w][0] for w in ids])
    sse_t = np.array([t[w][0] for w in ids])
    keep = np.ones(len(ids), dtype=bool)
    curve = []
    for k in range(min(len(ids), 120)):
        if k > 0:
            keep[order[k - 1]] = False
        nn = n[keep].sum()
        if nn <= 0:
            break
        curve.append(float(np.sqrt(sse_t[keep].sum() / nn)
                           - np.sqrt(sse_b[keep].sum() / nn)))
    curve = np.array(curve)
    neg = np.flatnonzero(curve >= 0)
    k_star = int(neg[0]) if len(neg) else len(curve)
    return dict(delta0=float(curve[0]) if len(curve) else 0.0,
                k_star=k_star, curve=curve.tolist(),
                removed=[ids[i] for i in order[:10]])


def accept(base_preds, treat_preds, wells, min_k: int = 52) -> dict:
    """Accept a candidate only if its gain survives removing `min_k` wells.

    `min_k` defaults to 52 -- the size of the public leaderboard, which is the
    scale at which 2nd place required a gain to be robust.
    """
    r = llco_curve(base_preds, treat_preds, wells)
    r["accepted"] = bool(r["delta0"] < 0 and r["k_star"] >= min_k)
    r["min_k"] = min_k
    return r
