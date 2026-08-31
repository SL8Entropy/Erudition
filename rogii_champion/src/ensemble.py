"""Blending OOF predictions, and the routing that 1st place used.

Two things worth stating up front.

*Routing beats one global weight vector.* 1st place kept two weight vectors --
one for general wells, one for the ~10% whose XY neighbourhood looks unlike
anything in training -- and routed between them on neighbourhood statistics.
The XY models are better on private and worse on public; the routing is what
makes that safe.

*Fitted weights are not free.* 4th place measured, with nested leave-one-group-
out over wells, that fitted weights lost to equal weights (6.578 vs 6.456 for
15 members). `fit_weights` is here because it is useful for diagnosis, but the
nested estimate is reported alongside it and equal weights are the default.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .metrics import accept, per_well_sse, pooled_rmse, score_report


# ---------------------------------------------------------------------- #
def load_oof(run_dir, variant: str = None):
    """Average every OOF file in a run directory, per well."""
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob("oof_*.npz"))
    if not files:
        raise FileNotFoundError(f"no oof_*.npz in {run_dir}")
    acc, cnt, sig = {}, {}, {}
    for f in files:
        z = np.load(f)
        for k in z.files:
            if k.startswith("pred__"):
                wid = k[len("pred__"):]
                v = z[k]
                acc[wid] = v if wid not in acc else acc[wid] + v
                cnt[wid] = cnt.get(wid, 0) + 1
            elif k.startswith("sigma__"):
                wid = k[len("sigma__"):]
                v = z[k]
                if v.size > 1:
                    sig[wid] = v if wid not in sig else sig[wid] + v
    preds = {w: acc[w] / cnt[w] for w in acc}
    sigmas = {w: sig[w] / cnt[w] for w in sig}
    return preds, sigmas


def blend(members: dict, weights: dict):
    """members[name][well] -> path. Weights need not sum to one."""
    names = [n for n in members if weights.get(n, 0.0) > 0]
    if not names:
        raise ValueError("no member has positive weight")
    wells = set(members[names[0]])
    for n in names[1:]:
        wells &= set(members[n])
    tot = sum(weights[n] for n in names)
    return {w: sum(weights[n] * members[n][w] for n in names) / tot
            for w in wells}


def route_blend(members: dict, weights_general: dict, weights_fallback: dict,
                xy_safe: dict):
    """Per-well routing between two weight vectors."""
    out = {}
    common = set.intersection(*[set(m) for m in members.values()])
    for w in common:
        wts = weights_general if xy_safe.get(w, True) else weights_fallback
        names = [n for n in members if wts.get(n, 0.0) > 0]
        tot = sum(wts[n] for n in names)
        if tot <= 0:
            continue
        out[w] = sum(wts[n] * members[n][w] for n in names) / tot
    return out


# ---------------------------------------------------------------------- #
def _design(members, wells, names):
    rows_y, rows_X = [], []
    for w in wells:
        if any(w.well_id not in members[n] for n in names):
            continue
        m = w.eval_mask & np.isfinite(w.tvt)
        if not m.any():
            continue
        rows_y.append(w.tvt[m])
        rows_X.append(np.stack([members[n][w.well_id][m] for n in names], 1))
    if not rows_y:
        raise RuntimeError("no overlapping wells for weight fitting")
    return np.concatenate(rows_X), np.concatenate(rows_y)


def fit_weights(members: dict, wells: list, names=None, nested_groups: int = 5):
    """Non-negative least squares weights, plus a nested estimate of their worth.

    The nested number is the one to believe: in-sample fitted weights always
    look good.
    """
    from scipy.optimize import nnls
    names = names or sorted(members)
    X, y = _design(members, wells, names)
    w, _ = nnls(X, y)
    w = w / max(w.sum(), 1e-9)
    fitted = {n: float(v) for n, v in zip(names, w)}

    rng = np.random.default_rng(0)
    ids = [w_.well_id for w_ in wells if all(w_.well_id in members[n] for n in names)]
    grp = {wid: int(g) for wid, g in
           zip(ids, rng.integers(0, nested_groups, len(ids)))}
    nested_pred, equal_pred = {}, {}
    for g in range(nested_groups):
        tr = [w_ for w_ in wells if grp.get(w_.well_id, -1) not in (g, -1)]
        te = [w_ for w_ in wells if grp.get(w_.well_id, -1) == g]
        if not tr or not te:
            continue
        Xt, yt = _design(members, tr, names)
        wg, _ = nnls(Xt, yt)
        wg = wg / max(wg.sum(), 1e-9)
        for w_ in te:
            nested_pred[w_.well_id] = sum(
                wg[i] * members[n][w_.well_id] for i, n in enumerate(names))
            equal_pred[w_.well_id] = np.mean(
                [members[n][w_.well_id] for n in names], axis=0)
    return dict(
        weights=fitted,
        in_sample=pooled_rmse({w_.well_id: sum(
            fitted[n] * members[n][w_.well_id] for n in names)
            for w_ in wells if all(w_.well_id in members[n] for n in names)}, wells),
        nested=pooled_rmse(nested_pred, wells),
        equal=pooled_rmse(equal_pred, wells),
    )


# ---------------------------------------------------------------------- #
def compare(base_preds, treat_preds, wells, label: str = "", min_k: int = 52):
    """Report a candidate against a baseline with the acceptance test."""
    b = score_report(base_preds, wells)
    t = score_report(treat_preds, wells)
    r = accept(base_preds, treat_preds, wells, min_k=min_k)
    print(f"\n=== {label or 'candidate'} ===")
    print(f"  base   pooled RMSE {b['pooled_rmse']:.4f}")
    print(f"  treat  pooled RMSE {t['pooled_rmse']:.4f}   "
          f"delta {t['pooled_rmse'] - b['pooled_rmse']:+.4f}")
    print(f"  leave-largest-contribution-out: k* = {r['k_star']} "
          f"(need >= {min_k})  ->  {'ACCEPT' if r['accepted'] else 'reject'}")
    print(f"  top-5 wells hold {t['top5_share'] * 100:.0f}% of the treat error")
    return dict(base=b, treat=t, llco=r)
