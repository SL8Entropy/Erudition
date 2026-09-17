#!/usr/bin/env python3
"""Is a model's per-well dip error predictable?  A cheap, cross-fitted answer.

Every model scored on this holdout carries most of its error as one wrong line per well:
removing a per-well linear trend takes the ConvNeXt 4.937 -> 3.072 and the AnchorCNN
6.024 -> 3.269.  That oracle uses the answer.  The question this script settles is whether
the same line can be *predicted* from things known before drilling -- which would turn an
oracle into a correction.

It is already known to work for the particle filter (28.98 -> 13.07 ft).  Whether it works
for a CNN, whose dip error is smaller and noisier, is a different question, and this
answers it in minutes instead of the 2.5 hours a proper out-of-fold training round costs.

Method.  The 155 holdout wells are split into ``--folds`` groups.  Each well's correction
comes from a corrector fitted on the *other* groups, so no well contributes to its own
correction.  The base model never saw any holdout well, and the neighbour pool is built
from training wells only, so nothing here is fitted on data it is then scored against.

Two controls decide whether the result means anything:

    oracle line   remove each well's true error line.  The ceiling.
    global line   remove ONE line, fitted on the other folds, from every well.  This is
                  what a plain calibration does, and the per-well corrector has to beat it
                  or it has learned nothing except the average.

    python anchor_diptest.py --preds runs/C_s1/holdout_predictions.pqt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from anchor_dip import DipCorrector, build_pool, error_line, well_features  # noqa: E402
from anchor_data import eval_rows, load_wells, split_wells, well_names      # noqa: E402


def main(args):
    names = well_names(args.data)
    tr_names, ho_names = split_wells(names, args.train_frac)
    wells = load_wells(Path(args.data), names, Path(args.cache) if args.cache else None,
                       log=lambda *a: None)
    df = pd.read_parquet(args.preds).sort_values(["well_id", "submit_index"]).reset_index(drop=True)
    has_std = "TVT_std" in df.columns
    pool = build_pool(wells, tr_names)          # neighbours are TRAINING wells only
    print(f"{args.preds}")
    print(f"  {df.well_id.nunique()} wells, neighbour pool {len(pool['ids'])} training wells, "
          f"model spread feature {'present' if has_std else 'absent'}")

    X, A, B, keep = [], [], [], []
    per = {}
    for nm, g in df.groupby("well_id", sort=True):
        w = wells[nm]
        ev = eval_rows(w)
        if len(ev) < 32 or len(g) != len(ev):
            continue
        md = w["md"][ev].astype(np.float64)
        pred = g.TVT_pred.to_numpy(np.float64)
        truth = g.TVT.to_numpy(np.float64)
        sd = g.TVT_std.to_numpy(np.float64) if has_std else np.zeros(len(g))
        a, b = error_line(pred, truth, md)
        X.append(well_features(w, pool, pred, sd))
        A.append(a); B.append(b); keep.append(nm)
        per[nm] = (md, pred, truth)
    X = np.asarray(X); A = np.asarray(A); B = np.asarray(B)
    print(f"  targets: offset a  mean {A.mean():+7.3f} sd {A.std():6.3f} ft ; "
          f"slope b  mean {B.mean():+.5f} sd {B.std():.5f} ft/ft")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(keep))
    folds = [order[i::args.folds] for i in range(args.folds)]
    a_hat = np.zeros(len(keep)); b_hat = np.zeros(len(keep))
    a_glb = np.zeros(len(keep)); b_glb = np.zeros(len(keep))
    for f in folds:
        tr = np.setdiff1d(order, f)
        c = DipCorrector(args.alpha).fit(X[tr], A[tr], B[tr], log=lambda *a: None)
        pa, pb = c.predict(X[f])
        a_hat[f], b_hat[f] = pa, pb
        a_glb[f], b_glb[f] = A[tr].mean(), B[tr].mean()   # the "one line for everyone" control

    r2 = lambda t, p: 1.0 - ((t - p) ** 2).sum() / max(((t - t.mean()) ** 2).sum(), 1e-12)
    print(f"  out-of-fold R2:  offset {r2(A, a_hat):+6.3f}   slope {r2(B, b_hat):+6.3f}")

    def pooled(mode):
        se = n = 0.0
        med = []
        for i, nm in enumerate(keep):
            md, pred, truth = per[nm]
            dm = md - md[0]
            if mode == "base":
                p = pred
            elif mode == "oracle":
                p = pred - (A[i] + B[i] * dm)
            elif mode == "global":
                p = pred - (a_glb[i] + b_glb[i] * dm)
            else:
                p = pred - (a_hat[i] + b_hat[i] * dm)
            e = p - truth
            se += (e ** 2).sum(); n += len(e)
            med.append(np.sqrt((e ** 2).mean()))
        return float(np.sqrt(se / n)), float(np.median(med))

    rows = [(k, *pooled(k)) for k in ("base", "global", "predicted", "oracle")]
    print(f"\n  {'':10s} {'pooled':>8s} {'median well':>12s}")
    for k, p, m in rows:
        label = {"base": "baseline", "global": "one global line", "predicted": "PREDICTED per-well",
                 "oracle": "oracle (cheats)"}[k]
        print(f"  {label:20s} {p:8.4f} {m:12.3f}")
    base, glob, pred_, orc = [r[1] for r in rows]
    print(f"\n  per-well correction vs baseline      {pred_ - base:+.4f} ft")
    print(f"  per-well correction vs global line   {pred_ - glob:+.4f} ft   <- must be negative "
          f"to have learned anything beyond the average")
    print(f"  fraction of the oracle gain captured  {(base - pred_) / max(base - orc, 1e-9):6.1%}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            dict(preds=str(args.preds), r2_offset=r2(A, a_hat), r2_slope=r2(B, b_hat),
                 pooled={k: v for k, v, _ in rows}), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds", required=True)
    p.add_argument("--data", default=str(HERE / "data"))
    p.add_argument("--cache", default=str(HERE / "cache/wells.npz"))
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    main(p.parse_args())
