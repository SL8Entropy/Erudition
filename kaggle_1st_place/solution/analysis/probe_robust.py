"""Robustness of the dip decode against the mean decode, for every probe_dipdecode_sse_*.csv
in this folder: cross-fitted RMSE (tau etc. chosen on one random half of the wells, scored on
the other, 50 splits), and k*, bootstrap, wins and median-well error for fixed settings.

    python analysis/probe_robust.py
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def kstar(g):
    order = np.argsort(g)
    for j in range(len(g) + 1):
        keep = np.ones(len(g), bool)
        keep[order[:j]] = False
        if g[keep].sum() >= 0:
            return j
    return len(g)


for f in sorted(glob.glob(str(Path(__file__).parent / "probe_dipdecode_sse_*.csv"))):
    d = pd.read_csv(f)
    N = d.rows.sum()
    rm = lambda k: np.sqrt(d[k].sum() / N)  # noqa: E731
    med = lambda k: np.median(np.sqrt(d[k] / d.rows))  # noqa: E731
    so = [c for c in d.columns if c.startswith("so_")]
    rng = np.random.default_rng(0)
    cf = []
    for _ in range(50):
        m = rng.permutation(len(d)) < len(d) // 2
        sse = 0.0
        for A, B in ((d[m], d[~m]), (d[~m], d[m])):
            sse += B[min(so, key=lambda k: A[k].sum())].sum()
        cf.append(np.sqrt(sse / N))
    name = Path(f).stem[len("probe_dipdecode_sse_"):]
    print(f"\n### {name}   mean {rm('mean'):.3f} | median well {med('mean'):.3f} | "
          f"first-order raw {rm('fo_t1.0'):.3f} | cross-fit 2nd-order {np.mean(cf):.3f} +- {np.std(cf):.3f}")
    for tau in (0.05, 0.1, 0.2, 0.3):
        k = f"so_t{tau}_e0.15_s0.2"
        g = (d[k] - d["mean"]).to_numpy()
        bs = np.mean([g[rng.integers(0, len(g), len(g))].sum() < 0 for _ in range(2000)])
        print(f"  tau={tau:<4} pooled {rm(k):.3f}  k*={kstar(g):<3} boot {bs:.0%}  wins {(g < 0).sum()}/{len(g)}  "
              f"median well {med(k):.3f}  (biggest gain: {d.loc[np.argmin(g), 'well_id']})")
