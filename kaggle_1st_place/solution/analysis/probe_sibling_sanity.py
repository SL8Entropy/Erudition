"""Sanity check for probe_sibling.py: is the GR matcher working, and does the sibling reference make
the true path fit better than the wrong hump's path?

    python analysis/probe_sibling_sanity.py

Result (2026-09-29): on single-hump columns the true path beats a +-10 ft shift 69% (typewell) /
67% (siblings) of the time; on multimodal columns the true path beats the wrong hump's path 60.5%
(typewell) vs 58% (siblings).  Siblings add no band information for this model.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from probe_common import RUN, columns, load_csv  # noqa: E402
from probe_sibling import SiblingBank, affine, load_all  # noqa: E402


def main():
    np.random.seed(0)
    wells, tr, ho = load_all()
    bank = SiblingBank().fit(wells, tr, all_names=tr + ho, log=lambda *a: None)
    cols = pd.read_parquet(RUN / "posterior_columns.parquet")
    W = 2
    res = {"uni": [], "mm": []}
    for wid in ho:
        df = load_csv(wid)
        suf, rb = columns(df)
        tvt0 = df.TVT_input.dropna().iloc[-1]
        y_row = df.TVT.to_numpy(float)[suf] - tvt0
        c = cols[cols.well_id == wid].set_index("bin")
        g_row = df.GR.to_numpy(float)[suf]
        kn = df[df.TVT_input.notna()].tail(1500)
        tw = wells[wid]
        refs = {"tw": lambda q, tw=tw: np.interp(q, tw["tw_tvt"], tw["tw_gr"], left=np.nan, right=np.nan),
                "sib": lambda q, wid=wid: bank.profile(wid, q)[0]}
        cal = {r: affine(f(kn.TVT_input.to_numpy(float)), kn.GR.to_numpy(float)) for r, f in refs.items()}

        def mis(r, q, g):
            ref = refs[r](q)
            if cal[r] is not None:
                ref = cal[r][0] * ref + cal[r][1]
            ok = np.isfinite(ref) & np.isfinite(g)
            return np.median(np.abs(g[ok] - ref[ok])) if ok.sum() >= 16 else np.nan

        for b, row in c.iterrows():
            sel = (rb >= b - W) & (rb <= b + W)
            g = g_row[sel]
            truth = tvt0 + y_row[sel]
            if row.n_sig == 1 and np.random.rand() < 0.15:
                for r in refs:
                    res["uni"].append((r, mis(r, truth, g) < min(mis(r, truth + d, g) for d in (-10, 10))))
            if row.n_sig >= 2 and row.truth_rank >= 0:
                wrong = [row[f"m{k}_centroid"] for k in (1, 2, 3)
                         if k - 1 != row.truth_rank and row[f"m{k}_mass"] >= 0.05]
                for r in refs:
                    m0 = mis(r, truth, g)
                    for cw in wrong:
                        res["mm"].append((r, m0 < mis(r, truth + (cw - row.y), g)))
    for k, v in res.items():
        d = pd.DataFrame(v, columns=["ref", "truth_better"]).groupby("ref").truth_better.mean()
        print(k, "truth path fits better than alternative:", d.round(3).to_dict())


if __name__ == "__main__":
    main()
