"""Does a sibling-lateral GR reference pick the true hump on multimodal columns more often than
the model's dominant hump?  Numpy/pandas only, all 155 holdout wells.  Sibling profiles come only
from the 618 training wells (kaggle2ndplace/anchor_sibling.SiblingBank: a well is never its own
sibling).

    python analysis/probe_sibling.py

Result (2026-09-29, 60-epoch baseline dump): GR matching picks the true hump ~50% of the time
(chance 46.5%, dominant hump 76.2%); the sibling profile is not more discriminative than the
typewell (see probe_sibling_sanity.py).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from probe_common import RUN, SOL, TRAIN, columns, load_csv, split  # noqa: E402

sys.path.insert(0, str(SOL.parents[1] / "kaggle2ndplace"))
from anchor_sibling import SiblingBank  # noqa: E402

WINDOWS = (2, 6)
REFS = ("tw", "sib", "blend")


def load_all():
    tr, ho = split()
    wells = {}
    for wid in tr + ho:
        t = pd.read_csv(TRAIN / f"{wid}__typewell.csv", usecols=["TVT", "GR"]).dropna()
        w = {"tw_tvt": t.TVT.to_numpy(float), "tw_gr": t.GR.to_numpy(float)}
        if wid in tr:
            h = pd.read_csv(TRAIN / f"{wid}__horizontal_well.csv", usecols=["TVT", "GR"])
            w["tvt"], w["gr"] = h.TVT.to_numpy(float), h.GR.to_numpy(float)
        wells[wid] = w
    return wells, tr, ho


def affine(ref_vals, gr):
    ok = np.isfinite(ref_vals) & np.isfinite(gr)
    if ok.sum() < 50 or np.std(ref_vals[ok]) < 1e-6:
        return None
    a, b = np.polyfit(ref_vals[ok], gr[ok], 1)
    return a, b


def main():
    wells, tr, ho = load_all()
    bank = SiblingBank().fit(wells, tr, all_names=tr + ho)
    cols = pd.read_parquet(RUN / "posterior_columns.parquet")
    recs, cover = [], []
    for wid in ho:
        df = load_csv(wid)
        suf, rb = columns(df)
        tvt0 = df.TVT_input.dropna().iloc[-1]
        c = cols[cols.well_id == wid].set_index("bin")
        mu_row = c.loc[rb, "mu"].to_numpy(float)
        g_row = df.GR.to_numpy(float)[suf]
        kn = df[df.TVT_input.notna()].tail(1500)
        tw = wells[wid]
        ref_fn = {"tw": lambda q, tw=tw: np.interp(q, tw["tw_tvt"], tw["tw_gr"], left=np.nan, right=np.nan)}
        if bank.n_siblings(wid) > 0:
            ref_fn["sib"] = lambda q, wid=wid: bank.profile(wid, q)[0]
        else:
            ref_fn["sib"] = lambda q: np.full(len(q), np.nan)

        def blend(q, f=ref_fn):
            a, s = f["tw"](q), f["sib"](q)
            return np.where(np.isfinite(s), 0.2 * a + 0.8 * s, a)   # 3rd-place weights
        ref_fn["blend"] = blend
        cal = {r: affine(ref_fn[r](kn.TVT_input.to_numpy(float)), kn.GR.to_numpy(float)) for r in REFS}
        cover.append({"well_id": wid, "n_sib": bank.n_siblings(wid),
                      "sib_cover": np.isfinite(ref_fn["sib"](tvt0 + mu_row)).mean()})
        for b, row in c[c.n_sig >= 2].iterrows():
            humps = [(row[f"m{k}_centroid"], row[f"m{k}_mass"]) for k in (1, 2, 3)
                     if row[f"m{k}_mass"] >= 0.05 and np.isfinite(row[f"m{k}_centroid"])]
            if len(humps) < 2:
                continue
            rec = {"well_id": wid, "bin": b, "n_rows": row.n_rows, "truth_rank": int(row.truth_rank),
                   "n_h": len(humps), "mu": row.mu, "y": row.y}
            for W in WINDOWS:
                sel = (rb >= b - W) & (rb <= b + W)
                g = g_row[sel]
                base = tvt0 + mu_row[sel]
                scale = np.nanstd(g) + 1e-6
                for r in REFS:
                    for k, (ch, _) in enumerate(humps):
                        ref = ref_fn[r](base + (ch - row.mu))
                        if cal[r] is not None:
                            ref = cal[r][0] * ref + cal[r][1]
                        ok = np.isfinite(ref) & np.isfinite(g)
                        rec[f"mis_{r}_W{W}_{k}"] = np.median(np.abs(g[ok] - ref[ok])) / scale if ok.sum() >= 16 else np.nan
            recs.append(rec)
    R, C = pd.DataFrame(recs), pd.DataFrame(cover)
    print(f"sibling coverage: {int((C.n_sib > 0).sum())}/{len(C)} holdout wells have training siblings")
    T = R[R.truth_rank >= 0]
    w = T.n_rows.to_numpy(float)
    print(f"multimodal columns {len(R)}, truth under a hump {len(T)}")
    print(f"  dominant hump is truth: {(w * (T.truth_rank.to_numpy() == 0)).sum() / w.sum():.3f}   "
          f"chance {(w / T.n_h.to_numpy()).sum() / w.sum():.3f}")
    wrong = T.truth_rank.to_numpy() != 0
    for W in WINDOWS:
        for r in REFS:
            M = np.stack([T.get(f"mis_{r}_W{W}_{k}", pd.Series(np.nan, index=T.index)).to_numpy(float)
                          for k in range(3)], 1)
            M = np.where(np.arange(3)[None, :] < T.n_h.to_numpy()[:, None], M, np.nan)
            have = np.isfinite(M).sum(1) == T.n_h.to_numpy()
            pick = np.where(have, np.nanargmin(np.where(np.isfinite(M), M, np.inf), 1), -1)
            hit = pick == T.truth_rank.to_numpy()
            print(f"  W={W:<2d} {r:5s} picks truth {(w * hit)[have].sum() / w[have].sum():.3f} | "
                  f"where dominant is wrong: {(w * hit)[have & wrong].sum() / max(w[have & wrong].sum(), 1):.3f}")


if __name__ == "__main__":
    main()
