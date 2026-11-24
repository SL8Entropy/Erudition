#!/usr/bin/env python3
"""Experiment A: does the XY geo prior point at the right hump when the posterior is multimodal?

    python posterior_geo_pick.py results/baseline_check_posterior_all

Reads a dump made by seq_NN_posterior_dump.py --save-all (posterior_all.npz,
posterior_columns.parquet, posterior_wells.csv). numpy + pandas only, no GPU.

The geo prior (`__geo`, TVT interpolated from neighbouring wells' XY, relative to the
anchor) is available at test time, and the network already sees it as the
`geo_tvt_diff` input channel. The question is whether the softmax under-uses it.

Part 1, diagnostic (uses truth, not a decoder): on multimodal columns whose truth sits
under some significant hump, how often is the hump nearest the geo prior the truth hump,
versus the dominant (largest-mass) hump, versus chance (1 / n_sig)?

Part 2, non-oracle decoders, all column-level (every row in a 32 ft column shares one
decode, so the within-column part of the SSE is decode-independent and carried over from
the mean decode, as in posterior_fb_decode.py):

  geo_hump        multimodal column -> centroid of the sig hump nearest the geo prior,
                  else the mean. No knob.
  geo_hump_gated  as above, but only when that hump holds >= tau of the mass
                  (tau chosen by split-half).
  geo_reweight    p'(x) ~ p(x) * N(x; geo, sigma), decode the mean of p'; sigma chosen
                  by split-half. Soft: it keeps hedging. Applied to every column.
  geo_reweight_mm the same, applied only to multimodal columns.

Knobs are never picked on the wells they are scored on: wells are split into two random
halves, the knob is chosen on one half and scored on the other (both directions, so every
well gets a held-out score), and this is repeated over SEEDS random splits. Each knob grid
includes "off" (tau = inf, sigma = inf), so the split-half is free to reject the geo prior.

Measured 2026-09-28 on results/baseline_check (60 epochs, raw 5.4736), all 155 wells --
a clean negative:
  * multimodal rows 15.7%, truth under some hump on 93.6% of them. The truth hump is the
    geo-nearest hump 52.4% of the time, the dominant hump 76.2%, chance (1/n_sig) 46.2%.
    Where the dominant hump is wrong, geo finds the truth hump only 39.8%.
  * on multimodal columns geo is on the truth's side of the mean 49.6% (a coin flip) and
    closer to the truth than the mean 33%; its |error| q50/q90 is 9.1/36.4 ft there vs
    5.2/17.3 ft overall -- it is worst exactly where the bands repeat (25-35 ft apart).
  * every decoder gets monotonically worse the more it trusts geo, even in-sample:
    hard pick 7.93, best gated 5.68, reweight sigma=80 ft 5.52 vs mean 5.474; split-half
    picks "off" (or the weakest setting) for all of them.
  So the softmax is not under-using the geo prior; on the hard columns the prior carries
  no band information. Band disambiguation has to come from new input (experiment D).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from posterior_analysis import MIN_MODE_MASS, TRUTH_BAND_FT, find_modes

HARD = ("fb03ae90", "f2d4c8c9", "f6d009f4", "e7818f7a", "fb0904bd", "f8afa78a",
        "fdfd57da", "d7eb0be8", "f6bc699b")
TAUS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, math.inf)  # inf = never override
SIGMAS = (2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0, 80.0, 160.0, math.inf)  # inf = plain mean
SEEDS = 50


def column_table(z, cols: pd.DataFrame) -> pd.DataFrame:
    """One row per (well, column): truth, mean, geo, hump picks, and reweighted means."""
    x = z["levels"].astype(np.float64)
    out = []
    for wid in (str(w) for w in z["focus"]):
        P = z[f"{wid}__p"].astype(np.float64)
        P /= P.sum(1, keepdims=True)
        bins = z[f"{wid}__bins"].astype(int)
        geo = z[f"{wid}__geo"].astype(np.float64)
        c = cols[cols.well_id == wid].set_index("bin").loc[bins]
        y, n, mu = (c[k].to_numpy(float) for k in ("y", "n_rows", "mu"))
        rw = {}
        for s in SIGMAS:
            q = P * np.exp(-0.5 * ((x[None, :] - geo[:, None]) / s) ** 2)
            q /= np.clip(q.sum(1, keepdims=True), 1e-300, None)
            rw[s] = (q * x).sum(1)
        for j in range(len(bins)):
            modes = find_modes(P[j], x)
            sig = [m for m in modes if m["mass"] >= MIN_MODE_MASS] or modes[:1]
            near = min(range(len(sig)), key=lambda r: abs(sig[r]["centroid"] - geo[j]))
            truth = -1
            for r, m in enumerate(sig):
                if m["core_lo"] - TRUTH_BAND_FT <= y[j] <= m["core_hi"] + TRUTH_BAND_FT:
                    truth = r
                    break
            row = {"well_id": wid, "bin": int(bins[j]), "n": n[j], "y": y[j], "mu": mu[j],
                   "geo": geo[j], "n_sig": len(sig), "truth_rank": truth, "geo_rank": near,
                   "geo_hump": sig[near]["centroid"], "geo_hump_mass": sig[near]["mass"]}
            row.update({f"rw_{s:g}": rw[s][j] for s in SIGMAS})
            out.append(row)
    return pd.DataFrame(out)


def decode(df: pd.DataFrame, kind: str, knob: float) -> np.ndarray:
    mm = df.n_sig.to_numpy() >= 2
    mu = df.mu.to_numpy()
    if kind == "geo_hump":
        return np.where(mm, df.geo_hump, mu)
    if kind == "geo_hump_gated":
        return np.where(mm & (df.geo_hump_mass.to_numpy() >= knob), df.geo_hump, mu)
    if kind == "geo_reweight":
        return df[f"rw_{knob:g}"].to_numpy()
    if kind == "geo_reweight_mm":
        return np.where(mm, df[f"rw_{knob:g}"], mu)
    if kind == "mean":
        return mu
    raise ValueError(kind)


def well_sse(df: pd.DataFrame, pred: np.ndarray, within: pd.Series) -> pd.Series:
    col = df.n.to_numpy() * (pred - df.y.to_numpy()) ** 2
    return pd.Series(col, index=df.index).groupby(df.well_id).sum() + within


KNOBS = {"geo_hump": (0.0,), "geo_hump_gated": TAUS, "geo_reweight": SIGMAS, "geo_reweight_mm": SIGMAS}


def split_half(df, within, rows, kind, seeds=SEEDS):
    """Held-out pooled RMSE over all wells, knob picked on the other half; per seed."""
    per_knob = {k: well_sse(df, decode(df, kind, k), within) for k in KNOBS[kind]}
    wells = np.array(sorted(rows.index))
    scores, picks = [], []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        perm = rng.permutation(wells)
        halves = (perm[: len(perm) // 2], perm[len(perm) // 2:])
        sse = 0.0
        for fit, test in ((halves[0], halves[1]), (halves[1], halves[0])):
            best = min(KNOBS[kind], key=lambda k: per_knob[k].loc[fit].sum())
            picks.append(best)
            sse += per_knob[best].loc[test].sum()
        scores.append(math.sqrt(sse / rows.sum()))
    return np.array(scores), picks, per_knob


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", type=Path)
    args = ap.parse_args(argv)
    run = args.run
    z = np.load(run / "posterior_all.npz")
    cols = pd.read_parquet(run / "posterior_columns.parquet")
    wells = pd.read_csv(run / "posterior_wells.csv").set_index("well_id")
    df = column_table(z, cols)
    rows = wells.rows
    within = rows * wells.rmse_mean ** 2 - (df.n * (df.mu - df.y) ** 2).groupby(df.well_id).sum()
    N = rows.sum()
    rep = {"wells": int(len(rows)), "rows": int(N)}

    # ---- Part 1: diagnostic ----
    mm = df[df.n_sig >= 2]
    under = mm[mm.truth_rank >= 0]
    w = under.n
    rep["diagnostic"] = {
        "multimodal_row_share": float(mm.n.sum() / N),
        "multimodal_truth_under_a_hump_row_share": float(w.sum() / mm.n.sum()),
        "geo_hump_is_truth_hump": float((w * (under.geo_rank == under.truth_rank)).sum() / w.sum()),
        "dominant_hump_is_truth_hump": float((w * (under.truth_rank == 0)).sum() / w.sum()),
        "chance_1_over_nsig": float((w / under.n_sig).sum() / w.sum()),
        "geo_abs_err_ft_on_multimodal_q50_q90": [float(v) for v in np.percentile(np.abs(mm.geo - mm.y), [50, 90])],
        "geo_abs_err_ft_all_q50_q90": [float(v) for v in np.percentile(np.abs(df.geo - df.y), [50, 90])],
    }
    # when the dominant hump is wrong (wrong_mode columns), does geo rescue it?
    wm = under[under.truth_rank > 0]
    rep["diagnostic"]["on_truth_not_dominant"] = {
        "row_share_of_under": float(wm.n.sum() / w.sum()),
        "geo_hump_is_truth_hump": float((wm.n * (wm.geo_rank == wm.truth_rank)).sum() / wm.n.sum()),
    }
    dm = under[under.truth_rank == 0]
    rep["diagnostic"]["on_truth_is_dominant"] = {
        "geo_hump_is_truth_hump": float((dm.n * (dm.geo_rank == 0)).sum() / dm.n.sum()),
    }

    # ---- Part 2: decoders ----
    base = well_sse(df, decode(df, "mean", 0), within)
    rep["pooled_mean"] = float(math.sqrt(base.sum() / N))
    rep["decoders"] = {}
    well_rmse = {"mean": np.sqrt(base / rows)}
    for kind in KNOBS:
        scores, picks, per_knob = split_half(df, within, rows, kind)
        # in-sample knob sweep, for context only (tuned on the test wells; not a result)
        sweep = {f"{k:g}": float(math.sqrt(v.sum() / N)) for k, v in per_knob.items()}
        # per-well view uses the most-often-picked knob
        k_mode = max(set(picks), key=picks.count)
        sse_k = per_knob[k_mode]
        wr = np.sqrt(sse_k / rows)
        well_rmse[kind] = wr
        diff = wr - well_rmse["mean"]
        rep["decoders"][kind] = {
            "heldout_pooled_rmse_mean_over_splits": float(scores.mean()),
            "heldout_pooled_rmse_sd_over_splits": float(scores.std()),
            "knob_picks": {f"{k:g}": picks.count(k) for k in sorted(set(picks))},
            "modal_knob": k_mode,
            "wins_at_modal_knob": int((diff < -1e-3).sum()),
            "losses_at_modal_knob": int((diff > 1e-3).sum()),
            "wins_by_0.5ft": int((diff < -0.5).sum()),
            "losses_by_0.5ft": int((diff > 0.5).sum()),
            "in_sample_sweep_NOT_A_RESULT": sweep,
        }

    wr = pd.DataFrame(well_rmse)
    wr["share_of_sse"] = base / base.sum()
    wr["multimodal_row_share"] = wells.multimodal_row_share
    wr.round(4).to_csv(run / "geo_pick_wells.csv")
    df.to_parquet(run / "geo_pick_columns.parquet")
    rep["hard_wells"] = wr.loc[[h for h in HARD if h in wr.index]].round(2).to_dict(orient="index")
    (run / "geo_pick_report.json").write_text(json.dumps(rep, indent=2))

    d = rep["diagnostic"]
    print(f"multimodal rows {d['multimodal_row_share']:.1%}; truth under a hump on "
          f"{d['multimodal_truth_under_a_hump_row_share']:.1%} of them")
    print(f"  truth hump == geo-nearest {d['geo_hump_is_truth_hump']:.1%} | == dominant "
          f"{d['dominant_hump_is_truth_hump']:.1%} | chance {d['chance_1_over_nsig']:.1%}")
    print(f"  when dominant is wrong: geo picks truth {d['on_truth_not_dominant']['geo_hump_is_truth_hump']:.1%}; "
          f"when dominant is right: geo agrees {d['on_truth_is_dominant']['geo_hump_is_truth_hump']:.1%}")
    print(f"pooled mean decode {rep['pooled_mean']:.3f}")
    for k, v in rep["decoders"].items():
        print(f"  {k:16s} held-out {v['heldout_pooled_rmse_mean_over_splits']:.3f} "
              f"+-{v['heldout_pooled_rmse_sd_over_splits']:.3f}  knob {v['modal_knob']:g}  "
              f"wins {v['wins_at_modal_knob']}/{rep['wells']} (>0.5 ft: {v['wins_by_0.5ft']} win, "
              f"{v['losses_by_0.5ft']} lose)")
    print(wr.loc[[h for h in HARD if h in wr.index]].round(2).to_string())


if __name__ == "__main__":
    main()
