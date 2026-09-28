#!/usr/bin/env python3
"""Continuity decode over the ConvNeXt's per-column depth posteriors (a negative result).

Idea: the reported mean sits between parallel bands of the posterior; the true path is
continuous from the known anchor, so a forward-backward pass that forbids large jumps
between neighbouring MD columns should pick the right band.

    python posterior_fb_decode.py results/baseline_check_posterior

Reads the focus wells saved by seq_NN_posterior_dump.py. Nothing is tuned on the
holdout: the transition is set from TRAINING wells, where the TVT change per 32 ft
column has std 0.78 ft, q99 2.0 ft and max 8.6 ft (16 training wells measured).

Measured 2026-09-28 on results/baseline_check (60 epochs, raw 5.4736), 16 focus wells:
pooled 11.32 (mean) -> 12.14 (smoothed), 11.62 (forward only); 4 of 6 typical wells got
worse. It wins big where it locks onto the right band (f6d009f4 10.4 -> 7.1) and loses
where it locks onto a wrong one -- the same reason arg-max decoding loses to the mean.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STEP = 0.5


def kernel(sig1=1.0, sig2=6.0, w2=0.03, half_ft=40.0):
    d = np.arange(-half_ft, half_ft + STEP / 2, STEP)
    k = (1 - w2) * np.exp(-0.5 * (d / sig1) ** 2) / sig1 + w2 * np.exp(-0.5 * (d / sig2) ** 2) / sig2
    return k / k.sum() + 1e-9


def forward_backward(p, x, k=None, start_sig=2.0):
    k = kernel() if k is None else k
    e = np.clip(p.astype(np.float64), 1e-12, None)
    a = np.empty_like(e)
    a[0] = e[0] * np.exp(-0.5 * (x / start_sig) ** 2)
    a[0] /= a[0].sum()
    for t in range(1, len(e)):
        a[t] = e[t] * np.convolve(a[t - 1], k, mode="same")
        a[t] /= a[t].sum()
    b = np.ones_like(e)
    for t in range(len(e) - 2, -1, -1):
        b[t] = np.convolve(e[t + 1] * b[t + 1], k, mode="same")
        b[t] /= b[t].sum()
    g = a * b
    g /= g.sum(axis=1, keepdims=True)
    return (g * x).sum(1), (a * x).sum(1)


def main(run):
    run = Path(run)
    z = np.load(run / "posterior_focus.npz")
    x = z["levels"].astype(float)
    cols = pd.read_parquet(run / "posterior_columns.parquet")
    wells = pd.read_csv(run / "posterior_wells.csv").set_index("well_id")
    rows = []
    for wid in [str(w) for w in z["focus"]]:
        if f"{wid}__p" not in z.files:
            continue
        bins = z[f"{wid}__bins"].astype(int)
        c = cols[cols.well_id == wid].set_index("bin").loc[bins]
        n, y, mu = (c[k].to_numpy(float) for k in ("n_rows", "y", "mu"))
        within = n.sum() * wells.loc[wid, "rmse_mean"] ** 2 - (n * (mu - y) ** 2).sum()
        sm, fw = forward_backward(z[f"{wid}__p"].astype(float), x)
        sse = {k: (n * (v - y) ** 2).sum() + within for k, v in (("mean", mu), ("smoothed", sm), ("forward", fw))}
        rows.append({"well_id": wid, "rows": int(n.sum()), **{f"rmse_{k}": np.sqrt(v / n.sum()) for k, v in sse.items()},
                     **{f"sse_{k}": v for k, v in sse.items()}})
    df = pd.DataFrame(rows)
    N = df.rows.sum()
    out = {k: float(np.sqrt(df[f"sse_{k}"].sum() / N)) for k in ("mean", "smoothed", "forward")}
    print(df[["well_id", "rows", "rmse_mean", "rmse_smoothed", "rmse_forward"]].round(2).to_string(index=False))
    print("pooled over focus wells:", {k: round(v, 3) for k, v in out.items()})
    df.to_csv(run / "fb_decode_focus.csv", index=False)
    (run / "fb_decode_focus.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
