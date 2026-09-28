#!/usr/bin/env python3
"""Plot the depth posteriors saved by seq_NN_posterior_dump.py.

    python plot_posteriors.py results/baseline_check_posterior

Writes, next to the dump:
    posterior_<well>.png   heat map of P(depth | MD column) with the truth, the reported
                           mean, the dominant hump and the geo prior drawn over it
    posterior_summary.png  where the squared error comes from, by failure type

Needs matplotlib, which the training requirements do not include.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    argv = sys.argv[1:] if argv is None else argv
    run = Path(argv[0])
    z = np.load(run / "posterior_focus.npz")
    rep = json.loads((run / "posterior_report.json").read_text(encoding="utf-8"))
    cols = pd.read_parquet(run / "posterior_columns.parquet")
    wells = pd.read_csv(run / "posterior_wells.csv").set_index("well_id")
    x = z["levels"].astype(float)
    hard = {r["well_id"] for r in rep["focus"]}

    for wid in [str(w) for w in z["focus"]]:
        if f"{wid}__p" not in z.files:
            continue
        p = z[f"{wid}__p"].astype(float)
        bins = z[f"{wid}__bins"].astype(int)
        y = z[f"{wid}__y"].astype(float)
        geo = z[f"{wid}__geo"].astype(float) if f"{wid}__geo" in z.files else None
        c = cols[cols.well_id == wid].set_index("bin").loc[bins]
        md = (bins - bins[0]) * 32.0  # ft after the prediction start

        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(11, 6.2), sharex=True, gridspec_kw={"height_ratios": [4, 1]}
        )
        pk = p / np.maximum(p.max(axis=1, keepdims=True), 1e-12)  # per-column normalised, so
        ax.imshow(                                                # a weak second hump is visible
            pk.T, origin="lower", aspect="auto", cmap="magma",
            extent=[md[0] - 16, md[-1] + 16, x[0] - 0.25, x[-1] + 0.25],
        )
        ax.plot(md, y, color="#39d98a", lw=1.8, label="truth")
        ax.plot(md, c["mu"], color="#4fc3f7", lw=1.4, label="reported (posterior mean)")
        ax.plot(md, c["m1_centroid"], color="white", lw=0.9, ls="--", label="dominant hump")
        if geo is not None:
            ax.plot(md, geo, color="#bdbdbd", lw=0.9, ls=":", label="geo prior")
        lo = np.nanmin([y.min(), c["mu"].min(), x[0]])
        hi = np.nanmax([y.max(), c["mu"].max(), x[-1]])
        ax.set_ylim(max(lo, x[0]), min(hi, x[-1]))
        ax.set_ylabel("TVT relative to anchor (ft)")
        r = wells.loc[wid]
        tag = "  [hard well]" if wid in hard else ""
        ax.set_title(
            f"{wid}{tag}   RMSE {r.rmse_mean:.2f} ft   argmax {r.rmse_argmax:.2f}   "
            f"best-hump oracle {r.rmse_oracle_mean_or_mode:.2f}   minus line {r.rmse_minus_line:.2f}",
            fontsize=10,
        )
        ax.legend(loc="upper left", fontsize=8, framealpha=0.6)
        ax2.plot(md, c["truth_mass"], color="#39d98a", lw=1.2, label="P(truth +-2 ft)")
        ax2.step(md, c["n_sig"] / 4.0, where="mid", color="#ff8a65", lw=1.0, label="humps / 4")
        ax2.set_ylim(0, 1.05)
        ax2.set_xlabel("MD after prediction start (ft)")
        ax2.legend(loc="upper right", fontsize=8, framealpha=0.6)
        fig.tight_layout()
        fig.savefig(run / f"posterior_{wid}.png", dpi=110)
        plt.close(fig)

    shares = rep["category_share_of_sse"]
    order = [k for k in ("unimodal_miss", "unimodal_offset", "blend", "wrong_mode",
                         "multimodal_miss", "out_of_window", "ok")]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(order[::-1], [100 * shares[k] for k in order[::-1]], color="#4fc3f7")
    for i, k in enumerate(order[::-1]):
        ax.text(100 * shares[k] + 0.5, i, f"{100 * shares[k]:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("share of pooled squared error (%)")
    pr = rep["pooled_rmse"]
    ax.set_title(
        f"mean {pr['mean']:.3f}  argmax {pr['argmax']:.3f}  dominant hump {pr['dominant_mode']:.3f}  "
        f"best-hump oracle {pr['oracle_mean_or_mode']:.3f}  minus line {pr['minus_line']:.3f}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(run / "posterior_summary.png", dpi=110)
    plt.close(fig)
    print(f"wrote plots to {run}")


if __name__ == "__main__":
    main()
