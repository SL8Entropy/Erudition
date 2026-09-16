#!/usr/bin/env python3
"""One table of every run under ``runs/``: training runs, quantization, ensembles.

    python anchor_summary.py
    python anchor_summary.py --seeds C_s1 C_s2 C_s3      # mean and spread of a seed group

Reads only the JSON each script writes, so it is safe to run while training is going.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def fmt(v, spec=".3f"):
    if v is None:
        return "-"
    if isinstance(v, bool) or isinstance(v, int):
        return str(v)
    return format(v, spec) if isinstance(v, float) else str(v)


def main(args):
    root = Path(args.runs)
    rows = []
    for f in sorted(root.glob("*/holdout_metrics.json")):
        m = json.loads(f.read_text())
        rows.append(dict(
            run=f.parent.name, arch=m.get("arch", "effnet"), row=m.get("row"), n_move=m.get("n_move"),
            seed=m.get("seed"), epoch_len=m.get("epoch_len"), prefilter=m.get("gr_prefilter_ft"),
            teachers=len(m.get("distill_teachers", []) or []) or None,
            ckpts=len(m["models"]) if "models" in m else None, params=m.get("params_m"),
            tta=m.get("tta"), pooled=m.get("pooled_rmse"), p1=m.get("pooled_rmse_1phase"),
            gain=m.get("tta_gain"), p50=m.get("well_rmse_p50")))
    if rows:
        hdr = ["run", "arch", "row", "n_move", "seed", "epoch_len", "prefilter", "teachers", "ckpts",
               "params", "tta", "pooled", "p1", "gain", "p50"]
        w = {h: max(len(h), *(len(fmt(r[h])) for r in rows)) for h in hdr}
        print("TRAINING / RESCORE RUNS   (pooled = with --tta phases, p1 = one phase, gain = p1 - pooled)")
        print("  ".join(h.rjust(w[h]) for h in hdr))
        for r in rows:
            print("  ".join(fmt(r[h]).rjust(w[h]) for h in hdr))

    if args.seeds:
        vals = [r["pooled"] for r in rows if r["run"] in args.seeds and r["pooled"] is not None]
        ones = [r["p1"] for r in rows if r["run"] in args.seeds and r["p1"] is not None]
        if vals:
            print(f"\nSEED GROUP {args.seeds}: pooled {np.mean(vals):.3f} +- {np.std(vals, ddof=1) if len(vals) > 1 else 0:.3f}"
                  f"  (min {min(vals):.3f}, max {max(vals):.3f}, n={len(vals)})"
                  + (f";  one phase {np.mean(ones):.3f}" if ones else ""))
            print("A single-seed treatment is only distinguishable from this group if it falls "
                  "outside the group's min-max range.")

    for f in sorted(root.glob("*/quantize_metrics.json")):
        m = json.loads(f.read_text())
        print(f"\nQUANTIZATION  {f.parent.name}  (keep_float={m.get('keep_float')})")
        for k in ("fp32", "ptq", "qat"):
            if k in m:
                q = m[k]
                print(f"  {k:5s} pooled {fmt(q.get('pooled_rmse'))}  one phase {fmt(q.get('pooled_rmse_1phase'))}"
                      f"  size {fmt(q.get('size_mb'), '.1f')} MB")
        for th, lat in m.get("latency_ms", {}).items():
            print(f"  {th:>2s} threads: fp32 {lat['fp32']:.0f} ms, int8 {lat['int8']:.0f} ms -> {lat['speedup']:.2f}x")

    for f in sorted(root.glob("*/ensemble_metrics.json")):
        m = json.loads(f.read_text())
        print(f"\nENSEMBLE  {f.parent.name}")
        for k in ("convnext", "anchor", "mean", "global_w", "gate", "gate_ce", "oracle_row"):
            if k in m:
                print(f"  {k:10s} pooled {m[k]['pooled_rmse']:.4f}  median well {m[k]['well_rmse_p50']:.3f}"
                      + ("   (uses the answer)" if k == "oracle_row" else ""))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default=str(HERE / "runs"))
    p.add_argument("--seeds", nargs="*", default=[])
    main(p.parse_args())
