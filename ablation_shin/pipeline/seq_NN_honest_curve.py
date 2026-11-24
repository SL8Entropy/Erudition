#!/usr/bin/env python3
"""Honest scores for holdout runs, read from their training logs.

`seq_NN_holdout_eval.py` validates on the 155 holdout wells, and the training loop keeps the
checkpoint with the lowest validation RMSE (`seq_NN_train.py`: `if val_rmse < best_score:
best_state = ...`, then `model.load_state_dict(best_state)`).  So the reported pooled RMSE of
every run is scored on a checkpoint *picked using the test wells*.  Across the runs checked,
the picked checkpoint sits ~0.14-0.15 ft below the same run's late-training average.

Comparing two runs by their reported numbers is still like-for-like (both picked the same
way), but noisier and optimistic.  This script prints the unpicked alternatives, from the
`val_rmse_raw` lines every run already logs (raw = before the final smoothing step):

    picked      the minimum, i.e. what the reported number is built on
    last        the final checkpoint, which nobody chose
    late avg    the mean over epochs >= --from, a stable plateau statistic

    python seq_NN_honest_curve.py results/0801_V2_ep150 results/cnx_tiny_ep150 results/cnx_fastvit_ep150
"""

import argparse
import re
from pathlib import Path

import numpy as np

LINE = re.compile(r"fold0 epoch (\d+): .*val_rmse_raw=([0-9.]+)")


def curve(run_dir):
    ep, val = [], []
    for line in open(Path(run_dir) / "seq_nn.log", encoding="utf-8", errors="ignore"):
        m = LINE.search(line)
        if m:
            ep.append(int(m.group(1)))
            val.append(float(m.group(2)))
    return np.array(ep), np.array(val)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+")
    p.add_argument("--from", dest="start", type=int, default=100, help="first epoch of the plateau average")
    a = p.parse_args()
    print(f"{'run':28s} {'picked':>14s} {'last':>14s} {'late avg':>9s} {'sd':>6s} {'flattery':>9s}")
    for r in a.runs:
        ep, v = curve(r)
        if len(v) == 0:
            print(f"{Path(r).name:28s} no val_rmse_raw lines found")
            continue
        late = v[ep >= a.start]
        avg = late.mean() if len(late) else float("nan")
        print(f"{Path(r).name:28s} {v.min():7.4f} @{ep[v.argmin()]:4d} {v[-1]:7.4f} @{ep[-1]:4d} "
              f"{avg:9.4f} {late.std() if len(late) else float('nan'):6.3f} {avg - v.min():9.3f}")


if __name__ == "__main__":
    main()
