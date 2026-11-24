"""Shared loaders for the CPU probes on a posterior dump (see analysis/README.md).

PROBE_RUN selects the dump directory, relative to kaggle_1st_place/solution
(default results/baseline_check_posterior_all, made by seq_NN_posterior_dump.py --save-all).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

SOL = Path(__file__).resolve().parents[1]   # kaggle_1st_place/solution
RUN = SOL / os.environ.get("PROBE_RUN", "results/baseline_check_posterior_all")
TRAIN = SOL / "data/train"
PREFIX_LEN, DS, TARGET_LEN = 1024, 32, 10000   # cfg.prefix_len, cfg.downsample, cfg.target_len


def load_csv(wid):
    return pd.read_csv(TRAIN / f"{wid}__horizontal_well.csv", usecols=["MD", "Z", "TVT", "GR", "TVT_input"])


def columns(df):
    """Suffix rows -> column bins, exactly as seq_NN_posterior_dump._row_bins."""
    suf = np.flatnonzero(df.TVT_input.isna().to_numpy())
    n = len(suf)
    kept = min(n, TARGET_LEN)
    rb = (PREFIX_LEN + np.minimum(np.arange(n), max(kept - 1, 0))) // DS
    return suf, rb


def split():
    s = pd.read_csv(RUN / "holdout_split.csv")
    return s[s.split == "train"].well_id.tolist(), s[s.split != "train"].well_id.tolist()
