"""Well loading, fold assignment and the small amount of global bookkeeping
the rest of the pipeline needs.

Competition layout (both `train/` and `test/` follow it):

    <root>/train/<well_id>__horizontal_well.csv   MD,X,Y,Z,GR,TVT_input[,TVT]
    <root>/train/<well_id>__typewell.csv          TVT,GR[,geology]
    <root>/sample_submission.csv                  id,tvt   with id = "<well>_<row>"

Everything downstream speaks `Well`, so this is the only module that knows
about CSV files.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# GR is standardised with fixed constants rather than per-well statistics: the
# absolute level carries real signal (shale vs sand), and per-well scaling would
# throw away exactly the calibration difference that the GR-affine augmentation
# is meant to teach robustness to.
GR_MEAN = 80.0
GR_STD = 40.0


@dataclass
class Well:
    """One horizontal well plus the typewell it is matched against."""

    well_id: str
    md: np.ndarray          # (N,) measured depth, 1 ft steps, ascending
    x: np.ndarray           # (N,)
    y: np.ndarray           # (N,)
    z: np.ndarray           # (N,) TVD of the bit -- known after PS too
    gr: np.ndarray          # (N,) gamma ray, may contain NaN
    tvt_input: np.ndarray   # (N,) expert TVT up to PS, NaN afterwards
    tw_tvt: np.ndarray      # (M,) typewell axis, ascending
    tw_gr: np.ndarray       # (M,)
    row_idx: np.ndarray     # (N,) original CSV row number -> submission id
    tvt: np.ndarray | None = None       # (N,) ground truth, train only
    tw_key: str = ""        # identity of the typewell, for sibling grouping
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @property
    def n(self) -> int:
        return len(self.md)

    @property
    def ps_idx(self) -> int:
        """First row of the prediction region (index of the first hidden TVT)."""
        known = np.isfinite(self.tvt_input)
        if not known.any():
            return 0
        return int(np.flatnonzero(known)[-1]) + 1

    @property
    def anchor_tvt(self) -> float:
        """Last visible TVT. Every TVT in this codebase is relative to it."""
        known = np.flatnonzero(np.isfinite(self.tvt_input))
        if len(known) == 0:
            return float("nan")
        return float(self.tvt_input[known[-1]])

    @property
    def anchor_md(self) -> float:
        known = np.flatnonzero(np.isfinite(self.tvt_input))
        return float(self.md[known[-1]]) if len(known) else float(self.md[0])

    @property
    def anchor_z(self) -> float:
        known = np.flatnonzero(np.isfinite(self.tvt_input))
        return float(self.z[known[-1]]) if len(known) else float(self.z[0])

    @property
    def eval_mask(self) -> np.ndarray:
        """Rows that are scored: everything after PS."""
        return ~np.isfinite(self.tvt_input)

    def z_layer(self) -> np.ndarray | None:
        """S = TVT + Z, the structural surface (2nd place calls it `z_layer`).

        The master identity dTVT = dz_layer - dz follows from this definition,
        and every geometry-preserving augmentation is written in these terms.
        """
        if self.tvt is None:
            return None
        return self.tvt + self.z

    def copy(self) -> "Well":
        return Well(
            well_id=self.well_id,
            md=self.md.copy(), x=self.x.copy(), y=self.y.copy(), z=self.z.copy(),
            gr=self.gr.copy(), tvt_input=self.tvt_input.copy(),
            tw_tvt=self.tw_tvt.copy(), tw_gr=self.tw_gr.copy(),
            row_idx=self.row_idx.copy(),
            tvt=None if self.tvt is None else self.tvt.copy(),
            tw_key=self.tw_key, meta=dict(self.meta),
        )


# ---------------------------------------------------------------------- #
# loading
# ---------------------------------------------------------------------- #
_NULLS = ["nan", "NaN", "NA", "null", ""]


def _typewell_key(tw_tvt: np.ndarray, tw_gr: np.ndarray) -> str:
    """Stable identity for a typewell so laterals sharing one can be grouped.

    Typewell files are duplicated per lateral, so hashing the curve itself is
    more reliable than any id column. Values are rounded before hashing so a
    float round-trip through CSV does not split a group.
    """
    a = np.round(np.asarray(tw_tvt, dtype=np.float64), 3)
    b = np.round(np.nan_to_num(np.asarray(tw_gr, dtype=np.float64), nan=-999.0), 3)
    h = hashlib.blake2b(a.tobytes() + b.tobytes(), digest_size=8)
    return h.hexdigest()


def list_well_ids(split_dir) -> list[str]:
    ids = sorted(p.name.replace("__horizontal_well.csv", "")
                 for p in Path(split_dir).glob("*__horizontal_well.csv"))
    if not ids:
        raise FileNotFoundError(f"no *__horizontal_well.csv under {split_dir}")
    return ids


def load_well(split_dir, well_id: str, require_tvt: bool = False):
    """Read one well. Returns None if it is unusable (no anchor, no typewell)."""
    split_dir = Path(split_dir)
    hw = pd.read_csv(split_dir / f"{well_id}__horizontal_well.csv", na_values=_NULLS)
    hw = hw.reset_index(names="_row_idx").sort_values("MD", kind="stable")

    tw_path = split_dir / f"{well_id}__typewell.csv"
    if not tw_path.exists():
        return None
    tw = pd.read_csv(tw_path, na_values=_NULLS)
    tw = tw.dropna(subset=["TVT", "GR"]).sort_values("TVT", kind="stable")
    if len(tw) < 8:
        return None

    def col(c):
        return hw[c].to_numpy(dtype=np.float64)

    tvt = col("TVT") if "TVT" in hw.columns else None
    if require_tvt and (tvt is None or not np.isfinite(tvt).any()):
        return None

    tw_tvt = tw["TVT"].to_numpy(dtype=np.float64)
    tw_gr = tw["GR"].to_numpy(dtype=np.float64)

    w = Well(
        well_id=well_id,
        md=col("MD"), x=col("X"), y=col("Y"), z=col("Z"), gr=col("GR"),
        tvt_input=col("TVT_input"),
        tw_tvt=tw_tvt, tw_gr=tw_gr,
        row_idx=hw["_row_idx"].to_numpy(dtype=np.int64),
        tvt=tvt,
        tw_key=_typewell_key(tw_tvt, tw_gr),
    )
    if not np.isfinite(w.anchor_tvt):
        return None            # nothing visible -> no anchor -> constant fallback
    return w


def load_split(split_dir, well_ids=None, require_tvt: bool = False,
               verbose: bool = True) -> list:
    split_dir = Path(split_dir)
    ids = well_ids if well_ids is not None else list_well_ids(split_dir)
    wells, dropped = [], []
    for wid in ids:
        w = load_well(split_dir, wid, require_tvt=require_tvt)
        if w is not None:
            wells.append(w)
        else:
            dropped.append(wid)
    if verbose:
        msg = f"[data] {split_dir.name}: loaded {len(wells)} wells"
        if dropped:
            msg += f", dropped {len(dropped)} ({dropped[:3]}...)"
        print(msg)
    return wells


COMPETITION = "rogii-wellbore-geology-prediction"


def find_data_root(root=None) -> Path:
    """Locate the data directory holding `train/` and `test/`.

    Order: the path given, then this repo's parent (where the data sits by
    default), then a `rogii-wellbore-geology-prediction` folder beside either.
    """
    here = Path(__file__).resolve().parent.parent
    cands = []
    if root is not None:
        root = Path(root)
        cands += [root, root / COMPETITION]
    cands += [here.parent, here.parent / COMPETITION,
              here.parent.parent / COMPETITION, Path.cwd()]
    for cand in cands:
        if (cand / "train").is_dir() or (cand / "test").is_dir():
            return cand
        if cand.is_dir() and list(cand.glob("*__horizontal_well.csv")):
            return cand
    if root is not None and Path(root).is_dir():
        for sub in sorted(Path(root).iterdir()):
            if sub.is_dir() and ((sub / "test").is_dir() or (sub / "train").is_dir()):
                return sub
    raise FileNotFoundError(
        "could not find a directory containing train/ or test/ well CSVs; "
        f"tried {[str(c) for c in cands]}. Pass --data-root explicitly.")


def split_dir(root, split: str = "test") -> Path:
    """Directory holding one split's well CSVs (`<root>/<split>/` or `<root>/`)."""
    root = Path(root)
    for cand in (root / split, root):
        if cand.is_dir() and list(cand.glob("*__horizontal_well.csv")):
            return cand
    raise FileNotFoundError(f"no {split} well CSVs under {root}")


def find_sample_submission(root) -> Path:
    root = Path(root)
    for cand in (root / "sample_submission.csv",
                 root.parent / "sample_submission.csv"):
        if cand.is_file():
            return cand
    hits = sorted(root.rglob("sample_submission.csv"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"sample_submission.csv not found under {root}")


# ---------------------------------------------------------------------- #
# folds
# ---------------------------------------------------------------------- #
def group_kfold(well_ids: list[str], n_folds: int = 5, seed: int = 0) -> dict:
    """GroupKFold by well, with the well order permuted by `seed`.

    Every row of a well lands in one fold. Changing `seed` changes the fold
    *composition*, which is 3rd place's "5 split patterns": averaging over
    several assignments removes the dependence on one lucky partition.
    """
    ids = sorted(well_ids)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    return {ids[j]: int(i % n_folds) for i, j in enumerate(order)}


def initial_structural_rate(well, n_rows: int = 256) -> float:
    """Slope of the structural surface S = TVT + Z, fitted on the prefix tail.

    3rd place measured that widening this window from 30 to 256 rows was worth
    real private-LB points: every foot of the prediction inherits this number,
    so a noisy estimate taken from just before PS costs far more than it looks.
    """
    vis = np.flatnonzero(np.isfinite(well.tvt_input))
    if len(vis) < 8:
        return 0.0
    idx = vis[-min(n_rows, len(vis)):]
    s = well.tvt_input[idx] + well.z[idx]
    md = well.md[idx]
    A = np.column_stack([md - md.mean(), np.ones(len(md))])
    coef, *_ = np.linalg.lstsq(A, s, rcond=None)
    return float(coef[0])


def typewell_groups(wells: list) -> dict:
    """typewell key -> well ids sharing it (sibling laterals)."""
    groups: dict = {}
    for w in wells:
        groups.setdefault(w.tw_key, []).append(w.well_id)
    return groups


def link_typewells(wells: list, min_overlap: float = 0.5, stride: int = 13,
                   max_per_point: int = 48, verbose: bool = True) -> dict:
    """Cluster typewells into master series, and rewrite `tw_key` in place.

    Typewells are *crops of a small number of master TVT-GR series* -- 2nd place
    consolidated the training set into 54 systems. Hashing each curve exactly
    therefore splits siblings apart: on the real data it yields 752 groups for
    773 wells, which makes the sibling reference useless. Two typewells belong
    to the same system when their (TVT, GR) samples coincide on the range they
    share, so cluster by point-set overlap instead.

    Idempotent: wells already carrying a `sys...` key are left alone.
    """
    todo = [w for w in wells if not str(w.tw_key).startswith("sys")]
    if not todo:
        return typewell_groups(wells)

    pts = [set(zip(np.round(w.tw_tvt, 2), np.round(w.tw_gr, 2))) for w in wells]
    n = len(wells)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # a point shared by several typewells makes them candidates; subsampling the
    # points keeps this near-linear, and union-find supplies transitivity
    buckets: dict = {}
    for i, s in enumerate(pts):
        for j, p in enumerate(s):
            if j % stride == 0:
                buckets.setdefault(p, []).append(i)

    for lst in buckets.values():
        if len(lst) < 2:
            continue
        lst = sorted(set(lst))[:max_per_point]
        a = lst[0]
        for b in lst[1:]:
            if find(a) == find(b):
                continue
            ov = len(pts[a] & pts[b])
            if ov >= min_overlap * min(len(pts[a]), len(pts[b])):
                parent[find(b)] = find(a)

    roots = sorted({find(i) for i in range(n)})
    label = {r: f"sys{k:03d}" for k, r in enumerate(roots)}
    for i, w in enumerate(wells):
        w.tw_key = label[find(i)]
    groups = typewell_groups(wells)
    if verbose:
        sizes = sorted((len(v) for v in groups.values()), reverse=True)
        print(f"[data] typewell systems: {len(groups)} "
              f"(largest {sizes[:5]}, singletons {sum(1 for s in sizes if s == 1)})")
    return groups


# ---------------------------------------------------------------------- #
# submission helpers
# ---------------------------------------------------------------------- #
def write_submission(preds: dict, sample_csv, out_csv):
    """`preds[well_id] = (row_idx, tvt)` -> submission CSV in sample order."""
    lut = {(wid, int(r)): float(v)
           for wid, (rows, vals) in preds.items() for r, v in zip(rows, vals)}
    sub = pd.read_csv(sample_csv)
    keys = [(i.rsplit("_", 1)[0], int(i.rsplit("_", 1)[1]))
            for i in sub["id"].astype(str)]
    vals = np.array([lut.get(k, np.nan) for k in keys], dtype=np.float64)
    n_missing = int(np.isnan(vals).sum())
    if n_missing:
        raise ValueError(f"{n_missing} submission rows have no prediction")
    sub["tvt"] = vals
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub[["id", "tvt"]].to_csv(out_csv, index=False)
    print(f"[data] wrote {out_csv}  rows={len(sub)}  "
          f"range=[{vals.min():.1f}, {vals.max():.1f}]")
    return sub


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float),
                    encoding="utf-8")
