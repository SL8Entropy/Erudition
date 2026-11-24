#!/usr/bin/env python3
"""Leave-largest-contribution-out acceptance test for a candidate change.

Pooled RMSE is proportional to a *sum* of squared errors, so a handful of
catastrophic wells move it far more than the bulk of the set does.  That makes
the headline number easy to win by luck: if a candidate happens to rescue three
bad wells, the pooled score improves whether or not the change is systematic.
The 2nd-place solution's answer, ported here, is to ask how much of the gain
survives when the wells that produced it are taken away.

For every well, take the difference of squared-error sums between the two runs

    g_w = SSE_treat(w) - SSE_base(w)

and drop wells one at a time in descending ``|g_w|`` -- largest contributors to
the difference first -- rescoring both runs on what is left after each removal.
``k*`` is the first removal count at which the improvement disappears.  A
candidate whose gain evaporates after a few wells was luck; one whose gain
survives a removal count comparable to the size of a real evaluation set is a
change worth keeping.

Both runs must have been scored on the same holdout wells.  Point it at two
directories written by ``seq_NN_holdout_eval.py`` or ``seq_NN_rescore.py``::

    python seq_NN_robust_compare.py --base results/0801_V2 --treat results/rb_v2_synth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PRED_FILENAME = "holdout_predictions.pqt"
DEFAULT_SURVIVAL_FRACTION = 52.0 / 773.0  # the bar the 2nd-place solution used
CURVE_STEM = "robust_compare_curve"
REPORT_STEM = "robust_compare"


def report_basename(base_path: Path) -> str:
    """Name the report after the baseline it was measured against.

    One candidate is usually compared to more than one baseline -- against the
    reference run and against its own no-TTA twin, say -- and those reports must
    not overwrite each other in the candidate's directory.
    """

    parent = base_path.parent.name
    return parent if parent else base_path.stem


def resolve_predictions_path(target: Path) -> Path:
    target = target.expanduser()
    if target.is_dir():
        candidate = target / PRED_FILENAME
        if not candidate.is_file():
            raise FileNotFoundError(f"{target} contains no {PRED_FILENAME}")
        return candidate
    if not target.is_file():
        raise FileNotFoundError(f"predictions file does not exist: {target}")
    return target


def load_predictions(path: Path, pred_col: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for column in ("well_id", "submit_index", "TVT", pred_col):
        if column not in df.columns:
            raise KeyError(f"{path} has no {column!r} column")
    out = df[["well_id", "submit_index", "TVT", pred_col]].copy()
    out = out.rename(columns={pred_col: "pred"})
    out = out.sort_values(["well_id", "submit_index"], kind="stable").reset_index(drop=True)
    if out[["well_id", "submit_index"]].duplicated().any():
        raise ValueError(f"{path} has duplicate (well_id, submit_index) rows")
    return out


def per_well_sse(df: pd.DataFrame) -> pd.DataFrame:
    error = (df["TVT"].to_numpy(dtype=np.float64) - df["pred"].to_numpy(dtype=np.float64)) ** 2
    frame = pd.DataFrame({"well_id": df["well_id"].to_numpy(), "sq_err": error})
    grouped = frame.groupby("well_id", sort=True)["sq_err"]
    return pd.DataFrame({"sse": grouped.sum(), "rows": grouped.size()})


def pooled_rmse(sse: np.ndarray, rows: np.ndarray) -> float:
    total_rows = float(rows.sum())
    if total_rows <= 0.0:
        return float("nan")
    return float(np.sqrt(sse.sum() / total_rows))


def removal_curve(base: pd.DataFrame, treat: pd.DataFrame, max_k: int) -> pd.DataFrame:
    """Rescore both runs after dropping the k largest contributors to the gap."""

    gap = treat["sse"].to_numpy(dtype=np.float64) - base["sse"].to_numpy(dtype=np.float64)
    order = np.argsort(-np.abs(gap), kind="stable")
    rows = base["rows"].to_numpy(dtype=np.float64)
    base_sse = base["sse"].to_numpy(dtype=np.float64)
    treat_sse = treat["sse"].to_numpy(dtype=np.float64)
    well_ids = np.asarray(base.index)

    keep = np.ones(len(well_ids), dtype=bool)
    records = []
    for k in range(max_k + 1):
        if k > 0:
            keep[order[k - 1]] = False
        if not keep.any():
            break
        records.append(
            {
                "k": k,
                "removed_well": "" if k == 0 else str(well_ids[order[k - 1]]),
                "removed_gap": 0.0 if k == 0 else float(gap[order[k - 1]]),
                "wells_left": int(keep.sum()),
                "rows_left": int(rows[keep].sum()),
                "base_rmse": pooled_rmse(base_sse[keep], rows[keep]),
                "treat_rmse": pooled_rmse(treat_sse[keep], rows[keep]),
            }
        )
    curve = pd.DataFrame.from_records(records)
    curve["delta"] = curve["treat_rmse"] - curve["base_rmse"]
    return curve


def first_k_without_gain(curve: pd.DataFrame) -> int | None:
    """Smallest k whose remaining wells no longer show an improvement."""

    for k, delta in zip(curve["k"].to_numpy(), curve["delta"].to_numpy()):
        if delta >= 0.0:
            return int(k)
    return None


def bootstrap_delta(base: pd.DataFrame, treat: pd.DataFrame, draws: int, seed: int) -> dict:
    """Well-level bootstrap of the pooled-RMSE difference."""

    rng = np.random.default_rng(seed)
    rows = base["rows"].to_numpy(dtype=np.float64)
    base_sse = base["sse"].to_numpy(dtype=np.float64)
    treat_sse = treat["sse"].to_numpy(dtype=np.float64)
    well_count = len(rows)
    deltas = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        pick = rng.integers(0, well_count, size=well_count)
        drawn_rows = rows[pick]
        deltas[draw] = pooled_rmse(treat_sse[pick], drawn_rows) - pooled_rmse(
            base_sse[pick], drawn_rows
        )
    return {
        "draws": int(draws),
        "mean": float(deltas.mean()),
        "p05": float(np.percentile(deltas, 5.0)),
        "p50": float(np.percentile(deltas, 50.0)),
        "p95": float(np.percentile(deltas, 95.0)),
        "share_improving": float(np.mean(deltas < 0.0)),
    }


def run(args: argparse.Namespace) -> int:
    base_path = resolve_predictions_path(args.base)
    treat_path = resolve_predictions_path(args.treat)
    base_df = load_predictions(base_path, args.pred_col)
    treat_df = load_predictions(treat_path, args.pred_col)

    if len(base_df) != len(treat_df) or not base_df["well_id"].equals(
        treat_df["well_id"]
    ) or not base_df["submit_index"].equals(treat_df["submit_index"]):
        raise ValueError(
            "the two runs were not scored on the same rows; "
            f"base={len(base_df):,} rows from {base_df['well_id'].nunique():,} wells, "
            f"treat={len(treat_df):,} rows from {treat_df['well_id'].nunique():,} wells"
        )
    if not np.allclose(
        base_df["TVT"].to_numpy(dtype=np.float64),
        treat_df["TVT"].to_numpy(dtype=np.float64),
        equal_nan=True,
    ):
        raise ValueError("the two runs disagree on the ground truth; check the holdout split")

    base_well = per_well_sse(base_df)
    treat_well = per_well_sse(treat_df)
    well_count = len(base_well)
    rows = base_well["rows"].to_numpy(dtype=np.float64)
    base_rmse = pooled_rmse(base_well["sse"].to_numpy(dtype=np.float64), rows)
    treat_rmse = pooled_rmse(treat_well["sse"].to_numpy(dtype=np.float64), rows)
    delta = treat_rmse - base_rmse

    max_k = well_count - 1 if args.max_k is None else min(int(args.max_k), well_count - 1)
    curve = removal_curve(base_well, treat_well, max_k)
    k_star = first_k_without_gain(curve)

    if args.min_k is None:
        min_k = max(5, int(round(DEFAULT_SURVIVAL_FRACTION * well_count)))
    else:
        min_k = int(args.min_k)

    if delta >= 0.0:
        verdict = "REJECT (no improvement at k=0)"
    elif k_star is None:
        verdict = "ACCEPT (gain survives every removal tested)"
    elif k_star >= min_k:
        verdict = "ACCEPT"
    else:
        verdict = "REJECT"

    gap = treat_well["sse"].to_numpy(dtype=np.float64) - base_well["sse"].to_numpy(dtype=np.float64)
    order = np.argsort(-np.abs(gap), kind="stable")
    top = [
        {
            "well_id": str(base_well.index[idx]),
            "sse_gap": float(gap[idx]),
            "rows": int(rows[idx]),
            "base_well_rmse": float(np.sqrt(base_well["sse"].to_numpy()[idx] / rows[idx])),
            "treat_well_rmse": float(np.sqrt(treat_well["sse"].to_numpy()[idx] / rows[idx])),
        }
        for idx in order[: args.top]
    ]
    improved_wells = int(np.sum(gap < 0.0))

    report = {
        "base": str(base_path),
        "treat": str(treat_path),
        "pred_col": args.pred_col,
        "wells": int(well_count),
        "rows": int(rows.sum()),
        "base_pooled_rmse": base_rmse,
        "treat_pooled_rmse": treat_rmse,
        "delta_pooled_rmse": delta,
        "wells_improved": improved_wells,
        "wells_improved_share": improved_wells / float(well_count),
        "k_star": k_star,
        "min_k": min_k,
        "verdict": verdict,
        "top_contributors": top,
    }
    if args.bootstrap > 0:
        report["bootstrap"] = bootstrap_delta(
            base_well, treat_well, args.bootstrap, args.bootstrap_seed
        )

    print(f"base   {base_path}")
    print(f"treat  {treat_path}")
    print(
        f"{well_count:,} holdout wells, {int(rows.sum()):,} scored rows, "
        f"prediction column {args.pred_col!r}"
    )
    print(f"pooled RMSE: base={base_rmse:.4f} ft  treat={treat_rmse:.4f} ft  delta={delta:+.4f} ft")
    print(
        f"wells where treat is better: {improved_wells:,}/{well_count:,} "
        f"({improved_wells / well_count:.1%})"
    )
    print("")
    print("removal curve (drop the largest contributors to the difference first):")
    print(f"{'k':>5}  {'wells':>6}  {'base':>9}  {'treat':>9}  {'delta':>9}  removed well")
    step = max(1, args.print_step)
    for _, row in curve.iterrows():
        k = int(row["k"])
        if k % step != 0 and not (k_star is not None and abs(k - k_star) <= 1):
            continue
        print(
            f"{k:>5}  {int(row['wells_left']):>6}  {row['base_rmse']:>9.4f}  "
            f"{row['treat_rmse']:>9.4f}  {row['delta']:>+9.4f}  {row['removed_well']}"
        )
    print("")
    if k_star is None:
        print(f"k* > {int(curve['k'].max())} (the gain never disappeared within the tested range)")
    else:
        print(f"k* = {k_star} (the gain disappears after removing {k_star} of {well_count:,} wells)")
    print(f"acceptance bar: k* >= {min_k}")
    if "bootstrap" in report:
        boot = report["bootstrap"]
        print(
            f"well-level bootstrap delta: p50={boot['p50']:+.4f} "
            f"[p05={boot['p05']:+.4f}, p95={boot['p95']:+.4f}] ft, "
            f"improving in {boot['share_improving']:.1%} of draws"
        )
    print(f"VERDICT: {verdict}")
    print("")
    print("largest contributors to the difference (negative gap = treat is better):")
    for entry in top:
        print(
            f"  {entry['well_id']}  gap={entry['sse_gap']:+.4g}  "
            f"base_rmse={entry['base_well_rmse']:.3f}  treat_rmse={entry['treat_well_rmse']:.3f}"
        )

    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        tag = report_basename(base_path)
        curve_path = output_dir / f"{CURVE_STEM}__vs_{tag}.csv"
        report_path = output_dir / f"{REPORT_STEM}__vs_{tag}.json"
        curve.to_csv(curve_path, index=False)
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print("")
        print(f"wrote {curve_path}")
        print(f"wrote {report_path}")
    return 0 if verdict.startswith("ACCEPT") else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Decide whether a candidate's holdout improvement survives removing the "
            "wells that produced it."
        )
    )
    parser.add_argument(
        "--base", type=Path, required=True,
        help="Baseline results directory (or holdout_predictions.pqt).",
    )
    parser.add_argument(
        "--treat", type=Path, required=True,
        help="Candidate results directory (or holdout_predictions.pqt).",
    )
    parser.add_argument(
        "--pred-col", default="TVT_pred",
        help="Prediction column to score; TVT_pred_raw skips the SG smoothing.",
    )
    parser.add_argument(
        "--min-k", type=int,
        help="Acceptance bar for k*. Default is 52/773 of the holdout well count, "
             "the public-leaderboard-sized bar the 2nd-place solution used.",
    )
    parser.add_argument(
        "--max-k", type=int,
        help="Stop the removal curve after this many wells (default: all but one).",
    )
    parser.add_argument(
        "--print-step", type=int, default=1,
        help="Print every Nth row of the removal curve (the full curve is still saved).",
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="How many largest contributors to list.",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=2000,
        help="Well-level bootstrap draws for a complementary read; 0 disables it.",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20260905)
    parser.add_argument(
        "--output-dir", type=Path,
        help="Write the full removal curve and a JSON report here.",
    )
    return parser


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
