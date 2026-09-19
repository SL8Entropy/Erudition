#!/usr/bin/env python3
"""Score an existing AnchorCNN checkpoint on the 20% holdout, without retraining.

Rebuilds the same split, the same wells and the same grid as ``anchor_train.py``, loads
one or more checkpoints, decodes with ``dp_expected_level`` and writes the same
``holdout_metrics.json`` / ``holdout_predictions.pqt`` / ``holdout_well_rmse.csv``.

Because inference is ~6 s per MD phase over the 155 holdout wells, changing the TTA or
averaging several checkpoints costs seconds rather than a training run -- the same
reason the 1st-place repo carries ``seq_NN_rescore.py`` next to its training entry point.

    python anchor_eval.py --models runs/dzl_w1/model_last.pt --out runs/dzl_w1_tta8 --tta 8

Several ``--models`` paths are averaged in prediction space, which is what the author's
submission does across its 15 checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

import gr2tvt_data as gd                                    # noqa: E402
from anchor_data import (eval_rows, load_wells, set_grid, split_wells,  # noqa: E402
                         well_names)
from anchor_train import build_model, predict_holdout, score, Logger    # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True, help="one or more .pt checkpoints")
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=str(HERE / "data"))
    p.add_argument("--cache", default=str(HERE / "cache/wells.npz"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--limit-wells", type=int, default=0)
    p.add_argument("--tta", type=int, default=1)
    p.add_argument("--val-batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16")
    # model geometry: must match the checkpoints
    p.add_argument("--backbone", default="efficientnet_b0")
    p.add_argument("--backbone-weights", default="")
    p.add_argument("--ps-col", type=int, default=16)
    p.add_argument("--fuse-div", type=int, default=4)
    p.add_argument("--n-move", type=int, default=10)
    p.add_argument("--drop-path", type=float, default=0.0)
    p.add_argument("--win", type=float, default=128.0)
    p.add_argument("--stem-stride", type=int, default=1, choices=[1, 2],
                   help="must match the checkpoint's training config")
    p.add_argument("--row", type=float, default=0.5,
                   help="must match the checkpoint's training config")
    p.add_argument("--convnext-stem-stride", type=int, default=2,
                   help="must match the checkpoint's training config")
    p.add_argument("--arch", choices=["effnet", "separable"], default="effnet",
                   help="must match the checkpoint's training config")
    p.add_argument("--pf-channels", type=int, default=0, metavar="N",
                   help="add 2 particle-filter input channels (belief + spread), computed "
                        "on the fly from N filter profiles.  0 = off.  Must match between "
                        "training and evaluation: it changes the model's input width.")
    p.add_argument("--sibling-sigma", action="store_true",
                   help="add 1 channel of sibling disagreement per depth: how much wells in "
                        "the same rock differ there, i.e. how much a GR match is worth")
    p.add_argument("--sibling-w", type=float, default=0.0,
                   help="sibling-lateral reference blend weight; must match the training run")
    p.add_argument("--sibling-bin", type=float, default=1.0)
    p.add_argument("--gr-prefilter-ft", type=float, default=0.0,
                   help="must match the checkpoint's training config")
    p.add_argument("--save-std", action="store_true",
                   help="also write TVT_std, the spread of the decoded belief combined across "
                        "checkpoints and phases (law of total variance)")
    p.add_argument("--dzl-w", type=float, default=1.0,
                   help="only decides whether the dz_layer head exists; it is unused at inference")
    a = p.parse_args(argv)
    a.pretrained = False           # weights come from --models, never from ImageNet
    if not a.cache:
        a.cache = None
    return a


def main(args):
    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(f"{out_dir} exists and is not empty; pass --force to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "eval.log")
    log(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(sys.argv)}")

    device = torch.device(args.device)
    set_grid(row=args.row, gr_prefilter_ft=args.gr_prefilter_ft, pf_profiles=args.pf_channels,
                    sib_sigma=args.sibling_sigma)
    log(f"grid: {gd.T} rows x {gd.H + args.ps_col} cols (row {gd.ROW} ft), "
        f"stem_stride {args.stem_stride}, arch {args.arch}, "
        f"GR prefilter {args.gr_prefilter_ft:g} ft")
    names = well_names(args.data)
    if args.limit_wells:
        names = names[:args.limit_wells]
    tr_names, ho_names = split_wells(names, args.train_frac)
    log(f"holdout: {len(ho_names)} wells (first {ho_names[0]})")
    wells = load_wells(Path(args.data), names, Path(args.cache) if args.cache else None, log=log)
    bank = None
    if args.sibling_w > 0:
        from anchor_sibling import apply_to_wells
        bank = apply_to_wells(wells, tr_names, names, args.sibling_w, args.sibling_bin, log=log)
    if args.sibling_sigma:
        from anchor_sibling import attach_spread
        attach_spread(wells, tr_names, names, bank=bank, bin_ft=args.sibling_bin, log=log)
    ho_wells = {n: wells[n] for n in ho_names}
    log(f"scored rows: {sum(len(eval_rows(w)) for w in ho_wells.values()):,}")

    phases = tuple(np.arange(args.tta) * (gd.COLW / 4.0)) if args.tta > 1 else (0.0,)
    log(f"MD phases: {tuple(round(p, 1) for p in phases)}")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[args.amp]

    model = build_model(args, device, log=log)
    acc, acc2, failed_any = {}, {}, set()
    for ck in args.models:
        sd = torch.load(ck, map_location=device)
        sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
        model.load_state_dict(sd)
        t0 = time.time()
        mom = {} if args.save_std else None
        preds, failed = predict_holdout(model, ho_wells, ho_names, device, args,
                                        phases=phases, amp_dtype=amp_dtype, moments=mom)
        failed_any |= set(failed)
        for k, v in preds.items():
            acc[k] = v if k not in acc else acc[k] + v
            if mom is not None:
                acc2[k] = mom[k] if k not in acc2 else acc2[k] + mom[k]
        m, _ = score(ho_wells, preds, failed)
        log(f"{Path(ck).name}: pooled RMSE {m['pooled_rmse']:.4f} ft  ({time.time()-t0:.0f}s)")

    preds = {k: v / len(args.models) for k, v in acc.items()}
    metrics, df = score(ho_wells, preds, sorted(failed_any))
    metrics.update(models=[str(m) for m in args.models], tta=args.tta, arch=args.arch,
                   row=args.row, n_move=args.n_move, gr_prefilter_ft=args.gr_prefilter_ft)

    if args.save_std:
        # E[x^2] and E[x] are each averaged over every (checkpoint, phase) view, so
        # var = E[x^2] - E[x]^2 is the mixture variance: within-view belief spread plus
        # disagreement between views.  Levels are relative to TVT_PS, which is shared.
        std = {}
        for k, e2 in acc2.items():
            ti = ho_wells[k]["tvt_input"]
            tvt_ps = float(ti[np.isfinite(ti)][-1])
            mean_rel = preds[k] - tvt_ps
            std[k] = np.sqrt(np.clip(e2 / len(args.models) - mean_rel ** 2, 0.0, None))
        df["TVT_std"] = np.nan
        for k, s_ in std.items():
            sel = (df["well_id"] == k).to_numpy()
            df.loc[sel, "TVT_std"] = s_       # both ordered by ascending row index
        abs_err = np.sqrt(df["sq_err"].to_numpy())
        ok = np.isfinite(df["TVT_std"].to_numpy())
        metrics["std_error_spearman"] = float(pd.Series(df["TVT_std"][ok]).corr(
            pd.Series(abs_err[ok]), method="spearman"))
        log(f"belief spread vs |error|, Spearman rho = {metrics['std_error_spearman']:.3f}")

    df.drop(columns=["sq_err"]).to_parquet(out_dir / "holdout_predictions.pqt", index=False)
    per = df.groupby("well_id")["sq_err"].agg(["mean", "sum", "size"])
    per = per.assign(rmse=np.sqrt(per["mean"])).rename(columns={"sum": "sse", "size": "rows"})
    per[["rmse", "sse", "rows"]].sort_values("sse", ascending=False).to_csv(
        out_dir / "holdout_well_rmse.csv")
    (out_dir / "holdout_metrics.json").write_text(json.dumps(metrics, indent=2))

    log("")
    log(f"HOLDOUT POOLED RMSE: {metrics['pooled_rmse']:.4f} ft "
        f"over {metrics['n_rows']:,} rows in {metrics['n_wells']} wells "
        f"({len(args.models)} ckpt x {len(phases)} phase)")
    log(f"per-well RMSE  mean {metrics['well_rmse_mean']:.3f}  "
        f"p50 {metrics['well_rmse_p50']:.3f}  p95 {metrics['well_rmse_p95']:.3f}")
    log(f"worst 5 wells carry {metrics['sse_share_top5']*100:.0f}% of the squared error: "
        f"{metrics['worst_wells']}")
    return metrics


if __name__ == "__main__":
    main(parse_args())
