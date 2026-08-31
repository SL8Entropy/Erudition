"""Held-out benchmark: train each variant, score it, and write a report.

**Which split, and why.** The `test/` folder shipped with the data holds 3
wells, they carry no `TVT` column, and they are duplicates of wells that are
also in `train/`. So it cannot produce an error metric -- it exists to check the
submission path. The evaluation that matters for this dataset is a split by
*well* over the 773 labelled training wells: rows within a well are strongly
correlated, so a row-wise split leaks badly and would report a fantasy score.

This module holds out one GroupKFold fold (155 wells, 20%) as the local test
set, trains on the other 618, and scores the held-out wells. That is the same
protocol every top-five team used, and the same one `train_fold` already
implements -- so the benchmark number and the OOF number are the same quantity.

Timing, metrics and configuration of every run land in a text report.
"""
from __future__ import annotations

import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .data import group_kfold, load_split, split_dir
from .features import FeatureProvider
from .metrics import score_report
from .model import build_model, count_params
from .train import predict_wells, train_fold


def _safe_node() -> str:
    try:
        return platform.node().encode("ascii", "replace").decode("ascii")
    except Exception:
        return "?"


def _archive_previous(report_path: Path) -> None:
    """Move an existing report aside instead of overwriting it.

    The report is rewritten in full after every variant, so a second run would
    otherwise destroy the first -- and comparing a 12-epoch run against a
    30-epoch one is the whole point of running it twice.
    """
    if not report_path.exists():
        return
    stamp = datetime.fromtimestamp(report_path.stat().st_mtime)
    dest = report_path.with_name(
        f"{report_path.stem}_{stamp:%Y%m%d_%H%M}{report_path.suffix}")
    n = 1
    while dest.exists():
        dest = report_path.with_name(
            f"{report_path.stem}_{stamp:%Y%m%d_%H%M}_{n}{report_path.suffix}")
        n += 1
    report_path.rename(dest)
    print(f"[bench] kept the previous report as {dest.name}")


def _fmt_time(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def _header(cfg_names, wells, holdout, n_holdout, epochs, device) -> str:
    gpu = (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    return "\n".join([
        "=" * 78,
        "ROGII wellbore geology prediction -- local held-out benchmark",
        "=" * 78,
        f"started        : {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"machine        : {_safe_node()}  |  {gpu}  |  torch {torch.__version__}",
        f"device         : {device}",
        "",
        "split          : GroupKFold by well over the 773 labelled train wells.",
        f"                 fold {holdout} of 5 held out as the local test set.",
        f"                 {len(wells) - n_holdout} wells train / {n_holdout} wells held out.",
        "                 The shipped test/ folder has 3 unlabelled wells that",
        "                 duplicate training wells, so it cannot be scored; it is",
        "                 used only to check the submission path.",
        "",
        f"epochs         : {epochs}",
        f"variants       : {', '.join(cfg_names)}",
        f"metric         : pooled RMSE in feet over every scored row of the",
        "                 held-out wells (the competition metric), plus the",
        "                 well-mean RMSE, which weights every well equally.",
        "=" * 78,
        "",
    ])


def _row(rec) -> str:
    lines = [
        f"--- {rec['variant']} " + "-" * (70 - len(rec['variant'])),
        f"  {rec['notes']}",
        "",
        f"  channels           : {rec['in_chans']}   "
        f"canvas {rec['canvas']}   params {rec['params']}",
        f"  heads              : {rec['heads']}",
        f"  decode             : {rec['decode']}",
        f"  train time         : "
        + (rec['train_time_s'] if isinstance(rec['train_time_s'], str)
           else f"{_fmt_time(rec['train_time_s'])}  ({rec['epochs']} epochs)"),
        f"  inference time     : {_fmt_time(rec['infer_time_s'])} "
        f"for {rec['n_wells']} wells "
        f"({rec['infer_time_s'] / max(rec['n_wells'], 1):.2f} s/well)",
        "",
        f"  pooled RMSE        : {rec['pooled_rmse']:.4f} ft",
        f"  well-mean RMSE     : {rec['well_mean_rmse']:.4f} ft",
        f"  rows scored        : {rec['n_rows']:,}",
        f"  worst 5 wells hold : {rec['top5_share'] * 100:.1f}% of the squared error",
        f"  worst wells        : "
        + ", ".join(f"{w} ({e:.1f} ft)" for w, e in rec["worst_wells"][:5]),
        f"  checkpoint         : {rec['checkpoint']}",
        "",
    ]
    return "\n".join(lines)


def run_benchmark(cfgs, data_root, out_dir, report_path, device="cuda",
                  holdout_fold: int = 0, epochs: int = None,
                  baselines: bool = True, make_submission: bool = True):
    """Train and score each config on the same held-out wells."""
    out_dir = Path(out_dir)
    report_path = Path(report_path)
    wells = load_split(split_dir(data_root, "train"), require_tvt=True)
    ids = [w.well_id for w in wells]
    n_folds = cfgs[0].n_folds
    assign = group_kfold(ids, n_folds, seed=0)
    held = [w for w in wells if assign[w.well_id] == holdout_fold]
    if epochs:
        for c in cfgs:
            c.epochs = epochs

    text = _header([c.name for c in cfgs], wells, holdout_fold, len(held),
                   epochs or cfgs[0].epochs, device)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _archive_previous(report_path)
    report_path.write_text(text, encoding="utf-8")
    records = []

    if baselines:
        text += _baselines(held)
        report_path.write_text(text, encoding="utf-8")

    for cfg in cfgs:
        print(f"\n{'=' * 60}\n[bench] {cfg.name}\n{'=' * 60}")
        rec = _run_one(cfg, wells, held, assign, holdout_fold, out_dir, device)
        records.append(rec)
        text += _row(rec)
        report_path.write_text(text, encoding="utf-8")
        print(f"[bench] {cfg.name}: pooled RMSE {rec['pooled_rmse']:.4f} ft  "
              f"(train {rec['train_time_s'] if isinstance(rec['train_time_s'], str) else _fmt_time(rec['train_time_s'])}, "
              f"infer {_fmt_time(rec['infer_time_s'])})")

    text += _summary(records)
    if make_submission:
        text += _submission_note(cfgs[0], out_dir, data_root, device, held)
    report_path.write_text(text, encoding="utf-8")
    print(f"\n[bench] report written to {report_path}")
    return records


def _baselines(held):
    """Reference points, so the model numbers mean something."""
    from .data import initial_structural_rate
    const = {w.well_id: np.full(w.n, w.anchor_tvt) for w in held}
    flat = {w.well_id: w.anchor_tvt - (w.z - w.anchor_z) for w in held}
    proj = {}
    for w in held:
        r0 = initial_structural_rate(w)
        proj[w.well_id] = (w.anchor_tvt + r0 * (w.md - w.anchor_md)
                           - (w.z - w.anchor_z))
    out = ["--- reference baselines (no model) " + "-" * 43, ""]
    for name, p in (("constant TVT (predict no change)", const),
                    ("flat layers (anchor - dZ)", flat),
                    ("dip projected from the prefix", proj)):
        r = score_report(p, held)
        out.append(f"  {name:34s} pooled RMSE {r['pooled_rmse']:8.3f} ft")
    out += ["", ""]
    return "\n".join(out)


def _run_one(cfg, wells, held, assign, holdout_fold, out_dir, device):
    spec = cfg.canvas_spec()
    run_dir = out_dir / cfg.name
    reuse = getattr(cfg, "reuse_weights", None)

    if reuse:                      # v1_tta: decode-only, runs the base weights
        train_s = f"none (reuses {reuse})"
        ck_path = sorted((out_dir / reuse).glob(
            f"split0_seed0_fold{holdout_fold}.pt"))
        if not ck_path:
            raise FileNotFoundError(
                f"{cfg.name} needs {reuse} weights; train {reuse} first")
        ck_path = ck_path[0]
    else:
        t0 = time.time()
        train_fold(cfg, wells, fold=holdout_fold, seed=0, split=0,
                   out_dir=run_dir, device=device)
        train_s = time.time() - t0
        ck_path = run_dir / f"split0_seed0_fold{holdout_fold}.pt"

    allowed = {w.well_id for w in wells if assign[w.well_id] != holdout_fold}
    provider = FeatureProvider(wells, allowed_full=allowed, use_pf=cfg.use_pf,
                               use_xy=cfg.use_xy, use_sibling=cfg.use_sibling,
                               correct_typewell=cfg.correct_typewell)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    model = build_model(cfg, ck.get("in_chans", spec.in_chans),
                        pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    t0 = time.time()
    oof = predict_wells(model, held, cfg, provider, device,
                        n_phases=cfg.n_phases, reanchor=cfg.reanchor,
                        adaptive_canvas=cfg.adaptive_canvas,
                        dp_weight=cfg.dp_weight)
    infer_s = time.time() - t0
    preds = {k: v["pred"] for k, v in oof.items()}
    rep = score_report(preds, held)

    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run_dir / f"heldout_fold{holdout_fold}.npz",
                        **{f"pred__{k}": v for k, v in preds.items()})
    n_params = count_params(model)
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    return dict(
        variant=cfg.name, notes=cfg.notes, in_chans=spec.in_chans,
        bench_note=getattr(cfg, 'bench_note', ''),
        canvas=f"{spec.n_rows}x{spec.n_cols}", params=n_params,
        heads=f"dz={cfg.head_dz} move={cfg.head_move} sigma={cfg.head_sigma}",
        decode=(f"phases={cfg.n_phases} reanchor={cfg.reanchor} "
                f"adaptive={cfg.adaptive_canvas} dp_weight={cfg.dp_weight}"),
        train_time_s=train_s, infer_time_s=infer_s,
        n_wells=len(held), epochs=cfg.epochs,
        checkpoint=str(ck_path), **{k: rep[k] for k in
                                    ("pooled_rmse", "well_mean_rmse", "n_rows",
                                     "top5_share", "worst_wells")})


def _summary(records):
    if not records:
        return ""
    best = min(records, key=lambda r: r["pooled_rmse"])
    lines = ["=" * 78, "SUMMARY (held-out pooled RMSE, lower is better)", "=" * 78,
             "",
             f"  {'variant':12s} {'pooled':>9s} {'well-mean':>10s} "
             f"{'train':>10s} {'infer':>9s}   {'vs base':>8s}",
             "  " + "-" * 64]
    base = next((r for r in records if r["variant"] == "base"), None)
    for r in records:
        d = ("" if base is None or r is base
             else f"{r['pooled_rmse'] - base['pooled_rmse']:+8.4f}")
        tt = (r["train_time_s"] if isinstance(r["train_time_s"], str)
              else _fmt_time(r["train_time_s"]))
        lines.append(f"  {r['variant']:12s} {r['pooled_rmse']:9.4f} "
                     f"{r['well_mean_rmse']:10.4f} {tt:>10s} "
                     f"{_fmt_time(r['infer_time_s']):>9s}   {d:>8s}")
    lines += ["",
              f"  best: {best['variant']} at {best['pooled_rmse']:.4f} ft",
              "",
              "  Caveat: this is ONE fold at a reduced epoch count. Run-to-run",
              "  spread on a 5-fold mean was 0.10-0.19 RMSE for 4th place, which",
              "  is larger than most differences you will see here. Use",
              "  `run.py compare` for the leave-largest-contribution-out test",
              "  before believing any ordering.",
              "", ""]
    return "\n".join(lines)


def _submission_note(cfg, out_dir, data_root, device, held):
    try:
        from .infer import run_inference
        t0 = time.time()
        out_csv = Path(out_dir) / "submission_base.csv"
        run_inference(cfg, out_dir / cfg.name, data_root, out_csv, device=device,
                      details_path=out_csv.with_name(f"details_{cfg.name}.csv"),
                      verbose=False)
        dt = time.time() - t0
        held_ids = {w.well_id for w in held}
        from .data import list_well_ids
        test_ids = list_well_ids(split_dir(data_root, "test"))
        overlap = sorted(set(test_ids) & held_ids)
        return "\n".join([
            "--- submission path " + "-" * 58, "",
            f"  wrote {out_csv} in {_fmt_time(dt)}",
            f"  test/ wells: {test_ids}",
            f"  of those, held out from training: {overlap or 'none'}",
            "  (the shipped test wells duplicate training wells and carry no",
            "   TVT column, so no error metric is computed for them)",
            "", ""])
    except Exception as e:
        return f"--- submission path ---\n  FAILED: {e}\n\n"
