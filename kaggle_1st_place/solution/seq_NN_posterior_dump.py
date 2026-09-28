#!/usr/bin/env python3
"""Dump a trained holdout run's depth posteriors and test whether its failures
are bimodal (mode confusion) or unimodal (drift).

The 1st-place model reports, for every 32 ft MD column, the *mean* of a softmax
over 400 candidate depths.  The project's open question is what the worst wells
look like underneath that mean: two humps with the truth under the one the model
under-weighted (which a mode-aware decoder or a unimodal loss could fix), or one
hump sliding away from the truth (which neither can).  ``posterior_analysis.py``
defines the categories; this script runs the model and feeds it.

Nothing is retrained.  Setup -- settings, recipe, 80/20 split, fold-safe geo
prior, loader, ``models.pkl`` -- is exactly ``seq_NN_rescore.py``'s; only the
scoring step is replaced, so the numbers line up with the run's own metrics.
The script checks that: the mean decode must reproduce the run's raw pooled
RMSE, and the model's own prediction must equal the mean of the logits it
returns.

    python seq_NN_posterior_dump.py --id 0801_V2 --models-dir results/baseline_check ^
        --output-dir results/baseline_check_posterior --device cuda --offline-timm

Outputs, in ``--output-dir``:

    posterior_report.json      pooled RMSE per decoder, failure categories, per-well tables
    posterior_wells.csv        one row per holdout well
    posterior_columns.parquet  one row per (well, MD column): the posterior's summary
    posterior_focus.npz        full posteriors for the focus wells, for plotting
    posterior_all.npz          full posteriors for EVERY holdout well (only with --save-all,
                               ~40 MB); same keys as the focus file, so decoder ideas can be
                               scored on all 155 wells instead of 16 hand-picked ones

    python seq_NN_posterior_dump.py --id 0801_V2 --models-dir results/baseline_check ^
        --output-dir results/baseline_check_posterior_all --save-all --device cuda --offline-timm
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

import posterior_analysis as pa
import seq_NN_holdout_eval as holdout
import seq_NN_rescore as rescore

# The six wells the 23 September report names as in the worst 15 for every network.
HARD_WELLS = ("d7eb0be8", "f2d4c8c9", "f6bc699b", "f6d009f4", "f8afa78a", "fb0904bd")
N_WORST = 10     # plus this run's own worst wells by squared error
N_CONTROL = 6    # plus wells around the median, as a reference for what "normal" looks like

_EXTRA = {"models_dir": None, "save_all": False}


def _row_bins(meta, cfg):
    """Canvas column each suffix row is read from (mirrors expand_target_prediction)."""
    if meta.get("stretch_to_full_size", False) or getattr(cfg, "strech_to_full_size", False):
        raise NotImplementedError("full-stretch recipes are not supported by this dump")
    suffix_len = int(meta["suffix_len"])
    kept = min(suffix_len, int(cfg.target_len))
    raw = np.minimum(np.arange(suffix_len), max(kept - 1, 0))
    return (int(cfg.prefix_len) + raw) // int(cfg.downsample)


def _forward_all(models, loader, cfg, seq_NN_train):
    """Mixture posterior (averaged probabilities) and averaged prediction, per sample."""
    import torch
    import torch.nn.functional as F

    device = torch.device(cfg.device)
    amp_dtype = getattr(torch, cfg.amp_dtype)
    amp_enabled = cfg.amp_dtype != "float32" and device.type == "cuda"
    prob_sum, pred_sum, metas = None, None, None
    for model in models.values():
        model = seq_NN_train.move_model_to_device(model, cfg)
        model.eval()
        probs, preds, these = [], [], []
        with torch.no_grad():
            for batch in loader:
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    pred, extra = seq_NN_train.model_forward(
                        model,
                        batch["unet_static"].to(device, non_blocking=True),
                        batch["typewell_aux"].to(device, non_blocking=True),
                        bin_count=batch["bin_count"].to(device, non_blocking=True),
                        row_mask=batch["row_mask"].to(device, non_blocking=True),
                        extra_return=True,
                    )
                logits = extra.get("alignment_logits")
                if logits is None:
                    raise RuntimeError("model returned no alignment_logits; this recipe has no depth posterior")
                probs.append(F.softmax(logits.float(), dim=-1).cpu().numpy())
                preds.append(pred.float().cpu().numpy())
                these.extend(batch["meta"])
        model.cpu()
        p = np.concatenate(probs, axis=0)
        q = np.concatenate(preds, axis=0)
        prob_sum = p if prob_sum is None else prob_sum + p
        pred_sum = q if pred_sum is None else pred_sum + q
        metas = these
    k = float(len(models))
    return prob_sum / k, pred_sum / k, metas


def _save_posteriors(path, well_ids, per_well_inputs, prob, x):
    """Per-column posteriors (float16) plus column bins, truth and geo prior, per well."""
    save = {"levels": x.astype(np.float32), "focus": np.array(well_ids)}
    for wid in well_ids:
        if wid not in per_well_inputs:
            continue
        row_bin, y_row, geo_row, tvt0, i = per_well_inputs[wid]
        used = np.unique(row_bin)
        save[f"{wid}__p"] = prob[i][used].astype(np.float16)
        save[f"{wid}__bins"] = used.astype(np.int16)
        save[f"{wid}__y"] = np.array([np.nanmean(y_row[row_bin == b]) for b in used], dtype=np.float32)
        if geo_row is not None and geo_row.size == row_bin.size:
            save[f"{wid}__geo"] = np.array([np.nanmean(geo_row[row_bin == b]) for b in used], dtype=np.float32)
    np.savez_compressed(path, **save)


def collect(cfg, main_module, seq_NN_train, models, train_wells, holdout_wells, train_frac):
    """Drop-in for seq_NN_rescore.score: same geo prior and loader, posterior analysis instead of scoring."""
    log = main_module.log
    main_module.log_section("setup")
    log(f"posterior dump: models={len(models):,}, holdout wells={len(holdout_wells):,}")
    split_path = holdout.write_split_file(cfg.output_dir, train_wells, holdout_wells)
    log(f"saved holdout split: {split_path}")
    main_module.save_cfg(cfg)

    main_module.log_section("holdout inference")
    geo_cfg = seq_NN_train.resolve_geo_prior_cfg(cfg)
    val_geo_prior, val_prior_summary = seq_NN_train.make_geo_prior_for_wells(
        cfg.train_path, cfg.train_path, list(train_wells), list(holdout_wells), geo_cfg,
        exclude_query_from_support=False,
    )
    log(f"geo_prior {geo_cfg.method}: rmse={val_prior_summary.get('rmse')}")
    pf_cache_dir = seq_NN_train.prepare_pf_cache_if_needed(
        "train", cfg.train_path, list(train_wells) + list(holdout_wells), cfg, log
    )
    loader = seq_NN_train.make_loader(
        path=cfg.train_path, well_ids=list(holdout_wells), cfg=cfg,
        training=True, simulation=False, shuffle=False,
        batch_size=cfg.val_batch_size, seed=cfg.seed + 2,
        pf_cache_dir=pf_cache_dir, pf_sample_cache_dir=None,
        geo_prior=val_geo_prior, drop_last=False,
    )
    prob, pred, metas = _forward_all(models, loader, cfg, seq_NN_train)
    log(f"posteriors: {prob.shape} (samples, MD columns, depth levels)")

    x = pa.level_axis(float(cfg.typewell_window), int(cfg.typewell_len))
    if prob.shape[-1] != x.size:
        raise ValueError(f"posterior has {prob.shape[-1]} levels, cfg says {x.size}")

    main_module.log_section("posterior analysis")
    wells, sanity_gap, model_sse, n_rows = [], 0.0, 0.0, 0
    per_well_inputs = {}
    for i, meta in enumerate(metas):
        wid = meta["well_id"]
        row_bin = _row_bins(meta, cfg)
        y_row = np.asarray(meta["TVT"], dtype=np.float64) - float(meta["tvt0"])
        # the model's own prediction, expanded exactly as the pipeline scores it
        tp = seq_NN_train.expand_target_prediction(pred[i], meta, cfg).astype(np.float64)
        ok = np.isfinite(y_row)
        model_sse += float(((tp - y_row)[ok] ** 2).sum())
        n_rows += int(ok.sum())
        mu_row = (prob[i][row_bin] * x).sum(axis=-1)
        sanity_gap = max(sanity_gap, float(np.nanmax(np.abs(mu_row - tp))))
        logp = np.log(np.clip(prob[i], 1e-30, None))
        w = pa.analyse_well(wid, logp, row_bin, y_row, x)
        wells.append(w)
        per_well_inputs[wid] = (row_bin, y_row, np.asarray(meta.get("geo_prior_TVT"), dtype=np.float64) - float(meta["tvt0"]) if meta.get("geo_prior_TVT") is not None else None, float(meta["tvt0"]), i)

    ranked = sorted(wells, key=lambda w: w["sse"]["mean"], reverse=True)
    worst = tuple(w["well_id"] for w in ranked[:N_WORST])
    med = len(ranked) // 2
    controls = tuple(w["well_id"] for w in ranked[med - N_CONTROL // 2: med + (N_CONTROL - N_CONTROL // 2)])
    focus = tuple(dict.fromkeys(HARD_WELLS + worst + controls))

    report = pa.build_report(wells, focus=HARD_WELLS)
    report["worst_this_run"] = list(worst)
    report["controls"] = list(controls)
    ref = None
    ref_path = Path(_EXTRA["models_dir"] or "") / holdout.METRICS_FILENAME
    if ref_path.is_file():
        with ref_path.open(encoding="utf-8") as fh:
            ref = json.load(fh).get("pooled_rmse_raw")
    report["sanity"] = {
        "model_pred_pooled_rmse_raw": math.sqrt(model_sse / max(n_rows, 1)),
        "run_reported_pooled_rmse_raw": ref,
        "max_abs_gap_model_pred_vs_posterior_mean_ft": sanity_gap,
        "note": "mean decode must equal the model's own prediction (gap ~0) and the run's raw pooled RMSE",
    }

    out = cfg.output_dir
    with (out / "posterior_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    import pandas as pd

    rows = []
    for w in wells:
        s = w["sse"]["mean"]
        r = max(w["n_rows"], 1)
        rows.append({
            "well_id": w["well_id"], "rows": w["n_rows"],
            "rmse_mean": math.sqrt(s / r),
            "rmse_argmax": math.sqrt(w["sse"]["argmax"] / r),
            "rmse_dominant_mode": math.sqrt(w["sse"]["dominant_mode"] / r),
            "rmse_oracle_mean_or_mode": math.sqrt(w["sse"]["oracle_mean_or_mode"] / r),
            "rmse_minus_offset": math.sqrt(w["sse"]["minus_offset"] / r),
            "rmse_minus_line": math.sqrt(w["sse"]["minus_line"] / r),
            "line_slope_ft_per_kft": w["line_slope_ft_per_kft"],
            "multimodal_row_share": w["multimodal_rows"] / r,
            **{f"sse_share_{c}": w["cat_sse"][c] / max(s, 1e-12) for c in pa.CATEGORIES},
            "hard_well": w["well_id"] in HARD_WELLS,
        })
    pd.DataFrame(rows).sort_values("rmse_mean", ascending=False).to_csv(out / "posterior_wells.csv", index=False)

    cols = []
    for w in wells:
        for c in w["columns"]:
            cols.append({"well_id": w["well_id"], **c})
    pd.DataFrame(cols).to_parquet(out / "posterior_columns.parquet")

    _save_posteriors(out / "posterior_focus.npz", focus, per_well_inputs, prob, x)
    if _EXTRA["save_all"]:
        every = tuple(w["well_id"] for w in wells)
        _save_posteriors(out / "posterior_all.npz", every, per_well_inputs, prob, x)
        log(f"saved posteriors for all {len(every)} wells: {out / 'posterior_all.npz'}")

    # headline
    pr = report["pooled_rmse"]
    log("")
    log(f"SANITY  model pred raw pooled RMSE {report['sanity']['model_pred_pooled_rmse_raw']:.4f}"
        f" (run reported {ref if ref is None else f'{ref:.4f}'});"
        f" max |pred - posterior mean| {sanity_gap:.4f} ft")
    log("POOLED RMSE BY DECODER (raw, no SG smoothing):")
    for k, v in pr.items():
        log(f"  {k:22s} {v:.4f}")
    log("SHARE OF POOLED SQUARED ERROR BY FAILURE TYPE:")
    for k, v in report["category_share_of_sse"].items():
        log(f"  {k:18s} {100 * v:5.1f}%")
    log(f"multimodal columns (>=2 humps), share of rows: {100 * report['multimodal_share_of_rows']:.1f}%")
    log("HARD WELLS:")
    for f in report["focus"]:
        top = max(f["sse_by_category"].items(), key=lambda kv: kv[1])
        log(f"  {f['well_id']}  rmse {f['rmse_mean']:6.2f}  argmax {f['rmse_argmax']:6.2f}"
            f"  oracle-mode {f['rmse_oracle_mode']:6.2f}  minus-line {f['rmse_minus_line']:6.2f}"
            f"  multimodal rows {100 * f['multimodal_row_share']:4.1f}%  mostly {top[0]} ({100 * top[1]:.0f}%)")
    if report["focus_missing"]:
        log(f"  not in this holdout: {report['focus_missing']}")
    log(f"saved: {out / 'posterior_report.json'}")
    return report


def main(argv=None):
    parser = rescore.build_parser()
    parser.add_argument("--save-all", action="store_true",
                        help="also write posterior_all.npz with every holdout well's posterior")
    args = parser.parse_args(argv)
    if args.offline_timm:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if args.tta is not None and args.tta > 1:
        raise SystemExit("--tta is not supported here: phase-shifted grids have different columns")
    _EXTRA["models_dir"] = str(args.models_dir.expanduser().resolve())
    _EXTRA["save_all"] = bool(args.save_all)
    rescore.score = collect  # everything else in rescore.run is reused unchanged
    return rescore.run(args)


if __name__ == "__main__":
    sys.exit(main())
