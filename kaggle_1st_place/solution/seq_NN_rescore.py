#!/usr/bin/env python3
"""Re-score an already-trained holdout run, optionally with MD-phase TTA.

``seq_NN_holdout_eval.py`` trains and scores in one pass, which is hours of GPU
time.  Some changes touch only inference -- averaging the prediction over
several MD column-grid phases is the motivating one -- and for those the
existing ``models.pkl`` is all that is needed.  This entrypoint rebuilds the
same 80/20 split, the same fold-safe geo prior and the same loader as the
training run did, loads the saved models, predicts, and writes the same
``holdout_predictions.pqt`` / ``holdout_metrics.json`` pair, so the output drops
straight into ``seq_NN_robust_compare.py``.

The saved models are plain pickled modules, so ``--source-dir`` has to point at
a pipeline whose model classes match the ones the run was trained with.  Pass
the patched fork to get MD-phase support::

    python seq_NN_rescore.py --id 0801_V2 --models-dir results/0801_V2 \\
        --source-dir experiments/bilzard --tta 8 \\
        --output-dir results/0801_V2_tta8 --device cuda --offline-timm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path

import numpy as np

import seq_NN_holdout_eval as holdout
import seq_NN_main_reproduce as repro


def resolve_cfg(args, cfg_module):
    canonical = repro.canonical_run_id(args.id)
    if args.cfg_name is None:
        canonical, _recipe, cfg = repro.select_recipe(canonical, cfg_module)
        return canonical, cfg
    registry = dict(cfg_module.SEQ_TRAIN_CFGS)
    if args.cfg_name not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(
            f"no SEQ_TRAIN_CFGS entry {args.cfg_name!r}; available entries: {available}"
        )
    cfg = deepcopy(registry[args.cfg_name])
    cfg.refresh()
    return canonical, cfg


def score(cfg, main_module, seq_NN_train, models, train_wells, holdout_wells, train_frac):
    log = main_module.log
    main_module.log_section("setup")
    main_module.log_cfg_summary(cfg, list(train_wells) + list(holdout_wells), [], include_test=False)
    log(f"rescore mode: train_frac={train_frac}, models={len(models):,}")
    log(f"MD phases: {seq_NN_train.resolve_md_phases(cfg)}")
    split_path = holdout.write_split_file(cfg.output_dir, train_wells, holdout_wells)
    log(f"saved holdout split: {split_path}")
    main_module.save_cfg(cfg)

    main_module.log_section("holdout inference")
    geo_cfg = seq_NN_train.resolve_geo_prior_cfg(cfg)
    val_geo_prior, val_prior_summary = seq_NN_train.make_geo_prior_for_wells(
        cfg.train_path,
        cfg.train_path,
        list(train_wells),
        list(holdout_wells),
        geo_cfg,
        exclude_query_from_support=False,
    )
    prior_rmse = val_prior_summary.get("rmse")
    log(
        f"geo_prior {geo_cfg.method}: support_wells={len(train_wells):,}, "
        f"query_wells={len(holdout_wells):,}, "
        f"rmse={'NA' if prior_rmse is None else f'{prior_rmse:.4f}'}, "
        f"elapsed={val_prior_summary['elapsed_sec']:.2f}s"
    )
    pf_cache_dir = seq_NN_train.prepare_pf_cache_if_needed(
        "train", cfg.train_path, list(train_wells) + list(holdout_wells), cfg, log
    )
    loader = seq_NN_train.make_loader(
        path=cfg.train_path,
        well_ids=list(holdout_wells),
        cfg=cfg,
        training=True,
        simulation=False,
        shuffle=False,
        batch_size=cfg.val_batch_size,
        seed=cfg.seed + 2,
        pf_cache_dir=pf_cache_dir,
        pf_sample_cache_dir=None,
        geo_prior=val_geo_prior,
        drop_last=False,
    )

    import time
    import torch

    cuda = str(cfg.device).startswith("cuda")
    if cuda:
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    if len(seq_NN_train.resolve_md_phases(cfg)) > 1:
        pred_df = seq_NN_train.predict_with_md_phase_tta(
            models,
            loader,
            cfg,
            include_target=True,
            apply_smooth=False,
            log=log,
        )
    else:
        pred_sum = None
        metas = None
        for model in models.values():
            pred, metas = seq_NN_train.predict_one_model(model, loader, cfg, return_meta=True)
            contribution = pred / float(len(models))
            pred_sum = contribution if pred_sum is None else pred_sum + contribution
        pred_df = seq_NN_train.make_prediction_df(
            pred_sum, metas, cfg, include_target=True, apply_smooth=False
        )
    if cuda:
        torch.cuda.synchronize()
    infer_sec = time.perf_counter() - t_start
    log(
        f"inference wall time: {infer_sec:.2f}s for {len(holdout_wells):,} wells "
        f"({infer_sec / max(len(holdout_wells), 1) * 1000:.1f} ms/well, data loading and "
        f"post-processing included, geo prior excluded)"
    )
    if getattr(cfg, "pred_sg_smooth", False):
        pred_df = seq_NN_train.apply_pred_sg_smooth(pred_df, cfg)
    pred_df = seq_NN_train.add_geo_prior_diagnostic_columns(pred_df, val_geo_prior)
    pred_df = seq_NN_train.add_geo_prior_well_summary_columns(pred_df)
    pred_df = seq_NN_train.add_typewell_matched_gr_columns(pred_df, cfg.train_path)

    main_module.log_section("holdout scoring")
    pred_path = cfg.output_dir / holdout.PRED_FILENAME
    pred_df.to_parquet(pred_path)
    log(f"saved holdout predictions: {pred_path}")

    metrics = holdout.collect_metrics(
        pred_df, seq_NN_train, train_wells, holdout_wells, cfg, train_frac
    )
    metrics["models_averaged"] = int(len(models))
    metrics["md_phases"] = list(seq_NN_train.resolve_md_phases(cfg))
    metrics["inference_sec"] = float(infer_sec)
    metrics["reparameterised"] = bool(getattr(cfg, "reparam_at_inference", False))
    scored = metrics["holdout_wells_scored"]
    if scored != len(holdout_wells):
        raise RuntimeError(
            f"expected predictions for {len(holdout_wells)} holdout wells, got {scored}"
        )
    metrics_path = cfg.output_dir / holdout.METRICS_FILENAME
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    log(f"saved holdout metrics: {metrics_path}")

    log("")
    log(
        f"HOLDOUT POOLED RMSE: {metrics['pooled_rmse']:.4f} ft "
        f"over {metrics['holdout_rows_scored']:,} suffix rows from {scored:,} wells"
    )
    if "pooled_rmse_raw" in metrics:
        log(f"HOLDOUT POOLED RMSE (pre-SG-smoothing): {metrics['pooled_rmse_raw']:.4f} ft")
    log(
        "HOLDOUT per-well RMSE: "
        f"{seq_NN_train.format_well_rmse_summary(metrics['well_rmse'])}"
    )
    return metrics


def run(args):
    repro.check_runtime_dependencies()
    settings, settings_path = repro.load_settings(args.settings)
    repro.rebuild_geo_map(settings)

    models_dir = args.models_dir.expanduser().resolve()
    source_arg = args.source_dir if args.source_dir is not None else models_dir
    source_dir = repro.resolve_source_dir(
        source_arg, settings, repro.canonical_run_id(args.id)
    )
    cfg_module, main_module = repro.import_snapshot(source_dir)
    seq_NN_train = main_module.seq_NN_train
    if args.tta is not None and args.tta > 1 and not hasattr(seq_NN_train, "resolve_md_phases"):
        raise RuntimeError(
            f"{source_dir} has no MD-phase support; pass --source-dir experiments/bilzard "
            "(or another fork that carries it) to use --tta"
        )

    canonical, cfg = resolve_cfg(args, cfg_module)
    cfg.f = 0
    run_label = args.cfg_name or canonical
    paths = settings["paths"]
    if args.output_dir is None:
        output_dir = paths["project_root"] / holdout.RESULTS_DIRNAME / f"{run_label}_rescore"
    else:
        output_dir = args.output_dir.expanduser().resolve()
    repro.apply_settings_paths(cfg, settings, output_dir)
    repro.apply_cli_overrides(cfg, args)
    if args.offline_timm:
        name = str(cfg.model_cfg.get("unet_timm_model_name", ""))
        if name.startswith("hf_hub:timm/"):
            cfg.model_cfg["unet_timm_model_name"] = name[len("hf_hub:timm/"):]
    if args.tta is not None:
        cfg.md_phase_tta = int(args.tta)
    cfg.cv_split_mode = "holdout"
    cfg.fold_count = 1
    cfg.refresh()

    if not cfg.train_path.is_dir():
        raise FileNotFoundError(f"train directory does not exist: {cfg.train_path}")
    all_wells = main_module.discover_well_ids(cfg.train_path)
    if cfg.well_limit is not None:
        all_wells = all_wells[: cfg.well_limit]
    cfg.well_limit = None
    train_wells, holdout_wells = holdout.split_wells(all_wells, args.train_frac)

    if str(cfg.device).startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"cfg.device={cfg.device!r}, but torch.cuda.is_available() is False"
            )

    repro._log(
        f"recipe={canonical}; cfg_name={args.cfg_name or '<archived default>'}; "
        f"source_dir={source_dir}; models_dir={models_dir}; settings={settings_path}"
    )
    output_dir_existed = cfg.output_dir.exists()
    if not main_module.prepare_output_dir(cfg.output_dir, cfg.f):
        return 0
    repro.copy_snapshot_sources(source_dir, cfg.output_dir)

    log_path = cfg.output_dir / cfg.log_filename
    with log_path.open("w", encoding="utf-8") as log_file:
        tee = main_module.Tee(sys.stdout, log_file)
        with redirect_stdout(tee):
            if output_dir_existed and cfg.f == 1:
                main_module.log(f"WARNING: output directory already exists: {cfg.output_dir}")
            models = main_module.load_models(models_dir, cfg)
            cfg.reparam_at_inference = bool(args.reparam)
            if args.reparam:
                if not hasattr(seq_NN_train, "reparameterize_for_inference"):
                    raise RuntimeError(
                        f"{source_dir} has no reparameterize_for_inference; pass "
                        "--source-dir experiments/bilzard"
                    )
                for model in models.values():
                    n = seq_NN_train.reparameterize_for_inference(model, log=main_module.log)
                    if n == 0:
                        main_module.log("reparam: no multi-branch blocks found (nothing to collapse)")
            score(
                cfg,
                main_module,
                seq_NN_train,
                models,
                train_wells,
                holdout_wells,
                args.train_frac,
            )
            main_module.log(f"saved print log: {log_path}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="Re-score a trained holdout run's models.pkl, optionally with MD-phase TTA.",
    )
    parser.add_argument("--id", required=True, choices=sorted(repro.ID_ALIASES))
    parser.add_argument(
        "--models-dir", type=Path, required=True,
        help="Directory holding the trained models.pkl, e.g. results/0801_V2.",
    )
    parser.add_argument(
        "--source-dir", type=Path,
        help="Pipeline to run the inference with; defaults to --models-dir. Use the "
             "patched fork for --tta.",
    )
    parser.add_argument(
        "--cfg-name",
        help="SEQ_TRAIN_CFGS entry to rebuild the config from, instead of the archived recipe.",
    )
    parser.add_argument(
        "--tta", type=int,
        help="Average the prediction over this many MD column-grid phases (8 matches "
             "the 2nd-place solution). Omit to keep the single archived grid.",
    )
    parser.add_argument(
        "--reparam", action="store_true",
        help="Collapse MobileOne/FastViT training-time branches before inference "
             "(same function up to float rounding, faster). No effect on ConvNeXt.",
    )
    parser.add_argument("--train-frac", type=float, default=holdout.DEFAULT_TRAIN_FRAC)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--well-limit", type=int)
    parser.add_argument("--val-batch-size", type=int)
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader workers (default 0). On Windows, starting workers costs ~13-15 s, more "
             "than building all 155 holdout inputs in the main process (~8 ms each); predictions "
             "are identical either way.",
    )
    parser.add_argument("--f", type=int, choices=[0, 1])
    parser.add_argument("--offline-timm", action="store_true")
    # Accepted so repro.apply_cli_overrides finds every attribute it reads.
    for suppressed in ("--epochs", "--fold-count", "--batch-size", "--grad-accum-steps"):
        parser.add_argument(suppressed, type=int, help=argparse.SUPPRESS)
    parser.add_argument("--repeats", dest="cv_repeats", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--persistent-workers", type=int, choices=[0, 1], help=argparse.SUPPRESS
    )
    parser.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.offline_timm:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
