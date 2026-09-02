#!/usr/bin/env python3
"""Well-level holdout evaluation for the archived sequence-NN snapshots.

The bundled ``data/test`` directory has no ``TVT`` column, so it cannot be
scored locally.  This entrypoint instead carves a scoreable holdout out of the
labelled ``data/train`` wells:

* the **first 80%** of the sorted well ids are used for training;
* the **remaining 20%** are held out and never seen by the model, the geo
  prior, or the checkpoint-selection metric;
* the model predicts the masked ``TVT_input`` suffix of every holdout well and
  the **pooled RMSE** is computed over the concatenation of all those suffix
  rows -- one global ``sqrt(mean((TVT - TVT_pred)**2))``, not a mean of
  per-well or per-fold RMSEs.

Everything that defines the recipe -- data loading, augmentation, model, loss,
checkpoint selection, inference -- still comes from the untouched snapshot in
``reference_results/<ID>/``.  Only the cross-validation split is replaced, by
substituting ``seq_NN_train.make_cv_splits`` with a single fixed split.  Run it
from this directory, for example::

    python seq_NN_holdout_eval.py --id 0801_V2 --device cuda

Note that the recipes' target normalisation constants (``cfg.target_stats``)
were fitted on the full original training set, so they carry a small amount of
information about the holdout wells.  Everything the run learns is split-safe.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

import seq_NN_main_reproduce as repro

DEFAULT_TRAIN_FRAC = 0.8

RESULTS_DIRNAME = "results"

SPLIT_FILENAME = "holdout_split.csv"
PRED_FILENAME = "holdout_predictions.pqt"
METRICS_FILENAME = "holdout_metrics.json"


def split_wells(well_ids, train_frac):
    """Contiguous first-``train_frac`` / remainder split of the sorted well ids."""

    well_ids = list(well_ids)
    total = len(well_ids)
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"--train-frac must be strictly between 0 and 1, got {train_frac}")
    n_train = int(np.floor(total * train_frac))
    if n_train < 1 or n_train >= total:
        raise ValueError(
            f"train_frac={train_frac} leaves {n_train} training and "
            f"{total - n_train} holdout wells out of {total}; both sides must be non-empty"
        )
    return well_ids[:n_train], well_ids[n_train:]


def install_holdout_split(seq_NN_train, n_train, n_total, repeats):
    """Replace k-fold construction with one fixed train/holdout split.

    ``kfold_training`` resolves ``make_cv_splits`` as a module global, so
    rebinding it here changes the split without editing the snapshot.  Returning
    the same split ``repeats`` times makes the snapshot train that many
    independently seeded models and average their holdout predictions, which is
    what ``average_repeated_oof_predictions`` already does for ``cv_repeats``.
    """

    train_index = np.arange(n_train, dtype=int)
    val_index = np.arange(n_train, n_total, dtype=int)

    def make_cv_splits(well_ids, cfg, log=print):
        if len(well_ids) != n_total:
            raise ValueError(
                f"holdout split was built for {n_total} wells but received {len(well_ids)}"
            )
        log(
            f"holdout split: train_wells={len(train_index):,}, "
            f"holdout_wells={len(val_index):,}, repeats={repeats}"
        )
        return [(train_index.copy(), val_index.copy()) for _ in range(repeats)]

    seq_NN_train.make_cv_splits = make_cv_splits


def write_split_file(output_dir, train_wells, holdout_wells):
    split_df = pd.DataFrame(
        {
            "well_id": list(train_wells) + list(holdout_wells),
            "split": ["train"] * len(train_wells) + ["holdout"] * len(holdout_wells),
        }
    )
    split_path = output_dir / SPLIT_FILENAME
    split_df.to_csv(split_path, index=False)
    return split_path


def collect_metrics(oof_df, seq_NN_train, train_wells, holdout_wells, cfg, train_frac):
    """Pooled RMSE over every holdout suffix row, plus per-well diagnostics."""

    metrics = {
        "train_frac": float(train_frac),
        "train_wells": len(train_wells),
        "holdout_wells": len(holdout_wells),
        "holdout_wells_scored": int(oof_df["well_id"].nunique()),
        "holdout_rows_scored": int(len(oof_df)),
        "models_averaged": int(seq_NN_train.resolve_cv_repeats(cfg)),
        "pooled_rmse": float(seq_NN_train.score_prediction_df(oof_df)),
        "well_rmse": seq_NN_train.well_rmse_summary(oof_df),
    }
    if "TVT_pred_raw" in oof_df.columns:
        metrics["pooled_rmse_raw"] = float(
            seq_NN_train.score_prediction_df(oof_df, pred_col="TVT_pred_raw")
        )
        metrics["well_rmse_raw"] = seq_NN_train.well_rmse_summary(
            oof_df, pred_col="TVT_pred_raw"
        )
    for sidecar, name in (("geo_prior_TVT", "geo_prior"), ("PF_TVT", "PF")):
        if sidecar in oof_df.columns and oof_df[sidecar].notna().all():
            metrics[f"pooled_rmse_{name}_baseline"] = float(
                seq_NN_train.score_prediction_df(oof_df, pred_col=sidecar)
            )
    return metrics


def run_holdout(cfg, main_module, seq_NN_train, train_wells, holdout_wells, train_frac):
    log = main_module.log
    all_wells = list(train_wells) + list(holdout_wells)

    main_module.log_section("setup")
    main_module.log_cfg_summary(cfg, all_wells, [], include_test=False)
    log(f"holdout mode: train_frac={train_frac}")
    log(f"train wells: {len(train_wells):,} ({train_wells[0]} .. {train_wells[-1]})")
    log(f"holdout wells: {len(holdout_wells):,} ({holdout_wells[0]} .. {holdout_wells[-1]})")
    split_path = write_split_file(cfg.output_dir, train_wells, holdout_wells)
    log(f"saved holdout split: {split_path}")
    main_module.save_cfg(cfg)

    main_module.log_section("model training")
    models, oof_df = seq_NN_train.kfold_training(well_ids=all_wells, cfg=cfg, log=log)
    if oof_df is None or len(oof_df) == 0:
        raise RuntimeError("holdout training produced no scored predictions")

    main_module.log_section("save models")
    main_module.save_models(models=models, cfg=cfg)

    main_module.log_section("holdout scoring")
    pred_path = cfg.output_dir / PRED_FILENAME
    oof_df.to_parquet(pred_path)
    log(f"saved holdout predictions: {pred_path}")

    metrics = collect_metrics(
        oof_df, seq_NN_train, train_wells, holdout_wells, cfg, train_frac
    )
    scored = metrics["holdout_wells_scored"]
    if scored != len(holdout_wells):
        raise RuntimeError(
            f"expected predictions for {len(holdout_wells)} holdout wells, got {scored}"
        )
    metrics_path = cfg.output_dir / METRICS_FILENAME
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    log(f"saved holdout metrics: {metrics_path}")

    log("")
    log(
        f"HOLDOUT POOLED RMSE: {metrics['pooled_rmse']:.4f} ft "
        f"over {metrics['holdout_rows_scored']:,} suffix rows "
        f"from {scored:,} wells"
    )
    if "pooled_rmse_raw" in metrics:
        log(f"HOLDOUT POOLED RMSE (pre-SG-smoothing): {metrics['pooled_rmse_raw']:.4f} ft")
    for key in ("pooled_rmse_geo_prior_baseline", "pooled_rmse_PF_baseline"):
        if key in metrics:
            log(f"  baseline {key}: {metrics[key]:.4f} ft")
    log(
        "HOLDOUT per-well RMSE: "
        f"{seq_NN_train.format_well_rmse_summary(metrics['well_rmse'])}"
    )
    return metrics


def run(args):
    repro.check_runtime_dependencies()
    settings, settings_path = repro.load_settings(args.settings)
    repro.rebuild_geo_map(settings)

    canonical_requested = repro.canonical_run_id(args.id)
    source_dir = repro.resolve_source_dir(args.source_dir, settings, canonical_requested)
    cfg_module, main_module = repro.import_snapshot(source_dir)
    if args.cfg_name is None:
        canonical, recipe, cfg = repro.select_recipe(canonical_requested, cfg_module)
    else:
        # Fork workflow: --source-dir points at an edited copy of a snapshot and
        # --cfg-name selects a SEQ_TRAIN_CFGS entry that the archived recipe map
        # knows nothing about, so the registry is resolved here instead.
        from copy import deepcopy

        canonical = canonical_requested
        recipe = repro.RUN_RECIPES[canonical]
        registry = dict(cfg_module.SEQ_TRAIN_CFGS)
        if args.cfg_name not in registry:
            available = ", ".join(sorted(registry))
            raise ValueError(
                f"{source_dir} has no SEQ_TRAIN_CFGS entry {args.cfg_name!r}; "
                f"available entries: {available}"
            )
        cfg = deepcopy(registry[args.cfg_name])
        cfg.refresh()
    # Deliberately NOT recipe["archived_f"]: 0803_V2 archived f=1 merely because the
    # author's directory already existed, and inheriting it would let that one recipe
    # silently overwrite a finished holdout run.  Always refuse unless --f 1 is given.
    cfg.f = 0
    run_label = args.cfg_name or canonical

    paths = settings["paths"]
    if args.output_dir is None:
        # Deliberately not SETTINGS.paths.output_dir: that tree holds the 15-fold
        # CV reproductions, and a holdout run writes the same filenames from a
        # different split.  Keep the two apart.
        output_dir = paths["project_root"] / RESULTS_DIRNAME / run_label
    else:
        output_dir = args.output_dir.expanduser().resolve()
    repro.apply_settings_paths(cfg, settings, output_dir)
    repro.apply_cli_overrides(cfg, args)
    if args.unet_pretrained is not None:
        # Escape hatch for machines that cannot reach the timm/ConvNeXt weights.
        # It changes the recipe, so it is only for smoke and offline debugging.
        cfg.model_cfg["unet_pretrained"] = bool(args.unet_pretrained)
    if args.unet_arch is not None:
        # unet_arch='unet' builds the plain U-Net and never touches timm or the
        # HuggingFace hub, which is the only way to smoke-test fully offline.
        cfg.model_cfg["unet_arch"] = args.unet_arch
    if args.offline_timm:
        # The recipes name the backbone as 'hf_hub:timm/<tag>', which makes timm
        # fetch config.json from huggingface.co even when the weights are already
        # cached.  The bare '<tag>' resolves the identical repository through
        # timm's own registry, so the same pretrained weights load from the local
        # cache with no network access.
        name = str(cfg.model_cfg.get("unet_timm_model_name", ""))
        if name.startswith("hf_hub:timm/"):
            cfg.model_cfg["unet_timm_model_name"] = name[len("hf_hub:timm/"):]
    if args.smoke and args.well_limit is not None:
        # apply_cli_overrides hardcodes well_limit=2 for smoke runs, which leaves
        # a single training well and a degenerate geo prior.  Keep our own value.
        cfg.well_limit = int(args.well_limit)

    # The holdout split supersedes k-fold geometry: fold_count is cosmetic from
    # here on, and cv_repeats becomes the seed-ensemble size on the same 80%.
    cfg.cv_split_mode = "holdout"
    cfg.fold_count = 1
    cfg.refresh()

    if not cfg.train_path.is_dir():
        raise FileNotFoundError(f"train directory does not exist: {cfg.train_path}")
    all_wells = main_module.discover_well_ids(cfg.train_path)
    if not all_wells:
        raise ValueError(f"no train wells found under {cfg.train_path}")
    if cfg.well_limit is not None:
        all_wells = all_wells[: cfg.well_limit]
    # kfold_training re-applies well_limit nowhere, but make_loader and the PF
    # cache both key off the ids we pass, so slice once here and then clear it.
    cfg.well_limit = None
    train_wells, holdout_wells = split_wells(all_wells, args.train_frac)

    if str(cfg.device).startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"cfg.device={cfg.device!r}, but torch.cuda.is_available() is False; "
                "install the pinned CUDA build and expose a compatible NVIDIA GPU"
            )

    repeats = main_module.seq_NN_train.resolve_cv_repeats(cfg)
    install_holdout_split(
        main_module.seq_NN_train,
        n_train=len(train_wells),
        n_total=len(train_wells) + len(holdout_wells),
        repeats=repeats,
    )

    repro._log(
        f"recipe={canonical}; cfg_name={args.cfg_name or '<archived default>'}; "
        f"source_dir={source_dir}; settings={settings_path}"
    )
    repro._log(
        f"train_wells={len(train_wells):,}; holdout_wells={len(holdout_wells):,}; "
        f"models={repeats}; output_dir={cfg.output_dir}"
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
                main_module.log(
                    f"WARNING: output directory already exists and f=1 was set: {cfg.output_dir}"
                )
            run_holdout(
                cfg,
                main_module,
                main_module.seq_NN_train,
                train_wells,
                holdout_wells,
                args.train_frac,
            )
            main_module.log(f"saved print log: {log_path}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Train one archived sequence-NN recipe on the first 80% of the labelled "
            "wells and report the pooled RMSE on the remaining 20%."
        )
    )
    parser.add_argument(
        "--id", required=True, choices=sorted(repro.ID_ALIASES),
        help="Recipe ID, e.g. 0801_V2.",
    )
    parser.add_argument(
        "--train-frac", type=float, default=DEFAULT_TRAIN_FRAC,
        help="Fraction of the sorted wells used for training (default 0.8).",
    )
    parser.add_argument(
        "--settings", type=Path,
        help="SETTINGS.json; auto-discovered beside this script when omitted.",
    )
    parser.add_argument(
        "--source-dir", type=Path,
        help="Directory containing the seq_NN_*.py pipeline to run. Point it at your own "
             "copy of a snapshot to train a modified pipeline.",
    )
    parser.add_argument(
        "--cfg-name",
        help="Name of a SEQ_TRAIN_CFGS entry in --source-dir's seq_NN_cfg.py, instead of "
             "the config the archived recipe used. Also becomes the results/ subdirectory.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Exact output directory; default is results/<cfg-name or id> under the package root.",
    )
    parser.add_argument("--device", help="Torch device override, such as cuda or cpu.")
    parser.add_argument(
        "--well-limit", type=int,
        help="Use only the first N wells before splitting (smoke/debug only).",
    )
    parser.add_argument("--epochs", type=int, help="Epoch override; omitted means the archived value.")
    parser.add_argument(
        "--repeats", dest="cv_repeats", type=int,
        help="Train this many independently seeded models on the same 80%% and average "
             "their holdout predictions before scoring (default 1).",
    )
    parser.add_argument("--batch-size", type=int, help="Training batch-size override.")
    parser.add_argument("--val-batch-size", type=int, help="Validation/inference batch-size override.")
    parser.add_argument("--num-workers", type=int, help="DataLoader worker-count override.")
    parser.add_argument("--grad-accum-steps", type=int, help="Gradient accumulation override.")
    parser.add_argument(
        "--persistent-workers", type=int, choices=[0, 1],
        help="DataLoader persistent worker override.",
    )
    parser.add_argument("--f", type=int, choices=[0, 1], help="Allow writing an existing output directory.")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny few-well, one-epoch smoke job.")
    parser.add_argument(
        "--unet-pretrained", type=int, choices=[0, 1],
        help="Override pretrained timm backbone loading. Pass 0 to build the U-Net from "
             "random weights when the timm checkpoint is unreachable; this changes the "
             "recipe and is for smoke/offline debugging only.",
    )
    parser.add_argument(
        "--offline-timm", action="store_true",
        help="Load the pretrained ConvNeXt backbone from the local HuggingFace cache "
             "without contacting huggingface.co. Same weights, same recipe; use it "
             "when the machine cannot verify the huggingface.co TLS certificate.",
    )
    parser.add_argument(
        "--unet-arch", choices=["unet", "convnext_small"],
        help="Override the 2D U-Net architecture. Pass 'unet' to avoid timm and the "
             "HuggingFace hub entirely; this changes the recipe and is for "
             "smoke/offline debugging only.",
    )
    # Accepted so repro.apply_cli_overrides finds every attribute it reads.
    parser.add_argument("--fold-count", type=int, help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.offline_timm:
        # Must precede the timm/huggingface_hub import in check_runtime_dependencies,
        # because huggingface_hub snapshots this flag at import time.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if args.cv_repeats is None:
        args.cv_repeats = 1
    if args.smoke and args.well_limit is None:
        # apply_cli_overrides would set well_limit=2, which cannot be split 80/20.
        args.well_limit = 10
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
