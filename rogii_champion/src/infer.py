"""Test-time inference: checkpoints -> submission.csv.

At test time every training label is legitimately available, so the feature
provider is built with all training wells allowed. The test wells contribute
their own visible prefixes to the XY cloud and the sibling bank, which is also
allowed -- and is what makes the sibling reference stronger at test time than
it looks in CV.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .data import (find_sample_submission, load_split, split_dir,
                   write_submission)
from .decode import predict_well
from .features import FeatureProvider
from .model import build_model


def load_checkpoints(run_dir, device: str = "cuda", limit: int = None):
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob("split*_seed*_fold*.pt"))
    if limit:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"no checkpoints in {run_dir}")
    return files


def run_inference(cfg, run_dir, data_root, out_csv, device: str = "cuda",
                  train_dir: str = "train", test_dir: str = "test",
                  limit_ckpt: int = None, verbose: bool = True,
                  details_path=None):
    data_root = Path(data_root)
    test_wells = load_split(data_root / test_dir)
    train_wells = load_split(data_root / train_dir, require_tvt=True) \
        if (data_root / train_dir).is_dir() else []

    provider = FeatureProvider(
        train_wells + test_wells,
        allowed_full={w.well_id for w in train_wells},
        use_pf=cfg.use_pf, use_xy=cfg.use_xy, use_sibling=cfg.use_sibling,
        correct_typewell=cfg.correct_typewell)

    spec = cfg.canvas_spec()
    files = load_checkpoints(run_dir, device, limit_ckpt)
    acc, cnt = {}, {}
    for i, f in enumerate(files):
        ck = torch.load(f, map_location="cpu", weights_only=False)
        model = build_model(cfg, ck.get("in_chans", spec.in_chans),
                            pretrained=False).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        for w in test_wells:
            try:
                pred = predict_well(model, w, spec, cfg, provider, device,
                                    n_phases=cfg.n_phases, reanchor=cfg.reanchor,
                                    adaptive_canvas=cfg.adaptive_canvas,
                                    dp_weight=cfg.dp_weight)
            except Exception as e:              # never lose a well to one model
                if verbose:
                    print(f"  ! {w.well_id} failed on {f.name}: {e}")
                continue
            acc[w.well_id] = pred if w.well_id not in acc else acc[w.well_id] + pred
            cnt[w.well_id] = cnt.get(w.well_id, 0) + 1
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        if verbose:
            print(f"[infer] {i + 1}/{len(files)} checkpoints")

    preds = {}
    n_fallback = 0
    for w in test_wells:
        m = w.eval_mask
        if w.well_id in acc and cnt[w.well_id] > 0:
            p = acc[w.well_id] / cnt[w.well_id]
        else:                                   # constant extrapolation fallback
            p = np.full(w.n, w.anchor_tvt)
            n_fallback += 1
        preds[w.well_id] = (w.row_idx[m], p[m])
    if verbose and n_fallback:
        print(f"[infer] {n_fallback} wells fell back to constant TVT")

    if details_path is not None:
        write_details(preds, test_wells, train_wells, provider, cfg,
                      details_path, verbose=verbose)
    return write_submission(preds, find_sample_submission(data_root), out_csv)


def write_details(preds, test_wells, train_wells, provider, cfg, details_path,
                  verbose: bool = True):
    """Per-row predictions plus the per-well XY-neighbourhood statistics.

    The statistics are what the ensemble notebook routes on: 1st place replaced
    XY-based predictions with GR + z_diff predictions wherever a well's
    neighbourhood fell outside the 95th percentile of the training
    distribution. Thresholds are fitted on the training wells here so the
    ensemble step needs nothing but these files.
    """
    import pandas as pd

    from .xy_neighbor import fit_safety_thresholds, xy_safety_mask

    rows = []
    for wid, (idx, vals) in preds.items():
        rows.append(pd.DataFrame(dict(
            id=[f"{wid}_{int(i)}" for i in idx], well_id=wid,
            row_idx=idx.astype(int), tvt=vals)))
    det = pd.concat(rows, ignore_index=True)

    safe = {}
    if cfg.use_xy and provider is not None and provider.cloud is not None:
        tr_stats = {w.well_id: provider.xy_stats(w) for w in train_wells}
        thr = fit_safety_thresholds(tr_stats) if tr_stats else {}
        te_stats = {w.well_id: provider.xy_stats(w) for w in test_wells}
        safe = xy_safety_mask(te_stats, thr) if thr else {}
        stats_df = pd.DataFrame(te_stats).T
        stats_df["xy_safe"] = [safe.get(i, True) for i in stats_df.index]
        stats_df.index.name = "well_id"
    else:
        stats_df = pd.DataFrame(index=pd.Index([w.well_id for w in test_wells],
                                               name="well_id"))
        stats_df["xy_safe"] = True

    details_path = Path(details_path)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        det.to_parquet(details_path, index=False)
    except Exception:
        details_path = details_path.with_suffix(".csv")
        det.to_csv(details_path, index=False)
    stats_csv = details_path.with_name(
        details_path.stem.replace("details", "wellstats") + ".csv")
    stats_df.to_csv(stats_csv)
    if verbose:
        n_bad = int((~stats_df["xy_safe"]).sum())
        print(f"[infer] wrote {details_path.name} and {stats_csv.name} "
              f"({n_bad}/{len(stats_df)} wells outside the XY-safe region)")
