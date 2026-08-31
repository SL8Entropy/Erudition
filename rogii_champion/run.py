"""CLI.

`--data-root` is optional everywhere: the folder holding `train/` and `test/`
is found automatically. Use `python -u` for long runs or stdout stays buffered.

    python run.py list
    python -u run.py train    --variant base --out runs --folds 0 --num-workers 3
    python -u run.py oof      --variant v1_tta --weights runs/base --out runs
    python -u run.py compare  --base runs/base --treat runs/v1_tta
    python -u run.py ensemble --members base=runs/base,v4=runs/v4_refgr --route
    python -u run.py infer    --variant base --run-dir runs/base --out submission.csv
    python -u run.py benchmark --epochs 12 --num-workers 3   # all 5, one report
    python -u run.py blend    --details details_base.csv,details_v4_refgr.csv
    python smoke_test.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs import describe, get_config          # noqa: E402
from src.data import (find_data_root, group_kfold, load_split, save_json,  # noqa: E402
                      split_dir)
from src.ensemble import compare as ens_compare   # noqa: E402
from src.ensemble import blend, fit_weights, load_oof, route_blend  # noqa: E402
from src.features import FeatureProvider         # noqa: E402
from src.metrics import score_report             # noqa: E402
from src.model import build_model                # noqa: E402
from src.train import predict_wells, train_fold  # noqa: E402
from src.xy_neighbor import fit_safety_thresholds, xy_safety_mask  # noqa: E402


def _device(arg: str) -> str:
    if arg != "auto":
        return arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def _apply_overrides(cfg, args):
    for key in ("epochs", "batch_size", "lr", "num_workers", "backbone",
                "n_phases", "accum_steps"):
        v = getattr(args, key, None)
        if v is not None:
            setattr(cfg, key, v)
    if getattr(args, "no_pretrained", False):
        cfg.pretrained = False
    if getattr(args, "folds", None):
        cfg.folds = tuple(args.folds)
    if getattr(args, "seeds", None):
        cfg.seeds = tuple(args.seeds)
    if getattr(args, "splits", None):
        cfg.split_patterns = tuple(args.splits)
    return cfg


# ---------------------------------------------------------------------- #
def cmd_train(args):
    cfg = _apply_overrides(get_config(args.variant), args)
    root = find_data_root(args.data_root)
    wells = load_split(split_dir(root, args.train_dir), require_tvt=True)
    device = _device(args.device)
    out = Path(args.out) / cfg.name
    print(f"[run] variant={cfg.name}  wells={len(wells)}  device={device}")
    print(f"[run] {cfg.notes}\n")

    reports = []
    for split in cfg.split_patterns:
        for seed in cfg.seeds:
            for fold in cfg.folds:
                _, rep = train_fold(cfg, wells, fold, seed, split, out,
                                    device=device)
                reports.append(dict(split=split, seed=seed, fold=fold, **rep))
    save_json(reports, out / "reports.json")

    preds, _ = load_oof(out)
    rep = score_report(preds, wells)
    print(f"\n[run] {cfg.name} full OOF pooled RMSE = {rep['pooled_rmse']:.4f}")
    save_json(rep, out / "oof_report.json")


def cmd_oof(args):
    """Recompute OOF for existing weights under a different inference config.

    This is how the decode-only variant is measured: identical weights, only
    the TTA and re-anchoring change, so any difference is attributable.
    """
    cfg = _apply_overrides(get_config(args.variant), args)
    root = find_data_root(args.data_root)
    wells = load_split(split_dir(root, args.train_dir), require_tvt=True)
    device = _device(args.device)
    wdir = Path(args.weights)
    out = Path(args.out) / cfg.name if args.out else Path(args.weights).parent / cfg.name
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(wdir.glob("split*_seed*_fold*.pt"))
    if not files:
        raise FileNotFoundError(f"no checkpoints in {wdir}")
    ids = [w.well_id for w in wells]
    spec = cfg.canvas_spec()

    for f in files:
        parts = f.stem.split("_")
        split = int(parts[0][5:])
        fold = int(parts[2][4:])
        assign = group_kfold(ids, cfg.n_folds, seed=split)
        va = [w for w in wells if assign[w.well_id] == fold]
        allowed = {w.well_id for w in wells if assign[w.well_id] != fold}
        provider = FeatureProvider(wells, allowed_full=allowed,
                                   use_pf=cfg.use_pf, use_xy=cfg.use_xy,
                                   use_sibling=cfg.use_sibling,
                                   correct_typewell=cfg.correct_typewell)
        ck = torch.load(f, map_location="cpu", weights_only=False)
        model = build_model(cfg, ck.get("in_chans", spec.in_chans),
                            pretrained=False).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        oof = predict_wells(model, va, cfg, provider, device,
                            n_phases=cfg.n_phases, reanchor=cfg.reanchor,
                            adaptive_canvas=cfg.adaptive_canvas,
                            dp_weight=cfg.dp_weight)
        rep = score_report({k: v["pred"] for k, v in oof.items()}, va)
        print(f"[oof] {f.stem}: pooled RMSE {rep['pooled_rmse']:.4f}")
        np.savez_compressed(
            out / f"oof_{f.stem}.npz",
            **{f"pred__{k}": v["pred"] for k, v in oof.items()},
            **{f"sigma__{k}": (v["sigma"] if v["sigma"] is not None
                               else np.zeros(1)) for k, v in oof.items()})
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    preds, _ = load_oof(out)
    rep = score_report(preds, wells)
    print(f"\n[oof] {cfg.name} full OOF pooled RMSE = {rep['pooled_rmse']:.4f}")
    save_json(rep, out / "oof_report.json")


def cmd_compare(args):
    root = find_data_root(args.data_root)
    wells = load_split(split_dir(root, args.train_dir), require_tvt=True)
    base, _ = load_oof(args.base)
    treat, _ = load_oof(args.treat)
    common = set(base) & set(treat)
    wells = [w for w in wells if w.well_id in common]
    ens_compare(base, treat, wells,
                label=f"{Path(args.treat).name} vs {Path(args.base).name}",
                min_k=args.min_k)


def cmd_ensemble(args):
    root = find_data_root(args.data_root)
    wells = load_split(split_dir(root, args.train_dir), require_tvt=True)
    members, sigmas = {}, {}
    for item in args.members.split(","):
        name, path = item.split("=")
        members[name], sigmas[name] = load_oof(path)
    common = set.intersection(*[set(m) for m in members.values()])
    wells = [w for w in wells if w.well_id in common]

    for name, m in members.items():
        print(f"  {name:12s} pooled RMSE {score_report(m, wells)['pooled_rmse']:.4f}")

    eq = blend(members, {n: 1.0 for n in members})
    print(f"\n  equal weights  {score_report(eq, wells)['pooled_rmse']:.4f}")

    if args.fit_weights:
        r = fit_weights(members, wells)
        print(f"  fitted weights {r['weights']}")
        print(f"    in-sample {r['in_sample']:.4f}   nested {r['nested']:.4f}"
              f"   equal {r['equal']:.4f}")
        if r["nested"] > r["equal"]:
            print("    -> nested is worse than equal weights: keep equal.")

    if args.route:
        prov = FeatureProvider(wells, allowed_full={w.well_id for w in wells},
                               use_xy=True)
        stats = {w.well_id: prov.xy_stats(w) for w in wells}
        thr = fit_safety_thresholds(stats)
        safe = xy_safety_mask(stats, thr)
        n_bad = sum(1 for v in safe.values() if not v)
        gen = {n: 1.0 for n in members}
        fb = {n: (1.0 if "xy" not in n and "base" in n else 0.0) for n in members}
        if not any(fb.values()):
            fb = gen
        routed = route_blend(members, gen, fb, safe)
        print(f"\n  routed ({n_bad} wells to the fallback vector): "
              f"{score_report(routed, wells)['pooled_rmse']:.4f}")


def cmd_infer(args):
    from src.infer import run_inference
    cfg = _apply_overrides(get_config(args.variant), args)
    root = find_data_root(args.data_root)
    out = Path(args.out)
    run_inference(cfg, args.run_dir, root, out, device=_device(args.device),
                  train_dir=args.train_dir, test_dir=args.test_dir,
                  limit_ckpt=args.limit_ckpt,
                  details_path=out.with_name(f"details_{cfg.name}.csv"))


def cmd_benchmark(args):
    """Train each variant on the same 80% and score the same held-out 20%."""
    from src.benchmark import run_benchmark
    root = find_data_root(args.data_root)
    cfgs = [_apply_overrides(get_config(n), args)
            for n in args.variants.split(",")]
    run_benchmark(cfgs, root, Path(args.out), Path(args.report),
                  device=_device(args.device), holdout_fold=args.holdout_fold,
                  epochs=args.epochs)


def cmd_blend(args):
    """Blend several variants' test predictions, routed per well.

    Reads the `details_<variant>.csv` files written by `infer`. Wells whose
    XY neighbourhood falls outside the training distribution get the second
    weight vector -- 1st place's routing, which is what makes the XY-dependent
    members safe to include.
    """
    import pandas as pd

    from src.data import find_sample_submission

    root = find_data_root(args.data_root)
    frames, stats = {}, {}
    for path in args.details.split(","):
        p = Path(path)
        name = p.stem.replace("details_", "")
        frames[name] = (pd.read_parquet(p) if p.suffix == ".parquet"
                        else pd.read_csv(p)).set_index("id")
        sp = p.with_name(p.stem.replace("details", "wellstats") + ".csv")
        if sp.is_file():
            stats[name] = pd.read_csv(sp).set_index("well_id")
    if not frames:
        raise SystemExit("no details files given")

    weights = {n: 1.0 for n in frames}
    fallback = dict(weights)
    if args.no_xy_members:
        for n in args.no_xy_members.split(","):
            fallback[n.strip()] = 0.0
    for spec in (args.weights or "").split(",") if args.weights else []:
        n, v = spec.split("=")
        weights[n] = float(v)

    ref = frames[sorted(frames)[0]]
    safe = {}
    for st in stats.values():
        if "xy_safe" in st.columns:
            safe = st["xy_safe"].astype(bool).to_dict()
            break
    is_safe = np.array([safe.get(w, True) for w in ref["well_id"]])

    acc = np.zeros(len(ref))
    wsum = np.zeros(len(ref))
    for n, df in frames.items():
        v = df.loc[ref.index, "tvt"].to_numpy()
        w = np.where(is_safe, weights.get(n, 0.0), fallback.get(n, 0.0))
        acc += w * v
        wsum += w
    out = ref[["well_id"]].copy()
    out["tvt"] = acc / np.maximum(wsum, 1e-9)

    sub = pd.read_csv(find_sample_submission(root))[["id"]]
    sub = sub.merge(out.reset_index()[["id", "tvt"]], on="id", how="left")
    if sub["tvt"].isna().any():
        raise SystemExit(f"{int(sub.tvt.isna().sum())} submission rows unmatched")
    sub.to_csv(args.out, index=False)
    print(f"[blend] {len(frames)} members -> {args.out}  rows={len(sub)}  "
          f"({int((~is_safe).sum())} rows on the no-XY weight vector)")


def cmd_smoke(args):
    from smoke_test import main as smoke_main
    smoke_main(device=_device(args.device), keep=args.keep)


# ---------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, need_data=True):
        if need_data:
            sp.add_argument("--data-root", default=None,
                            help="defaults to the folder holding train/ and test/")
            sp.add_argument("--train-dir", default="train")
        sp.add_argument("--device", default="auto")
        return sp

    sp = sub.add_parser("list");  sp.set_defaults(fn=lambda a: print(describe()))

    sp = common(sub.add_parser("train"))
    sp.add_argument("--variant", default="base")
    sp.add_argument("--out", default="runs")
    sp.add_argument("--epochs", type=int)
    sp.add_argument("--batch-size", type=int)
    sp.add_argument("--accum-steps", type=int)
    sp.add_argument("--lr", type=float)
    sp.add_argument("--num-workers", type=int)
    sp.add_argument("--backbone")
    sp.add_argument("--no-pretrained", action="store_true")
    sp.add_argument("--n-phases", type=int)
    sp.add_argument("--folds", type=int, nargs="+")
    sp.add_argument("--seeds", type=int, nargs="+")
    sp.add_argument("--splits", type=int, nargs="+")
    sp.set_defaults(fn=cmd_train)

    sp = common(sub.add_parser("oof"))
    sp.add_argument("--variant", required=True)
    sp.add_argument("--weights", required=True)
    sp.add_argument("--out", default="runs")
    sp.add_argument("--n-phases", type=int)
    sp.set_defaults(fn=cmd_oof)

    sp = common(sub.add_parser("compare"))
    sp.add_argument("--base", required=True)
    sp.add_argument("--treat", required=True)
    sp.add_argument("--min-k", type=int, default=52)
    sp.set_defaults(fn=cmd_compare)

    sp = common(sub.add_parser("ensemble"))
    sp.add_argument("--members", required=True,
                    help="name=dir,name=dir")
    sp.add_argument("--fit-weights", action="store_true")
    sp.add_argument("--route", action="store_true")
    sp.set_defaults(fn=cmd_ensemble)

    sp = common(sub.add_parser("infer"))
    sp.add_argument("--variant", default="base")
    sp.add_argument("--run-dir", required=True)
    sp.add_argument("--test-dir", default="test")
    sp.add_argument("--out", default="submission.csv")
    sp.add_argument("--limit-ckpt", type=int)
    sp.add_argument("--n-phases", type=int)
    sp.set_defaults(fn=cmd_infer)

    sp = common(sub.add_parser("benchmark"))
    sp.add_argument("--variants",
                    default="base,v1_tta,v2_slope,v3_moves,v4_refgr,v5_gate")
    sp.add_argument("--out", default="runs")
    sp.add_argument("--report", default="benchmark_results.txt")
    sp.add_argument("--holdout-fold", type=int, default=0)
    sp.add_argument("--epochs", type=int)
    sp.add_argument("--batch-size", type=int)
    sp.add_argument("--num-workers", type=int)
    sp.set_defaults(fn=cmd_benchmark)

    sp = common(sub.add_parser("blend"))
    sp.add_argument("--details", required=True,
                    help="comma-separated details_*.csv from `infer`")
    sp.add_argument("--weights", help="name=w,name=w (default: equal)")
    sp.add_argument("--no-xy-members",
                    help="members zeroed for wells outside the XY-safe region")
    sp.add_argument("--out", default="submission.csv")
    sp.set_defaults(fn=cmd_blend)

    sp = common(sub.add_parser("smoke"), need_data=False)
    sp.add_argument("--keep", action="store_true")
    sp.set_defaults(fn=cmd_smoke)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
