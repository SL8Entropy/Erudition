"""End-to-end self-test on generated wells.

The real competition data is not needed: this writes a small set of wells in
the competition's CSV layout, then exercises every variant's full path --
canvas -> features -> augmentation -> model -> loss -> decode -> OOF ->
ensemble -> submission. It is a plumbing test, not an accuracy test; the scores
it prints mean nothing.

    python smoke_test.py            # or: python run.py smoke
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs import get_config                                   # noqa: E402
from src.data import find_data_root, load_split, split_dir       # noqa: E402
from src.ensemble import compare, load_oof                       # noqa: E402
from src.metrics import score_report                             # noqa: E402
from src.train import train_fold                                 # noqa: E402


# ---------------------------------------------------------------------- #
def make_typewell(rng, tvt_lo=10000.0, tvt_hi=10600.0, step=0.5):
    tvt = np.arange(tvt_lo, tvt_hi, step)
    gr = np.full_like(tvt, 75.0)
    for _ in range(14):                     # layered "barcode" structure
        c = rng.uniform(tvt_lo, tvt_hi)
        w = rng.uniform(3.0, 25.0)
        gr += rng.uniform(-45, 55) * np.exp(-0.5 * ((tvt - c) / w) ** 2)
    gr += rng.normal(0, 2.0, len(tvt))
    return tvt, gr


def make_well(rng, tw_tvt, tw_gr, n=2400, ps_frac=0.3, with_label=True):
    md = np.arange(n, dtype=float) + rng.uniform(0, 500)
    head = rng.uniform(0, 2 * np.pi)
    x = rng.uniform(-4000, 4000) + np.cos(head) * (md - md[0])
    y = rng.uniform(-4000, 4000) + np.sin(head) * (md - md[0])

    slope = rng.normal(0.0, 0.004)          # structural slope of the layers
    s0 = rng.uniform(tw_tvt[0] + 80, tw_tvt[-1] - 80)
    S = s0 + slope * (md - md[0])
    if rng.random() < 0.25:                 # a fault
        S[int(n * rng.uniform(0.4, 0.9)):] += rng.normal(0, 10)

    wiggle = np.cumsum(rng.normal(0, 0.02, n))
    wiggle -= np.linspace(0, wiggle[-1], n)
    z = np.cumsum(rng.normal(0, 0.01, n)) + wiggle
    z -= z[0]
    tvt = S - z
    tvt = np.clip(tvt, tw_tvt[0] + 4, tw_tvt[-1] - 4)

    gr = np.interp(tvt, tw_tvt, tw_gr) + rng.normal(0, 6.0, n)
    gr += np.convolve(rng.normal(0, 5, n + 60), np.ones(60) / 60, "same")[:n]
    gr[rng.random(n) < 0.02] = np.nan

    ps = int(n * ps_frac)
    tvt_input = np.where(np.arange(n) < ps, tvt, np.nan)
    df = pd.DataFrame(dict(MD=md, X=x, Y=y, Z=z, GR=gr, TVT_input=tvt_input))
    if with_label:
        df["TVT"] = tvt
    return df


def build_dataset(root: Path, n_train=12, n_test=3, n_systems=3, seed=0):
    rng = np.random.default_rng(seed)
    systems = [make_typewell(rng) for _ in range(n_systems)]
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "test").mkdir(parents=True, exist_ok=True)

    sub_rows = []
    for split, n_wells, labelled in (("train", n_train, True),
                                     ("test", n_test, False)):
        for i in range(n_wells):
            wid = f"{split[0]}{i:04d}"
            tw_tvt, tw_gr = systems[i % n_systems]
            df = make_well(rng, tw_tvt, tw_gr, with_label=True)
            if not labelled:
                hidden = df["TVT_input"].isna().to_numpy()
                sub_rows += [f"{wid}_{j}" for j in np.flatnonzero(hidden)]
                df = df.drop(columns=["TVT"])
            df.to_csv(root / split / f"{wid}__horizontal_well.csv", index=False)
            pd.DataFrame(dict(TVT=tw_tvt, GR=tw_gr)).to_csv(
                root / split / f"{wid}__typewell.csv", index=False)
    pd.DataFrame(dict(id=sub_rows, tvt=0.0)).to_csv(
        root / "sample_submission.csv", index=False)
    return root


# ---------------------------------------------------------------------- #
def smoke_config(name: str):
    """A variant, shrunk to smoke-test size but with its head/feature flags."""
    base = get_config("smoke")
    cfg = get_config(name)
    for k in ("n_rows", "row_ft", "md_stride", "n_vis_cols", "n_tgt_cols",
              "backbone", "pretrained", "epochs", "batch_size", "num_workers",
              "n_folds", "folds", "seeds", "synth_per_epoch"):
        setattr(cfg, k, getattr(base, k))
    cfg.use_pf = cfg.name in ("base", "v1_tta")     # keep the PF path covered
    cfg.split_patterns = (0,)
    cfg.n_phases = min(cfg.n_phases, 2)
    return cfg


def main(device: str = None, keep: bool = False):
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tmp = Path(tempfile.mkdtemp(prefix="rogii_smoke_"))
    print(f"[smoke] workspace {tmp}   device={device}")
    ok, failures = [], []
    try:
        root = build_dataset(tmp / "data")
        root = find_data_root(root)
        wells = load_split(split_dir(root, "train"), require_tvt=True)
        assert len(wells) >= 8, "well loading failed"

        runs = tmp / "runs"
        for name in ("base", "v2_slope", "v3_moves", "v4_refgr", "v5_gate"):
            cfg = smoke_config(name)
            try:
                oof, rep = train_fold(cfg, wells, fold=0, seed=0, split=0,
                                      out_dir=runs / name, device=device,
                                      verbose=False)
                preds = {k: v["pred"] for k, v in oof.items()}
                assert all(np.isfinite(p).all() for p in preds.values()), "NaN pred"
                print(f"[smoke] {name:10s} ok   in_chans="
                      f"{cfg.canvas_spec().in_chans:2d}  "
                      f"RMSE={rep['pooled_rmse']:.2f}")
                ok.append(name)
            except Exception as e:
                failures.append((name, e, traceback.format_exc()))
                print(f"[smoke] {name:10s} FAILED: {e}")

        # v1 is inference-only: reuse the base weights, change only the decode
        try:
            from src.features import FeatureProvider
            from src.model import build_model
            from src.train import predict_wells
            from src.data import group_kfold
            cfg = smoke_config("v1_tta")
            assign = group_kfold([w.well_id for w in wells], cfg.n_folds, seed=0)
            va = [w for w in wells if assign[w.well_id] == 0]
            allowed = {w.well_id for w in wells if assign[w.well_id] != 0}
            prov = FeatureProvider(wells, allowed_full=allowed, use_pf=cfg.use_pf,
                                   use_xy=cfg.use_xy, use_sibling=cfg.use_sibling,
                                   correct_typewell=cfg.correct_typewell)
            ck = torch.load(next((runs / "base").glob("split0_seed0_fold0.pt")),
                            map_location="cpu", weights_only=False)
            model = build_model(cfg, ck["in_chans"]).to(device)
            model.load_state_dict(ck["model"])
            model.eval()
            o = predict_wells(model, va, cfg, prov, device, n_phases=cfg.n_phases,
                              reanchor=cfg.reanchor,
                              adaptive_canvas=cfg.adaptive_canvas)
            r = score_report({k: v["pred"] for k, v in o.items()}, va)
            print(f"[smoke] v1_tta     ok   (decode only) RMSE={r['pooled_rmse']:.2f}")
            ok.append("v1_tta")
        except Exception as e:
            failures.append(("v1_tta", e, traceback.format_exc()))
            print(f"[smoke] v1_tta     FAILED: {e}")

        # gate + acceptance test + submission
        try:
            from src.gate import apply_gate, fit_gate
            cfg = smoke_config("v5_gate")
            prov = FeatureProvider(wells, allowed_full={w.well_id for w in wells},
                                   use_pf=True, use_xy=True)
            preds, sigmas = load_oof(runs / "v5_gate")
            sub = [w for w in wells if w.well_id in preds]
            cands = {w.well_id: np.stack([
                preds[w.well_id],
                prov.pf_candidate(w),
                prov.xy_candidate(w),
                np.full(w.n, w.anchor_tvt) - (w.z - w.anchor_z),
            ]) for w in sub}
            g = fit_gate(sub, cands, sigmas, epochs=2, device=device, verbose=False)
            gp = {w.well_id: apply_gate(g, w, cands[w.well_id],
                                        sigmas.get(w.well_id), device=device)
                  for w in sub}
            assert all(np.isfinite(v).all() for v in gp.values())
            print(f"[smoke] gate       ok   RMSE={score_report(gp, sub)['pooled_rmse']:.2f}")
            ok.append("gate")
        except Exception as e:
            failures.append(("gate", e, traceback.format_exc()))
            print(f"[smoke] gate       FAILED: {e}")

        try:
            b, _ = load_oof(runs / "base")
            t, _ = load_oof(runs / "v2_slope")
            sub = [w for w in wells if w.well_id in b and w.well_id in t]
            compare(b, t, sub, label="smoke llco", min_k=1)
            ok.append("compare")
        except Exception as e:
            failures.append(("compare", e, traceback.format_exc()))
            print(f"[smoke] compare    FAILED: {e}")

        try:
            from src.infer import run_inference
            cfg = smoke_config("base")
            out_csv = tmp / "submission.csv"
            sub = run_inference(cfg, runs / "base", root, out_csv, device=device,
                                verbose=False)
            assert sub["tvt"].notna().all() and len(sub) > 0
            print(f"[smoke] submission ok   rows={len(sub)}")
            ok.append("submission")
        except Exception as e:
            failures.append(("submission", e, traceback.format_exc()))
            print(f"[smoke] submission FAILED: {e}")
    finally:
        if not keep:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"[smoke] kept {tmp}")

    print(f"\n[smoke] passed {len(ok)}: {', '.join(ok)}")
    if failures:
        print(f"[smoke] FAILED {len(failures)}:")
        for name, e, tb in failures:
            print(f"\n--- {name} ---\n{tb}")
        sys.exit(1)
    print("[smoke] all good")


if __name__ == "__main__":
    main(keep="--keep" in sys.argv)
