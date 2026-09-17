#!/usr/bin/env python3
"""Particle filter predictions on the same holdout, as a decorrelated ensemble partner.

Screening every model already scored on these 155 wells showed the problem with
ensembling here: the 1st-place variants all correlate 0.84-0.97 with each other, our
AnchorCNN correlates 0.669, and the only genuinely decorrelated thing (their non-ML geo
prior, 0.510) is far too weak at 11.6 ft to carry any weight.  The blend maths says a
partner at ~6 ft with correlation 0.5 would be worth ~0.3 ft, an order more than
everything extracted so far.

A particle filter is the obvious candidate: Bayesian state tracking rather than pattern
matching over an image, so it fails differently by construction.  3rd place's particle
filter and HMM families scored 5.6-6.2 standalone and they credited the diverse error
structure, not the accuracy, for their ensemble's generalisation.

The filter itself is ``rogii_champion/src/particle_filter.py``, reused unmodified: 512
particles over (level, structural slope, GR bias), three configuration profiles blended,
forward filter plus a genealogy backward pass.  It needs **no training** -- it is pure
per-well inference from the typewell, the lateral GR and the known prefix -- so there is
no split to respect and no leakage to worry about.  This file is the glue: our wells in,
a ``holdout_predictions.pqt`` out, in the same schema everything else here writes.

``--compare`` prints the number that actually decides whether this was worth doing: the
correlation of its residuals with another model's, and the blend that follows.

    python anchor_pf.py --out runs/X_pf --compare ../kaggle_1st_place/solution/results/rb_v3_tta_ep150/holdout_predictions.pqt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE.parent))          # so rogii_champion is importable

from anchor_data import eval_rows, load_wells, split_wells, well_names  # noqa: E402
from anchor_train import Logger, score                                  # noqa: E402


_SIBLING = None          # (SiblingBank, weight) installed by main(); see --sibling-w


def sibling_reference(name, tw_tvt, tw_gr):
    """Blend the supplied typewell with this well's siblings' lateral GR, on the tw_tvt axis.

    Injecting here means every consumer of the filter -- tuning, dip fitting, prediction --
    sees the same reference without any of them knowing about it.  Depths no sibling logged
    keep the typewell unchanged, so the blend never removes information.
    """
    if _SIBLING is None:
        return tw_gr
    bank, wgt = _SIBLING
    sib, cov, n = bank.profile(name, tw_tvt)
    if n == 0 or not cov.any():
        return tw_gr
    out = tw_gr.copy()
    out[cov] = (1.0 - wgt) * tw_gr[cov] + wgt * sib[cov]
    return out


def to_champion_well(name: str, w: dict):
    """Our well dict -> the Well dataclass the particle filter expects.

    X and Y are not used by the filter (it tracks structure along MD only), so they are
    passed as zeros rather than loaded.
    """
    from rogii_champion.src.data import Well
    n = len(w["md"])
    f64 = lambda k: w[k].astype(np.float64)  # noqa: E731
    tw_tvt = f64("tw_tvt")
    return Well(well_id=name, md=f64("md"), x=np.zeros(n), y=np.zeros(n), z=f64("z"),
                gr=f64("gr"), tvt_input=f64("tvt_input"), tw_tvt=tw_tvt,
                tw_gr=sibling_reference(name, tw_tvt, f64("tw_gr")),
                row_idx=np.arange(n), tvt=f64("tvt"))


def pf_predict(well_obj, md_query, profiles):
    """Weighted mean level per profile, mapped to MD.  Returns (mean, std) in TVT feet.

    Each profile contributes a mean and a spread; they are combined with the law of total
    variance, so the reported spread includes both the particle cloud's width and the
    disagreement between profiles.
    """
    from rogii_champion.src.particle_filter import run_pf
    e1 = np.zeros(len(md_query))
    e2 = np.zeros(len(md_query))
    for cfg in profiles:
        out = run_pf(well_obj, cfg)
        grid, levels, w = out["md_grid"], out["levels"], out["weights"]
        centres = 0.5 * (grid[:-1] + grid[1:])
        mean_bin = (levels * w[None, :]).sum(1)
        var_bin = (levels ** 2 * w[None, :]).sum(1) - mean_bin ** 2
        m = np.interp(md_query, centres, mean_bin)
        v = np.interp(md_query, centres, np.clip(var_bin, 0.0, None))
        e1 += m
        e2 += v + m ** 2
    e1 /= len(profiles)
    e2 /= len(profiles)
    return well_obj.anchor_tvt + e1, np.sqrt(np.clip(e2 - e1 ** 2, 0.0, None))


def tune(wells, names, base_profiles, n_wells, log, seed=0, temps=(1.0, 8.0),
         rates=(1.0, 4.0, 16.0, 64.0), particles=(0,), jumps=(0.0,), ess=(0.0,)):
    """Pick the filter's likelihood weight and drift rate on TRAINING wells.

    The filter fits nothing, so tuning it on the 618 training wells is leak-free -- the
    holdout is never touched.  Doing it on the holdout would be fitting to the test set.

    The grid targets the two knobs the failure mode implicates.  Error grows steadily with
    distance past the prediction start (4 ft in the first fifth of the lateral, 30 ft in
    the last), which is drift: the state's random walk outruns the evidence correcting it.
    ``obs_temp`` scales how strongly each bin's GR match pulls the particles back;
    ``rate_std`` is how fast the structural slope is allowed to wander.
    """
    from dataclasses import replace
    rng = np.random.default_rng(seed)
    sample = list(rng.choice(names, size=min(n_wells, len(names)), replace=False))
    log(f"tuning on {len(sample)} training wells (the holdout is not touched)")
    best, table = None, []
    # The filter averages log-likelihood over the samples in a bin instead of summing,
    # so the evidence is many times weaker than the state's random walk.  The grid has to
    # reach well past 1 for that reason; an optimum at the edge means widening it again.
    # The diagnosis is slope exploration, not evidence weight: the filter's slope random
    # walk wanders ~0.006 over a lateral while the prefix estimate of the slope is wrong by
    # 0.021 at p90, so no particle ever carries the right slope and the error grows with
    # distance.  The grid therefore has to reach far ABOVE the default rate_std, not below.
    # particles / jump probability / resampling threshold attack a different failure than
    # drift: a filter that has collapsed onto one layer hypothesis cannot recover, and the
    # remaining tail wells are exactly that.  0 means 'leave the profile's own value'.
    import itertools
    for temp_mult, rate_mult, npart, jp, ef in itertools.product(temps, rates, particles, jumps, ess):
        if True:
            over = {}
            if npart:
                over["n_particles"] = int(npart)
            if jp:
                over["jump_prob"] = float(jp)
            if ef:
                over["ess_frac"] = float(ef)
            profs = [replace(p, obs_temp=p.obs_temp * temp_mult,
                             rate_std=p.rate_std * rate_mult, **over) for p in base_profiles]
            se, n = 0.0, 0
            for nm in sample:
                w = wells[nm]
                ev = eval_rows(w)
                if len(ev) == 0:
                    continue
                try:
                    m, _ = pf_predict(to_champion_well(nm, w), w["md"][ev].astype(np.float64), profs)
                except Exception:
                    continue
                err = m - w["tvt"][ev].astype(np.float64)
                se += float((err ** 2).sum())
                n += len(err)
            rmse = float(np.sqrt(se / max(n, 1)))
            table.append((rmse, temp_mult, rate_mult))
            log(f"   temp x{temp_mult:<4g} rate x{rate_mult:<4g} N={npart or 512:<5d} "
                f"jump={jp or 0.004:<6g} ess={ef or 0.5:<4g} -> {rmse:7.3f} ft")
            if best is None or rmse < best[0]:
                best = (rmse, temp_mult, rate_mult, profs)
    log(f"   chosen: obs_temp x{best[1]:g}, rate_std x{best[2]:g}  ({best[0]:.3f} ft on train)")
    return best[3], dict(train_rmse=best[0], obs_temp_mult=best[1], rate_std_mult=best[2],
                         grid=[(r, t, s) for r, t, s in table])


def fit_drift_correction(wells, names, profiles, n_wells, log, seed=0):
    """Fit the filter's systematic drift on TRAINING wells: err ~ a + b * (MD - MD_PS).

    Measured on the holdout before correcting anything, the error was not scattered -- it
    was one-directional and growing: -1.7, -6.1, -8.5, -9.6 ft across the four quarters of
    the lateral, negative in 72% of wells.  That is the filter under-estimating the
    structural slope in a consistent direction, so every well drifts the same way and the
    worst wells are simply where it compounds longest.

    Two parameters, least squares, fitted on training wells only.  It cannot fix a well
    that locks onto the wrong layer, but it removes the part of the error that is common
    to all of them.
    """
    rng = np.random.default_rng(seed)
    sample = list(rng.choice(names, size=min(n_wells, len(names)), replace=False))
    dm, er = [], []
    for nm in sample:
        w = wells[nm]
        ev = eval_rows(w)
        if len(ev) == 0:
            continue
        md = w["md"][ev].astype(np.float64)
        try:
            m, _ = pf_predict(to_champion_well(nm, w), md, profiles)
        except Exception:
            continue
        dm.append(md - md[0])
        er.append(m - w["tvt"][ev].astype(np.float64))
    if not dm:
        return 0.0, 0.0
    dm, er = np.concatenate(dm), np.concatenate(er)
    A = np.column_stack([np.ones_like(dm), dm])
    a, b = np.linalg.lstsq(A, er, rcond=None)[0]
    log(f"drift correction fitted on {len(sample)} training wells: "
        f"err = {a:+.3f} ft {b:+.5f} x (MD - MD_PS)  "
        f"[{b * 5000:+.1f} ft over a 5000 ft lateral]")
    return float(a), float(b)


def fit_dip(wells, tr_names, profiles, n_wells, log, seed=0, alpha=1.0):
    """Fit a per-well dip corrector on training wells (see anchor_dip.py)."""
    from anchor_dip import DipCorrector, build_pool, error_line, well_features
    rng = np.random.default_rng(seed)
    sample = list(rng.choice(tr_names, size=min(n_wells, len(tr_names)), replace=False))
    pool = build_pool(wells, tr_names)
    log(f"dip corrector: neighbour pool {len(pool['ids'])} training wells, "
        f"fitting on {len(sample)}")
    X, A, B = [], [], []
    for nm in sample:
        w = wells[nm]
        ev = eval_rows(w)
        if len(ev) < 32:
            continue
        md = w["md"][ev].astype(np.float64)
        try:
            m, sd = pf_predict(to_champion_well(nm, w), md, profiles)
        except Exception:
            continue
        a, b = error_line(m, w["tvt"][ev].astype(np.float64), md)
        X.append(well_features(w, pool, m, sd, exclude=nm))   # never its own neighbour
        A.append(a)
        B.append(b)
    X = np.asarray(X)
    return DipCorrector(alpha).fit(X, np.asarray(A), np.asarray(B), log=log), pool


def compare(df: pd.DataFrame, other_path: Path, log):
    """Residual correlation against another prediction file, and the resulting blend."""
    o = pd.read_parquet(other_path).sort_values(["well_id", "submit_index"]).reset_index(drop=True)
    d = df.sort_values(["well_id", "submit_index"]).reset_index(drop=True)
    if len(o) != len(d) or not o[["well_id", "submit_index"]].equals(d[["well_id", "submit_index"]]):
        log("compare: the other file is not on the same rows; skipping")
        return {}
    y = d["TVT"].to_numpy()
    pa, pc = d["TVT_pred"].to_numpy(), o["TVT_pred"].to_numpy()
    rmse = lambda p: float(np.sqrt(((p - y) ** 2).mean()))  # noqa: E731
    rho = float(np.corrcoef(pa - y, pc - y)[0, 1])
    w = float(np.clip(((y - pc) * (pa - pc)).sum() / max(((pa - pc) ** 2).sum(), 1e-9), 0.0, 1.0))
    blend = rmse(w * pa + (1 - w) * pc)
    res = dict(other=str(other_path), other_rmse=rmse(pc), pf_rmse=rmse(pa),
               residual_corr=rho, best_weight_pf=w, blend_rmse=blend,
               blend_gain=blend - rmse(pc))
    log("")
    log(f"against {Path(other_path).parent.name}: it {res['other_rmse']:.4f} ft, "
        f"PF {res['pf_rmse']:.4f} ft")
    log(f"  residual correlation {rho:.3f}  (AnchorCNN manages 0.669; below ~0.5 is the "
        f"prize)")
    log(f"  best blend: {w:.3f} PF -> {blend:.4f} ft  ({res['blend_gain']:+.4f} ft)")
    return res


def main(args):
    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(f"{out_dir} exists and is not empty; pass --force to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "pf.log")
    log(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(sys.argv)}")

    from rogii_champion.src.particle_filter import PROFILES
    profiles = PROFILES[:args.profiles]
    names = well_names(args.data)
    if args.limit_wells:
        names = names[:args.limit_wells]
    tr_names, ho_names = split_wells(names, args.train_frac)
    wells = load_wells(Path(args.data), names, Path(args.cache) if args.cache else None, log=log)
    if args.sibling_w > 0:
        global _SIBLING
        from anchor_sibling import SiblingBank
        bank = SiblingBank(bin_ft=args.sibling_bin).fit(wells, tr_names, all_names=names, log=log)
        _SIBLING = (bank, args.sibling_w)
        cov = np.array([bank.n_siblings(n) for n in ho_names])
        log(f"sibling reference at weight {args.sibling_w:.2f}: {int((cov > 0).sum())}/"
            f"{len(ho_names)} holdout wells have siblings (median {int(np.median(cov))})")

    ho = {n: wells[n] for n in ho_names}
    log(f"holdout {len(ho_names)} wells, {sum(len(eval_rows(w)) for w in ho.values()):,} rows; "
        f"{len(profiles)} particle-filter profile(s), no training involved")
    tuning = None
    if args.tune:
        profiles, tuning = tune(wells, tr_names, profiles, args.tune, log, seed=args.seed,
                                temps=tuple(args.tune_temps), rates=tuple(args.tune_rates),
                                particles=tuple(args.tune_particles), jumps=tuple(args.tune_jump),
                                ess=tuple(args.tune_ess))

    dip, pool = (None, None)
    if args.dip_correct:
        dip, pool = fit_dip(wells, tr_names, profiles, args.dip_correct, log,
                            seed=args.seed, alpha=args.dip_alpha)

    drift = (0.0, 0.0)
    if args.calibrate and not args.dip_correct:
        drift = fit_drift_correction(wells, tr_names, profiles, args.calibrate, log, seed=args.seed)

    t0 = time.time()
    preds, stds, failed = {}, {}, []
    for i, nm in enumerate(ho_names):
        ev = eval_rows(ho[nm])
        if len(ev) == 0:
            continue
        mdq = ho[nm]["md"][ev].astype(np.float64)
        try:
            m, s = pf_predict(to_champion_well(nm, ho[nm]), mdq, profiles)
        except Exception as exc:                       # a well the filter cannot run on
            log(f"  {nm}: {type(exc).__name__}: {str(exc)[:80]}")
            failed.append(nm)
            continue
        if dip is not None:
            from anchor_dip import well_features
            fa, fb = dip.predict(well_features(ho[nm], pool, m, s)[None, :])
            m = m - (fa[0] + fb[0] * (mdq - mdq[0]))
        elif args.calibrate:
            m = m - (drift[0] + drift[1] * (mdq - mdq[0]))
        preds[nm], stds[nm] = m, s
        if (i + 1) % 50 == 0:
            log(f"  {i + 1}/{len(ho_names)} wells ({time.time() - t0:.0f}s)")
    log(f"filtered {len(preds)} wells in {time.time() - t0:.0f}s")

    metrics, df = score(ho, preds, failed)
    df["TVT_std"] = np.nan
    for nm, s in stds.items():
        df.loc[(df["well_id"] == nm).to_numpy(), "TVT_std"] = s
    ok = np.isfinite(df["TVT_std"].to_numpy())
    metrics["std_error_spearman"] = float(pd.Series(df["TVT_std"][ok]).corr(
        pd.Series(np.sqrt(df["sq_err"].to_numpy()[ok])), method="spearman"))
    metrics["profiles"] = len(profiles)
    if tuning is not None:
        metrics["tuning"] = tuning

    log("")
    log(f"PARTICLE FILTER POOLED RMSE: {metrics['pooled_rmse']:.4f} ft "
        f"over {metrics['n_rows']:,} rows in {metrics['n_wells']} wells")
    log(f"per-well RMSE  mean {metrics['well_rmse_mean']:.3f}  p50 {metrics['well_rmse_p50']:.3f}"
        f"  p95 {metrics['well_rmse_p95']:.3f}")
    log(f"spread vs |error|, Spearman rho = {metrics['std_error_spearman']:.3f}")
    if args.compare:
        metrics["compare"] = compare(df, Path(args.compare), log)

    df.drop(columns=["sq_err"]).to_parquet(out_dir / "holdout_predictions.pqt", index=False)
    per = df.groupby("well_id")["sq_err"].agg(["mean", "sum", "size"])
    per = per.assign(rmse=np.sqrt(per["mean"])).rename(columns={"sum": "sse", "size": "rows"})
    per[["rmse", "sse", "rows"]].sort_values("sse", ascending=False).to_csv(
        out_dir / "holdout_well_rmse.csv")
    (out_dir / "pf_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=str(HERE / "data"))
    p.add_argument("--cache", default=str(HERE / "cache/wells.npz"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--limit-wells", type=int, default=0)
    p.add_argument("--profiles", type=int, default=3, help="how many of the filter's blended "
                   "configuration profiles to average (1-3)")
    p.add_argument("--tune", type=int, default=0,
                   help="tune the filter on this many TRAINING wells first (leak-free: it fits "
                        "nothing and the holdout is untouched). 100 costs about a minute")
    p.add_argument("--tune-particles", type=int, nargs="+", default=[0])
    p.add_argument("--tune-jump", type=float, nargs="+", default=[0.0])
    p.add_argument("--tune-ess", type=float, nargs="+", default=[0.0])
    p.add_argument("--dip-correct", type=int, default=0,
                   help="fit a per-well dip corrector on this many TRAINING wells and apply "
                        "it to the holdout; supersedes --calibrate")
    p.add_argument("--dip-alpha", type=float, default=1.0, help="ridge strength")
    p.add_argument("--calibrate", type=int, default=0,
                   help="fit a two-parameter drift correction on this many TRAINING wells "
                        "and apply it to the holdout (leak-free)")
    p.add_argument("--sibling-w", type=float, default=0.0,
                   help="blend weight on the sibling-lateral reference GR (0 = typewell only). "
                        "Siblings are TRAINING wells sharing a master typewell system.")
    p.add_argument("--sibling-bin", type=float, default=1.0, help="sibling profile bin, feet")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tune-temps", type=float, nargs="+", default=[1.0, 8.0],
                   help="multipliers on the likelihood weight to try")
    p.add_argument("--tune-rates", type=float, nargs="+", default=[1.0, 4.0, 16.0, 64.0],
                   help="multipliers on the slope random walk; the filter's default is far too "
                        "small for the measured slope error, so these run upward")
    p.add_argument("--compare", default="", help="another holdout_predictions.pqt to correlate against")
    a = p.parse_args(argv)
    if not a.cache:
        a.cache = None
    return a


if __name__ == "__main__":
    main(parse_args())
