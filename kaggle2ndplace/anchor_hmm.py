#!/usr/bin/env python3
"""An HMM tracker over (level, structural slope), as a decorrelated ensemble partner.

Why this exists.  Screening every prediction set on these 155 wells showed that CNN
variants correlate 0.84-0.97 with each other and our AnchorCNN 0.669, while a particle
filter -- a different *method*, not a different architecture -- correlates 0.201.  At that
correlation a partner at 8 ft would be worth -0.40 ft on the blend and one at 6 ft -0.77
ft, against the 0.03 ft everything else has produced.  The particle filter could not cash
it: 28.98 ft, useless.  This is the stronger version of the same idea; 3rd place's HMM
scored 5.97 standalone and their whole ensemble was built on it.

Why the particle filter drifted, measured on 200 training wells:

    structural slope dS/dMD over the eval zone   p5 -0.041, p50 0.026, p95 0.046
    error of the prefix-tail estimate of it      p50 0.008, p90 0.021
    TVT drift if that error is never corrected   p50 37 ft, p90 103 ft

The filter's slope random walk can wander about 0.006 over a whole lateral, so it can
never recover from a bad initial estimate -- hence error growing monotonically with
distance.  The fix is structural, not a tuning knob: carry the slope as a *state over a
grid wide enough to contain the truth* and let the GR evidence revise it.

Model
-----
State      (level, slope), level = TVT - TVT_PS on a 2 ft grid over +-128 ft (129 states),
           slope = d(TVT + Z)/dMD on an absolute grid spanning the measured range
           (53 states), so the truth is inside the grid whatever the prefix suggested.
Transition level moves by slope*dMD - dZ, with a small Gaussian jitter, plus a slow random
           walk on the slope and a rare uniform jump for fault recovery.
Emission   Student-t on the GR residual against the reference profile at that level --
           heavy tails, so one bad GR sample cannot dominate.  Depends only on level, so
           it is computed once per MD bin and shared across slopes.
Prefix     the chain starts before the prediction point with the known TVT clamped, so the
           slope posterior is already informed by data when the eval zone begins.  3rd
           place found this mattered more than the initial estimate itself.
Inference  exact scaled forward-backward, then the posterior mean level per bin.

It fits nothing, so tuning on training wells is leak-free and the holdout is untouched.

    python anchor_hmm.py --out runs/X_hmm --tune 60 --compare ../kaggle_1st_place/solution/results/rb_v3_tta_ep150/holdout_predictions.pqt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

import gr2tvt_data as gd                                                  # noqa: E402
from anchor_data import eval_rows, load_wells, split_wells, well_names    # noqa: E402
from anchor_train import Logger, score                                    # noqa: E402


@dataclass
class HMMConfig:
    win: float = 128.0          # level half-window, ft
    level_step: float = 2.0     # level grid, ft
    bin_ft: float = 32.0        # MD bin, matching the CNN's column width
    rate_lo: float = -0.070     # slope grid: the measured range is [-0.041, +0.046]
    rate_hi: float = 0.075
    n_rates: int = 53
    rate_prior_sd: float = 0.020   # prior around the prefix estimate; p90 of its error
    rate_walk: float = 0.0015      # slope random walk per bin
    level_jitter: float = 1.0      # ft of level jitter per bin
    jump_prob: float = 0.004       # rare uniform component, fault recovery
    obs_scale: float = 0.45        # Student-t scale, normalised GR units
    obs_df: float = 2.0
    obs_temp: float = 8.0          # evidence weight per bin
    self_ref: float = 0.5          # blend of the well's own prefix profile into the reference
    prefix_ft: float = 1024.0      # how much known hole to run before the prediction start
    off_ref_penalty: float = -3.0  # log-likelihood where the reference does not cover


def _reference(w, anchor, levels, cfg):
    """Normalised reference GR at each level: the typewell, optionally blended with the
    well's own pre-PS log, which is the same tool in the same hole and better resolved."""
    tvt = anchor + levels
    tw = np.interp(tvt, w["tw_tvt"].astype(np.float64), w["tw_gr"].astype(np.float64),
                   left=np.nan, right=np.nan)
    ref = (tw - gd.GR_MU) / gd.GR_SD
    if cfg.self_ref > 0:
        ti, gr = w["tvt_input"].astype(np.float64), w["gr"].astype(np.float64)
        ok = np.isfinite(ti) & np.isfinite(gr)
        if ok.sum() >= 32:
            order = np.argsort(ti[ok])
            t_s, g_s = ti[ok][order], (gr[ok][order] - gd.GR_MU) / gd.GR_SD
            edges = np.arange(t_s[0], t_s[-1] + 1.0, 1.0)
            idx = np.clip(np.searchsorted(edges, t_s) - 1, 0, len(edges) - 2)
            cnt = np.bincount(idx, minlength=len(edges) - 1)
            tot = np.bincount(idx, weights=g_s, minlength=len(edges) - 1)
            have = cnt > 0
            if have.any():
                prof = np.interp(tvt, 0.5 * (edges[:-1] + edges[1:])[have], (tot[have] / cnt[have]),
                                 left=np.nan, right=np.nan)
                both = np.isfinite(prof) & np.isfinite(ref)
                ref = np.where(both, (1 - cfg.self_ref) * ref + cfg.self_ref * prof,
                               np.where(np.isfinite(ref), ref, prof))
    return ref


def _emissions(w, md_edges, ref, cfg, bias):
    """Student-t log-likelihood of each MD bin's GR under every level hypothesis."""
    md, gr = w["md"].astype(np.float64), w["gr"].astype(np.float64)
    good = np.isfinite(ref)
    out = np.full((len(md_edges) - 1, len(ref)), 0.0)
    lo_all = np.searchsorted(md, md_edges[:-1], "left")
    hi_all = np.searchsorted(md, md_edges[1:], "left")
    for k in range(len(md_edges) - 1):
        g = gr[lo_all[k]:hi_all[k]]
        g = g[np.isfinite(g)]
        if len(g) == 0:
            continue
        g = (g - gd.GR_MU) / gd.GR_SD - bias
        res = (g[None, :] - ref[good][:, None]) / cfg.obs_scale
        ll = -0.5 * (cfg.obs_df + 1.0) * np.log1p(res ** 2 / cfg.obs_df)
        out[k, good] = cfg.obs_temp * ll.mean(axis=1)
        out[k, ~good] = cfg.obs_temp * cfg.off_ref_penalty
    return out


def _shift_blur(A, shifts_bins, sigma_bins):
    """A is (R, L) over levels; shift each row by its own (fractional) amount, then blur."""
    R, L = A.shape
    pos = np.arange(L)[None, :] - shifts_bins[:, None]
    p0 = np.floor(pos).astype(np.int64)
    frac = pos - p0
    ok = (p0 >= 0) & (p0 + 1 < L)
    i0 = np.clip(p0, 0, L - 1)
    i1 = np.clip(p0 + 1, 0, L - 1)
    rows = np.arange(R)[:, None]
    out = np.where(ok, A[rows, i0] * (1 - frac) + A[rows, i1] * frac, 0.0)
    if sigma_bins > 0:
        out = gaussian_filter1d(out, sigma_bins, axis=1, mode="constant")
    return out


def run_hmm(w, cfg: HMMConfig):
    """Posterior mean and sd of the level at each MD bin, from PS to the end of the well."""
    md = w["md"].astype(np.float64)
    ti = w["tvt_input"].astype(np.float64)
    known = np.isfinite(ti)
    if known.sum() < 32 or (~known).sum() < 2:
        raise ValueError("no usable prefix or eval zone")
    i0 = int(np.flatnonzero(~known)[0])
    anchor = float(ti[known][-1])
    md_ps = float(md[i0])

    levels = np.arange(-cfg.win, cfg.win + 1e-9, cfg.level_step)
    rates = np.linspace(cfg.rate_lo, cfg.rate_hi, cfg.n_rates)
    L, R = len(levels), len(rates)

    md_start = max(md[0], md_ps - cfg.prefix_ft)
    edges = np.arange(md_start, md[-1] + cfg.bin_ft, cfg.bin_ft)
    if len(edges) < 3:
        raise ValueError("well too short")
    n_bins = len(edges) - 1

    ref = _reference(w, anchor, levels, cfg)
    ok = (np.isfinite(ti) & np.isfinite(w["gr"].astype(np.float64))
          & (np.abs(ti - anchor) <= cfg.win))     # inside the window: the build section is not
    if ok.sum() >= 32:
        r_at = np.interp(ti[ok], anchor + levels, np.nan_to_num(ref, nan=0.0))
        bias = float(np.median((w["gr"].astype(np.float64)[ok] - gd.GR_MU) / gd.GR_SD - r_at))
    else:
        bias = 0.0

    E = _emissions(w, edges, ref, cfg, bias)
    z_at = np.interp(edges, md, w["z"].astype(np.float64))

    # slope prior from the tail of the known prefix, on an absolute grid so a bad estimate
    # cannot put the truth outside the state space
    tail = np.flatnonzero(known)[-256:]
    S = ti[tail] + w["z"].astype(np.float64)[tail]
    r0 = float(np.polyfit(md[tail], S, 1)[0]) if len(tail) >= 8 else 0.0
    prior_r = np.exp(-0.5 * ((rates - r0) / cfg.rate_prior_sd) ** 2)
    prior_r /= prior_r.sum()

    # level clamp: known inside the prefix, free afterwards
    centres = 0.5 * (edges[:-1] + edges[1:])
    clamp = np.full(n_bins, np.nan)
    pre = centres < md_ps
    clamp[pre] = np.interp(centres[pre], md[known], ti[known]) - anchor
    clamp[np.abs(clamp) > cfg.win - cfg.level_step] = np.nan

    rate_kernel = np.exp(-0.5 * ((rates[:, None] - rates[None, :]) / max(cfg.rate_walk, 1e-9)) ** 2)
    rate_kernel /= rate_kernel.sum(axis=0, keepdims=True)
    sigma_bins = cfg.level_jitter / cfg.level_step

    def observe(a, k):
        e = E[k] - E[k].max()
        a = a * np.exp(e)[None, :]
        if np.isfinite(clamp[k]):
            g = np.exp(-0.5 * ((levels - clamp[k]) / max(cfg.level_step, 1e-9)) ** 2)
            a = a * g[None, :]
        s = a.sum()
        return (a / s, np.log(s)) if s > 0 else (np.full_like(a, 1.0 / a.size), -1e3)

    def propagate(a, k, forward=True):
        dz = z_at[k + 1] - z_at[k] if forward else z_at[k] - z_at[k - 1]
        sh = (rates * cfg.bin_ft - dz) / cfg.level_step
        out = _shift_blur(a, sh if forward else -sh, sigma_bins)
        out = rate_kernel @ out
        return (1 - cfg.jump_prob) * out + cfg.jump_prob / out.size

    # forward
    alpha = np.zeros((R, L))
    j0 = int(np.argmin(np.abs(levels - (clamp[0] if np.isfinite(clamp[0]) else 0.0))))
    alpha[:, j0] = prior_r
    alpha, _ = observe(alpha, 0)
    A = [alpha]
    for k in range(n_bins - 1):
        alpha, _ = observe(propagate(alpha, k, True), k + 1)
        A.append(alpha)

    # backward
    beta = np.full((R, L), 1.0 / (R * L))
    B = [None] * n_bins
    B[-1] = beta
    for k in range(n_bins - 1, 0, -1):
        e = E[k] - E[k].max()
        b = beta * np.exp(e)[None, :]
        if np.isfinite(clamp[k]):
            b = b * np.exp(-0.5 * ((levels - clamp[k]) / max(cfg.level_step, 1e-9)) ** 2)[None, :]
        b = propagate(b, k, False)
        s = b.sum()
        beta = b / s if s > 0 else np.full_like(b, 1.0 / b.size)
        B[k - 1] = beta

    mean = np.empty(n_bins)
    sd = np.empty(n_bins)
    for k in range(n_bins):
        g = A[k] * B[k]
        p = g.sum(axis=0)
        t = p.sum()
        p = p / t if t > 0 else np.full_like(p, 1.0 / len(p))
        m = float((p * levels).sum())
        mean[k] = m
        sd[k] = float(np.sqrt(max((p * levels ** 2).sum() - m ** 2, 0.0)))
    return dict(centres=centres, mean=mean, sd=sd, anchor=anchor)


def hmm_predict(w, md_query, cfg):
    out = run_hmm(w, cfg)
    m = np.interp(md_query, out["centres"], out["mean"])
    s = np.interp(md_query, out["centres"], out["sd"])
    return out["anchor"] + m, s


def tune(wells, names, cfg, n_wells, log, seed=0):
    """Grid over the knobs the drift analysis implicates, on TRAINING wells only."""
    rng = np.random.default_rng(seed)
    sample = list(rng.choice(names, size=min(n_wells, len(names)), replace=False))
    log(f"tuning on {len(sample)} training wells (the holdout is not touched)")
    best = None
    for obs_temp in (2.0, 8.0, 30.0):
        for prior_sd in (0.008, 0.020):
            for self_ref in (0.0, 0.5):
                c = replace(cfg, obs_temp=obs_temp, rate_prior_sd=prior_sd, self_ref=self_ref)
                se, n = 0.0, 0
                for nm in sample:
                    w = wells[nm]
                    ev = eval_rows(w)
                    if len(ev) == 0:
                        continue
                    try:
                        m, _ = hmm_predict(w, w["md"][ev].astype(np.float64), c)
                    except Exception:
                        continue
                    err = m - w["tvt"][ev].astype(np.float64)
                    se += float((err ** 2).sum())
                    n += len(err)
                rmse = float(np.sqrt(se / max(n, 1)))
                log(f"   obs_temp {obs_temp:<5g} prior_sd {prior_sd:<7g} self_ref {self_ref:<4g}"
                    f" -> {rmse:7.3f} ft")
                if best is None or rmse < best[0]:
                    best = (rmse, c)
    log(f"   chosen: obs_temp {best[1].obs_temp:g}, prior_sd {best[1].rate_prior_sd:g}, "
        f"self_ref {best[1].self_ref:g}  ({best[0]:.3f} ft on train)")
    return best[1], dict(train_rmse=best[0], obs_temp=best[1].obs_temp,
                         rate_prior_sd=best[1].rate_prior_sd, self_ref=best[1].self_ref)


def compare(df, other_path, log):
    o = pd.read_parquet(other_path).sort_values(["well_id", "submit_index"]).reset_index(drop=True)
    d = df.sort_values(["well_id", "submit_index"]).reset_index(drop=True)
    if len(o) != len(d) or not o[["well_id", "submit_index"]].equals(d[["well_id", "submit_index"]]):
        log("compare: the other file is not on the same rows; skipping")
        return {}
    y = d["TVT"].to_numpy()
    pa, pc = d["TVT_pred"].to_numpy(), o["TVT_pred"].to_numpy()
    rmse = lambda p: float(np.sqrt(((p - y) ** 2).mean()))  # noqa: E731
    rho = float(np.corrcoef(pa - y, pc - y)[0, 1])
    wgt = float(np.clip(((y - pc) * (pa - pc)).sum() / max(((pa - pc) ** 2).sum(), 1e-9), 0.0, 1.0))
    blend = rmse(wgt * pa + (1 - wgt) * pc)
    log("")
    log(f"against {Path(other_path).parent.name}: it {rmse(pc):.4f} ft, HMM {rmse(pa):.4f} ft")
    log(f"  residual correlation {rho:.3f}   (AnchorCNN 0.669, particle filter 0.201)")
    log(f"  best blend: {wgt:.3f} HMM -> {blend:.4f} ft  ({blend - rmse(pc):+.4f} ft)")
    return dict(other_rmse=rmse(pc), hmm_rmse=rmse(pa), residual_corr=rho,
                best_weight_hmm=wgt, blend_rmse=blend, blend_gain=blend - rmse(pc))


def main(args):
    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(f"{out_dir} exists and is not empty; pass --force to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "hmm.log")
    log(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(sys.argv)}")

    cfg = HMMConfig(bin_ft=args.bin_ft, level_step=args.level_step, n_rates=args.n_rates,
                    obs_temp=args.obs_temp, self_ref=args.self_ref, prefix_ft=args.prefix_ft)
    names = well_names(args.data)
    if args.limit_wells:
        names = names[:args.limit_wells]
    tr_names, ho_names = split_wells(names, args.train_frac)
    wells = load_wells(Path(args.data), names, Path(args.cache) if args.cache else None, log=log)
    ho = {n: wells[n] for n in ho_names}
    log(f"holdout {len(ho_names)} wells; state grid {int(2 * cfg.win / cfg.level_step) + 1} levels "
        f"x {cfg.n_rates} slopes over [{cfg.rate_lo:+.3f}, {cfg.rate_hi:+.3f}]; no training involved")

    tuning = None
    if args.tune:
        cfg, tuning = tune(wells, tr_names, cfg, args.tune, log, seed=args.seed)

    t0 = time.time()
    preds, stds, failed = {}, {}, []
    for i, nm in enumerate(ho_names):
        ev = eval_rows(ho[nm])
        if len(ev) == 0:
            continue
        try:
            m, s = hmm_predict(ho[nm], ho[nm]["md"][ev].astype(np.float64), cfg)
        except Exception as exc:
            log(f"  {nm}: {type(exc).__name__}: {str(exc)[:80]}")
            failed.append(nm)
            continue
        preds[nm], stds[nm] = m, s
        if (i + 1) % 50 == 0:
            log(f"  {i + 1}/{len(ho_names)} wells ({time.time() - t0:.0f}s)")
    log(f"decoded {len(preds)} wells in {time.time() - t0:.0f}s")

    metrics, df = score(ho, preds, failed)
    df["TVT_std"] = np.nan
    for nm, s in stds.items():
        df.loc[(df["well_id"] == nm).to_numpy(), "TVT_std"] = s
    okm = np.isfinite(df["TVT_std"].to_numpy())
    metrics["std_error_spearman"] = float(pd.Series(df["TVT_std"][okm]).corr(
        pd.Series(np.sqrt(df["sq_err"].to_numpy()[okm])), method="spearman"))
    if tuning:
        metrics["tuning"] = tuning

    log("")
    log(f"HMM POOLED RMSE: {metrics['pooled_rmse']:.4f} ft over {metrics['n_rows']:,} rows "
        f"in {metrics['n_wells']} wells")
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
    (out_dir / "hmm_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=str(HERE / "data"))
    p.add_argument("--cache", default=str(HERE / "cache/wells.npz"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--limit-wells", type=int, default=0)
    p.add_argument("--bin-ft", type=float, default=32.0)
    p.add_argument("--level-step", type=float, default=2.0)
    p.add_argument("--n-rates", type=int, default=53)
    p.add_argument("--obs-temp", type=float, default=8.0)
    p.add_argument("--self-ref", type=float, default=0.5)
    p.add_argument("--prefix-ft", type=float, default=1024.0)
    p.add_argument("--tune", type=int, default=0, help="tune on this many TRAINING wells first")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compare", default="")
    a = p.parse_args(argv)
    if not a.cache:
        a.cache = None
    return a


if __name__ == "__main__":
    main(parse_args())
