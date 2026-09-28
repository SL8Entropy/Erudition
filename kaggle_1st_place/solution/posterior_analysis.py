"""Is the ConvNeXt's error bimodal (mode confusion) or unimodal (drift)?

Pure numpy; no torch, no pipeline imports, so it can be tested on synthetic
posteriors and re-run on a saved dump without a GPU.

The 1st-place model outputs, for every 32 ft MD column, a softmax over 400
candidate depths (0.5 ft apart, +-100 ft around the anchor TVT) and reports its
mean.  If rock layers repeating is what drives the catastrophic wells, the
columns where those wells go wrong should show two or more humps, with the
truth sitting under one of them.  If instead the wells are drifting -- the
report's "one wrong straight line per well" -- the posterior should be a single
hump sliding away from the truth.

Vocabulary used throughout:

  basin     the stretch of levels between two local minima of the (lightly
            smoothed) posterior; each basin holds one hump
  core      the part of a basin where its hump stays above CORE_FRAC of its
            peak; "the truth sits under a hump" means inside its core (+-2 ft)
  mode      a basin, summarised by its probability mass and its centroid
  sig mode  a mode with mass >= MIN_MODE_MASS
  dominant  the sig mode with the largest mass

Row-level categories, assigned only to rows whose mean-decoded error is at
least BAD_FT (the other rows are "ok"):

  out_of_window    truth lies outside the +-100 ft window; nothing on the
                   canvas can represent it
  unimodal_miss    one sig mode, truth outside it: a single hump in the wrong
                   place (drift)
  unimodal_offset  one sig mode containing the truth, but the mean is still
                   BAD_FT away: a broad or skewed hump
  blend            >=2 sig modes, truth under the dominant one: the mean was
                   pulled off it by a secondary hump; picking the dominant
                   mode would have helped
  wrong_mode       >=2 sig modes, truth under a secondary one: the model
                   preferred the wrong hump
  multimodal_miss  >=2 sig modes, truth under none of them
"""

from __future__ import annotations

import math

import numpy as np

MIN_MODE_MASS = 0.05     # a hump must carry 5% of the probability to count
SMOOTH_SIGMA_BINS = 2.0  # 1 ft Gaussian smoothing before finding local extrema
PROMINENCE = 0.5         # merge humps whose separating dip is > 50% of the lower peak
TRUTH_BAND_FT = 2.0      # "probability at the truth" = mass within +-2 ft
BAD_FT = 5.0             # rows with |mean error| >= this get a failure category
CORE_FRAC = 0.1          # a hump's core = where it stays above 10% of its own peak

CATEGORIES = (
    "ok",
    "out_of_window",
    "unimodal_miss",
    "unimodal_offset",
    "blend",
    "wrong_mode",
    "multimodal_miss",
)


def level_axis(window_ft: float = 100.0, n_levels: int = 400) -> np.ndarray:
    """The pipeline's candidate depths, relative to the anchor (seq_NN_models)."""
    step = window_ft / n_levels
    return np.linspace(-window_ft + step, window_ft - step, n_levels)


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _gauss_smooth(p: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return p
    half = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(p, half, mode="edge"), k, mode="valid")


def find_modes(p: np.ndarray, x: np.ndarray) -> list[dict]:
    """Split a 1-D posterior into humps. Returns modes sorted by mass, largest first.

    Local maxima of a lightly smoothed copy define the humps; the boundary
    between two neighbours is the lowest point between them.  Neighbours whose
    dip is shallow (valley > PROMINENCE * lower peak) are merged, so noise on
    one hump is not counted as a second mode.  Masses and centroids are taken
    from the unsmoothed posterior.
    """
    s = _gauss_smooth(p, SMOOTH_SIGMA_BINS)
    n = s.size
    # plateau-safe local maxima
    d = np.diff(s)
    peaks = [i for i in range(1, n - 1) if d[i - 1] > 0 and d[i] <= 0]
    if s[0] > s[1]:
        peaks.insert(0, 0)
    if s[-1] > s[-2]:
        peaks.append(n - 1)
    if not peaks:
        peaks = [int(np.argmax(s))]
    # boundaries = argmin between consecutive peaks
    bounds = [int(peaks[i] + np.argmin(s[peaks[i]:peaks[i + 1] + 1])) for i in range(len(peaks) - 1)]
    # merge shallow dips, repeatedly
    changed = True
    while changed and len(peaks) > 1:
        changed = False
        for i in range(len(bounds)):
            lo_peak = min(s[peaks[i]], s[peaks[i + 1]])
            if s[bounds[i]] > PROMINENCE * lo_peak:
                keep = peaks[i] if s[peaks[i]] >= s[peaks[i + 1]] else peaks[i + 1]
                peaks[i:i + 2] = [keep]
                del bounds[i]
                changed = True
                break
    edges = [0] + [b + 1 for b in bounds] + [n]
    modes = []
    for j in range(len(peaks)):
        lo, hi = edges[j], edges[j + 1]
        m = float(p[lo:hi].sum())
        c = float((p[lo:hi] * x[lo:hi]).sum() / m) if m > 0 else float(x[peaks[j]])
        # core: contiguous run around the peak, inside the basin, above CORE_FRAC * peak.
        # A basin can run to the edge of the window through a flat tail; the core is
        # where the hump actually is, so "truth under this hump" is tested against it.
        pk = int(peaks[j])
        thr = CORE_FRAC * s[pk]
        a = pk
        while a - 1 >= lo and s[a - 1] >= thr:
            a -= 1
        b = pk
        while b + 1 < hi and s[b + 1] >= thr:
            b += 1
        modes.append({"mass": m, "centroid": c, "lo": lo, "hi": hi, "peak": pk,
                      "core_lo": float(x[a]), "core_hi": float(x[b])})
    modes.sort(key=lambda d: d["mass"], reverse=True)
    return modes


def summarise_column(p: np.ndarray, x: np.ndarray, y: float) -> dict:
    """Everything the analysis needs about one column's posterior and its truth y."""
    mu = float((p * x).sum())
    sd = float(math.sqrt(max((p * (x - mu) ** 2).sum(), 0.0)))
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    modes = find_modes(p, x)
    sig = [m for m in modes if m["mass"] >= MIN_MODE_MASS] or modes[:1]
    step = float(x[1] - x[0])
    in_window = bool(np.isfinite(y) and (x[0] - step / 2) <= y <= (x[-1] + step / 2))
    truth_mass = float(p[np.abs(x - y) <= TRUTH_BAND_FT].sum()) if np.isfinite(y) else float("nan")
    truth_rank = -1  # rank among sig modes of the hump the truth sits under; -1 = none
    if in_window:
        for r, m in enumerate(sig):
            if m["core_lo"] - TRUTH_BAND_FT <= y <= m["core_hi"] + TRUTH_BAND_FT:
                truth_rank = r
                break
    out = {
        "mu": mu,
        "argmax": float(x[int(np.argmax(p))]),
        "sd": sd,
        "entropy": ent,
        "n_sig": len(sig),
        "in_window": in_window,
        "truth_mass": truth_mass,
        "truth_rank": truth_rank,
        "best_mode": float(min((m["centroid"] for m in sig), key=lambda c: abs(c - y))) if np.isfinite(y) else mu,
    }
    for r in range(3):
        out[f"m{r + 1}_centroid"] = sig[r]["centroid"] if r < len(sig) else float("nan")
        out[f"m{r + 1}_mass"] = sig[r]["mass"] if r < len(sig) else 0.0
    return out


def categorise(err_mean: float, col: dict) -> str:
    if not np.isfinite(err_mean) or abs(err_mean) < BAD_FT:
        return "ok"
    if not col["in_window"]:
        return "out_of_window"
    if col["n_sig"] == 1:
        return "unimodal_offset" if col["truth_rank"] == 0 else "unimodal_miss"
    if col["truth_rank"] == 0:
        return "blend"
    if col["truth_rank"] > 0:
        return "wrong_mode"
    return "multimodal_miss"


def analyse_well(well_id: str, logits: np.ndarray, row_bin: np.ndarray, y_row: np.ndarray,
                 x: np.ndarray | None = None) -> dict:
    """Summarise one well.

    logits   (n_bins, n_levels) alignment logits for the well's canvas
    row_bin  (n_rows,) the canvas column each suffix row is read from
    y_row    (n_rows,) true TVT minus the anchor TVT, per suffix row
    """
    if x is None:
        x = level_axis(n_levels=logits.shape[1])
    used = np.unique(row_bin)
    p_all = softmax(logits[used])
    cols = {}
    for k, b in enumerate(used):
        rows = row_bin == b
        yb = float(np.nanmean(y_row[rows])) if np.isfinite(y_row[rows]).any() else float("nan")
        c = summarise_column(p_all[k], x, yb)
        c["bin"] = int(b)
        c["y"] = yb
        c["n_rows"] = int(rows.sum())
        cols[int(b)] = c

    # row-level decodes (each row reads its column's summary)
    def per_row(key):
        return np.array([cols[int(b)][key] for b in row_bin], dtype=np.float64)

    mu = per_row("mu")
    decodes = {
        "mean": mu,
        "argmax": per_row("argmax"),
        "dominant_mode": per_row("m1_centroid"),
        "oracle_best_mode": per_row("best_mode"),
    }
    ok = np.isfinite(y_row)
    err = {k: (v - y_row) for k, v in decodes.items()}
    # oracle ceiling for any per-column mode switch: the better of the mean and the best mode
    err["oracle_mean_or_mode"] = np.where(
        np.abs(err["mean"]) <= np.abs(err["oracle_best_mode"]), err["mean"], err["oracle_best_mode"]
    )
    sse = {k: float(np.nansum(v[ok] ** 2)) for k, v in err.items()}

    # the report's decomposition: remove one offset, or one straight line, per well
    e = err["mean"][ok]
    t = np.arange(row_bin.size, dtype=np.float64)[ok]
    sse["minus_offset"] = float(((e - e.mean()) ** 2).sum()) if e.size else 0.0
    if e.size >= 2:
        A = np.column_stack([np.ones_like(t), t])
        coef, *_ = np.linalg.lstsq(A, e, rcond=None)
        sse["minus_line"] = float(((e - A @ coef) ** 2).sum())
        slope_ft_per_kft = float(coef[1] * 1000.0)
    else:
        sse["minus_line"] = 0.0
        slope_ft_per_kft = 0.0

    cat_row = np.array(
        [categorise(err["mean"][i], cols[int(row_bin[i])]) for i in range(row_bin.size)],
        dtype=object,
    )
    cat_sse = {c: float(np.nansum(np.where(cat_row == c, err["mean"] ** 2, 0.0))) for c in CATEGORIES}
    cat_rows = {c: int((cat_row == c).sum()) for c in CATEGORIES}
    n_multi_rows = int(sum(cols[int(b)]["n_sig"] >= 2 for b in row_bin))
    return {
        "well_id": well_id,
        "n_rows": int(ok.sum()),
        "sse": sse,
        "cat_sse": cat_sse,
        "cat_rows": cat_rows,
        "multimodal_rows": n_multi_rows,
        "line_slope_ft_per_kft": slope_ft_per_kft,
        "columns": [cols[int(b)] for b in used],
    }


def pooled(wells: list[dict], key: str) -> float:
    n = sum(w["n_rows"] for w in wells)
    return math.sqrt(sum(w["sse"][key] for w in wells) / max(n, 1))


def build_report(wells: list[dict], focus: tuple[str, ...] = ()) -> dict:
    n = sum(w["n_rows"] for w in wells)
    total_sse = sum(w["sse"]["mean"] for w in wells)
    decoders = ("mean", "argmax", "dominant_mode", "oracle_best_mode",
                "oracle_mean_or_mode", "minus_offset", "minus_line")
    rep = {
        "wells": len(wells),
        "rows": n,
        "pooled_rmse": {k: pooled(wells, k) for k in decoders},
        "category_share_of_rows": {
            c: sum(w["cat_rows"][c] for w in wells) / max(n, 1) for c in CATEGORIES
        },
        "category_share_of_sse": {
            c: sum(w["cat_sse"][c] for w in wells) / max(total_sse, 1e-12) for c in CATEGORIES
        },
        "multimodal_share_of_rows": sum(w["multimodal_rows"] for w in wells) / max(n, 1),
        "thresholds": {
            "MIN_MODE_MASS": MIN_MODE_MASS, "SMOOTH_SIGMA_BINS": SMOOTH_SIGMA_BINS,
            "PROMINENCE": PROMINENCE, "TRUTH_BAND_FT": TRUTH_BAND_FT, "BAD_FT": BAD_FT,
            "CORE_FRAC": CORE_FRAC,
        },
    }
    ranked = sorted(wells, key=lambda w: w["sse"]["mean"], reverse=True)
    rep["worst10_share_of_sse"] = sum(w["sse"]["mean"] for w in ranked[:10]) / max(total_sse, 1e-12)

    def well_row(w):
        s = w["sse"]["mean"]
        rows = max(w["n_rows"], 1)
        return {
            "well_id": w["well_id"],
            "rmse_mean": math.sqrt(s / rows),
            "rmse_argmax": math.sqrt(w["sse"]["argmax"] / rows),
            "rmse_oracle_mode": math.sqrt(w["sse"]["oracle_mean_or_mode"] / rows),
            "rmse_minus_line": math.sqrt(w["sse"]["minus_line"] / rows),
            "share_of_pooled_sse": s / max(total_sse, 1e-12),
            "multimodal_row_share": w["multimodal_rows"] / rows,
            "line_slope_ft_per_kft": w["line_slope_ft_per_kft"],
            "sse_by_category": {c: w["cat_sse"][c] / max(s, 1e-12) for c in CATEGORIES if c != "ok"},
        }

    rep["worst10"] = [well_row(w) for w in ranked[:10]]
    by_id = {w["well_id"]: w for w in wells}
    rep["focus"] = [well_row(by_id[f]) for f in focus if f in by_id]
    rep["focus_missing"] = [f for f in focus if f not in by_id]
    return rep
