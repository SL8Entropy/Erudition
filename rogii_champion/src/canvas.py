"""The 2D alignment canvas -- the representation the whole solution rests on.

Rows are TVT hypotheses relative to the anchor (the last visible TVT); columns
are MD bins. Pixel (h, m) answers "if the bit were at relative TVT h when we
measured column m, how well would the gamma ray agree with the reference?".
The true path is then a curve to be traced across the image, which turns a
geosteering problem into a dense-prediction vision problem.

Grid follows 1st place: 400 rows x 0.5 ft (+-100 ft) and MD downsampled by 32,
32 visible columns (1024 ft) + 320 target columns (10240 ft) = 352. Measured on
the real training set, the +-100 ft window is well chosen -- exactly 1 well of
773 leaves it -- and the longest prediction region is 10052 ft, so the target
region has to be a little wider than 1st place's stated 10000 ft to cover it.

Two knobs exist for test-time augmentation and are used by the variants:

* `md_phase`  shifts the column grid origin inside one 32 ft column. Column
  averaging is coarse relative to the ~18 ft GR correlation length, so where
  you cut the columns changes the input (aliasing). Averaging over 8 phases
  removes it. (2nd place.)
* `n_rows` / `row_ft` widen the TVT window for wells whose path leaves it.
  (5th place's adaptive canvas.)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import GR_MEAN, GR_STD, Well, initial_structural_rate

MD_STRIDE = 32.0        # ft per column
N_VIS_COLS = 32         # 1024 ft of visible region
N_TGT_COLS = 320        # 10240 ft: the longest real prediction region
                        # is 10052 ft, and 32+320 = 352 = 11*32 needs no padding
N_ROWS = 400
ROW_FT = 0.5


# ---------------------------------------------------------------------- #
@dataclass
class CanvasSpec:
    """Which channel groups go into the image, and at what resolution."""

    n_rows: int = N_ROWS
    row_ft: float = ROW_FT
    md_stride: float = MD_STRIDE
    n_vis_cols: int = N_VIS_COLS
    n_tgt_cols: int = N_TGT_COLS
    use_pf: bool = False            # particle-filter channels
    use_xy: bool = False            # XY-neighbour channels
    use_sibling: bool = False       # V4: sibling-lateral + self-prefix reference
    tw_calib_blend: float = 0.5     # weight of the visible-region recalibration
    tw_calib_bin_ft: float = 1.0

    @property
    def n_cols(self) -> int:
        return self.n_vis_cols + self.n_tgt_cols

    @property
    def channels(self) -> list:
        ch = list(BASE_CHANNELS)
        if self.use_pf:
            ch += list(PF_CHANNELS)
        if self.use_xy:
            ch += list(XY_CHANNELS)
        if self.use_sibling:
            ch += list(SIB_CHANNELS)
        return ch

    @property
    def in_chans(self) -> int:
        return len(self.channels)


BASE_CHANNELS = (
    "tw_gr",           # reference barcode, vertical stripes
    "tw_valid",        # typewell coverage for this row
    "row_coord",       # relative TVT of the row
    "hw_gr_mean",      # lateral GR per column, horizontal stripes
    "hw_gr_nanrate",
    "hw_gr_std",       # within-bin structure rescued from the x32 downsample
    "hw_gr_slope",
    "hw_gr_lastfirst",
    "hw_gr_quad",
    "hw_gr_quadres",
    "hw_vis_tvt",      # visible-region TVT mean (relative)
    "known_col",       # 1 before PS, 0 after
    "mismatch_abs",    # |tw_gr - hw_gr|  -- the heart of the picture
    "mismatch_sgn",    # signed version: which side of the reference we are on
    "vis_tvt_dist",    # |row_tvt - visible TVT|
    "struct_prior",    # dip-projected geometric prior, r0*dMD - dZ
    "struct_prior_dist",  # |row_tvt - struct_prior|   (the z_diff channel)
    "dip_z",           # dZ/dMD per column
)
PF_CHANNELS = ("pf_heat", "pf_dist", "pf_spread")
XY_CHANNELS = ("xy_pred", "xy_dtvt", "xy_dist")
SIB_CHANNELS = ("sib_gr", "sib_valid", "sib_mismatch",
                "self_gr", "self_valid", "self_mismatch", "median_mismatch")


@dataclass
class CanvasMeta:
    """Everything decode needs to map columns back to well rows."""

    col_md: np.ndarray        # (W,) column centre MD
    col_valid: np.ndarray     # (W,) column overlaps the well
    col_known: np.ndarray     # (W,) column is before PS
    row_tvt: np.ndarray       # (H,) relative TVT of each row
    anchor_tvt: float
    anchor_md: float
    y_col: np.ndarray         # (W,) true relative TVT at the column centre
    y_valid: np.ndarray       # (W,) target is defined there
    gr_gap: np.ndarray | None = None   # (H, W) normalised |tw - hw|, for the GR loss
    extras: dict = field(default_factory=dict)


# ---------------------------------------------------------------------- #
# typewell calibration
# ---------------------------------------------------------------------- #
def calibrate_typewell(well: Well, blend: float = 0.5, bin_ft: float = 1.0,
                       extra_profile=None):
    """Re-calibrate the typewell GR using the well's own visible region.

    The lateral log is same-tool, same-well and higher resolution than the
    typewell, so where the visible region covers a TVT range it is the better
    estimate of "what GR looks like at this depth". 1st place bins the visible
    (TVT, GR) pairs, interpolates back onto the typewell axis and blends.

    `extra_profile` is an optional (tvt, gr) pair from sibling laterals (V4).
    """
    tw_tvt, tw_gr = well.tw_tvt, well.tw_gr
    if blend <= 0:
        return tw_gr.copy()

    vis = np.isfinite(well.tvt_input) & np.isfinite(well.gr)
    profiles = []
    if vis.sum() >= 16:
        profiles.append((well.tvt_input[vis], well.gr[vis], 1.0))
    if extra_profile is not None and len(extra_profile[0]) >= 16:
        profiles.append((extra_profile[0], extra_profile[1], 1.0))
    if not profiles:
        return tw_gr.copy()

    lo, hi = float(tw_tvt[0]), float(tw_tvt[-1])
    n_bins = max(int(np.ceil((hi - lo) / bin_ft)), 1)
    edges = lo + np.arange(n_bins + 1) * bin_ft
    centres = 0.5 * (edges[:-1] + edges[1:])

    num = np.zeros(n_bins)
    den = np.zeros(n_bins)
    for t, g, w in profiles:
        idx = np.clip(((t - lo) / bin_ft).astype(int), 0, n_bins - 1)
        np.add.at(num, idx, w * g)
        np.add.at(den, idx, w)
    have = den > 0
    if have.sum() < 4:
        return tw_gr.copy()

    obs = np.interp(tw_tvt, centres[have], num[have] / den[have],
                    left=np.nan, right=np.nan)
    # only trust the observed profile inside its own support
    inside = (tw_tvt >= centres[have][0]) & (tw_tvt <= centres[have][-1])
    out = tw_gr.copy()
    ok = inside & np.isfinite(obs) & np.isfinite(tw_gr)
    out[ok] = (1.0 - blend) * tw_gr[ok] + blend * obs[ok]
    only_obs = inside & np.isfinite(obs) & ~np.isfinite(tw_gr)
    out[only_obs] = obs[only_obs]
    return out


# ---------------------------------------------------------------------- #
# per-column summaries of the lateral log
# ---------------------------------------------------------------------- #
def _column_index(md: np.ndarray, edges: np.ndarray, width: int):
    """Gather indices (W, width) of the raw rows falling in each column."""
    lo = np.searchsorted(md, edges[:-1], side="left")
    hi = np.searchsorted(md, edges[1:], side="left")
    idx = lo[:, None] + np.arange(width)[None, :]
    valid = idx < hi[:, None]
    idx = np.clip(idx, 0, len(md) - 1)
    return idx, valid, lo, hi


def column_features(well: Well, edges: np.ndarray, anchor_tvt: float):
    """Rich per-column summary. Downsampling by 32 would normally destroy the
    within-bin structure, so level / trend / curvature / roughness / quality are
    kept explicitly (1st place)."""
    md = well.md
    counts = np.diff(np.searchsorted(md, edges, side="left"))
    width = int(max(counts.max(), 1))
    width = int(min(width, 4 * int(round(edges[1] - edges[0])) + 8))
    idx, valid, lo, hi = _column_index(md, edges, width)

    gr = (well.gr[idx] - GR_MEAN) / GR_STD
    ok = valid & np.isfinite(gr)
    gr_f = np.where(ok, gr, 0.0)
    n_ok = ok.sum(axis=1)
    n_all = np.maximum(valid.sum(axis=1), 1)

    mean = gr_f.sum(axis=1) / np.maximum(n_ok, 1)
    var = (np.where(ok, (gr - mean[:, None]) ** 2, 0.0).sum(axis=1)
           / np.maximum(n_ok, 1))
    std = np.sqrt(np.maximum(var, 0.0))
    nanrate = 1.0 - n_ok / n_all

    # weighted quadratic fit y ~ c0 + c1 t + c2 t^2 on t in [-1, 1]
    t = np.linspace(-1.0, 1.0, width)[None, :] * np.ones((len(counts), 1))
    X = np.stack([np.ones_like(t), t, t * t], axis=-1)          # (W, width, 3)
    w = ok.astype(np.float64)
    XtWX = np.einsum("wkp,wk,wkq->wpq", X, w, X)
    XtWy = np.einsum("wkp,wk,wk->wp", X, w, gr_f)
    XtWX += 1e-3 * np.eye(3)[None]
    coef = np.linalg.solve(XtWX, XtWy[..., None])[..., 0]        # (W, 3)
    fit = np.einsum("wkp,wp->wk", X, coef)
    resid = np.sqrt((np.where(ok, (gr_f - fit) ** 2, 0.0).sum(axis=1)
                     / np.maximum(n_ok, 1)))

    first = np.where(n_ok > 0, np.take_along_axis(
        gr_f, np.argmax(ok, axis=1)[:, None], 1)[:, 0], 0.0)
    last_pos = width - 1 - np.argmax(ok[:, ::-1], axis=1)
    last = np.where(n_ok > 0, np.take_along_axis(
        gr_f, last_pos[:, None], 1)[:, 0], 0.0)

    # median is robust to GR spikes where the mean is not (5th place's ch12)
    med = np.zeros(len(counts))
    has = n_ok > 0
    if has.any():
        med[has] = np.nanmedian(np.where(ok[has], gr[has], np.nan), axis=1)

    # Columns that fall outside the well have no Z. Leaving them at zero puts a
    # step of thousands of feet at the edge of the data, and np.gradient smears
    # that into the last real column -- so fill them by interpolation first and
    # let `col_valid` mask them afterwards.
    z = well.z[idx]
    n_z = valid.sum(axis=1)
    z_mean = np.where(n_z > 0, np.where(valid, z, 0.0).sum(axis=1)
                      / np.maximum(n_z, 1), np.nan)
    ok_z = np.isfinite(z_mean)
    if ok_z.any():
        z_mean = np.interp(np.arange(len(z_mean)), np.flatnonzero(ok_z),
                           z_mean[ok_z])
    else:
        z_mean = np.zeros(len(z_mean))
    dz = np.gradient(z_mean) / (edges[1] - edges[0])

    ti = well.tvt_input[idx]
    ti_ok = valid & np.isfinite(ti)
    n_ti = ti_ok.sum(axis=1)
    vis_tvt = (np.where(ti_ok, ti - anchor_tvt, 0.0).sum(axis=1)
               / np.maximum(n_ti, 1))

    return dict(
        mean=mean, std=std, nanrate=nanrate, slope=coef[:, 1],
        quad=coef[:, 2], quadres=resid, lastfirst=last - first, median=med,
        z_mean=z_mean, dip=dz, vis_tvt=vis_tvt, has_vis=(n_ti > 0).astype(float),
        col_valid=(counts > 0).astype(float), has_gr=(n_ok > 0).astype(float),
        lo=lo, hi=hi,
    )


# ---------------------------------------------------------------------- #
# the builder
# ---------------------------------------------------------------------- #
def build_canvas(well: Well, spec: CanvasSpec, md_phase: float = 0.0,
                 anchor_tvt: float = None, anchor_md: float = None,
                 extras: dict = None, tw_gr_override=None):
    """Return (C, H, W) float32 image and its `CanvasMeta`.

    `anchor_tvt` / `anchor_md` override the PS anchor, which is what
    re-anchoring TTA needs: pretend a point 25% into the prediction region was
    known, and re-render from there.
    """
    extras = extras or {}
    a_tvt = well.anchor_tvt if anchor_tvt is None else float(anchor_tvt)
    a_md = well.anchor_md if anchor_md is None else float(anchor_md)

    H, W = spec.n_rows, spec.n_cols
    stride = spec.md_stride
    md0 = a_md + md_phase - spec.n_vis_cols * stride
    edges = md0 + np.arange(W + 1) * stride
    col_md = 0.5 * (edges[:-1] + edges[1:])
    row_tvt = (np.arange(H) - (H - 1) / 2.0) * spec.row_ft

    cf = column_features(well, edges, a_tvt)
    col_valid = cf["col_valid"]
    col_known = (col_md < a_md).astype(np.float64) * col_valid

    tw_gr = calibrate_typewell(well, spec.tw_calib_blend, spec.tw_calib_bin_ft) \
        if tw_gr_override is None else np.asarray(tw_gr_override, dtype=np.float64)
    tw_at_row = np.interp(a_tvt + row_tvt, well.tw_tvt, tw_gr,
                          left=np.nan, right=np.nan)
    tw_valid = np.isfinite(tw_at_row).astype(np.float64)
    tw_row = np.nan_to_num((tw_at_row - GR_MEAN) / GR_STD)

    ones_r = np.ones((H, 1))
    ones_c = np.ones((1, W))
    has_gr = cf["has_gr"][None, :]        # zero out mismatch where GR is absent
    R = row_tvt[:, None] * ones_c            # (H, W) row coordinate in ft
    TW = tw_row[:, None] * ones_c
    HW = cf["mean"][None, :] * ones_r

    # Every "position or distance in TVT" channel is clipped to a little beyond
    # the canvas window. The visible columns can reach back into the build
    # section, where the trajectory and the known TVT move by thousands of feet;
    # the exact value there is off-canvas and unusable, and feeding +-3000 in
    # alongside channels of order 1 only wrecks the conditioning.
    CLIP = spec.n_rows * spec.row_ft / 2.0 + 50.0

    # The geometric prior is the *dip-projected* structural path, not a flat
    # layer. Measured over the 773 training wells: assuming flat layers
    # (TVT = anchor - dZ) is wrong by a median of 91 ft per well and leaves the
    # +-100 ft canvas for 88% of wells, because the driller steers down with the
    # formation -- Z moves hundreds of feet while TVT barely moves. Projecting
    # the structural slope estimated from the prefix instead is wrong by 21 ft
    # and stays inside the window for 88%.
    r0 = initial_structural_rate(well)
    z_anchor = float(np.interp(a_md, well.md, well.z))
    struct_prior = np.clip(r0 * (col_md - a_md) - (cf["z_mean"] - z_anchor),
                           -CLIP, CLIP)
    vis_tvt = np.clip(cf["vis_tvt"], -CLIP, CLIP)
    ch = {
        "tw_gr": TW,
        "tw_valid": tw_valid[:, None] * ones_c,
        "row_coord": R / 100.0,
        "hw_gr_mean": HW,
        "hw_gr_nanrate": cf["nanrate"][None, :] * ones_r,
        "hw_gr_std": cf["std"][None, :] * ones_r,
        "hw_gr_slope": cf["slope"][None, :] * ones_r,
        "hw_gr_lastfirst": cf["lastfirst"][None, :] * ones_r,
        "hw_gr_quad": cf["quad"][None, :] * ones_r,
        "hw_gr_quadres": cf["quadres"][None, :] * ones_r,
        "hw_vis_tvt": (vis_tvt * cf["has_vis"])[None, :] * ones_r / 100.0,
        "known_col": col_known[None, :] * ones_r,
        "mismatch_abs": np.abs(TW - HW) * has_gr,
        "mismatch_sgn": (TW - HW) * has_gr,
        "vis_tvt_dist": (np.abs(R - vis_tvt[None, :]) / 100.0
                         * cf["has_vis"][None, :]),
        "struct_prior": struct_prior[None, :] * ones_r / 100.0,
        "struct_prior_dist": np.abs(R - struct_prior[None, :]) / 100.0,
        "dip_z": cf["dip"][None, :] * ones_r,
    }

    if spec.use_pf:
        heat = extras.get("pf_heat")
        pf_mean = extras.get("pf_mean")
        pf_spread = extras.get("pf_spread")
        if heat is None:
            heat = np.zeros((H, W))
            pf_mean = np.zeros(W)
            pf_spread = np.zeros(W)
        pf_mean = np.clip(pf_mean, -CLIP, CLIP)
        pf_spread = np.clip(pf_spread, 0.0, CLIP)
        ch["pf_heat"] = heat
        ch["pf_dist"] = np.abs(R - pf_mean[None, :]) / 100.0
        ch["pf_spread"] = pf_spread[None, :] * ones_r / 100.0

    if spec.use_xy:
        xy = extras.get("xy_pred")
        if xy is None:
            xy = np.zeros(W)
        xy = np.clip(xy, -CLIP, CLIP)
        # as a rate in ft per ft, so it is on the same scale as dip_z and the
        # two can be compared directly by the network
        dxy = np.clip(np.gradient(xy) / spec.md_stride, -0.5, 0.5)
        ch["xy_pred"] = xy[None, :] * ones_r / 100.0
        ch["xy_dtvt"] = dxy[None, :] * ones_r
        ch["xy_dist"] = np.abs(R - xy[None, :]) / 100.0

    if spec.use_sibling:
        sib = extras.get("sib_gr_row")          # (H,) already on the row axis
        sib_v = extras.get("sib_valid_row")
        slf = extras.get("self_gr_row")
        slf_v = extras.get("self_valid_row")
        zero_r = np.zeros(H)
        sib = zero_r if sib is None else sib
        sib_v = zero_r if sib_v is None else sib_v
        slf = zero_r if slf is None else slf
        slf_v = zero_r if slf_v is None else slf_v
        SB = sib[:, None] * ones_c
        SF = slf[:, None] * ones_c
        ch["sib_gr"] = SB
        ch["sib_valid"] = sib_v[:, None] * ones_c
        ch["sib_mismatch"] = np.abs(SB - HW) * (sib_v[:, None] * ones_c) * has_gr
        ch["self_gr"] = SF
        ch["self_valid"] = slf_v[:, None] * ones_c
        ch["self_mismatch"] = np.abs(SF - HW) * (slf_v[:, None] * ones_c) * has_gr
        ch["median_mismatch"] = np.abs(TW - cf["median"][None, :] * ones_r) * has_gr

    names = spec.channels
    img = np.stack([ch[n] for n in names], axis=0).astype(np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img *= col_valid[None, None, :].astype(np.float32)

    y_col, y_valid = _target_columns(well, col_md, a_tvt)
    gr_gap = (np.abs(TW - HW) * tw_valid[:, None] * has_gr).astype(np.float32)
    meta = CanvasMeta(col_md=col_md, col_valid=col_valid.astype(bool),
                      col_known=col_known.astype(bool), row_tvt=row_tvt,
                      anchor_tvt=a_tvt, anchor_md=a_md,
                      y_col=y_col, y_valid=y_valid, gr_gap=gr_gap,
                      extras=dict(dip=cf["dip"], struct_prior=struct_prior,
                                  z_mean=cf["z_mean"], struct_rate=r0))
    return img, meta


def _target_columns(well: Well, col_md: np.ndarray, anchor_tvt: float):
    """True relative TVT at each column centre (NaN where undefined)."""
    if well.tvt is None:
        return np.full(len(col_md), np.nan), np.zeros(len(col_md), dtype=bool)
    ok = np.isfinite(well.tvt)
    if ok.sum() < 2:
        return np.full(len(col_md), np.nan), np.zeros(len(col_md), dtype=bool)
    y = np.interp(col_md, well.md[ok], well.tvt[ok],
                  left=np.nan, right=np.nan) - anchor_tvt
    inside = (col_md >= well.md[ok][0]) & (col_md <= well.md[ok][-1])
    valid = inside & np.isfinite(y)
    return y, valid


def columns_to_rows(pred_col: np.ndarray, meta: CanvasMeta, md_query: np.ndarray):
    """Interpolate a per-column relative-TVT path onto raw well rows -> absolute TVT."""
    ok = meta.col_valid
    if ok.sum() < 2:
        return np.full(len(md_query), meta.anchor_tvt)
    rel = np.interp(md_query, meta.col_md[ok], pred_col[ok])
    return rel + meta.anchor_tvt
