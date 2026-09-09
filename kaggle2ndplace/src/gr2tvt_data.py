#!/usr/bin/env python3
"""Input and output representation for the GR -> TVT inverse model.

Grid:
  vertical axis   typewell level (TVT - TVT_PS), a +-128 ft window at 0.5 ft per row
  horizontal axis measured depth over the evaluation zone, 32 ft per column by default
  normalisation   a single global constant applied to GR: (GR - 88) / 30
  anchor          TVT_PS, the last known TVT_input immediately before the evaluation
                  zone; it is available at test time

Channels:
  0  GR difference map between the well's column GR and the typewell's row GR
  1  the column GR broadcast across rows, which restores the absolute level that the
     difference in channel 0 removes
  2  fraction of valid GR samples in the column, broadcast across rows
  3  typewell coverage flag per row, marking where the row value is padded

Label: per column, the mean TVT within the column minus TVT_PS. Columns with no coverage
are NaN. cols_to_md() converts a per-column prediction back to arbitrary MD by linear
interpolation between column centres.
"""

from pathlib import Path

import numpy as np
import polars as pl

WIN = 128.0
ROW = 0.5
COLW = 32.0
T = int(2 * WIN / ROW)
H = 320
GR_MU, GR_SD = 88.0, 30.0
DZ_SLOPE_SD = 0.044
ZREL_SD = 128.0
LEVELS = np.arange(-WIN + ROW / 2, WIN, ROW)
NULLS = ["nan", "NaN", "NA", ""]


def load_well(train_dir: Path, name: str, with_buda: bool = False) -> dict:
    cols = ["MD", "X", "Y", "Z", "TVT", "GR", "TVT_input"] + (["BUDA"] if with_buda else [])
    df = pl.read_csv(Path(train_dir) / f"{name}__horizontal_well.csv", null_values=NULLS,
                     columns=cols).sort("MD")
    tw = pl.read_csv(Path(train_dir) / f"{name}__typewell.csv", null_values=NULLS)
    m = np.isfinite(tw["TVT"].cast(pl.Float64).to_numpy()) & np.isfinite(tw["GR"].cast(pl.Float64).to_numpy())
    twt = tw["TVT"].cast(pl.Float64).to_numpy()[m]
    twg = tw["GR"].cast(pl.Float64).to_numpy()[m]
    o = np.argsort(twt)
    out = dict(
        md=df["MD"].cast(pl.Float64).to_numpy(),
        x=df["X"].cast(pl.Float64).to_numpy(),
        y=df["Y"].cast(pl.Float64).to_numpy(),
        z=df["Z"].cast(pl.Float64).to_numpy(),
        tvt=df["TVT"].cast(pl.Float64).to_numpy(),
        gr=df["GR"].cast(pl.Float64).to_numpy(),
        tvt_input=df["TVT_input"].cast(pl.Float64).to_numpy(),
        tw_tvt=twt[o], tw_gr=twg[o],
    )
    if with_buda:
        out["buda"] = df["BUDA"].cast(pl.Float64).to_numpy()
    return out


def attach_z_layer(w: dict) -> dict:
    tvt, z, md = w["tvt"], w["z"], w["md"]
    ok = np.isfinite(tvt) & np.isfinite(z)
    zl = np.where(ok, tvt + z, np.nan)
    gap = ~np.isfinite(zl)
    if gap.any():
        buda = w.get("buda")
        if buda is not None:
            bok = ok & np.isfinite(buda)
            if bok.sum() >= 2:
                b_hat = float(np.median(zl[bok] - buda[bok]))
                fill = gap & np.isfinite(buda)
                zl[fill] = buda[fill] + b_hat
                gap = ~np.isfinite(zl)
        if gap.any() and (~gap).sum() >= 2:
            zl[gap] = np.interp(md[gap], md[~gap], zl[~gap])
    return dict(w, z_layer=zl)


def pre_ps_gr_profile(w: dict, i0: int, anchor0: float) -> tuple[np.ndarray, np.ndarray]:
    s_n = np.zeros(T, dtype=np.float32)
    s_cover = np.zeros(T, dtype=np.float32)
    ti, gr = w["tvt_input"][:i0], w["gr"][:i0]
    ok = np.isfinite(ti) & np.isfinite(gr)
    if ok.sum() < 2:
        return s_n, s_cover
    idx = np.floor((ti[ok] - anchor0 + WIN) / ROW).astype(np.int64)
    m = (idx >= 0) & (idx < T)
    if not m.any():
        return s_n, s_cover
    idx, g = idx[m], gr[ok][m]
    cnt = np.bincount(idx, minlength=T).astype(np.float64)
    gsum = np.bincount(idx, weights=g, minlength=T)
    hit = cnt > 0
    prof = np.full(T, np.nan)
    prof[hit] = gsum[hit] / cnt[hit]
    lo = int(np.argmax(hit))
    hi = T - 1 - int(np.argmax(hit[::-1]))
    cov = np.zeros(T, dtype=bool)
    cov[lo:hi + 1] = True
    gap = cov & ~hit
    if gap.any():
        prof[gap] = np.interp(np.where(gap)[0], np.where(hit)[0], prof[hit])
    s_n[cov] = ((prof[cov] - GR_MU) / GR_SD).astype(np.float32)
    s_cover[cov] = 1.0
    return s_n, s_cover


def build_compact(w: dict, with_label: bool = True, colw_in: float | None = None,
                  pre_ps_cols: int = 0, pre_ps_label: bool = False,
                  tvt_shift: float = 0.0, pre_ps_gr: bool = False,
                  z_dip: bool = False, z_rel: bool = False,
                  md_phase: float = 0.0, in_phase: float = 0.0) -> dict:
    fin_in = np.isfinite(w["tvt_input"])
    ev = ~fin_in
    if not ev.any():
        raise ValueError("no eval zone (TVT_input all finite)")
    i0 = int(np.argmax(ev))
    ik = np.where(fin_in[:i0])[0]
    if len(ik) == 0:
        raise ValueError("no known TVT_input before eval zone")
    tvt_ps = float(w["tvt_input"][ik[-1]])
    anchor0 = tvt_ps - float(tvt_shift)

    P = int(pre_ps_cols)
    H_TOT = H + P
    md_ev0 = float(w["md"][ev][0])
    md0 = md_ev0 - P * COLW - float(md_phase)
    sel = w["md"] >= md0
    md_e = w["md"][sel]
    gr_e = w["gr"][sel]
    b = ((md_e - md0) // COLW).astype(np.int64)
    ncol_tot = min(int(b.max()) + 1, H_TOT)
    keep = b < H_TOT
    b, md_e, gr_e = b[keep], md_e[keep], gr_e[keep]
    ncol = ncol_tot - P

    cw = float(colw_in if colw_in else COLW)
    if not (COLW / cw).is_integer():
        raise ValueError(f"COLW({COLW}) must be an integer multiple of colw_in({cw})")
    h_len = int(H_TOT * COLW / cw)
    bi = ((md_e - md0 + float(in_phase)) // cw).astype(np.int64)
    ki = bi < h_len
    bi_k, gr_k = bi[ki], gr_e[ki]
    ncol_in = min(int(bi.max()) + 1, h_len)

    cnt = np.bincount(bi_k, minlength=h_len).astype(np.float64)
    gv = np.isfinite(gr_k)
    gcnt = np.bincount(bi_k[gv], minlength=h_len).astype(np.float64)
    gsum = np.bincount(bi_k[gv], weights=gr_k[gv], minlength=h_len)
    h_gr = np.full(h_len, np.nan)
    ok = gcnt > 0
    h_gr[ok] = gsum[ok] / gcnt[ok]
    h_valid = np.zeros(h_len)
    real = cnt > 0
    h_valid[real] = gcnt[real] / cnt[real]
    miss = real & ~ok
    if miss.any():
        if ok.any():
            h_gr[miss] = np.interp(np.where(miss)[0], np.where(ok)[0], h_gr[ok])
        else:
            h_gr[miss] = GR_MU
    h_gr[~real] = 0.0

    t_gr = np.interp(anchor0 + LEVELS, w["tw_tvt"], w["tw_gr"])
    t_cover = ((anchor0 + LEVELS >= w["tw_tvt"][0]) & (anchor0 + LEVELS <= w["tw_tvt"][-1])).astype(np.float32)

    t_n = ((t_gr - GR_MU) / GR_SD).astype(np.float32)
    h_n = np.zeros(h_len, dtype=np.float32)
    h_n[real] = ((h_gr[real] - GR_MU) / GR_SD).astype(np.float32)

    y = np.full(H_TOT, np.nan)
    y_bnd = np.full(H_TOT + 1, np.nan, dtype=np.float32)
    if with_label and "tvt" in w and w["tvt"] is not None:
        tvt_sel = w["tvt"][sel][keep]
        in_ev = (b >= P)
        sup = in_ev if not pre_ps_label else np.ones_like(in_ev)
        tv = np.isfinite(tvt_sel) & sup
        tcnt = np.bincount(b[tv], minlength=H_TOT).astype(np.float64)
        tsum = np.bincount(b[tv], weights=tvt_sel[tv], minlength=H_TOT)
        lok = tcnt > 0
        y[lok] = tsum[lok] / tcnt[lok] - anchor0
        if tv.sum() >= 2:
            k0 = 0 if pre_ps_label else P
            xb = np.arange(k0, ncol_tot + 1) * COLW
            y_bnd[k0: ncol_tot + 1] = np.interp(xb, (md_e - md0)[tv], tvt_sel[tv]) - anchor0

    k_lv = np.zeros(h_len, dtype=np.float32)
    k_flag = np.zeros(h_len, dtype=np.float32)
    if P > 0:
        ti = w["tvt_input"][sel][keep]
        kv = np.isfinite(ti) & (bi_k < int(P * COLW / cw))
        if kv.any():
            kcnt = np.bincount(bi_k[kv], minlength=h_len).astype(np.float64)
            ksum = np.bincount(bi_k[kv], weights=(ti[kv] - anchor0), minlength=h_len)
            m = kcnt > 0
            k_lv[m] = (ksum[m] / kcnt[m]).astype(np.float32)
            k_flag[m] = 1.0

    h_cnt = gcnt.astype(np.float32)
    h_rows = cnt.astype(np.float32)
    d_n = None
    zr_n = None
    if z_dip or z_rel:
        if "z" not in w or w["z"] is None:
            raise ValueError("z_dip=True requires w['z'] (supplied by load_well / derive_z)")
        z_sel = w["z"][sel][keep][ki]
        zv = np.isfinite(z_sel)
        zcnt = np.bincount(bi_k[zv], minlength=h_len).astype(np.float64)
        zsum = np.bincount(bi_k[zv], weights=z_sel[zv], minlength=h_len)
        zok = zcnt > 0
        d_n = np.zeros(h_len, dtype=np.float32)
        if zok.sum() >= 2:
            zc = np.full(h_len, np.nan)
            zc[zok] = zsum[zok] / zcnt[zok]
            j0, j1 = int(np.argmax(zok)), h_len - 1 - int(np.argmax(zok[::-1]))
            gap = ~zok
            gap[:j0] = False
            gap[j1 + 1:] = False
            if gap.any():
                zc[gap] = np.interp(np.where(gap)[0], np.where(zok)[0], zc[zok])
            dz = np.diff(zc[j0: j1 + 1]) / (cw * DZ_SLOPE_SD)
            d_n[j0 + 1: j1 + 1] = dz.astype(np.float32)
            d_n[j0] = d_n[j0 + 1] if j1 > j0 else 0.0
        if z_rel:
            zr_n = np.zeros(h_len, dtype=np.float32)
            if zok.sum() >= 2:
                acc = np.zeros(h_len)
                acc[j0 + 1: j1 + 1] = np.cumsum(d_n[j0 + 1: j1 + 1].astype(np.float64)
                                                * cw * DZ_SLOPE_SD)
                acc[j1 + 1:] = acc[j1]
                ps_j = min(int(P * COLW / cw), h_len - 1)
                zr_n = ((acc - acc[ps_j]) / ZREL_SD).astype(np.float32)
        if not z_dip:
            d_n = None
    s_n, s_cover = pre_ps_gr_profile(w, i0, anchor0) if pre_ps_gr else (None, None)
    return dict(t_n=t_n, h_n=h_n, h_valid=h_valid.astype(np.float32), t_cover=t_cover,
                y=y, y_bnd=y_bnd, h_cnt=h_cnt, h_rows=h_rows, k_lv=k_lv, k_flag=k_flag,
                s_n=s_n, s_cover=s_cover, d_n=d_n, zr_n=zr_n,
                meta=dict(tvt_ps=tvt_ps, md0=md0, ncol=ncol, ncol_tot=ncol_tot, ps_col=P,
                          ncol_in=ncol_in, colw_in=cw, tvt_shift=float(tvt_shift)))


def assemble_np(c: dict) -> np.ndarray:
    ncol = c["meta"]["ncol"]
    act = (np.arange(H) < ncol).astype(np.float32)
    x = np.zeros((4, T, H), dtype=np.float32)
    x[0] = (c["h_n"][None, :] - c["t_n"][:, None]) * act[None, :]
    x[1] = np.broadcast_to(c["h_n"][None, :] * act[None, :], (T, H))
    x[2] = np.broadcast_to(c["h_valid"][None, :] * act[None, :], (T, H))
    x[3] = np.broadcast_to(c["t_cover"][:, None], (T, H))
    return x


def build_sample(w: dict, with_label: bool = True) -> dict:
    c = build_compact(w, with_label=with_label)
    col_mask = np.arange(H) < c["meta"]["ncol"]
    return dict(x=assemble_np(c), y=c["y"], col_mask=col_mask, meta=c["meta"])


def interp_weights(xc: np.ndarray, xq: np.ndarray):
    i1 = np.searchsorted(xc, xq).clip(1, len(xc) - 1)
    i0 = i1 - 1
    dx = xc[i1] - xc[i0]
    w1 = np.where(dx > 0, (xq - xc[i0]) / np.where(dx > 0, dx, 1.0), 0.0).clip(0.0, 1.0)
    w1 = np.where(xq <= xc[0], 0.0, w1)
    w1 = np.where(xq >= xc[-1], 1.0, w1)
    return i0, i1, (1.0 - w1), w1


def col_centers(meta: dict) -> np.ndarray:
    p0, nc = meta.get("ps_col", 0), meta["ncol"]
    return meta["md0"] + (p0 + np.arange(nc) + 0.5) * COLW


def cols_to_md(pred_lv: np.ndarray, meta: dict, md_query: np.ndarray) -> np.ndarray:
    nc = meta["ncol"]
    p0 = meta.get("ps_col", 0)
    xc = meta["md0"] + (p0 + np.arange(nc) + 0.5) * COLW
    lv = np.asarray(pred_lv, dtype=np.float64)[p0: p0 + nc]
    return meta["tvt_ps"] - meta.get("tvt_shift", 0.0) + np.interp(md_query, xc, lv)
