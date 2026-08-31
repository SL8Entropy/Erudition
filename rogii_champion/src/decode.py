"""Decoding a canvas into a TVT path, and the test-time augmentations.

Four things live here, three of which are borrowed:

* **expectation decode** (1st place) -- softmax down each column, take the mean
  over TVT. The distribution is collapsed to a point estimate only at the very
  last step.
* **MD-phase TTA** (2nd place) -- the column grid is 32 ft wide but the GR
  correlation length is ~18 ft, so *where* the columns are cut changes the
  input. Average over sub-column phases to remove the aliasing.
* **quarter re-anchoring** (4th place) -- predict the whole well, then pretend
  the prediction a quarter of the way in was known, re-render, and predict the
  rest. Averaging the two halves the long-range drift.
* **DP expectation over anchor moves** (2nd place) -- marginalise the
  conditional move field exactly, column by column, instead of reading each
  column independently.
* **adaptive canvas** (5th place) -- re-run the few wells whose path leaves the
  TVT window on a taller canvas.
"""
from __future__ import annotations

import numpy as np
import torch

from .canvas import build_canvas, columns_to_rows


# ---------------------------------------------------------------------- #
def md_phases(n: int, stride: float = 32.0):
    """Sub-column offsets. n=1 disables the augmentation."""
    return (np.arange(n) * (stride / max(n, 1))).tolist()


@torch.no_grad()
def forward_view(model, well, spec, cfg, provider, device, md_phase=0.0,
                 anchor_tvt=None, anchor_md=None, amp_dtype=torch.bfloat16):
    """Build one canvas view, run the model, return the per-column path."""
    a_tvt = well.anchor_tvt if anchor_tvt is None else anchor_tvt
    a_md = well.anchor_md if anchor_md is None else anchor_md

    stride = spec.md_stride
    md0 = a_md + md_phase - spec.n_vis_cols * stride
    edges = md0 + np.arange(spec.n_cols + 1) * stride
    col_md = 0.5 * (edges[:-1] + edges[1:])
    row_tvt = (np.arange(spec.n_rows) - (spec.n_rows - 1) / 2.0) * spec.row_ft

    extras = provider.extras(well, col_md, row_tvt, a_tvt) if provider else {}
    tw = provider.typewell_gr(well) if provider is not None else None
    img, meta = build_canvas(well, spec, md_phase=md_phase, anchor_tvt=a_tvt,
                             anchor_md=a_md, extras=extras, tw_gr_override=tw)

    x = torch.from_numpy(img).unsqueeze(0).to(device)
    with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype,
                        enabled=(device != "cpu")):
        out = model(x)
    logits = out["logits"].float()
    rows = torch.from_numpy(meta.row_tvt).float().to(device)
    p = torch.softmax(logits, dim=1)
    pred_col = (p * rows.view(1, -1, 1)).sum(dim=1)[0].cpu().numpy()

    res = dict(pred_col=pred_col, meta=meta, col_md=col_md, row_tvt=meta.row_tvt)
    if "move_logits" in out:
        res["dp_col"] = dp_expectation(
            out["move_logits"][0].float(), meta.row_tvt, cfg.move_row_stride,
            cfg.move_step, meta.col_known)
    if "log_sigma" in out:
        res["sigma"] = torch.sqrt(
            torch.nn.functional.softplus(out["log_sigma"][0].float()) + 1e-3
        ).cpu().numpy()
    if "dz" in out:
        res["dz"] = out["dz"][0].float().cpu().numpy()
    return res


# ---------------------------------------------------------------------- #
def dp_expectation(move_logits, row_tvt, row_stride: int, move_step: float,
                   col_known: np.ndarray):
    """Exact marginalisation of P(dTVT | TVT) along MD (2nd place, eq. 5-6).

    Starting from a point mass at the anchor level, propagate the level
    marginal column by column and report its mean. No sampling: the whole
    distribution is carried forward, so ambiguity survives to the last step.
    """
    M, Hs, W = move_logits.shape
    half = (M - 1) // 2
    P = torch.softmax(move_logits, dim=0)                 # (M, Hs, W)
    levels = torch.from_numpy(row_tvt[::row_stride][:Hs].copy()).float().to(P.device)

    start = int(np.argmax(~col_known)) if (~col_known).any() else 0
    p = torch.zeros(Hs, device=P.device)
    p[int(torch.argmin(levels.abs()))] = 1.0

    out = np.zeros(W, dtype=np.float64)
    out[:start] = 0.0
    for k in range(start, W):
        out[k] = float((p * levels).sum())
        if k + 1 >= W:
            break
        nxt = torch.zeros_like(p)
        contrib = p.unsqueeze(0) * P[:, :, k]             # (M, Hs)
        for j in range(M):
            sh = j - half
            if sh == 0:
                nxt += contrib[j]
            elif sh > 0:
                nxt[sh:] += contrib[j][:-sh]
            else:
                nxt[:sh] += contrib[j][-sh:]
        s = nxt.sum()
        p = nxt / s if s > 1e-12 else p
    return out


# ---------------------------------------------------------------------- #
def _extend_prefix(well, rel_pred, meta, upto_md: float):
    """Copy of the well with the model's own prediction promoted to 'known'."""
    w = well.copy()
    abs_pred = columns_to_rows(rel_pred, meta, well.md)
    fill = (~np.isfinite(w.tvt_input)) & (w.md <= upto_md)
    w.tvt_input = w.tvt_input.copy()
    w.tvt_input[fill] = abs_pred[fill]
    return w


def predict_well(model, well, spec, cfg, provider, device,
                 n_phases: int = 1, reanchor: float = 0.0,
                 adaptive_canvas: bool = False, dp_weight: float = 0.0,
                 amp_dtype=torch.bfloat16, return_detail: bool = False):
    """Full inference for one well -> absolute TVT on the raw MD grid."""
    phases = md_phases(n_phases, spec.md_stride)
    acc = np.zeros(well.n)
    sigma_acc = np.zeros(well.n)
    n_acc = 0
    for ph in phases:
        r = forward_view(model, well, spec, cfg, provider, device, md_phase=ph,
                         amp_dtype=amp_dtype)
        col = r["pred_col"]
        if dp_weight > 0 and "dp_col" in r:
            col = (1.0 - dp_weight) * col + dp_weight * r["dp_col"]
        acc += columns_to_rows(col, r["meta"], well.md) - well.anchor_tvt
        if "sigma" in r:
            sigma_acc += np.interp(well.md, r["col_md"], r["sigma"])
        n_acc += 1
    rel = acc / max(n_acc, 1)
    sigma = sigma_acc / max(n_acc, 1) if n_acc else None

    # 5th place: only the wells that need the taller canvas pay for it
    if adaptive_canvas:
        half_window = spec.n_rows * spec.row_ft / 2.0
        if np.nanmax(np.abs(rel)) > 0.55 * half_window:
            wide = _widen(spec)
            acc = np.zeros(well.n)
            for ph in phases:
                r = forward_view(model, well, wide, cfg, provider, device,
                                 md_phase=ph, amp_dtype=amp_dtype)
                col = r["pred_col"]
                if dp_weight > 0 and "dp_col" in r:
                    col = (1.0 - dp_weight) * col + dp_weight * r["dp_col"]
                acc += columns_to_rows(col, r["meta"], well.md) - well.anchor_tvt
            rel = acc / max(len(phases), 1)

    # 4th place: re-anchor a quarter of the way into the prediction region
    if reanchor > 0:
        md_end = float(well.md[-1])
        md_q = well.anchor_md + reanchor * (md_end - well.anchor_md)
        if md_q < md_end - 500.0:
            base_meta = _fake_meta(well)
            w2 = _extend_prefix(well, rel, base_meta, md_q)
            acc2 = np.zeros(well.n)
            n2 = 0
            for ph in phases:
                r = forward_view(model, w2, spec, cfg, provider, device,
                                 md_phase=ph, amp_dtype=amp_dtype)
                col = r["pred_col"]
                if dp_weight > 0 and "dp_col" in r:
                    col = (1.0 - dp_weight) * col + dp_weight * r["dp_col"]
                acc2 += columns_to_rows(col, r["meta"], well.md)
                n2 += 1
            second = acc2 / max(n2, 1) - well.anchor_tvt
            tail = well.md > md_q
            rel = rel.copy()
            rel[tail] = 0.5 * rel[tail] + 0.5 * second[tail]

    pred = rel + well.anchor_tvt
    vis = np.isfinite(well.tvt_input)
    pred[vis] = well.tvt_input[vis]          # the given prefix is never overwritten
    if return_detail:
        return pred, dict(sigma=sigma)
    return pred


def _widen(spec, factor: float = 1.6):
    import copy
    wide = copy.copy(spec)
    wide.n_rows = int(round(spec.n_rows * factor / 32) * 32)
    return wide


def _fake_meta(well):
    """Identity meta so `columns_to_rows` can be reused on a per-row array."""
    from .canvas import CanvasMeta
    return CanvasMeta(col_md=well.md, col_valid=np.ones(well.n, dtype=bool),
                      col_known=np.isfinite(well.tvt_input),
                      row_tvt=np.zeros(1), anchor_tvt=well.anchor_tvt,
                      anchor_md=well.anchor_md,
                      y_col=np.zeros(well.n), y_valid=np.zeros(well.n, dtype=bool))

