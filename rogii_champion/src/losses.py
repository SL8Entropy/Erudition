"""Losses.

The main objective is 1st place's: cross-entropy against an exponentially
smoothed distribution over TVT bins, normalised down each column. Two
auxiliaries ride along -- a Huber loss on the *decoded* expected path, so part
of the objective is computed on the actual answer rather than an intermediate
map, and a GR penalty that pushes probability mass toward cells where the
gamma ray actually agrees.

The variant-specific losses are here too: the structural-slope head (V2), the
teacher-forced anchor move distribution (V3) and Gaussian NLL (V5).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def column_log_prob(logits: torch.Tensor) -> torch.Tensor:
    """(B, H, W) logits -> log-probabilities normalised along the TVT axis."""
    return F.log_softmax(logits, dim=1)


def expected_path(logits: torch.Tensor, row_tvt: torch.Tensor) -> torch.Tensor:
    """Collapse the distribution to a point estimate -- at the very last step."""
    p = torch.softmax(logits, dim=1)
    return (p * row_tvt.view(1, -1, 1)).sum(dim=1)


def smoothed_ce(logits, y_col, row_tvt, valid, tau: float = 2.0,
                col_weight=None):
    """Cross-entropy against exp(-|row - truth| / tau), normalised per column.

    A hard one-hot target would tell the model that a bin 1 ft away is as wrong
    as one 50 ft away. The smoothed target keeps the geometry of the TVT axis
    in the loss.
    """
    logp = column_log_prob(logits)                        # (B, H, W)
    d = (row_tvt.view(1, -1, 1) - y_col.unsqueeze(1)).abs()
    q = torch.softmax(-d / tau, dim=1)
    ce = -(q * logp).sum(dim=1)                           # (B, W)
    w = valid.float()
    if col_weight is not None:
        w = w * col_weight
    return (ce * w).sum() / w.sum().clamp_min(1.0)


def path_huber(pred, y_col, valid, delta: float = 5.0, col_weight=None):
    d = pred - y_col
    a = d.abs()
    loss = torch.where(a <= delta, 0.5 * d * d, delta * (a - 0.5 * delta))
    w = valid.float()
    if col_weight is not None:
        w = w * col_weight
    return (loss * w).sum() / w.sum().clamp_min(1.0)


def gr_penalty(logits, gr_gap, valid):
    """mean(probability x GR gap): a soft physical prior on where mass may go."""
    p = torch.softmax(logits, dim=1)
    pen = (p * gr_gap).sum(dim=1)
    w = valid.float()
    return (pen * w).sum() / w.sum().clamp_min(1.0)


def gaussian_nll(pred, y_col, log_sigma, valid, min_var: float = 1e-3):
    """0.5 [log s^2 + (y - mu)^2 / s^2]  -- the model reports its own confidence.

    3rd place measured a real gain from swapping RMSE for this on the same
    network, and the predicted sigma is what the ensemble gate keys on.
    """
    var = F.softplus(log_sigma) + min_var
    nll = 0.5 * (torch.log(var) + (y_col - pred) ** 2 / var)
    w = valid.float()
    return (nll * w).sum() / w.sum().clamp_min(1.0)


# ---------------------------------------------------------------------- #
# V2: structural slope
# ---------------------------------------------------------------------- #
def dz_layer_loss(dz_pred, y_col, dip, valid, md_stride: float,
                  delta: float = 0.02, integ_weight: float = 1.0,
                  integ_delta: float = 5.0):
    """Supervise dz_layer directly, and the path it integrates to.

    dTVT = dz_layer - dz is an identity, so predicting the smooth structural
    slope and integrating it against the *known* trajectory is a strictly
    easier problem than predicting TVT: the coefficient on dz is fixed at -1
    and does not have to be learned.
    """
    dy = torch.zeros_like(y_col)
    dy[:, 1:] = y_col[:, 1:] - y_col[:, :-1]
    v2 = valid.clone()
    v2[:, 0] = False
    v2[:, 1:] = valid[:, 1:] & valid[:, :-1]

    dz_true = dy / md_stride + dip                     # dz_layer/dMD
    d = dz_pred - dz_true
    a = d.abs()
    hub = torch.where(a <= delta, 0.5 * d * d, delta * (a - 0.5 * delta))
    w = v2.float()
    loss = (hub * w).sum() / w.sum().clamp_min(1.0) / max(delta, 1e-6)

    if integ_weight > 0:
        step = (dz_pred - dip) * md_stride             # dTVT per column
        path = torch.cumsum(step * v2.float(), dim=1)
        anchor = _anchor_offset(path, y_col, valid)
        loss = loss + integ_weight * path_huber(
            path - anchor, y_col, valid, delta=integ_delta)
    return loss


def _anchor_offset(path, y_col, valid):
    """Align an integrated path to the anchor column (the last known column)."""
    B, W = path.shape
    idx = torch.zeros(B, dtype=torch.long, device=path.device)
    for b in range(B):
        v = torch.nonzero(valid[b], as_tuple=False)
        idx[b] = v[0, 0] if len(v) else 0
    off = path.gather(1, idx.view(-1, 1)) - y_col.gather(1, idx.view(-1, 1))
    return off


# ---------------------------------------------------------------------- #
# V3: anchor move distribution
# ---------------------------------------------------------------------- #
def move_ce(move_logits, y_col, row_tvt, valid, move_step: float = 2.0,
            row_stride: int = 4):
    """Teacher-forced cross-entropy on P(dTVT | TVT) along the true path.

    Only the anchor sitting on the ground-truth level at each column receives a
    loss. The other anchors are trained implicitly by convolutional weight
    sharing, which is exactly why this head can be supervised so cheaply.
    """
    B, M, Hs, W = move_logits.shape
    half = (M - 1) // 2
    row_lo = row_tvt[0]
    row_step = (row_tvt[1] - row_tvt[0]) * row_stride

    lvl = ((y_col - row_lo) / row_step).round().long().clamp(0, Hs - 1)
    dy = torch.zeros_like(y_col)
    dy[:, :-1] = y_col[:, 1:] - y_col[:, :-1]
    m = dy / move_step
    m_lo = m.floor().clamp(-half, half - 1e-6)
    frac = (m - m_lo).clamp(0.0, 1.0)
    i_lo = (m_lo + half).long().clamp(0, M - 1)
    i_hi = (i_lo + 1).clamp(0, M - 1)

    logp = F.log_softmax(move_logits, dim=1)                   # (B, M, Hs, W)
    idx = lvl.view(B, 1, 1, W).expand(B, M, 1, W)
    logp_at = logp.gather(2, idx).squeeze(2)                   # (B, M, W)

    ll = ((1.0 - frac) * logp_at.gather(1, i_lo.unsqueeze(1)).squeeze(1)
          + frac * logp_at.gather(1, i_hi.unsqueeze(1)).squeeze(1))
    v = valid.clone()
    v[:, -1] = False
    v[:, :-1] = valid[:, :-1] & valid[:, 1:]
    w = v.float()
    return -(ll * w).sum() / w.sum().clamp_min(1.0)


# ---------------------------------------------------------------------- #
def total_loss(out, batch, cfg, row_tvt):
    """Assemble the loss stack for whichever heads this variant has."""
    logits = out["logits"]
    y = batch["y_col"]
    valid = batch["y_valid"]
    cw = batch.get("col_weight")

    parts = {}
    parts["ce"] = cfg.w_ce * smoothed_ce(logits, y, row_tvt, valid,
                                         tau=cfg.ce_tau, col_weight=cw)
    pred = expected_path(logits, row_tvt)
    parts["huber"] = cfg.w_huber * path_huber(pred, y, valid,
                                              delta=cfg.huber_delta, col_weight=cw)
    if cfg.w_gr > 0:
        parts["gr"] = cfg.w_gr * gr_penalty(logits, batch["gr_gap"], valid)
    if cfg.head_dz and "dz" in out:
        parts["dz"] = cfg.w_dz * dz_layer_loss(
            out["dz"], y, batch["dip"], valid, cfg.md_stride,
            integ_weight=cfg.w_dz_integ, integ_delta=cfg.huber_delta)
    if cfg.head_move and "move_logits" in out:
        parts["move"] = cfg.w_move * move_ce(
            out["move_logits"], y, row_tvt, valid,
            move_step=cfg.move_step, row_stride=cfg.move_row_stride)
    if cfg.head_sigma and "log_sigma" in out:
        parts["nll"] = cfg.w_nll * gaussian_nll(pred, y, out["log_sigma"], valid)

    total = sum(parts.values())
    return total, {k: float(v.detach()) for k, v in parts.items()}
