"""Per-row SoftMax gating over candidate paths (3rd place).

Fixed ensemble weights treat the first 100 ft after PS, a 9,000 ft
extrapolation, a missing-GR stretch and a region where the candidates violently
disagree all the same way. A gate lets the weights vary along the well using
that context.

The weights are non-negative and sum to one, so the output is a convex
combination of paths that are each already physically plausible. With 773 wells
that constraint is doing real work: 3rd place measured an unconstrained
refiner at 5.40 CV against 5.28 for the gated mixture, losing in all five
folds. The cost is a real ceiling -- a convex mixture cannot fix drift shared
by every candidate.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftMaxGate(nn.Module):
    def __init__(self, n_feat: int, n_cand: int, hidden: int = 32,
                 kernel: int = 7, blocks: int = 2):
        super().__init__()
        self.inp = nn.Conv1d(n_feat, hidden, kernel, padding=kernel // 2)
        self.blocks = nn.ModuleList()
        for _ in range(blocks):
            self.blocks.append(nn.Sequential(
                nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2),
                nn.BatchNorm1d(hidden), nn.GELU(),
                nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2),
                nn.BatchNorm1d(hidden)))
        self.rnn = nn.LSTM(hidden, hidden // 2, batch_first=True,
                           bidirectional=True)
        self.out = nn.Conv1d(hidden, n_cand, 1)

    def forward(self, x):                     # x: (B, F, T)
        h = self.inp(x)
        for b in self.blocks:
            h = F.gelu(h + b(h))
        h, _ = self.rnn(h.transpose(1, 2))
        return self.out(h.transpose(1, 2))    # logits (B, K, T)


# ---------------------------------------------------------------------- #
def gate_features(well, cands: np.ndarray, sigma=None, step: int = 8):
    """Build (F, T) gate inputs on a subsampled row grid.

    `cands` is (K, N) absolute TVT. Everything the gate sees is either
    candidate disagreement, distance from the anchor, or local data quality --
    never the absolute depth, which does not generalise across wells.
    """
    m = np.flatnonzero(well.eval_mask)[::step]
    if len(m) < 4:
        return None, None, None
    c = cands[:, m]                                   # (K, T)
    a = well.anchor_tvt
    off = c - a
    mean = off.mean(axis=0, keepdims=True)
    d1 = np.diff(off, axis=1, prepend=off[:, :1])

    md = well.md[m]
    z = well.z[m]
    gr = well.gr[m]
    pos = (md - well.anchor_md) / max(well.md[-1] - well.anchor_md, 1.0)

    feats = [off / 20.0, (off - mean) / 10.0, d1 * 4.0]
    feats.append(np.repeat(off.std(axis=0, keepdims=True) / 10.0, 1, axis=0))
    feats.append(np.repeat((off.max(axis=0) - off.min(axis=0))[None] / 20.0, 1, 0))
    feats.append(((md - well.anchor_md) / 5000.0)[None])
    feats.append(((z - well.anchor_z) / 100.0)[None])
    feats.append((np.gradient(z, md) * 10.0)[None])
    feats.append((np.nan_to_num((gr - 80.0) / 40.0))[None])
    feats.append((~np.isfinite(gr)).astype(float)[None])
    feats.append(pos[None])
    if sigma is not None:
        feats.append((np.asarray(sigma)[m] / 10.0)[None])
    else:
        feats.append(np.zeros((1, len(m))))
    X = np.concatenate(feats, axis=0).astype(np.float32)
    y = well.tvt[m] - a if well.tvt is not None else None
    return X, off.astype(np.float32), (None if y is None else y.astype(np.float32))


def fit_gate(wells, cand_by_well: dict, sigma_by_well=None, epochs: int = 8,
             lr: float = 3e-3, device: str = "cuda", hidden: int = 32,
             seq_len: int = 256, verbose: bool = True):
    """Train the gate on OOF candidates. `cand_by_well[wid] = (K, N) array`."""
    items = []
    for w in wells:
        c = cand_by_well.get(w.well_id)
        if c is None or w.tvt is None:
            continue
        sg = (sigma_by_well or {}).get(w.well_id)
        X, off, y = gate_features(w, c, sg)
        if X is None:
            continue
        items.append((X, off, y))
    if not items:
        raise RuntimeError("no gate training data")

    n_feat, n_cand = items[0][0].shape[0], items[0][1].shape[0]
    model = SoftMaxGate(n_feat, n_cand, hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(0)

    for ep in range(epochs):
        rng.shuffle(items)
        tot, nb = 0.0, 0
        model.train()
        for X, off, y in items:
            T = X.shape[1]
            s = 0 if T <= seq_len else int(rng.integers(0, T - seq_len))
            sl = slice(s, s + min(seq_len, T))
            xb = torch.from_numpy(X[:, sl]).unsqueeze(0).to(device)
            cb = torch.from_numpy(off[:, sl]).unsqueeze(0).to(device)
            yb = torch.from_numpy(y[sl]).unsqueeze(0).to(device)

            logits = model(xb)
            w = torch.softmax(logits, dim=1)
            pred = (w * cb).sum(dim=1)
            best = (cb - yb.unsqueeze(1)).abs().argmin(dim=1)
            loss = (F.mse_loss(pred, yb)
                    + F.cross_entropy(logits.transpose(1, 2).reshape(-1, n_cand),
                                      best.reshape(-1)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)
            nb += 1
        if verbose:
            print(f"  [gate] ep{ep + 1}/{epochs} loss={tot / max(nb, 1):.4f}")
    return model


@torch.no_grad()
def apply_gate(model, well, cands: np.ndarray, sigma=None, step: int = 8,
               device: str = "cuda"):
    """Blend candidates for one well; the weights are interpolated back to 1 ft."""
    X, off, _ = gate_features(well, cands, sigma, step=step)
    if X is None:
        return cands.mean(axis=0)
    model.eval()
    logits = model(torch.from_numpy(X).unsqueeze(0).to(device))
    w = torch.softmax(logits, dim=1)[0].cpu().numpy()     # (K, T)

    m = np.flatnonzero(well.eval_mask)[::step]
    md_s = well.md[m]
    W = np.stack([np.interp(well.md, md_s, w[k]) for k in range(w.shape[0])])
    W /= np.maximum(W.sum(axis=0, keepdims=True), 1e-9)
    pred = (W * cands).sum(axis=0)
    vis = np.isfinite(well.tvt_input)
    pred[vis] = well.tvt_input[vis]
    return pred
