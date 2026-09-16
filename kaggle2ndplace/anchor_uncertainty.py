#!/usr/bin/env python3
"""Mean *and spread* of the AnchorCNN's decoded path.

``gr2tvt_model.dp_expected_level`` propagates a full belief over level from column to
column but returns only its expectation.  ``dp_moments`` is that same recursion, line for
line, additionally accumulating the second moment, so the spread of the belief comes out
at no extra model cost.  Its first output is verified equal to ``dp_expected_level``.

Why it matters: the pooled metric is dominated by a few catastrophic wells, and the
likely mechanism is a belief that splits between look-alike layers.  A wide or two-peaked
belief should therefore flag those wells -- which makes the spread a natural input to a
gate that decides when to trust this model and when to trust another.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def dp_moments(cls_logits: torch.Tensor, b: torch.Tensor, win: float = 128.0,
               start_col: int = 0, start_level: float = 0.0):
    """Per-column E[level] and E[level^2] under the DP belief.

    Identical recursion to ``dp_expected_level`` (no mask branch, scalar start level,
    which is how ``predict_level`` calls it).  Returns two (B, M) tensors at column
    centres; the variance is ``E2 - E1**2``.
    """
    B, V, Tq, M = cls_logits.shape
    n = (V - 1) // 2
    h_ft = 2 * win / Tq
    dev = cls_logits.device
    probs = torch.softmax(cls_logits, dim=1)
    idx = (torch.arange(Tq, device=dev).view(1, Tq) + 2 * n
           - torch.arange(V, device=dev).view(V, 1))
    idx_b = idx.unsqueeze(0).expand(B, V, Tq)
    alpha = cls_logits.new_zeros(B, Tq)
    alpha[:, int((start_level + win) / h_ft)] = 1.0
    rows = torch.arange(Tq, device=dev, dtype=cls_logits.dtype)
    e1 = [cls_logits.new_zeros(B) + float(start_level)]
    e2 = [cls_logits.new_zeros(B) + float(start_level) ** 2]
    for _ in range(start_col):
        e1.append(e1[-1])
        e2.append(e2[-1])
    for k in range(start_col, M):
        p_pad = F.pad(probs[:, :, :, k], (n, n))
        a_pad = F.pad(alpha, (n, n)).unsqueeze(1).expand(B, V, Tq + 2 * n)
        new = (a_pad.gather(2, idx_b) * p_pad.gather(2, idx_b)).sum(1)
        alpha = new / new.sum(1, keepdim=True).clamp_min(1e-30)
        bk = b[:, :, min(k + 1, M - 1)]
        pos = (rows + bk) * h_ft - win
        e1.append((alpha * pos).sum(1))
        e2.append((alpha * pos * pos).sum(1))
    E1 = torch.stack(e1, dim=1)
    E2 = torch.stack(e2, dim=1)
    return 0.5 * (E1[:, :-1] + E1[:, 1:]), 0.5 * (E2[:, :-1] + E2[:, 1:])
