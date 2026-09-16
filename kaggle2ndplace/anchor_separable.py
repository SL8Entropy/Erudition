#!/usr/bin/env python3
"""A separable-encoder AnchorCNN: 1-D encoders plus an explicit matching volume.

The released model feeds a 9-channel image to EfficientNet-B0, but that image is built
from about 3.7k one-dimensional numbers -- a typewell profile along level, a lateral log
along MD -- expanded into ~0.8M pixels by broadcasts and outer differences.  The CNN then
spends most of its compute re-deriving structure that was manufactured from 1-D vectors.

This model keeps the 1-D inputs 1-D for as long as possible:

    level branch   1-D residual CNN over the typewell profile        (B, C, T)
    MD branch      1-D residual CNN over the lateral log and trajectory (B, C, M)
    volume         their outer difference and product, plus the raw mismatch and
                   known-path channels, at level T/2 and MD M          (B, 2C+7, T/2, M)
    2-D pyramid    depthwise-separable residual stages whose feature maps are exactly
                   the released trunk's: [T/2, T/4, T/8, T/16] x [M/2, M/4, M/8, M/16]
    decoder/heads  identical to ``GR2TVTAnchorNet``: FPN sum, fuse blocks, move head,
                   sub-bin head, dz_layer head with FiLM

Only the encoder changes.  The decoder, heads, loss and DP decode are the author's
design, so a difference against the baseline is attributable to the encoder.  The
feature pyramid sits at the same resolutions as the baseline trunk's, which keeps the
2:1 ratio between finest features and output grid that the resolution ablation showed
is required (see TRAINING.md).

One confound is unavoidable and should be stated with any result: the baseline trunk
starts from ImageNet weights, and this encoder has none to start from.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from gr2tvt_model import dp_expected_level, known_path_map  # noqa: E402


class Res1d(nn.Module):
    def __init__(self, c: int, k: int = 5, dilation: int = 1):
        super().__init__()
        pad = dilation * (k // 2)
        self.conv1 = nn.Conv1d(c, c, k, padding=pad, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(c)
        self.conv2 = nn.Conv1d(c, c, k, padding=pad, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(c)

    def forward(self, x):
        return F.gelu(x + self.bn2(self.conv2(F.gelu(self.bn1(self.conv1(x))))))


class DSBlock(nn.Module):
    """Depthwise 3x3 then a 2x pointwise expansion, residual (ConvNeXt-lite)."""

    def __init__(self, c: int, expand: int = 2):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.pw1 = nn.Conv2d(c, c * expand, 1)
        self.pw2 = nn.Conv2d(c * expand, c, 1)

    def forward(self, x):
        return x + self.pw2(F.gelu(self.pw1(self.bn(self.dw(x)))))


def _down(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 2, stride=2, bias=False),
                         nn.BatchNorm2d(cout), nn.GELU())


class SeparableAnchorNet(nn.Module):
    """Drop-in replacement for ``GR2TVTAnchorNet`` that consumes the 1-D items directly."""

    takes_items = True

    def __init__(self, T: int, M: int, ps_col: int = 16, win: float = 128.0,
                 n_move: int = 5, c1d: int = 64, stages=(32, 64, 128, 192),
                 blocks=(2, 2, 2, 2), d: int = 64, n_blocks: int = 2, fuse_div: int = 4,
                 ps_sigma: float = 4.0, dzl_head: bool = True):
        super().__init__()
        if T % 16 or M % 16:
            raise ValueError(f"T={T} and M={M} must both be divisible by 16")
        self.T, self.M, self.ps_col, self.win = int(T), int(M), int(ps_col), float(win)
        self.n_move, self.fuse_div, self.ps_sigma = int(n_move), int(fuse_div), float(ps_sigma)
        self.md_stem = None                       # read by items_to_x; unused here
        self.Tv = self.T // 2
        rowv = 2 * self.win / self.Tv
        self.register_buffer("levels_v", torch.arange(-self.win + rowv / 2, self.win, rowv))

        # 1-D encoders
        self.t_stem = nn.Sequential(nn.Conv1d(4, c1d, 5, padding=2, bias=False),
                                    nn.BatchNorm1d(c1d), nn.GELU())
        self.t_body = nn.Sequential(*[Res1d(c1d, 5, dl) for dl in (1, 2, 4, 8, 1, 2, 4, 8)])
        self.h_stem = nn.Sequential(nn.Conv1d(6, c1d, 5, padding=2, bias=False),
                                    nn.BatchNorm1d(c1d), nn.GELU())
        self.h_body = nn.Sequential(*[Res1d(c1d, 5, dl) for dl in (1, 2, 4, 8, 16, 32)])

        # volume -> first pyramid stage at (T/2, M/2), matching the baseline trunk's first map
        vol_ch = 2 * c1d + 7
        self.vol_stem = nn.Sequential(nn.Conv2d(vol_ch, stages[0], (1, 2), stride=(1, 2), bias=False),
                                      nn.BatchNorm2d(stages[0]), nn.GELU())
        self.stages = nn.ModuleList()
        cin = stages[0]
        for i, (c, nb) in enumerate(zip(stages, blocks)):
            layers = [] if i == 0 else [_down(cin, c)]
            layers += [DSBlock(c) for _ in range(nb)]
            self.stages.append(nn.Sequential(*layers))
            cin = c

        # decoder and heads: the GR2TVTAnchorNet design, unchanged
        self.proj = nn.ModuleList(nn.Conv2d(c, d, 1, bias=False) for c in stages)
        fuse = []
        for _ in range(n_blocks):
            fuse += [nn.Conv2d(d, d, 3, padding=1, bias=False), nn.BatchNorm2d(d), nn.GELU()]
        self.blocks = nn.Sequential(*fuse)
        self.up_conv = nn.Sequential(nn.Conv2d(d, d, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(d), nn.GELU())
        self.cls_head = nn.Conv2d(d, 2 * n_move + 1, 1)
        self.b_head = nn.Conv2d(d, 1, 1)
        self.aux_head = nn.Conv2d(d, 1, 1)
        self.dzl_head = bool(dzl_head)
        if self.dzl_head:
            self.dzl_proj = nn.Conv1d(d, 1, 1)
            self.dzl_film = nn.Conv1d(1, 2 * d, 1)
            for m in (self.dzl_proj, self.dzl_film):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        for m in (self.cls_head, self.b_head, self.aux_head):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ inputs
    def inputs_from_items(self, items, device) -> dict:
        """Stack the 1-D arrays out of ``anchor_data.build_item`` tuples."""
        def st(k):
            return torch.tensor(np.stack([it[k] for it in items]), device=device)
        return dict(t_n=st(0), h_n=st(1), h_valid=st(2), t_cover=st(3),
                    ncol=torch.tensor([it[5] for it in items], device=device),
                    k_lv=st(9), k_flag=st(10), s_n=st(11), s_cover=st(12), d_n=st(13))

    # ----------------------------------------------------------------- forward
    def forward(self, x: dict, aux: bool = False, dzl: bool = False):
        t_n, t_cover, s_n, s_cover = x["t_n"], x["t_cover"], x["s_n"], x["s_cover"]
        h_n, h_valid, d_n = x["h_n"], x["h_valid"], x["d_n"]
        k_lv, k_flag, ncol = x["k_lv"], x["k_flag"], x["ncol"]
        B = t_n.shape[0]
        act = (torch.arange(self.M, device=h_n.device)[None, :] < ncol[:, None]).to(h_n.dtype)

        # level branch at full T, then pooled to T/2
        ft = self.t_body(self.t_stem(torch.stack([t_n, t_cover, s_n * s_cover, s_cover], 1)))
        ft = F.avg_pool1d(ft, 2)                                           # (B, C, Tv)
        # MD branch at full M
        fh = self.h_body(self.h_stem(torch.stack(
            [h_n * act, h_valid * act, d_n * act, k_flag * act, act, (k_lv / 32.0) * k_flag * act], 1)))
        fh = fh * act[:, None, :]                                          # (B, C, M)

        pool = lambda v: F.avg_pool1d(v[:, None, :], 2)[:, 0]              # noqa: E731
        t_v, s_v, sc_v, tc_v = pool(t_n), pool(s_n), pool(s_cover), pool(t_cover)
        A = act[:, None, :]
        raw = torch.stack([
            (h_n[:, None, :] - t_v[:, :, None]) * A,
            (h_n[:, None, :] - s_v[:, :, None]) * A * sc_v[:, :, None],
            sc_v[:, :, None].expand(B, self.Tv, self.M),
            tc_v[:, :, None].expand(B, self.Tv, self.M),
            known_path_map(k_lv, k_flag, self.levels_v, self.ps_sigma) * A,
            (k_flag * act)[:, None, :].expand(B, self.Tv, self.M),
            A.expand(B, self.Tv, self.M),
        ], dim=1)
        vol = torch.cat([fh[:, :, None, :] - ft[:, :, :, None],
                         fh[:, :, None, :] * ft[:, :, :, None], raw], dim=1)

        feats, h = [], self.vol_stem(vol)
        for stage in self.stages:
            h = stage(h)
            feats.append(h)

        tq, am = self.T // 4, self.M
        out = None
        for f, proj in zip(feats, self.proj):
            f = F.interpolate(proj(f), size=(tq, am // self.fuse_div), mode="bilinear",
                              align_corners=False)
            out = f if out is None else out + f
        h = self.blocks(out)
        h = self.up_conv(F.interpolate(h, size=(tq, am), mode="bilinear", align_corners=False))

        d_pred = None
        if self.dzl_head:
            d_pred = self.dzl_proj(h.mean(dim=2))[:, 0]
            gamma, beta = self.dzl_film(d_pred[:, None, :]).chunk(2, dim=1)
            h = h * (1.0 + gamma[:, :, None, :]) + beta[:, :, None, :]
        res = (self.cls_head(h), torch.sigmoid(self.b_head(h))[:, 0])
        if aux:
            res = res + (F.interpolate(self.aux_head(h), size=(self.T, am), mode="bilinear",
                                       align_corners=False).squeeze(1),)
        if dzl:
            if d_pred is None:
                raise ValueError("dzl=True requires dzl_head=True")
            res = res + (d_pred,)
        return res

    @torch.no_grad()
    def predict_level(self, x: dict) -> torch.Tensor:
        cls_logits, b = self.forward(x)
        return dp_expected_level(cls_logits.float(), b.float(), self.win, start_col=self.ps_col)
