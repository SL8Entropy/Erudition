#!/usr/bin/env python3
"""Network definition for the GR -> TVT inverse model (AnchorCNN).

An EfficientNet-B0 trunk consumes the 2D (level x MD) input grid. At each scale the TVT
axis is folded into the channel dimension by concatenation rather than pooled, so that
vertical position is preserved. The scales are merged with a light FPN-style addition
along MD, a 1D CNN runs along MD, and a per-column head regresses the level.

The head used by the winning submission predicts, for each anchor column, a distribution
over discrete TVT moves. Decoding turns those distributions into a path. An optional
auxiliary head regresses layer thickness (dz_layer); it shapes the representation during
training and is discarded at inference.
"""

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

LV_SCALE = 16.0
COLW = 32.0


def _apply_resolution_mods(trunk, stem_stride: int = 2, md_keep: int = 0,
                           tvt_keep: int = 0) -> None:
    if stem_stride == 1:
        trunk.conv_stem.stride = (1, 1)
    if md_keep > 0 or tvt_keep > 0:
        downs = [m for m in trunk.modules()
                 if isinstance(m, nn.Conv2d) and m is not trunk.conv_stem
                 and m.stride == (2, 2)]
        for i, m in enumerate(downs):
            m.stride = (1 if i < tvt_keep else 2, 1 if i < md_keep else 2)


def make_trunk(backbone: str, in_chans: int, pretrained: bool,
               stem_stride: int = 2, md_keep: int = 0, tvt_keep: int = 0,
               drop_path: float = 0.0):
    is_effnet = backbone.startswith("efficientnet")
    if not is_effnet and (stem_stride != 2 or md_keep or tvt_keep):
        raise ValueError(f"stem_stride/md_keep/tvt_keep are EfficientNet-specific and "
                         f"unsupported for backbone={backbone} (see make_trunk)")
    kw = dict(features_only=True, in_chans=in_chans, pretrained=pretrained,
              out_indices=(1, 2, 3, 4) if is_effnet else (0, 1, 2, 3))
    if drop_path > 0:
        kw["drop_path_rate"] = drop_path
    trunk = timm.create_model(backbone, **kw)
    reds = list(trunk.feature_info.reduction())
    if reds != [4, 8, 16, 32]:
        raise ValueError(f"backbone={backbone} has reductions {reds}, which do not match the "
                         f"decoder assumption [4, 8, 16, 32]")
    if is_effnet:
        _apply_resolution_mods(trunk, stem_stride, md_keep, tvt_keep)
    return trunk


def known_path_map(k_lv: torch.Tensor, k_flag: torch.Tensor, levels: torch.Tensor,
                   sigma: float = 4.0) -> torch.Tensor:
    d = (levels.view(1, -1, 1) - k_lv[:, None, :]) / sigma
    return torch.exp(-0.5 * d * d) * k_flag[:, None, :]


def assemble_x(t_n: torch.Tensor, h_n: torch.Tensor, h_valid: torch.Tensor,
               t_cover: torch.Tensor, ncol: torch.Tensor,
               k_map: torch.Tensor | None = None,
               k_flag: torch.Tensor | None = None,
               s_n: torch.Tensor | None = None,
               s_cover: torch.Tensor | None = None,
               d_n: torch.Tensor | None = None,
               zr_n: torch.Tensor | None = None, zr_in: bool = True,
               pos_enc: bool = False, ps_col: int = 0,
               zr_head: bool = False, dip_head: bool = False) -> torch.Tensor:
    B, T_ = t_n.shape
    M = h_n.shape[1]
    act = (torch.arange(M, device=h_n.device)[None, :] < ncol[:, None]).to(h_n.dtype)
    x0 = (h_n[:, None, :] - t_n[:, :, None]) * act[:, None, :]
    x1 = (h_n * act)[:, None, :].expand(B, T_, M)
    x2 = (h_valid * act)[:, None, :].expand(B, T_, M)
    x3 = t_cover[:, :, None].expand(B, T_, M)
    chans = [x0, x1, x2, x3]
    if k_map is not None:
        chans.append(k_map * act[:, None, :])
        if k_flag is not None:
            chans.append((k_flag * act)[:, None, :].expand(B, T_, M))
    if s_n is not None:
        chans.append((h_n[:, None, :] - s_n[:, :, None]) * act[:, None, :] * s_cover[:, :, None])
        chans.append(s_cover[:, :, None].expand(B, T_, M))
    if d_n is not None:
        chans.append((d_n * act)[:, None, :].expand(B, T_, M))
    if zr_n is not None and zr_in:
        chans.append((zr_n * act)[:, None, :].expand(B, T_, M))
    if pos_enc:
        dev, dt = h_n.device, h_n.dtype
        p_t = torch.linspace(-1.0 + 1.0 / T_, 1.0 - 1.0 / T_, T_, device=dev, dtype=dt)
        chans.append(p_t[None, :, None].expand(B, T_, M))
        p_m = (torch.arange(M, device=dev, dtype=dt) - float(ps_col)) * (COLW / 1000.0)
        chans.append((p_m[None, :] * act)[:, None, :].expand(B, T_, M))
    if zr_head:
        if zr_n is None:
            raise ValueError("zr_head=True requires zr_n")
        chans.append((zr_n * act)[:, None, :].expand(B, T_, M))
    if dip_head:
        if d_n is None:
            raise ValueError("dip_head=True requires d_n")
        chans.append((d_n * act)[:, None, :].expand(B, T_, M))
    return torch.stack(chans, dim=1)


class MDStem(nn.Module):

    def __init__(self, k: int = 1, stride: int = 1, valid_conv: bool = False):
        super().__init__()
        self.stride, self.k, self.valid_conv = stride, k, valid_conv
        self.conv = None
        if k > 1:
            if k < stride or (k - stride) % 2 != 0:
                raise ValueError(f"k must be >= stride and k-stride must be even"
                                 f"（k={k}, stride={stride}）")
            self.conv = nn.Conv1d(2, 2, k, stride=stride, padding=(k - stride) // 2,
                                  groups=2, bias=False)
            nn.init.zeros_(self.conv.weight)

    def forward(self, h_n, h_valid, h_cnt, h_rows):
        s = self.stride
        if s == 1 and self.conv is None:
            return h_n, h_valid
        B, L = h_n.shape
        if L % s:
            raise ValueError(f"input column count {L} is not divisible by stride {s}")
        g = h_cnt.view(B, -1, s).double()
        r = h_rows.view(B, -1, s).double()
        hn = h_n.view(B, -1, s).double()
        gs, rs = g.sum(-1), r.sum(-1)
        num = torch.where(gs > 0, (hn * g).sum(-1), (hn * r).sum(-1))
        den = torch.where(gs > 0, gs, rs).clamp_min(1e-6)
        base_h = (num / den).to(h_n.dtype)
        base_v = (gs / rs.clamp_min(1e-6)).to(h_n.dtype)
        if self.conv is None:
            return base_h, base_v
        d = self.conv(torch.stack([h_n, h_valid], 1))
        if d.shape[-1] != base_h.shape[-1]:
            raise RuntimeError(f"pre-filter output length {d.shape[-1]} != expected {base_h.shape[-1]}")
        out_v = base_v + d[:, 1] if self.valid_conv else base_v
        return base_h + d[:, 0], out_v


class Conv1dBlock(nn.Module):
    def __init__(self, d: int, k: int = 5):
        super().__init__()
        self.conv = nn.Conv1d(d, d, k, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm1d(d)

    def forward(self, x):
        return F.gelu(self.bn(self.conv(x))) + x


class GR2TVTNet(nn.Module):

    def __init__(self, backbone: str = "efficientnet_b0", in_chans: int = 4,
                 d: int = 256, n_blocks: int = 4, pretrained: bool = True,
                 t_size: int = 512, md_stride: int = 4,
                 stem_stride: int = 2, md_keep: int = 0, tvt_keep: int = 0):
        super().__init__()
        self.md_stride = md_stride
        self.trunk = timm.create_model(backbone, features_only=True, in_chans=in_chans,
                                       pretrained=pretrained, out_indices=(1, 2, 3, 4))
        _apply_resolution_mods(self.trunk, stem_stride, md_keep, tvt_keep)
        chs = self.trunk.feature_info.channels()
        reds = self.trunk.feature_info.reduction()
        self.proj = nn.ModuleList(
            nn.Conv1d(c * (t_size // r), d, 1, bias=False) for c, r in zip(chs, reds))
        self.blocks = nn.Sequential(*[Conv1dBlock(d) for _ in range(n_blocks)])
        self.head = nn.Conv1d(d, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, T, M = x.shape
        m4 = M // self.md_stride
        feats = self.trunk(x)
        h = None
        for f, proj in zip(feats, self.proj):
            b, c, t, m = f.shape
            f = proj(f.reshape(b, c * t, m))
            f = F.interpolate(f, size=m4, mode="linear", align_corners=False)
            h = f if h is None else h + f
        h = self.blocks(h)
        y = self.head(h)
        y = F.interpolate(y, size=M, mode="linear", align_corners=False)
        return y.squeeze(1) * LV_SCALE

    def predict_level(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class GR2TVTFieldNet(nn.Module):

    def __init__(self, backbone: str = "efficientnet_b0", in_chans: int = 4,
                 d: int = 64, n_blocks: int = 2, pretrained: bool = True,
                 win: float = 128.0, row: float = 0.5,
                 stem_stride: int = 2, md_keep: int = 0, tvt_keep: int = 0,
                 md_k: int = 1, md_stride: int = 1, md_valid_conv: bool = False,
                 anchor_m: int = 0, fuse_div: int = 4, ps_col: int = 0, ps_sigma: float = 4.0,
                 ps_mask: bool = True):
        super().__init__()
        self.ps_col = ps_col
        self.ps_sigma = ps_sigma
        self.ps_mask = ps_mask
        self.md_stem = (MDStem(md_k, md_stride, md_valid_conv)
                        if (md_k > 1 or md_stride > 1) else None)
        self.anchor_m = anchor_m
        self.fuse_div = fuse_div
        self.trunk = timm.create_model(backbone, features_only=True, in_chans=in_chans,
                                       pretrained=pretrained, out_indices=(1, 2, 3, 4))
        _apply_resolution_mods(self.trunk, stem_stride, md_keep, tvt_keep)
        chs = self.trunk.feature_info.channels()
        self.proj = nn.ModuleList(nn.Conv2d(c, d, 1, bias=False) for c in chs)
        blocks = []
        for _ in range(n_blocks):
            blocks += [nn.Conv2d(d, d, 3, padding=1, bias=False), nn.BatchNorm2d(d), nn.GELU()]
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv2d(d, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        levels = torch.arange(-win + row / 2, win, row, dtype=torch.float32)
        self.register_buffer("levels", levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, T, M = x.shape
        size4 = (T // 4, M // 4)
        feats = self.trunk(x)
        h = None
        for f, proj in zip(feats, self.proj):
            f = proj(f)
            f = F.interpolate(f, size=size4, mode="bilinear", align_corners=False)
            h = f if h is None else h + f
        h = self.blocks(h)
        z = self.head(h)
        z = F.interpolate(z, size=(T, M), mode="bilinear", align_corners=False)
        return z.squeeze(1)

    def predict_level(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(self.forward(x), dim=1)
        return (p * self.levels.view(1, -1, 1)).sum(1)


class _UpBlock(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin + cskip, cout, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return F.gelu(self.bn(self.conv(torch.cat([x, skip], dim=1))))


class GR2TVTUNet(nn.Module):

    def __init__(self, backbone: str = "efficientnet_b0", in_chans: int = 4,
                 pretrained: bool = True, win: float = 128.0, row: float = 0.5,
                 stem_stride: int = 2, md_keep: int = 0, tvt_keep: int = 0):
        super().__init__()
        self.trunk = timm.create_model(backbone, features_only=True, in_chans=in_chans,
                                       pretrained=pretrained, out_indices=(0, 1, 2, 3, 4))
        _apply_resolution_mods(self.trunk, stem_stride, md_keep, tvt_keep)
        chs = self.trunk.feature_info.channels()
        dec = [128, 96, 64, 48, 32]
        self.bottom = nn.Sequential(nn.Conv2d(chs[4], 128, 1, bias=False),
                                    nn.BatchNorm2d(128), nn.GELU())
        self.up1 = _UpBlock(128, chs[3], dec[0])
        self.up2 = _UpBlock(dec[0], chs[2], dec[1])
        self.up3 = _UpBlock(dec[1], chs[1], dec[2])
        self.up4 = _UpBlock(dec[2], chs[0], dec[3])
        self.up5 = _UpBlock(dec[3], in_chans, dec[4])
        self.head = nn.Conv2d(dec[4], 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        levels = torch.arange(-win + row / 2, win, row, dtype=torch.float32)
        self.register_buffer("levels", levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f2, f4, f8, f16, f32 = self.trunk(x)
        h = self.bottom(f32)
        h = self.up1(h, f16)
        h = self.up2(h, f8)
        h = self.up3(h, f4)
        h = self.up4(h, f2)
        h = self.up5(h, x)
        return self.head(h).squeeze(1)

    def predict_level(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(self.forward(x), dim=1)
        return (p * self.levels.view(1, -1, 1)).sum(1)


class GR2TVTFieldMDUp(nn.Module):

    def __init__(self, backbone: str = "efficientnet_b0", in_chans: int = 4,
                 d: int = 128, pretrained: bool = True, win: float = 128.0, row: float = 0.5,
                 stem_stride: int = 2, md_keep: int = 0, tvt_keep: int = 0):
        super().__init__()
        self.trunk = timm.create_model(backbone, features_only=True, in_chans=in_chans,
                                       pretrained=pretrained, out_indices=(1, 2, 3, 4))
        _apply_resolution_mods(self.trunk, stem_stride, md_keep, tvt_keep)
        chs = self.trunk.feature_info.channels()
        self.proj = nn.ModuleList(nn.Conv2d(c, d, 1, bias=False) for c in chs)
        self.pre = nn.Sequential(nn.Conv2d(d, d, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(d), nn.GELU())
        def md_conv(cin, cout, k=5):
            return nn.Sequential(nn.Conv2d(cin, cout, (1, k), padding=(0, k // 2), bias=False),
                                 nn.BatchNorm2d(cout), nn.GELU())
        self.up1 = md_conv(d, d)
        self.up2 = md_conv(d, d)
        self.skip = nn.Conv2d(in_chans, d // 4, 1, bias=False)
        self.mix = md_conv(d + d // 4, d, k=3)
        self.head = nn.Conv2d(d, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        levels = torch.arange(-win + row / 2, win, row, dtype=torch.float32)
        self.register_buffer("levels", levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, T, M = x.shape
        t4, m4 = T // 4, M // 4
        feats = self.trunk(x)
        h = None
        for f, proj in zip(feats, self.proj):
            f = proj(f)
            f = F.interpolate(f, size=(t4, m4), mode="bilinear", align_corners=False)
            h = f if h is None else h + f
        h = self.pre(h)
        h = self.up1(F.interpolate(h, size=(t4, M // 2), mode="bilinear", align_corners=False))
        h = self.up2(F.interpolate(h, size=(t4, M), mode="bilinear", align_corners=False))
        skip = self.skip(F.avg_pool2d(x, kernel_size=(T // t4, 1)))
        h = self.mix(torch.cat([h, skip], dim=1))
        z = self.head(h)
        z = F.interpolate(z, size=(T, M), mode="bilinear", align_corners=False)
        return z.squeeze(1)

    def predict_level(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(self.forward(x), dim=1)
        return (p * self.levels.view(1, -1, 1)).sum(1)


def field_ce_loss(logits: torch.Tensor, y: torch.Tensor, levels: torch.Tensor,
                  sigma: float = 3.0, kernel: str = "gauss") -> torch.Tensor:
    mask = torch.isfinite(y)
    y0 = torch.where(mask, y, torch.zeros_like(y))
    d = (levels.view(1, -1, 1) - y0.unsqueeze(1)) / sigma
    q = torch.softmax(-torch.abs(d) if kernel == "laplace" else -0.5 * d * d, dim=1)
    ce = -(q * torch.log_softmax(logits, dim=1)).sum(1)
    per_well = (ce * mask).sum(1) / mask.sum(1).clamp(min=1)
    return per_well.mean()


def masked_huber_laplacian(pred: torch.Tensor, y: torch.Tensor, lam: float = 0.0,
                           delta: float = 3.0) -> torch.Tensor:
    mask = torch.isfinite(y)
    y0 = torch.where(mask, y, torch.zeros_like(y))
    per_col = F.huber_loss(pred, y0, delta=delta, reduction="none") * mask
    per_well = per_col.sum(1) / mask.sum(1).clamp(min=1)
    loss = per_well.mean()
    if lam > 0:
        m3 = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
        curv = (pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]) * m3
        loss = loss + lam * (curv ** 2).sum(1).div(m3.sum(1).clamp(min=1)).mean()
    return loss


ANCHOR_N = 10
ANCHOR_V = 2 * ANCHOR_N + 1


def dzl_target(y_bnd: torch.Tensor, d_n: torch.Tensor, colw_in: float,
               dz_slope_sd: float) -> tuple[torch.Tensor, torch.Tensor]:
    dtvt = y_bnd[:, 1:] - y_bnd[:, :-1]
    dz = dzl_dz_bnd(d_n, colw_in, dz_slope_sd)
    valid = torch.isfinite(dtvt)
    return torch.where(valid, dtvt, torch.zeros_like(dtvt)) + dz, valid


def dzl_dz_bnd(d_n: torch.Tensor, colw_in: float, dz_slope_sd: float) -> torch.Tensor:
    dz_b = 0.5 * (d_n[:, :-1] + d_n[:, 1:]) * dz_slope_sd * colw_in
    return F.pad(dz_b, (0, 1), mode="replicate")


def dip_penalty(d_pred: torch.Tensor, d_n: torch.Tensor, colw_in: float,
                dz_slope_sd: float, win: float, tq: int, n_cls: int) -> torch.Tensor:
    h_ft = 2.0 * win / tq
    dstar = (d_pred - dzl_dz_bnd(d_n, colw_in, dz_slope_sd)) / h_ft
    n = (n_cls - 1) // 2
    cv = torch.arange(-n, n + 1, device=d_pred.device, dtype=d_pred.dtype)
    return (cv.view(1, -1, 1) - dstar[:, None, :]).abs()


def dzl_loss(d_pred: torch.Tensor, tgt: torch.Tensor, valid: torch.Tensor,
             tv_w: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    if d_pred.shape != tgt.shape:
        raise ValueError(f"dzl: column grid mismatch, D {tuple(d_pred.shape)} vs target "
                         f"{tuple(tgt.shape)} (check --colw-in / --md-stride / anchor_m)")
    n = valid.sum().clamp(min=1)
    l1 = ((d_pred - tgt).abs() * valid).sum() / n
    if tv_w <= 0:
        return l1, l1.new_zeros(())
    vv = valid[:, 1:] & valid[:, :-1]
    tv = ((d_pred[:, 1:] - d_pred[:, :-1]).abs() * vv).sum() / vv.sum().clamp(min=1)
    return l1, tv


class HistoryHead(nn.Module):

    D_STD = 0.712

    def __init__(self, d: int, hidden: int = 64, n_cls: int = ANCHOR_V,
                 d_std: float = D_STD, d_clip_sigma: float = 3.0, delta_dim: int = 0):
        super().__init__()
        self.hidden, self.d_std, self.k = hidden, d_std, d_clip_sigma
        self.d_emb = (nn.Sequential(nn.Linear(1, delta_dim), nn.GELU())
                      if delta_dim > 0 else None)
        self.gru = nn.GRU(d + (delta_dim or 1), hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_cls)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def h0(self, n: int, device, dtype) -> torch.Tensor:
        return torch.zeros(1, n, self.hidden, device=device, dtype=dtype)

    def encode_delta(self, d_prev: torch.Tensor) -> torch.Tensor:
        u = (self.k * torch.tanh(d_prev / (self.k * self.d_std))).unsqueeze(-1)
        return u if self.d_emb is None else self.d_emb(u)

    def forward(self, f_seq: torch.Tensor, d_prev: torch.Tensor,
                h: torch.Tensor | None = None):
        y, h = self.gru(torch.cat([f_seq, self.encode_delta(d_prev)], dim=-1), h)
        return self.out(y), h


def path_bins(y_bnd: torch.Tensor, win: float, tq: int) -> tuple:
    h_ft = 2 * win / tq
    val = torch.isfinite(y_bnd)
    q = (torch.where(val, y_bnd, torch.zeros_like(y_bnd)) + win) / h_ft
    bi = torch.floor(q).long().clamp(0, tq - 1)
    return bi, (q - bi.float()).clamp(0.0, 1.0), val, q


def warmup_bins(k_lv: torch.Tensor, k_flag: torch.Tensor, ncol: int,
                win: float, tq: int) -> torch.Tensor:
    B, L = k_lv.shape
    r = max(1, L // ncol)
    lv = k_lv[:, :ncol * r].view(B, ncol, r)
    fl = k_flag[:, :ncol * r].view(B, ncol, r)
    n_ok = fl.sum(-1)
    lv = torch.where(n_ok > 0, (lv * fl).sum(-1) / n_ok.clamp_min(1), torch.zeros_like(n_ok))
    h_ft = 2 * win / tq
    bi = torch.floor((lv + win) / h_ft).long().clamp(0, tq - 1)
    ok = (n_ok > 0).long()
    idx = torch.cummax(ok * torch.arange(ncol, device=bi.device).view(1, ncol), dim=1)[0]
    return bi.gather(1, idx)


class GR2TVTAnchorNet(nn.Module):

    def __init__(self, backbone: str = "efficientnet_b0", in_chans: int = 4,
                 d: int = 64, n_blocks: int = 2, pretrained: bool = True,
                 win: float = 128.0, row: float = 0.5,
                 stem_stride: int = 2, md_keep: int = 0, tvt_keep: int = 0,
                 md_k: int = 1, md_stride: int = 1, md_valid_conv: bool = False,
                 anchor_m: int = 0, fuse_div: int = 4, ps_col: int = 0, ps_sigma: float = 4.0,
                 ps_mask: bool = True, hist_dim: int = 0, hist_delta_dim: int = 0,
                 drop_path: float = 0.0, input_up: int = 1, ps_gr: bool = False,
                 z_dip: bool = False, z_rel: bool = False, z_rel_head: bool = False,
                 z_dip_head: bool = False, pos_enc: bool = False,
                 dzl_head: bool = False, dzl_bias0: float = 0.0,
                 dzl_film: str = "full", dzl_arch: str = "linear",
                 n_move: int = ANCHOR_N):
        super().__init__()
        self.n_move = n_move
        self.ps_col = ps_col
        self.ps_sigma = ps_sigma
        self.ps_mask = ps_mask
        self.ps_gr = ps_gr
        self.z_dip = z_dip
        self.z_rel = z_rel
        self.z_rel_head = z_rel_head
        self.z_dip_head = z_dip_head
        self.n_head_ch = int(z_rel_head) + int(z_dip_head)
        self.pos_enc = pos_enc
        self.md_stem = (MDStem(md_k, md_stride, md_valid_conv)
                        if (md_k > 1 or md_stride > 1) else None)
        self.anchor_m = anchor_m
        self.fuse_div = fuse_div
        self.input_up = input_up
        self.trunk = make_trunk(backbone, in_chans, pretrained,
                                stem_stride, md_keep, tvt_keep, drop_path)
        chs = self.trunk.feature_info.channels()
        self.proj = nn.ModuleList(nn.Conv2d(c, d, 1, bias=False) for c in chs)
        blocks = []
        for _ in range(n_blocks):
            blocks += [nn.Conv2d(d, d, 3, padding=1, bias=False), nn.BatchNorm2d(d), nn.GELU()]
        self.blocks = nn.Sequential(*blocks)
        self.up_conv = nn.Sequential(nn.Conv2d(d, d, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(d), nn.GELU())
        self.cls_head = nn.Conv2d(d, 2 * n_move + 1, 1)
        self.b_head = nn.Conv2d(d, 1, 1)
        self.aux_head = nn.Conv2d(d, 1, 1)
        self.hist = (HistoryHead(d, hist_dim, n_cls=2 * n_move + 1,
                                 delta_dim=hist_delta_dim)
                     if hist_dim > 0 else None)
        self.z_head_proj = (nn.Conv2d(self.n_head_ch, d, 1, bias=False)
                            if self.n_head_ch > 0 else None)
        if self.z_head_proj is not None:
            nn.init.zeros_(self.z_head_proj.weight)
        self.dzl_head = dzl_head
        self.dzl_film_mode = dzl_film
        if dzl_head:
            if dzl_arch == "mdconv":
                self.dzl_proj = nn.Sequential(
                    nn.Conv1d(d, d, 5, padding=2, bias=False), nn.BatchNorm1d(d), nn.GELU(),
                    nn.Conv1d(d, d, 5, padding=2, bias=False), nn.BatchNorm1d(d), nn.GELU(),
                    nn.Conv1d(d, 1, 1))
            else:
                self.dzl_proj = nn.Conv1d(d, 1, 1)
            self.dzl_film = (None if dzl_film == "off" else
                             nn.Conv1d(1, 2 * d, 1, bias=(dzl_film != "cond")))
            last = (self.dzl_proj[-1] if isinstance(self.dzl_proj, nn.Sequential)
                    else self.dzl_proj)
            for m in (last, self.dzl_film):
                if m is None:
                    continue
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if dzl_film == "bias":
                self.dzl_film.weight.requires_grad_(False)
            nn.init.constant_(last.bias, float(dzl_bias0))
            self.dzl_bias0 = float(dzl_bias0)
        for m in (self.cls_head, self.b_head, self.aux_head):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.win, self.row = win, row
        levels = torch.arange(-win + row / 2, win, row, dtype=torch.float32)
        self.register_buffer("levels", levels)

    def assemble(self, t_n, h_n, h_valid, t_cover, ncol, h_cnt=None, h_rows=None,
                 k_lv=None, k_flag=None, s_n=None, s_cover=None, d_n=None, zr_n=None):
        if self.md_stem is not None:
            if h_cnt is None or h_rows is None:
                raise ValueError("the pre-filter requires h_cnt / h_rows (returned by build_compact)")
            h_n, h_valid = self.md_stem(h_n, h_valid, h_cnt, h_rows)
            if k_lv is not None:
                st = self.md_stem.stride
                k_lv = k_lv.view(k_lv.shape[0], -1, st).mean(-1)
                k_flag = k_flag.view(k_flag.shape[0], -1, st).mean(-1)
            if d_n is not None:
                st = self.md_stem.stride
                d_n = d_n.view(d_n.shape[0], -1, st).sum(-1)
            if zr_n is not None:
                st = self.md_stem.stride
                zr_n = zr_n.view(zr_n.shape[0], -1, st).mean(-1)
        k_map = (known_path_map(k_lv, k_flag, self.levels, self.ps_sigma)
                 if (k_lv is not None and self.ps_col > 0) else None)
        use_s = self.ps_gr and s_n is not None
        use_z = (self.z_rel or self.z_rel_head) and zr_n is not None
        use_dh = self.z_dip_head and d_n is not None
        return assemble_x(t_n, h_n, h_valid, t_cover, ncol, k_map,
                          k_flag if (k_map is not None and self.ps_mask) else None,
                          s_n if use_s else None, s_cover if use_s else None,
                          d_n if ((self.z_dip or use_dh) and d_n is not None) else None,
                          zr_n if use_z else None, zr_in=self.z_rel,
                          pos_enc=self.pos_enc, ps_col=self.ps_col,
                          zr_head=self.z_rel_head and use_z, dip_head=use_dh)

    def forward(self, x: torch.Tensor, aux: bool = False, feat: bool = False,
                dzl: bool = False):
        B, _, T, M = x.shape
        tq = T // 4
        am = self.anchor_m or M
        zr = None
        if self.n_head_ch > 0:
            n = self.n_head_ch
            zr = x[:, -n:, 0, :]
            x = x[:, :-n]
        if self.input_up > 1:
            x = F.interpolate(x, scale_factor=self.input_up, mode="bilinear",
                              align_corners=False)
        feats = self.trunk(x)
        h = None
        for f, proj in zip(feats, self.proj):
            f = proj(f)
            f = F.interpolate(f, size=(tq, am // self.fuse_div), mode="bilinear", align_corners=False)
            h = f if h is None else h + f
        h = self.blocks(h)
        h = F.interpolate(h, size=(tq, am), mode="bilinear", align_corners=False)
        h = self.up_conv(h)
        if zr is not None:
            zz = zr[:, :, None, :]
            if zz.shape[-1] != am:
                zz = F.interpolate(zz, size=(1, am), mode="bilinear", align_corners=False)
            h = h + self.z_head_proj(zz)
        d_pred = None
        if self.dzl_head:
            d_pred = self.dzl_proj(h.mean(dim=2))[:, 0]
            if self.dzl_film is not None:
                cin = (d_pred - self.dzl_bias0 if self.dzl_film_mode == "cond" else d_pred)
                gamma, beta = self.dzl_film(cin[:, None, :]).chunk(2, dim=1)
                h = h * (1.0 + gamma[:, :, None, :]) + beta[:, :, None, :]
        out = (self.cls_head(h), torch.sigmoid(self.b_head(h))[:, 0])
        if aux:
            z = F.interpolate(self.aux_head(h), size=(T, am), mode="bilinear", align_corners=False)
            out = out + (z.squeeze(1),)
        if dzl:
            if d_pred is None:
                raise ValueError("dzl=True requires a model built with dzl_head=True")
            out = out + (d_pred,)
        if feat:
            out = out + (h,)
        return out

    @torch.no_grad()
    def predict_level(self, x: torch.Tensor) -> torch.Tensor:
        cls_logits, b = self.forward(x)
        return dp_expected_level(cls_logits.float(), b.float(), self.win, start_col=self.ps_col)


@torch.no_grad()
def items_to_x(items, dev, model=None):
    t_n, h_n, h_valid, t_cover, y, ncol = (np.stack([it[k] for it in items]) for k in range(6))
    tt = lambda a: torch.tensor(a, device=dev)
    has_k = len(items[0]) > 10 and items[0][9] is not None
    k_lv = tt(np.stack([it[9] for it in items])) if has_k else None
    k_flag = tt(np.stack([it[10] for it in items])) if has_k else None
    has_s = len(items[0]) > 12 and items[0][11] is not None
    s_n = tt(np.stack([it[11] for it in items])) if has_s else None
    s_cover = tt(np.stack([it[12] for it in items])) if has_s else None
    has_d = len(items[0]) > 13 and items[0][13] is not None
    d_n = tt(np.stack([it[13] for it in items])) if has_d else None
    has_zr = len(items[0]) > 14 and items[0][14] is not None
    zr_n = tt(np.stack([it[14] for it in items])) if has_zr else None
    md_stem = getattr(model, "md_stem", None) if model is not None else None
    if model is not None and (md_stem is not None or (has_k and getattr(model, "ps_col", 0) > 0)):
        h_cnt, h_rows = (np.stack([it[k] for it in items]) for k in (7, 8))
        x = model.assemble(tt(t_n), tt(h_n), tt(h_valid), tt(t_cover), tt(ncol),
                           tt(h_cnt), tt(h_rows), k_lv, k_flag, s_n, s_cover, d_n, zr_n)
    else:
        x = assemble_x(tt(t_n), tt(h_n), tt(h_valid), tt(t_cover), tt(ncol))
    return x, torch.tensor(y, device=dev)


def rollout_paths(cls_logits: torch.Tensor, b: torch.Tensor, ncol: int, n_sample: int,
                  temp: float, gen: torch.Generator, win: float = 128.0,
                  start_col: int = 0) -> torch.Tensor:
    V, Tq, M = cls_logits.shape
    n = (V - 1) // 2
    h = 2 * win / Tq
    lsm = torch.log_softmax(cls_logits.float() / temp, dim=0)
    state = torch.full((n_sample,), Tq // 2, dtype=torch.long, device=cls_logits.device)
    pos = [torch.zeros(n_sample, device=cls_logits.device)]
    for k in range(start_col, start_col + ncol):
        p = lsm[:, state, k].T
        c = torch.multinomial(p.exp(), 1, generator=gen)[:, 0]
        state = (state + c - n).clamp(0, Tq - 1)
        bk = b[state, min(k + 1, M - 1)]
        pos.append((state.float() + bk) * h - win)
    return torch.stack(pos, 1)


def hist_logits_tf(hist: HistoryHead, feat: torch.Tensor, y_bnd: torch.Tensor,
                   win: float = 128.0, ps_col: int = 0,
                   k_lv: torch.Tensor | None = None,
                   k_flag: torch.Tensor | None = None) -> torch.Tensor:
    B, C, Tq, M = feat.shape
    bi, _, _, _ = path_bins(y_bnd, win, Tq)
    bins = bi[:, :M].clone()
    if ps_col > 0 and k_lv is not None:
        bins[:, :ps_col] = warmup_bins(k_lv, k_flag, ps_col, win, Tq)
    g = feat.gather(2, bins.view(B, 1, 1, M).expand(B, C, 1, M)).squeeze(2)
    d_prev = torch.zeros(B, M, device=feat.device, dtype=feat.dtype)
    d_prev[:, 1:] = (bins[:, 1:] - bins[:, :-1]).to(feat.dtype)
    return hist(g.permute(0, 2, 1), d_prev)[0]


def rollout_paths_hist(cls_logits: torch.Tensor, b: torch.Tensor, ncol: int, n_sample: int,
                       temp: float, gen: torch.Generator, win: float = 128.0,
                       start_col: int = 0, hist: HistoryHead | None = None,
                       feat: torch.Tensor | None = None,
                       warm_bins: torch.Tensor | None = None) -> torch.Tensor:
    V, Tq, M = cls_logits.shape
    n = (V - 1) // 2
    h_ft = 2 * win / Tq
    dev = cls_logits.device
    cl = cls_logits.float() if hist is not None else torch.log_softmax(cls_logits.float() / temp, dim=0)
    state = torch.full((n_sample,), Tq // 2, dtype=torch.long, device=dev)
    d_prev = torch.zeros(n_sample, device=dev)
    hs = None
    if hist is not None:
        if feat is None:
            raise ValueError("passing hist also requires feat (the fused feature map)")
        hs = hist.h0(n_sample, dev, feat.dtype)
        if warm_bins is not None and start_col > 0:
            wb = warm_bins[:start_col]
            cols = torch.arange(start_col, device=dev)
            fw = feat[:, wb, cols].T
            dw = torch.zeros(start_col, device=dev, dtype=feat.dtype)
            dw[1:] = (wb[1:] - wb[:-1]).to(feat.dtype)
            _, hs = hist(fw.unsqueeze(0).expand(n_sample, -1, -1),
                         dw.unsqueeze(0).expand(n_sample, -1), hs)
    pos = [torch.zeros(n_sample, device=dev)]
    for k in range(start_col, start_col + ncol):
        lg = cl[:, state, k].T
        if hist is not None:
            step, hs = hist(feat[:, state, k].T.unsqueeze(1),
                            d_prev.to(feat.dtype).unsqueeze(1), hs)
            lg = torch.log_softmax((lg + step[:, 0].float()) / temp, dim=1)
        c = torch.multinomial(lg.exp(), 1, generator=gen)[:, 0]
        nxt = (state + c - n).clamp(0, Tq - 1)
        d_prev = (nxt - state).float()
        state = nxt
        bk = b[state, min(k + 1, M - 1)]
        pos.append((state.float() + bk) * h_ft - win)
    return torch.stack(pos, 1)


@torch.no_grad()
def rollout_paths_batch(cls_logits: torch.Tensor, b: torch.Tensor, ncol: torch.Tensor,
                        n_sample: int, temp: float, gen: torch.Generator,
                        win: float = 128.0, start_col: int = 0) -> torch.Tensor:
    B, V, Tq, M = cls_logits.shape
    n = (V - 1) // 2
    h_ft = 2 * win / Tq
    dev = cls_logits.device
    lsm = torch.log_softmax(cls_logits.float() / temp, dim=1)
    state = torch.full((B, n_sample), Tq // 2, dtype=torch.long, device=dev)
    wi = torch.arange(B, device=dev).view(-1, 1).expand(B, n_sample).reshape(-1)
    pos = [torch.zeros(B, n_sample, device=dev)]
    for k in range(start_col, start_col + int(ncol.max())):
        act = (k - start_col < ncol).view(-1, 1)
        p = lsm[wi, :, state.reshape(-1), k]
        c = torch.multinomial(p.exp(), 1, generator=gen)[:, 0].view(B, n_sample)
        state = torch.where(act, (state + c - n).clamp(0, Tq - 1), state)
        bk = b[wi, state.reshape(-1), min(k + 1, M - 1)].view(B, n_sample)
        pos.append(torch.where(act, (state.float() + bk) * h_ft - win, pos[-1]))
    return torch.stack(pos, 2)


def dp_expected_level(cls_logits: torch.Tensor, b: torch.Tensor,
                      win: float = 128.0, start_col: int = 0,
                      start_level: torch.Tensor | float = 0.0,
                      mask: torch.Tensor | None = None) -> torch.Tensor:
    B, V, Tq, M = cls_logits.shape
    n = (V - 1) // 2
    h_ft = 2 * win / Tq
    probs = torch.softmax(cls_logits, dim=1)
    idx = (torch.arange(Tq, device=cls_logits.device).view(1, Tq) + 2 * n
           - torch.arange(V, device=cls_logits.device).view(V, 1))
    idx_b = idx.unsqueeze(0).expand(B, V, Tq)
    alpha = cls_logits.new_zeros(B, Tq)
    if isinstance(start_level, torch.Tensor):
        sb = ((start_level + win) / h_ft).long().clamp(0, Tq - 1)
        alpha.scatter_(1, sb[:, None], 1.0)
    else:
        alpha[:, int((start_level + win) / h_ft)] = 1.0
    if mask is not None:
        m = mask.to(alpha.dtype)
        m = torch.where(((alpha * m).sum(1) > 0)[:, None], m, torch.ones_like(m))
        alpha = alpha * m
        alpha = alpha / alpha.sum(1, keepdim=True).clamp_min(1e-30)
    rows = torch.arange(Tq, device=cls_logits.device, dtype=cls_logits.dtype)
    e_bnd = [cls_logits.new_zeros(B) + float(start_level) if not isinstance(start_level, torch.Tensor)
             else start_level.clone()]
    for _ in range(start_col):
        e_bnd.append(e_bnd[-1])
    for k in range(start_col, M):
        p_pad = F.pad(probs[:, :, :, k], (n, n))
        a_pad = F.pad(alpha, (n, n)).unsqueeze(1).expand(B, V, Tq + 2 * n)
        new = (a_pad.gather(2, idx_b) * p_pad.gather(2, idx_b)).sum(1)
        if mask is not None:
            nm = new * m
            new = torch.where(nm.sum(1, keepdim=True) > 0, nm, new)
        alpha = new / new.sum(1, keepdim=True).clamp_min(1e-30)
        bk = b[:, :, min(k + 1, M - 1)]
        e_bnd.append((alpha * ((rows + bk) * h_ft - win)).sum(1))
    eb = torch.stack(e_bnd, dim=1)
    return 0.5 * (eb[:, :-1] + eb[:, 1:])


def anchor_loss(cls_logits: torch.Tensor, b_pred: torch.Tensor, y_bnd: torch.Tensor,
                win: float = 128.0, huber_w: float = 1.0, soft: bool = False,
                hist_logits: torch.Tensor | None = None):
    B, V, Tq, M = cls_logits.shape
    n = (V - 1) // 2
    h_ft = 2 * win / Tq
    val = torch.isfinite(y_bnd)
    q = (torch.where(val, y_bnd, torch.zeros_like(y_bnd)) + win) / h_ft
    bi = torch.floor(q).long().clamp(0, Tq - 1)
    bt = (q - bi.float()).clamp(0.0, 1.0)

    vt = val[:, :-1] & val[:, 1:]
    cls_t = bi[:, 1:] - bi[:, :-1]
    in_vocab = cls_t.abs() <= n
    oov = (vt & ~in_vocab).float().sum() / vt.float().sum().clamp_min(1)
    vt = vt & in_vocab
    bidx, kidx = vt.nonzero(as_tuple=True)
    lg = cls_logits[bidx, :, bi[:, :-1][bidx, kidx], kidx]
    if hist_logits is not None:
        lg = lg + hist_logits[bidx, kidx].to(lg.dtype)
    if soft:
        d = (q[:, 1:] - q[:, :-1])[bidx, kidx]
        low = torch.floor(d)
        w_hi = d - low
        li = (low.long() + n).clamp(0, V - 1)
        hi = (li + 1).clamp(0, V - 1)
        lp = torch.log_softmax(lg, dim=1)
        ce = -((1.0 - w_hi) * lp.gather(1, li[:, None])[:, 0]
               + w_hi * lp.gather(1, hi[:, None])[:, 0]).mean()
    else:
        ce = F.cross_entropy(lg, cls_t[bidx, kidx] + n)

    bb, kb = val.nonzero(as_tuple=True)
    kcol = kb.clamp(max=M - 1)
    hb = F.huber_loss(b_pred[bb, bi[bb, kb], kcol], bt[bb, kb])
    return ce + huber_w * hb, ce.detach(), hb.detach(), oov.detach()
