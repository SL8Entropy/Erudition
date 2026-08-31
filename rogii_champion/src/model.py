"""ConvNeXt U-Net over the alignment canvas, plus the variant heads.

1st place's model is deliberately unexotic: a standard 2D U-Net with a
ConvNeXt-Small backbone, residual blocks in the decoder, LayerNorm replaced by
BatchNorm (which needs BF16 to stay finite), and average-pool / interpolate
resampling instead of learnable up- and down-sampling. The innovation is in the
formulation, the features and the augmentation -- not here.

Heads, switched on per variant:

* `map`    (always) one logit per pixel; softmax down each column is a
           distribution over TVT hypotheses at that MD.
* `dz`     (V2) per-column structural slope dz_layer/dMD.
* `move`   (V3) per-anchor distribution over discrete dTVT moves, for the DP
           expectation decode.
* `sigma`  (V5) per-column predictive standard deviation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    _HAS_TIMM = True
except Exception:  # pragma: no cover
    timm = None
    _HAS_TIMM = False


# ---------------------------------------------------------------------- #
# LayerNorm -> BatchNorm surgery
# ---------------------------------------------------------------------- #
class BNChannelsLast(nn.Module):
    """BatchNorm for the NHWC tensors inside a ConvNeXt block."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.bn = nn.BatchNorm2d(dim, eps=eps)

    def forward(self, x):
        if x.dim() == 4:
            return self.bn(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return self.bn(x)


def convert_ln_to_bn(module: nn.Module) -> nn.Module:
    """Replace every LayerNorm with BatchNorm, in place.

    1st place: "Replacing LayerNorm in ConvNeXt with BatchNorm consistently
    works better, although BF16 training is required to avoid NaN losses."
    """
    for name, child in list(module.named_children()):
        cname = type(child).__name__
        if cname in ("LayerNorm2d", "LayerNormExp2d"):
            dim = getattr(child, "normalized_shape", None)
            dim = dim[0] if dim else child.weight.shape[0]
            setattr(module, name, nn.BatchNorm2d(dim, eps=getattr(child, "eps", 1e-5)))
        elif isinstance(child, nn.LayerNorm):
            dim = child.normalized_shape[0]
            setattr(module, name, BNChannelsLast(dim, eps=child.eps))
        else:
            convert_ln_to_bn(child)
    return module


# ---------------------------------------------------------------------- #
# blocks
# ---------------------------------------------------------------------- #
class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = (nn.Identity() if in_ch == out_ch
                     else nn.Conv2d(in_ch, out_ch, 1, bias=False))
        self.act = nn.GELU()

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act(y + self.skip(x))


class TinyEncoder(nn.Module):
    """Fallback backbone so the pipeline runs without timm (tests, CPU boxes)."""

    def __init__(self, in_chans: int, dims=(48, 96, 192, 384)):
        super().__init__()
        self.feature_info = [dict(num_chs=d, reduction=r)
                             for d, r in zip(dims, (4, 8, 16, 32))]
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], 7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(dims[0]), nn.GELU())
        self.stages = nn.ModuleList()
        for i in range(1, 4):
            self.stages.append(nn.Sequential(
                nn.AvgPool2d(2),
                ResBlock(dims[i - 1], dims[i])))

    def forward(self, x):
        out = [self.stem(x)]
        for s in self.stages:
            out.append(s(out[-1]))
        return out


# ---------------------------------------------------------------------- #
def _make_encoder(backbone: str, pretrained: bool, in_chans: int,
                  weights_file: str = None):
    """Build the timm encoder, tolerating an offline environment.

    Kaggle notebooks have no internet, so backbone weights must come from an
    attached dataset. `weights_file` is passed to timm as a local checkpoint;
    if a download is attempted and fails, we fall back to random init with a
    loud warning rather than killing the run.
    """
    kw = dict(features_only=True, in_chans=in_chans, out_indices=(0, 1, 2, 3))
    if weights_file:
        return timm.create_model(
            backbone, pretrained=True,
            pretrained_cfg_overlay=dict(file=str(weights_file)), **kw)
    if pretrained:
        # Behind a TLS-inspecting proxy, certifi's bundle does not contain the
        # injected root CA and the huggingface download fails; the OS trust
        # store does have it. Harmless when there is no proxy.
        try:
            import truststore
            truststore.inject_into_ssl()
        except Exception:
            pass
    try:
        return timm.create_model(backbone, pretrained=pretrained, **kw)
    except Exception as e:
        if not pretrained:
            raise
        print(f"[model] pretrained weights unavailable ({type(e).__name__}); "
              f"falling back to random init. Pass backbone_weights=<file> to "
              f"load them from disk.")
        return timm.create_model(backbone, pretrained=False, **kw)


class AlignmentUNet(nn.Module):
    def __init__(self, in_chans: int, backbone: str = "convnext_small.in12k_ft_in1k_384",
                 pretrained: bool = True, decoder_ch=(224, 160, 96, 48, 32),
                 use_bn: bool = True, head_dz: bool = False,
                 head_move: bool = False, n_moves: int = 21,
                 move_row_stride: int = 4, head_sigma: bool = False,
                 grad_checkpoint: bool = False, backbone_weights: str = None):
        super().__init__()
        self.n_moves = n_moves
        self.move_row_stride = move_row_stride
        self.want_dz = head_dz
        self.want_move = head_move
        self.want_sigma = head_sigma

        if _HAS_TIMM and backbone:
            self.encoder = _make_encoder(backbone, pretrained, in_chans,
                                         backbone_weights)
            enc_ch = [f["num_chs"] for f in self.encoder.feature_info]
        else:
            if backbone:
                print("[model] timm unavailable -> fallback encoder "
                      "(fine for tests, not for a real run)")
            self.encoder = TinyEncoder(in_chans)
            enc_ch = [f["num_chs"] for f in self.encoder.feature_info]
        if use_bn:
            convert_ln_to_bn(self.encoder)
        if grad_checkpoint and hasattr(self.encoder, "set_grad_checkpointing"):
            self.encoder.set_grad_checkpointing(True)

        d0, d1, d2, d3, d4 = decoder_ch
        self.dec3 = ResBlock(enc_ch[3] + enc_ch[2], d0)      # /16
        self.dec2 = ResBlock(d0 + enc_ch[1], d1)             # /8
        self.dec1 = ResBlock(d1 + enc_ch[0], d2)             # /4
        self.dec0 = ResBlock(d2, d3)                         # /2
        self.dec_out = ResBlock(d3, d4)                      # /1
        self.head_map = nn.Conv2d(d4, 1, 1)

        if head_dz:
            self.head_dz = nn.Sequential(
                nn.Conv1d(d1, 64, 5, padding=2), nn.BatchNorm1d(64), nn.GELU(),
                nn.Conv1d(64, 1, 1))
        if head_move:
            self.head_move = nn.Sequential(
                nn.Conv2d(d2, 96, 3, padding=1), nn.BatchNorm2d(96), nn.GELU(),
                nn.Conv2d(96, n_moves, 1))
        if head_sigma:
            self.head_sigma = nn.Sequential(
                nn.Conv1d(d1, 64, 5, padding=2), nn.BatchNorm1d(64), nn.GELU(),
                nn.Conv1d(64, 1, 1))

    # ------------------------------------------------------------------ #
    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear",
                             align_corners=False)

    def forward(self, x):
        B, _, H, W = x.shape
        ph = (32 - H % 32) % 32
        pw = (32 - W % 32) % 32
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")

        f0, f1, f2, f3 = self.encoder(x)
        y = self.dec3(torch.cat([self._up(f3, f2), f2], 1))
        y2 = self.dec2(torch.cat([self._up(y, f1), f1], 1))
        y1 = self.dec1(torch.cat([self._up(y2, f0), f0], 1))
        y0 = self.dec0(F.interpolate(y1, scale_factor=2, mode="bilinear",
                                     align_corners=False))
        yf = self.dec_out(F.interpolate(y0, size=x.shape[-2:], mode="bilinear",
                                        align_corners=False))

        out = {"logits": self.head_map(yf)[:, 0, :H, :W]}

        if self.want_dz or self.want_sigma:
            col = y2.mean(dim=2)                       # pool over TVT rows
        if self.want_dz:
            dz = self.head_dz(col)
            out["dz"] = F.interpolate(dz, size=W, mode="linear",
                                      align_corners=False)[:, 0]
        if self.want_sigma:
            sg = self.head_sigma(col)
            out["log_sigma"] = F.interpolate(sg, size=W, mode="linear",
                                             align_corners=False)[:, 0]
        if self.want_move:
            m = self.head_move(y1)                     # (B, n_moves, H/4, W/4)
            hs = max(H // self.move_row_stride, 1)
            m = F.interpolate(m, size=(hs, W), mode="bilinear", align_corners=False)
            out["move_logits"] = m
        return out


# ---------------------------------------------------------------------- #
class EMA:
    """Exponential moving average of the weights, kept on the training device."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.buffers = {k: v.detach().clone()
                        for k, v in model.state_dict().items()
                        if not v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(),
                                                     alpha=1.0 - self.decay)
            else:
                self.buffers[k] = v.detach().clone()

    def state_dict(self):
        return {**{k: v.clone() for k, v in self.shadow.items()},
                **{k: v.clone() for k, v in self.buffers.items()}}


def build_model(cfg, in_chans: int, pretrained: bool = None):
    """`pretrained=False` is forced at inference: the checkpoint carries the
    weights, and Kaggle has no internet to download a backbone with."""
    return AlignmentUNet(
        in_chans=in_chans,
        backbone=cfg.backbone,
        pretrained=cfg.pretrained if pretrained is None else pretrained,
        backbone_weights=getattr(cfg, "backbone_weights", None),
        use_bn=cfg.use_batchnorm,
        head_dz=cfg.head_dz,
        head_move=cfg.head_move,
        n_moves=cfg.n_moves,
        move_row_stride=cfg.move_row_stride,
        head_sigma=cfg.head_sigma,
        grad_checkpoint=cfg.grad_checkpoint,
    )


def count_params(model: nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters())
    return f"{n / 1e6:.1f}M"
