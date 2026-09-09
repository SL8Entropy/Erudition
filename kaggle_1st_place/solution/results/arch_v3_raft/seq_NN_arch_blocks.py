"""Architecture experiments for the sequence U-Net.

Three self-contained building blocks, each attacking one weakness of the
archived design.  All three are written so that a freshly built model starts
*numerically identical* to the baseline -- every added path is gated by a
zero-initialised scale, or begins from the baseline's own prediction -- so a
run that fails does so because the idea is wrong, not because the initial
weights were disturbed.

Geometry note used throughout.  The feature maps are ``(B, C, H, W)`` with

* ``H`` = MD, distance along the hole.  345 columns at full resolution,
  falling to 173 / 87 / 44 / 22 through the ConvNeXt stages.
* ``W`` = candidate stratigraphic level.  400 rows at full resolution.

The two axes mean entirely different things, and the archived model treats them
identically.  Two of the three blocks here exist only to stop doing that.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# 1. Anisotropic large-kernel branch  (long along MD, short across level)
# --------------------------------------------------------------------------


class AnisotropicLargeKernelDW(nn.Module):
    """Wrap a pretrained depthwise conv with a parallel long-along-MD kernel.

    RepLKNet/SLaK show that a single very large depthwise kernel buys effective
    receptive field far more cheaply than stacking depth.  A *square* 31x31 or
    51x51 kernel is mostly wasted here though: the level axis needs only a few
    rows of context, while the MD axis needs as much as it can get.  So the
    added kernel is a long thin rectangle, e.g. 31 tall (MD) by 3 wide (level).

    The branch is summed onto the original conv through a per-channel scale
    initialised to zero, so at step zero the block computes exactly what the
    pretrained 7x7 depthwise conv computed.  The large kernel can only be an
    addition to the pretrained function, never a replacement for it.
    """

    def __init__(self, base_conv: nn.Conv2d, md_kernel: int, level_kernel: int = 3):
        super().__init__()
        if not isinstance(base_conv, nn.Conv2d):
            raise TypeError(f"expected a Conv2d to wrap, got {type(base_conv).__name__}")
        channels = base_conv.in_channels
        if base_conv.groups != channels:
            raise ValueError(
                "AnisotropicLargeKernelDW expects a depthwise conv "
                f"(groups={base_conv.groups}, channels={channels})"
            )
        md_kernel = int(md_kernel)
        level_kernel = int(level_kernel)
        if md_kernel % 2 == 0 or level_kernel % 2 == 0:
            raise ValueError(f"kernel sizes must be odd, got ({md_kernel}, {level_kernel})")
        self.base_conv = base_conv
        self.large_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(md_kernel, level_kernel),
            padding=(md_kernel // 2, level_kernel // 2),
            groups=channels,
            bias=False,
        )
        nn.init.zeros_(self.large_conv.weight)
        # Per-channel gate, zero at init: the branch starts switched off.
        self.gate = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        return self.base_conv(x) + self.gate * self.large_conv(x)


def apply_anisotropic_large_kernels(backbone_stages, md_kernels, level_kernel=3, log=None):
    """Add a long-along-MD branch to every depthwise conv in each stage.

    ``md_kernels`` gives one kernel length per stage.  Later stages have fewer
    MD columns left (173 / 87 / 44 / 22 for the archived geometry), so the
    kernel is tapered rather than kept constant -- a 31-long kernel on a
    22-column feature map is just an expensive global average.
    """

    if len(md_kernels) != len(backbone_stages):
        raise ValueError(
            f"need one MD kernel per stage: {len(md_kernels)} given for "
            f"{len(backbone_stages)} stages"
        )
    wrapped = 0
    for stage, md_kernel in zip(backbone_stages, md_kernels):
        if int(md_kernel) <= 1:
            continue
        for module in stage.modules():
            conv = getattr(module, "conv_dw", None)
            if isinstance(conv, nn.Conv2d) and conv.groups == conv.in_channels:
                module.conv_dw = AnisotropicLargeKernelDW(
                    conv,
                    md_kernel=int(md_kernel),
                    level_kernel=int(level_kernel),
                )
                wrapped += 1
    if log is not None:
        log(
            f"anisotropic large kernels: wrapped {wrapped} depthwise convs, "
            f"md_kernels={tuple(int(k) for k in md_kernels)}, level_kernel={int(level_kernel)}"
        )
    return wrapped


# --------------------------------------------------------------------------
# 2. Axial attention along MD
# --------------------------------------------------------------------------


class AxialMDAttention(nn.Module):
    """Self-attention along the MD axis only, applied independently per level.

    Resolving an ambiguous GR match needs evidence from thousands of feet away,
    and a convolution stack only relays that through depth, diluting it at every
    hop.  Full 2-D attention over 345x400 positions is impossible in 6 GB --
    the attention matrix alone would be over a terabyte.  Attending along one
    axis at a time costs O(H*W*(H+W)) instead of O((H*W)^2), which at these
    sizes is the difference between megabytes and terabytes.

    Position is supplied by a depthwise 3x1 convolution rather than a learned
    table (conditional positional encoding), so the block does not care what MD
    length it is given and can be dropped into any stage.

    LayerScale starts at zero: at initialisation the block is the identity.
    """

    def __init__(self, channels, num_heads=4, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        if channels % num_heads != 0:
            # Fall back to the largest head count that divides the width.
            num_heads = math.gcd(channels, num_heads) or 1
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.pos_conv = nn.Conv2d(
            channels, channels, kernel_size=(3, 1), padding=(1, 0), groups=channels, bias=True
        )
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3, bias=True)
        self.proj = nn.Linear(channels, channels, bias=True)
        self.norm2 = nn.LayerNorm(channels)
        hidden = max(1, int(channels * float(mlp_ratio)))
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
        )
        # The positional conv is applied outside the gated residual, so it has
        # to be zero-initialised too or the block would perturb the pretrained
        # features on step one and we would be measuring that damage instead of
        # the idea.
        nn.init.zeros_(self.pos_conv.weight)
        nn.init.zeros_(self.pos_conv.bias)
        self.gamma_attn = nn.Parameter(torch.zeros(channels))
        self.gamma_mlp = nn.Parameter(torch.zeros(channels))
        self.dropout = float(dropout)

    def forward(self, x):
        # x: (B, C, H=MD, W=level)
        x = x + self.pos_conv(x)
        batch, channels, height, width = x.shape
        # One sequence per (sample, level): length H along MD.
        tokens = x.permute(0, 3, 2, 1).reshape(batch * width, height, channels)

        residual = tokens
        normed = self.norm1(tokens)
        qkv = self.qkv(normed).reshape(
            batch * width, height, 3, self.num_heads, channels // self.num_heads
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(
            qkv[0], qkv[1], qkv[2], dropout_p=self.dropout if self.training else 0.0
        )
        attended = attended.transpose(1, 2).reshape(batch * width, height, channels)
        tokens = residual + self.gamma_attn * self.proj(attended)
        tokens = tokens + self.gamma_mlp * self.mlp(self.norm2(tokens))

        return tokens.reshape(batch, width, height, channels).permute(0, 3, 2, 1).contiguous()


class AxialMDStack(nn.Module):
    """A short stack of MD-axial attention blocks."""

    def __init__(self, channels, depth=1, num_heads=4, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            AxialMDAttention(channels, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(int(depth))
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# --------------------------------------------------------------------------
# 3. Iterative refinement of the level path (RAFT / IGEV style)
# --------------------------------------------------------------------------


class ConvGRU1d(nn.Module):
    """Gated recurrent update over the MD axis, applied convolutionally."""

    def __init__(self, hidden_dim, input_dim, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.conv_z = nn.Conv1d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=padding)
        self.conv_r = nn.Conv1d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=padding)
        self.conv_q = nn.Conv1d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=padding)

    def forward(self, hidden, inputs):
        joined = torch.cat([hidden, inputs], dim=1)
        update = torch.sigmoid(self.conv_z(joined))
        reset = torch.sigmoid(self.conv_r(joined))
        candidate = torch.tanh(self.conv_q(torch.cat([reset * hidden, inputs], dim=1)))
        return (1.0 - update) * hidden + update * candidate


class IterativeLevelRefiner(nn.Module):
    """Refine a per-column level estimate by repeated small corrections.

    The archived model gets one shot: it softmaxes the match scores down the
    level axis and reports the mean.  Where the posterior is genuinely bimodal
    -- which is the normal case when rock patterns repeat -- that mean lands in
    the empty valley between the two modes, and the answer belongs to neither.
    This is the documented failure mode of soft-argmin disparity regression.

    Instead this head starts from that same one-shot estimate and then, for
    ``num_iters`` rounds, reads the match scores in a window *around wherever it
    currently thinks it is* and predicts a small correction.  A gated recurrent
    unit running along MD carries state between rounds, so corrections stay
    consistent along the hole.  Refinement commits to one mode rather than
    averaging two, and the running estimate is free to walk outside the window
    the one-shot expectation was confined to.

    ``num_iters`` is a test-time dial: more rounds cost time, not training.
    """

    def __init__(
        self,
        feature_dim,
        hidden_dim=96,
        context_dim=96,
        num_iters=8,
        lookup_radius=4,
        lookup_dilations=(1, 4, 16),
        gru_kernel_size=5,
    ):
        super().__init__()
        self.num_iters = int(num_iters)
        self.lookup_radius = int(lookup_radius)
        self.lookup_dilations = tuple(int(d) for d in lookup_dilations)
        self.hidden_dim = int(hidden_dim)

        lookup_dim = len(self.lookup_dilations) * (2 * self.lookup_radius + 1)
        self.context_encoder = nn.Sequential(
            nn.Conv1d(feature_dim * 2, context_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(context_dim, context_dim + hidden_dim, kernel_size=3, padding=1),
        )
        self.context_dim = int(context_dim)
        motion_in = lookup_dim + 1
        self.motion_encoder = nn.Sequential(
            nn.Conv1d(motion_in, context_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(context_dim, context_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.gru = ConvGRU1d(hidden_dim, context_dim * 2, kernel_size=gru_kernel_size)
        self.delta_head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1),
        )
        # Start with zero corrections so iteration 0 reproduces the one-shot
        # answer and the refiner has to earn every departure from it.
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def _lookup(self, cost, level_index):
        """Sample the match-score volume around the current level estimate.

        ``cost`` is (B, H, W) match scores; ``level_index`` is a continuous
        per-column position along W.  Returns (B, L, H) samples taken at a
        multi-scale set of offsets around that position.
        """

        batch, height, width = cost.shape
        offsets = []
        for dilation in self.lookup_dilations:
            offsets.append(
                torch.arange(
                    -self.lookup_radius, self.lookup_radius + 1, device=cost.device, dtype=cost.dtype
                )
                * float(dilation)
            )
        offsets = torch.cat(offsets)  # (L,)
        sample_count = offsets.numel()

        # grid_sample over a (B*H, 1, 1, W) "image": one row per MD column.
        positions = level_index.reshape(batch * height, 1, 1, 1) + offsets.reshape(1, 1, -1, 1)
        normalized = 2.0 * positions / max(width - 1, 1) - 1.0
        grid = torch.cat([normalized, torch.zeros_like(normalized)], dim=-1)
        source = cost.reshape(batch * height, 1, 1, width)
        sampled = F.grid_sample(
            source, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        return sampled.reshape(batch, height, sample_count).permute(0, 2, 1)

    def forward(self, features, cost, init_level_index, axis_scale):
        """features (B, C, H, W); cost (B, H, W); init_level_index (B, H)."""

        pooled_mean = features.mean(dim=-1)
        pooled_max = features.amax(dim=-1)
        context = self.context_encoder(torch.cat([pooled_mean, pooled_max], dim=1))
        context, hidden = torch.split(context, [self.context_dim, self.hidden_dim], dim=1)
        context = F.gelu(context)
        hidden = torch.tanh(hidden)

        width = cost.shape[-1]
        level_index = init_level_index
        predictions = []
        for _ in range(self.num_iters):
            lookup = self._lookup(cost, level_index.detach())
            normalized_level = (2.0 * level_index / max(width - 1, 1) - 1.0).unsqueeze(1)
            motion = self.motion_encoder(torch.cat([lookup, normalized_level], dim=1))
            hidden = self.gru(hidden, torch.cat([motion, context], dim=1))
            delta = self.delta_head(hidden).squeeze(1)
            level_index = (level_index + delta).clamp(0.0, float(width - 1))
            predictions.append(level_index)

        return [index * axis_scale[0] + axis_scale[1] for index in predictions]
