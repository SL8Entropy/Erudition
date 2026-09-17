#!/usr/bin/env python3
"""Put a ConvNeXt trunk inside Bilzard's AnchorCNN, at the right resolution.

The 1st-place solution uses a ConvNeXt U-Net, the 2nd-place AnchorCNN an EfficientNet-B0.
Swapping one for the other isolates the backbone from everything else -- input channels,
loss, decoder and training recipe all stay Bilzard's.

The obstacle is the stem.  ConvNeXt begins with a 4x4 stride-4 patchify conv, so its
finest feature map lands at input/4, which is exactly the output state grid: a 1:1 ratio.
Both configurations at 1:1 failed in the resolution ablation (+0.9 and +1.7 ft), while
both at 2:1 worked, so a stock ConvNeXt is predicted to fail here for reasons that have
nothing to do with ConvNeXt.

Replacing that stem with a stride-2 conv restores the 2:1 ratio and reproduces the
EfficientNet baseline's geometry exactly:

    EfficientNet-B0, stem_stride=1   feature rows [128, 64, 32, 16], output 64  -> 2.0x
    ConvNeXt-small, retrofitted      feature rows [128, 64, 32, 16], output 64  -> 2.0x

The 1st-place pipeline solves the same problem the same way -- its notes record that the
patchify stem is skipped and a 1x1 conv feeds stage 0 instead.

Caveat to state with any result: the replacement stem is randomly initialised, so the
trunk keeps its pretrained stages but loses its pretrained stem.  Both trunks are
therefore pretrained-except-the-stem, which keeps the comparison fair.
"""

from __future__ import annotations

import torch.nn as nn

# The tag actually present in this machine's HuggingFace cache.  timm's default pretrained
# tag for "convnext_small" is a different one, and the hub is unreachable here, so the
# plain name is rewritten to this before loading.
OFFLINE_TAGS = {
    "convnext_small": "convnext_small.in12k_ft_in1k_384",
    "convnext_tiny": "convnext_tiny.in12k_ft_in1k_384",
}


def is_convnext(backbone: str) -> bool:
    return backbone.startswith("convnext")


def cached_weights(backbone: str):
    """Path to the ConvNeXt weights already in the local HuggingFace cache, if present.

    ``HF_HUB_OFFLINE`` is read by ``huggingface_hub`` at import time, and timm has already
    imported it by the time a model is built, so setting the variable from inside the
    process is too late.  Handing timm the file directly sidesteps the hub altogether --
    the same route the EfficientNet weights take.
    """
    from pathlib import Path
    tag = OFFLINE_TAGS.get(backbone, backbone)
    root = Path.home() / ".cache/huggingface/hub" / f"models--timm--{tag.replace('/', '--')}"
    hits = sorted(root.glob("snapshots/*/model.safetensors")) + sorted(root.glob("snapshots/*/*.bin"))
    return hits[0] if hits else None


def retrofit_stem(model, in_chans: int = 9, stride: int = 2, log=print) -> None:
    """Replace ConvNeXt's stride-4 patchify stem with a stride-``stride`` conv, in place.

    ``features_only`` ConvNeXt flattens its children, so the stem is ``trunk.stem_0``
    (a Conv2d) followed by ``trunk.stem_1`` (a LayerNorm2d, left alone).  Only the
    convolution changes, so channel counts -- and therefore the FPN projections built
    from ``feature_info.channels()`` -- are untouched.
    """
    trunk = getattr(model, "trunk", None)
    stem = getattr(trunk, "stem_0", None)
    if not isinstance(stem, nn.Conv2d):
        raise ValueError("expected a ConvNeXt features_only trunk with a Conv2d at stem_0; "
                         f"got {type(stem).__name__}")
    old = f"{stem.kernel_size[0]}x{stem.kernel_size[1]} stride {stem.stride[0]}"
    trunk.stem_0 = nn.Conv2d(in_chans, stem.out_channels, kernel_size=3, stride=stride,
                             padding=1, bias=stem.bias is not None)
    log(f"convnext stem: {old} -> 3x3 stride {stride} (randomly initialised); "
        f"the pretrained stages are kept")
