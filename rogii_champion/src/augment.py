"""The augmentation stack.

1st place names two augmentations as far more important than the rest: Z-shift
(a resampled trajectory through fixed geology, in `synth.py`) and the GR affine
transform on the typewell, which forces the model to match *shape* rather than
absolute level. Everything else is here.

Two are structural rather than cosmetic and come from 2nd place: work in
z_layer space so that scaling or flipping the structural slope keeps the
identity dTVT = dz_layer - dz exact, and mask GR spans in exactly the format
real missing data takes.

Canvas-level augmentations (channel masking, PF corruption) run after the image
is built. PF corruption is the anti-shortcut measure: if a strong model's
output is a channel, corrupt it during training or the network will copy it and
stop reading the gamma ray.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .synth import PathBank, synth_well


@dataclass
class AugConfig:
    p_synth: float = 0.5           # Z-shift / synthetic regeneration
    p_gr_affine: float = 0.6       # typewell GR' = a*GR + b
    gr_affine_a: tuple = (0.80, 1.25)
    gr_affine_b: tuple = (-12.0, 12.0)
    p_reverse: float = 0.20
    p_md_stretch: float = 0.35
    md_stretch: tuple = (0.85, 1.18)
    p_level_shift: float = 0.35
    level_shift_ft: float = 32.0
    p_level_flip: float = 0.15
    p_struct_scale: float = 0.30   # stretch the structural slope in z_layer space
    struct_scale: tuple = (0.70, 1.40)
    p_gr_noise_shift: float = 0.30
    p_gr_specaug: float = 0.40
    specaug_max_ft: float = 672.0  # 21 columns, as 2nd place used
    p_tail_crop: float = 0.25
    p_tw_mask: float = 0.25
    p_tw_jitter: float = 0.30
    tw_jitter: float = 0.05
    p_hw_noise: float = 0.40
    hw_noise: float = 0.06
    p_prefix_resample: float = 0.30   # randomise how much is visible
    prefix_frac: tuple = (0.25, 0.75)
    p_channel_mask: float = 0.30
    channel_mask_n: int = 2
    p_pf_corrupt: float = 0.35
    pf_shift_ft: float = 20.0


# ---------------------------------------------------------------------- #
# well-level transforms
# ---------------------------------------------------------------------- #
def _resample(w, new_md):
    """Re-interpolate every per-row array onto a new MD grid."""
    old = w.md
    for name in ("x", "y", "z", "tvt"):
        v = getattr(w, name)
        if v is not None:
            setattr(w, name, np.interp(new_md, old, v))
    # NaNs must survive interpolation as NaNs, so carry a validity curve
    for name in ("gr", "tvt_input"):
        v = getattr(w, name)
        ok = np.isfinite(v).astype(float)
        vi = np.interp(new_md, old, np.nan_to_num(v))
        oi = np.interp(new_md, old, ok)
        setattr(w, name, np.where(oi > 0.5, vi, np.nan))
    w.md = new_md
    w.row_idx = np.arange(len(new_md))
    return w


def md_stretch(w, rng, cfg):
    """Same geology traversed at a different rate: beds get wider or narrower
    in MD while the 1 ft row spacing is preserved."""
    f = float(rng.uniform(*cfg.md_stretch))
    step = float(np.median(np.diff(w.md))) if w.n > 1 else 1.0
    n = max(int(round(w.n * f)), 640)
    new_md = w.md[0] + np.arange(n) * step
    src_md = np.clip(w.md[0] + (new_md - w.md[0]) / f, w.md[0], w.md[-1])
    w2 = _resample(w, src_md)
    w2.md = new_md
    return w2


def reverse_path(w, rng, cfg, keep_frac: float = 0.35):
    """Traverse the well backwards. The physics is direction-symmetric, and the
    prefix is rebuilt from the new start so the task shape is unchanged."""
    if w.tvt is None:
        return w
    n = w.n
    for name in ("x", "y", "z", "gr", "tvt"):
        v = getattr(w, name)
        if v is not None:
            setattr(w, name, v[::-1].copy())
    d = np.diff(w.md)
    w.md = w.md[0] + np.concatenate([[0.0], np.cumsum(d[::-1])])
    w.row_idx = np.arange(n)
    ps = int(np.clip(round(n * float(rng.uniform(0.2, keep_frac + 0.2))), 64, n - 64))
    w.tvt_input = np.where(np.arange(n) < ps, w.tvt, np.nan)
    return w


def level_shift(w, rng, cfg):
    """Move the whole system in level space: TVT and the typewell axis together."""
    c = float(rng.uniform(-cfg.level_shift_ft, cfg.level_shift_ft))
    w.tw_tvt = w.tw_tvt + c
    for name in ("tvt", "tvt_input"):
        v = getattr(w, name)
        if v is not None:
            setattr(w, name, v + c)
    return w


def level_flip(w, rng, cfg):
    """Turn the whole subsurface upside down: mirror TVT, the typewell axis and
    the trajectory together.

    This is 4th place's physically consistent vertical flip rather than a plain
    image flip. Mirroring TVT about the anchor and Z about its anchor value
    leaves GR matching exact and negates dTVT, dZ and dz_layer together, so
    every geometry channel stays self-consistent.
    """
    if w.tvt is None:
        return w
    a, z0 = w.anchor_tvt, w.anchor_z
    w.tw_tvt, w.tw_gr = (2 * a - w.tw_tvt)[::-1].copy(), w.tw_gr[::-1].copy()
    for name in ("tvt", "tvt_input"):
        v = getattr(w, name)
        if v is not None:
            setattr(w, name, 2 * a - v)
    w.z = 2 * z0 - w.z
    return w


def struct_scale(w, rng, cfg):
    """Scale the structural slope in z_layer space, keeping TVT and GR intact.

    S = TVT + Z is stretched about its anchor and Z is re-derived as S - TVT,
    so dTVT = dz_layer - dz still holds exactly. The model sees a steeper or
    flatter formation without any inconsistency in the gamma ray.
    """
    if w.tvt is None:
        return w
    a = float(rng.uniform(*cfg.struct_scale))
    S = w.tvt + w.z
    i0 = max(w.ps_idx - 1, 0)
    S_new = S[i0] + (S - S[i0]) * a
    w.z = S_new - w.tvt
    return w


def gr_affine(w, rng, cfg):
    a = float(rng.uniform(*cfg.gr_affine_a))
    b = float(rng.uniform(*cfg.gr_affine_b))
    w.tw_gr = a * w.tw_gr + b
    return w


def tw_jitter(w, rng, cfg):
    sd = cfg.tw_jitter * float(np.nanstd(w.tw_gr) + 1e-6)
    w.tw_gr = w.tw_gr + rng.normal(0.0, sd, len(w.tw_gr))
    return w


def tw_mask(w, rng, cfg):
    """Blank spans of the typewell: real typewells have gaps in coverage."""
    n = len(w.tw_gr)
    for _ in range(int(rng.integers(1, 3))):
        ln = int(rng.integers(max(n // 40, 2), max(n // 8, 4)))
        s = int(rng.integers(0, max(n - ln, 1)))
        w.tw_gr[s:s + ln] = np.nan
    return w


def gr_specaug(w, rng, cfg):
    n = w.n
    for _ in range(int(rng.integers(1, 4))):
        ln = int(rng.integers(16, max(int(cfg.specaug_max_ft), 32)))
        s = int(rng.integers(0, max(n - ln, 1)))
        w.gr[s:s + ln] = np.nan
    return w


def gr_noise_shift(w, rng, cfg):
    """Circularly shift the GR residual inside the well.

    Keeps the residual's real texture but decorrelates it from the true path,
    so the model cannot memorise a well by its noise.
    """
    if w.tvt is None:
        return w
    ref = np.interp(w.tvt, w.tw_tvt, w.tw_gr, left=np.nan, right=np.nan)
    res = np.nan_to_num(w.gr - ref)
    k = int(rng.integers(200, max(w.n // 2, 400)))
    w.gr = np.where(np.isfinite(w.gr), ref + np.roll(res, k), np.nan)
    return w


def hw_noise(w, rng, cfg):
    sd = cfg.hw_noise * float(np.nanstd(w.gr) + 1e-6)
    w.gr = w.gr + rng.normal(0.0, sd, w.n)
    return w


def tail_crop(w, rng, cfg):
    n = w.n
    lo = min(max(w.ps_idx + 512, n // 2), n - 1)
    if lo >= n:
        return w
    keep = int(rng.integers(lo, n + 1))
    for name in ("md", "x", "y", "z", "gr", "tvt", "tvt_input", "row_idx"):
        v = getattr(w, name)
        if v is not None:
            setattr(w, name, v[:keep])
    return w


def prefix_resample(w, rng, cfg):
    """Randomise the visible fraction (4th place: ~25-65%).

    The prefix length varies between wells, and a model tuned to one ratio
    degrades on the others.
    """
    if w.tvt is None:
        return w
    f = float(rng.uniform(*cfg.prefix_frac))
    ps = int(np.clip(round(w.n * f), 128, w.n - 128))
    w.tvt_input = np.where(np.arange(w.n) < ps, w.tvt, np.nan)
    return w


WELL_AUGS = (
    ("p_prefix_resample", prefix_resample),
    ("p_reverse", reverse_path),
    ("p_md_stretch", md_stretch),
    ("p_tail_crop", tail_crop),
    ("p_level_flip", level_flip),
    ("p_struct_scale", struct_scale),
    ("p_level_shift", level_shift),
    ("p_gr_affine", gr_affine),
    ("p_tw_jitter", tw_jitter),
    ("p_tw_mask", tw_mask),
    ("p_gr_noise_shift", gr_noise_shift),
    ("p_hw_noise", hw_noise),
    ("p_gr_specaug", gr_specaug),
)


def augment_well(well, cfg: AugConfig, rng, bank: PathBank = None):
    w = well.copy()
    if bank is not None and rng.random() < cfg.p_synth:
        w = synth_well(w, bank, rng)
    for pname, fn in WELL_AUGS:
        if rng.random() < getattr(cfg, pname):
            try:
                w = fn(w, rng, cfg)
            except Exception:
                pass                     # never let an augmentation kill a batch
    return w


# ---------------------------------------------------------------------- #
# canvas-level transforms
# ---------------------------------------------------------------------- #
def augment_canvas(img: np.ndarray, channels, cfg: AugConfig, rng):
    """Channel dropout, and deliberate corruption of the cross-model channels."""
    if rng.random() < cfg.p_channel_mask:
        idx = rng.choice(len(channels), size=min(cfg.channel_mask_n, len(channels)),
                         replace=False)
        img[idx] = 0.0
    if cfg.p_pf_corrupt > 0 and rng.random() < cfg.p_pf_corrupt:
        row_ft = 0.5
        for group, roll_axis in (("pf_", 1), ("xy_", None)):
            ids = [i for i, c in enumerate(channels) if c.startswith(group)]
            if not ids:
                continue
            if rng.random() < 0.35:
                img[ids] = 0.0                       # drop it entirely
            elif roll_axis is not None:
                k = int(round(rng.normal(0.0, cfg.pf_shift_ft / row_ft)))
                img[ids] = np.roll(img[ids], k, axis=roll_axis)
            else:
                img[ids] += rng.normal(0.0, cfg.pf_shift_ft / 100.0)
    return img
