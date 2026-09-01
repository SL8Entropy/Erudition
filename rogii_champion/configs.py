"""Configuration registry.

`base` is a faithful reproduction of the 1st place solution. Each of the five
variants changes exactly one thing, borrowed from a different team, and each
attacks a different stage of the pipeline -- so that when a variant wins or
loses you know what caused it.

    variant      borrowed from   stage changed        cost
    ---------------------------------------------------------------------
    v1_tta       2nd + 4th + 5th decode / TTA         none (reuses weights)
    v2_slope     2nd             target parameters    retrain
    v3_moves     2nd             head + decoder       retrain
    v4_refgr     3rd + 5th       input features       retrain
    v5_gate      3rd             uncertainty/ensemble retrain + gate fit

Run `python run.py list` to see them.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace

from src.augment import AugConfig
from src.canvas import MD_STRIDE, N_ROWS, N_TGT_COLS, N_VIS_COLS, ROW_FT


@dataclass
class Config:
    name: str = "base"
    notes: str = ""

    # ---- canvas ---------------------------------------------------- #
    n_rows: int = N_ROWS               # 400 rows x 0.5 ft = +-100 ft
    row_ft: float = ROW_FT
    md_stride: float = MD_STRIDE       # 32 ft per column
    n_vis_cols: int = N_VIS_COLS       # 1024 ft of visible region
    n_tgt_cols: int = N_TGT_COLS       # 10016 ft of prediction region
    use_pf: bool = True                # particle-filter channels
    use_xy: bool = True                # XY-neighbour channels
    use_sibling: bool = False          # V4
    correct_typewell: bool = False     # V4 / V1 inference
    tw_calib_blend: float = 0.5

    # ---- model ----------------------------------------------------- #
    backbone: str = "convnext_small.in12k_ft_in1k_384"
    pretrained: bool = True
    backbone_weights: str = None   # local weights file for offline runs
    reuse_weights: str = None      # train nothing; run another variant's weights
    bench_note: str = ""           # caveat printed in the benchmark report
    use_batchnorm: bool = True         # needs bf16, per 1st place
    grad_checkpoint: bool = False
    head_dz: bool = False
    head_move: bool = False
    head_sigma: bool = False
    n_moves: int = 21                  # {0, +-2, ..., +-20} ft
    move_step: float = 2.0
    move_row_stride: int = 4           # move grid at 4 x 0.5 = 2 ft per level bin

    # ---- loss ------------------------------------------------------ #
    w_ce: float = 1.0
    ce_tau: float = 2.0                # ft; width of the smoothed CE target
    w_huber: float = 0.08
    huber_delta: float = 5.0
    w_gr: float = 0.02
    w_dz: float = 0.0
    w_dz_integ: float = 1.0
    w_move: float = 0.0
    w_nll: float = 0.0
    vis_col_weight: float = 0.2        # visible columns are nearly free wins

    # ---- training -------------------------------------------------- #
    epochs: int = 30
    batch_size: int = 4
    accum_steps: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    ema_decay: float = 0.999
    amp: str = "bf16"                  # bf16 is required with BatchNorm
    num_workers: int = 4
    grad_clip: float = 1.0
    seed: int = 0
    n_folds: int = 5
    folds: tuple = (0, 1, 2, 3, 4)
    seeds: tuple = (0, 1, 2)           # 3 seeds x 5 folds, as 1st place
    split_patterns: tuple = (0,)       # >1 = 3rd place's multiple assignments
    synth_per_epoch: float = 1.0       # synthetic wells per real well per epoch

    # ---- inference ------------------------------------------------- #
    n_phases: int = 1                  # MD-phase TTA views
    reanchor: float = 0.0              # fraction into the prediction region
    adaptive_canvas: bool = False
    dp_weight: float = 0.0             # weight of the DP path vs the map path

    # ---- augmentation ---------------------------------------------- #
    aug: AugConfig = field(default_factory=AugConfig)

    # ---------------------------------------------------------------- #
    def canvas_spec(self):
        from src.canvas import CanvasSpec
        return CanvasSpec(
            n_rows=self.n_rows, row_ft=self.row_ft, md_stride=self.md_stride,
            n_vis_cols=self.n_vis_cols, n_tgt_cols=self.n_tgt_cols,
            use_pf=self.use_pf, use_xy=self.use_xy, use_sibling=self.use_sibling,
            tw_calib_blend=self.tw_calib_blend)

    def clone(self, **kw):
        c = copy.deepcopy(self)
        return replace(c, **kw) if kw else c


# ---------------------------------------------------------------------- #
BASE = Config(
    name="base",
    notes=("1st place reproduction: 2D alignment on a 400x352 canvas, "
           "ConvNeXt-Small U-Net with BatchNorm, smoothed cross-entropy plus "
           "expected-path Huber and a GR penalty, PF and XY-neighbour "
           "channels, Z-shift and GR-affine augmentation."),
)

# -- V1 -- decode only. No retraining: it reuses whatever weights exist.
V1 = BASE.clone(
    name="v1_tta",
    notes=("Inference-side only. MD-phase TTA over 8 sub-column offsets (2nd), "
           "quarter-point re-anchoring (4th), adaptive taller canvas for wells "
           "that leave the TVT window (5th), and a sibling-corrected typewell "
           "(5th). Same weights as base, so any change is purely the decode."),
    n_phases=8,
    reanchor=0.25,
    adaptive_canvas=True,
    correct_typewell=True,
    reuse_weights="base",
)

# -- V2 -- predict the structural slope instead of the path.
V2 = BASE.clone(
    name="v2_slope",
    notes=("2nd place's key insight. dTVT = dz_layer - dz is an identity and dz "
           "is known at test time, so an auxiliary head predicts the smooth "
           "structural slope dz_layer/dMD and the path is integrated against "
           "the real trajectory. The coefficient on dz never has to be learned. "
           "Paired with z_layer-space slope-scaling augmentation."),
    head_dz=True,
    w_dz=0.06,          # measured: at 0.35 the dz term ran 5x the CE term,
    w_dz_integ=0.25,    # so the model optimised the aux head instead
)
V2.aug.p_struct_scale = 0.55

# -- V3 -- conditional move field + exact DP decode.
V3 = BASE.clone(
    name="v3_moves",
    notes=("2nd place's AnchorCNN head alongside the alignment map: at every "
           "(level, column) anchor predict P(dTVT | TVT) over 21 discrete "
           "moves, teacher-forced on the true path, then decode by exact DP "
           "marginalisation. The map reads each column independently; the DP "
           "path cannot teleport between columns. Blended 70/30."),
    head_move=True,
    w_move=0.30,
    dp_weight=0.30,
)

# -- V4 -- better reference GR.
V4 = BASE.clone(
    name="v4_refgr",
    notes=("3rd place's largest private-LB lever plus 5th place's typewell "
           "correction: match against sibling laterals sharing the typewell "
           "(0.25 ft bins) and against the well's own pre-PS prefix, and "
           "correct the typewell itself with the sibling median residual. "
           "Adds 7 channels."),
    use_sibling=True,
    correct_typewell=True,
)

# -- V5 -- know what you do not know, then gate on it.
V5 = BASE.clone(
    name="v5_gate",
    notes=("3rd place's uncertainty + gating. A Gaussian-NLL head reports a "
           "per-row sigma, and a small sequence gate mixes the candidates "
           "(map path, PF, XY plane, flat-layer prior) with softmax weights "
           "that vary along the well. Convex mixing is a hard constraint that "
           "beat an unconstrained refiner by 0.12 for 3rd place. Trained over "
           "3 GroupKFold split patterns rather than 1."),
    head_sigma=True,
    w_nll=0.15,
    split_patterns=(0, 101, 202),
    seeds=(0,),
    bench_note=("The gate is NOT part of this number. A gate trained on the "
                "held-out fold would be fitting the test set; it needs OOF "
                "candidates from every fold, so it can only be fitted after a "
                "full 5-fold run. What is measured here is the Gaussian-NLL "
                "model on its own."),
)

# -- diagnostic -- is the augmentation load the reason the score plateaus?
LOWAUG = BASE.clone(
    name="base_lowaug",
    notes=("Diagnostic, not an idea from anyone. Identical to base except the "
           "augmentation load: a quarter as many synthetic wells and roughly "
           "half the per-sample augmentation probability. base sees ~75% "
           "synthetic-or-augmented samples, which suits a 100+ epoch schedule; "
           "this asks whether that is what is holding a 30-epoch run back."),
    synth_per_epoch=0.25,
)
LOWAUG.aug.p_synth = 0.15
for _k in ("p_gr_affine", "p_reverse", "p_md_stretch", "p_level_shift",
           "p_level_flip", "p_struct_scale", "p_gr_noise_shift", "p_gr_specaug",
           "p_tail_crop", "p_tw_mask", "p_tw_jitter", "p_hw_noise",
           "p_prefix_resample", "p_channel_mask", "p_pf_corrupt"):
    setattr(LOWAUG.aug, _k, getattr(LOWAUG.aug, _k) * 0.5)


# A tiny configuration for smoke tests and CPU boxes.
SMOKE = BASE.clone(
    name="smoke",
    notes="Tiny grid and fallback backbone; for the end-to-end self-test.",
    backbone="", pretrained=False, n_rows=128, row_ft=1.0, md_stride=32.0,
    n_vis_cols=8, n_tgt_cols=56, epochs=1, batch_size=2, seeds=(0,), folds=(0,),
    n_folds=2, use_pf=False, use_xy=True, num_workers=0, synth_per_epoch=0.5,
)

REGISTRY = {c.name: c for c in (BASE, V1, V2, V3, V4, V5, LOWAUG, SMOKE)}


def get_config(name: str) -> Config:
    if name not in REGISTRY:
        raise KeyError(f"unknown variant {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name].clone()


def describe() -> str:
    lines = []
    for name, c in REGISTRY.items():
        lines.append(f"{name:10s}  {c.notes}")
    return "\n\n".join(lines)
