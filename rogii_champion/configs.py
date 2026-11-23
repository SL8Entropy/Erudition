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


# ====================================================================== #
# ABLATION CONFIGS                                                        #
# ====================================================================== #

# -- Group 1: Feature channel ablations -------------------------------- #
ABL_NO_PF = BASE.clone(
    name="abl_no_pf",
    notes="Ablation: no particle-filter channels.",
    use_pf=False,
)
ABL_NO_XY = BASE.clone(
    name="abl_no_xy",
    notes="Ablation: no XY-neighbour channels.",
    use_xy=False,
)
ABL_CORE = BASE.clone(
    name="abl_core_only",
    notes="Ablation: core GR match only (no PF, no XY).",
    use_pf=False, use_xy=False,
)
ABL_NO_PF_CORRUPT = BASE.clone(
    name="abl_no_pf_corrupt",
    notes="Ablation: PF channels present but never corrupted during training.",
)
ABL_NO_PF_CORRUPT.aug.p_pf_corrupt = 0.0

ABL_TW_CORRECT_ONLY = BASE.clone(
    name="abl_tw_correct_only",
    notes="Ablation: typewell correction without sibling reference channels.",
    correct_typewell=True, use_sibling=False,
)
ABL_V4_NO_SIB = V4.clone(
    name="abl_v4_no_sib",
    notes="Ablation: V4 config but with sibling channels disabled.",
    use_sibling=False,
)

# -- Group 2: Augmentation ablations (drop-one) ------------------------ #
_AUG_NOTES = {
    "p_synth":           "Ablation: no synthetic well generation (1st place's #1 aug).",
    "p_gr_affine":       "Ablation: no GR affine transform on typewell (1st place's #2 aug).",
    "p_reverse":         "Ablation: no backwards traversal.",
    "p_md_stretch":      "Ablation: no MD-axis stretching.",
    "p_level_shift":     "Ablation: no vertical level shifting.",
    "p_level_flip":      "Ablation: no vertical flip (4th place's physics-consistent flip).",
    "p_struct_scale":    "Ablation: no structural slope scaling (2nd place's z_layer aug).",
    "p_gr_noise_shift":  "Ablation: no GR residual circular shift.",
    "p_gr_specaug":      "Ablation: no GR spectral masking.",
    "p_tail_crop":       "Ablation: no random tail cropping.",
    "p_tw_mask":         "Ablation: no typewell gap masking.",
    "p_tw_jitter":       "Ablation: no typewell noise jitter.",
    "p_hw_noise":        "Ablation: no horizontal-well GR noise.",
    "p_prefix_resample": "Ablation: no visible-fraction randomisation.",
    "p_channel_mask":    "Ablation: no random channel dropout at canvas level.",
    "p_pf_corrupt":      "Ablation: no PF/XY corruption (anti-shortcut measure).",
}
for _ak, _note in _AUG_NOTES.items():
    _short = _ak[2:]                     # e.g. "synth", "gr_affine", ...
    _cfg = BASE.clone(name=f"abl_no_{_short}", notes=_note)
    setattr(_cfg.aug, _ak, 0.0)
    if _ak == "p_synth":
        _cfg.synth_per_epoch = 0.0
    REGISTRY[_cfg.name] = _cfg

# No augmentation at all
ABL_NO_AUG = BASE.clone(
    name="abl_no_aug",
    notes="Ablation: all augmentations disabled. Raw data only.",
    synth_per_epoch=0.0,
)
for _ak in _AUG_NOTES:
    setattr(ABL_NO_AUG.aug, _ak, 0.0)

# Double augmentation load
ABL_HI_AUG = BASE.clone(
    name="abl_hi_aug",
    notes="Ablation: 2x augmentation load (all probabilities doubled, capped at 1.0).",
    synth_per_epoch=2.0,
)
for _ak in _AUG_NOTES:
    setattr(ABL_HI_AUG.aug, _ak, min(getattr(ABL_HI_AUG.aug, _ak) * 2.0, 1.0))

# -- Group 3: Loss weight ablations ----------------------------------- #
ABL_NO_HUBER = BASE.clone(
    name="abl_no_huber",
    notes="Ablation: no expected-path Huber loss.",
    w_huber=0.0,
)
ABL_NO_GR = BASE.clone(
    name="abl_no_gr",
    notes="Ablation: no GR penalty.",
    w_gr=0.0,
)
ABL_HI_HUBER = BASE.clone(
    name="abl_hi_huber",
    notes="Ablation: 4x Huber weight (0.32).",
    w_huber=0.32,
)
ABL_HI_GR = BASE.clone(
    name="abl_hi_gr",
    notes="Ablation: 4x GR penalty weight (0.08).",
    w_gr=0.08,
)
ABL_TAU1 = BASE.clone(
    name="abl_tau1",
    notes="Ablation: sharper CE target (tau=1.0 ft).",
    ce_tau=1.0,
)
ABL_TAU4 = BASE.clone(
    name="abl_tau4",
    notes="Ablation: broader CE target (tau=4.0 ft).",
    ce_tau=4.0,
)
ABL_TAU8 = BASE.clone(
    name="abl_tau8",
    notes="Ablation: very broad CE target (tau=8.0 ft).",
    ce_tau=8.0,
)
ABL_HARD_CE = BASE.clone(
    name="abl_hard_ce",
    notes="Ablation: near-hard one-hot CE target (tau=0.01).",
    ce_tau=0.01,
)
ABL_VIS_W0 = BASE.clone(
    name="abl_vis_w0",
    notes="Ablation: equal weight on visible and prediction columns.",
    vis_col_weight=0.0,
)
ABL_VIS_W1 = BASE.clone(
    name="abl_vis_w1",
    notes="Ablation: full weight on visible columns (1.0).",
    vis_col_weight=1.0,
)

# V2-specific loss ablations
ABL_V2_DZ_LO = V2.clone(
    name="abl_v2_dz_lo",
    notes="Ablation: V2 with lower slope head weight (w_dz=0.02).",
    w_dz=0.02,
)
ABL_V2_DZ_HI = V2.clone(
    name="abl_v2_dz_hi",
    notes="Ablation: V2 with higher slope head weight (w_dz=0.15).",
    w_dz=0.15,
)
ABL_V2_NO_INTEG = V2.clone(
    name="abl_v2_no_integ",
    notes="Ablation: V2 without integrated-path supervision.",
    w_dz_integ=0.0,
)

# V3-specific loss ablations
ABL_V3_MOVE_LO = V3.clone(
    name="abl_v3_move_lo",
    notes="Ablation: V3 with lower move head weight (w_move=0.10).",
    w_move=0.10,
)
ABL_V3_MOVE_HI = V3.clone(
    name="abl_v3_move_hi",
    notes="Ablation: V3 with higher move head weight (w_move=0.50).",
    w_move=0.50,
)

# -- Group 4: Canvas resolution ablations ------------------------------ #
ABL_COARSE_TVT = BASE.clone(
    name="abl_coarse_tvt",
    notes="Ablation: coarser TVT axis (200 rows at 1.0 ft = same +/-100 ft window).",
    n_rows=200, row_ft=1.0,
)
ABL_NARROW_WIN = BASE.clone(
    name="abl_narrow_win",
    notes="Ablation: narrower TVT window (+/-50 ft instead of +/-100 ft).",
    n_rows=200, row_ft=0.5,
)
ABL_COARSE_MD = BASE.clone(
    name="abl_coarse_md",
    notes="Ablation: coarser MD axis (64 ft stride, half the columns).",
    md_stride=64.0, n_tgt_cols=160,
)
ABL_FINE_MD = BASE.clone(
    name="abl_fine_md",
    notes="Ablation: finer MD axis (16 ft stride, matches 2nd place r16). Needs batch_size=2.",
    md_stride=16.0, n_tgt_cols=640, batch_size=2, accum_steps=2,
)
ABL_SHORT_VIS = BASE.clone(
    name="abl_short_vis",
    notes="Ablation: shorter visible region (16 cols = 512 ft).",
    n_vis_cols=16,
)
ABL_LONG_VIS = BASE.clone(
    name="abl_long_vis",
    notes="Ablation: longer visible region (64 cols = 2048 ft).",
    n_vis_cols=64,
)

# -- Group 5: Backbone / architecture ablations ------------------------ #
ABL_NO_PT = BASE.clone(
    name="abl_no_pretrain",
    notes="Ablation: random init (no ImageNet pretraining).",
    pretrained=False,
)
ABL_CNX_TINY = BASE.clone(
    name="abl_cnx_tiny",
    notes="Ablation: ConvNeXt-Tiny backbone (smaller model).",
    backbone="convnext_tiny.in12k_ft_in1k_384",
)
ABL_CNX_BASE = BASE.clone(
    name="abl_cnx_base",
    notes="Ablation: ConvNeXt-Base backbone (larger model). May need batch_size=2.",
    backbone="convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384",
    batch_size=2, accum_steps=2,
)
ABL_EFFNET = BASE.clone(
    name="abl_effnetv2s",
    notes="Ablation: EfficientNetV2-S backbone (different architecture family).",
    backbone="tf_efficientnetv2_s.in21k_ft_in1k",
)
ABL_GRAD_CK = BASE.clone(
    name="abl_grad_ckpt",
    notes="Ablation: gradient checkpointing (trade compute for VRAM).",
    grad_checkpoint=True,
)

# -- Group 7: Training recipe ablations -------------------------------- #
ABL_LR_LO = BASE.clone(
    name="abl_lr_1e4",
    notes="Ablation: lower learning rate (1e-4).",
    lr=1e-4,
)
ABL_LR_HI = BASE.clone(
    name="abl_lr_1e3",
    notes="Ablation: higher learning rate (1e-3).",
    lr=1e-3,
)
ABL_NO_EMA = BASE.clone(
    name="abl_no_ema",
    notes="Ablation: no exponential moving average of weights.",
    ema_decay=0.0,
)
ABL_EMA_HI = BASE.clone(
    name="abl_ema_hi",
    notes="Ablation: slower EMA decay (0.9999).",
    ema_decay=0.9999,
)
ABL_NO_WD = BASE.clone(
    name="abl_no_wd",
    notes="Ablation: no weight decay.",
    weight_decay=0.0,
)
ABL_HI_WD = BASE.clone(
    name="abl_hi_wd",
    notes="Ablation: higher weight decay (0.05).",
    weight_decay=0.05,
)
ABL_NO_WARMUP = BASE.clone(
    name="abl_no_warmup",
    notes="Ablation: no learning rate warmup.",
    warmup_frac=0.0,
)
ABL_LONG_WARMUP = BASE.clone(
    name="abl_long_warmup",
    notes="Ablation: longer warmup (20% of training).",
    warmup_frac=0.2,
)
ABL_CLIP_LO = BASE.clone(
    name="abl_clip_lo",
    notes="Ablation: tighter gradient clipping (0.5).",
    grad_clip=0.5,
)
ABL_CLIP_HI = BASE.clone(
    name="abl_clip_hi",
    notes="Ablation: looser gradient clipping (5.0).",
    grad_clip=5.0,
)
ABL_NO_CLIP = BASE.clone(
    name="abl_no_clip",
    notes="Ablation: no gradient clipping.",
    grad_clip=0.0,
)
ABL_EP60 = BASE.clone(
    name="abl_ep60",
    notes="Ablation: longer training (60 epochs).",
    epochs=60,
)
ABL_EP100 = BASE.clone(
    name="abl_ep100",
    notes="Ablation: much longer training (100 epochs).",
    epochs=100,
)
ABL_BS2 = BASE.clone(
    name="abl_bs2",
    notes="Ablation: smaller batch size (2) with 2x accumulation.",
    batch_size=2, accum_steps=2,
)
ABL_BS8 = BASE.clone(
    name="abl_bs8",
    notes="Ablation: larger batch size (8).",
    batch_size=8,
)

# -- Register all named ablation configs ------------------------------- #
for _c in (
    ABL_NO_PF, ABL_NO_XY, ABL_CORE, ABL_NO_PF_CORRUPT, ABL_TW_CORRECT_ONLY,
    ABL_V4_NO_SIB,
    ABL_NO_AUG, ABL_HI_AUG,
    ABL_NO_HUBER, ABL_NO_GR, ABL_HI_HUBER, ABL_HI_GR,
    ABL_TAU1, ABL_TAU4, ABL_TAU8, ABL_HARD_CE,
    ABL_VIS_W0, ABL_VIS_W1,
    ABL_V2_DZ_LO, ABL_V2_DZ_HI, ABL_V2_NO_INTEG,
    ABL_V3_MOVE_LO, ABL_V3_MOVE_HI,
    ABL_COARSE_TVT, ABL_NARROW_WIN, ABL_COARSE_MD, ABL_FINE_MD,
    ABL_SHORT_VIS, ABL_LONG_VIS,
    ABL_NO_PT, ABL_CNX_TINY, ABL_CNX_BASE, ABL_EFFNET, ABL_GRAD_CK,
    ABL_LR_LO, ABL_LR_HI, ABL_NO_EMA, ABL_EMA_HI,
    ABL_NO_WD, ABL_HI_WD, ABL_NO_WARMUP, ABL_LONG_WARMUP,
    ABL_CLIP_LO, ABL_CLIP_HI, ABL_NO_CLIP,
    ABL_EP60, ABL_EP100, ABL_BS2, ABL_BS8,
):
    REGISTRY[_c.name] = _c


def get_config(name: str) -> Config:
    if name not in REGISTRY:
        raise KeyError(f"unknown variant {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name].clone()


def describe() -> str:
    lines = []
    for name, c in REGISTRY.items():
        lines.append(f"{name:10s}  {c.notes}")
    return "\n\n".join(lines)
