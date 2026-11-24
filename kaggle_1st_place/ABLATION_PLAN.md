# Ablation plan for the TVT paper (`TVT_IEEE.pdf`)

**Model under test.** Teacher: ConvNeXt-small U-Net trained with the slope loss (`x_slope`).
Deployed model: FastViT-SA12 student distilled from that teacher (`cnx_fastvit_kd_xslope`).
Everything here uses the `kaggle_1st_place` pipeline only. `rogii_champion` results stay out.

## Protocol (same for every arm)

- Fixed sorted 80/20 split: train 618 wells, score 155 holdout wells (754,122 rows).
- 150 epochs, batch 4 × accumulation 4, `--num-workers 1`.
- Two seeds per arm, **7 and 11**. Only training randomness changes; the split is fixed.
- **Primary metric:** late-average RMSE (mean holdout RMSE over epochs 100–150), from `seq_NN_honest_curve.py`.
  "Reported" (best-checkpoint) RMSE goes in a secondary column only.
- **A change counts as real** only if:
  - both seeds move in the same direction,
  - the two-seed mean moves by at least 0.10 ft,
  - k\* ≥ 10 and bootstrap ≥ 90% in `seq_NN_robust_compare.py` against the reference arm with the same seed.

  Anything else is reported as "not shown", never as "no effect".

## The four ablations

✅ = already run. 🆕 = to run. Each run is ~2 h on laptop D (RTX 5060).

### A. Slope loss (the paper's new loss term)
Question: does the slope term improve the ConvNeXt-small network?

| Arm | Recipe | seed 7 | seed 11 |
|---|---|---|---|
| A0 no slope loss (control) | `0801_V2` / `small_s11` | ✅ `0801_V2_ep150_local` | 🆕 `small_s11` |
| A1 with slope loss (teacher) | `x_slope` / `x_slope_s11` | ✅ | ✅ |

New runs: **1**.

### B. Knowledge distillation (how the deployed model is obtained)
Question: does distillation help FastViT, and does a slope-loss teacher make a better student?

| Arm | Recipe | seed 7 | seed 11 |
|---|---|---|---|
| B0 FastViT, no teacher | `cnx_fastvit` | ✅ (`cnx_fastvit_ep150`, older GPU) | 🆕 `cnx_fastvit_s11` |
| B1 FastViT ← plain ConvNeXt-small | `cnx_fastvit_kd_small_local` | ✅ | 🆕 `cnx_fastvit_kd_small_local_s11` |
| B2 FastViT ← slope teacher (**final model**) | `cnx_fastvit_kd_xslope` | ✅ | 🆕 `cnx_fastvit_kd_xslope_s11` |

The teacher stays fixed across seeds: `0801_V2_ep150_local` for B1, and `x_slope_s11_ep150` for B2.
New runs: **3**. Run these on laptop D, because the teacher weights (`models.pkl`) exist only there.

### C. Synthetic training wells (85% of training samples)
Question: are the synthetic wells needed? A synthetic well replaces a real sample, so an epoch has the same number of samples; only the share of synthetic wells changes.

| Arm | Change | seed 7 | seed 11 |
|---|---|---|---|
| C0 reference | `x_slope` / `x_slope_s11` | ✅ | ✅ |
| C1 no synthetic wells | slope recipe + `sim_cfg['z_shift']['apply_prob'] = 0` | 🆕 `x_slope_nosynth` | 🆕 `x_slope_nosynth_s11` |

New runs: **2**.

### D. Input representation (the mismatch volume and the geo prior)
Question: do the two key engineered channels carry the result?

| Arm | Change | seed 7 | seed 11 |
|---|---|---|---|
| D0 reference | `x_slope` / `x_slope_s11` | ✅ | ✅ |
| D1 no mismatch (cost-volume) channel | slope recipe without `gr_abs_diff` (channel 3) | 🆕 `x_slope_nocost` | 🆕 `x_slope_nocost_s11` |
| D2 no geo-prior channel | slope recipe without `geo_tvt_diff` (channel 16) | 🆕 `x_slope_nogeo` | 🆕 `x_slope_nogeo_s11` |

New runs: **4**. Build these with `make_feature_remove_cfg(static_remove=...)` plus the slope settings.

**Total new runs: 10** (about 20 GPU-hours on laptop D).

## Recipes to add in `experiments/bilzard/seq_NN_cfg.py`

Copy `slope_cfg` exactly, change one thing, and add a seed-11 twin of each:
`cnx_fastvit_kd_small_local_s11`, `cnx_fastvit_kd_xslope_s11`, `x_slope_nosynth(_s11)`,
`x_slope_nocost(_s11)`, `x_slope_nogeo(_s11)`.

`small_s11` and `cnx_fastvit_s11` already exist.

Do not use the existing `cnx_fastvit_kd_small_s11` recipe. Its teacher, `results/0801_V2_ep150`, has no weights on this machine.

## Machine assignment

| Laptop | Runs |
|---|---|
| D (RTX 5060 8 GB) | the three B runs, `small_s11`, D1 ×2 (≈12 h) |
| C (RTX 3050 6 GB, ~5 h/run) | C1 ×2 and D2 ×2 if time allows; otherwise D takes D2 |
| A, B (4 GB) | ConvNeXt-small peaks at ~3.9 GiB at batch 4, so it probably will not fit. Run a 1-epoch fit test first; if it fits, take D2. |

All laptops must run the same commit of `main`.

## Commands (cmd.exe, from `kaggle_1st_place\solution`)

```bat
python seq_NN_holdout_eval.py --id 0801_V2 --source-dir experiments/bilzard --cfg-name <recipe> --epochs 150 --output-dir results/<recipe>_ep150 --device cuda --offline-timm --batch-size 4 --val-batch-size 4 --grad-accum-steps 4 --num-workers 1
python seq_NN_honest_curve.py results/<reference>_ep150 results/<recipe>_ep150
python seq_NN_robust_compare.py --base results/<reference>_ep150 --treat results/<recipe>_ep150 --output-dir results/<recipe>_ep150
```

To run several recipes back to back: `bash analysis/run_chain.sh <recipe> <recipe> ...`.

## Paper table (one row per arm)

| Arm | Late-avg seed 7 | Late-avg seed 11 | Mean | Δ vs reference | k\* / bootstrap | Network ms |
|---|---|---|---|---|---|---|

Supplementary result, no training needed: the decoder comparison already measured on saved posteriors
(mean decode is best: 5.480 vs median 5.636 vs argmax 6.274; `EXPERIMENTS_2026-10.md` §4).
