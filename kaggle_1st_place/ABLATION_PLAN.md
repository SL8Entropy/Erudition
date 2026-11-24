# Ablation plan for the TVT paper (`TVT_IEEE.pdf`)

**Model under test.** Teacher: ConvNeXt-small U-Net trained with the slope loss.
Deployed model: FastViT-SA12 student distilled from that teacher.
Everything here uses the `kaggle_1st_place` pipeline only. `rogii_champion` results stay out.

## Protocol (same for every arm)

- Fixed sorted 80/20 split: train 618 wells, score 155 holdout wells (754,122 rows).
- 150 epochs, batch 4 × accumulation 4, `--num-workers 1`.
- **Seeds 1 and 2 for every arm.** Add seeds 3 and 4 only if a result is unclear.
  Only training randomness changes; the split is fixed.
- Earlier runs used seeds 7 and 11. They stay in `EXPERIMENTS_2026-10.md` as history and do not go in the paper tables.
- **Primary metric:** late-average RMSE (mean holdout RMSE over epochs 100–150), from `seq_NN_honest_curve.py`.
  "Reported" (best-checkpoint) RMSE goes in a secondary column only.
- **A change counts as real** only if:
  - both seeds move in the same direction,
  - the two-seed mean moves by at least 0.10 ft,
  - k\* ≥ 10 and bootstrap ≥ 90% in `seq_NN_robust_compare.py` against the reference arm with the same seed.

  Anything else is reported as "not shown", never as "no effect".

## The four ablations

Every arm needs two runs: `<recipe>_s1` and `<recipe>_s2`. Each run takes about 2 h on laptop D (RTX 5060).

### A. Slope loss (the paper's new loss term)
Question: does the slope term improve the ConvNeXt-small network?

| Arm | Recipe |
|---|---|
| A0 ConvNeXt-small, no slope loss | `small_s1`, `small_s2` |
| A1 ConvNeXt-small + slope loss (**teacher**) | `x_slope_s1`, `x_slope_s2` |

### B. Knowledge distillation (how the deployed model is obtained)
Question: does distillation help FastViT, and does a slope-loss teacher make a better student?

| Arm | Recipe | Teacher |
|---|---|---|
| B0 FastViT, no teacher | `cnx_fastvit_s1`, `_s2` | none |
| B1 FastViT ← plain ConvNeXt-small | `cnx_fastvit_kd_small_s1`, `_s2` | A0 with the same seed |
| B2 FastViT ← slope teacher (**final model**) | `cnx_fastvit_kd_xslope_s1`, `_s2` | A1 with the same seed |

Each student learns from the teacher that has the same seed, so no teacher is picked after seeing results.
B1 and B2 can start only after their teacher run has finished.

### C. Synthetic training wells (85% of training samples)
Question: are the synthetic wells needed? A synthetic well replaces a real sample, so an epoch has the same number of samples; only the share of synthetic wells changes.

| Arm | Change |
|---|---|
| C1 `x_slope_nosynth_s1`, `_s2` | A1 + `sim_cfg['z_shift']['apply_prob'] = 0`; compared with A1 |

### D. Input representation (the mismatch volume and the geo prior)
Question: do the two key engineered channels carry the result?

| Arm | Change |
|---|---|
| D1 `x_slope_nocost_s1`, `_s2` | A1 without `gr_abs_diff` (channel 3); compared with A1 |
| D2 `x_slope_nogeo_s1`, `_s2` | A1 without `geo_tvt_diff` (channel 16); compared with A1 |

**Total: 8 arms × 2 seeds = 16 runs, about 32 GPU-hours.**

## Run order

1. **Teachers first:** `small_s1`, `x_slope_s1`. Everything else depends on these two.
2. **Seed 1 of every other arm:** `cnx_fastvit_s1`, `cnx_fastvit_kd_small_s1`, `cnx_fastvit_kd_xslope_s1`, `x_slope_nosynth_s1`, `x_slope_nocost_s1`, `x_slope_nogeo_s1`.
3. **Seed 2 of everything,** in the same order. Teachers first again.

Stopping after step 2 gives a complete seed-1 table to start writing from.

## Recipes to add in `experiments/bilzard/seq_NN_cfg.py`

All 16 recipes are new; no existing recipe uses seed 1 or 2.

- Build each one from the existing recipe of the same name with `_seeded(..., seed)`.
- Point the teacher paths at `results/<teacher>_s<seed>_ep150`.
- `x_slope_nosynth`, `x_slope_nocost` and `x_slope_nogeo` copy `slope_cfg` exactly and change one thing. The two channel arms use `make_feature_remove_cfg(static_remove=...)` plus the slope settings.

## Machine assignment

| Laptop | Runs |
|---|---|
| D (RTX 5060 8 GB) | both teachers, then every B run (the students need the teacher weights, which stay on D) |
| C (RTX 3050 6 GB, ~5 h/run) | C1 and D1/D2, once `x_slope_s1` has finished on D (C needs a copy of that result as its comparison reference) |
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

| Arm | Late-avg seed 1 | Late-avg seed 2 | Mean | Δ vs reference | k\* / bootstrap | Network ms |
|---|---|---|---|---|---|---|

Supplementary result, no training needed: the decoder comparison already measured on saved posteriors
(mean decode is best: 5.480 vs median 5.636 vs argmax 6.274; `EXPERIMENTS_2026-10.md` §4).
