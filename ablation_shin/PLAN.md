# Ablation plan

**Model under test.** Teacher: ConvNeXt-small U-Net trained with the slope loss.
Deployed model: FastViT-SA12 student distilled from that teacher.
Only this pipeline is used; results from `rogii_champion` or from earlier seed-7/11 runs do not go in these tables.

## Protocol (identical for every run)

- Fixed sorted 80/20 split: train on 618 wells, score 155 holdout wells (754,122 rows). No randomness in the split.
- 150 epochs, batch 4 × gradient accumulation 4, `--num-workers 1`.
- **Seeds 1, 2.** Seeds 3 and 4 only if the decision rule below asks for them.
- **Main number: late-average RMSE**, the mean holdout RMSE over epochs 100–150, from `honest_curve.txt`.
  The "reported" RMSE picks the best epoch using the test wells, so it is optimistic. It is logged as a secondary number only.

## Decision rule

- Δ = late-average(arm) − late-average(reference), both with the same seed. Negative means the arm is better.
- **One seed is enough** when |Δ| ≥ 0.30 ft. That is about 3× the run-to-run noise measured earlier (0.10–0.13 ft).
- **Otherwise run seed 2.** If the two seeds disagree in sign, run seeds 3 and 4.
- **A change counts as real** only if all three hold:
  - every seed run has the same sign,
  - the mean |Δ| is at least 0.10 ft,
  - every seed's compare step shows k\* ≥ 10 and bootstrap ≥ 90%.

  Otherwise write "not shown", never "no effect".
- Ablations A and B always get 2 seeds: they carry the paper's main claims.

## The four ablations

### A. Slope loss: does the new loss term help?
| Arm | Recipe | Reference |
|---|---|---|
| A0 ConvNeXt-small, no slope loss | `small_s{1,2}` | — |
| A1 ConvNeXt-small + slope loss (**the teacher**) | `x_slope_s{1,2}` | A0 |

### B. Distillation: how the deployed model is obtained
| Arm | Recipe | Teacher | Reference |
|---|---|---|---|
| B0 FastViT, no teacher | `cnx_fastvit_s{1,2}` | none | — |
| B1 FastViT ← plain ConvNeXt-small | `cnx_fastvit_kd_small_s{1,2}` | A0, same seed | B0 |
| B2 FastViT ← slope teacher (**the final model**) | `cnx_fastvit_kd_xslope_s{1,2}` | A1, same seed | B0 and B1 |

The student itself has slope-loss weight 0. It learns the slope behaviour only through its teacher, as described in the paper.

### C. Synthetic training wells (85% of training samples)
| Arm | Recipe | Reference |
|---|---|---|
| C1 synthetic wells off (`z_shift.apply_prob = 0`) | `x_slope_nosynth_s{1,2}` | A1 |

A synthetic well replaces a real sample instead of being added, so every epoch still has the same number of samples.

### D. Input representation
| Arm | Recipe | Reference |
|---|---|---|
| D1 without the GR mismatch channel (`gr_abs_diff`, channel 3) | `x_slope_nocost_s{1,2}` | A1 |
| D2 without the geo-prior channel (`geo_tvt_diff`, channel 16) | `x_slope_nogeo_s{1,2}` | A1 |

## Run count

| | Runs |
|---|---|
| Seed 1, all 8 arms | 8 |
| Seed 2 of A and B (5 arms) | 5 |
| **Planned total** | **13** (~26 h on laptop D) |
| Seed 2 of C1 / D1 / D2, only if their seed-1 \|Δ\| < 0.30 ft | 0–3 |

## Where it runs

**Everything runs on laptop D (RTX 5060 8 GB, ~2 h per run).** It has the cached weights, and the students need the teacher weights it produces.

Laptop C (RTX 3050 6 GB, ~5 h per run) can take C1/D1/D2 if time runs short. First copy D's whole
`results\x_slope_s1_ep150\` folder into C's `pipeline\results\`, because it is their reference. The 4 GB laptops are not used:
ConvNeXt-small needs ~3.9 GiB at batch 4.

Every laptop must use the same commit of this repo.

## Recipes

They are defined at the end of `make_bilzard_cfgs()` in `pipeline/experiments/bilzard/seq_NN_cfg.py`. Each one was checked:

| Recipe | Differs from its reference only in |
|---|---|
| `small_s*` | `seed` (vs the archived 0801_V2 recipe) |
| `x_slope_s*` | `seed` |
| `cnx_fastvit_s*` | `seed` |
| `cnx_fastvit_kd_small_s*`, `cnx_fastvit_kd_xslope_s*` | `seed` and the teacher path |
| `x_slope_nosynth_s*` | `seed` and `z_shift.apply_prob` |
| `x_slope_nocost_s*`, `x_slope_nogeo_s*` | `seed` and one removed input channel (16 → 15) |
