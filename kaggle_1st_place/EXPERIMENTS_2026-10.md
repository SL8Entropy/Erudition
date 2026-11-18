# Experiment log: training round, 28 Sep – 10 Oct 2026

This is the full record of the round run on the RTX 5060 laptop after the 23 September report.
It covers the literature review, the CPU checks on saved posteriors, the reproduction runs, and
four training rounds. Every number can be regenerated from `kaggle_1st_place/solution/results/`
with the commands at the end.

## 1. Summary

- **Two training changes each lowered the honest error of the 1st-place network by about 0.26 ft,
  and both held on a second seed.** A supervised *slope loss* (new code) and *deep supervision*
  (an existing option that was switched off). Late-average RMSE over two seeds: slope 4.849,
  deep supervision 4.854, controls 5.112.
- **They do not combine.** Together they score no better than the controls, with or without
  restricting the slope loss to the final output.
- **A FastViT student taught by the best slope-loss model ties the plain 1st-place network at
  half the network time** (4.9762 vs 4.9746 reported; late-average 5.053 vs 5.062), and beats the
  student taught by a plain teacher by 0.20–0.26 ft. It stays ~0.25 ft behind its own teacher.
- **The September headline student (4.84) did not reproduce.** Retrained on this machine with a
  freshly retrained teacher it scored 5.18 reported / 5.31 late-average, worse than its teacher.
- **Decoding and calibration are exhausted.** The mean decode is already the best decoder for RMSE
  and the posterior is calibrated. Remaining error has to be removed in training.
- Decoy injection, a CRPS loss, real-only fine-tuning, sibling-well GR references and a
  second-order continuity decode did not show a reliable gain.

## 2. How to read the numbers

All figures are pooled RMSE in feet over the 754,122 suffix rows of the 155 holdout wells
(the fixed sorted 80/20 split in `seq_NN_holdout_eval.py`). Lower is better.

| term | meaning |
|---|---|
| **reported** | What the training script prints. It keeps the checkpoint with the best holdout score, so it is optimistic by 0.06–0.19 ft depending on the run. Includes Savitzky-Golay smoothing. |
| **final** | Raw RMSE of the last checkpoint (epoch 150). No selection on the test wells. |
| **late average** | Mean raw RMSE of the evaluations from epoch 100 to 150. No selection on the test wells. **The main honest number.** |
| **k\*** | From `seq_NN_robust_compare.py`: how many of the most helpful wells must be removed before the gain disappears. Team bar: 10. A no-change rerun once reached k\*=21, so k\* alone is not proof. |
| **bootstrap** | Share of 2,000 well resamples in which the change is better. |
| **controls** | Two runs of the unchanged recipe: `0801_V2_ep150` (team, seed 7, RTX 3050) and `0801_V2_ep150_local` (retrained on the 5060). Reported 4.9785 / 4.9746. |

`final` and `late average` come from `seq_NN_honest_curve.py`.

## 3. Literature and competition review

31 topics, each researched by a separate agent and written to
`wellbore-ml-gaps/results/*.json`; compiled report `wellbore-ml-gaps/report.md`
(regenerate with `python wellbore-ml-gaps/generate_report.py`). Topics: manual geosteering
practice, expert studies, DTW, DP patents, particle filters, ensemble Kalman, RL geosteering,
deep-learning geosteering, multi-realization interpretation, Siamese depth matching, well-log
foundation models, multi-well RGT correlation, the 3rd/6th/8th/10th/14th/22nd/silver write-ups,
stereo matching, multimodal decoding, multi-hypothesis heads, differentiable DP and particle
filters, seismic RGT, retinal-layer segmentation, calibration, test-time adaptation, synthetic
data, BiFPN.

Findings that shaped the experiments:

- Three top teams (3rd, 6th, 22nd) credit **sibling-well GR references** for their most
  decorrelated evidence. Tested in §4: not supported for this model.
- 8th place broke a plateau with **decoy matches at ±15 ft** in synthetic wells. Tested in §6.
- Most error is **one straight-line drift per well** (removing a per-well line would take 4.94 to
  3.07 ft). Led to the slope loss.
- **TVT + Z − ANCC is constant per well** (std 0.0066 ft on all 773 wells), but test files have
  no formation columns, so it cannot be used at prediction time.
- **Mahto 2026** (Research Square preprint, cited in `literature review.ods`) gets its 0.59 ft
  headline from the train-only formation columns. Do not cite it as evidence.
- **BiFPN**: no evidence for full-resolution dense regression; the bottom-up pass feeds levels no
  head reads. Not pursued.

## 4. CPU checks on saved posteriors

Dumps made with `seq_NN_posterior_dump.py --save-all` (per-column 400-level probability maps for
all 155 wells). Scripts in `solution/analysis/`.

**Decoders** (60-epoch baseline dump, in-sample, so optimistic for the alternatives):
mean 5.480, best temperature 5.478, top-64 5.517, median 5.636, dominant mode 6.061,
argmax 6.274. The mean decode is the right decoder for RMSE.

**Calibration**: truth = −0.19 + 0.993 × mean; predicted variance 30.5 ft² against actual
squared error 29.9. Cross-validated recalibration of band weights gains 0.017 ft. Posterior
spread ranks per-well error (Spearman 0.64), useful for flagging wells, not for lowering RMSE.

**Sibling-well GR reference** (`probe_sibling*.py`): GR matching picks the true hump ~50% of
the time (chance 46.5%, dominant hump 76.2%). On multimodal columns the true path fits better
than the wrong hump's path 60.5% of the time with the typewell and 58% with siblings. Siblings
add no band information for this model.

**Second-order continuity decode** (`probe_dipdecode.py`): state = (TVT level, dip of TVT+Z),
emissions tempered to p^τ, dip prior from the known prefix. Explains why the team's
`posterior_fb_decode.py` failed: it is first-order and multiplies per-column posteriors that
already share evidence. Untempered (τ=1) the fix is also worse than the mean (5.84 vs 5.47).

| dump | mean | cross-fitted 2nd-order | k\* | bootstrap |
|---|---|---|---|---|
| 60-epoch baseline | 5.474 | 5.295 ± 0.149 | 3 | 93% |
| FastViT student (local) | 5.220 | 5.047 ± 0.066 | 4 | 97% |
| ConvNeXt-small 150 ep (local) | 5.003 | 4.975 ± 0.043 | 2 | 81% |

Helps weak models, not the strong one. Not adopted.

## 5. Reproduction on the RTX 5060

| run | reported | final | late avg |
|---|---|---|---|
| ConvNeXt-small, team (`0801_V2_ep150`) | 4.9785 | 5.091 | 5.163 |
| ConvNeXt-small, retrained (`0801_V2_ep150_local`) | **4.9746** | 5.017 | 5.062 |
| FastViT student, team (`cnx_fastvit_kd_small_ep150`) | 4.8402 | 4.924 | 4.968 |
| FastViT student, retrained (`cnx_fastvit_kd_small_local_ep150`) | **5.1787** | 5.198 | 5.310 |

The teacher reproduces. The student does not: the retrained one is worse than its teacher. Recipe
`cnx_fastvit_kd_small_local` is the student recipe with the teacher path pointed at the local teacher.

## 6. Training rounds

All ConvNeXt-small, 150 epochs, one change per run, `experiments/bilzard`, batch 4 × accum 4,
`--num-workers 1`.

### Round 2 (8–9 Oct): five single changes, seed 7

| recipe | change | reported | final | late avg | k\* (orig/local) | bootstrap | verdict |
|---|---|---|---|---|---|---|---|
| `x_deepsup` | deep supervision on (`deep_sup=True`, decay 0.5) | 4.7469 | 4.783 | 4.871 | 6 / 7 | 92% / 94% | gain |
| `x_slope` | slope loss, windows 16/64/128, weight 5× regression | 4.8061 | 4.836 | 4.914 | 5 / 35 | 90% / 88% | gain |
| `x_decoy` | decoy GR segments in 25% of synthetic wells | 4.8056 | 5.002 | 4.992 | 2 / 8 | 79% / 89% | not shown |
| `x_crps` | CDF (ranked-probability) loss added at alignment/4 | 4.9473 | 5.018 | 5.043 | – | – | no effect |
| `x_ft_real` | 10 extra epochs on real wells only | 4.8761 | 5.207* | 5.058 | – / 20 | 87% | worse |

\* `x_ft_real`: final after the 10 finetune epochs. Its validation rose every finetune epoch
(5.08 → 5.21); the reported 4.876 and the k\*=20 come from scoring the first finetune epoch.

Decoys match the slope loss on the reported score but not on the honest scores: the gain is
checkpoint-picking luck resting on 2–8 wells.

### Round 3 (9–10 Oct): second seeds and a combination

| recipe | reported | final | late avg | k\* (orig/local) | bootstrap |
|---|---|---|---|---|---|
| `x_deepsup_s11` | 4.7224 | 4.898 | 4.836 | 3 / 19 | 92% / 92% |
| `x_slope_s11` | **4.6692** | **4.752** | **4.783** | 4 / 22 | 95% / 96% |
| `x_slope_deepsup` | 4.9430 | 5.202 | 5.178 | 3 / 6 | 58% / 57% |

Both winners replicate. The combination is worse than either alone.

### Round 4 (10 Oct): fixing the combination, distilling the winner

| recipe | change | reported | final | late avg | notes |
|---|---|---|---|---|---|
| `x_slope_deepsup_final` | combination with `slope_final_only=True` | 4.9353 | 5.031 | 5.128 | vs local control: −0.04, bootstrap 62%. No gain. |
| `cnx_fastvit_kd_xslope` | FastViT student taught by `x_slope_s11` | 4.9762 | 5.001 | 5.053 | vs previous student: −0.20 pooled, k\*=41, bootstrap 85%. Ties the plain ConvNeXt-small. |

### Two-seed summary (late average)

| | seed 7 | seed 11 | mean | vs controls (5.112) |
|---|---|---|---|---|
| slope loss | 4.914 | 4.783 | 4.849 | −0.26 |
| deep supervision | 4.871 | 4.836 | 4.854 | −0.26 |

Against the original control every k\* is below 10 while the bootstrap is 88–96%. Fair wording:
probable, replicated on two seeds. A third seed would strengthen it.

## 7. Code changes

All in `kaggle_1st_place/solution/experiments/bilzard/`. With every new option at its default,
training batches are bit-identical to the previous commit (120 arrays compared with
`analysis/gen_items.py`).

**`seq_NN_train.py`**
- `masked_multiscale_slope_huber_loss(pred, target, target_mask, cfg, ...)`: Huber loss on
  `(pred[t+w] − pred[t]) − (target[t+w] − target[t])` for `w` in `cfg.slope_windows`, averaged
  over windows. Exactly 0 when pred = target and blind to constant offsets (unit-tested).
- New loss weight key `slope` in `unet_loss_weights`; wired into `_unet_loss_single`.
- `cfg.slope_final_only`: skips the slope term on deep-supervision stages (`is_deep_sup_stage`).

**`seq_NN_dataset.py`**
- `Simulator._inject_decoys` and `_decoy_period`: inside `z_shift`, with probability
  `sim_cfg['z_shift']['decoy_prob']`, 1–2 suffix stretches of 200–800 rows get GR read from the
  typewell at `TVT_sim ± P`, where P is the typewell's strongest autocorrelation lag between 10
  and 40 ft (fallback 15 ft), blended over 16 rows. Labels unchanged. The probability check runs
  first, so `decoy_prob=0` draws no random numbers.

**`seq_NN_cfg.py`**
- `"slope": 0.0` added to the default `unet_loss_weights` (the validator rejects unknown keys).
- Recipes: `x_decoy`, `x_slope`, `x_crps`, `x_deepsup`, `x_ft_real`, `x_slope_deepsup`,
  `x_deepsup_s11`, `x_slope_s11`, `x_slope_deepsup_final`, `cnx_fastvit_kd_small_local`,
  `cnx_fastvit_kd_xslope`.

Design notes:
- The slope loss was first planned at windows 4/16/64 and 1× the regression weight. A unit test
  on a synthetic 16 ft drift showed that term ~28× smaller than the regression term, so it was
  raised to 16/64/128 at 5×. It is still ~10% of the total loss (alignment dominates).
- The planned "stretch/squeeze channel" run was dropped: `tw_gr_m0p25` / `tw_gr_p0p25` are the
  typewell shifted by ±0.25 ft, not stretch channels.

## 8. Corrections to earlier notes

- `z_shift.apply_prob` is **0.85** in every run here: the `CFG` class default is 0.75, but the 0801_V2 recipe
  overrides it (`make_seq_requested_ablation_cfgs`), and the run logs confirm 0.85. (An earlier version of this
  log wrongly "corrected" it to 0.75.)
- "The decoder holds ~0.05M parameters" is true only for the full-resolution parts; the three
  mid up-blocks hold ~4.07M (~10 GMAC).
- US 11480045 is assigned to RNA Capital, not Chevron.
- The 4.84 student is one favourable seed (§5).

## 9. Reproducing

From `kaggle_1st_place/solution`:

```bash
python seq_NN_holdout_eval.py --id 0801_V2 --source-dir experiments/bilzard --cfg-name x_slope_s11 --epochs 150 --output-dir results/x_slope_s11_ep150 --device cuda --offline-timm --batch-size 4 --val-batch-size 4 --grad-accum-steps 4 --num-workers 1
python seq_NN_honest_curve.py results/0801_V2_ep150 results/0801_V2_ep150_local results/x_slope_s11_ep150
python seq_NN_robust_compare.py --base results/0801_V2_ep150_local --treat results/x_slope_s11_ep150 --output-dir results/x_slope_s11_ep150
```

Several recipes in a row: `bash analysis/run_chain.sh x_slope x_deepsup`.
Round summaries: `results/round2_summary.txt`, `round3_summary.txt`, `round4_summary.txt`.
Model weights (`models.pkl`) are git-ignored and exist only on the machine that trained them.

## 10. Machine notes

- 16 GB RAM: use `--num-workers 1`. Four workers ran out of commit memory ("paging file is too
  small").
- RTX 5060: ~40 s per ConvNeXt-small epoch, ~1.8–2.2 h per 150-epoch run; FastViT ~1.6 h.
- Windows idle sleep (5 min on AC) and an overnight update reboot stopped runs. For overnight
  chains: plugged in, lid open, `powercfg /change standby-timeout-ac 0`, Windows Update paused.

## 11. Next steps

1. Third seeds of `x_slope` and `x_deepsup`, to clear k\* ≥ 10 against both controls.
2. Two or three more seeds of the FastViT student before any "matches its teacher" claim.
3. Distil from a deep-supervision teacher, and from a two-teacher mix, to close the student's
   0.25 ft gap to its teacher.
