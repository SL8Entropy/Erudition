# Erudition — Project Report

**Date:** 22 September 2026
**Covers:** all work in `kaggle_1st_place/` and `kaggle2ndplace/` up to this date.
**Replaces:** `Erudition_Report_2026-09-21.md`.
**Short version for learners:** `Erudition_Study_Notes_2026-09-22.md`.

---

## 0. How to read the numbers in this report

Every number in this report means one of the following, and nothing else.

**Error (ft).** Root-mean-square error, in feet, over all 754,122 predicted points on the 155 test
wells. Lower is better. RMSE counts big mistakes more than small ones. Every "error" in this report
is this number unless the row says otherwise.

**Reported score vs unpicked score.** The 1st-place training code checks itself on the 155 test
wells during training and keeps the checkpoint that scored best there. That makes its **reported
score** too good. We measured by how much, as the reported score minus the late average (below):
0.14–0.15 ft for every ConvNeXt run, 0.07 ft for FastViT alone, 0.15 ft for FastViT taught by
tiny, and 0.10 ft for FastViT taught by small. Two honest alternatives are read from the training
log:
- **last checkpoint:** the model at the end of training, which nobody chose;
- **late average:** the average of all evaluations from round 100 to round 150.

Comparing two reported scores is fair, because both were picked the same way. The absolute
reported numbers are slightly flattering. AnchorCNN numbers are always last-checkpoint numbers.

**Verdicts.** Every comparison gets exactly one of these labels:

| label | meaning | the test behind it |
|---|---|---|
| ✅ **Confirmed** | The difference is real. | The gain survives removing the 10 test wells that help it most (k\* ≥ 10) **and** at least 90% of 2,000 random re-samples of the test wells agree. Older experiments that were not run through these two tests get ✅ only when the evidence in their row is as strong: for example, the same direction on every repeat, or on nearly all 155 wells. |
| 🟡 **Probable** | Likely real, but not certain. | k\* ≥ 10, but only 70–90% of re-samples agree. |
| ⚪ **Not shown** | We can't tell it from luck. | The gain disappears after removing fewer than 10 wells, or it is smaller than the 0.39 ft noise of a single comparison. This does **not** mean "no effect". It means one run on 155 wells can't see it. |
| ❌ **Confirmed worse** | The change hurts. | The same tests, in the other direction. |
| ⚠️ **Invalid test** | The experiment was broken. | A bug or a flawed setup. The result says nothing about the idea. |
| ⏳ **Not run** | Built, or planned, but not trained yet. | — |

**Network time (ms).** The time the network alone needs for one well, measured on 22 September in
one session, under identical conditions: idle laptop GPU (RTX 3050), 16-bit arithmetic, one well
at a time. The same model can run up to 30% faster or slower in another session, because the
laptop changes GPU speed with its power state. So compare network times only within this report.

**Whole-pipeline time (ms).** Everything needed to predict one well: the neighbour-well map, the
input image, the network, and turning the output into depths (§7.4).

**Noise.** Retraining an identical model with a different random seed moves the error by ±0.28 ft
(standard deviation of 3 AnchorCNN seeds). So a comparison of two single runs carries ±0.39 ft of
luck (√2 × 0.28). Differences smaller than that can't be detected by one pair of runs.

---

## 1. Summary

1. **Best single model: FastViT-SA12 taught by ConvNeXt-small. Error 4.83 ft, network time 13.1 ms.**
   - "Taught" means knowledge distillation: the small FastViT network is trained to copy a bigger
     network's predictions (§5.7).
   - It is ✅ confirmed better than FastViT trained alone (5.21 ft).
   - It is ⚪ not distinguishable from its teacher ConvNeXt-small (4.98 ft), nor from ConvNeXt-tiny
     (4.90 ft). The honest claim is "as accurate as a teacher with 4.4× its parameters and 2× its
     network time".
   - One seed only.
2. **Best combination: the average of ConvNeXt-tiny and that taught FastViT. Error 4.69 ft, network time 31.5 ms** (18.4 + 13.1).
   ✅ Confirmed better than ConvNeXt-tiny alone (4.90 ft).
3. **Recommendation for a drilling rig:** the taught FastViT (4.83 ft, 13.1 ms), once a second
   seed confirms it (commands in §10.1). Until then, ConvNeXt-tiny (4.90 ft, 18.4 ms).
4. **Whole-pipeline time dropped from 158 ms to 46 ms per well on 22 September,** with
   bit-identical predictions. The CPU work, not the network, had been the bottleneck (§7.4).
5. **Every reported score from the 1st-place code is flattered** by checkpoint picking (§0):
   by 0.14–0.15 ft for the ConvNeXts, and by 0.07–0.15 ft for the FastViTs. Comparisons between
   them are still fair, but read the unpicked columns in §5.5 and §5.7 for absolute numbers.
6. **Most ideas were ⚪ not shown:** their effect was smaller than the 0.39 ft noise. The things
   that clearly worked were:
   - training longer;
   - averaging models of different sizes;
   - distillation;
   - making the AnchorCNN half as expensive at no measurable cost;
   - repairing our particle filter.
7. **The biggest untested gap:** the 2nd-place model (AnchorCNN) has never been trained properly
   with synthetic data (§8 item 2).

---

## 2. The problem and the data

- **The drill.** A drill goes sideways through underground rock layers ("horizontal well") and
  can't see where it is. It knows its drilled length, its depth, and one sensor reading: **gamma
  ray (GR)**, the rock's natural radioactivity. Each layer has its own GR level, so GR works like
  a fingerprint.
- **The reference.** A **typewell** is the GR log of a nearby vertical well, which passes through
  every layer once.
- **The task.** For the first part of each well, the drill's position within the layers is
  known. For the rest, predict that position (**TVT**, in feet) at every point, from the GR, the
  drilling path and the typewell.
- **Why it's hard.** Layers repeat, so a wrong match can fit the GR as well as the right one.
- **The data.** 773 wells from the ROGII Kaggle competition. We train on 618 wells and test on
  155 wells that no model ever trains on (the **holdout**). Both repositories use the same split.

| reference point | error (ft) |
|---|---|
| Assume the drill stays at its last known depth | 9.8 (recorded only to one decimal) |
| The 1st-place model's neighbour-well map alone ("geo prior") | 11.57 |
| Error caused only by rounding answers onto the model's depth grid | 0.06 |

**What kind of machine-learning problem this is.**
- **Dense regression:** predict a continuous number at every point along the well.
- **In image terms, stereo matching** (disparity estimation). The 1st-place network builds a
  "cost volume" of how well each candidate depth matches. Its final step is the soft-argmin used
  by stereo networks: a probability-weighted average over the candidate depths.
- **Path finding:** the 2nd-place model's last step is a shortest-path search (dynamic
  programming) through a cost image.
- **Close relatives in the literature:** stereo disparity, retinal-layer segmentation, and
  seismic horizon tracking.

---

## 3. The models and methods

| name | from | how it works | size |
|---|---|---|---|
| **ConvNeXt U-Net** | 1st place (Ruby) | Turns each well into a 16-layer image (345 positions along the well × 400 candidate depths) and runs an image network shaped as a U-Net. Outputs a probability for every candidate depth at every position. Trains on 85% synthetic (computer-generated) samples. | 53.6 M parameters with ConvNeXt-small |
| **AnchorCNN** | 2nd place (Bilzard) | A small image network predicts the depth *change* at each step (21 options), then a shortest-path search picks the best overall path. | 3.7 M parameters |
| **Particle filter** | ours | No learning. Tracks hundreds of candidate positions step by step, keeping those whose expected GR matches the reading. | — |
| **HMM tracker** | ours | No learning. Exact probabilistic tracking over 129 depth levels. | — |

**Backbone.** The U-Net's main feature extractor, downloaded pre-trained on ImageNet photos. The
1st-place model uses ConvNeXt-small. We swapped in ConvNeXt-tiny, FastViT-SA12 and MobileOne-S1 (§5.5).

**The computer.** Laptop, RTX 3050 GPU (6 GB), 16 GB RAM. One AnchorCNN training run takes 1 hour.
One 150-round ConvNeXt run takes 3–5 hours, depending on the backbone.

---

## 4. How we judged results

Three checks, all run with `seq_NN_robust_compare.py`:

1. **Paired control.** Compare against the same settings without the change, at the same length of training.
2. **k\* (leave-wells-out).** Remove, one at a time, the test wells that contribute most to the
   gain. k\* is how many must go before the gain vanishes. We require k\* ≥ 10. A gain that
   vanishes after 1–3 wells came from a few lucky wells.
3. **Bootstrap.** Re-score on 2,000 random re-samples of the 155 wells, and report the percentage
   in which the change is better.

Why these are needed:
- The worst 1, 3, 10 and 52 test wells carry 8%, 20%, 43% and 86% of all error.
- Luck on single hard wells is large. Identical AnchorCNN settings with different seeds scored
  between 9.1 and 25.2 ft on well `f6d009f4`.
- A model that turned out to be identical to its baseline once "passed" by 0.03 ft, spread over 21 wells.

---

## 5. Results

### 5.1 Every system on the 155 test wells

The error is the reported score (§0). Network times come from the 22 September session. "Merged"
means FastViT's or MobileOne's training branches were folded into single layers for inference;
this changes accuracy by less than 0.02 ft (§7.2).

| system | error (ft) | attention? | network time | verdict / note |
|---|---|---|---|---|
| Assume "no change" | 9.8 | — | 0 | baseline |
| Our particle filter (final) | 12.0 | not a network | CPU tracker | §5.3 |
| 1st-place author's particle filter | 7.35 | not a network | 5.6 s per well (CPU) | §5.3 |
| AnchorCNN, original settings | 6.49 | none | 14 ms + 127 ms path search | §5.2 |
| AnchorCNN, "C" settings, 3 seeds | 6.02 / 6.57 / 6.20 | none | half the compute | §5.2 |
| AnchorCNN + tracker inputs, 2 seeds | 5.84 / 6.11 | none | + tracker on CPU | ⚪ §5.2 |
| ConvNeXt-small, 60 rounds | 5.16 | none | 26.1 ms | |
| **ConvNeXt-small, 150 rounds** (1st-place model) | **4.98** | none | 26.1 ms | reference |
| ConvNeXt-small + 8 shifted views averaged (TTA) | 4.94 | none | 209 ms (8 passes) | ⚪ vs 4.98 |
| **ConvNeXt-tiny, 150 rounds** | **4.90** | none | **18.4 ms** | ⚪ vs small (k\*=4) |
| FastViT-SA12, 150 rounds | 5.21 | last stage only | 12.6 ms merged | ❌ vs tiny |
| MobileOne-S1, 150 rounds | 6.51 | none | 11.9 ms merged | ❌ vs tiny (1.61 ft worse) |
| FastViT-SA12 taught by ConvNeXt-tiny | 4.93 | last stage only | 13.1 ms merged | ✅ vs FastViT alone |
| **FastViT-SA12 taught by ConvNeXt-small** | **4.84 (4.83 merged)** | last stage only | **13.1 ms merged** | ✅ vs FastViT alone; ⚪ vs small and vs tiny |
| Average: ConvNeXt-small + ConvNeXt-tiny | 4.78 | none | 44.5 ms | ✅ vs tiny |
| Average: small + tiny + a synthetic-data variant | 4.77 | none | 70.6 ms | passes k\* (65) |
| Average: ConvNeXt-tiny + FastViT | 4.85 | in FastViT | 31.0 ms | 🟡 vs tiny |
| Average: ConvNeXt-small + FastViT | 4.90 | in FastViT | 38.7 ms | ⚪ vs small (k\*=2) |
| Average: ConvNeXt-tiny + FastViT taught by tiny | 4.77 | in FastViT | 31.5 ms | ✅ vs tiny |
| Average: ConvNeXt-small + FastViT taught by tiny | 4.82 | in FastViT | 39.2 ms | ✅ vs small |
| Average: ConvNeXt-small + FastViT taught by small | 4.75 | in FastViT | 39.2 ms | ✅ vs small |
| **Average: ConvNeXt-tiny + FastViT taught by small** | **4.69** | in FastViT | **31.5 ms** | ✅ vs tiny — **best overall** |
| ConvNeXt + AnchorCNN, one fixed weight | 4.85 | none | 5–6× ConvNeXt alone | ⚪ vs 4.94 (0.085 ft) |

For context: the 1st-place author reported 4.80 ft using 15 models × 300 rounds and a different
test split. It is not directly comparable.

### 5.2 AnchorCNN (2nd-place model) experiments

| experiment | result | verdict |
|---|---|---|
| Rebuilt the training code from scratch | 6.49 ft. Three bugs found and fixed: training windows cut from the curved start of wells (labels up to 1,021 ft), a stuck EMA, a cache overwritten between runs | — |
| Wiring check: the author's own trained weights through our code | 4.03 ft. Not a fair score: those weights had seen our test wells. It proves the pipeline is correct. | — |
| **"C" settings:** coarser depth grid, fewer step options | Compute 15.7 → 7.9 GFLOP; training 56 → 29 min. Error 6.02 / 6.57 / 6.20 over 3 seeds against 6.49 for one original run | ✅ half the cost; ⚪ for accuracy (no loss detected) |
| **TTA:** average predictions over 8 shifted copies | 0.26–0.56 ft better on every run; 8× the cost | ✅ |
| Even coarser grid (8 ft steps) | 1.46 ft worse than C; worse on 124 of 155 wells; 100% of re-samples agree | ❌ |
| Cheaper first layer ("stride-2 stem") | Worse on 145 of 155 wells; median-well error 3.67 → 4.11 ft; unstable training | ❌ |
| Both cheaper changes together | 7.64 ft | ❌ |
| "Separable" (row/column) network design | 11.4 ft against 6.0–6.6 | ❌ |
| 8-bit numbers (int8) | 25.1 ft as is; 10.8 ft after retraining; runs at 0.63–0.76× the speed on this CPU; file 15.3 → 4.8 MB | ❌ |
| Tracker's guess and confidence as 2 extra inputs | Seed 1: 6.02 → 5.84. Seed 2: 6.57 → 6.11. Each gain disappears after removing 1–2 wells | ⚪ (third seed ⏳) |
| Sibling reference as an extra input | 0.25 ft worse, one run, only tried on top of the tracker inputs | ⚪ |
| Average of 3 seeds | 5.99 against 6.02 for the best single seed | ⚪ |
| A smoothing filter instead of TTA | Did not reproduce TTA's gain | ❌ |
| Synthetic data (`C_synth_s1`) | A bug gave 32% synthetic wells instead of 77%, and the run did 2.3× the training steps of its control. Final 6.30. It led the control at 18 evaluations in a row, and averaged 6.11 over the second half against 6.26–6.38, then overfitted | ⚠️ (bug fixed; proper run ⏳) |
| Training on longer stretches ("far-anchor"), version 1 | A bug made every sample from a well identical (window overlap 1.00 against 0.87). The error rose from 6.49 to 7.92 over 250 rounds | ⚠️ (fixed; not re-run) |
| Distillation from a 3-seed ensemble | The teacher (5.99) was no better than the student's own baseline (5.94), so there was nothing to learn | ⚠️ |
| "Twice the training data doesn't help" | The runs differed in both seed and amount of data | ⚠️ |

### 5.3 Trackers (no learning)

| step | error (ft) | what changed |
|---|---|---|
| Our particle filter as first built | 28.98 | It lost track: error grew from 4 ft early in a well to 30 ft at the end |
| Allow the rock's tilt to change 9× faster | 16.82 | Measured best value; the first tuning had moved it the wrong way |
| Remove a steady downward lean | 15.84 | The error leaned the same way in 72% of wells; one correction fitted on training wells |
| Predict each well's own correction | 13.07 | Fitted on training wells; the strongest clue was the drilling path's angle |
| Sibling reference (§5.9) | 12.08 / 11.95 | Two repeats; before: 12.73 / 12.65. ✅ |
| 8× more particles | changed by 0.02 ft | ❌ not the bottleneck |
| **1st-place author's particle filter** | **7.35** (7.90 without smoothing) | 5.6 s per well; uses only information available at prediction time (checked in its code) |
| HMM tracker | about 23 in early tuning | 0.11 s per well; never scored on the test wells ⏳ |

As a blend partner, every tracker received a weight of 0.00–0.06: it adds nothing (§5.6).

### 5.4 ConvNeXt (1st-place model) experiments

| experiment | result | verdict |
|---|---|---|
| Train longer: 60 → 150 rounds | 5.16 → 4.98 | ✅ |
| TTA (8 shifted views) | 4.98 → 4.94. Better in 4 of 4 tries, by 0.02–0.15 ft each | ⚪ |
| More realistic synthetic wells (`rb_v2_synth`) | 0.15 ft worse at 150 rounds, but better on most wells (the gain survives from k=10 to k=105), and still improving when stopped | ⚪ |
| Neighbour-well inputs (`rb_v1_neighbor`) | Far behind at 60 rounds, but undertrained: it was at the level the baseline reached by round 19 | ⚪ |
| **Axial attention** added (`arch_v2_axial`) | −0.024 ft. Two identical runs differ by 0.030 ft from luck alone | ⚪ |
| "Unimodal" and "RAFT" redesigns | Slightly worse at 60 rounds; their own notes call 60 rounds too short | ⚪ |
| "Large-kernel" redesign | Built so it could never learn: all 36 gating weights stayed exactly 0 | ⚠️ (fixed; not re-run) |
| Hidden data constant 0.99904 | Reproduced (0.99904 ± 0.00102), but correcting depths with it made every setting worse (5.00–20.05 against 4.94) | ❌ |
| Predicting each well's error line in advance | No input predicted it, on either network | ❌ |

### 5.5 Replacing the backbone

**Trained.** Unpicked scores are from §0. The test is against ConvNeXt-tiny unless stated.

| backbone | backbone parameters | attention | reported | last checkpoint | late average | network time | verdict |
|---|---|---|---|---|---|---|---|
| ConvNeXt-small | 49.5 M | none | 4.98 | 5.09 | 5.16 | 26.1 ms | ⚪ vs tiny (k\*=4, 67%) |
| **ConvNeXt-tiny** | 27.8 M | none | **4.90** | 5.14 | 5.09 | 18.4 ms | reference |
| FastViT-SA12 | 10.4 M | last stage | 5.21 | 5.30 | 5.32 | 12.6 ms merged | ❌ (94% of re-samples favour tiny) |
| MobileOne-S1 | 3.6 M | none | 6.51 | — | — | 11.9 ms merged | ❌ (1.61 ft worse) |

**The size rule.** From 50 M down to 28 M costs nothing: the data runs out before the model size
does. Below 10 M, accuracy falls sharply. Distillation moves that rule: taught, the 10 M FastViT
matches the 28 M and 50 M ConvNeXts (§5.7).

**Screened but not trained.** Each was built inside the U-Net and timed.

| candidate | fits the U-Net unchanged? | network time | why it was not trained |
|---|---|---|---|
| ConvNeXt-nano | yes | 14.8 ms | not yet: runs built ⏳ |
| InceptionNeXt-tiny | yes | 19.7 ms | not yet: runs built ⏳; only ImageNet-1k weights exist |
| FastViT-SA24 | yes | 17.7 ms merged | not yet: run built ⏳ |
| FastViT-T12 (no attention) | yes | 12.0 ms merged | not yet: run built ⏳ |
| ConvNeXt-base | yes | 36.6 ms | bigger doesn't help (size rule); likely too big to train on 6 GB |
| ConvNeXt-V2 nano / tiny | yes | slower than V1 at the same arithmetic | an extra normalisation layer |
| ConvFormer-S18, CAFormer-S18 | yes | slower than tiny | no advantage |
| FasterNet | yes | 1.8× slower than ConvNeXt-small, despite fewer operations | slower |
| EfficientNet-B3/B4 | no | 9.5 / 11.6 ms against tiny's 7.2 ms in a standalone test | slower |
| MambaOut, MobileNetV4, RepViT, RDNet, Hiera, EdgeNeXt, EfficientViT, TinyViT, HGNetV2, Swin | no | — | each needs adapter code |
| HRNet / Lite-HRNet | no | — | fails at our input size; Lite-HRNet is unavailable |
| DINOv2 | no | — | one coarse 25 × 29 grid; this task needs fine detail |
| FasterViT, CMT | not in the library | — | would need new dependencies |

The 1st-place author also built 10 transformer backbones, and kept ConvNeXt-small in all 6 submissions.

### 5.6 Combining models

**Rule-based combinations** (ConvNeXt + AnchorCNN):

| method | error | verdict |
|---|---|---|
| One fixed weight, cross-fitted (never tuned on the wells it is scored on) | 4.852 | ⚪ against the ConvNeXt's 4.94 |
| The same weight tuned on the test wells | 4.834 | not valid: tuned on its own test |
| A learned "gate" choosing per point | 5.18 against 4.87 for a fixed weight | ❌ |
| The gate plus the 3rd-place team's training rule | 5.01 | ❌ |
| Switch where the two models disagree most | 5.09–5.12 against 4.85 (disagreement predicts error with correlation 0.32) | ❌ |
| Stacking all 17 prediction sets | 4.88–4.95 against 4.85 | ❌ |
| A tracker as a partner | weight 0.00–0.06 | ❌ |

**Averages of two ConvNeXt-family networks** (50/50 average of the predicted depths):

| pair | error | similarity of their errors | vs the better member | verdict |
|---|---|---|---|---|
| small + tiny | 4.78 | 0.870 | −0.13 vs tiny; k\*=67; 91% | ✅ |
| tiny + FastViT | 4.85 | 0.838 | −0.06 vs tiny; k\*=81; 72.5% | 🟡 |
| small + FastViT | 4.90 | 0.854 | −0.07 vs small; k\*=2; 76% | ⚪ |
| tiny + FastViT taught by tiny | 4.77 | 0.879 | −0.14 vs tiny; k\*=31; 90.3% | ✅ |
| small + FastViT taught by tiny | 4.82 | 0.890 | −0.16 vs small; k\*=54; 98.4% | ✅ |
| small + FastViT taught by small | 4.75 | 0.874 | −0.23 vs small; k\*=32; 97.1% | ✅ |
| **tiny + FastViT taught by small** | **4.69** | 0.857 | **−0.21 vs tiny; k\*=19; 98.2%** | ✅ |
| FastViT taught by tiny + FastViT taught by small | 4.80 | 0.935 | not tested | — |

What decides whether a partner helps:
- **Different mistakes help.** The error similarity is the correlation of two models' errors
  (1.0 = identical mistakes). The AnchorCNN (similarity 0.67) was a better partner than other
  ConvNeXt versions (0.84–0.97).
- **Accuracy helps too.** Teaching FastViT raised its similarity to tiny from 0.84 to 0.88, but
  it cut its own error from 5.21 to 4.93. The net effect made it a better partner (4.85 → 4.77).
  Our tracker also became more similar as it improved, but never became accurate enough to help.
- **Simple wins.** One fixed weight beat every learned chooser.

### 5.7 Distillation: teaching a cheap network

**What it is.** A finished, better network (the **teacher**) is frozen. A cheaper network (the
**student**) trains on two targets:
1. the true depths, as normal;
2. the teacher's probability for every one of the 400 candidate depths at every position.

The second target also passes on "this other depth looked plausible too", which matters when
layers repeat. The loss is the KL divergence at temperature 2, with weight 1.0. Neither setting
has been tuned.

**Results** (FastViT-SA12 student in every row):

| teacher | reported | last checkpoint | late average | vs FastViT alone | vs its teacher |
|---|---|---|---|---|---|
| none (FastViT alone) | 5.21 | 5.30 | 5.32 | — | — |
| ConvNeXt-tiny | 4.93 | 5.12 | 5.11 | ✅ −0.28 (k\*=43, 91.1%) | ⚪ +0.03 |
| **ConvNeXt-small** | **4.84** | **4.92** | **4.97** | ✅ −0.37 (k\*=35, 95.2%) | ⚪ −0.14 (k\*=8, 73.2%; better on 77 of 155 wells) |

- The two teachers give students that are ⚪ indistinguishable from each other (k\*=4, 67%).
- The student taught by small is ⚪ indistinguishable from ConvNeXt-tiny (k\*=8, 62.5%).
- The student taught by small had the steadiest training of any run: its evaluations from round
  100 to 150 varied by 0.04 ft (standard deviation), against 0.12 ft for its teacher.
- **Why a student can match its teacher:** the teacher's full probability map is a smoother target
  than the single true depth. It works like a regulariser, which keeps the student from
  memorising 618 wells.
- **An earlier distillation failed** (AnchorCNN, §5.2) because its teacher was no better than
  the student. Every teacher here was measured better first.

**Teachers considered and rejected:**
- **DINOv2:** it would first have to be trained on this task. Its coarse 25 × 29 grid would
  likely make it worse than tiny, and a worse teacher teaches nothing.
- **ConvNeXt-base:** the size rule says bigger doesn't help, and it likely won't train on a 6 GB GPU.

**Built, not yet run** ⏳ (commands in §10.1):
- ConvNeXt-tiny, FastViT and ConvNeXt-nano taught by the small + tiny average;
- InceptionNeXt taught by small;
- FastViT-SA24 and FastViT-T12 taught by small.

### 5.8 Other teams' claims we checked

| claim | whose | our result |
|---|---|---|
| The 773 typewells are crops of 54 master logs | 22nd place | ✅ 54 |
| A sibling reference improves a particle filter by 0.78 ft | 22nd place | ✅ 0.68 ft |
| Hidden constant 0.99904 ± 0.001 | 22nd place | ✅ 0.99904 ± 0.00102, but unusable (they couldn't use it either) |
| About 75% of error lies far from the known part of the well | 22nd place | ✅ 69–70% |
| Most error is one wrong straight line per well | 8th place | ✅ on every model |
| Learned choosers don't transfer; averaging seeds doesn't help | 22nd place | ✅ same here |

### 5.9 Other findings

- **Master logs and siblings.** The 773 typewells overlap exactly within 54 master logs. Wells
  on the same master log drill the same rock, and we call them siblings. Pooling their GR gives
  a better reference: on 77% of test wells it matched the real GR better than the typewell did
  (error units 10.78 → 10.18).
- **Coverage gap.** AnchorCNN training windows reached at most 7,700 ft past the known part of a
  well. Test wells reach 10,100 ft, and 69–70% of the error lies in the far half. The fix
  (far-anchor) exists but hasn't been fairly tested.
- **No leakage.** None of the 155 test wells duplicates a training well.
- **The 1st-place code already had tracker inputs,** switched off. The author's own later versions
  that used them scored worse on the author's tests (5.01 and 5.54 against 4.80), but other
  settings changed too.

---

## 6. Why the score is hard to improve

**A few wells decide it.** The worst 10 of 155 wells carry 43% of all error.

**Most error is one straight line per well.** The prediction drifts from the truth at a steady
angle. Removing that line (which needs the true answer) would cut the error sharply:

| model | as is | minus one offset per well | minus one straight line per well |
|---|---|---|---|
| ConvNeXt | 4.94 | 3.76 | 3.07 |
| AnchorCNN | 6.02 | 4.15 | 3.27 |
| Our tracker | 15.85 | 9.49 | 5.48 |

Nothing we tried predicted the line in advance: the known part of the well, the drilling path,
neighbouring wells' slopes, the model's confidence, and the hidden constant all failed.

**The same six wells defeat every method:** `d7eb0be8`, `f2d4c8c9`, `f6bc699b`, `f6d009f4`,
`f8afa78a`, `fb0904bd`.
- They are in the worst 15 for all four networks compared.
- They carry 20–35% of each network's error.
- The author's much better tracker also scores 13–20 ft on them.

**Choosing per well can't be learned from 155 wells.** Picking the best model for each well with
hindsight would give 3.96 ft instead of 4.94. But every method for choosing without hindsight did
worse than one fixed weight.

---

## 7. Speed

### 7.1 Network time per model

The 22 September session, one well at a time; §0 explains why these numbers are only compared
with each other.

| model | parameters (whole model) | attention | network time |
|---|---|---|---|
| MobileOne-S1 | 10.2 M | none | 17.5 ms; **11.9 ms merged** |
| FastViT-T12 | 8.2 M | none | 16.7 ms; **12.0 ms merged** |
| FastViT-SA12 (plain or taught) | 12.3 M | last stage | 17.0–17.1 ms; **12.6–13.1 ms merged** |
| ConvNeXt-nano | 17.8 M | none | 14.8 ms |
| FastViT-SA24 | 22.2 M | last stage | 25.9 ms; **17.7 ms merged** |
| ConvNeXt-tiny | 31.9 M | none | 18.4 ms |
| InceptionNeXt-tiny | 28.1 M | none | 19.7 ms |
| ConvNeXt-small | 53.6 M | none | 26.1 ms |
| ConvNeXt-small + axial attention | 59.8 M | axial | 29.2 ms |
| ConvNeXt-base | 94.8 M | none | 36.6 ms |
| AnchorCNN (an earlier session) | 3.7 M | none | 14 ms network + 127 ms path search = 141 ms; 29 ms per well when 8 wells run together |

**Rules this produced:**
1. **Judge a model by measured milliseconds, not by operation counts.**
   - FasterNet does fewer operations than ConvNeXt-small and runs 1.8× slower.
   - FastViT does half of tiny's operations (40 against 82 GFLOP) but, before merging, runs only
     7% faster (17.1 against 18.4 ms).
   - At this input size, moving data through memory limits speed, not arithmetic.
2. **Changing spatial filters can't save much.** ConvNeXt spends 1.5% of its arithmetic on
   spatial (7×7) filters, 70.2% on channel mixing and 25.9% on other convolutions.
3. **The decoder is already small** (32 channels against the encoder's 768). What makes it cost
   time is the resolution it works at, not its width.

### 7.2 Precision and merging

The precision rows come from an earlier session with the same method, so compare their ratios,
not their milliseconds.

| change | effect on network time | effect on error | status |
|---|---|---|---|
| 32-bit → mixed 16-bit arithmetic (AMP, bf16) | 46.5 → 25–26 ms for tiny (−45%) | none measurable | already on in every run |
| Whole model in fp16, normalisation layers kept in 32-bit | a further −16% (tiny 22.0 ms, FastViT 19.0 ms) | output differs by 0.07% (bf16: 0.37%) | built only in a benchmark; not in scoring ⏳ |
| **Merging FastViT's training branches** | 17.0 → 13.1 ms (−23%) for the taught FastViT | 4.840 → 4.832 (and plain FastViT 5.213 → 5.229): rounding | ✅ **on by default** for every new run (`reparam_at_inference`) |
| CUDA graphs (replay recorded GPU work) | −0.5 to −2 ms | none | not worth it |
| Pruning the decoder | at most −3 to −4 ms, estimated | needs retraining; risks accuracy | not done |
| TensorRT | not measured; not installed on Windows | — | not done |
| Changing FastViT's attention | at most −1.6 ms (its share, §7.5) | — | not worth it |

### 7.3 What merging is

FastViT and MobileOne train each layer as several parallel branches: a 3×3 convolution, a 1×1
convolution, and a skip connection, each with batch normalisation. After training, the weights
are fixed, so the branches can be added into one 3×3 convolution that computes the same function.
The taught FastViT had 33 such layers. Merging is exact up to floating-point rounding, and needs
no retraining.

### 7.4 The whole pipeline, and the 22 September CPU fixes

Predicting one well has four steps. The network runs **once** per well.

| step | runs on | before (ms per well) | after (ms per well) | what changed |
|---|---|---|---|---|
| 1. Neighbour-well map (geo prior) | CPU | 54–57 | **7.9** | parsed CSVs cached; small tree searches run on one thread |
| 2. Build the 16 × 345 × 400 input | CPU | 12.5 | **8.3** | channels written once instead of copied three times |
| 3. Network (taught FastViT, merged) | GPU | 13.1 | 13.1 | — |
| 4. Output → depths | CPU | 4–6 | 4–6 | — |
| **Steps 2–4 as scoring runs them** | | 104 | **38.1** | worker processes no longer started for scoring |
| **Whole pipeline** | | **158** | **46** | **3.4× faster; predictions bit-identical** |

- **Worker start-up was the hidden cost.** The data loader used 4 worker processes. On Windows,
  starting them takes 13–15 s (each re-imports PyTorch). Spread over 155 wells, that looked like
  90 ms of "input building" per well. Building the inputs in the main process is now faster for
  155 wells, and `seq_NN_rescore.py` defaults to that. During training the workers start once and
  stay alive, so training was not affected.
- **The geo prior re-parsed 773 CSV files on every call.** That was 77% of its time. Each well's
  parsed columns are now cached in `data/.geo_cache/` (121 MB, ignored by git). The cache is keyed
  on each CSV's size and modification time, so an edited CSV is re-read.
- **Checked:** all 155 inputs and all 155 geo-prior results are bit-identical to before, and the
  error is unchanged (4.8322).

### 7.5 Attention in every model

"Attention" is a layer that lets each position weigh every other position. Its cost grows with the
square of the number of positions, so networks use it only where the image is small.

| model | attention | where | measured effect |
|---|---|---|---|
| ConvNeXt-small / tiny / nano / base | none | 7×7 convolutions only | the family of the best results |
| **FastViT-SA12** | self-attention, last stage | 2 blocks on a 22 × 13 grid (286 positions) | 1.58 ms of 19.6 (8%). Its accuracy value is untested: the T12 run ⏳ answers it |
| FastViT-SA24 | self-attention, last stage | 4 blocks on the same grid | ⏳ |
| FastViT-T12 | none | SA12 with the attention stage replaced by convolutions | ⏳ |
| MobileOne-S1, InceptionNeXt-tiny, AnchorCNN | none | convolutions | — |
| **ConvNeXt + axial attention** | axial | 3 blocks attending along the well | ⚪ −0.024 ft |
| 1st-place author's 10 transformer backbones | full / windowed | — | never submitted by the author |
| CAFormer / ConvFormer | last two stages / none | — | screened only |
| DINOv2 | global, everywhere | 14 × 14-pixel patches | rejected (§5.5) |
| "Bottleneck attention" (a guide we reviewed) | coarsest stage | — | not built: FastViT-SA12 already has this design |

No gain in this project came from attention.

---

## 8. Caveats

1. **Every reported score from the 1st-place code is picked on the test wells** (§0). The picked
   checkpoint beats the late average by 0.142 (small), 0.147 (tiny), 0.149 (small + more synthetic
   data), 0.067 (FastViT), 0.151 (FastViT taught by tiny) and 0.096 ft (FastViT taught by small).
   Consequences:
   - ConvNeXt runs compare fairly with each other.
   - ConvNeXt-vs-AnchorCNN comparisons tilt towards the ConvNeXt by that amount.
   - The ConvNeXt + AnchorCNN blend (4.85) inherits the flattery.
2. **The AnchorCNN was never trained properly with synthetic data.**
   - The 2nd-place recipe used 77% synthetic wells; the ConvNeXt always trained with 85%.
   - Our one attempt delivered 32%, because of a bug that is now fixed (the generator retries 12
     times and delivers 73%).
   - If synthetic data brings the AnchorCNN near 5.0 ft, blending would be worth revisiting.
     Results that don't depend on this: the failed cheaper designs, int8, gates and stacks, the
     error-line analysis, and every tracker result.
3. **One run per setting**, except the AnchorCNN control (3 seeds) and the tracker inputs (2 seeds).
   The ConvNeXt's own seed-to-seed noise has never been measured.
4. **Some settings were tuned on all 773 wells** by the original authors, including our test
   wells: the 1st-place scaling constants and its tracker settings.
5. **`model_best.pt` is never reported** (it is picked on the test wells). AnchorCNN numbers come
   from `model_last.pt`.
6. **The tracker is random.** Unless `PYTHONHASHSEED` is fixed, repeat runs differ slightly.
   Tracker numbers here are repeated.

---

## 9. Corrections log

| we said | what was wrong | corrected to |
|---|---|---|
| Far-anchor training degrades the model | A bug made every sample identical | ⚠️ invalid test; fixed, not re-run |
| "Still improving at round 114" | A ±0.3 ft wobble | The AnchorCNN peaks around round 100 |
| One run changed three things | The cause of its failure was unclear | Rule: one change per run |
| "Tracker inputs fixed the hard wells" | Plain runs range 9.1–25.2 ft on that well | ⚪ luck |
| "Siblings conclusively hurt the AnchorCNN" | One tangled run, below the noise | ⚪ |
| The first tracker tuning | Moved the tilt setting the wrong way | Measured best: ×9 |
| Tracker-input cache | Masked the last column; up to 49 ft wrong on 0.3% of columns | Fixed; now identical (0.000 ft) |
| Blend score 4.834 | Tuned on its own test wells | 4.852, cross-fitted |
| Synthetic generator | Gave up after one try: 32% instead of 77% | Retries 12 times: 73% |
| Synthetic run "worse" | Judged on its last score only | ⚠️ invalid test |
| ConvNeXt scores treated as unpicked | The code picks on the test wells | Flattery measured per run: 0.07–0.15 ft (§8) |
| MobileOne merging "broken" | Compared absolute differences on values reaching 500 million | Relative difference 5.2 × 10⁻⁷: exact |
| Teaching FastViT "not worth it" | Judged by its 3 ms saving alone | Run anyway: 5.21 → 4.84 |
| A taught student is "a worse averaging partner" | Considered similarity, not accuracy | Best pair found: 4.69 |
| Pseudo-labelling synthetic wells | The ConvNeXt already trains on 85% synthetic wells with exact labels | Dropped |
| A 6.5-hour `cnx_tiny_synth` run | Its expected effect (0.1 ft) is below what 155 wells can detect | Dropped |
| "Several network passes per well" | There is exactly one | Removed |
| "~100 ms per well of CPU input building" | It was worker start-up (13–15 s once); building takes 8.3 ms | §7.4 |

---

## 10. Open work

### 10.1 Runs waiting, in priority order

Every ConvNeXt command runs from `kaggle_1st_place\solution`. If the GPU runs out of memory,
replace `--batch-size 4 --grad-accum-steps 4` with `--batch-size 2 --grad-accum-steps 8`, which
trains on the same number of samples per step. Every new run merges its branches before scoring.

**1. Seed repeats of the best result.** The paper and the rig recommendation both depend on these.
Run once for each `NAME` in: `cnx_fastvit_kd_small_s11`, `cnx_fastvit_s11`,
`cnx_fastvit_kd_small_s23`, `cnx_fastvit_s23`. Optionally also `small_s11` and `small_s23`
(teacher repeats, ~5 h each).
```
python seq_NN_holdout_eval.py --id 0801_V2 --source-dir experiments/bilzard --cfg-name NAME --epochs 150 --output-dir results/NAME_ep150 --device cuda --offline-timm --batch-size 4 --val-batch-size 4 --grad-accum-steps 4
```
Pass criterion for the recommendation: the taught FastViT's last-checkpoint and late-average
scores are within 0.1 ft of ConvNeXt-tiny's (5.14 / 5.09) in every seed.

**2. FastViT-SA24 taught by small** (twice SA12's capacity, 17.7 ms merged).
```
python seq_NN_holdout_eval.py --id 0801_V2 --source-dir experiments/bilzard --cfg-name cnx_fastvit_sa24_kd_small --epochs 150 --output-dir results/cnx_fastvit_sa24_kd_small_ep150 --device cuda --offline-timm --batch-size 4 --val-batch-size 4 --grad-accum-steps 4
```

**3. FastViT-T12 taught by small** (the attention test: identical to SA12 except for the attention stage).
```
python seq_NN_holdout_eval.py --id 0801_V2 --source-dir experiments/bilzard --cfg-name cnx_fastvit_t12_kd_small --epochs 150 --output-dir results/cnx_fastvit_t12_kd_small_ep150 --device cuda --offline-timm --batch-size 4 --val-batch-size 4 --grad-accum-steps 4
```

**4. ConvNeXt-nano, alone and taught by the small + tiny average.** Use `--cfg-name cnx_nano`, then
`cnx_nano_kd_ens`, in the same command pattern.

**5. ConvNeXt-tiny and FastViT taught by the small + tiny average.** Use `cnx_tiny_kd_ens` and `cnx_fastvit_kd_ens`.

**6. InceptionNeXt-tiny, alone and taught by small.** Use `inx_tiny` and `inx_tiny_kd_small`. Low
priority: it is slower than tiny, and only ImageNet-1k weights exist.

**7. AnchorCNN with synthetic data, done properly.** Run from `kaggle2ndplace`:
```
python -u anchor_train.py --out runs/C_synth2_s1 --row 1.0 --n-move 5 --epoch-len 1150 --epochs 120 --eval-every 5 --tta 8 --seed 1 --synth-prob 0.77
```
Compare with `runs/C_s1` (6.024), both on the final score and on the average of the evaluations
from round 60 onwards (control: 6.26–6.38).

**8. Tracker inputs, third seed.** Run from `kaggle2ndplace`, `set "PYTHONHASHSEED=0"` on its own line first:
```
python -u anchor_train.py --out runs/C_s3_pfchan --row 1.0 --n-move 5 --epoch-len 1150 --epochs 120 --eval-every 5 --tta 8 --seed 3 --pf-channels 2
```
Compare with `runs/C_s3` (6.200).

**After any ConvNeXt run,** compare the unpicked scores:
```
python seq_NN_honest_curve.py results/cnx_tiny_ep150 results/NAME_ep150
```

### 10.2 Never tested

| item | state |
|---|---|
| ConvNeXt inside the AnchorCNN | built, never run |
| Sibling reference in the AnchorCNN, on its own | only tried on top of tracker inputs |
| Grid vs step options inside "C", separately | never separated |
| Far-anchor, fixed version (`--far-anchor 0.5`) | never run |
| Large-kernel ConvNeXt, fixed | never re-run |
| Neighbour-well inputs at 300 rounds | never run |
| More synthetic data past 150 rounds | never run |
| HMM tracker on the test wells | never scored |
| ConvNeXt seed-to-seed noise | never measured (repeats in §10.1 measure it) |
| Sibling disagreement as an input (`--sibling-sigma`) | built, never trained |
| Tracker inputs in the ConvNeXt (`pf_v1`) | set up, never run |
| Distillation weight and temperature | only weight 1, temperature 2 tried |
| fp16 weights in scoring | benchmarked only |

### 10.3 Making the measurement sharper

1. **3 seeds per important setting.** Three wins out of three is evidence; one win is not.
2. **5-fold cross-validation over all 773 wells** (AnchorCNN only): 5× the test wells. Needs a `--fold` option (not built).
3. **Report the median well's error too:** it moves about half as much with luck as the total does.

---

## 11. Paper plan

**Main claim:** a FastViT-SA12 taught by ConvNeXt-small is as accurate as its teacher, with 4.4×
fewer parameters (12.3 M against 53.6 M) and half the network time (13.1 against 26.1 ms).

| | error | last checkpoint | late average | network time | parameters |
|---|---|---|---|---|---|
| ConvNeXt-small (teacher; the 1st-place network) | 4.98 | 5.09 | 5.16 | 26.1 ms | 53.6 M |
| ConvNeXt-tiny | 4.90 | 5.14 | 5.09 | 18.4 ms | 31.9 M |
| FastViT-SA12 alone | 5.21 | 5.30 | 5.32 | 12.6 ms | 12.3 M |
| **FastViT-SA12 taught by ConvNeXt-small** | **4.84** | **4.92** | **4.97** | **13.1 ms** | **12.3 M** |

**Supporting results:**
- bigger backbones don't help (§5.5);
- cheaper backbones lose accuracy on their own (§5.5);
- blending and TTA cost 2–8× the time (§5.6, §7).

**Before submitting:**
- [ ] 3 seeds each of the student, FastViT alone and ideally the teacher (§10.1).
- [ ] Unpicked scores as the headline numbers.
- [ ] Ablations: distillation weight 0.5 / 1 / 2 and temperature 1 / 2 / 4.
- [ ] Report whole-pipeline time (46 ms) alongside network time.
- [ ] State that distillation is a standard method. The contribution is showing that it works
      with 618 training wells, and that the student matches a teacher 4× its size.
- [ ] Check the Kaggle data licence.
- [ ] Credit Ruby (1st), Bilzard (2nd), and the 3rd, 8th and 22nd-place write-ups.

A second, smaller paper, on why most improvements can't be detected on this benchmark (§6), would
suit a workshop that accepts negative results.

---

## 12. Lessons

1. **One change per run,** against a control at the same training length.
2. **Measure the noise first.** Here, one comparison carries ±0.39 ft of luck.
3. **Check which wells a gain comes from.** If it vanishes after removing 1–3 wells, it was luck.
4. **Check training samples before training.** Two bugs (identical far-anchor samples; 32% instead
   of 77% synthetic) each took under a minute to find once looked for.
5. **Don't read one run's curve as a trend.** It wobbles by ±0.3 ft.
6. **Simple beats clever with 155 test wells.** One fixed weight beat every learned chooser.
7. **A partner must make different mistakes and be accurate enough.** Both matter (§5.6).
8. **A teacher must be measurably better than the student.**
9. **Time every step of the pipeline before optimising one.** The "slow CPU step" was worker start-up.
10. **Measure instead of predicting.** Two confident predictions in §9 were wrong.
11. **Verify claims against the files.** Several things believed tested had never been run.

---

## 13. Where things live

**Commands and machine notes**
- Commands run in **cmd.exe**. Set variables with `set "NAME=value"` on their own line.
- ConvNeXt commands run from `kaggle_1st_place\solution`; AnchorCNN commands from `kaggle2ndplace`.
- ConvNeXt training needs `--batch-size 4 --grad-accum-steps 4` on this GPU.
- `huggingface_hub` downloads fail here (a certificate error). `urllib` works. Stored with verified
  checksums: ConvNeXt-small, -tiny and -nano; FastViT-SA12, -SA24 and -T12; InceptionNeXt-tiny.
- Windows scripts that start worker processes need `if __name__ == "__main__":`.

**Files**

| file | contents |
|---|---|
| `kaggle_1st_place/CLAUDE.md` | Every ConvNeXt experiment with full numbers |
| `kaggle2ndplace/TRAINING.md` | Every AnchorCNN, tracker and blend experiment |
| `kaggle_1st_place/solution/results/` | Every ConvNeXt run: config, log, model, predictions |
| `kaggle2ndplace/runs/` | Every AnchorCNN run |
| `…/experiments/bilzard/seq_NN_cfg.py` | All ConvNeXt recipes, including every taught and seed-repeat variant |
| `…/experiments/bilzard/seq_NN_train.py` | Training loop: distillation (`load_distill_teacher`, `distill_alignment_loss`) and merging (`reparameterize_for_inference`) |
| `…/experiments/bilzard/seq_NN_dataset.py` | Builds each well's input image |
| `…/experiments/bilzard/seq_NN_geo_prior.py` | Neighbour-well map, with the CSV cache |
| `kaggle_1st_place/solution/seq_NN_rescore.py` | Re-scores a finished run without retraining; `--reparam` merges branches; prints pipeline time |
| `kaggle_1st_place/solution/seq_NN_honest_curve.py` | Prints unpicked scores from a run's log |
| `kaggle_1st_place/solution/seq_NN_robust_compare.py` | k\* and bootstrap for two runs |
| `kaggle2ndplace/anchor_*.py` | AnchorCNN data, training, evaluation, blending, trackers, siblings |

---

## 14. Glossary

| term | meaning |
|---|---|
| **TVT** | The drill's position within the rock layers, in feet. What we predict. |
| **GR (gamma ray)** | The rock's natural radioactivity; the main sensor reading. |
| **Typewell** | The GR log of a nearby vertical well; the reference fingerprint of the layers. |
| **Holdout / test wells** | The 155 wells never trained on, used only for scoring. |
| **Error / RMSE** | See §0. |
| **Seed** | The random starting point of training. |
| **Round / epoch** | One pass of training. |
| **Checkpoint** | A saved copy of the model at one point in training. |
| **Reported / last checkpoint / late average** | See §0. |
| **k\*** | How many of the most helpful test wells must be removed before a gain vanishes. |
| **Bootstrap** | Re-scoring on random re-samples of the test wells. |
| **Cross-fitted** | Tuned on some wells and scored on others. |
| **Synthetic data** | Computer-generated training wells with exact labels. |
| **TTA** | Averaging predictions over several shifted copies of the input. |
| **Blend / average / ensemble** | Combining several models' predictions. |
| **Backbone** | The pre-trained feature extractor inside the U-Net. |
| **U-Net** | A network that shrinks the image to understand it, then enlarges it back to predict every position. |
| **Distillation (teacher / student)** | Training a cheaper network (student) to copy a better one's (teacher's) predictions. |
| **Attention** | A layer where each position weighs every other position. |
| **Merging (reparameterisation)** | Folding training-time parallel branches into one layer (§7.3). |
| **AMP / bf16 / fp16** | 16-bit arithmetic instead of 32-bit. |
| **Geo prior** | The first guess of depth made from neighbouring wells. |
| **Particle filter / HMM** | Trackers that follow the drill step by step without learning. |
| **Siblings** | Wells whose typewells come from the same master log. |
| **GFLOP** | Billions of arithmetic operations. |
