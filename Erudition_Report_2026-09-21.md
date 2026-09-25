# Erudition — Full Project Report

**Date:** 21 September 2026
**Covers:** all work in `kaggle_1st_place/` and `kaggle2ndplace/` up to this date (last updated
22 September: attention and speed for every model, merged FastViT branches, FastViT-SA24).
**Written for:** anyone, including people who don't work in drilling or machine learning.
Technical words are explained the first time they appear, and there is a glossary at the end (Part 12).

**How to read this:** Part 1 is a one-page summary. Parts 2–3 explain the problem and the
models. Parts 4–8 go through every experiment and its result. Parts 9–11 cover what is still
open and how to carry on.

---

## Contents

1. [The short version](#1-the-short-version)
2. [Background](#2-background)
3. [The story of the project](#3-the-story-of-the-project)
4. [Results](#4-results)
5. [Why this problem is hard to beat](#5-why-this-problem-is-hard-to-beat)
6. [Cost: running a model on a drilling rig](#6-cost-running-a-model-on-a-drilling-rig)
7. [Caveats that apply to the results](#7-caveats-that-apply-to-the-results)
8. [Mistakes we made, and what we corrected](#8-mistakes-we-made-and-what-we-corrected)
9. [What is still open](#9-what-is-still-open)
10. [Lessons and rules of thumb](#10-lessons-and-rules-of-thumb)
11. [Practical notes and where everything lives](#11-practical-notes-and-where-everything-lives)
12. [Glossary](#12-glossary)

---

## 1. The short version

- **Best single model: 4.94 ft of average error.** The 1st-place competition model (a
  "ConvNeXt"), trained for 150 rounds, with its predictions averaged over 8 slightly shifted
  views. For comparison, just assuming the drill stays at its last known depth gives about 9.8 ft.
- **Best combination: 4.85 ft.** That ConvNeXt blended with the 2nd-place model (an
  "AnchorCNN") using one fixed weight. The gain is tiny (0.085 ft), smaller than our test can
  reliably detect, and it makes the system 5–6× slower. **For a drilling rig, one model is the
  better choice.**
- **The rig question is settled: ConvNeXt-tiny, run once, no blending, no averaging.** Tested on
  19 September, it was as accurate as the standard ConvNeXt-small within what we can measure,
  while needing about a third less computing and training 38% faster (§6). Two cheaper backbones
  were then tested and the ladder stops here: FastViT lost 0.3 ft and MobileOne lost 1.6 ft. The
  accuracy-versus-size curve has a knee and tiny sits on it (§6).
- **Newest result, and the one a paper can be built on: a cheap model taught by ConvNeXt-small
  matches or beats its teacher.** FastViT alone scored 5.21. Taught by ConvNeXt-small, it scored
  **4.84, the best single model we have**, against its teacher's 4.98, while running about
  **2× faster** (13.1 vs 26.1 ms for the network, with its branches merged, §6.3). The gain over FastViT alone passes the "is it real?"
  check. The gain over the teacher does not (the student wins 77 of 155 wells), so the honest
  claim is **"as accurate as the teacher, at half the cost"**. This is one run each; a paper needs
  repeats (§9.5).
- **The first teaching result:** FastViT on its own scored
  5.21. Trained to imitate ConvNeXt-tiny's predictions ("knowledge distillation"), it scored
  **4.93**, closing about 90% of the gap to tiny (4.90). This is one run, so it is promising
  rather than proven, but it is the first idea in a long while to do clearly what it was meant to (§4.10).
- **Averaging two ConvNeXts is the most reliable gain we have found: 4.78 ft** (small + tiny,
  0.13 ft better than tiny alone, and it passed the "is it real?" check convincingly). It doubles
  the inference cost, so the plan is to *teach* that pair's knowledge to one cheap model rather
  than deploy two (§4.10, §9.1). **Best pair so far: tiny + the FastViT taught by small,
  4.69 ft** (passes the check, 98% of resamples agree), at about the cost of small alone.
- **Inference speed:** FastViT's training-time branches are now merged automatically before
  scoring. Merging makes the best student ~23% faster with the same accuracy (4.84 → 4.83), so it
  runs in **13.1 ms against its teacher's 26.1 ms, about 2× faster** (§6.3). A deeper
  FastViT-SA24 student is built: once merged, it runs at about today's un-merged SA12 cost, with
  twice the capacity (§9.1). The model already uses reduced-precision arithmetic (AMP). Switching
  to full half-precision weights saves a further ~16% for free. Pruning the decoder or installing
  TensorRT would save little: the network is only about a third of the total time per well (§6.3).
- **This changes the rig recommendation:** the FastViT taught by small (4.84, 13.1 ms merged)
  is both more accurate and faster than ConvNeXt-tiny alone (4.90, 18.4 ms). Two caveats: one
  seed, and the gap to tiny is within noise. So the recommendation is now "**the taught FastViT,
  once a second seed confirms it**; tiny until then" (§9.1).
- **Correction to all ConvNeXt scores:** the 1st-place code picks the checkpoint that scores best
  on the test wells, so its reported numbers are about 0.15 ft flattering (§7.7). Comparisons
  between ConvNeXt runs are still fair, because they were all picked the same way.
- **Most ideas we tried did not measurably help.** A few failed badly and can be ruled out.
  Many more changed the score by less than luck does, so we can't tell whether they help.
  Some tests turned out to be broken.
- **The main reason it's hard to improve:** a handful of very difficult wells dominate the
  score, and how badly a model does on them changes a lot just from luck in training. With only
  155 test wells, any single test run can't see improvements smaller than about 0.4 ft.
- **An important gap:** the 2nd-place model has not yet been properly tested with its synthetic
  (computer-generated) training wells, which made up about 77% of its original recipe. The one
  attempt only delivered 32% synthetic wells because of a bug (now fixed), and it looked
  promising for most of its training before it overfitted. Several conclusions about this model
  could change once it is tested properly.

## 2. Background

### 2.1 The problem

Many oil and gas wells are drilled **sideways** ("horizontal wells") through a thin layer of rock
that is worth producing from. The drill can't see where it is. It knows how far it has drilled
along the well and how deep it is, and it has one sensor reading: **gamma ray (GR)**, the natural
radioactivity of the surrounding rock. Each rock layer gives off a different amount, so the GR
reading works like a fingerprint for the layer the drill is in.

There is also a **reference log**, called a **typewell**: the GR readings from a nearby
**vertical** well, which passes straight down through every layer. A geologist locates the
horizontal well by sliding its GR readings along the typewell until the patterns match.

**The task:** for the first part of each well, the drill's position among the layers is known.
For the rest of the well, predict that position (called **TVT**, measured in feet) at every
point, using only the GR readings, the drilling path and the typewell.

**Why it's hard:** rock layers repeat. The same GR pattern can appear at several depths, so a
wrong match can look just as convincing as the right one.

### 2.2 The data and the scoring

- **773 wells** from a public Kaggle competition run by the company ROGII.
- We trained on **618 wells** and kept **155 wells** completely aside for testing (the
  **holdout**). The models never see those 155 during training. Both repos use exactly the same
  split, so their scores can be compared directly.
- Every score is the **average error in feet** over all **754,122** predicted points on the 155
  test wells. It is calculated as an RMSE, which counts large mistakes extra. Lower is better.

Reference points:

| what | error (ft) |
|---|---|
| Assume the drill stays at its last known depth | ~9.8 |
| The 1st-place model's map-based guess from neighbouring wells alone | 11.57 |
| Error that comes just from rounding the answer onto the model's grid | 0.06 |

### 2.3 The two starting models

We started from the two best published solutions to the competition.

| | author | how it works |
|---|---|---|
| **ConvNeXt U-Net** ("the ConvNeXt") | 1st place (Ruby) | Turns each well into a picture (GR pattern against depth) and uses a large image-recognition network to find the path. Trains on real wells plus many computer-generated ("synthetic") wells. About 54 million adjustable numbers. |
| **AnchorCNN** | 2nd place (Bilzard) | A smaller image network (about 3.7 million numbers) that predicts, step by step, how much the position changes, then adds the steps up. A final "decoding" step picks the most likely overall path. |

A third kind of method was also used: **trackers** (a *particle filter* and an *HMM*). These
don't learn from examples; they follow the drill step by step using simple physics and the GR
match.

### 2.4 The computer

A laptop with an **RTX 3050 GPU (6 GB)** and 16 GB of memory. This limits batch sizes and
training time:
- one AnchorCNN training run takes about **1 hour**
- one ConvNeXt run of 150 rounds takes about **5 hours**

### 2.5 How we checked that results are real

Because the score is noisy, we used three checks alongside the raw number:
- **Paired controls:** compare a change against the *same* settings without it, trained with the
  same random seed.
- **The acceptance test (k\*):** remove the test wells that contribute most to a difference,
  one at a time. k\* is how many you must remove before the improvement disappears. A high k\*
  means the gain is spread across many wells. A low k\* (1–2) means it rests on a couple of wells.
- **Resampling (bootstrap):** re-score on many random re-samples of the test wells and see how
  often the improvement holds.

### 2.6 What kind of machine-learning problem this is

Useful to know, because it decides which published methods are relevant. **It is not
classification.**

- **The overall task is dense regression**: for every point along the well, predict a continuous
  number (depth within the rock layers).
- **In image terms it is stereo matching**, also called disparity estimation. The 1st-place model
  is literally one: the gamma-ray mismatch forms a "cost volume", the network aggregates that
  cost, and the final averaging step is the standard soft-argmin used in stereo networks. In
  stereo you work out how far each pixel shifts between two camera views; here you work out how
  far the well sits from its reference log. Same mathematics.
- **The final decoding step is path finding.** The AnchorCNN searches for the best path through a
  2-D cost image using dynamic programming — a shortest-path search.
- **Classification appears inside** the AnchorCNN (each step is sorted into one of 21 "how far did
  it move" options, then the steps are added up), but that is a means, not the task.

Closest named cousins if you want literature: stereo disparity estimation; retinal layer
segmentation in eye scans (finding boundary curves through an image, usually with dynamic
programming); and seismic horizon tracking, which is the same problem in the same industry.

## 3. The story of the project

1. **Rebuilding the 2nd-place model.** We wrote the AnchorCNN's training code from scratch on
   the same split as the 1st-place code. Three bugs were found and fixed along the way:
   - Training samples were being cut from the curved start of each well. There, 4.3% of the
     moves fell outside what the model could express, with labels reaching 1021 ft. After the
     fix: 0.01% and 118 ft.
   - A smoothing trick (EMA) was stuck on the untrained model, so evaluations showed a
     meaningless 18.75 ft.
   - A data cache was being overwritten between runs.

   As a wiring check, we ran the 2nd-place author's own trained weights through our code:
   4.03 ft. That proves the pipeline is correct. It is not a fair score, because those weights
   had seen our test wells. Our own first model scored 6.49 ft.
2. **Testing the 1st-place model.** We built the 80/20 test for it, a "was that improvement
   real?" check, and ports of three ideas from the 2nd-place write-up (neighbour-well inputs,
   more realistic synthetic wells, and averaging over shifted views). We trained at 60 and then
   150 rounds, and tried four redesigns of the network.
3. **Making the AnchorCNN cheaper.** We tried coarser grids, cheaper first layers, a cheaper
   network design, 8-bit numbers, a smoothing filter, and copying from a larger ensemble.
4. **Combining the two models.** We tried a single weight, a learned chooser ("gate"), and a
   chooser with an extra training rule from the 3rd-place team.
5. **Building up the trackers.** We repaired our particle filter step by step (28.98 → ~12 ft),
   built an HMM, and added a per-well slope correction.
6. **Ideas from other teams' write-ups.** From the 8th place: where the error lives. From the
   22nd place: rock-system grouping, a hidden constant, using the tracker as an extra input,
   training on longer stretches, and better synthetic data. We checked their claims against our data.
7. **A full review.** We scored the 1st-place author's much stronger tracker, tested every way
   of combining models, measured model sizes for a rig, and audited what had really been tested.
8. **Making it cheap enough for a rig.** We screened a dozen replacement networks, trained the
   two most promising (ConvNeXt-tiny: as good, a third cheaper; FastViT: worse), built a third
   (MobileOne: 36% faster but 1.6 ft worse), and found that the reported scores of every 1st-place run
   were flattered by about 0.15 ft because the code keeps whichever checkpoint scored best on the
   test wells.
9. **Averaging and teaching (21 September).** We found that averaging ConvNeXt-small and
   ConvNeXt-tiny gives the most reliable improvement yet, and that a cheap model can be taught to
   imitate a better one (FastViT: 5.21 → 4.93). We built "teacher/student" training for several
   combinations, screened more backbones (ConvNeXt-nano came out fastest), and measured where
   inference time goes.

## 4. Results

### 4.1 Headline numbers (all on the same 155 test wells)

**How to read the two new columns.**
- **Attention:** whether the network uses "attention" (a layer that lets every position look
  at every other position), and where. See §6.4 for what each kind means.
- **Speed:** the time the network needs for one well. It was measured for every model in one
  session under identical conditions (22 September; idle laptop GPU, 16-bit arithmetic, one
  well at a time). Averages add up their members' times. Compare the numbers within this table
  only: the same model can run ~30% faster or slower on another day, depending on the laptop's
  power state. The whole pipeline takes roughly 180 ms per well, and most of it is CPU work
  that prepares each well's input; the network is about 10% of that (§6.3). "Merged" means FastViT's or
  MobileOne's training branches were collapsed for inference (§6.3), which doesn't change
  accuracy.

| system | error (ft) | attention | network time per well | notes |
|---|---|---|---|---|
| Assume "no change" | ~9.8 | — | 0 | the baseline any model must beat |
| AnchorCNN, original settings | 6.49 | none | ~14 ms + ~127 ms decoding | one run; decoding runs one well at a time (§6) |
| AnchorCNN, "C" settings (half the compute) | 6.02 / 6.57 / 6.20 | none | less than the original | three seeds; average 6.26 |
| AnchorCNN + tracker inputs | 5.84 / 6.11 | none | + ~24 ms CPU tracker | two seeds; each beat its matching plain run |
| AnchorCNN + synthetic data (flawed run) | 6.30 | none | as "C" | only 32% synthetic instead of 77%, and twice the training steps; see §7.1 |
| ConvNeXt, 60 rounds | 5.16 | none | 26.1 ms | |
| ConvNeXt-small, 150 rounds | 4.98 | none | 26.1 ms | checkpoint picked on test wells (§7.7) |
| ConvNeXt-small + averaging over 8 shifted views | 4.94 | none | ~209 ms (8 passes) | same caveat |
| ConvNeXt-small with axial attention (redesign), 60 rounds | no gain (−0.024 ft vs its control) | **axial attention**, 3 blocks | 29.2 ms | §4.3 |
| **ConvNeXt-tiny**, 150 rounds | **4.90** | none | 18.4 ms | same caveat; a third less computing than small |
| FastViT-SA12, 150 rounds | 5.21 | **self-attention, last stage only** | 17.1 ms (12.6 merged) | a cheaper backbone; worse on its own (§6) |
| MobileOne-S1, 150 rounds | 6.51 | none | 17.5 ms (11.9 merged) | the cheapest backbone tried; far worse (§6) |
| FastViT-SA12 taught by ConvNeXt-tiny, 150 rounds | 4.93 | self-attention, last stage | 17.0 ms (13.1 merged) | one run; 0.28 better than FastViT alone (§4.10) |
| **FastViT-SA12 taught by ConvNeXt-small**, 150 rounds | **4.84** (4.83 merged) | self-attention, last stage | **13.1 ms merged** | **best single model**; one run; about **2× faster than its teacher** (26.1 ms) (§4.10) |
| Average of ConvNeXt-small + ConvNeXt-tiny | 4.78 | none | 44.5 ms | passed the "is it real?" check (§4.10) |
| Average of small + tiny + a synthetic-data variant | 4.77 | none | ~71 ms | three models; barely better than two |
| Average of ConvNeXt-tiny + FastViT | 4.85 | in the FastViT half | 31.0 ms (merged) | |
| Average of ConvNeXt-small + FastViT | 4.90 | in the FastViT half | 38.7 ms (merged) | failed the "is it real?" check |
| Average of ConvNeXt-tiny + the FastViT taught by tiny | 4.77 | in the FastViT half | 31.5 ms (merged) | passes (§4.10) |
| Average of ConvNeXt-small + the FastViT taught by tiny | 4.82 | in the FastViT half | 39.2 ms (merged) | passes (98%) |
| Average of ConvNeXt-small + the FastViT it taught | 4.75 | in the FastViT half | 39.2 ms (merged) | passes (97%) |
| **Average of ConvNeXt-tiny + the FastViT taught by small** | **4.69** | in the FastViT half | **31.5 ms (merged)** | **best overall**; passes (98%) |
| ConvNeXt + AnchorCNN, one fixed weight | 4.85 | none | 5–6× the ConvNeXt alone | the earlier "best combination" |
| Our particle filter (best version) | ~12.0 | not a network | ~24 ms on the CPU | a tracker, not a learned model |
| The 1st-place author's particle filter | 7.35 | not a network | ~5.6 s on the CPU (4 workers) | much better tracker than ours |

For context: the 1st-place author reported 4.80 ft using 15 models × 300 rounds and a
different testing method. That is not directly comparable, but it shows better scores are
possible with far more computing.

### 4.2 What worked

| change | result | how sure |
|---|---|---|
| **"C" settings for the AnchorCNN** (coarser vertical grid, fewer step sizes) | **Half the compute** (15.7 → 7.9 GFLOP; training 56 → 29 min), no measurable loss in accuracy | Solid for "no loss". "Better" is not proven. |
| **Averaging over 8 shifted views (TTA) on the AnchorCNN** | Improved every run, by 0.26–0.56 ft | Solid: always the same direction, measured on the same trained model |
| **Training the ConvNeXt longer** (60 → 150 rounds) | 5.16 → 4.98 ft | Solid |
| **Repairing our particle filter** | 28.98 → 16.82 → 15.84 → 13.07 → ~12.0 ft (four measured fixes, §4.6) | Solid as a tracker |
| **"Sibling" reference for the particle filter** (§4.8) | 12.69 → 12.01 ft | Solid: two repeats, a gap five times the run-to-run noise |
| **Blending ConvNeXt + AnchorCNN with one weight** | 4.94 → 4.85 ft | Real but tiny, below what the test can resolve, and 5–6× slower |
| **Averaging ConvNeXt-tiny + the FastViT it taught** | 4.90 → 4.77 ft, at ~31.5 ms per well (merged) | Good: the gain survives removing 31 wells, and 90% of resamples agree. The best pair found, and cheaper than small + tiny |
| **Averaging ConvNeXt-small + the taught FastViT** | 4.98 → 4.82 ft | Good: survives removing 54 wells, and 98% of resamples agree |
| **Teaching FastViT with ConvNeXt-small** | 5.21 → 4.84 ft (teacher: 4.98). Unpicked: last 5.30 → 4.92, late average 5.32 → 4.97 | vs FastViT alone: good (survives removing 35 wells, 95%). vs its teacher: *as good*, not provably better (77 of 155 wells) |
| **Averaging ConvNeXt-tiny + the FastViT taught by small** | 4.90 → 4.69 ft | Good: survives removing 19 wells, 98% of resamples agree |
| **Removing a repeated computation** in the tracker inputs | 17× faster, identical results (checked to 0.000 ft) | Solid |
| **Averaging ConvNeXt-small + ConvNeXt-tiny** | 4.90 → 4.78 ft | Good: the gain survives removing the 67 wells that contribute most, and 91% of resampling checks agree |
| **Teaching FastViT to imitate ConvNeXt-tiny** (distillation) | 5.21 → 4.93 ft; unpicked checkpoints agree (last 5.30 → 5.12, late average 5.32 → 5.11) | Promising: one run, but every way of scoring it points the same way |

### 4.3 Decisively negative: failed by a wide margin, or consistently under careful testing

These are safe to rule out.

| idea | what happened |
|---|---|
| **Separable encoder** (a much cheaper network design) | 11.4 ft vs ~6.2 for the normal network |
| **8-bit numbers (int8)** to run faster | 25.1 ft as is, 10.8 ft after retraining, and *slower* on this computer's CPU (0.63–0.76× speed). Only the file got smaller (15.3 → 4.8 MB). |
| **8 ft vertical steps** (coarser than C) | 1.46 ft worse than C. The loss spread over 124 of 155 wells, and 100% of resampling checks agreed. The cause is known: at that coarseness, real moves round to "no move". |
| **Cheaper first layer ("stride-2 stem")** | Worse across 145 of 155 wells; typical-well error 3.67 → 4.11 ft; training became unstable |
| **Both cheapening changes together** | 7.64 ft, worse than the two separately would predict |
| **A learned "gate"** choosing between the two models point by point | 5.18 ft vs 4.87 for one fixed weight. It helped typical wells but chose confidently and wrongly on the hard ones. |
| **The gate plus the 3rd-place team's extra training rule** | No better than the plain gate (5.01) |
| **A "disagreement" rule** (switch models where they disagree most) | 5.09–5.12 vs 4.85. Disagreement barely predicts where the blend goes wrong (correlation 0.32). |
| **Stacking all 17 sets of predictions** we had | 4.88–4.95 vs 4.85; more members made it worse |
| **Tracker as a blend partner** (ours, and the author's better one) | Blend weight 0.00–0.06: adds nothing (§4.7) |
| **The hidden "0.99904" constant** in how the data was prepared | Reproduced exactly (0.99904 ± 0.00102), but using it to correct each well's depth made every setting worse (5.00–20.05 ft vs 4.94). It pins the average GR level, not the depth. |
| **Predicting each well's error line** from anything known in advance | No predictive power on either network (cross-checked); a model built just for this did worse than what the network already implies |

### 4.4 Unresolved: the effect is smaller than the test can detect

This does **not** mean "doesn't work". It means the effect is too small or too inconsistent for
a 155-well test to separate from luck (§5.1).

| idea | what we saw |
|---|---|
| **Averaging over shifted views on the ConvNeXt** | Helped 4 times out of 4, but only by 0.02–0.15 ft each time |
| **Tracker inputs to the AnchorCNN** | Better in both seeds (−0.18 and −0.46 ft), but each gain rests on 1–2 wells. A third seed is waiting. |
| **Sibling reference inside the AnchorCNN** | One run, only tried on top of the tracker inputs, 0.25 ft worse |
| **Averaging 3 seeds of the AnchorCNN** | 5.99 vs 6.02 for the best single seed; made the blend slightly worse |
| **A smoothing filter to replace view-averaging** | Didn't replace it; overall result within normal seed variation |
| **More synthetic data for the ConvNeXt** | 0.15 ft worse at 150 rounds, but better on most wells and still improving when stopped |
| **Synthetic data for the AnchorCNN** (flawed first run) | Final score 6.30, inside the plain runs' range. Better than all three plain runs over the second half of training, then it overfitted. Only 32% synthetic because of a bug (§7.1). |
| **Neighbour-well inputs for the ConvNeXt** | Far behind after 60 rounds, but badly undertrained (at the level the baseline reached by round 19) |
| **Axial attention in the ConvNeXt** | −0.024 ft, smaller than a pure-luck difference measured at −0.030 |
| **Two other ConvNeXt redesigns** ("unimodal", "RAFT") | Slightly worse at 60 rounds, which their own notes call too short |
| **C settings versus the original AnchorCNN** | No measurable loss; the apparent gain rests on one well |

### 4.5 Invalid tests: broken bugs or flawed setups

These say nothing about the ideas themselves.

| test | what was wrong |
|---|---|
| **Longer-stretch training ("far-anchor"), version 1** | A bug made every training sample from a well *identical* (overlap 1.00 vs 0.87 normally). The model memorised them and got steadily worse, from 6.49 to 7.92 ft over 250 rounds. Fixed, never re-run. |
| **"Large-kernel" redesign** (1st-place code) | Built so it could never learn: two parts that each needed the other to be non-zero, both started at zero. After training, all 36 were still exactly zero. Fixed, never re-run. |
| **First version of the gate** | Stuck at 50/50 by how it was set up. Fixed; the fixed version still lost. |
| **Copying from a larger model ("distillation")** | The ensemble being copied (5.99) was no better than the small model already was, so there was nothing to learn |
| **"Training on twice as much data didn't help"** | Compared runs with different seeds *and* different amounts of data. Not a real test. |

### 4.6 The particle filter story (a tracker that doesn't learn)

The filter imagines hundreds of possible positions and, at each step, keeps the ones whose
predicted GR matches the reading.

| stage | error (ft) | what fixed it |
|---|---|---|
| as first built | 28.98 | It kept losing track: error grew from 4 ft early in the well to 30 ft at the end. |
| allow the slope to change faster (×9) | 16.82 | The rock's tilt varies more than the filter allowed. Our first tuning moved this setting the *wrong way*; measuring showed it had to go up, with a clear best value at ×9. |
| correct a steady downward lean | 15.84 | The error leaned the same way in 72% of wells (worth about 13 ft over a typical well), so one correction fitted on training wells removed it. |
| predict each well's own correction | 13.07 | Fitted on training wells. The strongest clue was the drilling path's own angle. |
| better reference ("siblings", §4.8) | ~12.0 | Two repeats: 12.73 / 12.65 → 12.08 / 11.95 |

We also ruled out the obvious suspect ("not enough imagined positions"): 8 times as many
changed the answer by 0.02 ft. The remaining errors come from wells where the GR evidence
genuinely points to the wrong layer.

**The 1st-place author's filter**, run on our test wells (it only uses information available at
prediction time, which we checked in its code): **7.35 ft** with smoothing, 7.90 without. It
costs about 5.6 seconds per well. It's a far better tracker than ours, but see §4.7.

**The HMM tracker** (a different kind of tracker) was built and runs fast (0.11 s per well), but
early tuning only reached about 23 ft, and it was never scored on the test wells.

### 4.7 The blending story (combining models)

Blending helps only when the partner makes **different** mistakes. We measured how similar
each candidate's mistakes are to the ConvNeXt's (1.0 = identical, 0 = unrelated):

| partner | its own error | similarity of mistakes | blend result |
|---|---|---|---|
| Other versions of the 1st-place model | 4.98–5.52 | 0.84–0.97 | 4.85–4.92 |
| AnchorCNN | 5.84–6.02 | 0.67–0.68 | **4.85** |
| 1st-place author's tracker | 7.35 | 0.59 | 4.93 |
| Our tracker | ~12.0 | 0.37 | 4.93 (almost no gain) |

Two lessons:
- **A different model design is worth more than a different version of the same model.** The
  AnchorCNN is the best partner because it works in a genuinely different way.
- **Making a partner more accurate also made it more similar.** Every time we improved our
  tracker, its mistakes became more like the ConvNeXt's, so its value as a partner never grew.

**Update, 21 September: the second lesson has a limit.** The pairs of ConvNeXt-family models in
§4.10 were measured later, against ConvNeXt-tiny:

| partner for tiny | its own error | similarity of mistakes | average of the two |
|---|---|---|---|
| FastViT | 5.21 | 0.84 | 4.85 |
| ConvNeXt-small | 4.98 | 0.87 | 4.78 |
| **FastViT taught by tiny** | **4.93** | **0.88** | **4.77** |

Teaching FastViT did make its mistakes more like tiny's (0.84 → 0.88). But it also made FastViT
so much more accurate that it became the *best* partner, not a worse one. Similarity and
accuracy both matter. A small rise in similarity can be outweighed by a big gain in accuracy,
while our tracker never gained enough accuracy to make up for becoming more similar.

The simplest blend won every time. One fixed weight (cross-checked so it was never tuned on the
wells it was scored on) beat every gate, rule and stack.

### 4.8 Other things we built and found

- **Rock systems and siblings.** The 773 typewells are really crops of just **54 master logs**:
  where two overlap, they agree exactly. Wells that share a master log are drilling the same
  rock, and we call them siblings. Pooling the GR from sibling training wells gives a better
  reference: it matched the real GR better than the typewell on 77% of test wells (10.78 → 10.18
  error units). It helped the tracker (§4.6); inside the AnchorCNN it's unresolved.
- **Tracker inputs for the AnchorCNN.** Feeding the tracker's best guess and its confidence into
  the network as two extra inputs, as the 22nd-place team suggested. The tracker's stated
  confidence matched its real error well (7.56 vs 7.51 ft).
- **The 1st-place code already had tracker inputs**, switched off. The author's own later
  versions that used them scored worse on the author's own tests (5.01 and 5.54 vs 4.80), though
  other things changed too.
- **Coverage gap.** The AnchorCNN had never been trained on positions more than 7.7 thousand ft
  past the known part of a well, while test wells go to 10.1 thousand ft, and about 70% of the
  error is in the far half. A fix exists but hasn't been fairly tested (§4.5).
- **No duplicate wells** between our training and test sets (0 of 155).

### 4.9 Other teams' claims we checked

| claim | whose | our result |
|---|---|---|
| Typewells collapse to 54 master systems | 22nd place | **54**, found independently |
| Sibling reference improves a particle filter by 0.78 ft | 22nd place | **0.68 ft** |
| Hidden constant 0.99904 ± 0.001 | 22nd place | **0.99904 ± 0.00102**, but unusable (they also failed to use it) |
| About 75% of error is far from the known part | 22nd place | **69–70%** |
| The error is mostly one wrong line per well | 8th place | Reproduced on every one of our models |
| Learned choosers and extra correctors don't transfer | 22nd place | Same here |
| Averaging many seeds of a strong model doesn't help | 22nd place | Same here |

### 4.10 Averaging and teaching (21 September)

**Averaging two ConvNeXts.** We averaged the depth predictions of ConvNeXt-small and
ConvNeXt-tiny, which were already trained, so this cost nothing to try.

Costs are network time per well from the 22 September benchmark (§4.1), with FastViT merged.

| combination | error (ft) | vs tiny alone | "is it real?" check | cost per well |
|---|---|---|---|---|
| tiny alone | 4.90 | — | — | 18.4 ms |
| **small + tiny** | **4.78** | **−0.13** | passes convincingly: survives removing the 67 most influential wells; 91% of resamples agree | 44.5 ms |
| small + tiny + synthetic variant | 4.77 | −0.14 | passes | three models |
| tiny + FastViT | 4.85 | −0.06 | passes weakly: 72.5% of resamples agree | 31.0 ms |
| small + FastViT | 4.90 | −0.07 vs small | **fails**: the gain rests on just 2 wells | 38.7 ms |
| **tiny + taught FastViT** | **4.77** | **−0.14** | passes: survives removing 31 wells; 90% of resamples agree | **31.5 ms** |
| small + taught FastViT | 4.82 | −0.16 vs small | passes: survives removing 54 wells; 98% of resamples agree | 39.2 ms |

- **This is not the same as "averaging seeds doesn't help"** (§4.9). small and tiny are different
  depths of network, and their mistakes overlap a little less (similarity 0.87) than repeat runs
  of one design would.
- FastViT is the weakest of the three models, yet its mistakes overlap *least* with tiny's (0.84).
  That is why it still helps as a partner (the lesson from §4.7 again).
- One caveat: the component scores use the test-picked checkpoints (§7.7). The comparison with
  tiny's 4.90 is fair because it was picked the same way, but the absolute 4.78 is a little flattering.
- For a rig, two models means double the inference. That is why we turned to teaching.

**Teaching a cheap model (knowledge distillation).** A "student" network is trained on the
normal labels *and* to imitate a finished "teacher" network. It copies the teacher's full
opinion of how likely every possible depth is, not just its final answer. This passes on things
a single label cannot, such as "this other depth also looked plausible", which matters when rock
layers repeat.

An earlier attempt on the AnchorCNN taught nothing (§4.5), because its teacher was no better than
the student. Here, we first checked that each teacher really was better.

| student | teacher | status | result |
|---|---|---|---|
| **FastViT** | ConvNeXt-tiny | **finished** | **5.21 → 4.93** (tiny: 4.90). Unpicked: last 5.30 → 5.12, late average 5.32 → 5.11. About 90% of the gap closed. |
| ConvNeXt-tiny | small + tiny average | built, waiting | can one tiny model keep the 0.13 ft averaging gain? |
| FastViT | small + tiny average | built, waiting | does a stronger teacher help further? |
| **FastViT** | **ConvNeXt-small** | **finished** | **5.21 → 4.84**, beating its teacher's 4.98 on the headline score and on both unpicked measures (last 5.09 → 4.92, late average 5.16 → 4.97). See below. |
| **ConvNeXt-nano** | small + tiny average | built, waiting | the most promising: nano is the fastest backbone that fits (§6.1) |
| InceptionNeXt-tiny | ConvNeXt-small | built, waiting | weak case: small is no better than tiny on unpicked checkpoints, and InceptionNeXt is slower than tiny |

- **Distillation and averaging can be combined. (We first predicted the opposite, and were
  wrong.)** A student taught by tiny does make more of tiny's mistakes: the overlap rose from
  0.84 to 0.88. But it became so much more accurate on its own (5.21 → 4.93) that tiny plus its
  own student (4.77) is the best pair found. It is also cheaper than small + tiny, because FastViT
  is the cheaper second model.
- **FastViT taught by ConvNeXt-small: the student matches its teacher.**

  | | teacher: ConvNeXt-small | FastViT alone | **FastViT taught by small** |
  |---|---|---|---|
  | reported score (picked checkpoint) | 4.98 | 5.21 | **4.84** |
  | last checkpoint (unpicked) | 5.09 | 5.30 | **4.92** |
  | average over rounds 100–150 | 5.16 | 5.32 | **4.97** |
  | wobble between rounds (sd) | 0.12 | 0.06 | **0.04** |
  | network time per well (22 Sept benchmark) | 26.1 ms | 17.1 ms (12.6 merged) | 17.0 ms (**13.1 merged**) |

  - **vs FastViT alone (−0.37 ft):** passes the "is it real?" check. The gain survives removing
    the 35 most influential wells, and 95% of resamples agree.
  - **vs its teacher (−0.14 ft):** every score favours the student, but the check does not pass.
    The gain disappears after removing 8 wells, only 73% of resamples agree, and the student is
    better on 77 of 155 wells, which is exactly half. Honest claim: **the student is as accurate
    as its teacher, at about half the inference time** (once merged).
  - **Why a student can match or beat its teacher:** imitating the teacher's full opinion about
    every depth is a gentler, smoother target than the single right answer alone. It acts as a
    guide that keeps the student from memorising the 618 training wells. The student's training
    was also the steadiest of any run (round-to-round wobble 0.04 ft).
  - **Teacher choice (small vs tiny) made no measurable difference:** 4.84 vs 4.93, and the check
    fails (67%).
  - **As an averaging partner it is the best we have found:** tiny + this student reached
    **4.69** (passes, 98%), and small + this student reached 4.75 (passes, 97%). As predicted,
    it shares fewer mistakes with tiny (0.86) than tiny's own student does (0.88).
- **Using DINOv2 as the teacher was considered and rejected.** It is not trained for this task.
  It would first need an adapter, a padded input and a full training run on the same 618 wells.
  Its coarse single-scale view (25×29 positions) would likely make it *worse* than tiny, and a
  teacher worse than the student teaches nothing. The small + tiny average is already a
  better-measured teacher, at no training cost.
- **Using ConvNeXt-base as the teacher was also rejected.** Bigger models don't help here, because
  the data runs out before the model size does (§6). It also probably can't be trained on a 6 GB GPU.

## 5. Why this problem is hard to beat

### 5.1 The measuring stick is blunt (the noise floor)

- **A few wells decide the score.** The worst 1, 3, 10 and 52 of the 155 test wells carry 8%,
  20%, 43% and 86% of all the error. In practice the score is decided by 10–20 wells.
- **Luck moves the hard wells a lot.** Identical AnchorCNN settings, retrained with different
  random seeds, scored anywhere from **9.1 to 25.2 ft** on one hard well (`f6d009f4`).
- **Run-to-run noise is bigger than most effects.** Three identical runs differing only in seed
  scored 6.02, 6.57 and 6.20 (spread ±0.28 ft). Comparing one new run with one old run carries
  about **±0.39 ft** of pure luck. Most ideas we tested moved the score by 0.02–0.3 ft.
- **An unchanged model "passed" the test.** In the 1st-place code, a run that turned out to be
  identical to the baseline passed the acceptance test anyway (0.03 ft better, spread over 21 wells).
- **Even within one run,** the score swings about ±0.3 ft between checkpoints a few rounds apart.

### 5.2 Where the remaining error lives

Most of the error is **one wrong straight line per well**: the prediction drifts away at a steady
angle. Removing that line (which needs the true answer) would cut the error a lot:

| model | as is | minus one offset per well | minus one straight line per well |
|---|---|---|---|
| ConvNeXt | 4.94 | 3.76 | **3.07** |
| AnchorCNN | 6.02 | 4.15 | 3.27 |
| Our tracker | 15.85 | 9.49 | 5.48 |

We tried to **predict** each well's line in advance from everything available: the known part of
the well, the drilling path, the slopes of neighbouring wells, how confident the model was, and
the hidden constant. Nothing predicted it. The 22nd-place team hit the same wall.

### 5.3 The same wells defeat everything

- **Six wells** (`d7eb0be8`, `f2d4c8c9`, `f6bc699b`, `f6d009f4`, `f8afa78a`, `fb0904bd`) are in
  the worst 15 for all four network versions we compared, and carry **20–35%** of each one's
  error. The author's much better tracker also fails on them (13–20 ft).
- If we could magically pick the best model for each well, the score would be **3.96 ft**
  instead of 4.94. But every method we tried for *choosing* did worse than one fixed weight.
  155 wells aren't enough evidence to learn who to trust, and when.

### 5.4 The honest headline

> On this benchmark, most published and proposed improvements can't be told apart from luck
> when judged by a single run. We measured how big that luck is, where the remaining error
> lives, and which negative results are real.

## 6. Cost: running a model on a drilling rig

Measured on this laptop's GPU, for one well at a time, counting the network only. All
U-Net rows come from one session under identical conditions (22 September; see the note in
§4.1). "Merged" = training branches collapsed for inference (§6.3).

| model | adjustable numbers (whole model) | attention | time per well | trained here? |
|---|---|---|---|---|
| FastViT-T12 | 8.2 M | none | 16.7 ms (**12.0 merged**) | built, waiting (§9.1) |
| MobileOne-S1 | 10.2 M | none | 17.5 ms (**11.9 merged**) | yes: 6.51 |
| **FastViT-SA12** | 12.3 M | self-attention, last stage | 17.1 ms (**12.6–13.1 merged**) | yes: 5.21 alone, **4.84 taught** |
| ConvNeXt-nano | 17.8 M | none | 14.8 ms | built, waiting |
| **ConvNeXt-tiny** | 31.9 M | none | 18.4 ms | yes: 4.90 |
| InceptionNeXt-tiny | 28.1 M | none | 19.7 ms | built, waiting |
| FastViT-SA24 | 22.2 M | self-attention, last stage | 25.9 ms (**17.7 merged**) | built, waiting (§9.1) |
| ConvNeXt-small (the teacher) | 53.6 M | none | 26.1 ms | yes: 4.98 |
| ConvNeXt-small + axial attention | 59.8 M | axial attention | 29.2 ms | yes, 60 rounds: no gain |
| ConvNeXt-base | 94.8 M | none | 36.6 ms | no |
| AnchorCNN | 3.7 M | none | 14 ms network + 127 ms decoding = **141 ms** (measured earlier) | yes |

- **The AnchorCNN is small but not fast.** Its decoding step takes 90% of its time when wells are
  processed one at a time. Processing 8 at once cuts this to about 29 ms per well. Swapping its
  network for a cheaper one saves only about 10%.
- **Blending costs 5–6× the ConvNeXt alone** (plus a tracker running on the CPU), for 0.085 ft.
- **View-averaging (TTA) costs 8×** for 0.04 ft on the ConvNeXt.
- **ConvNeXt-tiny has been tested, and it loses nothing measurable.** It is the same design with
  fewer layers in one section and exactly the same pretraining: about a third less computing and
  ~27% faster. Its reported score was 4.90 vs 4.98 for small, but both were picked on the test
  wells (§7.7). The unpicked checkpoints disagree about which is ahead, and both gaps are under
  0.1 ft:

  | | small | tiny |
  |---|---|---|
  | last checkpoint | 5.09 | 5.14 |
  | average over rounds 100–150 | 5.16 | 5.09 |

  So: no measurable difference. Tiny's training was also steadier.
- **FastViT-SA12 was tested on 20 September, and it is worse.** 40 GFLOP, 20 ms, and it trained
  in 2.1 hours, but it lost on every measure: reported 5.21 vs tiny's 4.90, last checkpoint 5.30
  vs 5.14, late average 5.32 vs 5.09. The acceptance test rejected it against both other models
  (only 6% of resampling checks favoured it over tiny). Caveats: its pretraining used a smaller
  image collection than the ConvNeXts', and a 0.2–0.3 ft gap is near the limit of what this test
  can resolve.
  **Update, 21 September:** taught by ConvNeXt-tiny (§4.10), FastViT reached 4.93, almost
  level with tiny. So "cheaper backbone plus a teacher" is back on the table.
- **So the cost ladder stops at ConvNeXt-tiny.** Halving the arithmetic from tiny to FastViT
  (82 → 40 GFLOP) bought only ~13% real speed (23 → 20 ms), because at this image size the work
  is limited by memory traffic, not arithmetic. Paying 0.2 ft for 3 ms is a bad trade.
- **A related measurement, for "separable" or row/column proposals:** ConvNeXt spends just
  **1.5%** of its computing on 2-D spatial filtering, and 70% on mixing channels. Replacing the
  spatial filters cannot save much. We tested that design directly in the AnchorCNN: 11.4 ft
  against a 6.0–6.6 baseline.
- **MobileOne-S1 was tested, and it is far worse: 6.51 ft against tiny's 4.90.** That is a
  1.61 ft loss — four times the noise floor, so decisive. It would have been the fastest option
  (14.0 ms against tiny's 21.9, a 36% saving), and the branch-collapsing trick it relies on was
  verified exact, but the accuracy cost is not close to acceptable. Two fixes were made first so
  the comparison would be fair: its channel list counts its own first layer, and its first stage
  shrinks the image where ConvNeXt's does not, which would otherwise have made every feature map
  coarser.
- **Put together, the backbone results show a knee, and ConvNeXt-tiny sits on it:**

  | backbone size | model | score |
  |---|---|---|
  | 49.5M | ConvNeXt-small | 4.98 |
  | 27.8M | **ConvNeXt-tiny** | **4.90** |
  | 10.4M | FastViT-SA12 | 5.21 |
  | 3.6M | MobileOne-S1 | 6.51 |

  Halving the model from small to tiny is free — above about 28M the model is limited by how
  much data it has, not by its size. Below about 10M it falls off a cliff. **The cost ladder was
  closed for models trained on their own.** Teaching (§4.10) moves the knee: taught by tiny,
  the 10.4M FastViT scored 4.93 instead of 5.21.
- **ConvNeXt-base** (bigger) is not recommended: 1.7× the computing, no matching pretraining,
  likely won't fit in training on a 6 GB GPU, and the model is limited by data, not size.
- **Putting a ConvNeXt inside the AnchorCNN** would cost about 15× the AnchorCNN's normal
  computing (122 GFLOP).

### 6.1 Every replacement we screened

The 1st-place author had already built 10 transformer backbones plus two other model types, and
**kept ConvNeXt-small in all six submitted versions.** On top of that we screened:

| candidate | fits without code changes? | notes |
|---|---|---|
| **ConvNeXt-nano** (V1) | yes | **the fastest that fits** (49.7 ms vs tiny's 60.3 and FastViT's 53.0 in the same busy-GPU test), with the same strong pretraining as tiny. Weights downloaded; runs built (§9.1) |
| ConvNeXt-V2 tiny / nano | yes | V2 is slower than V1 at identical arithmetic (its extra normalisation layer): nano 70.7 vs 49.7 ms |
| InceptionNeXt-tiny | yes | slightly slower than tiny (64.9 ms), and only weaker ImageNet-1k weights exist. Weights downloaded; runs built (§9.1) |
| ConvFormer-S18, CAFormer-S18 | yes | ~101 ms, slower than tiny: no reason to try |
| MambaOut (femto / kobe) | no | a stem layout mismatch; would need adapter code |
| **FastViT-SA12** | yes | **trained: worse** (§6) |
| **MobileOne-S1** | needed a small fix, now done | **trained: 1.6 ft worse** (§6) |
| FasterNet | yes | fewer operations, **1.8× slower** than ConvNeXt-small |
| EfficientNet-B3/B4, EfficientNetV2 | no | and B3/B4 are **slower** than tiny (9.5 / 11.6 vs 7.2 ms) at a quarter of the arithmetic |
| Swin | no (but the author's transformer path supports it) | never used in any submitted version |
| MobileNetV4, RepViT, RDNet, Hiera, EdgeNeXt, EfficientViT, TinyViT, HGNetV2 | no | would need adapters |
| HRNet / Lite-HRNet | no | fails outright at our image size; Lite-HRNet not available |
| **DINOv2** | no | single-scale transformer built for 518×518 photos; needs a whole feature-pyramid adapter, and our input is a synthetic matching image, not a photo |
| FasterViT, CMT | not available | not in the library; would need new dependencies or writing from scratch |

### 6.2 Two rules this produced

1. **Judge candidates by measured milliseconds, never by "operations".** At our image size the
   work is limited by memory traffic. FasterNet used fewer operations and ran 1.8× slower;
   FastViT halved the arithmetic for 13% real speed; MobileOne's collapse trick changed the
   arithmetic by 2% and the speed by 3.7×.
2. **"Use a big encoder with a small decoder" is already the design.** The decoder's width is 32
   channels against the encoder's 768, and the projection layers that connect them already exist.
   Timed inside the model, the decoder parts hold about 0.05M numbers — their cost is the
   resolution they run at, not their width, so shrinking them further saves almost nothing.

**Recommendation: ConvNeXt-tiny, one pass, no blend, no averaging.** A third less computing and
38% faster training than the original, for no accuracy we can measure. Cheaper backbones than
tiny cost accuracy when trained on their own. The open question is whether a *taught* cheaper
model (ConvNeXt-nano or FastViT) can match or beat tiny (§4.10, §9.1).

(The timings in the §6.1 screen were taken while another training run shared the GPU, so only
their order is reliable. The idle-GPU numbers are in §6.3.)

### 6.3 Making inference faster without changing the model

We measured three common suggestions on the trained models, on an idle GPU, for one well, U-Net
only. Times are in ms per well:

| setting | ConvNeXt-tiny | FastViT |
|---|---|---|
| full precision (32-bit numbers) | 46.5 | 32.1 |
| **current: mixed 16-bit arithmetic (AMP, bf16) + memory layout tweak** | **25–26** | **22–23** |
| full 16-bit weights (fp16), normalisation layers kept at 32-bit | **22.0** | **19.0** |
| the same + "CUDA graphs" (pre-recorded GPU work) | 21.5 | 18.3 |

- **AMP (reduced precision): already on.** The configuration already uses 16-bit arithmetic
  (bf16), which is where the big 46 → 25 ms saving came from. The remaining free step is storing
  the whole model in fp16 (keeping the normalisation layers at 32-bit, otherwise it crashes).
  That is ~16% faster, and its output is actually *closer* to full precision than the current
  setting: a relative difference of 0.07% against 0.37%. No retraining is needed. It has not
  been wired into the scoring script or checked on the test wells yet.
- **Pruning the decoder: not worth it.** The decoder is about 8 of tiny's 25 ms. The costly parts
  are the ones working at full resolution, which the 2:1 rule (§6.2) says the model needs.
  Cutting them would need retraining and risk accuracy, to save perhaps 3–4 ms.
- **TensorRT: probably little gain, and a painful setup.** It is not installed, and neither is the
  library `torch.compile` needs on Windows. CUDA graphs capture the easy part of what TensorRT
  does (removing overhead between steps), and they saved only 0.5–2 ms. So the GPU is simply
  busy computing, and TensorRT's remaining advantage (merging steps) is maybe 10–30%, unmeasured.
- **Merging FastViT's training branches: built, and now automatic (22 September).** FastViT
  (like MobileOne) trains with extra parallel branches that can be collapsed into single layers
  once training is over. The trained student still had all 33 of them un-merged.

  | FastViT taught by small | network time | test error |
  |---|---|---|
  | as trained | 17.0 ms | 4.840 |
  | **merged** | **13.1 ms** | **4.832** |

  The error moved by 0.008 ft, which is float rounding (on plain FastViT it moved +0.016 the other
  way). Both are far below the 0.28 ft run-to-run noise. From now on, every run merges before
  its final scoring and saves the merged model (setting `reparam_at_inference`, on by default; it
  does nothing to ConvNeXt). Existing runs can be re-scored merged with
  `seq_NN_rescore.py --reparam` (§11).
- **The bigger picture, measured piece by piece (22 September, best student).** Each well
  goes through four steps. The network runs **once** per well:

  | step | where it runs | time per well |
  |---|---|---|
  | 1. neighbour-well map ("geo prior") | CPU | ~65 ms |
  | 2. building the input image (16 layers × 345 × 400) | CPU, 4 workers in parallel | **~100 ms** |
  | 3. the network | GPU | **~17 ms merged** (~21 as trained) |
  | 4. turning the output into depths, plus smoothing | CPU | ~4–6 ms |

  Steps 2 and 3 overlap: the CPU prepares the next well while the GPU works on the current one.
  So the GPU spends most of its time **waiting for the CPU**. The network is only about 10% of
  the ~180 ms total. Making it faster (merging, fp16, TensorRT) can only shrink that 10%. Steps
  1 and 2 are where real speed-ups would come from, and they haven't been profiled yet.
  Whole-pipeline timings are also noisy: the same model measured 105–170 ms per well on
  different runs, depending on what else the laptop was doing. (An earlier version of this
  report said a merging saving of 11 ms came from "several network passes per well". That was
  wrong: there is one pass per well, and the 11 ms was noise.)

### 6.4 Attention: where it is in every model we tried

"Attention" is a kind of layer that lets each position in the input look at every other
position and decide which ones matter. Convolutions only look at a small neighbourhood.
Attention is powerful but expensive: its cost grows with the *square* of the number of
positions, so on a large image it is usually only used where the image has been shrunk.

| model | attention? | where and what kind | effect we measured |
|---|---|---|---|
| ConvNeXt-small / tiny / nano / base | **none** | only convolutions (large 7×7 filters) | the family behind the best results |
| **FastViT-SA12** (our best student) | **yes, in its last stage only** | 2 ordinary self-attention blocks on a 22 × 13 grid (286 positions, each covering ~16 rows along the well × ~31 depth levels). The first three stages use cheap convolutions ("RepMixer"). | 1.6 ms of 19.6 (8%). Whether it helps accuracy is untested; the T12 run answers it |
| FastViT-SA24 | yes, last stage only | 4 attention blocks on the same 22 × 13 grid | built, waiting |
| FastViT-T12 | **none** | SA12 with its attention stage replaced by convolutions | built, waiting (the attention test) |
| MobileOne-S1 | none | convolutions only, merged for inference | 6.51: too small |
| InceptionNeXt-tiny | none | convolutions split into several shapes | built, waiting |
| **ConvNeXt with axial attention** (1st-place redesign we ran) | **yes** | 3 "axial" blocks that attend along one direction at a time (along the well), which is cheaper than full attention | **no gain**: −0.024 ft, smaller than a pure-luck difference |
| The 1st-place author's 10 transformer backbones | yes | full or windowed attention | not trained by us; the author kept ConvNeXt in all 6 submissions |
| CAFormer / ConvFormer | CAFormer: yes (last two stages); ConvFormer: none | — | screened only: slower than tiny |
| DINOv2 | yes, everywhere | a plain transformer on 14 × 14-pixel patches: one coarse 25 × 29 grid | rejected: throws away the fine detail this task needs (§6.1) |
| "Bottleneck" attention (from a guide we reviewed) | — | attention inserted in the coarsest stage | not built: it is essentially what FastViT-SA12 already has |
| AnchorCNN (2nd place) | none | convolutions + a step-by-step decoder | — |
| Particle filter / HMM | not networks | — | — |

**What this adds up to.**
- **None of our gains came from attention.** The two things that clearly worked were teaching
  (§4.10) and averaging. The one experiment that added attention on purpose (axial) gained
  nothing measurable.
- **Attention is cheap where FastViT uses it.** Changing it can save at most ~1.6 ms.
- **The accuracy lever for cheap models is capacity, not attention.** The size curve (§6) says
  models below ~28M are limited by their size. Hence the FastViT-SA24 student (twice SA12's
  capacity, about the same cost once merged) and the T12 test (does the attention matter at all?).

## 7. Caveats that apply to the results

1. **The AnchorCNN has not been properly tested with synthetic data.** Every AnchorCNN run
   except one had it switched off. The original 2nd-place recipe was about **77% synthetic**
   wells, and the ConvNeXt always trained with 85% synthetic samples, so the comparisons between
   the two were lopsided.

   The one synthetic run (`C_synth_s1`) was flawed twice. A bug meant it asked for 77% synthetic
   wells but got only **32%**: most attempts to make one were rejected, and the code gave up
   after one try (now fixed, it delivers 73%). It also did 2.3× as many training steps per
   round as its comparison run, so it overfitted within the same number of rounds.

   Even so, it led the plain run at 18 evaluations in a row, and its average over the second
   half of training (6.11 ft) was better than all three plain runs (6.26–6.38). Its final score
   (6.30) sits inside the plain runs' normal range (6.02–6.57), because it degraded over its
   last 30 rounds. So synthetic data is **leaning positive but unresolved**.

   What could change once this is tested properly:
   - If synthetic data brought the AnchorCNN to about 5.0 ft, the blend could reach roughly
     **4.5–4.7 ft** instead of 4.85, which would make blending worth reconsidering.
   - "The AnchorCNN stops improving after about 100 rounds" may only be true without synthetic data.
   - Results that don't depend on this: the cheaper-design failures, int8, the gates and stacks,
     the error-line analysis (it also failed on the ConvNeXt, which did use synthetic data), and
     all tracker results.
2. **Only 155 test wells.** Small effects can't be seen (§5.1). "Unresolved" results may be real
   gains or real losses.
3. **Mostly one run per setting.** Only the AnchorCNN control (3 seeds) and the tracker inputs
   (2 seeds) have repeats. The ConvNeXt's own seed-to-seed noise has never been measured.
4. **Some settings were tuned on all 773 wells** by the original authors, including our test
   wells: the 1st-place model's scaling constants and its tracker settings. The effect is
   probably small, but it isn't zero.
5. **`model_best.pt` must never be reported.** The training script saves the checkpoint that
   scored best on the test wells, which is choosing by the answer. Every **AnchorCNN** number in
   this report comes from the final checkpoint (`model_last.pt`). The ConvNeXt numbers are a
   different story; see item 7.
6. **The tracker is random.** It draws a different random pattern each time the program starts
   unless `PYTHONHASHSEED` is fixed, so repeat runs differ slightly. Tracker results in this
   report were repeated to account for this.
7. **Every ConvNeXt score was picked using the test wells** (found 19 September). In our test
   setup, the 1st-place code checks its progress on the 155 test wells and keeps whichever
   checkpoint scored best there. In every run checked, that picked checkpoint is **about 0.14–0.15
   ft better** than the same run's average over its last 50 rounds. What this means:
   - ConvNeXt runs can still be compared with each other fairly, since all were picked the same way.
   - ConvNeXt-vs-AnchorCNN comparisons were slightly tilted in the ConvNeXt's favour, because the
     AnchorCNN numbers are unpicked.
   - The blend scores (4.85) use a picked ConvNeXt, so they are somewhat flattering too.

   The earlier project notes wrongly said checkpoint choice was safe. The script
   `seq_NN_honest_curve.py` reads a run's log and prints the unpicked alternatives (last
   checkpoint, and average over late rounds).

## 8. Mistakes we made, and what we corrected

| mistake | what happened | correction |
|---|---|---|
| Far-anchor bug | Every training sample from a well was identical, and the run degraded for 180 rounds | Fixed. New rule: check sample variety before training. |
| Reading noise as a trend | "Still improving at round 114" was a ±0.3 ft wobble; it led to a wasted 250-round run | The AnchorCNN actually peaks around round 100 |
| Three changes in one run | When that run failed, it took detective work to find out why | Rule: one change per run |
| "Tracker inputs fixed the hard wells" | Plain runs alone range 9.1–25.2 ft on the key well, so this was luck | Claim withdrawn |
| "Siblings conclusively hurt the AnchorCNN" | One tangled run, smaller than the noise | Now called unresolved |
| First tracker tuning moved a setting the wrong way | Nothing improved until we measured and reversed it | Fixed (×9) |
| A tracker-inputs cache first masked the last column | Up to 49 ft wrong on 0.3% of columns | Fixed; now identical to 0.000 ft |
| Output files landed in the wrong folder | A script changed directory before saving | Moved; nothing overwritten |
| Blend score first quoted as 4.834 | That number was tuned on the same wells it was scored on | The honest cross-checked figure is 4.852 |
| Synthetic generator gave up after one try | Asked for 77% synthetic wells, delivered 32% | Now retries up to 12 pairings; delivers 73% |
| Synthetic run changed two things | It also did 2.3× the training steps, so it overfitted within the run | The proper test (§9.1, run 1) matches steps with its comparison run |
| Called the synthetic run "worse" from its final score alone | The final score is inside the normal range, and the whole curve leaned the other way | Judge on the curve's average as well as the endpoint |
| Treated ConvNeXt scores as unpicked | The 1st-place code keeps the checkpoint that scores best on the test wells, and the notes wrongly called this safe | Flattery measured at ~0.15 ft; unpicked scores now read from the logs (§7.7) |
| Called MobileOne's branch-collapsing "broken" | Compared raw difference sizes on an untrained network whose internal values reach 500 million, so a large-looking gap was actually 5 parts in 10 million | Checked as a *proportion* instead: it is exact |
| Wrote off teaching FastViT as "not worth it" | Judged on its 3 ms prize alone, without asking how much accuracy a teacher could return | Run anyway: 5.21 → 4.93 (§4.10). Rule: a teacher is worth trying whenever it is *measurably* better than the student |
| Predicted a taught model would be a worse averaging partner for its teacher | Reasoned from the rise in shared mistakes alone | Measured: tiny + its student is the best pair (4.77). Accuracy gains can outweigh a rise in shared mistakes |
| Suggested pseudo-labelling synthetic wells with a teacher | The ConvNeXt already trains on 85% synthetic wells with exact labels, and distillation already runs the teacher on them | Dropped as redundant |
| Recommended a 6.5-hour run without asking whether its effect was measurable | The 1st-place synthesis refinements are worth about 0.1 ft, below what 155 wells can resolve | Dropped; the same question on the AnchorCNN is a 0%-vs-73% contrast and takes 75 minutes |

## 9. What is still open

### 9.0 Does any of it still matter?

Worth stating plainly before the list. **The deployment question is answered** (§6), and nothing
below changes it:

- The AnchorCNN only matters through the blend, and the blend costs 5–6× the inference for
  0.085 ft, so it has been set aside.
- Anything that makes the AnchorCNN cheaper or better therefore improves a model we are not
  planning to deploy.

**The exception is the teaching runs (run 0 below).** A taught ConvNeXt-nano or FastViT that
matches tiny would be cheaper to run on a rig, and a taught tiny that keeps the small + tiny
averaging gain would be more accurate at the same cost. Those runs *can* change the deployment
choice. Everything else is worth doing for **completeness or a write-up**, not for the rig. If the
goal is a write-up, the highest-value work is not another single run — it is making the existing
claims measurable (§9.4), because single runs keep returning "unresolved".

### 9.1 Runs waiting, most important first

The two cost runs that used to head this list (ConvNeXt-tiny, FastViT) are finished; see §6.

**0) The teaching runs (most important; they can change the rig choice).** All are built, and
the weights they need are downloaded. Run from `kaggle_1st_place\solution`, one at a time; each
takes roughly 3–4 hours. Every command ends with
`--device cuda --offline-timm --batch-size 4 --val-batch-size 4 --grad-accum-steps 4`, written
out in full below. If the GPU runs out of memory, use `--batch-size 2 --grad-accum-steps 8`
instead; it trains on the same amount of data per step.

In order of importance:

a) ConvNeXt-nano on its own (the comparison run for b), then nano taught by the small + tiny average:
```
python seq_NN_holdout_eval.py --id 0801_V2 --source-dir experiments/bilzard --cfg-name cnx_nano --epochs 150 --output-dir results/cnx_nano_ep150 --device cuda --offline-timm --batch-size 4 --val-batch-size 4 --grad-accum-steps 4
```
```
python seq_NN_holdout_eval.py --id 0801_V2 --source-dir experiments/bilzard --cfg-name cnx_nano_kd_ens --epochs 150 --output-dir results/cnx_nano_kd_ens_ep150 --device cuda --offline-timm --batch-size 4 --val-batch-size 4 --grad-accum-steps 4
```
b) ConvNeXt-tiny taught by the small + tiny average (its comparison run is `results/cnx_tiny_ep150`, already done):
```
python seq_NN_holdout_eval.py --id 0801_V2 --source-dir experiments/bilzard --cfg-name cnx_tiny_kd_ens --epochs 150 --output-dir results/cnx_tiny_kd_ens_ep150 --device cuda --offline-timm --batch-size 4 --val-batch-size 4 --grad-accum-steps 4
```
c) FastViT taught by the small + tiny average (compare with `cnx_fastvit_kd_ep150`, which was taught by tiny alone):
```
python seq_NN_holdout_eval.py --id 0801_V2 --source-dir experiments/bilzard --cfg-name cnx_fastvit_kd_ens --epochs 150 --output-dir results/cnx_fastvit_kd_ens_ep150 --device cuda --offline-timm --batch-size 4 --val-batch-size 4 --grad-accum-steps 4
```
d) Lowest priority: InceptionNeXt-tiny on its own, then taught by ConvNeXt-small (`inx_tiny`,
`inx_tiny_kd_small`; same command pattern). A weak case, for the reasons in §4.10.

Always compare them with the checkpoints nobody picked:
```
python seq_NN_honest_curve.py results/cnx_tiny_ep150 results/cnx_nano_ep150 results/cnx_nano_kd_ens_ep150 results/cnx_tiny_kd_ens_ep150 results/cnx_fastvit_kd_ep150 results/cnx_fastvit_kd_ens_ep150
```
How to read them: a taught model counts as matching tiny if its last checkpoint and its late
average are both within about 0.1 ft of tiny's (5.14 / 5.09). One run each, so a result near the
line needs a second seed before it decides anything.

c2) **Done** (`results/cnx_fastvit_kd_small_ep150`, 4.84; §4.10). **Next: repeat it and its
comparison run with new random seeds** (the paper needs at least three of each, §9.5). The
original command, for reference:
```
python seq_NN_holdout_eval.py --id 0801_V2 --source-dir experiments/bilzard --cfg-name cnx_fastvit_kd_small --epochs 150 --output-dir results/cnx_fastvit_kd_small_ep150 --device cuda --offline-timm --batch-size 4 --val-batch-size 4 --grad-accum-steps 4
```

c2b) **FastViT-SA24 taught by ConvNeXt-small** (the capacity attempt; weights downloaded). Merged,
it runs in 17.7 ms, about the un-merged SA12's cost, with twice the capacity. Then **FastViT-T12
taught by small** (the attention test: identical, but with no attention stage):
```
python seq_NN_holdout_eval.py --id 0801_V2 --source-dir experiments/bilzard --cfg-name cnx_fastvit_sa24_kd_small --epochs 150 --output-dir results/cnx_fastvit_sa24_kd_small_ep150 --device cuda --offline-timm --batch-size 4 --val-batch-size 4 --grad-accum-steps 4
```
```
python seq_NN_holdout_eval.py --id 0801_V2 --source-dir experiments/bilzard --cfg-name cnx_fastvit_t12_kd_small --epochs 150 --output-dir results/cnx_fastvit_t12_kd_small_ep150 --device cuda --offline-timm --batch-size 4 --val-batch-size 4 --grad-accum-steps 4
```
SA24 is twice as deep, so training takes longer (roughly 4–5 h). If it runs out of GPU memory,
use `--batch-size 2 --grad-accum-steps 8`. Both runs merge their branches automatically before
scoring.

c3) **Repeats for the paper** (recipes built; the seed is the only change, and the test wells
stay the same). The two FastViT pairs matter most: about 2 h for FastViT alone and about 3 h for
the taught one. The teacher repeats are optional (~5 h each) and show how much the teacher itself
varies. Replace `NAME` in the command below with each of these:
`cnx_fastvit_s11`, `cnx_fastvit_kd_small_s11`, `cnx_fastvit_s23`, `cnx_fastvit_kd_small_s23`,
then optionally `small_s11`, `small_s23`.
```
python seq_NN_holdout_eval.py --id 0801_V2 --source-dir experiments/bilzard --cfg-name NAME --epochs 150 --output-dir results/NAME_ep150 --device cuda --offline-timm --batch-size 4 --val-batch-size 4 --grad-accum-steps 4
```
The students keep the original seed-7 ConvNeXt-small as their teacher.

**0b) Free speed-up, not yet done:** make the scoring script store the model in fp16 (keeping the
normalisation layers at 32-bit, §6.3), and check the test score doesn't move.

**1) AnchorCNN with synthetic data, done properly.** The first attempt was flawed (§7.1). This
version changes only one thing (synthetic data on), with the generator bug fixed and the same
number of training steps as its comparison run. About 1 h 15. Run from `kaggle2ndplace`:
```
python -u anchor_train.py --out runs/C_synth2_s1 --row 1.0 --n-move 5 --epoch-len 1150 --epochs 120 --eval-every 5 --tta 8 --seed 1 --synth-prob 0.77
```
```
python ../kaggle_1st_place/solution/seq_NN_robust_compare.py --base runs/C_s1 --treat runs/C_synth2_s1 --output-dir runs/C_synth2_s1
```
How to read it. Compare both the final score (against 6.024) and the average of the evaluations
from round 60 onward (against the plain runs' 6.26–6.38):
- clearly better on both: synthetic data helps, and blending is worth reconsidering
- final score inside 6.02–6.57 but a better average: helps during training, still overfits by
  the end; stopping earlier needs a rule that doesn't peek at the test wells
- no better on either: the gap to the ConvNeXt really is the model design

**2) Tracker inputs, third seed.** Confirms or rejects the best AnchorCNN result. About 1 h 15.
Run from `kaggle2ndplace`:
```
set "PYTHONHASHSEED=0"
```
```
python -u anchor_train.py --out runs/C_s3_pfchan --row 1.0 --n-move 5 --epoch-len 1150 --epochs 120 --eval-every 5 --tta 8 --seed 3 --pf-channels 2
```
Compare against `runs/C_s3` (6.200 ft). Three wins out of three would be meaningful evidence.

**3) Copying the 8-view average into a single-view model** (knowledge distillation, not yet
built). This is the one accuracy idea left that targets a gap **bigger than the noise**: view
averaging is worth 0.26–0.56 ft on every AnchorCNN run and costs 8× the inference, so teaching a
single-view model to imitate it would pay that back. Two caveats: our earlier attempt at the same
goal (a smoothing filter) failed, and our existing distillation code copies the model's internal
step predictions, which live on different grids for each view — the teacher's signal has to be
the final decoded path instead. On the ConvNeXt the same idea isn't worth it (averaging is only
worth 0.04 ft there).

**Closed since the last version:** MobileOne-S1 was trained and lost 1.6 ft (§6). It is too far
behind to rescue with a teacher. (An earlier version of this report also called teaching FastViT
"not worth it". It was run anyway and recovered about 90% of its gap (§4.10), which reopened the
cheaper-model line, see run 0.)

**Also dropped:** the 1st-place synthesis refinements at 300 rounds (`cnx_tiny_synth`,
registered and ready). The baseline already generates synthetic wells for 85% of its samples, so
that run tests three *refinements* worth about 0.1 ft — below what 155 wells can resolve, for
6.5 hours of GPU.

### 9.2 Never tested at all (including some we once thought were)

| item | status |
|---|---|
| Synthetic data in the AnchorCNN, done properly | Tried once, but flawed (§7.1). Run 1 in §9.1 is the proper test. |
| ConvNeXt inside the AnchorCNN | Built, never run. "Is the 1 ft gap just the network?" is open. |
| Sibling reference in the AnchorCNN on its own | Only tried on top of tracker inputs |
| Separating the two changes in "C" (grid vs step sizes) | Never run, so "C is free" can't be credited to either alone |
| Fixed far-anchor training (`--far-anchor 0.5`) | Never run |
| Large-kernel ConvNeXt with its bug fixed | Never re-run |
| Neighbour-well inputs trained longer (ConvNeXt, 300 rounds) | Never run; was undertrained |
| More synthetic data trained longer (ConvNeXt, past 150 rounds) | Never run; was still improving |
| HMM tracker on the test wells | Never scored |
| The ConvNeXt's seed-to-seed noise | Never measured |
| Cross-rock-system synthetic wells (`--synth-prob`) at the intended share | Used once at only 32% because of a bug; fixed (now delivers 73% when 77% is asked for) |
| Sibling disagreement as an input (`--sibling-sigma`) | Built, never trained with |
| Tracker inputs in the ConvNeXt (`pf_v1`) | Set up, never run; low priority (the tracker agrees with the ConvNeXt and fails the same wells) |

### 9.3 Tested, but needs a proper redo

| test | what a proper test needs |
|---|---|
| Far-anchor | The fixed version, alone, against its matching control |
| Distillation | An ensemble that clearly beats a single model to copy from |
| "Twice the data" | Same seed, only the amount of data changed |
| Resolution comparisons | 2–3 seeds of the original baseline, which was unusual on one well |
| 1st-place redesigns | 150 rounds to match the baseline, not 60 |

### 9.4 Make the measuring stick sharper

1. **More seeds.** Run important settings 3 times. Three wins out of three means something;
   one win doesn't. Costs 3× the GPU time.
2. **5-fold cross-validation over all 773 wells.** Every well is tested once, giving 5× as many
   test wells. The AnchorCNN trainer would need a `--fold` option (not built yet). This only
   works for AnchorCNN-only questions, because matching ConvNeXt predictions don't exist.
3. **Look at the typical well as well as the total.** The median well's error moves about half
   as much with luck as the total does.

### 9.5 Before writing a paper

**The strongest paper now: "a cheap student taught by ConvNeXt-small matches its teacher at 40%
less cost."** One clean story, with the negative results as supporting context:

| | error (ft) | network time | parameters |
|---|---|---|---|
| ConvNeXt-small (the 1st-place model's network; teacher) | 4.98 | 26.1 ms | 53.6 M |
| FastViT-SA12 alone | 5.21 | 12.6 ms (merged) | 12.3 M |
| **FastViT-SA12 taught by ConvNeXt-small** | **4.84** | **13.1 ms (merged)** | **12.3 M** |

Why it is a sound paper topic:
- It is practical: rig computers are limited, and this is 4× fewer parameters and ~2× faster
  (network only, branches merged) for no accuracy loss. But the whole pipeline is ~180 ms per
  well either way, mostly CPU-side input preparation (§6.3). The paper must state both numbers,
  or speed up the input preparation too.
- The gain over the untaught cheap model (0.37 ft) is large enough to pass our check on one run.
- The negative results explain *why* this is the right route: bigger models don't help, because
  the data runs out before the model size does (§6). Cheaper models lose accuracy on their own
  (§6), and averaging or blending cost 2–6× the inference (§4.7, §4.10).

**Leaving ConvNeXt-tiny out is fine.** Including it doesn't hurt the story: the taught FastViT
(4.84, 22 ms) is at least as good as tiny (4.90, 25 ms), and a reviewer who asks "why not just a
smaller ConvNeXt?" can be answered with that row. Leaving it out entirely would invite that question.

What is needed before submitting:
- [ ] **Repeats:** at least 3 seeds each of the teacher, FastViT alone and the taught FastViT.
      One run can't separate a 0.14 ft gap from luck (§5.1). Budget: ~6 runs × ~3 h.
- [ ] **Unpicked scores as the headline** (last checkpoint or late average), because the reported
      numbers pick the checkpoint on the test wells (§7.7).
- [ ] **Small ablations:** teaching strength (weight 0.5 / 1 / 2) and temperature (1 / 2 / 4).
      Both are untuned at 1 and 2.
- [ ] **Full-pipeline timing,** not only the network (the network is ~1/3 of time per well, §6.3).
- [ ] State plainly that knowledge distillation itself is a standard method. The contribution is
      showing it works on this task with only 618 training wells, where a student matches a teacher 4× its size.

A second, smaller paper about **what didn't work, and why the test can't tell**, is also
realistic, for example at a workshop that welcomes negative results. A paper claiming "this is
the best possible" is not, because others scored better with more computing.

- [ ] Check the data licence: Kaggle competition data often can't be used outside the competition
- [ ] Credit Ruby (1st place) and Bilzard (2nd place) clearly, plus the 3rd, 8th and 22nd-place write-ups
- [ ] Back the key claims with seeds or cross-validation (§9.4)
- [ ] State the synthetic-data caveat (§7.1)
- [ ] Use the headline in §5.4 as the single main claim

## 10. Lessons and rules of thumb

1. **One change per run**, compared with a matching control at the same training length.
2. **Measure the noise before believing a result.** Here it was ±0.39 ft per comparison, and
   most changes were smaller than that.
3. **Look at which wells a gain comes from.** If it disappears when one or two wells are
   removed, it's probably luck.
4. **Check training samples before training** after any change to how they are made: are they
   varied, and does a setting deliver what it says? (Far-anchor made identical samples;
   "77% synthetic" delivered 32%.) Both checks take under a minute.
5. **Don't read a single run's curve as a trend.** It wobbles by ±0.3 ft.
6. **Simple beats clever with small data.** One blend weight beat every learned chooser.
7. **For a partner model, being different matters more than being accurate.** Making a partner
   more accurate can make it more similar, and so less useful.
8. **Check what a baseline actually trains on** before comparing against it. The synthetic-data
   gap went unnoticed for most of the project.
9. **Verify claims against the files.** Several things we believed had been tested had never
   been run.
10. **Teaching needs a teacher that is measurably better.** It failed on the AnchorCNN, where the
    teacher was no better than the student. It worked on FastViT, where the gap had been measured first.
11. **Time the whole pipeline before optimising a part.** The network is only about a third of the
    time per well, so even a perfect network speed-up is capped.
12. **Measure a partner, don't predict it.** "More similar mistakes means a worse partner" sounded
    right and was wrong for the taught FastViT (§4.7). Averages of finished models cost nothing
    to score, so score them.

## 11. Practical notes and where everything lives

**Running things**
- The user runs commands in **cmd.exe**. Set environment variables with `set "NAME=value"` on
  their own line. `set NAME= 0` (with a space) or `set NAME=0 && python ...` put stray spaces
  into the value, and Python refuses to start.
- 1st-place commands run from `kaggle_1st_place\solution`; AnchorCNN commands run from `kaggle2ndplace`.
- The 1st-place model needs `--batch-size 4 --grad-accum-steps 4` on this GPU. The AnchorCNN
  gets slow above about 4 GB of GPU memory, so keep its default batch size.
- **Downloading from HuggingFace fails** on this machine (a certificate error in one library).
  Downloading with `urllib` or `curl` works. Stored locally, with checksums verified:
  ConvNeXt-small, ConvNeXt-tiny, ConvNeXt-nano, FastViT-SA12, FastViT-SA24, FastViT-T12 and
  InceptionNeXt-tiny.
- Any script that starts worker processes on Windows must wrap its code in
  `if __name__ == "__main__":`, or it hangs.

**Where the details are**

| file | contents |
|---|---|
| `kaggle2ndplace/TRAINING.md` | Every AnchorCNN, tracker and blend experiment, with full numbers and commands |
| `kaggle_1st_place/CLAUDE.md` | Every ConvNeXt experiment, the repo layout and machine quirks |
| `kaggle_1st_place/REPORT.md` | A readable account of the 1st-place porting work |
| `kaggle_1st_place/EXPLAINER.md` | A from-scratch explanation of how the ConvNeXt model works |
| `kaggle2ndplace/runs/` | Every AnchorCNN run: settings, logs, predictions |
| `kaggle_1st_place/solution/results/` | Every ConvNeXt run |
| `kaggle_1st_place/solution/experiments/bilzard/seq_NN_cfg.py` | The ConvNeXt variants: `cnx_tiny`, `cnx_base`, `cnx_nano`, `cnx_fastvit`, `cnx_mobileone`, `inx_tiny`, `cnx_tiny_synth`, `pf_v1`, and the taught ones: `cnx_fastvit_kd`, `cnx_tiny_kd_ens`, `cnx_fastvit_kd_ens`, `cnx_fastvit_kd_small`, `cnx_fastvit_sa24_kd_small`, `cnx_fastvit_t12_kd_small`, `cnx_nano_kd_ens`, `inx_tiny_kd_small` |
| `kaggle_1st_place/solution/experiments/bilzard/seq_NN_train.py` | The training loop, including teaching: `load_distill_teacher` loads one or more finished runs as frozen teachers, and `distill_alignment_loss` makes the student imitate their averaged opinion |
| `kaggle_1st_place/solution/seq_NN_rescore.py` | Re-scores a finished run from its saved model without retraining. `--reparam` merges FastViT/MobileOne branches first; it also prints the full-pipeline time per well. Example: `python seq_NN_rescore.py --id 0801_V2 --models-dir results/cnx_fastvit_kd_small_ep150 --source-dir experiments/bilzard --cfg-name cnx_fastvit_kd_small --output-dir results/cnx_fastvit_kd_small_rescore_reparam --device cuda --offline-timm --val-batch-size 4 --reparam` |
| `kaggle_1st_place/solution/seq_NN_honest_curve.py` | Reads a run's log and prints the checkpoints nobody picked (§7.7) — use this, not the headline score, to compare ConvNeXt runs |

**Main code we wrote (in `kaggle2ndplace/`)**

| file | what it does |
|---|---|
| `anchor_data.py` | Loads wells, splits train/test, builds and varies training samples, generates synthetic wells |
| `anchor_train.py` / `anchor_eval.py` | Train and score the AnchorCNN |
| `anchor_ensemble.py` | Blends the two models, with the gates |
| `anchor_pf.py` / `anchor_hmm.py` | The two trackers |
| `anchor_pfchan.py` | Feeds the tracker's output into the AnchorCNN |
| `anchor_sibling.py` | Rock-system grouping and sibling references |
| `anchor_dip.py` / `anchor_diptest.py` | The per-well error-line corrector and its test |
| `anchor_separable.py`, `anchor_quantize.py`, `anchor_convnext.py` | The cheaper-design, int8 and ConvNeXt-inside experiments |

## 12. Glossary

| term | meaning |
|---|---|
| **TVT** | The drill's position within the stack of rock layers, in feet. What we predict. |
| **GR (gamma ray)** | The natural radioactivity of the rock; the main sensor reading. |
| **Typewell** | GR readings from a nearby vertical well, used as a reference fingerprint of the layers. |
| **Horizontal well / lateral** | A well drilled sideways through a rock layer. |
| **Holdout / test wells** | The 155 wells kept aside and never trained on, used only for scoring. |
| **RMSE / error** | Average error in feet, counting large mistakes extra. Lower is better. |
| **Model / network** | A program that learns to make predictions from examples. |
| **Training round (epoch)** | One pass of training over the data. |
| **Seed** | The random starting point of training. Same settings with a different seed give a slightly different model. |
| **Noise floor** | How much scores move from luck alone. Smaller effects can't be seen. |
| **Control** | The same run without the change being tested, for a fair comparison. |
| **k\*** | How many of the most influential test wells you must remove before an improvement disappears. |
| **Bootstrap** | Re-scoring on random re-samples of the test wells to see how often a result holds. |
| **Cross-checked (cross-fitted)** | Tuned on some wells and scored on others, so nothing is judged on wells it was tuned on. |
| **Synthetic data** | Computer-generated wells made from real ones, to give a model more variety. |
| **TTA (view-averaging)** | Predicting on several slightly shifted versions of the input and averaging them. |
| **Blend / ensemble** | Combining the predictions of several models. |
| **Gate** | A learned rule that chooses between models point by point. |
| **Stacking** | Fitting weights for many models at once. |
| **Particle filter / HMM** | Trackers that follow the drill step by step using simple physics, rather than learning from examples. |
| **Tracker inputs (PF channels)** | A tracker's guess and confidence, fed into a network as extra inputs. |
| **Siblings** | Wells whose typewells come from the same master log, so they drill the same rock. |
| **ConvNeXt (nano / tiny / small / base)** | A family of image networks; the name gives the size. |
| **AnchorCNN** | The 2nd-place model: predicts step-by-step changes, then decodes a path. |
| **Backbone** | The main image-recognition part of a network. |
| **GFLOP** | Billions of arithmetic operations; a rough measure of computing cost. |
| **int8** | Storing a network's numbers in 8 bits to save space and, sometimes, time. |
| **Overfitting** | A model memorising its training examples instead of learning general patterns. |
| **Reparameterisation** | Training a network with extra parallel branches, then mathematically folding them into one plain filter for deployment. Same answers, less memory traffic, so it runs faster. |
| **Knowledge distillation** | Training a small, fast model to imitate a bigger or slower one, so it inherits the better model's behaviour at the cheaper model's speed. |
| **Teacher / student** | In distillation, the finished model being imitated (teacher) and the model being trained to imitate it (student). |
| **AMP / fp16 / bf16** | Doing the arithmetic with 16-bit numbers instead of 32-bit. It is roughly twice as fast on this GPU and, done carefully, just as accurate. fp16 and bf16 are two 16-bit formats. |
| **Attention** | A network layer that lets each position look at every other position and weigh which ones matter. Powerful, but its cost grows with the square of the number of positions. |
| **Merging / reparameterisation** | Some networks (FastViT, MobileOne) train with several parallel branches per layer, which can be folded into one layer after training. Same answers, less work. |
| **CUDA graph** | Recording the GPU's whole sequence of work once and replaying it, which removes the small overhead of starting each step separately. |
| **TensorRT** | NVIDIA's tool that rewrites a trained network into a faster form for one specific GPU, mainly by merging steps. |
| **Stereo matching / disparity** | Working out how far something shifts between two views of a scene. The closest named computer-vision problem to this task (§2.6). |
| **Cost volume** | An image of "how badly does each candidate depth match here", which the network then cleans up. |
