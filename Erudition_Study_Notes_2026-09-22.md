# Erudition — Study Notes

*22 September 2026. Everything we tested, with its result, for a reader who knows basic computer
science but not machine learning. Full evidence and commands: `Erudition_Report_2026-09-22.md`.*

---

## 1. The problem

- A drill moves **sideways** through underground rock layers. It can't see where it is.
- Its one sensor is **gamma ray (GR)**: the rock's natural radioactivity. Each layer has a different
  GR level, so GR works as a fingerprint.
- A **typewell** is the GR log of a nearby vertical well. It shows every layer once, in order.
- **Task:** for the unknown part of each well, predict the drill's depth within the layers (**TVT**,
  in feet) at every point.
- **Why it's hard:** layers repeat, so a wrong position can match the GR as well as the right one.
- **Data:** 773 wells. Models train on 618 and are tested on the other 155, which they never see.

## 2. How results are measured

- **Error** = root-mean-square error in feet over all 754,122 test points. Lower is better. Big
  mistakes count extra.
- **Baseline:** assuming the drill stays at its last known depth gives 9.8 ft.
- **Noise:** training the same model again with a different random seed changes the error by
  ±0.28 ft. Comparing two single runs therefore carries ±0.39 ft of pure luck.
- **Two checks decide whether a difference is real:**
  - **k\*:** remove the test wells that help the change most, one at a time. k\* is how many must
    go before the gain disappears. **k\* ≥ 10 required.**
  - **Bootstrap:** re-score on 2,000 random re-samples of the 155 wells. **≥ 90% must agree.**
- **Checkpoint trap:** the 1st-place code keeps whichever saved model scored best *on the test
  wells*. Its reported scores are therefore too good, by 0.07–0.15 ft. Comparisons between its runs
  stay fair, because every run was picked the same way.

**Verdict legend** (the same in the report):

| mark | meaning |
|---|---|
| ✅ | **Confirmed:** passes both checks (or equally strong older evidence) |
| 🟡 | **Probable:** k\* ≥ 10, but only 70–90% of re-samples agree |
| ⚪ | **Not shown:** can't be told apart from luck. Not the same as "no effect" |
| ❌ | **Confirmed worse** |
| ⚠️ | **Invalid test:** the experiment had a bug |
| ⏳ | **Built, not run yet** |

## 3. Words you need

| term | meaning |
|---|---|
| **Model / network** | A program with millions of adjustable numbers (parameters), tuned by training on examples. |
| **Training round (epoch)** | One pass over the training data. |
| **Seed** | The random number that starts training. A different seed gives a slightly different model. |
| **Overfitting** | Memorising training examples instead of learning the pattern. |
| **Backbone** | The feature-extracting core of a network, downloaded pre-trained on millions of photos. |
| **U-Net** | A network that shrinks an image to understand it, then enlarges it back to answer at every pixel. |
| **Synthetic data** | Computer-generated training wells with exact answers. |
| **TTA** | Predict on 8 slightly shifted copies of the input and average the answers. |
| **Averaging (ensemble)** | Run several models and average their predicted depths. |
| **Distillation (teacher → student)** | Train a cheap network (student) to copy a better network's (teacher's) full predictions. |
| **Attention** | A layer where every position looks at every other position. Its cost grows with the square of the input size. |
| **Merging (reparameterisation)** | Some networks train each layer as several parallel branches; after training these are added into one layer. Same answers, less work. |
| **AMP / 16-bit** | Doing arithmetic with 16-bit numbers instead of 32-bit: about 2× faster. |
| **Network time** | Milliseconds for the network alone to process one well (measured 22 September, one session). |

## 4. The starting models

| model | how it works | parameters |
|---|---|---|
| **ConvNeXt U-Net** (1st place) | Turns each well into a 16-layer image (345 positions × 400 candidate depths). An image network outputs a probability for each candidate depth, and the answer is their weighted average. Like **stereo matching** in computer vision. | 53.6 M |
| **AnchorCNN** (2nd place) | A small network predicts each step's depth change, then a **shortest-path search** (dynamic programming) picks the best path. | 3.7 M |
| **Particle filter, HMM** (ours) | No learning: follow the drill step by step with simple physics and GR matching. | — |

---

## 5. What we tested

### 5.1 AnchorCNN (2nd-place model)

| idea | result | |
|---|---|---|
| Rebuild its training code | 6.49 ft; 3 bugs fixed on the way | — |
| Coarser depth grid, fewer step options ("C") | Half the compute (15.7 → 7.9 GFLOP); 6.02 / 6.57 / 6.20 over 3 seeds against 6.49 | ✅ cheaper, ⚪ accuracy change |
| TTA (8 shifted copies) | 0.26–0.56 ft better on every run; 8× the cost | ✅ |
| Even coarser grid (8 ft) | 1.46 ft worse; worse on 124 of 155 wells | ❌ |
| Cheaper first layer | Worse on 145 of 155 wells | ❌ |
| "Separable" design (rows and columns separately) | 11.4 ft | ❌ |
| 8-bit numbers (int8) | 10.8 ft after retraining, and slower on this CPU | ❌ |
| Tracker's guess as 2 extra inputs | 6.02 → 5.84 and 6.57 → 6.11 (2 seeds); each gain rests on 1–2 wells | ⚪ |
| Average of 3 seeds | 5.99 against 6.02 | ⚪ |
| Synthetic data | A bug delivered 32% synthetic wells instead of 77% | ⚠️ |
| Training on longer stretches | A bug made every training sample identical | ⚠️ |
| Distillation from an ensemble | The teacher (5.99) was no better than the student (5.94) | ⚠️ |

### 5.2 Trackers (no learning)

| idea | result | |
|---|---|---|
| Fix our particle filter step by step | 28.98 → 16.82 → 15.84 → 13.07 → 12.0 ft | ✅ |
| Better reference from "sibling" wells drilling the same rock | 12.7 → 12.0 ft, repeated twice | ✅ |
| 8× more particles | 0.02 ft change | ❌ |
| The 1st-place author's particle filter | 7.35 ft; 5.6 s per well | ✅ best tracker |
| HMM tracker | 23 ft in early tuning; never scored on the test wells | ⏳ |

### 5.3 ConvNeXt (1st-place model)

| idea | result | |
|---|---|---|
| Train 150 rounds instead of 60 | 5.16 → 4.98 | ✅ |
| TTA | 4.98 → 4.94 | ⚪ |
| More realistic synthetic wells | 0.15 ft worse, but better on most wells and still improving | ⚪ |
| Inputs from neighbouring wells | Behind at 60 rounds, but undertrained | ⚪ |
| **Axial attention** added | −0.024 ft; luck alone gives 0.030 ft | ⚪ |
| Two other redesigns | Slightly worse at 60 rounds, which is too short to judge | ⚪ |
| "Large-kernel" redesign | A bug kept its weights at exactly 0 | ⚠️ |
| Correcting depths with a hidden data constant | Every setting worse (5.00–20.05) | ❌ |

### 5.4 Swapping the backbone

| backbone | backbone parameters | error | network time | against ConvNeXt-tiny |
|---|---|---|---|---|
| ConvNeXt-small (original) | 49.5 M | 4.98 | 26.1 ms | ⚪ no difference |
| **ConvNeXt-tiny** | 27.8 M | **4.90** | 18.4 ms | reference |
| FastViT-SA12 | 10.4 M | 5.21 | 12.6 ms (merged) | ❌ |
| MobileOne-S1 | 3.6 M | 6.51 | 11.9 ms (merged) | ❌ |

- **Rule:** from 50 M down to 28 M parameters costs nothing, because the data runs out first.
  Below 10 M, accuracy drops, unless the model is taught (§5.6).
- **Timed but not trained:** ConvNeXt-nano (14.8 ms), InceptionNeXt-tiny (19.7 ms), FastViT-SA24
  (17.7 ms merged), FastViT-T12 (12.0 ms merged), ConvNeXt-base (36.6 ms). Runs are built ⏳.
- **Rejected without training:** DINOv2 (its coarse 25 × 29 grid loses the detail this task
  needs), and a dozen others that don't fit the U-Net without new code.

### 5.5 Combining models

| combination | error | |
|---|---|---|
| ConvNeXt + AnchorCNN, one fixed weight | 4.85 against 4.94 | ⚪ (5–6× slower) |
| A learned "gate" choosing between them | 5.18 | ❌ |
| Switch where the two models disagree | 5.09–5.12 | ❌ |
| Stacking all 17 prediction sets | 4.88–4.95 | ❌ |
| ConvNeXt-small + ConvNeXt-tiny, averaged | 4.78 against tiny's 4.90 | ✅ |
| ConvNeXt-tiny + FastViT | 4.85 | 🟡 |
| ConvNeXt-small + FastViT | 4.90 | ⚪ |
| **ConvNeXt-tiny + FastViT taught by small** | **4.69** | ✅ best overall |

**Lesson:** a partner helps if it makes *different* mistakes **and** is accurate enough. The simplest
method (one fixed average) beat every learned chooser.

### 5.6 Distillation — the main result

The student (FastViT-SA12) learns from the true depths **and** from the teacher's probability for
every one of the 400 candidate depths.

| student | teacher | error | at the end of training (unpicked) | against FastViT alone | against its teacher |
|---|---|---|---|---|---|
| FastViT alone | — | 5.21 | 5.30 | — | — |
| FastViT | ConvNeXt-tiny | 4.93 | 5.12 | ✅ | ⚪ |
| **FastViT** | **ConvNeXt-small** | **4.84** (4.83 merged) | **4.92** | ✅ | ⚪ |

- The taught FastViT **matches** a teacher with 4.4× its parameters, at half the network time
  (13.1 against 26.1 ms).
- "Beats its teacher" is **not** shown: it wins on 77 of the 155 wells, which is half.
- Why it works: the teacher's full probability map is a smoother target than the single true
  depth, which stops the student memorising the 618 training wells.

### 5.7 Speed

| trick | result | |
|---|---|---|
| 16-bit arithmetic (AMP) | 46.5 → 25 ms (ConvNeXt-tiny) | ✅ already on |
| **Merging FastViT's training branches** | 17.0 → 13.1 ms, same accuracy | ✅ automatic in every new run |
| Whole model in 16-bit (fp16) | 16% faster, same accuracy | ⏳ not in scoring yet |
| CUDA graphs (replay recorded GPU work) | 0.5–2 ms saved | ❌ not worth it |
| Pruning the decoder, TensorRT | at most 3–4 ms, or unmeasured | ❌ not done |
| Changing FastViT's attention | attention is only 1.6 ms of 19.6 | ❌ nothing to gain |
| **Fixing the CPU steps (22 September)** | whole pipeline **158 → 46 ms per well**, identical predictions | ✅ |

**The CPU fix, step by step** (per well):

| step | runs on | before | after | the fix |
|---|---|---|---|---|
| Neighbour-well map | CPU | 55 ms | 8 ms | cache the parsed CSV files; don't start 16 threads for tiny searches |
| Build the input image | CPU | 12.5 ms | 8.3 ms | write each layer once instead of copying three times |
| Network | GPU | 13.1 ms | 13.1 ms | — |
| Loop as scoring ran it | — | 104 ms | 38 ms | stop starting 4 worker processes (13–15 s on Windows) for 155 wells |

Before the fix, the GPU sat idle, waiting for the CPU.

### 5.8 Other teams' claims we checked

| claim | our result |
|---|---|
| The 773 typewells come from 54 master logs | ✅ 54 |
| A sibling reference helps a particle filter by 0.78 ft | ✅ 0.68 ft |
| Most error lies far from the known part of the well | ✅ 69–70% |
| The error is mostly one wrong straight line per well | ✅ on every model |

---

## 6. Scoreboard

| system | error (ft) | network time | number of networks |
|---|---|---|---|
| Assume "no change" | 9.8 | 0 | 0 |
| AnchorCNN | 6.02–6.57 | 141 ms (mostly the path search) | 1 |
| ConvNeXt-small (the 1st-place model) | 4.98 | 26.1 ms | 1 |
| ConvNeXt-tiny | 4.90 | 18.4 ms | 1 |
| **FastViT taught by ConvNeXt-small, merged** | **4.83** | **13.1 ms** | **1 — best single model** |
| **ConvNeXt-tiny + that student, averaged** | **4.69** | 31.5 ms | **2 — best overall** |

Errors are the 1st-place code's reported scores (§2 checkpoint trap). One training run each. The
whole pipeline, not just the network, takes 46 ms per well for the best single model.

## 7. Lessons

1. **Measure the noise first:** one comparison carries ±0.39 ft of luck, and most ideas moved the score less than that.
2. **Change one thing per run,** against a matching control.
3. **Check which wells a gain comes from:** a gain resting on 1–3 wells is luck.
4. **Check the training data before training:** two bugs silently broke whole experiments.
5. **Simple beats clever with 155 test wells:** one fixed average beat every learned chooser.
6. **A teacher must be measurably better than its student,** or distillation teaches nothing.
7. **Time every step before optimising one:** the "slow" step was starting worker processes, not computing.
8. **Measure, don't predict:** two of our confident predictions were wrong (report §9).

## 8. Waiting to run

1. **Two more seeds** of the best student and of FastViT alone: needed before the result is trusted.
2. **FastViT-SA24 taught by small:** twice the capacity at 17.7 ms.
3. **FastViT-T12 taught by small:** the same model without attention. Does attention matter?
4. **ConvNeXt-nano**, alone and taught by the small + tiny average.
5. **AnchorCNN with synthetic data,** done properly.
