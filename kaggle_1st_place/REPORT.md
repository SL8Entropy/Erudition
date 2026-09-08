# Porting the 2nd-place ideas into the 1st-place solution

**A report on what Ruby's winning pipeline does, the four variants we built from
Bilzard's write-up, and which one actually won.**

Everything below was measured locally on one machine (RTX 3050 6 GB) using an
80/20 holdout carved out of the 773 labelled training wells. No number here is a
competition leaderboard score.

---

## 1. The problem, in plain terms

A **horizontal well** is drilled sideways through layers of rock, and the crew
wants to stay inside one particular layer — the productive one. This is called
*geosteering*.

Here is the difficulty. You always know where the drill bit is **in space**: its
easting, northing and depth (`X`, `Y`, `Z`) are surveyed continuously. What you
do *not* know is where the bit is **inside the rock layers**. Layers tilt, roll
and jump across faults, so being at a depth of 10,000 ft tells you nothing about
whether you are in the top of the target layer, the bottom of it, or the layer
above. That position-within-the-rock is the quantity to predict, and the
competition calls it **TVT** (true vertical thickness — read it as "how far down
the rock column am I").

What you get to help you:

| Input | What it is |
|---|---|
| `GR` | A gamma-ray reading taken continuously along the well. Different rock layers emit different amounts of natural radiation, so `GR` is a fingerprint of the rock right at the bit. |
| `X, Y, Z, MD` | The surveyed trajectory. `MD` is measured depth — distance travelled along the well. |
| **Typewell** | A nearby *vertical* reference well, already interpreted, giving the `GR` value at every rock level. This is the fingerprint chart you match against. |
| `TVT_input` | The answer for the *first* part of the well, then blank. |

The rows to predict are exactly the rows where `TVT_input` is blank — a
contiguous stretch at the end of each well. Scoring is **pooled RMSE in feet**:
one root-mean-square over every predicted row of every well, pooled together
rather than averaged per well. That detail matters enormously later.

### How a geologist actually solves it

They slide the well's `GR` curve up and down against the typewell's `GR` chart
and look for where the wiggles line up. If the well's log shows "high, high, low,
spike, low" and the typewell shows that same sequence at level 9,840 ft, you are
probably at 9,840 ft.

The catch — and it is the whole difficulty of the task — is that **rock layers
repeat**. The same wiggle pattern appears at several different levels, so a
stretch of log often matches three or four places equally well. The correct
answer is genuinely ambiguous from local evidence, and only becomes clear from
context further along the well.

---

## 2. Ruby's solution (1st place)

Ruby's core insight is to stop treating this as a curve-fitting problem and
**turn it into an image problem**, then hand the image to an image neural
network.

### 2.1 Building the picture

For each well she builds a 2-D grid:

- **Vertical axis — candidate level.** 400 rows spanning ±100 ft around the last
  known position, so each row is 0.5 ft. Row *i* means "suppose the well is
  sitting at level *i* right now".
- **Horizontal axis — distance along the well.** The well's rows are chunked into
  groups of 32, giving 345 columns. Each column is roughly 32 ft of hole.

So the picture is 345 columns × 400 rows, and the question the network answers is
"for each column, which row is the well actually at?"

### 2.2 The 16 channels

The picture has 16 stacked layers of information, like the red/green/blue
channels of a photo. Grouped by what they do:

**The matching evidence (the heart of it)**
- `tw_gr` — the typewell's GR at each candidate level.
- `gr` — the well's own measured GR for that column, copied down every row.
- `gr_abs_diff` — the absolute difference between those two. **This is the match
  score**: wherever it is small, the well's reading agrees with what that level
  should look like. The network is essentially reading a heat map of "how well
  does each hypothesis fit".

**What the GR looks like inside each column** (because averaging 32 ft into one
number throws away detail, these summarise what was lost)
- `gr_std`, `gr_slope`, `gr_first_last_delta` — spread, trend, and net change.
- `gr_quadratic_a/b/c/rmse` — a curve fitted through the column, plus how well it
  fitted.
- `gr_isnan_rate` — what fraction of readings are missing (real logs have gaps).

**Where we are and what we know**
- `tw_tvt_rel` — the level axis itself, so the network knows up from down.
- `seen_tvt_rel` — the last position we were sure about.
- `tw_seen_tvt_abs_diff` — distance from that known point.
- `tw_gr_is_nan` — where the typewell has no data.

**Information from the neighbours**
- `geo_tvt_diff` — derived from other wells nearby (see §2.5).

### 2.3 The network and how it answers

A **2-D U-Net** with a **ConvNeXt-small encoder pretrained on ImageNet**. Using a
photo-trained backbone sounds odd for well logs, but the job is pattern-matching
in a 2-D image, and pretrained edge and texture detectors transfer well.

The output is one score per (column, level). A **softmax down the level axis**
turns each column's scores into a probability distribution — "I'm 60% sure we're
at level 12, 30% at level 40, 10% elsewhere". The final prediction for that column
is the **mean of that distribution**.

This is worth pausing on, because it is the same idea Bilzard reached
independently. Both solutions refuse to commit to a single answer where the
evidence is ambiguous; they keep a probability spread over all the plausible
levels and report its average. Ruby computes one distribution per column
directly; Bilzard propagates a distribution of *moves* along the well with
dynamic programming. Different machinery, same philosophy.

### 2.4 Training signal

Three losses, added together:

| Loss | Weight | What it does |
|---|---:|---|
| Regression (Huber) | 0.0167 | Predicted level should be close to the truth, with reduced sensitivity to outliers. |
| Alignment (cross-entropy) | 0.0667 | The *distribution* should put its mass on the true level. Target is a smoothed bump (σ = 1.25) rather than a spike, so near-misses are partially rewarded. |
| GR penalty | 0.0167 | The predicted level should be consistent with the observed GR. |

Optimiser AdamW, learning rate 2e-4 on a cosine decay, weight averaging (EMA
0.99), bfloat16 mixed precision.

### 2.5 The geological prior from neighbouring wells

Rock layers are continuous across a field, so if three interpreted wells sit
nearby, their surfaces tell you roughly where yours should be. Ruby fits a local
tilted plane through the 12 nearest wells using inverse-distance weighting, with
**anisotropy** — the search ellipse is stretched 1.3:1 along a 50° bearing,
because geology has a grain direction and neighbours along strike are more
informative than neighbours across it.

Crucially it is built **fold-safe**: when predicting a held-out well, only
training wells contribute. On its own this prior scores **11.57 ft** — that is
the "geology alone, no log matching" reference point, and the model has to beat
it by a wide margin to be earning its keep.

### 2.6 Synthetic training data

773 wells is not many. Ruby manufactures more:

1. Take a real well.
2. Invent a new stratigraphic path by **bootstrapping chunks of real level-change
   sequences** — chop up how real wells wandered up and down, stitch the pieces
   back together in a new order.
3. **Hold `TVT + Z` fixed.** This is the clever constraint. `TVT + Z` is the
   depth of the rock surface itself; keeping it unchanged means the *rock stays
   where it is* and only the well's position within it changes. Physically
   coherent by construction.
4. Regenerate the GR log: read what the typewell says the GR should be at the new
   path, then add back a modelled residual (the well-specific deviation).

Plus conventional augmentations: reversing the path, stretching along MD, cutting
the tail short, masking channels, masking stretches of sequence, jittering and
drifting the typewell GR.

### 2.7 Finishing touches

- **Savitzky-Golay smoothing** of the predicted *surface* (`TVT_pred + Z`, not
  `TVT` itself — smoothing the geology, not the trajectory), window 577, blended
  72.6% with the raw prediction. Worth about 0.03–0.04 ft.
- **Ensembling**: geographic 5-fold × 3 repeats = 15 models per recipe, and six
  recipes = **90 models** in the final submission.

### 2.8 What we run locally, and why our numbers look worse

We train **one** model on an 80/20 split, not 90 models on full cross-validation.
The archived `0801_V2` figure of **4.8045 ft** comes from 15 models × 300 epochs
× geographic CV over all 773 wells. Our baseline is **5.1618 ft** at 60 epochs.
Those are not comparable, and the gap is mostly ensemble size and training
length. What our setup *is* good for is **A/B comparison** — every variant is
trained and scored identically, so differences between them are meaningful even
though the absolute level is not competitive.

---

## 3. What Bilzard (2nd place) did differently

Three ideas from his write-up were worth borrowing.

### Idea 1 — A test for whether an improvement is real

Ruby found that adding neighbouring-well information improved her validation
score but hurt her competition score, could not explain it, and put it down to
messy labels.

Bilzard built a test for exactly this. Pooled RMSE is a **sum** of squared
errors, so a handful of disastrous wells dominate it. If a change happens to
rescue three bad wells by luck, the headline score improves even though the
change is worthless.

His answer — **leave-largest-contribution-out**:

1. For each well compute `g_w = SSE_new(w) − SSE_old(w)`, the change in that
   well's total squared error.
2. Remove wells one at a time, largest `|g_w|` first — i.e. delete the wells most
   responsible for the difference.
3. Rescore both models after each removal.
4. **`k*`** is the number of removals at which the improvement disappears.

Small `k*` means the gain rested on a few lucky wells. Large `k*` means it is
systematic. Bilzard rejected a candidate whose apparent −0.28 gain evaporated
after removing 8 of 773 wells, and accepted one that survived 52.

### Idea 2 — More physically honest synthetic data

Bilzard noticed the 773 typewells are **not independent**: they are windows cut
from about **54 master rock sequences**. Knowing that, you can crop *any* level
window from a master rather than being limited to windows that happen to exist.

His synthesis then: blend two real trajectories (`TVT_mix = λ·TVT₁ + (1−λ)·TVT₂`);
place the result into another rock system at a **realistic depth** — specifically
at the same relative position within that system's *drilling band* (the range of
levels wells in that system are actually steered through); regenerate the GR from
that system's master; and add residuals **pasted from real wells** rather than
sampled from a fitted model.

### Idea 3 — Fixing the blurry-measurement problem

Both solutions average ~32 ft of hole into one column. Where exactly you place
the cut lines changes the averages — like how moving a photo's crop lines changes
what is in frame. With a GR correlation length of about 18 ft, a 32 ft box
average is coarse enough that this **aliasing** measurably shifts the answer.

Bilzard's fix costs nothing: run inference 8 times with the column grid shifted
by a fraction of a column each time, and average the results. **No retraining.**

---

## 4. What we built

One edited copy of Ruby's `0801_V2` snapshot lives in `solution/experiments/bilzard/`.
At its original settings it is **bit-identical** to the archived pipeline —
verified by comparing built input tensors, maximum difference 0.0 — so every
variant differs from the baseline only by its own switches.

### Variant 1 — `rb_v1_neighbor` (tests Idea 1)

Idea 1 is a *method*, not a model, so it needed something to judge. We turned on
**all nine** neighbouring-well channels (16 → 24 input channels) — precisely the
change Ruby could not evaluate. This is not a proposed improvement; it is the
defendant.

The test itself is `solution/seq_NN_robust_compare.py`. It reads two runs'
predictions, produces the removal curve, reports `k*`, the per-well win rate, and
a well-level bootstrap. Sanity-checked against a null case (a model versus a
rerun of its own weights): delta +0.0002 ft, `k*` = 0, correctly rejected.

### Variant 2 — `rb_v2_synth` (Idea 2)

Three changes inside the simulator:

1. **Trajectory mixup** — blend two real paths, on top of the existing bootstrap.
2. **Master-series re-skinning** — new module `seq_NN_master_typewell.py`. It
   rebuilds Bilzard's consolidation with one dense correlation pass: put every
   typewell on a shared level grid, compute all pairwise correlations as a single
   matrix multiply, and union those overlapping ≥100 ft with correlation ≥0.9.
   Result: **58 systems** from the 618 training wells (60 using all 773), against
   Bilzard's 54 — and members of a group agree to **0.000 GR** on their overlap,
   confirming outright that they are crops of one series. Takes ~1.3 seconds.
   A simulated well is then re-placed into another system at the same
   drilling-band quantile it held in its own, with GR regenerated from a crop of
   the target master.
3. **Real residual pasting** — the GR noise becomes a 70/30 mixture of the fitted
   model and real pasted residuals.

One subtlety worth recording: Ruby's simulator already held `TVT + Z` fixed, so
she was *already* obeying the physical constraint Bilzard emphasises.
Re-skinning is the one operation that deliberately moves it — so every absolute
surface quantity is shifted with it to keep the geo prior aligned.

### Variant 3 — `rb_v3_tta` (Idea 3)

Added `md_phase_shift` (move the column cut lines by N rows) and `md_phase_tta`
(average inference over several such grids). Predictions from different phases
live on different grids, so they are mapped back onto original rows *before*
averaging. Training and checkpoint selection stay single-phase, as Bilzard did.

Because it changes nothing about training, `solution/seq_NN_rescore.py` applies
it to weights that already exist — **5 minutes instead of 5 hours**.

### Variant 4 — `rb_v4_all`

Variants 2 and 3 together.

---

## 5. Results

### Round 1 — 60 epochs

| Run | Pooled RMSE | vs baseline | `k*` | Bootstrap improving | Verdict |
|---|---:|---:|---:|---:|---|
| `0801_V2` (baseline) | **5.1618** | — | — | — | — |
| `rb_v3_tta` | **5.1119** | −0.050 | 1 | 66% | REJECT |
| `rb_v4_all` | 5.3711 | +0.209 | 0 | 20% | REJECT |
| `rb_v2_synth` | 5.5241 | +0.362 | 0 | 9% | REJECT |
| `rb_v1_neighbor` | 6.2556 | +1.094 | 0 | 0% | REJECT |

Nothing beat the baseline. But every validation curve was still falling when the
schedule ran out, and the two changes that lost worst — heavier augmentation and
a wider input — are exactly the two that pay off *late*. Placing each run on the
baseline's own trajectory showed how starved they were: `rb_v2_synth` at epoch 60
sat where the baseline sat at epoch 35; `rb_v1_neighbor` sat where the baseline
sat at **epoch 19**. So the 60-epoch verdict was not trustworthy.

### Round 2 — 150 epochs

| Run | Pooled RMSE | vs baseline | best epoch | `k*` | Bootstrap improving | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `0801_V2_ep150` (baseline) | 4.9785 | — | 110/150 | — | — | — |
| **`rb_v3_tta_ep150`** | **4.9374** | **−0.041** | — | 1 | 65% | REJECT |
| `rb_v4_all_ep150` | 5.1092 | +0.131 | — | 0 | 26% | REJECT |
| `rb_v2_synth_ep150` | 5.1250 | +0.147 | 115/150 | 0 | 19% | REJECT |

More epochs helped both, and helped the synthesis model far more:

- Baseline 5.1618 → 4.9785 (−0.183)
- `rb_v2_synth` 5.5241 → 5.1250 (−0.399)
- **The gap closed from +0.362 to +0.147 — 60% of it was undertraining.**

---

## 6. Which one did the best

**`rb_v3_tta_ep150` — 4.9374 ft.** The 150-epoch baseline with 8-phase MD
test-time averaging. Down from 5.1618 at the start of this work, a **−0.224 ft**
improvement overall.

It is also the cheapest thing here: it required no training of its own, just a
5-minute rescore of weights that already existed.

**The honest caveat:** it fails the strict acceptance test, every time. But the
pattern across four independent measurements is consistent:

| Model | Budget | TTA effect | `k*` | Bootstrap |
|---|---|---:|---:|---:|
| baseline | 60 | −0.050 | 1 | 66% |
| baseline | 150 | −0.041 | 1 | 65% |
| `rb_v2_synth` | 60 | −0.153 | 2 | 86% |
| `rb_v2_synth` | 150 | −0.016 | 4 | 57% |

**Four out of four helped.** The effect shrinks as a model converges, which makes
physical sense — a better-trained model is already less sensitive to the aliasing
TTA averages away. Four consistent signs, plus Bilzard's independent −0.21 on 773
wells, plus zero cost, is the whole case for keeping it. Our holdout simply
cannot certify an effect this small.

### Why the synthesis variant is not a failure either

`rb_v2_synth` still loses on the headline (+0.147), but **how** it loses changed
completely between rounds. At 60 epochs it was worse everywhere — worse even
after deleting the 8 most catastrophic wells. At 150 epochs its removal curve
goes **negative from k=10 all the way to k≈105**: on the bulk of the holdout it
is *better* than the baseline, and it loses only on a handful of disasters. Its
median well is better too (2.796 vs 2.929 ft).

It was also **still improving at epoch 150** (−0.684 over epochs 50→150) while
the baseline had largely flattened (−0.201, oscillating in a 5.02–5.76 band).

---

## 7. The thing that limits every conclusion here

The holdout is too small and too tail-dominated to resolve the effects we are
chasing.

**Error concentration on our 155 holdout wells:**

| Worst N wells | Share of total squared error |
|---:|---:|
| 1 | 8% |
| 3 | 20% |
| 10 | 43% |
| 52 | 86% |

Ten wells carry nearly half the metric. The effective sample size is closer to
10–20 than to 155.

Three symptoms of that:

1. **The remaining +0.147 gap is smaller than the baseline's own checkpoint
   noise** (σ = 0.161 across epochs 50–150). And the baseline's `best_epoch=110`
   is the luckiest point on a noisy plateau, while `rb_v2_synth`'s 115 sits on a
   monotone descent — the comparison is biased in the baseline's favour by
   roughly the size of the gap itself.
2. **The deciding wells change identity every run.** At 150 epochs the baseline
   blows up on `d1457cc5` (11.66 ft) which `rb_v2_synth` fixes (4.76 ft), while
   `rb_v2_synth` destroys `f0188a48` (0.83 → 8.67 ft), a well the baseline nearly
   nails. These are coin flips on individual hard wells.
3. **The bootstrap confidence interval on any delta is about ±0.4 ft**, and we
   are trying to measure 0.1 ft.

Ruby's original confusion — validation improving while the real score got worse —
now looks less like bad labels and more like this: **the metric is decided by
about ten coin flips**, and a single run cannot tell a real 0.1 ft gain from
luck. That is precisely the problem Bilzard's test was built to expose, and
building it was worth doing even though it rejected everything we fed it.

---

## 8. What to do next

1. **Add seeds, not epochs.** `--repeats 3` on the baseline and `rb_v2_synth` at
   150 epochs, or 5-fold CV over all 773 wells. Nothing smaller can separate
   these candidates. This is the single highest-value next run.
2. **Fix a calibration mismatch in re-skinning.** Real samples get a typewell
   blended toward the well's own visible log; a re-skinned sample gets the raw
   master crop. Both halves of a re-skinned sample are self-consistent, but its
   *statistics* differ from a real sample's — a plausible contributor to the
   remaining gap.
3. **Ablate the three synthesis sub-changes separately.** Mixup, re-skinning and
   residual pasting were bundled; the loss is not attributed to any of them. Also
   try a gentler dose (`reskin.apply_prob` 0.15–0.25 rather than 0.5).
4. **Investigate the tail directly.** A few wells decide everything. Find what
   they share — rock system, GR gaps, suffix length, distance to the nearest
   interpreted well.
5. **`rb_v1_neighbor` remains unproven, not disproven.** At epoch 60 it sat where
   the baseline sat at epoch 19. Give it 300 epochs against the baseline's 150;
   losing with a 2× compute advantage would be decisive, and costs half what a
   symmetric comparison would.

---

## 9. Reproducing this

Everything runs from `solution/`. Full commands are in `CLAUDE.md` under
"Running the four recipes". In short:

```bash
python seq_NN_holdout_eval.py --id 0801_V2 --source-dir experiments/bilzard --cfg-name rb_v2_synth --epochs 150 --output-dir results/rb_v2_synth_ep150 --device cuda --offline-timm --batch-size 4 --val-batch-size 4 --grad-accum-steps 4
```

```bash
python seq_NN_rescore.py --id 0801_V2 --models-dir results/0801_V2_ep150 --source-dir experiments/bilzard --tta 8 --output-dir results/rb_v3_tta_ep150 --device cuda --offline-timm --val-batch-size 4 --num-workers 4
```

```bash
python seq_NN_robust_compare.py --base results/0801_V2_ep150 --treat results/rb_v2_synth_ep150 --output-dir results/rb_v2_synth_ep150
```

**Files added by this work:**

| File | Purpose |
|---|---|
| `solution/experiments/bilzard/` | The edited fork; four recipes in its config registry |
| `solution/experiments/bilzard/seq_NN_master_typewell.py` | Master rock-sequence consolidation |
| `solution/seq_NN_robust_compare.py` | The leave-largest-contribution-out acceptance test |
| `solution/seq_NN_rescore.py` | Re-score existing weights, optionally with phase TTA |

Timing on the RTX 3050: ~110 s/epoch, so 150 epochs is about 5 hours. Training is
GPU-bound — the data pipeline sits idle ~92% of the time, and the usual speed
knobs (bfloat16, channels-last, TF32, cuDNN autotuning) are already enabled.
