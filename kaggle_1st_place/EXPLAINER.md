# Understanding the wellbore-geology model, from scratch

*A teaching walkthrough of the 1st-place solution and four proposed changes to it.*

**Who this is for.** You know what a neural network is, what training and a loss
function are, and roughly what a convolution does. You know nothing about oil
wells, and you have not met U-Nets, attention, or stereo matching. Every term
beyond the basics is defined the first time it appears.

**How to read it.** Parts 1–8 build up the existing solution one idea at a time.
Part 9 is the hinge: it reframes what the whole thing is, and that reframing is
what generates the four improvements in Parts 10–12. Read in order; each part
uses the one before it.

---

# Part 1 — The physical problem

## 1.1 What a horizontal well is

Drilling for oil used to mean drilling straight down. Modern wells go down and
then **turn sideways**, running horizontally for thousands of feet through one
particular layer of rock — the layer that holds the oil.

Staying inside that layer is the entire game. Drift up or down out of it and you
are drilling through worthless rock. Steering the drill bit to stay in the layer
while you drill is called **geosteering**.

## 1.2 The thing you don't know

Here is the part that surprises people. You always know exactly where the drill
bit is **in space**. Its position is surveyed continuously:

- `X`, `Y` — how far east and north
- `Z` — how deep below the surface
- `MD` — "measured depth", how far the bit has travelled *along* the hole

But knowing you are 10,000 feet below the surface tells you nothing useful,
because **the rock layers are not flat**. They tilt, roll, thin out, and jump
across faults. At 10,000 ft you might be in the top of the target layer, the
bottom of it, or the layer above.

So the quantity you actually need is: *how far down the rock column am I?* The
competition calls this **TVT**, and predicting it is the task. Think of `Z` as
"depth below the surface of the earth" and TVT as "depth below the top of the
rock sequence" — two completely different things, because the rock sequence
itself is at a different depth in different places.

## 1.3 What you get to work with

| Input | What it is |
|---|---|
| `GR` | A **gamma-ray** reading taken continuously along the hole. Different rocks emit different amounts of natural radiation, so `GR` is a fingerprint of the rock at the bit *right now*. |
| `X, Y, Z, MD` | The surveyed trajectory. Always known. |
| **Typewell** | A nearby *vertical* well that someone has already interpreted, giving the `GR` value at **every rock level**. This is your reference chart. |
| `TVT_input` | The correct answer for the first stretch of the well, then blank. |

The typewell is the key to everything. It is a lookup table: "at rock level
9,840 ft, the gamma-ray reads 74". Sampled about every half foot, spanning
roughly 1,000–2,000 feet of rock.

## 1.4 The exact task and how it's scored

Each well is a table of rows, one row every foot or so along the hole. The
`TVT_input` column is filled in for the first part and **blank for the rest**.
Predict the blanks.

Scoring is **pooled RMSE in feet**:

```
score = sqrt( mean over EVERY predicted row of EVERY well of (true - predicted)² )
```

Two details that matter enormously later:

1. **Squared error.** Being wrong by 20 ft is not twice as bad as 10 ft, it's
   *four times* as bad. Big mistakes dominate.
2. **Pooled, not averaged per well.** All rows from all wells go into one pot.
   A long well with big errors can outweigh dozens of short accurate ones.

There are 773 training wells with answers. (The shipped test set has only 3
wells and no answers, which is why all local work uses a held-out slice of the
773.)

---

# Part 2 — Why this is hard

## 2.1 How a human solves it

A geologist slides the well's `GR` curve up and down against the typewell chart,
looking for where the wiggles line up.

Suppose your last 200 ft of drilling read: *high, high, low, sharp spike, low,
medium*. You scan the typewell chart for that sequence. You find it at level
9,840. You conclude you are at 9,840.

This is **pattern matching between two signals**, and it's the whole job.

## 2.2 The catch: rock layers repeat

Sedimentary rock is laid down in cycles. The same *high, high, low, spike, low*
sequence appears at 9,840 — and again at 9,790 — and maybe again at 9,910.

So a stretch of log genuinely matches three places about equally well. The
correct answer is **ambiguous from local evidence alone**. It only becomes clear
from context: one of those three options is consistent with where you were
3,000 ft ago, and the others aren't.

Hold on to this. **Ambiguity is not noise in this problem — it is the structure
of the problem.** Two of the four improvements later exist purely because of it.

---

# Part 3 — The core trick: turn it into a picture

Ruby's key move is to stop treating this as curve-fitting and turn it into an
**image**, so that image-processing neural networks can be used.

## 3.1 Building the image

Make a 2-D grid:

- **Vertical axis = candidate level.** 400 rows covering ±100 ft around the last
  position you were sure about, so each row is 0.5 ft. Row *i* means the
  hypothesis *"suppose the well is at level i right now."*
- **Horizontal axis = distance along the well.** The rows of the well are
  grouped into chunks of 32 and averaged, giving 345 columns. Each column is
  about 32 ft of hole.

The picture is **345 columns × 400 rows**, and the question the network answers
is: *for each column, which row is the well actually at?*

Notice what this does. It converts "find where two wiggly lines match" into
"find a path across an image" — and finding structure in images is exactly what
convolutional networks are good at.

## 3.2 Why chunk into 32-ft columns?

Purely for size. The raw well is up to 11,024 rows. At 400 candidate levels
that's 4.4 million cells per well, times 16 channels. Chunking by 32 makes it
345 × 400 = 138,000 cells — 32× cheaper.

The cost is **blur**: averaging 32 ft into one number throws away detail
finer than 32 ft. Ruby partly compensates by adding extra channels that describe
what each chunk looked like inside (Section 4.2).

## 3.3 The 16 channels

A colour photo has 3 channels (red, green, blue) stacked at each pixel. This
image has 16, each carrying different information at every (column, level) cell.

**The matching evidence — the heart of the whole thing**

- `tw_gr` — what the typewell says the gamma-ray *should* be at this candidate
  level.
- `gr` — what the well *actually* measured in this column (the same value
  repeated down every row, since the measurement doesn't depend on which
  hypothesis you're testing).
- `gr_abs_diff` — the absolute difference between those two.

**That third channel is the match score.** Where it is small, the measurement
agrees with what that level should look like. The network is essentially
reading a heat map of "how well does each hypothesis fit". Everything else is
support.

**Recovering the detail that chunking destroyed**

- `gr_std` — how much the gamma-ray varied inside the chunk
- `gr_slope` — was it trending up or down
- `gr_first_last_delta` — net change across the chunk
- `gr_quadratic_a/b/c/rmse` — a curve fitted through the chunk, plus how well
  the curve fitted
- `gr_isnan_rate` — what fraction of readings were missing (real logs have gaps)

**Orientation and known information**

- `tw_tvt_rel` — the level axis itself, so the network knows which way is up
- `seen_tvt_rel` — the last level we were certain about
- `tw_seen_tvt_abs_diff` — distance from that certain point
- `tw_gr_is_nan` — where the typewell has no data

**Information from other wells**

- `geo_tvt_diff` — derived from nearby interpreted wells (Part 7.1)

---

# Part 4 — The network

## 4.1 What a U-Net is

A **U-Net** is the standard design for tasks where the output is the same shape
as the input — here, an image in, a per-cell score out.

It has two halves:

1. **The encoder (going down).** Repeatedly shrink the image while growing the
   number of channels. Early layers see fine detail in small neighbourhoods;
   later layers see coarse structure over large areas. In this model the along-
   the-well axis shrinks **345 → 173 → 87 → 44 → 22**, and the level axis
   shrinks **400 → 100 → 50 → 25 → 13**.
2. **The decoder (going back up).** Expand back to full size, and at each step
   glue on ("skip connection") the matching-size output from the encoder.

The skip connections are the point. The decoder gets both *what* is there (from
the deep, coarse layers) and *exactly where* it is (from the shallow, fine
layers). Without them you'd know a match exists but not precisely at which
level. Drawn out, the shrink-then-grow shape looks like a **U** — hence the
name.

## 4.2 Why a photo-trained encoder

The encoder here is **ConvNeXt-small**, a modern CNN, and it arrives
**pretrained on ImageNet** — it has already been trained to recognise millions
of ordinary photographs.

Using a photo network on gamma-ray data sounds absurd. It works because the
early layers of any image network learn generic things — edges, corners,
textures, stripes, gradients. Those are useful for *any* image, and a
match-score heat map has edges and stripes like anything else.

This is **transfer learning**: start from weights that already know something
general, rather than from random noise. With only 618 training wells it matters
a lot — there is nowhere near enough data to learn good edge detectors from
scratch.

("Small" refers to model size; ConvNeXt comes in tiny/small/base/large. This
one has about 50 million parameters, and the full model is 54.1M.)

---

# Part 5 — How the answer comes out

This section is the most important in the document. Three of the four proposed
improvements are aimed at what happens here.

## 5.1 From image to answer

The U-Net outputs a small feature vector for every (column, level) cell. A final
1×1 convolution — a learned weighted sum across those features — squashes each
cell to **one number**: a score for "the well is at this level in this column".

So we have a 345 × 400 grid of scores.

## 5.2 Softmax: scores into probabilities

Raw scores aren't probabilities. **Softmax** converts a row of numbers into
positive numbers that sum to 1, preserving order and exaggerating differences:

```
softmax(s)_i  =  exp(s_i) / Σ_j exp(s_j)
```

Softmax is applied **down each column** (across the 400 levels). Now each column
has a genuine probability distribution: *"60% chance we're at level 12, 30% at
level 40, 10% spread elsewhere."*

## 5.3 The answer is the mean of that distribution

The final prediction for a column is the **expectation** — each candidate level
weighted by its probability:

```
prediction = Σ_k  probability(level k) × depth(level k)
```

If the distribution is a single narrow hump at level 12, this gives ≈ level 12.
Good.

It's also **differentiable**, which matters: you cannot train through "just pick
the highest-scoring level" (`argmax`), because a tiny change in the scores
either doesn't change the pick at all or changes it in a jump — there is no
gradient to learn from. The weighted mean is smooth, so gradients flow.

## 5.4 The flaw hiding in that step

**Averaging is only safe when there is one hump.**

Remember Part 2.2: rock layers repeat, so the model often — correctly — believes
two things at once:

> "Either level 12 or level 40. I genuinely cannot tell which."

The average of those two humps is **level 26**: the empty valley between them,
a level the model assigns almost no probability to.

So the model confidently reports an answer it believes is *wrong*, and wrong by
14 ft. Not a small error — the signature of a catastrophic well. And because the
metric squares errors and pools everything (Part 1.4), catastrophic wells are
what the score is made of.

This is a known, named problem in another field, which we'll get to in Part 9.

---

# Part 6 — How it's trained

## 6.1 The three losses

The model is trained on three objectives added together.

**Regression loss (weight 0.017).** The predicted level should be close to the
truth. Uses **Huber loss**, which behaves like squared error for small mistakes
and like absolute error for big ones — so a single wild outlier can't dominate
the gradient.

**Alignment loss (weight 0.067 — the largest).** Not about the final number, but
about the *shape of the distribution*: it should put its probability mass on the
correct level. It's a **cross-entropy** against a target that is a smooth bump
centred on the truth rather than a single spike, so being one level off is
partly rewarded instead of punished as hard as being fifty off. The bump's width
is a fixed constant. *(Remember this — improvement 1 changes exactly it.)*

**GR-penalty loss (weight 0.017).** The predicted level should be consistent
with the gamma-ray actually observed.

Alongside: **AdamW** optimiser, learning rate 2e-4 on a **cosine schedule**
(smoothly decaying to near zero by the last epoch), and **EMA** — an
exponentially-weighted running average of the weights, which is usually a bit
better than the final weights because it smooths out the last few noisy steps.

## 6.2 Manufacturing extra training data

618 training wells is very few. So fake ones are generated, and the method is
cleverer than it first looks:

1. Take a real well.
2. Invent a new path through the rock by chopping up how real wells wandered up
   and down, then stitching the pieces back together in a new order.
3. **Hold `TVT + Z` fixed.** This is the important constraint. `TVT + Z` is the
   depth of the rock surface itself. Keeping it unchanged means *the geology
   stays where it is* and only the well's position within it changes. Without
   this you'd be generating wells drilled through impossible geology, and the
   model would learn from physically meaningless examples.
4. Regenerate the gamma-ray log: look up what the typewell says `GR` should be
   along the new path, then add back realistic well-specific deviation.

Plus ordinary augmentations: reversing the path, stretching along the hole,
cutting the tail short, hiding random channels, masking stretches of the
sequence, and jittering the typewell chart.

---

# Part 7 — The supporting machinery

## 7.1 Guessing from the neighbours

Rock layers are continuous across a field, so nearby interpreted wells tell you
roughly where yours should be. A local tilted plane is fitted through the 12
nearest wells, weighted by distance.

One nice touch: the search region is an **ellipse, not a circle** — stretched
1.3:1 along a 50° compass bearing. Geology has a grain direction, and a
neighbour along that grain is more informative than one across it.

This is built **fold-safe**: when predicting a held-out well, only training
wells contribute, so no answer leaks.

On its own this prior scores **11.57 ft**. That's the "geology alone, no
pattern matching" reference — the full model must beat it by a wide margin to be
earning its keep. It does: about 5 ft.

## 7.2 Smoothing the answer

Predictions come out slightly jittery column to column, but real geology is
smooth. So a **Savitzky–Golay filter** (fits a line through a sliding window and
takes the fitted value — smoothing without flattening genuine trends) is applied
and blended 73% with the raw prediction.

The subtlety: it smooths `TVT_pred + Z`, the **rock surface**, not `TVT`
itself. The well's trajectory has real sharp turns that should be preserved;
the geology underneath shouldn't. Worth about 0.04 ft.

## 7.3 Ensembling

The competition submission trained the whole thing 5 times on different data
splits, repeated 3 times = 15 models per recipe, across 6 recipe variants =
**90 models**, all averaged. Averaging many models cancels their independent
errors. It's the least clever and most reliable trick in machine learning.

---

# Part 8 — What the numbers look like

| Setup | Score (ft) |
|---|---|
| Neighbour-wells prior alone, no pattern matching | 11.57 |
| One model, 60 epochs, local 80/20 split | 5.16 |
| One model, 150 epochs, local 80/20 split | 4.98 |
| Competition version: 15 models × 300 epochs × full cross-validation | 4.80 |

The local single-model numbers are worse than the competition figure mostly
because of ensemble size and training length, not because anything is broken.
Their value is **comparison**: variants trained and scored identically can be
compared to each other even though the absolute level isn't competitive.

One statistic to carry into the rest of the document. On the 155 held-out wells:

| Worst N wells | Share of total squared error |
|---:|---:|
| 1 | 8% |
| 10 | 43% |
| 52 | 86% |

**Ten wells out of 155 carry nearly half the score.** Improving typical wells
barely moves the number. Fixing catastrophic ones moves it a lot. That is why
Part 5.4 matters so much.

---

# Part 9 — The reframing

Here is the observation that generates all four improvements.

## 9.1 Stereo matching

To judge depth with two eyes, your brain finds the same object in the left and
right image and measures how far apart it appears. Nearby objects shift a lot
between the two views, distant ones barely at all. That shift is called
**disparity**, and computing it for every pixel is **stereo matching** — a
problem computer vision has worked on for decades.

The standard method:

1. For each pixel, try every candidate shift.
2. Score how well the two images agree at that shift. Stack all those scores
   into a 3-D block: (row, column, candidate shift). This block is called a
   **cost volume**.
3. Run a neural network over the cost volume to clean it up, using context to
   resolve places where the raw matching is ambiguous. This is **cost
   aggregation**.
4. Convert scores to probabilities with softmax and take the weighted mean to
   get a smooth, sub-pixel answer. This is **soft-argmin**.

## 9.2 Ruby built a stereo matcher

Compare that list to Parts 3–5:

| Ruby's pipeline | Stereo matching |
|---|---|
| `gr_abs_diff` — match score at every (column, candidate level) | **cost volume** |
| U-Net over the stacked channels | **cost aggregation** |
| softmax down the level axis, take the mean | **soft-argmin** |
| a level for every column | **disparity map** |

It's the same architecture, arrived at independently. The "two views" being
matched are the well's own log and the typewell chart, and "disparity" is which
rock level you're at.

**Why this matters:** it means the useful literature isn't image classification
at all. It's fifteen years of people optimising exactly this shape of problem —
and they have already found and named the failure mode from Part 5.4.

## 9.3 The named failure

From that literature, on soft-argmin:

> It approximates the correct answer *when the distribution is unimodal and
> symmetric*. When that assumption is not met, it **blends the modes and may
> produce a solution far from all of them.**

"Unimodal" means one hump. That is Part 5.4 exactly, described by people who hit
it years earlier — and who developed fixes.

Independently, the 2nd-place competitor in this competition reported the same
thing: a model trained to predict a single path "collapses onto the average of
the hypotheses — a path that corresponds to none of them."

Two independent observations of the same failure in the same task. That is a
strong reason to attack it.

---

# Part 10 — Improvement 1: stop averaging things that shouldn't be averaged

**Variant name: `arch_v1_unimodal`**

## 10.1 The problem, restated

The model reports the mean of its distribution over levels. When that
distribution has two humps, the mean lands between them, on a level the model
thinks is wrong (Part 5.4).

## 10.2 The idea

Force the distribution to have **one hump** — and let the model control how wide
that hump is.

The existing alignment loss (Part 6.1) already trains the distribution toward a
smooth bump on the truth, but the bump's width is a **fixed constant** for every
column of every well. That's the weakness. A column where the rock signature is
unmistakable and a column where three levels match equally well are given the
same target shape, so the model is never asked to *say* which situation it's in.

## 10.3 How it works

Add a small **confidence head** — a few layers producing one number `c` between
0 and 1 per column, meaning "how sure am I here?"

Build the training target from that confidence:

```
width σ   = 1 bin + 8 bins × (1 − c)          (1 bin = 0.5 ft)
target(k) = softmax( −|level_k − truth| / σ )
```

Read that off:

- `c` near 1 → σ ≈ 1 bin → a narrow spike. "I'm certain."
- `c` near 0 → σ ≈ 9 bins → a wide but still **single** hump. "I'm unsure, and
  here is my honest uncertainty."

The point is the second case. The model may express doubt — but only as *one
wide hump*, never as two separate humps. It cannot quietly split, so the mean
can never land in an empty valley.

Two supporting pieces:

- **Focal weighting.** Bins near the peak are weighted more heavily, so the loss
  concentrates on getting the hump right rather than on the long flat tails.
- **A confidence penalty.** Without it there's a cheat: set `c = 0` everywhere,
  making every target maximally wide and easy to match — while saying nothing.
  A small pressure toward high confidence blocks it, so the model must *earn*
  the right to be uncertain.

## 10.4 Why it's the first thing to try

It's a **loss-function change**. No new layers of consequence, no extra memory,
no slowdown, no new dependencies. Days of work rather than weeks. And it targets
a mechanism that two independent sources identified in this exact task.

It's also a clean experiment: the new term *replaces* the old alignment term at
the same weight, in the same role. Any difference is attributable to the
formulation — fixed-width smoothing versus learned-width unimodal — and nothing
else.

---

# Part 11 — Improvement 2: seeing further along the well

Two variants attack this, in different ways: **`arch_v4_lkconv`** and
**`arch_v2_axial`**. They are **alternatives, not a pair.**

## 11.1 The problem: receptive field

A neuron's **receptive field** is how much of the input it can possibly be
influenced by. One 3×3 convolution sees a 3×3 patch. Stack a second and the
neuron sees 5×5; stack a third, 7×7. Receptive field grows with depth, slowly,
and information from far away arrives diluted through many layers.

Now look at the two axes of our image again — they mean completely different
things:

- **Level axis (400 rows).** Local. You care about a neighbourhood of candidate
  depths, and several may be plausible at once.
- **Along-the-well axis (345 columns ≈ 11,000 ft).** This is the *only* thing
  that can settle ambiguity. You break a tie between two candidate levels by
  noticing one is consistent with where you were 3,000 ft ago and the other
  isn't.

A standard CNN uses square filters and treats both axes identically. So the
network is structurally weakest at exactly the reasoning that would rescue the
ambiguous wells.

## 11.2 Fix A — long, lopsided filters (`arch_v4_lkconv`)

Some recent CNNs use enormous filters — 31×31, even 51×51 — so a single layer
takes in a huge area directly instead of relaying it through depth. On dense
prediction tasks this measurably helps.

**The insight for this problem: a square 51×51 filter is mostly wasted here.**
You don't need 51 rows of level context; five would do. What you need is
*length along the hole*. So the right filter is a long thin rectangle: about 31
tall (along the well) by 3 wide (across levels).

Two practical details that make it safe:

- **Depthwise convolution.** Instead of mixing all channels, each channel gets
  its own filter. Cost drops by a factor of the channel count, which is what
  makes a 31-long filter affordable at all.
- **Tapering.** Later layers have fewer columns left (173 → 87 → 44 → 22), so
  the filter lengths taper: 31, 31, 15, 7. A 31-long filter on a 22-column map
  is just an expensive global average.

**The safety property.** The new filter is added *in parallel* to the existing
pretrained one, through a multiplier that starts at **exactly zero**:

```
output = pretrained_filter(x)  +  gate × new_long_filter(x)      # gate starts at 0
```

At the first training step this computes exactly what the pretrained network
computed. Training starts from the baseline and the long filter can only be
learned as an addition to it. (Verified in this implementation: 36 filters
wrapped, output bit-identical to the original at initialisation.)

Without that gate you'd be scrambling pretrained ImageNet features with random
noise on step one, and a poor result would tell you nothing about the idea.

## 11.3 Fix B — attention along the well only (`arch_v2_axial`)

**Attention** lets every position look directly at every other position and
decide what to pay attention to — no relaying through depth, no dilution. It is
the ideal tool for "check whether this is consistent with 3,000 ft ago".

The problem is cost. Attention compares every position to every other, so cost
grows with the **square** of the number of positions. At 345 × 400 = 138,000
positions, the comparison table alone would exceed a terabyte. Completely
impossible in 6 GB.

**The fix: do the two directions separately.** Instead of comparing every cell
to every other cell:

- attend along the level axis only (400 positions), and
- attend along the well axis only (345 positions),

as two operations. Cost goes from *(345×400)²* to *345×400×(345+400)* —
roughly 185 times cheaper, and megabytes instead of terabytes.

This is **axial attention**, and here it is more than a cost trick: it matches
the structure of the problem. The two axes are different, so treat them
differently. In this implementation attention is applied along the well axis at
the three coarsest stages, where the well axis is 87, 44 and 22 long — each
short enough that attention spans the *entire hole* at a few megabytes.

Two details:

- **Position information.** Attention has no built-in sense of order — shuffle
  the inputs and you get the same answer. Usually you add a learned position
  table, but that fixes the input length. Instead a small 3×1 convolution is
  applied first: it can only see neighbours, which injects a sense of "nearby",
  and it works at any length.
- **Same zero-start safety.** The block's contribution is scaled by a parameter
  starting at zero, so it begins as the identity function.

  *(A real bug caught here: the positional convolution sits outside that gate,
  so zeroing the gate alone wasn't enough — the block still perturbed the
  features at step one. The convolution had to be zero-initialised too. Worth
  mentioning because it's the kind of thing that silently ruins an experiment:
  the run would have "failed", and the conclusion would have been wrong.)*

## 11.4 Choosing between them

They fix the same weakness. Run **one**, measure, then consider the other. If
you run both at once and the score improves, you have learned nothing about
which one earned it.

Large filters are the cheaper first swing. Axial attention is the more
principled one — it can genuinely connect two distant points, while a 31-long
filter only reaches 31 columns per layer.

---

# Part 12 — Improvement 3: guess, then correct, repeatedly

**Variant name: `arch_v3_raft`**

## 12.1 The problem

The current model gets **one shot**. Build the match-score image, process it
once, softmax, average, done. Everything rides on a single pass — including the
averaging step from Part 5.4.

## 12.2 The idea

Don't try to be right immediately. Make a rough guess, then improve it in small
steps — the way you'd solve a jigsaw puzzle by placing pieces approximately and
adjusting, rather than computing every final position up front.

This is how modern stereo matching works (the RAFT family), and it's now the
dominant approach in that field.

## 12.3 How it works

Start from the current model's one-shot answer. Then repeat 8 times:

1. **Look around where you currently think you are.** Read the match scores at
   your current level and at a set of offsets around it — nearby, medium, far
   (offsets of ±1, ±4, ±16 levels). This is a small local window, not the whole
   400-row column.
2. **Predict a correction.** A small recurrent network — a **GRU**, which
   carries a memory as it scans along the well and learns what to keep and what
   to forget — runs along the columns and outputs "move down 2 ft here, up 1 ft
   there".
3. **Apply it** and go round again.

The GRU's memory is what keeps corrections *coherent*: the adjustment at one
column is informed by adjustments made at its neighbours, so the whole path
moves sensibly rather than each column wobbling on its own.

## 12.4 Why this suits the problem, specifically

**It commits.** Each round nudges toward one interpretation rather than
averaging two. Where the one-shot mean is stuck in the valley between two humps,
refinement can walk into one of them. It attacks Part 5.4 structurally, not just
through a loss.

**It escapes the fixed window.** The ±100 ft window is a hard commitment: if the
truth lies outside, the current model can *never* reach it, at any confidence.
Iterative correction lets the estimate walk to wherever the evidence leads.

**The number of rounds is a dial you set after training.** Want more accuracy
and have time? Run 16 rounds instead of 8 — no retraining. Few architecture
changes give you a free accuracy/time trade at test time.

## 12.5 Costs

This is a genuine rebuild — weeks, not days. Training an iterative model is
fussier to stabilise, and the new parts can't inherit pretrained weights (though
the feature extractor still can). It adds 0.55M parameters to 54.1M, so it's
cheap in size; the cost is engineering and risk, not compute.

**Safety property again:** the correction head is initialised to output exactly
zero, so at the first step all 8 rounds reproduce the one-shot answer precisely.
The refiner can only earn departures from the baseline. (Verified: iterate 0
matches the one-shot expectation to 2.4 × 10⁻⁷.)

---

# Part 13 — How you'd know if any of this worked

The hardest lesson from this project isn't architectural.

Remember Part 8: ten wells carry nearly half the score. That means the metric on
155 wells is effectively decided by **about ten coin flips**. Measured
consequences:

- The uncertainty on any comparison between two models is roughly **±0.4 ft**.
- Which wells blow up **changes from run to run**. In one comparison the
  baseline failed badly on a well the challenger nailed, while the challenger
  destroyed a well the baseline handled easily.
- A single model's own score wanders by **±0.16 ft** between neighbouring
  training epochs, purely from training noise.

So an improvement worth 0.1 ft **cannot be detected** by one run of each model.
It's inside the noise.

This is why a specific test is used, borrowed from the 2nd-place solution:
compute how much each well contributed to the difference between two models,
then delete the biggest contributors one at a time and re-score. If the
improvement vanishes after removing three wells, it was luck. If it survives
removing fifty, it's real.

**The general lesson, and it applies far beyond this project: before chasing a
0.1-unit improvement, find out whether your evaluation can measure 0.1 units at
all.** Often it can't, and the fix is more evaluation data or more random seeds
— not a better model.

---

# Glossary

| Term | Meaning |
|---|---|
| **Attention** | An operation letting every position look directly at every other and weight what's relevant. Cost grows with the square of position count. |
| **Axial attention** | Attention applied along one axis at a time instead of over the whole grid. Far cheaper, and a good fit when the axes mean different things. |
| **Cost volume** | A stack of match scores for every position × every candidate answer. Here: every column × every candidate level. |
| **Cross-entropy** | A loss measuring how far a predicted probability distribution is from a target one. |
| **Depthwise convolution** | A convolution where each channel gets its own filter instead of mixing channels. Much cheaper; makes very large filters affordable. |
| **Disparity** | In stereo vision, how far an object shifts between two views. Here, the analogue is which rock level you're at. |
| **EMA** | Exponential moving average of weights during training; usually slightly better than the final weights. |
| **Expectation / soft-argmin** | The probability-weighted mean of a distribution. Differentiable, unlike picking the maximum — but wrong when the distribution has two humps. |
| **Geosteering** | Steering a drill bit to stay inside a target rock layer. |
| **GR (gamma ray)** | Natural radiation measured along the hole; a fingerprint of the rock. |
| **GRU** | A small recurrent network that scans a sequence carrying a memory, learning what to keep and forget. |
| **Huber loss** | Squared error for small mistakes, absolute error for large ones. Limits outlier influence. |
| **MD** | Measured depth: distance travelled along the hole. |
| **Multimodal / bimodal** | A distribution with more than one hump — more than one plausible answer. |
| **Pooled RMSE** | Root-mean-square error over all rows of all wells combined, not averaged per well. |
| **Receptive field** | How much of the input a given neuron can be influenced by. |
| **Softmax** | Turns arbitrary scores into probabilities summing to 1. |
| **Stereo matching** | Computing depth by finding the same content in two views. The framework this whole task turns out to fit. |
| **Transfer learning** | Starting from weights trained on another task instead of from random. |
| **TVT** | The prediction target: position within the rock column, as opposed to depth below the surface. |
| **Typewell** | A reference vertical well giving GR at every rock level — the chart you match against. |
| **Unimodal** | Having exactly one hump. The condition under which taking the mean is a sound answer. |
| **U-Net** | Encoder-decoder network with skip connections, for when output shape matches input shape. |
