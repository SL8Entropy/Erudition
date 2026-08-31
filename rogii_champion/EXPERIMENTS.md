# The five iterations

The baseline is Ruby's 1st place solution: a 2D alignment canvas, a ConvNeXt-Small
U-Net, cross-entropy down each column, expectation decoding. Private LB 5.639,
CV 4.627 over 3 seeds × 5 folds.

Each iteration changes one thing, at a different stage, taken from a different
team. That is deliberate: five changes at the same stage would compete for the
same headroom, and five changes at once would be unattributable.

| | change | source | stage | retrain | inference cost |
|---|---|---|---|---|---|
| v1_tta | decode-time augmentation | 2nd, 4th, 5th | decode | none | ×16 |
| v2_slope | predict `dz_layer`, integrate | 2nd | target | 1× | ×1 |
| v3_moves | move-distribution head + DP | 2nd | head/decoder | 1× | ×1.1 |
| v4_refgr | sibling + self reference GR | 3rd, 5th | features | 1× | ×1 |
| v5_gate | sigma head + per-row gate | 3rd | ensembling | 3× | ×1 |

---

## v1_tta — spend compute at decode time

**Change.** Four inference-side additions, no retraining:

1. **MD-phase TTA, 8 views.** Columns are 32 ft wide but the GR correlation
   length is ~18 ft, so the per-column average is coarse and *where* the column
   boundaries fall changes the input. 2nd place shifts the column-grid origin
   through 8 sub-column offsets and averages. They measured OOF 5.624 → 5.417.
2. **Quarter-point re-anchoring.** Predict the whole well; promote the
   prediction 25% into the target region to "known"; re-render and predict the
   remaining 75%; average. 4th place: 0.18 CV. They tested a second re-anchor at
   50% — it helped CV and hurt private — and stopped. One point only.
3. **Adaptive canvas.** If the predicted path exceeds 55% of the ±100 ft window,
   re-run that well on a 1.6× taller canvas. 5th place's trick: cheap where it
   can be, expensive only where it must be.
4. **Sibling-corrected typewell** at inference (see v4).

**Why first.** It costs nothing to train and is measurable against the base
weights with a single command, so it isolates the decode contribution
completely. If it holds, fold it into the baseline before judging v2–v5.

**Watch for.** 16 forward passes per well per checkpoint, so re-scoring the OOF
set costs 16× a plain pass; `--n-phases 4` halves it while you iterate.
Re-anchoring feeds the model its
own prediction, so a bad first pass can be reinforced; the original prefix is
hard-constrained, which bounds the damage.

```bash
python -u run.py oof     --variant v1_tta --weights runs/base --out runs
python -u run.py compare --base runs/base --treat runs/v1_tta
```

---

## v2_slope — predict the geology, not the path

**Change.** An auxiliary head predicts the per-column structural slope
`dz_layer/dMD`; the path is recovered by integrating `dTVT = dz_layer − dz`
against the known trajectory. Paired with z_layer-space slope-scaling
augmentation (`p_struct_scale` 0.30 → 0.55).

**Why.** This is 2nd place's central insight and the strongest idea in the field
that Ruby did not use. `z_layer = TVT + Z − b` makes `dTVT = dz_layer − dz` an
identity, and `dz` is known at test time even after PS. So the coefficient on
`dz` is fixed at −1 and never has to be learned; the model only has to estimate
the smooth structural slope, which is piecewise-constant in the real data with
steps in about 10% of columns. It is a strictly easier estimation problem
carrying the same information.

**Watch for.** Integrated quantities drift: a small constant bias in `dz_layer`
becomes a large TVT error 9,000 ft later. That is why `w_dz_integ` supervises
the integrated path as well as the slope, and why the map head is kept as the
primary output rather than replaced. If v2 loses, try `w_dz` 0.35 → 0.15 before
concluding the idea fails — an auxiliary head that dominates the loss stops
being auxiliary.

---

## v3_moves — keep the ambiguity alive through the decoder

**Change.** A second head predicts, at every `(level, column)` anchor, a
distribution over 21 discrete moves `{0, ±2, …, ±20}` ft, teacher-forced on the
ground-truth path. Decoding marginalises it exactly with dynamic programming and
blends 70/30 with the map path.

**Why.** The column-softmax map reads each MD column independently — nothing in
the decode forbids a path that teleports 40 ft between adjacent columns and back.
The DP decode propagates a level marginal forward one column at a time through
the model's own conditional move distribution, so continuity is structural
rather than hoped for. 2nd place built their entire solution on this head and
reached private 5.802 with it alone; the two decoders make different mistakes,
which is exactly what a blend wants.

**Watch for.** The move vocabulary caps movement at 20 ft per 32 ft column. Real
fault jumps exceed that, and the DP path cannot represent them — which is part
of why the blend keeps the map path at 0.70. Teacher forcing only supervises the
anchor on the true path; the rest of the field is trained implicitly by weight
sharing, so the head is cheap but its off-path calibration is unverified. Sweep
`dp_weight` over {0.0, 0.3, 0.5, 1.0} on OOF — it costs one forward pass to
re-score, since both decodes come from the same run.

---

## v4_refgr — fix the reference you are matching against

**Change.** Seven channels: sibling-lateral reference GR aggregated in 0.25 ft
TVT bins from wells sharing the typewell, its coverage mask and mismatch; the
same three built from the well's own pre-PS prefix; and a median-based mismatch.
The typewell itself is corrected by the sibling median residual.

**Why.** 3rd place's clearest statement about the private leaderboard is that
reference-GR design mattered more than any network change — a distance-weighted
sibling reference moved them 5.817 → 5.691. 5th place arrived at the same place
from the other direction: the typewell is smoothed along TVT, so beds below some
thickness are erased from it and it systematically disagrees with the sharper
lateral logs. The laterals are the better estimate of what GR looks like at a
given depth; the typewell should be corrected toward them. The self-prefix
reference is the host's own hint, and 2nd place added it late for a gain.

**Watch for.** This variant lives or dies on the typewell clustering. The real
training set holds 54 master series, but only when typewells are grouped by
curve *overlap*: hashing each curve exactly gives 752 groups for 773 wells and
leaves 739 wells with no siblings at all. `data.link_typewells` reproduces the
54 — if you touch it, check that number first.

The fold-safety surface is also largest here. A sibling bank built from
validation labels leaks the answer straight into a channel, and the CV becomes
fiction. `SiblingBank.profile(exclude_well=...)` removes the query well's
own contribution and `allowed_full` gates whose labels count — both are wired in
`FeatureProvider`, but if you touch this code, re-check them. Note also that this
channel is *stronger at test time than in CV*: test wells sit alongside all 773
training wells, while a CV fold only sees 618.

---

## v5_gate — know what you do not know

**Change.** A Gaussian-NLL head reports a per-column sigma. A small
CNN+BiLSTM gate then mixes four candidate paths — the map path, the particle
filter, the XY plane fit, and the flat-layer prior — with softmax weights that
vary row by row, keyed on candidate disagreement, distance from PS, GR quality
and the predicted sigma. Trained over 3 GroupKFold split patterns instead of 1.

**Why.** Fixed ensemble weights treat the first 100 ft after PS, a 9,000 ft
extrapolation and a missing-GR stretch identically. 3rd place's gate does not,
and their measurements are unusually clean: gating beat an unconstrained BiLSTM
refiner 5.28 vs 5.40, losing in zero of five folds, because non-negative weights
summing to one restrict the output to a convex combination of paths that are
each already plausible — a real constraint with 773 wells. Separately, going
from 3 to 5 split patterns moved their private LB 5.903 → 5.836.

**Watch for.** A convex mixture cannot fix drift shared by every candidate; 3rd
place confirmed that ceiling with an oracle analysis. Each extra split pattern
is another full 5-fold retrain, making this the most expensive iteration by wall
clock — run it at `split_patterns=(0,)` first and add patterns only once the
single-split version has earned them. The gate is also the most
overfittable component here — it trains on OOF candidates, so its own validation
must be nested (base models and gate must share the fold assignment, which
`split_patterns` handles). And 4th place's nested leave-one-group-out result is
worth respecting: fitted ensemble weights lost to equal weights, 6.578 vs 6.456.
Let the gate vary weights *within* a well; keep equal weights *between* members.

---

## How to judge a result

Pooled RMSE is dominated by a handful of catastrophic wells — squaring makes a
40 ft error 400× a 2 ft error, per row. So is any *difference* between two
models. Two runs can differ by 0.2 RMSE because of five wells.

Use the acceptance test from 2nd place, which `run.py compare` prints:

1. per well, `g_w = SSE_treat(w) − SSE_base(w)`;
2. remove wells in descending `|g_w|`, recomputing pooled RMSE each time;
3. `k*` is where the improvement disappears.

Accept only if `k*` comfortably exceeds ~52 wells — the size of the public
leaderboard. 2nd place rejected a candidate that looked 0.28 better because 8
wells out of 773 accounted for nearly all of it, and accepted one whose gain
survived removing 52.

Two more habits from the write-ups, both earned the hard way:

- **The public LB was actively misleading.** Every XY-feature model of Ruby's
  scored worse on public and better on private. 5th place's public and private
  scores were nearly anti-correlated while CV and private rank-correlated well.
  Ruby investigated the CV/LB gap exhaustively, found no legitimate cause, and
  trusted the larger sample. That decision won the competition.
- **Beware fold 0.** 4th place had four configurations win on fold 0 and lose on
  the 5-fold mean. Their run-to-run standard deviation of the 5-fold mean was
  0.10–0.19 RMSE — larger than most of the effects being chased. Report all
  folds, and prefer multiple split patterns over more seeds on one split.

## A suggested order

1. Train `base`, 5 folds × 1 seed. That is the reference every comparison uses.
2. Run `v1_tta` on those weights. Free, and it settles the decode question first.
3. Train `v2_slope` and `v4_refgr` — the two changes most likely to move the
   score, one on the target and one on the input.
4. Train `v3_moves`. Sweep `dp_weight` on OOF; it is a re-score, not a retrain.
5. Train `v5_gate` last: it consumes the others' candidates, so it wants them to
   exist.
6. Blend with equal weights and two-vector XY routing. Check `--fit-weights`
   only to see whether the nested estimate beats equal weights; if it does not,
   keep equal.

Expected shape of the result, based on what the source teams measured for these
same changes in their own pipelines: v1 and v4 are the most likely to hold up,
v2 and v3 are the most likely to be worth more than their standalone number
suggests because they diversify the ensemble, and v5's value is mostly in the
blend rather than as a single model. **None of that is a measurement** — it is
the prior, and the acceptance test exists precisely to overrule it.
