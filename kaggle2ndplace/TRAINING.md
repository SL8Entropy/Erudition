# Training the AnchorCNN locally

Trains **one** model — the 32 ft `plan147/dzl_w1` family — on 80% of the wells and
scores the held-out 20% with pooled RMSE.

```bash
python anchor_train.py --out runs/dzl_w1 --epochs 120 --eval-every 5 --tta 8
```

| file | role |
|---|---|
| `anchor_data.py` | well loading + cache, the 80/20 split, augmentation, the dataset |
| `anchor_train.py` | model build, training loop, holdout decode, scoring, CLI |
| `src/*.py` | **the author's inference modules, unmodified** — model, loss, decoder |

Nothing in `src/` is edited. It already contains everything needed both to build a
sample (`gr2tvt_data.build_compact`) and to train on one (`gr2tvt_model.anchor_loss`);
what was missing was the training-side plumbing, which is the two files above.

## The protocol

**Split.** The 773 well ids are sorted; the first 618 train, the last 155 are held out.
That is the same rule, the same rounding and the same sort order as the 1st-place repo's
`seq_NN_holdout_eval.py`, verified to land on the same wells (`d07aed8f` is the first
holdout well in both) and the same **754,122** scored rows. Numbers from the two
solutions are therefore directly comparable.

**Scored rows.** Every row of a holdout well whose `TVT_input` is NaN — the suffix the
competition asks for. Train wells carry `TVT` everywhere and `TVT_input` on the prefix;
the eval zone is exactly where `TVT_input` runs out.

**Metric.** Pooled RMSE: one `sqrt(mean((TVT - TVT_pred)**2))` over all 754,122 rows,
not a mean of per-well RMSEs. Per-well mean/median/p95 and the worst wells are reported
next to it, because pooled RMSE is dominated by a handful of catastrophic wells — on
this holdout the worst 5 wells of 155 carry roughly half the squared error.

`holdout_predictions.pqt` is written in the schema the 1st-place repo's
`seq_NN_robust_compare.py` expects (`well_id`, `submit_index`, `TVT`, `TVT_pred`), so a
run here can be put through that script's leave-largest-contribution-out test against a
run there:

```bash
python ../kaggle_1st_place/solution/seq_NN_robust_compare.py --base ../kaggle_1st_place/solution/results/0801_V2_ep150 --treat runs/dzl_w1
```

## Result of the reference run

One model, 120 epochs, 618 train wells, scored on the 155 holdout wells / 754,122 rows.
About 70 minutes on an RTX 3050 6 GB. `runs/dzl_w1/`.

| decode | pooled RMSE |
|---|---:|
| single phase (last epoch) | 6.870 ft |
| 8 MD phases, 4 ft spacing | **6.485 ft** |
| 8 MD phases, 8 ft spacing (the author's `[0, 8, … 56]`) | 6.562 ft |

Per-well RMSE mean 4.65, median 3.66, p95 11.33; the worst 5 wells of 155 carry 43% of
the squared error. Best epoch was 99 (6.615) against 6.870 at 120, i.e. the run is on a
noisy plateau, which is what a constant learning rate is supposed to produce.

MD-phase TTA is worth **−0.39 ft** here, far more than the −0.04 to −0.15 the 1st-place
repo measured on its own models. That is consistent with its stated mechanism: it
averages out column-grid aliasing, and a less converged model is more sensitive to it.

The two spacings differ by 0.077 ft, which is at the edge of what this holdout can
resolve — the 1st-place repo measured a same-model, different-seed null draw of 0.030 ft,
and treats anything under ~0.05 ft as unmeasured.

### Against the 1st-place solution, on identical wells and rows

Their best local run is `rb_v3_tta_ep150` at **4.937 ft**. Running their own acceptance
test on this run's predictions file:

```
bootstrap delta p50 = +1.479 ft [+0.421, +2.701], improving in 0.6% of draws
k* = 0    VERDICT: REJECT
```

So this single model is clearly behind their tuned one, which is the expected result: it
is one checkpoint at roughly a quarter of the author's per-model sample budget, against a
150-epoch run of a solution that was itself tuned on this holdout.

What is more interesting is that the difference is **not uniform**. The AnchorCNN wins
enormously on some wells the 1st-place model fails (`f5859199` 15.14 → 4.03,
`d1457cc5` 8.45 → 2.42) and fails on others it handles easily (`ea3a0e38` 2.84 → 23.15,
`e5864244` 5.85 → 26.48). Two methods with uncorrelated catastrophic wells are the
textbook case for ensembling, and pooled RMSE is dominated by exactly those wells.

## Switching models: the resolution ablation

Two flags change the geometry. **Both default to the released `dzl_w1` config, so every
command already in this file keeps running unchanged** — verified by rescoring
`runs/dzl_w1/model_last.pt` after the flags were added and getting 6.5618 ft, bit-identical
to before.

| flag | default | effect |
|---|---|---|
| `--stem-stride` | `1` | `2` uses the stock EfficientNet stem instead of running stage 0 at full input resolution |
| `--row` | `0.5` | vertical grid sampling in ft; `1.0` halves the input rows |

`--row` also changes the **output** grid, because the model derives its state grid as
`tq = T // 4`. So `--row 1.0` gives 256 input rows *and* 64 level bins of 4 ft instead of
128 bins of 2 ft. Pair it with `--n-move 5` to keep the move vocabulary at the same
physical ±20 ft it has in the baseline — otherwise `n_move=10` at 4 ft bins spans ±40 ft,
and since real moves never exceed 8.4 ft per 32 ft column, most of those classes would
never fire. (This is the same reasoning behind the author's own 16 ft family using
`n_move=5`.)

### The four variants

Measured on this GPU at batch 6 / accum 2, with per-epoch times that match the real
baseline run to within 0.4%:

| | `--row` | `--stem-stride` | GFLOP/fwd | ms/sample | s/epoch | 120 epochs | peak VRAM |
|---|---|---|---:|---:|---:|---:|---:|
| **A** baseline | 0.5 | 1 | 15.7 | 45.1 | 27.9 | 56 min | 3.77 GiB |
| **B** | 0.5 | 2 | 7.6 | 16.8 | 10.4 | **21 min** | 1.26 GiB |
| **C** | 1.0 | 1 | 7.9 | 23.1 | 14.3 | 29 min | 1.91 GiB |
| **D** | 1.0 | 2 | 3.8 | 10.0 | 6.2 | **12 min** | 0.64 GiB |

B is 2.7× faster in wall clock against 2.1× in FLOPs, because it also drops clear of the
memory-pressure point this 6 GB card hits above ~4 GiB.

**B is the clean ablation.** It cuts compute while leaving the output grid, the bin size
and the move vocabulary exactly as the baseline has them, so any RMSE change is
attributable to compute alone. C and D additionally coarsen the output resolution, so
they answer a different question — "can the task tolerate 4 ft bins?" — and a loss there
cannot be blamed on the FLOP cut by itself.

### Results

All four at 120 epochs, one seed, 8-phase TTA, scored on the same 155 wells.

| | pooled RMSE | vs A | excl. `ea3a0e38` | plateau mean | plateau std | median well | wins | GFLOP | time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A** baseline | 6.485 | — | 5.949 | 6.865 | 0.167 | 3.665 | — | 15.7 | 56 m |
| **B** stride 2 | 6.505 | +0.019 | 6.526 | 7.750 | 0.713 | 4.113 | 32% | 7.6 | 21 m |
| **C** row 1.0 | **5.942** | −0.544 | 5.955 | **6.389** | 0.382 | 3.679 | 41% | 7.9 | 29 m |
| **D** both | 7.644 | +1.159 | 7.684 | 8.571 | 0.655 | 4.789 | 26% | 3.8 | 12 m |
| **E** row 2.0 | 7.405 | +0.920 | — | 8.006 | 0.418 | 4.838 | — | 3.9 | 18 m |

"plateau mean/std" is over the eight single-phase evals from epoch 60 on — a far more
stable statistic than one endpoint checkpoint. "wins" is the share of the 155 wells where
the variant beats A.

**`stem_stride=2` is not free — the prior behind this ablation was wrong.** B looks
neutral in pooled RMSE (+0.019) but that is an artefact: run the acceptance test in the
direction "is A better than B" and it gives **k\* = 145 of 155**, i.e. A's advantage
survives deleting 145 wells, so the deficit is spread across essentially the whole
holdout. It shows up everywhere else too — plateau mean +0.885, median well 3.665 →
4.113, B wins on only 32% of wells, and B's checkpoint scatter is 4× A's (std 0.713 vs
0.167). Running stage 0 at full input resolution is doing real work, and removing it also
destabilises training. Keep `stem_stride=1`.

**Halving the vertical sampling costs nothing measurable, and may help.** C matches A on
the bulk — median well 3.679 vs 3.665, and excluding one well the pooled figures are
5.955 vs 5.949 — while using **2× fewer FLOPs** and running 1.95× faster. Its plateau
mean is 0.476 *better* than A's. The safe claim is "no measurable loss at half the
compute"; the stronger claim ("C is better") is not established, because its endpoint
pooled gain has **k\* = 1** — it rests on a single well.

**Do not combine them.** D is worse than either change predicts: B alone +0.885 and C
alone −0.476 on plateau means would predict about +0.41, but D is +1.706. The penalty is
super-additive, which fits — D halves the stage-0 resolution twice over.

**One well decides the headline.** `ea3a0e38` is 23.15 ft for A and 3.7–4.9 ft for all
three variants; it alone moves the A-vs-C pooled delta from −0.544 to +0.006. The
1st-place model scores 2.8 ft on it, so it is A that is anomalous, not the variants —
A's 6.485 is likely a pessimistic single-seed draw and the honest baseline is nearer
5.95. This is the same "the wells that decide it change identity every run" effect the
1st-place repo documents, and it is why the endpoint pooled number should never be read
on its own here.

**The resolution ladder stops at `--row 1.0`.** E (`--row 2.0 --n-move 3`: 128 rows → 32
bins of 8 ft) keeps the 2:1 ratio the rule demands and is 4× cheaper than A, but it is
**1.46 ft worse than C**, with `k* = 124` of 155 and **100% of bootstrap draws** favouring
C — the most decisive result in this whole study. The damage is broad, not tail-driven:
median well 4.838 against C's 3.679.

The mechanism is the one predicted before the run. At 8 ft bins the maximum real move of
8.4 ft per 32 ft column is about one bin, so the move distribution collapses onto {0, ±1}
and carries almost no information.

Do not read the cross-entropy across these variants. Final CE is A 0.49 (21 classes),
C 0.33 (11 classes), E 0.22 (7 classes): it falls mechanically as the vocabulary shrinks,
and it ranks C below A even though C decodes at least as well. **CE is only comparable
between runs with the same bin size and `--n-move`**; across them, compare decoded RMSE.

One caveat: E was still descending at 120 epochs (−0.435 ft over epochs 90→119, while A
was flat at +0.154), so it is undertrained. But it would have to find another 1.4 ft to
reach C, for a saving of only 2× over C, so the trade is bad regardless.

Net: the 2:1 rule is necessary but not sufficient. Resolution can come down until the
move vocabulary stops resolving the physical step size, and that limit sits between 4 ft
and 8 ft bins.

**Confound to close.** C changed two things at once: `--row 1.0` *and* `--n-move 5`. To
attribute the result to resolution rather than to the narrower vocabulary, run
`--row 1.0` with `--n-move 10` (29 min). A clean `--n-move 5` control at `--row 0.5` is
not available — it would cap moves at ±10 ft, below the 8.4 ft maximum in the data.

Next step if you want C settled: 2–3 seeds of C against 2–3 of A. At 29 and 56 minutes
that is roughly 4 hours, and it is the only thing that will resolve a 0.5 ft effect on
155 wells.

### Commands

Run from `kaggle2ndplace/`. `runs/dzl_w1` is already variant A, so it needs no re-run.

```bash
python anchor_train.py --out runs/res_B_stride2 --stem-stride 2 --epochs 120 --eval-every 5 --tta 8
```
```bash
python anchor_train.py --out runs/res_C_row1 --row 1.0 --n-move 5 --epochs 120 --eval-every 5 --tta 8
```
```bash
python anchor_train.py --out runs/res_D_both --row 1.0 --stem-stride 2 --n-move 5 --epochs 120 --eval-every 5 --tta 8
```

Keep `--batch-size 6 --grad-accum 2` (the defaults) across all four. B and D leave a lot
of VRAM free and would run faster at a larger batch, but changing it changes the effective
batch and the BatchNorm statistics, which would confound the comparison.

Rescoring a variant **must repeat its geometry flags**, or the grid will not match the
checkpoint:

```bash
python anchor_eval.py --models runs/res_D_both/model_last.pt --out runs/res_D_both_tta8 --row 1.0 --stem-stride 2 --n-move 5 --tta 8
```

Judging a variant against the baseline, with the 1st-place repo's acceptance test:

```bash
python ../kaggle_1st_place/solution/seq_NN_robust_compare.py --base runs/dzl_w1 --treat runs/res_B_stride2 --output-dir runs/res_B_stride2
```

B + C + D together are about 62 minutes of GPU. Remember the noise floor: a single seed
here cannot resolve differences under ~0.05 ft, so read a small regression as "no
measurable cost", not as a win for the baseline.

## Round 2: cost and ensemble experiments

Every run below **adopts the two proven changes**: the C geometry (`--row 1.0 --n-move 5`)
and batch 6 (the default). Nothing here changes a default, so every command earlier in
this file still reproduces its old result — checked by rescoring `runs/res_C_row1` through
the new code: 5.9417 ft against 5.9416 at training time, the size of CUDA nondeterminism.

| id | tests | new code | depends on | measured time |
|---|---|---|---|---:|
| `C_s1..3` | the control: C geometry at 2x the earlier sample budget, 3 seeds | — | — | ~60 min each |
| `X_antialias` | fold the 8 MD phases into the model: GR prefilter before column binning | `--gr-prefilter-ft` | — | ~60 min |
| `X_separable` | 1-D encoders + matching volume instead of EfficientNet | `anchor_separable.py` | — | ~31 min |
| `X_distill` | one model distilled from the 3-seed ensemble's move field | `--distill-teachers` | `C_s1..3` | ~60 min |
| `X_int8` | 8-bit quantization: post-training, then quantization-aware fine-tune | `anchor_quantize.py` | `C_s1` | ~60 min |
| `X_ensemble` | AnchorCNN + 1st-place ConvNeXt, blended by a cross-fitted gate | `anchor_ensemble.py` | `C_s1..3` | ~2 min |

Times are from real epochs on this machine, not estimates: per 240 samples, C takes 6 s,
separable 3 s, distillation with 3 teachers 11 s, quantization-aware training 46 s.

### What each one is

**Anti-aliased GR (`--gr-prefilter-ft 32`).** The input box-averages GR into 32 ft
columns at a fixed phase, which aliases everything shorter than a column; the 8-phase TTA
averages that out afterwards, for 8x the inference. A 32 ft moving average applied before
the column box turns the effective kernel into a 64 ft triangle, the standard
anti-aliasing fix. Gaps are never filled from neighbours. **Success looks like** this
run's *one-phase* score matching the control's *8-phase* score — i.e. `tta_gain` near
zero with no loss in the 8-phase number. One honest doubt: TTA also acts as a small
ensemble of 8 views, and a prefilter cannot reproduce that part.

**Separable encoder (`--arch separable`).** The 9-channel input image is a broadcast of
~3.7k one-dimensional numbers. This model keeps them 1-D: a residual 1-D CNN over the
typewell profile, another over the lateral log, their outer difference and product (plus
the raw mismatch and known-path channels) as a matching volume, then a light 2-D pyramid.
Its four feature maps are at *exactly* the baseline trunk's resolutions — (128, 168),
(64, 84), (32, 42), (16, 21) — so the 2:1 rule from the ablation holds. Decoder, heads,
loss and DP are unchanged, so any difference is the encoder. Measured: **1.35M params
against 3.74M, 4.82 GFLOP against 7.85 (1.6x), 2x faster training.** Two confounds to
keep in mind: it starts from random weights while the baseline trunk starts from ImageNet,
and it runs the same sample budget as the control, which is why it finishes in half the
time.

**Distillation (`--distill-teachers`).** Teacher-forced cross-entropy only ever supervises
the one anchor on the true path. The loss added here is KL(teacher ‖ student) over the
*whole* move field — every covered level at every predicted column — averaged over the
three frozen control seeds. That field says how the well would move *if* it were on a
different layer, which is exactly what the DP marginalises when a match is ambiguous.
Deliberately run at `--epoch-len 618 --seed 46013`, the same budget and seed as
`runs/res_C_row1`, so it is a clean with/without-distillation pair at equal data; the
teachers cost ~1.8x per step, which is why the budget is the smaller one. Judge it two
ways: against `res_C_row1` (did distillation help a single model?) and against
`C_seeds_tta8` (how much of the 3-model ensemble does one model keep, at a third of the
inference?).

**8-bit quantization (`anchor_quantize.py`).** FX graph-mode int8 on the x86 backend,
calibrated on 64 training wells, then 25 epochs of quantization-aware fine-tuning. The DP
decoder stays in float. Measured in smoke tests, and worth knowing before spending the
hour:

- **int8 was slower than float32 on this CPU**: 0.64x at 1 thread, 0.78x at 8.
  EfficientNet's SiLU has no int8 kernel, so every block converts back to float around
  its activations. The weights do shrink 3.2x (15.3 → 4.8 MB).
- **Post-training quantization destroyed accuracy** (tens of feet on a 12-well subset),
  and keeping the stem, heads or whole decoder in float did not rescue it — the damage is
  in the trunk. A 2-epoch quantization-aware fine-tune roughly halved the error, which is
  why the real run uses 25.

So the hour answers whether accuracy is recoverable at int8. On *this* hardware with
PyTorch it will not deliver a speedup; it only matters if the target has real int8
acceleration (TensorRT, OpenVINO, a Jetson, an NPU). `--keep-float ends|ends+decoder`
exists for mixed-precision variants.

**A more decisive gate (`--gate-ce-w`).** 3rd place trained their gate on two terms:
`MSE(blend) + CE(w, best candidate)`. The first asks for the most accurate mixture; the
second asks the weight to *commit* to whichever candidate is actually closer at that row,
instead of hedging near 0.5. With two candidates that cross-entropy is a binary one on the
weight itself.

Passing `--gate-ce-w > 0` reports a second method, `gate_ce`, beside the plain `gate`,
cross-fitted identically, so one run compares them directly. Default is 0, so runs without
the flag are unchanged (verified: `gate` reproduces 4.6452 ft on a fixed 40-well subset
either way). Early stopping always watches the *pure* validation MSE, because pooled RMSE
is what the run is scored on; the extra term only shapes training.

Mind the scale: the MSE term is in ft² (order 20 here) and the cross-entropy is order 0.7,
so `--gate-ce-w` has to be of order 10 before it bites. Both terms and a `decisiveness`
statistic (0 = always 50/50, 1 = always a hard pick) are logged per fold and per method.

**Ensemble (`anchor_ensemble.py`).** Five methods on identical rows: ConvNeXt alone,
AnchorCNN alone, a fixed 50/50 mean, one cross-fitted blend weight, and a per-row gate
(a small bidirectional GRU whose output is a convex weight, initialised at 0.5 so it starts
as the plain mean), plus a row-wise oracle as an upper bound. The gate's inputs include
the AnchorCNN's belief spread from `--save-std` — which, measured on `res_C_row1`,
**correlates with its actual error at Spearman rho = 0.61**, so the model does know when
it is unsure. It is leak-free by cross-fitting: the 155 holdout wells are split into 5
groups, each well's blend comes from a gate fitted on the other four (early-stopped on an
inner split), and no hyperparameter is tuned on the result. The caveat is size — each
gate learns from ~124 wells.

### Commands

Run from `kaggle2ndplace/`, one at a time (two GPU trainings at once on 6 GB just slow
each other down). The order respects the dependencies.

**1. Control, 3 seeds (~3 h):**

```bash
python anchor_train.py --out runs/C_s1 --row 1.0 --n-move 5 --epoch-len 1150 --epochs 120 --eval-every 5 --tta 8 --seed 1
```
```bash
python anchor_train.py --out runs/C_s2 --row 1.0 --n-move 5 --epoch-len 1150 --epochs 120 --eval-every 5 --tta 8 --seed 2
```
```bash
python anchor_train.py --out runs/C_s3 --row 1.0 --n-move 5 --epoch-len 1150 --epochs 120 --eval-every 5 --tta 8 --seed 3
```

**2. Anti-aliased GR (~60 min).** Seed 1, so it is paired with `C_s1`: same model
construction, same augmentation draws, same data order.

```bash
python anchor_train.py --out runs/X_antialias --row 1.0 --n-move 5 --epoch-len 1150 --epochs 120 --eval-every 5 --tta 8 --seed 1 --gr-prefilter-ft 32
```

**3. Separable encoder (~31 min):**

```bash
python anchor_train.py --out runs/X_separable --arch separable --row 1.0 --n-move 5 --epoch-len 1150 --epochs 120 --eval-every 5 --tta 8 --seed 1
```

**4. The 3-seed ensemble, scored once (~2 min)** — needed by both 5 and 7:

```bash
python anchor_eval.py --models runs/C_s1/model_last.pt runs/C_s2/model_last.pt runs/C_s3/model_last.pt --out runs/C_seeds_tta8 --row 1.0 --n-move 5 --tta 8 --save-std
```

**5. Distillation (~60 min):**

```bash
python anchor_train.py --out runs/X_distill --row 1.0 --n-move 5 --epoch-len 618 --epochs 120 --eval-every 5 --tta 8 --seed 46013 --distill-teachers runs/C_s1/model_last.pt runs/C_s2/model_last.pt runs/C_s3/model_last.pt
```

**6. 8-bit quantization (~60 min):**

```bash
python anchor_quantize.py --model runs/C_s1/model_last.pt --out runs/X_int8 --row 1.0 --n-move 5 --tta 8 --calib-wells 64 --threads 1 4 8 --qat-epochs 25 --qat-epoch-len 618
```

**7. ConvNeXt + AnchorCNN ensemble (~2 min; measured 56 s on 80 wells):**

```bash
python anchor_ensemble.py --anchor runs/C_seeds_tta8/holdout_predictions.pqt --convnext ../kaggle_1st_place/solution/results/rb_v3_tta_ep150/holdout_predictions.pqt --out runs/X_ensemble
```

Add 3rd place's second loss term to get the decisive gate reported alongside the plain one
(slower, because those fits early-stop later):

```bash
python anchor_ensemble.py --anchor runs/C_seeds_tta8/holdout_predictions.pqt --convnext ../kaggle_1st_place/solution/results/rb_v3_tta_ep150/holdout_predictions.pqt --out runs/X_ensemble_ce --gate-ce-w 10
```

### Round 2 results

Control: three seeds of C at `--epoch-len 1150`, pooled **6.024 / 6.200 / 6.565**
(mean 6.263, sd 0.276). One-phase mean 6.512. That spread is the yardstick.

| run | pooled | one phase | median well | verdict |
|---|---:|---:|---:|---|
| control seeds | 6.024–6.565 | 6.415–6.690 | 3.42–3.61 | — |
| 3-seed ensemble | 5.993 | — | 3.442 | k\*=1 vs best seed: nothing |
| `X_antialias` | 6.110 | **6.737** | 3.530 | **failed its purpose** |
| `X_distill` | 6.252 | 6.366 | 3.455 | no gain; see below |
| `X_separable` | **11.395** | 11.556 | 6.417 | **failed** |
| `X_int8` (QAT) | **10.795** | 11.498 | — | **failed, and slower** |
| `X_ensemble` best blend | **4.905** | — | 2.761 | +0.03 over ConvNeXt alone |

**int8 is a dead end here, for two independent reasons.** Post-training quantization gave
25.06 ft and 25 epochs of quantization-aware training only got to 10.79 ft, against 5.98
for the same checkpoint in float32. And it is *slower* on this CPU at every thread count
measured — 0.63x at 1 thread, 0.76x at 4, 0.74x at 8. The only gain is file size, 15.3 →
4.8 MB. Revisit only on hardware with real int8 acceleration, and the accuracy problem
would still need solving.

**The separable encoder failed, and not marginally** — 11.4 ft against a control of
6.0–6.6. It was still creeping down at the end (−0.16 ft over the last 30 epochs) but
would need to find 5.4 ft. Its confounds are real (no ImageNet initialisation, same
sample budget as a model that has one) and yet cannot explain a gap that size. Taken with
the earlier finding that cheapening the stem costs 0.9 ft, the encoder appears to be doing
substantially more than re-deriving an outer product.

**Anti-aliasing did not absorb the TTA.** Its whole point was to make one phase as good as
eight. Instead its one-phase score (6.737) is worse than every control seed's (6.415,
6.415, 6.690), and its TTA gain *grew* to 0.627 against a control mean of 0.25. Its
8-phase score is unremarkable (inside the seed range), so it did no harm — it simply did
not do its job. The likely reading: the 8 phases are not only correcting aliasing, they
are acting as an 8-member ensemble, and a prefilter cannot reproduce that.

**Distillation had nothing to transfer.** The student came out at 6.252 against its
matched no-distillation twin `res_C_row1` at 5.942 (k\*=7, below the bar of 10, 89% of
bootstrap draws favouring the twin) — suggestive of harm, not conclusive. But the test was
flawed: the teacher ensemble scored 5.993, i.e. *no better than the student's own
baseline*, so there was no ensemble advantage to distil in the first place. Worth one
retry only with teachers that actually beat the baseline. Note the student's median well
did improve (3.455 vs 3.679); it lost on the tail.

**Seed ensembling is nearly worthless here.** Three seeds averaged give 5.993 against the
best single seed's 6.024 — k\*=1, 55% of bootstrap draws. It recovers roughly the luck of
seed selection (the seed mean is 6.263) and no more, for 3x the inference.

**On combining with the ConvNeXt, the simplest blend wins and the learned gate loses.**

| method | pooled | vs ConvNeXt alone |
|---|---:|---|
| ConvNeXt alone | 4.937 | — |
| AnchorCNN (3 seeds) | 5.993 | +1.06 |
| 50/50 mean | 5.035 | +0.10 |
| one fitted weight (~0.2 anchor) | **4.905** | −0.032, k\*=154/155, ACCEPT |
| learned per-row gate | 5.017 | +0.080, k\*=0, REJECT |
| gate + 3rd place's CE term | 5.013 | +0.076 |
| row-wise oracle | 4.011 | −0.93 (uses the answer) |

The single fitted weight improves on ConvNeXt by 0.032 ft, and the acceptance test puts
k\* at **154 of 155** — the gain is spread across essentially every well rather than
riding on a few. But 0.032 ft is below this holdout's resolution and the bootstrap
interval crosses zero, so treat it as "blending in ~20% AnchorCNN costs nothing and may
help a hair", not as a win. The learned gate, with or without the decisiveness term, is
worse than doing nothing — 124 wells is not enough to fit a per-row chooser, exactly the
direction 3rd place warned about when they found their constrained gate beat an
unconstrained refiner.

**The complementarity was weaker than the two cherry-picked wells suggested.** Across all
155 wells the AnchorCNN beats the ConvNeXt on only 45%, wins by more than 3 ft on 6 wells
while losing by more than 3 ft on 22, and their per-well RMSEs correlate at 0.657. The
errors are substantially correlated and one model is simply better. The 0.93 ft oracle gap
says per-row selection *could* pay, but nothing tried here captures any of it.

**The doubled sample budget did not help.** `res_C_row1` at `--epoch-len 618` scored 5.942,
better than all three 2x-budget seeds. One seed against three is not proof, but there is no
evidence for "train on more material", which had been a recommendation.

## Swapping the backbone: a ConvNeXt trunk in the AnchorCNN

There is a ~1 ft gap between this AnchorCNN (~6.0) and the 1st-place ConvNeXt (4.94) on
identical wells, and four things could explain it: the backbone, their extra geo-prior
input channels, their longer tuned recipe, or their output formulation. `--backbone
convnext_small` isolates the first one — everything else stays Bilzard's.

**The stem has to be retrofitted or the comparison is rigged.** ConvNeXt starts with a 4x4
stride-4 patchify conv, which puts its finest feature map at input/4 — exactly the output
grid, a 1:1 ratio. Both 1:1 configurations in the resolution ablation failed (+0.9 and
+1.7 ft), so a stock ConvNeXt would lose here for reasons unrelated to ConvNeXt.
`anchor_convnext.retrofit_stem` replaces it with a 3x3 stride-2 conv, which reproduces the
EfficientNet baseline's geometry exactly (measured):

| trunk | feature rows | cols | output bins | ratio |
|---|---|---|---:|---:|
| EfficientNet-B0, `stem_stride=1` | [128, 64, 32, 16] | [168, 84, 42, 21] | 64 | 2.0x |
| ConvNeXt-small, retrofitted | [128, 64, 32, 16] | [168, 84, 42, 21] | 64 | 2.0x |
| ConvNeXt-small, stock (`--convnext-stem-stride 4`) | [64, 32, 16, 8] | — | 64 | 1.0x |

The 1st-place pipeline solves the same problem the same way: its notes record that the
patchify stem is skipped and a 1x1 conv feeds stage 0.

Pretrained weights come from the local HuggingFace cache. `HF_HUB_OFFLINE` is read at
import time so setting it from inside the process is too late; `cached_weights()` finds the
snapshot and hands the file to timm directly, the same route the EfficientNet weights take.
The replacement stem is randomly initialised, so the ConvNeXt keeps its pretrained stages
but loses its pretrained stem — state that with any result.

**It is not a compute-matched comparison.** Measured on this GPU:

| trunk | params | GFLOP/fwd | ms/sample | peak VRAM (batch 4 / 6) | 150 epochs |
|---|---:|---:|---:|---:|---:|
| EfficientNet-B0 | 3.7M | 7.8 | 24 | 1.3 / 1.9 GiB | ~70 min |
| ConvNeXt-small | 49.7M | **122.1** | 69–73 | 2.9 / 4.6 GiB | **~3.5 h** |

15.6x the compute per sample, because a stride-2 stem quadruples every stage's resolution
relative to what ConvNeXt was designed for. Use `--batch-size 4 --grad-accum 3` (effective
batch 12, same as the default 6x2): batch 6 peaks at 4.6 GiB, past the ~4 GiB point where
this card's throughput collapses.

### Commands

The control first, so a usable comparison exists after ~70 minutes rather than 4.5 hours:

```bash
python -u anchor_train.py --out runs/C_s1_ep150 --row 1.0 --n-move 5 --epoch-len 1150 --epochs 150 --eval-every 5 --tta 8 --seed 1 && python -u anchor_train.py --out runs/X_convnext --backbone convnext_small --convnext-stem-stride 2 --row 1.0 --n-move 5 --epoch-len 1150 --epochs 150 --eval-every 5 --tta 8 --seed 1 --batch-size 4 --grad-accum 3
```

Then:

```bash
python ../kaggle_1st_place/solution/seq_NN_robust_compare.py --base runs/C_s1_ep150 --treat runs/X_convnext --output-dir runs/X_convnext
```

Rescoring a ConvNeXt checkpoint must repeat `--backbone convnext_small
--convnext-stem-stride 2`, or the rebuilt architecture will not match the weights.

## Ensemble partners: what the correlation screen found

Blend gain is governed by how *decorrelated* a partner is, not how accurate. Screening
every prediction set already scored on these 155 wells (`anchor_pf.py --compare` prints
this for any new one):

| partner | RMSE | residual corr | best blend | gain |
|---|---:|---:|---:|---:|
| AnchorCNN `C_s1` | 6.02 | **0.669** | 4.838 | −0.099 |
| eight 1st-place variants | 4.98–5.52 | 0.84–0.97 | 4.85–4.92 | −0.02…−0.09 |
| their non-ML geo prior | 11.57 | 0.510 | 4.937 | 0.000 |
| **particle filter** (`anchor_pf.py`) | 28.98 | **0.201** | 4.937 | 0.000 |

Two things follow. Our AnchorCNN is the *best partner that exists* for that ConvNeXt —
more decorrelated than any of the 1st place's own variants, because it is a genuinely
different formulation. And a physics tracker decorrelates far further (0.20), which is
exactly the hypothesis: architecture diversity buys ρ≈0.7, method diversity buys ρ≈0.2.

**The prize, if a competitive tracker existed.** At ρ=0.20:

| tracker accuracy | blend | gain |
|---|---:|---:|
| 6 ft (3rd place's HMM was 5.97) | 4.166 | **−0.77** |
| 8 ft | 4.542 | **−0.40** |
| 12 ft | 4.826 | −0.11 |
| 29 ft (what we have) | 4.935 | −0.003 |

That is 4–25x everything extracted from ensembling so far, and it is the strongest
argument for building an HMM next.

**But this particular filter is far too weak.** 28.98 ft, worse than predicting a flat
line (9.8 ft). The failure is drift, cleanly: mean |error| grows 4.3 → 10.9 → 17.2 → 23.4
→ 29.7 ft across the five fifths of the lateral. It tracks well and then loses the thread.
Tuning its likelihood weight and drift rate on 80 *training* wells (leak-free — the filter
fits nothing) moved it only to 26.7, and a 100x sweep of the likelihood weight changed
nothing, so it is not a tuning problem.

It is also not an integration problem: the `Well` built by `to_champion_well` is
byte-identical to `rogii_champion`'s own loader on every field. The filter was written to
generate a *channel* for a CNN, not to predict alone, and its own docstring says so.

### Fixing the particle filter: 28.98 -> 16.82 ft

The drift had one cause, and it is measurable rather than a matter of taste. On 200
training wells the structural slope over the eval zone spans [-0.041, +0.046], and the
prefix-tail estimate of it is wrong by 0.021 at p90 — worth 103 ft of drift at p90 if it
is never corrected. The filter's slope random walk can wander about 0.006 over a whole
lateral, so no particle ever carries the right slope.

The first tuning grid swept the slope walk *downward* (x0.25, x1) and the likelihood
weight up to x100, and nothing helped, because neither is the mechanism. Sweeping the
slope walk **upward** does:

| rate_std multiplier | x1 | x4 | x6 | **x9** | x12 | x18 | x64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RMSE on training wells | 36.4 | 17.4 | 15.7 | **15.3** | 17.2 | 22.4 | 81.1 |

Holdout: **28.98 -> 16.82 ft**, an interior optimum at x9 (`--tune-rates`).

**And it still contributes nothing.** A partner only helps when its RMSE is below
`sigma_convnext / rho` — below that line the blend weight goes negative and is clipped to
zero. The catch is that the threshold moves as the tracker improves, because a tracker
that follows the GR better also agrees with the CNN more:

| PF version | RMSE | corr | break-even | blend weight |
|---|---:|---:|---:|---:|
| as shipped | 28.98 | 0.201 | 24.6 ft | 0.000 |
| slope walk x4 | 18.67 | 0.276 | 17.9 ft | 0.000 |
| slope walk x9 | **16.82** | 0.296 | 16.7 ft | 0.005 (-0.0006 ft) |

It closed from 4.4 ft above break-even to 0.14 ft above, and sits exactly on the line. At
rho ~ 0.30 the prize is smaller than the rho = 0.20 table suggested: a tracker at 10 ft is
worth -0.10 ft, at 8 ft -0.25 ft, at 6 ft -0.61 ft. **A tracker has to reach about 8 ft to
be worth anything**, which is where 3rd place's HMM (5.97) sat.

`anchor_hmm.py` is built and runs (0.11 s/well, exact forward-backward over 129 levels x
53 slopes, prefix clamped) but is not competitive yet: ~23 ft at its best setting so far,
worse than the repaired particle filter. Its slope grid is already wide enough by
construction, so its problem is elsewhere; lower likelihood weights help, which is where
tuning should start.

### Attacking the tail: a systematic bias, not mode collapse

Before correcting anything, the error was one-directional and growing — quarter by quarter
across the lateral: **-1.69, -6.11, -8.54, -9.62 ft**, negative in 72% of wells, mean
-6.49. The twenty worst wells had |mean error| / RMSE = 0.87 with *zero* sign flips, i.e.
they were not wandering, they were leaning. The filter under-estimates the structural
slope in a consistent direction, so every well drifts the same way and the tail wells are
simply where it compounds longest.

Two parameters fitted on training wells (`--calibrate`) remove it: `err = -1.01 ft
-0.00260 x (MD - MD_PS)`, a slope bias worth **-13 ft over a 5000 ft lateral**.

| stage | RMSE | corr | blend weight |
|---|---:|---:|---:|
| as shipped | 28.98 | 0.201 | 0.000 |
| slope random walk x9 | 16.82 | 0.296 | 0.005 |
| **+ drift calibration** | **15.84** | 0.266 | 0.016 (-0.006 ft) |

**Mode collapse is ruled out.** Sweeping the knobs that would fix a degenerate particle
cloud — 512 vs 4096 particles, jump probability 0.004 vs 0.03, resampling threshold 0.5 vs
0.2 — moves nothing: all eight configurations land between 16.5 and 17.3 ft, and eight
times the particles changes the answer by 0.02 ft. So the remaining tail is not a sampling
artefact; the posterior genuinely prefers the wrong layer on those wells, which is a
statement about the *evidence*, not the inference.

That points at the reference profile, and it is the same conclusion 3rd place reached from
the other direction when they called reference-GR choice their biggest lever. A tracker's
only evidence is GR against the reference, so a well whose typewell does not represent it
has nothing to lock onto. **Sibling-lateral reference GR is the next thing to build**, and
it would help the AnchorCNN and the tracker for the same reason.

### Per-well dip correction (`anchor_dip.py`)

The 8th-place team's error decomposition -- "the error is per-well dip lock-in, not tail
difficulty" -- reproduces on every model here, theirs included:

| | baseline | − per-well offset | − per-well linear trend |
|---|---:|---:|---:|
| ConvNeXt (1st place) | 4.937 | 3.761 | **3.072** |
| AnchorCNN | 6.024 | 4.148 | **3.269** |
| particle filter | 15.845 | 9.491 | **5.478** |
| *8th place U-Net* | *6.44* | *4.58* | *3.33* |

One wrong line per well is worth **1.9 ft even on the best model available**, which is
sixty times anything ensembling has produced. The filter is the extreme case: its shape is
fine and nearly all its error is a single wrong dip.

A global two-parameter calibration only removes the population average of that.
`anchor_dip.py` predicts each well's own line -- two ridge regressions for the error's
offset and slope, from twelve features available at inference (prefix slope, the filter's
own implied slope, neighbouring *training* wells' structural slopes, trajectory slope, GR
coverage, filter spread). A training well is never its own neighbour.

This is only possible for the filter because it needs no training, so it can be run over
all 618 training wells and the corrector fitted there with the holdout untouched. The CNNs
would need out-of-fold predictions, which we do not have.

| stage | RMSE | median well | corr | blend weight | blend gain |
|---|---:|---:|---:|---:|---:|
| as shipped | 28.98 | — | 0.201 | 0.000 | 0.000 |
| slope random walk x9 | 16.82 | 10.05 | 0.296 | 0.005 | −0.001 |
| + global drift calibration | 15.84 | 8.97 | 0.266 | 0.016 | −0.006 |
| **+ per-well dip corrector** | **13.07** | **7.68** | 0.308 | 0.030 | **−0.014** |

**28.98 → 13.07 ft, a 55% cut**, median well 7.68. The prediction that neighbouring wells'
dip would be the dominant feature was wrong: the strongest predictor of the error slope is
`dz_mean`, the well's *own* trajectory slope, with the neighbour slope fourth. Since
dTVT = dz_layer − dz, the filter is mishandling the part of the motion that is already
known exactly.

Still only −0.014 ft on the blend, because the bar is set by a 4.937 ft partner. Reaching
10 ft would be worth ≈ −0.10, and 8 ft ≈ −0.25. The median well is already 7.7 ft; the
pooled figure is held up by the tail (p95 26.8).

## The gate, with its initialisation bug fixed

The first gate was pinned at w=0.5 by its initialisation and by weight decay on the output
bias, so it reproduced the plain mean. Fixed — it now starts at the fitted constant weight
and the bias is exempt from decay — it moves decisively (decisiveness 0.63, mean w 0.20)
and gets the **best median well of any method, 2.747** against 2.835 for the ConvNeXt
alone. But pooled it is **5.178** against 4.870 for a single constant weight.

So per-row gating genuinely helps typical wells and hurts catastrophic ones, where it
picks confidently and wrongly. With the confound removed the conclusion is firmer than
before: **a single fitted weight beats a learned per-row gate here.** Using one AnchorCNN
seed rather than the 3-seed average also blends better (4.870 vs 4.905) at a third of the
inference, because seed-averaging removes the idiosyncrasy that made the partner useful.

### Reading the results

One table for everything, with the control seeds' mean and spread:

```bash
python anchor_summary.py --seeds C_s1 C_s2 C_s3
```

**The control's seed spread is the yardstick.** Each treatment is one seed, so a result
inside the min–max range of `C_s1..3` is not distinguishable from the control, however it
looks. Then the acceptance test, pairing each treatment with its matched control:

```bash
python ../kaggle_1st_place/solution/seq_NN_robust_compare.py --base runs/C_s1 --treat runs/X_antialias --output-dir runs/X_antialias
```
```bash
python ../kaggle_1st_place/solution/seq_NN_robust_compare.py --base runs/C_s1 --treat runs/X_separable --output-dir runs/X_separable
```
```bash
python ../kaggle_1st_place/solution/seq_NN_robust_compare.py --base runs/res_C_row1 --treat runs/X_distill --output-dir runs/X_distill
```
```bash
python ../kaggle_1st_place/solution/seq_NN_robust_compare.py --base runs/C_seeds_tta8 --treat runs/X_distill --output-dir runs/X_distill
```
```bash
python ../kaggle_1st_place/solution/seq_NN_robust_compare.py --base runs/C_s1 --treat runs/X_int8 --output-dir runs/X_int8
```
```bash
python ../kaggle_1st_place/solution/seq_NN_robust_compare.py --base ../kaggle_1st_place/solution/results/rb_v3_tta_ep150 --treat runs/X_ensemble/gate --output-dir runs/X_ensemble/gate
```

For the anti-aliasing question specifically, the comparison that matters is its *one-phase*
predictions against the control's 8-phase ones:

```bash
python anchor_eval.py --models runs/X_antialias/model_last.pt --out runs/X_antialias_1phase --row 1.0 --n-move 5 --tta 1 --gr-prefilter-ft 32
```
```bash
python ../kaggle_1st_place/solution/seq_NN_robust_compare.py --base runs/C_s1 --treat runs/X_antialias_1phase --output-dir runs/X_antialias_1phase
```

Rescoring any of these later must repeat the flags it was trained with — `--arch`,
`--row`, `--n-move`, `--gr-prefilter-ft` — or the grid will not match the checkpoint.

## Is the pipeline right?

The architecture defaults were read back from the released checkpoint
`model/plan147/dzl_w1/ep119_seed46013_fold0.pt`, not guessed: 9 input channels, `d=64`,
2 fuse blocks, `n_move=10` (a 21-class move vocabulary), `dzl_arch="linear"`,
`dzl_film="full"`. A freshly built model matches that checkpoint with **0 missing, 0
unexpected and 0 mismatched tensors**.

Because they match, the author's own trained weights can be pushed through this
repository's data pipeline and scoring code. Doing so gives **4.03 ft** pooled RMSE over
the 155 holdout wells. If the channel construction, the anchoring, the DP decode or the
column→MD mapping were wrong, that number would be nonsense. It is *not* a
generalisation estimate — those checkpoints were trained on all 773 wells, so they have
seen the holdout — it is a correctness check on the plumbing.

Two other measured floors, for calibration:

| reference | pooled RMSE |
|---|---|
| grid quantisation (ground-truth labels pushed through the column grid) | 0.06 ft |
| constant TVT_PS (predict the last known value) | ~9.8 ft |
| released 15-checkpoint family, one checkpoint, holdout-contaminated | 4.03 ft |

## What the input looks like

A `(9, 512, 336)` tensor per well: 512 rows of typewell level (`TVT - TVT_PS`, ±128 ft
at 0.5 ft/row) by 336 columns of MD (32 ft each — 16 known pre-PS columns then 320
predicted ones). The head emits a `(21, 128, 336)` field of move distributions
`P(dTVT | level, MD)` over `{0, ±2, …, ±20}` ft per column, which
`dp_expected_level` marginalises into the expected path.

Channels 0–5 come from `assemble_x`, 6–8 from the flags this trainer sets
(`ps_gr=True`, `z_dip=True`): GR mismatch against the typewell, lateral GR, GR validity,
typewell coverage, the known-TVT ridge, the known-column flag, GR mismatch against the
well's own pre-PS profile, its coverage, and dip z.

## Augmentation, and one thing deliberately left out

Applied to real wells, all of them exactly consistency-preserving:

- **re-anchor** — pick a new prediction-start row, so one well yields many samples
- **level shift** — move the ±128 ft window by up to ±32 ft (`tvt_shift`)
- **MD phase** — move the 32 ft column boundaries within one column (`md_phase`)
- **MD flip** — traverse the well backwards
- **level flip** — mirror TVT, the typewell TVT axis and Z about one common constant
- **specaug** — blank spans of GR, in the same format as real missing data

Two constraints keep a re-anchored sample representable on the grid, and both are
derived from the data rather than assumed. A well's known prefix is *not* all lateral:
it contains the curve landing into it, where TVT sweeps ~700 ft and moves reach ~40 ft
per 32 ft column. The move vocabulary stops at ±20 ft and the window at ±128 ft, so a
crop taken there could not be expressed at all — the label would be silently clipped and
the supervision would be a lie. `lateral_span()` finds the longest run of representable
columns, and a deviation cap keeps the label inside the window with the level shift's
budget subtracted. Without them, 4.3% of moves fell outside the vocabulary and labels
reached 1021 ft; with them, 0.01% and 118 ft.

**Structural scaling in z_layer space is not applied to real wells, on purpose.**
Stretching the formation slope maps TVT by an amount that varies along the well, so the
observed GR would no longer equal f(TVT) for *any* typewell f. It is only consistent
when the GR is regenerated from the typewell — i.e. on synthetic wells. Applying it to a
real well would corrupt the GR↔TVT correspondence the whole method rests on.

## Synthetic wells: partial

`--synth-prob` (default `0`, off) enables `anchor_data.synth_well`, which implements
steps 3–5 of the writeup's pipeline: mix two real TVT trajectories, read GR back out of
a typewell along the mixed path, and paste a real well's residual on top.

As the writeup requires, `z_layer` is the internal representation: the source well's real
formation surface is held fixed and the trajectory z is *derived* as `z = z_layer − TVT`.
That makes `dTVT = dz_layer − dz` true by construction, so the dip channel and the
dz_layer head stay consistent with the new path — measured violation is exactly 0.0 ft.
A mixed trajectory that would need GR outside the typewell's span is rejected rather than
clipped (clipping would fabricate flat sections and then supervise on them); about 14% of
draws are rejected that way.

Steps 1–2 are **not** implemented — consolidating the 773 typewells into master series
and re-skinning a trajectory into a different system at the same quantile of its drilling
band. The vertical material therefore stays the well's own typewell, and the diversity is
well below the author's. Treat the flag as a starting point, not a reproduction. For
reference, the 1st-place repo's port of the same idea (`rb_v2_synth`, which *did* include
master-series consolidation) came out 0.147 ft **worse** than its baseline at a 150-epoch
budget, so this is not obviously free.

## Environment notes for this machine

- **`requests` cannot reach huggingface.co here** (`CERTIFICATE_VERIFY_FAILED`), while
  `urllib` and `curl` reach the same host fine. `timm`'s `pretrained=True` goes through
  `huggingface_hub`, so it fails. `ensure_backbone_weights()` downloads the ImageNet
  weights once with `urllib` into `cache/` and hands them to timm via
  `pretrained_cfg_overlay`; the weights are identical and timm still adapts the
  3-channel stem to 9 channels. Override with `--backbone-weights`.
- **Batch size has a hard cliff on this 6 GB card.** Measured, with bf16 autocast and
  channels-last: batch 4 → 2.54 GiB and 60 ms/sample, batch 6 → 3.77 GiB and 59
  ms/sample, batch 8 → 5.01 GiB and **389** ms/sample, batch 10 → 6.23 GiB and 470
  ms/sample. Past ~4 GiB it spills into shared memory and throughput collapses 6×.
  Batch 6 with `--grad-accum 2` is the default; drop to 4 if the display driver needs
  the room.
- A 618-sample epoch takes ~37 s, a single-phase holdout pass ~6 s, so the default
  120-epoch run is about 80 minutes.
- `--num-workers` defaults to 0. Sample construction costs ~0.7 ms against a ~350 ms
  training step, so workers buy nothing, and on Windows each one would receive a pickled
  copy of the whole well set.

## Budget, and what it means

The default is 120 epochs × 618 samples ≈ 74k samples. The writeup's per-model budget is
120 epochs × (~620 real + 2048 synthetic) ≈ 320k samples, so this default trains on
about **4× less material** than the author's single model, and the final submission
averaged 15 such checkpoints across 5 folds and 3 seeds plus 8 MD phases and a second
16 ft-column family. Raise `--epoch-len` to close the volume gap (`--epoch-len 2472`
quadruples it at ~2.5 min/epoch).

`--eval-every` scores the EMA weights so the curve is readable during the run. The final
number scores the **last** epoch by default; `--score-best` scores the best-by-holdout
checkpoint instead, which is selection on the test set and should be reported as such.

## The sibling-lateral reference GR

Every model here works the same way underneath: slide the lateral's gamma ray against a
**reference profile** of what GR reads at each depth, until the patterns line up.  Every
model has used the same reference -- the supplied typewell -- and the typewell has two
problems.  It is a different, lower-resolution tool than the one in the lateral (the
competition host said so outright), and it may simply not describe the rock this well is
actually in.  That second failure is the catastrophic-well signature.

3rd place replaced it.  They called the choice of reference GR their single biggest lever
on the private leaderboard and weighted typewell against an alternative 0.2 / 0.8.  This is
that idea, rebuilt here.

### Master typewell systems

The 773 typewells are not 773 logs.  Interpolating them all onto a common 1 ft depth grid
and correlating every pair over the depths they share (one matmul, 0.2 s) then joining pairs
with >= 100 ft of overlap and correlation >= 0.9 gives:

    54 systems over 773 wells, members median 8, max 55, 12 singletons

Members of a system agree to 0.000 GR where they overlap -- they are one log, cropped
differently.  Wells sharing a system are drilling the same rock, so call them **siblings**.
153 of the 155 holdout wells have at least one sibling among the training wells; the median
has 21.

### Siblings as a reference

For a sibling that is a *training* well the true TVT is known at every row, so its own
**lateral** GR can be binned by depth into a profile -- measured by a horizontal tool, at
lateral resolution, in the right rock.  Accumulating counts across siblings so a depth
logged by many wells outweighs one logged by a single well gives a depth -> GR curve that
can be sampled anywhere the typewell is sampled.

How well does it describe a holdout well's actual lateral GR?  Median RMS over 153 wells:

    typewell                       10.78 API
    sibling profile                10.49          beats typewell on 58% of wells
    0.5 typewell + 0.5 sibling     10.18          beats typewell on 77% of wells
    0.2 typewell + 0.8 sibling     10.44

Two things that seemed obviously right and were not:

  * **Standardising each sibling's GR** before averaging (median/IQR, to remove tool
    calibration differences) makes it *much* worse -- 17.11 vs 10.49 -- because the
    typewell's robust statistics are taken over its full ~900 ft span while the lateral
    lives in a narrow TVT window, so the rescale maps onto the wrong range.
  * **Affine-calibrating the sibling profile on the known prefix**, the inference-legal
    analogue of `calibrate_typewell`, is a wash (10.80 vs 10.49).  The scales already agree;
    fitting two more parameters only adds variance.

Coarser bins are worse (2 ft -> 11.67, 5 ft -> 13.14), which is not smoothing noise away:
the lateral's TVT range is narrow, so a 5 ft bin leaves only a handful of samples across it.
1 ft is the setting.

### Where it plugs in

`anchor_sibling.apply_to_wells` overwrites each well's `tw_gr` in place after loading, so
`gr2tvt_data.build_compact` fills the typewell channel from the blend.  Nothing in `src/`
changes, the input stays 9 channels, checkpoints stay compatible, and flip augmentation
reverses the blended array alongside `tw_tvt` exactly as it did before.  The particle filter
gets the same treatment through `anchor_pf.sibling_reference`, which means tuning, dip
fitting and prediction all see one consistent reference without any of them knowing about it.

Leakage is asserted, not assumed:

    [ok] bank holds 618 laterals, all from the 618 training wells
    [ok] no well is its own sibling (checked all 773)
    [ok] uncovered depths untouched (482 of 2175 on d07aed8f)
    [ok] at weight 1.0 covered depths equal the sibling profile exactly

Depths no sibling logged keep the original typewell, so the blend never removes information.

### The blend weight, chosen on training wells

Swept over 120 training wells with the filter, holdout untouched:

    sibling-w   pooled ft   mean well   p50 well
      0.00        18.80       13.91       10.01
      0.25        18.17       13.05        8.70
      0.50        18.18       12.82        8.76
      0.75        15.72       11.49        7.96
      1.00        15.87       11.79        9.33

Pooled improves with weight; the *median* well is best around 0.25.  Siblings help the
wells that are badly wrong and cost a little on the wells that were already fine, which is
what a better-matched reference should do.  Pooled RMSE is the competition metric, so
**0.75** is the setting -- best pooled in this sweep and better median than 1.00.

The filter resamples stochastically and is not seeded, so repeat runs of the same config
differ; the w = 0 arm reproduces to 0.05 ft but the sibling arms move by over a foot between
repeats, because a better reference moves marginal wells onto a decision boundary where the
resampling can flip them.  Every number involving the filter below is therefore reported
with two repeats.

### Result on the particle filter

The filter is the right thing to test first: it needs no training, so a reference change is
measurable in minutes rather than an hour.  Best known configuration (3 profiles, temp 1.0,
slope walk x9, per-well dip correction fitted on 500 training wells), two repeats each:

    sibling-w   pooled ft            mean well        p50 well        p95 well
      0.00      12.733 / 12.652    10.269 / 10.298   7.770 / 7.931  25.119 / 24.068
      0.75      12.077 / 11.950     9.661 /  9.619   7.428 / 7.630  23.386 / 23.699

    -0.68 ft pooled, against a within-arm spread of 0.08-0.13 ft.

Every statistic moves the right way, and the gap is five times the repeat noise.  On the
plain untuned filter the same change is 17.18 -> 16.03.  **This is the first change in the
whole campaign that improved a prediction by changing the model's input rather than by
recalibrating its output.**

### It does not help the ensemble -- and the reason is the interesting part

A better filter should be a better blend partner.  It is not:

    partner        RMSE     corr with ConvNeXt   optimal weight   blended RMSE   break-even
    pf w=0        12.733          0.340              0.0216          4.9303        14.50 ft
    pf w=0.75     12.077          0.367              0.0201          4.9321        13.45 ft

Improving the filter by 0.66 ft also raised its correlation with the ConvNeXt from 0.340 to
0.367, which lifted the break-even threshold (`RMSE < sigma_c / rho`) from 14.50 ft to
13.45 ft by more than the accuracy gain closed.  The net blend is 0.002 ft *worse*.

This is the blend algebra being explicit about something that is easy to miss: **a partner
gets better at the ensemble only if its accuracy improves faster than its correlation.**
The sibling reference improves the filter by teaching it the same thing the CNN already
knows -- what the rock looks like -- so it buys accuracy and redundancy in one move.  For
the filter standalone that is a clear win.  For the filter as an ensemble member it is a
wash.

The corollary is that the sibling reference has to be tested *inside* the CNN to pay off on
the leaderboard, not bolted onto a second model beside it.

### Running it

    :: AnchorCNN at C geometry with the sibling reference (~1 h on this machine)
    python -u anchor_train.py --out runs/C_s1_sib --row 1.0 --n-move 5 --epoch-len 1150 --epochs 120 --eval-every 5 --tta 8 --seed 1 --sibling-w 0.75

    :: its paired control is the existing runs/C_s1 (6.0241 ft), same seed, same geometry
    python ../kaggle_1st_place/solution/seq_NN_robust_compare.py --base runs/C_s1 --treat runs/C_s1_sib --output-dir runs/C_s1_sib

    :: rescoring an existing checkpoint needs the same weight it was trained with
    python anchor_eval.py --models runs/C_s1_sib/model_last.pt --out runs/C_s1_sib_tta8 --row 1.0 --n-move 5 --tta 8 --sibling-w 0.75

`--sibling-w 0` is the default and reproduces every earlier run byte for byte; the sibling
code is not reached at all.
