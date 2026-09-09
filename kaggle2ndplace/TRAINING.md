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
