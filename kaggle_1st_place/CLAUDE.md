# kaggle_1st_place

ROGII wellbore-geology Kaggle 1st-place solution, repackaged for local reproduction.

**Read this file before exploring. Do not re-derive the facts below.**

## Layout

```
kaggle_1st_place/
`-- solution/                  # the whole package; run every command from here
    |-- SETTINGS.json          # single path authority, all paths relative to itself
    |-- data/train/            # 773 labelled wells (x3 files each), 1.3 GB
    |-- data/test/             # only 3 wells, UNLABELLED -- not scoreable
    |-- Original_Comp_Models/<ID>/  # six archived snapshots, READ-ONLY (was reference_results/)
    |-- experiments/bilzard/   # the one edited fork; four recipes live in its cfg registry
    |-- seq_NN_main_reproduce.py  # original: 15-fold CV repro -> reproduction_outputs/<ID>/
    |-- seq_NN_holdout_eval.py    # added: 80/20 holdout -> results/<ID>/
    `-- submit-reproduce.ipynb    # six-family ensemble; loads models.pkl from output dirs
```

## Do not re-explore `Original_Comp_Models/`

Six snapshots (`0719_V1 0724_V1 0729_V3 0801_V1 0801_V2 0803_V2`), each a **full private
copy of the same ~22,000-line pipeline**. Reading or diffing them is expensive and has
already been done. The findings:

- `MMDD_Vn` = the date the author snapshotted their tree, plus a variant number.
- They differ by **code evolution**, not config. Consecutive `.py` churn:
  599 / 1781 / 933 / **0** / 70 lines. `0801_V1` and `0801_V2` are byte-identical source;
  they differ in exactly one input channel (`z_diff` vs `geo_tvt_diff`), which moved the
  archived OOF from 5.1668 to 4.8045 ft.
- Identical across all six: `batch_size=16`, `fold_count=5`, `cv_repeats=3`, `seed=7`,
  architecture, optimizer, LR. Only `epochs` (300/200) and `min_epochs` vary.
- `0801_V1` onward add `seq_NN_trf_backbones.py` / `seq_NN_trf_unet.py`.
- Archived OOF RMSE: 0719_V1 5.0910, 0724_V1 4.8586, 0729_V3 5.5360, 0801_V1 5.1668,
  **0801_V2 4.8045 (best)**, 0803_V2 5.0055. Final submission = all six x 15 = 90 models.
- No `models.pkl` ships with the repo, by design. The only `.pkl`s are `cfg.pkl` configs.

Snapshot module roles are tabulated in `solution/README.md`; read that table instead of
opening the files.

## Data schema

Train wells have `TVT` (ground truth) **and** `TVT_input`; test wells have only
`TVT_input`. In both, the rows to predict are exactly where `TVT_input` is NaN
(the suffix). `data/test/` has no `TVT`, so it cannot be scored locally -- that is
why the holdout script exists.

## The 80/20 holdout evaluation

`solution/seq_NN_holdout_eval.py` sorts the 773 wells by id, trains on the first
618 (80%), scores the last 155 (20%), and reports **pooled RMSE**: one
`sqrt(mean((TVT - TVT_pred)**2))` over all 754,122 held-out suffix rows.

It edits **no snapshot**. It rebinds `seq_NN_train.make_cv_splits` at runtime to one
fixed split; model, data, loss, and inference stay as archived. The split is contiguous
over sorted ids, so it needs no seed and is exactly reproducible.

Outputs go to `solution/results/<ID>/`, kept separate from `reproduction_outputs/<ID>/`
because both write `models.pkl`/`cfg.pkl`/`seq_nn.log` with incompatible meanings
(1 model on 80% vs 15 models on 100%), and the notebook builds its ensemble from those
files. `prepare_output_dir` refuses an existing directory unless `--f 1`.

## Results so far

All at 1 model, `--epochs 60`, batch 4x4 accum 4, scored on the same 155 wells /
754,122 rows. `k*` and the bootstrap come from `seq_NN_robust_compare.py` against
`results/0801_V2`.

| run | pooled RMSE | vs base | best_epoch | k\* | bootstrap p50 [p05,p95] | verdict |
|---|---:|---:|---:|---:|---|---|
| `results/0801_V2/` (baseline) | **5.1618** | -- | 60/60 | -- | -- | -- |
| `results/rb_v3_tta/` | **5.1119** | -0.050 | -- (rescore) | 1 | -0.046 [-0.245, +0.125], 66% improving | REJECT |
| `results/rb_v4_all/` | 5.3711 | +0.209 | 55/60 | 0 | +0.203 [-0.195, +0.644], 20% | REJECT |
| `results/rb_v2_synth/` | 5.5241 | +0.362 | 55/60 | 0 | +0.355 [-0.066, +0.825], 9% | REJECT |
| `results/rb_v1_neighbor/` | 6.2556 | +1.094 | 60/60 | 0 | +1.072 [+0.508, +1.720], 0% | REJECT |

Nothing beat the baseline. Two things the numbers do say:

- **MD-phase TTA is 2/2 positive and free.** On the baseline weights -0.050 ft
  (66% of bootstrap draws improving); on the `rb_v2_synth` weights **-0.153 ft**
  (k\* = 2, 86% of draws improving). It never touches training, so it is worth
  carrying even though 155 wells cannot resolve a 0.05-0.15 ft effect to the
  k\* >= 10 bar. Absence of proof here is a power problem, not evidence of absence.
- **The 60-epoch budget confounds everything else.** Every validation curve was
  still descending at 60 (baseline 5.375@50 -> 5.235@55 -> 5.205@60), and the
  archived recipe trains 300. Stronger augmentation (`rb_v2_synth`) and a wider
  input (`rb_v1_neighbor`, which fell 0.14 ft over the last 10 epochs and is
  nowhere near converged) are exactly the two changes that pay off late, so this
  budget is systematically biased against both. Extending is not possible after
  the fact -- the cosine schedule is tied to `epochs`, so a longer run is a
  rerun.

`rb_v2_synth` is not merely tail-unlucky: dropping the 8 worst-SSE baseline wells
it is still 4.370 vs the baseline's 4.182. `rb_v4_all` has the best **median**
well (2.888 vs 2.929) but a fatter tail (q95 11.05 vs 9.86) -- more variance, not
a uniform regression.

Error concentration on this holdout, which is why `k*` matters: the worst 1 / 3 /
10 / 52 wells carry 8% / 20% / 43% / 86% of the total squared error. With 155
wells the effective sample size is closer to 10-20.

Reference points for that run: geo-prior-only baseline 11.5716 ft; archived 0801_V2 OOF
4.8045 ft (but that is 15 models x 300 epochs x geographic CV over all 773 wells, so not
a like-for-like comparison). Per-well RMSE mean=4.00, q50=2.93, q95=9.86,
mean_lt_q95=3.47 -- a heavy tail, so the pooled figure is dominated by roughly the worst
5% of wells, not by typical ones.

Re-running the same `--id` refuses rather than overwrites (`prepare_output_dir`); the
script forces `cfg.f = 0` rather than inheriting `archived_f`, because `0803_V2`'s
archived value is 1 and would otherwise clobber a finished run. Use `--output-dir` to
keep variants side by side.

## The 150-epoch round

Same protocol, `--epochs 150`, 1 model each. `rb_v3_tta` / `rb_v4_all` are the two
trained models rescored with `--tta 8`; they cost 5 min each, not a training run.

| run | pooled RMSE | vs base@150 | best_epoch | k\* | bootstrap improving | verdict |
|---|---:|---:|---:|---:|---:|---|
| `0801_V2_ep150` | 4.9785 | -- | 110/150 | -- | -- | -- |
| `rb_v3_tta_ep150` | **4.9374** | -0.041 | -- | 1 | 65% | REJECT |
| `rb_v4_all_ep150` | 5.1092 | +0.131 | -- | 0 | 26% | REJECT |
| `rb_v2_synth_ep150` | 5.1250 | +0.147 | 115/150 | 0 | 19% | REJECT |

**Settled by this round:**

- *The epoch confound was real.* Baseline 5.1618 -> 4.9785 (-0.183);
  `rb_v2_synth` 5.5241 -> 5.1250 (-0.399). The gap closed from +0.362 to +0.147,
  i.e. 60% of it was undertraining, exactly as predicted.
- *The baseline is near its ceiling; `rb_v2_synth` is not.* Over epochs 50->150
  the baseline improved -0.201 and oscillates in a 5.02-5.76 band (std 0.161);
  `rb_v2_synth` improved -0.684 and was still descending at 150.
- *`rb_v2_synth`'s failure mode inverted.* At 60 epochs it was worse everywhere
  (4.370 vs 4.182 even after dropping the 8 worst wells). At 150 its removal
  curve goes **negative from k=10 through k~105** -- it is better than the
  baseline on the bulk of the holdout, and loses only on a few catastrophic
  wells. Median well 2.796 vs the baseline's 2.929.

**Not settled, and the reason is the holdout, not the ideas.** The remaining
+0.147 is smaller than the baseline's own checkpoint-to-checkpoint scatter
(0.161), and `best_epoch=110` is the luckiest point on a noisy plateau while
`rb_v2_synth`'s 115 sits on a monotone descent -- the comparison is biased toward
the baseline by roughly the size of the gap. The wells that decide it change
identity every run: at 150 the baseline blows up on `d1457cc5` (11.66) which
`rb_v2_synth` fixes (4.76), while `rb_v2_synth` destroys `f0188a48` (0.83 ->
8.67), a well the baseline nearly nails. **The pooled metric on 155 wells is
decided by about ten coin flips**, so no single-seed run can resolve a 0.1 ft
effect. Add seeds or folds, not epochs.

**MD-phase TTA is now 4/4 positive** across two budgets and two models:
-0.050 (base@60), -0.041 (base@150), -0.153 (synth@60), -0.016 (synth@150).
The magnitude shrinks as a model converges, which is what you would expect if it
averages out column-grid aliasing that a better-trained model is already less
sensitive to. Every one of the four still fails the k\* >= 10 bar; four
consistent signs plus Bilzard's independent -0.21 on 773 wells is the whole case
for it, and it is free.

Best result so far: **4.9374 ft** (`rb_v3_tta_ep150`), against 5.1618 at the
start of this work. Archived `0801_V2` OOF is 4.8045 on a different protocol
(15 models x 300 epochs x geographic CV over all 773 wells).

## The architecture round (60 epochs)

Against `results/0801_V2` (5.1618), same budget, same archived recipe.

| run | pooled RMSE | vs base | best_epoch | s/epoch | k\* | bootstrap improving | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `arch_v4_lkconv` | 5.1314 | -0.030 | 60/60 | 170 | 21 | 64% | **ACCEPT -- but see below** |
| `arch_v2_axial` | 5.1382 | -0.024 | 60/60 | 110 | 6 | 59% | REJECT |
| `arch_v3_raft` | 5.2395 | +0.078 | **40**/60 | 121 | 0 | 26% | REJECT |
| `arch_v1_unimodal` | 5.3233 | +0.162 | 60/60 | 117 | 0 | 21% | REJECT |

### `arch_v4_lkconv`'s ACCEPT is spurious: the branch never trained

`AnisotropicLargeKernelDW` shipped with **both** factors of `gate *
large_conv(x)` zero-initialised. That is a gradient deadlock:
`d(loss)/d(gate)` is proportional to `large_conv(x)` = 0, and
`d(loss)/d(large_conv.weight)` is proportional to `gate` = 0. Neither can ever
move.

Confirmed on the trained checkpoint: **all 36 gates and all 36 weight tensors
are still exactly 0.0** after 60 epochs. The model is functionally the plain
baseline -- it just paid 170 s/epoch instead of 111 to convolve zeros.

Fixed by leaving the weight at its normal initialisation and gating with zero
alone, which keeps identity-at-init *and* gives the gate a live gradient
(verified both). The run is worth repeating; the result above is not about
large kernels at all.

### The accident is the most useful result: a seed-level null

Because that model is functionally identical to the baseline, the comparison is
a **same-model / different-random-draw null** -- something this project never
had. It gives, from pure training nondeterminism:

> delta **-0.030 ft**, **k\* = 21**, bootstrap **64% improving**, ACCEPT.

So a k\* of 21 against a bar of 10, and a 64% bootstrap, are reachable with
*no model change whatsoever*. Consequences:

- **The k\* >= 10 bar is far too lenient at this effect size.** The acceptance
  test can and did hand ACCEPT to noise. `k*` measures whether a difference is
  *spread across wells*, not whether it is *real*; a diffuse noise draw scores
  well on it.
- Every delta in this project smaller than about 0.05 ft should be treated as
  unmeasured, whatever `k*` says. That includes the MD-phase TTA numbers
  (-0.050, -0.041, -0.016), whose case now rests on sign consistency across
  four measurements and Bilzard's independent result, not on this holdout.
- `arch_v2_axial`'s -0.024 is *smaller than the null draw of -0.030*. It cannot
  be distinguished from zero.

### What the other three say

- **`arch_v2_axial` did train, but barely.** `gamma_attn` mean 0.011 (max
  0.026), `gamma_mlp` 0.003 -- the attention contributes a few percent of the
  residual stream. Still descending at epoch 60 (5.184@55 -> 5.177@60), so
  undertrained like everything else at this budget. Unproven, not disproven.
- **`arch_v3_raft` is the only variant that peaked early**: `best_epoch=40`,
  then the curve rose (5.282@40 -> 5.400@50 -> 5.285@60). That pattern is
  instability in the refinement loop rather than undertraining, and it is the
  known failure mode of iterative decoders. Its `delta_head` did learn
  (mean |w| 5.4e-4), so it was doing something, just not something that helped.
- **`arch_v1_unimodal` is the cheapest test of the mode-blending hypothesis, and
  it did not pay** (+0.162). The confidence head trained normally. This is the
  clearest evidence available against the premise that soft-argmin mode blending
  drives the catastrophic wells -- though it replaces the alignment loss rather
  than adding to it, so it changes the training signal substantially, and 60
  epochs may be too short for a learned-width target to calibrate.

Net: **nothing in this round beat the baseline in a way that survives scrutiny**,
and the round's real contribution is the noise calibration above.

## Which runs are starved (the 60-epoch diagnosis that led to the 150 tier)

`early_stopping_rounds=50` never fires in these runs because it is gated on
`epoch >= min_epochs` and `min_epochs=170` exceeds any budget used so far. So
every run trained for exactly its full cosine schedule, and every one of them
stopped while still descending.

Placing each run's final validation RMSE on the **baseline's own trajectory**
says how starved it is (val_rmse_raw, log_freq=5):

| run | final @60 | change over epochs 45->60 | sits at baseline epoch | deficit |
|---|---:|---:|---:|---:|
| `0801_V2` | 5.2048 | -0.281 | 60 | 1.0x |
| `rb_v2_synth` | 5.5777 | -0.331 | 35 | 1.7x |
| `rb_v4_all` | 5.6168 | -0.224 | 29 | 2.1x |
| `rb_v1_neighbor` | 6.3039 | -0.313 | **19** | **3.1x** |

(The deficit is a reading of where a run sits on the baseline's curve, not a
promise that N times the epochs closes it -- a regularized model is slower *and*
may have a lower asymptote.)

**Only two runs need training.** `rb_v3_tta` is baseline weights + inference-time
phase averaging and `rb_v4_all` is `rb_v2_synth` weights + the same, so both are
`seq_NN_rescore.py --tta 8` against the trained models: 5 minutes each instead of
5 hours. The 60-epoch round trained `rb_v4_all` separately; that was avoidable.
(Its `best_epoch=55` and near-identical curve to `rb_v2_synth` make the pair a
free read on run-to-run nondeterminism: about 0.02-0.04 ft.)

Output directories are epoch-tagged (`results/<name>_ep150/`) because
`prepare_output_dir` refuses to overwrite the finished 60-epoch runs, and the two
budgets must stay side by side.

`seq_NN_robust_compare.py` writes `robust_compare__vs_<base>.json` and
`robust_compare_curve__vs_<base>.csv`, so a candidate can be compared against
several baselines into its own directory without clobbering earlier reports.

## Open leads, in the order worth trying

1. **Rerun baseline vs `rb_v2_synth` at `--epochs 150`, then rescore both with
   `--tta 8`.** This is the decisive experiment, not a new idea. Better synthetic
   data is regularization, and regularization is invisible at a budget where the
   model is still underfitting. Two trainings (~5 h each) plus two rescores
   (~5 min each) produce all four models at the new budget. Commands are under
   "Running the four recipes".
2. **Fix the calibration mismatch in re-skinning.** Real samples get a typewell
   blended toward the seen prefix (`typewell_calibration_with_seen=True`); a
   re-skinned sample gets the raw master crop. Both sides of a re-skinned sample
   are self-consistent, but its *statistics* differ from a real sample's. Re-apply
   `GR_calibration` to the crop after `reskin_typewell` and re-measure.
3. **Ablate `rb_v2_synth`'s three sub-changes separately** (mixup / re-skin /
   `real_block` residuals). They were bundled; the 0.36 ft loss is not attributed.
   Also try a gentler dose: `reskin.apply_prob` 0.15-0.25 instead of 0.5.
4. **Raise statistical power before chasing 0.1 ft.** The bootstrap CI on a delta
   is about +-0.4 ft here. Either 3 seeds per config, or 5-fold CV over all 773
   wells. Anything smaller cannot separate these candidates.
5. **Attack the tail.** `fb03ae90`, `f6d009f4`, `f2d4c8c9`, `e7818f7a`,
   `d7eb0be8` dominate every comparison. Check what they share -- typewell
   system, GR gaps, suffix length, distance to the nearest support well. The
   remaining 0.35 ft between this holdout and the archived 4.80 OOF lives there.
6. **`rb_v1_neighbor` is unproven, not disproven** -- at epoch 60 it sat where the
   baseline sat at epoch 19. Matching its budget to the baseline's would not
   settle it; give it *more* compute than the control instead (300 epochs against
   the baseline's 150). If a candidate loses while holding a 2x compute
   advantage, that is decisive, and it costs 10 h rather than the 20 h a
   symmetric 300-vs-300 pair would.

## Environment gotchas (both already hit and solved)

1. **huggingface.co TLS fails on this machine** (`CERTIFICATE_VERIFY_FAILED`, not a
   sandbox artifact). The recipes name the backbone `hf_hub:timm/convnext_small.in12k_ft_in1k_384`,
   which fetches `config.json` on every model build. Pass `--offline-timm`: it rewrites
   that to the bare `convnext_small.in12k_ft_in1k_384` tag (timm's registry resolves it to
   the same repo, weights already cached) and sets `HF_HUB_OFFLINE=1`. Same weights.
2. **GPU is an RTX 3050 6 GB laptop.** The archived `batch_size=16` needs ~11 GiB and
   OOMs. Use `--batch-size 4 --val-batch-size 4 --grad-accum-steps 4` to keep the
   effective batch (BatchNorm stats then come from 4, a small deviation).

~110-130 s/epoch at full scale with those settings, so archived `epochs=300` is 6-11 h.

3. **Training is GPU-bound; do not bother tuning the data pipeline.** Measured on
   this machine: the loader delivers a batch in **0.056 s** at the current
   `num_workers=4, prefetch_factor=1`, while training consumes one every
   **0.714 s** (110 s / 154 batches) -- it is idle ~92% of the time.
   `workers=8, prefetch=4` gets it to 0.044 s, which buys nothing. Consistent
   with that, all four 60-epoch runs took 109-117 s/epoch despite
   `rb_v1_neighbor` carrying 50% more input channels and `rb_v2_synth` running
   the full re-skin simulation every sample.
   The GPU-side knobs are **already on**: `amp_dtype=bfloat16`,
   `channels_last_2d=True`, `cudnn.benchmark=True` and TF32 (both follow from
   `deterministic=False`). There is no free win sitting there, and running from
   Jupyter changes nothing -- same interpreter, same GPU.
4. **VRAM is nearly full at `--batch-size 4`**: 3.84 GiB peak allocated / 3.98
   reserved of 6.00 GiB, for a (4, 16, 345, 400) input. Batch 6 would sit on the
   edge with the display driver, batch 8 would OOM. `--grad-accum-steps 4`
   already emulates the archived effective batch of 16.
5. **Host RAM is the tight resource, not VRAM.** 16 GiB total with only ~4 GiB
   free at rest. Windows DataLoader workers use `spawn`, so each one receives a
   pickled copy of the whole dataset (all 618 wells plus the initialised
   `Simulator`), which is why raising `num_workers` is not free here. Two
   attempts to micro-benchmark GPU step time were destroyed by paging (they
   reported ~37 s/batch against a real 0.71 s/batch and ended with 0.2 GiB free);
   if you profile on this box, watch free RAM or the numbers are meaningless.

```bash
cd solution && python seq_NN_holdout_eval.py --id 0801_V2 --device cuda --offline-timm --batch-size 4 --val-batch-size 4 --grad-accum-steps 4
```

The user runs commands in **cmd.exe**, not bash: `VAR=value cmd` fails there. Use
`set VAR=value` on its own line, or omit `PYTORCH_CUDA_ALLOC_CONF` (it is only an
allocator hint).

## Making a new attempt (fork the pipeline, edit, retrain)

Never edit `Original_Comp_Models/`. Copy a snapshot, edit the copy, and point at it:

```bash
cp -r Original_Comp_Models/0801_V2 experiments/my_v1
rm experiments/my_v1/cfg.pkl experiments/my_v1/seq_nn.log   # stale provenance
# edit experiments/my_v1/*.py, adding a SEQ_TRAIN_CFGS entry named my_v1
python seq_NN_holdout_eval.py --id 0801_V2 --source-dir experiments/my_v1   --cfg-name my_v1 --device cuda --offline-timm   --batch-size 4 --val-batch-size 4 --grad-accum-steps 4
```

`--source-dir` wins over the ID's archived directory in `resolve_source_dir`, `--cfg-name`
selects a registry entry the archived recipe map does not know about, and results land in
`results/<cfg-name>/` with the edited source copied beside `models.pkl`. `--id` is still
required: it supplies the recipe metadata, not the code.

Where to change what (module roles are tabulated in `solution/README.md`):

| Change | File |
|---|---|
| Hyperparameters, channels, aug/sim probabilities, loss weights | `seq_NN_cfg.py` (`CFG`, `SEQ_TRAIN_CFGS`) |
| Model architecture, heads, prediction outputs | `seq_NN_models.py` |
| ConvNeXt encoder/decoder blocks | `seq_NN_pretrained_unet.py` |
| Loss functions, optimizer, scheduler, training loop | `seq_NN_train.py` |
| Input features, windowing, simulation, augmentation | `seq_NN_dataset.py` |
| Static channel construction | `seq_NN_data_prep.py` |
| XY geo prior | `seq_NN_geo_prior.py` |

For config-only changes prefer a new `SEQ_TRAIN_CFGS` entry over mutating `CFG` defaults,
so the original recipe stays runnable from the same fork for A/B comparison. Note that
`seq_NN_holdout_eval.py` forwards only a subset of the snapshot's CLI overrides
(epochs, batch, workers, repeats); anything else goes in the config entry.

## The Bilzard fork: `experiments/bilzard/`

One edited copy of `0801_V2` carrying three ideas ported from the 2nd-place
solution, plus four `SEQ_TRAIN_CFGS` entries that switch them on. At its
archived settings the fork is **bit-identical** to `Original_Comp_Models/0801_V2`
(verified by comparing built items: max abs difference 0.0), so every recipe
below differs from the baseline only by the knobs named in its row.

| cfg-name | idea | what changes |
|---|---|---|
| `rb_v1_neighbor` | 1 | all nine neighbouring-well (geo-prior) channels on, 16 -> 24 input channels. **Not a proposed improvement** -- it is the candidate the acceptance test exists to judge. |
| `rb_v2_synth` | 2 | trajectory mixup, master-series re-skinning, real residual pasting |
| `rb_v3_tta` | 3 | inference averaged over 8 MD column-grid phases; **no training change** |
| `rb_v4_all` | 2+3 | `rb_v2_synth` plus the 8-phase averaging |

### Idea 1 -- is the improvement real? (`seq_NN_robust_compare.py`)

Leave-largest-contribution-out. Per well, `g_w = SSE_treat(w) - SSE_base(w)`;
drop wells in descending `|g_w|` and rescore both runs on what is left; `k*` is
the first removal count at which the gain disappears. The default bar is
52/773 of the holdout well count (the public-LB-sized bar Bilzard used) = **10**
wells here. Also reports the per-well win rate and a well-level bootstrap.

It reads `holdout_predictions.pqt`, so it works on any pair of runs from
`seq_NN_holdout_eval.py` or `seq_NN_rescore.py` scored on the same split.

Sanity-checked against a null: baseline vs a rerun of the same weights gives
delta +0.0002 ft, k* = 0, REJECT.

### Idea 2 -- physically grounded synthesis (`seq_NN_master_typewell.py`)

Three changes inside the simulator, all switched on by `sim_cfg`:

1. **Trajectory mixup** (`z_shift.trend_mixup_prob`): `TVT_mix = k*TVT1 +
   (1-k)*TVT2` over two real suffix paths, on top of the archived diff-block
   bootstrap.
2. **Master-series re-skinning** (`sim_cfg['reskin']`): the 773 typewells are
   consolidated into master systems by one dense correlation pass -- **60
   systems** here, against the 54 Bilzard reports, and members agree to 0.000 GR
   on their overlap, confirming they are crops of one series. A simulated well
   is then re-placed into another system at the *same quantile of that system's
   drilling band* it occupied in its own, and its GR is regenerated from a crop
   of the target master. `z` is untouched, so the whole move lands on
   `z_layer = TVT + z`; every absolute-surface sidecar (`geo_s_rel_prior_abs`,
   `S_avg`) is shifted with it so the geo prior stays aligned.
3. **Real residual pasting**: `z_shift.noise_mode` becomes a 0.7/0.3 mixture of
   the fitted `TVT_bias_v2` model and the existing `real_block` residual bank.

Note Ruby's `z_shift` already holds `TVT + Z` fixed, i.e. it was already
`z_layer`-consistent; re-skinning is the one place that deliberately moves it.

The bank is built in-process from the loaded wells (~1.3 s for 773) -- no cache
file, and it only runs when `reskin.apply_prob > 0`.

### Idea 3 -- MD-phase TTA (`cfg.md_phase_shift` / `cfg.md_phase_tta`)

The pipeline averages `downsample=32` raw rows per column and pins the cut lines
to the prediction-start row, so everything below 32 ft is aliased away.
`md_phase_shift` moves the window boundary back by N rows, which re-phases every
cut line; `md_phase_tta=8` averages inference over 8 evenly spread phases
(0, 4, ..., 28 rows). Bin-space predictions live on different grids per phase,
so the averaging happens after expanding back to original rows. Training and
checkpoint selection stay single-phase, exactly as Bilzard did it.

Because it changes nothing about training, it can be applied to weights that
already exist -- that is what `seq_NN_rescore.py` is for.

### `seq_NN_rescore.py`

Rebuilds the same 80/20 split, geo prior and loader, loads an existing
`models.pkl`, predicts (optionally with `--tta N`) and writes the same
`holdout_predictions.pqt` / `holdout_metrics.json`. Single-phase inference over
the 155 holdout wells is ~35 s, so 8 phases is ~5 min against ~2 h to retrain.
`--source-dir` must point at a pipeline whose model classes match the pickle;
use `experiments/bilzard` to get MD-phase support.

Verified: rescoring `results/0801_V2` with no TTA reproduces 5.1619 vs the
training run's 5.1618 (CUDA nondeterminism).

### Running the four recipes

Run everything from `solution/`, in `cmd.exe`. Shared flags for this 6 GB laptop
GPU: `--device cuda --offline-timm --batch-size 4 --val-batch-size 4
--grad-accum-steps 4`. Output directories carry the epoch budget because
`prepare_output_dir` refuses to overwrite a finished run.

**Tier 1 -- the 150-epoch pair (~10 h of GPU, produces all four models).**
Train the control and the candidate:

```bash
python seq_NN_holdout_eval.py --id 0801_V2 --source-dir experiments/bilzard --epochs 150 --output-dir results/0801_V2_ep150 --device cuda --offline-timm --batch-size 4 --val-batch-size 4 --grad-accum-steps 4
```
```bash
python seq_NN_holdout_eval.py --id 0801_V2 --source-dir experiments/bilzard --cfg-name rb_v2_synth --epochs 150 --output-dir results/rb_v2_synth_ep150 --device cuda --offline-timm --batch-size 4 --val-batch-size 4 --grad-accum-steps 4
```

Then derive the two TTA models from those weights, ~5 min each, no training:

```bash
python seq_NN_rescore.py --id 0801_V2 --models-dir results/0801_V2_ep150 --source-dir experiments/bilzard --tta 8 --output-dir results/rb_v3_tta_ep150 --device cuda --offline-timm --val-batch-size 4 --num-workers 4
```
```bash
python seq_NN_rescore.py --id 0801_V2 --models-dir results/rb_v2_synth_ep150 --source-dir experiments/bilzard --cfg-name rb_v2_synth --tta 8 --output-dir results/rb_v4_all_ep150 --device cuda --offline-timm --val-batch-size 4 --num-workers 4
```

Score everything against the new control, plus the TTA-only contrast:

```bash
python seq_NN_robust_compare.py --base results/0801_V2_ep150 --treat results/rb_v2_synth_ep150 --output-dir results/rb_v2_synth_ep150
python seq_NN_robust_compare.py --base results/0801_V2_ep150 --treat results/rb_v3_tta_ep150 --output-dir results/rb_v3_tta_ep150
python seq_NN_robust_compare.py --base results/0801_V2_ep150 --treat results/rb_v4_all_ep150 --output-dir results/rb_v4_all_ep150
python seq_NN_robust_compare.py --base results/rb_v2_synth_ep150 --treat results/rb_v4_all_ep150 --output-dir results/rb_v4_all_ep150
```

**Tier 2 -- settle `rb_v1_neighbor` (~10 h), only if the GPU is otherwise idle.**
It gets 300 epochs against the control's 150 on purpose: at 60 it sat where the
baseline sat at 19, so a matched budget cannot answer the question, while losing
with a 2x compute advantage would.

```bash
python seq_NN_holdout_eval.py --id 0801_V2 --source-dir experiments/bilzard --cfg-name rb_v1_neighbor --epochs 300 --output-dir results/rb_v1_neighbor_ep300 --device cuda --offline-timm --batch-size 4 --val-batch-size 4 --grad-accum-steps 4
```
```bash
python seq_NN_robust_compare.py --base results/0801_V2_ep150 --treat results/rb_v1_neighbor_ep300 --output-dir results/rb_v1_neighbor_ep300
```

### Running the four architecture recipes

At 60 epochs against `results/0801_V2` (5.1618), which was trained at the same
budget from the same archived recipe. ~2 h each; `arch_v2_axial` and
`arch_v3_raft` are somewhat slower for the added compute.

```bash
python seq_NN_holdout_eval.py --id 0801_V2 --source-dir experiments/arch --cfg-name arch_v1_unimodal --epochs 60 --output-dir results/arch_v1_unimodal --device cuda --offline-timm --batch-size 4 --val-batch-size 4 --grad-accum-steps 4
python seq_NN_holdout_eval.py --id 0801_V2 --source-dir experiments/arch --cfg-name arch_v2_axial    --epochs 60 --output-dir results/arch_v2_axial    --device cuda --offline-timm --batch-size 4 --val-batch-size 4 --grad-accum-steps 4
python seq_NN_holdout_eval.py --id 0801_V2 --source-dir experiments/arch --cfg-name arch_v3_raft     --epochs 60 --output-dir results/arch_v3_raft     --device cuda --offline-timm --batch-size 4 --val-batch-size 4 --grad-accum-steps 4
python seq_NN_holdout_eval.py --id 0801_V2 --source-dir experiments/arch --cfg-name arch_v4_lkconv   --epochs 60 --output-dir results/arch_v4_lkconv   --device cuda --offline-timm --batch-size 4 --val-batch-size 4 --grad-accum-steps 4
```

Then judge each against the 60-epoch baseline:

```bash
python seq_NN_robust_compare.py --base results/0801_V2 --treat results/arch_v1_unimodal --output-dir results/arch_v1_unimodal
```

The 60-epoch commands that produced the table above are in git history; the
`_ep150` suffix keeps both budgets side by side rather than replacing them.

## The architecture fork: `experiments/arch/`

A second fork of `0801_V2`, independent of `experiments/bilzard/`, holding four
architecture experiments. New module `seq_NN_arch_blocks.py`; the rest is
surgical edits to `seq_NN_models.py`, `seq_NN_train.py` and `seq_NN_cfg.py`.

The framing that produced these: **Ruby's pipeline is a stereo-matching network
without the name.** `gr_abs_diff` is a cost volume over disparities (levels),
the U-Net is cost aggregation, and softmax-expectation down the level axis is
soft-argmin disparity regression -- GC-Net's design. That means the relevant
literature is stereo matching, not ImageNet backbones, and it comes with a
diagnosed failure: soft-argmin is only sound on a *unimodal* posterior; on a
bimodal one it "blends the modes and may produce a solution far from all the
modes". Repeating rock layers make the posterior bimodal by construction, and
this is a candidate mechanism for the catastrophic wells that dominate pooled
RMSE.

| cfg-name | attacks | change |
|---|---|---|
| `arch_v1_unimodal` | mode blending | confidence head + AcfNet-style learned-width unimodal target, **replacing** the fixed-width alignment term (same weight, same role) |
| `arch_v2_axial` | short sight along MD | MD-axial attention after ConvNeXt stages 1-3 |
| `arch_v3_raft` | one-shot decoding | 8 rounds of RAFT-style residual refinement over the cost volume |
| `arch_v4_lkconv` | short sight along MD | parallel long-along-MD depthwise kernel in all 36 ConvNeXt blocks |

**v2 and v4 are competing fixes for the same weakness**, not complements. Run
one, measure, then consider the other; running both at once attributes nothing.

### Everything starts numerically inert

Each added path is gated so a fresh model reproduces the archived behaviour
exactly. This matters because the branches graft onto a pretrained ConvNeXt: a
randomly initialised parallel branch would scramble the pretrained features on
step one and the run would measure that damage, not the idea.

Verified directly, not assumed:

- `AnisotropicLargeKernelDW`: 36 depthwise convs wrapped; output bit-identical
  to the unwrapped conv at init (zero weight *and* zero per-channel gate).
- `AxialMDStack`: identity at init. The positional conv sits outside the gated
  residual, so it needs zero-init too -- it was the one place the claim was
  false until fixed.
- `IterativeLevelRefiner`: 8 rounds, all identical at init, and iterate 0
  reproduces the one-shot expectation to 2.4e-07. Adds 0.55M params to 54.1M.

### Geometry the blocks rely on

Feature maps are `(B, C, H=MD, W=level)`. The ConvNeXt patchify stem is skipped
(`to_backbone` is a 1x1 conv into stage 0), so MD runs
**345 -> 173 -> 87 -> 44 -> 22** and level runs **400 -> 100 -> 50 -> 25 -> 13**.
Hence: axial attention on stages 1-3 (MD 87/44/22, a few MB of attention);
large kernels tapered `(31, 31, 15, 7)` because a 31-long kernel on a 22-column
map is an expensive global average; and full-resolution 2-D attention is out of
the question (345x400 tokens).

### Traps hit while building this

- `SeqUNet2DModel.model_named_parameters()` enumerates an explicit module list.
  A new submodule that is not added there **is silently never optimised**. The
  confidence head and the refiner are registered in both `model_modules()` and
  `model_named_parameters()`.
- `make_model` passes `cfg.model_cfg` straight into the constructor, so every
  new knob needs a matching keyword argument or the run dies at model build.
- `arch_v3_raft` sets `regression` weight to 0 and moves it to `iters`: the
  per-iteration loss already contains the final prediction at full weight, so
  keeping both would double-count it.

## Known caveat

`cfg.target_stats` are hardcoded constants fitted on the full original training set, so
they carry slight information about the holdout wells. Weights and the geo prior are
split-safe. **Checkpoint choice is not** (corrected 2026-09-19): the holdout wells are the
validation set, and `seq_NN_train.py` keeps the lowest-validation checkpoint (`if val_rmse <
best_score: best_state = ...`; `model.load_state_dict(best_state)`). Every reported holdout
number is therefore scored on a checkpoint picked using the test wells -- about 0.14-0.15 ft
better than the same run's epoch>=100 average, in every run checked. Comparisons between runs
stay like-for-like but are optimistic and noisier. `seq_NN_honest_curve.py <run dirs>` prints
the unpicked alternatives (last checkpoint, late average) from each run's log.

## PF channels and backbone size (added from the 2nd-place workspace)

- The pipeline already implements 22 PF input channels (`PF_HEATMAP_CHANNELS` in
  `seq_NN_data_prep.py`); `uses_pf_heatmap_channels(cfg)` just checks whether any is in
  `unet_static_channels`. 0801_V2 uses none. The author's 0729_V3 and 0803_V2 did, and their
  archived OOF was worse (5.536, 5.006 vs 4.8045) -- but those snapshots also differ by code
  churn and, for 0803_V2, two geo channels, so it is not a clean ablation.
- **Cache cost, measured:** 10.7 s/well at 4 workers, ~2.3 h for 773 wells, 0.04 GB,
  numba warmup 8 s. A probe script must wrap its body in `if __name__ == "__main__":`
  or the Windows `spawn` Pool hangs. Its summary on the first 4 wells: TVT RMSE 5.17
  filtered / **3.80 FFBSi** -- 4 wells only, and the settings were tuned on all 773.
- New `experiments/bilzard` entries: `pf_v1` (0801_V2 + `pf_particle_density_prob`,
  `pf_prob_ffbsi`, the author's 0803_V2 additions and nothing else), `cnx_tiny`
  (`convnext_tiny.in12k_ft_in1k_384`, the author's exact pretraining recipe), `cnx_base`
  (`convnext_base.fb_in22k_ft_in1k_384`, nearest available). `unet_arch` is only a label the
  builder validates ('convnext_small'); the network is `unet_timm_model_name`. tiny/base
  weights are not cached and must be downloaded before `--offline-timm` will work. **tiny is now cached** (commit bac32564, 114.4 MB, checksum verified against the HF LFS sha256, fetched with urllib since `requests` fails TLS here); `cnx_tiny` built offline loads it exactly (backbone tensor diff 0.0, stage 3 = 9 blocks).
- Size in this U-Net (400x345, fp16): nano 17.8M/54 GFLOP/21 ms, tiny 31.9M/82/22 ms,
  small 53.6M/127/30 ms, base 94.8M/214/40 ms (batch 1, network only).

## Backbone replacements and the bottleneck-MHSA proposal (checked 2026-09-19)

**The author already explored replacements and kept ConvNeXt-small every time.** The code
has four model types (`unet`, `trf_unet`, `two_stage_unet`, `bidir_convgru`) and the
transformer path lists 10 backbones in `seq_NN_trf_backbones.py` (Swin tiny/small/base,
CAFormer-S18, MaxViT, CoAtNet 1/2, PVTv2-b2, Twins-SVT). All six submitted snapshots use
`model: unet` with `convnext_small`.

**Drop-in test** (built through `make_model`, real 16x400x345 forward, fp16, batch 1,
no code changes; `unet_arch` kept as the label, `unet_timm_model_name` swapped):

    works:  convnext_small 53.6M/127 GFLOP/33 ms, convnext_tiny 31.9M/82/23,
            convnextv2_tiny 32.0M/82/33, convnextv2_nano 17.9M/54/27,
            inception_next_tiny 28.1M/79/25, convformer_s18 26.3M/69/38,
            caformer_s18 25.8M/83/40, fastvit_sa12 12.3M/40/20
    fails:  mambaout_* (NHWC layout breaks the BN norm swap), tf_efficientnetv2_s,
            regnety_016, rdnet_tiny, hiera_tiny_224 (no ConvNeXt-style stages),
            edgenext_small

"Works" means builds and runs, not trained or validated.  Real Mamba/VMamba/Samba need
`mamba_ssm` CUDA kernels with no official Windows build (not installed here); Samba is a
language-model architecture with no pretrained vision weights.

**`arch_v5_bottleneck_mhsa` guide, verified against `experiments/arch`:**
- Correct: wiring at `seq_NN_models.py:679`; `backbone_stages` is an `nn.ModuleList`
  (`seq_NN_pretrained_unet.py:543`); `model_named_parameters` includes `("unet", self.unet)`
  (line 779); `model_cfg` is the *same dict object* as `unet_cfg` (`set_model_name`), so the
  guide's in-place edits need no `refresh()`; stage 3 is 22x13 = 286 tokens.
- Layer-wise LR decay would give a block inside `backbone_stages[3]` stage 3's reduced rate,
  but `layer_wise_lr` is **False** in 0801_V2 (lr 2e-4 everywhere), so it is not an issue.
- **Flaw in Step 4:** same-seed builds are not comparable. Adding a block before the heads
  shifts their random init -- measured with `arch_v2_axial`: 4 of 110 shared tensors differ,
  all in `unet_gr_rmse_head`. Copy the plain model's `state_dict` into the new one
  (`strict=False`) before comparing outputs.
- Expected value is low: the same zero-gated design in `arch_v2_axial` trained to
  gamma_attn mean 0.011 in 60 epochs and scored -0.024 (inside the -0.030 null draw); stage-3
  ConvNeXt blocks already cover most of a 22x13 map; the model is data-limited.  If run, use
  150 epochs against `0801_V2_ep150` and report the trained gamma values.

## ConvNeXt-tiny result and FastViT (2026-09-19)

`results/cnx_tiny_ep150`, 150 epochs, 1 model, same recipe as `0801_V2_ep150`:

    reported (test-picked, smoothed)   tiny 4.9044   small 4.9785
    picked, raw                        tiny 4.9457   small 5.0212   both at epoch 110
    last checkpoint, raw               tiny 5.1435   small 5.0907
    average epochs >= 100, raw         tiny 5.0926   small 5.1628   (sd 0.066 vs 0.118)

The two unpicked statistics disagree in sign, both by < 0.1 ft: **tiny is as good as small
within what this holdout can measure**, at 64% of the compute (82 vs 127 GFLOP) and ~27%
faster. "Tiny is better" is not supportable.

`cnx_fastvit` added to `experiments/bilzard` (`fastvit_sa12.apple_dist_in1k`, ImageNet-1k
distilled; the ConvNeXts use stronger ImageNet-12k pretraining). Weights cached (46.5 MB,
checksum verified); built offline, 12.3M params, all 94 backbone weight tensors identical to
the download, stages [2, 2, 6, 2]. 40 GFLOP / 20 ms in this U-Net.

### More backbone candidates screened (2026-09-20)

    fasternet_s            works: 35.9M / 98.2 GFLOP / 59.3 ms  -- fewer FLOPs than
                           convnext_small yet 1.8x SLOWER here
    mobileone_s3/s4        no: "4 stages vs 5 channel entries" (stem counted as a stage);
                           a small adapter in _stage_channels_from_timm_model would fix it
    efficientnet_b0,       no: do not expose ConvNeXt-style stages
    tf_efficientnetv2_b0,
    mobilenetv4_conv_medium
    swin_tiny              no in the conv path -- but the author's `trf_unet` path already
                           supports swin tiny/small/base; no trf recipe is registered in
                           either fork (all 10 recipes are model_name='unet')
    repvit_m1_1            no: does not accept drop_path_rate
    FasterViT (NVIDIA)     not in timm, `fastervit` package not installed
    CMT                    not in timm at all

**FLOPs do not predict wall clock in this U-Net.** The patchify stem is skipped and the
input is 400x345, so backbones run far above their design resolution: fasternet_s 98 GFLOP
= 59 ms against convnext_small 127 GFLOP = 33 ms, and fastvit_sa12 40 GFLOP = 20 ms against
convnext_tiny 82 GFLOP = 23 ms (half the FLOPs, 13% faster). Screen candidates by measured
ms, never by GFLOP.

### Where the U-Net's FLOPs actually are (measured, 400x345, ConvNeXt-small)

    linear / channel mixing (block MLPs)   89.3 GFLOP   70.2%
    other spatial convs (decoder, stem)    32.9 GFLOP   25.9%
    1x1 convs                               3.2 GFLOP    2.5%
    7x7 depthwise spatial kernels           1.9 GFLOP    1.5%
    TOTAL                                 127.2 GFLOP

ConvNeXt is already depthwise-separable, so its spatial mixing is **1.5% of compute**.
Proposals to replace 2-D kernels with row+column 1-D convolutions therefore cannot cut 70%:
the ceiling is ~1.5% plus about two thirds of the 25.9% decoder/stem convs, i.e. ~20%, and
separable convs are often slower per FLOP (two passes, memory-bound). Cutting the 70% needs
fewer channels or blocks -- which is what convnext_tiny (-35% FLOPs, no measurable loss) and
fastvit_sa12 (-69%) do, while keeping pretrained weights.

The 2nd-place workspace already tested the row/column idea end to end
(`kaggle2ndplace/anchor_separable.py`: 1-D level encoder + 1-D MD encoder, outer-difference
matching volume, separable 2-D pyramid at the same feature resolutions): **11.40 ft against a
6.02-6.57 control**, at 4.82 vs 7.9 GFLOP. Confound stated there: no ImageNet initialisation.

### Third backbone screen: MobileOne, HRNet, EfficientNet-B3/B4 (2026-09-20)

Encoder only, `features_only`, in_chans=16, 400x345, fp16, batch 1:

    convnext_small        49.5M   46.7 GFLOP   13.8 ms
    convnext_tiny         27.8M   24.1 GFLOP    7.2 ms
    fastvit_sa12          10.4M    9.0 GFLOP    9.3 ms
    mobileone_s1           3.6M    5.2 GFLOP    7.1 ms  ->  reparameterised  5.1 GFLOP  1.9 ms
    mobileone_s2           5.8M    7.9 GFLOP    7.5 ms  ->  reparameterised  7.8 GFLOP  2.6 ms
    mobileone_s4          12.9M   17.3 GFLOP   10.5 ms
    efficientnet_b3       10.1M    5.6 GFLOP    9.5 ms
    efficientnet_b4       16.7M    8.7 GFLOP   11.6 ms
    tf_efficientnetv2_s   19.9M   16.1 GFLOP   14.0 ms
    hrnet_w18, hrnet_w18_small_v2   FAIL at 400x345 (fusion size mismatch; needs padding)

**Reparameterisation does not change FLOPs (5.2 -> 5.1) but cuts wall clock 3.7x** (7.1 ->
1.9 ms): it removes multi-branch memory traffic, which is what actually costs time here.

**EfficientNet-B3/B4 are slower than convnext_tiny** (9.5 / 11.6 vs 7.2 ms) at a quarter of
the FLOPs -- the claim that they run faster on this GPU is measurably wrong, as is the claim
that ConvNeXt's 7x7 depthwise kernels are the bandwidth problem (they are 1.5% of compute).

**Backbone share of the real model** (timed in place with CUDA events, whole `PretrainedUNet2d`):

    convnext_small  29.1 ms total | stages 19.8 ms (68%) | stem+decoder+heads 9.4 ms
    convnext_tiny   20.4 ms total | stages 11.2 ms (55%) | stem+decoder+heads 9.2 ms

In-model stages cost ~1.55x their standalone figure, because the pipeline skips patchify and
runs the backbone at higher resolution. A reparameterised MobileOne encoder would plausibly
land near 3 ms in-model, i.e. ~12 ms total against tiny's 20.4 -- a real ~40% saving, but it
needs a stage/stem adapter, and the nearest tested relative (fastvit_sa12, same Apple reparam
family, 3x the parameters) lost 0.16-0.31 ft.

Input-channel pruning is not worth it: 1x1 convs are 2.5% of compute, and the author's own
0801_V1 -> V2 differ by exactly one input channel for 5.1668 -> 4.8045 OOF.

### `cnx_mobileone` built (2026-09-20)

`experiments/bilzard`, MobileOne-S1. Two things had to be handled or the comparison would be rigged:

1. **Channel list includes the stem.** `feature_info` is `[64, 96, 192, 512, 1280]` for 4
   stages, and unlike ConvNeXt (stem 96 -> stage0 in 96 -> out 96) MobileOne's stage 0 changes
   width (in 64 -> out 96). `seq_NN_pretrained_unet.py` now separates *what feeds stage 0*
   (`backbone_in_ch`) from *what each stage emits* (`stage_channels`), and accepts a list of
   `len(stages) + 1`. ConvNeXt/FastViT behaviour is unchanged (the extra branch only triggers
   on the 5-entry case).
2. **MobileOne's stage 0 downsamples; ConvNeXt's does not.** Left alone every feature map would
   be one level coarser, which the 2:1 resolution rule says costs real accuracy. The recipe sets
   `unet_stem_stride=(1, 2)` instead of `(2, 4)`, handing that factor of two to stage 0.
   Verified: stage maps land on ConvNeXt's geometry, (173,100) (87,50) (44,25) (22,13) against
   tiny's (173,100) (86,50) (43,25) (21,12).

Measured in the real U-Net, 16ch 345x400, fp16, batch 1:

    cnx_tiny                       31.9M   21.9 ms
    cnx_mobileone (branched)       10.2M   18.6 ms
    cnx_mobileone (reparameterised) 10.2M  14.0 ms   -36% against tiny

**Reparameterisation is exact** -- relative output difference 5.2e-07 over the whole stage
stack. (Absolute differences are useless here: an untrained MobileOne's activations reach
4.9e+08, so a 2.6e+02 absolute gap is 5e-07 relative. A first check that compared absolute
values wrongly called it broken.) Accuracy is therefore identical branched or reparameterised;
reparameterise only for deployment speed.

Weights `mobileone_s1.apple_in1k` (19.6 MB) are **not yet downloaded**.

### "Big encoder + pruned decoder" is already the shipped design (2026-09-20)

`unet_emb_dim=32`, `unet_mults=(1,2,4,4)`, `unet_stacks=1` against an encoder that ends at 768
channels, and `to_backbone` is already a 1x1 projection. Timed inside the U-Net (tiny, 345x400,
fp16, batch 1; total 22.2 ms): backbone_stages 11.2, up_blocks ~5, full_up_block 2.2, out 1.5,
stem 1.8, stem_down + to_backbone 0.16. The decoder parts hold ~0.05M parameters -- their cost
is the resolution they run at, not their width, so pruning width further saves ~nothing while
cutting resolution is what the 2:1 rule says costs accuracy.

Encoder options screened for that idea:

    convnext_tiny     27.8M  24.1 GFLOP   7.2 ms
    convnextv2_tiny   27.9M  24.1 GFLOP  13.6 ms   (GRN makes V2 ~2x slower at equal FLOPs)
    convnextv2_base   87.7M  82.4 GFLOP  27.0 ms   -> ~36 ms whole model vs tiny's 21.9
    DINOv2 (vit_*_patch14_dinov2)  plain ViT: no `.stages`, patch 14, built for 518x518.
        features_only returns single-scale tokens, so it needs a ViTDet-style feature pyramid
        before this decoder can use it -- substantial work, and it is photo-pretrained while
        the input here is a 16-channel synthetic cost image.

### MobileOne-S1 trained: decisively worse, and the capacity curve has a knee (2026-09-20)

Run done outside this repo, so `results/cnx_mobileone_ep150/` is not present here; copy it in when
convenient so `seq_NN_honest_curve.py` can read its unpicked checkpoints.

    HOLDOUT POOLED RMSE 6.5128 ft   (pre-SG-smoothing 6.5823)
    per-well mean 4.6052, q50 3.2559, q95 11.3391

Against `cnx_tiny_ep150` (4.9044, same test-picked convention) that is **+1.61 ft** -- four times
the single-run noise floor, so decisive without needing the acceptance test. The weights were
genuinely loaded: `_create_timm_model_with_retries` re-raises rather than falling back to random
init, so a completed run implies the pretrained checkpoint was found.

Accuracy against backbone size, all 150 epochs, same recipe:

    49.5M  convnext_small   4.98
    27.8M  convnext_tiny    4.90     <- the knee
    10.4M  fastvit_sa12     5.21
     3.6M  mobileone_s1     6.51

**The model is data-limited above ~28M and capacity-limited below ~10M.** Halving from small to
tiny is free; going further is not. The cost ladder is closed: ConvNeXt-tiny is the floor.

This also retires the idea of distilling ConvNeXt-tiny into a cheaper backbone: for FastViT the
prize is 3 ms (20.8 vs 23.0), and for MobileOne distillation would have to recover 1.6 ft.
MobileCLIP-pretrained FastViT variants do not help either -- `fastvit_mci0` is 26.0 ms, slower
than tiny itself.

### Distillation built (2026-09-21): `cnx_fastvit_kd`

Added to `experiments/bilzard`, teacher = a finished run's `models.pkl` (which stores the model
object itself, so the teacher brings its own architecture):

- `seq_NN_train.load_distill_teacher(cfg, device, log)` -- reads `cfg.distill_teacher_dir`,
  takes the first fold, moves to device, `.eval()`, `requires_grad_(False)`. Returns None unless
  `distill_weight > 0`.
- `seq_NN_train.distill_alignment_loss(...)` -- KL(teacher || student) over the depth axis of
  `extra["alignment_logits"]` at `distill_temperature`, scaled by T^2, masked through the
  existing `_weighted_masked_mean`. Matching the full distribution is the point: it passes on
  which *other* depths the teacher found plausible, which is exactly the information a hard
  label discards when rock layers repeat.
- Wired into the training step right before the non-finite check; logged as `distill=`.
- Recipe `cnx_fastvit_kd`: fastvit_sa12 student, `distill_teacher_dir="results/cnx_tiny_ep150"`,
  weight 1.0, temperature 2.0 (both untuned).

Verified before running: teacher loads (31.9M, frozen); both models emit `alignment_logits` of
(1, 345, 400) so the grids match; KL(teacher||teacher) = 1.0e-08; KL(student||teacher) = 1.68;
gradients reach the student (total |grad| 2.2e3) and **no** teacher parameter receives one.

Why this is worth running when `X_distill` in the 2nd-place workspace was not: there the teacher
(5.993) was no better than the student's own baseline (5.942), so nothing could transfer. Here
the gap is measured -- tiny 5.14/5.09 against fastvit 5.30/5.32 on unpicked checkpoints. Note the
prize is small either way: FastViT is only 13% faster than tiny (20.8 vs 23.0 ms).

### DINOv2 as a ConvNeXt replacement: no (2026-09-21)

Measured, encoder only, 16 channels (timm adapts the patch embed), fp16, batch 1. 345 is not a
multiple of the 14-pixel patch, so the input had to be padded to 350x406:

    convnext_tiny            27.8M   24.1 GFLOP   8.6 ms   maps (86,100) (43,50) (21,25) (10,12)
    vit_small_dinov2         22.8M   42.3 GFLOP   8.5 ms   maps (25,29) x3  -- single scale
    vit_base_dinov2          88.0M  146.2 GFLOP  16.3 ms   maps (25,29) x3  -- single scale

Every DINOv2 map is 25x29, about 24x fewer positions than ConvNeXt's finest in-model map
(173,100). The 2:1 resolution rule measured in the 2nd-place workspace (one extra halving cost
0.9 ft, two cost 1.7 ft) says fine resolution is exactly what this task needs, and patch-14
tokenisation discards it in the first layer. It would also need a multi-scale adapter and
padding, and it is photo-pretrained while the input is a synthetic 16-channel cost image.

Recent stereo work agrees on *how* such models are used: DEFOM-Stereo (CVPR 2025) keeps a CNN
for matching features and adds Depth Anything V2 (DINOv2-based) as a second source for
initialisation; MonSter (CVPR 2025) fuses a monocular prior with matching. Add alongside, never
replace. The analogue already exists here: the particle-filter channels are an independent prior
fed into the matcher (`pf_v1` would use the author's 7.35 ft filter).

Pointers from the same survey: IGEV++ (TPAMI 2025, multi-range cost volumes: coarse for large
shifts, fine for detail -- maps onto large drift vs fine wiggles); Fast-FoundationStereo
(arXiv 2512.11130: distillation + architecture search + pruning, >10x faster at near-equal
accuracy, using 1.4M pseudo-labelled pairs); a CMT-enhanced Hiformer for well-log stratigraphic
correlation (CNN+transformer with a geological loss term); WLFM (arXiv 2509.18152), a well-log
foundation model pretrained on 1,200 wells for porosity/lithology -- no correlation task and no
released weights, so not usable here.

### A ConvNeXt ensemble is the strongest, most robust result so far (2026-09-21)

Plain averages of existing holdout predictions (no weights fitted, so nothing tuned on the test
wells). Row-level error correlation between the variants is 0.83-0.87 -- enough diversity to
average:

    tiny alone                                  4.9044
    small + tiny                                4.7782   -0.126   k*=67  bootstrap 91.1%  ACCEPT
    small + tiny + synth                        4.7652   -0.139   k*=65  bootstrap 88.9%  ACCEPT
    small+TTA + tiny + synth+TTA + fastvit      4.7467   -0.158

k*=67 is three times the accidental noise draw's k*=21, and every other change in this project
sat at k*=0-2. Better than the ConvNeXt + AnchorCNN blend (4.852). Caveat: every member's
checkpoint was picked on the test wells (see "Known caveat"), so absolute values are optimistic by
~0.15 ft; the gain *relative to tiny* compares like with like.

Two ways to use it: deploy small + tiny directly (~52 ms per well vs tiny's 22, i.e. 2.4x the cost
for a robust 0.13 ft -- a far better trade than the AnchorCNN blend's 5-6x for 0.085), or distil
the ensemble into tiny. The existing `load_distill_teacher` takes one teacher; an ensemble teacher
needs it to load several dirs and average their softmax probabilities -- all ConvNeXt-family runs
share the (345, 400) alignment grid.

### Correction: "pseudo-label synthetic wells" is already happening (2026-09-21)

`seq_NN_dataset.py:6301` replaces each training sample with a `z_shift` simulation with
probability `sim_cfg['z_shift']['apply_prob']` (0.85), and a simulated well carries its exact
generated path as the label. So synthetic wells never needed pseudo-labels, and the distillation
term in `seq_NN_train.py` already runs the teacher on every training batch -- 85% of which are
synthetic. The Fast-FoundationStereo lesson (1.4M pseudo-labelled pairs) applies to large pools of
*unlabelled real* data, which this dataset does not have (`data/test` holds 3 wells). What remains
open is only the choice of teacher and student.

ConvNeXt-Base as teacher is not supported by the evidence: small (49.5M) already scores the same
as tiny (27.8M), so above ~28M the model is limited by the 618 real wells, and a bigger teacher
trained on the same wells knows no more. It would also need a different pretraining recipe
(fb_in22k_ft_in1k_384; no in12k_ft_in1k_384 exists for base), a ~350 MB download, likely batch 2
on 6 GB, and roughly 8-9 h for 150 epochs. The measured stronger teacher is the small + tiny
ensemble (-0.126 ft vs tiny, k*=67).

### Ensemble-teacher distillation built; single-teacher KD interim result (2026-09-21)

**Interim `cnx_fastvit_kd_ep150` (epoch 130/150, still running), unpicked statistics:**

    run                    picked         last          late avg (ep>=100)
    cnx_tiny_ep150         4.9457 @110    5.1435 @150   5.0926   (teacher)
    cnx_fastvit_ep150      5.2516 @100    5.3027 @150   5.3189   (no teacher)
    cnx_fastvit_kd_ep150   4.9592 @105    5.1552 @130   5.1073   (taught by tiny)

The student reached its teacher: late average -0.21 ft against plain FastViT, 0.015 ft from tiny.
One seed each, and the A/B noise for single checkpoints is sd 0.39, so this is promising, not proven.
The loss is active: train_loss is 0.399 vs 0.210 for plain FastViT at epoch 120. The epoch log
prints only fixed keys, so `distill` never appears there. Epoch time ~65 s, about 2 h 40 min per 150 epochs.

**Multi-teacher support:** `load_distill_teacher` now reads `cfg.distill_teacher_dirs` (a list),
and the single `distill_teacher_dir` still works. It returns a list of frozen teachers.
`distill_alignment_loss` accepts a list of teacher extras and averages their tempered softmaxes,
so the target is the ensemble's mixture distribution. The train step runs every teacher under no_grad.

**Recipes:** `cnx_tiny_kd_ens` (student tiny) and `cnx_fastvit_kd_ens` (student fastvit_sa12).
Teacher = [`results/0801_V2_ep150`, `results/cnx_tiny_ep150`], weight 1, T 2.

**Verified:**
- 53.6M + 31.9M teachers load, frozen; 0.35 GB of weights.
- self-KL with two identical teachers is 2e-9.
- A dict and a one-element list give identical loss.
- The mixture KL equals a hand-computed reference (7.336539).
- A student equal to the mixture gives KL 1e-8.
- The gradient reaches the student only.

Check script: scratchpad `kd_ens_check.py`.

**DINOv2 as teacher: not worth it.** It is not a trained teacher yet: it would need an adapter,
padding to 350x406, and a full training run on the same 618 wells. Its single-scale 25x29 maps
violate the 2:1 resolution rule, so it is expected to score *worse* than tiny as a student, and
a teacher weaker than the student transfers nothing (the 2nd-place `X_distill` lesson). The
ensemble teacher already provides a measured 0.126 ft margin at zero training cost.

### Ensemble tiny + FastViT (2026-09-21)

50/50 average of `TVT_pred`, holdout, same test-picked convention as every other number here:

    tiny alone                 4.9044
    tiny + fastvit             4.8487   -0.056 vs tiny, k*=81, bootstrap better in 72.5%  ACCEPT (weak)
    small + fastvit            4.9049   no gain
    small + tiny (earlier)     4.7782   -0.126 vs tiny, k*=67, bootstrap 91%

- Error correlation: tiny/fastvit 0.838, small/fastvit 0.854, small/tiny 0.870.
- The best FastViT weight on the holdout is 0.3 (4.8189), but that weight is chosen in-sample.
  Only 50/50 is an honest figure.
- Cost is ~44 ms/well (23.0 + 20.8) against ~52 ms for small+tiny: 16% cheaper for under half the gain.
- A distilled FastViT is taught to copy tiny, so its errors should correlate *more* with tiny.
  Expect it to be a worse ensemble partner than plain FastViT. Check with `cnx_fastvit_kd_ep150`
  once that run writes its predictions.

### Wider backbone drop-in screen (2026-09-21)

Screened with `dropin3.py` in the scratchpad: the real U-Net, random weights, 345x400, fp16.
**Timings were taken while `cnx_fastvit_kd` was training on the same GPU**, so absolute ms are
inflated; only the ordering is meaningful. Params are for the whole model.

    fits unchanged      params  GFLOP   ms
    convnext_nano       17.8M   54.7    49.7   <- fastest; same in12k_ft_in1k recipe as tiny
    fastvit_sa12        12.3M   39.9    53.0
    convnext_tiny       31.9M   82.6    60.3
    inception_next_tiny 28.1M   79.7    64.9   in1k weights only
    convnextv2_nano     17.9M   54.7    70.7   GRN layer makes it slower than v1
    convnextv2_tiny     32.0M   82.6    84.3
    convformer_s18      26.3M   68.9   100.7
    caformer_s18        25.8M   82.6   101.9

- Do not fit: mambaout, repvit, efficientvit, edgenext, tiny_vit, hgnetv2 (constructor kwargs,
  channel-layout or stem mismatches). Each would need adapter code.
- Recipe `cnx_nano` added (`hf_hub:timm/convnext_nano.in12k_ft_in1k`). Its weights are NOT downloaded yet.

### InceptionNeXt student, ConvNeXt-small teacher (2026-09-21)

- Recipe `inx_tiny_kd_small`: `inception_next_tiny.sail_in1k` student, teacher `results/0801_V2_ep150`, weight 1, T 2.
- Recipe `inx_tiny`: the matched no-teacher control.
- `_kd_ens` now takes `teachers=`.
- Weights (in1k only) are not downloaded yet.
- Caveat: the small teacher is not better than tiny on unpicked checkpoints (small last 5.091 /
  late 5.163; tiny 5.144 / 5.093). The teacher therefore sets a ceiling of roughly tiny level; it
  can only help if InceptionNeXt's in1k start leaves it below that.

### ConvNeXt-nano student, small+tiny ensemble teacher (2026-09-21)

- Recipe `cnx_nano_kd_ens`: `convnext_nano.in12k_ft_in1k` student, teachers [`0801_V2_ep150`, `cnx_tiny_ep150`], weight 1, T 2.
- Control: `cnx_nano` (no teacher).
- Weights are not downloaded yet.

### cnx_fastvit_kd final + inference-speed options (2026-09-21)

**`cnx_fastvit_kd_ep150` final:**
- Pooled 4.9292 (test-picked convention), against fastvit 5.2132 and tiny 4.9044.
- Unpicked: last 5.119 (tiny 5.144), late avg 5.110 (tiny 5.093).
- Distillation closed ~90% of the gap. One seed.

**Weights downloaded** (urllib, sha256 verified; huggingface_hub fails SSL on this machine, but
urllib uses the Windows cert store): `convnext_nano.in12k_ft_in1k` 62.4 MB,
`inception_next_tiny.sail_in1k` 112.3 MB. Both load with HF_HUB_OFFLINE=1.

**Inference profile.** Trained U-Nets, 1x16x345x400, idle RTX 3050, scratchpad
`infer_profile.py` / `infer_profile2.py`. ms per well, U-Net only:

    variant                                   tiny   fastvit  small
    fp32                                      46.5   32.1     68.5
    bf16 autocast + channels_last (CURRENT)   25-26  22-23    37.5
    fp16 autocast                             31.0   22.9     36.2
    bf16 CUDA graph                           25.4   20.8
    pure fp16 weights, norms kept fp32        22.0   19.0
    pure fp16 + CUDA graph                    21.5   18.3

- AMP is already on (cfg `amp_dtype=bfloat16`, `channels_last_2d=True`).
- Pure fp16 with fp32 norms is ~16% faster and *closer* to fp32 than current bf16: relative
  output drift 7e-4 vs 3.7e-3. Plain `.half()` crashes because a custom norm casts its input to
  fp32; the norms must stay fp32.
- CUDA graphs save only 0.5-2 ms, so the model is compute-bound, not launch-bound.
- Decoder is ~8 ms of tiny's 25: up blocks 0.8/1.2/1.9, full_up_block 2.6, out head 1.8.
  The costliest parts are the full-resolution ones the 2:1 rule says matter.
- TensorRT, torch_tensorrt and triton are not installed (so no torch.compile on Windows).
  onnxruntime is CPU-only.
- Whole-pipeline holdout prediction ~12 s / 155 wells, about 75-80 ms per well (log timestamps,
  rough). The U-Net is about a third of per-well cost; the rest is CPU-side prep and
  post-processing.

### Averages with FastViT, including the distilled one (2026-09-21) -- corrects a prediction

50/50 `TVT_pred` averages, holdout (test-picked convention), robust_compare against the stronger member:

    pair                     pooled   err corr   check
    small + fastvit          4.9049   0.854      vs small: k*=2, 76%     REJECT
    small + fastvit_kd       4.8153   0.890      vs small: k*=54, 98.4%  ACCEPT
    tiny + fastvit           4.8487   0.838      vs tiny:  k*=81, 72.5%  ACCEPT (weak)
    tiny + fastvit_kd        4.7656   0.879      vs tiny:  k*=31, 90.3%  ACCEPT
    small + tiny             4.7782   0.870      vs tiny:  k*=67, 91%    ACCEPT

**The prediction "a distilled student is a worse averaging partner for its teacher" was wrong.**
Correlation did rise (0.838 -> 0.879), but the student's own accuracy rose more (5.21 -> 4.93).
tiny + its own FastViT student (4.766) is the best pair found, at ~44 ms/well against small+tiny's ~52.

- New recipe `cnx_fastvit_kd_small`: FastViT taught by `0801_V2_ep150` alone. It is verified to
  resolve and load its teacher.
