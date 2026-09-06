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

## Which runs are starved, and the 150-epoch tier

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

The 60-epoch commands that produced the table above are in git history; the
`_ep150` suffix keeps both budgets side by side rather than replacing them.

## Known caveat

`cfg.target_stats` are hardcoded constants fitted on the full original training set, so
they carry slight information about the holdout wells. Everything actually learned --
weights, geo prior, checkpoint choice -- is split-safe.
