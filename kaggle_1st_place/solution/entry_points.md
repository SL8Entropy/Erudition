# Entry Points

Unless a command says otherwise, run the commands below with the current working
directory set to the repository root, the directory containing this file. The
defaults in `SETTINGS.json` are relative to that file and already point to the
bundled data, six bundled snapshots, internal cache directories, and
`reproduction_outputs/`. Running from a result subdirectory is not the
documented workflow.

The launcher is Bash, not Python. Set `PYTHON_BIN` when the active Python is
not the environment containing the pinned packages. Before a reproduction,
raise the open-file limit in the same shell:

```bash
cd /path/to/your/clone
ulimit -n 8192
```

The launcher uses `python3` by default. To use a custom environment, set the
interpreter once before running the commands below:

```bash
export PYTHON_BIN=/path/to/python
```

`reproduce_workflow.sh` first rebuilds
`data/train_png_typewell_map.csv` from the 773 bundled training horizontal
CSVs. It verifies the generated `well_id,horizontal_avg_X,horizontal_avg_Y`
projection against the existing file before writing the compact map. The
historical PNG-derived/group columns are intentionally not generated because
the sequence CV code never reads them.

## 1. Use the self-contained defaults

No path editing or archive download is required for this repository. The
following command should work as-is after dependencies are installed:

```bash
./reproduce_workflow.sh --train 0801_V2 --device cuda
```

All supported IDs are `0719_V1`, `0724_V1`, `0729_V3`, `0801_V1`, `0801_V2`,
and `0803_V2`. `SETTINGS.json` remains the single path authority if the
package is intentionally relocated or a different test directory is supplied;
relative values continue to resolve relative to `SETTINGS.json`.

## 2. Optional setup verification

This is a fast, debug-oriented check of all six source snapshots and recipe
configs. It imports each snapshot, discovers the bundled train/test wells,
serializes a temporary config, and compares every non-environment field with
the bundled `cfg.pkl`. It stops before model allocation or training. It can be
skipped when the goal is simply to run part 3.

```bash
./reproduce_workflow.sh --verify
```

Use `--keep-temp` only when inspecting the generated temporary configs:

```bash
./reproduce_workflow.sh --verify --keep-temp
```

## 3. Train an exact recipe

Train one family, including its 15 folds and individual test inference:

```bash
./reproduce_workflow.sh --train 0801_V2 --device cuda
```

The result is written to `reproduction_outputs/0801_V2/`. Replace the ID to
run another family. The destination must not already exist; choose a new
`paths.output_dir` in `SETTINGS.json` or pass `--force` when overwriting is
intentional.

To train all six sequentially in canonical order:

```bash
./reproduce_workflow.sh --train-all --device cuda
```

The launcher validates all six bundled snapshots and output destinations before
starting the first run, then waits for each complete family before starting the
next. Each run copies its exact `seq_NN*.py` files beside its newly written
`models.pkl`, config, log, and prediction artifacts. No archived `models.pkl`
is required because these commands retrain the families.

CPU smoke overrides, reduced folds, and shortened epochs are intentionally not
part of the documented reproduction command; they change the recipe and are
only appropriate for local debugging.

## 4. Optional direct entrypoint

This is an optional equivalent to part 3 for users who want to call Python
directly. It produces the same `reproduction_outputs/<ID>/` artifacts and
automatically selects `reference_results/<ID>/` from `SETTINGS.json`:

```bash
"${PYTHON_BIN:-python3}" ./generate_train_geo_map.py \
  --train-dir data/train \
  --output data/train_png_typewell_map.csv
"${PYTHON_BIN:-python3}" ./seq_NN_main_reproduce.py \
  --id 0801_V2 \
  --device cuda
```

The map command is shown explicitly so the direct path has the same first step
as the launcher. `--settings`, `--source-dir`, and `--output-dir` are available
for an intentional relocation or an isolated output, but are unnecessary for
the self-contained defaults.

## 5. 80/20 holdout evaluation (pooled RMSE)

`reference_results/*/cfg.pkl` and the six recipes above are scored by
repeated geographic cross-validation, and `data/test/` carries no `TVT`
column, so nothing in parts 1-4 produces a single held-out RMSE. Part 5 does.

`seq_NN_holdout_eval.py` sorts the 773 labelled wells under `data/train/` by
well id, trains on the **first 80%** (618 wells), and scores the **remaining
20%** (155 wells), which the model, the geo prior, and the checkpoint-selection
metric never see. It reports the **pooled RMSE**: one
`sqrt(mean((TVT - TVT_pred)**2))` over the concatenation of every masked
`TVT_input` suffix row of every holdout well (754,122 rows), not a mean of
per-well or per-fold RMSEs.

```bash
python seq_NN_holdout_eval.py --id 0801_V2 --device cuda
```

The recipe itself is untouched. Data loading, augmentation, model, loss,
checkpoint selection, and inference all still come from the archived snapshot;
only `seq_NN_train.make_cv_splits` is replaced, at runtime, with the single
fixed split. Because the split is contiguous over sorted well ids it is exactly
reproducible and needs no seed.

`results/<ID>/` receives:

- `holdout_split.csv`: every well id and its `train`/`holdout` side;
- `holdout_predictions.pqt`: per-row holdout predictions and diagnostics;
- `holdout_metrics.json`: `pooled_rmse` plus per-well quantiles and the
  geo-prior baseline;
- `models.pkl`, `cfg.pkl`, `seq_nn.log`, and the `seq_NN*.py` snapshot copy.

### Useful options

| Flag | Effect |
|---|---|
| `--train-frac F` | Split fraction; default `0.8`. |
| `--repeats N` | Train N independently seeded models on the same 80% and average their holdout predictions before scoring. Default 1. |
| `--offline-timm` | Load the pretrained ConvNeXt backbone from the local HuggingFace cache with no network access. Same weights and same recipe. |
| `--batch-size` / `--grad-accum-steps` | Fit the run into less VRAM; see below. |
| `--epochs` | Shorten the run. This also rescales the cosine LR schedule, so a short run is not a truncated full run. |
| `--smoke --unet-arch unet` | Tiny fully offline plumbing check. |

### Environment notes

`--offline-timm` exists because the recipes name the backbone as
`hf_hub:timm/convnext_small.in12k_ft_in1k_384`, which makes timm fetch
`config.json` from huggingface.co on every model construction even when the
weights are already in `~/.cache/huggingface`. The flag rewrites that to the
bare `convnext_small.in12k_ft_in1k_384` tag, which timm resolves through its own
registry to the identical repository, and sets `HF_HUB_OFFLINE=1`. Use it on any
machine that cannot verify the huggingface.co TLS certificate.

The archived `batch_size=16` needs roughly 11 GiB of VRAM. On a smaller card,
keep the effective batch with gradient accumulation:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python seq_NN_holdout_eval.py --id 0801_V2 --device cuda --offline-timm --batch-size 4 --val-batch-size 4 --grad-accum-steps 4
```

Note that this makes BatchNorm statistics come from 4 samples rather than 16, a
small deviation from the archived recipe.

At full scale one epoch over the 618 training wells takes about 128 s on an
RTX 3050 6 GB laptop GPU with the settings above, so the archived
`epochs=300` (`min_epochs=170`, `early_stopping_rounds=50`) is a 6-11 hour run.

### Known caveat

The recipes' target normalisation constants (`cfg.target_stats`) are hardcoded
values fitted on the full original training set, so they carry a small amount of
information about the holdout wells. Everything the run actually learns --
weights, geo prior, checkpoint choice -- is split-safe.

## Final ensemble inference

After the six individual families are trained, use the existing notebook for
saved-model inference and the final ensemble:

<https://www.kaggle.com/code/w5833946/submit-reproduce>

Point its six result inputs at the corresponding
`reproduction_outputs/<ID>/` directories. Keep each `models.pkl` paired with
the `seq_NN*.py` snapshot copied into the same directory because pickle class
imports depend on that snapshot.
