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
    |-- reference_results/<ID>/ # six archived snapshots, 5.9 MB total, READ-ONLY
    |-- seq_NN_main_reproduce.py  # original: 15-fold CV repro -> reproduction_outputs/<ID>/
    |-- seq_NN_holdout_eval.py    # added: 80/20 holdout -> results/<ID>/
    `-- submit-reproduce.ipynb    # six-family ensemble; loads models.pkl from output dirs
```

## Do not re-explore `reference_results/`

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

| run | recipe | models | epochs | pooled RMSE | notes |
|---|---|---|---|---|---|
| `results/0801_V2/` | 0801_V2 | 1 | 60 | **5.1618 ft** | `best_epoch=60`, i.e. still improving when the cosine schedule ran out |

Reference points for that run: geo-prior-only baseline 11.5716 ft; archived 0801_V2 OOF
4.8045 ft (but that is 15 models x 300 epochs x geographic CV over all 773 wells, so not
a like-for-like comparison). Per-well RMSE mean=4.00, q50=2.93, q95=9.86,
mean_lt_q95=3.47 -- a heavy tail, so the pooled figure is dominated by roughly the worst
5% of wells, not by typical ones.

Re-running the same `--id` refuses rather than overwrites (`prepare_output_dir`); the
script forces `cfg.f = 0` rather than inheriting `archived_f`, because `0803_V2`'s
archived value is 1 and would otherwise clobber a finished run. Use `--output-dir` to
keep variants side by side.

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

Never edit `reference_results/`. Copy a snapshot, edit the copy, and point at it:

```bash
cp -r reference_results/0801_V2 experiments/my_v1
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

## Known caveat

`cfg.target_stats` are hardcoded constants fitted on the full original training set, so
they carry slight information about the holdout wells. Everything actually learned --
weights, geo prior, checkpoint choice -- is split-safe.
