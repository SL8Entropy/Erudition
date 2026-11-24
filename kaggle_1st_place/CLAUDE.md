# kaggle_1st_place: the research pipeline

The ROGII Kaggle 1st-place solution, repackaged for local training and scoring. Project-wide rules,
the paper model and machine notes are in `../CLAUDE.md`. **Paper ablation runs do not happen here:** they use the
copy in `../ablation_study/pipeline/`.

## Layout (run every command from `solution/`)

```
solution/
|-- SETTINGS.json             path authority, relative to itself
|-- data/                     773 labelled wells + 3 unlabelled test wells (1.4 GB, not in git)
|-- Original_Comp_Models/     six archived author snapshots: READ-ONLY, never edit or re-explore
|-- experiments/bilzard/      the edited fork: every recipe lives in seq_NN_cfg.py, make_bilzard_cfgs()
|-- experiments/arch/         second fork: four architecture experiments (all negative)
|-- seq_NN_holdout_eval.py    train one recipe, score the holdout -> results/<name>/
|-- seq_NN_rescore.py         rescore existing weights (e.g. --tta 8, --reparam), no training
|-- seq_NN_honest_curve.py    picked / last / late-average RMSE from a run's log (prints only)
|-- seq_NN_robust_compare.py  k* and bootstrap between two runs (writes json + csv)
|-- analysis/                 posterior probes, run_chain.sh
`-- results/<name>/           one folder per run; models.pkl is git-ignored
```

## Protocol

- **Split:** sort the 773 wells by id; the first 618 train, the last 155 are the holdout (754,122 suffix rows).
  No randomness: the split does not depend on the seed.
- **Rows to predict:** where `TVT_input` is NaN. The test wells have no `TVT`, so they cannot be scored.
- **Metric:** pooled RMSE over all holdout rows. The training loop keeps the best-scoring epoch on the holdout, which makes
  "reported" optimistic. Use the late average from `seq_NN_honest_curve.py`.
- **Errors are concentrated:** the worst 10 wells carry about 43% of the squared error. k\* counts how many of the most helpful wells
  must be removed before a gain disappears. The bar is 10, but noise alone has reached 21.

Standard command (cmd.exe):

```bat
python seq_NN_holdout_eval.py --id 0801_V2 --source-dir experiments/bilzard --cfg-name <recipe> --epochs 150 --output-dir results/<recipe>_ep150 --device cuda --offline-timm --batch-size 4 --val-batch-size 4 --grad-accum-steps 4 --num-workers 1
```

- `--id 0801_V2` supplies the archived recipe metadata; `--source-dir` supplies the code.
- An existing output folder is refused; pass `--output-dir` to make a new one, and never overwrite a finished run.
- Batch 4 × accumulation 4 reproduces the archived effective batch of 16 (batch 16 does not fit in memory).

## Adding a recipe

Add an entry to the list returned by `make_bilzard_cfgs()` in `experiments/bilzard/seq_NN_cfg.py`, built
from `base` (the 0801_V2 recipe) or `slope_cfg` (x_slope) with one change. Never change `CFG` defaults.

| Change | File |
|---|---|
| Hyperparameters, channels, simulation, loss weights | `seq_NN_cfg.py` |
| Model, heads | `seq_NN_models.py`, `seq_NN_pretrained_unet.py` |
| Losses, training loop, distillation | `seq_NN_train.py` |
| Inputs, synthetic wells, augmentation | `seq_NN_dataset.py` |

Traps:
- A new submodule must be registered in `model_named_parameters()`, or it is never trained.
- Every new `model_cfg` key needs a matching constructor argument.
- `unet_loss_weights` rejects unknown keys.
- Synthetic wells *replace* real samples (`z_shift.apply_prob` = 0.85, set by the 0801_V2 recipe; the class default of 0.75 does not apply).

## What has been settled (do not re-test without a new reason)

Late-average RMSE in ft, 150 epochs, seed 7 unless noted.

| Result | Late avg |
|---|---|
| ConvNeXt-small control (two runs) | 5.163 / 5.062 |
| + slope loss (seeds 7 / 11) | 4.914 / 4.783 |
| + deep supervision (seeds 7 / 11) | 4.871 / 4.836 |
| slope + deep supervision | 5.178 (worse: do not combine) |
| ConvNeXt-tiny | 5.093 (ties small) |
| FastViT, no teacher | 5.319 |
| FastViT ← plain small | first run 4.968, **did not reproduce** (5.310) |
| FastViT ← slope teacher | 5.053 |

- Capacity has a knee near ConvNeXt-tiny: MobileOne-S1 scored 6.51 reported, far worse.
- **No reliable gain:**
  - training changes: decoys, CRPS loss, real-only finetune, synthetic-data re-skin (`rb_v2_synth`), all-neighbour input channels;
  - architecture: the four `arch` experiments, DINOv2;
  - decoding: second-order decode, temperature, recalibration, sibling-well GR. The mean decode is already the best decoder.
- MD-phase TTA (`--tta 8`) was 4/4 slightly positive (−0.02 to −0.15), but always within noise.

Details: `EXPERIMENTS_2026-10.md` (Oct round) and `NOTES_ARCHIVE_2026-09.md` (September, historical).

## Environment

- huggingface.co fails TLS on this laptop. `--offline-timm` uses the cached weights: ConvNeXt small/tiny/nano, FastViT sa12/sa24/t12, MobileOne-S1, InceptionNeXt-tiny.
- The Windows DataLoader uses `spawn`, so each worker holds a full copy of the dataset in RAM. Use `--num-workers 1`.
- Any probe script that uses multiprocessing must wrap its body in `if __name__ == "__main__":`, or it hangs on Windows.
