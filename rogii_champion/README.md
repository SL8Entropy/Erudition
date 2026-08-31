# ROGII Wellbore Geology Prediction — 1st place, plus five iterations

A working reimplementation of Ruby's 1st place solution (private **5.639**), and
five iterations on it, each borrowing one idea from a different team in the top
five. Runs locally against the `train/` and `test/` folders next to this repo.

The base model is the 2D-alignment formulation: render each well as an image
whose rows are TVT hypotheses and columns are MD bins, and trace the path across
it with a ConvNeXt U-Net trained with cross-entropy down each column.

```
rogii solutions/
├── train/                     773 wells (MD,X,Y,Z,GR,TVT_input,TVT + surfaces)
├── test/                      the 3-well public sample
├── sample_submission.csv
└── rogii_champion/
    ├── configs.py             base + the five variants; one dataclass each
    ├── run.py                 CLI: train / oof / compare / ensemble / infer / blend
    ├── smoke_test.py          end-to-end self-test on generated wells
    └── src/
        ├── data.py            well loading, typewell clustering, folds
        ├── canvas.py          the alignment image and its channels
        ├── features.py        per-well feature provider (fold safety lives here)
        ├── xy_neighbor.py     structural plane fit + neighbourhood statistics
        ├── particle_filter.py PF over (level, slope, GR bias)
        ├── sibling_ref.py     sibling-lateral and self-prefix reference GR
        ├── synth.py           physically consistent synthetic wells
        ├── augment.py         the augmentation stack
        ├── model.py           ConvNeXt U-Net, LayerNorm→BatchNorm, variant heads
        ├── losses.py          smoothed CE + expected-path Huber + GR penalty
        ├── decode.py          expectation, DP, MD-phase TTA, re-anchoring
        ├── gate.py            per-row SoftMax gate over candidate paths
        ├── metrics.py         pooled RMSE + the acceptance test
        ├── train.py           dataset and training loop
        ├── infer.py           checkpoints → submission.csv
        └── ensemble.py        blending, routing, weight fitting
```

## Quick start

```bash
pip install -r requirements.txt
python run.py list
```

`--data-root` is optional — the folder holding `train/` and `test/` is found
automatically. Add `-u` to `python` for any long run, or stdout stays buffered
and the log looks empty while it works.

```bash
python -u run.py train --variant base --out runs --folds 0 --num-workers 3
```

That trains one fold and prints its out-of-fold pooled RMSE. Drop `--folds` to
train all five, and add `--seeds 0 1 2` for the full 15-checkpoint ensemble.

Then, in order:

```bash
# a decode-only variant: same weights, only the TTA changes
python -u run.py oof     --variant v1_tta --weights runs/base --out runs

# is the difference real, or five lucky wells?
python -u run.py compare --base runs/base --treat runs/v1_tta

# blend out-of-fold predictions, with XY-safety routing
python -u run.py ensemble --members base=runs/base,v4=runs/v4_refgr --route

# predictions for test/, written as submission.csv + details_<variant>.csv
python -u run.py infer   --variant base --run-dir runs/base --out submission.csv

# combine several variants' test predictions into one submission
python -u run.py blend   --details details_base.csv,details_v4_refgr.csv \
                         --no-xy-members v4_refgr --out submission.csv
```

`run.py compare` is the one to lean on. It prints the pooled RMSE of both runs
*and* the leave-largest-contribution-out curve that says whether the difference
survives removing the handful of wells producing it — see
[EXPERIMENTS.md](EXPERIMENTS.md).

## Benchmarking all six at once

```bash
python -u run.py benchmark --epochs 12 --num-workers 3 --report benchmark_results.txt
```

Trains every variant on the same 618 wells, scores every variant on the same
155 held-out wells, and writes one report containing — per variant — the
configuration, train time, inference time, held-out pooled RMSE, well-mean
RMSE, rows scored, the worst wells, and the checkpoint path. No-model baselines
are included at the top so the numbers have a scale, and a summary table at the
end ranks the variants against `base`.

### Which split, and why

The shipped `test/` folder **cannot give you an error metric**. It holds 3
wells, they have no `TVT` column, and they are duplicates of wells that are also
in `train/`. It exists to check that the submission path works.

The evaluation that means something here is a split *by well*: rows within a
well are strongly correlated, so a row-wise split leaks and reports a fantasy
score. The benchmark holds out one GroupKFold fold (155 wells, 20%), trains on
the other 618, and scores the held-out wells with the competition's pooled
RMSE. That is the same protocol all five top teams used, and the same quantity
`train_fold` reports as OOF.

One of the three `test/` wells does land in the held-out fold, so the report
also scores that one honestly against its twin's label in `train/`.

### What gets saved

Training writes, per `(split, seed, fold)`:

```
runs/<variant>/split0_seed0_fold0.pt      EMA weights, in_chans, variant, epochs_done
runs/<variant>/manifest.json              variant, canvas, heads, backbone
runs/<variant>/report_split0_seed0_fold0.json
runs/<variant>/oof_split0_seed0_fold0.npz     per-well OOF predictions
runs/<variant>/heldout_fold0.npz              benchmark predictions
```

`infer` and `ensemble` pick those up automatically; nothing has to be moved.

The checkpoint is the trained model — not a crash-recovery file. It is written
once, when training finishes, so a run that dies at epoch 8 of 12 leaves
nothing.

### Re-running

Re-running **trains from scratch and overwrites** the checkpoint. It warns
first, naming how many epochs the existing one had:

```bash
python -u run.py benchmark --epochs 30            # all six, from scratch
python -u run.py benchmark --variants v3_moves    # just one, leaving the rest
```

There is deliberately no resume: a full 30-epoch fold is ~55 minutes, and
continuing a finished run would splice two halves of a cosine learning-rate
schedule together, giving a model that is not the same as one trained straight
through. Use `--variants` to avoid redoing work.

## The five iterations

Each changes **one** thing, at a different stage, from a different team — so a
win or loss is attributable.

| | idea | from | stage | retrain? |
|---|---|---|---|---|
| **v1_tta** | MD-phase TTA ×8, quarter re-anchoring, adaptive canvas, sibling-corrected typewell | 2nd, 4th, 5th | decode | **no** — runs the base weights |
| **v2_slope** | predict the structural slope `dz_layer` and integrate it against the known `dz` | 2nd | target | yes |
| **v3_moves** | per-anchor `P(dTVT｜TVT)` head, decoded by exact DP marginalisation | 2nd | head/decoder | yes |
| **v4_refgr** | sibling-lateral + self-prefix reference GR, corrected typewell | 3rd, 5th | features | yes |
| **v5_gate** | Gaussian-NLL sigma + per-row SoftMax gate over candidates | 3rd | ensembling | yes |

v1 is first on purpose: it costs no training, and if it holds up it becomes part
of the baseline the other four are measured against.

## What the real data says

Measured over all 773 training wells, and it shaped several decisions:

| | |
|---|---|
| wells / rows | 773 / 4.6M, 1 ft spacing |
| visible prefix | 850–2391 ft (median 1702), always contiguous |
| prediction region | 407–10052 ft (median 4840) |
| max ｜TVT − anchor｜ after PS | median 20 ft, p99 84 ft, **max 104 ft** |
| GR missing | median 28% of rows, up to 80% |
| typewell master series | **54** (largest holds 71 wells; 13 singletons) |

Three of those changed the code.

**The canvas window is right, the column count was not.** Exactly one well of
773 leaves ±100 ft after PS, so 1st place's window is well chosen — but the
longest prediction region is 10,052 ft and needs 320 target columns, not the 313
implied by "10,000 ft".

**Typewells only cluster into 54 master series if you group them by curve
overlap.** Hashing each curve exactly gives 752 groups for 773 wells, which
would have left v4's sibling reference with nothing to work with. The overlap
clustering reproduces 2nd place's stated 54 exactly (largest group 71 wells).

**The flat-layer prior is badly wrong on this field.** Predicting
`TVT = anchor − ΔZ` scores 107 ft RMSE and leaves the canvas window for 88% of
wells: the driller steers *down with the formation*, so Z moves hundreds of feet
while TVT barely moves — which is also why "predict no change at all" scores
15.9 ft. Projecting the structural slope estimated from the prefix instead
(`r₀·ΔMD − ΔZ`) scores 20.9 ft and stays in the window for 88% of wells. The
geometry channel now carries the projected prior, not the flat one.

## Measured cost

RTX 3050 6 GB laptop GPU, full 400 × 352 canvas, bf16:

| | |
|---|---|
| training step, batch 4 | 0.26 s, **2.56 GiB** VRAM |
| inference, batch 1 | 0.90 GiB |
| dataloader item (augment + PF + XY + canvas) | ~100 ms |
| **one epoch** (618 train wells + 618 synthetic) | **109 s** with 3 workers, 215 s with 0 |
| scoring 155 held-out wells | 26 s (0.16 s/well) |

How long to train everything:

| scope | per variant | all five |
|---|---|---|
| 1 fold, 12 epochs (the `benchmark` default) | ~23 min | ~2 h |
| 1 fold, 30 epochs (full recipe, one model) | ~55 min | ~4.5 h |
| 5 folds × 1 seed — full OOF, the usable minimum | ~5 h | ~25 h |
| 5 folds × 3 seeds — 1st place's actual ensemble | ~15 h | ~75 h |

`v1_tta` never trains, so "all five" is four trainings plus `v5_gate`'s extra
split patterns. Inference scales with `n_phases × (1 + reanchor)`: `v1_tta` runs
16 views per well, so scoring the held-out set takes ~7 min rather than 26 s.

VRAM is not the constraint; **host RAM** is. Each dataloader worker gets a
pickled copy of the wells and the feature provider (~0.9 GB), so
`--num-workers 3` costs about 5.5 GB of system RAM. Use `--num-workers 2` on a
16 GB machine.

## Self-test

Generates its own wells, so it needs no data and takes a couple of minutes:

```bash
python smoke_test.py
```

It exercises every variant's full path — canvas, features, augmentation, model,
loss, decode, OOF, gate, acceptance test, submission.

## What is faithful and what is not

Faithful to the 1st place write-up: the canvas anchored on the last visible TVT;
the ConvNeXt-Small U-Net with LayerNorm replaced by BatchNorm and
average-pool/interpolate resampling; smoothed cross-entropy down the typewell
axis plus expected-path Huber plus the GR penalty; the per-column statistical
summary of the lateral log; typewell calibration from the visible region; the
particle-filter and XY-neighbour channels with PF-corruption augmentation
against shortcut learning; Z-shift and GR-affine augmentation; GroupKFold by
well with two-vector routing on the neighbourhood statistics.

Not specified in the write-up, so chosen here and marked in `configs.py`:
learning rate, epoch count, loss weights, PF hyperparameters, ensemble weights.

The pretrained backbone downloads from HuggingFace on first use. Behind a
TLS-inspecting proxy that fails; `model.py` routes through the OS trust store
(`truststore`), which fixes it. If it still cannot download, it falls back to
random init with a warning — pass `cfg.backbone_weights` to load a local file
instead.
