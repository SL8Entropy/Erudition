# ROGII Wellbore Geology Prediction — 2nd place weights

Everything the winning submission notebook loads at inference time, and nothing else.
Private leaderboard 5.802.

> **Local training** (added here, not part of the released bundle): `anchor_data.py` and
> `anchor_train.py` train one model of the 32 ft `dzl_w1` family on 80% of the wells in
> `data/` and score the held-out 20%, reusing the author's `src/` modules unmodified.
> `anchor_eval.py` rescores a checkpoint. See [TRAINING.md](TRAINING.md).

## Contents

| path | count | contents |
|---|---:|---|
| `model/plan147/dzl_w1/ep119_seed*_fold*.pt` | 15 | dzl (32 ft MD columns, auxiliary dz_layer head) |
| `model/plan148/r16/ep119_seed*_fold*.pt` | 15 | r16 (16 ft MD columns) |
| `src/*.py` | 4 | inference modules, byte-identical to those in the code archive |
| `wheels/timm-*.whl` | 1 | offline install, used only if timm is absent from the Kaggle image |

Weights total 436 MB. Each family is 3 seeds x 5 folds, taken at epoch 119.

Licensed under the Apache License, Version 2.0. See `LICENSE` and `NOTICE`.

The two families differ only in the MD column width of the input and state grid, 32 ft
and 16 ft. The submission runs both, averages the 15 checkpoints and 8 test-time MD phase
shifts within each family, then averages the two families 1:1.

## Reproducing these weights

Train them with `script/train_dzl.sh` and `script/train_r16.sh` from the code archive.
Those scripts write to exactly the paths above, so a freshly trained set drops straight in.
See the archive's README for hardware, environment and run times.
