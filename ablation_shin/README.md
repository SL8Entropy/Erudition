# ablation_shin

Ablation study for the paper *Deep Learning-Based Probabilistic Stratigraphic Depth Estimation
for Real-Time Geosteering* (`TVT_IEEE.pdf`). Self-contained: everything needed to run it is in this folder,
except the data and the pretrained weights (see Setup).

| File | What it is |
|---|---|
| `PLAN.md` | What is ablated, which runs, and how a result is judged. **Read first.** |
| `TEAM_PLAN.md` | Who runs what on laptops A–D: setup, fit test, per-laptop commands, hand-over to D. |
| `COMMANDS.md` | Every command, in the order to run them, with a one-line explanation each. |
| `RESULTS_LOG.md` | Fill in one row after each run finishes. |
| `pipeline/` | The training code: a copy of `kaggle_1st_place/solution` (commit `2ac58d6`), trimmed to what these runs use. Run everything from here, not from `kaggle_1st_place/solution`. |

This folder lives at `Erudition/ablation_shin/`, inside the Erudition repo.

## Layout of `pipeline/`

```
pipeline/
|-- SETTINGS.json               paths, all relative to this file
|-- seq_NN_holdout_eval.py      trains one recipe and scores it on the 155 holdout wells
|-- seq_NN_main_reproduce.py    loader used by the script above (do not run directly)
|-- generate_train_geo_map.py   rebuilds data/train_png_typewell_map.csv at start-up (do not run directly)
|-- seq_NN_honest_curve.py      late-average RMSE (the paper's main number) from a run's log
|-- seq_NN_robust_compare.py    k* and bootstrap test between two runs
|-- requirements.txt            dependency pins (requirements-local.txt: exact lock)
|-- experiments/bilzard/        the model code; the 16 ablation recipes are at the end of
|                               make_bilzard_cfgs() in seq_NN_cfg.py
|-- Original_Comp_Models/0801_V2/  archived recipe metadata read via --id 0801_V2 (do not edit)
|-- data/                       NOT in git: copy it in (Setup step 2)
`-- results/<recipe>_ep150/     one folder per finished run (created by the runs)
```

## Setup (once per laptop)

1. **Python with CUDA PyTorch.** On laptop D this already exists: `Erudition\.venv` (torch 2.11 + CUDA 12.8,
   timm 1.0.26). `COMMANDS.md` step 0 activates it. **The plain `python` on PATH has no torch.**
   On another laptop, create a venv, install CUDA PyTorch, then from `pipeline\`:
   `python -m pip install -r requirements.txt`
2. **Data.** `pipeline\data` must hold the same files as `kaggle_1st_place\solution\data`
   (1.4 GB: `train\`, `test\`, `train_png_typewell_map.csv`). Neither is in git. On laptop D,
   `pipeline\data` is already a junction to that folder. Elsewhere, make one from `ablation_shin\pipeline`:
   `mklink /J data ..\..\kaggle_1st_place\solution\data`, or copy the folder in.
3. **Pretrained weights.** On a machine where huggingface.co works, run once without `--offline-timm`:
   `python -c "import timm; timm.create_model('hf_hub:timm/convnext_small.in12k_ft_in1k_384', pretrained=True); timm.create_model('hf_hub:timm/fastvit_sa12.apple_dist_in1k', pretrained=True)"`
   Laptop D has both cached already. Every command then uses `--offline-timm`.
4. **Overnight runs:** plug in, keep the lid open, and turn sleep off with `powercfg /change standby-timeout-ac 0`.
   Turn it back on afterwards with `powercfg /change standby-timeout-ac 5`. Pause Windows Update.

## Where the results go

Nothing important stays only in the terminal. Each run writes to `pipeline\results\<recipe>_ep150\`:

| File | Contents | In git? |
|---|---|---|
| `seq_nn.log` | full training log, including the holdout RMSE every 5 epochs | yes |
| `holdout_metrics.json` | final "reported" pooled RMSE and per-well stats | yes |
| `honest_curve.txt` | picked / last / **late-average** RMSE (written by the COMMANDS.md redirect) | yes |
| `compare_vs_<ref>.txt` + `robust_compare__vs_<ref>.json` | k\*, bootstrap, verdict | yes |
| `holdout_predictions.pqt` (37 MB) | every holdout row's prediction; needed by the compare step | no, copy by hand |
| `models.pkl` (~205 MB) | trained weights; teachers' weights are needed by the student runs | no, copy by hand |

After each run, the folder goes to laptop D, and D copies its numbers into `RESULTS_LOG.md` and commits (see `TEAM_PLAN.md` §5–6).
