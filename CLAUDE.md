# Erudition

TVT (stratigraphic depth) prediction for horizontal wells, built on the ROGII Kaggle 1st-place
pipeline, written up as an IEEE TGRS paper. **This file is the entry point. Read it first.**

## Where things are

| Path | What it is |
|---|---|
| `ablation_shin/` | **Current work:** the paper's ablation study. Read its `README.md`, `PLAN.md`, `COMMANDS.md`; log every run in `RESULTS_LOG.md`. Runs start from `ablation_shin/pipeline/`. |
| `kaggle_1st_place/` | The research pipeline the paper is built on. Its own `CLAUDE.md` covers how it works. |
| `kaggle_1st_place/EXPERIMENTS_2026-10.md` | The training round that produced the slope loss (seeds 7/11). |
| `kaggle_1st_place/NOTES_ARCHIVE_2026-09.md` | Archive of the September experiments. Out of date: read only for history. |
| `kaggle_1st_place/fig*.png`, `make_methodology_figures.py` | The paper's methodology figures. |
| `rogii_champion/` | A separate reimplementation (30-epoch base 13.52 ft). **Not part of the paper.** Never mix its numbers with the pipeline's. |
| `kaggle2ndplace/` | 2nd-place solution code, used only as a reference. |
| `wellbore-ml-gaps/report.md` | Literature and competition review (31 topics). |

The canonical paper draft is `TVT_IEEE.pdf`, which is kept outside the repo (the owner's Downloads folder). There is no other draft.
Everything is on `main`; the branches `g` and `analysis/posterior-bimodality` are fully merged into it.

## The paper model

- **Teacher:** a ConvNeXt-small U-Net trained with the slope loss (`x_slope` recipe).
- **Deployed model:** a FastViT-SA12 student distilled from that teacher (`cnx_fastvit_kd_xslope`).
- **The student itself has slope-loss weight 0.** It gets the slope behaviour only from the teacher.

Numbers from the October runs (seeds 7/11), late-average RMSE in ft:

| Model | Late avg |
|---|---|
| ConvNeXt-small, no slope loss (control) | 5.112 (two runs) |
| Teacher, mean of 2 seeds | 4.849 |
| Student, 1 seed | 5.053 |

Paper Table II still quotes an older student (4.83) that did not reproduce. The paper's tables will be filled from `ablation_shin`, seeds 1 and 2.

## Rules

- **Seeds 1 and 2** for paper runs, then 3 and 4 if needed. A student's teacher is the run with the same seed. Seed-7/11 runs are history, not paper rows.
- **Main metric: late-average RMSE** (mean holdout RMSE over epochs 100–150, `seq_NN_honest_curve.py`).
  The "reported" RMSE keeps the epoch that scored best on the test wells, so it is 0.06–0.19 ft optimistic.
- **Noise:** two identical runs differ by 0.10–0.13 ft (students by up to 0.34). A seed-only null run reached k\*=21.
  So treat |Δ| < 0.10 ft as unmeasured, and k\* alone as no proof.
- **Commits** carry no Claude co-author lines.

## Machine (laptop D: RTX 5060 8 GB, 16 GB RAM, Windows)

- Python: `Erudition\.venv` (torch 2.11 + CUDA). The plain `python` on PATH has no torch.
- The user runs commands in **cmd.exe**.
- Always pass `--num-workers 1` (more workers run out of RAM) and `--offline-timm` (huggingface.co fails TLS here).
- One 150-epoch run takes ~2 h. For overnight runs, turn sleep off with `powercfg /change standby-timeout-ac 0`, and restore it to 5 afterwards.
- `git fetch`/`push` need the sandbox disabled (no DNS inside it).
