# Commands for all 13 runs

This is the master command list, written from the point of view of **laptop D**.
Who trains which run is in `TEAM_PLAN.md`.

## Rules (read first)

1. **Shell:** use **cmd.exe** (not PowerShell, not bash).
2. **One run at a time.** Each training command takes about 2 h on D. Do not start the next run until the current one has finished.
3. **Runs trained on laptops A, B or C** (runs 3, 6, 7, 8 and 11): D does not train them. Once their results folders have been copied to D, D runs only their compare lines (runs 3 and 11 have none).
4. **Compare lines need their reference folders on D first.** The "Needs" column in the table below lists which folders. If a reference folder comes from another laptop, wait until it has been copied over.
5. **A run refuses to start if its `results/...` folder already exists.** Never delete a finished run to retry it. Rename the failed folder to `results/_failed_<recipe>` first.
6. **Exit code 1 from `seq_NN_robust_compare.py` is normal.** It means the verdict is REJECT, not that the script crashed. The output file is still written.
7. **After every run, fill in its row in `RESULTS_LOG.md`.** Only D edits that file.

## Run overview

All folders are under `results/` and end in `_ep150` (for example, `small_s1` means `results/small_s1_ep150`).

| Run | Recipe | Arm | Needs before it can run | Compared with |
|----:|--------|-----|-------------------------|---------------|
| 1 | `small_s1` | A0: ConvNeXt-small, no slope loss (seed 1) | nothing | none |
| 2 | `x_slope_s1` | A1: ConvNeXt-small with slope loss (seed 1) | nothing | `small_s1` (**ablation A**) |
| 3 | `cnx_fastvit_s1` | B0: FastViT, no teacher (seed 1) | nothing | none |
| 4 | `cnx_fastvit_kd_small_s1` | B1: FastViT taught by run 1 | `small_s1/models.pkl` | `cnx_fastvit_s1` |
| 5 | `cnx_fastvit_kd_xslope_s1` | B2: FastViT taught by run 2 (**final model**) | `x_slope_s1/models.pkl` | `cnx_fastvit_s1`, `cnx_fastvit_kd_small_s1` |
| 6 | `x_slope_nosynth_s1` | C1: teacher without synthetic wells | nothing | `x_slope_s1` (**ablation C**) |
| 7 | `x_slope_nocost_s1` | D1: teacher without the GR-mismatch channel | nothing | `x_slope_s1` (**ablation D**, first half) |
| 8 | `x_slope_nogeo_s1` | D2: teacher without the geo-prior channel | nothing | `x_slope_s1` (**ablation D**, second half) |
| 9 | `small_s2` | A0, seed 2 | nothing | none |
| 10 | `x_slope_s2` | A1, seed 2 | nothing | `small_s2` |
| 11 | `cnx_fastvit_s2` | B0, seed 2 | nothing | none |
| 12 | `cnx_fastvit_kd_small_s2` | B1, seed 2 | `small_s2/models.pkl` (run 9) | `cnx_fastvit_s2` |
| 13 | `cnx_fastvit_kd_xslope_s2` | B2, seed 2 (**final model**) | `x_slope_s2/models.pkl` (run 10) | `cnx_fastvit_s2`, `cnx_fastvit_kd_small_s2` |

Runs 1-5 are the seed-1 core, runs 6-8 are the seed-1 ablations, and runs 9-13 repeat runs 1-5 with seed 2.

## Setup (only for manual typing)

If you start a run with `open_terminal.bat` or any other `.bat` file in the project, setup is automatic. Skip this section.

**Option 1:** run `open_terminal.bat`, then type your commands in the terminal that opens.

**Option 2:** set up by hand.

1. Open cmd.exe **in the `ablation_study` folder** (in File Explorer, click the address bar, type `cmd`, press Enter). All paths below are relative to that folder, so the project works wherever it is copied.
2. Run:

```bat
cd pipeline
call "../../.venv/Scripts/activate.bat"
set COMMON=--id 0801_V2 --source-dir experiments/bilzard --epochs 150 --device cuda --offline-timm --batch-size 4 --val-batch-size 4 --grad-accum-steps 4 --num-workers 1
powercfg /change standby-timeout-ac 0
```

`COMMON` holds the settings shared by every run. If a command complains about `%COMMON%`, you are in a new window: repeat this setup.

## Step 1: the two teachers

Everything later depends on these.

### Run 1: `small_s1` (A0)

The no-slope control, and the teacher for run 4.

```bat
REM Train
python seq_NN_holdout_eval.py %COMMON% --cfg-name small_s1 --output-dir results/small_s1_ep150
REM Honest curve
python seq_NN_honest_curve.py results/small_s1_ep150 > results/small_s1_ep150/honest_curve.txt
```

### Run 2: `x_slope_s1` (A1)

The paper's teacher, and the teacher for run 5. Compared with run 1, this is **ablation A**.

```bat
REM Train
python seq_NN_holdout_eval.py %COMMON% --cfg-name x_slope_s1 --output-dir results/x_slope_s1_ep150
REM Honest curve
python seq_NN_honest_curve.py results/x_slope_s1_ep150 > results/x_slope_s1_ep150/honest_curve.txt
REM Compare vs run 1
python seq_NN_robust_compare.py --base results/small_s1_ep150 --treat results/x_slope_s1_ep150 --output-dir results/x_slope_s1_ep150 > results/x_slope_s1_ep150/compare_vs_small_s1.txt 2>&1
```

## Step 2: seed 1 of the other six arms

### Run 3: `cnx_fastvit_s1` (B0)

FastViT with no teacher. This is the reference for runs 4 and 5.

```bat
REM Train
python seq_NN_holdout_eval.py %COMMON% --cfg-name cnx_fastvit_s1 --output-dir results/cnx_fastvit_s1_ep150
REM Honest curve
python seq_NN_honest_curve.py results/cnx_fastvit_s1_ep150 > results/cnx_fastvit_s1_ep150/honest_curve.txt
```

### Run 4: `cnx_fastvit_kd_small_s1` (B1)

FastViT taught by the teacher from run 1 (no slope loss). Needs `results/small_s1_ep150/models.pkl`.

```bat
REM Train
python seq_NN_holdout_eval.py %COMMON% --cfg-name cnx_fastvit_kd_small_s1 --output-dir results/cnx_fastvit_kd_small_s1_ep150
REM Honest curve
python seq_NN_honest_curve.py results/cnx_fastvit_kd_small_s1_ep150 > results/cnx_fastvit_kd_small_s1_ep150/honest_curve.txt
REM Compare vs run 3
python seq_NN_robust_compare.py --base results/cnx_fastvit_s1_ep150 --treat results/cnx_fastvit_kd_small_s1_ep150 --output-dir results/cnx_fastvit_kd_small_s1_ep150 > results/cnx_fastvit_kd_small_s1_ep150/compare_vs_cnx_fastvit_s1.txt 2>&1
```

### Run 5: `cnx_fastvit_kd_xslope_s1` (B2, the final model)

FastViT taught by the teacher from run 2 (with slope loss). Needs `results/x_slope_s1_ep150/models.pkl`.

```bat
REM Train
python seq_NN_holdout_eval.py %COMMON% --cfg-name cnx_fastvit_kd_xslope_s1 --output-dir results/cnx_fastvit_kd_xslope_s1_ep150
REM Honest curve
python seq_NN_honest_curve.py results/cnx_fastvit_kd_xslope_s1_ep150 > results/cnx_fastvit_kd_xslope_s1_ep150/honest_curve.txt
REM Compare vs run 3 (student without a teacher)
python seq_NN_robust_compare.py --base results/cnx_fastvit_s1_ep150 --treat results/cnx_fastvit_kd_xslope_s1_ep150 --output-dir results/cnx_fastvit_kd_xslope_s1_ep150 > results/cnx_fastvit_kd_xslope_s1_ep150/compare_vs_cnx_fastvit_s1.txt 2>&1
REM Compare vs run 4 (student of the plain teacher)
python seq_NN_robust_compare.py --base results/cnx_fastvit_kd_small_s1_ep150 --treat results/cnx_fastvit_kd_xslope_s1_ep150 --output-dir results/cnx_fastvit_kd_xslope_s1_ep150 > results/cnx_fastvit_kd_xslope_s1_ep150/compare_vs_cnx_fastvit_kd_small_s1.txt 2>&1
```

### Run 6: `x_slope_nosynth_s1` (C1, **ablation C**)

The teacher without synthetic wells. Compared with run 2.

```bat
REM Train
python seq_NN_holdout_eval.py %COMMON% --cfg-name x_slope_nosynth_s1 --output-dir results/x_slope_nosynth_s1_ep150
REM Honest curve
python seq_NN_honest_curve.py results/x_slope_nosynth_s1_ep150 > results/x_slope_nosynth_s1_ep150/honest_curve.txt
REM Compare vs run 2
python seq_NN_robust_compare.py --base results/x_slope_s1_ep150 --treat results/x_slope_nosynth_s1_ep150 --output-dir results/x_slope_nosynth_s1_ep150 > results/x_slope_nosynth_s1_ep150/compare_vs_x_slope_s1.txt 2>&1
```

### Run 7: `x_slope_nocost_s1` (D1, **ablation D**, first half)

The teacher without the GR-mismatch channel. Compared with run 2.

```bat
REM Train
python seq_NN_holdout_eval.py %COMMON% --cfg-name x_slope_nocost_s1 --output-dir results/x_slope_nocost_s1_ep150
REM Honest curve
python seq_NN_honest_curve.py results/x_slope_nocost_s1_ep150 > results/x_slope_nocost_s1_ep150/honest_curve.txt
REM Compare vs run 2
python seq_NN_robust_compare.py --base results/x_slope_s1_ep150 --treat results/x_slope_nocost_s1_ep150 --output-dir results/x_slope_nocost_s1_ep150 > results/x_slope_nocost_s1_ep150/compare_vs_x_slope_s1.txt 2>&1
```

### Run 8: `x_slope_nogeo_s1` (D2, **ablation D**, second half)

The teacher without the geo-prior channel. Compared with run 2.

```bat
REM Train
python seq_NN_holdout_eval.py %COMMON% --cfg-name x_slope_nogeo_s1 --output-dir results/x_slope_nogeo_s1_ep150
REM Honest curve
python seq_NN_honest_curve.py results/x_slope_nogeo_s1_ep150 > results/x_slope_nogeo_s1_ep150/honest_curve.txt
REM Compare vs run 2
python seq_NN_robust_compare.py --base results/x_slope_s1_ep150 --treat results/x_slope_nogeo_s1_ep150 --output-dir results/x_slope_nogeo_s1_ep150 > results/x_slope_nogeo_s1_ep150/compare_vs_x_slope_s1.txt 2>&1
```

**Checkpoint.** The seed-1 table in `RESULTS_LOG.md` is now complete. For each of runs 6, 7 and 8 whose |Δ| is below 0.30 ft, add its seed-2 run from Step 4.

## Step 3: seed 2 of arms A and B (always run)

These repeat runs 1-5 with seed 2, in the same order and with the same comparisons.

### Run 9: `small_s2` (A0)

```bat
REM Train
python seq_NN_holdout_eval.py %COMMON% --cfg-name small_s2 --output-dir results/small_s2_ep150
REM Honest curve
python seq_NN_honest_curve.py results/small_s2_ep150 > results/small_s2_ep150/honest_curve.txt
```

### Run 10: `x_slope_s2` (A1)

```bat
REM Train
python seq_NN_holdout_eval.py %COMMON% --cfg-name x_slope_s2 --output-dir results/x_slope_s2_ep150
REM Honest curve
python seq_NN_honest_curve.py results/x_slope_s2_ep150 > results/x_slope_s2_ep150/honest_curve.txt
REM Compare vs run 9
python seq_NN_robust_compare.py --base results/small_s2_ep150 --treat results/x_slope_s2_ep150 --output-dir results/x_slope_s2_ep150 > results/x_slope_s2_ep150/compare_vs_small_s2.txt 2>&1
```

### Run 11: `cnx_fastvit_s2` (B0)

```bat
REM Train
python seq_NN_holdout_eval.py %COMMON% --cfg-name cnx_fastvit_s2 --output-dir results/cnx_fastvit_s2_ep150
REM Honest curve
python seq_NN_honest_curve.py results/cnx_fastvit_s2_ep150 > results/cnx_fastvit_s2_ep150/honest_curve.txt
```

### Run 12: `cnx_fastvit_kd_small_s2` (B1)

Teacher is run 9. Needs `results/small_s2_ep150/models.pkl`.

```bat
REM Train
python seq_NN_holdout_eval.py %COMMON% --cfg-name cnx_fastvit_kd_small_s2 --output-dir results/cnx_fastvit_kd_small_s2_ep150
REM Honest curve
python seq_NN_honest_curve.py results/cnx_fastvit_kd_small_s2_ep150 > results/cnx_fastvit_kd_small_s2_ep150/honest_curve.txt
REM Compare vs run 11
python seq_NN_robust_compare.py --base results/cnx_fastvit_s2_ep150 --treat results/cnx_fastvit_kd_small_s2_ep150 --output-dir results/cnx_fastvit_kd_small_s2_ep150 > results/cnx_fastvit_kd_small_s2_ep150/compare_vs_cnx_fastvit_s2.txt 2>&1
```

### Run 13: `cnx_fastvit_kd_xslope_s2` (B2, the final model)

Teacher is run 10. Needs `results/x_slope_s2_ep150/models.pkl`.

```bat
REM Train
python seq_NN_holdout_eval.py %COMMON% --cfg-name cnx_fastvit_kd_xslope_s2 --output-dir results/cnx_fastvit_kd_xslope_s2_ep150
REM Honest curve
python seq_NN_honest_curve.py results/cnx_fastvit_kd_xslope_s2_ep150 > results/cnx_fastvit_kd_xslope_s2_ep150/honest_curve.txt
REM Compare vs run 11 (student without a teacher)
python seq_NN_robust_compare.py --base results/cnx_fastvit_s2_ep150 --treat results/cnx_fastvit_kd_xslope_s2_ep150 --output-dir results/cnx_fastvit_kd_xslope_s2_ep150 > results/cnx_fastvit_kd_xslope_s2_ep150/compare_vs_cnx_fastvit_s2.txt 2>&1
REM Compare vs run 12 (student of the plain teacher)
python seq_NN_robust_compare.py --base results/cnx_fastvit_kd_small_s2_ep150 --treat results/cnx_fastvit_kd_xslope_s2_ep150 --output-dir results/cnx_fastvit_kd_xslope_s2_ep150 > results/cnx_fastvit_kd_xslope_s2_ep150/compare_vs_cnx_fastvit_kd_small_s2.txt 2>&1
```

## Step 4: seed 2 of ablations C and D (only if the checkpoint after run 8 asks for it)

These repeat runs 6-8 with seed 2, and are compared with run 10 (`x_slope_s2`).

| Ablation run | Replace `<abl>` with |
|--------------|----------------------|
| C1 (run 6) | `nosynth` |
| D1 (run 7) | `nocost` |
| D2 (run 8) | `nogeo` |

Example for C1 (`<abl>` = `nosynth`). For D1 or D2, change `nosynth` to `nocost` or `nogeo` on every line.

```bat
REM Train
python seq_NN_holdout_eval.py %COMMON% --cfg-name x_slope_nosynth_s2 --output-dir results/x_slope_nosynth_s2_ep150
REM Honest curve
python seq_NN_honest_curve.py results/x_slope_nosynth_s2_ep150 > results/x_slope_nosynth_s2_ep150/honest_curve.txt
REM Compare vs run 10
python seq_NN_robust_compare.py --base results/x_slope_s2_ep150 --treat results/x_slope_nosynth_s2_ep150 --output-dir results/x_slope_nosynth_s2_ep150 > results/x_slope_nosynth_s2_ep150/compare_vs_x_slope_s2.txt 2>&1
```

## When everything is done

Restore the normal sleep timeout (5 minutes):

```bat
powercfg /change standby-timeout-ac 5
```
