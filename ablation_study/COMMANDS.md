# Commands, in order

This is the master list of all 13 runs, written from laptop D's point of view.
**Who runs which block: `TEAM_PLAN.md`.**
- Blocks 3, 6, 7, 8 and 11 are trained on laptops A, B and C. D runs only their compare lines, once their folders have been copied to D.
- In blocks 4, 5, 12 and 13, run each compare line once its reference folder (`cnx_fastvit_s1` or `cnx_fastvit_s2`, from A or B) is on D.

Run in **cmd.exe** (not PowerShell, not bash), one block at a time.
Each training command takes about 2 h on D. Do not start the next block until the current one has finished.
After every block, fill in its row in `RESULTS_LOG.md`. Only D edits that file.

## 0. Every time you open a new terminal

```bat
cd /d "C:\Users\Aadinarayan R\Desktop\CSE\projects\Erudition\ablation_study\pipeline"
call "C:\Users\Aadinarayan R\Desktop\CSE\projects\Erudition\.venv\Scripts\activate.bat"
set COMMON=--id 0801_V2 --source-dir experiments/bilzard --epochs 150 --device cuda --offline-timm --batch-size 4 --val-batch-size 4 --grad-accum-steps 4 --num-workers 1
powercfg /change standby-timeout-ac 0
```

`COMMON` holds the settings shared by every run. If a later command complains about `%COMMON%`, you are in a new window: run block 0 again.
`seq_NN_robust_compare.py` exits with code 1 when its verdict is REJECT. That is normal, not a crash; its output file is still written.
A run refuses to start if its `results\...` folder already exists. Never delete a finished run to "retry"; rename the failed folder to `results\_failed_<recipe>` first.

## Step 1: the two teachers (everything later depends on these)

### 1. `small_s1`: A0, ConvNeXt-small without slope loss, seed 1
The no-slope control, and the teacher for run 4.
```bat
python seq_NN_holdout_eval.py %COMMON% --cfg-name small_s1 --output-dir results/small_s1_ep150
python seq_NN_honest_curve.py results/small_s1_ep150 > results/small_s1_ep150/honest_curve.txt
```

### 2. `x_slope_s1`: A1, ConvNeXt-small with slope loss, seed 1
The paper's teacher. Compared with run 1, this is **ablation A**.
```bat
python seq_NN_holdout_eval.py %COMMON% --cfg-name x_slope_s1 --output-dir results/x_slope_s1_ep150
python seq_NN_honest_curve.py results/x_slope_s1_ep150 > results/x_slope_s1_ep150/honest_curve.txt
python seq_NN_robust_compare.py --base results/small_s1_ep150 --treat results/x_slope_s1_ep150 --output-dir results/x_slope_s1_ep150 > results/x_slope_s1_ep150/compare_vs_small_s1.txt 2>&1
```

## Step 2: seed 1 of the other six arms

### 3. `cnx_fastvit_s1`: B0, FastViT with no teacher, seed 1
The reference for the two students.
```bat
python seq_NN_holdout_eval.py %COMMON% --cfg-name cnx_fastvit_s1 --output-dir results/cnx_fastvit_s1_ep150
python seq_NN_honest_curve.py results/cnx_fastvit_s1_ep150 > results/cnx_fastvit_s1_ep150/honest_curve.txt
```

### 4. `cnx_fastvit_kd_small_s1`: B1, FastViT taught by run 1, seed 1
Distillation from a teacher without the slope loss. Needs `results\small_s1_ep150\models.pkl`.
```bat
python seq_NN_holdout_eval.py %COMMON% --cfg-name cnx_fastvit_kd_small_s1 --output-dir results/cnx_fastvit_kd_small_s1_ep150
python seq_NN_honest_curve.py results/cnx_fastvit_kd_small_s1_ep150 > results/cnx_fastvit_kd_small_s1_ep150/honest_curve.txt
python seq_NN_robust_compare.py --base results/cnx_fastvit_s1_ep150 --treat results/cnx_fastvit_kd_small_s1_ep150 --output-dir results/cnx_fastvit_kd_small_s1_ep150 > results/cnx_fastvit_kd_small_s1_ep150/compare_vs_cnx_fastvit_s1.txt 2>&1
```

### 5. `cnx_fastvit_kd_xslope_s1`: B2, FastViT taught by run 2, seed 1 (**the final model**)
Needs `results\x_slope_s1_ep150\models.pkl`. It is compared with the student without a teacher (run 3) and with the student of the plain teacher (run 4).
```bat
python seq_NN_holdout_eval.py %COMMON% --cfg-name cnx_fastvit_kd_xslope_s1 --output-dir results/cnx_fastvit_kd_xslope_s1_ep150
python seq_NN_honest_curve.py results/cnx_fastvit_kd_xslope_s1_ep150 > results/cnx_fastvit_kd_xslope_s1_ep150/honest_curve.txt
python seq_NN_robust_compare.py --base results/cnx_fastvit_s1_ep150 --treat results/cnx_fastvit_kd_xslope_s1_ep150 --output-dir results/cnx_fastvit_kd_xslope_s1_ep150 > results/cnx_fastvit_kd_xslope_s1_ep150/compare_vs_cnx_fastvit_s1.txt 2>&1
python seq_NN_robust_compare.py --base results/cnx_fastvit_kd_small_s1_ep150 --treat results/cnx_fastvit_kd_xslope_s1_ep150 --output-dir results/cnx_fastvit_kd_xslope_s1_ep150 > results/cnx_fastvit_kd_xslope_s1_ep150/compare_vs_cnx_fastvit_kd_small_s1.txt 2>&1
```

### 6. `x_slope_nosynth_s1`: C1, the teacher without synthetic wells, seed 1
**Ablation C.** Compared with run 2.
```bat
python seq_NN_holdout_eval.py %COMMON% --cfg-name x_slope_nosynth_s1 --output-dir results/x_slope_nosynth_s1_ep150
python seq_NN_honest_curve.py results/x_slope_nosynth_s1_ep150 > results/x_slope_nosynth_s1_ep150/honest_curve.txt
python seq_NN_robust_compare.py --base results/x_slope_s1_ep150 --treat results/x_slope_nosynth_s1_ep150 --output-dir results/x_slope_nosynth_s1_ep150 > results/x_slope_nosynth_s1_ep150/compare_vs_x_slope_s1.txt 2>&1
```

### 7. `x_slope_nocost_s1`: D1, the teacher without the GR-mismatch channel, seed 1
**Ablation D**, first half. Compared with run 2.
```bat
python seq_NN_holdout_eval.py %COMMON% --cfg-name x_slope_nocost_s1 --output-dir results/x_slope_nocost_s1_ep150
python seq_NN_honest_curve.py results/x_slope_nocost_s1_ep150 > results/x_slope_nocost_s1_ep150/honest_curve.txt
python seq_NN_robust_compare.py --base results/x_slope_s1_ep150 --treat results/x_slope_nocost_s1_ep150 --output-dir results/x_slope_nocost_s1_ep150 > results/x_slope_nocost_s1_ep150/compare_vs_x_slope_s1.txt 2>&1
```

### 8. `x_slope_nogeo_s1`: D2, the teacher without the geo-prior channel, seed 1
**Ablation D**, second half. Compared with run 2.
```bat
python seq_NN_holdout_eval.py %COMMON% --cfg-name x_slope_nogeo_s1 --output-dir results/x_slope_nogeo_s1_ep150
python seq_NN_honest_curve.py results/x_slope_nogeo_s1_ep150 > results/x_slope_nogeo_s1_ep150/honest_curve.txt
python seq_NN_robust_compare.py --base results/x_slope_s1_ep150 --treat results/x_slope_nogeo_s1_ep150 --output-dir results/x_slope_nogeo_s1_ep150 > results/x_slope_nogeo_s1_ep150/compare_vs_x_slope_s1.txt 2>&1
```

**Checkpoint:** the seed-1 table in `RESULTS_LOG.md` is now complete. For runs 6–8, if |Δ| < 0.30 ft, add their seed-2 run to step 4.

## Step 3: seed 2 of A and B (always)

These repeat runs 1–5 with seed 2, in the same order and with the same comparisons.

### 9. `small_s2`: A0, seed 2
```bat
python seq_NN_holdout_eval.py %COMMON% --cfg-name small_s2 --output-dir results/small_s2_ep150
python seq_NN_honest_curve.py results/small_s2_ep150 > results/small_s2_ep150/honest_curve.txt
```

### 10. `x_slope_s2`: A1, seed 2
```bat
python seq_NN_holdout_eval.py %COMMON% --cfg-name x_slope_s2 --output-dir results/x_slope_s2_ep150
python seq_NN_honest_curve.py results/x_slope_s2_ep150 > results/x_slope_s2_ep150/honest_curve.txt
python seq_NN_robust_compare.py --base results/small_s2_ep150 --treat results/x_slope_s2_ep150 --output-dir results/x_slope_s2_ep150 > results/x_slope_s2_ep150/compare_vs_small_s2.txt 2>&1
```

### 11. `cnx_fastvit_s2`: B0, seed 2
```bat
python seq_NN_holdout_eval.py %COMMON% --cfg-name cnx_fastvit_s2 --output-dir results/cnx_fastvit_s2_ep150
python seq_NN_honest_curve.py results/cnx_fastvit_s2_ep150 > results/cnx_fastvit_s2_ep150/honest_curve.txt
```

### 12. `cnx_fastvit_kd_small_s2`: B1, seed 2 (teacher = run 9)
```bat
python seq_NN_holdout_eval.py %COMMON% --cfg-name cnx_fastvit_kd_small_s2 --output-dir results/cnx_fastvit_kd_small_s2_ep150
python seq_NN_honest_curve.py results/cnx_fastvit_kd_small_s2_ep150 > results/cnx_fastvit_kd_small_s2_ep150/honest_curve.txt
python seq_NN_robust_compare.py --base results/cnx_fastvit_s2_ep150 --treat results/cnx_fastvit_kd_small_s2_ep150 --output-dir results/cnx_fastvit_kd_small_s2_ep150 > results/cnx_fastvit_kd_small_s2_ep150/compare_vs_cnx_fastvit_s2.txt 2>&1
```

### 13. `cnx_fastvit_kd_xslope_s2`: B2, the final model, seed 2 (teacher = run 10)
```bat
python seq_NN_holdout_eval.py %COMMON% --cfg-name cnx_fastvit_kd_xslope_s2 --output-dir results/cnx_fastvit_kd_xslope_s2_ep150
python seq_NN_honest_curve.py results/cnx_fastvit_kd_xslope_s2_ep150 > results/cnx_fastvit_kd_xslope_s2_ep150/honest_curve.txt
python seq_NN_robust_compare.py --base results/cnx_fastvit_s2_ep150 --treat results/cnx_fastvit_kd_xslope_s2_ep150 --output-dir results/cnx_fastvit_kd_xslope_s2_ep150 > results/cnx_fastvit_kd_xslope_s2_ep150/compare_vs_cnx_fastvit_s2.txt 2>&1
python seq_NN_robust_compare.py --base results/cnx_fastvit_kd_small_s2_ep150 --treat results/cnx_fastvit_kd_xslope_s2_ep150 --output-dir results/cnx_fastvit_kd_xslope_s2_ep150 > results/cnx_fastvit_kd_xslope_s2_ep150/compare_vs_cnx_fastvit_kd_small_s2.txt 2>&1
```

## Step 4: only if the checkpoint after run 8 asks for it

Same as runs 6–8 with `_s2`, compared with run 10 (`x_slope_s2`). Example for C1:
```bat
python seq_NN_holdout_eval.py %COMMON% --cfg-name x_slope_nosynth_s2 --output-dir results/x_slope_nosynth_s2_ep150
python seq_NN_honest_curve.py results/x_slope_nosynth_s2_ep150 > results/x_slope_nosynth_s2_ep150/honest_curve.txt
python seq_NN_robust_compare.py --base results/x_slope_s2_ep150 --treat results/x_slope_nosynth_s2_ep150 --output-dir results/x_slope_nosynth_s2_ep150 > results/x_slope_nosynth_s2_ep150/compare_vs_x_slope_s2.txt 2>&1
```
For D1 or D2, replace `nosynth` with `nocost` or `nogeo`.

## When everything is done

```bat
powercfg /change standby-timeout-ac 5
```
