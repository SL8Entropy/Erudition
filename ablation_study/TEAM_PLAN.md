# Ablation study: team plan (4 laptops)

**What we are doing.** The paper's final model is a ConvNeXt-small U-Net trained with the slope loss (the *teacher*),
distilled into a FastViT-SA12 network (the *student*, the deployed model). We test the four things the paper claims matter
by training the model with each one removed:

| Ablation | Question | Arms |
|---|---|---|
| **A. Slope loss** | Does the new slope loss term help? | A0 ConvNeXt-small, no slope loss · A1 with slope loss (the teacher) |
| **B. Distillation** | Does a teacher help FastViT, and does the slope-loss teacher help more? | B0 FastViT alone · B1 FastViT ← A0 · B2 FastViT ← A1 (**the final model**) |
| **C. Synthetic wells** | Are the synthetic training wells (85% of samples) needed? | C1 = A1 with synthetic wells off |
| **D. Input channels** | Do the GR-mismatch and geo-prior input channels matter? | D1 = A1 without `gr_abs_diff` · D2 = A1 without `geo_tvt_diff` |

Protocol, identical for every run:
- Fixed 80/20 split: train on 618 wells, test on 155.
- 150 epochs, batch 4 × gradient accumulation 4.
- Seeds 1 and 2.
- Main score: late-average RMSE (epochs 100–150).

Full rules are in `ablation_study/PLAN.md`.

**13 runs total.** Expected wall time is about **15–18 hours**, with all four laptops working at once.

---

## 1. Who runs what

| Laptop | GPU | Queue, in order | Est. time |
|---|---|---|---|
| **A** | RTX 3050 4 GB | fit test → **A-1** `cnx_fastvit_s1` | ~3 h |
| **B** | RTX 2050 4 GB | fit test → **B-1** `cnx_fastvit_s2` | ~4 h |
| **C** | RTX 3050 6 GB | **C-1** `x_slope_nosynth_s1` → **C-2** `x_slope_nocost_s1` → **C-3** `x_slope_nogeo_s1` | ~15 h |
| **D** | RTX 5060 8 GB | **D-1** `small_s1` → **D-2** `x_slope_s1` → **D-3** `cnx_fastvit_kd_small_s1` → **D-4** `cnx_fastvit_kd_xslope_s1` → **D-5** `small_s2` → **D-6** `x_slope_s2` → **D-7** `cnx_fastvit_kd_small_s2` → **D-8** `cnx_fastvit_kd_xslope_s2` | ~15–18 h |

**Why this split:**
- **D runs every teacher and every student.** A student needs its teacher's weights (`models.pkl`, 205 MB) on the same machine. D is also the fastest laptop.
- **C runs the ConvNeXt ablations C1, D1, D2.** Their training does not depend on any other run, so C starts immediately.
- **A and B run only FastViT-without-teacher (B0).** ConvNeXt-small needs ~3.9 GiB of GPU memory, so it will not fit on a 4 GB card. FastViT is much smaller, but it still has to pass the fit test (section 3).
- **If A or B fails the fit test,** that run moves to the end of D's queue (+~1.6 h).
- **Possible extra runs.** If C1, D1 or D2 changes the score by less than 0.30 ft on seed 1, it needs a seed-2 run. Those runs (`x_slope_nosynth_s2`, `x_slope_nocost_s2`, `x_slope_nogeo_s2`) go to whichever of C or D finishes first. D decides once C's results are in.

**Never change** the batch size, the gradient accumulation, the epoch count or the recipe on one laptop to make a run fit. That breaks the comparison. If a run does not fit, report it and stop.

---

## 2. Setup (every laptop, once, before anything else)

1. **Get the code.**
   - `git pull` on `main` of the Erudition repo.
   - Run `git rev-parse --short HEAD` and send the commit hash to the group.
   - **Everyone must be on the same commit.** Do not edit any code.
2. **Python.** You need Python with CUDA PyTorch.
   - In `ablation_study\pipeline`, run `python -m pip install -r requirements.txt`.
   - Check with `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`. It must print `True` and your GPU.
3. **Data.** Copy the `data` folder (1.4 GB, from laptop D's `kaggle_1st_place\solution\data`) to `ablation_study\pipeline\data`.
   It must contain `train\`, `test\` and `train_png_typewell_map.csv`.
4. **Pretrained weights.** Run this once, online, without `--offline-timm`:
   - Laptops A and B: `python -c "import timm; timm.create_model('hf_hub:timm/fastvit_sa12.apple_dist_in1k', pretrained=True)"`
   - Laptop C: `python -c "import timm; timm.create_model('hf_hub:timm/convnext_small.in12k_ft_in1k_384', pretrained=True)"`

   If the download fails, copy the matching `models--timm--...` folder from laptop D's `%USERPROFILE%\.cache\huggingface\hub\` into the same place on your laptop.
5. **Power.**
   - Plug in, keep the lid open and pause Windows Update.
   - Turn sleep off with `powercfg /change standby-timeout-ac 0`.
   - When your queue is done, turn it back on with `powercfg /change standby-timeout-ac 5`.

**Every new terminal (cmd.exe)** starts with these three lines. Replace the path with your own clone, and activate your own environment if you use one:

```bat
cd /d "<your Erudition folder>\ablation_study\pipeline"
set COMMON=--id 0801_V2 --source-dir experiments/bilzard --epochs 150 --device cuda --offline-timm --batch-size 4 --val-batch-size 4 --grad-accum-steps 4 --num-workers 1
echo %COMMON%
```

On laptop D only, also run `call "C:\Users\Aadinarayan R\Desktop\CSE\projects\Erudition\.venv\Scripts\activate.bat"`: the plain `python` there has no torch.

If you see "paging file is too small" or a memory error from the data loader, change `--num-workers 1` to `--num-workers 0` and tell D.

---

## 3. Fit test (laptops A and B only, before your real run)

```bat
python seq_NN_holdout_eval.py %COMMON% --epochs 1 --cfg-name cnx_fastvit_s1 --output-dir results/_fittest
```

- **Pass:** it finishes with `HOLDOUT POOLED RMSE` near the end. Then delete `results\_fittest` (`rmdir /s /q results\_fittest`) and start your run.
- **Fail:** it stops with `CUDA out of memory`. Tell D: your run moves to D. Delete `results\_fittest`.

(`--epochs 1` comes after `%COMMON%`, so it overrides the 150 in COMMON. It is for this test only.)

---

## 4. Running your queue (unattended, overnight)

**Double-click your laptop's file in `ablation_study\bat\`:**

| Laptop | File | Runs, in order |
|---|---|---|
| A | `queue_A.bat` | `cnx_fastvit_s1` |
| B | `queue_B.bat` | `cnx_fastvit_s2` |
| C | `queue_C.bat` | `x_slope_nosynth_s1` → `x_slope_nocost_s1` → `x_slope_nogeo_s1` |
| D | `queue_D.bat` | `small_s1` → `x_slope_s1` → `cnx_fastvit_kd_small_s1` → `cnx_fastvit_kd_xslope_s1` → `small_s2` → `x_slope_s2` → `cnx_fastvit_kd_small_s2` → `cnx_fastvit_kd_xslope_s2` |

A queue:
- **Never waits for a key between runs.** If a run fails, the queue logs the failure and starts the next run.
- **Saves the full console output of every run:**
  - finished run: `results\<recipe>_ep150\console_log.txt`
  - failed run: `results\_console\<recipe>.txt`
- **Writes one start/end line per run** to `results\_console\queue_<computer name>.txt`. Check that file in the morning.
- **Turns sleep off at the start and back on at the end.**

To watch a run live, open its log in a second window:
`powershell Get-Content -Wait results\_console\<recipe>.txt`

Each run also writes its training curve to `seq_nn.log`: training loss, each loss term, holdout RMSE and learning rate, every 5 epochs.

Do not click inside a running cmd window. Clicking starts text selection, which **freezes the program** until you press Esc.
A run refuses to start if its results folder already exists. Never delete a finished run; if a run fails, rename its folder to `results\_failed_<recipe>` and tell D.

The commands below are what each queue runs. Use them only if you are running by hand.

### Laptop A
```bat
python seq_NN_holdout_eval.py %COMMON% --cfg-name cnx_fastvit_s1 --output-dir results/cnx_fastvit_s1_ep150
python seq_NN_honest_curve.py results/cnx_fastvit_s1_ep150 > results/cnx_fastvit_s1_ep150/honest_curve.txt
```

### Laptop B
```bat
python seq_NN_holdout_eval.py %COMMON% --cfg-name cnx_fastvit_s2 --output-dir results/cnx_fastvit_s2_ep150
python seq_NN_honest_curve.py results/cnx_fastvit_s2_ep150 > results/cnx_fastvit_s2_ep150/honest_curve.txt
```

### Laptop C (three runs, in this order)
```bat
python seq_NN_holdout_eval.py %COMMON% --cfg-name x_slope_nosynth_s1 --output-dir results/x_slope_nosynth_s1_ep150
python seq_NN_honest_curve.py results/x_slope_nosynth_s1_ep150 > results/x_slope_nosynth_s1_ep150/honest_curve.txt

python seq_NN_holdout_eval.py %COMMON% --cfg-name x_slope_nocost_s1 --output-dir results/x_slope_nocost_s1_ep150
python seq_NN_honest_curve.py results/x_slope_nocost_s1_ep150 > results/x_slope_nocost_s1_ep150/honest_curve.txt

python seq_NN_holdout_eval.py %COMMON% --cfg-name x_slope_nogeo_s1 --output-dir results/x_slope_nogeo_s1_ep150
python seq_NN_honest_curve.py results/x_slope_nogeo_s1_ep150 > results/x_slope_nogeo_s1_ep150/honest_curve.txt
```

### Laptop D
D follows `ablation_study/COMMANDS.md`: blocks 1, 2, 4, 5, 9, 10, 12 and 13 (= D-1 … D-8). It skips blocks 3, 6, 7, 8 and 11, which the other laptops run.
D-3/D-4 must wait for D-1/D-2, and D-7/D-8 for D-5/D-6, because each student loads its teacher. The queue order already guarantees this.

### Before comparing anything: check the protocol (D)

Once runs are on D, run this in `ablation_study\pipeline`:
```bat
python check_protocol.py
```
For every finished run, it shows the one setting its recipe changes, plus every run-time setting (epochs, batch size, ...).
It prints `OK` only if all runs share identical run-time settings. Otherwise it lists the run that differs.
`num_workers` may differ between laptops: it changes speed, not the method.

---

## 5. After each run: send the folder to D

When a run finishes on A, B or C:
1. **Copy the whole folder** `ablation_study\pipeline\results\<recipe>_ep150\` to laptop D's `ablation_study\pipeline\results\` (USB or shared drive).
   It must include `holdout_predictions.pqt` (37 MB), `seq_nn.log`, `holdout_metrics.json` and `honest_curve.txt`. `models.pkl` (205 MB) is optional.
2. **Post in the group:** the recipe name, your laptop, the finish time, and the `late avg` number from `honest_curve.txt`.

**Do not commit or push results from A, B or C.** Only D commits, so `RESULTS_LOG.md` never gets merge conflicts.

---

## 6. What D does with the folders (consolidation)

When both folders of a comparison are on D, D runs its compare step from `COMMANDS.md` and fills the row in `RESULTS_LOG.md`:

| Arriving run | Compare against | COMMANDS.md block |
|---|---|---|
| `cnx_fastvit_s1` (A) | needed by the student compares | 4, 5 |
| `cnx_fastvit_s2` (B) | needed by the student compares | 12, 13 |
| `x_slope_nosynth_s1` (C) | `x_slope_s1` | 6 |
| `x_slope_nocost_s1` (C) | `x_slope_s1` | 7 |
| `x_slope_nogeo_s1` (C) | `x_slope_s1` | 8 |

After the seed-1 rows are complete, D checks C1/D1/D2 for |Δ| < 0.30 ft and assigns any seed-2 runs (section 1).
D commits `RESULTS_LOG.md` after each update. Large files (`*.pqt`, `*.pkl`) stay out of git automatically.

---

## 7. Checklist

- [ ] All four laptops report the same commit hash.
- [ ] A and B: fit test passed (or the run was handed to D).
- [ ] C: setup done, C-1 started.
- [ ] D: D-1 started.
- [ ] Every finished run: folder copied to D, late avg posted in the group.
- [ ] D: every compare done, `RESULTS_LOG.md` complete, seed-2 decision made for C1/D1/D2.
- [ ] Everyone: sleep turned back on.
