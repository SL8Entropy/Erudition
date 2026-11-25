# Commands, in order

This is the master list of all 13 runs, written from laptop D's point of view.
**Who runs which block: `TEAM\_PLAN.md`.**

* Blocks 3, 6, 7, 8 and 11 are trained on laptops A, B and C. D runs only their compare lines, once their folders have been copied to D.
* In blocks 4, 5, 12 and 13, run each compare line once its reference folder (`cnx\_fastvit\_s1` or `cnx\_fastvit\_s2`, from A or B) is on D.

Run in **cmd.exe** (not PowerShell, not bash), one block at a time.
Each training command takes about 2 h on D. Do not start the next block until the current one has finished.
After every block, fill in its row in `RESULTS\_LOG.md`. Only D edits that file.

## 0\. Every time you open a new terminal. This is only if you want to manually enter any terminal commands. all other bat files automatically set this up regardless.

Run open\_terminal.bat, and then enter commands into the ensuing terminal that opens. or alternatively:

Open cmd.exe **in the `ablation\_study` folder** (in File Explorer, click the address bar, type `cmd`, press Enter). All paths below are relative to that folder, so the project works wherever it is copied.

```bat
cd pipeline
call ..\\..\\.venv\\Scripts\\activate.bat
set COMMON=--id 0801\_V2 --source-dir experiments/bilzard --epochs 150 --device cuda --offline-timm --batch-size 4 --val-batch-size 4 --grad-accum-steps 4 --num-workers 1
powercfg /change standby-timeout-ac 0
```

`COMMON` holds the settings shared by every run. If a later command complains about `%COMMON%`, you are in a new window: run block 0 again.
`seq\_NN\_robust\_compare.py` exits with code 1 when its verdict is REJECT. That is normal, not a crash; its output file is still written.
A run refuses to start if its `results\\...` folder already exists. Never delete a finished run to "retry"; rename the failed folder to `results\\\_failed\_<recipe>` first.

## Step 1: the two teachers (everything later depends on these)

### 1\. `small\_s1`: A0, ConvNeXt-small without slope loss, seed 1

The no-slope control, and the teacher for run 4.

```bat
python seq\_NN\_holdout\_eval.py %COMMON% --cfg-name small\_s1 --output-dir results/small\_s1\_ep150
python seq\_NN\_honest\_curve.py results/small\_s1\_ep150 > results/small\_s1\_ep150/honest\_curve.txt
```

### 2\. `x\_slope\_s1`: A1, ConvNeXt-small with slope loss, seed 1

The paper's teacher. Compared with run 1, this is **ablation A**.

```bat
python seq\_NN\_holdout\_eval.py %COMMON% --cfg-name x\_slope\_s1 --output-dir results/x\_slope\_s1\_ep150
python seq\_NN\_honest\_curve.py results/x\_slope\_s1\_ep150 > results/x\_slope\_s1\_ep150/honest\_curve.txt
python seq\_NN\_robust\_compare.py --base results/small\_s1\_ep150 --treat results/x\_slope\_s1\_ep150 --output-dir results/x\_slope\_s1\_ep150 > results/x\_slope\_s1\_ep150/compare\_vs\_small\_s1.txt 2>\&1
```

## Step 2: seed 1 of the other six arms

### 3\. `cnx\_fastvit\_s1`: B0, FastViT with no teacher, seed 1

The reference for the two students.

```bat
python seq\_NN\_holdout\_eval.py %COMMON% --cfg-name cnx\_fastvit\_s1 --output-dir results/cnx\_fastvit\_s1\_ep150
python seq\_NN\_honest\_curve.py results/cnx\_fastvit\_s1\_ep150 > results/cnx\_fastvit\_s1\_ep150/honest\_curve.txt
```

### 4\. `cnx\_fastvit\_kd\_small\_s1`: B1, FastViT taught by run 1, seed 1

Distillation from a teacher without the slope loss. Needs `results\\small\_s1\_ep150\\models.pkl`.

```bat
python seq\_NN\_holdout\_eval.py %COMMON% --cfg-name cnx\_fastvit\_kd\_small\_s1 --output-dir results/cnx\_fastvit\_kd\_small\_s1\_ep150
python seq\_NN\_honest\_curve.py results/cnx\_fastvit\_kd\_small\_s1\_ep150 > results/cnx\_fastvit\_kd\_small\_s1\_ep150/honest\_curve.txt
python seq\_NN\_robust\_compare.py --base results/cnx\_fastvit\_s1\_ep150 --treat results/cnx\_fastvit\_kd\_small\_s1\_ep150 --output-dir results/cnx\_fastvit\_kd\_small\_s1\_ep150 > results/cnx\_fastvit\_kd\_small\_s1\_ep150/compare\_vs\_cnx\_fastvit\_s1.txt 2>\&1
```

### 5\. `cnx\_fastvit\_kd\_xslope\_s1`: B2, FastViT taught by run 2, seed 1 (**the final model**)

Needs `results\\x\_slope\_s1\_ep150\\models.pkl`. It is compared with the student without a teacher (run 3) and with the student of the plain teacher (run 4).

```bat
python seq\_NN\_holdout\_eval.py %COMMON% --cfg-name cnx\_fastvit\_kd\_xslope\_s1 --output-dir results/cnx\_fastvit\_kd\_xslope\_s1\_ep150
python seq\_NN\_honest\_curve.py results/cnx\_fastvit\_kd\_xslope\_s1\_ep150 > results/cnx\_fastvit\_kd\_xslope\_s1\_ep150/honest\_curve.txt
python seq\_NN\_robust\_compare.py --base results/cnx\_fastvit\_s1\_ep150 --treat results/cnx\_fastvit\_kd\_xslope\_s1\_ep150 --output-dir results/cnx\_fastvit\_kd\_xslope\_s1\_ep150 > results/cnx\_fastvit\_kd\_xslope\_s1\_ep150/compare\_vs\_cnx\_fastvit\_s1.txt 2>\&1
python seq\_NN\_robust\_compare.py --base results/cnx\_fastvit\_kd\_small\_s1\_ep150 --treat results/cnx\_fastvit\_kd\_xslope\_s1\_ep150 --output-dir results/cnx\_fastvit\_kd\_xslope\_s1\_ep150 > results/cnx\_fastvit\_kd\_xslope\_s1\_ep150/compare\_vs\_cnx\_fastvit\_kd\_small\_s1.txt 2>\&1
```

### 6\. `x\_slope\_nosynth\_s1`: C1, the teacher without synthetic wells, seed 1

**Ablation C.** Compared with run 2.

```bat
python seq\_NN\_holdout\_eval.py %COMMON% --cfg-name x\_slope\_nosynth\_s1 --output-dir results/x\_slope\_nosynth\_s1\_ep150
python seq\_NN\_honest\_curve.py results/x\_slope\_nosynth\_s1\_ep150 > results/x\_slope\_nosynth\_s1\_ep150/honest\_curve.txt
python seq\_NN\_robust\_compare.py --base results/x\_slope\_s1\_ep150 --treat results/x\_slope\_nosynth\_s1\_ep150 --output-dir results/x\_slope\_nosynth\_s1\_ep150 > results/x\_slope\_nosynth\_s1\_ep150/compare\_vs\_x\_slope\_s1.txt 2>\&1
```

### 7\. `x\_slope\_nocost\_s1`: D1, the teacher without the GR-mismatch channel, seed 1

**Ablation D**, first half. Compared with run 2.

```bat
python seq\_NN\_holdout\_eval.py %COMMON% --cfg-name x\_slope\_nocost\_s1 --output-dir results/x\_slope\_nocost\_s1\_ep150
python seq\_NN\_honest\_curve.py results/x\_slope\_nocost\_s1\_ep150 > results/x\_slope\_nocost\_s1\_ep150/honest\_curve.txt
python seq\_NN\_robust\_compare.py --base results/x\_slope\_s1\_ep150 --treat results/x\_slope\_nocost\_s1\_ep150 --output-dir results/x\_slope\_nocost\_s1\_ep150 > results/x\_slope\_nocost\_s1\_ep150/compare\_vs\_x\_slope\_s1.txt 2>\&1
```

### 8\. `x\_slope\_nogeo\_s1`: D2, the teacher without the geo-prior channel, seed 1

**Ablation D**, second half. Compared with run 2.

```bat
python seq\_NN\_holdout\_eval.py %COMMON% --cfg-name x\_slope\_nogeo\_s1 --output-dir results/x\_slope\_nogeo\_s1\_ep150
python seq\_NN\_honest\_curve.py results/x\_slope\_nogeo\_s1\_ep150 > results/x\_slope\_nogeo\_s1\_ep150/honest\_curve.txt
python seq\_NN\_robust\_compare.py --base results/x\_slope\_s1\_ep150 --treat results/x\_slope\_nogeo\_s1\_ep150 --output-dir results/x\_slope\_nogeo\_s1\_ep150 > results/x\_slope\_nogeo\_s1\_ep150/compare\_vs\_x\_slope\_s1.txt 2>\&1
```

**Checkpoint:** the seed-1 table in `RESULTS\_LOG.md` is now complete. For runs 6–8, if |Δ| < 0.30 ft, add their seed-2 run to step 4.

## Step 3: seed 2 of A and B (always)

These repeat runs 1–5 with seed 2, in the same order and with the same comparisons.

### 9\. `small\_s2`: A0, seed 2

```bat
python seq\_NN\_holdout\_eval.py %COMMON% --cfg-name small\_s2 --output-dir results/small\_s2\_ep150
python seq\_NN\_honest\_curve.py results/small\_s2\_ep150 > results/small\_s2\_ep150/honest\_curve.txt
```

### 10\. `x\_slope\_s2`: A1, seed 2

```bat
python seq\_NN\_holdout\_eval.py %COMMON% --cfg-name x\_slope\_s2 --output-dir results/x\_slope\_s2\_ep150
python seq\_NN\_honest\_curve.py results/x\_slope\_s2\_ep150 > results/x\_slope\_s2\_ep150/honest\_curve.txt
python seq\_NN\_robust\_compare.py --base results/small\_s2\_ep150 --treat results/x\_slope\_s2\_ep150 --output-dir results/x\_slope\_s2\_ep150 > results/x\_slope\_s2\_ep150/compare\_vs\_small\_s2.txt 2>\&1
```

### 11\. `cnx\_fastvit\_s2`: B0, seed 2

```bat
python seq\_NN\_holdout\_eval.py %COMMON% --cfg-name cnx\_fastvit\_s2 --output-dir results/cnx\_fastvit\_s2\_ep150
python seq\_NN\_honest\_curve.py results/cnx\_fastvit\_s2\_ep150 > results/cnx\_fastvit\_s2\_ep150/honest\_curve.txt
```

### 12\. `cnx\_fastvit\_kd\_small\_s2`: B1, seed 2 (teacher = run 9)

```bat
python seq\_NN\_holdout\_eval.py %COMMON% --cfg-name cnx\_fastvit\_kd\_small\_s2 --output-dir results/cnx\_fastvit\_kd\_small\_s2\_ep150
python seq\_NN\_honest\_curve.py results/cnx\_fastvit\_kd\_small\_s2\_ep150 > results/cnx\_fastvit\_kd\_small\_s2\_ep150/honest\_curve.txt
python seq\_NN\_robust\_compare.py --base results/cnx\_fastvit\_s2\_ep150 --treat results/cnx\_fastvit\_kd\_small\_s2\_ep150 --output-dir results/cnx\_fastvit\_kd\_small\_s2\_ep150 > results/cnx\_fastvit\_kd\_small\_s2\_ep150/compare\_vs\_cnx\_fastvit\_s2.txt 2>\&1
```

### 13\. `cnx\_fastvit\_kd\_xslope\_s2`: B2, the final model, seed 2 (teacher = run 10)

```bat
python seq\_NN\_holdout\_eval.py %COMMON% --cfg-name cnx\_fastvit\_kd\_xslope\_s2 --output-dir results/cnx\_fastvit\_kd\_xslope\_s2\_ep150
python seq\_NN\_honest\_curve.py results/cnx\_fastvit\_kd\_xslope\_s2\_ep150 > results/cnx\_fastvit\_kd\_xslope\_s2\_ep150/honest\_curve.txt
python seq\_NN\_robust\_compare.py --base results/cnx\_fastvit\_s2\_ep150 --treat results/cnx\_fastvit\_kd\_xslope\_s2\_ep150 --output-dir results/cnx\_fastvit\_kd\_xslope\_s2\_ep150 > results/cnx\_fastvit\_kd\_xslope\_s2\_ep150/compare\_vs\_cnx\_fastvit\_s2.txt 2>\&1
python seq\_NN\_robust\_compare.py --base results/cnx\_fastvit\_kd\_small\_s2\_ep150 --treat results/cnx\_fastvit\_kd\_xslope\_s2\_ep150 --output-dir results/cnx\_fastvit\_kd\_xslope\_s2\_ep150 > results/cnx\_fastvit\_kd\_xslope\_s2\_ep150/compare\_vs\_cnx\_fastvit\_kd\_small\_s2.txt 2>\&1
```

## Step 4: only if the checkpoint after run 8 asks for it

Same as runs 6–8 with `\_s2`, compared with run 10 (`x\_slope\_s2`). Example for C1:

```bat
python seq\_NN\_holdout\_eval.py %COMMON% --cfg-name x\_slope\_nosynth\_s2 --output-dir results/x\_slope\_nosynth\_s2\_ep150
python seq\_NN\_honest\_curve.py results/x\_slope\_nosynth\_s2\_ep150 > results/x\_slope\_nosynth\_s2\_ep150/honest\_curve.txt
python seq\_NN\_robust\_compare.py --base results/x\_slope\_s2\_ep150 --treat results/x\_slope\_nosynth\_s2\_ep150 --output-dir results/x\_slope\_nosynth\_s2\_ep150 > results/x\_slope\_nosynth\_s2\_ep150/compare\_vs\_x\_slope\_s2.txt 2>\&1
```

For D1 or D2, replace `nosynth` with `nocost` or `nogeo`.

## When everything is done

```bat
powercfg /change standby-timeout-ac 5
```

