#!/usr/bin/env bash
# Train several recipes back to back (150 epochs, ConvNeXt-sized settings for an 8 GB laptop GPU
# with 16 GB RAM), then write an honest-score + robustness summary against both control seeds.
#   bash analysis/run_chain.sh x_slope x_deepsup          (from kaggle_1st_place/solution)
#   PY=path/to/python bash analysis/run_chain.sh ...      (defaults to "python")
# A failed run is logged and skipped. Keep the laptop plugged in with the lid open and
# idle sleep off (powercfg /change standby-timeout-ac 0); sleep and update reboots killed runs.
cd "$(dirname "$0")/.."
PY="${PY:-python}"
COMMON="--device cuda --offline-timm --batch-size 4 --val-batch-size 4 --grad-accum-steps 4 --num-workers 1"
RUNS="$*"
for r in $RUNS; do
  echo "[chain] $(date) $r start"
  "$PY" seq_NN_holdout_eval.py --id 0801_V2 --source-dir experiments/bilzard --cfg-name "$r" --epochs 150 --output-dir "results/${r}_ep150" $COMMON
  echo "[chain] $(date) $r exit=$?"
done
SUM="results/chain_summary_$(date +%Y%m%d_%H%M).txt"
{
  echo "== unpicked scores (picked / last / late avg) =="
  "$PY" seq_NN_honest_curve.py results/0801_V2_ep150 results/0801_V2_ep150_local $(for r in $RUNS; do [ -d "results/${r}_ep150" ] && echo "results/${r}_ep150"; done)
  for r in $RUNS; do
    [ -f "results/${r}_ep150/holdout_predictions.pqt" ] || { echo "== $r: no predictions"; continue; }
    for base in 0801_V2_ep150 0801_V2_ep150_local; do
      echo "== $r vs $base"
      "$PY" seq_NN_robust_compare.py --base "results/$base" --treat "results/${r}_ep150" --output-dir "results/${r}_ep150" 2>&1 | grep -aE "pooled RMSE|k\* =|bootstrap|VERDICT"
    done
  done
} > "$SUM" 2>&1
echo "[chain] $(date) ALL DONE -> $SUM"
