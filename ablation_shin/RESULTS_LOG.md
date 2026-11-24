# Results log

Fill in one row as soon as a run (one block of `COMMANDS.md`) finishes, then commit.
Never edit a filled row except to add to its Notes. If a run is redone, add a new row.

**Where each number comes from** (all files are in `pipeline\results\<recipe>_ep150\`):

| Column | Source |
|---|---|
| Reported | `holdout_metrics.json`, key `pooled_rmse` (secondary number: checkpoint picked on the test wells) |
| Last / **Late avg** | `honest_curve.txt`, columns `last` and `late avg`. **Late avg is the main number.** |
| Δ | late avg of this run − late avg of the reference run. Negative means this run is better. |
| k\* / Bootstrap / Verdict | `compare_vs_<reference>.txt`: the lines `k* =`, `bootstrap` and `VERDICT` |

## Per-run log

| # | Recipe | Arm | Seed | Finished (date) | Laptop | Reported | Last | **Late avg** | Reference | Δ late avg | k\* | Bootstrap % | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `small_s1` | A0 | 1 | | D | | | | — | — | — | — | |
| 2 | `x_slope_s1` | A1 | 1 | | D | | | | `small_s1` | | | | |
| 3 | `cnx_fastvit_s1` | B0 | 1 | | D | | | | — | — | — | — | |
| 4 | `cnx_fastvit_kd_small_s1` | B1 | 1 | | D | | | | `cnx_fastvit_s1` | | | | |
| 5a | `cnx_fastvit_kd_xslope_s1` | B2 | 1 | | D | | | | `cnx_fastvit_s1` | | | | |
| 5b | ″ | B2 | 1 | | | | | | `cnx_fastvit_kd_small_s1` | | | | same run as 5a, second comparison |
| 6 | `x_slope_nosynth_s1` | C1 | 1 | | D | | | | `x_slope_s1` | | | | |
| 7 | `x_slope_nocost_s1` | D1 | 1 | | D | | | | `x_slope_s1` | | | | |
| 8 | `x_slope_nogeo_s1` | D2 | 1 | | D | | | | `x_slope_s1` | | | | |
| 9 | `small_s2` | A0 | 2 | | D | | | | — | — | — | — | |
| 10 | `x_slope_s2` | A1 | 2 | | D | | | | `small_s2` | | | | |
| 11 | `cnx_fastvit_s2` | B0 | 2 | | D | | | | — | — | — | — | |
| 12 | `cnx_fastvit_kd_small_s2` | B1 | 2 | | D | | | | `cnx_fastvit_s2` | | | | |
| 13a | `cnx_fastvit_kd_xslope_s2` | B2 | 2 | | D | | | | `cnx_fastvit_s2` | | | | |
| 13b | ″ | B2 | 2 | | | | | | `cnx_fastvit_kd_small_s2` | | | | same run as 13a, second comparison |

Optional seed-2 runs from step 4 (add rows only if they are run):

| # | Recipe | Arm | Seed | Finished (date) | Laptop | Reported | Last | **Late avg** | Reference | Δ late avg | k\* | Bootstrap % | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

## Ablation summary (fill when all seeds of an ablation are done; this is the paper table)

Decision rule from `PLAN.md`: every seed has the same sign, mean |Δ| ≥ 0.10 ft, and every compare shows k\* ≥ 10 and bootstrap ≥ 90%. Otherwise the verdict is "not shown".

| Ablation | Comparison | Δ seed 1 | Δ seed 2 | Mean Δ | Verdict |
|---|---|---|---|---|---|
| A. Slope loss | A1 vs A0 | | | | |
| B. Distillation | B1 vs B0 | | | | |
| B. Distillation | B2 vs B0 | | | | |
| B. Slope teacher | B2 vs B1 | | | | |
| C. Synthetic wells | C1 vs A1 | | | | |
| D. Mismatch channel | D1 vs A1 | | | | |
| D. Geo-prior channel | D2 vs A1 | | | | |
