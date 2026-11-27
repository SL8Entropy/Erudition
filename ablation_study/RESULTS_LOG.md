# Results log

Laptop D fills in one row as soon as a run finishes (its own, or a folder copied in from A, B or C), then commits. Nobody else edits this file.
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
| 1 | `small_s1` | A0 | 1 | 2026-11-26 | D | 4.778 | 5.002 | **5.025** | — | — | — | — | picked ep 75 (raw 4.825); late sd 0.052; flattery 0.200 |
| 2 | `x_slope_s1` | A1 | 1 | 2026-11-27 | D | 4.820 | 4.909 | **5.006** | `small_s1` | −0.020 | 0 | 40.0 | VERDICT REJECT (picked-checkpoint delta +0.043). picked ep 120 (raw 4.854); late sd 0.195, spikes at ep 105 (5.23) and 110 (5.54); flattery 0.151. Laptop was in standby 20:44–23:32 (charger out) between ep 65 and 70; the run resumed. |
| 3 | `cnx_fastvit_s1` | B0 | 1 | | A | | | | — | — | — | — | |
| 4 | `cnx_fastvit_kd_small_s1` | B1 | 1 | 2026-11-27 | D | 4.901 | 4.924 | **4.955** | `cnx_fastvit_s1` | | | | teacher `small_s1`. picked ep 135 (raw 4.924); late sd 0.037; flattery 0.031. Compare pending: `cnx_fastvit_s1` (laptop A) not on D yet. |
| 5a | `cnx_fastvit_kd_xslope_s1` | B2 | 1 | 2026-11-27 | D | 4.803 | 4.866 | **4.916** | `cnx_fastvit_s1` | | | | teacher `x_slope_s1`. picked ep 120 (raw 4.854); late sd 0.056; flattery 0.061. Compare pending: `cnx_fastvit_s1` (laptop A) not on D yet. |
| 5b | ″ | B2 | 1 | | | | | | `cnx_fastvit_kd_small_s1` | −0.039 | 65 | 76.9 | same run as 5a, second comparison. Script VERDICT ACCEPT (picked-checkpoint delta −0.099), but bootstrap < 90% and \|Δ\| < 0.10 |
| 6 | `x_slope_nosynth_s1` | C1 | 1 | | C | | | | `x_slope_s1` | | | | |
| 7 | `x_slope_nocost_s1` | D1 | 1 | | C | | | | `x_slope_s1` | | | | |
| 8 | `x_slope_nogeo_s1` | D2 | 1 | | C | | | | `x_slope_s1` | | | | |
| 9 | `small_s2` | A0 | 2 | 2026-11-27 | D | 5.173 | 5.199 | **5.597** | — | — | — | — | picked ep 150 (raw 5.199); late sd 0.329 (swings 5.20–6.23); flattery 0.398 |
| 10 | `x_slope_s2` | A1 | 2 | 2026-11-27 | D | 5.145 | 5.359 | **5.419** | `small_s2` | −0.178 | 16 | 56.6 | Script VERDICT ACCEPT (picked-checkpoint delta −0.028), but bootstrap < 90%. picked ep 90 (raw 5.177); late sd 0.081; flattery 0.243 |
| 11 | `cnx_fastvit_s2` | B0 | 2 | | B | | | | — | — | — | — | |
| 12 | `cnx_fastvit_kd_small_s2` | B1 | 2 | 2026-11-27 | D | 5.343 | 5.554 | **5.567** | `cnx_fastvit_s2` | | | | teacher `small_s2`. picked ep 90 (raw 5.368); late sd 0.084; flattery 0.199. Compare pending: `cnx_fastvit_s2` (laptop B) not on D yet. |
| 13a | `cnx_fastvit_kd_xslope_s2` | B2 | 2 | | D | | | | `cnx_fastvit_s2` | | | | |
| 13b | ″ | B2 | 2 | | | | | | `cnx_fastvit_kd_small_s2` | | | | same run as 13a, second comparison |

Optional seed-2 runs from step 4 (add rows only if they are run):

| # | Recipe | Arm | Seed | Finished (date) | Laptop | Reported | Last | **Late avg** | Reference | Δ late avg | k\* | Bootstrap % | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

## Ablation summary (fill when all seeds of an ablation are done; this is the paper table)

Decision rule from `PLAN.md`: every seed has the same sign, mean |Δ| ≥ 0.10 ft, and every compare shows k\* ≥ 10 and bootstrap ≥ 90%. Otherwise the verdict is "not shown".

| Ablation | Comparison | Δ seed 1 | Δ seed 2 | Mean Δ | Verdict |
|---|---|---|---|---|---|
| A. Slope loss | A1 vs A0 | −0.020 | −0.178 | −0.099 | not shown (mean \|Δ\| 0.099 < 0.10; bootstrap 40.0% / 56.6% < 90%; k\* 0 on seed 1) |
| B. Distillation | B1 vs B0 | | | | |
| B. Distillation | B2 vs B0 | | | | |
| B. Slope teacher | B2 vs B1 | | | | |
| C. Synthetic wells | C1 vs A1 | | | | |
| D. Mismatch channel | D1 vs A1 | | | | |
| D. Geo-prior channel | D2 vs A1 | | | | |
