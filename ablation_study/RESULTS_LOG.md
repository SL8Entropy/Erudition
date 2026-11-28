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
| 3 | `cnx_fastvit_s1` | B0 | 1 | 2026-11-26 | A | 5.272 | 5.364 | **5.390** | — | — | — | — | **Protocol deviation:** trained with batch 2 × grad accum 8 (protocol 4 × 4; same 16 samples per step, but FastViT's BatchNorm sees 2-sample batches). Kept by the owner's decision, 2026-11-27. picked ep 135 (raw 5.323); late sd 0.053; flattery 0.067. `check_protocol.py` will flag it once `cfg.pkl` is on D. |
| 4 | `cnx_fastvit_kd_small_s1` | B1 | 1 | 2026-11-27 | D | 4.901 | 4.924 | **4.955** | `cnx_fastvit_s1` | −0.436 | 11 | 90.5 | teacher `small_s1`. picked ep 135 (raw 4.924); late sd 0.037; flattery 0.031. Compare pending: `cnx_fastvit_s1` (laptop A) not on D yet. 2026-11-27: Δ from A's honest curve; reference has a batch deviation (row 3); k\*/bootstrap wait for A's `holdout_predictions.pqt`. 2026-11-28: compare run (A's `.pqt` via WhatsApp): VERDICT ACCEPT, picked-checkpoint delta −0.371, 77/155 wells better. |
| 5a | `cnx_fastvit_kd_xslope_s1` | B2 | 1 | 2026-11-27 | D | 4.803 | 4.866 | **4.916** | `cnx_fastvit_s1` | −0.475 | 117 | 94.3 | teacher `x_slope_s1`. picked ep 120 (raw 4.854); late sd 0.056; flattery 0.061. Compare pending: `cnx_fastvit_s1` (laptop A) not on D yet. 2026-11-27: Δ from A's honest curve; reference has a batch deviation (row 3); k\*/bootstrap wait for A's `holdout_predictions.pqt`. 2026-11-28: compare run: VERDICT ACCEPT, picked-checkpoint delta −0.470, 92/155 wells better. |
| 5b | ″ | B2 | 1 | | | | | | `cnx_fastvit_kd_small_s1` | −0.039 | 65 | 76.9 | same run as 5a, second comparison. Script VERDICT ACCEPT (picked-checkpoint delta −0.099), but bootstrap < 90% and \|Δ\| < 0.10 |
| 6 | `x_slope_nosynth_s1` | C1 | 1 | 2026-11-26 | C | 6.630 | 7.656 | **7.626** | `x_slope_s1` | +2.621 | 139 | 100.0 | picked ep 30 (raw 6.684); late sd 0.063; flattery 0.942. Compare pending: C's `holdout_predictions.pqt` and `cfg.pkl` not on D. 2026-11-28: k\*/bootstrap test the claimed direction (full teacher beats the removal): `compare_full_teacher_vs_x_slope_nosynth_s1.txt`, ACCEPT, picked-checkpoint delta −1.810, teacher better on 107/155 wells. The arm-vs-reference file `compare_vs_x_slope_s1.txt` says REJECT (k\* 0), as expected for a removal. |
| 7 | `x_slope_nocost_s1` | D1 | 1 | 2026-11-26 | C | 4.838 | 5.027 | **5.045** | `x_slope_s1` | +0.040 | 12 | 51.8 | picked ep 80 (raw 4.865); late sd 0.064; flattery 0.180. \|Δ\| < 0.30: needs seed 2 (`x_slope_nocost_s2`). Compare pending: C's `.pqt` not on D. 2026-11-28: full teacher vs removal (`compare_full_teacher_vs_x_slope_nocost_s1.txt`): picked-checkpoint delta −0.018, teacher better on 78/155 wells. |
| 8 | `x_slope_nogeo_s1` | D2 | 1 | 2026-11-26 | C | 8.405 | 8.844 | **8.911** | `x_slope_s1` | +3.906 | 146 | 100.0 | picked ep 85 (raw 8.599); late sd 0.087; flattery 0.312. Compare pending: C's `.pqt` not on D. 2026-11-28: full teacher vs removal (`compare_full_teacher_vs_x_slope_nogeo_s1.txt`): ACCEPT, picked-checkpoint delta −3.584, teacher better on 119/155 wells. |
| 9 | `small_s2` | A0 | 2 | 2026-11-27 | D | 5.173 | 5.199 | **5.597** | — | — | — | — | picked ep 150 (raw 5.199); late sd 0.329 (swings 5.20–6.23); flattery 0.398 |
| 10 | `x_slope_s2` | A1 | 2 | 2026-11-27 | D | 5.145 | 5.359 | **5.419** | `small_s2` | −0.178 | 16 | 56.6 | Script VERDICT ACCEPT (picked-checkpoint delta −0.028), but bootstrap < 90%. picked ep 90 (raw 5.177); late sd 0.081; flattery 0.243 |
| 11 | `cnx_fastvit_s2` | B0 | 2 | 2026-11-26 | B | 5.308 | 5.339 | **5.443** | — | — | — | — | batch 4 × 4 (protocol). picked ep 150 (raw 5.339); late sd 0.097; flattery 0.105. `.pqt` and `cfg.pkl` not on D yet. |
| 12 | `cnx_fastvit_kd_small_s2` | B1 | 2 | 2026-11-27 | D | 5.343 | 5.554 | **5.567** | `cnx_fastvit_s2` | +0.124 | 0 | 42.4 | teacher `small_s2`. picked ep 90 (raw 5.368); late sd 0.084; flattery 0.199. Compare pending: `cnx_fastvit_s2` (laptop B) not on D yet. 2026-11-27: Δ from B's honest curve; k\*/bootstrap wait for B's `holdout_predictions.pqt`. 2026-11-28: compare run: VERDICT REJECT, picked-checkpoint delta +0.035, 79/155 wells better. |
| 13a | `cnx_fastvit_kd_xslope_s2` | B2 | 2 | 2026-11-27 | D | 4.998 | 5.124 | **5.142** | `cnx_fastvit_s2` | −0.301 | 3 | 91.4 | 2026-11-28: compare run: VERDICT REJECT, picked-checkpoint delta −0.310, 79/155 wells better; wells e32da4ec and f2d4c8c9 carry the gain here and in row 13b. 2026-11-27: Δ from B's honest curve; k\*/bootstrap wait for B's `.pqt`. teacher `x_slope_s2`. picked ep 70 (raw 5.022); late sd 0.030; flattery 0.120. Compare pending: `cnx_fastvit_s2` (laptop B) not on D yet. Laptop hibernated 08:44–10:46 (lid closed, charger out) between ep 40 and 45; the run resumed. |
| 13b | ″ | B2 | 2 | | | | | | `cnx_fastvit_kd_small_s2` | −0.425 | 3 | 93.3 | same run as 13a, second comparison. VERDICT REJECT (k\* 3 < 10; picked-checkpoint delta −0.345; 77/155 wells better) |

Optional seed-2 runs from step 4 (add rows only if they are run):

| # | Recipe | Arm | Seed | Finished (date) | Laptop | Reported | Last | **Late avg** | Reference | Δ late avg | k\* | Bootstrap % | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 15 | `x_slope_nocost_s2` | D1 | 2 | | D | | | | `x_slope_s2` | | | | in `queue_D_s3.bat`, last. Test k\*/bootstrap in the claimed direction (full teacher vs removal). |

Seed-3 round (added 2026-11-28 at the supervisor's request; arms A and B only):

| # | Recipe | Arm | Seed | Finished (date) | Laptop | Reported | Last | **Late avg** | Reference | Δ late avg | k\* | Bootstrap % | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 17 | `small_s3` | A0 | 3 | | D | | | | — | — | — | — | |
| 18 | `x_slope_s3` | A1 | 3 | | D | | | | `small_s3` | | | | |
| 19 | `cnx_fastvit_s3` | B0 | 3 | | B | | | | — | — | — | — | |
| 20 | `cnx_fastvit_kd_small_s3` | B1 | 3 | | D | | | | `cnx_fastvit_s3` | | | | |
| 21a | `cnx_fastvit_kd_xslope_s3` | B2 | 3 | | D | | | | `cnx_fastvit_s3` | | | | |
| 21b | ″ | B2 | 3 | | | | | | `cnx_fastvit_kd_small_s3` | | | | same run as 21a, second comparison |

## Ablation summary (fill when all seeds of an ablation are done; this is the paper table)

Decision rule from `PLAN.md`: every seed has the same sign, mean |Δ| ≥ 0.10 ft, and every compare shows k\* ≥ 10 and bootstrap ≥ 90%. Otherwise the verdict is "not shown".

| Ablation | Comparison | Δ seed 1 | Δ seed 2 | Mean Δ | Verdict |
|---|---|---|---|---|---|
| A. Slope loss | A1 vs A0 | −0.020 | −0.178 | −0.099 | not shown (mean \|Δ\| 0.099 < 0.10; bootstrap 40.0% / 56.6% < 90%; k\* 0 on seed 1) |
| B. Distillation | B1 vs B0 | −0.436 | +0.124 | −0.156 | not shown: seeds disagree in sign, so the rule asks for seeds 3 and 4. Seed 1 passes (k\* 11, 90.5%), seed 2 does not (k\* 0, 42.4%). Seed-1 reference ran at batch 2 × 8 (row 3). |
| B. Distillation | B2 vs B0 | −0.475 | −0.301 | −0.388 | not shown: same sign and mean \|Δ\| ≥ 0.10, seed 1 passes (k\* 117, 94.3%), but seed 2 has k\* 3 (bootstrap 91.4%). Seed-1 reference ran at batch 2 × 8 (row 3). |
| B. Slope teacher | B2 vs B1 | −0.039 | −0.425 | −0.232 | not shown (same sign, mean \|Δ\| ≥ 0.10, but k\* 3 on seed 2 and bootstrap 76.9% on seed 1) |
| C. Synthetic wells | C1 vs A1 | +2.621 | — | +2.621 | **shown**: removing synthetic wells hurts. \|Δ\| ≥ 0.30 (one seed is enough), k\* 139, bootstrap 100% |
| D. Mismatch channel | D1 vs A1 | +0.040 | | | not shown yet: \|Δ\| < 0.30, needs seed 2 (`x_slope_nocost_s2`); seed 1 bootstrap 51.8% |
| D. Geo-prior channel | D2 vs A1 | +3.906 | — | +3.906 | **shown**: removing the geo-prior channel hurts. \|Δ\| ≥ 0.30 (one seed is enough), k\* 146, bootstrap 100% |
