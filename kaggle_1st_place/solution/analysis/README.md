# analysis/

Scripts from the Sep–Oct 2026 round. Run everything from `kaggle_1st_place/solution`.
Results and verdicts are in `kaggle_1st_place/EXPERIMENTS_2026-10.md`.

| script | what it does | needs |
|---|---|---|
| `run_chain.sh` | Trains several `experiments/bilzard` recipes back to back (150 epochs) and writes an honest-score + k*/bootstrap summary against both control seeds. `bash analysis/run_chain.sh x_slope x_deepsup` | GPU |
| `gen_items.py` | Builds the first training batches (simulation on, fixed seeds, 0 workers) from one copy of the fork. Used to prove new options are bit-identical to the old code when switched off: run it on a `git archive HEAD` copy and on the working tree and compare the `.npz` files. | CPU |
| `probe_dipdecode.py` | Second-order (level + dip) forward-backward decode with tempered emissions over a saved posterior dump. Writes `probe_dipdecode_sse_<dump>.csv`. | a `seq_NN_posterior_dump.py --save-all` dump |
| `probe_robust.py` | Cross-fitted RMSE, k*, bootstrap, wins and median-well error for every `probe_dipdecode_sse_*.csv`. | the CSVs above |
| `probe_sibling.py` | Does a sibling-well GR reference pick the true hump on multimodal columns more often than the dominant hump? | a dump, `kaggle2ndplace/anchor_sibling.py` |
| `probe_sibling_sanity.py` | Checks the GR matcher works and compares typewell vs sibling discrimination on true paths. | same |
| `probe_common.py` | Shared loaders; `PROBE_RUN=results/<dump>` selects the dump (default `results/baseline_check_posterior_all`). | – |

On a 16 GB Windows laptop keep `--num-workers 1` for training and at most 4 processes for the probes.
