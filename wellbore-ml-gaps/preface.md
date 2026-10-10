## Summary of findings

Error anatomy we are attacking (60-epoch baseline dump, 5.47 ft): **~51%** of squared error is
wrong / blended repeated band, **~25%** a single hump drifting (one straight line per well),
**~13%** broad humps. Oracle best-mode decode 4.27 ft; per-well line removal 3.30 ft.

### Ruled out (measured on the saved holdout posteriors)

- **Decoding and calibration.** Mean decode 5.480; best temperature (tuned in-sample) 5.478; median
  5.636; dominant mode 6.061; argmax 6.274 (B03). Posterior is calibrated: truth = -0.19 + 0.993 x mean,
  predicted variance 30.5 vs MSE 29.9; cross-validated band-weight recalibration gains 0.017 ft (B09).
  Band confusion can only fall through **more evidence** or **better training**, not re-weighting.
- **BiFPN** decoder: no evidence for full-resolution dense regression; bottom-up path feeds unused
  levels. FLOP savings, if wanted, come from narrowing the three mid up-blocks (~4.07M params) (B12).
- **Learned per-well gates**: improve CV but lose on the leaderboard (8th, 22nd place; our own 5.018 vs 4.905).
- **TVT + Z - ANCC identity**: exact on all 773 wells (std 0.0066 ft) but test files have no formation
  columns, so it is only a change of variables (A18).
- **Mahto 2026**: headline 0.59 ft comes from the train-only formation columns (label leakage) — do not cite (A13).

### Gaps worth attacking, ranked by evidence

| # | Gap | Error mode | Evidence | Cost |
|---|---|---|---|---|
| 1 | Sibling (same master log) and own-prefix GR references as ConvNeXt/FastViT input channels | band | 3rd (-0.126 private), 6th, 22nd place independently; our PF +0.68 ft | medium |
| 2 | Decoy segments in the Simulator (GR from typewell at TVT +- repeat period, labels unchanged) — today the true band is always the best local match | band | 8th place broke a 6.6 plateau | low |
| 3 | Supervised slope loss on S = TVT + Z over MD windows (`dS_penalty` exists, weight 0) | drift | 8th/silver work in u-space; line oracle 4.94 -> 3.07 | low |
| 4 | Second-order (level, dip) forward-backward decode with tempered emissions (p/prior)^tau — the failed decode was first-order and double-counted evidence | drift, band | silver tracker; B05 and B06 independently | CPU only |
| 5 | Switched-off features: CRPS loss, `deep_sup`, stretch/squeeze channels, `pf_v1` heatmaps, `finetune_epochs` real-only stage | mixed | implemented, weight 0 / False / None | config |
| 6 | Learned GR matching embeddings; attention along the level axis; GA-Net-style learned MD aggregation | band | 3rd-place 1D SDF, FoundationStereo, GA-Net | high |

### Repo notes to fix

- `z_shift` apply_prob is **0.75** in the configs; `kaggle_1st_place/CLAUDE.md` says 0.85.
- Decoder "~0.05M params" holds only for the full-resolution parts; mid up-blocks hold ~4.07M.
- US 11480045 is RNA Capital, not Chevron.
- A05's mapping of the 7.35 ft PF to `C_pf_far_ep250` is wrong (the report attributes 7.35 ft to the 1st-place author's PF).
- `pypdf` was installed into `.venv` during research.

### Evidence caveat

Many paywalled sources (OnePetro, Wiley, MDPI) returned 403; affected fields are marked uncertain
and omitted below. Measurements in B03, B09, A18 and B10 were run locally on the 60-epoch baseline
dump and the `0801_V2_ep150` predictions, not on the 4.84 ft FastViT student.
