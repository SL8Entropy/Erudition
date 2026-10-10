# Research report: wellbore geology prediction — gaps and additions for Erudition

_Topic:_ Wellbore geology prediction (ROGII geosteering / TVT prediction) and the machine-learning solutions developed for it -- identify gaps in the Erudition implementation and additions that could lower holdout RMSE.

_Items:_ 31 · _generated from_ `results/`

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

## Contents

1. [A01 Manual / commercial geosteering practice (typewell correlation, TVT modelling, ROGII StarSteer)](#a01-manual-commercial-geosteering-practice-typewell-correlation-tvt-modelling-rogii-starsteer) - Targets: Drift | Effort: E0/E1 low
2. [A02 Expert decision-making in geosteering (Geosteering World Cup studies, 349 participants, 2025)](#a02-expert-decision-making-in-geosteering-geosteering-world-cup-studies-349-participants-2025) - Targets: Band confusion | Effort: E1 low
3. [A03 Dynamic Time Warping for well-log correlation (classic + constrained variants)](#a03-dynamic-time-warping-for-well-log-correlation-classic-constrained-variants) - Targets: Band confusion | Effort: Low for A03-2
4. [A04 DP geosteering patents (US 11180981 Chevron; US 11480045 RNA Capital, likely ROGII-linked)](#a04-dp-geosteering-patents-us-11180981-chevron-us-11480045-rna-capital-likely-rogii-linked) - Targets: Drift | Effort: A04-1 medium
5. [A05 Bayesian / particle-filter geosteering (NORCE, Alyaev et al.)](#a05-bayesian-particle-filter-geosteering-norce-alyaev-et-al) - Targets: Drift | Effort: Low for E1
6. [A06 Ensemble Kalman / ensemble-based geosteering and well-log interpretation (DISTINGUISH, 2021+)](#a06-ensemble-kalman-ensemble-based-geosteering-and-well-log-interpretation-distinguish-2021) - Targets: Drift | Effort: Low-medium: ES-MDA is ~50 lines of numpy
7. [A07 Reinforcement-learning geosteering and the NORCE/UiS geosteering robot line (2024-2026)](#a07-reinforcement-learning-geosteering-and-the-norceuis-geosteering-robot-line-2024-2026) - Targets: Drift | Effort: Low for E1
8. [A08 Deep-learning automated geosteering (GAN earth models, CNN/LSTM relative-position predictors, GenAI decision support)](#a08-deep-learning-automated-geosteering-gan-earth-models-cnnlstm-relative-position-predictors-genai-decision-support) - Targets: Band confusion | Effort: Medium for E1: a new head plus loss plus decode, about 150-250 lines
9. [A09 Alyaev et al. multi-realization probabilistic log interpretation (SPWLA 2022) and direct multi-modal inversion (ESS 2022)](#a09-alyaev-et-al-multi-realization-probabilistic-log-interpretation-spwla-2022-and-direct-multi-modal-inversion-ess-2022) - Targets: Band confusion | Effort: Medium
10. [A10 CNN / Siamese well-log depth matching (1D-CNN vs cross-correlation 2022, late-fusion 2024, Siamese SPWLA 2025, deep-Q 2021, SPWLA depth-shifting contest)](#a10-cnn-siamese-well-log-depth-matching-1d-cnn-vs-cross-correlation-2022-late-fusion-2024-siamese-spwla-2025-deep-q-2021-spwla-depth-shifting-contest) - Targets: Band confusion | Effort: Medium
11. [A11 Transformer / foundation models for well logs (WLFM, LithoFormer, TS foundation model for logs, TGS promptable model, CMT-Hiformer)](#a11-transformer-foundation-models-for-well-logs-wlfm-lithoformer-ts-foundation-model-for-logs-tgs-promptable-model-cmt-hiformer) - Targets: Band confusion | Effort: Medium: ~300 lines
12. [A12 Multi-well correlation via relative geologic time and loop consistency (Sylvester 2023, graph/sequence methods)](#a12-multi-well-correlation-via-relative-geologic-time-and-loop-consistency-sylvester-2023-graphsequence-methods) - Targets: Band confusion | Effort: Low for E0
13. [A13 Mahto 2026 "Learning-Driven Autonomy Pipeline for Horizontal Wellbore Geology Prediction" (Research Square)](#a13-mahto-2026-learning-driven-autonomy-pipeline-for-horizontal-wellbore-geology-prediction-research-square) - Targets: None of ours | Effort: Low: the audit is a grep plus a single test
14. [A14 ROGII 3rd place -- five candidate trajectories (HMM, PF, 2 NNs, Adaptive 1D SDF) + sequential SoftMax gate](#a14-rogii-3rd-place----five-candidate-trajectories-hmm-pf-2-nns-adaptive-1d-sdf-sequential-softmax-gate) - Targets: Band confusion | Effort: E1 medium
15. [A15 ROGII silver structural-coordinate tracker (EdidiongA; u = TVT + Z, explicit dip state, banded trellis, windowed GR correlation)](#a15-rogii-silver-structural-coordinate-tracker-edidionga-u-tvt-z-explicit-dip-state-banded-trellis-windowed-gr-correlation) - Targets: Drift | Effort: Low-medium
16. [A16 ROGII 14th place hierarchical U-Net stack with seed averaging (keithtyser)](#a16-rogii-14th-place-hierarchical-u-net-stack-with-seed-averaging-keithtyser) - Targets: Blend/calibration part of band confusion | Effort: E1 low
17. [A17 Other ROGII writeups and repos (8th, 22nd, mycarta dcor toolkit, Lukyanenko blog, remaining top-30)](#a17-other-rogii-writeups-and-repos-8th-22nd-mycarta-dcor-toolkit-lukyanenko-blog-remaining-top-30) - Targets: Band confusion | Effort: E1 low-medium
18. [A18 Kaggle discussion identity TVT + Z - ANCC = const per well (~0.03 ft)](#a18-kaggle-discussion-identity-tvt-z---ancc-const-per-well-003-ft) - Targets: Drift | Effort: E1 low
19. [A19 Public geosteering data beyond Kaggle (GWC 2020 typelog on DataverseNO, other open typelogs)](#a19-public-geosteering-data-beyond-kaggle-gwc-2020-typelog-on-dataverseno-other-open-typelogs) - Targets: Band confusion | Effort: E0 low
20. [B01 Stereo matching cost-volume architectures (GC-Net, PSMNet, GA-Net/SGM-Net, AcfNet, RAFT-Stereo, IGEV/IGEV++, CREStereo)](#b01-stereo-matching-cost-volume-architectures-gc-net-psmnet-ga-netsgm-net-acfnet-raft-stereo-igevigev-crestereo) - Targets: Band confusion | Effort: E0 low
21. [B02 Foundation-prior stereo (FoundationStereo, Fast-FoundationStereo, DEFOM-Stereo, MonSter)](#b02-foundation-prior-stereo-foundationstereo-fast-foundationstereo-defom-stereo-monster) - Targets: Band confusion | Effort: E1 medium: a new prior pipeline plus dataset channels, but the zero-i…
22. [B03 Multimodal stereo decoding and losses (ADL + dominant-modal estimator, SMD-Nets, Stereo Risk, top-k soft-argmax)](#b03-multimodal-stereo-decoding-and-losses-adl-dominant-modal-estimator-smd-nets-stereo-risk-top-k-soft-argmax) - Targets: Band confusion | Effort: E0 done / low
23. [B04 Multi-hypothesis heads (mixture density networks, annealed MCL NeurIPS 2024, resilient MCL)](#b04-multi-hypothesis-heads-mixture-density-networks-annealed-mcl-neurips-2024-resilient-mcl) - Targets: Band confusion | Effort: Medium: head and loss are small, but path parameterisation, masking o…
24. [B05 Differentiable alignment / structured DP losses (Mensch & Blondel, soft-DTW, Drop-DTW, Deep Declarative DTW, neural CRF/HMM)](#b05-differentiable-alignment-structured-dp-losses-mensch-blondel-soft-dtw-drop-dtw-deep-declarative-dtw-neural-crfhmm) - Targets: Band confusion | Effort: Medium: the banded forward-backward and its memory-efficient backward…
25. [B06 Differentiable particle filters (survey 2023, normalizing-flow DPF, learned GMM proposals, semi-supervised DPF)](#b06-differentiable-particle-filters-survey-2023-normalizing-flow-dpf-learned-gmm-proposals-semi-supervised-dpf) - Targets: Drift | Effort: Low for E1
26. [B07 Seismic horizon tracking and deep relative geologic time (Geng 2020, Wu 2020, Bi 2021, transformer RGT 2023, sinusoidal RGT 2026)](#b07-seismic-horizon-tracking-and-deep-relative-geologic-time-geng-2020-wu-2020-bi-2021-transformer-rgt-2023-sinusoidal-rgt-2026) - Targets: Drift | Effort: Low for E1
27. [B08 Retinal-layer segmentation with topology constraints (He et al. 2021 MedIA column-softmax + soft-argmax, graph-search CNNs)](#b08-retinal-layer-segmentation-with-topology-constraints-he-et-al-2021-media-column-softmax-soft-argmax-graph-search-cnns) - Targets: Band confusion in short excursions | Effort: Medium: banded soft forward-backward over 345 x 400 is ~100 lines of…
28. [B09 Calibration and conformal prediction for histogram/classification-as-regression outputs (temperature/isotonic, conformal histogram regression, multimodal conformal, conformal under HMM)](#b09-calibration-and-conformal-prediction-for-histogramclassification-as-regression-outputs-temperatureisotonic-conformal-histogram-regression-multimodal-conformal-conformal-under-hmm) - Targets: Measurement/protocol and blend quality | Effort: Low: numpy-only post-hoc analysis
29. [B10 Test-time adaptation / per-well self-supervision on the known prefix (TTL for time series, PETSA, closed-form offset/slope correction)](#b10-test-time-adaptation-per-well-self-supervision-on-the-known-prefix-ttl-for-time-series-petsa-closed-form-offsetslope-correction) - Targets: Drift | Effort: E0 low
30. [B11 Synthetic data, domain randomisation and sim-to-real alignment (hard repeated-layer generation, pretrain-to-alignment 2026)](#b11-synthetic-data-domain-randomisation-and-sim-to-real-alignment-hard-repeated-layer-generation-pretrain-to-alignment-2026) - Targets: Band confusion | Effort: E1 low-medium
31. [B12 Efficient multi-scale decoders (BiFPN/EfficientDet, FPN variants) for dense full-resolution prediction](#b12-efficient-multi-scale-decoders-bifpnefficientdet-fpn-variants-for-dense-full-resolution-prediction) - Targets: None of the three main modes directly | Effort: Low-medium

---

## A01 Manual / commercial geosteering practice (typewell correlation, TVT modelling, ROGII StarSteer)

### Basics

| Field | Value |
|---|---|
| Name | Manual / commercial geosteering practice: typewell (type-log) correlation, TVT modelling, ROGII StarSteer, Landmark and H&P segment/heat-map correlation |
| Authors org | Industry practice; ROGII (StarSteer, AutoSteer, multi-typewell with variable thickness in 2025.2); Landmark Graphics (Lagenwalter, Poland, Daffin; patent US20130238306A1 / US10318662B2); Helmerich & Payne (M. Willerth; automated heat-map geosteering, IADC/SPE 2022; Bayesian-Viterbi method, SPE/IADC Drilling Conference 2023); SES 'Geosteering Mechanics' practitioner notes (makinhole.com) |
| Year | Practice codified 2010-2025 (patent filed 2010, granted 2019; H&P 2022 and 2023; StarSteer 2025.2 multi-typewell webinar Oct 2025) |
| Venue link | https://rogii.com/products/starsteer ; https://patents.google.com/patent/US20130238306 ; https://drillingcontractor.org/automated-geosteering-algorithm-shows-promising-ability-to-match-human-geological-interpretations-62826 ; https://onepetro.org/SPEDC/proceedings-abstract/23DC/23DC/D031S025R001/516827 ; https://makinhole.com/SES_Geosteering.htm ; https://www.youtube.com/watch?v=OV4ftFHUqHM |
| Code availability | None. StarSteer / AutoSteer and H&P tools are proprietary; the patent describes the workflow only. |
| Evidence level | Mixed: granted patent text (read via Google Patents), trade-press article (Drilling Contractor), practitioner web notes, vendor pages; the SPE 2023 Bayesian-Viterbi paper is known from abstract/search snippets only (full text paywalled, HTTP 403). |
| Extreme summary | Experts convert lateral GR to TVT by assuming a piecewise-planar formation (dip segments plus fault throws plus stretch/squeeze) and fitting it until lateral GR matches the typewell; repeated-layer ties are broken by continuity from an unambiguous landing, preference for few segments, peak/valley sequence-and-spacing matching, regional dip from offsets, and (hardware permitting) azimuthal up/down gamma. |

### Problem formulation

| Field | Value |
|---|---|
| Task definition | Estimate the bit's stratigraphic position (TVT relative to a typewell / target top) along the lateral in real time, i.e. a mapping MD -> TVT, from lateral GR and trajectory, in order to steer the well in zone. |
| Inputs | Lateral LWD gamma ray (sometimes resistivity, azimuthal/imaging gamma, continuous inclination); survey trajectory (MD, inclination, azimuth, TVD/Z); one or several typewells / offset vertical pilots; regional 3D structure grids or seismic surfaces; the landing (build) section where the well crosses stratigraphy steeply and correlation is least ambiguous. |
| Target parameterisation | Explicit structural model: a sequence of segments, each with an apparent dip, optional fault throw at segment boundaries and optional thickness scaling. TVT at each MD follows from TVD minus the modelled surface depth, i.e. the structural-coordinate form (u = TVT + Z) with a dip state. The typewell is projected parallel to the modelled surfaces ('TVT type log'); the inflection point where wellbore inclination equals apparent dip marks a TVT extremum. |
| Output representation | Classic manual practice: a single best interpretation (polyline of dip segments + faults). Commercial automation adds multiple scenario interpretations per lateral (StarSteer: several interpretations from multiple offset wells), probability heat maps over stratigraphic depth vs MD (H&P 2022), and a Viterbi best path over a Bayesian state space with complexity-weighted likelihood (H&P 2023). |

### Method

| Field | Value |
|---|---|
| Architecture | Human-in-the-loop fitting: (1) establish a 3D interpreted surface / regional dip; (2) initialise a straight-line 2D model; (3) compute predicted lateral GR from the typewell using the vertical distance the wellbore penetrates the modelled formation at each sample; (4) segment by segment, adjust dip and fault sliders until predicted and actual GR segments match (optional cross-correlation QC curve); (5) place tie lines/markers where the curves diverge and open a new segment; (6) push the 2D model back to 3D surface hardpoints. Automated variants replace step 4 with misfit heat maps plus path search (H&P), Viterbi on a Bayesian state space (H&P 2023), a particle filter plus DTW against one or more offsets (Geosciences 2024), or a proprietary ML AutoSteer trained on thousands of wells (ROGII). |
| Core mechanism | A strong structural prior: beds are locally planar, so TVT(MD) is piecewise-linear along hole with few breakpoints, and the lateral GR must reproduce the typewell GR under this geometric warp. Repeated-layer ambiguity is resolved by (a) continuity from an unambiguous start (landing/heel, 'one wrong pick and subsequent picks are often all wrong'), (b) a parsimony/complexity preference for fewer dip changes and faults, (c) matching the sequence and spacing of peaks and valleys rather than a single feature's amplitude, (d) stretch/squeeze diagnosis: features stretched along MD mean near-parallel drilling, compressed means crossing, which reveals dip relative to inclination, (e) regional dip from offsets and 3D grids as a prior, (f) azimuthal gamma high-side vs low-side to tell up-section from down-section exits, (g) multiple typewells / variable thickness for lateral facies and thickness change. |
| Training objective | Not learned. Manual: visual GR misfit minimisation with parsimony judgement. Automated: GR misfit (heat-map probability) plus geometric/complexity prior; Bayesian likelihood with a complexity-related correctness prior (H&P 2023). |
| Decoding | Manual: a single most-plausible path chosen by the geosteerer. H&P 2023: Viterbi best path over the full stratigraphic state space. H&P 2022: path through the heat map plus human control points. All mode-seeking, path-level. |
| Metric consistent decode | Mode-seeking (single interpretation / Viterbi). Not Bayes-optimal for pooled RMSE; practice optimises in-zone footage, where committing to one band is rational, whereas pooled RMSE favours the posterior mean. |
| Evidence scope | Whole lateral from landing to current bit, plus typewell(s), offset wells and regional structure (multi-well context). |
| Prefix conditioning | Central: interpretation is sequential and anchored at the already-interpreted section (landing curve and earlier lateral); dip is calibrated during landing (SES 'Block' parameters) and new segments extend the existing polyline. |
| Propose vs select | Propose-then-select: several candidate interpretations (different typewells, fault hypotheses) are built and compared on GR misfit and geological plausibility; StarSteer supports multiple interpretations per lateral. |
| Physical constraints | Layer ordering from the typewell; locally planar beds (piecewise-constant apparent dip); faults as discrete vertical throws at segment boundaries; thickness scaling (stretch/squeeze, variable-thickness multi-typewell); TVT extremum where inclination equals apparent dip; consistency with 3D surfaces and neighbour wells. |

### Data

| Field | Value |
|---|---|
| Dataset | Not applicable (practice). H&P Bayesian method validated on 1,440 synthetic laterals (layer cake + realistically corrupted gamma); H&P 2022 field test on two Haynesville wells (Sabine Oil & Gas). |
| Synthetic to real | Humans work on real data only; alignment is by projecting the typewell parallel to modelled surfaces. H&P synthetic benchmark used geologically realistic layer cakes with corrupted gamma; no quantitative synthetic-to-real alignment stage reported. |

### Results

| Field | Value |
|---|---|
| Metric and performance | No RMSE-type metric published. H&P 2022: automated interpretation matched human geosteerers on landing point, modelled structure and footage in zone for two wells; control-point interventions reduced to 0 and 2 after refinement. H&P 2023 Bayesian-Viterbi: comparable to their earlier automated method on 59% of 1,440 synthetic laterals, significantly better on 34%, 30x faster. Human context (see A02): in GWC 2021 only 14 of 329 participants scored >= 60%. |
| Validation protocol | Not applicable for manual practice; H&P used synthetic trials and a two-well field test; no spatially grouped CV reported. |

### Ambiguity handling

| Field | Value |
|---|---|
| Repeated layers | Handled by context rather than local match: continuity from the landing, multi-feature sequence and spacing matching, rejecting interpretations that need implausibly many dip changes or faults, and regional dip priors. Azimuthal gamma (up vs down) is the hardware remedy but is absent from the Kaggle data. Practitioners stress that one wrong pick propagates, i.e. errors are band-level and persistent. |
| Drift | Handled explicitly through the dip state: a slow systematic TVT drift is exactly a small apparent-dip error, diagnosed via stretch/squeeze of GR features and fixed by re-dipping the segment, i.e. one parameter per segment rather than per point. |
| Uncertainty and calibration | Traditional practice gives no uncertainty. Commercial automation offers heat maps / probabilities over stratigraphic depth and multi-scenario sets, but no published calibration (reliability/ECE) analysis. |

### Relevance to erudition

| Field | Value |
|---|---|
| Where to use | (1) Output regulariser / post-hoc structure: a piecewise-linear-dip (few breakpoints) prior, either as a per-well refit of the expectation curve or as a training loss on the soft-argmax curve. (2) Input channels: typewell stretch/squeeze variants (already stubbed as tw_gr_m0p25 / tw_gr_p0p25 and gr_abs_diff_tw_m0p25 / gr_abs_diff_tw_p0p25 in experiments/arch/seq_NN_cfg.py, currently commented out). (3) Drift correction: per-well dip (slope) + offset re-estimation anchored on the known heel prefix. (4) Candidate generation: DP/Viterbi with a segment-count penalty as a propose-and-select candidate for a gate or averaging. |
| Error mode targeted | Drift (~25%) primarily, via an explicit dip state; band confusion (~51%) secondarily, via continuity plus parsimony priors. |
| Gap vs our implementation | Already present: whole-lateral evidence (345 MD columns in one image), typewell-vs-lateral GR cost channel (gr_abs_diff), prefix conditioning (seen_tvt_rel, tw_seen_tvt_abs_diff), trajectory via z_diff, and the 2nd-place AnchorCNN+DP, which is a path-level method close to automated practice. Missing relative to expert practice: (a) no explicit dip/segment state or parsimony prior; the U-Net outputs an independent distribution per column and the expectation curve is not encouraged to be piecewise-linear, so a single-slope drift is not corrected as one parameter; (b) typewell stretch/squeeze (thickness variation) is not in the active channel set; (c) no scenario-level output with plausibility scores, band choice is implicit in per-column mass; (d) no explicit apparent-dip / inflection-point cue (inclination vs candidate dip); (e) experts use neighbours for regional dip and typewell selection rather than for picking humps, which may explain why our XY geo-prior hump picking failed. Note the post-hoc forward-backward continuity decode already failed, so continuity alone is insufficient; the specifically expert ingredient not yet tried is the low-complexity (few dip segments) constraint. |
| Proposed experiment | E0 (diagnostic, CPU): fit continuous piecewise-linear curves with a breakpoint penalty to the training-well TVT labels to measure how piecewise-linear the labels are (segments per 1000 ft, residual RMS); this sets K/lambda for E1/E3 and tests the label-provenance hypothesis. E1 (post-hoc, CPU): new script posterior_segment_refit.py next to posterior_fb_decode.py; on posterior dumps (results/baseline_check_posterior_all), fit TVT(MD) as continuous piecewise-linear with a per-breakpoint penalty to the expectation curve, weighted by inverse posterior variance and anchored to the prefix; output alpha*refit + (1-alpha)*mean with alpha in {0, 0.25, 0.5, 1}; control = raw expectation on the same 155 holdout wells; report pooled RMSE and the drift/band/hump decomposition from posterior_analysis.py. E2 (training, low effort): re-enable tw_gr_m0p25, tw_gr_p0p25, gr_abs_diff_tw_m0p25, gr_abs_diff_tw_p0p25 in unet_static_channels (experiments/arch/seq_NN_cfg.py); 60-epoch ConvNeXt vs baseline with 2 seeds each because of the +-0.39 ft noise floor. E3 (training, medium): add an L1 penalty on the second difference of the soft-argmax TVT curve along MD (total variation of dip, which favours piecewise-linear and tolerates rare jumps) with small weight; 60-epoch comparison. E4 (medium-high): DP over (TVT, dip) states with emissions from the U-Net column posterior and a breakpoint penalty, used only as a gate/average candidate, not as a replacement for the mean decode. |
| Implementation effort | E0/E1 low (numpy/scipy on existing dumps, about a day); E2 low (config toggle, feature code exists); E3 medium (loss term plus weight tuning); E4 medium-high (2D-state DP; must differ materially from the failed continuity decode). |
| Distillation compatibility | E2/E3 change teacher inputs/loss and transfer through KD on posteriors (the student needs the extra channels for E2). E1/E4 are post-hoc on any posterior and apply equally to the FastViT student. |
| Error correlation with ours | Segment refit or dip-state DP uses our posterior as emission, so band-choice errors remain highly correlated; diversity is limited to drift-shaped errors. Low value as an independent blend member, moderate as a structured corrector. |
| Risk | The network may already produce near-piecewise-linear expectation curves, leaving little headroom. A hard refit can snap a hedged curve onto one band and raise RMSE, like other mode-seeking decodes. Stretch/squeeze channels were commented out upstream, possibly because they did not help. Faults (vertical throws) violate smoothness priors unless jumps are allowed. Gains may sit inside the +-0.39 ft noise floor. Azimuthal gamma, the strongest human cue for up/down ambiguity, is unavailable in the data. |

### Other info

- **Id**: A01
- **Uncertain fields (omitted above)**:
  - comparability
  - gpu_cost
  - inference_budget
  - notes

---

## A02 Expert decision-making in geosteering (Geosteering World Cup studies, 349 participants, 2025)

### Basics

| Field | Value |
|---|---|
| Name | Analyzing expert decision-making in geosteering: Statistical insights from a large-scale controlled experiment (Geosteering World Cup 2021, 349 participants) and related NORCE/UiS human-vs-machine studies |
| Authors org | Yasaman Cheraghi, Sergey Alyaev, Reidar B. Bratvold, Aojie Hong, Igor Kuvaev, Stephen Clark, Andrei Zhuravlev (University of Stavanger, NORCE Norwegian Research Centre, ROGII) |
| Year | 2025 (journal); preceded by URTeC 2022 paper 'What Can We Learn After 10,000 Geosteering Decisions?' (URTeC-2022-3722510); dataset released 2025 |
| Venue link | Applied Computing and Geosciences 2025: https://www.sciencedirect.com/science/article/pii/S2590197425000199 ; SSRN preprint: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4951357 ; URTeC 2022: https://onepetro.org/URTECONF/proceedings-abstract/22URTC/22URTC/D021S022R002/489183 ; JPT summary: https://jpt.spe.org/a-study-suggests-geosteers-often-are-missing-the-target ; Related robot benchmark (SPE J. 2025, doi 10.2118/218444-PA): https://onepetro.org/SJ/article/30/03/995/631101/Geosteering-Robot-Powered-by-Multiple |
| Code availability | Data public under CC-BY 4.0: Zenodo https://zenodo.org/records/15190744 (262 MB; JSON full player data following schema.json / schema_answer.json, plus CSV of interpolated interpretations per participant per step) and GitHub https://github.com/geosteering-no/10000-geosteering-interpretations-and-decisions (release 1.0). No analysis notebooks listed in the README. |
| Evidence level | Peer-reviewed (Applied Computing and Geosciences 2025) plus conference paper and trade-press summary; full text not accessible (HTTP 403), so findings are taken from abstracts, the Zenodo/GitHub records and the JPT summary. |
| Extreme summary | About 10,000 time-stamped interpretations and steering decisions from 349 experts in a controlled GWC 2021 simulation show large, inconsistent variation among experts using identical tools and data; interpretation quality is the main driver of good placement, misinterpretation (especially missed fault-induced shifts) is very costly, and only a small minority score well, so outcomes mix skill and luck. |

### Problem formulation

| Field | Value |
|---|---|
| Task definition | Humans interpret streaming LWD logs against a typewell using stratigraphy-based steering (SBS) in a ROGII cloud simulator, and every 2 minutes (per drilled segment) update their interpretation and choose the next bit target; scored on in-reservoir contact (target-zone footage), reservoir quality and drilling speed (JPT: zone contact 75%, ROP 25%). |
| Inputs | Real-time lateral log (gamma ray) and trajectory, typewell / offset log, SBS interpretation tools (dip segments, faults) in a StarSteer-like cloud environment; true geomodels based on real fields with varying formation dips and faults; two wells (one conventional, one unconventional). |
| Target parameterisation | Human interpretations as SBS dip-segment models (sequence of segments with dips and fault throws) snapshotted over time; decisions as target positions / trajectory. |
| Output representation | Per-participant, per-step interpretation snapshots (interpolated to common steps in CSV) and trajectories; outcome scores per participant. |

### Method

| Field | Value |
|---|---|
| Architecture | Not an ML method. Controlled experiment (competition) plus statistical analysis of correlation and causation between interpretation quality, decisions and well-placement scores; related work builds algorithmic baselines (particle filter + RL robot 'PluRaListic', earlier DSS-1) benchmarked against these humans. |
| Core mechanism | Using a single common synthetic truth with many independent human attempts makes it possible to separate interpretation skill, decision skill and luck. Findings reported: (1) interpretation quality is crucial and misinterpretation has a significant negative impact on placement; (2) faults cause large scoring drops because many participants missed fault-induced zone shifts; (3) a large fraction of participants passively followed the plan (177 excluded; about 24% actively geosteered), and active continuous steering helped; (4) the same individuals performed inconsistently across the two wells, and regional (North American) experience was not predictive; (5) interpretation errors did not always map to lower reservoir contact, i.e. decision quality and outcome are not linearly related under uncertainty; (6) authors attribute variability to the lack of unified guidelines for SBS decisions. |
| Training objective | Not applicable (statistical study). |
| Decoding | Humans commit to one interpretation per step (mode-seeking); no probabilistic hedging in SBS tools. |
| Metric consistent decode | Humans are mode-seeking: they pick one band and steer to it. That is appropriate for an in-zone objective, not for pooled RMSE, where the posterior mean is optimal for a calibrated posterior. |
| Evidence scope | Whole lateral drilled so far plus typewell; sequential (causal) evidence only, no look-ahead, unlike our offline whole-lateral setting. |
| Prefix conditioning | Inherent: every new interpretation extends the participant's previous interpretation; errors made early persist (supports the 'one wrong pick propagates' practitioner view in A01). |
| Propose vs select | Single human interpretation per step; the related PF+RL robot keeps hundreds of plausible interpretations (particles) and optimises decisions over them. |
| Physical constraints | SBS model: piecewise dips and fault throws relative to the typewell stratigraphy. |

### Data

| Field | Value |
|---|---|
| Dataset | GWC 2021: 349 participants (JPT cites 329 scored), about 10,000 time-stamped interpretation+decision snapshots, two synthetic wells based on real fields with dips and faults, 2-minute decision cadence. CC-BY 4.0 on Zenodo/GitHub. |
| Synthetic to real | Logs are synthetic, generated from truth geomodels built on real fields; no alignment stage relevant to ML training. The dataset could be used to build realistic hard cases (faults, dip changes) for our synthetic generator, and gives human interpretation traces over known truth. |

### Results

| Field | Value |
|---|---|
| Metric and performance | JPT/URTeC summary: 50% of participants scored 30% or less; only 14 of 329 (4.3%) scored >= 60%. Related benchmarks: PF+RL robot (SPE J. 2025) median 77.3% reservoir contact over 1,000 runs, top 14% of GWC humans, best run above all humans; earlier study: DSS-1 algorithm beat 28 of 29 experts (arXiv 2005.08916, later withdrawn by authors); JPT cites a study where only 2 of 54 professionals outperformed uncertainty-based algorithms. |
| Validation protocol | Controlled experiment with common truth; statistical correlation/causation analysis. Details of tests not verified (full text inaccessible). |
| Inference budget | Humans: 2-minute decision interval under time pressure; not relevant to our compute budget. |

### Ambiguity handling

| Field | Value |
|---|---|
| Repeated layers | Main human failure is misinterpretation of geological features, notably missing fault-induced shifts, which in TVT space look like jumping to a different repeated band. Humans lack probabilistic hedging, so a wrong band choice becomes a persistent error. No specific tie-breaking protocol was found; the authors emphasise that no unified guideline exists. |
| Uncertainty and calibration | Human interpretations are single-valued, without uncertainty; the paper's point is that decisions should account for uncertainty. The related robot uses particle-filter multi-interpretation (hundreds of hypotheses) and outperforms most humans, supporting calibrated multi-hypothesis outputs. |

### Relevance to erudition

| Field | Value |
|---|---|
| Where to use | (1) Label-noise prior: if Kaggle TVT labels come from human interpretations, there is an irreducible label noise and systematic fault/band picking bias; use this to set expectations and to consider robust losses near faults. (2) Synthetic data: human trace data and fault-heavy truth models can inform harder synthetic cases (faults as abrupt band shifts). (3) Evaluation: split holdout RMSE by presence of fault-like jumps in the label to see whether our band-confusion errors concentrate there. (4) Justification for a calibrated multi-hypothesis posterior over single-path decoding. |
| Error mode targeted | Band confusion (~51%), especially the fault-induced subset; secondarily measurement/protocol (label noise from human interpretation). |
| Proposed experiment | E1 (analysis, CPU, low): in posterior_analysis.py, detect fault-like jumps in holdout labels (/dTVT/dMD - local dip trend/ above a threshold, or piecewise-linear fit residual jumps) and report the share of squared error within +-N columns after jumps vs elsewhere, for the 155 holdout wells. If band-confusion error concentrates after jumps, prioritise fault-aware synthetic data. E2 (data, medium): add fault throws (random vertical offsets at random MD, magnitude matched to typical layer thickness, i.e. a jump to an adjacent repeated band) to the synthetic well generator used for the 85% synthetic portion; 60-epoch ConvNeXt with and without, 2 seeds, compare pooled RMSE and the wrong_mode component. E3 (label-noise estimate, low): on training wells, measure disagreement between label TVT and the best piecewise-linear refit, and check whether GWC human interpretation spread (from the Zenodo CSVs, where truth is known) gives a realistic human error scale; use this as the floor for expectations. |
| Implementation effort | E1 low (analysis on existing dumps); E2 medium (generator change + retrain); E3 low-medium (download 262 MB CC-BY data, parse JSON/CSV). |
| Distillation compatibility | E2 changes teacher training data only; KD on posteriors to FastViT unaffected. E1/E3 are analysis. |
| Error correlation with ours | Not a model; no blend member. Human interpretations are a different error source (causal, single-path) and are not available at test time. |
| Risk | The GWC study measures steering outcomes, not TVT accuracy, and its findings about humans may not transfer to an offline, whole-lateral ML setting. GWC wells are synthetic from two field-based geomodels, so statistics may not represent the Kaggle distribution. Full text was not accessible; specific statistical effects (e.g. effect sizes for interpretation quality, skill vs luck split) are unverified. Adding faults to synthetic data can hurt if Kaggle holdout wells rarely contain faults. |

### Other info

- **Id**: A02
- **Uncertain fields (omitted above)**:
  - comparability
  - drift
  - gap_vs_our_implementation
  - gpu_cost
  - notes

---

## A03 Dynamic Time Warping for well-log correlation (classic + constrained variants)

### Basics

| Field | Value |
|---|---|
| Name | Dynamic Time Warping (DTW) for well-log correlation: classic DTW and constrained variants (Sakoe-Chiba band, slope/stretch limits, stratigraphic tie constraints, local/subsequence DTW) |
| Authors org | Lineage: Sakoe & Chiba (1978, speech); Zoraster et al. (SPE 90471, 2004, first DP curve alignment for well-to-well logs); Hladil et al. (2010, DTW for log correlation, cited by Chevron US11180981); Sylvester (Basin Research 2023, DTW + relative geologic time multi-well); Samant et al. (Paleoceanography and Paleoclimatology 2025, stratigraphic constraints in DTW); Align app (GSA Today, Feb 2024). In-competition: ROGII 3rd-place Local-DTW HMM ensemble member. |
| Year | 1978 (algorithm); 2004 (first SPE well-log DP alignment); 2023-2025 (constrained and multi-well variants); 2025/2026 (ROGII 3rd-place local DTW) |
| Venue link | SPE ATCE 2004 https://onepetro.org/SPEATCE/proceedings/04ATCE/All-04ATCE/SPE-90471-MS/71708 ; Sylvester 2023 https://onlinelibrary.wiley.com/doi/full/10.1111/bre.12787 ; Samant 2025 https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024PA005082 ; Align (GSA Today 2024) https://www.geosociety.org/gsa-today/february-2024/align-a-userfriendly-app-for-numerical-stratigraphic-correlation ; Semi-automatic DTW log correlation (2023) https://www.researchgate.net/publication/374950910 ; ROGII 3rd-place writeup (local copy Erudition/3rd.txt) |
| Evidence level | Peer-reviewed for classic/constrained DTW on vertical logs and sedimentary sections; competition writeup (ROGII 3rd place) for local DTW on this exact task; search-snippet only for the 2025 Samant details (full text returned 403). |
| Extreme summary | DTW finds the minimum-cost monotone stretch/squeeze alignment between two logs by dynamic programming, with bands, slope limits and tie points keeping it geologically plausible. Plain DTW assumes a monotone depth-to-depth mapping, which breaks on laterals that go up and down through strata, so on our task it only works locally or after being recast as a lateral-sample to typewell-level DP (the Chevron patent and 2nd-place designs). |

### Problem formulation

| Field | Value |
|---|---|
| Task definition | Classic: given two vertical logs (reference/typewell and target), find a monotone warping path mapping each depth in one to a depth in the other, giving correlated tops. Geosteering adaptation: map each lateral MD sample to a typewell TVT position; plain DTW is valid only on stretches where the bit moves steadily up- or down-section. |
| Inputs | GR (or other logs) of both wells; optional tie points/tops (user picks, cf. US11906683 'DTW of signals plus user picks'), sedimentation-rate bounds, age constraints (Samant 2025). For laterals: lateral GR vs typewell GR, optionally trajectory Z. |
| Target parameterisation | Warping path (index pairs i to j), i.e. a monotone depth-shift function; for a monotone lateral segment, a TVT(MD) curve. |
| Output representation | Single optimal path (argmin); soft-DTW gives a smoothed expected alignment matrix. No native calibrated uncertainty. |

### Method

| Field | Value |
|---|---|
| Architecture | Non-learned DP over a cost matrix C[i,j] = /a(i) - b(j)/ after normalization (z-score or histogram matching). Accumulated cost D[i,j] = C[i,j] + min(D[i-1,j], D[i,j-1], D[i-1,j-1]) with step patterns/weights, then backtrack. Variants: Sakoe-Chiba band /i-j/ <= w; Itakura parallelogram; slope constraints (max consecutive horizontal/vertical moves = max stretch ratio); open-begin/open-end subsequence DTW (match a short lateral window anywhere in the typewell); derivative DTW; multi-log vector costs; amercing penalty per warping step (Herrmann & Webb 2021); soft-DTW (Cuturi & Blondel 2017) for differentiable use. |
| Core mechanism | Global optimal alignment under monotonicity: repeated/self-similar beds are disambiguated by ordering (bed A must precede bed B) plus stretch limits. On laterals this ordering constraint disappears because the bit can re-enter the same bed, which is exactly why band confusion is the dominant error in our task. |
| Training objective | None (classical). Differentiable relatives (soft-DTW, Drop-DTW) can serve as structured training losses; see item B05. |
| Decoding | Min-cost path via backtracking (Viterbi-like hard argmin). Soft-DTW gives a temperature-controlled expected alignment. |
| Metric consistent decode | Mode-seeking (argmin path), not Bayes-optimal for RMSE. The 2nd-place writeup and our own continuity decode (posterior_fb_decode.py) both found mode/continuity paths worse than the expectation. |
| Evidence scope | Whole log pair (global DTW) or a local window (subsequence/local DTW). |
| Prefix conditioning | Via fixed start point / tie points: the last known heel TVT can anchor an open-end DTW. |
| Propose vs select | Single output per run; candidates can come from several band widths/windows or k-best paths. Usable as an ensemble member: 3rd place used a Local-DTW HMM at weight 0.20 purely for error diversity. |
| Physical constraints | Monotonicity (layer ordering), bounded stretch/squeeze (thickness-ratio limits), optional tie points and rate bounds. Monotonicity is correct vertically but wrong for undulating laterals. |

### Data

| Field | Value |
|---|---|
| Dataset | Classic papers: pairs/sets of vertical wells (Zoraster 2004 field logs; Sylvester 2023 multi-well sets; Samant 2025 sediment cores/proxy records). ROGII 3rd place: Kaggle ROGII lateral + typewell data. |
| Synthetic to real | Not applicable (no training). Normalization between logs is critical: std normalization (as in US11180981) or percentile/histogram matching. |

### Results

| Field | Value |
|---|---|
| Metric and performance | Classic papers report qualitative correlation quality or tie errors, no lateral RMSE. ROGII 3rd place: adding the Local-DTW HMM family improved HMM-mixture OOF RMSE 6.0492 -> 5.9703 and combined HMMPF 5.6648 -> 5.6155, though it was not the strongest standalone model. |
| Comparability | Only the 3rd-place number is on comparable data, and it is OOF CV inside an HMM/PF pipeline at about 5.6 ft, not our 155-well holdout. Our single model is 4.84 ft; the DTW-driven gain (~0.05 ft) is far below our +-0.39 ft single-run noise floor. |
| Validation protocol | Classic DTW: unsupervised, no held-out protocol. 3rd place: group K-fold OOF; their CV 5.29 vs private 5.84 gap warns about fitting blend weights on OOF. |
| Inference budget | O(N*M) per pair, O(N*w) with a band; milliseconds on CPU for a 345-column x 400-level grid. Negligible next to the CNN. |

### Ambiguity handling

| Field | Value |
|---|---|
| Repeated layers | Handled only through ordering and stretch limits. On laterals repeated patterns are re-crossings of the same bed plus self-similar beds at other levels; monotone DTW cannot represent re-crossing and forces a wrong ordering. Local/subsequence DTW handles re-crossings by matching windows independently, but then loses the global tie-breaker and becomes just a local matching cost, like our gr_abs_diff channels. |
| Drift | Not addressed; DTW has no dip state. A slow systematic offset is absorbed as a warping shift unless rate constraints are added. The Chevron patent adds a pathbias term toward the survey trajectory for this reason. |
| Uncertainty and calibration | None natively. Soft-DTW or forward-backward over the same lattice gives an alignment posterior, but it is uncalibrated. |

### Relevance to erudition

| Field | Value |
|---|---|
| Where to use | (1) As an extra cost channel in the 16-channel input image: a windowed subsequence/derivative-DTW matching cost per (MD column, TVT level) that tolerates local bed thickening/thinning between the typewell and the lateral location, which a pointwise /GR - typewell GR/ channel does not. (2) As a cheap, low-correlation blend member (as in 3rd place). Not as a decoder over our posterior. |
| Error mode targeted | Band confusion (~51%) via a stretch-tolerant matching cost; secondarily broad humps (~13%) where thickness change smears the pointwise cost. Does not target drift. |
| Gap vs our implementation | We have no stretch-tolerant (warped) matching cost: the cost image is pointwise GR difference per TVT hypothesis, and any tolerance to thickness change must be learned by the ConvNeXt receptive field. Neither the 1st-place nor the 2nd-place inputs explicitly model local bed-thickness change between typewell and lateral. We also have no DTW-family blend member (3rd place did). A global DTW-style decoder over our posterior is effectively covered already by the 2nd-place DP and the failed continuity forward-backward decode, so the remaining gap is on the evidence/input side. |
| Proposed experiment | A03-2 first (cheap diversity check): implement a standalone local-DTW tracker (lateral GR windows vs typewell, anchored at the last known TVT, open-end, Sakoe-Chiba band allowing +-20% thickness change), numba, script next to posterior_fb_decode.py in kaggle_1st_place/solution. Report its holdout RMSE on the 155 wells and its error correlation with ConvNeXt/FastViT predictions (results/cnx_mobileone_ep150). Keep only if correlation < ~0.7 and a FIXED-weight blend chosen on training OOF improves holdout by > 0.39 ft (no learned gates, which already failed). A03-1 (input channel): in the cost-image builder that produces the gr_abs_diff channels, add 1-2 channels: for each (MD column c, TVT level l), the subsequence-DTW cost between the lateral GR window [c-W, c+W] (W 3-5 columns, projected to TVT via trajectory Z under hypothesis l at c) and the typewell GR around l, normalized per well; precompute and cache under data/. Train ConvNeXt 60 ep with 17-18 channels vs the identical 16-channel control and seed; require > 0.39 ft gain, then confirm with a second seed and the 150-ep recipe before distilling to FastViT. |
| Implementation effort | Low for A03-2 (standalone tracker, ~200 lines numba). Medium for A03-1 (feature pipeline change, recache, retrain; TVT-hypothesis projection must match existing channels). |
| Gpu cost | Feature precompute is CPU-only (minutes with numba). A03-1 needs one 60-epoch ConvNeXt run on the RTX 5060 8 GB, same cost as the baseline; inference unchanged apart from 1-2 more input channels. A03-2 is CPU-only. |
| Distillation compatibility | Yes. An extra input channel is transparent to FastViT KD on output posteriors (the student gets the same channel). A standalone tracker is not distilled but can be a blend member. |
| Risk | (1) The ConvNeXt may already learn stretch-tolerant matching from the 16-channel image, making the channel redundant. (2) Weak standalone members rarely help a 4.8 ft model, learned gates already failed here, and the 3rd-place gain was ~0.05 ft at 5.6 ft, below our noise floor. (3) DTW is mode-seeking and monotone while laterals are non-monotone in TVT, so any global DTW decode repeats the failed continuity-decode result. (4) GR normalization mismatch between lateral and typewell can dominate the DTW cost. |

### Other info

- **Id**: A03
- **Uncertain fields (omitted above)**:
  - code_availability
  - error_correlation_with_ours
  - notes

---

## A04 DP geosteering patents (US 11180981 Chevron; US 11480045 RNA Capital, likely ROGII-linked)

### Basics

| Field | Value |
|---|---|
| Name | Industrial DP/correlation geosteering patents: US 11180981 B2 (Chevron, automated post-geosteering by dynamic programming) and US 11480045 B2 (assigned to RNA Capital Inc., not Chevron: methods for assisted and automated horizontal-well geosteering) |
| Year | US11180981: priority 2018-11-28, filed 2019-11-04, granted 2021-11-23. US11480045: filed 2020-04-29, granted 2022-10-25. |
| Venue link | https://patents.google.com/patent/US11180981 (family: US20200165911A1, WO2020109890A1, EP3887637B1, CN113227529B, AU2019387627B2, AR117152A1) ; https://patents.google.com/patent/US11480045 |
| Code availability | No code. US11180981 states its recurrence at pseudocode level, so it is easy to reimplement; US11480045 describes a multi-algorithm workflow without implementable detail for its neural-network component. |
| Evidence level | Granted patents (legal disclosure, not peer-reviewed); illustrative synthetic and field figures only, no quantitative benchmark. |
| Extreme summary | US11180981 post-geosteers a lateral with a DP over (lateral sample, pilot-well index): std-normalized /GR/ mismatch cost, a smoothness penalty on index jumps and a penalty pulling toward the survey-based path, then a min-cost backtrack; the mapping may go up and down section. US11480045 splits the lateral into overlapping segments, scores many dip hypotheses per segment with a weighted similarity K-value (Pearson, RMS deviation, dip-change penalties), detects re-crossings by self-correlation, and reports per-segment confidence plus a 'geosteering spectrum' of alternatives. |

### Problem formulation

| Field | Value |
|---|---|
| Task definition | Estimate the stratigraphic position (TVT / typewell index) of each lateral sample from the lateral LWD log and one or more typewell (pilot) logs; US11180981 post-drill (retro-steering), US11480045 in real time as MWD/LWD data arrive. |
| Inputs | US11180981: pilot (typewell) log, lateral log, initial well path from gyro/survey. US11480045: one or more typewell logs, lateral MWD/LWD logs, trajectory, regional dip ranges, formation-thickness constraints. |
| Target parameterisation | US11180981: typewell sample index n(ns) per lateral sample ns (equivalent to TVT). US11480045: piecewise-linear formation dip per segment (explicit dip state), from which TVT follows. |
| Output representation | US11180981: single min-cost path. US11480045: best-scoring interpretation, per-segment confidence factors, and a 2D similarity spectrum over (dip, MD) showing alternative interpretations. |

### Method

| Field | Value |
|---|---|
| Architecture | US11180981: cost[n][ns] = /pilot[n]/sd_pilot - lateral[ns]/sd_lateral/; mincost[n][ns] = min over m of (mincost[m][ns-1] + smoothbias*/m-n/) + cost[n][ns] + pathbias*/pathw[ns]-n/; backtrack from the minimum at the last lateral sample. US11480045: overlapping 'big segments'; per segment several similarity algorithms (MSE, Pearson, cumulative absolute difference, and a neural network trained on original plus synthetically distorted logs, scoring by Euclidean distance in an n-dimensional embedding); K-value combines correlation T, RMS deviation SQ and dip differences with tunable exponents pow1-pow5; the algorithm/dip with the highest K wins; local-extremum search tests several extrema penalized against the previous segment's dip; self-correlation and repeated-section methods detect re-entry into the same formation; when new data arrive the method steps back a fixed depth and reprocesses. |
| Core mechanism | Non-monotone alignment: the state is the typewell level at each lateral sample, and transitions penalize level change instead of forbidding backward moves, so the path can go up and down section. An extra prior ties the solution to the survey trajectory (pathbias), acting as a drift regularizer. US11480045 adds explicit dip-continuity scoring and multi-hypothesis display with confidence. |
| Training objective | US11180981: none (smoothbias/pathbias set by the operator or automatically). US11480045: a similarity network trained on original logs plus synthetically distorted copies (metric-learning style); the K-value is hand-weighted. |
| Decoding | US11180981: Viterbi-style min-cost backtrack (mode). US11480045: argmax K-value per segment with sequential dip continuity (greedy/local), plus alternative extrema for review. |
| Metric consistent decode | Mode-seeking in both. For our pooled RMSE the Bayes-optimal counterpart is the expectation over the same lattice (as in the 2nd-place DP); the 2nd-place author found Viterbi consistently worse than the expectation. |
| Evidence scope | US11180981: whole lateral jointly against one pilot well. US11480045: overlapping segments, multiple typewells. |
| Prefix conditioning | US11180981: start point and pathbias toward the initial path; no learned use of known heel labels. US11480045: previous segments' dips constrain the next segment (sequential). |
| Propose vs select | US11180981: single path. US11480045: candidate generation (several algorithms and extrema) plus selection by K-value; the spectrum is effectively a hypothesis set. |
| Physical constraints | Smooth stratigraphic position (index jumps penalized); adherence to survey trajectory; in US11480045 also regional dip range, bounded dip change between adjacent segments, formation-thickness bounds. |

### Data

| Field | Value |
|---|---|
| Dataset | US11180981: one synthetic example and three field GR cases. US11480045: unspecified field wells. |
| Synthetic to real | US11180981: std normalization of each log. US11480045: scale normalization of typewell to lateral; similarity NN trained on synthetic distortions of real logs (augmentation-style synthetic data). |

### Results

| Field | Value |
|---|---|
| Metric and performance | No quantitative metrics. US11180981 shows the corrected path closely matching the true path in a synthetic case and plausible modelled-vs-observed GR on field cases. |
| Comparability | Not comparable (no numbers). Conceptually US11180981 is a hand-crafted analogue of the 2nd-place AnchorCNN + DP, with fixed costs where the 2nd place learns move distributions. |
| Validation protocol | None reported. |
| Inference budget | US11180981 naive O(N_lateral * N_pilot^2), or O(N_lateral * N_pilot * W) with a bounded jump window: well under a second per well on CPU for a 345 x 400 grid. US11480045 heavier but still CPU-cheap. |

### Ambiguity handling

| Field | Value |
|---|---|
| Repeated layers | US11180981: only smoothness and pathbias break ties between self-similar levels; no multi-hypothesis output. US11480045: explicitly handles re-crossing via self-correlation and repeated-section detection, tests multiple local extrema, and exposes ambiguity through the spectrum and low-confidence flags for geologist review. |
| Drift | US11180981's pathbias pulls the solution toward the survey trajectory, an explicit prior against slow drift. US11480045 bounds dip change between segments and to a regional range, i.e. an explicit dip state. |
| Uncertainty and calibration | US11180981: none. US11480045: per-segment confidence factor and similarity spectrum (uncalibrated). |

### Relevance to erudition

| Field | Value |
|---|---|
| Where to use | (1) Confirms the 2nd-place design (DP over typewell level per lateral column with a move penalty); keep the sum-product expectation, not the patent's min-cost path. (2) The pathbias idea maps to a trajectory/structural prior (cf. item A18, TVT + Z identity), best introduced as an input channel rather than a hard decode term. (3) US11480045's dip-state scoring maps to a second-order (level, previous move) state in the AnchorCNN DP, and its confidence output to a per-well flag. |
| Error mode targeted | Drift (~25%) via pathbias / dip-continuity priors; band confusion (~51%) only indirectly (multi-hypothesis exposure, confidence). |
| Gap vs our implementation | Already have: DP over (MD column, TVT level) with learned move distributions (2nd-place AnchorCNN) and expectation decode, which is stronger than the patent's min-cost path. Gaps: (a) the 2nd-place DP is first-order in level, P(dTVT / anchor at level, column), with no explicit dip/rate state; US11480045 scores dip continuity, and a (level, dip) state is the principled version. (b) No path-level prior toward a structurally/trajectory-consistent path (pathbias analogue); the ConvNeXt path has none at all. (c) No per-well or per-segment confidence output used for anything. (d) US11480045 allows multiple typewells per lateral; our inputs use a single typewell (plus the 2nd-place self-prefix reference). |
| Proposed experiment | A04-1 (dip-state DP, no retraining): in the kaggle2ndplace inference DP (eq. 5 of 2nd.txt), expand the state from level s to (s, d), d = previous move bin (21 classes); transition P(s', d' / s, d) = P_anchor(d' / s, k) * g(d' - d), g a fixed Gaussian on move change fitted on TRAINING wells (analogous to the 0.78 ft per-column std measured in posterior_fb_decode.py), renormalized. Keep expectation decode. Compare with the current first-order DP on the 155 holdout wells using the same AnchorCNN checkpoints; state size 128 x 21 per column is cheap. Also sweep g's width including infinity (which recovers the baseline) on training wells only. A04-2 (pathbias as evidence, not decode): since the hard-continuity decode failed on ConvNeXt posteriors, add a channel with a Gaussian ridge at the prefix-extrapolated TVT (TVT_last + prefix rate * dMD, sigma growing with distance from the heel) to the ConvNeXt input; 60 ep vs identical control, >0.39 ft required. A04-3 (baseline, ~1 day): reimplement US11180981 verbatim with smoothbias/pathbias tuned on training wells; report holdout RMSE and error correlation with ConvNeXt/FastViT to size the gap between hand-crafted DP and learned models. |
| Implementation effort | A04-1 medium (modify the 2nd-place DP inference, check normalization; no training). A04-2 medium (new channel + retrain). A04-3 low (~100 lines numba). |
| Gpu cost | A04-1: inference only, trivial even with a 21x larger state (< 1 s per well). A04-2: one 60-epoch ConvNeXt retrain (hours on RTX 5060 8 GB). A04-3: CPU only. |
| Distillation compatibility | A04-1 applies after AnchorCNN outputs, independent of the FastViT student (its DP marginals could also serve as soft KD targets). A04-2's channel is KD-compatible. A04-3 not applicable. |
| Risk | (1) Patents provide no benchmark; the ideas are standard DP engineering. (2) Our continuity forward-backward decode already failed on ConvNeXt posteriors because smoothing locks onto wrong bands; a dip-state DP may likewise sharpen onto wrong modes and lose the averaging benefit of the expectation. (3) AnchorCNN move distributions already see local context that encodes dip, so explicit dip continuity may add less than the 0.39 ft noise floor. (4) A pathbias-style prior only helps when structural dip is small; a strong trajectory prior would bias wells with real dip changes. (5) US11180981 is in force until about 2040: fine for research, relevant for commercial deployment. |

### Other info

- **Id**: A04
- **Uncertain fields (omitted above)**:
  - authors_org
  - error_correlation_with_ours
  - notes

---

## A05 Bayesian / particle-filter geosteering (NORCE, Alyaev et al.)

### Basics

| Field | Value |
|---|---|
| Name | Bayesian / particle-filter geosteering: Veettil & Clark SMC geosteering (2020); Muhammad, Srivastava, Alyaev et al. RL + particle filter (2024/2025); ROGII Kaggle PF components |
| Authors org | D. R. A. Veettil & K. Clark (Petrophysics 2020); R. B. Muhammad, A. Srivastava, S. Alyaev, R. B. Bratvold, D. M. Tartakovsky (UiS / NORCE / Stanford); ROGII Kaggle 1st/3rd/4th/5th place PF components |
| Year | 2020 (SMC geosteering); 2024 arXiv / 2025 Computational Geosciences (RL+PF); 2026 (Kaggle ROGII PF variants) |
| Venue link | Petrophysics 61(1):99-111, doi:10.30632/PJV61N1-2020a4 (https://onepetro.org/petrophysics/article-abstract/61/01/99/171623); Computational Geosciences 29, art. 14 (2025), arXiv:2402.06377 (https://arxiv.org/abs/2402.06377); ROGII 3rd-place writeup (https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/3rd-place-solution) |
| Code availability | No public code for Veettil & Clark or the RL+PF paper. NORCE code hub: https://github.com/geosteering-no (benchmark, DISTINGUISH, 10000 GWC interpretations). A PF already exists in our repo: rogii_champion/src/particle_filter.py (512 particles; state = level, structural slope, GR bias; 3 profiles; genealogy backward pass), glue in kaggle2ndplace/anchor_pf.py and anchor_pfchan.py; the 1st-place pipeline has PF-heatmap config (experiments/bilzard/seq_NN_cfg.py PF_heatmap_*: 500 particles x 128 seeds). |
| Evidence level | Peer-reviewed (Petrophysics 2020; Computational Geosciences 2025), plus Kaggle competition writeups (1st/3rd/4th place PF numbers), plus our own holdout logs |
| Extreme summary | Treat the well's stratigraphic position (plus dip and GR bias) as a hidden Markov state and track it along MD with a sequential Monte Carlo filter whose likelihood compares lateral GR to the typewell; the particle cloud is a non-Gaussian, possibly multimodal posterior. In ROGII it is a weak standalone predictor (6.6-7.4 ft for winners, 13.1 ft for ours) but a decorrelated feature/channel source. |

### Problem formulation

| Field | Value |
|---|---|
| Task definition | Sequentially estimate the tool position relative to stratigraphic layers (equivalently TVT along the lateral) and stratigraphic variables (boundary depths / dip) from LWD measurements as the well is drilled; in RL+PF the estimate feeds a DQN steering policy. |
| Inputs | Lateral gamma ray, a typewell / offset-well GR reference (GWC 2020 logs in Muhammad et al.), well trajectory (inclination / TVD), prior on layer structure; in Kaggle variants also the known heel TVT prefix and optionally sibling-well lateral GR as an alternative reference. |
| Target parameterisation | Explicit state space: Veettil & Clark estimate stratigraphic variables jointly with tool location; Muhammad et al. state = reservoir boundary positions b_x on a 10 ft MD grid; our PF and ROGII 3rd: structural coordinate S = TVT + Z with a slope/rate state (dTVT = r*dMD - dZ) plus GR bias and observation-family flags. |
| Output representation | Weighted particle cloud (empirical posterior). Summaries used: best particle (MAP-like) and top-5 weighted particles (Muhammad et al.); weighted mean, q10-q90, spread, skew, IQR (ROGII 3rd); 2D particle-probability heatmap on the TVT x MD grid (ROGII 1st); Gaussian bump + confidence (our pf_bump/pf_conf channels). |

### Method

| Field | Value |
|---|---|
| Architecture | Bootstrap particle filter: propagate particles with a random-walk (with momentum) transition on position/slope/bias, including rare large jumps for faults; weight by a GR likelihood (Gaussian sigma=0.2 on normalised GR in Muhammad et al.; Student-t df=2 in our PF); systematic resampling when ESS falls below a threshold (or at fixed points before each decision in Muhammad et al.); optional backward smoothing through particle genealogy; several configuration profiles / seeds blended. Muhammad et al. used 128 particles (tested 32-1024); ROGII 3rd used 500 particles x 240 seeds; our PF 512 particles x 3 profiles, updates in 64 ft MD bins. |
| Core mechanism | Bayesian recursive filtering handles the nonlinear, non-Gaussian observation model (GR vs depth is many-to-one when layers repeat), so several stratigraphic hypotheses can coexist in the particle cloud until later GR disambiguates them. The explicit slope state encodes the physical prior that dip changes slowly, which constrains drift. |
| Training objective | None for the filter itself (pure inference; hyperparameters hand-tuned or grid-searched). Muhammad et al. train only the downstream DQN policy. ROGII 4th trained a LightGBM residual corrector (L2+L1 blend) on ~115 PF self-diagnostic features. |
| Decoding | Best particle or top-5 particles (Muhammad et al.); weighted mean of genealogy-smoothed paths, with seed paths weighted by full-sequence likelihood (ROGII 3rd; our PF); forward + backward runs (ROGII 4th). |
| Metric consistent decode | Mixed: the weighted mean of smoothed particles is RMSE-consistent; best-particle / top-k decodes (Muhammad et al.) are mode-seeking and would raise our pooled RMSE under band ambiguity. |
| Evidence scope | Whole lateral processed sequentially (causal forward filter, optional backward smoothing) against the typewell; ROGII 3rd also uses sibling-well references. |
| Prefix conditioning | In ROGII variants the known heel prefix initialises particles tightly at the last known row and calibrates the typewell GR mapping; in the geosteering papers the prior comes from the pre-drill geomodel. |
| Propose vs select | Proposes a distribution / many candidate paths; ROGII 1st and 3rd use it as a candidate and feature generator feeding a NN or gate rather than as the final output. |
| Physical constraints | Smooth dip (random walk on slope with momentum), rare fault jumps, TVT+Z structural identity (our PF, ROGII 3rd), slowly drifting GR bias, heavy-tailed GR likelihood. |

### Data

| Field | Value |
|---|---|
| Dataset | Veettil & Clark: simulated formation-tops examples. Muhammad et al.: synthetic geomodels built from real GR logs of the Geosteering World Cup 2020 semifinal. ROGII Kaggle: real laterals with typewells; our holdout = 155 wells. |
| Synthetic to real | The filter needs no training data, so there is no learned synthetic-to-real gap; the gap appears as typewell-vs-lateral GR mismatch, handled by the GR bias state, typewell calibration ('seen_blend' in the 1st-place cfg) and heavy-tailed likelihoods. |

### Results

| Field | Value |
|---|---|
| Metric and performance | Muhammad et al.: best-particle GR MAE 0.0099 (normalised) with 128 particles, yet boundary estimates can disagree with truth (multimodality); RL with PF estimates reached 88.67% reservoir contact vs 66.84% using raw logs, within 1-2% of an oracle given true boundaries. ROGII 3rd: PF CV 6.58 / public 7.01 / private 6.73 ft RMSE. ROGII 1st: PF standalone CV ~7.4 ft; PF channels cut CV 7.7 -> 6.7 early, but gave no gain for the final strong single model. Ours (155 holdout wells): PF standalone best 13.07 ft (runs/X_pf_final), residual correlation 0.31 with rb_v3_tta_ep150 (4.94 ft), best blend weight 0.03 for -0.014 ft; AnchorCNN with PF channels (runs/C_pf_far_ep250) 7.35 ft pooled. |
| Comparability | Kaggle PF numbers are CV/LB RMSE in ft on the same competition data but other splits; our 13.07 ft and 7.35 ft are on our 155-well holdout. Journal results are not comparable (synthetic, boundary/decision metrics). |
| Validation protocol | Journal work: synthetic scenarios only. ROGII 3rd reported CV vs public vs private (small PF gap 6.58/7.01/6.73), consistent with a training-free method having no selection bias. Our PF is training-free, so fold safety is trivial (anchor_pfchan.py documents no tvt leakage). |
| Inference budget | Cheap CPU: our PF ~24 ms per well per profile in numpy; 3rd place's 500 x 240-seed version is much heavier but still CPU-feasible offline; fits the Kaggle budget. |

### Ambiguity handling

| Field | Value |
|---|---|
| Repeated layers | In principle multiple hypotheses stay alive in the particle cloud; in practice ESS-triggered resampling collapses minority modes and the PF commits to a wrong band (Muhammad et al. note the multimodality; ROGII 4th calls wrong-layer matches the main PF failure and mitigates with bidirectional decoding, a Viterbi second opinion with typewell-rung 'aliasing companions' and a learned corrector). ROGII 3rd kept hard/soft observation families and typewell vs sibling reference flags to preserve diversity. |
| Drift | An explicit slope/rate state with momentum (3rd place rate momentum 0.9995) models a straight-line structural trend, which matches the one-line-per-well drift mode; PF drift mostly comes from a mis-set initial rate or GR bias. |
| Uncertainty and calibration | Particle spread is an uncertainty proxy; our logs show Spearman rho 0.37-0.58 between PF spread and /error/ (rank-informative, not calibrated). No reliability/ECE analysis in the papers. Likelihood temperature (obs_temp, lik_scale) is the main calibration knob. |

### Relevance to erudition

| Field | Value |
|---|---|
| Where to use | (1) Input channels to the ConvNeXt/FastViT U-Net (the 1st-place code path exists: PF_heatmap_* in experiments/bilzard/seq_NN_cfg.py). (2) Rich per-row features (quantiles, spread, skew, forward/backward disagreement) for a gate or residual corrector. (3) A test-time tracker that uses the U-Net posterior as its emission model (see B06). |
| Error mode targeted | Drift (~25%) via the explicit slope state; secondarily band confusion (~51%) by adding sequential whole-lateral evidence. |
| Proposed experiment | E1 (cheap, most direct): train FastViT-SA12 (distilled) with the 1st-place 2D PF heatmap channels enabled in experiments/bilzard/seq_NN_cfg.py (PF_heatmap_n_particles=500, PF_heatmap_n_seeds=128; PF rotate/shift corruption augmentation on), 150 epochs, same seed; control = current 4.84 ft student. Score with seq_NN_holdout_eval.py and seq_NN_robust_compare.py on the 155 holdout wells; accept if gain exceeds the 0.39 ft noise floor, or if RMSE is equal but residual correlation with the non-PF model drops enough to improve the best pair (4.69 ft). E2: upgrade rogii_champion/src/particle_filter.py toward the 3rd-place spec (more seeds, hard/soft families, rate momentum) and export q10/q25/q50/q75/q90 + spread channels instead of pf_bump/pf_conf in kaggle2ndplace/anchor_pfchan.py; gate on standalone PF holdout RMSE < 9 ft before any NN retraining. E3 (diagnostic, no training): on posteriors from seq_NN_posterior_dump.py --save-all, check per well whether the PF modal band is correct on the U-Net's wrong_mode wells; if yes on a meaningful share, the PF carries evidence the network lacks and E1/E2 are worth running. |
| Implementation effort | Low for E1 (code path exists in the 1st-place config; mainly PF cache generation for real and synthetic wells); medium for E2 (PF engineering and tuning); low for E3 (analysis only). |
| Distillation compatibility | Yes: PF features are inputs, so teacher and student both consume them and KD on output posteriors is unchanged. PF features must also be computed for synthetic training wells (1st place cached them). |
| Error correlation with ours | High diversity as a standalone member (residual correlation 0.20-0.33 with the ConvNeXt) but too inaccurate to earn blend weight; as a channel the diversity is partly absorbed. 1st place kept PF-channel models specifically as decorrelated ensemble members. |
| Risk | 1st place found PF channels stopped helping once backbones and augmentation became stronger; our AnchorCNN PF-channel result (7.35 ft) never became competitive. Shortcut learning: the network may copy the PF path (1st place needed PF rotate/shift corruption). Resampling collapses minority modes, so PF evidence for the correct band may be missing exactly where needed. Our PF is much weaker than winners', and improving it costs engineering time for a payoff that may sit inside the +-0.39 ft noise floor. |

### Other info

- **Uncertain fields (omitted above)**:
  - gap_vs_our_implementation
  - gpu_cost
  - notes

---

## A06 Ensemble Kalman / ensemble-based geosteering and well-log interpretation (DISTINGUISH, 2021+)

### Basics

| Field | Value |
|---|---|
| Name | Ensemble Kalman / ensemble-smoother geosteering and well-log interpretation: LM-EnRML log interpretation (Jahani et al.), GAN + EnRML earth-model updating, FlexIES Goliat workflow, DISTINGUISH workflow |
| Authors org | NORCE / UiS: N. Jahani, J. Ambia Garrido, S. Alyaev, K. Fossum, E. Suter, C. Torres-Verdin (log interpretation); K. Fossum, S. Alyaev, J. Tveranger, A. H. Elsheikh (GAN-EnRML); M. H. Rammay, S. Alyaev, D. S. Larsen, R. B. Bratvold, C. Saint (Goliat FlexIES); S. Alyaev, K. Fossum, H. E. Djecta, J. Tveranger, A. H. Elsheikh (DISTINGUISH). Earlier EnKF geosteering: Chen, Lorentzen, Vefring (2015); Luo et al. (2015). |
| Year | 2015 (EnKF proactive geosteering), 2021 (ensemble log interpretation, arXiv), 2022 (GAN-EnRML, J. Comput. Sci.), 2024 (Goliat, Geophysics), 2024/2025 (DISTINGUISH, ECMOR 2024) |
| Venue link | arXiv:2103.05384 (https://arxiv.org/abs/2103.05384); J. Computational Science 65 (2022), arXiv:2207.03596; Geophysics 89(6):D301-D314 (2024), arXiv:2210.15548; ECMOR 2024, arXiv:2503.08509; DISTINGUISH code https://github.com/geosteering-no (DISTINGUISH-WF, GAN-geosteering) |
| Code availability | Yes for DISTINGUISH: github.com/geosteering-no/DISTINGUISH-WF and GAN-geosteering (ECMOR 2024 reproduction), spade-gan-inversion, GAN-EM-model. No code linked for the LM-EnRML log-interpretation paper or Goliat FlexIES. |
| Evidence level | Peer-reviewed (Geophysics 2024, J. Comput. Sci. 2022, ECMOR 2024) plus arXiv preprints; no ROGII / GR-typewell TVT benchmark |
| Extreme summary | Represent the uncertain earth model (layer boundaries, resistivities or GAN latent vectors) as an ensemble of realizations and update it with iterative ensemble smoothers (LM-EnRML, FlexIES, ES-MDA-like) as new logs arrive; this gives fast, derivative-free, approximately Bayesian uncertainty propagation, but Gaussian-linear update steps handle multimodal band ambiguity poorly. |

### Problem formulation

| Field | Value |
|---|---|
| Task definition | Real-time inversion of LWD logs (mostly extra-deep azimuthal resistivity, also density/GR) into layer boundaries and properties around the well, plus look-ahead prediction and steering decisions (DISTINGUISH adds dynamic-programming decisions). |
| Inputs | Extra-deep EM resistivity logs (Goliat, GAN-EnRML, DISTINGUISH), multiple logging measurements incl. density/resistivity (Jahani et al.), trajectory, prior geomodel; GR-typewell correlation is not the main use case. |
| Target parameterisation | Layer boundary depths and petrophysical properties (Jahani et al.); GAN latent vector that decodes to a 2D facies/geomodel section (GAN-EnRML, DISTINGUISH); 1D layered models with model-error terms (Goliat). No TVT-per-column parameterisation. |
| Output representation | Ensemble of realizations (typically tens to ~100 members) with empirical mean/median, percentiles and probability maps. |

### Method

| Field | Value |
|---|---|
| Architecture | Prior ensemble -> forward model (analytical EM solver, or DNN surrogate in Goliat/DISTINGUISH) -> ensemble-based update of parameters using cross-covariances between parameters and predicted data (Kalman-type gain), iterated (LM-EnRML, FlexIES) to handle nonlinearity. GAN-EnRML performs the update in GAN latent space so posteriors stay geologically realistic; DISTINGUISH wraps this with a global discrete DP optimiser for steering. |
| Core mechanism | Kalman-type updates estimate gradients statistically from ensemble covariances, so no adjoint is needed and the cost is a few ensemble forward runs; iterating the smoother approximates randomized maximum likelihood sampling of the posterior. Latent-space updating (GAN) keeps realizations on a learned geological manifold, a way to impose non-Gaussian structure despite the Gaussian update. |
| Training objective | The update itself is not trained. DNN surrogates (Goliat, DISTINGUISH FNN) are trained on simulated EM responses with MSE-type losses; GANs trained adversarially on geological training images. |
| Decoding | Ensemble mean or median (Goliat reports the median as on par with proprietary inversion); probability maps for decisions. |
| Metric consistent decode | Ensemble mean is RMSE-consistent and the median MAE-consistent; but the posterior itself is biased toward unimodal Gaussian-like spread, so the mean may sit between bands without the calibrated weights our metric needs. |
| Evidence scope | Whole drilled section up to the current bit position, local around the well; single well (plus prior); no typewell correlation or multi-well joint inference. |
| Prefix conditioning | Sequential assimilation means all past measurements condition the ensemble; analogous to conditioning on the known heel, but not on labels. |
| Propose vs select | Proposes an ensemble of hypotheses; could serve as a candidate generator for a reranker, but the literature uses mean/median. |
| Physical constraints | Layered-earth forward physics; geological realism via GAN prior; model-error term for surrogate inaccuracy (Goliat). |

### Data

| Field | Value |
|---|---|
| Dataset | Synthetic and field-inspired layered cases (Jahani et al.); synthetic fluvial GAN models (GAN-EnRML, DISTINGUISH); historic Goliat field well, Barents Sea, extra-deep EM data (Rammay et al.). |
| Synthetic to real | DNN surrogate trained on synthetic EM responses and applied to field data with an explicit model-error estimate (Goliat); no normalisation analogous to lateral-vs-typewell GR. |

### Results

| Field | Value |
|---|---|
| Metric and performance | Jahani et al.: accurate boundaries and properties, about 10-125x lower compute than Metropolis-Hastings MCMC. GAN-EnRML verified statistically against MCMC. Goliat: median estimate on par with proprietary deterministic inversion while adding uncertainty. DISTINGUISH: produces meaningful steering recommendations and maps boundaries in a low net-to-gross synthetic case. No RMSE in TVT ft. |
| Comparability | Not comparable to our pooled RMSE: different sensors (EM resistivity), synthetic or single-well cases, boundary / decision metrics. |
| Validation protocol | MCMC reference posterior on synthetic cases; single field case study; no spatially grouped held-out benchmark. |

### Ambiguity handling

| Field | Value |
|---|---|
| Repeated layers | Weak point: EnKF/ES updates are linear-Gaussian in the parameters, so bimodal posteriors (two candidate bands) get averaged or collapse; GAN latent parameterisation helps realism but not multimodal weights. Gaussian-mixture EnKF variants (e.g. Bayesian EnKF for GMMs, Math. Geosci. 2024) are the known remedy but are not used in the geosteering papers. |
| Drift | Well suited in principle: a low-dimensional structural parameter vector (offset, dip nodes) is nearly Gaussian and near-linear locally, so an ensemble smoother can fit a per-well straight-line or piecewise-linear dip correction against whole-lateral GR misfit. |
| Uncertainty and calibration | Uncertainty = ensemble spread; validated against MCMC on synthetic cases (GAN-EnRML), but ensemble collapse / under-dispersion with small ensembles is a known issue; no reliability diagrams or ECE. |

### Relevance to erudition

| Field | Value |
|---|---|
| Where to use | Post-hoc test-time refinement of low-dimensional structural parameters (offset + piecewise-linear dip) per well, initialised from the U-Net posterior; or as a principled way to combine an ensemble of network outputs (ConvNeXt/FastViT/AnchorCNN/TTA members) into a calibrated per-well distribution instead of per-column softmax averaging. |
| Error mode targeted | Drift (~25%) and broad/skewed humps (~13%); weak on band confusion (~51%) because of Gaussian update assumptions. |
| Gap vs our implementation | We do per-column softmax over TVT with independent columns and an expectation decode; there is no well-level parametric state updated against whole-lateral GR misfit, and ensembling is limited to averaging member posteriors (TTA, pairs). What the ensemble-smoother line offers that we lack: (a) a low-dimensional per-well structural model (offset, dip nodes, typewell stretch) fitted to all GR evidence jointly, giving coherent path-level hypotheses rather than column-wise marginals; (b) sample-based posterior propagation that yields path samples (useful for uncertainty flags and for scoring hypotheses on whole-lateral GR). What it does not offer: a remedy for band confusion, since Gaussian updates blend modes, which our mean decode already does. Our failed forward-backward continuity decode is a related post-hoc structure prior, which lowers expectations. |
| Proposed experiment | Per-well ES-MDA drift correction (test-time, no training): for each holdout well, take the U-Net expectation path, parameterise a correction delta(md) = a + b*(md - md0) (optionally + 2-4 piecewise-linear dip knots), sample an ensemble of N=100 corrections from a prior N(0, sigma) with sigma set from training-well residual statistics, and run 4 ES-MDA iterations where the data are (1) the lateral GR in the eval zone vs typewell GR interpolated at corrected TVT and (2) continuity with the known prefix at the anchor. Output the ensemble-mean corrected path and blend with the uncorrected path by a weight tuned on a training fold. New file: kaggle_1st_place/solution/posterior_esmda_correct.py reading the dumps from seq_NN_posterior_dump.py --save-all; control = the same FastViT 4.84 ft predictions; report pooled RMSE and the drift-component share from posterior_analysis.py. Only restrict correction to wells whose posterior is unimodal (single hump) to avoid mode blending, since the target is the ~25% single-hump drift mode. |
| Implementation effort | Low-medium: ES-MDA is ~50 lines of numpy; the forward model is typewell interpolation; main work is choosing priors, noise levels and the GR normalisation already used by the canvas. |
| Gpu cost | None: CPU only, roughly seconds per well (100 members x 4 iterations x cheap interpolation); no retraining. |
| Distillation compatibility | Orthogonal: a post-processing step on the student's decoded path; no effect on KD. Cannot itself be distilled easily unless the correction is later used as a training target. |
| Error correlation with ours | Low incremental diversity: it refines our own path rather than producing an independent member, so its value is in correcting systematic drift, not in blending. |
| Risk | GR misfit is exactly the evidence the network already sees through its cost-volume channels, so a hand-crafted likelihood may double-count or be worse than the learned one (as with the failed geo-prior and continuity decode). GR is non-monotone and self-similar, so the ES-MDA objective is multimodal and a Gaussian update can pull a correct path toward a wrong band. Small, noisy GR signal in some wells; per-well improvements may fall inside the +-0.39 ft noise floor. Literature evidence is EM-resistivity, not GR-typewell, so transfer is unproven. |

### Other info

- **Uncertain fields (omitted above)**:
  - inference_budget
  - notes

---

## A07 Reinforcement-learning geosteering and the NORCE/UiS geosteering robot line (2024-2026)

### Basics

| Field | Value |
|---|---|
| Name | Reinforcement-learning geosteering and the NORCE/UiS geosteering robot line (RL+PF 'PluRaListic' robot, DQN / Dual-DRL / ADP decision frameworks), 2023-2026 |
| Authors org | Ressi Bonti Muhammad, Apoorv Srivastava, Sergey Alyaev, Reidar B. Bratvold, Daniel M. Tartakovsky, Igor Cheraghi (Kristian Fossum, Hibat Errahmen Djecta on 2026 paper). University of Stavanger (UiS), NORCE Norwegian Research Centre, Stanford; tested in ROGII StarSteer / Geosteering World Cup environments. |
| Year | 2023 (DQN vs ADP, arXiv 2310.04772); 2024 (SPE Norway Subsurface Conf. robot in cloud environment; arXiv 2402.06377); 2025 (Computational Geosciences 29:14; SPE Journal 30(03):995; Muhammad PhD dissertation UiS; Dual-DRL book chapter); 2026 (arXiv 2606.17331, June 2026) |
| Venue link | Muhammad et al., 'High-precision geosteering via RL and particle filters', Computational Geosciences 29, 14 (2025) https://link.springer.com/article/10.1007/s10596-025-10352-y (arXiv https://arxiv.org/abs/2402.06377); Muhammad, Cheraghi, Alyaev, Srivastava, Bratvold, 'Geosteering Robot Powered by Multiple Probabilistic Interpretation and AI: Benchmarking Against Human Experts', SPE Journal 30(03) 995 (2025) https://doi.org/10.2118/218444-PA; 'Enhancing Geosteering With AI: Integrating a Decision-Making Robot Into a Cloud-Based Environment' SPE Norway Subsurface 2024 https://onepetro.org/SPEBERG/proceedings-abstract/24BERG/24BERG/D011S002R008/544174; 'Optimal Sequential Decision-Making in Geosteering: A RL Approach' https://arxiv.org/abs/2310.04772; 'Uncertainty-Aware Well Placement: Simulator-Verified Dual-Network RL Meets Particle Filters' https://link.springer.com/chapter/10.1007/978-3-031-97554-7_14; Djecta et al. 'Decision-Driven Geosteering Under Uncertainty' arXiv 2606.17331 https://arxiv.org/abs/2606.17331; dissertation record https://nva.sikt.no/registration/0198cc847557-814be692-c20c-408f-9a2b-7c3f757a6a4a |
| Evidence level | Peer-reviewed (Computational Geosciences 2025, SPE Journal 2025) plus arXiv preprint (2026). Details of the SPE Journal robot come only from the abstract and JPT synopsis because OnePetro full text was blocked (403). |
| Extreme summary | A particle filter over the bit's position relative to stratigraphic boundaries (offset plus local dip), weighted by matching GR against an offset/reference log, produces hundreds of interpretations. A DQN-family agent then picks steering actions from that belief. The robot reached a median 77.3% reservoir contact, placing it in the top 14% of GWC human experts, at about 4 s per decision. |

### Problem formulation

| Field | Value |
|---|---|
| Task definition | Sequential decision-making: choose inclination/targetline changes at decision points along a lateral to maximise reservoir contact, given a probabilistic interpretation of where the well is relative to the target layers. The interpretation sub-task (PF) matches our task: estimating the stratigraphic position (TVT-like offset) of the drilled well from GR and a reference log. The RL sub-task has no counterpart in our offline, fixed-trajectory TVT prediction. |
| Inputs | Real-time GR measured along the lateral; GR of an offset/reference (type) well; the trajectory (minimum curvature, DLS ≤3°/100 ft); in the 2026 work, reference stratigraphic interpretations used to fit a KDE of dip-angle increments. The 2025 paper used real GR from the Geosteering World Cup 2020 semifinal, scaled 0-1. No neighbour wells and no multi-log input. |
| Target parameterisation | PF state is geometric. In the 2025 paper it is the boundary positions b_x of a 3-layer faulted reservoir (20 ft constant thickness, 2-3 faults). In the 2026 paper it is xi = (offset_t, angle_t): vertical offset of the target boundaries plus local dip angle. This amounts to an explicit dip/slope state, the same idea as u = TVT + Z with a dip state in the silver-medal tracker (A15). The RL state is either raw GR (RL-Log) or the PF-estimated distance to the boundaries (RL-Estimation). |
| Output representation | A particle cloud (128 particles in 2025; 256 in training and 2064 in testing in 2026) giving 'hundreds of most-likely interpretations'. The decision module consumes either the single highest-weight particle or the top-5 particles ('pluralistic'). The final output is a steering action and trajectory, not a TVT curve. |

### Method

| Field | Value |
|---|---|
| Architecture | (1) Bootstrap/sequential-importance-resampling particle filter. Transition: boundary evolution with faults (2025) or KDE-sampled dip-angle increments (2026). Likelihood: Gaussian on the mismatch between measured and reference-predicted GR (sigma = 0.2 on the 0-1 scaled log), data assimilated every 10 ft. (2) Value-based RL: DQN with experience replay and a target network, 3 fully connected ReLU hidden layers (2-4x input width), gamma 0.95, lr 5e-4, 20k episodes, 50k replay, batch 64. 11 discrete actions (inclination change -5° to +5°, inclination held within 86-94°), 10 decision points every 320 ft. The 2026 variant adds a dueling value/advantage 'Dual DRL' network and compares it with Approximate Dynamic Programming. The SPE robot swaps in a targetline action space and realistic DLS constraints inside a StarSteer-like cloud environment. |
| Core mechanism | Decouple interpretation (PF belief over offset/dip) from decision (RL policy trained on simulated geology). Feeding the RL agent PF estimates instead of raw logs was the key to generalisation. RL-Log collapsed on unseen GR logs, while RL-Estimation stayed robust across log depths and fault counts. Its reservoir contact came within 1-2% of an RL agent given true look-ahead boundaries. For our problem, the transferable pieces are: an explicit dip state with a data-learned (KDE) dip-increment transition prior; keeping several interpretations instead of one; and a belief summary that downstream modules consume. |
| Training objective | PF: none (Bayesian filtering, hand-set Gaussian GR likelihood). RL: Q-learning TD loss. Reward is 0 inside the reservoir and -1 outside, summed over 32 points (10 ft spacing) per decision. The 2026 paper also reports RMS jerk (steering smoothness) as a stability metric. There is no supervised loss on stratigraphic position. |
| Decoding | Interpretation decode is mode-like: the highest-weight particle, or the top-5 particles passed to the policy. Actions come from the argmax of Q-values. The 2026 ADP baseline uses robust optimisation over the ensemble. |
| Metric consistent decode | No. The decode picks the highest-weight particle (MAP-like, mode-seeking), which can raise pooled RMSE in our setting. The PF weighted mean (RMSE-consistent) is available but was not what the robot used. |
| Evidence scope | Whole lateral drilled so far (causal, filtering only, no smoothing) plus a single reference/typewell. Nothing from multiple wells jointly. |
| Prefix conditioning | Implicit. The PF is initialised at landing (first 960 ft) with a known position and assimilates GR causally, so earlier interpretation constrains later interpretation through the state. There is no explicit fine-tuning or closed-form correction from known labels. |
| Propose vs select | Propose-and-select. The PF proposes many interpretations and the policy selects an action. For us, a PF could slot in as a candidate generator (K particle clusters) for a reranker, or as a cheap CPU ensemble member. |
| Physical constraints | Constant layer thickness, discrete faults, smooth dip evolution (KDE-sampled increments), DLS limits on the trajectory, and inclination bounds of 86-94°. |

### Data

| Field | Value |
|---|---|
| Synthetic to real | 100% synthetic geology with real GR textures. GR scaled 0-1 and shifted to target TVD. No sim-to-real alignment stage. The human benchmark ran in a GWC-like synthetic scenario, not on real laterals. |

### Results

| Field | Value |
|---|---|
| Metric and performance | Computational Geosciences 2025, reservoir contact: rule-based with 1 particle 54.75%; RL-Log 66.84%; RL-Estimation with 1 particle 88.67% (input MAE 1.29); with 5 particles 88.91%. Oracle RL with true look-ahead >90%. SPE Journal 2025: median 77.3% reservoir contact over 1,000 runs, top 14% of human experts; best run beat every expert. 2026: Dual DRL RMS jerk 0.8 vs ADP 1.32 and DQN 1.38, with the highest median return. None of these papers reports a TVT or stratigraphic-position RMSE. |
| Comparability | Not comparable. The metrics are reservoir contact and steering smoothness, not pooled TVT RMSE on 155 holdout wells. The PF input MAE of 1.29 is in grid cells on a toy 3-layer model. |
| Validation protocol | Monte Carlo over randomly generated synthetic geologies, plus generalisation tests to unseen GR logs, different log depths and fault counts. There are no real-well holdouts, no spatial grouping and no leaderboard. The human benchmark compares against archived GWC expert results. |
| Inference budget | RL-Estimation <1 s per scenario (2025). Robot about 4 s per steering decision vs a 2-min stand time (SPE 2025). Training 30 min (RL-Log) to 300-420 min (RL-Estimation), with the PF taking about 95% of compute. A CPU PF at 128-2064 particles fits a Kaggle/rig budget. |

### Ambiguity handling

| Field | Value |
|---|---|
| Repeated layers | Not addressed explicitly. The synthetic reservoirs have 3 layers of constant thickness, so self-similar repeated bands barely occur. Keeping top-5 interpretations gave only +0.24% contact over the single best particle. PF weights over a GR likelihood are known to struggle to break repeated-band ties without extra evidence. |
| Drift | Handled structurally by the explicit dip/angle state with a KDE transition prior learned from reference interpretations (2026). This is the most relevant piece for our single-straight-line drift mode, but no drift metric is reported. |
| Uncertainty and calibration | The particle cloud represents uncertainty and belief-informed control is the point of the work. There is no reliability or calibration analysis of particle weights, and the likelihood sigma is hand-set (0.2). |

### Relevance to erudition

| Field | Value |
|---|---|
| Where to use | (a) As a cheap CPU ensemble/blend member or candidate generator: an (offset, dip) PF whose emissions come from our U-Net per-column posteriors or from GR-vs-typewell correlation. (b) As a data-driven prior: fit a KDE of dip increments on real training-well labels, and use it to check and retune the dip process of our synthetic well generator (85% of training data). (c) RL itself has no place in an offline, fixed-trajectory TVT-regression pipeline. |
| Error mode targeted | Drift (~25%) via the explicit dip state and learned dip prior; weakly band confusion (~51%) via multi-hypothesis particles. The RL component targets no Erudition error mode. |
| Proposed experiment | E1 (cheap diagnostic, CPU, about 1 h): fit a KDE of per-step dip increments d(TVT+Z)/dMD, taken at the 345-column resolution, on real training wells. Compare it with the same statistic from our synthetic generator (the synthetic-well code used by experiments/bilzard/seq_NN_data_prep.py / seq_NN_dataset.py). Compare histograms, tail mass and autocorrelation of dip. If the synthetic dips are too smooth or too rough, retune the generator's dip process to match. Retrain 60-epoch ConvNeXt vs the unchanged 60-epoch baseline (5.47 ft) on the same seed, and require a gain beyond the +-0.39 ft noise floor (ideally 2 seeds). E2 (moderate): implement a 2-state (offset, dip) SIR particle filter with the real-data KDE dip transition and emission p(obs / TVT) taken from the per-column softmax dumped by seq_NN_posterior_dump.py --save-all. Add forward-backward smoothing, since the robot's filtering-only setup is weaker for offline use. Decode with the particle-weighted mean, not the top particle. Evaluate pooled RMSE on the 155 holdout wells with a new posterior_pf_decode.py next to posterior_fb_decode.py. Control: the expectation decode on the same dumps, plus the failed fb decode. Also score the PF output as a blend member with ConvNeXt/FastViT at fixed weights chosen on training folds only. |
| Implementation effort | Low for E1 (statistics script plus one generator knob and retrain). Medium for E2 (PF plus smoother on existing posterior dumps, about 200-300 lines; no network changes). |
| Gpu cost | E1: one 60-epoch ConvNeXt retrain on the RTX 5060 8 GB (same cost as the existing baseline run). E2: no GPU, since the PF runs on CPU over saved posteriors (1-2k particles x 345 columns x 155 wells takes seconds to minutes). |
| Distillation compatibility | E1 changes only training data, so it is fully compatible with FastViT KD. E2 is a post-hoc decode on output posteriors and works on teacher or student posteriors equally. The RL policy has no role. |
| Error correlation with ours | Low diversity for E2 if emissions come from our own posteriors, since it re-weights the same evidence. Moderate diversity if emissions are raw GR-vs-typewell correlation, as in the robot's PF, but that PF baseline is much weaker (7.35 ft). E1 changes the model itself, not a blend member. |
| Risk | The RL part is irrelevant to RMSE. The PF part has already been tried in spirit: a forward-backward continuity decode over our posteriors failed, and the 7.35 ft PF is far behind 4.84 ft. Particle weights inherit any miscalibration of the softmax, and MAP-style top-particle decoding raises RMSE. The robot line was validated only on simple synthetic 3-layer or GWC-like geologies with reservoir-contact metrics, so it gives no evidence on band confusion in real repeated-layer stratigraphy. E1 may show that our generator already matches real dip statistics, in which case it yields no gain. |
| Notes | Useful lineage points: the 2026 Djecta et al. paper runs on ROGII StarSteer via the Solo API, the same vendor ecosystem as the Kaggle competition. The GWC 2020 semifinal GR log they use is public (see A19). The robot's '4 s per decision' and 'top 14% of experts' figures are from the SPE abstract and JPT synopsis; full-text details (particle count, exact state, action space) could not be verified because OnePetro returned 403. The key lesson for our pipeline is not RL but the separation of belief and decision, together with the data-learned dip-increment prior. |

### Other info

- **Id**: A07
- **Uncertain fields (omitted above)**:
  - code_availability
  - dataset
  - gap_vs_our_implementation

---

## A08 Deep-learning automated geosteering (GAN earth models, CNN/LSTM relative-position predictors, GenAI decision support)

### Basics

| Field | Value |
|---|---|
| Name | Deep-learning automated geosteering: GAN earth models plus ensemble data assimilation (DISTINGUISH), MDN/multiple-trajectory-prediction relative-position predictors, CNN/RNN stratigraphic correlators, and GenAI explainable decision support (2020-2026) |
| Authors org | Main line: Sergey Alyaev, Kristian Fossum, Jan Tveranger, Ahmed H. Elsheikh, Reidar Bratvold, Igor Cheraghi and co-authors, from NORCE, Heriot-Watt and University of Stavanger (DISTINGUISH project, geosteering-no GitHub org). Related correlators: SPE ATCE 2020 'Deep-Learning-Based Automated Stratigraphic Correlation'; Yang et al. 2023 ASDNet (attention dense net); 'Stratigraphic Correlation of Well Logs Using Geology-Informed Deep Learning Networks' (Processes 2025); intra/inter-well constrained stratigraphic-layer identification (2025). |
| Year | 2021 (First Break GAN forecasting; ICCS 2021 GAN+FDNN+EnRML), 2022 (ESS 'Direct multi-modal inversion' MDN-MTP; Journal of Computational Science GAN verification; SPWLA 2022 sequential multi-realization; Goliat strategic workflow arXiv 2210.15548), 2023 (ASDNet), 2024 (EAGE GenAI decision support), 2025 (Processes geology-informed DL), 2026 (arXiv 2503.08509 v2 extended) |
| Venue link | Alyaev, Tveranger, Fossum, Elsheikh, 'Probabilistic forecasting for geosteering in fluvial successions using a GAN', First Break 39(7) 2021 https://arxiv.org/abs/2207.01374; Fossum, Alyaev, Tveranger, Elsheikh, 'Deep learning for prediction of complex geology ahead of drilling', ICCS 2021 https://arxiv.org/abs/2104.02550; Fossum et al., 'Verification of a real-time ensemble-based method for updating earth model based on GAN', J. Comput. Sci. 2022 https://arxiv.org/abs/2207.03596; Alyaev & Elsheikh, 'Direct multi-modal inversion of geophysical logs using deep learning', Earth and Space Science 2022 https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021EA002186 (arXiv https://arxiv.org/abs/2201.01871); 'Sequential Multi-Realization Probabilistic Interpretation of Well Logs and Geological Prediction by a Deep-Learning Method', SPWLA 2022 https://onepetro.org/SPWLAALS/proceedings-abstract/SPWLA22/SPWLA22/D051S022R001/487879; 'Strategic Geosteering Workflow with UQ and DL: Goliat Field' https://arxiv.org/abs/2210.15548; Alyaev et al., 'A Generative-AI Modeling Framework for Explainable Decision Support in Complex Geosteering Scenarios' https://arxiv.org/abs/2503.08509; ASDNet https://www.sciencedirect.com/science/article/pii/S2666544123000278; Geology-informed DL correlation https://doi.org/10.3390/pr13051288; SPE ATCE 2020 https://onepetro.org/SPEATCE/proceedings-abstract/20ATCE/2-20ATCE/449943 |
| Evidence level | Peer-reviewed (First Break 2021, ICCS 2021, J. Comput. Sci. 2022, ESS 2022, Processes 2025, ASDNet 2023) plus arXiv preprints (2503.08509 v2, 2210.15548). All GAN/EnKF geosteering results are synthetic. Full-text numbers were verified only for 2503.08509 (HTML); the rest come from abstracts. |
| Extreme summary | Two DL routes to geosteering. (1) GANs trained offline encode complex 2D geology in a roughly 60-dimensional latent vector, a neural forward proxy simulates EM/resistivity logs, and an ensemble smoother (EnRML/EnKF) updates the latent ensemble in real time, followed by DP/robust optimisation for decisions. (2) Direct inversion networks (MDN trained with a multiple-trajectory-prediction loss) map GR along the well to several probable stratigraphic curves with probabilities in milliseconds, avoiding the mode collapse of standard MDNs. |

### Problem formulation

| Field | Value |
|---|---|
| Task definition | Route 1: estimate the 2D geology around and ahead of the bit (facies/resistivity section), then choose the next steering move. Route 2 (closest to our task): from a GR log measured along the lateral plus a reference stratigraphy, predict the well's stratigraphic position/offset curve (the relative position to the layers) as multiple weighted hypotheses, including ahead of the data. Correlators (ASDNet etc.) label formation tops per depth, mostly in vertical wells. |
| Inputs | Route 1: ultra-deep azimuthal resistivity (UDAR) / extra-deep EM measurements, trajectory, a prior ensemble of 250 latent vectors. Route 2: gamma-ray along the trajectory plus the implicit reference stratigraphy learned from training geology; single-well only. Correlators: multi-log suites (GR, resistivity, density, etc.) across vertical wells; the 2025 work adds intra- and inter-well constraints. |
| Target parameterisation | Route 1: GAN latent vector z (about 60-dimensional standard Gaussian) generating 64x64 facies/property sections. Route 2: stratigraphic offset/position curve along MD relative to the typewell, a TVT-like curve, discretised in grid cells. Correlators: per-depth formation class labels. |
| Output representation | Route 1: an ensemble of 250 geomodel realisations (a posterior sample set) plus an optimal next action. Route 2: K hypothesis curves with probabilities (mixture/MTP head), i.e. explicit multi-hypothesis output. Correlators: per-depth class probabilities (segmentation). |

### Method

| Field | Value |
|---|---|
| Architecture | Route 1 (DISTINGUISH): a convolutional GAN generator (offline-trained on outcrop-based fluvial models, Williams Fork analogues) maps z to a 2D section. A forward deep neural network (FDNN/FNN) emulates a commercial extra-deep EM simulator, so the full forward model G(z) to logs is differentiable and fast. An ensemble randomized maximum likelihood (EnRML) or EnKF iteratively updates the 250-member latent ensemble from logs. A dynamic-programming algorithm builds reward matrices per realisation, and a robust step picks the move with maximal mean expected reward. Route 2 (ESS 2022 / SPWLA 2022): a deep network mapping the GR sequence to K candidate stratigraphic curves plus K mixture weights (mixture density network head trained with a multiple-trajectory-prediction loss borrowed from autonomous-driving trajectory forecasting). Correlators: 1D CNN/RNN, attention-based dense nets (ASDNet) and semantic-segmentation networks over log depth. |
| Core mechanism | Route 1: amortise geological realism into a GAN prior so that data assimilation searches a low-dimensional, geologically plausible manifold, keeping uncertainty as an ensemble. Route 2: a winner-takes-all / MTP loss trains K heads to cover distinct modes, such as distinct repeated-band alignments, and a classification loss learns their probabilities. This avoids the averaging or collapse of a unimodal regressor and gives explicit, scoreable alternative interpretations in milliseconds. |
| Training objective | GAN: adversarial loss (offline). FDNN: supervised regression to simulator outputs. Data assimilation: EnRML/EnKF misfit minimisation (no network training at run time). MDN-MTP: multiple-trajectory-prediction loss, i.e. regression on the best-matching hypothesis plus cross-entropy on which hypothesis wins. Correlators: cross-entropy per depth; the 2025 geology-informed work adds stratigraphic-order constraints to fix implausible results such as older formations above younger ones or repeated formations at wrong depths. |
| Decoding | Route 1: ensemble statistics and a robust (mean-reward) decision. Route 2: returns all K curves with probabilities; a single answer would be the argmax hypothesis (mode) or the probability-weighted mean. Papers emphasise the multi-modal set rather than one decode. |
| Metric consistent decode | Route 2's natural 'most probable curve' decode is mode-seeking (bad for pooled RMSE). The probability-weighted mean of the K curves is RMSE-consistent if the weights are calibrated. Route 1 uses ensemble means for decisions (RMSE-consistent in spirit). |
| Evidence scope | Whole lateral so far (sequential/real-time), single well plus prior geology. There is no joint multi-well evidence except the 2025 inter-well-constrained correlator. |
| Propose vs select | Route 2 is natively propose-and-select (K hypotheses plus scores), so it can feed a gate/reranker or act as an ensemble member. Route 1 proposes an ensemble and selects by decision optimisation. |
| Physical constraints | Route 1: geological realism via the GAN prior, physics via the EM forward proxy. Route 2: whatever geology the training generator contains (fixed-thickness stratigraphy with possible faults). Correlators: explicit stratigraphic ordering (2025). |

### Data

| Field | Value |
|---|---|
| Dataset | Route 1: synthetic 2D 64x64 sections cut from a 3D fluvial model based on Williams Fork Formation outcrops, prior ensemble of 250; the Goliat field case study (Barents Sea) for the strategic workflow. Route 2: synthetic layered geologies with GR logs; later NORCE work used the GWC 2020 semifinal GR log. Correlators: field well sets (e.g. Norwegian/Chinese basins), region-specific. |
| Synthetic to real | Route 1 and Route 2 are trained entirely on synthetic geology; no sim-to-real alignment is reported. Route 2 uses real GR textures in some variants. None report a real-lateral holdout comparable to ROGII. |

### Results

| Field | Value |
|---|---|
| Comparability | Not comparable. Metrics are in grid cells on synthetic geology, qualitative, or formation-label accuracy in vertical wells; none are pooled TVT RMSE in feet on real laterals. |
| Validation protocol | Synthetic hold-out realisations and twin experiments (truth sampled from the prior). No spatially grouped real-well CV and no leaderboard. |

### Ambiguity handling

| Field | Value |
|---|---|
| Repeated layers | Route 2 is the only DL approach here that targets ambiguity directly. MTP/WTA heads keep separate hypotheses for alternative alignments instead of blending them, which addresses exactly our 'blend' failure (15%) and makes 'wrong_mode' (24%) visible as a scored alternative. The 2025 geology-informed correlator adds ordering constraints against repeated-formation errors. Route 1 relies on EM depth-of-investigation to see layers ahead, which GR-only laterals lack. |
| Drift | Not specifically addressed. Route 1's ensemble update can absorb slow structural trends via the latent, but no drift metric is reported. |
| Uncertainty and calibration | Both routes output uncertainty (ensemble or K weighted hypotheses). No reliability diagrams, ECE or temperature/isotonic calibration of hypothesis probabilities is reported. EnKF ensembles are known to under-disperse with small ensembles. |

### Relevance to erudition

| Field | Value |
|---|---|
| Where to use | (a) A multi-hypothesis (MTP/WTA) auxiliary head on the ConvNeXt/FastViT U-Net decoder: K band-hypothesis TVT curves plus logits per well or per window. Uses: a better-calibrated probability-weighted mean; K-hypothesis features for a reranker/gate; KD targets for the student. (b) A GAN/deep-generative prior for the synthetic well generator, to make structurally realistic hard cases with repeated layers, faults and dip changes. (c) Geology-ordering constraints from DL correlators as a loss term. The EM-based DISTINGUISH workflow itself does not apply, since ROGII has GR only and no UDAR. |
| Error mode targeted | Band confusion (~51%: blend 15%, wrong_mode 24%, multimodal_miss 12%) via explicit multi-hypothesis heads; broad/skewed humps (~13%) via mixture components. Not drift. |
| Gap vs our implementation | Our per-column softmax over 400 TVT levels is already a nonparametric multimodal distribution, so a mixture head does not add expressiveness per column. What we lack: (1) hypotheses that are coherent along the whole lateral. The softmax is per column, so modes are not linked into consistent whole-path alternatives, while MTP outputs K complete curves, each internally consistent, with one probability each. That is a path-level mixture, which our failed post-hoc fb continuity decode tried to recover after the fact but never trained for. (2) A training signal that rewards covering the true band as a separate hypothesis (WTA) rather than hedging (blend). (3) Per-hypothesis probabilities that could be calibrated as a K-way classification (temperature scaling on K logits is far easier than calibrating 345x400 softmaxes). (4) A structured generative prior for synthetic data: we use a procedural generator (85% synthetic), not a learned one. Already have: multimodal posterior, expectation decode (RMSE-consistent), an oracle best-mode headroom figure (4.27 ft). The AcfNet unimodal target failed, which is consistent with needing MORE modes handled explicitly, not fewer. |
| Proposed experiment | E1 (MTP head): in experiments/bilzard/seq_NN_models.py, add an auxiliary head on the U-Net bottleneck/decoder that outputs K=3-4 whole-lateral TVT offset curves (345 columns each), each parameterised as the column-wise soft-argmax over its own 400-level softmax (K extra 400-channel logit maps), plus K well-level logits from global pooling. Loss in seq_NN_train.py: the existing per-column CE plus lambda * [min_k RMSE(curve_k, target) + CE(logits, argmin_k)] with annealed WTA (anneal temperature over epochs to avoid dead heads, per annealed MCL NeurIPS 2024). Decode two ways: (a) the existing expectation decode (control), (b) the calibrated K-weighted mean of curves, with temperature fit on training-fold wells only. Also dump K curves and weights as gate features for the failed ConvNeXt/AnchorCNN gate. Control: the identical 60-epoch ConvNeXt run without the head (5.47 ft), 2 seeds each, holdout 155 wells via seq_NN_holdout_eval.py; success means more than 0.39 ft gain, or oracle-of-K RMSE well below 4.27 ft with weighted-mean RMSE improving. E2 (diagnostic, cheap): from existing posterior dumps (seq_NN_posterior_dump.py --save-all), extract the top-2 path-consistent modes by DP and fit a K=2 weight calibrator (logistic on mode mass, entropy, GR-match score) to measure how much RMSE a perfectly coherent 2-hypothesis representation could gain before building E1. |
| Implementation effort | Medium for E1: a new head plus loss plus decode, about 150-250 lines; WTA training needs care (hypothesis collapse, annealing schedule). Low for E2 (post-hoc on saved posteriors). High for a GAN-based synthetic generator (training a generative model of real TVT-vs-MD structure from limited real wells). |
| Distillation compatibility | Compatible. The student can distil both the per-column posteriors and the K hypothesis curves and weights (KL on K-way weights, RMSE on curves). The head adds negligible inference cost to FastViT (22 ms budget). |
| Error correlation with ours | As an auxiliary head on the same backbone, errors will be highly correlated, so gains must come from calibration and coherence, not diversity. As features for a gate/reranker, the K-curve set exposes mode alternatives the current gate never sees, which is useful even when correlated. |
| Risk | (1) Under pooled RMSE with a calibrated posterior, the K-weighted mean approaches the expectation decode, so gains appear only if WTA training yields better-calibrated mode weights or brings in evidence the per-column head misses. (2) WTA/MTP heads are prone to collapse and dead hypotheses in data-limited regimes (only real wells carry true ambiguity; 85% synthetic may not contain realistic repeated-band confusions). (3) Prior failures of the unimodal target, axial attention and RAFT at 60 epochs suggest 60 epochs may be too short to judge new heads; budget 150 epochs for a final check. (4) All literature evidence is synthetic and GR-only or EM-based with no RMSE-in-feet numbers, so there is no external proof of benefit on ROGII-like data. (5) GAN geomodels need EM look-ahead data that ROGII lacks. |
| Notes | Overlap: A09 (Alyaev multi-realization / ESS 2022) covers the MDN-MTP head in more depth, and B04 covers annealed/resilient MCL theory; A06 covers the EnKF side of DISTINGUISH. The 2503.08509 paper is labelled GenAI but uses GAN plus EnKF plus DP, not LLMs. No LLM-based geosteering interpretation paper with quantitative results was found. Searches for supervised CNN/LSTM 'relative position to target' predictors on ROGII-like GR laterals found mostly vertical-well correlators (ASDNet, semantic segmentation 2024, Processes 2025) and patents (Chevron US 11480045, covered in A04). The strongest transferable idea for Erudition is the path-level multi-hypothesis output with WTA training and calibrated hypothesis weights. |

### Other info

- **Id**: A08
- **Uncertain fields (omitted above)**:
  - code_availability
  - gpu_cost
  - inference_budget
  - metric_and_performance
  - prefix_conditioning

---

## A09 Alyaev et al. multi-realization probabilistic log interpretation (SPWLA 2022) and direct multi-modal inversion (ESS 2022)

### Basics

| Field | Value |
|---|---|
| Name | Alyaev et al.: Direct multi-modal inversion of geophysical logs using deep learning (MDN + multiple-trajectory-prediction loss, ESS 2022) and its sequential version, Sequential multi-realization probabilistic interpretation of well logs (SPWLA 2022) |
| Authors org | Sergey Alyaev (NORCE Norwegian Research Centre) and Ahmed H. Elsheikh (Heriot-Watt University) for ESS 2022; S. Alyaev, A. Ambrus, N. Jahani, A. H. Elsheikh for SPWLA 2022. Funded by the DigiWells SFI (Aker BP, ConocoPhillips, Equinor, and others) |
| Year | 2022 (arXiv preprint Jan 2022; ESS published Sep 2022; SPWLA 63rd Symposium June 2022; NFES/NORCE workshop poster Nov 2022) |
| Venue link | Earth and Space Science 9(9), e2021EA002186, doi:10.1029/2021EA002186 (https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021EA002186; arXiv:2201.01871 https://arxiv.org/abs/2201.01871). SPWLA 63rd Annual Logging Symposium, Stavanger, doi:10.30632/SPWLA-2022-0112 (https://onepetro.org/SPWLAALS/proceedings-abstract/SPWLA22/SPWLA22/D051S022R001/487879). Poster: https://nfes.org/assets/workshop2022/ambrus_sequential_multi_mode_inversion_poster.pdf |
| Evidence level | Peer-reviewed journal (ESS 2022, open access, full text read). SPWLA 2022 paper known only from its abstract plus the NFES 2022 poster (full paper paywalled, not read). |
| Extreme summary | A small CNN reads the lateral-GR minus typewell-GR difference heat map, which is the same object as our gr_abs_diff cost image, and an MDN head outputs M=7 whole stratigraphic curves plus their probabilities in one ~1.5 ms pass. It is trained with Cui et al.'s multiple-trajectory-prediction (MTP) loss: best-of-M L1 regression plus cross-entropy on the winning mode, which avoids MDN mode collapse. The SPWLA version chains windows, seeding new curves from old endpoints and merging or pruning them, so it tracks hundreds of realizations. |

### Problem formulation

| Field | Value |
|---|---|
| Task definition | Stratigraphic inversion for geosteering. Given a horizontal-well log g(x) along MD and a vertical offset-well log f(z) along stratigraphic vertical depth (SVD), find the SVD function b*(x) with f(b*(x)) = g(x): the well's stratigraphic position along MD, which is equivalent to our TVT. The model also extrapolates b(x) ahead of the data (look-ahead prediction). It outputs several likely b(x) curves and their probabilities. |
| Inputs | Gamma ray only in ESS 2022, from the GWC 2020 semi-final synthetic typelog of the Middle Woodford (0.5 ft sampling, normalised to 0-1). The SPWLA/poster version adds rate of penetration (ROP) as a second log pair. The input is the heat map r_ij = g_j - f_i on a 64-cell (32 ft TVD) typewell window x 16 ft of observed lateral. No trajectory or inclination channels (layer-cake assumption, no dip input). No neighbour wells. |
| Target parameterisation | The SVD function b(x) relative to the window start: a 32-point curve (16 ft inversion over observed data plus 16 ft prediction ahead). Every second training sample is re-centred so b(0)=0, which favours continuity with the previous inversion. No explicit dip state; dip, curvature and faults are learned from the synthetic curve generator. |
| Output representation | K hypotheses: M whole curves (M=7 selected; 1-9 tested) plus M unnormalised log-probabilities, as one concatenated vector of size M x (l+ + 1). Each mode is a coherent path over the whole window, not a per-column marginal. No per-mode width is predicted; the kernel width is fixed implicitly by the loss weight alpha_class=0.1. |

### Method

| Field | Value |
|---|---|
| Architecture | Regression head: three 3x3 Conv2D + 2x2 max-pool pairs over the normalised difference image, then two fully connected layers, all ReLU. MDN predictor: a 4096-wide dense ReLU layer and a linear output of M x (32+1) values (dense design adopted from Cui et al. 2019). Tiny by modern standards and not tuned; the authors leave hyperparameter optimisation to future work. Sequential version (poster): the network runs on consecutive windows, each previous realization's endpoint seeds the next window's inversion, coinciding endpoints merge with summed probability, low-probability branches are pruned, and hundreds of realizations are tracked, similar to beam search with learned proposals. |
| Core mechanism | Winner-takes-all training. Only the hypothesis closest to the truth (L1 distance over the curve) gets the regression gradient, so the M heads specialise on distinct zero-isolines of the difference image: repeated-band alternatives, dip up vs dip down, fault throws. A classification head learns how often each specialised hypothesis wins, which gives implicit probabilities learned from the frequency of geological configurations in training. Rupprecht et al. 2017 show this approximates a Voronoi-cell mixture of the conditional density. Collapse is measured, not assumed: 'collapsed modes' are pairs closer to each other than the best mode is to the truth (12.1% at M=7). |
| Training objective | L_MTP = L_class + alpha * L_reg (poster writes alpha on the MAE term; paper alpha_class = 0.1). m* = argmin_m //b_m - b*//_1. L_reg = mean /b_m* - b*/ (MAE, length-normalised), applied only to m*. L_class = -log softmax(p)_{m*}. Adam, lr 5e-5, batch 512, 28 M synthetic training samples, early stopping (patience 3 epochs, 4 validation checks per epoch). 5.2 h on one RTX 2080 Ti. Correlated noise (exponential kernel, correlation length 8 cells) added at train time in Section 5.5. |
| Metric consistent decode | Not RMSE-consistent as published. The reported numbers are oracle best-of-M; picking the highest-probability mode is mode-seeking; the per-mode L1 loss is median-like. A probability-weighted mean of the M curves approximates the posterior mean (RMSE-optimal) only if the mode probabilities are calibrated. The paper's bucketed reliability check shows this roughly holds up to about P35 and drifts above that. |
| Evidence scope | Local window: 16 ft of lateral and a 32 ft typewell window per evaluation. The sequential version accumulates evidence along the lateral only through realization bookkeeping (endpoint seeding, merge and prune), not through a learned long-range receptive field. |
| Prefix conditioning | Implicit. Each window starts from the endpoint of a previous realization (half the training samples are centred at 0), so the known start acts as an anchor. There is no fine-tuning or closed-form correction from known heel labels. |
| Propose vs select | Propose. It emits M candidate paths with scores in one pass, so it fits naturally as a candidate generator for a downstream reranker, gate or decision optimiser. The sequential version adds a hand-coded select step (merge and prune by probability). |
| Physical constraints | Layer-cake assumption: the log depends only on SVD position; constant thickness; faults and dips only as represented in the synthetic curve generator. Geological plausibility is learned from the data prior, not enforced. No ordering, smooth-dip or TVT + Z identity constraints. |

### Data

| Field | Value |
|---|---|
| Dataset | Fully synthetic. 28 M training triples, 12 k validation, 10 k test: a random 64-cell section of the GWC 2020 semi-final typelog (Middle Woodford, South-Central Oklahoma), a random SVD curve with trend 0, and the lateral log generated by linear interpolation of the typelog along the curve. Extra hand-made scenarios: flat, slope (82/98 deg), fault (3.75 ft throw). The SPWLA/poster version uses GWC 2020 GR + ROP with realistic stratigraphy and a fault at 2750 ft MD. |
| Synthetic to real | 100% synthetic, with a single typelog; the lateral log is an exact resample of the offset log (noise-free in the reference model). Section 5.5 shows the model is brittle: the 0%-noise model falls below the constant baseline once test noise exceeds about 2%. Training with 10% correlated noise gives robustness past the 6-7% GWC noise standard, at the cost of 'trusting' the data less. No lateral-vs-typewell GR normalisation or bias mismatch is modelled, and there is no real-data alignment stage. |

### Results

| Field | Value |
|---|---|
| Metric and performance | ESS 2022 test set (cells of 0.5 ft): best-mode MAE 2.34 cells for M=1 (deterministic) vs 1.13 for M=7, i.e. a 52% reduction; 1.60 for M=2, 1.32 for M=3. Mixture NLL is lowest for M=1 (0.365; single-mode predicts averages), and the well-log NLL is best for M=2 and M=7. Fault scenario best-mode MAE 3.08 (M=1) vs 1.40 (M=7). With 1% noise: 5.67 (M=1) vs 2.75 (M=7). Constant-zero baseline: 6.04 cells. Inference 1.5 ms per window. Sequential GWC case (poster): tracks the true curve through a fault, 443 ms total on a standard PC; no RMSE given. |
| Comparability | Low. The numbers are oracle best-of-M MAE in 0.5 ft cells over 32-ft windows (half of which is extrapolation) on synthetic single-typelog data. They are not pooled RMSE of a deployable decode over whole real laterals. The only transferable fact is the relative gain of best-of-M over deterministic (about 2x), which mirrors our own oracle best-mode gap (4.84 deployed vs 4.27 oracle on the 60-ep model). |
| Validation protocol | Random synthetic train/val/test splits generated with the same rules and the same typelog, so there is no spatial or well grouping and no real-well holdout. The fault/slope/flat scenarios are hand-constructed probes. Calibration is checked with probability-bucket reliability on validation and test. |
| Inference budget | About 1.5 ms per window on GPU for ESS; 443 ms for a whole GWC lateral including sequential bookkeeping on a standard PC. Far inside the Kaggle budget; a K-path head added to our U-Net would cost essentially nothing at inference. |

### Ambiguity handling

| Field | Value |
|---|---|
| Repeated layers | The paper's core target. The difference image shows several zero-isolines (e.g. four modes, symmetric around 0 and around 4 ft); separate MTP heads lock onto different isolines, and the classifier ranks them by learned geological frequency. Failure mode shown: the correct band is often not the highest-probability mode (p = 8% and 17% in the worked examples). Even a perfect multi-hypothesis generator leaves the selection problem, which is exactly our wrong_mode + multimodal_miss share. |
| Drift | Handled only through the synthetic curve prior (learned dip up vs down, curvature) and through look-ahead extrapolation. There is no explicit dip state or straight-line drift correction. The re-centring trick favours continuity from the previous endpoint, which can also propagate drift. |
| Uncertainty and calibration | Mode probabilities come from a softmax cross-entropy on the winning index. A reliability analysis over probability buckets (P0-P10 ... P30+) shows good agreement below about P35 and deviation above, within error bars because the high buckets have few samples. No temperature scaling or ECE. Mode width is fixed, not predicted. Collapse rate is reported per M (2.3% at M=2 up to 15.1% at M=9); M was chosen as the largest M whose collapse rate stays below 1/M. |

### Relevance to erudition

| Field | Value |
|---|---|
| Where to use | (1) As an auxiliary K-path head on the ConvNeXt/FastViT decoder: K whole-lateral TVT curves (345 columns) plus K logits trained with MTP alongside the existing per-column softmax loss. (2) As a candidate generator for a whole-lateral reranker scoring each path by full-lateral GR misfit, prefix consistency and single-straight-line drift, which is what our failed per-point gates could not see. (3) As a structured source of mode-level calibration for a probability-weighted mean decode. (4) The sequential merge/prune bookkeeping as a beam-search alternative to our post-hoc forward-backward decode. |
| Error mode targeted | Band confusion (wrong_mode 24%, multimodal_miss 12%, blend 15%): about 51% of squared error. Secondary: faulted or broad-hump cases. |
| Gap vs our implementation | What we already have: the input representation is effectively identical (their r_ij = g - f heat map is our gr_abs_diff/gr_signed_diff cost image, and ours adds 14 more channels), and our 400-bin per-column softmax is strictly more expressive per column than 7 Gaussian-less modes. So the per-column representation is not the gap. What we lack: (a) path-level (joint) hypotheses. Our softmax gives marginals per MD column, so we have no object that says 'band A along the whole lateral, with probability p'. Per-point RMSE with a calibrated marginal is already Bayes-optimal, so the benefit is not a better decode of the same posterior. It is a training signal and a selection interface: MTP forces coherent whole-path alternatives, and a whole-lateral reranker can then use evidence that per-point gates did not see (GR misfit integrated over the full lateral, drift linearity, prefix match). (b) Explicit mode probabilities with a measured collapse rate and a reliability check at mode level; our calibration is only at bin level. (c) Winner-takes-all diversity pressure. Our cross-entropy on per-column bins rewards hedged blends (the 15% blend error), whereas MTP rewards committing. Confirmation: the 2nd-place AnchorCNN + DP carries multiple level hypotheses via P(dTVT/TVT), which is related but per-level rather than per-path. Our oracle best-mode (4.27 vs 4.84 ft) quantifies the ceiling of a perfect selector over modes, and Alyaev reports the same kind of oracle-only win. |
| Proposed experiment | E1 (cheap, diagnostic, no training): from existing posterior dumps (seq_NN_posterior_dump.py --save-all), extract per-well top-K coherent paths by K-best Viterbi or by clustering per-column modes into band-consistent tracks (posterior_fb_decode.py already has the forward-backward machinery). Measure oracle-best-path RMSE vs oracle per-point best-mode (4.27) vs deployed expectation. If path-level oracle is not clearly below expectation, stop. E2 (main): add an MTP head to experiments/arch/seq_NN_models.py (and seq_NN_trf_unet.py for FastViT). Pool the decoder feature map over the TVT axis into a per-column feature, apply a small 1D conv/transformer along MD, and output K=4,6,8 curves of 345 TVT values (parameterised as offsets from the current expectation or from the last known heel TVT) plus K logits from a globally pooled vector. Loss in seq_NN_train.py: existing column CE + lambda * [best-of-K L2 on masked unknown columns + CE on the winner], with lambda in {0.1, 0.3} and L2 (not L1) so each mode tends to the conditional mean. Optionally use annealed or relaxed WTA (epsilon=0.05 to non-winners) against collapse. Config flag in seq_NN_cfg.py (mtp_k, mtp_lambda). Decodes: (i) unchanged expectation from the softmax head (does the auxiliary loss alone help?), (ii) probability-weighted mean of K paths, (iii) 50/50 blend. Control: same config with mtp_lambda=0, same seed(s), 150 epochs, evaluated with seq_NN_holdout_eval.py on the 155 holdout wells. Given the +-0.39 ft single-run noise floor, use at least 2 seeds per arm or seq_NN_robust_compare.py. Log the collapse rate and mode reliability on OOF. E3 (conditional on E1/E2 showing oracle headroom): a whole-lateral reranker (logistic or LightGBM) over the K paths with per-path features: integrated GR misfit, prefix agreement, straight-line drift residual, mode logit. Train on OOF only, and output soft weights for a weighted mean, not a hard pick. |
| Implementation effort | Medium. The head and loss are about 150 lines and the Zenodo MTP loss can be adapted. The work is in the path parameterisation over masked or unknown columns, evaluation plumbing and multi-seed runs. E1 is low effort. |
| Gpu cost | The head adds under 5% FLOPs. Training cost is the same as a baseline run (about the cost of our 150-epoch ConvNeXt run on the RTX 5060 8 GB); two arms x 2 seeds is about 4 baseline runs. Inference cost is negligible. E1 is CPU only. |
| Distillation compatibility | Good. The student can copy the K logits (KL on the softmax over modes) and the K curves (L2 after Hungarian or identity matching, since the heads are index-stable once trained). Alternatively, render the K paths plus probabilities into a per-column mixture and add it to the existing posterior KD target. The student's own softmax head is unchanged. |
| Error correlation with ours | Moderate. It shares the backbone and input channels, so errors in the evidence stay correlated; the diversity comes from path-level commitment vs hedged marginals. As a blend member it is most useful where the softmax blends two bands (the blend 15%), and least useful for drift. |
| Risk | (1) It does not add evidence. Per the metric fact in our context, band-confusion error falls only via more evidence or better calibration, and MTP helps only if the whole-path structure or reranker extracts evidence the U-Net ignores. (2) The published gains are oracle best-of-M; the deployable decode (highest-probability mode) picked the wrong band in 2 of 3 worked examples. (3) Learned gates and selectors already failed three times (per-point ConvNeXt/AnchorCNN gate, XY geo-prior pick, forward-backward continuity), and the 3rd-place CV-to-private gap (5.29 to 5.84) warns about selection overfitting. (4) Mode collapse and heads idling (12% at K=7). The auxiliary loss can hurt the main head at 60 epochs, as happened with AcfNet, axial attention and RAFT. (5) Real laterals are much longer than 16 ft windows and have prefix labels, so K fixed paths over 345 columns may need too many modes to cover per-segment band switching. |
| Notes | A09 is the closest in-domain predecessor of a multi-hypothesis head and uses the same difference-image input as our cost volume, so it validates the idea but not its RMSE value. The honest reading is that the paper shows multi-modality exists and can be enumerated cheaply; it does not show that enumeration improves a point estimate. For us the decisive quantity is the gap between oracle-best-path and expectation on real holdout wells (E1) before any training. The sequential merge/prune scheme is a hand-coded particle filter with learned proposals, which links to A05 (PF) and B06 (differentiable PF). Related transferable work: B04 (annealed/resilient MCL fixes WTA collapse and gives calibrated scores), B03 (multimodal stereo decoding). Poster title reference: 'AI-based multi-modal interpretation of logs for ahead-of-bit probabilistic ROP prediction' (Ambrus et al., NFES 2022). |

### Other info

- **Uncertain fields (omitted above)**:
  - code_availability
  - decoding

---

## A10 CNN / Siamese well-log depth matching (1D-CNN vs cross-correlation 2022, late-fusion 2024, Siamese SPWLA 2025, deep-Q 2021, SPWLA depth-shifting contest)

### Basics

| Field | Value |
|---|---|
| Name | CNN / Siamese well-log depth matching family: 1D-CNN vs cross-correlation (Petrophysics 2022); late-fusion multimodal 1D-CNNs (Geophysical Prospecting 2022/2024); Siamese NN depth alignment (SPWLA 2025); Deep Q-learning multi-well depth matching (Petrophysics 2021); SPWLA PDDA 2023 depth-shifting contest; plus Deep Hierarchical Graph Correlation (triplet 1D-CNN embeddings + DP, JMSE 2026) and ROGII 3rd-place Adaptive 1D SDF as in-domain learned correlation |
| Year | 2021 (Deep Q), 2022 (1D-CNN vs CC; late fusion online), 2023 (PDDA contest; summary paper Petrophysics 67(2), 2026), 2024 (late fusion in print, GP 72(1)), 2025 (Siamese SPWLA), 2026 (DHGC; ROGII Kaggle) |
| Venue link | Petrophysics 63(1):12 (2022) https://onepetro.org/petrophysics/article/63/01/12/477822 ; Geophysical Prospecting 72(1):155-182, doi:10.1111/1365-2478.13200 https://onlinelibrary.wiley.com/doi/full/10.1111/1365-2478.13200 ; SPWLA-2025-0048 https://onepetro.org/SPWLAALS/proceedings/SPWLA25/SPWLA25/D041S010R002/678930 ; Petrophysics 62(4):353 (2021) https://onepetro.org/petrophysics/article-pdf/62/04/353/2459476/spwla-2021-v62n4a1.pdf ; contest repo https://github.com/pddasig/Machine-Learning-Competition-2023 and summary PJV67N2-2026a12 https://www.spwla.org/SPWLA/Publications/Publication_Detail.aspx?iProductCode=PJV67N2-2026a12 ; DHGC https://doi.org/10.3390/jmse14010066 ; ROGII 3rd place writeup (local 3rd.txt, Adaptive 1D SDF section) |
| Evidence level | Peer-reviewed journals (Petrophysics 2021/2022, Geophysical Prospecting, JMSE); the SPWLA 2025 conference paper is known from its abstract only; a public competition with leaderboard (PDDA 2023); a Kaggle writeup (ROGII 3rd). Full texts were mostly paywalled or 403; details come from abstracts and search snippets. |
| Extreme summary | This literature replaces hand-made similarity (cross-correlation, sample-wise differences) with learned similarity. The forms are: 1D-CNNs that regress a bulk shift between two log windows, late-fused across log types; Siamese or triplet 1D-CNN embeddings whose window-to-window similarity feeds DTW or DP; and RL agents that pick shifts. The consistent lesson, including the PDDA contest where all top 5 teams used ML + DTW, is learned local cost + global DP alignment. Its target, though, is small shifts between logs of the same or a nearby well, not 400-level repeated-band ambiguity. |

### Problem formulation

| Field | Value |
|---|---|
| Task definition | Depth matching / alignment: estimate the depth shift (bulk or depth-varying) that aligns a log to a reference log, e.g. LWD vs wireline in the same well (Torres Caceres; Siamese: GR from different logging runs in one well), multiple logs to a reference GR (PDDA contest: shift RHOB, NPHI, RD to GR), or multi-well correlation (Deep Q, DHGC). This is closely analogous to our per-MD-column choice of typewell TVT level, but the shifts are small and the mapping is monotone. |
| Inputs | Pairs of 1D log windows. Torres Caceres: seven log types (GR, resistivity, P/S sonic, density, neutron, PEF) from 6 Ivar Aasen wells (Norwegian North Sea). Siamese: GR vs GR. PDDA: reference GR + misaligned RHOB/NPHI/RD from 9+3 wells. DHGC: GR windows from 8 North Sea wells. ROGII 3rd SDF: lateral GR + trajectory (4-block TCN) and typewell GR/TVT context (3-block TCN). |
| Target parameterisation | Bulk shift scalar (1D-CNN per log type; late fusion aggregates shifts); a warping path over a learned cost matrix (Siamese/DHGC + DP); discrete shift actions (Deep Q); corrected log values + per-sample shift (PDDA). ROGII SDF: classification over 256 typewell TVT candidates per lateral row, with the expected TVT as output. |
| Output representation | Point shift estimate or a single warping path. ROGII SDF gives a per-row softmax over TVT candidates, i.e. the same per-column distribution representation as ours. |

### Method

| Field | Value |
|---|---|
| Core mechanism | Learn a similarity metric over windows (receptive field of tens of samples) that is invariant to tool-dependent scale, bias, noise and small stretch, and discriminates geologically distinct windows that look alike sample by sample. Metric learning (Siamese/triplet/contrastive, Rank-N-Contrast in ROGII SDF) makes the similarity reflect correspondence distance rather than raw amplitude difference. A global DP/DTW then enforces path consistency. The Siamese abstract explicitly claims robustness where cross-correlation fails on repetitive lithologies (unverified beyond the abstract). |
| Decoding | Argmin or regressed shift; DP/DTW optimal path over the learned cost matrix; greedy policy rollout (DQN); ROGII SDF expectation over the candidate softmax. |
| Metric consistent decode | Mostly mode-seeking (DTW/DP single path, argmin shift). ROGII SDF expectation is RMSE-consistent. For us the embedding would be an input channel, not a decode, so this concern moves to our unchanged expectation decode. |
| Evidence scope | Local windows scored pairwise plus global path via DP/DTW (whole log). ROGII SDF: whole packed lateral sequence vs typewell context. |
| Prefix conditioning | None in the depth-matching papers. ROGII SDF estimates well-specific GR bias and scale from the known prefix, an explicit in-domain use of heel labels for the similarity function. |
| Propose vs select | Mostly single output. In the ROGII use the learned correlation model was a weak standalone candidate (not the strongest) kept for diversity in a softmax gate. For us the natural slot is as an extra input channel (a learned cost slice) rather than as a candidate. |
| Physical constraints | Monotone depth warping (DTW/DP), bounded shifts; bulk-shift assumption in the Torres Caceres CNN. None of these methods handle non-monotone laterals that go up and down through strata; ROGII SDF handles this by per-row classification without monotonicity. |

### Data

| Field | Value |
|---|---|
| Synthetic to real | Semi-synthetic shifting of real aligned logs is the standard: apply known shifts to aligned logs to create labels (late fusion; PDDA training wells are aligned, so shifts can be simulated). Siamese is claimed label-free (self-supervised pairs). No real-to-real domain gap handling reported beyond normalisation. |

### Results

| Field | Value |
|---|---|
| Comparability | Very low for the depth-matching papers: different task (small monotone shifts, same or nearby well), different metrics (Pearson, NMSE, MAD), tiny well counts. ROGII 3rd SDF is directly comparable in task and data: a learned-embedding correlation volume gave a weaker standalone model than the physics/tracker candidates, and its value was diversity. |
| Validation protocol | Well-level holdout (2 inference wells in Torres Caceres; 3 hidden test wells in PDDA), which is fold-safe at the well level but tiny. Siamese: single-well demonstration. ROGII 3rd: GroupKFold OOF, with a known CV 5.29 vs private 5.84 gap for the full gated system. |
| Inference budget | Cheap. A 1D-CNN or TCN embedding of 345 lateral windows and 400 typewell windows plus a 345x400 cosine volume costs milliseconds on GPU and under a second on CPU. The Siamese paper notes it is slower than CC, which is irrelevant at our scale. |

### Ambiguity handling

| Field | Value |
|---|---|
| Repeated layers | This is the stated motivation (the Siamese abstract says cross-correlation gives false matches on repetitive lithologies). But a window embedding can only separate repeated bands whose windows differ somewhere inside the receptive field. Truly self-similar bands stay ambiguous under any local similarity, which is why every method pairs it with a global DP. In our setting the U-Net already aggregates local evidence over a large receptive field, so the marginal gain on band confusion depends on whether a pretrained metric captures invariances (GR bias/scale per well, stretch from relative dip) that the U-Net fails to learn from the raw /GR_lat - GR_tw/ slice. |
| Drift | Not addressed directly. Invariance to GR scale and bias in a learned embedding could reduce drift caused by GR calibration mismatch, which ROGII SDF handles explicitly by estimating bias and scale from the prefix. Geometric drift (slow dip error) is outside the scope of similarity learning. |
| Uncertainty and calibration | None of the depth-matching papers report calibrated uncertainty. ROGII SDF uses a fixed temperature 1.75 softmax, which is a crude calibration. |

### Relevance to erudition

| Field | Value |
|---|---|
| Where to use | As extra channels in the 16-channel cost image built in experiments/arch/seq_NN_dataset.py: a learned cosine-similarity slice between embedded lateral windows (along MD) and embedded typewell windows (along TVT), next to or replacing gr_abs_diff (+-0.25 ft shifted variants) and gr_signed_diff. Alternatively, as a learnable feature-extractor stage inside the model (stereo-style learned correlation volume, see B01) computed end-to-end with the U-Net. |
| Error mode targeted | Band confusion (about 51%), specifically cases where amplitude-level differences look alike but window shape differs. Secondarily, drift induced by GR calibration mismatch between lateral and typewell. |
| Gap vs our implementation | Today the cost image uses sample-wise hand-made interactions: _make_gr_abs_diff_channel computes /(GR_lat[i] - GR_tw[j]) / std/ for every MD column i and TVT level j (plus +-0.25 ft shifted typewell variants, signed diff, ratio/compression residuals, /TVT_tw - visible TVT/, PF and XY-neighbour channels). Window-level similarity is learned only implicitly by the ConvNeXt/FastViT encoder from these slices, trained on our small real set plus 85% synthetic. What we do not have: (a) a similarity metric pretrained with a metric-learning objective (triplet, InfoNCE, Rank-N-Contrast) that explicitly contrasts the true TVT level against hard repeated-band negatives; (b) invariance to per-well GR gain and bias learned by design (we use a global diff_std normaliser; the ROGII 3rd SDF estimated per-well bias and scale from the prefix); (c) stretch/compression invariance, since a lateral MD window maps to a typewell TVT interval whose length depends on relative dip, which a 1D embedding trained with stretch augmentation handles and a pointwise difference cannot; (d) learnable, not fixed, feature channels, which is the core stereo lesson (GC-Net/PSMNet correlate learned features, not raw pixels). What we already do: the U-Net is effectively the learned aggregator plus global reasoning, which plays the DP role these papers bolt on, so any gain must come from the feature/metric side only. |
| Proposed experiment | E1 (pretrain the embedding, fold-safe): new module experiments/arch/seq_NN_log_embed.py. Two 1D TCN/CNN encoders (or one shared, Siamese) over GR windows of W in {32, 64} samples at our downsampled MD and TVT resolution, 32-64-dim L2-normalised output. For each training lateral column i with known TVT t_i, the positive is the typewell window centred at t_i and the negatives are all other 399 TVT levels. Use InfoNCE over the column, which is a per-column softmax, plus a Rank-N-Contrast or distance-weighted soft target so near levels are not penalised like far bands. Mine hard negatives from the top-k gr_abs_diff minima at repeated bands. Augment with random per-well GR gain/bias, noise and lateral stretch in [0.5, 2] to emulate relative dip. Train only on training-fold wells (same split as the U-Net) plus synthetic. Report the embedding-alone expectation RMSE on holdout as a diagnostic. E2 (channel ablation): add channel names 'gr_emb_cos' (and optionally 'gr_emb_cos_w32', 'gr_emb_cos_w64') to seq_NN_dataset.py next to gr_abs_diff, computing the cosine volume from the frozen encoder at data-prep time and caching it. Config in seq_NN_cfg.py. Arms: (A) control = current 16 channels; (B) +gr_emb_cos; (C) gr_emb_cos replacing gr_abs_diff and its shifted variants. ConvNeXt-small, 150 epochs, 2 seeds each, evaluated with seq_NN_holdout_eval.py on the 155 holdout wells and seq_NN_robust_compare.py (noise floor +-0.39 ft). Break down the delta by error anatomy (wrong_mode / blend / drift) with posterior_analysis.py. E3 (only if B beats A): an end-to-end version where the encoder is trainable inside seq_NN_models.py (correlation-volume layer before the U-Net stem, initialised from E1), then distil into FastViT as usual. |
| Implementation effort | Medium. The encoder and InfoNCE training are about 300 lines; the channel plumbing in seq_NN_dataset.py already has a pattern for new cost channels; caching the volumes avoids retraining cost per epoch. The end-to-end E3 is medium-high because of memory for a 345x400xD volume. |
| Gpu cost | E1: small, well under an hour on the RTX 5060 8 GB (tiny 1D encoders, 345x400 logits per lateral). E2: the same as baseline training per arm (3 arms x 2 seeds = 6 baseline-scale runs), with one extra input channel. E3: a D=32 correlation volume at 345x400 is about 4.4 M floats per sample, which fits in 8 GB at modest batch size. |
| Distillation compatibility | Fully compatible: the embedding channel is an input feature, computed identically for teacher and student, so KD on output posteriors is unchanged. For E3, the student can share the frozen encoder. |
| Error correlation with ours | Low diversity if used as a channel (same model, same decode); its purpose is to improve the single model, not to be a blend member. As a standalone candidate (embedding-only expectation), ROGII 3rd shows such a model is weak but decorrelated. Our prior learned-gate failures argue against using it as a gated candidate. |
| Risk | (1) The U-Net may already learn equivalent window features from gr_abs_diff; the 1D-CNN vs CC paper found CNN only on par with plain cross-correlation, and ROGII 3rd's learned-correlation model was the weakest candidate. (2) Pretraining data is the same limited set (15% real), so the metric can overfit to training typewells; fold leakage is a risk if the encoder sees holdout wells. (3) Local similarity cannot resolve genuinely self-similar bands, and the band-confusion share needs more evidence, which an embedding of the same GR does not add. (4) Depth-matching results transfer poorly: small monotone shifts within the same well are a different regime from choosing among 400 TVT levels on a non-monotone lateral. (5) Negative results at 60 epochs for other architecture changes (axial attention, RAFT) show that input or architecture tweaks are often within the +-0.39 ft noise floor, so a multi-seed comparison is needed. |
| Notes | The strongest in-domain evidence is not the depth-matching papers but ROGII 3rd place's Adaptive 1D SDF (TCN encoders + cosine correlation volume + Rank-N-Contrast + prefix-estimated GR bias/scale), which confirms that the idea runs on this exact data and is diverse but weak alone. The PDDA 2023 result (all top-5 are DTW hybrids) and DHGC (triplet embeddings + DP) confirm the 'learned local cost + global alignment' template, which in our pipeline corresponds to 'learned cost channel + U-Net'. Siamese SPWLA 2025 authorship (Acharya, Fabian, Yazidi, Westeng) overlaps with the NTNU/OsloMet BRU21 group behind Torres Caceres. Related items: B01 (stereo learned features and correlation volumes), A11 (pretrained well-log foundation encoders as an alternative source of embeddings), A03 (DTW baseline). Another potentially relevant paper not read: 'Attention-driven convolutional neural network for automated well log depth alignment' (2026, ScienceDirect S2590123026009382). |

### Other info

- **Uncertain fields (omitted above)**:
  - architecture
  - authors_org
  - code_availability
  - dataset
  - metric_and_performance
  - training_objective

---

## A11 Transformer / foundation models for well logs (WLFM, LithoFormer, TS foundation model for logs, TGS promptable model, CMT-Hiformer)

### Basics

| Field | Value |
|---|---|
| Name | Transformer / foundation models for well logs: WLFM (USTC 2025), LithoFormer (2026), TimeGPT for well logs (2024), TGS promptable ViT-MAE well-log foundation model (IMAGE 2025), CMT-enhanced Hiformer (Processes 2025) |
| Year | 2024 (TimeGPT well logs, Dec 2024), 2025 (WLFM Sep 2025; TGS IMAGE 2025; CMT-Hiformer Apr 2025), 2026 (LithoFormer Jul 2026) |
| Venue link | WLFM arXiv 2509.18152 https://arxiv.org/abs/2509.18152 ; LithoFormer arXiv 2607.22804 https://arxiv.org/abs/2607.22804 ; 'Leveraging Time-Series Foundation Model for Subsurface Well Logs Prediction and Anomaly Detection' arXiv 2412.05681 https://arxiv.org/abs/2412.05681 ; TGS 'Well log foundation model - making promptable AI models for interpretation', SEG/AAPG IMAGE 2025 expanded abstract https://www.tgs.com/technical-library/well-log-foundation-model-making-promptable-ai-models-for-interpretation and https://pubs.geoscienceworld.org/segeab/proceedings-abstract/SEGEAB.44/1/2910/725127 ; CMT-Hiformer, Processes 2025 https://doi.org/10.3390/pr13051288 ; related: Deep Hierarchical Graph Correlation (CNN + DP well-log alignment), JMSE 14(1):66 2026 https://doi.org/10.3390/jmse14010066 |
| Evidence level | WLFM, LithoFormer, TimeGPT-logs: arXiv preprints (abstract + HTML read for WLFM). CMT-Hiformer: peer-reviewed MDPI Processes (abstract/snippet only; full text 403). TGS: industry conference expanded abstract (summary page only). None evaluated on horizontal-well TVT/geosteering. |
| Extreme summary | Well-log transformers are now pretrained on 1k to 1.1M vertical wells with masked-patch reconstruction (WLFM VQ tokens + stratigraphy-aware cross-well contrastive loss; TGS ViT-MAE) and fine-tuned for lithology, porosity, formation tops or zonation, often with geology-informed losses (superposition/ordering penalties in LithoFormer and CMT-Hiformer). They give layer-aware GR embeddings and ordering constraints, but all target vertical-well interpretation, not lateral-to-typewell alignment. |

### Problem formulation

| Field | Value |
|---|---|
| Task definition | Vertical-well interpretation: per-depth lithology/zonation classification, porosity regression, formation-top depth prediction (TGS, prompted by formation name), missing-log imputation, anomaly detection, and cross-well stratigraphic correlation via layer-aware embeddings (WLFM) or boundary prediction (CMT-Hiformer, LithoFormer). Not geosteering: no model predicts TVT of a lateral against a typewell. |
| Inputs | Multi-curve logs in depth: GR, SP, AC (sonic), resistivity, density, neutron (WLFM uses GR/SP/AC among others); TGS uses 1.1M North American logs of many curve types; TimeGPT uses single-curve series. No trajectory, no typewell pairing, no XY neighbours except WLFM's cross-well positive-pair mining by relative depth. |
| Target parameterisation | Per-depth class labels (lithology, zone), per-depth continuous properties, boundary probability per depth (LithoFormer decoupled head), formation-top depth scalar given a prompt (TGS). None parameterise TVT or relative geologic time directly, although WLFM's contrastive positives are defined by relative-depth alignment across wells, i.e. an implicit RGT. |
| Output representation | Per-depth softmax (zonation), per-depth regression, boundary probability sequence, scalar depth per prompted top; WLFM also outputs discrete geological tokens and embeddings reusable as features. |

### Method

| Field | Value |
|---|---|
| Architecture | WLFM: patch the multi-curve log, encode, vector-quantise (z_q = argmin //z_e - e_k//) into a learned geological codebook, transformer encoder pretrained with masked-token modelling plus stratigraphy-aware InfoNCE contrastive loss, then multi-task heads with few-shot fine-tuning (RTX A6000, batch 256). LithoFormer: Seq2Seq transformer on the whole log in one pass, channel-independent PatchTST backbone with rotary positional embeddings (RoPE), decoupled multi-task head for zonation and boundary probabilities. TGS: 60M-parameter 1D vision transformer trained as ViT-MAE (masked autoencoding of log patches) on 1.1M logs for imputation, then a promptable head that takes a formation token and regresses its top depth; fine-tuned on 271,972 tops from 44,062 Permian wells, 37 formations. TimeGPT: generic pretrained time-series transformer used zero-shot/fine-tuned for forecasting and anomaly detection. CMT-Hiformer: hybrid CNN-meets-transformer (CMT) blocks inside a hierarchical multi-scale transformer (Hiformer) for per-depth stratigraphic labelling. |
| Core mechanism | Self-supervised pretraining on many unlabeled wells gives an encoder that maps GR texture to layer identity (WLFM reports emergent layer awareness: same-layer patches cluster across wells, ARI 0.78 and 82.3% purity without lithology labels). The contrastive term explicitly pulls together patches from the same stratigraphic interval in different wells, which is exactly the invariance a lateral-to-typewell matcher needs (same layer, different tool calibration/noise). Geology-informed losses (Law of Superposition, ordering) remove order violations that per-depth classifiers make. |
| Training objective | WLFM: masked-token prediction over VQ codes + InfoNCE stratigraphy-aware contrastive (positives by relative-depth alignment within tolerance, filtered by low-frequency similarity, in-batch negatives) + downstream task losses. LithoFormer: zonation CE + boundary probability loss + geology-informed penalty enforcing superposition/order. TGS: MAE reconstruction, then depth regression per prompt. CMT-Hiformer: CE plus a geological constraint regulariser with tunable weights that promotes correct boundary placement. TimeGPT: generic forecasting loss (pretrained by vendor). |
| Decoding | Argmax per depth or boundary thresholding (LithoFormer, CMT-Hiformer); direct regression (TGS, TimeGPT); embeddings for clustering (WLFM). |
| Metric consistent decode | Mode-seeking (argmax/boundary picks) for the classification models; regression heads approximate the mean. Irrelevant for us if only the encoder is reused as a feature extractor, because our expectation decode stays unchanged. |
| Evidence scope | Single vertical well per forward pass (whole log in LithoFormer, windows elsewhere); WLFM pretraining uses cross-well pairs, but inference is single-well. |
| Prefix conditioning | Not applicable (no known-prefix concept). TGS prompting is the closest analogue: the geologist's prompt conditions the output. |
| Propose vs select | Single output. As an extra input channel or embedding it would feed our U-Net, not act as a separate ensemble member. |
| Physical constraints | Superposition / stratigraphic order penalties (LithoFormer, CMT-Hiformer); none on dip, trajectory or TVT+Z identities. |

### Data

| Field | Value |
|---|---|
| Synthetic to real | All real vertical-well data; no synthetic stage. For us the domain gap is large: lateral GR sampled along MD through shallow-dipping strata versus vertical GR in depth; ROGII GR comes with per-well calibration offsets. |

### Results

| Field | Value |
|---|---|
| Comparability | Not comparable: vertical-well classification accuracy, boundary error or porosity MSE; no pooled TVT RMSE, no laterals, no typewell alignment. |
| Validation protocol | WLFM uses well-level splits and 5 seeds; others unclear from abstracts. No spatially grouped CV reported; no leaderboard-style blind test. |
| Inference budget | Encoders of 10-60M parameters over a 1D sequence: milliseconds per well on GPU, seconds on CPU; a precomputed per-level typewell embedding costs nothing at inference. Fits both rig and Kaggle budgets. |

### Ambiguity handling

| Field | Value |
|---|---|
| Repeated layers | Indirect. A contrastively trained encoder can separate layers that look alike at raw GR amplitude but differ in multi-scale context (WLFM layer-awareness), which could sharpen the matching cost between similar bands. However, truly repeated cycles (same GR signature at different depths) are indistinguishable by any local encoder with a limited receptive field; only longer context or other evidence (neighbour wells, trajectory) separates them. Ordering losses (LithoFormer) prevent order violations in vertical wells; the analogue for us is monotone consistency of band identity along the lateral, which our per-column softmax does not enforce. |
| Drift | Not addressed. Encoders provide texture similarity, not dip estimates. |
| Uncertainty and calibration | No calibration analysis reported by any of the five; WLFM reports seed std only. Boundary probability heads (LithoFormer) are uncalibrated probabilities. |

### Relevance to erudition

| Field | Value |
|---|---|
| Where to use | (1) As a learned matching-cost channel: pretrain a small 1D GR encoder self-supervised (masked patch + cross-well contrastive) on all typewell GR and lateral GR (real + synthetic), then add channels such as cosine similarity between the lateral-GR window embedding at each MD column and the typewell embedding at each TVT level to the 16-channel cost image, next to gr_abs_diff. (2) Replace tw_gr / gr raw channels with a few PCA components of embeddings (multi-scale context). (3) Borrow the geology-informed-loss idea as a band-consistency regulariser on the U-Net output. The foundation models themselves (weights) are not usable: none are public, and all are vertical-well. |
| Error mode targeted | Band confusion (~51%) via a more discriminative multi-scale similarity cost; possibly broad humps (~13%) via sharper cost. Not drift. |
| Gap vs our implementation | Our cost image uses hand-crafted channels only: tw_gr, gr, gr_abs_diff, windowed GR stats (std, slope, quadratic fit a/b/c, rmse, isnan rate, first-last delta), tw_tvt_rel, seen_tvt_rel, tw_seen_tvt_abs_diff, z_diff (experiments/bilzard/seq_NN_cfg.py unet_static_channels). The only learned similarity is inside the ConvNeXt/FastViT 2D backbone, which is ImageNet-pretrained, not pretrained on log signals. Missing: (a) any log-domain self-supervised pretraining; (b) a learned, calibration-invariant lateral-vs-typewell similarity channel with a long receptive field (hundreds of ft of GR context on both sides); (c) an explicit cross-well invariance objective (WLFM's contrastive positives by relative depth map directly onto our labelled wells: the lateral GR sample at TVT t and the typewell GR at TVT t are a positive pair). Note the 1st place writeup reports transformer backbones lost to ConvNeXt and synthetic-first two-stage training lost to joint training, so the gap is specifically the 1D log encoder as an input feature, not a transformer replacement of the 2D backbone (our axial attention also failed). |
| Proposed experiment | E1 (cheap, decisive): train a 1D GR encoder (e.g. 4-6 layer 1D ConvNeXt or small transformer, receptive field ~256 ft, 64-d output) with a supervised contrastive/InfoNCE objective on training-fold wells only: anchor = lateral GR window centred at MD column i, positive = typewell GR window centred at the true TVT(i), negatives = typewell windows at other TVT levels, especially the other repeated-band candidates (hard negatives mined from our posterior dumps via seq_NN_posterior_dump.py --save-all humps), plus masked-patch reconstruction on all typewells and synthetic wells as an auxiliary. Add new module experiments/bilzard/seq_NN_gr_encoder.py; compute channel 'gr_emb_cos' (345 x 400 cosine similarity map) in seq_NN_dataset.py next to gr_abs_diff; register it in unet_static_channels in seq_NN_cfg.py (17 channels). First diagnostic without retraining the U-Net: on 155 holdout wells, for multimodal columns, check whether argmax of gr_emb_cos over the sig humps picks the truth hump more often than the dominant hump (76.2% in posterior_geo_pick.py) -- if it does not beat 76.2%, stop. E2: 60-epoch ConvNeXt with the extra channel vs control 5.47 ft, 2 seeds, same folds; then 150 epochs + FastViT KD if positive (>0.39 ft). Fold safety: the encoder must be trained only on training-fold wells for each holdout split, else it leaks labels. |
| Implementation effort | Medium: ~300 lines (encoder, contrastive data sampler, channel computation, caching of typewell embeddings); fold-safe retraining of the encoder per split adds bookkeeping. |
| Gpu cost | Encoder pretraining: under 1-2 GPU-hours on RTX 5060 8 GB (1D, small). Channel computation: a 345x400 dot product per well, negligible. U-Net retrain: same as baseline 60/150-epoch runs. |
| Distillation compatibility | Compatible: the channel is an input, so the FastViT student receives it too; adds one encoder forward pass (few ms) to student inference. |
| Error correlation with ours | Same backbone and decode, so errors stay highly correlated; gains, if any, come from better evidence in the cost volume, not diversity. A standalone embedding-DP tracker using the learned cost could be a more diverse ensemble member. |
| Risk | (1) Our 2D backbone already learns local GR similarity from gr/tw_gr/gr_abs_diff, so a 1D encoder may add nothing (1st place: PF features stopped helping once backbones got stronger). (2) Repeated bands are by definition similar in GR; no encoder with bounded context separates identical cycles, so the gain is capped at the fraction of confusions caused by weak similarity rather than true self-similarity. (3) Only ~773 real wells (54 master typewell systems) supply real positives; synthetic positives may teach synthetic-generator artefacts. (4) Label leakage if the encoder is not fold-safe. (5) No public weights; external foundation models are vertical-well and multi-curve, while ROGII has GR only. |
| Notes | 1st place writeup (1st.txt, 'What didn't work'): transformer backbones and simulation-first pretraining lost. 4th place used a synthetic-only 'foundation model' (5.9 ft CV without real finetuning) as an ensemble member for diversity -- that is the only in-competition foundation-model evidence, and it is a 2D image model, not a log encoder. 5th place synthetic-only model scored 6.34 private. Overlaps A10 (Siamese depth matching) -- the proposed encoder is essentially a Siamese similarity learner; B01 (learned stereo features) is the same idea from stereo. TimeGPT-style forecasting foundation models are not useful here (forecasting GR is not the task). Related CNN+DP alignment: Deep Hierarchical Graph Correlation (JMSE 2026). |

### Other info

- **Id**: A11
- **Uncertain fields (omitted above)**:
  - authors_org
  - code_availability
  - dataset
  - metric_and_performance

---

## A12 Multi-well correlation via relative geologic time and loop consistency (Sylvester 2023, graph/sequence methods)

### Basics

| Field | Value |
|---|---|
| Name | Multi-well correlation via relative geologic time and loop consistency: Sylvester 2023 ChronoLog (pairwise DTW + least-squares RGT), Wheeler & Hale 2014 simultaneous correlation, incremental/hierarchical multi-well correlation (2018, 2025), and the in-competition analogue (ROGII 3rd place sibling-lateral reference GR) |
| Year | 2014 (Wheeler & Hale), 2018 (incremental correlation), 2022 (Sylvester EarthArXiv preprint), 2023 (Sylvester, Basin Research 35:1961), 2025 (correlation-path paper; ROGII competition) |
| Venue link | Sylvester 2023, 'Automated multi-well stratigraphic correlation and model building using relative geologic time', Basin Research https://onlinelibrary.wiley.com/doi/10.1111/bre.12787 (preprint https://eartharxiv.org/repository/view/3508/); Wheeler & Hale 2014, 'Simultaneous correlation of multiple well logs', SEG Expanded Abstracts 618-622 https://onepetro.org/SEGAM/proceedings-pdf/SEG14/All-SEG14/SEG-2014-0227/1513284/seg-2014-0227.pdf ; incremental correlation https://www.researchgate.net/publication/325604900 ; correlation path 2025 https://link.springer.com/chapter/10.1007/978-3-031-92870-3_5 ; ROGII 3rd place writeup (local C:/Users/Aadinarayan R/Desktop/CSE/projects/Erudition/3rd.txt); 1st place XY-neighbour section (local 1st.txt); 2nd place 'explicit formation modeling' rejection (local 2nd.txt) |
| Evidence level | Sylvester and Wheeler & Hale: peer-reviewed (abstract/preprint summary read; full text 403). Incremental correlation: peer-reviewed (snippet only). ROGII 3rd place: competition writeup with private-LB ablations (read in full locally). |
| Extreme summary | Instead of correlating each well to one reference, run many pairwise DTW correlations between neighbouring wells (e.g. over a triangulated well graph), then solve one global least-squares problem for a per-well depth-to-RGT shift so all pairwise ties are satisfied jointly; this closes loops and removes the path dependence of chained DTW. In ROGII, the working version of 'use the neighbours' was not an XY surface prior but using sibling laterals' GR binned in TVT as additional reference curves, which gave the 3rd place team their largest private-LB gains (-0.10 to -0.13 ft). |

### Problem formulation

| Field | Value |
|---|---|
| Task definition | Classical: assign every depth sample in every well a relative geologic time (chronostratigraphic position) so that correlative layers share the same RGT across the basin. For ROGII (our reading): predict TVT along a lateral, where TVT against a typewell is exactly an RGT coordinate; labelled training laterals near or sharing a typewell with the target provide extra GR(TVT) references and extra structural constraints. |
| Inputs | Classical: GR (or other) logs of hundreds of vertical wells plus their map locations. ROGII 3rd place: target lateral GR, its typewell GR, and GR of training laterals sharing the same typewell (siblings), binned by their known TVT; optionally distance-weighted by XY. |
| Target parameterisation | Relative geologic time per sample (monotone in depth for vertical wells); per-well shift function depth -> RGT. For laterals, TVT (or u = TVT + Z in structural coordinates) plays the role of RGT and is not monotone in MD. |
| Output representation | Classical: a single consistent RGT field (point estimate) plus chronostratigraphic (Wheeler) diagram and 3D model. 3rd place: HMM/PF posterior mean paths using a mixture of references. |

### Method

| Field | Value |
|---|---|
| Core mechanism | Consistency through joint optimisation: each pairwise correlation is noisy and may lock onto a wrong repeated cycle, but a wrong tie cannot be satisfied around every loop of the well graph, so the least-squares solution down-weights it (loop closure). In ROGII terms, a sibling lateral that crossed the same interval gives an independent, calibrated GR(TVT) sample of the same stratigraphy acquired with lateral-tool characteristics, so the band ambiguity of the typewell (which may be a different vintage/tool) is reduced, and several references vote on the band. |
| Training objective | No learning: DTW cost (squared/abs log difference) per pair, then an L2 least-squares misfit over all pairwise ties with smoothness regularisation. 3rd place HMM: Student-t (nu=1) GR emission, smooth transitions, forward-backward posterior mean. |
| Decoding | Least-squares solution (L2 -> mean-like); 3rd place: forward-backward posterior mean (RMSE-consistent). |
| Metric consistent decode | Yes for the least-squares RGT (mean under Gaussian residuals) and for the 3rd place posterior mean. Pairwise DTW itself is mode-seeking (single best path); the global L2 averaging partly corrects that. |
| Evidence scope | Multiple wells jointly (whole basin for ChronoLog; target + siblings + typewell for 3rd place). |
| Prefix conditioning | Classical: none. 3rd place: known prefix used as self-reference GR template and as clamped HMM states (last 128 rows prepended, initial rate from last 256 rows). |
| Propose vs select | Single consistent solution. As used in ROGII, sibling-reference trackers are separate candidate trajectories fed to a SoftMax gate, i.e. usable as ensemble members or as extra channels. |
| Physical constraints | Monotone RGT in each vertical well (no layer crossing), loop closure across the well graph, smooth shifts; 3rd place: rate caps (/rate/ <= 0.25, integrated correction <= 10 ft), smooth dip state. |

### Data

| Field | Value |
|---|---|
| Synthetic to real | Not applicable for the classical methods (real logs, no learning). For Erudition, sibling-reference channels exist only for real wells; synthetic wells would need synthetic siblings (the master-typewell simulator can generate several laterals through one master system) or the channel set to a neutral value, with the risk that the network learns to ignore it (85% synthetic). |

### Results

| Field | Value |
|---|---|
| Comparability | 3rd place numbers are pooled RMSE in ft on the ROGII private LB (directly comparable in units, not in wells or baseline strength; their base was an HMM ~5.8, weaker than our 4.84 ft single model on our holdout). Classical RGT papers are not quantitatively comparable. |
| Validation protocol | 3rd place reports private-LB deltas across solution branches (not identical pipelines) and uses fold-specific reference pools to avoid leakage; GroupKFold by well. Sibling features are leakage-prone: a holdout well must only see siblings from the training fold, and siblings must never include itself. |
| Inference budget | Pairwise DTW over a few siblings plus a small least-squares solve: well under a second per well on CPU. Building binned sibling reference curves: milliseconds. Fits Kaggle and rig budgets. |

### Ambiguity handling

| Field | Value |
|---|---|
| Repeated layers | This is the main attraction. Band confusion arises because the typewell GR has self-similar cycles; a sibling lateral that drilled the same interval provides GR(TVT) with the same tool response and local thickness, so its signature at the candidate bands can differ where the typewell's does not (thickness changes, local facies). Loop consistency further rejects a band choice that disagrees with multiple neighbours. 3rd place keeps reference families as separate hidden states rather than averaging, preserving multimodality until evidence resolves it. |
| Drift | Partly. Siblings constrain local structure (dip) near the target only if they are laterally close; the XY-surface route (1st place, our failed geo prior) attacked drift directly but failed at the leaderboard, probably due to label inconsistency between interpreters. Distance-weighted siblings give an implicit local dip prior without imposing an explicit surface. |
| Uncertainty and calibration | Classical least squares gives no calibrated uncertainty (residual misfit per tie can flag unreliable wells). Incremental/stochastic correlation literature (RING) samples multiple consistent correlations for uncertainty. 3rd place uses PF multimodality and reference-family posteriors as gate features, no calibration analysis. |

### Relevance to erudition

| Field | Value |
|---|---|
| Where to use | (1) New cost-image channels built from sibling laterals: for each target, collect training-fold laterals whose typewell belongs to the same master system (seq_NN_master_typewell.py already finds the 54 systems), bin their GR by labelled TVT into a sibling reference curve sib_gr(TVT) on the 400-level grid (optionally inverse-distance weighted by XY), and add 'sib_gr' and '/gr - sib_gr/' (and a sibling-count/coverage mask) as channels alongside tw_gr and gr_abs_diff. (2) Consistency-based reranking of humps: for multimodal columns, score each significant hump by GR match against several independent references (typewell, siblings, self-prefix) instead of the XY-surface position the failed geo prior used. (3) A cheap 3rd-place-style sibling-reference HMM/DP as a diverse ensemble or gate member. |
| Error mode targeted | Band confusion (~51%: wrong_mode 24%, blend 15%, multimodal_miss 12%) primarily; drift (~25%) secondarily via local structural agreement. |
| Gap vs our implementation | Erudition uses exactly one reference curve per well: the provided typewell (channels tw_gr, gr_abs_diff, tw_tvt_rel). The only multi-well information tried is the XY geo prior (geo_tvt_diff and related channels, commented out in unet_static_channels; posterior_geo_pick.py hump picking), which models neighbours as a structural surface S = TVT + Z + C -- the one form of neighbour use that failed on the Kaggle LB for 1st place and failed in our own experiment (geo-nearest hump = truth only 52.4% vs dominant hump 76.2%, chance 46.2%). What we do not do: (a) use neighbour/sibling wells' GR as additional reference signals (grep for 'sibling' in the solution tree finds nothing); (b) use the target's own known prefix GR vs its labelled TVT as a self-reference channel (3rd place found it useful because GR calibration is shared between prefix and suffix); (c) enforce any cross-well consistency of band choice. The master-typewell consolidation already exists but is used only for synthetic generation, not for building reference channels. |
| Proposed experiment | E0 (diagnostic, CPU, no retraining, uses existing dump results/baseline_check_posterior_all): for each holdout well, build sib_gr(TVT) from training-fold siblings in the same master system (fold-safe; exclude holdout wells), and a self-prefix reference from the well's own known section. On multimodal columns, pick the significant hump whose GR match (mean /gr - ref/ over the column's 32 ft window) is best under sib_gr, and under typewell for comparison; report truth-hump hit rate vs the dominant hump's 76.2% and the geo prior's 52.4%, plus a split-half-tuned soft reweight p'(x) ~ p(x) * exp(-match(x)/T) like posterior_geo_pick.py's geo_reweight. Also report sibling coverage (fraction of holdout wells and TVT levels with >= 1 sibling). Stop if the hit rate is not clearly above 76.2% on covered columns. E1 (if E0 positive): in experiments/bilzard/seq_NN_dataset.py compute channels sib_gr, sib_gr_abs_diff, sib_count (and self_prefix_gr, self_prefix_abs_diff); for synthetic wells, generate 1-3 synthetic siblings from the same master series via the existing Simulator so the channel is populated in training; register channels in seq_NN_cfg.py (16 -> 19-21 channels). 60-epoch ConvNeXt vs control 5.47 ft, 2 seeds; success > 0.39 ft; then 150 epochs and FastViT KD. E2 (optional): a ChronoLog-style least-squares consistency step -- pairwise DTW between the target's predicted GR(TVT) profile and each sibling's, solve for a per-well TVT shift minimising all ties, apply as a closed-form offset correction to the expectation decode; test on the dump only. |
| Implementation effort | Low for E0 (~150 lines numpy on existing dumps plus master-system grouping). Medium for E1 (channel building in dataset, fold-safe sibling pools, synthetic siblings). Medium-high for a full RGT least-squares solver (E2). |
| Gpu cost | E0 and E2: CPU only. E1: same as a baseline 60/150-epoch run on RTX 5060 8 GB (+3-5 input channels, negligible compute). |
| Distillation compatibility | Compatible: new inputs are available to the FastViT student; E2 is a post-hoc correction on any model's output. |
| Error correlation with ours | Channels inside the same net: correlated errors, gains via new evidence. A sibling-reference HMM/DP tracker (3rd place style) would have substantially different errors from ConvNeXt/FastViT (different evidence and inductive bias), making it a good gate/blend member -- though our learned per-point gates have failed before, so a simple well-level blend is safer. |
| Notes | Strongest evidence is in-competition: 3rd place called reference-GR design 'the most important HMM decision' and the source of their largest private gains (sibling-lateral GR, fine TVT bins, self-prefix reference, keeping reference families separate). 2nd place's rejection of explicit formation modelling and 1st place's XY LB degradation agree with our failed geo prior: neighbours help as GR references, not as structural surfaces. Overlaps A03 (DTW), A15 (structural coordinate u = TVT + Z), A18 (TVT + Z - ANCC identity), B07 (seismic RGT). Deep-learning multi-well correlators (DHGC CNN+DP 2026, KANFormer, ASDNet) remain vertical-well and single-pair; none implement loop consistency end-to-end. |

### Other info

- **Id**: A12
- **Uncertain fields (omitted above)**:
  - architecture
  - authors_org
  - code_availability
  - dataset
  - metric_and_performance
  - risk

---

## A13 Mahto 2026 "Learning-Driven Autonomy Pipeline for Horizontal Wellbore Geology Prediction" (Research Square)

### Basics

| Field | Value |
|---|---|
| Name | Learning-Driven Autonomy Pipeline for Horizontal Wellbore Geology Prediction |
| Authors org | Lakshman Mahto (sole author), Quantitative Economics and Data Science, Birla Institute of Technology Mesra, Ranchi, India |
| Year | 2026 (manuscript dated May 14, 2026; posted July 20, 2026) |
| Venue link | Research Square preprint (status 'In Review', not peer reviewed), DOI 10.21203/rs.3.rs-9902401/v1, https://www.researchsquare.com/article/rs-9902401/v1 (PDF: https://www.researchsquare.com/article/rs-9902401/v1.pdf). CITATION VERIFIED via Crossref (https://api.crossref.org/works/10.21203/rs.3.rs-9902401/v1); ordinary web search does not index it, which is why earlier searches failed. |
| Code availability | No code released. Four supplementary CSVs on Research Square (diagnosticmetrics.csv, uncertaintycalibrationbins.csv, rollingperformance.csv, diagnosticpredictions.csv). The data-availability statement is boilerplate about LaTeX sources, with no code or data link. License CC BY 4.0. |
| Evidence level | Non-peer-reviewed preprint; single-well experiment; full text read (22-page PDF). Results are invalidated by label leakage that we reproduced locally (see notes). |
| Extreme summary | A single-well geosteering 'autonomy' pipeline (ridge regression backbone, tree baselines, typewell GR-score, RFF kernel mean embeddings, residual sparse-attention CVAE, uncertainty-weighted fusion, steering rules) on ROGII training well 000d7d20. It reports ridge at 0.59 ft RMSE, but ridge gets that score only because it takes the train-only formation-surface columns (ANCC etc.) as inputs, and TVT is an exact linear function of Z - ANCC. |

### Problem formulation

| Field | Value |
|---|---|
| Task definition | Predict interpreted TVT for every MD station after the prediction start (first missing TVT_input; row 1,442, MD 12,909 ft) of one horizontal well. It also outputs a TVT 90% interval, a GR-score lithologic confidence, the predicted typewell layer, a favourable-layer probability, a facies-state label and a steering action (continue / up / down / hold). |
| Inputs | Geometry: MD, X, Y, Z, horizontal distance, dZ, inclination/azimuth/curvature proxies. Log: GR plus rolling mean/std, gradient, second gradient and a missing mask. Stratigraphic: 'interpreted formation surfaces' S_i and Z - S_i. These are the ROGII train-only columns ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA, which the test files do not contain. Also the typewell (TVT, GR, Geology). No neighbour or sibling wells; one well only. |
| Target parameterisation | TVT directly for ridge/trees; the CVAE predicts the ridge residual r = TVT - TVT_ridge and GR over a 6-step horizon. No explicit dip state and no TVT + Z identity stated. The ridge implicitly learns TVT = c - (Z - ANCC) through the leaked surface features. |
| Output representation | Point estimate (fused TVT) with a Gaussian-style 90% interval (+-1.645 sigma_fused). The CVAE can sample multiple futures but only 6 steps ahead. No per-column depth distribution and no K hypotheses. |

### Method

| Field | Value |
|---|---|
| Architecture | (1) Ridge regression on the engineered features, trained on the pre-start interval of the same well. (2) Random forest, extra trees and gradient boosting baselines. (3) Typewell GR interpolated at predicted TVT gives GR-score = exp(-(dGR/30)^2). (4) Per-modality MLP encoders, Random Fourier Features and window mean-pooling (RFF-KME, W_raw = 6) give fused tokens over L_seq = 8 windows. (5) A GRU-attention CVAE, then a causal block-sparse-attention CVAE, forecasting the ridge residual plus GR for H = 6 steps. (6) Fusion by inverse-variance or 1/RMSE^2 weights, with sigma inflated by model spread, GR std and (1 - GR-score). (7) A rule-based favourability and steering layer. |
| Core mechanism | Residual learning around a strong deterministic backbone: ridge carries the trend and the deep model corrects the residual. The backbone is strong only because of label leakage through the formation-surface columns. The GR-score is a pointwise typewell-consistency check used as a confidence signal, not as evidence for picking position. |
| Training objective | Ridge L2 least squares. CVAE ELBO: lambda_r //r - r_hat//^2 + lambda_GR //GR - GR_hat//^2 + beta KL(q(z/y,c) // p(z/c)). Fusion and uncertainty inflation are heuristic, not trained. |
| Decoding | Fused prediction = weighted mean of component TVTs (precision or 1/RMSE^2 weights). The CVAE output is added to ridge. Layer comes from a typewell interval lookup and the steering action from thresholds. |
| Metric consistent decode | Mean-type fusion, so nominally RMSE-consistent; it is not mode-seeking. Irrelevant in practice because the backbone leaks the label. |
| Evidence scope | Single well: the model is trained on that well's own heel and evaluated on its own lateral, with local rolling windows (6 x 8 samples). No multiwell training. |
| Prefix conditioning | The model is fit only on the known prefix (rows 0-1,441) of the same well, i.e. per-well fitting. In principle this is the most extreme form of prefix conditioning, but here it only calibrates the leaked Z - surface relation. |
| Propose vs select | Single output from a fixed-weight fusion of components. Not a candidate generator in any useful sense for us. |
| Physical constraints | None imposed explicitly. The leaked surface features encode the TVT + Z - ANCC identity by construction. |

### Data

| Field | Value |
|---|---|
| Synthetic to real | No synthetic data. No normalisation between lateral and typewell beyond a fixed s_GR = 30 API in the GR-score. No alignment stage. |

### Results

| Field | Value |
|---|---|
| Metric and performance | Table 1 (forward interval, one well): ridge RMSE 0.5876 ft, MAE 0.5129, bias -0.4628, R2 0.9867. 'gr_match' has identical numbers (a likely copy or pass-through bug). Uncertainty-fused 0.6389. Extra trees 5.39, random forest 7.99, gradient boosting 19.55 ft. Our local check on the same well: a linear fit on Z - ANCC trained on the prefix gives 0.007 ft; all six surfaces + Z give 0.042 ft; geometry only (MD, X, Y, Z) gives 58.9 ft; last-value carry-forward gives 7.45 ft. |
| Comparability | Not comparable. One well versus our 155; it uses train-only interpreted surfaces that do not exist at test time (a label proxy); there is no cross-well generalisation. An inference-legal version would be the geometry-only or last-value numbers (58.9 / 7.45 ft on this well), far worse than our 4.84 ft pooled. |
| Validation protocol | Single temporal split within one well (prefix train, lateral test). No well-level CV, no leaderboard. Section 7 recommends well-ID splits and train-only normalisation for future multiwell work, but none was done. |
| Inference budget | Trivial (ridge plus a small CVAE on one well). Not reported. |

### Ambiguity handling

| Field | Value |
|---|---|
| Repeated layers | Not handled. The GR-score is pointwise and unimodal, and it would score a wrong repeated band as high confidence. The author lists windowed or DTW-based lithologic confidence as future work. |
| Drift | Not handled explicitly. Within one well the leaked surfaces remove drift entirely; with legal inputs, a ridge on geometry extrapolates linearly and diverges (58.9 ft on this well in our check). |
| Uncertainty and calibration | Heuristic sigma_fused = sqrt(sigma_resid^2 + spread^2) * (1 + lambda_noise*GR_std/30 + lambda_GR*(1 - GR-score)), with a Gaussian 90% interval. There is a reliability-style plot (Fig. 5: predicted sigma bins vs observed /error/ and empirical 90% coverage), but numbers are only in the supplementary CSV. No temperature/isotonic calibration and no ECE. It explicitly notes that fusion underestimates uncertainty when components share a bias. |

### Relevance to erudition

| Field | Value |
|---|---|
| Where to use | Only as reporting and presentation inspiration (the lit-review's intended use): rolling-RMSE-along-MD plots, error by typewell layer, calibration-by-sigma-bin plots, and GR-score versus uncertainty. Also as a cautionary citation showing that the ROGII train files contain label-proxy surface columns, and why any paper using them must be read critically. Not a model component. |
| Error mode targeted | None of ours (band confusion, drift, broad hump). Relevant only to measurement/protocol, as an example of leakage. |
| Proposed experiment | No model experiment is recommended. Two cheap hygiene/reporting tasks. (1) Leakage audit: grep the data loaders (kaggle_1st_place/solution/experiments/*/seq_NN_*.py, kaggle2ndplace/src/*, rogii_champion/src/*) for ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA and assert that none reaches a holdout-time input tensor. Add a unit test that zeroes those columns for holdout wells and checks the predictions are bit-identical. (2) A reliability figure for the paper: from seq_NN_posterior_dump.py --save-all outputs, bin rows by posterior std (deciles) and plot the RMS error per bin against the mean std, plus empirical coverage of the posterior 90% interval, for the ConvNeXt teacher and the FastViT student. Report ECE-style gaps. This directly quantifies the calibration lever. Control: none needed, since it is analysis only. |
| Implementation effort | Low: the audit is a grep plus a single test; the reliability plot is about 50 lines over existing posterior dumps. |
| Gpu cost | None (CPU analysis over existing posterior dumps). |
| Distillation compatibility | Not applicable (no model to distil). The reliability analysis applies to both teacher and student posteriors. |
| Error correlation with ours | Not applicable. An inference-legal version (geometry ridge) would be far too inaccurate (tens of ft) to earn blend weight. |
| Risk | Citing it as evidence that 'TVT is mostly geometry-controlled' would be wrong and could embarrass our write-up. The conclusion is an artefact of feeding interpreted formation surfaces (a direct function of the label) to the model. It is a single well, has no multiwell validation, and 'gr_match' has metrics identical to ridge (suspect code). No code is available to check further. |

### Other info

- **Uncertain fields (omitted above)**:
  - dataset
  - gap_vs_our_implementation
  - notes

---

## A14 ROGII 3rd place -- five candidate trajectories (HMM, PF, 2 NNs, Adaptive 1D SDF) + sequential SoftMax gate

### Basics

| Field | Value |
|---|---|
| Name | ROGII Wellbore Geology Prediction 3rd Place Solution: five candidate TVT trajectories (3-family HMM, particle filter, Last-PS NN, Delta NN, Adaptive 1D SDF) fused by a sequential SoftMax gate |
| Authors org | Takoi (takoihiraokazu) and tereka, Kaggle team |
| Year | 2026 (writeup posted Aug 6, 2026) |
| Venue link | Kaggle competition writeup: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/3rd-place-solution ; local copy C:/Users/Aadinarayan R/Desktop/CSE/projects/Erudition/3rd.txt |
| Evidence level | Competition writeup with CV, public LB and private LB for every component (self-reported but backed by Kaggle's hidden-test scoring); inference code public. |
| Extreme summary | Generate physically plausible TVT candidates (HMM, PF), add NNs and a correlation model that make different errors, and let a small BiLSTM gate output per-row convex (SoftMax) weights over the five candidates, trained on MSE plus best-candidate CE. Fold-safe CV 5.29 ft, but private 5.84 ft (public 6.04). The largest private gains came from reference-GR design (sibling laterals, self-prefix) and prefix-based rate initialisation, not from the gate. |

### Problem formulation

| Field | Value |
|---|---|
| Task definition | Predict TVT for every hidden row after the prediction start (PS) of each horizontal well from the trajectory, lateral GR, the typewell (TVT, GR, Geology) and the known heel TVT prefix; row-level RMSE. |
| Inputs | MD, X, Y, Z, lateral GR (plus missing flag), known TVT_input prefix, typewell GR/TVT. Also sibling-lateral GR from training wells sharing the same typewell (aggregated into 0.25 ft or 0.0625 ft TVT bins), a self-reference (TVT_input, GR) template from the target well's own prefix, and spatial-distance-weighted siblings in one branch. |
| Target parameterisation | Physics models: hidden state = TVT position (structural coordinate TVT + Z), formation rate d(TVT+Z)/dMD, GR bias, reference family. Last-PS NN: offset TVT - last_known_tvt. Delta NN: row-to-row delta TVT_t - TVT_(t-1), integrated from PS. SDF: a softmax over 256 typewell TVT candidates, expectation output. Gate: convex weights over the five candidate values. |
| Output representation | Five point-estimate trajectories (the PF also exports quantile, spread, skew and IQR features, and the Last-PS NN a Gaussian sigma), then a per-row softmax weight vector over K = 5 candidates, then a point estimate. |

### Method

| Field | Value |
|---|---|
| Architecture | (1) 3-family HMM mixture: Exp417 base (weight 0.5; two typewell/sibling reference mixtures 0.2/0.8, self-prefix reference, bias state, exact forward-backward), Local-DTW HMM (0.2), fine-bin 0.0625 ft sibling HMM (0.3). Student-t (df=1) GR emission. The last 128 prefix rows are prepended with clamped TVT; initial rate comes from the last 256 prefix rows; output projected with a rate cap of 0.25 and at most 10 ft integrated correction. (2) PF: 500 particles x 240 seeds x 3 seed bases, rate momentum 0.9995, hard/soft observation families (f015), reference flag per particle, genealogy smoothing, likelihood-weighted seed averaging; candidate = 0.6 f015 + 0.4 soft. (3) Last-PS NN: 49 features into 96 channels; 16 dilated-TCN residual blocks (kernel 7, dilation 1..128 twice); MaxViT1D attention after blocks 4/8/12/16; downsampled typewell cross-attention after block 8 (residual encoder + gated BiLSTM); gated LSTM head; Gaussian mean/variance head. (4) Delta NN: same family, 31 difference features, dropout 0.5, manifold mixup. (5) Adaptive 1D SDF: TCN encoders for lateral and typewell, 128-d cosine correlation volume over 256 typewell TVT candidates fused with trajectory/rate/graph-edge cost, prefix-estimated GR bias/scale, softmax T = 1.75, expected TVT. (6) Gate: BiLSTM (hidden 32, 1 layer) or CNN (hidden 32, k = 7, 2 residual blocks) producing per-row logits over the 5 candidates. (7) 5 GroupKFold folds x 5 split patterns = 25 checkpoints per family; final = BiLSTM gate only (CNN weight 0). |
| Core mechanism | Propose-then-select under a convex constraint. Diverse candidates with different error structures (smooth global HMM path, multimodal PF, anchor-offset NN, local-slope delta NN, direct correlation SDF). The gate only reweights them per row using context: candidate disagreement, distance from PS, GR-missing flags and the NN's predictive sigma. The convex hull prevents extreme extrapolation with only 773 wells, at the cost of never correcting drift shared by all candidates. The deeper mechanism behind private generalisation was better reference GR and prefix use in the physics models. |
| Training objective | HMM/PF: none (probabilistic inference). Last-PS NN: Gaussian NLL with sigma^2 = softplus + 0.001; RMSE to GNLL improved 5.4176 to 5.3645 on the 3-split ensemble. Delta NN: delta RMSE (a GNLL pilot was worse). SDF: candidate CE + 0.25 SmoothL1(TVT/10) + 0.10 delta + 0.05 Rank-N-Contrast. Gate: MSE(y, sum w_k c_k) + CE(w, argmin_k /c_k - y/), 8 epochs. |
| Decoding | Per-row convex combination sum_k softmax(logits)_k * candidate_k, averaged over 25 gate checkpoints. The HMM uses the forward-backward posterior mean; the PF uses the likelihood-weighted mean of smoothed paths; the SDF uses the softmax expectation; the NN uses the Gaussian mean. |
| Metric consistent decode | Mostly RMSE-consistent. The MSE term and the convex mixing hedge between candidates, and the candidates are mean decodes. The CE-to-best-candidate term pushes toward hard selection (mode-seeking), which raised our pooled RMSE when we tried it (see gap). |
| Evidence scope | Whole lateral (bidirectional HMM/BiLSTM), against the typewell plus sibling laterals from other wells sharing the typewell, plus the target well's own prefix as a self-reference. |
| Prefix conditioning | Strong: the last 128 prefix rows are prepended inside the HMM with clamped TVT states; initial formation rate uses a robust estimate over the last 256 prefix rows (30 to 256 rows gave -0.042 private); a self-prefix GR template is mixed into the reference; the NN anchors on last_known_tvt; the SDF estimates GR bias/scale from the prefix. |
| Propose vs select | Explicit propose (5 candidates) plus select (sequential SoftMax gate). An unconstrained BiLSTM refiner was worse (5.4023 vs 5.2797 CV, lost in 5/5 folds). |
| Physical constraints | TVT + Z structural coordinate with a slowly varying rate state (momentum), rate capped at 0.25 and integrated correction capped at 10 ft, slowly drifting GR bias, heavy-tailed emission, convex-hull constraint on the final output. |

### Data

| Field | Value |
|---|---|
| Dataset | ROGII Kaggle: 773 training wells, 3,783,989 rows, with hidden test wells; 5-fold GroupKFold by well gives 155/155/155/154/154 validation wells (the same fold size as our 155-well holdout). |
| Synthetic to real | No synthetic wells. Lateral-vs-typewell mismatch is handled by the GR bias state, sibling-lateral references (same tool and resolution as the lateral), self-prefix references and prefix-estimated GR bias/scale in the SDF. |

### Results

| Field | Value |
|---|---|
| Metric and performance | RMSE ft (CV / public / private): 3-family HMM 5.970 / 6.207 / 6.229; PF 6.583 / 7.011 / 6.725; Last-PS NN 5.472 / 6.074 / 5.903; Delta NN 5.770 / 6.093 / 6.102; Adaptive 1D SDF 9.542 / 8.502 / 8.837; CNN gate 5.426 / 6.027 / 5.814; BiLSTM gate 5.333 / 6.043 / 5.836; final 5-split ensemble 5.288 / 6.043 / 5.836. The gate's gain over the best single candidate (Last-PS NN) is -0.14 CV but only -0.07 private. Private-LB interventions: spatial-distance-weighted sibling HMM 5.817 to 5.691 (-0.126); rational distance weighting -0.097; 30 to 256-row rate window -0.042; 3 to 5 split patterns 5.903 to 5.836 (-0.067); Rao-Blackwellized PF bias -0.003; fold-specific HMM reference pools -0.008. |
| Validation protocol | Two-stage fold-safe OOF: base candidates for gate training come only from fold models that did not see the well; base models and gates share fold assignments; fold-local normalisation; 5 splits x 5 folds; no public-LB selection. The CV 5.29 vs private 5.84 gap (+0.55) is NOT mainly gate selection bias. Training-free components show similar gaps (HMM +0.26, PF +0.14), and the NN shows +0.43, so most of the gap is a test-set distribution shift. The gate itself realised about half of its CV gain on private (-0.07 of -0.14). |

### Ambiguity handling

| Field | Value |
|---|---|
| Repeated layers | Handled indirectly by diversity: the PF keeps multimodal hypotheses (quantile/spread/skew features), the HMM keeps reference families as separate hidden states, and the gate can shift weight toward a different candidate where candidates disagree. Band confusion is only fixed if some candidate is on the right band, because a convex mix of two wrong-band candidates is still wrong or blended. The largest lever was reference quality (sibling laterals), which sharpens the likelihood and so reduces band ambiguity at the source. |
| Drift | Explicit: rate state with momentum, a robust 256-row prefix rate estimate, prepended prefix with clamped TVT, and rate/total-correction caps. The Delta NN models local slope. They state the gate cannot fix drift shared by all candidates (an OOF oracle confirmed this ceiling) and name drift as the dominant private-test error for long hidden sections. |
| Uncertainty and calibration | Gaussian predictive sigma from the Last-PS NN (GNLL) and PF spread/IQR are used as gate features, not as outputs. No reliability/ECE analysis. GNLL training improved the mean itself (acting as a learned loss attenuation). |

### Relevance to erudition

| Field | Value |
|---|---|
| Where to use | (1) As a design template for a K-candidate gate whose candidates are hypotheses extracted from our own posterior (per-mode means of the ConvNeXt/FastViT per-column softmax), plus the expectation decode and physics candidates. (2) The physics-side levers (sibling references, self-prefix reference, 256-row prefix rate estimate, prefix prepended in the HMM) as new input channels or reference curves for our cost image. (3) The 5-split x 5-fold protocol as a lesson on selection: evaluate any learned gate with nested cross-fitting. |
| Error mode targeted | Band confusion (~51%) via diverse candidates and sibling references; drift (~25%) via the prefix rate state; the gate itself mainly targets blend/wrong-mode choice. |
| Implementation effort | E1 medium (feature builder plus synthetic-well sibling proxy); E2 low-medium (mode extraction and adapting the existing gate); E3 low-medium (code exists, needs tuning and a full holdout run). |
| Gpu cost | E1: one 150-epoch FastViT run on the RTX 5060 8 GB (same as the baseline; 2-4 extra input channels add negligible memory), plus about the same again if the ConvNeXt teacher is retrained for KD. E2 and E3: CPU or a few GPU-minutes. |
| Distillation compatibility | E1: yes; sibling channels are inputs consumed by teacher and student, and KD on posteriors is unchanged (the teacher should also see the channels). E2: post-hoc on the student's posterior, so fully compatible. A multi-model candidate gate like theirs is not distillable into a single FastViT, but it can sit after it. |
| Error correlation with ours | Their candidates are highly diverse (HMM/PF are training-free physics; our PF residual correlation with the ConvNeXt is 0.20-0.33). Modes of our own posterior are, by construction, correlated with our expectation decode. Sibling-reference evidence is new information, which is the property the context says is needed. |
| Risk | Our own two-candidate gate already lost to a single global weight (5.02-5.18 vs 4.87-4.91 ft), because about 124 fitting wells are too few for a per-row chooser and the gate is confidently wrong on catastrophic wells. Their gate had 773 OOF wells and still only realised -0.07 private versus -0.14 CV. Since our expectation decode is Bayes-optimal for a calibrated posterior, reweighting modes without new evidence cannot beat it. The 3rd place's final 5.84 private is worse than the 1st place approach we reproduce, so copying the whole stack is not a path to lower RMSE; the transferable parts are the reference-GR and prefix levers. Sibling references also carry a leakage risk if the query well or holdout wells enter the sibling bank (anchor_sibling.py guards the query well; the holdout wells must also be excluded). |
| Notes | Warning about selection: the CV 5.29 to private 5.84 gap is common to all components (training-free HMM +0.26, PF +0.14), so it reflects a harder test set rather than gate overfitting, but the gate's marginal gain halved out of sample. Their own takeaway (paraphrased): prioritise the right reference GR, stable initial-rate estimation, diverse HMM error modes and well-level CV diversity before adding model complexity. Increasing the number of CV split patterns was itself a private-LB gain (-0.067). Related items: A05 (PF), 1st place writeup (1st.txt) for PF-heatmap channels, B06 (differentiable PF). The repo already contains the 3rd-place inference notebook (rogii_exp417_gnll_five_groupkfold_submit.py), which can be mined for exact HMM/PF hyperparameters. |

### Other info

- **Uncertain fields (omitted above)**:
  - code_availability
  - comparability
  - gap_vs_our_implementation
  - inference_budget
  - proposed_experiment

---

## A15 ROGII silver structural-coordinate tracker (EdidiongA; u = TVT + Z, explicit dip state, banded trellis, windowed GR correlation)

### Basics

| Field | Value |
|---|---|
| Name | ROGII silver-medal structural-coordinate tracker (EdidiongA / Edidiong-Abasi Anwanane): u = TVT + Z, second-order (u, dip) banded trellis, windowed GR emissions against a merged typewell/pseudo-typewell reference, inverse-variance branch fusion with a spatial field, deployment-side dose calibration |
| Authors org | Edidiong-Abasi Anwanane (Kaggle / GitHub: EdidiongA), solo |
| Year | 2026 |
| Venue link | Kaggle ROGII - Wellbore Geology Prediction (241/6,125, silver, top 4%); GitHub https://github.com/EdidiongA/wellbore-tvt-prediction ; preprint 'Blind-Zone TVT Prediction in Horizontal Wells: An Information-Ceiling Study with Deployment-Side Dose Calibration' (PDF in repo papers/rogii_preprint_Anwanane.pdf, 29 pages; arXiv link announced as forthcoming); author site https://edidionga.github.io/ |
| Code availability | Public, MIT license: tracker notebooks (final base model/rogii_tvt_tracker_v22.ipynb), neural line (1D U-Net, BiGRU with 5-fold OOF), postprocessing v25-v58 one mechanism per file, diagnostics L1-L8, full campaign ledger docs/campaign_log.md. Written for the Kaggle CPU offline environment; competition data not redistributed. |
| Evidence level | Competition writeup plus public code plus a self-published (not peer-reviewed) preprint with a full submission ledger (public and some private scores per version) and leave-one-well-out measurements. |
| Extreme summary | Reparameterise the target as the smooth structural coordinate u = TVT + Z and track (u, dip) with a second-order HMM whose prior makes constant dip free and dip changes expensive. Emissions are clipped /GR - ref(u - Z)/ against a typewell blended with the well's own known-zone GR. CPU only, about 4 s per well, private 8.354 (base 8.416). Much weaker than the U-Net line, but built from very different assumptions. |

### Problem formulation

| Field | Value |
|---|---|
| Task definition | Predict TVT at every blind-zone station (median ~4,800 stations, ~74% of the lateral) of a horizontal well from the known-zone TVT prefix, trajectory (X, Y, Z, MD), horizontal GR, and one vertical typewell (GR vs TVT). The method tracks the smooth structural curve u(MD) and recovers TVT = u - Z. |
| Inputs | Horizontal GR; trajectory Z (and X, Y for the spatial field); known-zone TVT_input (prefix labels); typewell GR vs TVT, affine-mapped to horizontal-tool GR units by a trimmed least-squares fit on the known zone; pseudo-typewell = known-zone (TVT, GR) pairs binned over TVT; for the spatial branch, the known-zone u-curves of training/neighbour wells (IDW local gradient fits over X, Y). No offset logs. |
| Target parameterisation | Structural coordinate u = TVT + Z (stratigraphic position = geology minus wellbore). Paper evidence: replacing true u by its 2,000 ft moving average costs only ~3 ft; an oracle knowing u smoothed at 300 ft scores 0.86 ft; high-passed TVT and Z correlate at -0.74 per well. Explicit dip state (grid steps per 30 ft block, within +-14 steps). Their neural line found the increment u - u_lastknown (drift relative to the anchor) much better than absolute datum-centred u (prototype 22.6 -> 12.65 ft). |
| Output representation | Discrete posterior over (u, dip) per 30 ft block. Two decodes, Viterbi MAP path and forward-backward posterior mean, ensembled at two positional-prior strengths. Five semi-independent branch paths are fused per station into one point estimate, weighted by the inverse variance of their disagreement. |

### Method

| Field | Value |
|---|---|
| Architecture | (1) Merged reference: the typewell is mapped to tool units and blended with the pseudo-typewell using weight w = n/(n+3) (n = local smoothed known-zone sample count), then EM re-estimates the reference along the pass-1 path. (2) Second-order HMM over 30 ft MD blocks, state (u on a 0.5 ft grid inside a per-well band, dip), with banded transitions vectorised through precomputed gather maps. Coarse-to-fine: a 1 ft scout pass locates the drift tube, the full ensemble runs at 0.5 ft within tube +-150 ft, the final grid is 0.25 ft, and the full band is redone when the path grazes the band edge. (3) Branches: drift-cancelling (windowed mean cancellation between observed GR and reference, targets slow gain drift; weight 0.40), typewell-only (0.28 / 0.15, a hedge against pseudo-reference failure), their combination, the pre-branch ensemble, and a GR-independent spatial structural field (IDW gradient fits of neighbour known-zone u-curves). (4) Inverse-variance fusion per station; the field can take weight up to 0.95 where the branches scatter. (5) Post-processing found by leaderboard dose-response: position-ramped toe damping of far-toe curvature toward the near-blind linear trend (S ~0.13-0.15), high-frequency shrinkage (h = 0.40, window 601), and a global signed cubic drift profile (toe -2.4 ft). (6) Guards fall back to constant continuation on malformed wells, low known-zone GR correlation, or non-finite output. |
| Core mechanism | The coordinate change turns a jagged 5,000-sample regression into estimating one smooth curve with few effective degrees of freedom, and TVT = u - Z restores every wiggle exactly. On top of that sits a second-order (dip-persistence) prior: steady dip costs nothing and dip changes cost lambda = 300. The author argues that first-order smoothness priors fail here because they penalise real steady drift, which is everywhere, in order to suppress noise. Other load-bearing ideas: using the well's own known-zone GR as a reference (largest classical gain, 11.63 -> 11.20), and using uncertain priors proportionally (inverse-variance pull) instead of snapping to them. The discrete field prior failed on hidden wells (10.200 vs 9.382) even though it won every local holdout. |
| Training objective | No gradient training. Hand-set costs: clipped absolute GR residual emission, dip-change penalty lambda = 300, a weak positional prior of 0.02 per ft of deviation from constant continuation, plus a zero-prior member weighted 40%. Branch weights and regularisation doses were tuned on a 'corrupted suite' of perturbed training wells and on leaderboard probes. The separate BiGRU line used masked L1 on blind stations only. |
| Decoding | Viterbi (MAP) and forward-backward (posterior mean) decodes averaged at two prior strengths, then inverse-variance fusion of branches, then damping, HF shrinkage and a drift profile. |
| Evidence scope | Whole lateral (sequence-level HMM over the full blind zone) plus typewell; the spatial branch adds multiple wells jointly through an IDW structural field. |
| Prefix conditioning | Heavy and closed-form. It uses the anchor at the known/blind boundary (audited at 0.01 ft median error), a per-well band centred on constant continuation, an affine typewell-to-tool GR calibration fitted on the known zone, the pseudo-typewell built from known-zone (TVT, GR) pairs, and the known-zone u-curves that feed the spatial field. Prefix self-calibration (hiding the tail of the known zone to choose configurations) was measured as invalid: the pseudo-blind zone is ~10x shorter and 2.7x easier, and selecting on it cost 0.249 ft. |
| Propose vs select | Produces one fused path per well. Internally it proposes (five branches, two decodes) and fuses them by disagreement-weighted averaging, not selection. Cheap enough (CPU) to act as one ensemble member or an input channel for us. The author's own routing test (tracker vs BiGRU) found a cross-fitted routing gain of exactly 0. |
| Physical constraints | u = TVT + Z identity (exact); locally constant dip (second-order persistence); banded u range; clipped emission for washouts; positional prior toward constant continuation; spatial structural field that assumes locally linear surfaces. |

### Data

| Field | Value |
|---|---|
| Synthetic to real | No synthetic data. Real-to-typewell GR normalisation by a trimmed least-squares affine fit on the known zone, blended with the pseudo-typewell (w = n/(n+3)), with EM refinement of the reference along the first-pass path. |

### Results

| Field | Value |
|---|---|
| Metric and performance | Leaderboard ladder: constant continuation 15.88 public; v2 bare (u, dip) tracker 13.196; + merged reference/EM/drift-cancel 11.630; typewell-only branch 11.197; proportional spatial field 9.624; inverse-variance fusion 8.926; sealed base v22 8.913 public / 8.416 private; final v55 8.720 public / 8.354 private (241/6,125). The tracker's honest leave-one-well-out OOF on all 773 wells is 5.507 ft, giving a 3.41 ft train-to-hidden gap. BiGRU line on identical inputs: 6.9-7.1 ft OOF (5 well-grouped folds). Tracker vs BiGRU per-well error correlation 0.171; the best fixed blend (0.10 toward the BiGRU) gains only +0.079 ft. Oracles: perfect routing 0.86 ft, perfect per-well slope removal 0.80 ft, perfect configuration selection 0.56 ft; none could be predicted at inference. |
| Validation protocol | Leave-one-well-out with the spatial field rebuilt excluding every point of the evaluated well (post-exclusion median nearest-neighbour distance 468 ft); a pad holdout (246 wells in 2,500 ft neighbourhoods); corrupted-well suites; a fitted local-to-leaderboard transfer law. Key warnings: local and hidden orderings inverted repeatedly (v25, best local at 5.459, gave the worst hidden score, 9.302); random holdouts flatter spatial mechanisms because pad-mates remain in the field; the final regularisation doses were fitted to the public leaderboard, which is not fold-safe. |
| Inference budget | About 4-6 s per well on one CPU core; the whole hidden set in under 40 min; no GPU, no training. Fits any rig or Kaggle budget easily. |

### Ambiguity handling

| Field | Value |
|---|---|
| Repeated layers | Not handled explicitly. The banded trellis limits hypotheses to a per-well band around constant continuation (a tube found by the scout pass). Clipped GR emissions and the pseudo-typewell (same tool, same well) sharpen matching. Forward-backward keeps several bands alive until the mean decode. The spatial field acts as a GR-independent tie-breaker. It is applied proportionally (inverse variance) instead of as a discrete snap, because the discrete version picked the wrong rung at high confidence on hidden wells. |
| Drift | The main target of the design. The explicit dip state with zero-cost persistence represents steady drift instead of penalising it; the drift-cancelling branch removes slow GR gain drift; a zero-prior member (40%) avoids pulling toward constant continuation. The author showed the residual error is slope-shaped (subtracting each well's fitted drift rate would gain 0.80 ft) but not predictable from any internal observable (OOF Spearman 0.095). A typewell-derived slope correction was the only CI-positive local effect (+0.082 ft) and lost 0.389 ft on the hidden set. The gains that did work were toe damping toward the near-blind linear trend and a global signed cubic drift ramp, both tuned on the leaderboard. |
| Uncertainty and calibration | Uses per-station branch disagreement (spread across five paths) as its uncertainty measure and weights by inverse variance. No reliability diagrams or ECE. Error grows along the lateral from ~3.6 ft near the anchor to ~33 ft at the toe. The author notes the field is sometimes wrong with high confidence on hidden wells. |

### Relevance to erudition

| Field | Value |
|---|---|
| Where to use | (a) As a structured decode over our per-column posteriors: replace the independent per-column expectation with a second-order (u, dip) forward-backward that uses the ConvNeXt/FastViT column softmax as its emission, and read out the posterior mean. (b) As a cheap CPU ensemble member or extra input channel: render the tracker path and its disagreement spread into the 16-channel cost image, as the 1st place did with PF features. (c) As design rules for the gate/blend: blend proportionally by inverse variance or disagreement, never snap to one choice, and build references from the well's own known zone. |
| Error mode targeted | Drift (~25%) through the explicit dip state and drift-cancel branch; secondarily band confusion (~51%) through whole-lateral evidence and a disagreement-weighted spatial pull. |
| Gap vs our implementation | What we already have: the 1st-place recipe calibrates typewell GR from the visible region by TVT binning (equivalent to their pseudo-typewell blend); the Z-shift augmentation keeps TVT + Z invariant, so the network implicitly learns the u identity; TVT is predicted relative to the anchor (their increment finding). What we lack: (1) Any explicit dip/slope state. Our per-column softmax + expectation decode treats columns independently. From the context description, our failed post-hoc forward-backward continuity decode used a first-order smoothness prior, and the preprint gives a concrete reason such priors fail (they tax steady drift). A second-order dip-persistence transition in u-space has not been tried and is the main untested variant. (2) Disagreement-based inverse-variance fusion. Our learned per-point ConvNeXt/AnchorCNN gates failed, while their unlearned weighting by live disagreement was their largest single hidden-set gain. (3) A GR-independent structural-field branch used proportionally. Our XY geo-prior hump picking failed as a discrete pick, which is exactly the failure they documented (discrete field prior 10.200 vs 9.382); the soft, confidence-weighted pull has not been tried. (4) Position-ramped toe damping and HF shrinkage are cheap post-hoc checks for the drift mode, but their doses were tuned on the leaderboard and local LOWO priced them as harmful, so gains may not show on our holdout. |
| Proposed experiment | E1 (main, CPU-only, ~1 day): add kaggle_1st_place/solution/posterior_dip_decode.py next to posterior_fb_decode.py. Load the dumped posteriors (seq_NN_posterior_dump.py --save-all, results/baseline_check_posterior_all). Per well, map each column's 400-level posterior into u = TVT + Z (shift by the column's Z relative to the anchor). Run forward-backward over state (u, dip), dip in {-K..K} grid steps per column, with u_{t+1} = u_t + dip_t, zero cost for persistence, and dip changes costing lambda (sweep 10, 30, 100, 300). Add a small jump probability for faults. Emission = log(posterior^(1/T)), T in {1, 1.5, 2}. Decode the posterior-mean u, then TVT = u - Z. Controls: the plain expectation decode on the same dumps (FastViT 4.84 / ConvNeXt), plus the old first-order FB decode. Pick lambda/T on training wells; report pooled RMSE on the 155 holdout wells with a paired per-well bootstrap against the +-0.39 ft noise floor, and split the change across wrong_mode / blend / drift buckets with posterior_analysis.py. E2: disagreement-weighted blend, w_i proportional to 1/variance over {ConvNeXt, FastViT, AnchorCNN} per column (no learning), against the fixed-weight pair (4.69). E3: soft XY field pull: add the 1st-place estimate dTVT = a*dX + b*dY - dZ as a Gaussian log-prior term inside E1's emission, with variance set from neighbourhood-quality stats (proportional, not a hump pick). E4 (optional): port the tracker notebook as an extra blend member and check per-well error correlation with ConvNeXt; expect a weight of 0.1 or less. |
| Implementation effort | Low-medium. E1/E2 are numpy on existing posterior dumps; a banded forward-backward with a few thousand states over 345 columns is cheap. Porting the full tracker with reference EM, branches and field is medium effort (MIT notebooks exist but are written for Kaggle). |
| Distillation compatibility | Compatible. The dip-aware decode runs after the network on its output posteriors, so it applies unchanged to the FastViT student. Dip-decoded posterior marginals could also serve as KD targets (untested). |
| Error correlation with ours | Tracker errors are likely close to independent of ours (in the author's study, tracker vs BiGRU per-well error correlation was 0.171). But the tracker is ~2 ft worse than U-Net solutions, and the author's best blend weight was only 0.10 (+0.079 ft), so as a raw blend member expect little. Its value lies in its structure: using its prior inside our decode, or its disagreement as a confidence signal. |
| Risk | (1) Our failed forward-backward continuity decode suggests the posterior already carries most of the sequential information the network can extract; a second-order prior may again be neutral. (2) Band confusion is the dominant error, and for pooled RMSE a mean decode improves only with more evidence or better calibration. A dip prior adds evidence only when the wrong band implies an implausible dip change. (3) The author's post-processing doses were fitted on the public leaderboard and ran opposite to local validation, so local gains from damping/drift ramps may be noise or misleading; the 3.4 ft LOWO-to-hidden gap shows local protocols can mislead. (4) Faults (jumps in TVT + Z, simulated in the 1st-place augmentation) break the dip-persistence prior, so the transition needs a small jump probability. (5) With an absolute score of ~8.35, the tracker as a standalone member is unlikely to help a 4.7-4.8 ft system. |
| Notes | The preprint's main value is methodological: the 'displacement law' (hidden-side optima of regularisation sit where local validation calls them harmful), the local/public/private inversions, and the rule that proportional mechanisms fail boundedly while discrete ones fail unboundedly (consistent with our failed XY hump picking and learned gates). The pseudo-typewell weight w = n/(n+3) and the clip on GR residuals are cheap details worth copying into our typewell-calibration channel. The author notes that public solutions converged on a robust low-order polynomial in U = T + Z. The related identity TVT + Z - ANCC = const (A18) could anchor u exactly. The preprint calls the metric MAE while other writeups report RMSE; treat its numbers with caution. |

### Other info

- **Id**: A15
- **Uncertain fields (omitted above)**:
  - comparability
  - dataset
  - gpu_cost
  - metric_consistent_decode

---

## A16 ROGII 14th place hierarchical U-Net stack with seed averaging (keithtyser)

### Basics

| Field | Value |
|---|---|
| Name | ROGII 14th place: hierarchical 1-D U-Net stack with seed averaging (keithtyser / Keith Tyser) |
| Authors org | Keith Tyser (Kaggle ktyser / keithtyser), solo; the author says coding agents wrote the code, launched and watched the runs, and drafted the writeup |
| Year | 2026 |
| Venue link | Kaggle writeup https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/14th-place-a-hierarchical-u-net-stack-with-seed-a ; training code https://github.com/keithtyser/rogii-wellbore-geology-solution (docs/RECIPE.md, submissions/README.md, features/README.md) |
| Evidence level | Competition writeup plus code, with a full ledger of all 13 submissions (CV, public, private). Not peer-reviewed. The Kaggle writeup page could not be rendered by the fetcher; details come from the GitHub docs. |
| Extreme summary | A 1-D U-Net per well with cosine candidate matching and an expected-value decode over 512 candidate depths (0.5 ft, temperature 2.0), trained with synthetic-well pretraining (lineage: 26th place, Tucker Arrants). Stacked hierarchically: 30-checkpoint blend -> per-well hard gate vs an XGBoost margin model -> width-96 minus width-48 'scale direction' -> seed-averaged width-64 direction, each added with a fold-safe coefficient. CV 6.2518, public 6.004, private 6.329 (14/6,125, gold). The unselected simple ridge/simplex blends scored better privately (6.266). |

### Problem formulation

| Field | Value |
|---|---|
| Task definition | Predict blind-zone TVT for every row of each horizontal well from GR, trajectory, typewell and the known prefix; scored by pooled row RMSE. |
| Inputs | Horizontal GR (including a long GR window of 51), trajectory with inclination trig features, typewell (candidate curve matching), a 32-column history (known prefix) and a 640-column future; 96 temporal context channels. The XGBoost margin model uses 143 features in the absolute field frame (trajectory, 'Q3D', six-path and typewell stencil). The gate uses 1,250 target-free well aggregates. |
| Output representation | Per-column (block of 16 rows) categorical distribution over 512 candidate depths; point estimate by mean decode. Final submission is a linear combination of member point predictions. |

### Method

| Field | Value |
|---|---|
| Architecture | 1-D encoder-decoder U-Net over the well log at widths 32 (tctx96 bank), 48, 64 and 96, block size 16, dropout 0.25, plus a cosine trajectory refiner head and multi-head cosine candidate matching against the typewell. Four-level stack: (1) base = 30-checkpoint U-Net blend (fixed weights 0.608/0.222/0.169 over three sub-banks) and an XGBoost model predicting a margin on that blend; (2) GateV2 = hard per-well choice between them from 1,250 target-free well features; (3) add c_fold * (pred_w96 - pred_w48) at block 16 (raw width-96 alone was worse and rejected twice); (4) add c_fold * (mean of 16 width-64 checkpoints over 4 seeds - gated base), with c between 0.34 and 0.52 per fold. |
| Core mechanism | Variance reduction and bias correction by fold-safe, shrunk residual stacking. Each stage adds a 'direction' (difference between a stronger/averaged model and the current base) scaled by one coefficient fitted on the other four folds. Seed averaging matters because single runs are very noisy: two runs with the same seed and config on different GPUs differed by up to 34.5 ft on single rows (residual correlation 0.857), the same spread as between different seeds. The XGBoost works only as a margin on the incumbent (6.34 vs 15.93 when predicting raw TVT). |
| Training objective | Classification 1.0 (soft target, sigma 1.0) + regression 0.10 (Cauchy, scale 1.5) + slope 0.10 + history supervision 0.25 (sigma 0.75) + candidate curve prior logit 0.75 (sigma 10.0), prior sigma 3.0. 60 epochs with 10 pretrain epochs, batch 1, 4 samples per well, AdamW-style lr 3e-4, wd 1e-4, clip 1.0, patience 12, cosine annealing with T_max = epochs (the author warns that changing epochs changes the schedule shape). Synthetic augmentation: synthetic probability 0.50, flip 0.25, path noise 6.0, GR noise 1.0. |
| Decoding | Mean (expected value) over 512 candidates with decode temperature 2.0 (softens the softmax before the expectation); MC dropout samples 1. Then linear stacking. |
| Evidence scope | Whole lateral (640 future columns + 32 history columns) against the typewell; single well only (the XGBoost features are in an absolute field frame but no explicit neighbour-well evidence is described). |
| Prefix conditioning | Known prefix enters as 32 history columns with its own supervision loss (weight 0.25). No test-time fine-tuning. |
| Propose vs select | Both. Blends and 'direction' additions are proposals averaged linearly; GateV2 is a hard per-well selection between two bases (hard gate 6.3358 vs soft blend 6.3412 vs 198-feature gate 6.3418: tiny differences). |
| Physical constraints | Candidate curve prior logit (sigma 10) and slope loss encourage plausible paths; the structural identity is acknowledged; no hard layer-ordering or continuity constraint in the decode. |

### Data

| Field | Value |
|---|---|
| Dataset | ROGII Kaggle: 773 training wells, 3,783,989 scored rows, fixed 5-fold whole-well random split (folds_random.csv, never changed after week one), plus synthetic wells at probability 0.5. |
| Synthetic to real | Synthetic-well pretraining (10 pretrain epochs) then mixed training with synthetic probability 0.5, synthetic flip 0.25, path noise 6.0, GR noise scale 1.0; recipe credited to the 26th place public work (Tucker Arrants). No separate real-only alignment stage is described. |

### Results

| Field | Value |
|---|---|
| Metric and performance | All pooled RMSE. Submitted Construction A: CV 6.2518 / public 6.004 / private 6.329 (14/6,125). d96 scale stack: 6.3174 / 6.102 / 6.341. ABS-frame XGB margin: 6.3429 / 6.094 / 6.306. 30-checkpoint blend: 6.4052 / 6.110 / 6.313. Sum-to-one ridge (penalties 10 -> 0.01): CV 6.455-6.548, private 6.283 -> 6.266. 9-member simplex: CV 6.5648, private 6.266. Early best-CV single U-Net: 6.6846 / 6.258 / 6.485. Per-fold CV of the final model: 6.59, 6.61, 5.90, 6.64, 5.43. Main takeaways: seed/checkpoint averaging moved a single U-Net from 6.685 to about 6.40 CV (-0.28), and 6.485 to about 6.27-6.31 private (-0.18 to -0.22). The later stacking stages gained 0.15 CV but nothing privately (6.313 -> 6.329). |
| Comparability | Moderate for relative effects, weak for absolutes. Same competition data and metric family (pooled RMSE), but 5-fold CV over all 773 wells vs our fixed 155-well holdout, and the model family is a 1-D U-Net (not the 1st-place 2D cost-image ConvNeXt). The relative gains (seed averaging ~0.2-0.3 ft; stacking ~0 private) should transfer as a rough guide. |
| Validation protocol | Fixed 5-fold whole-well random split (not spatially/pad grouped); every weight, coefficient, penalty and gate fitted on 4 folds and applied to the 5th, checked by verify_construction_a.py (fold-safe refit within 0.033 of frozen coefficients; a leaky same-fold refit would score 6.207). CV-to-private gap: CV ordering did not predict private ordering. The best CV (6.2518) scored 6.329, while worse-CV ridge/simplex blends scored 6.266, a warning that complex stacking fits fold noise. |

### Ambiguity handling

| Field | Value |
|---|---|
| Repeated layers | Not addressed specifically. Implicitly through the soft classification over candidates with a curve-prior logit, multi-head cosine matching, and averaging many checkpoints, which averages across band choices (softer posteriors closer to the mean decode). |
| Drift | Slope loss (0.10) and the cosine trajectory refiner head target path shape; averaging seeds reduces per-run random drift. No explicit dip state or drift correction. |
| Uncertainty and calibration | No calibration analysis reported. Decode temperature 2.0 is a de facto calibration choice (softening). Their seed-reproducibility measurement (residual correlation 0.857 between identical-config runs) is a useful estimate of run-to-run variance. |

### Relevance to erudition

| Field | Value |
|---|---|
| Where to use | (a) Seed/checkpoint averaging of our ConvNeXt teacher (and FastViT student) by averaging per-column probabilities before the expectation decode, and distilling FastViT from the seed-averaged teacher posterior. (b) A decode-temperature sweep before the expectation, i.e. a calibration fix. (c) Fold-safe sum-to-one ridge or simplex blending of ConvNeXt/FastViT/AnchorCNN instead of learned per-point gates. (d) The 'margin on incumbent' GBM idea as a residual corrector (low expected value). |
| Error mode targeted | Blend/calibration part of band confusion (~15% blend + some wrong_mode through softer, better-calibrated posteriors) and run-to-run variance; also the measurement/protocol noise floor (seed variance). |
| Gap vs our implementation | Confirms our recipe family: expected-value decode over a depth axis, synthetic pretraining, soft classification targets plus a regression auxiliary; independent confirmation that this is what gold looks like. Gaps: (1) We report single models (FastViT 4.84) and one pair (4.69); we do not seed-average. Their data shows single U-Nets are very noisy (identical config reruns correlate 0.857 in residuals, which fits our +-0.39 ft single-run noise floor), and averaging many checkpoints gave -0.2 to -0.3 ft. (2) We do not apply a decode temperature; they used T = 2.0. For pooled RMSE with a mean decode, a softer (better-calibrated) posterior is exactly the 'better calibration' route in our metric analysis. (3) We have not tried fold-safe linear stacking with a shrinkage coefficient. Our learned per-point gates failed; their data say simple ridge/simplex blends beat complex gates on the private set, and hard per-well gates added ~0.007 CV. (4) Their 'width-96 minus width-48' scale direction suggests a cheap multi-capacity difference feature, but it helped only in CV and not privately. (5) Their history supervision loss (weight 0.25) supervises the known prefix explicitly; check whether our cost-image model is supervised on visible columns. |
| Proposed experiment | E1 (calibration, zero training): in kaggle_1st_place/solution/seq_NN_rescore.py or posterior_analysis.py, apply softmax temperature T in {0.8, 1.0, 1.25, 1.5, 2.0, 3.0} to the dumped logits/posteriors (or p^(1/T) renormalised) before the expectation decode. Fit T on training/OOF wells and report pooled RMSE on the 155 holdout wells for ConvNeXt, FastViT and the pair, with a paired bootstrap against the 4.84 / 4.69 controls. Also report blend/wrong_mode error shares. E2 (seed averaging): train 3 seeds of the best ConvNeXt-small recipe (same config as cnx_mobileone_ep150 or the 60-epoch baseline) and average per-column probabilities (not point predictions) before decoding; compare 1 vs 2 vs 3 seeds to estimate the ensemble curve; control = the existing single seed. E3 (distillation): distill FastViT-SA12 from the seed-averaged teacher posterior versus the single-teacher KD; control = the current 4.84 student. E4 (blending): replace learned gates with a sum-to-one ridge on {ConvNeXt, FastViT, AnchorCNN} point predictions, fitted fold-safe on training-well OOF (or a split of the holdout with cross-fitting), penalty chosen in-fold; control = equal-weight pair 4.69. |
| Implementation effort | E1 low (post-hoc, minutes). E4 low. E2/E3 medium: no new code, just repeated training runs and a probability-averaging script. |
| Distillation compatibility | Very good. A seed-averaged teacher posterior is a smoother, better-calibrated KD target (standard ensemble distillation), and the student stays a single FastViT, so rig inference cost is unchanged. The temperature finding transfers directly (KD already uses temperature; decode temperature is separate). |
| Error correlation with ours | Same model family, so members built this way (seeds of our own ConvNeXt) will be highly but not fully correlated. Their measured residual correlation between identical-config reruns is 0.857, which leaves real averaging headroom. A 1-D U-Net member in their style would be more diverse than another ConvNeXt seed, but it is weaker (single model ~6.7 CV on their protocol). |
| Risk | (1) Seed averaging mainly removes variance; it cannot fix systematic wrong-band choices when all seeds share the same bias (~51% band confusion). Oracle best-mode is 4.27, and averaging moves toward the ensemble posterior, not the oracle. (2) Their complex stacking gave CV gains that did not transfer (private got worse), so do not over-engineer per-well gates or scale directions; our own learned gates already failed. (3) Temperature may already be implicitly near-optimal if our training loss (smoothed target + Huber on expectation) calibrates well; the gain may sit below the +-0.39 ft noise floor. (4) GPU time on an 8 GB laptop makes 3+ ConvNeXt seeds expensive; memory might force smaller batch sizes that change the recipe. (5) Their split is random whole-well, not pad-grouped, so their absolute CV may be optimistic. |
| Notes | Lineage: the U-Net-per-well treatment, synthetic-well pretraining and expected-value decode come from the 26th place public work (Tucker Arrants, 'stacked U-Net synthetic wells' writeup), so this is a third independent confirmation of the family our 1st-place reproduction belongs to. Useful facts: the scheduler T_max = epochs couples epoch count to anneal shape (relevant to our 60 vs 150 epoch comparisons); runs are not bit-reproducible across GPUs; margin-on-incumbent beats raw prediction for GBMs by a huge factor (6.34 vs 15.93). The author finished 80th public -> 14th private. Private best 6.266 came from unselected simple blends, which argues for simple fold-safe blending in our setting too. |

### Other info

- **Id**: A16
- **Uncertain fields (omitted above)**:
  - code_availability
  - gpu_cost
  - inference_budget
  - metric_consistent_decode
  - target_parameterisation

---

## A17 Other ROGII writeups and repos (8th, 22nd, mycarta dcor toolkit, Lukyanenko blog, remaining top-30)

### Basics

| Field | Value |
|---|---|
| Name | Other ROGII writeups and repos: 8th (PF ranker + synthetic-pretrained U-Net), 22nd (five engines), 6th (91-PF row-level bagging), 7th (HMM + refiner U-Net), 10th (compass not a map), mycarta dcor toolkit, Lukyanenko blog (93rd), taichiiiiiiii physics fork |
| Year | 2026 |
| Venue link | 8th https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/8th-place-pf-ranker ; 22nd https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/22nd-place-solution-five-engines-and-a-hard-less (DOI 10.34740/kaggle/w/109172) ; 6th https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/6th-place-diverse-gr-representations-91-pfs ; 7th https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/7th-place-solution-hmm-unet-agent-is-all-you ; 10th https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/10th-place-solution-a-compass-not-a-map ; toolkit https://github.com/mycarta/rogii-geosteering-toolkit ; blog https://andlukyane.com/blog/rogii-geosteering-kaggle ; fork https://github.com/taichiiiiiiii/rogii-wellbore-geology-prediction ; index of all writeups https://kaggle.farid.one/ (lists 1st-14th, 17th-29th incl. 9th, 11th, 13th which were not read here) |
| Evidence level | Competition writeups with detailed ablation tables (8th, 22nd, 6th, 7th, 10th), one blog, one GitHub toolkit README (no scores). Not peer-reviewed. Read in full via a browser render in this study (Kaggle pages need JavaScript). |
| Extreme summary | Beyond the top 5, the gold writeups converge on three lessons: (1) error is per-well dip lock-in that accumulates toward the toe (8th: per-well line oracle 6.44 -> 3.33; 10th: 89% of SSE beyond 1,500 ft); (2) diversity of evidence sources (typewell vs self-prefix vs neighbour/sibling pseudo-typewells, PF/HMM legs, alternate CV splits) beats model sophistication, with soft averaging rather than selection; (3) public LB (52 wells) was anti-correlated with private while leak-free 773-well CV tracked private, and learned per-well gates/routers/drift correctors consistently failed to transfer. The mycarta distance-correlation cost was only a LightGBM feature with no reported score. |

### Problem formulation

| Field | Value |
|---|---|
| Task definition | Same competition task for all: predict blind-zone TVT per row of each horizontal well from GR, XYZ trajectory, typewell and known heel prefix; pooled row RMSE. |
| Inputs | 8th U-Net: 9 channels (hengck23 MTP input set incl. 4 dip/azimuth channels; removing them -> 11.37 vs 6.43). 8th ranker: 20 PF trajectories + trajectory dynamics, PF disagreement, GR match, geometry, nearby-surface agreement, spatial PF reliability from training-fold neighbours. 6th: 91 PF candidate curves from 6 PF configs x 5 GR references (typewell, self-prefix, XY neighbour within 1,500 ft, top-5 GR-similar neighbours, self-graft) + GR-free physical dip prior. 22nd: sibling-lateral likelihood tables (same master-log system, per-2 ft TVT bin GR bias/sigma), typewell templates (773 typewells -> 107 tops templates -> 54 master logs), PF channels, kriging prior. 10th: formation-top KD-tree point banks + regional polynomial, PF (16 seeds) + HMM candidates, 2-D ranking U-Nets. mycarta: multi-scale NCC, sliding distance correlation vs typewell, self-correlation to own prefix, Q-3D tortuosity (-0.107 RMSE, largest single group gain), signed azimuth, offset-well priors, landing-zone state. |
| Target parameterisation | 8th U-Net: TVT lattice H=384 rows x 0.5 ft (+-96 ft), W=768 MD columns x 16 ft; synthetic surfaces invented in u = TVT + Z space. 22nd solo: per-row dTVT with TVT = anchor + cumsum (dilated TCN); Ochir: beam DP in U = TVT + Z; Adil: signed-distance field over residual-TVT x MD chunks. 10th: state lattice on TVT, smoothness prior on dA/dMD. mycarta/Lukyanenko: row-level residual regression (persistence residual), TVT = kriged ANCC - Z + C. |
| Output representation | 8th U-Net: per-column categorical over lattice, expectation decode; 8th ranker: per-segment scores over 20 PF candidates converted to weights and interpolated along MD. 6th: row-wise attention soft-average over 91 candidate curves (top-k restriction hurt; full softmax best). 7th: HMM posterior -> learned emissions -> refiner U-Net point path + gated hedges. 22nd: 40-50 member shrink-NNLS + Ridge stack; Adil MTP K=4 + selector. 10th: LightGBM selector tempered-softmax blend of candidates + TCN/U-Net correctors. |

### Method

| Field | Value |
|---|---|
| Architecture | 8th (Yuto leg): smp ResNet34 U-Net, 9 input channels, InstanceNorm, column-wise CE; 200 epochs synthetic-only then 50 epochs real 50% + synthetic 50% (lr 3e-5, frozen BN stats, hflip 0.5, anchor-real-prob 0.2). 8th (bluepill leg): bank of ~30 then 20 CPU PFs (GR/typewell, self-reference, spatial-neighbour, smoothing, change-point) + ~40 selector models tried (TCN, GRU/PF-MoE, CatBoost ranking, LightGBM, Perceiver/Transformer), five selectors kept, fold-safe weights. Final: 0.5 PF-ranker + 0.5 U-Net bank with a tail-25% residual correction and MeanShift spatial folds. 6th: GPU PF (about 200x faster) with fixed-lag then full ancestral smoother (Kitagawa 1996; Doucet & Johansen 2009), 91 candidates -> GBM + TCN + candidate-curve attention ensembler, global gain a about 1.03. 7th: HMM (numba banded forward-backward) 5.676 -> learned emissions 5.301 -> refiner U-Net pretrained on ~6,000 synthetic wells with fabricated first-pass errors (u_ship = truth + lognormal-amplitude low-passed random walk), finetuned 8 epochs real -> 5.209 -> disagreement-gated hedges 5.2025 (CV). 22nd: five engines merged via shrink-NNLS + Ridge with decode-parity and solo-LB-proven membership rules and a Q-vs-HMM disagreement gate (>= 11.5 ft). 10th: physics engine (kriging/IDW + PF + HMM) -> 2-D ranking U-Nets -> LightGBM selector (49 features) -> TCN corrector (eta 0.10) -> second U-Net with per-row disagreement-gated alpha -> heel-anchored gain s = 1.024. |
| Core mechanism | Recurring mechanisms: (a) synthetic surfaces in u = TVT + Z space with realistic piecewise-linear statistics and adversarial decoys at +-15 ft (the Eagle Ford aliasing period), kept at 50% throughout real training as a regulariser (8th: f0 7.41 -> 6.92, f1 6.76 -> 6.50, f2 5.39 -> 5.23; fold 0 later 5.06); (b) decorrelated evidence sources: self-prefix and neighbour pseudo-typewells correlate only 0.72-0.79 with the typewell leg (6th); sibling-lateral likelihood tables carry +0.73 nat/row at the true depth (22nd); (c) soft hedging instead of hard selection under genuine ambiguity (6th: top-k hurt; 8th: scores -> weights interpolated along MD); (d) training a refiner on fabricated confidently-wrong inputs so it learns when to distrust the first pass (7th); (e) disagreement between two models as a gate for how much to trust a second model (7th, 10th, 22nd). |
| Training objective | 8th U-Net: column-wise CE; 22nd TCN: global RMSE on reconstructed cumsum + local SmoothL1 on increments (lambda = 30 set by gradient-norm balancing, since the cumsum term front-loads gradients ~500x); 22nd Typewell Editor: measurement-consistency loss (corrected typewell sampled at true TVT must match lateral GR); 7th refiner: point regression on synthetic fabricated-error inputs then real; 6th: PF parameters Optuna-tuned on 150-600 wells. |
| Decoding | 8th U-Net: posterior expectation, temperature 1.0. 6th: soft attention average over all candidates + global gain 1.03. 7th: HMM forward-backward posterior -> U-Net refiners averaged -> gated hedges g = sigma/(sigma+1). 10th: tempered softmax blend + projection to structural prior + gated corrections. 22nd: NNLS linear stack. |
| Metric consistent decode | Mostly yes: expectations and soft averages (RMSE-optimal); 6th explicitly found hard/top-k selection worse than full soft averaging, consistent with our note that mode-seeking raises pooled RMSE. The 10th-place retrieval test is the clearest warning: over 1,025 candidates the oracle best was 2.60 but the objective's pick scored 7.09 and the posterior-weighted average 6.54 vs the chain's 6.01. |
| Evidence scope | Whole lateral + typewell, plus multi-well evidence: XY neighbours, GR-similar neighbours, same-master-log sibling laterals, kriged formation surfaces (6th, 8th, 10th, 22nd, Lukyanenko). |
| Prefix conditioning | Self-prefix used as a pseudo-typewell (6th selfGR; 22nd Ochir directional self-reference split by drilling direction; mycarta self-correlation); prefix-calibrated typewell affine (6th tw-affine; 22nd Typewell Editor robust affine on prefix). 22nd fake-anchor augmentation replays each training well with the anchor pulled back (~3x far-anchor labels; CV 6.88, LB 6.367). |
| Propose vs select | Strongly propose-then-select/soft-average: 6th (91 candidates), 8th (20 PFs + ranker), 10th (engine candidates + selector), 22nd (40-50 member stack). Their main lesson: headroom comes from candidates that are wrong in a different direction; stronger selectors saturate at ~0.99 correlation with the base (6th). |
| Physical constraints | u = TVT + Z surface space for synthetic generation and beam DP (8th, 22nd); GR-free physical dip prior injected into PF likelihood (6th); 22nd's exact GR-normalization invariant beta = 0.99904 +- 0.001; bounded tanh gates chained to a validated prior (22nd Ochir); projection toward structural prior (10th). |

### Data

| Field | Value |
|---|---|
| Dataset | ROGII Kaggle: 773 training wells, 3,783,989 scored rows; public LB 52 wells, private 148 wells (per 10th and 22nd). Hidden set more isolated: median nearest-training-well distance 683 ft vs 470 ft among training wells (10th). Hidden set contains re-interpreted copies of some training wells (22nd, Adil: 0.45 train_TVT + 0.55 pred blend gave about -0.03 to -0.05 LB). |
| Synthetic to real | 8th: 200 epochs synthetic-only gives zero-shot 7.62; replacing pretrain -> finetune with mixed continuation at synth_prob 0.5 broke a 6.6 plateau. Generator: u_new = RDP skeleton of real surface (eps 2.0) + fold-local piecewise-linear deviation (segment median ~98 ft, AR(1) slope rho ~0.248, jumps 3.1%), TVT = u - Z, rejection if outside real drift envelope (<= 12 resamples), decoy_prob 0.25 at +-15 ft, real trajectory and GR residual/missingness preserved. Pretraining seed is a lottery uncorrelated with pretrain val (9.08 -> 6.50 by re-rolling). 7th: ~6,000 synthetic wells for refiner pretraining. 22nd: donor-relocation synthetic wells gave blend +-0.00 ('clearest miss'). Lukyanenko: synthetic pretraining tried once, not explored. |

### Results

| Field | Value |
|---|---|
| Metric and performance | 8th: private gold (8th); PF-ranker OOF (stratified) 5.709, median well 2.92, P90 8.51; U-Net stratified 6.437 / group 6.847 CV, 0.7/0.3 blend 6.309; public 5.5 went to 6.7-7.0 private. 22nd: public 4.901 (rank 2) -> private 6.543 (22nd); all 50 submissions +1.47 +- 0.95 ft worse on private; CV(773) vs private Pearson +0.92, public vs private Spearman -0.48 in endgame band; best private 6.386 was a sub labeled 'overfit to CV'. 6th: CV 5.458 / public 5.626 / private 5.984. 7th: CV 5.2025 / public 5.518 / private 6.057 (later variants 6.009-6.015 private, 5.083 CV). 10th: CV 5.719 / public 5.620 / private 6.269; dropped neighbour stage would have scored 6.179. Lukyanenko: final public 6.042 / private 7.570 (93rd); particle filter 9.001 -> 7.801 public; AnchorCNN reproduction 7.148 private. taichiiiiiiii: public 6.389 / private 9.327. mycarta: no scores disclosed. |
| Comparability | Weak-moderate: 5-fold CV over 773 wells or public/private LB, not our fixed 155-well holdout. Relative effects (e.g. group-vs-stratified blend -0.13, seed variance, soft vs hard selection) are the transferable part. Our best single model (4.84) is already below all their CVs, but numbers are on different well sets. |
| Validation protocol | 8th: GroupKFold by geographic clusters (MeanShift bw = 7000 ft, 40 clusters) and stratified 5-fold; the two splits make different OOF errors (residual corr 0.826; worst-30 overlap 19/30) and mixing 30% group-scheme models helped. mycarta: StratifiedGroupKFold stratified by signed azimuth, median TVT bin and XY grid bin, grouped by well; spatial BlockKFold (Verde) rejected as over-pessimistic because test wells interleave with training wells. 3rd (local 3rd.txt): 5 GroupKFold patterns averaged. 10th: paired SD of a configuration difference 0.168 on 52 wells vs 0.093 on 148, so public gaps < 0.10 are unreadable; byte-identical resubmission spread about 0.03. 22nd: 'cv_refit' (leave-fold-out refit of the whole stacker); decode-parity rule. 7th: paired bootstrap on 773 wells gave false confidence (1.75%) in a public inversion. |

### Ambiguity handling

| Field | Value |
|---|---|
| Repeated layers | Explicitly the dominant failure: 8th adds decoy matches at +-15 ft (Eagle Ford aliasing period) during synthetic training as adversarial training against adjacent-bed lock-on; row-shift TTA (+-4/8 rows) did not help (+0.05 to +0.08). 6th: remaining error is a per-segment upper/lower branch choice that GR alone cannot resolve, so hedge by soft averaging. 22nd: lateral GR tracks the master log with corr 0.865 and ~12 API scatter, which is the multimodality; sharpened (honest-sigma) likelihoods failed 5/5 in PF because broad likelihoods provide implicit robustness. 10th: local matcher AUC 0.87-0.95 on perturbations but cannot find the truth among 1,025 candidates. |
| Drift | 8th: last 25% of the well holds 41.6% of SSE because dip error accumulates; per-well constant oracle 4.58, per-well line 3.33, perfect tail only 4.61. 10th: error grows like a random walk from the heel (mean /error/ 0.32 ft at 50 ft -> 4.55 ft at 5,000 ft), signed bias ~0, correlation length 400-500 ft; per-well line/quadratic/cubic oracles 6.70/5.34/4.42; drift extrapolation from the prefix 12.60 vs flat 15.91; first fifth of the scored zone predicts the last with r = 0.131. 22nd: OOF-trained drift-direction correctors CV -0.25 but LB +0.34 ('drift direction is the info wall'). Our holdout (0801_V2_ep150, 4.978): per-well line oracle 3.08, 58.6% of SSE beyond 3,000 ft past the heel. |
| Uncertainty and calibration | No reliability/ECE analysis in any of them. Practical uncertainty proxies: disagreement between two refiners (7th, g = sigma/(sigma+1)), Q-vs-HMM disagreement >= 11.5 ft (22nd, CV -0.052 -> LB -0.167), log-disagreement gated alpha (10th, b0 0.35-0.50, b1 -0.08 to -0.14, stable across folds). 10th's triage: a leak-free bad-well detector (engine disagreement + matching-cost entropy) reached rank corr 0.41 but every treatment of flagged wells made things worse. 6th/10th global de-shrink gains 1.03/1.024. |

### Relevance to erudition

| Field | Value |
|---|---|
| Where to use | (1) Synthetic generator (seq_NN_dataset.py Simulator): add 8th-place surface-space generation with +-15 ft decoy injection and keep 50% synthetic during real training (we train with 85% synthetic; check whether we switch off synthetic late). (2) Alternative references as extra cost-image channels: self-prefix pseudo-typewell GR and sibling/neighbour pseudo-typewell GR (gr_abs_diff against each), or sibling-lateral per-TVT-bin GR bias/sigma tables as a 'bias-corrected emission' channel (22nd: -0.42 NN screen win). (3) Ensemble protocol: train members on different fold/split schemes (stratified vs spatial-cluster group) and blend at a fixed weight rather than seed-only ensembles. (4) Gating: a disagreement-gated alpha between ConvNeXt and AnchorCNN/FastViT (per-row alpha = clip(b0 + b1 log1p/d/)) as a 2-parameter replacement for our failed learned per-point gates. (5) Evaluation: adopt paired-difference noise estimates and a decode-parity check. |
| Error mode targeted | Band confusion (decoy training, alternate references, hedging), drift (surface-space synthetics), and measurement/protocol (split diversity, noise floors, decode parity). |
| Proposed experiment | E1 (decoy synthetic, highest priority): in experiments/bilzard/seq_NN_dataset.py Simulator, add an option decoy_prob = 0.25 that, for a synthetic well, overwrites a random 200-800 ft segment of the rendered GR with the typewell response at TVT +- 15 ft (or +-P where P is the dominant autocorrelation period of that typewell), keeping labels at the true TVT; train 60 epochs vs the 5.47 baseline, report wrong_mode/blend/multimodal_miss shares from posterior_analysis.py and pooled RMSE with paired bootstrap. E2 (self-prefix reference channel): build a pseudo-typewell from the well's own known prefix (TVT_input, GR) binned at 0.5 ft, and add 'self_gr_abs_diff' (/GR - selfTW(level)/, NaN outside prefix coverage + an is_nan mask) to unet_static_channels in seq_NN_cfg.py; for synthetic wells generate it from the synthetic prefix. Control: baseline at 60 epochs. E3 (sibling tables): group training typewells into master-log systems (seq_NN_master_typewell.py already exists), pool laterals of same-system training wells (fold-safe, leave-one-well-out) into per-2 ft TVT-bin GR bias and sigma, and add 'sib_bias_corrected_diff' = /GR - (TW + bias)/ / sigma as a channel. E4 (split diversity): train a second ConvNeXt on a spatial-cluster split of the 618 training wells (MeanShift bandwidth ~7000 ft on well XY centroids) for model selection/early stopping, then blend 0.7/0.3 with the standard member by averaging posteriors; control = two seeds of the standard split (to separate split diversity from seed diversity). E5 (tiny gate): replace learned per-point gates with alpha_i = clip(b0 + b1 log1p/p_cnx_i - p_anchor_i/, 0, 0.6), fit b0, b1 by cross-fitting on holdout halves or training OOF; control = equal-weight pair 4.69. |
| Implementation effort | E1 low-medium (Simulator option). E2 medium (new channel for real and synthetic wells). E3 medium-high (fold-safe table building, master-log grouping exists). E4 medium (clustering + one extra training run). E5 low. |
| Distillation compatibility | Good for E1-E3 (training-data or input-channel changes; FastViT sees the same inputs, KD on posteriors unchanged). E4: a split-diverse teacher pair gives a smoother KD target. E5 operates after decoding, so the student would need its own pair partner. |
| Error correlation with ours | Alternative references (self, neighbour, sibling) are the most decorrelated sources documented (0.72-0.79 correlation vs typewell legs in 6th; sibling NN channels biggest screen win in 22nd). Split-scheme diversity gives residual corr 0.826 (8th) vs 0.857-0.992 for seed/checkpoint ensembles (14th, 8th). PF/HMM legs have much lower correlation (0.357 for HMM vs CNN in 8th) but are weak alone (11.94). |
| Risk | (1) Public/private instability: several findings (neighbour stage in 10th, gates in 8th/22nd) moved CV and LB in opposite directions; our single 155-well holdout has the same small-sample problem (noise floor +-0.39 ft). (2) Neighbour/sibling information may over-represent training geometry; hidden wells are more isolated (683 vs 470 ft), so price E3 under reduced donor density. (3) Decoy injection may make the model overly hedged (broader humps, ~13% of our error) if the decoy rate is too high. (4) Seed ensembles did not help 8th and 10th (residual corr 0.992; 0.186 ft spread), contradicting 14th; measure our own diversity first. (5) Many writeups were agent-written with limited verification; numbers are self-reported. |
| Notes | Useful measured facts: 8th pretraining-seed lottery (fold 9.08 -> 6.50 by re-rolling; pretrain val uncorrelated with final), stage-dependent noise floor (0.955 ft at pretrain ep10 vs 0.008 ft for mixed continuation); 8th post-hoc savgol/median smoothing best -0.014 (consistent with our failed continuity decode); 10th error concentration: worst 39 wells (5%) = 43.9% of SSE; per-well RMSE p50 3.31. 22nd: 773 typewells collapse to 54 master logs (same as 2nd place), and the hidden set contains re-interpreted copies of training wells (small free gain, but this is a leaderboard artifact, not rig-relevant). mycarta: within a lateral TVT and Z are uncorrelated (r ~ 0) though globally r = -0.96; tortuosity (Q-3D) was the largest feature gain for LightGBM (-0.107), AEON Catch22/ClaSP features overfit (+0.476). Lukyanenko blog: AI-agent failure modes (hardcoding the 3 visible test wells, silent OOF row drops, unverified attribution, cheap smoke tests rejecting ideas, memory pollution); kriged ANCC surface + PF + stacking reached 7.43-7.57 private, i.e. row-level tabular framing was the ceiling. Not read in this pass: 9th, 11th, 13th (synthetic factory cited by 22nd), 17th-29th except 22nd; worth a follow-up for the 13th generator. |

### Other info

- **Id**: A17
- **Uncertain fields (omitted above)**:
  - authors_org
  - code_availability
  - gap_vs_our_implementation
  - gpu_cost
  - inference_budget

---

## A18 Kaggle discussion identity TVT + Z - ANCC = const per well (~0.03 ft)

### Basics

| Field | Value |
|---|---|
| Name | Structural identity TVT + Z - ANCC = const per well (ROGII Kaggle discussion; reused in 2nd, 4th, 8th, 10th, 14th, 22nd writeups and the Lukyanenko blog) |
| Year | 2026 |
| Venue link | Kaggle ROGII competition discussion + writeups: 2nd place (local C:/Users/Aadinarayan R/Desktop/CSE/projects/Erudition/2nd.txt, eq. 1 dTVT = dz_layer - dz); 10th place https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/10th-place-solution-a-compass-not-a-map ; 8th place https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/8th-place-pf-ranker ; 22nd place https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/22nd-place-solution-five-engines-and-a-hard-less ; blog https://andlukyane.com/blog/rogii-geosteering-kaggle ; surface columns in TVD: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/701034 |
| Code availability | No dedicated code; the identity is a one-line data fact. Local verification script written for this study (scratchpad idcheck.py / seg.py / orc.py, not committed). In our repo the surface S = TVT + Z already appears in experiments/bilzard/seq_NN_dataset.py (S_avg from the six formation columns, used by the Simulator's init_xy_local_fits for synthetic XY shifts) and in seq_NN_geo_prior.py (fold-safe IDW prior on S_rel = S - S0). |
| Evidence level | Verified directly on the local training data (773 wells, 3,783,989 suffix rows) in this study, plus consistent statements in 6+ competition writeups. Not peer-reviewed, but it is an exact property of how the host built the labels. |
| Extreme summary | TVT is exactly the expert-interpreted formation surface minus the wellbore Z plus a per-well constant: TVT + Z - ANCC is constant within every well to CSV rounding (per-well std 0.0066 ft, range 0.03-0.07 ft), and the six formation columns are parallel offsets of one surface (ANCC - BUDA std 0.003 ft). It is not a test-time constraint, because ANCC is absent from test files; it is a reparameterisation: predicting TVT = predicting the smooth surface A(MD) = TVT + Z, and all drift is error in the dip dA/dMD. |

### Problem formulation

| Field | Value |
|---|---|
| Task definition | Not a model. A data identity: for each training well, C_well = TVT(MD) + Z(MD) - ANCC(MD) is constant. Consequence: TVT(MD) = TVT_anchor + [A(MD) - A_anchor] - [Z(MD) - Z_anchor], so the only unknown in the blind zone is the change in the geological surface A along the path. |
| Inputs | Train horizontal CSV columns: MD, X, Y, Z, ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA, TVT, GR, TVT_input. Test horizontal CSV columns: MD, X, Y, Z, GR, TVT_input only (verified on data/test/000d7d20__horizontal_well.csv). So ANCC and the other formation columns exist for training wells only (7 of 773 wells have ANCC entirely NaN; BUDA is complete for all 773). Our U-Net inputs (experiments/bilzard/seq_NN_cfg.py unet_static_channels) do not include any formation column; they include z_diff (row-to-row dZ in the suffix), tw_tvt_rel, seen_tvt_rel and tw_seen_tvt_abs_diff; the geo_* channels (geo_s_rel, geo_tvt_rel, geo_tvt_abs_diff, geo_dS, geo_tvt_diff, delta_geo_trend_rbf_5) are commented out in the baseline channel list. |
| Target parameterisation | Surface space u = A = TVT + Z (8th place U-Net synthetic generator, 22nd place Ochir beam-DP 'U = TVT + Z', 15th-silver style trackers) versus TVT space with dZ as input (2nd place: feed dz/dMD so the model only estimates the smooth dz_layer; our 1st-place reproduction: z_diff channel, TVT lattice). The 10th place author reports getting this backwards twice: build the state lattice on TVT (median suffix swing 26.4 ft, p95 54.4 ft) not on A (157.5 ft median swing, same as Z's 160.6 ft), but put the smoothness prior on dA/dMD (dip: p1/median/p99 = -0.065/+0.017/+0.072; std of dip change per 20 ft node 0.0096). |
| Output representation | Not applicable (identity). Implied options: per-column distribution over TVT (ours), or a piecewise-linear A(MD) with few breakpoints (expert interpretations are 'sequences of straight lines': 8th place measured median segment length about 98 ft and fault-like jumps at 3.1% of segments). |

### Method

| Field | Value |
|---|---|
| Architecture | Not applicable. How top solutions exploited the identity: (1) 2nd place: dz/dMD input so the AnchorCNN learns only the structural slope; steps in dz_layer/dMD occur in about 10% of columns. (2) 4th place: LightGBM quantile regression predicts dA/dMD (A = ANCC) per 750 ft segment from X, Y, Z, azimuth and pitch, integrates it to a topography TVT prior plus a confidence band, and renders the band as the green image channel for a ConvNeXt V2 that predicts 8 control-point offsets. (3) 8th place: synthetic generator invents u_new = S_real (RDP piecewise-linear skeleton of a real surface) + d (piecewise-linear deviation field, AR(1) slope deviations rho about 0.248), then TVT_new = u_new - Z with the real trajectory; inventing TVT directly was detected by a real-vs-synthetic domain gate at AUC 1.0. (4) 10th place and Lukyanenko: kriged/IDW formation-top surfaces from training wells evaluated along the query path as a structural prior. (5) Our repo: Simulator.init_xy_local_fits fits dS = a dX + b dY on S_avg for synthetic XY shifts; seq_NN_geo_prior.py builds a fold-safe anisotropic IDW WLS prior on S. |
| Core mechanism | Drillers steer Z to follow the formation, so dTVT = dA - dZ with dZ known exactly for every row at test time. The high-frequency part of TVT is therefore free (it is -dZ), and all remaining uncertainty is the low-frequency geological surface A (dip plus occasional kinks/faults). Our own measurements on the 155-well holdout (0801_V2_ep150 predictions, pooled 4.978 ft) show the residual is dominated by low-order structure: removing a per-well constant gives 3.80 ft, a per-well line 3.08 ft, a heel-anchored slope only 3.51 ft, a per-well quadratic 2.46 ft; 58.6% of squared error lies more than 3,000 ft past the heel and 87% more than 1,500 ft past it. This is the same 'random-walk dip error' picture the 8th (6.44 -> 3.33 with per-well line oracle) and 10th (error correlation length 400-500 ft; 89% of SSE beyond 1,500 ft) places report. |
| Training objective | Not applicable. Possible uses: an auxiliary loss on dA/dMD (dip) or on the piecewise-linear structure of A (training labels available from TVT + Z for every training row, even for the 7 wells without ANCC), or a smoothness penalty on second differences of A along the decoded path. |
| Decoding | Not applicable. Note for decoders: a per-column expectation over the TVT lattice is still Bayes-optimal for pooled RMSE; converting to A = TVT + Z changes nothing in the decode itself (Z is deterministic). |
| Metric consistent decode | Neutral. The identity is a change of variables with a known deterministic offset (Z), so RMSE on TVT equals RMSE on A; any mean decode in either space is equally RMSE-optimal. |
| Evidence scope | Per well (the constant C_well is per well); cross-well use requires spatial interpolation of A (kriging/IDW/tabular dip models), which several teams found transfers poorly to the hidden set. |
| Prefix conditioning | The identity lets the known prefix fix C_well and A_anchor exactly (TVT_input + Z at the last known row), which our dataset already does (s0 = TVT_input[last_seen] + Z[last_seen] in seq_NN_dataset.py; tw_tvt_rel / seen_tvt_rel are relative to that anchor). The prefix also gives the recent dip dA/dMD; extrapolating it is weak (our check: last-200-row prefix slope extrapolation of A = 43.4 ft pooled over all 773 training suffixes vs 107.5 ft for holding A constant; 10th place: 12.60 vs 15.91 flat). |
| Propose vs select | Not applicable; can feed a proposal (structural prior channel / candidate) or a constraint. |

### Data

| Field | Value |
|---|---|
| Dataset | ROGII Kaggle training set: 773 horizontal wells with paired typewells, 3,783,989 scored suffix rows, MD step 1 ft (median). Test/hidden files lack the formation columns. |

### Results

| Field | Value |
|---|---|
| Metric and performance | Identity accuracy (this study, local data): per-well std of TVT + Z - ANCC mean 0.00657 ft (max 0.0087), per-well range 0.03-0.07 ft (matches the discussion's '~0.03 ft' and the 10th place '0.07 ft = CSV rounding'); identical for ASTNU and BUDA. Oracles on our holdout (0801_V2_ep150, pooled 4.978 ft): per-well constant removed 3.80, per-well line removed 3.08, heel-anchored slope removed 3.51, per-well quadratic removed 2.46. External oracles: 10th place on A with piecewise-linear fits: one anchored line 7.59, one free line 5.96, two segments 2.43, three segments 1.44 (relative to their chain). Uses: 4th place topography-only prior private 12.6 (weak) -> 10.8 with a stronger TabPFN/RealMLP/LightGBM dip ensemble; 6th place GR-free physical dip prior alone CV 10.1 / private 9.34, but injected into PF likelihood CV 6.98 -> 6.50 and aligned CV with LB. |
| Comparability | Oracle numbers on our holdout are directly comparable (same 155 wells, same pooled RMSE). External numbers are 5-fold CV or LB on other models; use them only for direction. |
| Validation protocol | The identity itself needs no validation. Anything built from other wells' A (kriging, IDW, dip regression) must be strictly fold-safe (6th place: leak in the dip field inflates CV and collapses on private), and must be priced under realistic donor density: 10th place found the hidden wells are more isolated (median distance to nearest training well 683 ft vs 470 ft among training wells); a neighbour-surface stage gained 0.18 CV but lost 0.33 public (2.95 sigma) [though it would have scored 6.179 vs 6.269 private, so the evidence is mixed]. |
| Inference budget | Negligible for the identity; a tabular dip model or IDW prior is CPU-cheap (our geo prior takes about 10 s for 155 wells after caching). |

### Ambiguity handling

| Field | Value |
|---|---|
| Repeated layers | No direct help: the identity does not tell which repeated GR band the bit is in. It helps indirectly by constraining how fast A can change (dip bounds), which narrows the set of plausible band jumps; a jump to a band 15 ft away within a short MD span would need an implausible dip or a fault. |
| Drift | This is the main target. Drift = integrated error in dip dA/dMD. Levers: (a) a better dip prior from geometry (4th place tabular dA/dMD with quantile band; 6th place fold-safe dip field), (b) a piecewise-linear structural prior on A, (c) synthetic training in A space so the network learns realistic dip statistics. Caveat from the 10th place: their final model already beat the best single straight line, so below about 5 ft on their scale gains must come from fitting bends, not a single drift slope; our ConvNeXt (4.98) is at 3.08 after removing a per-well line, so a single-line correction could at most gain about 1.9 ft if it were perfectly predictable, but 10th and 22nd place found drift direction essentially unpredictable from observables (OOF drift correctors: CV -0.25, LB +0.34). |
| Uncertainty and calibration | The 4th place produced calibrated confidence intervals for dA/dMD (Pearson corr between predicted sigma and actual error 0.31 -> 0.40 with the stronger ensemble) and rendered them as an image channel, letting the vision model learn when to trust topography. Our geo prior exposes geo_nbr_std, geo_nbr_dS_std and geo_radial_extrap_score diagnostics that serve the same purpose but are not in the baseline channel list. |

### Relevance to erudition

| Field | Value |
|---|---|
| Where to use | (1) Input channel: a structural-prior probability channel in the cost image (P(TVT > level / dip prior)), like the 4th place green channel, built from a fold-safe dip model with an uncertainty band; our geo_* channels are the closest existing code. (2) Auxiliary target: predict dA/dMD per column as a second head (label = d(TVT + Z)/dMD, available for every training row). (3) Synthetic generator audit: make sure synthetic TVT = A_synthetic - Z with A piecewise linear. (4) Diagnostics: report errors in A-slope space to separate drift from band confusion. |
| Error mode targeted | Drift (~25% of squared error), secondarily broad humps; not band confusion. |
| Gap vs our implementation | What we already do: (a) z_diff is an input channel, so the network sees dZ and can learn dTVT = dA - dZ (the 2nd-place insight); (b) the anchor fixes C_well via s0 = TVT_input + Z at the last known row; (c) the Simulator uses S_avg (mean of the six formation columns, i.e. TVT + Z + const) for XY-shift synthetic augmentation; (d) seq_NN_geo_prior.py builds a fold-safe anisotropic IDW WLS prior on S with neighbour diagnostics. What we do not do: (1) no geo/structural channel in the baseline U-Net input (geo_* channels commented out; the XY geo-prior was tried only as a post-hoc hump picker, which failed); (2) no uncertainty-banded structural prior channel (4th place style) and no tabular dip model from X, Y, Z, azimuth, inclination; (3) no auxiliary dip (dA/dMD) head or smoothness loss on A; (4) no explicit check that synthetic paths follow piecewise-linear A. The identity cannot be used as a hard test-time constraint: ANCC is absent from test, so TVT + Z - ANCC = const adds no information beyond what z_diff and the anchor already give. |
| Proposed experiment | E1 (zero training, diagnostic): extend posterior_analysis.py (or a new script next to posterior_geo_pick.py) to decompose holdout error into A-space components per well: residual after removing constant, line, quadratic, and after a 2-3 segment piecewise-linear fit; attribute each well's error to drift vs band confusion using the existing wrong_mode/blend labels. Control: current 60-epoch and ep150 predictions. E2 (input channel): re-enable a subset of the existing geo channels in experiments/bilzard/seq_NN_cfg.py unet_static_channels, specifically geo_tvt_abs_diff (/level - prior TVT/) plus a new 'geo_prior_cdf' channel = Phi((level - prior_TVT)/sigma_prior) where sigma_prior comes from geo_nbr_dS_std scaled by distance past the heel; keep everything else fixed; train 60 epochs and compare with the 60-epoch baseline (5.47) on the 155 holdout wells with paired bootstrap; then 150 epochs if it passes the +-0.39 noise floor. Also run it with the prior computed under 'realistic donor density' (drop training wells within ~500 ft of each holdout well) to price the 10th-place isolation effect. E3 (tabular dip prior, 4th-place style): fit LightGBM quantile models (q10/q50/q90) for dA/dMD per 250-750 ft segment from segment-midpoint X, Y, Z, azimuth, inclination, fold-safe over the 618 training wells; integrate to a TVT prior band and use it as the E2 channel instead of IDW. E4 (auxiliary head): add a per-column regression head for dA/dMD (label d(TVT+Z)/dMD smoothed over 16-32 ft) with weight 0.1 in seq_NN_train.py; compare with baseline at 60 epochs. E5 (synthetic audit): in seq_NN_dataset.py Simulator, verify that synthetic TVT paths satisfy TVT = A - Z with A piecewise linear; if not, add an 8th-place style generator mode (RDP skeleton of a real A + AR(1) slope deviation + decoy +-15 ft) and compare. |
| Implementation effort | E1 low (analysis script). E2 low-medium (channels already implemented; add one CDF channel and retrain). E3 medium (new tabular model + fold-safe integration). E4 low-medium (one extra head + loss). E5 medium. |
| Distillation compatibility | Good. Extra input channels or an auxiliary head carry over to the FastViT student unchanged (the student gets the same input tensor; KD on output posteriors is unaffected). |
| Error correlation with ours | A structural-prior channel changes what the same network sees, so diversity is modest; a standalone tabular dip prior has very different errors (weak alone: 4th place 12.6/10.8, 6th place CV 10.1) and is most useful as an input or likelihood term, not as a blend member. |
| Risk | (1) Our XY geo-prior already failed as a post-hoc hump picker, and 10th and 4th place both observed that neighbour/topography information improves CV but can hurt the hidden set because hidden wells are more isolated and the model over-trusts topography (4th place deliberately kept a weaker topography model). (2) Drift direction appeared unpredictable in 10th and 22nd place studies (drift correctors CV -0.25 / LB +0.34; per-well blend weights no better than a scalar). (3) The identity itself is not a new constraint at test time; expectations of 'removing drift' from it are overstated. (4) Gains may fall under the +-0.39 ft single-run noise floor, requiring paired bootstrap or multiple seeds. (5) Auxiliary dip loss could conflict with the per-column CE if weighted too high. |
| Notes | Local verification details: 773 training wells; ANCC NaN in 7 wells (0.9% of rows); per-well std of TVT + Z - ANCC mean 0.00657 ft (range 0.0057-0.0087), per-well max-min 0.03-0.07 ft; identical behaviour for ASTNU and BUDA; ANCC - BUDA constant within a well (std mean 0.0027 ft), i.e. the six columns are one surface plus constants, so the formation columns carry one degree of freedom per row. Surface columns are in negative TVD units like Z (discussion 701034). The 22nd place also found a second exact per-well invariant from the host's GR normalization: beta = <GR_lateral, TW_GR(TVT_true)> / <GR, GR> = 0.99904 +- 0.001, which pins a scalar of the hidden path but has a multimodal root set (median 8 solutions within +-40 ft); worth a separate look as a test-legal constraint. Holdout gain check (this study): a heel-anchored global gain on (pred - TVT_anchor) is 0.98 in-sample (4.978 -> 4.971) and 0.96 cross-fitted (5.011 +- 0.064, worse), so the 6th/10th place 'de-shrink' gains (a ~1.02-1.03) do not transfer to our ConvNeXt, which is not amplitude-shrunk. |

### Other info

- **Id**: A18
- **Uncertain fields (omitted above)**:
  - authors_org
  - gpu_cost
  - physical_constraints
  - synthetic_to_real

---

## A19 Public geosteering data beyond Kaggle (GWC 2020 typelog on DataverseNO, other open typelogs)

### Basics

| Field | Value |
|---|---|
| Name | Public geosteering data beyond Kaggle: GWC 2020 semi-final typelog (DataverseNO), Alyaev & Elsheikh 2022 stratigraphy-realization generator (DataverseNO), GWC 2021 '10,000 interpretations and decisions' (Zenodo), Kaggle Geology Forecast Challenge 2025 (horizon traces), and open vertical/deviated log repositories usable as extra typelogs (FORCE 2020, Volve, KGS, UK NDR, NLOG, BC Energy Regulator) |
| Authors org | GWC 2020 typelog: D. Miner, I. Kuvaev (ROGII), S. Alyaev (NORCE). Stratigraphy realizations: S. Alyaev (NORCE) with A. H. Elsheikh (Heriot-Watt). GWC 2021 dataset: S. Alyaev, Y. Cheraghi (UiS), I. Kuvaev, S. Clark, A. Zhuravlev (ROGII). Geology Forecast Challenge: S. Alyaev, N. Blaser, A. Fedorov, I. Kuvaev (NORCE / UiB / ROGII). FORCE 2020: FORCE / Norwegian Offshore Directorate (Sodir) with Equinor et al. Volve: Equinor and licence partners. KGS: Kansas Geological Survey. UK NDR: North Sea Transition Authority. NLOG: TNO for the Dutch ministry. BC: BC Energy Regulator. |
| Year | GWC 2020 typelog dataset 2021; stratigraphy realizations 2022; GWC 2021 dataset released 2025 (competition 2021); Geology Forecast Challenge 2025; FORCE 2020 (2020); Volve released 2018; UK NDR open since 2019; KGS/NLOG/BCER ongoing |
| Venue link | GWC 2020 typelog: DataverseNO doi:10.18710/20VIVT https://dataverse.no/dataset.xhtml?persistentId=doi:10.18710/20VIVT ; mirror https://github.com/alin256/gwc2020-semi-unconventional . Stratigraphy realizations + generator script (supporting data for Alyaev & Elsheikh, 'Direct multi-modal inversion of geophysical logs using deep learning', Earth and Space Science 2022, arXiv 2201.01871): DataverseNO doi:10.18710/1F9GYH . GWC 2021: Zenodo doi:10.5281/zenodo.15190734 https://zenodo.org/records/15190744 and https://github.com/geosteering-no/10000-geosteering-interpretations-and-decisions (paper: Cheraghi et al., Applied Computing and Geosciences 2025, doi:10.1016/j.acags.2025.100237). Geology Forecast Challenge: https://www.kaggle.com/competitions/geology-forecast-challenge-open ; 1st-place code https://github.com/geosteering-no/Geology-Forecast-Challenge-Gold . FORCE 2020: https://www.sodir.no/en/force/Previous-events/2020/results-of-the-FORCE-2020-lithology-competition/ . Volve: https://www.equinor.com/energy/volve-data-sharing . KGS LAS: https://www.kgs.ku.edu/Magellan/Logs/ . UK NDR: https://ndr.nstauthority.co.uk/ . NLOG: https://www.nlog.nl/en/data . BCER: https://www.bc-er.ca/energy-professionals/online-systems/elibrary/ |
| Extreme summary | Public geosteering data with the Kaggle task's exact structure (lateral GR + typewell GR + TVT truth) is almost nonexistent: the only open ROGII-style typelog is one synthetic Middle Woodford GR curve (GWC 2020), and the GWC 2021 release adds two ground-truth scenarios with ~10,000 expert interpretation snapshots. Beyond that, real typelogs must be taken from generic vertical-well repositories (FORCE 2020, UK NDR, KGS, NLOG), and real structural shapes from the Geology Forecast Challenge horizon traces, all of which need domain adaptation before they can feed our simulator. |

### Problem formulation

| Field | Value |
|---|---|
| Task definition | Not a method; a data-sourcing item. The goal is to find typelogs (GR vs TVT/TVD of an offset vertical well) and geometry (TVT/horizon vs MD/VS along laterals) outside the Kaggle set, to (a) diversify the 85% synthetic training wells, especially cases with repeated, self-similar layers, and (b) provide independent evaluation or calibration material. GWC 2020/2021 are ROGII StarSteer-style stratigraphy-based-steering tasks (match lateral GR to an offset typelog by adjusting the stratigraphic model); Geology Forecast Challenge asks for probabilistic forecasts of a layer-depth (horizon) curve ahead of the bit (10 realizations). |
| Target parameterisation | GWC/StarSteer answer files parameterise truth as horizon depths at MD points plus TVD shifts (a piecewise stratigraphic model); convertible to TVT along the trajectory, i.e. the same target as Kaggle TVT. Alyaev 2022 uses SVD(MD), which is exactly TVT up to sign/offset. Geology Forecast Challenge: HORIZON_Z_adjusted vs VS_APPROX_adjusted, i.e. a structural surface trace (equivalent to our z_layer = TVT + Z along the lateral). Vertical-well repositories provide only GR(TVD), i.e. typelog material, no TVT target. |
| Output representation | Not applicable (datasets). GWC 2021 is notable because it contains many human interpretations of the same well, an empirical distribution of plausible TVT paths per scenario; Geology Forecast Challenge scores 10 realizations per sample (probabilistic output). |

### Method

| Field | Value |
|---|---|
| Architecture | Not applicable (datasets). Relevant generators: Alyaev 2022 builds training triples (reference typelog randomly trimmed to a short section, an SVD curve realization, and the synthetic lateral log) and trained a mixture density network on 28 million samples; the stratigraphy realization script is public. NORCE's interactive-geosteering-benchmark (github.com/geosteering-no/interactive-geosteering-benchmark, BSD-4) contains a multi-layer toy geomodel with an EM (resistivity) tool, not GR. Geology Forecast Challenge 1st place: Conv1D + multi-head attention with positional encoding, 5-fold, trained on 314,360 windows cut by the organisers' interpolation script. |
| Core mechanism | The value of external data here is distributional: our synthetic wells are generated only from the 773 Kaggle typewells (consolidated into ~60 master systems by seq_NN_master_typewell.py), so every synthetic repeated-layer pattern is one that already exists in Kaggle masters. New typelogs add new GR textures and new self-similarity structures; new horizon traces add new structural (dip/curvature) shapes for the z_shift / trend bank; GWC 2021 interpretations give human ambiguity statistics (spread of expert TVT paths) on known-truth wells. |
| Training objective | Not applicable (datasets). Geology Forecast Challenge metric is equivalent to a weighted MSE over forecast positions (per 1st-place writeup), with 10 realizations; Alyaev 2022 used a multiple-trajectory-prediction (winner-takes-all style) MDN loss. |
| Decoding | Not applicable (datasets). |
| Metric consistent decode | Not applicable (datasets). |
| Evidence scope | GWC 2020/2021: single lateral vs single typelog per scenario (2 scenarios in 2021). Geology Forecast Challenge: single horizon trace windows. Repositories: single vertical wells (multi-well fields in FORCE/Volve/NDR allow building typelog families from nearby wells). |
| Propose vs select | Not applicable (datasets). GWC 2021 human interpretations could serve as a candidate set for studying how often experts choose the wrong repeated band. |

### Data

| Field | Value |
|---|---|
| Synthetic to real | GWC 2020 typelog is itself synthetic (built from Middle Woodford observations) and normalized to 0-1, so it must be quantile-mapped to Kaggle GR units (API) and noise-matched before use; Alyaev notes GWC used 6-7% log noise. Vertical-well repositories are real but from different basins and tools (North Sea clastics vs US unconventional carbonates/shales), so their GR textures and cycle thicknesses differ from Kaggle's; they need (a) GR normalization per well (quantiles), (b) resampling to 0.25 ft TVT levels, (c) optional thickness scaling to Kaggle cycle-length statistics. Our pipeline already has the machinery: master-typewell re-skinning (seq_NN_master_typewell.py) regenerates a simulated well's GR from a crop of another master, so external typelogs can enter as extra 'masters' in the re-skin bank. |

### Results

| Field | Value |
|---|---|
| Validation protocol | For any use in training: external typelogs must be kept out of the internal 155-well holdout evaluation and results must be compared with a Kaggle-only control at the same epochs and seeds; because our acceptance noise is +-0.39 ft for a single run (and a seed-level null delta of -0.03 ft at 60 epochs), 2+ seeds or 5-fold CV are needed. For GWC 2021, the known-truth answer must not leak into any tuning if used as an external test. |
| Inference budget | No inference cost: data only affects training. Preprocessing (LAS parsing, normalization, resampling) is CPU minutes. GWC 2021 zips 262 MB; FORCE 2020 ~ hundreds of MB; Volve full set is large (tens of GB+) but only LAS logs are needed. |

### Ambiguity handling

| Field | Value |
|---|---|
| Repeated layers | This is the motivation, but the evidence is mixed. Real vertical logs from cyclic successions (e.g. North Sea Jurassic/Cretaceous, Kansas Pennsylvanian cyclothems) contain many self-similar GR cycles, which would create hard band-confusion cases the simulator does not currently produce from Kaggle masters alone. Caveat: our band confusion is defined relative to Kaggle typewells; if external logs are more (or differently) repetitive, the network may learn an ambiguity prior that is miscalibrated for Kaggle wells, which under the expectation decode directly raises RMSE (calibration, not accuracy, is what matters for pooled RMSE). A cheaper and in-distribution alternative is to mine Kaggle masters themselves for self-similar windows and upweight those in the simulator. |
| Drift | Geology Forecast Challenge horizon traces are the relevant resource: they are real structural shapes along laterals (Z of a horizon vs VS), i.e. realistic dip/curvature histories, which could extend the real residual / trend bank used by z_shift (real_block residuals, TVT_bias_v2) and trajectory mixup; this targets the ~25% one-line-drift error mode by teaching the network more realistic slow dip changes. Typelogs do not help drift. |
| Uncertainty and calibration | GWC 2021 gives, for the same well and same data, the spread of 349 expert interpretations over time, an empirical measure of how ambiguous each MD segment is; it could be used to check whether our posterior spread (width, number of humps) is larger where experts disagree. Geology Forecast Challenge is explicitly probabilistic (10 realizations). No calibration analyses exist in these releases. |

### Relevance to erudition

| Field | Value |
|---|---|
| Where to use | (1) Synthetic generator: add external typelogs as extra master systems in the re-skin bank of experiments/bilzard/seq_NN_master_typewell.py (sim_cfg['reskin']), with low probability, after GR quantile normalization and 0.25 ft resampling. (2) Hard-case mining: compute a band-ambiguity score per typewell window (count and depth of secondary minima of the typewell self-cost /tw_gr(t) - tw_gr(t + s)/ over the +-50 ft level window) and oversample high-ambiguity crops, from Kaggle masters first, external logs second. (3) Drift augmentation: add Geology Forecast Challenge horizon traces (converted to z_layer increments per MD) to the z_shift real-residual bank. (4) External sanity test: convert GWC 2021 scenarios (if GR is included) into Kaggle format and score the model plus compare with the human-expert spread; tiny N, qualitative only. |
| Error mode targeted | Band confusion (~51%) via harder / more diverse repeated-layer synthetic cases; drift (~25%) via real structural traces (Geology Forecast Challenge); calibration of hump weights secondarily. |
| Gap vs our implementation | Today all synthetic wells are generated from Kaggle's own 773 typewells / ~60 master systems and Kaggle trajectories (Ruby's simulator + diff-block bootstrap, trajectory mixup, real_block residuals, re-skinning between Kaggle masters). No external typelog, horizon trace, or expert-interpretation data is used anywhere, and there is no explicit hard-case (self-similarity) sampling: synthetic crops are drawn without regard to how ambiguous the typewell window is. The closest experiment, rb_v2_synth (mixup + re-skin + real residuals, bundled), was worse than baseline at 60 epochs (+0.362 ft) and still +0.147 at 150 epochs but still descending and better on the bulk of wells (median 2.796 vs 2.929), with losses on a few catastrophic wells; the three sub-changes were never ablated. So the 'more diverse typelogs' direction is unresolved rather than refuted, and adding out-of-basin typelogs is a stronger version of the same intervention, with larger distribution shift. |
| Proposed experiment | E0 (CPU, no training, half a day): download the GWC 2020 gr.csv plus ~50-100 GR logs from FORCE 2020 (NLOD) and KGS; normalize each by quantiles to the Kaggle typewell GR distribution, resample to 0.25 ft. Define an ambiguity index per 100 ft window: number of secondary minima within 20% of the primary minimum of the windowed self-matching cost over shifts 5-50 ft. Compare the index distribution of Kaggle masters vs external logs and, crucially, correlate the index of each holdout well's typewell window with its per-well SSE and wrong_mode share from the existing posterior dump (results/baseline_check_posterior_all). Go/no-go: proceed only if high-ambiguity typewell windows explain a large share of wrong_mode error AND Kaggle's own masters contain few such windows (i.e. there is a coverage gap external data could fill). E1 (in-distribution control first): in experiments/bilzard/seq_NN_master_typewell.py add ambiguity-weighted crop sampling from Kaggle masters only (new sim_cfg key, e.g. 'hard_crop_weight_power'); 150 epochs, 2 seeds, vs baseline 150 epochs 2 seeds, MD-phase TTA on both. E2: add external typelogs as extra masters in the re-skin bank (new sim_cfg['reskin']['external_masters'] path list, apply_prob 0.1-0.2, so external logs are <=10% of synthetic wells); same protocol; success = pooled RMSE gain beyond run-to-run scatter across both seeds, with the tail wells (fb03ae90, f6d009f4, f2d4c8c9, e7818f7a, d7eb0be8) inspected. E3 (drift): convert Geology Forecast Challenge traces into per-MD z_layer increment series, append to the real_block residual bank with a mixing probability, 150 epochs 2 seeds. E4 (optional): if GWC 2021 zips include the lateral and typelog GR, build two Kaggle-format wells and compare model posterior spread with the expert-interpretation spread per MD. |
| Implementation effort | E0 low (LAS parsing with lasio, numpy). E1 low-medium (sampling weights inside the existing simulator). E2 medium (external master ingestion, GR normalization, making re-skin quantile placement work for masters without Kaggle drilling bands). E3 medium (coordinate conversion of VS/Z traces). E4 medium, blocked on verifying zip contents. |
| Gpu cost | E0 CPU only. E1-E3: each is a pair of 150-epoch ConvNeXt-small trainings per seed (roughly 5 h per run on the RTX 5060 8 GB at the documented ~110-120 s/epoch), so 2 seeds x (candidate + control) ~ 20 h per hypothesis; control runs can be shared across E1-E3. |
| Distillation compatibility | Fully compatible: data changes only affect the teacher's training (and optionally the student's KD data); the FastViT student distilled on output posteriors inherits any gain without architectural change. |
| Error correlation with ours | Low diversity: same architecture and decode, so errors stay correlated; the benefit, if any, is a better-calibrated single model rather than a new blend member. Models trained with and without external masters could be averaged as a mild ensemble, similar to the existing ConvNeXt ensemble. |
| Notes | Downloads were not performed in this research pass (per session rules); zip contents of GWC 2021 remain unverified. Alyaev & Elsheikh 2022 is also covered under A09 (multi-hypothesis MDN); their public stratigraphy-realization script is a simple, documented synthetic-curve generator that could be compared with Ruby's simulator. Geology Forecast Challenge is a ROGII/NORCE-organised sibling competition (2025) of the ROGII Kaggle competition and is the only other public dataset with real lateral-scale structural traces; it was co-authored by I. Kuvaev (ROGII), linking it to the Kaggle data source. CREStereo (see B01) is the stereo-matching precedent for this item: its authors attribute robustness on repetitive-texture regions largely to a synthetic dataset designed around hard cases, which supports hard-case mining (E1) over indiscriminate diversification. Overlaps: A01/A02 (GWC studies), A09 (Alyaev MDN), B11 (synthetic data and sim-to-real). |

### Other info

- **Id**: A19
- **Uncertain fields (omitted above)**:
  - code_availability
  - comparability
  - dataset
  - evidence_level
  - inputs
  - metric_and_performance
  - physical_constraints
  - prefix_conditioning
  - risk

---

## B01 Stereo matching cost-volume architectures (GC-Net, PSMNet, GA-Net/SGM-Net, AcfNet, RAFT-Stereo, IGEV/IGEV++, CREStereo)

### Basics

| Field | Value |
|---|---|
| Name | Stereo matching cost-volume architectures: GC-Net (soft-argmin), PSMNet (SPP + stacked hourglass), GwcNet (group-wise correlation), SGM-Net and GA-Net (learned semi-global aggregation), AcfNet (unimodal cost filtering), CFNet (cascade, uncertainty-adaptive range), ACVNet (attention concatenation volume), RAFT-Stereo, IGEV / IGEV++ (geometry encoding volumes, multi-range), CREStereo (cascaded recurrent, adaptive correlation, hard-case synthetic data) |
| Authors org | GC-Net: Kendall et al. (Skydio/Cambridge). PSMNet: J.-R. Chang, Y.-S. Chen (NCTU). GwcNet: X. Guo et al. (CUHK/SenseTime). SGM-Net: A. Seki, M. Pollefeys (Toshiba/ETH). GA-Net: F. Zhang, V. Prisacariu, R. Yang, P. Torr (Oxford/Baidu). AcfNet: Y. Zhang et al. (DeepMotion). CFNet: Z. Shen, Y. Dai, Z. Rao (NWPU). ACVNet: G. Xu et al. (HUST). RAFT-Stereo: L. Lipson, Z. Teed, J. Deng (Princeton). IGEV / IGEV++: G. Xu, X. Wang, Z. Zhang, J. Cheng, C. Liao, X. Yang (HUST). CREStereo: J. Li et al. (Megvii). |
| Year | 2017 (GC-Net, SGM-Net), 2018 (PSMNet), 2019 (GwcNet, GA-Net), 2020 (AcfNet), 2021 (CFNet, RAFT-Stereo), 2022 (ACVNet, CREStereo), 2023 (IGEV-Stereo), 2025 (IGEV++ in TPAMI) |
| Venue link | GC-Net ICCV 2017 arXiv 1703.04309; PSMNet CVPR 2018 arXiv 1803.08669; GwcNet CVPR 2019 arXiv 1903.04025; SGM-Net CVPR 2017; GA-Net CVPR 2019 arXiv 1904.06587; AcfNet AAAI 2020 arXiv 1909.03751; CFNet CVPR 2021 arXiv 2104.04314; ACVNet CVPR 2022 arXiv 2203.02146; RAFT-Stereo 3DV 2021 arXiv 2109.07547; IGEV-Stereo CVPR 2023 arXiv 2303.06615; IGEV++ TPAMI 2025 arXiv 2409.00638; CREStereo CVPR 2022 arXiv 2203.11483; survey: Tosi, Bartolomei, Poggi, 'A Survey on Deep Stereo Matching in the Twenties' arXiv 2407.07816 |
| Evidence level | Peer-reviewed (ICCV/CVPR/AAAI/3DV/TPAMI) with public benchmarks; IGEV++ and CREStereo abstracts verified this session; other details from the papers as widely documented. Transfer to our 1-D-lateral x level setting is our own reasoning, not evidence. |
| Extreme summary | Deep stereo builds a cost volume (per pixel, per candidate disparity), aggregates it with 3D convolutions or learned SGM-like recurrences, and reads out disparity by soft-argmin, with later work adding learned matching features, group-wise correlation, multi-range volumes and iterative GRU refinement. Our model already is a GC-Net-style design collapsed by one dimension (MD x level cost image, 2D U-Net aggregation, softmax expectation). The untried pieces that map most directly onto our error anatomy are learned semi-global aggregation along MD (GA-Net SGA), learned matching features/group-wise correlation instead of a single handcrafted /gr - tw_gr/ cost, PSMNet-style intermediate supervision, and CREStereo-style hard-case synthetic data; RAFT/IGEV-style iterative refinement is the family that already failed here. |

### Problem formulation

| Field | Value |
|---|---|
| Task definition | Stereo: predict per-pixel disparity d(x, y) of a rectified image pair, i.e. for each left-image pixel choose which of D candidate right-image positions matches. Ours (analogy documented in kaggle_1st_place/CLAUDE.md): for each MD column of the lateral, choose which of 400 typewell levels (0.25 ft over a 100 ft window) the lateral is at, i.e. TVT is disparity, the typewell is the right image scanline, and the lateral GR is the left image. |
| Inputs | Stereo: two RGB images; features from a shared (Siamese) CNN. Ours: 16-channel image of 345 MD columns x 400 levels built by hand: tw_gr, gr, gr_abs_diff (a single-scale absolute-difference cost), per-column lateral window statistics (gr_std, gr_slope, quadratic fit, first-last delta, isnan rate), tw_gr_is_nan, tw_tvt_rel, seen_tvt_rel, tw_seen_tvt_abs_diff, z_diff (optionally PF channels). |
| Target parameterisation | Stereo: disparity (continuous), regressed via expectation over discrete candidates; cascade methods (CFNet, CREStereo) re-parameterise as residual disparity within a narrowed range. Ours: TVT - TVT0 over the fixed 400-level window, supervised by regression plus an 'alignment' density target (exp_smooth, sigma 1.25 levels). |
| Output representation | Per-pixel softmax over disparity (GC-Net, PSMNet, GwcNet, GA-Net, AcfNet, CFNet stages, ACVNet) or per-pixel scalar disparity updated iteratively (RAFT-Stereo, IGEV, CREStereo). AcfNet additionally predicts a per-pixel confidence (variance) of the target distribution. Ours: per-column softmax over 400 levels, expectation decode. |

### Method

| Field | Value |
|---|---|
| Architecture | GC-Net: Siamese 2D CNN features at 1/2 res -> concatenation cost volume (C x D/2 x H/2 x W/2) -> 3D conv encoder-decoder -> soft-argmin. PSMNet: ResNet features + spatial pyramid pooling (global context) -> concatenation volume at 1/4 res -> three stacked 3D hourglasses with intermediate outputs, each supervised (weights 0.5/0.7/1.0) -> upsample + soft-argmin. GwcNet: split feature channels into G groups, correlation per group gives a G-channel volume (plus a reduced concatenation volume) -> cheaper and better-conditioned aggregation. SGM-Net: CNN predicts the SGM P1/P2 penalties per pixel/direction, classic SGM does the aggregation. GA-Net: semi-global guided aggregation (SGA) layer: for each direction r, C_A(p,d) = w0*C(p,d) + w1*C_A(p-r,d) + w2*C_A(p-r,d-1) + w3*C_A(p-r,d+1) + w4*max_i C_A(p-r,i), weights predicted per pixel by a guidance subnet and normalised, max over 4 directions; plus local guided aggregation (LGA) for thin structures; a few GA layers replace most 3D convs at far lower cost. AcfNet: standard PSMNet-like pipeline, but supervises the cost volume with a per-pixel unimodal (Laplacian) distribution centred on ground truth with learned variance from a confidence head. CFNet: fused multi-scale volumes at low res, then cascade stages whose disparity search range is set per pixel from the previous stage's variance (mean +- k*sigma). ACVNet: correlation volume used to build attention weights that filter the concatenation volume, suppressing redundant/ambiguous matches before aggregation. RAFT-Stereo: all-pairs 1-D correlation pyramid (no 3D conv), multi-level ConvGRU updates the disparity iteratively (sequence loss). IGEV: a lightweight 3D-regularised geometry encoding volume (GEV) + all-pairs correlation; soft-argmin from the GEV initialises the ConvGRU updates, cutting iterations. IGEV++: multi-range GEVs (coarse volumes for large disparities / ill-posed regions, fine ones for detail), adaptive patch matching (APM) to build large-range costs cheaply, selective geometry feature fusion (SGFF) to combine ranges per pixel. CREStereo: cascaded recurrent refinement over a feature pyramid, adaptive group correlation layer (local 2D-1D alternating search with learned deformable offsets and group-wise correlation), stacked cascade at inference for high resolution, trained on a new synthetic dataset built around hard cases. |
| Core mechanism | Three recurring ideas. (1) Matching is done in a learned feature space with receptive field, not on raw intensities, and the cost is expressed as correlation/concatenation of Siamese features (GC-Net, GwcNet, ACVNet). (2) Ambiguity (repeated texture, textureless or occluded regions) is resolved by propagating evidence across the image under a smoothness prior: 3D conv hourglasses with global context (PSMNet SPP), or explicitly DP-like learned aggregation with a jump penalty and an 'any disparity' reset term (SGM-Net, GA-Net). (3) Coarse-to-fine: a coarse global volume handles large displacements and ambiguous regions, fine volumes or iterative updates handle detail (CFNet, IGEV++, CREStereo, RAFT-Stereo). AcfNet adds a fourth: regularising the shape of the predicted distribution so soft-argmin is not a blend of modes. |
| Training objective | Smooth-L1 on soft-argmin disparity (GC-Net, PSMNet with multi-output weights, GwcNet, GA-Net); AcfNet adds stereo focal loss / cross-entropy to a unimodal target with learned variance plus confidence loss; CFNet multi-stage smooth-L1; RAFT-Stereo, IGEV, CREStereo exponentially weighted sequence loss over refinement iterations. Ours: regression (0.05/3) + alignment density (0.2/3, exp_smooth sigma 1.25) + GR_penalty (0.05/3); deep supervision implemented in the decoder but off (deep_sup=False). |
| Decoding | Soft-argmin / expectation over disparity (GC-Net family); iterative scalar updates (RAFT family); some works use top-k or local soft-argmin around the argmax to avoid mode blending (covered in B03). |
| Metric consistent decode | Soft-argmin = posterior mean, Bayes-optimal for squared error given a calibrated posterior, which matches our pooled RMSE; stereo's benchmarks (bad-pixel %, EPE) reward mode-seeking, so stereo papers that replace soft-argmin with argmax-local decodes optimise a different metric and may raise our RMSE. Cascade range narrowing (CFNet/IGEV++ fine stages, CREStereo) is implicitly mode-seeking when the posterior is multimodal. |
| Evidence scope | Whole image pair (global context via SPP, hourglasses, or directional recurrences spanning the full width). For us the equivalent is the whole lateral suffix plus prefix in one pass, which our U-Net already sees (bottleneck 22 MD x 13 levels). |
| Prefix conditioning | Not applicable in stereo. In ours the known prefix enters as channels (seen_tvt_rel, tw_seen_tvt_abs_diff); a GA-Net-style directional aggregation run from heel to toe would naturally propagate the known prefix's band identity along MD. |
| Propose vs select | Single output per pixel; CFNet/IGEV++ can be read as propose (coarse range) then select (fine), and multi-range volumes could provide band hypotheses to a reranker. Stereo nets are ensemble-compatible only as additional members with shared inputs. |
| Physical constraints | Smoothness with discontinuity tolerance (SGM penalties P1 for +-1 disparity, P2 for jumps; GA-Net's learned w2/w3 and w4 terms); uniqueness/occlusion handling is not modelled in most. Maps to ours as: TVT changes slowly along MD (small /dTVT/dMD/ per column), with rare faults (jumps). |

### Results

| Field | Value |
|---|---|
| Comparability | Not comparable numerically (image benchmarks, bad-pixel/EPE metrics). Architectural comparability is high: our 2D cost image is GC-Net's 3D volume with the image-height axis removed, so 3D-aggregation ideas transfer as 2D ops and 2D-recurrence ideas (SGA along image rows) transfer as 1D recurrences along MD. |
| Validation protocol | Stereo: fixed public benchmarks with hidden test sets. For our repo: 155-well holdout, single-run noise +-0.39 ft, seed-level null -0.03 ft at 60 epochs; 60-epoch comparisons are confounded by undertraining (baseline 5.16 -> 4.98 at 150), so architecture candidates need 150 epochs and 2 seeds. |

### Ambiguity handling

| Field | Value |
|---|---|
| Repeated layers | In stereo, repeated texture is the canonical ambiguity. Remedies with evidence: (a) wider context from SPP/hourglasses (PSMNet), which our U-Net already has; (b) learned semi-global aggregation (SGM-Net, GA-Net) that carries a match hypothesis along a scanline with learned continuity and jump penalties, so a locally ambiguous region inherits the band chosen where evidence is unambiguous; (c) attention filtering of the volume using correlation confidence (ACVNet); (d) hard-case synthetic training data (CREStereo). For us, (b) is the in-network, end-to-end trained version of the post-hoc forward-backward continuity decode that failed; unlike the post-hoc decode it can learn when to trust continuity and its outputs are trained against the RMSE-consistent loss. Caveat: our posterior dump shows much of the error is choosing/blending between bands that are all locally consistent over long MD stretches, and continuity cannot break a tie that persists along the whole lateral. |
| Drift | Coarse/multi-range volumes (IGEV++, CFNet) target large displacements, relevant if part of the drift error comes from truth leaving the +-50 ft window or sitting near its edge (clipping); fine-resolution detail is lost in our stem (level axis stride 4 at the stem, 400 -> 100), analogous to building the volume at 1/4 resolution. A slow one-line drift is a smooth bias, not an ambiguity, so matching architectures help only if the evidence to correct it (fine GR texture alignment) is being discarded early. |
| Uncertainty and calibration | AcfNet's confidence head and CFNet's variance-based ranges are the stereo uncertainty mechanisms; neither is calibrated in the ECE sense. For our RMSE, calibration of band weights is what matters (B09); none of these architectures addresses it directly. |

### Relevance to erudition

| Field | Value |
|---|---|
| Where to use | (1) Aggregation: insert GA-Net-style SGA layers along MD (heel-to-toe and toe-to-heel) into the decoder at the 1/2-MD, 1/4-level scale (173 x 100) or the next scale (87 x 50), zero-gated residual, guidance weights predicted from decoder features. (2) Cost construction: add learned matching-feature channels before the U-Net: a 1-D CNN over typewell GR along the level axis (full 0.25 ft resolution) and a 1-D CNN over the raw 32 lateral rows of each column (GR vs relative z within the column), combined by group-wise correlation (GwcNet) into G extra channels, replacing or supplementing the single handcrafted gr_abs_diff. (3) Supervision: turn on the already-implemented deep_sup (PSMNet-style intermediate outputs). (4) Range: diagnose whether truth ever leaves the 100 ft window; if yes, an IGEV++-style coarse long-range volume (e.g. 300 ft at 1 ft) as extra channels or branch. (5) Data: CREStereo's hard-case synthetic design (see A19/B11). |
| Error mode targeted | Band confusion (~51%) via SGA propagation and learned matching features; broad/skewed humps (~13%) via finer-resolution learned costs; drift (~25%) only via the range/resolution diagnostics. |
| Gap vs our implementation | What we already have: cost image + 2D encoder-decoder aggregation + softmax expectation (GC-Net/PSMNet skeleton); global context through a ConvNeXt-small U-Net; concatenation-volume-like inputs (tw_gr broadcast along MD and gr broadcast along levels). What we tried and failed at 60 epochs: AcfNet unimodal target (replacing alignment, +0.162), RAFT refinement (+0.078, unstable), MD-axial attention (-0.024, gates barely opened), large-kernel convs (never trained due to a zero-init deadlock; still untested). What we do not do: (a) no learned matching features; the only cost is pointwise /gr - tw_gr/ at a single scale (the +-0.25 ft shifted variants tw_gr_m0p25/p0p25 and signed/min/max diffs exist but are commented out), and the lateral descriptors are handcrafted window statistics; (b) no DP-like learned aggregation along MD (SGM-Net/GA-Net): long-range propagation relies on the U-Net's downsampled bottleneck and failed post-hoc DP; (c) no intermediate supervision (deep_sup implemented but False); (d) level axis is downsampled 4x at the stem, so aggregation never runs at 0.25 ft level resolution, unlike stereo nets that keep 1/4 but build the cost from learned features; (e) fixed single 100 ft range with no coarse long-range volume; (f) no hard-case synthetic design. |
| Proposed experiment | E0 (CPU diagnostics, no training): from results/baseline_check_posterior_all compute (i) fraction of holdout rows whose true TVT - TVT0 lies outside or within 2 ft of the +-50 ft window edge and their share of squared error (decides the multi-range lead); (ii) for wrong_mode columns, the MD run length over which the wrong band stays locally consistent with GR (if wrong bands persist over most of the lateral, SGA cannot help; if they are islands between confidently-correct stretches, SGA can). E1 (SGA along MD, main candidate): new module SGAlongMD in experiments/arch/seq_NN_arch_blocks.py: input decoder feature map F (B,C,H=MD,W=level), a 1x1 conv to K=16 'cost' channels, a guidance head predicting 5 weights per (MD, channel group) normalised by softmax, recurrence over MD in both directions (torch loop over H, or cumulative formulation), output = F + gamma * conv1x1(max(fwd, bwd)), gamma initialised to 0 with the inner convs at normal init (avoid the arch_v4 zero-times-zero deadlock). Register in model_modules() and model_named_parameters() (trap documented in CLAUDE.md). Place at decoder scale 173 x 100. cfg arch_v5_sga; 150 epochs, 2 seeds, vs 0801_V2_ep150 plus a second baseline seed; accept only if both seeds improve and the mean gain exceeds the seed scatter; log gamma to confirm it opened. E2 (learned cost): channels from a Siamese-style pair of small 1-D CNNs (typewell along levels at 0.25 ft; lateral per column over its 32 raw rows with relative z), group-wise correlation G=8 -> +8 input channels; trained end-to-end (cfg arch_v6_gwc); same protocol. E3 (cheap): flip deep_sup=True with deep_sup_stage_decay 0.5 on the baseline, 150 epochs 2 seeds. E4 (only if E0(i) shows window-edge error): coarse 300 ft x 1 ft range channels resampled onto the 400-level grid or a parallel coarse branch. |
| Implementation effort | E0 low (numpy on existing dumps). E1 medium (~150 lines; a Python loop over 173 MD steps is simple but slow in training, a scan/cumulative-max CUDA-friendly form or torch.compile helps; must respect gating/registration traps). E2 medium (dataset must expose raw 32-row lateral windows per column to the model, currently only summary stats are passed). E3 trivial (config flag). E4 medium. |
| Distillation compatibility | KD on output posteriors works regardless of teacher architecture. For the FastViT student, SGA can be added to the student's decoder too (cheap, but a sequential op on CPU) or left teacher-only so the student learns the propagated posterior. Learned cost channels (E2) must be computed for the student as well; they are small 1-D CNNs and fit the CPU budget. |
| Error correlation with ours | Moderate-to-high correlation: same inputs and decode. SGA changes the inductive bias (explicit continuity along MD) and may produce different errors on wells where the U-Net blends bands, so an SGA model may be a useful member of the existing ConvNeXt average; learned per-point gates have failed before, so plain averaging is preferred. |
| Risk | (1) Our model already sees the whole lateral in one pass and the posterior-analysis summary says band-confusion error falls only with more evidence or better calibration; SGA and learned costs add no new evidence, only better use of existing evidence, so gains may be small relative to the +-0.39 ft single-run noise. (2) Post-hoc continuity decoding already failed, suggesting wrong bands are often continuous along MD, which no continuity prior can fix (E0(ii) tests this). (3) Recurrent/iterative modules were unstable here (RAFT peaked at epoch 40) and residual branches grafted onto the pretrained ConvNeXt barely opened in 60 epochs (axial gates ~0.01); 150 epochs and a verified-live gate are required. (4) Mode-seeking parts of stereo (cascade range narrowing, argmax-local decodes, unimodal targets) conflict with the RMSE-optimal expectation decode; AcfNet-style unimodality already hurt (+0.162). (5) Benchmarks are images with bad-pixel metrics; no evidence that stereo gains transfer to 1-D well-log alignment. (6) Holdout resolution: the pooled metric is decided by ~10 wells, so E1-E3 need seeds or 5-fold CV to conclude. |
| Notes | Mapping table (ours -> stereo): MD column = image column; TVT level = disparity; typewell GR = right scanline; lateral GR = left scanline; gr_abs_diff = AD cost volume (SAD without window); ConvNeXt U-Net = 3D cost aggregation (2D here because the image height axis does not exist); softmax expectation = GC-Net soft-argmin; alignment loss = AcfNet-like fixed-width target; PF channels = external prior as in DEFOM-Stereo/MonSter (B02). A key difference from stereo: laterals revisit TVT non-monotonically and there is no epipolar uniqueness, so occlusion reasoning does not transfer. The survey by Tosi et al. (2024) is the best entry point for newer variants (Selective-Stereo CVPR 2024 with multi-frequency GRUs; Stereo Anywhere; foundation-prior models in B02). Multimodal decoding and losses (ADL, SMD-Nets, top-k soft-argmax) are split into B03; learned DP losses into B05. Overlaps: A19 (CREStereo hard-case synthetic data), B02, B03, B05. |

### Other info

- **Id**: B01
- **Uncertain fields (omitted above)**:
  - code_availability
  - dataset
  - gpu_cost
  - inference_budget
  - metric_and_performance
  - synthetic_to_real

---

## B02 Foundation-prior stereo (FoundationStereo, Fast-FoundationStereo, DEFOM-Stereo, MonSter)

### Basics

| Field | Value |
|---|---|
| Name | Foundation-prior stereo: FoundationStereo, Fast-FoundationStereo, DEFOM-Stereo, MonSter (and MonSter++) |
| Authors org | FoundationStereo: Bowen Wen, Matthew Trepte, Joseph Aribido, Jan Kautz, Orazio Gallo, Stan Birchfield (NVIDIA). Fast-FoundationStereo: Bowen Wen, Shaurya Dewan, Stan Birchfield (NVIDIA). DEFOM-Stereo: Hualie Jiang, Zhiqiang Lou, Laiyan Ding, Rui Xu, Minglang Tan, Wenjie Jiang, Rui Huang (CUHK-Shenzhen / Insta360). MonSter: Junda Cheng et al. (Huazhong Univ. of Science and Technology and collaborators). |
| Year | 2025 (FoundationStereo, DEFOM-Stereo, MonSter: CVPR 2025); Fast-FoundationStereo arXiv Dec 2025; MonSter++ arXiv 2025 |
| Venue link | FoundationStereo CVPR 2025 oral, best-paper nomination: https://arxiv.org/abs/2501.09898 , https://nvlabs.github.io/FoundationStereo/ ; Fast-FoundationStereo: https://arxiv.org/abs/2512.11130 , https://nvlabs.github.io/Fast-FoundationStereo/ ; DEFOM-Stereo CVPR 2025: https://arxiv.org/abs/2501.09466 ; MonSter CVPR 2025 highlight: https://openaccess.thecvf.com/content/CVPR2025/html/Cheng_MonSter_Marry_Monodepth_to_Stereo_Unleashes_Power_CVPR_2025_paper.html ; MonSter++: https://arxiv.org/abs/2501.08643 |
| Evidence level | Peer-reviewed (CVPR 2025) for FoundationStereo, DEFOM-Stereo and MonSter; arXiv preprint for Fast-FoundationStereo and MonSter++. Details read from the arXiv HTML versions. The CVF PDFs for DEFOM and MonSter returned 403, so some MonSter ablation numbers come from the MonSter++ extension and are marked. |
| Extreme summary | Four 2025 stereo models add a strong monocular prior (Depth Anything V2) next to classical cost-volume matching, never in place of it. They inject it as extra features (FoundationStereo STA, DEFOM), as the starting disparity (DEFOM), or through a confidence-gated mutual refinement that first fits a global scale and shift of the prior to confident matches (MonSter). Fast-FoundationStereo then compresses the result about 10x with feature-level distillation, block-wise NAS and structured pruning. |

### Problem formulation

| Field | Value |
|---|---|
| Task definition | Dense disparity (inverse depth) for every pixel of a rectified stereo pair, zero-shot across domains. Mapping to Erudition: disparity maps to TVT level, pixel maps to MD column, and the left/right images map to lateral GR and typewell GR. The monocular prior maps to any estimate of TVT that does not come from GR matching. |
| Inputs | Left and right RGB images. The frozen Depth Anything V2 (DINOv2 ViT) monocular depth network sees the left image. In our analogue the independent prior sources would be the particle-filter track (Ruby's 7.35 ft PF), the 2nd-place AnchorCNN+DP path, the XY geo prior / neighbour wells, and a trajectory-plus-constant-dip extrapolation from the known heel prefix. |
| Target parameterisation | Disparity in pixels. The monocular prior is relative inverse depth up to an unknown scale and shift. DEFOM normalises it (d0 = eta*w*z/max(z)) and then learns per-pixel multiplicative scale updates. MonSter fits one global scale and shift by least squares, then learns a per-pixel residual shift. The TVT analogue of the scale/shift ambiguity is a per-well offset plus a per-well slope along MD. That is exactly the 'one straight-line drift per well' error mode. |
| Output representation | Point disparity map. FoundationStereo builds a cost volume, gets an initial disparity by soft-argmin, then refines it with ConvGRU iterations. DEFOM and MonSter are RAFT/IGEV-style iterative refiners. Uncertainty is not a first-class output, though MonSter computes an internal confidence. |

### Method

| Field | Value |
|---|---|
| Architecture | FoundationStereo: a side-tuning adapter (STA). A frozen DepthAnythingV2 feature is taken before its output head, reduced with a 4x4 stride-4 conv and concatenated with a same-scale CNN feature at 1/4 resolution. The adapter beat both direct ViT pyramid use and bidirectional feature exchange. Next comes a cost volume. Attentive Hybrid Cost Filtering (AHCF) follows: a 3D hourglass built from Axial-Planar Convolutions, which split a 3x3x3 conv into a spatial Ks x Ks x 1 conv and a disparity-only 1x1xKd conv, plus a Disparity Transformer that runs self-attention along the disparity axis of the cost volume. Last are soft-argmin init and coarse-to-fine ConvGRU refinement, about 0.7 s per 375x1242 image on an A100. DEFOM-Stereo builds on RAFT-Stereo. Its matching and context encoders fuse CNN features with Depth Anything V2 features through a trainable DPT head. The recurrent disparity starts from the normalised monocular depth, and a scale update module (a ConvGRU that predicts dense scale maps, d_n = s * d_{n-1}) runs a 'scale lookup' that samples correlation at a set of scale factors, giving global rather than local search. It costs +67% parameters and +42% runtime over RAFT-Stereo with ViT-L. MonSter has two branches. The mono branch is DepthAnythingV2; the stereo branch is IGEV-like. After N1 (about 2) stereo iterations, mono inverse depth is aligned to stereo disparity by a global least-squares scale and shift. Then N2 (about 2) rounds of mutual refinement follow. SGA (Stereo-Guided Alignment) uses confident stereo cues to estimate a per-pixel residual shift for the mono prior. MGR (Mono-Guided Refinement) uses the aligned mono disparity to fix stereo in ill-posed regions. Fast-FoundationStereo uses a single efficient student backbone in place of the hybrid ViT+CNN. Its cost-filtering blocks come from block-wise NAS solved with integer linear programming under a latency budget, and its ConvGRU refiner is pruned structurally along recurrent dependency graphs with first-order Taylor importance. It runs in 49 ms, or 21 ms with TensorRT, against 496 ms for the teacher. |
| Core mechanism | (1) The prior is a second, independent source of evidence that is strongest exactly where matching is ill-posed: occlusions, textureless and repetitive regions. Repeated rock layers are our version of a repetitive texture. (2) The prior is added, not substituted. The matching path still owns fine detail, and the prior is fused through features, initialisation or confidence-weighted refinement. (3) Scale ambiguity of the prior is resolved against the matcher. MonSter fits a global scale/shift to confident stereo pixels, then learns only local residuals. DEFOM learns multiplicative scale corrections with a global scale lookup. (4) The confidence gate means the prior only overrides matching where matching is unreliable. MonSter's confidence comes from the feature-warping residual /F_L(x) - F_R(x - D)/. (5) Global reasoning along the disparity axis, via FoundationStereo's Disparity Transformer, lets the network compare distant candidate disparities directly. This is the stereo analogue of comparing two repeated bands tens of levels apart, which local convolutions cannot see in one receptive field. (6) Fast-FoundationStereo shows that feature-level distillation, not only output supervision, carries much of the teacher's generalisation into a small student. |
| Training objective | FoundationStereo: smooth-L1 on the initial soft-argmin disparity plus exponentially weighted L1 over the GRU iterations, trained on its 1M-pair synthetic FSD dataset. An iterative self-curation loop drops and regenerates samples on which the model's BP-2 exceeds 60% (ambiguous or unlearnable). DEFOM and MonSter: RAFT/IGEV-style sequence L1 losses with gamma = 0.9 weighting. Fast-FoundationStereo: MSE between student and frozen-teacher backbone pyramid features, with each NAS block trained by distillation, then output-level training on synthetic data plus 1.4M pseudo-labelled internet stereo pairs. Pseudo-labels are filtered by agreement with monocular depth, normal-map consistency and sky masking. |
| Decoding | FoundationStereo initialises with soft-argmin (the expectation over disparity) and then refines by regression. DEFOM and MonSter output the last refinement iterate as a point estimate. None of them decode a distribution at the end. |
| Metric consistent decode | Partly. The soft-argmin init is the posterior mean (L2-consistent), but the final GRU regression under L1 loss is pulled toward a median-like, L1-optimal estimate. For our pooled RMSE, only the prior-fusion ideas transfer. The decode should stay the expectation over our per-column softmax. |
| Evidence scope | Whole image jointly. The monocular prior is global per image, and the cost volume is local per pixel plus aggregation. In our analogue the prior (PF / DP / geo) spans the whole lateral. |
| Prefix conditioning | Not applicable in stereo. In our analogue the known heel prefix is the natural anchor for MonSter's global scale/shift fit: fit the prior's offset and slope to the prefix labels and to the high-confidence U-Net columns, then pass only residuals to the network. This overlaps with item B10. |
| Propose vs select | Fusion, not selection. The prior enters as features or initialisation, and a learned confidence decides where it dominates. It is not a candidate-plus-gate design. That matters here because our learned per-point gates between ConvNeXt and AnchorCNN already failed, while warm-started feature-level fusion has not been tried. |
| Physical constraints | Monocular depth carries object-level geometric plausibility. MonSter's global scale/shift is a two-parameter structural constraint, and its per-pixel residual is learned. The TVT analogue is a structural constraint of offset plus slope (a constant dip) per well, with learned local residuals. |

### Data

| Field | Value |
|---|---|
| Dataset | FoundationStereo: FSD, 1M synthetic stereo pairs rendered in NVIDIA Omniverse, plus public synthetic sets. Fast-FoundationStereo adds 1.4M pseudo-labelled in-the-wild pairs. DEFOM and MonSter: SceneFlow and mixed datasets, evaluated zero-shot on KITTI 2012/2015, Middlebury and ETH3D. |
| Synthetic to real | This is the headline claim of all four papers: training on synthetic data only, then zero-shot to real. FoundationStereo credits the frozen foundation features (STA) with closing the sim-to-real gap, and self-curation for removing ambiguous synthetic samples. Fast-FoundationStereo: pseudo-labelled real data moved RT-IGEV from 11.52% to 8.69% BP-2 on Middlebury-H. Relevance: Erudition trains on 85% synthetic wells. The pseudo-label lesson needs a large pool of unlabelled real wells, which we lack (data/test has 3 wells). The CLAUDE.md correction of 2026-09-21 already records this. |

### Results

| Field | Value |
|---|---|
| Comparability | Not comparable numerically. These are image stereo benchmarks with bad-pixel or EPE metrics. Only the architectural and training recipes transfer to our pooled RMSE over 155 holdout wells. |
| Validation protocol | Standard stereo benchmarks with official held-out test servers (KITTI, Middlebury, ETH3D) and zero-shot cross-domain evaluation. There is no analogue of our spatial-leakage or test-picked-checkpoint issues. |
| Inference budget | FoundationStereo about 0.5-0.7 s per image on an A100. DEFOM +42% over RAFT-Stereo. Fast-FoundationStereo 21-49 ms. For us the relevant cost is only the extra prior channels or a small second stage: a few ms per well on the RTX 5060. The PF or AnchorCNN prior costs whatever it already costs. AnchorCNN+DP made the blend 5-6x slower in the earlier measurement, so a PF or geo prior is the budget-friendly source. |

### Ambiguity handling

| Field | Value |
|---|---|
| Repeated layers | In stereo, repetitive textures are a named failure case that the monocular prior resolves, because the prior does not depend on matching. The analogue for band confusion (about 51% of our squared error) is an independent absolute-position prior that is correct to within half a band spacing. When the prior is itself ambiguous, as the XY geo-prior hump-picking experiment found, fusion cannot help. The Disparity Transformer (attention along the disparity/level axis) is a second, prior-free mechanism: it lets the network compare repeated candidate bands globally. |
| Drift | MonSter's global scale-and-shift alignment is structurally a per-image two-parameter fit. The TVT equivalent is a per-well offset plus slope, which matches the drift mode directly. Our posterior dump (results/baseline_check_posterior_all/posterior_report.json, 60-epoch baseline) shows that an oracle per-well straight-line removal would cut pooled RMSE from 5.47 to 3.30 ft, and a per-well offset removal gives 4.05 ft. The drift component is therefore large, and a confident anchor for that line could remove much of it. |
| Uncertainty and calibration | MonSter uses a confidence estimate (the warping residual) to gate fusion. FoundationStereo and DEFOM do not output calibrated uncertainty. None report calibration. For us the useful point is the confidence gate: the prior should affect only columns where the U-Net posterior is broad or multimodal. |

### Relevance to erudition

| Field | Value |
|---|---|
| Where to use | (a) As an input-side fusion. Feed an independent TVT prior as extra channels to the U-Net, following the DEFOM/FoundationStereo 'add alongside' pattern, warm-started from a trained checkpoint so the starved-training failure of rb_v1_neighbor does not repeat. (b) As a MonSter-lite second stage. Align the prior to the stage-1 posterior with a confidence-weighted per-well offset+slope fit, then have a small refiner take [stage-1 posterior, aligned prior, confidence] and output a corrected per-column softmax. (c) As an architectural block. Add a Disparity-Transformer-style attention along the level axis (not MD) inside the U-Net. (d) In the distillation step. Add feature-level MSE distillation from ConvNeXt-small to FastViT, per Fast-FoundationStereo, beyond the current output KL. |
| Error mode targeted | Band confusion (about 51%) via an independent prior and level-axis attention; drift (about 25%) via confidence-weighted offset+slope alignment; distillation quality for the FastViT student. |
| Gap vs our implementation | What we already do: an independent prior as input exists in principle. rb_v1_neighbor added 9 geo-prior channels (16 to 24 inputs) but was starved (3.1x behind the baseline at 60 epochs) and never tested warm-started. The pf_v1 particle-filter channel is only planned. Output-level KD (KL at T=2, single or ensemble teacher) is built and works: FastViT-SA12 taught by ConvNeXt-small is the best single model at 4.84 ft. DINOv2 as a backbone replacement was correctly rejected (25x29 single-scale maps break the 2:1 resolution rule). What we do not do: (1) No confidence-gated fusion. Every prior use so far has been either a raw input channel trained from scratch, a post-hoc hump pick (geo pick) or a learned hard gate between models. None used MonSter's two-step 'fit the prior's global offset+slope on confident columns, then learn a residual' design. (2) No initialisation from a prior. The U-Net is one-shot, and the RAFT refiner (arch_v3) started from the network's own expectation, not from an independent prior. (3) No attention along the level axis. arch_v2_axial attended along MD. FoundationStereo's Disparity Transformer attends along disparity, the axis on which the repeated bands lie. At ConvNeXt stage 1 the level axis has 100 positions and at stage 2 it has 50, so this is cheap. (4) No axial-planar split of spatial and level kernels. arch_v4_lkconv was dead (zero-init deadlock) and never measured, so the anisotropic kernel idea is still untested. (5) Distillation is output-only. Fast-FoundationStereo's ablation credits backbone feature MSE with a 23% relative BP-2 gain (2.87% to 2.20%). (6) No self-curation of synthetic data. FoundationStereo regenerates synthetic samples the model finds unlearnable. We could audit which z_shift simulations produce truly unresolvable repeated-band cases. Whether to drop them (FoundationStereo) or keep them (calibration) is open. Keeping them is probably right under a mean decode. |
| Proposed experiment | E1, a MonSter-lite prior fusion that is warm-started and cheap. In experiments/arch (fork of 0801_V2): (i) produce a per-column independent prior from the AnchorCNN+DP holdout path if it is cached, otherwise from the PF; (ii) compute a confidence-weighted per-well offset+slope alignment of that prior to the stage-1 expectation, weighting columns by 1/posterior-variance restricted to unimodal columns and including the heel prefix where labels are known; (iii) render the aligned prior as a Gaussian 'prior plane' over the 400 levels with width equal to its residual std, plus a scalar confidence plane, giving 2 extra input channels; (iv) extend the U-Net's input projection (the to_backbone 1x1 conv) with zero-initialised weights for the new channels so the model is bit-identical at init, then fine-tune from results/cnx_tiny_ep150 for 30-40 epochs. Touch seq_NN_dataset.py (new channels), seq_NN_models.py (input width and zero-init; register in model_named_parameters), seq_NN_cfg.py (recipe arch_v5_prior_fuse). Control: the identical fine-tune schedule from the same checkpoint with the new channels filled with zeros. This removes the extra-epochs confound; the noise floor is ±0.39 ft per single run, so use seq_NN_robust_compare.py and two seeds per arm, and report the SSE change split by posterior_analysis.py categories (wrong_mode, blend, unimodal_offset). E2, level-axis attention: add a Disparity-Transformer-style block (gated residual, zero-init gamma, like AxialMDStack but transposed to attend over W=level with MD positions folded into the batch) after ConvNeXt stages 1-2 (level 100/50 positions). Recipe arch_v6_level_attn, 150 epochs; the control is cnx_tiny_ep150. Verify identity at init and live gradients on both gate and weights, given the lkconv lesson. E3, feature-level KD: in seq_NN_train.py add an MSE loss between 1x1-projected FastViT decoder features and frozen ConvNeXt-small decoder features at 2-3 U-Net scales (weight about 0.1 of the KL term), alongside the existing distill_alignment_loss. Recipe cnx_fastvit_kd_small_feat; the control is cnx_fastvit_kd_small_ep150. |
| Implementation effort | E1 medium: a new prior pipeline plus dataset channels, but the zero-init warm-start pattern already exists in experiments/arch. E2 low-medium: it mirrors AxialMDStack with the axes swapped. E3 low: a forward hook on decoder features plus 1x1 projection heads. |
| Gpu cost | E1: a 30-40 epoch fine-tune of tiny at about 65-110 s per epoch is about 1 h per arm, so about 4 h for 2 seeds x 2 arms, plus prior generation (PF on CPU). E2: a full 150-epoch run (about 5 h) or a 40-epoch warm start (about 1.2 h). Attention over 100 level positions x 173 MD positions x batch fits in 8 GB. E3: about 2 h 40 min per 150-epoch FastViT run plus teacher forward passes (already paid by the current KD). |
| Distillation compatibility | Good. E1's prior channels and E2's block are part of the student's inputs and architecture, so a FastViT student can include them and the output-posterior KD stays unchanged. E3 is itself a distillation upgrade. Feature KD needs 1x1 adapters because FastViT and ConvNeXt channel widths differ. |
| Error correlation with ours | An independent prior (PF or DP) has different errors from GR-matching U-Nets. AnchorCNN blends showed that diversity exists, but a learned gate could not exploit it. Fusion at the feature level inside the network is the stereo field's answer to exactly this problem. A level-attention variant of the same U-Net would correlate highly (about 0.85, like the other ConvNeXt variants), so as an ensemble member it adds only seed-style diversity. |
| Risk | (1) The prior may not be informative where it matters. The XY geo prior failed at hump picking, and in stereo the monocular prior is strong because depth networks are trained on millions of images, while our PF/DP priors are much weaker (PF 7.35 ft against U-Net 4.8-5.2 ft). (2) Data-limited regime: extra channels trained from scratch starved rb_v1_neighbor. The warm-start plus zero-init plan addresses this, but the fine-tune may learn to ignore the channels. (3) Leakage: the AnchorCNN/PF prior for holdout wells must be produced without holdout labels and without holdout-picked checkpoints. The known caveat already inflates numbers by about 0.15 ft. (4) Level attention might repeat arch_v2's pattern, where gammas barely move in 60 epochs. Use 150 epochs or a warm start. (5) Effects below ±0.39 ft are unmeasurable on a single run. The accidental null draw produced k*=21 and 64% bootstrap with no model change. |
| Notes | The CLAUDE.md survey (2026-09-21) already drew the right high-level lesson ('add alongside, never replace') and rejected DINOv2. This item adds the concrete mechanisms that were not yet mapped. First, MonSter's global scale/shift followed by a learned residual maps onto a per-well offset+slope, and the posterior dump shows a per-well line-removal oracle of 3.30 ft against 5.47. Second, FoundationStereo's Disparity Transformer is attention along the level axis, which differs from the MD-axial attention already tested. Third, Fast-FoundationStereo's clearest distillation lever is feature-level MSE, not output KL alone. The pseudo-labelling lesson does not apply (no unlabelled real wells), and self-curation is ambiguous under a mean decode. Ambiguous synthetic samples teach correct uncertainty, so do not drop them blindly. |

### Other info

- **Id**: B02
- **Uncertain fields (omitted above)**:
  - code_availability
  - metric_and_performance

---

## B03 Multimodal stereo decoding and losses (ADL + dominant-modal estimator, SMD-Nets, Stereo Risk, top-k soft-argmax)

### Basics

| Field | Value |
|---|---|
| Name | Multimodal stereo decoding and losses: ADL + dominant-modal estimator (DME), SMD-Nets, Stereo Risk (L1 risk), top-k soft-argmax (CoEx), single-modal weighted average (SMWA, Chen 2019) |
| Authors org | ADL/DME: Peng Xu, Zhiyu Xiang, Chengyu Qiao, Jingyun Fu, Tianyu Pu (Zhejiang University). SMD-Nets: Fabio Tosi, Yiyi Liao, Carolin Schmitt, Andreas Geiger (Univ. Bologna / MPI-IS / Tuebingen). Stereo Risk: Ce Liu, Suryansh Kumar, Shuhang Gu, Radu Timofte, Yao Yao, Luc Van Gool (ETH Zurich and others). CoEx top-k: Antyanta Bangunharcana et al. (KAIST). SMWA: Chuangrong Chen, Xiaozhi Chen, Hui Cheng (Sun Yat-sen Univ. / DJI). |
| Year | SMWA 2019 (ICCV); SMD-Nets 2021 (CVPR); CoEx 2021 (IROS); ADL 2024 (CVPR; arXiv 2023); Stereo Risk 2024 (ICML) |
| Venue link | ADL: https://arxiv.org/abs/2306.15612 , CVPR 2024 https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_Adaptive_Multi-Modal_Cross-Entropy_Loss_for_Stereo_Matching_CVPR_2024_paper.pdf ; SMD-Nets: https://openaccess.thecvf.com/content/CVPR2021/papers/Tosi_SMD-Nets_Stereo_Mixture_Density_Networks_CVPR_2021_paper.pdf , https://autonomousvision.github.io/smdnets/ ; Stereo Risk: https://arxiv.org/abs/2407.03152 , PMLR 235:31326-31343 https://proceedings.mlr.press/v235/liu24af.html ; CoEx: https://arxiv.org/abs/2108.05773 , https://github.com/antabangun/coex ; SMWA: https://openaccess.thecvf.com/content_ICCV_2019/html/Chen_On_the_Over-Smoothing_Problem_of_CNN_Based_Disparity_Estimation_ICCV_2019_paper.html |
| Evidence level | Peer-reviewed (ICCV/CVPR/ICML/IROS), plus an in-house measurement: every decode family in this item was evaluated on the saved ConvNeXt posteriors for all 155 holdout wells (results/baseline_check_posterior_all/posterior_all.npz, 23,634 posterior columns, 60-epoch 0801_V2 baseline) with scratchpad scripts decode_probe.py and decode_probe2.py. The in-house numbers are single-model diagnostics, not trained ablations. |
| Extreme summary | Stereo papers fix the 'soft-argmin blends two modes' problem with mode-seeking decodes: pick the dominant mode (DME, SMWA, SMD-Nets), keep only the top-k bins (CoEx), or use the median via L1 risk (Stereo Risk). All of these optimise EPE, bad-pixel rate or edge sharpness, not squared error. On our posteriors every one of them raises pooled RMSE, from 5.48 ft (mean) to 5.52-6.28 ft. Stereo Risk proves the expectation is the L2-risk minimiser, which is exactly our metric, so the transferable parts are the training losses and calibration, not the decoders. |

### Problem formulation

| Field | Value |
|---|---|
| Task definition | Per-pixel disparity from a stereo cost volume whose per-pixel distribution is multimodal at object edges (foreground vs background) or in repetitive texture. Mapping to Erudition: the per-MD-column softmax over 400 TVT levels, which is multimodal when repeated rock bands match the GR (about 16% of rows are multimodal in the dump). |
| Inputs | Stereo cost volumes (PSMNet, GwcNet, GANet, IGEV backbones). For us: the 16-channel 345 x 400 cost image and the U-Net's alignment_logits. |
| Target parameterisation | Discrete disparity bins with a distribution over them (ADL, SMWA, CoEx, Stereo Risk), or a continuous bimodal Laplacian mixture (mu1, b1, mu2, b2, pi) per pixel (SMD-Nets). Ours: TVT - TVT0 in 400 half-foot levels (-99.75 to 99.75 ft). |
| Output representation | A per-pixel distribution in every case, turned into a point estimate by the decoder. SMD-Nets outputs a 2-component mixture, which could feed a K-hypothesis reranker (see B04). |

### Method

| Field | Value |
|---|---|
| Architecture | Loss and decoder changes on standard cost-volume networks; the backbones are unchanged. ADL: multimodal ground truth built per pixel by DBSCAN clustering (eps = 3 px) of ground-truth disparities in an m x n local window, turned into a Laplacian mixture (b = 0.8). The centre pixel's own cluster gets weight at least alpha = 0.5, and the rest is split by cluster size. SMD-Nets: any 2D/3D stereo backbone plus a continuous MLP head that predicts bimodal Laplacian parameters at arbitrary sub-pixel query locations. CoEx: guided cost-volume excitation, then top-k (e.g. k = 2) selection before soft-argmin. Stereo Risk: IGEV-style backbone with Laplacian-kernel interpolation of the discrete distribution into a continuous density; the output is the L1-risk minimiser (the median) found by binary search, and gradients come from the implicit function theorem. |
| Training objective | ADL: cross-entropy against the adaptive multimodal Laplacian-mixture target. SMWA: cross-entropy against a Gaussian-smoothed one-hot target (variance 2), better than L1. SMD-Nets: negative log-likelihood of the bimodal Laplacian mixture. Stereo Risk: L1 loss on the risk-minimiser output, trained end-to-end through the implicit function theorem; L1 risk at both training and test time works best, test-time-only gives marginal gains. CoEx: smooth-L1 on the top-k soft-argmin. Ours today (experiments/arch/seq_NN_cfg.py): alignment cross-entropy to an 'exp_smooth' Gaussian target with sigma 1.25 levels (weight 0.05*4/3), plus Huber (delta 1.0) regression on the expectation (weight 0.05/3), plus a GR penalty (0.05/3). A distance-aware ranked-probability loss ('CDF', masked_typewell_cdf_loss, squared difference of predicted and target CDFs) is implemented but set to weight 0.0. |
| Decoding | DME: mode split, then the mode with maximum cumulative probability, then soft-argmax restricted to that mode. SMWA: argmax peak, monotone-descent bounds, local weighted mean. SMD-Nets: mean of the dominant mixture component. CoEx: soft-argmin over the top-k bins. Stereo Risk: median of the Laplacian-interpolated continuous density. Erudition: full expectation over 400 levels (the model prediction equals the posterior mean to 9e-6 ft). |
| Metric consistent decode | All the stereo decoders here are mode-seeking or median (L1) decoders, not Bayes-optimal for RMSE. Measured on our 155-well posteriors (23,634 columns): mean 5.480 ft; median / L1 risk 5.636; SMWA 6.282; argmax 6.274; posterior_analysis 'dominant_mode' (DME-like) 6.061 at row level; top-k k = 2/4/8/16/32/64 gives 6.226/6.117/5.946/5.697/5.548/5.517. Temperature sharpening p^(1/T): T = 0.5 5.583, 0.7 5.500, 0.85 5.478, 1.0 5.480, 1.2 5.507, 1.5 5.586, 2.0 5.795. The global temperature optimum is flat around 0.85-1.0, a gain of 0.001 ft. The mean decode is confirmed as the right choice. |
| Evidence scope | Per-pixel decoders using local context only (ADL's target uses a local window). No whole-scene or multi-image evidence. In our terms: a single MD column's posterior. |
| Prefix conditioning | None. Stereo has no labelled prefix. Mode choice could in principle be conditioned on the known heel TVT, but that is continuity or prefix evidence (B05/B10), not a decode trick. The post-hoc forward-backward continuity decode has already failed. |
| Propose vs select | Selection at decode time: pick one mode out of the distribution. SMD-Nets' 2-component head is a natural proposal generator (two hypotheses plus weights) that a downstream reranker or gate could select from. That is the B04 path, and the oracle best-mode decode (4.27 ft) bounds its value. |
| Physical constraints | None beyond per-pixel unimodality assumptions (SMWA, DME) or bimodality (SMD-Nets). ADL's multimodal target encodes the fact that edges straddle two surfaces. |

### Data

| Field | Value |
|---|---|
| Dataset | SceneFlow (synthetic), KITTI 2012/2015, Middlebury, ETH3D; SMD-Nets adds its own 4K synthetic UnrealStereo4K dataset. |
| Synthetic to real | ADL and Stereo Risk report better synthetic-to-real generalisation: ADL on KITTI after SceneFlow pretraining; Stereo Risk on Middlebury, 12.63% against 13.76% bad-pixel when trained only on synthetic data. No calibration transfer analysis. |

### Results

| Field | Value |
|---|---|
| Comparability | Not comparable. The stereo metrics are EPE (a mean absolute error, where median-like decodes help) and bad-pixel rates (0/1 losses, where mode decodes help). Our metric is pooled RMSE, for which the conditional mean is optimal. The in-house decode sweep is directly comparable: the same 155 holdout wells, but at posterior-column rather than row resolution (mean 5.480 on columns against 5.474 at row level). |
| Validation protocol | Standard benchmark splits and servers. For our in-house probe, temperature and top-k were swept on the holdout itself (in-sample), which can only flatter the alternatives, and they still lost. |
| Inference budget | Negligible for all decoders (per-column operations over 400 levels). Stereo Risk's binary search added about 50% runtime in stereo. For us, any of these costs well under 1 ms per well. |

### Ambiguity handling

| Field | Value |
|---|---|
| Repeated layers | This is the item's central relevance: repeated bands produce exactly the multimodal per-column posterior these papers target. The evidence says mode-seeking is wrong for our metric. In the dump, band-related categories are wrong_mode 24%, blend 15% and multimodal_miss 12% of SSE. DME-style mode picking turns 'blend' rows (the mean sits between the modes) into 'wrong_mode' rows whenever the dominant mode is wrong. Because the model's dominant mode is wrong often enough, pooled RMSE rises from 5.47 to 6.06. The oracle best-mode decode (4.27 ft) shows the prize exists but can only be claimed with new evidence that identifies the right mode, not with a decoding rule. |
| Drift | Not addressed by any of these methods. The low-spread group in our dump (posterior sd below 3.4 ft, half the columns) is overconfident: mean predicted variance 5.5 ft^2 against an actual MSE of 9.3 ft^2. That is the drift/offset signature (a unimodal hump centred slightly wrong), and no decode fixes a displaced unimodal hump. |
| Uncertainty and calibration | SMD-Nets gives explicit aleatoric uncertainty through its mixture. ADL, Stereo Risk and CoEx do not evaluate calibration. In-house calibration read-out (decode_probe.py / decode_probe2.py): central-interval coverage is 0.586 at the nominal 50% level, 0.826 at 80% and 0.894 at 90%, so calibration is fair, slightly wide in the centre. Split by posterior sd quartile (cut points 3.4 / 6.2 / 10.4 ft): predicted variance against actual MSE is 5.5 vs 9.3 (lowest half, overconfident), 21.2 vs 21.5, 63.8 vs 67.4, and 238 vs 177 (top 5%, underconfident in variance). Their SSE shares are 15%, 21%, 34% and 29%. An in-sample oracle temperature per spread group improves pooled RMSE only from 5.480 to 5.464 ft (-0.016, below the noise floor). Conclusion: temperature-style recalibration of band weights is not a lever. Moving mass to the correct band needs new evidence. |

### Relevance to erudition

| Field | Value |
|---|---|
| Where to use | (a) As a closed question: record that the whole mode-seeking decode family, plus L1/median, top-k and global or spread-conditioned temperature, is measured and rejected for pooled RMSE on our posteriors, so nobody spends GPU time on it. (b) On the training side: the ADL/SMWA finding that the distribution-shaped loss matters more than the decoder suggests trying a distance-aware proper scoring rule (our dormant 'CDF' ranked-probability loss, i.e. discrete CRPS) alongside the alignment cross-entropy. Unlike cross-entropy, it penalises mass on a far-away wrong band more than mass on a near one, which is what squared error cares about. (c) As a proposal generator: SMD-Nets-style K = 2 mixture outputs feeding a reranker or gate (B04), where the mixture mean pi1*mu1 + pi2*mu2 stays RMSE-consistent as the fallback. (d) Monitoring: use per-column posterior sd and multimodality as the confidence signal for B02-style prior fusion. |
| Error mode targeted | Band confusion (wrong_mode, blend, multimodal_miss; about 51% of SSE): the decoders make it worse, and a CRPS-style loss may reduce it slightly. Broad or skewed humps (about 13%): the CRPS loss sharpens them in a distance-aware way. |
| Gap vs our implementation | Already done or already known: expectation decode, which is Bayes-optimal for RMSE under calibration and confirmed empirically here. Cross-entropy to a Gaussian-smoothed target (sigma 1.25 levels), which is exactly the SMWA/Chen 2019 recommendation, plus Huber on the expectation, the L2-risk analogue of Stereo Risk's end-to-end risk training. A DME-like 'dominant_mode' decode, which posterior_analysis.py already measures (6.06 ft against 5.47). An AcfNet-style learned-width unimodal target (arch_v1_unimodal, rejected, +0.162 ft at 60 epochs). The literature is consistent with that rejection: forcing unimodality is the opposite of ADL, whose gain comes from letting targets be multimodal. Not done: (1) The measured sweep of top-k, SMWA, median/L1-risk and temperature. This item did it: all are worse or neutral. (2) A distance-aware proper scoring rule. The 'CDF' (ranked-probability) loss exists in seq_NN_train.py (masked_typewell_cdf_loss) but has weight 0.0 in every recipe. (3) A mixture-density or K-hypothesis head (SMD-Nets) that exposes the two band candidates explicitly for a reranker. (4) An ADL-style multimodal target has no clean analogue. Our ground truth is a single TVT per column, and neighbour-column clustering only matters at faults. It would teach the network to put mass on aliases the label rules out, so it should not be adopted. |
| Proposed experiment | E0 (done in this research, zero cost): decode sweep on results/baseline_check_posterior_all/posterior_all.npz. Scripts are in the session scratchpad (decode_probe.py, decode_probe2.py); fold them into posterior_analysis.py as extra columns (median, topk_k, smwa, temp_T) so every future dump reports them. Result: nothing beats the mean. E1, the ranked-probability (CRPS) loss: in experiments/arch/seq_NN_cfg.py create recipe arch_v7_crps = 0801_V2 with unet_loss_weights['CDF'] set to 0.05/3 (arm A) and 0.05*4/3 (arm B), keeping 'alignment' and 'regression' unchanged. This adds to the loss rather than replacing it, which avoids the confound that sank arch_v1_unimodal. First confirm that masked_typewell_cdf_loss receives the same target_probs as the alignment term and the same mask. Warm-start from results/cnx_tiny_ep150 for 40 epochs, with a control fine-tune of the same length at CDF = 0, two seeds per arm. Judge with seq_NN_robust_compare.py plus the posterior_analysis SSE split (expect a reduction in far wrong_mode SSE, possibly at a cost on ok rows). Also rerun decode_probe2 to check the low-sd overconfidence. E2 (optional, only if B04 proceeds): an SMD-style 2-component head alongside the softmax, trained with mixture NLL at a small weight. Its (mu_k, pi_k) become reranker features, and the softmax mean remains the default output. |
| Implementation effort | E0 done / low (copy into posterior_analysis.py). E1 low: a config-only change once the CDF loss's target and mask wiring is verified. E2 medium: a new head, a loss, and registration in model_modules() / model_named_parameters(). |
| Gpu cost | E0 about 1 minute on CPU. E1 about 1 h per 40-epoch warm-started tiny fine-tune, so about 6 h for 2 arms + control x 2 seeds; or 5 h per arm at 150 epochs from scratch. The CDF loss is a cumsum over 400 levels, a negligible overhead. E2 adds under 1% in parameters and time. |
| Distillation compatibility | Fully compatible. E1 only changes the training loss of the teacher or student, and the KD target (the teacher softmax) is unchanged. The CRPS term can also be applied between teacher and student CDFs as a distance-aware complement to the KL term. E2 heads can be distilled with a mixture-matching loss or ignored by the student. |
| Error correlation with ours | Decode variants of the same posterior are almost perfectly correlated with the mean decode and strictly worse, so they are useless as blend members. A CRPS-trained model is the same architecture and would correlate at about 0.85 or more with the other ConvNeXt runs, adding seed-style diversity only. |
| Risk | (1) The metric argument is decisive against all mode-seeking decoders: for pooled RMSE with a reasonably calibrated posterior, the mean is optimal, and our measurements confirm it (every alternative is +0.04 to +0.8 ft). (2) The CRPS loss could over-sharpen: a proper scoring rule is still minimised by the true distribution, but at finite data it may trade calibration for sharpness differently from cross-entropy, and the expected effect is small (likely within the ±0.39 ft single-run noise). (3) Global calibration is already near-optimal (a temperature gain of 0.001 ft; per-spread-group oracle 0.016 ft), so any calibration-only approach has a ceiling of a few hundredths of a foot on this model. (4) The oracle best-mode decode (4.27 ft) invites mode-selection schemes, but it uses the label. Without new evidence (neighbour wells, a prefix, a longer MD context), mode selection is the failed geo-pick / gate / continuity story again. |
| Notes | Key takeaway for the team: the stereo 'multimodal decoding' literature solves a different loss. Stereo Risk makes it explicit: the expectation is the L2-risk solution, and moving to the L1 median helps EPE. Our metric is L2, so we are already at their L2 optimum. The measured numbers on our own posteriors settle it: mean 5.480; T=0.85 5.478; top-64 5.517; median 5.636; top-8 5.946; DME-like dominant mode 6.061; argmax 6.274; SMWA 6.282. The ADL ablation shows the loss mattered more than the decoder (multimodal + SME 0.80 against unimodal + DME 0.82). That points our effort at training-side changes (a distance-aware CRPS loss, and above all more evidence per column), not decode rules. arch_v1_unimodal's failure is consistent with this literature, since it pushed the opposite way from ADL (forcing unimodality). |

### Other info

- **Id**: B03
- **Uncertain fields (omitted above)**:
  - code_availability
  - core_mechanism
  - metric_and_performance

---

## B04 Multi-hypothesis heads (mixture density networks, annealed MCL NeurIPS 2024, resilient MCL)

### Basics

| Field | Value |
|---|---|
| Name | Multi-hypothesis heads: Mixture Density Networks (Bishop 1994), Multiple Choice Learning / winner-takes-all (Guzman-Rivera 2012, sMCL Lee 2016, relaxed WTA Rupprecht 2017, EWTA Makansi 2019, MTP loss Cui 2019), resilient MCL with learned scoring heads (rMCL, NeurIPS 2023), WTA-as-density-estimator (ICML 2024), annealed MCL (aMCL, NeurIPS 2024), TimeMCL (2025) |
| Authors org | MDN: C. Bishop (Aston, 1994). rMCL / WTA-density / aMCL / TimeMCL: V. Letzelter, D. Perera, M. Fontaine, M. Chen, P. Perez, S. Essid, G. Richard, A. Cortes and co-authors (LTCI Telecom Paris, Valeo.ai). Relaxed WTA: Rupprecht et al. (TUM). EWTA: Makansi et al. (Freiburg). MTP: Cui et al. (Uber ATG). |
| Year | 1994 (MDN); 2012-2019 (MCL family); 2023 (rMCL); 2024 (WTA density, aMCL); 2025 (TimeMCL) |
| Venue link | aMCL: NeurIPS 2024, arXiv:2407.15580 (https://arxiv.org/abs/2407.15580). rMCL: NeurIPS 2023, arXiv:2311.01052 (https://arxiv.org/abs/2311.01052). WTA learners are geometry-aware conditional density estimators: ICML 2024, PMLR 235:27254-27287 (https://proceedings.mlr.press/v235/letzelter24a.html, arXiv:2406.04706). TimeMCL: arXiv:2506.05515 (https://arxiv.org/abs/2506.05515). Relaxed WTA: ICCV 2017 'Learning in an Uncertain World'. EWTA: CVPR 2019 'Overcoming Limitations of Mixture Density Networks'. MTP: ICRA 2019 (arXiv:1809.10732). |
| Evidence level | Peer-reviewed (NeurIPS 2023/2024, ICML 2024, ICCV/CVPR/ICRA). aMCL full text read (HTML); rMCL and WTA-density read at abstract level; TimeMCL arXiv preprint read. No application to geosteering except Alyaev et al. 2022 (see A09). |
| Extreme summary | Predict K hypotheses (plus K scores) in one forward pass and train only the closest one (winner-takes-all), optionally softened by a temperature annealed to zero (aMCL) or by epsilon-relaxation. The hypotheses become an optimal K-point quantization of the conditional distribution, and learned scoring heads (rMCL) turn them into a Voronoi-cell density whose score-weighted mean estimates the conditional mean. |

### Problem formulation

| Field | Value |
|---|---|
| Task definition | Generic ambiguous regression: given x, output a set of plausible y values (sound-source directions, speech sources, future trajectories, time-series futures) instead of one averaged y. Applied to our task: given the lateral/typewell cost image, output K whole-lateral TVT curves (band hypotheses) plus the probability of each. |
| Inputs | Method-agnostic; it is an output head. In our setting it reuses the ConvNeXt/FastViT U-Net features over the 16-channel 345 MD x 400 TVT cost image (GR diffs, trajectory, typewell, prefix channels). |
| Target parameterisation | Each hypothesis is a vector in output space. For us: K curves of 345 TVT values, preferably parameterised as offsets from the known heel TVT or from the per-column expectation so different hypotheses correspond to different bands. MDN alternative: per-column Gaussian mixture over TVT, which our 400-bin softmax already subsumes. |
| Output representation | K hypotheses f_1..f_K(x) plus K scores gamma_k(x) summing to one (rMCL/aMCL), or for MDN K means, K widths and K mixture weights. rMCL interprets the pair as a piecewise-constant density over the Voronoi cells of the hypotheses; ICML 2024 adds kernel-smoothed Voronoi densities for NLL evaluation. |

### Method

| Field | Value |
|---|---|
| Architecture | Shared backbone -> K hypothesis heads (linear or small MLP each) + one scoring head with K logits (softmax or K sigmoids). TimeMCL puts K heads on a TimeGrad-style RNN encoder and outputs K whole multivariate trajectories. MTP (Cui 2019; Alyaev 2022) concatenates K trajectories and K logits in one vector. For our U-Net: pool decoder features along the TVT axis (e.g. posterior-weighted pooling) to one feature per MD column, a 1D conv/transformer along MD, K curve heads (345 values each) and a globally pooled K-logit head. |
| Core mechanism | Winner-takes-all: the loss is min_k l(f_k(x), y), so each head specialises on a region of output space and the heads converge to a centroidal Voronoi tessellation (conditional K-means) of p(y/x) rather than collapsing to the mean. Known failure: hypothesis collapse (some heads never win). Fixes: relaxed WTA (epsilon weight on losers, but it biases hypotheses toward the barycenter), EWTA (top-n winners evolving from n=K to 1), and aMCL: soft Boltzmann assignment q_k ~ exp(-l_k/T) with T annealed to 0 (deterministic annealing); hypotheses split at critical temperatures (first split at T = 2*lambda_max of the conditional covariance), following the rate-distortion curve and avoiding greedy local minima. rMCL: scoring heads trained with binary cross-entropy on 'this head won', giving calibrated cell probabilities; conditional-mean estimator = sum_k gamma_k f_k. |
| Training objective | WTA: min_k //f_k - y//^2 plus a scoring loss (CE/BCE on the winner index). aMCL: sum_k softmin_T(l)_k * l_k with the assignment detached, temperature schedule T(t) decaying to 0, plus scoring loss. Relaxed WTA: (1-eps) on the winner + eps/(K-1) on others. MDN: Gaussian-mixture negative log-likelihood (prone to mode collapse and numerical instability, the motivation for EWTA). Use L2 (not L1) per hypothesis so each cell converges to its conditional mean, which is what pooled RMSE needs. |
| Decoding | (i) score-weighted mean sum_k gamma_k f_k (conditional-mean estimate); (ii) argmax-score hypothesis (mode-seeking); (iii) pass all K hypotheses + scores to a reranker/gate that uses extra evidence and outputs new weights, then take the reweighted mean. Papers report distortion (min over hypotheses, an oracle) and RMSE of the weighted mean. |
| Metric consistent decode | Only the score-weighted (or reweighted) mean is RMSE-consistent. Picking the top-scored hypothesis is mode-seeking and will raise pooled RMSE unless the selector is far more accurate than the posterior (the same reason arg-max lost to the expectation in our repo). Oracle 'distortion' numbers in MCL papers are not achievable metrics. |
| Evidence scope | Whatever the backbone sees; for us the whole lateral plus typewell (U-Net receptive field), single well. A reranker on top can add whole-lateral path-level evidence. |
| Prefix conditioning | Not part of the method. Hypotheses can be anchored to the last known heel TVT (offset parameterisation); prefix channels already enter the backbone. |
| Propose vs select | Propose-and-score in one pass. The K hypotheses are an explicit candidate set, so it slots into a propose + rerank/gate pipeline (like 3rd place's five candidates + softmax gate, or 2nd place's sampled rollouts + shape-gate top-k average). |
| Physical constraints | None intrinsic. Each hypothesis is a whole path, so it can be regularised (smoothness / dS curvature, start at known TVT, near-linear drift), which is impossible for per-column marginals. |

### Data

| Field | Value |
|---|---|
| Dataset | UCI regression, WSJ0-mix speech separation (aMCL), sound-event localisation (rMCL), GluonTS Solar/Electricity/Exchange/Traffic/Taxi/Wikipedia (TimeMCL), vehicle trajectories (MTP/EWTA). In-domain: Alyaev 2022 on the GWC 2020 synthetic typelog (A09). |
| Synthetic to real | Not addressed. WTA learns diversity from the spread of targets among similar-looking inputs; our 85% synthetic generator provides this at scale for repeated-layer cases, but scores learned on synthetic data may be miscalibrated on real wells. |

### Results

| Field | Value |
|---|---|
| Metric and performance | aMCL: WSJ0-2mix SI-SDR 16.85+-0.13 vs PIT 16.88+-0.10; 3-mix 10.00 vs 10.01, lower seed variance than MCL; UCI RMSE comparable to relaxed WTA and competitive with MC-dropout / deep ensembles. TimeMCL: distortion on par with TimeGrad/TACTiS-2 at orders of magnitude fewer FLOPs, smoother trajectories. rMCL: better calibrated than MCL for sound-source localisation. None report RMSE gains of the weighted-mean decode over a strong density model; the wins are in distortion (oracle-of-K) and multimodal fidelity. |
| Comparability | Not comparable; no geosteering numbers. The relevant internal ceiling is our oracle best-mode decode 4.27 ft vs 4.84 ft deployed (a perfect selector over per-column modes). |
| Validation protocol | Standard fixed splits; multiple seeds reported for aMCL. Not applicable to our well-level holdout. |
| Inference budget | Negligible: K extra heads on the existing backbone (well under 1 ms). A reranker scoring K paths with a GR-misfit integral is O(K x columns) on CPU. Fits the Kaggle/rig budget. |

### Ambiguity handling

| Field | Value |
|---|---|
| Repeated layers | Exactly the target case: each repeated band can be owned by one hypothesis, so the model states 'band A with prob p, band B with prob 1-p' as two coherent whole paths instead of a per-column blend. WTA pressure rewards committing a head to a band, removing hedged blends inside a head (our blend error, 15%). The mixture still carries the ambiguity; RMSE drops only if the scores (or a reranker) are better calibrated than today's softmax band weights. |
| Drift | Indirect. Hypotheses are whole curves, so heads can represent 'same band, different slope', a quantization of drift uncertainty. Adds no evidence against drift by itself. |
| Uncertainty and calibration | rMCL scores are trained as winner probabilities and reported better calibrated than vanilla MCL; aMCL reduces collapse. Hypothesis-level calibration can be checked with a reliability diagram of 'hypothesis k won' over holdout wells and fixed by temperature scaling of score logits. Our current calibration is only per bin. |

### Relevance to erudition

| Field | Value |
|---|---|
| Where to use | Auxiliary head on the ConvNeXt/FastViT U-Net decoder (experiments/arch/seq_NN_models.py, seq_NN_trf_unet.py) trained jointly with the existing column cross-entropy; its K paths + scores feed a whole-lateral reranker that outputs reweighted scores; final prediction = reweighted mean, blended with the existing expectation. |
| Error mode targeted | Band confusion (~51%: wrong_mode 24%, blend 15%, multimodal_miss 12%); secondarily drift via slope hypotheses. |
| Gap vs our implementation | Today the ConvNeXt outputs per-column marginals only (400-bin softmax, expectation decode); nothing represents a joint whole-lateral hypothesis, and the loss (alignment soft CE + expectation regression + GR_penalty; unet_loss_weights in experiments/arch/seq_NN_cfg.py / seq_NN_train.py) has no diversity or path-coherence term. Consequences: (1) we cannot assign a probability to 'the well is in band B along the whole lateral', so evidence visible only at path level (integrated GR misfit of the entire path, linearity of drift, consistency with the prefix) never reaches the weights; our failed per-point gates and XY prior acted at point level. (2) Per-column MDN/MCL is NOT a gap: a 400-bin softmax is already a more flexible per-column density. (3) The 2nd-place code already does a stochastic propose + rerank (rollout_paths samples paths from the AnchorCNN move field, then averages the top-k by shape distance to a reference MLP), but its proposals come from a weaker model (AnchorCNN 5.94 ft holdout). (4) Theory: with a calibrated mixture, the score-weighted mean of K WTA hypotheses and our expectation both estimate E[y/x], so an MCL head alone cannot beat a calibrated softmax on pooled RMSE; any gain must come from path-level evidence used by a reranker, or from the auxiliary training signal improving the shared backbone. |
| Proposed experiment | E0 (no training, go/no-go, ~1 h CPU): from posterior_all.npz (seq_NN_posterior_dump.py --save-all) extract K=2-4 coherent paths per well by tracking per-column modes, or by forward-filtering-backward-sampling many paths (reuse posterior_fb_decode.py) and k-means clustering them. Report (a) path-oracle RMSE vs point-oracle 4.27 vs expectation; (b) path-level soft reweighting: w_k proportional to mass_k * exp(-beta * integrated GR misfit of path k against the typewell over the unknown suffix - beta2 * linear-drift residual), beta fitted on TRAINING wells only; decode = sum_k w_k path_k, blended with the expectation. Keep only if the pooled gain on the 155 holdout wells exceeds the +-0.39 ft noise floor. E1 (training): add mhyp_k (4/6), mhyp_lambda (0.1/0.3), mhyp_mode ('amcl' with T0 ~ 25 ft^2 decaying to 0 over the first 60% of epochs, or 'rwta' eps=0.05) to seq_NN_cfg.py; head in seq_NN_models.py: posterior-weighted TVT pooling of decoder features -> 1D conv along MD -> K offset curves from the heel TVT + K score logits; loss in seq_NN_train.py: masked L2 per hypothesis over unknown columns with aMCL soft assignment + CE scoring loss, added as a new unet_loss_weights key. Evaluate with seq_NN_holdout_eval.py: (i) expectation of the unchanged softmax (does the auxiliary loss help the backbone?), (ii) score-weighted mean of paths, (iii) E0 reranker on learned paths, (iv) 50/50 blends. Control: identical config with mhyp_lambda=0, same seed, 150 epochs; 2 seeds per arm because of the noise floor. Log hypothesis-collapse rate and score reliability. |
| Implementation effort | Medium: head and loss are small, but path parameterisation, masking of variable-length suffixes, the annealing schedule and the reranker need care. E0 alone is low effort. |
| Distillation compatibility | Compatible. Either give the FastViT student the same head and distil scores by KL and hypotheses by L2 after Hungarian matching of teacher/student heads (heads are permutation-ambiguous), or keep KD on per-column posteriors as today and let the student learn its own heads. The existing per-column KD path is unaffected. |
| Error correlation with ours | High correlation if used as a weighted mean (same backbone, same information). Diversity comes only from the reranker's path-level evidence, so it is a gate/reweighting component rather than an independent blend member. |
| Risk | (1) Metric: any hard selection is mode-seeking; our continuity decode and arg-max both lost for this reason, so only soft reweighting is safe. (2) Selection overfitting: 3rd place's gate had CV 5.29 vs private 5.84, and our learned per-point gates were rejected (5.017 vs 4.905 for one weight). (3) Data-limited: only 618 real training wells to calibrate path scores. (4) Collapse and temperature-schedule sensitivity (a stated aMCL limitation, tied to the LR). (5) If the path oracle is barely below 4.27 or GR-misfit reweighting does not beat the softmax weights, the band information is not in the data and no head will help. |
| Notes | In-domain precedent: Alyaev et al. 2022 (A09) used MTP best-of-M on the same kind of GR difference image (M=7) and reported mostly oracle-level wins. The ICML 2024 result (WTA hypotheses + scores form a Voronoi conditional density) is the formal reason the weighted-mean decode cannot beat a calibrated softmax by itself. aMCL is the most robust training choice (lower seed variance, no barycenter bias); prefer it over relaxed WTA. A trained CRF (item B05) with forward-filtering-backward-sampling is a training-free generator of coherent path hypotheses and could replace the learned head as the proposal stage. |

### Other info

- **Uncertain fields (omitted above)**:
  - code_availability
  - gpu_cost

---

## B05 Differentiable alignment / structured DP losses (Mensch & Blondel, soft-DTW, Drop-DTW, Deep Declarative DTW, neural CRF/HMM)

### Basics

| Field | Value |
|---|---|
| Name | Differentiable alignment and structured DP losses: smoothed-max DP (Mensch & Blondel, ICML 2018), soft-DTW (Cuturi & Blondel, ICML 2017) and soft-DTW divergence (Blondel, Mensch & Vert, AISTATS 2021), Drop-DTW (NeurIPS 2021), Deep Declarative DTW (DecDTW, ICLR 2023), neural linear-chain CRF / HMM layers (incl. hybrid CNN-CRF stereo 2017, SGM-Nets 2017) |
| Authors org | A. Mensch, M. Blondel (Inria / NTT, later Google); M. Cuturi, M. Blondel; M. Blondel, A. Mensch, J.-P. Vert (Google); N. Dvornik, I. Hadji, K. Derpanis, A. Garg, A. Jepson (Samsung AI Centre Toronto); M. Xu, S. Garg, M. Milford, S. Gould (QUT / ANU); P. Knobelreiter, C. Reinbacher, A. Shekhovtsov, T. Pock (TU Graz); A. Seki, M. Pollefeys (Toshiba / ETH) |
| Year | 2017-2023 (soft-DTW 2017, CNN-CRF stereo 2017, SGM-Nets 2017, Mensch & Blondel 2018, soft-DTW divergence 2021, Drop-DTW 2021, DecDTW 2023) |
| Venue link | Mensch & Blondel, ICML 2018, arXiv:1802.03676 (https://arxiv.org/abs/1802.03676). Soft-DTW, ICML 2017, arXiv:1703.01541. Differentiable Divergences Between Time Series, AISTATS 2021, arXiv:2010.08354. Drop-DTW, NeurIPS 2021, arXiv:2108.11996 (https://github.com/SamsungLabs/Drop-DTW). DecDTW, ICLR 2023, arXiv:2303.10778 (https://openreview.net/forum?id=UClBPxIZqnY). CNN-CRF stereo, CVPR 2017, arXiv:1611.10229. SGM-Nets, CVPR 2017. |
| Evidence level | Peer-reviewed (ICML, AISTATS, NeurIPS, ICLR, CVPR). Read at abstract / summary level plus known method details; no application to geosteering found. Internal evidence from our repo: the 2nd-place AnchorCNN is a locally normalised, teacher-forced path model (MEMM-like) scoring 5.94 ft holdout, and the post-hoc forward-backward decode on ConvNeXt posteriors failed (16 focus wells 11.32 -> 12.14). |
| Extreme summary | Replace the hard max/min in a dynamic program (Viterbi, DTW, CRF forward) with a smoothed max (log-sum-exp or other strongly convex regulariser) so the DP value and its argmax relaxation become differentiable; gradients of the smoothed value are the expected path (edge marginals). This lets a network learn unary and pairwise potentials end-to-end against a path-level loss, i.e. a trained version of the continuity decode, with Drop-DTW adding skip costs and DecDTW giving exact-path gradients. |

### Problem formulation

| Field | Value |
|---|---|
| Task definition | Structured prediction over monotone or Markov paths: sequence labelling (Viterbi/CRF), time-series alignment and averaging (DTW), step localisation in video (Drop-DTW), audio-to-score alignment and visual place recognition (DecDTW), stereo disparity (CNN-CRF, SGM-Nets). For us: predict the TVT-level path over 345 MD columns given the cost image, with a learned Markov transition between neighbouring columns. |
| Inputs | Unary scores from a network (for us the ConvNeXt/FastViT per-column logits over 400 TVT levels) plus pairwise scores: a fixed or learned transition kernel over dTVT per column, optionally conditioned on column features (inclination, predicted dip, GR contrast), as SGM-Nets predicts penalties from image patches. |
| Target parameterisation | Discrete state = TVT level (0.5 ft bins) per MD column; path = sequence of levels. DTW variants instead align lateral GR samples to typewell depths monotonically, which is wrong for laterals that go up and down through strata; the linear-chain CRF over TVT levels is the right form (non-monotone, like the 2nd-place DP and the patents in A04). |
| Output representation | Path-level distribution: log-partition, per-column marginals (forward-backward), pairwise marginals, MAP path (Viterbi), or samples (forward-filtering backward-sampling). The per-column marginals are directly comparable to our current softmax. |

### Method

| Field | Value |
|---|---|
| Architecture | CNN backbone -> unary potentials theta_t(s) (T=345 columns x S=400 levels) + transition potentials A_t(s, s') (banded, /s - s'/ <= ~80 bins i.e. 40 ft, translation-invariant kernel k_t(ds) predicted per column by a small head, as a Toeplitz/banded convolution) -> differentiable forward-backward layer (log-semiring, i.e. Mensch & Blondel with negentropy regulariser) -> marginals. Soft-DTW: DP over an alignment cost matrix with soft-min. Drop-DTW: extra drop states with learned drop costs. DecDTW: continuous-time DTW solved as an inner optimisation, gradients by implicit differentiation. CNN-CRF stereo: CNN unaries + contrast-sensitive pairwise terms, trained with a structured SVM loss. |
| Core mechanism | Smoothed Bellman recursion: max_Omega(x) = log sum exp(x) (entropy) or sparsemax-like (squared L2) turns DP into a differentiable operator whose gradient w.r.t. potentials equals expected edge occupancies. With the entropic smoother, the forward value is log Z of a linear-chain CRF and the gradient is the CRF marginal, so training by CRF negative log-likelihood of the true path gives gradient = model marginal - one-hot truth. Crucially the unaries are learned jointly with the transitions, so they become local evidence (likelihood-like potentials) rather than already-contextualised marginals; the global normalisation then produces calibrated path marginals. Globally normalised CRFs also avoid the label bias / exposure bias of locally normalised teacher-forced models (MEMM), which is what the 2nd-place AnchorCNN is. |
| Training objective | Options: (a) CRF NLL of the discretised true TVT path: -score(y) + log Z (log Z by forward algorithm); (b) marginal (posterior) likelihood / CE on forward-backward marginals against our existing soft typewell targets, which matches a per-point metric better (Kakade et al. 2002 alternative objective); (c) expected-RMSE loss on the marginal mean sum_s gamma_t(s) x_s (differentiable through forward-backward), i.e. the existing 'regression' term applied after the CRF; (d) soft-DTW divergence / Drop-DTW as alignment losses (not the right structure for non-monotone laterals); (e) structured SVM (CNN-CRF stereo). Recommended for pooled RMSE: (b)+(c), optionally with a small (a) weight. |
| Decoding | Forward-backward marginals then expectation (RMSE-consistent). Viterbi / MAP path is mode-seeking. Forward-only filtering is what the 2nd-place dp_expected_level does (exact for its locally normalised chain because the move field already sees the whole image). DecDTW returns the exact optimal path. |
| Metric consistent decode | Marginal expectation from forward-backward is Bayes-optimal for pooled RMSE if the CRF marginals are calibrated; Viterbi, argmax and DecDTW's optimal path are mode-seeking and would raise RMSE on ambiguous wells. |
| Evidence scope | Whole lateral jointly (every column's marginal conditions on all unaries through the chain), single well plus typewell. Longer-range dependence than first-order (e.g. constant drift slope) needs a second-order state (level, slope), as in the silver structural tracker (A15). |
| Prefix conditioning | Natural: clamp the chain to the known heel TVT at the first unknown column (initial distribution = delta or narrow Gaussian), as posterior_fb_decode.py and dp_expected_level already do; the prefix columns can also be clamped observations. |
| Propose vs select | Single structured output (marginals), but forward-filtering-backward-sampling yields coherent path samples that can feed a propose + rerank stage (item B04) or the 2nd-place-style shape gate. |
| Physical constraints | Continuity (banded transitions), learned step-size distribution per column, optional asymmetric up/down penalties (SGM-Nets style), optional second-order smoothness via augmented state; monotonicity is imposed only by DTW variants and is inappropriate here. |

### Data

| Field | Value |
|---|---|
| Dataset | Mensch & Blondel: CoNLL NER and machine translation attention; soft-DTW: UCR time-series; Drop-DTW: CrossTask, COIN, YouCook2 instructional videos; DecDTW: audio-to-score, Oxford RobotCar / Nordland visual place recognition; CNN-CRF: Middlebury 2014, KITTI 2015. No well-log datasets. |
| Synthetic to real | Not addressed. For us the transition kernel could be fitted on real training wells only (TVT change per 32 ft column: std 0.78 ft, q99 2.0 ft, max 8.6 ft measured on training wells), while unaries learn from the synthetic-heavy mix; a real-only fine-tuning stage for the pairwise head is cheap. |

### Results

| Field | Value |
|---|---|
| Metric and performance | Mensch & Blondel: structured attention/NER results competitive with standard CRF training; soft-DTW: better barycenters and forecasting than Euclidean loss; Drop-DTW: SOTA weakly supervised step localisation on CrossTask/COIN; DecDTW: SOTA audio-to-score alignment and place recognition vs soft-DTW; CNN-CRF stereo: strong Middlebury/KITTI results with shallow CNNs and no post-processing. None of these are RMSE-on-ambiguous-path numbers. |
| Comparability | Not comparable externally. Internal anchors: ConvNeXt expectation 4.84 ft (FastViT distilled) / 5.47 ft (60 ep); AnchorCNN locally normalised path model 5.94 ft; post-hoc FB decode worse than expectation on 16 focus wells. |
| Validation protocol | Fixed benchmark splits in the papers; not applicable to our well-level holdout. |
| Inference budget | Banded forward-backward: 345 sequential steps of B x 400 x ~161 multiply-adds, a few ms per well on GPU and ~tens of ms on CPU; the 2nd-place repo measured its DP decode at 28.7 ms/sample batched. Training memory: store alpha/beta (345 x 400 floats per well) or use a custom autograd that recomputes; negligible vs the U-Net. |

### Ambiguity handling

| Field | Value |
|---|---|
| Repeated layers | A trained chain helps only if pairwise evidence disambiguates bands: a band switch requires a large jump, which the learned transition makes expensive, so a path that is consistent with the heel anchor and continuous is preferred. Because training calibrates the product unary x transition, it should not over-commit as the post-hoc decode did; but if the correct band is also reachable by a smooth path, the chain cannot choose it and the marginal stays bimodal (correctly, for RMSE). |
| Drift | First-order transitions do not penalise a constant-slope drift (each step is small). Needs a second-order state (level x slope bins, or a transition conditioned on a predicted dip per column) to encode 'one straight-line drift per well'; the state-space grows (400 x ~9 slope bins) but stays tractable. |
| Uncertainty and calibration | CRF marginals are calibrated under the model when trained by likelihood; our post-hoc product of already-contextualised marginals double-counts evidence and over-sharpens, which is the mechanism behind the failed decode. Calibration can be checked with the existing per-bin reliability tools and a temperature on unaries/transitions fitted on training wells. |

### Relevance to erudition

| Field | Value |
|---|---|
| Where to use | A CRF head after the ConvNeXt/FastViT decoder logits: alignment_logits become unaries, a small head predicts per-column transition kernels, forward-backward produces marginals that replace the softmax in the alignment CE, regression and GR_penalty losses (experiments/arch/seq_NN_train.py) and in the expectation decode (seq_NN_holdout_eval.py). Also applicable to the AnchorCNN: replace its teacher-forced local CE with global normalisation. |
| Error mode targeted | Band confusion (~51%), in particular blend (15%) and wrong_mode picked after a spurious jump; drift (~25%) only with a second-order (slope) state. |
| Gap vs our implementation | What we do: per-column independent softmax trained with per-column soft CE + expectation regression + GR_penalty; the U-Net's receptive field gives implicit context but no explicit pairwise consistency; the post-hoc forward-backward in kaggle_1st_place/solution/posterior_fb_decode.py multiplies the network's per-column POSTERIORS (already conditioned on neighbouring columns) by a fixed Gaussian-mixture transition kernel fitted on training wells, which treats marginals as independent likelihoods, double-counts shared evidence and over-sharpens toward whichever band has more mass near the anchor (hence 'wins big where it locks onto the right band and loses where it locks onto a wrong one'). The 2nd-place AnchorCNN is a trained path model, but locally normalised and trained by teacher forcing on the true path only (anchor_loss in kaggle2ndplace/src/gr2tvt_model.py), then decoded by forward filtering or sampled rollouts; it never sees off-path states during training except via the distillation experiment. Gap: nobody in our stack trains unaries and transitions JOINTLY under a globally normalised path likelihood or a marginal-based loss. That is exactly what Mensch & Blondel / neural CRF offer: the unaries learn to be local evidence and the transitions learn what continuity is worth, so the chain's marginals are calibrated by construction instead of hand-tuned (sig1, sig2, w2 in kernel()). DTW-family losses (soft-DTW, Drop-DTW, DecDTW) are NOT a direct fit because they assume monotone alignment; their contribution is Drop-DTW's idea of skip states (faults / missing GR) and DecDTW's exact-path gradients, both secondary. |
| Proposed experiment | E0 (cheap check of the double-counting hypothesis, no training): in posterior_fb_decode.py add a 'tempered emissions' option e = p^tau with tau in {0.1, 0.2, 0.3, 0.5} (fitting tau and the kernel on TRAINING-well posteriors, which requires dumping training wells with seq_NN_posterior_dump.py), run on posterior_all.npz for all 155 holdout wells. If some tau beats the raw expectation by more than the 0.39 ft noise floor, that already proves pairwise structure adds information once double counting is removed. E1 (main, trained CRF head): new module experiments/arch/seq_NN_crf.py with a banded log-space forward-backward (log-sum-exp over a +-80 bin window via unfold, custom autograd or checkpointing), initial state clamped at the heel TVT, transition kernel k_t(ds) = softmax over 161 offsets predicted from decoder features pooled per column (initialise to the training-well dTVT histogram). Config flags in seq_NN_cfg.py: crf_head=True, crf_band_bins=80, crf_kernel='learned'/'fixed', crf_nll_weight (0 or 0.05). In seq_NN_train.py route extra['alignment_logits'] through the CRF and apply the existing alignment soft CE, regression and GR_penalty to the CRF marginals (loss (b)+(c)), plus optional CRF NLL of the discretised true path. Decode: marginal expectation. Control: same config with crf_head=False, same seed, 150 epochs, 2 seeds each, seq_NN_holdout_eval.py on 155 wells; also report the focus-well subset used by posterior_fb_decode.py for continuity with the negative result. E2 (if E1 helps): second-order state (level x 9 slope bins) for drift; FastViT student with the same CRF head, KD on CRF marginals. |
| Implementation effort | Medium: the banded forward-backward and its memory-efficient backward are the only new code (~150 lines with tests against brute force on tiny grids); masking variable-length suffixes and the prefix clamp are already solved in posterior_fb_decode.py and dp_expected_level. E0 is low effort. Second-order state is medium-high. |
| Distillation compatibility | Good: KD on output posteriors becomes KD on CRF marginals (or on unaries + kernels). A FastViT student can carry the same cheap CRF head; the head's cost is independent of the backbone. |
| Error correlation with ours | Moderate: same backbone and inputs, so errors stay correlated on truly ambiguous wells, but the decisions on blend / jump columns change systematically; a CRF-head model is a reasonable blend member with the plain ConvNeXt, and more diverse than a second seed. |
| Risk | (1) The ConvNeXt's receptive field may already capture most continuity, so the learned transitions add little (the model can learn a flat kernel). (2) A first-order chain cannot fix band confusion where both bands are smoothly reachable; the metric-optimal answer there is a hedge, which we already output. (3) CRF NLL on the true path is mode-seeking-ish in training and can sharpen marginals; prefer marginal/expectation losses. (4) Training-time sequential loop and numerical stability in log space. (5) The AnchorCNN precedent: a trained path model at 5.94 ft is worse than the per-column U-Net, though for backbone and normalisation reasons, not proof against structure. (6) Our ablations at 60 epochs (axial attention, RAFT, AcfNet) were all null; a 150-epoch comparison with two seeds is needed to resolve effects near 0.39 ft. |
| Notes | Why the post-hoc decode failed and the trained one might not: the forward-backward product assumes emissions are conditionally independent given the state, but U-Net column posteriors share almost all of their evidence with their neighbours; multiplying 345 of them with a smoothness prior raises the effective confidence enormously. Training end-to-end removes this by letting the unaries carry only the evidence not explained by the chain. Closest analogues in stereo: SGM-Nets (learned DP penalties) and hybrid CNN-CRF (end-to-end structured SVM). The soft-DTW divergence fixes soft-DTW's self-similarity bias if a DTW-type loss is ever used for typewell-to-lateral GR alignment. Related items: A03 (DTW), A04 (DP patents), A15 (explicit dip state), B06 (differentiable particle filters), B08 (retinal layer graph-search CNNs). |

### Other info

- **Uncertain fields (omitted above)**:
  - code_availability
  - gpu_cost

---

## B06 Differentiable particle filters (survey 2023, normalizing-flow DPF, learned GMM proposals, semi-supervised DPF)

### Basics

| Field | Value |
|---|---|
| Name | Differentiable particle filters (DPFs): Chen & Li survey (2023); conditional-normalizing-flow DPF (2021) and normalizing-flow DPF (2024/25); StateMixNN learned Gaussian-mixture proposals (2024/25); semi-supervised DPF (ICRA 2021, revisited 2024); Mixture Density Particle Filter (NeurIPS 2023); PyDPF library (2025) |
| Authors org | Xiongjie Chen & Yunpeng Li (Univ. of Surrey / King's College London) for the survey, CNF-DPF and NF-DPF; Hao Wen, Xiongjie Chen, Georgios Papagiannis, Conghui Hu, Yunpeng Li (semi-supervised DPF); Benjamin Cox, Santiago Segarra, Victor Elvira (StateMixNN, Edinburgh / Rice); Ali Younis & Erik Sudderth (UC Irvine, MDPF); John-Joseph Brady, Benjamin Cox, Victor Elvira, Yunpeng Li (PyDPF). Foundations: Jonschkowski et al. (RSS 2018), Karkus et al. PF-net (CoRL 2018), Corenflos et al. OT resampling (ICML 2021). |
| Year | 2021-2025 (survey 2023; CNF-DPF 2021; semi-supervised DPF 2020/2021, revisited 2024; MDPF 2023; NF-DPF arXiv 2024, TSP 2025; StateMixNN arXiv 2024, Signal Processing 2025; PyDPF 2025) |
| Venue link | Survey: Foundations of Data Science (AIMS), arXiv:2302.09639 (https://arxiv.org/abs/2302.09639). CNF-DPF: IEEE conference 2021, arXiv:2107.00488 (https://arxiv.org/abs/2107.00488; https://ieeexplore.ieee.org/document/9626998/). NF-DPF: IEEE Transactions on Signal Processing, arXiv:2403.01499 (https://arxiv.org/abs/2403.01499). StateMixNN: Signal Processing (Elsevier) 2025, arXiv:2411.15638 (https://arxiv.org/abs/2411.15638; https://www.sciencedirect.com/science/article/pii/S0165168425001124). Semi-supervised DPF: ICRA 2021, arXiv:2011.05748 (https://arxiv.org/abs/2011.05748); revisited objectives arXiv:2405.01251. MDPF: NeurIPS 2023, arXiv:2404.08789 (https://proceedings.neurips.cc/paper_files/paper/2023/hash/85b2ff7574ef265f3a4800db9112ce14-Abstract-Conference.html). PyDPF: arXiv:2510.25693 (https://arxiv.org/abs/2510.25693). |
| Evidence level | Peer-reviewed (ICRA, NeurIPS, IEEE TSP, Signal Processing, FoDS) plus arXiv preprints; no application to well logs, geosteering or ROGII found. Relevance judgements below are our analysis, not reported results. |
| Extreme summary | DPFs make the particle filter's transition, proposal and measurement models neural networks and train them end-to-end through the filter (differentiable or gradient-corrected resampling), so the emission model is learned for use inside sequential inference rather than in isolation. For us the valuable idea is not the particles (a 1D TVT state is solved exactly on a grid) but training / tempering the U-Net posterior so that it composes correctly with a sequential dip-aware tracker. |

### Problem formulation

| Field | Value |
|---|---|
| Task definition | Sequential Bayesian state estimation (filtering / smoothing) in state-space models whose dynamics and observation likelihood are unknown or high-dimensional (images), learning those models from data; benchmark tasks are robot localisation (maze / house environments), visual object tracking and synthetic nonlinear SSMs. |
| Inputs | Observation sequences (images, sensor readings, low-dimensional synthetic observations) and optionally actions; labelled state trajectories for supervised variants, few or no labels for semi-supervised / likelihood-trained variants (SDPF, StateMixNN). In a ROGII transfer the observation at each MD column would be the lateral GR window plus the typewell, or directly the U-Net per-column posterior. |
| Target parameterisation | Explicit latent state x_t (pose, object position, generic hidden state). A ROGII instance would use (TVT level, structural slope r = dS/dMD with S = TVT + Z, optional GR bias), the same state as our PF. |
| Output representation | Weighted particle set, i.e. an empirical (multimodal) posterior per step; MDPF turns it into a continuous kernel-mixture density; NF-DPF gives valid normalised densities; StateMixNN uses Gaussian-mixture proposal and transition densities. Point estimates are usually the weighted particle mean. |

### Method

| Field | Value |
|---|---|
| Architecture | Classical PF loop with neural components: (1) dynamic model p_theta(x_t / x_{t-1}) as an MLP-parameterised Gaussian / mixture or a conditional normalizing flow; (2) proposal q_phi(x_t / x_{t-1}, y_t) that looks at the latest observation (CNF-DPF, NF-DPF, StateMixNN Gaussian mixtures) instead of the bootstrap proposal; (3) measurement model l_psi(y_t / x_t) as a learned feature-similarity network or a conditional flow density; (4) differentiable resampling: soft resampling (Karkus 2018), entropy-regularised optimal-transport resampling (Corenflos 2021), or none/stop-gradient; MDPF instead represents the posterior as a kernel mixture and uses an importance-sampling gradient estimator so resampling gradients are unbiased and low variance. PyDPF implements these variants behind one PyTorch API. |
| Core mechanism | End-to-end learning through the filter: the loss is computed on the filter's output (e.g. RMSE of the weighted mean against ground truth, or the marginal log-likelihood / ELBO of observations), so the measurement model is trained to be a good likelihood for sequential fusion, including its sharpness (temperature) relative to the dynamics; algorithmic structure (predict-update-resample) acts as a prior that reduces data needs. MDPF adds that multimodal posteriors can be tracked stably over long horizons when gradients through resampling are unbiased, whereas truncated-gradient DPFs are unstable and lose modes. |
| Decoding | Weighted particle mean (most common) or mixture-density statistics; smoothing variants use backward passes. For RMSE the weighted mean of the (smoothed) posterior is the right summary. |
| Metric consistent decode | RMSE-consistent when the weighted mean is used and trained with RMSE; mode-seeking only if a best-particle / MAP summary is taken. The real risk for us is not the decode but an over-confident fused posterior (see gap field), which makes the mean behave like an arg-max. |
| Evidence scope | Sequence-level: causal filtering over the whole trajectory, optional smoothing (whole sequence). No multi-sequence (multi-well) evidence sharing. |
| Prefix conditioning | Not addressed in the literature beyond initial-state priors; in ROGII the known heel prefix would set a tight initial distribution at the last known row and an initial slope estimate (as our PF already does via _initial_rate). |
| Propose vs select | Produces a full posterior per step, so it can serve as a candidate path, an input channel, or a posterior re-weighting of the U-Net output; it slots in as an ensemble member or as a decode stage. |
| Physical constraints | Only whatever the dynamic model encodes; in our transfer: smooth structural slope random walk (dip), rare fault jumps, S = TVT + Z identity, GR bias drift. Learned dynamics could learn well-type-specific dip statistics from training wells. |

### Data

| Field | Value |
|---|---|
| Dataset | Robotics / vision benchmarks: DeepMind Lab maze localisation, House3D, KITTI-style visual odometry, disk / object tracking, synthetic linear-Gaussian and stochastic-volatility SSMs. No geoscience data. |
| Synthetic to real | Mostly synthetic or simulated; SDPF evaluates on a real robot dataset. For us the relevant point is that DPF emissions are trained on the same distribution they are applied to, whereas our U-Net is trained on 85% synthetic wells, so any learned temperature / emission scaling should be fitted on real training wells only. |

### Results

| Field | Value |
|---|---|
| Comparability | Not comparable: different domains and metrics (pose RMSE, NLL). Only our own PF / HMM / fb-decode numbers are on the 155-well holdout. |
| Validation protocol | Standard train / test splits of simulated trajectories; multiple training seeds in MDPF (stability across runs is a headline result). No spatial grouping issues. For us any learned emission scaling must be fitted on training wells and scored on the holdout with seq_NN_robust_compare.py paired bootstrap. |
| Inference budget | Cheap for a 1D/2D state: a grid HMM over 400 levels x ~50 slopes x 345 columns is ~7M multiply-adds per well (anchor_hmm.py runs 0.11 s/well at 129 x 53); a 512-particle PF is tens of ms per well. Training a DPF end-to-end adds a sequential loop over 345 columns per well to the backward pass, which is slow in Python but fine if written as batched log-sum-exp on the grid. |

### Ambiguity handling

| Field | Value |
|---|---|
| Repeated layers | Particles or mixture components can keep several band hypotheses alive; MDPF specifically targets stable tracking of multiple posterior modes. But a filter can only resolve band confusion if the sequence contains disambiguating evidence the per-column posterior has not already used; the U-Net's receptive field already spans the whole 345-column image, so most of that evidence is already inside each column posterior. Our PF notes: mode collapse was ruled out (512 vs 4096 particles moved results 0.02 ft) and 'the posterior genuinely prefers the wrong layer', i.e. an evidence limit. |
| Drift | An explicit slope state is the natural fix for one-straight-line-per-well drift: a tracker with dip momentum extrapolates the heel dip and corrects it as evidence accumulates. Our PF showed the slope prior is the dominant knob (slope walk x9: 28.98 -> 16.82 ft) and that a residual linear drift remained (calibrated -0.0026 ft/ft). The failed posterior_fb_decode.py has NO slope state (zero-mean level kernel), so a dip-aware version is untested. |
| Uncertainty and calibration | Filtered posterior spread is the uncertainty output; DPF training with NLL (MDPF) calibrates the mixture. No reliability / ECE analysis in the surveyed papers. Key for us: fusing N column posteriors assumes conditionally independent emissions; ours are highly correlated (shared receptive field), so the raw product is badly over-confident unless tempered. |

### Relevance to erudition

| Field | Value |
|---|---|
| Where to use | (1) Decode stage after the ConvNeXt / FastViT: treat per-column posteriors p_t(level) as scaled likelihoods (p_t / prior)^tau inside an exact grid HMM over (level, slope), take the expectation of the smoothed marginal. (2) Training-time: a differentiable forward-backward layer on the posterior grid with an RMSE loss on the smoothed mean (DPF-style end-to-end training), learning tau and the transition parameters. (3) As a PF-channel generator that uses the network's posterior (instead of raw GR likelihood) as the measurement model, feeding back into a second-stage network. |
| Error mode targeted | Drift (~25%) via the slope state; band confusion (~51%) only through calibration of how evidence is fused (tempering), not through new evidence. |
| Gap vs our implementation | What we have: a bootstrap PF with hand-set Student-t GR likelihood (rogii_champion/src/particle_filter.py; 3 profiles x 512 particles; state L, r, b; genealogy smoothing), an exact grid HMM on GR (anchor_hmm.py, ~23 ft, not competitive), PF channels to AnchorCNN (5.84 ft), and a post-hoc forward-backward decode over ConvNeXt posteriors that failed (posterior_fb_decode.py). What the DPF literature offers that we do not do: (a) LEARNED emission scaling. posterior_fb_decode.py multiplies the raw per-column softmax outputs directly (exponent 1, no division by the training prior). Because every column posterior already integrates evidence from the whole lateral, multiplying ~345 of them counts the same evidence hundreds of times, making the fused posterior extremely peaked; the smoothed mean then behaves like an arg-max and inherits arg-max's wrong-band losses, exactly the failure its docstring reports ('wins where it locks onto the right band, loses where it locks onto a wrong one'). DPF training (or simply fitting a temperature tau on training wells) is the principled correction; hybrid NN-HMM practice divides posteriors by the class prior to get scaled likelihoods. This knob has not been tried. (b) A slope / dip state in the decode: our fb decode uses a zero-mean level random walk, so it cannot represent the per-well straight-line drift that is ~25% of error and worth 1.9 ft (per-well linear-trend oracle 4.94 -> 3.07 ft on ConvNeXt). (c) End-to-end training through the sequential layer (the network never sees the smoother during training, so it is not trained to produce posteriors that compose). (d) Mode-preserving posterior representations (MDPF mixtures) - low priority because our 1D state is exact on a grid, so particles and differentiable resampling solve a problem we do not have. Confirmation of what we already do: bootstrap PF with fault jumps, slope state, GR bias, backward smoothing, multi-profile blending - all present. |
| Proposed experiment | E1 (no training, 1-2 h, decides whether E2 is worth it): new script kaggle_1st_place/solution/posterior_hmm_decode.py extending posterior_fb_decode.py: state = (level on the 400-level posterior grid, slope r on ~41 bins spanning [-0.05, 0.05] ft/ft); transition: level_t = level_{t-1} + r*dMD - dZ + N(0, s_L), r random walk N(0, s_r) (reuse the S = TVT + Z convention of particle_filter.py); emission = (p_t(level) / pi(level))^tau with pi the training-set marginal of levels (or uniform); initial state from the heel prefix and _initial_rate. Fit tau in {0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0}, s_r, s_L on TRAINING-well posteriors (dump them with seq_NN_posterior_dump.py --save-all on a training subset), then score once on the 155 holdout wells from results/baseline_check_posterior_all (expectation of smoothed marginal vs raw expectation). Also report the blend 'mean of raw expectation and HMM expectation' and the per-well linear-trend residual to see whether drift shrinks. Control: raw expectation decode of the same dump (5.47 ft at 60 epochs; repeat on the 4.84 ft FastViT dump). Accept only if paired bootstrap gain > 0.39 ft or consistent gain on both dumps. E2 (only if E1 shows tau << 1 helps): make the grid HMM a differentiable layer (log-space forward-backward with torch.logsumexp over level and slope; banded transition as a 1D convolution) placed after the U-Net softmax, learnable tau, s_r, s_L; fine-tune the FastViT student 20-30 epochs from its checkpoint with loss = RMSE of smoothed mean + existing per-column loss (keep the per-column KD term so distillation is untouched); control = same fine-tune schedule without the layer. E3 (cheap diagnostic): on holdout wells tagged wrong_mode by the error-anatomy script, check whether the tempered HMM's smoothed marginal puts more mass on the true band than the raw column posterior; if not, stop - the evidence is not there. |
| Implementation effort | Low for E1 (numpy, ~150 lines, reuse posterior_fb_decode.py I/O); medium for E2 (differentiable banded 2D forward-backward in torch, memory care over 345 columns x 400 levels x 41 slopes, plus a fine-tune run); low for E3. |
| Distillation compatibility | E1 is a post-hoc decode, fully compatible (applies to teacher or student posteriors). E2 adds a layer after the softmax; KD on per-column posteriors can stay on the pre-HMM output while the HMM loss acts on the smoothed mean, so the FastViT + KD recipe is preserved. |
| Error correlation with ours | Low diversity: the decode reuses our own posterior, so it corrects our errors rather than adding an independent member; it cannot fix wells where the posterior has no mass on the true band. A GR-likelihood PF (our existing one) is the decorrelated member (residual corr 0.27-0.34) but too weak (15.8 ft) to earn weight. |
| Risk | (1) Metric fact from context: for pooled RMSE the expectation of a calibrated posterior is already Bayes-optimal, so a smoother only helps if the per-column posterior is miscalibrated relative to the path-level posterior or if the slope prior adds information (dip continuity) the network ignores; the prior fb decode failed, and tempering may just converge back to the raw mean as tau -> 0 (a null result is likely). (2) Evidence double counting cannot be removed exactly by a single scalar tau; the right exponent varies with well length and posterior sharpness. (3) Mode-seeking lock-in reappears if tau is too large. (4) E2 end-to-end training is the trained version of B05's neural CRF and overlaps with it; 60-epoch structured experiments (AcfNet target, axial attention, RAFT) all failed here, so expect a small effect within the +-0.39 ft noise floor. (5) DPF papers solve high-dimensional-state / unknown-likelihood problems; our state is 1-2D and our likelihood is already a strong learned network, so most DPF machinery (differentiable resampling, flows) is irrelevant. |
| Notes | Mapping to our stack: the U-Net per-column softmax IS a learned measurement model in DPF terms (PF-net / Jonschkowski style), and 400 discrete levels make the exact grid filter preferable to particles (no resampling variance, trivially differentiable). The distinctive DPF lesson worth importing is to train or calibrate the emission for use inside the sequential model (MDPF: NLL-trained mixtures stay multimodal and stable). Semi-supervised DPF (pseudo-likelihood on unlabelled sequences) suggests a transductive option: use the unlabelled parts of holdout / test laterals to fit tau and slope noise by marginal likelihood without labels - fold-safe because no TVT is used, but only worth trying after E1. Related items: A05 (Bayesian / PF geosteering), B05 (differentiable DP / neural CRF), B09 (calibration), B10 (test-time adaptation on the prefix). |

### Other info

- **Uncertain fields (omitted above)**:
  - code_availability
  - gpu_cost
  - metric_and_performance
  - training_objective

---

## B07 Seismic horizon tracking and deep relative geologic time (Geng 2020, Wu 2020, Bi 2021, transformer RGT 2023, sinusoidal RGT 2026)

### Basics

| Field | Value |
|---|---|
| Name | Seismic horizon tracking via deep relative geologic time (RGT): Geng et al. 2020 (2D CNN RGT), Wu et al. 2020 (realistic synthetic structure models for CNN training), Bi et al. 2021 (Deep RGT, 3D attention U-Net + structural-similarity loss), Yang et al. 2023 (transformer multi-task RGT/horizons/faults with horizon priors), Dou et al. 2026 (RGT-Est, sinusoidal-phase RGT); plus pretrain-to-alignment RGT (2026) |
| Year | 2020 (Geng; Wu), 2021 (Bi), 2023 (Yang), 2026 (Dou sinusoidal RGT, arXiv May 2026; pretrain-to-alignment, arXiv May 2026) |
| Evidence level | Peer-reviewed (Geophysics, JGR Solid Earth, IEEE TGRS) for 2020-2023; arXiv preprints for 2026 work. Field results are mostly qualitative (visual horizon alignment); no well-log / geosteering evaluation. Transfer to ROGII is our analysis. |
| Extreme summary | Instead of tracking each horizon, a network regresses one scalar field (relative geologic time) over the whole seismic image; every horizon is an iso-contour of that field, so all horizons are mutually consistent (non-crossing, ordered) by construction, and faults appear as RGT discontinuities. Later work adds structure-aware losses (SSIM, perceptual, adversarial), sparse horizon priors, and multi-frequency sinusoidal encodings of RGT that make closely spaced, similar-looking layers distinguishable. |

### Problem formulation

| Field | Value |
|---|---|
| Task definition | From a 2D seismic section or 3D seismic volume, predict at every sample the relative geologic time (stratigraphic age / depositional order), then extract horizons as RGT iso-surfaces and faults as lateral RGT discontinuities; optional sparse interpreted horizons as constraints. |
| Inputs | Seismic amplitude image / volume; optionally sparse horizon picks (2D lines or 3D surfaces) as prior channels (Yang 2023, Dou 2026); fault labels in multi-task setups. ROGII analogue: the lateral-vs-typewell GR cost image over (MD column x TVT level), with the heel prefix as the sparse 'horizon pick'. |
| Target parameterisation | Relative geologic time: a scalar field monotonically increasing with depth along each vertical trace (monotonicity not hard-enforced by the networks); Dou 2026 maps the scalar RGT r in [-1, 1] to three sinusoidal channels sin/cos-style at frequencies 2.0, 1.0, 0.5 and optimises in that phase space as well as on the scalar. ROGII analogue: TVT along the lateral is exactly an RGT sample (stratigraphic depth of the bit), and S = TVT + Z is the structural elevation of a reference horizon at each MD. |
| Output representation | Dense scalar field (point regression per pixel/voxel); horizons derived as iso-contours; no per-pixel distribution or multiple hypotheses. |

### Method

| Field | Value |
|---|---|
| Architecture | Geng 2020: 2D ResNet-50 encoder + U-Net-style decoder + refinement module. Bi 2021: simplified 3D U-shaped encoder-decoder with multi-scale residual learning and attention (channel/spatial highlighting), plus a sub-volume-then-merge workflow to handle large volumes without boundary artefacts. Yang 2023: transformer-based multi-task network with shared features and two mutually constraining branches (RGT, faults), lighter than CNNs with larger structure-oriented receptive field, horizon priors as input. Dou 2026: 3D HRNet (parallel multi-resolution branches), optional sparse 2D/3D horizon guidance channel, output RGT in [-1, 1] then sinusoidal transform, 3D PatchGAN discriminator conditioned on seismic. |
| Core mechanism | (1) Global scalar-field parameterisation: predicting one field whose level sets are the horizons couples all horizons and all traces, giving built-in stratigraphic consistency and fault-crossing correlation without explicit tracking. (2) Structure-aware losses (Bi's structural-similarity loss; Dou's multi-directional 3D LPIPS and adversarial loss) supervise spatial relationships rather than independent pixels, countering over-smoothing. (3) Dou's multi-frequency sinusoidal phase targets amplify small RGT differences (reported ~62x larger L1 gradients) so thin, similar layers get distinct codes while the low-frequency channel removes wrap-around ambiguity. (4) Sparse horizon priors pin the field where interpretations exist. (5) Heavy synthetic training from forward-modelled folded/faulted structure models (Wu 2020 generator line). |
| Decoding | Direct regression output; horizons = iso-contours of the predicted RGT; faults = lateral discontinuities of RGT. |
| Metric consistent decode | Scalar MSE regression estimates the conditional mean (RMSE-consistent); SSIM/LPIPS and especially the adversarial term are distribution-matching / mode-seeking and favour sharp single hypotheses, which would raise pooled RMSE under band ambiguity. |
| Evidence scope | Whole 2D section or 3D volume (large receptive field; sub-volume stitching in Bi 2021); single survey, no multi-survey joint inference. |
| Prefix conditioning | Sparse interpreted horizons as extra input channels (Yang 2023; Dou 2026 shows 2D/3D horizon constraints significantly improve precision and lateral consistency). Direct analogue of our known heel TVT prefix. |
| Propose vs select | Single output field; no candidate generation. Could be an auxiliary head or target reparameterisation rather than a separate ensemble member. |
| Physical constraints | Consistency and ordering of horizons implied by the scalar-field formulation; monotonicity with depth is NOT hard-enforced (Dou 2026 lists lack of hard monotonicity / non-intersection constraints as a limitation); faults as discontinuities; SSIM/LPIPS/adversarial terms as soft structural priors. |

### Data

| Field | Value |
|---|---|
| Synthetic to real | All core methods train purely on synthetic data generated from structural models (Wu 2020 line: flat layers + folding + faulting + wavelet convolution + noise) and apply zero-shot to field data. The 2026 pretrain-to-alignment paper argues this leaves a synthetic-to-field gap and proposes four stages: self-supervised pretraining on field data, synthetic supervision, prior-driven refinement, and domain-adaptation fine-tuning on scarce field labels; jiarunyangustc/rgt_finetune fine-tunes a pretrained RGT net on field data with multi-source constraints. |

### Results

| Field | Value |
|---|---|
| Comparability | Not comparable: seismic RGT fields vs well-log TVT RMSE on 155 holdout wells; no shared metric. |
| Validation protocol | Synthetic held-out sets plus visual checks on field surveys; no spatially grouped CV concept. Weak quantitative evidence for field gains. |
| Inference budget | 3D volumes at 512^3 need large GPUs, but the ROGII analogue is a 1D path per well or our existing 345 x 400 image, so any transplanted head / loss adds negligible inference cost. |

### Ambiguity handling

| Field | Value |
|---|---|
| Repeated layers | Handled in principle by global context (the whole section constrains which reflector is which) and, in Dou 2026, by multi-frequency phase codes: high frequency separates thin layers locally, low frequency tells which cycle, avoiding wrap-around. This mirrors our band confusion: GR repeats with a quasi-period, so 'position within the cycle' is easy and 'which cycle' is the hard part. None of these works outputs a distribution, so ambiguity is resolved by regression toward the mean (or sharpened by adversarial loss). |
| Drift | RGT fields are driven by global structural consistency and by sparse horizon priors; smooth structural dip is implicit in the synthetic folding models. The ROGII analogue of drift is a wrong dip of the structural surface S(MD); a gradient-consistency view of RGT suggests supervising the derivative dS/dMD (dip) directly, which is not done in our pipeline (see gap). |
| Uncertainty and calibration | None: point regression without uncertainty or calibration analysis; limitations sections note unreliable predictions where reflectors are obscured. |

### Relevance to erudition

| Field | Value |
|---|---|
| Where to use | (1) Loss / target design in the ConvNeXt / FastViT U-Net: an RGT-style supervised dip (gradient) consistency loss on the expected path's structural surface S = E[TVT] + Z over multiple MD windows; and an auxiliary multi-frequency sinusoidal head (sin/cos of 2*pi*TVT/P_k) sharing the decoder. (2) Training schedule: a pretrain-to-alignment style final stage on real wells only after the 85%-synthetic phase. (3) Conceptual: treat the heel prefix like sparse horizon priors (already done via tvt_input channels) and neighbour wells like a multi-trace RGT field (belongs to A12). |
| Error mode targeted | Drift (~25%) via dip-consistency supervision and real-data alignment; band confusion (~51%) weakly via multi-frequency auxiliary targets (feature learning) and via synthetic-to-real alignment. |
| Proposed experiment | E1 (lowest effort, targets drift): add a supervised dip-consistency loss 'dS_match' next to dS_penalty in experiments/arch/seq_NN_train.py: S_pred = E_p[TVT] + z_rel, S_true = TVT + z_rel; loss = mean over windows w in (7, 15, 31, 63) of masked L1 of (S_pred[t+w] - S_pred[t]) - (S_true[t+w] - S_true[t]), scaled by surface_stats; weight sweep {0.01, 0.03} x the alignment weight. Config: new key in unet_loss_weights in experiments/bilzard/seq_NN_cfg.py. Run first as a cheap 60-epoch ConvNeXt screen against the 5.47 ft baseline with the same seed, then 150-epoch teacher -> FastViT-SA12 distillation vs the 4.84 ft student. Metrics: pooled RMSE via seq_NN_holdout_eval.py, paired bootstrap via seq_NN_robust_compare.py, AND the per-well linear-trend residual decomposition (does the drift share fall?). Also run a control with the existing dS_penalty switched on (weight 0.01) to separate unsupervised smoothness from supervised dip matching. E2 (targets synthetic-to-real gap): pretrain-to-alignment stage - after the 150-epoch mixed run, continue 15-20 epochs on real training wells only (synthetic ratio 0), LR x0.3, z_shift off (use the existing finetune_* machinery if it can be restricted to real wells); control: 15-20 extra epochs of the same mixed schedule, so epoch count is matched. E3 (targets band confusion via features): auxiliary head of 2K channels predicting sin/cos(2*pi*TVT/P_k), P_k in {8, 32, 128} ft (or P set from the typewell GR autocorrelation period per well), L1 loss weight 0.02, decode unchanged (expectation of the softmax); measure wrong_mode share and pooled RMSE vs control. Run E1 and E2 before E3. |
| Implementation effort | Low for E1 (about 30 lines beside masked_surface_curvature_l1_loss, one config key); low-medium for E2 (dataset filter for real-only wells plus schedule; check what the finetune path does); medium for E3 (new head, target construction, loss, KD plumbing for the extra head). |
| Distillation compatibility | Yes. E1 and E3 are training losses usable for teacher and student (the student can take dS_match on its own expected path alongside KD on posteriors); E2 applies to the teacher, the student, or both. Nothing changes the output posterior format. |
| Error correlation with ours | Low diversity: these are modifications of our own model, not an independent member; a dS_match-trained model may however drift differently (smaller per-well slope bias) and so could lower the best-pair RMSE if its residual correlation with the current student drops. |
| Risk | (1) Dip supervision may just trade drift for more wrong-band blending: the expectation path under a bimodal posterior has a spurious slope when mass shifts between bands, and the loss would push mass shifts to be smoother rather than correct. (2) Per-column CE on alignment targets already supervises the path densely; the extra term may be redundant and inside the +-0.39 ft noise floor (60-epoch structural changes AcfNet target, axial attention, RAFT all failed). (3) Real-only fine-tuning risks overfitting to ~600 real wells and losing the synthetic-induced robustness; spatial leakage between neighbouring real wells can inflate apparent gains. (4) Sinusoidal auxiliary targets encode position, not evidence: band confusion falls only with more evidence or better calibration (context metric fact), so E3 helps only if it improves features. (5) Seismic RGT evidence is mostly qualitative, so there are no hard numbers to expect a gain size from. |
| Notes | Strong conceptual analogy: ROGII TVT is an RGT value sampled along the well path, and our cost image in (MD, TVT) coordinates is effectively a flattened (RGT-domain) section. The 'monotone RGT-like output parameterisation' in the outline's why-field does not transfer literally - TVT along a lateral is not monotone because the bit moves up and down through strata; the transferable invariant is smoothness of the structural surface S = TVT + Z (dip), which is what E1 supervises. Repeating reflectors in seismic are handled by global context rather than a special mechanism; Dou 2026's multi-frequency phase encoding is the closest to an explicit repeated-layer device. A multi-well RGT field (all laterals and typewells in an area mapped into one geologic-time field) is the principled version of the failed XY geo-prior and belongs to A12. Pretrain-to-alignment also informs B11 (synthetic data / sim-to-real). |

### Other info

- **Uncertain fields (omitted above)**:
  - authors_org
  - code_availability
  - dataset
  - gap_vs_our_implementation
  - gpu_cost
  - metric_and_performance
  - training_objective
  - venue_link

---

## B08 Retinal-layer segmentation with topology constraints (He et al. 2021 MedIA column-softmax + soft-argmax, graph-search CNNs)

### Basics

| Field | Value |
|---|---|
| Name | Retinal OCT layer-surface segmentation with column-wise distributions and topology/smoothness constraints: He et al. 2021 FCRN (column softmax + soft-argmax + topology module), He et al. 2018/2019 topology-guaranteed S-Net/R-Net and boundary regression, graph-search CNN hybrids (CNN-GS), Xie/Wu constrained IPM (2022) and differentiable DP (DDP, 2022) layers |
| Authors org | Y. He, A. Carass, Y. Liu, B. Jedynak, S. Solomon, S. Saidha, P. Calabresi, J. Prince (Johns Hopkins IACL); H. Xie, Z. Pan, W. Xu, X. Wu et al. (University of Iowa); CNN-GS: L. Fang, S. Farsiu et al. (Duke) |
| Year | 2017 (CNN-GS); 2018 (topology-guaranteed S-Net/R-Net, arXiv); 2019 (FCBR, MICCAI); 2021 (FCRN, MedIA vol. 68, 101856); 2022 (constrained IPM, Optics Express 30(2); DDP, arXiv 2210.06335) |
| Venue link | FCRN: Medical Image Analysis 68:101856, doi:10.1016/j.media.2020.101856 (https://pubmed.ncbi.nlm.nih.gov/33260113/, PMC7855873). FCBR: MICCAI 2019 (https://link.springer.com/chapter/10.1007/978-3-030-32239-7_14). Topology-guaranteed CNN: arXiv:1803.05120 (https://arxiv.org/abs/1803.05120). DDP: arXiv:2210.06335 (https://arxiv.org/abs/2210.06335). Constrained IPM: Optics Express 30(2):2453-2471, 2022 (https://pubmed.ncbi.nlm.nih.gov/35209385/); mutually interacting surfaces: arXiv:2007.01259. |
| Evidence level | Peer-reviewed (MedIA 2021, MICCAI 2019, Optics Express 2022); DDP is an arXiv preprint. DDP full text read (method and tables extracted). FCRN read at abstract and secondary-description level (PMC full text was behind a bot check), so its exact topology-module formula and loss weights come from secondary descriptions. |
| Extreme summary | Retinal-layer networks output, for every A-scan column, a softmax over depth per surface and read the position with a differentiable soft-argmax (exactly our per-column softmax + expectation decode), then enforce anatomy inside the network: ordering between surfaces via non-negative ReLU increments and inter-column smoothness via a constrained differentiable DP or IPM layer. Reported gains of the in-network smoothness layer over plain soft-argmax are tiny on MASD (2.77 to 2.75 um, 1.90 to 1.88 um) and larger on Hausdorff/outlier metrics. |

### Problem formulation

| Field | Value |
|---|---|
| Task definition | For each column x of an OCT B-scan, predict the depth z_x(i) of N retinal surfaces (N=3 Duke AMD, N=9 JHU MS) from the image, with ordering z(i) <= z(i+1) and bounded column-to-column change. |
| Inputs | The OCT B-scan intensity image (plus 7 image-gradient channels in the Iowa IPM/DDP models). No neighbour images, no known prefix. Our analogue: the 16-channel 345 MD-column x 400 TVT-level cost image. |
| Target parameterisation | Absolute depth per column per surface. He 2018 R-Net: non-negative layer thicknesses cumulatively summed into surfaces. IPM: per-column Gaussian (mu, sigma) surface cost. Our analogue: TVT offset per MD column relative to the anchor. |
| Output representation | Per-column, per-surface softmax over depth; point estimate by soft-argmax (expectation); after the topology module or DP layer, one smooth surface per layer (a path). No multi-hypothesis output. |

### Method

| Field | Value |
|---|---|
| Core mechanism | (1) Treat surface localisation as per-column classification over depth and decode with the expectation: fully differentiable, sub-pixel, and supervisable with both CE on the distribution and L1 on the expectation (identical to our alignment CE + expectation loss). (2) Enforce structure by construction inside the forward pass instead of post hoc: ordering by ReLU-positive increments (topology can never be violated), smoothness by a hard slope bound inside a DP layer whose soft max/argmax makes it trainable end-to-end, so feature learning receives the gradient of the structured output. |
| Decoding | Soft-argmax (expectation) per column, then topology module (FCRN), constrained soft DP with soft-argmax backtracking (DDP), or IPM QP (constrained least squares around the column means). |
| Metric consistent decode | Soft-argmax alone is the RMSE-consistent mean. The structured stages are not: DDP/IPM build a unimodal quadratic cost -(z-mu)^2 around the column mean, discarding multimodality, then solve for a max-score path, i.e. a projection of the mean sequence onto a smoothness set. For band-confusion columns this is neither the posterior mean nor mode-aware; it only smooths the mean curve. |
| Evidence scope | Whole B-scan (all columns of one image) jointly; single image, no multi-image evidence. |
| Prefix conditioning | None; the DP start is unconstrained. For us the known heel TVT would be a hard DP start state (all mass at the known TVT in the last known column). |
| Propose vs select | Single output. Not a candidate generator, although the soft (LogSumExp) forward pass of the DP could be sampled to give path hypotheses. |
| Physical constraints | Surface ordering (non-crossing) and bounded inter-column slope /z_x - z_(x-1)/ <= delta; both hard. |

### Data

| Field | Value |
|---|---|
| Dataset | JHU MS: 35 Spectralis macular scans (14 healthy, 21 MS), 49 B-scans of 128 x 1024 each, 9 surfaces. Duke DME (10 subjects). Duke AMD: 384 volumes (115 normal, 269 AMD), 51 B-scans of 512 x 361 around the fovea, 3 surfaces. |
| Synthetic to real | Not applicable; all real, manually traced data; augmentation only (noise, flips). |

### Results

| Field | Value |
|---|---|
| Metric and performance | JHU MS overall MASD: FCRN 2.83 um (as re-reported in the DDP paper), IPM 2.78, DDP 2.75, DDP-less ablation 2.77 um; He 2018 S-Net/R-Net 2.82 um vs graph-based AURA 2.83 um. Duke AMD: DDP total MASD 1.88 um vs same net without DDP 1.90 um and FCRN re-implementation 2.78 um; 95% HD Inter-RPEDC 6.53 (DDP) vs 6.59 (no DDP) vs 9.73 um (FCRN). DDP ablation on JHU: average-HD gains 1.5-16% per surface; authors note visible over-smoothing on some AMD cases. |
| Comparability | Not numerically comparable (different domain, sub-pixel errors on clean images, no repeated structures). Qualitatively comparable: same output formulation; ablations show in-network smoothness mainly helps outliers/weak-evidence regions, analogous to our band-confusion regime but in a far less ambiguous problem. |
| Validation protocol | Fixed subject-level splits (JHU MS: train on last 6 HC + 9 MS by ID, test on the other 20; Duke AMD random 266/59/59 subjects). Single split, no seed variance reported; p-values computed over columns overstate significance. |
| Inference budget | Forward pass plus O(X * Z * delta) DP per surface; for us 345 columns x 400 levels x (2*delta+1) is milliseconds on GPU or CPU. Fits Kaggle/rig budget easily. |

### Ambiguity handling

| Field | Value |
|---|---|
| Repeated layers | Not addressed: retinal surfaces are distinct and ordered, so column posteriors are near-unimodal. The ordering trick depends on knowing which surface is which (fixed identity), which is precisely what is ambiguous for us. A slope-bounded DP can suppress an isolated wrong-band excursion shorter than the jump the slope bound forbids, but cannot fix a lateral that sits in the wrong band for its whole length. Reducing each column to a quadratic around its mean (DDP/IPM) would lock in blends instead of choosing a band. |
| Drift | Not addressed. The He 2018 increment parameterisation (cumulative sum of predicted non-negative increments) is the analogue of predicting per-column dTVT and integrating from the heel, which would worsen drift through error accumulation unless tied back by absolute per-column evidence. |
| Uncertainty and calibration | None reported; column distributions serve only as a differentiable route to the expectation. No reliability analysis. |

### Relevance to erudition

| Field | Value |
|---|---|
| Where to use | (a) Loss/decoder level in experiments/arch/seq_NN_train.py and seq_NN_models.py: a trained, soft, slope-bounded DP (forward-backward) layer over the per-column log-posteriors with the known heel TVT as start state, i.e. the end-to-end version of the failed post-hoc continuity decode. (b) Post-hoc variant on saved posteriors already exists (posterior_fb_decode.py). (c) The CE + expectation-regression supervision is already in our loss; nothing to add there. |
| Error mode targeted | Band confusion in short excursions (blend / wrong_mode segments shorter than the lateral) and broad-hump noise; not drift. |
| Gap vs our implementation | Already done: per-column softmax over 400 TVT levels + expectation decode (He's soft-argmax); smoothed alignment CE targets (alignment_exp_smooth_sigma 1.25, Laplace targets) + expectation regression loss (He's CE + smooth-L1); post-hoc forward-backward continuity decode (posterior_fb_decode.py, negative result on 155 holdout wells: smoothed 5.78 and forward 5.70 vs mean 5.47 ft, only 67/155 wells improved). Not done: (1) structure inside training, i.e. a differentiable slope-bounded DP whose output is supervised so the backbone learns column posteriors that are consistent along MD (DDP style, pretrain then fine-tune). Our continuity decode may have failed partly because the U-Net posterior was never trained to be consumed by a path model. (2) A hard prefix start state inside the network. (3) Ordering/topology between multiple surfaces has no direct analogue: we predict one surface (bit TVT); the only analogue would be jointly predicting several typewell marker horizons with non-crossing constraints, which the target does not require. Caution: DDP/IPM use a cost -(z - mu)^2 that collapses each column to its mean and is useless for band confusion; an adaptation must use the full log-posterior as emission, which makes it the B05 differentiable-alignment item rather than a new idea. |
| Proposed experiment | Low priority (subsumed by B05; the post-hoc version already lost). If run: E1 (training, one 150-epoch ConvNeXt run + control). Add aux head 'ddp' in experiments/arch/seq_NN_train.py: emissions = log_softmax of column logits (not the quadratic cost); transition = log-Gaussian in dTVT per 32 ft column, band-limited to /dTVT/ <= 3x training max, sigma either fixed from training wells or predicted per well by a small MLP on trajectory features; start = delta at known heel TVT; soft forward-backward in log space giving smoothed marginals. Loss = L2 of the smoothed-marginal expectation vs truth on unknown columns, weight 0.1-0.3 as a new unet_loss_weights key (config in seq_NN_cfg.py); enable only after 60% of epochs (DDP paper's pretrain-then-finetune). Evaluate with seq_NN_holdout_eval.py: (i) unchanged per-column expectation (does the aux loss improve the backbone?), (ii) smoothed-marginal expectation, (iii) 50/50 blend. Two seeds per arm; accept only if pooled-RMSE gain on the 155 holdout wells exceeds the +-0.39 ft noise floor. Control: same config, aux weight 0. |
| Implementation effort | Medium: banded soft forward-backward over 345 x 400 is ~100 lines of PyTorch; log-space stability and masking of variable suffix lengths need care. |
| Distillation compatibility | Compatible: the DP is parameter-light; the FastViT student can reuse it, and KD can target raw column posteriors (as today) or DP-smoothed marginals. |
| Error correlation with ours | High: same backbone and evidence; the DP only redistributes mass along MD. Not a useful independent blend member. |
| Risk | (1) The post-hoc forward-backward decode already lost (5.70-5.78 vs 5.47 ft), indicating the Markov smoothness prior conflicts with how errors arise (whole-lateral band commitment, not isolated excursions). (2) Retinal ablations show only ~0.02 um MASD gain: in-network smoothness fixes rare outliers in weak-contrast regions. (3) Mean-collapsing costs (DDP/IPM quadratic) would harden blends. (4) Over-smoothing reported by the DDP authors; laterals have real faults and dip changes where a slope bound is wrong. (5) Topology/ordering has no target in a single-surface problem. |
| Notes | Main value is confirmation: our formulation is the retinal FCRN formulation, and the retinal community's extra step (structure in the network) is the B05 differentiable DP idea, with small reported gains in a much less ambiguous domain. The one transferable insight is to train the posterior jointly with the path model (pretrain, then fine-tune) instead of smoothing post hoc. He 2018's ReLU-increment construction is a clean way to encode hard monotone structure if a monotone quantity (e.g. relative geologic time) is ever predicted (see B07). |

### Other info

- **Uncertain fields (omitted above)**:
  - architecture
  - code_availability
  - gpu_cost
  - training_objective

---

## B09 Calibration and conformal prediction for histogram/classification-as-regression outputs (temperature/isotonic, conformal histogram regression, multimodal conformal, conformal under HMM)

### Basics

| Field | Value |
|---|---|
| Name | Calibration and conformal prediction for histogram / classification-as-regression outputs: temperature scaling (Guo 2017), quantile/isotonic recalibration (Kuleshov 2018), distribution calibration (Song 2019), conformal histogram regression CHR (Sesia & Romano 2021), probabilistic/multimodal conformal PCP (Wang et al. 2023), conformal prediction under HMMs and dependent data (2022-2024), plus an empirical calibration audit of Erudition's 155-well holdout posteriors |
| Authors org | Guo, Pleiss, Sun, Weinberger (Cornell); Kuleshov, Fenner, Ermon (Stanford); Song, Diethe, Kull, Flach (Bristol); Sesia (USC) & Romano (Technion); Wang, Gao, Yin, Zhou, Blei (Columbia) for PCP; Su, Zhou, Luo (HMM particle-filter ACI); Oliveira et al. (split conformal for dependent data); audit: this project. |
| Year | 2017 (temperature scaling); 2018 (calibrated regression); 2019 (distribution calibration); 2021 (CHR); 2022-2023 (PCP, conformal for dependent data / HMM via Markov de Finetti); 2024 (ACI by particle filtering under HMMs); audit 2026-09-28 |
| Code availability | CHR: github.com/msesia/chr (Python). Temperature scaling / isotonic: a few lines (sklearn IsotonicRegression). Kuleshov tutorial: github.com/AnthonyRentsch/calibrated_regression. Audit scripts for our posteriors: scratchpad calib.py / modecal.py (numpy only, reproducible on posterior_all.npz); not committed. |
| Evidence level | Peer-reviewed methods (ICML/NeurIPS). Our own audit on 155 holdout wells (23,634 columns, 60-epoch baseline dump) is a direct measurement; out-of-sample recalibration numbers use 2-fold well-grouped CV within the holdout set (fitting on training wells would be the proper protocol). |
| Extreme summary | Post-hoc calibration (temperature, isotonic/quantile, mode-mass Platt) and conformal wrappers (CHR, PCP, well-level split conformal) can make histogram posteriors reliable and give coverage-guaranteed intervals, but for pooled RMSE under the mean decode they only help if the posterior mean is mis-calibrated. Our audit shows the 60-epoch ConvNeXt posterior is already mean- and variance-calibrated (slope of truth on mean 0.993, predicted variance 30.5 vs actual MSE 29.9 ft^2, NLL minimal at T=1.0): temperature scaling does not help out of sample (5.52 vs 5.47 ft) and CV mode-mass recalibration gains 0.017 ft, far below the 0.39 ft noise floor. The useful product is per-well confidence: posterior spread ranks well RMSE with Spearman 0.64 and well-level split conformal gives valid 90% coverage. |

### Problem formulation

| Field | Value |
|---|---|
| Task definition | Given a model's predictive distribution p(y/x) (here a 400-bin histogram over TVT per MD column), (a) recalibrate it so that probabilities match frequencies, and/or (b) produce prediction sets with finite-sample coverage guarantees; downstream (c) flag low-confidence wells. |
| Inputs | Model posterior histograms (our per-column 400-bin softmax), a held-out calibration set of wells with truth, optionally per-well covariates (entropy, number of significant modes, well length). |
| Target parameterisation | Unchanged (TVT offset per column). Calibration acts on the output distribution only. |
| Output representation | Recalibrated per-column histogram; conformal interval or union of intervals (CHR gives shortest interval from the histogram; PCP gives unions of balls, i.e. multimodal sets); per-well predicted RMSE with conformal upper bound. |

### Method

| Field | Value |
|---|---|
| Architecture | Post-hoc, no network change. Temperature: p_T proportional to p^(1/T), one scalar. Isotonic/quantile recalibration (Kuleshov): monotone map on the predicted CDF value (PIT) fitted on calibration data. Distribution calibration (Song): GP with Beta link mapping predicted to calibrated distributions. Mode-mass calibration (our variant): per-column basins from posterior_analysis.find_modes, dominant-mode mass recalibrated by Platt a*logit+b, other modes rescaled. CHR: nested histogram-based intervals conformalised on a calibration set. PCP: sample K points from the predictive, the set is a union of balls with conformal radius. HMM/dependent conformal: exchangeability restored at the level of blocks/chains or via online adaptive conformal with particle-filter states. |
| Core mechanism | Under pooled squared loss the Bayes decode is the posterior mean, and RMSE depends on the posterior only through its mean. So calibration lowers RMSE exactly when E[y / mu_hat] != mu_hat, e.g. systematic over/under-confidence of dominant band weights that biases the mean toward the wrong band. Temperature and isotonic maps change the mean only by reshaping multimodal columns (moving weight between humps); for unimodal symmetric columns they change intervals but not the mean. Conformal methods guarantee marginal coverage under exchangeability (valid across wells if wells are exchangeable) but do not change the point prediction. |
| Training objective | Post-hoc: NLL (temperature), isotonic least squares on PIT (Kuleshov), GP marginal likelihood (Song), or directly the SSE of the recalibrated mean (what matters for us; our mode-mass Platt fit minimised SSE). Conformal: quantile of nonconformity scores on the calibration set; no training. In-training alternatives (not reviewed in depth): label smoothing, focal loss, mixup; our targets are already smoothed (alignment_exp_smooth_sigma 1.25). |
| Decoding | Keep the expectation of the recalibrated histogram (RMSE-consistent). Conformal sets are for reporting/flagging, not for the point estimate. |
| Metric consistent decode | Yes if the expectation of the (re)calibrated posterior is used. Anything that uses calibration to justify picking the dominant mode is mode-seeking and raised RMSE in our repo (argmax 6.26, dominant mode 6.06 vs mean 5.47 ft). |
| Evidence scope | Per column (temperature, isotonic, mode mass); per well for conformal flags (well-level exchangeability). |
| Prefix conditioning | Not intrinsic. A per-well temperature fitted on the known heel prefix would be a test-time calibration (B10 territory) but the prefix posterior is conditioned on known labels and not representative of the suffix. |
| Propose vs select | Neither; it reweights an existing posterior. Can calibrate the inputs of a gate or blend (calibrated posteriors blend better). |
| Physical constraints | None. |

### Data

| Field | Value |
|---|---|
| Dataset | Literature: UCI regression, time series, CIFAR/ImageNet (temperature), synthetic/skewed regression (CHR). Our audit: 155 holdout wells, 23,634 MD columns (32 rows each), posterior_all.npz from the 60-epoch ConvNeXt baseline (seq_NN_posterior_dump.py --save-all). |
| Synthetic to real | Relevant: 85% synthetic training could miscalibrate real-well posteriors, but the audit on real holdout wells shows no material miscalibration of mean or variance, so the synthetic mix has not produced a systematic confidence bias at the 60-epoch checkpoint. |

### Results

| Field | Value |
|---|---|
| Metric and performance | Audit on 155 holdout wells (column-level pooled RMSE, mean decode 5.469 ft): (1) mean calibration: weighted regression truth = -0.19 + 0.993 * mu. (2) Variance: mean predicted variance 30.5 ft^2 vs actual MSE 29.9 ft^2; z-score std 1.17 (heavy tails). (3) Shape: PIT deciles 0.078/0.060/0.061/0.074/0.101/0.128/0.143/0.136/0.112/0.106; central 50% interval covers 64% (core too wide), 80% covers 84%, 90% covers 90.3%, 95% covers 93.8% (tails too thin), mean 90% width 13.2 ft. (4) Temperature (in-sample): T=0.8/0.9/1.0/1.1/1.25 give RMSE 5.472/5.466/5.469/5.479/5.506 and NLL minimum at T=1.0; well-grouped 2-fold CV temperature (fit on SSE) 5.517 +- 0.049 ft, i.e. worse. (5) Mode-mass reliability (3,730 multimodal columns): dominant-mass bins 0.55/0.65/0.75/0.86/0.93 have truth-under-dominant rates 0.59/0.70/0.81/0.76/0.85 (roughly calibrated, slightly over-confident above 0.8). CV Platt recalibration of mode masses (sharpening, a~1.4-1.6): 5.463 vs 5.480 ft (unweighted column RMSE), gain 0.017 ft; in-sample optimum 5.452 ft. (6) Per-well confidence: Spearman(posterior sd, actual well RMSE) = 0.64 (p=5e-19); the 10 wells with the largest predicted sd hold 33% of well-level SSE (oracle top-10: 46%); well-level split conformal on ratio actual RMSE / predicted sd, 90% level: empirical coverage 0.91 over 200 random 77/78 splits, threshold ratio 1.65 (median ratio 0.73). Literature: CHR attains nominal coverage with shorter intervals than CQR on skewed data; Kuleshov recalibration gives calibrated intervals and improves forecasting/RL downstream. |
| Comparability | The audit is directly comparable (same 155 holdout wells, same metric), with two caveats: column-level truth is the mean of 32 rows (so 5.469 vs row-level 5.474 ft), and it is the 60-epoch baseline, not the 4.84 ft FastViT or 4.69 ft blend, which should be re-audited. |
| Validation protocol | CV within holdout wells, grouped by well, repeated random 2-fold (20 repeats for temperature, 10 for mode-mass Platt). Proper deployment would fit calibrators on training/OOF wells and evaluate once on the holdout; with synthetic-heavy training that requires real-well OOF posteriors. |
| Inference budget | Negligible (one power and renormalisation per column, or a 1-D isotonic lookup; conformal is a quantile lookup). |

### Ambiguity handling

| Field | Value |
|---|---|
| Repeated layers | Calibration can only redistribute weight between humps that already exist; it cannot identify the correct band. The audit shows dominant-band weights are roughly calibrated, so the 51% of SSE from band confusion is irreducible by recalibration of this posterior; it needs more evidence (inputs, neighbour wells, path-level evidence) or a better model. Multimodal conformal (PCP, CHR with disjoint sets) can express 'band A or band B' as a two-piece set for reporting. |
| Drift | Not addressed: a hump sliding away from truth is a mean error that calibration cannot correct unless it is systematic across wells (the regression intercept of -0.19 ft and slope 0.993 show it is not). |
| Uncertainty and calibration | Core topic. Findings: well calibrated in mean and variance; mild shape miscalibration (under-dispersed core, heavy tails, PIT hump at deciles 5-8); per-well spread is a useful ranking signal and supports conformal flagging with valid coverage. Ensembles/blends (averaging probabilities of two models) typically become under-confident and multimodal averages shift weight, so blended posteriors need a separate audit. |

### Relevance to erudition

| Field | Value |
|---|---|
| Where to use | (1) Diagnostic gate in the evaluation pipeline (new script next to posterior_analysis.py, e.g. posterior_calibration.py): run on every new posterior dump and on blends before trusting a blend weight. (2) Blend-time calibration: temperature per member before probability averaging in the pair/ensemble blend (ConvNeXt + FastViT or ConvNeXt + AnchorCNN), fitted on OOF training wells. (3) Per-well confidence flag and conformal bound for reporting / rig use (not for RMSE). |
| Error mode targeted | Measurement/protocol and blend quality; marginally 'blend' (15%) and 'wrong_mode' (24%) via mode-mass reweighting, but the audit shows little recoverable error there. |
| Gap vs our implementation | Not done before: no calibration audit existed; now measured (see metric_and_performance). Confirmed not a gap for single-model RMSE: temperature and mode-mass recalibration give <= 0.02 ft out of sample on the 60-epoch baseline, and the posterior mean is essentially unbiased. Remaining gaps: (1) the audit has not been run on the deployed FastViT-SA12 distilled student (4.84 ft) or on the 4.69 ft pair; KD with a temperature can change calibration, and probability averaging of two models changes dispersion, so per-member temperature before averaging is untested. (2) No per-well confidence output: posterior sd predicts well RMSE (Spearman 0.64) and a well-level conformal bound is valid at 90%, which could flag the worst wells (the worst 10 wells carry ~48% of SSE) for fallback to a different model or for human review. (3) Heavy tails (z std 1.17, 95% interval covers 93.8%) mean interval reporting should use conformal quantiles rather than raw posterior quantiles. |
| Proposed experiment | E0 (done, CPU, ~2 min): calibration audit on baseline_check_posterior_all (numbers above). E1 (CPU, <1 h): rerun the audit on posterior dumps of the FastViT-SA12 student and the ConvNeXt teacher (seq_NN_posterior_dump.py --save-all for each) and on their probability average; report mean-calibration slope, var ratio, PIT, NLL-vs-T. If the average is under-confident (var ratio > 1.1 or NLL minimum at T < 0.9), E2: fit a temperature per member on training-well OOF posteriors (or 2-fold within holdout as a quick check), average tempered posteriors, decode by expectation; compare with the current blend at matched weight on the 155 wells; accept only if gain > 0.39 ft (expected gain is small; mainly a sanity check). E3 (confidence flag): compute per-well predicted RMSE = sqrt(mean posterior variance over suffix columns); fit the 90% conformal ratio threshold on training wells; on holdout report coverage, and test a routing rule 'if predicted well RMSE > q, use the other model / blend weight 0.5' with the threshold chosen on training wells only. Files: new kaggle_1st_place/solution/posterior_calibration.py (reuse posterior_analysis.softmax/level_axis and the npz loader from posterior_fb_decode.py); no training changes. |
| Implementation effort | Low: numpy-only post-hoc analysis; the audit script already exists in prototype form. |
| Gpu cost | None beyond one inference dump per model (minutes on the RTX 5060); calibration fitting is CPU-only. |
| Distillation compatibility | Compatible and relevant: KD temperature and teacher calibration shape the student's posterior; auditing the student and, if needed, tempering the teacher targets during KD is a zero-cost check. |
| Error correlation with ours | Not a blend member; it is a monotone transform of our own posterior and so perfectly correlated. Its value is in making blends and gates better conditioned. |
| Risk | (1) Measured: the posterior is already calibrated, so expected RMSE gain for a single model is ~0.01-0.02 ft, well below the 0.39 ft noise floor. (2) Fitting calibrators on the holdout itself leaks; with only 618 real training wells and 85% synthetic training data, OOF real-well posteriors are needed for a clean fit. (3) Conformal validity assumes wells are exchangeable between calibration and test; spatial clustering of wells (and Kaggle private test) can break this. (4) Per-well flags only help RMSE if there is a better fallback for flagged wells; our previous learned gates failed (5.017 vs 4.905). |
| Notes | Key conclusion for the metric: band-confusion error cannot be removed by recalibrating the current posterior; the evidence must change. This is consistent with the outline's framing and now quantified. Interesting side facts: the PIT hump at deciles 5-8 with mean z near 0 indicates a slight skew (median below mean) in typical columns; in-sample, mild sharpening (T=0.9) is optimal for RMSE while T=1.0 is optimal for NLL, but the difference (0.003 ft) is noise. The 'conformal under HMM' literature (Markov de Finetti, particle-filter ACI) matters only if intervals are produced sequentially along the lateral in real time; for offline pooled-RMSE scoring, well-level split conformal is sufficient. |

### Other info

- **Uncertain fields (omitted above)**:
  - venue_link

---

## B10 Test-time adaptation / per-well self-supervision on the known prefix (TTL for time series, PETSA, closed-form offset/slope correction)

### Basics

| Field | Value |
|---|---|
| Name | Test-time adaptation (TTA) using the known heel prefix: label-driven TTA for time-series forecasting (TAFAS, AAAI 2025; PETSA, ICML 2025 TTA workshop; FAC 'principled TTA' 2026), test-time training modules (TTL/TTT for forecasting, arXiv 2409.14012), online self-adaptive stereo (MADNet/MAD, CVPR 2019), entropy-based TTA (Tent) as a counter-example, and closed-form per-well offset / slope / gain correction (10th-place ROGII heel-anchored gain) |
| Year | 2019-2026 (MADNet 2019, Tent 2021, TTL 2024, TAFAS 2025, PETSA 2025, FAC May 2026, ROGII writeups 2026) |
| Venue link | TAFAS, AAAI 2025, arXiv:2501.04970 (https://arxiv.org/abs/2501.04970); PETSA, ICML 2025 Workshop on Test-Time Adaptation, arXiv:2506.23424 (https://arxiv.org/abs/2506.23424); FAC 'Towards Principled Test-Time Adaptation for Time Series Forecasting', arXiv:2605.17250 (https://arxiv.org/abs/2605.17250); TTL 'Test Time Learning for Time Series Forecasting', arXiv:2409.14012 (https://arxiv.org/abs/2409.14012); MADNet 'Real-time self-adaptive deep stereo', CVPR 2019, arXiv:1810.05424 (https://openaccess.thecvf.com/content_CVPR_2019/html/Tonioni_Real-Time_Self-Adaptive_Deep_Stereo_CVPR_2019_paper.html); Tent, ICLR 2021, arXiv:2006.10726; ROGII 10th place: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/10th-place-solution-a-compass-not-a-map |
| Evidence level | Peer-reviewed for TAFAS (AAAI), MADNet (CVPR), Tent (ICLR); workshop paper for PETSA; arXiv preprints for FAC and TTL; ROGII numbers are competition writeups (read via our A17 extraction; the Kaggle pages are JS-rendered and could not be re-fetched here). Plus an internal diagnostic run for this report on results/baseline_check/holdout_predictions.pqt (60-epoch baseline, 155 wells, pooled 5.43 ft): a global heel-anchored gain is flat (optimum s=1.00), and offset transfer only works WITHIN a suffix (see metric_and_performance). No geosteering-specific TTA paper found. |
| Extreme summary | Label-driven TTA freezes the trained model and fits a tiny calibration module (TAFAS gated input/output calibration, PETSA low-rank gated adapters) to ground truth that has already been revealed for part of the current series, giving about 2-10% MSE gains on non-stationary forecasting benchmarks. For us the revealed labels are the heel prefix (~1,700 ft median), so the transferable recipe is: move the heel back inside the known prefix (pseudo-heels), predict the known part, and fit either a closed-form correction or a few adapter / BN steps per well before predicting the real suffix; entropy-minimisation TTA (Tent) is mode-seeking and should be avoided under an RMSE-expectation decode. |

### Problem formulation

| Field | Value |
|---|---|
| Task definition | Adapt a pretrained predictor at inference time to the particular test series using information available only then. Forecasting TTA (TAFAS, PETSA, FAC): multivariate forecasting under distribution shift, with the first p steps of the forecast window revealed before the rest (partially observed ground truth, POGT). MADNet: continuous self-supervised adaptation of a stereo network to a new domain from photometric reprojection loss. For us: predict TVT over the unknown suffix of one lateral given its GR, trajectory, typewell and the known TVT over the heel prefix; the adaptation signal is the prefix TVT labels (and optionally prefix GR-typewell self-consistency). |
| Inputs | Forecasting TTA: look-back window X plus revealed partial targets Y[:p]. MADNet: stereo pairs only (unsupervised). Ours: the unchanged 16-channel cost image plus, for TTA, K pseudo-heel versions of the same well built by shifting the prefix/suffix boundary back by D in the known prefix (so that the prefix labels become pseudo-suffix targets). |
| Target parameterisation | Forecasting TTA works on the raw target (MSE/Huber) or its spectrum (PETSA frequency loss, FAC frequency-domain correction). For us: the per-column TVT expectation (or per-column posterior over 400 levels); closed-form corrections parameterise the correction as offset + slope in MD (drift in the surface A = TVT + Z, see A18) or as a heel-anchored gain s on (TVT_pred - TVT_heel). |
| Output representation | Point forecasts corrected by an additive calibration term (TAFAS/PETSA output GCM: Y_hat + tanh(alpha) * (W Y_hat + b)). For us the posterior can be shifted (a per-well shift or per-column shift of the 400-level softmax, which preserves the expectation decode) or re-tempered. |

### Method

| Field | Value |
|---|---|
| Architecture | TAFAS: frozen source forecaster wrapped by an input Gated Calibration Module GCM(X) = X + tanh(alpha) * (W X + b) and an output GCM of the same form, with W, b zero-initialised so adaptation starts as a no-op; only W, b, alpha are updated. PETSA: same placement, W factorised low-rank (A B) and gating tanh(alpha * X) input-dependent, about 33x less memory than TAFAS at window 720. FAC: correction parameterised directly in the frequency domain. TTL (2409.14012): replaces Mamba blocks with Test-Time-Training layers whose hidden state is updated by a self-supervised inner loss per sequence (no labels). MADNet/MAD: lightweight pyramid stereo network; MAD updates one sub-module per frame chosen by a reward, making online backprop real-time. Tent: updates BatchNorm affine parameters to minimise prediction entropy. |
| Core mechanism | Partial ground truth is the most reliable adaptation signal available in forecasting: TAFAS uses L_partial = MSE(Y_cali[:p], Y[:p]) on the revealed prefix of the current window plus L_full on earlier fully-revealed windows, and chooses p by the dominant FFT period (PAAS). Freezing the backbone and learning only a zero-initialised, gated correction keeps the source semantics and prevents the few revealed points from overfitting the whole model. The key question for any transfer is whether the error on the revealed part is informative about the error on the part still to predict. In forecasting that holds because the shift is in the series statistics; in our task the model is re-anchored to the true TVT at the heel, so errors restart near zero and grow like a random walk (10th place: mean /err/ 0.32 ft at 50 ft to 4.55 ft at 5,000 ft; first fifth of the scored zone predicts the last fifth with r = 0.131). Therefore the transferable part is per-WELL systematic bias (typewell-lateral GR mismatch, stretch, dip-bias of this well), measured by forecasting inside the prefix from pseudo-heels, not per-window persistence. |
| Training objective | TAFAS: L_partial + L_full (MSE). PETSA: Huber (delta 0.5) + frequency (FFT spectrum) loss + patch-wise structural loss (local mean/variance/correlation), each on partial and delayed-full labels. Tent: entropy minimisation (label-free). MADNet: photometric reprojection loss (label-free). For us, candidate TTA losses on pseudo-heel forecasts: MSE of the expectation vs prefix TVT (RMSE-consistent) plus the existing soft CE to the typewell targets; avoid pure entropy minimisation. |
| Decoding | Unchanged: after adaptation the forecaster produces its usual output (for us: posterior expectation, optionally MD-phase TTA x8). |
| Metric consistent decode | Label-driven TTA with an MSE/expectation loss keeps the decode RMSE-consistent. Tent-style entropy minimisation sharpens posteriors toward the dominant mode (mode-seeking) and is expected to raise pooled RMSE on bimodal (repeated-band) wells, where our hedged expectation is Bayes-optimal. |
| Evidence scope | Single series / single well: the adapter is fit per test series (or continuously across a stream). For us: one well plus its typewell; pseudo-heels reuse the same well's prefix. |
| Prefix conditioning | This item IS prefix conditioning. Forecasting TTA: fine-tunes a calibration module on revealed ground truth. Our current pipeline already uses the prefix in three closed ways: (1) seen_tvt_rel and tw_seen_tvt_abs_diff input channels; (2) GR_calibration (typewell_calibration_with_seen=True) blends the typewell GR toward a pseudo-curve built from prefix (TVT_input, GR) within TVT0 +- 100 ft with weight 0.7 -- a closed-form test-time calibration of the reference; (3) the prediction is anchored at the heel TVT. What is missing: any use of the prefix LABELS as a supervised adaptation signal for the network or its output (fine-tuning, offset/slope/temperature fitted per well). |
| Propose vs select | Single adapted output; can slot in as a per-well post-processor applied to any member (ConvNeXt, FastViT, AnchorCNN) or produce an adapted copy of a member for blending. |
| Physical constraints | None intrinsic. For us a correction restricted to offset + slope of the surface A(MD) = TVT + Z is physically interpretable (constant dip bias); slope extrapolation must be heavily shrunk (see risk). |

### Data

| Field | Value |
|---|---|
| Dataset | TAFAS/PETSA/FAC: ETTh1/ETTh2/ETTm1/ETTm2, Weather, Electricity, Traffic, Exchange, etc. (standard long-horizon forecasting benchmarks). MADNet: KITTI, synthetic-to-real stereo sequences. Ours: 773 Kaggle wells (618 train / 155 holdout), known prefix median 1,702 ft (q10 1,422, q90 1,944 ft of MD from the first row, measured here from data/train), unknown suffix median 4,840 ft (q10 3,263, q90 6,349 ft). |
| Synthetic to real | TTA is a sim-to-real tool in stereo (MADNet adapts synthetic-trained networks online to real scenes). For us it could absorb per-well synthetic-to-real mismatch (85% synthetic training), since the adapter only sees real prefix labels of the test well. |

### Results

| Field | Value |
|---|---|
| Metric and performance | TAFAS (Table 1): average MSE gains ~4.95% (iTransformer) and 5.20% (DLinear) at H=336, 5.76% and 6.30% at H=720; up to 45% on Chronos on unseen data; ablation iTransformer H=720: 0.786 no-TTA -> 0.704 full TAFAS; removing GCM 0.832 (worse than no TTA), output-only calibration 0.808. PETSA: ~2-5% average MSE reduction vs no-TTA; 127 best-MSE wins vs TAFAS 88; e.g. ETTh1 iTransformer 0.449 -> 0.432 (TAFAS 0.435); ~33.6x less memory. ROGII (via A17): 10th place applied a heel-anchored gain s = 1.024 at the end of its stack; drift extrapolation from the prefix 12.60 vs flat 15.91 (their scale); 22nd: OOF drift-direction correctors CV -0.25 but LB +0.34. Internal diagnostic on our 60-epoch baseline (pooled 5.431 ft): global heel-anchored gain s in {0.97..1.05} gives 5.442/5.431/5.431/5.434/5.442/5.453/5.486 -- flat, optimum 1.00, so the 10th-place gain does not transfer; using the first 20% of each SUFFIX as if revealed, a constant offset (mean error of its last 50 rows) lowers the RMSE of the remaining 80% from 5.92 to 5.51 ft (first 30%: 6.13 -> 5.60), whereas slope extrapolation explodes (12.8 ft); correlation of early vs late mean error 0.20 / 0.42 / 0.53 for 10/20/30% -- i.e. errors are persistent once they exist, but the prefix itself carries no suffix error because the prediction is re-anchored at the heel. |
| Comparability | Forecasting-benchmark MSE percentages are not comparable to pooled RMSE on 155 wells. The internal diagnostic is directly comparable but uses suffix labels (an upper-bound style probe, not a deployable method). |
| Validation protocol | TAFAS/PETSA use online streams with delayed labels; FAC argues protocols are heterogeneous and some leak (evaluating on steps already used for updates) and proposes a matured-ground-truth-only protocol. For us: TTA must use only prefix labels of each holdout well; any hyperparameter (D, steps, lr, shrinkage) must be tuned on training wells (or cross-fitted halves of the holdout) to avoid holdout tuning. |

### Ambiguity handling

| Field | Value |
|---|---|
| Repeated layers | Weak lever. Prefix labels tell the model which band the heel is in (already used via the anchor). A per-well adapter fitted on pseudo-heels could learn that for this well the lateral GR systematically matches the band above/below (e.g. a GR scale/offset mismatch vs the typewell), which is one cause of wrong-band picks. Entropy-based TTA would make band confusion worse by committing to one mode. |
| Drift | Main target. Drift = one straight-line error per well (~25% of our squared error; per-well line oracle 3.08 ft on ep150). A per-well dip-bias estimated from pseudo-heel forecasts inside the prefix (predict prefix segments, measure the slope of error in A-space) could be applied, shrunk, to the suffix. External evidence is pessimistic: 10th and 22nd found drift direction essentially unpredictable from observables, and our diagnostic shows naive slope extrapolation is catastrophic; an offset correction helps only once error has accumulated, which never happens in the prefix. |
| Uncertainty and calibration | Pseudo-heel forecasts give a per-well calibration check: compare predicted posterior spread against realised error on known prefix segments and fit a per-well temperature (or a shrunk global-plus-well temperature) -- a label-driven, per-well version of B09 temperature scaling that keeps the expectation decode. No TTA paper reports calibration of histogram outputs. |

### Relevance to erudition

| Field | Value |
|---|---|
| Where to use | A post-processing / inference stage between the trained model and seq_NN_holdout_eval.py (or seq_NN_rescore.py, which already rescored MD-phase TTA without retraining): for each well build K pseudo-heel items by moving the boundary back inside the prefix (reuse Simulator.start_point_shift logic deterministically, plus _recalibrate_typewell_after_start_shift so GR_calibration only sees the pseudo-prefix), predict, compare with known TVT, then (a) closed-form: offset / shrunk-slope / temperature correction; or (b) gradient TTA: zero-initialised output calibration head (TAFAS-style GCM on the 400-level logits or on the expectation) or BN-affine / last-decoder-block update for a few steps; then predict the real suffix. |
| Error mode targeted | Drift (~25%) primarily; per-well calibration (broad/skewed humps ~13%, blend 15%) secondarily; band confusion only indirectly (per-well GR mismatch). |
| Gap vs our implementation | We already do: prefix as input channels (seen_tvt_rel, tw_seen_tvt_abs_diff), closed-form reference calibration from the prefix (GR_calibration, weight 0.7 within TVT0 +- 100 ft), heel anchoring, MD-phase TTA x8 (test-time augmentation, not adaptation; -0.04 to -0.15 ft, 4/4 positive), and a start_point_shift augmentation that is switched off. We do NOT: (1) use prefix TVT labels as a supervised signal at inference -- no pseudo-heel forecasting, no per-well fine-tuning, no per-well output calibration; (2) estimate per-well bias or temperature from the model's own behaviour on the known part; (3) train the model to be adaptable (start_point_shift off means the network has only ever seen the true heel position, so pseudo-heel predictions may be slightly out of distribution). The 10th-place global heel gain (s=1.024) is NOT a gap: our diagnostic shows the optimum is s=1.00 for our model. |
| Proposed experiment | E0 (zero training, go/no-go, ~1 h): new script kaggle_1st_place/solution/seq_NN_pseudo_heel.py that loads a trained run (models.pkl via the seq_NN_rescore.py loader), and for each of the 155 holdout wells builds K=4 pseudo-heel items with the boundary moved back by D in {300, 600, 900, 1200} ft inside the known prefix (deterministic version of Simulator.start_point_shift; typewell recalibrated on the pseudo-prefix only). Record per-well pseudo-forecast error features: mean error, error slope in A = TVT + Z per 1,000 ft, error at the pseudo-suffix end, and posterior spread vs /error/. Then correlate them with the real suffix error (mean error, fitted line slope) per well. Go only if Spearman /rho/ >= 0.3 on the slope or offset (cross-fitted); otherwise stop -- this mirrors the 10th/22nd 'drift is an info wall' finding at no cost. E1 (closed-form, if E0 passes): correct the suffix by beta * (estimated A-space slope) * (MD - MD_heel) with beta fitted by cross-fitting on holdout halves (or on training-well OOF), plus an optional per-well temperature tau_w = clip(global_tau * f(spread ratio)); control = same weights without correction; score with seq_NN_robust_compare.py (paired bootstrap, k*). E2 (gradient TTA): TAFAS-style zero-initialised output GCM on the per-column logits (a per-well shift + scale along the level axis, 2-10 params) or BN affine of the decoder, optimised 10 steps Adam lr 1e-3 on the K pseudo-heel items with loss = MSE(expectation, prefix TVT) + existing soft CE; control = no TTA. E3 (make the model adaptable): retrain one 150-epoch run with aug_cfg['start_point_shift']['apply_prob']=0.25 so pseudo-heels are in distribution, then repeat E0-E2; control = baseline 150-epoch run (also tells whether start_point_shift alone helps as augmentation). |
| Implementation effort | E0 low (one script reusing rescore loader and start_point_shift; ~150 lines). E1 low. E2 medium (per-well autograd loop, careful BN handling: freeze running stats). E3 low code, one 150-epoch training (~5 h). |
| Distillation compatibility | Fully compatible: closed-form and output-calibration TTA act after the student; BN/adapter TTA works on the FastViT student equally (it has BN). The teacher does not need to change. |
| Error correlation with ours | Not a new member; it modifies the error of an existing member per well. If it works, it corrects exactly the component the ensemble cannot average away (shared per-well drift), so its benefit should survive blending. |
| Notes | Why forecasting TTA works and ours may not: TAFAS/PETSA exploit persistence of error within one forecast window (the first p steps of the SAME window), which our diagnostic confirms exists within a suffix (offset from first 20% cuts late RMSE 5.92 -> 5.51) but is unavailable at test time because nothing in the suffix is revealed. The only analogue is a rig setting where labels arrive progressively while drilling (real-time geosteering): there, TAFAS-style partial-truth adaptation is directly applicable and the gain could be large -- worth noting for the thesis framing even though the Kaggle protocol hides it. Related items: B09 (calibration), A15/A18 (dip state, A-space), A17 (10th/22nd drift findings), B11 (start_point_shift as augmentation). TTL (2409.14012) is test-time TRAINING layers inside the architecture (self-supervised inner loop, no labels), only loosely related. |

### Other info

- **Uncertain fields (omitted above)**:
  - authors_org
  - code_availability
  - gpu_cost
  - inference_budget
  - risk

---

## B11 Synthetic data, domain randomisation and sim-to-real alignment (hard repeated-layer generation, pretrain-to-alignment 2026)

### Basics

| Field | Value |
|---|---|
| Name | Synthetic data, domain randomisation and sim-to-real alignment: adversarial decoy synthesis (ROGII 8th place, +-15 ft decoys, mixed synthetic continuation), pretrain-to-alignment paradigm for geophysical AI (Gao, Wu et al., USTC 2026), domain-randomised synthetic data with ambiguity self-curation (FoundationStereo, CVPR 2025), fabricated-error synthetic pretraining (ROGII 7th place refiner), distribution alignment over realism (sim-to-real RF positioning 2026), and synthetic replay against forgetting |
| Year | 2017-2026 (domain randomisation 2017; FoundationStereo 2025; ROGII writeups 2026; pretrain-to-alignment May 2026; RF sim-to-real July 2026) |
| Venue link | ROGII 8th: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/8th-place-pf-ranker ('PF ranker + Synthetic-Pretrained U-Net'); 7th: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/7th-place-solution-hmm-unet-agent-is-all-you ; 14th (synthetic pretraining lineage): https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/14th-place-a-hierarchical-u-net-stack-with-seed-a ; Pretrain-to-alignment: arXiv:2605.16783 (https://arxiv.org/abs/2605.16783); FoundationStereo: CVPR 2025, arXiv:2501.09898 (https://nvlabs.github.io/FoundationStereo/); RF positioning: arXiv:2607.04400 (https://arxiv.org/abs/2607.04400); Tobin et al. 'Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World', IROS 2017, arXiv:1703.06907. |
| Evidence level | Competition writeups with ablations (8th, 7th; numbers taken from our A17 extraction because Kaggle pages are JS-rendered and could not be re-fetched); arXiv preprint for pretrain-to-alignment (qualitative gains only in the summary read); peer-reviewed CVPR for FoundationStereo; code audit of our Simulator and training loop for this report (config defaults, finetune stage, run logs). |
| Extreme summary | Synthetic wells only help if they reproduce the real failure mode: the 8th-place U-Net broke a 6.6 plateau by (a) generating surfaces in u = TVT + Z space with real piecewise-linear statistics and injecting adversarial decoy matches at +-15 ft (decoy_prob 0.25), and (b) keeping 50% synthetic data during real training instead of pretrain-then-finetune. Geophysical pretrain-to-alignment (USTC 2026) adds a real-data alignment stage with frozen backbone + LoRA and label-free physical losses, then per-sample adaptation. Our generator already mixes ~75% synthetic throughout and has an unused real-emphasis finetune stage, but has no decoy / hard repeated-layer mode, and our one attempt to make synthetics more diverse (rb_v2_synth) hurt. |

### Problem formulation

| Field | Value |
|---|---|
| Task definition | Improve a model trained mostly on simulated examples so that it generalises to real data: by making simulations harder or more realistic where the model fails (decoys, fabricated confident errors), by randomising nuisance factors (domain randomisation), by removing or keeping ambiguous samples (self-curation), and by adding a real-data alignment stage (fine-tuning, LoRA refinement with physics losses, mixed continuation). |
| Inputs | 8th: real trajectory and real GR residual/missingness preserved; synthetic surface u_new from the RDP skeleton of a real surface; GR rendered from the typewell at TVT = u - Z. Pretrain-to-alignment: 4,000 labelled synthetic seismic images + 3,000 unlabelled field images. Ours: real trajectory (Z), real typewell (773 typewells / ~60 master systems), synthetic TVT path from diff-block bootstrap of real paths, GR = typewell(TVT_sim) + residual noise (TVT_bias_v2 by default; real_block optional). |
| Target parameterisation | 8th: surfaces generated in u = TVT + Z space (A-space), TVT = u - Z. Ours: TVT path generated directly (z_shift adds a sampled trend to TVT at the heel; Ruby's z_shift keeps TVT + Z consistent, per CLAUDE.md), labels exact for synthetic wells. Pretrain-to-alignment: relative geologic time (RGT) volumes. |
| Output representation | Unchanged by synthetic data; the model keeps its per-column softmax over 400 TVT levels with expectation decode. |

### Method

| Field | Value |
|---|---|
| Architecture | Data-side methods; architecture-agnostic. 8th U-Net leg: smp ResNet34 U-Net, 9 channels, InstanceNorm, column-wise CE; 200 epochs synthetic-only, then 50 epochs real 50% + synthetic 50% (lr 3e-5, frozen BN stats, hflip 0.5, anchor-real-prob 0.2). Pretrain-to-alignment: MAE ViT encoder pretrained on field data -> ViT encoder + conv decoder supervised on synthetic (MSE + MS-SSIM) -> frozen backbone + LoRA refined on field data with label-free geological losses -> per-sample adaptation. 7th: refiner U-Net pretrained on ~6,000 synthetic wells whose first-pass input is truth plus a lognormal-amplitude low-passed random walk (fabricated confident errors), finetuned 8 epochs on real. |
| Core mechanism | (1) Hard-case synthesis: a decoy is a segment where the lateral GR is rendered from the typewell at TVT +- 15 ft (the dominant aliasing period of the Eagle Ford typewells) while the label stays at the true TVT, so the network learns that a locally better GR match at the adjacent repeated band is not sufficient evidence to jump -- exactly the wrong_mode / blend failure. (2) Mixed continuation: keeping synthetic at 50% during the real stage acts as a regulariser and avoids forgetting the broad synthetic prior (the same finding as replay against catastrophic forgetting). (3) A-space generation: dip statistics (segment median ~98 ft, AR(1) slope rho ~0.248, jumps 3.1%) are realistic by construction and rejection keeps drift inside the real envelope. (4) Alignment: when field labels are scarce, adapt only small modules (LoRA) with physically motivated label-free losses on real data (orientation alignment, isochron consistency, horizon consistency), then per-sample. (5) Distribution alignment matters more than realism or size (RF positioning 2026). (6) FoundationStereo removes ambiguous samples (repetitive texture) from its randomised data -- the opposite choice to decoys; for an RMSE-expectation decode with exact synthetic labels, ambiguous samples teach calibrated hedging and should be kept, while samples whose label is wrong should be removed. |
| Training objective | Unchanged supervised losses on synthetic samples (our soft CE to typewell targets + expectation regression + GR_penalty). Alignment stage options: real-only or real-weighted supervised finetune (our finetune_epochs stage), LoRA/adapter refinement, or label-free consistency losses (pretrain-to-alignment) -- our GR_penalty (lateral GR vs typewell GR at predicted TVT) is already such a label-free physical loss and could be applied to unlabelled suffixes. |
| Decoding | Unchanged (posterior expectation, MD-phase TTA x8). |
| Metric consistent decode | Not applicable directly; decoys with exact labels push the network toward calibrated bimodal posteriors on genuinely ambiguous wells, which the expectation decode handles optimally for RMSE. Self-curation that deletes ambiguous samples would make the model overconfident on them. |
| Evidence scope | Training-data scope; per-well inference unchanged. |
| Prefix conditioning | 8th: anchor-real-prob 0.2 (real heel anchoring in synthetic). Ours: synthetic wells keep the real prefix (GR_sim[seen] = GR[seen] when range_shift == 0) and GR_calibration uses the prefix; the pretrain-to-alignment stage-4 per-sample adaptation corresponds to B10. |
| Propose vs select | Not applicable; a better-trained member slots into the existing ensemble / distillation unchanged. |
| Physical constraints | 8th: TVT = u - Z with u piecewise linear and rejection outside the real drift envelope (<= 12 resamples). Ours: z_shift preserves TVT + Z consistency; fault_ratio 0.05 with up to 3 faults (std 1.75 ft); safe_range_guard off by default. |

### Data

| Field | Value |
|---|---|
| Dataset | ROGII Kaggle: 773 wells (we use 618 train / 155 holdout). 8th: synthetic surfaces from fold-local real surfaces. Pretrain-to-alignment: global shelf-edge clinothems, 4,000 synthetic + 3,000 field seismic images. FoundationStereo: ~1M synthetic stereo pairs (FSD). |

### Results

| Field | Value |
|---|---|
| Comparability | 8th-place fold RMSEs are on their scale (CV 6.4-6.8 for the U-Net) and not directly comparable to our 4.84-5.47 ft on 155 holdout wells; relative gains (~0.2-0.5 ft) are the transferable quantity. |
| Inference budget | No inference cost (training-time only). |

### Ambiguity handling

| Field | Value |
|---|---|
| Repeated layers | Central. Our synthetic GR is typewell(TVT_sim) + residual noise whose TVT-bias component is small (TVT_bias_v2_tvt_std 0.95 ft), so the lateral GR almost always matches the TRUE band best; the only repeated-band ambiguity the network sees in synthetic data is the one intrinsic to the typewell. Real laterals often match an adjacent band better over hundreds of feet (22nd: lateral vs master corr 0.865, ~12 API scatter). The decoy mode injects exactly this: an adversarial segment where the wrong band is the better local match, teaching the network to keep mass on the anchored band unless evidence persists. This targets wrong_mode (24%) and blend (15%) directly. |
| Drift | A-space generation (8th) gives realistic dip statistics; our diff-block bootstrap of real TVT paths (lengths 1024/2048/4096 rows, weights 0.45/0.40/0.15) plus optional drift_ratio (0.0 default) already reuses real path shapes. Generating in A = TVT + Z space would make synthetic drift consistent with the real one-straight-line-per-well error mode (see A18). |
| Uncertainty and calibration | Synthetic data shape the prior over ambiguity: if synthetic wells are easier than real ones (no decoys), the network's posteriors are over-confident on real ambiguous wells, which raises RMSE under the expectation decode. Checkable: compare reliability curves / ECE of band-weight on synthetic validation wells vs real holdout wells (posterior_analysis.py). A real-only alignment stage mainly recalibrates. |

### Relevance to erudition

| Field | Value |
|---|---|
| Where to use | Simulator.z_shift / Simulator.process in kaggle_1st_place/solution/experiments/arch/seq_NN_dataset.py (new decoy step after GR_sim is rendered), sim_cfg in experiments/arch/seq_NN_cfg.py (new 'decoy' block, synthetic ratio), and the existing finetune stage in seq_NN_train.py (finetune_epochs / finetune_disabled_augs). Distillation: the FastViT student inherits any teacher improvement and can itself be trained on decoy data. |
| Error mode targeted | Band confusion (~51%: wrong_mode 24%, blend 15%, multimodal_miss 12%) via decoys; drift (~25%) via A-space generation; calibration of broad humps (~13%) via a real-data alignment stage. |
| Gap vs our implementation | Already done: heavy synthetic mixing throughout training (~75% z_shift, not pretrain-then-finetune, i.e. the regime the 8th place found best), real trajectories and typewells, real residual banks (real_block), trajectory mixup and master-series re-skinning (Idea 2, net negative so far), TVT + Z consistency. Missing: (1) no decoy / adversarial repeated-band mode -- nothing renders GR from the wrong band while keeping the true label; (2) no A-space (surface) generator with RDP skeleton + AR(1) slope deviations and drift-envelope rejection (diff-block bootstrap is path-space); (3) the real-emphasis finetune stage exists but was never run (finetune_epochs=None everywhere), and there is no ablation of the synthetic ratio (0.75 is inherited, never swept; keep_ori_path has a sweep grid (0.0, 0.025, 0.05) in seq_NN_cfg.py but no results reported); (4) no calibration comparison between synthetic and real validation; (5) no label-free alignment losses on unlabelled real suffixes (pretrain-to-alignment stage 3) -- although GR_penalty already implements one; (6) CLAUDE.md lead 2 (re-apply GR_calibration after reskin) still open, which is a sim-to-real statistics mismatch of exactly the kind the RF paper warns about. |
| Proposed experiment | E1 (decoys, highest priority): add sim_cfg['decoy'] = {apply_prob: 0.25, period_ft: 15 or 'acf', seg_len_ft: (200, 800), n_max: 2, blend: (0.6, 1.0)} and a Simulator.decoy step applied only to z_shift-simulated wells: pick a random suffix segment, compute the typewell's dominant autocorrelation period P near the segment's TVT (fallback 15 ft), and replace GR_sim on that segment with typewell(TVT_sim +- P) + the same residual noise (blend weight w from the range, cosine-tapered edges), leaving TVT labels unchanged. Train 150 epochs vs the 150-epoch baseline (4.9785) with seed-matched control; report pooled RMSE (seq_NN_robust_compare.py, paired bootstrap, k*) and the wrong_mode / blend / multimodal_miss shares via posterior_analysis.py on a posterior dump. Also a dose sweep at 60 epochs (apply_prob 0.1 / 0.25 / 0.4). E2 (alignment stage, zero new code): rerun the best config with finetune_epochs = 10 and 20 (z_shift disabled, lr x0.3) from the best checkpoint, plus a variant that keeps z_shift at 0.3 during finetune (mixed continuation, 8th-style) by setting finetune_disabled_augs=[] and overriding apply_prob; control = same checkpoint without finetune. E3 (synthetic ratio): z_shift apply_prob 0.5 vs 0.75 vs 0.9 at 150 epochs (2 seeds each if budget allows). E4 (A-space generator): new sample_model 'surface_rdp' in TrendSampler: u = TVT + Z of a real training well, RDP skeleton (eps 2 ft), add piecewise-linear deviations (segment length ~ lognormal median 98 ft, AR(1) slope rho 0.25, jump prob 3%), TVT = u - Z, reject if outside the real drift envelope; compare against diff_block_bootstrap. E5 (calibration check, no training): dump posteriors on 155 synthetic validation wells and on the real holdout, compare band-weight reliability; if synthetic is much better calibrated, prioritise E1/E2. |
| Implementation effort | E1 low-medium (~80 lines in the Simulator plus config validation). E2 low (flags exist). E3 low (config). E4 medium (new sampler mode and rejection). E5 low (posterior dump script exists). |
| Gpu cost | Each arm is one standard training run: ~5 h per 150-epoch ConvNeXt run on the RTX 5060 8 GB (per CLAUDE.md), ~2 h at 60 epochs; finetune arms add 10-20 epochs (~20-40 min). Decoy rendering is CPU-side and negligible. No inference cost. |
| Distillation compatibility | Fully compatible: better teacher posteriors propagate through KD; the student's KD batches are already 75-85% synthetic, so decoy samples also directly teach the student to hedge. |
| Error correlation with ours | A decoy-trained model should differ from the baseline mainly on repeated-band wells (the tail that dominates pooled RMSE), so it is a more diverse blend member than a second seed; a finetuned copy of the same run is highly correlated. |
| Notes | Decoys vs self-curation is the key design choice: FoundationStereo deletes ambiguous samples because in its data the ambiguity makes labels unlearnable for a mode-seeking metric (EPE/bad-pixel); in our RMSE-expectation setting with exact synthetic labels, ambiguity is the signal the model must learn to hedge on, so we should ADD ambiguous (decoy) samples and only delete synthetic samples whose label is physically implausible. Pretrain-to-alignment's stage 3 (frozen backbone + LoRA, label-free geological losses on field data) maps to training a small adapter with GR_penalty on unlabelled suffix GR of holdout/test wells (transductive), which is closer to B10. Related items: A17 (8th/7th/22nd writeups), A18 (A-space identity), A19 (external typelogs to diversify masters), B09 (calibration), B10 (per-sample adaptation). |

### Other info

- **Uncertain fields (omitted above)**:
  - authors_org
  - code_availability
  - metric_and_performance
  - risk
  - synthetic_to_real
  - validation_protocol

---

## B12 Efficient multi-scale decoders (BiFPN/EfficientDet, FPN variants) for dense full-resolution prediction

### Basics

| Field | Value |
|---|---|
| Name | Efficient multi-scale decoders: BiFPN / EfficientDet (weighted bidirectional FPN), with FPN, PANet, Panoptic/semantic FPN, RevBiFPN and BiFPN-in-U-Net (U-Det) as comparators, evaluated as a replacement for the PretrainedUNet2d decoder |
| Authors org | M. Tan, R. Pang, Q. V. Le (Google Brain) for EfficientDet/BiFPN; T.-Y. Lin et al. (FAIR) for FPN; S. Liu et al. (CUHK/SenseTime) for PANet; A. Kirillov, R. Girshick, K. He, P. Dollar (FAIR) for Panoptic FPN; V. Chiley et al. (Cerebras) for RevBiFPN; N. V. Keetha et al. for U-Det |
| Year | 2020 (EfficientDet, CVPR 2020); FPN 2017; PANet 2018; Panoptic FPN 2019; U-Det 2020; RevBiFPN 2023 |
| Venue link | EfficientDet: CVPR 2020, arXiv:1911.09070 (https://arxiv.org/abs/1911.09070). Panoptic FPN: CVPR 2019, arXiv:1901.02446 (https://arxiv.org/abs/1901.02446). U-Det: arXiv:2003.09293 (https://arxiv.org/abs/2003.09293). RevBiFPN: MLSys 2023 (https://proceedings.mlsys.org/paper_files/paper/2023/file/c4f3fb058f957d5f753f9f9f0f08d991-Paper-mlsys2023.pdf). Enhanced YOLOv5 PANet->BiFPN ablation: PMC10611198 (https://pmc.ncbi.nlm.nih.gov/articles/PMC10611198/). |
| Code availability | Public: google/automl (official EfficientDet TF), rwightman/efficientdet-pytorch (PyTorch BiFPN, timm-compatible), zylo117/Yet-Another-EfficientDet-Pytorch; FPN decoders in segmentation_models_pytorch (Unet, FPN, PAN, UPerNet). No BiFPN code exists in the Erudition repo today (grep for BiFPN/FPN in kaggle_1st_place/solution finds nothing). |
| Evidence level | Peer-reviewed for EfficientDet/FPN/Panoptic FPN (object detection and 1/4-resolution semantic segmentation). No peer-reviewed evidence found for BiFPN beating a U-Net decoder on full-resolution dense regression (depth, disparity, or column-wise distributions). BiFPN-vs-U-Net evidence is limited to small medical-segmentation papers (U-Det, arXiv, non-pretrained baseline) and detection-neck ablations in YOLO variants. |
| Extreme summary | BiFPN is a detection neck: repeated top-down plus bottom-up fusion over P3-P7 with learned, normalised per-input weights. In its own ablation the weighted fusion adds only +0.45 COCO AP over unweighted BiFPN, and +0.31 AP over repeated FPN+PANet at 0.68x FLOPs. Its one segmentation result predicts at 1/4 resolution and confounds the backbone. Nothing here supports a lower RMSE for our full-resolution column-softmax output, and the only FLOP saving on offer (narrower mid-level fusion) can be had inside the existing U-Net. |

### Problem formulation

| Field | Value |
|---|---|
| Task definition | EfficientDet: multi-scale object detection (box + class) on COCO, with a secondary semantic-segmentation experiment on Pascal VOC 2012. In Erudition the analogous role would be the decoder that turns ConvNeXt/FastViT stage maps into a full-resolution 345 MD x 400 TVT feature map, which a per-column softmax over TVT levels turns into a distribution that is expectation-decoded to TVT. |
| Inputs | RGB images for EfficientDet. In our setting the decoder input is the 4 encoder stage maps (convnext_small at 16ch 400x345, measured: 96ch 200x87, 192ch 100x43, 384ch 50x21, 768ch 25x10 with the default stem_stride (2,4)) plus the full-resolution stem skip (emb_dim 32). |
| Target parameterisation | Not applicable (a generic decoder). It leaves our target unchanged: a per-column categorical distribution over 400 TVT levels. |
| Output representation | EfficientDet emits box/class heads at P3-P7 (1/8 to 1/128). Its segmentation variant adds P2 and predicts at 1/4 resolution, then upsamples bilinearly. The Panoptic/semantic FPN head likewise merges levels at 1/4 and upsamples 4x. Neither natively outputs full resolution; our output has to be full resolution along the 400-level TVT axis, because that axis is the class axis of the softmax. |

### Method

| Field | Value |
|---|---|
| Architecture | BiFPN layer: every level is projected to a single width W (64-384 depending on D0-D7). A top-down pass is followed by a bottom-up pass; nodes with a single input are removed, and an extra same-level edge runs from input to output at the middle levels. Each fusion node computes O = sum_i w_i * I_i / (eps + sum_j w_j) with ReLU-constrained scalar weights (fast normalised fusion), followed by a depthwise-separable 3x3 conv, BN and swish. The layer is repeated 3-8 times (compound scaling). Segmentation variant: levels P2-P7, W=128, 3 BiFPN repeats, classification head width 256 repeated 3 times, per-pixel classification on P2 only. For comparison, the current PretrainedUNet2d decoder (experiments/bilzard/seq_NN_pretrained_unet.py): 3 BilinearResUpBlock2d (bilinear upsample, concat skip, 1x1 proj to the skip width, ResBlock of two 3x3 convs) at widths 384/192/96. Then to_full_resolution (1x1 to 32), BilinearAddResUpBlock2d with the full-resolution stem skip, and an out ResBlock at 32 channels. Measured parameter counts (convnext_small, this session): up_blocks 3.10M / 0.78M / 0.19M, full_up_block 0.019M, out 0.019M, stem 0.015M. Each up-block level is about 3.3 GMAC. The three full-resolution parts are about 2.0-2.6 GMAC each, together about 34 GFLOP, which matches the 32.9 GFLOP 'other spatial convs (decoder, stem)' row in kaggle_1st_place/CLAUDE.md. |
| Core mechanism | (1) Bidirectional cross-scale flow, so coarse levels also receive fine detail. (2) Learned scalar per-input fusion weights. (3) Uniform narrow width with depthwise-separable convs, which is where the FLOP savings come from. (4) Repetition. For a detector, (1) matters because predictions are made at every level. For a single full-resolution output, the last bottom-up pass feeds levels that are never read, so the useful part reduces to a repeated top-down FPN with extra lateral edges. |
| Training objective | Focal loss + smooth-L1 box loss for detection; per-pixel cross-entropy for segmentation. In Erudition the existing column-softmax CE/KD losses and deep supervision would stay. Note that deep supervision is wired to unet.deep_sup_stage_channels (experiments/bilzard/seq_NN_models.py:613), so a BiFPN decoder must expose the same per-level feature list or deep supervision breaks. |
| Decoding | Not applicable (the decoder changes features, not the decode). Our expectation decode over the per-column softmax stays as is. |
| Metric consistent decode | Neutral: a decoder swap leaves the expectation decode, which is Bayes-optimal for RMSE, unchanged. |
| Evidence scope | Whole image (local-to-global via the pyramid). In our case the whole 345x400 cost image of one lateral. It adds no new evidence sources. |
| Prefix conditioning | None. Whatever the input channels already encode passes through unchanged. |
| Propose vs select | Single output. A BiFPN-decoder model could be trained as a separate blend member, but its only source of diversity would be decoder topology. |
| Physical constraints | None. BiFPN imposes no ordering, smoothness or continuity. |

### Data

| Field | Value |
|---|---|
| Dataset | COCO 2017 (detection), Pascal VOC 2012 (segmentation), LUNA-16 (U-Det, 1,186 nodules). None resemble our 16-channel synthetic cost images (85% synthetic training wells, 155 real holdout wells). |
| Synthetic to real | Not addressed by any BiFPN source. The architecture has no bearing on the synthetic-to-real gap. |

### Results

| Field | Value |
|---|---|
| Comparability | Low. Detection AP and 1/4-resolution mIoU gains of 0.3-1.7 points do not translate into a predicted ft-RMSE change on 155 holdout wells. The closest in-repo evidence is internal: the 2:1 resolution rule (2nd-place workspace: one extra halving cost 0.9 ft, two cost 1.7 ft) and the decoder-cost breakdown in kaggle_1st_place/CLAUDE.md. |
| Validation protocol | COCO test-dev / VOC val, single model, single scale. The ablations are clean for detection (same backbone and heads). The segmentation comparison is not controlled for backbone. |

### Ambiguity handling

| Field | Value |
|---|---|
| Repeated layers | Not addressed. Band confusion (choosing or blending the wrong repeated band) comes from ambiguous evidence. The deepest stage (25x10 at convnext_small, 768 ch) already covers the whole image, so extra cross-scale fusion adds little receptive field and no new evidence. |
| Drift | Not addressed. Slow straight-line drift per well is an evidence/prior problem, not a multi-scale fusion problem. |
| Uncertainty and calibration | None. BiFPN has no calibration mechanism. Any change in posterior calibration from a decoder swap would be incidental and must be measured (e.g. per-column NLL / ECE on holdout). |

### Relevance to erudition

| Field | Value |
|---|---|
| Where to use | Only as an alternative decoder inside PretrainedUNet2d (experiments/bilzard/seq_NN_pretrained_unet.py, and the copy in experiments/arch/), replacing the three BilinearResUpBlock2d. to_full_resolution, full_up_block (stem skip) and the out block stay, since the full-resolution TVT axis cannot be removed without breaking the 2:1 rule. Possible secondary use: the decoder for the FastViT-SA12 student, if it measurably cuts student latency at equal RMSE. |
| Error mode targeted | None of the three main modes directly (band confusion ~51%, drift ~25%, broad hump ~13%). At best a small general feature-quality or latency effect. The teammate's 'lower RMSE' claim has no mechanism tied to a known error mode. |
| Gap vs our implementation | What we already do: the decoder is a bilinear U-Net with concat skips at every encoder level plus a full-resolution stem skip. That is a single top-down multi-scale pathway with lateral connections, i.e. FPN-like, but predicting at full resolution. What BiFPN would add: (a) a bottom-up pass, useless for a single full-resolution output unless followed by another top-down pass; (b) learned scalar fusion weights, +0.45 AP in the source paper, well below anything that would clear our +-0.39 ft noise floor by analogy; (c) uniform narrow width with depthwise convs, the real source of its FLOP savings. Point (c) is not specific to BiFPN. Our up-blocks run at the skip width (384/192/96), so a 'narrow U-Net' (up-block width capped at 64-128) gets the same saving with a one-line change. Correction to repo notes: kaggle_1st_place/CLAUDE.md says the decoder parts hold ~0.05M parameters. That holds only for the full-resolution parts (full_up_block + out + stem ~0.05M). The three up-blocks hold ~4.07M (3.10/0.78/0.19M at convnext_small), about 7.6% of the 53.6M model and about 10 GMAC. So there is some width to prune in the mid-level decoder, contrary to that note, but the time is still dominated by resolution. Resolution warning: with stem_stride (2,4) along (MD, TVT), even the finest encoder map is already 1/4 on the TVT (class) axis (100 of 400 levels). EfficientDet-style heads predict at P2/P3 or coarser, so any BiFPN variant that drops the full-resolution stem path would violate the 2:1 rule, which cost 0.9 ft per halving in the 2nd-place workspace. The shipped design ('big encoder + pruned decoder') already reflects the BiFPN lesson of spending little on the neck. |
| Proposed experiment | Only worth running as a cheap falsification test. (1) Implement unet_decoder in {'unet' (default), 'unet_narrow', 'bifpn'} in experiments/bilzard/seq_NN_pretrained_unet.py. 'unet_narrow' is BilinearResUpBlock2d with out_ch = min(skip_ch, W) and W = 64 or 128. 'bifpn' projects each stage map to W with a 1x1 conv, runs 2 BiFPN layers (top-down + bottom-up, fast normalised fusion, depthwise-separable 3x3 + BN + SiLU), then takes the finest node (the 1/2 MD x 1/4 TVT level) into the unchanged to_full_resolution -> full_up_block(stem_skip) -> out path. Expose deep_sup_stage_channels = (W, W, W, emb_dim) so deep supervision in seq_NN_models.py:613 still works. Add a recipe flag in seq_NN_cfg.py; reuse the cnx_tiny recipe otherwise. (2) Timing first, no training: in-model CUDA-event timing at 345x400, fp16, batch 1 for all three decoders, with convnext_tiny and fastvit_sa12. If 'bifpn' is not faster than 'unet_narrow', stop: the fusion topology then brings no efficiency. (3) Accuracy: 60-epoch screen of unet vs unet_narrow vs bifpn at W=64 on convnext_tiny, same seed and data. Compare pooled holdout RMSE with the existing robust_compare against the +-0.39 ft single-run noise floor, and also per-column NLL to catch calibration changes. Go to 150 epochs only if bifpn beats unet_narrow by more than 0.39 ft or matches baseline RMSE with at least 15% lower measured latency. (4) Ablate weighted fusion off (plain sum) to attribute any gain. Control run: the current cnx_tiny_ep150 (4.90 ft test-picked) and a fresh 60-epoch baseline. |
| Implementation effort | Low-medium. About 150 lines for a BiFPN module over 4 levels with odd spatial sizes (200x87, 100x43 ... need size-matched interpolation or pooling in the bottom-up pass, not a fixed stride of 2). 'unet_narrow' is a few lines. Deep-supervision plumbing and recipe flags add a little. |
| Distillation compatibility | Fully compatible: the output posterior format is unchanged, so FastViT-student KD on output posteriors works unmodified. A narrower decoder is a natural extra lever for the student's latency. |
| Error correlation with ours | Expected to be highly correlated with the baseline, since encoder, inputs, loss and decode are identical and only the mid-level fusion topology differs. Little value as a separate blend or gate member; seed averaging would likely give similar diversity. |
| Risk | (1) No mechanism aimed at our dominant errors, which need more evidence or better calibration, not better fusion. (2) The evidence base is detection and 1/4-resolution segmentation. The gains are small (+0.3 to +0.45 AP for the topology and weighting parts) and would sit well inside our +-0.39 ft noise floor. (3) Resolution trap: a detection-style BiFPN predicts at P2-P3; dropping the full-resolution path would cost ~0.9 ft per halving under the 2:1 rule. (4) FLOPs are not wall clock here: depthwise-separable repeated fusion is memory-bound, and EfficientNet-B3 is already slower than convnext_tiny in this repo. A lower-FLOP BiFPN may not be faster. (5) Architecture swaps have repeatedly failed at 60 epochs (axial attention, RAFT refinement, AcfNet target), and MobileOne showed capacity cuts below ~10M hurt. The model is data-limited above ~28M, so decoder capacity changes are unlikely to matter much either way. (6) Deep-supervision heads would have to be re-wired, a source of silent bugs. |

### Other info

- **Uncertain fields (omitted above)**:
  - gpu_cost
  - inference_budget
  - metric_and_performance
  - notes
