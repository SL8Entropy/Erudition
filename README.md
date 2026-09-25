# ROGII Wellbore Geology Prediction (Erudition)

This repository contains the codebase for AI wellbore geology prediction, drawing heavily on top-performing solutions (1st and 2nd place) from the SPWLA 2024 AI competition. It contains different iterations and architectures of sequence-based neural networks (ConvNeXt U-Nets, AnchorCNN, etc.) used to map horizontal well gamma-ray logs to true vertical thickness (TVT).

## Directory Structure

- `/train/`: Contains the training dataset, structured as 773 pairs of `*__horizontal_well.csv` and `*__typewell.csv`.
- `/rogii_champion/`: A working framework based on the 1st place solution (private LB 5.639) plus five iterative enhancements.
- `/kaggle2ndplace/`: Source code and local training tools reproducing the 2nd place weights (private LB 5.802).
- `/kaggle_1st_place/`: Comprehensive repository of the 1st place solution, containing all original competition models and reproduction scripts.
- Root scripts and notebooks (`*.py`, `*.ipynb`): High-level inference and ensemble submission scripts.

## Commands to Train and Test Models

### 1. The `rogii_champion` Framework

The main entry point for the champion framework is `rogii_champion/run.py`. 

- **Training a model:** 
  ```bash
  python rogii_champion/run.py train --variant base --out rogii_champion/runs --folds 0 --num-workers 3
  ```
  *What it does:* Creates a new folder inside `rogii_champion/runs/base/` containing model checkpoints (`.pt`), tensorboard logs, and training metrics for the specified variant.

- **Out-Of-Fold (OOF) Inference / Testing:**
  ```bash
  python rogii_champion/run.py oof --variant base --weights rogii_champion/runs/base --out rogii_champion/runs
  ```
  *What it does:* Runs evaluation on the validation folds, outputting prediction files (e.g., CSV or PQT) into the `rogii_champion/runs/base/` directory for analysis.

- **Inference on Test Set:**
  ```bash
  python rogii_champion/run.py infer --variant base --run-dir rogii_champion/runs/base --out submission.csv
  ```
  *What it does:* Uses the trained checkpoints to generate `submission.csv` in the root directory for Kaggle.

### 2. The `kaggle2ndplace` Framework

The 2nd place solution is trained via `kaggle2ndplace/anchor_train.py`.

- **Training AnchorCNN:**
  ```bash
  python kaggle2ndplace/anchor_train.py --out kaggle2ndplace/runs/dzl_w1 --epochs 120 --tta 8 --score-best
  ```
  *What it does:* Trains the AnchorCNN on 80% of the training wells and scores the remaining 20%. It creates the `kaggle2ndplace/runs/dzl_w1/` folder containing the model checkpoints and `holdout_predictions.pqt` which stores the prediction metrics.

- **Evaluation:**
  ```bash
  python kaggle2ndplace/anchor_eval.py --weights kaggle2ndplace/runs/dzl_w1/model.pt
  ```
  *What it does:* Evaluates an existing checkpoint on the 20% holdout set and prints RMSE metrics.

### 3. The `kaggle_1st_place` Framework

The 1st place reproducible pipeline is managed via `seq_NN_main_reproduce.py`.

- **Training/Reproducing archived models:**
  ```bash
  python kaggle_1st_place/solution/seq_NN_main_reproduce.py --id 0801_V2 --output-dir kaggle_1st_place/solution/results/0801_V2
  ```
  *What it does:* Creates the `kaggle_1st_place/solution/results/0801_V2/` folder containing configuration files (`cfg.pkl`), trained neural network weights, and validation logs for the specific original recipe `0801_V2`.

- **Evaluating Experimental Forks (e.g., Bilzard / Arch):**
  ```bash
  python seq_NN_holdout_eval.py --id 0801_V2 --source-dir experiments/bilzard --cfg-name NAME --epochs 150 --output-dir results/NAME_ep150 --device cuda --offline-timm --batch-size 4 --val-batch-size 4 --grad-accum-steps 4
  ```
  *What it does:* Runs an 80/20 train/holdout evaluation using an edited experimental fork (`experiments/bilzard` or `experiments/arch`), pulling a specific configuration `NAME` from that fork's registry rather than the archived defaults. Output lands in `results/NAME_ep150`.

  **What is `experiments/bilzard`?**
  An edited fork of the `0801_V2` solution originally created to port ideas from the 2nd place (Bilzard) team. It has since become the sandbox registry for most post-competition iterations (knowledge distillation, FastViT). Possible `NAME`s:
  *   `rb_v1_neighbor`: Enables all nine neighboring-well geo-prior channels (16 -> 24 input channels).
  *   `rb_v2_synth`: Enables trajectory mixup, master-series re-skinning, and real residual pasting (synthetic generation).
  *   `rb_v3_tta`: Enables inference averaged over 8 MD column-grid phases.
  *   `rb_v4_all`: Combines `rb_v2_synth` and `rb_v3_tta`.
  *   `pf_v1`: Adds particle filter density probabilities to the inputs.
  *   `cnx_fastvit`: Uses the `fastvit_sa12.apple_dist_in1k` backbone for faster inference.
  *   `cnx_fastvit_kd` / `cnx_fastvit_kd_ens`: Uses FastViT as a student trained via knowledge distillation from a larger teacher (or an ensemble).
  *   `cnx_tiny_kd_ens`: Uses ConvNeXt-tiny as a student distilled from an ensemble teacher.

  **What is `experiments/arch`?**
  A second independent fork of `0801_V2` dedicated exclusively to architectural experiments, primarily targeting the stereo-matching / soft-argmin weakness on bimodal posteriors (repeating rock layers). Possible `NAME`s:
  *   `arch_v1_unimodal`: Replaces the fixed-width alignment term with a confidence head + learned-width unimodal target to stop mode blending.
  *   `arch_v2_axial`: Adds MD-axial attention after ConvNeXt stages 1-3 for better long-range sight.
  *   `arch_v3_raft`: Implements 8 rounds of RAFT-style residual refinement over the cost volume for one-shot decoding.
  *   `arch_v4_lkconv`: Adds parallel long-along-MD depthwise large-kernel convolutions.

  **Knowledge Distillation Implementation Details**
  The `experiments/bilzard` fork implements Knowledge Distillation (KD) to transfer multi-modal depth uncertainty from large models to smaller, faster ones (like `fastvit_sa12`).
  *   **How it works:** Repeating rock layers make the model's depth predictions multi-modal. A hard label discards this ambiguity. By minimizing the Kullback-Leibler (KL) Divergence between the teacher's depth distribution and the student's, the student learns alternative plausible depths. Ensembles are averaged into a mixture distribution.
  *   **Where it lives:** Implemented in `kaggle_1st_place/solution/experiments/bilzard/seq_NN_train.py` via `load_distill_teacher()` (freezes the teacher in `.eval()`) and `distill_alignment_loss()` (computes the KL divergence).
  *   **When it runs:** It runs live during every training step. The frozen teacher evaluates the exact same augmented synthetic `z_shift` batches as the student using `torch.no_grad()`, generating high-quality pseudo-labels on the fly.
  *   **What is added:** Architecturally, nothing is added to the student model. A penalty term (`distill_weight * distill_term`) is simply added to the student's standard Cross-Entropy loss before backpropagation, scaled by the temperature squared.

---

## File Explanations (`.py` Files)

### Root Directory
- **`rogii_exp417_gnll_five_groupkfold_submit.py`**: The main Kaggle submission script. It bundles the required features, coordinates the 5-fold × 5-GroupKFold ensemble inference, and writes out the final predictions.

### `rogii_champion/`
- **`configs.py`**: Contains Python dataclasses defining hyperparameters and architectures for the base model and its five variants.
- **`run.py`**: The master CLI used to trigger training, out-of-fold evaluations, benchmarking, and ensembling.
- **`smoke_test.py`**: Runs a rapid, end-to-end self-test on generated synthetic wells to ensure the pipeline isn't broken.
- **`src/augment.py`**: Implements data augmentations applied during training to improve robustness.
- **`src/benchmark.py`**: Generates benchmark reports and evaluates model performance across runs.
- **`src/canvas.py`**: Constructs the 2D alignment grid (the "canvas") and populates the feature channels for the ConvNeXt U-Net.
- **`src/data.py`**: Handles dataset loading, typewell clustering logic, and sets up cross-validation folds.
- **`src/decode.py`**: Decodes the raw neural network grid predictions back into physical True Vertical Thickness (TVT) trajectories.
- **`src/ensemble.py`**: Contains the logic to ensemble multiple checkpoint predictions.
- **`src/features.py`**: Generates and engineers fold-safe features for each horizontal well.
- **`src/gate.py`**: A gating mechanism/model module used to dynamically combine different sub-models.
- **`src/infer.py`**: Orchestrates the inference pipeline for generating test-set predictions.
- **`src/losses.py`**: Implements custom loss functions, notably smoothed Cross-Entropy, expected-path Huber loss, and gamma-ray penalties.
- **`src/metrics.py`**: Computes evaluation metrics (like RMSE) to track performance.
- **`src/model.py`**: Defines the PyTorch neural architectures, specifically the ConvNeXt U-Net and variant heads.
- **`src/particle_filter.py`**: Implements a particle filter applied over geological slope, level, and GR bias.
- **`src/sibling_ref.py`**: Computes reference gamma-ray values by comparing a well to its sibling laterals and self-prefixes.
- **`src/synth.py`**: Generates physically consistent synthetic horizontal wells for data augmentation.
- **`src/train.py`**: Houses the core PyTorch training loops, optimizers, and gradient updates.
- **`src/xy_neighbor.py`**: Fits structural geological planes and computes spatial neighbourhood statistics.

### `kaggle2ndplace/`
- **`anchor_train.py`**: The training script for the AnchorCNN family, managing the 80/20 train/holdout split.
- **`anchor_eval.py`**: Rescores and evaluates a saved AnchorCNN checkpoint.
- **`anchor_convnext.py`**: Defines a ConvNeXt-based backbone for the anchor models.
- **`anchor_data.py`**: Responsible for loading data and creating synthetic wells specifically for the 2nd place solution.
- **`anchor_dip.py` & `anchor_diptest.py`**: Scripts to calculate and test geological dip (angle/slope) features.
- **`anchor_ensemble.py`**: Ensembles the AnchorCNN with the 1st-place ConvNeXt gating mechanism.
- **`anchor_hmm.py`**: Implements a Hidden Markov Model (HMM) logic for sequence predictions.
- **`anchor_pf.py` & `anchor_pfchan.py`**: Particle Filter logic and the associated input channels tailored for the 2nd place architecture.
- **`anchor_quantize.py`**: Converts and quantizes models (e.g., to int8) to optimize inference speed and resource cost.
- **`anchor_separable.py`**: Defines 1-D separable encoder architectures as a lightweight alternative to full 2D models.
- **`anchor_sibling.py`**: Evaluates sibling-well disagreement to adjust confidence when gamma-ray profiles match.
- **`anchor_summary.py`**: Generates summaries of model architectures and training stats.
- **`anchor_uncertainty.py`**: Models the uncertainty of the geological path predictions.
- **`src/anchor_deploy.py`**: Handles packaging and inference deployment for the AnchorCNN models.
- **`src/gr2tvt_data.py` & `src/gr2tvt_model.py`**: Core data structures and neural models mapping Gamma Ray arrays to TVT predictions.
- **`src/zlayer_mlp.py`**: Implements an auxiliary Multilayer Perceptron (MLP) for z-layer depth predictions.

### `kaggle_1st_place/solution/`
- **`seq_NN_main_reproduce.py`**: CLI script to reproduce legacy training recipes using archived snapshots.
- **`seq_NN_holdout_eval.py`**: Evaluates the trained Sequence-NN on a holdout set, producing scores directly comparable across iterations.
- **`seq_NN_honest_curve.py`**: Calculates validation metrics while preventing target leakage.
- **`seq_NN_rescore.py`**: A utility script to rescore existing predictions against the ground truth.
- **`seq_NN_robust_compare.py`**: Compares prediction outputs (e.g., `holdout_predictions.pqt`) between different model runs to analyze improvements.
- **`generate_train_geo_map.py`**: Pre-computes spatial/geological mapping features for the training data.
- **`verify_cfg.py`**: Validates configuration integrity before kicking off large training jobs.
- **`Original_Comp_Models/*/seq_NN_*.py`**: This directory structure contains multiple archived iterations (e.g., `0719_V1`, `0801_V2`). In each subfolder, identical copies of sequence-NN scripts are kept (`seq_NN_cfg.py`, `seq_NN_train.py`, `seq_NN_models.py`, `seq_NN_dataset.py`, `seq_NN_geo_prior.py`) that lock in the exact configuration, data pipeline, spatial priors, and architecture for that particular historical competition model.
