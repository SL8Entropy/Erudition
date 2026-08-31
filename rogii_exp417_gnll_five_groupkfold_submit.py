"""Create the Eager momentum exp417 5-fold x 5-GroupKFold submission.

The mounted hidden test can change, so every dynamic feature is recomputed in
the Kaggle run. The HMM/PF candidates and 49/31-feature NN inputs share one
exp417 pass; the legacy exp384 HMM/PF pass is excluded. Each split uses its
matching 49/31-feature NN, Adaptive 1D SDF, CNN, and BiLSTM checkpoints.
"""

from __future__ import annotations

import json
import gc
import importlib.util
import os
import subprocess
import sys
import zipfile
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter


def release_cuda_cache() -> None:
    """Release Python and CUDA memory after large temporary computations."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


ARTIFACT_DATASET = "rogii-exp417-eager-gnll-5split-artifacts"
SUPPORT_ARTIFACT_DATASET = "rogii-exp384-hmmpf5p816-nn-softmax-artifacts"
GATE_ARTIFACT_DATASET = ARTIFACT_DATASET
SPLIT_NAMES = ("default", "split101", "split102", "split202", "split303")
ARTIFACT_NAME = "exp417_hmmpf5p759_cut30_rowtype_grnull_downsample23_p025_e100"
DELTA_ARTIFACT_NAME = "exp417_hmmpf5p759_core31_delta_d05wd1e2_e100"
CATBOOST_ARTIFACT_NAME = ""
HMMPF_SOURCE_DATASET = "rogii-exp384-hmmpf5p816-source"
HMMPF_SOURCE_NAME = "exp384_inference.py"
EXP417_SOURCE_DATASET = ARTIFACT_DATASET
EXP417_SOURCE_NAME = "exp417_inference.py"
NEIGHBOR_SURFACE_BLEND_MODULE = "neighbor_surface_blend.py"
NEIGHBOR_SURFACE_BLEND_WEIGHTS = "neighbor_surface_blend_bin_weights.json"
CORRELATION_ARTIFACT_NAME = "correlation_adaptive_5fold"
LSTM_GATE_ARTIFACT_NAME = "exp417_hmm_pf_exp417nn49nn31_softmax5_lstm"
PF4_TABLE_NAME = "hmm_pfbase_foldsafe_multigrid_components_773w.parquet"
EXP202_LIKPF_TABLE_PATTERN = "exp202_likpf_sb{seed}_test.parquet"
HMMPF_MULTIGRID_TABLE_NAME = "hmmpf_nu1_multigrid_test.parquet"
SOURCE_DATASET = "rogii-gridhmm-pfbase-source"

PF_PARTICLE_COUNT = 500
PF_SEED_COUNT = 256
PF4_SEED_COUNT = 128
PF_BACKEND = "torch"
PF_TORCH_SEED_CHUNK_SIZE = 256
PF4_TORCH_SEED_CHUNK_SIZE = 128
PF_JAX_SEED_CHUNK_SIZE = 256
PF4_JAX_SEED_CHUNK_SIZE = 128
PF_PASS_NAME = "mom9995rn001t60b025"
PF4_PASS_NAMES = (
    "mom9995rn001t60b025",
    "mom1rn0005",
    "mom9995rn001t60b025off5r2s10",
    "win8tnu4",
)
PF_SCALEAVG_SCALES = (2.0, 3.5, 5.0)
SELECTOR_SCALES = (3.0, 5.0, 8.0, 12.0)
SELECTOR_N_EVAL_THRESHOLD = 4840.0
SELECTOR_Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)
SELECTOR_BIN_VARIANTS = {
    0: "pf_scale_5_hold_0.2",
    1: "pf_scale_3_hold_0.15",
    2: "pf_scale_12_beam_0.2_hold_0.15",
    3: "pf_scale_5_hold_0.15",
    4: "pf_scale_5_beam_0.05_hold_0.05",
    5: "pf_scale_12_beam_0.2_hold_0.05",
}
SELECTOR_GLOBAL_VARIANT = "pf_scale_8_hold_0.2"
GRID_POSITION_STEP_05 = 0.5
GRID_POSITION_STEP_025 = 0.25
GRID_RATE_STEP = 0.005
GRID_RATE_MARGIN = 0.08
GRID_MAX_RATE_BINS = 41
GRID_BACKWARD_DTYPE = torch.float32
GRID_POSTERIOR_GUARD_THRESHOLD = 1e-20
GRID_BETA_RESET_THRESHOLD = 1e-30
HMM_SMOOTH05_INTERCEPT = 0.38089257609157
HMM_SMOOTH05_SLOPE = -0.3340671645882282
HMM_SMOOTH025_INTERCEPT = 0.19979711950562692
HMM_SMOOTH025_SLOPE = 0.04445938997836881
HMM_FILT05_INTERCEPT = 0.1849790347297589
HMM_FILT05_SLOPE = -0.16786202292850785
BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2),
    (10, 8.0, 64.0, 2),
    (8, 35.0, 220.0, 1),
    (10, 14.0, 90.0, 5),
    (20, 4.0, 36.0, 3),
    (12, 12.0, 100.0, 3),
    (15, 25.0, 180.0, 2),
    (20, 30.0, 200.0, 2),
    (15, 10.0, 80.0, 4),
    (25, 6.0, 50.0, 3),
    (10, 40.0, 300.0, 1),
    (12, 18.0, 120.0, 5),
    (30, 8.0, 70.0, 2),
    (10, 50.0, 400.0, 0),
]
HMMPF_OUTPUT_COLUMNS = [
    "well_id",
    "row_index",
    "hmmpf_pf_smooth",
    "hmmpf_pf_filt",
    "hmmpf_scaleavg_smooth",
    "hmmpf_scaleavg_filt",
    "hmmpf_grid05_smooth_smooth",
    "hmmpf_grid05_smooth_filt",
    "hmmpf_grid025_smooth_smooth",
    "hmmpf_grid025_smooth_filt",
    "hmmpf_grid05_filt_smooth",
    "hmmpf_grid05_filt_filt",
    "hmmpf_multigrid_smooth",
    "hmmpf_multigrid_filt",
]
PF4_GATE_OUTPUT_COLUMNS: list[str] = []
DYNAMIC_OUTPUT_COLUMNS = HMMPF_OUTPUT_COLUMNS + PF4_GATE_OUTPUT_COLUMNS
GATE_CANDIDATE_NAMES = ["hmm", "pf", "nn", "sdf1d", "delta"]
GATE_CANDIDATE_COLUMNS = ["hmm_tvt", "pf2_tvt", "nn", "sdf1d", "delta"]
BLEND_METRICS_NAME = "cnn_lstm_blend_metrics.json"
PHYSICS_KNOWN_ROWS = 256
PHYSICS_LOWER_QUANTILE = 0.01
PHYSICS_UPPER_QUANTILE = 0.99
PHYSICS_PADDING = 0.05
PHYSICS_CORRECTION_WEIGHT = 0.5
PHYSICS_HARD_RATE_LIMIT = 0.25
PHYSICS_MAX_ABS_CORRECTION = 10.0


class LSTMGate(torch.nn.Module):
    """Candidate予測を凸結合するBiLSTM softmax gate。"""

    def __init__(self, n_features: int, n_candidates: int, hidden_channels: int, num_layers: int) -> None:
        """Initialize the gate.

        Args:
            n_features: 入力特徴数。
            n_candidates: 候補数。
            hidden_channels: LSTM hidden size。
            num_layers: LSTM layer数。
        """

        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(n_features, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.SiLU(),
        )
        self.lstm = torch.nn.LSTM(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            num_layers=max(1, num_layers),
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_channels * 2),
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_channels, n_candidates),
        )
        self.base_logits = torch.nn.Parameter(torch.zeros(n_candidates))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return candidate logits for ``(batch, length, features)`` input."""

        hidden, _ = self.lstm(self.proj(features))
        return self.head(hidden) + self.base_logits


class ResidualGateBlock(torch.nn.Module):
    """系列長を保つCNN gate用residual block。"""

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        """Residual blockを初期化する。

        Args:
            channels: 入出力channel数。
            kernel_size: Conv1d kernel size。
            dilation: Dilation rate。
        """

        super().__init__()
        padding = dilation * (kernel_size // 2)
        groups = 8 if channels % 8 == 0 else 1
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            torch.nn.GroupNorm(groups, channels),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.0),
            torch.nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            torch.nn.GroupNorm(groups, channels),
            torch.nn.Dropout(0.0),
        )
        self.act = torch.nn.SiLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Residual blockを適用する。"""

        return self.act(features + self.net(features))


class CNNGate(torch.nn.Module):
    """Candidate予測を凸結合するCNN softmax gate。"""

    def __init__(
        self,
        n_features: int,
        n_candidates: int,
        hidden_channels: int,
        num_blocks: int,
        kernel_size: int,
    ) -> None:
        """CNN gateを初期化する。

        Args:
            n_features: 入力特徴数。
            n_candidates: 候補数。
            hidden_channels: Hidden channel数。
            num_blocks: Residual block数。
            kernel_size: Conv1d kernel size。
        """

        super().__init__()
        self.input = torch.nn.Sequential(
            torch.nn.Conv1d(n_features, hidden_channels, kernel_size=1),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.0),
        )
        self.blocks = torch.nn.Sequential(
            *[
                ResidualGateBlock(hidden_channels, kernel_size, dilation=2**idx)
                for idx in range(max(1, num_blocks))
            ]
        )
        self.head = torch.nn.Conv1d(hidden_channels, n_candidates, kernel_size=1)
        self.base_logits = torch.nn.Parameter(torch.zeros(n_candidates))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """``(batch, length, features)``からcandidate logitsを返す。"""

        hidden = self.blocks(self.input(features.transpose(1, 2)))
        return self.head(hidden).transpose(1, 2) + self.base_logits


def ensure_source_import_path() -> Path:
    """Find the bundled PF/GridHMM source Dataset and add it to ``sys.path``.

    Returns:
        Source root containing ``src/pf_gpu`` and ``src/pf_grid``.
    """

    input_root = Path("/kaggle/input")
    candidates = [
        Path.cwd(),
        Path("kaggle_work/datasets") / SOURCE_DATASET,
        input_root / SOURCE_DATASET,
        input_root / "datasets" / SOURCE_DATASET,
        input_root / "datasets" / "tereka" / SOURCE_DATASET,
    ]
    if input_root.is_dir():
        candidates.extend(path for path in input_root.iterdir() if path.name == SOURCE_DATASET)
        candidates.extend(path.parent.parent.parent for path in input_root.rglob("pf_grid/hmm_torch.py"))
    for candidate in dict.fromkeys(candidates):
        if (candidate / "src" / "pf_gpu" / "runner.py").is_file() and (
            candidate / "src" / "pf_grid" / "hmm_torch.py"
        ).is_file():
            candidate_str = str(candidate.resolve())
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate
    raise FileNotFoundError(f"Bundled PF/GridHMM source Dataset was not found: {SOURCE_DATASET}")


SOURCE_ROOT = ensure_source_import_path()

from src.pf_gpu.common import BackendConfig, PASS_SPECS, prepare_well_arrays  # noqa: E402
from src.pf_gpu.runner import run_variant_ensembles  # noqa: E402
from src.pf_grid.hmm_numpy import GridHMMConfig  # noqa: E402
from src.pf_grid.hmm_torch import TorchGridHMMConfig, run_grid_hmm_torch  # noqa: E402


def find_competition_root() -> Path:
    """Find the mounted ROGII competition input directory.

    Returns:
        Directory containing ``test`` and ``sample_submission.csv``.
    """

    candidates = [
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("data"),
    ]
    for root in candidates:
        if (root / "test").is_dir() and (root / "sample_submission.csv").is_file():
            return root
    raise FileNotFoundError("ROGII competition input was not found.")


def find_hmmpf_source() -> Path:
    """Find the bundled exp303 HMMPF nu1 inference source.

    Returns:
        Path to ``exp303_inference.py``.
    """

    candidates = [
        Path("/kaggle/input") / HMMPF_SOURCE_DATASET / HMMPF_SOURCE_NAME,
        Path("/kaggle/input") / "datasets" / "tereka" / HMMPF_SOURCE_DATASET / HMMPF_SOURCE_NAME,
        Path("kaggle_work/datasets") / HMMPF_SOURCE_DATASET / HMMPF_SOURCE_NAME,
    ]
    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        candidates.extend(input_root.rglob(HMMPF_SOURCE_NAME))
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"HMMPF source was not found: {HMMPF_SOURCE_NAME}")


def find_exp417_source() -> Path:
    """Find the mounted exp417 self-reference HMM source.

    Returns:
        Path to ``exp417_inference.py``.
    """

    candidates = [
        Path("/kaggle/input") / EXP417_SOURCE_DATASET / EXP417_SOURCE_NAME,
        Path("/kaggle/input") / "datasets" / "tereka" / EXP417_SOURCE_DATASET / EXP417_SOURCE_NAME,
        Path("kaggle_work/datasets") / EXP417_SOURCE_DATASET / EXP417_SOURCE_NAME,
    ]
    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        candidates.extend(input_root.rglob(EXP417_SOURCE_NAME))
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"exp417 source was not found: {EXP417_SOURCE_NAME}")


def find_hmm_variant_blend_source() -> Path | None:
    """5split artifactに同梱された3系統HMM推論sourceを探す。

    Returns:
        helper path。3split artifactでは存在しないため``None``。
    """

    name = "predict_exp417_hmm_variant_blend.py"
    candidates = [
        Path("/kaggle/input")
        / EXP417_SOURCE_DATASET
        / "bundle"
        / "variant_runtime"
        / "src"
        / "inference"
        / name,
        Path("kaggle_work/datasets")
        / EXP417_SOURCE_DATASET
        / "bundle"
        / "variant_runtime"
        / "src"
        / "inference"
        / name,
    ]
    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        candidates.extend(input_root.rglob(name))
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file() and "variant_runtime" in candidate.parts:
            return candidate
    return None


def run_exp303_hmmpf(competition_root: Path) -> pd.DataFrame:
    """Run exp303 PF2 plus 3D bias-HMM inference on mounted test wells.

    Args:
        competition_root: Mounted competition data root.

    Returns:
        ``id``、HMMPF、PF2、3D HMMの絶対TVTを持つDataFrame。
    """

    source_path = find_hmmpf_source()
    os.environ["ROGII_COMPETITION_ROOT"] = str(competition_root)
    namespace = {"__name__": "hmmpf_exp303_inference", "__file__": str(source_path)}
    code = compile(source_path.read_text(encoding="utf-8"), str(source_path), "exec")
    exec(code, namespace)
    submission = namespace.get("sub")
    if not isinstance(submission, pd.DataFrame) or not {"id", "tvt"}.issubset(submission.columns):
        raise ValueError("exp303 source did not create an id/tvt submission")
    test_partial = namespace.get("test_partial")
    hmm_frame = namespace.get("hmm_df")
    seed_bases = namespace.get("LIK_SEED_BASES")
    if (
        not isinstance(test_partial, pd.DataFrame)
        or not isinstance(hmm_frame, pd.DataFrame)
        or not isinstance(seed_bases, (list, tuple))
    ):
        raise ValueError("exp384 source did not expose its component predictions")

    ids = test_partial["id"].astype(str).to_numpy()
    last_tvt = test_partial["last_TVT"].to_numpy(dtype=np.float64)
    pf2_distances: list[np.ndarray] = []
    for seed_base in seed_bases:
        maximum = (
            pd.read_parquet(f"lik_f015_sb{seed_base}.parquet")
            .set_index("id")
            .loc[ids, "pf_lik_d"]
            .to_numpy(dtype=np.float64)
        )
        soft = (
            pd.read_parquet(f"lik_soft_sb{seed_base}.parquet")
            .set_index("id")
            .loc[ids, "pf_lik_d"]
            .to_numpy(dtype=np.float64)
        )
        # 学習用exp384表と同じ0.6/0.4のPF2定義を使用する。
        pf2_distances.append(0.6 * maximum + 0.4 * soft)
    pf2_distance = np.mean(np.stack(pf2_distances, axis=0), axis=0)
    hmm_distance = hmm_frame.loc[ids, "hmm3_d_anc"].to_numpy(dtype=np.float64)
    hmm_distance = np.where(np.isfinite(hmm_distance), hmm_distance, pf2_distance)

    components = pd.DataFrame(
        {
            "id": ids,
            "pf2_tvt": last_tvt + pf2_distance,
            "hmm_tvt": last_tvt + hmm_distance,
        }
    )
    result = (
        submission[["id", "tvt"]]
        .rename(columns={"tvt": "hmmpf_tvt"})
        .merge(components, on="id", how="left", validate="one_to_one")
    )
    value_columns = ["hmmpf_tvt", "pf2_tvt", "hmm_tvt"]
    if result["id"].duplicated().any() or not np.isfinite(result[value_columns]).all().all():
        raise ValueError("exp303 HMMPF prediction is invalid")
    return result


def register_variant_runtime_source(variant_source: Path) -> Path:
    """追加HMM runtimeの``src`` namespaceをimport対象へ登録する。

    Args:
        variant_source: ``variant_runtime/src/inference``配下のhelper path。

    Returns:
        ``src`` directoryを含むruntime root。
    """

    runtime_src = variant_source.parents[1]
    runtime_root = runtime_src.parent
    runtime_root_str = str(runtime_root)
    if runtime_root_str not in sys.path:
        sys.path.insert(0, runtime_root_str)
    # PF source側で``src`` namespaceが既にimport済みでも追加runtimeを探索させる。
    src_package = sys.modules.get("src")
    if src_package is not None:
        package_path = getattr(src_package, "__path__", None)
        if package_path is None:
            raise ImportError("loaded src module is not a package")
        runtime_src_str = str(runtime_src)
        if runtime_src_str not in package_path:
            package_path.append(runtime_src_str)
    # ``src.pf_grid``等が通常packageとして既に読込済みの場合も探索pathを拡張する。
    for module_name, loaded_module in tuple(sys.modules.items()):
        if not module_name.startswith("src."):
            continue
        package_path = getattr(loaded_module, "__path__", None)
        if package_path is None:
            continue
        relative_parts = module_name.split(".")[1:]
        runtime_package = runtime_src.joinpath(*relative_parts)
        runtime_package_str = str(runtime_package)
        if runtime_package.is_dir() and runtime_package_str not in package_path:
            package_path.append(runtime_package_str)
    return runtime_root


def run_exp417_components(competition_root: Path) -> pd.DataFrame:
    """Run exp417 on the currently mounted test and return HMM/PF components.

    Args:
        competition_root: Mounted competition data root.

    Returns:
        Row-keyed exp417 HMM and PF2 absolute TVT table.
    """

    source_path = find_exp417_source()
    os.environ["ROGII_COMPETITION_ROOT"] = str(competition_root)
    namespace = {"__name__": "hmmpf_exp417_inference", "__file__": str(source_path)}
    code = compile(source_path.read_text(encoding="utf-8"), str(source_path), "exec")
    exec(code, namespace)
    components = namespace.get("exp417_components")
    required = ["well_id", "row_index", "hmm_tvt", "pf2_tvt"]
    if not isinstance(components, pd.DataFrame) or not set(required).issubset(components.columns):
        raise ValueError("exp417 source did not expose its HMM/PF component table")
    component_columns = required + (
        ["last_tvt"] if "last_tvt" in components.columns else []
    )
    result = components[component_columns].copy()
    values = result[["hmm_tvt", "pf2_tvt"]].to_numpy(dtype=np.float64)
    if result[["well_id", "row_index"]].duplicated().any() or not np.isfinite(values).all():
        raise ValueError("exp417 HMM/PF component prediction is invalid")
    variant_source = find_hmm_variant_blend_source()
    if variant_source is not None:
        register_variant_runtime_source(variant_source)
        spec = importlib.util.spec_from_file_location(
            "exp417_hmm_variant_blend_runtime",
            variant_source,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load HMM variant helper: {variant_source}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.predict_and_blend_test(competition_root, result)
        print("Applied exp417/local-DTW/bin0.0625 HMM deploy blend", flush=True)
    return result


def formation_rate(tvt: np.ndarray, md: np.ndarray, z: np.ndarray) -> np.ndarray:
    """地層面rate ``d(TVT+Z)/dMD`` を計算する。

    Args:
        tvt: TVT path。
        md: MD path。
        z: Z path。

    Returns:
        隣接行間の地層面rate。
    """

    delta_md = np.diff(np.asarray(md, dtype=np.float64))
    if (delta_md <= 0.0).any() or not np.isfinite(delta_md).all():
        raise ValueError("MD must be finite and strictly increasing")
    return np.diff(np.asarray(tvt, dtype=np.float64) + np.asarray(z, dtype=np.float64)) / delta_md


def estimate_physics_rate_bounds(known: pd.DataFrame) -> tuple[float, float]:
    """既知prefixから物理射影用rate範囲を推定する。

    Args:
        known: 有効なMD、Z、TVT_inputを持つ既知prefix末尾。

    Returns:
        paddingと絶対上限を適用したrate下限・上限。
    """

    rates = formation_rate(
        known["TVT_input"].to_numpy(),
        known["MD"].to_numpy(),
        known["Z"].to_numpy(),
    )
    rates = rates[np.isfinite(rates)]
    if rates.size == 0:
        return -PHYSICS_HARD_RATE_LIMIT, PHYSICS_HARD_RATE_LIMIT
    lower = max(
        float(np.quantile(rates, PHYSICS_LOWER_QUANTILE) - PHYSICS_PADDING),
        -PHYSICS_HARD_RATE_LIMIT,
    )
    upper = min(
        float(np.quantile(rates, PHYSICS_UPPER_QUANTILE) + PHYSICS_PADDING),
        PHYSICS_HARD_RATE_LIMIT,
    )
    if lower > upper:
        center = float(np.clip(np.median(rates), -PHYSICS_HARD_RATE_LIMIT, PHYSICS_HARD_RATE_LIMIT))
        return center, center
    return lower, upper


def project_hmmpf_path(
    prediction: np.ndarray,
    hidden_md: np.ndarray,
    hidden_z: np.ndarray,
    known: pd.DataFrame,
    rate_lower: float,
    rate_upper: float,
) -> np.ndarray:
    """HMMPF pathへ保守的な地層dip射影を適用する。

    Args:
        prediction: hidden区間のraw HMMPF TVT。
        hidden_md: hidden区間のMD。
        hidden_z: hidden区間のZ。
        known: 既知prefix末尾。
        rate_lower: 許容rate下限。
        rate_upper: 許容rate上限。

    Returns:
        50%反映・10 ft上限を適用した物理補正HMMPF TVT。
    """

    anchor_tvt = float(known["TVT_input"].iloc[-1])
    anchor_md = float(known["MD"].iloc[-1])
    anchor_z = float(known["Z"].iloc[-1])
    pred = np.asarray(prediction, dtype=np.float64)
    full_md = np.concatenate(([anchor_md], np.asarray(hidden_md, dtype=np.float64)))
    full_z = np.concatenate(([anchor_z], np.asarray(hidden_z, dtype=np.float64)))
    full_tvt = np.concatenate(([anchor_tvt], pred))
    projected_rate = np.clip(formation_rate(full_tvt, full_md, full_z), rate_lower, rate_upper)
    projected_level = np.empty(len(pred) + 1, dtype=np.float64)
    projected_level[0] = anchor_tvt + anchor_z
    projected_level[1:] = projected_level[0] + np.cumsum(projected_rate * np.diff(full_md))
    fully_projected = projected_level[1:] - full_z[1:]
    correction = PHYSICS_CORRECTION_WEIGHT * (fully_projected - pred)
    correction = np.clip(correction, -PHYSICS_MAX_ABS_CORRECTION, PHYSICS_MAX_ABS_CORRECTION)
    return pred + correction


def add_physics_hmmpf_candidate(
    competition_root: Path,
    table: pd.DataFrame,
    diagnostics_path: Path | None = None,
) -> pd.DataFrame:
    """raw HMMPFを維持したまま物理補正gate候補を追加する。

    Args:
        competition_root: mounted competition data root。
        table: hidden行のHMMPF/MultiGridテーブル。
        diagnostics_path: 井戸別補正診断の保存先。未指定なら保存しない。

    Returns:
        ``hmmpf_physics_tvt``を追加したcopy。
    """

    result = table.copy()
    result["hmmpf_physics_tvt"] = np.nan
    diagnostics: list[dict[str, float | int | str]] = []
    for well_id, group in result.groupby("well_id", sort=False):
        ordered = group.sort_values("row_index")
        row_index = ordered["row_index"].to_numpy(dtype=np.int64)
        horizontal = pd.read_csv(
            competition_root / "test" / f"{well_id}__horizontal_well.csv",
            usecols=["MD", "Z", "TVT_input"],
        )
        known = (
            horizontal.iloc[: int(row_index[0])]
            .dropna(subset=["TVT_input", "MD", "Z"])
            .tail(PHYSICS_KNOWN_ROWS)
        )
        if len(known) < 2:
            raise ValueError(f"At least two known rows are required for well={well_id}")
        hidden = horizontal.iloc[row_index]
        lower, upper = estimate_physics_rate_bounds(known)
        raw = ordered["hmmpf_tvt"].to_numpy(dtype=np.float64)
        refined = project_hmmpf_path(
            raw,
            hidden["MD"].to_numpy(),
            hidden["Z"].to_numpy(),
            known,
            lower,
            upper,
        )
        result.loc[ordered.index, "hmmpf_physics_tvt"] = refined
        diagnostics.append(
            {
                "well_id": str(well_id),
                "rate_lower": lower,
                "rate_upper": upper,
                "mean_abs_correction": float(np.mean(np.abs(refined - raw))),
                "max_abs_correction": float(np.max(np.abs(refined - raw))),
            }
        )
    if not np.isfinite(result["hmmpf_physics_tvt"].to_numpy(dtype=np.float64)).all():
        raise ValueError("Physics HMMPF candidate contains non-finite values")
    if diagnostics_path is not None:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(diagnostics).to_csv(diagnostics_path, index=False)
    return result


def build_exp417_nn_feature_table(
    exp417_components: pd.DataFrame,
    dynamic_table_path: Path,
    output_path: Path,
) -> Path:
    """Build the NN feature table from one exp417 HMM/PF result.

    Args:
        exp417_components: Row-aligned exp417 HMM/PF predictions.
        dynamic_table_path: Dynamic PF/GridHMM component parquet.
        output_path: Destination parquet path.

    Returns:
        Path to the validated feature table.
    """

    if os.environ.get("ROGII_REUSE_DYNAMIC") == "1" and output_path.is_file():
        print(f"Reusing local HMMPF/MultiGrid table: {output_path}", flush=True)
        return output_path

    dynamic = pd.read_parquet(
        dynamic_table_path,
        columns=["well_id", "row_index", "hmmpf_multigrid_smooth"],
    )
    required = ["well_id", "row_index", "hmm_tvt", "pf2_tvt"]
    if not set(required).issubset(exp417_components.columns):
        raise ValueError("exp417 components are missing NN feature columns")
    component_columns = required + (
        ["hmmpf_tvt"] if "hmmpf_tvt" in exp417_components.columns else []
    )
    components = exp417_components[component_columns].copy()
    merged = components.merge(
        dynamic,
        on=["well_id", "row_index"],
        how="left",
        validate="one_to_one",
    )
    table = merged.rename(columns={"hmmpf_multigrid_smooth": "multigrid_tvt"})
    if "hmmpf_tvt" not in table.columns:
        # 旧3splitは0.30 F015 + 0.20 Soft + 0.50 HMMを維持する。
        table["hmmpf_tvt"] = 0.5 * (
            table["pf2_tvt"].to_numpy(dtype=np.float64)
            + table["hmm_tvt"].to_numpy(dtype=np.float64)
        )
    table = table[
        [
            "well_id",
            "row_index",
            "hmmpf_tvt",
            "pf2_tvt",
            "hmm_tvt",
            "multigrid_tvt",
        ]
    ]
    value_cols = [
        "hmmpf_tvt",
        "pf2_tvt",
        "hmm_tvt",
        "multigrid_tvt",
    ]
    if len(table) != len(components) or table[["well_id", "row_index"]].isna().any().any():
        raise ValueError("exp417 NN feature keys do not match components")
    if table[["well_id", "row_index"]].duplicated().any() or not np.isfinite(table[value_cols]).all().all():
        raise ValueError("exp417 NN feature values are invalid")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_path, index=False)
    print(f"Saved exp417 NN feature table: {output_path} rows={len(table)}", flush=True)
    return output_path


def is_artifact_root(path: Path) -> bool:
    """Return whether a path contains the exp202 HMM Grid DStd artifact.

    Args:
        path: Candidate artifact directory.

    Returns:
        True when required model, config, feature, and source files exist.
    """

    return (
        path.is_dir()
        and (path / "baseline.py").is_file()
        and (path / "config.json").is_file()
        and (path / "feature_names.json").is_file()
        and (path / "src" / "features" / "precompute_exp202_likpf_table.py").is_file()
        and len(list((path / "models").glob("fold_*.pt"))) == 5
    )


def artifact_search_roots() -> list[Path]:
    """Build possible Kaggle and local artifact dataset roots.

    Returns:
        Candidate roots that may contain ``ARTIFACT_NAME``.
    """

    kaggle_input = Path("/kaggle/input")
    roots = [
        kaggle_input / ARTIFACT_DATASET,
        kaggle_input / "datasets" / ARTIFACT_DATASET,
        kaggle_input / "datasets" / "tereka" / ARTIFACT_DATASET,
        Path("kaggle_work/datasets") / ARTIFACT_DATASET,
    ]
    dataset_parent = kaggle_input / "datasets"
    if dataset_parent.is_dir():
        # Kaggleの配置差異に備えて owner/dataset 形式も探索する。
        for owner_root in dataset_parent.iterdir():
            if owner_root.is_dir():
                roots.append(owner_root / ARTIFACT_DATASET)
    return list(dict.fromkeys(roots))


def support_artifact_search_roots() -> list[Path]:
    """Build roots for the existing Core31/SDF support Dataset.

    Returns:
        Candidate mounted support Dataset roots.
    """

    kaggle_input = Path("/kaggle/input")
    roots = [
        kaggle_input / SUPPORT_ARTIFACT_DATASET,
        kaggle_input / "datasets" / SUPPORT_ARTIFACT_DATASET,
        kaggle_input / "datasets" / "tereka" / SUPPORT_ARTIFACT_DATASET,
        Path("kaggle_work/datasets") / SUPPORT_ARTIFACT_DATASET,
    ]
    dataset_parent = kaggle_input / "datasets"
    if dataset_parent.is_dir():
        for owner_root in dataset_parent.iterdir():
            if owner_root.is_dir():
                roots.append(owner_root / SUPPORT_ARTIFACT_DATASET)
    return list(dict.fromkeys(roots))


def candidate_artifact_roots(dataset_root: Path) -> list[Path]:
    """List artifact candidates under one dataset root.

    Args:
        dataset_root: Mounted or local dataset directory.

    Returns:
        Candidate artifact directories.
    """

    candidates = [dataset_root, dataset_root / ARTIFACT_NAME]
    if dataset_root.is_dir():
        candidates.extend(dataset_root.rglob(ARTIFACT_NAME))
        candidates.extend(path.parent for path in dataset_root.rglob("baseline.py"))
    return candidates


def find_artifact_root() -> Path:
    """Find or extract the mounted HMM/PFBase artifact directory.

    Returns:
        Directory containing ``baseline.py`` and ``models/fold_*.pt``.
    """

    dataset_roots = artifact_search_roots()
    for dataset_root in dataset_roots:
        for candidate in candidate_artifact_roots(dataset_root):
            if is_artifact_root(candidate):
                return candidate

    for dataset_root in dataset_roots:
        if not dataset_root.is_dir():
            continue
        zip_paths = sorted(dataset_root.rglob("*.zip"))
        if not zip_paths:
            continue
        # Datasetを複数zipへ分割しても、artifact同士の相対配置を維持する。
        extract_root = (
            Path("/kaggle/working/rogii_hmmpfbase_fsmgcomp")
            / ARTIFACT_DATASET
        )
        extract_root.mkdir(parents=True, exist_ok=True)
        for zip_path in zip_paths:
            print(f"Extracting {zip_path} to {extract_root}", flush=True)
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(extract_root)
        for candidate in candidate_artifact_roots(extract_root):
            if is_artifact_root(candidate):
                return candidate

    kaggle_input = Path("/kaggle/input")
    entries = sorted(path.name for path in kaggle_input.iterdir()) if kaggle_input.is_dir() else []
    raise FileNotFoundError(
        "HMM/PFBase artifact root was not found: "
        f"dataset={ARTIFACT_DATASET} artifact={ARTIFACT_NAME} "
        f"kaggle_input_entries={entries[:30]}"
    )


def find_named_nn_artifact_root(artifact_root: Path, artifact_name: str) -> Path:
    """分割Datasetを含めて指定名のNN artifactを探す。

    Args:
        artifact_root: 先に発見した47特徴NN artifact。
        artifact_name: 探索するNN artifact名。

    Returns:
        ``is_artifact_root``を満たすartifact directory。
    """

    candidates = [artifact_root.parent / artifact_name]
    for dataset_root in [*artifact_search_roots(), *support_artifact_search_roots()]:
        candidates.append(dataset_root / artifact_name)
        if dataset_root.is_dir():
            candidates.extend(dataset_root.rglob(artifact_name))
    for candidate in dict.fromkeys(candidates):
        if is_artifact_root(candidate):
            return candidate
    raise FileNotFoundError(f"NN artifact root was not found: {artifact_name}")


def find_catboost_artifact_root(artifact_root: Path) -> Path:
    """分割Datasetを含めてCatBoost artifactを探す。

    Args:
        artifact_root: 先に発見した47特徴NN artifact。

    Returns:
        CatBoost推論器と15モデルを含むdirectory。
    """

    candidates = [artifact_root.parent / CATBOOST_ARTIFACT_NAME]
    for dataset_root in artifact_search_roots():
        candidates.append(dataset_root / CATBOOST_ARTIFACT_NAME)
        if dataset_root.is_dir():
            candidates.extend(dataset_root.rglob(CATBOOST_ARTIFACT_NAME))
    for candidate in dict.fromkeys(candidates):
        predictor = candidate / "predict_exp356_hmmpf_catboost.py"
        models = list((candidate / "models").glob("seed*_fold*.cbm"))
        if predictor.is_file() and len(models) == 15:
            return candidate
    raise FileNotFoundError(
        f"CatBoost artifact root was not found: {CATBOOST_ARTIFACT_NAME}"
    )


def find_correlation_artifact_root(artifact_root: Path) -> Path:
    """Adaptive 1D SDFのcode・config・5fold checkpointを探す。

    Args:
        artifact_root: 47特徴NN artifact root。

    Returns:
        Correlation artifact directory。
    """

    candidates = [
        artifact_root.parent / CORRELATION_ARTIFACT_NAME,
        Path("kaggle_work/datasets")
        / SUPPORT_ARTIFACT_DATASET
        / CORRELATION_ARTIFACT_NAME,
    ]
    for dataset_root in [*artifact_search_roots(), *support_artifact_search_roots()]:
        candidates.append(dataset_root / CORRELATION_ARTIFACT_NAME)
        if dataset_root.is_dir():
            candidates.extend(dataset_root.rglob(CORRELATION_ARTIFACT_NAME))
    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        candidates.extend(
            path.parent
            for path in input_root.rglob(
                f"{CORRELATION_ARTIFACT_NAME}/config.json"
            )
        )
    for candidate in dict.fromkeys(candidates):
        code_path = (
            candidate
            / "code"
            / "src"
            / "train"
            / "train_correlation_volume_1d.py"
        )
        if (
            code_path.is_file()
            and (candidate / "config.json").is_file()
            and all(
                (candidate / "models" / f"fold{fold}_model.pt").is_file()
                for fold in range(5)
            )
        ):
            return candidate
    raise FileNotFoundError("Adaptive 1D SDF artifact root was not found.")


def run_correlation_submission(
    competition_root: Path,
    correlation_root: Path,
    output_path: Path,
    sdf_table_path: Path,
) -> pd.DataFrame:
    """Adaptive 1D SDF 5foldを推論しsubmissionとNN特徴tableを作る。

    Args:
        competition_root: Competition data root。
        correlation_root: Adaptive 1D SDF artifact root。
        output_path: 1D SDF submission path。
        sdf_table_path: NNへ渡す``sdf_tvt`` table path。

    Returns:
        Sample submission順の1D SDF予測。
    """

    import src as source_package  # noqa: PLC0415

    correlation_src = str((correlation_root / "code" / "src").resolve())
    if correlation_src not in source_package.__path__:
        source_package.__path__ = [correlation_src, *list(source_package.__path__)]
    from src.train.train_correlation_volume_1d import (  # noqa: PLC0415
        load_correlation_checkpoint,
        load_correlation_config,
        load_correlation_inference_sample,
        predict_correlation_ensemble,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_correlation_config(
        correlation_root / "config.json",
        data_root=str(competition_root),
        device=str(device),
    )
    models = [
        load_correlation_checkpoint(
            correlation_root / "models" / f"fold{fold}_model.pt",
            config,
            device,
        )
        for fold in range(5)
    ]
    sample_index = load_sample_index(competition_root)
    prediction_parts: list[pd.DataFrame] = []
    for well_id in sample_index["well_id"].drop_duplicates():
        sample = load_correlation_inference_sample(
            competition_root / "test" / f"{well_id}__horizontal_well.csv",
            competition_root / "test" / f"{well_id}__typewell.csv",
            config,
        )
        prediction = predict_correlation_ensemble(models, sample, device)
        prediction_parts.append(
            pd.DataFrame(
                {
                    "well_id": well_id,
                    "row_index": sample["hidden_index"],
                    "sdf_tvt": prediction,
                }
            )
        )
    del models
    release_cuda_cache()
    predictions = pd.concat(prediction_parts, ignore_index=True)
    if bool(predictions.duplicated(["well_id", "row_index"]).any()):
        raise ValueError("1D SDF inference produced duplicate row keys.")
    aligned = sample_index.merge(
        predictions,
        on=["well_id", "row_index"],
        how="left",
        validate="one_to_one",
    )
    if aligned["sdf_tvt"].isna().any():
        raise ValueError("1D SDF inference contains missing predictions.")
    submission = aligned[["id"]].copy()
    submission["tvt"] = aligned["sdf_tvt"].to_numpy(dtype=np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    sdf_table_path.parent.mkdir(parents=True, exist_ok=True)
    aligned[["well_id", "row_index", "sdf_tvt"]].to_parquet(
        sdf_table_path,
        index=False,
    )
    return submission


def is_gate_artifact_root(path: Path) -> bool:
    """Return whether a path contains the LSTM gate artifact.

    Args:
        path: Candidate gate artifact directory.

    Returns:
        True when required metrics and five fold checkpoints exist.
    """

    return (
        path.is_dir()
        and (path / "metrics.json").is_file()
        and (path / "models").is_dir()
        and len(list((path / "models").glob("fold_*.pt"))) == 5
    )


def find_gate_artifact_root(artifact_root: Path, gate_name: str) -> Path:
    """Find one CNN/LSTM gate artifact.

    Args:
        artifact_root: HMM/PFBase NN artifact root.

    Returns:
        gate_name: ``softmax_cnn``または``softmax_lstm``。

    Returns:
        Directory containing ``models/fold_*.pt`` and ``metrics.json``.
    """

    input_root = Path("/kaggle/input")
    dataset_roots = [
        *artifact_search_roots(),
        input_root / GATE_ARTIFACT_DATASET,
        input_root / "datasets" / GATE_ARTIFACT_DATASET,
        input_root / "datasets" / "tereka" / GATE_ARTIFACT_DATASET,
        Path("kaggle_work/datasets") / GATE_ARTIFACT_DATASET,
        input_root / ARTIFACT_DATASET,
        input_root / "datasets" / ARTIFACT_DATASET,
        input_root / "datasets" / "tereka" / ARTIFACT_DATASET,
        Path("kaggle_work/datasets") / ARTIFACT_DATASET,
    ]
    dataset_parent = input_root / "datasets"
    if dataset_parent.is_dir():
        for owner_root in dataset_parent.iterdir():
            if owner_root.is_dir():
                dataset_roots.append(owner_root / ARTIFACT_DATASET)
    candidates = [artifact_root.parent / "gates" / gate_name]
    if input_root.is_dir():
        candidates.extend(path.parent for path in input_root.rglob(f"{gate_name}/metrics.json"))
    for dataset_root in dataset_roots:
        candidates.extend([dataset_root / "gates" / gate_name, dataset_root / gate_name])
        if dataset_root.is_dir():
            candidates.extend(dataset_root.rglob(f"gates/{gate_name}"))
    for candidate in dict.fromkeys(candidates):
        if is_gate_artifact_root(candidate):
            return candidate
    raise FileNotFoundError(f"Gate artifact root was not found: {gate_name}")


def load_well(root: Path, wid: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one test horizontal/typewell pair.

    Args:
        root: Competition data root.
        wid: Well id.

    Returns:
        Horizontal well and typewell frames.
    """

    hw = pd.read_csv(root / "test" / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(root / "test" / f"{wid}__typewell.csv")
    return hw, tw


def selector_well_code(hw: pd.DataFrame) -> tuple[int, str, float, float]:
    """Choose the existing target-free PF selector variant for one well."""

    eval_mask = hw["TVT_input"].isna().to_numpy()
    n_eval = float(eval_mask.sum())
    z_eval = hw.loc[eval_mask, "Z"].to_numpy(dtype=float)
    z_span = float(np.nanmax(z_eval) - np.nanmin(z_eval)) if len(z_eval) else 0.0
    n_bin = int(n_eval > SELECTOR_N_EVAL_THRESHOLD)
    z_bin = int(np.searchsorted(SELECTOR_Z_SPAN_THRESHOLDS, z_span, side="right"))
    code = n_bin + 2 * z_bin
    return code, SELECTOR_BIN_VARIANTS.get(code, SELECTOR_GLOBAL_VARIANT), n_eval, z_span


def parse_selector_variant(name: str) -> tuple[float, float, float]:
    """Parse a PF selector variant name into scale, beam weight, and hold weight."""

    parts = name.split("_")
    scale = float(parts[2])
    beam_weight = float(parts[parts.index("beam") + 1]) if "beam" in parts else 0.0
    hold_weight = float(parts[parts.index("hold") + 1]) if "hold" in parts else 0.0
    return scale, beam_weight, hold_weight


def apply_selector_variant(
    name: str,
    pf_by_scale: dict[str, np.ndarray],
    tvt_beam: np.ndarray,
    last_known_tvt: float,
) -> np.ndarray:
    """Apply the target-free selector policy to PF scale curves."""

    scale, beam_weight, hold_weight = parse_selector_variant(name)
    base = pf_by_scale.get(f"pf_scale_{scale:g}", pf_by_scale.get("pf_scale_8"))
    if base is None:
        raise KeyError(f"PF scale for selector {name!r} is missing.")
    pred = (1.0 - beam_weight) * base + beam_weight * tvt_beam
    return (1.0 - hold_weight) * pred + hold_weight * last_known_tvt


def apply_selector_variant_to_base_curve(
    name: str,
    base_curve: np.ndarray,
    tvt_beam: np.ndarray,
    last_known_tvt: float,
) -> np.ndarray:
    """Apply only the selector beam/hold weights to one PF base curve."""

    _, beam_weight, hold_weight = parse_selector_variant(name)
    pred = (1.0 - beam_weight) * base_curve + beam_weight * tvt_beam
    return (1.0 - hold_weight) * pred + hold_weight * last_known_tvt


def beam_search(
    hgr: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    last_tvt: float,
    beam_size: int,
    move_cost: float,
    error_scale: float,
    radius: int,
) -> np.ndarray:
    """Run a small GR-matching beam search used by the PF selector."""

    n_rows = len(hgr)
    n_tvt = len(tw_tvt)
    if n_rows == 0:
        return np.array([last_tvt], dtype=float)
    if radius > 0 and n_rows > max(3, 2 * radius + 1):
        window = min(2 * radius + 1, n_rows if n_rows % 2 == 1 else n_rows - 1)
        smooth_gr = savgol_filter(hgr, window, min(2, window - 1))
    else:
        smooth_gr = hgr.copy()

    start_idx = int(np.argmin(np.abs(tw_tvt - last_tvt)))
    moves = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    move_costs = move_cost * np.array([2.0, 1.0, 0.0, 1.0, 2.0])
    beam_idx = np.full(beam_size, start_idx, dtype=np.int64)
    beam_cost = np.full(beam_size, np.inf)
    beam_cost[0] = 0.0
    beam_count = 1
    parents: list[np.ndarray] = []
    states: list[np.ndarray] = []

    for gr_value in smooth_gr:
        cand_idx: list[int] = []
        cand_cost: list[float] = []
        cand_parent: list[int] = []
        for parent in range(beam_count):
            base_idx = int(beam_idx[parent])
            base_cost = float(beam_cost[parent])
            for move, move_penalty in zip(moves, move_costs):
                next_idx = int(np.clip(base_idx + int(move), 0, n_tvt - 1))
                emission = ((float(gr_value) - float(tw_gr[next_idx])) ** 2) / error_scale
                cand_idx.append(next_idx)
                cand_cost.append(base_cost + float(move_penalty) + emission)
                cand_parent.append(parent)
        order = np.argsort(cand_cost)[:beam_size]
        states.append(np.asarray(cand_idx, dtype=np.int64)[order])
        parents.append(np.asarray(cand_parent, dtype=np.int64)[order])
        beam_idx = states[-1]
        beam_cost = np.asarray(cand_cost, dtype=float)[order]
        beam_count = len(order)

    best = int(np.argmin(beam_cost[:beam_count]))
    path = np.empty(n_rows, dtype=np.int64)
    for step in range(n_rows - 1, -1, -1):
        path[step] = states[step][best]
        best = parents[step][best]
    return tw_tvt[path]


def run_beam_ensemble(hw: pd.DataFrame, tw: pd.DataFrame) -> np.ndarray:
    """Compute the mean beam-search TVT curve for one well."""

    eval_mask = hw["TVT_input"].isna().to_numpy()
    tw_sorted = tw.sort_values("TVT")
    tw_tvt = tw_sorted["TVT"].to_numpy(dtype=float)
    tw_gr = tw_sorted["GR"].interpolate(limit_direction="both").fillna(tw_sorted["GR"].mean()).to_numpy(dtype=float)
    known = hw.loc[~eval_mask]
    last_known = float(known["TVT_input"].iloc[-1]) if len(known) else float(tw_tvt[0])
    gr_interp = hw["GR"].interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr)))
    hgr = gr_interp.to_numpy(dtype=float)[eval_mask]
    curves = [
        beam_search(hgr, tw_tvt, tw_gr, last_known, beam_size, move_cost, error_scale, radius)
        for beam_size, move_cost, error_scale, radius in BEAM_CONFIGS
    ]
    eval_curve = np.mean(np.stack(curves, axis=0), axis=0)
    out = hw["TVT_input"].fillna(last_known).to_numpy(dtype=float)
    out[np.where(eval_mask)[0]] = eval_curve
    return out


def compute_pf_selector_curve(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    scales: tuple[float, ...],
    average_scales: bool,
) -> tuple[np.ndarray, str, str]:
    """Compute selector-applied PF curve for one well.

    Args:
        hw: Horizontal well.
        tw: Typewell.
        scales: PF scales to run.
        average_scales: When true, average selector-applied curves for all scales.

    Returns:
        Curve, selector variant, and backend used.
    """

    arrays = prepare_well_arrays(hw, tw)
    if arrays is None:
        fallback = hw["TVT_input"].interpolate(limit_direction="both").to_numpy(dtype=float)
        return fallback, "no_eval", "none"
    _, selector_variant, _, _ = selector_well_code(hw)
    try:
        tvt_beam = run_beam_ensemble(hw, tw)
    except Exception as exc:  # noqa: BLE001
        print(f"  Beam failed, fallback to hold: {exc}", flush=True)
        tvt_beam = np.full(len(hw), arrays.last_tvt, dtype=float)
    ensembles = run_variant_ensembles(
        arrays,
        scales=scales,
        n_particles=PF_PARTICLE_COUNT,
        n_seeds=PF_SEED_COUNT,
        spec=PASS_SPECS[PF_PASS_NAME],
        backend_config=BackendConfig(
            backend=PF_BACKEND,
            jax_seed_chunk_size=PF_JAX_SEED_CHUNK_SIZE,
            torch_seed_chunk_size=PF_TORCH_SEED_CHUNK_SIZE,
        ),
    )
    if average_scales:
        curves = [
            apply_selector_variant_to_base_curve(
                selector_variant,
                ensembles.by_scale[True][f"pf_scale_{scale:g}"],
                tvt_beam,
                arrays.last_tvt,
            )
            for scale in scales
        ]
        return np.mean(np.stack(curves, axis=0), axis=0), selector_variant, ensembles.backend_used
    return apply_selector_variant(selector_variant, ensembles.by_scale[True], tvt_beam, arrays.last_tvt), selector_variant, ensembles.backend_used


def compute_grid_hmm_curves(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    position_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GridHMM smoothed and filtered curves for one well."""

    arrays = prepare_well_arrays(hw, tw)
    if arrays is None:
        fallback = hw["TVT_input"].interpolate(limit_direction="both").to_numpy(dtype=float)
        return fallback, fallback
    result = run_grid_hmm_torch(
        arrays,
        PASS_SPECS[PF_PASS_NAME],
        TorchGridHMMConfig(
            grid=GridHMMConfig(
                position_step=position_step,
                rate_step=GRID_RATE_STEP,
                rate_margin=GRID_RATE_MARGIN,
                max_rate_bins=GRID_MAX_RATE_BINS,
            ),
            device="cuda" if torch.cuda.is_available() else "cpu",
            backward_dtype=GRID_BACKWARD_DTYPE,
            posterior_guard_threshold=GRID_POSTERIOR_GUARD_THRESHOLD,
            beta_reset_threshold=GRID_BETA_RESET_THRESHOLD,
        ),
    )
    return result.smoothed, result.filtered


def compute_pf4_gate_curves(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Compute PF4 np500/ns128 smoothed curves used by the LSTM gate.

    Args:
        hw: Horizontal well.
        tw: Typewell.

    Returns:
        Candidate curve dictionary and diagnostics.
    """

    arrays = prepare_well_arrays(hw, tw)
    if arrays is None:
        fallback = hw["TVT_input"].interpolate(limit_direction="both").to_numpy(dtype=float)
        return {column: fallback for column in PF4_GATE_OUTPUT_COLUMNS}, {"pf4_backend": "none"}
    _, selector_variant, _, _ = selector_well_code(hw)
    try:
        tvt_beam = run_beam_ensemble(hw, tw)
    except Exception as exc:  # noqa: BLE001
        print(f"  PF4 beam failed, fallback to hold: {exc}", flush=True)
        tvt_beam = np.full(len(hw), arrays.last_tvt, dtype=float)

    output_map = {
        "mom9995rn001t60b025": "pf4_mom9995_smooth",
        "mom1rn0005": "pf4_mom1_smooth",
        "mom9995rn001t60b025off5r2s10": "pf4_tnu4_smooth",
        "win8tnu4": "pf4_win8tnu4_smooth",
    }
    curves: dict[str, np.ndarray] = {}
    backends: dict[str, str] = {}
    logliks: dict[str, float] = {}
    for pass_name in PF4_PASS_NAMES:
        ensembles = run_variant_ensembles(
            arrays,
            scales=SELECTOR_SCALES,
            n_particles=PF_PARTICLE_COUNT,
            n_seeds=PF4_SEED_COUNT,
            spec=PASS_SPECS[pass_name],
            backend_config=BackendConfig(
                backend=PF_BACKEND,
                jax_seed_chunk_size=PF4_JAX_SEED_CHUNK_SIZE,
                torch_seed_chunk_size=PF4_TORCH_SEED_CHUNK_SIZE,
            ),
        )
        curves[output_map[pass_name]] = apply_selector_variant(
            selector_variant,
            ensembles.by_scale[True],
            tvt_beam,
            arrays.last_tvt,
        )
        backends[pass_name] = ensembles.backend_used
        logliks[pass_name] = float(ensembles.mean_loglik)
    diagnostics = {
        "pf4_selector_variant": selector_variant,
        "pf4_backends": backends,
        "pf4_logliks": logliks,
    }
    return curves, diagnostics


def target_fraction_curve(hw: pd.DataFrame) -> np.ndarray:
    """Return 0..1 target fraction on evaluation rows and 0 elsewhere."""

    eval_index = np.where(hw["TVT_input"].isna().to_numpy())[0]
    frac = np.zeros(len(hw), dtype=float)
    if len(eval_index) > 1:
        frac[eval_index] = np.arange(len(eval_index), dtype=float) / float(len(eval_index) - 1)
    return frac


def multigrid_curve(
    scaleavg_curve: np.ndarray,
    grid05_smooth: np.ndarray,
    grid025_smooth: np.ndarray,
    grid05_filt: np.ndarray,
    frac: np.ndarray,
) -> np.ndarray:
    """Apply the public submit multigrid linear4 blend to component curves."""

    smooth05 = np.clip(HMM_SMOOTH05_INTERCEPT + HMM_SMOOTH05_SLOPE * frac, 0.0, None)
    smooth025 = np.clip(HMM_SMOOTH025_INTERCEPT + HMM_SMOOTH025_SLOPE * frac, 0.0, None)
    filt05 = np.clip(HMM_FILT05_INTERCEPT + HMM_FILT05_SLOPE * frac, 0.0, None)
    grid_sum = smooth05 + smooth025 + filt05
    overflow = grid_sum > 1.0
    if overflow.any():
        smooth05[overflow] /= grid_sum[overflow]
        smooth025[overflow] /= grid_sum[overflow]
        filt05[overflow] /= grid_sum[overflow]
        grid_sum = smooth05 + smooth025 + filt05
    scaleavg = 1.0 - grid_sum
    return (
        scaleavg * scaleavg_curve
        + smooth05 * grid05_smooth
        + smooth025 * grid025_smooth
        + filt05 * grid05_filt
    )


def load_sample_index(competition_root: Path) -> pd.DataFrame:
    """Load sample submission ids and split them into well id and row index.

    Args:
        competition_root: Competition data root.

    Returns:
        DataFrame with ``id``, ``well_id``, and ``row_index``.
    """

    sample = pd.read_csv(competition_root / "sample_submission.csv")
    sample_parts = sample["id"].astype(str).str.rsplit("_", n=1, expand=True)
    return pd.DataFrame(
        {
            "id": sample["id"].astype(str),
            "well_id": sample_parts[0].astype(str),
            "row_index": sample_parts[1].astype(int),
        }
    )


def read_numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Read a numeric horizontal-well column with interpolation fallback.

    Args:
        frame: Horizontal-well frame.
        column: Column name.

    Returns:
        Numeric series without NaNs.
    """

    if column not in frame:
        return pd.Series(np.zeros(len(frame), dtype=float), index=frame.index)
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.interpolate(limit_direction="both").ffill().bfill().fillna(0.0).astype(float)


def diff_from_last_known(values: np.ndarray, last_value: float) -> np.ndarray:
    """Return first differences, comparing the first row with the last visible value."""

    return np.diff(values, prepend=last_value)


def load_horizontal_context_for_gate(
    competition_root: Path,
    well_id: str,
    row_indices: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Build the same row context features used during gate OOF training.

    Args:
        competition_root: Competition data root.
        well_id: Well id.
        row_indices: Sample row indices.

    Returns:
        Gate context arrays and ``last_known_tvt``.
    """

    hw, _ = load_well(competition_root, well_id)
    known = hw[hw["TVT_input"].notna()]
    if len(known) == 0:
        last_known_tvt = float(pd.to_numeric(hw["TVT_input"], errors="coerce").fillna(0.0).iloc[0])
        last_md = float(hw["MD"].iloc[0])
        last_z = float(hw["Z"].iloc[0])
        last_x = float(read_numeric_column(hw, "X").iloc[0])
        last_y = float(read_numeric_column(hw, "Y").iloc[0])
        last_tvt_input = last_known_tvt
    else:
        last = known.iloc[-1]
        last_known_tvt = float(last["TVT_input"])
        last_md = float(last["MD"])
        last_z = float(last["Z"])
        last_x = float(read_numeric_column(hw, "X").loc[last.name])
        last_y = float(read_numeric_column(hw, "Y").loc[last.name])
        last_tvt_input = float(last["TVT_input"])

    rows = hw.iloc[row_indices.astype(int)]
    gr_filled = read_numeric_column(hw, "GR")
    gr_values = gr_filled.iloc[row_indices.astype(int)].to_numpy(dtype=np.float64)
    gr_missing = rows["GR"].isna().to_numpy(dtype=np.float64)
    x_values = read_numeric_column(hw, "X").iloc[row_indices.astype(int)].to_numpy(dtype=np.float64)
    y_values = read_numeric_column(hw, "Y").iloc[row_indices.astype(int)].to_numpy(dtype=np.float64)
    tvt_input_filled = (
        pd.to_numeric(hw["TVT_input"], errors="coerce")
        .ffill()
        .bfill()
        .fillna(last_known_tvt)
        .astype(float)
    )
    tvt_input_values = tvt_input_filled.iloc[row_indices.astype(int)].to_numpy(dtype=np.float64)
    md_values = rows["MD"].to_numpy(dtype=np.float64)
    z_values = rows["Z"].to_numpy(dtype=np.float64)
    md_since = np.maximum(md_values - last_md, 0.0)
    z_since = z_values - last_z
    md_delta = np.diff(md_values, prepend=last_md)
    x_delta = diff_from_last_known(x_values, last_x)
    y_delta = diff_from_last_known(y_values, last_y)
    z_delta = np.diff(z_values, prepend=last_z)
    xyz_delta_norm = np.sqrt(np.square(x_delta) + np.square(y_delta) + np.square(z_delta))
    gr_delta = np.diff(gr_values, prepend=gr_values[0])
    tvt_input_delta = diff_from_last_known(tvt_input_values, last_tvt_input)
    return {
        "last_known_tvt": last_known_tvt,
        "md_since": md_since,
        "z_since": z_since,
        "md_delta": md_delta,
        "z_delta": z_delta,
        "xyz_delta_norm": xyz_delta_norm,
        "gr_values": gr_values,
        "gr_delta": gr_delta,
        "gr_abs_delta": np.abs(gr_delta),
        "gr_missing": gr_missing,
        "tvt_input_delta": tvt_input_delta,
        "position_frac": np.linspace(0.0, 1.0, len(row_indices), dtype=np.float64),
    }


def build_gate_features(
    competition_root: Path,
    gate_frame: pd.DataFrame,
    candidate_cols: list[str],
    extra_feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build padded gate feature and candidate tensors for all test wells.

    Args:
        competition_root: Competition data root.
        gate_frame: Row-aligned candidate frame.
        candidate_cols: Candidate prediction column names.
        extra_feature_cols: Candidateではなくgate判断だけに使う追加特徴列。

    Returns:
        Features, candidates, and row ids in tensor order.
    """

    feature_batches: list[np.ndarray] = []
    candidate_batches: list[np.ndarray] = []
    id_batches: list[np.ndarray] = []
    max_len = 0
    for well_id, group in gate_frame.groupby("well_id", sort=False):
        group = group.sort_values("row_index").copy()
        row_index = group["row_index"].to_numpy(dtype=np.int64)
        candidates = group[candidate_cols].to_numpy(dtype=np.float64)
        context = load_horizontal_context_for_gate(competition_root, str(well_id), row_index)
        last_known_tvt = float(context["last_known_tvt"])
        candidate_from_last = candidates - last_known_tvt
        candidate_centered = candidates - candidates.mean(axis=1, keepdims=True)
        candidate_steps = np.diff(candidates, axis=0, prepend=np.full((1, candidates.shape[1]), last_known_tvt))
        candidate_std = candidates.std(axis=1, keepdims=True)
        candidate_range = (candidates.max(axis=1) - candidates.min(axis=1)).reshape(-1, 1)
        context_features = np.column_stack(
            [
                context["md_since"],
                context["z_since"],
                context["md_delta"],
                context["z_delta"],
                context["xyz_delta_norm"],
                context["gr_values"],
                context["gr_delta"],
                context["gr_abs_delta"],
                context["gr_missing"],
                context["tvt_input_delta"],
                context["position_frac"],
            ]
        )
        feature_parts: list[np.ndarray] = [
            candidate_from_last,
            candidate_centered,
            candidate_steps,
            candidate_std,
            candidate_range,
            context_features,
        ]
        if extra_feature_cols:
            feature_parts.append(
                group[extra_feature_cols].to_numpy(dtype=np.float64)
            )
        features = np.column_stack(
            feature_parts
        ).astype(np.float32)
        feature_batches.append(features)
        candidate_batches.append(candidates.astype(np.float32))
        id_batches.append(group["id"].astype(str).to_numpy())
        max_len = max(max_len, len(group))

    n_wells = len(feature_batches)
    n_features = feature_batches[0].shape[1]
    n_candidates = candidate_batches[0].shape[1]
    features_tensor = np.zeros((n_wells, max_len, n_features), dtype=np.float32)
    candidates_tensor = np.zeros((n_wells, max_len, n_candidates), dtype=np.float32)
    ids_ordered: list[str] = []
    for idx, (features, candidates, ids) in enumerate(zip(feature_batches, candidate_batches, id_batches)):
        length = len(features)
        features_tensor[idx, :length] = features
        candidates_tensor[idx, :length] = candidates
        ids_ordered.extend(ids.tolist())
    return features_tensor, candidates_tensor, ids_ordered


def load_gate_models(
    gate_root: Path,
    device: torch.device,
) -> list[tuple[torch.nn.Module, np.ndarray, np.ndarray]]:
    """Load fold gate checkpoints.

    Args:
        gate_root: Gate artifact root.
        device: Inference device.

    Returns:
        Tuples of model, feature mean, and feature std.
    """

    models: list[tuple[torch.nn.Module, np.ndarray, np.ndarray]] = []
    for checkpoint_path in sorted((gate_root / "models").glob("fold_*.pt")):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        candidate_names = [str(name) for name in checkpoint.get("candidate_names", [])]
        if candidate_names != GATE_CANDIDATE_NAMES:
            raise ValueError(
                "Gate checkpoint candidate order differs from submission order: "
                f"checkpoint={candidate_names} submission={GATE_CANDIDATE_NAMES} "
                f"path={checkpoint_path}"
            )
        if checkpoint["model_type"] == "cnn":
            model: torch.nn.Module = CNNGate(
                n_features=int(checkpoint["n_features"]),
                n_candidates=int(checkpoint["n_candidates"]),
                hidden_channels=int(checkpoint["hidden_channels"]),
                num_blocks=int(checkpoint["num_blocks"]),
                kernel_size=int(checkpoint["kernel_size"]),
            )
        elif checkpoint["model_type"] == "lstm":
            model = LSTMGate(
                n_features=int(checkpoint["n_features"]),
                n_candidates=int(checkpoint["n_candidates"]),
                hidden_channels=int(checkpoint["hidden_channels"]),
                num_layers=int(checkpoint["num_blocks"]),
            )
        else:
            raise ValueError(f"Unsupported gate model: {checkpoint['model_type']}")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        models.append(
            (
                model,
                np.asarray(checkpoint["feature_mean"], dtype=np.float32),
                np.asarray(checkpoint["feature_std"], dtype=np.float32),
            )
        )
    if len(models) != 5:
        raise ValueError(f"Expected 5 gate checkpoints, found {len(models)} in {gate_root}")
    return models


def load_cnn_lstm_blend_weights(artifact_root: Path) -> tuple[float, float]:
    """OOF全体でfitしたCNN/LSTM推論重みをartifactから読む。

    Args:
        artifact_root: NN artifact root。

    Returns:
        CNN重みとLSTM重み。

    Raises:
        FileNotFoundError: blend metricsが存在しない場合。
        ValueError: 重みが有限な凸結合になっていない場合。
    """

    candidates = [artifact_root.parent / BLEND_METRICS_NAME]
    for dataset_root in artifact_search_roots():
        candidates.append(dataset_root / BLEND_METRICS_NAME)
        if dataset_root.is_dir():
            candidates.extend(dataset_root.rglob(BLEND_METRICS_NAME))
    metrics_path = next(
        (path for path in dict.fromkeys(candidates) if path.is_file()),
        None,
    )
    if metrics_path is None:
        raise FileNotFoundError(BLEND_METRICS_NAME)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    cnn_weight = float(metrics.get("cnn_lstm_global_weight", float("nan")))
    lstm_weight = float(metrics.get("lstm_global_weight", float("nan")))
    if (
        not np.isfinite(cnn_weight)
        or not np.isfinite(lstm_weight)
        or not 0.0 <= cnn_weight <= 1.0
        or not 0.0 <= lstm_weight <= 1.0
        or not np.isclose(cnn_weight + lstm_weight, 1.0, atol=1e-9)
    ):
        raise ValueError(
            "CNN/LSTM blend weights must be a finite convex combination: "
            f"cnn={cnn_weight} lstm={lstm_weight} path={metrics_path}"
        )
    return cnn_weight, lstm_weight


def run_softmax_gate_submission(
    competition_root: Path,
    gate_root: Path,
    dynamic_table_path: Path,
    nn_submission_path: Path,
    sdf_submission_path: Path,
    delta_submission_path: Path,
    output_path: Path,
    nn_details_path: Path | None = None,
) -> pd.DataFrame:
    """Run one fold-averaged five-candidate SoftMax gate and write submission.

    Args:
        competition_root: Competition data root.
        gate_root: Gate artifact root.
        dynamic_table_path: Physics HMMPF/MultiGrid table path.
        nn_submission_path: Submission produced by HMM/PFBase NN.
        sdf_submission_path: Adaptive 1D SDF submission path。
        delta_submission_path: 31特徴raw-delta NN submission path。
        output_path: Final submission destination.
        nn_details_path: Gaussian NLL NN49の平均・Std詳細CSV。

    Returns:
        Final submission frame.
    """

    sample_index = load_sample_index(competition_root)
    nn_submission = pd.read_csv(nn_submission_path).rename(columns={"tvt": "nn", "tvd": "nn", "TVT": "nn"})
    if "nn" not in nn_submission.columns:
        value_col = [col for col in nn_submission.columns if col != "id"][0]
        nn_submission = nn_submission.rename(columns={value_col: "nn"})
    sdf_submission = pd.read_csv(sdf_submission_path).rename(
        columns={"tvt": "sdf1d", "tvd": "sdf1d", "TVT": "sdf1d"}
    )
    if "sdf1d" not in sdf_submission.columns:
        value_col = [col for col in sdf_submission.columns if col != "id"][0]
        sdf_submission = sdf_submission.rename(columns={value_col: "sdf1d"})
    delta_submission = pd.read_csv(delta_submission_path).rename(
        columns={"tvt": "delta", "tvd": "delta", "TVT": "delta"}
    )
    if "delta" not in delta_submission.columns:
        value_col = [col for col in delta_submission.columns if col != "id"][0]
        delta_submission = delta_submission.rename(columns={value_col: "delta"})
    dynamic = pd.read_parquet(dynamic_table_path)
    gate_frame = sample_index.merge(nn_submission[["id", "nn"]], on="id", how="left", validate="one_to_one")
    extra_feature_cols: list[str] = []
    if nn_details_path is not None:
        nn_details = pd.read_csv(nn_details_path, usecols=["id", "pred_tvt_std"])
        nn_details = nn_details.rename(columns={"pred_tvt_std": "nn49_std"})
        gate_frame = gate_frame.merge(
            nn_details,
            on="id",
            how="left",
            validate="one_to_one",
        )
        extra_feature_cols.append("nn49_std")
    gate_frame = gate_frame.merge(
        sdf_submission[["id", "sdf1d"]],
        on="id",
        how="left",
        validate="one_to_one",
    )
    gate_frame = gate_frame.merge(
        delta_submission[["id", "delta"]],
        on="id",
        how="left",
        validate="one_to_one",
    )
    gate_frame = gate_frame.merge(dynamic, on=["well_id", "row_index"], how="left", validate="one_to_one")
    del nn_submission, sdf_submission, delta_submission, dynamic
    release_cuda_cache()
    candidate_cols = GATE_CANDIDATE_COLUMNS
    required_gate_cols = [*candidate_cols, *extra_feature_cols]
    if gate_frame[required_gate_cols].isna().any().any():
        raise ValueError("Gate candidate frame contains NaN values.")
    well_lengths = [int(length) for length in gate_frame.groupby("well_id", sort=False).size().to_list()]
    features, candidates, ids_ordered = build_gate_features(
        competition_root,
        gate_frame,
        candidate_cols,
        extra_feature_cols,
    )
    del gate_frame
    release_cuda_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_gate_models(gate_root, device)
    prediction_sum = np.zeros((features.shape[0], features.shape[1]), dtype=np.float32)
    weight_sum = np.zeros((features.shape[0], features.shape[1], len(candidate_cols)), dtype=np.float32)
    # 全fold modelを先にGPUへ読み込み、井戸ごとに全modelを連続適用する。
    with torch.no_grad():
        for well_idx, length in enumerate(well_lengths):
            well_candidates = torch.from_numpy(candidates[well_idx : well_idx + 1, :length]).to(device)
            for model, mean, std in models:
                norm_features = (
                    (features[well_idx : well_idx + 1, :length] - mean.reshape(1, 1, -1))
                    / std.reshape(1, 1, -1)
                ).astype(np.float32)
                logits = model(torch.from_numpy(norm_features).to(device))
                weights = torch.softmax(logits, dim=-1)
                pred = torch.sum(weights * well_candidates, dim=-1)
                prediction_sum[well_idx, :length] += pred.cpu().numpy()[0]
                weight_sum[well_idx, :length] += weights.cpu().numpy()[0]
                del norm_features, logits, weights, pred
            del well_candidates
    pred_mean = prediction_sum / float(len(models))
    weight_mean = weight_sum / float(len(models))
    del models, features, candidates
    release_cuda_cache()

    rows: list[pd.DataFrame] = []
    valid_weight_rows: list[np.ndarray] = []
    cursor = 0
    for well_idx, length in enumerate(well_lengths):
        rows.append(
            pd.DataFrame(
                {
                    "id": ids_ordered[cursor : cursor + length],
                    "tvt": pred_mean[well_idx, :length].astype(float),
                }
            )
        )
        valid_weight_rows.append(weight_mean[well_idx, :length])
        cursor += length
    final = pd.concat(rows, ignore_index=True)
    final = sample_index[["id"]].merge(final, on="id", how="left", validate="one_to_one")
    if final["tvt"].isna().any():
        raise ValueError("Final SoftMax gate submission contains NaN predictions.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_path, index=False)

    diagnostics = {
        "rows": int(len(final)),
        "wells": int(sample_index["well_id"].nunique()),
        "candidate_columns": candidate_cols,
        "weight_mean": {
            name: float(np.concatenate(valid_weight_rows, axis=0)[:, idx].mean())
            for idx, name in enumerate(candidate_cols)
        },
        "nn_submission_path": str(nn_submission_path),
        "sdf_submission_path": str(sdf_submission_path),
        "delta_submission_path": str(delta_submission_path),
        "dynamic_table_path": str(dynamic_table_path),
        "gate_root": str(gate_root),
    }
    diagnostics_name = f"{gate_root.name}_submission_diagnostics.json"
    (output_path.parent / diagnostics_name).write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, indent=2), flush=True)
    del (
        rows,
        valid_weight_rows,
        prediction_sum,
        weight_sum,
        pred_mean,
        weight_mean,
        sample_index,
        ids_ordered,
        diagnostics,
        well_lengths,
    )
    release_cuda_cache()
    return final


def compute_dynamic_hmmpf_rows(
    competition_root: Path,
    sample_index: pd.DataFrame,
    well_ids: list[str],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Compute dynamic HMM/PFBase rows for selected wells.

    Args:
        competition_root: Competition data root.
        sample_index: Sample submission index frame.
        well_ids: Well ids assigned to this worker.

    Returns:
        Dynamic feature table shard and diagnostics.
    """

    rows: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    worker_device = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    for well_id in well_ids:
        target_rows = sample_index[sample_index["well_id"] == str(well_id)]
        if target_rows.empty:
            raise ValueError(f"Assigned well has no sample rows: {well_id}")
        print(
            f"Computing dynamic HMM/PFBase features for {well_id} on CUDA_VISIBLE_DEVICES={worker_device}",
            flush=True,
        )
        hw, tw = load_well(competition_root, str(well_id))
        pf_curve, selector_variant, pf_backend = compute_pf_selector_curve(
            hw,
            tw,
            scales=SELECTOR_SCALES,
            average_scales=False,
        )
        scaleavg_curve, scaleavg_selector, scaleavg_backend = compute_pf_selector_curve(
            hw,
            tw,
            scales=PF_SCALEAVG_SCALES,
            average_scales=True,
        )
        grid05_smooth, grid05_filt = compute_grid_hmm_curves(hw, tw, GRID_POSITION_STEP_05)
        grid025_smooth, _ = compute_grid_hmm_curves(hw, tw, GRID_POSITION_STEP_025)
        frac = target_fraction_curve(hw)
        multigrid = multigrid_curve(scaleavg_curve, grid05_smooth, grid025_smooth, grid05_filt, frac)
        pf4_curves: dict[str, np.ndarray] = {}
        pf4_diagnostics: dict[str, object] = {}
        if PF4_GATE_OUTPUT_COLUMNS:
            pf4_curves, pf4_diagnostics = compute_pf4_gate_curves(hw, tw)
        row_idx = target_rows["row_index"].to_numpy(dtype=int)
        part_payload: dict[str, object] = {
            "well_id": str(well_id),
            "row_index": row_idx,
            "hmmpf_pf_smooth": pf_curve[row_idx],
            "hmmpf_pf_filt": pf_curve[row_idx],
            "hmmpf_scaleavg_smooth": scaleavg_curve[row_idx],
            "hmmpf_scaleavg_filt": scaleavg_curve[row_idx],
            "hmmpf_grid05_smooth_smooth": grid05_smooth[row_idx],
            "hmmpf_grid05_smooth_filt": grid05_smooth[row_idx],
            "hmmpf_grid025_smooth_smooth": grid025_smooth[row_idx],
            "hmmpf_grid025_smooth_filt": grid025_smooth[row_idx],
            "hmmpf_grid05_filt_smooth": grid05_filt[row_idx],
            "hmmpf_grid05_filt_filt": grid05_filt[row_idx],
            "hmmpf_multigrid_smooth": multigrid[row_idx],
            "hmmpf_multigrid_filt": multigrid[row_idx],
        }
        for column in PF4_GATE_OUTPUT_COLUMNS:
            part_payload[column] = pf4_curves[column][row_idx]
        part = pd.DataFrame(part_payload)
        rows.append(part)
        diagnostics.append(
            {
                "well_id": str(well_id),
                "target_rows": int(len(row_idx)),
                "worker_cuda_visible_devices": worker_device,
                "selector_variant": selector_variant,
                "scaleavg_selector_variant": scaleavg_selector,
                "pf_backend": pf_backend,
                "scaleavg_backend": scaleavg_backend,
                "pf_mean": float(np.nanmean(pf_curve[row_idx])),
                "scaleavg_mean": float(np.nanmean(scaleavg_curve[row_idx])),
                "grid05_smooth_mean": float(np.nanmean(grid05_smooth[row_idx])),
                "grid025_smooth_mean": float(np.nanmean(grid025_smooth[row_idx])),
                "grid05_filt_mean": float(np.nanmean(grid05_filt[row_idx])),
                "multigrid_mean": float(np.nanmean(multigrid[row_idx])),
                **pf4_diagnostics,
            }
        )
        for column in PF4_GATE_OUTPUT_COLUMNS:
            diagnostics[-1][f"{column}_mean"] = float(np.nanmean(pf4_curves[column][row_idx]))
        # 井戸単位のPF/GridHMM計算は一時配列が大きいため、次の井戸へ進む前に解放する。
        del (
            hw,
            tw,
            part,
            pf_curve,
            scaleavg_curve,
            grid05_smooth,
            grid05_filt,
            grid025_smooth,
            frac,
            multigrid,
            pf4_curves,
            pf4_diagnostics,
            row_idx,
            part_payload,
        )
        release_cuda_cache()
    if not rows:
        return pd.DataFrame(columns=DYNAMIC_OUTPUT_COLUMNS), diagnostics
    table = pd.concat(rows, ignore_index=True).loc[:, DYNAMIC_OUTPUT_COLUMNS]
    table = table.sort_values(["well_id", "row_index"]).reset_index(drop=True)
    del rows
    release_cuda_cache()
    return table, diagnostics


def write_dynamic_hmmpf_table(
    table: pd.DataFrame,
    diagnostics: list[dict[str, object]],
    output_path: Path,
    expected_rows: int | None = None,
    diagnostics_path: Path | None = None,
) -> Path:
    """Validate and write a dynamic HMM/PFBase table.

    Args:
        table: Dynamic feature table.
        diagnostics: Per-well diagnostics.
        output_path: Destination parquet path.
        expected_rows: Optional expected row count.
        diagnostics_path: Optional destination for diagnostics JSON.

    Returns:
        Written parquet path.
    """

    table = table.loc[:, DYNAMIC_OUTPUT_COLUMNS].sort_values(["well_id", "row_index"]).reset_index(drop=True)
    if expected_rows is not None and len(table) != expected_rows:
        raise ValueError(f"Dynamic HMM/PFBase table rows mismatch: {len(table)} vs {expected_rows}")
    value_cols = [col for col in table.columns if col not in {"well_id", "row_index"}]
    if table[value_cols].isna().any().any():
        raise ValueError("Dynamic HMM/PFBase table contains NaN values.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_path, index=False)
    diagnostics_output_path = diagnostics_path or output_path.parent / "dynamic_hmmpf_diagnostics.json"
    diagnostics_output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_output_path.write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )
    print(f"Saved dynamic HMM/PFBase feature table: {output_path} rows={len(table)}", flush=True)
    return output_path


def split_wells_for_workers(well_ids: list[str], worker_count: int) -> list[list[str]]:
    """Split well ids into non-empty round-robin worker shards."""

    shards = [[] for _ in range(worker_count)]
    for idx, well_id in enumerate(well_ids):
        shards[idx % worker_count].append(well_id)
    return [shard for shard in shards if shard]


def build_dynamic_hmmpf_table_parallel(
    competition_root: Path,
    output_path: Path,
    sample_index: pd.DataFrame,
    well_ids: list[str],
) -> Path:
    """Build dynamic HMM/PFBase table using one subprocess per visible GPU."""

    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    max_workers = int(os.environ.get("ROGII_DYNAMIC_GPU_WORKERS", max(device_count, 1)))
    worker_count = min(max_workers, max(device_count, 1), len(well_ids))
    if worker_count <= 1:
        table, diagnostics = compute_dynamic_hmmpf_rows(competition_root, sample_index, well_ids)
        return write_dynamic_hmmpf_table(table, diagnostics, output_path, expected_rows=len(sample_index))

    shard_dir = output_path.parent / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards = split_wells_for_workers(well_ids, worker_count)
    processes: list[tuple[subprocess.Popen[bytes], Path, Path, int, list[str]]] = []
    for worker_idx, shard_wells in enumerate(shards):
        shard_path = shard_dir / f"hmmpf_worker{worker_idx}.parquet"
        diagnostics_path = shard_path.with_suffix(".diagnostics.json")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--dynamic-worker",
            "--competition-root",
            str(competition_root),
            "--output-path",
            str(shard_path),
            "--well-ids",
            ",".join(shard_wells),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(worker_idx)
        print(
            f"Launching dynamic worker {worker_idx} on CUDA_VISIBLE_DEVICES={worker_idx}: wells={shard_wells}",
            flush=True,
        )
        process = subprocess.Popen(command, env=env)
        processes.append((process, shard_path, diagnostics_path, worker_idx, shard_wells))

    for process, _, _, worker_idx, shard_wells in processes:
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"Dynamic worker {worker_idx} failed with code {return_code}: wells={shard_wells}"
            )

    tables: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    for _, shard_path, diag_path, _, _ in processes:
        tables.append(pd.read_parquet(shard_path))
        diagnostics.extend(json.loads(diag_path.read_text(encoding="utf-8")))
    table = pd.concat(tables, ignore_index=True)
    result = write_dynamic_hmmpf_table(table, diagnostics, output_path, expected_rows=len(sample_index))
    del tables, table, diagnostics, sample_index
    release_cuda_cache()
    return result


def build_dynamic_hmmpf_table(competition_root: Path, output_path: Path) -> Path:
    """Compute HMM/PFBase component table for mounted test wells."""

    if os.environ.get("ROGII_REUSE_DYNAMIC") == "1" and output_path.is_file():
        print(f"Reusing local dynamic HMM table: {output_path}", flush=True)
        return output_path

    sample_index = load_sample_index(competition_root)
    well_ids = list(dict.fromkeys(sample_index["well_id"].astype(str).tolist()))
    return build_dynamic_hmmpf_table_parallel(competition_root, output_path, sample_index, well_ids)


def dynamic_worker_main() -> None:
    """Worker entry point for one dynamic HMM/PFBase shard."""

    parser = ArgumentParser()
    parser.add_argument("--dynamic-worker", action="store_true")
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--well-ids", required=True)
    args = parser.parse_args()
    well_ids = [well_id for well_id in args.well_ids.split(",") if well_id]
    sample_index = load_sample_index(args.competition_root)
    table, diagnostics = compute_dynamic_hmmpf_rows(args.competition_root, sample_index, well_ids)
    expected_rows = int(sample_index["well_id"].isin(well_ids).sum())
    write_dynamic_hmmpf_table(
        table,
        diagnostics,
        args.output_path,
        expected_rows=expected_rows,
        diagnostics_path=args.output_path.with_suffix(".diagnostics.json"),
    )
    del table, diagnostics, sample_index
    release_cuda_cache()


def build_exp202_likpf_table(
    artifact_root: Path,
    competition_root: Path,
    output_path: Path,
    seed_base: int = 0,
) -> Path:
    """Build one exp202 likelihood-PF realization for mounted test wells.

    Args:
        artifact_root: Mounted model artifact root containing the precompute script.
        competition_root: Mounted competition data root.
        output_path: Destination parquet path.
        seed_base: likelihood-PF seed base。

    Returns:
        Path: Written parquet table path.
    """

    if os.environ.get("ROGII_REUSE_DYNAMIC") == "1" and output_path.is_file():
        print(f"Reusing local exp202 lik-PF table: {output_path}", flush=True)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(artifact_root / "src" / "features" / "precompute_exp202_likpf_table.py"),
        "--data-root",
        str(competition_root),
        "--split",
        "test",
        "--output-path",
        str(output_path),
        "--seed-base",
        str(seed_base),
        "--n-particles",
        "500",
        "--n-seeds",
        "240",
        "--batch-wells",
        "16",
        "--w-sib",
        "0.3",
        "--k-aug",
        "0",
        "--device",
        "cuda" if torch.cuda.is_available() else "cpu",
    ]
    env = os.environ.copy()
    python_path = str(artifact_root)
    if env.get("PYTHONPATH"):
        python_path = f"{python_path}{os.pathsep}{env['PYTHONPATH']}"
    env["PYTHONPATH"] = python_path
    print("Running command:", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)
    table = pd.read_parquet(output_path)
    expected_rows = len(load_sample_index(competition_root))
    if len(table) != expected_rows:
        raise ValueError(f"exp202 lik-PF table rows mismatch: {len(table)} vs {expected_rows}")
    value_cols = [col for col in table.columns if col not in {"well_id", "row_index"}]
    if table[value_cols].isna().any().any():
        raise ValueError("exp202 lik-PF table contains NaN values.")
    print(f"Saved exp202 lik-PF feature table: {output_path} rows={len(table)}", flush=True)
    del table
    release_cuda_cache()
    return output_path


def validate_submission_file(
    competition_root: Path,
    submission_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Validate submission shape and write a JSON report.

    Args:
        competition_root: Mounted competition data root.
        submission_path: Created submission file.
        output_path: JSON validation destination.

    Returns:
        dict[str, object]: Validation report.
    """

    sample = pd.read_csv(competition_root / "sample_submission.csv")
    submission = pd.read_csv(submission_path)
    report: dict[str, object] = {
        "submission_path": str(submission_path),
        "rows": int(len(submission)),
        "sample_rows": int(len(sample)),
        "same_id_order": bool(submission["id"].equals(sample["id"])),
        "duplicate_id_count": int(submission["id"].duplicated().sum()),
        "missing_tvt_count": int(submission["tvt"].isna().sum()),
        "finite_tvt": bool(np.isfinite(submission["tvt"].to_numpy(dtype=float)).all()),
        "min_tvt": float(submission["tvt"].min()),
        "max_tvt": float(submission["tvt"].max()),
        "mean_tvt": float(submission["tvt"].mean()),
    }
    if report["rows"] != report["sample_rows"] or not report["same_id_order"]:
        raise ValueError(f"Submission row/id validation failed: {report}")
    if report["duplicate_id_count"] or report["missing_tvt_count"] or not report["finite_tvt"]:
        raise ValueError(f"Submission value validation failed: {report}")
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Submission validation: {json.dumps(report, indent=2)}", flush=True)
    return report


def apply_neighbor_surface_blend_submission(
    competition_root: Path,
    artifact_root: Path,
    submission_path: Path,
    diagnostics_path: Path,
) -> None:
    """Apply OOF-fitted neighbor surface blend to the final submission.

    Args:
        competition_root: Mounted competition data root.
        artifact_root: Mounted model artifact root containing blend code and weights.
        submission_path: Submission CSV to update in place.
        diagnostics_path: JSON destination for blend coverage/change diagnostics.
    """

    module_path = artifact_root / NEIGHBOR_SURFACE_BLEND_MODULE
    weights_path = artifact_root / NEIGHBOR_SURFACE_BLEND_WEIGHTS
    if not module_path.is_file():
        raise FileNotFoundError(f"Neighbor surface blend module not found: {module_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Neighbor surface blend weights not found: {weights_path}")

    spec = importlib.util.spec_from_file_location("neighbor_surface_blend_submit", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load neighbor surface blend module: {module_path}")
    nsb = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = nsb
    spec.loader.exec_module(nsb)

    submission = pd.read_csv(submission_path)
    original_tvt = submission["tvt"].to_numpy(dtype=np.float64).copy()
    parts = submission["id"].str.rsplit("_", n=1)
    submission["well_id"] = parts.str[0]
    submission["row_index"] = parts.str[1].astype(int)

    weights = json.loads(weights_path.read_text(encoding="utf-8"))
    bank = nsb.load_surface_bank(competition_root / "train", step=5)
    blended = original_tvt.copy()
    nb_covered = np.zeros(len(submission), dtype=bool)
    nearest_distance = np.full(len(submission), np.inf, dtype=np.float64)
    for well_id, group in submission.groupby("well_id", sort=False):
        frame = pd.read_csv(
            competition_root / "test" / f"{well_id}__horizontal_well.csv",
            usecols=["X", "Y", "Z", "TVT_input"],
        )
        nb_pred, nb_dnn = nsb.neighbor_transfer_for_well(frame, bank, exclude_well=str(well_id))
        rows = group["row_index"].to_numpy(dtype=np.int64)
        idx = group.index.to_numpy()
        base_values = original_tvt[idx]
        blended[idx] = nsb.apply_bin_weights(base_values, nb_pred[rows], nb_dnn[rows], weights)
        nb_covered[idx] = np.isfinite(nb_pred[rows])
        nearest_distance[idx] = nb_dnn[rows]

    submission["tvt"] = blended
    final = submission[["id", "tvt"]]
    if final["tvt"].isna().any() or not np.isfinite(final["tvt"].to_numpy(dtype=np.float64)).all():
        raise ValueError("Neighbor surface blended submission contains non-finite predictions.")
    final.to_csv(submission_path, index=False)

    delta = blended - original_tvt
    diagnostics = {
        "module_path": str(module_path),
        "weights_path": str(weights_path),
        "weights": weights,
        "rows": int(len(final)),
        "covered_rows": int(nb_covered.sum()),
        "coverage": float(nb_covered.mean()),
        "changed_rows": int(np.count_nonzero(np.abs(delta) > 1e-12)),
        "mean_abs_change": float(np.mean(np.abs(delta))),
        "max_abs_change": float(np.max(np.abs(delta))),
        "finite_tvt": bool(np.isfinite(final["tvt"].to_numpy(dtype=np.float64)).all()),
        "nearest_distance_min": float(np.nanmin(np.where(np.isfinite(nearest_distance), nearest_distance, np.nan))),
        "nearest_distance_median": float(np.nanmedian(np.where(np.isfinite(nearest_distance), nearest_distance, np.nan))),
    }
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(f"Neighbor surface blend diagnostics: {json.dumps(diagnostics, indent=2)}", flush=True)


def run_physics_crf_submission(
    competition_root: Path,
    output_table_path: Path,
    submission_path: Path,
) -> Path:
    """固定p=0.001 Physics Grid CRFのtest予測を作る。

    Args:
        competition_root: competition data root。
        output_table_path: CRF行テーブルの保存先。
        submission_path: CRF単体submissionの保存先。

    Returns:
        検証済みCRF行テーブルpath。
    """

    if (
        os.environ.get("ROGII_REUSE_DYNAMIC") == "1"
        and output_table_path.is_file()
        and submission_path.is_file()
    ):
        print(f"Reusing local Physics CRF table: {output_table_path}", flush=True)
        return output_table_path
    source_root = find_hmmpf_source().parent
    predictor = source_root / "src" / "inference" / "predict_exp384_physics_crf.py"
    if not predictor.is_file() and (source_root / "src.zip").is_file():
        extract_root = Path("/kaggle/working/crf_source/src")
        if not Path("/kaggle/working").is_dir():
            extract_root = Path("working/crf_source/src")
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source_root / "src.zip") as archive:
            archive.extractall(extract_root)
        predictor = next(
            extract_root.rglob("predict_exp384_physics_crf.py"),
            extract_root / "inference" / "predict_exp384_physics_crf.py",
        )
        src_package = next((parent for parent in predictor.parents if parent.name == "src"), extract_root)
        source_root = src_package.parent
    if not predictor.is_file():
        raise FileNotFoundError(f"Physics CRF predictor was not found: {predictor}")
    command = [
        sys.executable,
        str(predictor),
        "--data-root",
        str(competition_root),
        "--output-path",
        str(output_table_path),
        "--submission-path",
        str(submission_path),
        "--devices",
        "cuda:0",
        "--reference-switch-probability",
        "0.001",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root) + os.pathsep + environment.get("PYTHONPATH", "")
    print("Running Physics CRF command:", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=environment)
    frame = pd.read_parquet(output_table_path, columns=["well_id", "row_index", "pred_crf"])
    sample = pd.read_csv(competition_root / "sample_submission.csv", usecols=["id"])
    if len(frame) != len(sample) or frame[["well_id", "row_index"]].duplicated().any():
        raise ValueError("Physics CRF rows do not match sample submission")
    if not np.isfinite(frame["pred_crf"].to_numpy(dtype=np.float64)).all():
        raise ValueError("Physics CRF contains non-finite predictions")
    return output_table_path


def main_single_split() -> None:
    """Run exp417-only dynamic candidates and the matching BiLSTM gate."""

    competition_root = find_competition_root()
    artifact_root = find_artifact_root()
    delta_artifact_root = find_named_nn_artifact_root(
        artifact_root,
        DELTA_ARTIFACT_NAME,
    )
    if Path("/kaggle/working").is_dir():
        working_root = Path("/kaggle/working")
    else:
        working_root = Path(os.environ.get("ROGII_WORKING_ROOT", "."))
        working_root.mkdir(parents=True, exist_ok=True)
    output_dir = working_root / "runs" / "hmmpf_multigrid_1dsdf_softmax_submit"
    pf4_table_path = build_dynamic_hmmpf_table(
        competition_root,
        working_root / "dynamic_hmmpfbase_features" / PF4_TABLE_NAME,
    )
    # HMM/PFはexp417を一度だけ実行し、gate候補と新版NN入力で共有する。
    exp417_table_path = working_root / "dynamic_exp417_features" / "exp417_components.parquet"
    exp417_table_path.parent.mkdir(parents=True, exist_ok=True)
    exp417_components = run_exp417_components(competition_root)
    exp417_components.to_parquet(exp417_table_path, index=False)
    release_cuda_cache()
    hmmpf_multigrid_table_path = build_exp417_nn_feature_table(
        exp417_components,
        pf4_table_path,
        working_root / "dynamic_hmmpf_multigrid_features" / HMMPF_MULTIGRID_TABLE_NAME,
    )
    # 新版49/31特徴NNが読むseed 0だけを生成する。
    exp202_table_path = build_exp202_likpf_table(
        artifact_root,
        competition_root,
        working_root
        / "dynamic_exp202_likpf_features"
        / EXP202_LIKPF_TABLE_PATTERN.format(seed=0),
        seed_base=0,
    )
    correlation_root = find_correlation_artifact_root(artifact_root)
    sdf_submission_path = working_root / "submission_sdf1d.csv"
    sdf_table_path = working_root / "dynamic_sdf_features" / "sdf1d_test.parquet"
    run_correlation_submission(
        competition_root,
        correlation_root,
        sdf_submission_path,
        sdf_table_path,
    )
    release_cuda_cache()

    # exp417 HMMPFで再学習した49特徴modelを推論する。
    command = [
        sys.executable,
        str(artifact_root / "baseline.py"),
        "--predict-only",
        "--data-root",
        str(competition_root),
        "--artifact-dir",
        str(artifact_root),
        "--output-dir",
        str(output_dir),
        "--no-root-submission",
        "--feature-plan-stages",
        "diff1,exp202_likpf,hmmpf_multigrid_d,sdf",
        "--exp202-likpf-table-path",
        str(exp202_table_path),
        "--hmmpf-multigrid-table-path",
        str(hmmpf_multigrid_table_path),
        "--sdf-table-path",
        str(sdf_table_path),
        "--exp202-likpf-particle-count",
        "500",
        "--exp202-likpf-seed-count",
        "240",
        "--exp202-likpf-seed-bases",
        "0",
        "--exp202-likpf-w-sib",
        "0.3",
        "--exp202-likpf-k-aug",
        "0",
    ]
    print("Running command:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    release_cuda_cache()

    submission_path = working_root / "submission.csv"
    generated_nn_path = output_dir / "submission.csv"
    if not generated_nn_path.is_file():
        raise FileNotFoundError(
            f"NN submission.csv was not created: {generated_nn_path}"
        )
    nn_submission_path = working_root / "submission_nn.csv"
    generated_nn_path.replace(nn_submission_path)

    # 同じexp417動的候補を使って再学習した31特徴raw-delta modelも推論する。
    delta_output_dir = working_root / "runs" / "finehmmpf_core31_delta_submit"
    delta_command = [
        sys.executable,
        str(delta_artifact_root / "baseline.py"),
        "--predict-only",
        "--data-root",
        str(competition_root),
        "--artifact-dir",
        str(delta_artifact_root),
        "--output-dir",
        str(delta_output_dir),
        "--no-root-submission",
        "--feature-plan-stages",
        "diff1,exp202_likpf,hmmpf_multigrid_d,sdf",
        "--exp202-likpf-table-path",
        str(exp202_table_path),
        "--hmmpf-multigrid-table-path",
        str(hmmpf_multigrid_table_path),
        "--sdf-table-path",
        str(sdf_table_path),
        "--exp202-likpf-particle-count",
        "500",
        "--exp202-likpf-seed-count",
        "240",
        "--exp202-likpf-seed-bases",
        "0",
        "--exp202-likpf-w-sib",
        "0.3",
        "--exp202-likpf-k-aug",
        "0",
    ]
    print("Running delta command:", " ".join(delta_command), flush=True)
    subprocess.run(delta_command, check=True)
    release_cuda_cache()
    generated_delta_path = delta_output_dir / "submission.csv"
    if not generated_delta_path.is_file():
        raise FileNotFoundError(f"Delta submission.csv was not created: {generated_delta_path}")
    delta_submission_path = working_root / "submission_delta.csv"
    generated_delta_path.replace(delta_submission_path)
    # 新版49/31特徴NNで学習したTypeWell IDなし・CRFなしBiLSTMだけを使う。
    lstm_gate_root = find_gate_artifact_root(artifact_root, LSTM_GATE_ARTIFACT_NAME)
    lstm_submission_path = working_root / "submission_softmax_lstm.csv"
    lstm_submission = run_softmax_gate_submission(
        competition_root,
        lstm_gate_root,
        exp417_table_path,
        nn_submission_path,
        sdf_submission_path,
        delta_submission_path,
        lstm_submission_path,
    )
    final = lstm_submission.copy()
    final.to_csv(submission_path, index=False)
    blend_diagnostics = {
        "method": "exp417_hmm_pf_exp417nn49nn31_softmax_bilstm",
        "cv_rmse": 5.333151466,
        "candidate_columns": GATE_CANDIDATE_COLUMNS,
        "rows": int(len(final)),
        "lstm_submission": str(lstm_submission_path),
        "exp417_table": str(exp417_table_path),
    }
    (working_root / "softmax_blend_diagnostics.json").write_text(
        json.dumps(blend_diagnostics, indent=2),
        encoding="utf-8",
    )
    release_cuda_cache()
    validate_submission_file(
        competition_root,
        submission_path,
        working_root / "submission_validation.json",
    )
    print(
        f"Created exp417 HMM/PF SoftMax BiLSTM submission: {submission_path}",
        flush=True,
    )
    release_cuda_cache()


def find_bundle_root(artifact_root: Path) -> Path:
    """manifestを持つ5 split artifact bundleを探す。

    Args:
        artifact_root: 自動探索済みのNN artifact。

    Returns:
        ``manifest.json``を持つbundle directory。
    """

    for candidate in (artifact_root, *artifact_root.parents):
        if (candidate / "manifest.json").is_file():
            return candidate
    raise FileNotFoundError("Three-split artifact manifest was not found")


def run_nn_artifact_submission(
    artifact_root: Path,
    competition_root: Path,
    output_dir: Path,
    exp202_table_path: Path,
    hmmpf_table_path: Path,
    sdf_table_path: Path,
    output_path: Path,
) -> Path:
    """1 split・5 foldの49または31特徴NNを推論する。

    Args:
        artifact_root: 5 fold NN artifact。
        competition_root: mounted competition root。
        output_dir: baseline推論出力先。
        exp202_table_path: likelihood-PF特徴表。
        hmmpf_table_path: Eager momentum HMMPF特徴表。
        sdf_table_path: 同じsplitのSDF特徴表。
        output_path: 保存するsubmission path。

    Returns:
        平均・Stdを含むprediction details path。
    """

    command = [
        sys.executable,
        str(artifact_root / "baseline.py"),
        "--predict-only",
        "--data-root",
        str(competition_root),
        "--artifact-dir",
        str(artifact_root),
        "--output-dir",
        str(output_dir),
        "--no-root-submission",
        "--feature-plan-stages",
        "diff1,exp202_likpf,hmmpf_multigrid_d,sdf",
        "--exp202-likpf-table-path",
        str(exp202_table_path),
        "--hmmpf-multigrid-table-path",
        str(hmmpf_table_path),
        "--sdf-table-path",
        str(sdf_table_path),
        "--exp202-likpf-particle-count",
        "500",
        "--exp202-likpf-seed-count",
        "240",
        "--exp202-likpf-seed-bases",
        "0",
        "--exp202-likpf-w-sib",
        "0.3",
        "--exp202-likpf-k-aug",
        "0",
    ]
    print("Running NN command:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    generated_path = output_dir / "submission.csv"
    if not generated_path.is_file():
        raise FileNotFoundError(generated_path)
    generated_path.replace(output_path)
    details_path = output_dir / "prediction_details.csv"
    if not details_path.is_file():
        raise FileNotFoundError(details_path)
    release_cuda_cache()
    return details_path


def average_submission_files(paths: list[Path], output_path: Path) -> pd.DataFrame:
    """ID整列した複数splitのsubmissionを等重み平均する。

    Args:
        paths: 平均するsubmission一覧。
        output_path: 平均結果の保存先。

    Returns:
        平均済みsubmission。
    """

    frames = [pd.read_csv(path) for path in paths]
    base = frames[0][["id"]].copy()
    predictions: list[np.ndarray] = []
    for path, frame in zip(paths, frames):
        if not base["id"].equals(frame["id"]):
            raise ValueError(f"Submission IDs are not aligned: {path}")
        predictions.append(frame["tvt"].to_numpy(dtype=np.float64))
    base["tvt"] = np.mean(np.stack(predictions), axis=0)
    base.to_csv(output_path, index=False)
    return base


def main() -> None:
    """Eager momentumの5 split NN/CNN/LSTMを推論して提出を作る。"""

    competition_root = find_competition_root()
    artifact_root = find_artifact_root()
    bundle_root = find_bundle_root(artifact_root)
    manifest = json.loads((bundle_root / "manifest.json").read_text(encoding="utf-8"))
    if not str(manifest.get("protocol", "")).startswith("5 folds x 5 GroupKFold splits"):
        raise ValueError("Unexpected artifact protocol")
    working_root = (
        Path("/kaggle/working")
        if Path("/kaggle/working").is_dir()
        else Path(os.environ.get("ROGII_WORKING_ROOT", "."))
    )
    working_root.mkdir(parents=True, exist_ok=True)

    pf4_table_path = build_dynamic_hmmpf_table(
        competition_root,
        working_root / "dynamic_hmmpfbase_features" / PF4_TABLE_NAME,
    )
    exp417_table_path = (
        working_root / "dynamic_exp417_features" / "exp417_components.parquet"
    )
    exp417_table_path.parent.mkdir(parents=True, exist_ok=True)
    exp417_components = run_exp417_components(competition_root)
    exp417_components.to_parquet(exp417_table_path, index=False)
    release_cuda_cache()
    hmmpf_table_path = build_exp417_nn_feature_table(
        exp417_components,
        pf4_table_path,
        working_root
        / "dynamic_hmmpf_multigrid_features"
        / HMMPF_MULTIGRID_TABLE_NAME,
    )
    exp202_table_path = build_exp202_likpf_table(
        artifact_root,
        competition_root,
        working_root
        / "dynamic_exp202_likpf_features"
        / EXP202_LIKPF_TABLE_PATTERN.format(seed=0),
        seed_base=0,
    )

    cnn_paths: list[Path] = []
    lstm_paths: list[Path] = []
    for split_name in SPLIT_NAMES:
        sdf_submission_path = working_root / f"submission_sdf_{split_name}.csv"
        sdf_table_path = (
            working_root / "dynamic_sdf_features" / f"{split_name}.parquet"
        )
        run_correlation_submission(
            competition_root,
            bundle_root / "sdf" / split_name,
            sdf_submission_path,
            sdf_table_path,
        )
        nn49_path = working_root / f"submission_nn49_{split_name}.csv"
        nn31_path = working_root / f"submission_nn31_{split_name}.csv"
        nn49_details_path = run_nn_artifact_submission(
            bundle_root / "nn49" / split_name,
            competition_root,
            working_root / "runs" / f"nn49_{split_name}",
            exp202_table_path,
            hmmpf_table_path,
            sdf_table_path,
            nn49_path,
        )
        run_nn_artifact_submission(
            bundle_root / "nn31" / split_name,
            competition_root,
            working_root / "runs" / f"nn31_{split_name}",
            exp202_table_path,
            hmmpf_table_path,
            sdf_table_path,
            nn31_path,
        )
        cnn_path = working_root / f"submission_cnn_{split_name}.csv"
        lstm_path = working_root / f"submission_lstm_{split_name}.csv"
        run_softmax_gate_submission(
            competition_root,
            bundle_root / "gates" / split_name / "cnn",
            exp417_table_path,
            nn49_path,
            sdf_submission_path,
            nn31_path,
            cnn_path,
            nn49_details_path,
        )
        run_softmax_gate_submission(
            competition_root,
            bundle_root / "gates" / split_name / "lstm",
            exp417_table_path,
            nn49_path,
            sdf_submission_path,
            nn31_path,
            lstm_path,
            nn49_details_path,
        )
        cnn_paths.append(cnn_path)
        lstm_paths.append(lstm_path)

    cnn = average_submission_files(
        cnn_paths, working_root / "submission_softmax_cnn.csv"
    )
    lstm = average_submission_files(
        lstm_paths, working_root / "submission_softmax_lstm.csv"
    )
    cnn_weight, lstm_weight = load_cnn_lstm_blend_weights(artifact_root)
    final = cnn.copy()
    final["tvt"] = (
        cnn_weight * cnn["tvt"].to_numpy(dtype=np.float64)
        + lstm_weight * lstm["tvt"].to_numpy(dtype=np.float64)
    )
    submission_path = working_root / "submission.csv"
    final.to_csv(submission_path, index=False)
    (working_root / "softmax_blend_diagnostics.json").write_text(
        json.dumps(
            {
                "method": "exp417_momentum_eager_5fold_x_5groupkfold_cnn_lstm",
                "cnn_weight": cnn_weight,
                "lstm_weight": lstm_weight,
                "rows": int(len(final)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    validate_submission_file(
        competition_root,
        submission_path,
        working_root / "submission_validation.json",
    )
    print(f"Created five-split submission: {submission_path}", flush=True)
    release_cuda_cache()


if __name__ == "__main__":
    if "--dynamic-worker" in sys.argv:
        dynamic_worker_main()
    else:
        main()
