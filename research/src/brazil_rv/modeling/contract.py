from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

CONTRACT_VERSION = "CROSS_ASSET_ITRANSFORMER_V1"
CLOUD_RUNTIME_CONTRACT_VERSION = "CROSS_ASSET_ITRANSFORMER_CLOUD_RUNTIME_V1"
MUON_COMPATIBILITY_CONTRACT_VERSION = "PYTORCH_2_13_MUON_COMPAT_V1"
TORCH_COMPILE_COMPATIBILITY_CONTRACT_VERSION = "TORCH_COMPILE_COMPATIBILITY_V1"
FEATURE_CONTRACT_VERSION = "M1_FEATURES_V1"


def resolve_project_root() -> Path:
    configured = os.environ.get("BRAZIL_RV_ROOT")
    root = (
        Path(configured).expanduser().resolve()
        if configured is not None
        else Path(__file__).absolute().parents[6]
    )
    required = (
        root / "quant" / "b3-quant" / "research",
        root / "quant-data" / "b3" / "processed" / "features",
    )
    if not all(path.is_dir() for path in required):
        raise FileNotFoundError(
            f"BRAZIL_RV_ROOT does not contain the required workspace layout: {root}"
        )
    return root


PROJECT_ROOT = resolve_project_root()
FEATURE_STORE_POINTER = (
    PROJECT_ROOT
    / "quant-data"
    / "b3"
    / "processed"
    / "features"
    / "m1_features_v1_canonical_path.txt"
)
RUN_OUTPUT_BASE = PROJECT_ROOT / "quant-data" / "b3" / "processed" / "model_runs"

TRAIN_START = date(2022, 3, 17)
TRAIN_END = date(2024, 6, 28)
VALIDATION_START = date(2024, 7, 8)
VALIDATION_END = date(2025, 6, 30)
TEST_START = date(2025, 7, 7)
TEST_END = date(2026, 7, 17)

EXPECTED_SAMPLE_COUNT = 59_565
EXPECTED_DECISIONS_PER_DATE = 55
EQUITY_COUNT = 158
CONTEXT_COUNT = 6
INSTRUMENT_COUNT = EQUITY_COUNT + CONTEXT_COUNT
HORIZON_COUNT = 3
DYNAMIC_CHANNEL_COUNT = 6
EQUITY_SLOW_COUNT = 1
CONTEXT_SLOW_COUNT = 3
CONTEXT_SYMBOLS = ("WIN$", "WDO$", "DI1F27", "DI1F28", "DI1F29", "DI1F31")
HORIZONS = (30, 60, 120)

PATCH_MINUTES = 5
PATCH_INPUT_WIDTH = PATCH_MINUTES * DYNAMIC_CHANNEL_COUNT
ABSOLUTE_PATCH_COUNT = 69
STATE_TOKEN_SLOT = 69
TEMPORAL_TOKEN_COUNT = 70
EQUITY_ABSOLUTE_START_PATCH = 12

FAMILY_EQUITY = 0
FAMILY_EQUITY_FUTURE = 1
FAMILY_FX_FUTURE = 2
FAMILY_RATE_FUTURE = 3
FAMILY_COUNT = 4
INSTRUMENT_FAMILY_IDS = (
    (FAMILY_EQUITY,) * EQUITY_COUNT
    + (FAMILY_EQUITY_FUTURE, FAMILY_FX_FUTURE)
    + (FAMILY_RATE_FUTURE,) * 4
)

MODEL_VARIANTS = ("full", "temporal_only")
OPTIMIZER_VARIANTS = ("hybrid", "adamw")
ALLOWED_SEEDS = (11, 29, 47)

D_MODEL = 256
ATTENTION_HEADS = 8
HEAD_DIM = 32
TEMPORAL_DEPTH = 2
CROSS_ASSET_DEPTH = 6
SWIGLU_WIDTH = 704
RMS_NORM_EPS = 1e-6
QK_NORM_EPS = 1e-6
ROPE_BASE = 10_000.0
INPUT_DROPOUT = 0.05
RESIDUAL_DROPOUT = 0.10
ATTENTION_DROPOUT = 0.0

EFFECTIVE_BATCH_SIZE = 32
MAX_EPOCHS = 20
EARLY_STOP_PATIENCE = 5
MIN_IC_IMPROVEMENT = 1e-4
GRADIENT_CLIP = 1.0
HUBER_DELTA = 1.0

MUON_LR = 0.02
MUON_MOMENTUM = 0.95
MUON_NESTEROV = True
MUON_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
MUON_EPS = 1e-7
MUON_NS_STEPS = 5
MUON_WEIGHT_DECAY = 0.01
MUON_ADJUST_LR_FN = "original"

ADAMW_LR = 3e-4
ADAMW_BETAS = (0.9, 0.95)
ADAMW_EPS = 1e-8
ADAMW_WEIGHT_DECAY = 0.01

WARMUP_FRACTION = 0.05
FINAL_LR_FACTOR = 0.1
COMPILE_WARMUP_PASS_COUNT = 5
COMPILE_STEADY_STATE_PASS_COUNT = 3

COMPILE_PARITY_PREDICTION_ATOL = 5e-3
COMPILE_PARITY_PREDICTION_RTOL = 5e-3
COMPILE_PARITY_LOSS_ATOL = 5e-4
COMPILE_PARITY_LOSS_RTOL = 5e-3
COMPILE_PARITY_GRADIENT_RELATIVE_L2_MAX = 1e-2
COMPILE_PARITY_GRADIENT_COSINE_MIN = 0.9999
COMPILE_PARITY_GRADIENT_MAX_ABSOLUTE_ATOL = 1e-3
COMPILE_PARITY_GRADIENT_MAX_ABSOLUTE_RTOL = 1e-2

MIN_IC_EQUITIES = 30
SANITY_SAMPLE_COUNT = 32
SANITY_DECISION_INDEX = 27
SANITY_MAX_STEPS = 1_000
SANITY_MAX_LOSS = 0.05
SANITY_MIN_SPEARMAN = 0.90

EXPECTED_ARRAY_SHAPES = {
    "equity_features.npy": (1248, 158, 405, 6),
    "equity_slow.npy": (1248, 158, 1),
    "equity_membership.npy": (1248, 158),
    "equity_data_ready.npy": (1248, 158),
    "context_features.npy": (1248, 6, 465, 6),
    "context_slow.npy": (1248, 6, 3),
    "context_data_ready.npy": (1248, 6),
    "raw_returns.npy": (1248, 158, 55, 3),
    "targets.npy": (1248, 158, 55, 3),
    "label_mask.npy": (1248, 158, 55, 3),
    "cross_section_median.npy": (1248, 55, 3),
    "horizon_mask.npy": (1248, 55, 3),
}


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    precision: str
    microbatch_size: int
    accumulation_steps: int
    evaluation_batch_size: int
    num_workers: int
    prefetch_factor: int
    compile_backend: str
    compile_mode: str
    compile_fullgraph: bool
    compile_dynamic: bool
    minimum_vram_bytes: int
    expected_compute_capability: tuple[int, int]
    required_device_name_fragment: str | None
    required_cpu_architecture: str | None


RUNTIME_PROFILES = {
    "a10": RuntimeProfile(
        name="a10",
        precision="bf16",
        microbatch_size=16,
        accumulation_steps=2,
        evaluation_batch_size=64,
        num_workers=4,
        prefetch_factor=2,
        compile_backend="inductor",
        compile_mode="reduce-overhead",
        compile_fullgraph=True,
        compile_dynamic=False,
        minimum_vram_bytes=22 * 1024**3,
        expected_compute_capability=(8, 6),
        required_device_name_fragment="A10",
        required_cpu_architecture=None,
    ),
    "a100": RuntimeProfile(
        name="a100",
        precision="bf16",
        microbatch_size=32,
        accumulation_steps=1,
        evaluation_batch_size=128,
        num_workers=6,
        prefetch_factor=4,
        compile_backend="inductor",
        compile_mode="reduce-overhead",
        compile_fullgraph=True,
        compile_dynamic=False,
        minimum_vram_bytes=38 * 1024**3,
        expected_compute_capability=(8, 0),
        required_device_name_fragment="A100",
        required_cpu_architecture=None,
    ),
    "gh200": RuntimeProfile(
        name="gh200",
        precision="bf16",
        microbatch_size=32,
        accumulation_steps=1,
        evaluation_batch_size=256,
        num_workers=8,
        prefetch_factor=4,
        compile_backend="inductor",
        compile_mode="reduce-overhead",
        compile_fullgraph=True,
        compile_dynamic=False,
        minimum_vram_bytes=90 * 1024**3,
        expected_compute_capability=(9, 0),
        required_device_name_fragment=None,
        required_cpu_architecture="aarch64",
    ),
}
RUNTIME_PROFILE_NAMES = tuple(RUNTIME_PROFILES)
if any(
    profile.microbatch_size * profile.accumulation_steps != EFFECTIVE_BATCH_SIZE
    for profile in RUNTIME_PROFILES.values()
):
    raise ValueError("Every runtime profile must preserve effective batch 32")


@dataclass(frozen=True)
class HardwareInfo:
    profile: str
    device_name: str
    compute_capability: tuple[int, int]
    total_vram_bytes: int
    cpu_architecture: str
    platform: str
    pytorch_version: str
    cuda_version: str | None
    cudnn_version: int | None


@dataclass(frozen=True)
class CompileSetupReport:
    api: str
    backend: str
    mode: str
    fullgraph: bool
    dynamic: bool
    backward_pass_autocast_control_available: bool
    backward_pass_autocast_policy: str


@dataclass(frozen=True)
class CompileParityThresholds:
    prediction_atol: float = COMPILE_PARITY_PREDICTION_ATOL
    prediction_rtol: float = COMPILE_PARITY_PREDICTION_RTOL
    loss_atol: float = COMPILE_PARITY_LOSS_ATOL
    loss_rtol: float = COMPILE_PARITY_LOSS_RTOL
    gradient_relative_l2_max: float = COMPILE_PARITY_GRADIENT_RELATIVE_L2_MAX
    gradient_cosine_min: float = COMPILE_PARITY_GRADIENT_COSINE_MIN
    gradient_max_absolute_atol: float = COMPILE_PARITY_GRADIENT_MAX_ABSOLUTE_ATOL
    gradient_max_absolute_rtol: float = COMPILE_PARITY_GRADIENT_MAX_ABSOLUTE_RTOL


@dataclass(frozen=True)
class CompileParityReport:
    mode: str
    dropout_enabled: bool
    batch_size: int
    passed: bool

    eager_predictions_finite: bool
    compiled_predictions_finite: bool
    prediction_allclose: bool
    prediction_max_absolute_difference: float
    prediction_relative_l2_error: float

    eager_loss: float
    compiled_loss: float
    losses_finite: bool
    loss_absolute_difference: float
    loss_tolerance: float

    gradient_presence_match: bool | None
    eager_gradients_finite: bool | None
    compiled_gradients_finite: bool | None
    gradient_parameter_count: int | None
    eager_gradient_l2_norm: float | None
    compiled_gradient_l2_norm: float | None
    eager_gradient_max_absolute: float | None
    gradient_relative_l2_error: float | None
    gradient_cosine_similarity: float | None
    gradient_max_absolute_difference: float | None
    gradient_max_absolute_tolerance: float | None


@dataclass(frozen=True)
class CompileEvaluationWarmupReport:
    evaluation_pass_seconds: tuple[float, float, float, float, float]
    evaluation_steady_state_median_seconds: float
    peak_allocated_cuda_memory_bytes: int
    peak_reserved_cuda_memory_bytes: int


@dataclass(frozen=True)
class CompileWarmupReport:
    training_pass_seconds: tuple[float, float, float, float, float]
    training_steady_state_median_seconds: float
    evaluation_pass_seconds: tuple[float, float, float, float, float]
    evaluation_steady_state_median_seconds: float
    peak_allocated_cuda_memory_bytes: int
    peak_reserved_cuda_memory_bytes: int


@dataclass(frozen=True)
class ArchitectureConstants:
    d_model: int = D_MODEL
    attention_heads: int = ATTENTION_HEADS
    head_dim: int = HEAD_DIM
    temporal_depth: int = TEMPORAL_DEPTH
    cross_asset_depth: int = CROSS_ASSET_DEPTH
    swiglu_width: int = SWIGLU_WIDTH
    rms_norm_eps: float = RMS_NORM_EPS
    qk_norm_eps: float = QK_NORM_EPS
    rope_base: float = ROPE_BASE
    input_dropout: float = INPUT_DROPOUT
    residual_dropout: float = RESIDUAL_DROPOUT
    attention_dropout: float = ATTENTION_DROPOUT
    output_horizons: int = HORIZON_COUNT


@dataclass(frozen=True)
class TrainingConstants:
    effective_batch_size: int = EFFECTIVE_BATCH_SIZE
    maximum_epochs: int = MAX_EPOCHS
    early_stop_patience: int = EARLY_STOP_PATIENCE
    minimum_ic_improvement: float = MIN_IC_IMPROVEMENT
    gradient_clip: float = GRADIENT_CLIP
    huber_delta: float = HUBER_DELTA


@dataclass(frozen=True)
class MuonConstants:
    lr: float = MUON_LR
    momentum: float = MUON_MOMENTUM
    nesterov: bool = MUON_NESTEROV
    ns_coefficients: tuple[float, float, float] = MUON_NS_COEFFICIENTS
    eps: float = MUON_EPS
    ns_steps: int = MUON_NS_STEPS
    weight_decay: float = MUON_WEIGHT_DECAY
    adjust_lr_fn: str = MUON_ADJUST_LR_FN


@dataclass(frozen=True)
class AdamWConstants:
    lr: float = ADAMW_LR
    betas: tuple[float, float] = ADAMW_BETAS
    eps: float = ADAMW_EPS
    decayed_weight_decay: float = ADAMW_WEIGHT_DECAY
    zero_decay_weight_decay: float = 0.0
    fused: bool = True


@dataclass(frozen=True)
class SchedulerConstants:
    schedule: str = "linear_warmup_cosine_decay"
    maximum_epochs: int = MAX_EPOCHS
    warmup_fraction: float = WARMUP_FRACTION
    final_lr_factor: float = FINAL_LR_FACTOR
    update_numbers: str = "1_through_total_steps"


@dataclass(frozen=True)
class SplitBoundaries:
    train_start: date = TRAIN_START
    train_end: date = TRAIN_END
    validation_start: date = VALIDATION_START
    validation_end: date = VALIDATION_END
    test_start: date = TEST_START
    test_end: date = TEST_END
