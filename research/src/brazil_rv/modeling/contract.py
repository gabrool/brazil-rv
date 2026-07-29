from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType

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

EXPECTED_DATE_COUNT = 1248
EXPECTED_SAMPLE_COUNT = 59_565
EXPECTED_DECISIONS_PER_DATE = 55
EQUITY_COUNT = 158
CONTEXT_COUNT = 6
INSTRUMENT_COUNT = EQUITY_COUNT + CONTEXT_COUNT
HORIZON_COUNT = 3
DYNAMIC_CHANNEL_COUNT = 26
SLOW_FEATURE_COUNT = 32
EQUITY_SLOW_COUNT = SLOW_FEATURE_COUNT
CONTEXT_SLOW_COUNT = SLOW_FEATURE_COUNT
CONTEXT_GENERIC_DYNAMIC_COUNT = 16
CONTEXT_SYMBOLS = ("WIN$", "WDO$", "DI1F27", "DI1F28", "DI1F29", "DI1F31")
HORIZONS = (30, 60, 120)

PATCH_MINUTES = 5
PATCH_INPUT_WIDTH = PATCH_MINUTES * DYNAMIC_CHANNEL_COUNT
ABSOLUTE_PATCH_COUNT = 69
STATE_TOKEN_SLOT = 69
TEMPORAL_TOKEN_COUNT = 70
EQUITY_ABSOLUTE_START_PATCH = 12
TABULAR_OFFSETS = (0, 15, 30, 60, 120)
TABULAR_VALIDITY_COUNT = (1 + CONTEXT_COUNT) * len(TABULAR_OFFSETS)
TABULAR_FEATURE_COUNT = (
    SLOW_FEATURE_COUNT
    + DYNAMIC_CHANNEL_COUNT * len(TABULAR_OFFSETS)
    + CONTEXT_GENERIC_DYNAMIC_COUNT * CONTEXT_COUNT * len(TABULAR_OFFSETS)
    + SLOW_FEATURE_COUNT * CONTEXT_COUNT
    + 2
    + TABULAR_VALIDITY_COUNT
)
if PATCH_INPUT_WIDTH != 130 or TABULAR_FEATURE_COUNT != 871:
    raise ValueError("Model input widths do not match the feature contract")

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

TRANSFORMER_MODELS = (
    "temporal_only",
    "context_only",
    "pooled_market",
    "context_pooled",
)
NEURAL_MODELS = (*TRANSFORMER_MODELS, "tcn", "mlp")
SUPPORTED_MODELS = (*NEURAL_MODELS, "xgboost")
OPTIMIZER_VARIANTS = ("adamw", "sam_adamw")
ALLOWED_SEEDS = (11, 29, 47)
SOFT_RANK_TEMPERATURES = (0.05, 0.10, 0.20, 0.50)
SAM_RHOS = (0.0025, 0.005, 0.010, 0.020, 0.035, 0.050)
SOFT_RANK_STANDARDIZATION_EPS = 1e-6
SOFT_SPEARMAN_CORRELATION_EPS = 1e-8
SAM_NORM_EPS = 1e-12

RMS_NORM_EPS = 1e-6
QK_NORM_EPS = 1e-6
ROPE_BASE = 10_000.0
INPUT_DROPOUT = 0.05
RESIDUAL_DROPOUT = 0.10
ATTENTION_DROPOUT = 0.0
TARGETED_FUSION_GATE_BIAS = -2.0
POOLED_INDUCING_TOKEN_COUNT = 4

TCN_WIDTH = 128
TCN_DEPTH = 6
TCN_KERNEL_SIZE = 3
TCN_DILATIONS = (1, 2, 4, 8, 16, 32)
TCN_FUSION_WIDTH = 256
MLP_WIDTH = 256
MLP_DEPTH = 3
MLP_SWIGLU_WIDTH = 512

EFFECTIVE_BATCH_SIZE = 512
MAX_EPOCHS = 20
EARLY_STOP_PATIENCE = 5
MIN_IC_IMPROVEMENT = 1e-4
GRADIENT_CLIP = 1.0

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
COMPILE_PARITY_PREDICTION_ATOL = 2e-2
COMPILE_PARITY_PREDICTION_RTOL = 5e-3
COMPILE_PARITY_LOSS_ATOL = 5e-4
COMPILE_PARITY_LOSS_RTOL = 5e-3
COMPILE_PARITY_GRADIENT_RELATIVE_L2_MAX = 1e-2
COMPILE_PARITY_GRADIENT_COSINE_MIN = 0.9999
COMPILE_PARITY_GRADIENT_MAX_ABSOLUTE_ATOL = 1e-3
COMPILE_PARITY_GRADIENT_MAX_ABSOLUTE_RTOL = 1e-2

MIN_IC_EQUITIES = 30
SANITY_SMOKE_SAMPLE_COUNT = 512
SANITY_MEMORIZATION_SAMPLE_COUNT = 8
SANITY_DECISION_INDEX = 27
SANITY_MAX_STEPS = 1_000
SANITY_MAX_LOSS = 0.10
SANITY_MIN_SPEARMAN = 0.90

XGBOOST_VERSION = "3.2.0"
XGBOOST_OBJECTIVE = "reg:pseudohubererror"
XGBOOST_HUBER_SLOPE = 1.0
XGBOOST_TREE_METHOD = "hist"
XGBOOST_DEVICE = "cuda"
XGBOOST_INNER_VALIDATION_FRACTION = 0.20
XGBOOST_INNER_EMBARGO_DATES = 5
XGBOOST_EARLY_STOPPING_ROUNDS = 50
XGBOOST_MAX_BOOSTING_ROUNDS = 4_000

EXPECTED_ARRAY_SHAPES = {
    "equity_features.npy": (1248, 158, 405, 26),
    "equity_slow.npy": (1248, 158, 32),
    "equity_membership.npy": (1248, 158),
    "equity_data_ready.npy": (1248, 158),
    "context_features.npy": (1248, 6, 465, 26),
    "context_slow.npy": (1248, 6, 32),
    "context_data_ready.npy": (1248, 6),
    "raw_returns.npy": (1248, 158, 55, 3),
    "targets.npy": (1248, 158, 55, 3),
    "label_mask.npy": (1248, 158, 55, 3),
    "cross_section_median.npy": (1248, 55, 3),
    "horizon_mask.npy": (1248, 55, 3),
}


@dataclass(frozen=True)
class RuntimeSettings:
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
    required_cpu_architecture: str


GH200_RUNTIME = RuntimeSettings(
    microbatch_size=64,
    accumulation_steps=8,
    evaluation_batch_size=256,
    num_workers=8,
    prefetch_factor=4,
    compile_backend="inductor",
    compile_mode="reduce-overhead",
    compile_fullgraph=True,
    compile_dynamic=False,
    minimum_vram_bytes=90 * 1024**3,
    expected_compute_capability=(9, 0),
    required_cpu_architecture="aarch64",
)
if (
    GH200_RUNTIME.microbatch_size * GH200_RUNTIME.accumulation_steps
    != EFFECTIVE_BATCH_SIZE
):
    raise ValueError("GH200 physical and accumulated batches must equal 512")


@dataclass(frozen=True)
class HardwareInfo:
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
class TransformerArchitecture:
    family: str
    d_model: int
    attention_heads: int
    head_dim: int
    temporal_depth: int
    swiglu_width: int
    rms_norm_eps: float
    qk_norm_eps: float
    rope_base: float
    input_dropout: float
    residual_dropout: float
    attention_dropout: float
    context_memory_tokens: int
    pooled_memory_tokens: int
    fusion_blocks: int
    output_horizons: int


@dataclass(frozen=True)
class TCNArchitecture:
    family: str
    patch_input_width: int
    width: int
    residual_blocks: int
    kernel_size: int
    dilations: tuple[int, ...]
    slow_width: int
    fusion_states: int
    fusion_width: int
    dropout: float
    output_horizons: int


@dataclass(frozen=True)
class MLPArchitecture:
    family: str
    input_width: int
    hidden_width: int
    residual_blocks: int
    swiglu_width: int
    norm_eps: float
    dropout: float
    output_horizons: int


_SHARED_TRANSFORMER = {
    "family": "transformer",
    "d_model": 256,
    "attention_heads": 8,
    "head_dim": 32,
    "temporal_depth": 2,
    "swiglu_width": 704,
    "rms_norm_eps": RMS_NORM_EPS,
    "qk_norm_eps": QK_NORM_EPS,
    "rope_base": ROPE_BASE,
    "input_dropout": INPUT_DROPOUT,
    "residual_dropout": RESIDUAL_DROPOUT,
    "attention_dropout": ATTENTION_DROPOUT,
    "output_horizons": HORIZON_COUNT,
}
NEURAL_ARCHITECTURES: Mapping[
    str, TransformerArchitecture | TCNArchitecture | MLPArchitecture
] = MappingProxyType(
    {
        "temporal_only": TransformerArchitecture(
            **_SHARED_TRANSFORMER,
            context_memory_tokens=0,
            pooled_memory_tokens=0,
            fusion_blocks=0,
        ),
        "context_only": TransformerArchitecture(
            **_SHARED_TRANSFORMER,
            context_memory_tokens=6,
            pooled_memory_tokens=0,
            fusion_blocks=1,
        ),
        "pooled_market": TransformerArchitecture(
            **_SHARED_TRANSFORMER,
            context_memory_tokens=0,
            pooled_memory_tokens=6,
            fusion_blocks=1,
        ),
        "context_pooled": TransformerArchitecture(
            **_SHARED_TRANSFORMER,
            context_memory_tokens=6,
            pooled_memory_tokens=6,
            fusion_blocks=1,
        ),
        "tcn": TCNArchitecture(
            family="tcn",
            patch_input_width=PATCH_INPUT_WIDTH,
            width=TCN_WIDTH,
            residual_blocks=TCN_DEPTH,
            kernel_size=TCN_KERNEL_SIZE,
            dilations=TCN_DILATIONS,
            slow_width=SLOW_FEATURE_COUNT,
            fusion_states=9,
            fusion_width=TCN_FUSION_WIDTH,
            dropout=RESIDUAL_DROPOUT,
            output_horizons=HORIZON_COUNT,
        ),
        "mlp": MLPArchitecture(
            family="mlp",
            input_width=TABULAR_FEATURE_COUNT,
            hidden_width=MLP_WIDTH,
            residual_blocks=MLP_DEPTH,
            swiglu_width=MLP_SWIGLU_WIDTH,
            norm_eps=RMS_NORM_EPS,
            dropout=RESIDUAL_DROPOUT,
            output_horizons=HORIZON_COUNT,
        ),
    }
)
EXPECTED_TRAINABLE_PARAMETER_COUNTS: Mapping[str, int] = MappingProxyType(
    {
        "temporal_only": 1_668_611,
        "context_only": 2_603_779,
        "pooled_market": 3_539_715,
        "context_pooled": 3_539_715,
        "tcn": 777_987,
        "mlp": 1_404_675,
    }
)


def architecture_for_model(
    model_name: str,
) -> TransformerArchitecture | TCNArchitecture | MLPArchitecture:
    try:
        return NEURAL_ARCHITECTURES[model_name]
    except KeyError as error:
        raise ValueError(f"Unknown neural model: {model_name}") from error


@dataclass(frozen=True)
class XGBoostCandidate:
    max_depth: int
    learning_rate: float
    min_child_weight: int


XGBOOST_CANDIDATES = tuple(
    XGBoostCandidate(max_depth, learning_rate, min_child_weight)
    for max_depth in (4, 6, 8)
    for learning_rate in (0.03, 0.07)
    for min_child_weight in (10, 50)
)
XGBOOST_FIXED_PARAMETERS: Mapping[str, object] = MappingProxyType(
    {
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "reg_lambda": 5.0,
        "reg_alpha": 0.1,
        "max_bin": 256,
        "objective": XGBOOST_OBJECTIVE,
        "huber_slope": XGBOOST_HUBER_SLOPE,
        "tree_method": XGBOOST_TREE_METHOD,
        "device": XGBOOST_DEVICE,
    }
)


@dataclass(frozen=True)
class TrainingConstants:
    effective_batch_size: int = EFFECTIVE_BATCH_SIZE
    maximum_epochs: int = MAX_EPOCHS
    early_stop_patience: int = EARLY_STOP_PATIENCE
    minimum_ic_improvement: float = MIN_IC_IMPROVEMENT
    gradient_clip: float = GRADIENT_CLIP


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
