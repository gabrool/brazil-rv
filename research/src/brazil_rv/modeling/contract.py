from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType

FEATURE_CONTRACT_VERSION = "M1_FEATURES_INTRADAY_DI_MASKED_CONTEXT_HUMAN_PRIORS_V4"


def resolve_project_root() -> Path:
    configured = os.environ.get("BRAZIL_RV_ROOT")
    root = (
        Path(configured).expanduser().resolve()
        if configured is not None
        else Path(__file__).absolute().parents[6]
    )
    if not (root / "quant" / "b3-quant" / "research").is_dir():
        raise FileNotFoundError(
            f"BRAZIL_RV_ROOT does not contain the research project: {root}"
        )
    return root


PROJECT_ROOT = resolve_project_root()
FEATURE_STORE_POINTER = (
    PROJECT_ROOT
    / "quant-data"
    / "b3"
    / "processed"
    / "features"
    / "m1_features_canonical_path.txt"
)
RUN_OUTPUT_BASE = PROJECT_ROOT / "quant-data" / "b3" / "processed" / "model_runs"

TRAIN_START = date(2021, 8, 16)
TRAIN_END = date(2024, 6, 28)
VALIDATION_START = date(2024, 7, 8)
VALIDATION_END = date(2025, 6, 30)
TEST_START = date(2025, 7, 7)
TEST_END = date(2026, 7, 17)

EXPECTED_DECISIONS_PER_DATE = 55
EXPECTED_SPLIT_DATE_COUNTS: Mapping[str, int] = MappingProxyType(
    {"train": 716, "embargo_1": 5, "validation": 244, "embargo_2": 4, "test": 259}
)
EXPECTED_SPLIT_SAMPLE_COUNTS: Mapping[str, int] = MappingProxyType(
    {
        name: count * EXPECTED_DECISIONS_PER_DATE
        for name, count in EXPECTED_SPLIT_DATE_COUNTS.items()
    }
)

EQUITY_COUNT = 158
LOCAL_CONTEXT_COUNT = 7
GLOBAL_CONTEXT_COUNT = 8
CONTEXT_COUNT = LOCAL_CONTEXT_COUNT + GLOBAL_CONTEXT_COUNT
INSTRUMENT_COUNT = EQUITY_COUNT + CONTEXT_COUNT
HORIZON_COUNT = 3
DYNAMIC_CHANNEL_COUNT = 26
SLOW_FEATURE_COUNT = 32
CONTEXT_GENERIC_DYNAMIC_COUNT = 16
LOCAL_CONTEXT_SYMBOLS = (
    "WIN$",
    "WDO$",
    "DI1F27",
    "DI1F28",
    "DI1F29",
    "DI1F31",
    "DI1$N",
)
GLOBAL_CONTEXT_SYMBOLS = (
    "ES.v.0",
    "NQ.v.0",
    "ZT.v.0",
    "ZN.v.0",
    "CL.v.0",
    "HG.v.0",
    "6E.v.0",
    "6M.v.0",
)
HORIZONS = (30, 60, 120)

# The completed context screen is the current feature policy.
CANONICAL_DROPPED_LOCAL_SLOTS = (0,)  # WIN$
CANONICAL_RETAINED_GLOBAL_SLOTS = (2, 3)  # ZT, ZN
CANONICAL_NEUTRALIZED_EQUITY_SLOW_INDICES = (20,)  # beta_to_WIN

PATCH_MINUTES = 5
PATCH_INPUT_WIDTH = PATCH_MINUTES * DYNAMIC_CHANNEL_COUNT
ABSOLUTE_PATCH_COUNT = 69
STATE_TOKEN_SLOT = 69
TEMPORAL_TOKEN_COUNT = 70
EQUITY_ABSOLUTE_START_PATCH = 12
GLOBAL_WINDOW_MINUTES = ABSOLUTE_PATCH_COUNT * PATCH_MINUTES
DECISION_GLOBAL_INDICES = tuple(345 + PATCH_MINUTES * index for index in range(55))
TABULAR_OFFSETS = (0, 15, 30, 60, 120)
TABULAR_FEATURE_COUNT = (
    SLOW_FEATURE_COUNT
    + DYNAMIC_CHANNEL_COUNT * len(TABULAR_OFFSETS)
    + CONTEXT_GENERIC_DYNAMIC_COUNT * CONTEXT_COUNT * len(TABULAR_OFFSETS)
    + SLOW_FEATURE_COUNT * CONTEXT_COUNT
    + 2
    + (1 + CONTEXT_COUNT) * len(TABULAR_OFFSETS)
    + LOCAL_CONTEXT_COUNT
    + GLOBAL_CONTEXT_COUNT
)

FAMILY_EQUITY = 0
FAMILY_EQUITY_FUTURE = 1
FAMILY_FX_FUTURE = 2
FAMILY_RATE_FUTURE = 3
FAMILY_EQUITY_INDEX = 4
FAMILY_RATE_TREASURY = 5
FAMILY_ENERGY = 6
FAMILY_INDUSTRIAL_METAL = 7
FAMILY_MAJOR_FX = 8
FAMILY_EMERGING_FX = 9
FAMILY_COUNT = 10
INSTRUMENT_FAMILY_IDS = (
    (FAMILY_EQUITY,) * EQUITY_COUNT
    + (FAMILY_EQUITY_FUTURE, FAMILY_FX_FUTURE)
    + (FAMILY_RATE_FUTURE,) * 5
    + (FAMILY_EQUITY_INDEX,) * 2
    + (FAMILY_RATE_TREASURY,) * 2
    + (FAMILY_ENERGY, FAMILY_INDUSTRIAL_METAL, FAMILY_MAJOR_FX, FAMILY_EMERGING_FX)
)

TRANSFORMER_MODELS = (
    "temporal_only",
    "context_only",
    "pooled_market",
    "context_pooled",
)
NEURAL_MODELS = (*TRANSFORMER_MODELS, "tcn", "mlp")
SUPPORTED_MODELS = (*NEURAL_MODELS, "xgboost")
GLOBAL_CONTEXT_SETTINGS = ("enabled", "masked")
OPTIMIZER_VARIANTS = ("adamw", "sam_adamw")
NEURAL_OBJECTIVES = ("soft_spearman", "rank_huber")
ALLOWED_SEEDS = (11, 29, 47)
DEFAULT_NEURAL_OBJECTIVE = "soft_spearman"
SOFT_RANK_TEMPERATURES = (0.05, 0.10, 0.20, 0.50)
SAM_RHOS = (0.025, 0.050, 0.075, 0.100, 0.125)
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

TCN_KERNEL_SIZE = 3
TCN_FUSIONS = ("none", "context_only", "pooled_market", "context_pooled")
TCN_WIDTHS = (64, 128, 192, 256)
TCN_BLOCK_VARIANTS = ("gelu", "silu", "swiglu")
TCN_RECEPTIVE_FIELDS: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {
        "short": (1, 1, 1, 1, 1, 2),
        "medium": (1, 2, 2, 2, 4, 4),
        "long": (1, 2, 4, 4, 4, 8),
        "full": (1, 2, 4, 8, 16, 32),
        "matched_full": (1, 2, 4, 8, 8, 12),
    }
)
TCN_SWIGLU_HIDDEN_WIDTHS: Mapping[int, int] = MappingProxyType(
    {64: 24, 128: 40, 192: 64, 256: 88}
)
CONTEXT_ROUTING_MODES = ("late_only", "early_concat", "film", "early_concat_film")
CONTEXT_ROUTING_MACRO_SYMBOLS = (
    "WDO$",
    "DI1F27",
    "DI1F28",
    "DI1F29",
    "DI1F31",
    "DI1$N",
    "ZT.v.0",
    "ZN.v.0",
)
CONTEXT_ROUTING_LOCAL_SOURCE_COUNT = 6
CONTEXT_ROUTING_GLOBAL_SOURCE_COUNT = 2
CONTEXT_ROUTING_SOURCE_COUNT = 8
CONTEXT_ROUTING_PATCH_SOURCE_WIDTH = PATCH_MINUTES * CONTEXT_GENERIC_DYNAMIC_COUNT
CONTEXT_ROUTING_MACRO_EARLY_SOURCE_WIDTH = CONTEXT_ROUTING_PATCH_SOURCE_WIDTH + 1
CONTEXT_ROUTING_EXCLUDED_GLOBAL_SLOW_CHANNEL = (
    "previous_b3_close_to_decision_return_normalized"
)
CONTEXT_ROUTING_GLOBAL_SLOW_WIDTH = SLOW_FEATURE_COUNT - 1
CONTEXT_ROUTING_MACRO_SLOW_INPUT_WIDTH = (
    CONTEXT_ROUTING_LOCAL_SOURCE_COUNT * SLOW_FEATURE_COUNT
    + CONTEXT_ROUTING_GLOBAL_SOURCE_COUNT * CONTEXT_ROUTING_GLOBAL_SLOW_WIDTH
    + CONTEXT_ROUTING_SOURCE_COUNT
)

# Only the selected sector/subsector representation is a current model input.
PEER_FEATURE_MODES = ("none", "selected")
PEER_STATE_WIDTH = 6
PEER_STATE_ORDER = (
    "return_vs_selected_peer_median_15m",
    "return_vs_selected_peer_median_60m",
    "selected_peer_return_rank_15m",
    "selected_peer_return_rank_60m",
    "selected_peer_15m_valid",
    "selected_peer_60m_valid",
)

EFFECTIVE_BATCH_SIZE = 512
MAX_EPOCHS = 20
EARLY_STOP_PATIENCE = 5
MIN_IC_IMPROVEMENT = 1e-4
GRADIENT_CLIP = 1.0
HUBER_DELTA = 1.0
MIN_IC_EQUITIES = 30

ADAMW_LR = 3e-4
ADAMW_BETAS = (0.9, 0.95)
ADAMW_EPS = 1e-8
ADAMW_WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.05
FINAL_LR_FACTOR = 0.1

MUON_LR = 0.02
MUON_MOMENTUM = 0.95
MUON_NESTEROV = True
MUON_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
MUON_EPS = 1e-7
MUON_NS_STEPS = 5
MUON_WEIGHT_DECAY = 0.01
MUON_ADJUST_LR_FN = "original"

MLP_WIDTH = 256
MLP_DEPTH = 3
MLP_SWIGLU_WIDTH = 512

XGBOOST_VERSION = "3.2.0"
XGBOOST_OBJECTIVE = "reg:pseudohubererror"
XGBOOST_HUBER_SLOPE = 1.0
XGBOOST_TREE_METHOD = "hist"
XGBOOST_DEVICE = "cuda"
XGBOOST_INNER_VALIDATION_FRACTION = 0.20
XGBOOST_INNER_EMBARGO_DATES = 5
XGBOOST_EARLY_STOPPING_ROUNDS = 50
XGBOOST_MAX_BOOSTING_ROUNDS = 4_000

SANITY_SMOKE_SAMPLE_COUNT = 512
SANITY_MEMORIZATION_SAMPLE_COUNT = 8
SANITY_DECISION_INDEX = 27
SANITY_MAX_STEPS = 1_000
SANITY_MAX_LOSS = 0.10
SANITY_MAX_LOSS_RATIO = 0.50
SANITY_MIN_SPEARMAN = 0.50


@dataclass(frozen=True)
class RuntimeSettings:
    microbatch_size: int = 64
    accumulation_steps: int = 8
    evaluation_batch_size: int = 256
    num_workers: int = 8
    prefetch_factor: int = 4
    compile_backend: str = "inductor"
    compile_mode: str = "default"
    compile_fullgraph: bool = True
    compile_dynamic: bool = False


GH200_RUNTIME = RuntimeSettings()


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
class TCNSettings:
    fusion: str = "context_pooled"
    width: int = 64
    receptive_field: str = "full"
    block: str = "swiglu"
    slow_routing: str = "late_only"
    macro_temporal_routing: str = "late_only"


BASELINE_TCN_SETTINGS = TCNSettings()


@dataclass(frozen=True)
class TCNArchitecture:
    family: str
    fusion_mode: str
    receptive_field: str
    block: str
    patch_input_width: int
    width: int
    swiglu_hidden_width: int | None
    residual_blocks: int
    kernel_size: int
    dilations: tuple[int, ...]
    slow_width: int
    fusion_states: int
    theoretical_receptive_field_patches: int
    theoretical_receptive_field_minutes: int
    maximum_effective_equity_receptive_field_patches: int
    maximum_effective_equity_receptive_field_minutes: int
    maximum_effective_context_receptive_field_patches: int | None
    maximum_effective_context_receptive_field_minutes: int | None
    fusion_width: int
    dropout: float
    output_horizons: int
    slow_routing: str
    macro_temporal_routing: str
    context_routing_rank: int


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


NeuralArchitecture = TransformerArchitecture | TCNArchitecture | MLPArchitecture

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
NEURAL_ARCHITECTURES: Mapping[str, TransformerArchitecture | MLPArchitecture] = (
    MappingProxyType(
        {
            "temporal_only": TransformerArchitecture(
                **_SHARED_TRANSFORMER,
                context_memory_tokens=0,
                pooled_memory_tokens=0,
                fusion_blocks=0,
            ),
            "context_only": TransformerArchitecture(
                **_SHARED_TRANSFORMER,
                context_memory_tokens=CONTEXT_COUNT,
                pooled_memory_tokens=0,
                fusion_blocks=1,
            ),
            "pooled_market": TransformerArchitecture(
                **_SHARED_TRANSFORMER,
                context_memory_tokens=0,
                pooled_memory_tokens=2 + POOLED_INDUCING_TOKEN_COUNT,
                fusion_blocks=1,
            ),
            "context_pooled": TransformerArchitecture(
                **_SHARED_TRANSFORMER,
                context_memory_tokens=CONTEXT_COUNT,
                pooled_memory_tokens=2 + POOLED_INDUCING_TOKEN_COUNT,
                fusion_blocks=1,
            ),
            "mlp": MLPArchitecture(
                "mlp",
                TABULAR_FEATURE_COUNT,
                MLP_WIDTH,
                MLP_DEPTH,
                MLP_SWIGLU_WIDTH,
                RMS_NORM_EPS,
                RESIDUAL_DROPOUT,
                HORIZON_COUNT,
            ),
        }
    )
)


def architecture_for_model(
    model_name: str, tcn_settings: TCNSettings | None = None
) -> NeuralArchitecture:
    if model_name == "tcn":
        if tcn_settings is None:
            raise ValueError("TCN settings are required")
        return resolve_tcn_architecture(tcn_settings)
    if tcn_settings is not None:
        raise ValueError(f"TCN settings are forbidden for {model_name}")
    try:
        return NEURAL_ARCHITECTURES[model_name]
    except KeyError as error:
        raise ValueError(f"Unknown neural model: {model_name}") from error


def model_consumes_context(
    model_name: str, tcn_settings: TCNSettings | None = None
) -> bool:
    if model_name == "tcn":
        if tcn_settings is None:
            raise ValueError("TCN settings are required")
        return tcn_settings.fusion in ("context_only", "context_pooled")
    if tcn_settings is not None:
        raise ValueError(f"TCN settings are forbidden for {model_name}")
    return model_name in ("context_only", "context_pooled", "mlp", "xgboost")


def validate_peer_feature_mode(model_name: str, mode: str) -> str:
    if mode not in PEER_FEATURE_MODES:
        raise ValueError(f"Invalid peer-feature mode: {mode}")
    if mode != "none" and model_name != "tcn":
        raise ValueError("Peer features are supported only for TCN")
    return mode


def resolve_tcn_architecture(settings: TCNSettings) -> TCNArchitecture:
    if (
        settings.fusion not in TCN_FUSIONS
        or settings.width not in TCN_WIDTHS
        or settings.block not in TCN_BLOCK_VARIANTS
    ):
        raise ValueError(f"Invalid TCN settings: {settings}")
    if (
        settings.slow_routing not in CONTEXT_ROUTING_MODES
        or settings.macro_temporal_routing not in CONTEXT_ROUTING_MODES
    ):
        raise ValueError(f"Invalid context routing: {settings}")
    if (
        settings.slow_routing != "late_only"
        or settings.macro_temporal_routing != "late_only"
    ) and settings.fusion != "context_pooled":
        raise ValueError("Early or FiLM routing requires context_pooled fusion")
    try:
        dilations = TCN_RECEPTIVE_FIELDS[settings.receptive_field]
    except KeyError as error:
        raise ValueError(
            f"Invalid TCN receptive field: {settings.receptive_field}"
        ) from error
    fusion_states = {
        "none": 0,
        "context_only": 1 + CONTEXT_COUNT,
        "pooled_market": 3,
        "context_pooled": 3 + CONTEXT_COUNT,
    }[settings.fusion]
    theoretical = 1 + (TCN_KERNEL_SIZE - 1) * sum(dilations)
    equity_patches = min(
        theoretical, ABSOLUTE_PATCH_COUNT - EQUITY_ABSOLUTE_START_PATCH
    )
    context_patches = (
        min(theoretical, ABSOLUTE_PATCH_COUNT)
        if settings.fusion in ("context_only", "context_pooled")
        else None
    )
    return TCNArchitecture(
        "tcn",
        settings.fusion,
        settings.receptive_field,
        settings.block,
        PATCH_INPUT_WIDTH,
        settings.width,
        TCN_SWIGLU_HIDDEN_WIDTHS[settings.width]
        if settings.block == "swiglu"
        else None,
        len(dilations),
        TCN_KERNEL_SIZE,
        dilations,
        SLOW_FEATURE_COUNT,
        fusion_states,
        theoretical,
        theoretical * PATCH_MINUTES,
        equity_patches,
        equity_patches * PATCH_MINUTES,
        context_patches,
        None if context_patches is None else context_patches * PATCH_MINUTES,
        2 * settings.width,
        RESIDUAL_DROPOUT,
        HORIZON_COUNT,
        settings.slow_routing,
        settings.macro_temporal_routing,
        min(32, settings.width),
    )


def routing_enabled(architecture: TCNArchitecture) -> bool:
    return (
        architecture.slow_routing != "late_only"
        or architecture.macro_temporal_routing != "late_only"
    )


def context_routing_metadata(architecture: TCNArchitecture) -> dict[str, object]:
    return {
        "slow_routing": architecture.slow_routing,
        "macro_temporal_routing": architecture.macro_temporal_routing,
        "ordered_sources": list(CONTEXT_ROUTING_MACRO_SYMBOLS),
    }


def peer_feature_metadata(
    model_name: str, architecture: NeuralArchitecture | None, mode: str
) -> dict[str, object]:
    mode = validate_peer_feature_mode(model_name, mode)
    return {
        "mode": mode,
        "state_order": list(PEER_STATE_ORDER),
        "state_width": PEER_STATE_WIDTH,
    }


@dataclass(frozen=True)
class XGBoostCandidate:
    max_depth: int
    learning_rate: float
    min_child_weight: int


XGBOOST_CANDIDATES = tuple(
    XGBoostCandidate(depth, rate, weight)
    for depth in (4, 6, 8)
    for rate in (0.03, 0.07)
    for weight in (10, 50)
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
