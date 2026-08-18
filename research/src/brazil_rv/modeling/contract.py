from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PureWindowsPath
from types import MappingProxyType


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


def workspace_path(
    value: str | Path,
    *,
    must_exist: bool = True,
    project_root: Path | None = None,
) -> Path:
    raw = os.fspath(value)
    if not raw.strip():
        raise ValueError("Workspace path is empty")
    native = Path(raw)
    normalized = raw.replace("\\", "/")
    lowered = normalized.casefold()
    relative: str | None = None
    recorded_root: str | None = None
    for prefix in ("c:/brazil-rv/quant-data", "c:/quant-data"):
        if lowered == prefix or lowered.startswith(prefix + "/"):
            recorded_root = normalized[: len(prefix)]
            relative = normalized[len(prefix) :].lstrip("/")
            break
    if relative is not None:
        depth = 0
        for part in relative.split("/"):
            if part in ("", "."):
                continue
            if part == "..":
                if depth == 0:
                    raise ValueError(f"Workspace path escapes quant-data: {value}")
                depth -= 1
            else:
                depth += 1
        if os.name == "nt" and native.is_absolute() and native.exists():
            assert recorded_root is not None
            data_root = Path(recorded_root).resolve()
            path = native.resolve()
        else:
            root = (PROJECT_ROOT if project_root is None else project_root).resolve()
            data_root = (root / "quant-data").resolve()
            path = (data_root / relative).resolve()
        if not path.is_relative_to(data_root):
            raise ValueError(f"Workspace path escapes quant-data: {value}")
    elif native.is_absolute():
        path = native.resolve()
    elif PureWindowsPath(raw).is_absolute():
        raise ValueError(f"Unsupported Windows workspace path: {value}")
    else:
        raise ValueError(f"Workspace path must be absolute: {value}")
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


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
HORIZONS = (30, 60, 120)

# This is the accepted context screen, not an experiment switch.
CANONICAL_DROPPED_LOCAL_SLOTS = (0,)  # WIN$
CANONICAL_RETAINED_GLOBAL_SLOTS = (2, 3)  # ZT, ZN
CANONICAL_NEUTRALIZED_EQUITY_SLOW_INDICES = (20,)  # beta_to_WIN

PATCH_MINUTES = 5
PATCH_INPUT_WIDTH = PATCH_MINUTES * DYNAMIC_CHANNEL_COUNT
ABSOLUTE_PATCH_COUNT = 69
STATE_TOKEN_SLOT = 69
EQUITY_ABSOLUTE_START_PATCH = 12
GLOBAL_WINDOW_MINUTES = ABSOLUTE_PATCH_COUNT * PATCH_MINUTES
DECISION_GLOBAL_INDICES = tuple(345 + PATCH_MINUTES * index for index in range(55))

TCN_WIDTH = 64
TCN_KERNEL_SIZE = 3
TCN_DILATIONS = (1, 2, 4, 8, 16, 32)
TCN_SWIGLU_HIDDEN_WIDTH = 24
TCN_FUSION_WIDTH = 2 * TCN_WIDTH
TCN_ATTENTION_HEADS = 4
RESIDUAL_DROPOUT = 0.10
TARGETED_FUSION_GATE_BIAS = -2.0

ALLOWED_SEEDS = (11, 29, 47)
RECENCY_POLICIES = ("uniform", "exp_504", "exp_252", "exp_126", "rolling_504")
RECENCY_HALF_LIVES: Mapping[str, int] = MappingProxyType(
    {"exp_504": 504, "exp_252": 252, "exp_126": 126}
)
ROLLING_WINDOW_DATES = 504

EFFECTIVE_BATCH_SIZE = 512
MAX_EPOCHS = 30
EARLY_STOP_PATIENCE = 7
MIN_IC_IMPROVEMENT = 1e-4
GRADIENT_CLIP = 1.0
MIN_IC_EQUITIES = 30
SOFT_RANK_TEMPERATURE = 0.50
SOFT_RANK_STANDARDIZATION_EPS = 1e-6
SOFT_SPEARMAN_CORRELATION_EPS = 1e-8
SAM_RHO = 0.125
SAM_NORM_EPS = 1e-12

ADAMW_LR = 3e-4
ADAMW_BETAS = (0.9, 0.95)
ADAMW_EPS = 1e-8
ADAMW_WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.05
FINAL_LR_FACTOR = 0.1


@dataclass(frozen=True)
class RuntimeSettings:
    effective_batch_size: int = EFFECTIVE_BATCH_SIZE
    loader_batch_size: int = 256
    microbatch_size: int = 256
    evaluation_batch_size: int = 256
    num_workers: int = 8
    prefetch_factor: int = 4
    compile_backend: str = "inductor"
    compile_mode: str = "default"
    compile_fullgraph: bool = True
    compile_dynamic: bool = False

    def __post_init__(self) -> None:
        if self.effective_batch_size % self.loader_batch_size:
            raise ValueError("Effective batch must divide into loader batches")
        if self.loader_batch_size % self.microbatch_size:
            raise ValueError("Loader batch must divide into microbatches")

    @property
    def loader_batches_per_effective_batch(self) -> int:
        return self.effective_batch_size // self.loader_batch_size


GH200_RUNTIME = RuntimeSettings()


@dataclass(frozen=True)
class TCNArchitecture:
    family: str = "tcn"
    patch_input_width: int = PATCH_INPUT_WIDTH
    width: int = TCN_WIDTH
    swiglu_hidden_width: int = TCN_SWIGLU_HIDDEN_WIDTH
    residual_blocks: int = len(TCN_DILATIONS)
    kernel_size: int = TCN_KERNEL_SIZE
    dilations: tuple[int, ...] = TCN_DILATIONS
    slow_width: int = SLOW_FEATURE_COUNT
    fusion_states: int = CONTEXT_COUNT + 3
    fusion_width: int = TCN_FUSION_WIDTH
    dropout: float = RESIDUAL_DROPOUT
    output_horizons: int = HORIZON_COUNT


TCN_ARCHITECTURE = TCNArchitecture()
