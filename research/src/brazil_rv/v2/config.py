from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .contract import (
    ALLOWED_LOOKBACKS,
    ALLOWED_SEEDS,
    DEFAULT_LOOKBACK,
    DECISION_MINUTE_INDEX,
    GBDT_SEEDS as GBDT_SEEDS,
    HORIZONS as HORIZONS,
    PRIMARY_HORIZONS as PRIMARY_HORIZONS,
    SOFT_RANK_TEMPERATURE,
)

PERSISTENCE_WEIGHTS = (0.0, 0.1, 0.3)
SOFT_RANK_TEMPERATURES = (SOFT_RANK_TEMPERATURE, 1.0)
DECISION_CUTOFF_INDEX = DECISION_MINUTE_INDEX
FAST_PRESENT_FLAG = "fast_present"
DAYS_SINCE_LAST_SLOW_ROW_FLAG = "days_since_last_slow_row"
PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROTOCOL_CONFIG_ROOT = PROJECT_ROOT / "research" / "configs" / "v2"


@dataclass(frozen=True)
class ModelConfig:
    """Frozen starter-model controls that are independent of a store manifest."""

    slow_feature_count: int
    slow_lookback: int = DEFAULT_LOOKBACK
    gru_layers: int = 1
    hidden_width: int = 64
    fusion_width: int = 128
    trunk_blocks: int = 2
    trunk_swiglu_hidden: int = 48
    dropout: float = 0.1
    fast_pretrained: bool = False
    fast_pretrained_checkpoint: Path | None = None
    fast_pretrained_sha256: str | None = None
    lambda_persistence: float = 0.0
    soft_rank_temperature: float = SOFT_RANK_TEMPERATURE
    use_bf16: bool = False
    compile_forward: bool = True
    time_decay_half_life_sessions: float | None = None

    def __post_init__(self) -> None:
        if self.slow_feature_count <= 0:
            raise ValueError("slow_feature_count must come from a nonempty store")
        if self.slow_lookback not in ALLOWED_LOOKBACKS:
            raise ValueError("slow_lookback must be 20, 60, or 120 sessions")
        if self.gru_layers not in (1, 2):
            raise ValueError("gru_layers must be one or two")
        if (
            min(
                self.hidden_width,
                self.fusion_width,
                self.trunk_blocks,
                self.trunk_swiglu_hidden,
            )
            <= 0
        ):
            raise ValueError("model widths and block count must be positive")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")
        if self.lambda_persistence not in PERSISTENCE_WEIGHTS:
            raise ValueError("lambda_persistence is outside the frozen grid")
        if self.soft_rank_temperature not in SOFT_RANK_TEMPERATURES:
            raise ValueError("soft-rank temperature is outside the frozen grid")
        if self.time_decay_half_life_sessions not in (None, 756.0):
            raise ValueError("time-decay half-life must be None or 756 sessions")
        has_fast_path = self.fast_pretrained_checkpoint is not None
        has_fast_sha = self.fast_pretrained_sha256 is not None
        if self.fast_pretrained != has_fast_path or has_fast_path != has_fast_sha:
            raise ValueError(
                "fast_pretrained, its checkpoint, and its expected SHA-256 must be set together"
            )
        if self.fast_pretrained_sha256 is not None and (
            len(self.fast_pretrained_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.fast_pretrained_sha256
            )
        ):
            raise ValueError("fast_pretrained_sha256 must be lowercase hexadecimal")


@dataclass(frozen=True)
class ProtocolPreset:
    name: str
    folds: tuple[str, ...]
    seeds: tuple[int, ...]
    max_epochs_override: int | None
    bootstrap_replications: int
    bootstrap_block_length: int
    max_parallel: int

    def __post_init__(self) -> None:
        if not self.name or not self.folds or not self.seeds:
            raise ValueError("protocol name, folds, and seeds must be nonempty")
        if any(seed not in ALLOWED_SEEDS for seed in self.seeds):
            raise ValueError("protocol contains a non-screen seed")
        if self.max_epochs_override is not None and self.max_epochs_override <= 0:
            raise ValueError("max_epochs_override must be positive")
        if self.bootstrap_replications < 0 or self.bootstrap_block_length != 20:
            raise ValueError("protocol bootstrap settings differ from v2")
        if not 1 <= self.max_parallel <= 6:
            raise ValueError("max_parallel must be between one and six")


TRIAGE_PROTOCOL = ProtocolPreset(
    name="triage",
    folds=("F1", "F2"),
    seeds=(11,),
    max_epochs_override=None,
    bootstrap_replications=0,
    bootstrap_block_length=20,
    max_parallel=1,
)
FULL_PROTOCOL = ProtocolPreset(
    name="full",
    folds=("F1", "F2", "F3"),
    seeds=ALLOWED_SEEDS,
    max_epochs_override=None,
    bootstrap_replications=10_000,
    bootstrap_block_length=20,
    max_parallel=6,
)
PROTOCOL_PRESETS = {
    TRIAGE_PROTOCOL.name: TRIAGE_PROTOCOL,
    FULL_PROTOCOL.name: FULL_PROTOCOL,
}


def _expected_protocol_payload(preset: ProtocolPreset) -> dict[str, object]:
    return {
        "schema": "BRAZIL_RV_V2_PROTOCOL_V1",
        "name": preset.name,
        "folds": list(preset.folds),
        "seeds": list(preset.seeds),
        "decision_minute_index": DECISION_MINUTE_INDEX,
        "slow_lookback": DEFAULT_LOOKBACK,
        "horizons_sessions": list(HORIZONS),
        "training": {
            "date_pairs_per_batch": 8,
            "epochs": 20,
            "patience": 3,
            "sam_rho": 0.125,
            "learning_rate": 0.0003,
            "pretrained_learning_rate_multiplier": 0.3,
            "weight_decay": 0.01,
            "ema_decay": 0.995,
            "lambda_persistence": 0.0,
            "use_bf16": False,
        },
        "evaluation": {
            "paired_bootstrap_replications": preset.bootstrap_replications,
            "paired_bootstrap_block_sessions": preset.bootstrap_block_length,
        },
        "max_parallel_trajectories": preset.max_parallel,
        "official_registration": None,
    }


def _exact_json_match(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    options = {"sort_keys": True, "separators": (",", ":"), "allow_nan": False}
    return json.dumps(left, **options) == json.dumps(right, **options)


def load_protocol_preset(path: Path) -> ProtocolPreset:
    """Load a named protocol only when every frozen JSON field is exact."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("v2 protocol config must contain a JSON object")
    name = payload.get("name")
    if not isinstance(name, str) or name not in PROTOCOL_PRESETS:
        raise ValueError(f"Unknown v2 protocol preset: {name}")
    preset = PROTOCOL_PRESETS[name]
    if path.stem != name:
        raise ValueError("v2 protocol filename must equal its frozen preset name")
    expected = _expected_protocol_payload(preset)
    if not _exact_json_match(payload, expected):
        raise ValueError(
            f"v2 protocol config differs from the frozen {name!r} preset"
        )
    return preset


def protocol_preset(
    name: str, *, config_root: Path = PROTOCOL_CONFIG_ROOT
) -> ProtocolPreset:
    try:
        PROTOCOL_PRESETS[name]
    except KeyError as error:
        raise ValueError(f"Unknown v2 protocol preset: {name}") from error
    return load_protocol_preset(config_root / f"{name}.json")
