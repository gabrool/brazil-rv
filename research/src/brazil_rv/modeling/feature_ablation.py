from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from brazil_rv.preprocessing.contract import (
    GLOBAL_SLOW_CHANNELS,
    GLOBAL_UNUSED_SLOW_CHANNEL_INDICES,
    SLOW_CHANNELS,
)

from .contract import FEATURE_CONTRACT_VERSION


@dataclass(frozen=True)
class FeatureAblation:
    key: str
    removed_slow_features: tuple[str, ...]
    description: str

    def specification(self) -> dict[str, object]:
        return {
            "key": self.key,
            "description": self.description,
            "removed_slow_features": list(self.removed_slow_features),
        }

    def serialized_specification(self) -> str:
        return json.dumps(self.specification(), sort_keys=True, separators=(",", ":"))

    def specification_sha256(self) -> str:
        return hashlib.sha256(self.serialized_specification().encode()).hexdigest()


@dataclass(frozen=True)
class ResolvedSlowPosition:
    index: int
    equity_local_name: str
    global_name: str
    global_structurally_unused: bool

    def metadata(self) -> dict[str, object]:
        return {
            "index": self.index,
            "equity_local_name": self.equity_local_name,
            "global_name": self.global_name,
            "global_structurally_unused": self.global_structurally_unused,
        }


def _serialized(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _axis_sha256(names: tuple[str, ...]) -> str:
    return _sha256_text(_serialized(list(names)))


@dataclass(frozen=True)
class ResolvedFeatureAblation:
    specification: FeatureAblation
    slow_indices: tuple[int, ...]
    position_mapping: tuple[ResolvedSlowPosition, ...]
    equity_local_axis_sha256: str
    global_axis_sha256: str

    @property
    def key(self) -> str:
        return self.specification.key

    def metadata(self) -> dict[str, object]:
        mapping = [position.metadata() for position in self.position_mapping]
        mapping_serialized = _serialized(mapping)
        mapping_sha256 = _sha256_text(mapping_serialized)
        resolved_identity = {
            "specification_sha256": self.specification.specification_sha256(),
            "equity_local_axis_sha256": self.equity_local_axis_sha256,
            "global_axis_sha256": self.global_axis_sha256,
            "resolved_position_mapping_sha256": mapping_sha256,
        }
        return {
            **self.specification.specification(),
            "mask_semantics": "shared_slow_position_mask",
            "shared_position_count": len(mapping),
            "resolved_position_mapping": mapping,
            "serialized_resolved_position_mapping": mapping_serialized,
            "resolved_position_mapping_sha256": mapping_sha256,
            "equity_local_axis_sha256": self.equity_local_axis_sha256,
            "global_axis_sha256": self.global_axis_sha256,
            "serialized_specification": self.specification.serialized_specification(),
            "specification_sha256": self.specification.specification_sha256(),
            "resolved_identity_sha256": _sha256_text(_serialized(resolved_identity)),
        }


_DROP_SLOW_LOW_PRIOR = (
    "vol_regime",
    "realized_vol_20d_log_ratio",
    "vol_of_vol_20d",
    "median_daily_real_volume_20d_log_scale",
    "daily_dollar_volume_regime_20d",
    "observed_fraction_5d",
    "observed_fraction_20d",
    "weekday_sin",
    "weekday_cos",
    "month_end_proximity",
    "quarter_end_proximity",
)

FEATURE_ABLATIONS = MappingProxyType(
    {
        "none": FeatureAblation("none", (), "Use every normally available feature."),
        "drop_slow_low_prior": FeatureAblation(
            "drop_slow_low_prior",
            _DROP_SLOW_LOW_PRIOR,
            "Jointly zero the 11 registered low-prior slow-feature positions.",
        ),
    }
)
FEATURE_ABLATION_KEYS = tuple(FEATURE_ABLATIONS)


def _validate_registry() -> None:
    if FEATURE_ABLATION_KEYS != ("none", "drop_slow_low_prior"):
        raise ValueError("Feature-ablation registry keys do not match the contract")
    for key, specification in FEATURE_ABLATIONS.items():
        names = specification.removed_slow_features
        if specification.key != key or not specification.description:
            raise ValueError(f"Invalid feature-ablation identity: {key}")
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicated slow feature in ablation: {key}")
        if not set(names) <= set(SLOW_CHANNELS):
            raise ValueError(f"Unknown slow feature in ablation: {key}")
    if len(_DROP_SLOW_LOW_PRIOR) != 11:
        raise ValueError("drop_slow_low_prior must remove exactly 11 channels")


_validate_registry()


def get_feature_ablation(key: str) -> FeatureAblation:
    try:
        return FEATURE_ABLATIONS[key]
    except KeyError as error:
        raise ValueError(f"Unknown feature ablation: {key}") from error


def resolve_feature_ablation(
    specification: FeatureAblation,
    *,
    slow_features: tuple[str, ...],
    global_slow_features: tuple[str, ...] = GLOBAL_SLOW_CHANNELS,
    global_structurally_unused_indices: tuple[
        int, ...
    ] = GLOBAL_UNUSED_SLOW_CHANNEL_INDICES,
) -> ResolvedFeatureAblation:
    if len(slow_features) != len(set(slow_features)):
        raise ValueError("Feature-store slow-feature names contain duplicates")
    if slow_features != SLOW_CHANNELS:
        raise ValueError("Feature-store slow-feature axis is not canonical")
    if len(global_slow_features) != len(set(global_slow_features)):
        raise ValueError("Feature-store global slow-feature names contain duplicates")
    if global_slow_features != GLOBAL_SLOW_CHANNELS:
        raise ValueError("Feature-store global slow-feature axis is not canonical")
    if (
        len(global_structurally_unused_indices)
        != len(set(global_structurally_unused_indices))
        or global_structurally_unused_indices != GLOBAL_UNUSED_SLOW_CHANNEL_INDICES
    ):
        raise ValueError(
            "Feature-store global structural slow positions are not canonical"
        )
    indices = tuple(
        slow_features.index(name) for name in specification.removed_slow_features
    )
    unused = set(global_structurally_unused_indices)
    return ResolvedFeatureAblation(
        specification,
        indices,
        tuple(
            ResolvedSlowPosition(
                index=index,
                equity_local_name=slow_features[index],
                global_name=global_slow_features[index],
                global_structurally_unused=index in unused,
            )
            for index in indices
        ),
        _axis_sha256(slow_features),
        _axis_sha256(global_slow_features),
    )


def _schema_axis(schema: dict[str, object], field: str) -> tuple[str, ...]:
    rows = schema.get(field)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Feature schema is missing {field} metadata")
    indices = tuple(row.get("index") for row in rows)
    if indices != tuple(range(len(rows))):
        raise ValueError(f"Feature schema {field} indices are not contiguous")
    names = tuple(row.get("name") for row in rows)
    if any(not isinstance(name, str) for name in names):
        raise ValueError(f"Feature schema {field} names are invalid")
    return names


def _schema_indices(schema: dict[str, object], field: str) -> tuple[int, ...]:
    values = schema.get(field)
    if not isinstance(values, list) or any(
        not isinstance(value, int) or isinstance(value, bool) for value in values
    ):
        raise ValueError(f"Feature schema is missing {field} metadata")
    return tuple(values)


def resolve_feature_ablation_for_store(
    store: Path, key: str
) -> ResolvedFeatureAblation:
    schema = json.loads((store / "feature_schema.json").read_text(encoding="utf-8"))
    if schema.get("contract_version") != FEATURE_CONTRACT_VERSION:
        raise ValueError("Feature schema has the wrong feature-ablation contract")
    slow_features = _schema_axis(schema, "slow_channels")
    global_slow_features = _schema_axis(schema, "global_slow_channels")
    global_structurally_unused = _schema_indices(schema, "global_slow")
    for registered in FEATURE_ABLATIONS.values():
        resolve_feature_ablation(
            registered,
            slow_features=slow_features,
            global_slow_features=global_slow_features,
            global_structurally_unused_indices=global_structurally_unused,
        )
    return resolve_feature_ablation(
        get_feature_ablation(key),
        slow_features=slow_features,
        global_slow_features=global_slow_features,
        global_structurally_unused_indices=global_structurally_unused,
    )


def apply_feature_ablation_to_slow_features(
    values: np.ndarray, ablation: ResolvedFeatureAblation
) -> np.ndarray:
    """Apply the registered position mask after context masking at model ingress."""
    if values.shape[-1] != len(SLOW_CHANNELS):
        raise ValueError("Slow-feature input has the wrong trailing axis")
    if not ablation.slow_indices:
        return values
    output = values.copy()
    output[..., ablation.slow_indices] = 0.0
    return output


NO_FEATURE_ABLATION = resolve_feature_ablation(
    FEATURE_ABLATIONS["none"], slow_features=SLOW_CHANNELS
)
