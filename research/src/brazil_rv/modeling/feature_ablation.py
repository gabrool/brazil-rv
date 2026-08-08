from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from brazil_rv.preprocessing.contract import GLOBAL_SLOW_CHANNELS, SLOW_CHANNELS

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
class ResolvedFeatureAblation:
    specification: FeatureAblation
    slow_indices: tuple[int, ...]

    @property
    def key(self) -> str:
        return self.specification.key

    def metadata(self) -> dict[str, object]:
        return {
            **self.specification.specification(),
            "resolved_slow_features": [
                {"name": name, "index": index}
                for name, index in zip(
                    self.specification.removed_slow_features,
                    self.slow_indices,
                    strict=True,
                )
            ],
            "serialized_specification": self.specification.serialized_specification(),
            "specification_sha256": self.specification.specification_sha256(),
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
) -> ResolvedFeatureAblation:
    if len(slow_features) != len(set(slow_features)):
        raise ValueError("Feature-store slow-feature names contain duplicates")
    if slow_features != SLOW_CHANNELS:
        raise ValueError("Feature-store slow-feature axis is not canonical")
    return ResolvedFeatureAblation(
        specification,
        tuple(
            slow_features.index(name) for name in specification.removed_slow_features
        ),
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


def resolve_feature_ablation_for_store(
    store: Path, key: str
) -> ResolvedFeatureAblation:
    schema = json.loads((store / "feature_schema.json").read_text(encoding="utf-8"))
    if schema.get("contract_version") != FEATURE_CONTRACT_VERSION:
        raise ValueError("Feature schema has the wrong feature-ablation contract")
    slow_features = _schema_axis(schema, "slow_channels")
    global_slow_features = _schema_axis(schema, "global_slow_channels")
    if global_slow_features != GLOBAL_SLOW_CHANNELS:
        raise ValueError("Feature-store global slow-feature axis is not canonical")
    for registered in FEATURE_ABLATIONS.values():
        resolve_feature_ablation(registered, slow_features=slow_features)
    return resolve_feature_ablation(
        get_feature_ablation(key), slow_features=slow_features
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
