from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from brazil_rv.v2.feature_spec import FeatureSpec, feature_schema_sha256
from brazil_rv.v2.store import write_store


def _inferred_feature_names(
    arrays: Mapping[str, NDArray[np.generic]],
) -> dict[str, tuple[str, ...]]:
    output: dict[str, tuple[str, ...]] = {}
    for family, array_name in (
        ("slow", "slow_values"),
        ("intraday", "intraday_values"),
        ("native_fast", "fast_patch_values"),
    ):
        value = arrays.get(array_name)
        if value is not None:
            output[family] = tuple(
                f"{family}_fixture_{index}" for index in range(value.shape[-1])
            )
    for array_name, value in sorted(arrays.items()):
        if not array_name.startswith("sidecar_") or not array_name.endswith("_values"):
            continue
        family = array_name.removesuffix("_values")
        output[family] = tuple(
            f"{family}_fixture_{index}" for index in range(value.shape[-1])
        )
    return output


def fixture_feature_schema(
    feature_names: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    specs = tuple(
        FeatureSpec(
            name=str(name),
            family=family,
            transform="signed_identity",
            source_units="synthetic_test_units",
            availability_rule="synthetic fixture available at the decision time",
            formula="synthetic fixture value",
            minimum_support=1,
            validity_rule="synthetic fixture validity mask",
            age_staleness_policy="synthetic fixture age",
            version="synthetic_fixture_1",
        )
        for family in (
            *(
                key
                for key in ("slow", "intraday", "native_fast")
                if key in feature_names
            ),
            *sorted(key for key in feature_names if key.startswith("sidecar_")),
        )
        for name in feature_names[family]
    )
    return {
        "specifications": [asdict(spec) for spec in specs],
        "sha256": feature_schema_sha256(specs),
    }


def write_fixture_store(
    output_dir: Path,
    *,
    dates: Sequence[object],
    isins: Sequence[str],
    arrays: Mapping[str, NDArray[np.generic]],
    feature_names: Mapping[str, Sequence[str]] | None = None,
    metadata: Mapping[str, object] | None = None,
    **kwargs: Any,
) -> Path:
    names = dict(feature_names or _inferred_feature_names(arrays))
    metadata_payload = dict(metadata or {})
    metadata_payload.setdefault("action_terms_source", "verified_contractual_terms")
    metadata_payload.setdefault("schedule_source", "explicit_versioned_schedule")
    metadata_payload.setdefault(
        "corporate_action_contract",
        {"stored_action_arrays": "retrospective outcome/accounting terms"},
    )
    metadata_payload["feature_schema"] = fixture_feature_schema(names)
    return write_store(
        output_dir,
        dates=dates,
        isins=isins,
        arrays=arrays,
        feature_names=names,
        metadata=metadata_payload,
        **kwargs,
    )
