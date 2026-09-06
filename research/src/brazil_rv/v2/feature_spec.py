from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from .normalization import rank_gauss

Transform = Literal[
    "rank_gauss",
    "binary",
    "bounded_fraction",
    "signed_identity",
    "age_sessions",
    "signed_clip",
    "annual_rate",
]


@dataclass(frozen=True)
class FeatureSpec:
    """One canonical model-feature definition and its fixed transform."""

    name: str
    family: str
    transform: Transform
    source_units: str
    clip: float | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.family or not self.source_units:
            raise ValueError("feature name, family, and source units are required")
        if self.transform == "signed_clip":
            if self.clip is None or not np.isfinite(self.clip) or self.clip <= 0:
                raise ValueError("signed_clip requires a finite positive clip")
        elif self.clip is not None:
            raise ValueError("clip is only valid for signed_clip")


_BOUNDED_FRACTIONS = {
    "close_location_value",
    "last_30_minute_return_share_lag1",
    "last_hour_volume_share_lag1",
    "oddlot_volume_share",
}
_AGES = {
    "observed_history_age_sessions",
    "sessions_since_financial_filing",
    "sessions_since_earnings",
    "sessions_until_announced_earnings",
}
_BINARY = {
    "observed_history_left_censored",
    "preview_add",
    "preview_delete",
}
_SIGNED = {
    "preview_delta_signed_sqrt",
    "preview_pressure",
    "pre_effective_ramp",
    "post_effective_reversal",
}
_SIGNED_CLIPPED = {
    "realized_skew_60",
    "realized_kurtosis_60",
    "standardized_unexpected_earnings",
    "put_skew",
    "oddlot_volume_share_change_5",
}
_ANNUAL_RATES = {"loan_rate", "loan_rate_change_5"}


def feature_specs(
    family: str, names: Sequence[str]
) -> tuple[FeatureSpec, ...]:
    """Return explicit transforms for an ordered enabled feature family."""

    output: list[FeatureSpec] = []
    for raw_name in names:
        name = str(raw_name)
        suffix = name.split("_", 1)[1] if family == "sidecar_rebalance" else name
        if suffix in _BINARY:
            transform: Transform = "binary"
            units = "flag"
            clip = None
        elif suffix in _BOUNDED_FRACTIONS:
            transform = "bounded_fraction"
            units = "fraction_0_1"
            clip = None
        elif suffix in _AGES:
            transform = "age_sessions"
            units = "sessions"
            clip = None
        elif suffix in _SIGNED:
            transform = "signed_identity"
            units = "signed_unitless"
            clip = None
        elif suffix in _SIGNED_CLIPPED:
            transform = "signed_clip"
            units = "standardized_unitless"
            clip = 5.0
        elif suffix in _ANNUAL_RATES:
            transform = "annual_rate"
            units = "annual_decimal"
            clip = None
        else:
            transform = "rank_gauss"
            units = "declared_raw_economic_unit"
            clip = None
        output.append(
            FeatureSpec(
                name=name,
                family=family,
                transform=transform,
                source_units=units,
                clip=clip,
            )
        )
    return tuple(output)


def feature_schema_sha256(specs: Sequence[FeatureSpec]) -> str:
    payload = json.dumps(
        [asdict(spec) for spec in specs],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def transform_feature_panel_into(
    values: NDArray[np.floating],
    valid: NDArray[np.bool_],
    active: NDArray[np.bool_],
    specs: Sequence[FeatureSpec],
    output: NDArray[np.float32],
    output_valid: NDArray[np.bool_],
    *,
    source_rows: NDArray[np.integer] | None = None,
    minimum_rank_names: int = 20,
) -> None:
    """Apply typed transforms row-by-row into preallocated float32 arrays."""

    raw = np.asarray(values)
    mask = np.asarray(valid, dtype=np.bool_)
    membership = np.asarray(active, dtype=np.bool_)
    destination = np.asarray(output)
    destination_valid = np.asarray(output_valid)
    if raw.ndim != 3 or raw.shape != mask.shape:
        raise ValueError("feature values/masks must align [date, name, feature]")
    if membership.shape != raw.shape[:2] or len(specs) != raw.shape[2]:
        raise ValueError("feature specs or active membership are misaligned")
    if destination.dtype != np.float32 or destination_valid.dtype != np.bool_:
        raise TypeError("feature destinations must be float32 and bool")
    if minimum_rank_names < 1:
        raise ValueError("minimum_rank_names must be positive")
    rows = (
        np.arange(raw.shape[0], dtype=np.int64)
        if source_rows is None
        else np.asarray(source_rows, dtype=np.int64)
    )
    if destination.shape != (rows.size, *raw.shape[1:]) or (
        destination_valid.shape != destination.shape
    ):
        raise ValueError("feature destinations are misaligned")
    for output_row, source_row in enumerate(rows):
        if source_row < 0 or source_row >= raw.shape[0]:
            raise ValueError("source_rows contains an out-of-range index")
        for feature, spec in enumerate(specs):
            cross = np.asarray(raw[source_row, :, feature], dtype=np.float64)
            usable = (
                mask[source_row, :, feature]
                & membership[source_row]
                & np.isfinite(cross)
            )
            transformed = np.zeros(cross.shape, dtype=np.float64)
            if spec.transform == "rank_gauss":
                if int(usable.sum()) >= minimum_rank_names:
                    ranked, usable = rank_gauss(cross, usable, membership[source_row])
                    transformed = ranked.astype(np.float64, copy=False)
                else:
                    usable[:] = False
            elif spec.transform == "binary":
                binary = usable & ((cross == 0.0) | (cross == 1.0))
                usable &= binary
                transformed[usable] = cross[usable]
            elif spec.transform == "bounded_fraction":
                bounded = usable & (cross >= 0.0) & (cross <= 1.0)
                usable &= bounded
                transformed[usable] = 2.0 * cross[usable] - 1.0
            elif spec.transform == "signed_identity":
                bounded = usable & (cross >= -1.0) & (cross <= 1.0)
                usable &= bounded
                transformed[usable] = cross[usable]
            elif spec.transform == "age_sessions":
                usable &= cross >= 0.0
                transformed[usable] = np.log1p(
                    np.minimum(cross[usable], 252.0)
                ) / np.log1p(252.0)
            elif spec.transform == "signed_clip":
                assert spec.clip is not None
                transformed[usable] = np.clip(
                    cross[usable], -spec.clip, spec.clip
                )
            elif spec.transform == "annual_rate":
                transformed[usable] = np.clip(
                    np.arcsinh(cross[usable] / 0.01), -5.0, 5.0
                )
            else:  # pragma: no cover - Literal plus FeatureSpec validation
                raise AssertionError(spec.transform)
            destination[output_row, :, feature] = 0.0
            destination[output_row, usable, feature] = transformed[usable].astype(
                np.float32
            )
            destination_valid[output_row, :, feature] = usable
