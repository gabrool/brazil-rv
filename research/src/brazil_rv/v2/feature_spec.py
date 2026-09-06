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
    "precomputed_native",
]


@dataclass(frozen=True)
class FeatureSpec:
    """Immutable semantic contract for one ordered model feature."""

    name: str
    family: str
    transform: Transform
    source_units: str
    clip: float | None = None
    availability_rule: str = "available_at_decision_time"
    formula: str = "declared_by_feature_name"
    minimum_support: int = 1
    validity_rule: str = "source_valid & active & finite"
    age_staleness_policy: str = "no_forward_fill_as_observed"
    version: str = "1"

    def __post_init__(self) -> None:
        required_text = (
            self.name,
            self.family,
            self.source_units,
            self.availability_rule,
            self.formula,
            self.validity_rule,
            self.age_staleness_policy,
            self.version,
        )
        if any(not value for value in required_text):
            raise ValueError("all feature semantic fields must be non-empty")
        if (
            isinstance(self.minimum_support, bool)
            or not isinstance(self.minimum_support, (int, np.integer))
            or self.minimum_support < 1
        ):
            raise ValueError("minimum_support must be a positive integer")
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


def _family_availability_rule(family: str) -> str:
    if family == "slow":
        return "causal daily history available before the decision session"
    if family == "intraday":
        return "completed bars before the decision minute; decision row excluded"
    if family.startswith("sidecar_"):
        return "publication-lagged sidecar snapshot available by the decision session"
    return "available_at_decision_time"


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
                availability_rule=_family_availability_rule(family),
                formula=f"canonical_{family}_producer:{name}",
                minimum_support=20 if transform == "rank_gauss" else 1,
                validity_rule=(
                    "source_valid & active & finite; rank_gauss additionally "
                    "requires 20 valid active names"
                    if transform == "rank_gauss"
                    else "source_valid & active & finite"
                ),
                age_staleness_policy=(
                    "clip at 252 sessions then log1p-scale"
                    if transform == "age_sessions"
                    else "family producer's causal mask; no forward-fill as observed"
                ),
            )
        )
    return tuple(output)


def native_fast_feature_specs() -> tuple[FeatureSpec, ...]:
    """Return the ordered, already-computed seven-channel fast contract."""

    from .intraday_features import NATIVE_FAST_FEATURES

    completed = "completed five-minute blocks strictly before the decision minute"
    definitions = {
        "adjacent_endpoint_log_return_over_s5": (
            "standardized_log_return",
            "log(C_p/C_{p-1}) / (sigma_asof * sqrt(5/continuous_minutes))",
            2,
            "current and previous block endpoints valid; sigma_asof positive",
            "no carry-forward across blocks or sessions",
        ),
        "block_log_high_low_over_s5": (
            "standardized_log_range",
            "log(H_p/L_p) / (sigma_asof * sqrt(5/continuous_minutes))",
            5,
            "all five price bars valid; H_p >= L_p > 0; sigma_asof positive",
            "no carry-forward across blocks or sessions",
        ),
        "signed_close_location": (
            "signed_fraction_minus1_1",
            "2 * (C_p-L_p)/(H_p-L_p) - 1; zero when H_p == L_p",
            5,
            "all five price bars valid and C_p within [L_p,H_p]",
            "no carry-forward across blocks or sessions",
        ),
        "relative_same_clock_volume": (
            "signed_log_ratio_clipped_5",
            "clip(log1p(V_p/median(V_past20_same_clock))-log(2),-5,5)",
            16,
            "complete nonnegative block volume and >=16 valid prior same-clock blocks",
            "median uses only the prior 20 sessions at the same clock minute",
        ),
        "observed_fraction": (
            "fraction_0_1",
            "count(observed bars in block) / 5",
            1,
            "session/name is supported at the decision date",
            "missing bars remain unobserved; no imputation",
        ),
        "elapsed_fraction": (
            "fraction_0_1",
            "completed_prefix_minutes / decision_prefix_minutes",
            1,
            "session/name is supported at the decision date",
            "derived from the dated session schedule",
        ),
        "last_price_age_fraction": (
            "fraction_0_1",
            "clip((block_end-last_observed_price_minute)/continuous_minutes,0,1)",
            1,
            "at least one observed positive close in the session prefix",
            "age resets each session and never carries a prior-session price",
        ),
    }
    if tuple(definitions) != tuple(NATIVE_FAST_FEATURES):
        raise AssertionError("native fast semantic registry is out of order")
    return tuple(
        FeatureSpec(
            name=name,
            family="native_fast",
            transform="precomputed_native",
            source_units=definitions[name][0],
            availability_rule=completed,
            formula=definitions[name][1],
            minimum_support=definitions[name][2],
            validity_rule=definitions[name][3],
            age_staleness_policy=definitions[name][4],
            version="native_fast_1",
        )
        for name in NATIVE_FAST_FEATURES
    )


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
    rank_specs = tuple(spec for spec in specs if spec.transform == "rank_gauss")
    if any(spec.minimum_support != minimum_rank_names for spec in rank_specs):
        raise ValueError(
            "minimum_rank_names must match every rank-gauss feature specification"
        )
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
            elif spec.transform == "precomputed_native":
                transformed[usable] = cross[usable]
            else:  # pragma: no cover - Literal plus FeatureSpec validation
                raise AssertionError(spec.transform)
            destination[output_row, :, feature] = 0.0
            destination[output_row, usable, feature] = transformed[usable].astype(
                np.float32
            )
            destination_valid[output_row, :, feature] = usable
