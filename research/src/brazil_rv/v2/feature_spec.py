from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from .contract import INTRADAY_PRIOR_SESSION_FEATURES
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


# These descriptions are part of the store identity, not documentation aliases.
# Keep them next to the transform registry so changing a producer's economic
# definition necessarily changes the FeatureSpec hash sealed by the store.
_SLOW_FORMULAS: dict[str, tuple[str, str]] = {
    **{
        f"log_return_{horizon}": (
            "decimal_log_return",
            f"exact log shareholder-wealth close return over {horizon} sessions",
        )
        for horizon in (1, 5, 21, 63, 126, 252)
    },
    "momentum_12_1": (
        "decimal_log_return",
        "exact 252-session log return minus exact 21-session log return",
    ),
    **{
        f"yang_zhang_vol_{window}": (
            "daily_decimal_volatility",
            f"Yang-Zhang volatility over the prior {window} completed sessions",
        )
        for window in (5, 20, 60)
    },
    "vol_of_vol_60": (
        "daily_decimal_volatility",
        "sample standard deviation of Yang-Zhang-5 over 60 completed sessions",
    ),
    "realized_skew_60": (
        "standardized_unitless",
        "sample skewness of one-session log returns over 60 completed sessions",
    ),
    "realized_kurtosis_60": (
        "standardized_unitless",
        "excess kurtosis of one-session log returns over 60 completed sessions",
    ),
    "max_return_21": (
        "decimal_log_return",
        "maximum one-session log return over 21 completed sessions",
    ),
    "distance_52_week_high": (
        "decimal_log_return",
        "log(close / maximum high over 252 completed sessions)",
    ),
    "beta_60": (
        "unitless_slope",
        "60-session covariance with the decision-universe market return divided by market variance",
    ),
    "idiosyncratic_vol_60": (
        "daily_decimal_volatility",
        "60-session sample volatility of residuals from the market beta regression",
    ),
    "log_volume_mean_20": (
        "log_brl",
        "log(mean daily BRL turnover over the inclusive 20 completed-session window; complete-source no-trade sessions are exact zero and incomplete sessions invalidate the window)",
    ),
    "volume_zscore_20": (
        "standardized_unitless",
        "current completed-session BRL turnover minus the inclusive 20-session mean, divided by that window's population deviation",
    ),
    "amihud_20": (
        "absolute_log_return_per_brl",
        "mean(abs(one-session log return) / BRL turnover) over an exact 20 completed-session window with positive turnover in every session",
    ),
    "trade_count_zscore_20": (
        "standardized_unitless",
        "current completed-session trade count minus the inclusive 20-session mean, divided by that window's population deviation",
    ),
    "turnover_proxy_20": (
        "ratio",
        "current completed-session BRL turnover divided by the inclusive 20 completed-session mean",
    ),
    "high_low_range_1": (
        "decimal_log_range",
        "log(completed-session high / completed-session low)",
    ),
    "high_low_range_5": (
        "decimal_log_range",
        "log(maximum high / minimum low) over five completed sessions",
    ),
    "close_location_value": (
        "fraction_0_1",
        "(close-low)/(high-low), with 0.5 for a valid zero-range session",
    ),
    "observed_history_age_sessions": (
        "sessions",
        "exchange-session distance from the first observed canonical daily row",
    ),
    "observed_history_left_censored": (
        "flag",
        "one when observed history begins at the left boundary of the canonical archive",
    ),
    "cluster_mean_return_5": (
        "decimal_log_return",
        "leave-one-out mean exact five-session return of prior-only correlation-cluster peers",
    ),
    "cluster_mean_return_21": (
        "decimal_log_return",
        "leave-one-out mean exact 21-session return of prior-only correlation-cluster peers",
    ),
    "name_minus_cluster_return_5": (
        "decimal_log_return",
        "name exact five-session return minus its leave-one-out cluster mean",
    ),
    "name_minus_cluster_return_21": (
        "decimal_log_return",
        "name exact 21-session return minus its leave-one-out cluster mean",
    ),
    "cluster_dispersion": (
        "daily_decimal_return_dispersion",
        "cross-sectional sample deviation of exact 21-session returns among leave-one-out cluster peers",
    ),
}

_INTRADAY_FORMULAS: dict[str, tuple[str, str]] = {
    "overnight_return": (
        "decimal_log_return",
        "log(M1 session open / exact adjacent M1 session close); prior return consistency and decision-known boundaries",
    ),
    "intraday_return_1545": (
        "decimal_log_return",
        "log(last completed-prefix price / session open)",
    ),
    "overnight_return_sum_5": (
        "decimal_log_return",
        "sum of observed overnight_return in five sessions ending at the decision; minimum 4/5; completed-session return validation",
    ),
    "overnight_return_sum_20": (
        "decimal_log_return",
        "sum of observed overnight_return in 20 sessions ending at the decision; minimum 16/20; completed-session return validation",
    ),
    "intraday_return_sum_5": (
        "decimal_log_return",
        "sum of observed intraday_return_1545 in five sessions ending at the decision; minimum 4/5",
    ),
    "intraday_return_sum_20": (
        "decimal_log_return",
        "sum of observed intraday_return_1545 in 20 sessions ending at the decision; minimum 16/20",
    ),
    "overnight_minus_intraday": (
        "decimal_log_return",
        "overnight_return minus intraday_return_1545",
    ),
    "overnight_minus_intraday_mean_20": (
        "decimal_log_return",
        "observed-sample mean overnight-minus-intraday differential in 20 sessions; minimum 16/20; completed-session return validation",
    ),
    "last_30_minute_return_share_lag1": (
        "signed_unbounded_ratio",
        "prior validated session M1 last-30-minute log return divided by M1 full-session log return",
    ),
    "last_hour_volume_share_lag1": (
        "fraction_0_1",
        "prior session last-hour BRL turnover divided by full-session BRL turnover",
    ),
    "close_vwap_deviation_lag1": (
        "decimal_log_return",
        "prior validated session log(M1 final close / M1 full-session VWAP)",
    ),
    "vwap_deviation_1545": (
        "decimal_log_return",
        "log(last completed-prefix price / completed-prefix VWAP)",
    ),
    "realized_vol_5m_1": (
        "decimal_log_return_volatility",
        "root-sum-square adjacent five-minute endpoint returns in the current completed prefix",
    ),
    "realized_vol_5m_5": (
        "decimal_log_return_volatility",
        "observed-sample mean of realized_vol_5m_1 in five sessions; minimum 4/5",
    ),
    "realized_vol_5m_20": (
        "decimal_log_return_volatility",
        "observed-sample mean of realized_vol_5m_1 in 20 sessions; minimum 16/20",
    ),
    "realized_skew_5m_20": (
        "standardized_unitless",
        "skewness of completed five-minute endpoint returns over 20 exact sessions",
    ),
    "roll_spread_20": (
        "decimal_return_spread",
        "Roll spread from adjacent completed five-minute returns over 20 exact sessions",
    ),
    "corwin_schultz_spread_20": (
        "decimal_return_spread",
        "observed-sample mean of causal Corwin-Schultz prefix high-low estimates in 20 sessions; minimum 16/20",
    ),
    "intraday_range_1545": (
        "decimal_log_range",
        "log(completed-prefix high / completed-prefix low)",
    ),
    "volume_1545_relative_median_20": (
        "ratio",
        "M1 completed-prefix turnover divided by median of at least 16/20 prior activity-complete, return-validated sessions at the same clock",
    ),
}

_SIDECAR_FORMULAS: dict[str, tuple[str, str]] = {
    "loan_balance_to_volume_20": (
        "ratio_sessions",
        "lending balance BRL divided by mean canonical BRL turnover over its exact source session and prior 19 sessions",
    ),
    "loan_balance_change_1": (
        "ratio_sessions",
        "one-exchange-session change in loan_balance_to_volume_20 at the vintage known by the decision",
    ),
    "loan_balance_change_5": (
        "ratio_sessions",
        "five-exchange-session change in loan_balance_to_volume_20 at the vintage known by the decision",
    ),
    "loan_rate": ("annual_decimal", "raw annual-decimal lending rate"),
    "loan_rate_change_5": (
        "annual_decimal",
        "five-exchange-session change in raw annual-decimal lending rate at the known vintage",
    ),
    "sessions_since_financial_filing": (
        "sessions",
        "exchange sessions since the latest timestamped ITR/DFP/financial-filing receipt available at the decision",
    ),
    "standardized_unexpected_earnings": (
        "standardized_unitless",
        "declared point-in-time standardized unexpected earnings from a true earnings-release source",
    ),
    "put_call_log_oi_ratio": (
        "log_ratio",
        "log((put open interest + 1 contract)/(call open interest + 1 contract)) from a complete snapshot",
    ),
    "delta_oi_to_volume_1": (
        "ratio",
        "one-source-session change in total option open interest divided by canonical stock ADV20",
    ),
    "atm_iv_to_median_20": (
        "ratio",
        "at-the-money implied volatility divided by its prior-20-source-session median",
    ),
    "put_skew": (
        "standardized_unitless",
        "declared put-versus-call implied-volatility skew from a complete option surface",
    ),
    "oddlot_volume_share": (
        "fraction_0_1",
        "odd-lot BRL turnover divided by odd-lot plus regular-lot BRL turnover",
    ),
    "oddlot_volume_share_change_5": (
        "fraction_change",
        "exact five-exchange-session change in oddlot_volume_share at the known vintage",
    ),
    "log_market_cap": ("log_brl", "log point-in-time market capitalization in BRL"),
    "book_to_market": (
        "ratio",
        "point-in-time book equity divided by market capitalization",
    ),
    "gross_profitability": (
        "ratio",
        "point-in-time gross profit divided by total assets",
    ),
    "liabilities_to_assets": (
        "ratio",
        "liabilities divided by assets from one coherent publicly available filing row",
    ),
}

_REBALANCE_FORMULAS: dict[str, tuple[str, str]] = {
    "current_weight_sqrt": (
        "sqrt_fraction",
        "sqrt(max(current published index weight, 0))",
    ),
    "preview_delta_signed_sqrt": (
        "signed_sqrt_fraction",
        "sign(preview-current) * sqrt(abs(preview-current)) at the known preview vintage",
    ),
    "preview_add": ("flag", "one for a known preview addition, else zero"),
    "preview_delete": ("flag", "one for a known preview deletion, else zero"),
    "preview_pressure": (
        "signed_unitless",
        "known preview weight delta divided by causal ADV shares and clipped to [-1,1]",
    ),
    "pre_effective_ramp": (
        "signed_unitless",
        "signed-sqrt preview delta times exp(-sessions_to_effective/5)",
    ),
    "post_effective_reversal": (
        "signed_unitless",
        "negative signed-sqrt effective delta times exp(-sessions_since_effective/10) through session 20",
    ),
}


def _semantic_definition(family: str, name: str) -> tuple[str, str]:
    if family == "slow":
        definitions = _SLOW_FORMULAS
        key = name
    elif family == "intraday":
        definitions = _INTRADAY_FORMULAS
        key = name
    elif family == "sidecar_rebalance":
        definitions = _REBALANCE_FORMULAS
        key = name.split("_", 1)[1]
    elif family.startswith("sidecar_"):
        definitions = _SIDECAR_FORMULAS
        key = name
    else:
        raise ValueError(f"unregistered feature family: {family}")
    try:
        return definitions[key]
    except KeyError as error:
        raise ValueError(f"unregistered semantic feature: {family}/{name}") from error


def _family_availability_rule(family: str) -> str:
    if family == "slow":
        return "causal daily history available before the decision session"
    if family == "intraday":
        return "completed bars before the decision minute; decision row excluded"
    if family.startswith("sidecar_"):
        return "publication-lagged sidecar snapshot available by the decision session"
    return "available_at_decision_time"


def feature_specs(
    family: str,
    names: Sequence[str],
    *,
    minimum_rank_names: int = 20,
) -> tuple[FeatureSpec, ...]:
    """Return explicit transforms for an ordered enabled feature family."""

    if minimum_rank_names < 1:
        raise ValueError("minimum_rank_names must be positive")

    output: list[FeatureSpec] = []
    for raw_name in names:
        name = str(raw_name)
        units, formula = _semantic_definition(family, name)
        suffix = name.split("_", 1)[1] if family == "sidecar_rebalance" else name
        if suffix in _BINARY:
            transform: Transform = "binary"
            clip = None
        elif suffix in _BOUNDED_FRACTIONS:
            transform = "bounded_fraction"
            clip = None
        elif suffix in _AGES:
            transform = "age_sessions"
            clip = None
        elif suffix in _SIGNED:
            transform = "signed_identity"
            clip = None
        elif suffix in _SIGNED_CLIPPED:
            transform = "signed_clip"
            clip = 5.0
        elif suffix in _ANNUAL_RATES:
            transform = "annual_rate"
            clip = None
        else:
            transform = "rank_gauss"
            clip = None
        output.append(
            FeatureSpec(
                name=name,
                family=family,
                transform=transform,
                source_units=units,
                clip=clip,
                availability_rule=(
                    "prior full-session observation available by decision t"
                    if family == "intraday" and name in INTRADAY_PRIOR_SESSION_FEATURES
                    else _family_availability_rule(family)
                ),
                formula=formula,
                minimum_support=(
                    minimum_rank_names if transform == "rank_gauss" else 1
                ),
                validity_rule=(
                    "source_valid & active & finite; rank_gauss additionally "
                    f"requires {minimum_rank_names} valid active names"
                    if transform == "rank_gauss"
                    else "source_valid & active & finite"
                ),
                age_staleness_policy=(
                    "clip at 252 sessions then log1p-scale"
                    if transform == "age_sessions"
                    else (
                        "source age is at least one exchange session"
                        if family == "intraday"
                        and name in INTRADAY_PRIOR_SESSION_FEATURES
                        else "family producer's causal mask; no forward-fill as observed"
                    )
                ),
                version="decision_feature_2",
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
    membership_rows: NDArray[np.integer] | None = None,
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
    membership_indices = (
        rows if membership_rows is None else np.asarray(membership_rows, dtype=np.int64)
    )
    if membership_indices.shape != rows.shape:
        raise ValueError("membership_rows must align with source_rows")
    if destination.shape != (rows.size, *raw.shape[1:]) or (
        destination_valid.shape != destination.shape
    ):
        raise ValueError("feature destinations are misaligned")
    for output_row, (source_row, membership_row) in enumerate(
        zip(rows, membership_indices, strict=True)
    ):
        if membership_row < 0 or membership_row >= raw.shape[0]:
            raise ValueError("membership_rows contains an out-of-range index")
        if source_row < -1 or source_row >= raw.shape[0]:
            raise ValueError("source_rows contains an out-of-range index")
        if source_row == -1:
            destination[output_row] = 0.0
            destination_valid[output_row] = False
            continue
        for feature, spec in enumerate(specs):
            cross = np.asarray(raw[source_row, :, feature], dtype=np.float64)
            usable = (
                mask[source_row, :, feature]
                & membership[membership_row]
                & np.isfinite(cross)
            )
            transformed = np.zeros(cross.shape, dtype=np.float64)
            if spec.transform == "rank_gauss":
                if int(usable.sum()) >= minimum_rank_names:
                    ranked, usable = rank_gauss(
                        cross, usable, membership[membership_row]
                    )
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
                transformed[usable] = np.clip(cross[usable], -spec.clip, spec.clip)
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


def observation_age_sessions_into(
    valid: NDArray[np.bool_],
    active: NDArray[np.bool_],
    output: NDArray[np.float32],
    *,
    source_rows: NDArray[np.integer] | None = None,
    decision_rows: NDArray[np.integer] | None = None,
    source_age_sessions: NDArray[np.floating] | None = None,
) -> None:
    """Write source ages independently of transforms or current membership.

    ``-1`` is the explicit left-censored/never-observed sentinel. A current
    raw observation has age zero.  ``source_rows`` identifies the raw snapshot
    consumed by each output decision; ``decision_rows`` identifies the decision
    session used for age arithmetic and membership.  The scan includes raw
    rows before the requested output window, so a pre-window observation keeps
    its honest as-of age.  Crucially, rank support and transformed validity do
    not reset the source clock.
    """

    mask = np.asarray(valid, dtype=np.bool_)
    membership = np.asarray(active, dtype=np.bool_)
    if mask.ndim != 3 or membership.shape != mask.shape[:2]:
        raise ValueError("feature validity and membership axes are misaligned")
    destination = np.asarray(output)
    rows = (
        np.arange(mask.shape[0], dtype=np.int64)
        if source_rows is None
        else np.asarray(source_rows, dtype=np.int64)
    )
    decisions = (
        rows if decision_rows is None else np.asarray(decision_rows, dtype=np.int64)
    )
    if rows.ndim != 1 or decisions.shape != rows.shape:
        raise ValueError("source_rows and decision_rows must be aligned vectors")
    if (
        np.any(rows < -1)
        or np.any(rows >= mask.shape[0])
        or np.any(decisions < 0)
        or np.any(decisions >= mask.shape[0])
        or (rows.size > 1 and np.any(np.diff(rows) < 0))
        or np.any(decisions < rows)
    ):
        raise ValueError("source/decision rows violate the causal calendar")
    if (
        destination.shape != (rows.size, *mask.shape[1:])
        or destination.dtype != np.float32
    ):
        raise ValueError("feature age destination must be aligned float32")
    destination[...] = -1.0
    last_seen = np.full(mask.shape[1:], -1, dtype=np.int32)
    scanned_through = -1
    for output_row, (source_row, decision_row) in enumerate(
        zip(rows, decisions, strict=True)
    ):
        if source_row >= 0:
            for raw_row in range(scanned_through + 1, source_row + 1):
                usable = mask[raw_row]
                last_seen[usable] = (
                    raw_row
                    if source_age_sessions is None
                    else raw_row - source_age_sessions[raw_row, usable].astype(np.int32)
                )
            scanned_through = max(scanned_through, int(source_row))
        seen = last_seen >= 0
        destination[output_row, seen] = (decision_row - last_seen[seen]).astype(
            np.float32
        )
        destination[output_row, ~membership[decision_row], :] = -1.0
