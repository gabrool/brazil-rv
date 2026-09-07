from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from brazil_rv.execution.stateful_ledger import (
    LedgerConfig,
    StatefulLedgerResult,
    TERMINAL_SETTLEMENT_CONVENTION,
    ledger_configurations,
    ledger_sensitivity_grid,
    simulate_stateful_ledger,
)
from brazil_rv.modeling.metrics import average_ranks, moving_block_bootstrap

from .artifacts import write_json_atomic
from .config import FULL_PROTOCOL, ProtocolPreset
from .contract import (
    HORIZONS,
    PRIMARY_HORIZONS,
    REGISTERED_PRIMARY_TARGET,
)
from .corporate_actions import AlignedActionTerms
from .splits import (
    PREREGISTRATION_ROOT,
    authorize_dates,
    validate_contiguous_session_axis,
)

MIN_CROSS_SECTION = 20
BOOTSTRAP_SEED = 20260903
ECONOMICS_COSTS_BPS = (2.0, 4.0, 7.0)
ECONOMICS_ANNUAL_BORROW_RATES = (0.02, 0.04)
ECONOMICS_HEADLINE = (4.0, 0.02)
EVALUATION_SCHEMA = "BRAZIL_RV_V2_EVALUATION_V10"
PAIRED_COMPARISON_SCHEMA = "BRAZIL_RV_V2_PAIRED_COMPARISON_V3"


def primary_population_protocol() -> dict[str, object]:
    """Describe the exact population and aggregation used by the primary IC."""

    return {
        "target": REGISTERED_PRIMARY_TARGET,
        "horizons_sessions": list(PRIMARY_HORIZONS),
        "requirements": [
            "active_at_entry",
            "finite_target_scale_sigma_greater_than_1e-8",
            "valid_and_finite_neutral_target_on_every_primary_horizon",
            "valid_and_finite_neutralization_characteristics",
            "valid_and_finite_score_on_every_primary_horizon",
        ],
        "minimum_cross_section_names": MIN_CROSS_SECTION,
        "per_horizon_metric": "tie_aware_spearman",
        "daily_aggregation": "equal_mean_of_all_primary_horizons_when_all_defined",
    }


def headline_ledger_protocol() -> dict[str, object]:
    """Return the exact registered headline signal and ledger configuration."""

    config = LedgerConfig()
    if (
        config.cost_bps_per_side != ECONOMICS_HEADLINE[0]
        or config.annual_borrow_rate != ECONOMICS_HEADLINE[1]
    ):
        raise RuntimeError("headline constants differ from the default ledger")
    return {
        "signal": "tie_aware_rank_average_D1_D2_D3_D5",
        "signal_horizons_sessions": list(PRIMARY_HORIZONS),
        "ledger": asdict(config),
    }


@dataclass(frozen=True)
class EvaluationInputs:
    dates: tuple[date, ...]
    session_indices: NDArray[np.integer]
    calendar_identity_sha256: str
    scores: NDArray[np.floating]
    score_mask: NDArray[np.bool_]
    scaled_midrank_targets: NDArray[np.floating]
    scaled_target_mask: NDArray[np.bool_]
    neutral_midrank_targets: NDArray[np.floating]
    neutral_target_mask: NDArray[np.bool_]
    shareholder_midrank_targets: NDArray[np.floating]
    shareholder_simple_returns: NDArray[np.floating]
    shareholder_target_mask: NDArray[np.bool_]
    price_midrank_targets: NDArray[np.floating]
    price_target_mask: NDArray[np.bool_]
    active: NDArray[np.bool_]
    raw_close: NDArray[np.floating]
    action_shares_per_prior_share: NDArray[np.floating]
    action_cash_per_prior_share: NDArray[np.floating]
    action_session_resolved: NDArray[np.bool_]
    action_has_action: NDArray[np.bool_]
    action_successor_index: NDArray[np.integer]
    action_payment_session: NDArray[np.integer]
    security_ids: tuple[str, ...]
    target_scale_sigma: NDArray[np.floating]
    prior_feature_values: Mapping[str, NDArray[np.floating]]
    cdi_returns: NDArray[np.floating]
    transfer_chronology_clean: bool
    action_terms_source: str = "verified_contractual_terms"
    schedule_source: str = "explicit_versioned_schedule"
    horizons: tuple[int, ...] = HORIZONS
    source_artifact_hashes: Mapping[str, str] | None = None
    history_age_sessions: NDArray[np.floating] | None = None
    source_archive_present: Mapping[str, NDArray[np.bool_]] | None = None
    source_feature_valid: Mapping[str, NDArray[np.bool_]] | None = None
    initial_reference_price: NDArray[np.floating] | None = None
    eventual_survives_to_final_year: NDArray[np.bool_] | None = None
    action_alignment: str = "retrospective"
    annual_borrow_rate_by_name: NDArray[np.floating] | None = None
    shortable: NDArray[np.bool_] | None = None


@dataclass(frozen=True)
class EvaluationResult:
    report: dict[str, object]
    dates: tuple[date, ...]
    daily_primary_ic: NDArray[np.float64]
    headline_economics_dates: tuple[date, ...]
    headline_net_excess_bps: NDArray[np.float64]
    primary_scores: NDArray[np.float64]
    primary_targets: NDArray[np.float64]
    primary_outcome_mask: NDArray[np.bool_]
    primary_score_mask: NDArray[np.bool_]


def _array_sha256(values: NDArray[np.generic]) -> str:
    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(value) for value in array.shape).encode("ascii"))
    digest.update(b"\0")
    if array.ndim == 0:
        digest.update(np.ascontiguousarray(array).tobytes())
    else:
        for row in array:
            digest.update(np.ascontiguousarray(row).tobytes())
    return digest.hexdigest()


def _dates_sha256(dates: Sequence[date]) -> str:
    digest = hashlib.sha256()
    for value in dates:
        digest.update(value.isoformat().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _strings_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _finite_mean(values: NDArray[np.floating]) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else math.nan


def _spearman_result(
    left: NDArray[np.floating],
    right: NDArray[np.floating],
    mask: NDArray[np.bool_],
) -> tuple[float, int, str | None]:
    """Return a rank correlation with explicit support and failure reason."""

    valid = np.asarray(mask, dtype=np.bool_).copy()
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    valid &= np.isfinite(left_values) & np.isfinite(right_values)
    count = int(valid.sum())
    if count < MIN_CROSS_SECTION:
        return math.nan, count, "fewer_than_20_valid_names"
    left_ranks = average_ranks(left_values[valid])
    right_ranks = average_ranks(right_values[valid])
    left_ranks -= left_ranks.mean()
    right_ranks -= right_ranks.mean()
    left_scale = float(np.square(left_ranks).sum())
    right_scale = float(np.square(right_ranks).sum())
    if left_scale == 0.0 and right_scale == 0.0:
        return math.nan, count, "constant_score_and_target"
    if left_scale == 0.0:
        return math.nan, count, "constant_score"
    if right_scale == 0.0:
        return math.nan, count, "constant_target"
    return (
        float(np.dot(left_ranks, right_ranks) / np.sqrt(left_scale * right_scale)),
        count,
        None,
    )


def _spearman(
    left: NDArray[np.floating],
    right: NDArray[np.floating],
    mask: NDArray[np.bool_],
) -> float:
    return _spearman_result(left, right, mask)[0]


def _validate(inputs: EvaluationInputs) -> None:
    dates = inputs.dates
    validate_contiguous_session_axis(dates, inputs.session_indices)
    _validate_sha256(inputs.calendar_identity_sha256, label="calendar_identity_sha256")
    if tuple(inputs.horizons) != HORIZONS:
        raise ValueError("evaluation horizon axis differs from the frozen v2 contract")
    if not isinstance(inputs.transfer_chronology_clean, bool):
        raise TypeError("transfer_chronology_clean must be an explicit Boolean")
    if not inputs.action_terms_source or not inputs.schedule_source:
        raise ValueError("evaluation requires action and schedule source-tier labels")
    if inputs.action_alignment != "retrospective":
        raise ValueError(
            "ledger accounting requires the retrospective action alignment"
        )
    scores = np.asarray(inputs.scores)
    active = np.asarray(inputs.active)
    if active.ndim != 2 or active.shape[0] != len(dates):
        raise ValueError("active mask must have date-by-name shape")
    expected = (len(dates), active.shape[1], len(HORIZONS))
    if scores.shape != expected:
        raise ValueError("scores must have date-by-name-by-horizon shape")
    matrix_shape = expected[:2]
    if active.shape != matrix_shape:
        raise ValueError("active mask shape differs from scores")
    score_mask = np.asarray(inputs.score_mask)
    if score_mask.shape != expected:
        raise ValueError("score_mask shape differs from scores")
    if score_mask.dtype != np.bool_:
        raise TypeError("score_mask must be a Boolean array")
    if active.dtype != np.bool_:
        raise TypeError("active must be a Boolean array")

    # Outcome masks are authorized and checked through their exact endpoint
    # window before any numeric target payload is decoded.  This ordering is
    # material at the F3 tail, whose later endpoints enter a sealed window.
    target_masks = (
        ("neutral_target_mask", inputs.neutral_target_mask),
        ("scaled_target_mask", inputs.scaled_target_mask),
        ("shareholder_target_mask", inputs.shareholder_target_mask),
        ("price_target_mask", inputs.price_target_mask),
    )
    for name, raw_values in target_masks:
        values = np.asarray(raw_values)
        if values.shape != expected:
            raise ValueError(f"{name} shape differs from scores")
        if values.dtype != np.bool_:
            raise TypeError(f"{name} must be a Boolean array")
        for horizon_index, horizon in enumerate(HORIZONS):
            if values[-horizon:, :, horizon_index].any():
                raise ValueError(
                    f"{name} permits a horizon endpoint outside the window"
                )

    numeric_targets = (
        ("neutral_midrank_targets", inputs.neutral_midrank_targets),
        ("scaled_midrank_targets", inputs.scaled_midrank_targets),
        ("shareholder_midrank_targets", inputs.shareholder_midrank_targets),
        ("shareholder_simple_returns", inputs.shareholder_simple_returns),
        ("price_midrank_targets", inputs.price_midrank_targets),
    )
    for name, values in numeric_targets:
        if getattr(values, "shape", None) != expected:
            raise ValueError(f"{name} shape differs from scores")
    scaled = np.asarray(inputs.scaled_midrank_targets)
    neutral = np.asarray(inputs.neutral_midrank_targets)
    shareholder_rank = np.asarray(inputs.shareholder_midrank_targets)
    shareholder_return = np.asarray(inputs.shareholder_simple_returns)
    price_rank = np.asarray(inputs.price_midrank_targets)
    for name, values, mask in (
        ("neutral_midrank_targets", neutral, inputs.neutral_target_mask),
        ("scaled_midrank_targets", scaled, inputs.scaled_target_mask),
        (
            "shareholder_midrank_targets",
            shareholder_rank,
            inputs.shareholder_target_mask,
        ),
        (
            "shareholder_simple_returns",
            shareholder_return,
            inputs.shareholder_target_mask,
        ),
        ("price_midrank_targets", price_rank, inputs.price_target_mask),
    ):
        if not np.isfinite(values[np.asarray(mask, dtype=np.bool_)]).all():
            raise ValueError(f"{name} contains a non-finite valid outcome")
    for name, values in (
        ("raw_close", inputs.raw_close),
        (
            "action_shares_per_prior_share",
            inputs.action_shares_per_prior_share,
        ),
        ("action_cash_per_prior_share", inputs.action_cash_per_prior_share),
        ("action_session_resolved", inputs.action_session_resolved),
        ("action_has_action", inputs.action_has_action),
        ("action_successor_index", inputs.action_successor_index),
        ("action_payment_session", inputs.action_payment_session),
        ("target_scale_sigma", inputs.target_scale_sigma),
    ):
        if np.asarray(values).shape != matrix_shape:
            raise ValueError(f"{name} shape differs from scores")
    if np.asarray(inputs.cdi_returns).shape != (len(dates),):
        raise ValueError("CDI return axis differs from evaluation dates")
    if inputs.initial_reference_price is not None:
        reference = np.asarray(inputs.initial_reference_price, dtype=np.float64)
        if reference.shape != (matrix_shape[1],):
            raise ValueError("initial reference-price axis differs from securities")
        if np.isinf(reference).any() or np.any(
            reference[np.isfinite(reference)] <= 0.0
        ):
            raise ValueError("initial reference prices must be positive or missing")
    if inputs.annual_borrow_rate_by_name is not None:
        rates = np.asarray(inputs.annual_borrow_rate_by_name, dtype=np.float64)
        if rates.shape != matrix_shape or np.isinf(rates).any():
            raise ValueError("lending borrow rates are misaligned or infinite")
    if inputs.shortable is not None:
        shortable = np.asarray(inputs.shortable)
        if shortable.shape != matrix_shape or shortable.dtype != np.bool_:
            raise ValueError("shortable mask must be Boolean and align names")
    for name, values in (
        ("action_session_resolved", inputs.action_session_resolved),
        ("action_has_action", inputs.action_has_action),
    ):
        if np.asarray(values).dtype != np.bool_:
            raise TypeError(f"{name} must be Boolean")
    for name, values in (
        ("action_successor_index", inputs.action_successor_index),
        ("action_payment_session", inputs.action_payment_session),
    ):
        if not np.issubdtype(np.asarray(values).dtype, np.integer):
            raise TypeError(f"{name} must be integer")
    if (
        len(inputs.security_ids) != matrix_shape[1]
        or len(set(inputs.security_ids)) != matrix_shape[1]
        or any(not value for value in inputs.security_ids)
    ):
        raise ValueError("security_ids must be nonempty, unique, and align names")
    expected_features = {
        "yang_zhang_vol_20",
        "beta_60",
        "log_volume_mean_20",
        "momentum_12_1",
        "log_return_5",
    }
    if set(inputs.prior_feature_values) != expected_features:
        raise ValueError("evaluation prior-feature diagnostic roster differs")
    if any(
        np.asarray(values).shape != matrix_shape
        for values in inputs.prior_feature_values.values()
    ):
        raise ValueError("evaluation prior-feature arrays are misaligned")
    if inputs.history_age_sessions is not None:
        history_age = np.asarray(inputs.history_age_sessions, dtype=np.float64)
        if history_age.shape != matrix_shape:
            raise ValueError("history-age diagnostics are misaligned")
        if np.isinf(history_age).any() or np.any(
            history_age[np.isfinite(history_age)] < 0
        ):
            raise ValueError("history-age diagnostics must be non-negative or missing")
    if inputs.eventual_survives_to_final_year is not None:
        survives = np.asarray(inputs.eventual_survives_to_final_year)
        if survives.shape != matrix_shape:
            raise ValueError("eventual-survival audit labels are misaligned")
        if survives.dtype != np.bool_:
            raise TypeError("eventual-survival audit labels must be Boolean")
    present = inputs.source_archive_present
    valid = inputs.source_feature_valid
    if (present is None) != (valid is None):
        raise ValueError(
            "source coverage diagnostics require both archive presence and validity"
        )
    if present is not None and valid is not None:
        if not present or set(present) != set(valid):
            raise ValueError("source coverage diagnostic rosters differ or are empty")
        for source in sorted(present):
            source_present = np.asarray(present[source])
            source_valid = np.asarray(valid[source])
            if (
                not source
                or source_present.shape != matrix_shape
                or source_valid.shape != matrix_shape
            ):
                raise ValueError("source coverage diagnostics are misaligned")
            if source_present.dtype != np.bool_ or source_valid.dtype != np.bool_:
                raise TypeError("source coverage diagnostics must be Boolean")
            if np.any(source_valid & ~source_present):
                raise ValueError(
                    "source feature validity exists without archive presence"
                )
    if not np.isfinite(np.asarray(inputs.cdi_returns, dtype=np.float64)).all():
        raise ValueError("CDI returns must be finite")
    if not inputs.source_artifact_hashes:
        raise ValueError("evaluation requires at least one source artifact identity")
    for name, digest in inputs.source_artifact_hashes.items():
        if not name:
            raise ValueError("source artifact hash names must be nonempty")
        _validate_sha256(digest, label="source artifact hashes")


def _validate_sha256(value: str, *, label: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")


def _aligned_action_terms(inputs: EvaluationInputs) -> AlignedActionTerms:
    return AlignedActionTerms(
        shares_per_prior_share=np.asarray(
            inputs.action_shares_per_prior_share, dtype=np.float64
        ),
        cash_per_prior_share=np.asarray(
            inputs.action_cash_per_prior_share, dtype=np.float64
        ),
        session_resolved=np.asarray(inputs.action_session_resolved, dtype=np.bool_),
        has_action=np.asarray(inputs.action_has_action, dtype=np.bool_),
        successor_index=np.asarray(inputs.action_successor_index, dtype=np.int64),
    )


def _primary_population_components(
    inputs: EvaluationInputs,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
    NDArray[np.bool_],
]:
    indexes = [HORIZONS.index(horizon) for horizon in PRIMARY_HORIZONS]
    scores = np.asarray(inputs.scores, dtype=np.float64)[..., indexes]
    targets = np.asarray(inputs.neutral_midrank_targets, dtype=np.float64)[..., indexes]
    target_mask = np.asarray(inputs.neutral_target_mask, dtype=np.bool_)[..., indexes]
    score_mask = np.asarray(inputs.score_mask, dtype=np.bool_)[..., indexes]
    scale = np.asarray(inputs.target_scale_sigma, dtype=np.float64)
    outcome_population = (
        np.asarray(inputs.active, dtype=np.bool_)
        & np.isfinite(scale)
        & (scale > 1e-8)
        & target_mask.all(axis=-1)
        & np.isfinite(targets).all(axis=-1)
    )
    score_population = score_mask.all(axis=-1) & np.isfinite(scores).all(axis=-1)
    return scores, targets, outcome_population, score_population


def _primary_daily_metrics(
    scores: NDArray[np.float64],
    targets: NDArray[np.float64],
    outcome_population: NDArray[np.bool_],
    score_population: NDArray[np.bool_],
    dates: Sequence[date],
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[dict[str, object]]]:
    head_ic = np.full((len(dates), len(PRIMARY_HORIZONS)), np.nan, dtype=np.float64)
    daily_primary = np.full(len(dates), np.nan, dtype=np.float64)
    rows: list[dict[str, object]] = []
    for day, day_value in enumerate(dates):
        outcome_count = int(outcome_population[day].sum())
        population = outcome_population[day] & score_population[day]
        common_count = int(population.sum())
        reasons: list[str] = []
        if outcome_count < MIN_CROSS_SECTION:
            reasons.append("fewer_than_20_common_neutral_outcomes")
        elif common_count < MIN_CROSS_SECTION:
            reasons.append("fewer_than_20_common_scores")
        else:
            for horizon_index, horizon in enumerate(PRIMARY_HORIZONS):
                value, _, reason = _spearman_result(
                    scores[day, :, horizon_index],
                    targets[day, :, horizon_index],
                    population,
                )
                head_ic[day, horizon_index] = value
                if reason is not None:
                    reasons.append(f"D{horizon}:{reason}")
        if not reasons:
            # Every head is defined on this exact population; a plain mean is
            # therefore the registered equal-four-head aggregation, not a
            # missing-head nanmean.
            daily_primary[day] = float(head_ic[day].mean())
        rows.append(
            {
                "date": day_value.isoformat(),
                "possible": outcome_count >= MIN_CROSS_SECTION,
                "used": not reasons,
                "common_neutral_outcome_name_count": outcome_count,
                "common_score_and_outcome_name_count": common_count,
                "score_support_loss_name_count": outcome_count - common_count,
                "head_neutral_target_spearman_ic": {
                    f"D{horizon}": _finite_or_none(head_ic[day, index])
                    for index, horizon in enumerate(PRIMARY_HORIZONS)
                },
                "primary_neutral_target_ic": _finite_or_none(daily_primary[day]),
                "undefined_reason": ";".join(reasons) if reasons else None,
            }
        )
    return head_ic, daily_primary, rows


def _daily_metrics(
    inputs: EvaluationInputs,
    primary_population: NDArray[np.bool_],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    list[dict[str, object]],
]:
    scores = np.asarray(inputs.scores, dtype=np.float64)
    score_mask = np.asarray(inputs.score_mask, dtype=np.bool_)
    active = np.asarray(inputs.active, dtype=np.bool_)
    scaled_mask = np.asarray(inputs.scaled_target_mask, dtype=np.bool_)
    neutral_mask = np.asarray(inputs.neutral_target_mask, dtype=np.bool_)
    shareholder_mask = np.asarray(inputs.shareholder_target_mask, dtype=np.bool_)
    price_mask = np.asarray(inputs.price_target_mask, dtype=np.bool_)
    scaled = np.asarray(inputs.scaled_midrank_targets, dtype=np.float64)
    neutral = np.asarray(inputs.neutral_midrank_targets, dtype=np.float64)
    shareholder_rank = np.asarray(inputs.shareholder_midrank_targets, dtype=np.float64)
    shareholder_return = np.asarray(inputs.shareholder_simple_returns, dtype=np.float64)
    price_rank = np.asarray(inputs.price_midrank_targets, dtype=np.float64)
    scale = np.asarray(inputs.target_scale_sigma, dtype=np.float64)
    days = len(inputs.dates)
    horizon_count = len(HORIZONS)
    neutral_ic = np.full((days, horizon_count), np.nan, dtype=np.float64)
    legacy_scaled_ic = np.full_like(neutral_ic, np.nan)
    shareholder_ic = np.full_like(neutral_ic, np.nan)
    price_ic = np.full_like(neutral_ic, np.nan)
    spread_total_bps = np.full_like(neutral_ic, np.nan)
    rows: list[dict[str, object]] = []
    primary_indexes = {
        HORIZONS.index(horizon): index for index, horizon in enumerate(PRIMARY_HORIZONS)
    }
    for day, day_value in enumerate(inputs.dates):
        for horizon_index, horizon in enumerate(HORIZONS):
            if horizon_index in primary_indexes:
                neutral_valid = primary_population[day]
                neutral_population = "common_D1_D2_D3_D5"
            else:
                neutral_valid = (
                    active[day]
                    & score_mask[day, :, horizon_index]
                    & neutral_mask[day, :, horizon_index]
                    & np.isfinite(scores[day, :, horizon_index])
                    & np.isfinite(neutral[day, :, horizon_index])
                )
                neutral_population = "per_horizon_D10"
            neutral_value, neutral_count, neutral_reason = _spearman_result(
                scores[day, :, horizon_index],
                neutral[day, :, horizon_index],
                neutral_valid,
            )
            neutral_ic[day, horizon_index] = neutral_value
            legacy_valid = (
                active[day]
                & score_mask[day, :, horizon_index]
                & scaled_mask[day, :, horizon_index]
                & np.isfinite(scale[day])
                & (scale[day] > 1e-8)
                & np.isfinite(scores[day, :, horizon_index])
                & np.isfinite(scaled[day, :, horizon_index])
            )
            legacy_value, legacy_count, legacy_reason = _spearman_result(
                scores[day, :, horizon_index],
                scaled[day, :, horizon_index],
                legacy_valid,
            )
            legacy_scaled_ic[day, horizon_index] = legacy_value

            shareholder_valid = (
                active[day]
                & score_mask[day, :, horizon_index]
                & shareholder_mask[day, :, horizon_index]
                & np.isfinite(scores[day, :, horizon_index])
                & np.isfinite(shareholder_rank[day, :, horizon_index])
            )
            shareholder_value, shareholder_count, shareholder_reason = _spearman_result(
                scores[day, :, horizon_index],
                shareholder_rank[day, :, horizon_index],
                shareholder_valid,
            )
            shareholder_ic[day, horizon_index] = shareholder_value

            price_valid = (
                active[day]
                & score_mask[day, :, horizon_index]
                & price_mask[day, :, horizon_index]
                & np.isfinite(scores[day, :, horizon_index])
                & np.isfinite(price_rank[day, :, horizon_index])
            )
            price_value, price_count, price_reason = _spearman_result(
                scores[day, :, horizon_index],
                price_rank[day, :, horizon_index],
                price_valid,
            )
            price_ic[day, horizon_index] = price_value

            spread_valid = shareholder_valid & np.isfinite(
                shareholder_return[day, :, horizon_index]
            )
            names = np.flatnonzero(spread_valid)
            spread_reason: str | None = None
            if names.size < MIN_CROSS_SECTION:
                spread_reason = "fewer_than_20_valid_names"
            else:
                decile_count = max(1, names.size // 10)
                order = names[
                    np.argsort(scores[day, names, horizon_index], kind="stable")
                ]
                bottom = shareholder_return[
                    day, order[:decile_count], horizon_index
                ].mean()
                top = shareholder_return[
                    day, order[-decile_count:], horizon_index
                ].mean()
                spread_total_bps[day, horizon_index] = float(top - bottom) * 10_000.0
            rows.append(
                {
                    "date": day_value.isoformat(),
                    "horizon_sessions": horizon,
                    "neutral_target_population": neutral_population,
                    "neutral_target_valid_name_count": neutral_count,
                    "neutral_target_spearman_ic": _finite_or_none(neutral_value),
                    "neutral_target_ic_undefined_reason": neutral_reason,
                    "legacy_scaled_target_valid_name_count": legacy_count,
                    "legacy_scaled_target_ic": _finite_or_none(legacy_value),
                    "legacy_scaled_target_ic_undefined_reason": legacy_reason,
                    "shareholder_rank_valid_name_count": shareholder_count,
                    "shareholder_rank_ic": _finite_or_none(shareholder_value),
                    "shareholder_rank_ic_undefined_reason": shareholder_reason,
                    "price_return_rank_valid_name_count": price_count,
                    "price_return_rank_ic": _finite_or_none(price_value),
                    "price_return_rank_ic_undefined_reason": price_reason,
                    "shareholder_return_spread_valid_name_count": int(names.size),
                    "shareholder_return_spread_total_bps": _finite_or_none(
                        spread_total_bps[day, horizon_index]
                    ),
                    "shareholder_return_spread_bps_per_holding_session": (
                        _finite_or_none(spread_total_bps[day, horizon_index] / horizon)
                    ),
                    "shareholder_return_spread_undefined_reason": spread_reason,
                }
            )
    return (
        neutral_ic,
        legacy_scaled_ic,
        shareholder_ic,
        price_ic,
        spread_total_bps,
        rows,
    )


def _persistence(
    inputs: EvaluationInputs,
) -> tuple[dict[int, NDArray[np.float64]], list[dict[str, object]]]:
    scores = np.asarray(inputs.scores, dtype=np.float64)
    score_mask = np.asarray(inputs.score_mask, dtype=bool)
    active = np.asarray(inputs.active, dtype=bool)
    results: dict[int, NDArray[np.float64]] = {}
    rows: list[dict[str, object]] = []
    for lag in (1, 5):
        values = np.full((len(inputs.dates), len(HORIZONS)), np.nan)
        for day in range(lag, len(inputs.dates)):
            for horizon_index, _ in enumerate(HORIZONS):
                valid = (
                    active[day]
                    & active[day - lag]
                    & score_mask[day, :, horizon_index]
                    & score_mask[day - lag, :, horizon_index]
                )
                values[day, horizon_index] = _spearman(
                    scores[day, :, horizon_index],
                    scores[day - lag, :, horizon_index],
                    valid,
                )
        results[lag] = values
        for day, day_value in enumerate(inputs.dates):
            for horizon_index, horizon in enumerate(HORIZONS):
                rows.append(
                    {
                        "date": day_value.isoformat(),
                        "lag_sessions": lag,
                        "horizon_sessions": horizon,
                        "path_model_count": 1,
                        "spearman": _finite_or_none(values[day, horizon_index]),
                    }
                )
    return results, rows


def _economics_signal(
    inputs: EvaluationInputs,
    scores_input: NDArray[np.floating] | None = None,
    score_mask_input: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    indexes = [HORIZONS.index(horizon) for horizon in PRIMARY_HORIZONS]
    scores = np.asarray(
        inputs.scores if scores_input is None else scores_input, dtype=np.float64
    )[..., indexes]
    masks = np.asarray(
        inputs.score_mask if score_mask_input is None else score_mask_input, dtype=bool
    )[..., indexes]
    active = np.asarray(inputs.active, dtype=bool)
    composite = np.full(scores.shape[:2], np.nan, dtype=np.float64)
    composite_mask = masks.all(axis=-1) & active & np.isfinite(scores).all(axis=-1)
    for day in range(scores.shape[0]):
        valid = composite_mask[day]
        count = int(valid.sum())
        if count:
            head_ranks = np.stack(
                [
                    average_ranks(scores[day, valid, horizon_index])
                    for horizon_index in range(len(PRIMARY_HORIZONS))
                ]
            )
            averaged_rank = head_ranks.mean(axis=0)
            composite[day, valid] = 2.0 * ((averaged_rank + 0.5) / count) - 1.0
    return composite, composite_mask


def _ledger_rows(
    result: StatefulLedgerResult,
    *,
    cost_bps: float,
    annual_borrow_rate: float,
) -> list[dict[str, object]]:
    order_count = {day: 0 for day in result.dates}
    fill_count = {day: 0 for day in result.dates}
    cancellation_count = {day: 0 for day in result.dates}
    for order in result.intended_orders:
        if order.decision_date in order_count:
            order_count[order.decision_date] += 1
    for fill in result.fills:
        if fill.fill_date in fill_count:
            fill_count[fill.fill_date] += 1
    for cancellation in result.cancellations:
        if cancellation.cancellation_date in cancellation_count:
            cancellation_count[cancellation.cancellation_date] += 1
    signed_values = np.where(
        (result.signed_shares != 0.0) & np.isfinite(result.mark_price),
        result.signed_shares * result.mark_price,
        0.0,
    )
    actual_net = np.divide(
        signed_values.sum(axis=1),
        result.nav,
        out=np.full(len(result.dates), np.nan, dtype=np.float64),
        where=result.nav != 0.0,
    )
    return [
        {
            "date": day.isoformat(),
            "cost_bps_per_side": cost_bps,
            "annual_borrow_rate": annual_borrow_rate,
            "borrow_source": result.borrow_source,
            "economics_resolved": not result.economics_unresolved,
            "gross_pnl_bps": _finite_or_none(result.gross_pnl_bps[index]),
            "interest_bps": _finite_or_none(result.interest_bps[index]),
            "turnover_fraction_nav": _finite_or_none(
                result.turnover_fraction_nav[index]
            ),
            "unresolved_stale_inventory_fraction_nav": _finite_or_none(
                result.unresolved_stale_inventory_fraction_nav[index]
            ),
            "unresolved_claim_inventory_fraction_nav": _finite_or_none(
                result.unresolved_claim_inventory_fraction_nav[index]
            ),
            "stale_mark_inventory_fraction_nav": _finite_or_none(
                result.stale_mark_inventory_fraction_nav[index]
            ),
            "turnover_cost_bps": _finite_or_none(result.cost_bps[index]),
            "borrow_cost_bps": _finite_or_none(result.borrow_bps[index]),
            "held_short_weighted_annual_borrow_rate": _finite_or_none(
                result.held_short_weighted_annual_borrow_rate[index]
            ),
            "excluded_short_entry_candidate_count": int(
                result.excluded_short_entry_candidate_count[index]
            ),
            "net_return": _finite_or_none(result.daily_net_return[index]),
            "net_excess_all_cash_bps": _finite_or_none(
                result.net_excess_all_cash_bps[index]
            ),
            "deployed_gross_fraction_nav": _finite_or_none(
                result.gross_fraction_nav[index]
            ),
            "deployed_net_fraction_nav": _finite_or_none(actual_net[index]),
            "nav": _finite_or_none(result.nav[index]),
            "free_cash": _finite_or_none(result.free_cash[index]),
            "restricted_cash": _finite_or_none(result.restricted_cash[index]),
            "receivables": _finite_or_none(result.receivables[index]),
            "payables": _finite_or_none(result.payables[index]),
            "marked_signed_holdings": _finite_or_none(
                result.marked_signed_holdings[index]
            ),
            "reconciliation_error": _finite_or_none(result.reconciliation_error[index]),
            "stale_mark_name_days": int(result.stale_mark_name_days[index]),
            "unresolved_action_name_days": int(
                result.unresolved_action_name_days[index]
            ),
            "valuation_scenario_count": int(result.valuation_scenario_count[index]),
            "terminal_settlement_count": int(result.terminal_settlement_count[index]),
            "terminal_settlement_notional": _finite_or_none(
                result.terminal_settlement_notional[index]
            ),
            "terminal_settlement_notional_fraction_nav": _finite_or_none(
                result.terminal_settlement_notional_fraction_nav[index]
            ),
            "settled_then_printed_count": int(result.settled_then_printed_count[index]),
            "terminal_settlement_haircut_scenario_nav": _finite_or_none(
                result.settlement_haircut_scenario_nav[index]
            ),
            "planned_gross_fraction_nav": _finite_or_none(
                result.planned_gross_fraction_nav[index]
            ),
            "planned_net_fraction_nav": _finite_or_none(
                result.planned_net_fraction_nav[index]
            ),
            "planned_name_weight_fraction_nav": _finite_or_none(
                result.planned_name_weight_fraction_nav[index]
            ),
            "actual_risk_breach": bool(result.actual_risk_breach[index]),
            "pending_entry_count": int(result.pending_entry_count[index]),
            "pending_exit_count": int(result.pending_exit_count[index]),
            "cancelled_entry_count": int(result.cancelled_entry_count[index]),
            "submitted_entry_count": int(result.submitted_entry_count[index]),
            "submitted_exit_count": int(result.submitted_exit_count[index]),
            "same_close_replacement_count": int(
                result.same_close_replacement_count[index]
            ),
            "blocked_entry_no_reference_count": int(
                result.blocked_entry_no_reference_count[index]
            ),
            "blocked_entry_gross_cap_count": int(
                result.blocked_entry_gross_cap_count[index]
            ),
            "blocked_entry_net_cap_count": int(
                result.blocked_entry_net_cap_count[index]
            ),
            "blocked_entry_name_cap_count": int(
                result.blocked_entry_name_cap_count[index]
            ),
            "zero_entry_small_universe": bool(result.zero_entry_small_universe[index]),
            "zero_entry_gross_cap": bool(result.zero_entry_gross_cap[index]),
            "zero_entry_net_cap": bool(result.zero_entry_net_cap[index]),
            "zero_entry_name_cap": bool(result.zero_entry_name_cap[index]),
            "zero_entry_no_reference": bool(result.zero_entry_no_reference[index]),
            "risk_trim_gross_notional": _finite_or_none(
                result.risk_trim_gross_notional[index]
            ),
            "risk_trim_net_notional": _finite_or_none(
                result.risk_trim_net_notional[index]
            ),
            "risk_trim_name_notional": _finite_or_none(
                result.risk_trim_name_notional[index]
            ),
            "pending_exit_mean_age_sessions": _finite_or_none(
                result.pending_exit_mean_age_sessions[index]
            ),
            "intended_order_count": order_count[day],
            "fill_count": fill_count[day],
            "order_cancellation_count": cancellation_count[day],
        }
        for index, day in enumerate(result.dates)
    ]


def _serialise_records(records: Sequence[object]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for record in records:
        row = asdict(record)
        for key, value in row.items():
            if isinstance(value, date):
                row[key] = value.isoformat()
        output.append(row)
    return output


def _holding_audit(
    result: StatefulLedgerResult,
    security_ids: Sequence[str],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    daily: list[dict[str, object]] = []
    observed_ages: list[int] = []
    for day_index, day_value in enumerate(result.dates):
        held = np.flatnonzero(result.signed_shares[day_index] != 0.0)
        ages = result.holding_age_sessions[day_index, held]
        observed_ages.extend(int(value) for value in ages)
        daily.append(
            {
                "date": day_value.isoformat(),
                "held_name_count": int(held.size),
                "mean_holding_age_sessions": (
                    _finite_or_none(float(np.mean(ages))) if ages.size else None
                ),
                "median_holding_age_sessions": (
                    _finite_or_none(float(np.median(ages))) if ages.size else None
                ),
                "maximum_holding_age_sessions": (
                    int(np.max(ages)) if ages.size else None
                ),
            }
        )
        for name in held:
            mark = float(result.mark_price[day_index, name])
            shares = float(result.signed_shares[day_index, name])
            nav = float(result.nav[day_index])
            rows.append(
                {
                    "date": day_value.isoformat(),
                    "security": security_ids[name],
                    "security_index": int(name),
                    "signed_shares": shares,
                    "mark_price": _finite_or_none(mark),
                    "signed_marked_value": _finite_or_none(shares * mark),
                    "signed_weight_fraction_nav": (
                        _finite_or_none(shares * mark / nav)
                        if math.isfinite(mark) and nav != 0.0
                        else None
                    ),
                    "holding_age_sessions": int(
                        result.holding_age_sessions[day_index, name]
                    ),
                }
            )
    age_array = np.asarray(observed_ages, dtype=np.int64)
    histogram = (
        {
            str(int(value)): int(count)
            for value, count in zip(
                *np.unique(age_array, return_counts=True), strict=True
            )
        }
        if age_array.size
        else {}
    )
    return {
        "population": "end_of_session_held_name_days",
        "held_name_day_count": int(age_array.size),
        "mean_sessions": (
            _finite_or_none(float(np.mean(age_array))) if age_array.size else None
        ),
        "median_sessions": (
            _finite_or_none(float(np.median(age_array))) if age_array.size else None
        ),
        "p25_sessions": (
            _finite_or_none(float(np.quantile(age_array, 0.25)))
            if age_array.size
            else None
        ),
        "p75_sessions": (
            _finite_or_none(float(np.quantile(age_array, 0.75)))
            if age_array.size
            else None
        ),
        "p95_sessions": (
            _finite_or_none(float(np.quantile(age_array, 0.95)))
            if age_array.size
            else None
        ),
        "maximum_sessions": int(np.max(age_array)) if age_array.size else None,
        "histogram": histogram,
        "daily": daily,
        "holdings": rows,
    }


def _action_attribution(
    inputs: EvaluationInputs,
    result: StatefulLedgerResult,
) -> dict[str, object]:
    action_rows: list[dict[str, object]] = []
    daily_total = np.zeros(len(result.dates), dtype=np.float64)
    daily_defined = np.zeros(len(result.dates), dtype=np.int64)
    daily_unresolved = np.zeros(len(result.dates), dtype=np.int64)
    has_action = np.asarray(inputs.action_has_action, dtype=np.bool_)
    resolved = np.asarray(inputs.action_session_resolved, dtype=np.bool_)
    action_q = np.asarray(inputs.action_shares_per_prior_share, dtype=np.float64)
    action_d = np.asarray(inputs.action_cash_per_prior_share, dtype=np.float64)
    successor = np.asarray(inputs.action_successor_index, dtype=np.int64)
    payment = np.asarray(inputs.action_payment_session, dtype=np.int64)
    for day_index in range(len(result.dates)):
        for name in np.flatnonzero(has_action[day_index]):
            prior_shares = (
                float(result.signed_shares[day_index - 1, name])
                if day_index > 0
                else 0.0
            )
            prior_mark = (
                float(result.mark_price[day_index - 1, name])
                if day_index > 0
                else math.nan
            )
            successor_index = int(successor[day_index, name])
            action_resolved = bool(resolved[day_index, name])
            cash_claim = (
                prior_shares * float(action_d[day_index, name])
                if action_resolved
                else math.nan
            )
            crossing_pnl = math.nan
            current_mark = math.nan
            if (
                action_resolved
                and day_index > 0
                and prior_shares != 0.0
                and math.isfinite(prior_mark)
                and 0 <= successor_index < len(inputs.security_ids)
            ):
                current_mark = float(result.mark_price[day_index, successor_index])
                q = float(action_q[day_index, name])
                d = float(action_d[day_index, name])
                if (
                    math.isfinite(current_mark)
                    and math.isfinite(q)
                    and math.isfinite(d)
                ):
                    crossing_pnl = prior_shares * (q * current_mark + d - prior_mark)
                    daily_total[day_index] += crossing_pnl
                    daily_defined[day_index] += 1
            if prior_shares != 0.0 and not action_resolved:
                daily_unresolved[day_index] += 1
            payment_index = int(payment[day_index, name])
            action_rows.append(
                {
                    "date": result.dates[day_index].isoformat(),
                    "security": inputs.security_ids[name],
                    "security_index": int(name),
                    "held_prior_to_action": prior_shares != 0.0,
                    "prior_signed_shares": prior_shares,
                    "prior_mark_price": _finite_or_none(prior_mark),
                    "action_resolved": action_resolved,
                    "shares_per_prior_share": (
                        _finite_or_none(action_q[day_index, name])
                        if action_resolved
                        else None
                    ),
                    "cash_per_prior_share": (
                        _finite_or_none(action_d[day_index, name])
                        if action_resolved
                        else None
                    ),
                    "successor_security": (
                        inputs.security_ids[successor_index]
                        if 0 <= successor_index < len(inputs.security_ids)
                        else None
                    ),
                    "successor_security_index": successor_index,
                    "current_successor_mark_price": _finite_or_none(current_mark),
                    "current_successor_mark_observed": bool(
                        0 <= successor_index < len(inputs.security_ids)
                        and np.isfinite(inputs.raw_close[day_index, successor_index])
                    ),
                    "cash_claim_signed": _finite_or_none(cash_claim),
                    "claim_payment_date": (
                        result.dates[payment_index].isoformat()
                        if 0 <= payment_index < len(result.dates)
                        else None
                    ),
                    "claim_payment_session": payment_index,
                    "action_crossing_shareholder_pnl": _finite_or_none(crossing_pnl),
                    "action_crossing_shareholder_pnl_bps_start_nav": (
                        _finite_or_none(crossing_pnl / result.nav[day_index - 1] * 1e4)
                        if day_index > 0
                        and math.isfinite(crossing_pnl)
                        and result.nav[day_index - 1] != 0.0
                        else None
                    ),
                }
            )
    return {
        "interpretation": (
            "contractual shareholder wealth change across action-affected held "
            "claims; includes contemporaneous underlying price movement and is "
            "not an isolated action-alpha estimate"
        ),
        "rows": action_rows,
        "daily": [
            {
                "date": day_value.isoformat(),
                "defined_held_action_count": int(daily_defined[index]),
                "unresolved_held_action_count": int(daily_unresolved[index]),
                "action_crossing_shareholder_pnl": (
                    _finite_or_none(daily_total[index])
                    if daily_defined[index]
                    else None
                ),
            }
            for index, day_value in enumerate(result.dates)
        ],
    }


def _liquidity_quartiles(inputs: EvaluationInputs) -> NDArray[np.int8]:
    values = np.asarray(
        inputs.prior_feature_values["log_volume_mean_20"], dtype=np.float64
    )
    active = np.asarray(inputs.active, dtype=np.bool_)
    quartiles = np.full(values.shape, -1, dtype=np.int8)
    for day in range(values.shape[0]):
        names = np.flatnonzero(active[day] & np.isfinite(values[day]))
        if not names.size:
            continue
        ranks = average_ranks(values[day, names])
        quartiles[day, names] = np.minimum(
            (4.0 * (ranks + 0.5) / names.size).astype(np.int8), 3
        )
    return quartiles


def _quality_stratification(
    inputs: EvaluationInputs,
    primary_scores: NDArray[np.float64],
    primary_targets: NDArray[np.float64],
    primary_outcome_mask: NDArray[np.bool_],
    primary_score_mask: NDArray[np.bool_],
) -> dict[str, object]:
    shape = primary_outcome_mask.shape
    rows: list[dict[str, object]] = []

    def add(group: str, label: str, stratum: NDArray[np.bool_]) -> None:
        mask = np.asarray(stratum, dtype=np.bool_)
        if mask.shape != shape:
            raise ValueError(f"quality stratum {group}/{label} is misaligned")
        outcome = primary_outcome_mask & mask
        score = primary_score_mask & mask
        _, daily, _ = _primary_daily_metrics(
            primary_scores,
            primary_targets,
            outcome,
            score,
            inputs.dates,
        )
        rows.append(
            {
                "dimension": group,
                "stratum": label,
                "possible_date_count": int(
                    (outcome.sum(axis=1) >= MIN_CROSS_SECTION).sum()
                ),
                "used_date_count": int(np.isfinite(daily).sum()),
                "possible_name_days": int(outcome.sum()),
                "used_name_days": int((outcome & score).sum()),
                "mean_daily_primary_neutral_target_ic": _finite_or_none(
                    _finite_mean(daily)
                ),
                "status": "supported" if np.isfinite(daily).any() else "unsupported",
                "undefined_reason": (
                    None
                    if np.isfinite(daily).any()
                    else "no_date_has_20_names_with_four_defined_heads"
                ),
            }
        )

    years = np.asarray([value.year for value in inputs.dates], dtype=np.int32)
    for year in np.unique(years):
        add(
            "calendar_year",
            str(int(year)),
            np.broadcast_to((years == year)[:, None], shape),
        )

    quartiles = _liquidity_quartiles(inputs)
    for quartile in range(4):
        add("causal_liquidity_quartile", f"Q{quartile + 1}", quartiles == quartile)

    if inputs.history_age_sessions is None:
        rows.append(
            {
                "dimension": "history_age_sessions",
                "stratum": "unavailable",
                "possible_date_count": 0,
                "used_date_count": 0,
                "possible_name_days": 0,
                "used_name_days": 0,
                "mean_daily_primary_neutral_target_ic": None,
                "status": "unsupported",
                "undefined_reason": "history_age_not_supplied_by_evaluation_source",
            }
        )
    else:
        age = np.asarray(inputs.history_age_sessions, dtype=np.float64)
        add("history_age_sessions", "0_to_59", np.isfinite(age) & (age < 60))
        add(
            "history_age_sessions",
            "60_to_251",
            np.isfinite(age) & (age >= 60) & (age < 252),
        )
        add("history_age_sessions", "252_plus", np.isfinite(age) & (age >= 252))

    if inputs.eventual_survives_to_final_year is None:
        rows.append(
            {
                "dimension": "eventual_survival_audit_label",
                "stratum": "unavailable",
                "possible_date_count": 0,
                "used_date_count": 0,
                "possible_name_days": 0,
                "used_name_days": 0,
                "mean_daily_primary_neutral_target_ic": None,
                "status": "unsupported",
                "undefined_reason": "eventual_survival_audit_label_not_supplied",
            }
        )
    else:
        survives = np.asarray(inputs.eventual_survives_to_final_year, dtype=np.bool_)
        add("eventual_survival_audit_label", "survives_to_final_year", survives)
        add("eventual_survival_audit_label", "delisted_within_panel", ~survives)
    rows.append(
        {
            "dimension": "verified_terminal_status",
            "stratum": "unsupported",
            "possible_date_count": 0,
            "used_date_count": 0,
            "possible_name_days": 0,
            "used_name_days": 0,
            "mean_daily_primary_neutral_target_ic": None,
            "status": "unsupported",
            "undefined_reason": (
                "no verified terminal-settlement roster is supplied; final-window "
                "quote absence is not treated as terminal status"
            ),
        }
    )

    final_quote = np.isfinite(np.asarray(inputs.raw_close, dtype=np.float64)[-1])
    add(
        "window_terminal_quote_status",
        "observed_on_final_evaluation_session",
        np.broadcast_to(final_quote[None, :], shape),
    )
    add(
        "window_terminal_quote_status",
        "not_observed_on_final_evaluation_session",
        np.broadcast_to((~final_quote)[None, :], shape),
    )

    source_rows: list[dict[str, object]] = []
    if (
        inputs.source_archive_present is not None
        and inputs.source_feature_valid is not None
    ):
        active = np.asarray(inputs.active, dtype=np.bool_)
        for source in sorted(inputs.source_archive_present):
            present = np.asarray(inputs.source_archive_present[source], dtype=np.bool_)
            valid = np.asarray(inputs.source_feature_valid[source], dtype=np.bool_)
            add("source_archive_availability", f"{source}:present", present)
            add("source_archive_availability", f"{source}:absent", ~present)
            active_present = active & present
            source_rows.append(
                {
                    "source": source,
                    "active_name_days": int(active.sum()),
                    "archive_present_name_days": int(active_present.sum()),
                    "archive_coverage_rate": _finite_or_none(
                        active_present.sum() / active.sum()
                    )
                    if active.any()
                    else None,
                    "feature_valid_name_days": int((active & valid).sum()),
                    "feature_valid_conditional_on_archive_presence": (
                        _finite_or_none((active & valid).sum() / active_present.sum())
                        if active_present.any()
                        else None
                    ),
                    "status": "supported" if active_present.any() else "unsupported",
                }
            )
    else:
        source_rows.append(
            {
                "source": "all_external_families",
                "status": "unsupported",
                "undefined_reason": "archive_presence_and_feature_validity_not_supplied",
            }
        )
    return {
        "definition": (
            "diagnostic only; every IC uses the registered four-head common "
            "population within the named causal stratum, and sparse strata stay "
            "explicitly unsupported"
        ),
        "rows": rows,
        "source_coverage": source_rows,
    }


def _declared_subperiod_readouts(
    inputs: EvaluationInputs,
    neutral_ic: NDArray[np.float64],
    shareholder_ic: NDArray[np.float64],
    price_ic: NDArray[np.float64],
    spread_total_bps: NDArray[np.float64],
    *,
    window_name: str,
) -> dict[str, object]:
    years = np.asarray([value.year for value in inputs.dates], dtype=np.int32)
    rows: list[dict[str, object]] = []
    for year in np.unique(years):
        selected = years == year
        for horizon_index, horizon in enumerate(HORIZONS):
            rows.append(
                {
                    "evaluation_window": window_name,
                    "subperiod": str(int(year)),
                    "horizon_sessions": horizon,
                    "calendar_date_count": int(selected.sum()),
                    "neutral_target_used_date_count": int(
                        np.isfinite(neutral_ic[selected, horizon_index]).sum()
                    ),
                    "mean_neutral_target_spearman_ic": _finite_or_none(
                        _finite_mean(neutral_ic[selected, horizon_index])
                    ),
                    "shareholder_rank_used_date_count": int(
                        np.isfinite(shareholder_ic[selected, horizon_index]).sum()
                    ),
                    "mean_shareholder_rank_ic": _finite_or_none(
                        _finite_mean(shareholder_ic[selected, horizon_index])
                    ),
                    "price_return_rank_used_date_count": int(
                        np.isfinite(price_ic[selected, horizon_index]).sum()
                    ),
                    "mean_price_return_rank_ic": _finite_or_none(
                        _finite_mean(price_ic[selected, horizon_index])
                    ),
                    "spread_used_date_count": int(
                        np.isfinite(spread_total_bps[selected, horizon_index]).sum()
                    ),
                    "mean_shareholder_return_spread_total_bps": _finite_or_none(
                        _finite_mean(spread_total_bps[selected, horizon_index])
                    ),
                }
            )
    return {
        "definition": "calendar-year subperiods within the named evaluation window",
        "rows": rows,
    }


def _input_hashes(inputs: EvaluationInputs) -> dict[str, str]:
    result = {
        "dates": _dates_sha256(inputs.dates),
        "canonical_calendar": inputs.calendar_identity_sha256,
        "session_indices": _array_sha256(np.asarray(inputs.session_indices)),
        "scores": _array_sha256(np.asarray(inputs.scores)),
        "score_mask": _array_sha256(np.asarray(inputs.score_mask)),
        "neutral_midrank_targets": _array_sha256(
            np.asarray(inputs.neutral_midrank_targets)
        ),
        "neutral_target_mask": _array_sha256(np.asarray(inputs.neutral_target_mask)),
        "scaled_midrank_targets": _array_sha256(
            np.asarray(inputs.scaled_midrank_targets)
        ),
        "scaled_target_mask": _array_sha256(np.asarray(inputs.scaled_target_mask)),
        "shareholder_midrank_targets": _array_sha256(
            np.asarray(inputs.shareholder_midrank_targets)
        ),
        "shareholder_simple_returns": _array_sha256(
            np.asarray(inputs.shareholder_simple_returns)
        ),
        "shareholder_target_mask": _array_sha256(
            np.asarray(inputs.shareholder_target_mask)
        ),
        "price_midrank_targets": _array_sha256(
            np.asarray(inputs.price_midrank_targets)
        ),
        "price_target_mask": _array_sha256(np.asarray(inputs.price_target_mask)),
        "active": _array_sha256(np.asarray(inputs.active)),
        "raw_close": _array_sha256(np.asarray(inputs.raw_close)),
        "action_shares_per_prior_share": _array_sha256(
            np.asarray(inputs.action_shares_per_prior_share)
        ),
        "action_cash_per_prior_share": _array_sha256(
            np.asarray(inputs.action_cash_per_prior_share)
        ),
        "action_session_resolved": _array_sha256(
            np.asarray(inputs.action_session_resolved)
        ),
        "action_has_action": _array_sha256(np.asarray(inputs.action_has_action)),
        "action_successor_index": _array_sha256(
            np.asarray(inputs.action_successor_index)
        ),
        "action_payment_session": _array_sha256(
            np.asarray(inputs.action_payment_session)
        ),
        "security_ids": _strings_sha256(inputs.security_ids),
        "target_scale_sigma": _array_sha256(np.asarray(inputs.target_scale_sigma)),
        "cdi_returns": _array_sha256(np.asarray(inputs.cdi_returns)),
        "transfer_chronology_clean": _array_sha256(
            np.asarray(inputs.transfer_chronology_clean, dtype=np.bool_)
        ),
        "action_terms_source": hashlib.sha256(
            inputs.action_terms_source.encode("utf-8")
        ).hexdigest(),
        "schedule_source": hashlib.sha256(
            inputs.schedule_source.encode("utf-8")
        ).hexdigest(),
        "history_age_sessions": _array_sha256(
            np.asarray(
                inputs.history_age_sessions
                if inputs.history_age_sessions is not None
                else "unsupported"
            )
        ),
        "initial_reference_price": _array_sha256(
            np.asarray(
                inputs.initial_reference_price
                if inputs.initial_reference_price is not None
                else "unsupported"
            )
        ),
        "eventual_survives_to_final_year": _array_sha256(
            np.asarray(
                inputs.eventual_survives_to_final_year
                if inputs.eventual_survives_to_final_year is not None
                else "unsupported"
            )
        ),
    }
    for name, values in sorted(inputs.prior_feature_values.items()):
        result[f"prior_feature_{name}"] = _array_sha256(np.asarray(values))
    for roster_name, roster in (
        ("source_archive_present", inputs.source_archive_present),
        ("source_feature_valid", inputs.source_feature_valid),
    ):
        digest = hashlib.sha256()
        if roster is None:
            digest.update(b"unsupported")
        else:
            for name, values in sorted(roster.items()):
                digest.update(name.encode("utf-8"))
                digest.update(b"\0")
                digest.update(_array_sha256(np.asarray(values)).encode("ascii"))
                digest.update(b"\n")
        result[roster_name] = digest.hexdigest()
    return result


def _economics_contract(inputs: EvaluationInputs) -> dict[str, object]:
    config = LedgerConfig()
    return {
        "signal_construction": (
            "arithmetic mean of each D=1,2,3,5 head's tie-aware "
            "cross-sectional ranks, centered and rescaled to [-1,1]; "
            "all four score masks required and D=10 excluded"
        ),
        "signal_horizons_sessions": list(PRIMARY_HORIZONS),
        "k_per_side": config.k_per_side,
        "effective_k_per_side": "min(k_per_side, floor(eligible_names / 2))",
        "buffer_per_side": config.buffer_per_side,
        "gross_target": config.gross_target,
        "planned_gross_cap": config.planned_gross_cap,
        "planned_absolute_net_cap": config.planned_absolute_net_cap,
        "planned_name_weight_cap": config.planned_name_weight_cap,
        "planned_net_cap_rationale": (
            "20% permits ordinary asynchronous imbalance across 30 independent "
            "slots per side while retaining a binding directional-risk limit"
        ),
        "annual_sessions": config.annual_sessions,
        "terminal_liquidation": True,
        "stateful_policy": (
            "buffered held inventory; same-close exits free same-decision entry "
            "slots while older pending exits remain occupied; independent side "
            "refills; partial lowest-conviction gross/net/name risk trims; no "
            "discretionary drift trades"
        ),
        "unresolved_stale_gate": (
            "each evaluation mean daily unresolved-or-stale marked inventory "
            "notional is strictly below 2% of NAV"
        ),
        "missing_print_policy": (
            "stale mark while an exit is pending; fill at the first print and "
            "report a valuation scenario after 10 missing sessions"
        ),
        "marking_basis": "raw contractual close with explicit signed shares",
        "action_terms_source": inputs.action_terms_source,
        "schedule_source": inputs.schedule_source,
        "corporate_action_basis": (
            "the explicitly labelled action tier supplies q/d terms, successor "
            "shares, and cash claims; each declared payment session settles its claim"
        ),
        "short_proceeds_remuneration": config.short_proceeds_remuneration,
        "costs_bps_per_side": list(ECONOMICS_COSTS_BPS),
        "annual_borrow_rates": list(ECONOMICS_ANNUAL_BORROW_RATES),
        "headline": {
            "cost_bps_per_side": ECONOMICS_HEADLINE[0],
            "annual_borrow_rate": ECONOMICS_HEADLINE[1],
        },
    }


def _realized_beta_diagnostic(
    inputs: EvaluationInputs, headline: StatefulLedgerResult
) -> dict[str, object]:
    d1 = HORIZONS.index(1)
    returns = np.asarray(inputs.shareholder_simple_returns, dtype=np.float64)
    valid = np.asarray(inputs.shareholder_target_mask, dtype=np.bool_)
    active = np.asarray(inputs.active, dtype=np.bool_)
    market = np.full(len(inputs.dates), np.nan, dtype=np.float64)
    for ledger_day in range(1, len(inputs.dates)):
        source_day = ledger_day - 1
        names = (
            active[source_day]
            & valid[source_day, :, d1]
            & np.isfinite(returns[source_day, :, d1])
        )
        if names.any():
            market[ledger_day] = float(returns[source_day, names, d1].mean())
    book = np.full(len(inputs.dates), np.nan, dtype=np.float64)
    book[: len(headline.daily_net_return)] = headline.daily_net_return
    used = np.isfinite(market) & np.isfinite(book)
    if int(used.sum()) < 3 or float(np.var(market[used])) == 0.0:
        return {
            "status": "unsupported",
            "observation_count": int(used.sum()),
            "intercept_daily": None,
            "slope_beta": None,
            "r_squared": None,
            "classification": "unsupported",
        }
    design = np.column_stack((np.ones(int(used.sum()), dtype=np.float64), market[used]))
    coefficients, *_ = np.linalg.lstsq(design, book[used], rcond=None)
    fitted = design @ coefficients
    residual = book[used] - fitted
    total = book[used] - float(book[used].mean())
    total_ss = float(np.dot(total, total))
    r_squared = 1.0 - float(np.dot(residual, residual)) / total_ss if total_ss else 0.0
    beta = float(coefficients[1])
    return {
        "status": "supported",
        "observation_count": int(used.sum()),
        "intercept_daily": float(coefficients[0]),
        "slope_beta": beta,
        "r_squared": r_squared,
        "classification": "directional" if abs(beta) > 0.30 else "beta_neutral",
        "directional_threshold_absolute_beta": 0.30,
        "alignment": "ledger_day_t versus equal_weight_active_D1_from_t_minus_1",
    }


def _lending_coverage(inputs: EvaluationInputs) -> dict[str, object]:
    if inputs.shortable is None or inputs.annual_borrow_rate_by_name is None:
        return {"status": "unsupported"}
    shortable = np.asarray(inputs.shortable, dtype=np.bool_)
    rate_present = np.isfinite(
        np.asarray(inputs.annual_borrow_rate_by_name, dtype=np.float64)
    )
    active = np.asarray(inputs.active, dtype=np.bool_)
    volatility = np.asarray(
        inputs.prior_feature_values["yang_zhang_vol_20"], dtype=np.float64
    )
    high_vol = np.zeros(active.shape, dtype=np.bool_)
    for day in range(len(inputs.dates)):
        names = np.flatnonzero(active[day] & np.isfinite(volatility[day]))
        if names.size:
            cutoff = float(np.quantile(volatility[day, names], 0.75))
            high_vol[day, names] = volatility[day, names] >= cutoff

    def fraction(values: NDArray[np.bool_], mask: NDArray[np.bool_]) -> float | None:
        return float(values[mask].mean()) if mask.any() else None

    return {
        "status": "supported",
        "active_name_days": int(active.sum()),
        "active_rate_observed_prior_20_fraction": fraction(rate_present, active),
        "active_shortable_fraction": fraction(shortable, active),
        "high_volatility_quartile_name_days": int(high_vol.sum()),
        "high_volatility_quartile_rate_observed_prior_20_fraction": fraction(
            rate_present, high_vol
        ),
        "high_volatility_quartile_shortable_fraction": fraction(shortable, high_vol),
    }


def _diagnostics(
    inputs: EvaluationInputs, headline: StatefulLedgerResult | None = None
) -> dict[str, object]:
    scores = np.asarray(inputs.scores, dtype=np.float64)
    masks = np.asarray(inputs.score_mask, dtype=np.bool_)
    active = np.asarray(inputs.active, dtype=np.bool_)
    target_mask = np.asarray(inputs.scaled_target_mask, dtype=np.bool_)
    scaled = np.asarray(inputs.scaled_midrank_targets, dtype=np.float64)
    composite, composite_mask = _economics_signal(inputs)
    exposure_rows = []
    exposure_summary = []
    for name, raw_values in sorted(inputs.prior_feature_values.items()):
        values = np.asarray(raw_values, dtype=np.float64)
        daily = np.asarray(
            [
                _spearman(composite[day], values[day], composite_mask[day])
                for day in range(len(inputs.dates))
            ],
            dtype=np.float64,
        )
        exposure_rows.extend(
            {
                "date": day.isoformat(),
                "feature": name,
                "spearman": _finite_or_none(daily[index]),
            }
            for index, day in enumerate(inputs.dates)
        )
        finite = daily[np.isfinite(daily)]
        interval = {
            "estimate": _finite_or_none(_finite_mean(daily)),
            "lower_95": None,
            "upper_95": None,
        }
        if finite.size >= 20:
            interval = _bootstrap_payload(
                daily, replications=10_000, block_length=min(20, len(daily))
            )
        exposure_summary.append({"feature": name, **interval})

    matched_rows = []
    matched_daily_rows = []
    sigma = np.asarray(inputs.target_scale_sigma, dtype=np.float64)
    all_horizon_outcome = (
        active
        & np.isfinite(sigma)
        & (sigma > 1e-8)
        & target_mask.all(axis=-1)
        & np.isfinite(scaled).all(axis=-1)
    )
    all_horizon_population = (
        all_horizon_outcome & masks.all(axis=-1) & np.isfinite(scores).all(axis=-1)
    )
    matched_values = np.full(
        (len(inputs.dates), len(HORIZONS)), np.nan, dtype=np.float64
    )
    for day, day_value in enumerate(inputs.dates):
        for horizon_index, horizon in enumerate(HORIZONS):
            value, count, reason = _spearman_result(
                scores[day, :, horizon_index],
                scaled[day, :, horizon_index],
                all_horizon_population[day],
            )
            matched_values[day, horizon_index] = value
            matched_daily_rows.append(
                {
                    "date": day_value.isoformat(),
                    "horizon_sessions": horizon,
                    "all_five_common_outcome_name_count": int(
                        all_horizon_outcome[day].sum()
                    ),
                    "all_five_common_score_and_outcome_name_count": count,
                    "scaled_target_spearman_ic": _finite_or_none(value),
                    "undefined_reason": reason,
                }
            )
    for horizon_index, horizon in enumerate(HORIZONS):
        values = matched_values[:, horizon_index]
        defined = np.isfinite(values)
        matched_rows.append(
            {
                "horizon_sessions": horizon,
                "population": "common_D1_D2_D3_D5_D10",
                "possible_date_count": int(
                    (all_horizon_outcome.sum(axis=1) >= MIN_CROSS_SECTION).sum()
                ),
                "used_date_count": int(defined.sum()),
                "mean_scaled_target_ic": _finite_or_none(_finite_mean(values)),
                "undefined_reason": (
                    None
                    if defined.any()
                    else "no_date_has_a_defined_all_five_horizon_ic"
                ),
            }
        )

    shareholder_returns = np.asarray(
        inputs.shareholder_simple_returns, dtype=np.float64
    )
    shareholder_valid = np.asarray(inputs.shareholder_target_mask, dtype=np.bool_)
    intervals = ((0, 1), (1, 3), (3, 5), (5, 10))
    incremental_rows = []
    for head_index, head_horizon in enumerate(HORIZONS):
        for start, stop in intervals:
            stop_index = HORIZONS.index(stop)
            start_index = HORIZONS.index(start) if start else None
            daily = []
            for day in range(len(inputs.dates)):
                stop_wealth = 1.0 + shareholder_returns[day, :, stop_index]
                if start_index is None:
                    start_wealth = np.ones_like(stop_wealth)
                    interval_valid = shareholder_valid[day, :, stop_index].copy()
                else:
                    start_wealth = 1.0 + shareholder_returns[day, :, start_index]
                    interval_valid = (
                        shareholder_valid[day, :, start_index]
                        & shareholder_valid[day, :, stop_index]
                    )
                interval_valid &= (
                    np.isfinite(start_wealth)
                    & (start_wealth > 0.0)
                    & np.isfinite(stop_wealth)
                )
                realized = np.full_like(stop_wealth, np.nan)
                realized[interval_valid] = (
                    stop_wealth[interval_valid] / start_wealth[interval_valid] - 1.0
                )
                valid = (
                    masks[day, :, head_index]
                    & active[day]
                    & interval_valid
                    & np.isfinite(sigma[day])
                    & (sigma[day] > 1e-8)
                    & np.isfinite(scores[day, :, head_index])
                )
                if valid.any():
                    realized[valid] = (realized[valid] - np.median(realized[valid])) / (
                        sigma[day, valid] * np.sqrt(stop - start)
                    )
                daily.append(_spearman(scores[day, :, head_index], realized, valid))
            incremental_rows.append(
                {
                    "head_horizon_sessions": head_horizon,
                    "increment_start_sessions": start,
                    "increment_end_sessions": stop,
                    "return_basis": "shareholder_wealth_same_holding",
                    "scale_basis": "fixed_decision_time_target_scale_sigma",
                    "mean_spearman_ic": _finite_or_none(
                        _finite_mean(np.asarray(daily))
                    ),
                }
            )
    return {
        "exposure_daily": exposure_rows,
        "exposure_summary": exposure_summary,
        "incremental_horizon_ic": incremental_rows,
        "matched_universe_ic": matched_rows,
        "matched_universe_daily": matched_daily_rows,
        "realized_beta": (
            _realized_beta_diagnostic(inputs, headline)
            if headline is not None
            else {"status": "unsupported", "classification": "unsupported"}
        ),
        "lending_coverage": _lending_coverage(inputs),
    }


def evaluate_scores(
    inputs: EvaluationInputs,
    *,
    window_name: str,
    registration_path: Path | None = None,
    preregistration_root: Path = PREREGISTRATION_ROOT,
) -> EvaluationResult:
    """Evaluate one score cube after enforcing the v2 access boundary."""
    ledger = authorize_dates(
        inputs.dates,
        purpose="evaluation",
        registration_path=registration_path,
        preregistration_root=preregistration_root,
    )
    if ledger.official_validation_accessed and not inputs.transfer_chronology_clean:
        raise PermissionError(
            "official validation refuses an artifact with contaminated transfer "
            "chronology"
        )
    _validate(inputs)
    if not window_name:
        raise ValueError("window_name must be nonempty")
    (
        primary_scores,
        primary_targets,
        primary_outcome_mask,
        primary_score_mask,
    ) = _primary_population_components(inputs)
    primary_population = primary_outcome_mask & primary_score_mask
    _, daily_primary, primary_rows = _primary_daily_metrics(
        primary_scores,
        primary_targets,
        primary_outcome_mask,
        primary_score_mask,
        inputs.dates,
    )
    (
        neutral_ic,
        legacy_scaled_ic,
        shareholder_ic,
        price_ic,
        spread_total_bps,
        metric_rows,
    ) = _daily_metrics(inputs, primary_population)
    persistence, persistence_rows = _persistence(inputs)
    economics_score, economics_mask = _economics_signal(inputs)
    action_terms = _aligned_action_terms(inputs)
    grid = ledger_sensitivity_grid(
        dates=inputs.dates,
        scores=economics_score,
        score_mask=economics_mask,
        active=np.asarray(inputs.active, dtype=np.bool_),
        raw_close=inputs.raw_close,
        action_terms=action_terms,
        action_payment_session=inputs.action_payment_session,
        cdi_returns=inputs.cdi_returns,
        security_ids=inputs.security_ids,
        initial_reference_price=inputs.initial_reference_price,
        annual_borrow_rate_by_name=inputs.annual_borrow_rate_by_name,
        shortable=inputs.shortable,
    )
    configurations = ledger_configurations()
    headline_name = f"cost_{ECONOMICS_HEADLINE[0]:g}_borrow_{ECONOMICS_HEADLINE[1]:g}"
    headline = grid[headline_name]
    d5_index = HORIZONS.index(5)
    d5_score = np.asarray(inputs.scores, dtype=np.float64)[..., d5_index]
    d5_score_mask = (
        np.asarray(inputs.score_mask, dtype=np.bool_)[..., d5_index]
        & np.asarray(inputs.active, dtype=np.bool_)
        & np.isfinite(d5_score)
    )
    d5_only = simulate_stateful_ledger(
        dates=inputs.dates,
        scores=d5_score,
        score_mask=d5_score_mask,
        active=np.asarray(inputs.active, dtype=np.bool_),
        raw_close=inputs.raw_close,
        action_terms=action_terms,
        action_payment_session=inputs.action_payment_session,
        cdi_returns=inputs.cdi_returns,
        security_ids=inputs.security_ids,
        config=LedgerConfig(),
        initial_reference_price=inputs.initial_reference_price,
    )
    active = np.asarray(inputs.active, dtype=np.bool_)
    scaled_mask = np.asarray(inputs.scaled_target_mask, dtype=np.bool_)
    neutral_mask = np.asarray(inputs.neutral_target_mask, dtype=np.bool_)
    shareholder_mask = np.asarray(inputs.shareholder_target_mask, dtype=np.bool_)
    price_mask = np.asarray(inputs.price_target_mask, dtype=np.bool_)
    horizon_rows: list[dict[str, object]] = []
    for horizon_index, horizon in enumerate(HORIZONS):
        if horizon in PRIMARY_HORIZONS:
            scaled_possible = primary_outcome_mask
        else:
            scaled_possible = (
                active
                & neutral_mask[..., horizon_index]
                & np.isfinite(
                    np.asarray(inputs.neutral_midrank_targets)[..., horizon_index]
                )
            )
        shareholder_possible = (
            active
            & shareholder_mask[..., horizon_index]
            & np.isfinite(
                np.asarray(inputs.shareholder_midrank_targets)[..., horizon_index]
            )
        )
        price_possible = (
            active
            & price_mask[..., horizon_index]
            & np.isfinite(np.asarray(inputs.price_midrank_targets)[..., horizon_index])
        )
        horizon_rows.append(
            {
                "horizon_sessions": horizon,
                "primary_horizon": horizon in PRIMARY_HORIZONS,
                "mean_neutral_target_spearman_ic": _finite_or_none(
                    _finite_mean(neutral_ic[:, horizon_index])
                ),
                "neutral_target_possible_date_count": int(
                    (scaled_possible.sum(axis=1) >= MIN_CROSS_SECTION).sum()
                ),
                "neutral_target_used_date_count": int(
                    np.isfinite(neutral_ic[:, horizon_index]).sum()
                ),
                "neutral_target_possible_name_days": int(scaled_possible.sum()),
                "legacy_scaled_target_ic": _finite_or_none(
                    _finite_mean(legacy_scaled_ic[:, horizon_index])
                ),
                "mean_shareholder_rank_ic": _finite_or_none(
                    _finite_mean(shareholder_ic[:, horizon_index])
                ),
                "shareholder_rank_possible_date_count": int(
                    (shareholder_possible.sum(axis=1) >= MIN_CROSS_SECTION).sum()
                ),
                "shareholder_rank_used_date_count": int(
                    np.isfinite(shareholder_ic[:, horizon_index]).sum()
                ),
                "shareholder_rank_possible_name_days": int(shareholder_possible.sum()),
                "mean_price_return_rank_ic": _finite_or_none(
                    _finite_mean(price_ic[:, horizon_index])
                ),
                "price_return_rank_possible_date_count": int(
                    (price_possible.sum(axis=1) >= MIN_CROSS_SECTION).sum()
                ),
                "price_return_rank_used_date_count": int(
                    np.isfinite(price_ic[:, horizon_index]).sum()
                ),
                "price_return_rank_possible_name_days": int(price_possible.sum()),
                "mean_shareholder_return_spread_total_bps": _finite_or_none(
                    _finite_mean(spread_total_bps[:, horizon_index])
                ),
                "mean_shareholder_return_spread_bps_per_holding_session": (
                    _finite_or_none(
                        _finite_mean(spread_total_bps[:, horizon_index]) / horizon
                    )
                ),
                "mean_persistence_1_session": _finite_or_none(
                    _finite_mean(persistence[1][:, horizon_index])
                ),
                "mean_persistence_5_sessions": _finite_or_none(
                    _finite_mean(persistence[5][:, horizon_index])
                ),
            }
        )
    economics_summaries: list[dict[str, object]] = []
    economics_daily: list[dict[str, object]] = []
    for scenario, result in grid.items():
        config = configurations[scenario]
        economics_summaries.append(
            {
                "scenario": scenario,
                "cost_bps_per_side": config.cost_bps_per_side,
                "annual_borrow_rate": config.annual_borrow_rate,
                "borrow_source": config.borrow_source,
                "buffer_per_side": config.buffer_per_side,
                "short_proceeds_remuneration": (config.short_proceeds_remuneration),
                "terminal_settlement_convention": TERMINAL_SETTLEMENT_CONVENTION,
                "settlement_grace_sessions": config.settlement_grace_sessions,
                "settlement_haircut": config.settlement_haircut,
                "path_model_count": 1,
                **{
                    key: _finite_or_none(value) if isinstance(value, float) else value
                    for key, value in result.summary().items()
                },
            }
        )
        if scenario.startswith("cost_"):
            economics_daily.extend(
                {
                    **row,
                    "scenario": scenario,
                }
                for row in _ledger_rows(
                    result,
                    cost_bps=config.cost_bps_per_side,
                    annual_borrow_rate=config.annual_borrow_rate,
                )
            )
    headline_daily = _ledger_rows(
        headline,
        cost_bps=ECONOMICS_HEADLINE[0],
        annual_borrow_rate=ECONOMICS_HEADLINE[1],
    )
    holding_audit = _holding_audit(headline, inputs.security_ids)
    action_attribution = _action_attribution(inputs, headline)
    deployed_net = np.asarray(
        [
            float(row["deployed_net_fraction_nav"])
            if row["deployed_net_fraction_nav"] is not None
            else math.nan
            for row in headline_daily
        ],
        dtype=np.float64,
    )
    quality_stratification = _quality_stratification(
        inputs,
        primary_scores,
        primary_targets,
        primary_outcome_mask,
        primary_score_mask,
    )
    report: dict[str, object] = {
        "schema": EVALUATION_SCHEMA,
        "window": {
            "name": window_name,
            "start": inputs.dates[0].isoformat(),
            "end": inputs.dates[-1].isoformat(),
            "date_count": len(inputs.dates),
            "date_identity_sha256": _dates_sha256(inputs.dates),
            "canonical_calendar_sha256": inputs.calendar_identity_sha256,
            "first_session_index": int(np.asarray(inputs.session_indices)[0]),
            "last_session_index": int(np.asarray(inputs.session_indices)[-1]),
        },
        "access": ledger.payload(),
        "official_validation_accessed": ledger.official_validation_accessed,
        "test_accessed": ledger.test_accessed,
        "transfer_chronology_clean": inputs.transfer_chronology_clean,
        "action_terms_source": inputs.action_terms_source,
        "schedule_source": inputs.schedule_source,
        "horizons_sessions": list(HORIZONS),
        "primary_horizons_sessions": list(PRIMARY_HORIZONS),
        "metric_contract": {
            "primary_target": REGISTERED_PRIMARY_TARGET,
            "primary_population": (
                "per date: active entry names with finite sigma>1e-8 and valid, "
                "finite score and neutral outcome on every D1/D2/D3/D5 head"
            ),
            "minimum_names": MIN_CROSS_SECTION,
            "daily_primary_aggregation": (
                "equal mean of all four head Spearman correlations only when "
                "every head is defined"
            ),
            "shareholder_return": "gross contractual holding simple return",
            "price_return": "contractually unit-adjusted price-only return",
        },
        "mean_daily_primary_neutral_target_ic": _finite_or_none(
            _finite_mean(daily_primary)
        ),
        "annual_borrow_rate_by_name": _array_sha256(
            np.asarray(
                inputs.annual_borrow_rate_by_name
                if inputs.annual_borrow_rate_by_name is not None
                else "unsupported"
            )
        ),
        "shortable": _array_sha256(
            np.asarray(
                inputs.shortable if inputs.shortable is not None else "unsupported"
            )
        ),
        "primary_support": {
            "possible_date_count": int(
                (primary_outcome_mask.sum(axis=1) >= MIN_CROSS_SECTION).sum()
            ),
            "used_date_count": int(np.isfinite(daily_primary).sum()),
            "possible_name_days": int(primary_outcome_mask.sum()),
            "used_name_days": int(primary_population.sum()),
            "score_support_loss_name_days": int(
                (primary_outcome_mask & ~primary_score_mask).sum()
            ),
            "undefined_reason": (
                None
                if np.isfinite(daily_primary).any()
                else "no_date_has_a_defined_four_head_primary_ic"
            ),
        },
        "daily_primary_ic": primary_rows,
        "horizon_readouts": horizon_rows,
        "declared_subperiod_readouts": _declared_subperiod_readouts(
            inputs,
            neutral_ic,
            shareholder_ic,
            price_ic,
            spread_total_bps,
            window_name=window_name,
        ),
        "daily_metric_table": metric_rows,
        "persistence_table": persistence_rows,
        "economics": {
            "contract": _economics_contract(inputs),
            "headline": {
                "scenario": headline_name,
                **{
                    key: _finite_or_none(value) if isinstance(value, float) else value
                    for key, value in headline.summary().items()
                },
                "mean_deployed_net_fraction_nav": _finite_or_none(
                    _finite_mean(deployed_net)
                ),
                "terminal_unresolved_inventory_fraction_nav": (
                    _finite_or_none(
                        headline.unresolved_inventory_notional / headline.nav[-1]
                    )
                    if headline.nav[-1] != 0.0
                    else None
                ),
            },
            "summaries": economics_summaries,
            "daily_table": economics_daily,
            "headline_audit": {
                "daily_state": headline_daily,
                "intended_orders": _serialise_records(headline.intended_orders),
                "fills": _serialise_records(headline.fills),
                "cancellations": _serialise_records(headline.cancellations),
                "holding_age_distribution": holding_audit,
                "claims_and_action_attribution": action_attribution,
            },
            "d5_only_diagnostic": {
                "horizon_sessions": 5,
                "contract": "D5 score head with the exact headline ledger settings",
                "summary": {
                    key: _finite_or_none(value) if isinstance(value, float) else value
                    for key, value in d5_only.summary().items()
                },
                "daily_table": _ledger_rows(
                    d5_only,
                    cost_bps=LedgerConfig().cost_bps_per_side,
                    annual_borrow_rate=LedgerConfig().annual_borrow_rate,
                ),
            },
            "coverage": {
                "possible_date_count": len(inputs.dates),
                "reported_date_count": len(headline.dates),
                "finite_net_excess_date_count": int(
                    np.isfinite(headline.net_excess_all_cash_bps).sum()
                ),
                "score_supported_date_count": int(economics_mask.any(axis=1).sum()),
                "score_supported_name_days": int(economics_mask.sum()),
                "deployed_date_count": int((headline.gross_fraction_nav > 0.0).sum()),
                "all_cash_date_count": int((headline.gross_fraction_nav == 0.0).sum()),
                "economics_unresolved": headline.economics_unresolved,
            },
        },
        "diagnostics": _diagnostics(inputs, headline),
        "quality_and_coverage_stratification": quality_stratification,
        "input_hashes": _input_hashes(inputs),
        "source_artifact_hashes": dict(
            sorted((inputs.source_artifact_hashes or {}).items())
        ),
        "mask_coverage": {
            "active_name_days": int(active.sum()),
            "score_mask_true": int(np.asarray(inputs.score_mask).sum()),
            "neutral_target_mask_true": int(neutral_mask.sum()),
            "legacy_scaled_target_mask_true": int(scaled_mask.sum()),
            "shareholder_target_mask_true": int(shareholder_mask.sum()),
            "price_target_mask_true": int(price_mask.sum()),
            "primary_common_outcome_name_days": int(primary_outcome_mask.sum()),
            "primary_common_score_and_outcome_name_days": int(primary_population.sum()),
            "economics_score_mask_true": int(economics_mask.sum()),
            "economics_path_model_count": 1,
            "raw_close_present_name_days": int(
                np.isfinite(np.asarray(inputs.raw_close, dtype=np.float64)).sum()
            ),
            "action_name_days": int(
                np.asarray(inputs.action_has_action, dtype=np.bool_).sum()
            ),
            "unresolved_action_source_name_days": int(
                (
                    np.asarray(inputs.action_has_action, dtype=np.bool_)
                    & ~np.asarray(inputs.action_session_resolved, dtype=np.bool_)
                ).sum()
            ),
            "known_action_payment_name_days": int(
                (np.asarray(inputs.action_payment_session) >= 0).sum()
            ),
            "stale_mark_name_days": int(headline.stale_mark_name_days.sum()),
            "unresolved_action_name_days": int(
                headline.unresolved_action_name_days.sum()
            ),
            "valuation_scenario_count": int(headline.valuation_scenario_count.sum()),
            "actual_risk_breach_dates": int(headline.actual_risk_breach.sum()),
        },
    }
    return EvaluationResult(
        report=report,
        dates=inputs.dates,
        daily_primary_ic=daily_primary,
        headline_economics_dates=headline.dates,
        headline_net_excess_bps=headline.net_excess_all_cash_bps,
        primary_scores=primary_scores,
        primary_targets=primary_targets,
        primary_outcome_mask=primary_outcome_mask,
        primary_score_mask=primary_score_mask,
    )


def _bootstrap_payload(
    values: NDArray[np.floating],
    *,
    replications: int,
    block_length: int,
) -> dict[str, float | int | str | None]:
    finite_count = int(np.isfinite(np.asarray(values, dtype=np.float64)).sum())
    support = {
        "possible_date_count": int(len(values)),
        "defined_date_count": finite_count,
    }
    if finite_count == 0:
        return {
            "estimate": None,
            "lower_95": None,
            "upper_95": None,
            **support,
            "undefined_reason": "no_defined_daily_values",
        }
    if replications == 0:
        return {
            "estimate": _finite_or_none(_finite_mean(values)),
            "lower_95": None,
            "upper_95": None,
            **support,
            "undefined_reason": None,
        }
    output = moving_block_bootstrap(
        values,
        replications=replications,
        block_length=block_length,
        seed=BOOTSTRAP_SEED,
    )
    return (
        {
            key: _finite_or_none(np.asarray(value).reshape(-1)[0])
            for key, value in output.items()
        }
        | support
        | {"undefined_reason": None}
    )


def _paired_primary_daily(
    candidate: EvaluationResult,
    baseline: EvaluationResult,
) -> tuple[NDArray[np.float64], list[dict[str, object]]]:
    expected_scores = (
        len(candidate.dates),
        candidate.primary_outcome_mask.shape[1],
        len(PRIMARY_HORIZONS),
    )
    for label, result in (("candidate", candidate), ("baseline", baseline)):
        if result.primary_scores.shape != expected_scores:
            raise ValueError(f"{label} retained primary scores are misaligned")
        if result.primary_targets.shape != expected_scores:
            raise ValueError(f"{label} retained primary targets are misaligned")
        if result.primary_outcome_mask.shape != expected_scores[:2]:
            raise ValueError(f"{label} retained outcome population is misaligned")
        if result.primary_score_mask.shape != expected_scores[:2]:
            raise ValueError(f"{label} retained score population is misaligned")
    common_outcome = candidate.primary_outcome_mask & baseline.primary_outcome_mask
    common_score = candidate.primary_score_mask & baseline.primary_score_mask
    candidate_heads, candidate_daily, candidate_rows = _primary_daily_metrics(
        candidate.primary_scores,
        candidate.primary_targets,
        common_outcome,
        common_score,
        candidate.dates,
    )
    baseline_heads, baseline_daily, baseline_rows = _primary_daily_metrics(
        baseline.primary_scores,
        baseline.primary_targets,
        common_outcome,
        common_score,
        baseline.dates,
    )
    delta = candidate_daily - baseline_daily
    rows: list[dict[str, object]] = []
    for day, day_value in enumerate(candidate.dates):
        candidate_reason = candidate_rows[day]["undefined_reason"]
        baseline_reason = baseline_rows[day]["undefined_reason"]
        reason = None
        if candidate_reason is not None or baseline_reason is not None:
            reason = (
                f"candidate={candidate_reason or 'defined'};"
                f"baseline={baseline_reason or 'defined'}"
            )
        rows.append(
            {
                "date": day_value.isoformat(),
                "common_outcome_name_count": int(common_outcome[day].sum()),
                "common_candidate_baseline_name_count": int(
                    (common_outcome[day] & common_score[day]).sum()
                ),
                "candidate_head_neutral_target_spearman_ic": {
                    f"D{horizon}": _finite_or_none(candidate_heads[day, index])
                    for index, horizon in enumerate(PRIMARY_HORIZONS)
                },
                "baseline_head_neutral_target_spearman_ic": {
                    f"D{horizon}": _finite_or_none(baseline_heads[day, index])
                    for index, horizon in enumerate(PRIMARY_HORIZONS)
                },
                "candidate_primary_neutral_target_ic": _finite_or_none(
                    candidate_daily[day]
                ),
                "baseline_primary_neutral_target_ic": _finite_or_none(
                    baseline_daily[day]
                ),
                "delta": _finite_or_none(delta[day]),
                "undefined_reason": reason,
            }
        )
    return delta, rows


def paired_comparison(
    candidate: EvaluationResult,
    baseline: EvaluationResult,
    *,
    registration_path: Path | None = None,
    preregistration_root: Path = PREREGISTRATION_ROOT,
    protocol: ProtocolPreset = FULL_PROTOCOL,
) -> dict[str, object]:
    """Compare aligned daily metrics without estimating on another window."""
    ledger = authorize_dates(
        candidate.dates,
        purpose="evaluation",
        registration_path=registration_path,
        preregistration_root=preregistration_root,
    )
    if ledger.official_validation_accessed:
        for label, result in (("candidate", candidate), ("baseline", baseline)):
            if result.report.get("transfer_chronology_clean") is not True:
                raise PermissionError(
                    "official validation refuses a paired comparison whose "
                    f"{label} has contaminated transfer chronology"
                )
    if candidate.dates != baseline.dates:
        raise ValueError("paired IC comparison requires identical date axes")
    if candidate.headline_economics_dates != baseline.headline_economics_dates:
        raise ValueError("paired economics comparison requires identical date axes")
    _validate_paired_identity(candidate.report, baseline.report)
    if candidate.headline_net_excess_bps.shape != (
        len(candidate.headline_economics_dates),
    ) or baseline.headline_net_excess_bps.shape != (
        len(baseline.headline_economics_dates),
    ):
        raise ValueError("paired economics arrays differ from their date axes")
    ic_delta, paired_population_rows = _paired_primary_daily(candidate, baseline)
    candidate_unresolved = _headline_economics_unresolved(candidate.report)
    baseline_unresolved = _headline_economics_unresolved(baseline.report)
    economics_undefined_reason = None
    if candidate_unresolved or baseline_unresolved:
        economics_delta = np.full(
            len(candidate.headline_economics_dates), np.nan, dtype=np.float64
        )
        labels = []
        if candidate_unresolved:
            labels.append("candidate")
        if baseline_unresolved:
            labels.append("baseline")
        economics_undefined_reason = (
            f"{'_and_'.join(labels)}_headline_economics_unresolved"
        )
    else:
        economics_delta = (
            candidate.headline_net_excess_bps - baseline.headline_net_excess_bps
        )
    if (
        protocol.bootstrap_replications > 0
        and min(len(ic_delta), len(economics_delta)) < protocol.bootstrap_block_length
    ):
        raise ValueError(
            "paired comparison requires at least the preset bootstrap block length"
        )
    return {
        "schema": PAIRED_COMPARISON_SCHEMA,
        "access": ledger.payload(),
        "official_validation_accessed": ledger.official_validation_accessed,
        "test_accessed": ledger.test_accessed,
        "date_identity_sha256": _dates_sha256(candidate.dates),
        "protocol": protocol.name,
        "block_length_sessions": protocol.bootstrap_block_length,
        "replications": protocol.bootstrap_replications,
        "seed": BOOTSTRAP_SEED,
        "daily_primary_ic_delta": _bootstrap_payload(
            ic_delta,
            replications=protocol.bootstrap_replications,
            block_length=protocol.bootstrap_block_length,
        ),
        "primary_population": {
            "definition": (
                "per-date intersection of candidate and baseline finite score "
                "support with active, sigma-valid outcomes on D1/D2/D3/D5"
            ),
            "minimum_names": MIN_CROSS_SECTION,
            "possible_date_count": len(candidate.dates),
            "used_date_count": int(np.isfinite(ic_delta).sum()),
        },
        "daily_headline_net_excess_bps_delta": _bootstrap_payload(
            economics_delta,
            replications=protocol.bootstrap_replications,
            block_length=protocol.bootstrap_block_length,
        ),
        "economics_comparison_undefined_reason": economics_undefined_reason,
        "daily_primary_ic_delta_table": paired_population_rows,
        "daily_headline_net_excess_bps_delta_table": [
            {
                "date": value.isoformat(),
                "delta": _finite_or_none(economics_delta[index]),
                "undefined_reason": economics_undefined_reason,
            }
            for index, value in enumerate(candidate.headline_economics_dates)
        ],
    }


def _headline_economics_unresolved(report: Mapping[str, object]) -> bool:
    economics = report.get("economics")
    if not isinstance(economics, Mapping):
        raise ValueError("evaluation report lacks economics")
    headline = economics.get("headline")
    if not isinstance(headline, Mapping):
        raise ValueError("evaluation report lacks headline economics")
    value = headline.get("economics_unresolved")
    if not isinstance(value, bool):
        raise ValueError(
            "evaluation report lacks an explicit economics resolution flag"
        )
    return value


_PAIRED_INPUT_KEYS = (
    "dates",
    "canonical_calendar",
    "session_indices",
    "scaled_midrank_targets",
    "scaled_target_mask",
    "shareholder_midrank_targets",
    "shareholder_simple_returns",
    "shareholder_target_mask",
    "price_midrank_targets",
    "price_target_mask",
    "active",
    "raw_close",
    "action_shares_per_prior_share",
    "action_cash_per_prior_share",
    "action_session_resolved",
    "action_has_action",
    "action_successor_index",
    "action_payment_session",
    "security_ids",
    "target_scale_sigma",
    "cdi_returns",
    "history_age_sessions",
    "source_archive_present",
    "source_feature_valid",
    "initial_reference_price",
    "eventual_survives_to_final_year",
    "prior_feature_beta_60",
    "prior_feature_log_return_5",
    "prior_feature_log_volume_mean_20",
    "prior_feature_momentum_12_1",
    "prior_feature_yang_zhang_vol_20",
)


def _validate_paired_identity(
    candidate_report: Mapping[str, object],
    baseline_report: Mapping[str, object],
) -> None:
    """Require both reports to describe the same paired evaluation population."""
    for label, report in (
        ("candidate", candidate_report),
        ("baseline", baseline_report),
    ):
        if report.get("schema") != EVALUATION_SCHEMA:
            raise ValueError(
                f"{label} evaluation report is stale or incompatible; "
                f"expected {EVALUATION_SCHEMA}"
            )
        input_hashes = report.get("input_hashes")
        if not isinstance(input_hashes, Mapping) or any(
            not isinstance(input_hashes.get(key), str) for key in _PAIRED_INPUT_KEYS
        ):
            raise ValueError(f"{label} report lacks paired input identities")
        for key in _PAIRED_INPUT_KEYS:
            digest = input_hashes[key]
            assert isinstance(digest, str)
            _validate_sha256(digest, label=f"{label} {key} identity")
        source_hashes = report.get("source_artifact_hashes")
        if not isinstance(source_hashes, Mapping) or not source_hashes:
            raise ValueError(f"{label} report lacks source artifact identities")
        if any(
            not isinstance(name, str) or not name or not isinstance(digest, str)
            for name, digest in source_hashes.items()
        ):
            raise ValueError(f"{label} report has malformed source identities")
        for digest in source_hashes.values():
            assert isinstance(digest, str)
            _validate_sha256(digest, label=f"{label} source identity")
        economics = report.get("economics")
        if not isinstance(economics, Mapping) or not isinstance(
            economics.get("contract"), Mapping
        ):
            raise ValueError(f"{label} report lacks the economics contract")
    candidate_inputs = candidate_report["input_hashes"]
    baseline_inputs = baseline_report["input_hashes"]
    assert isinstance(candidate_inputs, Mapping)
    assert isinstance(baseline_inputs, Mapping)
    mismatched = [
        key
        for key in _PAIRED_INPUT_KEYS
        if candidate_inputs[key] != baseline_inputs[key]
    ]
    if mismatched:
        raise ValueError(
            "paired comparison requires identical dates, outcome targets/masks, "
            "raw closes, contractual actions/payment sessions, security identities, "
            "target scales, and CDI inputs; "
            "mismatched identities: "
            f"{mismatched}"
        )
    if (
        candidate_report["source_artifact_hashes"]
        != baseline_report["source_artifact_hashes"]
    ):
        raise ValueError("paired comparison requires identical source identities")
    candidate_economics = candidate_report["economics"]
    baseline_economics = baseline_report["economics"]
    assert isinstance(candidate_economics, Mapping)
    assert isinstance(baseline_economics, Mapping)
    if candidate_economics["contract"] != baseline_economics["contract"]:
        raise ValueError("paired comparison requires an identical economics contract")


def write_evaluation_report(path: Path, result: EvaluationResult) -> str:
    return write_json_atomic(path, result.report)
