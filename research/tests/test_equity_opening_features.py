from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from brazil_rv.preprocessing.audit import opening_feature_family_stats
from brazil_rv.preprocessing.build import _sample_date_is_eligible
from brazil_rv.preprocessing.contract import (
    DECISION_EQUITY_INDICES,
    EQUITY_SESSION_MINUTES,
    HORIZONS,
)
from brazil_rv.preprocessing.transforms import (
    _completed_session_open,
    add_slow_cross_sectional_ranks,
    build_causal_features,
    build_dynamic_features,
    build_equity_features,
    build_raw_returns,
    center_cross_section,
    centered_midranks,
)


def _synthetic_equity_grid(days: int = 21) -> tuple[np.ndarray, np.ndarray]:
    minute = np.arange(EQUITY_SESSION_MINUTES, dtype=np.float64)
    raw = np.zeros((days, EQUITY_SESSION_MINUTES, 5), dtype=np.float64)
    for day_idx in range(days):
        base = 80.0 + 0.05 * day_idx
        close = base * np.exp(0.0002 * minute + 0.001 * np.sin(minute / 11.0 + day_idx))
        open_ = close * np.exp(0.0001 * np.cos(minute / 7.0 + day_idx))
        raw[day_idx, :, 0] = open_
        raw[day_idx, :, 1] = np.maximum(open_, close) * 1.0002
        raw[day_idx, :, 2] = np.minimum(open_, close) * 0.9998
        raw[day_idx, :, 3] = close
        raw[day_idx, :, 4] = np.exp(
            6.0 + 0.035 * day_idx + 0.15 * np.sin(minute / 19.0 + day_idx)
        )
    return raw, np.ones(raw.shape[:2], dtype=bool)


def _market_dates(days: int) -> tuple[date, ...]:
    return tuple(date(2024, 1, 2) + timedelta(days=index) for index in range(days))


def test_completed_session_uses_first_observed_open_and_empty_session_is_invalid() -> (
    None
):
    raw, observed = _synthetic_equity_grid()
    observed[19, :7] = False
    result = build_equity_features(
        raw,
        observed,
        np.ones(raw.shape[0], dtype=bool),
        market_dates=_market_dates(raw.shape[0]),
    )
    expected = np.log(raw[19, -1, 3] / raw[19, 7, 0]) / (
        result.sigma[20] * np.sqrt(EQUITY_SESSION_MINUTES)
    )
    np.testing.assert_allclose(result.slow[20, 3], np.clip(expected, -10.0, 10.0))

    session_open, valid = _completed_session_open(
        raw[19], np.zeros(EQUITY_SESSION_MINUTES, dtype=bool)
    )
    assert session_open == 0.0
    assert not valid


def test_daily_gap_uses_first_early_bar_and_never_backfills_from_cutoff() -> None:
    raw, observed = _synthetic_equity_grid()
    cutoff = DECISION_EQUITY_INDICES[0]
    observed[20, :7] = False
    baseline = build_equity_features(raw, observed, np.ones(21, dtype=bool))
    expected = np.log(raw[20, 7, 0] / raw[19, -1, 3]) / (
        baseline.sigma[20] * np.sqrt(EQUITY_SESSION_MINUTES)
    )
    np.testing.assert_allclose(baseline.slow[20, 1], np.clip(expected, -10.0, 10.0))
    assert baseline.slow_rank_valid[20, 0]

    future_changed = raw.copy()
    future_changed[20, cutoff:, :4] *= 1.9
    future_changed[20, cutoff:, 4] *= 50.0
    changed = build_equity_features(future_changed, observed, np.ones(21, dtype=bool))
    np.testing.assert_array_equal(baseline.slow[20], changed.slow[20])
    np.testing.assert_array_equal(
        baseline.slow_rank_valid[20], changed.slow_rank_valid[20]
    )
    offsets = np.linspace(-0.5, 0.5, 30, dtype=np.float32)
    baseline_slow = np.repeat(baseline.slow[20][None], 30, axis=0)
    changed_slow = np.repeat(changed.slow[20][None], 30, axis=0)
    baseline_slow[:, 1] += offsets
    changed_slow[:, 1] += offsets
    baseline_validity = np.repeat(baseline.slow_rank_valid[20][None], 30, axis=0)
    changed_validity = np.repeat(changed.slow_rank_valid[20][None], 30, axis=0)
    add_slow_cross_sectional_ranks(
        baseline_slow, baseline_validity, np.ones(30, dtype=bool)
    )
    add_slow_cross_sectional_ranks(
        changed_slow, changed_validity, np.ones(30, dtype=bool)
    )
    np.testing.assert_array_equal(baseline_slow[:, 17], changed_slow[:, 17])

    late_observed = observed.copy()
    late_observed[20, :cutoff] = False
    late = build_equity_features(raw, late_observed, np.ones(21, dtype=bool))
    assert late.slow[20, 1] == 0.0
    assert not late.slow_rank_valid[20, 0]


def test_dynamic_return_since_open_activates_at_delayed_first_observation() -> None:
    raw, observed = _synthetic_equity_grid()
    first_observed = DECISION_EQUITY_INDICES[0]
    observed[20, :first_observed] = False
    baseline = build_equity_features(raw, observed, np.ones(21, dtype=bool))
    assert not baseline.dynamic[20, :first_observed, 6].any()
    for position, elapsed in ((first_observed, 1), (first_observed + 5, 6)):
        expected = np.log(raw[20, position, 3] / raw[20, first_observed, 0]) / (
            baseline.sigma[20] * np.sqrt(elapsed)
        )
        np.testing.assert_allclose(
            baseline.dynamic[20, position, 6], np.clip(expected, -10.0, 10.0)
        )
    assert baseline.slow[20, 1] == 0.0

    changed_raw = raw.copy()
    changed_raw[20, first_observed + 10, :4] *= 2.0
    changed = build_equity_features(changed_raw, observed, np.ones(21, dtype=bool))
    np.testing.assert_array_equal(
        baseline.dynamic[20, : first_observed + 10, 6],
        changed.dynamic[20, : first_observed + 10, 6],
    )


def test_first_observed_open_preserves_mapping_change_boundaries() -> None:
    raw, observed = _synthetic_equity_grid(days=1)
    first_observed = 5
    observed[0, :first_observed] = False
    mapping_changed = np.zeros_like(observed)
    mapping_changed[0, 15] = True
    sigma = np.array([0.01], dtype=np.float64)
    dynamic, _ = build_dynamic_features(
        raw,
        observed,
        np.ones(1, dtype=bool),
        sigma,
        is_rate=False,
        mapping_changed=mapping_changed,
        first_observed_open=True,
    )

    assert not dynamic[0, :first_observed, 6].any()
    for position, elapsed in ((first_observed, 1), (first_observed + 5, 6)):
        expected = np.log(raw[0, position, 3] / raw[0, first_observed, 0]) / (
            sigma[0] * np.sqrt(elapsed)
        )
        np.testing.assert_allclose(
            dynamic[0, position, 6], np.clip(expected, -10.0, 10.0)
        )
    assert not dynamic[0, 15:, 6].any()


def test_gap_rank_uses_only_active_ready_equities_with_valid_early_opens() -> None:
    slow = np.zeros((31, 32), dtype=np.float32)
    validity = np.zeros((31, 3), dtype=bool)
    values = np.linspace(-1.0, 1.0, 30, dtype=np.float32)
    slow[:30, 1] = values
    slow[30, 1] = 1_000.0
    validity[:30, 0] = True
    active = np.ones(31, dtype=bool)
    add_slow_cross_sectional_ranks(slow, validity, active)

    np.testing.assert_allclose(slow[:30, 17], centered_midranks(values))
    assert np.isfinite(slow[:30, 17]).all()
    assert abs(float(slow[:30, 17].mean())) < 1e-7
    assert np.all(np.abs(slow[:30, 17]) < 1.0)
    assert slow[30, 17] == 0.0

    too_small = np.zeros((29, 32), dtype=np.float32)
    too_small[:, 1] = np.arange(29, dtype=np.float32)
    too_small_validity = np.zeros((29, 3), dtype=bool)
    too_small_validity[:, 0] = True
    add_slow_cross_sectional_ranks(
        too_small, too_small_validity, np.ones(29, dtype=bool)
    )
    assert not too_small[:, 17].any()


def test_missing_nominal_open_does_not_change_equity_or_sample_eligibility() -> None:
    raw, observed = _synthetic_equity_grid()
    delayed = observed.copy()
    delayed[20, 0] = False
    valid_day = np.ones(21, dtype=bool)
    baseline = build_equity_features(raw, observed, valid_day)
    changed = build_equity_features(raw, delayed, valid_day)
    np.testing.assert_array_equal(baseline.data_ready, changed.data_ready)

    baseline_returns, _ = build_raw_returns(raw, observed)
    changed_returns, _ = build_raw_returns(raw, delayed)
    np.testing.assert_array_equal(baseline_returns[20], changed_returns[20])
    entry = np.asarray(DECISION_EQUITY_INDICES)
    baseline_endpoints = np.stack(
        [observed[20, entry + horizon - 1] for horizon in HORIZONS], axis=1
    )
    changed_endpoints = np.stack(
        [delayed[20, entry + horizon - 1] for horizon in HORIZONS], axis=1
    )
    np.testing.assert_array_equal(baseline_endpoints, changed_endpoints)

    raw_cross_section = np.repeat(baseline_returns[20][None], 30, axis=0)
    baseline_candidate = np.broadcast_to(
        observed[20, entry, None] & baseline_endpoints,
        raw_cross_section.shape,
    ).copy()
    changed_candidate = np.broadcast_to(
        delayed[20, entry, None] & changed_endpoints,
        raw_cross_section.shape,
    ).copy()
    baseline_centered = center_cross_section(
        raw_cross_section, baseline_candidate, np.ones(30, dtype=np.float64)
    )
    changed_centered = center_cross_section(
        raw_cross_section, changed_candidate, np.ones(30, dtype=np.float64)
    )
    for expected, actual in zip(baseline_centered, changed_centered, strict=True):
        np.testing.assert_array_equal(expected, actual)
    assert _sample_date_is_eligible(30)


def _opening_audit_arrays(valid_count: int) -> dict[str, np.ndarray]:
    equity_count = max(valid_count + 1, 30)
    minute_count = max(DECISION_EQUITY_INDICES)
    dynamic = np.zeros((2, equity_count, minute_count, 26), dtype=np.float32)
    dynamic[1, :valid_count, 7, 5] = 1.0
    dynamic[1, :valid_count, 7, 6] = np.linspace(
        -0.5, 0.5, valid_count, dtype=np.float32
    )
    slow = np.zeros((2, equity_count, 32), dtype=np.float32)
    slow[1, :valid_count, 1] = np.linspace(-1.0, 1.0, valid_count, dtype=np.float32)
    slow[1, :valid_count, 3] = np.linspace(-0.25, 0.25, valid_count, dtype=np.float32)
    if valid_count >= 30:
        slow[1, :valid_count, 17] = centered_midranks(slow[1, :valid_count, 1])
    membership = np.zeros((2, equity_count), dtype=bool)
    data_ready = np.zeros_like(membership)
    membership[1, :valid_count] = True
    data_ready[1, :valid_count] = True
    return {
        "equity_features.npy": dynamic,
        "equity_slow.npy": slow,
        "equity_membership.npy": membership,
        "equity_data_ready.npy": data_ready,
    }


def _opening_rows(summary: dict[str, object]) -> dict[str, dict[str, object]]:
    return {row["feature"]: row for row in summary["features"]}


def test_opening_audit_accepts_warmup_boundary_and_reports_counts() -> None:
    arrays = _opening_audit_arrays(30)
    assert not arrays["equity_features.npy"][0, :, :, 5].any()
    summary = opening_feature_family_stats(arrays, np.array([1], dtype=np.int64))
    assert summary["equity_days_with_valid_early_open_proxy"] == 30
    assert summary["rank_decision_cross_sections_computed"] == len(
        DECISION_EQUITY_INDICES
    )
    assert summary["rank_decision_cross_sections_below_minimum_population"] == 0
    rows = _opening_rows(summary)
    assert rows["return_since_open_normalized"]["valid_observation_count"] == 30
    assert rows["overnight_gap_normalized"]["valid_observation_count"] == 30
    assert (
        rows["previous_open_to_close_return_normalized"]["valid_observation_count"]
        == 30
    )
    rank = rows["overnight_gap_cross_section_rank"]
    assert rank["valid_observation_count"] == 30
    assert rank["nonzero_count"] == 30
    assert rank["std"] > 0.0


def test_opening_audit_rejects_all_zero_rank_at_warmup_boundary() -> None:
    dead = _opening_audit_arrays(30)
    dead["equity_slow.npy"][..., 17] = 0.0
    with pytest.raises(ValueError, match="does not match centered midranks"):
        opening_feature_family_stats(dead, np.array([1], dtype=np.int64))


@pytest.mark.parametrize("corruption", ("permuted", "negated", "centered_scaled"))
def test_opening_audit_rejects_inexact_centered_ranks(corruption: str) -> None:
    arrays = _opening_audit_arrays(30)
    rank = arrays["equity_slow.npy"][1, :30, 17].copy()
    if corruption == "permuted":
        changed = np.roll(rank, 1)
    elif corruption == "negated":
        changed = -rank
    else:
        changed = 0.5 * rank
    arrays["equity_slow.npy"][1, :30, 17] = changed
    with pytest.raises(ValueError, match="does not match centered midranks"):
        opening_feature_family_stats(arrays, np.array([1], dtype=np.int64))


@pytest.mark.parametrize("active", (False, True), ids=("inactive", "active-invalid"))
def test_opening_audit_rejects_nonzero_rank_outside_valid_set(active: bool) -> None:
    arrays = _opening_audit_arrays(30)
    invalid_slot = arrays["equity_slow.npy"].shape[1] - 1
    if active:
        arrays["equity_membership.npy"][1, invalid_slot] = True
        arrays["equity_data_ready.npy"][1, invalid_slot] = True
    arrays["equity_slow.npy"][1, invalid_slot, 17] = 0.25
    with pytest.raises(ValueError, match="must be neutral outside valid"):
        opening_feature_family_stats(arrays, np.array([1], dtype=np.int64))


def test_opening_audit_accepts_subminimum_neutral_fallback_and_counts() -> None:
    tiny = opening_feature_family_stats(
        _opening_audit_arrays(29), np.array([1], dtype=np.int64)
    )
    assert tiny["equity_days_with_valid_early_open_proxy"] == 29
    assert tiny["rank_decision_cross_sections_computed"] == 0
    assert tiny["rank_decision_cross_sections_below_minimum_population"] == len(
        DECISION_EQUITY_INDICES
    )
    rows = _opening_rows(tiny)
    assert rows["return_since_open_normalized"]["valid_observation_count"] == 29
    assert rows["overnight_gap_normalized"]["valid_observation_count"] == 29
    assert (
        rows["previous_open_to_close_return_normalized"]["valid_observation_count"]
        == 29
    )
    assert rows["overnight_gap_cross_section_rank"]["valid_observation_count"] == 0


def test_local_context_builder_keeps_existing_nominal_open_semantics() -> None:
    raw, observed = _synthetic_equity_grid()
    observed[20, :7] = False
    result = build_causal_features(
        raw, observed, np.ones(21, dtype=bool), is_rate=False
    )
    assert result.slow[20, 1] == 0.0
    assert not result.dynamic[20, :, 6].any()

    nominal_observed = np.ones_like(observed)
    nominal = build_causal_features(
        raw, nominal_observed, np.ones(21, dtype=bool), is_rate=False
    )
    position = 20
    expected = np.log(raw[20, position, 3] / raw[20, 0, 0]) / (
        nominal.sigma[20] * np.sqrt(position + 1)
    )
    np.testing.assert_allclose(
        nominal.dynamic[20, position, 6], np.clip(expected, -10.0, 10.0)
    )
