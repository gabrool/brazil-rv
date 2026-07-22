from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from brazil_rv.preprocessing.contract import (
    DECISION_EQUITY_INDICES,
    DYNAMIC_CHANNELS,
    EQUITY_SESSION_MINUTES,
    HORIZONS,
    output_array_specs,
)
from brazil_rv.preprocessing.io import (
    create_output_memmaps,
    expand_membership,
    full_session_final_closes,
    validate_assignments,
    validate_physical_source_identity,
    validate_source_date_isolation,
)
from brazil_rv.preprocessing.transforms import (
    build_causal_features,
    build_prior_rate_level,
    build_raw_returns,
    center_cross_section,
    time_to_expiry_scaled,
)


def _synthetic_grid(
    days: int = 23,
    *,
    price_scale: float = 1.0,
    volume_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    minute = np.arange(EQUITY_SESSION_MINUTES, dtype=np.float64)
    raw = np.zeros((days, EQUITY_SESSION_MINUTES, 5), dtype=np.float64)
    for day_idx in range(days):
        base = price_scale * (80.0 + 0.05 * day_idx)
        close = base * np.exp(0.0002 * minute + 0.001 * np.sin(minute / 11.0 + day_idx))
        open_ = close * np.exp(0.0001 * np.cos(minute / 7.0 + day_idx))
        raw[day_idx, :, 0] = open_
        raw[day_idx, :, 1] = np.maximum(open_, close) * 1.0002
        raw[day_idx, :, 2] = np.minimum(open_, close) * 0.9998
        raw[day_idx, :, 3] = close
        raw[day_idx, :, 4] = volume_scale * np.exp(
            6.0 + 0.035 * day_idx + 0.15 * np.sin(minute / 19.0 + day_idx)
        )
    return raw, np.ones(raw.shape[:2], dtype=bool)


def _synthetic_rate_grid(days: int = 23) -> tuple[np.ndarray, np.ndarray]:
    minute = np.arange(EQUITY_SESSION_MINUTES, dtype=np.float64)
    raw = np.zeros((days, EQUITY_SESSION_MINUTES, 5), dtype=np.float64)
    for day_idx in range(days):
        close = (
            10.0
            + 0.01 * day_idx
            + 0.0005 * minute
            + 0.005 * np.sin(minute / 13.0 + day_idx)
        )
        open_ = close + 0.0002 * np.cos(minute / 9.0 + day_idx)
        raw[day_idx, :, 0] = open_
        raw[day_idx, :, 1] = np.maximum(open_, close) + 0.0003
        raw[day_idx, :, 2] = np.minimum(open_, close) - 0.0003
        raw[day_idx, :, 3] = close
        raw[day_idx, :, 4] = np.exp(
            6.0 + 0.03 * day_idx + 0.1 * np.sin(minute / 17.0 + day_idx)
        )
    return raw, np.ones(raw.shape[:2], dtype=bool)


def test_future_mutation_causality() -> None:
    raw, observed = _synthetic_grid(days=21)
    observed[-1] = False
    valid = np.ones(raw.shape[0], dtype=bool)
    baseline = build_causal_features(raw, observed, valid, is_rate=False)
    decision = DECISION_EQUITY_INDICES[0]

    changed_observed = observed.copy()
    changed_observed[-1, decision + 1] = True
    changed = build_causal_features(raw, changed_observed, valid, is_rate=False)

    np.testing.assert_array_equal(
        baseline.dynamic[-1, :decision], changed.dynamic[-1, :decision]
    )
    assert baseline.data_ready[-1] and changed.data_ready[-1]

    baseline_active = int(np.full(30, baseline.data_ready[-1]).sum())
    changed_active = int(np.full(30, changed.data_ready[-1]).sum())
    baseline_context_ready = np.full(6, baseline.data_ready[-1])
    changed_context_ready = np.full(6, changed.data_ready[-1])
    baseline_feature_sample = baseline_context_ready.all() and baseline_active >= 30
    changed_feature_sample = changed_context_ready.all() and changed_active >= 30

    assert baseline_active == changed_active
    np.testing.assert_array_equal(baseline_context_ready, changed_context_ready)
    assert baseline_feature_sample == changed_feature_sample


def test_price_scale_invariance() -> None:
    raw, observed = _synthetic_grid()
    scaled = raw.copy()
    daily_multipliers = np.linspace(0.7, 3.1, raw.shape[0])
    scaled[:, :, :4] *= daily_multipliers[:, None, None]
    valid = np.ones(raw.shape[0], dtype=bool)
    baseline = build_causal_features(raw, observed, valid, is_rate=False)
    changed = build_causal_features(scaled, observed, valid, is_rate=False)
    np.testing.assert_allclose(
        baseline.dynamic[:, :, :4], changed.dynamic[:, :, :4], atol=2e-5
    )
    returns, endpoints = build_raw_returns(raw, observed)
    scaled_returns, scaled_endpoints = build_raw_returns(scaled, observed)
    np.testing.assert_array_equal(endpoints, scaled_endpoints)
    np.testing.assert_allclose(returns, scaled_returns, atol=1e-7)

    day_returns = np.repeat(returns[-1][None, :, :], 30, axis=0)
    offsets = np.linspace(-0.001, 0.001, 30, dtype=np.float32)[:, None, None]
    day_returns += offsets
    scaled_day_returns = np.repeat(scaled_returns[-1][None, :, :], 30, axis=0)
    scaled_day_returns += offsets
    candidates = np.ones(day_returns.shape, dtype=bool)
    sigma = np.repeat(baseline.sigma[-1], 30)
    scaled_sigma = np.repeat(changed.sigma[-1], 30)
    _, _, targets, _, _ = center_cross_section(day_returns, candidates, sigma)
    _, _, scaled_targets, _, _ = center_cross_section(
        scaled_day_returns, candidates, scaled_sigma
    )
    np.testing.assert_allclose(targets, scaled_targets, atol=1e-6)


def test_volume_scale_invariance() -> None:
    raw, observed = _synthetic_grid(volume_scale=1.0)
    scaled, _ = _synthetic_grid(volume_scale=19.0)
    valid = np.ones(raw.shape[0], dtype=bool)
    baseline = build_causal_features(raw, observed, valid, is_rate=False)
    changed = build_causal_features(scaled, observed, valid, is_rate=False)
    np.testing.assert_allclose(
        baseline.dynamic[:, :, 4], changed.dynamic[:, :, 4], atol=2e-5
    )


def test_identity_isolation_and_overlap_failure() -> None:
    raw, observed = _synthetic_grid(days=42)
    first_valid = np.zeros(42, dtype=bool)
    first_valid[:21] = True
    second_valid = ~first_valid
    first_observed = observed & first_valid[:, None]
    second_observed = observed & second_valid[:, None]
    first = build_causal_features(raw, first_observed, first_valid, is_rate=False)
    second = build_causal_features(raw, second_observed, second_valid, is_rate=False)
    assert first.data_ready[20]
    assert not first.data_ready[21:].any()
    assert not second.data_ready[:41].any()
    assert second.data_ready[41]
    _, first_endpoints = build_raw_returns(raw, first_observed)
    assert not first_endpoints[21:].any()

    start = date(2024, 1, 2)
    source = str(Path("shared.parquet"))
    assignments = pl.DataFrame(
        {
            "security_id": ["A", "B"],
            "source_file": [source, source],
        }
    )
    disjoint = {
        "A": frozenset(start + timedelta(days=i) for i in range(2)),
        "B": frozenset(start + timedelta(days=i) for i in range(2, 4)),
    }
    validate_source_date_isolation(assignments, disjoint)
    assert not disjoint["A"].intersection(disjoint["B"])
    overlapping = {**disjoint, "B": frozenset({start + timedelta(days=1)})}
    with pytest.raises(ValueError, match="Overlapping accepted identity dates"):
        validate_source_date_isolation(assignments, overlapping)


def test_membership_isolation() -> None:
    market_dates = tuple(date(2024, 1, 2) + timedelta(days=i) for i in range(5))
    membership = pl.DataFrame(
        {
            "security_id": ["A"],
            "effective_from": [market_dates[2]],
            "effective_to_exclusive": [None],
            "is_member": [True],
        },
        schema_overrides={
            "effective_from": pl.Date,
            "effective_to_exclusive": pl.Date,
        },
    )
    mask = expand_membership(membership, market_dates, ("A",))
    np.testing.assert_array_equal(mask[:, 0], [False, False, True, True, True])


def test_entry_causality_and_horizon_indices() -> None:
    raw, observed = _synthetic_grid(days=21)
    entry = DECISION_EQUITY_INDICES[0]
    raw[-1, entry, 0] = 100.0
    raw[-1, entry + 29, 3] = 103.0
    raw[-1, entry + 59, 3] = 106.0
    raw[-1, entry + 119, 3] = 112.0
    returns, endpoints = build_raw_returns(raw, observed)
    np.testing.assert_allclose(
        returns[-1, 0], np.log(np.array([103.0, 106.0, 112.0]) / 100.0)
    )
    assert endpoints[-1, 0].all()
    features = build_causal_features(
        raw, observed, np.ones(21, dtype=bool), is_rate=False
    ).dynamic
    prefix = features[-1, :entry]
    assert prefix.shape[0] == 15
    assert features[-1, entry, 5] == 1.0
    assert prefix[-1, 5] == 1.0


def test_missing_bar_semantics() -> None:
    raw, observed = _synthetic_grid(days=21)
    valid = np.ones(21, dtype=bool)
    baseline = build_causal_features(raw, observed, valid, is_rate=False)
    missing_observed = observed.copy()
    missing_observed[-1, DECISION_EQUITY_INDICES[0]] = False
    changed = build_causal_features(raw, missing_observed, valid, is_rate=False)
    deleted = DECISION_EQUITY_INDICES[0]
    np.testing.assert_array_equal(changed.dynamic[-1, deleted], np.zeros(6))
    np.testing.assert_array_equal(
        baseline.dynamic[-1, :deleted], changed.dynamic[-1, :deleted]
    )
    np.testing.assert_array_equal(
        baseline.dynamic[-1, deleted + 2 :], changed.dynamic[-1, deleted + 2 :]
    )
    _, endpoint_mask = build_raw_returns(raw, missing_observed)
    assert not endpoint_mask[-1, 0].any()


def test_state_causality() -> None:
    raw, observed = _synthetic_grid(days=22)
    valid = np.ones(22, dtype=bool)
    baseline = build_causal_features(raw, observed, valid, is_rate=False)
    changed_raw = raw.copy()
    minute_factor = np.exp(
        0.002 * np.sin(np.arange(EQUITY_SESSION_MINUTES, dtype=np.float64))
    )
    changed_raw[20, :, :4] *= minute_factor[:, None]
    changed_raw[20, :, 4] *= 5.0
    changed = build_causal_features(changed_raw, observed, valid, is_rate=False)
    assert baseline.sigma[20] == changed.sigma[20]
    assert baseline.sigma[21] != changed.sigma[21]

    position = 100
    prior_logs = np.log(raw[:20, position, 4])
    median = np.median(prior_logs)
    scale = max(1.4826 * np.median(np.abs(prior_logs - median)), 0.1)
    expected = np.clip(
        (np.log(changed_raw[20, position, 4]) - median) / scale, -6.0, 6.0
    )
    np.testing.assert_allclose(changed.dynamic[20, position, 4], expected, atol=1e-6)


def test_di_causal_state_and_scaling() -> None:
    raw, observed = _synthetic_rate_grid()
    observed[21] = False
    valid = np.ones(raw.shape[0], dtype=bool)
    daily_close = raw[:, -1, 3]
    daily_close_observed = np.ones(raw.shape[0], dtype=bool)
    daily_close_observed[21] = False
    prior_rate, prior_ready = build_prior_rate_level(daily_close, daily_close_observed)
    result = build_causal_features(
        raw, observed, valid, is_rate=True, extra_ready=prior_ready
    )

    np.testing.assert_allclose(
        prior_rate[20], np.clip(raw[19, -1, 3] / 10.0, -1.0, 3.0)
    )
    np.testing.assert_allclose(
        prior_rate[21], np.clip(raw[20, -1, 3] / 10.0, -1.0, 3.0)
    )
    assert prior_ready[21] and prior_ready[22]
    assert prior_rate[22] == prior_rate[21]
    assert result.data_ready[21]
    assert not result.dynamic[21].any()

    position = 25
    anchor = raw[20, position - 1, 3]
    expected_moves = np.clip(
        100.0 * (raw[20, position, :4] - anchor) / result.sigma[20],
        -10.0,
        10.0,
    )
    np.testing.assert_allclose(
        result.dynamic[20, position, :4], expected_moves, atol=1e-6
    )

    mutated = raw.copy()
    mutated[20, :, :4] += (
        0.003 * np.sin(np.arange(EQUITY_SESSION_MINUTES, dtype=np.float64))
    )[:, None]
    mutated_prior, mutated_prior_ready = build_prior_rate_level(
        mutated[:, -1, 3], daily_close_observed
    )
    changed = build_causal_features(
        mutated,
        observed,
        valid,
        is_rate=True,
        extra_ready=mutated_prior_ready,
    )
    assert mutated_prior[20] == prior_rate[20]
    assert changed.sigma[20] == result.sigma[20]
    assert changed.sigma[21] != result.sigma[21]

    market_dates = (date(2016, 1, 1), date(2025, 1, 1), date(2026, 1, 1))
    expiry = date(2026, 1, 1)
    scaled_expiry = time_to_expiry_scaled(market_dates, expiry)
    expected_one_year = np.log1p((expiry - market_dates[1]).days / 365.25) / np.log(
        11.0
    )
    np.testing.assert_allclose(scaled_expiry[1], expected_one_year)
    assert scaled_expiry[0] == 1.0
    assert scaled_expiry[2] == 0.0


def test_di_prior_level_uses_full_session_final_close() -> None:
    market_dates = (date(2024, 1, 2), date(2024, 1, 3))
    source = pl.DataFrame(
        {
            "ts_exchange": [
                datetime(2024, 1, 2, 16, 44),
                datetime(2024, 1, 2, 17, 5),
            ],
            "close": [10.25, 11.75],
        }
    )
    daily_close, observed = full_session_final_closes(source, market_dates)
    prior_rate, prior_ready = build_prior_rate_level(daily_close, observed)

    assert prior_ready.tolist() == [False, True]
    np.testing.assert_allclose(prior_rate[1], 11.75 / 10.0)


def test_assignment_and_physical_source_identity_rejection() -> None:
    security_ids = [f"SECURITY_{index:03d}" for index in range(158)]
    security_ids[-1] = security_ids[0]
    duplicate_assignments = pl.DataFrame(
        {
            "security_id": security_ids,
            "manual_decision": ["ACCEPTED"] * 158,
            "normalization_rule": ["FILTER_TO_COTAHIST_SECURITY_DATES"] * 158,
        }
    )
    with pytest.raises(ValueError, match="158 unique security_id"):
        validate_assignments(duplicate_assignments)

    source_path = Path("mismatched.parquet")
    assignments = pl.DataFrame(
        {"source_file": [str(source_path)], "xp_symbol": ["EXPECTED"]}
    )
    source = pl.DataFrame({"symbol": ["ACTUAL"]})
    with pytest.raises(ValueError, match="symbol mismatch"):
        validate_physical_source_identity(assignments, source, source_path)


def test_output_contract(tmp_path: Path) -> None:
    arrays = create_output_memmaps(tmp_path / "synthetic", date_count=1)
    specs = output_array_specs(1)
    for filename, spec in specs.items():
        assert arrays[filename].shape == spec.shape
        assert arrays[filename].dtype == spec.dtype
        assert not arrays[filename].any()
        if np.issubdtype(spec.dtype, np.floating):
            assert np.isfinite(arrays[filename]).all()
    assert DYNAMIC_CHANNELS == (
        "open_move_normalized",
        "high_move_normalized",
        "low_move_normalized",
        "close_move_normalized",
        "volume_surprise",
        "observed",
    )

    raw = np.zeros((30, 1, len(HORIZONS)), dtype=np.float32)
    raw[:, 0, :] = np.linspace(-0.01, 0.01, 30)[:, None]
    candidate = np.ones(raw.shape, dtype=bool)
    masked_raw, label_mask, targets, medians, horizon_mask = center_cross_section(
        raw, candidate, np.ones(30, dtype=np.float64)
    )
    assert label_mask.all() and horizon_mask.all()
    assert np.isfinite(masked_raw).all()
    assert np.isfinite(targets).all()
    assert np.isfinite(medians).all()
    assert not targets[~label_mask].any()
