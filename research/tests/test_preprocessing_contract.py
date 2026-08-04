from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.lib.format import open_memmap

from brazil_rv.preprocessing import audit as audit_module
from brazil_rv.preprocessing import build as build_module
from brazil_rv.preprocessing.contract import (
    DECISION_EQUITY_INDICES,
    DYNAMIC_CHANNELS,
    EQUITY_SESSION_MINUTES,
    EQUITY_SLOW_CHANNELS,
    GLOBAL_SLOW_CHANNELS,
    GLOBAL_UNUSED_SLOW_CHANNEL_INDICES,
    HORIZONS,
    SLOW_CHANNELS,
    output_array_specs,
)
from brazil_rv.preprocessing.build import _write_feature_schema
from brazil_rv.preprocessing.io import (
    create_output_memmaps,
    expand_membership,
    full_session_final_closes,
    validate_assignments,
    validate_physical_source_identity,
    validate_source_date_isolation,
)
from brazil_rv.preprocessing.transforms import (
    add_equity_cross_sectional_dynamic,
    build_causal_features,
    build_daily_changes,
    build_prior_rate_level,
    build_raw_returns,
    causal_exposure_betas,
    centered_midranks,
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
    np.testing.assert_array_equal(changed.dynamic[-1, deleted, :6], np.zeros(6))
    np.testing.assert_array_equal(
        baseline.dynamic[-1, :deleted, :6], changed.dynamic[-1, :deleted, :6]
    )
    np.testing.assert_array_equal(
        baseline.dynamic[-1, deleted + 2 :, :6], changed.dynamic[-1, deleted + 2 :, :6]
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
        "return_since_open_normalized",
        "return_15m_normalized",
        "return_30m_normalized",
        "return_60m_normalized",
        "realized_vol_15m_log_ratio",
        "realized_vol_30m_log_ratio",
        "realized_vol_60m_log_ratio",
        "cumulative_volume_surprise",
        "session_range_position",
        "observed_fraction_30m",
        "market_median_return_15m",
        "market_median_return_60m",
        "market_breadth_15m",
        "market_breadth_60m",
        "market_dispersion_15m",
        "market_dispersion_60m",
        "cross_section_return_rank_15m",
        "cross_section_return_rank_60m",
        "cross_section_volume_rank",
        "cross_section_volatility_rank_30m",
    )
    assert EQUITY_SLOW_CHANNELS == SLOW_CHANNELS
    assert len(SLOW_CHANNELS) == 32

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


def test_centered_midranks_and_rank_target_groups() -> None:
    values = np.array([3.0, 1.0, 1.0, 8.0])
    np.testing.assert_allclose(
        centered_midranks(values), np.array([0.25, -0.5, -0.5, 0.75])
    )
    np.testing.assert_array_equal(centered_midranks(np.array([5.0])), [0.0])
    assert centered_midranks(values).mean() == 0.0

    raw = np.zeros((30, 2, len(HORIZONS)), dtype=np.float32)
    raw[:, 0, 0] = np.arange(30)
    raw[:, 1, 0] = np.arange(30)[::-1]
    raw[:, :, 1] = np.repeat(np.arange(30)[:, None], 2, axis=1)
    raw[:, :, 2] = 4.0
    candidate = np.ones(raw.shape, dtype=bool)
    candidate[0, 0, 0] = False
    sigma = np.linspace(0.5, 2.0, 30)
    _, label_mask, targets, _, horizon_mask = center_cross_section(
        raw, candidate, sigma
    )
    assert not label_mask[:, 0, 0].any()
    assert not targets[:, 0, 0].any()
    assert not horizon_mask[0, 0]
    for decision_idx in range(2):
        for horizon_idx in range(len(HORIZONS)):
            valid = label_mask[:, decision_idx, horizon_idx]
            if not valid.any():
                continue
            assert abs(float(targets[valid, decision_idx, horizon_idx].mean())) < 1e-7
            assert np.all(
                np.diff(np.sort(targets[valid, decision_idx, horizon_idx])) >= 0
            )
    assert np.all(targets[label_mask] > -1.0)
    assert np.all(targets[label_mask] < 1.0)
    assert np.all(targets[:, :, 2][label_mask[:, :, 2]] == 0.0)


def test_exact_trailing_features_and_no_forward_fill() -> None:
    raw, observed = _synthetic_grid(days=21)
    result = build_causal_features(
        raw, observed, np.ones(21, dtype=bool), is_rate=False
    )
    day = 20
    minute = 60
    sigma = result.sigma[day]
    expected_return_15 = np.log(raw[day, minute, 3] / raw[day, minute - 15, 3]) / (
        sigma * np.sqrt(15)
    )
    np.testing.assert_allclose(result.dynamic[day, minute, 7], expected_return_15)
    adjacent = np.log(
        raw[day, minute - 29 : minute + 1, 3] / raw[day, minute - 30 : minute, 3]
    )
    expected_rv_30 = np.clip(np.log(np.sqrt(np.mean(adjacent**2)) / sigma), -4, 4)
    np.testing.assert_allclose(result.dynamic[day, minute, 11], expected_rv_30)

    missing = observed.copy()
    missing[day, minute - 15] = False
    changed = build_causal_features(
        raw, missing, np.ones(21, dtype=bool), is_rate=False
    )
    assert changed.dynamic[day, minute, 7] == 0.0
    for position in (32, 34, 36, 38, 40, 42, 44):
        missing[day, position] = False
    changed = build_causal_features(
        raw, missing, np.ones(21, dtype=bool), is_rate=False
    )
    assert changed.dynamic[day, minute, 11] == 0.0


def test_prefix_cumulative_volume_and_cutoff_causality() -> None:
    raw, observed = _synthetic_grid(days=21)
    valid = np.ones(21, dtype=bool)
    baseline = build_causal_features(raw, observed, valid, is_rate=False)

    volume_changed = raw.copy()
    volume_changed[-1, 201:, 4] *= 1000.0
    changed = build_causal_features(volume_changed, observed, valid, is_rate=False)
    np.testing.assert_array_equal(
        baseline.dynamic[-1, :201, 13], changed.dynamic[-1, :201, 13]
    )

    cutoff = DECISION_EQUITY_INDICES[0]
    future_changed = raw.copy()
    future_changed[-1, cutoff:, :4] *= 1.7
    future_changed[-1, cutoff:, 4] *= 11.0
    changed = build_causal_features(future_changed, observed, valid, is_rate=False)
    np.testing.assert_array_equal(
        baseline.dynamic[-1, :cutoff], changed.dynamic[-1, :cutoff]
    )
    np.testing.assert_array_equal(baseline.slow[-1], changed.slow[-1])


def test_overnight_gap_and_slow_state_ignore_current_close() -> None:
    raw, observed = _synthetic_grid(days=21)
    dates = tuple(date(2024, 1, 2) + timedelta(days=index) for index in range(21))
    baseline = build_causal_features(
        raw,
        observed,
        np.ones(21, dtype=bool),
        is_rate=False,
        market_dates=dates,
    )
    expected_gap = np.log(raw[20, 0, 0] / raw[19, -1, 3]) / (
        baseline.sigma[20] * np.sqrt(EQUITY_SESSION_MINUTES)
    )
    np.testing.assert_allclose(baseline.slow[20, 1], np.clip(expected_gap, -10.0, 10.0))

    changed_raw = raw.copy()
    changed_raw[20, 1:, :4] *= 1.9
    changed_raw[20, 1:, 4] *= 50.0
    changed = build_causal_features(
        changed_raw,
        observed,
        np.ones(21, dtype=bool),
        is_rate=False,
        market_dates=dates,
    )
    np.testing.assert_array_equal(baseline.slow[20], changed.slow[20])
    assert np.isfinite(baseline.slow[20, 7:12]).all()


def test_cross_sectional_leave_one_out_permutation_and_isolation() -> None:
    equity_count = 31
    dynamic = np.zeros((equity_count, 1, len(DYNAMIC_CHANNELS)), dtype=np.float32)
    validity = np.zeros((equity_count, 1, 4), dtype=bool)
    values = np.linspace(-1.0, 1.0, equity_count, dtype=np.float32)
    dynamic[:, 0, 7] = values
    validity[:, 0, 0] = True
    active = np.ones(equity_count, dtype=bool)
    add_equity_cross_sectional_dynamic(dynamic, validity, active)
    np.testing.assert_allclose(dynamic[:, 0, 22], centered_midranks(values))
    for focal in (0, 15, 30):
        peers = np.delete(values, focal)
        np.testing.assert_allclose(dynamic[focal, 0, 16], np.median(peers))
        np.testing.assert_allclose(
            dynamic[focal, 0, 18], 2.0 * np.mean(peers > 0.0) - 1.0
        )

    permutation = np.arange(equity_count)[::-1]
    permuted = np.zeros_like(dynamic)
    permuted[:, :, :16] = dynamic[permutation, :, :16]
    permuted_validity = validity[permutation].copy()
    add_equity_cross_sectional_dynamic(permuted, permuted_validity, active)
    np.testing.assert_allclose(permuted[:, :, 16:26], dynamic[permutation, :, 16:26])

    extended = np.zeros((equity_count + 1, 1, len(DYNAMIC_CHANNELS)), dtype=np.float32)
    extended[:equity_count, :, :16] = dynamic[:, :, :16]
    extended[-1, 0, 7] = 1_000.0
    extended_validity = np.zeros((equity_count + 1, 1, 4), dtype=bool)
    extended_validity[:equity_count, 0, 0] = True
    extended_validity[-1, 0, 0] = True
    extended_active = np.r_[active, False]
    add_equity_cross_sectional_dynamic(extended, extended_validity, extended_active)
    np.testing.assert_allclose(extended[:equity_count, :, 16:26], dynamic[:, :, 16:26])

    too_small = dynamic[:29].copy()
    too_small[:, :, 16:26] = 0.0
    add_equity_cross_sectional_dynamic(
        too_small, validity[:29], np.ones(29, dtype=bool)
    )
    assert not too_small[:, :, 16:26].any()


def test_causal_exposure_beta_uses_prior_paired_sessions() -> None:
    context = np.arange(1.0, 23.0)[:, None]
    equity = np.column_stack((2.0 * context[:, 0], -context[:, 0]))
    context_valid = np.ones(context.shape, dtype=bool)
    equity_valid = np.ones(equity.shape, dtype=bool)
    baseline = causal_exposure_betas(equity, equity_valid, context, context_valid)
    np.testing.assert_allclose(baseline[20, :, 0], [2.0, -1.0], atol=1e-6)

    changed_equity = equity.copy()
    changed_equity[20] = [20_000.0, 30_000.0]
    changed = causal_exposure_betas(
        changed_equity, equity_valid, context, context_valid
    )
    np.testing.assert_array_equal(baseline[20], changed[20])
    assert not np.array_equal(baseline[21], changed[21])

    closes = np.array([100.0, 0.0, 102.0])
    changes, valid = build_daily_changes(
        closes, np.array([True, False, True]), is_rate=False
    )
    np.testing.assert_allclose(changes[2], np.log(1.02))
    assert valid.tolist() == [False, False, True]


def test_context_family_zero_fields_and_deterministic_fixture() -> None:
    raw, observed = _synthetic_grid(days=21)
    valid = np.ones(21, dtype=bool)
    first = build_causal_features(
        raw,
        observed,
        valid,
        is_rate=False,
        include_dollar_volume=False,
    )
    second = build_causal_features(
        raw,
        observed,
        valid,
        is_rate=False,
        include_dollar_volume=False,
    )
    np.testing.assert_array_equal(first.dynamic, second.dynamic)
    np.testing.assert_array_equal(first.slow, second.slow)
    assert not first.dynamic[..., 16:26].any()
    assert not first.slow[..., 13:15].any()
    assert not first.slow[..., 17:26].any()
    assert not first.slow[..., 30:32].any()


def test_global_unused_slow_channel_indices_are_exact() -> None:
    assert GLOBAL_UNUSED_SLOW_CHANNEL_INDICES == (
        *range(13, 16),
        *range(17, 26),
        30,
    )


def test_global_slow_validation_accepts_calendar_and_expiry_channels() -> None:
    slow = np.zeros((2, 1, 3, len(GLOBAL_SLOW_CHANNELS)), dtype=np.float32)
    ready = np.ones(slow.shape[:-1], dtype=bool)
    slow[..., 26:30] = [0.5, -0.5, 0.25, 0.75]
    slow[..., 31] = 0.6

    audit_module.validate_global_slow_fields(slow, ready)


@pytest.mark.parametrize("channel", GLOBAL_UNUSED_SLOW_CHANNEL_INDICES)
def test_every_unused_global_slow_channel_must_be_zero(channel: int) -> None:
    slow = np.zeros((1, 1, 1, len(GLOBAL_SLOW_CHANNELS)), dtype=np.float32)
    ready = np.ones(slow.shape[:-1], dtype=bool)
    slow[..., channel] = 1.0

    with pytest.raises(ValueError, match="Global unused slow channels"):
        audit_module.validate_global_slow_fields(slow, ready)


def test_unready_global_slow_row_must_remain_zero() -> None:
    slow = np.zeros((1, 1, 2, len(GLOBAL_SLOW_CHANNELS)), dtype=np.float32)
    ready = np.ones(slow.shape[:-1], dtype=bool)
    ready[..., 1] = False
    slow[..., 1, 26] = 1.0

    with pytest.raises(ValueError, match="Unready global slow rows"):
        audit_module.validate_global_slow_fields(slow, ready)


def test_generated_schema_matches_channel_contract(tmp_path: Path) -> None:
    _write_feature_schema(tmp_path)
    schema = json.loads((tmp_path / "feature_schema.json").read_text(encoding="utf-8"))
    assert [row["name"] for row in schema["dynamic_channels"]] == list(DYNAMIC_CHANNELS)
    assert [row["name"] for row in schema["slow_channels"]] == list(SLOW_CHANNELS)
    assert [row["name"] for row in schema["global_slow_channels"]] == list(
        GLOBAL_SLOW_CHANNELS
    )
    assert schema["global_slow"] == list(GLOBAL_UNUSED_SLOW_CHANNEL_INDICES)
    assert schema["global_slow_semantics"]["26:30"].startswith("Deterministic")
    assert "17:31" not in schema["global_slow_semantics"]
    assert "average_one_based_midrank" in schema["stored_target"]


def _atomic_build_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, Path, Path, datetime]:
    output_base = tmp_path / "features"
    output_base.mkdir()
    pointer = output_base / "m1_features_canonical_path.txt"
    previous = output_base / "previous_complete"
    previous.mkdir()
    pointer.write_text(str(previous), encoding="utf-8")
    monkeypatch.setattr(build_module, "OUTPUT_BASE", output_base)
    monkeypatch.setattr(build_module, "CANONICAL_OUTPUT_POINTER", pointer)
    created_at = datetime(2026, 1, 2, 3, 4, 5, 6789, tzinfo=UTC)
    final = output_base / f"m1_features_global_context_{created_at:%Y%m%dT%H%M%S%fZ}"
    return pointer, previous, final, created_at


def test_feature_audit_summary_references_final_existing_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_store = tmp_path / "feature_store"
    feature_store.mkdir()
    audit_base = tmp_path / "audits"
    monkeypatch.setattr(audit_module, "AUDIT_BASE", audit_base)

    def generate(
        source: Path,
        partial: Path,
        final: Path,
        _: datetime,
    ) -> None:
        assert source == feature_store
        assert partial.name.endswith(".partial")
        assert not final.exists()
        partial.mkdir(parents=True)
        (partial / "audit_summary.json").write_text(
            json.dumps(
                {
                    "features_dir": str(source),
                    "audit_output_dir": str(final),
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(audit_module, "_generate_feature_audit", generate)
    audit_dir = audit_module.audit_feature_store(feature_store)
    summary = json.loads((audit_dir / "audit_summary.json").read_text(encoding="utf-8"))

    assert Path(summary["features_dir"]) == feature_store
    assert Path(summary["features_dir"]).is_dir()
    assert Path(summary["audit_output_dir"]) == audit_dir
    assert audit_dir.is_dir()
    assert ".partial" not in summary["features_dir"]
    assert not tuple(audit_base.glob("*.partial"))


def test_construct_feature_store_closes_memmaps_on_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "partial"
    handles: list[np.memmap] = []

    def populate(
        path: Path,
        _: datetime,
        __: float,
        memmaps: list[np.memmap],
    ) -> None:
        path.mkdir()
        paths = (
            path / "artifact.npy",
            *(path / name for name in build_module._TEMPORARY_MEMMAP_FILENAMES),
        )
        for number, memmap_path in enumerate(paths):
            array = open_memmap(memmap_path, mode="w+", dtype=np.float32, shape=(2,))
            array[...] = number + 1
            handles.append(array)
            memmaps.append(array)

    monkeypatch.setattr(build_module, "_populate_feature_store", populate)

    build_module._construct_feature_store(output, datetime.now(UTC), 0.0)

    assert all(handle._mmap.closed for handle in handles)
    np.testing.assert_array_equal(np.load(output / "artifact.npy"), [1.0, 1.0])
    assert all(
        not (output / name).exists()
        for name in build_module._TEMPORARY_MEMMAP_FILENAMES
    )


def test_feature_build_failure_closes_memmaps_before_removing_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pointer, previous, final, created_at = _atomic_build_paths(monkeypatch, tmp_path)
    handles: list[np.memmap] = []
    real_rmtree = build_module.shutil.rmtree

    def populate(
        partial: Path,
        _: datetime,
        __: float,
        memmaps: list[np.memmap],
    ) -> None:
        partial.mkdir()
        array = open_memmap(
            partial / "held.npy", mode="w+", dtype=np.float32, shape=(2,)
        )
        array[...] = 1.0
        handles.append(array)
        memmaps.append(array)
        raise RuntimeError("injected failure after memmap creation")

    def remove_after_close(path: Path) -> None:
        assert handles and all(handle._mmap.closed for handle in handles)
        real_rmtree(path)

    monkeypatch.setattr(build_module, "_populate_feature_store", populate)
    monkeypatch.setattr(build_module.shutil, "rmtree", remove_after_close)

    with pytest.raises(RuntimeError, match="failure after memmap creation"):
        build_module.build_feature_store(created_at=created_at)

    assert pointer.read_text(encoding="utf-8") == str(previous)
    assert previous.is_dir()
    assert not final.exists()
    assert not tuple(final.parent.glob(f"{final.name}.*.partial"))


def test_feature_build_cleanup_failure_is_not_suppressed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pointer, previous, final, created_at = _atomic_build_paths(monkeypatch, tmp_path)

    def construct(partial: Path, *_: object) -> None:
        partial.mkdir()
        raise RuntimeError("injected build failure")

    def fail_cleanup(_: Path) -> None:
        raise PermissionError("injected Windows cleanup failure")

    monkeypatch.setattr(build_module, "_construct_feature_store", construct)
    monkeypatch.setattr(build_module.shutil, "rmtree", fail_cleanup)

    with pytest.raises(RuntimeError, match="Failed to remove incomplete") as error:
        build_module.build_feature_store(created_at=created_at)

    assert isinstance(error.value.__cause__, PermissionError)
    assert pointer.read_text(encoding="utf-8") == str(previous)
    assert not final.exists()
    assert tuple(final.parent.glob(f"{final.name}.*.partial"))


def test_feature_build_renames_then_audits_then_promotes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pointer, _, final, created_at = _atomic_build_paths(monkeypatch, tmp_path)
    audit_dir = tmp_path / "audit"
    events: list[str] = []
    real_replace = build_module.os.replace

    def construct(partial: Path, *_: object) -> None:
        events.append("construct partial")
        assert partial.parent == final.parent
        assert partial.name.endswith(".partial")
        assert not final.exists()
        partial.mkdir()
        (partial / "artifact").write_text("complete", encoding="utf-8")

    def replace(source: Path, destination: Path) -> None:
        events.append("rename final")
        assert Path(source).name.endswith(".partial")
        assert Path(destination) == final
        real_replace(source, destination)

    def audit(path: Path) -> Path:
        events.append("audit final")
        assert path == final
        assert path.is_dir()
        assert pointer.read_text(encoding="utf-8") != str(final)
        audit_dir.mkdir()
        return audit_dir

    def promote(path: Path) -> None:
        events.append("promote pointer")
        assert path == final
        pointer.write_text(str(path), encoding="utf-8")

    monkeypatch.setattr(build_module, "_construct_feature_store", construct)
    monkeypatch.setattr(build_module.os, "replace", replace)
    monkeypatch.setattr(build_module, "audit_feature_store", audit)
    monkeypatch.setattr(build_module, "_promote", promote)

    output, published_audit = build_module.build_feature_store(created_at=created_at)

    assert events == [
        "construct partial",
        "rename final",
        "audit final",
        "promote pointer",
    ]
    assert output == final
    assert published_audit == audit_dir
    assert (final / "artifact").read_text(encoding="utf-8") == "complete"
    assert pointer.read_text(encoding="utf-8") == str(final)
    assert not tuple(final.parent.glob(f"{final.name}.*.partial"))


def test_feature_build_failure_before_rename_cleans_only_its_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pointer, previous, final, created_at = _atomic_build_paths(monkeypatch, tmp_path)
    unrelated = final.parent / "unrelated.partial"
    unrelated.mkdir()

    def construct(partial: Path, *_: object) -> None:
        partial.mkdir()
        (partial / "artifact").write_text("incomplete", encoding="utf-8")
        raise RuntimeError("injected build failure")

    monkeypatch.setattr(build_module, "_construct_feature_store", construct)

    with pytest.raises(RuntimeError, match="injected build failure"):
        build_module.build_feature_store(created_at=created_at)

    assert pointer.read_text(encoding="utf-8") == str(previous)
    assert previous.is_dir()
    assert not final.exists()
    assert not tuple(final.parent.glob(f"{final.name}.*.partial"))
    assert unrelated.is_dir()


def test_feature_audit_failure_cleans_outputs_and_keeps_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pointer, previous, final, created_at = _atomic_build_paths(monkeypatch, tmp_path)
    audit_base = tmp_path / "audits"
    unrelated_audit = audit_base / "existing_audit"
    unrelated_audit.mkdir(parents=True)

    def construct(partial: Path, *_: object) -> None:
        partial.mkdir()
        (partial / "artifact").write_text("complete", encoding="utf-8")

    def fail_audit(
        source: Path,
        partial: Path,
        _: Path,
        __: datetime,
    ) -> None:
        assert source == final
        partial.mkdir(parents=True)
        (partial / "incomplete").write_text("incomplete", encoding="utf-8")
        raise RuntimeError("injected audit failure")

    monkeypatch.setattr(audit_module, "AUDIT_BASE", audit_base)
    monkeypatch.setattr(audit_module, "_generate_feature_audit", fail_audit)
    monkeypatch.setattr(build_module, "_construct_feature_store", construct)
    monkeypatch.setattr(
        build_module, "audit_feature_store", audit_module.audit_feature_store
    )

    with pytest.raises(RuntimeError, match="injected audit failure"):
        build_module.build_feature_store(created_at=created_at)

    assert pointer.read_text(encoding="utf-8") == str(previous)
    assert previous.is_dir()
    assert not final.exists()
    assert not tuple(final.parent.glob(f"{final.name}.*.partial"))
    assert tuple(audit_base.iterdir()) == (unrelated_audit,)


def test_pointer_failure_before_replacement_rolls_back_invocation_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pointer, previous, final, created_at = _atomic_build_paths(monkeypatch, tmp_path)
    audit_dir = tmp_path / "audits" / "new_audit"
    unrelated_audit = tmp_path / "audits" / "existing_audit"
    unrelated_audit.mkdir(parents=True)
    real_replace = build_module.os.replace

    def construct(partial: Path, *_: object) -> None:
        partial.mkdir()
        (partial / "artifact").write_text("complete", encoding="utf-8")

    def audit(path: Path) -> Path:
        assert path == final
        audit_dir.mkdir()
        (audit_dir / "audit_summary.json").write_text("{}", encoding="utf-8")
        return audit_dir

    def fail_pointer_replace(source: Path, destination: Path) -> None:
        if Path(destination) == pointer:
            raise RuntimeError("injected pointer failure")
        real_replace(source, destination)

    monkeypatch.setattr(build_module, "_construct_feature_store", construct)
    monkeypatch.setattr(build_module, "audit_feature_store", audit)
    monkeypatch.setattr(build_module.os, "replace", fail_pointer_replace)

    with pytest.raises(RuntimeError, match="injected pointer failure"):
        build_module.build_feature_store(created_at=created_at)

    assert pointer.read_text(encoding="utf-8") == str(previous)
    assert previous.is_dir()
    assert not final.exists()
    assert not audit_dir.exists()
    assert unrelated_audit.is_dir()
    assert not tuple(final.parent.glob(f"{final.name}.*.partial"))
    assert not tuple(pointer.parent.glob(f"{pointer.name}.*.tmp"))


def test_pointer_interruption_after_replacement_preserves_committed_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pointer, previous, final, created_at = _atomic_build_paths(monkeypatch, tmp_path)
    audit_dir = tmp_path / "audits" / "new_audit"
    unrelated_audit = tmp_path / "audits" / "existing_audit"
    unrelated_audit.mkdir(parents=True)
    real_replace = build_module.os.replace

    def construct(partial: Path, *_: object) -> None:
        partial.mkdir()
        (partial / "artifact").write_text("complete", encoding="utf-8")

    def audit(path: Path) -> Path:
        assert path == final
        audit_dir.mkdir()
        (audit_dir / "audit_summary.json").write_text("{}", encoding="utf-8")
        return audit_dir

    def replace_then_interrupt(source: Path, destination: Path) -> None:
        real_replace(source, destination)
        if Path(destination) == pointer:
            raise KeyboardInterrupt("injected post-replacement interruption")

    monkeypatch.setattr(build_module, "_construct_feature_store", construct)
    monkeypatch.setattr(build_module, "audit_feature_store", audit)
    monkeypatch.setattr(build_module.os, "replace", replace_then_interrupt)

    with pytest.raises(KeyboardInterrupt, match="post-replacement interruption"):
        build_module.build_feature_store(created_at=created_at)

    assert pointer.read_text(encoding="utf-8") == str(final)
    assert final.is_dir()
    assert (final / "artifact").is_file()
    assert audit_dir.is_dir()
    assert (audit_dir / "audit_summary.json").is_file()
    assert previous.is_dir()
    assert unrelated_audit.is_dir()
    assert not tuple(final.parent.glob(f"{final.name}.*.partial"))
    assert not tuple(pointer.parent.glob(f"{pointer.name}.*.tmp"))
