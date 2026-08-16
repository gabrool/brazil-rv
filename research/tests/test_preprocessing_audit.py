from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from brazil_rv.preprocessing import preprocessing_audit as audit_cli
from brazil_rv.preprocessing import preprocessing_audit_features as feature_audit
from brazil_rv.preprocessing import preprocessing_audit_target as target_audit
from brazil_rv.preprocessing.analyze_preprocessing import (
    OUTPUT_FILES,
    AuditArrays,
    AuditDates,
    causal_factor_betas,
    distribution_metrics,
    fit_di_curve,
    maturity_hull_intersection,
    pca_summary,
    redundancy_tables,
    shift_metrics,
    target_group_metrics,
    validate_audit_indices,
)
from brazil_rv.preprocessing.contract import (
    DYNAMIC_CHANNELS,
    GLOBAL_SLOW_CHANNELS,
    SLOW_CHANNELS,
)
from brazil_rv.preprocessing.transforms import centered_midranks


class FakeArrays:
    def __init__(self, values: dict[str, np.ndarray]):
        self.values = values

    def array(self, filename: str, **_: object) -> np.ndarray:
        return self.values[filename]


def test_distribution_excludes_padding_and_counts_clips_and_zeros() -> None:
    values = np.asarray([0.0, 0.0, 1.0, 10.0, -10.0])
    observed = np.asarray([False, True, True, True, True])
    result = distribution_metrics(values, observed, lower_clip=-10.0, upper_clip=10.0)
    assert result["valid_count"] == 4
    assert result["possible_count"] == 5
    assert result["zero_fraction"] == pytest.approx(0.25)
    assert result["lower_clipping_fraction"] == pytest.approx(0.25)
    assert result["upper_clipping_fraction"] == pytest.approx(0.25)
    assert result["observed_fraction"] == pytest.approx(0.8)


def test_year_and_thirty_minute_normalization_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(feature_audit, "DECISION_EQUITY_INDICES", (1, 31))
    monkeypatch.setattr(feature_audit, "DECISION_CONTEXT_INDICES", (1, 31))
    monkeypatch.setattr(feature_audit, "DECISION_GLOBAL_INDICES", (1, 31))
    date_count, minutes = 3, 31
    equity = np.zeros((date_count, 1, minutes, len(DYNAMIC_CHANNELS)), np.float32)
    equity[:2, 0, :, 0] = np.asarray([1.0, 3.0])[:, None]
    equity[:2, 0, :, 5] = 1.0
    equity[0, 0, 0, 5] = 0.0
    local = np.zeros((date_count, 7, minutes, len(DYNAMIC_CHANNELS)), np.float32)
    global_features = np.zeros(
        (date_count, 8, minutes, len(DYNAMIC_CHANNELS)), np.float32
    )
    values = {
        "equity_features.npy": equity,
        "equity_slow.npy": np.zeros((date_count, 1, len(SLOW_CHANNELS)), np.float32),
        "equity_membership.npy": np.ones((date_count, 1), bool),
        "equity_data_ready.npy": np.ones((date_count, 1), bool),
        "context_features.npy": local,
        "context_slow.npy": np.zeros((date_count, 7, len(SLOW_CHANNELS)), np.float32),
        "context_data_ready.npy": np.zeros((date_count, 7), bool),
        "global_features.npy": global_features,
        "global_slow.npy": np.zeros(
            (date_count, 8, 2, len(GLOBAL_SLOW_CHANNELS)), np.float32
        ),
        "global_data_ready.npy": np.zeros((date_count, 8, 2), bool),
    }
    dates = AuditDates(
        (date(2023, 12, 29), date(2024, 1, 2), date(2024, 7, 8)),
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([2], dtype=np.int64),
    )
    equity_index = pl.DataFrame(
        {"equity_slot": [0], "security_id": ["S0"], "latest_ticker": ["T0"]}
    )
    rows, _ = feature_audit.run_normalization_audit(
        FakeArrays(values), dates, equity_index
    )

    def count(scope: str, value: str) -> int:
        row = next(
            row
            for row in rows
            if row["entity_kind"] == "equity"
            and row["feature"] == "open_move_normalized"
            and row["scope_kind"] == scope
            and row["scope_value"] == value
        )
        return int(row["valid_count"])

    assert count("year", "2023") == 30
    assert count("year", "2024") == 31
    assert count("time_bin_30m", "0") == 59
    assert count("time_bin_30m", "1") == 2


def test_held_out_index_is_rejected() -> None:
    trade_dates = (date(2024, 1, 2), date(2026, 1, 2))
    with pytest.raises(ValueError, match="Held-out test|prohibited split"):
        validate_audit_indices(np.asarray([1]), trade_dates, allow_validation=True)


def test_validation_target_is_rejected_before_array_open(tmp_path: Path) -> None:
    dates = AuditDates(
        (date(2024, 1, 2), date(2024, 7, 8)),
        np.asarray([0], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
    )
    arrays = AuditArrays(tmp_path, dates)
    with pytest.raises(ValueError, match="prohibited split"):
        arrays.target_slice("targets.npy", dates.validation)


def test_target_ties_coverage_and_reconstruction_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(target_audit, "DECISION_EQUITY_INDICES", (1,))
    monkeypatch.setattr(target_audit, "HORIZONS", (1,))
    monkeypatch.setattr(target_audit, "MIN_ACTIVE_EQUITIES", 2)
    raw = np.asarray([0.1, 0.1, 0.2], dtype=np.float32)
    prerank = raw.astype(np.float64) / 1e-4
    stored = centered_midranks(prerank)
    direct = target_group_metrics(
        raw,
        stored,
        np.ones(3, bool),
        0.0,
        np.zeros(3),
        1,
    )
    assert direct["parity_mismatch_count"] == 0
    assert direct["tie_fraction"] == pytest.approx(1 / 3)
    values = {
        "raw_returns.npy": raw.reshape(1, 3, 1, 1),
        "targets.npy": stored.reshape(1, 3, 1, 1),
        "label_mask.npy": np.ones((1, 3, 1, 1), bool),
        "cross_section_median.npy": np.zeros((1, 1, 1), np.float32),
        "horizon_mask.npy": np.ones((1, 1, 1), bool),
        "equity_slow.npy": np.zeros((1, 3, len(SLOW_CHANNELS)), np.float32),
        "equity_features.npy": np.ones((1, 3, 2, len(DYNAMIC_CHANNELS)), np.float32),
        "equity_membership.npy": np.ones((1, 3), bool),
        "equity_data_ready.npy": np.ones((1, 3), bool),
    }
    dates = AuditDates(
        (date(2024, 1, 2),),
        np.asarray([0], dtype=np.int64),
        np.empty(0, dtype=np.int64),
    )
    equity_index = pl.DataFrame(
        {
            "equity_slot": [0, 1, 2],
            "security_id": ["A", "B", "C"],
            "latest_ticker": ["A3", "B3", "C3"],
        }
    )
    summary, coverage, _ = target_audit.run_target_audit(
        FakeArrays(values), dates, equity_index
    )
    assert coverage[0]["candidate_count"] == 3
    assert coverage[0]["tie_fraction"] == pytest.approx(1 / 3)
    assert summary["target_parity"]["mismatch_count"] == 0


def test_exact_sigma_recovers_target_order_when_stored_regime_is_clipped() -> None:
    raw = np.asarray([1e-3, 2e-3, 3e-3])
    sigma = np.asarray([1e-4, 4e-4, 1e-4])
    stored = centered_midranks(raw / sigma)
    clipped_regime = np.full(3, 4.0)
    approximate = target_group_metrics(
        raw,
        stored,
        np.ones(3, dtype=bool),
        0.0,
        clipped_regime,
        1,
    )
    exact = target_group_metrics(
        raw,
        stored,
        np.ones(3, dtype=bool),
        0.0,
        clipped_regime,
        1,
        causal_sigma=sigma,
    )
    assert approximate["parity_mismatch_count"] > 0
    assert exact["parity_mismatch_count"] == 0


def test_redundancy_detects_duplicate_without_structural_zeros() -> None:
    values = np.asarray(
        [[1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [3.0, 3.0, 7.0], [4.0, 4.0, 8.0]]
    )
    valid = np.asarray(
        [
            [True, True, False],
            [True, True, False],
            [True, True, True],
            [True, True, True],
        ]
    )
    pairs, summary = redundancy_tables(values, valid, ("left", "copy", "family"))
    duplicate = next(
        row
        for row in pairs
        if {row["feature_left"], row["feature_right"]} == {"left", "copy"}
    )
    assert duplicate["spearman_rho"] == pytest.approx(1.0)
    family = next(row for row in summary if row["feature"] == "family")
    assert family["valid_count"] == 2
    assert family["std"] == pytest.approx(0.5)


def test_shift_metrics_recover_center_scale_and_availability() -> None:
    training = np.asarray([-1.0, 0.0, 1.0])
    validation = np.asarray([1.0, 3.0, 5.0])
    result = shift_metrics(
        training,
        validation,
        training_possible=3,
        validation_possible=6,
    )
    assert result["validation_mean"] == pytest.approx(3.0)
    assert result["validation_std"] == pytest.approx(2.0 * result["training_std"])
    assert result["observed_fraction_change"] == pytest.approx(-0.5)
    assert result["ks_statistic"] > 0.0


def test_pca_is_fitted_only_on_training_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(7)
    width = len(DYNAMIC_CHANNELS) + len(SLOW_CHANNELS)
    names = tuple((*DYNAMIC_CHANNELS, *SLOW_CHANNELS))
    kinds = tuple(
        (*("dynamic" for _ in DYNAMIC_CHANNELS), *("slow" for _ in SLOW_CHANNELS))
    )
    training_values = rng.normal(size=(8, width))
    validation_values = training_values + 100.0
    valid = np.ones_like(training_values, dtype=bool)
    train = feature_audit.FeatureSample(
        training_values,
        valid,
        names,
        kinds,
        np.zeros(8, dtype=np.int64),
        np.zeros(8, dtype=np.int64),
        np.zeros(8, dtype=np.int64),
        8,
    )
    validation = feature_audit.FeatureSample(
        validation_values,
        valid,
        names,
        kinds,
        np.ones(8, dtype=np.int64),
        np.zeros(8, dtype=np.int64),
        np.zeros(8, dtype=np.int64),
        8,
    )
    seen: list[np.ndarray] = []

    def spy(values: np.ndarray, mask: np.ndarray) -> dict[str, object]:
        seen.append(values.copy())
        return pca_summary(values, mask)

    monkeypatch.setattr(feature_audit, "pca_summary", spy)
    feature_audit.run_redundancy_and_shift(
        {entity: train for entity in ("equity", "local", "global")},
        {entity: validation for entity in ("equity", "local", "global")},
    )
    assert seen
    assert max(float(values.max()) for values in seen) < 10.0


def test_di_curve_recovers_level_and_tilt() -> None:
    maturity = np.asarray([1.0, 2.0, 4.0, 7.0])
    z = (maturity - maturity.mean()) / maturity.std()
    changes = 2.5 - 1.2 * z
    fit = fit_di_curve(changes, maturity, np.ones(4, bool))
    assert fit is not None
    assert fit.level == pytest.approx(2.5)
    assert fit.tilt == pytest.approx(-1.2)
    assert fit.residual_rmse == pytest.approx(0.0, abs=1e-12)


def test_di_curve_handles_one_missing_and_rejects_fewer_than_three() -> None:
    maturity = np.asarray([1.0, 2.0, 4.0, 7.0])
    changes = np.asarray([1.0, 2.0, 3.0, 4.0])
    assert (
        fit_di_curve(changes, maturity, np.asarray([True, False, True, True]))
        is not None
    )
    assert (
        fit_di_curve(changes, maturity, np.asarray([True, False, False, True])) is None
    )


def test_curvature_is_only_evaluated_with_four_contracts() -> None:
    maturity = np.asarray([1.0, 2.0, 4.0, 7.0])
    changes = maturity**2
    three = fit_di_curve(changes, maturity, np.asarray([True, True, True, False]))
    four = fit_di_curve(changes, maturity, np.ones(4, bool))
    assert three is not None and three.curvature is None
    assert four is not None and four.curvature is not None


def test_constant_maturity_hull_intersection() -> None:
    maturity = np.asarray([[1.0, 2.0, 4.0, 7.0], [0.5, 1.5, 3.0, 6.0]])
    result = maturity_hull_intersection(maturity, np.ones_like(maturity, bool))
    assert result["intersection_minimum_maturity_years"] == pytest.approx(1.0)
    assert result["intersection_maximum_maturity_years"] == pytest.approx(6.0)
    assert result["constant_maturity_without_extrapolation_full_interval"]


def test_factor_beta_uses_only_prior_sessions_and_minimum_pairs() -> None:
    count = 22
    equity = np.arange(1.0, count + 1.0)[:, None]
    factors = np.column_stack((equity[:, 0], 2.0 * equity[:, 0]))
    valid_equity = np.ones_like(equity, bool)
    valid_factors = np.ones_like(factors, bool)
    baseline, ready = causal_factor_betas(equity, valid_equity, factors, valid_factors)
    changed_factors = factors.copy()
    changed_factors[20] = 1_000_000.0
    changed, _ = causal_factor_betas(
        equity, valid_equity, changed_factors, valid_factors
    )
    assert not ready[19].any()
    assert ready[20].all()
    np.testing.assert_array_equal(baseline[20], changed[20])


def test_stratified_sampling_is_deterministic() -> None:
    dates = np.asarray([1, 2, 3], dtype=np.int64)
    applicable = np.asarray([[True, True, False], [True, False, True], [True] * 3])
    first = feature_audit.deterministic_stratified_keys(
        dates,
        applicable,
        decision_count=5,
        decisions_per_date=3,
        entities_per_decision=2,
    )
    second = feature_audit.deterministic_stratified_keys(
        dates,
        applicable,
        decision_count=5,
        decisions_per_date=3,
        entities_per_decision=2,
    )
    np.testing.assert_array_equal(first.date_idx, second.date_idx)
    np.testing.assert_array_equal(first.entity_idx, second.entity_idx)
    np.testing.assert_array_equal(first.decision_idx, second.decision_idx)


def test_output_directory_is_complete_and_atomic(tmp_path: Path) -> None:
    output = tmp_path / "audit"

    def complete(path: Path) -> None:
        for filename in OUTPUT_FILES:
            (path / filename).write_text(filename, encoding="utf-8")

    result = audit_cli.atomic_output_directory(output, complete)
    assert result == output.resolve()
    assert {path.name for path in result.iterdir()} == set(OUTPUT_FILES)

    incomplete = tmp_path / "incomplete"
    with pytest.raises(ValueError, match="incomplete"):
        audit_cli.atomic_output_directory(
            incomplete,
            lambda path: (path / OUTPUT_FILES[0]).write_text(
                "partial", encoding="utf-8"
            ),
        )
    assert not incomplete.exists()
    assert not list(tmp_path.glob("*.partial"))
