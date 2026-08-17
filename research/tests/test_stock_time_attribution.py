from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import brazil_rv.modeling.analyze_stock_time_attribution as attribution
from brazil_rv.modeling.analyze_stock_time_attribution import (
    AttributionInputs,
    OpeningDiagnostics,
    bootstrap_summary,
    causal_observation_completeness,
    horizon_attribution,
    learn_overnight_thresholds,
    moving_block_bootstrap_matrix,
    opening_attribution,
    opening_context_attribution,
    primary_time_bins,
    rank_decomposition,
    stock_attribution,
    time_of_day_30m_attribution,
    time_of_day_attribution,
)
from brazil_rv.modeling.metrics import average_ranks, sample_level_ic


def _opening(sample_count: int) -> OpeningDiagnostics:
    return OpeningDiagnostics(
        np.linspace(-2.0, 2.0, sample_count),
        (-1.0, 1.0),
        np.linspace(15.0, 250.0, sample_count),
        np.linspace(0.70, 1.0, sample_count),
        np.linspace(0.75, 1.0, sample_count),
        np.linspace(0.0, 45.0, sample_count),
        np.arange(sample_count) % 2 == 0,
    )


def _inputs(sample_count: int = 6, equity_count: int = 30) -> AttributionInputs:
    rng = np.random.default_rng(20260813)
    predictions = rng.normal(size=(sample_count, equity_count, 3)).astype(np.float32)
    targets = (predictions + rng.normal(scale=0.2, size=predictions.shape)).astype(
        np.float32
    )
    returns = rng.normal(scale=0.01, size=predictions.shape).astype(np.float32)
    mask = np.ones_like(predictions, dtype=bool)
    dates = np.repeat(np.arange((sample_count + 2) // 3), 3)[:sample_count]
    decisions = np.tile(np.arange(3), (sample_count + 2) // 3)[:sample_count]
    return AttributionInputs(
        "run-a",
        predictions,
        targets,
        returns,
        mask,
        dates,
        decisions,
        tuple(f"SEC-{slot}" for slot in range(equity_count)),
        _opening(sample_count),
    )


def test_exact_stock_contributions_reconcile_with_ties_and_unequal_coverage() -> None:
    rng = np.random.default_rng(17)
    predictions = rng.normal(size=(6, 35, 3)).astype(np.float32)
    targets = (predictions + rng.normal(scale=0.4, size=predictions.shape)).astype(
        np.float32
    )
    predictions[0, :5, 0] = 1.0
    targets[1, :4, 1] = -0.5
    mask = np.ones_like(predictions, dtype=bool)
    mask[0, :3, 0] = False
    mask[1, 4:8, 0] = False
    mask[2, :5, 1] = False
    mask[3, 10:15, 1] = False
    mask[4, :6, 2] = False
    predictions[5, :, 2] = 2.0  # zero rank variance is ineligible
    expected, _ = sample_level_ic(predictions, targets, mask)
    decomposition = rank_decomposition(predictions, targets, mask)
    np.testing.assert_array_equal(np.isnan(decomposition.sample_ic), np.isnan(expected))
    np.testing.assert_allclose(
        decomposition.sample_ic, expected, rtol=0.0, atol=2e-15, equal_nan=True
    )
    for sample, horizon in np.argwhere(decomposition.eligible):
        invalid = ~mask[sample, :, horizon]
        assert not decomposition.contributions[sample, invalid, horizon].any()
        assert np.isclose(
            decomposition.contributions[sample, :, horizon].sum(),
            expected[sample, horizon],
            rtol=0.0,
            atol=2e-15,
        )

    inputs = AttributionInputs(
        "run",
        predictions,
        targets,
        np.zeros_like(predictions),
        mask,
        np.arange(6),
        np.zeros(6, dtype=np.int64),
        tuple(f"SEC-{slot}" for slot in range(35)),
        _opening(6),
    )
    frame = stock_attribution(inputs)
    for horizon_index, minutes in enumerate((30, 60, 120)):
        contribution_sum = (
            frame.filter(attribution.pl.col("horizon_minutes") == minutes)
            .get_column("mean_spearman_contribution")
            .sum()
        )
        assert np.isclose(
            contribution_sum,
            np.nanmean(expected[:, horizon_index]),
            rtol=0.0,
            atol=2e-15,
        )
    assert frame.get_column("coverage").n_unique() > 1


def test_time_series_rank_skill_uses_normalized_cross_sectional_ranks() -> None:
    rng = np.random.default_rng(29)
    base = rng.normal(size=(20, 30, 3)).astype(np.float32)
    targets = (base + rng.normal(scale=0.25, size=base.shape)).astype(np.float32)
    predictions = base.copy()
    trend = 100.0 * np.arange(20, dtype=np.float32)
    predictions += trend[:, None, None]
    targets -= trend[:, None, None]
    inputs = AttributionInputs(
        "run",
        predictions,
        targets,
        np.zeros_like(predictions),
        np.ones_like(predictions, dtype=bool),
        np.arange(20),
        np.zeros(20, dtype=np.int64),
        tuple(f"SEC-{slot}" for slot in range(30)),
        _opening(20),
    )
    predicted_coordinates = []
    target_coordinates = []
    for sample in range(20):
        predicted = average_ranks(predictions[sample, :, 0])
        actual = average_ranks(targets[sample, :, 0])
        predicted -= predicted.mean()
        actual -= actual.mean()
        predicted_coordinates.append(predicted[0] / np.sqrt(np.sum(predicted**2)))
        target_coordinates.append(actual[0] / np.sqrt(np.sum(actual**2)))
    expected = np.corrcoef(predicted_coordinates, target_coordinates)[0, 1]
    raw = np.corrcoef(predictions[:, 0, 0], targets[:, 0, 0])[0, 1]
    skill = (
        stock_attribution(inputs)
        .filter(
            (attribution.pl.col("equity_slot") == 0)
            & (attribution.pl.col("horizon_minutes") == 30)
        )
        .item(0, "time_series_rank_skill")
    )
    assert np.isclose(skill, expected, rtol=0.0, atol=1e-15)
    assert abs(skill - raw) > 0.5


def test_time_horizon_and_opening_reports_keep_economic_diagnostics() -> None:
    inputs = _inputs()
    reports = (
        time_of_day_attribution(inputs),
        time_of_day_30m_attribution(inputs),
        horizon_attribution(inputs),
        opening_attribution(inputs),
        opening_context_attribution(inputs),
    )
    for frame in reports:
        assert set(frame.columns) >= {
            "mean_spearman_ic",
            "mean_top_minus_bottom",
            "mean_one_way_turnover",
            "label_coverage",
        }
    regimes = set(reports[3].get_column("opening_regime"))
    assert regimes == {"large_negative", "middle", "large_positive"}


def test_thirty_minute_bins_use_historical_six_decision_partition() -> None:
    bins = primary_time_bins()
    assert bins[0] == tuple(range(6))
    assert bins[-1] == tuple(range(48, 55))
    assert tuple(value for group in bins for value in group) == tuple(range(55))
    assert len(bins) == 9


def test_day_block_bootstrap_is_deterministic() -> None:
    values = np.arange(24, dtype=np.float64).reshape(8, 3)
    values[2, 1] = np.nan
    first = moving_block_bootstrap_matrix(
        values, replications=200, block_length=5, seed=31
    )
    second = moving_block_bootstrap_matrix(
        values, replications=200, block_length=5, seed=31
    )
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


def test_opening_thresholds_are_fit_only_from_training_values() -> None:
    training = np.arange(100, dtype=np.float64)
    thresholds = learn_overnight_thresholds(training)
    assert thresholds == tuple(np.quantile(training, (0.10, 0.90)))
    validation_a = np.array([-1_000.0, 50.0, 1_000.0])
    validation_b = validation_a * 100
    assert learn_overnight_thresholds(training) == thresholds
    assert not np.array_equal(validation_a, validation_b)


def test_opening_context_completeness_uses_only_causal_prefix() -> None:
    observed = np.zeros((2, 2, 12), dtype=bool)
    observed[0, 0, :5] = True
    observed[0, 1, [0, 3]] = True
    observed[1, :, :8] = True
    result = causal_observation_completeness(
        observed,
        np.array([0, 1]),
        np.array([5, 8]),
        readiness=np.array([[True, False], [True, True]]),
        preopen_cutoff=4,
    )
    np.testing.assert_array_equal(result["observed_bars"], [[5, 2], [8, 8]])
    np.testing.assert_allclose(result["observed_fraction"], [[1.0, 0.4], [1.0, 1.0]])
    np.testing.assert_array_equal(
        result["minutes_since_most_recent_observed_bar"], [[0, 1], [0, 0]]
    )
    np.testing.assert_allclose(
        result["preopen_observed_fraction"], [[1.0, 0.5], [1.0, 1.0]]
    )
    assert not result["ready"][0, 1]
    future_mutated = observed.copy()
    future_mutated[0, :, 5:] = True
    future_mutated[1, :, 8:] = True
    unchanged = causal_observation_completeness(
        future_mutated,
        np.array([0, 1]),
        np.array([5, 8]),
        readiness=np.array([[True, False], [True, True]]),
        preopen_cutoff=4,
    )
    for name in result:
        np.testing.assert_array_equal(result[name], unchanged[name])


def test_global_completeness_uses_model_window_and_fixed_preopen_prefix() -> None:
    observed = np.zeros((1, 2, 500), dtype=bool)
    observed[0, 0, 10] = True
    observed[0, 1, 120] = True
    arguments = (
        np.array([0]),
        np.array([450]),
    )
    result = causal_observation_completeness(
        observed,
        *arguments,
        preopen_cutoff=330,
        current_window_length=345,
    )
    np.testing.assert_array_equal(result["observed_bars"], [[0, 1]])
    np.testing.assert_allclose(result["observed_fraction"], [[0.0, 1 / 345]])
    assert np.isnan(result["minutes_since_most_recent_observed_bar"][0, 0])
    assert result["minutes_since_most_recent_observed_bar"][0, 1] == 329
    np.testing.assert_allclose(
        result["preopen_observed_fraction"], [[1 / 330, 1 / 330]]
    )

    before_window = observed.copy()
    before_window[0, 0, 50] = True
    before_window[0, 1, 100] = True
    before_result = causal_observation_completeness(
        before_window,
        *arguments,
        preopen_cutoff=330,
        current_window_length=345,
    )
    for name in (
        "observed_bars",
        "observed_fraction",
        "minutes_since_most_recent_observed_bar",
    ):
        np.testing.assert_allclose(result[name], before_result[name], equal_nan=True)
    assert not np.array_equal(
        result["preopen_observed_fraction"],
        before_result["preopen_observed_fraction"],
    )

    inside_window = observed.copy()
    inside_window[0, 0, 449] = True
    inside_result = causal_observation_completeness(
        inside_window,
        *arguments,
        preopen_cutoff=330,
        current_window_length=345,
    )
    np.testing.assert_array_equal(inside_result["observed_bars"], [[1, 1]])
    assert inside_result["minutes_since_most_recent_observed_bar"][0, 0] == 0
    np.testing.assert_array_equal(
        result["preopen_observed_fraction"],
        inside_result["preopen_observed_fraction"],
    )

    after_cutoff = observed.copy()
    after_cutoff[:, :, 450:] = True
    after_result = causal_observation_completeness(
        after_cutoff,
        *arguments,
        preopen_cutoff=330,
        current_window_length=345,
    )
    for name in result:
        np.testing.assert_array_equal(result[name], after_result[name])


def test_attribution_inference_is_validation_only(monkeypatch, tmp_path) -> None:
    inputs = _inputs()
    observed_split: list[str] = []

    def collect(run_dir, split, *, identity_cache=None):
        assert identity_cache is None
        observed_split.append(split)
        observations = SimpleNamespace(
            predictions=inputs.predictions,
            targets=inputs.targets,
            raw_returns=inputs.raw_returns,
            label_mask=inputs.label_mask,
            date_idx=inputs.date_idx,
            decision_idx=inputs.decision_idx,
        )
        return observations, {}, [], {}, tmp_path

    monkeypatch.setattr(attribution, "collect_neural_evaluation", collect)
    monkeypatch.setattr(attribution, "_security_ids", lambda _: inputs.security_ids)
    monkeypatch.setattr(
        attribution,
        "_opening_diagnostics",
        lambda _store, _dates, _decisions: inputs.opening,
    )
    loaded = attribution.load_attribution_inputs(tmp_path / "run")
    assert observed_split == ["validation"]
    assert loaded.run_name == "run"


def test_bootstrap_output_reports_method_parameters(monkeypatch) -> None:
    inputs = _inputs(sample_count=18)
    monkeypatch.setattr(attribution, "BOOTSTRAP_REPLICATIONS", 20)
    frame = bootstrap_summary(inputs)
    assert set(frame.get_column("scope")) == {
        "time_of_day_5m",
        "time_of_day_30m",
        "horizon_attribution",
    }
    assert set(frame.get_column("block_trading_days")) == {5}
    assert set(frame.get_column("seed")) == {20260805}


def test_analysis_surface_is_validation_only() -> None:
    args = attribution.parse_args(["--run-dir", "run", "--output-dir", "output"])
    assert not hasattr(args, "split")
