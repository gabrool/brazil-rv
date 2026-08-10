from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from brazil_rv.preprocessing.contract import (
    DECISION_EQUITY_INDICES,
    DECISION_GLOBAL_INDICES,
    DYNAMIC_CHANNELS,
    GLOBAL_CONTEXT_SYMBOLS,
    LOCAL_CONTEXT_SYMBOLS,
    SLOW_CHANNELS,
)

from brazil_rv.modeling.analyze_stock_time_attribution import (
    _atomic_write_json,
    _build_context_time_deltas,
    _build_core_outputs,
    _coverage_summary,
    _build_liquidity_outputs,
    _build_opening_regimes,
    _daily_grid,
    _independent_bucket_ic,
    _load_analysis_metadata,
    _paired_delta_row,
    _point_in_time_bucket_contribution_vector,
    _record_artifact,
    _opening_condition_masks,
    _stock_identity_rows,
    _with_economic_ratios,
    adaptive_liquidity_buckets,
    additive_spearman_contributions,
    aggregate_additive_contributions,
    causal_observation_completeness,
    deterministic_liquidity_buckets,
    economic_stock_attribution,
    economic_window_accounting,
    learn_overnight_thresholds,
    moving_block_bootstrap,
    moving_block_bootstrap_matrix,
    moving_block_bootstrap_indices,
    named_time_scopes,
    overnight_regimes,
    parse_args,
    run_analysis,
    per_stock_time_series_skill,
    primary_time_bins,
    stock_contribution_opportunity_accounting,
    standardized_rank_scores,
)
from brazil_rv.modeling.stage3_context_addition import (
    STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION,
    STAGE3_LOGICAL_CONFIGURATION_ORDER,
    STAGE3_SEEDS,
)
from brazil_rv.modeling.stock_time_cache import (
    CACHE_VERSION,
    INFERENCE_CODE_PATHS,
    METRIC_REPRODUCTION_DAILY_IC_ABSOLUTE_TOLERANCE,
    METRIC_REPRODUCTION_DAILY_THRESHOLDS,
    METRIC_REPRODUCTION_ECONOMIC_RETURN_ABSOLUTE_TOLERANCE,
    METRIC_REPRODUCTION_GATE_SCHEMA_VERSION,
    METRIC_REPRODUCTION_HORIZON_IC_ABSOLUTE_TOLERANCE,
    METRIC_REPRODUCTION_PRIMARY_IC_ABSOLUTE_TOLERANCE,
    METRIC_REPRODUCTION_TURNOVER_ABSOLUTE_TOLERANCE,
    adopt_or_infer_caches as _adopt_or_infer_caches,
    atomic_write_npy as _atomic_write_npy,
    job_cache_identity as _job_cache_identity,
    metric_reproduction_gate,
    metric_reproduction_thresholds,
    remove_recognized_partial_cache as _remove_recognized_partial_cache,
    validate_cache_manifest as _validate_cache_manifest,
    validate_metric_reproduction_gate as _validate_metric_reproduction_gate,
)
from brazil_rv.modeling.stock_time_inference import (
    AnalysisInputs,
    Stage3AnalysisJob,
    inference_code_identity as _inference_code_identity,
    reject_test_derived_path as _reject_test_derived_path,
)
from brazil_rv.modeling.engine import EvaluationObservations
from brazil_rv.modeling.metrics import average_ranks
from brazil_rv.modeling.process_lock import (
    PRODUCTION_TRAINING_LOCK,
    ProcessLockLease,
    exclusive_process_lock,
)


def _random_arrays(
    sample_count: int = 4,
    equity_count: int = 30,
    horizon_count: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(17)
    predictions = generator.normal(
        size=(sample_count, equity_count, horizon_count)
    ).astype(np.float32)
    targets = generator.normal(size=(sample_count, equity_count, horizon_count)).astype(
        np.float32
    )
    mask = np.ones_like(predictions, dtype=bool)
    return predictions, targets, mask


def _metric_payloads() -> tuple[dict[str, object], list[dict[str, object]]]:
    summary: dict[str, object] = {
        "primary_score": 0.0,
        "horizons": [
            {
                "horizon_minutes": horizon,
                "mean_daily_spearman_ic": 0.0,
            }
            for horizon in (30, 60, 120)
        ],
    }
    daily_rows = [
        {
            "date_idx": date_index,
            "horizon_minutes": horizon,
            "spearman_ic": 0.0,
            "rank_target_pearson_ic": 0.0,
            "top_return": 0.0,
            "bottom_return": 0.0,
            "top_minus_bottom": 0.0,
            "long_only_top": 0.0,
            "one_way_turnover": 0.0,
        }
        for date_index in (793, 794)
        for horizon in (30, 60, 120)
    ]
    return summary, daily_rows


def _write_recorded_metric_artifacts(
    run_dir: Path,
    summary: dict[str, object],
    daily_rows: list[dict[str, object]],
    trade_dates: dict[int, date] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(run_dir / "validation_metrics.json", summary)
    if trade_dates is None:
        trade_dates = {
            date_index: date(2024, 9, 18) + timedelta(days=position)
            for position, date_index in enumerate(
                sorted({int(row["date_idx"]) for row in daily_rows})
            )
        }
    recorded_rows = [
        {"trade_date": trade_dates[int(row["date_idx"])], **copy.deepcopy(row)}
        for row in daily_rows
    ]
    pl.DataFrame(recorded_rows).write_parquet(
        run_dir / "validation_daily_metrics.parquet"
    )


def _valid_metric_gate(tmp_path: Path) -> dict[str, object]:
    summary, daily_rows = _metric_payloads()
    run_dir = tmp_path / "valid-metric-gate"
    _write_recorded_metric_artifacts(run_dir, summary, daily_rows)
    return metric_reproduction_gate(run_dir, summary, daily_rows)


def test_average_rank_ties_are_stable_exact_midranks() -> None:
    values = np.asarray([2.0, 1.0, 2.0, 4.0, 2.0, 1.0])
    np.testing.assert_array_equal(
        average_ranks(values),
        np.asarray([3.0, 0.5, 3.0, 5.0, 3.0, 0.5]),
    )


def test_additive_spearman_reconstructs_with_ties_and_masking() -> None:
    predictions, targets, mask = _random_arrays(sample_count=2, equity_count=35)
    predictions[0, :4, 0] = 1.0
    targets[0, 5:9, 0] = -2.0
    mask[:, -5:, 1] = False
    result = additive_spearman_contributions(predictions, targets, mask)
    np.testing.assert_allclose(
        result.contributions.sum(axis=1),
        result.sample_ic,
        atol=5e-12,
        rtol=0.0,
        equal_nan=True,
    )
    assert np.count_nonzero(result.contributions[:, -5:, 1]) == 0


def test_additive_spearman_skips_small_and_zero_variance_cross_sections() -> None:
    predictions, targets, mask = _random_arrays(sample_count=2)
    mask[0, :2] = False
    predictions[1, :, 0] = 3.0
    result = additive_spearman_contributions(predictions, targets, mask)
    assert np.isnan(result.sample_ic[0]).all()
    assert np.isnan(result.sample_ic[1, 0])
    assert np.isnan(result.contributions[0]).all()
    assert np.isnan(result.contributions[1, :, 0]).all()


def test_daily_horizon_primary_weighting_uses_dates_not_raw_samples() -> None:
    contributions = np.zeros((3, 2, 1), dtype=np.float64)
    contributions[0, 0, 0] = 1.0
    contributions[1, 0, 0] = 3.0
    contributions[2, 0, 0] = 10.0
    sample_ic = contributions.sum(axis=1)
    result = aggregate_additive_contributions(
        contributions, sample_ic, np.asarray([0, 0, 1])
    )
    assert result["primary_ic"] == pytest.approx(6.0)
    assert result["primary_contributions"][0] == pytest.approx(6.0)
    assert result["primary_ic"] != pytest.approx(sample_ic.mean())


def test_time_bins_merge_final_singleton_and_opening_scopes_are_exact() -> None:
    bins = primary_time_bins()
    assert len(bins) == 9
    assert all(len(group) == 6 for group in bins[:-1])
    assert bins[-1] == tuple(range(48, 55))
    assert tuple(index for group in bins for index in group) == tuple(range(55))
    scopes = named_time_scopes()
    assert scopes["opening_30"] == tuple(range(6))
    assert scopes["opening_60"] == tuple(range(12))
    assert scopes["rest_of_day"] == tuple(range(12, 55))


def test_security_identity_is_stable_while_ticker_remains_display_metadata() -> None:
    index = pl.DataFrame(
        {
            "equity_slot": [0],
            "security_id": ["permanent-security"],
            "isin": ["BR0000000001"],
            "latest_ticker": ["OLD3"],
            "xp_symbol": ["Issuer"],
        }
    )
    before = _stock_identity_rows(index)[0]
    after = _stock_identity_rows(
        index.with_columns(pl.lit("NEW3").alias("latest_ticker"))
    )[0]
    assert before["security_id"] == after["security_id"] == "permanent-security"
    assert before["isin"] == after["isin"] == "BR0000000001"
    assert before["display_ticker"] == "OLD3"
    assert after["display_ticker"] == "NEW3"


def test_five_minute_and_stock_time_grid_reconstructs_cells() -> None:
    predictions, targets, mask = _random_arrays(sample_count=4)
    additive = additive_spearman_contributions(predictions, targets, mask)
    dates, grid = _daily_grid(
        additive.sample_ic,
        np.asarray([5, 5, 6, 6]),
        np.asarray([0, 1, 0, 1]),
    )
    assert dates.tolist() == [5, 6]
    np.testing.assert_allclose(grid[0, :2], additive.sample_ic[:2])
    for sample in range(4):
        np.testing.assert_allclose(
            additive.contributions[sample].sum(axis=0), additive.sample_ic[sample]
        )


def test_economic_spread_and_turnover_attribution_reconstruct_exactly() -> None:
    sample_count, equity_count, horizon_count = 3, 30, 1
    predictions = np.empty((sample_count, equity_count, horizon_count), np.float32)
    predictions[0, :, 0] = np.arange(equity_count)
    predictions[1, :, 0] = np.roll(np.arange(equity_count), 3)
    predictions[2, :, 0] = np.arange(equity_count)[::-1]
    returns = np.linspace(-0.03, 0.03, equity_count, dtype=np.float32)[
        None, :, None
    ].repeat(sample_count, axis=0)
    mask = np.ones_like(predictions, dtype=bool)
    result = economic_stock_attribution(
        predictions,
        returns,
        mask,
        np.asarray([0, 0, 0]),
        np.asarray([0, 1, 2]),
    )
    for sample in range(sample_count):
        ranked = np.argsort(predictions[sample, :, 0], kind="mergesort")
        k = equity_count // 10
        expected = (
            returns[sample, ranked[-k:], 0].mean()
            - returns[sample, ranked[:k], 0].mean()
        )
        assert result.return_contributions[sample, :, 0].sum() == pytest.approx(
            expected
        )
    assert result.flat_entry_turnover[0, :, 0].sum() == pytest.approx(1.0)
    assert result.flat_exit_turnover[2, :, 0].sum() == pytest.approx(1.0)
    assert result.flat_entry_turnover[1:].sum() == 0.0
    assert result.flat_exit_turnover[:2].sum() == 0.0
    for sample in (1, 2):
        expected = (
            0.5
            * np.abs(
                result.weights[sample, :, 0] - result.weights[sample - 1, :, 0]
            ).sum()
        )
        assert result.intraday_turnover[sample, :, 0].sum() == pytest.approx(expected)


def test_per_stock_time_series_skill_and_minimum_coverage() -> None:
    date_count, equity_count = 8, 31
    base = np.linspace(-1.0, 1.0, equity_count, dtype=np.float32)
    predictions = np.stack([np.roll(base, day) for day in range(date_count)])[
        :, :, None
    ]
    targets = predictions.copy()
    mask = np.ones_like(predictions, dtype=bool)
    mask[:6, 0, 0] = False
    result = per_stock_time_series_skill(
        predictions,
        targets,
        mask,
        np.arange(date_count),
        minimum_days=5,
        minimum_coverage=0.5,
        bootstrap_replications=50,
    )
    assert np.isnan(result["skill"][0, 0])
    np.testing.assert_allclose(result["skill"][1:, 0], 1.0, atol=1e-12)


def test_liquidity_buckets_are_deterministic_with_ties_and_adaptive() -> None:
    values = np.repeat(np.arange(5, dtype=np.float64), 12)
    eligible = np.ones(values.size, dtype=bool)
    first = deterministic_liquidity_buckets(values, eligible)
    second = deterministic_liquidity_buckets(values, eligible)
    np.testing.assert_array_equal(first, second)
    for value in np.unique(values):
        assert np.unique(first[values == value]).size == 1
    adaptive, count = adaptive_liquidity_buckets(
        np.arange(90, dtype=np.float64), eligible=np.ones(90, dtype=bool)
    )
    assert count == 3
    assert np.bincount(adaptive).tolist() == [30, 30, 30]


def test_overnight_thresholds_depend_only_on_supplied_training_values() -> None:
    training = np.linspace(-2.0, 2.0, 100)
    thresholds = learn_overnight_thresholds(training)
    validation = np.asarray([-10.0, 0.0, 10.0])
    regimes = overnight_regimes(validation, thresholds)
    assert regimes["large"].tolist() == [True, False, True]
    assert thresholds == learn_overnight_thresholds(training.copy())


def test_context_completeness_is_causal_and_staleness_is_exact() -> None:
    observed = np.zeros((1, 2, 20), dtype=bool)
    observed[0, 0, [0, 4, 9, 15]] = True
    observed[0, 1, [2, 8, 19]] = True
    before = causal_observation_completeness(
        observed,
        np.asarray([0]),
        np.asarray([10]),
        preopen_cutoff=5,
        recent_minutes=5,
    )
    observed[:, :, 10:] = ~observed[:, :, 10:]
    after = causal_observation_completeness(
        observed,
        np.asarray([0]),
        np.asarray([10]),
        preopen_cutoff=5,
        recent_minutes=5,
    )
    for key in before:
        np.testing.assert_allclose(before[key], after[key], equal_nan=True)
    assert before["minutes_since_most_recent_observed_bar"][0, 0] == 0
    assert before["minutes_since_most_recent_observed_bar"][0, 1] == 1


def test_moving_block_bootstrap_is_deterministic_and_preserves_blocks() -> None:
    first = moving_block_bootstrap_indices(12, replications=20, block_length=5, seed=8)
    second = moving_block_bootstrap_indices(12, replications=20, block_length=5, seed=8)
    np.testing.assert_array_equal(first, second)
    for row in first:
        assert np.all(np.diff(row[:5]) == 1)
        assert np.all(np.diff(row[5:10]) == 1)
    values = np.arange(12, dtype=np.float64)
    assert moving_block_bootstrap(
        values, replications=20, block_length=5, seed=8
    ) == moving_block_bootstrap(values, replications=20, block_length=5, seed=8)


def test_same_seed_delta_rejects_misalignment_and_is_deterministic() -> None:
    values = np.linspace(-0.1, 0.1, 10)
    kwargs = {
        "logical": "core_plus_es",
        "seed": 11,
        "scope_type": "named_scope",
        "scope_name": "opening_30",
        "decisions": tuple(range(6)),
        "horizon_minutes": 0,
        "current_ic": values + 0.02,
        "core_ic": values,
        "current_spread": values + 0.01,
        "core_spread": values,
        "current_turnover": np.ones(10),
        "core_turnover": np.full(10, 0.9),
        "regime": None,
        "freshness": None,
        "bootstrap_replications": 30,
        "bootstrap_seed": 4,
    }
    first = _paired_delta_row(**kwargs)
    second = _paired_delta_row(**kwargs)
    assert first == second
    assert first["mean_paired_ic_delta"] == pytest.approx(0.02)
    kwargs["core_ic"] = values[:-1]
    with pytest.raises(ValueError, match="misaligned"):
        _paired_delta_row(**kwargs)


def test_metric_reproduction_gate_accepts_exact_gh200_diagnostic(
    tmp_path: Path,
) -> None:
    recorded_summary, recorded_daily = _metric_payloads()
    recorded_summary["primary_score"] = 0.04176565907570362
    recorded_horizons = recorded_summary["horizons"]
    assert isinstance(recorded_horizons, list)
    for row, value in zip(recorded_horizons, (0.02, 0.03, 0.04), strict=True):
        assert isinstance(row, dict)
        row["mean_daily_spearman_ic"] = value
    turnover_row = next(
        row
        for row in recorded_daily
        if row["date_idx"] == 794 and row["horizon_minutes"] == 60
    )
    turnover_row["one_way_turnover"] = 1.1428170594837261
    _write_recorded_metric_artifacts(
        tmp_path,
        recorded_summary,
        recorded_daily,
        {793: date(2024, 9, 18), 794: date(2024, 9, 19)},
    )

    recomputed_summary = copy.deepcopy(recorded_summary)
    recomputed_summary["primary_score"] = 0.04176573994209834
    recomputed_horizons = recomputed_summary["horizons"]
    assert isinstance(recomputed_horizons, list)
    for row, difference in zip(
        recomputed_horizons,
        (
            3.701404472505887e-08,
            7.025438509417059e-08,
            2.0935884377515368e-07,
        ),
        strict=True,
    ):
        assert isinstance(row, dict)
        row["mean_daily_spearman_ic"] = (
            float(row["mean_daily_spearman_ic"]) + difference
        )
    recomputed_daily = copy.deepcopy(recorded_daily)
    recomputed_daily[0]["spearman_ic"] = 2.7086860318291384e-05
    recomputed_daily[1]["rank_target_pearson_ic"] = 9.971929088893605e-06
    recomputed_turnover = next(
        row
        for row in recomputed_daily
        if row["date_idx"] == 794 and row["horizon_minutes"] == 60
    )
    recomputed_turnover["one_way_turnover"] = 1.1411335578002246

    gate = metric_reproduction_gate(tmp_path, recomputed_summary, recomputed_daily)
    assert gate["schema_version"] == METRIC_REPRODUCTION_GATE_SCHEMA_VERSION
    assert gate["passed"] is True
    assert gate["thresholds"] == metric_reproduction_thresholds()
    assert gate["primary_ic"]["absolute_difference"] == pytest.approx(
        8.086639471938106e-08
    )
    assert [
        gate["horizons"][f"{horizon}m"]["absolute_difference"]
        for horizon in (30, 60, 120)
    ] == pytest.approx(
        [
            3.701404472505887e-08,
            7.025438509417059e-08,
            2.0935884377515368e-07,
        ]
    )
    assert gate["daily_metrics"]["spearman_ic"][
        "maximum_absolute_difference"
    ] == pytest.approx(2.7086860318291384e-05)
    assert gate["daily_metrics"]["rank_target_pearson_ic"][
        "maximum_absolute_difference"
    ] == pytest.approx(9.971929088893605e-06)
    turnover = gate["daily_metrics"]["one_way_turnover"]
    assert turnover["maximum_absolute_difference"] == pytest.approx(
        0.0016835016835015093
    )
    assert turnover["worst_row"] == {
        "date_idx": 794,
        "trade_date": "2024-09-19",
        "horizon_minutes": 60,
        "recorded": 1.1428170594837261,
        "recomputed": 1.1411335578002246,
        "absolute_difference": pytest.approx(0.0016835016835015093),
    }


@pytest.mark.parametrize("location", ("primary", "horizon"))
def test_metric_reproduction_scalar_ic_boundaries(
    tmp_path: Path, location: str
) -> None:
    recorded_summary, daily_rows = _metric_payloads()
    _write_recorded_metric_artifacts(tmp_path, recorded_summary, daily_rows)
    threshold = (
        METRIC_REPRODUCTION_PRIMARY_IC_ABSOLUTE_TOLERANCE
        if location == "primary"
        else METRIC_REPRODUCTION_HORIZON_IC_ABSOLUTE_TOLERANCE
    )

    at_boundary = copy.deepcopy(recorded_summary)
    if location == "primary":
        at_boundary["primary_score"] = threshold
    else:
        horizons = at_boundary["horizons"]
        assert isinstance(horizons, list) and isinstance(horizons[0], dict)
        horizons[0]["mean_daily_spearman_ic"] = threshold
    assert metric_reproduction_gate(tmp_path, at_boundary, daily_rows)["passed"]

    above_boundary = copy.deepcopy(at_boundary)
    above = math.nextafter(threshold, math.inf)
    if location == "primary":
        above_boundary["primary_score"] = above
    else:
        horizons = above_boundary["horizons"]
        assert isinstance(horizons, list) and isinstance(horizons[0], dict)
        horizons[0]["mean_daily_spearman_ic"] = above
    with pytest.raises(ValueError, match="validation metric parity"):
        metric_reproduction_gate(tmp_path, above_boundary, daily_rows)


@pytest.mark.parametrize("metric", ("spearman_ic", "rank_target_pearson_ic"))
def test_metric_reproduction_daily_ic_boundaries(tmp_path: Path, metric: str) -> None:
    summary, recorded_daily = _metric_payloads()
    _write_recorded_metric_artifacts(tmp_path, summary, recorded_daily)
    at_boundary = copy.deepcopy(recorded_daily)
    at_boundary[0][metric] = METRIC_REPRODUCTION_DAILY_IC_ABSOLUTE_TOLERANCE
    assert metric_reproduction_gate(tmp_path, summary, at_boundary)["passed"]
    above_boundary = copy.deepcopy(at_boundary)
    above_boundary[0][metric] = math.nextafter(
        METRIC_REPRODUCTION_DAILY_IC_ABSOLUTE_TOLERANCE, math.inf
    )
    with pytest.raises(ValueError, match="validation metric parity"):
        metric_reproduction_gate(tmp_path, summary, above_boundary)


@pytest.mark.parametrize(
    "metric",
    ("top_return", "bottom_return", "top_minus_bottom", "long_only_top"),
)
def test_metric_reproduction_economic_return_boundaries(
    tmp_path: Path, metric: str
) -> None:
    summary, recorded_daily = _metric_payloads()
    _write_recorded_metric_artifacts(tmp_path, summary, recorded_daily)
    at_boundary = copy.deepcopy(recorded_daily)
    at_boundary[0][metric] = METRIC_REPRODUCTION_ECONOMIC_RETURN_ABSOLUTE_TOLERANCE
    assert metric_reproduction_gate(tmp_path, summary, at_boundary)["passed"]
    above_boundary = copy.deepcopy(at_boundary)
    above_boundary[0][metric] = math.nextafter(
        METRIC_REPRODUCTION_ECONOMIC_RETURN_ABSOLUTE_TOLERANCE,
        math.inf,
    )
    with pytest.raises(ValueError, match="validation metric parity"):
        metric_reproduction_gate(tmp_path, summary, above_boundary)


def test_metric_reproduction_turnover_boundaries(tmp_path: Path) -> None:
    summary, recorded_daily = _metric_payloads()
    _write_recorded_metric_artifacts(tmp_path, summary, recorded_daily)
    for accepted in (
        0.0016835016835015093,
        METRIC_REPRODUCTION_TURNOVER_ABSOLUTE_TOLERANCE,
    ):
        recomputed = copy.deepcopy(recorded_daily)
        recomputed[0]["one_way_turnover"] += accepted
        assert metric_reproduction_gate(tmp_path, summary, recomputed)["passed"]
    recomputed = copy.deepcopy(recorded_daily)
    recomputed[0]["one_way_turnover"] += math.nextafter(
        METRIC_REPRODUCTION_TURNOVER_ABSOLUTE_TOLERANCE, math.inf
    )
    with pytest.raises(ValueError, match="validation metric parity"):
        metric_reproduction_gate(tmp_path, summary, recomputed)


@pytest.mark.parametrize("metric", ("spearman_ic", "top_return"))
def test_turnover_allowance_does_not_leak_to_other_metrics(
    tmp_path: Path, metric: str
) -> None:
    summary, recorded_daily = _metric_payloads()
    _write_recorded_metric_artifacts(tmp_path, summary, recorded_daily)
    recomputed = copy.deepcopy(recorded_daily)
    recomputed[0][metric] = 0.0016835016835015093
    with pytest.raises(ValueError, match="validation metric parity"):
        metric_reproduction_gate(tmp_path, summary, recomputed)


@pytest.mark.parametrize(
    "mutation",
    (
        "row_count",
        "row_order",
        "date_idx",
        "horizon",
        "missing_column",
        "extra_column",
        "nan_pattern",
        "infinity",
    ),
)
def test_metric_reproduction_rejects_exact_structure_and_finiteness_mismatches(
    tmp_path: Path, mutation: str
) -> None:
    summary, recorded_daily = _metric_payloads()
    _write_recorded_metric_artifacts(tmp_path, summary, recorded_daily)
    recomputed = copy.deepcopy(recorded_daily)
    if mutation == "row_count":
        recomputed.pop()
    elif mutation == "row_order":
        recomputed.reverse()
    elif mutation == "date_idx":
        recomputed[0]["date_idx"] = 999
    elif mutation == "horizon":
        recomputed[0]["horizon_minutes"] = 45
    elif mutation == "missing_column":
        for row in recomputed:
            del row["top_return"]
    elif mutation == "extra_column":
        for row in recomputed:
            row["unexpected_metric"] = 0.0
    elif mutation == "nan_pattern":
        recomputed[0]["spearman_ic"] = math.nan
    elif mutation == "infinity":
        recomputed[0]["spearman_ic"] = math.inf
    else:
        raise AssertionError(mutation)
    with pytest.raises(ValueError, match="validation metric parity"):
        metric_reproduction_gate(tmp_path, summary, recomputed)


def test_metric_reproduction_allows_only_matching_nan_patterns(
    tmp_path: Path,
) -> None:
    summary, recorded_daily = _metric_payloads()
    recorded_daily[0]["spearman_ic"] = math.nan
    _write_recorded_metric_artifacts(tmp_path, summary, recorded_daily)
    gate = metric_reproduction_gate(tmp_path, summary, recorded_daily)
    assert gate["passed"] is True
    assert json.dumps(gate, allow_nan=False)


def test_metric_reproduction_failure_reports_every_failed_comparison(
    tmp_path: Path,
) -> None:
    summary, recorded_daily = _metric_payloads()
    _write_recorded_metric_artifacts(tmp_path, summary, recorded_daily)
    recomputed = copy.deepcopy(recorded_daily)
    recomputed[0]["spearman_ic"] = 0.001
    recomputed[1]["top_return"] = 0.002
    with pytest.raises(ValueError, match="validation metric parity") as exc_info:
        metric_reproduction_gate(tmp_path, summary, recomputed)
    details = json.loads(str(exc_info.value).split("\n", 1)[1])
    failures = {row["metric"]: row for row in details["failures"]}
    assert set(failures) == {"spearman_ic", "top_return"}
    for metric, expected in (("spearman_ic", 0.001), ("top_return", 0.002)):
        failure = failures[metric]
        assert failure["recorded"] == 0.0
        assert failure["recomputed"] == expected
        assert failure["maximum_absolute_difference"] == expected
        assert failure["difference"] == expected
        assert failure["threshold"] == METRIC_REPRODUCTION_DAILY_THRESHOLDS[metric]
        assert failure["passed"] is False
        assert failure["worst_row"]["date_idx"] == 793
        assert failure["worst_row"]["trade_date"] == "2024-09-18"
        assert failure["worst_row"]["horizon_minutes"] in (30, 60)


def test_cache_manifest_rejects_nested_identity_hash_shape_and_dtype(
    tmp_path: Path,
) -> None:
    valid_gate = _valid_metric_gate(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    predictions = np.zeros((2, 3, 1), dtype=np.float32)
    prediction_path = cache / "predictions.npy"
    _atomic_write_npy(prediction_path, predictions)
    identity = {
        "cache_name": "stock_time_attribution",
        "cache_version": CACHE_VERSION,
        "inference_code_sha256": {
            path: f"hash-{position}"
            for position, path in enumerate(INFERENCE_CODE_PATHS)
        },
        "seed": 11,
        "context": {"key": "core", "hash": "abc"},
        "prediction_shape": [2, 3, 1],
        "prediction_dtype": "float32",
    }
    manifest_path = cache / "manifest.json"

    def write_manifest(cache_identity: dict[str, object]) -> None:
        _atomic_write_json(
            manifest_path,
            {
                "status": "completed",
                "identity": cache_identity,
                "creation_provenance": {
                    "analyzer_worktree_clean": True,
                    "analyzer_source_sha256": "source",
                    "analyzer_git_commit_sha": "commit",
                },
                "prediction_file": {
                    "name": prediction_path.name,
                    "sha256": __import__("hashlib")
                    .sha256(prediction_path.read_bytes())
                    .hexdigest(),
                },
                "metric_reproduction_gate": copy.deepcopy(valid_gate),
            },
        )

    write_manifest(identity)
    path, _ = _validate_cache_manifest(manifest_path, identity)
    assert path == prediction_path
    wrong = {**identity, "context": {"key": "core", "hash": "wrong"}}
    with pytest.raises(ValueError, match="identity"):
        _validate_cache_manifest(manifest_path, wrong)
    wrong_shape = {**identity, "prediction_shape": [2, 4, 1]}
    write_manifest(wrong_shape)
    with pytest.raises(ValueError, match="shape or dtype"):
        _validate_cache_manifest(manifest_path, wrong_shape)
    wrong_dtype = {**identity, "prediction_dtype": "float64"}
    write_manifest(wrong_dtype)
    with pytest.raises(ValueError, match="shape or dtype"):
        _validate_cache_manifest(manifest_path, wrong_dtype)
    legacy_identity = {**identity, "cache_version": CACHE_VERSION - 1}
    write_manifest(legacy_identity)
    with pytest.raises(ValueError, match="version mismatch"):
        _validate_cache_manifest(manifest_path, legacy_identity)
    _atomic_write_json(
        manifest_path,
        {
            "status": "completed",
            "identity": identity,
        },
    )
    with pytest.raises(ValueError, match="provenance"):
        _validate_cache_manifest(manifest_path, identity)
    _atomic_write_json(
        manifest_path,
        {
            "status": "completed",
            "identity": identity,
            "creation_provenance": {
                "analyzer_worktree_clean": True,
                "analyzer_source_sha256": "source",
                "analyzer_git_commit_sha": "commit",
            },
            "prediction_file": {
                "name": prediction_path.name,
                "sha256": __import__("hashlib")
                .sha256(prediction_path.read_bytes())
                .hexdigest(),
            },
            "test_data_used": True,
        },
    )
    with pytest.raises(ValueError, match="test-derived"):
        _validate_cache_manifest(manifest_path, identity)
    write_manifest(identity)
    prediction_path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        _validate_cache_manifest(manifest_path, identity)


@pytest.mark.parametrize(
    ("failure_stage", "error_type", "message"),
    (
        ("inference", RuntimeError, "synthetic inference failure"),
        ("metric_reproduction", ValueError, "validation metric parity"),
    ),
)
def test_inference_failure_is_recorded_without_mutating_inputs_or_writing_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
    error_type: type[Exception],
    message: str,
) -> None:
    run_dir = tmp_path / "run"
    feature_store = tmp_path / "feature-store"
    run_dir.mkdir()
    feature_store.mkdir()
    run_marker = run_dir / "immutable.txt"
    feature_marker = feature_store / "immutable.txt"
    run_marker.write_text("run", encoding="utf-8")
    feature_marker.write_text("feature", encoding="utf-8")
    job = Stage3AnalysisJob(
        position=0,
        logical_configuration="core",
        context_ablation=STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION["core"],
        seed=11,
        run_dir=run_dir,
        run_manifest_path=run_dir / "run_manifest.json",
        run_manifest_sha256="run-sha",
        checkpoint_path=run_dir / "best.pt",
        checkpoint_sha256="checkpoint-sha",
        producing_git_commit_sha="producer",
        manifest={},
    )
    inputs = AnalysisInputs(
        state_path=tmp_path / "stage3-state.json",
        state_sha256="state-sha",
        state={},
        configuration={},
        feature_store=feature_store,
        feature_identity={"manifest_sha256": "feature-sha"},
        feature_manifest={},
        sample_index=pl.DataFrame(),
        validation_rows=pl.DataFrame(
            {"sample_id": [7], "date_idx": [0], "decision_idx": [0]}
        ),
        jobs=(job,),
        analyzer_git_commit_sha="analyzer",
        analyzer_worktree_clean=True,
        analyzer_source_sha256="source-sha",
    )
    state = {
        "status": "running",
        "jobs": [{"status": "pending"}],
        "artifacts": {},
        "pending_artifact": None,
    }
    state_path = tmp_path / "analysis_state.json"

    monkeypatch.setattr(
        "brazil_rv.modeling.stock_time_cache.validate_runtime",
        lambda: None,
    )

    def fail_inference(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("synthetic inference failure")

    if failure_stage == "inference":
        monkeypatch.setattr(
            "brazil_rv.modeling.stock_time_cache.collect_neural_evaluation",
            fail_inference,
        )
    else:
        shape = (1, 158, 3)
        observations = EvaluationObservations(
            sample_id=np.asarray([7], dtype=np.int64),
            date_idx=np.asarray([0], dtype=np.int64),
            decision_idx=np.asarray([0], dtype=np.int64),
            predictions=np.zeros(shape, dtype=np.float32),
            targets=np.zeros(shape, dtype=np.float32),
            raw_returns=np.zeros(shape, dtype=np.float32),
            label_mask=np.ones(shape, dtype=bool),
        )
        evaluation = SimpleNamespace(
            observations=observations, summary={}, daily_rows=[]
        )

        def successful_inference(*args: object, **kwargs: object) -> SimpleNamespace:
            del args, kwargs
            return evaluation

        monkeypatch.setattr(
            "brazil_rv.modeling.stock_time_cache.collect_neural_evaluation",
            successful_inference,
        )
        recorded_summary, recorded_daily = _metric_payloads()
        _write_recorded_metric_artifacts(run_dir, recorded_summary, recorded_daily)
        evaluation.summary = copy.deepcopy(recorded_summary)
        evaluation.summary["primary_score"] = (
            METRIC_REPRODUCTION_PRIMARY_IC_ABSOLUTE_TOLERANCE * 2.0
        )
        evaluation.daily_rows = recorded_daily
    cache_root = tmp_path / "output"
    with pytest.raises(error_type, match=message):
        _adopt_or_infer_caches(
            cache_root,
            inputs,
            state,
            state_path,
        )
    assert state["jobs"][0]["status"] == "failed"
    assert state["jobs"][0]["error"].startswith(f"{error_type.__name__}: ")
    assert message in state["jobs"][0]["error"]
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["jobs"][0]["status"] == "failed"
    assert run_marker.read_text(encoding="utf-8") == "run"
    assert feature_marker.read_text(encoding="utf-8") == "feature"
    assert not cache_root.exists()


def test_interrupted_cache_recovery_removes_only_recognized_partial_files(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "predictions.npy").write_bytes(b"complete-but-uncommitted")
    (cache / "manifest.json.tmp").write_bytes(b"partial-manifest")
    _remove_recognized_partial_cache(
        cache,
        {"predictions.npy", "manifest.json"},
    )
    assert not any(cache.iterdir())

    unexpected = cache / "unknown.bin"
    unexpected.write_bytes(b"ambiguous")
    with pytest.raises(ValueError, match="unexpected entries"):
        _remove_recognized_partial_cache(
            cache,
            {"predictions.npy", "manifest.json"},
        )
    assert unexpected.is_file()


def test_artifact_promotion_resumes_from_hash_bound_pending_state(
    tmp_path: Path,
) -> None:
    final_path = tmp_path / "result.parquet"
    staged_path = tmp_path / "result.parquet.staged"
    staged_path.write_bytes(b"deterministic-artifact")
    digest = __import__("hashlib").sha256(staged_path.read_bytes()).hexdigest()
    state = {
        "artifacts": {},
        "pending_artifact": {
            "name": final_path.name,
            "path": str(final_path),
            "staged_path": str(staged_path),
            "sha256": digest,
            "bytes": staged_path.stat().st_size,
        },
    }
    state_path = tmp_path / "analysis_state.json"

    def must_not_rewrite(path: Path) -> None:
        del path
        raise AssertionError("pending artifact should be promoted without rewriting")

    _record_artifact(
        state,
        state_path,
        final_path.name,
        final_path,
        must_not_rewrite,
    )
    assert final_path.read_bytes() == b"deterministic-artifact"
    assert not staged_path.exists()
    assert state["pending_artifact"] is None
    assert state["artifacts"][final_path.name]["sha256"] == digest


def test_atomic_json_preserves_target_and_cleans_temp_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    _atomic_write_json(path, {"value": 1})

    def fail_replace(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(
        "brazil_rv.modeling.analyze_stock_time_attribution.os.replace", fail_replace
    )
    with pytest.raises(OSError, match="synthetic"):
        _atomic_write_json(path, {"value": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1}
    assert not path.with_name("state.json.tmp").exists()


def test_analysis_lock_prevents_duplicate_execution(tmp_path: Path) -> None:
    lock = tmp_path / "analysis.lock"
    with exclusive_process_lock(lock, "holder"):
        with pytest.raises(RuntimeError, match="already active"):
            with exclusive_process_lock(lock, "second"):
                raise AssertionError("duplicate lock was acquired")


def test_final_test_isolation_has_no_split_argument_and_rejects_test_paths() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--stage3-state",
                "state.json",
                "--output-dir",
                "out",
                "--scope",
                "core",
                "--split",
                "test",
            ]
        )
    with pytest.raises(ValueError, match="test-derived"):
        _reject_test_derived_path(
            Path("/model_runs/evaluations/test/result.json"), "synthetic"
        )


def test_mixed_valid_cells_are_nan_and_never_dilute_additive_aggregates() -> None:
    predictions, targets, mask = _random_arrays(sample_count=3)
    predictions[1, :, 0] = 1.0
    mask[1, :, 1:] = False
    result = additive_spearman_contributions(predictions, targets, mask)
    assert np.isnan(result.contributions[1]).all()
    aggregate = aggregate_additive_contributions(
        result.contributions,
        result.sample_ic,
        np.asarray([0, 0, 1], dtype=np.int64),
    )
    expected = np.nanmean(np.stack([result.sample_ic[0], result.sample_ic[2]]), axis=0)
    np.testing.assert_allclose(aggregate["horizon_ic"], expected, atol=5e-12)
    np.testing.assert_allclose(
        np.asarray(aggregate["horizon_contributions"]).sum(axis=0),
        aggregate["horizon_ic"],
        atol=5e-12,
    )


def test_population_rank_scores_hold_under_varying_universe_sizes() -> None:
    for size in (3, 7, 30):
        scores = standardized_rank_scores(np.arange(size, dtype=np.float64))
        assert scores.mean() == pytest.approx(0.0, abs=1e-15)
        assert np.std(scores, ddof=0) == pytest.approx(1.0, abs=1e-15)

    predictions, targets, mask = _random_arrays(
        sample_count=5, equity_count=40, horizon_count=1
    )
    mask[0, :10, 0] = False
    result = per_stock_time_series_skill(
        predictions,
        targets,
        mask,
        np.arange(5, dtype=np.int64),
        minimum_days=1,
        minimum_coverage=0.5,
        bootstrap_replications=8,
    )
    first = result["daily_prediction_scores"][0, 10:, 0]
    second = result["daily_prediction_scores"][1, :, 0]
    assert np.std(first, ddof=0) == pytest.approx(1.0)
    assert np.std(second, ddof=0) == pytest.approx(1.0)
    assert np.isnan(result["daily_prediction_scores"][0, :10, 0]).all()


def test_economic_window_accounting_uses_flat_boundaries_and_daily_sums() -> None:
    decision_count = 55
    equity_count = 30
    predictions = np.tile(
        np.arange(equity_count, dtype=np.float32),
        (decision_count, 1),
    )[:, :, None]
    returns = np.zeros_like(predictions)
    returns[:, :3, 0] = -0.01
    returns[:, -3:, 0] = 0.01
    mask = np.ones_like(predictions, dtype=bool)
    dates = np.zeros(decision_count, dtype=np.int64)
    decisions = np.arange(decision_count, dtype=np.int64)

    constant = economic_stock_attribution(predictions, returns, mask, dates, decisions)
    accounting = economic_window_accounting(constant, dates, decisions)
    assert accounting.daily_gross_contribution.sum() == pytest.approx(55 * 0.02)
    assert accounting.daily_entry_turnover.sum() == pytest.approx(1.0)
    assert accounting.daily_intraday_turnover.sum() == pytest.approx(0.0)
    assert accounting.daily_exit_turnover.sum() == pytest.approx(1.0)
    total = float(accounting.daily_total_turnover.sum())
    gross = float(accounting.daily_gross_contribution.sum())
    assert 10_000.0 * gross / total == pytest.approx(5_500.0)

    alternating_predictions = predictions.copy()
    alternating_predictions[1::2] *= -1.0
    alternating_returns = np.zeros_like(returns)
    for decision in range(decision_count):
        order = np.argsort(alternating_predictions[decision, :, 0])
        alternating_returns[decision, order[:3], 0] = -0.01
        alternating_returns[decision, order[-3:], 0] = 0.01
    alternating = economic_stock_attribution(
        alternating_predictions,
        alternating_returns,
        mask,
        dates,
        decisions,
    )
    alternating_accounting = economic_window_accounting(alternating, dates, decisions)
    assert alternating_accounting.daily_intraday_turnover.sum() == pytest.approx(
        2.0 * (decision_count - 1)
    )

    sparse_mask = np.zeros_like(mask)
    sparse_mask[[0, 53]] = True
    sparse = economic_stock_attribution(
        alternating_predictions,
        alternating_returns,
        sparse_mask,
        dates,
        decisions,
    )
    sparse_accounting = economic_window_accounting(sparse, dates, decisions)
    assert sparse_accounting.daily_entry_turnover.sum() == pytest.approx(1.0)
    assert sparse_accounting.daily_intraday_turnover.sum() == pytest.approx(2.0)
    assert sparse_accounting.daily_exit_turnover.sum() == pytest.approx(1.0)

    zero = economic_stock_attribution(
        alternating_predictions,
        np.zeros_like(returns),
        mask,
        dates,
        decisions,
    )
    zero_accounting = economic_window_accounting(zero, dates, decisions)
    zero_gross = float(zero_accounting.daily_gross_contribution.sum())
    zero_turnover = float(zero_accounting.daily_total_turnover.sum())
    assert zero_gross == 0.0
    assert 10_000.0 * zero_gross / zero_turnover == 0.0


def test_point_in_time_liquidity_vector_tracks_bucket_moves() -> None:
    grid = np.full((2, 55, 3, 1), np.nan, dtype=np.float64)
    grid[0, 0, :, 0] = [0.1, -0.05, -0.05]
    grid[1, 0, :, 0] = [0.2, -0.1, -0.1]
    buckets = np.asarray([[0, 0, 1], [1, 0, 1]], dtype=np.int8)
    bucket_zero = _point_in_time_bucket_contribution_vector(grid, buckets, 0, (0,), 0)
    bucket_one = _point_in_time_bucket_contribution_vector(grid, buckets, 1, (0,), 0)
    assert bucket_zero[0] == pytest.approx(0.05)
    assert bucket_one[0] == pytest.approx(0.10)
    np.testing.assert_allclose(
        bucket_zero + bucket_one,
        np.asarray([0.15, -0.075, -0.075]),
        atol=5e-12,
    )


def test_bootstrap_uses_only_finite_replicates_and_reports_unavailable() -> None:
    values = np.full((12, 2), np.nan, dtype=np.float64)
    values[0, 0] = 1.0
    result = moving_block_bootstrap_matrix(
        values, replications=100, block_length=3, seed=19
    )
    assert 0 < result["finite_replication_count"][0] < 100
    assert result["finite_replication_count"][1] == 0
    assert np.isnan(result["lower_95"][1])
    assert np.isnan(result["probability_positive"][1])
    assert result["probability_positive"][0] == 1.0


def test_metadata_loader_reads_only_training_and_validation_dates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_store = tmp_path / "feature-store"
    feature_store.mkdir()
    equity_count = 158
    date_count = 32
    equity_index = pl.DataFrame(
        {
            "equity_slot": np.arange(equity_count),
            "security_id": [f"security-{index}" for index in range(equity_count)],
            "isin": [f"BR{index:010d}" for index in range(equity_count)],
            "latest_ticker": [f"T{index}" for index in range(equity_count)],
            "xp_symbol": [f"Stock {index}" for index in range(equity_count)],
        }
    )
    equity_index.write_parquet(feature_store / "equity_index.parquet")
    trade_dates = [
        *[date(2024, 5, 30) + timedelta(days=offset) for offset in range(30)],
        date(2024, 7, 8),
        date(2025, 7, 7),
    ]
    pl.DataFrame(
        {"date_idx": np.arange(date_count), "trade_date": trade_dates}
    ).write_parquet(feature_store / "date_index.parquet")
    sample_index = pl.DataFrame(
        {
            "sample_id": np.arange(date_count),
            "date_idx": np.arange(date_count),
            "decision_idx": np.zeros(date_count, dtype=np.int64),
            "trade_date": trade_dates,
        }
    )
    validation_rows = sample_index.filter(pl.col("date_idx") == 30)
    observed_channel = DYNAMIC_CHANNELS.index("observed")
    equity_minutes = DECISION_EQUITY_INDICES[0]
    global_minutes = DECISION_GLOBAL_INDICES[0]
    arrays: dict[str, np.ndarray] = {
        "equity_membership.npy": np.ones((date_count, equity_count), dtype=bool),
        "equity_data_ready.npy": np.ones((date_count, equity_count), dtype=bool),
        "equity_slow.npy": np.zeros(
            (date_count, equity_count, len(SLOW_CHANNELS)), dtype=np.float32
        ),
        "equity_features.npy": np.ones(
            (date_count, equity_count, equity_minutes, len(DYNAMIC_CHANNELS)),
            dtype=np.float32,
        ),
        "context_features.npy": np.ones(
            (
                date_count,
                len(LOCAL_CONTEXT_SYMBOLS),
                75,
                len(DYNAMIC_CHANNELS),
            ),
            dtype=np.float32,
        ),
        "context_data_ready.npy": np.ones(
            (date_count, len(LOCAL_CONTEXT_SYMBOLS)), dtype=bool
        ),
        "global_features.npy": np.ones(
            (
                date_count,
                len(GLOBAL_CONTEXT_SYMBOLS),
                global_minutes,
                len(DYNAMIC_CHANNELS),
            ),
            dtype=np.float32,
        ),
        "global_data_ready.npy": np.ones(
            (date_count, len(GLOBAL_CONTEXT_SYMBOLS), 55), dtype=bool
        ),
    }
    liquidity_channel = SLOW_CHANNELS.index("median_daily_dollar_volume_20d_log_scale")
    arrays["equity_slow.npy"][:, :, liquidity_channel] = np.log1p(10_000_000.0)
    arrays["equity_slow.npy"][31] = 999_999.0
    arrays["equity_features.npy"][31] = 999_999.0
    accessed_dates: list[int] = []

    class GuardedArray:
        def __init__(self, values: np.ndarray) -> None:
            self.values = values
            self.shape = values.shape

        def __getitem__(self, key: object) -> np.ndarray:
            if not isinstance(key, tuple) or not isinstance(key[0], np.ndarray):
                raise AssertionError("Feature array was not date-indexed first")
            selected = np.asarray(key[0], dtype=np.int64)
            if np.any(selected == 31):
                raise AssertionError("Final-test feature value was read")
            accessed_dates.extend(selected.tolist())
            return self.values[key]

        def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
            del args, kwargs
            raise AssertionError("Full feature array was materialized")

    monkeypatch.setattr(
        "brazil_rv.modeling.analyze_stock_time_attribution.np.load",
        lambda path, **kwargs: GuardedArray(arrays[Path(path).name]),
    )
    monkeypatch.setattr(
        "brazil_rv.modeling.analyze_stock_time_attribution._feature_axes",
        lambda inputs: {
            "liquidity_channel_name": "median_daily_dollar_volume_20d_log_scale",
            "liquidity_channel_index": liquidity_channel,
            "observed_channel_index": observed_channel,
            "dollar_volume_log_affine": {"center": 0.0, "scale": 1.0},
        },
    )
    monkeypatch.setattr(
        "brazil_rv.modeling.analyze_stock_time_attribution._universe_liquidity_threshold",
        lambda inputs: None,
    )
    inputs = AnalysisInputs(
        state_path=tmp_path / "state.json",
        state_sha256="state",
        state={},
        configuration={},
        feature_store=feature_store,
        feature_identity={"manifest_sha256": "feature"},
        feature_manifest={},
        sample_index=sample_index,
        validation_rows=validation_rows,
        jobs=(),
        analyzer_git_commit_sha="commit",
        analyzer_worktree_clean=True,
        analyzer_source_sha256="source",
    )
    metadata = _load_analysis_metadata(
        inputs,
        {
            "date_idx": np.asarray([30], dtype=np.int64),
            "decision_idx": np.asarray([0], dtype=np.int64),
        },
    )
    assert set(accessed_dates) == set(range(31))
    assert metadata["active"].shape == (1, equity_count)
    assert metadata["trade_dates"].shape == (1,)


def test_dirty_execution_cache_identity_and_lock_ownership_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    job = Stage3AnalysisJob(
        position=0,
        logical_configuration="core",
        context_ablation=STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION["core"],
        seed=11,
        run_dir=run_dir,
        run_manifest_path=run_dir / "run_manifest.json",
        run_manifest_sha256="run",
        checkpoint_path=run_dir / "best.pt",
        checkpoint_sha256="checkpoint",
        producing_git_commit_sha="producer",
        manifest={},
    )
    inputs = AnalysisInputs(
        state_path=tmp_path / "state.json",
        state_sha256="state",
        state={},
        configuration={},
        feature_store=tmp_path / "feature-store",
        feature_identity={"manifest_sha256": "feature"},
        feature_manifest={},
        sample_index=pl.DataFrame(),
        validation_rows=pl.DataFrame({"sample_id": [1]}),
        jobs=(job,),
        analyzer_git_commit_sha="commit",
        analyzer_worktree_clean=False,
        analyzer_source_sha256="source",
        inference_code_sha256=_inference_code_identity(),
    )
    monkeypatch.setattr(
        "brazil_rv.modeling.analyze_stock_time_attribution.validate_analysis_inputs",
        lambda *args: inputs,
    )
    payload = __import__(
        "brazil_rv.modeling.analyze_stock_time_attribution",
        fromlist=["dry_run_payload"],
    ).dry_run_payload(tmp_path / "state.json", tmp_path / "dry", "core")
    assert payload["analyzer_worktree_clean"] is False
    assert payload["artifacts_created"] is False
    with pytest.raises(RuntimeError, match="clean worktree"):
        run_analysis(tmp_path / "state.json", tmp_path / "run-output", "core")
    assert not (tmp_path / "run-output").exists()

    clean_inputs = AnalysisInputs(
        **{
            **inputs.__dict__,
            "analyzer_worktree_clean": True,
        }
    )
    identity = _job_cache_identity(clean_inputs, job)
    assert "scope" not in identity
    assert identity["inference_code_sha256"] == _inference_code_identity()

    cache = tmp_path / "legacy-cache"
    cache.mkdir()
    manifest_path = cache / "manifest.json"
    dirty_identity = identity
    _atomic_write_json(
        manifest_path,
        {
            "status": "completed",
            "identity": dirty_identity,
            "creation_provenance": {
                "analyzer_worktree_clean": False,
                "analyzer_source_sha256": "source",
                "analyzer_git_commit_sha": "commit",
            },
        },
    )
    with pytest.raises(ValueError, match="provenance"):
        _validate_cache_manifest(manifest_path, dirty_identity)

    with exclusive_process_lock(
        PRODUCTION_TRAINING_LOCK, "synthetic inference holder"
    ) as lease:
        lease.assert_owned()
        with pytest.raises(RuntimeError, match="already active"):
            with exclusive_process_lock(
                PRODUCTION_TRAINING_LOCK, "synthetic training competitor"
            ):
                raise AssertionError("competing owner acquired production lock")
    lost = ProcessLockLease(
        path=tmp_path / "missing.lock", token="missing", purpose="synthetic"
    )
    with pytest.raises(RuntimeError, match="ownership was lost"):
        lost.assert_owned()


def test_core_output_integration_reconstructs_and_emits_canonical_rows(
    tmp_path: Path,
) -> None:
    date_count = 32
    decision_count = 55
    equity_count = 30
    sample_count = date_count * decision_count
    generator = np.random.default_rng(20260808)
    targets = generator.normal(size=(sample_count, equity_count, 3)).astype(np.float32)
    raw_returns = generator.normal(
        scale=0.01, size=(sample_count, equity_count, 3)
    ).astype(np.float32)
    label_mask = np.ones_like(targets, dtype=bool)
    date_idx = np.repeat(np.arange(date_count, dtype=np.int64), decision_count)
    decision_idx = np.tile(np.arange(decision_count, dtype=np.int64), date_count)
    shared = {
        "sample_id": np.arange(sample_count, dtype=np.int64),
        "date_idx": date_idx,
        "decision_idx": decision_idx,
        "targets": targets,
        "raw_returns": raw_returns,
        "label_mask": label_mask,
    }
    equity_index = pl.DataFrame(
        {
            "equity_slot": np.arange(equity_count),
            "security_id": [f"security-{index:03d}" for index in range(equity_count)],
            "isin": [f"BR{index:010d}" for index in range(equity_count)],
            "latest_ticker": [f"T{index:03d}" for index in range(equity_count)],
            "xp_symbol": [f"Stock {index:03d}" for index in range(equity_count)],
        }
    )
    metadata = {
        "trade_dates": np.arange(
            np.datetime64("2024-12-16"),
            np.datetime64("2024-12-16") + np.timedelta64(date_count, "D"),
        ),
        "equity_index": equity_index,
        "active": np.ones((date_count, equity_count), dtype=bool),
        "dollar_liquidity": np.tile(
            np.linspace(2_000_000.0, 20_000_000.0, equity_count),
            (date_count, 1),
        ),
        "eligibility_liquidity_threshold": {"value_brl": 2_000_000.0},
        "liquidity_quintile": np.tile(
            np.repeat(np.arange(5, dtype=np.int8), equity_count // 5),
            (date_count, 1),
        ),
        "adaptive_liquidity": np.zeros((date_count, equity_count), dtype=np.int8),
        "adaptive_liquidity_bucket_count": np.ones(date_count, dtype=np.int8),
        "equity_completeness": {
            "scheduled_minutes": np.asarray(DECISION_EQUITY_INDICES, dtype=np.int64)[
                decision_idx
            ],
            "observed_bars": np.floor(
                np.asarray(DECISION_EQUITY_INDICES, dtype=np.float64)[
                    decision_idx, None
                ]
                * np.tile(np.linspace(0.7, 1.0, equity_count), (sample_count, 1))
            ).astype(np.int64),
            "observed_fraction": np.tile(
                np.linspace(0.7, 1.0, equity_count), (sample_count, 1)
            ),
            "recent_observed_fraction": np.full((sample_count, equity_count), 0.9),
        },
        "local_completeness": {
            "observed_fraction": np.full(
                (sample_count, len(LOCAL_CONTEXT_SYMBOLS)), 0.95
            ),
            "preopen_observed_fraction": np.full(
                (sample_count, len(LOCAL_CONTEXT_SYMBOLS)), 0.9
            ),
            "minutes_since_most_recent_observed_bar": np.full(
                (sample_count, len(LOCAL_CONTEXT_SYMBOLS)), 2.0
            ),
            "ready": np.ones((sample_count, len(LOCAL_CONTEXT_SYMBOLS)), dtype=bool),
        },
        "global_completeness": {
            "observed_fraction": np.full(
                (sample_count, len(GLOBAL_CONTEXT_SYMBOLS)), 0.97
            ),
            "preopen_observed_fraction": np.full(
                (sample_count, len(GLOBAL_CONTEXT_SYMBOLS)), 0.92
            ),
            "minutes_since_most_recent_observed_bar": np.full(
                (sample_count, len(GLOBAL_CONTEXT_SYMBOLS)), 1.0
            ),
            "ready": np.ones((sample_count, len(GLOBAL_CONTEXT_SYMBOLS)), dtype=bool),
        },
        "overnight_regimes": {
            "normal": np.ones(date_count, dtype=bool),
            "large": np.arange(date_count) % 2 == 0,
            "large_positive": np.arange(date_count) % 4 == 0,
            "large_negative": np.arange(date_count) % 4 == 2,
        },
    }
    cache_paths = {}
    for seed in STAGE3_SEEDS:
        predictions = (
            targets + generator.normal(scale=0.5, size=targets.shape)
        ).astype(np.float32)
        path = tmp_path / f"predictions-{seed}.npy"
        np.save(path, predictions, allow_pickle=False)
        cache_paths[("core", seed)] = path

    outputs, core_by_seed, reconstruction = _build_core_outputs(
        cache_paths,
        shared,
        metadata,
        bootstrap_replications=8,
    )
    liquidity, liquidity_time, liquidity_checks = _build_liquidity_outputs(
        cache_paths,
        core_by_seed,
        shared,
        metadata,
    )
    opening = _build_opening_regimes(core_by_seed, shared, metadata)
    for logical in STAGE3_LOGICAL_CONFIGURATION_ORDER:
        for seed in STAGE3_SEEDS:
            cache_paths[(logical, seed)] = cache_paths[("core", seed)]
    context = _build_context_time_deltas(
        cache_paths,
        core_by_seed,
        shared,
        metadata,
        bootstrap_replications=8,
    )

    stock = outputs["stock_attribution"]
    assert stock.height == equity_count
    assert stock["valid_decision_count"].min() == decision_count
    np.testing.assert_allclose(
        stock["long_gross_return_contribution"].to_numpy()
        + stock["short_gross_return_contribution"].to_numpy(),
        stock["net_gross_spread_contribution"].to_numpy(),
        atol=5e-9,
        rtol=0.0,
    )
    time_bins = outputs["time_of_day_bins"]
    time_5m = outputs["time_of_day_5m"]
    assert set(time_5m["horizon_minutes"].unique()) == {0, 30, 60, 120}
    assert time_5m.filter(pl.col("aggregation") == "across_seed").height == (
        decision_count * 4
    )
    stock_time = outputs["stock_time_attribution"]
    assert "across_seed_minimum_contribution" in stock_time.columns
    selected_stock_time = stock_time.filter(
        (pl.col("aggregation") == "across_seed")
        & (pl.col("scope_type") == "decision_5m")
        & (pl.col("decision_idx") == 0)
        & (pl.col("horizon_minutes") == 0)
    )
    selected_time = time_5m.filter(
        (pl.col("aggregation") == "across_seed")
        & (pl.col("decision_idx") == 0)
        & (pl.col("horizon_minutes") == 0)
    )
    assert selected_stock_time["additive_ic_contribution"].sum() == pytest.approx(
        selected_time["mean_spearman_ic"].item(), abs=5e-12
    )
    assert all(
        values["stock_absolute_difference"] <= 5e-12
        and values["time_decomposition_maximum_absolute_difference"] <= 5e-12
        for values in reconstruction["by_seed"].values()
    )
    assert (
        liquidity.filter(
            (pl.col("bucket_kind") == "daily_liquidity_quintile")
            & (pl.col("aggregation") == "across_seed")
        ).height
        == 5 * 4
    )
    assert liquidity_time.filter(pl.col("aggregation") == "across_seed").height == (
        len(primary_time_bins()) * 5 * 3
    )
    assert all(check["passed"] is True for check in liquidity_checks.values())
    assert {
        "equity_history_stratification",
        "overnight_regime",
        "retained_local_context",
        "retained_global_context",
        "core_b3_observed_bar_count",
        "core_b3_history_missingness",
        "core_b3_history_completeness",
        "core_retained_local_completeness",
        "core_retained_global_completeness",
        "core_retained_global_freshness",
        "core_retained_context_preopen",
        "core_retained_local_preopen",
        "core_retained_global_preopen",
    }.issubset(set(opening["diagnostic_type"].unique()))
    assert set(
        opening.filter(pl.col("diagnostic_type").str.starts_with("core_"))[
            "horizon_minutes"
        ].drop_nulls()
    ) == {0, 30, 60, 120}
    assert set(
        context.filter(pl.col("scope_type") == "decision_5m")["horizon_minutes"]
    ) == {0, 30, 60, 120}
    assert {
        "b3_observed_bar_count",
        "b3_history_missingness",
        "b3_history_completeness",
        "retained_local_completeness",
        "retained_global_completeness",
        "retained_global_freshness",
        "retained_context_preopen",
        "overnight_regime",
        "retained_local_preopen",
        "retained_global_preopen",
        "added_context_freshness",
    }.issubset(set(context["condition_type"].drop_nulls()))
    finite_delta = context["mean_paired_ic_delta"].drop_nulls().to_numpy()
    np.testing.assert_allclose(finite_delta, 0.0, atol=5e-12)
    time_values = time_5m.filter(pl.col("aggregation") == "across_seed")
    np.testing.assert_allclose(
        time_values["mean_gross_top_return"].to_numpy()
        - time_values["mean_gross_bottom_return"].to_numpy(),
        time_values["mean_gross_top_minus_bottom"].to_numpy(),
        atol=5e-9,
        rtol=0.0,
    )
    coverage_columns = (
        "mean_valid_equity_count",
        "label_coverage",
        "valid_decision_cell_count",
        "valid_date_count",
    )
    for frame, keys in (
        (time_5m, ("decision_idx", "horizon_minutes")),
        (time_bins, ("scope", "horizon_minutes")),
    ):
        seed_coverage = (
            frame.filter(pl.col("seed") == STAGE3_SEEDS[0])
            .sort(keys)
            .select(coverage_columns)
            .to_numpy()
        )
        across_coverage = (
            frame.filter(pl.col("aggregation") == "across_seed")
            .sort(keys)
            .select(coverage_columns)
            .to_numpy()
        )
        np.testing.assert_allclose(seed_coverage, across_coverage, rtol=0.0, atol=0.0)
    assert "decision_count" in time_bins.columns
    assert set(time_bins["horizon_minutes"]) == {0, 30, 60, 120}

    active_stock = stock.filter(pl.col("mean_daily_total_one_way_turnover") > 0)
    np.testing.assert_allclose(
        active_stock["gross_contribution_per_unit_turnover"].to_numpy(),
        active_stock["mean_daily_gross_contribution"].to_numpy()
        / active_stock["mean_daily_total_one_way_turnover"].to_numpy(),
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        active_stock["break_even_one_way_cost_bps"].to_numpy(),
        10_000.0 * active_stock["gross_contribution_per_unit_turnover"].to_numpy(),
        rtol=0.0,
        atol=1e-11,
    )
    across_liquidity = liquidity.filter(
        (pl.col("aggregation") == "across_seed")
        & (pl.col("mean_daily_total_one_way_turnover") > 0)
    )
    np.testing.assert_allclose(
        across_liquidity["gross_contribution_per_unit_turnover"].to_numpy(),
        across_liquidity["mean_daily_gross_contribution"].to_numpy()
        / across_liquidity["mean_daily_total_one_way_turnover"].to_numpy(),
        rtol=0.0,
        atol=1e-15,
    )

    core_history = opening.filter(
        pl.col("diagnostic_type").is_in(
            [
                "core_b3_observed_bar_count",
                "core_b3_history_missingness",
                "core_b3_history_completeness",
            ]
        )
    )
    assert {
        "category_lower_bound",
        "category_upper_bound",
        "category_lower_bound_inclusive",
        "category_upper_bound_inclusive",
        "category_unit",
        "condition_decision_cell_count",
        "condition_date_count",
        "mean_median_observed_bar_count",
        "median_observed_fraction",
        "mean_fraction_eligible_meeting_expected_history",
    }.issubset(core_history.columns)
    assert core_history["condition_decision_cell_count"].is_not_null().all()
    assert set(core_history["horizon_minutes"]) == {0, 30, 60, 120}


def test_fixed_liquidity_quintile_ic_reranks_within_each_date_bucket() -> None:
    equity_count = 158
    groups = np.full((2, equity_count), -1, dtype=np.int8)
    groups[0, :150] = np.repeat(np.arange(5, dtype=np.int8), 30)
    groups[1] = groups[0]
    groups[1, 0] = 1
    predictions = np.broadcast_to(
        np.arange(equity_count, dtype=np.float32)[None, :, None],
        (2, equity_count, 1),
    ).copy()
    targets = predictions.copy()
    targets[:, :30, 0] = targets[:, :30, 0][:, ::-1]
    mask = groups[:, :, None] >= 0
    within = _independent_bucket_ic(
        predictions,
        targets,
        mask,
        groups,
        np.asarray([0, 1], dtype=np.int64),
        5,
    )
    assert np.isfinite(within[0, :, 0]).all()
    assert within[0, 0, 0] == pytest.approx(-1.0)
    assert np.isnan(within[1, 0, 0])
    assert np.isfinite(within[1, 1:, 0]).all()
    additive = additive_spearman_contributions(predictions, targets, mask)
    additive_bucket = additive.contributions[0, groups[0] == 0, 0].sum()
    assert additive_bucket != pytest.approx(within[0, 0, 0])


def test_economic_ratios_recompute_from_across_seed_primitives() -> None:
    seed_rows = pl.DataFrame(
        {
            "group": [1, 1],
            "mean_daily_gross_contribution": [1.0, 3.0],
            "mean_daily_total_one_way_turnover": [1.0, 9.0],
            "gross_contribution_per_unit_turnover": [1.0, 1.0 / 3.0],
            "break_even_one_way_cost_bps": [10_000.0, 10_000.0 / 3.0],
        }
    )
    primitives = seed_rows.group_by("group").agg(
        pl.col("mean_daily_gross_contribution").mean(),
        pl.col("mean_daily_total_one_way_turnover").mean(),
    )
    across = _with_economic_ratios(primitives)
    assert across["gross_contribution_per_unit_turnover"].item() == pytest.approx(0.4)
    assert across["break_even_one_way_cost_bps"].item() == pytest.approx(4_000.0)
    assert across["gross_contribution_per_unit_turnover"].item() != pytest.approx(
        seed_rows["gross_contribution_per_unit_turnover"].mean()
    )


def test_stock_opportunity_support_is_not_diluted_by_ineligible_cells() -> None:
    contributions = np.zeros((2, 31, 1), dtype=np.float64)
    contributions[0, 0, 0] = 0.2
    additive = __import__(
        "brazil_rv.modeling.analyze_stock_time_attribution",
        fromlist=["AdditiveSpearmanResult"],
    ).AdditiveSpearmanResult(
        contributions=contributions,
        sample_ic=np.asarray([[0.2], [0.1]], dtype=np.float64),
    )
    label_mask = np.ones_like(contributions, dtype=bool)
    label_mask[1, 0, 0] = False
    accounting = stock_contribution_opportunity_accounting(additive, label_mask)
    assert accounting["valid_opportunity_count"][0] == 1
    assert accounting["portfolio_valid_cell_count"][0] == 2
    assert accounting["conditional_contribution"][0] == pytest.approx(0.2)
    assert accounting["unconditional_contribution"][0] == pytest.approx(0.1)


def test_opening_history_categories_use_counts_completeness_and_readiness() -> None:
    sample_count = 3
    equity_count = 30
    scheduled = np.asarray([15, 20, 25], dtype=np.int64)
    observed = np.asarray(
        [
            np.full(equity_count, 15),
            np.full(equity_count, 17),
            np.full(equity_count, 15),
        ],
        dtype=np.int64,
    )
    local_ready = np.ones((sample_count, len(LOCAL_CONTEXT_SYMBOLS)), dtype=bool)
    local_ready[1, LOCAL_CONTEXT_SYMBOLS.index("WDO$")] = False
    global_ready = np.ones((sample_count, len(GLOBAL_CONTEXT_SYMBOLS)), dtype=bool)
    global_ready[2, GLOBAL_CONTEXT_SYMBOLS.index("ZT.v.0")] = False
    metadata = {
        "active": np.ones((sample_count, equity_count), dtype=bool),
        "equity_completeness": {
            "scheduled_minutes": scheduled,
            "observed_bars": observed,
            "observed_fraction": observed / scheduled[:, None],
        },
        "local_completeness": {
            "ready": local_ready,
            "observed_fraction": np.ones_like(local_ready, dtype=np.float64),
            "preopen_observed_fraction": np.ones_like(local_ready, dtype=np.float64),
        },
        "global_completeness": {
            "ready": global_ready,
            "observed_fraction": np.ones_like(global_ready, dtype=np.float64),
            "preopen_observed_fraction": np.ones_like(global_ready, dtype=np.float64),
            "minutes_since_most_recent_observed_bar": np.zeros_like(
                global_ready, dtype=np.float64
            ),
        },
        "overnight_regimes": {
            "normal": np.ones(sample_count, dtype=bool),
        },
    }
    conditions = {
        (condition.condition_type, condition.category): condition
        for condition in _opening_condition_masks(
            {"date_idx": np.arange(sample_count, dtype=np.int64)}, metadata
        )
    }
    np.testing.assert_array_equal(
        conditions[("b3_history_missingness", "complete")].sample_mask,
        [True, False, False],
    )
    np.testing.assert_array_equal(
        conditions[("b3_history_missingness", "missing_1_to_5_bars")].sample_mask,
        [False, True, False],
    )
    np.testing.assert_array_equal(
        conditions[("b3_history_completeness", "below_80pct")].sample_mask,
        [False, False, True],
    )
    assert conditions[("b3_history_completeness", "80_to_95pct")].category_unit
    np.testing.assert_array_equal(
        conditions[("retained_local_preopen", "complete")].sample_mask,
        [True, False, True],
    )
    np.testing.assert_array_equal(
        conditions[("retained_global_preopen", "complete")].sample_mask,
        [True, True, False],
    )
    np.testing.assert_array_equal(
        conditions[("retained_context_preopen", "complete")].sample_mask,
        [True, False, False],
    )


def test_absolute_b3_observed_bar_bands_are_exact_exhaustive_and_distinct() -> None:
    observed_counts = np.asarray(
        [0, 30, 31, 60, 61, 90, 91, 120, 121, 180, 181], dtype=np.int64
    )
    sample_count = observed_counts.size
    equity_count = 30
    scheduled = np.maximum(observed_counts, 1)
    observed = np.broadcast_to(
        observed_counts[:, None], (sample_count, equity_count)
    ).copy()
    local_ready = np.ones((sample_count, len(LOCAL_CONTEXT_SYMBOLS)), dtype=bool)
    global_ready = np.ones((sample_count, len(GLOBAL_CONTEXT_SYMBOLS)), dtype=bool)
    metadata = {
        "active": np.ones((sample_count, equity_count), dtype=bool),
        "equity_completeness": {
            "scheduled_minutes": scheduled,
            "observed_bars": observed,
            "observed_fraction": observed / scheduled[:, None],
        },
        "local_completeness": {
            "ready": local_ready,
            "observed_fraction": np.ones_like(local_ready, dtype=np.float64),
            "preopen_observed_fraction": np.ones_like(local_ready, dtype=np.float64),
        },
        "global_completeness": {
            "ready": global_ready,
            "observed_fraction": np.ones_like(global_ready, dtype=np.float64),
            "preopen_observed_fraction": np.ones_like(global_ready, dtype=np.float64),
            "minutes_since_most_recent_observed_bar": np.zeros_like(
                global_ready, dtype=np.float64
            ),
        },
        "overnight_regimes": {"normal": np.ones(sample_count, dtype=bool)},
    }
    conditions = _opening_condition_masks(
        {"date_idx": np.arange(sample_count, dtype=np.int64)}, metadata
    )
    absolute = [
        condition
        for condition in conditions
        if condition.condition_type == "b3_observed_bar_count"
    ]
    assert [condition.category for condition in absolute] == [
        "0_to_30_bars",
        "31_to_60_bars",
        "61_to_90_bars",
        "91_to_120_bars",
        "121_to_180_bars",
        "over_180_bars",
    ]
    membership = np.stack([condition.sample_mask for condition in absolute])
    np.testing.assert_array_equal(membership.sum(axis=0), np.ones(sample_count))
    np.testing.assert_array_equal(
        np.argmax(membership, axis=0),
        np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5]),
    )
    assert [condition.category_lower_bound for condition in absolute] == [
        0.0,
        30.0,
        60.0,
        90.0,
        120.0,
        180.0,
    ]
    assert [condition.category_upper_bound for condition in absolute] == [
        30.0,
        60.0,
        90.0,
        120.0,
        180.0,
        None,
    ]
    assert absolute[0].category_lower_bound_inclusive is True
    assert all(
        condition.category_lower_bound_inclusive is False for condition in absolute[1:]
    )
    assert all(
        condition.category_upper_bound_inclusive is True for condition in absolute[:-1]
    )
    by_identity = {
        (condition.condition_type, condition.category): condition
        for condition in conditions
    }
    early = 1
    late = 10
    assert (
        by_identity[("b3_history_missingness", "complete")]
        .sample_mask[[early, late]]
        .all()
    )
    assert (
        by_identity[("b3_history_completeness", "at_least_95pct")]
        .sample_mask[[early, late]]
        .all()
    )
    assert membership[0, early]
    assert membership[5, late]


def test_scope_coverage_is_date_weighted_and_h0_counts_finite_cells() -> None:
    counts = np.zeros((2, 3, 3), dtype=np.float64)
    coverage = np.zeros_like(counts)
    counts[:, :2, 0] = [[30.0, 40.0], [50.0, 0.0]]
    coverage[:, :2, 0] = [[0.3, 0.4], [0.5, 0.0]]
    horizon = _coverage_summary(counts, coverage, (0, 1), 0)
    assert horizon == {
        "mean_valid_equity_count": pytest.approx(42.5),
        "label_coverage": pytest.approx(0.425),
        "valid_decision_cell_count": 3,
        "valid_date_count": 2,
    }
    counts[:, :2, 1:] = [
        [[0.0, 40.0], [50.0, 0.0]],
        [[60.0, 70.0], [0.0, 0.0]],
    ]
    coverage[counts >= 30] = counts[counts >= 30] / 100.0
    primary = _coverage_summary(counts, coverage, (0, 1), None)
    assert primary["valid_decision_cell_count"] == 7
    assert primary["valid_date_count"] == 2
    assert primary["mean_valid_equity_count"] == pytest.approx(50.0)


def test_portable_cache_reuses_core_for_full_scope_and_rejects_semantic_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid_gate = _valid_metric_gate(tmp_path)
    feature_store = tmp_path / "feature-store"
    feature_store.mkdir()
    targets = np.linspace(-1.0, 1.0, 158 * 3, dtype=np.float32).reshape(1, 158, 3)
    raw_returns = targets / 100.0
    label_mask = np.ones_like(targets, dtype=bool)
    jobs: list[Stage3AnalysisJob] = []
    for position, (logical, seed) in enumerate(
        (logical, seed)
        for logical in STAGE3_LOGICAL_CONFIGURATION_ORDER
        for seed in STAGE3_SEEDS
    ):
        run_dir = tmp_path / "runs" / f"{logical}_{seed}"
        run_dir.mkdir(parents=True)
        jobs.append(
            Stage3AnalysisJob(
                position=position,
                logical_configuration=logical,
                context_ablation=STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION[logical],
                seed=seed,
                run_dir=run_dir,
                run_manifest_path=run_dir / "run_manifest.json",
                run_manifest_sha256=f"run-{position}",
                checkpoint_path=run_dir / "best.pt",
                checkpoint_sha256=f"checkpoint-{position}",
                producing_git_commit_sha="producer",
                manifest={"synthetic_index": position},
            )
        )

    def inputs_for(
        selected: tuple[Stage3AnalysisJob, ...], reporting: str
    ) -> AnalysisInputs:
        return AnalysisInputs(
            state_path=tmp_path / "stage3-state.json",
            state_sha256="state",
            state={},
            configuration={},
            feature_store=feature_store,
            feature_identity={"manifest_sha256": "feature"},
            feature_manifest={},
            sample_index=pl.DataFrame(),
            validation_rows=pl.DataFrame(
                {
                    "sample_id": [0],
                    "date_idx": [0],
                    "decision_idx": [0],
                }
            ),
            jobs=selected,
            analyzer_git_commit_sha=f"commit-{reporting}",
            analyzer_worktree_clean=True,
            analyzer_source_sha256=f"source-{reporting}",
            inference_code_sha256=_inference_code_identity(),
        )

    calls: list[int] = []

    def collect(manifest: dict[str, object], *args: object) -> SimpleNamespace:
        del args
        index = int(manifest["synthetic_index"])
        calls.append(index)
        observations = EvaluationObservations(
            sample_id=np.asarray([0], dtype=np.int64),
            date_idx=np.asarray([0], dtype=np.int64),
            decision_idx=np.asarray([0], dtype=np.int64),
            predictions=(targets + index / 1000.0).astype(np.float32),
            targets=targets,
            raw_returns=raw_returns,
            label_mask=label_mask,
        )
        return SimpleNamespace(observations=observations, summary={}, daily_rows=[])

    monkeypatch.setattr(
        "brazil_rv.modeling.stock_time_cache.validate_runtime",
        lambda: None,
    )
    monkeypatch.setattr(
        "brazil_rv.modeling.stock_time_cache.collect_neural_evaluation",
        collect,
    )
    monkeypatch.setattr(
        "brazil_rv.modeling.stock_time_cache.metric_reproduction_gate",
        lambda *args: copy.deepcopy(valid_gate),
    )
    cache_root = tmp_path / "portable-cache"
    core_inputs = inputs_for(tuple(jobs[:3]), "core")
    core_state = {
        "status": "running",
        "jobs": [{"status": "pending"} for _ in core_inputs.jobs],
    }
    core_output = tmp_path / "core-output"
    core_output.mkdir()
    core_paths, _ = _adopt_or_infer_caches(
        cache_root,
        core_inputs,
        core_state,
        core_output / "analysis_state.json",
    )
    assert calls == [0, 1, 2]
    core_cache_bytes = {
        path.relative_to(cache_root): path.read_bytes()
        for path in cache_root.rglob("*")
        if path.is_file()
    }
    first_manifest = json.loads(
        (core_paths[("core", 11)].parent / "manifest.json").read_text(encoding="utf-8")
    )
    assert set(first_manifest["identity"]["inference_code_sha256"]) == set(
        INFERENCE_CODE_PATHS
    )
    assert first_manifest["creation_provenance"]["analyzer_git_commit_sha"] == (
        "commit-core"
    )
    assert first_manifest["creation_provenance"]["analyzer_source_sha256"] == (
        "source-core"
    )
    assert first_manifest["metric_reproduction_gate"] == valid_gate
    _validate_metric_reproduction_gate(first_manifest["metric_reproduction_gate"])

    full_inputs = inputs_for(tuple(jobs), "reporting-only-change")
    assert _job_cache_identity(core_inputs, jobs[0]) == _job_cache_identity(
        full_inputs, jobs[0]
    )
    full_state = {
        "status": "running",
        "jobs": [{"status": "pending"} for _ in full_inputs.jobs],
    }
    full_output = tmp_path / "full-output"
    full_output.mkdir()
    _adopt_or_infer_caches(
        cache_root,
        full_inputs,
        full_state,
        full_output / "analysis_state.json",
    )
    assert calls == list(range(24))
    assert full_state["jobs"][0]["cache_manifest_path"].startswith(str(cache_root))
    assert core_paths[("core", 11)].parent.parent.parent == cache_root
    assert core_cache_bytes == {
        relative: (cache_root / relative).read_bytes() for relative in core_cache_bytes
    }
    completed_cache_bytes = {
        path.relative_to(cache_root): path.read_bytes()
        for path in cache_root.rglob("*")
        if path.is_file()
    }
    resumed_state = {
        "status": "running",
        "jobs": [{"status": "pending"} for _ in full_inputs.jobs],
    }
    resumed_output = tmp_path / "resumed-output"
    resumed_output.mkdir()
    _adopt_or_infer_caches(
        cache_root,
        full_inputs,
        resumed_state,
        resumed_output / "analysis_state.json",
    )
    assert calls == list(range(24))
    assert completed_cache_bytes == {
        relative: (cache_root / relative).read_bytes()
        for relative in completed_cache_bytes
    }

    changed_job = Stage3AnalysisJob(
        **{**jobs[0].__dict__, "checkpoint_sha256": "changed-checkpoint"}
    )
    changed_inputs = inputs_for((changed_job,), "another-reporting-change")
    with pytest.raises(ValueError, match="identity"):
        _validate_cache_manifest(
            core_paths[("core", 11)].parent / "manifest.json",
            _job_cache_identity(changed_inputs, changed_job),
        )
    manifest_path = core_paths[("core", 11)].parent / "manifest.json"
    for changed_path in (
        "research/src/brazil_rv/modeling/contract.py",
        "research/src/brazil_rv/modeling/engine.py",
        "research/src/brazil_rv/modeling/stock_time_cache.py",
    ):
        changed_hashes = dict(core_inputs.inference_code_sha256)
        changed_hashes[changed_path] = "changed-code-hash"
        changed_code_inputs = AnalysisInputs(
            **{
                **core_inputs.__dict__,
                "inference_code_sha256": changed_hashes,
            }
        )
        with pytest.raises(ValueError, match="identity"):
            _validate_cache_manifest(
                manifest_path,
                _job_cache_identity(changed_code_inputs, jobs[0]),
            )
    mismatched_hash_state = {
        "status": "running",
        "jobs": [{"status": "pending"} for _ in changed_code_inputs.jobs],
    }
    with pytest.raises(ValueError, match="identity"):
        _adopt_or_infer_caches(
            cache_root,
            changed_code_inputs,
            mismatched_hash_state,
            tmp_path / "mismatched-hash-state.json",
        )

    def mutate_gate(manifest: dict[str, object], mutation: str) -> None:
        gate = manifest["metric_reproduction_gate"]
        assert isinstance(gate, dict)
        if mutation == "passed_false":
            primary = gate["primary_ic"]
            assert isinstance(primary, dict)
            primary["recomputed"] = float(primary["recorded"]) + 2e-6
            primary["absolute_difference"] = 2e-6
            primary["passed"] = False
            gate["passed"] = False
        elif mutation == "missing_version":
            del gate["schema_version"]
        elif mutation == "legacy_gate":
            manifest["metric_reproduction_gate"] = {"passed": True}
        elif mutation == "missing_metric":
            daily_metrics = gate["daily_metrics"]
            assert isinstance(daily_metrics, dict)
            del daily_metrics["spearman_ic"]
        elif mutation == "altered_threshold":
            thresholds = gate["thresholds"]
            assert isinstance(thresholds, dict)
            daily_thresholds = thresholds["daily_metric_absolute_tolerances"]
            assert isinstance(daily_thresholds, dict)
            daily_thresholds["one_way_turnover"] = 0.01
            daily_metrics = gate["daily_metrics"]
            assert isinstance(daily_metrics, dict)
            turnover = daily_metrics["one_way_turnover"]
            assert isinstance(turnover, dict)
            turnover["threshold"] = 0.01
        elif mutation == "inconsistent_pass":
            daily_metrics = gate["daily_metrics"]
            assert isinstance(daily_metrics, dict)
            spearman = daily_metrics["spearman_ic"]
            assert isinstance(spearman, dict)
            spearman["passed"] = False
        elif mutation == "missing_metric_pass":
            daily_metrics = gate["daily_metrics"]
            assert isinstance(daily_metrics, dict)
            spearman = daily_metrics["spearman_ic"]
            assert isinstance(spearman, dict)
            del spearman["passed"]
        else:
            raise AssertionError(mutation)

    for mutation in (
        "passed_false",
        "missing_version",
        "legacy_gate",
        "missing_metric",
        "altered_threshold",
        "inconsistent_pass",
        "missing_metric_pass",
    ):
        rejected_manifest = copy.deepcopy(first_manifest)
        mutate_gate(rejected_manifest, mutation)
        _atomic_write_json(manifest_path, rejected_manifest)
        rejected_state = {"status": "running", "jobs": [{"status": "pending"}]}
        with pytest.raises(
            ValueError, match="Metric reproduction gate|lacks metric parity"
        ):
            _adopt_or_infer_caches(
                cache_root,
                inputs_for((jobs[0],), f"rejected-{mutation}"),
                rejected_state,
                tmp_path / f"rejected-{mutation}-state.json",
            )


def test_full_synthetic_orchestration_core_full_cache_resume_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from brazil_rv.modeling import (
        analyze_stage3_context_addition as stage3_analyzer,
    )
    from brazil_rv.modeling import stage2_context_ablation as stage2
    from brazil_rv.modeling import stage3_context_addition as stage3
    from brazil_rv.modeling import stock_time_cache as cache_semantics
    from brazil_rv.modeling import stock_time_inference as inference_semantics
    from brazil_rv.modeling.context_ablation import get_context_ablation
    from brazil_rv.modeling.contract import (
        FEATURE_CONTRACT_VERSION,
        SplitBoundaries,
        VALIDATION_END,
        VALIDATION_START,
    )
    from brazil_rv.modeling.contract import TCNSettings, architecture_for_model
    from brazil_rv.modeling.evaluate import _CHECKPOINT_IDENTITY_FIELDS

    analyzer = __import__(
        "brazil_rv.modeling.analyze_stock_time_attribution",
        fromlist=["FINAL_ARTIFACT_NAMES", "_sha256"],
    )
    feature_store = tmp_path / "feature-store"
    feature_store.mkdir()
    _atomic_write_json(
        feature_store / "manifest.json",
        {
            "contract_version": FEATURE_CONTRACT_VERSION,
            "global_context": {
                "source_hashes": {"source": "synthetic-source"},
                "normalized_store_hashes": {"store": "synthetic-store"},
            },
            "canonical_inputs": {},
            "constants": {"dollar_volume_log_affine": {"center": 0.0, "scale": 1.0}},
        },
    )
    _atomic_write_json(
        feature_store / "feature_schema.json",
        {
            "slow_channels": [{"name": name} for name in SLOW_CHANNELS],
            "dynamic_channels": [{"name": name} for name in DYNAMIC_CHANNELS],
        },
    )

    training_dates = [
        date(2021, 8, 16) + timedelta(days=offset) for offset in range(30)
    ]
    validation_dates = [
        VALIDATION_START,
        date(2024, 12, 30),
        date(2024, 12, 31),
        date(2025, 1, 2),
        VALIDATION_END,
    ]
    final_test_date = SplitBoundaries().test_start
    trade_dates = [*training_dates, *validation_dates, final_test_date]
    training_count = len(training_dates)
    validation_indices = np.arange(training_count, training_count + 5, dtype=np.int64)
    final_test_index = len(trade_dates) - 1
    sample_rows = [
        {
            "sample_id": date_index * 55 + decision,
            "date_idx": date_index,
            "decision_idx": decision,
            "trade_date": trade_date,
            "equity_cutoff_index": int(DECISION_EQUITY_INDICES[decision]),
            "context_cutoff_index": 75 + 5 * decision,
            "active_equity_count": 158,
        }
        for date_index, trade_date in enumerate(trade_dates)
        for decision in range(55)
    ]
    sample_index = pl.DataFrame(sample_rows)
    sample_index.write_parquet(feature_store / "sample_index.parquet")
    pl.DataFrame(
        {"date_idx": np.arange(len(trade_dates)), "trade_date": trade_dates}
    ).write_parquet(feature_store / "date_index.parquet")
    pl.DataFrame(
        {
            "equity_slot": np.arange(158),
            "security_id": [f"security-{index:03d}" for index in range(158)],
            "isin": [f"BR{index:010d}" for index in range(158)],
            "latest_ticker": [f"T{index:03d}" for index in range(158)],
            "xp_symbol": [f"Stock {index:03d}" for index in range(158)],
        }
    ).write_parquet(feature_store / "equity_index.parquet")

    date_count = len(trade_dates)
    observed_channel = DYNAMIC_CHANNELS.index("observed")
    liquidity_channel = SLOW_CHANNELS.index("median_daily_dollar_volume_20d_log_scale")
    overnight_channel = SLOW_CHANNELS.index("overnight_gap_normalized")

    def write_feature_array(name: str, values: np.ndarray) -> None:
        _atomic_write_npy(feature_store / name, values)

    write_feature_array("equity_membership.npy", np.ones((date_count, 158), dtype=bool))
    write_feature_array("equity_data_ready.npy", np.ones((date_count, 158), dtype=bool))
    equity_slow = np.zeros((date_count, 158, len(SLOW_CHANNELS)), dtype=np.float32)
    equity_slow[:, :, liquidity_channel] = np.log1p(
        np.linspace(2_000_000.0, 20_000_000.0, 158)
    )
    equity_slow[:, :, overnight_channel] = np.linspace(-0.02, 0.02, date_count)[:, None]
    write_feature_array("equity_slow.npy", equity_slow)
    del equity_slow
    equity_features = np.zeros(
        (date_count, 158, 405, len(DYNAMIC_CHANNELS)), dtype=np.float32
    )
    equity_features[..., observed_channel] = 1.0
    equity_features[final_test_index] = -777.0
    write_feature_array("equity_features.npy", equity_features)
    del equity_features
    write_feature_array(
        "context_slow.npy",
        np.zeros(
            (date_count, len(LOCAL_CONTEXT_SYMBOLS), len(SLOW_CHANNELS)),
            dtype=np.float32,
        ),
    )
    context_features = np.zeros(
        (
            date_count,
            len(LOCAL_CONTEXT_SYMBOLS),
            465,
            len(DYNAMIC_CHANNELS),
        ),
        dtype=np.float32,
    )
    context_features[..., observed_channel] = 1.0
    context_features[final_test_index] = -777.0
    write_feature_array("context_features.npy", context_features)
    del context_features
    write_feature_array(
        "context_data_ready.npy",
        np.ones((date_count, len(LOCAL_CONTEXT_SYMBOLS)), dtype=bool),
    )
    write_feature_array(
        "global_slow.npy",
        np.zeros(
            (date_count, len(GLOBAL_CONTEXT_SYMBOLS), 55, len(SLOW_CHANNELS)),
            dtype=np.float32,
        ),
    )
    global_features = np.zeros(
        (
            date_count,
            len(GLOBAL_CONTEXT_SYMBOLS),
            615,
            len(DYNAMIC_CHANNELS),
        ),
        dtype=np.float32,
    )
    global_features[..., observed_channel] = 1.0
    global_features[final_test_index] = -777.0
    write_feature_array("global_features.npy", global_features)
    del global_features
    write_feature_array(
        "global_data_ready.npy",
        np.ones((date_count, len(GLOBAL_CONTEXT_SYMBOLS), 55), dtype=bool),
    )
    dense_target_shape = (date_count, 158, 55, 3)
    write_feature_array(
        "raw_returns.npy", np.zeros(dense_target_shape, dtype=np.float32)
    )
    write_feature_array("targets.npy", np.zeros(dense_target_shape, dtype=np.float32))
    write_feature_array("label_mask.npy", np.ones(dense_target_shape, dtype=bool))
    write_feature_array(
        "cross_section_median.npy",
        np.zeros((date_count, 55, 3), dtype=np.float32),
    )
    write_feature_array("horizon_mask.npy", np.ones((date_count, 55, 3), dtype=bool))

    def canonical_validation_dates() -> list[date]:
        weekdays: list[date] = []
        current = VALIDATION_START
        while current <= VALIDATION_END:
            if current.weekday() < 5:
                weekdays.append(current)
            current += timedelta(days=1)
        indices = np.linspace(0, len(weekdays) - 1, 244, dtype=int)
        result = [weekdays[index] for index in indices]
        assert len(set(result)) == 244
        return result

    validation_artifact_dates = canonical_validation_dates()
    metrics_by_position: dict[
        int, tuple[dict[str, object], list[dict[str, object]]]
    ] = {}

    def write_completed_run(
        run_dir: Path,
        configuration: dict[str, object],
        key: str,
        seed: int,
        commit: str,
        score: float,
        position: int,
    ) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        identity = configuration["feature_store"]
        training_semantics = copy.deepcopy(configuration["training_semantics"])
        assert isinstance(identity, dict)
        assert isinstance(training_semantics, dict)
        tcn_settings = training_semantics["tcn_settings"]
        assert isinstance(tcn_settings, dict)
        architecture = architecture_for_model("tcn", TCNSettings(**tcn_settings))
        training_semantics["architecture_constants"] = asdict(architecture)
        manifest = {
            **training_semantics,
            "status": "completed",
            "seed": seed,
            "git_commit_sha": commit,
            "resolved_feature_store_path": str(feature_store),
            "feature_manifest_contract_version": configuration["feature_contract"],
            "split_boundaries": copy.deepcopy(configuration["split_boundaries"]),
            "context_ablation": get_context_ablation(key).metadata(),
            "global_context_source_hashes": identity["global_context_source_hashes"],
            "global_context_normalized_store_hashes": identity[
                "global_context_normalized_store_hashes"
            ],
            "resolved_source_paths": identity["canonical_inputs"],
            "best_validation_primary_score": score,
            "best_epoch": 2,
            "stopped_epoch": 3,
            "successful_optimizer_updates": 231,
            "training_duration_seconds": 1.0,
            "scheduler_steps": {"steps_per_epoch": 77},
            "synthetic_index": position,
        }
        checkpoint = {
            field: copy.deepcopy(manifest[field])
            for field in _CHECKPOINT_IDENTITY_FIELDS
        }
        checkpoint["context_ablation"] = copy.deepcopy(manifest["context_ablation"])
        _atomic_write_json(run_dir / "run_manifest.json", manifest)
        torch.save(checkpoint, run_dir / "best.pt")
        pl.DataFrame(
            {
                "epoch": [1, 2, 3],
                "optimizer_steps": [77, 77, 77],
                "epoch_seconds": [0.3, 0.3, 0.4],
            }
        ).write_csv(run_dir / "history.csv")
        horizon_scores = {30: score - 0.0001, 60: score, 120: score + 0.0001}
        daily_rows = [
            {
                "trade_date": trade_date,
                "date_idx": date_index,
                "horizon_minutes": horizon,
                "spearman_ic": horizon_scores[horizon],
                "rank_target_pearson_ic": horizon_scores[horizon] / 2.0,
                "top_return": 0.002 + score,
                "bottom_return": -0.001,
                "top_minus_bottom": 0.003 + score,
                "long_only_top": 0.002 + score,
                "one_way_turnover": 0.4 + score,
            }
            for date_index, trade_date in enumerate(validation_artifact_dates)
            for horizon in (30, 60, 120)
        ]
        summary = {
            "primary_score": score,
            "horizons": [
                {
                    "horizon_minutes": horizon,
                    "mean_daily_spearman_ic": horizon_scores[horizon],
                    "mean_top_minus_bottom": 0.003 + score,
                    "mean_one_way_turnover": 0.4 + score,
                }
                for horizon in (30, 60, 120)
            ],
        }
        _atomic_write_json(run_dir / "validation_metrics.json", summary)
        pl.DataFrame(daily_rows).write_parquet(
            run_dir / "validation_daily_metrics.parquet"
        )
        recomputed_daily_rows = [
            {key: value for key, value in row.items() if key != "trade_date"}
            for row in daily_rows
        ]
        metrics_by_position[position] = (summary, recomputed_daily_rows)

    stage1_state = tmp_path / "stage1-state.json"
    _atomic_write_json(stage1_state, {"status": "completed"})
    source_configuration = stage2._configuration(
        stage3.STAGE2_PRODUCING_COMMIT, feature_store, stage1_state
    )
    stage2_jobs: list[dict[str, object]] = []
    adopted_key = stage3.ADOPTED_STAGE2_CONTEXT_ABLATION
    adopted_logical = stage3.ADOPTED_STAGE2_LOGICAL_CONFIGURATION
    for base in stage2.stage2_jobs():
        key = str(base["context_ablation"])
        seed = int(base["seed"])
        run_dir = tmp_path / "stage2-runs" / f"{key}_{seed}"
        position = STAGE3_LOGICAL_CONFIGURATION_ORDER.index(
            adopted_logical
        ) * 3 + STAGE3_SEEDS.index(seed)
        score = 0.02 + position / 10_000.0
        manifest_sha: str | None = None
        if key == adopted_key:
            write_completed_run(
                run_dir,
                source_configuration,
                key,
                seed,
                stage3.STAGE2_PRODUCING_COMMIT,
                score,
                position,
            )
            manifest_sha = analyzer._sha256(run_dir / "run_manifest.json")
        stage2_jobs.append(
            {
                **base,
                "status": "completed",
                "result_origin": "trained_stage2",
                "run_dir": str(run_dir),
                "run_manifest_sha256": manifest_sha,
                "producing_git_commit_sha": stage3.STAGE2_PRODUCING_COMMIT,
                "source_stage1_state": None,
                "source_stage1_job": None,
                "primary_validation_ic": score if key == adopted_key else 0.0,
            }
        )
    source_stage2_state = tmp_path / "stage2-state.json"
    _atomic_write_json(
        source_stage2_state,
        {
            "state_version": stage2.STATE_VERSION,
            "sweep_name": stage2.SWEEP_NAME,
            "status": "completed",
            "configuration": source_configuration,
            "jobs": stage2_jobs,
        },
    )

    feature_identity = stage2._feature_store_identity(feature_store)
    monkeypatch.setattr(
        stage3, "PACKAGED_FEATURE_MANIFEST_SHA256", feature_identity["manifest_sha256"]
    )
    monkeypatch.setattr(
        stage3_analyzer,
        "PACKAGED_FEATURE_MANIFEST_SHA256",
        feature_identity["manifest_sha256"],
    )
    isolated_run_root = tmp_path / "isolated-model-runs"
    isolated_run_root.mkdir()
    monkeypatch.setattr(stage3, "RUN_OUTPUT_BASE", isolated_run_root)
    orchestrator_commit = "a" * 40
    configuration = stage3._configuration(
        orchestrator_commit, feature_store, source_stage2_state
    )
    adopted = stage3._validated_stage2_adoptions(source_stage2_state, configuration)
    stage3_state = stage3._new_state(configuration, source_stage2_state, adopted)
    for position, job in enumerate(stage3_state["jobs"]):
        logical = str(job["logical_configuration"])
        seed = int(job["seed"])
        if logical == adopted_logical:
            continue
        run_dir = tmp_path / "stage3-runs" / f"{logical}_{seed}"
        score = 0.02 + position / 10_000.0
        write_completed_run(
            run_dir,
            configuration,
            str(job["context_ablation"]),
            seed,
            orchestrator_commit,
            score,
            position,
        )
        job.update(
            {
                "status": "completed",
                "result_origin": "trained_stage3",
                "run_dir": str(run_dir),
                "run_manifest_sha256": analyzer._sha256(run_dir / "run_manifest.json"),
                "producing_git_commit_sha": orchestrator_commit,
                "primary_validation_ic": score,
                "completed_at_utc": "2026-08-09T00:00:00+00:00",
                "error": None,
            }
        )
    stage3_state["status"] = "completed"
    stage3_state["completed_at_utc"] = "2026-08-09T00:00:00+00:00"
    stage3_state_path = tmp_path / "stage3-state.json"
    _atomic_write_json(stage3_state_path, stage3_state)

    validation_rows = sample_index.filter(
        pl.col("trade_date").is_between(VALIDATION_START, VALIDATION_END)
    ).sort("sample_id")
    sample_id = validation_rows["sample_id"].to_numpy().astype(np.int64)
    date_idx = validation_rows["date_idx"].to_numpy().astype(np.int64)
    decision_idx = validation_rows["decision_idx"].to_numpy().astype(np.int64)
    sample_count = validation_rows.height
    generator = np.random.default_rng(20260809)
    targets = generator.normal(size=(sample_count, 158, 3)).astype(np.float32)
    raw_returns = (targets * 0.01).astype(np.float32)
    label_mask = np.ones_like(targets, dtype=bool)
    evaluations: dict[int, SimpleNamespace] = {}
    for position in range(24):
        job_generator = np.random.default_rng(10_000 + position)
        predictions = (
            targets
            + job_generator.normal(scale=0.2 + position / 200.0, size=targets.shape)
        ).astype(np.float32)
        summary, daily_rows = metrics_by_position[position]
        evaluations[position] = SimpleNamespace(
            observations=EvaluationObservations(
                sample_id=sample_id,
                date_idx=date_idx,
                decision_idx=decision_idx,
                predictions=predictions,
                targets=targets,
                raw_returns=raw_returns,
                label_mask=label_mask,
            ),
            summary=summary,
            daily_rows=daily_rows,
        )

    original_load = np.load
    guarded_feature_arrays = {path.name for path in feature_store.glob("*.npy")}
    accessed_feature_dates: list[int] = []

    class GuardedArray:
        def __init__(self, values: np.ndarray) -> None:
            self.values = values
            self.shape = values.shape

        def __getitem__(self, key: object) -> np.ndarray:
            if not isinstance(key, tuple) or not isinstance(key[0], np.ndarray):
                raise AssertionError(
                    "Feature array was materialized without date slicing"
                )
            selected = np.asarray(key[0], dtype=np.int64)
            if np.any(selected == final_test_index):
                raise AssertionError("Final-test feature observation was accessed")
            accessed_feature_dates.extend(selected.tolist())
            return np.asarray(self.values[key])

        def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
            del args, kwargs
            raise AssertionError("Full feature array was materialized")

    def guarded_load(path: object, **kwargs: object) -> object:
        candidate = Path(path)
        if (
            candidate.parent.resolve() == feature_store.resolve()
            and candidate.name in guarded_feature_arrays
        ):
            return GuardedArray(original_load(candidate, **kwargs))
        return original_load(path, **kwargs)

    inference_calls: list[int] = []
    inference_row_dates: list[set[date]] = []
    runtime_validations: list[bool] = []

    def collect(
        manifest: dict[str, object], store: Path, rows: pl.DataFrame
    ) -> SimpleNamespace:
        assert store.resolve() == feature_store.resolve()
        row_dates = set(rows["trade_date"].to_list())
        assert row_dates <= set(validation_dates)
        assert final_test_date not in row_dates
        inference_row_dates.append(row_dates)
        position = int(manifest["synthetic_index"])
        inference_calls.append(position)
        return evaluations[position]

    monkeypatch.setattr(analyzer.np, "load", guarded_load)
    monkeypatch.setattr(cache_semantics, "collect_neural_evaluation", collect)
    monkeypatch.setattr(
        cache_semantics,
        "validate_runtime",
        lambda: runtime_validations.append(True),
    )
    monkeypatch.setattr(
        inference_semantics, "_git_identity", lambda: ("analysis-commit", True)
    )
    monkeypatch.setattr(analyzer, "BOOTSTRAP_REPLICATIONS", 4)
    monkeypatch.setattr(
        analyzer, "PRODUCTION_TRAINING_LOCK", tmp_path / "production-training.lock"
    )

    immutable_inputs = [
        path
        for root in (
            feature_store,
            tmp_path / "stage2-runs",
            tmp_path / "stage3-runs",
        )
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
    ]
    immutable_inputs.extend([stage1_state, source_stage2_state, stage3_state_path])
    before_inputs = {path: analyzer._sha256(path) for path in immutable_inputs}

    first_run_dir = Path(str(stage3_state["jobs"][0]["run_dir"]))
    persisted_manifest = json.loads(
        (first_run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    persisted_checkpoint = torch.load(
        first_run_dir / "best.pt", map_location="cpu", weights_only=False
    )
    assert isinstance(persisted_manifest["architecture_constants"]["dilations"], list)
    assert isinstance(
        persisted_checkpoint["architecture_constants"]["dilations"], tuple
    )

    core_output = tmp_path / "core-output"
    full_output = tmp_path / "full-output"
    core_manifest_path = run_analysis(stage3_state_path, core_output, "core")
    assert core_manifest_path.is_file()
    assert inference_calls == [0, 1, 2]
    cache_root = tmp_path / "_stock_time_attribution_cache"
    core_cache_bytes = {
        path.relative_to(cache_root): path.read_bytes()
        for path in cache_root.rglob("*")
        if path.is_file()
    }
    for job in stage3_state["jobs"][:3]:
        manifest = json.loads(
            (
                cache_root
                / "predictions"
                / f"{job['logical_configuration']}_seed{job['seed']}"
                / "manifest.json"
            ).read_text(encoding="utf-8")
        )
        assert set(manifest["identity"]["inference_code_sha256"]) == set(
            INFERENCE_CODE_PATHS
        )
        assert manifest["creation_provenance"]["analyzer_git_commit_sha"] == (
            "analysis-commit"
        )
        assert isinstance(
            manifest["creation_provenance"]["analyzer_source_sha256"], str
        )
        _validate_metric_reproduction_gate(manifest["metric_reproduction_gate"])

    full_manifest_path = run_analysis(stage3_state_path, full_output, "full-stage3")
    assert full_manifest_path.is_file()
    assert inference_calls == list(range(24))
    assert runtime_validations == [True, True]
    assert core_cache_bytes == {
        relative: (cache_root / relative).read_bytes() for relative in core_cache_bytes
    }
    assert all(final_test_date not in dates for dates in inference_row_dates)
    assert final_test_index not in accessed_feature_dates
    assert set(accessed_feature_dates) == {
        *range(training_count),
        *validation_indices.tolist(),
    }

    full_state_path = full_output / "analysis_state.json"
    full_state = json.loads(full_state_path.read_text(encoding="utf-8"))
    assert full_state["status"] == "completed"
    assert full_state["pending_artifact"] is None
    assert set(full_state["artifacts"]) == set(analyzer.FINAL_ARTIFACT_NAMES)
    assert all(job["status"] == "completed" for job in full_state["jobs"])
    full_manifest = json.loads(full_manifest_path.read_text(encoding="utf-8"))
    assert full_manifest["analysis_version"] == 4
    assert len(full_manifest["prediction_cache_manifests"]) == 24
    assert full_manifest["configuration"]["inference_code_sha256"] == (
        inference_semantics.inference_code_identity()
    )
    for name, metadata in full_manifest["artifacts"].items():
        artifact = Path(str(metadata["path"]))
        assert artifact.is_file(), name
        assert analyzer._sha256(artifact) == metadata["sha256"]
        assert full_state["artifacts"][name]["sha256"] == metadata["sha256"]
    summary = json.loads((full_output / "summary.json").read_text(encoding="utf-8"))
    assert summary["analysis_version"] == 4
    assert summary["inputs"]["job_count"] == 24
    assert all(count > 0 for count in summary["artifact_row_counts"].values())
    assert all(
        check["passed"] is True for check in summary["metric_reproduction_checks"]
    )
    warnings = " ".join(summary["coverage_warnings"])
    assert "Fixed point-in-time liquidity-quintile IC" in warnings
    assert "Adaptive fallback-bucket IC" in warnings
    assert (
        "Within-liquidity IC is emitted only for adaptive buckets meeting "
        "MIN_IC_EQUITIES." not in warnings
    )
    assert summary["test_data_used"] is False
    assert summary["final_test_remained_sealed"] is True

    cache_bytes_before_resume = {
        path.relative_to(cache_root): path.read_bytes()
        for path in cache_root.rglob("*")
        if path.is_file()
    }
    artifact_hashes_before_resume = {
        name: analyzer._sha256(Path(str(values["path"])))
        for name, values in full_state["artifacts"].items()
    }
    assert (
        run_analysis(stage3_state_path, full_output, "full-stage3")
        == full_manifest_path
    )
    assert inference_calls == list(range(24))
    assert cache_bytes_before_resume == {
        relative: (cache_root / relative).read_bytes()
        for relative in cache_bytes_before_resume
    }
    resumed_state = json.loads(full_state_path.read_text(encoding="utf-8"))
    assert artifact_hashes_before_resume == {
        name: analyzer._sha256(Path(str(values["path"])))
        for name, values in resumed_state["artifacts"].items()
    }
    assert before_inputs == {path: analyzer._sha256(path) for path in immutable_inputs}
