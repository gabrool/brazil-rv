from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

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
    METRIC_REPRODUCTION_ABSOLUTE_TOLERANCE,
    AnalysisInputs,
    Stage3AnalysisJob,
    STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION,
    STAGE3_LOGICAL_CONFIGURATION_ORDER,
    _atomic_write_json,
    _atomic_write_npy,
    _adopt_or_infer_caches,
    _build_context_time_deltas,
    _build_core_outputs,
    _build_liquidity_outputs,
    _build_opening_regimes,
    _daily_grid,
    _job_cache_identity,
    _load_analysis_metadata,
    _paired_delta_row,
    _point_in_time_bucket_contribution_vector,
    _record_artifact,
    _reject_test_derived_path,
    _remove_recognized_partial_cache,
    _stock_identity_rows,
    _validate_cache_manifest,
    adaptive_liquidity_buckets,
    additive_spearman_contributions,
    aggregate_additive_contributions,
    causal_observation_completeness,
    deterministic_liquidity_buckets,
    economic_stock_attribution,
    economic_window_accounting,
    learn_overnight_thresholds,
    metric_reproduction_gate,
    moving_block_bootstrap,
    moving_block_bootstrap_matrix,
    moving_block_bootstrap_indices,
    named_time_scopes,
    overnight_regimes,
    parse_args,
    run_analysis,
    per_stock_time_series_skill,
    primary_time_bins,
    standardized_rank_scores,
    STAGE3_SEEDS,
)
from brazil_rv.modeling.metrics import average_ranks, create_metric_table
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


def test_metric_reproduction_gate_passes_and_fails_closed(tmp_path: Path) -> None:
    predictions, targets, mask = _random_arrays(sample_count=4)
    returns = np.random.default_rng(4).normal(size=predictions.shape).astype(np.float32)
    dates = np.asarray([10, 10, 11, 11], dtype=np.int64)
    decisions = np.asarray([0, 1, 0, 1], dtype=np.int64)
    summary, daily_rows = create_metric_table(
        predictions, targets, returns, mask, dates, decisions
    )
    (tmp_path / "validation_metrics.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    pl.DataFrame(daily_rows).write_parquet(
        tmp_path / "validation_daily_metrics.parquet"
    )
    result = metric_reproduction_gate(tmp_path, summary, daily_rows)
    assert result["passed"] is True
    mutated = dict(summary)
    mutated["primary_score"] = float(summary["primary_score"]) + 1e-5
    with pytest.raises(ValueError, match="parity"):
        metric_reproduction_gate(tmp_path, mutated, daily_rows)
    assert METRIC_REPRODUCTION_ABSOLUTE_TOLERANCE < 1e-5


def test_cache_manifest_rejects_nested_identity_hash_shape_and_dtype(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    predictions = np.zeros((2, 3, 1), dtype=np.float32)
    prediction_path = cache / "predictions.npy"
    _atomic_write_npy(prediction_path, predictions)
    identity = {
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
                "prediction_file": {
                    "name": prediction_path.name,
                    "sha256": __import__("hashlib")
                    .sha256(prediction_path.read_bytes())
                    .hexdigest(),
                },
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
    write_manifest(identity)
    prediction_path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        _validate_cache_manifest(manifest_path, identity)


def test_inference_failure_is_recorded_without_mutating_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
        validation_rows=pl.DataFrame({"sample_id": [7]}),
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
        "brazil_rv.modeling.analyze_stock_time_attribution.validate_runtime",
        lambda: None,
    )

    def fail_inference(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("synthetic inference failure")

    monkeypatch.setattr(
        "brazil_rv.modeling.analyze_stock_time_attribution.collect_neural_evaluation",
        fail_inference,
    )
    with pytest.raises(RuntimeError, match="synthetic inference failure"):
        _adopt_or_infer_caches(
            tmp_path / "output",
            inputs,
            "core",
            state,
            state_path,
        )
    assert state["jobs"][0]["status"] == "failed"
    assert state["jobs"][0]["error"] == "RuntimeError: synthetic inference failure"
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["jobs"][0]["status"] == "failed"
    assert run_marker.read_text(encoding="utf-8") == "run"
    assert feature_marker.read_text(encoding="utf-8") == "feature"


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
        inference_code_sha256={"module.py": "hash"},
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
    assert identity["inference_code_sha256"] == {"module.py": "hash"}

    cache = tmp_path / "legacy-cache"
    cache.mkdir()
    manifest_path = cache / "manifest.json"
    dirty_identity = {
        **identity,
        "analyzer_worktree_clean": False,
    }
    _atomic_write_json(
        manifest_path,
        {"status": "completed", "identity": dirty_identity},
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
            "observed_bars": np.ones((sample_count, equity_count), dtype=np.int64),
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
        == 5 * 3
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
        "core_b3_bars_before_decision",
        "core_retained_local_completeness",
        "core_retained_global_completeness",
        "core_retained_global_freshness",
        "core_retained_context_preopen",
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
        "b3_bars_before_decision",
        "retained_local_completeness",
        "retained_global_completeness",
        "retained_global_freshness",
        "retained_context_preopen",
        "overnight_regime",
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
