from __future__ import annotations

import copy
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest

from brazil_rv.v2.config import TRIAGE_PROTOCOL
from brazil_rv.v2.evaluate import (
    EvaluationInputs,
    _economics_signal,
    evaluate_scores,
    paired_comparison,
    write_evaluation_report,
)


def _weekdays(start: date, count: int) -> tuple[date, ...]:
    values: list[date] = []
    current = start
    while len(values) < count:
        if current.weekday() < 5:
            values.append(current)
        current += timedelta(days=1)
    return tuple(values)


def _fixture() -> EvaluationInputs:
    dates = _weekdays(date(2024, 1, 2), 25)
    names = 60
    horizons = (1, 2, 3, 5, 10)
    base = np.arange(names, dtype=np.float64)
    scores = np.empty((len(dates), names, len(horizons)), dtype=np.float64)
    residual = np.empty_like(scores)
    raw_rank = np.empty_like(scores)
    raw_return = np.empty_like(scores)
    for day in range(len(dates)):
        for horizon_index, horizon in enumerate(horizons):
            phase = 0.23 * day + 0.11 * horizon_index
            score = np.sin(base * 0.19 + phase) + 0.35 * np.cos(
                base * 0.07 - 0.31 * day
            )
            target = 0.55 * score + np.cos(base * 0.31 + 0.17 * day)
            scores[day, :, horizon_index] = score
            residual[day, :, horizon_index] = target
            raw_rank[day, :, horizon_index] = target
            raw_return[day, :, horizon_index] = target * horizon / 10_000.0
    score_mask = np.ones_like(scores, dtype=bool)
    target_mask = np.ones_like(scores, dtype=bool)
    for horizon_index, horizon in enumerate(horizons):
        target_mask[-horizon:, :, horizon_index] = False
    target_mask[:, 0] = False
    matrix_shape = (len(dates), names)
    return EvaluationInputs(
        dates=dates,
        session_indices=np.arange(100, 100 + len(dates), dtype=np.int64),
        calendar_identity_sha256="b" * 64,
        scores=scores,
        score_mask=score_mask,
        median_residual_midrank_targets=residual,
        raw_midrank_targets=raw_rank,
        raw_log_returns=raw_return,
        target_mask=target_mask,
        raw_target_mask=target_mask.copy(),
        active=np.ones(matrix_shape, dtype=bool),
        adjusted_close=np.full(matrix_shape, 100.0),
        neutralized_log_return=np.zeros(matrix_shape),
        neutralized_log_return_valid=np.ones(matrix_shape, dtype=bool),
        return_neutralized_event=np.zeros(matrix_shape, dtype=bool),
        cross_sectional_median_log_return=np.zeros(len(dates)),
        target_scale_sigma=np.full(matrix_shape, 0.02),
        prior_feature_values={
            "yang_zhang_vol_20": np.broadcast_to(base, matrix_shape),
            "beta_60": np.broadcast_to(np.sin(base), matrix_shape),
            "log_volume_mean_20": np.broadcast_to(np.log1p(base), matrix_shape),
            "momentum_12_1": np.broadcast_to(np.cos(base), matrix_shape),
            "log_return_5": np.broadcast_to(base[::-1], matrix_shape),
        },
        cdi_returns=np.zeros(len(dates)),
        source_artifact_hashes={"store_manifest": "a" * 64},
    )


def test_harness_metrics_are_nontrivial_on_rotating_fixture() -> None:
    report = evaluate_scores(_fixture(), window_name="F2").report

    assert 0.0 < report["pooled_primary_median_residual_ic"] < 1.0
    for row in report["horizon_readouts"]:
        assert 0.0 < abs(row["mean_median_residual_spearman_ic"]) < 1.0
        assert 0.0 < abs(row["mean_raw_rank_ic"]) < 1.0
        assert row["mean_decile_spread_bps_per_holding_session"] != 0.0
        assert 0.0 < abs(row["mean_persistence_1_session"]) < 1.0
        assert 0.0 < abs(row["mean_persistence_5_sessions"]) < 1.0
    assert len(report["diagnostics"]["incremental_horizon_ic"]) == 20
    assert len(report["diagnostics"]["matched_universe_ic"]) == 5
    assert report["official_validation_accessed"] is False
    assert report["test_accessed"] is False


def test_target_mask_never_becomes_the_economics_score_mask() -> None:
    inputs = _fixture()
    changed_residual = np.asarray(inputs.median_residual_midrank_targets).copy()
    changed_raw_rank = np.asarray(inputs.raw_midrank_targets).copy()
    changed_raw_return = np.asarray(inputs.raw_log_returns).copy()
    changed_residual[:, 0] = 1e9
    changed_raw_rank[:, 0] = -1e9
    changed_raw_return[:, 0] = 1e9
    changed = replace(
        inputs,
        median_residual_midrank_targets=changed_residual,
        raw_midrank_targets=changed_raw_rank,
        raw_log_returns=changed_raw_return,
    )

    before = evaluate_scores(inputs, window_name="F2").report
    after = evaluate_scores(changed, window_name="F2").report

    assert before["pooled_primary_median_residual_ic"] == after[
        "pooled_primary_median_residual_ic"
    ]
    assert before["horizon_readouts"] == after["horizon_readouts"]
    assert before["economics"] == after["economics"]
    assert before["mask_coverage"]["economics_score_mask_true"] == 25 * 60


def test_raw_rank_ic_uses_its_own_target_mask() -> None:
    inputs = _fixture()
    raw_mask = np.asarray(inputs.raw_target_mask).copy()
    for horizon_index, horizon in enumerate(inputs.horizons):
        raw_mask[: len(inputs.dates) - horizon, 0, horizon_index] = True

    report = evaluate_scores(
        replace(inputs, raw_target_mask=raw_mask), window_name="F2"
    ).report
    first = report["daily_metric_table"][0]

    assert first["median_residual_valid_name_count"] == 59
    assert first["raw_rank_valid_name_count"] == 60
    assert first["decile_valid_name_count"] == 60


def test_economics_signal_is_primary_head_rank_average_and_excludes_d10() -> None:
    inputs = _fixture()
    scores = np.zeros_like(inputs.scores[:1, :4])
    scores[0, :, 0] = [0.0, 1.0, 2.0, 3.0]
    scores[0, :, 1] = [300.0, 200.0, 100.0, 0.0]
    scores[0, :, 2] = [0.0, 1.0, 2.0, 3.0]
    scores[0, :, 3] = [0.0, 1.0, 2.0, 3.0]
    scores[0, :, 4] = [1e9, -1e9, -2e9, 2e9]
    tiny = replace(
        inputs,
        dates=inputs.dates[:1],
        session_indices=np.asarray([100], dtype=np.int64),
        scores=scores,
        score_mask=np.ones_like(scores, dtype=bool),
        active=np.ones((1, 4), dtype=bool),
    )

    composite, mask = _economics_signal(tiny)

    np.testing.assert_allclose(composite, [[-0.375, -0.125, 0.125, 0.375]])
    assert mask.all()


def test_report_json_is_byte_deterministic(tmp_path) -> None:
    first = evaluate_scores(_fixture(), window_name="F2")
    second = evaluate_scores(_fixture(), window_name="F2")
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    first_sha = write_evaluation_report(first_path, first)
    second_sha = write_evaluation_report(second_path, second)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_sha == second_sha
    assert first_path.with_suffix(".json.sha256").is_file()


def test_official_gate_runs_before_array_validation() -> None:
    malformed_official = replace(_fixture(), dates=(date(2025, 1, 2),))

    with pytest.raises(PermissionError, match="preregistration"):
        evaluate_scores(malformed_official, window_name="official")


def test_gappy_session_axis_is_rejected() -> None:
    inputs = _fixture()
    indices = np.asarray(inputs.session_indices).copy()
    indices[5:] += 1

    with pytest.raises(ValueError, match="contiguous canonical sessions"):
        evaluate_scores(replace(inputs, session_indices=indices), window_name="gappy")


def test_paired_bootstrap_uses_daily_primary_and_headline_deltas() -> None:
    evaluated = evaluate_scores(_fixture(), window_name="F2")
    baseline = replace(
        evaluated,
        daily_primary_ic=np.zeros(25),
        headline_net_excess_bps=np.zeros(25),
    )
    candidate = replace(
        evaluated,
        daily_primary_ic=np.full(25, 2.0),
        headline_net_excess_bps=np.full(25, 3.0),
    )

    comparison = paired_comparison(candidate, baseline)

    assert comparison["daily_primary_ic_delta"] == {
        "estimate": 2.0,
        "lower_95": 2.0,
        "upper_95": 2.0,
    }
    assert comparison["daily_headline_net_excess_bps_delta"] == {
        "estimate": 3.0,
        "lower_95": 3.0,
        "upper_95": 3.0,
    }
    triage = paired_comparison(candidate, baseline, protocol=TRIAGE_PROTOCOL)
    assert triage["replications"] == 0
    assert triage["daily_primary_ic_delta"]["lower_95"] is None


@pytest.mark.parametrize(
    "field",
    [
        "target_mask",
        "adjusted_close",
        "return_neutralized_event",
        "neutralized_log_return",
        "cdi_returns",
    ],
)
def test_paired_comparison_rejects_different_evaluation_population(
    field: str,
) -> None:
    baseline = evaluate_scores(_fixture(), window_name="F2")
    values = np.asarray(getattr(_fixture(), field)).copy()
    if values.dtype == np.bool_:
        index = (0, 1, 0) if values.ndim == 3 else (0, 1)
        values[index] = ~values[index]
    else:
        values.flat[0] += 1.0
    candidate = evaluate_scores(replace(_fixture(), **{field: values}), window_name="F2")

    with pytest.raises(ValueError, match="identical dates, targets, masks"):
        paired_comparison(candidate, baseline)


def test_paired_comparison_rejects_source_or_economics_contract_mismatch() -> None:
    baseline = evaluate_scores(_fixture(), window_name="F2")
    other_source = evaluate_scores(
        replace(_fixture(), source_artifact_hashes={"store_manifest": "c" * 64}),
        window_name="F2",
    )
    with pytest.raises(ValueError, match="identical source identities"):
        paired_comparison(other_source, baseline)

    changed_report = copy.deepcopy(baseline.report)
    changed_report["economics"]["contract"]["rank_band"] = 0.4
    changed_contract = replace(baseline, report=changed_report)
    with pytest.raises(ValueError, match="identical economics contract"):
        paired_comparison(changed_contract, baseline)


def test_ledger_reports_every_evaluation_day_even_with_an_event() -> None:
    inputs = _fixture()
    events = np.asarray(inputs.return_neutralized_event).copy()
    closes = np.asarray(inputs.adjusted_close).copy()
    events[1, 0] = True
    closes[1:, 0] = 80.0

    report = evaluate_scores(
        replace(inputs, return_neutralized_event=events, adjusted_close=closes),
        window_name="F2",
    ).report
    headline = [
        row
        for row in report["economics"]["daily_table"]
        if row["cost_bps_per_side"] == 4.0
        and row["annual_borrow_rate"] == 0.02
    ]
    assert len(headline) == len(inputs.dates)
    assert all(row["interval_valid"] is True for row in headline)
    assert report["mask_coverage"]["return_neutralized_event_true"] == 1
    assert "return_neutralized_event" in report["input_hashes"]
