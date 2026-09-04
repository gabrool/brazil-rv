from __future__ import annotations

import copy
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest

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
            scores[day, :, horizon_index] = base + day * 0.01 + horizon * 0.001
            residual[day, :, horizon_index] = base / (names - 1)
            raw_rank[day, :, horizon_index] = base / (names - 1)
            raw_return[day, :, horizon_index] = base * horizon / 10_000.0
    score_mask = np.ones_like(scores, dtype=bool)
    target_mask = np.ones_like(scores, dtype=bool)
    for horizon_index, horizon in enumerate(horizons):
        target_mask[-horizon:, :, horizon_index] = False
    target_mask[:, 0] = False
    return EvaluationInputs(
        dates=dates,
        session_indices=np.arange(100, 100 + len(dates), dtype=np.int64),
        calendar_identity_sha256="b" * 64,
        scores=scores,
        score_mask=score_mask,
        residual_midrank_targets=residual,
        raw_midrank_targets=raw_rank,
        raw_log_returns=raw_return,
        target_mask=target_mask,
        raw_target_mask=target_mask.copy(),
        active=np.ones((len(dates), names), dtype=bool),
        total_return_close=np.full((len(dates), names), 100.0),
        unresolved_action=np.zeros((len(dates), names), dtype=bool),
        cdi_returns=np.zeros(len(dates)),
        source_artifact_hashes={"store_manifest": "a" * 64},
    )


def test_harness_metrics_match_monotone_hand_fixture() -> None:
    result = evaluate_scores(_fixture(), window_name="F2")
    report = result.report

    assert report["pooled_primary_ic"] == pytest.approx(1.0)
    for row in report["horizon_readouts"]:
        assert row["mean_residual_spearman_ic"] == pytest.approx(1.0)
        assert row["mean_raw_rank_ic"] == pytest.approx(1.0)
        assert row["mean_decile_spread_bps_per_holding_session"] == pytest.approx(54.0)
        assert row["mean_persistence_1_session"] == pytest.approx(1.0)
        assert row["mean_persistence_5_sessions"] == pytest.approx(1.0)
    assert report["official_validation_accessed"] is False
    assert report["test_accessed"] is False


def test_crossfit_persistence_uses_complete_model_paths_not_stitched_boundaries() -> None:
    inputs = _fixture()
    first = np.asarray(inputs.scores).copy()
    second = -first
    stitched = first.copy()
    for day in range(len(inputs.dates)):
        if (day // 5) % 2 == 0:
            stitched[day] = second[day]
    stitched_result = evaluate_scores(
        replace(inputs, scores=stitched), window_name="F2_stitched"
    )
    pathwise_result = evaluate_scores(
        replace(
            inputs,
            scores=stitched,
            pathwise_scores=(first, second),
            pathwise_score_masks=(inputs.score_mask, inputs.score_mask),
        ),
        window_name="F2_pathwise",
    )
    stitched_readout = stitched_result.report["horizon_readouts"][0]
    pathwise_readout = pathwise_result.report["horizon_readouts"][0]
    assert stitched_readout["mean_persistence_1_session"] < 1.0
    assert pathwise_readout["mean_persistence_1_session"] == pytest.approx(1.0)
    summaries = pathwise_result.report["economics"]["summaries"]
    assert all(row["path_model_count"] == 2 for row in summaries)


def test_target_mask_never_becomes_the_economics_score_mask() -> None:
    inputs = _fixture()
    changed_residual = np.asarray(inputs.residual_midrank_targets).copy()
    changed_raw_rank = np.asarray(inputs.raw_midrank_targets).copy()
    changed_raw_return = np.asarray(inputs.raw_log_returns).copy()
    changed_residual[:, 0] = 1e9
    changed_raw_rank[:, 0] = -1e9
    changed_raw_return[:, 0] = 1e9
    changed = replace(
        inputs,
        residual_midrank_targets=changed_residual,
        raw_midrank_targets=changed_raw_rank,
        raw_log_returns=changed_raw_return,
    )

    before = evaluate_scores(inputs, window_name="F2").report
    after = evaluate_scores(changed, window_name="F2").report

    assert before["pooled_primary_ic"] == after["pooled_primary_ic"]
    assert before["horizon_readouts"] == after["horizon_readouts"]
    assert before["economics"] == after["economics"]
    assert before["mask_coverage"]["economics_score_mask_true"] == 25 * 60
    assert (
        before["mask_coverage"]["target_mask_true"]
        < before["mask_coverage"]["score_mask_true"]
    )


def test_raw_rank_ic_uses_its_own_target_mask() -> None:
    inputs = _fixture()
    raw_mask = np.asarray(inputs.raw_target_mask).copy()
    for horizon_index, horizon in enumerate(inputs.horizons):
        raw_mask[: len(inputs.dates) - horizon, 0, horizon_index] = True

    report = evaluate_scores(
        replace(inputs, raw_target_mask=raw_mask), window_name="F2"
    ).report
    first = report["daily_metric_table"][0]

    assert first["residual_valid_name_count"] == 59
    assert first["raw_rank_valid_name_count"] == 60
    assert first["decile_valid_name_count"] == 60


def test_economics_signal_is_primary_head_rank_average_and_excludes_d10() -> None:
    scores = np.zeros((1, 4, 5), dtype=np.float64)
    scores[0, :, 0] = [0.0, 1.0, 2.0, 3.0]
    scores[0, :, 1] = [300.0, 200.0, 100.0, 0.0]
    scores[0, :, 2] = [0.0, 1.0, 2.0, 3.0]
    scores[0, :, 3] = [0.0, 1.0, 2.0, 3.0]
    scores[0, :, 4] = [1e9, -1e9, -2e9, 2e9]
    cube_mask = np.ones_like(scores, dtype=bool)
    matrix = np.ones((1, 4), dtype=bool)
    inputs = EvaluationInputs(
        dates=(date(2024, 1, 2),),
        session_indices=np.asarray([100], dtype=np.int64),
        calendar_identity_sha256="b" * 64,
        scores=scores,
        score_mask=cube_mask,
        residual_midrank_targets=np.zeros_like(scores),
        raw_midrank_targets=np.zeros_like(scores),
        raw_log_returns=np.zeros_like(scores),
        target_mask=np.zeros_like(cube_mask),
        raw_target_mask=np.zeros_like(cube_mask),
        active=matrix,
        total_return_close=np.ones((1, 4)),
        unresolved_action=np.zeros((1, 4), dtype=bool),
        cdi_returns=np.zeros(1),
        source_artifact_hashes={"store_manifest": "a" * 64},
    )

    composite, mask = _economics_signal(inputs)

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


def test_gappy_or_parity_only_session_axis_is_rejected() -> None:
    inputs = _fixture()
    indices = np.asarray(inputs.session_indices).copy()
    indices[5:] += 1

    with pytest.raises(ValueError, match="contiguous canonical sessions"):
        evaluate_scores(
            replace(inputs, session_indices=indices),
            window_name="parity_only",
        )


def test_paired_bootstrap_uses_daily_primary_and_headline_deltas() -> None:
    evaluated = evaluate_scores(_fixture(), window_name="F2")
    baseline = replace(
        evaluated,
        daily_primary_ic=np.zeros(25),
        headline_net_excess_bps=np.zeros(24),
    )
    candidate = replace(
        evaluated,
        daily_primary_ic=np.full(25, 2.0),
        headline_net_excess_bps=np.full(24, 3.0),
    )

    comparison = paired_comparison(candidate, baseline)

    assert comparison["block_length_sessions"] == 20
    assert comparison["replications"] == 10_000
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


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("target_mask", "identical dates, targets, masks, close"),
        ("total_return_close", "identical dates, targets, masks, close"),
        ("unresolved_action", "identical dates, targets, masks, close"),
        ("cdi_returns", "identical dates, targets, masks, close"),
    ],
)
def test_paired_comparison_rejects_different_evaluation_population(
    field: str, message: str
) -> None:
    baseline = evaluate_scores(_fixture(), window_name="F2")
    values = np.asarray(getattr(_fixture(), field)).copy()
    if values.dtype == np.bool_:
        index = (0, 1, 0) if values.ndim == 3 else (0, 1)
        values[index] = ~values[index]
    else:
        values.flat[0] += 1.0
    candidate = evaluate_scores(
        replace(_fixture(), **{field: values}),
        window_name="F2",
    )

    with pytest.raises(ValueError, match=message):
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
    economics = changed_report["economics"]
    assert isinstance(economics, dict)
    contract = economics["contract"]
    assert isinstance(contract, dict)
    contract["rank_band"] = 0.4
    changed_contract = replace(baseline, report=changed_report)
    with pytest.raises(ValueError, match="identical economics contract"):
        paired_comparison(changed_contract, baseline)


def test_evaluation_hashes_and_reports_unresolved_economic_intervals() -> None:
    inputs = _fixture()
    unresolved = np.asarray(inputs.unresolved_action).copy()
    unresolved[1, 0] = True

    result = evaluate_scores(
        replace(inputs, unresolved_action=unresolved), window_name="F2"
    )
    report = result.report
    economics = report["economics"]
    assert isinstance(economics, dict)
    daily = economics["daily_table"]
    assert isinstance(daily, list)
    first_headline = next(
        row
        for row in daily
        if row["cost_bps_per_side"] == 4.0 and row["annual_borrow_rate"] == 0.02
    )
    assert first_headline["interval_valid"] is False
    assert first_headline["unresolved_action_position_count"] == 1
    assert report["mask_coverage"]["unresolved_action_true"] == 1
    assert "unresolved_action" in report["input_hashes"]
