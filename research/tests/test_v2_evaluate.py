from __future__ import annotations

import copy
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest

import brazil_rv.v2.evaluate as evaluate_module
from brazil_rv.v2.config import TRIAGE_PROTOCOL
from brazil_rv.v2.evaluate import (
    EvaluationInputs,
    _bootstrap_payload,
    _diagnostics,
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
    scaled_rank = np.empty_like(scores)
    shareholder_rank = np.empty_like(scores)
    shareholder_return = np.empty_like(scores)
    price_rank = np.empty_like(scores)
    for day in range(len(dates)):
        for horizon_index, horizon in enumerate(horizons):
            phase = 0.23 * day + 0.11 * horizon_index
            score = np.sin(base * 0.19 + phase) + 0.35 * np.cos(
                base * 0.07 - 0.31 * day
            )
            target = 0.55 * score + np.cos(base * 0.31 + 0.17 * day)
            scores[day, :, horizon_index] = score
            scaled_rank[day, :, horizon_index] = target
            shareholder_rank[day, :, horizon_index] = target
            shareholder_return[day, :, horizon_index] = target * horizon / 10_000.0
            price_rank[day, :, horizon_index] = target
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
        scaled_midrank_targets=scaled_rank,
        scaled_target_mask=target_mask,
        neutral_midrank_targets=scaled_rank.copy(),
        neutral_target_mask=target_mask.copy(),
        shareholder_midrank_targets=shareholder_rank,
        shareholder_simple_returns=shareholder_return,
        shareholder_target_mask=target_mask.copy(),
        price_midrank_targets=price_rank,
        price_target_mask=target_mask.copy(),
        active=np.ones(matrix_shape, dtype=bool),
        raw_close=np.full(matrix_shape, 100.0),
        action_shares_per_prior_share=np.ones(matrix_shape),
        action_cash_per_prior_share=np.zeros(matrix_shape),
        action_session_resolved=np.ones(matrix_shape, dtype=bool),
        action_has_action=np.zeros(matrix_shape, dtype=bool),
        action_successor_index=np.broadcast_to(
            np.arange(names, dtype=np.int64), matrix_shape
        ).copy(),
        action_payment_session=np.full(matrix_shape, -1, dtype=np.int64),
        security_ids=tuple(f"SEC-{index}" for index in range(names)),
        target_scale_sigma=np.full(matrix_shape, 0.02),
        prior_feature_values={
            "yang_zhang_vol_20": np.broadcast_to(base, matrix_shape),
            "beta_60": np.broadcast_to(np.sin(base), matrix_shape),
            "log_volume_mean_20": np.broadcast_to(np.log1p(base), matrix_shape),
            "momentum_12_1": np.broadcast_to(np.cos(base), matrix_shape),
            "log_return_5": np.broadcast_to(base[::-1], matrix_shape),
        },
        cdi_returns=np.zeros(len(dates)),
        transfer_chronology_clean=True,
        source_artifact_hashes={"store_manifest": "a" * 64},
        annual_borrow_rate_by_name=np.full(matrix_shape, 0.02),
        shortable=np.ones(matrix_shape, dtype=np.bool_),
    )


def test_harness_metrics_are_nontrivial_on_rotating_fixture() -> None:
    report = evaluate_scores(_fixture(), window_name="F2").report

    assert 0.0 < report["mean_daily_primary_neutral_target_ic"] < 1.0
    for row in report["horizon_readouts"]:
        assert 0.0 < abs(row["mean_neutral_target_spearman_ic"]) < 1.0
        assert 0.0 < abs(row["mean_shareholder_rank_ic"]) < 1.0
        assert 0.0 < abs(row["mean_price_return_rank_ic"]) < 1.0
        assert row["mean_shareholder_return_spread_total_bps"] != 0.0
        assert row["mean_shareholder_return_spread_bps_per_holding_session"] != 0.0
        assert 0.0 < abs(row["mean_persistence_1_session"]) < 1.0
        assert 0.0 < abs(row["mean_persistence_5_sessions"]) < 1.0
    assert len(report["diagnostics"]["incremental_horizon_ic"]) == 20
    assert len(report["diagnostics"]["matched_universe_ic"]) == 5
    assert report["official_validation_accessed"] is False
    assert report["test_accessed"] is False


def test_target_mask_never_becomes_the_economics_score_mask() -> None:
    inputs = _fixture()
    changed_scaled = np.asarray(inputs.scaled_midrank_targets).copy()
    changed_shareholder_rank = np.asarray(inputs.shareholder_midrank_targets).copy()
    changed_shareholder_return = np.asarray(inputs.shareholder_simple_returns).copy()
    changed_price_rank = np.asarray(inputs.price_midrank_targets).copy()
    changed_scaled[:, 0] = 1e9
    changed_shareholder_rank[:, 0] = -1e9
    changed_shareholder_return[:, 0] = 1e9
    changed_price_rank[:, 0] = -1e9
    changed = replace(
        inputs,
        scaled_midrank_targets=changed_scaled,
        shareholder_midrank_targets=changed_shareholder_rank,
        shareholder_simple_returns=changed_shareholder_return,
        price_midrank_targets=changed_price_rank,
    )

    before = evaluate_scores(inputs, window_name="F2").report
    after = evaluate_scores(changed, window_name="F2").report

    assert (
        before["mean_daily_primary_neutral_target_ic"]
        == after["mean_daily_primary_neutral_target_ic"]
    )
    assert [
        row["mean_neutral_target_spearman_ic"] for row in before["horizon_readouts"]
    ] == [row["mean_neutral_target_spearman_ic"] for row in after["horizon_readouts"]]
    assert before["economics"] == after["economics"]
    assert before["mask_coverage"]["economics_score_mask_true"] == 25 * 60


def test_return_family_ics_use_their_own_target_masks() -> None:
    inputs = _fixture()
    shareholder_mask = np.asarray(inputs.shareholder_target_mask).copy()
    price_mask = np.asarray(inputs.price_target_mask).copy()
    for horizon_index, horizon in enumerate(inputs.horizons):
        shareholder_mask[: len(inputs.dates) - horizon, 0, horizon_index] = True

    report = evaluate_scores(
        replace(
            inputs,
            shareholder_target_mask=shareholder_mask,
            price_target_mask=price_mask,
        ),
        window_name="F2",
    ).report
    first = report["daily_metric_table"][0]

    assert first["neutral_target_valid_name_count"] == 59
    assert first["shareholder_rank_valid_name_count"] == 60
    assert first["price_return_rank_valid_name_count"] == 59
    assert first["shareholder_return_spread_valid_name_count"] == 60


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


def test_incremental_diagnostic_uses_same_holding_shareholder_wealth() -> None:
    inputs = _fixture()
    original = _diagnostics(inputs)["incremental_horizon_ic"]
    altered_price_rank = _diagnostics(
        replace(
            inputs,
            price_midrank_targets=-np.asarray(inputs.price_midrank_targets),
        )
    )["incremental_horizon_ic"]
    shareholder = np.asarray(inputs.shareholder_simple_returns).copy()
    shareholder[..., 2] *= -1.0
    altered_shareholder = _diagnostics(
        replace(inputs, shareholder_simple_returns=shareholder)
    )["incremental_horizon_ic"]

    assert altered_price_rank == original
    assert altered_shareholder != original
    assert {row["return_basis"] for row in original} == {
        "shareholder_wealth_same_holding"
    }


def test_block_bootstrap_receives_missing_calendar_dates(monkeypatch) -> None:
    observed = None

    def capture(values, *, replications, block_length, seed):
        nonlocal observed
        observed = np.asarray(values).copy()
        return {
            "estimate": np.asarray([2.0]),
            "lower_95": np.asarray([1.0]),
            "upper_95": np.asarray([3.0]),
        }

    monkeypatch.setattr(evaluate_module, "moving_block_bootstrap", capture)
    values = np.asarray([1.0, np.nan, 3.0, np.nan, 5.0])

    result = _bootstrap_payload(values, replications=100, block_length=2)

    np.testing.assert_array_equal(observed, values)
    assert result["possible_date_count"] == 5
    assert result["defined_date_count"] == 3


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
        primary_scores=-evaluated.primary_targets,
        headline_net_excess_bps=np.zeros(25),
    )
    candidate = replace(
        evaluated,
        primary_scores=evaluated.primary_targets.copy(),
        headline_net_excess_bps=np.full(25, 3.0),
    )

    comparison = paired_comparison(candidate, baseline)

    assert comparison["daily_primary_ic_delta"] == {
        "estimate": 2.0,
        "lower_95": 2.0,
        "upper_95": 2.0,
        "possible_date_count": 25,
        "defined_date_count": 20,
        "undefined_reason": None,
    }
    assert comparison["daily_headline_net_excess_bps_delta"] == {
        "estimate": 3.0,
        "lower_95": 3.0,
        "upper_95": 3.0,
        "possible_date_count": 25,
        "defined_date_count": 25,
        "undefined_reason": None,
    }
    triage = paired_comparison(candidate, baseline, protocol=TRIAGE_PROTOCOL)
    assert triage["replications"] == 0
    assert triage["daily_primary_ic_delta"]["lower_95"] is None


@pytest.mark.parametrize(
    "field",
    [
        "scaled_target_mask",
        "shareholder_target_mask",
        "price_target_mask",
        "raw_close",
        "action_session_resolved",
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
    candidate = evaluate_scores(
        replace(_fixture(), **{field: values}), window_name="F2"
    )

    with pytest.raises(ValueError, match="identical dates, outcome targets/masks"):
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


def test_paired_comparison_rejects_different_contractual_action_timeline() -> None:
    baseline = evaluate_scores(_fixture(), window_name="F2")
    inputs = _fixture()
    cash = np.asarray(inputs.action_cash_per_prior_share).copy()
    has_action = np.asarray(inputs.action_has_action).copy()
    payment = np.asarray(inputs.action_payment_session).copy()
    cash[2, 0] = 1.0
    has_action[2, 0] = True
    payment[2, 0] = 3
    candidate = evaluate_scores(
        replace(
            inputs,
            action_cash_per_prior_share=cash,
            action_has_action=has_action,
            action_payment_session=payment,
        ),
        window_name="F2",
    )

    with pytest.raises(ValueError, match="contractual actions/payment sessions"):
        paired_comparison(candidate, baseline)


def test_ledger_reports_every_evaluation_day_with_a_paid_cash_action() -> None:
    inputs = _fixture()
    cash = np.asarray(inputs.action_cash_per_prior_share).copy()
    has_action = np.asarray(inputs.action_has_action).copy()
    payment = np.asarray(inputs.action_payment_session).copy()
    cash[2, 0] = 1.0
    has_action[2, 0] = True
    payment[2, 0] = 3

    report = evaluate_scores(
        replace(
            inputs,
            action_cash_per_prior_share=cash,
            action_has_action=has_action,
            action_payment_session=payment,
        ),
        window_name="F2",
    ).report
    headline = [
        row
        for row in report["economics"]["daily_table"]
        if row["cost_bps_per_side"] == 4.0
        and row["annual_borrow_rate"] == 0.02
        and row["borrow_source"] == "uniform"
    ]
    assert len(headline) == len(inputs.dates)
    assert {row["economics_resolved"] for row in headline} == {
        not report["economics"]["headline"]["economics_unresolved"]
    }
    assert report["mask_coverage"]["action_name_days"] == 1
    assert report["mask_coverage"]["known_action_payment_name_days"] == 1
    assert "action_payment_session" in report["input_hashes"]


def test_primary_uses_one_four_head_population_and_twenty_name_minimum() -> None:
    inputs = _fixture()
    target_mask = np.asarray(inputs.neutral_target_mask).copy()
    target_mask[0, :, :4] = False
    target_mask[0, :20, :4] = True
    # Each head has ten additional names, but those head-specific names must
    # not enter any of the four primary correlations.
    for head in range(4):
        target_mask[0, 20 + 10 * head : 30 + 10 * head, head] = True

    report = evaluate_scores(
        replace(inputs, neutral_target_mask=target_mask), window_name="F2"
    ).report
    primary = report["daily_primary_ic"][0]
    primary_rows = [
        row
        for row in report["daily_metric_table"]
        if row["date"] == inputs.dates[0].isoformat()
        and row["horizon_sessions"] in (1, 2, 3, 5)
    ]

    assert primary["common_neutral_outcome_name_count"] == 20
    assert primary["common_score_and_outcome_name_count"] == 20
    assert primary["used"] is True
    assert primary["undefined_reason"] is None
    assert {row["neutral_target_valid_name_count"] for row in primary_rows} == {20}


def test_primary_reports_why_a_day_is_undefined_instead_of_averaging_heads() -> None:
    inputs = _fixture()
    scores = np.asarray(inputs.scores).copy()
    scores[0, :, 2] = 1.0

    report = evaluate_scores(replace(inputs, scores=scores), window_name="F2").report
    first = report["daily_primary_ic"][0]

    assert first["primary_neutral_target_ic"] is None
    assert first["used"] is False
    assert first["undefined_reason"] == "D3:constant_score"
    assert first["head_neutral_target_spearman_ic"]["D1"] is not None
    assert first["head_neutral_target_spearman_ic"]["D3"] is None


def test_primary_reports_insufficient_common_support() -> None:
    inputs = _fixture()
    target_mask = np.asarray(inputs.neutral_target_mask).copy()
    target_mask[0, :, :4] = False
    target_mask[0, :19, :4] = True

    report = evaluate_scores(
        replace(inputs, neutral_target_mask=target_mask), window_name="F2"
    ).report
    first = report["daily_primary_ic"][0]

    assert first["common_neutral_outcome_name_count"] == 19
    assert first["primary_neutral_target_ic"] is None
    assert first["undefined_reason"] == "fewer_than_20_common_neutral_outcomes"


def test_all_five_horizon_diagnostic_intersects_d10_score_and_outcome() -> None:
    inputs = _fixture()
    target_mask = np.asarray(inputs.scaled_target_mask).copy()
    score_mask = np.asarray(inputs.score_mask).copy()
    target_mask[0, :, 4] = False
    score_mask[0, :, 4] = False
    target_mask[0, 1:21, 4] = True
    score_mask[0, 1:21, 4] = True

    report = evaluate_scores(
        replace(inputs, scaled_target_mask=target_mask, score_mask=score_mask),
        window_name="F2",
    ).report
    rows = [
        row
        for row in report["diagnostics"]["matched_universe_daily"]
        if row["date"] == inputs.dates[0].isoformat()
    ]

    assert len(rows) == 5
    assert {row["all_five_common_score_and_outcome_name_count"] for row in rows} == {20}
    assert all(row["scaled_target_spearman_ic"] is not None for row in rows)

    score_mask[0, 20, 4] = False
    failed = evaluate_scores(
        replace(inputs, scaled_target_mask=target_mask, score_mask=score_mask),
        window_name="F2",
    ).report
    failed_rows = [
        row
        for row in failed["diagnostics"]["matched_universe_daily"]
        if row["date"] == inputs.dates[0].isoformat()
    ]
    assert {
        row["all_five_common_score_and_outcome_name_count"] for row in failed_rows
    } == {19}
    assert all(
        row["undefined_reason"] == "fewer_than_20_valid_names" for row in failed_rows
    )


def test_scaled_shareholder_and_price_rank_ics_are_separate() -> None:
    inputs = _fixture()
    shareholder = -np.asarray(inputs.scaled_midrank_targets)
    price = np.ones_like(inputs.price_midrank_targets)

    report = evaluate_scores(
        replace(
            inputs,
            shareholder_midrank_targets=shareholder,
            price_midrank_targets=price,
        ),
        window_name="F2",
    ).report
    first = report["daily_metric_table"][0]

    assert first["neutral_target_spearman_ic"] > 0.0
    assert first["legacy_scaled_target_ic"] > 0.0
    assert first["shareholder_rank_ic"] < 0.0
    assert first["price_return_rank_ic"] is None
    assert first["price_return_rank_ic_undefined_reason"] == "constant_target"


def test_paired_comparison_intersects_different_candidate_score_populations() -> None:
    inputs = _fixture()
    candidate_mask = np.zeros_like(inputs.score_mask)
    baseline_mask = np.zeros_like(inputs.score_mask)
    candidate_mask[:, :40] = True
    baseline_mask[:, 20:] = True
    candidate = evaluate_scores(
        replace(inputs, score_mask=candidate_mask), window_name="F2"
    )
    baseline = evaluate_scores(
        replace(inputs, score_mask=baseline_mask), window_name="F2"
    )

    comparison = paired_comparison(candidate, baseline, protocol=TRIAGE_PROTOCOL)
    first = comparison["daily_primary_ic_delta_table"][0]

    assert first["common_candidate_baseline_name_count"] == 20
    assert first["delta"] == pytest.approx(0.0)
    assert comparison["primary_population"]["used_date_count"] == 20


def test_paired_comparison_preserves_undefined_dates_and_reason() -> None:
    inputs = _fixture()
    candidate_mask = np.zeros_like(inputs.score_mask)
    baseline_mask = np.zeros_like(inputs.score_mask)
    candidate_mask[:, :30] = True
    baseline_mask[:, 15:45] = True
    candidate = evaluate_scores(
        replace(inputs, score_mask=candidate_mask), window_name="F2"
    )
    baseline = evaluate_scores(
        replace(inputs, score_mask=baseline_mask), window_name="F2"
    )

    comparison = paired_comparison(candidate, baseline, protocol=TRIAGE_PROTOCOL)

    assert comparison["daily_primary_ic_delta"]["estimate"] is None
    assert comparison["daily_primary_ic_delta"]["defined_date_count"] == 0
    assert (
        comparison["daily_primary_ic_delta"]["undefined_reason"]
        == "no_defined_daily_values"
    )
    assert (
        comparison["daily_primary_ic_delta_table"][0][
            "common_candidate_baseline_name_count"
        ]
        == 15
    )
    assert (
        "fewer_than_20_common_scores"
        in comparison["daily_primary_ic_delta_table"][0]["undefined_reason"]
    )


def test_paired_comparison_refuses_stale_evaluation_schema() -> None:
    evaluated = evaluate_scores(_fixture(), window_name="F2")
    stale_report = copy.deepcopy(evaluated.report)
    stale_report["schema"] = "BRAZIL_RV_V2_EVALUATION_V3"

    with pytest.raises(ValueError, match="stale or incompatible"):
        paired_comparison(replace(evaluated, report=stale_report), evaluated)


def test_paired_comparison_does_not_treat_transfer_flag_as_population_identity() -> (
    None
):
    baseline = evaluate_scores(_fixture(), window_name="F2")
    changed_report = copy.deepcopy(baseline.report)
    changed_report["transfer_chronology_clean"] = False
    changed_report["input_hashes"]["transfer_chronology_clean"] = "c" * 64

    comparison = paired_comparison(
        replace(baseline, report=changed_report),
        baseline,
        protocol=TRIAGE_PROTOCOL,
    )

    assert comparison["daily_primary_ic_delta"]["estimate"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("candidate_unresolved", "baseline_unresolved", "expected_reason"),
    [
        (True, False, "candidate_headline_economics_unresolved"),
        (False, True, "baseline_headline_economics_unresolved"),
        (True, True, "candidate_and_baseline_headline_economics_unresolved"),
    ],
)
def test_paired_economics_is_wholly_undefined_when_either_ledger_is_unresolved(
    candidate_unresolved: bool,
    baseline_unresolved: bool,
    expected_reason: str,
) -> None:
    evaluated = evaluate_scores(_fixture(), window_name="F2")
    candidate_report = copy.deepcopy(evaluated.report)
    baseline_report = copy.deepcopy(evaluated.report)
    candidate_report["economics"]["headline"]["economics_unresolved"] = (
        candidate_unresolved
    )
    baseline_report["economics"]["headline"]["economics_unresolved"] = (
        baseline_unresolved
    )
    candidate = replace(
        evaluated,
        report=candidate_report,
        headline_net_excess_bps=np.full(len(evaluated.dates), 3.0),
    )
    baseline = replace(
        evaluated,
        report=baseline_report,
        headline_net_excess_bps=np.zeros(len(evaluated.dates)),
    )

    comparison = paired_comparison(candidate, baseline, protocol=TRIAGE_PROTOCOL)

    assert comparison["economics_comparison_undefined_reason"] == expected_reason
    assert comparison["daily_headline_net_excess_bps_delta"] == {
        "estimate": None,
        "lower_95": None,
        "upper_95": None,
        "possible_date_count": len(evaluated.dates),
        "defined_date_count": 0,
        "undefined_reason": "no_defined_daily_values",
    }
    assert {
        row["undefined_reason"]
        for row in comparison["daily_headline_net_excess_bps_delta_table"]
    } == {expected_reason}
    assert all(
        row["delta"] is None
        for row in comparison["daily_headline_net_excess_bps_delta_table"]
    )


def test_official_evaluation_refuses_contaminated_transfer_before_array_access(
    tmp_path,
) -> None:
    root = tmp_path / "research" / "preregistrations"
    root.mkdir(parents=True)
    registration = root / "official.md"
    registration.write_text("frozen\n", encoding="utf-8")
    inputs = replace(
        _fixture(),
        dates=_weekdays(date(2025, 1, 2), 25),
        transfer_chronology_clean=False,
    )

    with pytest.raises(PermissionError, match="contaminated transfer chronology"):
        evaluate_scores(
            inputs,
            window_name="official",
            registration_path=registration,
            preregistration_root=root,
        )


def test_invalid_tail_mask_is_rejected_before_target_payload_decode() -> None:
    class UndecodableTarget:
        shape = _fixture().scores.shape

        def __array__(self, *args, **kwargs):
            raise AssertionError("numeric target payload was decoded")

    inputs = _fixture()
    invalid_mask = np.asarray(inputs.scaled_target_mask).copy()
    invalid_mask[-1, 0, 0] = True

    with pytest.raises(
        ValueError, match="scaled_target_mask permits a horizon endpoint"
    ):
        evaluate_scores(
            replace(
                inputs,
                scaled_target_mask=invalid_mask,
                scaled_midrank_targets=UndecodableTarget(),
            ),
            window_name="F2",
        )


def test_report_exposes_primary_and_economic_coverage() -> None:
    report = evaluate_scores(_fixture(), window_name="F2").report

    assert report["primary_support"]["possible_date_count"] == 20
    assert report["primary_support"]["used_date_count"] == 20
    assert report["economics"]["coverage"]["possible_date_count"] == 25
    assert report["economics"]["coverage"]["finite_net_excess_date_count"] == 25
    assert report["economics"]["d5_only_diagnostic"]["horizon_sessions"] == 5
    assert len(report["economics"]["d5_only_diagnostic"]["daily_table"]) == 25


def test_report_persists_state_transactions_holding_ages_and_action_claims() -> None:
    inputs = _fixture()
    has_action = np.asarray(inputs.action_has_action).copy()
    cash = np.asarray(inputs.action_cash_per_prior_share).copy()
    payment = np.asarray(inputs.action_payment_session).copy()
    has_action[4, 0] = True
    cash[4, 0] = 1.25
    payment[4, 0] = 7

    report = evaluate_scores(
        replace(
            inputs,
            action_has_action=has_action,
            action_cash_per_prior_share=cash,
            action_payment_session=payment,
        ),
        window_name="F2",
    ).report
    audit = report["economics"]["headline_audit"]

    assert len(audit["daily_state"]) == len(inputs.dates)
    assert all("deployed_net_fraction_nav" in row for row in audit["daily_state"])
    assert all("reconciliation_error" in row for row in audit["daily_state"])
    assert audit["intended_orders"]
    assert audit["fills"]
    assert isinstance(audit["cancellations"], list)
    assert (
        audit["holding_age_distribution"]["population"]
        == "end_of_session_held_name_days"
    )
    assert audit["holding_age_distribution"]["holdings"]
    action_rows = audit["claims_and_action_attribution"]["rows"]
    assert len(action_rows) == 1
    assert action_rows[0]["cash_per_prior_share"] == 1.25
    assert action_rows[0]["claim_payment_date"] == inputs.dates[7].isoformat()
    assert "mean_deployed_net_fraction_nav" in report["economics"]["headline"]


def test_quality_strata_report_archive_and_conditional_validity_separately() -> None:
    inputs = _fixture()
    shape = inputs.active.shape
    present = np.zeros(shape, dtype=np.bool_)
    valid = np.zeros(shape, dtype=np.bool_)
    present[:, :40] = True
    valid[:, :20] = True
    age = np.broadcast_to(np.arange(shape[1], dtype=np.float64), shape).copy()
    survives = np.broadcast_to(np.arange(shape[1])[None, :] < 30, shape).copy()

    report = evaluate_scores(
        replace(
            inputs,
            history_age_sessions=age,
            source_archive_present={"options": present},
            source_feature_valid={"options": valid},
            eventual_survives_to_final_year=survives,
        ),
        window_name="F2",
    ).report
    diagnostics = report["quality_and_coverage_stratification"]
    source = diagnostics["source_coverage"][0]

    assert source["source"] == "options"
    assert source["archive_coverage_rate"] == pytest.approx(2.0 / 3.0)
    assert source["feature_valid_conditional_on_archive_presence"] == pytest.approx(0.5)
    assert {row["dimension"] for row in diagnostics["rows"]} >= {
        "calendar_year",
        "causal_liquidity_quartile",
        "history_age_sessions",
        "source_archive_availability",
        "eventual_survival_audit_label",
        "verified_terminal_status",
        "window_terminal_quote_status",
    }
    assert report["declared_subperiod_readouts"]["rows"]


def test_source_validity_without_archive_presence_is_rejected() -> None:
    inputs = _fixture()
    shape = inputs.active.shape
    with pytest.raises(ValueError, match="without archive presence"):
        evaluate_scores(
            replace(
                inputs,
                source_archive_present={"options": np.zeros(shape, dtype=np.bool_)},
                source_feature_valid={"options": np.ones(shape, dtype=np.bool_)},
            ),
            window_name="F2",
        )


def test_evaluator_rejects_nonretrospective_action_alignment() -> None:
    with pytest.raises(ValueError, match="retrospective action alignment"):
        evaluate_scores(
            replace(_fixture(), action_alignment="decision_known"),
            window_name="F2",
        )
