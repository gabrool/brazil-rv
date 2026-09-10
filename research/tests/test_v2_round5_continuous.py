from dataclasses import replace

import numpy as np
import pytest

from brazil_rv.v2.execution_policy import ExecutionPolicy
from brazil_rv.v2.round5_continuous import (
    boundary_evidence,
    evaluate_continuous,
    stitch_scores,
)
from test_v2_evaluate import _fixture
from test_v2_stateful_ledger import _action_terms, _run


def _spans(boundary, length):
    return [
        {"fold": "F1", "start_offset": 0, "end_offset": boundary - 1},
        {"fold": "F2", "start_offset": boundary, "end_offset": length - 1},
    ]


def test_stitch_preserves_missing_tail_scores_and_rejects_missing_calendar():
    inputs = _fixture()
    mask = inputs.score_mask.copy()
    mask[11] = False
    panels = [
        ("F1", inputs.session_indices[:12], inputs.scores[:12], mask[:12]),
        ("F2", inputs.session_indices[12:], inputs.scores[12:], mask[12:]),
    ]
    indices, scores, stitched_mask, spans = stitch_scores(panels)
    np.testing.assert_array_equal(indices, inputs.session_indices)
    np.testing.assert_array_equal(scores, inputs.scores)
    np.testing.assert_array_equal(stitched_mask, mask)
    assert spans[0]["missing_entire_score_dates"] == [111]
    for drift in (-1, 1):
        broken = [panels[0], ("F2", panels[1][1] + drift, *panels[1][2:])]
        with pytest.raises(ValueError, match="adjacent, nonoverlapping"):
            stitch_scores(broken)


def test_switch_preserves_prior_book_under_new_model_score_mutation():
    inputs = replace(
        _fixture(),
        execution_policy=ExecutionPolicy(
            theta=0.5, horizons=(3, 5, 10), buffer_per_quintile=9
        ),
    )
    spans = _spans(12, len(inputs.dates))
    _, original, evidence = evaluate_continuous(inputs, spans)
    scores = inputs.scores.copy()
    scores[12:] *= -1
    _, changed, _ = evaluate_continuous(replace(inputs, scores=scores), spans)
    for field in (
        "nav",
        "signed_shares",
        "free_cash",
        "restricted_cash",
        "receivables",
        "payables",
        "pending_entry_count",
        "pending_exit_count",
    ):
        np.testing.assert_array_equal(
            getattr(original, field)[:12], getattr(changed, field)[:12]
        )
    assert len(evidence) == 1
    assert evidence[0]["nav_continuity_exact"]
    assert not original.exit_instructions_terminal[:-1].any()
    assert original.exit_instructions_terminal[-1] > 0


def test_pending_entry_keeps_same_order_across_model_boundary():
    close = np.full((5, 3), 100.0)
    scores = np.broadcast_to([3.0, 0.0, -3.0], close.shape).copy()
    fills = np.ones_like(close)
    fills[1, 0] = 0.5
    result = _run(close, scores, fill_fraction=fills)
    evidence = boundary_evidence(result, _spans(2, 5))[0]
    long_entry = next(
        order
        for order in result.intended_orders
        if order.purpose == "entry" and order.side == "buy"
    )
    assert evidence["prior_pending_entry_count"] == 1
    assert long_entry.order_id in evidence["pre_switch_orders_filled_after_switch"]
    assert [
        f.fill_session for f in result.fills if f.order_id == long_entry.order_id
    ] == [1, 2]


def test_unpaid_corporate_claims_cross_boundary_until_actual_payment():
    close = np.full((7, 2), 100.0)
    close[2:] = 45.0
    scores = np.broadcast_to([1.0, -1.0], close.shape).copy()
    payment = np.full(close.shape, -1, dtype=np.int64)
    payment[2] = 4
    result = _run(
        close,
        scores,
        actions=_action_terms(close.shape, day=2, q=2, d=10),
        payment_session=payment,
    )
    evidence = boundary_evidence(result, _spans(3, 7))[0]
    for field in ("receivables", "payables"):
        assert evidence["prior_close_state_carried"][field] == pytest.approx(0.1)
        assert evidence["next_close_state"][field] == pytest.approx(0.1)
        assert getattr(result, field)[4] == 0


def test_boundary_audit_rejects_an_internal_terminal_reset():
    result = _run(np.full((5, 2), 100.0), np.tile([1.0, -1.0], (5, 1)))
    result.exit_instructions_terminal[1] = 1
    with pytest.raises(ValueError, match="internal date"):
        boundary_evidence(result, _spans(3, 5))
