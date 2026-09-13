def test_three_traded_heads_pair_with_five_head_anchor_on_the_same_population():
    import numpy as np
    import pytest
    from dataclasses import replace
    from test_v2_evaluate import _fixture, TRIAGE_PROTOCOL
    from brazil_rv.v2.evaluate import evaluate_scores, paired_comparison

    inputs = _fixture()
    mask = inputs.score_mask.copy()
    mask[..., :2] = False
    c1 = evaluate_scores(replace(inputs, score_mask=mask), window_name="F2")
    s0 = evaluate_scores(inputs, window_name="F2")
    result = paired_comparison(c1, s0, protocol=TRIAGE_PROTOCOL)
    defined = [
        row["delta"]
        for row in result["daily_primary_ic_delta_table"]
        if row["delta"] is not None
    ]
    assert defined and np.max(np.abs(defined)) == pytest.approx(0.0)


def test_advancement_calendar_and_confirmation_rejection_cannot_be_bypassed():
    import pytest
    from brazil_rv.v2.round7 import SCREEN_FOLDS
    from brazil_rv.v2.round7_decisions import advance, finalize, IC, NET

    def point(value):
        return {"pooled": {IC: {"estimate": value}, NET: {"estimate": 2.0}}}

    screen = {
        "folds": list(SCREEN_FOLDS),
        "seeds": [11, 29, 47],
        "cells": ["A0", "A1", "B1", "B3", "B4", "B9"],
        "paired": {
            f"{c}_minus_A0": point(v)
            for c, v in [
                ("A1", -0.003),
                ("B1", 0.003),
                ("B3", 0.0035),
                ("B4", 0.005),
                ("B9", 0.006),
            ]
        },
    }
    assert advance(screen)["cells"] == ["A1", "B9", "B4"]
    with pytest.raises(ValueError, match="four-fold"):
        advance({**screen, "folds": ["F2", "F5", "F10", "F14"]})
    three = {"paired": {"A1_minus_A0": point(0.001)}, "readouts": {"A1": point(0.026)}}
    six = {
        "seeds": [11, 29, 47, 61, 79, 97],
        "paired": {"B4_minus_A0": point(0.004)},
        "readouts": {"B4": point(0.03)},
    }
    omissions = {"candidate": "B4", "all_omissions_positive": False}
    fallback = finalize(three, six, omissions, "B4")
    assert fallback["designation"] == "A1" and len(fallback["seeds"]) == 3
    assert (
        finalize(three, six, {**omissions, "all_omissions_positive": True}, "B4")[
            "designation"
        ]
        == "B4"
    )
    rejected_recipe = {
        **six,
        "paired": {"A1_minus_A0": point(-0.001)},
        "readouts": {"A1": point(0.024)},
    }
    assert (
        finalize(
            three,
            rejected_recipe,
            {"candidate": "A1", "all_omissions_positive": False},
            "A1",
        )["designation"]
        == "A0"
    )
