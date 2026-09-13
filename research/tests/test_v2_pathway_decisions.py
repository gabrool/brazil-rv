import numpy as np
import pytest

from brazil_rv.v2.contract import DEVELOPMENT_FOLDS
from brazil_rv.v2.pathway_statistics import holm_primary, infer
from brazil_rv.v2.round7 import SCREEN_FOLDS, SEEDS
from brazil_rv.v2.round7_pathway import IC, NET, ORDER, advance, finalize, select


def point(value):
    return {"estimate": value, "lower_95": value - 0.001, "upper_95": value + 0.001}


def screen(deltas):
    pairs = {}
    for c, value in zip(ORDER, deltas, strict=True):
        # Real paired_readouts only attaches folds to one orientation.
        pairs[f"B4_minus_{c}"] = {
            "pooled": {IC: point(-value)},
            "folds": {f: {IC: point(-value)} for f in SCREEN_FOLDS},
        }
        pairs[f"{c}_minus_B4"] = {
            "pooled": {IC: point(value)},
            "opposite_of": f"B4_minus_{c}",
        }
    return {"seeds": SEEDS, "folds": SCREEN_FOLDS, "paired": pairs}


def test_screen_requires_practical_gain_and_keeps_matched_controls():
    assert advance(screen([0.001, 0.0019, 0.001, 0.001]))["cells"] == []
    assert advance(screen([-0.001, 0.004, 0.001, 0.001]))["cells"] == ["GL", "GE"]
    assert advance(screen([-0.001, 0.003, 0.0015, 0.001]))["cells"] == list(ORDER)
    data = screen([-0.001, 0.004, 0.001, 0.001])
    for fold in SCREEN_FOLDS[:2]:
        data["paired"]["B4_minus_GE"]["folds"][fold][IC] = point(0.002)
    assert advance(data)["cells"] == []
    with pytest.raises(ValueError, match="four registered"):
        advance({**data, "folds": SCREEN_FOLDS[:2]})


def test_confirmation_and_fresh_seeds_do_not_shop_for_runner_up():
    panel = {
        "cells": ["GL", "GE", "B4"],
        "seeds": SEEDS,
        "folds": DEVELOPMENT_FOLDS,
        "readouts": {
            c: {"pooled": {IC: point(v), NET: point(1.0)}}
            for c, v in (("GL", 0.026), ("GE", 0.03), ("B4", 0.024))
        },
        "paired": {"GE_minus_B4": {"pooled": {IC: point(0.006), NET: point(0.2)}}},
    }
    omissions = {
        "candidate": "GE",
        "reference": "B4",
        "seeds": SEEDS,
        "all_omissions_positive": True,
    }
    decision = select(panel, omissions, "B4")
    assert decision["eligible"] and decision["extension_cells"] == ["GE", "GL"]
    rejected = select(panel, {**omissions, "all_omissions_positive": False}, "B4")
    assert finalize(rejected)["designation"] == "B4"
    six = {**panel, "seeds": (11, 29, 47, 61, 79, 97)}
    six_omissions = {**omissions, "seeds": six["seeds"]}
    assert finalize(decision, six, six_omissions)["designation"] == "GE"
    assert (
        finalize(decision, six, {**six_omissions, "all_omissions_positive": False})[
            "designation"
        ]
        == "B4"
    )
    panel["paired"]["GE_minus_B4"]["pooled"][IC]["lower_95"] = -0.001
    assert not select(panel, omissions, "B4")["eligible"]


def test_nw_preserves_fold_boundaries_and_holm_keeps_two_test_family():
    arrays = [np.linspace(-1, 1, 30), np.linspace(1, -1, 30) + 0.3]
    result = infer(arrays)
    mean = np.concatenate(arrays).mean()
    variance = sum(np.sum((a - mean) ** 2) for a in arrays)
    for a in arrays:
        z = a - mean
        variance += sum(2 * (1 - k / 11) * (z[k:] @ z[:-k]) for k in range(1, 11))
    assert result["nw_lag10_se"] == pytest.approx(np.sqrt(variance) / 60)
    assert result["estimate"] == pytest.approx(mean)
    rows = {
        "GE_minus_GL": {IC: {"nominal_two_sided_p": 0.02}},
        "TE_minus_TL": {IC: {"nominal_two_sided_p": 0.03}},
    }
    holm_primary(rows)
    assert [r[IC]["holm_two_primary_tests_p"] for r in rows.values()] == [0.04, 0.04]
    single = {"GE_minus_GL": {IC: {"nominal_two_sided_p": 0.02}}}
    holm_primary(single)
    assert single["GE_minus_GL"][IC]["holm_two_primary_tests_p"] == 0.04
