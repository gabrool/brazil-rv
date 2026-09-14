from copy import deepcopy

import numpy as np
import pytest

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.controller_synthetic import synthetic_data
from brazil_rv.v2.decision_program import (
    benchmark_excess_returns,
    calibration_payload,
    calibrations,
)
from brazil_rv.v2 import objective_readouts as program
from brazil_rv.v2.objective_readouts import cardinal_readout, choose_blend, rank_view
from brazil_rv.v2.portfolio_program import read


def test_causal_blend_selection_ignores_future_and_retains_population():
    data, _, _ = synthetic_data(days=160)
    other = deepcopy(data)
    other.ranks = np.roll(other.ranks, 3, axis=1)
    bounds = {"fit": np.arange(70), "selection": np.arange(80, 120)}
    first = choose_blend(data, other, bounds)
    for values in (data, other):
        values.ranks[120:] *= -100
        values.inputs.raw_close[120:] *= 20
        values.inputs.shareholder_simple_returns[120:] = 1e5
        values.inputs.cdi_returns[120:] = 0.99
    assert choose_blend(data, other, bounds) == first
    other.valid[12, 4] = False
    with pytest.raises(ValueError, match="intersect"):
        choose_blend(data, other, bounds)


def test_new_rank_view_has_no_old_forecasts_or_mutation_and_cardinal_masks():
    data, _, _ = synthetic_data(days=80)
    original = data.ranks.copy()
    rows = np.arange(20, 30)
    view = rank_view(data, rows, -data.ranks[rows], data.valid[rows])
    assert not view.valid[:20].any() and not view.valid[30:].any()
    np.testing.assert_array_equal(data.ranks, original)
    np.testing.assert_array_equal(view.ranks[rows], -original[rows])
    assert view.inputs is data.inputs
    y = np.array([[0.01, 0.02, np.nan], [0.03, np.nan, np.nan]])
    mask = np.isfinite(y)
    p = np.nan_to_num(y) / 2
    result = cardinal_readout(p, y, mask, np.zeros((*y.shape, 3)))
    assert result["calibration_slope"] == pytest.approx(2)
    assert result["realized_mean_bps"] == pytest.approx(225)
    assert result["predicted_mean_bps"] == pytest.approx(112.5)


def test_identical_new_forecasts_reproduce_matched_fold_accounts(tmp_path, monkeypatch):
    data, _, _ = synthetic_data(days=160)
    rows = np.arange(80, 160)
    mapping = calibrations(data, np.arange(70), benchmark_excess_returns(data))[
        "equal_rank"
    ]
    payload = calibration_payload(mapping)
    write_json_atomic(
        tmp_path / "phase3/mappings/F2.json",
        {
            "arms": {arm: payload for arm in program.ARMS},
            "blend": {"selected": {"te_weight": 0.5, "calibration": payload}},
        },
    )
    monkeypatch.setattr(program, "load_data", lambda *_: (data, "fixture"))
    monkeypatch.setattr(program, "windows", lambda *_: {"evaluation": rows})
    monkeypatch.setattr(program, "_git_identity", lambda: {"commit": "fixture"})

    def panels(root, source, arm, variant, fold, indices):
        ranks = {member: data.ranks[indices].copy() for member in program.MEMBERS}
        heads = (
            {member: np.zeros(data.valid[indices].shape) for member in program.MEMBERS}
            if variant == "economic"
            else {}
        )
        return ranks, heads, data.valid[indices].copy(), {"fixture": True}

    monkeypatch.setattr(program, "read_panel", panels)
    program.evaluate_fold(tmp_path, "F2")
    for arm in (*program.ARMS, "blend"):
        for member in program.MEMBERS:
            path = tmp_path / "phase3/books/F2" / arm
            a, b = (
                read(path / variant / member / "book.json")
                for variant in program.VARIANTS
            )
            assert a["burn_in_sessions"] == b["burn_in_sessions"] == 0
            assert a["daily"] == b["daily"]
            assert a["dates"] == [data.inputs.dates[i].isoformat() for i in rows]
