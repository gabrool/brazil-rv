from copy import deepcopy

import numpy as np
import pytest

from brazil_rv.v2.controller_synthetic import synthetic_data
from brazil_rv.v2.objective_readouts import cardinal_readout, choose_blend, rank_view


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
