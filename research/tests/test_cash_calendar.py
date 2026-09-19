from copy import copy
import numpy as np
import pytest

from brazil_rv.v2.cash_calendar import apply_cash_calendar, close_interval_returns
from test_portfolio_policy import policy_fixture


def test_money_only_day_compounds_and_current_or_future_rate_cannot_leak():
    dates = np.array(
        ["2022-12-28", "2022-12-29", "2023-01-02", "2023-01-03"], dtype="datetime64[D]"
    )
    money = np.array(
        ["2022-12-29", "2022-12-30", "2023-01-02", "2023-01-03"], dtype="datetime64[D]"
    )
    rates = np.array([0.001, 0.002, 0.003, 0.004])
    actual = close_interval_returns(money, rates, dates)
    assert np.isnan(actual[:2]).all()
    assert actual[2:] == pytest.approx([1.001 * 1.002 - 1, 0.003])
    changed = rates.copy()
    changed[2:] = [0.05, 0.09]
    np.testing.assert_array_equal(
        actual[:3], close_interval_returns(money, changed, dates)[:3]
    )
    assert np.isnan(close_interval_returns(money[:2], rates[:2], dates)[2:]).all()


def test_explicit_cash_amendment_preserves_frozen_policy_coordinates():
    original = policy_fixture()
    data = copy(original)
    calendar = np.full(max(data.inputs.session_indices) + 1, np.datetime64("NaT", "D"))
    calendar[data.inputs.session_indices] = np.asarray(
        data.inputs.dates, dtype="datetime64[D]"
    )
    values = np.arange(len(calendar)) * 1e-6
    data.inputs = apply_cash_calendar(
        data.inputs, calendar, values, "cash source fixture"
    )
    assert data.static is original.static
    assert data.inputs.scores is original.inputs.scores
    assert data.inputs.active is original.inputs.active
    assert data.inputs.cdi_returns == pytest.approx(values[data.inputs.session_indices])
    assert data.inputs.source_artifact_hashes["cash_calendar"] == "cash source fixture"
    with pytest.raises(ValueError, match="already applied"):
        apply_cash_calendar(data.inputs, calendar, values, "duplicate")
