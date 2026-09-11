import numpy as np

from brazil_rv.v2.round6_diagnostics import field_readout


def test_cross_sectional_ic_does_not_pool_time_shifts_or_invalid_payloads():
    values = np.array([np.arange(25), np.arange(25) + 100.0])
    targets = np.array([np.arange(25) + 100.0, np.arange(25)])
    valid = np.ones_like(values, bool)
    valid[:, -1] = False
    values[:, -1], targets[:, -1] = 1e20, -1e20
    dates = np.array(["2020-12-30", "2021-01-04"], dtype="datetime64[D]")
    result = field_readout(values, targets, valid, dates)
    assert result["pooled"]["mean_daily_cross_sectional_ic"] == 1.0
    assert result["pooled"]["pooled_name_day_spearman_secondary"] < 0
    assert result["pooled"]["supported_name_days"] == 48
    assert result["by_year"]["2020"]["defined_daily_ics"] == 1


def test_constant_sparse_and_missing_dates_are_undefined_not_zero():
    values = np.tile(np.arange(25, dtype=float), (3, 1))
    targets = values.copy()
    values[0] = 0
    valid = np.ones_like(values, bool)
    valid[1, 19:] = False
    valid[2] = False
    dates = np.arange("2020-01-01", "2020-01-04", dtype="datetime64[D]")
    result = field_readout(values, targets, valid, dates)
    assert result["daily_ic"] == [None, None, None]
    assert result["pooled"]["mean_daily_cross_sectional_ic"] is None
    assert result["pooled"]["constant_rank_dates_with_minimum_support"] == 1
