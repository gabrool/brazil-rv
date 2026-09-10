from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl

from brazil_rv.v2.round5_exposures import (
    exposure_panel,
    market_shocks,
    sector_relative_panel,
    sector_shrink,
    shock_axes,
)
from brazil_rv.v2.round5_market import OBSERVATION_SCHEMA


def _source():
    days = [date(2024, 1, 2) + timedelta(days=i) for i in range(10)]
    rows = [
        {
            "series": "ptax_brl_per_usd",
            "reference_date": day,
            "available_at": datetime.combine(day, datetime.min.time(), UTC)
            + timedelta(hours=16),
            "value": 5 + i * 0.1,
            "source_file": "fixture",
        }
        for i, day in enumerate(days)
    ]
    returns = pl.DataFrame(
        schema={
            "series": pl.String,
            "reference_date": pl.Date,
            "previous_date": pl.Date,
            "available_at": pl.Datetime("us", "UTC"),
            "log_return": pl.Float64,
        }
    )
    return days, rows, returns


def test_market_mutation_enters_first_decision_without_extra_lag():
    days, rows, returns = _source()
    # 18:46 UTC is after the 18:45 UTC decision; reference stays day six.
    rows[6]["available_at"] = rows[6]["available_at"].replace(hour=18, minute=46)
    original = shock_axes(
        market_shocks(pl.DataFrame(rows, schema=OBSERVATION_SCHEMA), returns), days
    )["fx"]
    rows[6]["value"] *= 1.1
    changed = shock_axes(
        market_shocks(pl.DataFrame(rows, schema=OBSERVATION_SCHEMA), returns), days
    )["fx"]
    np.testing.assert_array_equal(original[0][:7], changed[0][:7])
    assert original[0][7, 0] != changed[0][7, 0]
    # A value that prints before the decision is admitted on its own date.
    rows[6]["available_at"] = rows[6]["available_at"].replace(hour=16, minute=0)
    same_day = shock_axes(
        market_shocks(pl.DataFrame(rows, schema=OBSERVATION_SCHEMA), returns), days
    )["fx"]
    assert same_day[0][6, 0] == np.log(rows[6]["value"] / rows[5]["value"])
    assert same_day[1][6, 0] == 0


def test_holiday_preserves_known_shock_and_age_not_a_fake_zero_print():
    days, rows, returns = _source()
    del rows[7:]
    current, age, historical, _ = shock_axes(
        market_shocks(pl.DataFrame(rows, schema=OBSERVATION_SCHEMA), returns), days
    )["fx"]
    np.testing.assert_array_equal(current[6], current[9])
    assert age[9, 0] == 3 and np.isnan(historical[9])


def test_futures_five_session_change_requires_linked_contract_returns():
    days, _, empty = _source()
    rows = [
        {
            "series": "dce_iron",
            "reference_date": days[i],
            "previous_date": days[i - 1],
            "available_at": datetime.combine(days[i], datetime.min.time(), UTC),
            "log_return": 0.01,
        }
        for i in range(1, 8)
    ]
    del rows[3]
    result = market_shocks(
        pl.DataFrame(schema=OBSERVATION_SCHEMA), pl.DataFrame(rows, schema=empty.schema)
    )
    assert result["shock_5"].null_count() == result.height
    assert all(value == 0.01 for value in result["shock_1"])


def test_ols_future_mutation_and_publication_clock():
    rng = np.random.default_rng(81)
    factor = rng.normal(0, 0.01, 155)
    y = factor[:, None] * np.array([0.5, 1, 1.5]) + rng.normal(0, 0.002, (155, 3))
    mask = np.ones_like(y, bool)
    sectors = np.full(y.shape, "sector", object)
    issuers = np.tile(["A", "B", "C"], (155, 1))
    published = np.arange(155) + 1
    original = exposure_panel(y, mask, factor, published, sectors, issuers)
    future_y, future_x = y.copy(), factor.copy()
    future_y[120:] += 10
    future_x[120:] += 10
    changed = exposure_panel(future_y, mask, future_x, published, sectors, issuers)
    for before, after in zip(original, changed):
        np.testing.assert_array_equal(before[:121], after[:121])
    assert not np.array_equal(original[0][121], changed[0][121])
    delayed = published.copy()
    delayed[100] = 130
    x2 = factor.copy()
    x2[100] = 99
    a = exposure_panel(y, mask, factor, delayed, sectors, issuers)
    b = exposure_panel(y, mask, x2, delayed, sectors, issuers)
    np.testing.assert_array_equal(a[0][:130], b[0][:130])
    assert not np.array_equal(a[0][130], b[0][130])


def test_sector_prior_counts_issuers_once_and_small_sector_keeps_ols():
    beta = np.array([5.0, 5.0, 1.0, 1.0, 1.0])
    variance = np.ones(5)
    sectors = np.full(5, "s", object)
    issuers = np.array(["A", "A", "B", "C", "D"])
    shrunk, age = sector_shrink(
        beta, variance, np.ones(5, bool), sectors, issuers, np.array([1, 1, 1, 2, 3])
    )
    assert shrunk[0] == shrunk[1] == 1
    assert age[0] == 3
    small, _ = sector_shrink(
        beta[:3], variance[:3], np.ones(3, bool), sectors[:3], issuers[:3], np.ones(3)
    )
    np.testing.assert_array_equal(small, beta[:3])


def test_sector_relative_excludes_other_share_classes():
    values = np.array(
        [[[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]]
    )
    mask = np.ones_like(values, bool)
    sectors = np.full((1, 4), "s", object)
    issuers = np.array([["A", "A", "B", "C"]])
    result, known = sector_relative_panel(
        values, mask, sectors, issuers, np.ones((1, 4), bool)
    )
    np.testing.assert_array_equal(result[0, 0], [7.0, 7.0, 3.0])
    assert known.all()
