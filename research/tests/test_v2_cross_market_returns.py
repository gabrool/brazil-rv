from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from brazil_rv.v2.cross_market_returns import (
    admit_brent,
    comparison_return_mask,
    relative_return_panel,
)


def test_inferred_crash_adjustment_cannot_become_an_adr_gap():
    args = list(inputs())
    actions = np.zeros_like(args[7])
    actions[1, 0] = True
    args[6][1, 0] = 0.46  # Artificial wealth return after an inferred unit change.
    args[7] = comparison_return_mask(args[7], actions)
    result = relative_return_panel(*args)
    assert not result[1][2, 0]
    assert result[1][3, 0]  # No additional lag or permanent history exclusion.


def inputs():
    days = [date(2024, 3, 4) + timedelta(days=i) for i in range(5)]

    def clock(d):
        return datetime(d.year, d.month, d.day, 20, tzinfo=UTC)

    us = pl.DataFrame(
        [
            {
                "series": s,
                "reference_date": d,
                "previous_date": days[i - 1],
                "available_at": clock(d),
                "log_return": 0.02,
            }
            for s in ("us_ADR", "us_EWZ")
            for i, d in enumerate(days)
            if i
        ]
    )
    fx = pl.DataFrame(
        [
            {
                "series": "ptax_brl_per_usd",
                "reference_date": d,
                "value": 5.0 * 1.01**i,
                "available_at": clock(d),
            }
            for i, d in enumerate(days)
        ]
    )
    cash = pl.DataFrame(
        {"source_trade_date": days, "ticker": ["AAAA3"] * 5, "isin": ["A"] * 5}
    )
    pairs = [
        {"symbol": "ADR", "ticker": "AAAA3", "start": "2010-01-01", "end": "2024-12-30"}
    ]
    return (
        us,
        fx,
        cash,
        pairs,
        days,
        ("A",),
        np.full((5, 1), 0.015),
        np.ones((5, 1), bool),
        np.full(5, 0.01),
    )


def test_currency_sign_same_endpoints_and_decision_lag():
    result = relative_return_panel(*inputs())
    assert not result[1][:2].any()
    assert result[0][2, 0] == pytest.approx(0.02 + np.log(1.01) - 0.015)
    assert result[2][2] == pytest.approx(0.02 + np.log(1.01) - 0.01)
    # EWZ must be converted to BRL too; subtracting a BRL ETF from USD leaks FX
    # exposure into a purported relative-market shock.


def test_identity_holidays_clocks_and_future_mutations():
    args = list(inputs())
    baseline = relative_return_panel(*args)
    args[0] = args[0].with_columns(
        pl.when(pl.col("reference_date") >= args[4][3])
        .then(99)
        .otherwise(pl.col("log_return"))
        .alias("log_return")
    )
    changed = relative_return_panel(*args)
    for a, b in zip(baseline, changed):
        np.testing.assert_array_equal(a[:4], b[:4])
    args = list(inputs())
    args[2] = args[2].with_columns(
        pl.when(pl.col("source_trade_date") >= args[4][1])
        .then(pl.lit("B"))
        .otherwise(pl.col("isin"))
        .alias("isin")
    )
    assert not relative_return_panel(*args)[1][2, 0]
    args = list(inputs())
    args[0] = args[0].with_columns(pl.lit(args[4][0]).alias("previous_date"))
    assert not relative_return_panel(*args)[1][3, 0]
    args = list(inputs())
    args[3][0]["start"] = args[4][1].isoformat()
    assert not relative_return_panel(*args)[1][2, 0]
    args = list(inputs())
    args[1] = args[1].with_columns(
        pl.lit(datetime(2024, 3, 20, tzinfo=UTC)).alias("available_at")
    )
    assert not relative_return_panel(*args)[1].any()


def test_brent_next_actual_session_and_other_clocks_exact():
    levels = pl.DataFrame(
        {
            "series": ["brent_spot", "ptax_brl_per_usd"],
            "reference_date": [date(2024, 3, 8)] * 2,
            "available_at": [None, datetime(2024, 3, 8, 16, tzinfo=UTC)],
        }
    )
    result = admit_brent(levels, [date(2024, 3, 8), date(2024, 3, 11)])
    assert result["available_at"][0] == datetime(2024, 3, 11, 18, 45, tzinfo=UTC)
    assert result["available_at"][1] == levels["available_at"][1]
