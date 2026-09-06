from __future__ import annotations

from datetime import date, time

import polars as pl
import pytest

from brazil_rv.v2.schedule_reconstruction import reconstruct_schedule


def _regimes() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "start_date": [date(2009, 1, 1)],
            "end_date": [date(2026, 12, 31)],
            "continuous_open": ["10:00:00"],
            "continuous_close": ["16:55:00"],
            "auction_close": ["17:00:00"],
            "component": ["dated_regime_17h"],
        }
    )


def test_reconstructed_schedule_requires_exact_exception_explanations() -> None:
    counts = pl.DataFrame(
        {
            "trade_date": [date(2021, 8, 13), date(2021, 8, 16), date(2021, 8, 18)],
            "len": [60, 60, 60],
        }
    )
    holidays = pl.DataFrame({"trade_date": pl.Series([], dtype=pl.Date)})
    bounds = pl.DataFrame(
        {
            "trade_date": [date(2021, 8, 16), date(2021, 8, 18)],
            "first_clock": [time(10), time(10)],
            "last_clock": [time(16, 54), time(17, 54)],
        }
    )
    with pytest.raises(ValueError, match="unexplained"):
        reconstruct_schedule(
            cotahist_counts=counts,
            holidays=holidays,
            explanations=pl.DataFrame(
                schema={
                    "trade_date": pl.Date,
                    "status": pl.String,
                    "reason": pl.String,
                    "evidence": pl.String,
                }
            ),
            regimes=_regimes(),
            m1_bounds=bounds,
        )

    explanations = pl.DataFrame(
        {
            "trade_date": [date(2021, 8, 17)],
            "status": ["weekday_nonholiday_without_archive_session"],
            "reason": ["exchange-specific closure"],
            "evidence": ["fixture"],
        }
    )
    schedule, exceptions = reconstruct_schedule(
        cotahist_counts=counts,
        holidays=holidays,
        explanations=explanations,
        regimes=_regimes(),
        m1_bounds=bounds,
    )
    assert exceptions.height == 1
    assert [row.trade_date for row in schedule] == counts["trade_date"].to_list()
    assert schedule[1].continuous_close == time(16, 55)
    assert schedule[2].continuous_close == time(17, 55)
    assert all(row.source.startswith("reconstructed_v1:") for row in schedule)


def test_reconstructed_schedule_rejects_stale_exception() -> None:
    counts = pl.DataFrame({"trade_date": [date(2020, 1, 2)], "len": [60]})
    holidays = pl.DataFrame({"trade_date": pl.Series([], dtype=pl.Date)})
    explanations = pl.DataFrame(
        {
            "trade_date": [date(2020, 1, 3)],
            "status": ["weekday_nonholiday_without_archive_session"],
            "reason": ["stale"],
            "evidence": ["fixture"],
        }
    )
    with pytest.raises(ValueError, match="stale"):
        reconstruct_schedule(
            cotahist_counts=counts,
            holidays=holidays,
            explanations=explanations,
            regimes=_regimes(),
            m1_bounds=pl.DataFrame(),
        )
