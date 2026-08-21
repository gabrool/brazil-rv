from __future__ import annotations

import math
from dataclasses import replace
from datetime import date, timedelta

import polars as pl
import pytest

from brazil_rv.preprocessing.adr_overnight import (
    PAIRS,
    Bar,
    _feature_series,
    _last_completed_us_session,
    _resolve_pair_identities,
    build_daily_frame,
)


def _bars(start: date, count: int, *, slope: float = 0.01) -> list[Bar]:
    return [
        Bar(
            start + timedelta(days=index),
            math.exp(slope * index),
            math.exp(slope * index),
            1_000,
        )
        for index in range(count)
    ]


def test_future_price_mutation_cannot_change_prior_features() -> None:
    start = date(2024, 1, 1)
    adr = _bars(start, 100, slope=0.012)
    ewz = _bars(start, 100, slope=0.004)
    parent = _feature_series(adr, ewz)
    mutation_date = adr[80].session_date
    changed_bars = list(adr)
    changed_bars[80] = replace(adr[80], adjusted_close=adr[80].adjusted_close * 2)
    changed = _feature_series(changed_bars, ewz)
    for field in ("return_1d", "return_5d", "residual", "surprise"):
        parent_values = getattr(parent, field)
        changed_values = getattr(changed, field)
        assert {
            key: value for key, value in parent_values.items() if key < mutation_date
        } == {
            key: value for key, value in changed_values.items() if key < mutation_date
        }


def test_last_completed_new_york_close_is_dst_aware_and_strict() -> None:
    spring_sessions = [date(2024, 3, 8), date(2024, 3, 11)]
    assert _last_completed_us_session(date(2024, 3, 11), spring_sessions) == date(
        2024, 3, 8
    )
    fall_sessions = [date(2024, 11, 1), date(2024, 11, 4)]
    assert _last_completed_us_session(date(2024, 11, 4), fall_sessions) == date(
        2024, 11, 1
    )


def _identity_frame(days: list[date]) -> tuple[pl.DataFrame, set[str]]:
    rows = []
    accepted = set()
    for pair in PAIRS:
        security_id = f"ISIN:{pair.local_ticker}"
        accepted.add(security_id)
        for current_date in days:
            rows.append(
                {
                    "trade_date": current_date,
                    "ticker": pair.local_ticker,
                    "security_id": security_id,
                }
            )
    return pl.DataFrame(rows).with_columns(pl.col("trade_date").cast(pl.Date)), accepted


def test_pair_identity_must_be_one_accepted_permanent_security() -> None:
    days = [date(2024, 1, 2)]
    frame, accepted = _identity_frame(days)
    identities = _resolve_pair_identities(frame, accepted, days[0], days[0])
    assert identities["ABEV3"] == "ISIN:ABEV3"

    corrupt = pl.concat(
        [
            frame,
            pl.DataFrame(
                {
                    "trade_date": days,
                    "ticker": ["ABEV3"],
                    "security_id": ["ISIN:WRONG"],
                }
            ).with_columns(pl.col("trade_date").cast(pl.Date)),
        ]
    )
    with pytest.raises(ValueError, match="one accepted permanent identity"):
        _resolve_pair_identities(corrupt, accepted | {"ISIN:WRONG"}, days[0], days[0])


def test_missing_adr_endpoint_is_masked_and_exactly_zero() -> None:
    source_start = date(2024, 1, 1)
    bars = {"EWZ": _bars(source_start, 40, slope=0.004)}
    bars.update(
        {pair.adr_symbol: _bars(source_start, 40, slope=0.01) for pair in PAIRS}
    )
    missing_source_date = source_start + timedelta(days=34)
    missing_index = 34
    abev = list(bars["ABEV"])
    abev[missing_index] = replace(abev[missing_index], volume=0)
    bars["ABEV"] = abev

    b3_dates = [source_start + timedelta(days=index) for index in range(35, 38)]
    identities, accepted = _identity_frame(b3_dates)
    frame, _ = build_daily_frame(
        bars,
        identities,
        accepted,
        available_start=b3_dates[0],
        available_end=b3_dates[-1],
    )
    row = frame.filter(
        (pl.col("available_date") == b3_dates[0])
        & (pl.col("security_id") == "ISIN:ABEV3")
    )
    assert row.get_column("source_session_date").item() == missing_source_date
    assert not row.get_column("adr_return_1d_mask").item()
    assert row.get_column("adr_return_1d").item() == 0.0
    assert not row.get_column("adr_minus_ewz_1d_mask").item()
    assert row.get_column("adr_minus_ewz_1d").item() == 0.0
