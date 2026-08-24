from __future__ import annotations

from datetime import date

import pytest

from brazil_rv.preprocessing.options_full import (
    _forward_option_price,
    _option_record,
    implied_volatility,
)


def _put(line: bytearray, start: int, end: int, value: str) -> None:
    encoded = value.encode("ascii")
    assert len(encoded) <= end - start
    line[start:end] = encoded.rjust(end - start, b"0")


def test_cotahist_option_parser_uses_close_then_average_and_quote_factor() -> None:
    line = bytearray(b" " * 245)
    line[:2] = b"01"
    line[2:10] = b"20240628"
    line[12:24] = b"RDORG246    "
    _put(line, 24, 27, "70")
    _put(line, 95, 108, "1234")
    _put(line, 108, 121, "1250")
    _put(line, 147, 152, "4")
    _put(line, 152, 170, "25")
    _put(line, 188, 201, "2464")
    line[202:210] = b"20240719"
    _put(line, 210, 217, "1")
    line[230:242] = b"BRRDORACNOR8"

    trade_date, record = _option_record(bytes(line))  # type: ignore[misc]

    assert trade_date == date(2024, 6, 28)
    assert record.ticker == "RDORG246"
    assert record.expiry == date(2024, 7, 19)
    assert record.strike == pytest.approx(24.64)
    assert record.average == pytest.approx(12.34)
    assert record.close == pytest.approx(12.50)
    assert record.premium == pytest.approx(12.50)
    assert record.quantity == 25
    assert record.trades == 4


@pytest.mark.parametrize("is_put", [False, True])
def test_implied_volatility_recovers_discounted_forward_price(is_put: bool) -> None:
    expected = 0.37
    premium = _forward_option_price(25.0, 24.0, 0.11, 30 / 365, expected, is_put)

    actual = implied_volatility(
        premium=premium,
        forward=25.0,
        strike=24.0,
        rate=0.11,
        years=30 / 365,
        is_put=is_put,
    )

    assert actual == pytest.approx(expected, abs=1e-10)


def test_implied_volatility_rejects_price_above_frozen_upper_bound() -> None:
    assert (
        implied_volatility(
            premium=100.0,
            forward=25.0,
            strike=24.0,
            rate=0.11,
            years=30 / 365,
            is_put=False,
        )
        is None
    )
