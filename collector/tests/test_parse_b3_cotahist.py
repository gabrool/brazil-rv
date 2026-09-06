from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import polars as pl
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from parse_b3_cotahist import (  # noqa: E402
    RECORD_LENGTH,
    collapse_security_days,
    parse_quote_line,
)


def _field(line: bytearray, start: int, stop: int, value: str) -> None:
    width = stop - start
    line[start:stop] = value.ljust(width)[:width].encode("latin-1")


def _quote_line(*, currency: str = "R$", quote_factor: int = 1) -> bytes:
    line = bytearray(b" " * RECORD_LENGTH)
    _field(line, 0, 2, "01")
    _field(line, 2, 10, "20240102")
    _field(line, 10, 12, "02")
    _field(line, 12, 24, "TEST3")
    _field(line, 24, 27, "010")
    _field(line, 27, 39, "TEST ISSUER")
    _field(line, 39, 49, "ON")
    _field(line, 52, 56, currency)
    for start, stop, value in (
        (56, 69, 1_000),
        (69, 82, 1_100),
        (82, 95, 900),
        (95, 108, 1_020),
        (108, 121, 1_050),
        (121, 134, 1_040),
        (134, 147, 1_060),
        (147, 152, 100),
        (152, 170, 1_000),
        (170, 188, 1_050_000),
        (210, 217, quote_factor),
        (242, 245, 1),
    ):
        _field(line, start, stop, str(value).zfill(stop - start))
    _field(line, 230, 242, "BRTESTACNOR1")
    return bytes(line)


def test_parser_requires_explicit_positive_quote_factor_and_supported_currency() -> None:
    parsed = parse_quote_line(_quote_line())
    assert parsed is not None
    assert parsed["trade_date"] == date(2024, 1, 2)
    assert parsed["close_brl"] == pytest.approx(10.5)
    assert parsed["currency"] == "R$"

    with pytest.raises(ValueError, match="invalid FATCOT"):
        parse_quote_line(_quote_line(quote_factor=0))
    with pytest.raises(ValueError, match="unsupported quote currency"):
        parse_quote_line(_quote_line(currency="USD"))


def test_daily_collapse_removes_only_byte_equivalent_economic_rows() -> None:
    parsed = parse_quote_line(_quote_line())
    assert parsed is not None
    frame = pl.DataFrame([parsed, parsed], infer_schema_length=None)
    collapsed, duplicate_count = collapse_security_days(frame)
    assert duplicate_count == 1
    assert collapsed.height == 1
    assert collapsed[0, "source_row_count"] == 1

    conflict = {**parsed, "close_brl": float(parsed["close_brl"]) + 0.01}
    with pytest.raises(ValueError, match="conflicting duplicate COTAHIST rows"):
        collapse_security_days(
            pl.DataFrame([parsed, conflict], infer_schema_length=None)
        )
