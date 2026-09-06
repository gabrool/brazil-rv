from __future__ import annotations

import sys
import zipfile
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
    parse_trailer,
    parse_year,
)


def _field(line: bytearray, start: int, stop: int, value: str) -> None:
    width = stop - start
    line[start:stop] = value.ljust(width)[:width].encode("latin-1")


def _quote_line(
    *,
    currency: str = "R$",
    quote_factor: int = 1,
    quoted_price_scale: int = 1,
) -> bytes:
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
        (56, 69, 1_000 * quoted_price_scale),
        (69, 82, 1_100 * quoted_price_scale),
        (82, 95, 900 * quoted_price_scale),
        (95, 108, 1_020 * quoted_price_scale),
        (108, 121, 1_050 * quoted_price_scale),
        (121, 134, 1_040 * quoted_price_scale),
        (134, 147, 1_060 * quoted_price_scale),
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


def test_equivalent_quote_units_have_identical_per_share_prices() -> None:
    unit_quote = parse_quote_line(_quote_line(quote_factor=1))
    ten_unit_quote = parse_quote_line(
        _quote_line(quote_factor=10, quoted_price_scale=10)
    )
    assert unit_quote is not None and ten_unit_quote is not None
    for field in (
        "open_brl",
        "high_brl",
        "low_brl",
        "average_brl",
        "close_brl",
        "best_bid_brl",
        "best_ask_brl",
    ):
        assert ten_unit_quote[field] == pytest.approx(unit_quote[field])


def test_malformed_record_length_prevents_archive_promotion(tmp_path: Path) -> None:
    trailer = bytearray(b" " * RECORD_LENGTH)
    _field(trailer, 0, 2, "99")
    _field(trailer, 31, 42, "123")
    assert parse_trailer(bytes(trailer)) == 123

    header = bytearray(b" " * RECORD_LENGTH)
    _field(header, 0, 2, "00")
    _field(header, 23, 31, "20240102")
    malformed = _quote_line()[:-1]
    # A structurally plausible count cannot rescue a malformed physical row.
    _field(trailer, 31, 42, "3")
    archive_path = tmp_path / "COTAHIST_A2024.ZIP"
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        archive.writestr(
            "COTAHIST_A2024.TXT",
            b"\r\n".join((bytes(header), malformed, bytes(trailer))) + b"\r\n",
        )
    output = tmp_path / "parsed"
    output.mkdir()
    audit = parse_year(archive_path, 2024, output)
    assert not audit.record_count_valid
    assert audit.malformed_length_records == 1
    assert "structural/trailer audit failed" in audit.error
    assert not (output / "year=2024").exists()


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
