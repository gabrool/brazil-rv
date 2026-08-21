from __future__ import annotations

import json
import zipfile
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from brazil_rv.preprocessing.options_activity import (
    FEATURES,
    OptionActivity,
    OptionRecord,
    StockDay,
    build_rows,
    build_sidecar,
    parse_option_line,
)

ISIN = "BRPETRACNPR6"
SECURITY_ID = f"ISIN:{ISIN}"


def _put(line: bytearray, start: int, end: int, value: str | int) -> None:
    width = end - start
    text = str(value)
    if isinstance(value, int):
        text = text.zfill(width)
    line[start:end] = text.ljust(width).encode("ascii")[:width]


def _option_line(
    trade_date: date,
    *,
    ticker: str = "UNRELATED123",
    isin: str = ISIN,
    market_type: int = 70,
    expiry: date | None = None,
    strike_cents: int = 3_000,
    trades: int = 10,
    quantity: int = 1_000,
    volume_cents: int = 100_000,
) -> bytes:
    line = bytearray(b" " * 245)
    _put(line, 0, 2, "01")
    _put(line, 2, 10, trade_date.strftime("%Y%m%d"))
    _put(line, 10, 12, "78")
    _put(line, 12, 24, ticker)
    _put(line, 24, 27, market_type)
    _put(line, 27, 39, "PETR")
    _put(line, 39, 49, "PN")
    _put(line, 147, 152, trades)
    _put(line, 152, 170, quantity)
    _put(line, 170, 188, volume_cents)
    _put(line, 188, 201, strike_cents)
    _put(line, 202, 210, (expiry or trade_date + timedelta(days=20)).strftime("%Y%m%d"))
    _put(line, 210, 217, 1)
    _put(line, 230, 242, isin)
    return bytes(line)


def _record(
    trade_date: date,
    *,
    is_put: bool,
    quantity: int,
    strike_brl: float = 30.0,
) -> OptionRecord:
    return OptionRecord(
        trade_date=trade_date,
        isin=ISIN,
        ticker="NO_PREFIX",
        is_put=is_put,
        expiry=trade_date + timedelta(days=20),
        strike_brl=strike_brl,
        trades=max(quantity // 10, 1),
        quantity=quantity,
        volume_brl=quantity * 2.0,
    )


def _activity(trade_date: date, index: int) -> OptionActivity:
    activity = OptionActivity()
    activity.add(_record(trade_date, is_put=False, quantity=100 + index), 30.0)
    activity.add(_record(trade_date, is_put=True, quantity=50 + index), 30.0)
    return activity


def test_parser_uses_market_type_and_exact_isin_not_ticker_prefix() -> None:
    trade_date = date(2023, 1, 2)
    call = parse_option_line(
        _option_line(trade_date, ticker="COMPLETELYX", market_type=70)
    )
    put = parse_option_line(
        _option_line(trade_date, ticker="OTHERTHING", market_type=80)
    )
    assert call is not None and not call.is_put and call.isin == ISIN
    assert put is not None and put.is_put and put.isin == ISIN
    assert parse_option_line(_option_line(trade_date, market_type=10)) is None


def test_rows_use_next_session_and_prior_only_rolling_state() -> None:
    start = date(2021, 8, 2)
    sessions = [start + timedelta(days=index) for index in range(27)]
    activities = {
        (source_date, SECURITY_ID): _activity(source_date, index)
        for index, source_date in enumerate(sessions[:-1])
    }
    stock_days = {
        (source_date, SECURITY_ID): StockDay(30.0, 1_000, 1_000_000, 30_000_000.0)
        for source_date in sessions[:-1]
    }
    rows, audit = build_rows(activities, stock_days, sessions)
    assert rows[0]["available_date"] == sessions[1]
    assert rows[0]["security_id"] == SECURITY_ID
    assert rows[0]["options_put_call_quantity_log_ratio_tanh_mask"]
    assert not rows[19]["options_quantity_log_surprise_20_scaled_mask"]
    assert rows[20]["options_quantity_log_surprise_20_scaled_mask"]
    assert not rows[4]["options_stock_quantity_log_ratio_change_5_tanh_mask"]
    assert rows[5]["options_stock_quantity_log_ratio_change_5_tanh_mask"]
    assert audit["output_rows"] == 26

    baseline = [{name: row[name] for name in FEATURES} for row in rows[:25]]
    activities[(sessions[25], SECURITY_ID)] = _activity(sessions[25], 1_000_000)
    mutated, _ = build_rows(activities, stock_days, sessions)
    assert [{name: row[name] for name in FEATURES} for row in mutated[:25]] == baseline


def _archive(path: Path, lines: list[bytes]) -> None:
    header = bytearray(b" " * 245)
    _put(header, 0, 2, "00")
    _put(header, 23, 31, "20211230")
    trailer = bytearray(b" " * 245)
    _put(trailer, 0, 2, "99")
    payload = b"\r\n".join([bytes(header), *lines, bytes(trailer)]) + b"\r\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("COTAHIST_A2021.TXT", payload)


def test_sidecar_writes_exact_identity_provenance_and_masks(tmp_path: Path) -> None:
    source_date = date(2021, 8, 16)
    archive = tmp_path / "COTAHIST_A2021.ZIP"
    _archive(
        archive,
        [
            _option_line(source_date, ticker="NO_SHARED_PREFIX", market_type=70),
            _option_line(source_date, ticker="ALSO_UNRELATED", market_type=80),
        ],
    )
    assignments = tmp_path / "assignments.parquet"
    pl.DataFrame(
        {
            "security_id": [SECURITY_ID],
            "isin": [ISIN],
            "latest_ticker": ["PETR4"],
            "first_overlap_date": ["2021-07-19"],
            "last_overlap_date": ["2026-07-17"],
        }
    ).write_parquet(assignments)
    calendar = tmp_path / "calendar" / "year=2021"
    calendar.mkdir(parents=True)
    pl.DataFrame(
        {
            "trade_date": [source_date, source_date + timedelta(days=1)],
            "security_id": [SECURITY_ID, SECURITY_ID],
            "close_brl": [30.0, 31.0],
            "trades": [1_000, 1_100],
            "quantity": [1_000_000, 1_100_000],
            "volume_brl": [30_000_000.0, 34_100_000.0],
        }
    ).write_parquet(calendar / "equities_daily_2021.parquet")
    output = tmp_path / "output"
    manifest = build_sidecar(
        [archive],
        calendar.parent,
        assignments,
        output,
        start=source_date,
        end=source_date,
    )
    frame = pl.read_parquet(output / "cotahist_options_activity.parquet")
    stored = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert frame.height == 1
    assert frame[0, "security_id"] == SECURITY_ID
    assert frame[0, "available_date"] == source_date + timedelta(days=1)
    assert stored["output_sha256"] == manifest["output_sha256"]
    assert set(stored["feature_valid_rows"]) == set(FEATURES)
    assert "ticker-prefix inference are never used" in stored["identity_rule"]
    with pytest.raises(FileExistsError):
        build_sidecar(
            [archive],
            calendar.parent,
            assignments,
            output,
            start=source_date,
            end=source_date,
        )
