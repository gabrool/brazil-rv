from __future__ import annotations

import json
import zipfile
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from brazil_rv.preprocessing.odd_lot_activity import (
    FEATURES,
    Activity,
    QuoteRecord,
    build_rows,
    build_sidecar,
    parse_quote_line,
)

ISIN = "BRPETRACNPR6"


def _put(line: bytearray, start: int, end: int, value: str | int) -> None:
    width = end - start
    text = str(value)
    if isinstance(value, int):
        text = text.zfill(width)
    line[start:end] = text.ljust(width).encode("ascii")[:width]


def _quote_line(
    trade_date: date,
    *,
    isin: str = ISIN,
    ticker: str = "PETR4",
    cod_bdi: str = "02",
    market_type: int = 10,
    spec: str = "PN",
    trades: int = 100,
    quantity: int = 1_000,
    volume_cents: int = 1_000_000,
    close_cents: int = 3_000,
) -> bytes:
    line = bytearray(b" " * 245)
    _put(line, 0, 2, "01")
    _put(line, 2, 10, trade_date.strftime("%Y%m%d"))
    _put(line, 10, 12, cod_bdi)
    _put(line, 12, 24, ticker)
    _put(line, 24, 27, market_type)
    _put(line, 39, 49, spec)
    _put(line, 52, 56, "R$")
    _put(line, 108, 121, close_cents)
    _put(line, 147, 152, trades)
    _put(line, 152, 170, quantity)
    _put(line, 170, 188, volume_cents)
    _put(line, 210, 217, 1)
    _put(line, 230, 242, isin)
    return bytes(line)


def _activity(
    trade_date: date,
    *,
    market: str,
    ticker: str,
    trades: int,
    volume_cents: int,
) -> Activity:
    value = Activity()
    value.add_record(
        QuoteRecord(
            trade_date=trade_date,
            isin=ISIN,
            market=market,
            ticker=ticker,
            trades=trades,
            quantity=trades * 10,
            volume_cents=volume_cents,
            close_brl=30.0,
        )
    )
    return value


def _archive(path: Path, lines: list[bytes]) -> None:
    header = bytearray(b" " * 245)
    _put(header, 0, 2, "00")
    _put(header, 23, 31, "20211230")
    trailer = bytearray(b" " * 245)
    _put(trailer, 0, 2, "99")
    payload = b"\r\n".join([bytes(header), *lines, bytes(trailer)]) + b"\r\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("COTAHIST_A2021.TXT", payload)


def test_parser_requires_exact_market_pair_equity_spec_and_isin() -> None:
    source_date = date(2021, 8, 16)
    regular = parse_quote_line(_quote_line(source_date))
    odd = parse_quote_line(
        _quote_line(
            source_date,
            ticker="PETR4F",
            cod_bdi="96",
            market_type=20,
            trades=4,
        )
    )
    assert regular is not None and regular.market == "regular"
    assert odd is not None and odd.market == "odd_lot"
    assert regular.isin == odd.isin == ISIN
    assert (
        parse_quote_line(_quote_line(source_date, cod_bdi="02", market_type=20)) is None
    )
    assert parse_quote_line(_quote_line(source_date, isin="")) is None
    assert parse_quote_line(_quote_line(source_date, spec="REC")) is None


def test_rows_use_next_session_and_keep_zero_distinct_from_missing() -> None:
    sessions = [date(2021, 8, 16) + timedelta(days=offset) for offset in range(4)]
    activity = {
        (sessions[0], ISIN, "regular"): _activity(
            sessions[0], market="regular", ticker="PETR4", trades=100, volume_cents=1000
        ),
        (sessions[1], ISIN, "regular"): _activity(
            sessions[1], market="regular", ticker="PETR4", trades=100, volume_cents=1000
        ),
        (sessions[1], ISIN, "odd_lot"): _activity(
            sessions[1], market="odd_lot", ticker="PETR4F", trades=10, volume_cents=100
        ),
        # An odd-lot-only row is missing regular activity, not an observed feature row.
        (sessions[2], ISIN, "odd_lot"): _activity(
            sessions[2], market="odd_lot", ticker="PETR4F", trades=5, volume_cents=50
        ),
    }
    rows, audit = build_rows(activity, sessions)
    assert len(rows) == 2
    assert rows[0]["source_trade_date"] == sessions[0]
    assert rows[0]["available_date"] == sessions[1]
    assert rows[0]["odd_lot_observed"] is False
    assert rows[0]["odd_volume_share_asin_sqrt"] == 0.0
    assert rows[0]["odd_volume_share_asin_sqrt_mask"] is True
    assert rows[1]["available_date"] == sessions[2]
    assert rows[1]["odd_lot_observed"] is True
    assert audit["orphan_odd_lot_security_days"] == 1


def test_rolling_features_use_prior_observations_only() -> None:
    start = date(2021, 8, 2)
    sessions = [start + timedelta(days=offset) for offset in range(24)]
    activity: dict[tuple[date, str, str], Activity] = {}
    for index, source_date in enumerate(sessions[:-1]):
        activity[(source_date, ISIN, "regular")] = _activity(
            source_date,
            market="regular",
            ticker="PETR4",
            trades=100,
            volume_cents=10_000,
        )
        activity[(source_date, ISIN, "odd_lot")] = _activity(
            source_date,
            market="odd_lot",
            ticker="PETR4F",
            trades=index + 1,
            volume_cents=(index + 1) * 100,
        )
    rows, _ = build_rows(activity, sessions)
    assert rows[4]["odd_volume_share_change_5_mask"] is False
    assert rows[5]["odd_volume_share_change_5_mask"] is True
    assert rows[19]["odd_volume_share_surprise_20_mask"] is False
    assert rows[20]["odd_volume_share_surprise_20_mask"] is True

    earlier = [{name: row[name] for name in FEATURES} for row in rows[:20]]
    activity[(sessions[21], ISIN, "odd_lot")] = _activity(
        sessions[21],
        market="odd_lot",
        ticker="PETR4F",
        trades=99_999,
        volume_cents=99_999_999,
    )
    mutated, _ = build_rows(activity, sessions)
    assert [{name: row[name] for name in FEATURES} for row in mutated[:20]] == earlier


def test_sidecar_writes_provenance_and_feature_masks(tmp_path: Path) -> None:
    source_date = date(2021, 8, 16)
    source = tmp_path / "COTAHIST_A2021.ZIP"
    _archive(
        source,
        [
            _quote_line(source_date),
            _quote_line(
                source_date,
                ticker="PETR4F",
                cod_bdi="96",
                market_type=20,
                trades=5,
                volume_cents=10_000,
            ),
        ],
    )
    calendar = tmp_path / "calendar" / "year=2021"
    calendar.mkdir(parents=True)
    pl.DataFrame(
        {"trade_date": [source_date, source_date + timedelta(days=1)]}
    ).write_parquet(calendar / "equities_daily_2021.parquet")
    output = tmp_path / "output"
    manifest = build_sidecar([source], calendar.parent, output)
    frame = pl.read_parquet(output / "odd_lot_activity.parquet")
    stored = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert frame.height == 1
    assert frame[0, "security_id"] == f"ISIN:{ISIN}"
    assert frame[0, "available_date"] == source_date + timedelta(days=1)
    assert stored["output_sha256"] == manifest["output_sha256"]
    assert set(stored["feature_valid_rows"]) == set(FEATURES)
    with pytest.raises(FileExistsError):
        build_sidecar([source], calendar.parent, output)
