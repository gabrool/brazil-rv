from __future__ import annotations

import json
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from brazil_rv.v2.artifacts import sha256_file
from brazil_rv.v2.bova11 import build_bova11_series, load_bova11_series


def _put(line: bytearray, start: int, end: int, value: str | int) -> None:
    width = end - start
    text = str(value)
    if isinstance(value, int):
        text = text.zfill(width)
    line[start:end] = text.ljust(width).encode("ascii")[:width]


def _quote_line(
    trade_date: date,
    *,
    isin: str = "BRBOVACTF003",
    ticker: str = "BOVA11",
    cod_bdi: str = "14",
    market_type: int = 10,
    spec: str = "CI",
    close_cents: int = 10_000,
) -> bytes:
    line = bytearray(b" " * 245)
    _put(line, 0, 2, "01")
    _put(line, 2, 10, trade_date.strftime("%Y%m%d"))
    _put(line, 10, 12, cod_bdi)
    _put(line, 12, 24, ticker)
    _put(line, 24, 27, market_type)
    _put(line, 39, 49, spec)
    _put(line, 108, 121, close_cents)
    _put(line, 210, 217, 1)
    _put(line, 230, 242, isin)
    return bytes(line)


def _archive(path: Path, year: int, lines: list[bytes]) -> None:
    header = bytearray(b" " * 245)
    _put(header, 0, 2, "00")
    trailer = bytearray(b" " * 245)
    _put(trailer, 0, 2, "99")
    payload = b"\r\n".join([bytes(header), *lines, bytes(trailer)]) + b"\r\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"COTAHIST_A{year}.TXT", payload)


def test_bova11_builder_filters_exact_contract_and_loader_hash_verifies(
    tmp_path: Path,
) -> None:
    first = date(2024, 1, 2)
    second = date(2024, 1, 3)
    source = tmp_path / "COTAHIST_A2024.ZIP"
    _archive(
        source,
        2024,
        [
            _quote_line(first, close_cents=10_000),
            _quote_line(second, cod_bdi="02", close_cents=10_100),
            _quote_line(first, isin="BRWRONG00000"),
            _quote_line(first, ticker="WRONG11"),
            _quote_line(first, spec="UNT"),
            _quote_line(first, market_type=20),
            _quote_line(first, cod_bdi="96"),
        ],
    )
    output = tmp_path / "bova"
    manifest = build_bova11_series([source], output)
    manifest_sha = sha256_file(output / "manifest.json")
    loaded = load_bova11_series(
        output,
        expected_manifest_sha256=manifest_sha,
        canonical_dates=[first, second, date(2024, 1, 4)],
    )

    assert manifest["observation_count"] == 2
    assert manifest["security"]["bdi_codes"] == ["02", "14"]
    assert manifest["sources"][0]["bdi_row_counts"] == {"02": 1, "14": 1}
    np.testing.assert_allclose(loaded.close_by_session[:2], [100.0, 101.0])
    assert np.isnan(loaded.close_by_session[2])
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        load_bova11_series(
            output,
            expected_manifest_sha256="0" * 64,
            canonical_dates=[first, second],
        )

    pl.DataFrame(
        {"trade_date": [first, second], "close_brl": [100.0, 999.0]},
        schema={"trade_date": pl.Date, "close_brl": pl.Float64},
    ).write_parquet(output / "bova11_close.parquet")
    with pytest.raises(ValueError, match="hash or byte count mismatch"):
        load_bova11_series(
            output,
            expected_manifest_sha256=manifest_sha,
            canonical_dates=[first, second],
        )


@pytest.mark.parametrize("second_close", [10_000, 10_100])
def test_bova11_classifications_cannot_duplicate_or_conflict(
    tmp_path: Path, second_close: int
) -> None:
    day = date(2019, 8, 19)
    source = tmp_path / "COTAHIST_A2019.ZIP"
    _archive(
        source,
        2019,
        [
            _quote_line(day, cod_bdi="14", close_cents=10_000),
            _quote_line(day, cod_bdi="02", close_cents=second_close),
        ],
    )
    output = tmp_path / "bova"
    if second_close != 10_000:
        with pytest.raises(ValueError, match="conflicting BOVA11 close"):
            build_bova11_series([source], output)
        assert not output.exists()
    else:
        manifest = build_bova11_series([source], output)
        assert manifest["observation_count"] == 1
        loaded = load_bova11_series(
            output,
            expected_manifest_sha256=sha256_file(output / "manifest.json"),
            canonical_dates=[day],
        )
        np.testing.assert_array_equal(loaded.close_by_session, [100.0])


def test_bova11_loader_can_compare_a_sealed_single_classification_artifact(
    tmp_path: Path,
) -> None:
    day = date(2019, 8, 16)
    source = tmp_path / "COTAHIST_A2019.ZIP"
    _archive(source, 2019, [_quote_line(day)])
    output = tmp_path / "bova"
    manifest = build_bova11_series([source], output)
    manifest["security"].pop("bdi_codes")
    manifest["security"]["bdi_code"] = "14"
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    loaded = load_bova11_series(
        output,
        expected_manifest_sha256=sha256_file(output / "manifest.json"),
        canonical_dates=[day],
    )
    np.testing.assert_array_equal(loaded.close_by_session, [100.0])


def test_bova11_builder_refuses_official_period(tmp_path: Path) -> None:
    source = tmp_path / "COTAHIST_A2025.ZIP"
    _archive(source, 2025, [_quote_line(date(2025, 1, 2))])
    with pytest.raises(PermissionError, match="2025/2026"):
        build_bova11_series([source], tmp_path / "output")


def test_bova11_loader_rejects_dates_outside_canonical_calendar(tmp_path: Path) -> None:
    source = tmp_path / "COTAHIST_A2024.ZIP"
    source_day = date(2024, 1, 2)
    _archive(source, 2024, [_quote_line(source_day)])
    output = tmp_path / "bova"
    build_bova11_series([source], output)
    with pytest.raises(ValueError, match="outside the canonical calendar"):
        load_bova11_series(
            output,
            expected_manifest_sha256=sha256_file(output / "manifest.json"),
            canonical_dates=[date(2024, 1, 1), date(2024, 1, 3)],
        )


def test_bova11_loader_ignores_valid_history_before_consumer_calendar(
    tmp_path: Path,
) -> None:
    source = tmp_path / "COTAHIST_A2024.ZIP"
    first = date(2024, 1, 2)
    second = date(2024, 1, 3)
    _archive(
        source,
        2024,
        [_quote_line(first), _quote_line(second, close_cents=10_100)],
    )
    output = tmp_path / "bova"
    build_bova11_series([source], output)

    loaded = load_bova11_series(
        output,
        expected_manifest_sha256=sha256_file(output / "manifest.json"),
        canonical_dates=[second],
    )

    np.testing.assert_allclose(loaded.close_by_session, [101.0])
