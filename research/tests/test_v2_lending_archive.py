from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from brazil_rv.v2.artifacts import sha256_file
from brazil_rv.v2.lending_archive import (
    BORROW_SOURCE_LABEL,
    LENDING_ARCHIVE_SCHEMA,
    _identical,
    load_lending_borrow_panels,
)


def _write_archive(
    root: Path, *, balances: pl.DataFrame, rates: pl.DataFrame
) -> str:
    root.mkdir()
    balance_path = root / "lending_balances.parquet"
    rate_path = root / "lending_rates.parquet"
    balances.write_parquet(balance_path)
    rates.write_parquet(rate_path)
    manifest = {
        "schema": LENDING_ARCHIVE_SCHEMA,
        "source_label": BORROW_SOURCE_LABEL,
        "status": "complete",
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (balance_path, rate_path)
        },
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return sha256_file(manifest_path)


def _balances(
    source: list[date],
    available: list[date],
    security: list[str],
    quantity: list[int],
    *,
    report: list[date] | None = None,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "source_position_date": pl.Series(source, dtype=pl.Date),
            "source_report_date": pl.Series(
                source if report is None else report, dtype=pl.Date
            ),
            "available_date": pl.Series(available, dtype=pl.Date),
            "security_id": security,
            "source_identity_method": ["test"] * len(source),
            "lending_balance_quantity": quantity,
            "lending_balance_brl": np.asarray(quantity, dtype=np.float64),
        }
    )


def _rates(
    source: list[date], available: list[date], security: list[str], rate: list[float]
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "source_trade_date": pl.Series(source, dtype=pl.Date),
            "available_date": pl.Series(available, dtype=pl.Date),
            "security_id": security,
            "registered_contracts": np.ones(len(source), dtype=np.int64),
            "registered_quantity": np.ones(len(source), dtype=np.int64),
            "annual_taker_rate": rate,
        }
    )


def test_overlap_identity_is_schema_and_value_exact() -> None:
    original = _rates(
        [date(2024, 6, 27)],
        [date(2024, 6, 28)],
        ["ISIN:BRA"],
        [0.08],
    )
    assert _identical(original, original.clone())
    changed = original.with_columns(pl.lit(0.081).alias("annual_taker_rate"))
    assert not _identical(original, changed)


def test_balance_without_trade_is_balance_shortable_and_rate_is_imputed(
    tmp_path: Path,
) -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    balance = _balances(
        [dates[0]], [dates[1]], ["ISIN:BRA"], [100]
    )
    rate = _rates(
        [dates[0]], [dates[1]], ["ISIN:BRB"], [0.08]
    )
    root = tmp_path / "archive"
    manifest_sha = _write_archive(root, balances=balance, rates=rate)

    panels = load_lending_borrow_panels(
        root,
        expected_manifest_sha256=manifest_sha,
        canonical_dates=dates,
        canonical_isins=["BRA", "BRB"],
    )

    assert panels.shortable_balance[1, 0]
    assert not panels.shortable_strict[1, 0]
    assert panels.shortable_open[1, 0]
    assert panels.rate_imputed[1, 0]
    assert panels.annual_taker_rate[1, 0] == pytest.approx(0.08)
    assert panels.shortable_strict[1, 1]
    assert not panels.rate_imputed[1, 1]
    assert panels.annual_taker_rate[0, 0] == pytest.approx(0.02)
    assert panels.rate_placeholder[0].all()
    assert panels.shortable_strict[0].all()
    assert panels.shortable_balance[0].all()
    assert panels.shortable_open[0].all()
    assert panels.source_unavailable_dates == ()
    assert panels.source_placeholder_dates == (dates[0],)


def test_balance_publication_lag_is_measured_from_report_not_position_date(
    tmp_path: Path,
) -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    root = tmp_path / "archive"
    manifest_sha = _write_archive(
        root,
        balances=_balances(
            [dates[0]],
            [dates[2]],
            ["ISIN:BRA"],
            [100],
            report=[dates[1]],
        ),
        rates=_rates([dates[0]], [dates[1]], ["ISIN:BRB"], [0.08]),
    )

    panels = load_lending_borrow_panels(
        root,
        expected_manifest_sha256=manifest_sha,
        canonical_dates=dates,
        canonical_isins=["BRA", "BRB"],
    )

    assert not panels.shortable_balance[1, 0]
    assert panels.shortable_balance[2, 0]
    assert not panels.shortable_strict[2, 0]


def test_loader_rejects_any_2025_or_2026_archive_row(tmp_path: Path) -> None:
    dates = [date(2024, 12, 30), date(2025, 1, 2)]
    root = tmp_path / "archive"
    manifest_sha = _write_archive(
        root,
        balances=_balances(
            [dates[0]], [dates[1]], ["ISIN:BRA"], [100]
        ),
        rates=_rates([dates[0]], [dates[1]], ["ISIN:BRA"], [0.08]),
    )

    with pytest.raises(PermissionError, match="refuses 2025/2026"):
        load_lending_borrow_panels(
            root,
            expected_manifest_sha256=manifest_sha,
            canonical_dates=dates,
            canonical_isins=["BRA"],
        )


def test_loader_fails_loudly_on_hash_mismatch(tmp_path: Path) -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    root = tmp_path / "archive"
    manifest_sha = _write_archive(
        root,
        balances=_balances(
            [dates[0]], [dates[1]], ["ISIN:BRA"], [100]
        ),
        rates=_rates([dates[0]], [dates[1]], ["ISIN:BRA"], [0.08]),
    )

    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        load_lending_borrow_panels(
            root,
            expected_manifest_sha256="0" * 64,
            canonical_dates=dates,
            canonical_isins=["BRA"],
        )
    assert manifest_sha != "0" * 64
