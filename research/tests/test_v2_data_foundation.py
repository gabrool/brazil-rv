from datetime import date

import polars as pl
import pytest

from brazil_rv.v2.data_foundation import (
    build_security_master,
    filter_cash_equities,
    load_cotahist,
    panel_from_daily,
    verify_v1_mapping,
)


def _row(day: date, isin: str, ticker: str, spec: str = "ON") -> dict[str, object]:
    return {
        "trade_date": day,
        "isin": isin,
        "ticker": ticker,
        "security_spec_base": spec,
        "bdi_code": "02",
        "market_type": 10,
        "open_brl": 10.0,
        "high_brl": 11.0,
        "low_brl": 9.0,
        "close_brl": 10.5,
        "volume_brl": 3_000_000.0,
        "trades": 100,
        "quantity": 1_000,
        "distribution_number": 1,
    }


def test_cash_filter_isin_identity_and_v1_exception() -> None:
    rows = [
        _row(date(2024, 1, 2), "BRTESTACNOR1", "TEST3"),
        _row(date(2024, 1, 2), "BRTESTACNPR0", "TEST4", "DRN"),
        {**_row(date(2024, 1, 3), "BRTESTACNOR1", "TEST3"), "market_type": 20},
    ]
    filtered = filter_cash_equities(
        pl.DataFrame(rows), v1_isins=("BRTESTACNPR0",)
    )
    assert filtered.select("isin").to_series().to_list() == [
        "BRTESTACNOR1",
        "BRTESTACNPR0",
    ]
    with pytest.raises(ValueError, match="one row"):
        filter_cash_equities(pl.DataFrame([rows[0], rows[0]]))


def test_cotahist_loader_applies_v1_exception_on_first_filter(tmp_path) -> None:
    path = tmp_path / "daily.parquet"
    pl.DataFrame(
        [_row(date(2024, 1, 2), "BRTESTACNPR0", "TEST4", "DRN")]
    ).write_parquet(path)
    loaded = load_cotahist([path], v1_isins=("BRTESTACNPR0",))
    assert loaded.get_column("isin").to_list() == ["BRTESTACNPR0"]


def test_security_master_splits_ticker_runs_and_panel_uses_isin() -> None:
    daily = pl.DataFrame(
        [
            _row(date(2024, 1, 2), "BRTESTACNOR1", "OLD3"),
            _row(date(2024, 1, 3), "BRTESTACNOR1", "OLD3"),
            _row(date(2024, 1, 4), "BRTESTACNOR1", "NEW3"),
        ]
    )
    master = build_security_master(daily)
    assert master.get_column("ticker").to_list() == ["OLD3", "NEW3"]
    assert master.get_column("last_date").to_list()[0] == date(2024, 1, 3)
    panel = panel_from_daily(daily)
    assert panel.isins == ("BRTESTACNOR1",)
    assert panel.observed.all()


def test_v1_mapping_is_strictly_one_to_one() -> None:
    valid = pl.DataFrame(
        {
            "security_id": ["ISIN:BRTESTACNOR1"],
            "isin": ["BRTESTACNOR1"],
        }
    )
    assert verify_v1_mapping(valid, ("BRTESTACNOR1",)).height == 1
    duplicate = pl.DataFrame(
        {
            "security_id": ["a", "b"],
            "isin": ["BRTESTACNOR1", "BRTESTACNOR1"],
        }
    )
    with pytest.raises(ValueError, match="multiple v1"):
        verify_v1_mapping(duplicate, ("BRTESTACNOR1",))
