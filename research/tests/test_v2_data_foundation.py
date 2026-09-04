from datetime import date

import numpy as np
import polars as pl
import pytest

from brazil_rv.v2.data_foundation import (
    build_security_master,
    continuation_identity_axis,
    detect_isin_successions,
    filter_cash_equities,
    inherit_linked_history,
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


def test_consecutive_same_ticker_isin_succession_links_history_and_survival_identity() -> None:
    predecessor = "BRTESTACNOR1"
    successor = "BRTESTACNPR0"
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    daily = pl.DataFrame(
        [
            _row(dates[0], predecessor, "TEST3"),
            _row(dates[1], predecessor, "TEST3"),
            _row(dates[2], successor, "TEST3"),
        ]
    )
    links = detect_isin_successions(daily)
    assert links.select(
        "ticker", "predecessor_isin", "successor_isin", "continuation_isin"
    ).row(0) == ("TEST3", predecessor, successor, predecessor)
    assert continuation_identity_axis((predecessor, successor), links) == (
        predecessor,
        predecessor,
    )
    master = build_security_master(daily)
    assert master.get_column("continuation_isin").to_list() == [
        predecessor,
        predecessor,
    ]

    values = np.asarray([[1.0, np.nan], [2.0, np.nan], [np.nan, 3.0]])
    inherited = inherit_linked_history(
        values, dates, (predecessor, successor), links
    )
    np.testing.assert_allclose(inherited[:, 1], [1.0, 2.0, 3.0])
    assert np.isnan(inherited[2, 0])


def test_ticker_reuse_after_gap_is_not_an_isin_succession() -> None:
    daily = pl.DataFrame(
        [
            _row(date(2024, 1, 2), "BRTESTACNOR1", "TEST3"),
            _row(date(2024, 1, 3), "BROTHERACN01", "OTHR3"),
            _row(date(2024, 1, 4), "BRTESTACNPR0", "TEST3"),
        ]
    )
    assert detect_isin_successions(daily).is_empty()


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
