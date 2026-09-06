from datetime import date

import polars as pl
import pytest

from brazil_rv.v2.data_foundation import (
    build_security_master,
    continuation_identity_axis,
    detect_isin_successions,
    filter_cash_equities,
    load_isin_link_allowlist,
    load_cotahist,
    panel_from_daily,
    prepare_cash_equities,
    validate_cotahist_daily,
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
        "currency": "R$",
        "quote_factor": 1,
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
    collapsed = filter_cash_equities(pl.DataFrame([rows[0], rows[0]]))
    assert collapsed.height == 1


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


def test_same_ticker_isin_succession_is_proposed_but_not_accepted_by_default(
    tmp_path,
) -> None:
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
        successor,
    ]
    allowlist = tmp_path / "isin_links_allowlist.csv"
    allowlist.write_text(
        "ticker,predecessor_isin,successor_isin,effective_date,first_known_at,"
        "shares_received_per_prior_share,cash_entitlement_per_prior_share,"
        "currency,source,evidence_sha256\n"
        f"TEST3,{predecessor},{successor},2024-01-04,2024-01-03T18:00:00Z,"
        f"1.0,0.0,BRL,issuer_notice,{'a' * 64}\n",
        encoding="utf-8",
    )
    accepted = load_isin_link_allowlist(allowlist, links)
    linked = build_security_master(daily, succession_links=accepted)
    assert linked.get_column("continuation_isin").to_list() == [
        predecessor,
        predecessor,
    ]


def test_empty_isin_allowlist_accepts_no_candidate(tmp_path) -> None:
    path = tmp_path / "isin_links_allowlist.csv"
    path.write_text(
        "ticker,predecessor_isin,successor_isin,effective_date,first_known_at,"
        "shares_received_per_prior_share,cash_entitlement_per_prior_share,"
        "currency,source,evidence_sha256\n",
        encoding="utf-8",
    )
    accepted = load_isin_link_allowlist(path, pl.DataFrame())
    assert accepted.is_empty()


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


def test_raw_validation_rejects_impossible_rows_and_conflicting_duplicates() -> None:
    valid = _row(date(2024, 1, 2), "BRTESTACNOR1", "TEST3")
    invalid = {
        **_row(date(2024, 1, 3), "BRTESTACNPR0", "TEST4"),
        "low_brl": 10.6,
    }
    result = prepare_cash_equities(
        pl.DataFrame([valid, invalid]), require_units=True
    )
    assert result.accepted.height == 1
    assert result.rejected[0, "raw_validation_reason"] == "inconsistent_ohlc_bounds"
    assert result.audit_by_year.filter(pl.col("reason") == "accepted")[
        0, "row_count"
    ] == 1

    conflicting = {**valid, "close_brl": 10.25}
    with pytest.raises(ValueError, match="conflicting COTAHIST"):
        validate_cotahist_daily(pl.DataFrame([valid, conflicting]))


def test_raw_validation_gate_reports_rejection_rate() -> None:
    rows = [
        _row(date(2024, 1, 2), "BRTESTACNOR1", "TEST3"),
        {
            **_row(date(2024, 1, 3), "BRTESTACNOR1", "TEST3"),
            "volume_brl": -1.0,
        },
    ]
    with pytest.raises(ValueError, match="invalid-row fraction"):
        validate_cotahist_daily(
            pl.DataFrame(rows), maximum_rejection_fraction=0.005
        )
