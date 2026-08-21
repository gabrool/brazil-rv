from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from brazil_rv.preprocessing import ccee_pld
from brazil_rv.preprocessing.ccee_pld import (
    FEATURES,
    POWER_ROLE_EXPOSURES,
    PldDay,
    build_power_rows,
    build_sidecar,
    derive_daily_features,
    parse_wide_rows,
)


def _day(trade_date: date, base: float) -> PldDay:
    return PldDay(
        trade_date=trade_date,
        values=tuple(
            tuple(base + hour + submarket / 10 for submarket in range(4))
            for hour in range(24)
        ),
    )


def _wide_rows(dates: list[date]) -> list[tuple[object, ...]]:
    header: tuple[object, ...] = (
        "Hora",
        "Submercado",
        *(datetime.combine(value, datetime.min.time()) for value in dates),
        None,
        "note",
    )
    rows: list[tuple[object, ...]] = [header]
    submarkets = ("SUDESTE", "SUL", "NORDESTE", "NORTE")
    for hour in range(24):
        for submarket_index, submarket in enumerate(submarkets):
            rows.append(
                (
                    hour,
                    submarket,
                    *(
                        50.0 + date_index + hour + submarket_index / 10
                        for date_index in range(len(dates))
                    ),
                    None,
                    None,
                )
            )
    # Ignored DST-repeat/note area after the canonical 96 rows.
    rows.append((23, "SUDESTE", *([999.0] + [None] * (len(dates) - 1)), None, None))
    rows.append((None, None, *([None] * len(dates)), None, "observation"))
    return rows


def test_wide_parser_uses_only_exact_24x4_block_and_filters_extras(
    tmp_path: Path,
) -> None:
    dates = [date(2021, 8, 16), date(2021, 8, 17)]
    days, audit = parse_wide_rows(_wide_rows(dates), tmp_path / "source.xlsx", "abc")
    assert [day.trade_date for day in days] == dates
    assert days[0].values[23][0] == 73.0
    assert audit.canonical_data_rows == 96
    assert audit.ignored_extra_rows == 2
    assert audit.ignored_extra_rows_with_date_values == 1
    assert audit.ignored_nondate_columns == 4


def test_wide_parser_accepts_complete_missing_dst_hour_but_not_partial() -> None:
    rows = _wide_rows([date(2018, 11, 4)])
    mutable = [list(row) for row in rows]
    for row in range(1, 5):
        mutable[row][2] = None
    days, audit = parse_wide_rows(
        [tuple(row) for row in mutable], Path("source.xlsx"), "abc"
    )
    assert days[0].values[0] == (None, None, None, None)
    assert audit.complete_23_hour_days == 1
    mutable[1][2] = 50.0
    with pytest.raises(ValueError, match="Partial submarket"):
        parse_wide_rows([tuple(row) for row in mutable], Path("source.xlsx"), "abc")


def test_shifted_normalization_and_future_mutation_are_causal() -> None:
    start = date(2021, 1, 1)
    days = [
        _day(start + timedelta(days=index), 50.0 + index / 3) for index in range(90)
    ]
    baseline = derive_daily_features(days)
    assert baseline[19]["pld_seco_daily_level_z60_mask"] is False
    assert baseline[20]["pld_seco_daily_level_z60_mask"] is True
    assert baseline[20]["pld_change_1d_surprise_z60_mask"] is False
    assert baseline[21]["pld_change_1d_surprise_z60_mask"] is True

    days[-1] = _day(days[-1].trade_date, 50_000.0)
    mutated = derive_daily_features(days)
    for earlier, changed in zip(baseline[:-1], mutated[:-1], strict=True):
        assert {
            feature: earlier[feature] for feature in FEATURES if feature in earlier
        } == {feature: changed[feature] for feature in FEATURES if feature in changed}


def test_power_rows_are_permanent_id_multi_hot_and_non_power_is_missing() -> None:
    rows = build_power_rows([_day(date(2021, 8, 16), 100.0)])
    assert len(rows) == len(POWER_ROLE_EXPOSURES)
    assert {str(row["security_id"]) for row in rows} == set(POWER_ROLE_EXPOSURES)
    eneva = next(row for row in rows if row["security_id"] == "ISIN:BRENEVACNOR8")
    assert eneva["power_role_thermal"] == 1.0
    assert eneva["power_role_hydro"] == 0.0
    assert eneva["power_role_hydro_mask"] is True
    assert not any(row["security_id"] == "ISIN:BRWEGEACNOR0" for row in rows)


def test_sidecar_is_immutable_and_records_source_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.xlsx"
    source.write_bytes(b"frozen workbook fixture")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    audit = ccee_pld.WorkbookAudit(
        source_file=str(source),
        source_sha256=source_hash,
        sheet_name="Test Sheet",
        workbook_rows=97,
        workbook_columns=3,
        canonical_data_rows=96,
        ignored_extra_rows=0,
        ignored_nondate_columns=2,
        date_columns=1,
        first_date="2021-08-16",
        last_date="2021-08-16",
        missing_calendar_dates=0,
        complete_24_hour_days=1,
        complete_23_hour_days=0,
        ignored_extra_rows_with_date_values=0,
    )
    monkeypatch.setattr(
        ccee_pld,
        "read_pld_workbook",
        lambda path, expected_sha256: ([_day(date(2021, 8, 16), 100.0)], audit),
    )
    output = tmp_path / "output"
    manifest = build_sidecar(source, output, expected_sha256=source_hash)
    frame = pl.read_parquet(output / "ccee_pld_daily_power.parquet")
    stored = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert frame.height == len(POWER_ROLE_EXPOSURES)
    assert stored["workbook_audit"]["source_sha256"] == source_hash
    assert stored["output_sha256"] == manifest["output_sha256"]
    assert set(stored["feature_valid_rows"]) == set(FEATURES)
    with pytest.raises(FileExistsError):
        build_sidecar(source, output, expected_sha256=source_hash)
