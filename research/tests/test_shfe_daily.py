from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from brazil_rv.preprocessing.shfe_daily import (
    EXPOSURE_TICKERS,
    FEATURES,
    PULP_FEATURES,
    RAW_FEATURES,
    STEEL_FEATURES,
    ContractObservation,
    SecurityExposure,
    Snapshot,
    _security_frame,
    availability_date,
    build_market_rows,
    normalize_market_rows,
    parse_snapshot,
    resolve_exposures,
)


def _raw_contract(
    product: str,
    delivery: str,
    settlement: float,
    open_interest: float,
    *,
    volume: float = 100.0,
) -> dict[str, object]:
    return {
        "PRODUCTID": f"{product}_f    ",
        "DELIVERYMONTH": delivery,
        "SETTLEMENTPRICE": settlement,
        "VOLUME": volume,
        "OPENINTEREST": open_interest,
    }


def _payload(
    report_date: date,
    update_date: str,
    rows: list[dict[str, object]],
) -> bytes:
    return json.dumps(
        {
            "report_date": report_date.strftime("%Y%m%d"),
            "update_date": update_date,
            "o_curinstrument": rows,
        },
        ensure_ascii=False,
    ).encode()


def _observation(
    product: str,
    contract: str,
    settlement: float,
    open_interest: float,
    *,
    year: int = 2022,
    month: int = 1,
) -> ContractObservation:
    return ContractObservation(
        product=product,
        contract=contract,
        delivery_year=year,
        delivery_month=month,
        settlement=settlement,
        volume=100.0,
        open_interest=open_interest,
    )


def test_parse_exact_update_timestamp_and_availability_boundary() -> None:
    trade_date = date(2021, 8, 2)
    snapshot = parse_snapshot(
        _payload(
            trade_date,
            "20210802 20:00:35",
            [
                _raw_contract("rb", "2110", 5_497, 1_158_574),
                _raw_contract("rb", "小计", 0, 2_000_000),
                _raw_contract("sp", "2109", 6_136, 109_984),
            ],
        ),
        trade_date,
    )
    assert snapshot.available_at == datetime(2021, 8, 2, 12, 0, 35, tzinfo=timezone.utc)
    assert {row.contract for row in snapshot.contracts} == {"rb2110", "sp2109"}
    sessions = [trade_date, trade_date + timedelta(days=1)]
    assert availability_date(snapshot.available_at, sessions) == trade_date
    assert (
        availability_date(datetime(2021, 8, 2, 13, 15, tzinfo=timezone.utc), sessions)
        == trade_date
    )
    assert availability_date(
        datetime(2021, 8, 2, 13, 15, 1, tzinfo=timezone.utc), sessions
    ) == trade_date + timedelta(days=1)


def test_roll_uses_prior_oi_and_returns_never_cross_contracts() -> None:
    start = date(2021, 8, 2)
    snapshots: list[Snapshot] = []
    for index in range(7):
        snapshots.append(
            Snapshot(
                trade_date=start + timedelta(days=index),
                available_at=datetime(2021, 8, 2 + index, 12, tzinfo=timezone.utc),
                contracts=(
                    _observation(
                        "rb",
                        "rb2201",
                        100.0 + index,
                        100.0 if index == 0 else 10.0,
                    ),
                    _observation(
                        "rb",
                        "rb2205",
                        200.0 + 2 * index,
                        10.0 if index == 0 else 1_000.0,
                        month=5,
                    ),
                    _observation(
                        "hc",
                        "hc2201",
                        110.0 + index,
                        1_000.0,
                    ),
                    _observation(
                        "hc",
                        "hc2203",
                        112.0 + index,
                        100.0,
                        month=3,
                    ),
                    _observation(
                        "hc",
                        "hc2205",
                        114.0 + index,
                        100.0,
                        month=5,
                    ),
                ),
            )
        )
    rows = build_market_rows(snapshots)
    # The day-1 OI flip cannot affect day 1's selection.
    assert rows[1]["selected_rb_contract"] == "rb2201"
    assert rows[1]["rb_return_1d"] == pytest.approx(math.log(101.0 / 100.0))
    assert rows[1]["hc_minus_rb_log_ratio"] == pytest.approx(math.log(111.0 / 101.0))
    assert rows[1]["hc_term_slope_mask"]
    # Day 5 selects rb2205 from day 4 and compares rb2205 at both endpoints.
    assert rows[5]["selected_rb_contract"] == "rb2205"
    assert rows[5]["rb_return_5d"] == pytest.approx(math.log(210.0 / 200.0))


def _raw_market_rows(count: int) -> list[dict[str, object]]:
    start = date(2021, 8, 2)
    rows: list[dict[str, object]] = []
    for index in range(count):
        row: dict[str, object] = {
            "source_trade_date": start + timedelta(days=index),
            "available_at": datetime(2021, 8, 2, 12, tzinfo=timezone.utc)
            + timedelta(days=index),
        }
        for feature_index, feature in enumerate(RAW_FEATURES):
            row[feature] = (index + 1) * (feature_index + 1) / 1_000.0
            row[f"{feature}_mask"] = True
        rows.append(row)
    return rows


def test_normalization_is_prior_only_under_future_mutation() -> None:
    source = _raw_market_rows(30)
    baseline = normalize_market_rows(source)
    assert not baseline[19][f"{FEATURES[0]}_mask"]
    assert baseline[20][f"{FEATURES[0]}_mask"]
    mutated_source = _raw_market_rows(30)
    for feature in RAW_FEATURES:
        mutated_source[25][feature] = 1_000_000.0
    mutated = normalize_market_rows(mutated_source)
    for index in range(25):
        for feature in FEATURES:
            assert mutated[index][feature] == baseline[index][feature]
            assert (
                mutated[index][f"{feature}_mask"] == baseline[index][f"{feature}_mask"]
            )


def _assignments(path: Path, *, omit: str | None = None) -> None:
    rows = []
    counter = 0
    for tickers in EXPOSURE_TICKERS.values():
        for ticker in tickers:
            if ticker == omit:
                continue
            counter += 1
            isin = f"BRTEST{counter:05d}0"
            rows.append(
                {
                    "security_id": f"ISIN:{isin}",
                    "isin": isin,
                    "latest_ticker": ticker,
                    "first_overlap_date": "2021-07-19",
                    "last_overlap_date": "2026-07-17",
                }
            )
    pl.DataFrame(rows).write_parquet(path)


def _normalized_market_row(
    source_date: date, available_at: datetime
) -> dict[str, object]:
    row: dict[str, object] = {
        "source_trade_date": source_date,
        "available_at": available_at,
    }
    for feature_index, feature in enumerate(FEATURES):
        row[feature] = float(feature_index + 1)
        row[f"{feature}_mask"] = True
    return row


def test_identity_mapping_is_exact_bounded_and_group_masked(tmp_path: Path) -> None:
    assignment_path = tmp_path / "assignments.parquet"
    _assignments(assignment_path)
    exposures = resolve_exposures(assignment_path)
    assert len(exposures) == 6
    assert all(row.security_id == f"ISIN:{row.isin}" for row in exposures)

    # Two publications map to the same next B3 session; only the later state survives.
    rows = [
        _normalized_market_row(
            date(2021, 8, 3), datetime(2021, 8, 3, 12, tzinfo=timezone.utc)
        ),
        _normalized_market_row(
            date(2021, 8, 4), datetime(2021, 8, 4, 12, tzinfo=timezone.utc)
        ),
    ]
    frame = _security_frame(rows, exposures, [date(2021, 8, 2), date(2021, 8, 4)])
    assert frame.height == 6
    assert set(frame.get_column("source_trade_date")) == {date(2021, 8, 4)}
    steel = frame.filter(pl.col("exposure_group") == "steel")
    pulp = frame.filter(pl.col("exposure_group") == "pulp")
    assert all(steel.get_column(f"{feature}_mask").all() for feature in STEEL_FEATURES)
    assert all(
        not steel.get_column(f"{feature}_mask").any() for feature in PULP_FEATURES
    )
    assert all(pulp.get_column(f"{feature}_mask").all() for feature in PULP_FEATURES)
    assert all(
        not pulp.get_column(f"{feature}_mask").any() for feature in STEEL_FEATURES
    )

    missing_path = tmp_path / "missing.parquet"
    _assignments(missing_path, omit="SUZB3")
    with pytest.raises(ValueError, match="Expected one accepted identity for SUZB3"):
        resolve_exposures(missing_path)


def test_identity_bounds_are_enforced() -> None:
    exposure = SecurityExposure(
        security_id="ISIN:BRTEST000010",
        isin="BRTEST000010",
        ticker="CSNA3",
        group="steel",
        effective_from=date(2021, 8, 4),
        effective_to_inclusive=date(2021, 8, 4),
    )
    rows = [
        _normalized_market_row(
            date(2021, 8, 3), datetime(2021, 8, 3, 12, tzinfo=timezone.utc)
        ),
        _normalized_market_row(
            date(2021, 8, 4), datetime(2021, 8, 4, 12, tzinfo=timezone.utc)
        ),
    ]
    frame = _security_frame(
        rows,
        [exposure],
        [date(2021, 8, 3), date(2021, 8, 4), date(2021, 8, 5)],
    )
    assert frame.height == 1
    assert frame[0, "available_date"] == date(2021, 8, 4)
