from __future__ import annotations

import io
import json
import math
import zipfile
from copy import deepcopy
from datetime import UTC, date, datetime, time

import polars as pl
import pytest

from brazil_rv.v2.round5_cross_market import (
    NEW_YORK,
    SAO_PAULO,
    SHANGHAI,
    TENORS,
    build,
    calendar_gaps,
    dce_prefix,
    parse_di,
    parse_us,
    roll_returns,
    us_close,
)
from brazil_rv.v2.round5_market import (
    OBSERVATION_SCHEMA,
    decision_snapshots,
    first_available_decision,
)


def _di_payload(
    day: date, *, duplicated: bool = False, tenors: tuple[int, ...] = TENORS
) -> bytes:
    lines = []
    for index, tenor in enumerate(tenors):
        # The nominal 90d fixed vertex matures on the following business day.
        days = tenor + (tenor == 90)
        lines.append(
            f"{index + 1:06d}00101{day:%Y%m%d}T1{'PRE':5}{'DIxPRE':15}"
            f"{days:05d}{days * 2 // 3:05d}+{123180000:014d}F{tenor:05d}"
        )
    # Same text in an unrelated curve must never pass the PRE code filter.
    lines.append(lines[0].replace("T1PRE  ", "T1APR  "))
    if duplicated:
        lines.append(lines[0])
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as archive:
        archive.writestr("TaxaSwap.txt", "\r\n".join(lines))
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as archive:
        archive.writestr(f"TS{day:%y%m%d}.ex_", inner.getvalue())
    return outer.getvalue()


def test_di_fixed_tenor_keeps_holiday_roll_and_native_rate_scale():
    day = date(2024, 12, 30)
    rows = parse_di(_di_payload(day), day)
    assert len(rows) == 6
    assert rows[1]["tenor"] == 90
    assert rows[1]["calendar_days"] == 91
    assert rows[1]["rate_pct_252"] == 12.318
    with pytest.raises(ValueError, match="source date"):
        parse_di(_di_payload(day), date(2024, 12, 27))
    with pytest.raises(ValueError, match="duplicate"):
        parse_di(_di_payload(day, duplicated=True), day)
    # Some official archives omit individual fixed vertices. Keep the other
    # observed quotes instead of dropping a whole curve or inventing a rate.
    partial = (30, 180, 720, 1080)
    assert [
        row["tenor"] for row in parse_di(_di_payload(day, tenors=partial), day)
    ] == list(partial)


def test_dce_prefix_stops_before_future_prices_even_if_payload_is_not_json(tmp_path):
    path = tmp_path / "I2501.json"
    past = b'{"d":"2024-12-30","s":"100","v":"12","p":"55"}'
    boundary = b'{"d":"2024-12-31"'
    prefix = b"var _brazil_rv=([" + past + b"," + boundary
    # Invalid later values make any full JSON parse fail. They must not be read.
    path.write_bytes(prefix + b',"s":THIS_PAYLOAD_MUST_NOT_BE_CONSUMED')
    rows, audit = dce_prefix(path)
    assert len(rows) == 1 and rows[0]["d"] == "2024-12-30"
    assert audit["consumed_bytes"] == len(prefix)
    assert audit["first_excluded_date_marker"] == "2024-12-31"


def test_dce_prefix_rejects_nonchronological_rows_and_missing_date_first(tmp_path):
    path = tmp_path / "I2401.json"
    path.write_text('var _brazil_rv=([{"d":"2024-01-03"},{"d":"2024-01-02"}]);')
    with pytest.raises(ValueError, match="chronological"):
        dce_prefix(path)
    path.write_text('var _brazil_rv=([{"s":"100","d":"2024-01-03"}]);')
    with pytest.raises(ValueError, match="first object field"):
        dce_prefix(path)


def _contract(day: int, name: str, price: float, oi: float, volume: float = 10) -> dict:
    reference = date(2024, 1, day)
    return {
        "product": "dce_iron",
        "reference_date": reference,
        "available_at": datetime.combine(reference, time(15), SHANGHAI).astimezone(UTC),
        "contract": name,
        "settlement": price,
        "open_interest": oi,
        "volume": volume,
        "source_file": f"{name}.json",
    }


def test_prior_oi_selects_same_contract_without_roll_jump_or_current_oi_leak():
    rows = [
        _contract(2, "I2405", 100, 100),
        _contract(2, "I2409", 200, 80),
        _contract(3, "I2405", 101, 50),
        _contract(3, "I2409", 210, 150),
        _contract(4, "I2405", 103, 40),
        _contract(4, "I2409", 212, 160),
    ]
    result = roll_returns(rows)
    assert result[0]["selected_contract"] == "I2405"
    assert result[0]["log_return"] == pytest.approx(math.log(101 / 100))
    assert result[1]["selected_contract"] == "I2409"
    assert result[1]["log_return"] == pytest.approx(math.log(212 / 210))
    mutated = deepcopy(rows)
    mutated[3]["open_interest"] = 1e9
    mutated[-1]["settlement"] = 1e6
    assert roll_returns(mutated)[0] == result[0]


def test_official_holiday_keeps_return_but_missing_session_and_unknown_gap_mask():
    rows = [_contract(2, "I2405", 100, 100), _contract(4, "I2405", 105, 100)]
    missing, audit = calendar_gaps(
        rows, "define(function(){return {2024:'20240103'};});"
    )
    assert missing == {"dce_iron": set()}
    assert len(roll_returns(rows, missing)) == 1
    missing, _ = calendar_gaps(rows, "define(function(){return {2024:'20240101'};});")
    assert roll_returns(rows, missing) == []
    missing, audit = calendar_gaps(rows, "")
    assert roll_returns(rows, missing) == []
    assert audit["calendar_years"] == []


def test_missing_selected_contract_is_masked_without_current_day_replacement():
    rows = [
        _contract(2, "I2405", 100, 100),
        _contract(2, "I2409", 200, 80),
        _contract(3, "I2409", 210, 150),
    ]
    assert roll_returns(rows) == []
    rows.append(_contract(3, "I2405", 101, 60, volume=0))
    assert roll_returns(rows) == []


def test_failed_source_day_does_not_become_a_one_session_return():
    rows = [_contract(2, "I2405", 100, 100), _contract(4, "I2405", 104, 100)]
    assert len(roll_returns(rows)) == 1
    assert roll_returns(rows, {"dce_iron": {date(2024, 1, 3)}}) == []


@pytest.mark.parametrize(
    "day, expected_hour",
    [
        (date(2010, 11, 26), 13),
        (date(2012, 7, 3), 13),
        (date(2013, 7, 3), 13),
        (date(2015, 7, 2), 16),
        (date(2018, 12, 24), 13),
        (date(2024, 7, 3), 13),
        (date(2024, 11, 29), 13),
        (date(2024, 12, 30), 16),
    ],
)
def test_us_regular_and_early_close_schedule(day, expected_hour):
    assert us_close(day).astimezone(NEW_YORK).hour == expected_hour


def test_half_day_first_eligible_decision_respects_historical_brazil_dst():
    # Brazil's old summer time made November's 13:00 NY close 16:00 locally.
    old = date(2010, 11, 26)
    new = date(2024, 11, 29)
    assert us_close(old).astimezone(SAO_PAULO).hour == 16
    assert first_available_decision(us_close(old), [old, date(2010, 11, 29)]) == 1
    assert us_close(new).astimezone(SAO_PAULO).hour == 15
    assert first_available_decision(us_close(new), [new, date(2024, 12, 2)]) == 0
    assert first_available_decision(us_close(date(2024, 7, 3)), [date(2024, 7, 3)]) == 0


def test_asia_and_di_source_mutation_first_changes_exact_eligible_decision():
    sessions = [date(2024, 1, d) for d in (2, 3, 4)]
    for zone, hour, first_changed in [(SHANGHAI, 15, 1), (SAO_PAULO, 23, 2)]:
        rows = [
            {
                "series": "test",
                "reference_date": day,
                "value": float(i + 1),
                "available_at": datetime.combine(day, time(hour), zone),
                "source_file": "fixture",
            }
            for i, day in enumerate(sessions)
        ]
        before = decision_snapshots(
            pl.DataFrame(rows, schema=OBSERVATION_SCHEMA), sessions
        )
        rows[1]["value"] = 99.0
        after = decision_snapshots(
            pl.DataFrame(rows, schema=OBSERVATION_SCHEMA), sessions
        )
        assert before.filter(pl.col("date") < sessions[first_changed]).equals(
            after.filter(pl.col("date") < sessions[first_changed])
        )
        assert (
            before.filter(pl.col("date") == sessions[first_changed])["value"][0]
            != (after.filter(pl.col("date") == sessions[first_changed])["value"][0])
        )


def test_us_preserves_vendor_fields_without_fabricating_cash_price_or_premium():
    timestamp = int(datetime(2024, 7, 3, 9, 30, tzinfo=NEW_YORK).timestamp())
    result = {
        "meta": {"symbol": "PBR", "currency": "USD", "regularMarketPrice": 999},
        "timestamp": [timestamp],
        "indicators": {
            "quote": [
                {
                    "open": [10],
                    "high": [11],
                    "low": [9],
                    "close": [10.5],
                    "volume": [100],
                }
            ],
            "adjclose": [{"adjclose": [5.25]}],
        },
    }
    payload = json.dumps({"chart": {"error": None, "result": [result]}}).encode()
    row = parse_us(payload, "PBR", "PBR.json")[0]
    assert row["close"] == 10.5 and row["adjusted_close"] == 5.25
    assert "regularMarketPrice" not in row
    assert row["available_at"] == us_close(date(2024, 7, 3))
    result["meta"]["symbol"] = "SUZ"
    result["timestamp"] = [
        int(datetime(2018, 7, 3, 9, 30, tzinfo=NEW_YORK).timestamp())
    ]
    payload = json.dumps({"chart": {"error": None, "result": [result]}}).encode()
    historical_otc = parse_us(payload, "SUZ", "SUZ.json")[0]
    assert historical_otc["available_at"] is None
    assert historical_otc["availability_status"] == "historical_otc_clock_unverified"


def test_source_build_preserves_raw_and_masks_multi_session_us_gap(tmp_path):
    import hashlib

    root = tmp_path / "output"
    root.mkdir()
    dce = tmp_path / "old_dce"
    dce.mkdir()
    (dce / "manifest.json").write_text(json.dumps({"files": []}))
    records = []
    source_bytes = {}
    day = date(2024, 7, 2)
    for name, family, key, payload in [
        ("di.zip", "di", day.isoformat(), _di_payload(day)),
    ]:
        path = root / name
        path.write_bytes(payload)
        source_bytes[path] = payload
        records.append(
            {
                "family": family,
                "key": key,
                "path": str(path),
                "status": "retrieved",
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    for symbol, days in [("EWZ", [2, 3, 5]), ("PBR", [2, 5])]:
        n = len(days)
        values = [10.0 + i for i in range(n)]
        payload = json.dumps(
            {
                "chart": {
                    "error": None,
                    "result": [
                        {
                            "meta": {"symbol": symbol, "currency": "USD"},
                            "timestamp": [
                                int(
                                    datetime(
                                        2024, 7, d, 9, 30, tzinfo=NEW_YORK
                                    ).timestamp()
                                )
                                for d in days
                            ],
                            "indicators": {
                                "quote": [
                                    {
                                        "open": values,
                                        "high": values,
                                        "low": values,
                                        "close": values,
                                        "volume": [100] * n,
                                    }
                                ],
                                "adjclose": [{"adjclose": values}],
                            },
                        }
                    ],
                }
            }
        ).encode()
        path = root / f"{symbol}.json"
        path.write_bytes(payload)
        source_bytes[path] = payload
        records.append(
            {
                "family": "us",
                "key": symbol,
                "path": str(path),
                "status": "retrieved",
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    acquisition = root / "acquisition.json"
    acquisition.write_text(json.dumps({"records": records, "dce_root": str(dce)}))
    result = build(root, acquisition)
    assert result["outputs"]["cross_market_observations.parquet"]["rows"] == 6
    obs = pl.read_parquet(root / "cross_market_observations.parquet")
    assert obs["value"].to_list() == pytest.approx([0.12318] * 6)
    returns = pl.read_parquet(root / "us_returns.parquet")
    assert returns["series"].to_list() == ["us_EWZ", "us_EWZ"]
    assert returns["reference_date"].to_list() == [date(2024, 7, 3), date(2024, 7, 5)]
    assert all(path.read_bytes() == original for path, original in source_bytes.items())
    with pytest.raises(FileExistsError):
        build(root, acquisition)
