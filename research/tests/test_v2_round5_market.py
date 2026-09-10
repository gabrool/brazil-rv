from datetime import UTC, date, datetime

import polars as pl

from brazil_rv.v2.round5_market import (
    OBSERVATION_SCHEMA,
    decision_snapshots,
    first_available_decision,
    parse_fred,
    parse_ptax,
    parse_treasury,
)


def test_ptax_actual_historical_time_and_minute_upper_bound():
    early = parse_ptax(
        b'{"value":[{"dataHoraCotacao":"2010-01-04 17:37:00", "cotacaoVenda":1.7232}]}',
        "historical.json",
    )[0]
    assert (
        first_available_decision(
            early["available_at"], [date(2010, 1, 4), date(2010, 1, 5)]
        )
        == 1
    )
    minute = parse_ptax(
        b'{"value":[{"dataHoraCotacao":"2024-01-02 15:44:00", "cotacaoVenda":4.9}]}',
        "fixture.json",
    )[0]
    assert (
        first_available_decision(
            minute["available_at"], [date(2024, 1, 2), date(2024, 1, 3)]
        )
        == 0
    )
    late = dict(minute, available_at=datetime(2024, 1, 2, 18, 45, 1, tzinfo=UTC))
    assert (
        first_available_decision(
            late["available_at"], [date(2024, 1, 2), date(2024, 1, 3)]
        )
        == 1
    )


def test_joined_future_mutation_changes_exact_first_available_decision():
    sessions = [date(2024, 3, day) for day in (7, 8, 11, 12)]
    rows = parse_fred(
        b"observation_date,VIXCLS\n2024-03-06,14.5\n2024-03-08,15.0\n",
        "vix.csv",
        "VIXCLS",
    )
    base = decision_snapshots(pl.DataFrame(rows, schema=OBSERVATION_SCHEMA), sessions)
    rows[-1]["value"] = 30.0
    changed = decision_snapshots(
        pl.DataFrame(rows, schema=OBSERVATION_SCHEMA), sessions
    )
    assert base.filter(pl.col("date") < date(2024, 3, 11)).equals(
        changed.filter(pl.col("date") < date(2024, 3, 11))
    )
    assert base["value"].to_list() == [14.5, 14.5, 15.0, 15.0]
    assert changed["value"].to_list() == [14.5, 14.5, 30.0, 30.0]
    assert base["age_sessions"].to_list() == [0, 1, 0, 1]


def test_us_dst_changes_utc_time_but_never_admits_same_day_close():
    rows = parse_fred(
        b"observation_date,VIXCLS\n2024-03-08,15\n2024-03-11,14\n", "vix.csv", "VIXCLS"
    )
    assert [row["available_at"].hour for row in rows] == [21, 20]
    sessions = [date(2024, 3, day) for day in (8, 11, 12)]
    assert [
        first_available_decision(row["available_at"], sessions) for row in rows
    ] == [1, 2]


def test_unknown_measurement_time_is_not_invented_and_missing_prints_stay_missing():
    rows = parse_fred(
        b"observation_date,DCOILBRENTEU\n2024-01-02,78\n2024-01-03,.\n",
        "brent.csv",
        "DCOILBRENTEU",
    )
    assert len(rows) == 1
    assert decision_snapshots(
        pl.DataFrame(rows, schema=OBSERVATION_SCHEMA),
        [date(2024, 1, 2), date(2024, 1, 3)],
    ).is_empty()


def test_treasury_units_and_next_decision():
    payload = b"""<feed xmlns="http://www.w3.org/2005/Atom" xmlns:m="urn:metadata" xmlns:d="urn:data"><entry><content><m:properties><d:NEW_DATE>2010-01-04T00:00:00</d:NEW_DATE><d:BC_3MONTH>0.08</d:BC_3MONTH><d:BC_2YEAR>1.09</d:BC_2YEAR><d:BC_5YEAR/><d:BC_10YEAR>3.85</d:BC_10YEAR></m:properties></content></entry></feed>"""
    rows = parse_treasury(payload, "treasury.xml")
    assert len(rows) == 3
    assert rows[0]["value"] == 0.0008
    assert (
        first_available_decision(
            rows[0]["available_at"], [date(2010, 1, 4), date(2010, 1, 5)]
        )
        == 1
    )
