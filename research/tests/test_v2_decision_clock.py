from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from brazil_rv.v2.decision_clock import (
    assert_calendar_complete,
    decision_timestamp,
    load_session_schedule,
    next_session_for_date_only,
    publication_available_at_decision,
)


def test_publication_clock_compares_absolute_instants_and_precision() -> None:
    session = date(2024, 7, 1)
    assert publication_available_at_decision(
        datetime(2024, 7, 1, 18, 44, 59, tzinfo=timezone.utc), session
    )
    assert not publication_available_at_decision(
        datetime(2024, 7, 1, 15, 45),
        session,
        naive_timezone="America/Sao_Paulo",
        precision="minute",
    )
    # 14:00 at UTC-5 is 19:00 UTC, after the 18:45 UTC decision.
    assert not publication_available_at_decision(
        datetime(2024, 7, 1, 14, 0, tzinfo=timezone(timedelta(hours=-5))),
        session,
    )
    with pytest.raises(ValueError, match="source timezone"):
        publication_available_at_decision(datetime(2024, 7, 1, 14, 0), session)
    assert decision_timestamp(session).tzinfo is not None


def test_date_only_publication_uses_next_exchange_session() -> None:
    sessions = [date(2024, 6, 28), date(2024, 7, 1), date(2024, 7, 2)]
    assert next_session_for_date_only(date(2024, 6, 28), sessions) == date(
        2024, 7, 1
    )
    assert next_session_for_date_only(date(2024, 6, 29), sessions) == date(
        2024, 7, 1
    )


def test_authoritative_schedule_loader_and_completeness_gate(tmp_path) -> None:
    schedule_path = tmp_path / "calendar.csv"
    schedule_path.write_text(
        "trade_date,continuous_open,decision_time,continuous_close,auction_close,timezone,source\n"
        "2024-01-02,10:00:00,15:45:00,16:55:00,17:00:00,America/Sao_Paulo,B3 fixture\n"
        "2024-01-03,10:00:00,15:45:00,16:55:00,17:00:00,America/Sao_Paulo,B3 fixture\n",
        encoding="utf-8",
    )
    schedule = load_session_schedule(schedule_path)
    assert [row.trade_date for row in schedule] == [
        date(2024, 1, 2),
        date(2024, 1, 3),
    ]
    assert_calendar_complete(schedule, [date(2024, 1, 2), date(2024, 1, 3)])
    with pytest.raises(ValueError, match="scheduled_session_missing"):
        assert_calendar_complete(schedule, [date(2024, 1, 2)])
