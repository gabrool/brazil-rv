from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import pytest

from brazil_rv.v2.decision_clock import (
    SessionDefinition,
    assert_calendar_complete,
    decision_timestamp,
    load_session_schedule,
    next_session_decision_cutoffs,
    next_session_for_date_only,
    publication_available_at_decision,
    schedule_source_label,
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
    assert next_session_for_date_only(date(2024, 6, 28), sessions) == date(2024, 7, 1)
    assert next_session_for_date_only(date(2024, 6, 29), sessions) == date(2024, 7, 1)


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


def test_in_memory_schedule_cannot_bypass_canonical_decision_clock() -> None:
    schedule = (
        SessionDefinition(
            trade_date=date(2024, 1, 2),
            continuous_open=time(10),
            decision_time=time(15, 44),
            continuous_close=time(16, 55),
            auction_close=time(17),
            source="fixture",
        ),
    )
    with pytest.raises(ValueError, match="15:45"):
        assert_calendar_complete(schedule, [date(2024, 1, 2)])


def test_schedule_source_label_and_next_decision_cutoffs() -> None:
    schedule = tuple(
        SessionDefinition(
            trade_date=value,
            continuous_open=time(10),
            decision_time=time(15, 45),
            continuous_close=time(16, 55),
            auction_close=time(17),
            source=f"reconstructed_v1:fixture_{index}",
        )
        for index, value in enumerate((date(2024, 1, 2), date(2024, 1, 3)))
    )
    assert schedule_source_label(schedule) == "reconstructed_v1"
    assert next_session_decision_cutoffs(schedule) == (schedule[1].decision_at, None)
    assert next_session_decision_cutoffs(
        schedule[:1], following_decision_at=schedule[1].decision_at
    ) == next_session_decision_cutoffs(schedule)[:1]
