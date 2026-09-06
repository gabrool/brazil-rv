from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Literal, Sequence
from zoneinfo import ZoneInfo

import polars as pl


SAO_PAULO = ZoneInfo("America/Sao_Paulo")
DECISION_TIME = time(15, 45)
TIMESTAMP_PRECISION = Literal["second", "minute", "date"]
SESSION_SCHEDULE_SCHEMA = "BRAZIL_RV_B3_EQUITY_SESSION_SCHEDULE_V1"


@dataclass(frozen=True)
class SessionDefinition:
    trade_date: date
    continuous_open: time
    decision_time: time
    continuous_close: time
    auction_close: time
    source: str

    @property
    def decision_at(self) -> datetime:
        return datetime.combine(self.trade_date, self.decision_time, tzinfo=SAO_PAULO)


def decision_timestamp(trade_date: date) -> datetime:
    """Return the v2 decision instant in historical Sao Paulo time."""

    return datetime.combine(trade_date, DECISION_TIME, tzinfo=SAO_PAULO)


def next_session_decision_cutoffs(
    schedule: Sequence[SessionDefinition],
) -> tuple[datetime | None, ...]:
    """Map a completed daily row to the first decision that may consume it."""

    validate_session_schedule(schedule)
    decisions = tuple(row.decision_at for row in schedule)
    return (*decisions[1:], None)


def schedule_source_label(schedule: Sequence[SessionDefinition]) -> str:
    """Return the one manifest label shared by an explicit session schedule."""

    validate_session_schedule(schedule)
    labels = {row.source.split(":", 1)[0] for row in schedule}
    if len(labels) != 1:
        raise ValueError(f"session schedule mixes source tiers: {sorted(labels)}")
    return labels.pop()


def normalize_publication_time(
    value: datetime,
    *,
    naive_timezone: str | None = None,
    precision: TIMESTAMP_PRECISION = "second",
    processing_delay: timedelta = timedelta(0),
) -> datetime:
    """Return the conservative UTC availability time for a publication.

    Offset-aware timestamps are absolute instants and are never relabelled.
    Naive values require an explicit IANA source timezone. Minute-resolution
    timestamps become usable only after the represented minute has ended.
    Date-only inputs require exchange-calendar resolution and are rejected here.
    """

    if precision == "date":
        raise ValueError("date-only publications require next-session resolution")
    if value.tzinfo is None or value.utcoffset() is None:
        if naive_timezone is None:
            raise ValueError("naive publication timestamp has no source timezone")
        value = value.replace(tzinfo=ZoneInfo(naive_timezone))
    available = value.astimezone(timezone.utc)
    if precision == "minute":
        available += timedelta(minutes=1)
    return available + processing_delay


def publication_available_at_decision(
    value: datetime,
    trade_date: date,
    *,
    naive_timezone: str | None = None,
    precision: TIMESTAMP_PRECISION = "second",
    processing_delay: timedelta = timedelta(0),
) -> bool:
    available = normalize_publication_time(
        value,
        naive_timezone=naive_timezone,
        precision=precision,
        processing_delay=processing_delay,
    )
    return available <= decision_timestamp(trade_date).astimezone(timezone.utc)


def next_session_for_date_only(
    publication_date: date, sessions: Sequence[date]
) -> date:
    """Apply the conservative D+1 rule on the authoritative session axis."""

    for session in sessions:
        if session > publication_date:
            return session
    raise ValueError("date-only publication has no following session")


def _parse_clock(value: object, *, field: str) -> time:
    if isinstance(value, time):
        return value.replace(tzinfo=None)
    text = str(value)
    try:
        return time.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"invalid {field}: {text}") from exc


def load_session_schedule(path: Path) -> tuple[SessionDefinition, ...]:
    """Load a versioned, source-attributed B3 equity session schedule."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = (
        pl.read_parquet(source)
        if source.suffix.lower() == ".parquet"
        else pl.read_csv(source, try_parse_dates=True)
    )
    required = {
        "trade_date",
        "continuous_open",
        "decision_time",
        "continuous_close",
        "auction_close",
        "timezone",
        "source",
    }
    if not required.issubset(frame.columns):
        raise ValueError(
            f"session schedule columns missing: {sorted(required - set(frame.columns))}"
        )
    frame = frame.with_columns(pl.col("trade_date").cast(pl.Date)).sort("trade_date")
    if frame.is_empty() or frame.get_column("trade_date").n_unique() != frame.height:
        raise ValueError("session schedule needs unique nonempty trading dates")
    if set(frame.get_column("timezone").cast(pl.String).to_list()) != {
        "America/Sao_Paulo"
    }:
        raise ValueError("session schedule timezone must be America/Sao_Paulo")
    rows: list[SessionDefinition] = []
    for row in frame.iter_rows(named=True):
        session = SessionDefinition(
            trade_date=row["trade_date"],
            continuous_open=_parse_clock(
                row["continuous_open"], field="continuous_open"
            ),
            decision_time=_parse_clock(row["decision_time"], field="decision_time"),
            continuous_close=_parse_clock(
                row["continuous_close"], field="continuous_close"
            ),
            auction_close=_parse_clock(row["auction_close"], field="auction_close"),
            source=str(row["source"]).strip(),
        )
        rows.append(session)
    schedule = tuple(rows)
    validate_session_schedule(schedule)
    return schedule


def validate_session_schedule(schedule: Sequence[SessionDefinition]) -> None:
    """Validate an in-memory schedule at the same boundary as a loaded one."""

    if not schedule:
        raise ValueError("session schedule must be nonempty")
    dates = [row.trade_date for row in schedule]
    if dates != sorted(dates) or len(set(dates)) != len(dates):
        raise ValueError("session schedule dates must be unique and chronological")
    for session in schedule:
        if not session.source.strip():
            raise ValueError("session schedule row lacks source attribution")
        if session.decision_time != DECISION_TIME:
            raise ValueError("v2 session schedule must use the 15:45 decision")
        if not (
            session.continuous_open
            < session.decision_time
            < session.continuous_close
            <= session.auction_close
        ):
            raise ValueError(f"invalid session clock ordering on {session.trade_date}")


def schedule_frame(schedule: Sequence[SessionDefinition]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "trade_date": [row.trade_date for row in schedule],
            "continuous_open": [row.continuous_open.isoformat() for row in schedule],
            "decision_time": [row.decision_time.isoformat() for row in schedule],
            "continuous_close": [row.continuous_close.isoformat() for row in schedule],
            "auction_close": [row.auction_close.isoformat() for row in schedule],
            "timezone": ["America/Sao_Paulo"] * len(schedule),
            "source": [row.source for row in schedule],
        }
    )


def calendar_completeness_table(
    schedule: Sequence[SessionDefinition], archive_dates: Sequence[date]
) -> pl.DataFrame:
    scheduled = {row.trade_date for row in schedule}
    archived = set(archive_dates)
    rows = [
        {"trade_date": value, "status": "scheduled_session_missing_from_archive"}
        for value in sorted(scheduled - archived)
    ]
    rows.extend(
        {"trade_date": value, "status": "archive_day_absent_from_schedule"}
        for value in sorted(archived - scheduled)
    )
    return pl.DataFrame(rows, schema={"trade_date": pl.Date, "status": pl.String})


def assert_calendar_complete(
    schedule: Sequence[SessionDefinition], archive_dates: Sequence[date]
) -> None:
    validate_session_schedule(schedule)
    mismatches = calendar_completeness_table(schedule, archive_dates)
    if mismatches.height:
        preview = mismatches.head(20).to_dicts()
        raise ValueError(f"B3 session/archive mismatch: {preview}")
