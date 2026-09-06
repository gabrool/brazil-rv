from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Sequence

import polars as pl

from .artifacts import sha256_file, write_json_atomic
from .decision_clock import DECISION_TIME, schedule_frame
from .decision_clock import SessionDefinition


SCHEDULE_RECONSTRUCTION_SCHEMA = "BRAZIL_RV_B3_SESSION_RECONSTRUCTION_V1"
EXCEPTION_STATUSES = {
    "weekday_nonholiday_without_archive_session",
    "archive_session_on_committed_holiday",
    "archive_day_with_1_to_49_records",
}
M1_START = date(2021, 8, 16)


def _read_table(path: Path) -> pl.DataFrame:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return (
        pl.read_parquet(source)
        if source.suffix.lower() == ".parquet"
        else pl.read_csv(source, try_parse_dates=True)
    )


def _cotahist_counts(paths: Sequence[Path]) -> pl.DataFrame:
    sources = tuple(Path(path).resolve() for path in paths)
    if not sources or any(not path.is_file() for path in sources):
        raise FileNotFoundError("COTAHIST daily inputs must all exist")
    return (
        pl.scan_parquet(sources)
        .select(pl.col("trade_date").cast(pl.Date))
        .group_by("trade_date")
        .len()
        .collect()
        .sort("trade_date")
    )


def _m1_bounds(assignments_path: Path) -> pl.DataFrame:
    assignments = _read_table(assignments_path)
    if "source_file" not in assignments.columns:
        raise ValueError("M1 assignments lack source_file")
    if assignments.height != 158:
        raise ValueError(
            f"reconstructed_v1 requires 158 accepted M1 names, got {assignments.height}"
        )
    sources = tuple(
        Path(value).resolve() for value in assignments["source_file"].to_list()
    )
    if any(not path.is_file() for path in sources):
        raise FileNotFoundError("an accepted M1 source file is missing")
    per_name = (
        pl.scan_parquet(sources)
        .select(
            pl.col("symbol"),
            pl.col("ts_exchange").dt.date().alias("trade_date"),
            pl.col("ts_exchange").dt.time().alias("clock"),
        )
        .group_by("trade_date", "symbol")
        .agg(
            pl.col("clock").min().alias("first_clock"),
            pl.col("clock").max().alias("last_clock"),
        )
        .collect()
    )
    first = (
        per_name.group_by("trade_date", "first_clock")
        .len()
        .sort(["trade_date", "len", "first_clock"], descending=[False, True, False])
        .group_by("trade_date", maintain_order=True)
        .first()
        .select("trade_date", "first_clock", pl.col("len").alias("first_support"))
    )
    last = (
        per_name.group_by("trade_date", "last_clock")
        .len()
        .sort(["trade_date", "len", "last_clock"], descending=[False, True, True])
        .group_by("trade_date", maintain_order=True)
        .first()
        .select("trade_date", "last_clock", pl.col("len").alias("last_support"))
    )
    return first.join(last, on="trade_date", how="inner", validate="1:1").sort(
        "trade_date"
    )


def _exception_candidates(counts: pl.DataFrame, holidays: pl.DataFrame) -> pl.DataFrame:
    if "trade_date" not in holidays.columns:
        raise ValueError("holiday table lacks trade_date")
    holiday_dates = set(
        holidays.select(pl.col("trade_date").cast(pl.Date))["trade_date"].to_list()
    )
    observed_counts = {
        row["trade_date"]: int(row["len"]) for row in counts.iter_rows(named=True)
    }
    sessions = {value for value, count in observed_counts.items() if count >= 50}
    if not sessions:
        raise ValueError("COTAHIST archive supplies no qualifying sessions")
    first = min(sessions)
    last = max(sessions)
    rows: list[dict[str, object]] = []
    current = first
    while current <= last:
        if (
            current.weekday() < 5
            and current not in holiday_dates
            and current not in sessions
        ):
            rows.append(
                {
                    "trade_date": current,
                    "status": "weekday_nonholiday_without_archive_session",
                    "archive_record_count": int(observed_counts.get(current, 0)),
                }
            )
        current += timedelta(days=1)
    for value in sorted(sessions & holiday_dates):
        rows.append(
            {
                "trade_date": value,
                "status": "archive_session_on_committed_holiday",
                "archive_record_count": observed_counts[value],
            }
        )
    for value, count in sorted(observed_counts.items()):
        if 1 <= count <= 49:
            rows.append(
                {
                    "trade_date": value,
                    "status": "archive_day_with_1_to_49_records",
                    "archive_record_count": count,
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "trade_date": pl.Date,
            "status": pl.String,
            "archive_record_count": pl.Int64,
        },
    ).sort("trade_date", "status")


def reconcile_schedule_exceptions(
    counts: pl.DataFrame,
    holidays: pl.DataFrame,
    explanations: pl.DataFrame,
) -> pl.DataFrame:
    """Return the exact explained exception inventory or fail closed."""

    required = {"trade_date", "status", "reason", "evidence"}
    if not required.issubset(explanations.columns):
        raise ValueError(
            f"schedule exception columns missing: {sorted(required - set(explanations.columns))}"
        )
    explained = explanations.select(
        pl.col("trade_date").cast(pl.Date),
        pl.col("status").cast(pl.String),
        pl.col("reason").cast(pl.String),
        pl.col("evidence").cast(pl.String),
    ).sort("trade_date", "status")
    if explained.select("trade_date", "status").n_unique() != explained.height:
        raise ValueError("schedule exception keys must be unique")
    if not set(explained["status"].to_list()).issubset(EXCEPTION_STATUSES):
        raise ValueError("schedule exception table contains an unknown status")
    if explained.filter(
        (pl.col("reason").str.strip_chars() == "")
        | (pl.col("evidence").str.strip_chars() == "")
    ).height:
        raise ValueError("every schedule exception requires a reason and evidence")
    candidates = _exception_candidates(counts, holidays)
    expected_keys = set(candidates.select("trade_date", "status").iter_rows())
    actual_keys = set(explained.select("trade_date", "status").iter_rows())
    if expected_keys != actual_keys:
        unexplained = sorted(expected_keys - actual_keys)
        stale = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"schedule reconstruction exception mismatch: unexplained={unexplained[:20]}, stale={stale[:20]}"
        )
    return candidates.join(
        explained, on=["trade_date", "status"], how="left", validate="1:1"
    ).sort("trade_date", "status")


def _regime_for_date(regimes: pl.DataFrame, value: date) -> dict[str, object]:
    required = {
        "start_date",
        "end_date",
        "continuous_open",
        "continuous_close",
        "auction_close",
        "component",
    }
    if not required.issubset(regimes.columns):
        raise ValueError(
            f"schedule regime columns missing: {sorted(required - set(regimes.columns))}"
        )
    matched = regimes.filter(
        (pl.col("start_date").cast(pl.Date) <= value)
        & (pl.col("end_date").cast(pl.Date) >= value)
    )
    if matched.height != 1:
        raise ValueError(f"schedule date {value} has {matched.height} dated regimes")
    return matched.row(0, named=True)


def _clock(value: object) -> time:
    return value if isinstance(value, time) else time.fromisoformat(str(value))


def _m1_clocks(row: dict[str, object]) -> tuple[time, time, time]:
    first_clock = _clock(row["first_clock"])
    last_clock = _clock(row["last_clock"])
    # The XP archive contains after-market prints and sparse trailing rows.  The
    # cross-name mode identifies the regular last minute; the next minute is the
    # start of the five-minute closing call and five minutes later is its end.
    if not (time(9, 45) <= first_clock <= time(13, 5)):
        raise ValueError(f"unsupported modal M1 open: {first_clock}")
    if time(16, 45) <= last_clock <= time(16, 59):
        continuous_close = time(16, 55)
        auction_close = time(17, 0)
    elif time(17, 45) <= last_clock <= time(18, 15):
        continuous_close = time(17, 55)
        auction_close = time(18, 0)
    else:
        raise ValueError(f"unsupported modal M1 close: {last_clock}")
    return first_clock, continuous_close, auction_close


def reconstruct_schedule(
    *,
    cotahist_counts: pl.DataFrame,
    holidays: pl.DataFrame,
    explanations: pl.DataFrame,
    regimes: pl.DataFrame,
    m1_bounds: pl.DataFrame,
) -> tuple[tuple[SessionDefinition, ...], pl.DataFrame]:
    """Reconstruct the labelled development calendar and its closed audit."""

    counts = cotahist_counts.select(
        pl.col("trade_date").cast(pl.Date), pl.col("len").cast(pl.Int64)
    ).sort("trade_date")
    exceptions = reconcile_schedule_exceptions(counts, holidays, explanations)
    bounds = {row["trade_date"]: row for row in m1_bounds.iter_rows(named=True)}
    sessions: list[SessionDefinition] = []
    for value in counts.filter(pl.col("len") >= 50)["trade_date"].to_list():
        if value >= M1_START:
            bound = bounds.get(value)
            if bound is None:
                raise ValueError(f"M1 schedule evidence missing for {value}")
            try:
                open_, continuous_close, auction_close = _m1_clocks(bound)
                component = "m1_cross_name_modal_bounds"
            except ValueError:
                regime = _regime_for_date(regimes, value)
                open_ = _clock(regime["continuous_open"])
                continuous_close = _clock(regime["continuous_close"])
                auction_close = _clock(regime["auction_close"])
                component = "dated_regime_clock_exception"
        else:
            regime = _regime_for_date(regimes, value)
            open_ = _clock(regime["continuous_open"])
            continuous_close = _clock(regime["continuous_close"])
            auction_close = _clock(regime["auction_close"])
            component = str(regime["component"])
        sessions.append(
            SessionDefinition(
                trade_date=value,
                continuous_open=open_,
                decision_time=DECISION_TIME,
                continuous_close=continuous_close,
                auction_close=auction_close,
                source=f"reconstructed_v1:{component}",
            )
        )
    return tuple(sessions), exceptions


def write_reconstructed_schedule(
    *,
    cotahist_paths: Sequence[Path],
    holidays_path: Path,
    exceptions_path: Path,
    regimes_path: Path,
    m1_assignments_path: Path,
    output_path: Path,
    audit_path: Path,
    source_urls: Sequence[str],
    holiday_source_fetch_date: date,
    holiday_source_archive_sha256: str,
) -> str:
    if len(holiday_source_archive_sha256) != 64:
        raise ValueError("holiday source archive SHA-256 must have 64 characters")
    counts = _cotahist_counts(cotahist_paths)
    holidays = _read_table(holidays_path)
    explanations = _read_table(exceptions_path)
    regimes = _read_table(regimes_path)
    bounds = _m1_bounds(m1_assignments_path)
    schedule, exception_audit = reconstruct_schedule(
        cotahist_counts=counts,
        holidays=holidays,
        explanations=explanations,
        regimes=regimes,
        m1_bounds=bounds,
    )
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    schedule_frame(schedule).write_csv(output)
    output_sha = sha256_file(output)
    clock_exception_dates = {
        row.trade_date
        for row in schedule
        if row.source.endswith(":dated_regime_clock_exception")
    }
    clock_exceptions = [
        {
            **row,
            "trade_date": row["trade_date"].isoformat(),
            "first_clock": str(row["first_clock"]),
            "last_clock": str(row["last_clock"]),
            "reason": "modal M1 boundary outside supported regular-session ranges; dated regime used",
        }
        for row in bounds.filter(
            pl.col("trade_date").is_in(sorted(clock_exception_dates))
        ).to_dicts()
    ]
    audit = {
        "schema": SCHEDULE_RECONSTRUCTION_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "schedule_source": "reconstructed_v1",
        "schedule_path": str(output),
        "schedule_sha256": output_sha,
        "session_count": len(schedule),
        "first_session": schedule[0].trade_date.isoformat(),
        "last_session": schedule[-1].trade_date.isoformat(),
        "minimum_cotahist_records": 50,
        "holiday_crosscheck": {
            "path": str(Path(holidays_path).resolve()),
            "sha256": sha256_file(Path(holidays_path).resolve()),
            "source_urls": list(source_urls),
            "source_fetch_date": holiday_source_fetch_date.isoformat(),
            "source_archive_sha256": holiday_source_archive_sha256.casefold(),
        },
        "exception_explanations": {
            "path": str(Path(exceptions_path).resolve()),
            "sha256": sha256_file(Path(exceptions_path).resolve()),
            "rows": [
                {
                    **row,
                    "trade_date": row["trade_date"].isoformat(),
                }
                for row in exception_audit.to_dicts()
            ],
            "unexplained_count": 0,
        },
        "dated_regimes": {
            "path": str(Path(regimes_path).resolve()),
            "sha256": sha256_file(Path(regimes_path).resolve()),
        },
        "m1_assignments": {
            "path": str(Path(m1_assignments_path).resolve()),
            "sha256": sha256_file(Path(m1_assignments_path).resolve()),
            "accepted_name_count": 158,
            "bounds_start": M1_START.isoformat(),
            "method": "cross-name modal first/last minute; five-minute closing call",
            "clock_exceptions": clock_exceptions,
        },
        "cotahist_inputs": [
            {
                "path": str(Path(path).resolve()),
                "sha256": sha256_file(Path(path).resolve()),
            }
            for path in cotahist_paths
        ],
    }
    return write_json_atomic(Path(audit_path).resolve(), audit)


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build reconstructed_v1 B3 schedule")
    parser.add_argument("--cotahist", type=Path, action="append", required=True)
    parser.add_argument("--holidays", type=Path, required=True)
    parser.add_argument("--exceptions", type=Path, required=True)
    parser.add_argument("--regimes", type=Path, required=True)
    parser.add_argument("--m1-assignments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--source-url", action="append", default=[])
    parser.add_argument(
        "--holiday-source-fetch-date", type=date.fromisoformat, required=True
    )
    parser.add_argument("--holiday-source-archive-sha256", required=True)
    args = parser.parse_args(arguments)
    digest = write_reconstructed_schedule(
        cotahist_paths=args.cotahist,
        holidays_path=args.holidays,
        exceptions_path=args.exceptions,
        regimes_path=args.regimes,
        m1_assignments_path=args.m1_assignments,
        output_path=args.output,
        audit_path=args.audit,
        source_urls=args.source_url,
        holiday_source_fetch_date=args.holiday_source_fetch_date,
        holiday_source_archive_sha256=args.holiday_source_archive_sha256,
    )
    print(
        json.dumps(
            {"schedule": str(args.output.resolve()), "audit_sha256": digest},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
