from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from numpy.lib.format import open_memmap
from numpy.typing import NDArray

from .contract import (
    ASSIGNMENTS_POINTER,
    CATALOGUE_PATH,
    CONTEXT_POINTER,
    GLOBAL_SOURCE_POINTER,
    HUMAN_PRIORS_POINTER,
    FIXED_RATE_CONTEXT_SYMBOLS,
    LOCAL_CONTEXT_SYMBOLS,
    LIQUIDITY_SELECTED_RATE_CONTEXT_SYMBOL,
    COTAHIST_POINTER,
    EXPECTED_EQUITIES,
    RATE_PERCENT_MAX,
    RATE_PERCENT_MIN,
    UNIVERSE_POINTER,
    output_array_specs,
)

SOURCE_COLUMNS = (
    "ts_exchange",
    "open",
    "high",
    "low",
    "close",
    "real_volume",
    "symbol",
)


@dataclass(frozen=True)
class CanonicalInputs:
    universe_dir: Path
    assignments_dir: Path
    cotahist_dir: Path
    context_dir: Path
    catalogue_path: Path
    global_source_dir: Path
    human_priors_dir: Path

    def manifest_entries(self) -> dict[str, dict[str, str]]:
        return {
            "point_in_time_universe": {
                "pointer": str(UNIVERSE_POINTER),
                "resolved_path": str(self.universe_dir),
            },
            "accepted_xp_assignments": {
                "pointer": str(ASSIGNMENTS_POINTER),
                "resolved_path": str(self.assignments_dir),
            },
            "parsed_cotahist": {
                "pointer": str(COTAHIST_POINTER),
                "resolved_path": str(self.cotahist_dir),
            },
            "xp_context_archive": {
                "pointer": str(CONTEXT_POINTER),
                "resolved_path": str(self.context_dir),
            },
            "xp_catalogue": {"resolved_path": str(self.catalogue_path)},
            "global_context_source": {
                "pointer": str(GLOBAL_SOURCE_POINTER),
                "resolved_path": str(self.global_source_dir),
            },
            "human_priors": {
                "pointer": str(HUMAN_PRIORS_POINTER),
                "resolved_path": str(self.human_priors_dir),
            },
        }


def resolve_pointer(pointer: Path) -> Path:
    resolved = Path(pointer.read_text(encoding="utf-8").strip())
    if not resolved.is_dir():
        raise FileNotFoundError(f"Canonical pointer {pointer} resolves to {resolved}")
    return resolved


def resolve_inputs() -> CanonicalInputs:
    return CanonicalInputs(
        universe_dir=resolve_pointer(UNIVERSE_POINTER),
        assignments_dir=resolve_pointer(ASSIGNMENTS_POINTER),
        cotahist_dir=resolve_pointer(COTAHIST_POINTER),
        context_dir=resolve_pointer(CONTEXT_POINTER),
        catalogue_path=CATALOGUE_PATH,
        global_source_dir=resolve_pointer(GLOBAL_SOURCE_POINTER),
        human_priors_dir=resolve_pointer(HUMAN_PRIORS_POINTER),
    )


def read_research_interval(universe_dir: Path) -> tuple[date, date]:
    manifest = json.loads((universe_dir / "manifest.json").read_text(encoding="utf-8"))
    return (
        date.fromisoformat(manifest["resolved_start_date"]),
        date.fromisoformat(manifest["resolved_end_date"]),
    )


def load_assignments(assignments_dir: Path) -> pl.DataFrame:
    path = assignments_dir / "xp_accepted_source_assignments_v1.parquet"
    assignments = (
        pl.read_parquet(path)
        .select(
            "security_id",
            "isin",
            "latest_ticker",
            "xp_symbol",
            "source_file",
            "source_assignment_type",
            "first_overlap_date",
            "last_overlap_date",
            "manual_decision",
            "normalization_rule",
        )
        .with_columns(
            pl.col("first_overlap_date").str.to_date(strict=True),
            pl.col("last_overlap_date").str.to_date(strict=True),
        )
        .sort("security_id")
    )
    validate_assignments(assignments)
    return assignments


def validate_assignments(assignments: pl.DataFrame) -> None:
    if assignments.height != EXPECTED_EQUITIES:
        raise ValueError(f"Expected {EXPECTED_EQUITIES} accepted assignments")
    if assignments.get_column("security_id").n_unique() != EXPECTED_EQUITIES:
        raise ValueError(
            "Accepted assignments must contain 158 unique security_id values"
        )
    if set(assignments.get_column("manual_decision")) != {"ACCEPTED"}:
        raise ValueError("Every accepted assignment must have manual_decision=ACCEPTED")
    if set(assignments.get_column("normalization_rule")) != {
        "FILTER_TO_COTAHIST_SECURITY_DATES"
    }:
        raise ValueError(
            "Every accepted assignment must use "
            "normalization_rule=FILTER_TO_COTAHIST_SECURITY_DATES"
        )


def cotahist_files(cotahist_dir: Path) -> list[Path]:
    files = sorted(cotahist_dir.glob("year=*/equities_daily_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No COTAHIST daily files under {cotahist_dir}")
    return files


def load_market_dates_and_security_dates(
    files: list[Path],
    security_ids: tuple[str, ...],
    research_start: date,
    research_end: date,
    *,
    allow_empty_security_dates: bool = False,
) -> tuple[tuple[date, ...], dict[str, frozenset[date]]]:
    daily = (
        pl.scan_parquet(files)
        .select("trade_date", "security_id")
        .filter(pl.col("trade_date").is_between(research_start, research_end))
        .collect()
    )
    market_dates = tuple(daily.get_column("trade_date").unique().sort().to_list())
    accepted = (
        daily.filter(pl.col("security_id").is_in(security_ids))
        .group_by("security_id")
        .agg(pl.col("trade_date").unique().sort())
    )
    dates_by_security = {
        row["security_id"]: frozenset(row["trade_date"])
        for row in accepted.iter_rows(named=True)
    }
    missing = [
        security_id
        for security_id in security_ids
        if not dates_by_security.get(security_id)
    ]
    if missing and not allow_empty_security_dates:
        raise ValueError(
            "Accepted securities without exact COTAHIST dates in requested "
            f"interval [{research_start}, {research_end}]: {missing}"
        )
    return market_dates, {
        security_id: dates_by_security.get(security_id, frozenset())
        for security_id in security_ids
    }


def expand_membership(
    membership: pl.DataFrame,
    market_dates: tuple[date, ...],
    security_ids: tuple[str, ...],
) -> NDArray[np.bool_]:
    date_axis = np.asarray(market_dates, dtype="datetime64[D]")
    slot_by_security = {
        security_id: slot for slot, security_id in enumerate(security_ids)
    }
    mask = np.zeros((len(market_dates), len(security_ids)), dtype=bool)
    for row in membership.filter(pl.col("is_member")).iter_rows(named=True):
        slot = slot_by_security.get(row["security_id"])
        if slot is None:
            continue
        start = int(np.searchsorted(date_axis, np.datetime64(row["effective_from"])))
        end_date = row["effective_to_exclusive"]
        end = (
            len(date_axis)
            if end_date is None
            else int(np.searchsorted(date_axis, np.datetime64(end_date)))
        )
        mask[start:end, slot] = True
    return mask


def validate_source_date_isolation(
    assignments: pl.DataFrame, valid_dates: dict[str, frozenset[date]]
) -> None:
    for group in assignments.partition_by("source_file"):
        rows = group.select("security_id", "source_file").iter_rows(named=True)
        seen: dict[date, str] = {}
        for row in rows:
            security_id = row["security_id"]
            for trade_date in valid_dates[security_id]:
                previous = seen.get(trade_date)
                if previous is not None:
                    raise ValueError(
                        "Overlapping accepted identity dates for source "
                        f"{row['source_file']}: {trade_date} belongs to "
                        f"{previous} and {security_id}"
                    )
                seen[trade_date] = security_id


def validate_physical_source_identity(
    assignments: pl.DataFrame, source: pl.DataFrame, source_path: Path
) -> None:
    assignment_symbols = assignments.get_column("xp_symbol").unique().to_list()
    if len(assignment_symbols) != 1:
        raise ValueError(
            f"Accepted source group {source_path} must have one unique xp_symbol"
        )
    internal_symbols = source.get_column("symbol").unique().to_list()
    if len(internal_symbols) != 1:
        raise ValueError(f"Physical XP source {source_path} must contain one symbol")
    if assignment_symbols[0] != internal_symbols[0]:
        raise ValueError(
            f"Physical XP source symbol mismatch for {source_path}: "
            f"assignment={assignment_symbols[0]}, parquet={internal_symbols[0]}"
        )


def discover_context_files(context_dir: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(context_dir.glob("*.parquet")):
        schema = pl.read_parquet_schema(path)
        if "symbol" not in schema:
            continue
        symbol_frame = pl.read_parquet(path, columns=["symbol"], n_rows=1)
        if symbol_frame.is_empty():
            continue
        symbol = symbol_frame.item()
        if symbol in LOCAL_CONTEXT_SYMBOLS:
            if symbol in found:
                raise ValueError(f"Multiple context sources found for {symbol}")
            found[symbol] = path
    missing = [symbol for symbol in LOCAL_CONTEXT_SYMBOLS if symbol not in found]
    if LIQUIDITY_SELECTED_RATE_CONTEXT_SYMBOL in missing:
        raise FileNotFoundError(
            "Missing required DI1$N M1 source in canonical XP context archive "
            f"{context_dir}. Extract only DI1$N for the research interval into "
            "the XP context archive; do not substitute DI1$D or DI1$."
        )
    if missing:
        raise FileNotFoundError(
            f"Missing context sources in canonical XP archive {context_dir}: {missing}"
        )
    return found


def load_context_expiries(catalogue_path: Path) -> dict[str, date]:
    fixed_di = FIXED_RATE_CONTEXT_SYMBOLS
    rows = (
        pl.read_parquet(catalogue_path, columns=["name", "expiration_time"])
        .filter(pl.col("name").is_in(fixed_di))
        .select("name", "expiration_time")
    )
    expiries: dict[str, date] = {}
    for row in rows.iter_rows(named=True):
        timestamp = row["expiration_time"]
        if timestamp is None or timestamp <= 0:
            continue
        expiries[row["name"]] = datetime.fromtimestamp(
            timestamp, tz=timezone.utc
        ).date()
    missing = [symbol for symbol in fixed_di if symbol not in expiries]
    if missing:
        raise ValueError(f"Missing or invalid fixed-DI expiries: {missing}")
    return expiries


def load_source_file(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path, columns=list(SOURCE_COLUMNS))


def full_session_final_closes(
    source: pl.DataFrame, market_dates: tuple[date, ...]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    date_count = len(market_dates)
    closes = np.zeros(date_count, dtype=np.float64)
    observed = np.zeros(date_count, dtype=bool)
    slot_by_date = {trade_date: slot for slot, trade_date in enumerate(market_dates)}
    final_closes = (
        source.with_columns(pl.col("ts_exchange").dt.date().alias("trade_date"))
        .filter(
            pl.col("trade_date").is_in(market_dates),
            pl.col("close").is_finite(),
            pl.col("close") > 0,
        )
        .sort("ts_exchange")
        .group_by("trade_date", maintain_order=True)
        .agg(pl.col("close").last())
    )
    for trade_date, close in final_closes.iter_rows():
        date_idx = slot_by_date[trade_date]
        closes[date_idx] = close
        observed[date_idx] = True
    return closes, observed


def prepare_session_bars(
    source: pl.DataFrame,
    source_path: Path,
    allowed_dates: frozenset[date],
    market_dates: tuple[date, ...],
    session_start_minute: int,
    session_minutes: int,
) -> pl.DataFrame:
    date_index = pl.DataFrame(
        {
            "trade_date": pl.Series(market_dates, dtype=pl.Date),
            "date_idx": pl.Series(range(len(market_dates)), dtype=pl.Int32),
        }
    )
    bars = (
        source.with_columns(
            pl.col("ts_exchange").dt.date().alias("trade_date"),
            (
                pl.col("ts_exchange").dt.hour().cast(pl.Int16) * 60
                + pl.col("ts_exchange").dt.minute().cast(pl.Int16)
                - session_start_minute
            )
            .cast(pl.Int16)
            .alias("minute_idx"),
        )
        .filter(
            pl.col("trade_date").is_in(tuple(allowed_dates)),
            pl.col("minute_idx").is_between(0, session_minutes - 1),
        )
        .join(date_index, on="trade_date", how="inner")
        .sort("ts_exchange")
    )
    validate_session_bars(bars, source_path)
    return bars


def validate_session_bars(bars: pl.DataFrame, source_path: Path) -> None:
    if bars.is_empty():
        return
    invalid = bars.filter(
        ~pl.col("open").is_finite()
        | ~pl.col("high").is_finite()
        | ~pl.col("low").is_finite()
        | ~pl.col("close").is_finite()
        | (pl.col("open") <= 0)
        | (pl.col("high") <= 0)
        | (pl.col("low") <= 0)
        | (pl.col("close") <= 0)
        | (pl.col("high") < pl.max_horizontal("open", "close"))
        | (pl.col("low") > pl.min_horizontal("open", "close"))
        | ~pl.col("real_volume").is_finite()
        | (pl.col("real_volume") <= 0)
        | (pl.col("ts_exchange").dt.second() != 0)
        | (pl.col("ts_exchange").dt.microsecond() != 0)
    )
    if not invalid.is_empty():
        _raise_bar_error("Invalid used bar", source_path, invalid.row(0, named=True))
    duplicates = bars.filter(pl.col("ts_exchange").is_duplicated())
    if not duplicates.is_empty():
        _raise_bar_error(
            "Duplicate used timestamp", source_path, duplicates.row(0, named=True)
        )


def validate_rate_source_scale(bars: pl.DataFrame, source_path: Path) -> None:
    """Require DI OHLC values in annual percentage-rate units."""
    if bars.is_empty():
        return
    values = bars.select("open", "high", "low", "close").to_numpy()
    minimum = float(values.min())
    maximum = float(values.max())
    if minimum < RATE_PERCENT_MIN or maximum > RATE_PERCENT_MAX:
        raise ValueError(
            "DI OHLC must use annual percentage-rate units within "
            f"[{RATE_PERCENT_MIN}, {RATE_PERCENT_MAX}]; "
            f"source={source_path}, observed_range=[{minimum}, {maximum}]"
        )


def _raise_bar_error(message: str, source_path: Path, row: dict[str, object]) -> None:
    raise ValueError(
        f"{message}: source={source_path}, symbol={row['symbol']}, "
        f"date={row['trade_date']}, timestamp={row['ts_exchange']}"
    )


def dense_grid(
    bars: pl.DataFrame, date_count: int, session_minutes: int
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    raw = np.zeros((date_count, session_minutes, 5), dtype=np.float64)
    observed = np.zeros((date_count, session_minutes), dtype=bool)
    if bars.is_empty():
        return raw, observed
    date_idx = bars.get_column("date_idx").to_numpy()
    minute_idx = bars.get_column("minute_idx").to_numpy()
    raw[date_idx, minute_idx] = bars.select(
        "open", "high", "low", "close", "real_volume"
    ).to_numpy()
    observed[date_idx, minute_idx] = True
    return raw, observed


def create_output_memmaps(output_dir: Path, date_count: int) -> dict[str, np.memmap]:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    arrays: dict[str, np.memmap] = {}
    for filename, spec in output_array_specs(date_count).items():
        array = open_memmap(
            output_dir / filename, mode="w+", dtype=spec.dtype, shape=spec.shape
        )
        array[...] = 0
        arrays[filename] = array
    return arrays
