from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import MetaTrader5 as mt5
import numpy as np
import polars as pl

UTC = timezone.utc
SCRIPT_VERSION = "10"


@dataclass
class SymbolAudit:
    broker: str
    server: str
    symbol: str
    requested_start_api_utc: str
    requested_end_api_utc: str
    exchange_timezone: str
    selected: bool = False
    rows: int = 0
    first_server_wall: str | None = None
    last_server_wall: str | None = None
    first_exchange: str | None = None
    last_exchange: str | None = None
    first_utc: str | None = None
    last_utc: str | None = None
    months_requested: int = 0
    months_nonempty: int = 0
    chunks_requested: int = 0
    chunks_nonempty: int = 0
    duplicate_timestamps_removed: int = 0
    out_of_range_rows_removed: int = 0
    zero_tick_volume_share: float | None = None
    zero_real_volume_share: float | None = None
    null_count: int = 0
    output_file: str | None = None
    market_data_sha256: str | None = None
    sha256: str | None = None
    error: str | None = None


def utc_now() -> datetime:
    return datetime.now(UTC)


def parse_utc_date(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date {value!r}; expected YYYY-MM-DD"
        ) from exc


def validate_timezone(name: str) -> str:
    try:
        ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise argparse.ArgumentTypeError(
            f"Unknown IANA timezone {name!r}. On Windows run: uv add tzdata"
        ) from exc
    return name


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_market_data(frame: pl.DataFrame) -> str:
    """Hash canonical MT5 market fields, excluding retrieval/timestamp metadata."""

    canonical = (
        ("time", np.dtype("<i8")),
        ("open", np.dtype("<f8")),
        ("high", np.dtype("<f8")),
        ("low", np.dtype("<f8")),
        ("close", np.dtype("<f8")),
        ("tick_volume", np.dtype("<i8")),
        ("spread", np.dtype("<i8")),
        ("real_volume", np.dtype("<f8")),
    )

    digest = hashlib.sha256()
    digest.update(f"rows={frame.height}\n".encode("ascii"))
    for column, dtype in canonical:
        if column not in frame.columns:
            raise RuntimeError(f"MT5 bar data is missing required column: {column}")
        values = np.asarray(frame[column].to_numpy(), dtype=dtype)
        values = np.ascontiguousarray(values)
        digest.update(column.encode("ascii") + b"\0")
        digest.update(dtype.str.encode("ascii") + b"\0")
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def safe_filename(symbol: str) -> str:
    """Return a Windows-safe, collision-resistant filename component.

    B3/XP continuous symbols use ``$`` and ``@`` semantically (for example,
    ``WIN$N`` and ``WIN@N``).  Replacing both with a generic underscore would
    make different symbols overwrite the same Parquet file.  Windows permits
    both characters in filenames, so preserve them and hex-escape only truly
    unsafe characters.
    """

    allowed = set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789._-$@"
    )
    encoded = "".join(
        char if char in allowed else f"_{ord(char):04X}_"
        for char in symbol
    )
    return encoded or "symbol"


def namedtuple_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "_asdict"):
        return dict(value._asdict())
    return {"value": str(value)}


def picked(mapping: dict[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    return {field: mapping.get(field) for field in fields if field in mapping}


def initialize_terminal(
    terminal_path: Path, allow_nonreal: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not terminal_path.is_file():
        raise FileNotFoundError(f"MT5 executable not found: {terminal_path}")

    if not mt5.initialize(str(terminal_path), timeout=60_000):
        raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")

    terminal = namedtuple_dict(mt5.terminal_info())
    account = namedtuple_dict(mt5.account_info())

    if not terminal.get("connected", False):
        raise RuntimeError("MT5 terminal is open but not connected to a broker server")
    if not account:
        raise RuntimeError("No trading account is logged into the selected terminal")

    server = str(account.get("server", ""))
    trade_mode = account.get("trade_mode")
    real_mode = getattr(mt5, "ACCOUNT_TRADE_MODE_REAL", 2)
    looks_demo = "DEMO" in server.upper()
    if not allow_nonreal and (trade_mode != real_mode or looks_demo):
        raise RuntimeError(
            f"Expected a real/production account, but server={server!r}, "
            f"trade_mode={trade_mode!r}."
        )

    if terminal.get("tradeapi_disabled") is False:
        raise RuntimeError(
            "External Python trading is enabled in MT5. In Tools > Options > "
            "Expert Advisors, enable 'Disable automated trading through the "
            "external Python API', restart MT5, and rerun."
        )

    return terminal, account


def connection_metadata(
    broker: str,
    terminal_path: Path,
    terminal: dict[str, Any],
    account: dict[str, Any],
    exchange_timezone: str,
) -> dict[str, Any]:
    terminal_fields = (
        "connected",
        "trade_allowed",
        "tradeapi_disabled",
        "dlls_allowed",
        "build",
        "maxbars",
        "path",
        "data_path",
        "commondata_path",
        "ping_last",
    )
    account_fields = (
        "server",
        "company",
        "currency",
        "trade_mode",
        "leverage",
        "margin_mode",
        "fifo_close",
        "trade_allowed",
        "trade_expert",
    )
    return {
        "retrieved_at_utc": utc_now().isoformat(),
        "broker_tag": broker,
        "requested_terminal_path": str(terminal_path),
        "metatrader5_python_package": getattr(mt5, "__version__", None),
        "audit_script_version": SCRIPT_VERSION,
        "terminal_version": mt5.version(),
        "terminal": picked(terminal, terminal_fields),
        "account": picked(account, account_fields),
        "exchange_timezone": exchange_timezone,
        "time_semantics_note": (
            "For the XP B3 feed, empirical session-clock checks show that the raw "
            "MT5 'time' integer represents exchange/server wall-clock labels encoded "
            "as Unix seconds. The output preserves raw 'time', creates naive "
            "'ts_server_wall', localizes it to 'ts_exchange', and then derives the "
            "correct absolute 'ts_utc'. Revalidate this assumption for each broker."
        ),
        "privacy_note": (
            "Account login, name, balance and equity are deliberately omitted."
        ),
    }


def export_symbol_catalogue(out_dir: Path) -> pl.DataFrame:
    symbols = mt5.symbols_get()
    if symbols is None:
        raise RuntimeError(f"symbols_get() failed: {mt5.last_error()}")
    records = [namedtuple_dict(item) for item in symbols]
    frame = pl.DataFrame(records, strict=False, infer_schema_length=None)
    if "name" in frame.columns:
        frame = frame.sort("name")
    frame.write_parquet(out_dir / "symbol_catalogue.parquet", compression="zstd")
    frame.write_csv(out_dir / "symbol_catalogue.csv")
    return frame


def add_months(value: datetime, months: int) -> datetime:
    if months < 1:
        raise ValueError("months must be >= 1")
    zero_based = value.month - 1 + months
    year = value.year + zero_based // 12
    month = zero_based % 12 + 1
    return value.replace(year=year, month=month, day=1)


def chunk_ranges(
    start: datetime, end_exclusive: datetime, chunk_months: int
) -> list[tuple[datetime, datetime]]:
    ranges: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor < end_exclusive:
        next_boundary = add_months(cursor.replace(day=1), chunk_months)
        stop = min(next_boundary, end_exclusive)
        ranges.append((cursor, stop))
        cursor = stop
    return ranges


def calendar_month_count(start: datetime, end_exclusive: datetime) -> int:
    if start >= end_exclusive:
        return 0
    last_included = end_exclusive - timedelta(seconds=1)
    return (last_included.year - start.year) * 12 + last_included.month - start.month + 1


def rates_to_frame(rates: Any, exchange_timezone: str) -> pl.DataFrame:
    if rates is None or len(rates) == 0:
        return pl.DataFrame()

    names = rates.dtype.names
    if not names:
        raise RuntimeError("MT5 returned an array without named fields")

    # MT5 returns a NumPy structured array whose individual field views can be
    # strided and, on some Windows/Polars combinations, can still trigger a
    # Rust AsSliceError even after np.ascontiguousarray().  Avoid Polars' NumPy
    # zero-copy constructor entirely: materialize ordinary Python scalars and
    # build explicitly typed Series. Monthly M1 chunks are small enough that
    # this copy is negligible relative to the broker request and is far more
    # robust.
    expected_dtypes: dict[str, pl.DataType] = {
        "time": pl.Int64,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "tick_volume": pl.UInt64,
        "spread": pl.Int64,
        "real_volume": pl.UInt64,
    }

    series: list[pl.Series] = []
    for name in names:
        values = np.asarray(rates[name])
        if values.ndim != 1:
            values = values.reshape(-1)
        # ndarray.tolist() produces normal Python int/float objects and bypasses
        # the failing NumPy buffer/slice conversion path in Polars.
        python_values = values.tolist()
        dtype = expected_dtypes.get(name)
        series.append(
            pl.Series(name=name, values=python_values, dtype=dtype, strict=False)
        )

    frame = pl.DataFrame(series)
    server_wall = pl.from_epoch("time", time_unit="s")
    return frame.with_columns(
        server_wall.alias("ts_server_wall"),
        server_wall.dt.replace_time_zone(exchange_timezone).alias("ts_exchange"),
        server_wall
        .dt.replace_time_zone(exchange_timezone)
        .dt.convert_time_zone("UTC")
        .alias("ts_utc"),
    )


def request_chunk(
    symbol: str,
    start: datetime,
    end_exclusive: datetime,
    exchange_timezone: str,
) -> tuple[pl.DataFrame, int]:
    """Request one calendar chunk and enforce the requested half-open range.

    Some MT5 broker servers return the first available bar for a symbol even
    when the requested interval predates that symbol's listing.  Without an
    explicit range filter this can make pre-listing months look nonempty and
    create duplicate timestamps when monthly chunks are concatenated.
    """

    end_inclusive = end_exclusive - timedelta(seconds=1)
    start_epoch = int(start.timestamp())
    end_epoch = int(end_exclusive.timestamp())
    delays = (0.0, 2.0, 5.0, 10.0)
    last_error: Any = None
    last_removed = 0

    for delay in delays:
        if delay:
            time.sleep(delay)
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start, end_inclusive)
        if rates is not None:
            frame = rates_to_frame(rates, exchange_timezone)
            if frame.height > 0:
                before = frame.height
                returned_min = int(frame["time"].min())
                returned_max = int(frame["time"].max())
                frame = frame.filter(
                    (pl.col("time") >= start_epoch)
                    & (pl.col("time") < end_epoch)
                )
                last_removed = before - frame.height
                if frame.height > 0:
                    return frame, last_removed
                last_error = (
                    "empty after requested-range filter; "
                    f"removed {last_removed} out-of-range row(s)"
                )

                # XP commonly returns the first available bar when the requested
                # interval predates a symbol's listing (or the last available bar
                # when it postdates retained history). That out-of-range sentinel
                # is definitive for this interval; retrying it with 2/5/10-second
                # sleeps only slows bulk backfills.
                all_after = returned_min >= end_epoch
                all_before = returned_max < start_epoch
                if last_removed > 0 and (all_after or all_before):
                    print(
                        f"WARNING: {symbol}: no M1 bars for {start.date()} to "
                        f"{end_exclusive.date()} ({last_error}; definitive "
                        "out-of-range fallback, no retries)",
                        file=sys.stderr,
                    )
                    return pl.DataFrame(), last_removed
            else:
                last_error = "empty response"
        else:
            last_error = mt5.last_error()

    print(
        f"WARNING: {symbol}: no M1 bars for {start.date()} to "
        f"{end_exclusive.date()} ({last_error})",
        file=sys.stderr,
    )
    return pl.DataFrame(), last_removed


def nullable_share(frame: pl.DataFrame, column: str, predicate: pl.Expr) -> float | None:
    if frame.height == 0 or column not in frame.columns:
        return None
    return float(frame.select(predicate.mean()).item())


def timestamp_summary(epoch: int, exchange_timezone: str) -> tuple[str, str, str]:
    # The raw XP/B3 integer's calendar/clock fields align with exchange wall time.
    server_wall = datetime.fromtimestamp(epoch, tz=UTC).replace(tzinfo=None)
    exchange = server_wall.replace(tzinfo=ZoneInfo(exchange_timezone))
    corrected_utc = exchange.astimezone(UTC)
    return server_wall.isoformat(), exchange.isoformat(), corrected_utc.isoformat()


def audit_symbol(
    broker: str,
    server: str,
    symbol: str,
    start: datetime,
    end_exclusive: datetime,
    out_dir: Path,
    exchange_timezone: str,
    chunk_months: int,
) -> SymbolAudit:
    result = SymbolAudit(
        broker=broker,
        server=server,
        symbol=symbol,
        requested_start_api_utc=start.isoformat(),
        requested_end_api_utc=end_exclusive.isoformat(),
        exchange_timezone=exchange_timezone,
    )

    try:
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"Symbol not found on this server: {symbol}")

        result.selected = bool(mt5.symbol_select(symbol, True))
        if not result.selected:
            raise RuntimeError(f"symbol_select() failed: {mt5.last_error()}")

        chunks: list[pl.DataFrame] = []
        ranges = chunk_ranges(start, end_exclusive, chunk_months)
        result.months_requested = calendar_month_count(start, end_exclusive)
        result.chunks_requested = len(ranges)
        for chunk_start, chunk_end in ranges:
            frame, removed = request_chunk(
                symbol, chunk_start, chunk_end, exchange_timezone=exchange_timezone
            )
            result.out_of_range_rows_removed += removed
            if frame.height:
                result.chunks_nonempty += 1
                chunks.append(frame)

        if not chunks:
            raise RuntimeError("No M1 bars were returned for the requested range")

        combined = pl.concat(chunks, how="vertical_relaxed").sort("time")

        # Defense in depth: ensure no broker/API response outside the overall
        # requested half-open interval can enter the stored dataset.
        overall_start_epoch = int(start.timestamp())
        overall_end_epoch = int(end_exclusive.timestamp())
        before_range_filter = combined.height
        combined = combined.filter(
            (pl.col("time") >= overall_start_epoch)
            & (pl.col("time") < overall_end_epoch)
        )
        result.out_of_range_rows_removed += before_range_filter - combined.height

        before = combined.height
        combined = combined.unique(subset=["time"], keep="last", maintain_order=True)
        result.duplicate_timestamps_removed = before - combined.height
        result.months_nonempty = int(
            combined.select(
                pl.col("ts_server_wall").dt.strftime("%Y-%m").n_unique()
            ).item()
        )

        retrieved_at = utc_now().isoformat()
        combined = combined.with_columns(
            pl.lit(broker).alias("broker"),
            pl.lit(server).alias("server"),
            pl.lit(symbol).alias("symbol"),
            pl.lit(retrieved_at).alias("retrieved_at_utc"),
        )

        file_path = out_dir / f"bars_m1_{safe_filename(symbol)}.parquet"
        combined.write_parquet(file_path, compression="zstd", statistics=True)

        result.rows = combined.height
        result.market_data_sha256 = sha256_market_data(combined)

        first_epoch = int(combined["time"].min())
        last_epoch = int(combined["time"].max())
        (
            result.first_server_wall,
            result.first_exchange,
            result.first_utc,
        ) = timestamp_summary(first_epoch, exchange_timezone)
        (
            result.last_server_wall,
            result.last_exchange,
            result.last_utc,
        ) = timestamp_summary(last_epoch, exchange_timezone)

        result.null_count = int(sum(combined.null_count().row(0)))
        result.zero_tick_volume_share = nullable_share(
            combined, "tick_volume", pl.col("tick_volume").eq(0)
        )
        result.zero_real_volume_share = nullable_share(
            combined, "real_volume", pl.col("real_volume").eq(0)
        )
        result.output_file = str(file_path)
        result.sha256 = sha256_file(file_path)

    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        # pyo3/Polars panic exceptions inherit from BaseException rather than
        # Exception. Record the symbol-level failure and allow the remaining
        # requested symbols to continue, while still honoring user interrupts.
        result.error = f"{type(exc).__name__}: {exc}"

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Connect to one real MT5 terminal and audit/export B3 M1 bars."
    )
    parser.add_argument("--terminal", type=Path, required=True, help="Path to terminal64.exe")
    parser.add_argument("--broker", default="XP", help="Broker tag written into metadata")
    parser.add_argument("--out", type=Path, required=True, help="Immutable output directory")
    parser.add_argument(
        "--exchange-timezone",
        type=validate_timezone,
        default="America/Sao_Paulo",
        help="IANA timezone used to interpret XP/B3 server wall-clock labels",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Export metadata/catalogue, then stop",
    )
    parser.add_argument(
        "--skip-catalogue",
        action="store_true",
        help="Skip the 74k-symbol catalogue on repeat data runs",
    )
    parser.add_argument("--symbols", nargs="+", help="Exact MT5 symbol names")
    parser.add_argument("--start", type=parse_utc_date, help="API start date, YYYY-MM-DD")
    parser.add_argument("--end", type=parse_utc_date, help="API exclusive end date, YYYY-MM-DD")
    parser.add_argument(
        "--chunk-months",
        type=int,
        default=1,
        help=(
            "Number of calendar months per MT5 request. Use 6 or 12 for "
            "multi-year bulk backfills; default 1 retains conservative monthly chunks."
        ),
    )
    parser.add_argument(
        "--allow-nonreal",
        action="store_true",
        help="Allow a demo/non-real account (not recommended)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        terminal, account = initialize_terminal(args.terminal, args.allow_nonreal)
        server = str(account.get("server", ""))

        args.out.mkdir(parents=True, exist_ok=False)
        metadata = connection_metadata(
            broker=args.broker,
            terminal_path=args.terminal,
            terminal=terminal,
            account=account,
            exchange_timezone=args.exchange_timezone,
        )
        write_json(args.out / "connection_metadata.json", metadata)

        if args.skip_catalogue:
            print(f"Connected to {server!r}; skipped catalogue export.")
        else:
            catalogue = export_symbol_catalogue(args.out)
            print(f"Connected to {server!r}; exported {catalogue.height:,} symbols.")

        if args.list_only:
            return 0

        if not args.symbols or args.start is None or args.end is None:
            raise ValueError(
                "For M1 extraction, provide --symbols, --start and --end. "
                "The --end date is exclusive."
            )
        if args.start >= args.end:
            raise ValueError("--start must be earlier than --end")
        if args.chunk_months < 1 or args.chunk_months > 24:
            raise ValueError("--chunk-months must be between 1 and 24")

        results = [
            audit_symbol(
                broker=args.broker,
                server=server,
                symbol=symbol,
                start=args.start,
                end_exclusive=args.end,
                out_dir=args.out,
                exchange_timezone=args.exchange_timezone,
                chunk_months=args.chunk_months,
            )
            for symbol in args.symbols
        ]

        summary = pl.DataFrame([asdict(item) for item in results], strict=False)
        summary.write_csv(args.out / "m1_audit_summary.csv")
        summary.write_parquet(args.out / "m1_audit_summary.parquet", compression="zstd")

        failures = [item for item in results if item.error]
        for item in results:
            status = f"ERROR: {item.error}" if item.error else f"{item.rows:,} rows"
            print(f"{item.symbol}: {status}")
        return 2 if failures else 0

    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
