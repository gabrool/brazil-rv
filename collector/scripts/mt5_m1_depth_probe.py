from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import MetaTrader5 as mt5
import polars as pl

UTC = timezone.utc


@dataclass
class ProbeResult:
    broker: str
    server: str
    symbol: str
    symbol_found: bool = False
    selected: bool = False
    requested_count: int = 0
    rows_returned: int = 0
    may_be_truncated: bool = False
    first_server_wall: str | None = None
    last_server_wall: str | None = None
    first_exchange: str | None = None
    last_exchange: str | None = None
    first_utc: str | None = None
    last_utc: str | None = None
    unique_dates: int = 0
    unique_months: int = 0
    error: str | None = None


def namedtuple_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "_asdict"):
        return dict(value._asdict())
    return {"value": str(value)}


def safe_metadata(terminal_path: Path, broker: str) -> dict[str, Any]:
    terminal = namedtuple_dict(mt5.terminal_info())
    account = namedtuple_dict(mt5.account_info())
    return {
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "broker": broker,
        "terminal_path": str(terminal_path),
        "terminal": {
            k: terminal.get(k)
            for k in (
                "connected",
                "trade_allowed",
                "tradeapi_disabled",
                "dlls_allowed",
                "build",
                "maxbars",
                "path",
                "data_path",
            )
        },
        "account": {
            k: account.get(k)
            for k in ("server", "company", "trade_mode", "trade_allowed")
        },
    }


def initialize(terminal_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not terminal_path.is_file():
        raise FileNotFoundError(f"MT5 executable not found: {terminal_path}")
    if not mt5.initialize(str(terminal_path), timeout=60_000):
        raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")
    terminal = namedtuple_dict(mt5.terminal_info())
    account = namedtuple_dict(mt5.account_info())
    if not terminal.get("connected", False):
        raise RuntimeError("MT5 is not connected")
    if not account:
        raise RuntimeError("No MT5 account is logged in")
    real_mode = getattr(mt5, "ACCOUNT_TRADE_MODE_REAL", 2)
    server = str(account.get("server", ""))
    if account.get("trade_mode") != real_mode or "DEMO" in server.upper():
        raise RuntimeError(f"Expected real account; server={server!r}")
    if terminal.get("tradeapi_disabled") is False:
        raise RuntimeError("External Python trading must be disabled")
    return terminal, account


def timestamp_triplet(epoch: int, timezone_name: str) -> tuple[str, str, str]:
    wall = datetime.fromtimestamp(epoch, tz=UTC).replace(tzinfo=None)
    local = wall.replace(tzinfo=ZoneInfo(timezone_name))
    return wall.isoformat(), local.isoformat(), local.astimezone(UTC).isoformat()


def probe_symbol(
    broker: str,
    server: str,
    symbol: str,
    count: int,
    timezone_name: str,
) -> ProbeResult:
    result = ProbeResult(
        broker=broker,
        server=server,
        symbol=symbol,
        requested_count=count,
    )
    try:
        info = mt5.symbol_info(symbol)
        result.symbol_found = info is not None
        if info is None:
            raise RuntimeError("Symbol not found on this server")
        result.selected = bool(mt5.symbol_select(symbol, True))
        if not result.selected:
            raise RuntimeError(f"symbol_select() failed: {mt5.last_error()}")

        rates = None
        last_error: Any = None
        for delay in (0, 2, 5, 10):
            if delay:
                time.sleep(delay)
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, count)
            if rates is not None and len(rates):
                break
            last_error = mt5.last_error()
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No M1 bars returned: {last_error}")

        frame = pl.DataFrame({name: rates[name] for name in rates.dtype.names}).sort("time")
        result.rows_returned = frame.height
        result.may_be_truncated = frame.height >= count
        first_epoch = int(frame["time"].min())
        last_epoch = int(frame["time"].max())
        (
            result.first_server_wall,
            result.first_exchange,
            result.first_utc,
        ) = timestamp_triplet(first_epoch, timezone_name)
        (
            result.last_server_wall,
            result.last_exchange,
            result.last_utc,
        ) = timestamp_triplet(last_epoch, timezone_name)

        wall = pl.from_epoch("time", time_unit="s")
        counts = frame.select(
            wall.dt.date().n_unique().alias("dates"),
            wall.dt.truncate("1mo").n_unique().alias("months"),
        ).row(0)
        result.unique_dates = int(counts[0])
        result.unique_months = int(counts[1])
    except Exception as exc:
        result.error = str(exc)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quickly probe the full M1 history currently exposed by one MT5 server."
    )
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--broker", default="XP")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument(
        "--count",
        type=int,
        default=2_000_000,
        help="Maximum bars requested per symbol; 2m exceeds ~10 years of normal B3 M1",
    )
    parser.add_argument("--exchange-timezone", default="America/Sao_Paulo")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        try:
            ZoneInfo(args.exchange_timezone)
        except ZoneInfoNotFoundError as exc:
            raise RuntimeError("Install timezone data with: uv add tzdata") from exc
        _, account = initialize(args.terminal)
        server = str(account.get("server", ""))
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "connection_metadata.json").write_text(
            json.dumps(safe_metadata(args.terminal, args.broker), indent=2),
            encoding="utf-8",
        )
        results = [
            probe_symbol(
                args.broker,
                server,
                symbol,
                args.count,
                args.exchange_timezone,
            )
            for symbol in args.symbols
        ]
        summary = pl.DataFrame([asdict(x) for x in results], strict=False)
        summary.write_csv(args.out / "m1_depth_probe.csv")
        summary.write_parquet(args.out / "m1_depth_probe.parquet", compression="zstd")
        for result in results:
            if result.error:
                print(f"{result.symbol}: ERROR: {result.error}")
            else:
                trunc = " (COUNT LIMIT HIT)" if result.may_be_truncated else ""
                print(
                    f"{result.symbol}: {result.rows_returned:,} bars, "
                    f"{result.first_exchange} -> {result.last_exchange}{trunc}"
                )
        return 2 if any(x.error for x in results) else 0
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
