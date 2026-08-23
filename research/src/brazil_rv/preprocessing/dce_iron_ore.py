from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import tempfile
import time
import urllib.request
from bisect import bisect_left
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

CONTRACT_VERSION = "SINA_DCE_IRON_ORE_PIT_V1"
RAW_CONTRACT_VERSION = "SINA_DCE_CONTRACT_DAILY_SNAPSHOT_V1"
URL = (
    "https://stock2.finance.sina.com.cn/futures/api/jsonp.php/"
    "var%20_brazil_rv=/InnerFuturesNewService.getDailyKLine"
    "?symbol={symbol}&type=2021_04_12"
)
PAYLOAD = re.compile(rb"=\((.*)\);?\s*$", re.DOTALL)
TARGETS = {
    "VALE3": "producer",
    "CMIN3": "producer",
    "CSNA3": "producer_steel",
    "GGBR4": "steel",
    "GOAU4": "steel",
    "USIM5": "steel",
}
ROLLING_OBSERVATIONS = 60
MIN_ROLLING_OBSERVATIONS = 20
ROBUST_CLIP = 5.0
FEATURES = (
    "dce_iron_ore_return_1d_z60",
    "dce_iron_ore_return_5d_z60",
    "dce_iron_ore_oi_change_1d_z60",
    "dce_iron_ore_curve_slope_z60",
    "dce_iron_ore_role_producer",
    "dce_iron_ore_role_steel",
)


@dataclass(frozen=True)
class ContractDay:
    symbol: str
    trade_date: date
    settlement: float
    open_interest: int
    volume: int


@dataclass(frozen=True)
class Exposure:
    security_id: str
    ticker: str
    role: str
    effective_from: date
    effective_to_inclusive: date


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _symbols(start_year: int, end_year: int) -> list[str]:
    return [
        f"I{year % 100:02d}{month:02d}"
        for year in range(start_year, end_year + 1)
        for month in range(1, 13)
    ]


def _download(symbol: str, destination: Path) -> dict[str, object]:
    url = URL.format(symbol=symbol)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 Brazil-RV research",
            "Referer": "https://finance.sina.com.cn/",
        },
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            match = PAYLOAD.search(payload)
            if match is None:
                raise ValueError(f"Unexpected Sina JSONP response: {symbol}")
            parsed = json.loads(match.group(1))
            if parsed is None:
                return {"symbol": symbol, "status": "not_listed", "url": url}
            if not isinstance(parsed, list) or not parsed:
                raise ValueError(f"Sina response has no daily rows: {symbol}")
            destination.write_bytes(payload)
            return {
                "symbol": symbol,
                "status": "downloaded",
                "url": url,
                "filename": destination.name,
                "bytes": len(payload),
                "sha256": _sha256(destination),
                "row_count": len(parsed),
                "first_date": parsed[0]["d"],
                "last_date": parsed[-1]["d"],
            }
        except OSError:
            if attempt == 2:
                raise
            time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"Failed to acquire Sina contract: {symbol}")


def acquire_sources(
    output_dir: Path,
    *,
    start_year: int = 2019,
    end_year: int = 2025,
    workers: int = 6,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    symbols = _symbols(start_year, end_year)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_download, symbol, temporary / f"{symbol}.json"): symbol
                for symbol in symbols
            }
            records = [future.result() for future in as_completed(futures)]
        records.sort(key=lambda row: str(row["symbol"]))
        manifest = {
            "contract_version": RAW_CONTRACT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "source": "Sina contract-specific InnerFuturesNewService daily JSONP",
            "source_limitation": (
                "Free unofficial mirror of DCE settlements and open interest. "
                "Contract-specific histories are used; the undocumented I0 continuous "
                "series is never used."
            ),
            "requested_symbols": symbols,
            "downloaded_count": sum(row["status"] == "downloaded" for row in records),
            "files": records,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
    except BaseException:
        if temporary.exists() and temporary.parent == output_dir.parent:
            shutil.rmtree(temporary)
        raise
    return output_dir


def parse_contract(path: Path) -> tuple[list[ContractDay], int]:
    match = PAYLOAD.search(path.read_bytes())
    if match is None:
        raise ValueError(f"Malformed Sina JSONP: {path}")
    payload = json.loads(match.group(1))
    if not isinstance(payload, list):
        raise ValueError(f"Sina contract response is not a row list: {path}")
    rows = []
    nonpositive_settlement_rows = 0
    symbol = path.stem.upper()
    for value in payload:
        settlement = float(value["s"])
        open_interest = int(value["p"])
        volume = int(value["v"])
        if open_interest < 0 or volume < 0:
            raise ValueError(f"Invalid DCE daily values in {path}")
        if settlement <= 0:
            nonpositive_settlement_rows += 1
            continue
        rows.append(
            ContractDay(
                symbol=symbol,
                trade_date=date.fromisoformat(value["d"]),
                settlement=settlement,
                open_interest=open_interest,
                volume=volume,
            )
        )
    if len({row.trade_date for row in rows}) != len(rows):
        raise ValueError(f"Duplicate DCE contract dates: {path}")
    return rows, nonpositive_settlement_rows


def _market_dates(calendar_dir: Path) -> list[date]:
    paths = sorted(calendar_dir.glob("year=*/equities_daily_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parsed COTAHIST files under {calendar_dir}")
    return (
        pl.scan_parquet(paths)
        .select(pl.col("trade_date").unique().sort())
        .collect()
        .get_column("trade_date")
        .to_list()
    )


def resolve_exposures(assignments_path: Path) -> list[Exposure]:
    frame = pl.read_parquet(assignments_path).select(
        "security_id",
        "isin",
        "latest_ticker",
        "first_overlap_date",
        "last_overlap_date",
    )
    rows = []
    for ticker, role in TARGETS.items():
        matches = frame.filter(pl.col("latest_ticker") == ticker)
        if matches.height != 1:
            raise ValueError(f"Expected one accepted assignment for {ticker}")
        row = matches.row(0, named=True)
        if row["security_id"] != f"ISIN:{row['isin']}":
            raise ValueError("DCE exposure identity must be an exact accepted ISIN")
        rows.append(
            Exposure(
                security_id=row["security_id"],
                ticker=ticker,
                role=role,
                effective_from=date.fromisoformat(row["first_overlap_date"]),
                effective_to_inclusive=date.fromisoformat(row["last_overlap_date"]),
            )
        )
    return rows


def _contract_month(symbol: str) -> int:
    year = 2000 + int(symbol[1:3])
    month = int(symbol[3:5])
    return year * 12 + month


def _curve_slope(rows: Iterable[ContractDay]) -> float | None:
    liquid = [row for row in rows if row.open_interest > 0 and row.settlement > 0]
    if len(liquid) < 3:
        return None
    liquid.sort(key=lambda row: _contract_month(row.symbol))
    base = _contract_month(liquid[0].symbol)
    x = [_contract_month(row.symbol) - base for row in liquid]
    y = [math.log(row.settlement) for row in liquid]
    x_mean = statistics.fmean(x)
    y_mean = statistics.fmean(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    if denominator <= 0:
        return None
    return (
        sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y, strict=True))
        / denominator
    )


def _robust(value: float | None, history: list[float]) -> tuple[float, bool]:
    if value is None or len(history) < MIN_ROLLING_OBSERVATIONS:
        return 0.0, False
    prior = history[-ROLLING_OBSERVATIONS:]
    center = statistics.median(prior)
    mad = statistics.median(abs(item - center) for item in prior)
    scale = 1.4826 * mad
    if scale <= 1e-12:
        return 0.0, False
    z = min(max((value - center) / scale, -ROBUST_CLIP), ROBUST_CLIP)
    return z / ROBUST_CLIP, True


def build_features(
    *,
    raw_dir: Path,
    calendar_dir: Path,
    assignments_path: Path,
    output_dir: Path,
    start: date,
    end: date,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    paths = sorted(raw_dir.glob("I*.json"))
    if not paths:
        raise FileNotFoundError(f"No DCE contract snapshots under {raw_dir}")
    by_date: dict[date, dict[str, ContractDay]] = {}
    nonpositive_settlement_rows = 0
    for path in paths:
        rows, dropped = parse_contract(path)
        nonpositive_settlement_rows += dropped
        for row in rows:
            if start <= row.trade_date <= end:
                by_date.setdefault(row.trade_date, {})[row.symbol] = row
    source_dates = sorted(by_date)
    market_dates = _market_dates(calendar_dir)
    exposures = resolve_exposures(assignments_path)
    raw_history: dict[str, list[float]] = {name: [] for name in FEATURES[:4]}
    previous_selected: str | None = None
    output_rows = []
    market_states = []
    for position, source_date in enumerate(source_dates):
        current = by_date[source_date]
        previous = by_date[source_dates[position - 1]] if position else {}
        if previous:
            previous_selected = min(
                previous,
                key=lambda symbol: (-previous[symbol].open_interest, symbol),
            )
        selected = previous_selected
        one_day = None
        five_day = None
        oi_change = None
        if selected in current and selected in previous:
            one_day = math.log(
                current[selected].settlement / previous[selected].settlement
            )
            if (
                previous[selected].open_interest > 0
                and current[selected].open_interest > 0
            ):
                oi_change = math.log(
                    current[selected].open_interest / previous[selected].open_interest
                )
        if selected in current and position >= 5:
            five_days_back = by_date[source_dates[position - 5]]
            if selected in five_days_back:
                five_day = math.log(
                    current[selected].settlement / five_days_back[selected].settlement
                )
        slope = _curve_slope(current.values())
        raw_values = (one_day, five_day, oi_change, slope)
        normalized = []
        for feature, value in zip(FEATURES[:4], raw_values, strict=True):
            normalized.append(_robust(value, raw_history[feature]))
        available_position = bisect_left(market_dates, source_date)
        if available_position < len(market_dates):
            available_date = market_dates[available_position]
            for exposure in exposures:
                if (
                    not exposure.effective_from
                    <= available_date
                    <= exposure.effective_to_inclusive
                ):
                    continue
                row: dict[str, object] = {
                    "source_trade_date": source_date,
                    "available_date": available_date,
                    "security_id": exposure.security_id,
                    "selected_contract": selected or "",
                }
                for feature, (value, mask) in zip(
                    FEATURES[:4], normalized, strict=True
                ):
                    row[feature] = value
                    row[f"{feature}_mask"] = mask
                producer = exposure.role in ("producer", "producer_steel")
                steel = exposure.role in ("steel", "producer_steel")
                row[FEATURES[4]] = float(producer)
                row[f"{FEATURES[4]}_mask"] = True
                row[FEATURES[5]] = float(steel)
                row[f"{FEATURES[5]}_mask"] = True
                output_rows.append(row)
        market_states.append(
            {
                "source_trade_date": source_date.isoformat(),
                "selected_contract": selected,
                "return_1d": one_day,
                "return_5d": five_day,
                "oi_change_1d": oi_change,
                "curve_slope": slope,
            }
        )
        for feature, value in zip(FEATURES[:4], raw_values, strict=True):
            if value is not None:
                raw_history[feature].append(value)

    frame = (
        pl.DataFrame(output_rows)
        .sort("source_trade_date")
        .unique(
            subset=("available_date", "security_id"), keep="last", maintain_order=True
        )
        .sort("available_date", "security_id")
    )
    if frame.is_empty():
        raise ValueError("No DCE features mapped to the B3 model axes")
    if frame.select(
        pl.struct("available_date", "security_id").is_duplicated().any()
    ).item():
        raise ValueError("DCE features contain duplicate keys")
    for feature in FEATURES:
        if frame.filter(~pl.col(f"{feature}_mask") & (pl.col(feature) != 0)).height:
            raise ValueError("Invalid DCE features must be exactly zero")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        parquet = temporary / "dce_iron_ore.parquet"
        frame.write_parquet(parquet, compression="zstd")
        state_path = temporary / "market_states.json"
        state_path.write_text(
            json.dumps(market_states, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "contract_version": CONTRACT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "features": list(FEATURES),
            "source": str(raw_dir.resolve()),
            "source_manifest_sha256": _sha256(raw_dir / "manifest.json"),
            "source_nonpositive_settlement_rows_dropped": nonpositive_settlement_rows,
            "availability_rule": (
                "DCE trading day D closes at 15:00 Asia/Shanghai (04:00 "
                "America/Sao_Paulo); use the first B3 session on or after D"
            ),
            "roll_rule": (
                "Select the same contract by maximum prior-DCE-session open interest; "
                "returns never splice different contracts"
            ),
            "normalization": (
                "Prior-only rolling-60 median/MAD z-score, minimum 20 observations, "
                "clipped to +/-5 and divided by 5"
            ),
            "target_exposures": [asdict(exposure) for exposure in exposures],
            "output_rows": frame.height,
            "output_security_count": frame.get_column("security_id").n_unique(),
            "first_source_date": str(frame.get_column("source_trade_date").min()),
            "last_source_date": str(frame.get_column("source_trade_date").max()),
            "output_sha256": _sha256(parquet),
            "market_states_sha256": _sha256(state_path),
            "official_validation_accessed": False,
            "test_accessed": False,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False, default=str)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
    except BaseException:
        if temporary.exists() and temporary.parent == output_dir.parent:
            shutil.rmtree(temporary)
        raise
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DCE iron-ore sidecar source")
    subparsers = parser.add_subparsers(dest="command", required=True)
    acquire = subparsers.add_parser("acquire")
    acquire.add_argument("--output-dir", type=Path, required=True)
    acquire.add_argument("--start-year", type=int, default=2019)
    acquire.add_argument("--end-year", type=int, default=2025)
    acquire.add_argument("--workers", type=int, default=6)
    build = subparsers.add_parser("build")
    build.add_argument("--raw-dir", type=Path, required=True)
    build.add_argument("--calendar-dir", type=Path, required=True)
    build.add_argument("--assignments-path", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--start", type=date.fromisoformat, required=True)
    build.add_argument("--end", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    if args.command == "acquire":
        print(
            acquire_sources(
                args.output_dir,
                start_year=args.start_year,
                end_year=args.end_year,
                workers=args.workers,
            )
        )
    else:
        print(
            build_features(
                **{key: value for key, value in vars(args).items() if key != "command"}
            )
        )


if __name__ == "__main__":
    main()
