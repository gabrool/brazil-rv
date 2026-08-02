from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from zoneinfo import ZoneInfo
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import databento as db
import polars as pl
import databento_dbn as dbn

from .contract import (
    GLOBAL_CONTEXT_FAMILIES,
    GLOBAL_CONTEXT_SYMBOLS,
    GLOBAL_CONTINUOUS_ROLL_RULE,
    DECISION_GLOBAL_INDICES,
    DECISION_TIMES,
    GLOBAL_DATABENTO_VERSION,
    GLOBAL_DATASET,
    GLOBAL_PROVIDER,
    GLOBAL_QUOTE_DIRECTIONS,
    GLOBAL_SCHEMA,
    GLOBAL_SESSION_END_MINUTE,
    GLOBAL_SESSION_START_MINUTE,
    GLOBAL_SOURCE_POINTER,
    PROJECT_ROOT,
)
from .io import read_research_interval, resolve_pointer

RAW_BASE = PROJECT_ROOT / "quant-data/b3/raw/databento/global_context"
NORMALIZED_BASE = PROJECT_ROOT / "quant-data/b3/interim/global_context"
API_KEY_ENV = "DATABENTO_API_KEY"
CONTINUOUS_STYPE = "continuous"
OUTRIGHT_STYPE = "raw_symbol"
REQUEST_WARMUP_DAYS = 45
DOWNLOAD_CHUNK_DAYS = 31
PRICE_COLUMNS = ("open", "high", "low", "close")
NORMALIZED_COLUMNS = (
    "ts_event_utc",
    "bar_end_utc",
    "received_at_utc",
    "continuous_symbol",
    "global_slot",
    "family",
    "quote_direction",
    "instrument_id",
    "raw_symbol",
    "expiration_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "mapping_changed",
)


class HistoricalClient(Protocol):
    metadata: Any
    timeseries: Any


@dataclass(frozen=True)
class RequestRange:
    start: date
    end: date


@dataclass(frozen=True)
class DownloadChunk:
    continuous_symbol: str
    start: date
    end: date
    bars_path: Path
    definitions_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_pointer(pointer: Path, target: Path) -> None:
    pointer.parent.mkdir(parents=True, exist_ok=True)
    temporary = pointer.with_name(f"{pointer.name}.tmp")
    temporary.write_text(str(target.resolve()), encoding="utf-8")
    os.replace(temporary, pointer)


def require_api_key(environment: dict[str, str] | None = None) -> str:
    key = (os.environ if environment is None else environment).get(API_KEY_ENV)
    if not key:
        raise RuntimeError(f"Set {API_KEY_ENV} before contacting Databento")
    return key


def authoritative_request_range(universe_dir: Path) -> RequestRange:
    start, end = read_research_interval(universe_dir)
    return RequestRange(
        start - timedelta(days=REQUEST_WARMUP_DAYS), end + timedelta(days=1)
    )


def chunk_ranges(start: date, end: date) -> tuple[RequestRange, ...]:
    chunks: list[RequestRange] = []
    cursor = start
    while cursor < end:
        stop = min(cursor + timedelta(days=DOWNLOAD_CHUNK_DAYS), end)
        chunks.append(RequestRange(cursor, stop))
        cursor = stop
    return tuple(chunks)


def estimate_cost(
    client: HistoricalClient,
    request: RequestRange,
    symbols: Sequence[str] = GLOBAL_CONTEXT_SYMBOLS,
) -> dict[str, float]:
    costs: dict[str, float] = {}
    try:
        for symbol in symbols:
            costs[symbol] = float(
                client.metadata.get_cost(
                    dataset=GLOBAL_DATASET,
                    start=request.start,
                    end=request.end,
                    symbols=[symbol],
                    schema=GLOBAL_SCHEMA,
                    stype_in=CONTINUOUS_STYPE,
                )
            )
    except Exception:
        raise RuntimeError("Databento cost estimate failed") from None
    return costs


def _safe_symbol(symbol: str) -> str:
    return symbol.replace(".", "_")


def _download_one(
    client: HistoricalClient,
    symbol: str,
    request: RequestRange,
    raw_dir: Path,
) -> DownloadChunk:
    stem = f"{_safe_symbol(symbol)}_{request.start}_{request.end}"
    bars_path = raw_dir / "bars" / f"{stem}.dbn.zst"
    definitions_path = raw_dir / "definitions" / f"{stem}.dbn.zst"
    metadata_path = raw_dir / "requests" / f"{stem}.json"
    if bars_path.is_file() and definitions_path.is_file() and metadata_path.is_file():
        return DownloadChunk(
            symbol, request.start, request.end, bars_path, definitions_path
        )
    for path in (bars_path, definitions_path, metadata_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    bars_partial = bars_path.with_suffix(f"{bars_path.suffix}.partial")
    definitions_partial = definitions_path.with_suffix(
        f"{definitions_path.suffix}.partial"
    )
    bars_partial.unlink(missing_ok=True)
    definitions_partial.unlink(missing_ok=True)
    try:
        client.timeseries.get_range(
            dataset=GLOBAL_DATASET,
            start=request.start,
            end=request.end,
            symbols=[symbol],
            schema=GLOBAL_SCHEMA,
            stype_in=CONTINUOUS_STYPE,
            stype_out=OUTRIGHT_STYPE,
            path=bars_partial,
        )
        client.timeseries.get_range(
            dataset=GLOBAL_DATASET,
            start=request.start,
            end=request.end,
            symbols=[symbol],
            schema="definition",
            stype_in=CONTINUOUS_STYPE,
            stype_out=OUTRIGHT_STYPE,
            path=definitions_partial,
        )
    except Exception:
        bars_partial.unlink(missing_ok=True)
        definitions_partial.unlink(missing_ok=True)
        raise RuntimeError("Databento historical download failed") from None
    if not bars_partial.is_file() or not definitions_partial.is_file():
        raise RuntimeError("Databento did not produce both requested DBN files")
    os.replace(bars_partial, bars_path)
    os.replace(definitions_partial, definitions_path)
    _atomic_json(
        metadata_path,
        {
            "provider": GLOBAL_PROVIDER,
            "dataset": GLOBAL_DATASET,
            "schema": GLOBAL_SCHEMA,
            "continuous_symbol": symbol,
            "stype_in": CONTINUOUS_STYPE,
            "stype_out": OUTRIGHT_STYPE,
            "start": str(request.start),
            "end": str(request.end),
            "bars_sha256": _sha256(bars_path),
            "definitions_sha256": _sha256(definitions_path),
        },
    )
    return DownloadChunk(
        symbol, request.start, request.end, bars_path, definitions_path
    )


def download_history(
    client: HistoricalClient,
    request: RequestRange,
    raw_dir: Path,
    *,
    confirmed_paid_download: bool,
    symbols: Sequence[str] = GLOBAL_CONTEXT_SYMBOLS,
) -> tuple[DownloadChunk, ...]:
    costs = estimate_cost(client, request, symbols)
    print(json.dumps({"estimated_cost_usd": costs, "total_usd": sum(costs.values())}))
    if not confirmed_paid_download:
        raise RuntimeError(
            "Re-run with --confirm-paid-download after reviewing the estimate"
        )
    chunks = tuple(
        _download_one(client, symbol, chunk, raw_dir)
        for symbol in symbols
        for chunk in chunk_ranges(request.start, request.end)
    )
    _atomic_json(
        raw_dir / "manifest.json",
        {
            "provider": GLOBAL_PROVIDER,
            "dataset": GLOBAL_DATASET,
            "schema": GLOBAL_SCHEMA,
            "databento_version": GLOBAL_DATABENTO_VERSION,
            "symbols": list(symbols),
            "requested_start": str(request.start),
            "requested_end": str(request.end),
            "continuous_roll_rule": GLOBAL_CONTINUOUS_ROLL_RULE,
            "chunks": [
                {
                    "continuous_symbol": chunk.continuous_symbol,
                    "start": str(chunk.start),
                    "end": str(chunk.end),
                    "bars_path": str(chunk.bars_path),
                    "bars_sha256": _sha256(chunk.bars_path),
                    "definitions_path": str(chunk.definitions_path),
                    "definitions_sha256": _sha256(chunk.definitions_path),
                }
                for chunk in chunks
            ],
        },
    )
    return chunks


def _dbn_frame(path: Path) -> pl.DataFrame:
    frame = db.DBNStore.from_file(path).to_df(map_symbols=True).reset_index()
    return pl.from_pandas(frame)


def _timestamp_column(frame: pl.DataFrame) -> str:
    for name in ("ts_event_utc", "ts_event", "index"):
        if name in frame.columns:
            return name
    raise ValueError("Databento frame has no event timestamp")


def normalize_bars(
    frame: pl.DataFrame,
    continuous_symbol: str,
    definitions: pl.DataFrame | None = None,
) -> pl.DataFrame:
    if (
        continuous_symbol not in GLOBAL_CONTEXT_SYMBOLS
        and continuous_symbol != "6L.v.0"
    ):
        raise ValueError(f"Unsupported continuous symbol: {continuous_symbol}")
    timestamp = _timestamp_column(frame)
    raw_symbol = "raw_symbol" if "raw_symbol" in frame.columns else "symbol"
    required = {timestamp, raw_symbol, "instrument_id", *PRICE_COLUMNS, "volume"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Databento frame is missing columns: {missing}")
    expiration = None
    if definitions is not None and not definitions.is_empty():
        definition_symbol = (
            "raw_symbol" if "raw_symbol" in definitions.columns else "symbol"
        )
        if (
            definition_symbol in definitions.columns
            and "expiration" in definitions.columns
        ):
            expiration = definitions.select(
                pl.col("instrument_id").cast(pl.UInt32),
                pl.col(definition_symbol).cast(pl.String).alias("raw_symbol"),
                pl.col("expiration")
                .cast(pl.Datetime("ns", "UTC"), strict=False)
                .alias("expiration_utc"),
            ).unique("instrument_id", keep="last")
    slot = (
        GLOBAL_CONTEXT_SYMBOLS.index(continuous_symbol)
        if continuous_symbol in GLOBAL_CONTEXT_SYMBOLS
        else -1
    )
    family = GLOBAL_CONTEXT_FAMILIES[slot] if slot >= 0 else "CANDIDATE_FX_6L"
    quote = GLOBAL_QUOTE_DIRECTIONS[slot] if slot >= 0 else "BRL_PER_USD"
    normalized = (
        frame.select(
            pl.col(timestamp).cast(pl.Datetime("ns", "UTC")).alias("ts_event_utc"),
            pl.col("instrument_id").cast(pl.UInt32),
            pl.col(raw_symbol).cast(pl.String).alias("raw_symbol"),
            *(pl.col(name).cast(pl.Float64) for name in PRICE_COLUMNS),
            pl.col("volume").cast(pl.Float64),
            (
                pl.col("received_at_utc").cast(pl.Datetime("ns", "UTC"))
                if "received_at_utc" in frame.columns
                else pl.lit(None, dtype=pl.Datetime("ns", "UTC"))
            ).alias("received_at_utc"),
        )
        .sort("ts_event_utc")
        .with_columns(
            (pl.col("ts_event_utc") + pl.duration(minutes=1))
            .cast(pl.Datetime("ns", "UTC"))
            .alias("bar_end_utc"),
            pl.lit(continuous_symbol).alias("continuous_symbol"),
            pl.lit(slot, dtype=pl.Int8).alias("global_slot"),
            pl.lit(family).alias("family"),
            pl.lit(quote).alias("quote_direction"),
            pl.col("raw_symbol")
            .ne(pl.col("raw_symbol").shift(1))
            .fill_null(False)
            .alias("mapping_changed"),
        )
    )
    if expiration is None:
        normalized = normalized.with_columns(
            pl.lit(None, dtype=pl.Datetime("ns", "UTC")).alias("expiration_utc")
        )
    else:
        normalized = normalized.join(
            expiration, on=["instrument_id", "raw_symbol"], how="left"
        )
    normalized = normalized.select(NORMALIZED_COLUMNS)
    validate_normalized_bars(normalized, expected_symbol=continuous_symbol)
    return normalized


def _with_mapping_changes(
    frame: pl.DataFrame, previous_raw_symbol: str | None = None
) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    raw_symbols = frame["raw_symbol"].to_list()
    changes = [False] * len(raw_symbols)
    changes[0] = (
        previous_raw_symbol is not None and raw_symbols[0] != previous_raw_symbol
    )
    for index in range(1, len(raw_symbols)):
        changes[index] = raw_symbols[index] != raw_symbols[index - 1]
    return frame.with_columns(pl.Series("mapping_changed", changes, dtype=pl.Boolean))


def validate_normalized_bars(
    frame: pl.DataFrame, *, expected_symbol: str | None = None
) -> None:
    if tuple(frame.columns) != NORMALIZED_COLUMNS:
        raise ValueError("Normalized global schema does not match the source contract")
    if expected_symbol is not None and set(frame["continuous_symbol"]) != {
        expected_symbol
    }:
        raise ValueError("Normalized rows do not match the requested continuous symbol")
    invalid = frame.filter(
        pl.any_horizontal(
            *(pl.col(name).is_null() for name in (*PRICE_COLUMNS, "volume"))
        )
        | pl.any_horizontal(*(~pl.col(name).is_finite() for name in PRICE_COLUMNS))
        | pl.any_horizontal(*(pl.col(name) <= 0 for name in PRICE_COLUMNS))
        | (pl.col("high") < pl.max_horizontal("open", "close"))
        | (pl.col("low") > pl.min_horizontal("open", "close"))
        | ~pl.col("volume").is_finite()
        | (pl.col("volume") < 0)
        | (pl.col("bar_end_utc") != pl.col("ts_event_utc") + pl.duration(minutes=1))
        | pl.col("instrument_id").is_null()
        | pl.col("raw_symbol").is_null()
        | (pl.col("raw_symbol").str.len_chars() == 0)
    )
    if invalid.height:
        raise ValueError("Malformed OHLCV row in normalized global source")
    if frame.select("continuous_symbol", "ts_event_utc").is_duplicated().any():
        raise ValueError("Duplicate continuous-symbol timestamp in normalized source")
    if not frame["ts_event_utc"].is_sorted():
        raise ValueError("Normalized global timestamps are out of order")

    if expected_symbol in GLOBAL_CONTEXT_SYMBOLS:
        expected_slot = GLOBAL_CONTEXT_SYMBOLS.index(expected_symbol)
        if set(frame["global_slot"]) != {expected_slot}:
            raise ValueError(
                "Normalized rows do not resolve to the expected global slot"
            )


def _chunk_metadata(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_download(raw_dir: Path, *, created_at: datetime | None = None) -> Path:
    created_at = datetime.now(UTC) if created_at is None else created_at
    request_files = sorted((raw_dir / "requests").glob("*.json"))
    if not request_files:
        raise FileNotFoundError(
            f"No completed Databento request chunks under {raw_dir}"
        )
    output_dir = NORMALIZED_BASE / f"global_context_{created_at:%Y%m%dT%H%M%S%fZ}"
    partial = output_dir.with_name(f"{output_dir.name}.partial")
    if output_dir.exists() or partial.exists():
        raise FileExistsError(f"Global normalized output already exists: {output_dir}")
    (partial / "bars").mkdir(parents=True)
    source_hashes: dict[str, str] = {}
    normalized_hashes: dict[str, str] = {}
    row_count = 0
    symbols_seen: set[str] = set()
    previous_raw_symbols: dict[str, str] = {}
    for request_path in request_files:
        metadata = _chunk_metadata(request_path)
        symbol = str(metadata["continuous_symbol"])
        bars_path = raw_dir / "bars" / f"{request_path.stem}.dbn.zst"
        definitions_path = raw_dir / "definitions" / f"{request_path.stem}.dbn.zst"
        if _sha256(bars_path) != metadata["bars_sha256"]:
            raise ValueError(f"Source hash mismatch: {bars_path}")
        if _sha256(definitions_path) != metadata["definitions_sha256"]:
            raise ValueError(f"Definition hash mismatch: {definitions_path}")
        normalized = normalize_bars(
            _dbn_frame(bars_path), symbol, _dbn_frame(definitions_path)
        )
        normalized = _with_mapping_changes(normalized, previous_raw_symbols.get(symbol))
        previous_raw_symbols[symbol] = str(normalized.item(-1, "raw_symbol"))
        partition = (
            partial / "bars" / f"slot={int(normalized.item(0, 'global_slot')):02d}"
        )
        partition.mkdir(parents=True, exist_ok=True)
        target = partition / f"{request_path.stem}.parquet"
        normalized.write_parquet(target, compression="zstd", statistics=True)
        source_hashes[str(bars_path)] = _sha256(bars_path)
        source_hashes[str(definitions_path)] = _sha256(definitions_path)
        normalized_hashes[str(target.relative_to(partial))] = _sha256(target)
        row_count += normalized.height
        symbols_seen.add(symbol)
    if symbols_seen != set(GLOBAL_CONTEXT_SYMBOLS):
        raise ValueError(
            "Completed source chunks do not contain the fixed global universe"
        )
    scan = pl.scan_parquet(partial / "bars/**/*.parquet", glob=True)
    summary = (
        scan.select(
            pl.col("ts_event_utc").min().alias("actual_start"),
            pl.col("bar_end_utc").max().alias("actual_end"),
            pl.len().alias("rows"),
        )
        .collect()
        .row(0, named=True)
    )
    if int(summary["rows"]) != row_count:
        raise ValueError("Normalized row count changed during store validation")
    raw_manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "status": "complete",
        "provider": GLOBAL_PROVIDER,
        "dataset": GLOBAL_DATASET,
        "schema": GLOBAL_SCHEMA,
        "databento_version": GLOBAL_DATABENTO_VERSION,
        "symbols": list(GLOBAL_CONTEXT_SYMBOLS),
        "families": list(GLOBAL_CONTEXT_FAMILIES),
        "quote_directions": list(GLOBAL_QUOTE_DIRECTIONS),
        "continuous_roll_rule": GLOBAL_CONTINUOUS_ROLL_RULE,
        "requested_start": raw_manifest["requested_start"],
        "requested_end": raw_manifest["requested_end"],
        "actual_start_utc": str(summary["actual_start"]),
        "actual_end_utc": str(summary["actual_end"]),
        "row_count": row_count,
        "normalized_columns": list(NORMALIZED_COLUMNS),
        "source_hashes": source_hashes,
        "normalized_hashes": normalized_hashes,
        "created_at_utc": created_at.isoformat(),
    }
    _atomic_json(partial / "manifest.json", manifest)
    os.replace(partial, output_dir)
    _atomic_pointer(GLOBAL_SOURCE_POINTER, output_dir)
    return output_dir


def load_global_symbol(source_dir: Path, symbol: str) -> pl.DataFrame:
    slot = GLOBAL_CONTEXT_SYMBOLS.index(symbol)
    files = sorted((source_dir / "bars" / f"slot={slot:02d}").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No normalized global bars for {symbol}")
    frame = pl.read_parquet(files).sort("ts_event_utc")
    frame = _with_mapping_changes(frame)
    validate_normalized_bars(frame, expected_symbol=symbol)
    return frame


def _atomic_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    frame.write_parquet(temporary, compression="zstd", statistics=True)
    os.replace(temporary, path)


def _write_shadow_manifest(output_dir: Path) -> None:
    files = sorted((output_dir / "bars").rglob("*.parquet"))
    if not files:
        return
    summary = (
        pl.scan_parquet([str(path) for path in files])
        .select(
            pl.col("ts_event_utc").min().alias("actual_start"),
            pl.col("bar_end_utc").max().alias("actual_end"),
            pl.len().alias("rows"),
        )
        .collect()
        .row(0, named=True)
    )
    subscription_path = output_dir / "shadow_subscription.json"
    subscription = (
        json.loads(subscription_path.read_text(encoding="utf-8"))
        if subscription_path.is_file()
        else {}
    )
    existing_path = output_dir / "manifest.json"
    existing = (
        json.loads(existing_path.read_text(encoding="utf-8"))
        if existing_path.is_file()
        else {}
    )
    actual_start = summary["actual_start"]
    actual_end = summary["actual_end"]
    _atomic_json(
        existing_path,
        {
            "status": "active",
            "mode": "shadow",
            "provider": GLOBAL_PROVIDER,
            "dataset": GLOBAL_DATASET,
            "schema": GLOBAL_SCHEMA,
            "databento_version": GLOBAL_DATABENTO_VERSION,
            "symbols": list(GLOBAL_CONTEXT_SYMBOLS),
            "families": list(GLOBAL_CONTEXT_FAMILIES),
            "quote_directions": list(GLOBAL_QUOTE_DIRECTIONS),
            "continuous_roll_rule": GLOBAL_CONTINUOUS_ROLL_RULE,
            "requested_start": str(actual_start.date()),
            "requested_end": str(actual_end.date() + timedelta(days=1)),
            "actual_start_utc": str(actual_start),
            "actual_end_utc": str(actual_end),
            "row_count": int(summary["rows"]),
            "normalized_columns": list(NORMALIZED_COLUMNS),
            "source_hashes": {},
            "normalized_hashes": {
                str(path.relative_to(output_dir)): _sha256(path) for path in files
            },
            "subscription": subscription,
            "created_at_utc": existing.get(
                "created_at_utc", datetime.now(UTC).isoformat()
            ),
            "updated_at_utc": datetime.now(UTC).isoformat(),
        },
    )


def write_shadow_daily_chunks(
    frame: pl.DataFrame,
    continuous_symbol: str,
    output_dir: Path,
) -> tuple[Path, ...]:
    normalized = normalize_bars(frame, continuous_symbol)
    normalized = normalized.with_columns(
        pl.col("ts_event_utc").dt.date().alias("_utc_date")
    )
    written: list[Path] = []
    slot = GLOBAL_CONTEXT_SYMBOLS.index(continuous_symbol)
    partition = output_dir / "bars" / f"slot={slot:02d}"
    for utc_date in normalized["_utc_date"].unique().sort().to_list():
        target = partition / f"date={utc_date}.parquet"
        daily = normalized.filter(pl.col("_utc_date") == utc_date).drop("_utc_date")
        if target.is_file():
            daily = pl.concat(
                [pl.read_parquet(target), daily], how="vertical_relaxed"
            ).unique(["continuous_symbol", "ts_event_utc"], keep="last")
        daily = daily.sort("ts_event_utc")
        prior_files = sorted(
            path for path in partition.glob("date=*.parquet") if path.name < target.name
        )
        previous_raw_symbol = (
            str(pl.read_parquet(prior_files[-1]).item(-1, "raw_symbol"))
            if prior_files
            else None
        )
        daily = _with_mapping_changes(daily, previous_raw_symbol)
        validate_normalized_bars(daily, expected_symbol=continuous_symbol)
        _atomic_parquet(daily, target)
        written.append(target)
    _write_shadow_manifest(output_dir)
    return tuple(written)


def _record_text(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


class ShadowCollector:
    def __init__(self, output_dir: Path, flush_records: int = 512) -> None:
        self.output_dir = output_dir
        self.flush_records = flush_records
        self.mappings: dict[int, tuple[str, str]] = {}
        self.rows: dict[str, list[dict[str, object]]] = {
            symbol: [] for symbol in GLOBAL_CONTEXT_SYMBOLS
        }

    def __call__(self, record: object) -> None:
        if isinstance(record, dbn.SymbolMappingMsg):
            continuous_symbol = _record_text(record.stype_in_symbol)
            if continuous_symbol in GLOBAL_CONTEXT_SYMBOLS:
                self.mappings[int(record.instrument_id)] = (
                    continuous_symbol,
                    _record_text(record.stype_out_symbol),
                )
            return
        if not isinstance(record, dbn.OHLCVMsg):
            return
        instrument_id = int(record.instrument_id)
        try:
            continuous_symbol, raw_symbol = self.mappings[instrument_id]
        except KeyError:
            raise ValueError(
                f"Live OHLCV record has no continuous-symbol mapping: {instrument_id}"
            ) from None
        ts_event = datetime.fromtimestamp(int(record.ts_event) / 1_000_000_000, UTC)
        received_at = datetime.now(UTC)
        if received_at < ts_event + timedelta(minutes=1):
            raise ValueError("Databento delivered an incomplete one-minute bar")
        self.rows[continuous_symbol].append(
            {
                "ts_event_utc": ts_event,
                "received_at_utc": received_at,
                "instrument_id": instrument_id,
                "symbol": raw_symbol,
                "open": float(record.pretty_open),
                "high": float(record.pretty_high),
                "low": float(record.pretty_low),
                "close": float(record.pretty_close),
                "volume": float(record.volume),
            }
        )
        if sum(len(rows) for rows in self.rows.values()) >= self.flush_records:
            self.flush()

    def flush(self) -> None:
        for symbol, rows in self.rows.items():
            if rows:
                write_shadow_daily_chunks(pl.DataFrame(rows), symbol, self.output_dir)
                rows.clear()


def run_shadow_collection(output_dir: Path, flush_records: int = 512) -> None:
    client = db.Live(key=require_api_key(), ts_out=True)
    collector = ShadowCollector(output_dir, flush_records)
    client.add_callback(collector)
    subscription_id = client.subscribe(
        dataset=GLOBAL_DATASET,
        schema=GLOBAL_SCHEMA,
        symbols=list(GLOBAL_CONTEXT_SYMBOLS),
        stype_in=CONTINUOUS_STYPE,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        output_dir / "shadow_subscription.json",
        {
            "subscription_id": subscription_id,
            "provider": GLOBAL_PROVIDER,
            "dataset": GLOBAL_DATASET,
            "schema": GLOBAL_SCHEMA,
            "symbols": list(GLOBAL_CONTEXT_SYMBOLS),
            "stype_in": CONTINUOUS_STYPE,
            "started_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    client.start()
    try:
        client.block_for_close()
    except KeyboardInterrupt:
        client.stop()
    finally:
        collector.flush()


def audit_candidate(frame: pl.DataFrame, output_path: Path) -> dict[str, object]:
    normalized = normalize_bars(frame, "6L.v.0")
    local = normalized.with_columns(
        pl.col("ts_event_utc").dt.convert_time_zone("America/Sao_Paulo").alias("b3_ts")
    ).with_columns(
        pl.col("b3_ts").dt.date().alias("b3_date"),
        (
            pl.col("b3_ts").dt.hour().cast(pl.Int16) * 60
            + pl.col("b3_ts").dt.minute().cast(pl.Int16)
        ).alias("b3_minute"),
    )
    grid = local.filter(
        pl.col("b3_minute").is_between(
            GLOBAL_SESSION_START_MINUTE, GLOBAL_SESSION_END_MINUTE - 1
        )
    )
    by_date = grid.group_by("b3_date").agg(
        pl.len().alias("observed_minutes"),
        pl.col("volume").sum().alias("volume"),
        pl.col("mapping_changed").sum().alias("rolls"),
        pl.col("b3_ts").max().alias("last_observed"),
    )

    decision_rows: list[dict[str, object]] = []
    b3_timezone = ZoneInfo("America/Sao_Paulo")
    for trade_date in sorted(grid["b3_date"].unique().to_list()):
        on_date = grid.filter(pl.col("b3_date") == trade_date)
        for decision_idx, (cutoff, decision_time) in enumerate(
            zip(DECISION_GLOBAL_INDICES, DECISION_TIMES, strict=True)
        ):
            end_minute = GLOBAL_SESSION_START_MINUTE + cutoff
            prefix = on_date.filter(
                pl.col("b3_minute").is_between(end_minute - 345, end_minute - 1)
            )
            decision = datetime.combine(
                trade_date, decision_time, b3_timezone
            ).astimezone(UTC)
            last_bar_end = (
                prefix["bar_end_utc"].max() if not prefix.is_empty() else None
            )
            observed_minutes = prefix.height
            decision_rows.append(
                {
                    "b3_date": str(trade_date),
                    "decision_idx": decision_idx,
                    "decision_time_utc": decision.isoformat(),
                    "observed_minutes": observed_minutes,
                    "missing_minutes": 345 - observed_minutes,
                    "observed_fraction": observed_minutes / 345,
                    "staleness_minutes": (
                        (decision - last_bar_end).total_seconds() / 60.0
                        if last_bar_end is not None
                        else None
                    ),
                    "volume": (
                        float(prefix["volume"].sum()) if observed_minutes else 0.0
                    ),
                    "roll_count": (
                        int(prefix["mapping_changed"].sum()) if observed_minutes else 0
                    ),
                }
            )
    report = {
        "continuous_symbol": "6L.v.0",
        "date_count": by_date.height,
        "observed_fraction": (
            float(by_date["observed_minutes"].sum())
            / (
                by_date.height
                * (GLOBAL_SESSION_END_MINUTE - GLOBAL_SESSION_START_MINUTE)
            )
            if by_date.height
            else 0.0
        ),
        "total_volume": float(by_date["volume"].sum()) if by_date.height else 0.0,
        "roll_count": int(by_date["rolls"].sum()) if by_date.height else 0,
        "decision_coverage": decision_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_path, report)
    return report


def _historical_client() -> db.Historical:
    return db.Historical(require_api_key())


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    estimate = subparsers.add_parser("estimate")
    estimate.add_argument("--universe-pointer", type=Path, required=True)
    download = subparsers.add_parser("download")
    download.add_argument("--universe-pointer", type=Path, required=True)
    download.add_argument("--raw-dir", type=Path, default=RAW_BASE)
    download.add_argument("--confirm-paid-download", action="store_true")
    normalize = subparsers.add_parser("normalize")
    normalize.add_argument("--raw-dir", type=Path, default=RAW_BASE)
    shadow = subparsers.add_parser("shadow")
    shadow.add_argument("--output-dir", type=Path, required=True)
    shadow.add_argument("--flush-records", type=int, default=512)
    candidate = subparsers.add_parser("audit-6l")
    candidate.add_argument("--input-parquet", type=Path, required=True)
    candidate.add_argument("--output", type=Path, required=True)
    return parser.parse_args(arguments)


def main() -> None:
    args = parse_args()
    if args.command in {"estimate", "download"}:
        request = authoritative_request_range(resolve_pointer(args.universe_pointer))
        client = _historical_client()
        if args.command == "estimate":
            costs = estimate_cost(client, request)
            print(
                json.dumps(
                    {"estimated_cost_usd": costs, "total_usd": sum(costs.values())}
                )
            )
        else:
            download_history(
                client,
                request,
                args.raw_dir,
                confirmed_paid_download=args.confirm_paid_download,
            )
    elif args.command == "normalize":
        print(normalize_download(args.raw_dir))
    elif args.command == "shadow":
        run_shadow_collection(args.output_dir, args.flush_records)
    else:
        report = audit_candidate(pl.read_parquet(args.input_parquet), args.output)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
