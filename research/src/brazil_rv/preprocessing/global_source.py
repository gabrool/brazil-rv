from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
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
INSTRUMENT_ID_STYPE = "instrument_id"
REQUEST_WARMUP_DAYS = 45
DOWNLOAD_CHUNK_DAYS = 31
HISTORICAL_SCHEMAS = (GLOBAL_SCHEMA, "definition")
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
class HistoricalRequest:
    continuous_symbol: str
    schema: str
    start: date
    end: date
    data_path: Path
    descriptor_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_temporary_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp")


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = _json_temporary_path(path)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
    requests: Sequence[HistoricalRequest],
) -> dict[str, object]:
    estimate_plan = metadata_estimate_plan(requests)
    by_schema = {
        schema: {"request_count": 0, "estimated_cost_usd": 0.0}
        for schema in HISTORICAL_SCHEMAS
    }
    for request in requests:
        by_schema[request.schema]["request_count"] += 1
    try:
        for number, request in enumerate(estimate_plan, start=1):
            print(
                f"Estimating metadata {number}/{len(estimate_plan)}: "
                f"{request.continuous_symbol} {request.schema} "
                f"[{request.start}, {request.end})",
                file=sys.stderr,
                flush=True,
            )
            cost = float(client.metadata.get_cost(**_request_kwargs(request)))
            by_schema[request.schema]["estimated_cost_usd"] += cost
    except Exception:
        raise RuntimeError("Databento cost estimate failed") from None
    return {
        "remaining_request_count": len(requests),
        "remaining_download_request_count": len(requests),
        "metadata_estimate_group_count": len(estimate_plan),
        "by_schema": by_schema,
        "total_usd": sum(
            float(summary["estimated_cost_usd"]) for summary in by_schema.values()
        ),
    }


def _safe_symbol(symbol: str) -> str:
    return symbol.replace(".", "_")


def request_plan(
    request: RequestRange,
    raw_dir: Path,
    symbols: Sequence[str] = GLOBAL_CONTEXT_SYMBOLS,
) -> tuple[HistoricalRequest, ...]:
    planned: list[HistoricalRequest] = []
    for symbol in symbols:
        for chunk in chunk_ranges(request.start, request.end):
            for schema in HISTORICAL_SCHEMAS:
                schema_name = schema.replace("-", "_")
                stem = f"{_safe_symbol(symbol)}_{chunk.start}_{chunk.end}_{schema_name}"
                directory = "bars" if schema == GLOBAL_SCHEMA else "definitions"
                planned.append(
                    HistoricalRequest(
                        continuous_symbol=symbol,
                        schema=schema,
                        start=chunk.start,
                        end=chunk.end,
                        data_path=raw_dir / directory / f"{stem}.dbn.zst",
                        descriptor_path=raw_dir / "requests" / f"{stem}.json",
                    )
                )
    return tuple(planned)


def metadata_estimate_plan(
    requests: Sequence[HistoricalRequest],
) -> tuple[HistoricalRequest, ...]:
    groups: list[HistoricalRequest] = []
    latest_by_contract: dict[tuple[str, str, str, str], int] = {}
    for request in requests:
        contract = (
            GLOBAL_DATASET,
            CONTINUOUS_STYPE,
            request.continuous_symbol,
            request.schema,
        )
        previous = latest_by_contract.get(contract)
        if previous is not None and groups[previous].end == request.start:
            groups[previous] = replace(groups[previous], end=request.end)
        else:
            latest_by_contract[contract] = len(groups)
            groups.append(request)
    return tuple(groups)


def _request_contract(request: HistoricalRequest) -> dict[str, object]:
    return {
        "request_id": request.descriptor_path.stem,
        "provider": GLOBAL_PROVIDER,
        "dataset": GLOBAL_DATASET,
        "schema": request.schema,
        "continuous_symbol": request.continuous_symbol,
        "stype_in": CONTINUOUS_STYPE,
        "stype_out": INSTRUMENT_ID_STYPE,
        "start": str(request.start),
        "end": str(request.end),
        "data_path": f"{request.data_path.parent.name}/{request.data_path.name}",
    }


def _request_descriptor(request: HistoricalRequest) -> dict[str, object]:
    descriptor = _request_contract(request)
    descriptor.update(
        {
            "data_sha256": _sha256(request.data_path),
            "data_size_bytes": request.data_path.stat().st_size,
        }
    )
    return descriptor


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _completed_request(request: HistoricalRequest) -> bool:
    data_exists = request.data_path.is_file()
    descriptor_exists = request.descriptor_path.is_file()
    if data_exists != descriptor_exists:
        raise ValueError(
            f"Incomplete Databento request artifacts: {request.descriptor_path.stem}"
        )
    if not data_exists:
        return False
    descriptor = _read_json(request.descriptor_path)
    expected = _request_contract(request)
    if set(descriptor) != {*expected, "data_sha256", "data_size_bytes"} or any(
        descriptor[name] != value for name, value in expected.items()
    ):
        raise ValueError(
            f"Databento request descriptor mismatch: {request.descriptor_path}"
        )
    if descriptor["data_sha256"] != _sha256(request.data_path):
        raise ValueError(f"Databento request hash mismatch: {request.data_path}")
    if descriptor["data_size_bytes"] != request.data_path.stat().st_size:
        raise ValueError(f"Databento request size mismatch: {request.data_path}")
    return True


def _validate_planned_files(
    raw_dir: Path,
    plan: Sequence[HistoricalRequest],
    *,
    complete: bool,
) -> None:
    expected_descriptors = {request.descriptor_path.resolve() for request in plan}
    expected_data = {request.data_path.resolve() for request in plan}
    actual_descriptors = {
        path.resolve() for path in (raw_dir / "requests").rglob("*") if path.is_file()
    }
    actual_data = {
        path.resolve()
        for directory in ("bars", "definitions")
        for path in (raw_dir / directory).rglob("*")
        if path.is_file()
    }
    if actual_descriptors - expected_descriptors or actual_data - expected_data:
        raise ValueError("Unexpected Databento request artifacts exist")
    if complete and (
        actual_descriptors != expected_descriptors or actual_data != expected_data
    ):
        raise ValueError("Expected Databento request artifacts are missing")


def _request_partial_path(request: HistoricalRequest) -> Path:
    return request.data_path.with_suffix(f"{request.data_path.suffix}.partial")


def _recover_request_states(raw_dir: Path, plan: Sequence[HistoricalRequest]) -> None:
    partials = tuple(_request_partial_path(request) for request in plan)
    recoverable_temporaries = (
        *(_json_temporary_path(request.descriptor_path) for request in plan),
        _json_temporary_path(raw_dir / "manifest.json"),
    )
    expected_partials = {path.absolute() for path in partials}
    actual_partials = {path.absolute() for path in raw_dir.rglob("*.partial")}
    if actual_partials - expected_partials:
        raise ValueError("Unexpected Databento partial artifact exists")
    if any(path.exists() and not path.is_file() for path in partials):
        raise ValueError("Malformed Databento partial artifact exists")

    expected_temporaries = {path.absolute() for path in recoverable_temporaries}
    actual_temporaries = {path.absolute() for path in raw_dir.rglob("*.tmp")}
    if actual_temporaries - expected_temporaries:
        raise ValueError("Unexpected Databento temporary artifact exists")
    if any(path.exists() and not path.is_file() for path in recoverable_temporaries):
        raise ValueError("Malformed Databento temporary artifact exists")

    for path in (*partials, *recoverable_temporaries):
        path.unlink(missing_ok=True)
    _validate_planned_files(raw_dir, plan, complete=False)

    for request in plan:
        data_exists = request.data_path.is_file()
        descriptor_exists = request.descriptor_path.is_file()
        if descriptor_exists and not data_exists:
            raise ValueError(
                f"Incomplete Databento request artifacts: "
                f"{request.descriptor_path.stem}"
            )
        if data_exists and not descriptor_exists:
            _validate_dbn_contract(request)
            _atomic_json(request.descriptor_path, _request_descriptor(request))


def remaining_requests(
    raw_dir: Path, plan: Sequence[HistoricalRequest]
) -> tuple[HistoricalRequest, ...]:
    _recover_request_states(raw_dir, plan)
    return tuple(request for request in plan if not _completed_request(request))


def _request_kwargs(request: HistoricalRequest) -> dict[str, object]:
    return {
        "dataset": GLOBAL_DATASET,
        "start": request.start,
        "end": request.end,
        "symbols": [request.continuous_symbol],
        "schema": request.schema,
        "stype_in": CONTINUOUS_STYPE,
    }


def _download_request(
    client: HistoricalClient,
    request: HistoricalRequest,
) -> None:
    for path in (request.data_path, request.descriptor_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    partial = _request_partial_path(request)
    partial.unlink(missing_ok=True)
    try:
        client.timeseries.get_range(
            **_request_kwargs(request),
            stype_out=INSTRUMENT_ID_STYPE,
            path=partial,
        )
    except Exception:
        partial.unlink(missing_ok=True)
        raise RuntimeError("Databento historical download failed") from None
    if not partial.is_file():
        raise RuntimeError("Databento did not produce the requested DBN file")
    os.replace(partial, request.data_path)
    _atomic_json(request.descriptor_path, _request_descriptor(request))


@contextmanager
def _acquisition_lock(raw_dir: Path) -> Iterator[None]:
    raw_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = raw_dir.parent / f".{raw_dir.name}.acquisition.lock"
    with lock_path.open("a+b") as lock:
        lock.seek(0, os.SEEK_END)
        if lock.tell() == 0:
            lock.write(b"\0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("Databento acquisition is already active") from None
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def download_history(
    client: HistoricalClient,
    request: RequestRange,
    raw_dir: Path,
    *,
    confirmed_paid_download: bool,
    symbols: Sequence[str] = GLOBAL_CONTEXT_SYMBOLS,
) -> tuple[HistoricalRequest, ...]:
    with _acquisition_lock(raw_dir):
        plan = request_plan(request, raw_dir, symbols)
        remaining = remaining_requests(raw_dir, plan)
        estimate = estimate_cost(client, remaining)
        print(json.dumps(estimate))
        if remaining and not confirmed_paid_download:
            raise RuntimeError(
                "Re-run with --confirm-paid-download after reviewing the estimate"
            )
        for planned_request in remaining:
            _download_request(client, planned_request)
        _validate_planned_files(raw_dir, plan, complete=True)
        descriptors = []
        for planned_request in plan:
            if not _completed_request(planned_request):
                raise ValueError("Completed Databento request is missing")
            descriptors.append(_read_json(planned_request.descriptor_path))
        _atomic_json(
            raw_dir / "manifest.json",
            {
                "status": "complete",
                "provider": GLOBAL_PROVIDER,
                "dataset": GLOBAL_DATASET,
                "schemas": list(HISTORICAL_SCHEMAS),
                "databento_version": GLOBAL_DATABENTO_VERSION,
                "symbols": list(symbols),
                "stype_in": CONTINUOUS_STYPE,
                "stype_out": INSTRUMENT_ID_STYPE,
                "requested_start": str(request.start),
                "requested_end": str(request.end),
                "continuous_roll_rule": GLOBAL_CONTINUOUS_ROLL_RULE,
                "requests": descriptors,
            },
        )
        return plan


def _date_ns(value: date) -> int:
    return int(datetime.combine(value, datetime.min.time(), UTC).timestamp() * 1e9)


def _dbn_metadata(path: Path) -> dict[str, object]:
    try:
        metadata = db.DBNStore.from_file(path).metadata
    except Exception:
        raise ValueError(f"Invalid Databento DBN file: {path}") from None
    return {
        "dataset": metadata.dataset,
        "schema": str(metadata.schema),
        "stype_in": str(metadata.stype_in),
        "stype_out": str(metadata.stype_out),
        "start": metadata.start,
        "end": metadata.end,
        "symbols": list(metadata.symbols),
        "partial": list(metadata.partial),
        "not_found": list(metadata.not_found),
        "mappings": metadata.mappings,
    }


def _mapping_by_date(
    metadata: dict[str, object],
    request: HistoricalRequest,
) -> dict[date, int]:
    mappings = metadata["mappings"]
    if not isinstance(mappings, dict) or set(mappings) != {request.continuous_symbol}:
        raise ValueError(f"DBN continuous mapping mismatch: {request.data_path}")
    entries = mappings[request.continuous_symbol]
    if not isinstance(entries, list):
        raise ValueError(f"Malformed DBN mapping metadata: {request.data_path}")
    output: dict[date, int] = {}
    cursor = request.start
    while cursor < request.end:
        matches = [
            entry
            for entry in entries
            if date.fromisoformat(str(entry["start_date"]))
            <= cursor
            < date.fromisoformat(str(entry["end_date"]))
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Missing or ambiguous continuous mapping: "
                f"{request.continuous_symbol} {cursor}"
            )
        try:
            instrument_id = int(matches[0]["symbol"])
        except (TypeError, ValueError):
            raise ValueError(
                f"Continuous mapping is not an instrument ID: {request.data_path}"
            ) from None
        if instrument_id <= 0:
            raise ValueError(
                f"Continuous mapping has an invalid instrument ID: {request.data_path}"
            )
        output[cursor] = instrument_id
        cursor += timedelta(days=1)
    return output


def _validate_dbn_contract(
    request: HistoricalRequest,
) -> tuple[dict[str, object], dict[date, int]]:
    metadata = _dbn_metadata(request.data_path)
    expected = {
        "dataset": GLOBAL_DATASET,
        "schema": request.schema,
        "stype_in": CONTINUOUS_STYPE,
        "stype_out": INSTRUMENT_ID_STYPE,
        "start": _date_ns(request.start),
        "end": _date_ns(request.end),
        "symbols": [request.continuous_symbol],
    }
    if any(metadata[name] != value for name, value in expected.items()):
        raise ValueError(f"DBN request metadata mismatch: {request.data_path}")
    if metadata["partial"] or metadata["not_found"]:
        raise ValueError(
            f"DBN request contains unresolved symbols: {request.data_path}"
        )
    return metadata, _mapping_by_date(metadata, request)


def _validate_manifest_intervals(
    entries: Sequence[dict[str, object]],
    request: RequestRange,
    symbols: Sequence[str],
) -> None:
    for symbol in symbols:
        for schema in HISTORICAL_SCHEMAS:
            intervals = sorted(
                (
                    date.fromisoformat(str(entry["start"])),
                    date.fromisoformat(str(entry["end"])),
                )
                for entry in entries
                if entry.get("continuous_symbol") == symbol
                and entry.get("schema") == schema
            )
            cursor = request.start
            for start, end in intervals:
                if start > cursor:
                    raise ValueError(
                        f"Databento request interval gap: {symbol} {schema}"
                    )
                if start < cursor:
                    raise ValueError(
                        f"Databento request interval overlap: {symbol} {schema}"
                    )
                if end <= start:
                    raise ValueError(
                        f"Invalid Databento request interval: {symbol} {schema}"
                    )
                cursor = end
            if cursor != request.end:
                raise ValueError(f"Databento request interval gap: {symbol} {schema}")


def _validate_raw_acquisition(
    raw_dir: Path,
    request: RequestRange,
    symbols: Sequence[str] = GLOBAL_CONTEXT_SYMBOLS,
) -> tuple[tuple[HistoricalRequest, ...], dict[str, dict[date, int]]]:
    plan = request_plan(request, raw_dir, symbols)
    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Completed raw manifest is missing: {manifest_path}")
    manifest = _read_json(manifest_path)
    expected_header = {
        "status": "complete",
        "provider": GLOBAL_PROVIDER,
        "dataset": GLOBAL_DATASET,
        "schemas": list(HISTORICAL_SCHEMAS),
        "databento_version": GLOBAL_DATABENTO_VERSION,
        "symbols": list(symbols),
        "stype_in": CONTINUOUS_STYPE,
        "stype_out": INSTRUMENT_ID_STYPE,
        "requested_start": str(request.start),
        "requested_end": str(request.end),
        "continuous_roll_rule": GLOBAL_CONTINUOUS_ROLL_RULE,
    }
    if set(manifest) != {*expected_header, "requests"} or any(
        manifest[name] != value for name, value in expected_header.items()
    ):
        raise ValueError("Raw Databento manifest contract mismatch")
    entries = manifest["requests"]
    if not isinstance(entries, list) or any(
        not isinstance(entry, dict) for entry in entries
    ):
        raise ValueError("Raw Databento manifest requests are malformed")
    request_ids = [entry.get("request_id") for entry in entries]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("Duplicate Databento request in raw manifest")
    _validate_manifest_intervals(entries, request, symbols)
    _validate_planned_files(raw_dir, plan, complete=True)

    descriptors: list[dict[str, object]] = []
    mappings: dict[str, dict[date, int]] = {}
    for planned_request in plan:
        if not _completed_request(planned_request):
            raise ValueError("Expected Databento request is missing")
        descriptor = _read_json(planned_request.descriptor_path)
        descriptors.append(descriptor)
        _, mappings[planned_request.descriptor_path.stem] = _validate_dbn_contract(
            planned_request
        )
    if entries != descriptors:
        raise ValueError("Raw manifest requests do not match request descriptors")
    return plan, mappings


def _dbn_frame(path: Path) -> pl.DataFrame:
    frame = db.DBNStore.from_file(path).to_df(map_symbols=False).reset_index()
    return pl.from_pandas(frame)


def _timestamp_column(frame: pl.DataFrame) -> str:
    for name in ("ts_event_utc", "ts_event", "index"):
        if name in frame.columns:
            return name
    raise ValueError("Databento frame has no event timestamp")


def _definition_identities(definitions: pl.DataFrame) -> pl.DataFrame:
    required = {"instrument_id", "raw_symbol"}
    missing = sorted(required.difference(definitions.columns))
    if missing:
        raise ValueError(f"Definition records are missing columns: {missing}")
    expiration = (
        pl.col("expiration").cast(pl.Datetime("ns", "UTC"), strict=False)
        if "expiration" in definitions.columns
        else pl.lit(None, dtype=pl.Datetime("ns", "UTC"))
    )
    rows = definitions.select(
        pl.col("instrument_id").cast(pl.UInt32),
        pl.col("raw_symbol").cast(pl.String),
        expiration.alias("expiration_utc"),
    )
    if rows.filter(
        pl.col("instrument_id").is_null()
        | pl.col("raw_symbol").is_null()
        | (pl.col("raw_symbol").str.len_chars() == 0)
    ).height:
        raise ValueError("Definition records contain a missing outright identity")
    conflicts = rows.group_by("instrument_id").agg(
        pl.col("raw_symbol").n_unique().alias("raw_symbol_count"),
        pl.col("expiration_utc").drop_nulls().n_unique().alias("expiration_count"),
    )
    if conflicts.filter(
        (pl.col("raw_symbol_count") != 1) | (pl.col("expiration_count") > 1)
    ).height:
        raise ValueError("Definition records contain an ambiguous outright mapping")
    return rows.group_by("instrument_id").agg(
        pl.col("raw_symbol").first(),
        pl.col("expiration_utc").drop_nulls().first(),
    )


def _attach_definition_identity(
    frame: pl.DataFrame,
    definitions: pl.DataFrame,
) -> pl.DataFrame:
    if "instrument_id" not in frame.columns:
        raise ValueError("Databento bars are missing instrument_id")
    bars = frame.drop("raw_symbol") if "raw_symbol" in frame.columns else frame
    mapped = bars.join(
        _definition_identities(definitions),
        on="instrument_id",
        how="left",
        validate="m:1",
    )
    if mapped.filter(pl.col("raw_symbol").is_null()).height:
        raise ValueError("Definition records are missing an instrument mapping")
    return mapped


def _validate_bar_instrument_mapping(
    frame: pl.DataFrame,
    request: HistoricalRequest,
    mapping: dict[date, int],
) -> None:
    timestamp = _timestamp_column(frame)
    expected = pl.DataFrame(
        {
            "_mapping_date": list(mapping),
            "_expected_instrument_id": list(mapping.values()),
        },
        schema={"_mapping_date": pl.Date, "_expected_instrument_id": pl.UInt32},
    )
    checked = (
        frame.select(
            pl.col(timestamp)
            .cast(pl.Datetime("ns", "UTC"))
            .dt.date()
            .alias("_mapping_date"),
            pl.col("instrument_id").cast(pl.UInt32),
        )
        .join(expected, on="_mapping_date", how="left", validate="m:1")
        .filter(
            pl.col("_expected_instrument_id").is_null()
            | (pl.col("instrument_id") != pl.col("_expected_instrument_id"))
        )
    )
    if checked.height:
        raise ValueError(
            f"Bars disagree with the continuous mapping: {request.data_path}"
        )


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
    if definitions is not None:
        frame = _attach_definition_identity(frame, definitions)
    timestamp = _timestamp_column(frame)
    raw_symbol = "raw_symbol" if "raw_symbol" in frame.columns else "symbol"
    required = {timestamp, raw_symbol, "instrument_id", *PRICE_COLUMNS, "volume"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Databento frame is missing columns: {missing}")
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
            (
                pl.col("expiration_utc").cast(pl.Datetime("ns", "UTC"))
                if "expiration_utc" in frame.columns
                else pl.lit(None, dtype=pl.Datetime("ns", "UTC"))
            ).alias("expiration_utc"),
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
        .select(NORMALIZED_COLUMNS)
    )
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


def normalize_download(
    raw_dir: Path,
    request: RequestRange,
    *,
    created_at: datetime | None = None,
) -> Path:
    plan, mappings = _validate_raw_acquisition(raw_dir, request)
    created_at = datetime.now(UTC) if created_at is None else created_at
    output_dir = NORMALIZED_BASE / f"global_context_{created_at:%Y%m%dT%H%M%S%fZ}"
    partial = output_dir.with_name(f"{output_dir.name}.partial")
    if output_dir.exists() or partial.exists():
        raise FileExistsError(f"Global normalized output already exists: {output_dir}")

    try:
        (partial / "bars").mkdir(parents=True)
        source_hashes: dict[str, str] = {}
        normalized_hashes: dict[str, str] = {}
        row_count = 0
        symbols_seen: set[str] = set()
        previous_raw_symbols: dict[str, str] = {}
        for index in range(0, len(plan), len(HISTORICAL_SCHEMAS)):
            bars_request, definitions_request = plan[
                index : index + len(HISTORICAL_SCHEMAS)
            ]
            if (
                bars_request.schema != GLOBAL_SCHEMA
                or definitions_request.schema != "definition"
                or bars_request.continuous_symbol
                != definitions_request.continuous_symbol
                or bars_request.start != definitions_request.start
                or bars_request.end != definitions_request.end
            ):
                raise ValueError("Historical request pair does not match the contract")
            bar_mapping = mappings[bars_request.descriptor_path.stem]
            definition_mapping = mappings[definitions_request.descriptor_path.stem]
            if bar_mapping != definition_mapping:
                raise ValueError("OHLCV and definition continuous mappings disagree")
            bars = _dbn_frame(bars_request.data_path)
            _validate_bar_instrument_mapping(bars, bars_request, bar_mapping)
            symbol = bars_request.continuous_symbol
            normalized = normalize_bars(
                bars,
                symbol,
                _dbn_frame(definitions_request.data_path),
            )
            normalized = _with_mapping_changes(
                normalized, previous_raw_symbols.get(symbol)
            )
            previous_raw_symbols[symbol] = str(normalized.item(-1, "raw_symbol"))
            partition = (
                partial / "bars" / f"slot={int(normalized.item(0, 'global_slot')):02d}"
            )
            partition.mkdir(parents=True, exist_ok=True)
            target = partition / f"{bars_request.descriptor_path.stem}.parquet"
            normalized.write_parquet(target, compression="zstd", statistics=True)
            for source_path in (
                bars_request.data_path,
                definitions_request.data_path,
            ):
                source_hashes[str(source_path)] = _sha256(source_path)
            normalized_hashes[str(target.relative_to(partial))] = _sha256(target)
            row_count += normalized.height
            symbols_seen.add(symbol)
        if symbols_seen != set(GLOBAL_CONTEXT_SYMBOLS):
            raise ValueError(
                "Completed source requests do not contain the fixed global universe"
            )
        summary = (
            pl.scan_parquet(partial / "bars/**/*.parquet", glob=True)
            .select(
                pl.col("ts_event_utc").min().alias("actual_start"),
                pl.col("bar_end_utc").max().alias("actual_end"),
                pl.len().alias("rows"),
            )
            .collect()
            .row(0, named=True)
        )
        if int(summary["rows"]) != row_count:
            raise ValueError("Normalized row count changed during store validation")
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
            "requested_start": str(request.start),
            "requested_end": str(request.end),
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
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
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
    estimate.add_argument("--raw-dir", type=Path, default=RAW_BASE)
    download = subparsers.add_parser("download")
    download.add_argument("--universe-pointer", type=Path, required=True)
    download.add_argument("--raw-dir", type=Path, default=RAW_BASE)
    download.add_argument("--confirm-paid-download", action="store_true")
    normalize = subparsers.add_parser("normalize")
    normalize.add_argument("--universe-pointer", type=Path, required=True)
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
    if args.command in {"estimate", "download", "normalize"}:
        request = authoritative_request_range(resolve_pointer(args.universe_pointer))
        if args.command == "estimate":
            client = _historical_client()
            with _acquisition_lock(args.raw_dir):
                plan = request_plan(request, args.raw_dir)
                print(
                    json.dumps(
                        estimate_cost(client, remaining_requests(args.raw_dir, plan))
                    )
                )
        elif args.command == "download":
            download_history(
                _historical_client(),
                request,
                args.raw_dir,
                confirmed_paid_download=args.confirm_paid_download,
            )
        else:
            print(normalize_download(args.raw_dir, request))
    elif args.command == "shadow":
        run_shadow_collection(args.output_dir, args.flush_records)
    else:
        report = audit_candidate(pl.read_parquet(args.input_parquet), args.output)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
