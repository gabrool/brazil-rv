from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import polars as pl

from .contract import (
    ASSIGNMENTS_POINTER,
    CANONICAL_OUTPUT_POINTER,
    COTAHIST_POINTER,
    MIN_ACTIVE_EQUITIES,
    PROJECT_ROOT,
    UNIVERSE_POINTER,
)
from .io import load_assignments, resolve_pointer

SCHEMA_VERSION = "B3_HUMAN_PRIORS_V1"
RAW_SCHEMA_VERSION = "B3_HUMAN_PRIORS_RAW_V1"
RAW_BASE = PROJECT_ROOT / "quant-data/b3/raw/b3/human_priors"
OUTPUT_BASE = PROJECT_ROOT / "quant-data/b3/interim/b3/human_priors_v1"
CANONICAL_POINTER = (
    PROJECT_ROOT / "quant-data/b3/interim/b3/human_priors_v1_canonical_path.txt"
)

CLASSIFICATION_PAGE_URL = (
    "https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/"
    "renda-variavel/acoes/consultas/classificacao-setorial/"
)
CLASSIFICATION_API_BASE = (
    "https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/"
    "CompanyCall/GetDownloadIndustryClassification/"
)
MARKET_CAP_PAGE_URL = (
    "https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/"
    "market-data/consultas/mercado-a-vista/valor-de-mercado-das-empresas-"
    "listadas/bolsa-de-valores-mensal/"
)
MARKET_CAP_API_BASE = (
    "https://sistemaswebb3-listados.b3.com.br/marketValueProxy/"
    "marketValueCall/GetStockExchangeMonthlyDownload/"
)
UNITS_PAGE_URL = (
    "https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/"
    "market-data/consultas/mercado-a-vista/units/"
)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
MANUAL_MARKET_CAP_FILENAME = re.compile(
    r"^Bolsa_Valores_Mensal_(\d{4})-(\d{2})\.csv$", re.IGNORECASE
)
MARKET_CAP_REQUIRED_START_MONTH = "2020-06"
MIN_COMPLETE_MARKET_CAP_ISSUERS = 21
MARKET_CAP_HISTORY_LIMITATION = (
    "The current official B3 monthly application exposes only its latest two "
    "reference months. Its first-party application sends only company, language, "
    "keyword, pageNumber, and pageSize; it exposes no historical date selector or "
    "supported historical request parameter. Older official B3 CSV exports must be "
    "provided through batch manual ingestion."
)

RAW_SUBDIRECTORY = {
    "classification": "classification",
    "market_cap": "market_cap",
    "units": "units",
}

EXCEPTION_SCHEMA = {
    "source_type": pl.String,
    "source_file": pl.String,
    "source_key": pl.String,
    "source_name": pl.String,
    "reason": pl.String,
    "candidate_issuer_id": pl.String,
    "candidate_issuer_name": pl.String,
    "candidate_score": pl.Float64,
    "status": pl.String,
}


@dataclass(frozen=True)
class HttpPayload:
    body: bytes
    content_type: str
    source_url: str


@dataclass(frozen=True)
class RawSource:
    kind: str
    path: Path
    source_url: str
    retrieved_at_utc: datetime
    content_type: str
    sha256: str
    bytes: int
    reference_dates: tuple[date, ...]
    acquisition_method: str


@dataclass(frozen=True)
class FeatureInputs:
    universe_dir: Path
    assignments_dir: Path
    cotahist_dir: Path
    feature_store: Path
    universe_pointer: Path = UNIVERSE_POINTER
    assignments_pointer: Path = ASSIGNMENTS_POINTER
    cotahist_pointer: Path = COTAHIST_POINTER
    feature_store_pointer: Path = CANONICAL_OUTPUT_POINTER


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
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
    try:
        temporary.write_text(str(target.resolve()), encoding="utf-8")
        os.replace(temporary, pointer)
    finally:
        temporary.unlink(missing_ok=True)


def _json_url(base: str, payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    token = base64.b64encode(encoded.encode("utf-8")).decode("ascii")
    return f"{base}{urllib.parse.quote(token, safe='')}"


def _http_get(
    url: str,
    *,
    timeout: float = 60.0,
    retries: int = 3,
) -> HttpPayload:
    last_error = ""
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/octet-stream,text/csv,text/html,*/*",
                "Referer": "https://www.b3.com.br/",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
                content_type = response.headers.get_content_type()
                final_url = response.geturl()
            if not body:
                raise ValueError("B3 returned an empty response")
            return HttpPayload(body, content_type, final_url)
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            OSError,
            ValueError,
        ) as exc:
            last_error = f"attempt {attempt}/{retries}: {type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 4))
    raise RuntimeError(f"B3 download failed: {last_error}")


def _looks_like_html(data: bytes) -> bool:
    preview = data[:1024].lstrip().lower()
    return preview.startswith(b"<!doctype html") or preview.startswith(b"<html")


def _validate_xlsx(data: bytes) -> None:
    if _looks_like_html(data):
        raise ValueError("B3 returned an HTML page instead of an XLSX workbook")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            required = {
                "[Content_Types].xml",
                "xl/workbook.xml",
                "xl/worksheets/sheet1.xml",
            }
            missing = required - set(archive.namelist())
            if missing:
                raise ValueError(
                    f"Classification workbook is missing XLSX members: {sorted(missing)}"
                )
    except zipfile.BadZipFile:
        raise ValueError(
            "B3 classification response is not a valid XLSX file"
        ) from None


def _decode_market_cap_response(data: bytes) -> bytes:
    if _looks_like_html(data):
        raise ValueError("B3 returned an HTML page instead of market-cap data")
    try:
        decoded = base64.b64decode(data.strip(), validate=True)
    except (ValueError, base64.binascii.Error):
        raise ValueError("B3 market-cap response is not valid base64") from None
    if _looks_like_html(decoded):
        raise ValueError("B3 encoded an HTML error page as market-cap data")
    return decoded


def _source_descriptor_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.json")


def _source_from_descriptor(path: Path) -> RawSource:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != RAW_SCHEMA_VERSION:
        raise ValueError(f"Unsupported raw-source descriptor schema: {path}")
    source_path = path.parent / str(payload["filename"])
    if not source_path.is_file():
        raise FileNotFoundError(f"Raw B3 source is missing: {source_path}")
    actual_hash = _sha256_file(source_path)
    actual_size = source_path.stat().st_size
    if payload.get("sha256") != actual_hash or payload.get("bytes") != actual_size:
        raise ValueError(f"Raw B3 source descriptor mismatch: {source_path}")
    return RawSource(
        kind=str(payload["kind"]),
        path=source_path,
        source_url=str(payload["source_url"]),
        retrieved_at_utc=datetime.fromisoformat(str(payload["retrieved_at_utc"])),
        content_type=str(payload["content_type"]),
        sha256=actual_hash,
        bytes=actual_size,
        reference_dates=tuple(
            date.fromisoformat(value) for value in payload.get("reference_dates", [])
        ),
        acquisition_method=str(payload["acquisition_method"]),
    )


def discover_raw_sources(raw_dir: Path, kind: str) -> tuple[RawSource, ...]:
    if kind not in RAW_SUBDIRECTORY:
        raise ValueError(f"Unknown B3 human-prior source kind: {kind}")
    directory = raw_dir / RAW_SUBDIRECTORY[kind]
    sources = tuple(
        _source_from_descriptor(path)
        for path in sorted(directory.glob("*.json"))
        if not path.name.endswith("manifest.json")
    )
    return tuple(
        sorted(sources, key=lambda item: (item.retrieved_at_utc, item.path.name))
    )


def _cache_source(
    *,
    raw_dir: Path,
    kind: str,
    data: bytes,
    extension: str,
    source_url: str,
    content_type: str,
    retrieved_at: datetime,
    reference_dates: Sequence[date],
    acquisition_method: str,
) -> RawSource:
    directory = raw_dir / RAW_SUBDIRECTORY[kind]
    directory.mkdir(parents=True, exist_ok=True)
    digest = _sha256_bytes(data)
    for source in discover_raw_sources(raw_dir, kind):
        if source.sha256 == digest:
            descriptor = _source_descriptor_path(source.path)
            payload = json.loads(descriptor.read_text(encoding="utf-8"))
            payload.setdefault("first_retrieved_at_utc", payload["retrieved_at_utc"])
            payload.update(
                {
                    "source_url": source_url,
                    "retrieved_at_utc": retrieved_at.astimezone(UTC).isoformat(),
                    "content_type": content_type,
                    "reference_dates": [str(value) for value in reference_dates],
                    "acquisition_method": acquisition_method,
                }
            )
            _atomic_json(descriptor, payload)
            return _source_from_descriptor(descriptor)
    anchor = max(reference_dates, default=retrieved_at.date())
    target = directory / f"{kind}_{anchor:%Y-%m-%d}_{digest[:12]}{extension}"
    descriptor = _source_descriptor_path(target)
    if target.exists() or descriptor.exists():
        raise FileExistsError(f"Conflicting raw B3 cache artifact: {target}")
    temporary = target.with_name(f"{target.name}.partial")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, target)
        _atomic_json(
            descriptor,
            {
                "schema_version": RAW_SCHEMA_VERSION,
                "kind": kind,
                "filename": target.name,
                "source_url": source_url,
                "retrieved_at_utc": retrieved_at.astimezone(UTC).isoformat(),
                "first_retrieved_at_utc": retrieved_at.astimezone(UTC).isoformat(),
                "content_type": content_type,
                "sha256": digest,
                "bytes": len(data),
                "reference_dates": [str(value) for value in reference_dates],
                "acquisition_method": acquisition_method,
            },
        )
    except BaseException:
        temporary.unlink(missing_ok=True)
        if target.exists() and not descriptor.exists():
            target.unlink()
        raise
    return _source_from_descriptor(descriptor)


def _write_raw_manifest(raw_dir: Path) -> Path:
    sources = [
        source
        for kind in RAW_SUBDIRECTORY
        for source in discover_raw_sources(raw_dir, kind)
    ]
    manifest = raw_dir / "manifest.json"
    _atomic_json(
        manifest,
        {
            "schema_version": RAW_SCHEMA_VERSION,
            "sources": [
                {
                    "kind": source.kind,
                    "path": str(source.path.resolve()),
                    "source_url": source.source_url,
                    "retrieved_at_utc": source.retrieved_at_utc.isoformat(),
                    "content_type": source.content_type,
                    "sha256": source.sha256,
                    "bytes": source.bytes,
                    "reference_dates": [str(value) for value in source.reference_dates],
                    "acquisition_method": source.acquisition_method,
                }
                for source in sources
            ],
        },
    )
    return manifest


def _cached_today(raw_dir: Path, kind: str, today: date) -> RawSource | None:
    sources = discover_raw_sources(raw_dir, kind)
    if not sources or sources[-1].retrieved_at_utc.date() != today:
        return None
    return sources[-1]


def acquire_official_sources(
    raw_dir: Path = RAW_BASE,
    *,
    refresh: bool = False,
    now: datetime | None = None,
    fetch: Callable[[str], HttpPayload] = _http_get,
) -> dict[str, RawSource]:
    now = datetime.now(UTC) if now is None else now.astimezone(UTC)
    classification_url = _json_url(CLASSIFICATION_API_BASE, {"language": "pt-br"})
    market_cap_url = _json_url(
        MARKET_CAP_API_BASE,
        {
            "company": "",
            "language": "pt-br",
            "keyword": "",
            "pageNumber": 1,
            "pageSize": 20,
        },
    )
    acquired: dict[str, RawSource] = {}

    cached = None if refresh else _cached_today(raw_dir, "classification", now.date())
    if cached is None:
        response = fetch(classification_url)
        if response.content_type not in {
            "application/octet-stream",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }:
            raise ValueError(
                f"Unexpected B3 classification content type: {response.content_type}"
            )
        _validate_xlsx(response.body)
        parse_classification_xlsx(response.body, now.date())
        cached = _cache_source(
            raw_dir=raw_dir,
            kind="classification",
            data=response.body,
            extension=".xlsx",
            source_url=response.source_url,
            content_type=response.content_type,
            retrieved_at=now,
            reference_dates=[now.date()],
            acquisition_method="official_b3_api",
        )
    acquired["classification"] = cached

    cached = None if refresh else _cached_today(raw_dir, "market_cap", now.date())
    if cached is None:
        response = fetch(market_cap_url)
        if response.content_type not in {"text/plain", "application/octet-stream"}:
            raise ValueError(
                f"Unexpected B3 market-cap content type: {response.content_type}"
            )
        decoded = _decode_market_cap_response(response.body)
        market_cap = parse_market_cap_csv(decoded)
        reference_dates = market_cap["reference_date"].unique().sort().to_list()
        if max(reference_dates) >= date(now.year, now.month, 1):
            raise ValueError(
                "B3 market-cap export includes an incomplete current month"
            )
        cached = _cache_source(
            raw_dir=raw_dir,
            kind="market_cap",
            data=decoded,
            extension=".csv",
            source_url=response.source_url,
            content_type="text/csv; charset=utf-8",
            retrieved_at=now,
            reference_dates=reference_dates,
            acquisition_method="official_b3_api",
        )
    acquired["market_cap"] = cached

    cached = None if refresh else _cached_today(raw_dir, "units", now.date())
    if cached is None:
        response = fetch(UNITS_PAGE_URL)
        if response.content_type != "text/html":
            raise ValueError(
                f"Unexpected B3 units content type: {response.content_type}"
            )
        parse_units_html(response.body, now.date())
        cached = _cache_source(
            raw_dir=raw_dir,
            kind="units",
            data=response.body,
            extension=".html",
            source_url=response.source_url,
            content_type=response.content_type,
            retrieved_at=now,
            reference_dates=[now.date()],
            acquisition_method="official_b3_page",
        )
    acquired["units"] = cached
    _write_raw_manifest(raw_dir)
    return acquired


def market_cap_manual_instructions(raw_dir: Path = RAW_BASE) -> str:
    return (
        f"{MARKET_CAP_HISTORY_LIMITATION}\n"
        "\n"
        "Download only official B3 'Bolsa de Valores - Mensal' CSV exports from:\n"
        f"  {MARKET_CAP_PAGE_URL}\n"
        "Place all official historical exports in one otherwise-empty directory. "
        "Name each Bolsa_Valores_Mensal_YYYY-MM.csv, where YYYY-MM is the newest "
        "BRL reference calendar month inside that file, then run one batch command:\n"
        "  uv run python -m brazil_rv.preprocessing.human_priors "
        f'ingest-market-cap-dir --directory <DIRECTORY> --raw-dir "{raw_dir}"\n'
        "Every CSV is validated before the cache is modified. Do not substitute CVM, "
        "Receita Federal, third-party, or unofficial data. Strict build refuses raw "
        "or usable normalized month gaps, empty normalized data, and reconciliation "
        "conflicts; --allow-incomplete-market-cap creates a diagnostic-only artifact."
    )


def _validate_manual_market_cap(
    path: Path,
) -> tuple[bytes, pl.DataFrame, tuple[date, ...]]:
    match = MANUAL_MARKET_CAP_FILENAME.fullmatch(path.name)
    if match is None:
        raise ValueError(
            "Manual B3 market-cap filename must be Bolsa_Valores_Mensal_YYYY-MM.csv"
        )
    data = path.read_bytes()
    parsed = parse_market_cap_csv(data)
    reference_dates = tuple(parsed["reference_date"].unique().sort().to_list())
    expected_month = f"{int(match.group(1)):04d}-{int(match.group(2)):02d}"
    latest_month = max(parsed["reference_month"].to_list())
    if latest_month != expected_month:
        raise ValueError(
            f"Filename month {expected_month} does not match the newest B3 "
            f"reference calendar month {latest_month}"
        )
    return data, parsed, reference_dates


def ingest_manual_market_cap(
    path: Path,
    raw_dir: Path = RAW_BASE,
    *,
    retrieved_at: datetime | None = None,
) -> RawSource:
    data, _, reference_dates = _validate_manual_market_cap(path)
    retrieved_at = (
        datetime.now(UTC) if retrieved_at is None else retrieved_at.astimezone(UTC)
    )
    source = _cache_source(
        raw_dir=raw_dir,
        kind="market_cap",
        data=data,
        extension=".csv",
        source_url=MARKET_CAP_PAGE_URL,
        content_type="text/csv; charset=utf-8",
        retrieved_at=retrieved_at,
        reference_dates=reference_dates,
        acquisition_method="manual_official_b3_download",
    )
    _write_raw_manifest(raw_dir)
    return source


def ingest_market_cap_directory(
    directory: Path,
    raw_dir: Path = RAW_BASE,
    *,
    retrieved_at: datetime | None = None,
) -> tuple[RawSource, ...]:
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Manual B3 market-cap directory is missing: {directory}"
        )
    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No manual B3 market-cap CSV files found under {directory}"
        )
    invalid_names = [
        path.name
        for path in csv_files
        if MANUAL_MARKET_CAP_FILENAME.fullmatch(path.name) is None
    ]
    if invalid_names:
        raise ValueError(
            "Manual market-cap directory contains unexpected CSV filenames: "
            f"{invalid_names}"
        )

    validated = [(path, *_validate_manual_market_cap(path)) for path in csv_files]
    retrieved_at = (
        datetime.now(UTC) if retrieved_at is None else retrieved_at.astimezone(UTC)
    )
    sources = tuple(
        _cache_source(
            raw_dir=raw_dir,
            kind="market_cap",
            data=data,
            extension=".csv",
            source_url=MARKET_CAP_PAGE_URL,
            content_type="text/csv; charset=utf-8",
            retrieved_at=retrieved_at,
            reference_dates=reference_dates,
            acquisition_method="manual_official_b3_batch_download",
        )
        for _, data, _, reference_dates in validated
    )
    _write_raw_manifest(raw_dir)
    return sources


def _xlsx_rows(data: bytes) -> list[dict[int, str]]:
    _validate_xlsx(data)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(node.itertext()) for node in root]
        sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    rows: list[dict[int, str]] = []
    for row in sheet.iter(f"{namespace}row"):
        values: dict[int, str] = {}
        for cell in row.findall(f"{namespace}c"):
            reference = cell.attrib.get("r", "")
            match = re.match(r"([A-Z]+)", reference)
            if match is None:
                raise ValueError("XLSX cell is missing a column reference")
            column = 0
            for character in match.group(1):
                column = column * 26 + ord(character) - ord("A") + 1
            column -= 1
            cell_type = cell.attrib.get("t")
            if cell_type == "inlineStr":
                inline = cell.find(f"{namespace}is")
                value = "" if inline is None else "".join(inline.itertext())
            else:
                node = cell.find(f"{namespace}v")
                value = "" if node is None or node.text is None else node.text
                if cell_type == "s" and value:
                    try:
                        value = shared[int(value)]
                    except (IndexError, ValueError):
                        raise ValueError(
                            "XLSX shared-string index is invalid"
                        ) from None
            values[column] = value.strip()
        rows.append(values)
    return rows


def normalize_identity_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = "".join(
        character for character in text if not unicodedata.combining(character)
    )
    return re.sub(r"[^A-Z0-9]", "", ascii_text.upper())


def parse_classification_xlsx(
    data: bytes,
    snapshot_date: date,
    *,
    source_file: str = "",
    source_url: str = CLASSIFICATION_PAGE_URL,
    retrieved_at_utc: datetime | None = None,
) -> pl.DataFrame:
    rows = _xlsx_rows(data)
    header_index = None
    positions: dict[str, int] = {}
    for index, row in enumerate(rows):
        canonical = {
            normalize_identity_name(value): column for column, value in row.items()
        }
        if {"SETOR", "SUBSETOR", "SEGMENTO", "EMISSOR"}.issubset(canonical):
            header_index = index
            positions = {
                "sector": canonical["SETOR"],
                "subsector": canonical["SUBSETOR"],
                "economic_segment": canonical["SEGMENTO"],
                "issuer": canonical["EMISSOR"],
            }
            break
    if header_index is None:
        raise ValueError("B3 classification workbook schema drift: headers not found")
    for row in rows[header_index + 1 : header_index + 4]:
        for column, value in row.items():
            canonical = normalize_identity_name(value)
            if canonical == "NOMEDEPREGÃO" or canonical == "NOMEDEPREGAO":
                positions["issuer"] = column
            elif canonical == "CODIGO":
                positions["code"] = column
            elif canonical == "SEGMENTODENEGOCIACAO":
                positions["trading_segment"] = column
    required = {"sector", "subsector", "economic_segment", "issuer", "code"}
    if not required.issubset(positions):
        raise ValueError(
            "B3 classification workbook schema drift: "
            f"missing columns {sorted(required - set(positions))}"
        )

    hierarchy = {"sector": "", "subsector": "", "economic_segment": ""}
    parsed: list[dict[str, object]] = []
    retrieved_at_utc = retrieved_at_utc or datetime.combine(
        snapshot_date, datetime.min.time(), UTC
    )
    for row in rows[header_index + 1 :]:
        issuer_name = row.get(positions["issuer"], "").strip()
        issuer_code = row.get(positions["code"], "").strip().upper()
        issuer_header = normalize_identity_name(issuer_name)
        if issuer_header in {"NOMEDEPREGÃO", "NOMEDEPREGAO", "EMISSOR"}:
            continue
        for field in hierarchy:
            value = row.get(positions[field], "").strip()
            if value and normalize_identity_name(value) not in {
                "SETOR",
                "SETORECONOMICO",
                "SUBSETOR",
                "SEGMENTO",
            }:
                hierarchy[field] = value
        if not issuer_name and not issuer_code:
            continue
        if not issuer_name or not re.fullmatch(r"[A-Z0-9]{4,12}", issuer_code):
            raise ValueError(
                "B3 classification workbook has an incomplete issuer row: "
                f"issuer={issuer_name!r}, code={issuer_code!r}"
            )
        if not all(hierarchy.values()):
            raise ValueError(
                f"B3 classification hierarchy is incomplete for {issuer_name}"
            )
        parsed.append(
            {
                "issuer_id": f"B3_ISSUER_CODE:{issuer_code}",
                "issuer_b3_code": issuer_code,
                "issuer_name": issuer_name,
                **hierarchy,
                "trading_segment": row.get(
                    positions.get("trading_segment", -1), ""
                ).strip(),
                "classification_snapshot_date": snapshot_date,
                "source_file": source_file,
                "source_url": source_url,
                "retrieved_at_utc": retrieved_at_utc,
            }
        )
    if not parsed:
        raise ValueError("B3 classification workbook contains no issuer rows")
    frame = pl.from_dicts(parsed, infer_schema_length=None).sort("issuer_id")
    conflicts = (
        frame.group_by("issuer_id")
        .agg(
            pl.struct("issuer_name", "sector", "subsector", "economic_segment")
            .n_unique()
            .alias("versions")
        )
        .filter(pl.col("versions") > 1)
    )
    if not conflicts.is_empty():
        raise ValueError(
            "B3 classification workbook contains conflicting issuer codes: "
            f"{conflicts['issuer_id'].to_list()}"
        )
    return frame.unique("issuer_id", keep="first", maintain_order=True)


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("B3 text response has an unsupported encoding")


def _parse_brl(value: str) -> float:
    text = value.strip().replace(" ", "")
    if not text:
        raise ValueError("Market capitalization is empty")
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".")
    elif "," in text:
        text = text.replace(",", ".")
    try:
        parsed = float(text)
    except ValueError:
        raise ValueError(f"Invalid B3 market capitalization: {value!r}") from None
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"Invalid B3 market capitalization: {value!r}")
    return parsed


def parse_market_cap_csv(
    data: bytes,
    *,
    source_file: str = "",
    source_url: str = MARKET_CAP_PAGE_URL,
    retrieved_at_utc: datetime | None = None,
) -> pl.DataFrame:
    if _looks_like_html(data):
        raise ValueError("B3 returned HTML instead of a market-cap CSV")
    rows = list(csv.reader(io.StringIO(_decode_text(data)), delimiter="|"))
    header_index = None
    issuer_column = None
    value_columns: list[tuple[int, date]] = []
    pattern = re.compile(
        r"^\s*Valor\s*\(R\$\)\s*em\s*(\d{2}/\d{2}/\d{4})\s*$",
        re.IGNORECASE,
    )
    for index, row in enumerate(rows[:10]):
        normalized = [normalize_identity_name(value) for value in row]
        if "EMPRESA" not in normalized:
            continue
        header_index = index
        issuer_column = normalized.index("EMPRESA")
        for column, value in enumerate(row):
            match = pattern.match(value)
            if match:
                value_columns.append(
                    (column, datetime.strptime(match.group(1), "%d/%m/%Y").date())
                )
        break
    if header_index is None or issuer_column is None or not value_columns:
        raise ValueError("B3 market-cap CSV schema drift: required columns not found")
    reference_months = {
        f"{reference_date.year:04d}-{reference_date.month:02d}"
        for _, reference_date in value_columns
    }
    if len(reference_months) != len(value_columns):
        raise ValueError("B3 market-cap CSV contains duplicate calendar-month columns")
    parsed: list[dict[str, object]] = []
    snapshot_date = None
    if rows and rows[0] and re.fullmatch(r"\d{8}", rows[0][0].strip()):
        snapshot_date = datetime.strptime(rows[0][0].strip(), "%Y%m%d").date()
    retrieved_at_utc = retrieved_at_utc or datetime.now(UTC)
    source_issuers: set[str] = set()
    declared_issuer_count: int | None = None
    for row in rows[header_index + 1 :]:
        if issuer_column >= len(row):
            continue
        issuer_name = row[issuer_column].strip()
        normalized_issuer = normalize_identity_name(issuer_name)
        total_match = re.fullmatch(r"TOTALGERAL(\d+)", normalized_issuer)
        if total_match:
            declared_issuer_count = int(total_match.group(1))
            continue
        if not issuer_name or normalized_issuer.startswith("TOTAL"):
            continue
        source_issuers.add(normalized_issuer)
        for column, reference_date in value_columns:
            if column >= len(row) or not row[column].strip():
                continue
            parsed.append(
                {
                    "reference_date": reference_date,
                    "reference_month": f"{reference_date.year:04d}-{reference_date.month:02d}",
                    "issuer_name": issuer_name,
                    "market_cap_brl": _parse_brl(row[column]),
                    "source_snapshot_date": snapshot_date,
                    "source_file": source_file,
                    "source_url": source_url,
                    "retrieved_at_utc": retrieved_at_utc,
                }
            )
    if not parsed:
        raise ValueError("B3 market-cap CSV contains no issuer observations")
    if declared_issuer_count is None:
        raise ValueError(
            "B3 market-cap CSV is missing the 'Total Geral (N)' completeness footer"
        )
    if len(source_issuers) < MIN_COMPLETE_MARKET_CAP_ISSUERS:
        raise ValueError(
            "B3 market-cap CSV appears pagination-truncated: "
            f"only {len(source_issuers)} issuer rows were returned"
        )
    if len(source_issuers) != declared_issuer_count:
        raise ValueError(
            "B3 market-cap CSV issuer rows do not match its completeness footer: "
            f"rows={len(source_issuers)}, declared={declared_issuer_count}"
        )
    frame = pl.from_dicts(parsed, infer_schema_length=None).sort(
        "reference_date", "issuer_name"
    )
    duplicates = frame.filter(
        pl.struct("reference_month", "issuer_name").is_duplicated()
    )
    if not duplicates.is_empty():
        raise ValueError("B3 market-cap CSV has duplicate issuer/month rows")
    return frame


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "tr":
            self._row = []
        elif tag.lower() in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"td", "th"} and self._cell is not None:
            assert self._row is not None
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag.lower() == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None


def _parse_component(text: str) -> tuple[int, str]:
    match = re.fullmatch(
        r"\s*(\d+)\s+(?:aç(?:ão|ões)\s+)?"
        r"(ON|PN[A-F]?|BDR\s+[A-Z])(?:\s+stock.*)?\s*",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"Unsupported B3 unit component: {text!r}")
    return int(match.group(1)), re.sub(r"\s+", " ", match.group(2).upper())


def parse_units_html(
    data: bytes,
    snapshot_date: date,
    *,
    source_file: str = "",
    source_url: str = UNITS_PAGE_URL,
    retrieved_at_utc: datetime | None = None,
) -> pl.DataFrame:
    parser = _TableParser()
    parser.feed(_decode_text(data))
    parsed: list[dict[str, object]] = []
    retrieved_at_utc = retrieved_at_utc or datetime.combine(
        snapshot_date, datetime.min.time(), UTC
    )
    for row in parser.rows:
        if len(row) < 3 or normalize_identity_name(row[1]) == "CODIGO":
            continue
        unit_ticker = row[1].strip().upper()
        if not re.fullmatch(r"[A-Z0-9]{4,8}", unit_ticker):
            continue
        components = re.split(r"\s*\+\s*", row[2].strip())
        if not components:
            raise ValueError(f"B3 unit has no components: {unit_ticker}")
        for component in components:
            quantity, share_class = _parse_component(component)
            parsed.append(
                {
                    "unit_ticker": unit_ticker,
                    "issuer_name": row[0].strip(),
                    "component_share_class": share_class,
                    "component_quantity": quantity,
                    "unit_snapshot_date": snapshot_date,
                    "source_file": source_file,
                    "source_url": source_url,
                    "retrieved_at_utc": retrieved_at_utc,
                }
            )
    if not parsed:
        raise ValueError("B3 units page schema drift: unit table not found")
    frame = pl.from_dicts(parsed, infer_schema_length=None).sort(
        "unit_ticker", "component_share_class"
    )
    duplicates = frame.filter(
        pl.struct("unit_ticker", "component_share_class").is_duplicated()
    )
    if not duplicates.is_empty():
        raise ValueError("B3 units page has duplicate unit/class components")
    return frame


def normalize_share_class(value: object) -> str:
    text = str(value or "").strip().upper().split()[0] if value else ""
    return {"OR": "ON", "UNT": "UNIT"}.get(text, text)


def _empty_exceptions() -> pl.DataFrame:
    return pl.DataFrame(schema=EXCEPTION_SCHEMA)


def _exception_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    if not rows:
        return _empty_exceptions()
    return pl.from_dicts(rows, schema=EXCEPTION_SCHEMA, strict=False)


def _fuzzy_candidates(
    name: str, classification: pl.DataFrame, limit: int = 3
) -> list[tuple[str, str, float]]:
    normalized = normalize_identity_name(name)
    candidates = [
        (
            str(row["issuer_id"]),
            str(row["issuer_name"]),
            SequenceMatcher(
                None, normalized, normalize_identity_name(row["issuer_name"])
            ).ratio(),
        )
        for row in classification.select("issuer_id", "issuer_name").to_dicts()
    ]
    return sorted(candidates, key=lambda item: (-item[2], item[0]))[:limit]


def _compatible_share_class(security_class: str, source_class: object) -> bool:
    candidate = normalize_share_class(source_class)
    if not candidate:
        return True
    if candidate == security_class:
        return True
    return candidate == "PN" and security_class.startswith("PN")


def reconcile_security_metadata(
    assignments: pl.DataFrame,
    ticker_history: pl.DataFrame,
    classification: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    required_assignment = {"security_id", "isin", "latest_ticker"}
    required_history = {
        "security_id",
        "ticker",
        "valid_from",
        "valid_to",
        "isin",
        "issuer_short_name",
        "security_spec",
    }
    if missing := required_assignment - set(assignments.columns):
        raise ValueError(f"Accepted assignments are missing columns: {sorted(missing)}")
    if missing := required_history - set(ticker_history.columns):
        raise ValueError(f"Ticker history is missing columns: {sorted(missing)}")
    latest = (
        ticker_history.filter(
            pl.col("security_id").is_in(assignments["security_id"].to_list())
        )
        .sort("security_id", "valid_to")
        .group_by("security_id", maintain_order=True)
        .last()
    )
    source = assignments.select("security_id", "isin", "latest_ticker").join(
        latest, on="security_id", how="left", suffix="_history"
    )
    if source["issuer_short_name"].null_count():
        missing_ids = source.filter(pl.col("issuer_short_name").is_null())[
            "security_id"
        ].to_list()
        raise ValueError(
            f"Accepted securities are missing ticker history: {missing_ids}"
        )

    by_name: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_isin: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_ticker: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in classification.to_dicts():
        by_name[normalize_identity_name(row["issuer_name"])].append(row)
        if row.get("isin"):
            by_isin[str(row["isin"]).upper()].append(row)
        if row.get("ticker"):
            by_ticker[str(row["ticker"]).upper()].append(row)

    metadata_rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    exception_rows: list[dict[str, object]] = []
    for security in source.sort("security_id").to_dicts():
        security_id = str(security["security_id"])
        ticker = str(security.get("ticker") or security["latest_ticker"])
        isin = str(security.get("isin_history") or security["isin"])
        cotahist_name = str(security["issuer_short_name"])
        share_class = normalize_share_class(security["security_spec"])
        matches: list[dict[str, object]] = []
        method = ""
        if "security_id" in classification.columns:
            matches = classification.filter(
                pl.col("security_id") == security_id
            ).to_dicts()
            method = "EXACT_SECURITY_ID"
        if not matches and isin:
            matches = [
                row
                for row in by_isin.get(isin.upper(), [])
                if _compatible_share_class(share_class, row.get("share_class"))
            ]
            method = "EXACT_ISIN"
        snapshot = classification["classification_snapshot_date"][0]
        overlaps_snapshot = bool(
            security["valid_from"] <= snapshot <= security["valid_to"]
        )
        if not matches and overlaps_snapshot:
            matches = [
                row
                for row in by_ticker.get(ticker.upper(), [])
                if _compatible_share_class(share_class, row.get("share_class"))
            ]
            method = "EXACT_TICKER_ON_SNAPSHOT_DATE"
        if not matches:
            matches = [
                row
                for row in by_name.get(normalize_identity_name(cotahist_name), [])
                if _compatible_share_class(share_class, row.get("share_class"))
            ]
            method = "EXACT_NORMALIZED_B3_ISSUER_NAME"

        issuer_ids = {str(row["issuer_id"]) for row in matches}
        mapped = len(issuer_ids) == 1
        chosen = (
            sorted(matches, key=lambda row: str(row["issuer_id"]))[0]
            if mapped
            else None
        )
        if mapped and chosen is not None:
            status = "MAPPED"
            reason = ""
        elif matches:
            status = "AMBIGUOUS"
            reason = "AMBIGUOUS_EXACT_MATCH"
            method = ""
        else:
            status = "UNRESOLVED"
            reason = "NO_EXACT_MATCH"
            method = ""

        candidate_text = ""
        candidate_score: float | None = None
        if not mapped:
            candidates = (
                [
                    (str(row["issuer_id"]), str(row["issuer_name"]), 1.0)
                    for row in matches
                ]
                if matches
                else _fuzzy_candidates(cotahist_name, classification)
            )
            candidate_text = " | ".join(
                f"{candidate_id}:{candidate_name}:{score:.6f}"
                for candidate_id, candidate_name, score in candidates
            )
            candidate_score = candidates[0][2] if candidates else None
            for candidate_id, candidate_name, score in candidates or [("", "", None)]:
                exception_rows.append(
                    {
                        "source_type": "security_classification",
                        "source_file": str(classification["source_file"][0]),
                        "source_key": security_id,
                        "source_name": cotahist_name,
                        "reason": reason,
                        "candidate_issuer_id": candidate_id,
                        "candidate_issuer_name": candidate_name,
                        "candidate_score": score,
                        "status": status,
                    }
                )

        common = {
            "security_id": security_id,
            "ticker": ticker,
            "isin": isin,
            "issuer_id": chosen["issuer_id"] if chosen else None,
            "issuer_name": chosen["issuer_name"] if chosen else cotahist_name,
            "cotahist_issuer_name": cotahist_name,
            "share_class": share_class,
            "sector": chosen["sector"] if chosen else None,
            "subsector": chosen["subsector"] if chosen else None,
            "economic_segment": chosen["economic_segment"] if chosen else None,
            "classification_snapshot_date": snapshot,
            "matching_method": method or "UNRESOLVED",
            "mapping_status": status,
            "source_file": str(classification["source_file"][0]),
            "source_url": str(classification["source_url"][0]),
        }
        metadata_rows.append(common)
        review_rows.append(
            {
                **common,
                "review_reason": reason,
                "fuzzy_candidates": candidate_text,
                "best_fuzzy_score": candidate_score,
            }
        )
    metadata = pl.from_dicts(metadata_rows, infer_schema_length=None).sort(
        "security_id"
    )
    review = pl.from_dicts(review_rows, infer_schema_length=None).sort("security_id")
    return metadata, review, _exception_frame(exception_rows)


def _issuer_aliases(
    classification: pl.DataFrame, security_metadata: pl.DataFrame
) -> dict[str, tuple[str, str, str] | None]:
    candidates: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for row in classification.select("issuer_id", "issuer_name").to_dicts():
        candidates[normalize_identity_name(row["issuer_name"])].add(
            (
                str(row["issuer_id"]),
                str(row["issuer_name"]),
                "EXACT_NORMALIZED_B3_ISSUER_NAME",
            )
        )
    for row in (
        security_metadata.filter(pl.col("issuer_id").is_not_null())
        .select("issuer_id", "issuer_name", "cotahist_issuer_name")
        .to_dicts()
    ):
        candidates[normalize_identity_name(row["cotahist_issuer_name"])].add(
            (
                str(row["issuer_id"]),
                str(row["issuer_name"]),
                "EXACT_NORMALIZED_COTAHIST_ISSUER_NAME",
            )
        )
    aliases: dict[str, tuple[str, str, str] | None] = {}
    for name, values in candidates.items():
        issuer_ids = {value[0] for value in values}
        if len(issuer_ids) != 1:
            aliases[name] = None
            continue
        aliases[name] = sorted(values, key=lambda value: value[2])[0]
    return aliases


def reconcile_market_cap(
    market_cap: pl.DataFrame,
    classification: pl.DataFrame,
    security_metadata: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, object]]:
    aliases = _issuer_aliases(classification, security_metadata)
    matched: list[dict[str, object]] = []
    exceptions: list[dict[str, object]] = []
    source_names = {
        normalize_identity_name(value) for value in market_cap["issuer_name"].to_list()
    }
    mapped_names: set[str] = set()
    for row in market_cap.sort("reference_date", "issuer_name").to_dicts():
        normalized_name = normalize_identity_name(row["issuer_name"])
        alias = aliases.get(normalized_name)
        if alias is not None:
            issuer_id, issuer_name, method = alias
            matched.append(
                {
                    "reference_date": row["reference_date"],
                    "reference_month": row["reference_month"],
                    "issuer_id": issuer_id,
                    "issuer_name": issuer_name,
                    "market_cap_brl": row["market_cap_brl"],
                    "matching_method": method,
                    "source_file": row["source_file"],
                    "source_url": row["source_url"],
                    "retrieved_at_utc": row["retrieved_at_utc"],
                }
            )
            mapped_names.add(normalized_name)
            continue
        ambiguous = normalized_name in aliases
        candidates = _fuzzy_candidates(str(row["issuer_name"]), classification)
        for candidate_id, candidate_name, score in candidates or [("", "", None)]:
            exceptions.append(
                {
                    "source_type": "market_cap",
                    "source_file": str(row["source_file"]),
                    "source_key": f"{row['reference_date']}:{row['issuer_name']}",
                    "source_name": str(row["issuer_name"]),
                    "reason": (
                        "AMBIGUOUS_EXACT_ISSUER_NAME"
                        if ambiguous
                        else "NO_EXACT_ISSUER_NAME"
                    ),
                    "candidate_issuer_id": candidate_id,
                    "candidate_issuer_name": candidate_name,
                    "candidate_score": score,
                    "status": "UNRESOLVED",
                }
            )

    matched_frame = (
        pl.from_dicts(matched, infer_schema_length=None)
        if matched
        else pl.DataFrame(
            schema={
                "reference_date": pl.Date,
                "reference_month": pl.String,
                "issuer_id": pl.String,
                "issuer_name": pl.String,
                "market_cap_brl": pl.Float64,
                "matching_method": pl.String,
                "source_file": pl.String,
                "source_url": pl.String,
                "retrieved_at_utc": pl.Datetime(time_zone="UTC"),
            }
        )
    )
    normalized: list[dict[str, object]] = []
    duplicate_groups = 0
    conflict_groups = 0
    for group in matched_frame.partition_by(
        ["reference_month", "issuer_id"], maintain_order=True
    ):
        rows = group.sort(
            "reference_date", "retrieved_at_utc", "source_file"
        ).to_dicts()
        values = [float(row["market_cap_brl"]) for row in rows]
        if len(rows) > 1:
            duplicate_groups += 1
        tolerance = max(0.01, max(values, default=0.0) * 1e-10)
        if values and max(values) - min(values) > tolerance:
            conflict_groups += 1
            for row in rows:
                exceptions.append(
                    {
                        "source_type": "market_cap",
                        "source_file": str(row["source_file"]),
                        "source_key": f"{row['reference_month']}:{row['issuer_id']}",
                        "source_name": str(row["issuer_name"]),
                        "reason": "CONFLICTING_ISSUER_MONTH_VALUES",
                        "candidate_issuer_id": str(row["issuer_id"]),
                        "candidate_issuer_name": str(row["issuer_name"]),
                        "candidate_score": 1.0,
                        "status": "CONFLICT",
                    }
                )
            continue
        normalized.append(rows[-1])
    output = (
        pl.from_dicts(normalized, infer_schema_length=None).sort(
            "reference_date", "issuer_id"
        )
        if normalized
        else matched_frame
    )
    stats = {
        "source_row_count": market_cap.height,
        "source_distinct_issuer_names": len(source_names),
        "mapped_distinct_issuer_names": len(mapped_names),
        "issuer_name_mapping_fraction": (
            len(mapped_names) / len(source_names) if source_names else 0.0
        ),
        "normalized_row_count": output.height,
        "normalized_issuer_count": (
            output["issuer_id"].n_unique() if output.height else 0
        ),
        "duplicate_issuer_month_groups": duplicate_groups,
        "conflicting_issuer_month_groups": conflict_groups,
    }
    return output, _exception_frame(exceptions), stats


def normalize_unit_components(
    units: pl.DataFrame,
    classification: pl.DataFrame,
    security_metadata: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    aliases = _issuer_aliases(classification, security_metadata)
    securities_by_ticker: dict[str, list[dict[str, object]]] = defaultdict(list)
    securities_by_issuer_class: dict[tuple[str, str], list[dict[str, object]]] = (
        defaultdict(list)
    )
    for row in security_metadata.to_dicts():
        securities_by_ticker[str(row["ticker"]).upper()].append(row)
        if row.get("issuer_id"):
            securities_by_issuer_class[
                (str(row["issuer_id"]), str(row["share_class"]))
            ].append(row)
    output: list[dict[str, object]] = []
    exceptions: list[dict[str, object]] = []
    for row in units.sort("unit_ticker", "component_share_class").to_dicts():
        unit_matches = securities_by_ticker[str(row["unit_ticker"]).upper()]
        unit_security = unit_matches[0] if len(unit_matches) == 1 else None
        alias = aliases.get(normalize_identity_name(row["issuer_name"]))
        issuer_id = (
            str(unit_security["issuer_id"])
            if unit_security is not None and unit_security.get("issuer_id")
            else alias[0]
            if alias is not None
            else None
        )
        component_class = normalize_share_class(row["component_share_class"])
        component_matches = (
            securities_by_issuer_class.get((issuer_id, component_class), [])
            if issuer_id
            else []
        )
        component_security = (
            component_matches[0] if len(component_matches) == 1 else None
        )
        if len(unit_matches) > 1:
            status = "AMBIGUOUS_UNIT_SECURITY"
        elif unit_security is None:
            status = "UNIT_NOT_IN_ACCEPTED_UNIVERSE"
        elif issuer_id is None:
            status = "UNRESOLVED_ISSUER"
        elif len(component_matches) > 1:
            status = "AMBIGUOUS_COMPONENT_SECURITY"
        elif component_security is None:
            status = "COMPONENT_NOT_IN_ACCEPTED_UNIVERSE"
        else:
            status = "MAPPED"
        normalized = {
            "unit_security_id": (
                unit_security["security_id"] if unit_security is not None else None
            ),
            "unit_ticker": row["unit_ticker"],
            "component_issuer_id": issuer_id,
            "component_share_class": component_class,
            "component_security_id": (
                component_security["security_id"]
                if component_security is not None
                else None
            ),
            "component_quantity": int(row["component_quantity"]),
            "mapping_status": status,
            "unit_snapshot_date": row["unit_snapshot_date"],
            "source_file": row["source_file"],
            "source_url": row["source_url"],
        }
        output.append(normalized)
        if status not in {"MAPPED", "COMPONENT_NOT_IN_ACCEPTED_UNIVERSE"}:
            exceptions.append(
                {
                    "source_type": "unit_component",
                    "source_file": str(row["source_file"]),
                    "source_key": f"{row['unit_ticker']}:{component_class}",
                    "source_name": str(row["issuer_name"]),
                    "reason": status,
                    "candidate_issuer_id": issuer_id or "",
                    "candidate_issuer_name": alias[1] if alias else "",
                    "candidate_score": 1.0 if alias else None,
                    "status": "UNRESOLVED",
                }
            )
    frame = pl.from_dicts(output, infer_schema_length=None).sort(
        "unit_ticker", "component_share_class"
    )
    return frame, _exception_frame(exceptions)


def active_security_days(
    dates: Sequence[date],
    security_ids: Sequence[str],
    membership: np.ndarray,
    data_ready: np.ndarray,
    *,
    minimum_active: int = MIN_ACTIVE_EQUITIES,
) -> tuple[pl.DataFrame, np.ndarray]:
    expected_shape = (len(dates), len(security_ids))
    if membership.shape != expected_shape or data_ready.shape != expected_shape:
        raise ValueError(
            "Feature membership/readiness shape does not match date/security axes"
        )
    active = np.asarray(membership, dtype=bool) & np.asarray(data_ready, dtype=bool)
    eligible_dates = active.sum(axis=1) >= minimum_active
    date_indices, security_indices = np.nonzero(active & eligible_dates[:, None])
    frame = pl.DataFrame(
        {
            "date_idx": date_indices.astype(np.int32),
            "trade_date": [dates[index] for index in date_indices],
            "equity_slot": security_indices.astype(np.int16),
            "security_id": [security_ids[index] for index in security_indices],
        }
    ).with_columns(pl.col("trade_date").cast(pl.Date))
    return frame, eligible_dates


def add_self_excluded_peer_counts(
    active: pl.DataFrame, security_metadata: pl.DataFrame
) -> pl.DataFrame:
    joined = active.join(
        security_metadata.select(
            "security_id", "issuer_id", "sector", "subsector", "economic_segment"
        ),
        on="security_id",
        how="left",
        validate="m:1",
    )
    for group, output in (
        ("issuer_id", "same_issuer_peer_count"),
        ("sector", "same_sector_peer_count"),
        ("subsector", "same_subsector_peer_count"),
        ("economic_segment", "same_economic_segment_peer_count"),
    ):
        counts = (
            joined.filter(pl.col(group).is_not_null())
            .group_by("trade_date", group)
            .len()
            .rename({"len": output})
            .with_columns((pl.col(output) - 1).cast(pl.Int32))
        )
        joined = joined.join(counts, on=["trade_date", group], how="left").with_columns(
            pl.col(output).fill_null(0)
        )
    return joined.sort("trade_date", "equity_slot")


def strictly_lagged_market_cap_index(
    reference_dates: Sequence[date], model_date: date
) -> int | None:
    index = bisect_left(reference_dates, model_date) - 1
    return index if index >= 0 else None


def _distribution(values: Sequence[int | float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "minimum": None,
            "p10": None,
            "median": None,
            "p90": None,
            "maximum": None,
        }
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(array.max()),
    }


def _peer_coverage(values: Sequence[int]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.int64)
    denominator = int(array.size)
    return {
        "security_day_count": denominator,
        "peer_count_distribution": _distribution(array),
        "percentage_with_at_least": {
            str(threshold): (
                float(np.mean(array >= threshold) * 100.0) if denominator else 0.0
            )
            for threshold in (1, 2, 3, 5)
        },
    }


def classification_peer_audit(
    peers: pl.DataFrame,
    security_metadata: pl.DataFrame,
    classification: pl.DataFrame,
) -> tuple[dict[str, object], pl.DataFrame]:
    level_definitions = (
        ("sector", "same_sector_peer_count"),
        ("subsector", "same_subsector_peer_count"),
        ("economic_segment", "same_economic_segment_peer_count"),
    )
    audit: dict[str, object] = {}
    group_rows: list[dict[str, object]] = []
    for level, peer_column in level_definitions:
        mapped = security_metadata.filter(pl.col(level).is_not_null())
        static = mapped.group_by(level).len().rename({"len": "accepted_security_count"})
        dynamic = (
            peers.filter(pl.col(level).is_not_null())
            .group_by("trade_date", level)
            .len()
            .rename({"len": "active_group_size"})
        )
        level_rows: list[dict[str, object]] = []
        for row in static.sort(level).to_dicts():
            group_name = str(row[level])
            sizes = dynamic.filter(pl.col(level) == group_name)[
                "active_group_size"
            ].to_list()
            group_peer_values = peers.filter(pl.col(level) == group_name)[
                peer_column
            ].to_list()
            sectors = (
                security_metadata.filter(pl.col(level) == group_name)["sector"]
                .drop_nulls()
                .unique()
                .sort()
                .to_list()
            )
            summary = {
                "level": level,
                "sector": " | ".join(str(value) for value in sectors),
                "group": group_name,
                "accepted_security_count": int(row["accepted_security_count"]),
                "active_group_size_min": int(min(sizes)) if sizes else 0,
                "active_group_size_p10": (
                    float(np.quantile(sizes, 0.10)) if sizes else 0.0
                ),
                "active_group_size_median": (float(np.median(sizes)) if sizes else 0.0),
                "active_group_size_p90": (
                    float(np.quantile(sizes, 0.90)) if sizes else 0.0
                ),
                "active_group_size_max": int(max(sizes)) if sizes else 0,
                "security_day_count": len(group_peer_values),
                "zero_peer_fraction": (
                    float(np.mean(np.asarray(group_peer_values) == 0))
                    if group_peer_values
                    else 0.0
                ),
                "at_most_one_peer_fraction": (
                    float(np.mean(np.asarray(group_peer_values) <= 1))
                    if group_peer_values
                    else 0.0
                ),
            }
            level_rows.append(summary)
            group_rows.append(summary)
        broad_threshold = (
            max(
                10,
                math.ceil(
                    float(
                        np.quantile(
                            [row["active_group_size_max"] for row in level_rows], 0.90
                        )
                    )
                ),
            )
            if level_rows
            else 10
        )
        peer_values = peers[peer_column].to_list()
        audit[level] = {
            "distinct_groups": static.height,
            "accepted_group_size_distribution": _distribution(
                static["accepted_security_count"].to_list()
            ),
            "security_day_weighted": _peer_coverage(peer_values),
            "groups_frequently_with_zero_or_one_other_peer": [
                row
                for row in level_rows
                if row["zero_peer_fraction"] >= 0.25
                or row["at_most_one_peer_fraction"] >= 0.25
            ],
            "broad_group_threshold_max_active_size": broad_threshold,
            "abnormally_broad_groups": [
                row
                for row in level_rows
                if row["active_group_size_max"] >= broad_threshold
            ],
        }

    sector_counts = (
        classification.group_by("subsector")
        .agg(pl.col("sector").n_unique().alias("sector_count"))
        .sort("subsector")
    )
    violations = sector_counts.filter(pl.col("sector_count") != 1)
    audit["subsector_sector_nesting"] = {
        "every_subsector_belongs_to_exactly_one_sector": violations.is_empty(),
        "subsector_count": sector_counts.height,
        "violations": violations.to_dicts(),
    }

    sector = peers["same_sector_peer_count"].to_numpy()
    subsector = peers["same_subsector_peer_count"].to_numpy()
    policies = {
        "sector_only": _peer_coverage(sector),
        "subsector_only": _peer_coverage(subsector),
        "subsector_if_at_least_one_other_else_sector": _peer_coverage(
            np.where(subsector >= 1, subsector, sector)
        ),
        "subsector_if_at_least_two_others_else_sector": _peer_coverage(
            np.where(subsector >= 2, subsector, sector)
        ),
        "sector_and_subsector_simultaneous": {
            "security_day_count": peers.height,
            "sector_relation_percentage_with_at_least_one_peer": (
                float(np.mean(sector >= 1) * 100.0) if peers.height else 0.0
            ),
            "subsector_relation_percentage_with_at_least_one_peer": (
                float(np.mean(subsector >= 1) * 100.0) if peers.height else 0.0
            ),
            "both_relation_types_percentage_with_at_least_one_peer": (
                float(np.mean((sector >= 1) & (subsector >= 1)) * 100.0)
                if peers.height
                else 0.0
            ),
        },
    }
    audit["candidate_policy_coverage"] = policies
    group_sizes = pl.from_dicts(group_rows, infer_schema_length=None).sort(
        "level", "sector", "group"
    )
    return audit, group_sizes


def _single_member_periods(
    issuer_security_ids: Sequence[str],
    security_slots: dict[str, int],
    dates: Sequence[date],
    eligible_dates: np.ndarray,
    membership: np.ndarray,
) -> list[dict[str, str]]:
    slots = [security_slots[security_id] for security_id in issuer_security_ids]
    eligible_indices = np.flatnonzero(eligible_dates)
    observations: list[tuple[int, str]] = []
    for position, date_index in enumerate(eligible_indices):
        active_slots = [
            slot for slot in slots if bool(membership[int(date_index), slot])
        ]
        if len(active_slots) == 1:
            security_id = next(
                security_id
                for security_id in issuer_security_ids
                if security_slots[security_id] == active_slots[0]
            )
            observations.append((position, security_id))
    periods: list[dict[str, str]] = []
    if not observations:
        return periods
    start_position, current_security = observations[0]
    prior_position = start_position
    for position, security_id in observations[1:]:
        if position != prior_position + 1 or security_id != current_security:
            start_date_index = int(eligible_indices[start_position])
            end_date_index = int(eligible_indices[prior_position])
            periods.append(
                {
                    "security_id": current_security,
                    "from": str(dates[start_date_index]),
                    "to": str(dates[end_date_index]),
                }
            )
            start_position, current_security = position, security_id
        prior_position = position
    periods.append(
        {
            "security_id": current_security,
            "from": str(dates[int(eligible_indices[start_position])]),
            "to": str(dates[int(eligible_indices[prior_position])]),
        }
    )
    return periods


def issuer_peer_audit(
    peers: pl.DataFrame,
    security_metadata: pl.DataFrame,
    dates: Sequence[date],
    security_ids: Sequence[str],
    eligible_dates: np.ndarray,
    membership: np.ndarray,
) -> tuple[dict[str, object], pl.DataFrame]:
    mapped = security_metadata.filter(pl.col("issuer_id").is_not_null())
    groups = (
        mapped.group_by("issuer_id")
        .agg(
            pl.col("issuer_name").first(),
            pl.col("security_id").sort(),
            pl.col("ticker").sort(),
            pl.col("share_class").sort(),
        )
        .filter(pl.col("security_id").list.len() > 1)
        .sort("issuer_id")
    )
    security_slots = {
        security_id: slot for slot, security_id in enumerate(security_ids)
    }
    rows: list[dict[str, object]] = []
    for group in groups.to_dicts():
        group_security_ids = [str(value) for value in group["security_id"]]
        periods = _single_member_periods(
            group_security_ids,
            security_slots,
            dates,
            eligible_dates,
            membership,
        )
        rows.append(
            {
                "issuer_id": group["issuer_id"],
                "issuer_name": group["issuer_name"],
                "accepted_security_count": len(group_security_ids),
                "security_ids": " | ".join(group_security_ids),
                "tickers": " | ".join(str(value) for value in group["ticker"]),
                "share_classes": " | ".join(
                    str(value) for value in group["share_class"]
                ),
                "single_pit_member_period_count": len(periods),
                "single_pit_member_periods_json": json.dumps(periods),
            }
        )
    frame = (
        pl.from_dicts(rows, infer_schema_length=None).sort("issuer_id")
        if rows
        else pl.DataFrame(
            schema={
                "issuer_id": pl.String,
                "issuer_name": pl.String,
                "accepted_security_count": pl.Int64,
                "security_ids": pl.String,
                "tickers": pl.String,
                "share_classes": pl.String,
                "single_pit_member_period_count": pl.Int64,
                "single_pit_member_periods_json": pl.String,
            }
        )
    )
    counts = peers["same_issuer_peer_count"].to_numpy()
    audit = {
        "issuers_with_multiple_accepted_securities": groups.height,
        "security_day_coverage_with_at_least_one_active_same_issuer_peer_percent": (
            float(np.mean(counts >= 1) * 100.0) if counts.size else 0.0
        ),
        "security_day_weighted_peer_count_distribution": _distribution(counts),
        "exact_groups": frame.to_dicts(),
    }
    return audit, frame


def unit_overlap_audit(
    components: pl.DataFrame,
    security_ids: Sequence[str],
    eligible_dates: np.ndarray,
    membership: np.ndarray,
    data_ready: np.ndarray,
) -> tuple[dict[str, object], pl.DataFrame, pl.DataFrame]:
    slot_by_security = {
        security_id: slot for slot, security_id in enumerate(security_ids)
    }
    component_rows: list[dict[str, object]] = []
    for component in components.to_dicts():
        unit_id = component["unit_security_id"]
        component_id = component["component_security_id"]
        unit_slot = slot_by_security.get(str(unit_id)) if unit_id else None
        component_slot = (
            slot_by_security.get(str(component_id)) if component_id else None
        )
        if unit_slot is not None and component_slot is not None:
            both_members = (
                membership[:, unit_slot]
                & membership[:, component_slot]
                & eligible_dates
            )
            both_ready = (
                data_ready[:, unit_slot]
                & data_ready[:, component_slot]
                & eligible_dates
            )
            member_overlap = int(both_members.sum())
            ready_overlap = int(both_ready.sum())
            active_overlap = int((both_members & both_ready).sum())
        else:
            member_overlap = ready_overlap = active_overlap = 0
        component_rows.append(
            {
                **component,
                "component_pit_membership_overlap_date_count": member_overlap,
                "component_m1_readiness_overlap_date_count": ready_overlap,
                "component_fully_active_overlap_date_count": active_overlap,
            }
        )
    component_frame = pl.from_dicts(component_rows, infer_schema_length=None).sort(
        "unit_ticker", "component_share_class"
    )

    parity_rows: list[dict[str, object]] = []
    for group in components.partition_by("unit_ticker", maintain_order=True):
        rows = group.sort("component_share_class").to_dicts()
        unit_ids = sorted(
            {str(row["unit_security_id"]) for row in rows if row["unit_security_id"]}
        )
        unit_id = unit_ids[0] if len(unit_ids) == 1 else None
        mapped_components = [
            row
            for row in rows
            if row["mapping_status"] == "MAPPED"
            and row["component_security_id"] is not None
        ]
        problems = [
            f"{row['component_share_class']}:{row['mapping_status']}"
            for row in rows
            if row not in mapped_components
        ]
        pit_overlap = ready_overlap = fully_active_overlap = 0
        if unit_id is not None and len(mapped_components) == len(rows):
            component_ids = [
                str(row["component_security_id"]) for row in mapped_components
            ]
            slots = [slot_by_security[unit_id]] + [
                slot_by_security[component_id] for component_id in component_ids
            ]
            all_members = np.all(membership[:, slots], axis=1) & eligible_dates
            all_ready = np.all(data_ready[:, slots], axis=1) & eligible_dates
            pit_overlap = int(all_members.sum())
            ready_overlap = int(all_ready.sum())
            fully_active_overlap = int((all_members & all_ready).sum())

        if unit_id is None:
            limitation = "UNIT_NOT_IN_ACCEPTED_UNIVERSE"
        elif len(mapped_components) != len(rows):
            limitation = "MISSING_OR_AMBIGUOUS_COMPONENTS"
        elif pit_overlap == 0:
            limitation = "NO_SIMULTANEOUS_PIT_MEMBERSHIP"
        elif ready_overlap == 0:
            limitation = "NO_SIMULTANEOUS_M1_READINESS"
        elif fully_active_overlap == 0:
            limitation = "NO_SIMULTANEOUS_PIT_AND_M1_READY_DATE"
        else:
            limitation = ""
        parity_rows.append(
            {
                "unit_ticker": rows[0]["unit_ticker"],
                "unit_security_id": unit_id,
                "required_component_count": len(rows),
                "mapped_component_count": len(mapped_components),
                "missing_or_ambiguous_components": " | ".join(problems),
                "pit_all_component_overlap_date_count": pit_overlap,
                "m1_ready_all_component_overlap_date_count": ready_overlap,
                "fully_active_all_component_overlap_date_count": fully_active_overlap,
                "exact_parity_possible": fully_active_overlap > 0,
                "exact_parity_limitation": limitation,
            }
        )
    parity_frame = pl.from_dicts(parity_rows, infer_schema_length=None).sort(
        "unit_ticker"
    )
    present = parity_frame.filter(pl.col("unit_security_id").is_not_null())
    limitation_counts = {
        str(row["exact_parity_limitation"]): int(row["len"])
        for row in parity_frame.filter(~pl.col("exact_parity_possible"))
        .group_by("exact_parity_limitation")
        .len()
        .to_dicts()
    }
    audit = {
        "units_in_accepted_universe": present["unit_security_id"].n_unique(),
        "unit_tickers_in_accepted_universe": present["unit_ticker"]
        .unique()
        .sort()
        .to_list(),
        "component_rows": component_frame.height,
        "mapped_component_security_rows": int(
            component_frame["component_security_id"].is_not_null().sum()
        ),
        "unit_rows": parity_frame.height,
        "units_with_exact_parity_possible": int(
            parity_frame["exact_parity_possible"].sum()
        ),
        "units_with_exact_parity_impossible": int(
            (~parity_frame["exact_parity_possible"]).sum()
        ),
        "exact_parity_limitation_counts": limitation_counts,
    }
    return audit, component_frame, parity_frame


def _month_key(value: date) -> str:
    return f"{value.year:04d}-{value.month:02d}"


def _parse_month_key(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d{4})-(\d{2})", value)
    if match is None or not 1 <= int(match.group(2)) <= 12:
        raise ValueError(f"Invalid calendar-month key: {value!r}")
    return int(match.group(1)), int(match.group(2))


def _calendar_months(start: str, end: str) -> list[str]:
    year, month = _parse_month_key(start)
    end_year, end_month = _parse_month_key(end)
    values: list[str] = []
    while (year, month) <= (end_year, end_month):
        values.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return values


def _last_completed_month(as_of: date) -> str:
    year, month = (
        (as_of.year - 1, 12) if as_of.month == 1 else (as_of.year, as_of.month - 1)
    )
    return f"{year:04d}-{month:02d}"


def market_cap_month_coverage(
    reference_dates: Sequence[date],
    *,
    as_of: date,
    required_start_month: str = MARKET_CAP_REQUIRED_START_MONTH,
) -> dict[str, object]:
    available = sorted({_month_key(value) for value in reference_dates})
    required_end_month = _last_completed_month(as_of)
    expected = _calendar_months(required_start_month, required_end_month)
    missing = sorted(set(expected) - set(available))
    return {
        "required_start_month": required_start_month,
        "required_end_month": required_end_month,
        "distinct_reference_month_count": len(available),
        "available_reference_months": available,
        "first_reference_month": available[0] if available else None,
        "last_reference_month": available[-1] if available else None,
        "missing_reference_months": missing,
        "missing_reference_month_count": len(missing),
        "market_cap_history_complete": not missing,
    }


def market_cap_readiness(
    raw_reference_dates: Sequence[date],
    usable_reference_dates: Sequence[date],
    *,
    normalized_row_count: int,
    conflicting_issuer_month_groups: int,
    as_of: date,
) -> dict[str, object]:
    raw = market_cap_month_coverage(raw_reference_dates, as_of=as_of)
    usable = market_cap_month_coverage(usable_reference_dates, as_of=as_of)
    reasons: list[str] = []
    if not raw["market_cap_history_complete"]:
        reasons.append(
            "missing raw calendar months: " + ", ".join(raw["missing_reference_months"])
        )
    if not usable["market_cap_history_complete"]:
        reasons.append(
            "missing usable normalized calendar months: "
            + ", ".join(usable["missing_reference_months"])
        )
    if normalized_row_count == 0:
        reasons.append("normalized market-cap data is empty")
    if conflicting_issuer_month_groups:
        reasons.append(
            f"conflicting issuer-month groups: {conflicting_issuer_month_groups}"
        )
    ready = (
        bool(raw["market_cap_history_complete"])
        and bool(usable["market_cap_history_complete"])
        and normalized_row_count > 0
        and conflicting_issuer_month_groups == 0
    )
    return {
        "required_start_month": raw["required_start_month"],
        "required_end_month": raw["required_end_month"],
        "raw_distinct_reference_month_count": raw["distinct_reference_month_count"],
        "raw_available_reference_months": raw["available_reference_months"],
        "raw_first_reference_month": raw["first_reference_month"],
        "raw_last_reference_month": raw["last_reference_month"],
        "raw_missing_reference_months": raw["missing_reference_months"],
        "raw_missing_reference_month_count": raw["missing_reference_month_count"],
        "raw_market_cap_history_complete": raw["market_cap_history_complete"],
        "usable_distinct_reference_month_count": usable[
            "distinct_reference_month_count"
        ],
        "usable_available_reference_months": usable["available_reference_months"],
        "usable_first_reference_month": usable["first_reference_month"],
        "usable_last_reference_month": usable["last_reference_month"],
        "usable_missing_reference_months": usable["missing_reference_months"],
        "usable_missing_reference_month_count": usable["missing_reference_month_count"],
        "usable_market_cap_history_complete": usable["market_cap_history_complete"],
        "normalized_market_cap_row_count": normalized_row_count,
        "market_cap_data_ready": ready,
        "market_cap_not_ready_reasons": reasons,
    }


def market_cap_audit(
    peers: pl.DataFrame,
    market_cap: pl.DataFrame,
    security_metadata: pl.DataFrame,
    mapping_stats: dict[str, object],
    *,
    as_of: date,
    source_reference_dates: Sequence[date] | None = None,
) -> tuple[dict[str, object], pl.DataFrame]:
    cap_by_issuer: dict[str, list[tuple[date, float]]] = defaultdict(list)
    for row in market_cap.sort("reference_date").to_dicts():
        cap_by_issuer[str(row["issuer_id"])].append(
            (row["reference_date"], float(row["market_cap_brl"]))
        )
    active_counts: dict[str, int] = defaultdict(int)
    joined_counts: dict[str, int] = defaultdict(int)
    staleness_by_issuer: dict[str, list[int]] = defaultdict(list)
    joined_by_security: dict[str, int] = defaultdict(int)
    active_by_security: dict[str, int] = defaultdict(int)
    all_staleness: list[int] = []
    for row in peers.select("trade_date", "security_id", "issuer_id").to_dicts():
        security_id = str(row["security_id"])
        active_by_security[security_id] += 1
        issuer_id = row["issuer_id"]
        if issuer_id is None:
            continue
        issuer_id = str(issuer_id)
        active_counts[issuer_id] += 1
        observations = cap_by_issuer.get(issuer_id, [])
        reference_dates = [value[0] for value in observations]
        index = strictly_lagged_market_cap_index(reference_dates, row["trade_date"])
        if index is None:
            continue
        reference_date = observations[index][0]
        staleness = (row["trade_date"] - reference_date).days
        joined_counts[issuer_id] += 1
        joined_by_security[security_id] += 1
        staleness_by_issuer[issuer_id].append(staleness)
        all_staleness.append(staleness)

    issuer_names = {
        str(row["issuer_id"]): str(row["issuer_name"])
        for row in security_metadata.filter(pl.col("issuer_id").is_not_null())
        .select("issuer_id", "issuer_name")
        .unique("issuer_id")
        .to_dicts()
    }
    coverage_rows: list[dict[str, object]] = []
    discontinuities: list[dict[str, object]] = []
    for issuer_id, issuer_name in sorted(issuer_names.items()):
        observations = cap_by_issuer.get(issuer_id, [])
        references = sorted(value[0] for value in observations)
        internal_missing: list[str] = []
        if references:
            observed_months = sorted({_month_key(value) for value in references})
            expected_internal = _calendar_months(
                observed_months[0], observed_months[-1]
            )
            internal_missing = sorted(set(expected_internal) - set(observed_months))
            if internal_missing:
                discontinuities.append(
                    {
                        "issuer_id": issuer_id,
                        "issuer_name": issuer_name,
                        "missing_months_between_first_and_last": [
                            value for value in internal_missing
                        ],
                    }
                )
        active_count = active_counts.get(issuer_id, 0)
        joined_count = joined_counts.get(issuer_id, 0)
        stale = staleness_by_issuer.get(issuer_id, [])
        coverage_rows.append(
            {
                "issuer_id": issuer_id,
                "issuer_name": issuer_name,
                "market_cap_month_count": len(references),
                "first_reference_date": references[0] if references else None,
                "last_reference_date": references[-1] if references else None,
                "internal_missing_month_count": len(internal_missing),
                "active_security_day_count": active_count,
                "strictly_lagged_joined_security_day_count": joined_count,
                "security_day_coverage_fraction": (
                    joined_count / active_count if active_count else 0.0
                ),
                "median_staleness_days": (float(np.median(stale)) if stale else None),
                "p90_staleness_days": (
                    float(np.quantile(stale, 0.90)) if stale else None
                ),
            }
        )
    coverage = pl.from_dicts(coverage_rows, infer_schema_length=None).sort("issuer_id")
    available_reference_dates = (
        market_cap["reference_date"].unique().sort().to_list()
        if market_cap.height
        else []
    )
    readiness = market_cap_readiness(
        (
            available_reference_dates
            if source_reference_dates is None
            else source_reference_dates
        ),
        available_reference_dates,
        normalized_row_count=market_cap.height,
        conflicting_issuer_month_groups=int(
            mapping_stats["conflicting_issuer_month_groups"]
        ),
        as_of=as_of,
    )
    total_active = peers.height
    total_joined = sum(joined_by_security.values())
    no_cap_securities = sorted(
        security_id
        for security_id, active_count in active_by_security.items()
        if active_count and joined_by_security.get(security_id, 0) == 0
    )
    audit = {
        **mapping_stats,
        **readiness,
        "available_reference_dates": [
            str(value) for value in available_reference_dates
        ],
        "eligible_security_day_count": total_active,
        "strictly_lagged_joined_security_day_count": total_joined,
        "strictly_lagged_security_day_coverage_fraction": (
            total_joined / total_active if total_active else 0.0
        ),
        "staleness_days_distribution": _distribution(all_staleness),
        "duplicate_or_conflicting_observations": {
            "duplicate_issuer_month_groups": mapping_stats[
                "duplicate_issuer_month_groups"
            ],
            "conflicting_issuer_month_groups": mapping_stats[
                "conflicting_issuer_month_groups"
            ],
        },
        "issuer_discontinuities": discontinuities,
        "issuers_with_no_usable_historical_market_cap": [
            row["issuer_id"]
            for row in coverage_rows
            if row["active_security_day_count"]
            and not row["strictly_lagged_joined_security_day_count"]
        ],
        "securities_with_no_usable_historical_market_cap": no_cap_securities,
        "strict_lag_rule": (
            "Use the latest issuer-level reference_date strictly before the model "
            "trade_date; same-date and future observations are never used."
        ),
    }
    return audit, coverage


def _render_markdown(audit: dict[str, object]) -> str:
    classification = audit["classification_and_peer_coverage"]
    policies = classification["candidate_policy_coverage"]
    status = (
        "> **COMPLETE:** Official B3 market-cap data passed raw and normalized "
        "publication-readiness checks."
        if audit["market_cap_data_ready"]
        else (
            "> **DIAGNOSTIC ONLY — MARKET-CAP DATA NOT READY:** This artifact is not "
            "eligible for downstream market-cap features and is not a complete PIT "
            "metadata audit."
        )
    )
    lines = [
        "# B3 human-prior metadata audit",
        "",
        status,
        "",
        "This audit uses the current B3 classification and unit snapshots as descriptive "
        "metadata across the historical PIT universe. It does not treat either snapshot "
        "as point-in-time historical truth and does not select a sector/subsector policy.",
        "",
        "## Scope",
        "",
        f"- Eligible model dates: {audit['scope']['eligible_model_date_count']}",
        f"- Eligible security-days: {audit['scope']['eligible_security_day_count']}",
        f"- Accepted securities: {audit['scope']['accepted_security_count']}",
        f"- Exactly classified securities: {audit['scope']['mapped_security_count']}",
        f"- Mapping exceptions: {audit['mapping_exception_count']}",
        "",
        "## Peer evidence",
        "",
        "| Relation | Groups | Median peers | P10 | P90 | ≥1 peer | ≥2 peers | ≥3 peers | ≥5 peers |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for level in ("sector", "subsector", "economic_segment"):
        summary = classification[level]
        weighted = summary["security_day_weighted"]
        distribution = weighted["peer_count_distribution"]
        thresholds = weighted["percentage_with_at_least"]
        lines.append(
            f"| {level} | {summary['distinct_groups']} | "
            f"{distribution['median']:.2f} | {distribution['p10']:.2f} | "
            f"{distribution['p90']:.2f} | {thresholds['1']:.2f}% | "
            f"{thresholds['2']:.2f}% | {thresholds['3']:.2f}% | "
            f"{thresholds['5']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "Subsector-to-sector nesting is one-to-one: "
            f"**{classification['subsector_sector_nesting']['every_subsector_belongs_to_exactly_one_sector']}**.",
            "",
            "## Candidate-policy coverage (evidence only)",
            "",
            "| Candidate | ≥1 peer | ≥2 peers | ≥3 peers | ≥5 peers |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in (
        "sector_only",
        "subsector_only",
        "subsector_if_at_least_one_other_else_sector",
        "subsector_if_at_least_two_others_else_sector",
    ):
        thresholds = policies[name]["percentage_with_at_least"]
        lines.append(
            f"| {name} | {thresholds['1']:.2f}% | {thresholds['2']:.2f}% | "
            f"{thresholds['3']:.2f}% | {thresholds['5']:.2f}% |"
        )
    simultaneous = policies["sector_and_subsector_simultaneous"]
    lines.extend(
        [
            "",
            "For simultaneous relation types, sector coverage with at least one peer is "
            f"{simultaneous['sector_relation_percentage_with_at_least_one_peer']:.2f}%, "
            "subsector coverage is "
            f"{simultaneous['subsector_relation_percentage_with_at_least_one_peer']:.2f}%, "
            "and both are available on "
            f"{simultaneous['both_relation_types_percentage_with_at_least_one_peer']:.2f}% "
            "of eligible security-days.",
            "",
            "## Same issuer",
            "",
            f"- Issuers with multiple accepted securities: "
            f"{audit['same_issuer']['issuers_with_multiple_accepted_securities']}",
            "- Eligible security-day coverage with an active same-issuer peer: "
            f"{audit['same_issuer']['security_day_coverage_with_at_least_one_active_same_issuer_peer_percent']:.2f}%",
            "",
            "## Units",
            "",
            f"- Units in the accepted universe: {audit['units']['units_in_accepted_universe']}",
            f"- Units with exact parity possible: "
            f"{audit['units']['units_with_exact_parity_possible']}",
            f"- Units with exact parity impossible: {audit['units']['units_with_exact_parity_impossible']}",
            "",
            "## Monthly market capitalization",
            "",
            f"- Issuer-name mapping coverage: "
            f"{audit['market_cap']['issuer_name_mapping_fraction']:.2%}",
            f"- Strictly lagged eligible security-day coverage: "
            f"{audit['market_cap']['strictly_lagged_security_day_coverage_fraction']:.2%}",
            f"- Raw calendar-month history complete: "
            f"{audit['market_cap']['raw_market_cap_history_complete']}",
            f"- Raw available calendar months: "
            f"{audit['market_cap']['raw_distinct_reference_month_count']}",
            f"- Missing raw calendar months: "
            f"{audit['market_cap']['raw_missing_reference_month_count']}",
            f"- Usable normalized calendar-month history complete: "
            f"{audit['market_cap']['usable_market_cap_history_complete']}",
            f"- Usable normalized calendar months: "
            f"{audit['market_cap']['usable_distinct_reference_month_count']}",
            f"- Missing usable normalized calendar months: "
            f"{audit['market_cap']['usable_missing_reference_month_count']}",
            f"- Usable normalized observations: "
            f"{audit['market_cap']['normalized_market_cap_row_count']}",
            f"- Conflicting issuer-month groups excluded: "
            f"{audit['market_cap']['conflicting_issuer_month_groups']}",
            f"- Market-cap data ready: {audit['market_cap_data_ready']}",
            "",
            "See the CSV outputs for exact groups, questionable classifications, unit "
            "overlaps, market-cap coverage, and unresolved deterministic mappings.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_raw_sources(
    raw_dir: Path = RAW_BASE,
    *,
    as_of: date | None = None,
    inputs: FeatureInputs | None = None,
    assignments_loader: Callable[[Path], pl.DataFrame] = load_assignments,
) -> dict[str, object]:
    as_of = date.today() if as_of is None else as_of
    inputs = _resolve_feature_inputs() if inputs is None else inputs
    _validate_feature_inputs(inputs)
    classification, raw_market_cap, units, sources = _load_raw_tables(raw_dir)
    assignments = assignments_loader(inputs.assignments_dir)
    ticker_history = pl.read_parquet(inputs.universe_dir / "ticker_history.parquet")
    security_metadata, _, security_exceptions = reconcile_security_metadata(
        assignments, ticker_history, classification
    )
    market_cap, market_exceptions, market_stats = reconcile_market_cap(
        raw_market_cap, classification, security_metadata
    )
    accepted_issuers = set(
        security_metadata.filter(pl.col("issuer_id").is_not_null())[
            "issuer_id"
        ].to_list()
    )
    market_issuers = set(market_cap["issuer_id"].to_list())
    readiness = market_cap_readiness(
        raw_market_cap["reference_date"].to_list(),
        market_cap["reference_date"].to_list(),
        normalized_row_count=market_cap.height,
        conflicting_issuer_month_groups=int(
            market_stats["conflicting_issuer_month_groups"]
        ),
        as_of=as_of,
    )
    return {
        "build_mode_if_built_now": (
            "complete"
            if readiness["market_cap_data_ready"]
            else "diagnostic_market_cap_not_ready"
        ),
        "eligible_for_strict_build": readiness["market_cap_data_ready"],
        "classification": {
            "row_count": classification.height,
            "issuer_count": classification["issuer_id"].n_unique(),
        },
        "market_cap": {
            "row_count": raw_market_cap.height,
            "source_issuer_count": raw_market_cap["issuer_name"].n_unique(),
            "all_company_footer_validated": True,
            **readiness,
            "mapped_accepted_universe_issuer_count": len(
                accepted_issuers & market_issuers
            ),
            "mapping_exception_count": market_exceptions.height,
            "duplicate_issuer_month_groups": market_stats[
                "duplicate_issuer_month_groups"
            ],
            "conflicting_issuer_month_groups": market_stats[
                "conflicting_issuer_month_groups"
            ],
        },
        "units": {
            "unit_count": units["unit_ticker"].n_unique(),
            "component_count": units.height,
        },
        "security_mapping_exception_count": security_exceptions.height,
        "sources": [
            {
                "kind": source.kind,
                "path": str(source.path.resolve()),
                "source_url": source.source_url,
                "sha256": source.sha256,
                "retrieved_at_utc": source.retrieved_at_utc.isoformat(),
                "reference_dates": [str(value) for value in source.reference_dates],
            }
            for source in sources
        ],
        "suspicious_truncation_or_schema_mismatch": [],
    }


def _repository_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Cannot locate the repository root")


def _repository_state() -> tuple[str, list[str]]:
    repository = _repository_root()
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return commit, status


def _validate_feature_inputs(inputs: FeatureInputs) -> None:
    manifest = json.loads(
        (inputs.feature_store / "manifest.json").read_text(encoding="utf-8")
    )
    expected = {
        "point_in_time_universe": inputs.universe_dir,
        "accepted_xp_assignments": inputs.assignments_dir,
        "parsed_cotahist": inputs.cotahist_dir,
    }
    recorded = manifest.get("canonical_inputs", {})
    for name, path in expected.items():
        recorded_path = Path(recorded.get(name, {}).get("resolved_path", ""))
        if recorded_path.resolve() != path.resolve():
            raise ValueError(
                f"Canonical feature store {name} does not match the current pointer"
            )


def _resolve_feature_inputs() -> FeatureInputs:
    inputs = FeatureInputs(
        universe_dir=resolve_pointer(UNIVERSE_POINTER),
        assignments_dir=resolve_pointer(ASSIGNMENTS_POINTER),
        cotahist_dir=resolve_pointer(COTAHIST_POINTER),
        feature_store=resolve_pointer(CANONICAL_OUTPUT_POINTER),
    )
    _validate_feature_inputs(inputs)
    return inputs


def _load_raw_tables(
    raw_dir: Path,
) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    tuple[RawSource, ...],
]:
    classification_sources = discover_raw_sources(raw_dir, "classification")
    units_sources = discover_raw_sources(raw_dir, "units")
    market_sources = discover_raw_sources(raw_dir, "market_cap")
    if not classification_sources or not units_sources or not market_sources:
        raise FileNotFoundError(
            "B3 human-prior raw cache is incomplete. Run the acquire command first."
        )
    classification_source = classification_sources[-1]
    units_source = units_sources[-1]
    classification = parse_classification_xlsx(
        classification_source.path.read_bytes(),
        classification_source.retrieved_at_utc.date(),
        source_file=str(classification_source.path.resolve()),
        source_url=classification_source.source_url,
        retrieved_at_utc=classification_source.retrieved_at_utc,
    )
    units = parse_units_html(
        units_source.path.read_bytes(),
        units_source.retrieved_at_utc.date(),
        source_file=str(units_source.path.resolve()),
        source_url=units_source.source_url,
        retrieved_at_utc=units_source.retrieved_at_utc,
    )
    market = pl.concat(
        [
            parse_market_cap_csv(
                source.path.read_bytes(),
                source_file=str(source.path.resolve()),
                source_url=source.source_url,
                retrieved_at_utc=source.retrieved_at_utc,
            )
            for source in market_sources
        ],
        how="vertical_relaxed",
    )
    return (
        classification,
        market,
        units,
        (classification_source, *market_sources, units_source),
    )


def _manifest_input(
    pointer: Path,
    resolved: Path,
    artifact_names: Sequence[str] = (),
) -> dict[str, object]:
    manifest = next(
        (
            path
            for name in ("manifest.json", "decision_manifest.json")
            if (path := resolved / name).is_file()
        ),
        None,
    )
    artifacts = list(artifact_names)
    if manifest is not None and manifest.name not in artifacts:
        artifacts.append(manifest.name)
    missing = [name for name in artifacts if not (resolved / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Canonical input {resolved} is missing lineage files: {missing}"
        )
    return {
        "pointer": str(pointer),
        "pointer_sha256": _sha256_file(pointer),
        "resolved_path": str(resolved.resolve()),
        "resolved_manifest": str(manifest.resolve()) if manifest else "",
        "artifact_sha256": {
            name: _sha256_file(resolved / name) for name in sorted(artifacts)
        },
    }


def build_human_priors(
    raw_dir: Path = RAW_BASE,
    output_base: Path = OUTPUT_BASE,
    pointer: Path = CANONICAL_POINTER,
    *,
    created_at: datetime | None = None,
    allow_incomplete_market_cap: bool = False,
    inputs: FeatureInputs | None = None,
    assignments_loader: Callable[[Path], pl.DataFrame] = load_assignments,
    repository_state: Callable[[], tuple[str, list[str]]] = _repository_state,
) -> Path:
    created_at = datetime.now(UTC) if created_at is None else created_at.astimezone(UTC)
    inputs = _resolve_feature_inputs() if inputs is None else inputs
    _validate_feature_inputs(inputs)
    classification, raw_market_cap, raw_units, raw_sources = _load_raw_tables(raw_dir)
    assignments = assignments_loader(inputs.assignments_dir)
    ticker_history = pl.read_parquet(inputs.universe_dir / "ticker_history.parquet")
    security_metadata, classification_review, security_exceptions = (
        reconcile_security_metadata(assignments, ticker_history, classification)
    )
    market_cap, market_exceptions, market_stats = reconcile_market_cap(
        raw_market_cap, classification, security_metadata
    )
    accepted_issuer_ids = set(
        security_metadata.filter(pl.col("issuer_id").is_not_null())[
            "issuer_id"
        ].to_list()
    )
    market_stats["mapped_accepted_universe_issuer_count"] = len(
        accepted_issuer_ids & set(market_cap["issuer_id"].to_list())
    )
    unit_components, unit_exceptions = normalize_unit_components(
        raw_units, classification, security_metadata
    )
    mapping_exceptions = pl.concat(
        [security_exceptions, market_exceptions, unit_exceptions],
        how="vertical_relaxed",
    ).sort("source_type", "source_key", "candidate_issuer_id")

    date_index = pl.read_parquet(inputs.feature_store / "date_index.parquet").sort(
        "date_idx"
    )
    equity_index = pl.read_parquet(inputs.feature_store / "equity_index.parquet").sort(
        "equity_slot"
    )
    dates = date_index["trade_date"].to_list()
    security_ids = equity_index["security_id"].to_list()
    if security_ids != assignments["security_id"].to_list():
        raise ValueError("Feature equity axis does not match accepted assignments")
    membership = np.load(inputs.feature_store / "equity_membership.npy", mmap_mode="r")
    data_ready = np.load(inputs.feature_store / "equity_data_ready.npy", mmap_mode="r")
    active, eligible_dates = active_security_days(
        dates, security_ids, membership, data_ready
    )
    feature_manifest = json.loads(
        (inputs.feature_store / "manifest.json").read_text(encoding="utf-8")
    )
    if int(eligible_dates.sum()) != int(feature_manifest["eligible_date_count"]):
        raise ValueError(
            "Recomputed eligible model dates do not match feature manifest"
        )
    peers = add_self_excluded_peer_counts(active, security_metadata)
    classification_audit, group_sizes = classification_peer_audit(
        peers, security_metadata, classification
    )
    issuer_audit, issuer_groups = issuer_peer_audit(
        peers,
        security_metadata,
        dates,
        security_ids,
        eligible_dates,
        membership,
    )
    units_audit, unit_overlap, unit_parity = unit_overlap_audit(
        unit_components,
        security_ids,
        eligible_dates,
        membership,
        data_ready,
    )
    market_audit, market_coverage = market_cap_audit(
        peers,
        market_cap,
        security_metadata,
        market_stats,
        as_of=created_at.date(),
        source_reference_dates=raw_market_cap["reference_date"].to_list(),
    )
    market_cap_data_ready = bool(market_audit["market_cap_data_ready"])
    if not market_cap_data_ready and not allow_incomplete_market_cap:
        raise ValueError(
            "Official B3 market-cap data is not ready for publication for "
            f"{market_audit['required_start_month']} through "
            f"{market_audit['required_end_month']}: "
            f"{'; '.join(market_audit['market_cap_not_ready_reasons'])}. "
            "Batch-ingest or correct official files, then rerun, or use "
            "--allow-incomplete-market-cap for a diagnostic-only build."
        )
    build_mode = (
        "complete" if market_cap_data_ready else "diagnostic_market_cap_not_ready"
    )
    audit = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": created_at.isoformat(),
        "build_mode": build_mode,
        "raw_market_cap_history_complete": market_audit[
            "raw_market_cap_history_complete"
        ],
        "usable_market_cap_history_complete": market_audit[
            "usable_market_cap_history_complete"
        ],
        "market_cap_data_ready": market_cap_data_ready,
        "eligible_for_downstream_market_cap_features": market_cap_data_ready,
        "scope": {
            "accepted_security_count": len(security_ids),
            "eligible_model_date_count": int(eligible_dates.sum()),
            "eligible_security_day_count": peers.height,
            "mapped_security_count": int(
                (security_metadata["mapping_status"] == "MAPPED").sum()
            ),
            "classification_snapshot_date": str(
                classification["classification_snapshot_date"][0]
            ),
            "units_snapshot_date": str(raw_units["unit_snapshot_date"][0]),
            "historical_classification_caveat": (
                "The current B3 classification and unit snapshots are descriptive "
                "audit metadata, not point-in-time historical truth."
            ),
        },
        "classification_and_peer_coverage": classification_audit,
        "questionable_or_ambiguous_security_count": int(
            (classification_review["mapping_status"] != "MAPPED").sum()
        ),
        "same_issuer": issuer_audit,
        "units": units_audit,
        "market_cap": market_audit,
        "mapping_exception_count": mapping_exceptions.height,
        "grouping_policy_selected": None,
    }

    output_dir = output_base / (
        f"human_priors_{build_mode}_{created_at:%Y%m%dT%H%M%S%fZ}"
    )
    partial = output_dir.with_name(f"{output_dir.name}.partial")
    if output_dir.exists() or partial.exists():
        raise FileExistsError(f"B3 human-prior output already exists: {output_dir}")
    partial.mkdir(parents=True)
    try:
        security_metadata.write_parquet(
            partial / "security_metadata.parquet",
            compression="zstd",
            statistics=True,
        )
        market_cap.write_parquet(
            partial / "issuer_market_cap_monthly.parquet",
            compression="zstd",
            statistics=True,
        )
        unit_components.write_parquet(
            partial / "unit_components.parquet",
            compression="zstd",
            statistics=True,
        )
        _atomic_json(partial / "metadata_audit.json", audit)
        (partial / "metadata_audit.md").write_text(
            _render_markdown(audit), encoding="utf-8"
        )
        group_sizes.write_csv(partial / "sector_subsector_group_sizes.csv")
        classification_review.write_csv(partial / "security_classification_review.csv")
        issuer_groups.write_csv(partial / "issuer_peer_groups.csv")
        unit_overlap.write_csv(partial / "unit_overlap_audit.csv")
        unit_parity.write_csv(partial / "unit_parity_coverage.csv")
        market_coverage.write_csv(partial / "market_cap_coverage.csv")
        mapping_exceptions.write_csv(partial / "mapping_exceptions.csv")

        commit, status = repository_state()
        outputs = sorted(
            path for path in partial.iterdir() if path.name != "manifest.json"
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": created_at.isoformat(),
            "repository_commit": commit,
            "repository_status_porcelain": status,
            "implementation_sha256": _sha256_file(Path(__file__).resolve()),
            "build_mode": build_mode,
            "raw_market_cap_history_complete": market_audit[
                "raw_market_cap_history_complete"
            ],
            "raw_missing_reference_months": market_audit[
                "raw_missing_reference_months"
            ],
            "usable_market_cap_history_complete": market_audit[
                "usable_market_cap_history_complete"
            ],
            "usable_missing_reference_months": market_audit[
                "usable_missing_reference_months"
            ],
            "normalized_market_cap_row_count": market_audit[
                "normalized_market_cap_row_count"
            ],
            "conflicting_issuer_month_groups": market_audit[
                "conflicting_issuer_month_groups"
            ],
            "market_cap_data_ready": market_cap_data_ready,
            "eligible_for_downstream_market_cap_features": market_cap_data_ready,
            "canonical_pointer_published": market_cap_data_ready,
            "official_b3_pages": {
                "classification": CLASSIFICATION_PAGE_URL,
                "market_cap_monthly": MARKET_CAP_PAGE_URL,
                "units": UNITS_PAGE_URL,
            },
            "official_b3_endpoints": {
                "classification_download_base": CLASSIFICATION_API_BASE,
                "classification_request": {"language": "pt-br"},
                "market_cap_current_download_base": MARKET_CAP_API_BASE,
                "market_cap_current_request": {
                    "company": "",
                    "language": "pt-br",
                    "keyword": "",
                    "pageNumber": 1,
                    "pageSize": 20,
                },
                "units_direct_page": UNITS_PAGE_URL,
            },
            "market_cap_historical_acquisition_limitation": MARKET_CAP_HISTORY_LIMITATION,
            "market_cap_export_completeness_rule": (
                "Every CSV must contain more than 20 distinct company rows and a "
                "Total Geral (N) footer whose declared N exactly matches those rows; "
                "otherwise it is rejected as truncated."
            ),
            "canonical_inputs": {
                "point_in_time_universe": _manifest_input(
                    inputs.universe_pointer,
                    inputs.universe_dir,
                    ("ticker_history.parquet",),
                ),
                "accepted_xp_assignments": _manifest_input(
                    inputs.assignments_pointer,
                    inputs.assignments_dir,
                    ("xp_accepted_source_assignments_v1.parquet",),
                ),
                "parsed_cotahist": _manifest_input(
                    inputs.cotahist_pointer,
                    inputs.cotahist_dir,
                    ("parse_audit.json",),
                ),
                "feature_store": _manifest_input(
                    inputs.feature_store_pointer,
                    inputs.feature_store,
                    (
                        "date_index.parquet",
                        "equity_index.parquet",
                        "equity_membership.npy",
                        "equity_data_ready.npy",
                    ),
                ),
            },
            "raw_sources": [
                {
                    "kind": source.kind,
                    "path": str(source.path.resolve()),
                    "source_url": source.source_url,
                    "retrieved_at_utc": source.retrieved_at_utc.isoformat(),
                    "content_type": source.content_type,
                    "sha256": source.sha256,
                    "bytes": source.bytes,
                    "reference_dates": [str(value) for value in source.reference_dates],
                    "acquisition_method": source.acquisition_method,
                }
                for source in raw_sources
            ],
            "normalized_schemas": {
                "security_metadata": str(security_metadata.schema),
                "issuer_market_cap_monthly": str(market_cap.schema),
                "unit_components": str(unit_components.schema),
                "unit_parity_coverage": str(unit_parity.schema),
            },
            "classification_is_point_in_time": False,
            "grouping_policy_selected": None,
            "output_dir": str(output_dir.resolve()),
            "output_sha256": {path.name: _sha256_file(path) for path in outputs},
            "artifact_status": {path.name: build_mode for path in outputs},
        }
        _atomic_json(partial / "manifest.json", manifest)
        os.replace(partial, output_dir)
        if market_cap_data_ready:
            _atomic_pointer(pointer, output_dir)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return output_dir


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Acquire and audit official B3 human-prior metadata."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    acquire = subparsers.add_parser(
        "acquire", help="Cache the current official B3 sources."
    )
    acquire.add_argument("--raw-dir", type=Path, default=RAW_BASE)
    acquire.add_argument("--refresh", action="store_true")
    ingest = subparsers.add_parser(
        "ingest-market-cap",
        help="Validate and cache a manually downloaded official B3 monthly CSV.",
    )
    ingest.add_argument("--file", type=Path, required=True)
    ingest.add_argument("--raw-dir", type=Path, default=RAW_BASE)
    build = subparsers.add_parser(
        "build", help="Normalize the cache and generate the PIT metadata audit."
    )
    build.add_argument("--raw-dir", type=Path, default=RAW_BASE)
    build.add_argument("--output-base", type=Path, default=OUTPUT_BASE)
    build.add_argument("--pointer", type=Path, default=CANONICAL_POINTER)
    build.add_argument(
        "--allow-incomplete-market-cap",
        action="store_true",
        help=(
            "Create a diagnostic-only artifact instead of failing market-cap "
            "publication-readiness checks."
        ),
    )
    ingest_directory = subparsers.add_parser(
        "ingest-market-cap-dir",
        help="Prevalidate and batch-cache official B3 monthly CSV exports.",
    )
    ingest_directory.add_argument("--directory", type=Path, required=True)
    ingest_directory.add_argument("--raw-dir", type=Path, default=RAW_BASE)
    validate = subparsers.add_parser(
        "validate-raw", help="Validate the existing raw cache without network access."
    )
    validate.add_argument("--raw-dir", type=Path, default=RAW_BASE)
    instructions = subparsers.add_parser(
        "instructions", help="Print official manual market-cap instructions."
    )
    instructions.add_argument("--raw-dir", type=Path, default=RAW_BASE)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    if args.command == "acquire":
        acquired = acquire_official_sources(args.raw_dir, refresh=args.refresh)
        for kind, source in acquired.items():
            print(f"{kind}: {source.path} sha256={source.sha256}")
        print(json.dumps(validate_raw_sources(args.raw_dir), indent=2))
        print(market_cap_manual_instructions(args.raw_dir))
    elif args.command == "ingest-market-cap":
        source = ingest_manual_market_cap(args.file, args.raw_dir)
        print(f"market_cap: {source.path} sha256={source.sha256}")
        print(json.dumps(validate_raw_sources(args.raw_dir), indent=2))
    elif args.command == "ingest-market-cap-dir":
        sources = ingest_market_cap_directory(args.directory, args.raw_dir)
        print(f"validated_and_cached_files: {len(sources)}")
        print(json.dumps(validate_raw_sources(args.raw_dir), indent=2))
    elif args.command == "validate-raw":
        print(json.dumps(validate_raw_sources(args.raw_dir), indent=2))
    elif args.command == "build":
        output = build_human_priors(
            raw_dir=args.raw_dir,
            output_base=args.output_base,
            pointer=args.pointer,
            allow_incomplete_market_cap=args.allow_incomplete_market_cap,
        )
        audit = json.loads((output / "metadata_audit.json").read_text(encoding="utf-8"))
        market = audit["market_cap"]
        print(output)
        print(
            json.dumps(
                {
                    "raw_market_cap_reference_months": market[
                        "raw_distinct_reference_month_count"
                    ],
                    "raw_missing_market_cap_calendar_months": market[
                        "raw_missing_reference_months"
                    ],
                    "raw_market_cap_history_complete": market[
                        "raw_market_cap_history_complete"
                    ],
                    "usable_market_cap_reference_months": market[
                        "usable_distinct_reference_month_count"
                    ],
                    "usable_missing_market_cap_calendar_months": market[
                        "usable_missing_reference_months"
                    ],
                    "usable_market_cap_history_complete": market[
                        "usable_market_cap_history_complete"
                    ],
                    "source_issuer_count": market["source_distinct_issuer_names"],
                    "mapped_accepted_universe_issuer_count": market[
                        "mapped_accepted_universe_issuer_count"
                    ],
                    "normalized_market_cap_row_count": market[
                        "normalized_market_cap_row_count"
                    ],
                    "conflicting_issuer_month_groups": market[
                        "conflicting_issuer_month_groups"
                    ],
                    "build_mode": audit["build_mode"],
                    "market_cap_data_ready": audit["market_cap_data_ready"],
                    "eligible_for_downstream_market_cap_features": audit[
                        "eligible_for_downstream_market_cap_features"
                    ],
                },
                indent=2,
            )
        )
    else:
        print(market_cap_manual_instructions(args.raw_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
