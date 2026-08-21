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
import urllib.error
import urllib.request
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
from pypdf import PdfReader

CONTRACT_VERSION = "B3_BDI_LENDING_OPEN_BALANCE_V1"
RAW_CONTRACT_VERSION = "B3_BDI_CHAPTER_05_PDF_SNAPSHOT_V1"
PDF_URL = "https://arquivos.b3.com.br/bdi/download/bdi/{day}/BDI_05_{compact}.pdf"
VALID_ISIN = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")
MODERN_ROW = re.compile(
    r"^\s*(?P<day>\d{2}/\d{2}/\d{4})\s+"
    r"(?P<ticker>[A-Z0-9]{4,12})\s+"
    r"(?P<isin>[A-Z]{2}[A-Z0-9]{9}[0-9])\s+"
    r"(?P<body>.*?)\s+"
    r"(?P<quantity>[0-9][0-9.,]*)\s+"
    r"(?P<price>-|[0-9][0-9.,]*)\s+"
    r"(?P<balance>[0-9][0-9.,]*)\s*$"
)
LEGACY_ROW = re.compile(
    r"^\s*(?P<ticker>[A-Z0-9]{4,12})\s+.*?\s+"
    r"(?P<quantity>[0-9][0-9.]*)\s+"
    r"(?P<balance>[0-9][0-9.,]*)\s*$"
)
MIN_PARSED_POSITIONS = 100
MIN_MAPPED_NONZERO_POSITIONS = 25
ADV_SESSIONS = 20
CHANGE_SESSIONS = (5, 20)
FEATURES = (
    "lending_balance_days_to_cover_log_tanh",
    "lending_balance_days_to_cover_change_5_tanh",
    "lending_balance_days_to_cover_change_20_tanh",
)


@dataclass(frozen=True)
class Identity:
    security_id: str
    isin: str
    effective_from: date
    effective_to_inclusive: date


@dataclass(frozen=True)
class StockDay:
    ticker: str
    quantity: int


@dataclass(frozen=True)
class Position:
    position_date: date
    ticker: str
    isin: str | None
    quantity: int
    balance_brl: float


@dataclass(frozen=True)
class Bulletin:
    report_date: date
    position_date: date
    layout: str
    positions: tuple[Position, ...]
    source_row_count: int
    used_total_row_count: int


@dataclass
class ParseAudit:
    report_date: str
    filename: str
    pdf_sha256: str
    page_count: int = 0
    status: str = ""
    layout: str = ""
    position_date: str = ""
    source_row_count: int = 0
    parsed_position_count: int = 0
    used_total_row_count: int = 0
    mapped_nonzero_position_count: int = 0
    unmapped_position_count: int = 0
    outside_identity_bound_count: int = 0
    output_security_count: int = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _parse_day(value: str) -> date:
    return datetime.strptime(value, "%d/%m/%Y").date()


def _quantity(value: str) -> int:
    return int(value.replace(".", "").replace(",", ""))


def _money(value: str) -> float:
    if "," in value and "." in value:
        decimal = "," if value.rfind(",") > value.rfind(".") else "."
        thousands = "." if decimal == "," else ","
        return float(value.replace(thousands, "").replace(decimal, "."))
    if "," in value:
        return float(value.replace(".", "").replace(",", "."))
    if "." in value and len(value.rsplit(".", 1)[1]) == 2:
        return float(value.replace(",", ""))
    return float(value.replace(".", "").replace(",", ""))


def _modern_section(lines: list[str]) -> list[str] | None:
    start = next(
        (
            index
            for index, line in enumerate(lines)
            if "Posi" in line and "Aberto" in line
        ),
        None,
    )
    if start is None:
        return None
    end = next(
        (
            index
            for index, line in enumerate(lines[start + 1 :], start + 1)
            if "Registrados" in line
        ),
        len(lines),
    )
    return lines[start + 1 : end]


def _parse_modern(lines: list[str], report_date: date) -> Bulletin | None:
    section = _modern_section(lines)
    if section is None:
        return None
    raw: list[tuple[Position, bool]] = []
    for line in section:
        match = MODERN_ROW.fullmatch(line)
        if match is None:
            continue
        position = Position(
            position_date=_parse_day(match.group("day")),
            ticker=match.group("ticker"),
            isin=match.group("isin"),
            quantity=_quantity(match.group("quantity")),
            balance_brl=_money(match.group("balance")),
        )
        is_total = re.search(r"(?:^|\s)Total(?:\s|$)", match.group("body")) is not None
        raw.append((position, is_total))
    if not raw:
        return None
    dates = {position.position_date for position, _ in raw}
    if len(dates) != 1:
        raise ValueError(f"Modern BDI contains multiple position dates: {dates}")
    position_date = dates.pop()
    if position_date > report_date:
        raise ValueError("BDI position date cannot follow its report date")

    grouped: dict[str, list[tuple[Position, bool]]] = {}
    for item in raw:
        grouped.setdefault(item[0].isin or "", []).append(item)
    positions: list[Position] = []
    total_rows = 0
    for isin, items in grouped.items():
        totals = [position for position, is_total in items if is_total]
        if len(totals) > 1:
            raise ValueError(f"Duplicate BDI Total row for {isin}")
        if totals:
            positions.append(totals[0])
            total_rows += 1
            continue
        first = items[0][0]
        positions.append(
            Position(
                position_date=position_date,
                ticker=first.ticker,
                isin=isin,
                quantity=sum(position.quantity for position, _ in items),
                balance_brl=sum(position.balance_brl for position, _ in items),
            )
        )
    return Bulletin(
        report_date=report_date,
        position_date=position_date,
        layout="modern_isin",
        positions=tuple(positions),
        source_row_count=len(raw),
        used_total_row_count=total_rows,
    )


def _parse_legacy(lines: list[str], report_date: date) -> Bulletin | None:
    start = next(
        (index for index, line in enumerate(lines) if "Banco de T" in line), None
    )
    if start is None:
        return None
    prefix = "\n".join(lines[start : start + 15])
    date_match = re.search(
        r"saldo\s+acumulado[\s\S]{0,400}?\bem\s+(\d{2}/\d{2}/\d{4})",
        prefix,
        flags=re.IGNORECASE,
    )
    if date_match is None:
        raise ValueError("Legacy BDI lending table has no stated balance date")
    position_date = _parse_day(date_match.group(1))
    if position_date > report_date:
        raise ValueError("BDI position date cannot follow its report date")
    end = len(lines)
    for index, line in enumerate(lines[start + 1 :], start + 1):
        lowered = line.lower()
        if (
            (
                "mercado" in lowered
                and "contratos" in lowered
                and "referencial" in lowered
            )
            or ("posi" in lowered and "garantias" in lowered)
            or ("op" in lowered and "flex" in lowered)
        ):
            end = index
            break
    by_ticker: dict[str, Position] = {}
    for line in lines[start + 1 : end]:
        match = LEGACY_ROW.fullmatch(line)
        if match is None:
            continue
        ticker = match.group("ticker")
        if ticker in by_ticker:
            raise ValueError(f"Duplicate legacy BDI ticker row: {ticker}")
        by_ticker[ticker] = Position(
            position_date=position_date,
            ticker=ticker,
            isin=None,
            quantity=_quantity(match.group("quantity")),
            balance_brl=_money(match.group("balance")),
        )
    if not by_ticker:
        return None
    return Bulletin(
        report_date=report_date,
        position_date=position_date,
        layout="legacy_ticker",
        positions=tuple(by_ticker.values()),
        source_row_count=len(by_ticker),
        used_total_row_count=0,
    )


def parse_bdi_pages(pages: list[str], report_date: date) -> Bulletin | None:
    """Parse only the official BDI securities-loan open-balance section."""
    lines = [line.rstrip() for page in pages for line in page.splitlines()]
    modern = _parse_modern(lines, report_date)
    if modern is not None:
        return modern
    return _parse_legacy(lines, report_date)


def parse_bdi_pdf(path: Path, report_date: date) -> tuple[Bulletin | None, int]:
    reader = PdfReader(path)
    pages = [page.extract_text(extraction_mode="layout") or "" for page in reader.pages]
    return parse_bdi_pages(pages, report_date), len(reader.pages)


def _calendar_dates(calendar_dir: Path) -> list[date]:
    paths = sorted(calendar_dir.glob("year=*/equities_daily_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parsed COTAHIST equities under {calendar_dir}")
    return (
        pl.scan_parquet(paths)
        .select(pl.col("trade_date").unique().sort())
        .collect()
        .get_column("trade_date")
        .to_list()
    )


def _download_pdf(day: date, destination: Path) -> dict[str, object]:
    url = PDF_URL.format(day=day.isoformat(), compact=day.strftime("%Y%m%d"))
    request = urllib.request.Request(url, headers={"User-Agent": "Brazil-RV research"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
                if not payload.startswith(b"%PDF"):
                    raise ValueError(f"B3 response is not a PDF: {url}")
                destination.write_bytes(payload)
                return {
                    "report_date": day.isoformat(),
                    "status": "downloaded",
                    "url": url,
                    "filename": destination.name,
                    "bytes": len(payload),
                    "sha256": _sha256(destination),
                    "http_last_modified": response.headers.get("Last-Modified"),
                    "http_etag": response.headers.get("ETag"),
                }
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return {
                    "report_date": day.isoformat(),
                    "status": "not_found",
                    "url": url,
                    "http_status": 404,
                }
            # The archive endpoint returns HTTP 500, rather than 404, for dates
            # whose chapter file was never migrated. Retry briefly to distinguish
            # a transient response, then retain the exact status in the manifest.
            if error.code == 500 and attempt == 2:
                return {
                    "report_date": day.isoformat(),
                    "status": "not_found_http_500",
                    "url": url,
                    "http_status": 500,
                }
            if error.code < 500 and error.code != 429:
                raise
        except urllib.error.URLError:
            if attempt == 2:
                raise
        time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"Failed to acquire BDI PDF after retries: {url}")


def acquire_sources(
    calendar_dir: Path,
    output_dir: Path,
    *,
    start: date,
    end: date,
    workers: int = 6,
) -> dict[str, object]:
    if start > end:
        raise ValueError("start cannot follow end")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    sessions = [day for day in _calendar_dates(calendar_dir) if start <= day <= end]
    if not sessions:
        raise ValueError("No B3 sessions fall in the requested interval")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        records: list[dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _download_pdf,
                    day,
                    temporary / f"BDI_05_{day:%Y%m%d}.pdf",
                ): day
                for day in sessions
            }
            for future in as_completed(futures):
                records.append(future.result())
        records.sort(key=lambda record: str(record["report_date"]))
        downloaded = sum(record["status"] == "downloaded" for record in records)
        manifest: dict[str, object] = {
            "contract_version": RAW_CONTRACT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "source_endpoint": PDF_URL,
            "requested_start": start.isoformat(),
            "requested_end": end.isoformat(),
            "requested_b3_sessions": len(sessions),
            "downloaded_pdf_count": downloaded,
            "not_found_count": len(records) - downloaded,
            "immutability_rule": (
                "Official B3 PDF response bytes are stored unchanged and checksummed; "
                "the snapshot builder refuses to overwrite an existing directory"
            ),
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
    return manifest


def load_identities(assignments_path: Path) -> dict[str, Identity]:
    assignments = pl.read_parquet(assignments_path).select(
        "security_id", "isin", "first_overlap_date", "last_overlap_date"
    )
    identities: dict[str, Identity] = {}
    for row in assignments.iter_rows(named=True):
        isin = row["isin"]
        if VALID_ISIN.fullmatch(isin) is None or row["security_id"] != f"ISIN:{isin}":
            raise ValueError("Accepted lending identity must be an exact valid ISIN")
        if isin in identities:
            raise ValueError(f"Duplicate accepted ISIN identity: {isin}")
        identities[isin] = Identity(
            security_id=row["security_id"],
            isin=isin,
            effective_from=date.fromisoformat(row["first_overlap_date"]),
            effective_to_inclusive=date.fromisoformat(row["last_overlap_date"]),
        )
    if not identities:
        raise ValueError("Accepted identity table is empty")
    return identities


def load_stock_days(
    calendar_dir: Path,
    identities: dict[str, Identity],
    end: date,
) -> tuple[list[date], dict[tuple[date, str], StockDay]]:
    paths = sorted(calendar_dir.glob("year=*/equities_daily_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parsed COTAHIST equities under {calendar_dir}")
    market_dates = _calendar_dates(calendar_dir)
    security_ids = [identity.security_id for identity in identities.values()]
    frame = (
        pl.scan_parquet(paths)
        .filter(
            pl.col("trade_date") <= end,
            pl.col("security_id").is_in(security_ids),
        )
        .select(
            "trade_date",
            "security_id",
            "security_id_is_fallback",
            "ticker",
            "quantity",
        )
        .collect()
    )
    if frame.filter(pl.col("security_id_is_fallback")).height:
        raise ValueError("Accepted COTAHIST lending rows cannot use fallback identity")
    if frame.select(
        pl.struct("trade_date", "security_id").is_duplicated().any()
    ).item():
        raise ValueError("COTAHIST stock denominator has duplicate date/security rows")
    stock_days = {
        (row["trade_date"], row["security_id"]): StockDay(
            ticker=row["ticker"], quantity=int(row["quantity"])
        )
        for row in frame.iter_rows(named=True)
    }
    return market_dates, stock_days


def _map_bulletin(
    bulletin: Bulletin,
    identities: dict[str, Identity],
    stock_days: dict[tuple[date, str], StockDay],
) -> tuple[dict[str, Position], dict[str, int]]:
    by_ticker: dict[str, str] = {}
    for identity in identities.values():
        stock = stock_days.get((bulletin.position_date, identity.security_id))
        if stock is None:
            continue
        if stock.ticker in by_ticker:
            raise ValueError(
                f"Ambiguous same-date COTAHIST ticker {stock.ticker} on "
                f"{bulletin.position_date}"
            )
        by_ticker[stock.ticker] = identity.security_id
    by_security = {identity.security_id: identity for identity in identities.values()}
    mapped: dict[str, Position] = {}
    unmapped = 0
    outside = 0
    for position in bulletin.positions:
        if position.isin is not None:
            identity = identities.get(position.isin)
        else:
            security_id = by_ticker.get(position.ticker)
            identity = by_security.get(security_id) if security_id is not None else None
        if identity is None:
            unmapped += 1
            continue
        if not (
            identity.effective_from
            <= bulletin.position_date
            <= identity.effective_to_inclusive
        ):
            outside += 1
            continue
        if identity.security_id in mapped:
            raise ValueError(
                f"Multiple BDI positions map to {identity.security_id} on "
                f"{bulletin.position_date}"
            )
        mapped[identity.security_id] = position
    return mapped, {"unmapped": unmapped, "outside": outside}


def read_bulletins(
    raw_dir: Path,
    identities: dict[str, Identity],
    stock_days: dict[tuple[date, str], StockDay],
    *,
    start: date,
    end: date,
) -> tuple[list[Bulletin], list[ParseAudit]]:
    raw_manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    bulletins: list[Bulletin] = []
    audits: list[ParseAudit] = []
    seen_position_dates: set[date] = set()
    for record in raw_manifest["files"]:
        report_date = date.fromisoformat(record["report_date"])
        if not start <= report_date <= end or record["status"] != "downloaded":
            continue
        path = raw_dir / record["filename"]
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"Raw BDI PDF hash mismatch: {path}")
        audit = ParseAudit(
            report_date=report_date.isoformat(),
            filename=path.name,
            pdf_sha256=record["sha256"],
        )
        bulletin, page_count = parse_bdi_pdf(path, report_date)
        audit.page_count = page_count
        if bulletin is None:
            audit.status = "no_open_balance_table"
            audits.append(audit)
            continue
        if len(bulletin.positions) < MIN_PARSED_POSITIONS:
            raise ValueError(
                f"Incomplete BDI open-balance parse for {report_date}: "
                f"{len(bulletin.positions)} positions"
            )
        if bulletin.position_date in seen_position_dates:
            raise ValueError(f"Duplicate BDI position date: {bulletin.position_date}")
        mapped, mapping_audit = _map_bulletin(bulletin, identities, stock_days)
        if len(mapped) < MIN_MAPPED_NONZERO_POSITIONS:
            raise ValueError(
                f"Too few accepted positions map on {report_date}: {len(mapped)}"
            )
        seen_position_dates.add(bulletin.position_date)
        audit.status = "parsed"
        audit.layout = bulletin.layout
        audit.position_date = bulletin.position_date.isoformat()
        audit.source_row_count = bulletin.source_row_count
        audit.parsed_position_count = len(bulletin.positions)
        audit.used_total_row_count = bulletin.used_total_row_count
        audit.mapped_nonzero_position_count = len(mapped)
        audit.unmapped_position_count = mapping_audit["unmapped"]
        audit.outside_identity_bound_count = mapping_audit["outside"]
        audit.output_security_count = sum(
            (bulletin.position_date, identity.security_id) in stock_days
            and identity.effective_from
            <= bulletin.position_date
            <= identity.effective_to_inclusive
            for identity in identities.values()
        )
        bulletins.append(
            Bulletin(
                report_date=bulletin.report_date,
                position_date=bulletin.position_date,
                layout=bulletin.layout,
                positions=tuple(mapped.values()),
                source_row_count=bulletin.source_row_count,
                used_total_row_count=bulletin.used_total_row_count,
            )
        )
        audits.append(audit)
    return bulletins, audits


def build_rows(
    bulletins: list[Bulletin],
    identities: dict[str, Identity],
    stock_days: dict[tuple[date, str], StockDay],
    market_dates: list[date],
) -> list[dict[str, object]]:
    session_index = {day: index for index, day in enumerate(market_dates)}
    raw_rows: list[dict[str, object]] = []
    for bulletin in sorted(bulletins, key=lambda item: item.report_date):
        available_index = bisect_right(market_dates, bulletin.report_date)
        if available_index == len(market_dates):
            continue
        if bulletin.position_date not in session_index:
            raise ValueError("BDI balance date is absent from the B3 session calendar")
        position_by_security = {
            position.isin
            if bulletin.layout == "modern_isin"
            else position.ticker: position
            for position in bulletin.positions
        }
        for identity in identities.values():
            stock = stock_days.get((bulletin.position_date, identity.security_id))
            if stock is None or not (
                identity.effective_from
                <= bulletin.position_date
                <= identity.effective_to_inclusive
            ):
                continue
            if bulletin.layout == "modern_isin":
                position = position_by_security.get(identity.isin)
                identity_method = "bdi_isin"
            else:
                position = position_by_security.get(stock.ticker)
                identity_method = "same_date_cotahist_ticker"
            raw_rows.append(
                {
                    "source_position_date": bulletin.position_date,
                    "source_report_date": bulletin.report_date,
                    "available_date": market_dates[available_index],
                    "security_id": identity.security_id,
                    "source_identity_method": identity_method,
                    "lending_balance_quantity": position.quantity if position else 0,
                    "lending_balance_brl": position.balance_brl if position else 0.0,
                }
            )
    if not raw_rows:
        raise ValueError("No BDI lending rows were constructed")

    quantity_by_security_date = {
        (security_id, day): stock.quantity
        for (day, security_id), stock in stock_days.items()
    }
    level_by_security_date: dict[tuple[str, date], float] = {}
    for row in raw_rows:
        security_id = str(row["security_id"])
        position_date = row["source_position_date"]
        index = session_index[position_date]
        identity = next(
            value for value in identities.values() if value.security_id == security_id
        )
        start_index = index - ADV_SESSIONS + 1
        adv_valid = (
            start_index >= 0 and market_dates[start_index] >= identity.effective_from
        )
        if adv_valid:
            quantities = [
                quantity_by_security_date.get((security_id, day), 0)
                for day in market_dates[start_index : index + 1]
            ]
            adv = statistics.fmean(quantities)
            adv_valid = adv > 0.0
        else:
            adv = 0.0
        balance = int(row["lending_balance_quantity"])
        level = math.log1p(balance / adv) if adv_valid else 0.0
        row["lending_balance_days_to_cover_log_tanh"] = (
            math.tanh(level / 3.0) if adv_valid else 0.0
        )
        row["lending_balance_days_to_cover_log_tanh_mask"] = adv_valid
        if adv_valid:
            level_by_security_date[(security_id, position_date)] = level

    for row in raw_rows:
        security_id = str(row["security_id"])
        position_date = row["source_position_date"]
        index = session_index[position_date]
        current = level_by_security_date.get((security_id, position_date))
        for lag in CHANGE_SESSIONS:
            feature = f"lending_balance_days_to_cover_change_{lag}_tanh"
            prior_date = market_dates[index - lag] if index >= lag else None
            prior = (
                level_by_security_date.get((security_id, prior_date))
                if prior_date is not None
                else None
            )
            valid = current is not None and prior is not None
            row[feature] = math.tanh((current - prior) / 2.0) if valid else 0.0
            row[f"{feature}_mask"] = valid
    return raw_rows


def _frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    frame = (
        pl.DataFrame(rows, infer_schema_length=None)
        .with_columns(
            pl.col("source_position_date").cast(pl.Date),
            pl.col("source_report_date").cast(pl.Date),
            pl.col("available_date").cast(pl.Date),
            pl.col("lending_balance_quantity").cast(pl.Int64),
            pl.col("lending_balance_brl").cast(pl.Float64),
            *[pl.col(feature).cast(pl.Float32) for feature in FEATURES],
            *[pl.col(f"{feature}_mask").cast(pl.Boolean) for feature in FEATURES],
        )
        .sort("available_date", "security_id")
    )
    if frame.select(
        pl.struct("available_date", "security_id").is_duplicated().any()
    ).item():
        raise ValueError(
            "BDI lending output has duplicate available-date/security keys"
        )
    return frame


def build_sidecar(
    raw_dir: Path,
    calendar_dir: Path,
    assignments_path: Path,
    output_dir: Path,
    *,
    start: date,
    end: date,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    identities = load_identities(assignments_path)
    market_dates, stock_days = load_stock_days(calendar_dir, identities, end)
    bulletins, audits = read_bulletins(
        raw_dir,
        identities,
        stock_days,
        start=start,
        end=end,
    )
    rows = build_rows(bulletins, identities, stock_days, market_dates)
    frame = _frame(rows)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        output_path = temporary / "bdi_lending_open_balance.parquet"
        audit_path = temporary / "pdf_parse_audit.parquet"
        frame.write_parquet(output_path, compression="zstd", statistics=True)
        pl.DataFrame([asdict(audit) for audit in audits]).write_parquet(
            audit_path, compression="zstd", statistics=True
        )
        raw_manifest = raw_dir / "manifest.json"
        parsed = [audit for audit in audits if audit.status == "parsed"]
        manifest: dict[str, object] = {
            "contract_version": CONTRACT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "raw_source_directory": str(raw_dir.resolve()),
            "raw_manifest": {
                "path": str(raw_manifest.resolve()),
                "sha256": _sha256(raw_manifest),
            },
            "calendar_directory": str(calendar_dir.resolve()),
            "assignments": {
                "path": str(assignments_path.resolve()),
                "sha256": _sha256(assignments_path),
            },
            "source_start": start.isoformat(),
            "source_end": end.isoformat(),
            "availability_rule": (
                "A chapter-05 bulletin dated D is used only on the first B3 session "
                "strictly after D. This follows B3's description of the source files "
                "as end-of-day regulatory data and never assumes an intraday release time"
            ),
            "identity_rule": (
                "Modern rows map by exact BDI ISIN. Legacy rows map the exact BDI "
                "ticker to the exact same-position-date COTAHIST regular-equity row, "
                "then enforce the accepted permanent ISIN identity and effective bounds; "
                "ticker prefixes and current-ticker backfills are never used"
            ),
            "zero_rule": (
                "After a complete official open-position table passes row-count and "
                "identity guards, an active accepted security absent from that table "
                "has observed zero open balance. A missing/incomplete bulletin emits no rows"
            ),
            "normalization_rule": (
                "log1p(open shares / arithmetic mean own share volume over the 20 "
                "sessions ending on the stated balance date), fixed tanh scale 3; "
                "5/20-session changes use exact prior B3-session levels and fixed tanh "
                "scale 2. Missing sessions never collapse into observation lags"
            ),
            "scope_exclusions": (
                "This source contains open-balance quantities and BRL balance only. "
                "It does not infer lending rates, fees, registered-loan flow, or utilization"
            ),
            "features": list(FEATURES),
            "adv_sessions": ADV_SESSIONS,
            "change_sessions": list(CHANGE_SESSIONS),
            "parsed_bulletin_count": len(parsed),
            "no_table_bulletin_count": sum(
                audit.status == "no_open_balance_table" for audit in audits
            ),
            "legacy_bulletin_count": sum(
                audit.layout == "legacy_ticker" for audit in parsed
            ),
            "modern_bulletin_count": sum(
                audit.layout == "modern_isin" for audit in parsed
            ),
            "output_rows": frame.height,
            "output_security_count": frame.get_column("security_id").n_unique(),
            "first_position_date": str(frame.get_column("source_position_date").min()),
            "last_position_date": str(frame.get_column("source_position_date").max()),
            "first_available_date": str(frame.get_column("available_date").min()),
            "last_available_date": str(frame.get_column("available_date").max()),
            "feature_valid_rows": {
                feature: int(frame.get_column(f"{feature}_mask").sum())
                for feature in FEATURES
            },
            "output_file": output_path.name,
            "output_sha256": _sha256(output_path),
            "parse_audit_file": audit_path.name,
            "parse_audit_sha256": _sha256(audit_path),
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
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Acquire and normalize B3 BDI securities-lending open balances"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    acquire = commands.add_parser("acquire")
    acquire.add_argument("--calendar-dir", type=Path, required=True)
    acquire.add_argument("--out", type=Path, required=True)
    acquire.add_argument("--start", type=date.fromisoformat, required=True)
    acquire.add_argument("--end", type=date.fromisoformat, required=True)
    acquire.add_argument("--workers", type=int, default=6)
    build = commands.add_parser("build")
    build.add_argument("--raw-dir", type=Path, required=True)
    build.add_argument("--calendar-dir", type=Path, required=True)
    build.add_argument("--assignments", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--start", type=date.fromisoformat, required=True)
    build.add_argument("--end", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    if args.command == "acquire":
        manifest = acquire_sources(
            args.calendar_dir,
            args.out,
            start=args.start,
            end=args.end,
            workers=args.workers,
        )
        print(
            f"Wrote {manifest['downloaded_pdf_count']:,} official BDI PDFs to "
            f"{args.out}",
            flush=True,
        )
    else:
        manifest = build_sidecar(
            args.raw_dir,
            args.calendar_dir,
            args.assignments,
            args.out,
            start=args.start,
            end=args.end,
        )
        print(
            f"Wrote {manifest['output_rows']:,} BDI lending rows for "
            f"{manifest['output_security_count']} securities to {args.out}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
