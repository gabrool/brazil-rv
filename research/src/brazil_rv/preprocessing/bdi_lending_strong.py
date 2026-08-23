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
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl
from pypdf import PdfReader

from .bdi_lending import load_identities, load_stock_days

CONTRACT_VERSION = "B3_BDI_LENDING_RATES_FLOWS_V1"
RAW_CONTRACT_VERSION = "B3_LEGACY_BDI_CHAPTER_05_SNAPSHOT_V1"
LEGACY_URL = (
    "https://bvmf.bmfbovespa.com.br/download/BOLETINSDIARIOS/BDI_05_{compact}.pdf"
)
NUMBER = r"[0-9][0-9.,]*"
PERCENT = rf"{NUMBER}%"
REGISTERED_ROW = re.compile(
    rf"^\s*(?P<day>\d{{2}}/\d{{2}}/20\d{{2}})\s+"
    rf"(?P<ticker>[A-Z0-9]{{4,12}})\s+"
    rf"(?P<isin>[A-Z]{{2}}[A-Z0-9]{{9}}[0-9])\s+.*?\s+"
    rf"(?P<contracts>{NUMBER})\s+(?P<quantity>{NUMBER})\s+"
    rf"(?P<value>{NUMBER})\s+"
    rf"(?P<donor_min>{PERCENT})\s+(?P<donor_avg>{PERCENT})\s+"
    rf"(?P<donor_max>{PERCENT})\s+(?P<taker_min>{PERCENT})\s+"
    rf"(?P<taker_avg>{PERCENT})\s+(?P<taker_max>{PERCENT})\s*$"
)
MIN_COMPLETE_TABLE_ROWS = 100
ADV_OBSERVATIONS = 20
CHANGE_OBSERVATIONS = 5
FEATURES = (
    "lending_taker_fee_level_log_tanh",
    "lending_taker_fee_change_5_tanh",
    "lending_registered_flow_adv20_log_tanh",
    "lending_registered_flow_change_5_tanh",
)


@dataclass(frozen=True)
class RegisteredLoan:
    report_date: date
    ticker: str
    isin: str
    contracts: int
    quantity: int
    value_brl: float
    donor_min: float
    donor_avg: float
    donor_max: float
    taker_min: float
    taker_avg: float
    taker_max: float


@dataclass
class AggregatedLoans:
    contracts: int = 0
    quantity: int = 0
    value_brl: float = 0.0
    weighted_taker_rate: float = 0.0

    def add(self, row: RegisteredLoan) -> None:
        self.contracts += row.contracts
        self.quantity += row.quantity
        self.value_brl += row.value_brl
        self.weighted_taker_rate += row.quantity * row.taker_avg

    @property
    def taker_rate(self) -> float | None:
        return self.weighted_taker_rate / self.quantity if self.quantity > 0 else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _integer(value: str) -> int:
    return int(value.replace(".", "").replace(",", ""))


def _decimal(value: str) -> float:
    value = value.removesuffix("%")
    if "," in value and "." in value:
        decimal = "," if value.rfind(",") > value.rfind(".") else "."
        thousands = "." if decimal == "," else ","
        return float(value.replace(thousands, "").replace(decimal, "."))
    if value.count(",") == 1:
        return float(value.replace(",", "."))
    if value.count(".") == 1 and len(value.rsplit(".", 1)[1]) <= 2:
        return float(value)
    return float(value.replace(".", "").replace(",", ""))


def parse_registered_lines(
    lines: Sequence[str], report_date: date
) -> list[RegisteredLoan]:
    rows = []
    for line in lines:
        match = REGISTERED_ROW.fullmatch(line)
        if match is None:
            continue
        stated = match.group("day")
        row_date = datetime.strptime(stated, "%d/%m/%Y").date()
        if row_date != report_date:
            raise ValueError("Registered-loan row date differs from the BDI report")
        rows.append(
            RegisteredLoan(
                report_date=row_date,
                ticker=match.group("ticker"),
                isin=match.group("isin"),
                contracts=_integer(match.group("contracts")),
                quantity=_integer(match.group("quantity")),
                value_brl=_decimal(match.group("value")),
                donor_min=_decimal(match.group("donor_min")),
                donor_avg=_decimal(match.group("donor_avg")),
                donor_max=_decimal(match.group("donor_max")),
                taker_min=_decimal(match.group("taker_min")),
                taker_avg=_decimal(match.group("taker_avg")),
                taker_max=_decimal(match.group("taker_max")),
            )
        )
    return rows


def parse_registered_pdf(
    path: Path, report_date: date
) -> tuple[list[RegisteredLoan], int]:
    reader = PdfReader(path)
    rows = []
    for page in reader.pages:
        text = page.extract_text(extraction_mode="layout") or ""
        if "Taxa Doador" in text and "Taxa Tomador" in text:
            rows.extend(parse_registered_lines(text.splitlines(), report_date))
    if rows and len(rows) < MIN_COMPLETE_TABLE_ROWS:
        raise ValueError(f"Suspiciously small registered-loan table: {path}")
    return rows, len(reader.pages)


def _download(day: date, destination: Path) -> dict[str, object]:
    url = LEGACY_URL.format(compact=day.strftime("%Y%m%d"))
    request = urllib.request.Request(url, headers={"User-Agent": "Brazil-RV research"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
                if not payload.startswith(b"%PDF"):
                    raise ValueError(f"Legacy BDI response is not a PDF: {url}")
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
            if error.code in (404, 500):
                return {
                    "report_date": day.isoformat(),
                    "status": f"not_found_http_{error.code}",
                    "url": url,
                }
            if attempt == 2:
                raise
        except urllib.error.URLError:
            if attempt == 2:
                raise
        time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"Failed to acquire legacy BDI: {url}")


def acquire_legacy_sources(
    output_dir: Path,
    *,
    start: date,
    end: date,
    workers: int = 6,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_download, day, temporary / f"BDI_05_{day:%Y%m%d}.pdf"): day
                for day in days
            }
            records = [future.result() for future in as_completed(futures)]
        records.sort(key=lambda row: str(row["report_date"]))
        manifest = {
            "contract_version": RAW_CONTRACT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "requested_start": start.isoformat(),
            "requested_end": end.isoformat(),
            "requested_weekdays": len(days),
            "downloaded_pdf_count": sum(
                row["status"] == "downloaded" for row in records
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
    return output_dir


def _pdf_paths(pdf_dirs: Sequence[Path], start: date, end: date) -> list[Path]:
    by_date: dict[date, Path] = {}
    for directory in pdf_dirs:
        for path in directory.glob("BDI_05_*.pdf"):
            day = datetime.strptime(path.stem[-8:], "%Y%m%d").date()
            if start <= day <= end:
                existing = by_date.get(day)
                if existing is not None and _sha256(existing) != _sha256(path):
                    raise ValueError(f"Conflicting BDI files for {day}")
                by_date[day] = path
    return [by_date[day] for day in sorted(by_date)]


def build_lending_strong_features(
    *,
    pdf_dirs: Sequence[Path],
    calendar_dir: Path,
    assignments_path: Path,
    output_dir: Path,
    start: date,
    end: date,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    identities = load_identities(assignments_path)
    market_dates, stock_days = load_stock_days(calendar_dir, identities, end)
    paths = _pdf_paths(pdf_dirs, start, end)
    if not paths:
        raise ValueError("No BDI PDFs fall in the requested interval")

    tables: dict[date, dict[str, AggregatedLoans]] = {}
    audits = []
    for path in paths:
        report_date = datetime.strptime(path.stem[-8:], "%Y%m%d").date()
        loans, page_count = parse_registered_pdf(path, report_date)
        grouped: dict[str, AggregatedLoans] = {}
        for loan in loans:
            grouped.setdefault(loan.isin, AggregatedLoans()).add(loan)
        if loans:
            tables[report_date] = grouped
        audits.append(
            {
                "report_date": report_date.isoformat(),
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "page_count": page_count,
                "registered_row_count": len(loans),
                "registered_isin_count": len(grouped),
            }
        )

    fee_history: dict[tuple[date, str], float | None] = {}
    flow_history: dict[tuple[date, str], float | None] = {}
    output_rows = []
    for source_date in sorted(tables):
        next_index = bisect_right(market_dates, source_date)
        if next_index >= len(market_dates):
            continue
        available_date = market_dates[next_index]
        grouped = tables[source_date]
        source_index = bisect_right(market_dates, source_date) - 1
        five_sessions_back = (
            market_dates[source_index - CHANGE_OBSERVATIONS]
            if source_index >= CHANGE_OBSERVATIONS
            else None
        )
        for isin, identity in identities.items():
            if (
                not identity.effective_from
                <= source_date
                <= identity.effective_to_inclusive
            ):
                continue
            stock = stock_days.get((source_date, identity.security_id))
            if stock is None:
                continue
            quantity_history = [
                stock_days[(prior_date, identity.security_id)].quantity
                for prior_date in market_dates[: source_index + 1]
                if (prior_date, identity.security_id) in stock_days
            ][-ADV_OBSERVATIONS:]
            adv_valid = len(quantity_history) == ADV_OBSERVATIONS
            adv = (
                statistics.fmean(quantity_history[-ADV_OBSERVATIONS:])
                if adv_valid
                else 0.0
            )
            aggregate = grouped.get(isin)
            registered_quantity = 0 if aggregate is None else aggregate.quantity
            fee = None if aggregate is None else aggregate.taker_rate
            flow = registered_quantity / adv if adv_valid and adv > 0 else None

            previous_fee = (
                fee_history.get((five_sessions_back, identity.security_id))
                if five_sessions_back is not None
                else None
            )
            previous_flow = (
                flow_history.get((five_sessions_back, identity.security_id))
                if five_sessions_back is not None
                else None
            )
            fee_change_valid = fee is not None and previous_fee is not None
            flow_change_valid = flow is not None and previous_flow is not None
            row: dict[str, object] = {
                "source_trade_date": source_date,
                "available_date": available_date,
                "security_id": identity.security_id,
                "registered_contracts": 0 if aggregate is None else aggregate.contracts,
                "registered_quantity": registered_quantity,
            }
            row["lending_taker_fee_level_log_tanh"] = (
                math.tanh(math.log1p(fee) / 2.0) if fee is not None else 0.0
            )
            row["lending_taker_fee_level_log_tanh_mask"] = fee is not None
            row["lending_taker_fee_change_5_tanh"] = (
                math.tanh((math.log1p(fee) - math.log1p(previous_fee)) / 2.0)
                if fee_change_valid
                else 0.0
            )
            row["lending_taker_fee_change_5_tanh_mask"] = fee_change_valid
            row["lending_registered_flow_adv20_log_tanh"] = (
                math.tanh(math.log1p(flow) / 3.0) if flow is not None else 0.0
            )
            row["lending_registered_flow_adv20_log_tanh_mask"] = flow is not None
            row["lending_registered_flow_change_5_tanh"] = (
                math.tanh((math.log1p(flow) - math.log1p(previous_flow)) / 2.0)
                if flow_change_valid
                else 0.0
            )
            row["lending_registered_flow_change_5_tanh_mask"] = flow_change_valid
            output_rows.append(row)
            fee_history[(source_date, identity.security_id)] = fee
            flow_history[(source_date, identity.security_id)] = flow

    frame = pl.DataFrame(output_rows)
    if frame.is_empty():
        raise ValueError("No registered-loan rows mapped to accepted securities")
    if frame.select(
        pl.struct("available_date", "security_id").is_duplicated().any()
    ).item():
        raise ValueError("Strong lending features contain duplicate keys")
    for feature in FEATURES:
        if frame.filter(~pl.col(f"{feature}_mask") & (pl.col(feature) != 0)).height:
            raise ValueError("Invalid strong lending values must be exactly zero")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        parquet = temporary / "bdi_lending_strong.parquet"
        frame.write_parquet(parquet, compression="zstd")
        manifest = {
            "contract_version": CONTRACT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "features": list(FEATURES),
            "availability_rule": (
                "BDI report D is published after session D and is first used at "
                "the next observed B3 session"
            ),
            "normalization": (
                "Fixed log/tanh transforms; ADV uses the trailing 20 observed cash "
                "sessions ending on source D; five-observation changes use only "
                "the exact fifth prior B3 session and mask across bulletin gaps"
            ),
            "source_limit": (
                "Historical legacy two-step CSV bodies were zero bytes. Exact rates "
                "and registered flows therefore begin with the first BDI rate table; "
                "earlier open-balance PDFs are audited but do not fabricate rates."
            ),
            "pdf_directories": [str(path.resolve()) for path in pdf_dirs],
            "pdf_audits": audits,
            "rate_table_count": len(tables),
            "first_rate_table": min(tables).isoformat(),
            "last_rate_table": max(tables).isoformat(),
            "output_rows": frame.height,
            "output_security_count": frame.get_column("security_id").n_unique(),
            "output_sha256": _sha256(parquet),
            "official_validation_accessed": False,
            "test_accessed": False,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build strong-form B3 lending features"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    acquire = subparsers.add_parser("acquire")
    acquire.add_argument("--output-dir", type=Path, required=True)
    acquire.add_argument("--start", type=date.fromisoformat, required=True)
    acquire.add_argument("--end", type=date.fromisoformat, required=True)
    acquire.add_argument("--workers", type=int, default=6)
    build = subparsers.add_parser("build")
    build.add_argument("--pdf-dirs", type=Path, nargs="+", required=True)
    build.add_argument("--calendar-dir", type=Path, required=True)
    build.add_argument("--assignments-path", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--start", type=date.fromisoformat, required=True)
    build.add_argument("--end", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    if args.command == "acquire":
        print(
            acquire_legacy_sources(
                args.output_dir, start=args.start, end=args.end, workers=args.workers
            )
        )
    else:
        print(
            build_lending_strong_features(
                pdf_dirs=args.pdf_dirs,
                calendar_dir=args.calendar_dir,
                assignments_path=args.assignments_path,
                output_dir=args.output_dir,
                start=args.start,
                end=args.end,
            )
        )


if __name__ == "__main__":
    main()
