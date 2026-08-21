from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import zipfile
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

SCRIPT_VERSION = "2"
RECORD_LENGTH = 245
VALID_EQUITY_BASE_SPECS = {"ON", "OR", "PN", "PNA", "PNB", "PNC", "PND", "PNE", "PNF", "UNT"}
VALID_ISIN = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")


@dataclass
class ParseAudit:
    year: int
    source_zip: str
    source_member: str
    source_sha256: str
    header_generation_date: str = ""
    trailer_total_records: int = 0
    physical_records: int = 0
    quote_records: int = 0
    header_records: int = 0
    trailer_records: int = 0
    other_records: int = 0
    trailer_count_basis: str = ""
    record_count_valid: bool = False
    record_count_warning: str = ""
    candidate_equity_records: int = 0
    block_variant_records_removed: int = 0
    duplicate_security_date_rows_collapsed: int = 0
    final_equity_security_days: int = 0
    distinct_securities: int = 0
    distinct_tickers: int = 0
    malformed_length_records: int = 0
    output_daily: str = ""
    output_ticker_observations: str = ""
    error: str = ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_int(raw: bytes) -> int:
    text = raw.decode("ascii", errors="ignore").strip()
    return int(text) if text else 0


def parse_text(raw: bytes) -> str:
    return raw.decode("latin-1", errors="replace").strip()


def parse_date_yyyymmdd(raw: bytes) -> date | None:
    text = raw.decode("ascii", errors="ignore").strip()
    if not text or text == "00000000":
        return None
    return datetime.strptime(text, "%Y%m%d").date()


def price_from_cents(raw: bytes) -> float:
    return parse_int(raw) / 100.0


def security_id_for(isin: str, ticker: str, spec: str) -> tuple[str, bool]:
    if VALID_ISIN.fullmatch(isin):
        return f"ISIN:{isin}", False
    return f"FALLBACK:{ticker}:{spec}", True


def choose_txt_member(archive: zipfile.ZipFile, year: int) -> str:
    members = [name for name in archive.namelist() if name.upper().endswith(".TXT")]
    if not members:
        raise ValueError("ZIP contains no TXT file")
    preferred = [name for name in members if "COTAHIST" in name.upper() and str(year) in Path(name).name]
    return preferred[0] if preferred else members[0]


def parse_header(line: bytes) -> str:
    if len(line) < 31 or line[:2] != b"00":
        return ""
    parsed = parse_date_yyyymmdd(line[23:31])
    return parsed.isoformat() if parsed else ""


def parse_trailer(line: bytes) -> int:
    if len(line) < 42 or line[:2] != b"99":
        return 0
    return parse_int(line[31:42])


def is_equity_candidate(cod_bdi: str, market_type: int, spec: str) -> bool:
    if cod_bdi != "02" or market_type != 10:
        return False
    base_spec = spec.split()[0] if spec else ""
    return base_spec in VALID_EQUITY_BASE_SPECS and "REC" not in spec


def parse_quote_line(line: bytes) -> dict[str, object] | None:
    if len(line) < RECORD_LENGTH or line[:2] != b"01":
        return None
    trade_date = parse_date_yyyymmdd(line[2:10])
    if trade_date is None:
        return None
    cod_bdi = parse_text(line[10:12])
    ticker = parse_text(line[12:24])
    market_type = parse_int(line[24:27])
    issuer_short_name = parse_text(line[27:39])
    spec = parse_text(line[39:49])
    if not is_equity_candidate(cod_bdi, market_type, spec):
        return None
    quote_factor = parse_int(line[210:217]) or 1
    isin = parse_text(line[230:242])
    security_id, fallback = security_id_for(isin, ticker, spec)
    return {
        "trade_date": trade_date,
        "security_id": security_id,
        "security_id_is_fallback": fallback,
        "isin": isin,
        "ticker": ticker,
        "issuer_short_name": issuer_short_name,
        "security_spec": spec,
        "security_spec_base": spec.split()[0] if spec else "",
        "bdi_code": cod_bdi,
        "market_type": market_type,
        "currency": parse_text(line[52:56]),
        "open_brl": price_from_cents(line[56:69]) / quote_factor,
        "high_brl": price_from_cents(line[69:82]) / quote_factor,
        "low_brl": price_from_cents(line[82:95]) / quote_factor,
        "average_brl": price_from_cents(line[95:108]) / quote_factor,
        "close_brl": price_from_cents(line[108:121]) / quote_factor,
        "best_bid_brl": price_from_cents(line[121:134]) / quote_factor,
        "best_ask_brl": price_from_cents(line[134:147]) / quote_factor,
        "trades": parse_int(line[147:152]),
        "quantity": parse_int(line[152:170]),
        "volume_brl": parse_int(line[170:188]) / 100.0,
        "quote_factor": quote_factor,
        "distribution_number": parse_int(line[242:245]),
    }


def detect_block_variants(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], int]:
    keys = {(row["trade_date"], row["isin"], row["ticker"]) for row in rows if row["isin"]}
    retained: list[dict[str, object]] = []
    removed = 0
    for row in rows:
        ticker = str(row["ticker"])
        isin = str(row["isin"])
        is_variant = (
            len(ticker) > 1
            and ticker[-1] in {"M", "Q", "R"}
            and bool(isin)
            and (row["trade_date"], isin, ticker[:-1]) in keys
        )
        if is_variant:
            removed += 1
        else:
            retained.append(row)
    return retained, removed


def rows_to_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col("trade_date").cast(pl.Date),
        pl.col("security_id_is_fallback").cast(pl.Boolean),
        pl.col("trades").cast(pl.Int64),
        pl.col("quantity").cast(pl.Int64),
        pl.col("quote_factor").cast(pl.Int64),
        pl.col("distribution_number").cast(pl.Int64),
    )


def collapse_security_days(frame: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    if frame.is_empty():
        return frame, 0
    duplicate_count = frame.height - frame.select(pl.struct(["trade_date", "security_id"]).n_unique()).item()
    if duplicate_count <= 0:
        return frame.with_columns(pl.lit(1).alias("source_row_count")), 0
    identity_and_price = [
        "ticker", "issuer_short_name", "security_spec", "security_spec_base", "bdi_code",
        "market_type", "currency", "open_brl", "high_brl", "low_brl", "average_brl",
        "close_brl", "best_bid_brl", "best_ask_brl", "quote_factor",
        "distribution_number", "isin", "security_id_is_fallback",
    ]
    expressions: list[pl.Expr] = [pl.col(column).sort_by("volume_brl").last().alias(column) for column in identity_and_price]
    expressions.extend([
        pl.col("trades").sum().alias("trades"),
        pl.col("quantity").sum().alias("quantity"),
        pl.col("volume_brl").sum().alias("volume_brl"),
        pl.len().alias("source_row_count"),
    ])
    collapsed = frame.group_by(["trade_date", "security_id"]).agg(expressions)
    return collapsed.sort(["trade_date", "security_id"]), duplicate_count


def parse_year(source_zip: Path, year: int, out_root: Path) -> ParseAudit:
    audit = ParseAudit(year=year, source_zip=str(source_zip), source_member="", source_sha256=sha256_file(source_zip))
    try:
        rows: list[dict[str, object]] = []
        with zipfile.ZipFile(source_zip) as archive:
            member = choose_txt_member(archive, year)
            audit.source_member = member
            with archive.open(member) as handle:
                for raw_line in handle:
                    audit.physical_records += 1
                    line = raw_line.rstrip(b"\r\n")
                    if len(line) != RECORD_LENGTH:
                        audit.malformed_length_records += 1
                    if line[:2] == b"00":
                        audit.header_records += 1
                        audit.header_generation_date = parse_header(line)
                        continue
                    if line[:2] == b"99":
                        audit.trailer_records += 1
                        audit.trailer_total_records = parse_trailer(line)
                        continue
                    if line[:2] != b"01":
                        audit.other_records += 1
                        continue
                    audit.quote_records += 1
                    parsed = parse_quote_line(line)
                    if parsed is not None:
                        rows.append(parsed)
        # B3's published layout says the trailer count includes header and trailer.
        # In observed 2025+ annual files, the trailer may instead equal the count
        # of type-01 quote records. Treat both conventions as internally valid,
        # while preserving the basis and warning in the audit output.
        if audit.trailer_total_records == audit.physical_records:
            audit.trailer_count_basis = "all_physical_records"
            audit.record_count_valid = True
        elif audit.trailer_total_records == audit.quote_records:
            audit.trailer_count_basis = "quote_records_only"
            audit.record_count_valid = True
            audit.record_count_warning = (
                "Trailer count excludes header/trailer, contrary to the published layout"
            )
        else:
            audit.trailer_count_basis = "mismatch"
            audit.record_count_valid = False
            audit.record_count_warning = (
                f"Trailer={audit.trailer_total_records}, physical={audit.physical_records}, "
                f"quotes={audit.quote_records}"
            )

        structural_ok = (
            audit.header_records == 1
            and audit.trailer_records == 1
            and audit.other_records == 0
            and audit.malformed_length_records == 0
        )
        if not structural_ok:
            audit.record_count_valid = False
            extra = (
                f"header_records={audit.header_records}, "
                f"trailer_records={audit.trailer_records}, "
                f"other_records={audit.other_records}, "
                f"malformed_length_records={audit.malformed_length_records}"
            )
            audit.record_count_warning = (
                f"{audit.record_count_warning}; {extra}"
                if audit.record_count_warning
                else extra
            )

        audit.candidate_equity_records = len(rows)
        rows, removed = detect_block_variants(rows)
        audit.block_variant_records_removed = removed
        observations = rows_to_frame(rows).sort(["trade_date", "security_id", "ticker"])
        daily, duplicate_count = collapse_security_days(observations)
        audit.duplicate_security_date_rows_collapsed = duplicate_count
        audit.final_equity_security_days = daily.height
        audit.distinct_securities = daily["security_id"].n_unique() if daily.height else 0
        audit.distinct_tickers = observations["ticker"].n_unique() if observations.height else 0
        year_dir = out_root / f"year={year}"
        year_dir.mkdir(parents=True, exist_ok=False)
        daily_path = year_dir / f"equities_daily_{year}.parquet"
        observations_path = year_dir / f"ticker_observations_{year}.parquet"
        daily.write_parquet(daily_path, compression="zstd", statistics=True)
        observations.write_parquet(observations_path, compression="zstd", statistics=True)
        audit.output_daily = str(daily_path)
        audit.output_ticker_observations = str(observations_path)
    except Exception as exc:  # noqa: BLE001
        audit.error = f"{type(exc).__name__}: {exc}"
    return audit


def parse_year_from_filename(path: Path) -> int:
    match = re.search(r"COTAHIST_A(\d{4})\.ZIP$", path.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot infer year from filename: {path.name}")
    return int(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse B3 COTAHIST annual ZIPs into central-market equity daily Parquet files.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--years", nargs="*", type=int)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Output directory already exists: {args.out}")
    zip_paths = sorted(args.input_dir.glob("COTAHIST_A*.ZIP"))
    if args.years:
        requested = set(args.years)
        zip_paths = [path for path in zip_paths if parse_year_from_filename(path) in requested]
    if not zip_paths:
        raise FileNotFoundError(f"No COTAHIST_A*.ZIP files found in {args.input_dir}")
    args.out.mkdir(parents=True, exist_ok=False)
    audits: list[ParseAudit] = []
    for path in zip_paths:
        year = parse_year_from_filename(path)
        print(f"Parsing {path.name} ...", flush=True)
        audit = parse_year(path, year, args.out)
        audits.append(audit)
        if audit.error:
            print(f"  ERROR: {audit.error}", file=sys.stderr)
        else:
            print(f"  {audit.final_equity_security_days:,} security-days; {audit.distinct_securities:,} securities; removed {audit.block_variant_records_removed:,} block variants")
    audit_csv = args.out / "parse_audit.csv"
    audit_json = args.out / "parse_audit.json"
    with audit_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(audits[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(item) for item in audits)
    audit_json.write_text(json.dumps({
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.out),
        "audits": [asdict(item) for item in audits],
    }, indent=2), encoding="utf-8")
    return 2 if any(item.error for item in audits) else 0


if __name__ == "__main__":
    raise SystemExit(main())
