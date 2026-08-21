from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import zipfile
from bisect import bisect_right
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

CONTRACT_VERSION = "B3_COTAHIST_ODD_LOT_ACTIVITY_V1"
RECORD_LENGTH = 245
REGULAR_MARKET = ("02", 10)
ODD_LOT_MARKET = ("96", 20)
VALID_EQUITY_BASE_SPECS = {
    "ON",
    "OR",
    "PN",
    "PNA",
    "PNB",
    "PNC",
    "PND",
    "PNE",
    "PNF",
    "UNT",
}
VALID_ISIN = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")
ROLLING_CHANGE_OBSERVATIONS = 5
ROLLING_SCALE_OBSERVATIONS = 20
ROBUST_Z_CLIP = 5.0

LEVEL_FEATURES = (
    "odd_volume_share_asin_sqrt",
    "odd_trade_share_asin_sqrt",
)
FEATURES = (
    *LEVEL_FEATURES,
    "odd_volume_share_change_5",
    "odd_trade_share_change_5",
    "odd_volume_share_surprise_20",
    "odd_trade_share_surprise_20",
    "odd_regular_avg_trade_value_log_ratio",
    "odd_regular_close_log_ratio",
)


@dataclass
class ArchiveAudit:
    source_zip: str
    source_member: str = ""
    source_sha256: str = ""
    header_generation_date: str = ""
    physical_records: int = 0
    quote_records: int = 0
    header_records: int = 0
    trailer_records: int = 0
    malformed_length_records: int = 0
    regular_records: int = 0
    odd_lot_records: int = 0
    invalid_isin_records: int = 0
    non_equity_spec_records: int = 0
    block_variant_records_removed: int = 0


@dataclass
class QuoteRecord:
    trade_date: date
    isin: str
    market: str
    ticker: str
    trades: int
    quantity: int
    volume_cents: int
    close_brl: float


@dataclass
class Activity:
    trades: int = 0
    quantity: int = 0
    volume_cents: int = 0
    close_brl: float = 0.0
    ticker: str = ""
    source_row_count: int = 0
    _close_selector: tuple[int, str] = (-1, "")

    def add_record(self, record: QuoteRecord) -> None:
        self.trades += record.trades
        self.quantity += record.quantity
        self.volume_cents += record.volume_cents
        self.source_row_count += 1
        selector = (record.volume_cents, record.ticker)
        if selector > self._close_selector:
            self._close_selector = selector
            self.close_brl = record.close_brl
            self.ticker = record.ticker

    def add_activity(self, other: Activity) -> None:
        self.trades += other.trades
        self.quantity += other.quantity
        self.volume_cents += other.volume_cents
        self.source_row_count += other.source_row_count
        if other._close_selector > self._close_selector:
            self._close_selector = other._close_selector
            self.close_brl = other.close_brl
            self.ticker = other.ticker


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_int(raw: bytes) -> int:
    text = raw.decode("ascii", errors="ignore").strip()
    return int(text) if text else 0


def _parse_text(raw: bytes) -> str:
    return raw.decode("latin-1", errors="replace").strip()


def _parse_date(raw: bytes) -> date | None:
    text = raw.decode("ascii", errors="ignore").strip()
    if not text or text == "00000000":
        return None
    return datetime.strptime(text, "%Y%m%d").date()


def _choose_txt_member(archive: zipfile.ZipFile, source: Path) -> str:
    members = [name for name in archive.namelist() if name.upper().endswith(".TXT")]
    if not members:
        raise ValueError(f"COTAHIST ZIP contains no TXT member: {source}")
    preferred = [name for name in members if "COTAHIST" in name.upper()]
    return sorted(preferred or members)[0]


def parse_quote_line(line: bytes) -> QuoteRecord | None:
    """Parse only exact regular-board or odd-lot equity activity records."""
    if len(line) != RECORD_LENGTH or line[:2] != b"01":
        return None
    trade_date = _parse_date(line[2:10])
    if trade_date is None:
        return None
    cod_bdi = _parse_text(line[10:12])
    market_type = _parse_int(line[24:27])
    pair = (cod_bdi, market_type)
    if pair == REGULAR_MARKET:
        market = "regular"
    elif pair == ODD_LOT_MARKET:
        market = "odd_lot"
    else:
        return None
    spec = _parse_text(line[39:49])
    base_spec = spec.split()[0] if spec else ""
    if base_spec not in VALID_EQUITY_BASE_SPECS or "REC" in spec:
        return None
    isin = _parse_text(line[230:242])
    if VALID_ISIN.fullmatch(isin) is None:
        return None
    quote_factor = _parse_int(line[210:217]) or 1
    return QuoteRecord(
        trade_date=trade_date,
        isin=isin,
        market=market,
        ticker=_parse_text(line[12:24]),
        trades=_parse_int(line[147:152]),
        quantity=_parse_int(line[152:170]),
        volume_cents=_parse_int(line[170:188]),
        close_brl=_parse_int(line[108:121]) / (100.0 * quote_factor),
    )


def _scan_archive(
    source: Path,
    ticker_groups: dict[tuple[date, str, str], dict[str, Activity]],
) -> ArchiveAudit:
    audit = ArchiveAudit(
        source_zip=str(source.resolve()), source_sha256=_sha256(source)
    )
    with zipfile.ZipFile(source) as archive:
        member = _choose_txt_member(archive, source)
        audit.source_member = member
        with archive.open(member) as handle:
            for raw_line in handle:
                audit.physical_records += 1
                line = raw_line.rstrip(b"\r\n")
                if len(line) != RECORD_LENGTH:
                    audit.malformed_length_records += 1
                    continue
                record_type = line[:2]
                if record_type == b"00":
                    audit.header_records += 1
                    generated = _parse_date(line[23:31])
                    audit.header_generation_date = (
                        generated.isoformat() if generated else ""
                    )
                    continue
                if record_type == b"99":
                    audit.trailer_records += 1
                    continue
                if record_type != b"01":
                    continue
                audit.quote_records += 1
                cod_bdi = _parse_text(line[10:12])
                market_type = _parse_int(line[24:27])
                if (cod_bdi, market_type) not in {REGULAR_MARKET, ODD_LOT_MARKET}:
                    continue
                spec = _parse_text(line[39:49])
                base_spec = spec.split()[0] if spec else ""
                if base_spec not in VALID_EQUITY_BASE_SPECS or "REC" in spec:
                    audit.non_equity_spec_records += 1
                    continue
                isin = _parse_text(line[230:242])
                if VALID_ISIN.fullmatch(isin) is None:
                    audit.invalid_isin_records += 1
                    continue
                record = parse_quote_line(line)
                if record is None:
                    raise ValueError(
                        f"Failed to parse accepted COTAHIST record in {source}"
                    )
                if record.market == "regular":
                    audit.regular_records += 1
                else:
                    audit.odd_lot_records += 1
                key = (record.trade_date, record.isin, record.market)
                by_ticker = ticker_groups.setdefault(key, {})
                activity = by_ticker.setdefault(record.ticker, Activity())
                activity.add_record(record)
    if (
        audit.header_records != 1
        or audit.trailer_records != 1
        or audit.malformed_length_records
    ):
        raise ValueError(
            f"Malformed COTAHIST archive {source}: headers={audit.header_records}, "
            f"trailers={audit.trailer_records}, "
            f"bad_lengths={audit.malformed_length_records}"
        )
    return audit


def _collapse_tickers(
    ticker_groups: dict[tuple[date, str, str], dict[str, Activity]],
    audits: list[ArchiveAudit],
) -> dict[tuple[date, str, str], Activity]:
    collapsed: dict[tuple[date, str, str], Activity] = {}
    audit_by_year: dict[int, ArchiveAudit] = {}
    for audit in audits:
        match = re.search(r"A(\d{4})", Path(audit.source_zip).stem, re.IGNORECASE)
        if match:
            audit_by_year[int(match.group(1))] = audit
    for key, by_ticker in ticker_groups.items():
        combined = Activity()
        tickers = set(by_ticker)
        for ticker, activity in by_ticker.items():
            if (
                len(ticker) > 1
                and ticker[-1] in {"M", "Q", "R"}
                and ticker[:-1] in tickers
            ):
                audit = audit_by_year.get(key[0].year)
                if audit is not None:
                    audit.block_variant_records_removed += activity.source_row_count
                continue
            combined.add_activity(activity)
        if combined.source_row_count:
            collapsed[key] = combined
    return collapsed


def read_activity_archives(
    archives: list[Path],
) -> tuple[dict[tuple[date, str, str], Activity], list[ArchiveAudit]]:
    ticker_groups: dict[tuple[date, str, str], dict[str, Activity]] = {}
    audits = [_scan_archive(path, ticker_groups) for path in archives]
    return _collapse_tickers(ticker_groups, audits), audits


def load_market_dates(calendar_dir: Path) -> list[date]:
    paths = sorted(calendar_dir.glob("year=*/equities_daily_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parsed COTAHIST daily files under {calendar_dir}")
    values = (
        pl.scan_parquet(paths)
        .select(pl.col("trade_date").unique())
        .collect()
        .get_column("trade_date")
        .to_list()
    )
    return sorted(values)


def _angle_share(numerator: int, denominator: int) -> tuple[float, bool]:
    if denominator <= 0:
        return 0.0, False
    share = min(max(numerator / denominator, 0.0), 1.0)
    return math.asin(math.sqrt(share)), True


def _robust_prior_z(current: float, prior: list[float]) -> tuple[float, bool]:
    if len(prior) < ROLLING_SCALE_OBSERVATIONS:
        return 0.0, False
    window = prior[-ROLLING_SCALE_OBSERVATIONS:]
    center = statistics.median(window)
    mad = statistics.median(abs(value - center) for value in window)
    scale = 1.4826 * mad
    if scale <= 1e-12:
        return 0.0, False
    return min(max((current - center) / scale, -ROBUST_Z_CLIP), ROBUST_Z_CLIP), True


def _base_row(
    source_date: date,
    available_date: date,
    isin: str,
    regular: Activity,
    odd: Activity | None,
) -> dict[str, object]:
    odd_observed = odd is not None
    odd = odd or Activity()
    volume_level, volume_mask = _angle_share(
        odd.volume_cents, regular.volume_cents + odd.volume_cents
    )
    trade_level, trade_mask = _angle_share(odd.trades, regular.trades + odd.trades)
    avg_trade_mask = (
        regular.trades > 0
        and odd.trades > 0
        and regular.volume_cents > 0
        and odd.volume_cents > 0
    )
    avg_trade_ratio = (
        math.log(
            (odd.volume_cents / odd.trades) / (regular.volume_cents / regular.trades)
        )
        if avg_trade_mask
        else 0.0
    )
    close_mask = odd_observed and regular.close_brl > 0.0 and odd.close_brl > 0.0
    close_ratio = math.log(odd.close_brl / regular.close_brl) if close_mask else 0.0
    row: dict[str, object] = {
        "source_trade_date": source_date,
        "available_date": available_date,
        "security_id": f"ISIN:{isin}",
        "isin": isin,
        "regular_observed": True,
        "odd_lot_observed": odd_observed,
        "regular_ticker": regular.ticker,
        "odd_lot_ticker": odd.ticker,
        "regular_trades": regular.trades,
        "odd_lot_trades": odd.trades,
        "regular_quantity": regular.quantity,
        "odd_lot_quantity": odd.quantity,
        "regular_volume_brl": regular.volume_cents / 100.0,
        "odd_lot_volume_brl": odd.volume_cents / 100.0,
        "regular_close_brl": regular.close_brl,
        "odd_lot_close_brl": odd.close_brl,
        "regular_source_row_count": regular.source_row_count,
        "odd_lot_source_row_count": odd.source_row_count,
        LEVEL_FEATURES[0]: volume_level,
        f"{LEVEL_FEATURES[0]}_mask": volume_mask,
        LEVEL_FEATURES[1]: trade_level,
        f"{LEVEL_FEATURES[1]}_mask": trade_mask,
        "odd_regular_avg_trade_value_log_ratio": avg_trade_ratio,
        "odd_regular_avg_trade_value_log_ratio_mask": avg_trade_mask,
        "odd_regular_close_log_ratio": close_ratio,
        "odd_regular_close_log_ratio_mask": close_mask,
    }
    return row


def build_rows(
    activity: dict[tuple[date, str, str], Activity],
    market_dates: list[date],
    *,
    available_start: date | None = None,
    available_end: date | None = None,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    sessions = sorted(set(market_dates))
    session_set = set(sessions)
    regular_keys = sorted(
        (source_date, isin)
        for source_date, isin, market in activity
        if market == "regular"
    )
    if any(source_date not in session_set for source_date, _ in regular_keys):
        raise ValueError(
            "A regular COTAHIST source date is absent from the B3 calendar"
        )

    rows_by_security: dict[str, list[dict[str, object]]] = {}
    missing_next_session = 0
    for source_date, isin in regular_keys:
        next_index = bisect_right(sessions, source_date)
        if next_index == len(sessions):
            missing_next_session += 1
            continue
        row = _base_row(
            source_date,
            sessions[next_index],
            isin,
            activity[(source_date, isin, "regular")],
            activity.get((source_date, isin, "odd_lot")),
        )
        rows_by_security.setdefault(isin, []).append(row)

    for security_rows in rows_by_security.values():
        histories: dict[str, list[float]] = {feature: [] for feature in LEVEL_FEATURES}
        for row in security_rows:
            for index, level_feature in enumerate(LEVEL_FEATURES):
                level_mask = bool(row[f"{level_feature}_mask"])
                history = histories[level_feature]
                change_name = (
                    "odd_volume_share_change_5"
                    if index == 0
                    else "odd_trade_share_change_5"
                )
                surprise_name = (
                    "odd_volume_share_surprise_20"
                    if index == 0
                    else "odd_trade_share_surprise_20"
                )
                if level_mask and len(history) >= ROLLING_CHANGE_OBSERVATIONS:
                    row[change_name] = float(row[level_feature]) - history[-5]
                    row[f"{change_name}_mask"] = True
                else:
                    row[change_name] = 0.0
                    row[f"{change_name}_mask"] = False
                if level_mask:
                    surprise, surprise_mask = _robust_prior_z(
                        float(row[level_feature]), history
                    )
                else:
                    surprise, surprise_mask = 0.0, False
                row[surprise_name] = surprise
                row[f"{surprise_name}_mask"] = surprise_mask
                # Append only after every current feature has used the prior history.
                if level_mask:
                    history.append(float(row[level_feature]))

    all_rows = sorted(
        (row for security_rows in rows_by_security.values() for row in security_rows),
        key=lambda row: (row["available_date"], row["security_id"]),
    )
    filtered = [
        row
        for row in all_rows
        if (available_start is None or row["available_date"] >= available_start)
        and (available_end is None or row["available_date"] <= available_end)
    ]
    odd_keys = {
        (source_date, isin)
        for source_date, isin, market in activity
        if market == "odd_lot"
    }
    regular_key_set = set(regular_keys)
    audit = {
        "regular_security_days": len(regular_keys),
        "odd_lot_security_days": len(odd_keys),
        "orphan_odd_lot_security_days": len(odd_keys - regular_key_set),
        "missing_next_session_security_days": missing_next_session,
        "output_rows": len(filtered),
        "observed_zero_odd_lot_rows": sum(
            not bool(row["odd_lot_observed"]) for row in filtered
        ),
    }
    return filtered, audit


def _rows_to_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    if not rows:
        raise ValueError("Odd-lot sidecar contains no rows")
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col("source_trade_date").cast(pl.Date),
        pl.col("available_date").cast(pl.Date),
        *[pl.col(feature).cast(pl.Float32) for feature in FEATURES],
        *[pl.col(f"{feature}_mask").cast(pl.Boolean) for feature in FEATURES],
    )


def build_sidecar(
    archives: list[Path],
    calendar_dir: Path,
    output_dir: Path,
    *,
    available_start: date | None = None,
    available_end: date | None = None,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    activity, archive_audits = read_activity_archives(archives)
    market_dates = load_market_dates(calendar_dir)
    rows, row_audit = build_rows(
        activity,
        market_dates,
        available_start=available_start,
        available_end=available_end,
    )
    frame = _rows_to_frame(rows)
    output_dir.mkdir(parents=True, exist_ok=False)
    data_path = output_dir / "odd_lot_activity.parquet"
    frame.write_parquet(data_path, compression="zstd", statistics=True)
    feature_coverage = {
        feature: int(frame.get_column(f"{feature}_mask").sum()) for feature in FEATURES
    }
    manifest: dict[str, object] = {
        "contract_version": CONTRACT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "source_archives": [asdict(audit) for audit in archive_audits],
        "calendar_source": str(calendar_dir.resolve()),
        "calendar_first_session": market_dates[0].isoformat(),
        "calendar_last_session": market_dates[-1].isoformat(),
        "calendar_session_count": len(market_dates),
        "availability_rule": (
            "COTAHIST trade-date D aggregate is first available on the next "
            "observed B3 session after D"
        ),
        "identity_rule": "exact valid ISIN, stored as security_id=ISIN:<ISIN>",
        "missingness_rule": (
            "regular present plus odd-lot absent is observed zero; regular absent "
            "produces no sidecar row"
        ),
        "rolling_rule": (
            "per-security prior valid observations only; current observation is appended "
            "after lag-5 change and prior-20 median/MAD surprise are computed"
        ),
        "features": list(FEATURES),
        "feature_valid_rows": feature_coverage,
        "available_start_filter": available_start.isoformat()
        if available_start
        else None,
        "available_end_filter": available_end.isoformat() if available_end else None,
        **row_audit,
        "distinct_securities": frame.get_column("security_id").n_unique(),
        "first_source_trade_date": frame.get_column("source_trade_date")
        .min()
        .isoformat(),
        "last_source_trade_date": frame.get_column("source_trade_date")
        .max()
        .isoformat(),
        "first_available_date": frame.get_column("available_date").min().isoformat(),
        "last_available_date": frame.get_column("available_date").max().isoformat(),
        "output_file": data_path.name,
        "output_sha256": _sha256(data_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return manifest


def _parse_iso_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a leakage-safe B3 COTAHIST odd-lot activity sidecar"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--calendar-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--years", nargs="*", type=int)
    parser.add_argument("--available-start", type=_parse_iso_date)
    parser.add_argument("--available-end", type=_parse_iso_date)
    args = parser.parse_args()
    archives = sorted(args.input_dir.glob("COTAHIST_A*.ZIP"))
    if args.years:
        year_tokens = {f"A{year}" for year in args.years}
        archives = [
            path
            for path in archives
            if any(token in path.stem for token in year_tokens)
        ]
    if not archives:
        raise FileNotFoundError(
            f"No selected COTAHIST_A*.ZIP files in {args.input_dir}"
        )
    manifest = build_sidecar(
        archives,
        args.calendar_dir,
        args.out,
        available_start=args.available_start,
        available_end=args.available_end,
    )
    print(
        f"Wrote {manifest['output_rows']:,} rows for "
        f"{manifest['distinct_securities']:,} securities to {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
