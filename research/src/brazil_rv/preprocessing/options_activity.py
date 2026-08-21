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

CONTRACT_VERSION = "B3_COTAHIST_OPTIONS_ACTIVITY_V1"
RECORD_LENGTH = 245
CALL_MARKET_TYPE = 70
PUT_MARKET_TYPE = 80
VALID_ISIN = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")
CHANGE_OBSERVATIONS = 5
SURPRISE_OBSERVATIONS = 20
SURPRISE_CLIP = 5.0

FEATURES = (
    "options_stock_quantity_log_ratio_tanh",
    "options_stock_turnover_log_ratio_tanh",
    "options_put_call_quantity_log_ratio_tanh",
    "options_put_call_trade_log_ratio_tanh",
    "options_near_expiry_share_asin_sqrt",
    "options_quantity_weighted_abs_log_moneyness_tanh",
    "options_quantity_log_surprise_20_scaled",
    "options_stock_quantity_log_ratio_change_5_tanh",
)


@dataclass(frozen=True)
class Identity:
    security_id: str
    isin: str
    ticker: str
    effective_from: date
    effective_to_inclusive: date


@dataclass(frozen=True)
class StockDay:
    close_brl: float
    trades: int
    quantity: int
    volume_brl: float


@dataclass(frozen=True)
class OptionRecord:
    trade_date: date
    isin: str
    ticker: str
    is_put: bool
    expiry: date | None
    strike_brl: float
    trades: int
    quantity: int
    volume_brl: float


@dataclass
class OptionActivity:
    call_trades: int = 0
    put_trades: int = 0
    call_quantity: int = 0
    put_quantity: int = 0
    call_volume_brl: float = 0.0
    put_volume_brl: float = 0.0
    valid_expiry_quantity: int = 0
    near_expiry_quantity: int = 0
    valid_moneyness_quantity: int = 0
    weighted_abs_log_moneyness: float = 0.0
    source_row_count: int = 0

    def add(self, record: OptionRecord, stock_close: float) -> None:
        if record.is_put:
            self.put_trades += record.trades
            self.put_quantity += record.quantity
            self.put_volume_brl += record.volume_brl
        else:
            self.call_trades += record.trades
            self.call_quantity += record.quantity
            self.call_volume_brl += record.volume_brl
        self.source_row_count += 1
        if record.expiry is not None and record.expiry >= record.trade_date:
            self.valid_expiry_quantity += record.quantity
            if (record.expiry - record.trade_date).days <= 30:
                self.near_expiry_quantity += record.quantity
        if record.strike_brl > 0.0 and stock_close > 0.0:
            self.valid_moneyness_quantity += record.quantity
            self.weighted_abs_log_moneyness += record.quantity * abs(
                math.log(record.strike_brl / stock_close)
            )


@dataclass
class ArchiveAudit:
    source_zip: str
    source_member: str = ""
    source_sha256: str = ""
    header_generation_date: str = ""
    physical_records: int = 0
    header_records: int = 0
    trailer_records: int = 0
    malformed_length_records: int = 0
    option_records: int = 0
    mapped_option_records: int = 0
    used_option_records: int = 0
    out_of_interval_records: int = 0
    unmapped_isin_records: int = 0
    outside_identity_bound_records: int = 0
    missing_stock_day_records: int = 0
    call_records_used: int = 0
    put_records_used: int = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _text(raw: bytes) -> str:
    return raw.decode("latin-1", errors="replace").strip()


def _integer(raw: bytes) -> int:
    value = raw.decode("ascii", errors="ignore").strip()
    return int(value) if value else 0


def _date(raw: bytes) -> date | None:
    value = raw.decode("ascii", errors="ignore").strip()
    if not value or value == "00000000":
        return None
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError:
        return None


def parse_option_line(line: bytes) -> OptionRecord | None:
    """Parse call/put semantics from TPMERC and identity from COTAHIST ISIN."""
    if len(line) != RECORD_LENGTH or line[:2] != b"01":
        return None
    market_type = _integer(line[24:27])
    if market_type not in (CALL_MARKET_TYPE, PUT_MARKET_TYPE):
        return None
    trade_date = _date(line[2:10])
    isin = _text(line[230:242])
    if trade_date is None or VALID_ISIN.fullmatch(isin) is None:
        return None
    quote_factor = _integer(line[210:217]) or 1
    return OptionRecord(
        trade_date=trade_date,
        isin=isin,
        ticker=_text(line[12:24]),
        is_put=market_type == PUT_MARKET_TYPE,
        expiry=_date(line[202:210]),
        strike_brl=_integer(line[188:201]) / (100.0 * quote_factor),
        trades=_integer(line[147:152]),
        quantity=_integer(line[152:170]),
        volume_brl=_integer(line[170:188]) / 100.0,
    )


def load_identities(assignments_path: Path) -> dict[str, Identity]:
    assignments = pl.read_parquet(assignments_path).select(
        "security_id",
        "isin",
        "latest_ticker",
        "first_overlap_date",
        "last_overlap_date",
    )
    identities: dict[str, Identity] = {}
    for row in assignments.iter_rows(named=True):
        isin = row["isin"]
        if VALID_ISIN.fullmatch(isin) is None or row["security_id"] != f"ISIN:{isin}":
            raise ValueError("Accepted option identity must be an exact valid ISIN")
        if isin in identities:
            raise ValueError(f"Duplicate accepted ISIN identity: {isin}")
        identities[isin] = Identity(
            security_id=row["security_id"],
            isin=isin,
            ticker=row["latest_ticker"],
            effective_from=date.fromisoformat(row["first_overlap_date"]),
            effective_to_inclusive=date.fromisoformat(row["last_overlap_date"]),
        )
    if not identities:
        raise ValueError("Accepted identity table is empty")
    return identities


def load_stock_days(
    calendar_dir: Path,
    identities: dict[str, Identity],
    start: date,
    end: date,
) -> tuple[list[date], dict[tuple[date, str], StockDay]]:
    paths = sorted(calendar_dir.glob("year=*/equities_daily_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parsed COTAHIST equities under {calendar_dir}")
    market_dates = (
        pl.scan_parquet(paths)
        .select(pl.col("trade_date").unique().sort())
        .collect()
        .get_column("trade_date")
        .to_list()
    )
    security_ids = [identity.security_id for identity in identities.values()]
    frame = (
        pl.scan_parquet(paths)
        .filter(
            pl.col("trade_date").is_between(start, end),
            pl.col("security_id").is_in(security_ids),
        )
        .select(
            "trade_date",
            "security_id",
            "close_brl",
            "trades",
            "quantity",
            "volume_brl",
        )
        .collect()
    )
    if frame.select(
        pl.struct("trade_date", "security_id").is_duplicated().any()
    ).item():
        raise ValueError("Parsed equity denominator has duplicate date/security rows")
    stock_days = {
        (row["trade_date"], row["security_id"]): StockDay(
            close_brl=float(row["close_brl"]),
            trades=int(row["trades"]),
            quantity=int(row["quantity"]),
            volume_brl=float(row["volume_brl"]),
        )
        for row in frame.iter_rows(named=True)
    }
    return market_dates, stock_days


def _choose_txt_member(archive: zipfile.ZipFile, source: Path) -> str:
    members = [name for name in archive.namelist() if name.upper().endswith(".TXT")]
    if not members:
        raise ValueError(f"COTAHIST archive contains no TXT member: {source}")
    preferred = [name for name in members if "COTAHIST" in name.upper()]
    return sorted(preferred or members)[0]


def scan_archive(
    source: Path,
    identities: dict[str, Identity],
    stock_days: dict[tuple[date, str], StockDay],
    start: date,
    end: date,
    activities: dict[tuple[date, str], OptionActivity],
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
                if line[:2] == b"00":
                    audit.header_records += 1
                    generated = _date(line[23:31])
                    audit.header_generation_date = (
                        generated.isoformat() if generated else ""
                    )
                    continue
                if line[:2] == b"99":
                    audit.trailer_records += 1
                    continue
                market_type = _integer(line[24:27]) if line[:2] == b"01" else 0
                if market_type not in (CALL_MARKET_TYPE, PUT_MARKET_TYPE):
                    continue
                audit.option_records += 1
                record = parse_option_line(line)
                if record is None:
                    continue
                if not start <= record.trade_date <= end:
                    audit.out_of_interval_records += 1
                    continue
                identity = identities.get(record.isin)
                if identity is None:
                    audit.unmapped_isin_records += 1
                    continue
                audit.mapped_option_records += 1
                if not (
                    identity.effective_from
                    <= record.trade_date
                    <= identity.effective_to_inclusive
                ):
                    audit.outside_identity_bound_records += 1
                    continue
                stock = stock_days.get((record.trade_date, identity.security_id))
                if stock is None:
                    audit.missing_stock_day_records += 1
                    continue
                activity = activities.setdefault(
                    (record.trade_date, identity.security_id), OptionActivity()
                )
                activity.add(record, stock.close_brl)
                audit.used_option_records += 1
                if record.is_put:
                    audit.put_records_used += 1
                else:
                    audit.call_records_used += 1
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


def read_activities(
    archives: list[Path],
    identities: dict[str, Identity],
    stock_days: dict[tuple[date, str], StockDay],
    start: date,
    end: date,
) -> tuple[dict[tuple[date, str], OptionActivity], list[ArchiveAudit]]:
    activities: dict[tuple[date, str], OptionActivity] = {}
    audits = [
        scan_archive(source, identities, stock_days, start, end, activities)
        for source in archives
    ]
    return activities, audits


def _ratio_tanh(numerator: float, denominator: float, scale: float) -> float:
    return math.tanh((math.log1p(numerator) - math.log1p(denominator)) / scale)


def _prior_robust_z(value: float, history: list[float]) -> tuple[float, bool]:
    if len(history) < SURPRISE_OBSERVATIONS:
        return 0.0, False
    prior = history[-SURPRISE_OBSERVATIONS:]
    center = statistics.median(prior)
    mad = statistics.median(abs(item - center) for item in prior)
    scale = 1.4826 * mad
    if scale <= 1e-12:
        return 0.0, False
    z = min(max((value - center) / scale, -SURPRISE_CLIP), SURPRISE_CLIP)
    return z / SURPRISE_CLIP, True


def _base_row(
    source_date: date,
    available_date: date,
    security_id: str,
    stock: StockDay,
    activity: OptionActivity,
) -> dict[str, object]:
    call_quantity = activity.call_quantity
    put_quantity = activity.put_quantity
    total_quantity = call_quantity + put_quantity
    call_trades = activity.call_trades
    put_trades = activity.put_trades
    total_trades = call_trades + put_trades
    total_volume = activity.call_volume_brl + activity.put_volume_brl
    row: dict[str, object] = {
        "source_trade_date": source_date,
        "available_date": available_date,
        "security_id": security_id,
        "option_series_rows": activity.source_row_count,
        "call_trades": call_trades,
        "put_trades": put_trades,
        "call_quantity": call_quantity,
        "put_quantity": put_quantity,
        "call_volume_brl": activity.call_volume_brl,
        "put_volume_brl": activity.put_volume_brl,
        "stock_trades": stock.trades,
        "stock_quantity": stock.quantity,
        "stock_volume_brl": stock.volume_brl,
    }
    quantity_ratio_mask = stock.quantity > 0
    row["options_stock_quantity_log_ratio_tanh"] = (
        _ratio_tanh(total_quantity, stock.quantity, 4.0) if quantity_ratio_mask else 0.0
    )
    row["options_stock_quantity_log_ratio_tanh_mask"] = quantity_ratio_mask
    turnover_ratio_mask = stock.volume_brl > 0.0
    row["options_stock_turnover_log_ratio_tanh"] = (
        _ratio_tanh(total_volume, stock.volume_brl, 4.0) if turnover_ratio_mask else 0.0
    )
    row["options_stock_turnover_log_ratio_tanh_mask"] = turnover_ratio_mask
    row["options_put_call_quantity_log_ratio_tanh"] = (
        _ratio_tanh(put_quantity, call_quantity, 3.0) if total_quantity > 0 else 0.0
    )
    row["options_put_call_quantity_log_ratio_tanh_mask"] = total_quantity > 0
    row["options_put_call_trade_log_ratio_tanh"] = (
        _ratio_tanh(put_trades, call_trades, 3.0) if total_trades > 0 else 0.0
    )
    row["options_put_call_trade_log_ratio_tanh_mask"] = total_trades > 0
    expiry_mask = (
        total_quantity > 0 and activity.valid_expiry_quantity == total_quantity
    )
    expiry_share = (
        activity.near_expiry_quantity / total_quantity if expiry_mask else 0.0
    )
    row["options_near_expiry_share_asin_sqrt"] = (
        2.0 * math.asin(math.sqrt(expiry_share)) / math.pi if expiry_mask else 0.0
    )
    row["options_near_expiry_share_asin_sqrt_mask"] = expiry_mask
    moneyness_mask = (
        total_quantity > 0 and activity.valid_moneyness_quantity == total_quantity
    )
    mean_abs_moneyness = (
        activity.weighted_abs_log_moneyness / total_quantity if moneyness_mask else 0.0
    )
    row["options_quantity_weighted_abs_log_moneyness_tanh"] = (
        math.tanh(mean_abs_moneyness / 0.25) if moneyness_mask else 0.0
    )
    row["options_quantity_weighted_abs_log_moneyness_tanh_mask"] = moneyness_mask
    row["_total_quantity_log"] = math.log1p(total_quantity)
    row["_stock_quantity_log_ratio"] = (
        math.log1p(total_quantity) - math.log1p(stock.quantity)
        if quantity_ratio_mask
        else 0.0
    )
    return row


def build_rows(
    activities: dict[tuple[date, str], OptionActivity],
    stock_days: dict[tuple[date, str], StockDay],
    market_dates: list[date],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    rows_by_security: dict[str, list[dict[str, object]]] = {}
    missing_next_session = 0
    for source_date, security_id in sorted(activities):
        next_index = bisect_right(market_dates, source_date)
        if next_index == len(market_dates):
            missing_next_session += 1
            continue
        stock = stock_days[(source_date, security_id)]
        row = _base_row(
            source_date,
            market_dates[next_index],
            security_id,
            stock,
            activities[(source_date, security_id)],
        )
        rows_by_security.setdefault(security_id, []).append(row)

    for security_rows in rows_by_security.values():
        quantity_history: list[float] = []
        ratio_history: list[float] = []
        for row in security_rows:
            quantity_log = float(row.pop("_total_quantity_log"))
            surprise, surprise_mask = _prior_robust_z(quantity_log, quantity_history)
            row["options_quantity_log_surprise_20_scaled"] = surprise
            row["options_quantity_log_surprise_20_scaled_mask"] = surprise_mask
            ratio = float(row.pop("_stock_quantity_log_ratio"))
            ratio_valid = bool(row["options_stock_quantity_log_ratio_tanh_mask"])
            if ratio_valid and len(ratio_history) >= CHANGE_OBSERVATIONS:
                row["options_stock_quantity_log_ratio_change_5_tanh"] = math.tanh(
                    (ratio - ratio_history[-CHANGE_OBSERVATIONS]) / 3.0
                )
                row["options_stock_quantity_log_ratio_change_5_tanh_mask"] = True
            else:
                row["options_stock_quantity_log_ratio_change_5_tanh"] = 0.0
                row["options_stock_quantity_log_ratio_change_5_tanh_mask"] = False
            # Current values enter state only after all current features are emitted.
            quantity_history.append(quantity_log)
            if ratio_valid:
                ratio_history.append(ratio)

    rows = sorted(
        (row for group in rows_by_security.values() for row in group),
        key=lambda row: (row["available_date"], row["security_id"]),
    )
    audit = {
        "activity_security_days": len(activities),
        "output_rows": len(rows),
        "distinct_securities": len(rows_by_security),
        "missing_next_session_security_days": missing_next_session,
    }
    return rows, audit


def _frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    if not rows:
        raise ValueError("COTAHIST options sidecar has no rows")
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col("source_trade_date").cast(pl.Date),
        pl.col("available_date").cast(pl.Date),
        *[pl.col(feature).cast(pl.Float32) for feature in FEATURES],
        *[pl.col(f"{feature}_mask").cast(pl.Boolean) for feature in FEATURES],
    )


def build_sidecar(
    archives: list[Path],
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
    market_dates, stock_days = load_stock_days(calendar_dir, identities, start, end)
    activities, archive_audits = read_activities(
        archives, identities, stock_days, start, end
    )
    rows, row_audit = build_rows(activities, stock_days, market_dates)
    frame = _frame(rows)
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "cotahist_options_activity.parquet"
    frame.write_parquet(output_path, compression="zstd", statistics=True)
    manifest: dict[str, object] = {
        "contract_version": CONTRACT_VERSION,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_start": start.isoformat(),
        "source_end": end.isoformat(),
        "source_archives": [asdict(audit) for audit in archive_audits],
        "calendar_source": str(calendar_dir.resolve()),
        "assignments_source": str(assignments_path.resolve()),
        "assignments_sha256": _sha256(assignments_path),
        "identity_rule": (
            "COTAHIST option CODISIN is matched exactly to accepted equity ISIN; "
            "option ticker text and ticker-prefix inference are never used"
        ),
        "option_type_rule": "TPMERC 070=call and TPMERC 080=put",
        "availability_rule": (
            "Trade-date D end-of-day COTAHIST aggregate is first usable on the next "
            "observed B3 session after D; the source has no intraday publication timestamp"
        ),
        "coverage_rule": (
            "Emit only a date/security with at least one mapped COTAHIST option-series "
            "row and an exact parsed COTAHIST equity denominator; absence is masked "
            "because no historical instrument master is available to prove listed-zero"
        ),
        "normalization_rule": (
            "Dimensionless log ratios use fixed monotone tanh compression; maturity "
            "share uses scaled asin-sqrt; moneyness uses a fixed 25% log-distance tanh; "
            "surprise uses only 20 prior observations and change uses the fifth prior "
            "observation, with current state appended after emission"
        ),
        "scope_limit": (
            "COTAHIST activity only: no open interest, delta-OI, covered/uncovered split, "
            "or implied-volatility claim"
        ),
        "features": list(FEATURES),
        "feature_valid_rows": {
            feature: int(frame.get_column(f"{feature}_mask").sum())
            for feature in FEATURES
        },
        **row_audit,
        "first_source_trade_date": frame.get_column("source_trade_date")
        .min()
        .isoformat(),
        "last_source_trade_date": frame.get_column("source_trade_date")
        .max()
        .isoformat(),
        "first_available_date": frame.get_column("available_date").min().isoformat(),
        "last_available_date": frame.get_column("available_date").max().isoformat(),
        "output_file": output_path.name,
        "output_sha256": _sha256(output_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _iso_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a leakage-safe COTAHIST options-activity sidecar"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--calendar-dir", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=_iso_date, required=True)
    parser.add_argument("--end", type=_iso_date, required=True)
    parser.add_argument("--years", nargs="+", type=int, required=True)
    args = parser.parse_args()
    archives = [args.input_dir / f"COTAHIST_A{year}.ZIP" for year in args.years]
    missing = [path for path in archives if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing COTAHIST archives: {missing}")
    manifest = build_sidecar(
        archives,
        args.calendar_dir,
        args.assignments,
        args.out,
        start=args.start,
        end=args.end,
    )
    print(
        f"Wrote {manifest['output_rows']:,} option-activity rows for "
        f"{manifest['distinct_securities']} securities to {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
