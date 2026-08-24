from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
import zipfile
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from .b3_options_open_interest import (
    FEATURES as OI_FEATURES,
    Identity,
    OptionInstrument,
    load_identities,
    parse_instruments,
)
from .io import discover_context_files, load_context_expiries
from .options_activity import StockDay, load_stock_days

CONTRACT_VERSION = "B3_FULL_OPTIONS_V1"
RECORD_LENGTH = 245
CALL_MARKET_TYPE = 70
PUT_MARKET_TYPE = 80
MIN_QUANTITY = 20
MIN_TRADES = 3
MIN_DTE = 5
MAX_DTE = 45
MIN_ATM_SERIES = 2
HISTORY = 20
IV_MIN = 0.05
IV_MAX = 3.0

VOLUME_FEATURES = (
    "options_stock_quantity_log_ratio_tanh",
    "options_put_call_quantity_log_ratio_tanh",
    "options_trade_count_surprise_20_scaled",
)
IV_FEATURES = (
    "options_atm_iv_prior20_robust_z_scaled",
    "options_atm_iv_change_1_tanh",
    "options_atm_iv_change_5_tanh",
    "options_put_skew_tanh",
    "options_iv_minus_realized20_tanh",
)
FEATURES = (*OI_FEATURES, *VOLUME_FEATURES, *IV_FEATURES)
_WORKER_IDENTITIES: Mapping[str, Identity] | None = None
_WORKER_STOCK_DAYS: Mapping[tuple[date, str], StockDay] | None = None


@dataclass(frozen=True)
class OptionDayRecord:
    ticker: str
    underlying_isin: str
    is_put: bool
    expiry: date
    strike: float
    close: float
    average: float
    trades: int
    quantity: int

    @property
    def premium(self) -> float:
        return self.close if self.close > 0 else self.average


@dataclass
class DayAudit:
    source_date: str
    cotahist_records: int = 0
    exact_instrument_matches: int = 0
    identity_mismatches: int = 0
    contract_mismatches: int = 0
    activity_rows: int = 0
    iv_filter_passes: int = 0
    iv_solutions: int = 0
    iv_bound_failures: int = 0
    parity_forward_series: int = 0
    cash_forward_series: int = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _integer(raw: bytes) -> int:
    value = raw.decode("ascii", errors="ignore").strip()
    return int(value) if value else 0


def _text(raw: bytes) -> str:
    return raw.decode("latin-1", errors="replace").strip()


def _date(raw: bytes) -> date | None:
    value = raw.decode("ascii", errors="ignore").strip()
    if not value or value == "00000000":
        return None
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError:
        return None


def _option_record(line: bytes) -> tuple[date, OptionDayRecord] | None:
    if len(line) != RECORD_LENGTH or line[:2] != b"01":
        return None
    market_type = _integer(line[24:27])
    if market_type not in (CALL_MARKET_TYPE, PUT_MARKET_TYPE):
        return None
    trade_date = _date(line[2:10])
    expiry = _date(line[202:210])
    if trade_date is None or expiry is None:
        return None
    quote_factor = _integer(line[210:217]) or 1
    price_scale = 100.0 * quote_factor
    return trade_date, OptionDayRecord(
        ticker=_text(line[12:24]),
        underlying_isin=_text(line[230:242]),
        is_put=market_type == PUT_MARKET_TYPE,
        expiry=expiry,
        strike=_integer(line[188:201]) / price_scale,
        close=_integer(line[108:121]) / price_scale,
        average=_integer(line[95:108]) / price_scale,
        trades=_integer(line[147:152]),
        quantity=_integer(line[152:170]),
    )


def _archive_member(archive: zipfile.ZipFile, source: Path) -> str:
    members = [name for name in archive.namelist() if name.upper().endswith(".TXT")]
    if not members:
        raise ValueError(f"COTAHIST archive contains no TXT member: {source}")
    preferred = [name for name in members if "COTAHIST" in name.upper()]
    return sorted(preferred or members)[0]


def read_option_records(
    archives: Sequence[Path], start: date, end: date
) -> tuple[dict[date, list[OptionDayRecord]], list[dict[str, object]]]:
    by_date: dict[date, list[OptionDayRecord]] = {}
    audits = []
    for source in archives:
        audit: dict[str, object] = {
            "source": str(source.resolve()),
            "sha256": _sha256(source),
            "physical_records": 0,
            "malformed_records": 0,
            "option_records": 0,
            "retained_option_records": 0,
        }
        with zipfile.ZipFile(source) as archive:
            member = _archive_member(archive, source)
            audit["member"] = member
            with archive.open(member) as handle:
                for raw in handle:
                    audit["physical_records"] += 1
                    line = raw.rstrip(b"\r\n")
                    if len(line) != RECORD_LENGTH:
                        audit["malformed_records"] += 1
                        continue
                    parsed = _option_record(line)
                    if parsed is None:
                        continue
                    audit["option_records"] += 1
                    trade_date, record = parsed
                    if start <= trade_date <= end:
                        by_date.setdefault(trade_date, []).append(record)
                        audit["retained_option_records"] += 1
        if audit["malformed_records"]:
            raise ValueError(f"Malformed COTAHIST records in {source}")
        audits.append(audit)
    if not by_date:
        raise ValueError("No COTAHIST option records in the requested interval")
    return by_date, audits


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _forward_option_price(
    forward: float, strike: float, rate: float, years: float, vol: float, is_put: bool
) -> float:
    scale = vol * math.sqrt(years)
    d1 = (math.log(forward / strike) + 0.5 * vol * vol * years) / scale
    d2 = d1 - scale
    discount = math.exp(-rate * years)
    if is_put:
        return discount * (strike * _normal_cdf(-d2) - forward * _normal_cdf(-d1))
    return discount * (forward * _normal_cdf(d1) - strike * _normal_cdf(d2))


def implied_volatility(
    *,
    premium: float,
    forward: float,
    strike: float,
    rate: float,
    years: float,
    is_put: bool,
) -> float | None:
    if min(premium, forward, strike, years) <= 0 or not 0 <= rate <= 1:
        return None
    low_price = _forward_option_price(
        forward, strike, rate, years, IV_MIN, is_put
    )
    high_price = _forward_option_price(
        forward, strike, rate, years, IV_MAX, is_put
    )
    tolerance = 1e-8 * max(1.0, premium)
    if premium < low_price - tolerance or premium > high_price + tolerance:
        return None
    low, high = IV_MIN, IV_MAX
    for _ in range(64):
        middle = 0.5 * (low + high)
        price = _forward_option_price(
            forward, strike, rate, years, middle, is_put
        )
        if price < premium:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _instrument_by_ticker(
    options: Sequence[OptionInstrument],
    cash_by_id: Mapping[str, str],
    source_date: date,
) -> dict[str, list[OptionInstrument]]:
    result: dict[str, list[OptionInstrument]] = {}
    for option in options:
        if (
            option.underlying_id not in cash_by_id
            or not option.trading_start <= source_date <= option.trading_end
            or option.expiry < source_date
        ):
            continue
        result.setdefault(option.ticker, []).append(option)
    return result


def _same_contract(record: OptionDayRecord, option: OptionInstrument) -> bool:
    expected_put = option.option_type in ("PUT", "PUTT")
    known_type = option.option_type == "CALL" or expected_put
    strike_tolerance = max(0.01, abs(option.strike) * 1e-6)
    return (
        known_type
        and record.is_put == expected_put
        and record.expiry == option.expiry
        and abs(record.strike - option.strike) <= strike_tolerance
    )


def _process_day(
    in_path: Path,
    source_date: date,
    records: Sequence[OptionDayRecord],
    annual_rate: float,
) -> tuple[date, dict[str, dict[str, object]], dict[str, object]]:
    if _WORKER_IDENTITIES is None or _WORKER_STOCK_DAYS is None:
        raise RuntimeError("Options worker was not initialized")
    identities = _WORKER_IDENTITIES
    stock_days = _WORKER_STOCK_DAYS
    cash_by_id, options, _ = parse_instruments(in_path, source_date, identities)
    option_by_ticker = _instrument_by_ticker(options, cash_by_id, source_date)
    audit = DayAudit(source_date=source_date.isoformat(), cotahist_records=len(records))
    mapped: dict[str, list[OptionDayRecord]] = {}
    for record in records:
        candidates = option_by_ticker.get(record.ticker, [])
        matches = [option for option in candidates if _same_contract(record, option)]
        if not matches:
            if candidates:
                audit.contract_mismatches += 1
            continue
        if len(matches) != 1:
            raise ValueError(
                f"Ambiguous exact BVBG option contract {record.ticker} on {source_date}"
            )
        option = matches[0]
        security_id = cash_by_id[option.underlying_id]
        if security_id != f"ISIN:{record.underlying_isin}":
            audit.identity_mismatches += 1
            continue
        if (source_date, security_id) not in stock_days:
            continue
        audit.exact_instrument_matches += 1
        mapped.setdefault(security_id, []).append(record)

    output: dict[str, dict[str, object]] = {}
    for security_id, security_records in mapped.items():
        audit.activity_rows += 1
        stock = stock_days[(source_date, security_id)]
        total_quantity = sum(record.quantity for record in security_records)
        call_quantity = sum(
            record.quantity for record in security_records if not record.is_put
        )
        put_quantity = total_quantity - call_quantity
        total_trades = sum(record.trades for record in security_records)
        qualifying = [
            record
            for record in security_records
            if record.quantity >= MIN_QUANTITY
            and record.trades >= MIN_TRADES
            and MIN_DTE <= (record.expiry - source_date).days <= MAX_DTE
            and record.premium > 0
            and record.strike > 0
        ]
        audit.iv_filter_passes += len(qualifying)
        pair_values: dict[tuple[date, float], dict[bool, list[float]]] = {}
        for record in qualifying:
            pair_values.setdefault((record.expiry, record.strike), {}).setdefault(
                record.is_put, []
            ).append(record.premium)
        iv_rows = []
        for record in qualifying:
            years = (record.expiry - source_date).days / 365.0
            pair = pair_values[(record.expiry, record.strike)]
            if set(pair) == {False, True}:
                forward = record.strike + math.exp(annual_rate * years) * (
                    statistics.median(pair[False]) - statistics.median(pair[True])
                )
                audit.parity_forward_series += 1
            else:
                forward = stock.close_brl
                audit.cash_forward_series += 1
            if forward <= 0:
                audit.iv_bound_failures += 1
                continue
            iv = implied_volatility(
                premium=record.premium,
                forward=forward,
                strike=record.strike,
                rate=annual_rate,
                years=years,
                is_put=record.is_put,
            )
            if iv is None:
                audit.iv_bound_failures += 1
                continue
            audit.iv_solutions += 1
            iv_rows.append(
                {
                    "iv": iv,
                    "log_moneyness": math.log(record.strike / forward),
                    "is_put": record.is_put,
                }
            )
        atm = [
            float(row["iv"])
            for row in iv_rows
            if abs(float(row["log_moneyness"])) <= 0.10
        ]
        otm_put = [
            float(row["iv"])
            for row in iv_rows
            if bool(row["is_put"])
            and -0.25 <= float(row["log_moneyness"]) <= -0.05
        ]
        output[security_id] = {
            "total_quantity": total_quantity,
            "call_quantity": call_quantity,
            "put_quantity": put_quantity,
            "total_trades": total_trades,
            "stock_quantity": stock.quantity,
            "atm_iv": statistics.median(atm) if len(atm) >= MIN_ATM_SERIES else None,
            "otm_put_iv": statistics.median(otm_put) if otm_put else None,
            "qualifying_atm_series": len(atm),
            "qualifying_otm_put_series": len(otm_put),
        }
    return source_date, output, asdict(audit)


def _initialize_worker(
    identities: Mapping[str, Identity],
    stock_days: Mapping[tuple[date, str], StockDay],
) -> None:
    global _WORKER_IDENTITIES, _WORKER_STOCK_DAYS
    _WORKER_IDENTITIES = identities
    _WORKER_STOCK_DAYS = stock_days


def load_shortest_active_di_rates(
    *, context_dir: Path, catalogue_path: Path, market_dates: Sequence[date]
) -> tuple[dict[date, float], dict[str, object]]:
    context_files = discover_context_files(context_dir)
    expiries = load_context_expiries(catalogue_path)
    closes: dict[str, dict[date, float]] = {}
    source_hashes = {}
    for symbol, expiry in sorted(expiries.items(), key=lambda item: item[1]):
        path = context_files[symbol]
        source_hashes[symbol] = {"path": str(path.resolve()), "sha256": _sha256(path)}
        frame = (
            pl.scan_parquet(path)
            .select(pl.col("ts_exchange").dt.date().alias("trade_date"), "close")
            .filter(pl.col("trade_date").is_in(market_dates), pl.col("close") > 0)
            .sort("trade_date")
            .group_by("trade_date", maintain_order=True)
            .agg(pl.col("close").last())
            .collect()
        )
        closes[symbol] = {
            row["trade_date"]: float(row["close"])
            for row in frame.iter_rows(named=True)
            if row["trade_date"] <= expiry
        }
    rates = {}
    chosen = {}
    for trade_date in market_dates:
        eligible = [
            (expiry, symbol)
            for symbol, expiry in expiries.items()
            if expiry >= trade_date and trade_date in closes[symbol]
        ]
        if not eligible:
            continue
        _, symbol = min(eligible)
        quote = closes[symbol][trade_date]
        if not 0 < quote < 100:
            raise ValueError(f"Invalid annual-percentage DI quote on {trade_date}: {quote}")
        rates[trade_date] = quote / 100.0
        chosen[symbol] = chosen.get(symbol, 0) + 1
    return rates, {
        "rule": "same-session final quote of the shortest non-expired fixed-maturity DI contract; annual percentage divided by 100",
        "chosen_date_count_by_symbol": chosen,
        "source_files": source_hashes,
    }


def _prior_robust_z(value: float, history: Sequence[float]) -> tuple[float, bool]:
    if len(history) < HISTORY:
        return 0.0, False
    prior = history[-HISTORY:]
    center = statistics.median(prior)
    mad = statistics.median(abs(item - center) for item in prior)
    scale = 1.4826 * mad
    if scale <= 1e-12:
        return 0.0, False
    return min(max((value - center) / scale, -5.0), 5.0) / 5.0, True


def _realized_volatility(
    closes: Mapping[date, float], dates: Sequence[date], position: int
) -> float | None:
    if position < HISTORY:
        return None
    window = dates[position - HISTORY : position + 1]
    values = [closes.get(day) for day in window]
    if any(value is None or value <= 0 for value in values):
        return None
    returns = np.diff(np.log(np.asarray(values, dtype=np.float64)))
    return float(np.std(returns, ddof=1) * math.sqrt(252.0))


def _feature_rows(
    *,
    parsed_days: Mapping[date, Mapping[str, Mapping[str, object]]],
    stock_days: Mapping[tuple[date, str], StockDay],
    market_dates: Sequence[date],
) -> list[dict[str, object]]:
    date_position = {value: index for index, value in enumerate(market_dates)}
    stock_closes: dict[str, dict[date, float]] = {}
    for (trade_date, security_id), stock in stock_days.items():
        stock_closes.setdefault(security_id, {})[trade_date] = stock.close_brl
    histories: dict[str, dict[str, list[tuple[date, float]]]] = {}
    rows = []
    for source_date in sorted(parsed_days):
        next_position = bisect_right(market_dates, source_date)
        if next_position >= len(market_dates):
            continue
        available_date = market_dates[next_position]
        for security_id, values in sorted(parsed_days[source_date].items()):
            history = histories.setdefault(security_id, {"trades": [], "iv": []})
            total_quantity = int(values["total_quantity"])
            call_quantity = int(values["call_quantity"])
            put_quantity = int(values["put_quantity"])
            total_trades = int(values["total_trades"])
            stock_quantity = int(values["stock_quantity"])
            row: dict[str, object] = {
                "source_trade_date": source_date,
                "available_date": available_date,
                "security_id": security_id,
                "qualifying_atm_series": int(values["qualifying_atm_series"]),
                "qualifying_otm_put_series": int(values["qualifying_otm_put_series"]),
            }
            ratio_mask = stock_quantity > 0
            row[VOLUME_FEATURES[0]] = (
                math.tanh(
                    (math.log1p(total_quantity) - math.log1p(stock_quantity)) / 4.0
                )
                if ratio_mask
                else 0.0
            )
            row[f"{VOLUME_FEATURES[0]}_mask"] = ratio_mask
            put_call_mask = total_quantity > 0
            row[VOLUME_FEATURES[1]] = (
                math.tanh(
                    (math.log1p(put_quantity) - math.log1p(call_quantity)) / 3.0
                )
                if put_call_mask
                else 0.0
            )
            row[f"{VOLUME_FEATURES[1]}_mask"] = put_call_mask
            trade_log = math.log1p(total_trades)
            trade_z, trade_mask = _prior_robust_z(
                trade_log, [item[1] for item in history["trades"]]
            )
            row[VOLUME_FEATURES[2]] = trade_z
            row[f"{VOLUME_FEATURES[2]}_mask"] = trade_mask

            atm = values["atm_iv"]
            iv_history = history["iv"]
            iv_values = [item[1] for item in iv_history]
            level, level_mask = (
                _prior_robust_z(float(atm), iv_values)
                if atm is not None
                else (0.0, False)
            )
            row[IV_FEATURES[0]] = level
            row[f"{IV_FEATURES[0]}_mask"] = level_mask
            previous_by_date = {item[0]: item[1] for item in iv_history}
            position = date_position[source_date]
            for feature, lag in zip(IV_FEATURES[1:3], (1, 5), strict=True):
                prior_date = market_dates[position - lag] if position >= lag else None
                valid = atm is not None and prior_date in previous_by_date
                row[feature] = (
                    math.tanh((float(atm) - previous_by_date[prior_date]) / 0.25)
                    if valid
                    else 0.0
                )
                row[f"{feature}_mask"] = valid
            skew_valid = atm is not None and values["otm_put_iv"] is not None
            row[IV_FEATURES[3]] = (
                math.tanh((float(values["otm_put_iv"]) - float(atm)) / 0.25)
                if skew_valid
                else 0.0
            )
            row[f"{IV_FEATURES[3]}_mask"] = skew_valid
            realized = _realized_volatility(
                stock_closes.get(security_id, {}), market_dates, position
            )
            spread_valid = atm is not None and realized is not None
            row[IV_FEATURES[4]] = (
                math.tanh((float(atm) - float(realized)) / 0.50)
                if spread_valid
                else 0.0
            )
            row[f"{IV_FEATURES[4]}_mask"] = spread_valid
            rows.append(row)
            history["trades"].append((source_date, trade_log))
            if atm is not None:
                history["iv"].append((source_date, float(atm)))
    return rows


def _merge_oi(
    rows: Sequence[Mapping[str, object]], oi_source: Path
) -> pl.DataFrame:
    new = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col("source_trade_date").cast(pl.Date),
        pl.col("available_date").cast(pl.Date),
    )
    oi = pl.read_parquet(oi_source).select(
        "source_trade_date", "available_date", "security_id",
        *OI_FEATURES, *(f"{feature}_mask" for feature in OI_FEATURES)
    )
    frame = oi.join(
        new,
        on=("source_trade_date", "available_date", "security_id"),
        how="full",
        coalesce=True,
        validate="1:1",
    ).sort("available_date", "security_id")
    for feature in FEATURES:
        frame = frame.with_columns(
            pl.col(feature).fill_null(0.0).cast(pl.Float32),
            pl.col(f"{feature}_mask").fill_null(False).cast(pl.Boolean),
        )
    return frame


def build_full_options_source(
    *,
    archives: Sequence[Path],
    bvbg_raw_dir: Path,
    calendar_dir: Path,
    assignments_path: Path,
    context_dir: Path,
    catalogue_path: Path,
    oi_source: Path,
    output_dir: Path,
    start: date,
    end: date,
    workers: int = 6,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not 1 <= workers <= 12:
        raise ValueError("Options build workers must be between one and twelve")
    identities = load_identities(assignments_path)
    market_dates, stock_days = load_stock_days(
        calendar_dir, identities, start, end
    )
    records_by_date, archive_audits = read_option_records(archives, start, end)
    rates, rate_audit = load_shortest_active_di_rates(
        context_dir=context_dir,
        catalogue_path=catalogue_path,
        market_dates=market_dates,
    )
    jobs = []
    missing_instrument_days = []
    missing_rate_days = []
    for source_date, records in sorted(records_by_date.items()):
        in_path = bvbg_raw_dir / f"IN{source_date:%y%m%d}.zip"
        if not in_path.is_file():
            missing_instrument_days.append(source_date.isoformat())
            continue
        rate = rates.get(source_date)
        if rate is None:
            missing_rate_days.append(source_date.isoformat())
            continue
        jobs.append(
            (in_path, source_date, records, rate)
        )
    if missing_instrument_days or missing_rate_days:
        raise ValueError(
            "Full options source is incomplete: "
            f"missing IN={missing_instrument_days[:5]}, DI={missing_rate_days[:5]}"
        )
    parsed = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(identities, stock_days),
    ) as pool:
        futures = [pool.submit(_process_day, *job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), 1):
            parsed.append(future.result())
            if completed % 50 == 0:
                print(f"options_days={completed}/{len(futures)}", flush=True)
    parsed.sort(key=lambda item: item[0])
    parsed_days = {item[0]: item[1] for item in parsed}
    day_audits = [item[2] for item in parsed]
    rows = _feature_rows(
        parsed_days=parsed_days, stock_days=stock_days, market_dates=market_dates
    )
    frame = _merge_oi(rows, oi_source)
    if frame.is_empty() or frame.select(
        pl.struct("available_date", "security_id").is_duplicated().any()
    ).item():
        raise ValueError("Full options source is empty or has duplicate keys")
    for feature in FEATURES:
        invalid = frame.filter(
            ~pl.col(f"{feature}_mask") & (pl.col(feature) != 0.0)
        ).height
        if invalid:
            raise ValueError(f"Invalid {feature} cells must be exactly zero")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        parquet = temporary / "full_options.parquet"
        frame.write_parquet(parquet, compression="zstd", statistics=True)
        audits_path = temporary / "daily_audits.json"
        audits_path.write_text(
            json.dumps(day_audits, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        iv_values = []
        for day in parsed_days.values():
            iv_values.extend(
                float(value["atm_iv"])
                for value in day.values()
                if value["atm_iv"] is not None
            )
        iv_array = np.asarray(iv_values, dtype=np.float64)
        if not iv_array.size:
            raise ValueError("No qualifying ATM implied-volatility observations")
        coverage = {
            feature: {
                "valid_count": int(frame.get_column(f"{feature}_mask").sum()),
                "valid_fraction": float(frame.get_column(f"{feature}_mask").mean()),
            }
            for feature in FEATURES
        }
        manifest = {
            "contract_version": CONTRACT_VERSION,
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "features": list(FEATURES),
            "optional_f2_trim_used": False,
            "source_archives": archive_audits,
            "bvbg_raw_dir": str(bvbg_raw_dir.resolve()),
            "bvbg_manifest_sha256": _sha256(bvbg_raw_dir / "manifest.json"),
            "oi_source": str(oi_source.resolve()),
            "oi_source_sha256": _sha256(oi_source),
            "oi_source_manifest_sha256": (
                _sha256(oi_source.parent / "manifest.json")
                if (oi_source.parent / "manifest.json").is_file()
                else None
            ),
            "risk_free_rate": rate_audit,
            "identity_rule": (
                "Exact same-date COTAHIST option ticker is joined to BVBG.028; "
                "type, expiry, and strike must agree; explicit BVBG underlying ID "
                "maps to the same-date accepted cash ISIN. No ticker prefix is used."
            ),
            "availability_rule": (
                "All D-dated option, cash, and DI inputs are computed after session D "
                "and first joined to the next observed B3 session without filling."
            ),
            "iv_contract": {
                "minimum_quantity": MIN_QUANTITY,
                "minimum_trades": MIN_TRADES,
                "expiry_calendar_days": [MIN_DTE, MAX_DTE],
                "premium": "COTAHIST close when positive, otherwise average",
                "forward": "same-strike/expiry qualifying put-call parity, otherwise cash close",
                "atm_log_moneyness": [-0.10, 0.10],
                "otm_put_log_moneyness": [-0.25, -0.05],
                "solver": "discounted-forward Black-Scholes bisection in [0.05,3.0]",
                "minimum_atm_series": MIN_ATM_SERIES,
                "american_option_approximation": "treated as European",
                "dividends": "ignored",
            },
            "normalization": (
                "Fixed log/tanh transforms and robust prior-20 median/MAD scaling; "
                "current values enter history only after current features are emitted."
            ),
            "output_rows": frame.height,
            "output_security_count": frame.get_column("security_id").n_unique(),
            "first_source_date": str(frame.get_column("source_trade_date").min()),
            "last_source_date": str(frame.get_column("source_trade_date").max()),
            "coverage": coverage,
            "atm_iv_percentiles": {
                str(percentile): float(np.quantile(iv_array, percentile / 100.0))
                for percentile in (1, 5, 25, 50, 75, 95, 99)
            },
            "daily_audits_sha256": _sha256(audits_path),
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
    parser = argparse.ArgumentParser(description="Build the full causal options source")
    parser.add_argument("--archive", type=Path, action="append", required=True)
    for name in (
        "bvbg_raw_dir", "calendar_dir", "assignments_path", "context_dir",
        "catalogue_path", "oi_source", "output_dir",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    print(build_full_options_source(archives=args.archive, **{
        key: value for key, value in vars(args).items() if key != "archive"
    }))


if __name__ == "__main__":
    main()
