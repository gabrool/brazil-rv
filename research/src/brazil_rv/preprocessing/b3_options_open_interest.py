from __future__ import annotations

import argparse
import hashlib
import io
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
import zipfile
from bisect import bisect_right
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from xml.etree.ElementTree import Element, iterparse

import polars as pl

from .bdi_lending import load_stock_days

CONTRACT_VERSION = "B3_BVBG_OPTIONS_OPEN_INTEREST_V1"
RAW_CONTRACT_VERSION = "B3_BVBG_028_086_OPTIONS_SNAPSHOT_V1"
DOWNLOAD_URL = "https://www.b3.com.br/pesquisapregao/download?filelist={name}"
VALID_ISIN = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")
ADV_OBSERVATIONS = 20
SURPRISE_OBSERVATIONS = 20
SURPRISE_CLIP = 5.0
FEATURES = (
    "options_oi_to_stock_adv20_log_tanh",
    "options_oi_change_to_stock_adv20_tanh",
    "options_put_call_oi_log_ratio_tanh",
    "options_near_expiry_oi_share_asin_sqrt",
    "options_oi_weighted_abs_log_moneyness_tanh",
    "options_total_oi_surprise_20_scaled",
)


@dataclass(frozen=True)
class Identity:
    security_id: str
    isin: str
    effective_from: date
    effective_to_inclusive: date


@dataclass(frozen=True)
class OptionInstrument:
    instrument_id: str
    underlying_id: str
    ticker: str
    option_type: str
    strike: float
    expiry: date
    trading_start: date
    trading_end: date


@dataclass
class OptionAggregate:
    active_series: int = 0
    reported_series: int = 0
    call_oi: int = 0
    put_oi: int = 0
    near_expiry_oi: int = 0
    valid_moneyness_oi: int = 0
    weighted_abs_log_moneyness: float = 0.0

    @property
    def total_oi(self) -> int:
        return self.call_oi + self.put_oi


@dataclass(frozen=True)
class ParsedDay:
    source_date: date
    cash_quantity: dict[str, int]
    aggregates: dict[str, OptionAggregate]
    instrument_count: int
    option_count: int
    mapped_option_count: int
    reported_option_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(element: Element, name: str) -> list[Element]:
    return [child for child in element if _local(child.tag) == name]


def _descendant(element: Element, name: str) -> Element | None:
    return next((child for child in element.iter() if _local(child.tag) == name), None)


def _text(element: Element | None, name: str) -> str | None:
    if element is None:
        return None
    child = _descendant(element, name)
    if child is None or child.text is None:
        return None
    value = child.text.strip()
    return value or None


def _nested_xml_member(path: Path) -> tuple[bytes, str]:
    with zipfile.ZipFile(path) as outer:
        members = [entry for entry in outer.infolist() if not entry.is_dir()]
        if len(members) != 1:
            raise ValueError(f"Expected one nested archive in {path}")
        nested = outer.read(members[0])
    with zipfile.ZipFile(io.BytesIO(nested)) as inner:
        xml = sorted(
            (entry for entry in inner.infolist() if entry.filename.endswith(".xml")),
            key=lambda entry: entry.filename,
        )
        if not xml:
            raise ValueError(f"Nested archive contains no XML: {path}")
        selected = xml[-1]
        return inner.read(selected), selected.filename


def _validate_download(payload: bytes) -> dict[str, object] | None:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as outer:
            members = [entry for entry in outer.infolist() if not entry.is_dir()]
            if not members:
                return None
            if len(members) != 1:
                raise ValueError("B3 archive must contain one nested ZIP")
            nested = outer.read(members[0])
        with zipfile.ZipFile(io.BytesIO(nested)) as inner:
            xml = sorted(
                entry.filename
                for entry in inner.infolist()
                if entry.filename.endswith(".xml") and entry.file_size > 0
            )
            if not xml:
                raise ValueError("B3 nested archive contains no nonempty XML")
            return {
                "nested_archive": members[0].filename,
                "xml_member_count": len(xml),
                "selected_latest_xml": xml[-1],
            }
    except zipfile.BadZipFile as error:
        raise ValueError("B3 response is not a valid ZIP") from error


def _download(name: str, destination: Path) -> dict[str, object]:
    url = DOWNLOAD_URL.format(name=name)
    request = urllib.request.Request(url, headers={"User-Agent": "Brazil-RV research"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
            audit = _validate_download(payload)
            if audit is None:
                return {"name": name, "status": "not_published", "url": url}
            destination.write_bytes(payload)
            return {
                "name": name,
                "status": "downloaded",
                "url": url,
                "filename": destination.name,
                "bytes": len(payload),
                "sha256": _sha256(destination),
                **audit,
            }
        except (OSError, urllib.error.URLError):
            if attempt == 2:
                raise
            time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"Failed to acquire B3 archive: {name}")


def acquire_sources(
    output_dir: Path,
    *,
    start: date,
    end: date,
    workers: int = 8,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    requests = [
        (day, f"{prefix}{day:%y%m%d}.zip") for day in days for prefix in ("PR", "IN")
    ]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_download, name, temporary / name): (day, name)
                for day, name in requests
            }
            records = []
            for future in as_completed(futures):
                day, _ = futures[future]
                records.append({"trade_date": day.isoformat(), **future.result()})
        records.sort(key=lambda row: (str(row["trade_date"]), str(row["name"])))
        published = {
            (str(row["trade_date"]), str(row["name"])[:2])
            for row in records
            if row["status"] == "downloaded"
        }
        complete_days = [
            day
            for day in days
            if (day.isoformat(), "PR") in published
            and (day.isoformat(), "IN") in published
        ]
        manifest = {
            "contract_version": RAW_CONTRACT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "requested_start": start.isoformat(),
            "requested_end": end.isoformat(),
            "requested_weekdays": len(days),
            "downloaded_archive_count": sum(
                row["status"] == "downloaded" for row in records
            ),
            "complete_pr_in_day_count": len(complete_days),
            "first_complete_day": complete_days[0].isoformat(),
            "last_complete_day": complete_days[-1].isoformat(),
            "selection_rule": (
                "Each public outer ZIP contains a nested ZIP; use the "
                "lexicographically latest timestamped XML member as the final daily state"
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


def load_identities(path: Path) -> dict[str, Identity]:
    frame = pl.read_parquet(path).select(
        "security_id", "isin", "first_overlap_date", "last_overlap_date"
    )
    identities = {}
    for row in frame.iter_rows(named=True):
        isin = row["isin"]
        if VALID_ISIN.fullmatch(isin) is None or row["security_id"] != f"ISIN:{isin}":
            raise ValueError("Option OI identity must be an exact accepted ISIN")
        identities[isin] = Identity(
            security_id=row["security_id"],
            isin=isin,
            effective_from=date.fromisoformat(row["first_overlap_date"]),
            effective_to_inclusive=date.fromisoformat(row["last_overlap_date"]),
        )
    return identities


def _instrument_id(container: Element | None) -> str | None:
    if container is None:
        return None
    identifier = _descendant(container, "FinInstrmId")
    return _text(identifier, "Id")


def parse_instruments(
    path: Path, source_date: date, identities: Mapping[str, Identity]
) -> tuple[dict[str, str], list[OptionInstrument], dict[str, object]]:
    payload, member = _nested_xml_member(path)
    cash_by_id: dict[str, str] = {}
    options: list[OptionInstrument] = []
    instrument_count = 0
    with io.BytesIO(payload) as handle:
        for _, element in iterparse(handle, events=("end",)):
            if _local(element.tag) != "Instrm":
                continue
            instrument_count += 1
            outer_id = _instrument_id(element)
            info = _descendant(element, "InstrmInf")
            equity = _descendant(info, "EqtyInf")
            option = _descendant(info, "OptnOnEqtsInf")
            if outer_id is not None and equity is not None:
                isin = _text(equity, "ISIN")
                identity = identities.get(isin or "")
                if identity is not None and (
                    identity.effective_from
                    <= source_date
                    <= identity.effective_to_inclusive
                ):
                    cash_by_id[outer_id] = identity.security_id
            elif outer_id is not None and option is not None:
                underlying = _descendant(option, "UndrlygInstrmId")
                underlying_id = _text(underlying, "Id")
                ticker = _text(option, "TckrSymb")
                option_type = _text(option, "OptnTp")
                strike = _text(option, "ExrcPric")
                expiry = _text(option, "XprtnDt")
                trading_start = _text(option, "TradgStartDt")
                trading_end = _text(option, "TradgEndDt")
                if all(
                    value is not None
                    for value in (
                        underlying_id,
                        ticker,
                        option_type,
                        strike,
                        expiry,
                        trading_start,
                        trading_end,
                    )
                ):
                    options.append(
                        OptionInstrument(
                            instrument_id=outer_id,
                            underlying_id=str(underlying_id),
                            ticker=str(ticker),
                            option_type=str(option_type),
                            strike=float(strike),
                            expiry=date.fromisoformat(str(expiry)),
                            trading_start=date.fromisoformat(str(trading_start)),
                            trading_end=date.fromisoformat(str(trading_end)),
                        )
                    )
            element.clear()
    return (
        cash_by_id,
        options,
        {
            "instrument_xml_member": member,
            "instrument_count": instrument_count,
            "accepted_cash_instrument_count": len(cash_by_id),
            "equity_option_instrument_count": len(options),
        },
    )


def parse_price_report(
    path: Path,
    source_date: date,
    cash_by_id: Mapping[str, str],
    options: Iterable[OptionInstrument],
) -> tuple[dict[str, int], dict[str, OptionAggregate], dict[str, object]]:
    option_by_id = {
        option.instrument_id: option
        for option in options
        if option.trading_start <= source_date <= option.trading_end
        and option.expiry >= source_date
        and option.underlying_id in cash_by_id
    }
    underlying_ids = {option.underlying_id for option in option_by_id.values()}
    aggregates: dict[str, OptionAggregate] = {}
    for option in option_by_id.values():
        security_id = cash_by_id[option.underlying_id]
        aggregates.setdefault(security_id, OptionAggregate()).active_series += 1
    cash_quantity: dict[str, int] = {}
    cash_close: dict[str, float] = {}
    price_rows = 0
    payload, member = _nested_xml_member(path)
    option_rows: list[tuple[OptionInstrument, int]] = []
    with io.BytesIO(payload) as handle:
        for _, element in iterparse(handle, events=("end",)):
            if _local(element.tag) != "PricRpt":
                continue
            price_rows += 1
            identifier = _instrument_id(element)
            if identifier in underlying_ids:
                security_id = cash_by_id[identifier]
                quantity = _text(element, "FinInstrmQty")
                close = _text(element, "LastPric")
                if quantity is not None:
                    cash_quantity[security_id] = int(quantity)
                if close is not None and float(close) > 0:
                    cash_close[security_id] = float(close)
            option = option_by_id.get(identifier or "")
            if option is not None:
                oi = _text(element, "OpnIntrst")
                if oi is not None and int(oi) >= 0:
                    option_rows.append((option, int(oi)))
            element.clear()

    for option, oi in option_rows:
        security_id = cash_by_id[option.underlying_id]
        aggregate = aggregates[security_id]
        aggregate.reported_series += 1
        if option.option_type == "CALL":
            aggregate.call_oi += oi
        elif option.option_type in ("PUT", "PUTT"):
            aggregate.put_oi += oi
        else:
            raise ValueError(f"Unknown option type: {option.option_type}")
        if (option.expiry - source_date).days <= 30:
            aggregate.near_expiry_oi += oi
        close = cash_close.get(security_id)
        if oi > 0 and close is not None and option.strike > 0:
            aggregate.valid_moneyness_oi += oi
            aggregate.weighted_abs_log_moneyness += oi * abs(
                math.log(option.strike / close)
            )
    return (
        cash_quantity,
        aggregates,
        {
            "price_xml_member": member,
            "price_row_count": price_rows,
            "active_mapped_option_count": len(option_by_id),
            "reported_mapped_option_count": len(option_rows),
        },
    )


def _parse_day_job(
    in_path: Path,
    pr_path: Path,
    source_date: date,
    identities: Mapping[str, Identity],
) -> tuple[ParsedDay, dict[str, object]]:
    cash_by_id, options, instrument_audit = parse_instruments(
        in_path, source_date, identities
    )
    cash_quantity, aggregates, price_audit = parse_price_report(
        pr_path, source_date, cash_by_id, options
    )
    if (
        int(instrument_audit["instrument_count"]) < 10_000
        or len(cash_by_id) < 50
        or int(price_audit["price_row_count"]) < 5_000
    ):
        raise ValueError(f"Incomplete BVBG.028/BVBG.086 daily pair for {source_date}")
    return (
        ParsedDay(
            source_date=source_date,
            cash_quantity=cash_quantity,
            aggregates=aggregates,
            instrument_count=int(instrument_audit["instrument_count"]),
            option_count=len(options),
            mapped_option_count=sum(
                option.underlying_id in cash_by_id for option in options
            ),
            reported_option_count=int(price_audit["reported_mapped_option_count"]),
        ),
        {
            "source_date": source_date.isoformat(),
            "in_path": str(in_path.resolve()),
            "in_sha256": _sha256(in_path),
            "pr_path": str(pr_path.resolve()),
            "pr_sha256": _sha256(pr_path),
            **instrument_audit,
            **price_audit,
        },
    )


def _prior_surprise(value: float, history: list[float]) -> tuple[float, bool]:
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


def build_features(
    *,
    raw_dir: Path,
    calendar_dir: Path,
    assignments_path: Path,
    output_dir: Path,
    start: date,
    end: date,
    workers: int = 6,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    identities = load_identities(assignments_path)
    pairs = []
    for pr_path in sorted(raw_dir.glob("PR*.zip")):
        source_date = datetime.strptime(pr_path.stem[2:], "%y%m%d").date()
        in_path = raw_dir / f"IN{source_date:%y%m%d}.zip"
        if start <= source_date <= end and in_path.is_file():
            pairs.append((in_path, pr_path, source_date, identities))
    if not pairs:
        raise ValueError("No complete BVBG.028/BVBG.086 daily pairs")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_parse_day_job, *job) for job in pairs]
        parsed = [future.result() for future in as_completed(futures)]
    parsed.sort(key=lambda item: item[0].source_date)
    days = [item[0] for item in parsed]
    audits = [item[1] for item in parsed]
    market_dates, stock_days = load_stock_days(calendar_dir, identities, end)

    quantity_history: dict[str, list[int]] = {
        identity.security_id: [] for identity in identities.values()
    }
    oi_history: dict[str, list[float]] = {
        identity.security_id: [] for identity in identities.values()
    }
    previous_oi: dict[str, tuple[date, int]] = {}
    output_rows = []
    for day in days:
        next_position = bisect_right(market_dates, day.source_date)
        if next_position >= len(market_dates):
            continue
        available_date = market_dates[next_position]
        for identity in identities.values():
            stock = stock_days.get((day.source_date, identity.security_id))
            if stock is not None:
                quantity_history.setdefault(identity.security_id, []).append(
                    stock.quantity
                )
        for security_id, aggregate in day.aggregates.items():
            if aggregate.active_series <= 0:
                continue
            quantities = quantity_history.get(security_id, [])
            adv_valid = len(quantities) >= ADV_OBSERVATIONS
            adv = statistics.fmean(quantities[-ADV_OBSERVATIONS:]) if adv_valid else 0.0
            total_oi = aggregate.total_oi
            previous = previous_oi.get(security_id)
            source_position = bisect_right(market_dates, day.source_date) - 1
            previous_source_date = (
                market_dates[source_position - 1] if source_position > 0 else None
            )
            oi_to_adv_valid = adv_valid and adv > 0
            change_valid = (
                oi_to_adv_valid
                and previous is not None
                and previous[0] == previous_source_date
            )
            put_call_valid = total_oi > 0
            near_valid = total_oi > 0
            moneyness_valid = total_oi > 0 and aggregate.valid_moneyness_oi == total_oi
            surprise, surprise_valid = _prior_surprise(
                math.log1p(total_oi), oi_history.setdefault(security_id, [])
            )
            row: dict[str, object] = {
                "source_trade_date": day.source_date,
                "available_date": available_date,
                "security_id": security_id,
                "active_option_series": aggregate.active_series,
                "total_open_interest": total_oi,
            }
            row[FEATURES[0]] = (
                math.tanh(math.log1p(total_oi / adv) / 4.0) if oi_to_adv_valid else 0.0
            )
            row[f"{FEATURES[0]}_mask"] = oi_to_adv_valid
            change = 0 if previous is None else total_oi - previous[1]
            row[FEATURES[1]] = (
                math.tanh(math.copysign(math.log1p(abs(change) / adv), change) / 3.0)
                if change_valid
                else 0.0
            )
            row[f"{FEATURES[1]}_mask"] = change_valid
            row[FEATURES[2]] = (
                math.tanh(
                    (math.log1p(aggregate.put_oi) - math.log1p(aggregate.call_oi)) / 3.0
                )
                if put_call_valid
                else 0.0
            )
            row[f"{FEATURES[2]}_mask"] = put_call_valid
            row[FEATURES[3]] = (
                math.asin(math.sqrt(aggregate.near_expiry_oi / total_oi))
                / (math.pi / 2)
                if near_valid
                else 0.0
            )
            row[f"{FEATURES[3]}_mask"] = near_valid
            row[FEATURES[4]] = (
                math.tanh(
                    aggregate.weighted_abs_log_moneyness
                    / aggregate.valid_moneyness_oi
                    / 2.0
                )
                if moneyness_valid
                else 0.0
            )
            row[f"{FEATURES[4]}_mask"] = moneyness_valid
            row[FEATURES[5]] = surprise
            row[f"{FEATURES[5]}_mask"] = surprise_valid
            output_rows.append(row)
            previous_oi[security_id] = (day.source_date, total_oi)
            oi_history[security_id].append(math.log1p(total_oi))

    frame = (
        pl.DataFrame(output_rows)
        .sort("source_trade_date")
        .unique(
            subset=("available_date", "security_id"), keep="last", maintain_order=True
        )
        .sort("available_date", "security_id")
    )
    if frame.is_empty():
        raise ValueError(
            "No complete option OI observations mapped to accepted equities"
        )
    if frame.select(
        pl.struct("available_date", "security_id").is_duplicated().any()
    ).item():
        raise ValueError("Option OI features contain duplicate keys")
    for feature in FEATURES:
        if frame.filter(~pl.col(f"{feature}_mask") & (pl.col(feature) != 0)).height:
            raise ValueError("Invalid option OI values must be exactly zero")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        parquet = temporary / "b3_options_open_interest.parquet"
        frame.write_parquet(parquet, compression="zstd")
        audit_path = temporary / "daily_audits.json"
        audit_path.write_text(
            json.dumps(audits, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        manifest = {
            "contract_version": CONTRACT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "features": list(FEATURES),
            "source": str(raw_dir.resolve()),
            "source_manifest_sha256": _sha256(raw_dir / "manifest.json"),
            "availability_rule": (
                "Final BVBG.086 report for session D is produced after all B3 "
                "decisions and first used at the next observed B3 session"
            ),
            "identity_rule": (
                "BVBG.086 option instrument ID -> same-date BVBG.028 option -> "
                "explicit UnderlyingInstrumentId -> same-date BVBG.028 cash ISIN -> "
                "accepted permanent security_id; ticker prefixes are never parsed"
            ),
            "source_limit": (
                "BVBG.086 supplies per-series open interest but not the covered/"
                "uncovered split in DerivativesOpenPositionFile; historical bodies "
                "for that separate file were zero bytes, so covered/uncovered and PIN "
                "were not fabricated"
            ),
            "absent_series_rule": (
                "An active BVBG.028 option series absent from the complete final "
                "BVBG.086 daily snapshot contributes zero open interest; the daily "
                "audit retains active and reported series counts"
            ),
            "normalization": (
                "Fixed log/tanh or asin-sqrt transforms; stock ADV and OI surprise "
                "use trailing observed histories with no future observations"
            ),
            "daily_pair_count": len(days),
            "output_rows": frame.height,
            "output_security_count": frame.get_column("security_id").n_unique(),
            "first_source_date": str(frame.get_column("source_trade_date").min()),
            "last_source_date": str(frame.get_column("source_trade_date").max()),
            "output_sha256": _sha256(parquet),
            "daily_audits_sha256": _sha256(audit_path),
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
        description="Build strong-form B3 option OI features"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    acquire = subparsers.add_parser("acquire")
    acquire.add_argument("--output-dir", type=Path, required=True)
    acquire.add_argument("--start", type=date.fromisoformat, required=True)
    acquire.add_argument("--end", type=date.fromisoformat, required=True)
    acquire.add_argument("--workers", type=int, default=8)
    build = subparsers.add_parser("build")
    build.add_argument("--raw-dir", type=Path, required=True)
    build.add_argument("--calendar-dir", type=Path, required=True)
    build.add_argument("--assignments-path", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--start", type=date.fromisoformat, required=True)
    build.add_argument("--end", type=date.fromisoformat, required=True)
    build.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    kwargs = {key: value for key, value in vars(args).items() if key != "command"}
    print(
        acquire_sources(**kwargs)
        if args.command == "acquire"
        else build_features(**kwargs)
    )


if __name__ == "__main__":
    main()
