"""Date-bounded exchange archives for the Round-5 cross-market data audit.

The acquisition step preserves public source responses. It does not infer
historical ADR conversion ratios or turn vendor-adjusted prices into cash prices.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import urllib.error
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import zipfile

import polars as pl

from brazil_rv.preprocessing.shfe_daily import parse_snapshot

from .artifacts import sha256_file, write_json_atomic
from .contract import DEVELOPMENT_END
from .round5_market import OBSERVATION_SCHEMA, _get

SHANGHAI = ZoneInfo("Asia/Shanghai")
SAO_PAULO = ZoneInfo("America/Sao_Paulo")
NEW_YORK = ZoneInfo("America/New_York")
TENORS = (30, 90, 180, 360, 720, 1080)
US_SYMBOLS = (
    "EWZ",
    "ABEV",
    "BBD",
    "BBDO",
    "BAK",
    "BSBR",
    "CIG",
    "SID",
    "EMBJ",
    "GGB",
    "ITUB",
    "PBR",
    "PBR-A",
    "SBS",
    "SUZ",
    "TIMB",
    "UGP",
    "VALE",
    "VIV",
)
SHFE_URL = "https://www.shfe.com.cn/data/tradedata/future/dailydata/kx{day:%Y%m%d}.dat"
DI_URL = "https://www.b3.com.br/pesquisapregao/download?filelist=TS{day:%y%m%d}.ex_,"
DCE_URL = (
    "https://stock2.finance.sina.com.cn/futures/api/jsonp.php/var%20_brazil_rv="
    "/InnerFuturesNewService.getDailyKLine?symbol={symbol}&type=2021_04_12"
)
LAYOUT_URL = (
    "https://www.b3.com.br/data/files/E6/B6/70/64/BB1379106B8BCB69AC094EA8/"
    "TaxaSwap%20para%20UP2DATA.xlsx"
)
SHFE_CALENDAR_URL = "https://www.shfe.com.cn/data/config/js/trade-data.js"


def _weekdays(start: date, end: date) -> list[date]:
    return [
        start + timedelta(days=i)
        for i in range((end - start).days + 1)
        if (start + timedelta(days=i)).weekday() < 5
    ]


def _fetch(job: tuple[str, Path, str, str]) -> dict:
    family, path, url, key = job
    try:
        source = _get(path, url)
        return {"family": family, "key": key, "status": "retrieved", **source}
    except (OSError, ValueError) as error:
        return {
            "family": family,
            "key": key,
            "path": str(path),
            "url": url,
            "status": "unavailable",
            "error": str(error),
        }


def acquire(
    root: Path, shfe_root: Path, dce_root: Path, families: list[str], workers: int = 4
) -> dict:
    """Resume immutable response caching; acquire only explicit historical dates.

    Expiries after 2024 are not requested. Existing 2025-expiry contracts are
    parsed through their <=2024 prefix in the separate build step.
    """
    root.mkdir(parents=True, exist_ok=True)
    jobs = []
    records = []
    if "di" in families:
        for day in _weekdays(date(2010, 1, 4), DEVELOPMENT_END):
            jobs.append(
                (
                    "di",
                    root / "raw" / "di" / f"TS{day:%y%m%d}.zip",
                    DI_URL.format(day=day),
                    day.isoformat(),
                )
            )
        jobs.append(
            (
                "documentation",
                root / "documentation" / "TaxaSwap_layout.xlsx",
                LAYOUT_URL,
                "TaxaSwap_layout",
            )
        )
    if "shfe" in families:
        manifest = json.loads((shfe_root / "manifest.json").read_text())
        existing = {row["report_date"]: row for row in manifest["source_files"]}
        for day in _weekdays(date(2013, 1, 1), DEVELOPMENT_END):
            if day.isoformat() in existing:
                row = existing[day.isoformat()]
                path = shfe_root / row["relative_path"]
                if sha256_file(path) != row["sha256"]:
                    raise ValueError(f"immutable SHFE source changed: {path}")
                records.append(
                    {
                        "family": "shfe",
                        "key": day.isoformat(),
                        "status": "retrieved",
                        "path": str(path),
                        "sha256": row["sha256"],
                        "reused": True,
                    }
                )
            else:
                jobs.append(
                    (
                        "shfe",
                        root / "raw" / "shfe" / f"kx{day:%Y%m%d}.dat",
                        SHFE_URL.format(day=day),
                        day.isoformat(),
                    )
                )
    if "dce" in families:
        # Full expired-contract responses have no held-out rows. Missing expired
        # contracts are recorded as null responses, never replaced by I0.
        for year in range(2014, 2019):
            for month in range(1, 13):
                symbol = f"I{year % 100:02d}{month:02d}"
                jobs.append(
                    (
                        "dce",
                        root / "raw" / "dce" / f"{symbol}.json",
                        DCE_URL.format(symbol=symbol),
                        symbol,
                    )
                )
    if "us" in families:
        p1 = int(datetime(2010, 1, 1, tzinfo=UTC).timestamp())
        p2 = int(
            datetime.combine(
                DEVELOPMENT_END + timedelta(days=1), time(), UTC
            ).timestamp()
        )
        for symbol in US_SYMBOLS:
            query = urllib.parse.urlencode(
                {
                    "period1": p1,
                    "period2": p2,
                    "interval": "1d",
                    "events": "div,splits",
                    "includeAdjustedClose": "true",
                }
            )
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{query}"
            jobs.append(("us", root / "raw" / "us" / f"{symbol}.json", url, symbol))
    if {"shfe", "dce"} & set(families):
        jobs.append(
            (
                "documentation",
                root / "documentation" / "shfe_trade_data.js",
                SHFE_CALENDAR_URL,
                "shfe_nontrading_calendar",
            )
        )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), start=1):
            records.append(future.result())
            if index % 50 == 0 or index == len(jobs):
                progress = {
                    "completed": index,
                    "requested": len(jobs),
                    "families": families,
                    "retrieved": sum(x["status"] == "retrieved" for x in records),
                    "unavailable": sum(x["status"] != "retrieved" for x in records),
                }
                write_json_atomic(
                    root / ("progress_" + "_".join(families) + ".json"), progress
                )
                print(json.dumps(progress), flush=True)
    result = {
        "schema": "BRAZIL_RV_ROUND5_CROSS_MARKET_ACQUISITION_V1",
        "created_at": datetime.now(UTC).isoformat(),
        "end": DEVELOPMENT_END.isoformat(),
        "shfe_root": str(shfe_root),
        "dce_root": str(dce_root),
        "records": sorted(records, key=lambda r: (r["family"], r["key"])),
    }
    write_json_atomic(root / ("acquisition_" + "_".join(families) + ".json"), result)
    return result


def parse_di(payload: bytes, expected_date: date) -> list[dict]:
    """Read the official fixed PRE vertices, with their actual rolled maturities.

    The layout's first-sheet date example says DDMMAAAA but the same workbook's
    CurveFile mapping and the historical bytes use YYYYMMDD. Date agreement is
    verified against the requested archive. Rates have seven implied decimals.
    Fixed vertex 90 can have 91 actual calendar days after a holiday/weekend roll;
    selecting by actual days would unnecessarily lose the requested 90d quote.
    """
    with zipfile.ZipFile(io.BytesIO(payload)) as outer:
        names = outer.namelist()
        if not names:
            return []
        if len(names) != 1:
            raise ValueError("TaxaSwap download has multiple source archives")
        nested = outer.read(names[0])
    with zipfile.ZipFile(io.BytesIO(nested)) as inner:
        text = inner.read("TaxaSwap.txt").decode("latin1")
    result = []
    for line in text.splitlines():
        if line[21:26].strip() != "PRE" or line[66:67] != "F":
            continue
        if len(line) != 72 or line[19:21] != "T1":
            raise ValueError("unrecognized TaxaSwap PRE layout")
        day = datetime.strptime(line[11:19], "%Y%m%d").date()
        if day != expected_date or day > DEVELOPMENT_END:
            raise ValueError(
                f"TaxaSwap source date {day} differs from bounded request {expected_date}"
            )
        tenor = int(line[67:72])
        if tenor not in TENORS:
            continue
        result.append(
            {
                "reference_date": day,
                "tenor": tenor,
                "calendar_days": int(line[41:46]),
                "working_days": int(line[46:51]),
                "rate_pct_252": int(line[51:66]) / 1e7,
            }
        )
    if len({row["tenor"] for row in result}) != len(result):
        raise ValueError("TaxaSwap has duplicate requested PRE vertices")
    return result


def dce_prefix(path: Path, end: date = DEVELOPMENT_END) -> tuple[list[dict], dict]:
    """Consume ordered JSONP rows only through end, without parsing future values.

    Date must be the first field in each object. The first later date is read as
    a boundary marker; its price/volume/OI payload is never decoded. An unbuffered
    handle prevents accidental read-ahead into the following payload.
    """
    rows = []
    digest = hashlib.sha256()
    consumed = 0
    boundary = None
    with path.open("rb", buffering=0) as handle:

        def read() -> bytes:
            nonlocal consumed
            value = handle.read(1)
            digest.update(value)
            consumed += len(value)
            return value

        prefix = b""
        while not prefix.endswith(b"=("):
            value = read()
            if not value:
                raise ValueError("unrecognized DCE JSONP wrapper")
            prefix += value
        first = read()
        if first == b"n":
            if b"n" + b"".join(read() for _ in range(3)) != b"null":
                raise ValueError("invalid DCE null response")
        elif first != b"[":
            raise ValueError("DCE JSONP does not contain an array")
        else:
            while True:
                value = read()
                while value in (b" ", b"\n", b"\r", b"\t", b","):
                    value = read()
                if value == b"]":
                    break
                if value != b"{":
                    raise ValueError("invalid DCE row boundary")
                fragment = value
                # The source's first field is exactly its reference date. Stop
                # after that quoted date, before reading the comma or any value.
                while not re.fullmatch(
                    rb'\{\s*"d"\s*:\s*"\d{4}-\d{2}-\d{2}"', fragment
                ):
                    value = read()
                    fragment += value
                    if not value or len(fragment) > 40:
                        raise ValueError("DCE date is not the first object field")
                day = date.fromisoformat(fragment[-11:-1].decode())
                if day > end:
                    boundary = day.isoformat()
                    break
                while not fragment.endswith(b"}"):
                    value = read()
                    if not value:
                        raise ValueError("truncated DCE source row")
                    fragment += value
                row = json.loads(fragment)
                if rows and row["d"] <= rows[-1]["d"]:
                    raise ValueError("DCE rows are not strictly chronological")
                rows.append(row)
    return rows, {
        "consumed_bytes": consumed,
        "consumed_prefix_sha256": digest.hexdigest(),
        "first_excluded_date_marker": boundary,
        "parsed_rows": len(rows),
        "last_parsed_date": rows[-1]["d"] if rows else None,
    }


def roll_returns(
    contracts: list[dict], unavailable_days: dict[str, set[date]] | None = None
) -> list[dict]:
    """Lagged-OI main contract, comparing that exact contract on adjacent days.

    Selection sees only the preceding source session. A missing chosen current
    contract yields no return; no replacement is chosen using current data. No
    absolute rolled price splice, future back-adjustment, or terminal OI is used.
    """
    unavailable_days = unavailable_days or {}
    by_product: dict[str, dict[date, dict[str, dict]]] = {}
    for row in contracts:
        if row["reference_date"] > DEVELOPMENT_END:
            raise ValueError("held-out contract observation")
        day_rows = by_product.setdefault(row["product"], {}).setdefault(
            row["reference_date"], {}
        )
        if row["contract"] in day_rows:
            raise ValueError("duplicate contract/session")
        day_rows[row["contract"]] = row
    output = []
    for product, days in sorted(by_product.items()):
        previous = None
        for day, current in sorted(days.items()):
            if previous is not None:
                eligible = [r for r in previous.values() if r["open_interest"] > 0]
                if eligible:
                    chosen = min(
                        eligible, key=lambda r: (-r["open_interest"], r["contract"])
                    )
                    now = current.get(chosen["contract"])
                    missing_between = any(
                        chosen["reference_date"] < failed < day
                        for failed in unavailable_days.get(product, set())
                    )
                    if (
                        now is not None
                        and now["volume"] > 0
                        and chosen["volume"] > 0
                        and not missing_between
                    ):
                        output.append(
                            {
                                "series": product,
                                "reference_date": day,
                                "previous_date": chosen["reference_date"],
                                "available_at": now["available_at"],
                                "log_return": math.log(
                                    now["settlement"] / chosen["settlement"]
                                ),
                                "selected_contract": chosen["contract"],
                                "selection_open_interest": chosen["open_interest"],
                                "source_file": now["source_file"],
                            }
                        )
            previous = current
    return output


def us_close(day: date) -> datetime:
    """Regular or scheduled early equity close for an observed 2010-24 US day.

    These years have 13:00 closes on the observed July 3, Christmas Eve, and
    Friday after the fourth Thursday in November. The source bars determine
    whether a session exists; this is not a synthetic holiday/trading calendar.
    Historical NYSE/ICE releases and US exchange notices are bound in the audit.
    """
    if not date(2010, 1, 1) <= day <= DEVELOPMENT_END:
        raise ValueError("US close schedule is bounded to the registered history")
    early = (
        (day.month == 7 and day.day == 3 and day.weekday() < 4)
        or (day.month == 12 and day.day == 24 and day.weekday() < 4)
        or (day.month == 11 and day.weekday() == 4 and 23 <= day.day <= 29)
    )
    return datetime.combine(day, time(13 if early else 16), NEW_YORK).astimezone(UTC)


def parse_us(payload: bytes, symbol: str, source_file: str) -> list[dict]:
    document = json.loads(payload)
    chart = document["chart"]
    if chart.get("error") or not chart.get("result"):
        raise ValueError(f"Yahoo chart unavailable for {symbol}: {chart.get('error')}")
    result = chart["result"][0]
    if result["meta"]["symbol"] != symbol or result["meta"].get("currency") != "USD":
        raise ValueError("Yahoo symbol/currency identity differs from request")
    quote = result["indicators"]["quote"][0]
    adjusted = result["indicators"].get("adjclose", [{}])[0].get("adjclose", [])
    timestamps = result["timestamp"]
    if any(
        len(quote[field]) != len(timestamps)
        for field in ("open", "high", "low", "close", "volume")
    ) or (adjusted and len(adjusted) != len(timestamps)):
        raise ValueError("Yahoo price arrays differ from timestamp axis")
    rows = []
    for index, timestamp in enumerate(timestamps):
        day = datetime.fromtimestamp(timestamp, UTC).astimezone(NEW_YORK).date()
        if day > DEVELOPMENT_END or day < date(2010, 1, 1):
            raise ValueError("Yahoo historical rows escaped bounded request")
        row = {
            "symbol": symbol,
            "reference_date": day,
            "source_file": source_file,
            "adjusted_close": adjusted[index] if adjusted else None,
            "available_at": None
            if symbol == "SUZ" and day < date(2018, 12, 10)
            else us_close(day),
            "availability_status": "historical_otc_clock_unverified"
            if symbol == "SUZ" and day < date(2018, 12, 10)
            else "us_cash_close",
        }
        row.update(
            {
                field: quote[field][index]
                for field in ("open", "high", "low", "close", "volume")
            }
        )
        # Source OHLC is split-adjusted vendor history, not a contemporaneous
        # cash-price archive. Ratios and raw price proof are absent: no premium.
        row["adr_premium_close"] = None
        rows.append(row)
    return rows


def _positive(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def _coverage(frame: pl.DataFrame, key: str) -> list[dict]:
    if frame.is_empty():
        return []
    summary = (
        frame.group_by(key)
        .agg(
            pl.len().alias("observations"),
            pl.col("reference_date").min().alias("first_date"),
            pl.col("reference_date").max().alias("last_date"),
        )
        .sort(key)
    )
    return json.loads(summary.write_json())


def calendar_gaps(contracts: list[dict], calendar_text: str) -> tuple[dict, dict]:
    """Identify missing mainland futures sessions without equating 404 with holiday.

    The official SHFE daily-data UI takes the complement of trade-data.js to
    mark trading dates. Its historical calendar starts in 2015. For earlier
    history, an intervening absent weekday is unresolved and masks the return.
    Applying this schedule to DCE is conservative: an extra DCE closure masks
    a move instead of silently calling multiple observed sessions one session.
    """
    calendars = {
        int(year): {datetime.strptime(day, "%Y%m%d").date() for day in days.split(",")}
        for year, days in re.findall(r"(\d{4}):'([\d,]+)'", calendar_text)
        if 2013 <= int(year) <= DEVELOPMENT_END.year
    }
    days_by_product: dict[str, set[date]] = {}
    for row in contracts:
        days_by_product.setdefault(row["product"], set()).add(row["reference_date"])
    gaps = {}
    for product, observed in days_by_product.items():
        gaps[product] = {
            day
            for day in _weekdays(min(observed), max(observed))
            if day not in observed and day not in calendars.get(day.year, set())
        }
    return gaps, {
        "calendar_years": sorted(calendars),
        "missing_or_unverified_weekdays": {
            product: sorted(day.isoformat() for day in days)
            for product, days in gaps.items()
        },
        "pre_calendar_gap_policy": "mask: missing weekday is not assumed to be an exchange holiday",
    }


def build(root: Path, acquisition: Path) -> dict:
    """Normalize the bounded archives without touching raw/canonical inputs."""
    acquisition_hash = sha256_file(acquisition)
    acquired = json.loads(acquisition.read_text())
    observations, di_rows, contracts, us_rows = [], [], [], []
    source_audit = []
    migration_dates = []
    dce_sources = []
    for record in acquired["records"]:
        if record["status"] != "retrieved" or record["family"] == "documentation":
            source_audit.append(record)
            continue
        path = Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"acquired source hash changed: {path}")
        family = record["family"]
        if family == "dce":
            dce_sources.append((path, record["key"], record["sha256"], True))
            continue
        payload = path.read_bytes()
        if family == "di":
            day = date.fromisoformat(record["key"])
            try:
                rows = parse_di(payload, day)
            except ValueError as error:
                source_audit.append(
                    {**record, "status": "unusable_payload", "error": str(error)}
                )
                continue
            for row in rows:
                # This is a decision-equivalent after-close bound, not a
                # claimed timestamp of the historical curve's publication.
                available = datetime.combine(
                    day, time(23, 59, 59), SAO_PAULO
                ).astimezone(UTC)
                di_rows.append(
                    {**row, "available_at": available, "source_file": str(path)}
                )
                observations.append(
                    {
                        "series": f"br_di_{row['tenor']}",
                        "reference_date": day,
                        "available_at": available,
                        "value": row["rate_pct_252"] / 100,
                        "source_file": str(path),
                    }
                )
            source_audit.append(
                {
                    **record,
                    "parsed_rows": len(rows),
                    "status": "parsed" if rows else "no_published_curve",
                    "missing_tenors": sorted(
                        set(TENORS) - {row["tenor"] for row in rows}
                    ),
                }
            )
        elif family == "shfe":
            day = date.fromisoformat(record["key"])
            try:
                snapshot = parse_snapshot(payload, day)
            except ValueError as error:
                source_audit.append(
                    {**record, "status": "unusable_payload", "error": str(error)}
                )
                continue
            if snapshot.available_at.astimezone(SHANGHAI).date() != day:
                migration_dates.append(
                    {
                        "reference_date": day.isoformat(),
                        "archive_update_at": snapshot.available_at.isoformat(),
                    }
                )
            for row in snapshot.contracts:
                contracts.append(
                    {
                        "product": f"shfe_{row.product}",
                        "reference_date": day,
                        "available_at": datetime.combine(
                            day, time(15), SHANGHAI
                        ).astimezone(UTC),
                        "archive_update_at": snapshot.available_at,
                        "contract": row.contract,
                        "settlement": row.settlement,
                        "volume": row.volume,
                        "open_interest": row.open_interest,
                        "source_file": str(path),
                    }
                )
            source_audit.append(
                {**record, "status": "parsed", "parsed_rows": len(snapshot.contracts)}
            )
        elif family == "us":
            rows = parse_us(payload, record["key"], str(path))
            us_rows.extend(rows)
            source_audit.append(
                {**record, "status": "parsed", "parsed_rows": len(rows)}
            )
    dce_root = Path(acquired["dce_root"])
    dce_manifest_path = dce_root / "manifest.json"
    dce_manifest = json.loads(dce_manifest_path.read_text())
    for record in dce_manifest["files"]:
        if record["status"] != "downloaded":
            source_audit.append({"family": "dce", "key": record["symbol"], **record})
            continue
        symbol = record["symbol"]
        # I25MM denotes an expiry. Only its explicitly bounded <=2024 prefix
        # is consumed; the old manifest's full-file hash is not recomputed.
        complete = int(symbol[1:3]) <= 24
        path = dce_root / record["filename"]
        if complete and sha256_file(path) != record["sha256"]:
            raise ValueError(f"immutable DCE source changed: {path}")
        dce_sources.append((path, symbol, record["sha256"], complete))
    for path, symbol, source_hash, hash_verified in dce_sources:
        rows, audit = dce_prefix(path)
        for row in rows:
            day = date.fromisoformat(row["d"])
            settlement, volume, oi = float(row["s"]), float(row["v"]), float(row["p"])
            if (
                not all(math.isfinite(x) for x in (settlement, volume, oi))
                or min(volume, oi) < 0
            ):
                raise ValueError(f"invalid DCE contract values: {path}")
            if settlement <= 0:
                continue
            contracts.append(
                {
                    "product": "dce_iron",
                    "reference_date": day,
                    "available_at": datetime.combine(
                        day, time(15), SHANGHAI
                    ).astimezone(UTC),
                    "archive_update_at": None,
                    "contract": symbol,
                    "settlement": settlement,
                    "volume": volume,
                    "open_interest": oi,
                    "source_file": str(path),
                }
            )
        source_audit.append(
            {
                "family": "dce",
                "key": symbol,
                "path": str(path),
                "status": "parsed" if rows else "not_retained_by_mirror",
                "source_file_sha256_from_manifest": source_hash,
                "full_file_hash_verified": hash_verified,
                **audit,
            }
        )
    calendar_path = acquisition.parent / "documentation" / "shfe_trade_data.js"
    if calendar_path.exists():
        _get(calendar_path, SHFE_CALENDAR_URL)  # Verify the immutable cached response.
    calendar_text = calendar_path.read_text("utf8") if calendar_path.exists() else ""
    failed_days, futures_calendar_audit = calendar_gaps(contracts, calendar_text)
    futures_calendar_audit["source"] = {
        "path": str(calendar_path),
        "url": SHFE_CALENDAR_URL,
        "sha256": sha256_file(calendar_path) if calendar_path.exists() else None,
    }
    returns = roll_returns(contracts, failed_days)
    # Preserve US return units rather than exposing arbitrary vendor level
    # scaling as a model input. Only adjacent source rows with real volume are
    # usable; missing rows never become zero observations.
    us_returns = []
    previous = {}
    us_sessions = {
        day: index
        for index, day in enumerate(
            sorted(row["reference_date"] for row in us_rows if row["symbol"] == "EWZ")
        )
    }
    for row in sorted(us_rows, key=lambda r: (r["symbol"], r["reference_date"])):
        before = previous.get(row["symbol"])
        if (
            before is not None
            and row["reference_date"] in us_sessions
            and before["reference_date"] in us_sessions
            and us_sessions[row["reference_date"]]
            - us_sessions[before["reference_date"]]
            == 1
            and _positive(before["adjusted_close"])
            and _positive(row["adjusted_close"])
            and _positive(before["volume"])
            and _positive(row["volume"])
        ):
            us_returns.append(
                {
                    "series": f"us_{row['symbol']}",
                    "reference_date": row["reference_date"],
                    "previous_date": before["reference_date"],
                    "available_at": row["available_at"],
                    "log_return": math.log(
                        row["adjusted_close"] / before["adjusted_close"]
                    ),
                    "source_file": row["source_file"],
                }
            )
        previous[row["symbol"]] = row
    frames = {
        "cross_market_observations.parquet": pl.DataFrame(
            observations, schema=OBSERVATION_SCHEMA
        ),
        "di_curve.parquet": pl.DataFrame(di_rows),
        "futures_contracts.parquet": pl.DataFrame(contracts, infer_schema_length=None),
        "futures_returns.parquet": pl.DataFrame(returns),
        "us_daily.parquet": pl.DataFrame(us_rows, infer_schema_length=None),
        "us_returns.parquet": pl.DataFrame(us_returns),
    }
    paths = [root / name for name in frames]
    if any(path.exists() for path in paths) or (root / "manifest.json").exists():
        raise FileExistsError(
            "cross-market normalized output root already contains a build"
        )
    output = {}
    for name, frame in frames.items():
        path = root / name
        frame.write_parquet(path)
        output[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": frame.height,
            "bytes": path.stat().st_size,
        }
    result = {
        "schema": "BRAZIL_RV_ROUND5_CROSS_MARKET_V1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "normalized_source_archive_pending_joined_consumer_proof",
        "acquisition": {"path": str(acquisition), "sha256": acquisition_hash},
        "canonical_dce_manifest": {
            "path": str(dce_manifest_path),
            "sha256": sha256_file(dce_manifest_path),
        },
        "source_documentation": {
            "path": str(root / "documentation" / "evidence.json"),
            "sha256": sha256_file(root / "documentation" / "evidence.json")
            if (root / "documentation" / "evidence.json").exists()
            else None,
        },
        "outputs": output,
        "source_audit": source_audit,
        "coverage": {
            "di": _coverage(frames["di_curve.parquet"], "tenor"),
            "futures_contracts": _coverage(
                frames["futures_contracts.parquet"], "product"
            ),
            "futures_returns": _coverage(frames["futures_returns.parquet"], "series"),
            "us": _coverage(frames["us_daily.parquet"], "symbol"),
            "us_returns": _coverage(frames["us_returns.parquet"], "series"),
        },
        "di_availability": "Official PRE curve of the reference day's DI1 settlements; first subsequent B3 15:45 decision. available_at is a decision-equivalent end-of-day bound, not a measured publication timestamp.",
        "di_units": "common observation value is annual decimal; native rate_pct_252 is percent per annum, 252-business-day compounding; nominal fixed vertex plus actual calendar/working days retained",
        "asia_availability": "The day-session traded-contract settlement is the completed-day volume-weighted exchange measurement at 15:00 Asia/Shanghai; first B3 15:45 at or after that fixing. Previous-session OI chooses the contract. Nontraded selected contracts yield no return. No archive upload lag.",
        "shfe_archive_migration_metadata": migration_dates,
        "shfe_revision_share": None,
        "dce_revision_share": None,
        "futures_calendar_audit": futures_calendar_audit,
        "dce_limitation": "Unofficial Sina individual-expiry mirror. Earlier expired contracts may not be retained. No undocumented I0 series, future roll weights or back-adjustment. 2025 expiry files consumed only through <=2024 prefix; their full-file hash is inherited from the sealed manifest, not re-read.",
        "us_availability": "Observed US day regular close 16:00 America/New_York, scheduled early closes 13:00; converted through historical DST to first eligible B3 15:45, which can be same-day on an early close. SUZ before its 2018-12-10 NYSE listing retains unverified availability: the prior SUZBY OTC segment is not assigned an NYSE closing time.",
        "us_price_semantics": "Vendor OHLC is historically split adjusted; adjusted_close also includes distributions. Raw response retained, but neither field proves contemporaneous cash price. Log returns cancel later common multiplicative adjustment factors; vendor correction/rounding revision share is unknown.",
        "us_identity": "18 vendor ADR symbols plus EWZ, not a historical ADR-universe selection. EMBJ carries vendor history previously under ERJ; no current-ticker-to-historical-ISIN inference is admitted by this archive.",
        "adr_premium_status": "unavailable: historical dated ADR conversion ratios and contemporaneous unadjusted price proof absent; every premium remains null",
        "us_revision_share": None,
    }
    write_json_atomic(root / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("acquire", "build"), nargs="?", default="acquire"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shfe-root", type=Path)
    parser.add_argument("--dce-root", type=Path)
    parser.add_argument("--acquisition", type=Path)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("di", "shfe", "dce", "us"),
        default=["di", "shfe", "dce", "us"],
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "build":
        if args.acquisition is None:
            parser.error("build requires --acquisition")
        result = build(args.root, args.acquisition)
        print(json.dumps(result["coverage"]))
    else:
        if args.shfe_root is None or args.dce_root is None:
            parser.error("acquire requires --shfe-root and --dce-root")
        result = acquire(
            args.root, args.shfe_root, args.dce_root, args.families, args.workers
        )
        print(json.dumps({"sources": len(result["records"])}))


if __name__ == "__main__":
    main()
