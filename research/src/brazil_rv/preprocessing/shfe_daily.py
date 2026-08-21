from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from bisect import bisect_left, bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, time as wall_time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

CONTRACT_VERSION = "SHFE_DAILY_FERROUS_PULP_V1"
SOURCE_URL = "https://www.shfe.com.cn/data/tradedata/future/dailydata/kx{date}.dat"
SHANGHAI = ZoneInfo("Asia/Shanghai")
SAO_PAULO = ZoneInfo("America/Sao_Paulo")
FIRST_MODEL_DECISION = wall_time(10, 15)
PRODUCT_IDS = {"rb": "rb_f", "hc": "hc_f", "sp": "sp_f"}
RETURN_WINDOWS = (1, 5)
NORMALIZATION_WINDOW = 60
NORMALIZATION_MIN_OBSERVATIONS = 20
NORMALIZATION_CLIP = 5.0

RAW_FEATURES = (
    "rb_return_1d",
    "rb_return_5d",
    "rb_term_slope",
    "hc_return_1d",
    "hc_return_5d",
    "hc_term_slope",
    "sp_return_1d",
    "sp_return_5d",
    "sp_term_slope",
    "hc_minus_rb_log_ratio",
)
FEATURES = tuple(f"shfe_{name}_z" for name in RAW_FEATURES)
STEEL_FEATURES = tuple(
    f"shfe_{name}_z"
    for name in RAW_FEATURES
    if name.startswith(("rb_", "hc_", "hc_minus_rb_"))
)
PULP_FEATURES = tuple(
    f"shfe_{name}_z" for name in RAW_FEATURES if name.startswith("sp_")
)
EXPOSURE_TICKERS = {
    "steel": ("CSNA3", "GGBR4", "GOAU4", "USIM5"),
    "pulp": ("KLBN11", "SUZB3"),
}


@dataclass(frozen=True)
class ContractObservation:
    product: str
    contract: str
    delivery_year: int
    delivery_month: int
    settlement: float
    volume: float
    open_interest: float


@dataclass(frozen=True)
class Snapshot:
    trade_date: date
    available_at: datetime
    contracts: tuple[ContractObservation, ...]


@dataclass(frozen=True)
class SecurityExposure:
    security_id: str
    isin: str
    ticker: str
    group: str
    effective_from: date
    effective_to_inclusive: date


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _positive_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0.0 else None


def _nonnegative_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0.0 else None


def _delivery_month(value: object) -> tuple[int, int] | None:
    text = str(value).strip()
    if len(text) == 4 and text.isdigit():
        year = 2000 + int(text[:2])
        month = int(text[2:])
    elif len(text) == 6 and text.isdigit():
        year = int(text[:4])
        month = int(text[4:])
    else:
        return None
    return (year, month) if 1 <= month <= 12 else None


def parse_snapshot(payload: bytes, expected_date: date | None = None) -> Snapshot:
    try:
        document = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("SHFE daily file is not valid UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise ValueError("SHFE daily file has no JSON object root")
    try:
        trade_date = datetime.strptime(str(document["report_date"]), "%Y%m%d").date()
        published_local = datetime.strptime(
            str(document["update_date"]), "%Y%m%d %H:%M:%S"
        ).replace(tzinfo=SHANGHAI)
    except (KeyError, ValueError) as error:
        raise ValueError(
            "SHFE daily file has invalid report_date/update_date"
        ) from error
    if expected_date is not None and trade_date != expected_date:
        raise ValueError(
            f"SHFE report date mismatch: expected={expected_date}, actual={trade_date}"
        )
    raw_rows = document.get("o_curinstrument")
    if not isinstance(raw_rows, list):
        raise ValueError("SHFE daily file has no o_curinstrument rows")

    product_by_id = {source: product for product, source in PRODUCT_IDS.items()}
    contracts: list[ContractObservation] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        product = product_by_id.get(str(raw.get("PRODUCTID", "")).strip())
        delivery = _delivery_month(raw.get("DELIVERYMONTH"))
        settlement = _positive_float(raw.get("SETTLEMENTPRICE"))
        volume = _nonnegative_float(raw.get("VOLUME"))
        open_interest = _nonnegative_float(raw.get("OPENINTEREST"))
        if (
            product is None
            or delivery is None
            or settlement is None
            or volume is None
            or open_interest is None
        ):
            continue
        contract = f"{product}{delivery[0] % 100:02d}{delivery[1]:02d}"
        key = (product, contract)
        if key in seen:
            raise ValueError(f"Duplicate SHFE contract row: {key}")
        seen.add(key)
        contracts.append(
            ContractObservation(
                product=product,
                contract=contract,
                delivery_year=delivery[0],
                delivery_month=delivery[1],
                settlement=settlement,
                volume=volume,
                open_interest=open_interest,
            )
        )
    if not contracts:
        raise ValueError("SHFE daily file has no usable rb/hc/sp contracts")
    return Snapshot(
        trade_date=trade_date,
        available_at=published_local.astimezone(timezone.utc),
        contracts=tuple(sorted(contracts, key=lambda row: (row.product, row.contract))),
    )


def parse_snapshot_file(path: Path) -> Snapshot:
    token = path.stem.removeprefix("kx")
    expected = datetime.strptime(token, "%Y%m%d").date()
    return parse_snapshot(path.read_bytes(), expected)


def _download_one(trade_date: date, output_dir: Path, timeout: float) -> str:
    destination = output_dir / f"year={trade_date.year}" / f"kx{trade_date:%Y%m%d}.dat"
    if destination.is_file():
        parse_snapshot_file(destination)
        return "existing"
    request = urllib.request.Request(
        SOURCE_URL.format(date=f"{trade_date:%Y%m%d}"),
        headers={"Accept": "application/json", "User-Agent": "Brazil-RV-research/1"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            parse_snapshot(payload, trade_date)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as target:
                target.write(payload)
            return "downloaded"
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return "not_published"
            if error.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise
        except (TimeoutError, urllib.error.URLError):
            if attempt == 2:
                raise
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def acquire_snapshots(
    start: date,
    end: date,
    output_dir: Path,
    *,
    workers: int = 4,
    timeout: float = 30.0,
) -> dict[str, object]:
    """Acquire official SHFE daily files without modifying an existing raw file."""
    if end < start:
        raise ValueError("end must be on or after start")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("requested_start") != start.isoformat()
            or manifest.get("requested_end") != end.isoformat()
        ):
            raise ValueError("Existing SHFE raw manifest covers a different interval")
        validate_raw_archive(output_dir)
        return manifest

    requested: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            requested.append(current)
        current += timedelta(days=1)
    outcomes: dict[date, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_one, trade_date, output_dir, timeout): trade_date
            for trade_date in requested
        }
        for future in as_completed(futures):
            outcomes[futures[future]] = future.result()

    files = sorted(output_dir.glob("year=*/kx*.dat"))
    source_files: list[dict[str, object]] = []
    product_session_counts = {product: 0 for product in PRODUCT_IDS}
    for path in files:
        snapshot = parse_snapshot_file(path)
        products = sorted({row.product for row in snapshot.contracts})
        for product in products:
            product_session_counts[product] += 1
        source_files.append(
            {
                "relative_path": path.relative_to(output_dir).as_posix(),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "report_date": snapshot.trade_date.isoformat(),
                "available_at": snapshot.available_at.isoformat(),
                "products": products,
            }
        )
    if not source_files:
        raise ValueError("No SHFE daily files were acquired")
    manifest: dict[str, object] = {
        "contract_version": CONTRACT_VERSION,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url_template": SOURCE_URL,
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "requested_weekday_count": len(requested),
        "downloaded_count": sum(value == "downloaded" for value in outcomes.values()),
        "reused_count": sum(value == "existing" for value in outcomes.values()),
        "not_published_dates": [
            value.isoformat()
            for value in requested
            if outcomes[value] == "not_published"
        ],
        "source_file_count": len(source_files),
        "first_report_date": source_files[0]["report_date"],
        "last_report_date": source_files[-1]["report_date"],
        "product_session_counts": product_session_counts,
        "source_files": source_files,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_raw_archive(raw_dir: Path) -> dict[str, object]:
    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"SHFE raw archive has no manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = manifest.get("source_files")
    if not isinstance(sources, list) or not sources:
        raise ValueError("SHFE raw manifest has no source files")
    listed: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("SHFE raw manifest source entry is malformed")
        relative = Path(str(source.get("relative_path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("SHFE raw manifest contains an unsafe relative path")
        relative_text = relative.as_posix()
        if relative_text in listed:
            raise ValueError("SHFE raw manifest contains a duplicate source path")
        listed.add(relative_text)
        path = raw_dir / relative
        if not path.is_file() or _sha256(path) != source.get("sha256"):
            raise ValueError(f"Immutable SHFE raw source changed: {path}")
    actual = {
        path.relative_to(raw_dir).as_posix() for path in raw_dir.glob("year=*/kx*.dat")
    }
    if actual != listed:
        raise ValueError("SHFE raw files differ from the immutable manifest inventory")
    return manifest


def load_snapshots(raw_dir: Path) -> list[Snapshot]:
    snapshots = [
        parse_snapshot_file(path) for path in sorted(raw_dir.glob("year=*/kx*.dat"))
    ]
    if not snapshots:
        raise FileNotFoundError(f"No SHFE kx*.dat files under {raw_dir}")
    if len({snapshot.trade_date for snapshot in snapshots}) != len(snapshots):
        raise ValueError("SHFE raw snapshot dates must be unique")
    return sorted(snapshots, key=lambda snapshot: snapshot.trade_date)


def _contracts(snapshot: Snapshot, product: str) -> dict[str, ContractObservation]:
    return {row.contract: row for row in snapshot.contracts if row.product == product}


def select_contract(prior: Snapshot, product: str) -> str | None:
    candidates = [
        row
        for row in prior.contracts
        if row.product == product and row.open_interest > 0.0
    ]
    if not candidates:
        return None
    # Earlier maturity is the deterministic tie-break; current-day data never enters.
    return min(
        candidates,
        key=lambda row: (-row.open_interest, row.delivery_year, row.delivery_month),
    ).contract


def _same_contract_return(
    current: Snapshot,
    past: Snapshot,
    product: str,
    contract: str | None,
) -> tuple[float, bool]:
    if contract is None:
        return 0.0, False
    current_row = _contracts(current, product).get(contract)
    past_row = _contracts(past, product).get(contract)
    if current_row is None or past_row is None:
        return 0.0, False
    return math.log(current_row.settlement / past_row.settlement), True


def _selected_log_ratio(
    current: Snapshot,
    numerator_product: str,
    numerator_contract: str | None,
    denominator_product: str,
    denominator_contract: str | None,
) -> tuple[float, bool]:
    if numerator_contract is None or denominator_contract is None:
        return 0.0, False
    numerator = _contracts(current, numerator_product).get(numerator_contract)
    denominator = _contracts(current, denominator_product).get(denominator_contract)
    if numerator is None or denominator is None:
        return 0.0, False
    return math.log(numerator.settlement / denominator.settlement), True


def _term_slope(snapshot: Snapshot, product: str) -> tuple[float, bool]:
    current_month = snapshot.trade_date.year * 12 + snapshot.trade_date.month
    points = []
    for row in snapshot.contracts:
        delivery = row.delivery_year * 12 + row.delivery_month
        if (
            row.product == product
            and row.volume > 0.0
            and row.open_interest > 0.0
            and delivery >= current_month
        ):
            points.append((float(delivery - current_month), math.log(row.settlement)))
    if len(points) < 3:
        return 0.0, False
    mean_x = statistics.fmean(point[0] for point in points)
    mean_y = statistics.fmean(point[1] for point in points)
    denominator = sum((point[0] - mean_x) ** 2 for point in points)
    if denominator <= 0.0:
        return 0.0, False
    monthly = (
        sum((point[0] - mean_x) * (point[1] - mean_y) for point in points) / denominator
    )
    return 12.0 * monthly, True


def build_market_rows(snapshots: list[Snapshot]) -> list[dict[str, object]]:
    if len(snapshots) < 6:
        raise ValueError("At least six SHFE sessions are required")
    rows: list[dict[str, object]] = []
    for index, current in enumerate(snapshots):
        row: dict[str, object] = {
            "source_trade_date": current.trade_date,
            "available_at": current.available_at,
        }
        selected: dict[str, str | None] = {}
        for product in PRODUCT_IDS:
            selected[product] = (
                None if index == 0 else select_contract(snapshots[index - 1], product)
            )
            row[f"selected_{product}_contract"] = selected[product] or ""
            for window in RETURN_WINDOWS:
                name = f"{product}_return_{window}d"
                if index < window:
                    value, mask = 0.0, False
                else:
                    value, mask = _same_contract_return(
                        current,
                        snapshots[index - window],
                        product,
                        selected[product],
                    )
                row[name] = value
                row[f"{name}_mask"] = mask
            slope, slope_mask = _term_slope(current, product)
            row[f"{product}_term_slope"] = slope
            row[f"{product}_term_slope_mask"] = slope_mask
        spread, spread_mask = _selected_log_ratio(
            current,
            "hc",
            selected["hc"],
            "rb",
            selected["rb"],
        )
        row["hc_minus_rb_log_ratio"] = spread
        row["hc_minus_rb_log_ratio_mask"] = spread_mask
        rows.append(row)
    return rows


def normalize_market_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    histories: dict[str, list[float]] = {feature: [] for feature in RAW_FEATURES}
    normalized: list[dict[str, object]] = []
    for source in rows:
        row = dict(source)
        for raw_feature, feature in zip(RAW_FEATURES, FEATURES, strict=True):
            valid = bool(source[f"{raw_feature}_mask"])
            history = histories[raw_feature]
            prior = history[-NORMALIZATION_WINDOW:]
            value = float(source[raw_feature])
            output = 0.0
            output_valid = False
            if valid and len(prior) >= NORMALIZATION_MIN_OBSERVATIONS:
                center = statistics.median(prior)
                mad = statistics.median(abs(item - center) for item in prior)
                scale = 1.4826 * mad
                if scale > 1e-12:
                    output = min(
                        max((value - center) / scale, -NORMALIZATION_CLIP),
                        NORMALIZATION_CLIP,
                    )
                    output_valid = True
            row[feature] = output
            row[f"{feature}_mask"] = output_valid
            if valid:
                history.append(value)
        normalized.append(row)
    return normalized


def load_b3_sessions(calendar_dir: Path) -> list[date]:
    paths = sorted(calendar_dir.glob("year=*/equities_daily_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parsed B3 daily files under {calendar_dir}")
    return (
        pl.scan_parquet(paths)
        .select(pl.col("trade_date").unique().sort())
        .collect()
        .get_column("trade_date")
        .to_list()
    )


def availability_date(available_at: datetime, b3_sessions: list[date]) -> date:
    if available_at.tzinfo is None:
        raise ValueError("available_at must be timezone-aware")
    local = available_at.astimezone(SAO_PAULO)
    same_or_next = bisect_left(b3_sessions, local.date())
    if (
        same_or_next < len(b3_sessions)
        and b3_sessions[same_or_next] == local.date()
        and local.time().replace(tzinfo=None) <= FIRST_MODEL_DECISION
    ):
        return local.date()
    next_session = bisect_right(b3_sessions, local.date())
    if next_session == len(b3_sessions):
        raise ValueError(f"No B3 session follows SHFE availability {available_at}")
    return b3_sessions[next_session]


def resolve_exposures(assignments_path: Path) -> list[SecurityExposure]:
    assignments = pl.read_parquet(assignments_path).select(
        "security_id",
        "isin",
        "latest_ticker",
        "first_overlap_date",
        "last_overlap_date",
    )
    exposures: list[SecurityExposure] = []
    for group, tickers in EXPOSURE_TICKERS.items():
        for ticker in tickers:
            matched = assignments.filter(pl.col("latest_ticker") == ticker)
            if matched.height != 1:
                raise ValueError(
                    f"Expected one accepted identity for {ticker}, found {matched.height}"
                )
            row = matched.row(0, named=True)
            if row["security_id"] != f"ISIN:{row['isin']}":
                raise ValueError(f"Accepted identity is not exact ISIN for {ticker}")
            exposures.append(
                SecurityExposure(
                    security_id=row["security_id"],
                    isin=row["isin"],
                    ticker=ticker,
                    group=group,
                    effective_from=date.fromisoformat(row["first_overlap_date"]),
                    effective_to_inclusive=date.fromisoformat(row["last_overlap_date"]),
                )
            )
    if len({row.security_id for row in exposures}) != len(exposures):
        raise ValueError("SHFE exposure identities must be unique")
    return sorted(exposures, key=lambda row: row.security_id)


def _security_frame(
    market_rows: list[dict[str, object]],
    exposures: list[SecurityExposure],
    b3_sessions: list[date],
) -> pl.DataFrame:
    output: list[dict[str, object]] = []
    for market in market_rows:
        available_date = availability_date(
            market["available_at"],  # type: ignore[arg-type]
            b3_sessions,
        )
        for exposure in exposures:
            if not (
                exposure.effective_from
                <= available_date
                <= exposure.effective_to_inclusive
            ):
                continue
            enabled = STEEL_FEATURES if exposure.group == "steel" else PULP_FEATURES
            row: dict[str, object] = {
                "source_trade_date": market["source_trade_date"],
                "available_at": market["available_at"],
                "available_date": available_date,
                "security_id": exposure.security_id,
                "identity_ticker": exposure.ticker,
                "exposure_group": exposure.group,
            }
            for feature in FEATURES:
                mask = feature in enabled and bool(market[f"{feature}_mask"])
                row[feature] = float(market[feature]) if mask else 0.0
                row[f"{feature}_mask"] = mask
            output.append(row)
    if not output:
        raise ValueError("SHFE normalized security frame has no rows")
    frame = pl.DataFrame(output, infer_schema_length=None).with_columns(
        pl.col("source_trade_date").cast(pl.Date),
        pl.col("available_date").cast(pl.Date),
        pl.col("available_at").dt.convert_time_zone("UTC"),
        *[pl.col(feature).cast(pl.Float32) for feature in FEATURES],
        *[pl.col(f"{feature}_mask").cast(pl.Boolean) for feature in FEATURES],
    )
    # Several Chinese sessions can become known before one post-holiday B3 open.
    # The latest exact publication is the state available to that B3 session.
    return (
        frame.sort("available_at")
        .group_by("available_date", "security_id", maintain_order=True)
        .last()
        .sort("available_date", "security_id")
    )


def _market_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col("source_trade_date").cast(pl.Date),
        pl.col("available_at").dt.convert_time_zone("UTC"),
        *[pl.col(feature).cast(pl.Float32) for feature in FEATURES],
        *[pl.col(f"{feature}_mask").cast(pl.Boolean) for feature in FEATURES],
    )


def build_sidecar(
    raw_dir: Path,
    calendar_dir: Path,
    assignments_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    validate_raw_archive(raw_dir)
    snapshots = load_snapshots(raw_dir)
    market_rows = normalize_market_rows(build_market_rows(snapshots))
    b3_sessions = load_b3_sessions(calendar_dir)
    exposures = resolve_exposures(assignments_path)
    market_frame = _market_frame(market_rows)
    security_frame = _security_frame(market_rows, exposures, b3_sessions)

    output_dir.mkdir(parents=True, exist_ok=False)
    market_path = output_dir / "shfe_market_daily.parquet"
    security_path = output_dir / "shfe_daily_features.parquet"
    market_frame.write_parquet(market_path, compression="zstd", statistics=True)
    security_frame.write_parquet(security_path, compression="zstd", statistics=True)
    raw_manifest = raw_dir / "manifest.json"
    manifest: dict[str, object] = {
        "contract_version": CONTRACT_VERSION,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_source_dir": str(raw_dir.resolve()),
        "raw_manifest_sha256": _sha256(raw_manifest)
        if raw_manifest.is_file()
        else None,
        "raw_snapshot_count": len(snapshots),
        "first_source_trade_date": snapshots[0].trade_date.isoformat(),
        "last_source_trade_date": snapshots[-1].trade_date.isoformat(),
        "calendar_source": str(calendar_dir.resolve()),
        "assignments_source": str(assignments_path.resolve()),
        "assignments_sha256": _sha256(assignments_path),
        "availability_rule": (
            "Parse each SHFE update_date in Asia/Shanghai; retain exact UTC "
            "available_at; use the same B3 date only when published by the first "
            "10:15 America/Sao_Paulo model decision, otherwise the next B3 session"
        ),
        "roll_rule": (
            "For SHFE session t, choose maximum-open-interest contract from session "
            "t-1 only; tie-break by earliest delivery; never fall back using t data"
        ),
        "return_rule": (
            "Log settlement return from t to t-1/t-5 for the identical contract "
            "selected from t-1 open interest; contract-roll jumps are never spliced"
        ),
        "cross_product_spread_rule": (
            "Log HRC-to-rebar settlement-price ratio using each product's independently "
            "selected prior-open-interest contract; dimensionless HRC relative premium"
        ),
        "term_slope_rule": (
            "Annualized OLS slope of log settlement versus delivery-month distance "
            "using current positive-volume, positive-open-interest contracts; >=3 points"
        ),
        "normalization_rule": (
            f"Prior-only trailing-{NORMALIZATION_WINDOW} median/MAD z-score; minimum "
            f"{NORMALIZATION_MIN_OBSERVATIONS} valid observations; clip "
            f"[-{NORMALIZATION_CLIP}, {NORMALIZATION_CLIP}]; append current after transform"
        ),
        "exposure_rule": (
            "Researcher-declared steel/pulp groups resolved one-to-one through the "
            "accepted assignment table to permanent ISIN security_id and bounded by "
            "that identity's accepted first/last overlap dates"
        ),
        "exposures": [
            {
                **asdict(exposure),
                "effective_from": exposure.effective_from.isoformat(),
                "effective_to_inclusive": exposure.effective_to_inclusive.isoformat(),
            }
            for exposure in exposures
        ],
        "features": list(FEATURES),
        "steel_features": list(STEEL_FEATURES),
        "pulp_features": list(PULP_FEATURES),
        "feature_valid_rows": {
            feature: int(security_frame.get_column(f"{feature}_mask").sum())
            for feature in FEATURES
        },
        "market_row_count": market_frame.height,
        "security_row_count": security_frame.height,
        "first_available_date": security_frame.get_column("available_date")
        .min()
        .isoformat(),
        "last_available_date": security_frame.get_column("available_date")
        .max()
        .isoformat(),
        "market_output_file": market_path.name,
        "market_output_sha256": _sha256(market_path),
        "security_output_file": security_path.name,
        "security_output_sha256": _sha256(security_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _iso_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Acquire/build SHFE daily sidecar data"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    acquire = subparsers.add_parser("acquire")
    acquire.add_argument("--start", type=_iso_date, required=True)
    acquire.add_argument("--end", type=_iso_date, required=True)
    acquire.add_argument("--out", type=Path, required=True)
    acquire.add_argument("--workers", type=int, default=4)
    build = subparsers.add_parser("build")
    build.add_argument("--raw-dir", type=Path, required=True)
    build.add_argument("--calendar-dir", type=Path, required=True)
    build.add_argument("--assignments", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "acquire":
        manifest = acquire_snapshots(
            args.start, args.end, args.out, workers=args.workers
        )
        print(
            f"Acquired {manifest['source_file_count']} official SHFE sessions in {args.out}",
            flush=True,
        )
    else:
        manifest = build_sidecar(
            args.raw_dir, args.calendar_dir, args.assignments, args.out
        )
        print(
            f"Wrote {manifest['security_row_count']} normalized rows to {args.out}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
