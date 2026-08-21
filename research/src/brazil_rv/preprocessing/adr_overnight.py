from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
import urllib.parse
import urllib.request
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

RAW_CONTRACT_VERSION = "YAHOO_CHART_ADR_OVERNIGHT_RAW_V1"
CONTRACT_VERSION = "ADR_OVERNIGHT_PIT_V1"
PERIOD_START = date(2021, 4, 1)
PERIOD_END_EXCLUSIVE = date(2024, 7, 3)
NEW_YORK = ZoneInfo("America/New_York")
SAO_PAULO = ZoneInfo("America/Sao_Paulo")
B3_DECISION_TIME = time(10, 15)
US_CLOSE_TIME = time(16, 0)


@dataclass(frozen=True)
class Pair:
    adr_symbol: str
    local_ticker: str


PAIRS = (
    Pair("ABEV", "ABEV3"),
    Pair("BBD", "BBDC4"),
    Pair("BBDO", "BBDC3"),
    Pair("BAK", "BRKM5"),
    Pair("BSBR", "SANB11"),
    Pair("CIG", "CMIG4"),
    Pair("SID", "CSNA3"),
    Pair("EMBJ", "EMBR3"),
    Pair("GGB", "GGBR4"),
    Pair("ITUB", "ITUB4"),
    Pair("PBR", "PETR3"),
    Pair("PBR-A", "PETR4"),
    Pair("SBS", "SBSP3"),
    Pair("SUZ", "SUZB3"),
    Pair("TIMB", "TIMS3"),
    Pair("UGP", "UGPA3"),
    Pair("VALE", "VALE3"),
    Pair("VIV", "VIVT3"),
)
EXCLUDED_PAIRS = {
    "CBDY": {
        "local_ticker": "PCAR3",
        "reason": (
            "Yahoo's retroactive CBDY history is an illiquid OTC series with repeated "
            "zero-volume sessions and unadjusted-looking one-day discontinuities above "
            "100%; it is not a reliable historical PCAR3 ADR return series."
        ),
    }
}
FEATURES = (
    "adr_return_1d",
    "adr_return_5d",
    "adr_minus_ewz_1d",
    "adr_residual_robust_surprise",
)
ALL_SYMBOLS = ("EWZ", *(pair.adr_symbol for pair in PAIRS), *EXCLUDED_PAIRS)


@dataclass(frozen=True)
class Bar:
    session_date: date
    adjusted_close: float
    close: float
    volume: int


@dataclass(frozen=True)
class SeriesFeatures:
    return_1d: dict[date, float]
    return_5d: dict[date, float]
    residual: dict[date, float]
    surprise: dict[date, float]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _epoch(value: date) -> int:
    return int(datetime.combine(value, time(), timezone.utc).timestamp())


def _safe_name(symbol: str) -> str:
    return symbol.replace("-", "_") + ".json"


def _chart_url(symbol: str) -> str:
    query = urllib.parse.urlencode(
        {
            "period1": _epoch(PERIOD_START),
            "period2": _epoch(PERIOD_END_EXCLUSIVE),
            "interval": "1d",
            "events": "div,splits",
            "includeAdjustedClose": "true",
        }
    )
    encoded = urllib.parse.quote(symbol, safe="")
    return f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}?{query}"


def _chart_result(payload: bytes, expected_symbol: str) -> dict:
    parsed = json.loads(payload)
    chart = parsed.get("chart", {})
    if chart.get("error") is not None or not chart.get("result"):
        raise ValueError(
            f"Yahoo chart failed for {expected_symbol}: {chart.get('error')}"
        )
    result = chart["result"][0]
    meta = result.get("meta", {})
    if meta.get("symbol") != expected_symbol:
        raise ValueError(
            f"Yahoo returned symbol {meta.get('symbol')!r} for {expected_symbol}"
        )
    if meta.get("currency") != "USD":
        raise ValueError(f"Yahoo {expected_symbol} history is not USD-denominated")
    return result


def acquire_snapshot(output_dir: Path) -> dict[str, object]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    sources: list[dict[str, object]] = []
    try:
        for symbol in ALL_SYMBOLS:
            url = _chart_url(symbol)
            request = urllib.request.Request(
                url, headers={"User-Agent": "Brazil-RV historical research/1.0"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            result = _chart_result(payload, symbol)
            path = temporary / _safe_name(symbol)
            path.write_bytes(payload)
            timestamps = result.get("timestamp") or []
            sources.append(
                {
                    "symbol": symbol,
                    "url": url,
                    "file": path.name,
                    "sha256": _sha256(path),
                    "byte_count": len(payload),
                    "timestamp_count": len(timestamps),
                    "exchange": result["meta"].get("exchangeName"),
                    "currency": result["meta"].get("currency"),
                }
            )
        manifest: dict[str, object] = {
            "contract_version": RAW_CONTRACT_VERSION,
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "provider": "Yahoo Finance query1 chart API",
            "period_start": PERIOD_START.isoformat(),
            "period_end_exclusive": PERIOD_END_EXCLUSIVE.isoformat(),
            "interval": "1d",
            "events": "div,splits",
            "include_adjusted_close": True,
            "sources": sources,
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


def _load_bars(path: Path, expected_symbol: str) -> tuple[list[Bar], dict]:
    result = _chart_result(path.read_bytes(), expected_symbol)
    timestamps = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    adjusted = (result.get("indicators", {}).get("adjclose") or [{}])[0].get(
        "adjclose"
    ) or []
    closes = quote.get("close") or []
    volumes = quote.get("volume") or []
    if not (len(timestamps) == len(adjusted) == len(closes) == len(volumes)):
        raise ValueError(
            f"Yahoo arrays have inconsistent lengths for {expected_symbol}"
        )
    bars: list[Bar] = []
    for timestamp, adjusted_close, close, volume in zip(
        timestamps, adjusted, closes, volumes
    ):
        if adjusted_close is None or close is None or volume is None:
            continue
        adjusted_close = float(adjusted_close)
        close = float(close)
        if (
            not math.isfinite(adjusted_close + close)
            or adjusted_close <= 0
            or close <= 0
        ):
            continue
        session_date = (
            datetime.fromtimestamp(timestamp, timezone.utc).astimezone(NEW_YORK).date()
        )
        bars.append(Bar(session_date, adjusted_close, close, int(volume)))
    if not bars:
        raise ValueError(f"Yahoo history contains no usable bars for {expected_symbol}")
    dates = [bar.session_date for bar in bars]
    if dates != sorted(set(dates)):
        raise ValueError(
            f"Yahoo session dates are duplicate or unsorted for {expected_symbol}"
        )
    return bars, result.get("events") or {}


def _load_snapshot(raw_dir: Path) -> tuple[dict[str, list[Bar]], dict[str, dict]]:
    manifest_path = raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("contract_version") != RAW_CONTRACT_VERSION
        or manifest.get("status") != "complete"
    ):
        raise ValueError("Yahoo raw snapshot is not complete")
    entries = {entry["symbol"]: entry for entry in manifest["sources"]}
    if set(entries) != set(ALL_SYMBOLS):
        raise ValueError("Yahoo raw snapshot symbol set differs from the ADR contract")
    bars: dict[str, list[Bar]] = {}
    events: dict[str, dict] = {}
    for symbol in ALL_SYMBOLS:
        entry = entries[symbol]
        path = raw_dir / entry["file"]
        if _sha256(path) != entry["sha256"]:
            raise ValueError(f"Yahoo raw response hash changed: {path}")
        bars[symbol], events[symbol] = _load_bars(path, symbol)
    return bars, events


def _log_returns(bars: Sequence[Bar], lag: int) -> dict[date, float]:
    output: dict[date, float] = {}
    for index in range(lag, len(bars)):
        current = bars[index]
        previous = bars[index - lag]
        max_days = 7 if lag == 1 else 14
        if (
            current.volume <= 0
            or previous.volume <= 0
            or (current.session_date - previous.session_date).days > max_days
        ):
            continue
        output[current.session_date] = math.log(
            current.adjusted_close / previous.adjusted_close
        )
    return output


def _compress(value: float, scale: float) -> float:
    return math.tanh(value / scale)


def _feature_series(adr: Sequence[Bar], ewz: Sequence[Bar]) -> SeriesFeatures:
    adr_1d = _log_returns(adr, 1)
    adr_5d = _log_returns(adr, 5)
    ewz_1d = _log_returns(ewz, 1)
    residual = {
        current_date: value - ewz_1d[current_date]
        for current_date, value in adr_1d.items()
        if current_date in ewz_1d
    }
    surprise: dict[date, float] = {}
    history: list[float] = []
    for current_date, value in sorted(residual.items()):
        prior = history[-60:]
        if len(prior) >= 20:
            center = statistics.median(prior)
            mad = statistics.median(abs(item - center) for item in prior)
            denominator = max(1.4826 * mad, 0.005)
            surprise[current_date] = (
                max(min((value - center) / denominator, 6.0), -6.0) / 6.0
            )
        history.append(value)
    return SeriesFeatures(
        return_1d={key: _compress(value, 0.05) for key, value in adr_1d.items()},
        return_5d={key: _compress(value, 0.10) for key, value in adr_5d.items()},
        residual={key: _compress(value, 0.03) for key, value in residual.items()},
        surprise=surprise,
    )


def _last_completed_us_session(
    b3_date: date, us_sessions: Sequence[date]
) -> date | None:
    decision = datetime.combine(b3_date, B3_DECISION_TIME, SAO_PAULO)
    index = bisect_right(us_sessions, b3_date) - 1
    while index >= 0:
        session = us_sessions[index]
        close = datetime.combine(session, US_CLOSE_TIME, NEW_YORK)
        if close < decision:
            return session
        index -= 1
    return None


def _cotahist_files(cotahist_dir: Path, through: date) -> list[Path]:
    paths = [
        path
        for path in sorted(cotahist_dir.glob("year=*/equities_daily_*.parquet"))
        if int(path.parent.name.removeprefix("year=")) <= through.year
    ]
    if not paths:
        raise FileNotFoundError(f"No parsed COTAHIST files under {cotahist_dir}")
    return paths


def _resolve_pair_identities(
    cotahist: pl.DataFrame, accepted_ids: set[str], start: date, end: date
) -> dict[str, str]:
    output: dict[str, str] = {}
    for pair in PAIRS:
        frame = cotahist.filter(
            pl.col("ticker") == pair.local_ticker,
            pl.col("trade_date").is_between(start, end),
        )
        identities = sorted(set(frame.get_column("security_id").to_list()))
        if len(identities) != 1 or identities[0] not in accepted_ids:
            raise ValueError(
                f"{pair.local_ticker} does not map to one accepted permanent identity: "
                f"{identities}"
            )
        output[pair.local_ticker] = identities[0]
    return output


def _split_audit(symbol: str, bars: Sequence[Bar], events: dict) -> dict[str, object]:
    adjusted_returns = _log_returns(bars, 1)
    max_return = max(abs(value) for value in adjusted_returns.values())
    split_rows = []
    for event in events.get("splits", {}).values():
        split_date = (
            datetime.fromtimestamp(event["date"], timezone.utc)
            .astimezone(NEW_YORK)
            .date()
        )
        adjusted_return = adjusted_returns.get(split_date)
        if adjusted_return is not None and abs(adjusted_return) > 0.35:
            raise ValueError(
                f"{symbol} adjusted close failed split audit on {split_date}"
            )
        split_rows.append(
            {
                "date": split_date.isoformat(),
                "ratio": event.get("splitRatio"),
                "adjusted_log_return": adjusted_return,
            }
        )
    if max_return > 0.35:
        raise ValueError(f"{symbol} has an implausible adjusted one-day return")
    return {
        "bar_count": len(bars),
        "first_session": bars[0].session_date.isoformat(),
        "last_session": bars[-1].session_date.isoformat(),
        "zero_volume_count": sum(bar.volume <= 0 for bar in bars),
        "max_abs_adjusted_log_return": max_return,
        "splits": split_rows,
    }


def build_daily_frame(
    bars: dict[str, list[Bar]],
    cotahist: pl.DataFrame,
    accepted_ids: set[str],
    *,
    available_start: date,
    available_end: date,
) -> tuple[pl.DataFrame, dict[str, object]]:
    if available_start > available_end:
        raise ValueError("available_start cannot be after available_end")
    identities = _resolve_pair_identities(
        cotahist, accepted_ids, available_start, available_end
    )
    b3_dates = (
        cotahist.filter(pl.col("trade_date").is_between(available_start, available_end))
        .get_column("trade_date")
        .unique()
        .sort()
        .to_list()
    )
    ewz_dates = [bar.session_date for bar in bars["EWZ"]]
    source_by_b3 = {
        current_date: _last_completed_us_session(current_date, ewz_dates)
        for current_date in b3_dates
    }
    features = {
        pair.adr_symbol: _feature_series(bars[pair.adr_symbol], bars["EWZ"])
        for pair in PAIRS
    }
    active_identity = {
        (row["trade_date"], row["ticker"]): row["security_id"]
        for row in cotahist.filter(
            pl.col("ticker").is_in([pair.local_ticker for pair in PAIRS]),
            pl.col("trade_date").is_between(available_start, available_end),
        )
        .select("trade_date", "ticker", "security_id")
        .iter_rows(named=True)
    }
    rows: list[dict[str, object]] = []
    missing_active_identity = 0
    for current_date in b3_dates:
        source_date = source_by_b3[current_date]
        if source_date is None:
            continue
        for pair in PAIRS:
            security_id = active_identity.get((current_date, pair.local_ticker))
            if security_id is None:
                missing_active_identity += 1
                continue
            if security_id != identities[pair.local_ticker]:
                raise ValueError(
                    f"{pair.local_ticker} identity changed unexpectedly on {current_date}"
                )
            series = features[pair.adr_symbol]
            values = {
                "adr_return_1d": series.return_1d.get(source_date),
                "adr_return_5d": series.return_5d.get(source_date),
                "adr_minus_ewz_1d": series.residual.get(source_date),
                "adr_residual_robust_surprise": series.surprise.get(source_date),
            }
            row: dict[str, object] = {
                "available_date": current_date,
                "source_session_date": source_date,
                "security_id": security_id,
            }
            for feature, value in values.items():
                valid = value is not None and math.isfinite(value)
                row[feature] = float(value) if valid else 0.0
                row[f"{feature}_mask"] = valid
            rows.append(row)
    if not rows:
        raise ValueError("No ADR rows map to active permanent identities")
    frame = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col("available_date", "source_session_date").cast(pl.Date),
        *[pl.col(feature).cast(pl.Float32) for feature in FEATURES],
        *[pl.col(f"{feature}_mask").cast(pl.Boolean) for feature in FEATURES],
    )
    frame = frame.select(
        "available_date",
        "source_session_date",
        "security_id",
        *[column for feature in FEATURES for column in (feature, f"{feature}_mask")],
    ).sort("available_date", "security_id")
    duplicate = (
        frame.group_by("available_date", "security_id").len().filter(pl.col("len") > 1)
    )
    if duplicate.height:
        raise ValueError("ADR daily frame contains duplicate date/security keys")
    invalid_nonzero = sum(
        int(((~frame[f"{feature}_mask"]) & (frame[feature] != 0)).sum())
        for feature in FEATURES
    )
    if invalid_nonzero:
        raise ValueError("Masked ADR feature values must be exactly zero")
    audit: dict[str, object] = {
        "pair_count": len(PAIRS),
        "identity_map": {
            pair.adr_symbol: {
                "local_ticker": pair.local_ticker,
                "security_id": identities[pair.local_ticker],
            }
            for pair in PAIRS
        },
        "output_row_count": frame.height,
        "output_date_count": frame.get_column("available_date").n_unique(),
        "output_security_count": frame.get_column("security_id").n_unique(),
        "missing_active_identity_count": missing_active_identity,
        "first_available_date": str(frame.get_column("available_date").min()),
        "last_available_date": str(frame.get_column("available_date").max()),
        "first_source_session": str(frame.get_column("source_session_date").min()),
        "last_source_session": str(frame.get_column("source_session_date").max()),
        "feature_valid_rows": {
            feature: int(frame.get_column(f"{feature}_mask").sum())
            for feature in FEATURES
        },
    }
    return frame, audit


def build_artifact(
    raw_dir: Path,
    cotahist_dir: Path,
    security_index: Path,
    output_dir: Path,
    *,
    available_start: date,
    available_end: date,
) -> dict[str, object]:
    raw_dir = raw_dir.resolve()
    cotahist_dir = cotahist_dir.resolve()
    security_index = security_index.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    bars, events = _load_snapshot(raw_dir)
    split_audits = {
        symbol: _split_audit(symbol, bars[symbol], events[symbol])
        for symbol in ("EWZ", *(pair.adr_symbol for pair in PAIRS))
    }
    excluded_audits = {}
    for symbol, exclusion in EXCLUDED_PAIRS.items():
        candidate = bars[symbol]
        returns = _log_returns(candidate, 1)
        excluded_audits[symbol] = {
            **exclusion,
            "bar_count": len(candidate),
            "zero_volume_count": sum(bar.volume <= 0 for bar in candidate),
            "max_abs_adjusted_log_return": max(
                abs(value) for value in returns.values()
            ),
        }
    cotahist_files = _cotahist_files(cotahist_dir, available_end)
    accepted_ids = set(
        pl.read_parquet(security_index).get_column("security_id").unique().to_list()
    )
    cotahist = (
        pl.scan_parquet(cotahist_files)
        .filter(
            pl.col("trade_date").is_between(available_start, available_end),
            pl.col("security_id").is_in(accepted_ids),
        )
        .select("trade_date", "ticker", "security_id")
        .collect()
    )
    frame, output_audit = build_daily_frame(
        bars,
        cotahist,
        accepted_ids,
        available_start=available_start,
        available_end=available_end,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        data_path = temporary / "adr_overnight.parquet"
        frame.write_parquet(data_path, compression="zstd", statistics=True)
        raw_manifest = raw_dir / "manifest.json"
        manifest: dict[str, object] = {
            "contract_version": CONTRACT_VERSION,
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "features": list(FEATURES),
            "raw_snapshot": {
                "path": str(raw_dir),
                "manifest_path": str(raw_manifest),
                "manifest_sha256": _sha256(raw_manifest),
            },
            "cotahist_identity_files": [
                {"path": str(path), "sha256": _sha256(path)} for path in cotahist_files
            ],
            "security_index": {
                "path": str(security_index),
                "sha256": _sha256(security_index),
            },
            "availability_rule": (
                "For each B3 10:15 America/Sao_Paulo decision, use only the last "
                "completed 16:00 America/New_York session. Session-close timestamps "
                "are constructed with IANA time zones, so US DST is explicit."
            ),
            "price_rule": (
                "Yahoo adjusted close only. One- and five-observation log returns require "
                "positive-volume endpoints and bounded session gaps. Declared splits are "
                "audited against adjusted-return continuity before construction."
            ),
            "feature_rule": (
                "Pair-specific ADR 1d/5d returns and ADR-minus-EWZ 1d residual use "
                "fixed tanh compression. Residual surprise uses the preceding 60 valid "
                "observations only, requires 20, applies median/MAD with a fixed floor, "
                "and clips to +/-6 before scaling. EWZ is never emitted or broadcast."
            ),
            "omitted_claims": (
                "No after-hours history, ADR parity premium, FX conversion, or standalone "
                "sample-constant EWZ feature is claimed."
            ),
            "selected_symbol_audits": split_audits,
            "excluded_pair_audits": excluded_audits,
            "available_start": available_start.isoformat(),
            "available_end": available_end.isoformat(),
            **output_audit,
            "output_file": data_path.name,
            "output_sha256": _sha256(data_path),
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


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Acquire or build point-in-time ADR overnight features"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    acquire = subparsers.add_parser("acquire")
    acquire.add_argument("--output-dir", type=Path, required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--raw-dir", type=Path, required=True)
    build.add_argument("--cotahist-dir", type=Path, required=True)
    build.add_argument("--security-index", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--available-start", type=_parse_date, required=True)
    build.add_argument("--available-end", type=_parse_date, required=True)
    args = parser.parse_args()
    if args.command == "acquire":
        manifest = acquire_snapshot(args.output_dir)
        print(f"Acquired {len(manifest['sources'])} immutable Yahoo responses")
        return 0
    manifest = build_artifact(
        args.raw_dir,
        args.cotahist_dir,
        args.security_index,
        args.output_dir,
        available_start=args.available_start,
        available_end=args.available_end,
    )
    print(f"Wrote {manifest['output_row_count']:,} ADR rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
