from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
from bisect import bisect_right
from collections.abc import Collection, Sequence
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

CONTRACT_VERSION = "B3_COTAHIST_REGULAR_TRADE_ACTIVITY_V1"
WINDOWS = (20, 60)
ROBUST_Z_CLIP = 5.0
SHARE_UNIT_BREAK_RATIO = 1.25
METRICS = (
    "log1p_trades",
    "log_avg_trade_value",
    "log_shares_per_trade",
)
FEATURES = tuple(
    f"regular_{metric}_robust_z{window}" for metric in METRICS for window in WINDOWS
)
SOURCE_COLUMNS = (
    "trade_date",
    "security_id",
    "security_id_is_fallback",
    "isin",
    "bdi_code",
    "market_type",
    "currency",
    "close_brl",
    "trades",
    "quantity",
    "volume_brl",
    "distribution_number",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _robust_prior_z(
    current: float | None, prior: list[float], window: int
) -> tuple[float, bool]:
    if current is None or len(prior) < window:
        return 0.0, False
    history = prior[-window:]
    center = statistics.median(history)
    mad = statistics.median(abs(value - center) for value in history)
    scale = 1.4826 * mad
    if scale <= 1e-12:
        return 0.0, False
    value = min(max((current - center) / scale, -ROBUST_Z_CLIP), ROBUST_Z_CLIP)
    return value, True


def _validate_source(frame: pl.DataFrame) -> None:
    missing = sorted(set(SOURCE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Parsed COTAHIST source is missing columns: {missing}")
    if frame.schema["trade_date"] != pl.Date:
        raise ValueError("trade_date must have Polars Date dtype")
    if frame.schema["security_id"] != pl.String or frame.schema["isin"] != pl.String:
        raise ValueError("security_id and isin must have Polars String dtype")
    if frame.schema["security_id_is_fallback"] != pl.Boolean:
        raise ValueError("security_id_is_fallback must have Polars Boolean dtype")
    if frame.select(pl.any_horizontal(pl.all().is_null()).any()).item():
        raise ValueError("Parsed COTAHIST source columns cannot be null")
    if frame.filter(
        (pl.col("trades") < 0)
        | (pl.col("quantity") < 0)
        | (pl.col("volume_brl") < 0)
        | (pl.col("close_brl") < 0)
    ).height:
        raise ValueError("Parsed COTAHIST activity fields cannot be negative")


def _feature_name(metric: str, window: int) -> str:
    return f"regular_{metric}_robust_z{window}"


def build_normalized_frame(
    daily: pl.DataFrame,
    market_dates: Sequence[date],
    *,
    security_ids: Collection[str] | None = None,
    available_start: date | None = None,
    available_end: date | None = None,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Build causal regular-session activity features from parsed COTAHIST rows."""
    if available_start is not None and available_end is not None:
        if available_start > available_end:
            raise ValueError("available_start cannot be after available_end")
    _validate_source(daily)
    requested = None if security_ids is None else set(security_ids)
    if requested is not None:
        if not requested or any(not value.startswith("ISIN:") for value in requested):
            raise ValueError("security_ids must be nonempty permanent ISIN identities")
        daily = daily.filter(pl.col("security_id").is_in(requested))
    fallback_rows = daily.filter(pl.col("security_id_is_fallback")).height
    daily = daily.filter(~pl.col("security_id_is_fallback"))
    if daily.is_empty():
        raise ValueError("No exact-identity COTAHIST rows remain")
    if daily.filter(pl.col("security_id") != pl.lit("ISIN:") + pl.col("isin")).height:
        raise ValueError("COTAHIST security_id must exactly equal ISIN:<isin>")
    if daily.filter(
        (pl.col("bdi_code") != "02")
        | (pl.col("market_type") != 10)
        | (pl.col("currency") != "R$")
    ).height:
        raise ValueError("Source contains non-regular-board equity activity")
    duplicate = (
        daily.group_by("trade_date", "security_id")
        .len()
        .filter(pl.col("len") > 1)
        .head(1)
    )
    if duplicate.height:
        raise ValueError("COTAHIST source contains duplicate security/date rows")

    sessions = sorted(set(market_dates))
    if not sessions:
        raise ValueError("market_dates cannot be empty")
    session_set = set(sessions)
    if any(value not in session_set for value in daily.get_column("trade_date")):
        raise ValueError("A COTAHIST source date is absent from the market calendar")

    rows: list[dict[str, object]] = []
    histories: dict[str, dict[str, list[float]]] = {}
    previous_distribution: dict[str, int] = {}
    previous_close: dict[str, float] = {}
    missing_next_session = 0
    unit_break_count = 0
    invalid_avg_trade_value = 0
    invalid_shares_per_trade = 0

    for source in daily.sort("security_id", "trade_date").iter_rows(named=True):
        source_date = source["trade_date"]
        next_index = bisect_right(sessions, source_date)
        if next_index == len(sessions):
            missing_next_session += 1
            continue
        available_date = sessions[next_index]
        security_id = source["security_id"]
        security_history = histories.setdefault(
            security_id, {metric: [] for metric in METRICS}
        )

        distribution = int(source["distribution_number"])
        close = float(source["close_brl"])
        prior_distribution = previous_distribution.get(security_id)
        prior_close = previous_close.get(security_id)
        distribution_changed = (
            prior_distribution is not None and distribution != prior_distribution
        )
        close_discontinuity = (
            prior_close is None
            or prior_close <= 0
            or close <= 0
            or abs(math.log(close / prior_close)) >= math.log(SHARE_UNIT_BREAK_RATIO)
        )
        share_unit_break = distribution_changed and close_discontinuity
        if share_unit_break:
            security_history["log_shares_per_trade"].clear()
            unit_break_count += 1

        trades = int(source["trades"])
        quantity = int(source["quantity"])
        volume_brl = float(source["volume_brl"])
        current: dict[str, float | None] = {
            "log1p_trades": math.log1p(trades),
            "log_avg_trade_value": (
                math.log(volume_brl / trades) if trades > 0 and volume_brl > 0 else None
            ),
            "log_shares_per_trade": (
                math.log(quantity / trades) if trades > 0 and quantity > 0 else None
            ),
        }
        invalid_avg_trade_value += current["log_avg_trade_value"] is None
        invalid_shares_per_trade += current["log_shares_per_trade"] is None
        output: dict[str, object] = {
            "source_trade_date": source_date,
            "available_date": available_date,
            "security_id": security_id,
        }
        for metric in METRICS:
            metric_history = security_history[metric]
            for window in WINDOWS:
                feature = _feature_name(metric, window)
                value, valid = _robust_prior_z(current[metric], metric_history, window)
                output[feature] = value
                output[f"{feature}_mask"] = valid
            if current[metric] is not None:
                metric_history.append(current[metric])

        previous_distribution[security_id] = distribution
        previous_close[security_id] = close
        if available_start is not None and available_date < available_start:
            continue
        if available_end is not None and available_date > available_end:
            continue
        rows.append(output)

    if not rows:
        raise ValueError("No rows fall inside the requested availability interval")
    frame = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col("source_trade_date").cast(pl.Date),
        pl.col("available_date").cast(pl.Date),
        *[pl.col(feature).cast(pl.Float32) for feature in FEATURES],
        *[pl.col(f"{feature}_mask").cast(pl.Boolean) for feature in FEATURES],
    )
    frame = frame.select(
        "source_trade_date",
        "available_date",
        "security_id",
        *[column for feature in FEATURES for column in (feature, f"{feature}_mask")],
    ).sort("available_date", "security_id")
    audit: dict[str, object] = {
        "input_row_count": daily.height + fallback_rows,
        "fallback_identity_rows_removed": fallback_rows,
        "exact_identity_row_count": daily.height,
        "missing_next_session_row_count": missing_next_session,
        "output_row_count": frame.height,
        "output_security_count": frame.get_column("security_id").n_unique(),
        "share_unit_break_count": unit_break_count,
        "invalid_avg_trade_value_source_rows": invalid_avg_trade_value,
        "invalid_shares_per_trade_source_rows": invalid_shares_per_trade,
        "feature_valid_rows": {
            feature: int(frame.get_column(f"{feature}_mask").sum())
            for feature in FEATURES
        },
    }
    return frame, audit


def _source_files(cotahist_dir: Path, available_end: date | None) -> list[Path]:
    files = sorted(cotahist_dir.glob("year=*/equities_daily_*.parquet"))
    if available_end is not None:
        files = [
            path
            for path in files
            if int(path.parent.name.removeprefix("year=")) <= available_end.year
        ]
    if not files:
        raise FileNotFoundError(f"No parsed COTAHIST equities under {cotahist_dir}")
    return files


def _load_security_ids(path: Path | None) -> tuple[str, ...] | None:
    if path is None:
        return None
    frame = pl.read_parquet(path).select("security_id")
    values = tuple(frame.get_column("security_id").to_list())
    if len(values) != len(set(values)):
        raise ValueError("security index contains duplicate security_id values")
    return values


def build_artifact(
    cotahist_dir: Path,
    output_dir: Path,
    *,
    security_index: Path | None = None,
    available_start: date | None = None,
    available_end: date | None = None,
) -> dict[str, object]:
    cotahist_dir = cotahist_dir.resolve()
    output_dir = output_dir.resolve()
    security_index = None if security_index is None else security_index.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    files = _source_files(cotahist_dir, available_end)
    security_ids = _load_security_ids(security_index)
    scan = pl.scan_parquet(files).select(SOURCE_COLUMNS)
    if available_end is not None:
        scan = scan.filter(pl.col("trade_date") < available_end)
    if security_ids is not None:
        scan = scan.filter(pl.col("security_id").is_in(security_ids))
    daily = scan.collect()
    calendar_scan = pl.scan_parquet(files).select(pl.col("trade_date").unique())
    if available_end is not None:
        calendar_scan = calendar_scan.filter(pl.col("trade_date") <= available_end)
    market_dates = (
        calendar_scan.collect().get_column("trade_date").unique().sort().to_list()
    )
    frame, audit = build_normalized_frame(
        daily,
        market_dates,
        security_ids=security_ids,
        available_start=available_start,
        available_end=available_end,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        data_path = temporary / "regular_trade_activity.parquet"
        frame.write_parquet(data_path, compression="zstd", statistics=True)
        parse_audit = cotahist_dir / "parse_audit.json"
        manifest: dict[str, object] = {
            "contract_version": CONTRACT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "source_directory": str(cotahist_dir),
            "source_files": [
                {"path": str(path.resolve()), "sha256": _sha256(path)} for path in files
            ],
            "source_parse_audit": (
                {"path": str(parse_audit), "sha256": _sha256(parse_audit)}
                if parse_audit.is_file()
                else None
            ),
            "security_index": (
                {"path": str(security_index), "sha256": _sha256(security_index)}
                if security_index is not None
                else None
            ),
            "availability_rule": (
                "Regular-session COTAHIST aggregate for trade date D is assigned "
                "exactly to the next observed B3 session after D"
            ),
            "identity_rule": (
                "Exact non-fallback permanent security_id=ISIN:<isin>; ticker is unused"
            ),
            "normalization_rule": (
                "Per-security clipped median/MAD surprise over the previous 20 or "
                "60 valid observations; the current observation is appended only "
                "after every feature is emitted"
            ),
            "share_unit_rule": (
                "Shares/trade history alone resets when distribution_number changes "
                f"and the causal close ratio crosses {SHARE_UNIT_BREAK_RATIO}; the "
                "event observation starts the new history and emits no share surprise"
            ),
            "missingness_rule": (
                "Missing security-dates emit no row; undefined ratios and insufficient "
                "prior history emit exact zero with an explicit false mask"
            ),
            "features": list(FEATURES),
            "windows": list(WINDOWS),
            "robust_z_clip": ROBUST_Z_CLIP,
            "available_start_filter": (
                available_start.isoformat() if available_start else None
            ),
            "available_end_filter": (
                available_end.isoformat() if available_end else None
            ),
            "calendar_first_session": market_dates[0].isoformat(),
            "calendar_last_session": market_dates[-1].isoformat(),
            "calendar_session_count": len(market_dates),
            **audit,
            "first_source_trade_date": str(frame.get_column("source_trade_date").min()),
            "last_source_trade_date": str(frame.get_column("source_trade_date").max()),
            "first_available_date": str(frame.get_column("available_date").min()),
            "last_available_date": str(frame.get_column("available_date").max()),
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
        description="Build causal regular-session COTAHIST activity features"
    )
    parser.add_argument("--cotahist-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--security-index", type=Path)
    parser.add_argument("--available-start", type=_parse_date)
    parser.add_argument("--available-end", type=_parse_date)
    args = parser.parse_args()
    manifest = build_artifact(
        args.cotahist_dir,
        args.output_dir,
        security_index=args.security_index,
        available_start=args.available_start,
        available_end=args.available_end,
    )
    print(
        f"Wrote {manifest['output_row_count']:,} regular-activity rows to "
        f"{args.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
