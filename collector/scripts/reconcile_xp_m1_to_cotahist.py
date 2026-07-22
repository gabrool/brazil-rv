from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SCRIPT_VERSION = "1"


@dataclass(frozen=True)
class MatchThresholds:
    min_overlap_days: int = 10
    strong_exact_ohlc_share: float = 0.90
    strong_exact_close_share: float = 0.95
    strong_return_corr: float = 0.995
    strong_ratio_mad: float = 0.002
    high_member_coverage: float = 0.80
    partial_member_coverage: float = 0.50


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_hhmm(value: str) -> int:
    parsed = datetime.strptime(value, "%H:%M").time()
    return parsed.hour * 60 + parsed.minute


def ensure_new_dir(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Output directory already exists: {path}")
    path.mkdir(parents=True, exist_ok=False)


def load_parquet_tree(root: Path, pattern: str) -> pd.DataFrame:
    paths = sorted(root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched {root / pattern}")
    frames = [pd.read_parquet(path) for path in paths]
    return pd.concat(frames, ignore_index=True, sort=False)


def discover_m1_files(roots: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"M1 root does not exist: {root}")
        for path in sorted(root.rglob("bars_m1_*.parquet")):
            try:
                symbol_frame = pd.read_parquet(path, columns=["symbol", "retrieved_at_utc"])
                if symbol_frame.empty:
                    rows.append({
                        "symbol": "",
                        "path": str(path),
                        "root": str(root),
                        "rows": 0,
                        "retrieved_at_utc": "",
                        "error": "empty parquet",
                    })
                    continue
                symbols = symbol_frame["symbol"].dropna().astype(str).unique().tolist()
                if len(symbols) != 1:
                    rows.append({
                        "symbol": "|".join(symbols),
                        "path": str(path),
                        "root": str(root),
                        "rows": len(symbol_frame),
                        "retrieved_at_utc": "",
                        "error": f"expected one symbol, found {len(symbols)}",
                    })
                    continue
                retrievals = symbol_frame["retrieved_at_utc"].dropna().astype(str)
                rows.append({
                    "symbol": symbols[0],
                    "path": str(path),
                    "root": str(root),
                    "rows": len(symbol_frame),
                    "retrieved_at_utc": retrievals.iloc[-1] if not retrievals.empty else "",
                    "error": "",
                })
            except Exception as exc:  # noqa: BLE001
                rows.append({
                    "symbol": "",
                    "path": str(path),
                    "root": str(root),
                    "rows": 0,
                    "retrieved_at_utc": "",
                    "error": f"{type(exc).__name__}: {exc}",
                })
    inventory = pd.DataFrame(rows)
    if inventory.empty:
        raise FileNotFoundError("No M1 parquet files were found")
    good = inventory[(inventory["error"] == "") & (inventory["symbol"] != "")].copy()
    if good.empty:
        raise ValueError("No valid M1 files were found")

    # Prefer the file with the greatest number of rows. Break ties using the
    # latest retrieval timestamp and then path. Duplicate files remain audited.
    good = good.sort_values(
        ["symbol", "rows", "retrieved_at_utc", "path"],
        ascending=[True, False, False, True],
    )
    chosen = good.groupby("symbol", as_index=False, sort=True).first()
    duplicate_counts = good.groupby("symbol").size().rename("file_count").reset_index()
    duplicates = good.merge(duplicate_counts, on="symbol", how="left")
    duplicates = duplicates[duplicates["file_count"] > 1].copy()
    chosen = chosen.merge(duplicate_counts, on="symbol", how="left")
    return chosen, pd.concat([inventory[inventory["error"] != ""], duplicates], ignore_index=True, sort=False)


def aggregate_one_m1_file(
    symbol: str,
    path: Path,
    start_date: date,
    end_date: date,
    cutoff_minute: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    columns = [
        "ts_exchange", "open", "high", "low", "close", "real_volume",
        "tick_volume", "spread",
    ]
    frame = pd.read_parquet(path, columns=columns)
    if frame.empty:
        raise ValueError("M1 file is empty")
    ts = pd.to_datetime(frame["ts_exchange"], errors="coerce")
    if ts.isna().any():
        raise ValueError(f"{int(ts.isna().sum())} invalid ts_exchange values")
    frame = frame.assign(ts_exchange=ts)
    frame["trade_date"] = frame["ts_exchange"].dt.date
    frame = frame[(frame["trade_date"] >= start_date) & (frame["trade_date"] <= end_date)].copy()
    if frame.empty:
        raise ValueError("No rows inside requested research interval")
    frame = frame.sort_values("ts_exchange", kind="stable")
    frame["minute_of_day"] = frame["ts_exchange"].dt.hour * 60 + frame["ts_exchange"].dt.minute

    def aggregate(source: pd.DataFrame, prefix: str) -> pd.DataFrame:
        grouped = source.groupby("trade_date", sort=True, observed=True)
        result = grouped.agg(
            **{
                f"{prefix}_open": ("open", "first"),
                f"{prefix}_high": ("high", "max"),
                f"{prefix}_low": ("low", "min"),
                f"{prefix}_close": ("close", "last"),
                f"{prefix}_quantity": ("real_volume", "sum"),
                f"{prefix}_tick_volume": ("tick_volume", "sum"),
                f"{prefix}_bar_count": ("close", "size"),
                f"{prefix}_first_ts": ("ts_exchange", "min"),
                f"{prefix}_last_ts": ("ts_exchange", "max"),
            }
        ).reset_index()
        return result

    all_day = aggregate(frame, "all")
    pre_cutoff_source = frame[frame["minute_of_day"] <= cutoff_minute]
    pre_cutoff = aggregate(pre_cutoff_source, "pre") if not pre_cutoff_source.empty else pd.DataFrame({"trade_date": []})
    daily = all_day.merge(pre_cutoff, on="trade_date", how="left")
    daily.insert(0, "xp_symbol", symbol)
    daily["source_file"] = str(path)

    audit = {
        "xp_symbol": symbol,
        "source_file": str(path),
        "m1_rows": int(len(frame)),
        "daily_rows": int(len(daily)),
        "first_exchange": frame["ts_exchange"].min().isoformat(),
        "last_exchange": frame["ts_exchange"].max().isoformat(),
        "first_trade_date": str(daily["trade_date"].min()),
        "last_trade_date": str(daily["trade_date"].max()),
        "days_with_pre_cutoff": int(daily["pre_close"].notna().sum()) if "pre_close" in daily else 0,
        "error": "",
    }
    return daily, audit


def add_price_keys(frame: pd.DataFrame, prefix: str, scale: int) -> pd.DataFrame:
    result = frame.copy()
    for field in ("open", "high", "low", "close"):
        source = f"{prefix}_{field}"
        result[f"{prefix}_{field}_key"] = np.rint(result[source].astype(float) * scale).astype("int64")
    result[f"{prefix}_quantity_key"] = np.rint(result[f"{prefix}_quantity"].fillna(0).astype(float)).astype("int64")
    return result


def build_variant_frame(xp_daily: pd.DataFrame, scale: int) -> pd.DataFrame:
    variants: list[pd.DataFrame] = []
    for variant, prefix in (("all", "all"), ("pre_cutoff", "pre")):
        required = [f"{prefix}_{name}" for name in ("open", "high", "low", "close", "quantity")]
        if not all(column in xp_daily.columns for column in required):
            continue
        part = xp_daily[["xp_symbol", "source_file", "trade_date", *required]].copy()
        part = part.dropna(subset=required)
        part = part.rename(columns={f"{prefix}_{name}": name for name in ("open", "high", "low", "close", "quantity")})
        part["variant"] = variant
        for field in ("open", "high", "low", "close"):
            part[f"{field}_key"] = np.rint(part[field].astype(float) * scale).astype("int64")
        part["quantity_key"] = np.rint(part["quantity"].astype(float)).astype("int64")
        variants.append(part)
    if not variants:
        raise ValueError("No XP daily variants could be constructed")
    return pd.concat(variants, ignore_index=True, sort=False)


def prepare_cotahist(daily_root: Path, start_date: date, end_date: date, scale: int) -> pd.DataFrame:
    cot = load_parquet_tree(daily_root, "year=*/equities_daily_*.parquet")
    cot["trade_date"] = pd.to_datetime(cot["trade_date"]).dt.date
    cot = cot[(cot["trade_date"] >= start_date) & (cot["trade_date"] <= end_date)].copy()
    cot = cot.drop_duplicates(["trade_date", "security_id"], keep="last")
    rename = {
        "open_brl": "open",
        "high_brl": "high",
        "low_brl": "low",
        "close_brl": "close",
        "quantity": "quantity",
    }
    cot = cot.rename(columns=rename)
    for field in ("open", "high", "low", "close"):
        cot[f"{field}_key"] = np.rint(cot[field].astype(float) * scale).astype("int64")
    cot["quantity_key"] = np.rint(cot["quantity"].fillna(0).astype(float)).astype("int64")
    return cot


def signature_counts(xp_variants: pd.DataFrame, cot: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["xp_symbol", "variant", "security_id"]
    key_sets = {
        "exact_ohlc_quantity_days": ["trade_date", "open_key", "high_key", "low_key", "close_key", "quantity_key"],
        "exact_ohlc_days": ["trade_date", "open_key", "high_key", "low_key", "close_key"],
        "exact_close_quantity_days": ["trade_date", "close_key", "quantity_key"],
    }
    outputs: list[pd.DataFrame] = []
    cot_columns = ["security_id", *set(sum(key_sets.values(), []))]
    cot_keys = cot[cot_columns].copy()
    for metric, keys in key_sets.items():
        joined = xp_variants[["xp_symbol", "variant", *keys]].merge(
            cot_keys[["security_id", *keys]],
            on=keys,
            how="inner",
        )
        if joined.empty:
            continue
        counts = joined.groupby(group_columns, as_index=False).size().rename(columns={"size": metric})
        outputs.append(counts)
    if not outputs:
        return pd.DataFrame(columns=[*group_columns, *key_sets])
    result = outputs[0]
    for output in outputs[1:]:
        result = result.merge(output, on=group_columns, how="outer")
    for metric in key_sets:
        if metric not in result:
            result[metric] = 0
        result[metric] = result[metric].fillna(0).astype("int64")
    return result


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    left = left[mask]
    right = right[mask]
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def ratio_stats(numerator: np.ndarray, denominator: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
    ratios = numerator[mask] / denominator[mask]
    if len(ratios) == 0:
        return float("nan"), float("nan")
    median = float(np.median(ratios))
    if median == 0:
        return median, float("nan")
    rel_mad = float(np.median(np.abs(ratios / median - 1.0)))
    return median, rel_mad


def member_mask(dates: pd.Series, intervals: list[tuple[date, date | None]], end_date: date) -> np.ndarray:
    values = dates.to_numpy()
    mask = np.zeros(len(values), dtype=bool)
    for start, end_exclusive in intervals:
        effective_end = end_exclusive
        if effective_end is None:
            mask |= np.array([value >= start for value in values], dtype=bool)
        else:
            mask |= np.array([(value >= start) and (value < effective_end) for value in values], dtype=bool)
    return mask


def build_membership_intervals(membership: pd.DataFrame) -> dict[str, list[tuple[date, date | None]]]:
    membership = membership.copy()
    membership["effective_from"] = pd.to_datetime(membership["effective_from"]).dt.date
    if "effective_to_exclusive" in membership.columns:
        membership["effective_to_exclusive"] = pd.to_datetime(membership["effective_to_exclusive"], errors="coerce").dt.date
    else:
        membership["effective_to_exclusive"] = None
    result: dict[str, list[tuple[date, date | None]]] = {}
    for sid, group in membership.groupby("security_id", sort=False):
        intervals = []
        for row in group[["effective_from", "effective_to_exclusive"]].drop_duplicates().sort_values("effective_from").itertuples(index=False):
            end_value = row.effective_to_exclusive
            intervals.append((row.effective_from, None if pd.isna(end_value) else end_value))
        result[str(sid)] = intervals
    return result


def pair_metrics(
    xp: pd.DataFrame,
    cot: pd.DataFrame,
    member_intervals: list[tuple[date, date | None]],
    research_end: date,
    price_scale: int,
) -> dict[str, object]:
    joined = xp.merge(cot, on="trade_date", how="inner", suffixes=("_xp", "_cot")).sort_values("trade_date")
    n = len(joined)
    cot_days = len(cot)
    if n == 0:
        return {
            "overlap_days": 0,
            "cot_days_in_research": cot_days,
            "member_cot_days": 0,
            "overlap_share_of_cot": 0.0,
        }

    exact_fields = {}
    for field in ("open", "high", "low", "close"):
        exact_fields[field] = np.rint(joined[f"{field}_xp"].to_numpy(float) * price_scale).astype(np.int64) == np.rint(
            joined[f"{field}_cot"].to_numpy(float) * price_scale
        ).astype(np.int64)
    exact_ohlc = exact_fields["open"] & exact_fields["high"] & exact_fields["low"] & exact_fields["close"]
    xp_qty = np.rint(joined["quantity_xp"].to_numpy(float)).astype(np.int64)
    cot_qty = np.rint(joined["quantity_cot"].to_numpy(float)).astype(np.int64)
    exact_qty = xp_qty == cot_qty

    member_joined = member_mask(joined["trade_date"], member_intervals, research_end) if member_intervals else np.zeros(n, dtype=bool)
    member_cot = member_mask(cot["trade_date"], member_intervals, research_end) if member_intervals else np.zeros(cot_days, dtype=bool)
    member_cot_days = int(member_cot.sum())

    xp_close = joined["close_xp"].to_numpy(float)
    cot_close = joined["close_cot"].to_numpy(float)
    close_ratio_median, close_ratio_rel_mad = ratio_stats(xp_close, cot_close)
    volume_ratio_median, volume_ratio_rel_mad = ratio_stats(xp_qty.astype(float), cot_qty.astype(float))

    xp_returns = np.diff(np.log(np.where(xp_close > 0, xp_close, np.nan)))
    cot_returns = np.diff(np.log(np.where(cot_close > 0, cot_close, np.nan)))
    return_corr = safe_corr(xp_returns, cot_returns)

    close_rel_error = np.abs(xp_close - cot_close) / np.maximum(np.abs(cot_close), 1e-12)
    result = {
        "overlap_days": n,
        "cot_days_in_research": cot_days,
        "member_cot_days": member_cot_days,
        "first_overlap_date": str(joined["trade_date"].min()),
        "last_overlap_date": str(joined["trade_date"].max()),
        "overlap_share_of_cot": n / cot_days if cot_days else 0.0,
        "exact_open_share": float(exact_fields["open"].mean()),
        "exact_high_share": float(exact_fields["high"].mean()),
        "exact_low_share": float(exact_fields["low"].mean()),
        "exact_close_share": float(exact_fields["close"].mean()),
        "exact_ohlc_share": float(exact_ohlc.mean()),
        "exact_quantity_share": float(exact_qty.mean()),
        "exact_ohlc_quantity_share": float((exact_ohlc & exact_qty).mean()),
        "exact_ohlc_days": int(exact_ohlc.sum()),
        "exact_close_days": int(exact_fields["close"].sum()),
        "exact_quantity_days": int(exact_qty.sum()),
        "median_close_relative_error": float(np.nanmedian(close_rel_error)),
        "p95_close_relative_error": float(np.nanquantile(close_rel_error, 0.95)),
        "close_ratio_median": close_ratio_median,
        "close_ratio_relative_mad": close_ratio_rel_mad,
        "volume_ratio_median": volume_ratio_median,
        "volume_ratio_relative_mad": volume_ratio_rel_mad,
        "return_correlation": return_corr,
        "member_overlap_days": int(member_joined.sum()),
        "member_exact_ohlc_days": int((exact_ohlc & member_joined).sum()),
        "member_exact_close_days": int((exact_fields["close"] & member_joined).sum()),
        "member_overlap_coverage": int(member_joined.sum()) / member_cot_days if member_cot_days else float("nan"),
        "member_exact_ohlc_coverage": int((exact_ohlc & member_joined).sum()) / member_cot_days if member_cot_days else float("nan"),
        "member_exact_close_coverage": int((exact_fields["close"] & member_joined).sum()) / member_cot_days if member_cot_days else float("nan"),
    }
    return result


def classify_pair(row: pd.Series, thresholds: MatchThresholds) -> str:
    overlap = int(row.get("overlap_days", 0))
    exact_ohlc = float(row.get("exact_ohlc_share", 0.0) or 0.0)
    exact_close = float(row.get("exact_close_share", 0.0) or 0.0)
    corr = float(row.get("return_correlation", float("nan")))
    ratio_mad = float(row.get("close_ratio_relative_mad", float("nan")))
    if overlap >= thresholds.min_overlap_days and exact_ohlc >= thresholds.strong_exact_ohlc_share and exact_close >= thresholds.strong_exact_close_share:
        return "STRONG_EXACT"
    if overlap >= max(20, thresholds.min_overlap_days) and exact_ohlc >= 0.80 and exact_close >= thresholds.strong_exact_close_share and (math.isnan(corr) or corr >= 0.99):
        return "STRONG_PRICE"
    if overlap >= 40 and not math.isnan(corr) and corr >= thresholds.strong_return_corr and not math.isnan(ratio_mad) and ratio_mad <= thresholds.strong_ratio_mad:
        return "STRONG_SCALED"
    if bool(row.get("direct_mapping_candidate", False)) and overlap >= 20 and (exact_close >= 0.80 or (not math.isnan(corr) and corr >= 0.98)):
        return "REVIEW_DIRECT"
    return "WEAK"


def score_pair(row: pd.Series) -> float:
    def finite(value: object, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if math.isfinite(parsed) else default

    return (
        0.40 * finite(row.get("exact_ohlc_share"))
        + 0.20 * finite(row.get("exact_close_share"))
        + 0.10 * finite(row.get("exact_quantity_share"))
        + 0.10 * finite(row.get("exact_ohlc_quantity_share"))
        + 0.10 * max(finite(row.get("return_correlation")), 0.0)
        + 0.10 * min(finite(row.get("member_overlap_coverage")), 1.0)
    )


def write_frame(frame: pd.DataFrame, stem: Path) -> None:
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    frame.to_parquet(stem.with_suffix(".parquet"), index=False, compression="zstd")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate XP M1 equities to daily bars and reconcile them to B3 COTAHIST permanent security IDs."
    )
    parser.add_argument("--universe-dir", type=Path, required=True)
    parser.add_argument("--daily-root", type=Path, required=True, help="Parsed COTAHIST root containing year=*/equities_daily_*.parquet")
    parser.add_argument("--mapping-csv", type=Path, required=True, help="security_xp_mapping.csv from the point-in-time-to-XP mapping audit")
    parser.add_argument("--m1-root", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", default="2021-07-19")
    parser.add_argument("--end", default="2026-07-17")
    parser.add_argument("--pre-cutoff", default="18:00")
    parser.add_argument("--price-scale", type=int, default=10000)
    parser.add_argument("--candidate-min-signature-days", type=int, default=3)
    args = parser.parse_args()

    ensure_new_dir(args.out)
    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)
    cutoff_minute = parse_hhmm(args.pre_cutoff)
    thresholds = MatchThresholds()

    mapping = pd.read_csv(args.mapping_csv)
    union = pd.read_parquet(args.universe_dir / "universe_union.parquet")
    membership = pd.read_parquet(args.universe_dir / "universe_membership_monthly.parquet")
    union_ids = set(union["security_id"].astype(str))
    membership_intervals = build_membership_intervals(membership)

    chosen_files, duplicate_file_audit = discover_m1_files(args.m1_root)
    write_frame(chosen_files, args.out / "m1_file_inventory")
    write_frame(duplicate_file_audit, args.out / "m1_duplicate_or_invalid_files")

    daily_frames: list[pd.DataFrame] = []
    aggregation_audit: list[dict[str, object]] = []
    for row in chosen_files.itertuples(index=False):
        print(f"Aggregating {row.symbol} ...", flush=True)
        try:
            daily, audit = aggregate_one_m1_file(
                str(row.symbol), Path(row.path), start_date, end_date, cutoff_minute
            )
            daily_frames.append(daily)
            aggregation_audit.append(audit)
        except Exception as exc:  # noqa: BLE001
            aggregation_audit.append({
                "xp_symbol": str(row.symbol),
                "source_file": str(row.path),
                "m1_rows": int(row.rows),
                "daily_rows": 0,
                "first_exchange": "",
                "last_exchange": "",
                "first_trade_date": "",
                "last_trade_date": "",
                "days_with_pre_cutoff": 0,
                "error": f"{type(exc).__name__}: {exc}",
            })
    aggregation_audit_frame = pd.DataFrame(aggregation_audit)
    write_frame(aggregation_audit_frame, args.out / "daily_aggregation_audit")
    failures = aggregation_audit_frame[aggregation_audit_frame["error"] != ""]
    if not failures.empty:
        print(f"WARNING: {len(failures)} M1 files failed daily aggregation", flush=True)
    if not daily_frames:
        raise RuntimeError("No M1 file could be aggregated")
    xp_daily = pd.concat(daily_frames, ignore_index=True, sort=False)
    xp_daily.to_parquet(args.out / "xp_daily_aggregates.parquet", index=False, compression="zstd")

    print("Loading COTAHIST daily data ...", flush=True)
    cot = prepare_cotahist(args.daily_root, start_date, end_date, args.price_scale)
    xp_variants = build_variant_frame(xp_daily, args.price_scale)

    print("Generating exact-signature candidate pairs ...", flush=True)
    counts = signature_counts(xp_variants, cot)
    if counts.empty:
        counts = pd.DataFrame(columns=[
            "xp_symbol", "variant", "security_id", "exact_ohlc_quantity_days",
            "exact_ohlc_days", "exact_close_quantity_days",
        ])

    # Add direct current-symbol candidates even if their raw daily signatures are
    # affected by adjustment conventions.
    direct = mapping[["security_id", "preferred_xp_symbol", "mapping_status"]].copy()
    direct = direct.rename(columns={"preferred_xp_symbol": "xp_symbol", "mapping_status": "original_mapping_status"})
    direct = direct[direct["xp_symbol"].notna() & (direct["xp_symbol"].astype(str) != "")]
    direct = direct[direct["xp_symbol"].astype(str).isin(set(xp_variants["xp_symbol"].astype(str)))]
    direct_variants = pd.DataFrame({"variant": ["all", "pre_cutoff"]})
    direct["__key"] = 1
    direct_variants["__key"] = 1
    direct = direct.merge(direct_variants, on="__key", how="inner").drop(columns="__key")

    candidate_counts = counts.copy()
    if not candidate_counts.empty:
        candidate_counts = candidate_counts[
            candidate_counts[["exact_ohlc_quantity_days", "exact_ohlc_days", "exact_close_quantity_days"]].max(axis=1)
            >= args.candidate_min_signature_days
        ]
    candidate_pairs = candidate_counts[["xp_symbol", "variant", "security_id"]].drop_duplicates() if not candidate_counts.empty else pd.DataFrame(columns=["xp_symbol", "variant", "security_id"])
    candidate_pairs = pd.concat(
        [candidate_pairs, direct[["xp_symbol", "variant", "security_id"]]],
        ignore_index=True,
    ).drop_duplicates()
    candidate_pairs = candidate_pairs.merge(candidate_counts, on=["xp_symbol", "variant", "security_id"], how="left")
    for column in ("exact_ohlc_quantity_days", "exact_ohlc_days", "exact_close_quantity_days"):
        candidate_pairs[column] = candidate_pairs[column].fillna(0).astype("int64")
    candidate_pairs = candidate_pairs.merge(
        direct[["xp_symbol", "variant", "security_id", "original_mapping_status"]].drop_duplicates(),
        on=["xp_symbol", "variant", "security_id"],
        how="left",
    )
    candidate_pairs["direct_mapping_candidate"] = candidate_pairs["original_mapping_status"].notna()

    xp_groups = {(symbol, variant): group.sort_values("trade_date") for (symbol, variant), group in xp_variants.groupby(["xp_symbol", "variant"], sort=False)}
    cot_groups = {str(sid): group.sort_values("trade_date") for sid, group in cot.groupby("security_id", sort=False)}

    print(f"Scoring {len(candidate_pairs):,} candidate pairs ...", flush=True)
    score_rows: list[dict[str, object]] = []
    for candidate in candidate_pairs.itertuples(index=False):
        sid = str(candidate.security_id)
        xp_key = (str(candidate.xp_symbol), str(candidate.variant))
        xp_group = xp_groups.get(xp_key)
        cot_group = cot_groups.get(sid)
        if xp_group is None or cot_group is None:
            continue
        metrics = pair_metrics(
            xp_group[["trade_date", "open", "high", "low", "close", "quantity"]],
            cot_group[["trade_date", "open", "high", "low", "close", "quantity"]],
            membership_intervals.get(sid, []),
            end_date,
            args.price_scale,
        )
        row = {
            "xp_symbol": str(candidate.xp_symbol),
            "variant": str(candidate.variant),
            "security_id": sid,
            "isin": sid[5:] if sid.startswith("ISIN:") else "",
            "is_union_security": sid in union_ids,
            "source_file": xp_group["source_file"].iloc[0],
            "direct_mapping_candidate": bool(candidate.direct_mapping_candidate),
            "original_mapping_status": getattr(candidate, "original_mapping_status", ""),
            "signature_exact_ohlc_quantity_days": int(candidate.exact_ohlc_quantity_days),
            "signature_exact_ohlc_days": int(candidate.exact_ohlc_days),
            "signature_exact_close_quantity_days": int(candidate.exact_close_quantity_days),
            **metrics,
        }
        score_rows.append(row)
    pair_scores = pd.DataFrame(score_rows)
    if pair_scores.empty:
        raise RuntimeError("No candidate pair could be scored")
    pair_scores["pair_classification"] = pair_scores.apply(lambda row: classify_pair(row, thresholds), axis=1)
    pair_scores["pair_score"] = pair_scores.apply(score_pair, axis=1)
    pair_scores = pair_scores.sort_values(
        ["is_union_security", "security_id", "pair_score", "overlap_days"],
        ascending=[False, True, False, False],
    )
    write_frame(pair_scores, args.out / "xp_cotahist_pair_scores")

    # Select the best variant for each XP symbol/security pair, then the best XP
    # source for each permanent security ID.
    best_variant = (
        pair_scores.sort_values(
            ["xp_symbol", "security_id", "pair_score", "overlap_days"],
            ascending=[True, True, False, False],
        )
        .drop_duplicates(["xp_symbol", "security_id"], keep="first")
        .reset_index(drop=True)
    )
    write_frame(best_variant, args.out / "best_variant_per_symbol_security")

    union_best = best_variant[best_variant["security_id"].isin(union_ids)].copy()
    union_best = union_best.sort_values(
        ["security_id", "pair_score", "member_exact_ohlc_coverage", "member_overlap_coverage"],
        ascending=[True, False, False, False],
    )
    chosen_security = (
        union_best.drop_duplicates(["security_id"], keep="first")
        .reset_index(drop=True)
    )

    union_table = union.copy()
    coverage = union_table.merge(chosen_security, on="security_id", how="left", suffixes=("", "_match"))
    coverage = coverage.merge(
        mapping[["security_id", "mapping_status", "preferred_xp_symbol", "already_archived"]],
        on="security_id",
        how="left",
    )

    def coverage_status(row: pd.Series) -> str:
        classification = str(row.get("pair_classification", ""))
        exact_coverage = row.get("member_exact_ohlc_coverage", np.nan)
        overlap_coverage = row.get("member_overlap_coverage", np.nan)
        exact_coverage = float(exact_coverage) if pd.notna(exact_coverage) else 0.0
        overlap_coverage = float(overlap_coverage) if pd.notna(overlap_coverage) else 0.0
        strong = classification in {"STRONG_EXACT", "STRONG_PRICE", "STRONG_SCALED"}
        if strong and max(exact_coverage, overlap_coverage) >= thresholds.high_member_coverage:
            return "RECONCILED_HIGH"
        if strong and max(exact_coverage, overlap_coverage) >= thresholds.partial_member_coverage:
            return "RECONCILED_PARTIAL"
        if classification in {"REVIEW_DIRECT", "STRONG_EXACT", "STRONG_PRICE", "STRONG_SCALED"}:
            return "REVIEW"
        return "UNRESOLVED"

    coverage["reconciliation_status"] = coverage.apply(coverage_status, axis=1)
    coverage["recovered_from_relabelled_xp_symbol"] = (
        (coverage["mapping_status"] == "NO_EXACT_XP_SYMBOL")
        & coverage["xp_symbol"].notna()
        & (coverage["reconciliation_status"].isin(["RECONCILED_HIGH", "RECONCILED_PARTIAL"]))
    )
    write_frame(coverage, args.out / "union_security_coverage")

    source_assignments = coverage[
        coverage["reconciliation_status"].isin(["RECONCILED_HIGH", "RECONCILED_PARTIAL"])
    ][[
        "security_id", "isin", "latest_ticker", "first_effective_from", "last_effective_from",
        "xp_symbol", "source_file", "variant", "pair_classification", "pair_score",
        "first_overlap_date", "last_overlap_date", "member_cot_days", "member_overlap_days",
        "member_exact_ohlc_days", "member_overlap_coverage", "member_exact_ohlc_coverage",
        "recovered_from_relabelled_xp_symbol",
    ]].copy()
    write_frame(source_assignments, args.out / "security_m1_source_assignments")

    recovered = coverage[coverage["recovered_from_relabelled_xp_symbol"]].copy()
    unresolved = coverage[coverage["reconciliation_status"] == "UNRESOLVED"].copy()
    review = coverage[coverage["reconciliation_status"] == "REVIEW"].copy()
    write_frame(recovered, args.out / "recovered_missing_security_ids")
    write_frame(unresolved, args.out / "unresolved_security_ids")
    write_frame(review, args.out / "review_security_ids")

    ambiguous = best_variant[
        best_variant["pair_classification"].isin(["STRONG_EXACT", "STRONG_PRICE", "STRONG_SCALED"])
    ].groupby("security_id").filter(lambda group: group["xp_symbol"].nunique() > 1)
    write_frame(ambiguous, args.out / "ambiguous_multiple_xp_sources")

    summary = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": iso_now(),
        "universe_dir": str(args.universe_dir),
        "daily_root": str(args.daily_root),
        "mapping_csv": str(args.mapping_csv),
        "m1_roots": [str(root) for root in args.m1_root],
        "research_start": args.start,
        "research_end": args.end,
        "pre_cutoff": args.pre_cutoff,
        "m1_symbols_discovered": int(chosen_files["symbol"].nunique()),
        "m1_daily_rows": int(len(xp_daily)),
        "cotahist_security_days": int(len(cot)),
        "cotahist_parent_securities": int(cot["security_id"].nunique()),
        "union_securities": int(len(union_ids)),
        "candidate_pairs_scored": int(len(pair_scores)),
        "reconciliation_status_counts": coverage["reconciliation_status"].value_counts(dropna=False).to_dict(),
        "recovered_from_relabelled_xp_symbol": int(coverage["recovered_from_relabelled_xp_symbol"].sum()),
        "ambiguous_security_ids": int(ambiguous["security_id"].nunique()) if not ambiguous.empty else 0,
        "aggregation_failures": int(len(failures)),
    }
    (args.out / "reconciliation_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
