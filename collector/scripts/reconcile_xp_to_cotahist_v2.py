from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_VERSION = "2"


@dataclass(frozen=True)
class RecoveryThresholds:
    min_overlap_days: int = 15
    high_overlap_coverage: float = 0.80
    high_shape_match_5ticks: float = 0.90
    high_ratio_stable_50bp: float = 0.90
    high_score: float = 0.80
    high_margin: float = 0.08
    review_score: float = 0.58


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_new_dir(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Output directory already exists: {path}")
    path.mkdir(parents=True, exist_ok=False)


def norm_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip().upper()


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x2 = x[mask]
    y2 = y[mask]
    if np.nanstd(x2) == 0 or np.nanstd(y2) == 0:
        return float("nan")
    return float(np.corrcoef(x2, y2)[0, 1])


def read_parquet_tree(root: Path, pattern: str) -> pd.DataFrame:
    paths = sorted(root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched {root / pattern}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True, sort=False)


def write_frame(frame: pd.DataFrame, stem: Path) -> None:
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    frame.to_parquet(stem.with_suffix(".parquet"), index=False, compression="zstd")


def load_catalogue(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, low_memory=False)
    required = {"name", "isin"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"XP catalogue is missing columns: {sorted(missing)}")
    keep = [column for column in ["name", "isin", "description", "path", "trade_tick_size"] if column in frame.columns]
    frame = frame[keep].copy()
    frame["name"] = frame["name"].map(norm_text)
    frame["isin"] = frame["isin"].map(norm_text)
    if "trade_tick_size" not in frame.columns:
        frame["trade_tick_size"] = np.nan
    frame = frame.drop_duplicates("name", keep="last")
    return frame


def load_cotahist(daily_root: Path, start: date, end: date) -> pd.DataFrame:
    frame = read_parquet_tree(daily_root, "year=*/equities_daily_*.parquet")
    required = {
        "trade_date", "security_id", "ticker", "isin", "open_brl", "high_brl",
        "low_brl", "close_brl", "quantity",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"COTAHIST data are missing columns: {sorted(missing)}")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    frame = frame[(frame["trade_date"] >= start) & (frame["trade_date"] <= end)].copy()
    frame = frame.drop_duplicates(["trade_date", "security_id"], keep="last")
    frame = frame.rename(columns={
        "open_brl": "open",
        "high_brl": "high",
        "low_brl": "low",
        "close_brl": "close",
    })
    frame["security_id"] = frame["security_id"].astype(str)
    frame["ticker"] = frame["ticker"].map(norm_text)
    frame["isin"] = frame["isin"].map(norm_text)
    return frame


def build_variants(xp_daily: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    groups: dict[tuple[str, str], pd.DataFrame] = {}
    for variant, prefix in (("all", "all"), ("pre_cutoff", "pre")):
        columns = [f"{prefix}_{field}" for field in ("open", "high", "low", "close", "quantity")]
        if not all(column in xp_daily.columns for column in columns):
            continue
        part = xp_daily[["xp_symbol", "source_file", "trade_date", *columns]].copy()
        part = part.rename(columns={f"{prefix}_{field}": field for field in ("open", "high", "low", "close", "quantity")})
        part = part.dropna(subset=["open", "high", "low", "close"])
        part["trade_date"] = pd.to_datetime(part["trade_date"]).dt.date
        part["xp_symbol"] = part["xp_symbol"].map(norm_text)
        for symbol, group in part.groupby("xp_symbol", sort=False):
            groups[(symbol, variant)] = group.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    if not groups:
        raise ValueError("No XP daily variants could be built")
    return groups


def pair_metrics(
    xp: pd.DataFrame,
    cot: pd.DataFrame,
    tick_size: float,
) -> dict[str, Any]:
    joined = xp.merge(cot, on="trade_date", how="inner", suffixes=("_xp", "_cot")).sort_values("trade_date")
    n = len(joined)
    cot_days = len(cot)
    if n == 0:
        return {
            "overlap_days": 0,
            "cot_days": cot_days,
            "overlap_coverage": 0.0,
        }

    xp_prices = joined[[f"{field}_xp" for field in ("open", "high", "low", "close")]].to_numpy(float)
    cot_prices = joined[[f"{field}_cot" for field in ("open", "high", "low", "close")]].to_numpy(float)
    valid_rows = np.all(np.isfinite(xp_prices) & np.isfinite(cot_prices) & (xp_prices > 0) & (cot_prices > 0), axis=1)
    xp_prices = xp_prices[valid_rows]
    cot_prices = cot_prices[valid_rows]
    joined_valid = joined.loc[valid_rows].copy()

    if len(joined_valid) == 0:
        return {
            "overlap_days": n,
            "cot_days": cot_days,
            "overlap_coverage": n / cot_days if cot_days else 0.0,
        }

    ratios = xp_prices / cot_prices
    daily_factor = np.nanmedian(ratios, axis=1)
    scaled_cot = cot_prices * daily_factor[:, None]

    effective_tick = tick_size if math.isfinite(tick_size) and tick_size > 0 else 0.01
    shape_error_ticks = np.nanmax(np.abs(xp_prices - scaled_cot) / effective_tick, axis=1)

    # Scale-invariant daily candle geometry. This remains comparable when the
    # broker back-adjusts historical prices by a multiplicative factor.
    xp_shape = np.log(xp_prices[:, :3] / xp_prices[:, [3]])
    cot_shape = np.log(cot_prices[:, :3] / cot_prices[:, [3]])
    shape_log_error = np.nanmax(np.abs(xp_shape - cot_shape), axis=1)

    xp_close = xp_prices[:, 3]
    cot_close = cot_prices[:, 3]
    close_ratio = xp_close / cot_close
    log_ratio = np.log(close_ratio)
    ratio_steps = np.diff(log_ratio)

    xp_returns = np.diff(np.log(xp_close))
    cot_returns = np.diff(np.log(cot_close))

    xp_qty = joined_valid["quantity_xp"].to_numpy(float)
    cot_qty = joined_valid["quantity_cot"].to_numpy(float)
    log_volume_corr = safe_corr(np.log1p(np.maximum(xp_qty, 0)), np.log1p(np.maximum(cot_qty, 0)))
    raw_return_corr = safe_corr(xp_returns, cot_returns)

    # Consecutive-date ratio stability is the adjustment-invariant return test:
    # for the same security, XP/COT price ratios are piecewise constant and only
    # jump on corporate-action adjustment dates.
    abs_steps = np.abs(ratio_steps[np.isfinite(ratio_steps)])
    if len(abs_steps):
        stable_10bp = float(np.mean(abs_steps <= 0.0010))
        stable_25bp = float(np.mean(abs_steps <= 0.0025))
        stable_50bp = float(np.mean(abs_steps <= 0.0050))
        step_p50 = float(np.nanmedian(abs_steps))
        step_p90 = float(np.nanquantile(abs_steps, 0.90))
        step_p95 = float(np.nanquantile(abs_steps, 0.95))
    else:
        stable_10bp = stable_25bp = stable_50bp = float("nan")
        step_p50 = step_p90 = step_p95 = float("nan")

    def q(values: np.ndarray, quantile: float) -> float:
        finite = values[np.isfinite(values)]
        return float(np.nanquantile(finite, quantile)) if len(finite) else float("nan")

    metrics = {
        "overlap_days": n,
        "valid_shape_days": len(joined_valid),
        "cot_days": cot_days,
        "overlap_coverage": n / cot_days if cot_days else 0.0,
        "first_overlap_date": str(joined["trade_date"].min()),
        "last_overlap_date": str(joined["trade_date"].max()),
        "shape_match_2ticks": float(np.mean(shape_error_ticks <= 2.000001)),
        "shape_match_5ticks": float(np.mean(shape_error_ticks <= 5.000001)),
        "shape_error_ticks_p50": q(shape_error_ticks, 0.50),
        "shape_error_ticks_p90": q(shape_error_ticks, 0.90),
        "shape_log_error_p50": q(shape_log_error, 0.50),
        "shape_log_error_p90": q(shape_log_error, 0.90),
        "shape_log_match_10bp": float(np.mean(shape_log_error <= 0.0010)),
        "shape_log_match_25bp": float(np.mean(shape_log_error <= 0.0025)),
        "ratio_stable_10bp": stable_10bp,
        "ratio_stable_25bp": stable_25bp,
        "ratio_stable_50bp": stable_50bp,
        "ratio_step_abs_p50": step_p50,
        "ratio_step_abs_p90": step_p90,
        "ratio_step_abs_p95": step_p95,
        "close_ratio_median": float(np.nanmedian(close_ratio)),
        "close_ratio_relative_mad": float(
            np.nanmedian(np.abs(close_ratio - np.nanmedian(close_ratio)))
            / max(abs(np.nanmedian(close_ratio)), 1e-12)
        ),
        "raw_return_correlation": raw_return_corr,
        "log_volume_correlation": log_volume_corr,
    }

    # A deliberately transparent score. It is used to rank missing-security
    # candidates, not to override exact ISIN evidence.
    def finite01(value: float, default: float = 0.0) -> float:
        return min(max(value, 0.0), 1.0) if math.isfinite(value) else default

    metrics["invariant_match_score"] = (
        0.30 * finite01(metrics["shape_match_5ticks"])
        + 0.20 * finite01(metrics["shape_log_match_25bp"])
        + 0.30 * finite01(metrics["ratio_stable_50bp"])
        + 0.10 * finite01(metrics["overlap_coverage"])
        + 0.07 * finite01(metrics["log_volume_correlation"])
        + 0.03 * finite01(metrics["raw_return_correlation"])
    )
    return metrics


def choose_best_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return sorted(
        rows,
        key=lambda row: (
            safe_float(row.get("invariant_match_score"), -1.0),
            safe_float(row.get("shape_match_5ticks"), -1.0),
            safe_float(row.get("ratio_stable_50bp"), -1.0),
            int(row.get("overlap_days", 0)),
        ),
        reverse=True,
    )[0]


def classify_missing(best: pd.Series, second_score: float, thresholds: RecoveryThresholds) -> str:
    overlap = int(best.get("overlap_days", 0))
    coverage = safe_float(best.get("overlap_coverage"), 0.0)
    shape = safe_float(best.get("shape_match_5ticks"), 0.0)
    stable = safe_float(best.get("ratio_stable_50bp"), 0.0)
    score = safe_float(best.get("invariant_match_score"), 0.0)
    margin = score - second_score if math.isfinite(second_score) else score

    if (
        overlap >= thresholds.min_overlap_days
        and coverage >= thresholds.high_overlap_coverage
        and shape >= thresholds.high_shape_match_5ticks
        and stable >= thresholds.high_ratio_stable_50bp
        and score >= thresholds.high_score
        and margin >= thresholds.high_margin
    ):
        return "RECOVERED_HIGH"
    if overlap >= max(8, thresholds.min_overlap_days // 2) and score >= thresholds.review_score:
        return "REVIEW"
    return "UNRESOLVED"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reconcile XP M1 daily aggregates to B3 permanent security IDs using "
            "exact XP-catalogue ISIN matches for direct assignments and "
            "scale-invariant candle/ratio fingerprints for historical predecessors."
        )
    )
    parser.add_argument("--universe-dir", type=Path, required=True)
    parser.add_argument("--daily-root", type=Path, required=True)
    parser.add_argument("--mapping-csv", type=Path, required=True)
    parser.add_argument("--xp-catalogue", type=Path, required=True)
    parser.add_argument("--previous-reconciliation-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", default="2021-07-19")
    parser.add_argument("--end", default="2026-07-17")
    parser.add_argument("--top-candidates", type=int, default=5)
    args = parser.parse_args()

    ensure_new_dir(args.out)
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    thresholds = RecoveryThresholds()

    union = pd.read_parquet(args.universe_dir / "universe_union.parquet")
    master = pd.read_parquet(args.universe_dir / "security_master.parquet")
    mapping = pd.read_csv(args.mapping_csv)
    catalogue = load_catalogue(args.xp_catalogue)
    xp_daily_path = args.previous_reconciliation_dir / "xp_daily_aggregates.parquet"
    if not xp_daily_path.exists():
        raise FileNotFoundError(f"Missing prior daily aggregate file: {xp_daily_path}")
    xp_daily = pd.read_parquet(xp_daily_path)
    cot = load_cotahist(args.daily_root, start, end)

    union["security_id"] = union["security_id"].astype(str)
    master["security_id"] = master["security_id"].astype(str)
    mapping["security_id"] = mapping["security_id"].astype(str)
    mapping["preferred_xp_symbol"] = mapping["preferred_xp_symbol"].map(norm_text)
    mapping["isin"] = mapping["isin"].map(norm_text)

    union_ids = set(union["security_id"])
    cot_union = cot[cot["security_id"].isin(union_ids)].copy()
    cot_groups = {
        str(sid): group.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
        for sid, group in cot_union.groupby("security_id", sort=False)
    }
    xp_groups = build_variants(xp_daily)
    xp_symbols = sorted({symbol for symbol, _ in xp_groups})

    catalogue_by_symbol = catalogue.set_index("name", drop=False).to_dict("index")
    source_by_symbol = (
        xp_daily[["xp_symbol", "source_file"]]
        .drop_duplicates("xp_symbol", keep="last")
        .assign(xp_symbol=lambda frame: frame["xp_symbol"].map(norm_text))
        .set_index("xp_symbol")["source_file"]
        .to_dict()
    )

    mapping = mapping.merge(
        catalogue.rename(columns={
            "name": "catalogue_symbol",
            "isin": "catalogue_isin",
            "description": "catalogue_description",
            "path": "catalogue_path",
            "trade_tick_size": "catalogue_tick_size",
        }),
        left_on="preferred_xp_symbol",
        right_on="catalogue_symbol",
        how="left",
    )
    mapping["catalogue_isin"] = mapping["catalogue_isin"].map(norm_text)
    mapping["direct_isin_match"] = (
        (mapping["preferred_xp_symbol"] != "")
        & (mapping["isin"] != "")
        & (mapping["isin"] == mapping["catalogue_isin"])
        & mapping["preferred_xp_symbol"].isin(xp_symbols)
    )

    direct_rows: list[dict[str, Any]] = []
    direct_variant_rows: list[dict[str, Any]] = []
    for row in mapping[mapping["direct_isin_match"]].itertuples(index=False):
        sid = str(row.security_id)
        symbol = norm_text(row.preferred_xp_symbol)
        cot_group = cot_groups.get(sid)
        if cot_group is None or cot_group.empty:
            continue
        tick_size = safe_float(getattr(row, "catalogue_tick_size", np.nan), 0.01)
        variant_metrics: list[dict[str, Any]] = []
        for variant in ("all", "pre_cutoff"):
            xp_group = xp_groups.get((symbol, variant))
            if xp_group is None:
                continue
            metrics = pair_metrics(
                xp_group[["trade_date", "open", "high", "low", "close", "quantity"]],
                cot_group[["trade_date", "open", "high", "low", "close", "quantity"]],
                tick_size,
            )
            candidate = {
                "security_id": sid,
                "isin": norm_text(row.isin),
                "latest_ticker": norm_text(row.latest_ticker),
                "xp_symbol": symbol,
                "variant": variant,
                "source_file": source_by_symbol.get(symbol, ""),
                "catalogue_isin": norm_text(row.catalogue_isin),
                "catalogue_description": getattr(row, "catalogue_description", ""),
                "catalogue_tick_size": tick_size,
                **metrics,
            }
            variant_metrics.append(candidate)
            direct_variant_rows.append(candidate)
        best = choose_best_variant(variant_metrics)
        if not best:
            continue
        best["assignment_status"] = "DIRECT_ISIN_CONFIRMED"
        best["quality_flag"] = (
            "HIGH"
            if safe_float(best.get("shape_match_5ticks"), 0.0) >= 0.80
            and safe_float(best.get("ratio_stable_50bp"), 0.0) >= 0.80
            else "REVIEW"
        )
        direct_rows.append(best)

    direct_assignments = pd.DataFrame(direct_rows)
    direct_variants = pd.DataFrame(direct_variant_rows)
    write_frame(direct_assignments, args.out / "direct_isin_assignments")
    write_frame(direct_variants, args.out / "direct_variant_quality")

    missing_mapping = mapping[~mapping["direct_isin_match"]].copy()
    missing_ids = [sid for sid in missing_mapping["security_id"].astype(str) if sid in union_ids]

    candidate_rows: list[dict[str, Any]] = []
    no_research_rows: list[dict[str, Any]] = []
    for index, sid in enumerate(missing_ids, start=1):
        cot_group = cot_groups.get(sid)
        map_row = missing_mapping[missing_mapping["security_id"] == sid].iloc[0]
        if cot_group is None or cot_group.empty:
            no_research_rows.append({
                "security_id": sid,
                "latest_ticker": norm_text(map_row.get("latest_ticker", "")),
                "reason": "NO_COTAHIST_DAYS_IN_RESEARCH_INTERVAL",
            })
            continue
        print(
            f"Scanning missing security {index}/{len(missing_ids)}: "
            f"{norm_text(map_row.get('latest_ticker', ''))} ({len(cot_group)} COTAHIST days)",
            flush=True,
        )
        for symbol in xp_symbols:
            catalogue_row = catalogue_by_symbol.get(symbol, {})
            tick_size = safe_float(catalogue_row.get("trade_tick_size"), 0.01)
            best_symbol_variant_rows: list[dict[str, Any]] = []
            for variant in ("all", "pre_cutoff"):
                xp_group = xp_groups.get((symbol, variant))
                if xp_group is None:
                    continue
                # Cheap date-overlap prefilter.
                xp_first = xp_group["trade_date"].min()
                xp_last = xp_group["trade_date"].max()
                cot_first = cot_group["trade_date"].min()
                cot_last = cot_group["trade_date"].max()
                if xp_last < cot_first or xp_first > cot_last:
                    continue
                metrics = pair_metrics(
                    xp_group[["trade_date", "open", "high", "low", "close", "quantity"]],
                    cot_group[["trade_date", "open", "high", "low", "close", "quantity"]],
                    tick_size,
                )
                if int(metrics.get("overlap_days", 0)) < 3:
                    continue
                best_symbol_variant_rows.append({
                    "security_id": sid,
                    "isin": norm_text(map_row.get("isin", "")),
                    "latest_ticker": norm_text(map_row.get("latest_ticker", "")),
                    "latest_issuer": map_row.get("latest_issuer", ""),
                    "membership_months": int(map_row.get("membership_months", 0) or 0),
                    "xp_symbol": symbol,
                    "variant": variant,
                    "source_file": source_by_symbol.get(symbol, ""),
                    "xp_catalogue_isin": norm_text(catalogue_row.get("isin", "")),
                    "xp_description": catalogue_row.get("description", ""),
                    "catalogue_tick_size": tick_size,
                    **metrics,
                })
            best_for_symbol = choose_best_variant(best_symbol_variant_rows)
            if best_for_symbol:
                candidate_rows.append(best_for_symbol)

    candidate_scores = pd.DataFrame(candidate_rows)
    if not candidate_scores.empty:
        candidate_scores = candidate_scores.sort_values(
            ["security_id", "invariant_match_score", "overlap_days"],
            ascending=[True, False, False],
        )
        candidate_scores["candidate_rank"] = candidate_scores.groupby("security_id").cumcount() + 1
        top_scores = candidate_scores[candidate_scores["candidate_rank"] <= args.top_candidates].copy()
    else:
        top_scores = pd.DataFrame()
    write_frame(candidate_scores, args.out / "missing_all_candidate_scores")
    write_frame(top_scores, args.out / "missing_top_candidate_scores")

    best_rows: list[dict[str, Any]] = []
    if not candidate_scores.empty:
        for sid, group in candidate_scores.groupby("security_id", sort=False):
            ordered = group.sort_values(["invariant_match_score", "overlap_days"], ascending=[False, False])
            best = ordered.iloc[0].copy()
            second_score = safe_float(ordered.iloc[1]["invariant_match_score"], float("nan")) if len(ordered) > 1 else float("nan")
            best["second_best_score"] = second_score
            best["score_margin"] = safe_float(best["invariant_match_score"], 0.0) - second_score if math.isfinite(second_score) else safe_float(best["invariant_match_score"], 0.0)
            best["recovery_status"] = classify_missing(best, second_score, thresholds)
            best_rows.append(best.to_dict())
    missing_best = pd.DataFrame(best_rows)
    write_frame(missing_best, args.out / "missing_best_candidates")

    high = missing_best[missing_best.get("recovery_status", pd.Series(dtype=str)) == "RECOVERED_HIGH"].copy() if not missing_best.empty else pd.DataFrame()
    review = missing_best[missing_best.get("recovery_status", pd.Series(dtype=str)) == "REVIEW"].copy() if not missing_best.empty else pd.DataFrame()
    unresolved = missing_best[missing_best.get("recovery_status", pd.Series(dtype=str)) == "UNRESOLVED"].copy() if not missing_best.empty else pd.DataFrame()
    no_research = pd.DataFrame(no_research_rows)
    write_frame(high, args.out / "missing_high_confidence_recoveries")
    write_frame(review, args.out / "missing_review")
    write_frame(unresolved, args.out / "missing_unresolved")
    write_frame(no_research, args.out / "missing_no_research_days")

    final_assignments = pd.concat(
        [
            direct_assignments.assign(source_assignment_type="DIRECT_ISIN") if not direct_assignments.empty else pd.DataFrame(),
            high.assign(source_assignment_type="RECOVERED_RELABELED") if not high.empty else pd.DataFrame(),
        ],
        ignore_index=True,
        sort=False,
    )
    write_frame(final_assignments, args.out / "provisional_source_assignments")

    summary = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": iso_now(),
        "universe_dir": str(args.universe_dir.resolve()),
        "daily_root": str(args.daily_root.resolve()),
        "mapping_csv": str(args.mapping_csv.resolve()),
        "xp_catalogue": str(args.xp_catalogue.resolve()),
        "previous_reconciliation_dir": str(args.previous_reconciliation_dir.resolve()),
        "union_security_count": int(len(union)),
        "xp_daily_symbol_count": int(len(xp_symbols)),
        "direct_isin_confirmed_count": int(len(direct_assignments)),
        "direct_quality_high_count": int((direct_assignments.get("quality_flag", pd.Series(dtype=str)) == "HIGH").sum()) if not direct_assignments.empty else 0,
        "direct_quality_review_count": int((direct_assignments.get("quality_flag", pd.Series(dtype=str)) == "REVIEW").sum()) if not direct_assignments.empty else 0,
        "missing_security_count": int(len(missing_ids)),
        "missing_no_research_days_count": int(len(no_research)),
        "missing_recovered_high_count": int(len(high)),
        "missing_review_count": int(len(review)),
        "missing_unresolved_count": int(len(unresolved)),
        "provisional_assignment_count": int(len(final_assignments)),
        "thresholds": thresholds.__dict__,
        "notes": [
            "Direct assignments are based on exact XP-catalogue ISIN equality, not raw price equality.",
            "Missing-security candidates are ranked using scale-invariant candle geometry and piecewise XP/COTAHIST price-ratio stability.",
            "The output is provisional; recovered historical predecessors should be inspected before normalizing M1 files by security_id.",
        ],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"Complete: {args.out}")
    print(f"Direct ISIN assignments: {len(direct_assignments):,}")
    print(f"Missing high-confidence recoveries: {len(high):,}")
    print(f"Missing review cases: {len(review):,}")
    print(f"Missing unresolved: {len(unresolved):,}")
    print(f"Missing with no COTAHIST days in research: {len(no_research):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
