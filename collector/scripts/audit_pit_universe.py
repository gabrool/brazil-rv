from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

SCRIPT_VERSION = "1"


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    details: str


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def bool_expr(col: str) -> pl.Expr:
    # Parquet outputs preserve booleans; this also tolerates text if a future
    # version writes them differently.
    dtype = None
    return pl.when(pl.col(col).cast(pl.String).str.to_lowercase().is_in(["true", "1"])) \
        .then(True).otherwise(False)


def compress_membership_intervals(membership: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for group in membership.sort(["security_id", "effective_from"]).partition_by(
        "security_id", maintain_order=True
    ):
        data = group.to_dicts()
        if not data:
            continue
        start = data[0]["effective_from"]
        end = data[0]["effective_to_exclusive"]
        review_start = data[0]["review_date"]
        review_end = data[0]["review_date"]
        tickers: set[str] = {str(data[0].get("ticker_asof") or "")}
        months = 1
        shares = [float(data[0].get("market_share") or 0.0)]
        presences = [float(data[0].get("presence_ratio") or 0.0)]
        prior_end = end

        def emit() -> None:
            nonlocal start, end, review_start, review_end, tickers, months, shares, presences
            rows.append(
                {
                    "security_id": data[0]["security_id"],
                    "effective_from": start,
                    "effective_to_exclusive": end,
                    "first_review_date": review_start,
                    "last_review_date": review_end,
                    "membership_months": months,
                    "tickers_asof": "|".join(sorted(t for t in tickers if t)),
                    "median_market_share": float(sorted(shares)[len(shares) // 2]),
                    "min_market_share": min(shares),
                    "median_presence_ratio": float(sorted(presences)[len(presences) // 2]),
                    "min_presence_ratio": min(presences),
                }
            )

        for current in data[1:]:
            current_start = current["effective_from"]
            contiguous = prior_end == current_start
            if not contiguous:
                emit()
                start = current_start
                review_start = current["review_date"]
                tickers = set()
                months = 0
                shares = []
                presences = []
            end = current["effective_to_exclusive"]
            review_end = current["review_date"]
            prior_end = end
            tickers.add(str(current.get("ticker_asof") or ""))
            months += 1
            shares.append(float(current.get("market_share") or 0.0))
            presences.append(float(current.get("presence_ratio") or 0.0))
        emit()

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows, infer_schema_length=None).sort(
        ["effective_from", "security_id"]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a point-in-time B3 universe.")
    parser.add_argument("--universe-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--low-market-share-threshold", type=float, default=0.93)
    parser.add_argument("--top-n-nonmembers", type=int, default=25)
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(f"Output directory already exists: {args.out}")

    universe_dir = args.universe_dir.resolve()
    files = {
        "metrics": require(universe_dir / "universe_metrics_monthly.parquet"),
        "membership": require(universe_dir / "universe_membership_monthly.parquet"),
        "changes": require(universe_dir / "universe_changes.parquet"),
        "union": require(universe_dir / "universe_union.parquet"),
        "master": require(universe_dir / "security_master.parquet"),
        "ticker_history": require(universe_dir / "ticker_history.parquet"),
        "summary": require(universe_dir / "universe_summary.parquet"),
        "manifest": require(universe_dir / "manifest.json"),
    }

    metrics = pl.read_parquet(files["metrics"])
    membership = pl.read_parquet(files["membership"])
    changes = pl.read_parquet(files["changes"])
    union = pl.read_parquet(files["union"])
    master = pl.read_parquet(files["master"])
    ticker_history = pl.read_parquet(files["ticker_history"])
    summary = pl.read_parquet(files["summary"])
    manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))

    args.out.mkdir(parents=True, exist_ok=False)

    checks: list[Check] = []

    def add_check(name: str, passed: bool, details: str) -> None:
        checks.append(Check(name, bool(passed), details))

    add_check(
        "master_security_id_unique",
        master["security_id"].n_unique() == master.height,
        f"rows={master.height}, unique={master['security_id'].n_unique()}",
    )
    add_check(
        "union_security_id_unique",
        union["security_id"].n_unique() == union.height,
        f"rows={union.height}, unique={union['security_id'].n_unique()}",
    )
    metric_dup = metrics.height - metrics.select(
        pl.struct(["effective_from", "security_id"]).n_unique()
    ).item()
    member_dup = membership.height - membership.select(
        pl.struct(["effective_from", "security_id"]).n_unique()
    ).item()
    add_check("metrics_key_unique", metric_dup == 0, f"duplicates={metric_dup}")
    add_check("membership_key_unique", member_dup == 0, f"duplicates={member_dup}")

    master_ids = set(master["security_id"].to_list())
    union_ids = set(union["security_id"].to_list())
    member_ids = set(membership["security_id"].unique().to_list())
    add_check(
        "union_subset_of_master",
        union_ids.issubset(master_ids),
        f"missing={len(union_ids-master_ids)}",
    )
    add_check(
        "membership_matches_union",
        member_ids == union_ids,
        f"membership_only={len(member_ids-union_ids)}, union_only={len(union_ids-member_ids)}",
    )

    # Summary/member reconciliation.
    recomputed_summary = (
        membership.group_by(["review_date", "effective_from"])
        .agg(
            pl.len().alias("member_count_recomputed"),
            pl.col("market_share").sum().alias("member_market_share_sum_recomputed"),
            pl.col("turnover_brl").sum().alias("member_turnover_brl_recomputed"),
        )
        .sort("effective_from")
    )
    summary_check = summary.join(
        recomputed_summary, on=["review_date", "effective_from"], how="full", coalesce=True
    ).with_columns(
        (pl.col("member_count") == pl.col("member_count_recomputed")).alias("count_match"),
        ((pl.col("member_market_share_sum") - pl.col("member_market_share_sum_recomputed")).abs() < 1e-12).alias("share_match"),
    )
    add_check(
        "summary_reconciles_to_membership",
        bool(summary_check.select((pl.col("count_match") & pl.col("share_match")).all()).item()),
        f"rows={summary_check.height}",
    )

    latest_count = int(summary.sort("effective_from")["member_count"][-1])
    add_check(
        "latest_count_matches_manifest",
        latest_count == int(manifest.get("latest_member_count", -1)),
        f"summary={latest_count}, manifest={manifest.get('latest_member_count')}",
    )
    add_check(
        "union_count_matches_manifest",
        union.height == int(manifest.get("union_security_count", -1)),
        f"union={union.height}, manifest={manifest.get('union_security_count')}",
    )
    add_check(
        "parent_count_matches_manifest",
        master.height == int(manifest.get("parent_security_count", -1)),
        f"master={master.height}, manifest={manifest.get('parent_security_count')}",
    )

    # Ticker history segment overlaps.
    overlap_rows: list[dict[str, Any]] = []
    for group in ticker_history.sort(["security_id", "valid_from"]).partition_by(
        "security_id", maintain_order=True
    ):
        data = group.to_dicts()
        for prior, current in zip(data, data[1:]):
            if current["valid_from"] <= prior["valid_to"]:
                overlap_rows.append(
                    {
                        "security_id": prior["security_id"],
                        "prior_ticker": prior["ticker"],
                        "prior_valid_from": prior["valid_from"],
                        "prior_valid_to": prior["valid_to"],
                        "current_ticker": current["ticker"],
                        "current_valid_from": current["valid_from"],
                        "current_valid_to": current["valid_to"],
                    }
                )
    add_check(
        "ticker_segments_non_overlapping",
        not overlap_rows,
        f"overlaps={len(overlap_rows)}",
    )

    # Monthly diagnostics.
    action_counts = (
        changes.group_by(["review_date", "effective_from", "action"])
        .agg(pl.len().alias("count"))
        .pivot(on="action", index=["review_date", "effective_from"], values="count")
        .fill_null(0)
    )
    monthly = summary.join(action_counts, on=["review_date", "effective_from"], how="left").fill_null(0)
    for col in ["ENTER", "EXIT"]:
        if col not in monthly.columns:
            monthly = monthly.with_columns(pl.lit(0).alias(col))
    monthly = monthly.with_columns(
        (1.0 - pl.col("member_market_share_sum")).alias("outside_market_share"),
        (pl.col("ENTER") + pl.col("EXIT")).alias("gross_churn_count"),
        ((pl.col("ENTER") + pl.col("EXIT")) / pl.col("member_count")).alias("gross_churn_rate"),
        (pl.col("member_market_share_sum") < args.low_market_share_threshold).alias("low_market_share_flag"),
    ).sort("effective_from")

    low_months = monthly.filter(pl.col("low_market_share_flag"))
    top_nonmembers: list[pl.DataFrame] = []
    for row in low_months.select(["review_date", "effective_from"]).iter_rows(named=True):
        frame = (
            metrics.filter(
                (pl.col("effective_from") == row["effective_from"])
                & (~pl.col("is_member"))
                & (pl.col("market_share") > 0)
            )
            .sort("market_share", descending=True)
            .head(args.top_n_nonmembers)
            .select(
                "review_date",
                "effective_from",
                "security_id",
                "ticker_asof",
                "issuer_short_name_asof",
                "age_sessions",
                "presence_ratio",
                "market_share",
                "last_close_brl",
                "was_member",
                "entry_pass",
                "retention_pass",
                "action",
            )
        )
        if frame.height:
            top_nonmembers.append(frame)
    top_nonmembers_df = (
        pl.concat(top_nonmembers, how="vertical_relaxed") if top_nonmembers else pl.DataFrame()
    )

    fallback = master.filter(pl.col("security_id_is_fallback"))
    multi_ticker = master.filter(pl.col("distinct_tickers") > 1)
    latest_effective = membership["effective_from"].max()
    latest_members = membership.filter(pl.col("effective_from") == latest_effective).sort(
        "market_share", descending=True
    )
    intervals = compress_membership_intervals(membership)

    # A useful status table for all union securities.
    latest_observed = master.select(
        "security_id",
        "isin",
        "latest_ticker",
        "latest_issuer_short_name",
        "first_observed_date",
        "last_observed_date",
        "observed_sessions",
        "distinct_tickers",
        "tickers",
        "security_id_is_fallback",
    )
    union_status = union.join(latest_observed, on="security_id", how="left", suffix="_master").with_columns(
        (pl.col("last_observed_date") == master["last_observed_date"].max()).alias("observed_on_dataset_last_date")
    )

    # Write outputs.
    def write(name: str, frame: pl.DataFrame) -> None:
        frame.write_parquet(args.out / f"{name}.parquet", compression="zstd", statistics=True)
        csv_frame = frame
        for col, dtype in zip(csv_frame.columns, csv_frame.dtypes):
            if isinstance(dtype, pl.List):
                csv_frame = csv_frame.with_columns(pl.col(col).list.join("|").alias(col))
        csv_frame.write_csv(args.out / f"{name}.csv")

    write("monthly_diagnostics", monthly)
    write("low_market_share_months", low_months)
    if top_nonmembers_df.height:
        write("top_nonmembers_low_share_months", top_nonmembers_df)
    else:
        pl.DataFrame({"note": ["No low-market-share months"]}).write_csv(
            args.out / "top_nonmembers_low_share_months.csv"
        )
    write("fallback_security_ids", fallback)
    write("multi_ticker_securities", multi_ticker)
    write("latest_members", latest_members)
    write("membership_intervals", intervals)
    write("union_status", union_status)
    if overlap_rows:
        write("ticker_segment_overlaps", pl.DataFrame(overlap_rows, infer_schema_length=None))

    check_df = pl.DataFrame(
        [{"check": c.name, "passed": c.passed, "details": c.details} for c in checks]
    )
    check_df.write_csv(args.out / "integrity_checks.csv")

    summary_payload = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "universe_dir": str(universe_dir),
        "output_dir": str(args.out.resolve()),
        "manifest": manifest,
        "counts": {
            "parent_securities": master.height,
            "union_securities": union.height,
            "latest_members": latest_members.height,
            "review_dates": summary.height,
            "membership_rows": membership.height,
            "membership_intervals": intervals.height,
            "fallback_security_ids": fallback.height,
            "multi_ticker_securities": multi_ticker.height,
            "low_market_share_months": low_months.height,
        },
        "ranges": {
            "member_count_min": int(summary["member_count"].min()),
            "member_count_max": int(summary["member_count"].max()),
            "member_market_share_min": float(summary["member_market_share_sum"].min()),
            "member_market_share_max": float(summary["member_market_share_sum"].max()),
        },
        "all_integrity_checks_passed": all(c.passed for c in checks),
    }
    (args.out / "audit_summary.json").write_text(
        json.dumps(summary_payload, indent=2, default=str), encoding="utf-8"
    )

    print(f"Audit complete: {args.out}")
    print(f"Integrity checks passed: {sum(c.passed for c in checks)}/{len(checks)}")
    print(f"Low-market-share months: {low_months.height}")
    print(f"Fallback security IDs: {fallback.height}")
    print(f"Multi-ticker securities: {multi_ticker.height}")
    return 0 if all(c.passed for c in checks) else 2


if __name__ == "__main__":
    raise SystemExit(main())
