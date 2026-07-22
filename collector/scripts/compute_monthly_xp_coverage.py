from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

SCRIPT_VERSION = "1"


def read_table(directory: Path, stem: str) -> pl.DataFrame:
    parquet = directory / f"{stem}.parquet"
    csv = directory / f"{stem}.csv"
    if parquet.exists():
        return pl.read_parquet(parquet)
    if csv.exists():
        return pl.read_csv(csv, try_parse_dates=True, infer_schema_length=10000)
    raise FileNotFoundError(f"Could not find {parquet} or {csv}")


def scalar(frame: pl.DataFrame, expression: pl.Expr) -> Any:
    return frame.select(expression).item()


def finite_or_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
    return value


def coverage_stats(monthly: pl.DataFrame, column: str) -> dict[str, Any]:
    ordered = monthly.sort("effective_from")
    min_row = ordered.sort(column).row(0, named=True)
    max_row = ordered.sort(column, descending=True).row(0, named=True)
    return {
        "min": finite_or_none(float(min_row[column])),
        "min_effective_from": str(min_row["effective_from"]),
        "median": finite_or_none(float(scalar(ordered, pl.col(column).median()))),
        "mean": finite_or_none(float(scalar(ordered, pl.col(column).mean()))),
        "max": finite_or_none(float(max_row[column])),
        "max_effective_from": str(max_row["effective_from"]),
        "latest": finite_or_none(float(ordered[column][-1])),
        "latest_effective_from": str(ordered["effective_from"][-1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute monthly XP M1 coverage of a point-in-time universe by member count, "
            "trailing turnover, and parent-market share."
        )
    )
    parser.add_argument("--universe-dir", type=Path, required=True)
    parser.add_argument("--accepted-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--low-count-threshold",
        type=float,
        default=0.90,
        help="Flag months below this member-count coverage ratio.",
    )
    parser.add_argument(
        "--low-turnover-threshold",
        type=float,
        default=0.95,
        help="Flag months below this turnover coverage ratio.",
    )
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(f"Output directory already exists: {args.out}")
    args.out.mkdir(parents=True, exist_ok=False)

    membership = read_table(args.universe_dir, "universe_membership_monthly")
    accepted = read_table(args.accepted_dir, "xp_accepted_source_assignments_v1")

    required_membership = {
        "review_date",
        "effective_from",
        "security_id",
        "ticker_asof",
        "issuer_short_name_asof",
        "turnover_brl",
        "market_share",
    }
    missing_membership = sorted(required_membership - set(membership.columns))
    if missing_membership:
        raise ValueError(f"Membership table is missing columns: {missing_membership}")

    required_accepted = {"security_id", "source_assignment_type"}
    missing_accepted = sorted(required_accepted - set(accepted.columns))
    if missing_accepted:
        raise ValueError(f"Accepted assignment table is missing columns: {missing_accepted}")

    membership = membership.with_columns(
        pl.col("security_id").cast(pl.Utf8).str.strip_chars(),
        pl.col("turnover_brl").cast(pl.Float64),
        pl.col("market_share").cast(pl.Float64),
        pl.col("review_date").cast(pl.Date),
        pl.col("effective_from").cast(pl.Date),
    )
    if "effective_to_exclusive" in membership.columns:
        membership = membership.with_columns(
            pl.col("effective_to_exclusive").cast(pl.Date)
        )

    accepted = accepted.select(
        pl.col("security_id").cast(pl.Utf8).str.strip_chars(),
        pl.col("source_assignment_type")
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.to_uppercase(),
        *(
            [pl.col("xp_symbol").cast(pl.Utf8).str.strip_chars()]
            if "xp_symbol" in accepted.columns
            else []
        ),
    )

    duplicate_accepted = accepted.group_by("security_id").len().filter(pl.col("len") > 1)
    if duplicate_accepted.height:
        raise ValueError(
            "Accepted assignments are not unique by security_id: "
            f"{duplicate_accepted.head(20).to_dicts()}"
        )

    accepted = accepted.with_columns(
        pl.lit(True).alias("xp_covered"),
        (pl.col("source_assignment_type") == "DIRECT_ISIN").alias("xp_direct_isin"),
        pl.col("source_assignment_type")
        .is_in(["RECOVERED_RELABELED", "RECOVERED_RELABELLED"])
        .alias("xp_recovered_relabelled"),
    )

    detailed = (
        membership.join(accepted, on="security_id", how="left")
        .with_columns(
            pl.col("xp_covered").fill_null(False),
            pl.col("xp_direct_isin").fill_null(False),
            pl.col("xp_recovered_relabelled").fill_null(False),
            pl.col("source_assignment_type").fill_null("UNRESOLVED"),
        )
        .sort(["effective_from", "security_id"])
    )

    group_keys = ["review_date", "effective_from"]
    if "effective_to_exclusive" in detailed.columns:
        group_keys.append("effective_to_exclusive")

    monthly = (
        detailed.group_by(group_keys, maintain_order=True)
        .agg(
            pl.len().alias("member_count"),
            pl.col("xp_covered").sum().cast(pl.Int64).alias("xp_member_count"),
            (~pl.col("xp_covered")).sum().cast(pl.Int64).alias(
                "unresolved_member_count"
            ),
            pl.col("xp_direct_isin").sum().cast(pl.Int64).alias(
                "xp_direct_member_count"
            ),
            pl.col("xp_recovered_relabelled").sum().cast(pl.Int64).alias(
                "xp_recovered_member_count"
            ),
            pl.col("turnover_brl").sum().alias("member_turnover_brl"),
            pl.when(pl.col("xp_covered"))
            .then(pl.col("turnover_brl"))
            .otherwise(0.0)
            .sum()
            .alias("xp_member_turnover_brl"),
            pl.when(~pl.col("xp_covered"))
            .then(pl.col("turnover_brl"))
            .otherwise(0.0)
            .sum()
            .alias("unresolved_member_turnover_brl"),
            pl.when(pl.col("xp_direct_isin"))
            .then(pl.col("turnover_brl"))
            .otherwise(0.0)
            .sum()
            .alias("xp_direct_turnover_brl"),
            pl.when(pl.col("xp_recovered_relabelled"))
            .then(pl.col("turnover_brl"))
            .otherwise(0.0)
            .sum()
            .alias("xp_recovered_turnover_brl"),
            pl.col("market_share").sum().alias("member_market_share_sum"),
            pl.when(pl.col("xp_covered"))
            .then(pl.col("market_share"))
            .otherwise(0.0)
            .sum()
            .alias("xp_member_market_share_sum"),
            pl.when(~pl.col("xp_covered"))
            .then(pl.col("market_share"))
            .otherwise(0.0)
            .sum()
            .alias("unresolved_member_market_share_sum"),
        )
        .with_columns(
            (pl.col("xp_member_count") / pl.col("member_count")).alias(
                "member_count_coverage"
            ),
            pl.when(pl.col("member_turnover_brl") > 0)
            .then(pl.col("xp_member_turnover_brl") / pl.col("member_turnover_brl"))
            .otherwise(None)
            .alias("turnover_coverage"),
            pl.when(pl.col("member_market_share_sum") > 0)
            .then(
                pl.col("xp_member_market_share_sum")
                / pl.col("member_market_share_sum")
            )
            .otherwise(None)
            .alias("member_market_share_coverage"),
        )
        .with_columns(
            (pl.col("member_count_coverage") < args.low_count_threshold).alias(
                "low_member_count_coverage"
            ),
            (pl.col("turnover_coverage") < args.low_turnover_threshold).alias(
                "low_turnover_coverage"
            ),
        )
        .sort("effective_from")
    )

    missing_detail = (
        detailed.filter(~pl.col("xp_covered"))
        .select(
            *group_keys,
            "security_id",
            "ticker_asof",
            "issuer_short_name_asof",
            "turnover_brl",
            "market_share",
            "presence_ratio",
            "last_close_brl",
            "source_assignment_type",
        )
        .sort(["effective_from", "turnover_brl"], descending=[False, True])
    )

    lowest = monthly.sort(
        ["turnover_coverage", "member_count_coverage"]
    ).head(min(20, monthly.height))

    total_membership_rows = detailed.height
    covered_membership_rows = int(scalar(detailed, pl.col("xp_covered").sum()))
    total_turnover = float(scalar(detailed, pl.col("turnover_brl").sum()))
    covered_turnover = float(
        scalar(
            detailed,
            pl.when(pl.col("xp_covered"))
            .then(pl.col("turnover_brl"))
            .otherwise(0.0)
            .sum(),
        )
    )

    summary = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "universe_dir": str(args.universe_dir),
        "accepted_dir": str(args.accepted_dir),
        "output_dir": str(args.out),
        "accepted_security_ids": accepted.height,
        "universe_membership_rows": total_membership_rows,
        "covered_membership_rows": covered_membership_rows,
        "membership_row_coverage": (
            covered_membership_rows / total_membership_rows
            if total_membership_rows
            else None
        ),
        "aggregate_member_turnover_brl": total_turnover,
        "aggregate_xp_member_turnover_brl": covered_turnover,
        "aggregate_turnover_coverage": (
            covered_turnover / total_turnover if total_turnover else None
        ),
        "months": monthly.height,
        "member_count_coverage": coverage_stats(monthly, "member_count_coverage"),
        "turnover_coverage": coverage_stats(monthly, "turnover_coverage"),
        "member_market_share_coverage": coverage_stats(
            monthly, "member_market_share_coverage"
        ),
        "months_below_member_count_threshold": int(
            scalar(monthly, pl.col("low_member_count_coverage").sum())
        ),
        "member_count_threshold": args.low_count_threshold,
        "months_below_turnover_threshold": int(
            scalar(monthly, pl.col("low_turnover_coverage").sum())
        ),
        "turnover_threshold": args.low_turnover_threshold,
    }

    monthly.write_parquet(
        args.out / "monthly_xp_coverage.parquet",
        compression="zstd",
        statistics=True,
    )
    monthly.write_csv(args.out / "monthly_xp_coverage.csv")
    missing_detail.write_parquet(
        args.out / "unresolved_members_monthly.parquet",
        compression="zstd",
        statistics=True,
    )
    missing_detail.write_csv(args.out / "unresolved_members_monthly.csv")
    lowest.write_csv(args.out / "lowest_coverage_months.csv")
    detailed.write_parquet(
        args.out / "membership_with_xp_coverage.parquet",
        compression="zstd",
        statistics=True,
    )
    (args.out / "coverage_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Complete: {args.out}")
    print(f"Months: {monthly.height}")
    print(
        "Membership-row coverage: "
        f"{summary['membership_row_coverage']:.2%} "
        f"({covered_membership_rows:,}/{total_membership_rows:,})"
    )
    print(f"Aggregate turnover coverage: {summary['aggregate_turnover_coverage']:.2%}")
    print(
        "Minimum monthly count coverage: "
        f"{summary['member_count_coverage']['min']:.2%} "
        f"on {summary['member_count_coverage']['min_effective_from']}"
    )
    print(
        "Minimum monthly turnover coverage: "
        f"{summary['turnover_coverage']['min']:.2%} "
        f"on {summary['turnover_coverage']['min_effective_from']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
