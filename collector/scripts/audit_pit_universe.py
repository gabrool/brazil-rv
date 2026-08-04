from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from build_point_in_time_universe import (
    ACCEPTED_FILENAME,
    load_parquets,
    sha256_file,
)
from pit_universe import (
    ACCEPTED_ASSIGNMENT_COUNT,
    CANONICAL_CONFIG,
    ELIGIBILITY_COLUMNS,
    MIN_RESEARCH_CROSS_SECTION,
    _review_records,
    build_universe_tables,
    eligibility_result,
    valid_observation_expr,
    validate_accepted_assignments,
)

SCRIPT_VERSION = "2"
TABLE_NAMES = (
    "universe_metrics_monthly",
    "universe_membership_monthly",
    "universe_changes",
    "universe_union",
    "security_master",
    "ticker_history",
    "universe_summary",
)


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    details: str


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def frames_equal(left: pl.DataFrame, right: pl.DataFrame) -> bool:
    return left.schema == right.schema and left.equals(right, null_equal=True)


def compress_membership_intervals(membership: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for group in membership.sort(["security_id", "effective_from"]).partition_by(
        "security_id", maintain_order=True
    ):
        data = group.to_dicts()
        start = data[0]["effective_from"]
        end = data[0]["effective_to_exclusive"]
        review_start = data[0]["review_date"]
        review_end = data[0]["review_date"]
        months = 1
        tickers = {str(data[0]["ticker_asof"] or "")}
        turnover = [float(data[0]["median_daily_turnover_brl"])]
        trades = [float(data[0]["median_daily_trade_count"])]

        def emit() -> None:
            rows.append(
                {
                    "security_id": data[0]["security_id"],
                    "effective_from": start,
                    "effective_to_exclusive": end,
                    "first_review_date": review_start,
                    "last_review_date": review_end,
                    "membership_months": months,
                    "tickers_asof": "|".join(
                        sorted(ticker for ticker in tickers if ticker)
                    ),
                    "median_daily_turnover_brl": float(
                        sorted(turnover)[len(turnover) // 2]
                    ),
                    "minimum_daily_turnover_brl": min(turnover),
                    "median_daily_trade_count": float(sorted(trades)[len(trades) // 2]),
                    "minimum_daily_trade_count": min(trades),
                }
            )

        prior_end = end
        for current in data[1:]:
            if prior_end != current["effective_from"]:
                emit()
                start = current["effective_from"]
                review_start = current["review_date"]
                months = 0
                tickers = set()
                turnover = []
                trades = []
            end = current["effective_to_exclusive"]
            review_end = current["review_date"]
            prior_end = end
            months += 1
            tickers.add(str(current["ticker_asof"] or ""))
            turnover.append(float(current["median_daily_turnover_brl"]))
            trades.append(float(current["median_daily_trade_count"]))
        emit()
    return pl.DataFrame(rows, infer_schema_length=None).sort(
        ["effective_from", "security_id"]
    )


def nearest_thresholds(metrics: pl.DataFrame) -> pl.DataFrame:
    thresholds = {
        "median_daily_turnover_brl": (
            CANONICAL_CONFIG.minimum_median_daily_turnover_brl
        ),
        "median_daily_trade_count": (CANONICAL_CONFIG.minimum_median_daily_trade_count),
    }
    rows: list[dict[str, object]] = []
    for group in metrics.partition_by(
        ["review_date", "effective_from"], maintain_order=True
    ):
        review_date = group.item(0, "review_date")
        effective_from = group.item(0, "effective_from")
        for column, threshold in thresholds.items():
            for status, selected in (
                ("MEMBER", group.filter(pl.col("is_member"))),
                ("REJECTED", group.filter(~pl.col("is_member"))),
            ):
                nearest = (
                    selected.with_columns(
                        (pl.col(column) - threshold).abs().alias("absolute_distance")
                    )
                    .sort(["absolute_distance", "security_id"])
                    .head(10)
                )
                for rank, row in enumerate(nearest.to_dicts(), start=1):
                    rows.append(
                        {
                            "review_date": review_date,
                            "effective_from": effective_from,
                            "metric": column,
                            "status": status,
                            "rank": rank,
                            "security_id": row["security_id"],
                            "ticker_asof": row["ticker_asof"],
                            "source_assignment_type": row["source_assignment_type"],
                            "value": row[column],
                            "threshold": threshold,
                            "signed_distance": row[column] - threshold,
                            "selection_reason": row["selection_reason"],
                        }
                    )
    return pl.DataFrame(rows, infer_schema_length=None).sort(
        ["effective_from", "metric", "status", "rank"]
    )


def source_segments_do_not_overlap(
    assignments: pl.DataFrame, valid_daily: pl.DataFrame
) -> tuple[bool, int]:
    overlap_count = 0
    dates_by_id = {
        row["security_id"]: set(row["trade_date"])
        for row in valid_daily.filter(
            pl.col("security_id").is_in(assignments["security_id"].to_list())
        )
        .group_by("security_id")
        .agg(pl.col("trade_date").unique())
        .iter_rows(named=True)
    }
    for group in assignments.partition_by("source_file"):
        security_ids = group["security_id"].to_list()
        claimed: set[object] = set()
        for security_id in security_ids:
            dates = dates_by_id.get(security_id, set())
            overlap_count += len(claimed & dates)
            claimed.update(dates)
    return overlap_count == 0, overlap_count


def future_append_invariance(
    metrics: pl.DataFrame,
    daily: pl.DataFrame,
    assignments: pl.DataFrame,
) -> tuple[bool, int]:
    valid_daily = daily.filter(valid_observation_expr()).sort(
        ["trade_date", "security_id"]
    )
    calendar = sorted(daily["trade_date"].unique().to_list())
    first_seen = dict(
        valid_daily.filter(
            pl.col("security_id").is_in(assignments["security_id"].to_list())
        )
        .group_by("security_id")
        .agg(pl.col("trade_date").min())
        .iter_rows()
    )
    failures = 0
    for group in metrics.partition_by(
        ["review_date", "effective_from"], maintain_order=True
    ):
        review_date = group.item(0, "review_date")
        future = valid_daily.filter(pl.col("trade_date") > review_date).head(1)
        if not future.height:
            continue
        future = future.with_columns(
            (pl.col("open_brl") * 1.5).alias("open_brl"),
            (pl.col("high_brl") * 1.5).alias("high_brl"),
            (pl.col("low_brl") * 1.5).alias("low_brl"),
            (pl.col("close_brl") * 1.5).alias("close_brl"),
            (pl.col("volume_brl") + 9_999_999.0).alias("volume_brl"),
            (pl.col("trades") + 999).alias("trades"),
        )
        kwargs = {
            "assignments": assignments,
            "calendar": calendar,
            "first_seen": first_seen,
            "review_date": review_date,
            "effective_from": group.item(0, "effective_from"),
            "effective_to_exclusive": group.item(0, "effective_to_exclusive"),
            "prior_members": set(
                group.filter(pl.col("was_member"))["security_id"].to_list()
            ),
            "config": CANONICAL_CONFIG,
        }
        baseline = pl.DataFrame(
            _review_records(valid_daily=valid_daily, **kwargs),
            infer_schema_length=None,
        )
        changed = pl.DataFrame(
            _review_records(valid_daily=pl.concat([valid_daily, future]), **kwargs),
            infer_schema_length=None,
        )
        failures += int(not frames_equal(baseline, changed))
    return failures == 0, failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit a causal liquidity-gated point-in-time B3 universe."
    )
    parser.add_argument("--universe-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(f"Output directory already exists: {args.out}")
    universe_dir = args.universe_dir.resolve()
    paths = {name: require(universe_dir / f"{name}.parquet") for name in TABLE_NAMES}
    manifest_path = require(universe_dir / "manifest.json")
    tables = {name: pl.read_parquet(path) for name, path in paths.items()}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    daily_root = Path(manifest["daily_root"])
    accepted_path = Path(manifest["accepted_assignment_file"])
    if accepted_path.name != ACCEPTED_FILENAME:
        raise ValueError(f"Unexpected accepted assignment file: {accepted_path}")
    assignments = validate_accepted_assignments(pl.read_parquet(accepted_path))
    daily, _ = load_parquets(daily_root, "year=*/equities_daily_*.parquet")
    observations, _ = load_parquets(daily_root, "year=*/ticker_observations_*.parquet")
    recomputed, metadata = build_universe_tables(
        daily, observations, assignments, CANONICAL_CONFIG
    )

    checks: list[Check] = []

    def add(name: str, passed: bool, details: str) -> None:
        checks.append(Check(name, bool(passed), details))

    assignment_counts = {
        assignment_type: count
        for assignment_type, count in assignments.group_by("source_assignment_type")
        .len()
        .iter_rows()
    }
    add(
        "accepted_assignment_axis",
        assignments.height == ACCEPTED_ASSIGNMENT_COUNT,
        f"rows={assignments.height}, composition={assignment_counts}",
    )
    accepted_ids = set(assignments["security_id"].to_list())
    metrics = tables["universe_metrics_monthly"]
    membership = tables["universe_membership_monthly"]
    summary = tables["universe_summary"]
    metric_ids = set(metrics["security_id"].unique().to_list())
    member_ids = set(membership["security_id"].unique().to_list())
    add(
        "candidate_axis_exactly_accepted",
        metric_ids == accepted_ids,
        f"candidate_only={len(metric_ids - accepted_ids)}, accepted_missing={len(accepted_ids - metric_ids)}",
    )
    add(
        "members_subset_of_accepted",
        member_ids.issubset(accepted_ids),
        f"unaccepted_members={len(member_ids - accepted_ids)}",
    )
    per_review = metrics.group_by("effective_from").agg(
        pl.len().alias("rows"),
        pl.col("security_id").n_unique().alias("unique_ids"),
    )
    add(
        "one_candidate_per_review_security",
        bool(
            per_review.select(
                (
                    (pl.col("rows") == ACCEPTED_ASSIGNMENT_COUNT)
                    & (pl.col("unique_ids") == ACCEPTED_ASSIGNMENT_COUNT)
                ).all()
            ).item()
        ),
        f"reviews={per_review.height}",
    )

    effective_dates = summary["effective_from"].to_list()
    expected_ends = {
        effective_from: (
            effective_dates[index + 1] if index + 1 < len(effective_dates) else None
        )
        for index, effective_from in enumerate(effective_dates)
    }
    invalid_intervals = sum(
        row["effective_to_exclusive"] != expected_ends[row["effective_from"]]
        for row in metrics.select("effective_from", "effective_to_exclusive").iter_rows(
            named=True
        )
    )
    add(
        "ordered_exclusive_intervals",
        invalid_intervals == 0 and effective_dates == sorted(effective_dates),
        f"invalid_rows={invalid_intervals}",
    )

    eligibility_failures = 0
    for row in metrics.iter_rows(named=True):
        expected = eligibility_result(
            accepted_identity=bool(row["accepted_identity"]),
            age_sessions=int(row["age_sessions"]),
            presence_sessions=int(row["presence_sessions"]),
            recent_observation=bool(row["eligible_recency"]),
            last_close_brl=row["last_close_brl"],
            median_daily_turnover_brl=float(row["median_daily_turnover_brl"]),
            median_daily_trade_count=float(row["median_daily_trade_count"]),
        )
        eligibility_failures += int(
            any(row[key] != value for key, value in expected.items())
            or row["is_member"] != row["equity_eligible"]
        )
    add(
        "eligibility_recomputes_without_override",
        eligibility_failures == 0,
        f"failed_rows={eligibility_failures}",
    )
    add(
        "members_pass_all_gates",
        bool(
            membership.select(
                pl.all_horizontal(
                    [pl.col(column) for column in ELIGIBILITY_COLUMNS]
                ).all()
            ).item()
        ),
        f"members={membership.height}",
    )
    nonmembers = metrics.filter(~pl.col("is_member"))
    add(
        "nonmembers_fail_at_least_one_gate",
        bool(
            nonmembers.select(
                (
                    ~pl.all_horizontal(
                        [pl.col(column) for column in ELIGIBILITY_COLUMNS]
                    )
                ).all()
            ).item()
        ),
        f"nonmembers={nonmembers.height}",
    )

    for name in TABLE_NAMES:
        add(
            f"{name}_exact_recomputation",
            frames_equal(tables[name], recomputed[name]),
            f"stored_rows={tables[name].height}, recomputed_rows={recomputed[name].height}",
        )

    reordered, _ = build_universe_tables(
        daily.reverse(), observations.reverse(), assignments.reverse(), CANONICAL_CONFIG
    )
    reorder_failures = [
        name
        for name in TABLE_NAMES
        if not frames_equal(recomputed[name], reordered[name])
    ]
    add(
        "input_row_order_determinism",
        not reorder_failures,
        f"different_tables={reorder_failures}",
    )

    causal_pass, causal_failures = future_append_invariance(metrics, daily, assignments)
    add(
        "future_append_invariance_every_review",
        causal_pass,
        f"failed_reviews={causal_failures}",
    )

    ticker_history = tables["ticker_history"]
    ticker_overlaps = 0
    for group in ticker_history.partition_by("security_id", maintain_order=True):
        rows = group.sort("valid_from").to_dicts()
        ticker_overlaps += sum(
            current["valid_from"] <= prior["valid_to"]
            for prior, current in zip(rows, rows[1:], strict=False)
        )
    add(
        "ticker_segments_non_overlapping",
        ticker_overlaps == 0,
        f"overlaps={ticker_overlaps}",
    )
    source_pass, source_overlaps = source_segments_do_not_overlap(
        assignments, daily.filter(valid_observation_expr())
    )
    add(
        "accepted_source_identity_dates_non_overlapping",
        source_pass,
        f"overlaps={source_overlaps}",
    )

    numeric_columns = (
        "presence_ratio",
        "turnover_brl",
        "market_turnover_brl",
        "market_share",
        "median_daily_turnover_brl",
        "median_daily_trade_count",
    )
    nonfinite = sum(
        metrics.filter(~pl.col(column).is_finite()).height for column in numeric_columns
    )
    member_price_nonfinite = membership.filter(
        pl.col("last_close_brl").is_null() | ~pl.col("last_close_brl").is_finite()
    ).height
    add(
        "required_numeric_fields_finite",
        nonfinite == 0 and member_price_nonfinite == 0,
        f"metric_nonfinite={nonfinite}, member_price_nonfinite={member_price_nonfinite}",
    )

    forbidden_policy_fields = {
        "target_count",
        "core_count",
        "buffer_count",
        "entry_presence",
        "retention_presence",
        "entry_market_share",
        "retention_market_share",
        "entry_min_price_brl",
        "retention_min_price_brl",
    }
    selection_contract = manifest.get("selection_contract", {})
    add(
        "manifest_count_and_rank_independent",
        selection_contract.get("count_independent") is True
        and selection_contract.get("rank_independent") is True
        and selection_contract.get("is_member") == "equity_eligible"
        and not (forbidden_policy_fields & set(manifest.get("config", {}))),
        f"forbidden_config={sorted(forbidden_policy_fields & set(manifest.get('config', {})))}",
    )
    add(
        "manifest_config_exact",
        manifest.get("config") == CANONICAL_CONFIG.manifest_dict(),
        f"config={manifest.get('config')}",
    )
    add(
        "minimum_research_cross_section",
        int(summary["member_count"].min()) >= MIN_RESEARCH_CROSS_SECTION,
        f"minimum={int(summary['member_count'].min())}",
    )
    add(
        "manifest_counts_reconcile",
        int(manifest.get("accepted_identity_count", -1)) == ACCEPTED_ASSIGNMENT_COUNT
        and int(manifest.get("review_count", -1)) == summary.height
        and int(manifest.get("union_security_count", -1))
        == tables["universe_union"].height
        and int(manifest.get("latest_member_count", -1))
        == int(summary["member_count"][-1])
        and int(manifest.get("invalid_observation_count", -1))
        == int(metadata["invalid_observation_count"]),
        "manifest versus recomputed metadata",
    )

    accepted_hash_pass = manifest.get("accepted_assignment_sha256") == sha256_file(
        accepted_path
    )
    source_hashes = manifest.get("cotahist_input_sha256", {})
    source_hash_failures = [
        path
        for path, expected in source_hashes.items()
        if not Path(path).is_file() or sha256_file(Path(path)) != expected
    ]
    parse_audit = manifest.get("cotahist_parse_audit")
    parse_hash_pass = bool(
        parse_audit
        and Path(parse_audit["path"]).is_file()
        and sha256_file(Path(parse_audit["path"])) == parse_audit["sha256"]
    )
    add(
        "input_hashes_match_manifest",
        accepted_hash_pass and not source_hash_failures and parse_hash_pass,
        f"accepted={accepted_hash_pass}, source_failures={len(source_hash_failures)}, parse_audit={parse_hash_pass}",
    )
    implementation_hashes = manifest.get("implementation_sha256", {})
    implementation_paths = {
        path.name: path
        for path in (
            Path(__file__).resolve(),
            Path(__file__).with_name("pit_universe.py").resolve(),
            Path(__file__).with_name("build_point_in_time_universe.py").resolve(),
        )
    }
    implementation_failures = [
        name
        for name, path in implementation_paths.items()
        if implementation_hashes.get(name) != sha256_file(path)
    ]
    add(
        "implementation_hashes_match_manifest",
        not implementation_failures,
        f"failures={implementation_failures}",
    )
    output_hashes = manifest.get("output_sha256", {})
    output_hash_failures = [
        filename
        for filename, expected in output_hashes.items()
        if not (universe_dir / filename).is_file()
        or sha256_file(universe_dir / filename) != expected
    ]
    add(
        "output_hashes_match_manifest",
        not output_hash_failures,
        f"failures={output_hash_failures}",
    )

    args.out.mkdir(parents=True, exist_ok=False)
    intervals = compress_membership_intervals(membership)
    nearest = nearest_thresholds(metrics)
    latest_effective = summary["effective_from"][-1]
    latest_members = membership.filter(
        pl.col("effective_from") == latest_effective
    ).sort("median_daily_turnover_brl")
    for name, frame in {
        "monthly_diagnostics": summary,
        "nearest_liquidity_thresholds": nearest,
        "membership_intervals": intervals,
        "latest_members": latest_members,
    }.items():
        frame.write_parquet(
            args.out / f"{name}.parquet", compression="zstd", statistics=True
        )
        frame.write_csv(args.out / f"{name}.csv")

    check_frame = pl.DataFrame(
        [
            {"check": check.name, "passed": check.passed, "details": check.details}
            for check in checks
        ]
    )
    check_frame.write_csv(args.out / "integrity_checks.csv")
    member_counts = summary["member_count"].to_list()
    audit_summary = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "universe_dir": str(universe_dir),
        "output_dir": str(args.out.resolve()),
        "checks_passed": sum(check.passed for check in checks),
        "check_count": len(checks),
        "all_integrity_checks_passed": all(check.passed for check in checks),
        "member_count": {
            "minimum": min(member_counts),
            "median": float(summary["member_count"].median()),
            "mean": float(summary["member_count"].mean()),
            "maximum": max(member_counts),
            "latest": member_counts[-1],
        },
        "latest_effective_from": latest_effective.isoformat(),
        "latest_gate_counts": {
            column: int(summary[column][-1])
            for column in summary.columns
            if column.endswith("_count")
        },
        "direct_member_rows": int(
            membership.filter(pl.col("source_assignment_type") == "DIRECT_ISIN").height
        ),
        "recovered_member_rows": int(
            membership.filter(
                pl.col("source_assignment_type") == "RECOVERED_RELABELED"
            ).height
        ),
    }
    (args.out / "audit_summary.json").write_text(
        json.dumps(audit_summary, indent=2), encoding="utf-8"
    )
    print(f"Audit complete: {args.out}")
    print(
        f"Integrity checks passed: {audit_summary['checks_passed']}/"
        f"{audit_summary['check_count']}"
    )
    return 0 if audit_summary["all_integrity_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
