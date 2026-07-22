from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

SCRIPT_VERSION = "1"


@dataclass(frozen=True)
class UniverseConfig:
    start_date: str
    end_date: str
    lookback_sessions: int
    implementation_lag_sessions: int
    entry_presence: float
    retention_presence: float
    entry_market_share: float
    retention_market_share: float
    entry_min_price_brl: float
    retention_min_price_brl: float


def load_parquets(root: Path, pattern: str) -> pl.DataFrame:
    paths = sorted(root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched {root / pattern}")
    return pl.concat([pl.read_parquet(path) for path in paths], how="diagonal_relaxed")


def month_ends(calendar: list[date]) -> list[date]:
    result: list[date] = []
    current_key: tuple[int, int] | None = None
    current_last: date | None = None
    for day in calendar:
        key = (day.year, day.month)
        if current_key is not None and key != current_key and current_last is not None:
            result.append(current_last)
        current_key, current_last = key, day
    if current_last is not None:
        result.append(current_last)
    return result


def ticker_segments(observations: pl.DataFrame) -> pl.DataFrame:
    chosen = (
        observations.sort(["security_id", "trade_date", "volume_brl"])
        .group_by(["security_id", "trade_date"], maintain_order=True)
        .agg(
            pl.col("ticker").last(),
            pl.col("isin").last(),
            pl.col("issuer_short_name").last(),
            pl.col("security_spec").last(),
            pl.col("security_id_is_fallback").last(),
        )
        .sort(["security_id", "trade_date"])
    )
    rows: list[dict[str, object]] = []
    for group in chosen.partition_by("security_id", maintain_order=True):
        data = group.to_dicts()
        if not data:
            continue
        security_id = data[0]["security_id"]
        start = data[0]["trade_date"]
        prior = data[0]
        for current in data[1:]:
            if current["ticker"] != prior["ticker"]:
                rows.append({
                    "security_id": security_id,
                    "ticker": prior["ticker"],
                    "valid_from": start,
                    "valid_to": prior["trade_date"],
                    "isin": prior["isin"],
                    "issuer_short_name": prior["issuer_short_name"],
                    "security_spec": prior["security_spec"],
                    "security_id_is_fallback": prior["security_id_is_fallback"],
                })
                start = current["trade_date"]
            prior = current
        rows.append({
            "security_id": security_id,
            "ticker": prior["ticker"],
            "valid_from": start,
            "valid_to": prior["trade_date"],
            "isin": prior["isin"],
            "issuer_short_name": prior["issuer_short_name"],
            "security_spec": prior["security_spec"],
            "security_id_is_fallback": prior["security_id_is_fallback"],
        })
    return pl.DataFrame(rows, infer_schema_length=None).sort(["security_id", "valid_from"])


def security_master(daily: pl.DataFrame, segments: pl.DataFrame) -> pl.DataFrame:
    latest = (
        daily.sort(["security_id", "trade_date", "volume_brl"])
        .group_by("security_id", maintain_order=True)
        .agg(
            pl.col("trade_date").min().alias("first_observed_date"),
            pl.col("trade_date").max().alias("last_observed_date"),
            pl.col("ticker").last().alias("latest_ticker"),
            pl.col("issuer_short_name").last().alias("latest_issuer_short_name"),
            pl.col("security_spec").last().alias("latest_security_spec"),
            pl.col("isin").last().alias("isin"),
            pl.col("security_id_is_fallback").max().alias("security_id_is_fallback"),
            pl.col("trade_date").n_unique().alias("observed_sessions"),
            pl.col("volume_brl").sum().alias("total_volume_brl"),
        )
    )
    ticker_info = segments.group_by("security_id").agg(
        pl.col("ticker").n_unique().alias("distinct_tickers"),
        pl.col("ticker").unique().sort().alias("tickers"),
    )
    return latest.join(ticker_info, on="security_id", how="left").sort("security_id")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a monthly point-in-time B3 equity universe.")
    parser.add_argument("--daily-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", default="2021-07-19")
    parser.add_argument("--end", default="2026-07-17")
    parser.add_argument("--lookback-sessions", type=int, default=63)
    parser.add_argument("--implementation-lag-sessions", type=int, default=5)
    parser.add_argument("--entry-presence", type=float, default=0.95)
    parser.add_argument("--retention-presence", type=float, default=0.90)
    parser.add_argument("--entry-market-share", type=float, default=0.001)
    parser.add_argument("--retention-market-share", type=float, default=0.0005)
    parser.add_argument("--entry-min-price", type=float, default=1.00)
    parser.add_argument("--retention-min-price", type=float, default=0.80)
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(f"Output directory already exists: {args.out}")
    args.out.mkdir(parents=True, exist_ok=False)

    cfg = UniverseConfig(
        args.start, args.end, args.lookback_sessions, args.implementation_lag_sessions,
        args.entry_presence, args.retention_presence, args.entry_market_share,
        args.retention_market_share, args.entry_min_price, args.retention_min_price,
    )
    requested_start = date.fromisoformat(args.start)
    requested_end = date.fromisoformat(args.end)

    print("Loading parsed COTAHIST files ...", flush=True)
    daily = load_parquets(args.daily_root, "year=*/equities_daily_*.parquet").sort(["trade_date", "security_id"])
    observations = load_parquets(args.daily_root, "year=*/ticker_observations_*.parquet").sort(["trade_date", "security_id", "ticker"])
    calendar = sorted(daily["trade_date"].unique().to_list())
    cal_idx = {day: i for i, day in enumerate(calendar)}
    start_date = next((d for d in calendar if d >= requested_start), None)
    end_candidates = [d for d in calendar if d <= requested_end]
    if start_date is None or not end_candidates:
        raise ValueError("Requested interval does not overlap the parsed market calendar")
    end_date = end_candidates[-1]

    first_seen = dict(daily.group_by("security_id").agg(pl.col("trade_date").min()).iter_rows())
    schedule: list[tuple[date, date]] = []
    for review in month_ends(calendar):
        i = cal_idx[review]
        if i + 1 < args.lookback_sessions:
            continue
        effective_i = i + args.implementation_lag_sessions
        if effective_i >= len(calendar):
            continue
        effective = calendar[effective_i]
        if effective <= end_date:
            schedule.append((review, effective))
    if not schedule:
        raise ValueError("No review schedule could be created")

    current_members: set[str] = set()
    all_ids = set(daily["security_id"].unique().to_list())
    metric_rows: list[dict[str, object]] = []
    member_rows: list[dict[str, object]] = []
    change_rows: list[dict[str, object]] = []

    for review, effective in schedule:
        i = cal_idx[review]
        window_dates = calendar[i - args.lookback_sessions + 1 : i + 1]
        window_start = window_dates[0]
        window = daily.filter(pl.col("trade_date").is_between(window_start, review))
        total_volume = float(window["volume_brl"].sum())
        grouped = (
            window.sort(["security_id", "trade_date", "volume_brl"])
            .group_by("security_id", maintain_order=True)
            .agg(
                pl.col("trade_date").n_unique().alias("presence_sessions"),
                pl.col("volume_brl").sum().alias("turnover_brl"),
                pl.col("trade_date").max().alias("last_observed_date"),
                pl.col("close_brl").sort_by("trade_date").last().alias("last_close_brl"),
                pl.col("ticker").sort_by("trade_date").last().alias("ticker_asof"),
                pl.col("issuer_short_name").sort_by("trade_date").last().alias("issuer_asof"),
                pl.col("security_spec").sort_by("trade_date").last().alias("spec_asof"),
                pl.col("isin").sort_by("trade_date").last().alias("isin"),
            )
        )
        by_id = {row[0]: row for row in grouped.iter_rows()}
        prior = set(current_members)
        next_members: set[str] = set()

        for sid in sorted(all_ids | prior):
            row = by_id.get(sid)
            if row is None:
                presence_sessions = 0
                turnover_brl = 0.0
                last_observed_date = None
                last_close = None
                ticker = issuer = spec = isin = ""
            else:
                _, presence_sessions, turnover_brl, last_observed_date, last_close, ticker, issuer, spec, isin = row
            seen = first_seen.get(sid)
            age_sessions = i - cal_idx[seen] + 1 if seen in cal_idx else 0
            presence = presence_sessions / args.lookback_sessions
            share = float(turnover_brl) / total_volume if total_volume > 0 else 0.0
            was_member = sid in prior
            entry_pass = bool(
                age_sessions >= args.lookback_sessions
                and presence >= args.entry_presence
                and share >= args.entry_market_share
                and last_close is not None
                and last_close >= args.entry_min_price
            )
            retention_pass = bool(
                presence >= args.retention_presence
                and share >= args.retention_market_share
                and last_close is not None
                and last_close >= args.retention_min_price
            )
            is_member = retention_pass if was_member else entry_pass
            if is_member:
                next_members.add(sid)
            action = "ENTER" if (not was_member and is_member) else "STAY" if (was_member and is_member) else "EXIT" if (was_member and not is_member) else "OUT"
            record = {
                "review_date": review,
                "effective_from": effective,
                "window_start": window_start,
                "window_end": review,
                "security_id": sid,
                "isin": isin,
                "ticker_asof": ticker,
                "issuer_short_name_asof": issuer,
                "security_spec_asof": spec,
                "age_sessions": age_sessions,
                "presence_sessions": presence_sessions,
                "presence_ratio": presence,
                "turnover_brl": float(turnover_brl),
                "market_share": share,
                "last_observed_date": last_observed_date,
                "last_close_brl": last_close,
                "was_member": was_member,
                "entry_pass": entry_pass,
                "retention_pass": retention_pass,
                "is_member": is_member,
                "action": action,
            }
            metric_rows.append(record)
            if is_member:
                member_rows.append(record)
            if action in {"ENTER", "EXIT"}:
                change_rows.append(record)
        current_members = next_members
        print(f"{review} -> {effective}: {len(current_members)} members ({len(current_members-prior)} entries, {len(prior-current_members)} exits)")

    metrics = pl.DataFrame(metric_rows, infer_schema_length=None).sort(["effective_from", "security_id"])
    membership = pl.DataFrame(member_rows, infer_schema_length=None).sort(["effective_from", "security_id"])
    changes = pl.DataFrame(change_rows, infer_schema_length=None).sort(["effective_from", "action", "security_id"])

    anchor_candidates = [effective for _, effective in schedule if effective <= start_date]
    anchor_effective = max(anchor_candidates) if anchor_candidates else min(effective for _, effective in schedule)
    metrics = metrics.filter(pl.col("effective_from") >= anchor_effective)
    membership = membership.filter(pl.col("effective_from") >= anchor_effective)
    changes = changes.filter(pl.col("effective_from") >= anchor_effective)

    effective_dates = sorted(membership["effective_from"].unique().to_list())
    next_map = {d: (effective_dates[i+1] if i+1 < len(effective_dates) else None) for i, d in enumerate(effective_dates)}
    membership = membership.with_columns(
        pl.col("effective_from").map_elements(lambda d: next_map.get(d), return_dtype=pl.Date).alias("effective_to_exclusive")
    )

    segments = ticker_segments(observations)
    master = security_master(daily, segments)
    union = (
        membership.group_by("security_id")
        .agg(
            pl.col("effective_from").min().alias("first_effective_from"),
            pl.col("effective_from").max().alias("last_effective_from"),
            pl.col("ticker_asof").last().alias("latest_member_ticker"),
            pl.col("issuer_short_name_asof").last().alias("latest_member_issuer"),
            pl.col("effective_from").n_unique().alias("membership_months"),
            pl.col("market_share").median().alias("median_market_share_while_member"),
        )
        .join(master, on="security_id", how="left")
        .sort(["first_effective_from", "security_id"])
    )
    summary = (
        membership.group_by(["review_date", "effective_from"])
        .agg(
            pl.len().alias("member_count"),
            pl.col("market_share").sum().alias("member_market_share_sum"),
            pl.col("turnover_brl").sum().alias("member_turnover_brl"),
        )
        .sort("effective_from")
    )

    outputs = {
        "universe_metrics_monthly": metrics,
        "universe_membership_monthly": membership,
        "universe_changes": changes,
        "universe_union": union,
        "security_master": master,
        "ticker_history": segments,
        "universe_summary": summary,
    }
    for name, frame in outputs.items():
        frame.write_parquet(args.out / f"{name}.parquet", compression="zstd", statistics=True)
        csv_frame = frame
        if "tickers" in csv_frame.columns:
            csv_frame = csv_frame.with_columns(
                pl.col("tickers").list.join("|").alias("tickers")
            )
        csv_frame.write_csv(args.out / f"{name}.csv")

    latest_member_count = int(summary["member_count"][-1]) if summary.height else 0
    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "daily_root": str(args.daily_root),
        "output_dir": str(args.out),
        "config": asdict(cfg),
        "resolved_start_date": start_date.isoformat(),
        "resolved_end_date": end_date.isoformat(),
        "anchor_effective_date": anchor_effective.isoformat(),
        "market_sessions": len(calendar),
        "review_count": len(effective_dates),
        "parent_security_count": len(all_ids),
        "union_security_count": union.height,
        "latest_member_count": latest_member_count,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (args.out / "config.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["parameter", "value"])
        for key, value in asdict(cfg).items(): writer.writerow([key, value])
    print(f"\nComplete: {args.out}\nParent securities: {len(all_ids):,}\nEver selected: {union.height:,}\nLatest membership: {latest_member_count:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
