from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import polars as pl

SCRIPT_VERSION = "1"


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def norm_symbol(value: object) -> str:
    return str(value or "").strip().upper()


def join_values(values: Iterable[object]) -> str:
    return "|".join(sorted({str(v) for v in values if v not in (None, "")}))


def load_required(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pl.read_parquet(path)


def scan_m1_archives(roots: list[Path]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"M1 root does not exist: {root}")
        for path in sorted(root.rglob("bars_m1_*.parquet")):
            resolved = str(path.resolve())
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            try:
                stats = (
                    pl.scan_parquet(path)
                    .select(
                        pl.len().alias("rows"),
                        pl.col("symbol").first().alias("symbol"),
                        pl.col("ts_exchange").min().alias("first_exchange"),
                        pl.col("ts_exchange").max().alias("last_exchange"),
                    )
                    .collect()
                    .row(0, named=True)
                )
                rows.append(
                    {
                        "symbol": norm_symbol(stats["symbol"]),
                        "rows": int(stats["rows"]),
                        "first_exchange": stats["first_exchange"],
                        "last_exchange": stats["last_exchange"],
                        "path": resolved,
                        "root": str(root.resolve()),
                        "error": "",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "symbol": norm_symbol(path.stem.removeprefix("bars_m1_")),
                        "rows": 0,
                        "first_exchange": None,
                        "last_exchange": None,
                        "path": resolved,
                        "root": str(root.resolve()),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    if not rows:
        return pl.DataFrame(
            schema={
                "symbol": pl.String,
                "rows": pl.Int64,
                "first_exchange": pl.Datetime("us", "America/Sao_Paulo"),
                "last_exchange": pl.Datetime("us", "America/Sao_Paulo"),
                "path": pl.String,
                "root": pl.String,
                "error": pl.String,
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None).sort(["symbol", "path"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a point-in-time B3 universe and map its historical security IDs "
            "to exact symbols available on an XP MetaTrader 5 server catalogue."
        )
    )
    parser.add_argument("--universe-dir", type=Path, required=True)
    parser.add_argument("--xp-catalogue", type=Path, required=True)
    parser.add_argument("--m1-root", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", default="2021-07-19")
    parser.add_argument("--end", default="2026-07-17")
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(f"Output directory already exists: {args.out}")
    args.out.mkdir(parents=True, exist_ok=False)

    research_start = parse_date(args.start)
    research_end = parse_date(args.end)

    membership = load_required(args.universe_dir / "universe_membership_monthly.parquet")
    union = load_required(args.universe_dir / "universe_union.parquet")
    master = load_required(args.universe_dir / "security_master.parquet")
    ticker_history = load_required(args.universe_dir / "ticker_history.parquet")
    load_required(args.universe_dir / "universe_changes.parquet")
    summary = load_required(args.universe_dir / "universe_summary.parquet")
    catalogue = load_required(args.xp_catalogue)

    if "name" not in catalogue.columns:
        raise ValueError("XP catalogue does not contain a 'name' column")

    # Basic PIT-universe integrity checks.
    membership_duplicate_rows = int(
        membership.height
        - membership.unique(subset=["effective_from", "security_id"]).height
    )
    union_distinct_ids = int(union["security_id"].n_unique())
    membership_distinct_ids = int(membership["security_id"].n_unique())
    union_duplicate_ids = int(union.height - union_distinct_ids)
    master_duplicate_ids = int(master.height - master["security_id"].n_unique())
    fallback_security_count = int(
        master.filter(pl.col("security_id_is_fallback") == True).height  # noqa: E712
    )
    invalid_intervals = int(
        membership.filter(
            pl.col("effective_to_exclusive").is_not_null()
            & (pl.col("effective_to_exclusive") <= pl.col("effective_from"))
        ).height
    )
    latest_effective = membership["effective_from"].max()
    latest_members = membership.filter(pl.col("effective_from") == latest_effective)
    latest_member_count = latest_members.height
    summary_latest_count = int(summary.sort("effective_from")["member_count"][-1])
    union_count_matches = union.height == membership_distinct_ids

    integrity_rows = [
        {"check": "membership_duplicate_effective_security_rows", "value": membership_duplicate_rows, "pass": membership_duplicate_rows == 0},
        {"check": "universe_union_duplicate_security_ids", "value": union_duplicate_ids, "pass": union_duplicate_ids == 0},
        {"check": "security_master_duplicate_security_ids", "value": master_duplicate_ids, "pass": master_duplicate_ids == 0},
        {"check": "invalid_membership_intervals", "value": invalid_intervals, "pass": invalid_intervals == 0},
        {"check": "union_matches_distinct_membership_security_ids", "value": f"{union.height}/{membership_distinct_ids}", "pass": union_count_matches},
        {"check": "latest_member_count_matches_summary", "value": f"{latest_member_count}/{summary_latest_count}", "pass": latest_member_count == summary_latest_count},
        {"check": "fallback_security_ids", "value": fallback_security_count, "pass": True},
    ]
    integrity = pl.DataFrame(integrity_rows, infer_schema_length=None)

    # Restrict ticker segments to those overlapping the research interval.
    overlapping = ticker_history.filter(
        (pl.col("valid_to") >= pl.lit(research_start))
        & (pl.col("valid_from") <= pl.lit(research_end))
    ).with_columns(pl.col("ticker").str.strip_chars().str.to_uppercase().alias("ticker"))

    union_ids = set(union["security_id"].to_list())
    overlapping = overlapping.filter(pl.col("security_id").is_in(list(union_ids)))

    xp_symbols = set(
        catalogue.select(pl.col("name").cast(pl.String).str.strip_chars().str.to_uppercase())
        .to_series()
        .to_list()
    )

    # Map each ticker to all PIT security IDs that used it. Ticker reuse is possible.
    ticker_to_ids: dict[str, set[str]] = defaultdict(set)
    for sid, ticker in overlapping.select("security_id", "ticker").iter_rows():
        ticker_to_ids[norm_symbol(ticker)].add(str(sid))

    archived = scan_m1_archives(args.m1_root)
    archived_by_symbol: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in archived.to_dicts():
        archived_by_symbol[norm_symbol(row["symbol"])].append(row)

    # Build source dictionaries.
    master_by_id = {str(row["security_id"]): row for row in master.to_dicts()}
    union_by_id = {str(row["security_id"]): row for row in union.to_dicts()}
    history_by_id: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in overlapping.sort(["security_id", "valid_from"]).to_dicts():
        history_by_id[str(row["security_id"])].append(row)

    mapping_rows: list[dict[str, object]] = []
    symbol_claims: dict[str, list[str]] = defaultdict(list)

    for sid in sorted(union_ids):
        u = union_by_id.get(str(sid), {})
        m = master_by_id.get(str(sid), {})
        history = history_by_id.get(str(sid), [])
        tickers = [norm_symbol(row.get("ticker")) for row in history]
        tickers = list(dict.fromkeys([ticker for ticker in tickers if ticker]))
        latest_ticker = norm_symbol(
            u.get("latest_member_ticker") or m.get("latest_ticker") or (tickers[-1] if tickers else "")
        )
        available = [ticker for ticker in tickers if ticker in xp_symbols]
        if latest_ticker and latest_ticker in xp_symbols and latest_ticker not in available:
            available.append(latest_ticker)
        available = list(dict.fromkeys(available))

        preferred = ""
        if latest_ticker in available:
            preferred = latest_ticker
        elif available:
            # Prefer the most recent available ticker in historical order.
            for ticker in reversed(tickers):
                if ticker in available:
                    preferred = ticker
                    break
            if not preferred:
                preferred = available[-1]

        archived_symbols = [ticker for ticker in available if ticker in archived_by_symbol]
        if preferred and preferred in archived_by_symbol and preferred not in archived_symbols:
            archived_symbols.append(preferred)

        ticker_collision = bool(preferred and len(ticker_to_ids.get(preferred, set())) > 1)
        fallback = bool(m.get("security_id_is_fallback", False))
        multi_ticker = len(tickers) > 1

        if fallback:
            status = "MANUAL_FALLBACK_SECURITY_ID"
        elif not available:
            status = "NO_EXACT_XP_SYMBOL"
        elif ticker_collision:
            status = "AMBIGUOUS_TICKER_REUSE"
        elif archived_symbols:
            status = "ALREADY_ARCHIVED_RECONCILE"
        else:
            status = "READY_FOR_DEPTH_PROBE"

        if preferred:
            symbol_claims[preferred].append(str(sid))

        archived_paths: list[str] = []
        archived_rows = 0
        archived_first: object = None
        archived_last: object = None
        for sym in archived_symbols:
            for item in archived_by_symbol.get(sym, []):
                archived_paths.append(str(item["path"]))
                archived_rows = max(archived_rows, int(item["rows"]))
                first = item["first_exchange"]
                last = item["last_exchange"]
                if first is not None and (archived_first is None or first < archived_first):
                    archived_first = first
                if last is not None and (archived_last is None or last > archived_last):
                    archived_last = last

        mapping_rows.append(
            {
                "security_id": str(sid),
                "isin": str(u.get("isin") or m.get("isin") or ""),
                "latest_ticker": latest_ticker,
                "latest_issuer": str(u.get("latest_member_issuer") or m.get("latest_issuer_short_name") or ""),
                "first_effective_from": u.get("first_effective_from"),
                "last_effective_from": u.get("last_effective_from"),
                "membership_months": int(u.get("membership_months") or 0),
                "first_observed_date": m.get("first_observed_date"),
                "last_observed_date": m.get("last_observed_date"),
                "distinct_tickers": len(tickers),
                "tickers": join_values(tickers),
                "xp_available_symbols": join_values(available),
                "xp_available_count": len(available),
                "preferred_xp_symbol": preferred,
                "preferred_symbol_security_claims": len(ticker_to_ids.get(preferred, set())) if preferred else 0,
                "multi_ticker_lineage": multi_ticker,
                "security_id_is_fallback": fallback,
                "already_archived": bool(archived_symbols),
                "archived_symbols": join_values(archived_symbols),
                "archived_rows": archived_rows,
                "archived_first_exchange": archived_first,
                "archived_last_exchange": archived_last,
                "archived_paths": join_values(archived_paths),
                "needs_daily_reconciliation": True,
                "mapping_status": status,
            }
        )

    mapping = pl.DataFrame(mapping_rows, infer_schema_length=None).sort(
        ["mapping_status", "latest_ticker", "security_id"]
    )

    collision_rows: list[dict[str, object]] = []
    for symbol, ids in sorted(symbol_claims.items()):
        unique_ids = sorted(set(ids))
        if len(unique_ids) > 1:
            collision_rows.append(
                {
                    "symbol": symbol,
                    "security_id_count": len(unique_ids),
                    "security_ids": "|".join(unique_ids),
                }
            )
    collisions = (
        pl.DataFrame(collision_rows, infer_schema_length=None)
        if collision_rows
        else pl.DataFrame(schema={"symbol": pl.String, "security_id_count": pl.Int64, "security_ids": pl.String})
    )

    manual = mapping.filter(
        pl.col("mapping_status").is_in(
            [
                "MANUAL_FALLBACK_SECURITY_ID",
                "NO_EXACT_XP_SYMBOL",
                "AMBIGUOUS_TICKER_REUSE",
            ]
        )
        | pl.col("multi_ticker_lineage")
    )

    ready_symbols = (
        mapping.filter(pl.col("mapping_status") == "READY_FOR_DEPTH_PROBE")
        .select("preferred_xp_symbol")
        .unique()
        .sort("preferred_xp_symbol")
        .to_series()
        .to_list()
    )
    archived_symbols = (
        mapping.filter(pl.col("mapping_status") == "ALREADY_ARCHIVED_RECONCILE")
        .select("preferred_xp_symbol")
        .filter(pl.col("preferred_xp_symbol") != "")
        .unique()
        .sort("preferred_xp_symbol")
        .to_series()
        .to_list()
    )

    # Latest membership snapshot for convenience.
    latest_snapshot = latest_members.select(
        [
            "effective_from",
            "effective_to_exclusive",
            "security_id",
            "isin",
            "ticker_asof",
            "issuer_short_name_asof",
            "market_share",
            "presence_ratio",
            "last_close_brl",
        ]
    ).sort(["market_share", "ticker_asof"], descending=[True, False])

    # Outputs.
    integrity.write_csv(args.out / "universe_integrity_checks.csv")
    mapping.write_parquet(args.out / "security_xp_mapping.parquet", compression="zstd", statistics=True)
    mapping.write_csv(args.out / "security_xp_mapping.csv")
    collisions.write_csv(args.out / "symbol_collisions.csv")
    manual.write_csv(args.out / "manual_review.csv")
    archived.write_csv(args.out / "existing_m1_inventory.csv")
    latest_snapshot.write_csv(args.out / "latest_membership_snapshot.csv")
    (args.out / "depth_probe_symbols.txt").write_text("\n".join(ready_symbols) + ("\n" if ready_symbols else ""), encoding="utf-8")
    (args.out / "already_archived_symbols.txt").write_text("\n".join(archived_symbols) + ("\n" if archived_symbols else ""), encoding="utf-8")

    status_counts = mapping.group_by("mapping_status").agg(pl.len().alias("count")).sort("mapping_status")
    status_counts.write_csv(args.out / "mapping_status_summary.csv")

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "universe_dir": str(args.universe_dir.resolve()),
        "xp_catalogue": str(args.xp_catalogue.resolve()),
        "m1_roots": [str(path.resolve()) for path in args.m1_root],
        "research_start": research_start.isoformat(),
        "research_end": research_end.isoformat(),
        "membership_rows": membership.height,
        "membership_distinct_security_ids": membership_distinct_ids,
        "universe_union_rows": union.height,
        "parent_security_rows": master.height,
        "fallback_security_ids": fallback_security_count,
        "latest_effective_from": latest_effective.isoformat() if latest_effective else None,
        "latest_member_count": latest_member_count,
        "xp_catalogue_symbol_count": len(xp_symbols),
        "existing_m1_file_count": archived.height,
        "mapping_status_counts": {row[0]: row[1] for row in status_counts.iter_rows()},
        "ready_depth_probe_symbol_count": len(ready_symbols),
        "already_archived_preferred_symbol_count": len(archived_symbols),
        "symbol_collision_count": collisions.height,
        "integrity_all_pass": all(bool(row["pass"]) for row in integrity.to_dicts()),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(f"Complete: {args.out}")
    print(f"Universe union securities: {union.height:,}")
    print(f"Latest members: {latest_member_count:,}")
    print(f"Existing M1 files inventoried: {archived.height:,}")
    print(f"Ready for depth probe: {len(ready_symbols):,}")
    print(f"Manual-review rows: {manual.height:,}")
    print(f"Symbol collisions: {collisions.height:,}")
    return 0 if manifest["integrity_all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
