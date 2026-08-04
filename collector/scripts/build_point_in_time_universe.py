from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from pit_universe import (
    CANONICAL_CONFIG,
    MIN_RESEARCH_CROSS_SECTION,
    SELECTION_REASON_PRIORITY,
    build_universe_tables,
    validate_accepted_assignments,
)

SCRIPT_VERSION = "2"
ACCEPTED_FILENAME = "xp_accepted_source_assignments_v1.parquet"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_parquets(root: Path, pattern: str) -> tuple[pl.DataFrame, list[Path]]:
    paths = sorted(root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched {root / pattern}")
    return (
        pl.concat([pl.read_parquet(path) for path in paths], how="diagonal_relaxed"),
        paths,
    )


def repository_commit() -> str:
    repository = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def repository_status() -> list[str]:
    repository = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def write_table(path: Path, frame: pl.DataFrame) -> None:
    frame.write_parquet(
        path.with_suffix(".parquet"), compression="zstd", statistics=True
    )
    csv_frame = frame
    for column, dtype in zip(csv_frame.columns, csv_frame.dtypes, strict=True):
        if isinstance(dtype, pl.List):
            csv_frame = csv_frame.with_columns(
                pl.col(column).list.join("|").alias(column)
            )
    csv_frame.write_csv(path.with_suffix(".csv"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the causal liquidity-gated point-in-time B3 equity universe."
        )
    )
    parser.add_argument("--daily-root", type=Path, required=True)
    parser.add_argument("--accepted-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(f"Output directory already exists: {args.out}")
    daily_root = args.daily_root.resolve()
    accepted_dir = args.accepted_dir.resolve()
    accepted_path = accepted_dir / ACCEPTED_FILENAME
    if not accepted_path.is_file():
        raise FileNotFoundError(accepted_path)

    print("Loading and validating canonical inputs ...", flush=True)
    assignments = validate_accepted_assignments(pl.read_parquet(accepted_path))
    daily, daily_paths = load_parquets(daily_root, "year=*/equities_daily_*.parquet")
    observations, observation_paths = load_parquets(
        daily_root, "year=*/ticker_observations_*.parquet"
    )
    tables, metadata = build_universe_tables(
        daily, observations, assignments, CANONICAL_CONFIG
    )
    summary = tables["universe_summary"]
    minimum_count = int(summary["member_count"].min())
    if minimum_count < MIN_RESEARCH_CROSS_SECTION:
        failed = summary.filter(pl.col("member_count") < MIN_RESEARCH_CROSS_SECTION)
        raise ValueError(
            "Liquidity gate violates the downstream 30-equity minimum: "
            f"{failed.select('review_date', 'effective_from', 'member_count').to_dicts()}"
        )

    args.out.mkdir(parents=True, exist_ok=False)
    for name, frame in tables.items():
        write_table(args.out / name, frame)

    config_path = args.out / "config.csv"
    with config_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value"])
        writer.writerows(CANONICAL_CONFIG.manifest_dict().items())

    input_paths = [*daily_paths, *observation_paths]
    parse_audit = daily_root / "parse_audit.json"
    cotahist_provenance = None
    if parse_audit.is_file():
        parsed = json.loads(parse_audit.read_text(encoding="utf-8"))
        cotahist_provenance = [
            {
                "year": row["year"],
                "source_zip": row["source_zip"],
                "source_sha256": row["source_sha256"],
            }
            for row in parsed.get("audits", [])
        ]
    output_paths = sorted(
        path for path in args.out.iterdir() if path.name != "manifest.json"
    )
    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": repository_commit(),
        "repository_status_porcelain": repository_status(),
        "implementation_sha256": {
            path.name: sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                Path(__file__).with_name("pit_universe.py").resolve(),
                Path(__file__).with_name("audit_pit_universe.py").resolve(),
            )
        },
        "daily_root": str(daily_root),
        "accepted_assignment_dir": str(accepted_dir),
        "accepted_assignment_file": str(accepted_path),
        "accepted_assignment_sha256": sha256_file(accepted_path),
        "accepted_identity_count": metadata["accepted_identity_count"],
        "accepted_assignment_composition": {
            assignment_type: count
            for assignment_type, count in assignments.group_by("source_assignment_type")
            .len()
            .iter_rows()
        },
        "output_dir": str(args.out.resolve()),
        "config": CANONICAL_CONFIG.manifest_dict(),
        "selection_contract": {
            "candidate_axis": "accepted security_id only",
            "missing_session_median_treatment": ("zero turnover and zero trades"),
            "count_independent": True,
            "rank_independent": True,
            "is_member": "equity_eligible",
            "selection_reason_priority": list(SELECTION_REASON_PRIORITY),
        },
        **{
            key: value.isoformat() if hasattr(value, "isoformat") else value
            for key, value in metadata.items()
        },
        "cotahist_input_sha256": {
            str(path.resolve()): sha256_file(path) for path in input_paths
        },
        "cotahist_parse_audit": (
            {
                "path": str(parse_audit.resolve()),
                "sha256": sha256_file(parse_audit),
                "source_archives": cotahist_provenance,
            }
            if parse_audit.is_file()
            else None
        ),
        "output_sha256": {path.name: sha256_file(path) for path in output_paths},
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"Complete: {args.out}")
    print(f"Accepted identities: {metadata['accepted_identity_count']:,}")
    print(f"Ever selected: {metadata['union_security_count']:,}")
    print(
        "Membership count min/latest/max: "
        f"{minimum_count:,}/{metadata['latest_member_count']:,}/"
        f"{int(summary['member_count'].max()):,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
