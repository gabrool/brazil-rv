"""Re-derive D+1 odd-lot volume from the bounded, immutable COTAHIST archives."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from datetime import date
from pathlib import Path

import polars as pl

from brazil_rv.preprocessing.odd_lot_activity import Activity, read_activity_archives

from .artifacts import inventory, sha256_file, write_json_atomic
from .build_store import _require_clean_implementation_commit
from .config import PROJECT_ROOT
from .contract import DEVELOPMENT_END
from .decision_clock import load_session_schedule
from .store import peak_rss_bytes


def activity_rows(
    activity: dict[tuple[date, str, str], Activity], dates: tuple[date, ...]
) -> pl.DataFrame:
    """An absent odd-lot quote is zero only beside an observed regular quote.

    The complete annual parser has already validated the archive structure. No
    source-date quote is available until the next session in the bounded calendar.
    """
    following = dict(zip(dates[:-1], dates[1:], strict=True))
    rows = []
    for (day, isin, market), regular in activity.items():
        if market != "regular" or day not in following:
            continue
        odd = activity.get((day, isin, "odd_lot"))
        rows.append(
            (
                day,
                following[day],
                isin,
                regular.volume_cents / 100.0,
                odd.volume_cents / 100.0 if odd is not None else 0.0,
            )
        )
    return pl.DataFrame(
        rows,
        schema={
            "source_trade_date": pl.Date,
            "available_date": pl.Date,
            "isin": pl.String,
            "regular_volume_brl": pl.Float64,
            "odd_lot_volume_brl": pl.Float64,
        },
        orient="row",
    ).sort("available_date", "isin")


def build(*, reference_store: Path, output: Path) -> dict[str, object]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    _require_clean_implementation_commit(PROJECT_ROOT, commit)
    manifest_path = reference_store / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    schedule_path = (
        PROJECT_ROOT / "research/configs/v2/b3_session_schedule_reconstructed_v1.csv"
    )
    dates = tuple(
        row.trade_date
        for row in load_session_schedule(schedule_path)
        if row.trade_date <= DEVELOPMENT_END
    )
    records = [
        row
        for row in manifest["metadata"]["cotahist_provenance"]["raw_archives"]
        if int(Path(row["path"]).stem[-4:]) <= DEVELOPMENT_END.year
    ]
    if {int(Path(row["path"]).stem[-4:]) for row in records} != set(range(2009, 2025)):
        raise ValueError("odd-lot archive must cover exactly 2009-2024")
    output.mkdir(parents=True, exist_ok=False)
    frames, audits = [], []
    # Parse and release one annual raw dictionary at a time. Only five scalar
    # columns survive each year; no v1 rolling indicators are materialized.
    for record in records:
        path = Path(record["path"])
        activity, source_audits = read_activity_archives([path])
        if source_audits[0].source_sha256 != record["sha256"]:
            raise ValueError(f"COTAHIST source hash changed: {path}")
        frames.append(activity_rows(activity, dates))
        audits.extend(asdict(item) for item in source_audits)
        del activity
        print(json.dumps({"archive": path.name, "rows": frames[-1].height}), flush=True)
    frame = pl.concat(frames).sort("available_date", "isin")
    archive_path = output / "odd_lot_activity.parquet"
    frame.write_parquet(archive_path)
    result = {
        "schema": "BRAZIL_RV_V2_ODDLOT_ARCHIVE",
        "implementation_commit": commit,
        "reference_manifest_sha256": sha256_file(manifest_path),
        "schedule_sha256": sha256_file(schedule_path),
        "source_end": str(DEVELOPMENT_END),
        "available_end": str(frame.get_column("available_date").max()),
        "last_consumed_source_date": str(frame.get_column("source_trade_date").max()),
        "rows": frame.height,
        "availability": "next session, no same-day use",
        "output": {"path": archive_path.name, "sha256": sha256_file(archive_path)},
        "source_audits": audits,
        "official_validation_accessed": False,
        "test_accessed": False,
        "peak_rss_gib": peak_rss_bytes() / 1024**3,
    }
    if result["peak_rss_gib"] > 8:
        raise RuntimeError("odd-lot derivation exceeded 8 GiB RSS")
    write_json_atomic(output / "manifest.json", result)
    write_json_atomic(output / "artifact_inventory.json", inventory(output))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-store", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build(reference_store=args.reference_store, output=args.output)
    print(json.dumps({key: result[key] for key in ("rows", "peak_rss_gib")}))


if __name__ == "__main__":
    main()
