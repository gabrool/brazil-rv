"""Input-only archive audit of the repaired multi-day intraday scalar contract."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from .artifacts import inventory, sha256_file, write_json_atomic
from .build_store import (
    _require_clean_implementation_commit,
    stream_intraday_from_assignments,
)
from .config import PROJECT_ROOT
from .contract import DEVELOPMENT_END, INTRADAY_DAILY_FEATURES
from .corporate_actions import (
    align_verified_action_terms,
    align_decision_known_action_terms,
    infer_cotahist_action_terms,
)
from .data_foundation import load_cotahist, panel_from_daily
from .decision_clock import load_session_schedule, next_session_decision_cutoffs
from .feature_spec import feature_specs, transform_feature_panel_into
from .intraday_features import decision_action_boundaries, shareholder_reference_returns
from .store import close_memmap, peak_rss_bytes
from .universe import build_daily_universe


def audit(*, reference_store: Path, output: Path) -> dict[str, object]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    _require_clean_implementation_commit(PROJECT_ROOT, commit)
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = reference_store / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    records = manifest["sources"]
    cotahist_records = [
        row
        for row in records
        if Path(row["path"]).name.startswith("equities_daily_")
        and int(Path(row["path"]).stem.rsplit("_", 1)[1]) <= DEVELOPMENT_END.year
    ]
    assignment_record = next(
        row
        for row in records
        if Path(row["path"]).name == "xp_accepted_source_assignments_v1.parquet"
    )
    source_bindings = [*cotahist_records, assignment_record]
    for record in source_bindings:
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"archive source identity changed: {record['path']}")
    assignments = pl.read_parquet(assignment_record["path"])
    m1_isins = assignments.get_column("isin").to_list()
    daily = load_cotahist(
        [Path(row["path"]) for row in cotahist_records],
        v1_isins=m1_isins,
        end_date=DEVELOPMENT_END,
    )
    schedule_path = (
        PROJECT_ROOT / "research/configs/v2/b3_session_schedule_reconstructed_v1.csv"
    )
    schedule = tuple(
        row
        for row in load_session_schedule(schedule_path)
        if row.trade_date <= DEVELOPMENT_END
    )
    dates = tuple(row.trade_date for row in schedule)
    axis_path = reference_store / "isin_index.npy"
    if sha256_file(axis_path) != manifest["indices"][axis_path.name]["sha256"]:
        raise ValueError("reference identity axis changed")
    isins = tuple(np.load(axis_path, allow_pickle=False).tolist())
    source_dates = set(daily.get_column("trade_date").to_list())
    panel = panel_from_daily(
        daily,
        dates=dates,
        isins=isins,
        source_session_complete=[value in source_dates for value in dates],
    )
    universe = build_daily_universe(
        panel.close_brl,
        panel.volume_brl,
        panel.observed,
        trade_observed=panel.trade_observed,
        activity_valid=panel.activity_valid,
        source_session_complete=panel.source_session_complete,
    )
    inferred = infer_cotahist_action_terms(
        panel.dates,
        panel.isins,
        panel.close_brl,
        panel.quantity,
        panel.trades,
        panel.distribution_number,
        panel.observed,
        universe.active,
    )
    actions = align_verified_action_terms(
        inferred.terms,
        panel.dates,
        panel.isins,
        coverage_resolved=inferred.coverage_resolved,
    )
    reference = shareholder_reference_returns(panel.close_brl, panel.observed, actions)
    completed_boundary = actions.has_action | ~actions.session_resolved
    daily_actions = align_decision_known_action_terms(
        inferred.terms,
        panel.dates,
        panel.isins,
        coverage_resolved=inferred.coverage_resolved,
        decision_timestamps=next_session_decision_cutoffs(schedule),
    )
    same_day_boundary = decision_action_boundaries(
        panel.open_brl,
        panel.close_brl,
        panel.observed,
        daily_actions.session_resolved,
        inferred.terms,
        schedule,
        isins,
    )
    del daily_actions
    selected = np.flatnonzero(panel.dates >= np.datetime64(date(2024, 1, 1)))
    m1_indices = np.asarray([isins.index(isin) for isin in m1_isins])
    # Only 20 prior rows are needed by the M1 estimators; daily inference above
    # retains the complete 2009 history so adjustment classification is identical.
    start = max(int(selected[0]) - 21, 0)
    local_daily = daily.filter(pl.col("trade_date") >= dates[start])
    local_selected = selected - start
    print(
        json.dumps(
            {
                "stage": "M1 archive scan",
                "names": len(m1_isins),
                "sessions": len(dates) - start,
            }
        ),
        flush=True,
    )
    with tempfile.TemporaryDirectory(
        prefix=".intraday-archive-", dir=output.parent, ignore_cleanup_errors=True
    ) as scratch:
        streamed = stream_intraday_from_assignments(
            assignments,
            local_daily,
            schedule[start:],
            isins,
            # Native patches are discarded. This positive constant affects their
            # unused scaling only; all reported scalar values/masks use raw M1.
            sigma_asof=np.full(
                (len(dates) - start, len(isins)), 0.02, dtype=np.float32
            ),
            kept_rows=local_selected,
            workspace=Path(scratch),
            official_log_return=reference[start:],
            completed_action_boundary=completed_boundary[start:],
            same_day_boundary=same_day_boundary[start:],
        )
        raw = streamed.result
        shape = (len(selected), len(isins), len(INTRADAY_DAILY_FEATURES))
        transformed = np.zeros(shape, dtype=np.float32)
        valid = np.zeros(shape, dtype=np.bool_)
        transform_feature_panel_into(
            raw.values,
            raw.valid,
            universe.active[start:],
            feature_specs("intraday", INTRADAY_DAILY_FEATURES, minimum_rank_names=20),
            transformed,
            valid,
            source_rows=local_selected,
            minimum_rank_names=20,
        )
        active = universe.active[selected][:, m1_indices]
        rows = []
        for feature, name in enumerate(INTRADAY_DAILY_FEATURES):
            observed = raw.valid[local_selected][:, m1_indices, feature]
            normalized = valid[:, m1_indices, feature]
            support = raw.support_fraction[local_selected][:, m1_indices, feature]
            rows.append(
                {
                    "feature": name,
                    "raw_valid_name_days": int(observed.sum()),
                    "possible_m1_name_days": int(observed.size),
                    "raw_fraction": float(observed.mean()),
                    "active_valid_name_days": int((observed & active).sum()),
                    "active_m1_name_days": int(active.sum()),
                    "active_raw_fraction": float(
                        (observed & active).sum() / active.sum()
                    ),
                    "transformed_valid_name_days": int(normalized.sum()),
                    "transformed_fraction": float(normalized.mean()),
                    "sessions_with_20_raw_active_names": int(
                        ((observed & active).sum(axis=1) >= 20).sum()
                    ),
                    "mean_support_fraction": float(support.mean()),
                }
            )
        close = np.asarray(raw.session_close[local_selected])
        usable = raw.session_close_valid[local_selected] & panel.observed[selected]
        day, name = np.nonzero(usable)
        pl.DataFrame(
            {
                "trade_date": panel.dates[selected[day]].astype("datetime64[ms]"),
                "isin": np.asarray(isins)[name],
                "m1_to_cotahist_close": close[day, name]
                / panel.close_brl[selected[day], name],
                "return_consistent": raw.return_consistent[local_selected][day, name],
                "inferred_action_or_unresolved": completed_boundary[
                    selected[day], name
                ],
            }
        ).with_columns(pl.col("trade_date").cast(pl.Date)).write_parquet(
            output / "level_ratio.parquet"
        )
        streamed.audit.write_parquet(output / "m1_source_audit.parquet")
        source_bindings.extend(
            {"path": str(path), "sha256": sha256_file(path)}
            for path in streamed.source_paths
        )
        consistency_count = int(
            raw.return_consistent[local_selected][:, m1_indices].sum()
        )
        for value in vars(raw).values():
            if isinstance(value, np.memmap):
                close_memmap(value)
        for value in (
            *streamed.native_arrays.values(),
            streamed.to_close_entry,
            streamed.to_close_entry_valid,
        ):
            close_memmap(value)
        del raw, streamed
        gc.collect()
    result = {
        "schema": "BRAZIL_RV_V2_INTRADAY_ARCHIVE_AUDIT",
        "implementation_commit": commit,
        "reference_store": str(reference_store),
        "reference_manifest_sha256": sha256_file(manifest_path),
        "source_end": str(DEVELOPMENT_END),
        "evaluated_year": 2024,
        "sessions": len(selected),
        "m1_names": len(m1_isins),
        "return_consistent_name_days": consistency_count,
        "coverage": rows,
        "sources": source_bindings,
        "schedule_sha256": sha256_file(schedule_path),
        "official_validation_accessed": False,
        "test_accessed": False,
        "peak_rss_gib": peak_rss_bytes() / 1024**3,
    }
    if result["peak_rss_gib"] > 8:
        raise RuntimeError("intraday archive audit exceeded 8 GiB RSS")
    write_json_atomic(output / "coverage.json", result)
    write_json_atomic(output / "artifact_inventory.json", inventory(output))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        reference_store=args.reference_store.resolve(), output=args.output.resolve()
    )
    print(json.dumps({key: result[key] for key in ("coverage", "peak_rss_gib")}))


if __name__ == "__main__":
    main()
