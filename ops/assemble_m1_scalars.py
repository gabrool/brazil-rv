"""Recover bounded M1 scalar cross-sections, reusing admitted rename outputs."""

import argparse
import json
import shutil
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from brazil_rv.preprocessing.io import (
    SOURCE_COLUMNS,
    dense_grid,
    validate_session_bars,
    validate_physical_source_identity,
)
from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.contract import INTRADAY_DAILY_FEATURES
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.decision_clock import load_session_schedule
from brazil_rv.v2.feature_spec import feature_specs, transform_feature_panel_into
from brazil_rv.v2.intraday_features import build_intraday_daily_features
from brazil_rv.v2.round5_derived import verified

PROJECT = Path(__file__).resolve().parents[1]
FIELDS = {
    "values": "full_intraday_values",
    "valid": "full_intraday_valid",
    "support_fraction": "full_intraday_support",
    "source_age_sessions": "full_intraday_source_age",
    "return_consistent": "full_m1_return_consistent",
    "entry": "full_to_close_entry",
    "entry_valid": "full_to_close_entry_valid",
    "session_close": "full_m1_session_close",
    "session_close_valid": "full_m1_session_close_valid",
    "realized_daily_vol": "full_intraday_realized",
    "fast_present": "full_fast_present",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualify-control", action="store_true")
    parser.add_argument("--qualify-date", default="2024-11-18")
    args = parser.parse_args()
    started = time.perf_counter()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    accepted = json.loads((PROJECT / "docs/v2_data_inputs.json").read_text())["store"]
    store = Path(accepted["root"])
    manifest = bound_json(
        {"path": str(store / "manifest.json"), "sha256": accepted["manifest_sha256"]}
    )
    rename = bound_json(run["rename_history_propagation"])
    m1 = bound_json(run["rename_m1_propagation"])
    root = Path(run["root"]) / "m1_scalar_assembly"
    original_root = root
    if args.qualify_control:
        root = root / "qualified_control"
    root.mkdir(exist_ok=args.qualify_control)
    if not (root / "executed_reproducer.py").exists():
        shutil.copyfile(__file__, root / "executed_reproducer.py")
    dates, isins = np.load(store / "date_index.npy"), np.load(store / "isin_index.npy")
    old_active = np.load(store / "active.npy", mmap_mode="r")
    new_active = np.load(verified(rename["arrays"]["active"]), mmap_mode="r")
    links = pl.read_parquet(verified(rename["history_mapping"]))
    all_sessions = {
        s.trade_date: s for s in load_session_schedule(Path(m1["schedule"]["path"]))
    }
    assignments = pl.read_parquet(verified(m1["assignments"])).sort("security_id")
    q = np.load(store / "action_shares_per_prior_share.npy", mmap_mode="r")
    cash = np.load(store / "action_cash_per_prior_share.npy", mmap_mode="r")
    resolved = np.load(store / "action_session_resolved.npy", mmap_mode="r")
    successor = np.load(store / "action_successor_index.npy", mmap_mode="r")
    has_action = np.load(store / "action_has_action.npy", mmap_mode="r")
    seen = np.load(store / "observed.npy", mmap_mode="r")
    close = np.load(store / "raw_close.npy", mmap_mode="r")
    boundary = np.load(
        store / "intraday_unit_or_unresolved_boundary_mask.npy", mmap_mode="r"
    )
    lookup = {v: j for j, v in enumerate(isins)}
    specs = feature_specs("intraday", INTRADAY_DAILY_FEATURES)
    reports = []
    for effect in sorted(set(links["effective_index"])):
        if args.qualify_control and str(dates[effect]) != args.qualify_date:
            continue
        relevant = links.filter(pl.col("effective_index") == effect)
        changed = (old_active != new_active)[
            :, relevant["successor_index"].to_numpy()
        ].any(axis=1)
        last = max(effect + 20, int(np.flatnonzero(changed)[-1]))
        start, stop = max(0, effect - 45), min(len(dates), last + 1)
        rows = np.arange(start, stop)
        keep = np.arange(effect - 1 - start, len(rows))
        calendar = dates[rows].astype(object).tolist()
        sessions = tuple(all_sessions[d] for d in calendar)
        folder = root / str(dates[effect])
        folder.mkdir()
        shutil.copyfile(__file__, folder / "executed_reproducer.py")
        shape = (len(rows), len(isins))
        panels = {}
        for key in FIELDS:
            dims = (
                (*shape, len(INTRADAY_DAILY_FEATURES))
                if key in {"values", "valid", "support_fraction", "source_age_sessions"}
                else shape
            )
            dtype = (
                bool
                if key
                in {
                    "valid",
                    "return_consistent",
                    "entry_valid",
                    "session_close_valid",
                    "fast_present",
                }
                else np.float32
            )
            fill = (
                False
                if dtype is bool
                else (
                    -1
                    if key == "source_age_sessions"
                    else (0 if key in {"values", "support_fraction"} else np.nan)
                )
            )
            panels[key] = np.full(dims, fill, dtype=dtype)
        if args.qualify_control:
            with np.load(
                original_root / str(dates[effect]) / "raw_control.npz"
            ) as saved:
                panels = {key: saved[key].copy() for key in FIELDS}
        daily_sources = [
            s
            for s in manifest["sources"]
            if Path(s["path"]).name
            in {f"equities_daily_{d.year}.parquet" for d in calendar}
        ]
        daily = pl.concat(
            [
                pl.scan_parquet(s["path"])
                .filter(pl.col("trade_date").is_between(calendar[0], calendar[-1]))
                .select("isin", "trade_date")
                .collect()
                for s in daily_sources
            ]
        ).unique()
        original_dates = {
            key[0]: set(frame["trade_date"]) for key, frame in daily.group_by("isin")
        }
        # The prior native repair has exact baseline scalar workspaces for both
        # physical rename files, including ALOS's separately recovered tail.
        reuse = {}
        for rec in m1["results"]:
            pair = next(
                r
                for r in links.to_dicts()
                if isins[r["successor_index"]] == rec["isin"]
            )
            original_start = int(
                np.searchsorted(dates, np.datetime64(rec["read_start"]))
            )
            original_end = (
                int(np.searchsorted(dates, np.datetime64(rec["read_end"]))) + 1
            )
            work = Path(rec["raw_intraday"]["path"]).parent / rec["isin"] / "control"
            for j, n in enumerate([pair["predecessor_index"], pair["successor_index"]]):
                reuse[n] = [(original_start, original_end, work, j)]
            # The prior scale-only tail lacks scalar return-consistency inputs.
            # Reuse only the fully source-qualified original workspaces.
        schedule = pl.DataFrame(
            {
                "trade_date": calendar,
                "date_idx": range(len(rows)),
                "open_minute": [
                    s.continuous_open.hour * 60 + s.continuous_open.minute
                    for s in sessions
                ],
                "minute_count": [
                    (s.continuous_close.hour * 60 + s.continuous_close.minute)
                    - (s.continuous_open.hour * 60 + s.continuous_open.minute)
                    for s in sessions
                ],
            }
        )
        max_minutes = max(schedule["minute_count"])
        receipts = []
        for group in assignments.partition_by("source_file"):
            needs = []
            for assignment in group.to_dicts():
                n = lookup[assignment["isin"]]
                if args.qualify_control and assignment["isin"] != "BRALOSACNOR5":
                    continue
                first, end = (
                    assignment.get("first_overlap_date"),
                    assignment.get("last_overlap_date"),
                )
                if isinstance(first, str):
                    first = date.fromisoformat(first)
                if isinstance(end, str):
                    end = date.fromisoformat(end)
                allowed = {
                    d
                    for d in original_dates.get(assignment["isin"], ())
                    if (first is None or d >= first) and (end is None or d <= end)
                }
                if not allowed:
                    continue
                covered_range = set().union(
                    *(set(range(a, b)) for a, b, _, _ in reuse.get(n, []))
                )
                required = {start + calendar.index(d) for d in allowed}
                if n in reuse and required <= covered_range:
                    covered = set()
                    for a, b, work, j in reuse[n]:
                        selected = rows[(rows >= a) & (rows < b)]
                        for key, filename in FIELDS.items():
                            array = np.load(work / (filename + ".npy"), mmap_mode="r")
                            panels[key][selected - start, n] = array[selected - a, j]
                        covered.update(selected.tolist())
                    required = {start + calendar.index(d) for d in allowed}
                    assert required <= covered, (
                        assignment["isin"],
                        sorted(required - covered),
                    )
                else:
                    needs.append((assignment, n, allowed))
            if not needs:
                continue
            path = Path(group[0, "source_file"])
            source = (
                pl.scan_parquet(path)
                .filter(
                    pl.col("ts_exchange")
                    .dt.date()
                    .is_between(calendar[0], calendar[-1])
                )
                .select(SOURCE_COLUMNS)
                .collect()
            )
            validate_physical_source_identity(group, source, path)
            for assignment, n, allowed in needs:
                bars = source.with_columns(
                    pl.col("ts_exchange").dt.date().alias("trade_date"),
                    (
                        pl.col("ts_exchange").dt.hour().cast(pl.Int16) * 60
                        + pl.col("ts_exchange").dt.minute()
                    ).alias("clock"),
                )
                bars = (
                    bars.filter(pl.col("trade_date").is_in(tuple(allowed)))
                    .join(schedule, on="trade_date")
                    .with_columns(
                        (pl.col("clock") - pl.col("open_minute"))
                        .cast(pl.Int16)
                        .alias("minute_idx")
                    )
                    .filter(
                        pl.col("minute_idx") >= 0,
                        pl.col("minute_idx") < pl.col("minute_count"),
                    )
                    .sort("ts_exchange")
                )
                validate_session_bars(bars, path)
                grid, observed = dense_grid(bars, len(rows), max_minutes)
                support = observed.any(axis=1)[:, None]
                prices = np.round(np.asarray(close[rows], dtype=np.float64), 2)
                terminal = np.take_along_axis(prices, successor[rows], axis=1)[:, n]
                terminal_seen = np.take_along_axis(seen[rows], successor[rows], axis=1)[
                    :, n
                ]
                refs = np.full((len(rows), 1), np.nan)
                valid_return = (
                    seen[rows[:-1], n] & terminal_seen[1:] & resolved[rows[1:], n]
                )
                refs[1:, 0] = np.where(
                    valid_return,
                    np.log(
                        (q[rows[1:], n] * terminal[1:] + cash[rows[1:], n])
                        / prices[:-1, n]
                    ),
                    np.nan,
                )
                result = build_intraday_daily_features(
                    *[grid[:, None, :, j] for j in range(5)],
                    observed[:, None],
                    volume_valid=observed[:, None],
                    session_valid=support,
                    sessions=sessions,
                    official_log_return=refs,
                    completed_action_boundary=(
                        has_action[rows, n]
                        | ~resolved[rows, n]
                        | (successor[rows, n] != n)
                    )[:, None],
                    same_day_boundary=boundary[rows, n, None],
                )
                entry_minute = np.array(
                    [
                        s.decision_time.hour * 60
                        + s.decision_time.minute
                        - s.continuous_open.hour * 60
                        - s.continuous_open.minute
                        for s in sessions
                    ]
                )
                entry = grid[np.arange(len(rows)), entry_minute, 0]
                panels["entry"][:, n] = entry
                panels["entry_valid"][:, n] = (
                    observed[np.arange(len(rows)), entry_minute]
                    & np.isfinite(entry)
                    & (entry > 0)
                )
                for key in FIELDS:
                    if key not in {"entry", "entry_valid"}:
                        panels[key][:, n] = getattr(result, key)[:, 0]
                receipts.append(
                    {
                        "isin": assignment["isin"],
                        "path": str(path),
                        "original_dates": len(allowed),
                        "bars": bars.height,
                    }
                )
        np.savez_compressed(folder / "raw_control.npz", date_indices=rows, **panels)
        write_json_atomic(
            folder / "source_scope.json",
            {
                "reads": receipts,
                "reused_names": list(reuse),
                "start": str(dates[start]),
                "end": str(dates[stop - 1]),
            },
        )
        values = np.empty((len(keep), len(isins), 20), dtype=np.float32)
        valid = np.empty(values.shape, dtype=bool)
        transform_feature_panel_into(
            panels["values"],
            panels["valid"],
            old_active[rows],
            specs,
            values,
            valid,
            source_rows=keep,
        )
        expected_values = np.load(store / "intraday_values.npy", mmap_mode="r")[
            rows[keep]
        ]
        expected_valid = np.load(store / "intraday_valid.npy", mmap_mode="r")[
            rows[keep]
        ]
        mismatches = {
            "valid": int((valid != expected_valid).sum()),
            "values": int((values != expected_values).sum()),
        }
        np.savez_compressed(
            folder / "control_transforms.npz", values=values, valid=valid
        )
        write_json_atomic(folder / "control_comparison.json", mismatches)
        reports.append({"date": str(dates[effect]), **mismatches})
        print(json.dumps(reports[-1]), flush=True)
    write_json_atomic(
        root
        / (
            "reconstruction_" + args.qualify_date + ".json"
            if args.qualify_control
            else "reconstruction.json"
        ),
        {
            "parent": accepted,
            "m1": run["rename_m1_propagation"],
            "reports": reports,
            "seconds": time.perf_counter() - started,
            "reproducer": binding(root / "executed_reproducer.py"),
        },
    )


if __name__ == "__main__":
    main()
