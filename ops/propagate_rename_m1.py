"""Bounded original-M1 reconstruction at the two admitted native renames."""

import json
import shutil
import time
from pathlib import Path

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.build_store import stream_intraday_from_assignments
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.decision_clock import load_session_schedule
from brazil_rv.v2.round5_derived import verified

PROJECT = Path(__file__).resolve().parents[1]


def main():
    clock = time.perf_counter()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    accepted = json.loads((PROJECT / "docs/v2_data_inputs.json").read_text())["store"]
    store = Path(accepted["root"])
    manifest = bound_json(
        {"path": str(store / "manifest.json"), "sha256": accepted["manifest_sha256"]}
    )
    history = bound_json(run["rename_history_propagation"])
    source_audit = bound_json(run["rename_m1_support_audit"])
    root = Path(run["root"]) / "rename_m1_propagation"
    root.mkdir(exist_ok=False)
    shutil.copyfile(__file__, root / "executed_reproducer.py")
    dates, isins = np.load(store / "date_index.npy"), np.load(store / "isin_index.npy")
    links = pl.read_parquet(verified(history["history_mapping"]))
    schedule_path = store / manifest["tables"]["b3_session_schedule"]["path"]
    schedule = {s.trade_date: s for s in load_session_schedule(schedule_path)}
    assignment_source = next(
        s
        for s in manifest["sources"]
        if Path(s["path"]).name == "xp_accepted_source_assignments_v1.parquet"
    )
    assignments = pl.read_parquet(verified(assignment_source))
    mapping = pl.read_parquet(
        store / manifest["tables"]["native_fast_security_mapping"]["path"]
    )
    fast_index = {r["store_name_index"]: r["fast_index"] for r in mapping.to_dicts()}
    old_sigma = np.load(store / "target_scale_sigma.npy", mmap_mode="r")
    new_sigma = np.load(
        verified(history["arrays"]["target_scale_sigma"]), mmap_mode="r"
    )
    old_boundary = np.load(
        store / "intraday_unit_or_unresolved_boundary_mask.npy", mmap_mode="r"
    )
    completed = (
        np.load(store / "action_has_action.npy")
        | ~np.load(store / "action_session_resolved.npy")
        | (np.load(store / "action_successor_index.npy") != np.arange(len(isins)))
    )
    raw_close = np.load(store / "raw_close.npy", mmap_mode="r")
    observed = np.load(store / "observed.npy", mmap_mode="r")
    q = np.load(store / "action_shares_per_prior_share.npy", mmap_mode="r")
    cash = np.load(store / "action_cash_per_prior_share.npy", mmap_mode="r")
    resolved = np.load(store / "action_session_resolved.npy", mmap_mode="r")
    old_fast = {
        k: np.load(store / f"{k}.npy", mmap_mode="r")
        for k in (
            "fast_patch_values",
            "fast_patch_valid",
            "fast_patch_mask",
            "fast_last_price_age_minutes",
            "fast_last_price_age_valid",
        )
    }
    results = []
    deltas = {}
    for link in links.to_dicts():
        pred, succ, effect = (
            link[k] for k in ("predecessor_index", "successor_index", "effective_index")
        )
        if succ not in fast_index:
            continue
        start, stop = max(0, effect - 45), min(len(dates), effect + 45)
        rows = np.arange(start, stop)
        names = [str(isins[pred]), str(isins[succ])]
        local_dates = dates[rows].astype(object).tolist()
        local_sessions = tuple(schedule[d] for d in local_dates)
        local_assignments = assignments.filter(pl.col("isin").is_in(names))
        assert local_assignments.height == 2
        source = [s for s in source_audit["selected_sources"] if s["isin"] in names]
        assert len({s["source_file"] for s in source}) == 1
        # Only bounded dated rows in 2023/2024 original COTAHIST derivatives.
        daily_sources = [
            s
            for s in manifest["sources"]
            if Path(s["path"]).name
            in {f"equities_daily_{d.year}.parquet" for d in local_dates}
        ]
        daily = pl.concat(
            [
                pl.scan_parquet(s["path"])
                .filter(
                    pl.col("isin").is_in(names),
                    pl.col("trade_date").is_between(local_dates[0], local_dates[-1]),
                )
                .select("isin", "trade_date")
                .collect()
                for s in daily_sources
            ]
        ).unique()
        local_links = pl.DataFrame(
            [
                {
                    **link,
                    "predecessor_index": 0,
                    "successor_index": 1,
                    "effective_index": effect - start,
                    "known_index": link["known_index"] - start,
                }
            ]
        )
        prices = np.round(
            np.asarray(raw_close[rows][:, [pred, succ]], dtype=np.float64), 2
        )
        seen = observed[rows][:, [pred, succ]]
        references = np.full(prices.shape, np.nan)
        ending = q[rows][:, [pred, succ]] * prices + cash[rows][:, [pred, succ]]
        usable = seen[1:] & seen[:-1] & resolved[rows[1:]][:, [pred, succ]]
        references[1:] = np.where(usable, np.log(ending[1:] / prices[:-1]), np.nan)
        amended_references = references.copy()
        e = effect - start
        amended_references[e, 0] = np.log(prices[e, 1] / prices[e - 1, 0])

        def build(label, sigma, refs, links=None, end=None):
            count = len(rows) if end is None else end
            workspace = root / names[1] / label
            workspace.mkdir(parents=True)
            return stream_intraday_from_assignments(
                local_assignments,
                daily,
                local_sessions[:count],
                names,
                sigma_asof=sigma[:count],
                kept_rows=np.arange(count),
                workspace=workspace,
                official_log_return=refs[:count],
                completed_action_boundary=completed[rows[:count]][:, [pred, succ]],
                same_day_boundary=old_boundary[rows[:count]][:, [pred, succ]],
                history_links=links,
            )

        baseline = build("control", old_sigma[rows][:, [pred, succ]], references)
        repaired = build(
            "continued",
            new_sigma[rows][:, [pred, succ]],
            amended_references,
            local_links,
        )
        f = int(
            repaired.native_mapping.filter(pl.col("isin") == names[1])[0, "fast_index"]
        )
        p = int(
            repaired.native_mapping.filter(pl.col("isin") == names[0])[0, "fast_index"]
        )
        # The original native successor and the fully warmed predecessor control
        # must reproduce the sealed store exactly before any interpretation.
        controls = 0
        for k, original in baseline.native_arrays.items():
            np.testing.assert_array_equal(
                original[:, f], old_fast[k][rows, fast_index[succ], : original.shape[2]]
            )
            np.testing.assert_array_equal(
                original[21:e, p],
                old_fast[k][rows[21:e], fast_index[pred], : original.shape[2]],
            )
            np.testing.assert_array_equal(repaired.native_arrays[k][:e], original[:e])
            np.testing.assert_array_equal(
                repaired.native_arrays[k][:, p], original[:, p]
            )
            assert not old_fast[k][rows, fast_index[succ], original.shape[2] :].any()
            controls += original[:, f].size + original[21:e, p].size
            actual = repaired.native_arrays[k][:, f]
            equal = (
                (actual == original[:, f])
                | (np.isnan(actual) & np.isnan(original[:, f]))
                if actual.dtype.kind == "f"
                else actual == original[:, f]
            )
            changed = np.where(~equal)
            deltas[f"{names[1]}__{k}__indices"] = np.column_stack(
                (
                    rows[changed[0]],
                    np.full(len(changed[0]), fast_index[succ]),
                    *changed[1:],
                )
            )
            deltas[f"{names[1]}__{k}__values"] = actual[changed]
        # Delete all future source/quote decisions from a second bounded replay.
        cutoff = e + 8
        prefix = build(
            "causal_prefix",
            new_sigma[rows][:, [pred, succ]],
            amended_references,
            local_links,
            cutoff,
        )
        for k in repaired.native_arrays:
            np.testing.assert_array_equal(
                prefix.native_arrays[k], repaired.native_arrays[k][:cutoff]
            )
        for k in ("values", "valid", "support_fraction", "source_age_sessions"):
            np.testing.assert_array_equal(
                getattr(prefix.result, k), getattr(repaired.result, k)[:cutoff]
            )
        raw = {
            k: np.asarray(getattr(repaired.result, k))
            for k in (
                "values",
                "valid",
                "support_fraction",
                "source_age_sessions",
                "return_consistent",
                "session_close",
                "session_close_valid",
                "decision_mark",
                "decision_mark_valid",
                "realized_daily_vol",
                "fast_present",
            )
        }
        raw.update(date_indices=rows, name_indices=np.array([pred, succ]))
        raw_path = root / f"{names[1]}_raw_intraday.npz"
        np.savez_compressed(raw_path, **raw)
        old_valid = baseline.native_arrays["fast_patch_valid"][:, f]
        valid = repaired.native_arrays["fast_patch_valid"][:, f]
        first = slice(e, e + 20)
        summary = {
            "isin": names[1],
            "source_receipts": source,
            "control_cells": controls,
            "read_start": str(dates[start]),
            "read_end": str(dates[stop - 1]),
            "old_first20_channels": old_valid[first].sum(axis=(0, 1)).tolist(),
            "new_first20_channels": valid[first].sum(axis=(0, 1)).tolist(),
            "gained_native_cells": int((valid & ~old_valid).sum()),
            "lost_native_cells": int((old_valid & ~valid).sum()),
            "raw_intraday_gains": int(
                (repaired.result.valid[e:, 1] & ~baseline.result.valid[e:, 1]).sum()
            ),
            "raw_intraday_losses": int(
                (baseline.result.valid[e:, 1] & ~repaired.result.valid[e:, 1]).sum()
            ),
            "raw_intraday": binding(raw_path),
            "future_deletion_prefix_cells_exact": True,
        }
        results.append(summary)
        print(json.dumps(summary), flush=True)
    delta = root / "native_deltas.npz"
    np.savez_compressed(delta, **deltas)
    write_json_atomic(
        root / "manifest.json",
        {
            "schema": "RENAME_M1_PROPAGATION_V1",
            "parent": accepted,
            "history": run["rename_history_propagation"],
            "source_audit": run["rename_m1_support_audit"],
            "assignments": assignment_source,
            "schedule": binding(schedule_path),
            "native_deltas": binding(delta),
            "results": results,
            "seconds": time.perf_counter() - clock,
            "reproducer": binding(root / "executed_reproducer.py"),
            "implementation": binding(
                PROJECT / "research/src/brazil_rv/v2/build_store.py"
            ),
            "heldout_access": False,
            "forecast_scoring": False,
            "status": "Bounded native repair and raw scalar amendments; final derived-store assembly remains separate.",
        },
    )


if __name__ == "__main__":
    main()
