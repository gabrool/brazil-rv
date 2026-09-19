"""Trace audited original EWZ returns through decision clocks to stored fields."""

from datetime import datetime, time, timezone
import json
from pathlib import Path
import time as timer
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    started = timer.perf_counter()
    pointer = json.loads(
        (PROJECT / "docs/v2_economic_data_scaling_run.json").read_text()
    )
    root = Path(pointer["root"])
    output = root / "ewz_decision_audit.json"
    if output.exists():
        raise FileExistsError(output)
    audit_path = root / "us_source_audit.json"
    audit = json.loads(audit_path.read_text())
    source = bound_json(audit["source_manifest"])
    us_record = source["outputs"]["us_returns.parquet"]
    assert sha256_file(Path(us_record["path"])) == us_record["sha256"]
    returns = (
        pl.read_parquet(us_record["path"])
        .filter(pl.col("series") == "us_EWZ")
        .sort("reference_date")
        .to_dicts()
    )
    accepted = json.loads((PROJECT / "docs/v2_data_inputs.json").read_text())
    store = Path(accepted["store"]["root"])
    manifest = bound_json(
        {
            "path": str(store / "manifest.json"),
            "sha256": accepted["store"]["manifest_sha256"],
        }
    )
    dates = np.load(store / "date_index.npy").astype(object)
    assert str(dates[-1]) <= "2024-12-31"
    active = np.load(store / "active.npy", mmap_mode="r")
    arrays = {
        suffix: np.load(store / f"sidecar_cross_market_{suffix}.npy", mmap_mode="r")
        for suffix in ("values", "valid", "age_sessions")
    }
    times = np.array([r["available_at"].timestamp() for r in returns])
    assert np.all(np.diff(times) > 0)
    results = []
    for horizon in (1, 5):
        f = manifest["feature_names"]["sidecar_cross_market"].index(
            f"shock_ewz_{horizon}"
        )
        known = np.zeros(len(dates), bool)
        values = np.zeros(len(dates), np.float32)
        ages = np.full(len(dates), -1, np.float32)
        same_date = []
        for i, day in enumerate(dates):
            cutoff = (
                datetime.combine(day, time(15, 45), ZoneInfo("America/Sao_Paulo"))
                .astimezone(timezone.utc)
                .timestamp()
            )
            last = int(np.searchsorted(times, cutoff, side="right")) - 1
            if last < horizon - 1:
                continue
            chunk = returns[last - horizon + 1 : last + 1]
            if any(
                a["reference_date"] != b["previous_date"]
                for a, b in zip(chunk, chunk[1:])
            ):
                continue
            known[i] = True
            values[i] = sum(row["log_return"] for row in chunk)
            # Count B3 sessions from the first session on/after the source date.
            ages[i] = np.count_nonzero(
                (dates >= chunk[-1]["reference_date"]) & (dates < day)
            )
            if chunk[-1]["reference_date"] == day:
                same_date.append(str(day))
        expected_valid = active & known[:, None]
        expected_values = np.where(expected_valid, values[:, None], 0).astype(
            np.float32
        )
        expected_ages = np.where(expected_valid, ages[:, None], -1).astype(np.float32)
        np.testing.assert_array_equal(arrays["valid"][..., f], expected_valid)
        np.testing.assert_array_equal(arrays["values"][..., f], expected_values)
        np.testing.assert_array_equal(arrays["age_sessions"][..., f], expected_ages)
        results.append(
            {
                "field": f"shock_ewz_{horizon}",
                "known_sessions": int(known.sum()),
                "active_valid_cells": int(expected_valid.sum()),
                "same_date_sessions": same_date,
            }
        )
    result = {
        "us_source_audit": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
        "store": accepted["store"],
        "reproducer": {
            "path": str(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
        "fields": results,
        "mismatches": 0,
        "method": "independent first-15:45 UTC comparison, exactly linked 1/5 US-return endpoints, source-date age and stored float32 values/masks/ages; no production shock alignment",
        "scope": "EWZ common shocks only; ADR gaps deliberately require paired prior B3/US endpoints and are a separate contract",
        "changed_arrays": 0,
        "elapsed_seconds": timer.perf_counter() - started,
    }
    write_json_atomic(output, result)
    print(json.dumps(result["fields"]))


if __name__ == "__main__":
    main()
