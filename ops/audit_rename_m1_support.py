"""Bound the remaining native-M1 history work without rebuilding passed sources."""

import json
from pathlib import Path

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data import V2DailyDataset
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.round5_derived import verified

PROJECT = Path(__file__).resolve().parents[1]


def main():
    run = json.loads(
        (PROJECT / "docs/v2_economic_data_scaling_run.json").read_text(encoding="utf8")
    )
    accepted = json.loads(
        (PROJECT / "docs/v2_data_inputs.json").read_text(encoding="utf8")
    )
    store = Path(accepted["store"]["root"])
    manifest = bound_json(
        {
            "path": str(store / "manifest.json"),
            "sha256": accepted["store"]["manifest_sha256"],
        }
    )
    rename = bound_json(run["rename_history_propagation"])
    axis, isins = np.load(store / "date_index.npy"), np.load(store / "isin_index.npy")
    before = np.load(store / "active.npy", mmap_mode="r")
    after = np.load(verified(rename["arrays"]["active"]), mmap_mode="r")
    gain = after & ~before
    mapping = pl.read_parquet(
        store / manifest["tables"]["native_fast_security_mapping"]["path"]
    )
    native = {r["store_name_index"]: r["fast_index"] for r in mapping.to_dicts()}
    present = np.load(store / "fast_present.npy", mmap_mode="r")
    patch = np.load(store / "fast_patch_mask.npy", mmap_mode="r")
    valid = np.load(store / "fast_patch_valid.npy", mmap_mode="r")
    values = np.load(store / "fast_patch_values.npy", mmap_mode="r")
    rows = sorted(set(np.where(gain)[0]))
    # Invoke the actual native consumer directly: no model forward, temporary
    # store or duplicated M1 data. Permanent output indices must already match.
    dataset = V2DailyDataset(
        store,
        rows,
        stage="finetune",
        lookback=60,
        include_fast=True,
        include_intraday=False,
    )
    checks = 0
    summaries = []
    for n in np.flatnonzero(gain.any(axis=0)):
        dates = np.flatnonzero(gain[:, n])
        f = native[n]
        assert present[dates, n].all() and patch[dates, f].any(axis=-1).all()
        for t in dates:
            actual_values, actual_valid, actual_patch, names, positions, _ = (
                dataset._fast(int(t))
            )
            j = int(np.flatnonzero(names == n)[0])
            np.testing.assert_array_equal(
                actual_values[j], np.where(valid[t, f], values[t, f], 0)
            )
            np.testing.assert_array_equal(actual_valid[j], valid[t, f])
            np.testing.assert_array_equal(actual_patch[j], patch[t, f])
            assert positions[j] == patch[t, f].sum()
            checks += (
                actual_values[j].size + actual_valid[j].size + actual_patch[j].size
            )
        summaries.append(
            {
                "isin": str(isins[n]),
                "eligible_days_gained": len(dates),
                "native_streams_present": int(present[dates, n].sum()),
                "first_date": str(axis[dates[0]]),
                "first_20_valid_channels": valid[dates[:20], f]
                .sum(axis=(0, 1))
                .tolist(),
                "all_gained_valid_channels": valid[dates, f].sum(axis=(0, 1)).tolist(),
                "native_first20_scaled_return_days": int(
                    valid[dates[:20], f, :, 0].any(axis=-1).sum()
                ),
                "native_first20_clock_volume_days": int(
                    valid[dates[:20], f, :, 3].any(axis=-1).sum()
                ),
            }
        )
    source_audit = pl.read_parquet(
        store / manifest["tables"]["m1_source_audit"]["path"]
    )
    linked = pl.read_parquet(verified(rename["history_mapping"]))
    affected = [
        str(isins[r[k]])
        for r in linked.to_dicts()
        for k in ("predecessor_index", "successor_index")
    ]
    selected_sources = source_audit.filter(pl.col("isin").is_in(affected)).to_dicts()
    dataset.store.close()
    output = Path(run["root"]) / "rename_m1_support_audit.json"
    assert not output.exists()
    write_json_atomic(
        output,
        {
            "schema": "RENAME_M1_SUPPORT_AUDIT_V1",
            "parent": accepted["store"],
            "history": run["rename_history_propagation"],
            "source_mapping": binding(
                store / manifest["tables"]["native_fast_security_mapping"]["path"]
            ),
            "source_audit": binding(
                store / manifest["tables"]["m1_source_audit"]["path"]
            ),
            "selected_sources": selected_sources,
            "summaries": summaries,
            "actual_consumer_cells": checks,
            "mismatches": 0,
            "reproducer": binding(Path(__file__)),
            "scope": "Existing current-day M1 source and permanent-index routing checked on all88 restored eligible days. Early missing sigma/clock-volume channels expose unpropagated predecessor history; this diagnostic is not a completed M1 repair or full source audit.",
            "next": "Reconstruct only affected M1 source windows with admitted predecessor history, unchanged same-clock/return-consistency support and decision cutoff. Preserve source-assignment bounds and pre-event arrays; do not use future closes or raw present-day observations.",
        },
    )
    print(json.dumps({"cells": checks, "summary": summaries}), flush=True)


if __name__ == "__main__":
    main()
