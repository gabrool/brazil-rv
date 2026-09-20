"""Exercise dated target amendments through the real dataset and collator."""

import json
import shutil
import time
from pathlib import Path

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data import V2DailyDataset, collate_v2_daily
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.feature_spec import FeatureSpec, feature_schema_sha256
from brazil_rv.v2.round5_derived import verified
from brazil_rv.v2.store import StoreStaging

PROJECT = Path(__file__).resolve().parents[1]


def main():
    started = time.perf_counter()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    root = Path(run["root"]) / "to_close_clock"
    clock = bound_json(binding(root / "manifest.json"))
    history = bound_json(run["rename_history_propagation"])
    peers = bound_json(run["rename_peer_propagation"])
    native = bound_json(run["rename_m1_propagation"])
    tail = bound_json(run["rename_m1_input_audit"])
    scalars = bound_json(run["m1_scalar_assembly"])
    store = Path(clock["parent"]["root"])
    original = bound_json(
        {
            "path": str(store / "manifest.json"),
            "sha256": clock["parent"]["manifest_sha256"],
        }
    )
    out = root / "consumer"
    out.mkdir()
    shutil.copyfile(__file__, out / "executed_reproducer.py")
    dates, isins = np.load(store / "date_index.npy"), np.load(store / "isin_index.npy")
    mapping = pl.read_parquet(
        store / original["tables"]["native_fast_security_mapping"]["path"]
    )
    links = pl.read_parquet(verified(history["history_mapping"]))
    amendments = {}
    for record in [scalars["deltas"], native["native_deltas"], tail["tail_deltas"]]:
        with np.load(verified(record)) as delta:
            for key in delta.files:
                if key.endswith("__indices"):
                    name = key.split("__")[1]
                    amendments.setdefault(name, []).append(
                        (
                            delta[key].copy(),
                            delta[key.replace("__indices", "__values")].copy(),
                        )
                    )
    with np.load(verified(clock["deltas"])) as delta:
        request = set(delta["target_to_close__indices"][:, 0].tolist())
        for key in delta.files:
            if key.endswith("__indices"):
                amendments.setdefault(key.split("__")[0], []).append(
                    (
                        delta[key].copy(),
                        delta[key.replace("__indices", "__values")].copy(),
                    )
                )
    with np.load(root / "raw_inputs.npz") as saved:
        combinations = np.column_stack((saved["session_minutes"], saved["cutoff"]))
        supported = saved["valid"].any(axis=1)
        for pair in np.unique(combinations[supported], axis=0):
            request.add(
                int(np.flatnonzero(supported & (combinations == pair).all(axis=1))[0])
            )
        request.add(int(np.flatnonzero(supported)[-1]))
    specs = [
        s
        for s in original["metadata"]["feature_schema"]["specifications"]
        if s["family"] in {"slow", "native_fast"}
    ]
    slow_names = [
        "active",
        "slow_values",
        "slow_valid",
        "slow_age_sessions",
        "slow_timestep_valid",
    ]
    slow = {
        key: np.load(
            verified(peers.get("arrays", {}).get(key, history["arrays"].get(key))),
            mmap_mode="r",
        )
        for key in slow_names
    }
    fast_keys = [
        "fast_patch_values",
        "fast_patch_valid",
        "fast_patch_mask",
        "fast_present",
        "fast_last_price_age_minutes",
        "fast_last_price_age_valid",
    ]
    target_keys = ["target_to_close", "target_to_close_valid"]
    arrays = {
        key: np.load(store / (key + ".npy"), mmap_mode="r")
        for key in fast_keys + target_keys
    }
    cells, reports = 0, []
    for day in sorted(request):
        start = day - 59
        assert start >= 0
        rows = np.arange(start, day + 1)
        view = out / str(dates[day])
        expected = {}
        with StoreStaging(view, dates=dates[rows], isins=isins.tolist()) as staging:
            for key, value in slow.items():
                staging.write_array(key, value[rows])
            for key, source in arrays.items():
                value = source[rows].copy()
                for indices, additions in amendments.get(key, []):
                    keep = (indices[:, 0] >= start) & (indices[:, 0] <= day)
                    ix = indices[keep].copy()
                    ix[:, 0] -= start
                    value[tuple(ix.T)] = additions[keep]
                staging.write_array(key, value)
                if key in target_keys + ["fast_present"]:
                    expected[key] = value[-1]
            staging.seal(
                feature_names={
                    k: original["feature_names"][k] for k in ["slow", "native_fast"]
                },
                sources=[
                    binding(root / "manifest.json"),
                    run["m1_scalar_assembly"],
                    run["rename_m1_propagation"],
                    run["rename_m1_input_audit"],
                ],
                tables={
                    "native_fast_security_mapping": mapping,
                    "slow_history_links": links.with_columns(
                        pl.col("effective_index") - start, pl.col("known_index") - start
                    ),
                },
                metadata={
                    "purpose": "Bounded dated-target CPU consumer audit, not model store",
                    "feature_schema": {
                        "minimum_rank_names": 20,
                        "specifications": specs,
                        "sha256": feature_schema_sha256(
                            [FeatureSpec(**s) for s in specs]
                        ),
                    },
                },
            )
        dataset = V2DailyDataset(
            view,
            [59],
            stage="finetune",
            lookback=60,
            include_fast=True,
            include_intraday=False,
        )
        sample = dataset[0]
        batch = collate_v2_daily([sample])
        mask = expected["target_to_close_valid"] & expected["fast_present"]
        target = np.where(mask, expected["target_to_close"], 0)
        for key, value in [("to_close_mask", mask), ("to_close_target", target)]:
            np.testing.assert_array_equal(sample[key], value)
            np.testing.assert_array_equal(batch[key][0].numpy(), value)
            cells += value.size * 2
        assert sample["slow_features"].shape[:2] == (933, 60)
        dataset.store.close()
        reports.append(
            {
                "date": str(dates[day]),
                "global_index": day,
                "minutes": combinations[day].tolist(),
                "view_manifest": binding(view / "manifest.json"),
            }
        )
    write_json_atomic(
        out / "manifest.json",
        {
            "clock": binding(root / "manifest.json"),
            "samples": len(request),
            "names": len(isins),
            "history": 60,
            "cells": cells,
            "mismatches": 0,
            "scopes": reports,
            "seconds": time.perf_counter() - started,
            "executed_reproducer": binding(out / "executed_reproducer.py"),
            "neural_forward": False,
            "forecast_scoring": False,
        },
    )
    print(
        json.dumps(
            {
                "samples": len(request),
                "cells": cells,
                "seconds": time.perf_counter() - started,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
