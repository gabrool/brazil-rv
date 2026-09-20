"""Verify assembled M1 scalars through the actual full-population consumer."""

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
    root = Path(run["root"]) / "m1_scalar_assembly"
    source = bound_json(binding(root / "amendments/manifest.json"))
    history = bound_json(run["rename_history_propagation"])
    peers = bound_json(run["rename_peer_propagation"])
    native = bound_json(run["rename_m1_propagation"])
    tail = bound_json(run["rename_m1_input_audit"])
    store = Path(source["parent"]["root"])
    original = bound_json(
        {
            "path": str(store / "manifest.json"),
            "sha256": source["parent"]["manifest_sha256"],
        }
    )
    out = root / "input_audit"
    out.mkdir()
    shutil.copyfile(__file__, out / "executed_reproducer.py")
    dates, isins = np.load(store / "date_index.npy"), np.load(store / "isin_index.npy")
    mapping = pl.read_parquet(
        store / original["tables"]["native_fast_security_mapping"]["path"]
    )
    links = pl.read_parquet(verified(history["history_mapping"]))
    active = np.load(verified(history["arrays"]["active"]), mmap_mode="r")
    prior_active = np.load(store / "active.npy", mmap_mode="r")
    amendments = {}
    for record in [source["deltas"], native["native_deltas"], tail["tail_deltas"]]:
        with np.load(verified(record)) as delta:
            for key in delta.files:
                if key.endswith("__indices"):
                    name = key.split("__")[1]
                    amendments.setdefault(name, []).append(
                        (delta[key], delta[key.replace("__indices", "__values")])
                    )
    specs = [
        s
        for s in original["metadata"]["feature_schema"]["specifications"]
        if s["family"] in {"slow", "intraday", "native_fast"}
    ]
    totals, scopes = 0, []
    for effect in sorted(set(links["effective_index"])):
        names = links.filter(pl.col("effective_index") == effect)[
            "successor_index"
        ].to_numpy()
        gained = np.flatnonzero(
            (active[:, names] & ~prior_active[:, names]).any(axis=1)
        )
        request = np.unique(np.r_[effect - 1, gained])
        start, stop = int(request.min()) - 59, int(request.max()) + 1
        rows = np.arange(start, stop)
        view = out / str(dates[effect])
        expected = {}
        with StoreStaging(view, dates=dates[rows], isins=isins.tolist()) as staging:
            for key in [
                "active",
                "slow_values",
                "slow_valid",
                "slow_age_sessions",
                "slow_timestep_valid",
            ]:
                record = peers.get("arrays", {}).get(key, history["arrays"].get(key))
                staging.write_array(key, np.load(verified(record), mmap_mode="r")[rows])
            for key in sorted(
                set(amendments)
                | {
                    "fast_present",
                    "intraday_values",
                    "intraday_valid",
                    "intraday_age_sessions",
                    "intraday_support_fraction",
                    "target_to_close",
                    "target_to_close_valid",
                }
            ):
                value = np.array(
                    np.load(store / (key + ".npy"), mmap_mode="r")[rows], copy=True
                )
                for ix, values in amendments.get(key, []):
                    pick = (ix[:, 0] >= start) & (ix[:, 0] < stop)
                    index = ix[pick].copy()
                    index[:, 0] -= start
                    value[tuple(index.T)] = values[pick]
                staging.write_array(key, value)
                if (
                    key.startswith("intraday")
                    or key.startswith("target_to_close")
                    or key == "fast_present"
                ):
                    expected[key] = value
            staging.seal(
                feature_names={
                    k: original["feature_names"][k]
                    for k in ["slow", "intraday", "native_fast"]
                },
                sources=[
                    binding(root / "amendments/manifest.json"),
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
                    "purpose": "Bounded M1 scalar/target consumer audit; not accepted model inputs",
                    "parent": source["parent"],
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
            (request - start).tolist(),
            stage="finetune",
            lookback=60,
            include_fast=True,
            include_intraday=True,
        )
        checks = 0
        for i, t in enumerate(request - start):
            sample = dataset[i]
            batch = collate_v2_daily([sample])
            expected_fields = {
                "current_features": np.where(
                    expected["intraday_valid"][t], expected["intraday_values"][t], 0
                ),
                "current_feature_mask": expected["intraday_valid"][t],
                "current_feature_age_sessions": expected["intraday_age_sessions"][t],
                "to_close_mask": expected["target_to_close_valid"][t]
                & expected["fast_present"][t],
            }
            expected_fields["to_close_target"] = np.where(
                expected_fields["to_close_mask"], expected["target_to_close"][t], 0
            )
            for key, value in expected_fields.items():
                np.testing.assert_array_equal(sample[key], value, err_msg=key)
                np.testing.assert_array_equal(batch[key][0].numpy(), value, err_msg=key)
                checks += value.size * 2
            assert sample["slow_features"].shape[:2] == (933, 60)
            if i == 0:
                for key in [
                    "intraday_values",
                    "intraday_valid",
                    "intraday_age_sessions",
                    "intraday_support_fraction",
                    "target_to_close",
                    "target_to_close_valid",
                ]:
                    np.testing.assert_array_equal(
                        expected[key][t],
                        np.load(store / (key + ".npy"), mmap_mode="r")[request[i]],
                    )
        dataset.store.close()
        totals += checks
        scopes.append(
            {
                "start": str(dates[start]),
                "end": str(dates[stop - 1]),
                "samples": len(request),
                "cells": checks,
                "view": str(view),
                "pre_event_exact": True,
            }
        )
    write_json_atomic(
        out / "manifest.json",
        {
            "source": binding(root / "amendments/manifest.json"),
            "all_names": 933,
            "history": 60,
            "scopes": scopes,
            "cells": totals,
            "mismatches": 0,
            "neural_forward": False,
            "forecast_scoring": False,
            "seconds": time.perf_counter() - started,
            "reproducer": binding(out / "executed_reproducer.py"),
        },
    )
    print(json.dumps({"cells": totals, "scopes": scopes}), flush=True)


if __name__ == "__main__":
    main()
