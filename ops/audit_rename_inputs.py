"""Check full-population 60-session tensors against explicit dated ISIN routing."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data import V2DailyDataset, collate_v2_daily
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.feature_spec import FeatureSpec, feature_schema_sha256
from brazil_rv.v2.store import StoreStaging

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peer", action="store_true")
    args = parser.parse_args()
    started = perf_counter()
    pointer = json.loads(
        (PROJECT / "docs/v2_economic_data_scaling_run.json").read_text(encoding="utf8")
    )
    root = Path(pointer["root"]) / "rename_history_propagation"
    destination = root.parent / "rename_peer_history" if args.peer else root

    def array_path(name):
        candidate = destination / f"{name}.npy"
        return candidate if candidate.exists() else root / f"{name}.npy"

    amendment = bound_json(binding(root / "manifest.json"))
    parent = amendment["parent"]
    source = Path(parent["root"])
    original = bound_json(
        {"path": str(source / "manifest.json"), "sha256": parent["manifest_sha256"]}
    )
    dates, isins = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    assert dates[-1] <= np.datetime64("2024-12-30")
    links = pl.read_parquet(root / "slow_history_links.parquet")
    specs = [
        s
        for s in original["metadata"]["feature_schema"]["specifications"]
        if s["family"] == "slow"
    ]
    audit_store = destination / "input_audit_store"
    # Only new derived arrays are copied into this disposable audit view; no
    # immutable input dataset or checkpoint is duplicated or scored.
    with StoreStaging(audit_store, dates=dates, isins=isins) as staging:
        for key in (
            "active",
            "slow_values",
            "slow_valid",
            "slow_age_sessions",
            "slow_timestep_valid",
        ):
            staging.copy_array(key, array_path(key))
        staging.seal(
            feature_names={"slow": original["feature_names"]["slow"]},
            sources=[
                binding(root / "manifest.json"),
                binding(destination / "manifest.json"),
            ],
            metadata={
                "purpose": "input-routing audit only; not an accepted model store",
                "parent": parent,
                "feature_schema": {
                    "minimum_rank_names": 20,
                    "specifications": specs,
                    "sha256": feature_schema_sha256([FeatureSpec(**s) for s in specs]),
                },
            },
            tables={"slow_history_links": links},
        )
    rows = sorted(
        {
            max(59, min(len(dates) - 1, row["effective_index"] + offset))
            for row in links.iter_rows(named=True)
            for offset in (-10, -1, 0, 1, 20, 59)
        }
    )
    arrays = {
        k: np.load(array_path(k), mmap_mode="r")
        for k in (
            "slow_values",
            "slow_valid",
            "slow_age_sessions",
            "slow_timestep_valid",
            "active",
        )
    }
    dataset = V2DailyDataset(
        audit_store,
        rows,
        stage="finetune",
        lookback=60,
        include_fast=False,
        include_intraday=False,
    )
    counts = []
    cells = 0
    ordered_links = links.sort("effective_index", descending=True).to_dicts()
    for item, t in enumerate(rows):
        sample = dataset[item]
        window = np.arange(t - 59, t + 1)
        source_names = np.tile(np.arange(len(isins)), (60, 1))
        # Independent literal row/name reconstruction, including each link's
        # current decision knowledge. This does not call the routing helper.
        for j in range(len(isins)):
            for h, day in enumerate(window):
                for row in ordered_links:
                    if (
                        t >= max(row["known_index"], row["effective_index"])
                        and day < row["effective_index"]
                        and source_names[h, j] == row["successor_index"]
                    ):
                        source_names[h, j] = row["predecessor_index"]
        ix = window[:, None]
        expected_mask = arrays["slow_valid"][ix, source_names].transpose(1, 0, 2)
        expected_value = np.where(
            expected_mask, arrays["slow_values"][ix, source_names].transpose(1, 0, 2), 0
        )
        expected_history = arrays["slow_timestep_valid"][ix, source_names].T
        expected_age = np.where(
            expected_history[..., None],
            arrays["slow_age_sessions"][ix, source_names].transpose(1, 0, 2),
            -1,
        )
        expected = {
            "slow_features": expected_value,
            "slow_feature_mask": expected_mask,
            "slow_history_mask": expected_history,
            "slow_feature_age_sessions": expected_age,
            "active_mask": arrays["active"][t],
        }
        batch = collate_v2_daily([sample])
        for key, values in expected.items():
            np.testing.assert_array_equal(
                sample[key], values, err_msg=f"{dates[t]}:{key}"
            )
            np.testing.assert_array_equal(
                batch[key][0].numpy(), values, err_msg=f"packed:{key}"
            )
            cells += values.size
        counts.append(
            {
                "date": str(dates[t]),
                "active": int(expected["active_mask"].sum()),
                "inherited_valid_cells": int(
                    (
                        expected_mask
                        & (
                            source_names.T[..., None]
                            != np.arange(len(isins))[:, None, None]
                        )
                    ).sum()
                ),
            }
        )
    dataset.store.close()
    # Remove every future identity link from a separate bounded data view. The
    # already published pre-event features and actual tensor must remain exact.
    first = min(links["effective_index"])
    early = V2DailyDataset(
        audit_store,
        [first - 1],
        stage="finetune",
        lookback=60,
        include_fast=False,
        include_intraday=False,
    )
    baseline = V2DailyDataset(
        source,
        [first - 1],
        stage="finetune",
        lookback=60,
        include_fast=False,
        include_intraday=False,
    )
    for key in (
        "slow_features",
        "slow_feature_mask",
        "slow_history_mask",
        "slow_feature_age_sessions",
        "active_mask",
    ):
        np.testing.assert_array_equal(early[0][key], baseline[0][key])
    early.store.close()
    baseline.store.close()
    result = {
        "status": "passed",
        "store": binding(audit_store / "manifest.json"),
        "amendment": binding(root / "manifest.json"),
        "peer_amendment": binding(destination / "manifest.json") if args.peer else None,
        "code": binding(Path(__file__)),
        "samples": counts,
        "full_population_names": len(isins),
        "lookback": 60,
        "array_and_packed_tensor_cells": cells,
        "mismatches": 0,
        "pre_rename_original_tensor_exact": True,
        "seconds": perf_counter() - started,
        "heldout_access": False,
        "neural_forward": False,
        "forecast_scoring": False,
        "audit_store_recovery": "rebuild this five-array view from the bound amendment; do not archive duplicate arrays",
    }
    write_json_atomic(destination / "input_audit.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
