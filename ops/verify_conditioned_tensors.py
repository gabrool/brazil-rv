"""Verify field identity, common/per-name routing and padded actual CPU batches."""

import json
from pathlib import Path
import time

import numpy as np
import torch

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data import V2DailyDataset
from brazil_rv.v2.round7_preprocessing import Round7Preprocessing


PROJECT = Path(__file__).resolve().parents[1]


def transformed(values, valid, scaler):
    x = np.arcsinh((values.astype(np.float64) - scaler["center"]) / scaler["scale"])
    return np.where(valid, np.where(scaler["passthrough"], values, x), 0).astype(
        np.float32
    )


def main():
    started = time.perf_counter()
    root = Path(
        json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())[
            "root"
        ]
    )
    source = root / "fit_conditioning_audit/report.json"
    audit = json.loads(source.read_text())
    destination = source.with_name("tensors.json")
    if destination.exists():
        raise FileExistsError(destination)
    store = Path(audit["store"]["root"])
    metadata = json.loads((store / "manifest.json").read_text())
    specs = {
        (s["family"], s["name"]): s["transform"]
        for s in metadata["metadata"]["feature_schema"]["specifications"]
    }
    seen = set()
    cells = 0
    samples = 0
    checkpoints = []
    store_active = np.load(store / "active.npy", mmap_mode="r")
    for record in audit["fits"]:
        path = Path(record["path"])
        manifest = json.loads(path.read_text())
        checkpoint = path.parent / "selected.pt"
        assert sha256_file(checkpoint) == manifest["artifacts"]["selected.pt"]
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=True, mmap=True
        )
        assert (
            payload["contract"]["preprocessing"]
            == manifest["contract"]["preprocessing"]
        )
        checkpoints.append(
            {"path": str(checkpoint), "sha256": manifest["artifacts"]["selected.pt"]}
        )
        del payload
        key = record["coordinates_sha256"]
        if key in seen:
            continue
        seen.add(key)
        path = Path(record["path"])
        assert sha256_file(path) == record["sha256"]
        c = json.loads(path.read_text())["contract"]
        prep = c["preprocessing"]
        rows = audit["coordinates"][key]["probe_dates"]
        dataset = V2DailyDataset(
            store,
            rows,
            stage="pretrain" if c["stage"] == "P" else "finetune",
            lookback=60,
            enabled_sidecars=tuple(prep["families"]),
            include_intraday=False,
            include_fast=False,
            include_common_state=prep["diagnostic"] is not None,
            compact_names=True,
            target_window_indices=c["fit_target_window"],
            purpose="training",
        )
        transformer = Round7Preprocessing.from_payload(prep)
        for i in range(len(rows)):
            raw = dataset[i]
            batch = transformer.collate([raw], fixed_name_count=c["padded_name_count"])
            active = np.flatnonzero(raw["active_mask"])
            np.testing.assert_array_equal(
                batch["name_index"][0, : len(active)].numpy(),
                np.flatnonzero(store_active[rows[i]]),
            )
            common = None
            for family, scaler in prep["families"].items():
                prefix = "sidecar_" + family
                names = prep["feature_names"][family]
                indices = [
                    metadata["feature_names"][prefix].index(name) for name in names
                ]
                np.testing.assert_array_equal(
                    scaler["passthrough"],
                    [
                        specs[prefix, name]
                        in {
                            "binary",
                            "bounded_fraction",
                            "signed_identity",
                            "age_sessions",
                        }
                        for name in names
                    ],
                )
                valid = raw[prefix + "_valid"][..., indices]
                age = raw[prefix + "_age_sessions"][..., indices]
                expected = transformed(
                    raw[prefix + "_values"][..., indices], valid, scaler
                )
                if family == "cross_market" and prep["common_columns"]:
                    shared = [
                        j
                        for j, n in enumerate(names)
                        if n.startswith("shock_")
                        or n
                        in {
                            "foreign_flow_1",
                            "foreign_flow_5",
                            "foreign_flow_month_reset",
                            "foreign_flow_methodology_change",
                            "ewz_minus_bova11_1",
                        }
                    ]
                    per_name = [j for j in range(len(names)) if j not in shared]
                    assert (
                        shared == prep["common_columns"]
                        and per_name == prep["per_name_columns"]
                    )
                    common_valid = valid[:, shared].any(axis=0)
                    positions = valid[:, shared].argmax(axis=0)
                    common_values = expected[positions, shared]
                    common_ages = np.max(age[:, shared], axis=0)
                    diagnostics = transformed(
                        raw["common_state_features"],
                        raw["common_state_feature_mask"],
                        prep["diagnostic"],
                    )
                    common = (
                        np.r_[common_values, diagnostics],
                        np.r_[common_valid, raw["common_state_feature_mask"]],
                        np.r_[
                            common_ages,
                            np.where(raw["common_state_feature_mask"], 1.0, -1.0),
                        ].astype(np.float32),
                    )
                    expected, valid, age = (
                        expected[:, per_name],
                        valid[:, per_name],
                        age[:, per_name],
                    )
                for suffix, value in [
                    ("values", expected),
                    ("valid", valid),
                    ("age_sessions", age),
                ]:
                    packed = np.zeros(
                        (c["padded_name_count"], value.shape[1]), dtype=value.dtype
                    )
                    packed[: len(active)] = value[active]
                    actual = batch[prefix + "_" + suffix][0].numpy()
                    assert actual.dtype == packed.dtype
                    np.testing.assert_array_equal(actual, packed)
                    cells += actual.size
            if common is not None:
                for suffix, value in zip(
                    ("features", "feature_mask", "age_sessions"), common
                ):
                    np.testing.assert_array_equal(
                        batch["common_state_" + suffix][0].numpy(), value
                    )
                    cells += value.size
            assert batch["slow_features"].shape[-2] == 60
            samples += 1
        dataset.store.close()
    write_json_atomic(
        destination,
        {
            "scaler_audit": {"path": str(source), "sha256": sha256_file(source)},
            "coordinate_systems": len(seen),
            "full_population_samples": samples,
            "verified_tensor_cells": cells,
            "mismatches": 0,
            "checkpoint_payloads": checkpoints,
            "method": "source-name indices and typed passthrough from sealed feature schema; independent common/per-name field selection, validity/ages, permanent name indices and exact padded float32/bool CPU tensors",
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    print(destination.read_text())


if __name__ == "__main__":
    main()
