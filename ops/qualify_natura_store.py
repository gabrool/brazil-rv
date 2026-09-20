"""Verify the complete Natura store and combined final CPU model tensors."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.contract import HORIZONS, TARGET_NEUTRALIZATION_FEATURES
from brazil_rv.v2.data import V2DailyDataset, collate_v2_daily
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.train import _verify_additive_parent_transfer
from audit_corporate_target_inputs import neutral_oracle

PROJECT = Path(__file__).resolve().parents[1]


def main():
    started = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    assembly = bound_json(run["natura_store_assembly"])
    contract = bound_json(assembly["contract"])
    root, parent = (
        Path(assembly["store"]["root"]),
        Path(assembly["parent_store"]["root"]),
    )
    m = bound_json(
        {
            "path": str(root / "manifest.json"),
            "sha256": assembly["store"]["manifest_sha256"],
        }
    )
    old = bound_json(
        {
            "path": str(parent / "manifest.json"),
            "sha256": assembly["parent_store"]["manifest_sha256"],
        }
    )
    dates, isins = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    np.testing.assert_array_equal(dates, np.load(parent / "date_index.npy"))
    assert (
        isins == np.load(parent / "isin_index.npy").tolist()
        and m["feature_schema_sha256"] == old["feature_schema_sha256"]
    )
    out = Path(run["natura_store_assembly"]["path"]).parent / "qualification"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    patches = {}
    bindings = [
        bound_json(contract["amendments"][k])["deltas"]
        for k in (
            "natura_wealth_amendment",
            "natura_daily_input_audit",
            "natura_auxiliary_qualification",
        )
    ]
    bindings.append(contract["action_deltas"])
    for rec in bindings:
        with np.load(rec["path"]) as z:
            for key in z.files:
                if key.endswith("__indices") and len(z[key]):
                    name = key.removesuffix("__indices")
                    assert name not in patches
                    patches[name] = (z[key].copy(), z[name + "__values"].copy())
    cached = {}

    def original(key):
        if key not in cached:
            cached[key] = np.load(parent / old["arrays"][key]["path"], mmap_mode="r")
        return cached[key]

    def expected(key, rows):
        result = np.array(original(key)[rows], copy=True)
        if key in patches:
            ix, values = patches[key]
            in_rows = np.isin(ix[:, 0], rows)
            local = ix[in_rows].copy()
            local[:, 0] = np.searchsorted(rows, local[:, 0])
            result[tuple(local.T)] = values[in_rows]
        return result

    cells = 0
    untouched = []
    for key, rec in m["arrays"].items():
        assert (
            rec["shape"] == old["arrays"][key]["shape"]
            and rec["dtype"] == old["arrays"][key]["dtype"]
        )
        if key not in patches:
            assert rec["sha256"] == old["arrays"][key]["sha256"]
            untouched.append(key)
            continue
        actual = np.load(root / rec["path"], mmap_mode="r")
        for start in range(0, len(dates), 64):
            rows = np.arange(start, min(start + 64, len(dates)))
            np.testing.assert_array_equal(
                actual[rows], expected(key, rows), err_msg=key
            )
            cells += actual[rows].size
    write_json_atomic(
        out / "arrays.json",
        {
            "cells": cells,
            "unchanged_arrays_hash_exact": untouched,
            "changed_arrays": list(patches),
            "mismatches": 0,
        },
    )
    daily = bound_json(run["natura_daily_propagation"])
    plan = bound_json(daily["plan"])
    _, first, stop = plan["rows"]
    chosen = (
        set(range(first - 10, stop, 14))
        | set(range(first - 2, first + 3))
        | set(range(stop - 3, stop))
    )
    for day in ("2019-11-07", "2019-12-18"):
        t = int(np.searchsorted(dates, np.datetime64(day)))
        chosen.update(range(t - 2, t + 3))
    samples = sorted(chosen)
    grants = sorted({t + u for t in samples for u in range(max(HORIZONS) + 1)})
    groups = [
        f.removeprefix("sidecar_")
        for f in m["feature_names"]
        if f.startswith("sidecar_")
    ]
    dataset = V2DailyDataset(
        root,
        samples,
        stage="finetune",
        enabled_sidecars=groups,
        include_common_state=True,
        lookback=60,
        target_window_indices=grants,
    )
    risk = [m["feature_names"]["slow"].index(f) for f in TARGET_NEUTRALIZATION_FEATURES]
    mapping = pl.read_parquet(
        root / m["tables"]["native_fast_security_mapping"]["path"]
    ).sort("fast_index")
    native_names = mapping["store_name_index"].to_numpy()
    packed = 0
    for i, t in enumerate(samples):
        sample = dataset[i]
        batch = collate_v2_daily([sample])
        rows = np.arange(t - 59, t + 1)
        current = np.array([t])
        valid = expected("slow_valid", rows).transpose(1, 0, 2)
        hist = expected("slow_timestep_valid", rows).T
        primary, known = neutral_oracle(
            expected("target_shareholder_simple_return", current)[0],
            expected("target_valid", current)[0],
            expected("target_scale_sigma", current)[0],
            expected("slow_values", current)[0][:, risk],
            expected("slow_valid", current)[0][:, risk].all(axis=1),
        )
        checks = {
            "active_mask": expected("active", current)[0],
            "slow_features": np.where(
                valid, expected("slow_values", rows).transpose(1, 0, 2), 0
            ),
            "slow_feature_mask": valid,
            "slow_history_mask": hist,
            "slow_feature_age_sessions": np.where(
                hist[..., None],
                expected("slow_age_sessions", rows).transpose(1, 0, 2),
                -1,
            ),
            "targets": np.where(known, primary, 0),
            "target_mask": known,
            "current_features": np.where(
                expected("intraday_valid", current)[0],
                expected("intraday_values", current)[0],
                0,
            ),
            "current_feature_mask": expected("intraday_valid", current)[0],
            "current_feature_age_sessions": expected("intraday_age_sessions", current)[
                0
            ],
        }
        for group in groups:
            k = "sidecar_" + group
            mask = expected(k + "_valid", current)[0]
            checks.update(
                {
                    k + "_values": np.where(
                        mask, expected(k + "_values", current)[0], 0
                    ),
                    k + "_valid": mask,
                    k + "_age_sessions": expected(k + "_age_sessions", current)[0],
                }
            )
        selected = expected("fast_present", current)[0, native_names] & expected(
            "fast_patch_mask", current
        )[0].any(axis=1)
        mask = expected("fast_patch_valid", current)[0, selected]
        checks.update(
            fast_name_index=native_names[selected],
            fast_patch_values=np.where(
                mask, expected("fast_patch_values", current)[0, selected], 0
            ),
            fast_patch_valid=mask,
            fast_patch_mask=expected("fast_patch_mask", current)[0, selected],
        )
        for key, value in checks.items():
            np.testing.assert_array_equal(
                sample[key], value, err_msg=f"{dates[t]}:{key}"
            )
            actual = batch[key][0].numpy()
            np.testing.assert_array_equal(
                actual[: len(value)], value, err_msg="packed:" + key
            )
            packed += value.size * 2
    dataset.store.close()
    causal = []
    for day in ("2019-09-18", "2019-11-07"):
        t = int(np.searchsorted(dates, np.datetime64(day)))
        data = V2DailyDataset(
            root,
            [t],
            stage="finetune",
            enabled_sidecars=groups,
            lookback=60,
            target_window_indices=[t],
        )
        read = data.store.read
        reads = []

        def bounded(name, selector):
            if not name.startswith("target_"):
                assert np.atleast_1d(np.arange(len(dates))[selector]).max() <= t
                reads.append(name)
            return read(name, selector)

        data.store.read = bounded
        item = data[0]
        assert not item["target_mask"].any() and not item["targets"].any()
        causal.append(
            {
                "date": day,
                "bounded_feature_reads": len(reads),
                "revoked_horizons_empty": True,
            }
        )
        data.store.close()
    try:
        _verify_additive_parent_transfer(
            {"training": {"store": assembly["parent_store"]}},
            {"training": {"store": assembly["store"]}},
            (parent, root),
        )
    except ValueError as error:
        rejection = str(error)
        assert rejection == "parent transfer changed a protected array, table or axis"
    else:
        raise AssertionError("Old weights accepted altered coordinates")
    result = {
        "status": "passed_complete_natura_store_and_combined_cpu_tensors",
        "assembly": run["natura_store_assembly"],
        "array_checks": binding(out / "arrays.json"),
        "samples": len(samples),
        "sample_dates": [str(dates[t]) for t in samples],
        "names": len(isins),
        "history": 60,
        "packed_cells": packed,
        "mismatches": 0,
        "causal": causal,
        "old_parent_rejected": rejection,
        "forward_scoring_or_heldout": False,
        "seconds": perf_counter() - started,
    }
    write_json_atomic(out / "manifest.json", result)
    run["natura_store_input_audit"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
