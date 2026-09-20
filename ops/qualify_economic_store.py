"""Verify composed arrays, independent family transforms and actual model inputs."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.contract import (
    HORIZONS,
    PRETRAIN_END,
    FINETUNE_START,
    TARGET_NEUTRALIZATION_FEATURES,
)
from brazil_rv.v2.data import V2DailyDataset, collate_v2_daily
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.store import close_memmap
from assemble_economic_store import inputs, verified
from audit_auxiliary_tensors import transformed
from audit_corporate_target_inputs import neutral_oracle, PAIRS

PROJECT = Path(__file__).resolve().parents[1]


def main():
    started = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    assembly = bound_json(run["derived_store_assembly"])
    contract = bound_json(assembly["contract"])
    reports, replacements, families, patches = inputs(contract)
    parent = Path(contract["parent"]["root"])
    original = bound_json(
        {
            "path": str(parent / "manifest.json"),
            "sha256": contract["parent"]["manifest_sha256"],
        }
    )
    root = Path(assembly["store"]["root"])
    manifest = bound_json(
        {
            "path": str(root / "manifest.json"),
            "sha256": assembly["store"]["manifest_sha256"],
        }
    )
    output = Path(contract["evidence_root"]) / "qualification"
    output.mkdir(exist_ok=True)
    executed = output / ("executed_" + sha256_file(Path(__file__))[:12] + ".py")
    if not executed.exists():
        executed.write_bytes(Path(__file__).read_bytes())
    identity = {"store": assembly["store"], "contract": assembly["contract"]}
    dates = np.load(root / "date_index.npy")
    isins = tuple(np.load(root / "isin_index.npy").tolist())
    np.testing.assert_array_equal(dates, np.load(parent / "date_index.npy"))
    assert isins == tuple(np.load(parent / "isin_index.npy"))
    assert manifest["feature_schema_sha256"] == original["feature_schema_sha256"]
    generated = {
        f"sidecar_{f}_{s}"
        for f in families
        for s in ("values", "valid", "age_sessions")
    }

    # Successful earlier boundaries can be reused after a consumer-only harness
    # correction; each receipt binds this exact store and frozen contract.
    array_receipt = output / "array_composition.json"
    if array_receipt.exists():
        array_audit = json.loads(array_receipt.read_text())
        assert array_audit["identity"] == identity
    else:
        goll = reports["goll_activity_admission"]
        with np.load(verified(goll["artifacts"]["slow_patch"])) as z:
            gp = {k: z[k].copy() for k in z.files}
        cells, untouched = 0, []
        for name, record in manifest["arrays"].items():
            if name in generated:
                continue
            base = replacements.get(name)
            path = (
                Path(base["path"])
                if base
                else parent / original["arrays"][name]["path"]
            )
            before = np.load(path, mmap_mode="r")
            actual = np.load(root / record["path"], mmap_mode="r")
            for t in range(0, len(dates), 64):
                end = min(t + 64, len(dates))
                expected = before[t:end].copy()
                if name == "slow_values":
                    keep = (gp["date_index"] >= t) & (gp["date_index"] < end)
                    expected[
                        gp["date_index"][keep] - t,
                        gp["name_index"][keep],
                        gp["field_index"][keep],
                    ] = gp["values"][keep]
                if (
                    name in ("volume_brl", "quantity", "trade_count", "trade_observed")
                    and t <= goll["date_index"] < end
                ):
                    expected[goll["date_index"] - t, goll["name_index"]] = goll[
                        "printed_activity"
                    ][name]
                for _, ix, value in patches.get(name, []):
                    coordinates = (
                        np.column_stack(np.unravel_index(ix, before.shape))
                        if ix.ndim == 1
                        else ix
                    )
                    keep = (coordinates[:, 0] >= t) & (coordinates[:, 0] < end)
                    local = coordinates[keep].copy()
                    local[:, 0] -= t
                    expected[tuple(local.T)] = value[keep]
                np.testing.assert_array_equal(actual[t:end], expected, err_msg=name)
                cells += expected.size
            if (
                not assembly["effects"][name].get("changed", 0)
                and name in original["arrays"]
            ):
                assert record["sha256"] == original["arrays"][name]["sha256"], name
                untouched.append(name)
            close_memmap(before)
            close_memmap(actual)
        array_audit = {
            "identity": identity,
            "cells": cells,
            "untouched_hash_exact": untouched,
            "mismatches": 0,
            "code": binding(executed),
        }
        write_json_atomic(array_receipt, array_audit)
        print(json.dumps({"composed_arrays_exact": cells}), flush=True)

    family_receipt = output / "family_transforms.json"
    if family_receipt.exists():
        family_audit = json.loads(family_receipt.read_text())
        assert family_audit["identity"] == identity
    else:
        active = np.load(root / "active.npy", mmap_mode="r")
        date_idx = {d: t for t, d in enumerate(dates.astype(object))}
        name_idx = {n: j for j, n in enumerate(isins)}
        cells = 0
        for family, record in families.items():
            f = "sidecar_" + family
            frame = pl.read_parquet(verified(record))
            d = np.array([date_idx[x] for x in frame["date"]], dtype=int)
            n = np.array([name_idx[x] for x in frame["isin"]], dtype=int)
            actual = [
                np.load(root / f"{f}_{s}.npy", mmap_mode="r")
                for s in ("values", "valid", "age_sessions")
            ]
            specs = {
                s["name"]: s
                for s in manifest["metadata"]["feature_schema"]["specifications"]
                if s["family"] == f
            }
            for j, name in enumerate(manifest["feature_names"][f]):
                payload = (
                    frame[name].cast(pl.Float32).to_numpy()
                    if name in frame.columns
                    else np.zeros(len(d), np.float32)
                )
                known = np.isfinite(payload) & (name in frame.columns)
                if name + "_mask" in frame.columns:
                    known &= frame[name + "_mask"].fill_null(False).to_numpy()
                age = (
                    frame[name + "_age_sessions"].cast(pl.Float32).to_numpy()
                    if name + "_age_sessions" in frame.columns
                    else np.full(len(d), -1, np.float32)
                )
                values, mask = transformed(payload, known & active[d, n], specs[name])
                expected = np.zeros(active.shape, np.float32)
                expected[d, n] = values
                np.testing.assert_array_equal(
                    actual[0][..., j], expected, err_msg=f + "/" + name
                )
                expected_valid = np.zeros(active.shape, bool)
                expected_valid[d, n] = mask
                np.testing.assert_array_equal(actual[1][..., j], expected_valid)
                expected.fill(-1)
                has_age = np.isfinite(age) & (age >= 0) & active[d, n]
                expected[d[has_age], n[has_age]] = age[has_age]
                np.testing.assert_array_equal(actual[2][..., j], expected)
                cells += active.size * 3
            for a in actual:
                close_memmap(a)
            del frame, actual
        family_audit = {
            "identity": identity,
            "cells": cells,
            "mismatches": 0,
            "code": binding(executed),
        }
        write_json_atomic(family_receipt, family_audit)
        close_memmap(active)
        print(json.dumps({"independent_family_cells_exact": cells}), flush=True)

    arrays = {}

    def a(k):
        if k not in arrays:
            arrays[k] = np.load(root / manifest["arrays"][k]["path"], mmap_mode="r")
        return arrays[k]

    active = a("active")
    old_active = np.load(parent / "active.npy", mmap_mode="r")
    gained_days = np.flatnonzero((active & ~old_active).any(axis=1))
    links = (
        pl.read_parquet(root / manifest["tables"]["slow_history_links"]["path"])
        .sort("effective_index", descending=True)
        .to_dicts()
    )
    samples = set(gained_days.tolist())
    for year in range(2010, 2025):
        for month in (3, 9):
            samples.add(
                int(np.searchsorted(dates, np.datetime64(f"{year}-{month:02}-15")))
            )
    for b in reports["corporate_claim_rows"]["barriers"]:
        t = int(np.searchsorted(dates, np.datetime64(b["start"])))
        samples.update(range(t - 10, min(t + 7, len(dates))))
    samples.update([279, 299, len(dates) - 1])
    for link in links:
        t = max(link["effective_index"], link["known_index"])
        samples.update(
            [t - 1, t, min(t + 59, len(dates) - 1), min(t + 125, len(dates) - 1)]
        )
    samples = sorted(
        t
        for t in samples
        if 59 <= t < len(dates)
        and (
            dates[t] <= np.datetime64(PRETRAIN_END)
            or dates[t] >= np.datetime64(FINETUNE_START)
        )
    )
    groups = sorted(
        f.removeprefix("sidecar_")
        for f in manifest["feature_names"]
        if f.startswith("sidecar_")
    )
    mapping = pl.read_parquet(
        root / manifest["tables"]["native_fast_security_mapping"]["path"]
    )
    print(
        json.dumps(
            {"consumer_dates": len(samples), "native_mapping_columns": mapping.columns}
        ),
        flush=True,
    )
    native_indices = mapping.sort("fast_index")["store_name_index"].to_numpy()
    risk = [
        manifest["feature_names"]["slow"].index(n)
        for n in TARGET_NEUTRALIZATION_FEATURES
    ]
    packed_cells = 0
    sample_receipts = []
    for stage, bound in (
        ("pretrain", dates <= np.datetime64(PRETRAIN_END)),
        ("finetune", dates >= np.datetime64(FINETUNE_START)),
    ):
        indices = [t for t in samples if bound[t]]
        target_window = sorted(
            {u for t in indices for u in range(t, min(t + 11, len(dates))) if bound[u]}
        )
        data = V2DailyDataset(
            root,
            indices,
            stage=stage,
            lookback=60,
            enabled_sidecars=groups,
            include_common_state=True,
            target_window_indices=target_window,
        )
        for i, t in enumerate(indices):
            sample = data[i]
            batch = collate_v2_daily([sample])
            window = np.arange(t - 59, t + 1)
            source_names = np.tile(np.arange(len(isins)), (60, 1))
            for link in links:
                if t >= max(link["effective_index"], link["known_index"]):
                    changed = (window[:, None] < link["effective_index"]) & (
                        source_names == link["successor_index"]
                    )
                    source_names[changed] = link["predecessor_index"]
            ix = window[:, None]
            slow_mask = a("slow_valid")[ix, source_names].transpose(1, 0, 2)
            hist_mask = a("slow_timestep_valid")[ix, source_names].T
            expected = {
                "active_mask": active[t],
                "slow_feature_mask": slow_mask,
                "slow_history_mask": hist_mask,
                "slow_features": np.where(
                    slow_mask, a("slow_values")[ix, source_names].transpose(1, 0, 2), 0
                ),
                "slow_feature_age_sessions": np.where(
                    hist_mask[..., None],
                    a("slow_age_sessions")[ix, source_names].transpose(1, 0, 2),
                    -1,
                ),
                "current_features": np.where(
                    a("intraday_valid")[t], a("intraday_values")[t], 0
                ),
                "current_feature_mask": a("intraday_valid")[t],
                "current_feature_age_sessions": a("intraday_age_sessions")[t],
            }
            for family in groups:
                f = "sidecar_" + family
                expected[f + "_values"] = np.where(
                    a(f + "_valid")[t], a(f + "_values")[t], 0
                )
                expected[f + "_valid"] = a(f + "_valid")[t]
                expected[f + "_age_sessions"] = a(f + "_age_sessions")[t]
            for _, maskkey, valuekey, targetmask in PAIRS[1:]:
                mask = a(maskkey)[t].copy()
                for j, h in enumerate(HORIZONS):
                    if any(u not in target_window for u in range(t, t + h + 1)):
                        mask[:, j] = False
                storekey = next(p[0] for p in PAIRS if p[2] == valuekey)
                expected[targetmask] = mask
                expected[valuekey] = np.where(mask, a(storekey)[t], 0)
            primary, pmask = neutral_oracle(
                a("target_shareholder_simple_return")[t],
                a("target_valid")[t],
                a("target_scale_sigma")[t],
                a("slow_values")[t][:, risk],
                a("slow_valid")[t][:, risk].all(axis=1),
            )
            for j, h in enumerate(HORIZONS):
                if any(u not in target_window for u in range(t, t + h + 1)):
                    pmask[:, j] = False
            expected.update(targets=np.where(pmask, primary, 0), target_mask=pmask)
            for key, value in expected.items():
                np.testing.assert_array_equal(
                    sample[key], value, err_msg=f"{dates[t]}:{key}"
                )
                np.testing.assert_array_equal(
                    batch[key][0].numpy(), value, err_msg=f"packed:{key}"
                )
                packed_cells += value.size * 2
            if stage == "finetune":
                selected = a("fast_present")[t, native_indices] & a("fast_patch_mask")[
                    t
                ].any(axis=1)
                mask = a("fast_patch_valid")[t, selected]
                ex = {
                    "fast_name_index": native_indices[selected],
                    "fast_patch_values": np.where(
                        mask, a("fast_patch_values")[t, selected], 0
                    ),
                    "fast_patch_valid": mask,
                    "fast_patch_mask": a("fast_patch_mask")[t, selected],
                    "fast_state_position": a("fast_patch_mask")[t, selected].sum(
                        axis=1
                    ),
                }
                for key, value in ex.items():
                    np.testing.assert_array_equal(sample[key], value)
                    np.testing.assert_array_equal(
                        batch[key][0].numpy()[: len(value)], value
                    )
                    packed_cells += value.size * 2
            else:
                assert sample["fast_name_index"].size == 0
            close_mask = a("target_to_close_valid")[t] & sample["fast_present"]
            np.testing.assert_array_equal(sample["to_close_mask"], close_mask)
            np.testing.assert_array_equal(
                sample["to_close_target"],
                np.where(close_mask, a("target_to_close")[t], 0),
            )
            sample_receipts.append(
                {
                    "date": str(dates[t]),
                    "stage": stage,
                    "active": int(active[t].sum()),
                    "native_names": len(sample["fast_name_index"]),
                }
            )
        data.store.close()
    for v in arrays.values():
        close_memmap(v)
    report = {
        "status": "passed_combined_store_and_actual_cpu_tensors",
        "identity": identity,
        "array_composition": binding(array_receipt),
        "family_transforms": binding(family_receipt),
        "samples": sample_receipts,
        "sample_count": len(samples),
        "names": len(isins),
        "history": 60,
        "packed_cells": packed_cells,
        "mismatches": 0,
        "original_schema_exact": True,
        "new_refit_required": True,
        "model_forward_or_heldout": False,
        "code": binding(executed),
        "seconds": perf_counter() - started,
    }
    write_json_atomic(output / "manifest.json", report)
    run["derived_store_input_audit"] = binding(output / "manifest.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("status", "sample_count", "packed_cells", "seconds")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
