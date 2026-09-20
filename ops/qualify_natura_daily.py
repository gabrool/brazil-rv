"""Qualify saved bounded Natura reducers without repeating their production."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.feature_spec import (
    FeatureSpec,
    observation_age_sessions_into,
    transform_feature_panel_into,
)

PROJECT = Path(__file__).resolve().parents[1]


def main():
    started = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    produced = bound_json(run["natura_daily_propagation"])
    plan = bound_json(produced["plan"])
    source = Path(produced["plan"]["path"]).parent
    out = source / "qualification_reportable"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    root = Path(plan["parent_store"]["root"])
    manifest = bound_json(
        {
            "path": str(root / "manifest.json"),
            "sha256": plan["parent_store"]["manifest_sha256"],
        }
    )
    selected = np.load(source / "selected.npy")
    start, first, stop = plan["rows"]
    global_rows = selected + start
    assert np.array_equal(global_rows, np.arange(first, stop))

    def old(key):
        return np.load(root / manifest["arrays"][key]["path"], mmap_mode="r")

    active = old("active")[start:stop]
    specifications = [
        FeatureSpec(**s)
        for s in manifest["metadata"]["feature_schema"]["specifications"]
        if s["family"] == "slow"
    ]
    delta_parts = {}
    checks = []
    effects = []

    def equal(label, actual, expected):
        same = (actual == expected) | (np.isnan(actual) & np.isnan(expected))
        bad = np.argwhere(~same)
        checks.append(
            {
                "name": label,
                "cells": actual.size,
                "mismatches": len(bad),
                "examples": [
                    {
                        "index": ix.tolist(),
                        "actual": float(actual[tuple(ix)])
                        if np.isfinite(actual[tuple(ix)])
                        else None,
                        "expected": float(expected[tuple(ix)])
                        if np.isfinite(expected[tuple(ix)])
                        else None,
                    }
                    for ix in bad[:5]
                ],
            }
        )

    def delta(key, before, after, field=None):
        same = (before == after) | (np.isnan(before) & np.isnan(after))
        ix = np.argwhere(~same)
        indices = ix.copy()
        indices[:, 0] = global_rows[ix[:, 0]]
        if field is not None:
            indices = np.column_stack((indices, np.full(len(ix), field)))
        delta_parts.setdefault(key, []).append((indices, after[tuple(ix.T)]))
        return len(ix)

    for field in plan["fields"]:
        outputs, masks, ages, raws = [], [], [], []
        for label in ("control", "corrected"):
            with np.load(source / label / f"{field}.npz") as z:
                raw, mask = z["values"], z["valid"]
            values = np.zeros((len(selected), active.shape[1], 1), np.float32)
            valid = np.zeros_like(values, bool)
            transform_feature_panel_into(
                raw[..., None],
                mask[..., None],
                active,
                specifications[field : field + 1],
                values,
                valid,
                source_rows=selected - 1,
                membership_rows=selected,
            )
            age = np.full_like(values, -1)
            observation_age_sessions_into(
                mask[..., None],
                active,
                age,
                source_rows=selected - 1,
                decision_rows=selected,
            )
            outputs.append(values[..., 0])
            masks.append(valid[..., 0])
            ages.append(age[..., 0])
            raws.append((raw, mask))
        before = old("slow_values")[global_rows, :, field]
        before_mask = old("slow_valid")[global_rows, :, field]
        before_age = old("slow_age_sessions")[global_rows, :, field]
        equal(f"values:{field}", outputs[0], before)
        equal(f"mask:{field}", masks[0], before_mask)
        # Unchanged source masks keep the exact sealed clocks, including clocks
        # older than this bounded reducer window. Only changed clocks need proof.
        changed_clock = (ages[0] != ages[1]) & active[selected]
        equal(
            f"changed_clock_control:{field}",
            ages[0][changed_clock],
            before_age[changed_clock],
        )
        after_age = before_age.copy()
        after_age[changed_clock] = ages[1][changed_clock]
        equal(f"pre_effect:{field}", outputs[1][:1], outputs[0][:1])
        equal(f"clean_tail:{field}", outputs[1][-3:], outputs[0][-3:])
        equal(f"clean_tail_mask:{field}", masks[1][-3:], masks[0][-3:])
        effects.append(
            {
                "field": specifications[field].name,
                "index": field,
                "values": delta("slow_values", before, outputs[1], field),
                "valid": delta("slow_valid", before_mask, masks[1], field),
                "ages": delta("slow_age_sessions", before_age, after_age, field),
                "gains": int((masks[1] & ~before_mask).sum()),
                "losses": int((~masks[1] & before_mask).sum()),
                "raw_shared_changes": int(
                    (
                        raws[0][1][selected - 1]
                        & raws[1][1][selected - 1]
                        & (raws[0][0][selected - 1] != raws[1][0][selected - 1])
                    ).sum()
                ),
            }
        )
        if field == 8:
            sigmas = [
                np.where(mask[selected - 1], raw[selected - 1], np.nan).astype(
                    np.float32
                )
                for raw, mask in raws
            ]
            equal("sigma_control", sigmas[0], old("target_scale_sigma")[global_rows])
            delta("target_scale_sigma", sigmas[0], sigmas[1])
        np.savez_compressed(
            out / f"field_{field}.npz",
            control=outputs[0],
            values=outputs[1],
            valid=masks[1],
            age=after_age,
        )
    clusters = [
        np.load(source / label / "clusters.npz")["values"][selected]
        for label in ("control", "corrected")
    ]
    equal("cluster_control", clusters[0], old("monthly_cluster_labels")[global_rows])
    equal("cluster_clean_tail", clusters[1][-3:], clusters[0][-3:])
    delta("monthly_cluster_labels", clusters[0], clusters[1])
    packed = {}
    for key, parts in delta_parts.items():
        packed[key + "__indices"] = np.concatenate([p[0] for p in parts])
        packed[key + "__values"] = np.concatenate([p[1] for p in parts])
    np.savez_compressed(out / "candidate_deltas.npz", **packed)
    report = {
        "status": "passed"
        if not any(c["mismatches"] for c in checks)
        else "control_mismatch_requires_qualification",
        "producer": run["natura_daily_propagation"],
        "checks": checks,
        "effects": effects,
        "candidate_deltas": binding(out / "candidate_deltas.npz"),
        "seconds": perf_counter() - started,
    }
    write_json_atomic(out / "manifest.json", report)
    run["natura_daily_qualification"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            {
                "status": report["status"],
                "failed": [c for c in checks if c["mismatches"]],
                "effects": effects,
                "seconds": report["seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
