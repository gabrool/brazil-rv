"""Qualify saved family precision and assemble only changed typed inputs."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.feature_spec import FeatureSpec, transform_feature_panel_into
from brazil_rv.v2.round5_store import align_family

PROJECT = Path(__file__).resolve().parents[1]


def main():
    started = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    report = bound_json(run["natura_auxiliary_propagation"])
    source = Path(run["natura_auxiliary_propagation"]["path"]).parent
    root = Path(report["parent_store"]["root"])
    m = bound_json(
        {
            "path": str(root / "manifest.json"),
            "sha256": report["parent_store"]["manifest_sha256"],
        }
    )
    dates, isins = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    rows = np.load(report["rows"]["path"])
    days = dates[rows].astype(object).tolist()
    active = np.load(root / "active.npy", mmap_mode="r")[rows]
    out = source / "qualified"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    deltas, effects = {}, {}
    controls = 0
    for family, rec in report["families"].items():
        full = "sidecar_" + family
        names = m["feature_names"][full]
        fields = [names.index(n) for n in rec["names"]]
        columns = np.array(rec["columns"])
        frame = pl.read_parquet(rec["source"]["path"]).filter(
            pl.col("date").is_between(days[0], days[-1])
        )
        prior = align_family(frame, days, tuple(isins), tuple(names))
        changed = [a.copy() for a in prior]
        for j, kind in enumerate(("values", "valid", "age")):
            control = np.load(source / f"{family}_control_{kind}.npy").astype(
                bool if j == 1 else np.float32
            )
            after = np.load(source / f"{family}_{kind}.npy").astype(
                bool if j == 1 else np.float32
            )
            ix = np.ix_(np.arange(len(rows)), columns, fields)
            expected = prior[j][ix]
            if j == 2:
                expected = np.where(prior[1][ix], expected, -1)
            # Earlier passed sector/magnitude raw controls are retained. Only
            # the failed product precision comparison is repeated after cast.
            if family == "cross_market" and j == 0:
                np.testing.assert_array_equal(control, expected)
                controls += control.size
            else:
                assert rec["checks"][j]["mismatches"] == 0
                controls += rec["checks"][j]["cells"]
            changed[j][ix] = after
        specs = [
            FeatureSpec(**s)
            for s in m["metadata"]["feature_schema"]["specifications"]
            if s["family"] == full
        ]
        transformed = []
        for raw, valid, age in (prior, changed):
            value, mask = np.zeros_like(raw), np.zeros_like(valid)
            transform_feature_panel_into(raw, valid, active, specs, value, mask)
            age[~active] = -1
            transformed.append((value, mask, age))
        result = {}
        for j, suffix in enumerate(("values", "valid", "age_sessions")):
            key = full + "_" + suffix
            before = np.load(root / m["arrays"][key]["path"], mmap_mode="r")[rows]
            np.testing.assert_array_equal(transformed[0][j], before, err_msg=key)
            controls += before.size
            after = transformed[1][j]
            same = (before == after) | (np.isnan(before) & np.isnan(after))
            local = np.argwhere(~same)
            ix = local.copy()
            ix[:, 0] = rows[local[:, 0]]
            deltas[key + "__indices"] = ix
            deltas[key + "__values"] = after[tuple(local.T)]
            result[suffix] = len(local)
            if j == 1:
                result["gains"] = int((after & ~before).sum())
                result["losses"] = int((before & ~after).sum())
            np.testing.assert_array_equal(
                after[:1], before[:1], err_msg="pre-effect:" + key
            )
            np.testing.assert_array_equal(
                after[-3:], before[-3:], err_msg="clean-tail:" + key
            )
        effects[family] = result
        np.savez_compressed(
            out / (family + "_rows.npz"),
            values=changed[0],
            valid=changed[1],
            ages=changed[2],
        )
    np.savez_compressed(out / "deltas.npz", **deltas)
    result = {
        "status": "passed_original_typed_family_controls",
        "producer": run["natura_auxiliary_propagation"],
        "effects": effects,
        "control_cells": controls,
        "deltas": binding(out / "deltas.npz"),
        "precision": "Original panel_frame casts all physical fields to Float32 before family alignment. Initial Float64 product comparisons were harness mismatches; saved regressions/shocks/portfolios are reused without rerunning.",
        "seconds": perf_counter() - started,
    }
    write_json_atomic(out / "manifest.json", result)
    run["natura_auxiliary_qualification"] = binding(out / "manifest.json")
    daily = bound_json(run["natura_daily_propagation"])
    run["natura_dependency_scope"] = binding(
        Path(daily["plan"]["path"]).parent / "dependency_scope/manifest.json"
    )
    write_json_atomic(pointer, run)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
