"""Verify actual full933/full60 slow consumers, without forecasts or targets."""

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
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text(encoding="utf8"))
    admission = bound_json(run["surviving_rename_admission"])
    qualified = bound_json(run["surviving_rename_daily_qualification"])
    assert qualified["status"] == "passed"
    source = Path(admission["plan"]["path"]).parent
    out = source / "consumer"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    parent = admission["parent"]
    root = Path(parent["root"])
    m = bound_json(
        dict(path=str(root / "manifest.json"), sha256=parent["manifest_sha256"])
    )
    dates, isins = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    links = pl.read_parquet(admission["history_mapping"]["path"])
    new_effects = [
        int(np.searchsorted(dates, np.datetime64(d)))
        for d in ("2019-08-06", "2024-08-01")
    ]
    first = min(new_effects)
    begin = first - 60
    samples = {e - 1 for e in new_effects}
    for e in new_effects:
        samples.update(range(e, min(e + 61, len(dates))))
    # Each later monthly risk/peer boundary, plus terminal tail and the chained
    # ALSO->ALOS boundary, covers propagation beyond restored eligibility.
    months = dates.astype("datetime64[M]")
    samples.update(
        int(t) for t in np.flatnonzero(months[1:] != months[:-1]) + 1 if t >= first
    )
    samples.add(len(dates) - 1)
    samples.add(int(np.searchsorted(dates, np.datetime64("2023-10-25"))))
    samples = np.array(sorted(samples))
    local = samples - begin
    patches = {}
    for record in (admission["liquidity_deltas"], qualified["deltas"]):
        with np.load(record["path"]) as z:
            for key in z.files:
                if key.endswith("__indices"):
                    name = key.removesuffix("__indices")
                    assert name not in patches
                    patches[name] = (z[key].copy(), z[name + "__values"].copy())
    arrays = {}
    for key in (
        "active",
        "slow_values",
        "slow_valid",
        "slow_age_sessions",
        "slow_timestep_valid",
    ):
        a = np.load(root / m["arrays"][key]["path"], mmap_mode="r")[begin:].copy()
        if key in patches:
            ix, values = patches[key]
            take = ix[:, 0] >= begin
            coords = ix[take].copy()
            coords[:, 0] -= begin
            a[tuple(coords.T)] = values[take]
        arrays[key] = a
    specs = [
        s
        for s in m["metadata"]["feature_schema"]["specifications"]
        if s["family"] == "slow"
    ]
    view = out / "slow_view"
    with StoreStaging(view, dates=dates[begin:], isins=isins) as staging:
        for k, a in arrays.items():
            staging.write_array(k, a)
        staging.seal(
            feature_names={"slow": m["feature_names"]["slow"]},
            sources=[
                run["surviving_rename_admission"],
                run["surviving_rename_daily_qualification"],
            ],
            tables={
                "slow_history_links": links.with_columns(
                    pl.col("effective_index") - begin, pl.col("known_index") - begin
                )
            },
            metadata={
                "purpose": "Slow-input audit only. No target/native/auxiliary completeness, model inference or accepted store.",
                "parent": parent,
                "feature_schema": {
                    "minimum_rank_names": 20,
                    "specifications": specs,
                    "sha256": feature_schema_sha256([FeatureSpec(**s) for s in specs]),
                },
            },
        )
    dataset = V2DailyDataset(
        view,
        local.tolist(),
        stage="finetune",
        lookback=60,
        include_fast=False,
        include_intraday=False,
        target_window_indices=local.tolist(),
    )
    current = [0]
    real_read = dataset.store.read

    def causal_read(name, selector):
        assert (
            np.atleast_1d(np.arange(len(dates) - begin)[selector]).max() <= current[0]
        )
        return real_read(name, selector)

    dataset.store.read = causal_read
    cells = 0
    for i, (global_day, day) in enumerate(zip(samples, local, strict=True)):
        current[0] = int(day)
        sample = dataset[i]
        batch = collate_v2_daily([sample])
        w = np.arange(day - 59, day + 1)
        route = np.broadcast_to(np.arange(933), (60, 933)).copy()
        # Independent per-destination ancestor walk, scoped by the current
        # decision's public knowledge and each historical row's effect boundary.
        known = {
            r["successor_index"]: r
            for r in links.iter_rows(named=True)
            if max(r["effective_index"], r["known_index"]) <= global_day
        }
        for j in known:
            for h, t in enumerate(w + begin):
                name = j
                while name in known and t < known[name]["effective_index"]:
                    name = known[name]["predecessor_index"]
                route[h, j] = name
        idx = (w[:, None], route)
        valid = arrays["slow_valid"][idx].transpose(1, 0, 2)
        history = arrays["slow_timestep_valid"][idx].T
        expected = {
            "active_mask": arrays["active"][day],
            "slow_features": np.where(
                valid, arrays["slow_values"][idx].transpose(1, 0, 2), 0
            ),
            "slow_feature_mask": valid,
            "slow_feature_age_sessions": np.where(
                history[..., None],
                arrays["slow_age_sessions"][idx].transpose(1, 0, 2),
                -1,
            ),
            "slow_history_mask": history,
        }
        assert sample["slow_features"].shape == (933, 60, 32)
        for key, value in expected.items():
            np.testing.assert_array_equal(
                sample[key], value, err_msg=f"{dates[global_day]}:{key}"
            )
            np.testing.assert_array_equal(batch[key][0].numpy(), value)
            cells += value.size * 2
    dataset.store.close()
    report = {
        "status": "qualified_actual_full_population_slow_history_only",
        "admission": run["surviving_rename_admission"],
        "daily": run["surviving_rename_daily_qualification"],
        "parent": parent,
        "view": binding(view / "manifest.json"),
        "samples": len(samples),
        "sample_dates": [str(dates[d]) for d in samples],
        "names": 933,
        "history": 60,
        "packed_cells": cells,
        "mismatches": 0,
        "future_feature_reads": 0,
        "seconds": perf_counter() - tick,
        "remaining": "Common state, issuer/financial, lending-denominator, native/scalar/auxiliary dependencies, target recomposition and complete-store acceptance. Jan25 calendar and integratedA remain; C/D unstarted.",
    }
    write_json_atomic(out / "manifest.json", report)
    run["surviving_rename_daily_input_audit"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            {k: report[k] for k in ("status", "samples", "packed_cells", "seconds")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
