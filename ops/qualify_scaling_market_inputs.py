"""Independent transforms and causal checks for newly composed market inputs."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.data_repair import bound_json
from brazil_rv.v2.decision_clock import load_session_schedule
from brazil_rv.v2.intraday_features import build_native_fast_features
from compose_surviving_store import read_layers
from propagate_surviving_m1 import grid_for, minute
from repair_scaling_inputs import context, record
from verify_surviving_market_inputs import independent


def main():
    tick = perf_counter()
    run, plan, root, m, _ = context()
    workspace = Path(bound_json(run["scaling_data_workspace"])["root"])
    out = workspace / "market_qualification"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    dates, names = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    active = np.load(root / "active.npy").copy()
    with np.load(bound_json(run["scaling_data_identity"])["deltas"]["path"]) as z:
        active[tuple(z["active__indices"].T)] = z["active__values"]
    layers = [
        bound_json(run["scaling_data_" + k])["deltas"]
        for k in ("context", "market", "daily_qualification")
    ]
    patches = read_layers(layers)

    def amended(k, rows):
        a = np.load(root / m["arrays"][k]["path"], mmap_mode="r")[rows].copy()
        for ix, v in patches.get(k, []):
            keep = np.isin(ix[:, 0], rows)
            local = ix[keep].copy()
            local[:, 0] = np.searchsorted(rows, local[:, 0])
            a[tuple(local.T)] = v[keep]
        return a

    checks = []
    for family, folder in (
        ("sector", "context"),
        ("magnitudes", "market"),
        ("cross_market", "market"),
    ):
        with np.load(workspace / folder / (family + "_corrected.npz")) as z:
            rows = z["rows"].copy()
            raw = z["values"].copy()
            valid = z["valid"].copy()
            ages = z["ages"].copy()
        spec = [
            s
            for s in m["metadata"]["feature_schema"]["specifications"]
            if s["family"] == "sidecar_" + family
        ]
        cells = 0
        for first in range(0, len(rows), 64):
            sl = slice(first, first + 64)
            selected = rows[sl]
            value, mask = independent(raw[sl], valid[sl], active[selected], spec)
            expected = [
                value,
                mask,
                np.where(active[selected, ..., None], ages[sl], -1),
            ]
            for key, a in zip(
                ("values", "valid", "age_sessions"), expected, strict=True
            ):
                np.testing.assert_array_equal(
                    a, amended("sidecar_" + family + "_" + key, selected)
                )
                cells += a.size
        checks.append(dict(family=family, cells=cells, mismatches=0))
    native = bound_json(run["scaling_data_m1"])
    schedule = {
        s.trade_date: s
        for s in load_session_schedule(
            root / m["tables"]["b3_session_schedule"]["path"]
        )
    }
    mutations = []
    for rec in native["cases"]:
        n = names.index(rec["isin"])
        path = Path(rec["source"]["path"])
        source = pl.read_parquet(path)
        rows = np.load(rec["date_indices"]["path"])
        with np.load(path.parent / (rec["isin"] + "_native.npz")) as z:
            all_values, all_valid = z["values"].copy(), z["valid"].copy()
        for pos in (0, len(rows) // 2, len(rows) - 1):
            t = int(rows[pos])
            s = schedule[dates[t].astype(object)]
            grid, seen, _ = grid_for(
                source.filter(pl.col("ts_exchange").dt.date() == s.trade_date),
                {s.trade_date},
                (s,),
                path,
            )
            prefix = minute(s.decision_time) - minute(s.continuous_open)
            future = grid.copy()
            future[:, prefix:, :4] *= 1.73
            future[:, prefix:, 4] *= 2.0
            results = []
            for current in (grid, future):
                result = build_native_fast_features(
                    *[current[:, None, :, j] for j in (1, 2, 3, 4)],
                    seen[:, None],
                    volume_valid=seen[:, None],
                    session_valid=seen.any(axis=1)[:, None],
                    sigma_asof=amended("target_scale_sigma", np.array([t]))[:, [n]],
                    sessions=(s,),
                    max_patches=all_values.shape[1],
                )
                results.append(result)
                np.testing.assert_array_equal(
                    result.values[0, 0, :, :2], all_values[pos]
                )
                np.testing.assert_array_equal(result.valid[0, 0, :, :2], all_valid[pos])
            np.testing.assert_array_equal(results[0].patch_mask, results[1].patch_mask)
            mutations.append(
                dict(
                    isin=rec["isin"],
                    date=str(dates[t]),
                    entry_and_later_minutes_mutated=True,
                    price_fields_exact=True,
                )
            )
    result = dict(
        status="qualified_market_formulas_and_native_causality",
        market_checks=checks,
        native_mutations=mutations,
        inputs={
            k: run["scaling_data_" + k] for k in ("market", "context", "m1", "scalars")
        },
        scalar_scope="Completed scalar source-clock oracle/future-predecessor isolation reused; no new scalar observations or optional to-close outcomes.",
        seconds=perf_counter() - tick,
    )
    record(run, "scaling_data_minute_qualification", out / "manifest.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
