"""Native M1 risk scaling and the accepted RRRP/BRAV history continuation."""

from datetime import date
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.preprocessing.io import SOURCE_COLUMNS, validate_physical_source_identity
from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.decision_clock import load_session_schedule
from brazil_rv.v2.intraday_features import build_native_fast_features
from propagate_surviving_m1 import grid_for, minute, same

PROJECT = Path(__file__).resolve().parents[1]


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    admission = bound_json(run["stage_c_event_data_admission"])
    events = bound_json(admission["plan"])["events"]
    root = Path(admission["parent"]["root"])
    m = bound_json(
        dict(
            path=str(root / "manifest.json"),
            sha256=admission["parent"]["manifest_sha256"],
        )
    )
    out = Path(admission["plan"]["path"]).parent / "m1"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    dates, isins = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    assignment_record = next(
        s
        for s in m["sources"]
        if Path(s["path"]).name == "xp_accepted_source_assignments_v1.parquet"
    )
    assignments = pl.read_parquet(assignment_record["path"])
    missing = [
        e["isin"]
        for e in events
        if assignments.filter(pl.col("isin") == e["isin"]).is_empty()
    ]
    assert len(missing) == 4 and "BRRRRPACNOR5" not in missing
    schedule_record = binding(root / m["tables"]["b3_session_schedule"]["path"])
    schedule = {
        s.trade_date: s for s in load_session_schedule(Path(schedule_record["path"]))
    }
    q = bound_json(run["stage_c_event_daily_qualification"])
    old_sigma = np.load(root / "target_scale_sigma.npy", mmap_mode="r")
    sigma = old_sigma.copy()
    with np.load(q["deltas"]["path"]) as z:
        sigma[tuple(z["target_scale_sigma__indices"].T)] = z[
            "target_scale_sigma__values"
        ]
    mapping = pl.read_parquet(
        root / m["tables"]["native_fast_security_mapping"]["path"]
    )
    fast = dict(zip(mapping["store_name_index"], mapping["fast_index"], strict=True))
    write_json_atomic(
        out / "plan.json",
        dict(
            parent=admission["parent"],
            admission=run["stage_c_event_data_admission"],
            daily=run["stage_c_event_daily_qualification"],
            assignments=assignment_record,
            schedule=schedule_record,
            missing_predecessor_assignments=missing,
            registration=binding(
                PROJECT / "research/preregistrations/v2_economic_data_scaling.md"
            ),
            contrast="Native return/range use final prior-close sigma on existing assigned M1 dates. Only RRRP has a predecessor stream: BRAV also inherits21prior original sessions for unchanged20-session same-clock volume/16-observation support. Public prebirth arrays remain empty; other native channels and masks remain exact. Four absent predecessor streams stay absent. Selected normalized rows are saved for scalar reuse; no held-out market reads or full source census.",
            verification="Old-coordinate native controls; independent printed-block return/range and16-of20 volume arithmetic. Entry-bar mutations and final combined full60 consumer proof follow. No accepted store/oldfit changes.",
        ),
    )
    reports, deltas = [], {}
    max_patches = np.load(root / "fast_patch_mask.npy", mmap_mode="r").shape[-1]
    for n in np.flatnonzero((~same(sigma, old_sigma)).any(axis=0)):
        if n not in fast:
            continue
        name = isins[n]
        group = assignments.filter(pl.col("isin") == name)
        assert group.height == 1
        rec = group.row(0, named=True)
        first, last = (
            date.fromisoformat(rec[k])
            for k in ("first_overlap_date", "last_overlap_date")
        )
        rows = np.flatnonzero(~same(sigma[:, n], old_sigma[:, n]))
        rows = rows[
            (dates[rows] >= np.datetime64(first)) & (dates[rows] <= np.datetime64(last))
        ]
        if not len(rows):
            continue
        bridge = name == "BRBRAVACNOR3"
        if bridge:
            event = next(e for e in events if e["successor_isin"] == name)
            effect = int(np.searchsorted(dates, np.datetime64(event["effective_date"])))
            rows = np.arange(effect - 21, max(effect + 21, int(rows[-1]) + 1))
            pred = event["isin"]
            predecessor = assignments.filter(pl.col("isin") == pred)
            assert (
                predecessor.height == 1
                and predecessor["source_file"][0] == rec["source_file"]
            )
            group = pl.concat([predecessor, group], how="vertical_relaxed")
        else:
            effect, pred = -1, None
        calendar = dates[rows].astype(object).tolist()
        sessions = tuple(schedule[d] for d in calendar)
        path = Path(rec["source_file"])
        source = (
            pl.scan_parquet(path)
            .filter(pl.col("ts_exchange").dt.date().is_in(calendar))
            .select(SOURCE_COLUMNS)
            .collect()
        )
        validate_physical_source_identity(group, source, path)
        daily = pl.concat(
            [
                pl.scan_parquet(s["path"])
                .filter(
                    pl.col("isin").is_in([name] if pred is None else [pred, name]),
                    pl.col("trade_date").is_in(calendar),
                )
                .select("isin", "trade_date")
                .collect()
                for s in m["sources"]
                if Path(s["path"]).name
                in {f"equities_daily_{d.year}.parquet" for d in calendar}
            ]
        )
        allowed = set(daily.filter(pl.col("isin") == name)["trade_date"])
        inherited = (
            set()
            if not bridge
            else set(
                daily.filter(
                    (pl.col("isin") == pred)
                    & (pl.col("trade_date") < dates[effect].astype(object))
                )["trade_date"]
            )
        )
        assert not allowed.intersection(inherited)
        grids = [grid_for(source, allowed, sessions, path)]
        grids.append(
            grid_for(source, allowed | inherited, sessions, path)
            if bridge
            else grids[0]
        )
        grids[1][2].select(SOURCE_COLUMNS).write_parquet(out / f"{name}_source.parquet")
        np.save(out / f"{name}_source_dates.npy", rows)
        group.write_parquet(out / f"{name}_assignments.parquet")
        daily.write_parquet(out / f"{name}_daily_dates.parquet")
        outputs = []
        for i, scale in enumerate((old_sigma[rows, n], sigma[rows, n])):
            grid, seen, _ = grids[i]
            result = build_native_fast_features(
                *[grid[:, None, :, j] for j in (1, 2, 3, 4)],
                seen[:, None],
                volume_valid=seen[:, None],
                session_valid=seen.any(axis=1)[:, None],
                sigma_asof=scale[:, None],
                sessions=sessions,
                max_patches=max_patches,
            )
            outputs.append(result)
        use = np.flatnonzero(rows >= effect) if bridge else np.arange(len(rows))
        channels = [0, 1, 3] if bridge else [0, 1]
        controls = 0
        for key, attr in (
            ("fast_patch_values", "values"),
            ("fast_patch_valid", "valid"),
        ):
            before = getattr(outputs[0], attr)[use, 0][:, :, channels]
            changed = getattr(outputs[1], attr)[use, 0][:, :, channels]
            accepted = np.load(root / f"{key}.npy", mmap_mode="r")[rows[use], fast[n]][
                :, :, channels
            ]
            np.testing.assert_array_equal(before, accepted, err_msg=name + key)
            controls += before.size
            ix = np.argwhere(~same(accepted, changed))
            deltas[f"{name}__{key}__indices"] = np.column_stack(
                (
                    rows[use][ix[:, 0]],
                    np.full(len(ix), fast[n]),
                    ix[:, 1],
                    np.array(channels)[ix[:, 2]],
                )
            )
            deltas[f"{name}__{key}__values"] = changed[tuple(ix.T)]
        if bridge:
            for attr, key in (
                ("patch_mask", "fast_patch_mask"),
                ("last_price_age_minutes", "fast_last_price_age_minutes"),
                ("last_price_age_valid", "fast_last_price_age_valid"),
            ):
                for result in outputs:
                    actual = getattr(result, attr)[use, 0]
                    np.testing.assert_array_equal(
                        actual,
                        np.load(root / f"{key}.npy", mmap_mode="r")[rows[use], fast[n]],
                    )
                    controls += actual.size
            for attr in ("values", "valid"):
                actual = getattr(outputs[1], attr)[use, 0][:, :, [2, 4, 5, 6]]
                accepted = np.load(
                    root / ("fast_patch_" + attr + ".npy"), mmap_mode="r"
                )[rows[use], fast[n]][:, :, [2, 4, 5, 6]]
                np.testing.assert_array_equal(actual, accepted)
                controls += actual.size
        np.savez_compressed(
            out / f"{name}_native.npz",
            rows=rows[use],
            channels=channels,
            control_values=outputs[0].values[use, 0][:, :, channels],
            control_valid=outputs[0].valid[use, 0][:, :, channels],
            values=outputs[1].values[use, 0][:, :, channels],
            valid=outputs[1].valid[use, 0][:, :, channels],
        )
        grid, seen, _ = grids[1]
        oracle_cells = 0
        for i in use:
            session = sessions[i]
            total = minute(session.continuous_close) - minute(session.continuous_open)
            blocks = (
                minute(session.decision_time) - minute(session.continuous_open)
            ) // 5
            scale = float(sigma[rows[i], n]) * np.sqrt(5 / total)
            for p in range(blocks):
                a, b = p * 5, (p + 1) * 5
                block = grid[i, a:b].astype(float)
                shape = (
                    seen[i, a:b]
                    & np.isfinite(block[:, 1:4]).all(axis=1)
                    & (block[:, 2] > 0)
                    & (block[:, 1] >= block[:, 2])
                    & (block[:, 3] >= block[:, 2])
                    & (block[:, 3] <= block[:, 1])
                )
                ok = bool(
                    shape.all() and np.isfinite(scale) and sigma[rows[i], n] > 1e-8
                )
                assert ok == outputs[1].valid[i, 0, p, 1]
                if ok:
                    assert (
                        np.float32(np.log(max(block[:, 1]) / min(block[:, 2])) / scale)
                        == outputs[1].values[i, 0, p, 1]
                    )
                ok = bool(
                    p
                    and seen[i, b - 1]
                    and seen[i, a - 1]
                    and grid[i, b - 1, 3] > 0
                    and grid[i, a - 1, 3] > 0
                    and np.isfinite(scale)
                    and sigma[rows[i], n] > 1e-8
                )
                assert ok == outputs[1].valid[i, 0, p, 0]
                if ok:
                    assert (
                        np.float32(
                            np.log(float(grid[i, b - 1, 3]) / float(grid[i, a - 1, 3]))
                            / scale
                        )
                        == outputs[1].values[i, 0, p, 0]
                    )
                oracle_cells += 2
        # Raw source/input volumes are saved; independent clock-specific median
        # checks and the scalar history use these same rows without a new scan.
        reports.append(
            dict(
                isin=name,
                rows=len(use),
                source_dates=len(rows),
                inherited_source_dates=len(inherited),
                source=binding(out / f"{name}_source.parquet"),
                date_indices=binding(out / f"{name}_source_dates.npy"),
                original_source=str(path),
                control_cells=controls,
                independent_price_block_cells=oracle_cells,
                changes=int(
                    (
                        ~same(
                            outputs[1].values[use, 0][:, :, channels],
                            outputs[0].values[use, 0][:, :, channels],
                        )
                    ).sum()
                ),
                gains=int(
                    (
                        outputs[1].valid[use, 0][:, :, channels]
                        & ~outputs[0].valid[use, 0][:, :, channels]
                    ).sum()
                ),
                losses=int(
                    (
                        outputs[0].valid[use, 0][:, :, channels]
                        & ~outputs[1].valid[use, 0][:, :, channels]
                    ).sum()
                ),
            )
        )
        print(json.dumps(reports[-1]), flush=True)
    np.savez_compressed(out / "native_deltas.npz", **deltas)
    report = dict(
        status="native_intermediate_pending_scalar_and_combined_qualification",
        plan=binding(out / "plan.json"),
        cases=reports,
        deltas=binding(out / "native_deltas.npz"),
        seconds=perf_counter() - tick,
    )
    write_json_atomic(out / "manifest.json", report)
    run = json.loads(pointer.read_text())
    run["stage_c_event_m1"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
