"""Propagate surviving-name risk scales and eligibility through existing M1 rules."""

import argparse
import json
import shutil
from datetime import date
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.preprocessing.io import (
    SOURCE_COLUMNS,
    dense_grid,
    validate_physical_source_identity,
    validate_session_bars,
)
from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.contract import INTRADAY_DAILY_FEATURES
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.decision_clock import load_session_schedule
from brazil_rv.v2.feature_spec import (
    feature_specs,
    observation_age_sessions_into,
    transform_feature_panel_into,
)
from brazil_rv.v2.intraday_features import (
    build_intraday_daily_features,
    build_native_fast_features,
)
from brazil_rv.v2.targets import build_to_close_target
from assemble_m1_scalars import FIELDS

PROJECT = Path(__file__).resolve().parents[1]


def minute(value):
    return value.hour * 60 + value.minute


def same(a, b):
    return (a == b) | (np.isnan(a) & np.isnan(b)) if a.dtype.kind == "f" else a == b


def grid_for(source, allowed, sessions, path):
    schedule = pl.DataFrame(
        dict(
            trade_date=[s.trade_date for s in sessions],
            date_idx=range(len(sessions)),
            opening=[minute(s.continuous_open) for s in sessions],
            count=[
                minute(s.continuous_close) - minute(s.continuous_open) for s in sessions
            ],
        )
    )
    bars = (
        source.with_columns(
            pl.col("ts_exchange").dt.date().alias("trade_date"),
            (
                pl.col("ts_exchange").dt.hour().cast(pl.Int16) * 60
                + pl.col("ts_exchange").dt.minute()
            ).alias("clock"),
        )
        .filter(pl.col("trade_date").is_in(tuple(allowed)))
        .join(schedule, on="trade_date")
        .with_columns(
            (pl.col("clock") - pl.col("opening")).cast(pl.Int16).alias("minute_idx")
        )
        .filter(pl.col("minute_idx") >= 0, pl.col("minute_idx") < pl.col("count"))
        .sort("ts_exchange")
    )
    validate_session_bars(bars, path)
    grid, seen = dense_grid(bars, len(sessions), max(schedule["count"]))
    return grid, seen, bars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    resume = parser.parse_args().resume
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    admission = bound_json(run["surviving_rename_admission"])
    root = Path(admission["parent"]["root"])
    m = bound_json(
        dict(
            path=str(root / "manifest.json"),
            sha256=admission["parent"]["manifest_sha256"],
        )
    )
    source_root = Path(admission["plan"]["path"]).parent
    out = source_root / "m1"
    out.mkdir(exist_ok=resume)
    assert not (out / "manifest.json").exists(), "Reuse completed outputs"
    (out / ("executed_resume.py" if resume else "executed.py")).write_bytes(
        Path(__file__).read_bytes()
    )
    if resume and not (out / "initial_failure").exists():
        failed = out / "initial_failure"
        failed.mkdir()
        for filename in [
            "control_scalars.npz",
            "corrected_scalars.npz",
            "controls.json",
        ]:
            shutil.copyfile(out / filename, failed / filename)
    dates = np.load(root / "date_index.npy")
    isins = np.load(root / "isin_index.npy").tolist()
    assignment_record = next(
        s
        for s in m["sources"]
        if Path(s["path"]).name == "xp_accepted_source_assignments_v1.parquet"
    )
    assignments = pl.read_parquet(assignment_record["path"])
    predecessors = ["BRSSBRACNOR1", "BRARZZACNOR3"]
    assert assignments.filter(pl.col("isin").is_in(predecessors)).is_empty()
    schedule_record = binding(root / m["tables"]["b3_session_schedule"]["path"])
    schedule = {
        s.trade_date: s for s in load_session_schedule(Path(schedule_record["path"]))
    }
    qualified = bound_json(run["surviving_rename_daily_qualification"])
    old_sigma = np.load(root / "target_scale_sigma.npy", mmap_mode="r")
    sigma = old_sigma.copy()
    with np.load(qualified["deltas"]["path"]) as z:
        sigma[tuple(z["target_scale_sigma__indices"].T)] = z[
            "target_scale_sigma__values"
        ]
    old_active = np.load(root / "active.npy", mmap_mode="r")
    active = old_active.copy()
    with np.load(admission["liquidity_deltas"]["path"]) as z:
        active[tuple(z["active__indices"].T)] = z["active__values"]
    if not resume:
        write_json_atomic(
            out / "plan.json",
            dict(
                parent=admission["parent"],
                admission=run["surviving_rename_admission"],
                daily=run["surviving_rename_daily_qualification"],
                assignments=assignment_record,
                schedule=schedule_record,
                registration=binding(
                    PROJECT / "research/preregistrations/v2_economic_data_scaling.md"
                ),
                closeout=run["stage_a_closeout_plan"],
                contrast="Only corrected daily sigma and newly qualified eligibility. SSBR/ARZZ have no accepted M1 assignment; no predecessor bars or source-age inheritance is invented. Native channels0/1 consume same-day pre-decision blocks and corrected prior-close sigma; every other native channel and its original20-session volume history are reused. Scalar raw reducers retain45prior sessions, original source assignments,20-session rules, min-rank support, and dated to-close normalization. All933 axes, no old-fit or accepted-store mutation.",
                verification="Exact original-coordinate controls; independent native block arithmetic and entry-bar exclusion; raw scalar/target support controls, seeded own-source age, independent target rank arithmetic and future-prefix deletion. Completed source census and previous tensor proofs are not repeated.",
            ),
        )
    mapping = pl.read_parquet(
        root / m["tables"]["native_fast_security_mapping"]["path"]
    )
    fast = dict(zip(mapping["store_name_index"], mapping["fast_index"], strict=True))
    native_reports, native_delta = [], {}
    native_report_path = out / "native.json"
    if native_report_path.exists():
        native_reports = json.loads(native_report_path.read_text())["cases"]
    else:
        for n in np.flatnonzero((~same(sigma, old_sigma)).any(axis=0)):
            if n not in fast:
                continue
            rows = np.flatnonzero(~same(sigma[:, n], old_sigma[:, n]))
            name = isins[n]
            assignment = assignments.filter(pl.col("isin") == name)
            assert assignment.height == 1
            rec = assignment.row(0, named=True)
            first, last = (
                date.fromisoformat(rec["first_overlap_date"]),
                date.fromisoformat(rec["last_overlap_date"]),
            )
            rows = rows[
                (dates[rows] >= np.datetime64(first))
                & (dates[rows] <= np.datetime64(last))
            ]
            if not len(rows):
                continue
            calendar = dates[rows].astype(object).tolist()
            sessions = tuple(schedule[d] for d in calendar)
            path = Path(rec["source_file"])
            selected = (
                pl.scan_parquet(path)
                .filter(pl.col("ts_exchange").dt.date().is_in(calendar))
                .select(SOURCE_COLUMNS)
                .collect()
            )
            validate_physical_source_identity(assignment, selected, path)
            daily_sources = [
                s
                for s in m["sources"]
                if Path(s["path"]).name
                in {f"equities_daily_{d.year}.parquet" for d in calendar}
            ]
            original_dates = set(
                pl.concat(
                    [
                        pl.scan_parquet(s["path"])
                        .filter(
                            pl.col("isin") == name, pl.col("trade_date").is_in(calendar)
                        )
                        .select("trade_date")
                        .collect()
                        for s in daily_sources
                    ]
                )["trade_date"]
            )
            grid, seen, bars = grid_for(selected, original_dates, sessions, path)
            bars.select(SOURCE_COLUMNS).write_parquet(
                out / (name + "_native_source.parquet")
            )
            outputs = []
            for scale in (old_sigma[rows, n], sigma[rows, n]):
                result = build_native_fast_features(
                    *[grid[:, None, :, j] for j in (1, 2, 3, 4)],
                    seen[:, None],
                    volume_valid=seen[:, None],
                    session_valid=seen.any(axis=1)[:, None],
                    sigma_asof=scale[:, None],
                    sessions=sessions,
                    max_patches=np.load(
                        root / "fast_patch_mask.npy", mmap_mode="r"
                    ).shape[-1],
                )
                outputs.append((result.values[:, 0, :, :2], result.valid[:, 0, :, :2]))
            np.savez_compressed(
                out / (name + "_native.npz"),
                rows=rows,
                control_values=outputs[0][0],
                control_valid=outputs[0][1],
                values=outputs[1][0],
                valid=outputs[1][1],
            )
            controls, oracle_cells = 0, 0
            for j, key in enumerate(("fast_patch_values", "fast_patch_valid")):
                old = np.load(root / (key + ".npy"), mmap_mode="r")[
                    rows, fast[n], :, :2
                ]
                np.testing.assert_array_equal(outputs[0][j], old, err_msg=name + key)
                controls += old.size
                ix = np.argwhere(~same(old, outputs[1][j]))
                native_delta[name + "__" + key + "__indices"] = np.column_stack(
                    (rows[ix[:, 0]], np.full(len(ix), fast[n]), ix[:, 1:])
                )
                native_delta[name + "__" + key + "__values"] = outputs[1][j][
                    tuple(ix.T)
                ]
            # Independent scalar block arithmetic uses original printed M1 values.
            for i, session in enumerate(sessions):
                total = minute(session.continuous_close) - minute(
                    session.continuous_open
                )
                blocks = (
                    minute(session.decision_time) - minute(session.continuous_open)
                ) // 5
                scale = float(sigma[rows[i], n]) * np.sqrt(5 / total)
                for p in range(blocks):
                    a, b = p * 5, (p + 1) * 5
                    block = grid[i, a:b].astype(float)
                    valid_price = (
                        seen[i, a:b]
                        & np.isfinite(block[:, 1:4]).all(axis=1)
                        & (block[:, 2] > 0)
                        & (block[:, 1] >= block[:, 2])
                        & (block[:, 3] >= block[:, 2])
                        & (block[:, 3] <= block[:, 1])
                    )
                    ok = bool(
                        valid_price.all()
                        and np.isfinite(scale)
                        and sigma[rows[i], n] > 1e-8
                    )
                    assert ok == outputs[1][1][i, p, 1]
                    if ok:
                        value = np.float32(
                            np.log(max(block[:, 1]) / min(block[:, 2])) / scale
                        )
                        assert value == outputs[1][0][i, p, 1]
                    ok = bool(
                        p
                        and seen[i, b - 1]
                        and seen[i, a - 1]
                        and grid[i, b - 1, 3] > 0
                        and grid[i, a - 1, 3] > 0
                        and np.isfinite(scale)
                        and sigma[rows[i], n] > 1e-8
                    )
                    assert ok == outputs[1][1][i, p, 0]
                    if ok:
                        value = np.float32(
                            np.log(float(grid[i, b - 1, 3]) / float(grid[i, a - 1, 3]))
                            / scale
                        )
                        assert value == outputs[1][0][i, p, 0]
                    oracle_cells += 2
            native_reports.append(
                dict(
                    isin=name,
                    rows=len(rows),
                    first=str(dates[rows[0]]),
                    last=str(dates[rows[-1]]),
                    source=binding(out / (name + "_native_source.parquet")),
                    original_source=str(path),
                    control_cells=controls,
                    independent_block_cells=oracle_cells,
                    changes=int((~same(outputs[1][0], outputs[0][0])).sum()),
                    gains=int((outputs[1][1] & ~outputs[0][1]).sum()),
                    losses=int((outputs[0][1] & ~outputs[1][1]).sum()),
                )
            )
            print(json.dumps(native_reports[-1]), flush=True)
        np.savez_compressed(out / "native_deltas.npz", **native_delta)
        write_json_atomic(
            native_report_path,
            dict(
                cases=native_reports,
                deltas=binding(out / "native_deltas.npz"),
                seconds=perf_counter() - tick,
            ),
        )

    # Neither 2019 identity has any M1 observations; the newly eligible ALSO
    # dates cannot alter scalar ranks, native history or to-close support.
    also, azza = isins.index("BRALSOACNOR5"), isins.index("BRAZZAACNOR9")
    added2019 = np.flatnonzero(active[:, also] & ~old_active[:, also])
    for key in ("intraday_valid", "fast_present", "target_to_close_valid"):
        assert not np.load(root / (key + ".npy"), mmap_mode="r")[added2019, also].any()
    effect = int(np.flatnonzero(dates == np.datetime64("2024-08-01"))[0])
    last_change = int(np.flatnonzero((active != old_active).any(axis=1))[-1])
    rows = np.arange(effect - 45, last_change + 2)
    calendar = dates[rows].astype(object).tolist()
    sessions = tuple(schedule[d] for d in calendar)
    keep = np.flatnonzero(rows >= effect - 1)
    output_rows = rows[keep]
    raw_path = out / "scalar_raw.npz"
    if raw_path.exists():
        with np.load(raw_path) as z:
            raw = {k: z[k].copy() for k in FIELDS}
    else:
        shape = (len(rows), len(isins))
        raw = {}
        for key in FIELDS:
            dims = (
                (*shape, 20)
                if key in {"values", "valid", "support_fraction", "source_age_sessions"}
                else shape
            )
            dtype = (
                bool
                if key
                in {
                    "valid",
                    "return_consistent",
                    "entry_valid",
                    "session_close_valid",
                    "fast_present",
                }
                else np.float32
            )
            fill = (
                False
                if dtype is bool
                else (
                    -1
                    if key == "source_age_sessions"
                    else (0 if key in {"values", "support_fraction"} else np.nan)
                )
            )
            raw[key] = np.full(dims, fill, dtype=dtype)
        daily_sources = [
            s
            for s in m["sources"]
            if Path(s["path"]).name
            in {f"equities_daily_{d.year}.parquet" for d in calendar}
        ]
        daily = pl.concat(
            [
                pl.scan_parquet(s["path"])
                .filter(pl.col("trade_date").is_between(calendar[0], calendar[-1]))
                .select("isin", "trade_date")
                .collect()
                for s in daily_sources
            ]
        ).unique()
        original_dates = {k[0]: set(v["trade_date"]) for k, v in daily.group_by("isin")}
        daily.write_parquet(out / "scalar_source_dates.parquet")
        price = np.round(
            np.asarray(
                np.load(root / "raw_close.npy", mmap_mode="r")[rows], dtype=float
            ),
            2,
        )
        q, cash, resolved, successor, has_action, seen_daily, boundary = [
            np.load(root / (k + ".npy"), mmap_mode="r")[rows]
            for k in (
                "action_shares_per_prior_share",
                "action_cash_per_prior_share",
                "action_session_resolved",
                "action_successor_index",
                "action_has_action",
                "observed",
                "intraday_unit_or_unresolved_boundary_mask",
            )
        ]
        receipts = []
        subset_dir = out / "scalar_sources"
        subset_dir.mkdir(exist_ok=True)
        for group in assignments.partition_by("source_file"):
            needs = []
            for rec in group.iter_rows(named=True):
                first, last = (
                    date.fromisoformat(rec["first_overlap_date"]),
                    date.fromisoformat(rec["last_overlap_date"]),
                )
                allowed = {
                    d for d in original_dates.get(rec["isin"], ()) if first <= d <= last
                }
                if allowed:
                    needs.append((rec, isins.index(rec["isin"]), allowed))
            if not needs:
                continue
            path = Path(group[0, "source_file"])
            source = (
                pl.scan_parquet(path)
                .filter(
                    pl.col("ts_exchange")
                    .dt.date()
                    .is_between(calendar[0], calendar[-1])
                )
                .select(SOURCE_COLUMNS)
                .collect()
            )
            validate_physical_source_identity(group, source, path)
            used = set()
            for rec, n, allowed in needs:
                assert not (used & allowed)
                used |= allowed
                grid, seen, bars = grid_for(source, allowed, sessions, path)
                subset = subset_dir / (rec["isin"] + ".parquet")
                bars.select(SOURCE_COLUMNS).write_parquet(subset)
                terminal = np.take_along_axis(price, successor, axis=1)[:, n]
                terminal_seen = np.take_along_axis(seen_daily, successor, axis=1)[:, n]
                refs = np.full((len(rows), 1), np.nan)
                valid_return = seen_daily[:-1, n] & terminal_seen[1:] & resolved[1:, n]
                refs[1:, 0] = np.where(
                    valid_return,
                    np.log((q[1:, n] * terminal[1:] + cash[1:, n]) / price[:-1, n]),
                    np.nan,
                )
                result = build_intraday_daily_features(
                    *[grid[:, None, :, j] for j in range(5)],
                    seen[:, None],
                    volume_valid=seen[:, None],
                    session_valid=seen.any(axis=1)[:, None],
                    sessions=sessions,
                    official_log_return=refs,
                    completed_action_boundary=(
                        has_action[:, n] | ~resolved[:, n] | (successor[:, n] != n)
                    )[:, None],
                    same_day_boundary=boundary[:, n, None],
                )
                entry_minute = np.array(
                    [
                        minute(s.decision_time) - minute(s.continuous_open)
                        for s in sessions
                    ]
                )
                entry = grid[np.arange(len(rows)), entry_minute, 0]
                raw["entry"][:, n] = entry
                raw["entry_valid"][:, n] = (
                    seen[np.arange(len(rows)), entry_minute]
                    & np.isfinite(entry)
                    & (entry > 0)
                )
                for key in FIELDS:
                    if key not in {"entry", "entry_valid"}:
                        raw[key][:, n] = getattr(result, key)[:, 0]
                receipts.append(
                    dict(
                        isin=rec["isin"],
                        original_source=str(path),
                        subset=binding(subset),
                        bars=bars.height,
                        dates=len(allowed),
                    )
                )
        np.savez_compressed(raw_path, date_indices=rows, **raw)
        write_json_atomic(out / "scalar_sources.json", receipts)

    # The canonical stream writes only through the final assignment/COTAHIST
    # date. A full-window scratch reducer must not emit trailing rolling data
    # after that per-security stop. Reuse every already recovered raw input.
    daily = pl.read_parquet(out / "scalar_source_dates.parquet")
    trims = []
    for rec in assignments.iter_rows(named=True):
        first, last = (
            date.fromisoformat(rec["first_overlap_date"]),
            date.fromisoformat(rec["last_overlap_date"]),
        )
        allowed = daily.filter(
            pl.col("isin") == rec["isin"], pl.col("trade_date").is_between(first, last)
        )["trade_date"]
        if not len(allowed):
            continue
        after = dates[rows] > np.datetime64(max(allowed))
        n = isins.index(rec["isin"])
        count = 0
        for key in FIELDS:
            fill = (
                False
                if raw[key].dtype.kind == "b"
                else (
                    -1
                    if key == "source_age_sessions"
                    else (0 if key in {"values", "support_fraction"} else np.nan)
                )
            )
            count += int(
                (
                    ~same(raw[key][after, n], np.full_like(raw[key][after, n], fill))
                ).sum()
            )
            raw[key][after, n] = fill
        if count:
            trims.append(
                dict(isin=rec["isin"], stop=str(max(allowed)), raw_cells=count)
            )
    np.savez_compressed(out / "scalar_raw_qualified.npz", date_indices=rows, **raw)
    write_json_atomic(
        out / "source_window_qualification.json",
        dict(
            initial_raw=binding(raw_path),
            changes=trims,
            reason="Match canonical stream per-security final assigned COTAHIST date. Reuse saved raw reducers; no source reread.",
        ),
    )
    specs = feature_specs("intraday", INTRADAY_DAILY_FEATURES)
    minutes = np.array(
        [minute(s.continuous_close) - minute(s.continuous_open) for s in sessions]
    )
    cutoff = np.array(
        [minute(s.decision_time) - minute(s.continuous_open) for s in sessions]
    )
    controls, outputs = {}, {}
    for label, membership in [("control", old_active), ("corrected", active)]:
        values = np.empty((len(keep), len(isins), 20), np.float32)
        valid = np.empty_like(values, dtype=bool)
        transform_feature_panel_into(
            raw["values"],
            raw["valid"],
            membership[rows],
            specs,
            values,
            valid,
            source_rows=keep,
        )
        target = build_to_close_target(
            raw["entry"],
            np.where(raw["session_close_valid"], raw["session_close"], np.nan),
            raw["realized_daily_vol"],
            membership[rows],
            raw["fast_present"] & raw["entry_valid"] & raw["return_consistent"],
            session_minutes=minutes,
            cutoff=cutoff,
        )
        arrays = dict(
            intraday_values=values,
            intraday_valid=valid,
            target_to_close=target.target[keep],
            target_to_close_valid=target.valid[keep],
            target_to_close_normalized_residual=target.normalized_residual[keep],
            target_to_close_raw_log_return=target.raw_log_return[keep],
        )
        np.savez_compressed(
            out / (label + "_scalars.npz"), date_indices=output_rows, **arrays
        )
        if label == "control":
            controls = arrays
        else:
            outputs = arrays
    checks = []
    for key, value in controls.items():
        expected = np.load(root / (key + ".npy"), mmap_mode="r")[output_rows]
        checks.append(
            dict(
                key=key,
                cells=value.size,
                mismatches=int((~same(value, expected)).sum()),
            )
        )
    for raw_key, key in [
        ("fast_present", "fast_present"),
        ("return_consistent", "m1_cotahist_return_consistent_mask"),
        ("support_fraction", "intraday_support_fraction"),
    ]:
        value = raw[raw_key][keep]
        expected = np.load(root / (key + ".npy"), mmap_mode="r")[output_rows]
        checks.append(
            dict(
                key=key,
                cells=value.size,
                mismatches=int((~same(value, expected)).sum()),
            )
        )
    write_json_atomic(out / "controls.json", checks)
    assert not any(c["mismatches"] for c in checks), checks
    # The successor has no earlier M1 source clock. Preserve all other sealed
    # ages; only current membership and its own newly available raw data matter.
    age = np.load(root / "intraday_age_sessions.npy", mmap_mode="r")[output_rows].copy()
    own_age = np.empty((len(keep), 1, 20), np.float32)
    observation_age_sessions_into(
        raw["valid"][:, azza : azza + 1],
        active[rows, azza : azza + 1],
        own_age,
        source_rows=keep,
        source_age_sessions=raw["source_age_sessions"][:, azza : azza + 1],
    )
    np.testing.assert_array_equal(own_age[-1, 0], age[-1, azza])
    age[:, azza] = own_age[:, 0]
    age[~active[output_rows]] = -1
    outputs["intraday_age_sessions"] = age
    np.savez_compressed(
        out / "corrected_scalars.npz", date_indices=output_rows, **outputs
    )
    deltas, effects = {}, {}
    for key, value in outputs.items():
        before = np.load(root / (key + ".npy"), mmap_mode="r")[output_rows]
        ix = np.argwhere(~same(value, before))
        deltas[key + "__indices"] = np.column_stack((output_rows[ix[:, 0]], ix[:, 1:]))
        deltas[key + "__values"] = value[tuple(ix.T)]
        effects[key] = len(ix)
    np.savez_compressed(out / "scalar_deltas.npz", **deltas)
    report = dict(
        status="produced_exact_controls_pending_independent_consumer_qualification",
        parent=admission["parent"],
        plan=binding(out / "plan.json"),
        native=binding(native_report_path),
        native_deltas=binding(out / "native_deltas.npz"),
        scalar_raw=binding(out / "scalar_raw_qualified.npz"),
        scalar_output=binding(out / "corrected_scalars.npz"),
        scalar_deltas=binding(out / "scalar_deltas.npz"),
        controls=binding(out / "controls.json"),
        effects=effects,
        scalar_gains=int(
            (outputs["intraday_valid"] & ~controls["intraday_valid"]).sum()
        ),
        scalar_losses=int(
            (controls["intraday_valid"] & ~outputs["intraday_valid"]).sum()
        ),
        target_gains=int(
            (
                outputs["target_to_close_valid"] & ~controls["target_to_close_valid"]
            ).sum()
        ),
        target_losses=int(
            (
                controls["target_to_close_valid"] & ~outputs["target_to_close_valid"]
            ).sum()
        ),
        missing_predecessor_assignments=predecessors,
        also_eligible_without_m1=len(added2019),
        seconds=perf_counter() - tick,
    )
    write_json_atomic(out / "manifest.json", report)
    run["surviving_rename_m1"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
