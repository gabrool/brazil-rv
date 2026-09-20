"""Recompose the BRAV scalar cross-section using saved accepted source windows."""

import argparse
import json
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
    admission = bound_json(run["stage_c_event_data_admission"])
    native = bound_json(run["stage_c_event_m1"])
    root = Path(admission["parent"]["root"])
    m = bound_json(
        dict(
            path=str(root / "manifest.json"),
            sha256=admission["parent"]["manifest_sha256"],
        )
    )
    out = Path(admission["plan"]["path"]).parent / "scalars"
    out.mkdir(exist_ok=resume)
    assert not (out / "manifest.json").exists()
    (out / ("executed_resume.py" if resume else "executed.py")).write_bytes(
        Path(__file__).read_bytes()
    )
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
    schedule_record = binding(root / m["tables"]["b3_session_schedule"]["path"])
    schedule = {
        s.trade_date: s for s in load_session_schedule(Path(schedule_record["path"]))
    }
    old_active = np.load(root / "active.npy", mmap_mode="r")
    active = old_active.copy()
    with np.load(admission["deltas"]["path"]) as z:
        active[tuple(z["active__indices"].T)] = z["active__values"]
    p, n = isins.index("BRRRRPACNOR5"), isins.index("BRBRAVACNOR3")
    links = pl.read_parquet(admission["history_mapping"]["path"])
    link = links.filter(pl.col("successor_index") == n).row(0, named=True)
    effect = link["effective_index"]
    last_change = int(np.flatnonzero(active[:, n] != old_active[:, n])[-1])
    rows = np.arange(effect - 45, last_change + 2)
    calendar = dates[rows].astype(object).tolist()
    sessions = tuple(schedule[d] for d in calendar)
    keep = np.flatnonzero(rows >= effect - 1)
    output_rows = rows[keep]
    missing = 0
    for event in bound_json(admission["plan"])["events"]:
        j = isins.index(event["successor_isin"])
        if j == n:
            continue
        added = np.flatnonzero(active[:, j] & ~old_active[:, j])
        missing += len(added)
        for key in ("intraday_valid", "fast_present", "target_to_close_valid"):
            assert not np.load(root / (key + ".npy"), mmap_mode="r")[added, j].any()
    assert missing == 180
    if not resume:
        write_json_atomic(
            out / "plan.json",
            dict(
                parent=admission["parent"],
                admission=run["stage_c_event_data_admission"],
                native=run["stage_c_event_m1"],
                prior_raw_history=run["rename_m1_propagation"],
                assignments=assignment_record,
                schedule=schedule_record,
                rows=[int(rows[0]), int(output_rows[0]), int(rows[-1]) + 1],
                registration=binding(
                    PROJECT / "research/preregistrations/v2_economic_data_scaling.md"
                ),
                contrast="Only RRRP/BRAV has a predecessor M1 stream;180other restored dates remain unsupported. Original45prior/20-session scalar reducers, all933 cross-sections and dated to-close formula. Reuse current native source subsets and the already-qualified ISAE raw21-session overlay intersecting this window. BRAV inherits strictly pre-effect21dates, own reference/knowledge/assignment clocks, and predecessor source ages. Never rescan the original ISAE history or change optional loss (inactive).",
                verification="Exact parent transformed/raw-support/target controls; preserve all recovered raw sources and outputs. Independently seeded source-age control and final combined target/formula/causality qualification; no separate old consumer matrix.",
            ),
        )
    if resume:
        # Resume only the source-age disposition after exact saved scalar controls.
        # Neither raw source/reducer nor passed cross-sectional controls are rerun.
        checks = json.loads((out / "controls.json").read_text())
        assert not any(c["mismatches"] for c in checks)
        with np.load(out / "raw_control.npz") as z:
            raw = {k: z[k].copy() for k in FIELDS}
        corrected = {k: a.copy() for k, a in raw.items()}
        with np.load(out / "brav_history_raw.npz") as z:
            used = z["date_indices"]
            take = used >= max(effect, link["known_index"])
            for k in FIELDS:
                corrected[k][used[take] - rows[0], n] = z[k][take]
        variants = {}
        for label in ("control", "corrected"):
            with np.load(out / f"{label}_scalars.npz") as z:
                variants[label] = {
                    k: z[k].copy() for k in z.files if k != "date_indices"
                }
        reused = json.loads((out / "reused_history.json").read_text())
        finish(
            root,
            out,
            rows,
            output_rows,
            raw,
            corrected,
            variants,
            active,
            old_active,
            effect,
            link,
            p,
            n,
            missing,
            reused,
            tick,
            pointer,
            dates,
        )
        return
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
            cached, covered = [], set()
            for rec0 in native["cases"]:
                if Path(rec0["original_source"]) == path:
                    used_rows = np.load(rec0["date_indices"]["path"])
                    covered.update(dates[used_rows].astype(object).tolist())
                    cached.append(
                        pl.scan_parquet(rec0["source"]["path"])
                        .filter(pl.col("ts_exchange").dt.date().is_in(calendar))
                        .select(SOURCE_COLUMNS)
                        .collect()
                    )
            missing_dates = set(calendar) - covered
            source = (
                pl.scan_parquet(path)
                .filter(pl.col("ts_exchange").dt.date().is_in(tuple(missing_dates)))
                .select(SOURCE_COLUMNS)
                .collect()
            )
            source = pl.concat([source, *cached], how="vertical_relaxed").sort(
                "ts_exchange"
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

    # This window crosses an earlier accepted ISA rename. Reuse its saved raw
    # continuation exactly, instead of rebuilding a previously qualified stream.
    previous = bound_json(run["rename_m1_propagation"])
    reused = []
    for rec in previous["results"]:
        j = isins.index(rec["isin"])
        old_link = (
            pl.read_parquet(root / "slow_history_links.parquet")
            .filter(pl.col("successor_index") == j)
            .row(0, named=True)
        )
        e = old_link["effective_index"]
        if e > rows[-1] or e + 21 <= rows[0]:
            continue
        with np.load(rec["raw_intraday"]["path"]) as z:
            ri = z["date_indices"]
            use = (ri >= max(e, old_link["known_index"], rows[0])) & (
                ri < min(e + 21, rows[-1] + 1)
            )
            for key in FIELDS:
                if key in z:
                    raw[key][ri[use] - rows[0], j] = z[key][use, 1]
        reused.append(
            dict(isin=rec["isin"], source=rec["raw_intraday"], rows=int(use.sum()))
        )
    write_json_atomic(out / "reused_history.json", reused)
    np.savez_compressed(out / "raw_control.npz", date_indices=rows, **raw)
    corrected = {key: a.copy() for key, a in raw.items()}

    # Only the newly admitted RRRP/BRAV stream is reduced here. Source subsets
    # are already saved, and only the original21-prior/21-following rows enter.
    p, n = isins.index("BRRRRPACNOR5"), isins.index("BRBRAVACNOR3")
    begin, end = effect - 21, effect + 21
    lo, hi = begin - rows[0], end - rows[0]
    assert 0 <= lo < hi <= len(rows)
    pair_source = pl.concat(
        [
            pl.read_parquet(out / "scalar_sources" / f"{isins[j]}.parquet")
            for j in (p, n)
        ]
    )
    allowed = set(calendar[lo:hi])
    grid, seen, _ = grid_for(
        pair_source, allowed, sessions[lo:hi], Path("saved_pair_sources")
    )
    price = np.round(
        np.load(root / "raw_close.npy", mmap_mode="r")[rows][:, [p, n]].astype(
            np.float64
        ),
        2,
    )
    observed = np.load(root / "observed.npy", mmap_mode="r")[rows][:, [p, n]]
    refs = np.full((hi - lo, 1), np.nan)
    for t in range(lo + 1, hi):
        current = 0 if rows[t] < effect else 1
        prior = 0 if rows[t - 1] < effect else 1
        if observed[t, current] and observed[t - 1, prior]:
            q = np.load(root / "action_shares_per_prior_share.npy", mmap_mode="r")[
                rows[t], (p, n)[current]
            ]
            cash = np.load(root / "action_cash_per_prior_share.npy", mmap_mode="r")[
                rows[t], (p, n)[current]
            ]
            resolved = np.load(root / "action_session_resolved.npy", mmap_mode="r")[
                rows[t], (p, n)[current]
            ]
            if resolved:
                refs[t - lo, 0] = np.log(
                    (float(q) * price[t, current] + float(cash)) / price[t - 1, prior]
                )
    boundary = np.load(
        root / "intraday_unit_or_unresolved_boundary_mask.npy", mmap_mode="r"
    )[begin:end, n].copy()
    action = np.load(root / "action_has_action.npy", mmap_mode="r")[begin:end, n].copy()
    resolved = np.load(root / "action_session_resolved.npy", mmap_mode="r")[
        begin:end, n
    ].copy()
    successor = np.load(root / "action_successor_index.npy", mmap_mode="r")[
        begin:end, n
    ].copy()
    pre = effect - begin
    boundary[:pre] = np.load(
        root / "intraday_unit_or_unresolved_boundary_mask.npy", mmap_mode="r"
    )[begin:effect, p]
    action[:pre] = np.load(root / "action_has_action.npy", mmap_mode="r")[
        begin:effect, p
    ]
    resolved[:pre] = np.load(root / "action_session_resolved.npy", mmap_mode="r")[
        begin:effect, p
    ]
    completed = action | ~resolved | (successor != n)
    completed[:pre] = (
        action[:pre]
        | ~resolved[:pre]
        | (
            np.load(root / "action_successor_index.npy", mmap_mode="r")[begin:effect, p]
            != p
        )
    )
    result = build_intraday_daily_features(
        *[grid[:, None, :, j] for j in range(5)],
        seen[:, None],
        volume_valid=seen[:, None],
        session_valid=seen.any(axis=1)[:, None],
        sessions=sessions[lo:hi],
        official_log_return=refs,
        completed_action_boundary=completed[:, None],
        same_day_boundary=boundary[:, None],
    )
    entry_minute = np.array(
        [minute(s.decision_time) - minute(s.continuous_open) for s in sessions[lo:hi]]
    )
    entry = grid[np.arange(hi - lo), entry_minute, 0]
    raw_pair = {
        key: getattr(result, key)[:, 0]
        for key in FIELDS
        if key not in ("entry", "entry_valid")
    }
    raw_pair["entry"], raw_pair["entry_valid"] = (
        entry,
        seen[np.arange(hi - lo), entry_minute] & np.isfinite(entry) & (entry > 0),
    )
    np.savez_compressed(
        out / "brav_history_raw.npz",
        date_indices=rows[lo:hi],
        official_log_return=refs,
        **raw_pair,
    )
    gate = max(effect, link["known_index"]) - rows[0]
    for key in FIELDS:
        corrected[key][gate:hi, n] = raw_pair[key][gate - lo :]

    specs = feature_specs("intraday", INTRADAY_DAILY_FEATURES)
    minutes = np.array(
        [minute(s.continuous_close) - minute(s.continuous_open) for s in sessions]
    )
    cutoff = np.array(
        [minute(s.decision_time) - minute(s.continuous_open) for s in sessions]
    )
    variants = {}
    for label, inputs, membership in (
        ("control", raw, old_active),
        ("corrected", corrected, active),
    ):
        values = np.zeros((len(keep), len(isins), 20), np.float32)
        valid = np.zeros_like(values, bool)
        transform_feature_panel_into(
            inputs["values"],
            inputs["valid"],
            membership[rows],
            specs,
            values,
            valid,
            source_rows=keep,
        )
        target = build_to_close_target(
            inputs["entry"],
            np.where(inputs["session_close_valid"], inputs["session_close"], np.nan),
            inputs["realized_daily_vol"],
            membership[rows],
            inputs["fast_present"]
            & inputs["entry_valid"]
            & inputs["return_consistent"],
            session_minutes=minutes,
            cutoff=cutoff,
        )
        variants[label] = dict(
            intraday_values=values,
            intraday_valid=valid,
            intraday_support_fraction=inputs["support_fraction"][keep],
            fast_present=inputs["fast_present"][keep],
            m1_cotahist_return_consistent_mask=inputs["return_consistent"][keep],
            target_to_close=target.target[keep],
            target_to_close_valid=target.valid[keep],
            target_to_close_normalized_residual=target.normalized_residual[keep],
            target_to_close_raw_log_return=target.raw_log_return[keep],
        )
        np.savez_compressed(
            out / f"{label}_scalars.npz", date_indices=output_rows, **variants[label]
        )
    checks = []
    for key, value in variants["control"].items():
        expected = np.load(root / (key + ".npy"), mmap_mode="r")[output_rows]
        checks.append(
            dict(
                key=key,
                cells=value.size,
                mismatches=int((~same(value, expected)).sum()),
                examples=np.argwhere(~same(value, expected))[:4].tolist(),
            )
        )
    write_json_atomic(out / "controls.json", checks)
    assert not any(c["mismatches"] for c in checks), checks

    finish(
        root,
        out,
        rows,
        output_rows,
        raw,
        corrected,
        variants,
        active,
        old_active,
        effect,
        link,
        p,
        n,
        missing,
        reused,
        tick,
        pointer,
        dates,
    )


def finish(
    root,
    out,
    rows,
    output_rows,
    raw,
    corrected,
    variants,
    active,
    old_active,
    effect,
    link,
    p,
    n,
    missing,
    reused,
    tick,
    pointer,
    dates,
):
    old_age = np.load(root / "intraday_age_sessions.npy", mmap_mode="r")
    age = old_age[output_rows].copy()
    mask = np.zeros((int(rows[-1]) + 1, 2, 20), bool)
    source_age = np.full(mask.shape, -1, np.float32)
    seed = old_age[effect - 1, [p, n]]
    for label, inputs, membership, rules in (
        ("control", raw, old_active | active, ()),
        (
            "corrected",
            corrected,
            active,
            (dict(link, predecessor_index=0, successor_index=1),),
        ),
    ):
        mask[effect - 1] = seed >= 0
        source_age[effect - 1] = seed
        use = rows >= effect
        mask[rows[use]] = inputs["valid"][use][:, [p, n]]
        source_age[rows[use]] = inputs["source_age_sessions"][use][:, [p, n]]
        result_age = np.empty((len(output_rows), 2, 20), np.float32)
        observation_age_sessions_into(
            mask,
            membership[: len(mask), [p, n]],
            result_age,
            source_rows=output_rows,
            source_age_sessions=source_age,
            history_links=rules,
        )
        if label == "control":
            expected = old_age[output_rows][:, [p, n]]
            np.testing.assert_array_equal(
                np.where(old_active[output_rows][:, [p, n], None], result_age, -1),
                expected,
            )
        else:
            age[:, [p, n]] = result_age
    age[~active[output_rows]] = -1
    variants["corrected"]["intraday_age_sessions"] = age
    deltas, effects = {}, {}
    for key, value in variants["corrected"].items():
        expected = np.load(root / (key + ".npy"), mmap_mode="r")[output_rows]
        ix = np.argwhere(~same(value, expected))
        deltas[key + "__indices"] = np.column_stack((output_rows[ix[:, 0]], ix[:, 1:]))
        deltas[key + "__values"] = value[tuple(ix.T)]
        effects[key] = len(ix)
    np.savez_compressed(out / "raw_corrected.npz", date_indices=rows, **corrected)
    np.savez_compressed(
        out / "corrected_scalars.npz", date_indices=output_rows, **variants["corrected"]
    )
    # Four inherited source clocks have no later successor observation in the
    # bounded reducer window. Carry their actual ages through the remaining
    # development dates; accepted own-source clocks supersede them when newer.
    tail_rows = np.arange(int(output_rows[-1]) + 1, len(dates))
    last = np.where(age[-1, n] >= 0, output_rows[-1] - age[-1, n], -1)
    tail = old_age[tail_rows, n].copy()
    for local, t in enumerate(tail_rows):
        own = old_age[t, n]
        own_last = np.where(own >= 0, t - own, -1)
        last = np.maximum(last, own_last)
        tail[local] = np.where(last >= 0, t - last, -1)
        if not active[t, n]:
            tail[local] = -1
    ix = np.argwhere(~same(tail, old_age[tail_rows, n]))
    tail_ix = np.column_stack((tail_rows[ix[:, 0]], np.full(len(ix), n), ix[:, 1]))
    key = "intraday_age_sessions"
    deltas[key + "__indices"] = np.concatenate((deltas[key + "__indices"], tail_ix))
    deltas[key + "__values"] = np.concatenate(
        (deltas[key + "__values"], tail[tuple(ix.T)])
    )
    effects[key] += len(ix)
    np.savez_compressed(
        out / "age_tail.npz", date_indices=tail_rows, isin_index=n, age=tail
    )
    write_json_atomic(
        out / "age_tail_disposition.json",
        dict(
            first=str(dates[tail_rows[0]]) if len(tail_rows) else None,
            changed_cells=len(ix),
            last_window_ages=age[-1, n].tolist(),
            reason="Inherited predecessor clocks remain honest elapsed ages even without a later raw observation; newer accepted own-source clocks replace them. No support/value/raw-source change.",
        ),
    )
    np.savez_compressed(out / "deltas.npz", **deltas)
    before, changed = variants["control"], variants["corrected"]
    report = dict(
        status="exact_controls_pending_combined_formula_consumer_qualification",
        plan=binding(out / "plan.json"),
        controls=binding(out / "controls.json"),
        raw_control=binding(out / "raw_control.npz"),
        raw_corrected=binding(out / "raw_corrected.npz"),
        outputs=binding(out / "corrected_scalars.npz"),
        deltas=binding(out / "deltas.npz"),
        effects=effects,
        scalar_gains=int((changed["intraday_valid"] & ~before["intraday_valid"]).sum()),
        scalar_losses=int(
            (before["intraday_valid"] & ~changed["intraday_valid"]).sum()
        ),
        target_gains=int(
            (changed["target_to_close_valid"] & ~before["target_to_close_valid"]).sum()
        ),
        target_losses=int(
            (before["target_to_close_valid"] & ~changed["target_to_close_valid"]).sum()
        ),
        unsupported_restored_dates=missing,
        reused_prior_history=reused,
        seconds=perf_counter() - tick,
    )
    write_json_atomic(out / "manifest.json", report)
    run = json.loads(pointer.read_text())
    run["stage_c_event_scalars"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
