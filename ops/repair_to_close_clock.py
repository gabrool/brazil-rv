"""Recover only admitted target inputs and attribute dated normalization."""

import argparse
import json
import shutil
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from brazil_rv.preprocessing.io import validate_physical_source_identity
from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.decision_clock import load_session_schedule
from brazil_rv.v2.round5_derived import verified
from brazil_rv.v2.targets import build_to_close_target

PROJECT = Path(__file__).resolve().parents[1]
KEYS = {
    "target_to_close": "target",
    "target_to_close_valid": "valid",
    "target_to_close_normalized_residual": "normalized_residual",
    "target_to_close_raw_log_return": "raw_log_return",
}


def minutes(t):
    return t.hour * 60 + t.minute


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-inputs", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    source = bound_json(run["m1_scalar_assembly"])
    m1 = bound_json(run["rename_m1_propagation"])
    store = Path(source["parent"]["root"])
    root = Path(run["root"]) / "to_close_clock"
    root.mkdir(exist_ok=args.reuse_inputs)
    shutil.copyfile(
        __file__,
        root
        / (
            "qualified_reproducer.py" if args.reuse_inputs else "executed_reproducer.py"
        ),
    )
    dates = np.load(store / "date_index.npy")
    isins = np.load(store / "isin_index.npy")
    baseline = {k: np.load(store / (k + ".npy")).copy() for k in KEYS}
    with np.load(verified(source["deltas"])) as delta:
        for key in delta.files:
            if key.endswith("__indices") and key.split("__")[1] in baseline:
                baseline[key.split("__")[1]][tuple(delta[key].T)] = delta[
                    key.replace("__indices", "__values")
                ]
    valid = baseline["target_to_close_valid"]
    all_sessions = {
        s.trade_date: s for s in load_session_schedule(Path(m1["schedule"]["path"]))
    }
    sessions = [all_sessions[d] for d in dates.astype(object)]
    total = np.array(
        [minutes(s.continuous_close) - minutes(s.continuous_open) for s in sessions]
    )
    prefix = np.array(
        [minutes(s.decision_time) - minutes(s.continuous_open) for s in sessions]
    )
    if args.reuse_inputs:
        with np.load(root / "raw_inputs.npz") as saved:
            inputs = {k: saved[k].copy() for k in ("entry", "close", "sigma")}
        recovery = json.loads((root / "input_recovery.json").read_text())
    else:
        inputs = {
            k: np.full(valid.shape, np.nan, np.float32)
            for k in ("entry", "close", "sigma")
        }
        filled = np.zeros_like(valid)
        reused = []
        scalar_root = Path(run["m1_scalar_assembly"]["path"]).parent.parent
        for effect in ("2023-10-25", "2024-11-18"):
            path = scalar_root / "qualified_control" / effect / "raw_control.npz"
            with np.load(path) as saved:
                rows = saved["date_indices"]
                for key, raw_key in {
                    "entry": "entry",
                    "close": "session_close",
                    "sigma": "realized_daily_vol",
                }.items():
                    inputs[key][rows] = np.where(valid[rows], saved[raw_key], np.nan)
                filled[rows] = valid[rows]
            reused.append(
                {"source": binding(path), "admitted_cells": int(valid[rows].sum())}
            )
        assignments = pl.read_parquet(verified(m1["assignments"]))
        lookup = {isin: n for n, isin in enumerate(isins)}
        scope = []
        for group in assignments.partition_by("source_file"):
            needs = []
            for rec in group.to_dicts():
                n = lookup[rec["isin"]]
                rows = np.flatnonzero(valid[:, n] & ~filled[:, n])
                if not len(rows):
                    continue
                first = date.fromisoformat(rec["first_overlap_date"])
                last = date.fromisoformat(rec["last_overlap_date"])
                allowed = dates[rows].astype(object).tolist()
                assert all(
                    first <= d <= last and d <= date(2024, 12, 30) for d in allowed
                )
                needs.append((n, rows, allowed))
            if not needs:
                continue
            required = sorted({d for _, _, allowed in needs for d in allowed})
            calendar = pl.DataFrame(
                {
                    "trade_date": required,
                    "date_idx": [
                        int(np.searchsorted(dates, np.datetime64(d))) for d in required
                    ],
                    "opening": [
                        minutes(all_sessions[d].continuous_open) for d in required
                    ],
                    "total": [
                        minutes(all_sessions[d].continuous_close)
                        - minutes(all_sessions[d].continuous_open)
                        for d in required
                    ],
                    "prefix": [
                        minutes(all_sessions[d].decision_time)
                        - minutes(all_sessions[d].continuous_open)
                        for d in required
                    ],
                }
            )
            path = Path(group[0, "source_file"])
            bars = (
                pl.scan_parquet(path)
                .filter(pl.col("ts_exchange").dt.date().is_in(required))
                .select("ts_exchange", "symbol", "open", "close")
                .with_columns(
                    pl.col("ts_exchange").dt.date().alias("trade_date"),
                    (
                        pl.col("ts_exchange").dt.hour().cast(pl.Int16) * 60
                        + pl.col("ts_exchange").dt.minute()
                    ).alias("clock"),
                )
                .join(calendar.lazy(), on="trade_date")
                .with_columns((pl.col("clock") - pl.col("opening")).alias("minute_idx"))
                .filter(
                    (
                        (pl.col("minute_idx") >= 0)
                        & (pl.col("minute_idx") < pl.col("prefix"))
                        & (pl.col("minute_idx") % 5 == 4)
                    )
                    | (pl.col("minute_idx") == pl.col("prefix"))
                    | (pl.col("minute_idx") == pl.col("total") - 1)
                )
                .collect()
            )
            validate_physical_source_identity(group, bars, path)
            by_day = {key[0]: frame for key, frame in bars.group_by("date_idx")}
            for n, rows, allowed in needs:
                for day in rows:
                    frame = by_day[int(day)]
                    ix = frame["minute_idx"].to_numpy()
                    assert len(np.unique(ix)) == len(ix)
                    close = np.zeros(total[day], np.float64)
                    seen = np.zeros(total[day], bool)
                    close[ix] = frame["close"].to_numpy()
                    seen[ix] = True
                    blocks, observed = (
                        close[4 : prefix[day] : 5],
                        seen[4 : prefix[day] : 5],
                    )
                    adjacent = (
                        observed[:-1]
                        & observed[1:]
                        & np.isfinite(blocks[:-1])
                        & np.isfinite(blocks[1:])
                        & (blocks[:-1] > 0)
                        & (blocks[1:] > 0)
                    )
                    assert adjacent.sum() >= np.ceil(0.8 * (prefix[day] // 5 - 1))
                    returns = np.zeros(len(blocks) - 1)
                    returns[adjacent] = np.log(
                        blocks[1:][adjacent] / blocks[:-1][adjacent]
                    )
                    entry = frame.filter(pl.col("minute_idx") == int(prefix[day]))[
                        "open"
                    ]
                    assert len(entry) == 1 and seen[total[day] - 1]
                    inputs["entry"][day, n] = entry[0]
                    inputs["close"][day, n] = close[total[day] - 1]
                    inputs["sigma"][day, n] = np.sqrt(np.sum(returns**2))
                    filled[day, n] = True
                scope.append(
                    {
                        "isin": str(isins[n]),
                        "source_file": str(path),
                        "dates": [str(d) for d in allowed],
                        "selected_bar_rows": int(
                            bars.filter(pl.col("date_idx").is_in(rows.tolist())).height
                        ),
                    }
                )
        np.testing.assert_array_equal(filled, valid)
        for value in inputs.values():
            assert np.all(np.isfinite(value[valid]) & (value[valid] > 0))
        np.savez_compressed(
            root / "raw_inputs.npz",
            **inputs,
            valid=valid,
            session_minutes=total,
            cutoff=prefix,
        )
        recovery = {
            "parent": source["parent"],
            "rename_assembly": run["m1_scalar_assembly"],
            "assignments": m1["assignments"],
            "schedule": m1["schedule"],
            "source_audit": binding(store / "m1_source_audit.parquet"),
            "reused": reused,
            "original_source_reads": scope,
            "valid_outcomes": int(valid.sum()),
            "seconds": time.perf_counter() - started,
            "inputs": binding(root / "raw_inputs.npz"),
            "selection": "Only already admitted name/date outcomes: five-minute completed block closes before decision, exact entry open, exact continuous close. Assignment bounded through 2024. No full raw/feature census; original source hashes reuse sealed audit receipts.",
        }
        write_json_atomic(root / "input_recovery.json", recovery)
        print(
            json.dumps({"recovered": int(valid.sum()), "seconds": recovery["seconds"]}),
            flush=True,
        )
    old = build_to_close_target(
        inputs["entry"],
        inputs["close"],
        inputs["sigma"],
        valid,
        valid,
        session_minutes=np.full(len(dates), 405),
        cutoff=np.full(len(dates), 345),
    )
    control = {}
    for key, attr in KEYS.items():
        value = getattr(old, attr)
        same = (value == baseline[key]) | (np.isnan(value) & np.isnan(baseline[key]))
        control[key] = int((~same).sum())
    write_json_atomic(root / "control.json", control)
    assert not any(control.values()), control
    corrected = build_to_close_target(
        inputs["entry"],
        inputs["close"],
        inputs["sigma"],
        valid,
        valid,
        session_minutes=total,
        cutoff=prefix,
    )
    np.testing.assert_array_equal(corrected.valid, old.valid)
    np.testing.assert_array_equal(corrected.raw_log_return, old.raw_log_return)
    deltas, effects = {}, {}
    for key, attr in KEYS.items():
        value = getattr(corrected, attr)
        prior = baseline[key]
        same = (value == prior) | (np.isnan(value) & np.isnan(prior))
        ix = np.argwhere(~same)
        deltas[key + "__indices"], deltas[key + "__values"] = ix, value[tuple(ix.T)]
        effects[key] = len(ix)
    np.savez_compressed(root / "deltas.npz", **deltas)
    oracle_count = 0
    for day in np.flatnonzero(valid.any(axis=1)):
        ok = valid[day]
        raw = np.log(
            inputs["close"][day, ok].astype(float)
            / inputs["entry"][day, ok].astype(float)
        )
        z = raw / (
            inputs["sigma"][day, ok].astype(float)
            * np.sqrt((total[day] - prefix[day]) / total[day])
        )
        z = np.clip(z - np.median(z), -5, 5)
        _, inverse, counts = np.unique(z, return_inverse=True, return_counts=True)
        ranks = (np.cumsum(counts) - counts + (counts - 1) / 2)[inverse]
        rank = ranks / (len(z) - 1) if len(z) > 1 else np.array([0.5])
        np.testing.assert_array_equal(
            rank.astype(np.float32), corrected.target[day, ok]
        )
        np.testing.assert_array_equal(
            z.astype(np.float32), corrected.normalized_residual[day, ok]
        )
        oracle_count += int(ok.sum())
    # Rebuilding an earlier date block cannot see later endpoints or clocks.
    cut = int(
        np.flatnonzero(valid.any(axis=1))[len(np.flatnonzero(valid.any(axis=1))) // 2]
    )
    earlier = build_to_close_target(
        inputs["entry"][:cut],
        inputs["close"][:cut],
        inputs["sigma"][:cut],
        valid[:cut],
        valid[:cut],
        session_minutes=total[:cut],
        cutoff=prefix[:cut],
    )
    for attr in KEYS.values():
        np.testing.assert_array_equal(
            getattr(earlier, attr), getattr(corrected, attr)[:cut]
        )
    write_json_atomic(
        root / "manifest.json",
        {
            "parent": source["parent"],
            "input_recovery": binding(root / "input_recovery.json"),
            "control": binding(root / "control.json"),
            "control_mismatches": 0,
            "deltas": binding(root / "deltas.npz"),
            "effects": effects,
            "valid_outcomes": int(valid.sum()),
            "independent_oracle_outcomes": oracle_count,
            "causal_prefix_end_exclusive": str(dates[cut]),
            "old_clipped": int((np.abs(old.normalized_residual[valid]) == 5).sum()),
            "new_clipped": int(
                (np.abs(corrected.normalized_residual[valid]) == 5).sum()
            ),
            "seconds": time.perf_counter() - started,
            "executed_reproducer": binding(
                root
                / (
                    "qualified_reproducer.py"
                    if args.reuse_inputs
                    else "executed_reproducer.py"
                )
            ),
            "target_code": binding(PROJECT / "research/src/brazil_rv/v2/targets.py"),
            "status": "Clock-only target amendment; not accepted store or refit. Existing RSS estimator, source support, raw returns and five-horizon primary targets unchanged.",
        },
    )
    print(json.dumps({"effects": effects, "oracle": oracle_count}), flush=True)


if __name__ == "__main__":
    main()
