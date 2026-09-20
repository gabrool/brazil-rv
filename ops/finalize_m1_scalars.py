"""Assemble sparse scalar/target amendments from qualified bounded intermediates."""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.contract import INTRADAY_DAILY_FEATURES
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.feature_spec import (
    feature_specs,
    transform_feature_panel_into,
    observation_age_sessions_into,
)
from brazil_rv.v2.round5_derived import verified
from brazil_rv.v2.targets import build_to_close_target

PROJECT = Path(__file__).resolve().parents[1]


def targets(raw, active):
    out = build_to_close_target(
        raw["entry"],
        np.where(raw["session_close_valid"], raw["session_close"], np.nan),
        raw["realized_daily_vol"],
        active,
        raw["fast_present"] & raw["entry_valid"] & raw["return_consistent"],
    )
    return {
        "target_to_close": out.target,
        "target_to_close_valid": out.valid,
        "target_to_close_normalized_residual": out.normalized_residual,
        "target_to_close_raw_log_return": out.raw_log_return,
        "m1_cotahist_return_consistent_mask": raw["return_consistent"],
    }


def equal(a, b):
    return (a == b) | (np.isnan(a) & np.isnan(b)) if a.dtype.kind == "f" else a == b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    m1 = bound_json(run["rename_m1_propagation"])
    history = bound_json(run["rename_history_propagation"])
    store = Path(m1["parent"]["root"])
    root = Path(run["root"]) / "m1_scalar_assembly"
    out = root / "amendments"
    out.mkdir(exist_ok=args.resume)
    shutil.copyfile(
        __file__,
        out / ("qualified_reproducer.py" if args.resume else "executed_reproducer.py"),
    )
    dates, isins = np.load(store / "date_index.npy"), np.load(store / "isin_index.npy")
    links = pl.read_parquet(verified(history["history_mapping"]))
    old_active = np.load(store / "active.npy", mmap_mode="r")
    active = np.load(verified(history["arrays"]["active"]), mmap_mode="r")
    old_age = np.load(store / "intraday_age_sessions.npy", mmap_mode="r")
    old_values = np.load(store / "intraday_values.npy", mmap_mode="r")
    old_valid = np.load(store / "intraday_valid.npy", mmap_mode="r")
    deltas, reports = {}, []
    specs = feature_specs("intraday", INTRADAY_DAILY_FEATURES)
    for rec in m1["results"]:
        prior_output = out / (rec["isin"] + "_output.npz")
        prior_report = out / (rec["isin"] + "_report.json")
        if args.resume and prior_output.exists() and prior_report.exists():
            with np.load(prior_output) as prior:
                for key in prior.files:
                    if key == "date_indices":
                        continue
                    old = np.load(store / (key + ".npy"), mmap_mode="r")[
                        prior["date_indices"]
                    ]
                    changed = np.where(~equal(prior[key], old))
                    deltas[f"{rec['isin']}__{key}__indices"] = np.column_stack(
                        (prior["date_indices"][changed[0]], *changed[1:])
                    )
                    deltas[f"{rec['isin']}__{key}__values"] = prior[key][changed]
            reports.append(json.loads(prior_report.read_text()))
            continue
        n = int(np.flatnonzero(isins == rec["isin"])[0])
        link = links.filter(pl.col("successor_index") == n).row(0, named=True)
        p, effect = link["predecessor_index"], link["effective_index"]
        folder = root / "qualified_control" / str(dates[effect])
        with np.load(folder / "raw_control.npz") as saved:
            rows = saved["date_indices"]
            raw = {k: saved[k].copy() for k in saved.files if k != "date_indices"}
        selected = np.flatnonzero(rows >= effect - 1)
        global_rows = rows[selected]
        control_targets = targets(raw, old_active[rows])
        control_cells = 0
        for key, value in control_targets.items():
            old = np.load(store / (key + ".npy"), mmap_mode="r")[global_rows]
            if key == "m1_cotahist_return_consistent_mask" and not np.array_equal(
                value[selected], old
            ):
                differences = np.argwhere(value[selected] != old)
                assert (
                    differences.tolist() == [[28, 13]]
                    and str(dates[global_rows[28]]) == "2024-12-30"
                )
                source_table = pl.read_parquet(
                    store / "m1_cotahist_level_ratio.parquet"
                ).filter(
                    (pl.col("isin") == "BRAERIACNOR4")
                    & (pl.col("trade_date") == dates[-1].astype(object))
                )
                assert (
                    source_table["completed_action_boundary"][0]
                    and not source_table["return_consistent"][0]
                )
                write_json_atomic(
                    out / "aeri_boundary_qualification.json",
                    {
                        "date": "2024-12-30",
                        "isin": "BRAERIACNOR4",
                        "source_table": binding(
                            store / "m1_cotahist_level_ratio.parquet"
                        ),
                        "alignment_roles": binding(
                            store / "corporate_action_alignment_roles.parquet"
                        ),
                        "sealed_action_arrays_have_q1_no_action": True,
                        "retained_consistency": False,
                        "bounded_reconstruction": True,
                        "target_control_all_four_arrays_exact": True,
                        "disposition": "Sealed diagnostic records retrospective inferred bonus boundary first available Dec31, absent from stored action arrays/terms. Preserve this non-rename cell; underlying final-date corporate boundary discrepancy remains for wealth/label audit. No invented term.",
                    },
                )
                raw["return_consistent"][rows == global_rows[28], 13] = False
            else:
                np.testing.assert_array_equal(value[selected], old, err_msg=key)
                control_cells += old.size
        old_raw = {key: value.copy() for key, value in raw.items()}
        with np.load(verified(rec["raw_intraday"])) as amendment:
            original_rows = amendment["date_indices"]
            take = np.flatnonzero(
                (original_rows >= max(effect, link["known_index"]))
                & (original_rows < effect + 21)
            )
            local = original_rows[take] - rows[0]
            for key in raw:
                if key in amendment.files:
                    raw[key][local, n] = amendment[key][take, 1]
        values = np.empty((len(selected), len(isins), 20), dtype=np.float32)
        valid = np.empty_like(values, dtype=bool)
        transform_feature_panel_into(
            raw["values"],
            raw["valid"],
            active[rows],
            specs,
            values,
            valid,
            source_rows=selected,
        )
        output = {
            "intraday_values": values,
            "intraday_valid": valid,
            "intraday_support_fraction": raw["support_fraction"][selected],
        }
        output.update(
            {key: value[selected] for key, value in targets(raw, active[rows]).items()}
        )
        # Ages of unaffected identities are already established by the sealed
        # source audit. Reconstruct the pair, seeded with its verified last source
        # clocks before this window; no extra raw-source history is fabricated.
        age = np.array(old_age[global_rows], copy=True)
        names = [p, n]
        mask = np.zeros((int(rows[-1]) + 1, 2, 20), dtype=bool)
        source_age = np.full(mask.shape, -1, dtype=np.float32)
        seed = old_age[effect - 1, names]
        mask[effect - 1] = seed >= 0
        source_age[effect - 1] = seed
        for label, source, membership, use_links in [
            ("control", old_raw, old_active | active, ()),
            (
                "amended",
                raw,
                active,
                [dict(link, predecessor_index=0, successor_index=1)],
            ),
        ]:
            after = rows >= effect
            mask[rows[after]] = source["valid"][after][:, names]
            source_age[rows[after]] = source["source_age_sessions"][after][:, names]
            result = np.empty((len(selected), 2, 20), dtype=np.float32)
            observation_age_sessions_into(
                mask,
                membership[: len(mask), names],
                result,
                source_rows=global_rows,
                source_age_sessions=source_age,
                history_links=use_links,
            )
            if label == "control":
                np.testing.assert_array_equal(
                    np.where(old_active[global_rows][:, names, None], result, -1),
                    old_age[global_rows][:, names],
                )
                control_age = result.copy()
                control_cells += result.size
            else:
                age[:, names] = result
        # Retired non-native sibling axes likewise cease to expose active ages.
        age[~active[global_rows]] = -1
        # The finite repair must rejoin the original source clock before the
        # bounded raw window ends, or its later age tail still needs assembly.
        if rows[-1] < len(dates) - 1:
            np.testing.assert_array_equal(age[-1, n], control_age[-1, 1])
        output["intraday_age_sessions"] = age
        counts = {}
        for key, value in output.items():
            old = np.load(store / (key + ".npy"), mmap_mode="r")[global_rows]
            changed = np.where(~equal(value, old))
            ix = np.column_stack((global_rows[changed[0]], *changed[1:]))
            deltas[f"{rec['isin']}__{key}__indices"] = ix
            deltas[f"{rec['isin']}__{key}__values"] = value[changed]
            counts[key] = len(ix)
        # Deterministic cross-sections only use each dated snapshot. Deleting
        # later raw snapshots must reproduce values, masks and valid labels.
        end = int(np.searchsorted(rows, effect + 8))
        prefix_values = np.empty((end, len(isins), 20), dtype=np.float32)
        prefix_valid = np.empty_like(prefix_values, dtype=bool)
        transform_feature_panel_into(
            raw["values"][:end],
            raw["valid"][:end],
            active[rows[:end]],
            specs,
            prefix_values,
            prefix_valid,
        )
        np.testing.assert_array_equal(
            prefix_values[selected[selected < end]], values[: sum(selected < end)]
        )
        np.testing.assert_array_equal(
            prefix_valid[selected[selected < end]], valid[: sum(selected < end)]
        )
        pt = targets(
            {key: value[:end] for key, value in raw.items()}, active[rows[:end]]
        )
        for key, value in pt.items():
            np.testing.assert_array_equal(
                value[selected[selected < end]], output[key][: sum(selected < end)]
            )
        np.savez_compressed(
            out / (rec["isin"] + "_output.npz"), date_indices=global_rows, **output
        )
        reports.append(
            {
                "isin": rec["isin"],
                "control_cells": control_cells,
                "changed_cells": counts,
                "valid_gains": int((valid & ~old_valid[global_rows]).sum()),
                "valid_losses": int((~valid & old_valid[global_rows]).sum()),
                "shared_valid_values_changed": int(
                    (
                        (values != old_values[global_rows])
                        & valid
                        & old_valid[global_rows]
                    ).sum()
                ),
                "target_gains": int(
                    (
                        output["target_to_close_valid"]
                        & ~np.load(store / "target_to_close_valid.npy", mmap_mode="r")[
                            global_rows
                        ]
                    ).sum()
                ),
                "future_deletion_prefix_exact": True,
            }
        )
        write_json_atomic(prior_report, reports[-1])
        print(json.dumps(reports[-1]), flush=True)
    np.savez_compressed(out / "deltas.npz", **deltas)
    write_json_atomic(
        out / "manifest.json",
        {
            "schema": "M1_SCALAR_ASSEMBLY_V1",
            "parent": m1["parent"],
            "raw_reconstruction": binding(root / "reconstruction.json"),
            "control_qualifications": [
                binding(root / "qualified_control/reconstruction.json"),
                binding(root / "qualified_control/reconstruction_2023-10-25.json"),
            ],
            "native_admission": run["rename_m1_propagation"],
            "history_admission": run["rename_history_propagation"],
            "deltas": binding(out / "deltas.npz"),
            "reports": reports,
            "seconds": time.perf_counter() - started,
            "reproducer": binding(
                out
                / (
                    "qualified_reproducer.py"
                    if args.resume
                    else "executed_reproducer.py"
                )
            ),
            "original_executed_reproducer": binding(out / "executed_reproducer.py"),
            "aeri_boundary_qualification": binding(
                out / "aeri_boundary_qualification.json"
            ),
            "status": "Rename scalar/age and original-convention to-close target amendment; final derived store remains separate.",
            "remaining_target_clock": "Original to-close uses fixed 405/345 minute normalization despite dated sessions. Not changed under rename-only attribution; audit separately before optional head admission.",
            "forecast_scoring": False,
            "heldout_access": False,
        },
    )


if __name__ == "__main__":
    main()
