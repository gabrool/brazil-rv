"""Finish inherited daily-scale propagation and audit actual packed M1 inputs."""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.build_store import stream_intraday_from_assignments
from brazil_rv.v2.data import V2DailyDataset, collate_v2_daily
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.decision_clock import load_session_schedule
from brazil_rv.v2.feature_spec import FeatureSpec, feature_schema_sha256
from brazil_rv.v2.round5_derived import verified
from brazil_rv.v2.store import StoreStaging

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-produced", action="store_true")
    reuse = parser.parse_args().reuse_produced
    started = time.perf_counter()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    root = Path(run["root"]) / "rename_m1_propagation"
    initial = bound_json(binding(root / "manifest.json"))
    parent = initial["parent"]
    store = Path(parent["root"])
    original = bound_json(
        {"path": str(store / "manifest.json"), "sha256": parent["manifest_sha256"]}
    )
    history = bound_json(run["rename_history_propagation"])
    peers = bound_json(run["rename_peer_propagation"])
    dates, isins = np.load(store / "date_index.npy"), np.load(store / "isin_index.npy")
    links = pl.read_parquet(verified(history["history_mapping"]))
    out = root / "qualification"
    out.mkdir(exist_ok=reuse)
    shutil.copyfile(
        __file__,
        out / ("qualified_reproducer.py" if reuse else "executed_reproducer.py"),
    )
    sigma = np.load(verified(history["arrays"]["target_scale_sigma"]), mmap_mode="r")
    old_sigma = np.load(store / "target_scale_sigma.npy", mmap_mode="r")
    tail = {}
    tail_effects = {}
    # The remaining ALOS scale differences are small Float32-path effects of
    # the already admitted daily repair. Still propagate them exactly; don't
    # silently switch back to old daily coordinates after the M1 warmup.
    tail_path = out / "native_scale_tail_deltas.npz"
    if not reuse:
        for r in initial["results"]:
            link = links.filter(
                pl.col("successor_index") == int(np.flatnonzero(isins == r["isin"])[0])
            ).to_dicts()[0]
            end = int(np.searchsorted(dates, np.datetime64(r["read_end"]))) + 1
            if end == len(dates):
                continue
            n = link["successor_index"]
            start = end - 21
            window = np.arange(start, len(dates))
            sessions_by_date = {
                s.trade_date: s
                for s in load_session_schedule(Path(initial["schedule"]["path"]))
            }
            sessions = tuple(sessions_by_date[d] for d in dates[window].astype(object))
            assignments = pl.read_parquet(verified(initial["assignments"])).filter(
                pl.col("isin") == r["isin"]
            )
            quote_sources = [
                s
                for s in original["sources"]
                if Path(s["path"]).name
                in {
                    f"equities_daily_{d.year}.parquet"
                    for d in dates[window].astype(object)
                }
            ]
            daily = pl.concat(
                [
                    pl.scan_parquet(s["path"])
                    .filter(
                        pl.col("isin") == r["isin"],
                        pl.col("trade_date").is_between(
                            dates[start].astype(object), dates[-1].astype(object)
                        ),
                    )
                    .select("isin", "trade_date")
                    .collect()
                    for s in quote_sources
                ]
            )
            results = []
            for label, scale in (("control", old_sigma), ("continued", sigma)):
                workspace = out / (r["isin"] + "_" + label)
                workspace.mkdir()
                results.append(
                    stream_intraday_from_assignments(
                        assignments,
                        daily,
                        sessions,
                        [r["isin"]],
                        sigma_asof=scale[window, n, None],
                        kept_rows=np.arange(len(window)),
                        workspace=workspace,
                    )
                )
            mapping = pl.read_parquet(
                store / original["tables"]["native_fast_security_mapping"]["path"]
            )
            f = int(mapping.filter(pl.col("store_name_index") == n)[0, "fast_index"])
            for k, actual in results[1].native_arrays.items():
                control = results[0].native_arrays[k]
                old = np.load(store / f"{k}.npy", mmap_mode="r")
                np.testing.assert_array_equal(
                    control[21:, 0], old[window[21:], f, : actual.shape[2]]
                )
                changed = np.where(actual[21:, 0] != control[21:, 0])
                tail[f"{r['isin']}__{k}__indices"] = np.column_stack(
                    (window[21:][changed[0]], np.full(len(changed[0]), f), *changed[1:])
                )
                tail[f"{r['isin']}__{k}__values"] = actual[21:, 0][changed]
                tail_effects[k] = len(changed[0])
        np.savez_compressed(tail_path, **tail)
    else:
        with np.load(tail_path) as saved:
            tail_effects = {
                k.split("__")[1]: len(saved[k])
                for k in saved.files
                if k.endswith("__values")
            }
    sparse = [np.load(verified(initial["native_deltas"])), np.load(tail_path)]
    amendments = {}
    for delta in sparse:
        for key in delta.files:
            if key.endswith("__indices"):
                name = key.split("__")[1]
                amendments.setdefault(name, []).append(
                    (delta[key], delta[key.replace("__indices", "__values")])
                )
    specs = [
        s
        for s in original["metadata"]["feature_schema"]["specifications"]
        if s["family"] in {"slow", "native_fast"}
    ]
    mapping = pl.read_parquet(
        store / original["tables"]["native_fast_security_mapping"]["path"]
    )
    old_active = np.load(store / "active.npy", mmap_mode="r")
    active = np.load(verified(history["arrays"]["active"]), mmap_mode="r")
    checks, samples, scopes = 0, 0, []
    for link in links.to_dicts():
        n, effect = link["successor_index"], link["effective_index"]
        gained = np.flatnonzero(active[:, n] & ~old_active[:, n])
        if not len(gained):
            continue
        # Every restored eligible date, plus the complete post-warmup scale tail
        # and a pre-event control. Each sample retains all933 names and60 days.
        requested = np.unique(
            np.r_[
                effect - 1,
                gained,
                *[
                    ix[:, 0]
                    for entries in amendments.values()
                    for ix, _ in entries
                    if len(ix)
                    and np.any(
                        ix[:, 1]
                        == int(
                            mapping.filter(pl.col("store_name_index") == n)[
                                0, "fast_index"
                            ]
                        )
                    )
                ],
            ]
        )
        requested = requested.astype(int)
        start, stop = max(0, int(requested.min()) - 59), int(requested.max()) + 1
        window = np.arange(start, stop)
        expected_fast = {}
        view = out / (str(isins[n]) + "_input_store")
        for k in (*amendments, "fast_present"):
            array = np.array(
                np.load(store / f"{k}.npy", mmap_mode="r")[window], copy=True
            )
            for ix, values in amendments.get(k, []):
                selected = (ix[:, 0] >= start) & (ix[:, 0] < stop)
                index = ix[selected].copy()
                index[:, 0] -= start
                array[tuple(index.T)] = values[selected]
            expected_fast[k] = array
        if not view.exists():
            with StoreStaging(
                view, dates=dates[window], isins=isins.tolist()
            ) as staging:
                for k in (
                    "active",
                    "slow_values",
                    "slow_valid",
                    "slow_age_sessions",
                    "slow_timestep_valid",
                ):
                    record = peers.get("arrays", {}).get(k, history["arrays"].get(k))
                    staging.write_array(
                        k, np.load(verified(record), mmap_mode="r")[window]
                    )
                for k, array in expected_fast.items():
                    staging.write_array(k, array)
                local_links = links.with_columns(
                    (pl.col("effective_index") - start), (pl.col("known_index") - start)
                )
                staging.seal(
                    feature_names={
                        k: original["feature_names"][k] for k in ("slow", "native_fast")
                    },
                    sources=[binding(root / "manifest.json"), binding(tail_path)],
                    tables={
                        "native_fast_security_mapping": mapping,
                        "slow_history_links": local_links,
                    },
                    metadata={
                        "purpose": "Bounded full-population 60-session input audit only; no forecast scoring",
                        "parent": parent,
                        "feature_schema": {
                            "minimum_rank_names": 20,
                            "specifications": specs,
                            "sha256": feature_schema_sha256(
                                [FeatureSpec(**s) for s in specs]
                            ),
                        },
                    },
                )
        dataset = V2DailyDataset(
            view,
            (requested - start).tolist(),
            stage="finetune",
            lookback=60,
            include_fast=True,
            include_intraday=False,
        )
        permanent = mapping["store_name_index"].to_numpy()
        for i, t in enumerate(requested - start):
            sample = dataset[i]
            batch = collate_v2_daily([sample])
            values, valid, patch, names, positions, present = dataset._fast(int(t))
            selected = expected_fast["fast_present"][t, permanent] & expected_fast[
                "fast_patch_mask"
            ][t].any(axis=-1)
            np.testing.assert_array_equal(names, permanent[selected])
            np.testing.assert_array_equal(
                valid, expected_fast["fast_patch_valid"][t, selected]
            )
            np.testing.assert_array_equal(
                values,
                np.where(valid, expected_fast["fast_patch_values"][t, selected], 0),
            )
            np.testing.assert_array_equal(
                patch, expected_fast["fast_patch_mask"][t, selected]
            )
            np.testing.assert_array_equal(positions, patch.sum(axis=1))
            # The ordinary collator must retain values, independent validity and
            # permanent compact-name indices, not just direct _fast readout.
            for key in (
                "fast_patch_values",
                "fast_patch_valid",
                "fast_patch_mask",
                "fast_name_index",
            ):
                np.testing.assert_array_equal(batch[key][0].numpy(), sample[key])
                checks += np.asarray(sample[key]).size
            assert sample["slow_features"].shape[:2] == (933, 60)
            samples += 1
        dataset.store.close()
        scopes.append(
            {"isin": str(isins[n]), "samples": len(requested), "store": str(view)}
        )
    write_json_atomic(
        out / "manifest.json",
        {
            "parent": binding(root / "manifest.json"),
            "tail_deltas": binding(tail_path),
            "tail_changed_cells": tail_effects,
            "actual_packed_cells": checks,
            "samples": samples,
            "all_names": 933,
            "history_sessions": 60,
            "mismatches": 0,
            "scopes": scopes,
            "seconds": time.perf_counter() - started,
            "reproducer": binding(
                out / ("qualified_reproducer.py" if reuse else "executed_reproducer.py")
            ),
            "remaining": "Raw scalar amendments still require the final full-population transform/age and to-close target assembly; these stores are audit views, never old-fit inputs.",
        },
    )
    print(
        json.dumps({"samples": samples, "cells": checks, "tail": tail_effects}),
        flush=True,
    )


if __name__ == "__main__":
    main()
