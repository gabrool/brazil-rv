"""Qualify auxiliary rename effects and full-population CPU consumer routing."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data import V2DailyDataset, collate_v2_daily
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.feature_spec import (
    FeatureSpec,
    feature_schema_sha256,
    transform_feature_panel_into,
)
from brazil_rv.v2.round5_derived import verified
from brazil_rv.v2.round5_store import align_family
from brazil_rv.v2.store import StoreStaging
from audit_auxiliary_tensors import transformed

PROJECT = Path(__file__).resolve().parents[1]


def main():
    start = perf_counter()
    pointer_path = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer_path.read_text(encoding="utf8"))
    report = bound_json(run["remaining_auxiliaries"])
    root = Path(run["remaining_auxiliaries"]["path"]).parent
    output = root / "qualification"
    output.mkdir(exist_ok=False)
    (output / "executed_reproducer.py").write_bytes(Path(__file__).read_bytes())
    store = Path(report["parent"]["root"])
    manifest = bound_json(
        {
            "path": str(store / "manifest.json"),
            "sha256": report["parent"]["manifest_sha256"],
        }
    )
    rename, peers = (
        bound_json(run["rename_history_propagation"]),
        bound_json(run["rename_peer_propagation"]),
    )
    dates, isins = (
        np.load(store / "date_index.npy"),
        np.load(store / "isin_index.npy").tolist(),
    )
    days = dates.astype(object).tolist()
    active = np.load(verified(rename["arrays"]["active"]))
    old_active = np.load(store / "active.npy")
    links = pl.read_parquet(verified(rename["history_mapping"]))
    specs = manifest["metadata"]["feature_schema"]["specifications"]
    groups = ["microstructure", "options", "rebalance", "oddlot"]
    amendments, fields, oracle_cells = {}, {}, 0
    final_arrays = {}
    affected = np.zeros(active.shape, bool)
    for link in links.to_dicts():
        affected[
            max(link["effective_index"], link["known_index"]) :,
            [link["predecessor_index"], link["successor_index"]],
        ] = True

    def assemble(frame, family):
        names = tuple(manifest["feature_names"][family])
        raw, valid, age = align_family(frame, days, tuple(isins), names)
        value, known = np.zeros_like(raw), np.zeros_like(valid)
        f_specs = [FeatureSpec(**s) for s in specs if s["family"] == family]
        transform_feature_panel_into(raw, valid, active, f_specs, value, known)
        age = np.where(active[..., None], age, -1).astype(np.float32)
        # Independent typed formulas verify the actual transformed loader.
        for k, spec in enumerate(f_specs):
            expected, mask = transformed(
                raw[..., k], valid[..., k] & active, spec.__dict__
            )
            np.testing.assert_array_equal(value[..., k], expected)
            np.testing.assert_array_equal(known[..., k], mask)
        return value, known, age

    for group in groups:
        family = "sidecar_" + group
        suffixes = ["values", "valid", "age_sessions"]
        old = [np.load(store / f"{family}_{s}.npy", mmap_mode="r") for s in suffixes]
        new = assemble(pl.read_parquet(verified(report["artifacts"][group])), family)
        if group == "oddlot":
            # This source has a previously audited store boundary, not a full
            # decision-dated family parquet. Only the three successor tails
            # are replaced, along with the explicitly changed membership mask.
            scope = np.zeros(active.shape, bool)
            for link in links.to_dicts():
                scope[
                    max(link["effective_index"], link["known_index"]) :,
                    link["successor_index"],
                ] = True
            new = tuple(np.where(scope[..., None], x, a) for x, a in zip(new, old))
            new = (
                np.where(active[..., None], new[0], 0).astype(np.float32),
                new[1] & active[..., None],
                np.where(active[..., None], new[2], -1).astype(np.float32),
            )
            before = pl.concat(
                [
                    pl.read_parquet(
                        root / r["successor"] / "odd_control.parquet"
                    ).filter(
                        (pl.col("isin") == r["successor"])
                        & (
                            pl.col("date")
                            >= days[
                                max(
                                    r["link"]["effective_index"],
                                    r["link"]["known_index"],
                                )
                            ]
                        )
                    )
                    for r in report["scopes"]
                ]
            )
            eligibility = assemble(before, family)
        elif group == "rebalance":
            # The historical index producer emitted active rows only. Its
            # archived eligibility-only rebuild is the matched membership
            # control; the sparse parent family is not that control.
            eligibility = assemble(
                pl.read_parquet(root / "index_eligibility_only.parquet"), family
            )
        else:
            eligibility = assemble(
                pl.read_parquet(verified(report["parent_families"][group]["data"])),
                family,
            )
        names = manifest["feature_names"][family]
        fields[group] = []
        for f, name in enumerate(names):
            a, b = old[1][..., f], new[1][..., f]
            fields[group].append(
                {
                    "name": name,
                    "gained": int((~a & b).sum()),
                    "lost": int((a & ~b).sum()),
                    "lost_still_eligible": int((a & ~b & active).sum()),
                    "eligibility_only_gains": int((~a & eligibility[1][..., f]).sum()),
                    "incremental_history_gains": int(
                        (~eligibility[1][..., f] & b & affected).sum()
                    ),
                    "changed_shared": int(
                        (a & b & (old[0][..., f] != new[0][..., f])).sum()
                    ),
                }
            )
        for suffix, before, after in zip(suffixes, old, new):
            np.testing.assert_array_equal(before[~affected], after[~affected])
            ix = np.argwhere(before != after)
            key = family + "_" + suffix
            amendments[key + "__indices"] = ix.astype(np.int32)
            amendments[key + "__values"] = after[tuple(ix.T)]
            final_arrays[key] = after
            oracle_cells += after.size
        assert not any(r["lost_still_eligible"] for r in fields[group]), fields[group]
        print(json.dumps({"family": group, "fields": fields[group]}), flush=True)
    np.savez_compressed(output / "deltas.npz", **amendments)
    # Samples include each newly eligible date, both pre-effect controls, and
    # the first post-warmup/last date; all 933 permanent names remain present.
    samples = set(np.flatnonzero((active & ~old_active).any(axis=1)).tolist())
    samples.update([len(days) - 1])
    for link in links.to_dicts():
        samples.add(link["effective_index"] - 1)
        samples.add(min(len(days) - 1, link["effective_index"] + 22))
    slow = {
        k: np.load(
            verified(peers.get("arrays", {}).get(k, rename["arrays"].get(k))),
            mmap_mode="r",
        )
        for k in [
            "active",
            "slow_values",
            "slow_valid",
            "slow_age_sessions",
            "slow_timestep_valid",
        ]
    }
    views, cells = [], 0
    view_specs = [
        s for s in specs if s["family"] in ["slow"] + ["sidecar_" + k for k in groups]
    ]
    for t in sorted(samples):
        first = t - 59
        rows = np.arange(first, t + 1)
        view = output / str(dates[t])
        with StoreStaging(view, dates=dates[rows], isins=isins) as staging:
            for key, value in slow.items():
                staging.write_array(key, value[rows])
            for key, value in final_arrays.items():
                staging.write_array(key, value[rows])
            staging.seal(
                feature_names={
                    f: manifest["feature_names"][f]
                    for f in ["slow"] + ["sidecar_" + k for k in groups]
                },
                sources=[run["remaining_auxiliaries"]],
                tables={
                    "slow_history_links": links.with_columns(
                        pl.col("effective_index") - first, pl.col("known_index") - first
                    )
                },
                metadata={
                    "purpose": "Auxiliary CPU consumer audit only; not accepted model input",
                    "feature_schema": {
                        "minimum_rank_names": 20,
                        "specifications": view_specs,
                        "sha256": feature_schema_sha256(
                            [FeatureSpec(**s) for s in view_specs]
                        ),
                    },
                },
            )
        data = V2DailyDataset(
            view,
            [59],
            stage="finetune",
            lookback=60,
            include_fast=False,
            include_intraday=False,
            enabled_sidecars=groups,
        )
        sample = data[0]
        batch = collate_v2_daily([sample])
        assert sample["slow_features"].shape[:2] == (933, 60)
        for key, value in final_arrays.items():
            np.testing.assert_array_equal(sample[key], value[t])
            np.testing.assert_array_equal(batch[key][0].numpy(), value[t])
            cells += value[t].size * 2
        data.store.close()
        views.append(
            {
                "date": str(dates[t]),
                "global_index": t,
                "manifest": binding(view / "manifest.json"),
            }
        )
    receipt = {
        "status": "passed",
        "input": run["remaining_auxiliaries"],
        "fields": fields,
        "deltas": binding(output / "deltas.npz"),
        "outside_scope_exact": True,
        "typed_array_cells": oracle_cells,
        "consumer_samples": len(samples),
        "names": len(isins),
        "history": 60,
        "consumer_cells": cells,
        "mismatches": 0,
        "views": views,
        "seconds": perf_counter() - start,
        "code": binding(Path(__file__)),
        "model_forward_or_heldout": False,
    }
    write_json_atomic(output / "manifest.json", receipt)
    run["remaining_auxiliary_input_audit"] = binding(output / "manifest.json")
    write_json_atomic(pointer_path, run)
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in ["status", "consumer_samples", "consumer_cells", "seconds"]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
