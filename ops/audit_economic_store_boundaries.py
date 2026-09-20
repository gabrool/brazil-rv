"""Qualify retained membership, support losses and bounded final consumers."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data import V2DailyDataset
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.store import close_memmap
from brazil_rv.v2.train import _verify_additive_parent_transfer

PROJECT = Path(__file__).resolve().parents[1]


def main():
    start = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    assembly = bound_json(run["derived_store_assembly"])
    root = Path(assembly["store"]["root"])
    parent = Path(assembly["parent"]["root"])
    manifest = json.loads((root / "manifest.json").read_text())
    dates = np.load(root / "date_index.npy")
    isins = np.load(root / "isin_index.npy")
    active = np.load(root / "active.npy")
    output = Path(run["root"]) / "derived_store_boundaries"
    output.mkdir(exist_ok=False)
    (output / "executed_reproducer.py").write_bytes(Path(__file__).read_bytes())
    denom = bound_json(run["lending_feature_propagation"])["artifacts"][
        "denominator_support_losses"
    ]
    prior_losses = set(pl.read_parquet(denom["path"])["date"])
    losses = []
    for key, effect in assembly["effects"].items():
        if not key.endswith("_valid") or effect.get("true_lost", 0) == 0:
            continue
        family = key.removesuffix("_valid")
        if family not in manifest["feature_names"]:
            continue
        before = np.load(parent / (key + ".npy"), mmap_mode="r")
        after = np.load(root / (key + ".npy"), mmap_mode="r")
        for t, n, f in np.argwhere(before & ~after & active[..., None]):
            field = manifest["feature_names"][family][f]
            if family == "sidecar_lending":
                assert isins[n] == "BRALOSACNOR5" and field == "utilization_proxy"
                assert dates[t].astype(object) in prior_losses
                reason = "previously enumerated capital-unit uncertainty"
            else:
                assert family == "slow" and isins[n] == "BREMBRACNOR4"
                assert "cluster" in field
                reason = "previously enumerated unchanged three-other-peer support"
            losses.append(
                {
                    "date": dates[t].astype(object),
                    "isin": str(isins[n]),
                    "family": family,
                    "field": field,
                    "reason": reason,
                }
            )
        close_memmap(before)
        close_memmap(after)
    assert len(losses) == 158
    pl.DataFrame(losses).write_parquet(output / "active_support_losses.parquet")
    links = pl.read_parquet(root / manifest["tables"]["slow_history_links"]["path"])
    groups = sorted(
        k.removeprefix("sidecar_")
        for k in manifest["feature_names"]
        if k.startswith("sidecar_")
    )
    checks = []
    # Delete future identity information in a private runtime view. Reject any
    # attempted access to future feature observations; the sealed files stay read-only.
    for date in ("2023-01-09", "2023-10-24", "2023-10-25", "2024-11-14", "2024-11-18"):
        t = int(np.searchsorted(dates, np.datetime64(date)))
        dataset = V2DailyDataset(
            root,
            [t],
            stage="finetune",
            enabled_sidecars=groups,
            target_window_indices=[t],
            lookback=60,
        )
        before = dataset[0]
        filtered = output / ("known_links_" + date + ".parquet")
        links.filter(
            (pl.col("known_index") <= t) & (pl.col("effective_index") <= t)
        ).write_parquet(filtered)
        dataset.store.manifest["tables"]["slow_history_links"]["path"] = str(filtered)
        original_read = dataset.store.read
        reads = []

        def bounded_read(name, date_selector):
            selected = np.atleast_1d(np.arange(len(dates))[date_selector])
            if not name.startswith("target_"):
                assert selected.max() <= t, (name, selected.max(), t)
                reads.append(name)
            return original_read(name, date_selector)

        dataset.store.read = bounded_read
        after = dataset[0]
        cells = 0
        for name, value in before.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(
                    value, after[name], err_msg=date + ":" + name
                )
                cells += value.size
        assert not after["target_mask"].any() and not after["targets"].any()
        checks.append(
            {
                "date": date,
                "cells": cells,
                "bounded_feature_reads": len(reads),
                "future_links_deleted": links.height - pl.read_parquet(filtered).height,
                "revoked_horizon_targets_zero": True,
            }
        )
        dataset.store.close()
    # The existing transfer admission checks reject even an explicitly requested
    # additive extension from the old store, before reading any old model weights.
    old_contract = {"training": {"store": assembly["parent"]}}
    new_contract = {"training": {"store": assembly["store"]}}
    try:
        _verify_additive_parent_transfer(old_contract, new_contract, (parent, root))
    except ValueError as error:
        assert str(error) == "parent transfer changed a protected array, table or axis"
        parent_rejection = str(error)
    else:
        raise AssertionError("Old parent incorrectly admitted")
    report = {
        "status": "passed",
        "assembly": run["derived_store_assembly"],
        "active_support_losses": binding(output / "active_support_losses.parquet"),
        "active_losses_vs_sealed": 158,
        "alos_utilization": 68,
        "embraer_peer_fields": 90,
        "difference_from_prior_126": "68 of the prior 126 ALOS losses were valid in the sealed store; the other dates gained identity/eligibility only in intermediate overlays.",
        "future_deletion_checks": checks,
        "old_parent_rejected": parent_rejection,
        "code": binding(output / "executed_reproducer.py"),
        "seconds": perf_counter() - start,
        "model_forward_or_heldout": False,
    }
    write_json_atomic(output / "manifest.json", report)
    run["derived_store_boundaries"] = binding(output / "manifest.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            {
                "support_losses": len(losses),
                "causal_checks": len(checks),
                "seconds": report["seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
