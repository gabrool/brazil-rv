"""Actual full-population consumer and capability audit of basket targets."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.contract import HORIZONS, TARGET_NEUTRALIZATION_FEATURES
from brazil_rv.v2.data import V2DailyDataset, collate_v2_daily
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.feature_spec import FeatureSpec, feature_schema_sha256
from brazil_rv.v2.store import StoreStaging

PROJECT = Path(__file__).resolve().parents[1]
PAIRS = [
    ("target_primary", "target_valid", "targets", "target_mask"),
    (
        "target_shareholder_midrank",
        "target_shareholder_valid",
        "shareholder_targets",
        "shareholder_target_mask",
    ),
    (
        "target_shareholder_simple_return",
        "target_shareholder_valid",
        "shareholder_simple_returns",
        "shareholder_target_mask",
    ),
    (
        "target_terminal_wealth",
        "target_shareholder_valid",
        "terminal_wealth",
        "shareholder_target_mask",
    ),
    (
        "target_terminal_loss",
        "target_shareholder_valid",
        "terminal_loss",
        "shareholder_target_mask",
    ),
    (
        "target_price_midrank",
        "target_price_valid",
        "price_targets",
        "price_target_mask",
    ),
    (
        "target_price_simple_return",
        "target_price_valid",
        "price_simple_returns",
        "price_target_mask",
    ),
]


def ranks(x):
    unique, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    del unique
    starts = np.cumsum(counts) - counts
    return (starts + (counts - 1) / 2)[inverse]


def neutral_oracle(returns, valid, sigma, z, zvalid):
    out = np.zeros(returns.shape, np.float32)
    mask = np.zeros(valid.shape, bool)
    for j, horizon in enumerate(HORIZONS):
        keep = (
            valid[:, j]
            & zvalid
            & np.isfinite(returns[:, j])
            & np.isfinite(sigma)
            & (sigma > 1e-8)
        )
        n = int(keep.sum())
        if n < 20:
            continue
        r = returns[keep, j].astype(np.float64)
        y = np.clip(
            (r - np.median(r)) / (sigma[keep].astype(np.float64) * np.sqrt(horizon)),
            -5,
            5,
        )
        chars = z[keep].astype(np.float64)
        if n < 40:
            design = np.column_stack([np.ones(n), chars])
        else:
            vg = np.minimum(np.floor(ranks(chars[:, 0]) * 10 / n), 9).astype(int)
            bg = np.minimum(np.floor(ranks(chars[:, 1]) * 5 / n), 4).astype(int)
            design = np.column_stack(
                [(vg == k).astype(float) for k in range(10)]
                + [(bg == k).astype(float) for k in range(5)]
                + [chars[:, 2]]
            )
        # Orthogonal projection from an SVD, independent of production lstsq.
        u, s, _ = np.linalg.svd(design, full_matrices=False)
        basis = u[:, s > np.finfo(float).eps * max(design.shape) * s[0]]
        residual = y - basis @ (basis.T @ y)
        unit = np.finfo(float).eps * max(1.0, float(np.max(np.abs(y))))
        if np.max(np.abs(residual)) <= 128 * unit:
            residual[:] = 0
        else:
            order = np.argsort(residual, kind="stable")
            start = 0
            for end in range(1, n + 1):
                if (
                    end == n
                    or residual[order[end]] - residual[order[start]] > 512 * unit
                ):
                    residual[order[start:end]] = residual[order[start]]
                    start = end
        out[keep, j] = (ranks(residual) / (n - 1)).astype(np.float32)
        mask[keep, j] = True
    return out, mask


def main():
    start = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    target = bound_json(run["corporate_target_propagation"])
    history, peers = [
        bound_json(run[k])
        for k in ["rename_history_propagation", "rename_peer_propagation"]
    ]
    store = Path(target["parent"]["root"])
    original = bound_json(
        {
            "path": str(store / "manifest.json"),
            "sha256": target["parent"]["manifest_sha256"],
        }
    )
    dates, isins = (
        np.load(store / "date_index.npy"),
        np.load(store / "isin_index.npy").tolist(),
    )
    requested = np.load(target["rows"]["path"])
    amendments = np.load(target["deltas"]["path"])
    out = (
        Path(run["corporate_target_propagation"]["path"]).parent
        / "consumer"
        / "qualified"
    )
    out.mkdir()
    (out / "executed_reproducer.py").write_bytes(Path(__file__).read_bytes())
    slow = {
        key: np.load(
            peers["arrays"].get(key, history["arrays"].get(key))["path"], mmap_mode="r"
        )
        for key in [
            "active",
            "slow_values",
            "slow_valid",
            "slow_age_sessions",
            "slow_timestep_valid",
        ]
    }
    sigma = np.load(history["arrays"]["target_scale_sigma"]["path"], mmap_mode="r")
    character_indices = [
        original["feature_names"]["slow"].index(k)
        for k in TARGET_NEUTRALIZATION_FEATURES
    ]
    arrays = {
        key: np.load(value["path"], mmap_mode="r")
        for key, value in history["arrays"].items()
        if key in target["target_arrays"]
    }
    specs = [
        s
        for s in original["metadata"]["feature_schema"]["specifications"]
        if s["family"] == "slow"
    ]
    links = pl.read_parquet(history["history_mapping"]["path"])
    checks, views = 0, []
    for day in requested:
        first, stop = int(day) - 59, min(int(day) + 11, len(dates))
        rows = np.arange(first, stop)
        view = out / str(dates[day])
        expected = {}
        with StoreStaging(view, dates=dates[rows], isins=isins) as staging:
            for key, value in slow.items():
                staging.write_array(key, value[rows])
            staging.write_array("target_scale_sigma", sigma[rows])
            for key, source in arrays.items():
                value = source[rows].copy()
                ix = amendments[key + "__indices"]
                keep = (ix[:, 0] >= first) & (ix[:, 0] < stop)
                index = ix[keep].copy()
                index[:, 0] -= first
                value[tuple(index.T)] = amendments[key + "__values"][keep]
                staging.write_array(key, value)
                expected[key] = value[59]
            staging.seal(
                feature_names={"slow": original["feature_names"]["slow"]},
                sources=[run["corporate_target_propagation"]],
                tables={
                    "slow_history_links": links.with_columns(
                        pl.col("effective_index") - first, pl.col("known_index") - first
                    )
                },
                metadata={
                    "purpose": "Corporate target CPU audit only, not accepted model store",
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
            [59],
            stage="finetune",
            lookback=60,
            include_fast=False,
            include_intraday=False,
            target_window_indices=np.arange(59, len(rows)),
        )
        sample = dataset[0]
        batch = collate_v2_daily([sample])
        assert sample["slow_features"].shape[:2] == (933, 60)
        chars = slow["slow_values"][day][:, character_indices]
        chars_valid = slow["slow_valid"][day][:, character_indices].all(
            axis=-1
        ) & np.isfinite(chars).all(axis=-1)
        primary, primary_mask = neutral_oracle(
            expected["target_shareholder_simple_return"],
            expected["target_valid"],
            sigma[day],
            chars,
            chars_valid,
        )
        for value_key, mask_key, dest, mask_dest in PAIRS:
            mask = (primary_mask if dest == "targets" else expected[mask_key]).copy()
            for hidx, horizon in enumerate(target["horizons"]):
                if 59 + horizon >= len(rows):
                    mask[:, hidx] = False
            value = np.where(
                mask, primary if dest == "targets" else expected[value_key], 0
            )
            for key, want in [(dest, value), (mask_dest, mask)]:
                np.testing.assert_array_equal(sample[key], want)
                np.testing.assert_array_equal(batch[key][0].numpy(), want)
                checks += want.size * 2
        dataset.store.close()
        # A revoked endpoint window must expose neither masks nor target payloads.
        restricted = V2DailyDataset(
            view,
            [59],
            stage="finetune",
            lookback=60,
            include_fast=False,
            include_intraday=False,
            target_window_indices=[59],
        )
        denied = restricted[0]
        for _, _, dest, mask_dest in PAIRS:
            assert not denied[mask_dest].any()
            assert not denied[dest].any()
        restricted.store.close()
        views.append(
            {"date": str(dates[day]), "manifest": binding(view / "manifest.json")}
        )
    receipt = {
        "status": "passed",
        "producer": run["corporate_target_propagation"],
        "samples": len(requested),
        "names": 933,
        "history": 60,
        "cells": checks,
        "mismatches": 0,
        "revoked_endpoint_windows": len(requested),
        "views": views,
        "seconds": perf_counter() - start,
        "model_forward_or_fit": False,
        "code": binding(Path(__file__)),
    }
    write_json_atomic(out / "manifest.json", receipt)
    run["corporate_target_input_audit"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps({k: receipt[k] for k in ["status", "samples", "cells", "seconds"]})
    )


if __name__ == "__main__":
    main()
