"""Rebuild only the five cluster fields after admitting rename history."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.feature_spec import (
    FeatureSpec,
    transform_feature_panel_into,
    observation_age_sessions_into,
)
from brazil_rv.v2.features import (
    exact_log_return,
    monthly_cluster_labels,
    _peer_features,
)

PROJECT = Path(__file__).resolve().parents[1]


def main():
    started = perf_counter()
    pointer = json.loads(
        (PROJECT / "docs/v2_economic_data_scaling_run.json").read_text(encoding="utf8")
    )
    root = Path(pointer["root"]) / "rename_history_propagation"
    parent = bound_json(binding(root / "manifest.json"))
    source = Path(parent["parent"]["root"])
    original = json.loads((source / "manifest.json").read_text(encoding="utf8"))
    output = root.parent / "rename_peer_history"
    output.mkdir(exist_ok=False)
    (output / "executed_reproducer.py").write_bytes(Path(__file__).read_bytes())
    dates = np.load(root / "date_index.npy")
    links = pl.read_parquet(root / "slow_history_links.parquet").to_dicts()
    start = int(np.searchsorted(dates, np.datetime64("2023-01-01")))
    first = min(r["effective_index"] for r in links)
    # This changed window has over 126 sessions before its first affected
    # monthly refit; no earlier history, clustering or slow field is rerun.
    days = dates[start:]
    active = np.load(root / "active.npy", mmap_mode="r")[start:]
    wealth = np.load(
        root / "continuation_history_basis/shareholder_wealth_close.npy", mmap_mode="r"
    )[start:]
    valid = np.load(
        root / "continuation_history_basis/shareholder_wealth_valid.npy", mmap_mode="r"
    )[start:]
    returns = {
        h: exact_log_return(wealth, h, shareholder_wealth_valid=valid)
        for h in (1, 5, 21)
    }
    specs = [
        FeatureSpec(**s)
        for s in original["metadata"]["feature_schema"]["specifications"]
        if s["family"] == "slow"
    ]
    count = len(days)
    selected = np.arange(first - start, count)
    # Reconstruct the consumed return fields independently from the persisted
    # wealth chain. Prove exact masks/values before using them for peer history;
    # this protects against an omitted ambiguity/restart boundary.
    for horizon, feature in ((1, 0), (5, 1), (21, 2)):
        values = np.empty((len(selected), active.shape[1], 1), np.float32)
        masks = np.empty(values.shape, bool)
        raw, raw_valid = returns[horizon]
        transform_feature_panel_into(
            raw[..., None],
            raw_valid[..., None],
            active,
            specs[feature : feature + 1],
            values,
            masks,
            source_rows=selected - 1,
            membership_rows=selected,
        )
        np.testing.assert_array_equal(
            values[..., 0],
            np.load(root / "slow_values.npy", mmap_mode="r")[first:, :, feature],
        )
        np.testing.assert_array_equal(
            masks[..., 0],
            np.load(root / "slow_valid.npy", mmap_mode="r")[first:, :, feature],
        )
    residual = returns[1][0].copy()
    residual_valid = returns[1][1] & active
    for day in range(count):
        mask = residual_valid[day]
        if mask.any():
            residual[day, mask] -= np.median(residual[day, mask])
    local_links = [
        {
            **r,
            "effective_index": r["effective_index"] - start,
            "known_index": r["known_index"] - start,
        }
        for r in links
    ]
    labels = monthly_cluster_labels(
        days, residual, residual_valid, active, history_links=local_links
    )
    raw, mask = _peer_features(*returns[5], *returns[21], labels, active)
    for row in local_links:
        t, a, b = (
            row["effective_index"] - 1,
            row["predecessor_index"],
            row["successor_index"],
        )
        raw[t, b], mask[t, b] = raw[t, a], mask[t, a]
    fields = {}
    for key in ("slow_values", "slow_valid", "slow_age_sessions"):
        fields[key] = np.lib.format.open_memmap(
            output / f"{key}.npy",
            mode="w+",
            dtype=np.load(root / f"{key}.npy", mmap_mode="r").dtype,
            shape=np.load(root / f"{key}.npy", mmap_mode="r").shape,
        )
        fields[key][:] = np.load(root / f"{key}.npy", mmap_mode="r")
    transform_feature_panel_into(
        raw,
        mask,
        active,
        specs[27:32],
        fields["slow_values"][first:, :, 27:32],
        fields["slow_valid"][first:, :, 27:32],
        source_rows=selected - 1,
        membership_rows=selected,
    )
    observation_age_sessions_into(
        mask,
        active,
        fields["slow_age_sessions"][first:, :, 27:32],
        source_rows=selected - 1,
        decision_rows=selected,
    )
    full_labels = np.load(root / "monthly_cluster_labels.npy")
    full_labels[first:] = labels[first - start :]
    np.save(output / "monthly_cluster_labels.npy", full_labels)
    for array in fields.values():
        array.flush()
    old_valid = np.load(root / "slow_valid.npy", mmap_mode="r")
    new_valid = fields["slow_valid"]
    old_base = np.load(source / "slow_valid.npy", mmap_mode="r")
    current_active = np.load(root / "active.npy", mmap_mode="r")
    outcomes = {
        "additional_valid": int((new_valid & ~old_valid).sum()),
        "removed_vs_prior_amendment": int((old_valid & ~new_valid).sum()),
        "lost_still_active_vs_sealed": int(
            (old_base & ~new_valid & current_active[..., None]).sum()
        ),
    }
    for key, array in fields.items():
        baseline = np.load(root / f"{key}.npy", mmap_mode="r")
        np.testing.assert_array_equal(array[:first], baseline[:first])
        np.testing.assert_array_equal(array[:, :, :27], baseline[:, :, :27])
    record = {
        "status": "passed_peer_history_overlay_new_store_pending",
        "parent": binding(root / "manifest.json"),
        "code": binding(output / "executed_reproducer.py"),
        "arrays": {p.stem: binding(p) for p in output.glob("*.npy")},
        "effects": outcomes,
        "return_fields_reproduced_exactly": [1, 5, 21],
        "other_27_fields_exact": True,
        "pre_event_prefix_exact": True,
        "seconds": perf_counter() - started,
        "source_clock": "source-bound effect/knowledge; monthly fit samples preceding 126 sessions with original minimum 101; carry current assignment on rename",
        "heldout_access": False,
        "forecast_scoring": False,
    }
    write_json_atomic(output / "manifest.json", record)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
