"""Accept bounded daily-risk amendments and test their actual CPU consumer."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.build_store import _common_state_diagnostic_panel
from brazil_rv.v2.contract import HORIZONS, TARGET_NEUTRALIZATION_FEATURES
from brazil_rv.v2.corporate_actions import (
    align_verified_action_terms,
    verified_action_terms_from_table,
)
from brazil_rv.v2.data import V2DailyDataset, collate_v2_daily
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.feature_spec import FeatureSpec, feature_schema_sha256
from brazil_rv.v2.store import StoreStaging
from brazil_rv.v2.targets import build_economic_multi_day_targets

from audit_corporate_target_inputs import neutral_oracle
from propagate_corporate_targets import FIELDS

PROJECT = Path(__file__).resolve().parents[1]


def same(a, b):
    return (a == b) | (np.isnan(a) & np.isnan(b))


def main():
    started = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    produced = bound_json(run["natura_daily_propagation"])
    initial = bound_json(run["natura_daily_qualification"])
    plan = bound_json(produced["plan"])
    source = Path(produced["plan"]["path"]).parent
    out = source / "qualified"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    failed = [r for r in initial["checks"] if r["mismatches"]]
    assert len(failed) == 1 and failed[0]["name"] == "sigma_control"
    root = Path(plan["parent_store"]["root"])
    manifest = bound_json(
        {
            "path": str(root / "manifest.json"),
            "sha256": plan["parent_store"]["manifest_sha256"],
        }
    )
    dates, isins = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    local_dates = np.load(source / "date_index.npy")
    selected = np.load(source / "selected.npy")
    start, first, stop = plan["rows"]
    rows = start + selected
    axis = isins.index("BRNATUACNOR6")

    def old(key):
        return np.load(root / manifest["arrays"][key]["path"], mmap_mode="r")

    with np.load(initial["candidate_deltas"]["path"]) as z:
        deltas = {k: z[k].copy() for k in z.files}

    def add(key, before, after, decision_rows):
        ix = np.argwhere(~same(before, after))
        indices = ix.copy()
        indices[:, 0] = decision_rows[ix[:, 0]]
        deltas[key + "__indices"] = indices
        deltas[key + "__values"] = after[tuple(ix.T)]

    # Sigma is an own-name reducer. Preserve every other accepted raw-risk cell;
    # a bounded scratch reconstruction is not authority to amend those names.
    sigmas = []
    for label in ("control", "corrected"):
        with np.load(source / label / "8.npz") as z:
            sigmas.append(
                np.where(
                    z["valid"][selected - 1], z["values"][selected - 1], np.nan
                ).astype(np.float32)
            )
    before_sigma = old("target_scale_sigma")[rows]
    off = np.argwhere(~same(sigmas[0], before_sigma))
    assert len(off) == failed[0]["mismatches"]
    assert not old("active")[rows[off[:, 0]], off[:, 1]].any()
    assert not (off[:, 1] == axis).any()
    np.testing.assert_array_equal(sigmas[0][:, axis], before_sigma[:, axis])
    mask = np.ones(sigmas[0].shape[1], bool)
    mask[axis] = False
    np.testing.assert_array_equal(sigmas[1][:, mask], sigmas[0][:, mask])
    sigma = np.array(old("target_scale_sigma"), copy=True)
    sigma[rows, axis] = sigmas[1][:, axis]
    assert (deltas["target_scale_sigma__indices"][:, 1] == axis).all()
    preserved = [
        {
            "date": str(dates[rows[t]]),
            "isin": isins[n],
            "reconstructed": float(sigmas[0][t, n]),
            "accepted": None,
            "active": False,
            "source_change": False,
        }
        for t, n in off
    ]
    write_json_atomic(
        out / "preserved_offscope_sigma.json",
        {
            "rows": preserved,
            "disposition": "All 18 unrelated inactive raw-risk cells retain accepted NaN; both scratch scenarios agree there. No source defect or repair is inferred. All Natura and all eligible risk controls match.",
        },
    )

    history = bound_json(run["rename_history_propagation"])
    close = np.load(
        history["history_basis"]["shareholder_wealth_close"]["path"], mmap_mode="r"
    )[start:stop].copy()
    seen = np.load(
        history["history_basis"]["shareholder_wealth_valid"]["path"], mmap_mode="r"
    )[start:stop]
    ambiguous = np.load(source / "ambiguous.npy")
    active = old("active")[start:stop]
    control_common = _common_state_diagnostic_panel(
        close,
        seen,
        ambiguous,
        active,
        old("target_scale_sigma")[start:stop],
        selected,
        local_dates,
        minimum_names=20,
    )
    for j, suffix in enumerate(("values", "valid")):
        np.testing.assert_array_equal(
            control_common[j], old("common_state_diagnostic_" + suffix)[rows]
        )
    wealth = bound_json(run["natura_wealth_amendment"])
    with np.load(wealth["deltas"]["path"]) as z:
        ix = z["shareholder_wealth_close__indices"].copy()
        ix[:, 0] -= start
        close[tuple(ix.T)] = z["shareholder_wealth_close__values"]
    corrected_common = _common_state_diagnostic_panel(
        close,
        seen,
        ambiguous,
        active,
        sigma[start:stop],
        selected,
        local_dates,
        minimum_names=20,
    )
    for j, suffix in enumerate(("values", "valid")):
        add(
            "common_state_diagnostic_" + suffix,
            control_common[j],
            corrected_common[j],
            rows,
        )
    corrected_common[2].write_parquet(out / "common_state_rows.parquet")

    gross = bound_json(run["natura_gross_target_attribution"])
    gross_rows = np.load(gross["rows"]["path"])
    risk_rows = np.unique(deltas["target_scale_sigma__indices"][:, 0])
    target_rows = np.union1d(gross_rows, risk_rows)
    terms = verified_action_terms_from_table(
        pl.read_parquet(gross["replacement_terms"]["path"])
    )
    actions = align_verified_action_terms(
        terms, dates, isins, coverage_resolved=old("action_session_resolved")
    )
    target_close = np.round(np.asarray(old("raw_close"), np.float64), 2)
    args = (target_close, old("observed"), old("active"))
    # Reuse the already accepted term-only endpoint proof. New controls cover
    # only dates outside that proof, where old terms equal the corrected terms.
    fresh = np.setdiff1d(target_rows, gross_rows)
    controls = build_economic_multi_day_targets(
        *args, old("target_scale_sigma"), actions, source_rows=fresh
    )
    control_cells = 0
    for field, key in FIELDS.items():
        np.testing.assert_array_equal(getattr(controls, field), old(key)[fresh])
        control_cells += getattr(controls, field).size
    corrected = build_economic_multi_day_targets(
        *args, sigma, actions, source_rows=target_rows
    )
    target_effects = {}
    for field, key in FIELDS.items():
        after = getattr(corrected, field)
        before = old(key)[target_rows]
        add(key, before, after, target_rows)
        target_effects[key] = len(deltas[key + "__values"])
        if after.dtype.kind == "b":
            np.testing.assert_array_equal(before, after)
    np.savez_compressed(out / "deltas.npz", **deltas)
    write_json_atomic(
        out / "composition.json",
        {
            "daily_control": run["natura_daily_qualification"],
            "gross_control": run["natura_gross_target_attribution"],
            "preserved_risk": binding(out / "preserved_offscope_sigma.json"),
            "new_target_control_cells": control_cells,
            "new_target_control_dates": len(fresh),
            "target_dates": len(target_rows),
            "target_effects": target_effects,
            "deltas": binding(out / "deltas.npz"),
            "seconds": perf_counter() - started,
        },
    )

    # A purpose-limited view has only the daily/risk/target contract qualified
    # here. It cannot be used as a complete new neural store or with old weights.
    view_start = int(target_rows[0]) - 59
    view_stop = stop + max(HORIZONS)
    window = slice(view_start, view_stop)
    keys = [
        "active",
        "slow_values",
        "slow_valid",
        "slow_age_sessions",
        "slow_timestep_valid",
        "monthly_cluster_labels",
        "target_scale_sigma",
        *FIELDS.values(),
    ]
    arrays = {}
    for key in keys:
        a = np.array(old(key)[window], copy=True)
        ix = deltas.get(key + "__indices")
        if ix is not None:
            ix = ix.copy()
            ix[:, 0] -= view_start
            a[tuple(ix.T)] = deltas[key + "__values"]
        arrays[key] = a
    specs = [
        s
        for s in manifest["metadata"]["feature_schema"]["specifications"]
        if s["family"] == "slow"
    ]
    view = out / "daily_audit_view"
    with StoreStaging(view, dates=dates[window], isins=isins) as staging:
        for key, a in arrays.items():
            staging.write_array(key, a)
        staging.seal(
            feature_names={"slow": manifest["feature_names"]["slow"]},
            sources=[binding(out / "composition.json")],
            metadata={
                "purpose": "Natura bounded daily and primary target CPU audit only; native/auxiliary dependencies pending; no scoring",
                "parent": plan["parent_store"],
                "feature_schema": {
                    "minimum_rank_names": 20,
                    "specifications": specs,
                    "sha256": feature_schema_sha256([FeatureSpec(**s) for s in specs]),
                },
            },
        )
    samples = np.arange(int(target_rows[0]), stop)
    local_samples = samples - view_start
    granted = np.arange(local_samples[0], view_stop - view_start)
    dataset = V2DailyDataset(
        view,
        local_samples.tolist(),
        stage="finetune",
        lookback=60,
        include_fast=False,
        include_intraday=False,
        target_window_indices=granted.tolist(),
    )
    risk = [
        manifest["feature_names"]["slow"].index(k)
        for k in TARGET_NEUTRALIZATION_FEATURES
    ]
    cells = 0
    neutral_changes = neutral_gains = neutral_losses = 0
    for i, t in enumerate(local_samples):
        sample = dataset[i]
        batch = collate_v2_daily([sample])
        w = slice(t - 59, t + 1)
        valid = arrays["slow_valid"][w].transpose(1, 0, 2)
        hist = arrays["slow_timestep_valid"][w].T
        primary, pmask = neutral_oracle(
            arrays["target_shareholder_simple_return"][t],
            arrays["target_valid"][t],
            arrays["target_scale_sigma"][t],
            arrays["slow_values"][t][:, risk],
            arrays["slow_valid"][t][:, risk].all(axis=1),
        )
        original_t = int(t) + view_start
        old_primary, old_mask = neutral_oracle(
            old("target_shareholder_simple_return")[original_t],
            old("target_valid")[original_t],
            old("target_scale_sigma")[original_t],
            old("slow_values")[original_t][:, risk],
            old("slow_valid")[original_t][:, risk].all(axis=1),
        )
        neutral_changes += int((~same(primary, old_primary)).sum())
        neutral_gains += int((pmask & ~old_mask).sum())
        neutral_losses += int((~pmask & old_mask).sum())
        expected = {
            "active_mask": arrays["active"][t],
            "slow_features": np.where(
                valid, arrays["slow_values"][w].transpose(1, 0, 2), 0
            ),
            "slow_feature_mask": valid,
            "slow_feature_age_sessions": np.where(
                hist[..., None], arrays["slow_age_sessions"][w].transpose(1, 0, 2), -1
            ),
            "slow_history_mask": hist,
            "targets": np.where(pmask, primary, 0),
            "target_mask": pmask,
        }
        for key, value in expected.items():
            np.testing.assert_array_equal(
                sample[key], value, err_msg=f"{dates[original_t]}:{key}"
            )
            np.testing.assert_array_equal(batch[key][0].numpy(), value, err_msg=key)
            cells += value.size * 2
    dataset.store.close()
    report = {
        "status": "qualified_daily_risk_and_primary_targets_intermediate_not_complete_store",
        "parent_store": plan["parent_store"],
        "composition": binding(out / "composition.json"),
        "deltas": binding(out / "deltas.npz"),
        "consumer_view": binding(view / "manifest.json"),
        "samples": len(samples),
        "names": len(isins),
        "history": 60,
        "packed_cells": cells,
        "mismatches": 0,
        "neutral": {
            "changes": neutral_changes,
            "gains": neutral_gains,
            "losses": neutral_losses,
        },
        "remaining": "Native/scalar M1, auxiliary/valuation/unit dependencies and combined final tensors require separate completion. No old fit or accepted store was changed.",
        "seconds": perf_counter() - started,
    }
    write_json_atomic(out / "manifest.json", report)
    run["natura_daily_input_audit"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
