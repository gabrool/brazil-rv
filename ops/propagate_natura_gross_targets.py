"""Isolate two source-proven scalar terms with risks held at the accepted baseline."""

from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.contract import HORIZONS
from brazil_rv.v2.corporate_actions import (
    align_verified_action_terms,
    verified_action_terms_from_table,
    verified_action_terms_to_table,
)
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.targets import build_economic_multi_day_targets

from propagate_corporate_targets import FIELDS

PROJECT = Path(__file__).resolve().parents[1]


def main():
    started = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    source = json.loads((PROJECT / "docs/v2_economic_data_inputs.json").read_text())[
        "store"
    ]
    store = Path(source["root"])
    manifest = bound_json(
        {"path": str(store / "manifest.json"), "sha256": source["manifest_sha256"]}
    )
    qualification = bound_json(run["held_event_source_qualification"])
    amendments = verified_action_terms_from_table(
        pl.read_parquet(qualification["amendments"]["path"])
    )
    out = Path(run["root"]) / "natura_gross_targets" / "qualified_endpoints"
    out.mkdir(parents=True, exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    (out / "target_fields_recipe.py").write_bytes(
        (PROJECT / "ops/propagate_corporate_targets.py").read_bytes()
    )

    def old(key):
        return np.load(store / manifest["arrays"][key]["path"], mmap_mode="r")

    dates, isins = (
        np.load(store / "date_index.npy"),
        np.load(store / "isin_index.npy").tolist(),
    )
    assert len(isins) == 933 and dates[-1] == np.datetime64("2024-12-30")
    axis = isins.index(amendments[0].isin)
    assert all(t.isin == isins[axis] for t in amendments)
    event_days = [
        int(np.searchsorted(dates, np.datetime64(t.ex_date))) for t in amendments
    ]
    rows = np.array(
        sorted({r for day in event_days for r in range(day - max(HORIZONS), day + 1)})
    )
    original_path = (
        store / manifest["tables"]["corporate_actions_verified_terms"]["path"]
    )
    original = verified_action_terms_from_table(pl.read_parquet(original_path))
    keys = {(t.isin, t.ex_date) for t in amendments}
    replaced = tuple(t for t in original if (t.isin, t.ex_date) in keys)
    assert len(replaced) == 2 and all(
        t.source == "inferred_cotahist_dismes_v1" for t in replaced
    )
    terms = tuple(t for t in original if (t.isin, t.ex_date) not in keys) + amendments
    verified_action_terms_to_table(terms).write_parquet(
        out / "corporate_actions_verified_terms.parquet"
    )
    verified_action_terms_to_table(replaced).write_parquet(
        out / "superseded_terms.parquet"
    )
    controls = align_verified_action_terms(
        original, dates, isins, coverage_resolved=old("action_session_resolved")
    )
    actions = align_verified_action_terms(
        terms, dates, isins, coverage_resolved=old("action_session_resolved")
    )
    changed_q = np.argwhere(
        actions.shares_per_prior_share != controls.shares_per_prior_share
    )
    np.testing.assert_array_equal(changed_q, np.array([[r, axis] for r in event_days]))
    np.testing.assert_array_equal(actions.session_resolved, controls.session_resolved)
    np.testing.assert_array_equal(actions.successor_index, controls.successor_index)
    close = np.round(np.asarray(old("raw_close"), np.float64), 2)
    observed, active, sigma = [
        old(k) for k in ("observed", "active", "target_scale_sigma")
    ]
    args = (close, observed, active, sigma)
    before = build_economic_multi_day_targets(*args, controls, source_rows=rows)
    control_cells = 0
    for field, key in FIELDS.items():
        np.testing.assert_array_equal(getattr(before, field), old(key)[rows])
        control_cells += getattr(before, field).size
    after = build_economic_multi_day_targets(*args, actions, source_rows=rows)
    oracles = []
    for p, day in enumerate(rows):
        for k, horizon in enumerate(HORIZONS):
            end = day + horizon
            if not any(day < event <= end for event in event_days):
                continue
            if not after.shareholder_valid[p, axis, k]:
                continue
            assert active[day, axis] and observed[day, axis] and observed[end, axis]
            qty, cash = Decimal(1), Decimal(0)
            for s in range(day + 1, end + 1):
                cash += qty * Decimal(str(actions.cash_per_prior_share[s, axis]))
                qty *= Decimal(str(actions.shares_per_prior_share[s, axis]))
            value = qty * Decimal(str(close[end, axis])) + cash
            wealth = value / Decimal(str(close[day, axis]))
            expected = np.float32(float(wealth))
            np.testing.assert_array_equal(after.terminal_wealth[p, axis, k], expected)
            np.testing.assert_array_equal(
                after.shareholder_simple_return[p, axis, k],
                np.float32(float(wealth - 1)),
            )
            oracles.append(
                {
                    "entry": str(dates[day]),
                    "exit": str(dates[end]),
                    "horizon": horizon,
                    "quantity": str(qty),
                    "gross_cash": str(cash),
                    "entry_price": float(close[day, axis]),
                    "exit_price": float(close[end, axis]),
                    "gross_terminal_wealth": str(wealth),
                }
            )
    # A term removed before effect cannot alter any exact endpoint that precedes it.
    prefix_checks = 0
    for term, event in zip(amendments, event_days, strict=True):
        q, d = (
            actions.shares_per_prior_share.copy(),
            actions.cash_per_prior_share.copy(),
        )
        q[event, axis] = controls.shares_per_prior_share[event, axis]
        d[event, axis] = controls.cash_per_prior_share[event, axis]
        deleted = build_economic_multi_day_targets(
            *args,
            replace(actions, shares_per_prior_share=q, cash_per_prior_share=d),
            source_rows=rows,
        )
        for k, horizon in enumerate(HORIZONS):
            prefix = rows + horizon < event
            for field in FIELDS:
                np.testing.assert_array_equal(
                    getattr(after, field)[prefix, ..., k],
                    getattr(deleted, field)[prefix, ..., k],
                )
                prefix_checks += getattr(after, field)[prefix, ..., k].size
    deltas, effects = {}, {}
    for field, key in FIELDS.items():
        a, b = getattr(before, field), getattr(after, field)
        same = (a == b) | (np.isnan(a) & np.isnan(b)) if a.dtype.kind == "f" else a == b
        local = np.argwhere(~same)
        indices = local.copy()
        indices[:, 0] = rows[local[:, 0]]
        deltas[key + "__indices"] = indices
        deltas[key + "__values"] = b[tuple(local.T)]
        effects[key] = {"changed": len(local)}
        if b.dtype.kind == "b":
            effects[key].update(gains=int((b & ~a).sum()), losses=int((a & ~b).sum()))
        np.save(out / (key + "_rows.npy"), b)
    np.save(out / "rows.npy", rows)
    np.savez_compressed(out / "deltas.npz", **deltas)
    write_json_atomic(out / "gross_endpoint_oracle.json", {"rows": oracles})
    report = {
        "status": "passed_gross_target_attribution_with_accepted_sigma_fixed_not_final_store",
        "parent_store": source,
        "source_qualification": run["held_event_source_qualification"],
        "original_terms": binding(original_path),
        "replacement_terms": binding(out / "corporate_actions_verified_terms.parquet"),
        "superseded_terms": binding(out / "superseded_terms.parquet"),
        "deltas": binding(out / "deltas.npz"),
        "oracle": binding(out / "gross_endpoint_oracle.json"),
        "rows": binding(out / "rows.npy"),
        "dates": len(rows),
        "names": len(isins),
        "control_cells": control_cells,
        "original_endpoint_oracles": len(oracles),
        "future_endpoint_prefix_cells": prefix_checks,
        "effects": effects,
        "scope": "Only two original Natura inferred q/cash events change. Original target builder, all933 cross-sections, horizons, entry/endpoints and accepted sigma are fixed. This is gross entitlement attribution before the required daily/native/auxiliary risk propagation. Physical target_primary is not the actual virtual characteristic-neutral target.",
        "remaining": "New final sigma/slow risk and virtual neutral targets, wealth/features/native/cash/JCP custody and account treatment need separate propagation/qualification. No old checkpoint may consume these changes.",
        "seconds": perf_counter() - started,
    }
    write_json_atomic(out / "manifest.json", report)
    run["natura_gross_target_attribution"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
