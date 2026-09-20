"""Compose the qualified Natura layer onto the immutable economic baseline."""

import copy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.build_store import _action_alignment_role_table
from brazil_rv.v2.corporate_actions import (
    align_action_payment_sessions,
    align_decision_known_action_terms,
    verified_action_terms_from_table,
)
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.decision_clock import (
    load_session_schedule,
    next_session_decision_cutoffs,
)
from brazil_rv.v2.intraday_features import decision_action_boundaries
from brazil_rv.v2.store import StoreStaging, close_memmap

PROJECT = Path(__file__).resolve().parents[1]


def main():
    started = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    records = {
        k: run[k]
        for k in (
            "natura_wealth_amendment",
            "natura_daily_input_audit",
            "natura_auxiliary_qualification",
            "natura_dependency_scope",
            "natura_gross_target_attribution",
            "held_event_source_qualification",
        )
    }
    reports = {k: bound_json(v) for k, v in records.items()}
    parent = reports["natura_daily_input_audit"]["parent_store"]
    root = Path(parent["root"])
    original = bound_json(
        {"path": str(root / "manifest.json"), "sha256": parent["manifest_sha256"]}
    )
    dates, isins = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    assert len(dates) == 3717 and len(isins) == 933 and str(dates[-1]) == "2024-12-30"
    out = Path(run["root"]) / "natura_store"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    destination = root.parent / "v2_economic_natura_store_20260920"
    deltas = {}
    for key in (
        "natura_wealth_amendment",
        "natura_daily_input_audit",
        "natura_auxiliary_qualification",
    ):
        with np.load(reports[key]["deltas"]["path"]) as z:
            for member in z.files:
                if member.endswith("__indices"):
                    field = member.removesuffix("__indices")
                    if len(z[member]):
                        assert field not in deltas
                        deltas[field] = (z[member].copy(), z[field + "__values"].copy())

    def old(key):
        return np.load(root / original["arrays"][key]["path"], mmap_mode="r")

    source = reports["held_event_source_qualification"]
    terms = verified_action_terms_from_table(
        pl.read_parquet(source["amendments"]["path"])
    )
    n = isins.index(terms[0].isin)
    event_rows = np.array(
        [np.searchsorted(dates, np.datetime64(t.ex_date)) for t in terms]
    )
    ix = np.column_stack((event_rows, np.full(len(terms), n)))
    payments = align_action_payment_sessions(terms, dates, [isins[n]])
    for key, values in (
        ("action_shares_per_prior_share", [t.shares_per_prior_share for t in terms]),
        ("action_cash_per_prior_share", [t.cash_per_prior_share for t in terms]),
        ("action_payment_session", payments[event_rows, 0]),
    ):
        values = np.asarray(values, dtype=old(key).dtype)
        changed = values != old(key)[tuple(ix.T)]
        if changed.any():
            deltas[key] = (ix[changed], values[changed])
    full_schedule = load_session_schedule(Path(source["schedule"]["path"]))
    sessions = tuple(
        s for s in full_schedule if dates[0] <= np.datetime64(s.trade_date) <= dates[-1]
    )
    following = next(
        s.decision_at for s in full_schedule if np.datetime64(s.trade_date) > dates[-1]
    )
    cutoffs = next_session_decision_cutoffs(sessions, following_decision_at=following)
    gross = reports["natura_gross_target_attribution"]
    superseded = verified_action_terms_from_table(
        pl.read_parquet(gross["superseded_terms"]["path"])
    )
    updated_boundary = []
    for term, row in zip(terms, event_rows, strict=True):
        window = slice(row - 2, row + 3)
        s = sessions[row - 2 : row + 3]
        known = align_decision_known_action_terms(
            terms,
            dates[window],
            [isins[n]],
            coverage_resolved=old("observed")[window, n, None],
            decision_timestamps=cutoffs[row - 2 : row + 3],
        )
        after = decision_action_boundaries(
            old("raw_open")[window, n, None],
            old("raw_close")[window, n, None],
            old("observed")[window, n, None],
            known.session_resolved,
            terms,
            s,
            [isins[n]],
        )
        before_known = align_decision_known_action_terms(
            superseded,
            dates[window],
            [isins[n]],
            coverage_resolved=old("observed")[window, n, None],
            decision_timestamps=cutoffs[row - 2 : row + 3],
        )
        before = decision_action_boundaries(
            old("raw_open")[window, n, None],
            old("raw_close")[window, n, None],
            old("observed")[window, n, None],
            before_known.session_resolved,
            superseded,
            s,
            [isins[n]],
        )
        np.testing.assert_array_equal(
            before[2, 0], old("intraday_unit_or_unresolved_boundary_mask")[row, n]
        )
        updated_boundary.append(after[2, 0])
    updated_boundary = np.array(updated_boundary, bool)
    changed = (
        updated_boundary
        != old("intraday_unit_or_unresolved_boundary_mask")[tuple(ix.T)]
    )
    if changed.any():
        deltas["intraday_unit_or_unresolved_boundary_mask"] = (
            ix[changed],
            updated_boundary[changed],
        )
    action_fields = {
        key: value
        for key, value in deltas.items()
        if key.startswith("action_")
        or key == "intraday_unit_or_unresolved_boundary_mask"
    }
    np.savez_compressed(
        out / "action_deltas.npz",
        **{
            key + suffix: part
            for key, pair in action_fields.items()
            for suffix, part in zip(("__indices", "__values"), pair, strict=True)
        },
    )
    alignment = pl.read_parquet(
        root / original["tables"]["corporate_action_alignment_roles"]["path"]
    )
    replaced = (pl.col("isin") == isins[n]) & pl.col("event_date").is_in(
        [t.ex_date for t in terms]
    )
    assert alignment.filter(replaced).height == 2
    alignment = pl.concat(
        [
            alignment.filter(~replaced),
            _action_alignment_role_table(terms, dates, cutoffs),
        ]
    ).sort("event_date", "isin")
    alignment.write_parquet(out / "corporate_action_alignment_roles.parquet")
    contract = {
        "parent": parent,
        "output": str(destination),
        "amendments": records,
        "action_deltas": binding(out / "action_deltas.npz"),
        "scope": "Exactly the Sep18 q2 and Nov7 gross JCP Natura corrections and their existing daily/risk/auxiliary/target dependencies. No NATU->NTCO succession or issuer inheritance. No custody, withholding or settled-account inference.",
        "refit_contract": "New P training-only conditioning and weights on this exact store; compatible new F children only. Preserve registered graph/folds/seeds/full933/full60/budget/optimizer/selector. Same schema does not mean compatible coordinates. Optional to-close loss stays inactive.",
        "remaining_account_program": "Stage A held events, delivery/tax/fraction/pricing and remaining adaptive bounds; C/D unstarted.",
        "unchanged_scope": reports["natura_dependency_scope"],
        "recovery_parent": run["economic_store_recovery"],
    }
    contract_path = PROJECT / "docs/v2_natura_store_contract.json"
    write_json_atomic(contract_path, contract)
    effects = {key: len(ix) for key, (ix, _) in deltas.items()}
    write_json_atomic(
        out / "plan.json",
        {
            "contract": binding(contract_path),
            "effects": effects,
            "tables": {
                "corporate_actions_verified_terms": gross["replacement_terms"],
                "corporate_action_alignment_roles": binding(
                    out / "corporate_action_alignment_roles.parquet"
                ),
            },
        },
    )
    metadata = copy.deepcopy(original["metadata"])
    metadata["natura_source_amendment"] = {
        "contract": binding(contract_path),
        "parent": parent,
        "refit_required": True,
    }
    tables = {key: root / rec["path"] for key, rec in original["tables"].items()}
    tables.update(
        corporate_actions_verified_terms=Path(gross["replacement_terms"]["path"]),
        corporate_action_alignment_roles=out
        / "corporate_action_alignment_roles.parquet",
    )
    common = reports["natura_daily_input_audit"]["composition"]["path"]
    common_rows = pl.read_parquet(Path(common).parent / "common_state_rows.parquet")
    original_common = pl.read_parquet(tables["common_state_diagnostics"])
    tables["common_state_diagnostics"] = pl.concat(
        [
            original_common.join(
                common_rows.select("trade_date"), on="trade_date", how="anti"
            ),
            common_rows,
        ]
    ).sort("trade_date")
    with StoreStaging(destination, dates=dates, isins=isins) as staging:
        for key, record in original["arrays"].items():
            staging.copy_array(key, root / record["path"])
            if key in deltas:
                a = staging.open_array(key, mode="r+")
                indices, values = deltas[key]
                a[tuple(indices.T)] = values
                close_memmap(a)
        staging.seal(
            feature_names=original["feature_names"],
            sources=[*original["sources"], binding(out / "plan.json")],
            metadata=metadata,
            tables=tables,
        )
    result = {
        "status": "assembled_complete_natura_store_pending_final_acceptance",
        "store": {
            "root": str(destination),
            "manifest_sha256": binding(destination / "manifest.json")["sha256"],
        },
        "parent_store": parent,
        "contract": binding(contract_path),
        "plan": binding(out / "plan.json"),
        "effects": effects,
        "seconds": perf_counter() - started,
    }
    write_json_atomic(out / "manifest.json", result)
    run["natura_store_assembly"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
