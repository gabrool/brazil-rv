"""Compose qualified feature layers and final targets into a new complete store."""

import copy
import json
from pathlib import Path
import sys
import subprocess
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.build_store import _action_alignment_role_table
from brazil_rv.v2.contract import HORIZONS
from brazil_rv.v2.corporate_actions import (
    align_verified_action_terms,
    verified_action_terms_from_table,
    verified_action_terms_to_table,
    verified_conversion_terms_from_links,
)
from brazil_rv.v2.corporate_replay import apply_corporate_replay, load_corporate_replay
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.decision_clock import (
    load_session_schedule,
    next_session_decision_cutoffs,
)
from brazil_rv.v2.store import StoreStaging, close_memmap
from brazil_rv.v2.targets import build_economic_multi_day_targets
from propagate_corporate_targets import FIELDS
from verify_corporate_replay import inputs_on_axes

PROJECT = Path(__file__).resolve().parents[1]
POINTER = PROJECT / "docs/v2_economic_data_scaling_run.json"
LAYERS = (
    ("surviving_rename_admission", "liquidity_deltas"),
    ("surviving_rename_wealth", "deltas"),
    ("surviving_rename_daily_qualification", "deltas"),
    ("surviving_rename_context", "common_deltas"),
    ("surviving_rename_dependency_input_audit", "deltas"),
    ("surviving_rename_auxiliary_input_audit", "deltas"),
    ("surviving_rename_m1", "native_deltas"),
    ("surviving_rename_m1", "scalar_deltas"),
    ("surviving_rename_market", "deltas"),
)


def read_layers(records):
    patches = {}
    for rec in records:
        assert binding(Path(rec["path"]))["sha256"] == rec["sha256"]
        with np.load(rec["path"]) as z:
            for key in z.files:
                if key.endswith("__indices") and len(z[key]):
                    name = key.split("__")[-2]
                    patches.setdefault(name, []).append(
                        (z[key].copy(), z[key.replace("__indices", "__values")].copy())
                    )
    return patches


def changed(a, b):
    return ~((a == b) | (np.isnan(a) & np.isnan(b)))


def main(mode):
    tick = perf_counter()
    run = json.loads(POINTER.read_text())
    out = Path(run["root"]) / "surviving_store"
    admission = bound_json(run["surviving_rename_admission"])
    parent = admission["parent"]
    root = Path(parent["root"])
    original = bound_json(
        dict(path=str(root / "manifest.json"), sha256=parent["manifest_sha256"])
    )
    dates = np.load(root / "date_index.npy")
    isins = np.load(root / "isin_index.npy").tolist()
    assert len(dates) == 3717 and len(isins) == 933 and str(dates[-1]) == "2024-12-30"

    def old(key):
        return np.load(root / original["arrays"][key]["path"], mmap_mode="r")

    if mode == "targets":
        out.mkdir(exist_ok=False)
        (out / "executed_targets.py").write_bytes(Path(__file__).read_bytes())
        for src in (
            PROJECT / "research/preregistrations/v2_economic_data_scaling.md",
            PROJECT / "docs/v2_STAGE_A_CLOSEOUT.md",
        ):
            (out / src.name).write_bytes(src.read_bytes())
        layers = [bound_json(run[k])[field] for k, field in LAYERS]
        evidence = {
            k: run[k]
            for k in dict.fromkeys(
                [k for k, _ in LAYERS]
                + [
                    "event_source_composition",
                    "event_composition_qualification",
                    "surviving_rename_issuers",
                    "surviving_rename_daily_input_audit",
                    "surviving_rename_unit_history",
                    "surviving_rename_auxiliary_arithmetic",
                    "surviving_rename_market_input_audit",
                    "composed_primary_event_terms",
                ]
            )
        }
        write_json_atomic(
            out / "plan.json",
            dict(
                parent=parent,
                feature_layers=layers,
                evidence=evidence,
                scope="Compose all qualified surviving-company features, two same-class target links and ready72 gross corporate outcomes using final sigma/slow risk. All933/full3717/full60, unchanged schema/support/normalization. No feature reducer/source census/old consumer repetition.",
                target_scope="Only rows changed by eligibility, sigma or crossing the two new unit renames, plus previously qualified corporate rows. Preserve exact unrelated raw outcomes; independent new endpoint arithmetic and actual final neutral projection. Prior72 endpoint proofs reused.",
                integration="New complete store and selected all-family consumers across new boundaries and monthly tails. Existing per-family full-history proofs reused; verify composition, not repeat their producers.",
                account="No new loan aliases, physical custody permission or acquired-issuer pooling. No account book/model forward/fit or heldout read. StageA remains pending calendar/integrated admission.",
                execution="User disabled the one-minute heartbeat; continue directly without scheduled checkpoint wakeups.",
            ),
        )
        patches = read_layers(layers)

        def corrected(key):
            value = np.array(old(key), copy=True)
            for ix, v in patches.get(key, []):
                value[tuple(ix.T)] = v
            return value

        active, sigma = corrected("active"), corrected("target_scale_sigma")
        links = pl.read_parquet(admission["links"]["path"])
        conversions = verified_conversion_terms_from_links(links)
        prior_terms = verified_action_terms_from_table(
            pl.read_parquet(
                root / original["tables"]["corporate_actions_verified_terms"]["path"]
            )
        )
        assert not any(t.resulting_isin is not None for t in prior_terms)
        terms = (*prior_terms, *conversions)
        verified_action_terms_to_table(terms).write_parquet(
            out / "corporate_actions_verified_terms.parquet"
        )
        actions = align_verified_action_terms(
            terms, dates, isins, coverage_resolved=old("action_session_resolved")
        )
        new_conversions = [
            t for t in conversions if t.isin in ("BRSSBRACNOR1", "BRARZZACNOR3")
        ]
        assert len(new_conversions) == 2
        action_deltas = {}
        for key, value in (
            ("action_shares_per_prior_share", actions.shares_per_prior_share),
            ("action_cash_per_prior_share", actions.cash_per_prior_share),
            ("action_session_resolved", actions.session_resolved),
            ("action_has_action", actions.has_action),
            ("action_successor_index", actions.successor_index),
        ):
            value = value.astype(old(key).dtype)
            ix = np.argwhere(changed(value, old(key)))
            # The three already-admitted conversions and all unrelated terms stay exact.
            allowed = {
                (
                    int(np.searchsorted(dates, np.datetime64(t.ex_date))),
                    isins.index(t.isin),
                )
                for t in new_conversions
            }
            assert all(tuple(v) in allowed for v in ix), key
            action_deltas[key + "__indices"] = ix
            action_deltas[key + "__values"] = value[tuple(ix.T)]
        np.savez_compressed(out / "action_deltas.npz", **action_deltas)
        composition = bound_json(run["event_source_composition"])
        event_rows = np.load(composition["rows"]["path"])
        rows = set(event_rows.tolist())
        for key in ("active", "target_scale_sigma"):
            for ix, _ in patches[key]:
                rows.update(ix[:, 0].tolist())
        for t in new_conversions:
            effect = int(np.searchsorted(dates, np.datetime64(t.ex_date)))
            rows.update(range(effect - max(HORIZONS), effect + 1))
        rows = np.array(sorted(rows))
        np.save(out / "target_rows.npy", rows)
        account_terms, calendar = load_corporate_replay(
            path=run["composed_primary_event_terms"]["path"],
            expected_sha256=run["composed_primary_event_terms"]["sha256"],
        )
        np.testing.assert_array_equal(dates, calendar)
        base = inputs_on_axes(
            Path(account_terms["store"]["root"]),
            dates,
            np.arange(933),
            np.arange(len(dates)),
        )
        events = apply_corporate_replay(
            base, account_terms, dates, run["composed_primary_event_terms"]["sha256"]
        ).share_distributions
        after = build_economic_multi_day_targets(
            np.round(np.asarray(old("raw_close"), np.float64), 2),
            old("observed"),
            active,
            sigma,
            actions,
            source_rows=rows,
            share_distributions=events,
        )
        deltas, effects = {}, {}
        for field, key in FIELDS.items():
            value = getattr(after, field)
            np.save(out / (key + "_rows.npy"), value)
            ix = np.argwhere(changed(value, old(key)[rows]))
            values = value[tuple(ix.T)]
            ix[:, 0] = rows[ix[:, 0]]
            deltas[key + "__indices"], deltas[key + "__values"] = ix, values
            effects[key] = len(ix)
        # Existing entry-permission proof is reused verbatim, with no target layer overwrite.
        with np.load(composition["deltas"]["path"]) as z:
            for suffix in ("__indices", "__values"):
                deltas["entry_fill_allowed" + suffix] = z["entry_fill_allowed" + suffix]
        np.savez_compressed(out / "target_deltas.npz", **deltas)
        write_json_atomic(
            out / "targets.json",
            dict(
                plan=binding(out / "plan.json"),
                rows=binding(out / "target_rows.npy"),
                deltas=binding(out / "target_deltas.npz"),
                action_deltas=binding(out / "action_deltas.npz"),
                effects=effects,
                target_dates=len(rows),
                seconds=perf_counter() - tick,
                status="produced_final_targets_pending_composition_qualification",
            ),
        )
        print(
            json.dumps(
                dict(
                    target_dates=len(rows),
                    effects=effects,
                    seconds=perf_counter() - tick,
                )
            ),
            flush=True,
        )
        return

    assert mode == "assemble"
    plan = json.loads((out / "plan.json").read_text())
    targets = json.loads((out / "targets.json").read_text())
    assert (out / "target_qualification.json").exists()
    (out / "executed_assembly.py").write_bytes(Path(__file__).read_bytes())
    records = [*plan["feature_layers"], targets["action_deltas"], targets["deltas"]]
    patches = read_layers(records)
    destination = (
        Path(run["root"]).parent.parent / "v2_economic_composed_store_20260920"
    )
    assert not destination.exists()
    tables = {key: root / rec["path"] for key, rec in original["tables"].items()}
    tables.update(
        isin_succession_links=Path(admission["links"]["path"]),
        slow_history_links=Path(admission["history_mapping"]["path"]),
        issuer_identity=Path(
            bound_json(run["surviving_rename_issuers"])["artifacts"]["identity"]["path"]
        ),
        corporate_actions_verified_terms=out
        / "corporate_actions_verified_terms.parquet",
    )
    context = bound_json(run["surviving_rename_context"])
    common = pl.read_parquet(context["artifacts"]["corrected_common_rows"]["path"])
    tables["common_state_diagnostics"] = pl.concat(
        [
            pl.read_parquet(tables["common_state_diagnostics"]).join(
                common.select("trade_date"), on="trade_date", how="anti"
            ),
            common,
        ]
    ).sort("trade_date")
    schedule = load_session_schedule(Path(admission["schedule"]["path"]))
    sessions = tuple(
        s for s in schedule if dates[0] <= np.datetime64(s.trade_date) <= dates[-1]
    )
    following = next(
        s.decision_at for s in schedule if np.datetime64(s.trade_date) > dates[-1]
    )
    cutoffs = next_session_decision_cutoffs(sessions, following_decision_at=following)
    # Original alignment already includes the earlier three links. Add only the two new rows.
    new_terms = [
        t
        for t in verified_conversion_terms_from_links(
            pl.read_parquet(tables["isin_succession_links"])
        )
        if t.isin in ("BRSSBRACNOR1", "BRARZZACNOR3")
    ]
    tables["corporate_action_alignment_roles"] = pl.concat(
        [
            pl.read_parquet(tables["corporate_action_alignment_roles"]),
            _action_alignment_role_table(new_terms, dates, cutoffs),
        ]
    ).sort("event_date", "isin")
    composition = bound_json(run["event_source_composition"])
    # Separate audit table preserves the original claim schema; neither table is a feature.
    tables["corporate_composed_claim_close_rows"] = pl.DataFrame(
        json.loads(Path(composition["claims"]["path"]).read_text())
    )
    barriers = []
    event_terms = {
        event["isin"]: event
        for event in bound_json(run["composed_primary_event_terms"])[
            "share_distributions"
        ]
    }
    for rec in json.loads(Path(composition["source_scope"]["path"]).read_text()):
        available = event_terms[rec["isin"]]["available_date"]
        barriers.append(
            dict(
                available_date=available,
                effective_date=rec["effective"],
                isin=rec["isin"],
                reason="sourced conversion closes new unquoted source entry",
                start=max(available, rec["effective"]),
            )
        )
    tables["corporate_source_entry_barriers"] = pl.concat(
        [
            pl.read_parquet(tables["corporate_source_entry_barriers"]),
            pl.DataFrame(barriers),
        ]
    )
    active = np.array(old("active"), copy=True)
    for ix, v in patches["active"]:
        active[tuple(ix.T)] = v
    tables["universe_size"] = pl.DataFrame(
        dict(date=dates, active_names=active.sum(axis=1))
    )
    effects = {key: sum(len(ix) for ix, _ in values) for key, values in patches.items()}
    tables["composed_array_changes"] = pl.DataFrame(
        [dict(array=k, changed_cells=v) for k, v in effects.items()]
    )
    contract = dict(
        parent=parent,
        output=str(destination),
        plan=binding(out / "plan.json"),
        layers=records,
        target_qualification=binding(out / "target_qualification.json"),
        refit="New training-only conditioning and P weights required; only compatible new F inherits. Same schema does not mean compatible coordinates. Original graph/folds/seeds/full933/full60/learning budget/optimizer/selector; optional to-close loss inactive.",
        accounting="Old forecasts use frozen OLDPolicyData and explicit account/source amendments. No inferred loan alias/custody permission. StageA calendar and integrated admission remain.",
        source_limits="Reuse prior source/revision/publication limits, ALSC locked unknown delivery/net cash and ENAT unknown auction. Preserve all existing peer/unit support barriers and missing M1 history.",
        storage="Lossless NTFS compression on the new staging directory only; independent files, unchanged NPY bytes, full store retained. No parent file compression or hard links.",
    )
    contract_path = PROJECT / "docs/v2_composed_store_contract.json"
    write_json_atomic(contract_path, contract)
    metadata = copy.deepcopy(original["metadata"])
    metadata["composed_source_data_contract"] = dict(
        contract=binding(contract_path),
        parent=parent,
        refit_required=True,
        inherited_diagnostics="Prior counts remain historical receipts; current arrays and composition evidence supersede them.",
    )
    metadata["isin_succession_link_count"] = 5
    with StoreStaging(destination, dates=dates, isins=isins) as staging:
        compression = subprocess.run(
            ["compact.exe", "/C", str(staging.staging)],
            capture_output=True,
            text=True,
            check=True,
        )
        (out / "storage_compression.txt").write_text(compression.stdout)
        for key, rec in original["arrays"].items():
            staging.copy_array(key, root / rec["path"])
            if key in patches:
                array = staging.open_array(key, mode="r+")
                seen = set()
                for ix, v in patches[key]:
                    flat = np.ravel_multi_index(ix.T, array.shape)
                    assert not seen.intersection(flat.tolist()), key
                    seen.update(flat.tolist())
                    array[tuple(ix.T)] = v
                close_memmap(array)
            packed = subprocess.run(
                ["compact.exe", "/C", "/F", str(staging.array_path(key))],
                capture_output=True,
                text=True,
                check=True,
            )
            with (out / "storage_compression.txt").open("a") as log:
                log.write(packed.stdout)
            print(json.dumps({"array_stored": key}), flush=True)
        staging.seal(
            feature_names=original["feature_names"],
            sources=[*original["sources"], binding(out / "plan.json")],
            metadata=metadata,
            tables=tables,
        )
    result = dict(
        status="complete_composed_store_pending_combined_consumer_qualification",
        store=dict(
            root=str(destination),
            manifest_sha256=binding(destination / "manifest.json")["sha256"],
        ),
        parent_store=parent,
        contract=binding(contract_path),
        plan=binding(out / "plan.json"),
        effects=effects,
        seconds=perf_counter() - tick,
    )
    write_json_atomic(out / "manifest.json", result)
    run["composed_store_assembly"] = binding(out / "manifest.json")
    write_json_atomic(POINTER, run)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
