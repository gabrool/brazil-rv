"""Compose qualified actual-exposure data layers and their final gross targets."""

import json
import copy
from pathlib import Path
import subprocess
import sys
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
from compose_surviving_store import read_layers, changed
from propagate_corporate_targets import FIELDS
from verify_corporate_replay import inputs_on_axes

PROJECT = Path(__file__).resolve().parents[1]
LAYERS = (
    "data_admission",
    "wealth",
    "daily_qualification",
    "context",
    "market",
    "m1",
    "scalars",
    "sidecars",
)


def main(mode):
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    admission = bound_json(run["stage_c_event_data_admission"])
    root = Path(admission["parent"]["root"])
    m = bound_json(
        dict(
            path=str(root / "manifest.json"),
            sha256=admission["parent"]["manifest_sha256"],
        )
    )
    out = Path(admission["plan"]["path"]).parent / "composition"
    dates, isins = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )

    def old(k):
        return np.load(root / m["arrays"][k]["path"], mmap_mode="r")

    if mode == "assemble":
        assemble(run, admission, root, m, out, dates, isins, pointer, tick)
        return
    assert mode == "targets"
    resume = out.exists()
    out.mkdir(exist_ok=True)
    assert not (out / "control_targets.npz").exists()
    (
        out / ("executed_targets_resume.py" if resume else "executed_targets.py")
    ).write_bytes(Path(__file__).read_bytes())
    layers = [bound_json(run["stage_c_event_" + k])["deltas"] for k in LAYERS]
    source = bound_json(run["stage_c_event_source_admission"])
    plan = dict(
        parent=admission["parent"],
        layers=layers,
        evidence={k: v for k, v in run.items() if k.startswith("stage_c_event_")},
        registration=binding(
            PROJECT / "research/preregistrations/v2_economic_data_scaling.md"
        ),
        contract="Combine five admitted market episodes,240eligible dates,source-disproved JSL scalar correction and fourteen actual-exposure event outcomes with final risk/slow coordinates. Reuse all saved features/source audits. Original all933/full3717/full60/schema/support, old stores immutable. Final complete consumer and new-P/compatible-F admission follow; no old fit reads changed coordinates.",
        targets="Exact parent controls on the newly affected union. Independent new/changed/unsupported endpoint walks and final neutral projection. Preserve original prior corporate outcomes outside actual new interactions. Class/unit or acquired-issuer transitions supply gross claims, not pooled successor history.",
    )
    if not resume:
        write_json_atomic(out / "plan.json", plan)
    else:
        assert bound_json(binding(out / "plan.json"))["layers"] == layers
    patches = read_layers(layers)

    def amended(k):
        a = old(k).copy()
        for ix, v in patches.get(k, []):
            a[tuple(ix.T)] = v
        return a

    active, sigma = amended("active"), amended("target_scale_sigma")
    original = verified_action_terms_from_table(
        pl.read_parquet(root / m["tables"]["corporate_actions_verified_terms"]["path"])
    )
    corrected = []
    jsl_proof = source["originals"]["805360"]
    for t in original:
        if t.isin == "BRJSLGACNOR2" and str(t.effective_date) == "2020-11-11":
            write_json_atomic(
                out / "rejected_jsl_scalar.json",
                dict(
                    isin=t.isin,
                    date=str(t.effective_date),
                    old_shares=t.shares_per_prior_share,
                    source=jsl_proof,
                    disposition="Remove disproved inferred split row. The separately sourced JSLG11->JSL q1 conversion owns the logistics identity; no synthetic same-ISIN conversion. Prior qualified wealth/daily used algebraically identical q1/cash0; has_action is diagnostic and JSL has no M1 assignment.",
                ),
            )
            continue
        corrected.append(t)
    combined = {(t.isin, t.effective_date, t.sequence): t for t in corrected}
    links = pl.read_parquet(admission["links"]["path"])
    for t in verified_conversion_terms_from_links(links):
        key = t.isin, t.effective_date, t.sequence
        if key in combined:
            assert combined[key] == t
        else:
            combined[key] = t
    terms = tuple(combined.values())
    verified_action_terms_to_table(terms).write_parquet(
        out / "corporate_actions_verified_terms.parquet"
    )
    actions = [
        align_verified_action_terms(
            ts, dates, isins, coverage_resolved=old("action_session_resolved")
        )
        for ts in (original, terms)
    ]
    ad = {}
    for k, value in (
        ("action_shares_per_prior_share", actions[1].shares_per_prior_share),
        ("action_cash_per_prior_share", actions[1].cash_per_prior_share),
        ("action_session_resolved", actions[1].session_resolved),
        ("action_has_action", actions[1].has_action),
        ("action_successor_index", actions[1].successor_index),
    ):
        value = value.astype(old(k).dtype)
        ix = np.argwhere(changed(value, old(k)))
        ad[k + "__indices"], ad[k + "__values"] = ix, value[tuple(ix.T)]
    np.savez_compressed(out / "action_deltas.npz", **ad)
    rows = set()
    for k in ("active", "target_scale_sigma"):
        for ix, _ in patches[k]:
            rows.update(ix[:, 0].tolist())
    for e in source["events"]:
        t = int(np.searchsorted(dates, np.datetime64(e["effective_date"])))
        rows.update(range(t - max(HORIZONS), t + 1))
    rows = np.array(sorted(rows))
    np.save(out / "target_rows.npy", rows)
    base = None
    event_sets = []
    for key in ("corporate_replay", "stage_c_event_candidate_terms"):
        rec = run[key]
        spec, calendar = load_corporate_replay(
            path=rec["path"], expected_sha256=rec["sha256"]
        )
        np.testing.assert_array_equal(dates, calendar)
        if base is None:
            base = inputs_on_axes(
                Path(spec["store"]["root"]),
                dates,
                np.arange(933),
                np.arange(len(dates)),
            )
        event_sets.append(
            apply_corporate_replay(base, spec, dates, rec["sha256"]).share_distributions
        )
    close = np.round(old("raw_close").astype(np.float64), 2)
    outputs = []
    controls = []
    for label, act, elig, risk, events in zip(
        ("control", "corrected"),
        actions,
        (old("active"), active),
        (old("target_scale_sigma"), sigma),
        event_sets,
        strict=True,
    ):
        target = build_economic_multi_day_targets(
            close,
            old("observed"),
            elig,
            risk,
            act,
            source_rows=rows,
            share_distributions=events,
        )
        values = {k: getattr(target, f) for f, k in FIELDS.items()}
        np.savez_compressed(out / (label + "_targets.npz"), date_indices=rows, **values)
        if label == "control":
            controls = [
                dict(
                    key=k, cells=v.size, mismatches=int(changed(v, old(k)[rows]).sum())
                )
                for k, v in values.items()
            ]
            write_json_atomic(out / "controls.json", controls)
            assert not any(x["mismatches"] for x in controls), controls
        outputs.append(values)
    deltas, effects = {}, {}
    for k, v in outputs[1].items():
        ix = np.argwhere(changed(v, old(k)[rows]))
        values = v[tuple(ix.T)]
        ix[:, 0] = rows[ix[:, 0]]
        deltas[k + "__indices"], deltas[k + "__values"] = ix, values
        effects[k] = len(ix)
    # New source entry closes only until an evidenced new identity episode.
    entry = old("entry_fill_allowed").copy()
    barriers = []
    for e in source["events"]:
        start = int(
            np.searchsorted(
                dates, np.datetime64(max(e["effective_date"], e["available_date"]))
            )
        )
        stop = (
            len(dates)
            if e["source_reopens_date"] is None
            else int(np.searchsorted(dates, np.datetime64(e["source_reopens_date"])))
        )
        n = isins.index(e["isin"])
        assert not old("observed")[start:stop, n].any(), e["id"]
        entry[start:stop, n] = False
        barriers.append(
            dict(
                isin=e["isin"],
                start=str(dates[start]),
                stop=e["source_reopens_date"],
                eligible_unquoted=int(active[start:stop, n].sum()),
            )
        )
    ix = np.argwhere(entry != old("entry_fill_allowed"))
    deltas["entry_fill_allowed__indices"], deltas["entry_fill_allowed__values"] = (
        ix,
        entry[tuple(ix.T)],
    )
    np.savez_compressed(out / "target_deltas.npz", **deltas)
    write_json_atomic(out / "entry_barriers.json", barriers)
    result = dict(
        status="targets_produced_pending_endpoint_and_complete_qualification",
        plan=binding(out / "plan.json"),
        controls=binding(out / "controls.json"),
        deltas=binding(out / "target_deltas.npz"),
        action_deltas=binding(out / "action_deltas.npz"),
        rows=binding(out / "target_rows.npy"),
        effects=effects,
        entry_changes=len(ix),
        target_dates=len(rows),
        seconds=perf_counter() - tick,
    )
    write_json_atomic(out / "targets.json", result)
    run = json.loads(pointer.read_text())
    run["stage_c_event_targets"] = binding(out / "targets.json")
    write_json_atomic(pointer, run)
    print(json.dumps(result), flush=True)


def assemble(run, admission, root, m, out, dates, isins, pointer, tick):
    plan = bound_json(binding(out / "plan.json"))
    targets = bound_json(run["stage_c_event_targets"])
    for key in (
        "stage_c_event_targets_qualification",
        "stage_c_event_minute_qualification",
        "stage_c_event_auxiliary_arithmetic",
        "stage_c_event_sidecars",
    ):
        bound_json(run[key])
    (
        out / ("executed_assembly_" + binding(Path(__file__))["sha256"][:12] + ".py")
    ).write_bytes(Path(__file__).read_bytes())
    records = [*plan["layers"], targets["action_deltas"], targets["deltas"]]
    patches = read_layers(records)
    destination = (
        Path(run["stage_c_root"]).parent.parent / "v2_economic_matched_store_20260920"
    )
    assert not destination.exists()
    tables = {k: root / r["path"] for k, r in m["tables"].items()}
    issuer = bound_json(run["stage_c_event_issuers"])
    tables.update(
        isin_succession_links=Path(admission["links"]["path"]),
        slow_history_links=Path(admission["history_mapping"]["path"]),
        issuer_identity=Path(issuer["artifacts"]["identity"]["path"]),
        financial_history_links=Path(
            issuer["artifacts"]["financial_history_links"]["path"]
        ),
        corporate_actions_verified_terms=out
        / "corporate_actions_verified_terms.parquet",
    )
    terms = verified_action_terms_from_table(
        pl.read_parquet(tables["corporate_actions_verified_terms"])
    )
    old_terms = verified_action_terms_from_table(
        pl.read_parquet(root / m["tables"]["corporate_actions_verified_terms"]["path"])
    )
    previous = {(t.isin, t.effective_date, t.sequence): t for t in old_terms}
    new_terms = [
        t for t in terms if previous.get((t.isin, t.effective_date, t.sequence)) != t
    ]
    schedule = load_session_schedule(
        Path(bound_json(admission["plan"])["schedule"]["path"])
    )
    sessions = tuple(
        s for s in schedule if dates[0] <= np.datetime64(s.trade_date) <= dates[-1]
    )
    following = next(
        s.decision_at for s in schedule if np.datetime64(s.trade_date) > dates[-1]
    )
    cutoffs = next_session_decision_cutoffs(sessions, following_decision_at=following)
    prior_roles = pl.read_parquet(tables["corporate_action_alignment_roles"])
    prior_roles = prior_roles.filter(
        ~(
            (pl.col("isin") == "BRJSLGACNOR2")
            & (pl.col("event_date").cast(pl.String) == "2020-11-11")
        )
    )
    tables["corporate_action_alignment_roles"] = pl.concat(
        [prior_roles, _action_alignment_role_table(new_terms, dates, cutoffs)]
    ).sort("event_date", "isin")
    context = bound_json(run["stage_c_event_context"])
    common = pl.read_parquet(context["artifacts"]["common_rows"]["path"])
    tables["common_state_diagnostics"] = pl.concat(
        [
            pl.read_parquet(tables["common_state_diagnostics"]).join(
                common.select("trade_date"), on="trade_date", how="anti"
            ),
            common,
        ]
    ).sort("trade_date")
    tables["matched_corporate_source_entry_barriers"] = pl.DataFrame(
        json.loads((out / "entry_barriers.json").read_text())
    )
    active = np.load(root / "active.npy").copy()
    for ix, v in patches["active"]:
        active[tuple(ix.T)] = v
    tables["universe_size"] = pl.DataFrame(
        dict(date=dates, active_names=active.sum(axis=1))
    )
    contract = dict(
        parent=admission["parent"],
        output=str(destination),
        plan=binding(out / "plan.json"),
        layers=records,
        evidence={
            k: run[k]
            for k in (
                "stage_c_event_targets_qualification",
                "stage_c_event_minute_qualification",
                "stage_c_event_auxiliary_arithmetic",
                "stage_c_event_sidecars",
            )
        },
        account_terms=run["stage_c_event_candidate_terms"],
        refit="New training-only conditioning and compatible newly fitted P/F weights. Preserve original graphs/folds/seeds/full933/full60/learning budget/optimizer/selector. Optional to-close loss inactive. Old accounting forecasts keep OLDPolicyData.",
        episodes="Historical public old-JSL values are preserved. The backward history walk may only traverse decreasing event times; new logistics-JSL history cannot enter old holding/SIMPAR histories. Market continuity never pools distinct legal issuer filings or loan-source aliases.",
        storage="Independent full NPY files, lossless NTFS while writing and LZX read-only compression after each array is finalized. Exact decoded bytes are verified by the manifest and independent composition. No raw/parent/old-fit changes or hardlinks. Prior ordinary-NTFS assembly exhausted space and its unsealed staging was automatically removed; saved numerical layers reused.",
    )
    contract_path = PROJECT / "docs/v2_matched_store_contract.json"
    write_json_atomic(contract_path, contract)
    metadata = copy.deepcopy(m["metadata"])
    metadata["matched_source_data_contract"] = dict(
        contract=binding(contract_path),
        parent=admission["parent"],
        refit_required=True,
        inherited_diagnostics="Earlier counts remain historical receipts; composition evidence and current arrays supersede them.",
    )
    metadata["isin_succession_link_count"] = 10
    effects = {key: sum(len(ix) for ix, _ in values) for key, values in patches.items()}
    tables["matched_array_changes"] = pl.DataFrame(
        [dict(array=k, changed_cells=v) for k, v in effects.items()]
    )
    with StoreStaging(destination, dates=dates, isins=isins) as stage:
        result = subprocess.run(
            ["compact.exe", "/C", str(stage.staging)],
            capture_output=True,
            text=True,
            check=True,
        )
        (out / "storage_compression.txt").write_text(result.stdout)
        for key, rec in m["arrays"].items():
            stage.copy_array(key, root / rec["path"])
            if key in patches:
                array = stage.open_array(key, mode="r+")
                touched = set()
                for ix, v in patches[key]:
                    flat = np.ravel_multi_index(ix.T, array.shape)
                    assert not touched.intersection(flat.tolist()), key
                    touched.update(flat.tolist())
                    array[tuple(ix.T)] = v
                close_memmap(array)
            result = subprocess.run(
                ["compact.exe", "/C", "/F", "/EXE:LZX", str(stage.array_path(key))],
                capture_output=True,
                text=True,
                check=True,
            )
            with (out / "storage_compression.txt").open("a") as log:
                log.write(result.stdout)
            print(json.dumps(dict(array_stored=key)), flush=True)
        stage.seal(
            feature_names=m["feature_names"],
            sources=[*m["sources"], binding(out / "plan.json")],
            metadata=metadata,
            tables=tables,
        )
    result = dict(
        status="complete_matched_store_pending_combined_consumer",
        store=dict(
            root=str(destination),
            manifest_sha256=binding(destination / "manifest.json")["sha256"],
        ),
        parent_store=admission["parent"],
        contract=binding(contract_path),
        plan=binding(out / "plan.json"),
        effects=effects,
        seconds=perf_counter() - tick,
    )
    write_json_atomic(out / "manifest.json", result)
    run = json.loads(pointer.read_text())
    run["stage_c_event_store_assembly"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
