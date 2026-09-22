"""Extend three surviving legal issuers using accepted own-version caches."""

import copy
from datetime import date
import json
from pathlib import Path
import pickle
from time import perf_counter

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from brazil_rv.v2 import round5_cvm as cvm
from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from propagate_fca_financials import compare
from repair_scaling_inputs import PROJECT, context, record

KEYS = ["date", "isin"]


def main():
    tick = perf_counter()
    run, plan, root, manifest, _ = context()
    admission = bound_json(run["scaling_data_identity"])
    prior = bound_json(run["stage_c_event_issuers"])
    out = Path(bound_json(run["scaling_data_workspace"])["root"]) / "issuers"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    (out / "executed_cvm.py").write_bytes(Path(cvm.__file__).read_bytes())
    dates, names = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    sessions = dates.astype(object).tolist()
    identity = pl.read_parquet(root / manifest["tables"]["issuer_identity"]["path"])
    old_links = pl.read_parquet(prior["artifacts"]["financial_history_links"]["path"])
    history = pl.read_parquet(admission["history_mapping"]["path"])
    new_links = history.filter(
        pl.col("predecessor_index").is_in(
            [names.index(e["isin"]) for e in plan["history"]]
        )
    )
    assert new_links.height == 4
    links = pl.concat([old_links, new_links]).sort("effective_index")
    write_json_atomic(
        out / "plan.json",
        dict(
            parent=plan["parent"],
            source_plan=run["scaling_data_plan"],
            admission=run["scaling_data_identity"],
            parent_issuers=run["stage_c_event_issuers"],
            contract="Four same-legal-company/class links for QGEP/ENAT,GPC/DEXP ON/PN,Wiz. Preserve all previously excluded acquired-issuer history. Reuse audited own-version caches; no source extraction/census or loan alias. Raw prices,DISMES and already accepted detected/ambiguous unit barriers are unchanged by the wealth-only scalar correction; retain them independently of retrospective q/cash. Verify old-coordinate controls for only the three affected legal issuers.",
        ),
    )
    updated = cvm.continue_issuer_identity(identity, new_links, sessions, names)
    added = updated.join(identity.select(KEYS), on=KEYS, how="anti")
    removed = identity.join(updated.select(KEYS), on=KEYS, how="anti")
    shared = updated.select(KEYS).join(identity.select(KEYS), on=KEYS)
    assert_frame_equal(
        identity.join(shared, on=KEYS).sort(KEYS),
        updated.join(shared, on=KEYS).sort(KEYS),
        check_exact=True,
    )
    selected_names = {e[k] for e in plan["history"] for k in ("isin", "successor_isin")}
    roots = sorted(
        set(identity.filter(pl.col("isin").is_in(selected_names))["cnpj"])
        | set(updated.filter(pl.col("isin").is_in(selected_names))["cnpj"])
    )
    assert len(roots) == 3
    affected = sorted(
        set(identity.filter(pl.col("cnpj").is_in(roots))["isin"])
        | set(updated.filter(pl.col("cnpj").is_in(roots))["isin"])
    )
    for link in new_links.iter_rows(named=True):
        first = max(link["effective_index"], link["known_index"])
        before = (
            identity.filter(
                (pl.col("isin") == names[link["predecessor_index"]])
                & (pl.col("date") < sessions[first])
            )
            .sort("date")
            .row(-1, named=True)
        )
        after = updated.filter(
            (pl.col("isin") == names[link["successor_index"]])
            & (pl.col("date") == sessions[first])
        ).row(0, named=True)
        assert all(
            before[k] == after[k]
            for k in (
                "cnpj",
                "cvm_code",
                "class",
                "preferred_class",
                "unit_composition",
            )
        )
        prefix = cvm.continue_issuer_identity(
            identity.filter(pl.col("date") < sessions[first]),
            new_links,
            sessions[:first],
            names,
        )
        assert_frame_equal(
            prefix.sort(KEYS),
            updated.filter(pl.col("date") < sessions[first]).sort(KEYS),
            check_exact=True,
        )
    active = np.load(root / manifest["arrays"]["active"]["path"]).copy()
    with np.load(admission["deltas"]["path"]) as z:
        active[tuple(z["active__indices"].T)] = z["active__values"]
    ti, ni = np.where(active)
    active_keys = pl.DataFrame(dict(date=dates[ti], isin=np.asarray(names)[ni]))
    assert removed.join(active_keys, on=KEYS).is_empty()
    for name, frame in (
        ("identity", updated),
        ("added", added),
        ("removed", removed),
        ("financial_history_links", links),
    ):
        frame.write_parquet(out / f"{name}.parquet")
    family = bound_json(
        json.loads((PROJECT / "docs/v2_data_inputs.json").read_text())[
            "financial_family"
        ]
    )
    fca = bound_json(run["fca_financial_propagation"])
    source_root = Path(bound_json(family["original_family"])["source_root"])
    documents, _ = pickle.loads(
        Path(family["extraction"]["cache"]["path"]).read_bytes()
    )
    extra, _ = pickle.loads(
        Path(fca["artifacts"]["new_issuer_documents"]["path"]).read_bytes()
    )
    roots8 = {c[:8] for c in roots}
    documents = [d for d in [*documents, *extra] if d["cnpj"][:8] in roots8]
    (out / "selected_documents.pkl").write_bytes(pickle.dumps(documents))
    codes = set(updated.filter(pl.col("cnpj").is_in(roots))["cvm_code"])
    rad = [r for r in cvm.rad_rows(source_root) if r["cvm_code"] in codes]
    changes = [
        r for r in bound_json(family["capital_changes"]) if r["cnpj"][:8] in roots8
    ]
    for r in changes:
        r["effective"] = date.fromisoformat(r["effective"])
    old_financial = pl.read_parquet(prior["artifacts"]["fundamentals"]["path"])
    old_events = pl.read_parquet(prior["artifacts"]["events"]["path"])
    market = cvm.valuation_market(root, history_links=old_links.to_dicts())
    outputs, audits = [], []
    for label, ids, mapping in (
        ("control", identity, old_links),
        ("corrected", updated, links),
    ):
        market["history_links"] = mapping.to_dicts()
        frame, audit = cvm.fundamental_features(
            copy.deepcopy(documents),
            rad,
            ids.filter(pl.col("cnpj").is_in(roots)),
            sessions,
            market,
            changes,
        )
        frame.write_parquet(out / f"{label}_fundamentals.parquet")
        outputs.append(frame)
        audits.append(audit)
        print(
            json.dumps(
                dict(stage=label, rows=frame.height, seconds=perf_counter() - tick)
            ),
            flush=True,
        )
    expected = old_financial.filter(pl.col("isin").is_in(affected)).sort(KEYS)
    assert_frame_equal(outputs[0].sort(KEYS), expected, check_exact=True)
    financial = pl.concat(
        [old_financial.filter(~pl.col("isin").is_in(affected)), outputs[1]]
    ).sort(KEYS)
    financial.write_parquet(out / "fundamentals.parquet")
    calendar = json.loads((source_root / "calendar_2025_announced.json").read_text())
    for key in ("available_date", "base_through", "through"):
        calendar[key] = date.fromisoformat(calendar[key])
    calendar["full_sessions"] = sessions + [
        date.fromisoformat(d) for d in calendar["sessions"]
    ]
    events = [r for r in rad if r["group"] != "cadastre"]
    headers = [
        d
        for d in cvm.filing_headers(source_root, "itr")
        + cvm.filing_headers(source_root, "dfp")
        if d["cvm_code"] in codes
    ]
    state = cvm.event_features(
        [*events, *cvm.header_only_filing_lags(events, headers)], sessions, calendar
    )

    def event_rows(ids):
        return (
            ids.filter(pl.col("cnpj").is_in(roots))
            .select("date", "isin", "cvm_code")
            .join(state, on=["date", "cvm_code"], how="left")
            .drop("cvm_code")
            .sort(KEYS)
        )

    assert_frame_equal(
        event_rows(identity),
        old_events.filter(pl.col("isin").is_in(affected)).sort(KEYS),
        check_exact=True,
    )
    event_frame = pl.concat(
        [old_events.filter(~pl.col("isin").is_in(affected)), event_rows(updated)]
    ).sort(KEYS)
    event_frame.write_parquet(out / "events.parquet")
    first = min(date.fromisoformat(e["effective_date"]) for e in plan["history"])
    for before, after in (
        (identity, updated),
        (old_financial, financial),
        (old_events, event_frame),
    ):
        assert_frame_equal(
            before.filter(pl.col("date") < first).sort(KEYS),
            after.filter(pl.col("date") < first).sort(KEYS),
            check_exact=True,
        )
    report = dict(
        status="own_issuer_dependencies_qualified",
        plan=binding(out / "plan.json"),
        parent=plan["parent"],
        affected_roots=roots,
        affected_isins=affected,
        added=added.height,
        removed=removed.height,
        added_active=added.join(active_keys, on=KEYS).height,
        controls=dict(
            financial_cells=expected.height * (expected.width - 2),
            event_cells=event_rows(identity).height * (event_frame.width - 2),
            exact=True,
            identity_future_deletion=True,
        ),
        selected_documents=len(documents),
        audits=audits,
        active_changes=dict(
            financial=compare(old_financial, financial, active_keys),
            events=compare(old_events, event_frame, active_keys),
        ),
        artifacts={p.stem: binding(p) for p in out.glob("*.parquet")},
        caches=[
            family["extraction"]["cache"],
            fca["artifacts"]["new_issuer_documents"],
        ],
        seconds=perf_counter() - tick,
        limits="Existing own-version sources only; no invented capital denominator or rate/balance/issuer alias. Announced2025 calendar is metadata only. Intermediate family; complete store and new-model validation remain pending.",
    )
    record(run, "scaling_data_issuers", out / "manifest.json", report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "added",
                    "removed",
                    "added_active",
                    "controls",
                    "seconds",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
