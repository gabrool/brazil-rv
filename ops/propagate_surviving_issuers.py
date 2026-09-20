"""Propagate the two admitted surviving issuers using audited own-version caches."""

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
from propagate_rename_issuers import bound_path

PROJECT = Path(__file__).resolve().parents[1]
KEYS = ["date", "isin"]


def main():
    tick = perf_counter()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    admission = bound_json(run["surviving_rename_admission"])
    prior = bound_json(run["rename_issuer_propagation"])
    out = Path(admission["plan"]["path"]).parent / "issuers"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    (out / "executed_cvm.py").write_bytes(Path(cvm.__file__).read_bytes())
    store = Path(admission["parent"]["root"])
    dates = np.load(store / "date_index.npy")
    sessions, isins = (
        dates.astype(object).tolist(),
        np.load(store / "isin_index.npy").tolist(),
    )
    assert sessions[-1] == date(2024, 12, 30)
    links = pl.read_parquet(bound_path(admission["history_mapping"]))
    old_links = pl.read_parquet(store / "slow_history_links.parquet")
    write_json_atomic(
        out / "plan.json",
        {
            "parent": admission["parent"],
            "admission": run["surviving_rename_admission"],
            "prior_issuer": run["rename_issuer_propagation"],
            "contrast": "Only source-clocked SSBR->ALSO and ARZZ->AZZA surviving issuer identities and their own-version financial/event dependencies. Keep old three links; no acquired issuer pooling, new extraction or loan alias.",
            "verification": "Rebuild old-coordinate controls only for the two affected legal issuers from accepted caches. Exact all-field control, unaffected rows and pre-effect prefix; independent identity core/clocks and future deletion. New data remain intermediates, no model inference.",
            "registration": binding(
                PROJECT / "research/preregistrations/v2_economic_data_scaling.md"
            ),
        },
    )
    identity = pl.read_parquet(bound_path(prior["artifacts"]["identity"]))
    updated = cvm.continue_issuer_identity(identity, links, sessions, isins)
    added = updated.join(identity.select(KEYS), on=KEYS, how="anti")
    removed = identity.join(updated.select(KEYS), on=KEYS, how="anti")
    shared = updated.select(KEYS).join(identity.select(KEYS), on=KEYS)
    assert_frame_equal(
        updated.join(shared, on=KEYS).sort(KEYS),
        identity.join(shared, on=KEYS).sort(KEYS),
        check_exact=True,
    )
    roots = sorted(set(added["cnpj"]) | set(removed["cnpj"]))
    assert len(roots) == 2
    names = sorted(
        set(identity.filter(pl.col("cnpj").is_in(roots))["isin"])
        | set(updated.filter(pl.col("cnpj").is_in(roots))["isin"])
    )
    for row in links.filter(
        pl.col("predecessor_index").is_in(
            [isins.index(x) for x in ["BRSSBRACNOR1", "BRARZZACNOR3"]]
        )
    ).iter_rows(named=True):
        begin = max(row["effective_index"], row["known_index"])
        predecessor = (
            identity.filter(
                (pl.col("isin") == isins[row["predecessor_index"]])
                & (pl.col("date") < sessions[begin])
            )
            .sort("date")
            .row(-1, named=True)
        )
        successor = updated.filter(
            (pl.col("isin") == isins[row["successor_index"]])
            & (pl.col("date") == sessions[begin])
        ).row(0, named=True)
        assert all(
            predecessor[k] == successor[k]
            for k in [
                "cnpj",
                "cvm_code",
                "class",
                "preferred_class",
                "unit_composition",
            ]
        )
        prefix = cvm.continue_issuer_identity(
            identity.filter(pl.col("date") < sessions[begin]),
            links,
            sessions[:begin],
            isins,
        )
        assert_frame_equal(
            prefix, updated.filter(pl.col("date") < sessions[begin]), check_exact=True
        )
    active = np.load(store / "active.npy").copy()
    with np.load(bound_path(admission["liquidity_deltas"])) as z:
        active[tuple(z["active__indices"].T)] = z["active__values"]
    ti, ni = np.where(active)
    active_keys = pl.DataFrame({"date": dates[ti], "isin": np.asarray(isins)[ni]})
    assert removed.join(active_keys, on=KEYS).is_empty()
    for name, frame in [("identity", updated), ("added", added), ("removed", removed)]:
        frame.write_parquet(out / f"{name}.parquet")
    print(
        json.dumps(
            {
                "stage": "identity",
                "roots": roots,
                "added": len(added),
                "removed": len(removed),
            }
        ),
        flush=True,
    )

    foundation = json.loads((PROJECT / "docs/v2_data_inputs.json").read_text())
    family = bound_json(foundation["financial_family"])
    previous = bound_json(run["fca_financial_propagation"])
    source_root = Path(bound_json(family["original_family"])["source_root"])
    documents, _ = pickle.loads(bound_path(family["extraction"]["cache"]).read_bytes())
    extra, _ = pickle.loads(
        bound_path(previous["artifacts"]["new_issuer_documents"]).read_bytes()
    )
    documents = [
        d for d in [*documents, *extra] if d["cnpj"][:8] in {c[:8] for c in roots}
    ]
    codes = set(updated.filter(pl.col("cnpj").is_in(roots))["cvm_code"])
    rad = [r for r in cvm.rad_rows(source_root) if r["cvm_code"] in codes]
    changes = [
        x
        for x in bound_json(family["capital_changes"])
        if x["cnpj"][:8] in {c[:8] for c in roots}
    ]
    for x in changes:
        x["effective"] = date.fromisoformat(x["effective"])
    write_json_atomic(
        out / "selected_documents.json",
        [
            {
                k: d[k].isoformat() if isinstance(d.get(k), date) else d.get(k)
                for k in [
                    "id",
                    "cnpj",
                    "cvm_code",
                    "kind",
                    "reference",
                    "version",
                    "receipt",
                ]
            }
            for d in documents
        ],
    )
    market = cvm.valuation_market(store, history_links=old_links.to_dicts())
    reports = []
    for label, ids, history in [
        ("control", identity, old_links),
        ("corrected", updated, links),
    ]:
        market["history_links"] = history.to_dicts()
        financial, audit = cvm.fundamental_features(
            copy.deepcopy(documents),
            rad,
            ids.filter(pl.col("cnpj").is_in(roots)),
            sessions,
            market,
            changes,
        )
        financial.write_parquet(out / f"{label}_fundamentals.parquet")
        reports.append(audit)
        print(
            json.dumps(
                {
                    "stage": label,
                    "rows": len(financial),
                    "seconds": perf_counter() - tick,
                }
            ),
            flush=True,
        )
    old_financial = pl.read_parquet(bound_path(prior["artifacts"]["fundamentals"]))
    control = pl.read_parquet(out / "control_fundamentals.parquet")
    assert_frame_equal(
        control.sort(KEYS),
        old_financial.filter(pl.col("isin").is_in(names)).sort(KEYS),
        check_exact=True,
    )
    financial = pl.concat(
        [
            old_financial.filter(~pl.col("isin").is_in(names)),
            pl.read_parquet(out / "corrected_fundamentals.parquet"),
        ]
    ).sort(KEYS)
    financial.write_parquet(out / "fundamentals.parquet")

    calendar = json.loads((source_root / "calendar_2025_announced.json").read_text())
    bound_path(calendar["source"])
    for key in ["available_date", "base_through", "through"]:
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
    old_events = pl.read_parquet(bound_path(prior["artifacts"]["events"]))

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
        old_events.filter(pl.col("isin").is_in(names)).sort(KEYS),
        check_exact=True,
    )
    event_frame = pl.concat(
        [old_events.filter(~pl.col("isin").is_in(names)), event_rows(updated)]
    ).sort(KEYS)
    event_frame.write_parquet(out / "events.parquet")
    for before, after in [(old_financial, financial), (old_events, event_frame)]:
        assert_frame_equal(
            before.filter(~pl.col("isin").is_in(names)),
            after.filter(~pl.col("isin").is_in(names)),
            check_exact=True,
        )
        assert_frame_equal(
            before.filter(pl.col("date") < date(2019, 8, 6)),
            after.filter(pl.col("date") < date(2019, 8, 6)),
            check_exact=True,
        )
    report = {
        "status": "qualified_surviving_issuer_intermediate",
        "plan": binding(out / "plan.json"),
        "parent": admission["parent"],
        "prior": run["rename_issuer_propagation"],
        "identity_rows": len(updated),
        "added_rows": len(added),
        "removed_rows": len(removed),
        "added_active": len(added.join(active_keys, on=KEYS)),
        "affected_roots": roots,
        "affected_isins": names,
        "selected_documents": len(documents),
        "controls": {
            "financial_cells": control.height * (control.width - 2),
            "event_cells": event_rows(identity).height * (event_frame.width - 2),
            "exact": True,
            "identity_future_deletion": True,
            "unaffected_exact": True,
        },
        "active_changes": {
            "financial": compare(old_financial, financial, active_keys),
            "events": compare(old_events, event_frame, active_keys),
        },
        "financial_audits": reports,
        "artifacts": {p.stem: binding(p) for p in out.glob("*.parquet")},
        "source_caches": [
            family["extraction"]["cache"],
            previous["artifacts"]["new_issuer_documents"],
        ],
        "source_family": foundation["financial_family"],
        "seconds": perf_counter() - tick,
        "limits": "Audited own-version caches reused; no extraction, disclosure-completeness/revision claim, acquired issuer pooling or loan alias. Announced2025 calendar is clock metadata only. All downstream family/tensor/store admission remains separate; no model result.",
    }
    write_json_atomic(out / "manifest.json", report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "status",
                    "seconds",
                    "controls",
                    "added_rows",
                    "removed_rows",
                    "added_active",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
