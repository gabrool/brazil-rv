"""Continue source-bound issuer identities and recompute only affected issuers.

The sealed store and earlier FCA/rename overlays remain immutable. No model is
scored; this is a separately attributed dependency layer for the next store.
"""

from datetime import date
import copy
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

PROJECT = Path(__file__).resolve().parents[1]
KEYS = ["date", "isin"]


def main():
    started = perf_counter()
    run = json.loads(
        (PROJECT / "docs/v2_economic_data_scaling_run.json").read_text(encoding="utf8")
    )
    pointer = json.loads(
        (PROJECT / "docs/v2_data_inputs.json").read_text(encoding="utf8")
    )
    output = Path(run["root"]) / "rename_issuer_propagation"
    output.mkdir(exist_ok=False)
    (output / "executed_reproducer.py").write_bytes(Path(__file__).read_bytes())
    store = Path(pointer["store"]["root"])
    sessions = np.load(store / "date_index.npy").astype("datetime64[D]").tolist()
    isins = np.load(store / "isin_index.npy").tolist()
    assert sessions[-1] <= date(2024, 12, 31)
    rename = bound_json(run["rename_history_propagation"])
    old_id = bound_json(run["fca_identity_admission"])
    previous = bound_json(run["fca_financial_propagation"])
    links = pl.read_parquet(bound_path(rename["history_mapping"]))
    identity = pl.read_parquet(bound_path(old_id["artifacts"]["identity"]))
    updated = cvm.continue_issuer_identity(identity, links, sessions, isins)
    updated.write_parquet(output / "identity.parquet")
    added = updated.join(identity.select(KEYS), on=KEYS, how="anti")
    removed = identity.join(updated.select(KEYS), on=KEYS, how="anti")
    added.write_parquet(output / "added.parquet")
    removed.write_parquet(output / "removed.parquet")
    shared_keys = updated.select(KEYS).join(identity.select(KEYS), on=KEYS, how="inner")
    assert_frame_equal(
        updated.join(shared_keys, on=KEYS).sort(KEYS),
        identity.join(shared_keys, on=KEYS).sort(KEYS),
        check_exact=True,
    )
    for cutoff in [int(links["effective_index"].min()), len(sessions) - 15]:
        prefix = cvm.continue_issuer_identity(
            identity.filter(pl.col("date") < sessions[cutoff]),
            links,
            sessions[:cutoff],
            isins,
        )
        assert_frame_equal(
            prefix, updated.filter(pl.col("date") < sessions[cutoff]), check_exact=True
        )
    active = np.load(bound_path(rename["arrays"]["active"]), mmap_mode="r")
    ti, ni = np.where(active)
    active_keys = pl.DataFrame(
        {
            "date": np.asarray(sessions, dtype="datetime64[D]")[ti],
            "isin": np.asarray(isins)[ni],
        }
    )
    assert removed.join(active_keys, on=KEYS).is_empty()
    roots = set(added["cnpj"]) | set(removed["cnpj"])
    names = set(updated.filter(pl.col("cnpj").is_in(roots))["isin"]) | set(
        identity.filter(pl.col("cnpj").is_in(roots))["isin"]
    )
    print(
        json.dumps(
            {
                "stage": "identity",
                "added": len(added),
                "removed": len(removed),
                "issuer_roots": len(roots),
            }
        ),
        flush=True,
    )
    family = bound_json(pointer["financial_family"])
    original = bound_json(family["original_family"])
    root = Path(original["source_root"])
    documents, _ = pickle.loads(bound_path(family["extraction"]["cache"]).read_bytes())
    extra, _ = pickle.loads(
        bound_path(previous["artifacts"]["new_issuer_documents"]).read_bytes()
    )
    documents = [
        d for d in [*documents, *extra] if d["cnpj"][:8] in {c[:8] for c in roots}
    ]
    changes = bound_json(family["capital_changes"])
    for change in changes:
        change["effective"] = date.fromisoformat(change["effective"])
    rad = cvm.rad_rows(root)
    relevant = updated.filter(pl.col("cnpj").is_in(roots))
    financial, audit = cvm.fundamental_features(
        copy.deepcopy(documents),
        rad,
        relevant,
        sessions,
        cvm.valuation_market(store, history_links=links.to_dicts()),
        changes,
    )
    old_financial = pl.read_parquet(bound_path(previous["artifacts"]["fundamentals"]))
    financial = pl.concat(
        [old_financial.filter(~pl.col("isin").is_in(names)), financial]
    ).sort(KEYS)
    financial.write_parquet(output / "fundamentals.parquet")
    calendar = json.loads(
        (root / "calendar_2025_announced.json").read_text(encoding="utf8")
    )
    bound_path(calendar["source"])
    for key in ("available_date", "base_through", "through"):
        calendar[key] = date.fromisoformat(calendar[key])
    calendar["full_sessions"] = sessions + [
        date.fromisoformat(d) for d in calendar["sessions"]
    ]
    codes = set(relevant["cvm_code"])
    events = [r for r in rad if r["cvm_code"] in codes and r["group"] != "cadastre"]
    headers = [
        d
        for d in cvm.filing_headers(root, "itr") + cvm.filing_headers(root, "dfp")
        if d["cvm_code"] in codes
    ]
    lags = cvm.header_only_filing_lags(events, headers)
    state = cvm.event_features([*events, *lags], sessions, calendar)
    event_frame = (
        relevant.select("date", "isin", "cvm_code")
        .join(state, on=["date", "cvm_code"], how="left")
        .drop("cvm_code")
    )
    old_events = pl.read_parquet(bound_path(previous["artifacts"]["events"]))
    event_frame = pl.concat(
        [old_events.filter(~pl.col("isin").is_in(names)), event_frame]
    ).sort(KEYS)
    event_frame.write_parquet(output / "events.parquet")
    for old, new in [(old_financial, financial), (old_events, event_frame)]:
        assert_frame_equal(
            old.filter(~pl.col("isin").is_in(names)).sort(KEYS),
            new.filter(~pl.col("isin").is_in(names)).sort(KEYS),
            check_exact=True,
        )
        first = sessions[int(links["effective_index"].min())]
        assert_frame_equal(
            old.filter(pl.col("date") < first).sort(KEYS),
            new.filter(pl.col("date") < first).sort(KEYS),
            check_exact=True,
        )
    deltas = {
        "financial": compare(old_financial, financial, active_keys),
        "events": compare(old_events, event_frame, active_keys),
    }
    write_json_atomic(
        output / "manifest.json",
        {
            "status": "verified_rename_issuer_financial_event_intermediate",
            "inputs": {
                k: run[k]
                for k in [
                    "rename_history_propagation",
                    "fca_identity_admission",
                    "fca_financial_propagation",
                ]
            },
            "identity_rows": len(updated),
            "added_rows": len(added),
            "added_active_rows": len(added.join(active_keys, on=KEYS)),
            "retired_inactive_rows": len(removed),
            "shared_identity_exact": True,
            "unaffected_families_exact": True,
            "future_deletion_prefix_exact": True,
            "affected_issuers": sorted(roots),
            "affected_isins": sorted(names),
            "active_changes": deltas,
            "financial_audit": audit,
            "artifacts": {p.stem: binding(p) for p in output.glob("*.parquet")},
            "code": {p.name: binding(p) for p in [Path(__file__), Path(cvm.__file__)]},
            "seconds": perf_counter() - started,
            "forecast_scoring": False,
            "limits": [
                "No accepted model store or saved fit coordinates changed.",
                "Other auxiliary families and recovered lending require their separately attributed propagation.",
                "Only same-class unit/no-cash source links carry identities; BDI aliases are not inferred.",
            ],
        },
    )
    print(
        json.dumps(
            {"stage": "complete", "seconds": perf_counter() - started, "deltas": deltas}
        ),
        flush=True,
    )


def bound_path(record):
    from brazil_rv.v2.artifacts import sha256_file

    path = Path(record["path"])
    assert sha256_file(path) == record["sha256"]
    return path


if __name__ == "__main__":
    main()
