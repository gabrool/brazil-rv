"""Own-legal-issuer identities and cached financial/event dependencies."""

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

PROJECT = Path(__file__).resolve().parents[1]
KEYS = ["date", "isin"]


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    admission = bound_json(run["stage_c_event_data_admission"])
    plan = bound_json(admission["plan"])
    sources = bound_json(run["stage_c_event_source_admission"])
    prior = bound_json(run["surviving_rename_issuers"])
    root = Path(admission["parent"]["root"])
    m = bound_json(
        dict(
            path=str(root / "manifest.json"),
            sha256=admission["parent"]["manifest_sha256"],
        )
    )
    dates, names = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    sessions = dates.astype(object).tolist()
    identity = pl.read_parquet(root / m["tables"]["issuer_identity"]["path"])
    out = Path(admission["plan"]["path"]).parent / "issuers"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    (out / "executed_cvm.py").write_bytes(Path(cvm.__file__).read_bytes())
    history = pl.read_parquet(admission["history_mapping"]["path"])
    old_history = pl.read_parquet(root / m["tables"]["slow_history_links"]["path"])
    separate = {"BRJSLGACNOR2", "BRTIMPACNOR1"}
    same_legal = history.filter(
        ~pl.col("predecessor_index").is_in([names.index(s) for s in separate])
    )
    new_same = same_legal.join(
        old_history.select("predecessor_index", "successor_index"),
        on=["predecessor_index", "successor_index"],
        how="anti",
    )
    assert new_same.height == 3
    write_json_atomic(
        out / "plan.json",
        dict(
            parent=admission["parent"],
            admission=run["stage_c_event_data_admission"],
            sources=run["stage_c_event_source_admission"],
            prior=run["surviving_rename_issuers"],
            contract="Three same-legal-company links extend their own identity. SIMPAR/TIM S.A. receive separately sourced legal CNPJ/class identities with unknown sector until their own dated metadata; never inherit predecessor filings or capital denominators. Retire old JSL only until its separately sourced logistics reopening. Recompute six affected legal issuers from audited own-version caches only; old-coordinate controls and unchanged pre-effect rows. No source extraction or acquired-issuer pooling.",
            financial_history_excludes=sorted(separate),
            registration=binding(
                PROJECT / "research/preregistrations/v2_economic_data_scaling.md"
            ),
        ),
    )
    updated = cvm.continue_issuer_identity(identity, new_same, sessions, names)
    seeds = {
        "BRSIMHACNOR0": ("07415333000120", "025003", "793185"),
        "BRTIMSACNOR5": ("02421421000111", "024929", "797479"),
    }
    for e in plan["events"]:
        if e["isin"] not in separate:
            continue
        begin = date.fromisoformat(e["effective_date"])
        cnpj, code, protocol = seeds[e["successor_isin"]]
        known = date.fromisoformat(sources["originals"][protocol]["available_date"])
        assert known <= begin
        if protocol == "797479":
            receipt = bound_json(sources["originals"][protocol]["receipt"])
            assert any(
                r["cvm"].replace("-", "") == code for r in receipt["issuer_rows"]
            )
        text = Path(sources["originals"][protocol]["text"]["path"]).read_text(
            encoding="utf8"
        )
        digits = "".join(c for c in text if c.isdigit())
        assert cnpj in digits
        existing = {
            r["date"]: r
            for r in updated.filter(pl.col("isin") == e["successor_isin"]).iter_rows(
                named=True
            )
        }
        inherited = dict.fromkeys(identity.columns)
        inherited.update(
            isin=e["successor_isin"],
            cnpj=cnpj,
            cvm_code=code,
            sector=None,
            sector_label=None,
            sector_known_date=None,
            sector_mapping_id=None,
            **{"class": "ON"},
            preferred_class="",
            unit_composition=None,
            fca_id=None,
            identity_known_date=known,
            identity_method="original_corporate_legal_identity:" + protocol,
            identity_effective_start=begin,
        )
        added = []
        for current in sessions:
            if current < begin:
                continue
            if current in existing:
                row = existing[current]
                assert (
                    row["cnpj"] == cnpj
                    and row["cvm_code"] == code
                    and row["class"] == "ON"
                )
                inherited = {**row}
            else:
                added.append({**inherited, "date": current})
        if added:
            updated = pl.concat([updated, pl.DataFrame(added, schema=identity.schema)])
        retired = (pl.col("isin") == e["isin"]) & (pl.col("date") >= begin)
        if e["source_reopens_date"]:
            retired &= pl.col("date") < date.fromisoformat(e["source_reopens_date"])
        updated = updated.filter(~retired)
    updated = updated.sort(KEYS)
    added = updated.join(identity.select(KEYS), on=KEYS, how="anti")
    removed = identity.join(updated.select(KEYS), on=KEYS, how="anti")
    shared = identity.select(KEYS).join(updated.select(KEYS), on=KEYS)
    assert_frame_equal(
        identity.join(shared, on=KEYS).sort(KEYS),
        updated.join(shared, on=KEYS).sort(KEYS),
        check_exact=True,
    )
    affected_roots = sorted(set(added["cnpj"]) | set(removed["cnpj"]))
    affected_names = sorted(
        set(identity.filter(pl.col("cnpj").is_in(affected_roots))["isin"])
        | set(updated.filter(pl.col("cnpj").is_in(affected_roots))["isin"])
    )
    for label, frame in (
        ("identity", updated),
        ("added", added),
        ("removed", removed),
        ("financial_history_links", same_legal),
    ):
        frame.write_parquet(out / f"{label}.parquet")
    active = np.load(root / m["arrays"]["active"]["path"]).copy()
    with np.load(admission["deltas"]["path"]) as z:
        active[tuple(z["active__indices"].T)] = z["active__values"]
    ti, ni = np.where(active)
    active_keys = pl.DataFrame({"date": dates[ti], "isin": np.array(names)[ni]})
    assert removed.join(active_keys, on=KEYS).is_empty()
    print(
        json.dumps(
            dict(
                stage="identities",
                added=added.height,
                removed=removed.height,
                roots=affected_roots,
                names=affected_names,
            )
        ),
        flush=True,
    )
    foundation = json.loads((PROJECT / "docs/v2_data_inputs.json").read_text())
    family = bound_json(foundation["financial_family"])
    fca = bound_json(run["fca_financial_propagation"])
    source_root = Path(bound_json(family["original_family"])["source_root"])
    documents, _ = pickle.loads(
        Path(family["extraction"]["cache"]["path"]).read_bytes()
    )
    extra, _ = pickle.loads(
        Path(fca["artifacts"]["new_issuer_documents"]["path"]).read_bytes()
    )
    roots8 = {c[:8] for c in affected_roots}
    documents = [d for d in [*documents, *extra] if d["cnpj"][:8] in roots8]
    codes = set(updated.filter(pl.col("cnpj").is_in(affected_roots))["cvm_code"])
    rad = [r for r in cvm.rad_rows(source_root) if r["cvm_code"] in codes]
    changes = [
        r for r in bound_json(family["capital_changes"]) if r["cnpj"][:8] in roots8
    ]
    for r in changes:
        r["effective"] = date.fromisoformat(r["effective"])
    with (out / "selected_documents.pkl").open("wb") as stream:
        pickle.dump(documents, stream)
    old_financial = pl.read_parquet(prior["artifacts"]["fundamentals"]["path"])
    old_events = pl.read_parquet(prior["artifacts"]["events"]["path"])
    market = cvm.valuation_market(root, history_links=old_history.to_dicts())
    outputs, audits = [], []
    for label, ids, links in (
        ("control", identity, old_history),
        ("corrected", updated, same_legal),
    ):
        market["history_links"] = links.to_dicts()
        frame, audit = cvm.fundamental_features(
            copy.deepcopy(documents),
            rad,
            ids.filter(pl.col("cnpj").is_in(affected_roots)),
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
    expected = old_financial.filter(pl.col("isin").is_in(affected_names)).sort(KEYS)
    assert_frame_equal(outputs[0].sort(KEYS), expected, check_exact=True)
    financial = pl.concat(
        [old_financial.filter(~pl.col("isin").is_in(affected_names)), outputs[1]]
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
            ids.filter(pl.col("cnpj").is_in(affected_roots))
            .select("date", "isin", "cvm_code")
            .join(state, on=["date", "cvm_code"], how="left")
            .drop("cvm_code")
            .sort(KEYS)
        )

    assert_frame_equal(
        event_rows(identity),
        old_events.filter(pl.col("isin").is_in(affected_names)).sort(KEYS),
        check_exact=True,
    )
    event_frame = pl.concat(
        [old_events.filter(~pl.col("isin").is_in(affected_names)), event_rows(updated)]
    ).sort(KEYS)
    event_frame.write_parquet(out / "events.parquet")
    prefix = pl.col("date") < date(2020, 9, 18)
    for before, after in (
        (identity, updated),
        (old_financial, financial),
        (old_events, event_frame),
    ):
        assert_frame_equal(
            before.filter(prefix).sort(KEYS),
            after.filter(prefix).sort(KEYS),
            check_exact=True,
        )
    report = dict(
        status="qualified_own_issuer_intermediate",
        plan=binding(out / "plan.json"),
        parent=admission["parent"],
        affected_roots=affected_roots,
        affected_isins=affected_names,
        added=added.height,
        removed=removed.height,
        added_active=added.join(active_keys, on=KEYS).height,
        controls=dict(
            financial_cells=expected.height * (expected.width - 2),
            event_cells=event_rows(identity).height * (event_frame.width - 2),
            exact=True,
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
        limits="Own-version cached filings only; SIMPAR/TIM legal predecessors do not supply issuer fundamentals, sectors or capital-unit histories. Source clocks and unsupported early sector/denominator fields stay explicit. Market continuation and lending rates remain separate. No accepted store/fit mutation.",
    )
    write_json_atomic(out / "manifest.json", report)
    run = json.loads(pointer.read_text())
    run["stage_c_event_issuers"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "controls",
                    "added",
                    "removed",
                    "added_active",
                    "seconds",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
