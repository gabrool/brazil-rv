"""Independently enumerate inherited unit barriers behind new support losses."""

from datetime import datetime, time, timedelta
import json
from pathlib import Path
import pickle
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2 import round5_cvm as cvm
from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main(scaling=False):
    tick = perf_counter()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    admission_key = "scaling_data_identity" if scaling else "surviving_rename_admission"
    a = bound_json(run[admission_key])
    root = Path(a["parent"]["root"])
    work = (
        Path(bound_json(run["scaling_data_workspace"])["root"])
        if scaling
        else Path(a["plan"]["path"]).parent
    )
    out = work / "unit_history"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    ids = pl.read_parquet(work / "issuers/identity.parquet")
    days, names = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    links = pl.read_parquet(
        work / "issuers/financial_history_links.parquet"
        if scaling
        else a["history_mapping"]["path"]
    ).to_dicts()
    by_successor = {r["successor_index"]: r for r in links}
    parent = bound_json(run["lending_feature_propagation"])
    float_inputs = bound_json(parent["source_receipts"]["prior_float_inputs"])
    bases = {
        "financial": root,
        "lending": Path(float_inputs["base_inputs"][0]["path"]).parent,
    }
    market = {}
    columns = [
        names.index(n)
        for n in bound_json(binding(work / "issuers/manifest.json"))["affected_isins"]
    ]
    for kind, base in bases.items():
        if scaling and kind == "lending":
            continue
        observed = np.load(base / "observed.npy", mmap_mode="r")
        dist = np.load(base / "distribution_number.npy", mmap_mode="r")
        splits = np.load(base / "detected_split_mask.npy", mmap_mode="r")
        ambiguous = np.load(base / "ambiguous_action_mask.npy", mmap_mode="r")
        observations = {}
        for col in columns:
            ix = np.flatnonzero(observed[:, col] & np.isfinite(dist[:, col]))
            changes = dict(
                zip(
                    ix[1:][np.diff(dist[ix, col]) != 0],
                    zip(
                        dist[ix[:-1], col][np.diff(dist[ix, col]) != 0],
                        dist[ix[1:], col][np.diff(dist[ix, col]) != 0],
                    ),
                    strict=True,
                )
            )
            rows = np.union1d(
                list(changes), np.flatnonzero(splits[:, col] | ambiguous[:, col])
            ).astype(int)
            observations[col] = [
                (
                    int(t),
                    dict(
                        date=str(days[t]),
                        isin=names[col],
                        distribution_change=list(map(float, changes[t]))
                        if t in changes
                        else None,
                        detected_split=bool(splits[t, col]),
                        ambiguous=bool(ambiguous[t, col]),
                    ),
                )
                for t in rows
            ]
        market[kind] = observations

    def barriers(kind, isin, measured, end, decision):
        first = int(np.searchsorted(days, np.datetime64(measured), side="right"))
        col = names.index(isin)
        findings = []
        # Walk each actual historical date's ancestor, independently of the
        # production prefix-sum implementation.
        for t in range(first, end + 1):
            owner = col
            while owner in by_successor:
                link = by_successor[owner]
                if (
                    max(link["effective_index"], link["known_index"]) > decision
                    or t >= link["effective_index"]
                ):
                    break
                owner = link["predecessor_index"]
            findings.extend(row for j, row in market[kind][owner] if j == t)
        return findings

    identity = {(r["date"], r["isin"]): r for r in ids.iter_rows(named=True)}
    balances = {
        (r["available_date"], r["security_id"].removeprefix("ISIN:")): r
        for r in pl.read_parquet(work / "lending/selected_balances.parquet").iter_rows(
            named=True
        )
    }
    floats = pl.read_parquet(work / "lending/selected_floats.parquet").to_dicts()
    records = []
    for row in (
        []
        if scaling
        else pl.read_parquet(work / "lending/denominator_losses.parquet").iter_rows(
            named=True
        )
    ):
        current, isin = row["date"], row["isin"]
        idrow, loan = identity[current, isin], balances[current, isin]
        candidates = [
            d
            for d in floats
            if d["cnpj"][:8] == idrow["cnpj"][:8]
            and d["cvm_code"] == idrow["cvm_code"]
            and d["class"] == idrow["class"]
            and d["date"] <= current
            and d["snapshot_date"] is not None
            and d["snapshot_date"] <= loan["source_position_date"]
            and d["free_float_shares"] > 0
        ]
        chosen = max(
            candidates,
            key=lambda d: (
                d["snapshot_date"],
                d["reference"],
                d["version"],
                d["date"],
                d["document_id"],
            ),
        )
        hits = barriers(
            "lending",
            isin,
            chosen["snapshot_date"],
            int(np.searchsorted(days, np.datetime64(loan["source_position_date"]))),
            int(np.searchsorted(days, np.datetime64(current))),
        )
        assert hits and any(h["isin"] != isin for h in hits)
        assert np.float32(
            loan["lending_balance_quantity"] / chosen["free_float_shares"]
        ) == np.float32(row["utilization_proxy"])
        records.append(
            dict(
                family="lending",
                date=str(current),
                isin=isin,
                document=str(chosen["document_id"]),
                snapshot=str(chosen["snapshot_date"]),
                known=str(chosen["date"]),
                old_value=row["utilization_proxy"],
                barriers=hits,
            )
        )
    im = bound_json(binding(work / "issuers/manifest.json"))
    family = bound_json(
        json.loads((PROJECT / "docs/v2_data_inputs.json").read_text())[
            "financial_family"
        ]
        if scaling
        else im["source_family"]
    )
    source_root = Path(bound_json(family["original_family"])["source_root"])
    codes = {
        d["cvm_code"] for d in identity.values() if d["isin"] in im["affected_isins"]
    }
    receipts = {
        r["id"]: r["receipt"]
        for r in cvm.rad_rows(source_root)
        if r["cvm_code"] in codes and r["group"] == "structured" and r["id"]
    }
    docs = (
        [
            d
            for d in pickle.loads(
                (work / "issuers/selected_documents.pkl").read_bytes()
            )
            if d.get("shares")
        ]
        if scaling
        else []
    )
    for source in [] if scaling else im["source_caches"]:
        selected, _ = pickle.loads(Path(source["path"]).read_bytes())
        docs.extend(
            d
            for d in selected
            if d["cnpj"][:8] in {c[:8] for c in im["affected_roots"]}
            and d.get("shares")
        )
    for d in docs:
        receipt = receipts.get(d["id"], d["receipt"])
        instant = (
            receipt + timedelta(minutes=1)
            if isinstance(receipt, datetime)
            else datetime.combine(receipt, time.max)
        )
        d["known"] = int(np.searchsorted(days, np.datetime64(instant.date())))
        if (
            d["known"] < len(days)
            and days[d["known"]] == np.datetime64(instant.date())
            and instant.time() > time(15, 45)
        ):
            d["known"] += 1
    before = pl.read_parquet(work / "issuers/control_fundamentals.parquet")
    after = pl.read_parquet(work / "issuers/corrected_fundamentals.parquet")
    lost = before.join(after, on=["date", "isin"], suffix="_after").filter(
        pl.col("log_market_cap").is_not_null()
        & pl.col("log_market_cap_after").is_null()
    )
    for row in lost.iter_rows(named=True):
        current, isin = row["date"], row["isin"]
        decision = int(np.searchsorted(days, np.datetime64(current)))
        idrow = identity[current, isin]
        chosen = max(
            (
                d
                for d in docs
                if d["cnpj"][:8] == idrow["cnpj"][:8]
                and d["cvm_code"] == idrow["cvm_code"]
                and d["known"] <= decision
            ),
            key=lambda d: (d["reference"], d["version"]),
        )
        hits = barriers("financial", isin, chosen["reference"], decision - 1, decision)
        assert hits and any(h["isin"] != isin for h in hits)
        records.append(
            dict(
                family="financial_three_valuation_fields",
                date=str(current),
                isin=isin,
                document=str(chosen["id"]),
                snapshot=str(chosen["reference"]),
                known=str(days[chosen["known"]]),
                old_value=row["log_market_cap"],
                barriers=hits,
            )
        )
    write_json_atomic(out / "records.json", records)
    summary = dict(
        status="qualified_independent_loss_lineage",
        inputs={
            "issuer": binding(work / "issuers/manifest.json"),
            "lending": binding(work / "lending/manifest.json"),
            "admission": run[admission_key],
        },
        records=binding(out / "records.json"),
        lending_rows=sum(r["family"] == "lending" for r in records),
        financial_rows=len(lost),
        distinct_barriers=[
            json.loads(s)
            for s in sorted(
                {json.dumps(h, sort_keys=True) for r in records for h in r["barriers"]}
            )
        ],
        seconds=perf_counter() - tick,
        limits="Existing DISMES/split/ambiguity rows are uncertainty barriers, not proof of a share split or wrong printed capital/float. No source-rate/quantity mutation or scope censoring; every newly lost live valuation/utilization is enumerated. Date selection/arithmetic independently reconstructed; audited original extraction/publication evidence reused.",
    )
    write_json_atomic(out / "manifest.json", summary)
    if scaling:
        summary["reused_lending_qualification"] = run["scaling_data_unit_history"]
        from repair_scaling_inputs import record

        record(run, "scaling_data_financial_units", out / "manifest.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
