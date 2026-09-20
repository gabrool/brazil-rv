"""Bound existing lending formulas to the surviving issuers and spot histories."""

from datetime import date
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from brazil_rv.v2 import round5_b3 as b3, round5_cvm as cvm
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
    parent = bound_json(run["lending_feature_propagation"])
    work = Path(admission["plan"]["path"]).parent
    issuer = bound_json(binding(work / "issuers/manifest.json"))
    names = issuer["affected_isins"]
    store = Path(admission["parent"]["root"])
    dates, isins = (
        np.load(store / "date_index.npy"),
        np.load(store / "isin_index.npy").tolist(),
    )
    assert str(dates[-1]) == "2024-12-30"
    sessions = dates.astype(object).tolist()
    columns = [isins.index(n) for n in names]
    out = work / "lending"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    write_json_atomic(
        out / "plan.json",
        {
            "parent": run["lending_feature_propagation"],
            "identity": binding(work / "issuers/manifest.json"),
            "admission": run["surviving_rename_admission"],
            "names": names,
            "contrast": "Only bounded surviving-issuer own-ISIN loan rows: prior20 spot ADV inherits admitted same-class history; free-float denominator preserves causal issuer/class/measurement and predecessor unit barriers. Do not copy balances, rates, flow, locates or contracts across identifiers. Reuse qualified source union and own-version float extractions.",
            "verification": "Exact old-coordinate controls on affected existing active rows, source-field equality, pre-effect/unaffected output equality; enumerate any denominator support loss. No source PDF or full source census, no model/account run.",
        },
    )
    old = pl.read_parquet(bound_path(parent["artifacts"]["features"]))
    balances = pl.read_parquet(
        bound_path(parent["artifacts"]["lending_balances"])
    ).filter(pl.col("security_id").is_in(["ISIN:" + n for n in names]))
    rates = pl.read_parquet(
        bound_path(parent["source_receipts"]["recovered_rates"])
    ).filter(pl.col("security_id").is_in(["ISIN:" + n for n in names]))
    balances.write_parquet(out / "selected_balances.parquet")
    rates.write_parquet(out / "selected_rates.parquet")
    fi = bound_json(parent["source_receipts"]["prior_float_inputs"])
    floats = pl.read_parquet(
        bound_path(fi["new_cvm_files"]["free_float_observations.parquet"])
    ).filter(
        pl.col("cnpj").str.slice(0, 8).is_in([c[:8] for c in issuer["affected_roots"]])
    )
    floats.write_parquet(out / "selected_floats.parquet")
    changes = [
        r
        for r in bound_json(fi["new_cvm_files"]["capital_change_observations.json"])
        if r["cnpj"][:8] in {c[:8] for c in issuer["affected_roots"]}
    ]
    for row in changes:
        row["effective"] = date.fromisoformat(row["effective"])
    base = Path(parent["source_receipts"]["base_manifest"]["path"]).parent
    volume = np.load(base / "volume_brl.npy", mmap_mode="r")
    old_links = pl.read_parquet(store / "slow_history_links.parquet")
    new_links = pl.read_parquet(admission["history_mapping"]["path"])
    old_id = bound_json(run["rename_issuer_propagation"])["artifacts"]["identity"]
    active_before = np.load(store / "active.npy", mmap_mode="r")
    active_after = active_before.copy()
    with np.load(admission["liquidity_deltas"]["path"]) as z:
        active_after[tuple(z["active__indices"].T)] = z["active__values"]

    def keys(mask):
        ti, ni = np.where(mask)
        return pl.DataFrame({"date": dates[ti], "isin": np.asarray(isins)[ni]})

    def typed(frame):
        return (
            frame.select(old.columns)
            .cast(old.schema)
            .with_columns(
                pl.col(c).fill_null(-1)
                for c in old.columns
                if c.endswith("_age_sessions")
            )
            .sort(KEYS)
        )

    market = cvm.valuation_market(Path(fi["base_inputs"][0]["path"]).parent)
    outputs, audits = [], []
    for label, links, ids, active in [
        ("control", old_links, old_id, active_before),
        ("corrected", new_links, issuer["artifacts"]["identity"], active_after),
    ]:
        v = volume.copy()
        for link in links.sort("effective_index").iter_rows(named=True):
            p, n, e = (
                link[k]
                for k in ["predecessor_index", "successor_index", "effective_index"]
            )
            v[:e, n] = v[:e, p]
        features = b3.lending_decision_features(
            balances, rates, sessions, names, np.ascontiguousarray(v[:, columns])
        )
        identity = pl.read_parquet(ids["path"]).filter(
            pl.col("cnpj").is_in(issuer["affected_roots"])
        )
        market["history_links"] = links.to_dicts()
        util, audit = b3.lending_utilization_features(
            balances, floats, identity, sessions, market, changes
        )
        features = typed(
            features.drop("utilization_proxy", "utilization_proxy_age_sessions").join(
                util, on=KEYS, how="full", coalesce=True
            )
        )
        features = features.join(keys(active), on=KEYS).sort(KEYS)
        features.write_parquet(out / f"{label}.parquet")
        outputs.append(features)
        audits.append(audit)
    expected = typed(old.filter(pl.col("isin").is_in(names)))
    assert_frame_equal(outputs[0], expected, check_exact=True)
    unchanged_sources = [
        c
        for c in old.columns
        if not c.startswith(("loan_balance_", "utilization_proxy"))
    ]
    shared = outputs[0].select(KEYS).join(outputs[1].select(KEYS), on=KEYS)
    assert_frame_equal(
        outputs[0].join(shared, on=KEYS).select(unchanged_sources),
        outputs[1].join(shared, on=KEYS).select(unchanged_sources),
        check_exact=True,
    )
    final = pl.concat([old.filter(~pl.col("isin").is_in(names)), outputs[1]]).sort(KEYS)
    assert_frame_equal(
        old.filter(pl.col("date") < date(2019, 8, 6)).sort(KEYS),
        final.filter(pl.col("date") < date(2019, 8, 6)).sort(KEYS),
        check_exact=True,
    )
    final.write_parquet(out / "features.parquet")
    joined = expected.join(
        outputs[1], on=KEYS, how="full", coalesce=True, suffix="_after"
    ).join(keys(active_after), on=KEYS)
    losses = joined.filter(
        pl.col("utilization_proxy").is_not_null()
        & pl.col("utilization_proxy_after").is_null()
    )
    losses.write_parquet(out / "denominator_losses.parquet")
    stats = compare(expected, outputs[1], keys(active_after))
    report = dict(
        status="qualified_surviving_lending_intermediate",
        plan=binding(out / "plan.json"),
        source_rows={
            "balances": len(balances),
            "rates": len(rates),
            "floats": len(floats),
        },
        controls={
            "rows": len(expected),
            "cells": len(expected) * (expected.width - 2),
            "exact": True,
            "rate_flow_shared_exact": True,
            "pre_effect_exact": True,
        },
        active_changes=stats,
        denominator_audits=audits,
        artifacts={p.stem: binding(p) for p in out.glob("*.parquet")},
        seconds=perf_counter() - tick,
        limits="No new loan alias/rate/balance union. Published rate/availability economic archive remains byte-identical and is not replayed. This is a sparse dependency layer, not full-store/model admission.",
    )
    write_json_atomic(out / "manifest.json", report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "status",
                    "source_rows",
                    "controls",
                    "active_changes",
                    "seconds",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
