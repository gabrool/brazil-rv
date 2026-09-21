"""Propagate accepted own-ISIN lending observations through four dated histories."""

from datetime import date
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from brazil_rv.v2 import round5_b3 as b3, round5_cvm as cvm
from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from propagate_fca_financials import compare
from repair_scaling_inputs import context, record

KEYS = ["date", "isin"]


def main():
    tick = perf_counter()
    run, plan, root, manifest, _ = context()
    admission = bound_json(run["scaling_data_identity"])
    issuer = bound_json(run["scaling_data_issuers"])
    parent = bound_json(run["stage_c_event_lending"])
    prior_issuer = bound_json(run["stage_c_event_issuers"])
    sources = bound_json(run["lending_feature_propagation"])
    names = issuer["affected_isins"]
    dates, isins = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    sessions = dates.astype(object).tolist()
    columns = [isins.index(n) for n in names]
    out = Path(bound_json(run["scaling_data_workspace"])["root"]) / "lending"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    write_json_atomic(
        out / "plan.json",
        dict(
            parent=run["stage_c_event_lending"],
            source_archive=run["lending_feature_propagation"],
            admission=run["scaling_data_identity"],
            issuer=run["scaling_data_issuers"],
            contract="Only accepted own-ISIN balances/rates/flows. Four new same-legal-company histories enter ADV and denominator interval walks at their effect/knowledge dates. No rate/balance record copied, loan alias, changed economic availability or source census. Preserve original Float32 volume precision, source clocks and independent unit barriers. Controls limited to these three newly affected issuers.",
        ),
    )
    old = pl.read_parquet(parent["artifacts"]["features"]["path"])
    balances = pl.read_parquet(sources["artifacts"]["lending_balances"]["path"]).filter(
        pl.col("security_id").is_in(["ISIN:" + n for n in names])
    )
    rates = pl.read_parquet(
        sources["source_receipts"]["recovered_rates"]["path"]
    ).filter(pl.col("security_id").is_in(["ISIN:" + n for n in names]))
    fi = bound_json(sources["source_receipts"]["prior_float_inputs"])
    roots8 = {c[:8] for c in issuer["affected_roots"]}
    floats = pl.read_parquet(
        fi["new_cvm_files"]["free_float_observations.parquet"]["path"]
    ).filter(pl.col("cnpj").str.slice(0, 8).is_in(roots8))
    changes = [
        c
        for c in bound_json(fi["new_cvm_files"]["capital_change_observations.json"])
        if c["cnpj"][:8] in roots8
    ]
    for c in changes:
        c["effective"] = date.fromisoformat(c["effective"])
    for label, frame in (
        ("selected_balances", balances),
        ("selected_rates", rates),
        ("selected_floats", floats),
    ):
        frame.write_parquet(out / f"{label}.parquet")
    volume_root = Path(sources["source_receipts"]["base_manifest"]["path"]).parent
    volume = np.load(volume_root / "volume_brl.npy", mmap_mode="r")
    market = cvm.valuation_market(Path(fi["base_inputs"][0]["path"]).parent)
    old_links = pl.read_parquet(root / manifest["tables"]["slow_history_links"]["path"])
    new_links = pl.read_parquet(admission["history_mapping"]["path"])
    old_units = pl.read_parquet(
        prior_issuer["artifacts"]["financial_history_links"]["path"]
    )
    new_units = pl.read_parquet(issuer["artifacts"]["financial_history_links"]["path"])
    before_id = pl.read_parquet(root / manifest["tables"]["issuer_identity"]["path"])
    after_id = pl.read_parquet(issuer["artifacts"]["identity"]["path"])
    active = np.load(root / manifest["arrays"]["active"]["path"], mmap_mode="r")
    corrected = active.copy()
    with np.load(admission["deltas"]["path"]) as z:
        corrected[tuple(z["active__indices"].T)] = z["active__values"]

    def keys(mask):
        ti, ni = np.where(mask)
        return pl.DataFrame(dict(date=dates[ti], isin=np.array(isins)[ni]))

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

    outputs, audits = [], []
    for label, links, units, ids, membership in (
        ("control", old_links, old_units, before_id, active),
        ("corrected", new_links, new_units, after_id, corrected),
    ):
        v = volume.copy()
        for link in links.sort("effective_index").iter_rows(named=True):
            p, n, e = (
                link[k]
                for k in ("predecessor_index", "successor_index", "effective_index")
            )
            v[:e, n] = v[:e, p]
        features = b3.lending_decision_features(
            balances, rates, sessions, names, np.ascontiguousarray(v[:, columns])
        )
        market["history_links"] = units.to_dicts()
        util, audit = b3.lending_utilization_features(
            balances,
            floats,
            ids.filter(pl.col("cnpj").is_in(issuer["affected_roots"])),
            sessions,
            market,
            changes,
        )
        features = (
            typed(
                features.drop(
                    "utilization_proxy", "utilization_proxy_age_sessions"
                ).join(util, on=KEYS, how="full", coalesce=True)
            )
            .join(keys(membership), on=KEYS)
            .sort(KEYS)
        )
        features.write_parquet(out / f"{label}.parquet")
        outputs.append(features)
        audits.append(audit)
    expected = typed(old.filter(pl.col("isin").is_in(names)))
    assert_frame_equal(outputs[0], expected, check_exact=True)
    starts = {
        e[k]: date.fromisoformat(e["effective_date"])
        for e in plan["history"]
        for k in ("isin", "successor_isin")
    }
    boundary = pl.DataFrame(dict(isin=list(starts), start=list(starts.values())))
    unchanged = (
        expected.join(boundary, on="isin")
        .filter(pl.col("date") < pl.col("start"))
        .drop("start")
    )
    changed = (
        outputs[1]
        .join(boundary, on="isin")
        .filter(pl.col("date") >= pl.col("start"))
        .drop("start")
    )
    # Other share classes of these same issuers retain their already accepted rows.
    affected = pl.concat(
        [expected.filter(~pl.col("isin").is_in(starts)), unchanged, changed]
    ).sort(KEYS)
    source_fields = [
        c
        for c in old.columns
        if not c.startswith(("loan_balance_", "utilization_proxy"))
    ]
    shared = expected.select(KEYS).join(affected.select(KEYS), on=KEYS)
    assert_frame_equal(
        expected.join(shared, on=KEYS).select(source_fields),
        affected.join(shared, on=KEYS).select(source_fields),
        check_exact=True,
    )
    final = pl.concat([old.filter(~pl.col("isin").is_in(names)), affected]).sort(KEYS)
    final.write_parquet(out / "features.parquet")
    joined = expected.join(
        affected, on=KEYS, how="full", coalesce=True, suffix="_after"
    ).join(keys(corrected), on=KEYS)
    losses = joined.filter(
        pl.col("utilization_proxy").is_not_null()
        & pl.col("utilization_proxy_after").is_null()
    )
    losses.write_parquet(out / "denominator_losses.parquet")
    report = dict(
        status="own_source_lending_dependencies_qualified",
        plan=binding(out / "plan.json"),
        parent=plan["parent"],
        source_rows=dict(
            balances=balances.height, rates=rates.height, floats=floats.height
        ),
        controls=dict(
            cells=expected.height * (expected.width - 2),
            exact=True,
            shared_rates_and_flows_exact=True,
        ),
        active_changes=compare(expected, affected, keys(corrected)),
        denominator_audits=audits,
        artifacts={p.stem: binding(p) for p in out.glob("*.parquet")},
        seconds=perf_counter() - tick,
        limits="Economic rate/availability archive unchanged. Enumerated denominator losses require selected-source interval qualification; no unchanged historical source campaign is repeated.",
    )
    record(run, "scaling_data_lending", out / "manifest.json", report)
    print(
        json.dumps(
            {k: report[k] for k in ("status", "source_rows", "controls", "seconds")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["units"]:
        from propagate_matched_event_lending import qualify_unit_losses

        qualify_unit_losses(
            "scaling_data_lending", "scaling_data_issuers", "scaling_data_unit_history"
        )
    else:
        main()
