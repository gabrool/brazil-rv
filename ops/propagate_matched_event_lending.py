"""Use accepted lending rows with dated market histories and own issuer units."""

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

PROJECT = Path(__file__).resolve().parents[1]
KEYS = ["date", "isin"]


def qualify_unit_losses():
    """Walk each selected measurement interval independently of prefix sums."""
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    source = bound_json(run["stage_c_event_lending"])
    issuer = bound_json(run["stage_c_event_issuers"])
    root = Path(source["parent"]["root"])
    work = Path(source["plan"]["path"]).parent
    out = work / "unit_qualification"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    days, names = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    ids = {
        (r["date"], r["isin"]): r
        for r in pl.read_parquet(issuer["artifacts"]["identity"]["path"])
        .filter(pl.col("isin").is_in(issuer["affected_isins"]))
        .iter_rows(named=True)
    }
    links = {
        r["successor_index"]: r
        for r in pl.read_parquet(
            issuer["artifacts"]["financial_history_links"]["path"]
        ).iter_rows(named=True)
    }
    original = bound_json(run["lending_feature_propagation"])
    f = bound_json(original["source_receipts"]["prior_float_inputs"])
    base = Path(f["base_inputs"][0]["path"]).parent
    seen, dist, split, ambiguous = (
        np.load(base / (k + ".npy"), mmap_mode="r")
        for k in (
            "observed",
            "distribution_number",
            "detected_split_mask",
            "ambiguous_action_mask",
        )
    )
    barriers = {}
    for isin in issuer["affected_isins"]:
        j = names.index(isin)
        positions = np.flatnonzero(seen[:, j] & np.isfinite(dist[:, j]))
        changed = positions[1:][np.diff(dist[positions, j]) != 0]
        flags = {
            int(t): dict(
                date=str(days[t]),
                isin=isin,
                distribution_change=[
                    float(dist[positions[np.searchsorted(positions, t) - 1], j]),
                    float(dist[t, j]),
                ],
                detected_split=bool(split[t, j]),
                ambiguous=bool(ambiguous[t, j]),
            )
            for t in changed
        }
        for t in np.flatnonzero(split[:, j] | ambiguous[:, j]):
            flags.setdefault(
                int(t),
                dict(
                    date=str(days[t]),
                    isin=isin,
                    distribution_change=None,
                    detected_split=bool(split[t, j]),
                    ambiguous=bool(ambiguous[t, j]),
                ),
            )
        barriers[j] = flags
    loans = {
        (r["available_date"], r["security_id"].removeprefix("ISIN:")): r
        for r in pl.read_parquet(
            source["artifacts"]["selected_balances"]["path"]
        ).iter_rows(named=True)
    }
    floats = pl.read_parquet(source["artifacts"]["selected_floats"]["path"]).to_dicts()
    records = []
    for row in pl.read_parquet(
        source["artifacts"]["denominator_losses"]["path"]
    ).iter_rows(named=True):
        current, isin = row["date"], row["isin"]
        legal, loan = ids[current, isin], loans[current, isin]
        selected = max(
            (
                d
                for d in floats
                if d["cnpj"][:8] == legal["cnpj"][:8]
                and d["cvm_code"] == legal["cvm_code"]
                and d["class"] == legal["class"]
                and d["date"] <= current
                and d["snapshot_date"] is not None
                and d["snapshot_date"] <= loan["source_position_date"]
                and d["free_float_shares"] > 0
            ),
            key=lambda d: (
                d["snapshot_date"],
                d["reference"],
                d["version"],
                d["date"],
                d["document_id"],
            ),
        )
        first = int(
            np.searchsorted(
                days, np.datetime64(selected["snapshot_date"]), side="right"
            )
        )
        last = int(np.searchsorted(days, np.datetime64(loan["source_position_date"])))
        decision = int(np.searchsorted(days, np.datetime64(current)))
        hits = []
        for t in range(first, last + 1):
            owner = names.index(isin)
            while owner in links:
                link = links[owner]
                if (
                    max(link["known_index"], link["effective_index"]) > decision
                    or t >= link["effective_index"]
                ):
                    break
                owner = link["predecessor_index"]
            if t in barriers[owner]:
                hits.append(barriers[owner][t])
        assert hits and any(h["isin"] != isin for h in hits)
        assert np.float32(
            loan["lending_balance_quantity"] / selected["free_float_shares"]
        ) == np.float32(row["utilization_proxy"])
        records.append(
            dict(
                date=str(current),
                isin=isin,
                document=str(selected["document_id"]),
                snapshot=str(selected["snapshot_date"]),
                known=str(selected["date"]),
                position_date=str(loan["source_position_date"]),
                balance_quantity=loan["lending_balance_quantity"],
                free_float_shares=selected["free_float_shares"],
                prior_ratio=row["utilization_proxy"],
                barriers=hits,
            )
        )
    write_json_atomic(out / "records.json", records)
    report = dict(
        status="qualified_independent_selected_unit_intervals",
        producer=run["stage_c_event_lending"],
        issuer=run["stage_c_event_issuers"],
        records=binding(out / "records.json"),
        losses=len(records),
        distinct_barriers=[
            json.loads(s)
            for s in sorted(
                {json.dumps(h, sort_keys=True) for r in records for h in r["barriers"]}
            )
        ],
        seconds=perf_counter() - tick,
        limits="Existing printed DISMES and ambiguity/split flags are uncertainty barriers, not proof of a split or an erroneous source denominator. Raw quantities/rates unchanged; no new source extraction. Same-legal links only, never TIM/SIMPAR predecessor capital units.",
    )
    write_json_atomic(out / "report.json", report)
    run["stage_c_event_unit_history"] = binding(out / "report.json")
    write_json_atomic(pointer, run)
    print(json.dumps(report), flush=True)


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    admission = bound_json(run["stage_c_event_data_admission"])
    admission_plan = bound_json(admission["plan"])
    issuer = bound_json(run["stage_c_event_issuers"])
    parent = bound_json(run["surviving_rename_lending"])
    sources = bound_json(run["lending_feature_propagation"])
    names = issuer["affected_isins"]
    root = Path(admission["parent"]["root"])
    dates, isins = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    sessions = dates.astype(object).tolist()
    columns = [isins.index(n) for n in names]
    out = Path(admission["plan"]["path"]).parent / "lending"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    write_json_atomic(
        out / "plan.json",
        dict(
            parent=run["surviving_rename_lending"],
            source_archive=run["lending_feature_propagation"],
            admission=run["stage_c_event_data_admission"],
            issuer=run["stage_c_event_issuers"],
            contract="Only existing own-ISIN balances/rates/flows; no alias or copied lending record. ADV inherits five market paths; free-float units follow only same-legal issuer paths. Old-JSL public feature rows remain unchanged before its sourced holding retirement, and new-logistics histories apply only from reopening. Original Float32 volumes/source clocks/barriers/support. New controls for nine newly affected names, no earlier source census or old passed campaign.",
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
    old_links = pl.read_parquet(root / "slow_history_links.parquet")
    new_links = pl.read_parquet(admission["history_mapping"]["path"])
    legal_links = pl.read_parquet(
        issuer["artifacts"]["financial_history_links"]["path"]
    )
    before_id = pl.read_parquet(root / "issuer_identity.parquet")
    after_id = pl.read_parquet(issuer["artifacts"]["identity"]["path"])
    active = np.load(root / "active.npy", mmap_mode="r")
    corrected_active = active.copy()
    with np.load(admission["deltas"]["path"]) as z:
        corrected_active[tuple(z["active__indices"].T)] = z["active__values"]

    def keys(mask):
        ti, ni = np.where(mask)
        return pl.DataFrame({"date": dates[ti], "isin": np.array(isins)[ni]})

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
        ("control", old_links, old_links, before_id, active),
        ("corrected", new_links, legal_links, after_id, corrected_active),
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
    starts = {}
    for e in admission_plan["events"]:
        for k in ("isin", "successor_isin"):
            starts[e[k]] = min(starts.get(e[k], "9999-12-31"), e["effective_date"])
    boundaries = pl.DataFrame(
        {
            "isin": list(starts),
            "start": [date.fromisoformat(d) for d in starts.values()],
        }
    )
    unchanged = (
        expected.join(boundaries, on="isin")
        .filter(pl.col("date") < pl.col("start"))
        .drop("start")
    )
    changed = (
        outputs[1]
        .join(boundaries, on="isin")
        .filter(pl.col("date") >= pl.col("start"))
        .drop("start")
    )
    affected = pl.concat([unchanged, changed]).sort(KEYS)
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
    ).join(keys(corrected_active), on=KEYS)
    losses = joined.filter(
        pl.col("utilization_proxy").is_not_null()
        & pl.col("utilization_proxy_after").is_null()
    )
    losses.write_parquet(out / "denominator_losses.parquet")
    report = dict(
        status="qualified_own_source_lending_intermediate",
        plan=binding(out / "plan.json"),
        parent=admission["parent"],
        source_rows=dict(
            balances=balances.height, rates=rates.height, floats=floats.height
        ),
        controls=dict(
            cells=expected.height * (expected.width - 2),
            exact=True,
            shared_rates_and_flows_exact=True,
        ),
        active_changes=compare(expected, affected, keys(corrected_active)),
        denominator_audits=audits,
        artifacts={p.stem: binding(p) for p in out.glob("*.parquet")},
        seconds=perf_counter() - tick,
        limits="Economic rate/availability archive unchanged. Denominator losses need independent selected measurement/date/unit walks before combined admission; no support threshold relaxed. Old fits/stores untouched.",
    )
    write_json_atomic(out / "manifest.json", report)
    run = json.loads(pointer.read_text())
    run["stage_c_event_lending"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "source_rows",
                    "controls",
                    "active_changes",
                    "seconds",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["--qualify-units"]:
        qualify_unit_losses()
    else:
        main()
