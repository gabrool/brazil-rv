"""Propagate four auxiliary families over five admitted market episodes."""

from datetime import date
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.auxiliary_history import (
    renamed_activity_features,
    renamed_oddlot_features,
)
from brazil_rv.v2.build_store import _prior_adv20
from brazil_rv.v2.data_repair import binding, bound_json, source_families
from brazil_rv.v2.round5_b3 import activity_decision_features
from brazil_rv.v2.round5_index import pressure_panel
from brazil_rv.v2.sidecars import derive_known_archive_features
from propagate_remaining_auxiliaries import exact_nullable, odd_family, replace

PROJECT = Path(__file__).resolve().parents[1]


def main():
    tick = perf_counter()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    admission = bound_json(run["stage_c_event_data_admission"])
    parent = Path(admission["parent"]["root"])
    manifest = bound_json(
        dict(
            path=str(parent / "manifest.json"),
            sha256=admission["parent"]["manifest_sha256"],
        )
    )
    prior = bound_json(run["surviving_rename_auxiliaries"])
    families = source_families(manifest)
    dates, isins = (
        np.load(parent / "date_index.npy"),
        np.load(parent / "isin_index.npy").tolist(),
    )
    sessions = dates.astype(object).tolist()
    old_links = pl.read_parquet(parent / "slow_history_links.parquet").to_dicts()
    links = pl.read_parquet(admission["history_mapping"]["path"]).to_dicts()
    events = bound_json(admission["plan"])["events"]
    new_pairs = {(e["isin"], e["successor_isin"]) for e in events}
    new_links = [
        x
        for x in links
        if (isins[x["predecessor_index"]], isins[x["successor_index"]]) in new_pairs
    ]
    for link in links:
        if isins[link["predecessor_index"]] == "BRJSLGACNOR2":
            link["source_reopens_index"] = sessions.index(date(2020, 11, 11))
    links.sort(key=lambda x: x["effective_index"])
    new_links.sort(key=lambda x: x["effective_index"])
    out = Path(admission["plan"]["path"]).parent / "auxiliaries"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    (out / "executed_index.py").write_bytes(
        (PROJECT / "research/src/brazil_rv/v2/round5_index.py").read_bytes()
    )
    write_json_atomic(
        out / "plan.json",
        dict(
            parent=admission["parent"],
            admission=run["stage_c_event_data_admission"],
            prior=run["surviving_rename_auxiliaries"],
            contrast="Five admitted market episodes: bounded22-prior/22-following activity/options and six-prior oddlot source; unchanged support/clocks. Index uses BOTH dated composition identities, originalADV20 and JSL source reopening by snapshot date; no option-series/loan alias or source census. Reuse accepted source normalizations and parent families; final combined typed/consumer verification follows.",
            verification="Save source subsets and control/corrected outputs once; exact old-family controls, future-source deletion prefixes, independent arithmetic and actual new-family consumers before full-store admission.",
            registration=binding(
                PROJECT / "research/preregistrations/v2_economic_data_scaling.md"
            ),
        ),
    )
    old_active = np.load(parent / "active.npy")
    active = old_active.copy()
    with np.load(admission["deltas"]["path"]) as z:
        active[tuple(z["active__indices"].T)] = z["active__values"]
    final = {
        k: pl.read_parquet(prior["artifacts"][k]["path"])
        for k in ("microstructure", "options", "rebalance")
    }
    parents = dict(final)
    am = bound_json(families["microstructure"]["source_manifest"])
    root = Path(am["builder"]["path"]).parent
    inventory = bound_json(am["source_inventory"])
    inputs = {
        Path(r["path"]).parent.name + "/" + Path(r["path"]).name: r
        for r in am["inputs"]
    }
    cm = bound_json(inputs["b3_cash_axis/manifest.json"])
    cash_rec = dict(
        path=str(root / "b3_cash_axis/cash.parquet"),
        sha256=cm["artifacts"]["cash.parquet"]["sha256"],
    )
    qm = bound_json(inputs["cotahist_option_volume/manifest.json"])
    odd_rec = next(
        r
        for r in manifest["sources"]
        if Path(r.get("path", "")).name == "odd_lot_activity.parquet"
    )
    receipts, scopes, odd_before, odd_after = {}, [], [], []

    def selected(rec, predicate):
        # Prior accepted source receipts bind these normalized files. No new census.
        receipts[rec["path"]] = rec
        return pl.scan_parquet(rec["path"]).filter(predicate).collect()

    for link in new_links:
        e = link["effective_index"]
        gate, last = max(e, link["known_index"]), min(e + 22, len(sessions) - 1)
        days = sessions[e - 22 : last + 1]
        p, n = (isins[link[k]] for k in ("predecessor_index", "successor_index"))
        spec = dict(
            predecessor=p,
            successor=n,
            effective=sessions[e],
            known=sessions[link["known_index"]],
        )
        scope = out / n
        scope.mkdir()
        predicate = pl.col("isin").is_in([p, n]) & pl.col(
            "source_trade_date"
        ).is_between(days[0], days[-1])
        cash = selected(cash_rec, predicate)
        qr = [
            r
            for r in qm["files"]
            if int(Path(r["output"]).stem[-4:]) in {d.year for d in days}
        ]
        quantity = pl.concat(
            [
                selected(
                    dict(
                        path=str(root / "cotahist_option_volume" / r["output"]),
                        sha256=r["output_sha256"],
                    ),
                    predicate,
                )
                for r in qr
            ],
            how="vertical_relaxed",
        )
        frames = [cash, quantity]
        for suffix in ("_oi.parquet", "_cash.parquet"):
            candidates = [r for r in inventory if Path(r["path"]).name.endswith(suffix)]
            chosen = [
                r
                for r in candidates
                if days[0] <= date.fromisoformat(Path(r["path"]).name[:10]) < days[-1]
            ]
            if chosen:
                frames.append(
                    pl.concat(
                        [selected(r, predicate) for r in chosen], how="vertical_relaxed"
                    )
                )
            else:
                # Before this archive begins, schema alone cannot create observations.
                frames.append(
                    pl.DataFrame(schema=pl.read_parquet_schema(candidates[0]["path"]))
                )
        for name, frame in zip(
            ("cash", "quantities", "snapshots", "nonregular"), frames
        ):
            frame.write_parquet(scope / (name + ".parquet"))
        before = activity_decision_features(*frames, days)
        after = renamed_activity_features(*frames, days, **spec)
        cells = 0
        for family, control, changed in zip(
            ("options", "microstructure"), before, after
        ):
            control.write_parquet(scope / (family + "_control.parquet"))
            changed.write_parquet(scope / (family + ".parquet"))
            cells += exact_nullable(
                control,
                parents[family],
                families[family]["feature_names"],
                sessions[gate],
                sessions[last],
                [p, n],
            )
            final[family] = replace(
                final[family], changed, n, sessions[gate], sessions[last]
            )
        cutoff = sessions[gate + 6]
        mutated = renamed_activity_features(
            *(f.filter(pl.col("source_trade_date") < cutoff) for f in frames),
            days,
            **spec,
        )
        for a, b in zip(mutated, after):
            assert_frame_equal(
                a.filter(pl.col("date") <= cutoff),
                b.filter(pl.col("date") <= cutoff),
                check_exact=True,
            )
        odd = selected(
            odd_rec,
            pl.col("isin").is_in([p, n])
            & pl.col("source_trade_date").is_between(sessions[e - 6], sessions[-1]),
        )
        odd.write_parquet(scope / "odd_source.parquet")
        odd_days = sessions[e - 6 :]
        a = odd_family(
            derive_known_archive_features(odd, odd_days, [p, n], group="oddlot"),
            odd_days,
        )
        b = odd_family(renamed_oddlot_features(odd, odd_days, **spec), odd_days).filter(
            pl.col("date") <= sessions[-1]
        )
        a = a.filter((pl.col("isin") == n) & (pl.col("date") >= sessions[gate]))
        a.write_parquet(scope / "oddlot_control.parquet")
        b.write_parquet(scope / "oddlot.parquet")
        odd_before.append(a)
        odd_after.append(b)
        scopes.append(
            dict(
                link=link,
                predecessor=p,
                successor=n,
                first=str(days[0]),
                last=str(days[-1]),
                control_cells=cells,
                source_rows={
                    k: len(f)
                    for k, f in zip(
                        ("cash", "quantities", "snapshots", "nonregular"), frames
                    )
                },
                odd_rows=len(odd),
                causal_prefix=str(cutoff),
            )
        )
        print(json.dumps(scopes[-1]), flush=True)
    pl.concat(odd_before).sort(["date", "isin"]).write_parquet(
        out / "oddlot_control.parquet"
    )
    final["oddlot"] = pl.concat(odd_after).sort(["date", "isin"])

    im = bound_json(families["rebalance"]["source_manifest"])
    snapshots = bound_json(im["snapshots"])
    portfolios = selected(
        snapshots["output"], pl.col("disclosure_date") >= date(2020, 4, 1)
    )
    portfolios.write_parquet(out / "index_portfolios.parquet")
    index_cash = selected(
        cash_rec, pl.col("source_trade_date") >= date(2020, 1, 1)
    ).select("source_trade_date", "isin", "ticker")
    index_cash.write_parquet(out / "index_tickers.parquet")
    base = Path(im["base_store"]["path"]).parent
    bound_json(im["base_store"])
    volume, seen = (
        np.load(base / "volume_brl.npy"),
        np.load(base / "activity_valid.npy"),
    )
    advs = []
    standalone = _prior_adv20(volume, seen)
    for chain in [old_links, links]:
        v, s = volume.copy(), seen.copy()
        for link in sorted(chain, key=lambda x: x["effective_index"]):
            p, n, e = (
                link[k]
                for k in ("predecessor_index", "successor_index", "effective_index")
            )
            v[:e, n], s[:e, n] = v[:e, p], s[:e, p]
        adv = _prior_adv20(v, s)
        for link in chain:
            gate = max(link["effective_index"], link["known_index"])
            adv[:gate, link["successor_index"]] = standalone[
                :gate, link["successor_index"]
            ]
        advs.append(adv)
    future = [date.fromisoformat(d) for d in bound_json(im["calendar"])["sessions"]]
    first = min(x["effective_index"] for x in new_links)
    index_reports = {}
    for label, membership, adv, chain in [
        ("control", old_active, advs[0], old_links),
        ("eligibility_only", active, advs[0], old_links),
        ("corrected", active, advs[1], links),
    ]:
        frame, report = pressure_panel(
            portfolios,
            index_cash,
            sessions,
            isins,
            membership,
            adv,
            future_sessions=future,
            history_links=tuple(chain),
        )
        frame = frame.filter(pl.col("date") >= sessions[first])
        frame.write_parquet(out / ("index_" + label + ".parquet"))
        index_reports[label] = report
        if label == "control":
            count = exact_nullable(
                frame,
                parents["rebalance"],
                families["rebalance"]["feature_names"],
                sessions[first],
                sessions[-1],
                isins,
            )
        elif label == "corrected":
            final["rebalance"] = pl.concat(
                [parents["rebalance"].filter(pl.col("date") < sessions[first]), frame]
            ).sort(["date", "isin"])
    cutoff = date(2024, 9, 17)
    mutation, _ = pressure_panel(
        portfolios.filter(pl.col("disclosure_date") <= cutoff),
        index_cash.filter(pl.col("source_trade_date") <= cutoff),
        sessions,
        isins,
        active,
        advs[1],
        future_sessions=future,
        history_links=tuple(links),
    )
    assert_frame_equal(
        mutation.filter(pl.col("date").is_between(sessions[first], cutoff)),
        final["rebalance"].filter(pl.col("date").is_between(sessions[first], cutoff)),
        check_exact=True,
    )
    for name, frame in final.items():
        frame.write_parquet(out / (name + ".parquet"))
    report = dict(
        status="qualified_auxiliary_reducers_pending_consumer",
        plan=binding(out / "plan.json"),
        parent=admission["parent"],
        prior=run["surviving_rename_auxiliaries"],
        scopes=scopes,
        index_control_cells=count,
        index_reports=index_reports,
        index_causal_prefix=str(cutoff),
        source_receipts=list(receipts.values()),
        artifacts={p.stem: binding(p) for p in out.glob("*.parquet")},
        seconds=perf_counter() - tick,
        limits="Four family intermediates; original input support and missingness remain. No raw census/source retrieval, new option-series/loan alias, fullstore or model result. Existing twelve index cases include one new reopening/future-mutation fixture; no workbook/source audit repeated.",
    )
    write_json_atomic(out / "manifest.json", report)
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    run["stage_c_event_auxiliaries"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            {k: report[k] for k in ("status", "index_control_cells", "seconds")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
