"""One bounded daily reconstruction; reuse previously qualified parent reducers."""

from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.corporate_actions import (
    align_decision_known_action_terms,
    verified_action_terms_from_table,
    verified_conversion_terms_from_links,
)
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.decision_clock import (
    load_session_schedule,
    next_session_decision_cutoffs,
)
from brazil_rv.v2.features import build_slow_features_into, exact_log_return

PROJECT = Path(__file__).resolve().parents[1]


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    admission = bound_json(run["stage_c_event_data_admission"])
    identities = bound_json(admission["plan"])
    wealth_report = bound_json(run["stage_c_event_wealth"])
    parent_daily = bound_json(run["surviving_rename_daily"])
    parent_proof = bound_json(run["surviving_rename_daily_qualification"])
    assert parent_proof["status"] == "passed"
    source = Path(admission["plan"]["path"]).parent
    root = Path(admission["parent"]["root"])
    m = bound_json(
        dict(
            path=str(root / "manifest.json"),
            sha256=admission["parent"]["manifest_sha256"],
        )
    )
    out = source / "daily" / "corrected"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    (out / "executed_features.py").write_bytes(
        (PROJECT / "research/src/brazil_rv/v2/features.py").read_bytes()
    )
    dates, names = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    first = int(
        np.searchsorted(dates, np.datetime64(identities["events"][0]["effective_date"]))
    )
    start, stop = first - 253, len(dates)
    days = dates[start:]
    selected = np.arange(first - start, len(days))
    write_json_atomic(
        out / "plan.json",
        dict(
            parent=admission["parent"],
            admission=run["stage_c_event_data_admission"],
            wealth=run["stage_c_event_wealth"],
            rows=[start, first, stop],
            parent_reducers=run["surviving_rename_daily"],
            parent_qualification=run["surviving_rename_daily_qualification"],
            source_coordinates=parent_daily["source_coordinates"],
            contract="Reuse accepted parent controls/source masks and original normalized full933 cross-sections. Run corrected32-field reducers once with253prior sessions and actual Float32 wealth tails. Preserve original20/60/126/252windows,48/101support,3other-peer rule and causal monthly clusters. Future reused JSL history never replaces its historical contribution to the market median or another issuer's membership. Unranked ages use full-axis birth metadata during qualification; this is intermediate, not complete-store admission.",
        ),
    )
    np.save(out / "date_index.npy", days)
    np.save(out / "selected.npy", selected)

    def old(key):
        return np.load(root / m["arrays"][key]["path"], mmap_mode="r")[start:]

    active = old("active").copy()
    with np.load(admission["deltas"]["path"]) as z:
        ix = z["active__indices"].copy()
        ix[:, 0] -= start
        active[tuple(ix.T)] = z["active__values"]
    wealth = [
        old("shareholder_wealth_" + k).copy()
        for k in ("open", "high", "low", "close", "valid")
    ]
    with np.load(wealth_report["deltas"]["path"]) as z:
        for k, array in zip(
            ("open", "high", "low", "close", "valid"), wealth, strict=True
        ):
            ix = z["shareholder_wealth_" + k + "__indices"].copy()
            ix[:, 0] -= start
            array[tuple(ix.T)] = z["shareholder_wealth_" + k + "__values"]
    # No source scan: these all-name normalized coordinates and the distinction
    # between feature rows and later exit-only quotes were already qualified.
    quotes = (
        pl.scan_parquet(parent_daily["source_coordinates"]["path"])
        .filter(pl.col("trade_date") >= days[0].astype(object))
        .collect()
    )
    lookup = {s: j for j, s in enumerate(names)}
    ti, ni = (
        np.searchsorted(days, quotes["trade_date"].to_numpy()),
        np.array([lookup[s] for s in quotes["isin"]]),
    )
    volume, trades = (
        old("volume_brl").astype(np.float64),
        old("trade_count").astype(np.float64),
    )
    raw = [
        np.round(old("raw_" + k).astype(np.float64), 2)
        for k in ("high", "low", "close")
    ]
    for array, key in zip(
        (volume, trades, *raw),
        ("volume_brl", "trades", "high_brl", "low_brl", "close_brl"),
        strict=True,
    ):
        array[ti, ni] = quotes[key].to_numpy()
    present = np.zeros_like(active)
    present[ti, ni] = True
    observed, activity = old("observed").copy() & present, old("activity_valid").copy()
    volume[activity & ~present] = 0
    trades[activity & ~present] = 0
    terms = verified_action_terms_from_table(
        pl.read_parquet(root / m["tables"]["corporate_actions_verified_terms"]["path"])
    )
    terms = tuple(
        replace(t, shares_per_prior_share=1.0, cash_per_prior_share=0.0)
        if t.isin == "BRJSLGACNOR2" and str(t.effective_date) == "2020-11-11"
        else t
        for t in terms
    )
    links = pl.read_parquet(admission["links"]["path"]).sort("successor_first_date")
    # The complete parent already contains its five admitted conversions.
    # Extend that term table; never append a second copy of an existing event.
    combined = {(t.isin, t.effective_date, t.sequence): t for t in terms}
    for term in verified_conversion_terms_from_links(links):
        key = term.isin, term.effective_date, term.sequence
        if key in combined:
            assert combined[key] == term
        else:
            combined[key] = term
    schedule = load_session_schedule(Path(identities["schedule"]["path"]))
    ss = tuple(
        s for s in schedule if days[0] <= np.datetime64(s.trade_date) <= days[-1]
    )
    following = next(
        s.decision_at for s in schedule if np.datetime64(s.trade_date) > days[-1]
    )
    actions = align_decision_known_action_terms(
        tuple(combined.values()),
        days,
        names,
        coverage_resolved=observed,
        decision_timestamps=next_session_decision_cutoffs(
            ss, following_decision_at=following
        ),
    )
    ambiguous = ~actions.session_resolved | (old("observed") & ~present)
    history_seen = observed.copy()
    mapping, saved_jsl = [], None
    for row in links.iter_rows(named=True):
        a, b = (names.index(row[k]) for k in ("predecessor_isin", "successor_isin"))
        boundary = (
            int(np.searchsorted(dates, np.datetime64(row["effective_date"]))) - start
        )
        item = dict(
            predecessor_index=a,
            successor_index=b,
            effective_index=boundary,
            known_index=boundary,
        )
        if row["predecessor_isin"] == "BRJSLGACNOR2":
            item["source_reopens_index"] = int(
                np.searchsorted(days, np.datetime64("2020-11-11"))
            )
        mapping.append(item)
        if boundary <= 0:
            continue
        assert row["first_known_at"] <= ss[boundary].decision_at
        if row["predecessor_isin"] == "BRJSLGA02OR1":
            saved_jsl = (
                b,
                boundary,
                wealth[3][:, b].copy(),
                wealth[4][:, b].copy(),
                ambiguous[:, b].copy(),
            )
        for array in (
            *wealth,
            volume,
            trades,
            *raw,
            observed,
            history_seen,
            activity,
            ambiguous,
        ):
            array[:boundary, b] = array[:boundary, a]
    assert saved_jsl is not None
    j, reopening, public_close, public_valid, public_ambiguous = saved_jsl
    public_returns = {}
    for horizon in (1, 5, 21):
        values, valid = exact_log_return(
            wealth[3], horizon, ambiguous, shareholder_wealth_valid=wealth[4]
        )
        old_values, old_valid = exact_log_return(
            public_close[:, None],
            horizon,
            public_ambiguous[:, None],
            shareholder_wealth_valid=public_valid[:, None],
        )
        values[:reopening, j], valid[:reopening, j] = (
            old_values[:reopening, 0],
            old_valid[:reopening, 0],
        )
        public_returns[horizon] = values, valid
        np.savez_compressed(
            out / f"public_returns_{horizon}.npz", values=values, valid=valid
        )
    for k, array in zip(("open", "high", "low", "close", "valid"), wealth, strict=True):
        np.save(out / ("wealth_" + k + ".npy"), array)
    for k, array in (
        ("active", active),
        ("ambiguous", ambiguous),
        ("volume", volume),
        ("activity", activity),
    ):
        np.save(out / (k + ".npy"), array)
    write_json_atomic(out / "history_mapping.json", mapping)

    def consume(field, values, valid):
        if field not in (25, 26):
            np.savez_compressed(out / f"{field}.npz", values=values, valid=valid)
        print(json.dumps(dict(field=field, seconds=perf_counter() - tick)), flush=True)

    labels = build_slow_features_into(
        *wealth[:4],
        volume,
        trades,
        wealth[4],
        active,
        days,
        raw_high=raw[0],
        raw_low=raw[1],
        raw_close=raw[2],
        price_observed=observed,
        history_observed=history_seen,
        activity_valid=activity,
        ambiguous_action=ambiguous,
        consume=consume,
        history_links=mapping,
        cross_section_returns=public_returns,
    )
    np.save(out / "clusters.npy", labels)
    report = dict(
        status="corrected_reducers_saved_pending_qualification",
        plan=binding(out / "plan.json"),
        source_coordinates=parent_daily["source_coordinates"],
        normalized_rows=quotes.height,
        parent_controls_reused=True,
        seconds=perf_counter() - tick,
    )
    write_json_atomic(out / "manifest.json", report)
    # Other independent account workers only read their frozen startup plan.
    run = json.loads(pointer.read_text())
    run["stage_c_event_daily"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
