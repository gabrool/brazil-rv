"""Bind five market continuations, including two distinct JSL identity episodes."""

from datetime import datetime
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.build_store import _route_decision_known_continuations
from brazil_rv.v2.contract import (
    UNIVERSE_MIN_HISTORY,
    UNIVERSE_MIN_MEDIAN_VOLUME_BRL,
    UNIVERSE_MIN_PRIOR_CLOSE_BRL,
    UNIVERSE_MIN_TRADED,
    UNIVERSE_PRIOR_SESSIONS,
)
from brazil_rv.v2.data_foundation import load_isin_link_allowlist, slow_history_links
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.decision_clock import load_session_schedule
from brazil_rv.v2.universe import build_daily_universe

PROJECT = Path(__file__).resolve().parents[1]
MARKET_LINKS = {
    "JSL_holding_SIMPAR": "SIMH3",
    "JSL_logistics_ticker": "JSLG3",
    "TIMP_TIMS": "TIMS3",
    "BKBR_ZAMP": "ZAMP3",
    "RRRP_BRAV": "BRAV3",
}


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    admission = bound_json(run["stage_c_event_source_admission"])
    parent = bound_json(run["economic_refit_inputs"])["store"]
    root = Path(parent["root"])
    m = bound_json(
        dict(path=str(root / "manifest.json"), sha256=parent["manifest_sha256"])
    )
    out = Path(run["stage_c_root"]) / "event_data"
    out.mkdir(exist_ok=False)
    (out / "executed_admission.py").write_bytes(Path(__file__).read_bytes())
    dates, names = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    events = sorted(
        (e for e in admission["events"] if e["id"] in MARKET_LINKS),
        key=lambda e: e["effective_date"],
    )
    first = int(np.searchsorted(dates, np.datetime64(events[0]["effective_date"])))
    start = first - 253
    schedule_record = next(
        s
        for s in m["sources"]
        if Path(s.get("path", "")).name == "b3_session_schedule_reconstructed_v1.csv"
    )
    schedule = {
        s.trade_date: s.decision_at
        for s in load_session_schedule(Path(schedule_record["path"]))
    }
    decisions = [schedule[d.astype(object)] for d in dates]
    plan = dict(
        parent=parent,
        sources=run["stage_c_event_source_admission"],
        input_scope=run["stage_c_event_input_scope"],
        registration=binding(
            PROJECT / "research/preregistrations/v2_economic_data_scaling.md"
        ),
        events=events,
        full_population=933,
        full_history=60,
        contrast="Five sourced1:1/no-cash market continuations: old holding JSL->SIMPAR, TIM reverse merger, BKBR->ZAMP, RRRP->BRAV, and separately sourced JSLG11->new-logistics JSLG3. Market price/activity continuity does not pool the legal issuers' financial filings. Acquired LCAM/SULA and VIVT/MODL/VVAR class/unit conversions supply gross claim targets only, never pooled successor feature history.",
        reused_isin="BRJSLGACNOR2 has old holding observations throughSeptember17 2020 and a separately sourced logistics episode fromNovember11. The old episode supplies SIMPAR history. New logistics history comes only from BRJSLGA02OR1. Preserve earlier public JSL rows; never use one final static root to overwrite both episodes. Frozen acceptance requires episode-aware daily/native/issuer/lending/auxiliary consumers, not this isolated liquidity proof.",
        data_scope="Reuse accepted normalized sources and prior algorithms. Five boundary liquidity controls first; final original-Float32 wealth determines daily/risk/native/market tails. Then own-version issuer/financial, lending unit barriers and affected auxiliaries, gross target recomposition and complete new store. Preserve all existing stores and source assignments. No raw/source census, oldfit-coordinate replacement, heldout read or new model hypothesis.",
        freezes="Remove source-disproved JSL November11 q3.058 in a separate source layer. Keep economic effect/knowledge/delivery/payment distinct. Existing corporate72/rename668 outcomes and every prior feature proof are reused, not rerun. Old source entry restrictions may reopen only at the sourced new episode. Unsupported raw M1/issuer values stay missing.",
        schedule=schedule_record,
        source_start=str(dates[start]),
        source_end=str(dates[-1]),
    )
    write_json_atomic(out / "plan.json", plan)
    selected_isins = sorted({e[k] for e in events for k in ("isin", "successor_isin")})
    years = range(int(str(dates[start])[:4]), 2025)
    sources = [
        s
        for s in m["sources"]
        if Path(s.get("path", "")).name
        in {f"equities_daily_{y}.parquet" for y in years}
    ]
    daily = pl.concat(
        [
            pl.scan_parquet(s["path"])
            .filter(
                pl.col("isin").is_in(selected_isins)
                & (pl.col("market_type") == 10)
                & (pl.col("trade_date") >= dates[start].astype(object))
            )
            .collect()
            for s in sources
        ]
    )
    daily.write_parquet(out / "normalized_quotes.parquet")
    evidence = binding(out / "plan.json")
    added, packed, reports = [], {}, []

    def old(key):
        return np.load(root / m["arrays"][key]["path"], mmap_mode="r")

    for event in events:
        label = event["id"]
        folder = out / label
        folder.mkdir()
        pair = [event["isin"], event["successor_isin"]]
        known = max(
            datetime.fromisoformat(admission["originals"][p]["available_at"])
            for p in event["source_protocols"]
        )
        row = dict(
            ticker=MARKET_LINKS[label],
            predecessor_isin=pair[0],
            successor_isin=pair[1],
            effective_date=event["effective_date"],
            first_known_at=known.isoformat(),
            shares_received_per_prior_share=1.0,
            cash_entitlement_per_prior_share=0.0,
            currency="BRL",
            source=evidence["path"],
            evidence_sha256=evidence["sha256"],
        )
        pl.DataFrame([row]).write_csv(folder / "allowlist.csv")
        effect_date = np.datetime64(event["effective_date"]).astype(object)
        # Explicit source-backed episode bounds, not a global assertion that the
        # reused JSL ISIN never appeared before/after this particular transition.
        source_rows = daily.filter(
            ((pl.col("isin") == pair[0]) & (pl.col("trade_date") < effect_date))
            | ((pl.col("isin") == pair[1]) & (pl.col("trade_date") >= effect_date))
        )
        link = load_isin_link_allowlist(folder / "allowlist.csv", source_rows)
        source_rows.write_parquet(folder / "bounded_identity_rows.parquet")
        t = int(np.searchsorted(dates, np.datetime64(event["effective_date"])))
        assert str(link["predecessor_last_date"][0]) == str(dates[t - 1])
        assert known <= decisions[t]
        added.append(link)
        axes = [names.index(s) for s in pair]
        begin, stop = t - 253, min(t + 61, len(dates))
        days = dates[begin:stop]
        selected = np.arange(t - begin, len(days))
        close = np.round(old("raw_close")[begin:stop][:, axes].astype(np.float64), 2)
        volume = old("volume_brl")[begin:stop][:, axes].astype(np.float64)
        quotes = daily.filter(
            pl.col("isin").is_in(pair)
            & pl.col("trade_date").is_between(
                days[0].astype(object), days[-1].astype(object)
            )
        )
        qi, qj = (
            np.searchsorted(days, quotes["trade_date"].to_numpy()),
            np.array([pair.index(s) for s in quotes["isin"]]),
        )
        volume[qi, qj] = quotes["volume_brl"].to_numpy()
        observed, traded, activity = (
            old(k)[begin:stop][:, axes].copy()
            for k in ("observed", "trade_observed", "activity_valid")
        )
        complete = old("source_session_complete")[begin:stop, 0]
        control = build_daily_universe(
            close,
            volume,
            observed,
            trade_observed=traded,
            activity_valid=activity,
            source_session_complete=complete,
        )
        np.testing.assert_array_equal(
            control.active[selected], old("active")[begin:stop][:, axes][selected]
        )
        route = dict(
            dates=days,
            isins=pair,
            links=link,
            decision_timestamps=decisions[begin:stop],
            raw_close=close,
            volume_brl=volume,
            trades=old("trade_count")[begin:stop][:, axes],
            observed=observed,
            trade_observed=traded,
            activity_valid=activity,
            ambiguous_action=np.zeros_like(observed),
            shareholder_wealth_arrays=(),
        )
        linked = _route_decision_known_continuations(**route)
        after = build_daily_universe(
            linked.close_brl,
            linked.volume_brl,
            linked.observed,
            trade_observed=linked.trade_observed,
            activity_valid=linked.activity_valid,
            source_session_complete=complete,
        )
        active = after.active & linked.claim_owner
        # Old JSL source output ends at reopening. The separately processed
        # logistics link exclusively owns every new-episode amendment thereafter.
        writable = np.ones((len(selected), 2), bool)
        if event["source_reopens_date"]:
            writable[:, 0] = days[selected] < np.datetime64(
                event["source_reopens_date"]
            )
        for day in selected:
            for j in (0, 1):
                if not writable[day - selected[0], j]:
                    continue
                first_print = np.flatnonzero(linked.observed[:, j])
                prior = np.flatnonzero(
                    linked.observed[day - UNIVERSE_PRIOR_SESSIONS : day, j]
                )
                price = (
                    linked.close_brl[day - UNIVERSE_PRIOR_SESSIONS + prior[-1], j]
                    if len(prior)
                    else np.nan
                )
                expected = bool(
                    len(first_print)
                    and linked.claim_owner[day, j]
                    and linked.activity_valid[
                        day - UNIVERSE_PRIOR_SESSIONS : day, j
                    ].all()
                    and linked.trade_observed[
                        day - UNIVERSE_PRIOR_SESSIONS : day, j
                    ].sum()
                    >= UNIVERSE_MIN_TRADED
                    and np.median(
                        linked.volume_brl[day - UNIVERSE_PRIOR_SESSIONS : day, j]
                    )
                    >= UNIVERSE_MIN_MEDIAN_VOLUME_BRL
                    and price >= UNIVERSE_MIN_PRIOR_CLOSE_BRL
                    and day - first_print[0] >= UNIVERSE_MIN_HISTORY
                )
                assert active[day, j] == expected
        delayed = _route_decision_known_continuations(
            **{
                **route,
                "links": link.with_columns(
                    pl.lit(decisions[t + 1]).alias("first_known_at")
                ),
            }
        )
        np.testing.assert_array_equal(delayed.close_brl, close)
        np.testing.assert_array_equal(delayed.observed, observed)
        for key, array in (
            ("active", active),
            ("prior_reference_close", after.prior_close_brl.astype(np.float32)),
        ):
            prior = old(key)[begin:stop][:, axes][selected]
            new = array[selected]
            ix = np.argwhere(
                writable & ~((prior == new) | (np.isnan(prior) & np.isnan(new)))
            )
            packed.setdefault(key + "__indices", []).append(
                np.column_stack((ix[:, 0] + t, np.array(axes)[ix[:, 1]]))
            )
            packed.setdefault(key + "__values", []).append(new[tuple(ix.T)])
        np.savez_compressed(
            folder / "liquidity.npz",
            dates=days,
            axes=axes,
            rows=np.arange(begin, stop),
            control=control.active,
            active=active,
            prior_close=after.prior_close_brl,
            close=close,
            volume=volume,
            observed=observed,
            traded=traded,
            activity=activity,
            writable=writable,
        )
        reports.append(
            dict(
                event=label,
                source=pair[0],
                successor=pair[1],
                control_cells=int(control.active[selected].size),
                gains=int(
                    (writable & active[selected] & ~control.active[selected]).sum()
                ),
                losses=int(
                    (writable & ~active[selected] & control.active[selected]).sum()
                ),
                gained_dates=[
                    str(d)
                    for d in days[selected][
                        active[selected, 1] & ~control.active[selected, 1]
                    ]
                ],
            )
        )
    old_links = pl.read_parquet(root / m["tables"]["isin_succession_links"]["path"])
    links = pl.concat([old_links, *added]).sort("successor_first_date")
    roots, ancestors = {}, []
    for row in links.iter_rows(named=True):
        root_name = roots.get(row["predecessor_isin"], row["predecessor_isin"])
        roots[row["successor_isin"]] = root_name
        ancestors.append(root_name)
    links = links.with_columns(pl.Series("continuation_isin", ancestors))
    links.write_parquet(out / "isin_succession_links.parquet")
    slow_history_links(links, dates, names, decisions).write_parquet(
        out / "slow_history_links.parquet"
    )
    np.savez_compressed(
        out / "liquidity_deltas.npz",
        **{k: np.concatenate(v) for k, v in packed.items()},
    )
    report = dict(
        status="bounded_market_identity_and_liquidity_qualified_remaining_dependencies_pending",
        parent=parent,
        plan=binding(out / "plan.json"),
        links=binding(out / "isin_succession_links.parquet"),
        history_mapping=binding(out / "slow_history_links.parquet"),
        deltas=binding(out / "liquidity_deltas.npz"),
        sources=sources,
        normalized_quotes=binding(out / "normalized_quotes.parquet"),
        cases=reports,
        seconds=perf_counter() - tick,
        limits="No global allowlist replacement or full builder permission for a reused ISIN. Public earlier rows and all accepted stores remain immutable. New complete assembly must bind episode-aware dependencies before refits.",
    )
    write_json_atomic(out / "manifest.json", report)
    run["stage_c_event_data_admission"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(json.dumps(dict(cases=reports, seconds=report["seconds"])))


if __name__ == "__main__":
    main()
