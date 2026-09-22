"""Bounded added-period data repairs using already qualified source evidence."""

from dataclasses import asdict, replace
from datetime import date, datetime, timezone
import json
from pathlib import Path
import sys
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
from brazil_rv.v2.corporate_actions import (
    VerifiedActionTerm,
    align_decision_known_action_terms,
    verified_action_terms_from_table,
    verified_action_terms_to_table,
    verified_conversion_terms_from_links,
)
from brazil_rv.v2.decision_clock import (
    load_session_schedule,
    next_session_decision_cutoffs,
)
from brazil_rv.v2.universe import build_daily_universe
from propagate_surviving_wealth import resume

PROJECT = Path(__file__).resolve().parents[1]
POINTER = PROJECT / "docs/v2_economic_data_scaling_run.json"


def context():
    run = json.loads(POINTER.read_text())
    plan = bound_json(run["scaling_data_plan"])
    root = Path(plan["parent"]["root"])
    manifest = bound_json(
        dict(path=str(root / "manifest.json"), sha256=plan["parent"]["manifest_sha256"])
    )
    return run, plan, root, manifest, Path(run["scaling_data_plan"]["path"]).parent


def record(run, key, path, value):
    write_json_atomic(path, value)
    # Preserve unrelated progress made by an independent storage operation.
    current = json.loads(POINTER.read_text())
    current[key] = binding(path)
    write_json_atomic(POINTER, current)


def prepare():
    run = json.loads(POINTER.read_text())
    parent = bound_json(run["economic_refit_inputs"])["store"]
    root = Path(parent["root"])
    manifest = bound_json(
        dict(path=str(root / "manifest.json"), sha256=parent["manifest_sha256"])
    )
    out = Path(run["scaling_investigation"]["path"]).parent / "data_repairs"
    out.mkdir(exist_ok=False)
    (out / "executed_preparation.py").write_bytes(Path(__file__).read_bytes())
    source_records, originals, terms = [], {}, []
    for key in ("scaling_expanded_source_plan", "scaling_expanded_later_source_plan"):
        prior = bound_json(run[key])
        source = bound_json(prior["source_admission"])
        source_records.append(prior["source_admission"])
        originals.update(source["originals"])
        terms.append(source.get("events", source.get("terms")))
    history = []
    for label, ticker, before, after, effect, protocols in (
        (
            "QGEP_ENAT",
            "ENAT3",
            "BRQGEPACNOR8",
            "BRENATACNOR0",
            "2019-04-24",
            ["681261"],
        ),
        (
            "GPC_DEXP_ON",
            "DEXP3",
            "BRGPCPACNOR4",
            "BRDEXPACNOR1",
            "2021-06-08",
            ["871753", "874238"],
        ),
        (
            "GPC_DEXP_PN",
            "DEXP4",
            "BRGPCPACNPR1",
            "BRDEXPACNPR8",
            "2021-06-08",
            ["871753", "874238"],
        ),
        (
            "WIZS_WIZC",
            "WIZC3",
            "BRWIZSACNOR1",
            "BRWIZCACNOR5",
            "2023-02-09",
            ["1052903", "1056754"],
        ),
    ):
        known = max(
            datetime.fromisoformat(originals[p]["available_at"]) for p in protocols
        )
        history.append(
            dict(
                id=label,
                ticker=ticker,
                isin=before,
                successor_isin=after,
                effective_date=effect,
                first_known_at=known.isoformat(),
                protocols=protocols,
            )
        )
    scalars = []
    for term, protocol in zip(
        terms[0]["scalar_actions"], ("680551", "685074"), strict=True
    ):
        scalars.append(
            dict(
                **term,
                protocol=protocol,
                first_known_at=originals[protocol]["available_at"],
            )
        )
    dates = np.load(root / "date_index.npy")
    names = np.load(root / "isin_index.npy").tolist()
    assert len(names) == 933 and len(dates) == 3717 and str(dates[-1]) == "2024-12-30"
    schedule = next(
        s
        for s in manifest["sources"]
        if Path(s.get("path", "")).name == "b3_session_schedule_reconstructed_v1.csv"
    )
    assignments = next(
        s
        for s in manifest["sources"]
        if Path(s.get("path", "")).name == "xp_accepted_source_assignments_v1.parquet"
    )
    selected = sorted(
        {e[k] for e in history for k in ("isin", "successor_isin")}
        | {e["isin"] for e in scalars}
    )
    assignment_rows = pl.read_parquet(assignments["path"]).filter(
        pl.col("isin").is_in(selected)
    )
    assignment_rows.write_parquet(out / "selected_assignment_metadata.parquet")
    plan = dict(
        status="frozen_before_new_data_or_model_outcomes",
        parent=parent,
        registration=binding(
            PROJECT / "research/preregistrations/v2_economic_data_scaling.md"
        ),
        sources=source_records,
        originals=originals,
        history=history,
        scalars=scalars,
        distributions=[e for t in terms for e in t["share_distributions"]],
        schedule=schedule,
        assignments=assignments,
        selected_assignment_metadata=binding(
            out / "selected_assignment_metadata.parquet"
        ),
        contract="Four same-legal-company/class histories, two source-disproved scalars and four gross distributions. Preserve all accepted stores,933 names,3717 dates and full60 history. No acquired/class-conversion history pooling, source/option/loan alias or missing-observation invention. Reuse dated source admissions and accepted normalized sources. Only changed dependencies and untested interactions need new verification.",
        propagation="Liquidity, original Float32 wealth tails, causal daily/risk/common state, own-version issuer/financial/lending unit barriers, native/scalars and affected auxiliary families; final gross/neutral outcomes and complete accepted store. Existing JSL identity episodes remain preserved. No old fit consumes changed coordinates.",
        target_boundary="Fibria cash revises50.12 to50.20 only at its saved knowledge date. Gross labels retain unremunerated entitlement cash and exact endpoints. GUARP/ON is a class conversion, not an extra history link. RLOG/Smiles are acquired claims. Unknown Smiles/ENAT auctions and unvalued Linx BDR remain unsupported, never zero-priced.",
        source_scope="Only selected source-assignment metadata is read beyond2024; no held-out market consumer. Daily normalized history begins253sessions before the first new history/action and ends2024-12-30; no new source census.",
    )
    record(run, "scaling_data_plan", out / "plan.json", plan)
    print(
        json.dumps(
            dict(
                history=history,
                scalar_names=[e["isin"] for e in scalars],
                assignment_metadata_rows=assignment_rows.height,
                distribution_names=[e["isin"] for e in plan["distributions"]],
            )
        )
    )


def admit():
    tick = perf_counter()
    run, plan, root, manifest, out = context()
    dest = out / "identity"
    dest.mkdir(exist_ok=False)
    (dest / "executed.py").write_bytes(Path(__file__).read_bytes())
    dates, names = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    schedule = {
        s.trade_date: s.decision_at
        for s in load_session_schedule(Path(plan["schedule"]["path"]))
    }
    decisions = [schedule[d.astype(object)] for d in dates]
    first = min(
        int(np.searchsorted(dates, np.datetime64(e["effective_date"])))
        for e in plan["history"]
    )
    start = first - 253
    selected_names = sorted(
        {e[k] for e in plan["history"] for k in ("isin", "successor_isin")}
        | {e["isin"] for e in plan["scalars"]}
    )
    # Chained successors are needed for wealth tails, with no additional identity admission.
    old_links = pl.read_parquet(
        root / manifest["tables"]["isin_succession_links"]["path"]
    )
    while True:
        extended = sorted(
            set(selected_names)
            | set(
                old_links.filter(pl.col("predecessor_isin").is_in(selected_names))[
                    "successor_isin"
                ]
            )
        )
        if extended == selected_names:
            break
        selected_names = extended
    years = range(int(str(dates[start])[:4]), 2025)
    sources = [
        s
        for s in manifest["sources"]
        if Path(s.get("path", "")).name
        in {f"equities_daily_{y}.parquet" for y in years}
    ]
    daily = pl.concat(
        [
            pl.scan_parquet(s["path"])
            .filter(
                pl.col("isin").is_in(selected_names)
                & (pl.col("market_type") == 10)
                & pl.col("trade_date").is_between(
                    dates[start].astype(object), dates[-1].astype(object)
                )
            )
            .collect()
            for s in sources
        ]
    )
    daily.write_parquet(dest / "normalized_quotes.parquet")
    packed, reports, added = {}, [], []

    def old(key):
        return np.load(root / manifest["arrays"][key]["path"], mmap_mode="r")

    for event in plan["history"]:
        folder = dest / event["id"]
        folder.mkdir()
        pair = [event["isin"], event["successor_isin"]]
        effect = np.datetime64(event["effective_date"])
        rows = daily.filter(
            (
                (pl.col("isin") == pair[0])
                & (pl.col("trade_date") < effect.astype(object))
            )
            | (
                (pl.col("isin") == pair[1])
                & (pl.col("trade_date") >= effect.astype(object))
            )
        )
        pl.DataFrame(
            [
                dict(
                    ticker=event["ticker"],
                    predecessor_isin=pair[0],
                    successor_isin=pair[1],
                    effective_date=event["effective_date"],
                    first_known_at=event["first_known_at"],
                    shares_received_per_prior_share=1.0,
                    cash_entitlement_per_prior_share=0.0,
                    currency="BRL",
                    source=str(out / "plan.json"),
                    evidence_sha256=binding(out / "plan.json")["sha256"],
                )
            ]
        ).write_csv(folder / "allowlist.csv")
        link = load_isin_link_allowlist(folder / "allowlist.csv", rows)
        rows.write_parquet(folder / "bounded_identity_rows.parquet")
        added.append(link)
        t = int(np.searchsorted(dates, effect))
        begin, stop = t - 253, min(t + 61, len(dates))
        days, axes = dates[begin:stop], [names.index(s) for s in pair]
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
        observed, traded, activity = [
            old(k)[begin:stop][:, axes].copy()
            for k in ("observed", "trade_observed", "activity_valid")
        ]
        complete = old("source_session_complete")[begin:stop, 0]
        control = build_daily_universe(
            close,
            volume,
            observed,
            trade_observed=traded,
            activity_valid=activity,
            source_session_complete=complete,
        )
        selected = np.arange(t - begin, len(days))
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
        for day in selected:
            for j in (0, 1):
                printed = np.flatnonzero(linked.observed[:, j])
                prior = np.flatnonzero(
                    linked.observed[day - UNIVERSE_PRIOR_SESSIONS : day, j]
                )
                price = (
                    linked.close_brl[day - UNIVERSE_PRIOR_SESSIONS + prior[-1], j]
                    if len(prior)
                    else np.nan
                )
                expected = bool(
                    len(printed)
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
                    and day - printed[0] >= UNIVERSE_MIN_HISTORY
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
        for key, array in (
            ("active", active),
            ("prior_reference_close", after.prior_close_brl.astype(np.float32)),
        ):
            before, after_rows = (
                old(key)[begin:stop][:, axes][selected],
                array[selected],
            )
            ix = np.argwhere(
                ~((before == after_rows) | (np.isnan(before) & np.isnan(after_rows)))
            )
            packed.setdefault(key + "__indices", []).append(
                np.column_stack((ix[:, 0] + t, np.array(axes)[ix[:, 1]]))
            )
            packed.setdefault(key + "__values", []).append(after_rows[tuple(ix.T)])
        np.savez_compressed(
            folder / "liquidity.npz",
            dates=days,
            axes=axes,
            control=control.active,
            active=active,
            close=close,
            volume=volume,
            observed=observed,
            traded=traded,
            activity=activity,
            prior_close=after.prior_close_brl,
        )
        reports.append(
            dict(
                event=event["id"],
                control_cells=int(control.active[selected].size),
                gains=int((active[selected] & ~control.active[selected]).sum()),
                losses=int((~active[selected] & control.active[selected]).sum()),
                gained_dates=[
                    str(d)
                    for d in days[selected][
                        active[selected, 1] & ~control.active[selected, 1]
                    ]
                ],
            )
        )
    links = pl.concat([old_links, *added]).sort("successor_first_date")
    roots, ancestors = {}, []
    for row in links.iter_rows(named=True):
        ancestor = roots.get(row["predecessor_isin"], row["predecessor_isin"])
        roots[row["successor_isin"]] = ancestor
        ancestors.append(ancestor)
    links = links.with_columns(pl.Series("continuation_isin", ancestors))
    links.write_parquet(dest / "isin_succession_links.parquet")
    slow_history_links(links, dates, names, decisions).write_parquet(
        dest / "slow_history_links.parquet"
    )
    np.savez_compressed(
        dest / "liquidity_deltas.npz",
        **{k: np.concatenate(v) for k, v in packed.items()},
    )
    report = dict(
        status="identity_liquidity_qualified_dependencies_pending",
        parent=plan["parent"],
        plan=run["scaling_data_plan"],
        links=binding(dest / "isin_succession_links.parquet"),
        history_mapping=binding(dest / "slow_history_links.parquet"),
        deltas=binding(dest / "liquidity_deltas.npz"),
        normalized_quotes=binding(dest / "normalized_quotes.parquet"),
        sources=sources,
        cases=reports,
        seconds=perf_counter() - tick,
    )
    record(run, "scaling_data_identity", dest / "manifest.json", report)
    print(json.dumps(dict(cases=reports, seconds=report["seconds"])))


def wealth():
    tick = perf_counter()
    run, plan, root, manifest, out = context()
    identity = bound_json(run["scaling_data_identity"])
    dest = out / "wealth"
    resuming = dest.exists()
    assert not (dest / "manifest.json").exists()
    dest.mkdir(exist_ok=True)
    attempt = len(list(dest.glob("executed*.py")))
    (
        dest / (f"executed_resume_{attempt}.py" if resuming else "executed.py")
    ).write_bytes(Path(__file__).read_bytes())
    dates, names = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    links = pl.read_parquet(identity["links"]["path"])
    old_terms = verified_action_terms_from_table(
        pl.read_parquet(
            root / manifest["tables"]["corporate_actions_verified_terms"]["path"]
        )
    )
    replacements = {
        (e["isin"], date.fromisoformat(e["effective_date"])): e for e in plan["scalars"]
    }
    combined, replaced = {}, []
    for term in old_terms:
        event = replacements.get((term.isin, term.effective_date))
        amended = term
        if event:
            q, cash = (
                event["shares_per_prior_share"],
                event["gross_cash_per_prior_share"],
            )
            assert term.shares_per_prior_share != q or term.cash_per_prior_share != cash
            amended = replace(
                term,
                action_type="dividend" if cash else "split",
                available_at=datetime.fromisoformat(event["first_known_at"]),
                announced_at=None,
                payment_date=date.fromisoformat(event["payment_date"])
                if event["payment_date"]
                else None,
                shares_per_prior_share=q,
                cash_per_prior_share=cash,
                source=plan["originals"][event["protocol"]]["pdf"]["path"],
                evidence="Previously visually qualified original " + event["protocol"],
                coverage_status="verified",
            )
            replaced.append(
                dict(
                    before={k: str(v) for k, v in asdict(term).items()},
                    after={k: str(v) for k, v in asdict(amended).items()},
                )
            )
        combined[(amended.isin, amended.effective_date, amended.sequence)] = amended
    assert len(replaced) == 2
    for term in verified_conversion_terms_from_links(links):
        key = term.isin, term.effective_date, term.sequence
        if key in combined:
            assert combined[key] == term
        else:
            combined[key] = term
    new_terms = tuple(combined.values())
    verified_action_terms_to_table(new_terms).write_parquet(
        dest / "corporate_actions_verified_terms.parquet"
    )
    write_json_atomic(dest / "scalar_replacements.json", replaced)
    # These two stale wealth factors were already rejected by sealed Round7.
    # Restore them ONLY in the audit control, to reproduce the inherited bytes.
    # Corrected paths use the accepted term table, in which they are absent.
    round7 = bound_json(manifest["metadata"]["round7_repair"]["source_audit"])
    rejected = [
        r
        for r in bound_json(round7["u2_events"])
        if r["date"] == "2020-03-18" and r["isin"] in ("BRENATACNOR0", "BRGUARACNOR4")
    ]
    assert len(rejected) == 2 and all(
        r["classification"] == "large_move_no_action" and not r["corroboration"]
        for r in rejected
    )
    legacy = []
    for row in rejected:
        day = date.fromisoformat(row["date"])
        assert not any(
            t.isin == row["isin"] and t.effective_date == day for t in new_terms
        )
        legacy.append(
            VerifiedActionTerm(
                action_type="split",
                isin=row["isin"],
                issuer_id=row["issuer"],
                effective_date=day,
                ex_date=day,
                payment_date=None,
                announced_at=None,
                available_at=datetime.combine(day, datetime.min.time(), timezone.utc),
                shares_per_prior_share=row["old_q"],
                cash_per_prior_share=0.0,
                currency="BRL",
                source=round7["u2_events"]["path"],
                evidence=row["old_evidence"],
                coverage_status="rejected_factor_audit_control_only",
            )
        )
    lineage_path = dest / "wealth_lineage_disposition.json"
    if not lineage_path.exists():
        write_json_atomic(
            lineage_path,
            dict(
                source=round7["u2_events"],
                records=rejected,
                disposition="Sealed Round7 rejects both March18 2020 factors as large_move_no_action. The inherited wealth arrays still contain them. Old-byte controls alone reinstate the printed old_q; corrected wealth removes them using already accepted q1/cash0. Earlier2010/2011 GPC dispositions lie outside the new recurrence windows; prior seeds remain frozen. No new source census or restored split claim.",
                frozen_before_corrected_outputs=True,
            ),
        )
    first_date = min(e["effective_date"] for e in [*plan["history"], *plan["scalars"]])
    start = int(np.searchsorted(dates, np.datetime64(first_date))) - 1
    days = dates[start:]
    quotes = pl.read_parquet(identity["normalized_quotes"]["path"]).filter(
        pl.col("trade_date") >= days[0].astype(object)
    )
    selected_names = sorted(set(quotes["isin"]))
    axes = [names.index(s) for s in selected_names]
    raw = np.full((4, len(days), len(axes)), np.nan, np.float64)
    ti = np.searchsorted(days, quotes["trade_date"].to_numpy())
    ni = np.array([selected_names.index(s) for s in quotes["isin"]])
    fields = ("open", "high", "low", "close")

    def old(key):
        return np.load(root / manifest["arrays"][key]["path"], mmap_mode="r")

    observed = old("observed")[start:][:, axes]
    required = np.zeros_like(observed)
    for event in [*plan["history"], *plan["scalars"]]:
        begin = max(
            0, int(np.searchsorted(days, np.datetime64(event["effective_date"]))) - 1
        )
        for name in (event["isin"], event.get("successor_isin", event["isin"])):
            required[begin:, selected_names.index(name)] = True
    for k, field in enumerate(fields):
        raw[k, ti, ni] = quotes[field + "_brl"].to_numpy()
        np.testing.assert_array_equal(
            raw[k].astype(np.float32)[observed & required],
            old("raw_" + field)[start:][:, axes][observed & required],
        )
    schedule = load_session_schedule(Path(plan["schedule"]["path"]))
    sessions = tuple(
        s for s in schedule if days[0] <= np.datetime64(s.trade_date) <= days[-1]
    )
    following = next(
        s.decision_at for s in schedule if np.datetime64(s.trade_date) > days[-1]
    )
    clocks = next_session_decision_cutoffs(sessions, following_decision_at=following)
    actions = [
        align_decision_known_action_terms(
            ts,
            days,
            selected_names,
            coverage_resolved=observed,
            decision_timestamps=clocks,
        )
        for ts in ((*old_terms, *legacy), new_terms)
    ]
    cases = [
        dict(
            id=e["id"],
            source=e["isin"],
            successor=e["successor_isin"],
            first=e["effective_date"],
        )
        for e in plan["history"]
    ]
    for event in plan["scalars"]:
        affected = next((c for c in cases if c["source"] == event["isin"]), None)
        if affected:
            affected["first"] = min(affected["first"], event["effective_date"])
        else:
            cases.append(
                dict(
                    id="GUAR_split",
                    source=event["isin"],
                    successor=event["isin"],
                    first=event["effective_date"],
                )
            )
    packed, reports, bases = {}, [], {}
    for case in cases:
        first = int(np.searchsorted(days, np.datetime64(case["first"])))
        a = selected_names.index(case["source"])
        seed = np.array(
            [old("shareholder_wealth_" + k)[start + first - 1, axes[a]] for k in fields]
        )
        seed_valid = bool(old("shareholder_wealth_valid")[start + first - 1, axes[a]])
        control, cv = resume(raw, observed, actions[0], first, a, seed, seed_valid)
        values, valid = resume(raw, observed, actions[1], first, a, seed, seed_valid)
        expected = np.array(
            [old("shareholder_wealth_" + k)[start:, axes[a]] for k in fields]
        )
        np.testing.assert_array_equal(control[:, first:], expected[:, first:])
        np.testing.assert_array_equal(
            cv[first:], old("shareholder_wealth_valid")[start + first :, axes[a]]
        )
        destinations = [(a, first)]
        if case["successor"] != case["source"]:
            event = next(e for e in plan["history"] if e["id"] == case["id"])
            effect = int(np.searchsorted(days, np.datetime64(event["effective_date"])))
            b = selected_names.index(case["successor"])
            bseed = np.array(
                [
                    old("shareholder_wealth_" + k)[start + effect - 1, axes[b]]
                    for k in fields
                ]
            )
            bvalid = bool(old("shareholder_wealth_valid")[start + effect - 1, axes[b]])
            standalone, sv = resume(raw, observed, actions[0], effect, b, bseed, bvalid)
            np.testing.assert_array_equal(
                standalone[:, effect:],
                np.array(
                    [
                        old("shareholder_wealth_" + k)[start + effect :, axes[b]]
                        for k in fields
                    ]
                ),
            )
            np.testing.assert_array_equal(
                sv[effect:], old("shareholder_wealth_valid")[start + effect :, axes[b]]
            )
            destinations.append((b, effect))
        changes = {}
        for axis, begin in destinations:
            for k, field in enumerate((*fields, "valid")):
                key = "shareholder_wealth_" + field
                after = valid[begin:] if field == "valid" else values[k, begin:]
                before = old(key)[start + begin :, axes[axis]]
                ix = np.flatnonzero(
                    ~((before == after) | (np.isnan(before) & np.isnan(after)))
                )
                packed.setdefault(key + "__indices", []).append(
                    np.column_stack((ix + start + begin, np.full(len(ix), axes[axis])))
                )
                packed.setdefault(key + "__values", []).append(after[ix])
                changes[selected_names[axis] + "/" + field] = len(ix)
        bases[case["id"] + "__wealth"], bases[case["id"] + "__valid"] = values, valid
        reports.append(
            dict(
                **case,
                control_cells=int(control[:, first:].size + cv[first:].size),
                changes=changes,
            )
        )
        np.savez_compressed(
            dest / (case["id"] + ".npz"),
            control=control,
            control_valid=cv,
            corrected=values,
            corrected_valid=valid,
            first=first,
            seed=seed,
        )
    np.savez_compressed(
        dest / "deltas.npz", **{k: np.concatenate(v) for k, v in packed.items()}
    )
    np.savez_compressed(
        dest / "source_basis.npz",
        raw=raw,
        observed=observed,
        dates=days,
        axes=axes,
        **bases,
    )
    report = dict(
        status="source_scalars_and_wealth_qualified_dependencies_pending",
        parent=plan["parent"],
        identity=run["scaling_data_identity"],
        terms=binding(dest / "corporate_actions_verified_terms.parquet"),
        deltas=binding(dest / "deltas.npz"),
        source_basis=binding(dest / "source_basis.npz"),
        scalar_replacements=binding(dest / "scalar_replacements.json"),
        cases=reports,
        seconds=perf_counter() - tick,
    )
    record(run, "scaling_data_wealth", dest / "manifest.json", report)
    print(json.dumps(dict(cases=reports, seconds=report["seconds"])))


if __name__ == "__main__":
    {"prepare": prepare, "identity": admit, "wealth": wealth}[sys.argv[1]]()
