"""Compose the evidenced added-period repairs and independent gross endpoints."""

import copy
from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.contract import HORIZONS
from brazil_rv.v2.corporate_actions import (
    align_action_payment_sessions,
    align_decision_known_action_terms,
    align_verified_action_terms,
    verified_action_terms_from_table,
)
from brazil_rv.v2.corporate_replay import apply_corporate_replay, load_corporate_replay
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.decision_clock import (
    load_session_schedule,
    next_session_decision_cutoffs,
)
from brazil_rv.v2.intraday_features import decision_action_boundaries
from brazil_rv.v2.targets import build_economic_multi_day_targets
from compose_surviving_store import changed, read_layers
from propagate_corporate_targets import FIELDS
from repair_scaling_inputs import PROJECT, context, record
from verify_corporate_replay import inputs_on_axes

LAYERS = (
    "identity",
    "wealth",
    "daily_qualification",
    "context",
    "market",
    "m1",
    "scalars",
    "sidecars",
)


def event_sets(run, root, dates, out, plan):
    original = bound_json(run["corporate_replay"])
    combined = copy.deepcopy(original)
    source_receipts = {r["path"]: r for r in original["sources"]}
    for key in ("scaling_expanded_source_plan", "scaling_expanded_later_source_plan"):
        receipt = bound_json(run[key])["terms"]
        other = bound_json(receipt)
        for category in ("share_distributions", "scalar_actions", "identity_actions"):
            field = "predecessor_isin" if category == "identity_actions" else "isin"
            prior = {
                (e[field], e["effective_date"]): e for e in combined.get(category, [])
            }
            for e in other.get(category, []):
                name = e[field], e["effective_date"]
                if name in prior:
                    assert prior[name] == e, (category, name)
                else:
                    combined.setdefault(category, []).append(e)
                    prior[name] = e
        source_receipts.update({r["path"]: r for r in other["sources"]})
    combined["sources"] = list(source_receipts.values())
    combined["scaling_composition"] = {
        "plan": run["scaling_data_plan"],
        "scope": "Union of separately qualified added-period terms; no new broker/custody/fraction assumptions. Original pending cases remain explicit.",
    }
    write_json_atomic(out / "account_terms.json", combined)
    new_receipt = binding(out / "account_terms.json")
    base = inputs_on_axes(
        Path(original["store"]["root"]), dates, np.arange(933), np.arange(len(dates))
    )
    results = []
    for receipt in (run["corporate_replay"], new_receipt):
        spec, calendar = load_corporate_replay(receipt["path"], receipt["sha256"])
        np.testing.assert_array_equal(dates, calendar)
        results.append(
            apply_corporate_replay(
                base, spec, dates, receipt["sha256"]
            ).share_distributions
        )
    expected = {(e["isin"], e["effective_date"]) for e in plan["distributions"]}
    names = np.load(root / "isin_index.npy").tolist()
    previous = {(e.source_index, e.effective_session): e for e in results[0]}
    for event in results[1]:
        key = event.source_index, event.effective_session
        if key in previous:
            assert replace(event, source=previous[key].source) == previous[key]
    added = {
        (names[e.source_index], str(dates[e.effective_session]))
        for e in results[1]
        if (e.source_index, e.effective_session) not in previous
    }
    assert added == expected
    return results, new_receipt


def main(mode):
    tick = perf_counter()
    run, plan, root, m, _ = context()
    out = Path(bound_json(run["scaling_data_workspace"])["root"]) / "composition"
    dates, names = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )

    def old(k):
        return np.load(root / m["arrays"][k]["path"], mmap_mode="r")

    if mode == "assemble":
        from compose_matched_event_store import assemble

        assemble(
            run,
            bound_json(run["scaling_data_identity"]),
            root,
            m,
            out,
            dates,
            names,
            PROJECT / "docs/v2_economic_data_scaling_run.json",
            tick,
            scaling=True,
        )
        return
    if mode == "qualify":
        qualify(run, plan, root, m, out, dates, names, tick)
        return
    assert mode == "targets"
    resume = out.exists()
    out.mkdir(exist_ok=True)
    assert not (out / "control_targets.npz").exists()
    (
        out
        / (
            ("executed_targets_" + binding(Path(__file__))["sha256"][:12] + ".py")
            if resume
            else "executed_targets.py"
        )
    ).write_bytes(Path(__file__).read_bytes())
    layers = [bound_json(run["scaling_data_" + k])["deltas"] for k in LAYERS]
    write_json_atomic(
        out / ("resume_plan.json" if resume else "plan.json"),
        dict(
            parent=plan["parent"],
            layers=layers,
            source_plan=run["scaling_data_plan"],
            evidence={k: v for k, v in run.items() if k.startswith("scaling_data_")},
            registration=binding(
                PROJECT / "research/preregistrations/v2_economic_data_scaling.md"
            ),
            contract="One complete store with four same-company histories,180restored eligible days, QGEP/Guararapes sourced scalars and rejected inherited2020wealth factors removed, four gross corporate claims and final dependent features/risks. Preserve all933/full3717/full60, schema and support thresholds; all earlier stores and fits remain immutable. Fibria cash revisions enter a gross label only when known by its endpoint, unremunerated/unreinvested; unknown Smiles fraction endpoints stay unsupported. No Linx BDR mark invented.",
            targets="Exact new-union parent controls, independent Decimal changed/new/unsupported endpoint walk and new integrated neutral projection. No prior source/feature campaign repeated.",
        ),
    )
    patches = read_layers(layers)

    def amended(k):
        a = old(k).copy()
        for ix, v in patches.get(k, []):
            a[tuple(ix.T)] = v
        return a

    active, sigma = amended("active"), amended("target_scale_sigma")
    paths = [
        root / m["tables"]["corporate_actions_verified_terms"]["path"],
        Path(bound_json(run["scaling_data_wealth"])["terms"]["path"]),
    ]
    terms = [verified_action_terms_from_table(pl.read_parquet(p)) for p in paths]
    (out / "corporate_actions_verified_terms.parquet").write_bytes(
        paths[1].read_bytes()
    )
    actions = [
        align_verified_action_terms(
            t, dates, names, coverage_resolved=old("action_session_resolved")
        )
        for t in terms
    ]
    ad = {}
    for k, a in (
        ("action_shares_per_prior_share", actions[1].shares_per_prior_share),
        ("action_cash_per_prior_share", actions[1].cash_per_prior_share),
        ("action_session_resolved", actions[1].session_resolved),
        ("action_has_action", actions[1].has_action),
        ("action_successor_index", actions[1].successor_index),
    ):
        value = a.astype(old(k).dtype)
        ix = np.argwhere(changed(value, old(k)))
        ad[k + "__indices"], ad[k + "__values"] = ix, value[tuple(ix.T)]
    scalar_names = [e["isin"] for e in plan["scalars"]]
    payments = align_action_payment_sessions(terms[1], dates, scalar_names)
    ix = np.array(
        [
            [
                int(np.searchsorted(dates, np.datetime64(e["effective_date"]))),
                names.index(e["isin"]),
            ]
            for e in plan["scalars"]
        ]
    )
    payment_values = np.array(
        [payments[t, j] for j, (t, _) in enumerate(ix)],
        dtype=old("action_payment_session").dtype,
    )
    use = payment_values != old("action_payment_session")[tuple(ix.T)]
    ad["action_payment_session__indices"], ad["action_payment_session__values"] = (
        ix[use],
        payment_values[use],
    )
    schedule = load_session_schedule(Path(plan["schedule"]["path"]))
    sessions = tuple(
        s for s in schedule if dates[0] <= np.datetime64(s.trade_date) <= dates[-1]
    )
    following = next(
        s.decision_at for s in schedule if np.datetime64(s.trade_date) > dates[-1]
    )
    cutoffs = next_session_decision_cutoffs(sessions, following_decision_at=following)
    boundaries = []
    for t, n in ix:
        window = slice(t - 2, t + 3)
        results = []
        for current in terms:
            known = align_decision_known_action_terms(
                current,
                dates[window],
                names,
                coverage_resolved=old("observed")[window],
                decision_timestamps=cutoffs[t - 2 : t + 3],
            )
            result = decision_action_boundaries(
                old("raw_open")[window],
                old("raw_close")[window],
                old("observed")[window],
                known.session_resolved,
                current,
                sessions[t - 2 : t + 3],
                names,
            )
            results.append(result[2, n])
        assert results[0] == old("intraday_unit_or_unresolved_boundary_mask")[t, n]
        boundaries.append(results[1])
    values = np.array(boundaries, bool)
    use = values != old("intraday_unit_or_unresolved_boundary_mask")[tuple(ix.T)]
    (
        ad["intraday_unit_or_unresolved_boundary_mask__indices"],
        ad["intraday_unit_or_unresolved_boundary_mask__values"],
    ) = ix[use], values[use]
    write_json_atomic(
        out / "scalar_settlement_clocks.json",
        dict(
            rows=ix.tolist(),
            payments=payment_values.tolist(),
            boundary_control_exact=True,
            boundaries=values.tolist(),
            no_assigned_intraday_observations_at_scalar_events=True,
        ),
    )
    np.savez_compressed(out / "action_deltas.npz", **ad)
    rows = set()
    for key in ("active", "target_scale_sigma"):
        for ix, _ in patches[key]:
            rows.update(ix[:, 0].tolist())
    for e in [*plan["history"], *plan["scalars"], *plan["distributions"]]:
        t = int(np.searchsorted(dates, np.datetime64(e["effective_date"])))
        rows.update(range(t - max(HORIZONS), t + 1))
    rows = np.array(sorted(rows))
    np.save(out / "target_rows.npy", rows)
    events, candidate = event_sets(run, root, dates, out, plan)
    close = np.round(old("raw_close").astype(float), 2)
    outputs = []
    for label, action, elig, risk, event in zip(
        ("control", "corrected"),
        actions,
        (old("active"), active),
        (old("target_scale_sigma"), sigma),
        events,
        strict=True,
    ):
        result = build_economic_multi_day_targets(
            close,
            old("observed"),
            elig,
            risk,
            action,
            source_rows=rows,
            share_distributions=event,
        )
        values = {k: getattr(result, f) for f, k in FIELDS.items()}
        np.savez_compressed(out / (label + "_targets.npz"), date_indices=rows, **values)
        if label == "control":
            checks = [
                dict(
                    key=k, cells=a.size, mismatches=int(changed(a, old(k)[rows]).sum())
                )
                for k, a in values.items()
            ]
            write_json_atomic(out / "controls.json", checks)
            assert not any(c["mismatches"] for c in checks), checks
        outputs.append(values)
    deltas, effects = {}, {}
    for key, a in outputs[1].items():
        ix = np.argwhere(changed(a, old(key)[rows]))
        values = a[tuple(ix.T)]
        ix[:, 0] = rows[ix[:, 0]]
        deltas[key + "__indices"], deltas[key + "__values"] = ix, values
        effects[key] = len(ix)
    entry = old("entry_fill_allowed").copy()
    barriers = []
    for e in [*plan["history"], *plan["distributions"]]:
        n = names.index(e["isin"])
        start = int(np.searchsorted(dates, np.datetime64(e["effective_date"])))
        assert not old("observed")[start:, n].any(), e
        entry[start:, n] = False
        barriers.append(
            dict(
                isin=e["isin"],
                start=str(dates[start]),
                stop=None,
                eligible_unquoted=int(active[start:, n].sum()),
            )
        )
    ix = np.argwhere(entry != old("entry_fill_allowed"))
    deltas["entry_fill_allowed__indices"], deltas["entry_fill_allowed__values"] = (
        ix,
        entry[tuple(ix.T)],
    )
    write_json_atomic(out / "entry_barriers.json", barriers)
    np.savez_compressed(out / "target_deltas.npz", **deltas)
    result = dict(
        status="new_targets_pending_independent_endpoints_and_complete_store",
        plan=binding(out / "plan.json"),
        controls=binding(out / "controls.json"),
        deltas=binding(out / "target_deltas.npz"),
        action_deltas=binding(out / "action_deltas.npz"),
        rows=binding(out / "target_rows.npy"),
        account_terms=candidate,
        effects=effects,
        target_dates=len(rows),
        entry_changes=len(ix),
        seconds=perf_counter() - tick,
    )
    record(run, "scaling_data_targets", out / "targets.json", result)
    print(json.dumps(result), flush=True)


def qualify(run, plan, root, m, folder, dates, names, tick):
    out = folder / "endpoint_qualification"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    target = bound_json(run["scaling_data_targets"])
    with np.load(folder / "corrected_targets.npz") as z:
        rows = z["date_indices"].copy()
        after = {k: z[k].copy() for k in z.files if k != "date_indices"}
    active = np.load(root / "active.npy").copy()
    with np.load(bound_json(run["scaling_data_identity"])["deltas"]["path"]) as z:
        active[tuple(z["active__indices"].T)] = z["active__values"]
    terms = verified_action_terms_from_table(
        pl.read_parquet(folder / "corporate_actions_verified_terms.parquet")
    )
    actions = align_verified_action_terms(
        terms,
        dates,
        names,
        coverage_resolved=np.load(root / "action_session_resolved.npy"),
    )
    receipt = target["account_terms"]
    spec, _ = load_corporate_replay(receipt["path"], receipt["sha256"])
    base = inputs_on_axes(
        Path(spec["store"]["root"]), dates, np.arange(933), np.arange(len(dates))
    )
    events = apply_corporate_replay(
        base, spec, dates, receipt["sha256"]
    ).share_distributions
    event_map = {(e.effective_session, e.source_index): e for e in events}
    close = np.round(np.load(root / "raw_close.npy").astype(float), 2)
    observed = np.load(root / "observed.npy")
    before_valid = np.load(root / "target_shareholder_valid.npy", mmap_mode="r")[rows]
    before_returns = np.load(
        root / "target_shareholder_simple_return.npy", mmap_mode="r"
    )[rows]
    scope = set(
        map(
            tuple,
            np.argwhere(
                (before_valid != after["target_shareholder_valid"])
                | changed(before_returns, after["target_shareholder_simple_return"])
            ),
        )
    )
    for e in [*plan["history"], *plan["scalars"], *plan["distributions"]]:
        effect = int(np.searchsorted(dates, np.datetime64(e["effective_date"])))
        name = names.index(e["isin"])
        for t in range(effect - max(HORIZONS), effect):
            if active[t, name] and observed[t, name]:
                for h, horizon in enumerate(HORIZONS):
                    if t < effect <= t + horizon:
                        scope.add((int(np.searchsorted(rows, t)), name, h))
    endpoints = []
    for local, name, h in sorted(scope):
        day = int(rows[local])
        end = day + HORIZONS[h]
        ok = bool(active[day, name] and observed[day, name] and end < len(dates))
        holdings, cash = {name: Decimal(1)}, Decimal(0)
        if ok:
            for t in range(day + 1, end + 1):
                nxt = {}
                for claim, q in holdings.items():
                    event = event_map.get((t, claim))
                    if event:
                        for leg in event.legs:
                            auction = leg.fractional_auction
                            if auction is not None and end >= (
                                leg.delivery_session
                                if auction.available_session is None
                                else auction.available_session
                            ):
                                ok = False
                            j = leg.successor_index
                            nxt[j] = nxt.get(j, Decimal(0)) + q * Decimal(
                                str(leg.shares_per_prior_share)
                            )
                        # Revalue this original entitlement at the endpoint's information set.
                        known = [(t, event.cash_per_prior_share), *event.cash_values]
                        amount = max(
                            (item for item in known if item[0] <= end),
                            key=lambda item: item[0],
                        )[1]
                        cash += q * Decimal(str(amount))
                    elif actions.session_resolved[t, claim]:
                        cash += q * Decimal(str(actions.cash_per_prior_share[t, claim]))
                        j = int(actions.successor_index[t, claim])
                        nxt[j] = nxt.get(j, Decimal(0)) + q * Decimal(
                            str(actions.shares_per_prior_share[t, claim])
                        )
                    else:
                        ok = False
                holdings = {j: q for j, q in nxt.items() if q}
            ok &= all(
                observed[end, j] and np.isfinite(close[end, j]) and close[end, j] > 0
                for j in holdings
            )
        assert bool(after["target_shareholder_valid"][local, name, h]) == ok, (
            dates[day],
            names[name],
            h,
            ok,
        )
        entry = dict(
            date=str(dates[day]), isin=names[name], horizon=HORIZONS[h], valid=bool(ok)
        )
        if ok:
            equity = sum(
                (q * Decimal(str(close[end, j])) for j, q in holdings.items()),
                Decimal(0),
            )
            price = Decimal(str(close[day, name]))
            for key, x in (
                ("target_terminal_wealth", (equity + cash) / price),
                ("target_shareholder_simple_return", (equity + cash) / price - 1),
                ("target_price_simple_return", equity / price - 1),
            ):
                actual, expected = after[key][local, name, h], np.float32(float(x))
                if actual != expected:
                    error = abs(float(actual) - float(expected))
                    bound = 32 * np.finfo(float).eps * max(1, abs(float(x)))
                    assert error <= bound, (entry, key, actual, expected, error, bound)
                    entry.setdefault("float64_roundoff", []).append(
                        dict(
                            field=key,
                            actual=float(actual),
                            decimal_float32=float(expected),
                            error=error,
                            bound=bound,
                        )
                    )
            entry.update(equity=str(equity), cash=str(cash), entry=str(price))
        endpoints.append(entry)
    write_json_atomic(out / "endpoints.json", endpoints)
    result = dict(
        status="qualified_new_primary_endpoints",
        input=run["scaling_data_targets"],
        checked=len(endpoints),
        valid=sum(r["valid"] for r in endpoints),
        gains=int((after["target_shareholder_valid"] & ~before_valid).sum()),
        losses=int((before_valid & ~after["target_shareholder_valid"]).sum()),
        records=binding(out / "endpoints.json"),
        seconds=perf_counter() - tick,
    )
    record(run, "scaling_data_targets_qualification", out / "manifest.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
