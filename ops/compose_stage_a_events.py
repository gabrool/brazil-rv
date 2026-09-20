"""Compose qualified event terms and attribute their remaining data consequences."""

from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.contract import HORIZONS
from brazil_rv.v2.corporate_actions import (
    align_verified_action_terms,
    verified_action_terms_from_table,
)
from brazil_rv.v2.corporate_replay import apply_corporate_replay, load_corporate_replay
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.targets import build_economic_multi_day_targets
from propagate_corporate_targets import FIELDS
from verify_corporate_replay import inputs_on_axes

PROJECT = Path(__file__).resolve().parents[1]
NEW_NAMES = ("BRNATUACNOR6", "BRALSCACNOR0", "BRSOMAACNOR3", "BRENATACNOR0")


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text(encoding="utf8"))
    root = Path(run["root"]) / "event_composition"
    root.mkdir(exist_ok=False)
    (root / "executed.py").write_bytes(Path(__file__).read_bytes())
    for relative in (
        "research/src/brazil_rv/v2/targets.py",
        "research/src/brazil_rv/v2/corporate_replay.py",
        "ops/propagate_corporate_targets.py",
        "ops/verify_corporate_replay.py",
    ):
        (root / Path(relative).name).write_bytes((PROJECT / relative).read_bytes())
    parent = bound_json(run["economic_refit_inputs"])["store"]
    store = Path(parent["root"])
    m = bound_json(
        dict(path=str(store / "manifest.json"), sha256=parent["manifest_sha256"])
    )
    terms, calendar = load_corporate_replay(
        **{
            "path": run["enat_settlement_terms"]["path"],
            "expected_sha256": run["enat_settlement_terms"]["sha256"],
        }
    )
    dates = np.load(store / "date_index.npy")
    isins = np.load(store / "isin_index.npy").tolist()
    np.testing.assert_array_equal(dates, calendar)
    assert len(isins) == 933 and len(dates) == 3717 and str(dates[-1]) == "2024-12-30"
    evidence = {
        k: run[k]
        for k in (
            "enat_settlement_terms",
            "remaining_held_source_audit",
            "opening_claim_identity",
            "natura_settlement_runtime_qualification",
            "enat_settlement_qualification",
            "corporate_custody_qualification",
            "spot_invoice_qualification",
            "economic_refit_inputs",
            "stage_a_closeout_plan",
        )
    }
    # This plan precedes target differences and scope counts; no model outcomes.
    write_json_atomic(
        root / "plan.json",
        dict(
            evidence=evidence,
            parent=parent,
            event_contract="Reuse all seven ENAT-composed distributions, Natura gross/net/credit terms and Cielo shareholder/loan cash unchanged; no new numeric account hypothesis.",
            data_scope=list(NEW_NAMES),
            targets="Gross continuous shareholder entitlement at observed own-successor endpoints; cash unremunerated. ENAT unknown-auction endpoints at/after physical credit remain unsupported. No lot-specific unit labels.",
            predictors="Preserve all eligibility, own-name OHLC/wealth, risks, native/auxiliary features and full history. These economic exchanges do not authorize predecessor history or issuer/loan aliases. Closing entitlements remain audit-only, never new quotes or predictors.",
            attribution="Source-only target and entry-permission amendment on accepted Natura coordinates. No accounting return, calibration, forward, optimizer, model fit or book replay.",
            additional_identity_scope="SSBR/ALSO and ARZZ/AZZA rename implications remain separately attributable; no identity/history inference from the four acquired-source exchanges.",
            controls="Reconstruct only new crossing-date full-name target rows against accepted store; reuse all earlier completed scopes. Decimal every new source endpoint, future-price prefix, unchanged prior events and loader identity.",
            admission="Candidate composition and data attribution, not integrated Stage A or final refit-store acceptance.",
        ),
    )
    primary = deepcopy(terms)
    primary["status"] = (
        "composed_qualified_primary_terms_pending_integrated_stage_a_admission"
    )
    primary["supersedes"] = run["enat_settlement_terms"]
    primary["composition_evidence"] = evidence
    primary["pending_cases"] = [
        "Two old clearing dates: cash/physical delivery and rent clocks distinct",
        "ALSC physical credit/exact net cash unknown; whole signed claim remains locked",
        "ENAT fraction auction unknown; signed residual stays locked",
        "Cielo held-loan cent and opposing spot-fill daytrade bounds only on actual exposure",
        "Separate succession data/identity admission and integrated Stage A",
    ]
    for event in primary["share_distributions"]:
        if event["isin"] in ("BRNATUACNOR6", "BRSOMAACNOR3"):
            event["unresolved"] = (
                "Event-specific lender terms remain unknown; the explicit primary loan convention and its qualified engineering bounds are hypotheses, not sourced instructions."
            )
        if event["isin"] == "BRALSCACNOR0":
            event["unresolved"] = (
                "Original732740 reports Jan15 2020 auction, first usable Jan31: gross54.26688776859/share, net unspecified fees, payment within seven following business days. Physical credit/exact net receipt remain unknown; do not unlock whole claims or treat gross as net."
            )
            event["auction_source_evidence"] = run["remaining_held_source_audit"]
    write_json_atomic(root / "primary_terms.json", primary)
    primary_binding = binding(root / "primary_terms.json")
    base = inputs_on_axes(
        Path(terms["store"]["root"]), dates, np.arange(933), np.arange(len(dates))
    )
    old_account = apply_corporate_replay(
        base, terms, dates, run["enat_settlement_terms"]["sha256"]
    )
    composed = apply_corporate_replay(base, primary, dates, primary_binding["sha256"])
    for key in (
        "action_shares_per_prior_share",
        "action_cash_per_prior_share",
        "action_session_resolved",
        "action_has_action",
        "action_payment_session",
    ):
        np.testing.assert_array_equal(getattr(old_account, key), getattr(composed, key))
    # Only provenance strings differ in immutable event objects.
    for old_event, new_event in zip(
        old_account.share_distributions, composed.share_distributions, strict=True
    ):
        assert replace(old_event, source=new_event.source) == new_event
    for key in (
        "scores",
        "scaled_midrank_targets",
        "active",
        "raw_close",
        "action_successor_index",
        "prior_feature_values",
    ):
        assert getattr(composed, key) is getattr(base, key)

    def old(key):
        return np.load(store / m["arrays"][key]["path"], mmap_mode="r")

    events = composed.share_distributions
    new_events = tuple(e for e in events if isins[e.source_index] in NEW_NAMES)
    prior_events = tuple(e for e in events if isins[e.source_index] not in NEW_NAMES)
    rows = np.array(
        sorted(
            {
                r
                for e in new_events
                for r in range(
                    e.effective_session - max(HORIZONS), e.effective_session + 1
                )
            }
        )
    )
    original = verified_action_terms_from_table(
        pl.read_parquet(store / m["tables"]["corporate_actions_verified_terms"]["path"])
    )
    actions = align_verified_action_terms(
        original, dates, isins, coverage_resolved=old("action_session_resolved")
    )
    close = np.round(np.asarray(old("raw_close"), np.float64), 2)
    observed, active, sigma = (
        old(k) for k in ("observed", "active", "target_scale_sigma")
    )
    args = (close, observed, active, sigma, actions)
    before = build_economic_multi_day_targets(
        *args, source_rows=rows, share_distributions=prior_events
    )
    control_cells = 0
    for field, key in FIELDS.items():
        np.testing.assert_array_equal(
            getattr(before, field), old(key)[rows], err_msg=key
        )
        control_cells += getattr(before, field).size
    after = build_economic_multi_day_targets(
        *args, source_rows=rows, share_distributions=events
    )
    oracles, scope = [], []
    entry_ix, claims = [], []
    for event in new_events:
        n, effect = event.source_index, event.effective_session
        known = max(effect, event.available_session)
        post = np.flatnonzero(active[known:, n]) + known
        assert not observed[known:, n].any()
        assert not old("shareholder_wealth_valid")[post, n].any()
        entry_ix.extend(
            (int(t), n)
            for t in range(known, len(dates))
            if old("entry_fill_allowed")[t, n]
        )
        for p, day in enumerate(rows):
            for h, horizon in enumerate(HORIZONS):
                end = int(day + horizon)
                if (
                    not day < effect <= end
                    or not active[day, n]
                    or not observed[day, n]
                ):
                    continue
                leg = event.legs[0]
                assert len(event.legs) == 1
                j = leg.successor_index
                assert np.all(actions.shares_per_prior_share[day + 1 : effect, n] == 1)
                assert np.all(actions.cash_per_prior_share[day + 1 : effect, n] == 0)
                auction = leg.fractional_auction
                barrier = (
                    None
                    if auction is None
                    else (
                        leg.delivery_session
                        if auction.available_session is None
                        else auction.available_session
                    )
                )
                valid = bool(
                    observed[end, j]
                    and np.all(actions.session_resolved[effect + 1 : end + 1, j])
                    and (barrier is None or end < barrier)
                )
                assert bool(after.shareholder_valid[p, n, h]) == valid
                row = dict(
                    isin=isins[n],
                    entry=str(dates[day]),
                    endpoint=str(dates[end]),
                    horizon=horizon,
                    valid=valid,
                )
                if valid:
                    qty, cash = (
                        Decimal(str(leg.shares_per_prior_share)),
                        Decimal(str(event.cash_per_prior_share)),
                    )
                    for t in range(effect + 1, end + 1):
                        assert actions.successor_index[t, j] == j
                        cash += qty * Decimal(str(actions.cash_per_prior_share[t, j]))
                        qty *= Decimal(str(actions.shares_per_prior_share[t, j]))
                    equity = qty * Decimal(str(close[end, j]))
                    entry = Decimal(str(close[day, n]))
                    for field, value in (
                        ("terminal_wealth", (equity + cash) / entry),
                        ("shareholder_simple_return", (equity + cash) / entry - 1),
                        ("price_simple_return", equity / entry - 1),
                    ):
                        np.testing.assert_array_equal(
                            getattr(after, field)[p, n, h], np.float32(float(value))
                        )
                    row.update(
                        quantity=str(qty),
                        cash=str(cash),
                        equity=str(equity),
                        entry_price=str(entry),
                    )
                else:
                    row["reason"] = (
                        "unknown-auction unit endpoint"
                        if barrier is not None and end >= barrier
                        else "missing observed successor endpoint or action support"
                    )
                oracles.append(row)
        for t in post:
            leg = event.legs[0]
            j = leg.successor_index
            # Exact continuous claim close only when an own successor print is observed.
            # ENAT post-delivery fractions are lot-dependent: omit unit marks there.
            a = leg.fractional_auction
            before_auction = a is None or t < (
                leg.delivery_session
                if a.available_session is None
                else a.available_session
            )
            usable = bool(observed[t, j] and before_auction)
            if usable:
                assert np.all(
                    actions.shares_per_prior_share[effect + 1 : t + 1, j] == 1
                )
                assert np.all(actions.cash_per_prior_share[effect + 1 : t + 1, j] == 0)
            claims.append(
                dict(
                    date=str(dates[t]),
                    isin=isins[n],
                    successor=isins[j],
                    observed_successor=bool(observed[t, j]),
                    exact_continuous_unit_mark=usable,
                    claim_close_brl=None
                    if not usable
                    else float(
                        Decimal(str(leg.shares_per_prior_share))
                        * Decimal(str(close[t, j]))
                        + Decimal(str(event.cash_per_prior_share))
                    ),
                    mark_available_next_decision=str(dates[t + 1]),
                    physical_credit_known=leg.delivery_session is not None,
                    source_quote_observed=False,
                )
            )
        scope.append(
            dict(
                isin=isins[n],
                effective=str(dates[effect]),
                eligible_source_dates=[str(dates[t]) for t in post],
                eligibility_unchanged=True,
                source_wealth_observed_after_effect=False,
                successor_history_inherited=False,
            )
        )
    deltas, effects = {}, {}
    for field, key in FIELDS.items():
        a, b = getattr(before, field), getattr(after, field)
        equal = (
            (a == b) | (np.isnan(a) & np.isnan(b)) if a.dtype.kind == "f" else a == b
        )
        ix = np.argwhere(~equal)
        global_ix = ix.copy()
        global_ix[:, 0] = rows[ix[:, 0]]
        deltas[key + "__indices"], deltas[key + "__values"] = global_ix, b[tuple(ix.T)]
        effects[key] = dict(changed=len(ix))
        if a.dtype.kind == "b":
            effects[key].update(gained=int((b & ~a).sum()), lost=int((a & ~b).sum()))
        np.save(root / (key + "_rows.npy"), b)
    ix = np.asarray(entry_ix, dtype=np.int64).reshape(-1, 2)
    deltas["entry_fill_allowed__indices"] = ix
    deltas["entry_fill_allowed__values"] = np.zeros(len(ix), bool)
    assert not observed[tuple(ix.T)].any()
    np.save(root / "rows.npy", rows)
    np.savez_compressed(root / "deltas.npz", **deltas)
    write_json_atomic(root / "endpoint_oracles.json", oracles)
    write_json_atomic(root / "source_scope.json", scope)
    write_json_atomic(root / "claim_close_rows.json", claims)
    future_cells = 0
    for cutoff in (
        new_events[0].effective_session + 3,
        new_events[-1].effective_session + 3,
    ):
        prefix_rows = rows[rows + max(HORIZONS) <= cutoff]
        mutated = close.copy()
        mutated[cutoff + 1 :] *= 7
        check = build_economic_multi_day_targets(
            mutated, *args[1:], source_rows=prefix_rows, share_distributions=events
        )
        take = np.searchsorted(rows, prefix_rows)
        for field in FIELDS:
            np.testing.assert_array_equal(
                getattr(check, field), getattr(after, field)[take]
            )
            future_cells += getattr(check, field).size
    report = dict(
        status="composed_primary_and_bounded_data_attribution_pending_store_and_integrated_admission",
        plan=binding(root / "plan.json"),
        parent=parent,
        primary_terms=primary_binding,
        source_terms_numeric_account_identity=True,
        previous_terms=run["enat_settlement_terms"],
        rows=binding(root / "rows.npy"),
        deltas=binding(root / "deltas.npz"),
        target_arrays={
            key: binding(root / (key + "_rows.npy")) for key in FIELDS.values()
        },
        endpoint_oracles=binding(root / "endpoint_oracles.json"),
        source_scope=binding(root / "source_scope.json"),
        claims=binding(root / "claim_close_rows.json"),
        control_cells=control_cells,
        target_dates=len(rows),
        effects=effects,
        new_source_entries_closed=len(ix),
        eligible_new_entries_closed=int(active[tuple(ix.T)].sum()),
        endpoint_checks=len(oracles),
        valid_endpoint_checks=sum(x["valid"] for x in oracles),
        future_prefix_cells=future_cells,
        feature_risk_wealth_history_arrays_changed=False,
        model_forward_or_books=False,
        seconds=perf_counter() - tick,
    )
    write_json_atomic(root / "manifest.json", report)
    run["event_source_composition"] = binding(root / "manifest.json")
    run["composed_primary_event_terms"] = primary_binding
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "control_cells",
                    "target_dates",
                    "effects",
                    "new_source_entries_closed",
                    "eligible_new_entries_closed",
                    "endpoint_checks",
                    "valid_endpoint_checks",
                    "seconds",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
