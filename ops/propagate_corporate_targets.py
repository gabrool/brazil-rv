"""Bounded sourced corporate basket endpoints on the repaired full name axis."""

from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.execution.share_distributions import (
    FractionAuction,
    ShareDelivery,
    ShareDistribution,
)
from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.corporate_actions import (
    align_verified_action_terms,
    verified_action_terms_from_table,
    verified_conversion_terms_from_links,
)
from brazil_rv.v2.contract import HORIZONS
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.targets import build_economic_multi_day_targets

PROJECT = Path(__file__).resolve().parents[1]
FIELDS = {
    "primary": "target_primary",
    "primary_valid": "target_valid",
    "normalized_residual": "target_normalized_residual",
    "normalized_cross_section_valid": "target_normalized_cross_section_valid",
    "shareholder_midrank": "target_shareholder_midrank",
    "shareholder_valid": "target_shareholder_valid",
    "shareholder_simple_return": "target_shareholder_simple_return",
    "terminal_wealth": "target_terminal_wealth",
    "terminal_loss": "target_terminal_loss",
    "price_midrank": "target_price_midrank",
    "price_valid": "target_price_valid",
    "price_simple_return": "target_price_simple_return",
}


def main():
    start = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    history = bound_json(run["rename_history_propagation"])
    terms = bound_json(run["corporate_replay"])
    parent = json.loads((PROJECT / "docs/v2_data_inputs.json").read_text())["store"]
    store = Path(parent["root"])
    manifest = bound_json(
        {"path": str(store / "manifest.json"), "sha256": parent["manifest_sha256"]}
    )
    output = Path(run["root"]) / "corporate_targets" / "qualified"
    output.mkdir(exist_ok=False)
    (output / "executed_reproducer.py").write_bytes(Path(__file__).read_bytes())
    dates, isins = (
        np.load(store / "date_index.npy"),
        np.load(store / "isin_index.npy").tolist(),
    )
    assert dates[-1] == np.datetime64("2024-12-30")

    def index(day):
        t = int(np.searchsorted(dates, np.datetime64(day)))
        assert dates[t] == np.datetime64(day)
        return t

    def previous(key):
        source = history["arrays"].get(key)
        return np.load(
            source["path"] if source else store / manifest["arrays"][key]["path"],
            mmap_mode="r",
        )

    events = []
    for event in terms["share_distributions"]:
        legs = []
        for leg in event["legs"]:
            auction = leg.get("fractional_auction")
            legs.append(
                ShareDelivery(
                    isins.index(leg["successor_isin"]),
                    leg["shares_per_prior_share"],
                    index(leg["delivery_date"]),
                    None
                    if auction is None
                    else FractionAuction(
                        index(auction["available_date"]),
                        auction["cash_per_share"],
                        index(auction["payment_date"]),
                    ),
                    leg["loan_principal_fraction"],
                )
            )
        events.append(
            ShareDistribution(
                isins.index(event["isin"]),
                index(event["effective_date"]),
                index(event["available_date"]),
                tuple(legs),
                run["corporate_replay"]["sha256"],
                event["cash_per_prior_share"],
                None if event["payment_date"] is None else index(event["payment_date"]),
            )
        )
    horizons = HORIZONS
    scope = set()
    for event in events:
        scope.update(
            range(event.effective_session - max(horizons), event.effective_session + 1)
        )
    for event in terms["cash_cancellations"]:
        scope.update(
            range(
                index(event["recognition_date"]) - max(horizons),
                index(event["recognition_date"]) + 1,
            )
        )
    rows = np.array(sorted(scope))
    close = np.round(np.asarray(previous("raw_close"), dtype=np.float64), 2)
    observed, active, sigma = [
        previous(k) for k in ["observed", "active", "target_scale_sigma"]
    ]
    # Target assembly consumes original Float64 terms before the store downcast.
    # Reusing the stored float32 q/d changes some cross-sectional ties.
    term_path = store / manifest["tables"]["corporate_actions_verified_terms"]["path"]
    original_terms = verified_action_terms_from_table(pl.read_parquet(term_path))
    conversions = verified_conversion_terms_from_links(
        pl.read_parquet(history["links"]["path"])
    )
    actions = align_verified_action_terms(
        (*original_terms, *conversions),
        dates,
        isins,
        coverage_resolved=previous("action_session_resolved"),
    )
    args = (close, observed, active, sigma)
    control = build_economic_multi_day_targets(*args, actions, source_rows=rows)
    control_cells = 0
    for field, key in FIELDS.items():
        np.testing.assert_array_equal(getattr(control, field), previous(key)[rows])
        control_cells += getattr(control, field).size
    q, cash, resolved = [
        np.array(v, copy=True)
        for v in [
            actions.shares_per_prior_share,
            actions.cash_per_prior_share,
            actions.session_resolved,
        ]
    ]
    for event in terms["cash_cancellations"]:
        n, t = isins.index(event["isin"]), index(event["recognition_date"])
        resolved[index(event["coverage_start"]) :, n] = True
        q[t, n], cash[t, n] = 0, event["cash_per_share"]
    corrected = replace(
        actions,
        shares_per_prior_share=q,
        cash_per_prior_share=cash,
        session_resolved=resolved,
    )
    result = build_economic_multi_day_targets(
        *args, corrected, source_rows=rows, share_distributions=tuple(events)
    )
    # Independent scalar closed form on each actual pre-conversion entry/horizon.
    # Subsequent successor cash/q terms enter separately; no basket helper reuse.
    oracle, cases = [], []
    for event in events:
        n, t = event.source_index, event.effective_session
        checked = 0
        for row, day in enumerate(rows):
            if not active[day, n] or not observed[day, n]:
                continue
            for hidx, horizon in enumerate(horizons):
                end = day + horizon
                if not day < t <= end < len(dates):
                    continue
                assert np.all(actions.shares_per_prior_share[day + 1 : t, n] == 1)
                assert np.all(actions.cash_per_prior_share[day + 1 : t, n] == 0)
                assert np.all(actions.session_resolved[day + 1 : t, n])
                terminal, equity = event.cash_per_prior_share, 0.0
                valid = True
                for leg in event.legs:
                    j = leg.successor_index
                    assert (
                        leg.fractional_auction is None
                        or end < leg.fractional_auction.available_session
                    )
                    assert np.all(actions.successor_index[t + 1 : end + 1, j] == j)
                    valid &= bool(
                        observed[end, j]
                        and np.all(actions.session_resolved[t + 1 : end + 1, j])
                    )
                    quantity = leg.shares_per_prior_share
                    for s in range(t + 1, end + 1):
                        terminal += quantity * actions.cash_per_prior_share[s, j]
                        quantity *= actions.shares_per_prior_share[s, j]
                    equity += quantity * close[end, j]
                assert bool(result.shareholder_valid[row, n, hidx]) == valid
                if valid:
                    wealth = (terminal + equity) / close[day, n]
                    np.testing.assert_array_equal(
                        result.terminal_wealth[row, n, hidx], np.float32(wealth)
                    )
                    np.testing.assert_array_equal(
                        result.price_simple_return[row, n, hidx],
                        np.float32(equity / close[day, n] - 1),
                    )
                    oracle.append(
                        {
                            "isin": isins[n],
                            "entry": str(dates[day]),
                            "horizon": horizon,
                            "exit": str(dates[end]),
                            "entry_price": float(close[day, n]),
                            "cash": float(terminal),
                            "share_value": float(equity),
                            "wealth": float(wealth),
                        }
                    )
                    checked += 1
        cases.append(
            {
                "isin": isins[n],
                "effective": str(dates[t]),
                "valid_closed_form_endpoints": checked,
                "eligible_after_effect": int(active[t:, n].sum()),
                "legs": len(event.legs),
            }
        )
    deltas, effects = {}, {}
    for field, key in FIELDS.items():
        before, after = getattr(control, field), getattr(result, field)
        same = (
            (before == after) | (np.isnan(before) & np.isnan(after))
            if before.dtype.kind == "f"
            else before == after
        )
        local = np.argwhere(~same)
        global_ix = local.copy()
        global_ix[:, 0] = rows[local[:, 0]]
        deltas[key + "__indices"] = global_ix
        deltas[key + "__values"] = after[tuple(local.T)]
        np.save(output / (key + "_rows.npy"), after)
        effect = {"changed_cells": len(local)}
        if after.dtype.kind == "b":
            effect.update(
                gained=int((after & ~before).sum()), lost=int((before & ~after).sum())
            )
        effects[key] = effect
    np.save(output / "rows.npy", rows)
    np.savez_compressed(output / "deltas.npz", **deltas)
    write_json_atomic(output / "endpoint_oracle.json", oracle)
    # Prefix and irrelevant future price mutation; only crossing labels may move.
    mutated = close.copy()
    cutoff = events[0].effective_session + max(horizons)
    mutated[cutoff + 1 :] *= 7
    prefix_rows = rows[rows + max(horizons) <= cutoff]
    prefix = build_economic_multi_day_targets(
        mutated,
        observed,
        active,
        sigma,
        corrected,
        source_rows=prefix_rows,
        share_distributions=tuple(events),
    )
    slots = np.searchsorted(rows, prefix_rows)
    for field in FIELDS:
        np.testing.assert_array_equal(
            getattr(prefix, field), getattr(result, field)[slots]
        )
    receipt = {
        "status": "bounded_corporate_target_endpoints_verified_combined_store_pending",
        "parent": parent,
        "history": run["rename_history_propagation"],
        "corporate_terms": run["corporate_replay"],
        "control_cells": control_cells,
        "dates": len(rows),
        "names": len(isins),
        "horizons": horizons,
        "effects": effects,
        "cases": cases,
        "oracle": binding(output / "endpoint_oracle.json"),
        "deltas": binding(output / "deltas.npz"),
        "rows": binding(output / "rows.npy"),
        "target_arrays": {
            key: binding(output / (key + "_rows.npy")) for key in FIELDS.values()
        },
        "code": binding(Path(__file__)),
        "target_code": binding(PROJECT / "research/src/brazil_rv/v2/targets.py"),
        "seconds": perf_counter() - start,
        "model_forward_or_fit": False,
        "contract": "Gross continuous entitlement at observed endpoint; cash remains unremunerated cash/receivable in labels. Economic effect applies independently of custody. No synthetic basket OHLC, execution or loan assumptions. Lot-dependent post-auction endpoints stay unsupported. All actual crossing labels precede auctions.",
        "wealth_scope": "Each converted source retains six sessions of liquidity-based eligibility after effect, with no source quotes. These 18 cells are preserved, not dropped. Keep observed OHLC and feature coordinates intact here; separately verify claim-close valuations and their feature/eligibility treatment before complete-store acceptance.",
    }
    write_json_atomic(output / "manifest.json", receipt)
    run["corporate_target_propagation"] = binding(output / "manifest.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in ["status", "control_cells", "cases", "effects", "seconds"]
            }
        )
    )


if __name__ == "__main__":
    main()
