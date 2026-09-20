"""Preserve liquidity membership while separating converted claims from quotes."""

import json
from decimal import Decimal
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    started = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    history = bound_json(run["rename_history_propagation"])
    peers = bound_json(run["rename_peer_propagation"])
    terms = bound_json(run["corporate_replay"])
    parent = json.loads((PROJECT / "docs/v2_data_inputs.json").read_text())["store"]
    root = Path(parent["root"])
    original = bound_json(
        {"path": str(root / "manifest.json"), "sha256": parent["manifest_sha256"]}
    )
    dates = np.load(root / "date_index.npy")
    isins = list(np.load(root / "isin_index.npy"))
    assert dates[-1] == np.datetime64("2024-12-30")
    output = Path(run["root"]) / "corporate_claim_rows"
    output.mkdir(exist_ok=False)
    (output / "executed_reproducer.py").write_bytes(Path(__file__).read_bytes())

    def prior(key):
        record = peers["arrays"].get(key, history["arrays"].get(key))
        return np.load(
            record["path"] if record else root / original["arrays"][key]["path"],
            mmap_mode="r",
        )

    active, seen, wealth_valid, allowed, slow_valid = [
        prior(k)
        for k in (
            "active",
            "observed",
            "shareholder_wealth_valid",
            "entry_fill_allowed",
            "slow_valid",
        )
    ]
    close = prior("raw_close")
    slow_values, slow_ages = prior("slow_values"), prior("slow_age_sessions")
    q, cash, resolved = [
        prior(k)
        for k in (
            "action_shares_per_prior_share",
            "action_cash_per_prior_share",
            "action_session_resolved",
        )
    ]
    barriers, rows, features = [], [], []
    for event in terms["share_distributions"]:
        n = isins.index(event["isin"])
        effect = int(np.searchsorted(dates, np.datetime64(event["effective_date"])))
        known = int(np.searchsorted(dates, np.datetime64(event["available_date"])))
        start = max(effect, known)
        assert not seen[start:, n].any(), "a resumed source quote needs new evidence"
        days = np.flatnonzero(active[start:, n]) + start
        assert len(days) == 6
        assert not wealth_valid[days, n].any()
        for t in days:
            components = []
            for leg in event["legs"]:
                j = isins.index(leg["successor_isin"])
                # This bounded six-session observation has no later action. Do
                # not use a retrospective action chain to price a known claim.
                assert seen[t, j] and resolved[effect + 1 : t + 1, j].all()
                assert np.all(q[effect + 1 : t + 1, j] == 1)
                assert np.all(cash[effect + 1 : t + 1, j] == 0)
                auction = leg.get("fractional_auction")
                assert auction is None or dates[t] < np.datetime64(
                    auction["available_date"]
                )
                price = round(float(close[t, j]), 2)
                components.append(
                    {
                        "isin": leg["successor_isin"],
                        "units": leg["shares_per_prior_share"],
                        "close_brl": price,
                        "delivered": str(dates[t]) >= leg["delivery_date"],
                    }
                )
            amount = event["cash_per_prior_share"]
            mark = amount + sum(c["units"] * c["close_brl"] for c in components)
            decimal_mark = Decimal(str(amount)) + sum(
                Decimal(str(c["units"])) * Decimal(str(c["close_brl"]))
                for c in components
            )
            np.testing.assert_allclose(mark, float(decimal_mark), rtol=1e-15)
            rows.append(
                {
                    "date": dates[t].astype(object),
                    "isin": event["isin"],
                    "effect_date": event["effective_date"],
                    "terms_available_date": event["available_date"],
                    "mark_available_next_decision": dates[t + 1].astype(object),
                    "cash_receivable_brl": amount,
                    "claim_close_brl": mark,
                    "liquidity_eligible": True,
                    "source_quote_observed": False,
                    "source_entry_allowed": False,
                    "legs": json.dumps(components, sort_keys=True),
                    "source_manifest_sha256": run["corporate_replay"]["sha256"],
                }
            )
            for f, name in enumerate(original["feature_names"]["slow"]):
                features.append(
                    {
                        "date": dates[t].astype(object),
                        "isin": event["isin"],
                        "field": name,
                        "valid": bool(slow_valid[t, n, f]),
                        "value": float(slow_values[t, n, f]),
                        "age": float(slow_ages[t, n, f]),
                    }
                )
        barriers.append(
            {
                "isin": event["isin"],
                "start": str(dates[start]),
                "effective_date": event["effective_date"],
                "available_date": event["available_date"],
                "reason": "sourced mandatory share distribution",
            }
        )
    for event in terms["cash_cancellations"]:
        start = max(event["coverage_start"], event["coverage_available_date"])
        n = isins.index(event["isin"])
        t = int(np.searchsorted(dates, np.datetime64(start)))
        assert not seen[t:, n].any()
        barriers.append(
            {
                "isin": event["isin"],
                "start": start,
                "effective_date": event["coverage_start"],
                "available_date": event["coverage_available_date"],
                "reason": "sourced closed shareholder register; later cash terms separate",
            }
        )
    # Membership is retained. Only the explicit entry permission changes; this
    # does not create a fill where the independent observation gate was false.
    indices = np.concatenate(
        [
            np.arange(np.searchsorted(dates, np.datetime64(b["start"])), len(dates))
            * len(isins)
            + isins.index(b["isin"])
            for b in barriers
        ]
    )
    indices = indices[allowed.ravel()[indices]]
    assert not seen.ravel()[indices].any()
    np.savez_compressed(
        output / "deltas.npz",
        entry_fill_allowed__indices=indices,
        entry_fill_allowed__values=np.zeros(len(indices), bool),
    )
    pl.DataFrame(rows).write_parquet(output / "claim_close_rows.parquet")
    pl.DataFrame(features).write_parquet(output / "preserved_slow_observations.parquet")
    report = {
        "status": "resolved_claim_valuation_and_preserved_model_membership",
        "parent": parent,
        "sources": {
            k: run[k]
            for k in (
                "corporate_replay",
                "rename_history_propagation",
                "rename_peer_propagation",
            )
        },
        "code": binding(output / "executed_reproducer.py"),
        "barriers": barriers,
        "source_entry_permissions_closed": len(indices),
        "eligible_entry_permissions_closed": int(active.ravel()[indices].sum()),
        "claims": binding(output / "claim_close_rows.parquet"),
        "preserved_features": binding(output / "preserved_slow_observations.parquet"),
        "deltas": binding(output / "deltas.npz"),
        "claim_rows": len(rows),
        "slow_fields_retained": len(features),
        "liquidity_membership_changed": False,
        "feature_or_wealth_arrays_changed": False,
        "causality": "max(effect, knowledge) entry barrier; exact closing marks available next decision; no future endpoints or later action terms consumed",
        "wealth_contract": "Closing entitlement observations only. Preserve coherent source-security reinvest-at-close OHLC masks; no synthetic basket high/low or new neural field. Gross cash is not reinvested/remunerated. Quantity-dependent post-auction and funded investor wealth remain in the account ledger.",
        "feature_contract": "All 18 liquidity-eligible rows and full prior history retained. First decision uses last pre-effect OHLC; subsequent source-return/volatility fields are unsupported. Activity reducers may consume independently certified zero activity, and valid history-age fields remain. Missing market observations never refresh from claim marks.",
        "seconds": perf_counter() - started,
        "model_forward_or_heldout": False,
    }
    write_json_atomic(output / "manifest.json", report)
    run["corporate_claim_rows"] = binding(output / "manifest.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "claim_rows",
                    "source_entry_permissions_closed",
                    "eligible_entry_permissions_closed",
                    "seconds",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
