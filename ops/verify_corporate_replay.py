"""Source-axis admission oracle; no saved predictions or historical labels read."""

from copy import copy
from dataclasses import replace
from pathlib import Path
import json
import time

import numpy as np

from brazil_rv.execution.portfolio_account import tensor
from brazil_rv.execution.portfolio_policy import (
    PolicyData,
    exact_replay,
    policy_ledger_config,
)
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.corporate_replay import apply_corporate_replay, load_corporate_replay
from brazil_rv.v2.evaluate import EvaluationInputs

PROJECT = Path(__file__).resolve().parents[1]
ACCOUNT_FIELDS = (
    "active",
    "raw_close",
    "action_shares_per_prior_share",
    "action_cash_per_prior_share",
    "action_session_resolved",
    "action_has_action",
    "action_successor_index",
    "action_payment_session",
)


def inputs_on_axes(store, calendar, names, rows):
    shape = (len(rows), len(names))
    payload = np.broadcast_to(np.float32(0), (*shape, 5))
    # Sentinels explicitly replace predictions/labels. This is not model scoring.
    masks = np.broadcast_to(True, payload.shape)
    arrays = {
        key: np.load(store / f"{key}.npy", mmap_mode="r")[np.ix_(rows, names)]
        for key in ACCOUNT_FIELDS
    }
    arrays["action_payment_session"] = np.where(
        arrays["action_payment_session"] >= 0,
        arrays["action_payment_session"] - rows[0],
        -1,
    )
    # Source indices are global; the bounded oracle below keeps all name axes.
    return EvaluationInputs(
        dates=tuple(calendar[rows].astype(object)),
        session_indices=rows,
        calendar_identity_sha256=sha256_file(store / "date_index.npy"),
        scores=payload,
        score_mask=masks,
        scaled_midrank_targets=payload,
        scaled_target_mask=masks,
        neutral_midrank_targets=payload,
        neutral_target_mask=masks,
        shareholder_midrank_targets=payload,
        shareholder_simple_returns=payload,
        shareholder_target_mask=masks,
        price_midrank_targets=payload,
        price_target_mask=masks,
        **arrays,
        security_ids=tuple(np.load(store / "isin_index.npy")[names]),
        target_scale_sigma=np.broadcast_to(0.02, shape),
        prior_feature_values={"yang_zhang_vol_20": np.broadcast_to(0.02, shape)},
        cdi_returns=np.zeros(len(rows)),
        transfer_chronology_clean=True,
        initial_unresolved_action=np.zeros(len(names), bool),
        annual_borrow_rate_by_name=np.zeros(shape),
        borrow_rate_imputed=np.zeros(shape, bool),
        borrow_rate_placeholder=np.zeros(shape, bool),
        shortable_by_borrow_source={"borrow_balance": np.ones(shape, bool)},
        bova11_close=np.full(len(rows), 100.0),
        initial_hedge_reference_price=100.0,
    )


def main():
    started = time.perf_counter()
    pointer_path = PROJECT / "docs/v2_economic_data_scaling_run.json"
    pointer = json.loads(pointer_path.read_text())
    record = pointer["corporate_replay"]
    terms, calendar = load_corporate_replay(record["path"], record["sha256"])
    store = Path(terms["store"]["root"])
    before_hashes = {k: sha256_file(store / f"{k}.npy") for k in ACCOUNT_FIELDS}
    names = np.arange(len(np.load(store / "isin_index.npy")))
    full = inputs_on_axes(store, calendar, names, np.arange(len(calendar)))
    after = apply_corporate_replay(full, terms, calendar, record["sha256"])
    counts = {}
    for key in ACCOUNT_FIELDS:
        a, b = getattr(full, key), getattr(after, key)
        equal = (a == b) | (np.isnan(a) & np.isnan(b))
        counts[key] = int((~equal).sum())
        if key in ("active", "raw_close", "action_successor_index"):
            assert a is b
    for key in (
        "scores",
        "score_mask",
        "scaled_midrank_targets",
        "prior_feature_values",
    ):
        assert getattr(full, key) is getattr(after, key)
    name = full.security_ids.index("BRCIELACNOR3")
    resolved_gain = after.action_session_resolved & ~full.action_session_resolved
    admitted_names = [name] + [
        full.security_ids.index(x["isin"]) for x in terms["share_distributions"]
    ]
    assert not (
        resolved_gain
        & np.broadcast_to(~np.isin(names, admitted_names), resolved_gain.shape)
    ).any()
    eligible_gain = int((resolved_gain & full.active).sum())
    rows = np.flatnonzero(
        (calendar >= np.datetime64("2024-08-26"))
        & (calendar <= np.datetime64("2024-09-27"))
    )
    small = inputs_on_axes(store, calendar, names, rows)
    sources = pointer["loan_source_panels"]
    assert sha256_file(Path(sources["path"])) == sources["sha256"]
    loan_manifest = json.loads(Path(sources["path"]).read_text())
    # Source panel manifest is verified separately by its established loader;
    # here bind the exact panel bytes used by the historical mechanics oracle.
    loan_path = Path(sources["path"]).parent / "panels.npz"
    loan_hash = sha256_file(loan_path)
    assert loan_hash == loan_manifest["panels"]["sha256"]
    with np.load(loan_path) as panels:
        np.testing.assert_array_equal(panels["dates"], calendar)
        np.testing.assert_array_equal(panels["isins"][:-1], full.security_ids)
        small = replace(
            small, loan_reference_prices=panels["loan_reference_prices"][rows]
        )
    # Published prior averages are sufficient prior marks for this zero-cost
    # predetermined-order oracle; all other securities and the hedge stay flat.
    small = replace(small, initial_reference_price=small.loan_reference_prices[0, :-1])
    shape = small.active.shape
    policy = PolicyData(
        small,
        np.ones(shape),
        np.full(shape, 0.0004),
        np.full(len(rows), 0.0001),
        np.zeros(len(rows)),
        small.loan_reference_prices[:, :-1],
    )
    revised = copy(policy)
    revised.inputs = apply_corporate_replay(small, terms, calendar, record["sha256"])
    assert revised.static is policy.static
    config = policy_ledger_config(
        initial_capital_brl=10_000_000,
        annual_borrow_rate=0,
        borrow_source="uniform",
        borrow_fee_multiplier=0,
        cost_bps_per_side=0,
        hedge_cost_bps_per_side=0,
    )
    entry = float(small.raw_close[0, name])
    quantity = 400_000 / entry
    outcomes = []
    for side in (1, -1):
        targets = np.zeros((len(rows), len(names) + 1))
        targets[0, name] = side * 0.04
        account = revised.initial_account(0, config)
        nav = [
            revised.step(account, tensor(t), d, terminal=d == len(rows) - 1)[
                "nav"
            ].item()
            for d, t in enumerate(targets)
        ]
        result, _, _ = exact_replay(
            revised, None, 0, len(rows), config=config, targets=targets
        )
        np.testing.assert_allclose(nav, result.nav, atol=2e-8, rtol=0)
        cash = (
            terms["cash_cancellations"][0]["cash_per_share"]
            if side > 0
            else terms["loan_cash_settlements"][0]["cash_per_share"]
        )
        expected = config.initial_capital_brl + side * quantity * (cash - entry)
        np.testing.assert_allclose(result.nav[-1], expected, atol=2e-8, rtol=0)
        outcomes.append(
            {
                "side": side,
                "quantity": quantity,
                "entry_brl": entry,
                "cash_per_share_brl": cash,
                "expected_nav_brl": expected,
                "actual_nav_brl": float(result.nav[-1]),
                "max_account_difference_brl": float(np.max(np.abs(result.nav - nav))),
                "loan_invoice_cent_bound_brl": quantity * 0.01 if side < 0 else 0,
            }
        )
    event = terms["share_distributions"][0]
    source = full.security_ids.index(event["isin"])
    destination = full.security_ids.index(event["successor_isin"])
    rows = np.flatnonzero(
        (calendar >= np.datetime64("2023-01-06"))
        & (calendar <= np.datetime64("2023-02-03"))
    )
    raw = inputs_on_axes(store, calendar, names, rows)
    with np.load(loan_path) as panels:
        raw = replace(raw, loan_reference_prices=panels["loan_reference_prices"][rows])
    raw = replace(raw, initial_reference_price=raw.loan_reference_prices[0, :-1])
    shape = raw.active.shape
    original = PolicyData(
        raw,
        np.ones(shape),
        np.full(shape, 0.0004),
        np.full(len(rows), 0.0001),
        np.zeros(len(rows)),
        raw.loan_reference_prices[:, :-1],
    )
    brml = copy(original)
    brml.inputs = apply_corporate_replay(raw, terms, calendar, record["sha256"])
    assert brml.static is original.static
    dates = calendar[rows].astype(str).tolist()
    delivery = dates.index(event["delivery_date"])
    entry = float(raw.raw_close[0, source])
    sale = float(raw.raw_close[delivery, destination])
    ratio = event["shares_per_prior_share"]
    cash = event["cash_per_prior_share"]
    brml_outcomes = []
    for capital in [1_000_000, 5_000_000, 10_000_000]:
        for side in [1, -1]:
            quantity = 0.04 * capital / entry
            entitled = quantity * ratio
            fraction = entitled % 1 if side > 0 else 0
            delivered = np.floor(entitled) if side > 0 else entitled
            targets = np.zeros((len(rows), len(names) + 1))
            targets[0, source] = side * 0.04
            cfg = replace(config, initial_capital_brl=capital)
            account = brml.initial_account(0, cfg)
            nav = [
                brml.step(account, tensor(t), d, terminal=d == len(rows) - 1)[
                    "nav"
                ].item()
                for d, t in enumerate(targets)
            ]
            result, _, _ = exact_replay(
                brml, None, 0, len(rows), config=cfg, targets=targets
            )
            np.testing.assert_allclose(nav, result.nav, atol=3e-8, rtol=0)
            expected = capital + side * (
                delivered * sale + quantity * cash - quantity * entry
            )
            expected += fraction * event["fractional_auction"]["cash_per_share"]
            np.testing.assert_allclose(result.nav[-1], expected, atol=3e-8, rtol=0)
            delivered_fills = [
                f for f in result.fills if f.security_index == destination
            ]
            assert (
                len(delivered_fills) == 1
                and delivered_fills[0].fill_session == delivery
            )
            np.testing.assert_allclose(
                delivered_fills[0].quantity, delivered, atol=1e-8
            )
            assert result.unsettled_cash[-1] == 0
            brml_outcomes.append(
                dict(
                    capital_brl=capital,
                    side=side,
                    source_quantity=quantity,
                    delivered_quantity=delivered,
                    retained_fraction=fraction,
                    expected_nav_brl=expected,
                    actual_nav_brl=float(result.nav[-1]),
                    max_account_difference_brl=float(np.max(np.abs(result.nav - nav))),
                )
            )
    # Actual Jan10 ALSO purchase offsets a BRML short on Jan11, but cannot return
    # its part of the converted loan until Jan12. The remaining Jan11 cover returns Jan13.
    cfg = replace(config, annual_borrow_rate=0.04)
    annual = np.full(shape, 0.04)
    brml = copy(brml)
    brml.inputs = replace(brml.inputs, annual_borrow_rate_by_name=annual)
    account = brml.initial_account(0, cfg)
    targets = np.zeros((len(rows), len(names) + 1))
    targets[0, source] = -0.04
    nav = []
    for d in range(len(rows)):
        account.prepare_day(d)
        if dates[d] == "2023-01-10":
            targets[d, destination] = 200_000 / float(account.nav)
        nav.append(
            float(
                brml.step(account, tensor(targets[d]), d, terminal=d == len(rows) - 1)[
                    "nav"
                ]
            )
        )
    result, _, _ = exact_replay(brml, None, 0, len(rows), config=cfg, targets=targets)
    np.testing.assert_allclose(nav, result.nav, atol=3e-8, rtol=0)
    quantity = 400_000 / entry
    purchased = 200_000 / float(raw.raw_close[dates.index("2023-01-10"), destination])
    portion = purchased / (quantity * ratio)
    reference = float(raw.loan_reference_prices[0, source])
    first_return = dates.index("2023-01-12")
    last_return = dates.index("2023-01-13")
    principal = quantity * reference
    rent1 = principal * portion * (1.04 ** (first_return / 252) - 1)
    rent2 = principal * (1 - portion) * (1.04 ** (last_return / 252) - 1)
    expected = (
        cfg.initial_capital_brl
        + quantity * (entry - ratio * sale - cash)
        + purchased * sale
        - 200_000
        - rent1
        - rent2
    )
    np.testing.assert_allclose(result.nav[-1], expected, atol=3e-8, rtol=0)
    np.testing.assert_allclose(
        result.loan_payment[first_return], rent1, atol=3e-8, rtol=0
    )
    np.testing.assert_allclose(
        result.loan_payment[last_return], rent2, atol=3e-8, rtol=0
    )
    brml_outcomes.append(
        dict(
            case="Jan10 owned purchase offsets loan at Jan11 custody",
            capital_brl=cfg.initial_capital_brl,
            source_quantity=quantity,
            owned_purchase_quantity=purchased,
            original_reference_brl=reference,
            original_principal_brl=principal,
            Jan12_rent_brl=rent1,
            Jan13_rent_brl=rent2,
            expected_nav_brl=expected,
            actual_nav_brl=float(result.nav[-1]),
            max_account_difference_brl=float(np.max(np.abs(result.nav - nav))),
        )
    )
    assert before_hashes == {k: sha256_file(store / f"{k}.npy") for k in ACCOUNT_FIELDS}
    receipt = {
        "status": "cielo_brmalls_source_admission_verified_not_model_profitability",
        "manifest": record,
        "store": terms["store"],
        "source_array_hashes": before_hashes,
        "loan_panel_sha256": loan_hash,
        "shape": list(full.active.shape),
        "changed_cells": counts,
        "resolved_eligible_stock_days": eligible_gain,
        "model_predictions_and_labels_read": False,
        "model_arrays_mutated": False,
        "accepted_store_mutated": False,
        "historical_mechanics_oracles": outcomes,
        "brmalls_historical_oracles": brml_outcomes,
        "scope": "predetermined 4% Cielo/BRML positions plus sourced Jan10 ALSO custody offset; zero execution fees; other names flat; BRML R$1m/R$5m/R$10m and original-principal 4% loan oracle",
        "elapsed_seconds": time.perf_counter() - started,
    }
    path = PROJECT / "docs/v2_corporate_replay_acceptance.json"
    digest = write_json_atomic(path, receipt)
    pointer["corporate_replay_acceptance"] = {
        "path": str(path.relative_to(PROJECT)),
        "sha256": digest,
    }
    write_json_atomic(pointer_path, pointer)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
