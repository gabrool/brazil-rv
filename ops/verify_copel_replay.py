"""Source-bound Copel split and explicit loan-allocation sensitivity oracles."""

from copy import copy
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from brazil_rv.execution.loan_contracts import spot_settlement_session
from brazil_rv.execution.portfolio_account import tensor
from brazil_rv.execution.portfolio_policy import (
    PolicyData,
    exact_replay,
    policy_ledger_config,
)
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.corporate_replay import apply_corporate_replay, load_corporate_replay
from verify_corporate_replay import ACCOUNT_FIELDS, inputs_on_axes

PROJECT = Path(__file__).resolve().parents[1]


def main():
    started = time.perf_counter()
    pointer_path = PROJECT / "docs/v2_economic_data_scaling_run.json"
    pointer = json.loads(pointer_path.read_text())
    record = pointer["corporate_replay"]
    terms, calendar = load_corporate_replay(record["path"], record["sha256"])
    store = Path(terms["store"]["root"])
    names = np.arange(len(np.load(store / "isin_index.npy")))
    full = inputs_on_axes(store, calendar, names, np.arange(len(calendar)))
    revised = apply_corporate_replay(full, terms, calendar, record["sha256"])
    source = full.security_ids.index("BRCPLECDAM13")
    destinations = [
        full.security_ids.index(x) for x in ("BRCPLEACNOR8", "BRCPLEACNPB9")
    ]
    gains = revised.action_session_resolved & ~full.action_session_resolved
    counts = {}
    hashes = {}
    for key in ACCOUNT_FIELDS:
        before, after = getattr(full, key), getattr(revised, key)
        counts[key] = int(
            (~((before == after) | (np.isnan(before) & np.isnan(after)))).sum()
        )
        hashes[key] = sha256_file(store / f"{key}.npy")
        if key in ("active", "raw_close", "action_successor_index"):
            assert before is after
    rows = np.flatnonzero(
        (calendar >= np.datetime64("2023-12-22"))
        & (calendar <= np.datetime64("2024-01-09"))
    )
    raw = inputs_on_axes(store, calendar, names, rows)
    binding = pointer["loan_source_panels"]
    manifest_path = Path(binding["path"])
    assert sha256_file(manifest_path) == binding["sha256"]
    loan_path = manifest_path.parent / "panels.npz"
    loan_hash = sha256_file(loan_path)
    assert loan_hash == json.loads(manifest_path.read_text())["panels"]["sha256"]
    with np.load(loan_path) as panel:
        np.testing.assert_array_equal(panel["dates"], calendar)
        references = panel["loan_reference_prices"][rows]
    raw = replace(
        raw,
        loan_reference_prices=references,
        initial_reference_price=references[0, :-1],
        annual_borrow_rate_by_name=np.full(raw.active.shape, 0.04),
    )
    original = PolicyData(
        raw,
        np.ones(raw.active.shape),
        np.full(raw.active.shape, 0.0004),
        np.full(len(rows), 0.0001),
        np.zeros(len(rows)),
        references[:, :-1],
    )
    data = copy(original)
    data.inputs = apply_corporate_replay(raw, terms, calendar, record["sha256"])
    assert data.static is original.static
    dates = calendar[rows].astype(str).tolist()
    delivery = dates.index("2023-12-28")
    price = float(raw.raw_close[0, source])
    reference = float(references[0, source])
    outcomes = []
    for capital in (1_000_000, 5_000_000, 10_000_000):
        for side in (1, -1):
            for delayed_on in (False, True):
                for k in (0.0, 0.2, 1.0):
                    events = tuple(
                        replace(
                            event,
                            legs=tuple(
                                replace(leg, loan_principal_fraction=fraction)
                                for leg, fraction in zip(event.legs, (k, 1 - k))
                            ),
                        )
                        if event.source_index == source
                        else event
                        for event in data.inputs.share_distributions
                    )
                    scenario = copy(data)
                    scenario.inputs = replace(data.inputs, share_distributions=events)
                    config = policy_ledger_config(
                        initial_capital_brl=capital,
                        annual_borrow_rate=0.04,
                        borrow_source="uniform",
                        borrow_fee_multiplier=0,
                        cost_bps_per_side=0,
                        hedge_cost_bps_per_side=0,
                    )
                    quantity = 0.04 * capital / price
                    targets = np.zeros((len(rows), len(names) + 1))
                    targets[0, source] = side * 0.04
                    account = scenario.initial_account(0, config)
                    nav = []
                    for day in range(len(rows)):
                        account.prepare_day(day)
                        if day == delivery and delayed_on:
                            targets[day, destinations[0]] = float(
                                account.market_weights[destinations[0]]
                            )
                        nav.append(
                            float(
                                scenario.step(
                                    account,
                                    tensor(targets[day]),
                                    day,
                                    terminal=day == len(rows) - 1,
                                )["nav"]
                            )
                        )
                    result, _, _ = exact_replay(
                        scenario, None, 0, len(rows), config=config, targets=targets
                    )
                    np.testing.assert_allclose(nav, result.nav, atol=3e-8, rtol=0)
                    exits = [delivery + int(delayed_on), delivery]
                    proceeds = quantity * sum(
                        ratio * float(raw.raw_close[day, destination])
                        for ratio, day, destination in zip((1, 4), exits, destinations)
                    )
                    due = [
                        spot_settlement_session(day, raw.dates[day]) for day in exits
                    ]
                    rent = (
                        0.0
                        if side > 0
                        else quantity
                        * reference
                        * sum(
                            fraction * (1.04 ** (day / 252) - 1)
                            for fraction, day in zip((k, 1 - k), due)
                        )
                    )
                    expected = capital + side * (proceeds - quantity * price) - rent
                    np.testing.assert_allclose(
                        result.nav[-1], expected, atol=3e-8, rtol=0
                    )
                    assert result.unsettled_cash[-1] == 0
                    fills = [
                        f for f in result.fills if f.security_index in destinations
                    ]
                    assert len(fills) == 2
                    outcomes.append(
                        dict(
                            capital_brl=capital,
                            side=side,
                            delayed_on_exit=delayed_on,
                            on_principal_fraction=k,
                            original_reference_brl=reference,
                            original_principal_brl=quantity * reference,
                            rent_brl=rent,
                            return_dates=[dates[d] for d in due],
                            expected_nav_brl=float(expected),
                            actual_nav_brl=float(result.nav[-1]),
                            max_account_difference_brl=float(
                                np.max(np.abs(result.nav - nav))
                            ),
                        )
                    )
    assert hashes == {key: sha256_file(store / f"{key}.npy") for key in ACCOUNT_FIELDS}
    receipt = dict(
        status="copel_sourced_shares_custody_and_bounded_loan_allocation_verified_not_model_profitability",
        manifest=record,
        source_array_hashes=hashes,
        loan_panel_sha256=loan_hash,
        shape=list(full.active.shape),
        changed_cells=counts,
        copel_resolved_cells=int(gains[:, source].sum()),
        copel_resolved_eligible_cells=int(
            (gains[:, source] & full.active[:, source]).sum()
        ),
        model_predictions_and_labels_read=False,
        accepted_store_mutated=False,
        static_policy_coordinates_preserved=True,
        outcomes=outcomes,
        scope="actual-calendar 4% prescribed positions, full name axes, 4% original-reference rent; zero execution/B3 fees and CDI for independent arithmetic; loan K bounds on fixed disposal paths, not on adaptive model policies",
        remaining="issuer K and loan conversion night not independently quoted; presettlement disposal and full clearing calendar remain explicit sensitivities",
        elapsed_seconds=time.perf_counter() - started,
    )
    output = PROJECT / "docs/v2_copel_replay_acceptance.json"
    digest = write_json_atomic(output, receipt)
    pointer["copel_replay_acceptance"] = {
        "path": str(output.relative_to(PROJECT)),
        "sha256": digest,
    }
    write_json_atomic(pointer_path, pointer)
    print(
        json.dumps(
            {
                key: value
                for key, value in receipt.items()
                if key not in ("outcomes", "source_array_hashes")
            }
        )
    )


if __name__ == "__main__":
    main()
