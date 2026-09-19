"""Actual-calendar, full-name default Dommo arithmetic; no predictions/labels."""

from copy import copy
from dataclasses import replace
import json
from pathlib import Path
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
from verify_corporate_replay import inputs_on_axes

PROJECT = Path(__file__).resolve().parents[1]


def main():
    start = time.perf_counter()
    pointer_path = PROJECT / "docs/v2_economic_data_scaling_run.json"
    pointer = json.loads(pointer_path.read_text())
    record = pointer["corporate_replay"]
    terms, calendar = load_corporate_replay(record["path"], record["sha256"])
    store = Path(terms["store"]["root"])
    names = np.arange(len(np.load(store / "isin_index.npy")))
    rows = np.flatnonzero(
        (calendar >= np.datetime64("2023-01-06"))
        & (calendar <= np.datetime64("2023-04-10"))
    )
    raw = inputs_on_axes(store, calendar, names, rows)
    loan_manifest = Path(pointer["loan_source_panels"]["path"])
    assert sha256_file(loan_manifest) == pointer["loan_source_panels"]["sha256"]
    panel_path = loan_manifest.parent / "panels.npz"
    assert (
        sha256_file(panel_path)
        == json.loads(loan_manifest.read_text())["panels"]["sha256"]
    )
    with np.load(panel_path) as panel:
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
    source = raw.security_ids.index("BRDMMOACNOR0")
    successor = raw.security_ids.index("BRPRIOACNOR1")
    delivery, returned = dates.index("2023-01-11"), dates.index("2023-01-13")
    price, reference = float(raw.raw_close[0, source]), float(references[0, source])
    successor_price = float(raw.raw_close[delivery, successor])
    results = []
    for capital in (1_000_000, 5_000_000, 10_000_000):
        for sign in (1, -1):
            config = policy_ledger_config(
                initial_capital_brl=capital,
                annual_borrow_rate=0.04,
                borrow_source="uniform",
                borrow_fee_multiplier=0,
                cost_bps_per_side=0,
                hedge_cost_bps_per_side=0,
            )
            targets = np.zeros((len(rows), len(names) + 1))
            targets[0, source] = sign * 0.04
            account = data.initial_account(0, config)
            nav = [
                float(
                    data.step(
                        account,
                        tensor(targets[day]),
                        day,
                        terminal=day == len(rows) - 1,
                    )["nav"]
                )
                for day in range(len(rows))
            ]
            book, _, _ = exact_replay(
                data, None, 0, len(rows), config=config, targets=targets
            )
            np.testing.assert_allclose(nav, book.nav, rtol=0, atol=3e-8)
            quantity = 0.04 * capital / price
            whole = np.floor(quantity * 0.0375)
            fraction = quantity * 0.0375 - whole
            rent = (
                quantity * reference * (1.04 ** (returned / 252) - 1) if sign < 0 else 0
            )
            expected = (
                capital
                + sign
                * (
                    whole * successor_price
                    + fraction * 31.94031
                    + quantity * 0.4625
                    - quantity * price
                )
                - rent
            )
            np.testing.assert_allclose(book.nav[-1], expected, rtol=0, atol=3e-8)
            assert book.receivables[-1] == 0 and book.payables[-1] == 0
            assert book.restricted_cash[-1] == 0 and book.unsettled_cash[-1] == 0
            fills = [f for f in book.fills if f.security_index == successor]
            assert len(fills) == 1 and fills[0].quantity == whole
            results.append(
                dict(
                    capital_brl=capital,
                    sign=sign,
                    original_quantity=quantity,
                    original_reference=reference,
                    original_principal=quantity * reference,
                    whole_successor_quantity=float(whole),
                    fractional_quantity=float(fraction),
                    rent_brl=rent,
                    terminal_nav_brl=float(book.nav[-1]),
                    expected_nav_brl=float(expected),
                    max_account_error_brl=float(np.max(np.abs(book.nav - nav))),
                )
            )
    # Count only accounting coverage amendments; never regenerate learned features.
    all_rows = np.arange(len(calendar))
    full = inputs_on_axes(store, calendar, names, all_rows)
    revised = apply_corporate_replay(full, terms, calendar, record["sha256"])
    gained = revised.action_session_resolved & ~full.action_session_resolved
    report = dict(
        status="default_pna_actual_calendar_oracles_pass_not_model_profitability",
        manifest=record,
        shape=list(full.active.shape),
        results=results,
        new_dommo_coverage_cells=int(gained[:, source].sum()),
        new_dommo_eligible_coverage_cells=int(
            (gained[:, source] & full.active[:, source]).sum()
        ),
        total_coverage_cells=int(gained.sum()),
        total_eligible_coverage_cells=int((gained & full.active).sum()),
        static_policy_coordinates_preserved=True,
        model_predictions_labels_or_heldout_read=False,
        scope="six predeclared full-name four-percent long/short books, original-reference 4% rent; zero execution/B3/CDI isolate cash and principal arithmetic, not the registered model-cost assumptions",
        boundaries="presettlement disposal; actual lender participation; invoice precision/custodian sweep; zero-whole-contract rent endpoints; changed clearing/return assumptions need corresponding physical custody terms",
        reproducer={"path": str(Path(__file__)), "sha256": sha256_file(Path(__file__))},
        seconds=time.perf_counter() - start,
    )
    path = PROJECT / "docs/v2_dommo_conversion_acceptance.json"
    digest = write_json_atomic(path, report)
    pointer["dommo_conversion_acceptance"] = {
        "path": str(path.relative_to(PROJECT)),
        "sha256": digest,
    }
    write_json_atomic(pointer_path, pointer)
    print(json.dumps({k: v for k, v in report.items() if k not in ["results"]}))


if __name__ == "__main__":
    main()
