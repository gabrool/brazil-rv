"""Actual-calendar lender-election cash and funding oracles, without predictions."""

from copy import copy
from dataclasses import replace
from datetime import datetime
import json
import math
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
from verify_corporate_replay import ACCOUNT_FIELDS, inputs_on_axes

PROJECT = Path(__file__).resolve().parents[1]


def main():
    started = time.perf_counter()
    pointer_path = PROJECT / "docs/v2_economic_data_scaling_run.json"
    pointer = json.loads(pointer_path.read_text())
    record = pointer["dommo_pnb_scenario"]
    terms, calendar = load_corporate_replay(record["path"], record["sha256"])
    store = Path(terms["store"]["root"])
    hashes = {key: sha256_file(store / f"{key}.npy") for key in ACCOUNT_FIELDS}
    names = np.arange(len(np.load(store / "isin_index.npy")))
    rows = np.flatnonzero(
        (calendar >= np.datetime64("2022-12-23"))
        & (calendar <= np.datetime64("2023-01-16"))
    )
    raw = inputs_on_axes(store, calendar, names, rows)
    source = raw.security_ids.index("BRDMMOACNOR0")
    binding = pointer["loan_source_panels"]
    manifest_path = Path(binding["path"])
    assert sha256_file(manifest_path) == binding["sha256"]
    loan_path = manifest_path.parent / "panels.npz"
    loan_hash = sha256_file(loan_path)
    assert loan_hash == json.loads(manifest_path.read_text())["panels"]["sha256"]
    with np.load(loan_path) as panel:
        np.testing.assert_array_equal(panel["dates"], calendar)
        refs = panel["loan_reference_prices"][rows]
    raw = replace(
        raw,
        loan_reference_prices=refs,
        initial_reference_price=refs[0, :-1],
        annual_borrow_rate_by_name=np.full(raw.active.shape, 0.04),
    )
    dates = calendar[rows].astype(str).tolist()
    cdi_source = next(s for s in terms["sources"] if "cdi_sgs12_20221024" in s["path"])
    rates = {
        datetime.strptime(r["data"], "%d/%m/%Y").date().isoformat(): float(r["valor"])
        / 100
        for r in json.loads(Path(cdi_source["path"]).read_text())
    }
    cdi = np.array(
        [
            0.0,
            *[
                math.prod(1 + v for d, v in rates.items() if before <= d < after) - 1
                for before, after in zip(dates, dates[1:])
            ],
        ]
    )
    outcomes = []
    for capital in (1_000_000, 5_000_000, 10_000_000):
        for funded in (False, True):
            initial = replace(raw, cdi_returns=cdi if funded else np.zeros(len(rows)))
            original = PolicyData(
                initial,
                np.ones(raw.active.shape),
                np.full(raw.active.shape, 0.0004),
                np.full(len(rows), 0.0001),
                np.zeros(len(rows)),
                refs[:, :-1],
            )
            data = copy(original)
            data.inputs = apply_corporate_replay(
                initial, terms, calendar, record["sha256"]
            )
            assert data.static is original.static
            for key in ("raw_close", "active", "scores", "scaled_midrank_targets"):
                assert getattr(data.inputs, key) is getattr(initial, key)
            config = policy_ledger_config(
                initial_capital_brl=capital,
                annual_borrow_rate=0.04,
                borrow_source="uniform",
                borrow_fee_multiplier=0,
                cost_bps_per_side=0,
                hedge_cost_bps_per_side=0,
            )
            account = data.initial_account(0, config)
            targets = np.zeros((len(rows), len(names) + 1))
            nav = []
            for day in range(len(rows)):
                account.prepare_day(day)
                targets[day] = account.market_weights.detach().numpy()
                if day == 0:
                    targets[day, source] = -0.04
                nav.append(
                    float(
                        data.step(
                            account,
                            tensor(targets[day]),
                            day,
                            terminal=day == len(rows) - 1,
                        )["nav"]
                    )
                )
            book, _, _ = exact_replay(
                data, None, 0, len(rows), config=config, targets=targets
            )
            np.testing.assert_allclose(nav, book.nav, atol=3e-8, rtol=0)
            price, reference = float(raw.raw_close[0, source]), float(refs[0, source])
            quantity = capital * 0.04 / price
            stop, rent_day, cash_day = [
                dates.index(d) for d in ("2022-12-26", "2022-12-28", "2023-01-13")
            ]
            principal = quantity * reference
            rent = principal * (1.04 ** (stop / 252) - 1)
            redemption = quantity * terms["dommo_election_contract"]["final_published"]
            free, restricted = float(capital), 0.0
            for day, rate in enumerate(data.inputs.cdi_returns):
                free += (free + restricted) * rate
                if day == 2:
                    restricted = capital * 0.04
                if day == rent_day:
                    free -= rent
                if day == cash_day:
                    free += restricted - redemption
                    restricted = 0
            np.testing.assert_allclose(book.nav[-1], free, atol=3e-8, rtol=0)
            np.testing.assert_allclose(
                book.loan_payment[rent_day], rent, atol=3e-8, rtol=0
            )
            assert book.loan_payment[rent_day + 1 :].sum() == 0
            np.testing.assert_allclose(
                book.loan_redemption_liability[cash_day - 1],
                redemption,
                atol=3e-8,
                rtol=0,
            )
            assert book.loan_redemption_liability[cash_day:].sum() == 0
            assert book.restricted_cash[cash_day:].sum() == 0
            assert len(book.fills) == 1
            outcomes.append(
                dict(
                    capital_brl=capital,
                    actual_CDI_funding=funded,
                    original_reference_brl=reference,
                    original_principal_brl=principal,
                    loan_quantity=quantity,
                    rent_brl=rent,
                    redemption_brl=redemption,
                    expected_terminal_nav_brl=free,
                    actual_terminal_nav_brl=float(book.nav[-1]),
                    max_account_difference_brl=float(np.max(np.abs(nav - book.nav))),
                    rent_paid_date=dates[rent_day],
                    redemption_paid_date=dates[cash_day],
                )
            )
    assert hashes == {key: sha256_file(store / f"{key}.npy") for key in ACCOUNT_FIELDS}
    receipt = dict(
        status="source_bound_Dommo_lender_election_endpoint_verified_default_PNA_incomplete",
        scenario=record,
        source_array_hashes=hashes,
        loan_panel_sha256=loan_hash,
        name_count=len(names),
        model_predictions_and_labels_read=False,
        static_policy_coordinates_preserved=True,
        accepted_store_mutated=False,
        CDI_convention="SGS12 monetary dates accrued across every interval, including equity closure Dec30; zero-funded oracles isolate loan arithmetic",
        outcomes=outcomes,
        remaining="default PNA loan fractions, actual lender participation, lifecycle and full clearing calendar remain; no corrected model profitability claimed",
        elapsed_seconds=time.perf_counter() - started,
    )
    path = PROJECT / "docs/v2_dommo_election_acceptance.json"
    digest = write_json_atomic(path, receipt)
    pointer["dommo_election_acceptance"] = {
        "path": str(path.relative_to(PROJECT)),
        "sha256": digest,
    }
    write_json_atomic(pointer_path, pointer)
    print(
        json.dumps(
            {
                k: v
                for k, v in receipt.items()
                if k not in ("source_array_hashes", "outcomes")
            }
        )
    )
    print(json.dumps(outcomes[-2:]))


if __name__ == "__main__":
    main()
