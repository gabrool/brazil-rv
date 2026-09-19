"""Small unchanged-book comparison against the recorded pre-election source."""

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from brazil_rv.execution import loan_contracts, stateful_ledger, portfolio_account
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic

PROJECT = Path(__file__).resolve().parents[1]
BASE = "a66cd423b899d6d9c5d8afa2a175caa72f43f77d"


def main():
    pointer = json.loads(
        (PROJECT / "docs/v2_economic_data_scaling_run.json").read_text()
    )
    root = Path(pointer["root"]) / "loan_election_unchanged"
    root.mkdir(exist_ok=True)

    def load(name):
        source = f"research/src/brazil_rv/execution/{name}.py"
        path = root / f"baseline_{name}.py"
        raw = subprocess.check_output(["git", "show", f"{BASE}:{source}"], cwd=PROJECT)
        if path.exists() and path.read_bytes() != raw:
            raise ValueError("baseline source changed")
        path.write_bytes(raw)
        spec = importlib.util.spec_from_file_location(
            f"brazil_rv.execution._baseline_{name}", path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    old_loans = load("loan_contracts")
    sys.modules["brazil_rv.execution.loan_contracts"] = old_loans
    try:
        old_ledger = load("stateful_ledger")
        old_account = load("portfolio_account")
    finally:
        sys.modules["brazil_rv.execution.loan_contracts"] = loan_contracts
    sys.path.insert(0, str(PROJECT / "research/tests"))
    import test_v2_stateful_ledger as fixture

    days = 30
    prices = 100 * np.exp(
        np.arange(days)[:, None] * np.array([0.001, -0.001, 0.0003])[None, :]
    )
    prices[6:8, 0] = np.nan
    outcomes = []
    for mixed in (False, True):
        weights = np.zeros((days, 3))
        weights[:12] = [0.04, -0.04 if mixed else 0.03, 0]
        weights[12:21] = [0.02, -0.02 if mixed else 0.015, 0.025]
        config = fixture._config(
            initial_capital_brl=10_000_000,
            annual_borrow_rate=0.04,
            cost_bps_per_side=4,
            borrow_fee_multiplier=1,
        )
        results = []
        account_results = []
        for ledger, accounts in (
            (old_ledger, old_account),
            (stateful_ledger, portfolio_account),
        ):
            fixture.simulate_stateful_ledger = ledger.simulate_stateful_ledger
            result = fixture._run(
                prices,
                np.ones_like(prices),
                portfolio_policy=lambda state: ledger.PortfolioTarget(
                    weights[state.day]
                ),
                config=config,
                cdi=np.full(days, 0.0004),
                initial_reference_price=np.full(3, 100.0),
            )
            results.append(result)
            account = accounts.PortfolioAccount.empty(np.full(4, 100.0), config=config)
            history = []
            for day in range(days):
                record = account.step(
                    accounts.tensor(np.r_[weights[day], 0]),
                    day=day,
                    close=np.r_[prices[day], 100],
                    cdi=0.0004,
                    session_date=result.dates[day],
                    annual_borrow=[0.04, 0.04, 0.04, 0],
                    loan_reference=np.full(4, 100.0),
                    terminal=day == days - 1,
                )
                history.append(
                    [
                        float(record[k])
                        for k in (
                            "nav",
                            "borrow",
                            "cost",
                            "interest",
                            "borrow_liability",
                        )
                    ]
                    + [float(account.cash), float(account.restricted.sum())]
                )
            account_results.append(np.array(history))
        arrays = []
        for key, value in vars(results[0]).items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(
                    value, getattr(results[1], key), strict=True, err_msg=key
                )
                arrays.append(key)
        for key in ("fills", "intended_orders", "loan_charges"):
            assert [asdict(x) for x in getattr(results[0], key)] == [
                asdict(x) for x in getattr(results[1], key)
            ]
        np.testing.assert_array_equal(
            account_results[0], account_results[1], strict=True
        )
        outcomes.append(
            dict(
                mixed=mixed,
                identical_arrays=arrays,
                fills=len(results[0].fills),
                nonzero_loan_charges=len(results[0].loan_charges),
                terminal_nav=float(results[0].nav[-1]),
            )
        )
    fixture.simulate_stateful_ledger = stateful_ledger.simulate_stateful_ledger
    receipt = dict(
        previous_source_commit=BASE,
        status="two_unchanged_books_bit_identical_in_both_accounts",
        outcomes=outcomes,
        reproducer_sha256=sha256_file(Path(__file__)),
        baseline_sources={p.name: sha256_file(p) for p in root.glob("baseline_*.py")},
    )
    output = root / "result.json"
    digest = write_json_atomic(output, receipt)
    pointer["loan_election_unchanged"] = {"path": str(output), "sha256": digest}
    write_json_atomic(PROJECT / "docs/v2_economic_data_scaling_run.json", pointer)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "outcomes": [
                    {k: v for k, v in x.items() if k != "identical_arrays"}
                    for x in outcomes
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
