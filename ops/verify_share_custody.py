"""Bounded synthetic baseline agreement and full-population account timing."""

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from brazil_rv.execution import portfolio_account, stateful_ledger
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.corporate_actions import AlignedActionTerms

PROJECT = Path(__file__).resolve().parents[1]
BASE = "56adc098a746c27ad4d62e7ecfd1e13ea9cd1108"


def prior_module(name, root):
    relative = f"research/src/brazil_rv/execution/{name}.py"
    content = subprocess.check_output(
        ["git", "show", f"{BASE}:{relative}"], cwd=PROJECT
    )
    path = root / f"baseline_{name}.py"
    if name in {"stateful_ledger", "portfolio_account"}:
        content = content.replace(
            b"from .loan_contracts import",
            b"from ._custody_baseline_loan_contracts import",
        )
    path.write_bytes(content)
    qualified = f"brazil_rv.execution._custody_baseline_{name}"
    spec = importlib.util.spec_from_file_location(qualified, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


def config(module):
    return module.LedgerConfig(
        initial_capital_brl=10_000_000,
        borrow_source="uniform",
        volatility_balanced_entries=False,
        beta_hedge=False,
        annual_borrow_rate=0.04,
        short_proceeds_remuneration=1,
        cost_bps_per_side=4,
        planned_name_weight_cap=0.1,
        planned_absolute_net_cap=1.1,
    )


def book(module, names, days, *, short=True):
    phase = np.arange(names)[None, :] + np.arange(days)[:, None] * 0.03
    close = 100 * np.exp(0.02 * np.sin(phase))
    ones = np.ones_like(close)
    terms = AlignedActionTerms(
        ones,
        np.zeros_like(close),
        ones.astype(bool),
        np.zeros_like(close, bool),
        np.broadcast_to(np.arange(names), close.shape),
    )
    dates = np.busday_offset("2024-01-02", np.arange(days)).astype(object)
    signs = np.where(np.arange(names) % 2, -1, 1) if short else np.ones(names)
    started = time.perf_counter()
    result = module.simulate_stateful_ledger(
        dates=dates,
        scores=ones,
        score_mask=ones.astype(bool),
        active=ones.astype(bool),
        raw_close=close,
        action_terms=terms,
        action_payment_session=np.full(close.shape, -1),
        cdi_returns=np.full(days, 0.0004),
        loan_reference_prices=np.full((days, names + 1), 100.0),
        initial_reference_price=np.full(names, 100.0),
        config=config(module),
        portfolio_policy=lambda state: module.PortfolioTarget(signs * (0.8 / names)),
    )
    return result, time.perf_counter() - started


def account(module):
    weight = torch.tensor(0.003, dtype=torch.float64, requires_grad=True)
    signs = torch.tensor([*np.where(np.arange(243) % 2, -1, 1), 0], dtype=torch.float64)
    instance = module.PortfolioAccount.empty(
        np.full(244, 100.0), config=config(stateful_ledger)
    )
    started = time.perf_counter()
    for day in range(64):
        instance.step(
            signs * weight,
            day=day,
            close=np.full(244, 100.0),
            cdi=0.0004,
            session_date="2024-01-02",
            annual_borrow=np.full(244, 0.04),
            loan_reference=np.full(244, 100.0),
            terminal=day == 63,
        )
    instance.nav.backward()
    return dict(
        seconds=time.perf_counter() - started,
        nav=float(instance.nav.detach()),
        gradient=float(weight.grad),
    )


def main():
    pointer_path = PROJECT / "docs/v2_economic_data_scaling_run.json"
    pointer = json.loads(pointer_path.read_text())
    root = Path(pointer["root"]) / "custody_acceptance"
    root.mkdir(exist_ok=True)
    prior_module("loan_contracts", root)
    old = prior_module("stateful_ledger", root)
    old_account = prior_module("portfolio_account", root)
    # Rebind old accounts to their exact old loan implementation; dated tariffs
    # are the sole shared loan dependency and are unchanged.
    for name in ("loan_fees",):
        path = f"research/src/brazil_rv/execution/{name}.py"
        prior = subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=PROJECT)
        assert prior.decode().replace("\r\n", "\n") == (PROJECT / path).read_text()
    equal = {}
    fields = (
        "nav",
        "signed_shares",
        "cost_bps",
        "borrow_bps",
        "loan_payment",
        "loan_liability",
        "free_cash",
        "restricted_cash",
        "unsettled_cash",
        "turnover_fraction_nav",
        "gross_fraction_nav",
    )
    for short in [False, True]:
        a, _ = book(old, 12, 32, short=short)
        b, _ = book(stateful_ledger, 12, 32, short=short)
        same = {
            key: bool(np.array_equal(getattr(a, key), getattr(b, key)))
            for key in fields
        }
        same["fills"] = [asdict(x) for x in a.fills] == [asdict(x) for x in b.fills]
        same["intentions"] = [asdict(x) for x in a.intended_orders] == [
            asdict(x) for x in b.intended_orders
        ]
        a_charges = [asdict(x) for x in a.loan_charges if x.rent != 0 or x.fee != 0]
        b_charges = [asdict(x) for x in b.loan_charges]
        same["all_nonzero_original_entry_charges"] = a_charges == b_charges
        assert all(same.values())
        equal["long_short" if short else "long_only"] = same
    a, old_seconds = book(old, 243, 252)
    b, new_seconds = book(stateful_ledger, 243, 252)
    np.testing.assert_array_equal(a.nav, b.nav)
    before, after = account(old_account), account(portfolio_account)
    assert before["nav"] == after["nav"] and before["gradient"] == after["gradient"]
    payload = dict(
        reference_commit=BASE,
        unaffected_bit_identical=equal,
        scope="synthetic account correctness and CPU timing; not model profitability or neural-fit ETA",
        full_population=dict(
            names=243,
            days=252,
            old_seconds=old_seconds,
            new_seconds=new_seconds,
            fills=len(b.fills),
            old_charge_rows=len(a.loan_charges),
            new_charge_rows=len(b.loan_charges),
        ),
        training_account_64_sessions=dict(before=before, after=after),
        heldout_accessed=False,
        immutable_datasets_duplicated=False,
    )
    path = root / "result.json"
    digest = write_json_atomic(path, payload)
    pointer["custody_runtime_acceptance"] = dict(
        path=str(path), sha256=digest, reproducer_sha256=sha256_file(Path(__file__))
    )
    write_json_atomic(pointer_path, pointer)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
