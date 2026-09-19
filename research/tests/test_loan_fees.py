from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest

from brazil_rv.execution.loan_fees import loan_fee_rates
from brazil_rv.execution.portfolio_account import tensor
from brazil_rv.execution.portfolio_policy import exact_replay, policy_ledger_config
from test_portfolio_policy import policy_fixture


@pytest.mark.parametrize(
    "modality,old,new",
    [
        ("normal", [0.001, 0.009], [0.0007, 0.0063]),
        ("direct", [0.0015, 0.011], [0.001, 0.0085]),
        ("otc", [0, 0.015], [0, 0.012]),
        ("compulsory", [0.0025, 0.0225], [0.0025, 0.0225]),
    ],
)
def test_dated_caps_follow_accrual_boundary_not_publication(modality, old, new):
    fees = loan_fee_rates(
        0.5,
        ["2022-07-07", "2022-11-10", "2022-11-11", "2022-11-14"],
        modality=modality,
    )
    np.testing.assert_array_equal(fees, [old, old, old, new])


def test_old_fixed_tariff_is_not_backfilled_with_rate_based_platform_fees():
    # 001/2020 announced, but did not implement, the new policy in January.
    dates = ["2016-07-18", "2020-01-02", "2020-10-23", "2020-10-26"]
    np.testing.assert_array_equal(
        loan_fee_rates(0.02, dates),
        [[0, 0.0025], [0, 0.0025], [0, 0.0025], [0.0004, 0.0036]],
    )
    assert loan_fee_rates(0.9, "2019-12-30", modality="compulsory")[1] == 0.005


def test_direct_component_caps_and_floors_cannot_be_combined():
    # At 4.2%, trading is capped; post-trading is not. A combined 95 bp
    # cap overcharges. At .244%, the separate post-trading floor still binds.
    np.testing.assert_array_equal(
        loan_fee_rates([0.042, 0.00244], "2024-01-02", modality="direct"),
        [[0.001, 0.00756], [0.000061, 0.00044]],
    )


@pytest.mark.parametrize("modality", ["normal", "direct", "otc"])
@pytest.mark.parametrize("uniform", [False, True])
def test_both_accounts_use_dated_equity_and_hedge_fees(modality, uniform):
    data = policy_fixture()
    days = len(data.inputs.dates)
    calendar = tuple(
        date(2022, 11, 7) + timedelta(days=i)
        for i in range(days * 2)
        if (date(2022, 11, 7) + timedelta(days=i)).weekday() < 5
    )[:days]
    inputs = replace(
        data.inputs,
        dates=calendar,
        raw_close=np.full_like(data.inputs.raw_close, 100.0),
        bova11_close=np.full(days, 100.0),
        annual_borrow_rate_by_name=np.full_like(data.inputs.raw_close, 0.4),
        hedge_annual_borrow_rate=np.full(days, 0.4),
        cdi_returns=np.zeros(days),
    )
    data = replace(data, inputs=inputs)
    config = policy_ledger_config(
        borrow_source="uniform" if uniform else "borrow_balance",
        annual_borrow_rate=0.3,
        borrow_fee_modality=modality,
        cost_bps_per_side=0,
        hedge_cost_bps_per_side=0,
    )
    stop = 9
    targets = np.zeros((stop, len(inputs.security_ids) + 1))
    targets[:-1, 0], targets[:-1, 1], targets[:-1, -1] = 0.02, -0.02, -0.01
    account = data.initial_account(0, config)
    records = [
        data.step(account, tensor(targets[day]), day, terminal=day == stop - 1)
        for day in range(stop)
    ]
    exact, _, _ = exact_replay(data, None, 0, stop, config=config, targets=targets)
    np.testing.assert_allclose(
        [record["nav"].item() for record in records], exact.nav, atol=1e-12, rtol=0
    )
    equity_rate = 0.3 if uniform else 0.4
    annual_fees = loan_fee_rates(equity_rate, calendar[:stop], modality=modality)
    opening_short = np.r_[0, -exact.signed_shares[:-1, 1] * 100]
    expected_fees = (
        opening_short
        * np.expm1(np.log1p(annual_fees) / 252).sum(axis=-1)
        / exact.start_nav
        * 1e4
    )
    np.testing.assert_allclose(exact.equity_borrow_fee_bps, expected_fees, atol=1e-12)
    np.testing.assert_allclose(
        exact.borrow_bps,
        exact.equity_borrow_raw_bps
        + exact.equity_borrow_fee_bps
        + exact.hedge_borrow_raw_bps
        + exact.hedge_borrow_fee_bps,
        atol=1e-12,
    )
    # The cap reduction affects November 14 onward without rewriting earlier NAV.
    assert exact.equity_borrow_fee_bps[5] < exact.equity_borrow_fee_bps[4]
    mutated_rates = inputs.annual_borrow_rate_by_name.copy()
    mutated_rates[7:] = 0.01
    changed_data = replace(
        data, inputs=replace(inputs, annual_borrow_rate_by_name=mutated_rates)
    )
    changed, _, _ = exact_replay(
        changed_data, None, 0, stop, config=config, targets=targets
    )
    np.testing.assert_array_equal(exact.nav[:7], changed.nav[:7])
