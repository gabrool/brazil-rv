from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest
import torch

from brazil_rv.execution.portfolio_account import PortfolioAccount, tensor
from brazil_rv.execution.spot_costs import MonthlySpotTariff, execution_bps
from brazil_rv.execution.stateful_ledger import LedgerConfig, PortfolioTarget
from brazil_rv.v2.portfolio_objective import clone_account
from test_portfolio_account import compare
from test_portfolio_ledger import replay
from test_v2_stateful_ledger import _config


def corporate(**changes):
    return _config(
        spot_cost_model="b3_spot_dated",
        hedge_cost_bps_per_side=0,
        initial_capital_brl=1000,
        **changes,
    )


def test_effective_date_and_no_bundled_or_fund_discount():
    cfg = corporate()
    for date in ["2021-02-02", "2023-10-05", "2024-12-30"]:
        assert execution_bps(cfg, date).tolist() == [[0, 0.5, 2.5, 0, 1]] * 2
    for date in ["2016-07-15", "2025-01-02"]:
        with pytest.raises(ValueError, match="admitted only"):
            execution_bps(cfg, date)
    with pytest.raises(ValueError, match="replace both"):
        LedgerConfig(spot_cost_model="b3_spot_dated")


def test_monthly_global_rate_validity_knowledge_and_missing_bound():
    late = MonthlySpotTariff("2017-01-03", "2017-02-01", "2017-01-16", 0.5, "fixture")
    january = MonthlySpotTariff(
        "2020-01-03", "2020-02-03", "2020-01-02", 0.366, "fixture"
    )
    cfg = corporate(monthly_spot_tariffs=(late, january))
    with pytest.raises(ValueError, match="explicit sourced bound"):
        execution_bps(cfg, "2017-01-13")
    for bound in (0.2, 0.5):
        bounded = replace(cfg, unrecovered_spot_trading_bps=bound)
        assert execution_bps(bounded, "2017-01-13")[0, 1:3].tolist() == [bound, 2.75]
        assert execution_bps(bounded, "2017-01-16")[0, 1:3].tolist() == [0.5, 2.75]
        assert execution_bps(bounded, "2020-02-03")[0, 1:3].tolist() == [0.366, 2.75]
        assert execution_bps(bounded, "2020-02-04")[0, 1:3].tolist() == [bound, 2.75]
        assert execution_bps(bounded, "2021-02-02")[0, 1:3].tolist() == [0.5, 2.5]
        # Future-rate mutation cannot affect any earlier interval or missing bound.
        changed = replace(
            bounded, monthly_spot_tariffs=(late, replace(january, trading_bps=0.4))
        )
        assert np.array_equal(
            execution_bps(changed, "2017-01-13"), execution_bps(bounded, "2017-01-13")
        )
        assert execution_bps(
            replace(bounded, spot_execution_phase="auction"), "2020-02-03"
        )[0, 1:3].tolist() == [0.7, 2.75]
    with pytest.raises(ValueError, match="overlapping"):
        replace(cfg, monthly_spot_tariffs=(late, late))
    with pytest.raises(ValueError, match="sourced"):
        replace(cfg, unrecovered_spot_trading_bps=0.1)


def test_historical_spot_and_custody_compose_in_both_accounts():
    from brazil_rv.execution.custody_fees import CustodyAssessment

    dates = np.array(
        [
            "2020-01-27",
            "2020-01-28",
            "2020-01-29",
            "2020-01-30",
            "2020-01-31",
            "2020-02-03",
            "2020-02-04",
            "2020-02-05",
        ]
    )
    cfg = corporate(
        monthly_spot_tariffs=(
            MonthlySpotTariff(
                "2020-01-03", "2020-02-03", "2020-01-02", 0.366, "fixture"
            ),
        ),
        unrecovered_spot_trading_bps=0.5,
        custody_assessments=(CustodyAssessment("2020-01-31", "2020-02-05"),),
    )
    close = np.full((8, 1), 100.0)
    targets = [[0.4]] * 3 + [[0.2]] * 3 + [[0], [0]]
    rows = compare(close, targets, dates=dates, config=cfg)
    result = replay(
        close,
        lambda s: PortfolioTarget(np.asarray(targets[s.day])),
        dates=dates,
        config=cfg,
    )
    for day, row in enumerate(rows):
        amount = sum(
            Decimal(str(f.gross_notional))
            for f in result.fills
            if f.fill_session == day
        )
        rate = Decimal(".366") if day < 6 else Decimal(".5")
        expected = [
            0,
            float(amount * rate / 10000),
            float(amount * Decimal("2.75") / 10000),
            0,
            float(amount / 10000),
        ]
        assert row["execution_charges"].numpy() == pytest.approx(
            expected, rel=0, abs=1e-12
        )
        assert result.execution_charges[day] == pytest.approx(
            expected, rel=0, abs=1e-12
        )
    assert result.custody_fee.sum() == result.custody_maintenance.sum() == 8.78
    assert result.custody_liability[4] == 8.78
    assert result.custody_payment[7] == 8.78


@pytest.mark.parametrize(
    "phase,broker,shortfall",
    [("regular", 0, 1), ("auction", 0, 1), ("regular", 0.5, 2)],
)
def test_both_accounts_signed_partial_missing_reversal_costs(phase, broker, shortfall):
    cfg = corporate(
        spot_execution_phase=phase,
        execution_brokerage_bps=broker,
        execution_shortfall_bps=shortfall,
    )
    close = np.array([[100, 100], [110, 90], [np.nan, 95], [105, 97], [103, 99]])
    targets = np.array([[0.4, -0.4], [-0.3, 0.3], [-0.3, 0.3], [0.2, -0.2], [0, 0]])
    fraction = np.ones_like(close)
    fraction[1] = 0.5
    records = compare(
        close, targets, cdi=np.full(5, 0.001), fractions=fraction, config=cfg
    )
    result = replay(
        close,
        lambda state: PortfolioTarget(targets[state.day]),
        cdi=np.full(5, 0.001),
        fill_fraction=fraction,
        config=cfg,
    )
    for day, row in enumerate(records):
        amount = sum(
            Decimal(str(f.gross_notional))
            for f in result.fills
            if f.fill_session == day
        )
        rates = [
            0,
            Decimal(".7") if phase == "auction" else Decimal(".5"),
            Decimal("2.5"),
            Decimal(str(broker)),
            Decimal(str(shortfall)),
        ]
        expected = np.array([float(amount * x / 10000) for x in rates])
        assert result.execution_charges[day] == pytest.approx(expected, abs=1e-12)
        assert row["execution_charges"].numpy() == pytest.approx(expected, abs=1e-12)
        assert sum(expected) == pytest.approx(float(row["cost"]), abs=1e-12)
    old = replay(
        close,
        lambda state: PortfolioTarget(targets[state.day]),
        cdi=np.full(5, 0.001),
        fill_fraction=fraction,
        config=replace(
            cfg,
            spot_cost_model="bundled",
            cost_bps_per_side=4,
            hedge_cost_bps_per_side=4,
        ),
    )
    assert result.intended_orders[0] == old.intended_orders[0]
    if phase == "regular" and broker == 0 and shortfall == 1:
        np.testing.assert_array_equal(result.nav, old.nav)
        np.testing.assert_array_equal(result.signed_shares, old.signed_shares)


def test_fee_gradient_hedge_funding_and_independent_copies():
    def run(weight, probe=False):
        cfg = corporate(beta_hedge=True, execution_brokerage_bps=0.5)
        account = PortfolioAccount.empty([100, 100, 100], config=cfg)
        rows = []
        for day in range(4):
            target = (
                torch.stack((weight, -weight, weight / 2))
                if day < 2
                else weight * torch.zeros(3)
            )
            rows.append(
                account.step(
                    target,
                    day=day,
                    close=[100, 100, 100],
                    cdi=0.001,
                    session_date="2024-01-02",
                    annual_borrow=[0, 0, 0],
                    loan_reference=[100, 100, 100],
                )
            )
            if day == 0:
                assert float(rows[0]["short_proceeds_interest"]) == 0
                assert float(rows[0]["execution_charges"].sum()) == pytest.approx(
                    4.5 / 1e4 * 1000 * float(weight) * 2.5
                )
                if probe:
                    other = clone_account(account)
                    other.trade_cash = other.trade_cash + 7
                    other.detach()
                    assert float(
                        other.trade_cash - account.trade_cash
                    ) == pytest.approx(7)
        return account.nav

    w = tensor(0.3).requires_grad_()
    nav = run(w, True)
    nav.backward()
    fd = (run(tensor(0.300001)) - run(tensor(0.299999))) / 0.000002
    assert float(w.grad) == pytest.approx(float(fd), abs=1e-5)


def test_actual_account_rejects_out_of_admission_date():
    account = PortfolioAccount.empty([100, 100], config=corporate())
    with pytest.raises(ValueError, match="admitted only"):
        account.step(
            tensor([0.4, 0]),
            day=0,
            close=[100, 100],
            cdi=0,
            session_date="2016-07-15",
            annual_borrow=[0, 0],
            loan_reference=[100, 100],
        )


def test_actual_settled_debit_spread_is_on_funded_debit_only():
    rows = []
    for spread in (0, 0.005, 0.01):
        cfg = corporate(annual_debit_spread=spread)
        records = compare(
            np.full((7, 1), 100.0),
            [[1.05]] * 5 + [[0], [0]],
            cdi=np.full(7, 0.001),
            config=cfg,
        )
        assert float(records[0]["debit_financing"]) == 0
        assert float(records[1]["debit_financing"]) == 0
        assert float(records[2]["debit_financing"]) == 0
        # Purchase settles after two sessions; next day's funding uses that close.
        funded = float(
            Decimal(1000) * Decimal("1.001") ** 3 - Decimal(1050) * Decimal("1.0004")
        )
        assert funded < 0
        assert float(records[3]["debit_financing"]) == pytest.approx(
            -funded * (0.001 + spread / 252), abs=1e-12
        )
        rows.append(records)
    for variant in rows[1:]:
        for day in range(3):
            assert float(variant[day]["nav"]) == float(rows[0][day]["nav"])
