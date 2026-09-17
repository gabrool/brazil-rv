from dataclasses import replace
from pathlib import Path
import runpy

import numpy as np
import pytest
import torch

from brazil_rv.execution.allocation import AllocationConfig, allocate
from brazil_rv.execution.portfolio_policy import policy_ledger_config
from brazil_rv.v2.opportunity_research import rolling_predictions
from brazil_rv.v2.performance import align_benchmarks, performance


def test_usd_conversion_calendar_cash_and_no_future_fx_fill():
    args = dict(
        dates=["2024-01-08", "2024-01-09"],
        previous_date="2024-01-05",
        fx_dates=["2024-01-05", "2024-01-09"],
        fx=[5.0, 5.5],
        rate_dates=["2024-01-05", "2024-01-08"],
        annual_us_percent=[3.6, 7.2],
    )
    bench = align_benchmarks(**args)
    assert bench["us_cash"] == pytest.approx([(1.0001) ** 3 - 1, 0.0002])
    assert bench["fx_ratio"] == pytest.approx([1, 5 / 5.5])
    assert bench["fx_age_days"].tolist() == [0, 3, 0]
    result = performance([0.01, 0.10], [0.001, 0.001], bench)
    assert result["absolute_compounded_return_usd"] == pytest.approx(0.01)
    assert result["fx_carried_endpoint_count"] == 1
    # A future revision/observation can never change the previous endpoint.
    args["fx"] = [5.0, 500.0]
    assert align_benchmarks(**args)["fx_ratio"][0] == 1


def test_zero_variance_and_initial_loss_drawdown():
    flat = performance([0.0, 0.0], [0.0, 0.0])
    assert flat["sharpe_brl_minus_zero"] is None
    assert flat["absolute_brl_flat_fraction"] == 1
    loss = performance([-0.2, 0.1], [0.0, 0.0])
    assert loss["maximum_drawdown_brl"] == pytest.approx(-0.2)
    assert loss["absolute_brl_winning_fraction"] == 0.5


def test_sector_constraints_preserve_unknown_names_and_inventory():
    mu = torch.tensor([0.005, -0.003, 0.006, -0.002], dtype=torch.float64)
    args = dict(
        previous=torch.zeros(4, dtype=torch.float64),
        beta=np.ones(4),
        idiosyncratic_variance=np.full(4, 0.0004),
        market_variance=0.0001,
        daily_borrow=np.zeros(4),
        lower=torch.full((4,), -0.05, dtype=torch.float64),
        upper=torch.full((4,), 0.05, dtype=torch.float64),
    )
    config = replace(AllocationConfig(), net_cap=0.45, beta_cap=0.45)
    original = allocate(mu, config=config, **args)
    neutral = allocate(
        mu,
        config=replace(config, sector_net_cap=0.0),
        sector_exposure=np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 1.0]]),
        **args,
    )
    assert abs(neutral[0]) < 1e-7  # singleton sector must hold no net exposure
    assert abs(neutral[1] + neutral[3]) < 1e-7
    assert neutral[2] > 0.049  # unknown-sector stock remains eligible
    assert original[0] > 0.049
    assert (
        policy_ledger_config(planned_absolute_net_cap=0.45).planned_absolute_net_cap
        == 0.45
    )


def test_supervised_fit_maturity_and_future_mutation():
    rng = np.random.default_rng(31)
    x = rng.normal(size=(620, 4))
    valid = np.ones_like(x, dtype=bool)
    y = x[:, :3] * 0.001 + rng.normal(size=(620, 3)) * 0.0001
    blocks = {"F5": np.arange(500, 620)}
    p, u, f = rolling_predictions(x, valid, y, blocks, 0)
    # First decision is 500: origin 494 matures at close 499. Origin 495 is
    # not yet usable even though the labelled return is present in the file.
    assert f["F5"]["last_origin"] == 494
    mutated = y.copy()
    mutated[495:] = 12345
    x2 = x.copy()
    x2[501:] = 1e6
    p2, u2, _ = rolling_predictions(x2, valid, mutated, blocks, 0)
    assert p2[500] == pytest.approx(p[500], abs=1e-12)
    assert u2[500] == pytest.approx(u[500], abs=1e-12)
    assert np.mean((p[500:] - y[500:]) ** 2) < np.mean((u[500:] - y[500:]) ** 2) / 10


def test_directional_market_forecast_enters_stock_and_hedge_once(monkeypatch):
    import brazil_rv.execution.portfolio_policy as policy
    from brazil_rv.v2.portfolio_inputs import Calibration
    from test_portfolio_policy import policy_fixture

    data = policy_fixture()
    data.beta[:] = 1.7
    model = policy.CalibratedPolicy(
        Calibration(np.zeros(1), np.ones(1), np.zeros(1), 0.0)
    )
    model.market_return_for = lambda day: 0.001
    recorded = {}

    def capture(preference, previous, **kwargs):
        recorded["preference"] = preference
        return torch.zeros_like(previous)

    monkeypatch.setattr(policy, "allocate", capture)
    account = data.initial_account(0, policy_ledger_config())
    policy.account_decision(data, model, account, 0)
    assert recorded["preference"][-1] == pytest.approx(0.001)
    assert recorded["preference"][:-1].numpy() == pytest.approx(
        np.full(len(recorded["preference"]) - 1, 0.0017)
    )


def test_reliability_followup_constant_fallback_and_future_isolation():
    module = runpy.run_path(
        str(
            Path(__file__).resolve().parents[2] / "ops/audit_opportunity_reliability.py"
        )
    )
    rng = np.random.default_rng(51)
    x = rng.normal(size=(540, 3))
    valid = np.ones_like(x, dtype=bool)
    y = np.zeros((540, 2))
    blocks = {"F5": np.arange(500, 540)}
    first, base, fits = module["predict"](x, valid, y, blocks, 0)
    assert fits["F5"]["selected_penalty"] == [None, None]
    y[495:] = 99999
    second, _, changed = module["predict"](x, valid, y, blocks, 0)
    np.testing.assert_array_equal(first[500:], second[500:])
    assert changed["F5"]["last_origin"] == 494
    np.testing.assert_array_equal(first[500:], base[500:])
