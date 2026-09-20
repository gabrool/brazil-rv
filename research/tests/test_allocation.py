from dataclasses import replace

import numpy as np
import pytest
import torch

from brazil_rv.execution.allocation import AllocationConfig, allocate


def solve(
    mu,
    previous=None,
    *,
    config=AllocationConfig(),
    cap=0.05,
    borrow=0.0,
    uncertainty=None,
):
    n = len(mu)
    return allocate(
        mu,
        torch.zeros(n, dtype=torch.float64) if previous is None else previous,
        beta=np.linspace(0.7, 1.3, n),
        idiosyncratic_variance=np.full(n, 0.0004),
        market_variance=0.0001,
        daily_borrow=np.full(n, borrow),
        lower=torch.full((n,), -cap, dtype=torch.float64),
        upper=torch.full((n,), cap, dtype=torch.float64),
        config=config,
        forecast_uncertainty=uncertainty,
    )


def test_cash_and_linear_cost_no_trade_region():
    mu = torch.tensor([1e-6, -1e-6], dtype=torch.float64)
    np.testing.assert_array_equal(solve(mu).detach().numpy(), [0, 0])
    prior = torch.tensor([0.02, -0.02], dtype=torch.float64)
    np.testing.assert_array_equal(
        solve(prior * 0.002, prior).detach().numpy(), prior.numpy()
    )


def test_joint_constraints_and_borrow_cost():
    mu = torch.linspace(-0.05, 0.05, 80, dtype=torch.float64)
    result = solve(mu).numpy()
    beta = np.linspace(0.7, 1.3, len(mu))
    assert np.abs(result).sum() <= 2.25 + 1e-6
    assert abs(result.sum()) <= 0.05 + 1e-6
    assert abs(beta @ result) <= 0.05 + 1e-6
    assert np.abs(result).max() <= 0.05 + 1e-6
    assert solve(torch.tensor([-0.0001]), borrow=0.001).item() == pytest.approx(
        0, abs=1e-7
    )


def test_adjoint_matches_finite_difference_and_inventory_gradient():
    mu = torch.tensor([0.00009, -0.000085, 0.000015], dtype=torch.float64)
    mu.requires_grad_()
    previous = torch.tensor([0.005, -0.01, 0.001], dtype=torch.float64)
    previous.requires_grad_()
    direction = torch.tensor([0.4, -0.3, 0.9], dtype=torch.float64)
    value = (solve(mu, previous) * direction).sum()
    derivatives = torch.autograd.grad(value, (mu, previous))
    for variable, analytical in zip((mu, previous), derivatives, strict=True):
        numerical = []
        for i in range(len(variable)):
            step = torch.zeros_like(variable)
            step[i] = 1e-8
            args_plus = (
                (mu + step, previous) if variable is mu else (mu, previous + step)
            )
            args_minus = (
                (mu - step, previous) if variable is mu else (mu, previous - step)
            )
            plus = (solve(*args_plus) * direction).sum().item()
            minus = (solve(*args_minus) * direction).sum().item()
            numerical.append((plus - minus) / 2e-8)
        assert analytical.numpy() == pytest.approx(numerical, rel=2e-3, abs=2e-3)
    assert derivatives[0].abs().sum() > 0
    assert derivatives[1].abs().sum() > 0


def test_higher_cost_reduces_turnover_without_minimum_gross():
    mu = torch.tensor([0.00011, -0.00011], dtype=torch.float64)
    base = solve(mu).abs().sum()
    costly = solve(mu, config=replace(AllocationConfig(), cost_bps=8)).abs().sum()
    assert costly < base


def test_forecast_uncertainty_can_choose_cash_without_changing_return_risk():
    mu = torch.tensor([0.00011, -0.00011], dtype=torch.float64)
    assert solve(mu).abs().sum() > 0.001
    protected = solve(mu, uncertainty=torch.full_like(mu, 0.0002))
    assert protected.abs().sum() < 1e-7


@pytest.mark.parametrize("seed", [3, 19, 47])
def test_stock_hedge_active_constraints_and_uncertainty_gradient(seed):
    rng = np.random.default_rng(seed)
    n = 40
    beta = rng.uniform(0.5, 1.5, n)
    diagonal = rng.uniform(0.0001, 0.0008, n)
    diagonal[-1] = 1e-8  # actual hedge conditioning, absent from old small tests
    beta[-1] = 1
    previous = torch.tensor(rng.uniform(-0.03, 0.03, n))
    mu = torch.tensor(rng.normal(0, 0.0007, n), requires_grad=True)
    u = torch.tensor(rng.uniform(0.00001, 0.00003, n), requires_grad=True)
    direction = torch.tensor(rng.normal(size=n))
    perturbation = torch.tensor(rng.normal(size=n) * 1e-8)

    def value(m, uncertainty):
        return (
            allocate(
                m,
                previous,
                beta=beta,
                idiosyncratic_variance=diagonal,
                market_variance=0.0002,
                daily_borrow=np.full(n, 0.0001),
                lower=torch.full((n,), -0.05),
                upper=torch.full((n,), 0.05),
                forecast_uncertainty=uncertainty,
            )
            @ direction
        )

    derivatives = torch.autograd.grad(value(mu, u), (mu, u))
    for index, analytic in enumerate(derivatives):
        plus = (
            value(mu + perturbation, u) if index == 0 else value(mu, u + perturbation)
        )
        minus = (
            value(mu - perturbation, u) if index == 0 else value(mu, u - perturbation)
        )
        assert (analytic @ perturbation).item() == pytest.approx(
            ((plus - minus) / 2).item(),
            rel=5e-3,
            abs=2e-8,
        )
