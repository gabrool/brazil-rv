from dataclasses import replace

import numpy as np
import pytest
import torch

from brazil_rv.execution.allocation import AllocationConfig, allocate


def solve(mu, previous=None, *, config=AllocationConfig(), cap=0.05, borrow=0.0):
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
    )


def test_cash_and_linear_cost_no_trade_region():
    mu = torch.tensor([1e-6, -1e-6], dtype=torch.float64)
    assert solve(mu).detach().numpy() == pytest.approx([0, 0], abs=1e-7)
    prior = torch.tensor([0.02, -0.02], dtype=torch.float64)
    assert solve(prior * 0.002, prior).detach().numpy() == pytest.approx(
        prior.numpy(), abs=1e-7
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


def test_native_adjoint_matches_finite_difference_and_inventory_gradient():
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


@pytest.mark.parametrize("status", [2, 7])
def test_incomplete_solve_retry_preserves_solution_and_adjoint(monkeypatch, status):
    import osqp

    mu = torch.tensor([0.00011, -0.00011], dtype=torch.float64, requires_grad=True)
    expected = solve(mu)
    expected_gradient = torch.autograd.grad(expected[0], mu)[0]
    original = osqp.OSQP.solve
    calls = 0

    def first_iteration_limit(self, **kwargs):
        nonlocal calls
        result = original(self, **kwargs)
        calls += 1
        if calls == 1:
            result.info.status_val = status
        return result

    monkeypatch.setattr(osqp.OSQP, "solve", first_iteration_limit)
    actual = solve(mu)
    actual_gradient = torch.autograd.grad(actual[0], mu)[0]
    assert calls == 2
    assert actual.detach().numpy() == pytest.approx(expected.detach().numpy(), abs=1e-7)
    assert actual_gradient.numpy() == pytest.approx(expected_gradient.numpy(), abs=1e-3)
