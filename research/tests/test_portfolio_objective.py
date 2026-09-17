from dataclasses import fields

import numpy as np
import pytest
import torch

from brazil_rv.execution.portfolio_policy import exact_replay, policy_ledger_config
from brazil_rv.v2.controller_synthetic import synthetic_data
from brazil_rv.v2.portfolio_objective import (
    TensorPreference,
    clone_account,
    full_preferences,
    smooth_ranks,
    utility_path,
)


def test_rank_bridge_has_only_same_date_active_peers_and_finite_gradients():
    x = torch.tensor(
        [[[1.0], [2.0], [4.0], [500.0]], [[3.0], [7.0], [2.0], [-100.0]]],
        requires_grad=True,
    )
    active = torch.tensor([[True, True, True, False]] * 2)
    first = smooth_ranks(x, active)
    changed = x.detach().clone()
    changed[1] += 100
    changed[0, 3] = -1e8
    torch.testing.assert_close(
        first[0], smooth_ranks(changed, active)[0], rtol=0, atol=0
    )
    torch.testing.assert_close(first.sum(1), torch.zeros((2, 1)), atol=1e-6, rtol=0)
    first[0, 0].backward()
    assert x.grad[0, :3].abs().sum() > 0
    assert x.grad[1].abs().sum() == 0
    assert x.grad[0, 3] == 0


def test_identity_restore_preserves_gradients_without_padded_security_zero():
    x = torch.tensor([[0.1, 0.2, 100.0]], requires_grad=True)
    result = full_preferences(x, torch.tensor([[2, 0, -1]]), 4)
    torch.testing.assert_close(
        result, torch.tensor([[0.2, 0.0, 0.1, 0.0]], dtype=torch.float64)
    )
    result.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([[1.0, 1.0, 0.0]]))


def test_chronological_objective_matches_independent_ledger_and_sam_restart():
    torch.set_num_threads(1)
    data, _, _ = synthetic_data(days=32)
    values = torch.tensor(data.ranks[:12].mean(-1) * 0.0003)
    initial = data.initial_account(0, policy_ledger_config())
    first = clone_account(initial)
    loss, first, targets, nav = utility_path(
        data, values, first, np.arange(12), terminal=True
    )
    repeat = utility_path(
        data, values, clone_account(initial), np.arange(12), terminal=True
    )
    assert float(loss) == float(repeat[0])
    np.testing.assert_array_equal(targets, repeat[2])
    result, planned, _ = exact_replay(data, TensorPreference(values), 0, 12)
    np.testing.assert_allclose(nav, result.nav, atol=1e-8, rtol=0)
    np.testing.assert_allclose(targets, planned, atol=1e-8, rtol=0)
    assert initial.shares.count_nonzero() == 0
    # Truncation is not liquidation: the same observations across two blocks
    # preserve inventory, state, payments and exact P&L.
    _, state, a, na = utility_path(
        data, values[:5], clone_account(initial), np.arange(5)
    )
    _, state, b, nb = utility_path(
        data, values[5:], clone_account(state), np.arange(5, 12), terminal=True
    )
    np.testing.assert_allclose(np.r_[na, nb], nav, atol=1e-10, rtol=0)
    np.testing.assert_allclose(np.concatenate((a, b)), targets, atol=1e-10, rtol=0)
    for field in fields(state):
        left, right = getattr(first, field.name), getattr(state, field.name)
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, atol=1e-10, rtol=0)


def test_account_economic_gradient_matches_finite_difference():
    torch.set_num_threads(1)
    data, _, _ = synthetic_data(days=24)
    base = torch.tensor(data.ranks[:8].mean(-1) * 0.00012)
    coefficient = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)

    def value(scale):
        return utility_path(
            data,
            base * scale,
            data.initial_account(0, policy_ledger_config()),
            np.arange(8),
            terminal=True,
        )[0]

    loss = value(coefficient)
    loss.backward()
    gradient = float(coefficient.grad)
    eps = 1e-4
    difference = float((value(1 + eps) - value(1 - eps)) / (2 * eps))
    assert abs(gradient) > 1e-6
    assert gradient == pytest.approx(difference, rel=0.02, abs=1e-4)
    assert float(value(1 - 0.001 * np.sign(gradient))) < float(loss)
