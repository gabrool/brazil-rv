from dataclasses import fields

import numpy as np
import pytest
import torch

from brazil_rv.execution.portfolio_policy import exact_replay, policy_ledger_config
from brazil_rv.v2.controller_synthetic import synthetic_data
from brazil_rv.v2.portfolio_objective import (
    NeuralPreference,
    TensorPreference,
    clone_account,
    full_preferences,
    smooth_ranks,
    utility_path,
)


def test_neural_preference_never_consumes_future_labels_and_keeps_cardinal_units():
    from brazil_rv.v2.config import ModelConfig
    from brazil_rv.v2.contract import HORIZONS
    from brazil_rv.v2.model import DailyMultiHorizonModel
    from brazil_rv.v2.portfolio_inputs import Calibration

    torch.set_num_threads(1)
    model = DailyMultiHorizonModel(
        ModelConfig(
            slow_feature_count=4,
            current_feature_count=0,
            disable_fast_stream=True,
            dropout=0,
        )
    )
    network = NeuralPreference(
        model,
        characteristic=False,
        horizons=HORIZONS,
        calibration=Calibration(np.zeros(1), np.ones(1), np.array([0.0001]), 0.0),
    ).eval()
    values = torch.randn(2, 8, 60, 4)
    batch = {
        "slow_features": values,
        "slow_feature_mask": torch.ones_like(values, dtype=torch.bool),
        "slow_history_mask": torch.ones(2, 8, 60, dtype=torch.bool),
        "slow_feature_age_sessions": torch.zeros_like(values),
        "active_mask": torch.ones(2, 8, dtype=torch.bool),
        "targets": torch.randn(2, 8, 5),
        "target_mask": torch.ones(2, 8, 5, dtype=torch.bool),
    }
    before = network(batch)
    changed = network(
        {
            **batch,
            "targets": -100 * batch["targets"],
            "target_mask": torch.zeros_like(batch["target_mask"]),
        }
    )
    for a, b in zip(before[:2], changed[:2]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with torch.no_grad():
        model.economic_head.bias.fill_(5.0)
    after = network(batch)
    # A common return forecast survives allocation inputs; ranking it away
    # would erase this exact five-basis-point shift.
    torch.testing.assert_close(
        after[1] - before[1], torch.full((2, 8), 0.0005), atol=1e-10, rtol=1e-6
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


def test_training_sam_restarts_state_and_carries_clean_inventory_between_blocks():
    from brazil_rv.v2.portfolio_objective_training import epoch

    torch.set_num_threads(1)
    data, _, _ = synthetic_data(days=35)

    class Cache:
        names = torch.arange(32).expand(35, -1)

        def gather(self, positions):
            return torch.tensor(data.ranks[positions].mean(-1), dtype=torch.float32)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.coefficient = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, x):
            score = x * self.coefficient
            return (
                score[..., None].expand(-1, -1, 3),
                score * 0.0003,
                (self.coefficient - 0.8).square(),
            )

    model = Model()
    optimizer = torch.optim.AdamW(
        [{"params": model.parameters(), "lr_multiplier": 1.0}], lr=1e-4
    )
    original = data.step
    starts = []

    def record(account, target, day, **kwargs):
        if day in (0, 32):
            starts.append(
                (day, float(account.nav.detach()), account.shares.detach().clone())
            )
        return original(account, target, day, **kwargs)

    data.step = record
    epoch(
        model,
        model,
        Cache(),
        np.arange(35),
        data,
        optimizer,
        None,
        "C6",
        "hybrid",
        1,
        0.01,
    )
    assert [r[0] for r in starts] == [0, 0, 32, 32]
    for a, b in ((starts[0], starts[1]), (starts[2], starts[3])):
        assert a[1] == b[1]
        torch.testing.assert_close(a[2], b[2], rtol=0, atol=0)
    assert starts[2][2].abs().sum() > 0
