import copy

import numpy as np
import torch

from brazil_rv.execution.opportunity_policy import OpportunityPolicy
from brazil_rv.execution.portfolio_policy import (
    CalibratedPolicy,
    account_decision,
    exact_replay,
    policy_ledger_config,
)
from brazil_rv.v2.controller_context import matured_shadow
from brazil_rv.v2.controller_synthetic import synthetic_data
from brazil_rv.v2.portfolio_inputs import Calibration


def test_shadow_uses_only_labels_matured_before_the_decision():
    rng = np.random.default_rng(12)
    ranks = rng.normal(size=(100, 20, 3))
    valid = np.ones((100, 20), bool)
    outcomes = rng.normal(0, 0.02, valid.shape)
    first = matured_shadow(ranks, valid, outcomes, valid)
    outcomes[65:] += 100
    second = matured_shadow(ranks, valid, outcomes, valid)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a[:71], b[:71])
    assert not np.array_equal(first[0][71:], second[0][71:])
    assert not first[1][:6].any()


def test_controller_initialization_and_scalers_preserve_the_fit_boundary():
    torch.set_num_threads(1)
    data, _, _ = synthetic_data(days=80)
    calibration = Calibration(np.zeros(3), np.ones(3), np.full(3, 0.001 / 3), 0.0)
    names = np.arange(32)
    state = torch.zeros((32, 10), dtype=torch.float64)
    for kind in ("reliability", "stateful"):
        original = OpportunityPolicy(data, calibration, np.arange(40), kind=kind)
        changed = copy.deepcopy(data)
        changed.static[40:, :, [8, 9, 10]] += 10000
        changed.context.common[40:] = 5000
        future = OpportunityPolicy(changed, calibration, np.arange(40), kind=kind)
        for key in ("common_center", "common_scale", "static_center", "static_scale"):
            torch.testing.assert_close(
                getattr(original, key), getattr(future, key), rtol=0, atol=0
            )
        baseline = CalibratedPolicy(calibration).preference_for(data, 20, names, state)
        torch.testing.assert_close(
            original.preference_for(data, 20, names, state), baseline, rtol=0, atol=0
        )


def test_conditional_policy_and_cash_switch_match_actual_account_transitions():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    data, _, _ = synthetic_data(days=80)
    calibration = Calibration(np.zeros(3), np.ones(3), np.full(3, 0.001 / 3), 0.0)
    model = OpportunityPolicy(data, calibration, np.arange(40), kind="stateful")
    # Nonzero conditional preferences exercise the new path, not only epoch zero.
    with torch.no_grad():
        model.network[-1].weight.fill_(0.0001)
    schedule = {d: model if d < 7 else None for d in range(12)}
    account = data.initial_account(0, policy_ledger_config())
    nav = []
    with torch.no_grad():
        for day in range(12):
            target = account_decision(data, schedule[day], account, day)
            data.step(account, target, day, terminal=day == 11)
            nav.append(float(account.nav))
    result, targets, _ = exact_replay(data, schedule, 0, 12)
    np.testing.assert_allclose(result.nav, nav, atol=1e-8, rtol=0)
    assert result.cost_bps[7] > 0
    assert np.abs(targets[7:]).max() == 0
