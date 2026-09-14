from dataclasses import replace

import numpy as np
import pytest
import torch

from brazil_rv.execution.portfolio_account import tensor
from brazil_rv.execution.portfolio_policy import (
    PolicyData,
    PreferenceModel,
    account_decision,
    exact_replay,
    policy_ledger_config,
    portfolio_variance,
)
from brazil_rv.v2.portfolio_inputs import Calibration, causal_risk, fit_calibration
from brazil_rv.v2.train import rank_average_ensemble
from test_v2_evaluate import _fixture


def policy_fixture():
    original = _fixture()
    inputs = replace(
        original,
        scores=rank_average_ensemble([original.scores], original.score_mask),
        initial_unresolved_action=np.zeros(original.active.shape[1], bool),
    )
    days, names = inputs.active.shape
    return PolicyData(
        inputs,
        np.ones((days, names)),
        np.full((days, names), 0.0004),
        np.full(days, 0.0001),
        np.full(days, 0.0004),
        np.full((days, names), 100.0),
    )


def test_causal_risk_does_not_see_today_or_future_and_missing_names_survive():
    rng = np.random.default_rng(42)
    wealth = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, (100, 4)), axis=0))
    seen = np.ones_like(wealth, bool)
    seen[:, 3] = False
    hedge = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 100)))
    beta, active = np.ones_like(wealth), np.ones_like(seen)
    first = causal_risk(wealth, seen, hedge, beta, active)
    wealth[80:] *= 100
    hedge[80:] *= 2
    second = causal_risk(wealth, seen, hedge, beta, active)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a[:81], b[:81])
    assert np.isfinite(first[0]).all() and (first[0] > 0).all()
    assert not first[2][:, 3].any()


def test_ridge_is_fit_only_and_does_not_read_crossing_endpoints():
    rng = np.random.default_rng(42)
    ranks = rng.normal(size=(30, 6, 3))
    valid = np.ones((30, 6), bool)
    returns = 0.01 * ranks[..., 0]
    cdi = np.full(30, 0.0004)
    fit = np.arange(20)
    first = fit_calibration(ranks, valid, returns, valid, cdi, fit)
    returns[15:] = np.nan
    ranks[20:] = 1e9
    cdi[20:] = 0.99
    second = fit_calibration(ranks, valid, returns, valid, cdi, fit)
    np.testing.assert_array_equal(first.coefficient, second.coefficient)
    assert first.intercept == second.intercept


def test_shared_controller_and_independent_exact_ledger_agree():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    data = policy_fixture()
    calibration = Calibration(np.zeros(3), np.ones(3), np.array([0.0008, 0, 0]), 0.0)
    model = PreferenceModel(data, calibration, np.arange(10))
    config = policy_ledger_config()
    account = data.initial_account(0, config)
    records, targets = [], []
    with torch.no_grad():
        for day in range(12):
            target = account_decision(data, model, account, day)
            targets.append(target.numpy())
            records.append(data.step(account, target, day, terminal=day == 11))
    exact, chosen, previous = exact_replay(data, model, 0, 12)
    np.testing.assert_allclose(chosen[:-1], targets[:-1], atol=1e-8)
    assert [r["nav"].item() for r in records] == pytest.approx(exact.nav, abs=1e-9)
    assert np.max(np.abs(exact.reconciliation_error)) < 1e-10
    assert previous.shape == (12, len(data.inputs.security_ids) + 1)


def test_preference_gradient_survives_sequential_account_and_detach_preserves_value():
    torch.set_num_threads(1)
    torch.manual_seed(29)
    data = policy_fixture()
    calibration = Calibration(
        np.zeros(3), np.ones(3), np.array([0.00015, 0.00012, 0]), 0.0
    )
    model = PreferenceModel(data, calibration, np.arange(10))
    account = data.initial_account(0, policy_ledger_config())
    objective = tensor(0.0)
    for day in range(6):
        risk = portfolio_variance(data, day, account.weights)
        target = account_decision(data, model, account, day)
        result = data.step(account, target, day)
        objective += (result["net_excess"] - 2.5 * risk) * 1e4
    objective.backward()
    gradients = torch.cat([p.grad.flatten() for p in model.parameters()])
    assert torch.isfinite(gradients).all()
    assert gradients.norm() > 1e-6
    before = account.nav.item()
    account.detach()
    assert account.nav.item() == before
    assert not account.shares.requires_grad


def test_missing_prior_cdi_stays_masked_and_never_uses_current_accrual():
    data = policy_fixture()
    prior = data.prior_cdi.copy()
    prior[0] = np.nan
    first = PolicyData(
        data.inputs, data.beta, data.diagonal, data.factor, prior, data.references
    )
    cdi = data.inputs.cdi_returns.copy()
    cdi[0] = 0.25
    second = PolicyData(
        replace(data.inputs, cdi_returns=cdi),
        data.beta,
        data.diagonal,
        data.factor,
        prior,
        data.references,
    )
    np.testing.assert_array_equal(first.static, second.static)
    assert np.isfinite(first.static[first.valid]).all()
    assert (first.static[0, :, -2:] == 0).all()


def test_policy_training_selects_only_on_exact_ledger_and_reuses_completed_trial(
    tmp_path, monkeypatch
):
    from brazil_rv.v2 import portfolio_training as training

    data = policy_fixture()
    (tmp_path / "frozen_design.json").write_text("{}")
    monkeypatch.setattr(training, "_git_identity", lambda: {"commit": "fixture"})
    monkeypatch.setattr(
        training,
        "windows",
        lambda *args: {
            "fit": np.arange(12),
            "selection": np.arange(15, 20),
            "evaluation": np.arange(21, 25),
        },
    )
    result = training.fit_policy(
        data, tmp_path, "S0", "F1", 11, "fixture", max_epochs=2
    )
    assert result["status"] == "completed"
    assert result["epochs_completed"] == 2
    assert len(result["history"]) == 3
    assert np.isfinite([r["utility_bps"] for r in result["history"]]).all()
    selected = tmp_path / "policies/S0/F1/seed_11/selected.pt"
    checkpoint = torch.load(selected, weights_only=True)
    assert checkpoint["epoch"] == result["selected_epoch"]
    assert (
        training.fit_policy(data, tmp_path, "S0", "F1", 11, "fixture", max_epochs=2)
        == result
    )
