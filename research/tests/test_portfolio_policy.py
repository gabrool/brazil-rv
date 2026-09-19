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


def test_benchmark_calibration_removes_common_market_return_without_losing_names():
    rng = np.random.default_rng(9)
    ranks = rng.normal(size=(90, 40, 3))
    valid = np.ones((90, 40), bool)
    beta = rng.uniform(0.4, 1.6, valid.shape)
    market = rng.normal(0.02, 0.01, 90)
    returns = beta * market[:, None]
    cdi = np.zeros(90)
    fit = np.arange(70)
    corrected = fit_calibration(
        ranks, valid, returns, valid, cdi, fit, benchmark_excess5=market, beta=beta
    )
    assert np.max(np.abs(corrected.predict(ranks))) < 1e-12
    assert corrected.diagnostics["fit_observations"] == 65 * 40
    market[65:] += 100
    changed = fit_calibration(
        ranks, valid, returns, valid, cdi, fit, benchmark_excess5=market, beta=beta
    )
    np.testing.assert_array_equal(corrected.coefficient, changed.coefficient)
    np.testing.assert_array_equal(corrected.covariance, changed.covariance)


def test_uncertainty_does_not_treat_duplicate_same_day_stocks_as_independent_dates():
    rng = np.random.default_rng(19)
    ranks = rng.normal(size=(100, 12, 3))
    valid = np.ones((100, 12), bool)
    returns = 0.002 * ranks[..., 0] + rng.normal(0, 0.03, (100, 1))
    first = fit_calibration(ranks, valid, returns, valid, np.zeros(100), np.arange(90))
    second = fit_calibration(
        np.repeat(ranks, 2, 1),
        np.repeat(valid, 2, 1),
        np.repeat(returns, 2, 1),
        np.repeat(valid, 2, 1),
        np.zeros(100),
        np.arange(90),
    )
    np.testing.assert_allclose(
        first.covariance, second.covariance, rtol=1e-9, atol=1e-16
    )
    assert first.covariance[0, 0] > 0


def test_validity_flags_keep_binary_units_when_fit_values_are_constant():
    data = policy_fixture()
    data.static[:10, :, [6, 7, 12]] = 1.0
    calibration = Calibration(np.zeros(3), np.ones(3), np.zeros(3), 0.0)
    model = PreferenceModel(data, calibration, np.arange(10))
    assert torch.equal(model.mean[[6, 7, 12]], torch.zeros(3))
    assert torch.equal(model.scale[[6, 7, 12]], torch.ones(3))


def test_tiny_inventory_does_not_amplify_marked_pnl_gradient(monkeypatch):
    import brazil_rv.execution.portfolio_policy as policy

    data = policy_fixture()
    account = data.initial_account(0, policy_ledger_config())
    account.shares[0] = 1e-16
    account.cost_basis[0] = 1e-15
    account.cost_basis.requires_grad_()
    # Isolate the actual state input before the shared network/allocator.
    monkeypatch.setattr(policy, "decide", lambda *args, **kwargs: args[7])
    pnl = policy.account_decision(data, None, account, 0)
    pnl.sum().backward()
    assert account.cost_basis.grad.abs().max() < 2
    assert pnl.abs().max() < 1e-10


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


@pytest.mark.parametrize("sector_cap", [None, 0.02])
def test_controller_reserves_delayed_claim_capacity_and_matches_actual_delivery(
    sector_cap,
):
    from brazil_rv.execution.allocation import AllocationConfig
    from brazil_rv.execution.share_distributions import ShareDelivery, ShareDistribution

    torch.set_num_threads(1)
    torch.manual_seed(11)
    data = policy_fixture()
    calibration = Calibration(np.zeros(3), np.ones(3), np.array([0.0008, 0, 0]), 0)
    model = PreferenceModel(data, calibration, np.arange(10))
    config = policy_ledger_config()
    allocation = AllocationConfig(sector_net_cap=sector_cap)
    names = len(data.inputs.security_ids)
    data.sectors = np.broadcast_to(
        np.array(["A", "B", "C"])[np.arange(names) % 3], data.valid.shape
    )
    account = data.initial_account(0, config)
    opening = account_decision(data, model, account, 0, allocation=allocation)
    source = int(opening[:-1].abs().argmax())
    assert opening[source].abs().item() > 0.001
    event = ShareDistribution(
        source,
        2,
        0,
        (
            ShareDelivery((source + 1) % names, 0.4, 4),
            ShareDelivery((source + 2) % names, 0.6, 5),
        ),
        "synthetic controller fixture",
    )
    closes = data.inputs.raw_close.copy()
    closes[2:, source] = np.nan
    data.inputs = replace(data.inputs, raw_close=closes, share_distributions=(event,))
    records, targets = [], []
    with torch.no_grad():
        for day in range(8):
            target = account_decision(data, model, account, day, allocation=allocation)
            if day in (3, 4):
                assert target[source].item() == pytest.approx(
                    account.weights[source].item(), abs=1e-8
                )
            targets.append(target.numpy())
            records.append(data.step(account, target, day, terminal=day == 7))
    exact, chosen, _ = exact_replay(data, model, 0, 8, allocation=allocation)
    np.testing.assert_allclose(chosen[:-1], targets[:-1], atol=1e-8)
    assert [record["nav"].item() for record in records] == pytest.approx(
        exact.nav, abs=1e-9
    )
    assert exact.undelivered_share_notional[2] > 0
    assert exact.undelivered_share_notional[5] == 0
    assert account.distributions == {}


@pytest.mark.parametrize("hedge_weight", [-0.02, 0.02])
@pytest.mark.parametrize("charge_fee", [True, False])
def test_hedge_borrow_rates_and_fee_scenarios_match_both_accounts(
    hedge_weight, charge_fee
):
    data = policy_fixture()
    rates = np.full(len(data.inputs.dates), np.nan)
    rates[4:8] = [0.0, 0.005, 0.03, np.nan]
    data.inputs = replace(data.inputs, hedge_annual_borrow_rate=rates)
    changes = (
        {}
        if charge_fee
        else dict(
            borrow_registration_fee_fraction=0.0,
            borrow_registration_fee_floor=0.0,
            borrow_registration_fee_cap=0.0,
        )
    )
    config = policy_ledger_config(
        cost_bps_per_side=0.0, hedge_cost_bps_per_side=0.0, **changes
    )
    start, stop = 3, 9
    targets = np.zeros((stop - start, len(data.inputs.security_ids) + 1))
    targets[:-1, -1] = hedge_weight
    account = data.initial_account(start, config)
    records = [
        data.step(account, tensor(target), day, terminal=day == stop - 1)
        for day, target in zip(range(start, stop), targets)
    ]
    exact, _, _ = exact_replay(data, None, start, stop, config=config, targets=targets)
    assert [r["nav"].item() for r in records] == pytest.approx(exact.nav, abs=1e-12)
    # Observed zero/below/above fallback rates, then missing. The first close
    # opens the hedge, so rent starts only on the following session.
    rent = np.array([0.0, 0.0, 0.005, 0.03, 0.02, 0.02])
    fee = np.array([0.0, 0.00025, 0.001, 0.006, 0.004, 0.004])
    expected = np.expm1(np.log1p(rent) / 252)
    if charge_fee:
        expected += np.expm1(np.log1p(fee) / 252)
    if hedge_weight > 0:
        expected[:] = 0.0
    opening_nav = np.r_[config.initial_capital_brl, exact.nav[:-1]]
    opening_hedge = np.r_[
        0.0, exact.hedge_signed_shares[:-1] * exact.hedge_mark_price[:-1]
    ]
    assert exact.hedge_borrow_bps == pytest.approx(
        np.abs(opening_hedge) / opening_nav * expected * 1e4, abs=1e-10
    )
    assert np.max(np.abs(exact.reconciliation_error)) < 1e-12
    # A later published rate cannot change an earlier accounting result.
    rates[7:] = 0.50
    changed, _, _ = exact_replay(
        data, None, start, stop, config=config, targets=targets
    )
    np.testing.assert_array_equal(changed.nav[:4], exact.nav[:4])


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


@pytest.mark.parametrize("weight", [-0.02, 0.02])
def test_unpriced_terminal_hedge_remains_in_both_accounts(weight):
    data = policy_fixture()
    close = data.inputs.bova11_close.copy()
    close[5:8] = np.nan
    data.inputs = replace(data.inputs, bova11_close=close)
    config = policy_ledger_config()
    targets = np.zeros((8, len(data.inputs.security_ids) + 1))
    targets[:-1, -1] = weight
    account = data.initial_account(0, config)
    records = [
        data.step(account, tensor(target), day, terminal=day == 7)
        for day, target in enumerate(targets)
    ]
    result, _, _ = exact_replay(data, None, 0, 8, targets=targets, config=config)
    assert [r["nav"].item() for r in records] == pytest.approx(result.nav, abs=1e-12)
    assert account.shares[-1].item() == pytest.approx(result.hedge_signed_shares[-1])
    assert result.hedge_signed_shares[-1] != 0
    assert result.terminal_unpriced_hedge_notional > 0
    assert not [f for f in result.fills if f.fill_session >= 5]
    no_haircut, _, _ = exact_replay(
        data, None, 0, 8, targets=targets, config=replace(config, unpriced_haircut=0.0)
    )
    np.testing.assert_array_equal(result.nav, no_haircut.nav)
    assert result.fills == no_haircut.fills
    assert result.unpriced_haircut_scenario_nav[-1] < no_haircut.nav[-1]


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
