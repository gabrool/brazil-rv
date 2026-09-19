"""Independent-path behavioral acceptance using the real QP and accounts."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import torch

from brazil_rv.execution.opportunity_policy import OpportunityPolicy
from brazil_rv.execution.allocation import AllocationConfig
from brazil_rv.execution.portfolio_account import tensor
from brazil_rv.execution.portfolio_policy import (
    CalibratedPolicy,
    PolicyData,
    account_decision,
    exact_replay,
    policy_ledger_config,
)
from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.controller_context import ControllerContext
from brazil_rv.v2.controller_training import initialize_stateful, learn, RECIPE
from brazil_rv.v2.evaluate import EvaluationInputs
from brazil_rv.v2.portfolio_inputs import Calibration
from brazil_rv.v2.portfolio_readouts import book_summary
from brazil_rv.v2.train import rank_average_ensemble


def synthetic_data(seed=201, days=640, names=32):
    """Regime is observed before action; future return noise uses separate draws.

    Every 16-session episode has an independently assigned rank ordering and
    announced opportunity/continuation state. Price shocks on day five are
    adverse to the original ranking; subsequent response is state-dependent.
    Borrow rates are known holding costs and vary independently by episode.
    """
    rng = np.random.default_rng(seed)
    signal, regime, cost = np.zeros((days, names)), np.zeros(days), np.zeros(days)
    states = np.concatenate(
        [rng.permutation([0.0, 1.0, 1.0, -1.0]) for _ in range((days + 63) // 64)]
    )
    for start in range(0, days, 16):
        stop = min(start + 16, days)
        signal[start:stop] = rng.permutation(np.linspace(-1, 1, names))
        regime[start:stop] = states[start // 16]
        cost[start:stop] = rng.choice([0.02, 0.02, 3.0])
    noise = rng.normal(0, 0.004, (days, names))
    returns = np.zeros_like(signal)
    returns[1:] = 0.003 * signal[:-1] * regime[:-1, None] + noise[1:]
    shock_days = np.arange(5, days, 16)
    # Zero-opportunity episodes contain only unpredictable noise. A systematic
    # adverse rank shock there would itself be a learnable short opportunity.
    shock_days = shock_days[regime[shock_days - 1] != 0]
    returns[shock_days] -= 0.02 * signal[shock_days - 1]
    close = 100 * np.cumprod(1 + returns, axis=0)
    dates, day = [], date(2021, 1, 4)
    while len(dates) < days:
        if day.weekday() < 5:
            dates.append(day)
        day += timedelta(days=1)
    active = np.ones((days, names), bool)
    shape = (days, names, 5)
    scores = np.repeat(signal[..., None], 5, -1)
    mask = np.ones(shape, bool)
    y, target_valid = np.zeros(shape), np.zeros(shape, bool)
    for h, span in enumerate((1, 2, 3, 5, 10)):
        y[:-span, :, h] = close[span:] / close[:-span] - 1
        target_valid[:-span, :, h] = True
    annual = np.broadcast_to(cost[:, None], active.shape).copy()
    inputs = EvaluationInputs(
        dates=tuple(dates),
        session_indices=np.arange(days),
        calendar_identity_sha256="synthetic",
        scores=rank_average_ensemble([scores], mask),
        score_mask=mask,
        scaled_midrank_targets=y,
        scaled_target_mask=target_valid,
        neutral_midrank_targets=y,
        neutral_target_mask=target_valid,
        shareholder_midrank_targets=y,
        shareholder_simple_returns=y,
        shareholder_target_mask=target_valid,
        price_midrank_targets=y,
        price_target_mask=target_valid,
        active=active,
        raw_close=close,
        action_shares_per_prior_share=np.ones_like(signal),
        action_cash_per_prior_share=np.zeros_like(signal),
        action_session_resolved=active,
        action_has_action=np.zeros_like(active),
        action_successor_index=np.broadcast_to(np.arange(names), active.shape).copy(),
        action_payment_session=np.full(active.shape, -1),
        security_ids=tuple(f"SYN-{n}" for n in range(names)),
        target_scale_sigma=np.full_like(signal, 0.01),
        prior_feature_values={"yang_zhang_vol_20": np.full_like(signal, 0.01)},
        cdi_returns=np.zeros(days),
        transfer_chronology_clean=True,
        annual_borrow_rate_by_name=annual,
        borrow_rate_imputed=np.zeros_like(active),
        borrow_rate_placeholder=np.zeros_like(active),
        shortable_by_borrow_source={"borrow_balance": active},
        borrow_source_label="synthetic_known_borrow",
        bova11_close=np.full(days, 100.0),
        # This synthetic market's published loan references equal the prior close.
        loan_reference_prices=np.column_stack(
            (np.vstack((close[0], close[:-1])), np.full(days, 100.0))
        ),
        bova11_manifest_sha256="synthetic",
        bova11_data_sha256="synthetic",
        hedge_beta=np.ones_like(signal),
        hedge_beta_valid=active,
        hedge_beta_manifest_sha256="synthetic",
        initial_reference_price=np.full(names, 100.0),
        initial_unresolved_action=np.zeros(names, bool),
        initial_hedge_reference_price=100.0,
    )
    data = PolicyData(
        inputs,
        np.ones_like(signal),
        np.full_like(signal, 0.0001),
        np.full(days, 0.0001),
        np.zeros(days),
        np.concatenate((np.full((1, names), 100.0), close[:-1])),
    )
    common = np.column_stack((regime == 0, regime == 1, regime == -1, cost > 1)).astype(
        np.float32
    )
    data.context = ControllerContext(
        common,
        np.ones_like(common, bool),
        (True,) * 4,
        ("zero_opportunity", "reversal", "continuation", "high_borrow"),
        np.zeros_like(signal, dtype=np.float32),
    )
    return data, regime, cost


class OraclePolicy(CalibratedPolicy):
    def preference_for(self, data, day, names, state):
        regime = float(data.context.common[day, 1] - data.context.common[day, 2])
        return 0.003 * tensor(data.ranks[day, names]).mean(-1) * regime


def assess(data, model, rows, regime, cost):
    start, stop = int(rows[0]), int(rows[-1] + 1)
    result, targets, previous = exact_replay(data, model, start, stop)
    summary, _ = book_summary(data, result, previous, start, start)
    gross = np.abs(targets).sum(1)
    local = np.arange(start, stop)
    settled = local % 16 >= 4
    return {
        "summary": summary,
        "terminal_residual_fraction": float(
            (
                result.unresolved_inventory_notional
                + abs(result.hedge_signed_notional[-1])
                + result.unresolved_receivable
                + result.unresolved_payable
            )
            / result.nav[-1]
        ),
        "gross_zero": float(gross[(regime[rows] == 0) & settled].mean()),
        "gross_useful": float(gross[(regime[rows] == 1) & settled].mean()),
        "gross_high_borrow": float(gross[(cost[rows] > 1) & settled].mean()),
        "gross_low_borrow": float(gross[(cost[rows] < 1) & settled].mean()),
    }


def behavioral_account_checks(data, model, rows):
    """Validate the trained conditional model, not just its zero initialization."""
    start, stop = int(rows[0]), int(rows[-1] + 1)
    exact, _, _ = exact_replay(data, model, start, stop)
    account = data.initial_account(start, policy_ledger_config())
    nav, trades, expensive_trades = [], [], []
    with torch.no_grad():
        for day in rows:
            day = int(day)
            target = account_decision(data, model, account, day)
            expensive = account_decision(
                data,
                model,
                account,
                day,
                allocation=replace(AllocationConfig(), cost_bps=12.0),
            )
            trades.append(float((target - account.weights).abs().sum()))
            expensive_trades.append(float((expensive - account.weights).abs().sum()))
            data.step(account, target, day, terminal=day == stop - 1)
            nav.append(float(account.nav))
    return {
        "max_account_nav_difference": float(
            np.max(np.abs(np.asarray(nav) - exact.nav))
        ),
        "mean_trade_cost4": float(np.mean(trades)),
        "mean_trade_cost12": float(np.mean(expensive_trades)),
        "largest_paired_trade_increase_cost12": float(
            np.max(np.asarray(expensive_trades) - trades)
        ),
    }


def run_synthetic(root, *, seed=11, kind="reliability", recipe=RECIPE):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    data, regime, cost = synthetic_data(days=1920)
    bounds = {
        "fit": np.arange(1152),
        "selection": np.arange(1168, 1472),
        "evaluation": np.arange(1488, 1920),
    }
    calibration = Calibration(np.zeros(3), np.ones(3), np.full(3, 0.001 / 3), 0.0)
    model = OpportunityPolicy(data, calibration, bounds["fit"], kind=kind)
    output = root / "phase2/behavioral_acceptance" / kind / f"seed_{seed}"
    parent = None
    if kind == "stateful":
        parent = initialize_stateful(
            model, root / "phase2/behavioral_acceptance/reliability/seed_11/selected.pt"
        )
    fit = learn(
        data,
        model,
        bounds,
        output,
        {
            "kind": kind,
            "seed": seed,
            "data_seed": 201,
            "scope": "synthetic",
            "conditional_parent_sha256": parent,
        },
        recipe=recipe,
    )
    rows = bounds["evaluation"]
    learned = assess(data, model, rows, regime, cost)
    control = assess(data, CalibratedPolicy(calibration), rows, regime, cost)
    oracle = assess(data, OraclePolicy(calibration), rows, regime, cost)
    behavior = behavioral_account_checks(data, model, rows)
    # Identical observable adverse inventory, differing announced response state.
    day = int(rows[0] + 8)
    account = data.initial_account(day, policy_ledger_config())
    names = np.flatnonzero(data.ranks[day].mean(-1) > 0)
    account.shares[names] = tensor(0.025 / account.marks[names].numpy())
    account.cash = account.cash - account.shares @ account.marks
    account.cost_basis[names] = account.shares[names] * account.marks[names] * 1.03
    account.entry_day[names] = day - 4
    original = data.context.common[day].copy()
    decisions = {}
    with torch.no_grad():
        for label, common in (
            ("reversal", [0, 1, 0, 0]),
            ("continuation", [0, 0, 1, 0]),
            ("zero", [1, 0, 0, 0]),
        ):
            data.context.common[day] = common
            target = account_decision(data, model, account, day)
            decisions[label] = float(target[names].sum())
    data.context.common[day] = original
    checks = {
        "profitable_independent_dates": learned["summary"]["mean"]["net_excess_bps"]
        > 0,
        "improves_on_unconditional": learned["summary"]["mean"]["utility_bps"]
        > control["summary"]["mean"]["utility_bps"],
        "less_zero_state_exposure": learned["gross_zero"]
        < 0.75 * control["gross_zero"],
        "differentiated_adverse_response": decisions["reversal"]
        > decisions["continuation"] + 0.05,
        "learned_checkpoint": fit["selected_epoch"] > 0 or parent is not None,
        "trained_account_parity": behavior["max_account_nav_difference"] < 1e-6,
        "known_cost_aware_inactivity": behavior["largest_paired_trade_increase_cost12"]
        < 1e-5,
        "no_material_unsettled_inventory": learned["terminal_residual_fraction"] < 1e-8,
    }
    result = {
        "checks": checks,
        "passed": all(checks.values()),
        "fit": fit,
        "learned": learned,
        "control": control,
        "oracle": oracle,
        "adverse_state_decisions": decisions,
        "behavioral_account_checks": behavior,
        "independent_evaluation_dates": len(rows),
        "synthetic_data_seed": 201,
        "conditional_parent_sha256": parent,
        "stateful_increment_selected": kind == "stateful" and fit["selected_epoch"] > 0,
        "episodes_by_split": {
            split: {
                str(state): int(np.count_nonzero(regime[indices][::16] == state))
                for state in (-1, 0, 1)
            }
            for split, indices in bounds.items()
        },
        "not_a_financial_result": True,
    }
    write_json_atomic(output / "acceptance.json", result)
    return result
