from __future__ import annotations

import numpy as np
import torch

from brazil_rv.execution.config import ExecutionConfig
from brazil_rv.execution.experiment57 import (
    RulePathPolicy,
    build_rule_schedule,
    cross_fold_conditional_means,
    neutrality_free_projection,
)
from brazil_rv.execution.simulator import MarketReplay, simulate


def test_cross_fold_conditional_means_exclude_evaluation_day() -> None:
    events = {
        "day": np.array([0, 1, 2]),
        "state_cell_id": np.array([7, 7, 7]),
    }
    edges = {
        "30m": np.array([2.0, 6.0, 100.0]),
        "60m": np.array([3.0, 7.0, 100.0]),
        "120m": np.array([4.0, 8.0, 100.0]),
        "to_close": np.array([5.0, 9.0, 100.0]),
    }
    mappings, table = cross_fold_conditional_means(
        events=events, edges=edges, estimation_days=np.array([0, 1])
    )
    assert mappings["30m"][7] == 4.0
    assert mappings["to_close"][7] == 7.0
    assert table["event_count"].to_list() == [2, 2, 2, 2]


def test_rule_schedule_ignores_events_until_exact_horizon_expiry() -> None:
    events = {
        "day": np.array([0, 0, 0]),
        "minute": np.array([5, 10, 35]),
        "name": np.array([0, 0, 0]),
        "direction": np.array([1, -1, -1]),
    }
    expected = np.array(
        [[9.0, 8.0, 7.0, 6.0], [9.0, 8.0, 7.0, 6.0], [9.0, 8.0, 7.0, 6.0]]
    )
    schedule, usage = build_rule_schedule(
        events=events,
        expected_net_bps=expected,
        selected_days=np.array([0]),
        threshold_bps=7.0,
        name_count=1,
    )
    assert schedule[0, 5, 0] == 1
    assert schedule[0, 34, 0] == 1
    assert schedule[0, 35, 0] == -1
    assert usage.height == 2
    assert usage["action_minute"].to_list() == [5, 35]


def test_neutrality_free_projection_respects_caps_and_gross_without_neutralizing() -> None:
    raw = torch.tensor([[0.5, 0.5, -0.1]])
    caps = torch.tensor([[0.4, 0.4, 0.4]])
    projected = neutrality_free_projection(
        raw, torch.ones_like(raw, dtype=torch.bool), caps, 0.6
    )
    assert torch.all(projected.abs() <= caps)
    assert torch.allclose(projected.abs().sum(dim=-1), torch.tensor([0.6]))
    assert projected.sum() > 0


def test_simulator_accepts_registered_neutrality_free_bounded_book() -> None:
    dtype = torch.float64
    market = MarketReplay(
        open_price=torch.ones((1, 2, 2), dtype=dtype) * 10,
        open_observed=torch.ones((1, 2, 2), dtype=torch.bool),
        active=torch.ones((1, 2), dtype=torch.bool),
        full_spread=torch.zeros((1, 2), dtype=dtype),
        adv20_brl=torch.ones((1, 2), dtype=dtype) * 10_000_000,
        minute_notional20_brl=torch.ones((1, 2, 2), dtype=dtype) * 10_000_000,
        daily_cdi_rate=torch.zeros(1, dtype=dtype),
    )
    ranks = torch.ones((1, 2, 2, 4), dtype=dtype)
    valid = torch.ones_like(ranks, dtype=torch.bool)
    refresh = torch.tensor([[True, False]])
    sigma = torch.ones((1, 2), dtype=dtype) * 0.01
    schedule = torch.tensor([[[1, 0]]], dtype=torch.int8)
    result = simulate(
        market,
        ranks,
        valid,
        refresh,
        sigma,
        RulePathPolicy(schedule, neutral=False),
        ExecutionConfig(
            gross_target=2.0,
            name_cap_fraction_of_gross=0.05,
            horizon_blend=(0.25, 0.25, 0.25, 0.25),
            taper_minutes=1,
        ),
    )
    assert torch.isfinite(result.net_pnl_brl).all()

