from __future__ import annotations

from dataclasses import replace

import torch

from brazil_rv.execution.config import ExecutionConfig
from brazil_rv.execution.features import (
    POLICY_STATE_SCHEMA_SHA256,
    build_policy_state,
    liquidity_tercile_one_hot,
    policy_state_feature_names,
    update_volume_weighted_cost_basis,
)
from brazil_rv.execution.simulator import MarketReplay, simulate


def _state() -> tuple[torch.Tensor, object]:
    dtype = torch.float64
    ranks = torch.tensor([[[-1.0, -0.5, 0.0], [0.0, 0.5, 1.0]]], dtype=dtype)
    shape = ranks.shape[:2]
    state = build_policy_state(
        ranks=ranks,
        rank_change=torch.full_like(ranks, 0.2),
        signal_age_minutes=torch.tensor([[5.0, 10.0]], dtype=dtype),
        current_weights=torch.tensor([[0.1, -0.1]], dtype=dtype),
        current_price=torch.tensor([[11.0, 9.0]], dtype=dtype),
        cost_basis_price=torch.tensor([[10.0, 10.0]], dtype=dtype),
        minutes_in_position=torch.tensor([[4.0, 7.0]], dtype=dtype),
        lagged_full_spread=torch.full(shape, 0.001, dtype=dtype),
        daily_sigma=torch.full(shape, 0.02, dtype=dtype),
        adv20_brl=torch.tensor([[1e6, 2e6]], dtype=dtype),
        participation_capacity_brl=torch.full(shape, 10_000.0, dtype=dtype),
        nav_brl=torch.tensor([1e6], dtype=dtype),
        initial_nav_brl=torch.tensor([1e6], dtype=dtype),
        tradeable_mask=torch.ones(shape, dtype=torch.bool),
        cap_weights=torch.full(shape, 0.2, dtype=dtype),
        gross_target=2.0,
        margin_fraction_of_gross=0.5,
        session_minute=10,
        session_minutes=405,
    )
    return ranks, state


def test_policy_state_schema_order_width_and_hash_are_frozen() -> None:
    ranks, state = _state()
    assert (
        POLICY_STATE_SCHEMA_SHA256
        == "96bc69bdb52656b556fb554f552e41ccc82e80ff56ee33f74ad6137d51354f7e"
    )
    assert state.schema_sha256 == POLICY_STATE_SCHEMA_SHA256
    assert state.features().shape == (1, 2, 21)
    assert policy_state_feature_names(ranks.shape[-1], state.horizon_names)[:6] == (
        "rank_30m",
        "rank_60m",
        "rank_120m",
        "rank_change_30m",
        "rank_change_60m",
        "rank_change_120m",
    )
    assert torch.isfinite(state.features()).all()


def test_volume_weighted_basis_add_reduce_cross_and_close() -> None:
    dtype = torch.float64
    old = torch.tensor([200.0, 200.0, 200.0, 200.0], dtype=dtype)
    fill = torch.tensor([100.0, -100.0, -300.0, -200.0], dtype=dtype)
    price = torch.full((4,), 20.0, dtype=dtype)
    basis = torch.full((4,), 10.0, dtype=dtype)

    shares, updated = update_volume_weighted_cost_basis(old, fill, price, basis, 1e-8)

    torch.testing.assert_close(shares[:3], torch.tensor([15.0, 5.0, -5.0], dtype=dtype))
    torch.testing.assert_close(
        updated[:3], torch.tensor([40 / 3, 10.0, 20.0], dtype=dtype)
    )
    assert torch.isnan(updated[3])


def test_liquidity_terciles_are_deterministic_and_ignore_ineligible_names() -> None:
    adv = torch.tensor(
        [[40.0, 10.0, 30.0, 20.0, torch.nan], [5.0, 4.0, 3.0, 2.0, 1.0]],
        dtype=torch.float64,
    )
    eligible = torch.tensor(
        [[True, True, True, True, True], [False, False, False, False, False]]
    )
    actual = liquidity_tercile_one_hot(adv, eligible)
    expected = torch.tensor(
        [
            [[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ],
        dtype=torch.float64,
    )
    assert torch.equal(actual, expected)


def test_masked_nonfinite_market_inputs_cannot_reach_policy_features() -> None:
    ranks, _ = _state()
    mask = torch.tensor([[True, False]])
    state = build_policy_state(
        ranks=ranks,
        rank_change=torch.zeros_like(ranks),
        signal_age_minutes=torch.zeros((1, 2), dtype=torch.float64),
        current_weights=torch.zeros((1, 2), dtype=torch.float64),
        current_price=torch.tensor([[10.0, torch.nan]], dtype=torch.float64),
        cost_basis_price=torch.full((1, 2), torch.nan, dtype=torch.float64),
        minutes_in_position=torch.zeros((1, 2), dtype=torch.float64),
        lagged_full_spread=torch.tensor([[0.001, torch.nan]], dtype=torch.float64),
        daily_sigma=torch.tensor([[0.02, torch.nan]], dtype=torch.float64),
        adv20_brl=torch.tensor([[1e6, torch.nan]], dtype=torch.float64),
        participation_capacity_brl=torch.zeros((1, 2), dtype=torch.float64),
        nav_brl=torch.tensor([1e6], dtype=torch.float64),
        initial_nav_brl=torch.tensor([1e6], dtype=torch.float64),
        tradeable_mask=mask,
        cap_weights=torch.tensor([[0.2, torch.nan]], dtype=torch.float64),
        gross_target=2.0,
        margin_fraction_of_gross=0.5,
        session_minute=0,
        session_minutes=405,
    )
    assert torch.isfinite(state.features()).all()
    assert torch.equal(state.per_name[:, 1], torch.zeros_like(state.per_name[:, 1]))


class _RecordingCashPolicy:
    projection_mode = "bounded"
    requires_policy_state = True

    def __init__(self) -> None:
        self.states: list[torch.Tensor] = []

    def step(
        self,
        ranks: torch.Tensor,
        refresh: torch.Tensor,
        current_weights: torch.Tensor,
        sigma: torch.Tensor,
        previous_target: torch.Tensor,
        initialized: torch.Tensor,
        tradeable_mask: torch.Tensor,
        cap_weights: torch.Tensor,
        full_spread: torch.Tensor,
        policy_state: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del ranks, refresh, current_weights, sigma, cap_weights, full_spread
        self.states.append(policy_state.features().detach().clone())
        return torch.zeros_like(previous_target), initialized | tradeable_mask.any(
            dim=-1
        )


def _replay() -> tuple[
    MarketReplay,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    ExecutionConfig,
]:
    dtype = torch.float64
    days, minutes, names = 1, 5, 4
    price = torch.full((days, minutes, names), 10.0, dtype=dtype)
    market = MarketReplay(
        open_price=price,
        open_observed=torch.ones_like(price, dtype=torch.bool),
        active=torch.ones((days, names), dtype=torch.bool),
        full_spread=torch.full((days, minutes, names), 0.001, dtype=dtype),
        adv20_brl=torch.arange(1, names + 1, dtype=dtype)[None] * 1e6,
        minute_notional20_brl=torch.full_like(price, 100_000.0),
        daily_cdi_rate=torch.tensor([0.001], dtype=dtype),
    )
    ranks = (
        torch.linspace(-1, 1, names, dtype=dtype)[None, None, :, None]
        .expand(days, minutes, names, 3)
        .clone()
    )
    valid = torch.ones_like(ranks, dtype=torch.bool)
    refresh = torch.zeros((days, minutes), dtype=torch.bool)
    refresh[:, 0] = refresh[:, 2] = True
    sigma = torch.full((days, names), 0.02, dtype=dtype)
    config = ExecutionConfig(
        nav_brl=1e6,
        gross_target=1.0,
        participation_rate=0.1,
        name_cap_fraction_of_gross=0.5,
        adv_cap_fraction=0.5,
        fee_bps=0.0,
        max_spread_bps=100.0,
        min_adv_brl=1.0,
        taper_minutes=1,
    )
    return market, ranks, valid, refresh, sigma, config


def test_simulator_policy_state_is_unchanged_by_future_mutation() -> None:
    market, ranks, valid, refresh, sigma, config = _replay()
    left_policy = _RecordingCashPolicy()
    simulate(market, ranks, valid, refresh, sigma, left_policy, config)

    changed_market = replace(
        market,
        open_price=market.open_price.clone(),
        full_spread=market.full_spread.clone(),
        minute_notional20_brl=market.minute_notional20_brl.clone(),
    )
    changed_ranks = ranks.clone()
    changed_market.open_price[:, 2:] *= 1.7
    changed_market.full_spread[:, 2:] *= 3
    changed_market.minute_notional20_brl[:, 2:] *= 9
    changed_ranks[:, 2:] *= -1
    right_policy = _RecordingCashPolicy()
    simulate(changed_market, changed_ranks, valid, refresh, sigma, right_policy, config)

    torch.testing.assert_close(
        left_policy.states[1], right_policy.states[1], rtol=0, atol=0
    )
