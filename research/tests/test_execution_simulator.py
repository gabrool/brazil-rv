from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest
import torch
from torch import nn

from brazil_rv.execution.config import ExecutionConfig
from brazil_rv.execution.policy import BandPolicy
from brazil_rv.execution.report import DailyExecutionResult
from brazil_rv.execution.simulator import (
    MarketReplay,
    cdi_interest_base,
    close_taper,
    simulate,
)


def _config(**changes: object) -> ExecutionConfig:
    values: dict[str, object] = {
        "nav_brl": 1_000.0,
        "gross_target": 1.0,
        "participation_rate": 1.0,
        "name_cap_fraction_of_gross": 0.5,
        "adv_cap_fraction": 1.0,
        "fee_bps": 2.0,
        "max_spread_bps": 500.0,
        "min_adv_brl": 1.0,
        "taper_minutes": 1,
        "horizon_blend": (1.0, 0.0, 0.0),
        "band": 0.0,
    }
    values.update(changes)
    return ExecutionConfig(**values)


def _case(
    *,
    minutes: int = 3,
    names: int = 2,
    full_spread: tuple[float, ...] | None = None,
    daily_cdi: float = 0.0,
    minute_notional: float = 10_000.0,
) -> tuple[MarketReplay, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = torch.float64
    price = torch.full((1, minutes, names), 10.0, dtype=dtype)
    observed = torch.ones_like(price, dtype=torch.bool)
    market = MarketReplay(
        open_price=price,
        open_observed=observed,
        active=torch.ones((1, names), dtype=torch.bool),
        full_spread=torch.tensor(
            [full_spread or tuple(0.0 for _ in range(names))], dtype=dtype
        ),
        adv20_brl=torch.full((1, names), 1_000_000.0, dtype=dtype),
        minute_notional20_brl=torch.full(
            (1, minutes, names), minute_notional, dtype=dtype
        ),
        daily_cdi_rate=torch.tensor([daily_cdi], dtype=dtype),
    )
    base = torch.linspace(-1.0, 1.0, names, dtype=dtype)
    ranks = base[None, None, :, None].expand(1, minutes, names, 3).clone()
    valid = torch.ones_like(ranks, dtype=torch.bool)
    refresh = torch.zeros((1, minutes), dtype=torch.bool)
    refresh[:, 0] = True
    sigma = torch.zeros((1, names), dtype=dtype)
    return market, ranks, valid, refresh, sigma


def test_costs_next_open_and_accounting_match_golden_example() -> None:
    market, ranks, valid, refresh, sigma = _case(full_spread=(0.01, 0.02))
    config = _config()

    result = simulate(
        market,
        ranks,
        valid,
        refresh,
        sigma,
        BandPolicy(config),
        config,
        return_path=True,
    )

    assert result.positions_brl is not None and result.fills_brl is not None
    torch.testing.assert_close(
        result.fills_brl[:, 0], torch.zeros((1, 2), dtype=torch.float64)
    )
    torch.testing.assert_close(
        result.fills_brl[0, 1], torch.tensor([-500.0, 500.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        result.fills_brl[0, 2], torch.tensor([500.0, -500.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        result.spread_cost_brl, torch.tensor([15.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        result.fees_brl, torch.tensor([0.4], dtype=torch.float64)
    )
    torch.testing.assert_close(
        result.net_pnl_brl, torch.tensor([-15.4], dtype=torch.float64)
    )
    torch.testing.assert_close(
        result.turnover_brl, torch.tensor([2_000.0], dtype=torch.float64)
    )
    assert result.forced_fill_count.item() == 0
    assert torch.equal(result.positions_brl[:, -1], torch.zeros((1, 2)))


def test_price_pnl_identity_is_exact_to_currency_tolerance() -> None:
    market, ranks, valid, refresh, sigma = _case()
    market.open_price[0, 2] = torch.tensor([9.0, 11.0])
    config = _config(fee_bps=0.0)

    result = simulate(market, ranks, valid, refresh, sigma, BandPolicy(config), config)

    torch.testing.assert_close(
        result.gross_pnl_brl,
        torch.tensor([100.0], dtype=torch.float64),
        rtol=0,
        atol=1e-8,
    )
    torch.testing.assert_close(
        result.net_pnl_brl,
        result.gross_pnl_brl
        - result.spread_cost_brl
        - result.fees_brl
        + result.cdi_earned_brl,
        rtol=0,
        atol=1e-8,
    )
    assert torch.equal(
        result.net_pnl_brl,
        result.gross_pnl_brl
        - result.spread_cost_brl
        - result.fees_brl
        + result.cdi_earned_brl,
    )
    DailyExecutionResult(
        trade_date=date(2024, 1, 2),
        net_pnl_brl=result.net_pnl_brl.item(),
        gross_pnl_brl=result.gross_pnl_brl.item(),
        spread_cost_brl=result.spread_cost_brl.item(),
        fees_brl=result.fees_brl.item(),
        cdi_earned_brl=result.cdi_earned_brl.item(),
        turnover_brl=result.turnover_brl.item(),
        max_intraday_gross_brl=result.max_intraday_gross_brl.item(),
        forced_fill_count=result.forced_fill_count.item(),
    )


def test_participation_carry_taper_and_forced_close_are_distinct() -> None:
    market, ranks, valid, refresh, sigma = _case(minutes=5, minute_notional=100.0)
    config = _config(participation_rate=1.0, taper_minutes=2, fee_bps=0.0)

    result = simulate(
        market,
        ranks,
        valid,
        refresh,
        sigma,
        BandPolicy(config),
        config,
        return_path=True,
    )

    assert result.fills_brl is not None and result.carried_demand_brl is not None
    assert torch.all(result.fills_brl[:, 1:-1].abs() <= 100.0)
    torch.testing.assert_close(
        result.carried_demand_brl[0, 1],
        torch.tensor([-400.0, 400.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.positions_brl[0, 3],
        torch.tensor([-250.0, 250.0], dtype=torch.float64),
    )
    assert result.forced_fill_count.item() == 2
    assert torch.equal(result.positions_brl[:, -1], torch.zeros((1, 2)))
    assert close_taper(3, 5, 2) == 0.5
    assert close_taper(4, 5, 2) == 0.0


def test_terminal_force_ignores_ordinary_liquidity_and_spread_gate() -> None:
    market, ranks, valid, refresh, sigma = _case(
        minutes=4, full_spread=(0.01, 0.01), minute_notional=100.0
    )
    minute_spread = market.full_spread[:, None, :].expand(1, 4, 2).clone()
    minute_spread[:, -1] = 0.10
    market.minute_notional20_brl[:, -1] = torch.nan
    market = replace(market, full_spread=minute_spread)
    config = _config(
        participation_rate=1.0,
        taper_minutes=1,
        fee_bps=0.0,
        force_spread_multiplier=2.0,
    )

    result = simulate(market, ranks, valid, refresh, sigma, BandPolicy(config), config)

    assert result.forced_fill_count.item() == 2
    torch.testing.assert_close(
        result.spread_cost_brl, torch.tensor([42.0], dtype=torch.float64)
    )


def test_all_cash_cdi_and_margin_line_are_explicit() -> None:
    market, ranks, valid, refresh, sigma = _case(daily_cdi=0.01)
    valid.zero_()
    refresh.zero_()
    config = _config(fee_bps=0.0)

    result = simulate(market, ranks, valid, refresh, sigma, BandPolicy(config), config)

    torch.testing.assert_close(
        result.final_nav_brl,
        torch.tensor([1_010.0], dtype=torch.float64),
        rtol=0,
        atol=1e-8,
    )
    base = cdi_interest_base(
        torch.tensor([1_000.0]), torch.tensor([[-1_000.0, 1_000.0]]), 0.5
    )
    assert torch.equal(base, torch.zeros(1))


def test_missing_open_is_not_stale_filled() -> None:
    market, ranks, valid, refresh, sigma = _case()
    market.open_observed[0, 2, 0] = False
    config = _config(fee_bps=0.0)

    try:
        simulate(market, ranks, valid, refresh, sigma, BandPolicy(config), config)
    except ValueError as error:
        assert "missing open" in str(error)
    else:
        raise AssertionError("A held position crossed an unobserved open")


def test_observed_open_and_valid_rank_values_must_be_finite() -> None:
    market, ranks, valid, refresh, sigma = _case()
    config = _config(fee_bps=0.0)
    market.open_price[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="Observed opens"):
        simulate(market, ranks, valid, refresh, sigma, BandPolicy(config), config)

    market.open_price[0, 0, 0] = 10.0
    ranks[0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="prediction ranks"):
        simulate(market, ranks, valid, refresh, sigma, BandPolicy(config), config)


def test_excluded_names_are_inert_and_do_not_affect_other_targets() -> None:
    market, ranks, valid, refresh, sigma = _case(minutes=3, names=4)
    valid[..., 0, :] = False
    config = _config(
        name_cap_fraction_of_gross=1.0,
        fee_bps=0.0,
    )
    left = simulate(
        market,
        ranks,
        valid,
        refresh,
        sigma,
        BandPolicy(config),
        config,
        return_path=True,
    )
    ranks[:, :, 0] = torch.nan
    right = simulate(
        market,
        ranks,
        valid,
        refresh,
        sigma,
        BandPolicy(config),
        config,
        return_path=True,
    )

    assert left.fills_brl is not None and right.fills_brl is not None
    assert torch.equal(left.fills_brl[..., 0], torch.zeros_like(left.fills_brl[..., 0]))
    torch.testing.assert_close(left.fills_brl[..., 1:], right.fills_brl[..., 1:])


class ParameterPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.raw = nn.Parameter(torch.tensor([-0.4, -0.3, 0.2, 0.1]))

    def step(
        self,
        ranks: torch.Tensor,
        refresh: torch.Tensor,
        current_weights: torch.Tensor,
        sigma: torch.Tensor,
        previous_target: torch.Tensor,
        initialized: torch.Tensor,
        tradeable: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del ranks, current_weights, sigma
        candidate = self.raw.to(previous_target).expand_as(previous_target)
        build = refresh & ~initialized
        output = torch.where(build[:, None], candidate, previous_target)
        return output * tradeable, initialized | build


def test_policy_gradient_flows_through_replay_without_mask_leakage() -> None:
    market, ranks, valid, refresh, sigma = _case(minutes=4, names=4)
    market.open_price[0, 2] = torch.tensor([9.5, 10.5, 11.0, 9.0])
    market.open_price[0, 3] = torch.tensor([9.0, 11.0, 12.0, 8.5])
    market.active[0, 0] = False
    config = _config(
        name_cap_fraction_of_gross=1.0,
        fee_bps=0.0,
        taper_minutes=1,
    )
    policy = ParameterPolicy().double()

    result = simulate(market, ranks, valid, refresh, sigma, policy, config)
    result.net_pnl_brl.sum().backward()

    assert policy.raw.grad is not None
    assert torch.isfinite(policy.raw.grad).all()
    assert policy.raw.grad[0].item() == 0.0
    assert torch.count_nonzero(policy.raw.grad).item() > 0
