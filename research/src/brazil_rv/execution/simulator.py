from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import ExecutionConfig
from .constraints import project_weights


@dataclass(frozen=True)
class MarketReplay:
    """Causal minute data for independent intraday session replays.

    Prices and minute notionals are shaped ``[day, minute, name]``. Spreads are
    full-spread fractions and may be either ``[day, name]`` or minute-resolved.
    ADV and activity are date-level because both are fixed before the session.
    """

    open_price: torch.Tensor
    open_observed: torch.Tensor
    active: torch.Tensor
    full_spread: torch.Tensor
    adv20_brl: torch.Tensor
    minute_notional20_brl: torch.Tensor
    daily_cdi_rate: torch.Tensor


@dataclass(frozen=True)
class SimulationResult:
    final_nav_brl: torch.Tensor
    net_pnl_brl: torch.Tensor
    gross_pnl_brl: torch.Tensor
    spread_cost_brl: torch.Tensor
    fees_brl: torch.Tensor
    cdi_earned_brl: torch.Tensor
    turnover_brl: torch.Tensor
    max_intraday_gross_brl: torch.Tensor
    forced_fill_count: torch.Tensor
    positions_brl: torch.Tensor | None = None
    fills_brl: torch.Tensor | None = None
    carried_demand_brl: torch.Tensor | None = None


def tradeable_universe(market: MarketReplay, config: ExecutionConfig) -> torch.Tensor:
    """Return the date/name mask without deleting excluded observations."""
    spread = _date_spread(market)
    adv = market.adv20_brl
    return (
        market.active.bool()
        & torch.isfinite(spread)
        & (spread >= 0)
        & (spread * 10_000 <= config.max_spread_bps)
        & torch.isfinite(adv)
        & (adv >= config.min_adv_brl)
    )


def close_taper(fill_minute: int, minute_count: int, taper_minutes: int) -> float:
    """Scale a target for its fill minute; the final open always targets zero."""
    if not 0 <= fill_minute < minute_count:
        raise ValueError("Fill minute lies outside the session")
    if taper_minutes < 0:
        raise ValueError("Taper minutes cannot be negative")
    remaining = minute_count - 1 - fill_minute
    if remaining == 0:
        return 0.0
    if taper_minutes == 0:
        return 1.0
    return min(1.0, remaining / taper_minutes)


def cdi_interest_base(
    nav_brl: torch.Tensor,
    positions_brl: torch.Tensor,
    margin_fraction_of_gross: float,
) -> torch.Tensor:
    """Undeployed cash under the explicit intraday margin assumption."""
    margin = margin_fraction_of_gross * positions_brl.abs().sum(dim=-1)
    return (nav_brl - margin).clamp_min(0)


def _date_spread(market: MarketReplay) -> torch.Tensor:
    days, minutes, names = market.open_price.shape
    spread = market.full_spread
    if spread.shape == (days, names):
        return spread
    if spread.shape == (days, minutes, names):
        return spread[:, 0]
    raise ValueError("Full spread must be [day,name] or [day,minute,name]")


def _minute_spread(market: MarketReplay) -> torch.Tensor:
    days, minutes, names = market.open_price.shape
    spread = market.full_spread
    if spread.shape == (days, names):
        return spread[:, None, :].expand(days, minutes, names)
    if spread.shape == (days, minutes, names):
        return spread
    raise ValueError("Full spread must be [day,name] or [day,minute,name]")


def _validate_inputs(
    market: MarketReplay,
    ranks: torch.Tensor,
    rank_valid: torch.Tensor,
    refresh_mask: torch.Tensor,
    sigma: torch.Tensor,
) -> tuple[int, int, int]:
    price = market.open_price
    if not price.is_floating_point() or price.ndim != 3:
        raise ValueError("Open prices must be a floating [day,minute,name] tensor")
    days, minutes, names = price.shape
    if minutes < 2:
        raise ValueError("A replay requires at least two session minutes")
    expected = (days, minutes, names)
    if market.open_observed.shape != expected:
        raise ValueError("Open observation mask differs from prices")
    if market.minute_notional20_brl.shape != expected:
        raise ValueError("Minute liquidity profile differs from prices")
    if market.active.shape != (days, names) or market.adv20_brl.shape != (
        days,
        names,
    ):
        raise ValueError("Date/name market inputs differ from prices")
    if market.daily_cdi_rate.shape != (days,):
        raise ValueError("CDI rates must contain one value per day")
    if ranks.ndim != 4 or ranks.shape[:3] != expected:
        raise ValueError("Ranks must be [day,minute,name,horizon]")
    if rank_valid.shape != ranks.shape:
        raise ValueError("Rank-valid mask differs from ranks")
    if refresh_mask.shape != (days, minutes):
        raise ValueError("Refresh mask must be [day,minute]")
    if sigma.shape not in ((days, names), expected):
        raise ValueError("Sigma must be [day,name] or [day,minute,name]")
    if price.device != ranks.device or price.dtype != ranks.dtype:
        raise ValueError("Market prices and ranks must share device and dtype")
    observed = market.open_observed.to(device=price.device, dtype=torch.bool)
    if (observed & (~torch.isfinite(price) | (price <= 0))).any():
        raise ValueError("Observed opens must be finite and strictly positive")
    valid_ranks = rank_valid.to(device=ranks.device, dtype=torch.bool)
    if (valid_ranks & ~torch.isfinite(ranks)).any():
        raise ValueError("Valid prediction ranks must be finite")
    if (
        not torch.isfinite(market.daily_cdi_rate).all()
        or not (market.daily_cdi_rate > -1).all()
    ):
        raise ValueError("Daily CDI rates must be finite and greater than -1")
    return days, minutes, names


def simulate(
    market: MarketReplay,
    ranks: torch.Tensor,
    rank_valid: torch.Tensor,
    refresh_mask: torch.Tensor,
    sigma: torch.Tensor,
    policy: object,
    config: ExecutionConfig = ExecutionConfig(),
    *,
    return_path: bool = False,
) -> SimulationResult:
    """Replay an intraday policy with conservative next-open execution.

    An action formed at minute ``t`` first trades at minute ``t + 1``. Holdings
    are marked open-to-open, then the pending target trades at that next open.
    This is deliberately one minute slower than the alpha label's entry-open
    convention and never substitutes a stale price for a missing open.
    """
    days, minutes, names = _validate_inputs(
        market, ranks, rank_valid, refresh_mask, sigma
    )
    dtype = market.open_price.dtype
    device = market.open_price.device
    spread = _minute_spread(market).to(device=device, dtype=dtype)
    adv = market.adv20_brl.to(device=device, dtype=dtype)
    liquidity = market.minute_notional20_brl.to(device=device, dtype=dtype)
    observed = market.open_observed.to(device=device, dtype=torch.bool)
    base_tradeable = tradeable_universe(market, config).to(device=device)
    blend = torch.as_tensor(config.horizon_blend, dtype=dtype, device=device)
    if blend.numel() != ranks.shape[-1]:
        raise ValueError("Horizon blend width differs from prediction heads")
    required_heads = blend != 0

    nav = torch.full((days,), config.nav_brl, dtype=dtype, device=device)
    initial_nav = nav.clone()
    position = torch.zeros((days, names), dtype=dtype, device=device)
    prior_target = torch.zeros_like(position)
    initialized = torch.zeros(days, dtype=torch.bool, device=device)

    gross_pnl = torch.zeros_like(nav)
    spread_cost = torch.zeros_like(nav)
    fees = torch.zeros_like(nav)
    cdi = torch.zeros_like(nav)
    turnover = torch.zeros_like(nav)
    maximum_gross = torch.zeros_like(nav)
    forced_count = torch.zeros(days, dtype=torch.int64, device=device)

    positions = [position]
    fills = [torch.zeros_like(position)]
    carried = [torch.zeros_like(position)]
    interval_cdi = torch.expm1(torch.log1p(market.daily_cdi_rate) / (minutes - 1))
    interval_cdi = interval_cdi.to(device=device, dtype=dtype)

    for action_minute in range(minutes - 1):
        minute_rank_valid = rank_valid[:, action_minute].to(
            device=device, dtype=torch.bool
        )
        name_rank_valid = minute_rank_valid[..., required_heads].all(dim=-1)
        policy_tradeable = base_tradeable & name_rank_valid
        policy_ranks = torch.where(
            minute_rank_valid,
            ranks[:, action_minute],
            torch.zeros_like(ranks[:, action_minute]),
        )
        current_weights = position / nav.unsqueeze(-1)
        current_sigma = (sigma if sigma.ndim == 2 else sigma[:, action_minute]).to(
            device=device, dtype=dtype
        )
        if (
            policy_tradeable & (~torch.isfinite(current_sigma) | (current_sigma < 0))
        ).any():
            raise ValueError("Tradeable policy sigma must be finite and non-negative")
        raw_target, initialized = policy.step(
            policy_ranks,
            refresh_mask[:, action_minute],
            current_weights,
            current_sigma,
            prior_target,
            initialized,
            policy_tradeable,
        )
        if (
            raw_target.shape != (days, names)
            or raw_target.device != device
            or raw_target.dtype != dtype
        ):
            raise ValueError(
                "Policy targets must match market day/name device and dtype"
            )
        if initialized.shape != (days,):
            raise ValueError("Policy initialized state must contain one value per day")

        cap_notional = torch.minimum(
            config.name_cap_fraction_of_gross * config.gross_target * nav.unsqueeze(-1),
            config.adv_cap_fraction * adv,
        )
        cap_weights = cap_notional / nav.unsqueeze(-1)
        active_rows = torch.nonzero(initialized, as_tuple=False).flatten()
        base_target = torch.zeros_like(raw_target)
        if active_rows.numel():
            projected = project_weights(
                raw_target[active_rows],
                policy_tradeable[active_rows],
                cap_weights[active_rows],
                config.gross_target,
            )
            base_target = base_target.index_copy(0, active_rows, projected)
        prior_target = base_target

        fill_minute = action_minute + 1
        taper = close_taper(fill_minute, minutes, config.taper_minutes)
        desired_notional = base_target * (nav.unsqueeze(-1) * taper)

        interest_base = cdi_interest_base(
            nav, position, config.margin_fraction_of_gross
        )
        interval_interest = interest_base * interval_cdi

        held = position.abs() > config.position_tolerance_brl
        adjacent = observed[:, action_minute] & observed[:, fill_minute]
        if (held & ~adjacent).any():
            raise ValueError(
                "A held position crosses a missing open; stale marking is forbidden"
            )
        current_open = torch.where(
            observed[:, action_minute],
            market.open_price[:, action_minute],
            torch.ones_like(position),
        )
        next_open = torch.where(
            observed[:, fill_minute],
            market.open_price[:, fill_minute],
            torch.ones_like(position),
        )
        price_return = torch.where(
            adjacent, next_open / current_open - 1, torch.zeros_like(position)
        )
        interval_pnl = (position * price_return).sum(dim=-1)
        position = position * (1 + price_return)
        nav = nav + interval_pnl + interval_interest
        gross_pnl = gross_pnl + interval_pnl
        cdi = cdi + interval_interest
        maximum_gross = torch.maximum(maximum_gross, position.abs().sum(dim=-1))

        minute_spread = spread[:, fill_minute]
        minute_liquidity = liquidity[:, fill_minute]
        fillable = (
            base_tradeable
            & observed[:, fill_minute]
            & torch.isfinite(minute_spread)
            & (minute_spread >= 0)
            & (minute_spread * 10_000 <= config.max_spread_bps)
            & torch.isfinite(minute_liquidity)
            & (minute_liquidity >= 0)
        )
        capacity = torch.where(
            fillable,
            config.participation_rate * minute_liquidity,
            torch.zeros_like(position),
        ).clamp_min(0)
        demand = desired_notional - position
        ordinary_fill = torch.minimum(torch.maximum(demand, -capacity), capacity)
        ordinary_fill = torch.where(fillable, ordinary_fill, 0)
        position = position + ordinary_fill
        residual = desired_notional - position

        forced_fill = torch.zeros_like(position)
        if fill_minute == minutes - 1:
            terminal = position.abs() > config.position_tolerance_brl
            terminal_pricable = (
                observed[:, fill_minute]
                & torch.isfinite(minute_spread)
                & (minute_spread >= 0)
            )
            if (terminal & ~terminal_pricable).any():
                raise ValueError(
                    "Terminal position lacks an observed priced final fill"
                )
            forced_count = terminal.sum(dim=-1)
            forced_fill = torch.where(terminal, -position, 0)
            position = position + forced_fill
            residual = torch.zeros_like(position)

        safe_spread = torch.where(
            torch.isfinite(minute_spread), minute_spread, torch.zeros_like(position)
        )
        regular_spread = (ordinary_fill.abs() * safe_spread * 0.5).sum(dim=-1)
        forced_spread = (
            forced_fill.abs() * safe_spread * 0.5 * config.force_spread_multiplier
        ).sum(dim=-1)
        total_fill = ordinary_fill + forced_fill
        fill_fees = total_fill.abs().sum(dim=-1) * config.fee_bps / 10_000
        step_spread = regular_spread + forced_spread
        nav = nav - step_spread - fill_fees
        spread_cost = spread_cost + step_spread
        fees = fees + fill_fees
        turnover = turnover + total_fill.abs().sum(dim=-1)
        maximum_gross = torch.maximum(maximum_gross, position.abs().sum(dim=-1))

        if return_path:
            positions.append(position)
            fills.append(total_fill)
            carried.append(residual)

    accounting = gross_pnl - spread_cost - fees + cdi
    scanned_net = nav - initial_nav
    tolerance = 64 * torch.finfo(dtype).eps * initial_nav
    if not torch.all((scanned_net - accounting).abs() <= tolerance):
        raise ArithmeticError("Execution accounting identity failed")
    if not torch.all(position.abs() <= config.position_tolerance_brl):
        raise ArithmeticError("Execution replay did not finish flat")
    # Canonicalize the reported totals from their separately accumulated
    # components. The scanned NAV is still checked above, while this exact
    # arithmetic order makes every valid replay directly reportable under the
    # 1e-8 DailyExecutionResult identity.
    net = accounting
    nav = initial_nav + net

    return SimulationResult(
        final_nav_brl=nav,
        net_pnl_brl=net,
        gross_pnl_brl=gross_pnl,
        spread_cost_brl=spread_cost,
        fees_brl=fees,
        cdi_earned_brl=cdi,
        turnover_brl=turnover,
        max_intraday_gross_brl=maximum_gross,
        forced_fill_count=forced_count,
        positions_brl=torch.stack(positions, dim=1) if return_path else None,
        fills_brl=torch.stack(fills, dim=1) if return_path else None,
        carried_demand_brl=torch.stack(carried, dim=1) if return_path else None,
    )
