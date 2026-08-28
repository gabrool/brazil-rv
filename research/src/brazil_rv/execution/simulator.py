from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import ExecutionConfig
from .constraints import project_weights
from .features import build_policy_state, update_volume_weighted_cost_basis


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
    mean_deployed_gross_brl: torch.Tensor
    turnover_by_name_brl: torch.Tensor
    gross_pnl_by_name_brl: torch.Tensor
    spread_cost_by_name_brl: torch.Tensor
    fees_by_name_brl: torch.Tensor
    round_trip_count_by_name: torch.Tensor
    round_trip_gross_pnl_by_name_brl: torch.Tensor
    round_trip_cost_by_name_brl: torch.Tensor
    selection_extended_count: torch.Tensor
    positions_brl: torch.Tensor | None = None
    fills_brl: torch.Tensor | None = None
    carried_demand_brl: torch.Tensor | None = None
    target_weights: torch.Tensor | None = None


def tradeable_universe(market: MarketReplay, config: ExecutionConfig) -> torch.Tensor:
    """Return the date/name mask without deleting excluded observations."""
    spread = _date_spread(market) * config.spread_schedule_multiplier
    adv = market.adv20_brl
    tradeable = (
        market.active.bool()
        & torch.isfinite(spread)
        & (spread >= 0)
        & (spread * 10_000 <= config.max_spread_bps)
        & torch.isfinite(adv)
        & (adv >= config.min_adv_brl)
    )
    if not config.top_half_adv:
        return tradeable
    selected = torch.zeros_like(tradeable)
    for day in range(tradeable.shape[0]):
        names = torch.nonzero(tradeable[day], as_tuple=False).flatten()
        keep = (names.numel() + 1) // 2
        if keep:
            order = torch.argsort(adv[day, names], descending=True, stable=True)
            selected[day, names[order[:keep]]] = True
    return selected


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
    convention. A missing open permits neither a fill nor a synthetic mark;
    an existing holding realizes its cumulative return when the next observed
    open arrives.
    """
    days, minutes, names = _validate_inputs(
        market, ranks, rank_valid, refresh_mask, sigma
    )
    dtype = market.open_price.dtype
    device = market.open_price.device
    spread = _minute_spread(market).to(device=device, dtype=dtype)
    spread = spread * config.spread_schedule_multiplier
    adv = market.adv20_brl.to(device=device, dtype=dtype)
    liquidity = market.minute_notional20_brl.to(device=device, dtype=dtype)
    observed = market.open_observed.to(device=device, dtype=torch.bool)
    base_tradeable = tradeable_universe(market, config).to(device=device)
    blend = torch.as_tensor(config.horizon_blend, dtype=dtype, device=device)
    if blend.numel() != ranks.shape[-1]:
        raise ValueError("Horizon blend width differs from prediction heads")
    requires_policy_state = bool(getattr(policy, "requires_policy_state", False))
    required_heads = (
        torch.ones_like(blend, dtype=torch.bool)
        if requires_policy_state
        else blend != 0
    )

    nav = torch.full((days,), config.nav_brl, dtype=dtype, device=device)
    initial_nav = nav.clone()
    position = torch.zeros((days, names), dtype=dtype, device=device)
    prior_target = torch.zeros_like(position)
    initialized = torch.zeros(days, dtype=torch.bool, device=device)
    last_open = torch.where(
        observed[:, 0],
        market.open_price[:, 0],
        torch.full_like(position, torch.nan),
    )
    last_spread = torch.where(observed[:, 0], spread[:, 0], torch.nan)
    cost_basis = torch.full_like(position, torch.nan)
    opened_minute = torch.full((days, names), -1, dtype=torch.int64, device=device)
    previous_refresh_rank = torch.zeros(
        (days, names, ranks.shape[-1]), dtype=dtype, device=device
    )
    rank_change = torch.zeros_like(previous_refresh_rank)
    has_previous_refresh = torch.zeros_like(previous_refresh_rank, dtype=torch.bool)
    last_valid_refresh_minute = torch.full(
        (days, names), -1, dtype=torch.int64, device=device
    )

    gross_pnl = torch.zeros_like(nav)
    spread_cost = torch.zeros_like(nav)
    fees = torch.zeros_like(nav)
    cdi = torch.zeros_like(nav)
    turnover = torch.zeros_like(nav)
    maximum_gross = torch.zeros_like(nav)
    forced_count = torch.zeros(days, dtype=torch.int64, device=device)
    gross_by_name = torch.zeros_like(position)
    spread_by_name = torch.zeros_like(position)
    fees_by_name = torch.zeros_like(position)
    turnover_by_name = torch.zeros_like(position)
    episode_gross = torch.zeros_like(position)
    episode_cost = torch.zeros_like(position)
    round_trip_count = torch.zeros((days, names), dtype=torch.int64, device=device)
    round_trip_gross = torch.zeros_like(position)
    round_trip_cost = torch.zeros_like(position)
    deployed_gross_sum = torch.zeros_like(nav)
    extension_path = torch.zeros((days, minutes), dtype=torch.int64, device=device)

    positions = [position]
    fills = [torch.zeros_like(position)]
    carried = [torch.zeros_like(position)]
    targets = [torch.zeros_like(position)]
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
        refresh_now = refresh_mask[:, action_minute].bool()[:, None, None]
        refreshed_head = refresh_now & minute_rank_valid
        rank_change = torch.where(
            refreshed_head & has_previous_refresh,
            policy_ranks - previous_refresh_rank,
            torch.where(refreshed_head, torch.zeros_like(rank_change), rank_change),
        )
        previous_refresh_rank = torch.where(
            refreshed_head, policy_ranks, previous_refresh_rank
        )
        has_previous_refresh = has_previous_refresh | refreshed_head
        refreshed_name = refresh_now.squeeze(-1) & name_rank_valid
        last_valid_refresh_minute = torch.where(
            refreshed_name,
            torch.full_like(last_valid_refresh_minute, action_minute),
            last_valid_refresh_minute,
        )
        signal_age = torch.where(
            last_valid_refresh_minute >= 0,
            action_minute - last_valid_refresh_minute,
            torch.zeros_like(last_valid_refresh_minute),
        ).to(dtype)
        current_weights = position / nav.unsqueeze(-1)
        current_sigma = (sigma if sigma.ndim == 2 else sigma[:, action_minute]).to(
            device=device, dtype=dtype
        )
        if (
            policy_tradeable & (~torch.isfinite(current_sigma) | (current_sigma < 0))
        ).any():
            raise ValueError("Tradeable policy sigma must be finite and non-negative")
        safe_adv = torch.where(base_tradeable, adv, torch.zeros_like(adv))
        cap_weights = torch.minimum(
            torch.as_tensor(
                config.name_cap_fraction_of_gross * config.gross_target,
                dtype=dtype,
                device=device,
            ),
            config.adv_cap_fraction * safe_adv / nav.unsqueeze(-1),
        )
        policy_state = None
        if requires_policy_state:
            action_liquidity = liquidity[:, action_minute]
            action_capacity = torch.where(
                policy_tradeable
                & torch.isfinite(action_liquidity)
                & (action_liquidity >= 0),
                config.participation_rate * action_liquidity,
                torch.zeros_like(position),
            )
            position_age = torch.where(
                opened_minute >= 0,
                action_minute - opened_minute,
                torch.zeros_like(opened_minute),
            ).to(dtype)
            policy_state = build_policy_state(
                ranks=policy_ranks,
                rank_change=rank_change,
                signal_age_minutes=signal_age,
                current_weights=current_weights,
                current_price=last_open,
                cost_basis_price=cost_basis,
                minutes_in_position=position_age,
                lagged_full_spread=torch.where(
                    torch.isfinite(spread[:, action_minute]),
                    spread[:, action_minute],
                    torch.zeros_like(position),
                ),
                daily_sigma=current_sigma,
                adv20_brl=adv,
                participation_capacity_brl=action_capacity,
                nav_brl=nav,
                initial_nav_brl=initial_nav,
                tradeable_mask=policy_tradeable,
                cap_weights=cap_weights,
                gross_target=config.gross_target,
                margin_fraction_of_gross=config.margin_fraction_of_gross,
                session_minute=action_minute,
                session_minutes=minutes,
                horizon_names=getattr(policy, "horizon_names", None),
            )
            raw_target, initialized = policy.step(
                policy_ranks,
                refresh_mask[:, action_minute],
                current_weights,
                current_sigma,
                prior_target,
                initialized,
                policy_tradeable,
                cap_weights,
                spread[:, action_minute],
                policy_state,
            )
        else:
            raw_target, initialized = policy.step(
                policy_ranks,
                refresh_mask[:, action_minute],
                current_weights,
                current_sigma,
                prior_target,
                initialized,
                policy_tradeable,
                cap_weights,
                spread[:, action_minute],
            )
        extension_count = getattr(policy, "last_selection_extended_count", None)
        if extension_count is not None:
            if extension_count.shape != (days,):
                raise ValueError(
                    "Selection-extension count must have one value per day"
                )
            extension_path[:, action_minute] = torch.where(
                refresh_mask[:, action_minute].bool(), extension_count, 0
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

        projection_mode = getattr(policy, "projection_mode", "exact")
        if projection_mode == "bounded":
            scale = torch.maximum(
                raw_target.abs().sum(dim=-1),
                torch.ones(days, dtype=dtype, device=device),
            )
            tolerance = 64 * torch.finfo(dtype).eps * scale
            if (~torch.isfinite(raw_target)).any() or (
                raw_target.masked_fill(policy_tradeable, 0).abs()
                > tolerance.unsqueeze(-1)
            ).any():
                raise ValueError("Bounded policy emitted invalid or masked weights")
            if (
                (raw_target.sum(dim=-1).abs() > tolerance).any()
                or (raw_target.abs() > cap_weights + tolerance.unsqueeze(-1)).any()
                or (
                    raw_target.abs().sum(dim=-1) > config.gross_target + tolerance
                ).any()
            ):
                raise ValueError("Bounded policy target violates neutrality or caps")
            base_target = torch.where(
                initialized[:, None], raw_target, torch.zeros_like(raw_target)
            )
        elif projection_mode == "exact":
            active_rows = torch.nonzero(initialized, as_tuple=False).flatten()
            base_target = torch.zeros_like(raw_target)
            if active_rows.numel():
                active_raw = raw_target[active_rows]
                active_mask = policy_tradeable[active_rows]
                active_caps = cap_weights[active_rows]
                side_target = config.gross_target / 2
                positive_capacity = torch.where(
                    active_mask & (active_raw > 0), active_caps, 0
                ).sum(dim=-1)
                negative_capacity = torch.where(
                    active_mask & (active_raw < 0), active_caps, 0
                ).sum(dim=-1)
                feasible = (positive_capacity >= side_target) & (
                    negative_capacity >= side_target
                )
                base_target = base_target.index_copy(
                    0, active_rows, prior_target[active_rows]
                )
                feasible_rows = active_rows[feasible]
                if feasible_rows.numel():
                    projected = project_weights(
                        raw_target[feasible_rows],
                        policy_tradeable[feasible_rows],
                        cap_weights[feasible_rows],
                        config.gross_target,
                    )
                    base_target = base_target.index_copy(0, feasible_rows, projected)
        else:
            raise ValueError(f"Unknown policy projection mode: {projection_mode}")
        prior_target = base_target

        fill_minute = action_minute + 1
        taper = close_taper(fill_minute, minutes, config.taper_minutes)
        desired_notional = base_target * (nav.unsqueeze(-1) * taper)

        interest_base = cdi_interest_base(
            nav, position, config.margin_fraction_of_gross
        )
        interval_interest = interest_base * interval_cdi

        held = position.abs() > config.position_tolerance_brl
        if (held & ~torch.isfinite(last_open)).any():
            raise ValueError("A held position lacks a prior observed open")
        next_observed = observed[:, fill_minute]
        next_open = torch.where(
            next_observed,
            market.open_price[:, fill_minute],
            torch.ones_like(position),
        )
        current_open = torch.where(torch.isfinite(last_open), last_open, 1)
        price_return = torch.where(
            next_observed,
            next_open / current_open - 1,
            torch.zeros_like(position),
        )
        interval_name_pnl = position * price_return
        interval_pnl = interval_name_pnl.sum(dim=-1)
        gross_by_name = gross_by_name + interval_name_pnl
        episode_gross = episode_gross + interval_name_pnl
        position = position * (1 + price_return)
        old_position = position
        last_open = torch.where(next_observed, next_open, last_open)
        last_spread = torch.where(next_observed, spread[:, fill_minute], last_spread)
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
                torch.isfinite(last_open)
                & torch.isfinite(last_spread)
                & (last_spread >= 0)
            )
            if (terminal & ~terminal_pricable).any():
                raise ValueError("Terminal position lacks a prior observed priced fill")
            forced_count = terminal.sum(dim=-1)
            forced_fill = torch.where(terminal, -position, 0)
            position = position + forced_fill
            residual = torch.zeros_like(position)

        safe_spread = torch.where(
            torch.isfinite(minute_spread), minute_spread, torch.zeros_like(position)
        )
        regular_spread = (ordinary_fill.abs() * safe_spread * 0.5).sum(dim=-1)
        terminal_spread = torch.where(
            torch.isfinite(last_spread), last_spread, torch.zeros_like(position)
        )
        forced_spread = (
            forced_fill.abs() * terminal_spread * 0.5 * config.force_spread_multiplier
        ).sum(dim=-1)
        total_fill = ordinary_fill + forced_fill
        basis_price = torch.where(
            ordinary_fill != 0,
            next_open,
            torch.where(
                torch.isfinite(last_open), last_open, torch.ones_like(last_open)
            ),
        )
        _, cost_basis = update_volume_weighted_cost_basis(
            old_position,
            total_fill,
            basis_price,
            cost_basis,
            config.position_tolerance_brl,
        )
        new_active_for_age = position.abs() > config.position_tolerance_brl
        old_active_for_age = old_position.abs() > config.position_tolerance_brl
        opened_or_crossed = new_active_for_age & (
            ~old_active_for_age | (torch.sign(old_position) != torch.sign(position))
        )
        opened_minute = torch.where(
            opened_or_crossed,
            torch.full_like(opened_minute, fill_minute),
            torch.where(
                new_active_for_age,
                opened_minute,
                torch.full_like(opened_minute, -1),
            ),
        )
        regular_spread_by_name = ordinary_fill.abs() * safe_spread * 0.5
        forced_spread_by_name = (
            forced_fill.abs() * terminal_spread * 0.5 * config.force_spread_multiplier
        )
        step_spread_by_name = regular_spread_by_name + forced_spread_by_name
        fill_fees_by_name = total_fill.abs() * config.fee_bps / 10_000
        fill_fees = fill_fees_by_name.sum(dim=-1)
        step_spread = regular_spread + forced_spread
        step_cost_by_name = step_spread_by_name + fill_fees_by_name
        gross_fill = total_fill.abs()
        turnover_by_name = turnover_by_name + gross_fill
        spread_by_name = spread_by_name + step_spread_by_name
        fees_by_name = fees_by_name + fill_fees_by_name

        old_active = old_position.abs() > config.position_tolerance_brl
        new_active = position.abs() > config.position_tolerance_brl
        same_side = (
            old_active & new_active & (torch.sign(old_position) == torch.sign(position))
        )
        closes = old_active & ~same_side
        crossing = closes & new_active
        close_fraction = torch.where(
            crossing,
            old_position.abs() / total_fill.abs().clamp_min(torch.finfo(dtype).tiny),
            torch.ones_like(position),
        )
        closing_cost = torch.where(closes, step_cost_by_name * close_fraction, 0)
        opening_cost = step_cost_by_name - closing_cost
        episode_cost = episode_cost + closing_cost
        round_trip_count = round_trip_count + closes.to(torch.int64)
        round_trip_gross = round_trip_gross + torch.where(closes, episode_gross, 0)
        round_trip_cost = round_trip_cost + torch.where(closes, episode_cost, 0)
        episode_gross = torch.where(
            closes, torch.zeros_like(episode_gross), episode_gross
        )
        episode_cost = torch.where(closes, torch.zeros_like(episode_cost), episode_cost)
        episode_cost = episode_cost + opening_cost
        nav = nav - step_spread - fill_fees
        spread_cost = spread_cost + step_spread
        fees = fees + fill_fees
        turnover = turnover + total_fill.abs().sum(dim=-1)
        maximum_gross = torch.maximum(maximum_gross, position.abs().sum(dim=-1))
        deployed_gross_sum = deployed_gross_sum + position.abs().sum(dim=-1)

        if return_path:
            positions.append(position)
            fills.append(total_fill)
            carried.append(residual)
            targets.append(base_target)

    accounting = gross_pnl - spread_cost - fees + cdi
    scanned_net = nav - initial_nav
    # The scanned NAV performs several additions at NAV scale per minute,
    # while the component ledger accumulates from zero. Bound their expected
    # floating-point drift by the replay length without changing reported PnL.
    rounding_steps = max(64, 2 * minutes)
    tolerance = rounding_steps * torch.finfo(dtype).eps * initial_nav
    if not torch.all((scanned_net - accounting).abs() <= tolerance):
        raise ArithmeticError("Execution accounting identity failed")
    if not torch.all(position.abs() <= config.position_tolerance_brl):
        raise ArithmeticError("Execution replay did not finish flat")
    if (round_trip_count == 0).logical_and(round_trip_cost != 0).any():
        raise ArithmeticError("Unclosed execution episode retained costs")
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
        mean_deployed_gross_brl=deployed_gross_sum / minutes,
        turnover_by_name_brl=turnover_by_name,
        gross_pnl_by_name_brl=gross_by_name,
        spread_cost_by_name_brl=spread_by_name,
        fees_by_name_brl=fees_by_name,
        round_trip_count_by_name=round_trip_count,
        round_trip_gross_pnl_by_name_brl=round_trip_gross,
        round_trip_cost_by_name_brl=round_trip_cost,
        selection_extended_count=extension_path,
        positions_brl=torch.stack(positions, dim=1) if return_path else None,
        fills_brl=torch.stack(fills, dim=1) if return_path else None,
        carried_demand_brl=torch.stack(carried, dim=1) if return_path else None,
        target_weights=torch.stack(targets, dim=1) if return_path else None,
    )
