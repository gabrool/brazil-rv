from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import cache

import torch
from torch import Tensor
from torch.nn import functional as F


POLICY_STATE_SCHEMA = {
    "schema": "BRAZIL_RV_POLICY_STATE_V1",
    "per_name": [
        {
            "name": "rank_by_horizon",
            "width": "horizon_count",
            "scaling": "identity; causal cross-sectional ranks in [-1,1]",
        },
        {
            "name": "rank_change_by_horizon",
            "width": "horizon_count",
            "scaling": "divide_by_2",
        },
        {
            "name": "signal_age_minutes",
            "width": 1,
            "scaling": "log1p(value)/log1p(session_minutes)",
        },
        {
            "name": "current_weight",
            "width": 1,
            "scaling": "divide_by_current_name_cap",
        },
        {
            "name": "unrealized_pnl_daily_sigma",
            "width": 1,
            "scaling": "asinh(signed_return/daily_sigma)",
        },
        {
            "name": "minutes_in_position",
            "width": 1,
            "scaling": "log1p(value)/log1p(session_minutes)",
        },
        {
            "name": "lagged_full_spread_bps",
            "width": 1,
            "scaling": "log1p(value)/log1p(75)",
        },
        {
            "name": "daily_sigma",
            "width": 1,
            "scaling": "log1p(value/0.01)",
        },
        {
            "name": "liquidity_tercile",
            "width": 3,
            "scaling": "one_hot_low_mid_high_from_causal_ADV20",
        },
        {
            "name": "minute_participation_capacity",
            "width": 1,
            "scaling": "log1p(capacity_fraction_NAV/0.01)",
        },
    ],
    "portfolio": [
        {"name": "gross", "scaling": "divide_by_gross_target"},
        {"name": "cash_fraction", "scaling": "CDI_margin_cash/NAV"},
        {
            "name": "day_pnl_bps_NAV",
            "scaling": "asinh(day_pnl_bps/10)",
        },
    ],
    "time": [
        {"name": "session_elapsed", "scaling": "minute/(session_minutes-1)"},
        {
            "name": "session_remaining",
            "scaling": "(session_minutes-1-minute)/(session_minutes-1)",
        },
    ],
}
DEFAULT_POLICY_HORIZON_NAMES = ("30m", "60m", "120m")


def policy_state_feature_names(
    horizon_count: int, horizon_names: tuple[str, ...] | None = None
) -> tuple[str, ...]:
    if horizon_count <= 0:
        raise ValueError("Policy state requires at least one prediction horizon")
    names = (
        tuple(f"h{index}" for index in range(horizon_count))
        if horizon_names is None
        else horizon_names
    )
    if (
        len(names) != horizon_count
        or len(set(names)) != horizon_count
        or not all(names)
    ):
        raise ValueError(
            "Ordered horizon names must be unique and match the rank width"
        )
    return (
        *(f"rank_{name}" for name in names),
        *(f"rank_change_{name}" for name in names),
        "signal_age_minutes",
        "current_weight",
        "unrealized_pnl_daily_sigma",
        "minutes_in_position",
        "lagged_full_spread_bps",
        "daily_sigma",
        "liquidity_tercile_low",
        "liquidity_tercile_mid",
        "liquidity_tercile_high",
        "minute_participation_capacity",
        "portfolio_gross",
        "portfolio_cash_fraction",
        "portfolio_day_pnl_bps_NAV",
        "session_elapsed",
        "session_remaining",
    )


def policy_state_feature_width(horizon_count: int) -> int:
    return len(policy_state_feature_names(horizon_count))


@cache
def policy_state_schema_sha256(horizon_names: tuple[str, ...]) -> str:
    payload = {
        "schema": POLICY_STATE_SCHEMA,
        "horizon_names": list(horizon_names),
        "feature_names": list(
            policy_state_feature_names(len(horizon_names), horizon_names)
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


POLICY_STATE_SCHEMA_SHA256 = policy_state_schema_sha256(DEFAULT_POLICY_HORIZON_NAMES)


@dataclass(frozen=True)
class PolicyState:
    """Causal state at one simulator action minute."""

    per_name: Tensor
    portfolio: Tensor
    time: Tensor
    tradeable_mask: Tensor
    cap_weights: Tensor
    horizon_names: tuple[str, ...] = DEFAULT_POLICY_HORIZON_NAMES

    def __post_init__(self) -> None:
        expected_per_name = 2 * len(self.horizon_names) + 10
        if self.per_name.ndim != 3 or self.per_name.shape[-1] != expected_per_name:
            raise ValueError(
                "PolicyState per-name width differs from its horizon schema"
            )
        shape = self.per_name.shape[:2]
        if (
            self.portfolio.shape != (shape[0], 3)
            or self.time.shape != (shape[0], 2)
            or self.tradeable_mask.shape != shape
            or self.cap_weights.shape != shape
        ):
            raise ValueError("PolicyState tensor axes do not align")

    @property
    def schema_sha256(self) -> str:
        return policy_state_schema_sha256(self.horizon_names)

    def features(self) -> Tensor:
        days, names, _ = self.per_name.shape
        portfolio = self.portfolio[:, None, :].expand(days, names, -1)
        time = self.time[:, None, :].expand(days, names, -1)
        return torch.cat((self.per_name, portfolio, time), dim=-1)


def liquidity_tercile_one_hot(adv20_brl: Tensor, eligible: Tensor) -> Tensor:
    """Deterministically bucket each day's eligible names by lagged ADV20."""
    if adv20_brl.ndim != 2 or eligible.shape != adv20_brl.shape:
        raise ValueError("ADV and eligibility must be aligned [day,name] tensors")
    eligible = eligible.bool() & torch.isfinite(adv20_brl)
    names = adv20_brl.shape[-1]
    order = torch.argsort(
        torch.where(eligible, adv20_brl, torch.full_like(adv20_brl, torch.inf)),
        dim=-1,
        stable=True,
    )
    position = torch.arange(names, device=adv20_brl.device)[None, :].expand_as(order)
    count = eligible.sum(dim=-1, keepdim=True)
    bucket = torch.div(
        position * 3,
        count.clamp_min(1),
        rounding_mode="floor",
    ).clamp_max(2)
    ordered_one_hot = F.one_hot(bucket, 3).to(adv20_brl.dtype)
    ordered_one_hot = ordered_one_hot * (position < count).unsqueeze(-1)
    output = torch.zeros_like(ordered_one_hot)
    return output.scatter(1, order.unsqueeze(-1).expand(-1, -1, 3), ordered_one_hot)


def update_volume_weighted_cost_basis(
    old_position_brl: Tensor,
    fill_brl: Tensor,
    fill_price: Tensor,
    old_cost_basis: Tensor,
    position_tolerance_brl: float,
) -> tuple[Tensor, Tensor]:
    """Update signed shares and their entry-price basis after one fill.

    Reductions preserve the old basis, same-side additions use an absolute-share
    VWAP, and a new or crossed position starts at the current fill price.
    """
    if not (
        old_position_brl.shape
        == fill_brl.shape
        == fill_price.shape
        == old_cost_basis.shape
    ):
        raise ValueError("Cost-basis tensors must have identical shapes")
    safe_price = torch.where(
        torch.isfinite(fill_price) & (fill_price > 0),
        fill_price,
        torch.ones_like(fill_price),
    )
    old_shares = old_position_brl / safe_price
    fill_shares = fill_brl / safe_price
    new_shares = old_shares + fill_shares
    old_active = old_position_brl.abs() > position_tolerance_brl
    new_position = old_position_brl + fill_brl
    new_active = new_position.abs() > position_tolerance_brl
    same_side = (
        old_active & new_active & (torch.sign(old_shares) == torch.sign(new_shares))
    )
    increase = same_side & (new_shares.abs() > old_shares.abs())
    safe_old_basis = torch.where(
        torch.isfinite(old_cost_basis), old_cost_basis, safe_price
    )
    added = (new_shares.abs() - old_shares.abs()).clamp_min(0)
    vwap = (
        old_shares.abs() * safe_old_basis + added * safe_price
    ) / new_shares.abs().clamp_min(torch.finfo(fill_price.dtype).tiny)
    basis = torch.where(increase, vwap, safe_old_basis)
    opened_or_crossed = new_active & (
        ~old_active | (torch.sign(old_shares) != torch.sign(new_shares))
    )
    basis = torch.where(opened_or_crossed, safe_price, basis)
    basis = torch.where(new_active, basis, torch.full_like(basis, torch.nan))
    return new_shares, basis


def build_policy_state(
    *,
    ranks: Tensor,
    rank_change: Tensor,
    signal_age_minutes: Tensor,
    current_weights: Tensor,
    current_price: Tensor,
    cost_basis_price: Tensor,
    minutes_in_position: Tensor,
    lagged_full_spread: Tensor,
    daily_sigma: Tensor,
    adv20_brl: Tensor,
    participation_capacity_brl: Tensor,
    nav_brl: Tensor,
    initial_nav_brl: Tensor,
    tradeable_mask: Tensor,
    cap_weights: Tensor,
    gross_target: float,
    margin_fraction_of_gross: float,
    session_minute: int,
    session_minutes: int,
    horizon_names: tuple[str, ...] | None = None,
) -> PolicyState:
    """Build the frozen, scaled state using information available through ``t``."""
    if ranks.ndim != 3 or rank_change.shape != ranks.shape:
        raise ValueError("Ranks and rank changes must be [day,name,horizon]")
    shape = ranks.shape[:2]
    per_name_inputs = (
        signal_age_minutes,
        current_weights,
        current_price,
        cost_basis_price,
        minutes_in_position,
        lagged_full_spread,
        daily_sigma,
        adv20_brl,
        participation_capacity_brl,
        tradeable_mask,
        cap_weights,
    )
    if any(value.shape != shape for value in per_name_inputs):
        raise ValueError("Per-name policy inputs do not align")
    if nav_brl.shape != shape[:1] or initial_nav_brl.shape != shape[:1]:
        raise ValueError("NAV inputs must contain one value per day")
    if not 0 <= session_minute < session_minutes or session_minutes < 2:
        raise ValueError("Policy-state minute lies outside the session")
    if gross_target <= 0 or not math.isfinite(gross_target):
        raise ValueError("Gross target must be finite and positive")
    resolved_horizons = horizon_names
    if resolved_horizons is None:
        if ranks.shape[-1] != len(DEFAULT_POLICY_HORIZON_NAMES):
            raise ValueError("Noncanonical rank widths require ordered horizon names")
        resolved_horizons = DEFAULT_POLICY_HORIZON_NAMES

    dtype = ranks.dtype
    mask = tradeable_mask.bool()
    session_denominator = math.log1p(session_minutes)
    safe_cap = torch.where(
        torch.isfinite(cap_weights),
        cap_weights.clamp_min(torch.finfo(dtype).tiny),
        torch.ones_like(cap_weights),
    )
    weight_fraction = torch.where(cap_weights > 0, current_weights / safe_cap, 0)
    held = current_weights != 0
    valid_basis = (
        held & torch.isfinite(current_price) & torch.isfinite(cost_basis_price)
    )
    safe_current_price = torch.where(
        valid_basis, current_price, torch.ones_like(current_price)
    )
    safe_cost_basis = torch.where(
        valid_basis, cost_basis_price, torch.ones_like(cost_basis_price)
    )
    signed_return = torch.where(
        valid_basis,
        torch.sign(current_weights) * (safe_current_price / safe_cost_basis - 1),
        0,
    )
    valid_sigma = torch.isfinite(daily_sigma) & (daily_sigma > 0)
    safe_sigma = torch.where(valid_sigma, daily_sigma, torch.ones_like(daily_sigma))
    unrealized = torch.where(
        valid_sigma,
        signed_return / safe_sigma,
        0,
    )
    safe_spread = torch.where(
        torch.isfinite(lagged_full_spread),
        lagged_full_spread.clamp_min(0),
        torch.zeros_like(lagged_full_spread),
    )
    spread_bps = safe_spread * 10_000
    liquidity = liquidity_tercile_one_hot(adv20_brl, mask)
    capacity_fraction = participation_capacity_brl.clamp_min(0) / nav_brl[:, None]
    per_name = torch.cat(
        (
            ranks,
            rank_change / 2,
            (torch.log1p(signal_age_minutes.clamp_min(0)) / session_denominator)[
                ..., None
            ],
            weight_fraction[..., None],
            torch.asinh(unrealized)[..., None],
            (torch.log1p(minutes_in_position.clamp_min(0)) / session_denominator)[
                ..., None
            ],
            (torch.log1p(spread_bps) / math.log1p(75.0))[..., None],
            torch.log1p(torch.where(valid_sigma, daily_sigma, 0) / 0.01)[..., None],
            liquidity,
            torch.log1p(capacity_fraction / 0.01)[..., None],
        ),
        dim=-1,
    )
    per_name = torch.where(mask[..., None], per_name, torch.zeros_like(per_name))

    gross = current_weights.abs().sum(dim=-1)
    cash = (
        nav_brl - margin_fraction_of_gross * current_weights.abs().sum(dim=-1) * nav_brl
    ).clamp_min(0)
    day_pnl_bps = (nav_brl / initial_nav_brl - 1) * 10_000
    portfolio = torch.stack(
        (gross / gross_target, cash / nav_brl, torch.asinh(day_pnl_bps / 10)),
        dim=-1,
    )
    time_denominator = session_minutes - 1
    elapsed = session_minute / time_denominator
    time = ranks.new_tensor([elapsed, 1 - elapsed]).expand(shape[0], 2)
    state = PolicyState(per_name, portfolio, time, mask, cap_weights, resolved_horizons)
    if state.features().shape[-1] != policy_state_feature_width(ranks.shape[-1]):
        raise AssertionError("Policy-state schema width differs from its tensor")
    return state
