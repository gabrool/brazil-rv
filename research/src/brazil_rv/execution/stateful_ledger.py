from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from typing import Literal, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from brazil_rv.v2.corporate_actions import AlignedActionTerms, apply_contractual_action
from brazil_rv.v2.hedge_beta import resolve_hedge_beta


OrderSide = Literal["buy", "sell"]
BorrowSource = Literal["uniform", "borrow_strict", "borrow_balance", "borrow_open"]
OrderPurpose = Literal[
    "entry", "exit", "risk_exit", "terminal_exit", "terminal_settlement", "hedge"
]
ExitInstructionCause = Literal[
    "ineligible_hold_exhausted",
    "settlement_grace",
    "rank_out_of_retention",
    "terminal",
]
CancellationReason = Literal[
    "band_exit",
    "expired",
    "evaluation_end",
    "exit_instruction",
    "prior_action_unresolved",
    "position_closed",
    "risk_net_cap",
    "risk_name_cap",
    "terminal_settlement",
]
_CANCELLATION_REASONS: tuple[CancellationReason, ...] = (
    "band_exit",
    "expired",
    "evaluation_end",
    "exit_instruction",
    "prior_action_unresolved",
    "position_closed",
    "risk_net_cap",
    "risk_name_cap",
    "terminal_settlement",
)

TERMINAL_SETTLEMENT_CONVENTION = "last_mark_after_10_sessions"
EXIT_INSTRUCTION_CAUSE_CODES: dict[ExitInstructionCause, int] = {
    "ineligible_hold_exhausted": 1,
    "settlement_grace": 2,
    "rank_out_of_retention": 3,
    "terminal": 4,
}


@dataclass(frozen=True)
class LedgerConfig:
    """Stateful long/short close-proxy ledger controls.

    The 20% planned-net cap is deliberately wider than the former 10% cap.
    With 30 independent slots per side and asynchronous fills/exits, 10% made
    ordinary temporary fill imbalance bind before the strategy could restore
    the light side.  The 20% cap still limits directional drift while allowing
    the six-slot daily turnover pattern to refill causally.
    """

    k_per_side: int = 30
    buffer_per_side: int = 30
    gross_target: float = 2.0
    planned_gross_cap: float = 2.25
    planned_absolute_net_cap: float = 0.20
    planned_name_weight_cap: float = 0.05
    cost_bps_per_side: float = 4.0
    annual_borrow_rate: float = 0.02
    borrow_source: BorrowSource = "borrow_balance"
    borrow_registration_fee_fraction: float = 0.20
    borrow_registration_fee_floor: float = 0.00025
    borrow_registration_fee_cap: float = 0.0070
    volatility_balanced_entries: bool = True
    volatility_group_count: int = 5
    small_stratum_scaling_threshold_multiple: int = 2
    beta_hedge: bool = True
    hedge_rebalance_threshold_nav: float = 0.05
    hedge_notional_cap_nav: float = 0.60
    hedge_cost_bps_per_side: float = 4.0
    hedge_annual_borrow_rate: float = 0.02
    annual_debit_spread: float = 0.0
    short_proceeds_remuneration: float = 1.0
    initial_capital_brl: float = 1.0
    entry_expiry_sessions: int = 3
    cancel_pending_outside_retention: bool = True
    ineligible_hold_sessions: int = 5
    settlement_grace_sessions: int = 10
    settlement_haircut: float = 0.30
    settlement_economics_unresolved_fraction_nav: float = 0.15
    settle_terminal_residuals: bool = False
    annual_sessions: int = 252

    def __post_init__(self) -> None:
        if self.k_per_side <= 0 or self.buffer_per_side < 0:
            raise ValueError("ledger K must be positive and its buffer non-negative")
        if self.gross_target <= 0 or self.cost_bps_per_side < 0:
            raise ValueError("ledger gross/cost controls are invalid")
        if (
            self.planned_gross_cap <= 0
            or self.planned_absolute_net_cap < 0
            or self.planned_name_weight_cap <= 0
        ):
            raise ValueError("planned risk caps are invalid")
        if (
            self.annual_borrow_rate < 0
            or self.annual_debit_spread < 0
            or self.annual_sessions <= 0
        ):
            raise ValueError("ledger financing controls are invalid")
        if self.borrow_source not in {
            "uniform",
            "borrow_strict",
            "borrow_balance",
            "borrow_open",
        }:
            raise ValueError("borrow source is not a registered rev4b cell")
        if (
            self.borrow_registration_fee_fraction < 0.0
            or self.borrow_registration_fee_floor < 0.0
            or self.borrow_registration_fee_cap < self.borrow_registration_fee_floor
        ):
            raise ValueError("borrow registration fee schedule is invalid")
        if self.volatility_group_count < 1:
            raise ValueError("volatility group count must be positive")
        if self.small_stratum_scaling_threshold_multiple < 2:
            raise ValueError("small-stratum scaling multiple must be at least two")
        if (
            self.hedge_rebalance_threshold_nav < 0.0
            or self.hedge_notional_cap_nav <= 0.0
            or self.hedge_cost_bps_per_side < 0.0
            or self.hedge_annual_borrow_rate < 0.0
        ):
            raise ValueError("hedge controls must be non-negative")
        if not 0 <= self.short_proceeds_remuneration <= 1:
            raise ValueError("short-proceeds remuneration must be in [0, 1]")
        if self.initial_capital_brl <= 0 or not np.isfinite(self.initial_capital_brl):
            raise ValueError("initial capital must be positive and finite")
        if self.entry_expiry_sessions < 1:
            raise ValueError("entry expiry must be at least one session")
        if self.ineligible_hold_sessions < 0:
            raise ValueError("ineligible hold must be non-negative")
        if self.settlement_grace_sessions < 1:
            raise ValueError("settlement grace must be at least one session")
        if not 0 <= self.settlement_haircut < 1:
            raise ValueError("settlement haircut must be in [0, 1)")
        if self.settlement_economics_unresolved_fraction_nav <= 0:
            raise ValueError("settlement incidence bound must be positive")


def equity_borrow_registration_fee(
    annual_rate: NDArray[np.float64] | float,
    *,
    config: LedgerConfig,
) -> NDArray[np.float64] | float:
    """Return the registered B3 borrower fee for an annual lending rate."""

    fee = np.clip(
        np.asarray(annual_rate, dtype=np.float64)
        * config.borrow_registration_fee_fraction,
        config.borrow_registration_fee_floor,
        config.borrow_registration_fee_cap,
    )
    return float(fee) if fee.ndim == 0 else fee


@dataclass(frozen=True)
class IntendedOrder:
    order_id: str
    security: str
    security_index: int
    decision_date: date
    decision_session: int
    side: OrderSide
    planned_notional: float | None
    position_fraction: float | None
    reference_price: float
    order_type: Literal["close_proxy"]
    limit_price: float | None
    expiry_date: date | None
    expiry_session: int | None
    purpose: OrderPurpose


@dataclass(frozen=True)
class Fill:
    order_id: str
    security: str
    security_index: int
    fill_date: date
    fill_session: int
    side: OrderSide
    quantity: float
    price: float
    gross_notional: float
    cost: float
    purpose: OrderPurpose


@dataclass(frozen=True)
class OrderCancellation:
    order_id: str
    security: str
    security_index: int
    cancellation_date: date
    cancellation_session: int
    unfilled_notional: float | None
    unfilled_position_fraction: float | None
    reason: CancellationReason


@dataclass
class _PendingOrder:
    order: IntendedOrder
    # Notional for entries/hedge; fraction of the CURRENT inventory for exits.
    # A partial exit rebases this fraction onto the inventory left after its fill.
    remaining_size: float

    def cancellation(
        self, day: int, day_date: date, reason: CancellationReason
    ) -> OrderCancellation:
        return OrderCancellation(
            order_id=self.order.order_id,
            security=self.order.security,
            security_index=self.order.security_index,
            cancellation_date=day_date,
            cancellation_session=day,
            unfilled_notional=self.remaining_size
            if self.order.planned_notional is not None
            else None,
            unfilled_position_fraction=self.remaining_size
            if self.order.position_fraction is not None
            else None,
            reason=reason,
        )


@dataclass(frozen=True)
class _PendingClaim:
    security_index: int
    signed_amount: float
    payment_session: int | None


@dataclass(frozen=True)
class _ValidatedInputs:
    dates: tuple[date, ...]
    score: NDArray[np.float64]
    score_valid: NDArray[np.bool_]
    membership: NDArray[np.bool_]
    raw_close: NDArray[np.float64]
    cdi: NDArray[np.float64]
    fill_fraction: NDArray[np.float64]
    action_q: NDArray[np.float64]
    action_d: NDArray[np.float64]
    action_resolved: NDArray[np.bool_]
    has_action: NDArray[np.bool_]
    action_successor: NDArray[np.int64]
    action_payment_session: NDArray[np.int64]
    securities: tuple[str, ...]
    initial_reference_price: NDArray[np.float64]
    annual_borrow_rate_by_name: NDArray[np.float64]
    borrow_rate_imputed: NDArray[np.bool_]
    borrow_rate_placeholder: NDArray[np.bool_]
    shortable: NDArray[np.bool_]
    selection_volatility: NDArray[np.float64]
    hedge_beta: NDArray[np.float64]
    hedge_beta_fallback: NDArray[np.bool_]
    hedge_close: NDArray[np.float64]
    hedge_annual_borrow_rate: NDArray[np.float64]


@dataclass(frozen=True)
class StatefulLedgerResult:
    dates: tuple[date, ...]
    start_nav: NDArray[np.float64]
    nav: NDArray[np.float64]
    daily_net_return: NDArray[np.float64]
    net_excess_all_cash_bps: NDArray[np.float64]
    gross_pnl_bps: NDArray[np.float64]
    equity_gross_pnl_bps: NDArray[np.float64]
    hedge_gross_pnl_bps: NDArray[np.float64]
    interest_bps: NDArray[np.float64]
    free_cash_interest_bps: NDArray[np.float64]
    short_proceeds_interest_bps: NDArray[np.float64]
    short_proceeds_interest_base: NDArray[np.float64]
    cost_bps: NDArray[np.float64]
    borrow_bps: NDArray[np.float64]
    equity_borrow_raw_bps: NDArray[np.float64]
    equity_borrow_fee_bps: NDArray[np.float64]
    cdi_benchmark_bps: NDArray[np.float64]
    held_short_weighted_annual_borrow_rate: NDArray[np.float64]
    held_short_notional_at_open: NDArray[np.float64]
    held_short_imputed_notional_at_open: NDArray[np.float64]
    held_short_placeholder_notional_at_open: NDArray[np.float64]
    excluded_short_entry_candidate_count: NDArray[np.int64]
    gross_fraction_nav: NDArray[np.float64]
    turnover_fraction_nav: NDArray[np.float64]
    unresolved_stale_inventory_fraction_nav: NDArray[np.float64]
    unresolved_claim_inventory_fraction_nav: NDArray[np.float64]
    stale_mark_inventory_fraction_nav: NDArray[np.float64]
    stale_mark_name_days: NDArray[np.int64]
    same_day_action_sessions: NDArray[np.bool_]
    unresolved_action_name_days: NDArray[np.int64]
    valuation_scenario_count: NDArray[np.int64]
    position_sign: NDArray[np.int8]
    signed_shares: NDArray[np.float64]
    mark_price: NDArray[np.float64]
    free_cash: NDArray[np.float64]
    restricted_cash: NDArray[np.float64]
    hedge_restricted_cash: NDArray[np.float64]
    receivables: NDArray[np.float64]
    payables: NDArray[np.float64]
    marked_signed_holdings: NDArray[np.float64]
    reconciliation_error: NDArray[np.float64]
    all_cash_nav: NDArray[np.float64]
    settlement_haircut_scenario_nav: NDArray[np.float64]
    unresolved_excluded_nav: NDArray[np.float64]
    pending_entry_count: NDArray[np.int64]
    pending_exit_count: NDArray[np.int64]
    cancelled_entry_count: NDArray[np.int64]
    blocked_entry_no_reference_count: NDArray[np.int64]
    blocked_entry_gross_cap_count: NDArray[np.int64]
    blocked_entry_net_cap_count: NDArray[np.int64]
    blocked_entry_name_cap_count: NDArray[np.int64]
    submitted_entry_count: NDArray[np.int64]
    submitted_exit_count: NDArray[np.int64]
    same_close_replacement_count: NDArray[np.int64]
    zero_entry_small_universe: NDArray[np.bool_]
    zero_entry_gross_cap: NDArray[np.bool_]
    zero_entry_net_cap: NDArray[np.bool_]
    zero_entry_name_cap: NDArray[np.bool_]
    zero_entry_no_reference: NDArray[np.bool_]
    risk_trim_gross_notional: NDArray[np.float64]
    risk_trim_net_notional: NDArray[np.float64]
    risk_trim_name_notional: NDArray[np.float64]
    pending_exit_mean_age_sessions: NDArray[np.float64]
    retention_width: NDArray[np.int64]
    entry_blocked_small_universe: NDArray[np.bool_]
    actual_risk_breach: NDArray[np.bool_]
    planned_gross_fraction_nav: NDArray[np.float64]
    planned_net_fraction_nav: NDArray[np.float64]
    planned_name_weight_fraction_nav: NDArray[np.float64]
    holding_age_sessions: NDArray[np.int64]
    terminal_settlement_count: NDArray[np.int64]
    terminal_settlement_notional: NDArray[np.float64]
    terminal_settlement_notional_fraction_nav: NDArray[np.float64]
    settled_then_printed_count: NDArray[np.int64]
    k_eff_per_side: NDArray[np.int64]
    held_count_start_of_day_long: NDArray[np.int64]
    held_count_start_of_day_short: NDArray[np.int64]
    held_count_end_of_day_long: NDArray[np.int64]
    held_count_end_of_day_short: NDArray[np.int64]
    occupied_after_submission_long: NDArray[np.int64]
    occupied_after_submission_short: NDArray[np.int64]
    open_slots_after_submission_long: NDArray[np.int64]
    open_slots_after_submission_short: NDArray[np.int64]
    band_candidates_long: NDArray[np.int64]
    band_candidates_short: NDArray[np.int64]
    band_excluded_unresolved_long: NDArray[np.int64]
    band_excluded_unresolved_short: NDArray[np.int64]
    band_excluded_settled_long: NDArray[np.int64]
    band_excluded_settled_short: NDArray[np.int64]
    band_candidates_without_prior_session_print_long: NDArray[np.int64]
    band_candidates_without_prior_session_print_short: NDArray[np.int64]
    band_exhausted_open_slots_long: NDArray[np.int64]
    band_exhausted_open_slots_short: NDArray[np.int64]
    blocked_open_slots_long: NDArray[np.int64]
    blocked_open_slots_short: NDArray[np.int64]
    pending_entries_end_of_day_long: NDArray[np.int64]
    pending_entries_end_of_day_short: NDArray[np.int64]
    pending_entries_without_print_today_long: NDArray[np.int64]
    pending_entries_without_print_today_short: NDArray[np.int64]
    slots_freed_by_exit_fill_today_long: NDArray[np.int64]
    slots_freed_by_exit_fill_today_short: NDArray[np.int64]
    exit_instructions_ineligible_hold_exhausted: NDArray[np.int64]
    exit_instructions_settlement_grace: NDArray[np.int64]
    exit_instructions_rank_out_of_retention: NDArray[np.int64]
    exit_instructions_terminal: NDArray[np.int64]
    exit_instruction_cause: NDArray[np.int8]
    exit_instruction_side: NDArray[np.int8]
    ineligible_streak: NDArray[np.int64]
    ineligible_held_sessions_score_invalid: NDArray[np.int64]
    ineligible_held_sessions_membership_invalid: NDArray[np.int64]
    ineligible_held_sessions_score_nonfinite: NDArray[np.int64]
    ineligible_exit_within_hold_window: NDArray[np.int64]
    net_cap_block_with_balanced_book: NDArray[np.int64]
    gross_cap_block_below_target: NDArray[np.int64]
    name_cap_block_on_fresh_entry: NDArray[np.int64]
    entry_pending_printed_unblocked_unfilled: NDArray[np.int64]
    entry_fill_quantity_short: NDArray[np.int64]
    entry_cost_basis: NDArray[np.float64]
    submission_nav: NDArray[np.float64]
    gross_shortfall_small_universe: NDArray[np.float64]
    gross_shortfall_occupancy_pending: NDArray[np.float64]
    gross_shortfall_occupancy_band_exhausted: NDArray[np.float64]
    gross_shortfall_occupancy_blocked: NDArray[np.float64]
    gross_shortfall_occupancy_exit_gap: NDArray[np.float64]
    gross_shortfall_occupancy_other: NDArray[np.float64]
    gross_shortfall_sizing_fill: NDArray[np.float64]
    gross_shortfall_sizing_mark_drift: NDArray[np.float64]
    gross_shortfall_sizing_nav_drift: NDArray[np.float64]
    ineligible_exit_reeligible_within_10_sessions_share: float
    intended_orders: tuple[IntendedOrder, ...]
    fills: tuple[Fill, ...]
    cancellations: tuple[OrderCancellation, ...]
    unresolved_inventory_count: int
    unresolved_inventory_notional: float
    terminal_unresolved_reason_breakdown: dict[str, dict[str, int | float]]
    unresolved_receivable: float
    unresolved_payable: float
    insolvent: bool
    insolvency_date: date | None
    economics_unresolved: bool
    terminal_boundary_unpriced_inventory_notional: float
    terminal_hedge_last_mark_settlement_notional: float
    gross_target: float
    ineligible_hold_sessions: int
    settlement_grace_sessions: int
    settlement_haircut: float
    settlement_economics_unresolved_fraction_nav: float
    share_sizing_mode: Literal[
        "fractional_notional_research_proxy", "round_lot_close_proxy"
    ]
    borrow_source: BorrowSource
    volatility_balanced_entries: bool
    beta_hedge: bool
    hedge_notional_cap_nav: float
    volatility_quota: NDArray[np.int64]
    volatility_group_size: NDArray[np.int64]
    entry_eligible_name_count: NDArray[np.int64]
    volatility_occupancy_long: NDArray[np.int64]
    volatility_occupancy_short: NDArray[np.int64]
    volatility_spilled_entries_long: NDArray[np.int64]
    volatility_spilled_entries_short: NDArray[np.int64]
    hedge_signed_shares: NDArray[np.float64]
    hedge_mark_price: NDArray[np.float64]
    hedge_signed_notional: NDArray[np.float64]
    hedge_unconstrained_target_notional: NDArray[np.float64]
    hedge_target_notional: NDArray[np.float64]
    hedge_beta_fallback_sessions: NDArray[np.bool_]
    hedge_capped: NDArray[np.bool_]
    hedge_turnover_fraction_nav: NDArray[np.float64]
    hedge_cost_bps: NDArray[np.float64]
    hedge_borrow_bps: NDArray[np.float64]
    ex_ante_beta_before_hedge: NDArray[np.float64]
    ex_ante_beta_after_hedge: NDArray[np.float64]
    gross_fraction_nav_including_hedge: NDArray[np.float64]
    net_notional_including_hedge: NDArray[np.float64]
    net_fraction_nav_including_hedge: NDArray[np.float64]

    def summary(self) -> dict[str, object]:
        excess = self.net_excess_all_cash_bps
        finite_excess = excess[np.isfinite(excess)]
        standard_deviation = (
            float(np.std(finite_excess, ddof=1)) if finite_excess.size > 1 else 0.0
        )
        sharpe = (
            float(np.mean(finite_excess) / standard_deviation * np.sqrt(252.0))
            if standard_deviation > 0
            else 0.0
        )
        finite_gross = self.gross_fraction_nav[np.isfinite(self.gross_fraction_nav)]
        finite_turnover = self.turnover_fraction_nav[
            np.isfinite(self.turnover_fraction_nav)
        ]
        mean_gross = float(np.mean(finite_gross)) if finite_gross.size else 0.0
        minimum_gross = float(np.min(finite_gross)) if finite_gross.size else 0.0
        maximum_gross = float(np.max(finite_gross)) if finite_gross.size else 0.0
        mean_turnover = float(np.mean(finite_turnover)) if finite_turnover.size else 0.0
        all_cash_terminal = float(self.all_cash_nav[-1])
        entry_orders = [
            order for order in self.intended_orders if order.purpose == "entry"
        ]
        intended_entry_notional = float(
            sum(order.planned_notional for order in entry_orders)
        )
        filled_entry_notional = float(
            sum(fill.gross_notional for fill in self.fills if fill.purpose == "entry")
        )
        pending_exit_age = self.pending_exit_mean_age_sessions[
            np.isfinite(self.pending_exit_mean_age_sessions)
            & (self.pending_exit_count > 0)
        ]
        cancellation_reasons = {
            reason: sum(
                cancellation.reason == reason for cancellation in self.cancellations
            )
            for reason in _CANCELLATION_REASONS
        }
        decomposition_arrays = {
            "small_universe": self.gross_shortfall_small_universe,
            "occupancy_pending": self.gross_shortfall_occupancy_pending,
            "occupancy_band_exhausted": (self.gross_shortfall_occupancy_band_exhausted),
            "occupancy_blocked": self.gross_shortfall_occupancy_blocked,
            "occupancy_exit_gap": self.gross_shortfall_occupancy_exit_gap,
            "occupancy_other": self.gross_shortfall_occupancy_other,
            "sizing_fill": self.gross_shortfall_sizing_fill,
            "sizing_mark_drift": self.gross_shortfall_sizing_mark_drift,
            "sizing_nav_drift": self.gross_shortfall_sizing_nav_drift,
        }
        decomposition = {
            name: float(np.mean(values)) if values.size else 0.0
            for name, values in decomposition_arrays.items()
        }
        decomposition["total"] = self.gross_target - mean_gross
        occupancy_total = decomposition["small_universe"] + sum(
            decomposition[name]
            for name in (
                "occupancy_pending",
                "occupancy_band_exhausted",
                "occupancy_blocked",
                "occupancy_exit_gap",
                "occupancy_other",
            )
        )
        decomposition["occupancy_share"] = (
            occupancy_total / decomposition["total"]
            if decomposition["total"] > 0.0
            else 0.0
        )
        decomposition_sum = sum(decomposition[name] for name in decomposition_arrays)
        if not np.isclose(
            decomposition_sum,
            decomposition["total"],
            rtol=0.0,
            atol=1e-9,
        ):
            raise RuntimeError("gross shortfall decomposition does not reconcile")
        defect_signatures = {
            "D1_entry_pending_printed_unblocked_unfilled": int(
                self.entry_pending_printed_unblocked_unfilled.sum()
            ),
            "D2_entry_fill_quantity_short": int(self.entry_fill_quantity_short.sum()),
            "D3_blocked_open_slots": int(
                self.blocked_open_slots_long.sum() + self.blocked_open_slots_short.sum()
            ),
            "D4_cap_block_defects": int(
                self.gross_cap_block_below_target.sum()
                + self.name_cap_block_on_fresh_entry.sum()
                + self.net_cap_block_with_balanced_book.sum()
            ),
            "D5_ineligible_exit_within_hold_window": int(
                self.ineligible_exit_within_hold_window.sum()
            ),
        }
        return {
            "mean_net_excess_bps_per_day": (
                float(np.mean(finite_excess)) if finite_excess.size else 0.0
            ),
            "annualized_net_excess_sharpe": sharpe,
            "compounded_net_excess_vs_all_cash": float(
                self.nav[-1] / all_cash_terminal - 1.0
            ),
            "compounded_net_excess_terminal_settlement_haircut_scenario": float(
                self.settlement_haircut_scenario_nav[-1] / all_cash_terminal - 1.0
            ),
            "compounded_net_excess_unresolved_excluded": float(
                self.unresolved_excluded_nav[-1] / all_cash_terminal - 1.0
            ),
            "mean_gross_fraction_nav": mean_gross,
            "minimum_gross_fraction_nav": minimum_gross,
            "maximum_gross_fraction_nav": maximum_gross,
            "mean_turnover_fraction_nav": mean_turnover,
            "borrow_source": self.borrow_source,
            "short_notional_weighted_borrow_rate": (
                float(
                    np.sum(
                        self.held_short_weighted_annual_borrow_rate
                        * self.held_short_notional_at_open
                    )
                    / np.sum(self.held_short_notional_at_open)
                )
                if np.sum(self.held_short_notional_at_open) > 0.0
                else 0.0
            ),
            "imputed_rate_share_of_short_notional": (
                float(
                    np.sum(self.held_short_imputed_notional_at_open)
                    / np.sum(self.held_short_notional_at_open)
                )
                if np.sum(self.held_short_notional_at_open) > 0.0
                else 0.0
            ),
            "placeholder_rate_share_of_short_notional": (
                float(
                    np.sum(self.held_short_placeholder_notional_at_open)
                    / np.sum(self.held_short_notional_at_open)
                )
                if np.sum(self.held_short_notional_at_open) > 0.0
                else 0.0
            ),
            "placeholder_priced_session_count": int(
                (self.held_short_placeholder_notional_at_open > 0.0).sum()
            ),
            "excluded_short_entry_candidate_count": int(
                self.excluded_short_entry_candidate_count.sum()
            ),
            "short_side_unshortable_candidate_share": (
                float(
                    self.excluded_short_entry_candidate_count.sum()
                    / (
                        self.excluded_short_entry_candidate_count.sum()
                        + self.band_candidates_short.sum()
                    )
                )
                if (
                    self.excluded_short_entry_candidate_count.sum()
                    + self.band_candidates_short.sum()
                )
                > 0
                else 0.0
            ),
            "mean_unresolved_stale_inventory_fraction_nav": float(
                np.mean(self.unresolved_stale_inventory_fraction_nav)
            ),
            "maximum_unresolved_stale_inventory_fraction_nav": float(
                np.max(self.unresolved_stale_inventory_fraction_nav)
            ),
            "mean_unresolved_claim_inventory_fraction_nav": float(
                np.mean(self.unresolved_claim_inventory_fraction_nav)
            ),
            "maximum_unresolved_claim_inventory_fraction_nav": float(
                np.max(self.unresolved_claim_inventory_fraction_nav)
            ),
            "mean_stale_mark_inventory_fraction_nav": float(
                np.mean(self.stale_mark_inventory_fraction_nav)
            ),
            "maximum_stale_mark_inventory_fraction_nav": float(
                np.max(self.stale_mark_inventory_fraction_nav)
            ),
            "average_holding_sessions_approximation": (
                2.0 * mean_gross / mean_turnover if mean_turnover > 0 else 0.0
            ),
            "stale_mark_name_days": int(self.stale_mark_name_days.sum()),
            "same_day_action_sessions": int(self.same_day_action_sessions.sum()),
            "unresolved_action_name_days": int(self.unresolved_action_name_days.sum()),
            "valuation_scenario_count": int(self.valuation_scenario_count.sum()),
            "terminal_settlement_convention": TERMINAL_SETTLEMENT_CONVENTION,
            "ineligible_hold_sessions": self.ineligible_hold_sessions,
            "settlement_grace_sessions": self.settlement_grace_sessions,
            "settlement_haircut": self.settlement_haircut,
            "terminal_settlement_count": int(self.terminal_settlement_count.sum()),
            "terminal_settlement_notional": float(
                self.terminal_settlement_notional.sum()
            ),
            "terminal_settlement_notional_fraction_nav": float(
                self.terminal_settlement_notional_fraction_nav.sum()
            ),
            "settled_then_printed_count": int(self.settled_then_printed_count.sum()),
            "gross_shortfall_decomposition": decomposition,
            "entry_defect_signatures": defect_signatures,
            "exit_instructions_by_cause": {
                "ineligible_hold_exhausted": int(
                    self.exit_instructions_ineligible_hold_exhausted.sum()
                ),
                "settlement_grace": int(self.exit_instructions_settlement_grace.sum()),
                "rank_out_of_retention": int(
                    self.exit_instructions_rank_out_of_retention.sum()
                ),
                "terminal": int(self.exit_instructions_terminal.sum()),
            },
            "ineligible_held_sessions_by_cause": {
                "score_valid_false": int(
                    self.ineligible_held_sessions_score_invalid.sum()
                ),
                "membership_false": int(
                    self.ineligible_held_sessions_membership_invalid.sum()
                ),
                "score_nonfinite": int(
                    self.ineligible_held_sessions_score_nonfinite.sum()
                ),
            },
            "ineligible_exit_reeligible_within_10_sessions_share": (
                self.ineligible_exit_reeligible_within_10_sessions_share
            ),
            "terminal_settlement_economics_unresolved": bool(
                self.terminal_settlement_notional_fraction_nav.sum()
                > self.settlement_economics_unresolved_fraction_nav
            ),
            "settlement_economics_unresolved_fraction_nav": (
                self.settlement_economics_unresolved_fraction_nav
            ),
            "cancelled_entry_count": int(self.cancelled_entry_count.sum()),
            "zero_entry_days_by_cause": {
                "small_universe": int(self.zero_entry_small_universe.sum()),
                "gross_cap": int(self.zero_entry_gross_cap.sum()),
                "net_cap": int(self.zero_entry_net_cap.sum()),
                "name_cap": int(self.zero_entry_name_cap.sum()),
                "missing_reference": int(self.zero_entry_no_reference.sum()),
            },
            "blocked_entry_candidates_by_cause": {
                "gross_cap": int(self.blocked_entry_gross_cap_count.sum()),
                "net_cap": int(self.blocked_entry_net_cap_count.sum()),
                "name_cap": int(self.blocked_entry_name_cap_count.sum()),
                "missing_reference": int(self.blocked_entry_no_reference_count.sum()),
            },
            "risk_trim_days_by_type": {
                "gross": int((self.risk_trim_gross_notional > 0.0).sum()),
                "net": int((self.risk_trim_net_notional > 0.0).sum()),
                "name": int((self.risk_trim_name_notional > 0.0).sum()),
            },
            "risk_trim_notional_by_type": {
                "gross": float(self.risk_trim_gross_notional.sum()),
                "net": float(self.risk_trim_net_notional.sum()),
                "name": float(self.risk_trim_name_notional.sum()),
            },
            "intended_entry_count": len(entry_orders),
            "mean_daily_exits_per_side": float(
                np.mean(self.submitted_exit_count) / 2.0
            ),
            "same_close_replacements": int(self.same_close_replacement_count.sum()),
            "intended_entry_fill_rate": (
                filled_entry_notional / intended_entry_notional
                if intended_entry_notional > 0.0
                else 0.0
            ),
            "mean_pending_exit_age_sessions": (
                float(np.mean(pending_exit_age)) if pending_exit_age.size else 0.0
            ),
            "cancellations_by_reason": cancellation_reasons,
            "intended_order_count": len(self.intended_orders),
            "fill_count": len(self.fills),
            "unresolved_inventory_count": self.unresolved_inventory_count,
            "unresolved_inventory_notional": self.unresolved_inventory_notional,
            "terminal_unresolved_reason_breakdown": (
                self.terminal_unresolved_reason_breakdown
            ),
            "terminal_no_print_dominates": (
                self.terminal_unresolved_reason_breakdown["no_terminal_print"][
                    "notional"
                ]
                > 0.5 * self.unresolved_inventory_notional
            ),
            "unresolved_receivable": self.unresolved_receivable,
            "unresolved_payable": self.unresolved_payable,
            "terminal_nav": float(self.nav[-1]),
            "terminal_nav_settlement_haircut_scenario": float(
                self.settlement_haircut_scenario_nav[-1]
            ),
            "terminal_nav_unresolved_excluded": float(self.unresolved_excluded_nav[-1]),
            "insolvent": self.insolvent,
            "insolvency_date": (
                self.insolvency_date.isoformat()
                if self.insolvency_date is not None
                else None
            ),
            "economics_unresolved": self.economics_unresolved,
            "terminal_boundary_unpriced_inventory_notional": self.terminal_boundary_unpriced_inventory_notional,
            "terminal_hedge_last_mark_settlement_notional": self.terminal_hedge_last_mark_settlement_notional,
            "mark_mode": "raw",
            "share_sizing_mode": self.share_sizing_mode,
            "volatility_balanced_entries": self.volatility_balanced_entries,
            "beta_hedge": self.beta_hedge,
            "mean_gross_fraction_nav_including_hedge": float(
                np.mean(self.gross_fraction_nav_including_hedge)
            ),
            "mean_absolute_net_fraction_nav_including_hedge": float(
                np.mean(np.abs(self.net_fraction_nav_including_hedge))
            ),
            "terminal_net_notional_including_hedge": float(
                self.net_notional_including_hedge[-1]
            ),
            "hedge_notional_cap_nav": self.hedge_notional_cap_nav,
            "hedge_capped_session_count": int(self.hedge_capped.sum()),
            "hedge_beta_fallback_sessions": int(
                self.hedge_beta_fallback_sessions.sum()
            ),
            "maximum_absolute_hedge_fraction_nav": float(
                np.max(
                    np.divide(
                        np.abs(self.hedge_signed_notional),
                        self.nav,
                        out=np.zeros_like(self.hedge_signed_notional),
                        where=self.nav > 0.0,
                    )
                )
            ),
            "mean_absolute_ex_ante_beta_after_hedge": float(
                np.mean(np.abs(self.ex_ante_beta_after_hedge))
            ),
            "mean_volatility_quota_by_quintile": np.mean(
                self.volatility_quota, axis=0
            ).tolist(),
            "mean_entry_eligible_name_count": float(
                np.mean(self.entry_eligible_name_count)
            ),
            "mean_volatility_group_size_by_quintile": np.mean(
                self.volatility_group_size, axis=0
            ).tolist(),
            "minimum_volatility_group_size_by_quintile": np.min(
                self.volatility_group_size, axis=0
            ).tolist(),
            "mean_volatility_occupancy_long_by_quintile": np.mean(
                self.volatility_occupancy_long, axis=0
            ).tolist(),
            "mean_volatility_occupancy_short_by_quintile": np.mean(
                self.volatility_occupancy_short, axis=0
            ).tolist(),
            "mean_absolute_volatility_occupancy_deviation_long": float(
                np.mean(np.abs(self.volatility_occupancy_long - self.volatility_quota))
            ),
            "mean_absolute_volatility_occupancy_deviation_short": float(
                np.mean(np.abs(self.volatility_occupancy_short - self.volatility_quota))
            ),
            "spilled_entries_long_by_quintile": np.sum(
                self.volatility_spilled_entries_long, axis=0
            ).tolist(),
            "spilled_entries_short_by_quintile": np.sum(
                self.volatility_spilled_entries_short, axis=0
            ).tolist(),
            "terminal_hedge_signed_notional": float(self.hedge_signed_notional[-1]),
        }


def _validate_inputs(
    dates: Sequence[date],
    scores: NDArray[np.floating],
    score_mask: NDArray[np.bool_],
    active: NDArray[np.bool_],
    raw_close: NDArray[np.floating],
    cdi_returns: NDArray[np.floating],
    *,
    fill_fraction: NDArray[np.floating] | None,
    action_terms: AlignedActionTerms,
    action_payment_session: NDArray[np.integer],
    security_ids: Sequence[str] | None,
    initial_reference_price: NDArray[np.floating] | None,
    annual_borrow_rate_by_name: NDArray[np.floating] | None,
    borrow_rate_imputed: NDArray[np.bool_] | None,
    borrow_rate_placeholder: NDArray[np.bool_] | None,
    shortable: NDArray[np.bool_] | None,
    selection_volatility: NDArray[np.floating] | None,
    hedge_beta: NDArray[np.floating] | None,
    hedge_beta_valid: NDArray[np.bool_] | None,
    hedge_beta_history: tuple[NDArray[np.floating], NDArray[np.bool_]] | None,
    hedge_close: NDArray[np.floating] | None,
    hedge_annual_borrow_rate: NDArray[np.floating] | None,
    borrow_source: BorrowSource,
    volatility_balanced_entries: bool,
    beta_hedge: bool,
) -> _ValidatedInputs:
    if not isinstance(action_terms, AlignedActionTerms):
        raise TypeError("action_terms must be explicit AlignedActionTerms")
    if action_payment_session is None:
        raise TypeError("action_payment_session is required")
    date_values = tuple(dates)
    if not date_values or any(
        left >= right for left, right in zip(date_values, date_values[1:], strict=False)
    ):
        raise ValueError("ledger dates must be nonempty and strictly chronological")
    score = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(score_mask, dtype=np.bool_)
    membership = np.asarray(active, dtype=np.bool_)
    close = np.asarray(raw_close, dtype=np.float64)
    cdi = np.asarray(cdi_returns, dtype=np.float64)
    matrix_shape = (len(date_values), score.shape[1] if score.ndim == 2 else 0)
    if score.ndim != 2 or any(
        value.shape != matrix_shape for value in (mask, membership, close)
    ):
        raise ValueError("ledger name panels must align [date, name]")
    if cdi.shape != (len(date_values),):
        raise ValueError("ledger CDI return axis is misaligned")
    if not np.isfinite(cdi).all() or (cdi <= -1.0).any():
        raise ValueError("ledger CDI returns must be finite and greater than -1")

    if borrow_source != "uniform":
        if annual_borrow_rate_by_name is None or shortable is None:
            raise ValueError("archive borrow requires rate and shortable panels")
        borrow_rate = np.asarray(annual_borrow_rate_by_name, dtype=np.float64)
        shortable_mask = np.asarray(shortable, dtype=np.bool_)
        if borrow_rate.shape != matrix_shape or shortable_mask.shape != matrix_shape:
            raise ValueError("lending borrow panels must align [date, name]")
        if (
            np.isinf(borrow_rate).any()
            or (borrow_rate[np.isfinite(borrow_rate)] < 0).any()
        ):
            raise ValueError("lending borrow rates must be non-negative or missing")
        if borrow_rate_imputed is None:
            raise ValueError("archive borrow requires an imputed-rate mask")
        imputed = np.asarray(borrow_rate_imputed)
        if imputed.shape != matrix_shape or imputed.dtype != np.bool_:
            raise ValueError("imputed-rate mask must be Boolean and align names")
        if np.any(imputed & ~np.isfinite(borrow_rate)):
            raise ValueError("an imputed borrow rate must be finite")
        if borrow_rate_placeholder is None:
            raise ValueError("archive borrow requires a placeholder-rate mask")
        placeholder = np.asarray(borrow_rate_placeholder)
        if placeholder.shape != matrix_shape or placeholder.dtype != np.bool_:
            raise ValueError("placeholder-rate mask must be Boolean and align names")
        if np.any(placeholder & ~np.isfinite(borrow_rate)):
            raise ValueError("a placeholder borrow rate must be finite")
        if np.any(imputed & placeholder):
            raise ValueError("borrow rates cannot be both imputed and placeholder")
    else:
        borrow_rate = np.full(matrix_shape, np.nan, dtype=np.float64)
        shortable_mask = np.ones(matrix_shape, dtype=np.bool_)
        imputed = np.zeros(matrix_shape, dtype=np.bool_)
        placeholder = np.zeros(matrix_shape, dtype=np.bool_)

    if volatility_balanced_entries:
        if selection_volatility is None:
            raise ValueError("volatility-balanced entries require a volatility panel")
        volatility = np.asarray(selection_volatility, dtype=np.float64)
        if volatility.shape != matrix_shape or np.isinf(volatility).any():
            raise ValueError(
                "selection volatility must align [date, name] without infinity"
            )
    else:
        volatility = np.full(matrix_shape, np.nan, dtype=np.float64)

    if beta_hedge:
        if hedge_beta is None or hedge_beta_valid is None or hedge_close is None:
            raise ValueError(
                "beta hedge requires economic hedge_beta, validity and BOVA11 closes"
            )
        beta, beta_fallback = resolve_hedge_beta(
            hedge_beta, hedge_beta_valid, history=hedge_beta_history
        )
        hedge = np.asarray(hedge_close, dtype=np.float64)
        if beta.shape != matrix_shape:
            raise ValueError("hedge_beta must align [date, name]")
        if hedge.shape != (len(date_values),) or np.isinf(hedge).any():
            raise ValueError("BOVA11 close must align the date axis without infinity")
        if (hedge[np.isfinite(hedge)] <= 0.0).any():
            raise ValueError("BOVA11 close must be positive when present")
        if hedge_annual_borrow_rate is None:
            hedge_borrow = np.full(len(date_values), np.nan, dtype=np.float64)
        else:
            hedge_borrow = np.asarray(hedge_annual_borrow_rate, dtype=np.float64)
            if (
                hedge_borrow.shape != (len(date_values),)
                or np.isinf(hedge_borrow).any()
                or (hedge_borrow[np.isfinite(hedge_borrow)] < 0.0).any()
            ):
                raise ValueError(
                    "BOVA11 borrow rate must align the date axis and be non-negative"
                )
    else:
        beta = np.full(matrix_shape, np.nan, dtype=np.float64)
        beta_fallback = np.zeros(matrix_shape, dtype=np.bool_)
        hedge = np.full(len(date_values), np.nan, dtype=np.float64)
        hedge_borrow = np.full(len(date_values), np.nan, dtype=np.float64)

    if np.isinf(close).any():
        raise ValueError("raw_close may be missing but cannot contain infinities")

    if fill_fraction is None:
        fill = np.ones(matrix_shape, dtype=np.float64)
    else:
        fill = np.asarray(fill_fraction, dtype=np.float64)
        if fill.shape != matrix_shape:
            raise ValueError("fill_fraction must align [date, name]")
        if not np.isfinite(fill).all() or (fill < 0).any() or (fill > 1).any():
            raise ValueError("fill fractions must be finite and in [0, 1]")

    name_count = matrix_shape[1]
    if security_ids is None:
        securities = tuple(str(index) for index in range(name_count))
    else:
        securities = tuple(security_ids)
        if len(securities) != name_count or len(set(securities)) != name_count:
            raise ValueError("security_ids must be unique and align the name axis")

    if initial_reference_price is None:
        initial_reference = np.full(name_count, np.nan, dtype=np.float64)
    else:
        initial_reference = np.asarray(initial_reference_price, dtype=np.float64).copy()
        if initial_reference.shape != (name_count,):
            raise ValueError("initial_reference_price must align the name axis")
        invalid_reference = ~np.isnan(initial_reference) & (
            ~np.isfinite(initial_reference) | (initial_reference <= 0.0)
        )
        if invalid_reference.any():
            raise ValueError(
                "initial_reference_price must contain positive finite values or NaN"
            )

    q = np.asarray(action_terms.shares_per_prior_share, dtype=np.float64)
    d = np.asarray(action_terms.cash_per_prior_share, dtype=np.float64)
    resolved = np.asarray(action_terms.session_resolved, dtype=np.bool_)
    has_action = np.asarray(action_terms.has_action, dtype=np.bool_)
    successor = (
        np.broadcast_to(np.arange(name_count, dtype=np.int64), matrix_shape).copy()
        if action_terms.successor_index is None
        else np.asarray(action_terms.successor_index, dtype=np.int64)
    )
    if any(
        value.shape != matrix_shape for value in (q, d, resolved, has_action, successor)
    ):
        raise ValueError("aligned action terms must align [date, name]")
    if (
        not np.isfinite(q).all()
        or not np.isfinite(d).all()
        or (q < 0).any()
        or (d < 0).any()
        or (successor < 0).any()
        or (successor >= name_count).any()
    ):
        raise ValueError("aligned action terms contain invalid values")
    identity_successor = np.broadcast_to(
        np.arange(name_count, dtype=np.int64), matrix_shape
    )
    silent_terms = ~has_action & (
        (q != 1.0) | (d != 0.0) | (successor != identity_successor)
    )
    if silent_terms.any():
        raise ValueError("economic action terms require has_action=true")

    raw_payment = np.asarray(action_payment_session)
    if raw_payment.shape != matrix_shape or not np.issubdtype(
        raw_payment.dtype, np.integer
    ):
        raise TypeError("action payment sessions must be an integer [date, name] panel")
    payment = raw_payment.astype(np.int64, copy=False)
    event_session = np.broadcast_to(
        np.arange(len(date_values), dtype=np.int64)[:, None], matrix_shape
    )
    if (payment < -1).any() or ((payment >= 0) & (payment < event_session)).any():
        raise ValueError("action payment sessions cannot precede their action session")
    invalid_payment_binding = (payment >= 0) & (~has_action | (d == 0.0))
    if invalid_payment_binding.any():
        raise ValueError(
            "payment sessions require a cash-bearing action on the same cell"
        )
    return _ValidatedInputs(
        dates=date_values,
        score=score,
        score_valid=mask,
        membership=membership,
        raw_close=close,
        cdi=cdi,
        fill_fraction=fill,
        action_q=q,
        action_d=d,
        action_resolved=resolved,
        has_action=has_action,
        action_successor=successor,
        action_payment_session=payment,
        securities=securities,
        initial_reference_price=initial_reference,
        annual_borrow_rate_by_name=borrow_rate,
        borrow_rate_imputed=imputed,
        borrow_rate_placeholder=placeholder,
        shortable=shortable_mask,
        selection_volatility=volatility,
        hedge_beta=beta,
        hedge_beta_fallback=beta_fallback,
        hedge_close=hedge,
        hedge_annual_borrow_rate=hedge_borrow,
    )


def _equity(
    free_cash: float,
    restricted_by_name: NDArray[np.float64],
    shares: NDArray[np.float64],
    marks: NDArray[np.float64],
    receivables: NDArray[np.float64],
    payables: NDArray[np.float64],
) -> float:
    held = shares != 0.0
    if held.any() and not np.isfinite(marks[held]).all():
        raise RuntimeError("held inventory has no finite raw-price mark")
    return float(
        free_cash
        + restricted_by_name.sum()
        + np.sum(shares[held] * marks[held], dtype=np.float64)
        + receivables.sum()
        - payables.sum()
    )


def _risk(signed_values: NDArray[np.float64], nav: float) -> tuple[float, float, float]:
    if nav <= 0:
        return np.inf, np.inf, np.inf
    weights = signed_values / nav
    return (
        float(np.abs(weights).sum()),
        float(weights.sum()),
        float(np.max(np.abs(weights), initial=0.0)),
    )


def _equal_count_groups(
    values: NDArray[np.float64], valid: NDArray[np.bool_], group_count: int
) -> NDArray[np.int64]:
    """Assign stable equal-count groups without admitting missing values."""

    names = np.flatnonzero(valid & np.isfinite(values))
    groups = np.full(values.shape, -1, dtype=np.int64)
    if names.size:
        order = names[np.argsort(values[names], kind="stable")]
        groups[order] = np.minimum(
            np.arange(order.size, dtype=np.int64) * group_count // order.size,
            group_count - 1,
        )
    return groups


def _balanced_quota(k_eff: int, group_count: int) -> NDArray[np.int64]:
    quota = np.full(group_count, k_eff // group_count, dtype=np.int64)
    centre = (group_count - 1) / 2.0
    middle_first = sorted(
        range(group_count), key=lambda group: (abs(group - centre), group)
    )
    for group in middle_first[: k_eff % group_count]:
        quota[group] += 1
    return quota


def _scaled_group_bands(
    *,
    k_eff: int,
    buffer: int,
    group_sizes: NDArray[np.integer],
    threshold_multiple: int,
    capacity_buffer: int | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Scale quota and retention widths together in undersized quintiles."""

    sizes = np.asarray(group_sizes, dtype=np.int64)
    quota = _balanced_quota(k_eff, len(sizes))
    retention_buffer = _balanced_quota(buffer, len(sizes))
    capacity = _balanced_quota(
        buffer if capacity_buffer is None else capacity_buffer, len(sizes)
    )
    for group, size in enumerate(sizes):
        width = int(quota[group] + capacity[group])
        if width == 0 or int(size) >= threshold_multiple * width:
            continue
        scale = float(size) / float(threshold_multiple * width)
        quota[group] = int(np.floor(quota[group] * scale))
        retention_buffer[group] = int(np.floor(retention_buffer[group] * scale))
    if capacity_buffer is not None:
        # Buffer ablations keep the reference quota/scaling fixed. Retention
        # still cannot overlap the opposite side in an undersized stratum.
        retention_buffer = np.minimum(retention_buffer, sizes // 2 - quota)
    return quota, retention_buffer


def _book_fill(
    *,
    name: int,
    side: OrderSide,
    quantity: float,
    price: float,
    shares: NDArray[np.float64],
    marks: NDArray[np.float64],
    restricted_by_name: NDArray[np.float64],
    free_cash: float,
    cost_rate: float,
) -> tuple[float, float]:
    old_shares = float(shares[name])
    remaining = quantity
    if side == "buy" and old_shares < 0.0:
        cover = min(remaining, -old_shares)
        release = restricted_by_name[name] * cover / -old_shares
        restricted_by_name[name] -= release
        free_cash += release - cover * price
        shares[name] += cover
        remaining -= cover
    if side == "sell" and old_shares > 0.0:
        sale = min(remaining, old_shares)
        free_cash += sale * price
        shares[name] -= sale
        remaining -= sale
    if remaining > 0.0:
        if side == "buy":
            free_cash -= remaining * price
            shares[name] += remaining
        else:
            restricted_by_name[name] += remaining * price
            shares[name] -= remaining
    notional = quantity * price
    free_cash -= cost_rate * notional
    marks[name] = price if shares[name] != 0.0 else np.nan
    return free_cash, notional


def simulate_stateful_ledger(
    *,
    dates: Sequence[date],
    scores: NDArray[np.floating],
    score_mask: NDArray[np.bool_],
    active: NDArray[np.bool_],
    raw_close: NDArray[np.floating],
    action_terms: AlignedActionTerms,
    action_payment_session: NDArray[np.integer],
    cdi_returns: NDArray[np.floating],
    config: LedgerConfig = LedgerConfig(),
    fill_fraction: NDArray[np.floating] | None = None,
    security_ids: Sequence[str] | None = None,
    initial_reference_price: NDArray[np.floating] | None = None,
    annual_borrow_rate_by_name: NDArray[np.floating] | None = None,
    borrow_rate_imputed: NDArray[np.bool_] | None = None,
    borrow_rate_placeholder: NDArray[np.bool_] | None = None,
    shortable: NDArray[np.bool_] | None = None,
    selection_volatility: NDArray[np.floating] | None = None,
    hedge_beta: NDArray[np.floating] | None = None,
    hedge_beta_valid: NDArray[np.bool_] | None = None,
    hedge_beta_history: tuple[NDArray[np.floating], NDArray[np.bool_]] | None = None,
    hedge_close: NDArray[np.floating] | None = None,
    initial_hedge_reference_price: float = np.nan,
    initial_unresolved_action: NDArray[np.bool_] | None = None,
    hedge_annual_borrow_rate: NDArray[np.floating] | None = None,
    entry_sizing_volatility: NDArray[np.floating] | None = None,
    capacity_buffer_per_side: int | None = None,
) -> StatefulLedgerResult:
    """Run causal close-proxy orders, fills, and a raw signed-share ledger.

    Optional entry volatility applies inverse-volatility sizing to each side's
    new-entry cohort; existing inventory is not rebalanced. It must already be
    available at the decision. Capacity can be bound to a reference buffer for
    a retention-only ablation without changing entry quotas.
    """

    inputs = _validate_inputs(
        dates,
        scores,
        score_mask,
        active,
        raw_close,
        cdi_returns,
        fill_fraction=fill_fraction,
        action_terms=action_terms,
        action_payment_session=action_payment_session,
        security_ids=security_ids,
        initial_reference_price=initial_reference_price,
        annual_borrow_rate_by_name=annual_borrow_rate_by_name,
        borrow_rate_imputed=borrow_rate_imputed,
        borrow_rate_placeholder=borrow_rate_placeholder,
        shortable=shortable,
        selection_volatility=selection_volatility,
        hedge_beta=hedge_beta,
        hedge_beta_valid=hedge_beta_valid,
        hedge_beta_history=hedge_beta_history,
        hedge_close=hedge_close,
        hedge_annual_borrow_rate=hedge_annual_borrow_rate,
        borrow_source=config.borrow_source,
        volatility_balanced_entries=config.volatility_balanced_entries,
        beta_hedge=config.beta_hedge,
    )
    day_count, name_count = inputs.score.shape
    sizing_volatility = (
        None
        if entry_sizing_volatility is None
        else np.asarray(entry_sizing_volatility, dtype=np.float64)
    )
    if sizing_volatility is not None and sizing_volatility.shape != inputs.score.shape:
        raise ValueError("entry sizing volatility must align with decision scores")
    initial_unresolved_action = (
        np.zeros(name_count, dtype=np.bool_)
        if initial_unresolved_action is None
        else np.asarray(initial_unresolved_action, dtype=np.bool_)
    )
    if initial_unresolved_action.shape != (name_count,):
        raise ValueError("initial action uncertainty must match the security axis")
    shares = np.zeros(name_count, dtype=np.float64)
    marks = np.full(name_count, np.nan, dtype=np.float64)
    hedge_shares = 0.0
    if not np.isnan(initial_hedge_reference_price) and (
        not np.isfinite(initial_hedge_reference_price)
        or initial_hedge_reference_price <= 0
    ):
        raise ValueError(
            "initial BOVA11 reference must be a positive prior close or missing"
        )
    hedge_mark = float(initial_hedge_reference_price)
    hedge_restricted_cash = 0.0
    # This is the last observed close strictly before the first decision.  It
    # lets the first evaluation session form the same causal order it would
    # form inside a longer continuous ledger, without reading that session's
    # later fill print.
    last_observed = inputs.initial_reference_price.copy()
    last_print_session = np.where(
        np.isfinite(inputs.initial_reference_price)
        & (inputs.initial_reference_price > 0.0),
        -1,
        -2,
    ).astype(np.int64)
    restricted_by_name = np.zeros(name_count, dtype=np.float64)
    receivable_by_name = np.zeros(name_count, dtype=np.float64)
    payable_by_name = np.zeros(name_count, dtype=np.float64)
    unresolved_action = np.zeros(name_count, dtype=np.bool_)
    explicit_unresolved_action = np.zeros(name_count, dtype=np.bool_)
    missing_sessions = np.zeros(name_count, dtype=np.int64)
    ineligible_streak = np.zeros(name_count, dtype=np.int64)
    entry_session = np.full(name_count, -1, dtype=np.int64)
    entry_cost_basis = np.zeros(name_count, dtype=np.float64)
    submission_nav = np.zeros(name_count, dtype=np.float64)
    scenario_seen = np.zeros(name_count, dtype=np.bool_)
    settled_names = np.zeros(name_count, dtype=np.bool_)
    settled_then_printed_seen = np.zeros(name_count, dtype=np.bool_)
    settlement_scenario_adjustment = 0.0
    free_cash = config.initial_capital_brl
    previous_nav = config.initial_capital_brl
    all_cash = config.initial_capital_brl
    pending_entries: dict[int, _PendingOrder] = {}
    pending_exits: dict[int, _PendingOrder] = {}
    pending_claims: list[_PendingClaim] = []
    orders: list[IntendedOrder] = []
    fills: list[Fill] = []
    cancellations: list[OrderCancellation] = []
    next_order_id = 0

    start_nav_rows: list[float] = []
    nav_rows: list[float] = []
    daily_rows: list[float] = []
    excess_rows: list[float] = []
    gross_pnl_rows: list[float] = []
    equity_gross_pnl_rows: list[float] = []
    hedge_gross_pnl_rows: list[float] = []
    interest_rows: list[float] = []
    free_cash_interest_rows: list[float] = []
    short_proceeds_interest_rows: list[float] = []
    short_proceeds_interest_base_rows: list[float] = []
    cost_rows: list[float] = []
    borrow_rows: list[float] = []
    equity_borrow_raw_rows: list[float] = []
    equity_borrow_fee_rows: list[float] = []
    cdi_benchmark_rows: list[float] = []
    held_short_borrow_rate_rows: list[float] = []
    held_short_notional_rows: list[float] = []
    held_short_imputed_notional_rows: list[float] = []
    held_short_placeholder_notional_rows: list[float] = []
    excluded_short_candidate_rows: list[int] = []
    gross_rows: list[float] = []
    turnover_rows: list[float] = []
    unresolved_stale_fraction_rows: list[float] = []
    unresolved_claim_fraction_rows: list[float] = []
    stale_mark_fraction_rows: list[float] = []
    stale_rows: list[int] = []
    unresolved_action_rows: list[int] = []
    same_day_action_rows: list[bool] = []
    scenario_count_rows: list[int] = []
    position_rows: list[NDArray[np.int8]] = []
    share_rows: list[NDArray[np.float64]] = []
    mark_rows: list[NDArray[np.float64]] = []
    free_cash_rows: list[float] = []
    restricted_rows: list[float] = []
    hedge_restricted_rows: list[float] = []
    receivable_rows: list[float] = []
    payable_rows: list[float] = []
    holding_value_rows: list[float] = []
    reconciliation_rows: list[float] = []
    all_cash_rows: list[float] = []
    haircut_rows: list[float] = []
    excluded_rows: list[float] = []
    pending_entry_rows: list[int] = []
    pending_exit_rows: list[int] = []
    cancelled_rows: list[int] = []
    blocked_reference_rows: list[int] = []
    blocked_gross_rows: list[int] = []
    blocked_net_rows: list[int] = []
    blocked_name_rows: list[int] = []
    submitted_entry_rows: list[int] = []
    submitted_exit_rows: list[int] = []
    same_close_replacement_rows: list[int] = []
    zero_entry_small_rows: list[bool] = []
    zero_entry_gross_rows: list[bool] = []
    zero_entry_net_rows: list[bool] = []
    zero_entry_name_rows: list[bool] = []
    zero_entry_reference_rows: list[bool] = []
    risk_trim_gross_rows: list[float] = []
    risk_trim_net_rows: list[float] = []
    risk_trim_name_rows: list[float] = []
    pending_exit_age_rows: list[float] = []
    retention_rows: list[int] = []
    small_universe_rows: list[bool] = []
    risk_breach_rows: list[bool] = []
    planned_gross_rows: list[float] = []
    planned_net_rows: list[float] = []
    planned_name_rows: list[float] = []
    age_rows: list[NDArray[np.int64]] = []
    settlement_count_rows: list[int] = []
    settlement_notional_rows: list[float] = []
    settlement_fraction_rows: list[float] = []
    settled_then_printed_rows: list[int] = []
    k_eff_rows: list[int] = []
    held_start_long_rows: list[int] = []
    held_start_short_rows: list[int] = []
    held_end_long_rows: list[int] = []
    held_end_short_rows: list[int] = []
    occupied_after_submission_long_rows: list[int] = []
    occupied_after_submission_short_rows: list[int] = []
    open_after_submission_long_rows: list[int] = []
    open_after_submission_short_rows: list[int] = []
    band_candidates_long_rows: list[int] = []
    band_candidates_short_rows: list[int] = []
    band_excluded_unresolved_long_rows: list[int] = []
    band_excluded_unresolved_short_rows: list[int] = []
    band_excluded_settled_long_rows: list[int] = []
    band_excluded_settled_short_rows: list[int] = []
    band_without_prior_print_long_rows: list[int] = []
    band_without_prior_print_short_rows: list[int] = []
    band_exhausted_long_rows: list[int] = []
    band_exhausted_short_rows: list[int] = []
    blocked_open_long_rows: list[int] = []
    blocked_open_short_rows: list[int] = []
    pending_end_long_rows: list[int] = []
    pending_end_short_rows: list[int] = []
    pending_without_print_long_rows: list[int] = []
    pending_without_print_short_rows: list[int] = []
    exit_fill_long_rows: list[int] = []
    exit_fill_short_rows: list[int] = []
    exit_ineligible_hold_exhausted_rows: list[int] = []
    exit_settlement_rows: list[int] = []
    exit_rank_rows: list[int] = []
    exit_terminal_rows: list[int] = []
    exit_cause_rows: list[NDArray[np.int8]] = []
    exit_side_rows: list[NDArray[np.int8]] = []
    ineligible_streak_rows: list[NDArray[np.int64]] = []
    ineligible_score_invalid_rows: list[int] = []
    ineligible_membership_invalid_rows: list[int] = []
    ineligible_score_nonfinite_rows: list[int] = []
    ineligible_exit_within_hold_rows: list[int] = []
    net_cap_balanced_rows: list[int] = []
    gross_cap_below_target_rows: list[int] = []
    name_cap_fresh_rows: list[int] = []
    pending_printed_unfilled_rows: list[int] = []
    entry_fill_short_rows: list[int] = []
    entry_cost_basis_rows: list[NDArray[np.float64]] = []
    submission_nav_rows: list[NDArray[np.float64]] = []
    shortfall_small_rows: list[float] = []
    shortfall_pending_rows: list[float] = []
    shortfall_band_exhausted_rows: list[float] = []
    shortfall_blocked_rows: list[float] = []
    shortfall_exit_gap_rows: list[float] = []
    shortfall_other_rows: list[float] = []
    shortfall_sizing_fill_rows: list[float] = []
    shortfall_sizing_mark_rows: list[float] = []
    shortfall_sizing_nav_rows: list[float] = []
    volatility_quota_rows: list[NDArray[np.int64]] = []
    volatility_group_size_rows: list[NDArray[np.int64]] = []
    entry_eligible_name_count_rows: list[int] = []
    volatility_occupancy_long_rows: list[NDArray[np.int64]] = []
    volatility_occupancy_short_rows: list[NDArray[np.int64]] = []
    volatility_spilled_long_rows: list[NDArray[np.int64]] = []
    volatility_spilled_short_rows: list[NDArray[np.int64]] = []
    hedge_share_rows: list[float] = []
    hedge_mark_rows: list[float] = []
    hedge_notional_rows: list[float] = []
    hedge_unconstrained_target_rows: list[float] = []
    hedge_target_rows: list[float] = []
    hedge_beta_fallback_rows: list[bool] = []
    hedge_capped_rows: list[bool] = []
    hedge_turnover_rows: list[float] = []
    hedge_cost_rows: list[float] = []
    hedge_borrow_rows: list[float] = []
    ex_ante_beta_before_rows: list[float] = []
    ex_ante_beta_after_rows: list[float] = []
    gross_including_hedge_rows: list[float] = []
    net_notional_including_hedge_rows: list[float] = []
    net_including_hedge_rows: list[float] = []
    insolvent = False
    insolvency_date: date | None = None
    action_uncertainty_seen = False
    terminal_printed = np.zeros(name_count, dtype=np.bool_)
    terminal_boundary_unpriced_inventory_notional = 0.0
    terminal_hedge_last_mark_settlement_notional = 0.0
    terminal_prior_pending_exit = np.zeros(name_count, dtype=np.bool_)

    def submit_order(
        day: int,
        name: int,
        side: OrderSide,
        size: float,
        reference_price: float,
        purpose: OrderPurpose,
        expiry_session: int | None,
        *,
        position_fraction: bool = False,
    ) -> _PendingOrder:
        nonlocal next_order_id
        order = IntendedOrder(
            order_id=f"order-{next_order_id:08d}",
            security="BRBOVACTF003" if purpose == "hedge" else inputs.securities[name],
            security_index=name,
            decision_date=inputs.dates[day],
            decision_session=day,
            side=side,
            planned_notional=None if position_fraction else size,
            position_fraction=size if position_fraction else None,
            reference_price=reference_price,
            order_type="close_proxy",
            limit_price=None,
            expiry_date=(
                inputs.dates[expiry_session]
                if expiry_session is not None and expiry_session < day_count
                else None
            ),
            expiry_session=expiry_session,
            purpose=purpose,
        )
        next_order_id += 1
        orders.append(order)
        return _PendingOrder(order=order, remaining_size=size)

    def cancel_entry(name: int, day: int, reason: CancellationReason) -> bool:
        pending = pending_entries.pop(name, None)
        if pending is None:
            return False
        cancellations.append(pending.cancellation(day, inputs.dates[day], reason))
        if shares[name] == 0.0:
            entry_cost_basis[name] = 0.0
            submission_nav[name] = 0.0
        return True

    for day in range(day_count):
        cancelled_today = 0
        if day == day_count - 1:
            prior_pending_names = [
                name
                for name, pending in pending_exits.items()
                if pending.order.decision_session < day
            ]
            if prior_pending_names:
                terminal_prior_pending_exit[prior_pending_names] = True
        start_nav = previous_nav
        if start_nav <= 0.0:
            raise RuntimeError("insolvent ledger attempted to continue")
        start_identity = _equity(
            free_cash,
            restricted_by_name,
            shares,
            marks,
            receivable_by_name,
            payable_by_name,
        )
        if hedge_shares != 0.0:
            if not np.isfinite(hedge_mark):
                raise RuntimeError("open BOVA11 hedge has no finite mark")
            start_identity += hedge_restricted_cash + hedge_shares * hedge_mark
        if not np.isclose(start_identity, start_nav, rtol=1e-12, atol=1e-12):
            raise RuntimeError("opening ledger identity does not reconcile")
        held_start_long_rows.append(int((shares > 0.0).sum()))
        held_start_short_rows.append(int((shares < 0.0).sum()))
        action_exposed = shares != 0.0
        for name in pending_entries.keys() | pending_exits.keys():
            action_exposed[name] = True

        # Only completed sessions can inform entry uncertainty. Current action
        # terms (including the inferred event flag) are accounting-only below.
        decision_unresolved = (
            ~inputs.action_resolved[day - 1] if day else initial_unresolved_action
        )
        for name in tuple(pending_entries):
            if decision_unresolved[name]:
                cancelled_today += int(
                    cancel_entry(name, day, "prior_action_unresolved")
                )

        eligible = (
            inputs.score_valid[day]
            & inputs.membership[day]
            & np.isfinite(inputs.score[day])
        )
        entry_eligible = eligible.copy()
        if config.volatility_balanced_entries:
            entry_eligible &= np.isfinite(inputs.selection_volatility[day])
        eligible_names = np.flatnonzero(eligible)
        order = (
            eligible_names[np.argsort(inputs.score[day, eligible_names], kind="stable")]
            if eligible_names.size
            else eligible_names
        )
        ranks = np.full(name_count, -1, dtype=np.int64)
        ranks[order] = np.arange(order.size)
        entry_names = np.flatnonzero(entry_eligible)
        entry_order = (
            entry_names[np.argsort(inputs.score[day, entry_names], kind="stable")]
            if entry_names.size
            else entry_names
        )
        k_eff = min(config.k_per_side, len(entry_order) // 2)
        k_eff_rows.append(k_eff)
        small_universe = k_eff == 0
        small_universe_rows.append(small_universe)
        if config.volatility_balanced_entries:
            group_eligible = eligible & np.isfinite(inputs.selection_volatility[day])
            volatility_groups = _equal_count_groups(
                inputs.selection_volatility[day],
                group_eligible,
                config.volatility_group_count,
            )
            group_sizes = np.bincount(
                volatility_groups[volatility_groups >= 0],
                minlength=config.volatility_group_count,
            )[: config.volatility_group_count]
            volatility_quota, volatility_buffer = _scaled_group_bands(
                k_eff=k_eff,
                buffer=config.buffer_per_side,
                group_sizes=group_sizes,
                threshold_multiple=config.small_stratum_scaling_threshold_multiple,
                capacity_buffer=capacity_buffer_per_side,
            )
            # An undersized stratum scales both its quota and buffer.  The
            # resulting quota sum is therefore the contractual side size for
            # this session; filling back to the unscaled k would undo the
            # small-stratum rule through the global spill pass.
            k_eff = int(volatility_quota.sum())
            k_eff_rows[-1] = k_eff
            small_universe = k_eff == 0
            small_universe_rows[-1] = small_universe
            group_orders: dict[int, NDArray[np.int64]] = {}
            long_retention = np.zeros(name_count, dtype=np.bool_)
            short_retention = np.zeros(name_count, dtype=np.bool_)
            for group in range(config.volatility_group_count):
                names = np.flatnonzero(volatility_groups == group)
                group_order = names[np.argsort(inputs.score[day, names], kind="stable")]
                group_orders[group] = group_order
                width = int(volatility_quota[group] + volatility_buffer[group])
                if width:
                    short_retention[group_order[:width]] = True
                    long_retention[group_order[-width:]] = True
            if np.any(long_retention & short_retention):
                raise RuntimeError("within-quintile long and short bands overlap")
            retention = int(np.sum(volatility_quota + volatility_buffer))
        else:
            volatility_groups = np.full(name_count, -1, dtype=np.int64)
            volatility_quota = np.zeros(config.volatility_group_count, dtype=np.int64)
            volatility_buffer = np.zeros(config.volatility_group_count, dtype=np.int64)
            group_sizes = np.zeros(config.volatility_group_count, dtype=np.int64)
            group_orders = {}
            long_retention = np.zeros(name_count, dtype=np.bool_)
            short_retention = np.zeros(name_count, dtype=np.bool_)
            retention = min(
                config.k_per_side + config.buffer_per_side,
                len(order) // 2,
            )
            if retention:
                long_retention[order[-retention:]] = True
                short_retention[order[:retention]] = True
        retention_rows.append(retention)

        if config.cancel_pending_outside_retention:
            for name, pending in tuple(pending_entries.items()):
                retained = (
                    long_retention[name]
                    if pending.order.side == "buy"
                    else short_retention[name]
                )
                if not retained:
                    cancelled_today += int(cancel_entry(name, day, "band_exit"))

        signed_values = np.zeros(name_count, dtype=np.float64)
        held = shares != 0.0
        ineligible_streak[~held | eligible] = 0
        ineligible_streak[held & ~eligible] += 1
        ineligible_score_invalid_rows.append(
            int((held & ~inputs.score_valid[day]).sum())
        )
        ineligible_membership_invalid_rows.append(
            int((held & ~inputs.membership[day]).sum())
        )
        ineligible_score_nonfinite_rows.append(
            int((held & ~np.isfinite(inputs.score[day])).sum())
        )
        signed_values[held] = shares[held] * marks[held]
        actual_gross, actual_net, actual_name = _risk(signed_values, start_nav)
        risk_breach = (
            actual_gross > config.planned_gross_cap + 1e-12
            or abs(actual_net) > config.planned_absolute_net_cap + 1e-12
            or actual_name > config.planned_name_weight_cap + 1e-12
        )

        exit_required: set[int] = set()
        exit_cause_by_name: dict[int, ExitInstructionCause] = {}
        for name in np.flatnonzero(held):
            if not eligible[name]:
                kept = (
                    retention > 0
                    and missing_sessions[name] < config.settlement_grace_sessions
                    and ineligible_streak[name] <= config.ineligible_hold_sessions
                )
            else:
                kept = (
                    retention > 0
                    and missing_sessions[name] < config.settlement_grace_sessions
                    and (
                        (
                            long_retention[name]
                            if shares[name] > 0.0
                            else short_retention[name]
                        )
                        if config.volatility_balanced_entries
                        else (
                            ranks[name] >= len(order) - retention
                            if shares[name] > 0.0
                            else ranks[name] < retention
                        )
                    )
                )
            if not kept:
                integer_name = int(name)
                exit_required.add(integer_name)
                if missing_sessions[name] >= config.settlement_grace_sessions:
                    exit_cause_by_name[integer_name] = "settlement_grace"
                elif not eligible[name]:
                    exit_cause_by_name[integer_name] = "ineligible_hold_exhausted"
                else:
                    exit_cause_by_name[integer_name] = "rank_out_of_retention"
        if day == day_count - 1:
            exit_required.update(int(name) for name in np.flatnonzero(held))
            for name, pending in tuple(pending_exits.items()):
                if pending.remaining_size < 1.0 - 1e-12:
                    cancellations.append(
                        pending.cancellation(day, inputs.dates[day], "evaluation_end")
                    )
                    del pending_exits[name]

        entries_to_cancel = exit_required & pending_entries.keys()
        for name in sorted(entries_to_cancel):
            cancelled_today += int(cancel_entry(name, day, "exit_instruction"))

        exit_cause_today = np.zeros(name_count, dtype=np.int8)
        exit_side_today = np.zeros(name_count, dtype=np.int8)
        ineligible_exit_within_hold_today = 0
        for name in sorted(exit_required):
            if name in pending_exits or shares[name] == 0.0:
                continue
            purpose: OrderPurpose
            if day == day_count - 1:
                purpose = "terminal_exit"
                cause: ExitInstructionCause = "terminal"
            else:
                purpose = "exit"
                cause = exit_cause_by_name[name]
            if (
                cause == "ineligible_hold_exhausted"
                and ineligible_streak[name] <= config.ineligible_hold_sessions
            ):
                ineligible_exit_within_hold_today += 1
            exit_cause_today[name] = EXIT_INSTRUCTION_CAUSE_CODES[cause]
            exit_side_today[name] = 1 if shares[name] > 0.0 else -1
            pending_exits[name] = submit_order(
                day,
                name,
                "sell" if shares[name] > 0.0 else "buy",
                1.0,
                float(marks[name]),
                purpose,
                None,
                position_fraction=True,
            )

        def projected_signed_values() -> NDArray[np.float64]:
            projected = signed_values.copy()
            for name, pending in pending_exits.items():
                projected[name] -= pending.remaining_size * signed_values[name]
            for name, pending in pending_entries.items():
                direction = 1.0 if pending.order.side == "buy" else -1.0
                projected[name] += direction * pending.remaining_size
            return projected

        def risk_projected_signed_values() -> NDArray[np.float64]:
            """Project risk without assuming any still-pending exit will fill."""

            projected = signed_values.copy()
            for name, pending in pending_entries.items():
                direction = 1.0 if pending.order.side == "buy" else -1.0
                projected[name] += direction * pending.remaining_size
            return projected

        risk_exit_quantity: dict[int, float] = {}
        risk_trim_gross = 0.0
        risk_trim_net = 0.0
        risk_trim_name = 0.0

        def trim_name(
            projected: NDArray[np.float64], name: int, notional: float
        ) -> float:
            if name in pending_exits or shares[name] == 0.0:
                return 0.0
            mark = float(marks[name])
            if not np.isfinite(mark) or mark <= 0.0:
                return 0.0
            amount = min(max(float(notional), 0.0), abs(float(projected[name])))
            quantity = min(amount / mark, abs(float(shares[name])))
            already = risk_exit_quantity.get(name, 0.0)
            quantity = min(quantity, abs(float(shares[name])) - already)
            if quantity <= 1e-12:
                return 0.0
            actual = quantity * mark
            risk_exit_quantity[name] = already + quantity
            projected[name] += -actual if shares[name] > 0.0 else actual
            return actual

        def trim_side(
            projected: NDArray[np.float64], side: OrderSide, notional: float
        ) -> float:
            """Plan lowest-conviction partial exits and return reduced notional."""

            remaining = max(float(notional), 0.0)
            if remaining <= 1e-12:
                return 0.0
            candidates = [
                int(name)
                for name in np.flatnonzero(
                    projected > 1e-12 if side == "sell" else projected < -1e-12
                )
                if name not in pending_exits and shares[name] != 0.0
            ]
            candidates.sort(
                key=lambda name: (
                    float(inputs.score[day, name])
                    if np.isfinite(inputs.score[day, name])
                    else (-np.inf if side == "sell" else np.inf),
                    name,
                ),
                reverse=side == "buy",
            )
            reduced = 0.0
            for name in candidates:
                if remaining <= 1e-12:
                    break
                actual = trim_name(projected, name, remaining)
                remaining -= actual
                reduced += actual
            return reduced

        if risk_breach and day < day_count - 1:
            projected = risk_projected_signed_values()

            name_limit = config.planned_name_weight_cap * start_nav
            for name in np.flatnonzero(np.abs(projected) > name_limit + 1e-12):
                name = int(name)
                if name in pending_entries:
                    cancelled_today += int(cancel_entry(name, day, "risk_name_cap"))
                    projected = risk_projected_signed_values()
                    for risk_name, quantity in risk_exit_quantity.items():
                        direction = -1.0 if shares[risk_name] > 0.0 else 1.0
                        projected[risk_name] += direction * quantity * marks[risk_name]
                excess = max(abs(float(projected[name])) - name_limit, 0.0)
                if excess > 1e-12:
                    risk_trim_name += trim_name(projected, name, excess)

            projected_gross, _, _ = _risk(projected, start_nav)
            if projected_gross > config.planned_gross_cap + 1e-12:
                total_reduction = max(
                    (projected_gross - config.gross_target) * start_nav, 0.0
                )
                long_notional = float(projected[projected > 0.0].sum())
                short_notional = float(-projected[projected < 0.0].sum())
                long_excess = max(
                    long_notional - config.gross_target * start_nav / 2.0, 0.0
                )
                short_excess = max(
                    short_notional - config.gross_target * start_nav / 2.0, 0.0
                )
                side_excess = long_excess + short_excess
                if side_excess <= 1e-12:
                    long_reduction = (
                        total_reduction
                        * long_notional
                        / (long_notional + short_notional)
                    )
                else:
                    long_reduction = total_reduction * long_excess / side_excess
                short_reduction = total_reduction - long_reduction
                risk_trim_gross += trim_side(projected, "sell", long_reduction)
                risk_trim_gross += trim_side(projected, "buy", short_reduction)

            _, projected_net, _ = _risk(projected, start_nav)
            if abs(projected_net) > config.planned_absolute_net_cap + 1e-12:
                heavy_side: OrderSide = "sell" if projected_net > 0.0 else "buy"
                heavy_entries = [
                    name
                    for name, pending in pending_entries.items()
                    if pending.order.side == ("buy" if heavy_side == "sell" else "sell")
                ]
                heavy_entries.sort(
                    key=lambda name: (
                        float(inputs.score[day, name]),
                        name,
                    ),
                    reverse=heavy_side == "buy",
                )
                target_net = config.planned_absolute_net_cap / 2.0
                for name in heavy_entries:
                    if abs(projected_net) <= target_net + 1e-12:
                        break
                    cancelled_today += int(cancel_entry(name, day, "risk_net_cap"))
                    projected = risk_projected_signed_values()
                    for risk_name, quantity in risk_exit_quantity.items():
                        direction = -1.0 if shares[risk_name] > 0.0 else 1.0
                        projected[risk_name] += direction * quantity * marks[risk_name]
                    _, projected_net, _ = _risk(projected, start_nav)
                net_reduction = max((abs(projected_net) - target_net) * start_nav, 0.0)
                risk_trim_net += trim_side(projected, heavy_side, net_reduction)

            for name, quantity in sorted(risk_exit_quantity.items()):
                quantity = min(quantity, abs(float(shares[name])))
                if quantity <= 1e-12 or name in pending_exits:
                    continue
                pending_exits[name] = submit_order(
                    day,
                    name,
                    "sell" if shares[name] > 0.0 else "buy",
                    quantity / abs(float(shares[name])),
                    float(marks[name]),
                    "risk_exit",
                    None,
                    position_fraction=True,
                )

        if day == day_count - 1:
            for name in tuple(pending_entries):
                cancelled_today += int(cancel_entry(name, day, "evaluation_end"))

        blocked_reference = 0
        blocked_gross = 0
        blocked_net = 0
        blocked_name = 0
        net_cap_balanced = 0
        gross_cap_below_target = 0
        name_cap_fresh = 0
        submitted_entries = 0
        same_close_replacements = 0
        occupied_after_submission_long = min(int((shares > 0.0).sum()), k_eff)
        occupied_after_submission_short = min(int((shares < 0.0).sum()), k_eff)
        open_after_submission_long = max(k_eff - occupied_after_submission_long, 0)
        open_after_submission_short = max(k_eff - occupied_after_submission_short, 0)
        band_candidates_long = 0
        band_candidates_short = 0
        band_excluded_unresolved_long = 0
        band_excluded_unresolved_short = 0
        band_excluded_settled_long = 0
        band_excluded_settled_short = 0
        band_without_prior_print_long = 0
        band_without_prior_print_short = 0
        band_exhausted_long = 0
        band_exhausted_short = 0
        blocked_open_long = 0
        blocked_open_short = 0
        excluded_short_candidates = 0
        spilled_entries: dict[OrderSide, NDArray[np.int64]] = {
            "buy": np.zeros(config.volatility_group_count, dtype=np.int64),
            "sell": np.zeros(config.volatility_group_count, dtype=np.int64),
        }
        if day < day_count - 1 and not small_universe:
            slot_notional = start_nav * config.gross_target / (2 * config.k_per_side)
            same_day_exit_names = {
                name
                for name, pending in pending_exits.items()
                if pending.order.decision_session == day
                and pending.remaining_size >= 1.0 - 1e-12
            }
            unavailable = set(np.flatnonzero(shares != 0.0).tolist()) | set(
                pending_entries
            )
            long_occupied = {
                int(name)
                for name in np.flatnonzero(shares > 0.0)
                if int(name) not in same_day_exit_names
            } | {
                name
                for name, pending in pending_entries.items()
                if pending.order.side == "buy"
            }
            short_occupied = {
                int(name)
                for name in np.flatnonzero(shares < 0.0)
                if int(name) not in same_day_exit_names
            } | {
                name
                for name, pending in pending_entries.items()
                if pending.order.side == "sell"
            }
            long_slots = max(k_eff - len(long_occupied), 0)
            short_slots = max(k_eff - len(short_occupied), 0)
            if config.volatility_balanced_entries:
                long_band = []
                short_band = []
                for group in range(config.volatility_group_count):
                    width = int(volatility_quota[group] + volatility_buffer[group])
                    if width == 0:
                        continue
                    group_order = group_orders[group]
                    long_band.extend(int(name) for name in group_order[-width:][::-1])
                    short_band.extend(int(name) for name in group_order[:width])
            else:
                long_band = [int(name) for name in entry_order[-k_eff:][::-1]]
                short_band = [
                    int(name)
                    for name in entry_order[
                        : (
                            len(entry_order) // 2
                            if config.borrow_source != "uniform"
                            else k_eff
                        )
                    ]
                ]
            if set(long_band) & set(short_band):
                raise RuntimeError("long and short entry bands overlap")
            long_candidates = [
                name
                for name in long_band
                if entry_eligible[name]
                if name not in unavailable
                and not decision_unresolved[name]
                and not settled_names[name]
            ]
            short_candidates = [
                name
                for name in short_band
                if entry_eligible[name]
                if name not in unavailable
                and not decision_unresolved[name]
                and not settled_names[name]
                and inputs.shortable[day, name]
            ]
            long_candidates.sort(
                key=lambda name: (float(inputs.score[day, name]), name), reverse=True
            )
            short_candidates.sort(
                key=lambda name: (float(inputs.score[day, name]), name)
            )
            excluded_short_candidates = sum(
                entry_eligible[name]
                and name not in unavailable
                and not decision_unresolved[name]
                and not settled_names[name]
                and not inputs.shortable[day, name]
                for name in short_band
            )
            band_candidates_long = len(long_candidates)
            band_candidates_short = len(short_candidates)
            band_excluded_unresolved_long = sum(
                name not in unavailable and bool(decision_unresolved[name])
                for name in long_band
            )
            band_excluded_unresolved_short = sum(
                name not in unavailable and bool(decision_unresolved[name])
                for name in short_band
            )
            band_excluded_settled_long = sum(
                name not in unavailable
                and not decision_unresolved[name]
                and bool(settled_names[name])
                for name in long_band
            )
            band_excluded_settled_short = sum(
                name not in unavailable
                and not decision_unresolved[name]
                and bool(settled_names[name])
                for name in short_band
            )
            band_without_prior_print_long = sum(
                last_print_session[name] < day - 1 for name in long_candidates
            )
            band_without_prior_print_short = sum(
                last_print_session[name] < day - 1 for name in short_candidates
            )
            replacement_capacity = {
                "buy": sum(shares[name] > 0.0 for name in same_day_exit_names),
                "sell": sum(shares[name] < 0.0 for name in same_day_exit_names),
            }
            planned_values = projected_signed_values()
            candidates_by_side = {
                "buy": long_candidates,
                "sell": short_candidates,
            }
            remaining_slots = {"buy": long_slots, "sell": short_slots}
            attempted_candidates: dict[OrderSide, set[int]] = {
                "buy": set(),
                "sell": set(),
            }
            volatility_occupancy: dict[OrderSide, NDArray[np.int64]] = {
                "buy": np.zeros(config.volatility_group_count, dtype=np.int64),
                "sell": np.zeros(config.volatility_group_count, dtype=np.int64),
            }
            if config.volatility_balanced_entries:
                for side, occupied in (
                    ("buy", long_occupied),
                    ("sell", short_occupied),
                ):
                    for name in occupied:
                        group = int(volatility_groups[name])
                        if group >= 0:
                            volatility_occupancy[side][group] += 1
                centre = (config.volatility_group_count - 1) / 2.0
                group_priority = sorted(
                    range(config.volatility_group_count),
                    key=lambda group: (abs(group - centre), group),
                )
            else:
                group_priority = []

            def next_candidate(side: OrderSide) -> tuple[int, bool] | None:
                candidates = candidates_by_side[side]
                attempted = attempted_candidates[side]
                if config.volatility_balanced_entries:
                    for group in group_priority:
                        if volatility_occupancy[side][group] >= volatility_quota[group]:
                            continue
                        for candidate in candidates:
                            if (
                                candidate not in attempted
                                and volatility_groups[candidate] == group
                            ):
                                return candidate, False
                    spill = next(
                        (
                            candidate
                            for candidate in candidates
                            if candidate not in attempted
                        ),
                        None,
                    )
                    return None if spill is None else (spill, True)
                candidate = next(
                    (
                        candidate
                        for candidate in candidates
                        if candidate not in attempted
                    ),
                    None,
                )
                return None if candidate is None else (candidate, False)

            side_budget = {
                "buy": long_slots * slot_notional,
                "sell": short_slots * slot_notional,
            }
            inverse_weight_scale: dict[OrderSide, float] = {}
            if sizing_volatility is not None:
                for side in ("buy", "sell"):
                    candidates = [
                        name
                        for name in candidates_by_side[side]
                        if np.isfinite(last_observed[name])
                        and last_observed[name] > 0.0
                    ]
                    sigma = sizing_volatility[day, candidates]
                    if np.any(~np.isfinite(sigma) | (sigma <= 0.0)):
                        raise ValueError(
                            "inverse-volatility entries require positive causal sigma"
                        )
                    planned: list[int] = []
                    for group in group_priority:
                        deficit = max(
                            int(
                                volatility_quota[group]
                                - volatility_occupancy[side][group]
                            ),
                            0,
                        )
                        planned.extend(
                            [
                                name
                                for name in candidates
                                if volatility_groups[name] == group
                            ][:deficit]
                        )
                    planned.extend(name for name in candidates if name not in planned)
                    planned = planned[: remaining_slots[side]]
                    inverse_weight_scale[side] = (
                        side_budget[side]
                        / float(np.sum(1.0 / sizing_volatility[day, planned]))
                        if planned
                        else 0.0
                    )

            if long_slots > short_slots:
                current_side: OrderSide = "buy"
            elif short_slots > long_slots:
                current_side = "sell"
            else:
                current_side = "buy"
            while True:
                other_side: OrderSide = "sell" if current_side == "buy" else "buy"
                candidate_row = (
                    next_candidate(current_side)
                    if remaining_slots[current_side] > 0
                    else None
                )
                other_candidate_row = (
                    next_candidate(other_side)
                    if remaining_slots[other_side] > 0
                    else None
                )
                available = candidate_row is not None
                other_available = other_candidate_row is not None
                if not available and not other_available:
                    break
                if not available:
                    current_side = other_side
                    candidate_row = other_candidate_row
                if candidate_row is None:
                    raise RuntimeError("entry scheduler lost an available candidate")
                name, is_spill = candidate_row
                attempted_candidates[current_side].add(name)
                reference = float(last_observed[name])
                if not np.isfinite(reference) or reference <= 0.0:
                    blocked_reference += 1
                    current_side = "sell" if current_side == "buy" else "buy"
                    continue
                entry_notional = slot_notional
                if sizing_volatility is not None:
                    entry_notional = min(
                        inverse_weight_scale[current_side]
                        / sizing_volatility[day, name],
                        start_nav * config.planned_name_weight_cap,
                        side_budget[current_side],
                    )
                    if entry_notional <= 1e-12 * start_nav:
                        current_side = "sell" if current_side == "buy" else "buy"
                        continue
                proposed = planned_values.copy()
                proposed[name] += (
                    entry_notional if current_side == "buy" else -entry_notional
                )
                gross_before, net_before, _ = _risk(planned_values, start_nav)
                planned_gross, planned_net, _ = _risk(proposed, start_nav)
                violates_gross = planned_gross > config.planned_gross_cap + 1e-12
                violates_net = (
                    abs(planned_net) > config.planned_absolute_net_cap + 1e-12
                )
                # A missing print can leave another name overweight while its
                # risk exit is pending. Its concentration cannot veto an entry
                # in this name; aggregate gross/net limits still apply.
                violates_name = (
                    abs(proposed[name]) / start_nav
                    > config.planned_name_weight_cap + 1e-12
                )
                blocked_gross += int(violates_gross)
                blocked_net += int(violates_net)
                blocked_name += int(violates_name)
                gross_cap_below_target += int(
                    violates_gross and gross_before < config.gross_target
                )
                net_cap_balanced += int(
                    violates_net
                    and abs(net_before)
                    <= config.planned_absolute_net_cap
                    - entry_notional / start_nav
                    - 1e-9
                )
                name_cap_fresh += int(
                    violates_name
                    and shares[name] == 0.0
                    and name not in pending_entries
                )
                if violates_gross or violates_net or violates_name:
                    current_side = "sell" if current_side == "buy" else "buy"
                    continue
                expiry = day + config.entry_expiry_sessions - 1
                submission_nav[name] = start_nav
                entry_cost_basis[name] = 0.0
                pending_entries[name] = submit_order(
                    day,
                    name,
                    current_side,
                    entry_notional,
                    reference,
                    "entry",
                    expiry,
                )
                remaining_slots[current_side] -= 1
                side_budget[current_side] -= entry_notional
                if config.volatility_balanced_entries:
                    volatility_occupancy[current_side][volatility_groups[name]] += 1
                    if is_spill:
                        spilled_entries[current_side][volatility_groups[name]] += 1
                submitted_entries += 1
                if replacement_capacity[current_side] > 0:
                    same_close_replacements += 1
                    replacement_capacity[current_side] -= 1
                planned_values = proposed
                current_side = "sell" if current_side == "buy" else "buy"
            occupied_after_submission_long = k_eff - remaining_slots["buy"]
            occupied_after_submission_short = k_eff - remaining_slots["sell"]
            open_after_submission_long = remaining_slots["buy"]
            open_after_submission_short = remaining_slots["sell"]
            band_exhausted_long = (
                remaining_slots["buy"]
                if len(attempted_candidates["buy"]) == len(long_candidates)
                else 0
            )
            band_exhausted_short = (
                remaining_slots["sell"]
                if len(attempted_candidates["sell"]) == len(short_candidates)
                else 0
            )
            blocked_open_long = (
                remaining_slots["buy"]
                if len(attempted_candidates["buy"]) < len(long_candidates)
                else 0
            )
            blocked_open_short = (
                remaining_slots["sell"]
                if len(attempted_candidates["sell"]) < len(short_candidates)
                else 0
            )
        blocked_reference_rows.append(blocked_reference)
        blocked_gross_rows.append(blocked_gross)
        blocked_net_rows.append(blocked_net)
        blocked_name_rows.append(blocked_name)
        submitted_entry_rows.append(submitted_entries)
        submitted_exit_rows.append(
            sum(
                order_.decision_session == day and order_.purpose != "entry"
                for order_ in orders
            )
        )
        same_close_replacement_rows.append(same_close_replacements)
        zero_entry_small_rows.append(submitted_entries == 0 and small_universe)
        zero_entry_gross_rows.append(submitted_entries == 0 and blocked_gross > 0)
        zero_entry_net_rows.append(submitted_entries == 0 and blocked_net > 0)
        zero_entry_name_rows.append(submitted_entries == 0 and blocked_name > 0)
        zero_entry_reference_rows.append(
            submitted_entries == 0 and blocked_reference > 0
        )
        risk_trim_gross_rows.append(risk_trim_gross)
        risk_trim_net_rows.append(risk_trim_net)
        risk_trim_name_rows.append(risk_trim_name)
        occupied_after_submission_long_rows.append(occupied_after_submission_long)
        occupied_after_submission_short_rows.append(occupied_after_submission_short)
        open_after_submission_long_rows.append(open_after_submission_long)
        open_after_submission_short_rows.append(open_after_submission_short)
        band_candidates_long_rows.append(band_candidates_long)
        band_candidates_short_rows.append(band_candidates_short)
        excluded_short_candidate_rows.append(excluded_short_candidates)
        band_excluded_unresolved_long_rows.append(band_excluded_unresolved_long)
        band_excluded_unresolved_short_rows.append(band_excluded_unresolved_short)
        band_excluded_settled_long_rows.append(band_excluded_settled_long)
        band_excluded_settled_short_rows.append(band_excluded_settled_short)
        band_without_prior_print_long_rows.append(band_without_prior_print_long)
        band_without_prior_print_short_rows.append(band_without_prior_print_short)
        band_exhausted_long_rows.append(band_exhausted_long)
        band_exhausted_short_rows.append(band_exhausted_short)
        blocked_open_long_rows.append(blocked_open_long)
        blocked_open_short_rows.append(blocked_open_short)
        exit_ineligible_hold_exhausted_rows.append(int((exit_cause_today == 1).sum()))
        exit_settlement_rows.append(int((exit_cause_today == 2).sum()))
        exit_rank_rows.append(int((exit_cause_today == 3).sum()))
        exit_terminal_rows.append(int((exit_cause_today == 4).sum()))
        exit_cause_rows.append(exit_cause_today)
        exit_side_rows.append(exit_side_today)
        ineligible_exit_within_hold_rows.append(ineligible_exit_within_hold_today)
        net_cap_balanced_rows.append(net_cap_balanced)
        gross_cap_below_target_rows.append(gross_cap_below_target)
        name_cap_fresh_rows.append(name_cap_fresh)

        # A last-mark settlement is conditional accounting, not an observed
        # market execution. Freeze its fraction before the possible final print;
        # expire the intention if a print makes the settlement unnecessary.
        settlement_orders = {
            int(name): submit_order(
                day,
                int(name),
                "sell" if shares[name] > 0.0 else "buy",
                1.0,
                float(marks[name]),
                "terminal_settlement",
                day,
                position_fraction=True,
            )
            for name in np.flatnonzero(
                (shares != 0.0)
                & ~settled_names
                & (
                    (missing_sessions >= config.settlement_grace_sessions - 1)
                    | (config.settle_terminal_residuals and day == day_count - 1)
                )
            )
        }

        # Freeze the hedge before reading any current-session fill or close.
        planned_equity = np.zeros(name_count, dtype=np.float64)
        held_for_hedge = shares != 0.0
        planned_equity[held_for_hedge] = shares[held_for_hedge] * marks[held_for_hedge]
        for name, pending in pending_entries.items():
            direction = 1.0 if pending.order.side == "buy" else -1.0
            planned_equity[name] += direction * pending.remaining_size
        for name, pending in pending_exits.items():
            planned_equity[name] -= pending.remaining_size * shares[name] * marks[name]
        beta_required = held_for_hedge.copy()
        for name in pending_entries:
            beta_required[name] = True
        hedge_beta_fallback_rows.append(
            bool(
                config.beta_hedge
                and np.any(beta_required & inputs.hedge_beta_fallback[day])
            )
        )
        exposure = planned_equity != 0.0
        equity_beta_notional = (
            float(np.sum(planned_equity[exposure] * inputs.hedge_beta[day, exposure]))
            if config.beta_hedge
            else 0.0
        )
        hedge_unconstrained_target_notional = (
            -equity_beta_notional if day != day_count - 1 else 0.0
        )
        hedge_limit = config.hedge_notional_cap_nav * start_nav
        hedge_target_notional = float(
            np.clip(hedge_unconstrained_target_notional, -hedge_limit, hedge_limit)
        )
        hedge_capped = (
            abs(hedge_unconstrained_target_notional - hedge_target_notional) > 1e-12
        )
        hedge_notional_before = hedge_shares * hedge_mark if hedge_shares else 0.0
        rebalance_required = config.beta_hedge and (
            abs(hedge_target_notional - hedge_notional_before)
            > config.hedge_rebalance_threshold_nav * start_nav
            or abs(hedge_notional_before) > hedge_limit + 1e-12
            or (day == day_count - 1 and hedge_shares != 0.0)
        )
        hedge_order = None
        if rebalance_required and np.isfinite(hedge_mark):
            change = hedge_target_notional - hedge_notional_before
            if change != 0.0:
                hedge_order = submit_order(
                    day,
                    name_count,
                    "buy" if change > 0 else "sell",
                    1.0 if day == day_count - 1 else abs(change),
                    hedge_mark,
                    "hedge",
                    day,
                    position_fraction=day == day_count - 1,
                )
        decision_hedge_notional = (
            hedge_target_notional if hedge_order is not None else hedge_notional_before
        )

        for name in pending_entries.keys() | pending_exits.keys():
            action_exposed[name] = True
        same_day_action_rows.append(
            bool(np.any(action_exposed & inputs.has_action[day]))
        )

        # Action-term uncertainty is a property of this session's claim, not
        # of the position for the rest of its life. A later resolved cell
        # clears the condition without requiring a fill.
        unresolved_action = ~inputs.action_resolved[day].copy()
        explicit_unresolved_action = inputs.has_action[day] & unresolved_action
        action_uncertainty_seen |= bool(
            ((shares != 0.0) & explicit_unresolved_action).any()
        )
        # Apply contractual terms to shares held before the session. Cash terms
        # become claims; only the later payment mask transfers them to cash.
        for name in np.flatnonzero(inputs.has_action[day]):
            successor = int(inputs.action_successor[day, name])
            if not inputs.action_resolved[day, name]:
                if (
                    shares[name] != 0.0
                    or receivable_by_name[name] != 0.0
                    or payable_by_name[name] != 0.0
                ):
                    unresolved_action[name] = True
                continue
            q = float(inputs.action_q[day, name])
            d = float(inputs.action_d[day, name])
            old_shares = float(shares[name])
            new_shares, signed_claim = apply_contractual_action(
                old_shares,
                0.0,
                shares_per_prior_share=q,
                cash_per_prior_share=d,
            )
            claim = float(signed_claim)
            if claim >= 0.0:
                receivable_by_name[name] += claim
            else:
                payable_by_name[name] -= claim
            if claim != 0.0:
                payment_session = int(inputs.action_payment_session[day, name])
                pending_claims.append(
                    _PendingClaim(
                        security_index=name,
                        signed_amount=claim,
                        payment_session=(
                            payment_session if payment_session >= 0 else None
                        ),
                    )
                )
            converted_mark = (
                (marks[name] - d) / q
                if old_shares != 0.0 and q > 0.0 and np.isfinite(marks[name])
                else np.nan
            )
            converted_reference = (
                (last_observed[name] - d) / q
                if q > 0.0 and np.isfinite(last_observed[name])
                else np.nan
            )
            if not np.isfinite(converted_reference) or converted_reference <= 0.0:
                converted_reference = np.nan
            if successor == name:
                shares[name] = float(new_shares)
                marks[name] = converted_mark if shares[name] != 0.0 else np.nan
                last_observed[name] = converted_reference
                if shares[name] == 0.0:
                    ineligible_streak[name] = 0
                    entry_cost_basis[name] = 0.0
                    submission_nav[name] = 0.0
            else:
                if shares[successor] != 0.0:
                    raise ValueError(
                        "conversion successor already has ledger inventory"
                    )
                shares[name] = 0.0
                marks[name] = np.nan
                last_observed[name] = np.nan
                for pending_map in (pending_exits, settlement_orders):
                    pending = pending_map.pop(int(name), None)
                    if pending is not None:
                        pending_map[successor] = pending
                missing_sessions[successor] = missing_sessions[name]
                missing_sessions[name] = 0
                shares[successor] = float(new_shares)
                marks[successor] = converted_mark
                # Reference marks belong to the tradable claim even when no
                # inventory existed at the conversion.  Do not overwrite the
                # causal predecessor-derived reference with the inventory-only
                # converted mark (NaN in that case).
                last_observed[successor] = converted_reference
                restricted_by_name[successor] += restricted_by_name[name]
                restricted_by_name[name] = 0.0
                if np.isfinite(converted_mark) and converted_mark <= 0.0:
                    unresolved_action[successor] = True
                    explicit_unresolved_action[successor] = True
                explicit_unresolved_action[successor] |= explicit_unresolved_action[
                    name
                ]
                unresolved_action[name] = False
                explicit_unresolved_action[name] = False
                entry_session[successor] = entry_session[name]
                entry_session[name] = -1
                ineligible_streak[successor] = ineligible_streak[name]
                ineligible_streak[name] = 0
                entry_cost_basis[successor] = entry_cost_basis[name]
                submission_nav[successor] = submission_nav[name]
                entry_cost_basis[name] = 0.0
                submission_nav[name] = 0.0
            if (
                successor == name
                and np.isfinite(converted_mark)
                and converted_mark <= 0.0
            ):
                unresolved_action[name] = True
                explicit_unresolved_action[name] = True
            if q == 0.0 and restricted_by_name[name] != 0.0:
                free_cash += restricted_by_name[name]
                restricted_by_name[name] = 0.0
                ineligible_streak[name] = 0
                entry_session[name] = -1
                entry_cost_basis[name] = 0.0
                submission_nav[name] = 0.0

        unpaid_claims: list[_PendingClaim] = []
        for claim in pending_claims:
            if claim.payment_session != day:
                unpaid_claims.append(claim)
                continue
            name = claim.security_index
            free_cash += claim.signed_amount
            if claim.signed_amount > 0.0:
                receivable_by_name[name] -= claim.signed_amount
                if abs(receivable_by_name[name]) <= 1e-12:
                    receivable_by_name[name] = 0.0
            else:
                payable_by_name[name] += claim.signed_amount
                if abs(payable_by_name[name]) <= 1e-12:
                    payable_by_name[name] = 0.0
        pending_claims = unpaid_claims

        short_at_open = shares < 0.0
        short_value_at_open = float(
            np.abs(shares[short_at_open] * marks[short_at_open]).sum()
        )
        cash_rate = float(inputs.cdi[day])
        debit_rate = cash_rate + config.annual_debit_spread / config.annual_sessions
        free_cash_interest = (
            max(free_cash, 0.0) * cash_rate + min(free_cash, 0.0) * debit_rate
        )
        short_proceeds_interest_base = float(
            restricted_by_name.sum() + hedge_restricted_cash
        )
        short_proceeds_interest = (
            short_proceeds_interest_base
            * cash_rate
            * config.short_proceeds_remuneration
        )
        interest = free_cash_interest + short_proceeds_interest
        imputed_short_notional = 0.0
        placeholder_short_notional = 0.0
        if config.borrow_source != "uniform" and short_at_open.any():
            short_values = np.abs(shares[short_at_open] * marks[short_at_open])
            raw_rates = inputs.annual_borrow_rate_by_name[day, short_at_open]
            if not np.isfinite(raw_rates).all():
                raise RuntimeError("held archive-borrow short has no finite rate")
            fee_rates = np.asarray(
                equity_borrow_registration_fee(raw_rates, config=config),
                dtype=np.float64,
            )
            effective_rates = raw_rates + fee_rates
            equity_borrow_raw = float(
                np.sum(
                    short_values
                    * np.expm1(np.log1p(raw_rates) / config.annual_sessions)
                )
            )
            equity_borrow_fee = float(
                np.sum(
                    short_values
                    * np.expm1(np.log1p(fee_rates) / config.annual_sessions)
                )
            )
            borrow = equity_borrow_raw + equity_borrow_fee
            weighted_borrow_rate = float(
                np.sum(short_values * effective_rates) / short_value_at_open
            )
            imputed_short_notional = float(
                np.sum(short_values[inputs.borrow_rate_imputed[day, short_at_open]])
            )
            placeholder_short_notional = float(
                np.sum(short_values[inputs.borrow_rate_placeholder[day, short_at_open]])
            )
        else:
            equity_borrow_raw = (
                np.expm1(np.log1p(config.annual_borrow_rate) / config.annual_sessions)
                * short_value_at_open
            )
            uniform_fee_rate = float(
                equity_borrow_registration_fee(
                    config.annual_borrow_rate,
                    config=config,
                )
            )
            equity_borrow_fee = (
                np.expm1(np.log1p(uniform_fee_rate) / config.annual_sessions)
                * short_value_at_open
            )
            borrow = equity_borrow_raw + equity_borrow_fee
            weighted_borrow_rate = (
                config.annual_borrow_rate + uniform_fee_rate
                if short_value_at_open > 0.0
                else 0.0
            )
        hedge_borrow = 0.0
        if config.beta_hedge and hedge_shares < 0.0:
            hedge_rate = max(
                float(inputs.hedge_annual_borrow_rate[day])
                if np.isfinite(inputs.hedge_annual_borrow_rate[day])
                else 0.0,
                config.hedge_annual_borrow_rate,
            )
            hedge_borrow = abs(hedge_shares * hedge_mark) * np.expm1(
                np.log1p(hedge_rate) / config.annual_sessions
            )
            borrow += hedge_borrow
        held_short_borrow_rate_rows.append(weighted_borrow_rate)
        held_short_notional_rows.append(short_value_at_open)
        held_short_imputed_notional_rows.append(imputed_short_notional)
        held_short_placeholder_notional_rows.append(placeholder_short_notional)
        free_cash += interest - borrow
        settlement_scenario_adjustment *= 1.0 + cash_rate

        # Only now may current-session prints affect the result. This makes the
        # immutable intended-order set invariant to those later observations.
        printed = np.isfinite(inputs.raw_close[day]) & (inputs.raw_close[day] > 0.0)
        if day == day_count - 1:
            terminal_printed = printed.copy()
        traded_notional = 0.0
        costs = 0.0
        entry_fill_short_today = 0
        exit_fill_long_today = 0
        exit_fill_short_today = 0
        cost_rate = config.cost_bps_per_side / 10_000.0
        for pending_map in (pending_exits, pending_entries):
            entries = pending_map is pending_entries
            for name, pending in tuple(pending_map.items()):
                # Uncertain action terms can block opening risk, but never a
                # printed exit, risk reduction, or terminal liquidation.
                if not entries and shares[name] == 0.0:
                    cancellations.append(
                        pending.cancellation(day, inputs.dates[day], "position_closed")
                    )
                    del pending_map[name]
                    continue
                if not printed[name]:
                    continue
                fraction = float(inputs.fill_fraction[day, name])
                remaining_before_fill = pending.remaining_size
                used_size = remaining_before_fill * fraction
                quantity = (
                    used_size / inputs.raw_close[day, name]
                    if entries
                    else used_size * abs(float(shares[name]))
                )
                if quantity <= 0.0:
                    continue
                before = float(shares[name])
                free_cash, notional = _book_fill(
                    name=name,
                    side=pending.order.side,
                    quantity=quantity,
                    price=float(inputs.raw_close[day, name]),
                    shares=shares,
                    marks=marks,
                    restricted_by_name=restricted_by_name,
                    free_cash=free_cash,
                    cost_rate=cost_rate,
                )
                after = float(shares[name])
                if entries:
                    entry_cost_basis[name] += notional
                    entry_fill_short_today += int(
                        used_size < remaining_before_fill - 1e-12
                    )
                elif before != 0.0 and abs(after) < abs(before):
                    if after == 0.0:
                        ineligible_streak[name] = 0
                        entry_cost_basis[name] = 0.0
                        submission_nav[name] = 0.0
                        exit_fill_long_today += int(before > 0.0)
                        exit_fill_short_today += int(before < 0.0)
                    else:
                        entry_cost_basis[name] *= abs(after) / abs(before)
                if before == 0.0 and shares[name] != 0.0:
                    entry_session[name] = day
                if before != 0.0 and shares[name] == 0.0:
                    entry_session[name] = -1
                fill_cost = cost_rate * notional
                fills.append(
                    Fill(
                        order_id=pending.order.order_id,
                        security=inputs.securities[name],
                        security_index=name,
                        fill_date=inputs.dates[day],
                        fill_session=day,
                        side=pending.order.side,
                        quantity=quantity,
                        price=float(inputs.raw_close[day, name]),
                        gross_notional=notional,
                        cost=fill_cost,
                        purpose=pending.order.purpose,
                    )
                )
                traded_notional += notional
                costs += fill_cost
                remaining = remaining_before_fill - used_size
                pending.remaining_size = (
                    remaining
                    if entries or after == 0.0
                    else remaining / (1.0 - used_size)
                )
                if pending.remaining_size <= 1e-12:
                    del pending_map[name]

        for name, pending in tuple(pending_entries.items()):
            expiry = pending.order.expiry_session
            if expiry is not None and day >= expiry:
                cancellations.append(
                    pending.cancellation(day, inputs.dates[day], "expired")
                )
                del pending_entries[name]
                cancelled_today += 1
                if shares[name] == 0.0:
                    entry_cost_basis[name] = 0.0
                    submission_nav[name] = 0.0

        pending_printed_unfilled_today = sum(
            bool(printed[name]) for name in pending_entries
        )
        pending_end_long = sum(
            pending.order.side == "buy" for pending in pending_entries.values()
        )
        pending_end_short = sum(
            pending.order.side == "sell" for pending in pending_entries.values()
        )
        pending_without_print_long = sum(
            pending.order.side == "buy" and not printed[name]
            for name, pending in pending_entries.items()
        )
        pending_without_print_short = sum(
            pending.order.side == "sell" and not printed[name]
            for name, pending in pending_entries.items()
        )

        newly_reprinted = printed & settled_names & ~settled_then_printed_seen
        settled_then_printed_seen |= newly_reprinted
        settled_then_printed_today = int(newly_reprinted.sum())

        for name in range(name_count):
            if printed[name]:
                last_print_session[name] = day
                last_observed[name] = inputs.raw_close[day, name]
                if shares[name] != 0.0:
                    marks[name] = inputs.raw_close[day, name]
                    missing_sessions[name] = 0
            elif shares[name] != 0.0:
                missing_sessions[name] += 1

        settlement_count = 0
        settlement_notional = 0.0
        if config.settle_terminal_residuals and day == day_count - 1:
            boundary_unpriced = (shares != 0.0) & ~printed
            terminal_boundary_unpriced_inventory_notional = float(
                np.abs(shares[boundary_unpriced] * marks[boundary_unpriced]).sum()
            )
        due_for_settlement = (
            (shares != 0.0)
            & ~printed
            & ~settled_names
            & (
                (missing_sessions >= config.settlement_grace_sessions)
                | (config.settle_terminal_residuals and day == day_count - 1)
            )
        )
        for name, pending in settlement_orders.items():
            if not due_for_settlement[name]:
                cancellations.append(
                    pending.cancellation(day, inputs.dates[day], "expired")
                )
        for name in np.flatnonzero(due_for_settlement):
            pending = pending_exits.pop(int(name), None)
            if pending is not None:
                cancellations.append(
                    pending.cancellation(day, inputs.dates[day], "terminal_settlement")
                )
            price = float(marks[name])
            if not np.isfinite(price) or price <= 0.0:
                raise RuntimeError("terminal settlement requires a positive last mark")
            before = float(shares[name])
            quantity = abs(before)
            side: OrderSide = "sell" if before > 0.0 else "buy"
            settlement = settlement_orders[int(name)]
            scenario_price = price * (
                1.0 - config.settlement_haircut
                if before > 0.0
                else 1.0 + config.settlement_haircut
            )
            scenario_price_delta = scenario_price - price
            settlement_scenario_adjustment += (
                before * scenario_price_delta
                - cost_rate * quantity * scenario_price_delta
            )
            free_cash, notional = _book_fill(
                name=int(name),
                side=side,
                quantity=quantity,
                price=price,
                shares=shares,
                marks=marks,
                restricted_by_name=restricted_by_name,
                free_cash=free_cash,
                cost_rate=cost_rate,
            )
            exit_fill_long_today += int(before > 0.0)
            exit_fill_short_today += int(before < 0.0)
            entry_cost_basis[name] = 0.0
            submission_nav[name] = 0.0
            fill_cost = cost_rate * notional
            fills.append(
                Fill(
                    order_id=settlement.order.order_id,
                    security=inputs.securities[name],
                    security_index=int(name),
                    fill_date=inputs.dates[day],
                    fill_session=day,
                    side=side,
                    quantity=quantity,
                    price=price,
                    gross_notional=notional,
                    cost=fill_cost,
                    purpose="terminal_settlement",
                )
            )
            traded_notional += notional
            costs += fill_cost
            settlement_count += 1
            settlement_notional += notional
            settled_names[name] = True
            entry_session[name] = -1
            ineligible_streak[name] = 0
            missing_sessions[name] = 0
            unresolved_action[name] = False
            explicit_unresolved_action[name] = False
        stale = int(np.sum((shares != 0.0) & ~printed))

        hedge_traded_notional = 0.0
        hedge_cost = 0.0
        hedge_gross_pnl = 0.0
        hedge_cost_rate = config.hedge_cost_bps_per_side / 10_000.0
        bova_printed = config.beta_hedge and np.isfinite(inputs.hedge_close[day])
        if bova_printed:
            current_hedge_close = float(inputs.hedge_close[day])
            if hedge_shares != 0.0:
                hedge_gross_pnl = hedge_shares * (current_hedge_close - hedge_mark)
            hedge_mark = current_hedge_close
        if hedge_order is not None:
            order = hedge_order.order
            settle_hedge_last_mark = (
                config.settle_terminal_residuals
                and day == day_count - 1
                and not bova_printed
                and hedge_shares != 0.0
            )
            if settle_hedge_last_mark:
                # Boundary accounting only: never present a stale mark as a print.
                current_hedge_close = float(hedge_mark)
                terminal_hedge_last_mark_settlement_notional = abs(
                    hedge_shares * hedge_mark
                )
                settlement_count += 1
                settlement_notional += terminal_hedge_last_mark_settlement_notional
                scenario_delta = (
                    -terminal_hedge_last_mark_settlement_notional
                    * config.settlement_haircut
                )
                settlement_scenario_adjustment += scenario_delta * (
                    1.0 - hedge_cost_rate if hedge_shares > 0 else 1.0 + hedge_cost_rate
                )
            if bova_printed or settle_hedge_last_mark:
                hedge_share_array = np.asarray([hedge_shares], dtype=np.float64)
                hedge_mark_array = np.asarray([hedge_mark], dtype=np.float64)
                hedge_restricted_array = np.asarray(
                    [hedge_restricted_cash], dtype=np.float64
                )
                hedge_quantity = (
                    abs(hedge_shares) * order.position_fraction
                    if order.position_fraction is not None
                    else order.planned_notional / current_hedge_close
                )
                free_cash, hedge_traded_notional = _book_fill(
                    name=0,
                    side=order.side,
                    quantity=hedge_quantity,
                    price=current_hedge_close,
                    shares=hedge_share_array,
                    marks=hedge_mark_array,
                    restricted_by_name=hedge_restricted_array,
                    free_cash=free_cash,
                    cost_rate=hedge_cost_rate,
                )
                hedge_shares = float(hedge_share_array[0])
                hedge_restricted_cash = float(hedge_restricted_array[0])
                hedge_cost = hedge_cost_rate * hedge_traded_notional
                traded_notional += hedge_traded_notional
                costs += hedge_cost
                fills.append(
                    Fill(
                        order_id=order.order_id,
                        security=order.security,
                        security_index=order.security_index,
                        fill_date=inputs.dates[day],
                        fill_session=day,
                        side=order.side,
                        quantity=hedge_quantity,
                        price=current_hedge_close,
                        gross_notional=hedge_traded_notional,
                        cost=hedge_cost,
                        purpose="terminal_settlement"
                        if settle_hedge_last_mark
                        else "hedge",
                    )
                )
            else:
                cancellations.append(
                    hedge_order.cancellation(
                        day,
                        inputs.dates[day],
                        "evaluation_end" if day == day_count - 1 else "expired",
                    )
                )

        held_now = shares != 0.0
        marked_holdings = float(np.sum(shares[held_now] * marks[held_now]))
        current_nav = (
            _equity(
                free_cash,
                restricted_by_name,
                shares,
                marks,
                receivable_by_name,
                payable_by_name,
            )
            + hedge_restricted_cash
            + (hedge_shares * hedge_mark if hedge_shares != 0.0 else 0.0)
        )
        identity = (
            free_cash
            + restricted_by_name.sum()
            + hedge_restricted_cash
            + marked_holdings
            + (hedge_shares * hedge_mark if hedge_shares != 0.0 else 0.0)
            + receivable_by_name.sum()
            - payable_by_name.sum()
        )
        reconciliation = current_nav - identity
        all_cash *= 1.0 + inputs.cdi[day]

        unresolved = held_now & (
            explicit_unresolved_action
            | (missing_sessions >= config.settlement_grace_sessions)
            | (day == day_count - 1)
        )
        newly_scenario = unresolved & ~scenario_seen
        scenario_seen |= unresolved
        settlement_haircut_nav = current_nav + settlement_scenario_adjustment
        excluded_nav = current_nav
        if unresolved.any():
            excluded_nav -= float(restricted_by_name[unresolved].sum())
            excluded_nav -= float(np.sum(shares[unresolved] * marks[unresolved]))
        unresolved_claim = (receivable_by_name != 0.0) | (payable_by_name != 0.0)
        excluded_names = unresolved | unresolved_claim
        if excluded_names.any():
            excluded_nav -= float(receivable_by_name[excluded_names].sum())
            excluded_nav += float(payable_by_name[excluded_names].sum())

        signed_values = np.zeros(name_count, dtype=np.float64)
        signed_values[held_now] = shares[held_now] * marks[held_now]
        gross, net, name_weight = _risk(signed_values, current_nav)
        hedge_notional = hedge_shares * hedge_mark if hedge_shares != 0.0 else 0.0
        gross_including_hedge = gross + abs(hedge_notional) / current_nav
        net_notional_including_hedge = float(signed_values.sum() + hedge_notional)
        net_including_hedge = net_notional_including_hedge / current_nav
        ex_ante_before = equity_beta_notional / start_nav
        ex_ante_after = (equity_beta_notional + decision_hedge_notional) / start_nav
        unresolved_stale = held_now & (
            ~printed
            | explicit_unresolved_action
            | (missing_sessions >= config.settlement_grace_sessions)
        )
        unresolved_claim_inventory = held_now & explicit_unresolved_action
        stale_mark_inventory = held_now & ~printed
        unresolved_stale_fraction = (
            float(np.abs(signed_values[unresolved_stale]).sum() / current_nav)
            if current_nav != 0.0
            else np.nan
        )
        unresolved_claim_fraction = (
            float(np.abs(signed_values[unresolved_claim_inventory]).sum() / current_nav)
            if current_nav != 0.0
            else np.nan
        )
        stale_mark_fraction = (
            float(np.abs(signed_values[stale_mark_inventory]).sum() / current_nav)
            if current_nav != 0.0
            else np.nan
        )
        end_risk_breach = (
            gross > config.planned_gross_cap + 1e-12
            or abs(net) > config.planned_absolute_net_cap + 1e-12
            or name_weight > config.planned_name_weight_cap + 1e-12
        )
        planned_values = signed_values.copy()
        for name, pending in pending_entries.items():
            sign = 1.0 if pending.order.side == "buy" else -1.0
            planned_values[name] += sign * pending.remaining_size
        planned_gross, planned_net, planned_name = _risk(planned_values, start_nav)

        if config.volatility_balanced_entries:
            long_groups = volatility_groups[(shares > 0.0) & (volatility_groups >= 0)]
            short_groups = volatility_groups[(shares < 0.0) & (volatility_groups >= 0)]
            long_volatility_occupancy = np.bincount(
                long_groups,
                minlength=config.volatility_group_count,
            )[: config.volatility_group_count]
            short_volatility_occupancy = np.bincount(
                short_groups,
                minlength=config.volatility_group_count,
            )[: config.volatility_group_count]
        else:
            long_volatility_occupancy = np.zeros(
                config.volatility_group_count, dtype=np.int64
            )
            short_volatility_occupancy = np.zeros(
                config.volatility_group_count, dtype=np.int64
            )

        held_end_long = int((shares > 0.0).sum())
        held_end_short = int((shares < 0.0).sum())
        target_per_slot = config.gross_target / (2.0 * config.k_per_side)
        small_universe_shortfall = (
            config.gross_target * (config.k_per_side - k_eff) / config.k_per_side
        )
        occupancy_slot_count = 2 * k_eff - int(held_now.sum())
        empty_pending_count = sum(shares[name] == 0.0 for name in pending_entries)
        occupancy_pending_count = min(max(occupancy_slot_count, 0), empty_pending_count)
        unclassified_occupancy = occupancy_slot_count - occupancy_pending_count
        occupancy_band_exhausted_count = min(
            max(unclassified_occupancy, 0),
            band_exhausted_long + band_exhausted_short,
        )
        unclassified_occupancy -= occupancy_band_exhausted_count
        occupancy_blocked_count = min(
            max(unclassified_occupancy, 0), blocked_open_long + blocked_open_short
        )
        unclassified_occupancy -= occupancy_blocked_count
        occupancy_exit_gap_count = min(
            max(unclassified_occupancy, 0),
            exit_fill_long_today + exit_fill_short_today,
        )
        unclassified_occupancy -= occupancy_exit_gap_count
        sizing_fill = float(
            np.sum(
                submission_nav[held_now] * target_per_slot - entry_cost_basis[held_now],
                dtype=np.float64,
            )
            / current_nav
        )
        sizing_mark = float(
            np.sum(
                entry_cost_basis[held_now] - np.abs(signed_values[held_now]),
                dtype=np.float64,
            )
            / current_nav
        )
        sizing_nav = float(
            np.sum(
                target_per_slot * (current_nav - submission_nav[held_now]),
                dtype=np.float64,
            )
            / current_nav
        )

        economic_pnl = current_nav - start_nav - interest + borrow + costs
        equity_gross_pnl = economic_pnl - hedge_gross_pnl
        daily_return = current_nav / start_nav - 1.0
        start_nav_rows.append(start_nav)
        nav_rows.append(current_nav)
        daily_rows.append(daily_return)
        excess_rows.append(10_000.0 * (daily_return - inputs.cdi[day]))
        gross_pnl_rows.append(10_000.0 * economic_pnl / start_nav)
        equity_gross_pnl_rows.append(10_000.0 * equity_gross_pnl / start_nav)
        hedge_gross_pnl_rows.append(10_000.0 * hedge_gross_pnl / start_nav)
        interest_rows.append(10_000.0 * interest / start_nav)
        free_cash_interest_rows.append(10_000.0 * free_cash_interest / start_nav)
        short_proceeds_interest_rows.append(
            10_000.0 * short_proceeds_interest / start_nav
        )
        short_proceeds_interest_base_rows.append(short_proceeds_interest_base)
        cost_rows.append(10_000.0 * costs / start_nav)
        borrow_rows.append(10_000.0 * borrow / start_nav)
        equity_borrow_raw_rows.append(10_000.0 * equity_borrow_raw / start_nav)
        equity_borrow_fee_rows.append(10_000.0 * equity_borrow_fee / start_nav)
        cdi_benchmark_rows.append(10_000.0 * inputs.cdi[day])
        gross_rows.append(gross if np.isfinite(gross) else np.nan)
        gross_including_hedge_rows.append(gross_including_hedge)
        net_notional_including_hedge_rows.append(net_notional_including_hedge)
        net_including_hedge_rows.append(net_including_hedge)
        turnover_rows.append(traded_notional / start_nav)
        volatility_quota_rows.append(volatility_quota.copy())
        volatility_group_size_rows.append(group_sizes.copy())
        entry_eligible_name_count_rows.append(int(entry_eligible.sum()))
        volatility_occupancy_long_rows.append(long_volatility_occupancy)
        volatility_occupancy_short_rows.append(short_volatility_occupancy)
        volatility_spilled_long_rows.append(spilled_entries["buy"])
        volatility_spilled_short_rows.append(spilled_entries["sell"])
        hedge_share_rows.append(hedge_shares)
        hedge_mark_rows.append(hedge_mark)
        hedge_notional_rows.append(hedge_notional)
        hedge_unconstrained_target_rows.append(hedge_unconstrained_target_notional)
        hedge_target_rows.append(hedge_target_notional)
        hedge_capped_rows.append(hedge_capped)
        hedge_turnover_rows.append(hedge_traded_notional / start_nav)
        hedge_cost_rows.append(10_000.0 * hedge_cost / start_nav)
        hedge_borrow_rows.append(10_000.0 * hedge_borrow / start_nav)
        ex_ante_beta_before_rows.append(ex_ante_before)
        ex_ante_beta_after_rows.append(ex_ante_after)
        unresolved_stale_fraction_rows.append(unresolved_stale_fraction)
        unresolved_claim_fraction_rows.append(unresolved_claim_fraction)
        stale_mark_fraction_rows.append(stale_mark_fraction)
        stale_rows.append(stale)
        unresolved_action_rows.append(int(unresolved_claim_inventory.sum()))
        scenario_count_rows.append(int(newly_scenario.sum()))
        signs = np.zeros(name_count, dtype=np.int8)
        signs[shares > 0.0] = 1
        signs[shares < 0.0] = -1
        position_rows.append(signs)
        share_rows.append(shares.copy())
        mark_rows.append(marks.copy())
        free_cash_rows.append(free_cash)
        restricted_rows.append(float(restricted_by_name.sum()))
        hedge_restricted_rows.append(hedge_restricted_cash)
        receivable_rows.append(float(receivable_by_name.sum()))
        payable_rows.append(float(payable_by_name.sum()))
        holding_value_rows.append(marked_holdings)
        reconciliation_rows.append(reconciliation)
        all_cash_rows.append(all_cash)
        haircut_rows.append(settlement_haircut_nav)
        excluded_rows.append(excluded_nav)
        pending_entry_rows.append(len(pending_entries))
        pending_exit_rows.append(len(pending_exits))
        pending_exit_age_rows.append(
            float(
                np.mean(
                    [
                        day - pending.order.decision_session + 1
                        for pending in pending_exits.values()
                    ]
                )
            )
            if pending_exits
            else np.nan
        )
        cancelled_rows.append(cancelled_today)
        risk_breach_rows.append(end_risk_breach)
        planned_gross_rows.append(planned_gross)
        planned_net_rows.append(planned_net)
        planned_name_rows.append(planned_name)
        ages = np.zeros(name_count, dtype=np.int64)
        held_with_age = held_now & (entry_session >= 0)
        ages[held_with_age] = day - entry_session[held_with_age] + 1
        age_rows.append(ages)
        ineligible_streak_rows.append(ineligible_streak.copy())
        settlement_count_rows.append(settlement_count)
        settlement_notional_rows.append(settlement_notional)
        settlement_fraction_rows.append(settlement_notional / start_nav)
        settled_then_printed_rows.append(settled_then_printed_today)
        held_end_long_rows.append(held_end_long)
        held_end_short_rows.append(held_end_short)
        pending_end_long_rows.append(pending_end_long)
        pending_end_short_rows.append(pending_end_short)
        pending_without_print_long_rows.append(pending_without_print_long)
        pending_without_print_short_rows.append(pending_without_print_short)
        exit_fill_long_rows.append(exit_fill_long_today)
        exit_fill_short_rows.append(exit_fill_short_today)
        pending_printed_unfilled_rows.append(pending_printed_unfilled_today)
        entry_fill_short_rows.append(entry_fill_short_today)
        entry_cost_basis_rows.append(entry_cost_basis.copy())
        submission_nav_rows.append(submission_nav.copy())
        shortfall_small_rows.append(small_universe_shortfall)
        shortfall_pending_rows.append(occupancy_pending_count * target_per_slot)
        shortfall_band_exhausted_rows.append(
            occupancy_band_exhausted_count * target_per_slot
        )
        shortfall_blocked_rows.append(occupancy_blocked_count * target_per_slot)
        shortfall_exit_gap_rows.append(occupancy_exit_gap_count * target_per_slot)
        shortfall_other_rows.append(unclassified_occupancy * target_per_slot)
        shortfall_sizing_fill_rows.append(sizing_fill)
        shortfall_sizing_mark_rows.append(sizing_mark)
        shortfall_sizing_nav_rows.append(sizing_nav)

        previous_nav = current_nav
        if not np.isfinite(current_nav):
            raise ValueError("ledger produced non-finite equity")
        if current_nav <= 0.0:
            insolvent = True
            insolvency_date = inputs.dates[day]
            break

    completed_days = len(nav_rows)
    final_shares = share_rows[-1]
    final_marks = mark_rows[-1]
    unresolved_inventory = final_shares != 0.0
    unresolved_count = int(unresolved_inventory.sum())
    unresolved_notional = float(
        np.abs(
            final_shares[unresolved_inventory] * final_marks[unresolved_inventory]
        ).sum()
    )
    terminal_reason_masks = {
        "no_terminal_print": unresolved_inventory & ~terminal_printed,
        "settlement_grace_sessions": unresolved_inventory
        & (missing_sessions >= config.settlement_grace_sessions),
        "unresolved_action": unresolved_inventory & explicit_unresolved_action,
        "prior_pending_exit": unresolved_inventory & terminal_prior_pending_exit,
    }
    terminal_reason_breakdown: dict[str, dict[str, int | float]] = {}
    for reason, reason_mask in terminal_reason_masks.items():
        terminal_reason_breakdown[reason] = {
            "count": int(reason_mask.sum()),
            "notional": float(
                np.abs(final_shares[reason_mask] * final_marks[reason_mask]).sum()
            ),
        }
    classified = np.logical_or.reduce(tuple(terminal_reason_masks.values()))
    other = unresolved_inventory & ~classified
    terminal_reason_breakdown["other"] = {
        "count": int(other.sum()),
        "notional": float(np.abs(final_shares[other] * final_marks[other]).sum()),
    }
    gross_array = np.asarray(gross_rows, dtype=np.float64)
    finite_gross = gross_array[np.isfinite(gross_array)]
    mean_gross = float(np.mean(finite_gross)) if finite_gross.size else 0.0
    settlement_fraction = float(np.sum(settlement_fraction_rows))
    economics_unresolved = (
        insolvent
        or action_uncertainty_seen
        or unresolved_count > 0
        or receivable_by_name.any()
        or payable_by_name.any()
        or hedge_shares != 0.0
        or terminal_boundary_unpriced_inventory_notional > 0.0
        or terminal_hedge_last_mark_settlement_notional > 0.0
        or settlement_fraction > config.settlement_economics_unresolved_fraction_nav
        or mean_gross < 0.5 * config.gross_target
    )
    exit_cause_array = np.stack(exit_cause_rows).astype(np.int8, copy=False)
    exit_side_array = np.stack(exit_side_rows).astype(np.int8, copy=False)
    nonterminal_exit_count = int(
        ((exit_cause_array >= 1) & (exit_cause_array <= 3)).sum()
    )
    reeligible_within_ten_count = 0
    for exit_day, name in np.argwhere(exit_cause_array == 1):
        side = int(exit_side_array[exit_day, name])
        for later_day in range(
            int(exit_day) + 1, min(int(exit_day) + 11, completed_days)
        ):
            later_eligible = (
                inputs.score_valid[later_day]
                & inputs.membership[later_day]
                & np.isfinite(inputs.score[later_day])
            )
            later_names = np.flatnonzero(later_eligible)
            later_order = (
                later_names[
                    np.argsort(inputs.score[later_day, later_names], kind="stable")
                ]
                if later_names.size
                else later_names
            )
            if config.volatility_balanced_entries:
                later_group_eligible = later_eligible & np.isfinite(
                    inputs.selection_volatility[later_day]
                )
                later_entry_eligible = later_group_eligible.copy()
                later_k_eff = min(
                    config.k_per_side, int(later_entry_eligible.sum()) // 2
                )
                later_groups = _equal_count_groups(
                    inputs.selection_volatility[later_day],
                    later_group_eligible,
                    config.volatility_group_count,
                )
                later_sizes = np.bincount(
                    later_groups[later_groups >= 0],
                    minlength=config.volatility_group_count,
                )[: config.volatility_group_count]
                later_quota, later_buffer = _scaled_group_bands(
                    k_eff=later_k_eff,
                    buffer=config.buffer_per_side,
                    group_sizes=later_sizes,
                    threshold_multiple=(
                        config.small_stratum_scaling_threshold_multiple
                    ),
                    capacity_buffer=capacity_buffer_per_side,
                )
                later_group = int(later_groups[name])
                if later_group < 0:
                    continue
                group_names = np.flatnonzero(later_groups == later_group)
                group_order = group_names[
                    np.argsort(inputs.score[later_day, group_names], kind="stable")
                ]
                width = int(later_quota[later_group] + later_buffer[later_group])
                inside_retention = bool(
                    width
                    and (
                        name in group_order[-width:]
                        if side > 0
                        else name in group_order[:width]
                    )
                )
            else:
                later_retention = min(
                    config.k_per_side + config.buffer_per_side,
                    len(later_order) // 2,
                )
                if not later_eligible[name] or later_retention == 0:
                    continue
                later_rank = int(np.flatnonzero(later_order == name)[0])
                inside_retention = (
                    later_rank >= len(later_order) - later_retention
                    if side > 0
                    else later_rank < later_retention
                )
            if inside_retention:
                reeligible_within_ten_count += 1
                break
    reeligible_within_ten_share = (
        reeligible_within_ten_count / nonterminal_exit_count
        if nonterminal_exit_count
        else 0.0
    )
    return StatefulLedgerResult(
        dates=inputs.dates[:completed_days],
        start_nav=np.asarray(start_nav_rows, dtype=np.float64),
        nav=np.asarray(nav_rows, dtype=np.float64),
        daily_net_return=np.asarray(daily_rows, dtype=np.float64),
        net_excess_all_cash_bps=np.asarray(excess_rows, dtype=np.float64),
        gross_pnl_bps=np.asarray(gross_pnl_rows, dtype=np.float64),
        equity_gross_pnl_bps=np.asarray(equity_gross_pnl_rows, dtype=np.float64),
        hedge_gross_pnl_bps=np.asarray(hedge_gross_pnl_rows, dtype=np.float64),
        interest_bps=np.asarray(interest_rows, dtype=np.float64),
        free_cash_interest_bps=np.asarray(free_cash_interest_rows, dtype=np.float64),
        short_proceeds_interest_bps=np.asarray(
            short_proceeds_interest_rows, dtype=np.float64
        ),
        short_proceeds_interest_base=np.asarray(
            short_proceeds_interest_base_rows, dtype=np.float64
        ),
        cost_bps=np.asarray(cost_rows, dtype=np.float64),
        borrow_bps=np.asarray(borrow_rows, dtype=np.float64),
        equity_borrow_raw_bps=np.asarray(equity_borrow_raw_rows, dtype=np.float64),
        equity_borrow_fee_bps=np.asarray(equity_borrow_fee_rows, dtype=np.float64),
        cdi_benchmark_bps=np.asarray(cdi_benchmark_rows, dtype=np.float64),
        held_short_weighted_annual_borrow_rate=np.asarray(
            held_short_borrow_rate_rows, dtype=np.float64
        ),
        held_short_notional_at_open=np.asarray(
            held_short_notional_rows, dtype=np.float64
        ),
        held_short_imputed_notional_at_open=np.asarray(
            held_short_imputed_notional_rows, dtype=np.float64
        ),
        held_short_placeholder_notional_at_open=np.asarray(
            held_short_placeholder_notional_rows, dtype=np.float64
        ),
        excluded_short_entry_candidate_count=np.asarray(
            excluded_short_candidate_rows, dtype=np.int64
        ),
        gross_fraction_nav=gross_array,
        turnover_fraction_nav=np.asarray(turnover_rows, dtype=np.float64),
        unresolved_stale_inventory_fraction_nav=np.asarray(
            unresolved_stale_fraction_rows, dtype=np.float64
        ),
        unresolved_claim_inventory_fraction_nav=np.asarray(
            unresolved_claim_fraction_rows, dtype=np.float64
        ),
        stale_mark_inventory_fraction_nav=np.asarray(
            stale_mark_fraction_rows, dtype=np.float64
        ),
        stale_mark_name_days=np.asarray(stale_rows, dtype=np.int64),
        same_day_action_sessions=np.asarray(same_day_action_rows, dtype=np.bool_),
        unresolved_action_name_days=np.asarray(unresolved_action_rows, dtype=np.int64),
        valuation_scenario_count=np.asarray(scenario_count_rows, dtype=np.int64),
        position_sign=np.stack(position_rows),
        signed_shares=np.stack(share_rows),
        mark_price=np.stack(mark_rows),
        free_cash=np.asarray(free_cash_rows, dtype=np.float64),
        restricted_cash=np.asarray(restricted_rows, dtype=np.float64),
        hedge_restricted_cash=np.asarray(hedge_restricted_rows, dtype=np.float64),
        receivables=np.asarray(receivable_rows, dtype=np.float64),
        payables=np.asarray(payable_rows, dtype=np.float64),
        marked_signed_holdings=np.asarray(holding_value_rows, dtype=np.float64),
        reconciliation_error=np.asarray(reconciliation_rows, dtype=np.float64),
        all_cash_nav=np.asarray(all_cash_rows, dtype=np.float64),
        settlement_haircut_scenario_nav=np.asarray(haircut_rows, dtype=np.float64),
        unresolved_excluded_nav=np.asarray(excluded_rows, dtype=np.float64),
        pending_entry_count=np.asarray(pending_entry_rows, dtype=np.int64),
        pending_exit_count=np.asarray(pending_exit_rows, dtype=np.int64),
        cancelled_entry_count=np.asarray(cancelled_rows, dtype=np.int64),
        blocked_entry_no_reference_count=np.asarray(
            blocked_reference_rows[:completed_days], dtype=np.int64
        ),
        blocked_entry_gross_cap_count=np.asarray(
            blocked_gross_rows[:completed_days], dtype=np.int64
        ),
        blocked_entry_net_cap_count=np.asarray(
            blocked_net_rows[:completed_days], dtype=np.int64
        ),
        blocked_entry_name_cap_count=np.asarray(
            blocked_name_rows[:completed_days], dtype=np.int64
        ),
        submitted_entry_count=np.asarray(
            submitted_entry_rows[:completed_days], dtype=np.int64
        ),
        submitted_exit_count=np.asarray(
            submitted_exit_rows[:completed_days], dtype=np.int64
        ),
        same_close_replacement_count=np.asarray(
            same_close_replacement_rows[:completed_days], dtype=np.int64
        ),
        zero_entry_small_universe=np.asarray(
            zero_entry_small_rows[:completed_days], dtype=np.bool_
        ),
        zero_entry_gross_cap=np.asarray(
            zero_entry_gross_rows[:completed_days], dtype=np.bool_
        ),
        zero_entry_net_cap=np.asarray(
            zero_entry_net_rows[:completed_days], dtype=np.bool_
        ),
        zero_entry_name_cap=np.asarray(
            zero_entry_name_rows[:completed_days], dtype=np.bool_
        ),
        zero_entry_no_reference=np.asarray(
            zero_entry_reference_rows[:completed_days], dtype=np.bool_
        ),
        risk_trim_gross_notional=np.asarray(
            risk_trim_gross_rows[:completed_days], dtype=np.float64
        ),
        risk_trim_net_notional=np.asarray(
            risk_trim_net_rows[:completed_days], dtype=np.float64
        ),
        risk_trim_name_notional=np.asarray(
            risk_trim_name_rows[:completed_days], dtype=np.float64
        ),
        pending_exit_mean_age_sessions=np.asarray(
            pending_exit_age_rows[:completed_days], dtype=np.float64
        ),
        retention_width=np.asarray(retention_rows[:completed_days], dtype=np.int64),
        entry_blocked_small_universe=np.asarray(
            small_universe_rows[:completed_days], dtype=np.bool_
        ),
        actual_risk_breach=np.asarray(risk_breach_rows, dtype=np.bool_),
        planned_gross_fraction_nav=np.asarray(planned_gross_rows, dtype=np.float64),
        planned_net_fraction_nav=np.asarray(planned_net_rows, dtype=np.float64),
        planned_name_weight_fraction_nav=np.asarray(
            planned_name_rows, dtype=np.float64
        ),
        holding_age_sessions=np.stack(age_rows),
        terminal_settlement_count=np.asarray(settlement_count_rows, dtype=np.int64),
        terminal_settlement_notional=np.asarray(
            settlement_notional_rows, dtype=np.float64
        ),
        terminal_settlement_notional_fraction_nav=np.asarray(
            settlement_fraction_rows, dtype=np.float64
        ),
        settled_then_printed_count=np.asarray(
            settled_then_printed_rows, dtype=np.int64
        ),
        k_eff_per_side=np.asarray(k_eff_rows, dtype=np.int64),
        held_count_start_of_day_long=np.asarray(held_start_long_rows, dtype=np.int64),
        held_count_start_of_day_short=np.asarray(held_start_short_rows, dtype=np.int64),
        held_count_end_of_day_long=np.asarray(held_end_long_rows, dtype=np.int64),
        held_count_end_of_day_short=np.asarray(held_end_short_rows, dtype=np.int64),
        occupied_after_submission_long=np.asarray(
            occupied_after_submission_long_rows, dtype=np.int64
        ),
        occupied_after_submission_short=np.asarray(
            occupied_after_submission_short_rows, dtype=np.int64
        ),
        open_slots_after_submission_long=np.asarray(
            open_after_submission_long_rows, dtype=np.int64
        ),
        open_slots_after_submission_short=np.asarray(
            open_after_submission_short_rows, dtype=np.int64
        ),
        band_candidates_long=np.asarray(band_candidates_long_rows, dtype=np.int64),
        band_candidates_short=np.asarray(band_candidates_short_rows, dtype=np.int64),
        band_excluded_unresolved_long=np.asarray(
            band_excluded_unresolved_long_rows, dtype=np.int64
        ),
        band_excluded_unresolved_short=np.asarray(
            band_excluded_unresolved_short_rows, dtype=np.int64
        ),
        band_excluded_settled_long=np.asarray(
            band_excluded_settled_long_rows, dtype=np.int64
        ),
        band_excluded_settled_short=np.asarray(
            band_excluded_settled_short_rows, dtype=np.int64
        ),
        band_candidates_without_prior_session_print_long=np.asarray(
            band_without_prior_print_long_rows, dtype=np.int64
        ),
        band_candidates_without_prior_session_print_short=np.asarray(
            band_without_prior_print_short_rows, dtype=np.int64
        ),
        band_exhausted_open_slots_long=np.asarray(
            band_exhausted_long_rows, dtype=np.int64
        ),
        band_exhausted_open_slots_short=np.asarray(
            band_exhausted_short_rows, dtype=np.int64
        ),
        blocked_open_slots_long=np.asarray(blocked_open_long_rows, dtype=np.int64),
        blocked_open_slots_short=np.asarray(blocked_open_short_rows, dtype=np.int64),
        pending_entries_end_of_day_long=np.asarray(
            pending_end_long_rows, dtype=np.int64
        ),
        pending_entries_end_of_day_short=np.asarray(
            pending_end_short_rows, dtype=np.int64
        ),
        pending_entries_without_print_today_long=np.asarray(
            pending_without_print_long_rows, dtype=np.int64
        ),
        pending_entries_without_print_today_short=np.asarray(
            pending_without_print_short_rows, dtype=np.int64
        ),
        slots_freed_by_exit_fill_today_long=np.asarray(
            exit_fill_long_rows, dtype=np.int64
        ),
        slots_freed_by_exit_fill_today_short=np.asarray(
            exit_fill_short_rows, dtype=np.int64
        ),
        exit_instructions_ineligible_hold_exhausted=np.asarray(
            exit_ineligible_hold_exhausted_rows, dtype=np.int64
        ),
        exit_instructions_settlement_grace=np.asarray(
            exit_settlement_rows, dtype=np.int64
        ),
        exit_instructions_rank_out_of_retention=np.asarray(
            exit_rank_rows, dtype=np.int64
        ),
        exit_instructions_terminal=np.asarray(exit_terminal_rows, dtype=np.int64),
        exit_instruction_cause=exit_cause_array,
        exit_instruction_side=exit_side_array,
        ineligible_streak=np.stack(ineligible_streak_rows),
        ineligible_held_sessions_score_invalid=np.asarray(
            ineligible_score_invalid_rows, dtype=np.int64
        ),
        ineligible_held_sessions_membership_invalid=np.asarray(
            ineligible_membership_invalid_rows, dtype=np.int64
        ),
        ineligible_held_sessions_score_nonfinite=np.asarray(
            ineligible_score_nonfinite_rows, dtype=np.int64
        ),
        ineligible_exit_within_hold_window=np.asarray(
            ineligible_exit_within_hold_rows, dtype=np.int64
        ),
        net_cap_block_with_balanced_book=np.asarray(
            net_cap_balanced_rows, dtype=np.int64
        ),
        gross_cap_block_below_target=np.asarray(
            gross_cap_below_target_rows, dtype=np.int64
        ),
        name_cap_block_on_fresh_entry=np.asarray(name_cap_fresh_rows, dtype=np.int64),
        entry_pending_printed_unblocked_unfilled=np.asarray(
            pending_printed_unfilled_rows, dtype=np.int64
        ),
        entry_fill_quantity_short=np.asarray(entry_fill_short_rows, dtype=np.int64),
        entry_cost_basis=np.stack(entry_cost_basis_rows),
        submission_nav=np.stack(submission_nav_rows),
        gross_shortfall_small_universe=np.asarray(
            shortfall_small_rows, dtype=np.float64
        ),
        gross_shortfall_occupancy_pending=np.asarray(
            shortfall_pending_rows, dtype=np.float64
        ),
        gross_shortfall_occupancy_band_exhausted=np.asarray(
            shortfall_band_exhausted_rows, dtype=np.float64
        ),
        gross_shortfall_occupancy_blocked=np.asarray(
            shortfall_blocked_rows, dtype=np.float64
        ),
        gross_shortfall_occupancy_exit_gap=np.asarray(
            shortfall_exit_gap_rows, dtype=np.float64
        ),
        gross_shortfall_occupancy_other=np.asarray(
            shortfall_other_rows, dtype=np.float64
        ),
        gross_shortfall_sizing_fill=np.asarray(
            shortfall_sizing_fill_rows, dtype=np.float64
        ),
        gross_shortfall_sizing_mark_drift=np.asarray(
            shortfall_sizing_mark_rows, dtype=np.float64
        ),
        gross_shortfall_sizing_nav_drift=np.asarray(
            shortfall_sizing_nav_rows, dtype=np.float64
        ),
        ineligible_exit_reeligible_within_10_sessions_share=(
            reeligible_within_ten_share
        ),
        intended_orders=tuple(orders),
        fills=tuple(fills),
        cancellations=tuple(cancellations),
        unresolved_inventory_count=unresolved_count,
        unresolved_inventory_notional=unresolved_notional,
        terminal_unresolved_reason_breakdown=terminal_reason_breakdown,
        unresolved_receivable=float(receivable_by_name.sum()),
        unresolved_payable=float(payable_by_name.sum()),
        insolvent=insolvent,
        insolvency_date=insolvency_date,
        economics_unresolved=economics_unresolved,
        terminal_boundary_unpriced_inventory_notional=terminal_boundary_unpriced_inventory_notional,
        terminal_hedge_last_mark_settlement_notional=terminal_hedge_last_mark_settlement_notional,
        gross_target=config.gross_target,
        ineligible_hold_sessions=config.ineligible_hold_sessions,
        settlement_grace_sessions=config.settlement_grace_sessions,
        settlement_haircut=config.settlement_haircut,
        settlement_economics_unresolved_fraction_nav=(
            config.settlement_economics_unresolved_fraction_nav
        ),
        share_sizing_mode="notional_entries_position_fraction_exits",
        borrow_source=config.borrow_source,
        volatility_balanced_entries=config.volatility_balanced_entries,
        beta_hedge=config.beta_hedge,
        hedge_notional_cap_nav=config.hedge_notional_cap_nav,
        volatility_quota=np.stack(volatility_quota_rows),
        volatility_group_size=np.stack(volatility_group_size_rows),
        entry_eligible_name_count=np.asarray(
            entry_eligible_name_count_rows, dtype=np.int64
        ),
        volatility_occupancy_long=np.stack(volatility_occupancy_long_rows),
        volatility_occupancy_short=np.stack(volatility_occupancy_short_rows),
        volatility_spilled_entries_long=np.stack(volatility_spilled_long_rows),
        volatility_spilled_entries_short=np.stack(volatility_spilled_short_rows),
        hedge_signed_shares=np.asarray(hedge_share_rows, dtype=np.float64),
        hedge_mark_price=np.asarray(hedge_mark_rows, dtype=np.float64),
        hedge_signed_notional=np.asarray(hedge_notional_rows, dtype=np.float64),
        hedge_unconstrained_target_notional=np.asarray(
            hedge_unconstrained_target_rows, dtype=np.float64
        ),
        hedge_target_notional=np.asarray(hedge_target_rows, dtype=np.float64),
        hedge_beta_fallback_sessions=np.asarray(
            hedge_beta_fallback_rows, dtype=np.bool_
        ),
        hedge_capped=np.asarray(hedge_capped_rows, dtype=np.bool_),
        hedge_turnover_fraction_nav=np.asarray(hedge_turnover_rows, dtype=np.float64),
        hedge_cost_bps=np.asarray(hedge_cost_rows, dtype=np.float64),
        hedge_borrow_bps=np.asarray(hedge_borrow_rows, dtype=np.float64),
        ex_ante_beta_before_hedge=np.asarray(
            ex_ante_beta_before_rows, dtype=np.float64
        ),
        ex_ante_beta_after_hedge=np.asarray(ex_ante_beta_after_rows, dtype=np.float64),
        gross_fraction_nav_including_hedge=np.asarray(
            gross_including_hedge_rows, dtype=np.float64
        ),
        net_notional_including_hedge=np.asarray(
            net_notional_including_hedge_rows, dtype=np.float64
        ),
        net_fraction_nav_including_hedge=np.asarray(
            net_including_hedge_rows, dtype=np.float64
        ),
    )


def ledger_sensitivity_grid(
    *,
    shortable_by_borrow_source: Mapping[BorrowSource, NDArray[np.bool_]],
    headline_config: LedgerConfig | None = None,
    **inputs: object,
) -> dict[str, StatefulLedgerResult]:
    """Run the registered financing/cost grid and structural sensitivities."""

    required: set[BorrowSource] = {
        "borrow_strict",
        "borrow_balance",
        "borrow_open",
    }
    if not required.issubset(shortable_by_borrow_source):
        raise ValueError(
            "rev4b ledger grid requires all three borrow availability cells"
        )
    results = {}
    for name, config in ledger_configurations(headline_config).items():
        shortable = (
            np.ones_like(next(iter(shortable_by_borrow_source.values())))
            if config.borrow_source == "uniform"
            else shortable_by_borrow_source[config.borrow_source]
        )
        results[name] = simulate_stateful_ledger(  # type: ignore[arg-type]
            config=config,
            shortable=shortable,
            **inputs,
        )
    return results


def ledger_configurations(
    headline_config: LedgerConfig | None = None,
) -> dict[str, LedgerConfig]:
    """Stress costs and borrow availability on the same constructed, hedged book."""

    headline = headline_config if headline_config is not None else LedgerConfig()
    configurations = {
        (
            f"borrow_{borrow}" if cost == 4.0 else f"cost_{cost:g}_borrow_{borrow}"
        ): replace(headline, cost_bps_per_side=cost, borrow_source=f"borrow_{borrow}")
        for cost in (2.0, 4.0, 7.0)
        for borrow in ("balance", "strict", "open")
    }
    configurations.update(
        {
            "sensitivity_buffer_0": replace(headline, buffer_per_side=0),
            "sensitivity_buffer_2k": replace(
                headline, buffer_per_side=2 * headline.k_per_side
            ),
            "comparator_sterile_proceeds": replace(
                headline, short_proceeds_remuneration=0.0
            ),
            "comparator_uniform_borrow": replace(
                headline,
                cost_bps_per_side=4.0,
                annual_borrow_rate=0.02,
                borrow_source="uniform",
            ),
        }
    )
    return configurations
