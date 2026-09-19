from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from typing import Callable, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from brazil_rv.v2.corporate_actions import AlignedActionTerms, apply_contractual_action
from brazil_rv.v2.hedge_beta import resolve_hedge_beta
from .share_distributions import (
    ShareDistribution,
    ShareClaimPosition,
    basket_prices,
    basket_betas,
)
from .loan_fees import LoanModality, loan_fee_rates
from .share_custody import ShareCustody
from .loan_contracts import (
    LoanCharge,
    LoanCashSettlement,
    LoanCashPayment,
    LoanContracts,
    LoanSession,
    spot_settlement_session,
)


OrderSide = Literal["buy", "sell"]
BorrowSource = Literal["uniform", "borrow_strict", "borrow_balance", "borrow_open"]
OrderPurpose = Literal["entry", "exit", "risk_exit", "terminal_exit", "hedge"]
ExitInstructionCause = Literal[
    "ineligible_hold_exhausted",
    "settlement_grace",
    "rank_out_of_retention",
    "terminal",
    "policy_rebalance",
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
    "policy_replaced",
    "corporate_action_netting",
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
    "policy_replaced",
    "corporate_action_netting",
)

MISSING_QUOTE_CONVENTION = "retain_inventory_until_quote_or_contractual_event"
EXIT_INSTRUCTION_CAUSE_CODES: dict[ExitInstructionCause, int] = {
    "ineligible_hold_exhausted": 1,
    "settlement_grace": 2,
    "rank_out_of_retention": 3,
    "terminal": 4,
    "policy_rebalance": 5,
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
    planned_absolute_beta_cap: float = 0.05
    planned_name_weight_cap: float = 0.05
    cost_bps_per_side: float = 4.0
    annual_borrow_rate: float = 0.02
    borrow_source: BorrowSource = "borrow_balance"
    borrow_fee_modality: LoanModality = "normal"
    borrow_fee_multiplier: float = 1.0
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
    unpriced_haircut: float = 0.30
    unpriced_economics_unresolved_fraction_nav: float = 0.15
    annual_sessions: int = 252

    def __post_init__(self) -> None:
        if self.k_per_side <= 0 or self.buffer_per_side < 0:
            raise ValueError("ledger K must be positive and its buffer non-negative")
        if self.gross_target <= 0 or self.cost_bps_per_side < 0:
            raise ValueError("ledger gross/cost controls are invalid")
        if (
            self.planned_gross_cap <= 0
            or self.planned_absolute_net_cap < 0
            or self.planned_absolute_beta_cap < 0
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
        if self.borrow_fee_modality not in {"normal", "direct", "otc", "compulsory"}:
            raise ValueError("unknown B3 loan modality")
        if self.borrow_fee_multiplier < 0 or not np.isfinite(
            self.borrow_fee_multiplier
        ):
            raise ValueError("borrow fee multiplier must be finite and non-negative")
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
        if not 0 <= self.unpriced_haircut < 1:
            raise ValueError("unpriced haircut must be in [0, 1)")
        if self.unpriced_economics_unresolved_fraction_nav <= 0:
            raise ValueError("unpriced exposure bound must be positive")


def daily_borrow_cost(annual_rate, dates, *, config: LedgerConfig):
    """One-session rent/fee estimate; each exchange component compounds alone."""
    fees = loan_fee_rates(annual_rate, dates, modality=config.borrow_fee_modality)
    return np.expm1(np.log1p(annual_rate) / config.annual_sessions) + (
        np.expm1(np.log1p(fees) / config.annual_sessions).sum(axis=-1)
        * config.borrow_fee_multiplier
    )


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
    free_cash_income_bps: NDArray[np.float64]
    debit_financing_bps: NDArray[np.float64]
    cost_bps: NDArray[np.float64]
    borrow_bps: NDArray[np.float64]
    unsettled_cash: NDArray[np.float64]
    loan_liability: NDArray[np.float64]
    loan_payment: NDArray[np.float64]
    loan_outstanding_principal: NDArray[np.float64]
    loan_charges: tuple[LoanCharge, ...]
    loan_cash_payments: tuple[LoanCashPayment, ...]
    equity_borrow_raw_bps: NDArray[np.float64]
    equity_borrow_fee_bps: NDArray[np.float64]
    cdi_benchmark_bps: NDArray[np.float64]
    borrowed_equity_weighted_annual_rate: NDArray[np.float64]
    borrowed_equity_principal_at_open: NDArray[np.float64]
    borrowed_equity_imputed_principal_at_open: NDArray[np.float64]
    borrowed_equity_placeholder_principal_at_open: NDArray[np.float64]
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
    undelivered_share_notional: NDArray[np.float64]
    share_claim_positions: tuple[ShareClaimPosition, ...]
    hedge_restricted_cash: NDArray[np.float64]
    receivables: NDArray[np.float64]
    payables: NDArray[np.float64]
    marked_signed_holdings: NDArray[np.float64]
    reconciliation_error: NDArray[np.float64]
    all_cash_nav: NDArray[np.float64]
    unpriced_haircut_scenario_nav: NDArray[np.float64]
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
    unpriced_inventory_count: NDArray[np.int64]
    unpriced_inventory_notional: NDArray[np.float64]
    unpriced_inventory_fraction_nav: NDArray[np.float64]
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
    terminal_unpriced_hedge_notional: float
    gross_target: float
    ineligible_hold_sessions: int
    settlement_grace_sessions: int
    unpriced_haircut: float
    unpriced_economics_unresolved_fraction_nav: float
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
    hedge_borrow_raw_bps: NDArray[np.float64]
    hedge_borrow_fee_bps: NDArray[np.float64]
    ex_ante_beta_before_hedge: NDArray[np.float64]
    ex_ante_beta_after_hedge: NDArray[np.float64]
    gross_fraction_nav_including_hedge: NDArray[np.float64]
    net_notional_including_hedge: NDArray[np.float64]
    net_fraction_nav_including_hedge: NDArray[np.float64]

    @property
    def equity_market_weights(self):
        """Economic risk exposures, including shares not yet credited to custody."""
        result = self.signed_shares * np.nan_to_num(self.mark_price) / self.nav[:, None]
        for claim in self.share_claim_positions:
            result[claim.session, claim.source_index] = 0.0
            result[claim.session, claim.successor_index] += (
                claim.signed_quantity * claim.mark / self.nav[claim.session]
            )
        return result

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
            "compounded_net_excess_unpriced_haircut_scenario": float(
                self.unpriced_haircut_scenario_nav[-1] / all_cash_terminal - 1.0
            ),
            "compounded_net_excess_unresolved_excluded": float(
                self.unresolved_excluded_nav[-1] / all_cash_terminal - 1.0
            ),
            "mean_gross_fraction_nav": mean_gross,
            "minimum_gross_fraction_nav": minimum_gross,
            "maximum_gross_fraction_nav": maximum_gross,
            "mean_turnover_fraction_nav": mean_turnover,
            "borrow_source": self.borrow_source,
            "borrow_weighting_basis": "outstanding fixed equity loan principal including pending returns",
            "terminal_unsettled_cash": float(self.unsettled_cash[-1]),
            "mean_free_cash_income_bps": float(self.free_cash_income_bps.mean()),
            "mean_debit_financing_bps": float(self.debit_financing_bps.mean()),
            "terminal_loan_liability": float(self.loan_liability[-1]),
            "terminal_loan_outstanding_principal": float(
                self.loan_outstanding_principal[-1]
            ),
            "total_loan_payment": float(self.loan_payment.sum()),
            "principal_weighted_borrow_rate": (
                float(
                    np.sum(
                        self.borrowed_equity_weighted_annual_rate
                        * self.borrowed_equity_principal_at_open
                    )
                    / np.sum(self.borrowed_equity_principal_at_open)
                )
                if np.sum(self.borrowed_equity_principal_at_open) > 0.0
                else 0.0
            ),
            "imputed_rate_share_of_borrowed_principal": (
                float(
                    np.sum(self.borrowed_equity_imputed_principal_at_open)
                    / np.sum(self.borrowed_equity_principal_at_open)
                )
                if np.sum(self.borrowed_equity_principal_at_open) > 0.0
                else 0.0
            ),
            "placeholder_rate_share_of_borrowed_principal": (
                float(
                    np.sum(self.borrowed_equity_placeholder_principal_at_open)
                    / np.sum(self.borrowed_equity_principal_at_open)
                )
                if np.sum(self.borrowed_equity_principal_at_open) > 0.0
                else 0.0
            ),
            "placeholder_priced_session_count": int(
                (self.borrowed_equity_placeholder_principal_at_open > 0.0).sum()
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
            "missing_quote_convention": MISSING_QUOTE_CONVENTION,
            "ineligible_hold_sessions": self.ineligible_hold_sessions,
            "settlement_grace_sessions": self.settlement_grace_sessions,
            "unpriced_haircut": self.unpriced_haircut,
            "unpriced_inventory_count": int(self.unpriced_inventory_count[-1]),
            "terminal_undelivered_share_notional": float(
                self.undelivered_share_notional[-1]
            ),
            "maximum_undelivered_share_notional": float(
                self.undelivered_share_notional.max()
            ),
            "unpriced_inventory_notional": float(self.unpriced_inventory_notional[-1]),
            "unpriced_inventory_fraction_nav": float(
                self.unpriced_inventory_fraction_nav[-1]
            ),
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
            "unpriced_economics_unresolved": bool(
                self.unpriced_inventory_fraction_nav.max()
                > self.unpriced_economics_unresolved_fraction_nav
            ),
            "unpriced_economics_unresolved_fraction_nav": (
                self.unpriced_economics_unresolved_fraction_nav
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
            "terminal_nav_unpriced_haircut_scenario": float(
                self.unpriced_haircut_scenario_nav[-1]
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
            "terminal_unpriced_hedge_notional": self.terminal_unpriced_hedge_notional,
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


def _settled_balances(free_cash, restricted, settlements):
    pending_free = sum(free for _, free, _ in settlements)
    pending_restricted = sum((r for _, _, r in settlements), np.zeros_like(restricted))
    funded_restricted = restricted - pending_restricted
    # Never release an unsettled sale receipt as if it were deposited cash.
    # Its negative earmark instead offsets the corresponding free-cash claim.
    funded_free = free_cash - pending_free + np.minimum(funded_restricted, 0).sum()
    return float(funded_free), np.maximum(funded_restricted, 0)


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
    loan_covers: NDArray[np.float64],
    loan_openings: NDArray[np.float64],
    loan_name: int,
    long_purchases: NDArray[np.float64],
    long_sales: NDArray[np.float64],
) -> tuple[float, float]:
    old_shares = float(shares[name])
    remaining = quantity
    if side == "buy" and old_shares < 0.0:
        cover = min(remaining, -old_shares)
        loan_covers[loan_name] += cover
        release = restricted_by_name[name] * cover / -old_shares
        restricted_by_name[name] -= release
        free_cash += release - cover * price
        shares[name] += cover
        remaining -= cover
    if side == "sell" and old_shares > 0.0:
        sale = min(remaining, old_shares)
        long_sales[loan_name] += sale
        free_cash += sale * price
        shares[name] -= sale
        remaining -= sale
    if remaining > 0.0:
        if side == "buy":
            long_purchases[loan_name] += remaining
            free_cash -= remaining * price
            shares[name] += remaining
        else:
            loan_openings[loan_name] += remaining
            restricted_by_name[name] += remaining * price
            shares[name] -= remaining
    notional = quantity * price
    free_cash -= cost_rate * notional
    marks[name] = price if shares[name] != 0.0 else np.nan
    return free_cash, notional


@dataclass(frozen=True)
class PortfolioDecisionState:
    """Observable pre-fill state; no current-session realization is exposed."""

    day: int
    weights: NDArray[np.float64]
    hedge_weight: float
    free_cash_fraction: float
    restricted_cash_fraction: float
    pending_entry_weights: NDArray[np.float64]
    pending_exit_fractions: NDArray[np.float64]
    holding_sessions: NDArray[np.int64]
    marked_pnl_fraction: NDArray[np.float64]
    entry_allowed: NDArray[np.bool_]
    shortable: NDArray[np.bool_]
    required_exit: NDArray[np.bool_]
    locked: NDArray[np.bool_]
    effective_beta: NDArray[np.float64]
    claim_exposure: NDArray[np.float64]


@dataclass(frozen=True)
class PortfolioTarget:
    weights: NDArray[np.float64]
    hedge_weight: float = 0.0


def simulate_stateful_ledger(
    *,
    dates: Sequence[date],
    loan_reference_prices: NDArray[np.floating] | None = None,
    scores: NDArray[np.floating],
    score_mask: NDArray[np.bool_],
    active: NDArray[np.bool_],
    raw_close: NDArray[np.floating],
    action_terms: AlignedActionTerms,
    action_payment_session: NDArray[np.integer],
    cdi_returns: NDArray[np.floating],
    share_distributions: Sequence[ShareDistribution] = (),
    loan_cash_settlements: Sequence[LoanCashSettlement] = (),
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
    entry_fill_allowed: NDArray[np.bool_] | None = None,
    portfolio_policy: Callable[[PortfolioDecisionState], PortfolioTarget] | None = None,
) -> StatefulLedgerResult:
    """Run causal close-proxy orders, fills, and a raw signed-share ledger.

    Optional entry volatility applies inverse-volatility sizing to each side's
    new-entry cohort; existing inventory is not rebalanced. It must already be
    available at the decision. Capacity can be bound to a reference buffer for
    a retention-only ablation without changing entry quotas.
    An execution-time entry-fill mask can suppress opening fills without changing
    earlier orders, printed exits, or inventory valuation.
    A portfolio policy replaces only rank/slot intentions. Cash, shares, action
    claims, financing and observed fills retain the same accounting. Its
    state is captured before the current session's retrospective observations.
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
    if portfolio_policy is not None and config.volatility_balanced_entries:
        raise ValueError("portfolio allocation does not use fixed volatility slots")
    day_count, name_count = inputs.score.shape
    if loan_cash_settlements:
        shortable_mask = inputs.shortable.copy()
        keys = set()
        for event in loan_cash_settlements:
            if not (
                0 <= event.security_index < name_count
                and event.effective_session < day_count
            ):
                raise ValueError(
                    "loan settlement must align with replay sessions and names"
                )
            key = (event.effective_session, event.security_index)
            if key in keys:
                raise ValueError("duplicate loan cash settlement")
            keys.add(key)
            shortable_mask[max(0, event.effective_session) :, event.security_index] = (
                False
            )
        inputs = replace(inputs, shortable=shortable_mask)
    distributions_by_day: dict[int, list[ShareDistribution]] = {}
    for event in share_distributions:
        if not (
            event.effective_session < day_count and 0 <= event.source_index < name_count
        ):
            raise ValueError("distribution must align with replay sessions and names")
        if any(not 0 <= leg.successor_index < name_count for leg in event.legs):
            raise ValueError("distribution successor is outside the security axis")
        if (
            event.effective_session >= 0
            and inputs.has_action[event.effective_session, event.source_index]
        ):
            raise ValueError("a distribution cannot duplicate a scalar source action")
        distributions_by_day.setdefault(event.effective_session, []).append(event)
    if entry_fill_allowed is not None:
        entry_fill_allowed = np.asarray(entry_fill_allowed, dtype=np.bool_)
        if entry_fill_allowed.shape != inputs.score.shape:
            raise ValueError("entry-fill availability must align with ledger panels")
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
    loans = LoanContracts(
        name_count + 1,
        config.borrow_fee_modality,
        config.borrow_fee_multiplier,
        config.annual_sessions,
    )
    loan_references = (
        np.full((day_count, name_count + 1), np.nan)
        if loan_reference_prices is None
        else np.asarray(loan_reference_prices, dtype=float)
    )
    if loan_references.shape != (day_count, name_count + 1):
        raise ValueError(
            "loan reference prices must align dates and equity-plus-hedge axes"
        )
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
    free_cash = config.initial_capital_brl
    # Trade-date cash includes dated settlement claims. Funding and public cash
    # balances below subtract those claims until their value date.
    settlements: list[tuple[int | None, float, NDArray[np.float64]]] = []
    custody = ShareCustody(name_count + 1)
    previous_nav = config.initial_capital_brl
    all_cash = config.initial_capital_brl
    pending_entries: dict[int, _PendingOrder] = {}
    pending_exits: dict[int, _PendingOrder] = {}
    pending_claims: list[_PendingClaim] = []
    pending_distributions = {}
    share_claim_positions = []
    retired_sources = {
        event.source_index
        for event in share_distributions
        if event.effective_session < 0
    }
    if retired_sources:
        last_observed[list(retired_sources)] = np.nan
    undelivered_rows = []
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
    free_cash_income_rows: list[float] = []
    debit_financing_rows: list[float] = []
    cost_rows: list[float] = []
    borrow_rows: list[float] = []
    loan_liability_rows: list[float] = []
    loan_payment_rows: list[float] = []
    loan_principal_rows: list[float] = []
    loan_charges: list[LoanCharge] = []
    loan_cash_payments: list[LoanCashPayment] = []
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
    unsettled_cash_rows: list[float] = []
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
    unpriced_count_rows: list[int] = []
    unpriced_notional_rows: list[float] = []
    unpriced_fraction_rows: list[float] = []
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
    hedge_borrow_raw_rows: list[float] = []
    hedge_borrow_fee_rows: list[float] = []
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
    terminal_unpriced_hedge_notional = 0.0
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

    def deliver_shares(
        day,
        name,
        successor,
        new_shares,
        allocation,
        converted_mark,
        converted_reference,
        *,
        final,
        ratio,
    ):
        nonlocal free_cash, cancelled_today
        transferred_restricted = restricted_by_name[name] * allocation
        for _, _, restricted_flow in settlements:
            transferred_pending = restricted_flow[name] * allocation
            restricted_flow[successor] += transferred_pending
            restricted_flow[name] -= transferred_pending
        transferred_basis = entry_cost_basis[name] * allocation
        prior_destination = float(shares[successor])
        combined = prior_destination + float(new_shares)
        loans.deliver(name, successor, ratio, allocation, final=final)
        custody_dates = custody.deliver(
            name, successor, ratio, new_shares, prior_destination, day, final=final
        )
        returns = np.zeros(name_count + 1)
        returns[successor] = max(
            float(loans.active_quantity[successor]) - max(-combined, 0), 0
        )
        # Delivery offsets opposite inventory without a market trade.
        # Only the extinguished short portion releases its proceeds.
        opposite = prior_destination * new_shares < 0.0
        source_left = (
            max(abs(new_shares) - abs(prior_destination), 0.0)
            if opposite
            else abs(new_shares)
        )
        destination_left = (
            max(abs(prior_destination) - abs(new_shares), 0.0)
            if opposite
            else abs(prior_destination)
        )
        source_fraction = source_left / abs(new_shares) if new_shares else 0.0
        destination_fraction = (
            destination_left / abs(prior_destination) if prior_destination else 0.0
        )
        restricted = (
            transferred_restricted * source_fraction
            + restricted_by_name[successor] * destination_fraction
        )
        release = transferred_restricted + restricted_by_name[successor] - restricted
        offset = min(abs(prior_destination), abs(new_shares)) if opposite else 0.0
        for due, quantity in custody_dates:
            fraction = float(quantity[successor]) / max(offset, 1e-30)
            loans.request_return(returns * fraction, due)
            if due > day:
                deferred = release * fraction
                restricted_flow = np.zeros(name_count + 1)
                restricted_flow[successor] = -deferred
                settlements.append((due, deferred, restricted_flow))
        free_cash += release
        restricted_by_name[name] -= transferred_restricted
        restricted_by_name[successor] = restricted
        entry_cost_basis[successor] = (
            transferred_basis * source_fraction
            + entry_cost_basis[successor] * destination_fraction
        )
        submission_nav[successor] = (
            submission_nav[successor]
            if destination_left
            else submission_nav[name]
            if source_left
            else 0.0
        )
        surviving_origins = [
            index
            for index, quantity in (
                (int(name), source_left),
                (successor, destination_left),
            )
            if quantity > 0.0
        ]
        entry_session[successor] = min(
            (entry_session[index] for index in surviving_origins), default=-1
        )
        ineligible_streak[successor] = max(
            (ineligible_streak[index] for index in surviving_origins), default=0
        )
        # Keep the successor's own causal mark when available. A source
        # with no holding must not erase an existing recipient position.
        destination_mark = marks[successor] if prior_destination else converted_mark
        if not np.isfinite(destination_mark) or destination_mark <= 0.0:
            destination_mark = converted_reference
        source_exit = pending_exits.get(int(name))
        if final:
            pending_exits.pop(int(name), None)
        destination_exit = pending_exits.pop(successor, None)
        exit_quantity = (
            new_shares * source_exit.remaining_size if source_exit else 0.0
        ) + (
            prior_destination * destination_exit.remaining_size
            if destination_exit
            else 0.0
        )
        exit_fraction = (
            float(np.clip(exit_quantity / combined, 0.0, 1.0)) if combined else 0.0
        )
        pending = None
        if exit_fraction > 0.0:
            side = "sell" if combined > 0.0 else "buy"
            pending = next(
                item
                for item in (destination_exit, source_exit)
                if item is not None and item.order.side == side
            )
            # The public intention remains unchanged. Its outstanding
            # claim follows contractual succession and the net quantity.
            adjusted_order = replace(
                pending.order,
                security=inputs.securities[successor],
                security_index=successor,
                side=side,
            )
            pending_exits[successor] = _PendingOrder(adjusted_order, exit_fraction)
        for item in (source_exit, destination_exit):
            if (
                item is not None
                and item is not pending
                and (item is not source_exit or final)
            ):
                cancellations.append(
                    item.cancellation(
                        day, inputs.dates[day], "corporate_action_netting"
                    )
                )
        cancelled_today += int(cancel_entry(int(name), day, "corporate_action_netting"))
        if final:
            shares[name] = 0.0
            marks[name] = np.nan
            last_observed[name] = np.nan
        if not prior_destination and new_shares:
            missing_sessions[successor] = missing_sessions[name]
        if final:
            missing_sessions[name] = 0
        shares[successor] = combined
        marks[successor] = destination_mark if combined else np.nan
        if not prior_destination and np.isfinite(converted_reference):
            last_observed[successor] = converted_reference
        if combined and (not np.isfinite(destination_mark) or destination_mark <= 0):
            unresolved_action[successor] = True
            explicit_unresolved_action[successor] = True
        explicit_unresolved_action[successor] |= explicit_unresolved_action[name]
        entry_cost_basis[name] -= transferred_basis
        if final:
            unresolved_action[name] = False
            explicit_unresolved_action[name] = False
            entry_session[name] = -1
            ineligible_streak[name] = 0
            entry_cost_basis[name] = 0.0
            submission_nav[name] = 0.0

    def deliver_due(day, *, realize_auctions=False):
        # Each delivery uses the same netting transition as a one-leg conversion.
        # Value weights allocate existing proceeds/basis; they create no cash.
        for name, legs in tuple(pending_distributions.items()):
            for leg in tuple(legs):
                auction = leg.fractional_auction
                if (
                    realize_auctions
                    and leg.delivery_session is None
                    and auction is not None
                    and auction.available_session == day
                ):
                    claim = (
                        shares[name]
                        * leg.shares_per_prior_share
                        * auction.cash_per_share
                    )
                    receivable_by_name[name] += claim
                    pending_claims.append(
                        _PendingClaim(name, claim, auction.payment_session)
                    )
                    shares[name], marks[name] = 0.0, np.nan
                    entry_cost_basis[name], entry_session[name] = 0.0, -1
                    pending_exits.pop(name, None)
                    legs.remove(leg)
                    continue
                if leg.delivery_session != day:
                    continue
                prices = basket_prices(legs, last_observed)
                values = [
                    item.shares_per_prior_share * price
                    for item, price in zip(legs, prices)
                ]
                allocation = values[legs.index(leg)] / sum(values)
                destination = leg.successor_index
                source_entry = entry_session[name]
                incoming = float(shares[name]) * leg.shares_per_prior_share
                fraction = fraction_basis = 0.0
                if auction is not None and incoming > 0:
                    if any(float(q[name]) > 1e-12 for _, q in custody.receipts):
                        raise ValueError(
                            "fraction auction requires settled source purchases"
                        )
                    fraction = incoming - np.floor(incoming)
                    fraction_basis = entry_cost_basis[name] * fraction / incoming
                    entry_cost_basis[name] -= fraction_basis
                    incoming = float(np.floor(incoming))
                deliver_shares(
                    day,
                    name,
                    destination,
                    incoming,
                    allocation,
                    float(last_observed[destination]),
                    float(last_observed[destination]),
                    final=len(legs) == 1,
                    ratio=leg.shares_per_prior_share,
                )
                if fraction > 0:
                    shares[name] = fraction / leg.shares_per_prior_share
                    entry_cost_basis[name], entry_session[name] = (
                        fraction_basis,
                        source_entry,
                    )
                    legs[legs.index(leg)] = replace(leg, delivery_session=None)
                else:
                    legs.remove(leg)
            if legs:
                prices = basket_prices(legs, last_observed)
                marks[name] = sum(
                    leg.shares_per_prior_share * price
                    for leg, price in zip(legs, prices)
                )
            else:
                del pending_distributions[name]

    for day in range(day_count):
        funding_cash, funding_restricted = _settled_balances(
            free_cash, np.r_[restricted_by_name, hedge_restricted_cash], settlements
        )
        settlements = [item for item in settlements if item[0] is None or item[0] > day]
        cancelled_today = 0
        custody.settle(day)
        deliver_due(day)
        locked = np.zeros(name_count, dtype=bool)
        if pending_distributions:
            locked[list(pending_distributions)] = True
        decision_beta = basket_betas(
            pending_distributions, last_observed, inputs.hedge_beta[day]
        )
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
        start_identity -= float(loans.liability)
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

        if portfolio_policy is not None:
            eligible = (
                inputs.score_valid[day]
                & inputs.membership[day]
                & np.isfinite(inputs.score[day])
            )
            held = shares != 0.0
            signed_values = np.where(held, shares * np.nan_to_num(marks), 0.0)
            weights = signed_values / start_nav
            claim_exposure = np.zeros(name_count)
            for name, legs in pending_distributions.items():
                for leg in legs:
                    claim_exposure[leg.successor_index] += (
                        shares[name]
                        * leg.shares_per_prior_share
                        * last_observed[leg.successor_index]
                        / start_nav
                    )
            ineligible_streak[~held | eligible] = 0
            ineligible_streak[held & ~eligible] += 1
            entry_eligible = (
                eligible
                & ~decision_unresolved
                & np.isfinite(last_observed)
                & (last_observed > 0.0)
            )
            if pending_distributions:
                entry_eligible[list(pending_distributions)] = False
            required_exit = (ineligible_streak > config.ineligible_hold_sessions) | (
                missing_sessions >= config.settlement_grace_sessions
            )
            required_exit[locked] = False
            pending_entry_weights = np.zeros(name_count, dtype=np.float64)
            pending_exit_fractions = np.zeros(name_count, dtype=np.float64)
            for name, pending in pending_entries.items():
                sign = 1.0 if pending.order.side == "buy" else -1.0
                pending_entry_weights[name] = sign * pending.remaining_size / start_nav
            for name, pending in pending_exits.items():
                pending_exit_fractions[name] = pending.remaining_size
            decision_cash, decision_restricted = _settled_balances(
                free_cash, np.r_[restricted_by_name, hedge_restricted_cash], settlements
            )
            decision_state = PortfolioDecisionState(
                day=day,
                weights=weights.copy(),
                hedge_weight=hedge_shares * hedge_mark / start_nav
                if hedge_shares
                else 0.0,
                free_cash_fraction=decision_cash / start_nav,
                restricted_cash_fraction=float(decision_restricted.sum()) / start_nav,
                pending_entry_weights=pending_entry_weights,
                pending_exit_fractions=pending_exit_fractions,
                holding_sessions=np.where(held, day - entry_session, 0),
                marked_pnl_fraction=np.where(
                    held,
                    np.sign(weights)
                    * (np.abs(signed_values) - entry_cost_basis)
                    / start_nav,
                    0.0,
                ),
                entry_allowed=entry_eligible.copy(),
                shortable=inputs.shortable[day].copy(),
                required_exit=required_exit,
                locked=locked.copy(),
                effective_beta=decision_beta,
                claim_exposure=claim_exposure,
            )
            target = (
                portfolio_policy(decision_state)
                if day < day_count - 1
                else PortfolioTarget(np.zeros(name_count))
            )
            desired = np.asarray(target.weights, dtype=np.float64)
            if desired.shape != weights.shape or not np.isfinite(desired).all():
                raise ValueError(
                    "portfolio targets must be finite on the security axis"
                )
            if not np.isfinite(target.hedge_weight):
                raise ValueError("portfolio hedge target is non-finite")
            tol = 2e-6
            opening = (
                (desired * weights < 0) | (np.abs(desired) > np.abs(weights) + tol)
            ) & (np.abs(desired) > tol)
            opening_short = desired < np.minimum(weights, 0.0) - tol
            if np.any(opening & ~entry_eligible) or np.any(
                opening_short & ~inputs.shortable[day]
            ):
                raise ValueError("portfolio requests an unavailable opening trade")
            if np.any(required_exit & (np.abs(desired) > tol)):
                raise ValueError("portfolio retains required exits")
            if (
                np.abs(desired).sum() + abs(target.hedge_weight)
                > config.planned_gross_cap + tol
                or abs(desired.sum() + target.hedge_weight)
                > config.planned_absolute_net_cap + tol
                or np.abs(desired[~locked]).max(initial=0.0)
                > config.planned_name_weight_cap + tol
                or abs(target.hedge_weight) > config.hedge_notional_cap_nav + tol
                or (not config.beta_hedge and abs(target.hedge_weight) > tol)
                or (
                    config.beta_hedge
                    and abs(desired @ decision_beta + target.hedge_weight)
                    > config.planned_absolute_beta_cap + tol
                )
            ):
                raise ValueError("portfolio target violates the planned risk budget")
            # Daily cancel/replace uses actual partial fills. Reversals open
            # only after their corresponding exit has completely filled.
            for name in tuple(pending_entries):
                cancelled_today += int(cancel_entry(name, day, "policy_replaced"))
            for pending in pending_exits.values():
                cancellations.append(
                    pending.cancellation(day, inputs.dates[day], "policy_replaced")
                )
            pending_exits.clear()
            submitted_entries = submitted_exits = 0
            exit_cause_today = np.zeros(name_count, dtype=np.int8)
            exit_side_today = np.zeros(name_count, dtype=np.int8)
            for name in range(name_count):
                old_weight, new_weight = weights[name], desired[name]
                reverse = old_weight * new_weight < 0.0
                reduction = (
                    abs(old_weight)
                    if reverse
                    else max(abs(old_weight) - abs(new_weight), 0.0)
                )
                if reduction > 1e-10 and old_weight != 0.0:
                    pending_exits[name] = submit_order(
                        day,
                        name,
                        "sell" if old_weight > 0 else "buy",
                        min(1.0, reduction / abs(old_weight)),
                        float(marks[name]),
                        "terminal_exit" if day == day_count - 1 else "exit",
                        None,
                        position_fraction=True,
                    )
                    submitted_exits += 1
                    exit_cause_today[name] = (
                        4 if day == day_count - 1 else 1 if required_exit[name] else 5
                    )
                    exit_side_today[name] = 1 if old_weight > 0 else -1
                increase = (
                    abs(new_weight)
                    if reverse
                    else max(abs(new_weight) - abs(old_weight), 0.0)
                )
                if increase > 1e-10 and day < day_count - 1:
                    pending_entries[name] = submit_order(
                        day,
                        name,
                        "buy" if new_weight > 0 else "sell",
                        increase * start_nav,
                        float(last_observed[name]),
                        "entry",
                        day,
                    )
                    submission_nav[name] = start_nav
                    submitted_entries += 1
            # Slot fields are inapplicable; policy reporting omits them.
            k_eff = 0
            volatility_groups = np.full(name_count, -1, dtype=np.int64)
            volatility_quota = np.zeros(config.volatility_group_count, dtype=np.int64)
            group_sizes = volatility_quota.copy()
            spilled_entries = {
                "buy": volatility_quota.copy(),
                "sell": volatility_quota.copy(),
            }
            band_exhausted_long = band_exhausted_short = 0
            blocked_open_long = blocked_open_short = 0
            k_eff_rows.append(k_eff)
            small_universe_rows.append(int(entry_eligible.sum()) < 2)
            retention_rows.append(0)
            ineligible_score_invalid_rows.append(
                int((held & ~inputs.score_valid[day]).sum())
            )
            ineligible_membership_invalid_rows.append(
                int((held & ~inputs.membership[day]).sum())
            )
            ineligible_score_nonfinite_rows.append(
                int((held & ~np.isfinite(inputs.score[day])).sum())
            )
            blocked_reference_rows.append(0)
            blocked_gross_rows.append(0)
            blocked_net_rows.append(0)
            blocked_name_rows.append(0)
            submitted_entry_rows.append(submitted_entries)
            submitted_exit_rows.append(submitted_exits)
            same_close_replacement_rows.append(0)
            zero_entry_small_rows.append(0)
            zero_entry_gross_rows.append(0)
            zero_entry_net_rows.append(0)
            zero_entry_name_rows.append(0)
            zero_entry_reference_rows.append(0)
            risk_trim_gross_rows.append(0)
            risk_trim_net_rows.append(0)
            risk_trim_name_rows.append(0)
            occupied_after_submission_long_rows.append(0)
            occupied_after_submission_short_rows.append(0)
            open_after_submission_long_rows.append(0)
            open_after_submission_short_rows.append(0)
            band_candidates_long_rows.append(0)
            band_candidates_short_rows.append(0)
            excluded_short_candidate_rows.append(0)
            band_excluded_unresolved_long_rows.append(0)
            band_excluded_unresolved_short_rows.append(0)
            band_without_prior_print_long_rows.append(0)
            band_without_prior_print_short_rows.append(0)
            band_exhausted_long_rows.append(0)
            band_exhausted_short_rows.append(0)
            blocked_open_long_rows.append(0)
            blocked_open_short_rows.append(0)
            exit_ineligible_hold_exhausted_rows.append(
                int((exit_cause_today == 1).sum())
            )
            exit_settlement_rows.append(0)
            exit_rank_rows.append(0)
            exit_terminal_rows.append(int((exit_cause_today == 4).sum()))
            exit_cause_rows.append(exit_cause_today)
            exit_side_rows.append(exit_side_today)
            ineligible_exit_within_hold_rows.append(0)
            net_cap_balanced_rows.append(0)
            gross_cap_below_target_rows.append(0)
            name_cap_fresh_rows.append(0)
        else:
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
                eligible_names[
                    np.argsort(inputs.score[day, eligible_names], kind="stable")
                ]
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
                group_eligible = eligible & np.isfinite(
                    inputs.selection_volatility[day]
                )
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
                    group_order = names[
                        np.argsort(inputs.score[day, names], kind="stable")
                    ]
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
                volatility_quota = np.zeros(
                    config.volatility_group_count, dtype=np.int64
                )
                volatility_buffer = np.zeros(
                    config.volatility_group_count, dtype=np.int64
                )
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
                            pending.cancellation(
                                day, inputs.dates[day], "evaluation_end"
                            )
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
                    if not locked[name]:
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
                            projected[risk_name] += (
                                direction * quantity * marks[risk_name]
                            )
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
                        if pending.order.side
                        == ("buy" if heavy_side == "sell" else "sell")
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
                            projected[risk_name] += (
                                direction * quantity * marks[risk_name]
                            )
                        _, projected_net, _ = _risk(projected, start_nav)
                    net_reduction = max(
                        (abs(projected_net) - target_net) * start_nav, 0.0
                    )
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
            open_after_submission_short = max(
                k_eff - occupied_after_submission_short, 0
            )
            band_candidates_long = 0
            band_candidates_short = 0
            band_excluded_unresolved_long = 0
            band_excluded_unresolved_short = 0
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
                slot_notional = (
                    start_nav * config.gross_target / (2 * config.k_per_side)
                )
                same_day_exit_names = {
                    name
                    for name, pending in pending_exits.items()
                    if pending.order.decision_session == day
                    and pending.remaining_size >= 1.0 - 1e-12
                    and not locked[name]
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
                        long_band.extend(
                            int(name) for name in group_order[-width:][::-1]
                        )
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
                    if name not in unavailable and not decision_unresolved[name]
                ]
                short_candidates = [
                    name
                    for name in short_band
                    if entry_eligible[name]
                    if name not in unavailable
                    and not decision_unresolved[name]
                    and inputs.shortable[day, name]
                ]
                long_candidates.sort(
                    key=lambda name: (float(inputs.score[day, name]), name),
                    reverse=True,
                )
                short_candidates.sort(
                    key=lambda name: (float(inputs.score[day, name]), name)
                )
                excluded_short_candidates = sum(
                    entry_eligible[name]
                    and name not in unavailable
                    and not decision_unresolved[name]
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
                            if (
                                volatility_occupancy[side][group]
                                >= volatility_quota[group]
                            ):
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
                        planned.extend(
                            name for name in candidates if name not in planned
                        )
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
                        raise RuntimeError(
                            "entry scheduler lost an available candidate"
                        )
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
            band_without_prior_print_long_rows.append(band_without_prior_print_long)
            band_without_prior_print_short_rows.append(band_without_prior_print_short)
            band_exhausted_long_rows.append(band_exhausted_long)
            band_exhausted_short_rows.append(band_exhausted_short)
            blocked_open_long_rows.append(blocked_open_long)
            blocked_open_short_rows.append(blocked_open_short)
            exit_ineligible_hold_exhausted_rows.append(
                int((exit_cause_today == 1).sum())
            )
            exit_settlement_rows.append(int((exit_cause_today == 2).sum()))
            exit_rank_rows.append(int((exit_cause_today == 3).sum()))
            exit_terminal_rows.append(int((exit_cause_today == 4).sum()))
            exit_cause_rows.append(exit_cause_today)
            exit_side_rows.append(exit_side_today)
            ineligible_exit_within_hold_rows.append(ineligible_exit_within_hold_today)
            net_cap_balanced_rows.append(net_cap_balanced)
            gross_cap_below_target_rows.append(gross_cap_below_target)
            name_cap_fresh_rows.append(name_cap_fresh)

        # Freeze the hedge before reading any current-session fill or close.
        planned_equity = np.zeros(name_count, dtype=np.float64)
        held_for_hedge = shares != 0.0
        planned_equity[held_for_hedge] = shares[held_for_hedge] * marks[held_for_hedge]
        for name, pending in pending_entries.items():
            direction = 1.0 if pending.order.side == "buy" else -1.0
            planned_equity[name] += direction * pending.remaining_size
        for name, pending in pending_exits.items():
            if not locked[name]:
                planned_equity[name] -= (
                    pending.remaining_size * shares[name] * marks[name]
                )
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
            float(np.sum(planned_equity[exposure] * decision_beta[exposure]))
            if config.beta_hedge
            else 0.0
        )
        hedge_unconstrained_target_notional = (
            target.hedge_weight * start_nav
            if portfolio_policy is not None
            else -equity_beta_notional
            if day != day_count - 1
            else 0.0
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
            > (
                0.0
                if portfolio_policy is not None
                else config.hedge_rebalance_threshold_nav
            )
            * start_nav
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
            or any(
                action_exposed[event.source_index]
                for event in distributions_by_day.get(day, ())
            )
        )

        # Action-term uncertainty is a property of this session's claim, not
        # of the position for the rest of its life. A later resolved cell
        # clears the condition without requiring a fill.
        unresolved_action = ~inputs.action_resolved[day].copy()
        explicit_unresolved_action = inputs.has_action[day] & unresolved_action
        action_uncertainty_seen |= bool(
            ((shares != 0.0) & explicit_unresolved_action).any()
        )
        loan_rent, loan_fees = loans.accrue(
            day, inputs.dates[day], charges=loan_charges
        )
        for event in loan_cash_settlements:
            if event.effective_session != day:
                continue
            name = event.security_index
            for due in np.unique(loans.return_day[loans.name == name]):
                if due > day:
                    restored = np.zeros(name_count + 1)
                    restored[name] = float(
                        loans.quantity[
                            (loans.name == name) & (loans.return_day == due)
                        ].sum()
                    )
                    custody.add(int(due), restored)
            quantity = float(loans.cash_settle(name, day))
            paid = quantity * event.cash_per_share
            free_cash += restricted_by_name[name] - paid
            restricted_by_name[name] = 0.0
            for index, (due, free, pending_restricted) in enumerate(settlements):
                remaining = pending_restricted.copy()
                amount = remaining[name]
                remaining[name] = 0
                settlements[index] = (due, free + amount, remaining)
            shares[name] += quantity
            if shares[name] != 0 and not np.isfinite(marks[name]):
                # A superseded cover can leave a long asset after net inventory
                # was flat. Retain its causal mark, not the loan cash price.
                marks[name] = last_observed[name]
            if quantity:
                loan_cash_payments.append(LoanCashPayment(day, name, quantity, paid))
            if abs(shares[name]) < 1e-12:
                shares[name] = 0.0
                pending_exits.pop(name, None)
                entry_session[name] = -1
                entry_cost_basis[name] = submission_nav[name] = 0.0
                ineligible_streak[name] = 0
        rates_today = (
            np.full(name_count, config.annual_borrow_rate)
            if config.borrow_source == "uniform"
            else inputs.annual_borrow_rate_by_name[day]
        )
        hedge_rate_today = (
            inputs.hedge_annual_borrow_rate[day]
            if np.isfinite(inputs.hedge_annual_borrow_rate[day])
            else config.hedge_annual_borrow_rate
        )
        loan_session = LoanSession(
            day,
            inputs.dates[day],
            loan_references[day],
            np.r_[rates_today, hedge_rate_today],
            spot_settlement_session(day, inputs.dates[day]),
            np.r_[inputs.borrow_rate_imputed[day], False],
            np.r_[
                inputs.borrow_rate_placeholder[day],
                not np.isfinite(inputs.hedge_annual_borrow_rate[day]),
            ],
        )
        # Apply contractual terms to shares held before the session. Cash terms
        # become claims; only the later payment mask transfers them to cash.
        for name in np.flatnonzero(inputs.has_action[day]):
            if name in pending_distributions or any(
                int(name) == leg.successor_index
                for legs in pending_distributions.values()
                for leg in legs
            ):
                raise ValueError(
                    "an action on an outstanding basket needs explicit claim terms"
                )
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
            if q > 0 and successor == name:
                loans.split(int(name), q)
                custody.split(int(name), q)
            elif q == 0 and inputs.action_payment_session[day, name] >= day:
                returns = np.zeros(name_count + 1)
                returns[name] = float(loans.active_quantity[name])
                loans.request_return(
                    returns, int(inputs.action_payment_session[day, name])
                )
            if q == 0:
                custody.split(int(name), 0.0)
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
                deliver_shares(
                    day,
                    int(name),
                    successor,
                    float(new_shares),
                    1.0,
                    converted_mark,
                    converted_reference,
                    final=True,
                    ratio=q,
                )
            if (
                successor == name
                and np.isfinite(converted_mark)
                and converted_mark <= 0.0
            ):
                unresolved_action[name] = True
                explicit_unresolved_action[name] = True
            if q == 0.0 and restricted_by_name[name] != 0.0:
                pay = int(inputs.action_payment_session[day, name])
                if pay != day:
                    release = np.zeros(name_count + 1)
                    release[name] = -restricted_by_name[name]
                    settlements.append(
                        (
                            pay if pay >= day else None,
                            float(restricted_by_name[name]),
                            release,
                        )
                    )
                free_cash += restricted_by_name[name]
                restricted_by_name[name] = 0.0
                ineligible_streak[name] = 0
                entry_session[name] = -1
                entry_cost_basis[name] = 0.0
                submission_nav[name] = 0.0

        # Recognize a non-tradable basket at its economic effective date. The
        # current source quote can no longer sell the cancelled predecessor.
        for event in distributions_by_day.get(day, ()):
            name = event.source_index
            retired_sources.add(name)
            if name in pending_distributions:
                raise ValueError("overlapping distributions on one predecessor")
            cancelled_today += int(cancel_entry(name, day, "corporate_action_netting"))
            claim = float(shares[name]) * event.cash_per_prior_share
            if claim >= 0:
                receivable_by_name[name] += claim
            else:
                payable_by_name[name] -= claim
            if claim:
                pending_claims.append(_PendingClaim(name, claim, event.payment_session))
            if shares[name] != 0 or (loans.name == name).any():
                prices = basket_prices(event.legs, last_observed)
                pending_distributions[name] = list(event.legs)
                marks[name] = sum(
                    leg.shares_per_prior_share * price
                    for leg, price in zip(event.legs, prices)
                )
                unresolved_action[name] = False
                explicit_unresolved_action[name] = False
            last_observed[name] = np.nan

        deliver_due(day, realize_auctions=True)

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

        cash_rate = float(inputs.cdi[day])
        debit_rate = cash_rate + config.annual_debit_spread / config.annual_sessions
        free_cash_income = max(funding_cash, 0.0) * cash_rate
        debit_financing = -min(funding_cash, 0.0) * debit_rate
        free_cash_interest = free_cash_income - debit_financing
        short_proceeds_interest_base = float(funding_restricted.sum())
        short_proceeds_interest = (
            short_proceeds_interest_base
            * cash_rate
            * config.short_proceeds_remuneration
        )
        interest = free_cash_interest + short_proceeds_interest
        equity_borrow_raw = float(loan_rent[:-1].sum())
        equity_borrow_fee = float(loan_fees[:-1].sum())
        hedge_borrow_raw = float(loan_rent[-1])
        hedge_borrow_fee = float(loan_fees[-1])
        hedge_borrow = hedge_borrow_raw + hedge_borrow_fee
        borrow = equity_borrow_raw + equity_borrow_fee + hedge_borrow
        equity_loans = loans.name < name_count
        principal = loans.principal.detach().numpy()[equity_loans]
        contract_rates = loans.annual_rate[equity_loans]
        contract_fees = (
            loan_fee_rates(
                contract_rates, inputs.dates[day], modality=config.borrow_fee_modality
            ).sum(axis=-1)
            * config.borrow_fee_multiplier
        )
        weighted_borrow_rate = (
            float(
                np.sum(principal * (contract_rates + contract_fees)) / principal.sum()
            )
            if principal.sum()
            else 0.0
        )
        # Quality attribution is bound to each loan's opening observation below.
        held_short_borrow_rate_rows.append(weighted_borrow_rate)
        held_short_notional_rows.append(float(principal.sum()))
        held_short_imputed_notional_rows.append(
            float(principal[loans.imputed[equity_loans]].sum())
        )
        held_short_placeholder_notional_rows.append(
            float(principal[loans.placeholder[equity_loans]].sum())
        )
        free_cash += interest
        fill_cash_before = free_cash
        fill_restricted_before = np.r_[restricted_by_name, hedge_restricted_cash]

        # Only now may current-session prints affect the result. This makes the
        # immutable intended-order set invariant to those later observations.
        printed = np.isfinite(inputs.raw_close[day]) & (inputs.raw_close[day] > 0.0)
        if retired_sources:
            printed[list(retired_sources)] = False
        if day == day_count - 1:
            terminal_printed = printed.copy()
        loan_covers = np.zeros(name_count + 1)
        loan_openings = np.zeros(name_count + 1)
        long_purchases = np.zeros(name_count + 1)
        long_sales = np.zeros(name_count + 1)
        long_before_fill = np.maximum(np.r_[shares, hedge_shares], 0)
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
                if entries and portfolio_policy is not None and name in pending_exits:
                    continue
                if (
                    entries
                    and entry_fill_allowed is not None
                    and not entry_fill_allowed[day, name]
                ):
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
                    loan_covers=loan_covers,
                    loan_openings=loan_openings,
                    loan_name=name,
                    long_purchases=long_purchases,
                    long_sales=long_sales,
                )
                after = float(shares[name])
                if entries:
                    if before * after < 0.0:
                        # An entry may cross a residual below the exit-order
                        # threshold. The new side starts a new holding period.
                        entry_cost_basis[name] = (
                            abs(after) * inputs.raw_close[day, name]
                        )
                    else:
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
                if (before == 0.0 or before * after < 0.0) and after != 0.0:
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
            bool(printed[name])
            and (entry_fill_allowed is None or bool(entry_fill_allowed[day, name]))
            for name in pending_entries
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

        for name in range(name_count):
            if printed[name]:
                last_print_session[name] = day
                last_observed[name] = inputs.raw_close[day, name]
                if shares[name] != 0.0:
                    marks[name] = inputs.raw_close[day, name]
                    missing_sessions[name] = 0
            elif shares[name] != 0.0:
                missing_sessions[name] += 1

        valued = printed.copy()
        for name, legs in pending_distributions.items():
            prices = basket_prices(legs, last_observed)
            marks[name] = sum(
                leg.shares_per_prior_share * price for leg, price in zip(legs, prices)
            )
            valued[name] = all(printed[leg.successor_index] for leg in legs)
            if valued[name]:
                missing_sessions[name] = 0
            last_observed[name] = np.nan
            share_claim_positions.extend(
                ShareClaimPosition(
                    day,
                    name,
                    leg.successor_index,
                    float(shares[name]) * leg.shares_per_prior_share,
                    float(last_observed[leg.successor_index]),
                    leg.delivery_session,
                )
                for leg in legs
            )
        undelivered_rows.append(
            sum(abs(shares[name] * marks[name]) for name in pending_distributions)
        )
        stale = int(np.sum((shares != 0.0) & ~valued))

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
            if bova_printed:
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
                    loan_covers=loan_covers,
                    loan_openings=loan_openings,
                    loan_name=name_count,
                    long_purchases=long_purchases,
                    long_sales=long_sales,
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
                        purpose="hedge",
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

        # Every name has at most one new-entry fill after reductions; contract
        # terms are shared within this session. Apply those actual quantities in
        # one vector operation, avoiding a whole-cohort scan for every fill.
        if traded_notional:
            settlements.append(
                (
                    spot_settlement_session(day, inputs.dates[day]),
                    free_cash - fill_cash_before,
                    np.r_[restricted_by_name, hedge_restricted_cash]
                    - fill_restricted_before,
                )
            )
        custody.fill(
            long_before_fill,
            long_purchases,
            long_sales,
            day,
            spot_settlement_session(day, inputs.dates[day]),
        )
        loans.fill(loan_covers, loan_openings, loan_session)
        rent_paid, fees_paid = loans.pay(day)
        loan_paid = float(rent_paid.sum() + fees_paid.sum())
        free_cash -= loan_paid
        loan_liability = float(loans.liability)
        loan_liability_rows.append(loan_liability)
        loan_payment_rows.append(loan_paid)
        loan_principal_rows.append(float(loans.principal.sum()))
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
            - loan_liability
        )
        settled_cash, settled_restricted = _settled_balances(
            free_cash, np.r_[restricted_by_name, hedge_restricted_cash], settlements
        )
        unsettled_cash = sum(free + float(r.sum()) for _, free, r in settlements)
        identity = (
            settled_cash
            + settled_restricted.sum()
            + unsettled_cash
            + marked_holdings
            + (hedge_shares * hedge_mark if hedge_shares != 0.0 else 0.0)
            + receivable_by_name.sum()
            - payable_by_name.sum()
            - loan_liability
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
        # This is a valuation sensitivity, never a fill or a source of cash.
        unpriced_notional = float(
            np.abs(shares[held_now & ~valued] * marks[held_now & ~valued]).sum()
        )
        unpriced_hedge = (
            abs(hedge_shares * hedge_mark)
            if hedge_shares != 0.0 and not bova_printed
            else 0.0
        )
        unpriced_notional += unpriced_hedge
        unpriced_count = int((held_now & ~valued).sum()) + int(unpriced_hedge > 0)
        unpriced_haircut_nav = current_nav - config.unpriced_haircut * unpriced_notional
        if day == day_count - 1:
            terminal_boundary_unpriced_inventory_notional = (
                unpriced_notional - unpriced_hedge
            )
            terminal_unpriced_hedge_notional = unpriced_hedge
        excluded_nav = current_nav
        if unpriced_hedge:
            excluded_nav -= hedge_restricted_cash + hedge_shares * hedge_mark
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
            ~valued
            | explicit_unresolved_action
            | (missing_sessions >= config.settlement_grace_sessions)
        )
        unresolved_claim_inventory = held_now & explicit_unresolved_action
        stale_mark_inventory = held_now & ~valued
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

        if portfolio_policy is not None:
            small_universe_shortfall = sizing_fill = sizing_mark = sizing_nav = np.nan
            occupancy_pending_count = occupancy_band_exhausted_count = np.nan
            occupancy_blocked_count = occupancy_exit_gap_count = (
                unclassified_occupancy
            ) = np.nan

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
        free_cash_income_rows.append(10_000.0 * free_cash_income / start_nav)
        debit_financing_rows.append(10_000.0 * debit_financing / start_nav)
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
        hedge_borrow_raw_rows.append(10_000.0 * hedge_borrow_raw / start_nav)
        hedge_borrow_fee_rows.append(10_000.0 * hedge_borrow_fee / start_nav)
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
        unsettled_cash_rows.append(unsettled_cash)
        free_cash_rows.append(settled_cash)
        restricted_rows.append(float(settled_restricted[:-1].sum()))
        hedge_restricted_rows.append(float(settled_restricted[-1]))
        receivable_rows.append(float(receivable_by_name.sum()))
        payable_rows.append(float(payable_by_name.sum()))
        holding_value_rows.append(marked_holdings)
        reconciliation_rows.append(reconciliation)
        all_cash_rows.append(all_cash)
        haircut_rows.append(unpriced_haircut_nav)
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
        unpriced_count_rows.append(unpriced_count)
        unpriced_notional_rows.append(unpriced_notional)
        unpriced_fraction_rows.append(unpriced_notional / current_nav)
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
    unpriced_fraction = float(np.max(unpriced_fraction_rows))
    economics_unresolved = (
        insolvent
        or action_uncertainty_seen
        or unresolved_count > 0
        or receivable_by_name.any()
        or payable_by_name.any()
        or bool(pending_distributions)
        or hedge_shares != 0.0
        or terminal_boundary_unpriced_inventory_notional > 0.0
        or terminal_unpriced_hedge_notional > 0.0
        or unpriced_fraction > config.unpriced_economics_unresolved_fraction_nav
        or (portfolio_policy is None and mean_gross < 0.5 * config.gross_target)
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
        free_cash_income_bps=np.asarray(free_cash_income_rows, dtype=np.float64),
        debit_financing_bps=np.asarray(debit_financing_rows, dtype=np.float64),
        cost_bps=np.asarray(cost_rows, dtype=np.float64),
        borrow_bps=np.asarray(borrow_rows, dtype=np.float64),
        unsettled_cash=np.asarray(unsettled_cash_rows, dtype=np.float64),
        loan_liability=np.asarray(loan_liability_rows, dtype=np.float64),
        loan_payment=np.asarray(loan_payment_rows, dtype=np.float64),
        loan_outstanding_principal=np.asarray(loan_principal_rows, dtype=np.float64),
        loan_charges=tuple(loan_charges),
        loan_cash_payments=tuple(loan_cash_payments),
        equity_borrow_raw_bps=np.asarray(equity_borrow_raw_rows, dtype=np.float64),
        equity_borrow_fee_bps=np.asarray(equity_borrow_fee_rows, dtype=np.float64),
        cdi_benchmark_bps=np.asarray(cdi_benchmark_rows, dtype=np.float64),
        borrowed_equity_weighted_annual_rate=np.asarray(
            held_short_borrow_rate_rows, dtype=np.float64
        ),
        borrowed_equity_principal_at_open=np.asarray(
            held_short_notional_rows, dtype=np.float64
        ),
        borrowed_equity_imputed_principal_at_open=np.asarray(
            held_short_imputed_notional_rows, dtype=np.float64
        ),
        borrowed_equity_placeholder_principal_at_open=np.asarray(
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
        undelivered_share_notional=np.asarray(undelivered_rows, dtype=np.float64),
        share_claim_positions=tuple(share_claim_positions),
        hedge_restricted_cash=np.asarray(hedge_restricted_rows, dtype=np.float64),
        receivables=np.asarray(receivable_rows, dtype=np.float64),
        payables=np.asarray(payable_rows, dtype=np.float64),
        marked_signed_holdings=np.asarray(holding_value_rows, dtype=np.float64),
        reconciliation_error=np.asarray(reconciliation_rows, dtype=np.float64),
        all_cash_nav=np.asarray(all_cash_rows, dtype=np.float64),
        unpriced_haircut_scenario_nav=np.asarray(haircut_rows, dtype=np.float64),
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
        unpriced_inventory_count=np.asarray(unpriced_count_rows, dtype=np.int64),
        unpriced_inventory_notional=np.asarray(
            unpriced_notional_rows, dtype=np.float64
        ),
        unpriced_inventory_fraction_nav=np.asarray(
            unpriced_fraction_rows, dtype=np.float64
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
        terminal_unpriced_hedge_notional=terminal_unpriced_hedge_notional,
        gross_target=config.gross_target,
        ineligible_hold_sessions=config.ineligible_hold_sessions,
        settlement_grace_sessions=config.settlement_grace_sessions,
        unpriced_haircut=config.unpriced_haircut,
        unpriced_economics_unresolved_fraction_nav=(
            config.unpriced_economics_unresolved_fraction_nav
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
        hedge_borrow_raw_bps=np.asarray(hedge_borrow_raw_rows, dtype=np.float64),
        hedge_borrow_fee_bps=np.asarray(hedge_borrow_fee_rows, dtype=np.float64),
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
