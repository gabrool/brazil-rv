from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from brazil_rv.v2.corporate_actions import AlignedActionTerms, apply_contractual_action


OrderSide = Literal["buy", "sell"]
OrderPurpose = Literal["entry", "exit", "risk_exit", "terminal_exit"]


@dataclass(frozen=True)
class LedgerConfig:
    k_per_side: int = 30
    buffer_per_side: int = 30
    gross_target: float = 2.0
    planned_gross_cap: float = 2.25
    planned_absolute_net_cap: float = 0.10
    planned_name_weight_cap: float = 0.05
    cost_bps_per_side: float = 4.0
    annual_borrow_rate: float = 0.02
    annual_debit_spread: float = 0.0
    short_proceeds_remuneration: float = 0.0
    initial_capital_brl: float = 1.0
    lot_size: int | None = None
    entry_expiry_sessions: int = 3
    forced_liquidation_haircut: float = 0.0
    max_missing_sessions: int = 10
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
        if not 0 <= self.short_proceeds_remuneration <= 1:
            raise ValueError("short-proceeds remuneration must be in [0, 1]")
        if self.initial_capital_brl <= 0 or not np.isfinite(self.initial_capital_brl):
            raise ValueError("initial capital must be positive and finite")
        if self.lot_size is not None and self.lot_size <= 0:
            raise ValueError("lot size must be positive when supplied")
        if self.entry_expiry_sessions < 1:
            raise ValueError("entry expiry must be at least one session")
        if not 0 <= self.forced_liquidation_haircut < 1:
            raise ValueError("valuation haircut must be in [0, 1)")
        if self.max_missing_sessions < 1:
            raise ValueError("max-missing sessions must be positive")


@dataclass(frozen=True)
class IntendedOrder:
    order_id: str
    security: str
    security_index: int
    decision_date: date
    decision_session: int
    side: OrderSide
    quantity: float
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
    unfilled_quantity: float
    reason: Literal["expired", "evaluation_end", "exit_instruction", "corporate_action"]


@dataclass
class _PendingOrder:
    order: IntendedOrder
    remaining_quantity: float


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


@dataclass(frozen=True)
class StatefulLedgerResult:
    dates: tuple[date, ...]
    nav: NDArray[np.float64]
    daily_net_return: NDArray[np.float64]
    net_excess_all_cash_bps: NDArray[np.float64]
    gross_pnl_bps: NDArray[np.float64]
    interest_bps: NDArray[np.float64]
    cost_bps: NDArray[np.float64]
    borrow_bps: NDArray[np.float64]
    gross_fraction_nav: NDArray[np.float64]
    turnover_fraction_nav: NDArray[np.float64]
    stale_mark_name_days: NDArray[np.int64]
    unresolved_action_name_days: NDArray[np.int64]
    valuation_scenario_count: NDArray[np.int64]
    position_sign: NDArray[np.int8]
    signed_shares: NDArray[np.float64]
    mark_price: NDArray[np.float64]
    free_cash: NDArray[np.float64]
    restricted_cash: NDArray[np.float64]
    receivables: NDArray[np.float64]
    payables: NDArray[np.float64]
    marked_signed_holdings: NDArray[np.float64]
    reconciliation_error: NDArray[np.float64]
    all_cash_nav: NDArray[np.float64]
    haircut_scenario_nav: NDArray[np.float64]
    unresolved_excluded_nav: NDArray[np.float64]
    pending_entry_count: NDArray[np.int64]
    pending_exit_count: NDArray[np.int64]
    cancelled_entry_count: NDArray[np.int64]
    blocked_entry_no_reference_count: NDArray[np.int64]
    retention_width: NDArray[np.int64]
    entry_blocked_small_universe: NDArray[np.bool_]
    actual_risk_breach: NDArray[np.bool_]
    planned_gross_fraction_nav: NDArray[np.float64]
    planned_net_fraction_nav: NDArray[np.float64]
    planned_name_weight_fraction_nav: NDArray[np.float64]
    holding_age_sessions: NDArray[np.int64]
    intended_orders: tuple[IntendedOrder, ...]
    fills: tuple[Fill, ...]
    cancellations: tuple[OrderCancellation, ...]
    unresolved_inventory_count: int
    unresolved_inventory_notional: float
    unresolved_receivable: float
    unresolved_payable: float
    insolvent: bool
    insolvency_date: date | None
    economics_unresolved: bool
    gross_target: float
    share_sizing_mode: Literal[
        "fractional_notional_research_proxy", "round_lot_close_proxy"
    ]

    def summary(self) -> dict[str, float | int | bool | str | None]:
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
        mean_turnover = float(np.mean(finite_turnover)) if finite_turnover.size else 0.0
        all_cash_terminal = float(self.all_cash_nav[-1])
        return {
            "mean_net_excess_bps_per_day": (
                float(np.mean(finite_excess)) if finite_excess.size else 0.0
            ),
            "annualized_net_excess_sharpe": sharpe,
            "compounded_net_excess_vs_all_cash": float(
                self.nav[-1] / all_cash_terminal - 1.0
            ),
            "compounded_net_excess_haircut_scenario": float(
                self.haircut_scenario_nav[-1] / all_cash_terminal - 1.0
            ),
            "compounded_net_excess_unresolved_excluded": float(
                self.unresolved_excluded_nav[-1] / all_cash_terminal - 1.0
            ),
            "mean_gross_fraction_nav": mean_gross,
            "mean_turnover_fraction_nav": mean_turnover,
            "average_holding_sessions_approximation": (
                2.0 * mean_gross / mean_turnover if mean_turnover > 0 else 0.0
            ),
            "stale_mark_name_days": int(self.stale_mark_name_days.sum()),
            "unresolved_action_name_days": int(self.unresolved_action_name_days.sum()),
            "valuation_scenario_count": int(self.valuation_scenario_count.sum()),
            "cancelled_entry_count": int(self.cancelled_entry_count.sum()),
            "intended_order_count": len(self.intended_orders),
            "fill_count": len(self.fills),
            "unresolved_inventory_count": self.unresolved_inventory_count,
            "unresolved_inventory_notional": self.unresolved_inventory_notional,
            "unresolved_receivable": self.unresolved_receivable,
            "unresolved_payable": self.unresolved_payable,
            "terminal_nav": float(self.nav[-1]),
            "terminal_nav_haircut_scenario": float(self.haircut_scenario_nav[-1]),
            "terminal_nav_unresolved_excluded": float(self.unresolved_excluded_nav[-1]),
            "insolvent": self.insolvent,
            "insolvency_date": (
                self.insolvency_date.isoformat()
                if self.insolvency_date is not None
                else None
            ),
            "economics_unresolved": self.economics_unresolved,
            "mark_mode": "raw",
            "share_sizing_mode": self.share_sizing_mode,
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


def _quantity(
    slot_notional: float, reference_price: float, lot_size: int | None
) -> float:
    quantity = slot_notional / reference_price
    if lot_size is not None:
        quantity = np.floor(quantity / lot_size) * lot_size
    return float(quantity)


def _risk(signed_values: NDArray[np.float64], nav: float) -> tuple[float, float, float]:
    if nav <= 0:
        return np.inf, np.inf, np.inf
    weights = signed_values / nav
    return (
        float(np.abs(weights).sum()),
        float(weights.sum()),
        float(np.max(np.abs(weights), initial=0.0)),
    )


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
) -> StatefulLedgerResult:
    """Run causal close-proxy orders, fills, and a raw signed-share ledger."""

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
    )
    day_count, name_count = inputs.score.shape
    shares = np.zeros(name_count, dtype=np.float64)
    marks = np.full(name_count, np.nan, dtype=np.float64)
    # This is the last observed close strictly before the first decision.  It
    # lets the first evaluation session form the same causal order it would
    # form inside a longer continuous ledger, without reading that session's
    # later fill print.
    last_observed = inputs.initial_reference_price.copy()
    restricted_by_name = np.zeros(name_count, dtype=np.float64)
    receivable_by_name = np.zeros(name_count, dtype=np.float64)
    payable_by_name = np.zeros(name_count, dtype=np.float64)
    unresolved_action = np.zeros(name_count, dtype=np.bool_)
    missing_sessions = np.zeros(name_count, dtype=np.int64)
    entry_session = np.full(name_count, -1, dtype=np.int64)
    scenario_seen = np.zeros(name_count, dtype=np.bool_)
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

    nav_rows: list[float] = []
    daily_rows: list[float] = []
    excess_rows: list[float] = []
    gross_pnl_rows: list[float] = []
    interest_rows: list[float] = []
    cost_rows: list[float] = []
    borrow_rows: list[float] = []
    gross_rows: list[float] = []
    turnover_rows: list[float] = []
    stale_rows: list[int] = []
    unresolved_action_rows: list[int] = []
    scenario_count_rows: list[int] = []
    position_rows: list[NDArray[np.int8]] = []
    share_rows: list[NDArray[np.float64]] = []
    mark_rows: list[NDArray[np.float64]] = []
    free_cash_rows: list[float] = []
    restricted_rows: list[float] = []
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
    retention_rows: list[int] = []
    small_universe_rows: list[bool] = []
    risk_breach_rows: list[bool] = []
    planned_gross_rows: list[float] = []
    planned_net_rows: list[float] = []
    planned_name_rows: list[float] = []
    age_rows: list[NDArray[np.int64]] = []
    insolvent = False
    insolvency_date: date | None = None
    action_uncertainty_seen = False

    def submit_order(
        day: int,
        name: int,
        side: OrderSide,
        quantity: float,
        reference_price: float,
        purpose: OrderPurpose,
        expiry_session: int | None,
    ) -> _PendingOrder:
        nonlocal next_order_id
        order = IntendedOrder(
            order_id=f"order-{next_order_id:08d}",
            security=inputs.securities[name],
            security_index=name,
            decision_date=inputs.dates[day],
            decision_session=day,
            side=side,
            quantity=quantity,
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
        return _PendingOrder(order=order, remaining_quantity=quantity)

    def cancel_for_action(
        pending_map: dict[int, _PendingOrder], name: int, day: int
    ) -> bool:
        """Cancel an unfilled order before its security units/claim can change."""

        pending = pending_map.pop(name, None)
        if pending is None:
            return False
        cancellations.append(
            OrderCancellation(
                order_id=pending.order.order_id,
                security=pending.order.security,
                security_index=pending.order.security_index,
                cancellation_date=inputs.dates[day],
                cancellation_session=day,
                unfilled_quantity=pending.remaining_quantity,
                reason="corporate_action",
            )
        )
        return pending.order.purpose == "entry"

    for day in range(day_count):
        cancelled_today = 0
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
        if not np.isclose(start_identity, start_nav, rtol=1e-12, atol=1e-12):
            raise RuntimeError("opening ledger identity does not reconcile")

        newly_uncertain_action = (shares != 0.0) & ~inputs.action_resolved[day]
        action_uncertainty_seen |= bool(newly_uncertain_action.any())
        unresolved_action |= newly_uncertain_action
        # Apply contractual terms to shares held before the session. Cash terms
        # become claims; only the later payment mask transfers them to cash.
        for name in np.flatnonzero(inputs.has_action[day]):
            successor = int(inputs.action_successor[day, name])
            affected_orders = {int(name), successor}
            for affected_name in sorted(affected_orders):
                cancelled_today += int(
                    cancel_for_action(pending_entries, affected_name, day)
                )
                cancel_for_action(pending_exits, affected_name, day)
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
            else:
                if shares[successor] != 0.0:
                    raise ValueError(
                        "conversion successor already has ledger inventory"
                    )
                shares[name] = 0.0
                marks[name] = np.nan
                last_observed[name] = np.nan
                shares[successor] = float(new_shares)
                marks[successor] = converted_mark
                # Reference marks belong to the tradable claim even when no
                # inventory existed at the conversion.  Do not overwrite the
                # causal predecessor-derived reference with the inventory-only
                # converted mark (NaN in that case).
                last_observed[successor] = converted_reference
                restricted_by_name[successor] += restricted_by_name[name]
                restricted_by_name[name] = 0.0
                unresolved_action[successor] |= unresolved_action[name]
                if np.isfinite(converted_mark) and converted_mark <= 0.0:
                    unresolved_action[successor] = True
                unresolved_action[name] = False
                entry_session[successor] = entry_session[name]
                entry_session[name] = -1
            if (
                successor == name
                and np.isfinite(converted_mark)
                and converted_mark <= 0.0
            ):
                unresolved_action[name] = True
            if q == 0.0 and restricted_by_name[name] != 0.0:
                free_cash += restricted_by_name[name]
                restricted_by_name[name] = 0.0
                entry_session[name] = -1

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
        interest = (
            max(free_cash, 0.0) * cash_rate
            + min(free_cash, 0.0) * debit_rate
            + restricted_by_name.sum() * cash_rate * config.short_proceeds_remuneration
        )
        borrow = (
            config.annual_borrow_rate / config.annual_sessions * short_value_at_open
        )
        free_cash += interest - borrow

        eligible = (
            inputs.score_valid[day]
            & inputs.membership[day]
            & np.isfinite(inputs.score[day])
        )
        eligible_names = np.flatnonzero(eligible)
        order = (
            eligible_names[np.argsort(inputs.score[day, eligible_names], kind="stable")]
            if eligible_names.size
            else eligible_names
        )
        ranks = np.full(name_count, -1, dtype=np.int64)
        ranks[order] = np.arange(order.size)
        retention = min(
            config.k_per_side + config.buffer_per_side,
            len(order) // 2,
        )
        retention_rows.append(retention)
        small_universe = len(order) < 2 * config.k_per_side
        small_universe_rows.append(small_universe)

        signed_values = np.zeros(name_count, dtype=np.float64)
        held = shares != 0.0
        signed_values[held] = shares[held] * marks[held]
        actual_gross, actual_net, actual_name = _risk(signed_values, start_nav)
        risk_breach = (
            actual_gross > config.planned_gross_cap + 1e-12
            or abs(actual_net) > config.planned_absolute_net_cap + 1e-12
            or actual_name > config.planned_name_weight_cap + 1e-12
        )

        exit_required: set[int] = set()
        for name in np.flatnonzero(held):
            kept = (
                eligible[name]
                and retention > 0
                and not unresolved_action[name]
                and missing_sessions[name] < config.max_missing_sessions
                and (
                    ranks[name] >= len(order) - retention
                    if shares[name] > 0.0
                    else ranks[name] < retention
                )
            )
            if not kept:
                exit_required.add(int(name))
        if risk_breach:
            # Full-name exits are the transparent reference reduction. They
            # remain risk until a later observed price actually fills them.
            exit_required.update(int(name) for name in np.flatnonzero(held))
        if day == day_count - 1:
            exit_required.update(int(name) for name in np.flatnonzero(held))

        entries_to_cancel = (
            set(pending_entries)
            if risk_breach
            else exit_required & pending_entries.keys()
        )
        for name in sorted(entries_to_cancel):
            pending = pending_entries.pop(name)
            cancellations.append(
                OrderCancellation(
                    order_id=pending.order.order_id,
                    security=inputs.securities[name],
                    security_index=name,
                    cancellation_date=inputs.dates[day],
                    cancellation_session=day,
                    unfilled_quantity=pending.remaining_quantity,
                    reason="exit_instruction",
                )
            )
            cancelled_today += 1

        for name in sorted(exit_required):
            if name in pending_exits or shares[name] == 0.0:
                continue
            purpose: OrderPurpose
            if day == day_count - 1:
                purpose = "terminal_exit"
            elif risk_breach:
                purpose = "risk_exit"
            else:
                purpose = "exit"
            pending_exits[name] = submit_order(
                day,
                name,
                "sell" if shares[name] > 0.0 else "buy",
                abs(float(shares[name])),
                float(marks[name]),
                purpose,
                None,
            )

        if day == day_count - 1:
            for name, pending in tuple(pending_entries.items()):
                cancellations.append(
                    OrderCancellation(
                        order_id=pending.order.order_id,
                        security=inputs.securities[name],
                        security_index=name,
                        cancellation_date=inputs.dates[day],
                        cancellation_session=day,
                        unfilled_quantity=pending.remaining_quantity,
                        reason="evaluation_end",
                    )
                )
                del pending_entries[name]
                cancelled_today += 1

        blocked_reference = 0
        if (
            day < day_count - 1
            and not small_universe
            and not risk_breach
            and not pending_exits
        ):
            slot_notional = start_nav * config.gross_target / (2 * config.k_per_side)
            occupied = set(np.flatnonzero(shares != 0.0).tolist()) | set(
                pending_entries
            )
            long_occupied = {int(name) for name in np.flatnonzero(shares > 0.0)} | {
                name
                for name, pending in pending_entries.items()
                if pending.order.side == "buy"
            }
            short_occupied = {int(name) for name in np.flatnonzero(shares < 0.0)} | {
                name
                for name, pending in pending_entries.items()
                if pending.order.side == "sell"
            }
            long_slots = max(config.k_per_side - len(long_occupied), 0)
            short_slots = max(config.k_per_side - len(short_occupied), 0)
            long_band = [int(name) for name in order[-config.k_per_side :][::-1]]
            short_band = [int(name) for name in order[: config.k_per_side]]
            if set(long_band) & set(short_band):
                raise RuntimeError("long and short entry bands overlap")
            long_candidates = [name for name in long_band if name not in occupied]
            short_candidates = [name for name in short_band if name not in occupied]
            planned_values = signed_values.copy()
            for name, pending in pending_entries.items():
                sign = 1.0 if pending.order.side == "buy" else -1.0
                planned_values[name] += (
                    sign * pending.remaining_quantity * pending.order.reference_price
                )
            for long_name, short_name in zip(
                long_candidates[:long_slots],
                short_candidates[:short_slots],
                strict=False,
            ):
                long_reference = float(last_observed[long_name])
                short_reference = float(last_observed[short_name])
                if (
                    not np.isfinite(long_reference)
                    or long_reference <= 0.0
                    or not np.isfinite(short_reference)
                    or short_reference <= 0.0
                ):
                    blocked_reference += 2
                    break
                long_quantity = _quantity(
                    slot_notional, long_reference, config.lot_size
                )
                short_quantity = _quantity(
                    slot_notional, short_reference, config.lot_size
                )
                if long_quantity <= 0.0 or short_quantity <= 0.0:
                    blocked_reference += 2
                    break
                proposed = planned_values.copy()
                proposed[long_name] += long_quantity * long_reference
                proposed[short_name] -= short_quantity * short_reference
                planned_gross, planned_net, planned_name = _risk(proposed, start_nav)
                if (
                    planned_gross > config.planned_gross_cap + 1e-12
                    or abs(planned_net) > config.planned_absolute_net_cap + 1e-12
                    or planned_name > config.planned_name_weight_cap + 1e-12
                ):
                    break
                expiry = day + config.entry_expiry_sessions - 1
                pending_entries[long_name] = submit_order(
                    day,
                    long_name,
                    "buy",
                    long_quantity,
                    long_reference,
                    "entry",
                    expiry,
                )
                pending_entries[short_name] = submit_order(
                    day,
                    short_name,
                    "sell",
                    short_quantity,
                    short_reference,
                    "entry",
                    expiry,
                )
                planned_values = proposed
        blocked_reference_rows.append(blocked_reference)

        # Only now may current-session prints affect the result. This makes the
        # immutable intended-order set invariant to those later observations.
        printed = np.isfinite(inputs.raw_close[day]) & (inputs.raw_close[day] > 0.0)
        traded_notional = 0.0
        costs = 0.0
        cost_rate = config.cost_bps_per_side / 10_000.0
        for pending_map in (pending_exits, pending_entries):
            for name, pending in tuple(pending_map.items()):
                if not printed[name] or unresolved_action[name]:
                    continue
                fraction = float(inputs.fill_fraction[day, name])
                quantity = pending.remaining_quantity * fraction
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
                if before == 0.0 and shares[name] != 0.0:
                    entry_session[name] = day
                if before != 0.0 and shares[name] == 0.0:
                    entry_session[name] = -1
                    unresolved_action[name] = False
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
                pending.remaining_quantity -= quantity
                tolerance = max(1e-12, pending.order.quantity * 1e-12)
                if pending.remaining_quantity <= tolerance:
                    del pending_map[name]

        for name, pending in tuple(pending_entries.items()):
            expiry = pending.order.expiry_session
            if expiry is not None and day >= expiry:
                cancellations.append(
                    OrderCancellation(
                        order_id=pending.order.order_id,
                        security=inputs.securities[name],
                        security_index=name,
                        cancellation_date=inputs.dates[day],
                        cancellation_session=day,
                        unfilled_quantity=pending.remaining_quantity,
                        reason="expired",
                    )
                )
                del pending_entries[name]
                cancelled_today += 1

        for name in range(name_count):
            if printed[name]:
                last_observed[name] = inputs.raw_close[day, name]
                if shares[name] != 0.0:
                    marks[name] = inputs.raw_close[day, name]
                    missing_sessions[name] = 0
            elif shares[name] != 0.0:
                missing_sessions[name] += 1
        stale = int(np.sum((shares != 0.0) & ~printed))

        held_now = shares != 0.0
        marked_holdings = float(np.sum(shares[held_now] * marks[held_now]))
        current_nav = _equity(
            free_cash,
            restricted_by_name,
            shares,
            marks,
            receivable_by_name,
            payable_by_name,
        )
        identity = (
            free_cash
            + restricted_by_name.sum()
            + marked_holdings
            + receivable_by_name.sum()
            - payable_by_name.sum()
        )
        reconciliation = current_nav - identity
        all_cash *= 1.0 + inputs.cdi[day]

        unresolved = held_now & (
            unresolved_action
            | (missing_sessions >= config.max_missing_sessions)
            | (day == day_count - 1)
        )
        newly_scenario = unresolved & ~scenario_seen
        scenario_seen |= unresolved
        haircut_adjustment = 0.0
        for name in np.flatnonzero(unresolved):
            scenario_mark = marks[name] * (
                1.0 - config.forced_liquidation_haircut
                if shares[name] > 0.0
                else 1.0 + config.forced_liquidation_haircut
            )
            haircut_adjustment += shares[name] * (scenario_mark - marks[name])
        haircut_nav = current_nav + haircut_adjustment
        excluded_nav = current_nav
        if unresolved.any():
            excluded_nav -= float(restricted_by_name[unresolved].sum())
            excluded_nav -= float(np.sum(shares[unresolved] * marks[unresolved]))
        unresolved_claim = ((receivable_by_name != 0.0) | (payable_by_name != 0.0)) & (
            (day == day_count - 1) | unresolved_action
        )
        excluded_names = unresolved | unresolved_claim
        if excluded_names.any():
            excluded_nav -= float(receivable_by_name[excluded_names].sum())
            excluded_nav += float(payable_by_name[excluded_names].sum())

        signed_values = np.zeros(name_count, dtype=np.float64)
        signed_values[held_now] = shares[held_now] * marks[held_now]
        gross, net, name_weight = _risk(signed_values, current_nav)
        end_risk_breach = (
            gross > config.planned_gross_cap + 1e-12
            or abs(net) > config.planned_absolute_net_cap + 1e-12
            or name_weight > config.planned_name_weight_cap + 1e-12
        )
        planned_values = signed_values.copy()
        for name, pending in pending_entries.items():
            sign = 1.0 if pending.order.side == "buy" else -1.0
            planned_values[name] += (
                sign * pending.remaining_quantity * pending.order.reference_price
            )
        planned_gross, planned_net, planned_name = _risk(planned_values, start_nav)

        economic_pnl = current_nav - start_nav - interest + borrow + costs
        daily_return = current_nav / start_nav - 1.0
        nav_rows.append(current_nav)
        daily_rows.append(daily_return)
        excess_rows.append(10_000.0 * (daily_return - inputs.cdi[day]))
        gross_pnl_rows.append(10_000.0 * economic_pnl / start_nav)
        interest_rows.append(10_000.0 * interest / start_nav)
        cost_rows.append(10_000.0 * costs / start_nav)
        borrow_rows.append(10_000.0 * borrow / start_nav)
        gross_rows.append(gross if np.isfinite(gross) else np.nan)
        turnover_rows.append(traded_notional / start_nav)
        stale_rows.append(stale)
        unresolved_action_rows.append(int(unresolved_action.sum()))
        scenario_count_rows.append(int(newly_scenario.sum()))
        signs = np.zeros(name_count, dtype=np.int8)
        signs[shares > 0.0] = 1
        signs[shares < 0.0] = -1
        position_rows.append(signs)
        share_rows.append(shares.copy())
        mark_rows.append(marks.copy())
        free_cash_rows.append(free_cash)
        restricted_rows.append(float(restricted_by_name.sum()))
        receivable_rows.append(float(receivable_by_name.sum()))
        payable_rows.append(float(payable_by_name.sum()))
        holding_value_rows.append(marked_holdings)
        reconciliation_rows.append(reconciliation)
        all_cash_rows.append(all_cash)
        haircut_rows.append(haircut_nav)
        excluded_rows.append(excluded_nav)
        pending_entry_rows.append(len(pending_entries))
        pending_exit_rows.append(len(pending_exits))
        cancelled_rows.append(cancelled_today)
        risk_breach_rows.append(end_risk_breach)
        planned_gross_rows.append(planned_gross)
        planned_net_rows.append(planned_net)
        planned_name_rows.append(planned_name)
        ages = np.zeros(name_count, dtype=np.int64)
        held_with_age = held_now & (entry_session >= 0)
        ages[held_with_age] = day - entry_session[held_with_age] + 1
        age_rows.append(ages)

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
    gross_array = np.asarray(gross_rows, dtype=np.float64)
    finite_gross = gross_array[np.isfinite(gross_array)]
    mean_gross = float(np.mean(finite_gross)) if finite_gross.size else 0.0
    economics_unresolved = (
        insolvent
        or action_uncertainty_seen
        or scenario_seen.any()
        or receivable_by_name.any()
        or payable_by_name.any()
        or mean_gross < 0.5 * config.gross_target
    )
    return StatefulLedgerResult(
        dates=inputs.dates[:completed_days],
        nav=np.asarray(nav_rows, dtype=np.float64),
        daily_net_return=np.asarray(daily_rows, dtype=np.float64),
        net_excess_all_cash_bps=np.asarray(excess_rows, dtype=np.float64),
        gross_pnl_bps=np.asarray(gross_pnl_rows, dtype=np.float64),
        interest_bps=np.asarray(interest_rows, dtype=np.float64),
        cost_bps=np.asarray(cost_rows, dtype=np.float64),
        borrow_bps=np.asarray(borrow_rows, dtype=np.float64),
        gross_fraction_nav=gross_array,
        turnover_fraction_nav=np.asarray(turnover_rows, dtype=np.float64),
        stale_mark_name_days=np.asarray(stale_rows, dtype=np.int64),
        unresolved_action_name_days=np.asarray(unresolved_action_rows, dtype=np.int64),
        valuation_scenario_count=np.asarray(scenario_count_rows, dtype=np.int64),
        position_sign=np.stack(position_rows),
        signed_shares=np.stack(share_rows),
        mark_price=np.stack(mark_rows),
        free_cash=np.asarray(free_cash_rows, dtype=np.float64),
        restricted_cash=np.asarray(restricted_rows, dtype=np.float64),
        receivables=np.asarray(receivable_rows, dtype=np.float64),
        payables=np.asarray(payable_rows, dtype=np.float64),
        marked_signed_holdings=np.asarray(holding_value_rows, dtype=np.float64),
        reconciliation_error=np.asarray(reconciliation_rows, dtype=np.float64),
        all_cash_nav=np.asarray(all_cash_rows, dtype=np.float64),
        haircut_scenario_nav=np.asarray(haircut_rows, dtype=np.float64),
        unresolved_excluded_nav=np.asarray(excluded_rows, dtype=np.float64),
        pending_entry_count=np.asarray(pending_entry_rows, dtype=np.int64),
        pending_exit_count=np.asarray(pending_exit_rows, dtype=np.int64),
        cancelled_entry_count=np.asarray(cancelled_rows, dtype=np.int64),
        blocked_entry_no_reference_count=np.asarray(
            blocked_reference_rows[:completed_days], dtype=np.int64
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
        intended_orders=tuple(orders),
        fills=tuple(fills),
        cancellations=tuple(cancellations),
        unresolved_inventory_count=unresolved_count,
        unresolved_inventory_notional=unresolved_notional,
        unresolved_receivable=float(receivable_by_name.sum()),
        unresolved_payable=float(payable_by_name.sum()),
        insolvent=insolvent,
        insolvency_date=insolvency_date,
        economics_unresolved=economics_unresolved,
        gross_target=config.gross_target,
        share_sizing_mode=(
            "fractional_notional_research_proxy"
            if config.lot_size is None
            else "round_lot_close_proxy"
        ),
    )


def ledger_sensitivity_grid(**inputs: object) -> dict[str, StatefulLedgerResult]:
    """Run the registered financing/cost grid and valuation sensitivities."""

    return {
        name: simulate_stateful_ledger(config=config, **inputs)  # type: ignore[arg-type]
        for name, config in ledger_configurations().items()
    }


def ledger_configurations() -> dict[str, LedgerConfig]:
    """Return named reference assumptions without changing intended orders."""

    headline = LedgerConfig()
    configurations = {
        f"cost_{cost:g}_borrow_{borrow:g}": replace(
            headline, cost_bps_per_side=cost, annual_borrow_rate=borrow
        )
        for cost in (2.0, 4.0, 7.0)
        for borrow in (0.02, 0.04)
    }
    configurations.update(
        {
            "sensitivity_buffer_0": replace(headline, buffer_per_side=0),
            "sensitivity_buffer_2k": replace(
                headline, buffer_per_side=2 * headline.k_per_side
            ),
            "sensitivity_short_proceeds_full": replace(
                headline, short_proceeds_remuneration=1.0
            ),
            "sensitivity_haircut_30pct": replace(
                headline, forced_liquidation_haircut=0.30
            ),
        }
    )
    return configurations
