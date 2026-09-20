"""Differentiable share/cash transitions for the frozen target-policy contract.

The independent NumPy stateful ledger verifies this training implementation.
Market/action/payment arrays are realizations, consumed only after target
notional and exit fractions have been computed from the previous state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import Tensor

from .stateful_ledger import LedgerConfig
from .share_distributions import (
    basket_prices,
    claim_delivery_session,
    recognize_distribution,
)
from .loan_contracts import LoanContracts, LoanSession, spot_settlement_session
from .share_custody import ShareCustody


def tensor(values) -> Tensor:
    return torch.as_tensor(values, dtype=torch.float64)


@dataclass
class PortfolioAccount:
    shares: Tensor
    marks: Tensor
    trade_restricted: Tensor
    claims: Tensor
    trade_cash: Tensor
    cost_basis: Tensor
    pending_exit: Tensor
    entry_day: np.ndarray
    missing_sessions: np.ndarray
    ineligible_sessions: np.ndarray
    proceeds_releases: list[tuple[int, Tensor]]
    bonus_proceeds: list[tuple[int, Tensor]]
    bonus_shares: Tensor
    bonus_delivery: np.ndarray
    payments: list[tuple[int, Tensor]]
    distributions: dict
    retired_sources: set[int]
    unquoted_deliveries: set[int]
    loans: LoanContracts
    custody: ShareCustody
    config: LedgerConfig
    settlements: list[tuple[int | None, Tensor, Tensor]]
    funding_day: int
    funding_cash: Tensor
    funding_restricted: Tensor
    prepared_loan_cash_payment: Tensor

    @classmethod
    def empty(cls, reference, *, config=LedgerConfig()):
        """Reference axis includes the optional hedge as the last coordinate."""
        n = len(reference)
        if any(event.security_index >= n for event in config.loan_recalls):
            raise ValueError("recall security index is outside the account axis")
        zero = torch.zeros(n, dtype=torch.float64)
        return cls(
            shares=zero.clone(),
            marks=tensor(np.nan_to_num(reference, nan=0.0)),
            trade_restricted=zero.clone(),
            claims=zero.clone(),
            trade_cash=tensor(config.initial_capital_brl),
            cost_basis=zero.clone(),
            pending_exit=zero.clone(),
            entry_day=np.full(n, -1, dtype=np.int64),
            missing_sessions=np.zeros(n, dtype=np.int64),
            ineligible_sessions=np.zeros(n, dtype=np.int64),
            proceeds_releases=[],
            bonus_proceeds=[],
            bonus_shares=zero.clone(),
            bonus_delivery=np.full(n, -1, dtype=np.int64),
            payments=[],
            distributions={},
            retired_sources=set(),
            unquoted_deliveries=set(),
            loans=LoanContracts(
                n,
                config.borrow_fee_modality,
                config.borrow_fee_multiplier,
                config.annual_sessions,
                config.electronic_loan_settlement_days,
            ),
            custody=ShareCustody(n),
            config=config,
            settlements=[],
            funding_day=-1,
            funding_cash=tensor(config.initial_capital_brl),
            funding_restricted=tensor(0.0),
            prepared_loan_cash_payment=tensor(0.0),
        )

    @property
    def reserved_weights(self):
        return self.bonus_shares * self.marks / self.nav

    def settled_balances(self):
        pending_free = sum((free for _, free, _ in self.settlements), tensor(0.0))
        pending_restricted = sum(
            (r for _, _, r in self.settlements), torch.zeros_like(self.trade_restricted)
        )
        restricted = self.trade_restricted - pending_restricted
        # A release cannot withdraw proceeds that have not yet settled. A
        # negative earmark reclassifies future cash; it is not an actual deposit.
        free = self.trade_cash - pending_free + restricted.clamp_max(0).sum()
        return free, restricted.clamp_min(0)

    @property
    def cash(self):
        return self.settled_balances()[0]

    @property
    def restricted(self):
        return self.settled_balances()[1]

    @property
    def unsettled_cash(self):
        return sum((f + r.sum() for _, f, r in self.settlements), tensor(0.0))

    @property
    def nav(self):
        return (
            self.trade_cash
            + self.trade_restricted.sum()
            + self.claims.sum()
            + (self.shares * self.marks).sum()
            - self.loans.liability
        )

    @property
    def weights(self):
        return self.shares * self.marks / self.nav

    @property
    def market_weights(self):
        """Expand basket entitlements to their actual underlying risk exposures."""
        values = self.shares * self.marks
        for name, legs in self.distributions.items():
            values = values.clone()
            values[name] = 0
            prices = basket_prices(legs, self.marks.detach().numpy())
            for leg, price in zip(legs, prices):
                destination = leg.successor_index
                values[destination] = values[destination] + (
                    self.shares[name] * leg.shares_per_prior_share * price
                )
        return values / self.nav

    def prepare_day(self, day, *, realize_auctions=False):
        """Known settlement/delivery dates inform decisions, never current prices."""
        fresh = day != self.funding_day
        if fresh:
            self.funding_cash, restricted = self.settled_balances()
            self.funding_restricted = restricted.sum()
            self.funding_day = day
            self._pay_claims(day)
            self.proceeds_releases = [
                (due, value) for due, value in self.proceeds_releases if due >= day
            ]
            held_proceeds = []
            for due, value in self.bonus_proceeds:
                if due <= day:
                    self.trade_cash = self.trade_cash + value.sum()
                    self.trade_restricted = self.trade_restricted - value
                else:
                    held_proceeds.append((due, value))
            self.bonus_proceeds = held_proceeds
            # A previously valued liability pays with the same known arrival
            # that releases its proceeds; do not expose spendable proceeds to
            # today's policy while hiding the simultaneous principal debit.
            self.prepared_loan_cash_payment = self.loans.settle_cash_claims(
                day, pay_only=True
            ).sum()
            self.trade_cash = self.trade_cash - self.prepared_loan_cash_payment
            self.custody.settle(day)
            self.bonus_shares = torch.where(
                torch.as_tensor(self.bonus_delivery > day), self.bonus_shares, 0.0
            )
            # Value-date obligations settle within this session. Close-to-close
            # interest below belongs to balances carried from the previous close.
            self.settlements = [
                item for item in self.settlements if item[0] is None or item[0] > day
            ]
        for name, legs in tuple(self.distributions.items()):
            for leg in tuple(legs):
                auction = leg.fractional_auction
                if (
                    realize_auctions
                    and leg.delivery_session is None
                    and auction is not None
                    and auction.available_session == day
                ):
                    amount = torch.zeros_like(self.claims)
                    amount[name] = (
                        self.shares[name]
                        * leg.shares_per_prior_share
                        * auction.cash_per_share
                    )
                    self.claims = self.claims + amount
                    self.payments.append((auction.payment_session, amount))
                    if auction.provision_loan_fractions:
                        release = self.trade_restricted[name]
                        r = torch.zeros_like(self.trade_restricted)
                        r[name] = -release
                        if auction.payment_session > day:
                            self.settlements.append((auction.payment_session, release, r))
                        self.trade_cash = self.trade_cash + release
                        self.trade_restricted = self.trade_restricted + r
                    for field in ("shares", "marks", "cost_basis", "pending_exit"):
                        values = getattr(self, field).clone()
                        values[name] = 0
                        setattr(self, field, values)
                    self.entry_day[name] = -1
                    legs.remove(leg)
                    continue
                early = (
                    leg.disposal_session == day
                    and self.shares[name].detach().item() > 0
                    and not (self.loans.name == name).any()
                )
                due = claim_delivery_session(
                    leg, borrowed=self.shares[name].detach().item() < 0
                )
                if due != day and not early:
                    continue
                prices = basket_prices(legs, self.marks.detach().numpy())
                values = [
                    item.shares_per_prior_share * price
                    for item, price in zip(legs, prices)
                ]
                allocation = values[legs.index(leg)] / sum(values)
                source_entry_day = self.entry_day[name]
                incoming = self.shares[name] * leg.shares_per_prior_share
                fraction = tensor(0.0)
                fraction_basis = tensor(0.0)
                if auction is not None and auction.provision_loan_fractions:
                    if not early and any(
                        float(q[name].detach()) > 1e-12
                        for _, q in self.custody.receipts
                    ):
                        raise ValueError(
                            "provisioned fractions require settled source purchases"
                        )
                    loan_fraction = self.loans.provision_fractions(
                        name, leg.shares_per_prior_share, day, auction
                    )
                    if loan_fraction.detach().item() > 0:
                        if incoming.detach().item() >= 0:
                            raise ValueError(
                                "gross custody offsets require separate fraction terms"
                            )
                        fraction = -loan_fraction
                        fraction_basis = self.cost_basis[name] * fraction / incoming
                        allocation = (incoming - fraction) / incoming
                        incoming = incoming - fraction
                if auction is not None and incoming.detach().item() > 0:
                    # Shareholder fractions remain a non-tradable claim. Loan
                    # quantities retain all fractions under the separate B3 rule.
                    if not early and any(
                        float(q[name].detach()) > 1e-12
                        for _, q in self.custody.receipts
                    ):
                        raise ValueError(
                            "fraction auction requires settled source purchases"
                        )
                    fraction = incoming - incoming.floor()
                    fraction_basis = self.cost_basis[name] * fraction / incoming
                    basis = self.cost_basis.clone()
                    basis[name] = basis[name] - fraction_basis
                    self.cost_basis = basis
                    incoming = incoming.floor()
                if (
                    self.marks[leg.successor_index].detach().item() <= 0
                    and leg.opening_mark is not None
                ):
                    self.unquoted_deliveries.add(leg.successor_index)
                self.pending_exit, _ = self._deliver_shares(
                    name,
                    leg.successor_index,
                    incoming,
                    allocation,
                    tensor(prices[legs.index(leg)]),
                    self.pending_exit,
                    torch.zeros_like(self.shares),
                    final=len(legs) == 1,
                    day=day,
                    ratio=leg.shares_per_prior_share,
                    receipt_day=leg.delivery_session if early else None,
                    loan_allocation=(
                        leg.loan_principal_fraction
                        / sum(item.loan_principal_fraction for item in legs)
                        if sum(item.loan_principal_fraction for item in legs) > 0
                        else 1.0
                    ),
                )
                if fraction.detach().item() != 0:
                    shares, basis = self.shares.clone(), self.cost_basis.clone()
                    shares[name] = fraction / leg.shares_per_prior_share
                    basis[name] = fraction_basis
                    self.shares, self.cost_basis = shares, basis
                    self.entry_day[name] = source_entry_day
                    legs[legs.index(leg)] = replace(
                        leg, delivery_session=None, disposal_session=None
                    )
                else:
                    legs.remove(leg)
            if legs:
                marks = self.marks.clone()
                prices = basket_prices(legs, self.marks.detach().numpy())
                marks[name] = sum(
                    leg.shares_per_prior_share * price
                    for leg, price in zip(legs, prices)
                )
                self.marks = marks
            else:
                del self.distributions[name]
        if fresh:
            # A known morning delivery changes the identity held at the notice.
            # The post-realization preparation of this same day is idempotent.
            self.loans.observe_recalls(day, self.config.loan_recalls)

    def _pay_claims(self, day):
        unpaid = []
        for pay, amount in self.payments:
            if pay == day:
                self.trade_cash = self.trade_cash + amount.sum()
                self.claims = self.claims - amount
            else:
                unpaid.append((pay, amount))
        self.payments = unpaid

    def detach(self):
        """Truncate gradients without resetting positions, financing or claims."""
        for name in (
            "shares",
            "marks",
            "trade_restricted",
            "claims",
            "bonus_shares",
            "trade_cash",
            "cost_basis",
            "pending_exit",
            "funding_cash",
            "funding_restricted",
            "prepared_loan_cash_payment",
        ):
            setattr(self, name, getattr(self, name).detach())
        for key in ("payments", "proceeds_releases", "bonus_proceeds"):
            setattr(
                self,
                key,
                [(day, amount.detach()) for day, amount in getattr(self, key)],
            )
        self.loans.detach()
        self.custody.detach()
        self.settlements = [(d, f.detach(), r.detach()) for d, f, r in self.settlements]

    def eligibility(self, active_score, prior_unresolved):
        held = self.shares.detach().numpy() != 0
        eligible = np.asarray(active_score, dtype=bool)
        self.ineligible_sessions[~held | eligible] = 0
        self.ineligible_sessions[held & ~eligible] += 1
        allowed = eligible & ~prior_unresolved & (self.marks.detach().numpy() > 0)
        if self.retired_sources:
            allowed[list(self.retired_sources)] = False
        if self.unquoted_deliveries:
            allowed[list(self.unquoted_deliveries)] = False
        required = (self.ineligible_sessions > self.config.ineligible_hold_sessions) | (
            self.missing_sessions >= self.config.settlement_grace_sessions
        )
        required |= self.loans.short_blocked(
            self.funding_day, self.config.loan_recalls
        ) & (self.shares.detach().numpy() < 0)
        if self.distributions:
            required[list(self.distributions)] = False
        return allowed, required

    def _fill(
        self, signed_quantity: Tensor, price: Tensor, cost_rate: Tensor, loan_session
    ):
        """Self-financing fill with segregated short proceeds, including reversals."""
        short_shares = (-self.shares).clamp_min(0)
        cover = torch.minimum(signed_quantity.clamp_min(0), short_shares)
        sell = torch.minimum((-signed_quantity).clamp_min(0), self.shares.clamp_min(0))
        held_proceeds = sum(
            (value for _, value in self.bonus_proceeds),
            torch.zeros_like(self.trade_restricted),
        )
        release = (
            (self.trade_restricted - held_proceeds)
            * cover
            / short_shares.clamp_min(1e-30)
        )
        new_long = (signed_quantity - cover).clamp_min(0)
        new_short = (-signed_quantity - sell).clamp_min(0)
        self.custody.fill(
            self.shares.clamp_min(0),
            new_long,
            sell,
            loan_session.day,
            spot_settlement_session(loan_session.day, loan_session.date),
        )
        self.loans.fill(
            cover,
            new_short,
            loan_session,
            full_return=((cover == short_shares) & (cover > 0)).detach().numpy(),
        )
        notional = signed_quantity.abs() * price
        costs = (cost_rate * notional).sum()
        cash_delta = (
            release - cover * price + sell * price - new_long * price
        ).sum() - costs
        restricted_delta = -release + new_short * price
        spot_day = spot_settlement_session(loan_session.day, loan_session.date)
        for name, due in self.loans.bonus_return_floors.items():
            if due > spot_day and release[name].detach().item() > 0:
                retained = torch.zeros_like(release)
                retained[name] = release[name]
                cash_delta = cash_delta - retained.sum()
                restricted_delta = restricted_delta + retained
                self.bonus_proceeds.append((due, retained))
                release = release - retained
        if bool((release.detach() > 0).any()):
            self.proceeds_releases.append((spot_day, release))
        self.trade_cash = self.trade_cash + cash_delta
        self.trade_restricted = self.trade_restricted + restricted_delta
        self.settlements.append(
            (
                spot_settlement_session(loan_session.day, loan_session.date),
                cash_delta,
                restricted_delta,
            )
        )
        self.shares = self.shares + signed_quantity
        return costs

    def _deliver_shares(
        self,
        name,
        destination,
        incoming,
        allocation,
        converted_mark,
        exit_fraction,
        entry_notional,
        *,
        final,
        day,
        ratio,
        loan_allocation=1.0,
        receipt_day=None,
    ):
        shares, marks = self.shares.clone(), self.marks.clone()
        transferred_restricted = self.trade_restricted[name] * allocation
        adjusted = []
        for due, free, restricted in self.settlements:
            r = restricted.clone()
            amount = r[name].clone() * allocation
            r[destination] = r[destination] + amount
            r[name] = r[name] - amount
            adjusted.append((due, free, r))
        self.settlements = adjusted
        transferred_basis = self.cost_basis[name] * allocation
        existing = self.shares[destination]
        combined = incoming + existing
        self.loans.deliver(name, destination, ratio, loan_allocation, final=final)
        custody_dates = self.custody.deliver(
            name,
            destination,
            ratio,
            incoming,
            existing,
            day,
            final=final,
            receipt_day=receipt_day,
        )
        remaining_short = (-combined).clamp_min(0)
        excess = torch.zeros_like(self.shares)
        excess[destination] = (
            self.loans.active_quantity[destination] - remaining_short
        ).clamp_min(0)
        offset = torch.where(
            incoming * existing < 0,
            torch.minimum(incoming.abs(), existing.abs()),
            0.0,
        )
        incoming_fraction = (incoming.abs() - offset) / incoming.abs().clamp_min(1e-30)
        existing_fraction = (existing.abs() - offset) / existing.abs().clamp_min(1e-30)
        restricted = self.trade_restricted.clone()
        restricted[destination] = (
            transferred_restricted * incoming_fraction
            + self.trade_restricted[destination] * existing_fraction
        )
        release = (
            transferred_restricted
            + self.trade_restricted[destination]
            - restricted[destination]
        )
        for due, quantity in custody_dates:
            fraction = quantity[destination] / offset.clamp_min(1e-30)
            self.loans.request_return(excess * fraction, due, request_day=day)
            if due > day:
                deferred = release * fraction
                restricted_flow = torch.zeros_like(self.trade_restricted)
                restricted_flow[destination] = -deferred
                self.settlements.append((due, deferred, restricted_flow))
        self.trade_cash = self.trade_cash + release
        restricted[name] = restricted[name] - transferred_restricted
        self.trade_restricted = restricted
        basis = self.cost_basis.clone()
        basis[destination] = (
            transferred_basis * incoming_fraction
            + self.cost_basis[destination] * existing_fraction
        )
        basis[name] = basis[name] - transferred_basis
        self.cost_basis = basis
        shares[destination] = combined
        marks[destination] = torch.where(
            existing != 0, self.marks[destination], converted_mark
        )
        if final:
            shares[name], marks[name] = 0, 0
        exit_quantity = (
            incoming * exit_fraction[name] + existing * exit_fraction[destination]
        )
        exit_fraction = exit_fraction.clone()
        denominator = torch.where(combined != 0, combined, 1.0)
        exit_fraction[destination] = torch.where(
            combined != 0, (exit_quantity / denominator).clamp(0, 1), 0.0
        )
        if final:
            exit_fraction[name] = 0
        # An entry into a cancelled predecessor is not an instruction
        # to open its successor. Realize only the existing entitlement.
        entry_notional = entry_notional.clone()
        entry_notional[name] = 0
        origins = [
            index
            for index, fraction in (
                (name, incoming_fraction),
                (destination, existing_fraction),
            )
            if fraction.detach().item() > 0
        ]
        self.entry_day[destination] = min(
            (self.entry_day[index] for index in origins), default=-1
        )
        self.ineligible_sessions[destination] = max(
            (self.ineligible_sessions[index] for index in origins), default=0
        )
        if existing.detach().item() == 0 and incoming.detach().item() != 0:
            self.missing_sessions[destination] = self.missing_sessions[name]
        if final:
            self.entry_day[name] = -1
            self.ineligible_sessions[name] = 0
            self.missing_sessions[name] = 0
        self.shares, self.marks = shares, marks
        return exit_fraction, entry_notional

    def convert_loan_cash(self, event, day):
        name = event.security_index
        active = self.loans.active_quantity[name]
        if not event.unreturned_only:
            for due in np.unique(self.loans.return_day[self.loans.name == name]):
                if due > day:
                    restored = torch.zeros_like(self.shares)
                    restored[name] = self.loans.quantity[
                        (self.loans.name == name) & (self.loans.return_day == due)
                    ].sum()
                    self.custody.add(int(due), restored)
        quantity = self.loans.cash_settle(event, day)
        fraction = (
            (quantity / active.clamp_min(1e-30)).clamp(max=1)
            if event.unreturned_only
            else tensor(1.0)
        )
        release = self.trade_restricted[name] * fraction
        self.trade_cash = self.trade_cash + release
        restricted = self.trade_restricted.clone()
        restricted[name] = restricted[name] - release
        self.trade_restricted = restricted
        if not event.unreturned_only:
            # Superseded returns retain their covering assets and original spot
            # value dates; an active-only election never cancels those returns.
            settlements = []
            for due, free, pending_restricted in self.settlements:
                amount = pending_restricted[name]
                remaining = pending_restricted.clone()
                remaining[name] = 0
                settlements.append((due, free + amount, remaining))
            self.settlements = settlements
        if event.cash_day > day:
            retained = torch.zeros_like(self.trade_restricted)
            retained[name] = -release
            self.settlements.append((event.cash_day, release, retained))
        shares = self.shares.clone()
        shares[name] = shares[name] + quantity
        self.shares = shares
        if abs(float(shares[name].detach())) < 1e-12:
            self.cost_basis = self.cost_basis.clone()
            self.cost_basis[name] = 0
            self.entry_day[name] = -1
            self.missing_sessions[name] = self.ineligible_sessions[name] = 0

    def step(
        self,
        target: Tensor,
        *,
        day: int,
        close,
        cdi: float,
        session_date,
        annual_borrow,
        loan_reference,
        action_q=None,
        action_d=None,
        action_resolved=None,
        successor=None,
        payment_session=None,
        action_settlements=(),
        share_distributions=(),
        loan_cash_settlements=(),
        fill_fraction=None,
        entry_fill_allowed=None,
        terminal=False,
        loan_return_session=None,
    ):
        self.retired_sources.update(
            event.source_index
            for event in share_distributions
            if event.effective_session < day
        )
        self.prepare_day(day)
        n = len(self.shares)
        start_nav = self.nav
        loan_session = LoanSession(
            day,
            session_date,
            np.asarray(loan_reference),
            np.asarray(annual_borrow),
            spot_settlement_session(day, session_date)
            if loan_return_session is None
            else loan_return_session,
        )
        before = self.weights
        target = torch.zeros_like(before) if terminal else target
        loan_blocked = self.loans.short_blocked(day, self.config.loan_recalls)
        if self.distributions:
            loan_blocked[list(self.distributions)] = False
        if np.any(
            loan_blocked
            & (
                target.detach().numpy()
                < np.minimum(0, self.reserved_weights.detach().numpy()) - 2e-6
            )
        ):
            raise ValueError("portfolio retains a called or unavailable loan")
        reverse = before * target < 0
        reduction = torch.where(
            reverse, before.abs(), (before.abs() - target.abs()).clamp_min(0)
        )
        # Match the exact ledger's order-submission threshold in NAV units.
        # Solver dust must not create held/age state only in the training account.
        mandatory = torch.as_tensor(loan_blocked) | terminal
        reduction = torch.where((reduction > 1e-10) | mandatory, reduction, 0.0)
        exit_fraction = (reduction / before.abs().clamp_min(1e-30)).clamp(max=1)
        increase = torch.where(
            reverse, target.abs(), (target.abs() - before.abs()).clamp_min(0)
        )
        increase = torch.where(increase > 1e-10, increase, 0.0)
        entry_notional = target.sign() * increase * start_nav
        hedge_trade_notional = (target[-1] - before[-1]) * start_nav
        for event in loan_cash_settlements:
            if event.prohibit_new_borrow and event.effective_session <= day:
                entry_notional = entry_notional.clone()
                name = event.security_index
                entry_notional[name] = entry_notional[name].clone().clamp_min(0)
            if event.effective_session == day and not event.unreturned_only:
                self.convert_loan_cash(event, day)

        # Realizations begin here. Terms alter prior inventory and marks once;
        # a cash entitlement remains a claim until its explicit payment day.
        q = np.ones(n) if action_q is None else np.asarray(action_q, dtype=float)
        d = np.zeros(n) if action_d is None else np.asarray(action_d, dtype=float)
        resolved = (
            np.ones(n, bool)
            if action_resolved is None
            else np.asarray(action_resolved, bool)
        )
        q, d = np.where(resolved, q, 1.0), np.where(resolved, d, 0.0)
        changed = (q != 1) | (d != 0)
        mapping = (
            np.arange(n) if successor is None else np.asarray(successor, dtype=np.int64)
        )
        changed |= resolved & (mapping != np.arange(n))
        settlements_today = {
            event.security_index: event
            for event in action_settlements
            if event.effective_session == day
        }
        withholding_accrual = tensor(0.0)
        lender_compensation = tensor(0.0)
        for name in np.flatnonzero(changed):
            name = int(name)
            if name in self.distributions or any(
                name == leg.successor_index
                for legs in self.distributions.values()
                for leg in legs
            ):
                raise ValueError(
                    "an action on an outstanding basket needs explicit claim terms"
                )
            amount = torch.zeros(n, dtype=torch.float64)
            event = settlements_today.get(name)
            bonus_delivery = None if event is None else event.bonus_delivery_session
            if event is not None:
                event.validate_action(q[name], d[name], mapping[name])
            if self.bonus_shares[name].detach().item() != 0:
                raise ValueError(
                    "an action on an undelivered bonus needs explicit terms"
                )
            gross = self.shares[name] * d[name]
            withheld = gross.clamp_min(0) * (
                0.0 if event is None else event.withholding_rate
            )
            compensation = -gross.clamp_max(0) * (
                1.0 if event is None else event.short_cash_fraction
            )
            amount[name] = gross.clamp_min(0) - withheld - compensation
            withholding_accrual = withholding_accrual + withheld
            lender_compensation = lender_compensation + compensation
            if bonus_delivery is not None and bonus_delivery > day:
                bonus = self.bonus_shares.clone()
                bonus[name] = self.shares[name] * (q[name] - 1)
                self.bonus_shares = bonus
                self.bonus_delivery[name] = bonus_delivery
            if bonus_delivery is not None:
                shifted = []
                for due, release in self.proceeds_releases:
                    if due < bonus_delivery and release[name].detach().item() > 0:
                        retained = torch.zeros_like(release)
                        retained[name] = release[name]
                        self.trade_cash = self.trade_cash - retained.sum()
                        self.trade_restricted = self.trade_restricted + retained
                        if due > day:
                            self.settlements.append((due, -retained.sum(), retained))
                        self.bonus_proceeds.append((bonus_delivery, retained))
                        release = release - retained
                    shifted.append((due, release))
                self.proceeds_releases = shifted
            self.claims = self.claims + amount
            if d[name] != 0:
                pay = int(payment_session[name]) if payment_session is not None else -1
                self.payments.append((pay, amount))
            shares, marks = self.shares.clone(), self.marks.clone()
            shares[name] = self.shares[name] * q[name]
            if q[name] > 0 and mapping[name] == name:
                self.loans.split(name, q[name], bonus_delivery=bonus_delivery)
                self.custody.split(name, q[name], bonus_delivery=bonus_delivery)
            elif (
                q[name] == 0
                and payment_session is not None
                and int(payment_session[name]) >= day
            ):
                returns = torch.zeros_like(self.shares)
                returns[name] = self.loans.active_quantity[name]
                self.loans.request_return(
                    returns, int(payment_session[name]), request_day=day
                )
            marks[name] = (self.marks[name] - d[name]) / q[name] if q[name] > 0 else 0
            if q[name] == 0:
                self.custody.split(name, 0.0)
                release = self.trade_restricted[name]
                release_vector = torch.zeros_like(self.trade_restricted)
                release_vector[name] = -release
                pay = int(payment_session[name]) if payment_session is not None else -1
                if pay != day:
                    self.settlements.append(
                        (pay if pay >= day else None, release, release_vector)
                    )
                self.trade_cash = self.trade_cash + release
                restricted = self.trade_restricted.clone()
                restricted[name] = 0
                self.trade_restricted = restricted
            destination = int(mapping[name])
            if destination != name:
                exit_fraction, entry_notional = self._deliver_shares(
                    name,
                    destination,
                    shares[name].clone(),
                    1.0,
                    marks[name].clone(),
                    exit_fraction,
                    entry_notional,
                    final=True,
                    day=day,
                    ratio=q[name],
                )
                continue
            self.shares, self.marks = shares, marks
        for event in share_distributions:
            if event.effective_session != day:
                continue
            name = event.source_index
            if changed[name] or name in self.distributions:
                raise ValueError(
                    "a distribution cannot duplicate an existing source action"
                )
            self.retired_sources.add(name)
            amount = torch.zeros_like(self.claims)
            amount[name] = self.shares[name] * event.cash_per_prior_share
            self.claims = self.claims + amount
            if event.cash_per_prior_share:
                self.payments.append((event.payment_session, amount))
            if (
                self.shares[name].detach().item() != 0
                or (self.loans.name == name).any()
            ):
                legs = recognize_distribution(event, self.marks.detach().numpy())
                self.distributions[name] = legs
                prices = basket_prices(legs, self.marks.detach().numpy())
                marks = self.marks.clone()
                marks[name] = sum(
                    leg.shares_per_prior_share * price
                    for leg, price in zip(legs, prices)
                )
                self.marks = marks
        # Same-session delivery of a newly recognized claim is a realization;
        # later deliveries were handled before this day's decision above.
        self.pending_exit = exit_fraction
        self.prepare_day(day, realize_auctions=True)
        exit_fraction = self.pending_exit
        if self.retired_sources:
            entry_notional = entry_notional.clone()
            entry_notional[list(self.retired_sources)] = 0
        self._pay_claims(day)

        config = self.config
        interest = (
            self.funding_cash * cdi
            + self.funding_cash.clamp_max(0)
            * config.annual_debit_spread
            / config.annual_sessions
            + self.funding_restricted * cdi * config.short_proceeds_remuneration
        )
        self.trade_cash = self.trade_cash + interest

        close = np.asarray(close, dtype=float)
        printed = np.isfinite(close) & (close > 0)
        if self.retired_sources:
            printed[list(self.retired_sources)] = False
        self.unquoted_deliveries.difference_update(np.flatnonzero(printed).tolist())
        prices = tensor(np.where(printed, close, 0.0))
        fractions = np.ones(n) if fill_fraction is None else np.asarray(fill_fraction)
        fractions = tensor(np.where(printed, fractions, 0.0))
        fractions[-1] = 0  # The hedge has one separate notional order below.
        cost_rate = torch.full(
            (n,), config.cost_bps_per_side / 1e4, dtype=torch.float64
        )
        cost_rate[-1] = config.hedge_cost_bps_per_side / 1e4
        shares_before_fill = self.shares
        available = (self.shares.abs() - self.bonus_shares.abs()).clamp_min(0)
        used = torch.minimum(
            exit_fraction * fractions, available / self.shares.abs().clamp_min(1e-30)
        )
        costs = self._fill(-self.shares * used, prices, cost_rate, loan_session)
        self.cost_basis = self.cost_basis * (1 - used)
        remaining_exit = (exit_fraction - used) / (1 - used).clamp_min(1e-30)
        remaining_exit = torch.where(self.shares.abs() > 1e-12, remaining_exit, 0.0)
        opening_allowed = remaining_exit.detach().numpy() <= 1e-12
        if entry_fill_allowed is not None:
            opening_allowed &= np.asarray(entry_fill_allowed, dtype=bool)
        opening_allowed[-1] = False
        quantity = (
            entry_notional
            / prices.clamp_min(1e-30)
            * fractions
            * torch.as_tensor(opening_allowed)
        )
        costs = costs + self._fill(quantity, prices, cost_rate, loan_session)
        self.cost_basis = self.cost_basis + quantity.abs() * prices

        # Hedge reductions use fixed decision notional; final liquidation uses
        # all actual hedge shares. Neither reads the close to choose intention.
        hedge_quantity = torch.zeros(n, dtype=torch.float64)
        if printed[-1]:
            hedge_quantity[-1] = (
                -self.shares[-1] if terminal else hedge_trade_notional / prices[-1]
            )
        costs = costs + self._fill(hedge_quantity, prices, cost_rate, loan_session)
        for event in loan_cash_settlements:
            if event.effective_session == day and event.unreturned_only:
                self.convert_loan_cash(event, day)
        cash_loan_payment = self.loans.settle_cash_claims(day).sum()
        self.trade_cash = self.trade_cash - cash_loan_payment
        cash_loan_payment = cash_loan_payment + self.prepared_loan_cash_payment
        self.loans.renew(
            loan_session,
            self.config.loan_term_sessions,
            approved=self.config.approve_loan_renewals,
        )
        overdue_principal = self.loans.overdue_principal(day).sum()
        loan_rent, loan_fee = self.loans.accrue(day, session_date)
        borrow = loan_rent.sum() + loan_fee.sum()
        rent_paid, fees_paid = self.loans.pay(day)
        self.trade_cash = self.trade_cash - rent_paid.sum() - fees_paid.sum()
        self.marks = torch.where(torch.as_tensor(printed), prices, self.marks)
        valued = printed.copy()
        for name, legs in self.distributions.items():
            marks = self.marks.clone()
            claim_prices = basket_prices(legs, self.marks.detach().numpy())
            marks[name] = sum(
                leg.shares_per_prior_share * price
                for leg, price in zip(legs, claim_prices)
            )
            self.marks = marks
            valued[name] = all(printed[leg.successor_index] for leg in legs)
        held = self.shares.detach().numpy() != 0
        self.missing_sessions[valued & held] = 0
        self.missing_sessions[~valued & held] += 1
        # No print means inventory remains, including at an evaluation boundary.
        unpriced_notional = (
            self.shares.abs() * self.marks * torch.as_tensor(~valued)
        ).sum()
        held_after = self.shares.detach().numpy() != 0
        held_before = shares_before_fill.detach().numpy() != 0
        changed_side = (
            shares_before_fill.detach().numpy() * self.shares.detach().numpy() < 0
        )
        self.cost_basis = torch.where(
            torch.as_tensor(changed_side), self.shares.abs() * prices, self.cost_basis
        )
        self.entry_day[(~held_before | changed_side) & held_after] = day
        self.entry_day[~held_after] = -1
        self.cost_basis = torch.where(torch.as_tensor(held_after), self.cost_basis, 0.0)
        self.pending_exit = torch.where(
            torch.as_tensor(held_after), remaining_exit, 0.0
        )
        self.ineligible_sessions[~held_after] = 0
        self.missing_sessions[~held_after] = 0
        nav = self.nav
        if not torch.isfinite(nav) or nav.detach().item() <= 0:
            raise FloatingPointError("non-finite or insolvent policy account")
        return {
            "start_nav": start_nav,
            "nav": nav,
            "net_excess": nav / start_nav - 1 - cdi,
            "interest": interest,
            "borrow": borrow,
            "withholding_accrual": withholding_accrual,
            "lender_compensation": lender_compensation,
            "borrow_paid": rent_paid.sum() + fees_paid.sum(),
            "borrow_liability": self.loans.liability,
            "loan_redemption_liability": self.loans.cash_liability,
            "loan_overdue_principal": overdue_principal,
            "loan_cash_settlement_payment": cash_loan_payment,
            "unsettled_cash": self.unsettled_cash,
            "free_cash_income": self.funding_cash.clamp_min(0) * cdi,
            "debit_financing": -self.funding_cash.clamp_max(0)
            * (cdi + config.annual_debit_spread / config.annual_sessions),
            "free_cash_interest": self.funding_cash * cdi
            + self.funding_cash.clamp_max(0)
            * config.annual_debit_spread
            / config.annual_sessions,
            "short_proceeds_interest": self.funding_restricted
            * cdi
            * config.short_proceeds_remuneration,
            "cost": costs,
            "unpriced_inventory_notional": unpriced_notional,
            "undelivered_share_notional": sum(
                (self.shares[name] * self.marks[name]).abs()
                for name in self.distributions
            )
            + (self.bonus_shares * self.marks).abs().sum(),
        }
