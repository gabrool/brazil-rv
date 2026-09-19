"""Differentiable share/cash transitions for the frozen target-policy contract.

The independent NumPy stateful ledger verifies this training implementation.
Market/action/payment arrays are realizations, consumed only after target
notional and exit fractions have been computed from the previous state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .stateful_ledger import LedgerConfig


def tensor(values) -> Tensor:
    return torch.as_tensor(values, dtype=torch.float64)


@dataclass
class PortfolioAccount:
    shares: Tensor
    marks: Tensor
    restricted: Tensor
    claims: Tensor
    cash: Tensor
    cost_basis: Tensor
    pending_exit: Tensor
    entry_day: np.ndarray
    missing_sessions: np.ndarray
    ineligible_sessions: np.ndarray
    payments: list[tuple[int, Tensor]]
    config: LedgerConfig

    @classmethod
    def empty(cls, reference, *, config=LedgerConfig()):
        """Reference axis includes the optional hedge as the last coordinate."""
        n = len(reference)
        zero = torch.zeros(n, dtype=torch.float64)
        return cls(
            shares=zero.clone(),
            marks=tensor(np.nan_to_num(reference, nan=0.0)),
            restricted=zero.clone(),
            claims=zero.clone(),
            cash=tensor(config.initial_capital_brl),
            cost_basis=zero.clone(),
            pending_exit=zero.clone(),
            entry_day=np.full(n, -1, dtype=np.int64),
            missing_sessions=np.zeros(n, dtype=np.int64),
            ineligible_sessions=np.zeros(n, dtype=np.int64),
            payments=[],
            config=config,
        )

    @property
    def nav(self):
        return (
            self.cash
            + self.restricted.sum()
            + self.claims.sum()
            + (self.shares * self.marks).sum()
        )

    @property
    def weights(self):
        return self.shares * self.marks / self.nav

    def detach(self):
        """Truncate gradients without resetting positions, financing or claims."""
        for name in (
            "shares",
            "marks",
            "restricted",
            "claims",
            "cash",
            "cost_basis",
            "pending_exit",
        ):
            setattr(self, name, getattr(self, name).detach())
        self.payments = [(day, amount.detach()) for day, amount in self.payments]

    def eligibility(self, active_score, prior_unresolved):
        held = self.shares.detach().numpy() != 0
        eligible = np.asarray(active_score, dtype=bool)
        self.ineligible_sessions[~held | eligible] = 0
        self.ineligible_sessions[held & ~eligible] += 1
        allowed = eligible & ~prior_unresolved & (self.marks.detach().numpy() > 0)
        required = (self.ineligible_sessions > self.config.ineligible_hold_sessions) | (
            self.missing_sessions >= self.config.settlement_grace_sessions
        )
        return allowed, required

    def _fill(self, signed_quantity: Tensor, price: Tensor, cost_rate: Tensor):
        """Self-financing fill with segregated short proceeds, including reversals."""
        short_shares = (-self.shares).clamp_min(0)
        cover = torch.minimum(signed_quantity.clamp_min(0), short_shares)
        sell = torch.minimum((-signed_quantity).clamp_min(0), self.shares.clamp_min(0))
        release = self.restricted * cover / short_shares.clamp_min(1e-30)
        new_long = (signed_quantity - cover).clamp_min(0)
        new_short = (-signed_quantity - sell).clamp_min(0)
        notional = signed_quantity.abs() * price
        costs = (cost_rate * notional).sum()
        self.cash = (
            self.cash
            + (release - cover * price + sell * price - new_long * price).sum()
            - costs
        )
        self.restricted = self.restricted - release + new_short * price
        self.shares = self.shares + signed_quantity
        return costs

    def step(
        self,
        target: Tensor,
        *,
        day: int,
        close,
        cdi: float,
        daily_borrow,
        action_q=None,
        action_d=None,
        action_resolved=None,
        successor=None,
        payment_session=None,
        fill_fraction=None,
        entry_fill_allowed=None,
        terminal=False,
    ):
        n = len(self.shares)
        start_nav = self.nav
        before = self.weights
        target = torch.zeros_like(before) if terminal else target
        reverse = before * target < 0
        reduction = torch.where(
            reverse, before.abs(), (before.abs() - target.abs()).clamp_min(0)
        )
        # Match the exact ledger's order-submission threshold in NAV units.
        # Solver dust must not create held/age state only in the training account.
        reduction = torch.where(reduction > 1e-10, reduction, 0.0)
        exit_fraction = (reduction / before.abs().clamp_min(1e-30)).clamp(max=1)
        increase = torch.where(
            reverse, target.abs(), (target.abs() - before.abs()).clamp_min(0)
        )
        increase = torch.where(increase > 1e-10, increase, 0.0)
        entry_notional = target.sign() * increase * start_nav
        hedge_trade_notional = (target[-1] - before[-1]) * start_nav

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
        for name in np.flatnonzero(changed):
            name = int(name)
            amount = torch.zeros(n, dtype=torch.float64)
            amount[name] = self.shares[name] * d[name]
            self.claims = self.claims + amount
            if d[name] != 0:
                pay = int(payment_session[name]) if payment_session is not None else -1
                self.payments.append((pay, amount))
            shares, marks = self.shares.clone(), self.marks.clone()
            shares[name] = self.shares[name] * q[name]
            marks[name] = (self.marks[name] - d[name]) / q[name] if q[name] > 0 else 0
            if q[name] == 0:
                self.cash = self.cash + self.restricted[name]
                restricted = self.restricted.clone()
                restricted[name] = 0
                self.restricted = restricted
            destination = int(mapping[name])
            if destination != name:
                if self.shares[destination].detach().item() != 0:
                    raise ValueError("conversion successor already has inventory")
                shares[destination], marks[destination] = (
                    shares[name].clone(),
                    marks[name].clone(),
                )
                shares[name], marks[name] = 0, 0
                for field in ("restricted", "cost_basis"):
                    values = getattr(self, field).clone()
                    values[destination] = values[destination] + values[name]
                    values[name] = 0
                    setattr(self, field, values)
                exit_fraction = exit_fraction.clone()
                exit_fraction[destination] = exit_fraction[name]
                exit_fraction[name] = 0
                for field in ("entry_day", "missing_sessions", "ineligible_sessions"):
                    values = getattr(self, field)
                    values[destination] = values[name]
                    values[name] = -1 if field == "entry_day" else 0
            self.shares, self.marks = shares, marks
        unpaid = []
        for pay, amount in self.payments:
            if pay == day:
                self.cash = self.cash + amount.sum()
                self.claims = self.claims - amount
            else:
                unpaid.append((pay, amount))
        self.payments = unpaid

        config = self.config
        interest = (
            self.cash * cdi
            + self.cash.clamp_max(0)
            * config.annual_debit_spread
            / config.annual_sessions
            + self.restricted.sum() * cdi * config.short_proceeds_remuneration
        )
        borrow = ((-self.shares).clamp_min(0) * self.marks * tensor(daily_borrow)).sum()
        self.cash = self.cash + interest - borrow

        close = np.asarray(close, dtype=float)
        printed = np.isfinite(close) & (close > 0)
        prices = tensor(np.where(printed, close, 0.0))
        fractions = np.ones(n) if fill_fraction is None else np.asarray(fill_fraction)
        fractions = tensor(np.where(printed, fractions, 0.0))
        fractions[-1] = 0  # The hedge has one separate notional order below.
        cost_rate = torch.full(
            (n,), config.cost_bps_per_side / 1e4, dtype=torch.float64
        )
        cost_rate[-1] = config.hedge_cost_bps_per_side / 1e4
        shares_before_fill = self.shares
        used = exit_fraction * fractions
        costs = self._fill(-self.shares * used, prices, cost_rate)
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
        costs = costs + self._fill(quantity, prices, cost_rate)
        self.cost_basis = self.cost_basis + quantity.abs() * prices

        # Hedge reductions use fixed decision notional; final liquidation uses
        # all actual hedge shares. Neither reads the close to choose intention.
        hedge_quantity = torch.zeros(n, dtype=torch.float64)
        if printed[-1]:
            hedge_quantity[-1] = (
                -self.shares[-1] if terminal else hedge_trade_notional / prices[-1]
            )
        costs = costs + self._fill(hedge_quantity, prices, cost_rate)
        self.marks = torch.where(torch.as_tensor(printed), prices, self.marks)
        held = self.shares.detach().numpy() != 0
        self.missing_sessions[printed & held] = 0
        self.missing_sessions[~printed & held] += 1
        # No print means inventory remains, including at an evaluation boundary.
        unpriced_notional = (
            self.shares.abs() * self.marks * torch.as_tensor(~printed)
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
            "cost": costs,
            "unpriced_inventory_notional": unpriced_notional,
        }
