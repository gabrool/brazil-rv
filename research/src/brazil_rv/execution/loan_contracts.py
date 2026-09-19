"""Fixed-principal loan cohorts, including outstanding returns and unpaid charges.

The share/cash accounts remain separate implementations. They share this contract
subledger, whose money transitions have closed-form tests against the B3 formulas.
All arithmetic is vectorized on CPU; quantities retain gradients in training and
have no autograd graph in NumPy replay. Rates and dates are historical constants.
"""

from dataclasses import dataclass, field, fields, replace
import math

import numpy as np
import torch

from .loan_fees import loan_fee_rates


def _tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


def _fee_period(session_date):
    return int(np.datetime64(session_date) >= np.datetime64("2020-10-26")) + int(
        np.datetime64(session_date) >= np.datetime64("2022-11-14")
    )


@dataclass(frozen=True)
class LoanSession:
    day: int
    date: object
    reference: np.ndarray
    annual_rate: np.ndarray
    return_day: int
    imputed: np.ndarray | None = None
    placeholder: np.ndarray | None = None


@dataclass(frozen=True)
class LoanCharge:
    session: int
    opening_session: int
    security_index: int
    rent: float
    fee: float


@dataclass(frozen=True)
class LoanCashValue:
    available_session: int
    cash_per_share: float


@dataclass(frozen=True)
class LoanCashSettlement:
    """Sourced compulsory cash closeout of loans, not shareholder redemption."""

    security_index: int
    effective_session: int
    available_session: int
    cash_per_share: float
    source: str
    rent_payment_session: int | None = None
    payment_session: int | None = None
    valuations: tuple[LoanCashValue, ...] = ()
    unreturned_only: bool = False
    prohibit_new_borrow: bool = True

    def __post_init__(self):
        if self.available_session > self.effective_session:
            raise ValueError("loan cash terms cannot be backdated")
        if (
            not self.source
            or not math.isfinite(self.cash_per_share)
            or self.cash_per_share <= 0
        ):
            raise ValueError("loan cash settlement requires a sourced positive price")
        for date in (self.rent_payment_session, self.payment_session):
            if date is not None and date < self.effective_session:
                raise ValueError("loan payment cannot precede quantity extinction")
        previous = self.effective_session
        for value in self.valuations:
            if (
                value.available_session <= previous
                or value.available_session >= self.cash_day
                or not math.isfinite(value.cash_per_share)
                or value.cash_per_share <= 0
            ):
                raise ValueError("loan cash valuations require ordered causal values")
            previous = value.available_session

    @property
    def cash_day(self):
        return (
            self.effective_session
            if self.payment_session is None
            else self.payment_session
        )

    @property
    def rent_day(self):
        return (
            self.effective_session
            if self.rent_payment_session is None
            else self.rent_payment_session
        )


@dataclass(frozen=True)
class LoanCashPayment:
    session: int
    security_index: int
    quantity: float
    cash_paid: float


def slice_loan_settlements(events, start, stop):
    return tuple(
        replace(
            event,
            effective_session=event.effective_session - start,
            available_session=event.available_session - start,
            rent_payment_session=None
            if event.rent_payment_session is None
            else event.rent_payment_session - start,
            payment_session=None
            if event.payment_session is None
            else event.payment_session - start,
            valuations=tuple(
                replace(value, available_session=value.available_session - start)
                for value in event.valuations
            ),
        )
        for event in events
        if event.effective_session < stop
    )


def spot_settlement_session(day, session_date):
    """Dated spot value date on the supplied complete trading-session axis.

    Spot moved from T+3 to T+2 on 2019-05-27. Same-settlement loan return
    is an execution assumption; broker/cutoff delays require an explicit override.
    """
    return day + (3 if np.datetime64(session_date) < np.datetime64("2019-05-27") else 2)


@dataclass
class LoanContracts:
    names: int
    modality: str = "normal"
    fee_multiplier: float = 1.0
    annual_sessions: int = 252
    name: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    opened: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    return_day: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    accrual_end: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    root: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    annual_rate: np.ndarray = field(default_factory=lambda: np.empty(0))
    fee_rate: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    fee_growth: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    fee_period: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    quantity: torch.Tensor = field(default_factory=lambda: _tensor([]))
    principal: torch.Tensor = field(default_factory=lambda: _tensor([]))
    rent_due: torch.Tensor = field(default_factory=lambda: _tensor([]))
    fees_due: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 2), dtype=torch.float64)
    )
    root_name: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    root_opened: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    imputed: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    placeholder: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    minimum: np.ndarray = field(default_factory=lambda: np.empty(0))
    started: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    root_fees: torch.Tensor = field(default_factory=lambda: _tensor([]))
    cash_claims: list[tuple[LoanCashSettlement, torch.Tensor, float]] = field(
        default_factory=list
    )

    def _by_name(self, values, names=None):
        return torch.zeros(self.names, dtype=torch.float64).index_add(
            0, torch.as_tensor(self.name if names is None else names), values
        )

    @property
    def minimum_provision(self):
        return (_tensor(self.minimum * self.started) - self.root_fees).clamp_min(0)

    @property
    def liability(self):
        return (
            self.rent_due.sum()
            + self.fees_due.sum()
            + self.minimum_provision.sum()
            + self.cash_liability
        )

    @property
    def cash_liability(self):
        return sum(
            (quantity * value for _, quantity, value in self.cash_claims), _tensor(0.0)
        )

    @property
    def active_quantity(self):
        return self._by_name(self.quantity * _tensor(self.return_day < 0))

    @property
    def outstanding_principal(self):
        return self._by_name(self.principal * _tensor(self.accrual_end < 0))

    def open(
        self,
        quantity,
        reference,
        annual_rate,
        day,
        session_date,
        imputed=None,
        placeholder=None,
    ):
        """Register only actually borrowed shares; never read a later quote/rate."""
        quantity = _tensor(quantity)
        ids = np.flatnonzero(quantity.detach().numpy() > 0)
        if not len(ids):
            return
        reference, rate = np.asarray(reference)[ids], np.asarray(annual_rate)[ids]
        if not np.all(np.isfinite(reference) & (reference > 0)):
            raise ValueError(
                "opening a loan requires its causal published-average reference"
            )
        if not np.all(np.isfinite(rate) & (rate >= 0)):
            raise ValueError("opening a loan requires a finite causal annual rate")
        roots = np.arange(len(self.root_name), len(self.root_name) + len(ids))
        self.name = np.r_[self.name, ids]
        self.opened = np.r_[self.opened, np.full(len(ids), day)]
        self.return_day = np.r_[self.return_day, np.full(len(ids), -1)]
        self.accrual_end = np.r_[self.accrual_end, np.full(len(ids), -1)]
        self.root = np.r_[self.root, roots]
        self.annual_rate = np.r_[self.annual_rate, rate]
        fees = loan_fee_rates(rate, session_date, modality=self.modality)
        self.fee_rate = np.concatenate((self.fee_rate, fees))
        self.fee_growth = np.concatenate((self.fee_growth, np.ones_like(fees)))
        self.fee_period = np.r_[
            self.fee_period, np.full(len(ids), _fee_period(session_date))
        ]
        self.quantity = torch.cat((self.quantity, quantity[ids]))
        self.principal = torch.cat((self.principal, quantity[ids] * _tensor(reference)))
        self.rent_due = torch.cat(
            (self.rent_due, torch.zeros(len(ids), dtype=torch.float64))
        )
        self.fees_due = torch.cat(
            (self.fees_due, torch.zeros((len(ids), 2), dtype=torch.float64))
        )
        old_voluntary = (
            np.datetime64(session_date) < np.datetime64("2020-10-26")
            and self.modality != "compulsory"
        )
        self.root_name = np.r_[self.root_name, ids]
        self.root_opened = np.r_[self.root_opened, np.full(len(ids), day)]
        self.imputed = np.r_[
            self.imputed,
            np.zeros(len(ids), dtype=bool)
            if imputed is None
            else np.asarray(imputed)[ids],
        ]
        self.placeholder = np.r_[
            self.placeholder,
            np.zeros(len(ids), dtype=bool)
            if placeholder is None
            else np.asarray(placeholder)[ids],
        ]
        self.minimum = np.r_[
            self.minimum,
            np.full(len(ids), 10.0 * self.fee_multiplier if old_voluntary else 0.0),
        ]
        self.started = np.r_[self.started, np.zeros(len(ids), dtype=bool)]
        self.root_fees = torch.cat(
            (self.root_fees, torch.zeros(len(ids), dtype=torch.float64))
        )

    def accrue(self, day, session_date, *, charges=None):
        """Registration exclusive, return inclusive; recognize expense, not payment."""
        if not len(self.name):
            zero = torch.zeros(self.names, dtype=torch.float64)
            return zero, zero.clone()
        age = day - self.opened
        if np.any(age <= 0):
            raise ValueError("loans accrue once per subsequent business session")
        rent_log = np.log1p(self.annual_rate) / self.annual_sessions
        rent = self.principal * _tensor(
            np.exp(rent_log * (age - 1)) * np.expm1(rent_log) * (self.accrual_end < 0)
        )
        # Tariff periods are charged separately; a new schedule must not
        # retrospectively reprice an earlier period's accrued fee.
        period = _fee_period(session_date)
        changed = self.fee_period != period
        if np.any(changed):
            self.fee_growth[changed] = 1.0
            self.fee_period[changed] = period
            self.fee_rate[changed] = loan_fee_rates(
                self.annual_rate[changed], session_date, modality=self.modality
            )
        daily_fee = np.expm1(np.log1p(self.fee_rate) / self.annual_sessions)
        fees = self.principal[:, None] * _tensor(
            self.fee_growth
            * daily_fee
            * self.fee_multiplier
            * (self.accrual_end < 0)[:, None]
        )
        self.fee_growth *= 1 + daily_fee
        before = self.minimum_provision
        self.started[:] = True
        self.root_fees = self.root_fees.index_add(
            0, torch.as_tensor(self.root), fees.sum(-1)
        )
        minimum_change = self.minimum_provision - before
        self.rent_due = self.rent_due + rent
        self.fees_due = self.fees_due + fees
        if charges is not None:
            root_index = torch.as_tensor(self.root)
            root_rent = torch.zeros_like(self.root_fees).index_add(0, root_index, rent)
            root_fee = (
                torch.zeros_like(self.root_fees).index_add(0, root_index, fees.sum(-1))
                + minimum_change
            )
            # Paid contract roots remain for attribution, but emitting zero rows
            # for every old root each day makes long replays needlessly quadratic.
            rent_values = root_rent.detach().numpy()
            fee_values = root_fee.detach().numpy()
            for index in np.flatnonzero((rent_values != 0) | (fee_values != 0)):
                charges.append(
                    LoanCharge(
                        day,
                        int(self.root_opened[index]),
                        int(self.root_name[index]),
                        float(rent_values[index]),
                        float(fee_values[index]),
                    )
                )
        return self._by_name(rent), self._by_name(fees.sum(-1)) + self._by_name(
            minimum_change, self.root_name
        )

    def request_return(self, quantity, settlement_day):
        """Allocate partial returns pro rata across unreturned contracts of a name."""
        quantity = _tensor(quantity)
        if not np.any(quantity.detach().numpy() > 0):
            return
        total = self.active_quantity
        if np.any(
            quantity.detach().numpy()
            > total.detach().numpy() + 1e-10 * np.maximum(1, total.detach().numpy())
        ):
            raise ValueError("loan return exceeds unreturned quantity")
        fraction = (quantity / total.clamp_min(1e-30)).clamp(0, 1)
        part = fraction[self.name] * _tensor(self.return_day < 0)
        ids = np.flatnonzero(part.detach().numpy() > 0)
        if not len(ids):
            return
        for key in ("quantity", "principal", "rent_due", "fees_due"):
            value = getattr(self, key)
            share = part[:, None] if value.ndim == 2 else part
            setattr(self, key, torch.cat((value * (1 - share), (value * share)[ids])))
        for key in (
            "name",
            "opened",
            "accrual_end",
            "root",
            "annual_rate",
            "fee_rate",
            "fee_growth",
            "fee_period",
            "imputed",
            "placeholder",
        ):
            value = getattr(self, key)
            setattr(self, key, np.concatenate((value, value[ids])))
        self.return_day = np.r_[self.return_day, np.full(len(ids), settlement_day)]
        self._keep(
            (self.quantity.detach().numpy() > 0)
            | (self.principal.detach().numpy() != 0)
        )

    def fill(self, cover, new_short, session):
        self.request_return(cover, session.return_day)
        self.open(
            new_short,
            session.reference,
            session.annual_rate,
            session.day,
            session.date,
            session.imputed,
            session.placeholder,
        )

    def pay(self, day):
        """Pay matured returns, retaining unreturned and future-return liabilities.

        The historical minimum is provisioned once per original contract and any
        residual is paid on its final return, never once per partial fill.
        """
        due = self.return_day == day
        if not np.any(due):
            zero = torch.zeros(self.names, dtype=torch.float64)
            return zero, zero.clone()
        rent = self._by_name(self.rent_due * _tensor(due))
        fees = self._by_name(self.fees_due.sum(-1) * _tensor(due))
        live_roots = np.unique(self.root[~due])
        completed = np.ones(len(self.root_name), dtype=bool)
        completed[live_roots] = False
        fees = fees + self._by_name(
            self.minimum_provision * _tensor(completed), self.root_name
        )
        self._keep(~due)
        # Keep tensor work proportional to open contracts, not the whole run.
        self.root_name = self.root_name[live_roots]
        self.root_opened = self.root_opened[live_roots]
        self.minimum = self.minimum[live_roots]
        self.started = self.started[live_roots]
        self.root_fees = self.root_fees[live_roots]
        self.root = np.searchsorted(live_roots, self.root)
        return rent, fees

    def cash_settle(self, event, day):
        """Extinguish all outstanding shares, including later requested returns.

        Rent/fees accrue through today before this call and pay via the ordinary
        liability path. Principal cash is a separate issuer-specific payment.
        A covering asset is retained if its planned physical return is superseded.
        """
        ids = self.name == event.security_index
        if event.unreturned_only:
            # Lender elections apply to whole eligible contracts. A pending
            # partial return excludes that root, not just its returned fragment.
            pending_roots = self.root[self.return_day >= 0]
            ids &= ~np.isin(self.root, pending_roots) & (self.opened < day)
        quantity = self.quantity[ids].sum()
        self.return_day[ids] = event.rent_day
        self.accrual_end[ids] = day
        if quantity.detach().item() > 0:
            self.cash_claims.append((event, quantity, event.cash_per_share))
        return quantity

    def settle_cash_claims(self, day, *, payments=None, pay_only=False):
        """Realize only today's published values and today's actual cash arrival.

        A future payment date is a realization, never used to discount the claim
        or choose a prior trade. Rent has its own stopped accrual/payment clock.
        """
        paid = torch.zeros(self.names, dtype=torch.float64)
        remaining = []
        for event, quantity, value in self.cash_claims:
            if pay_only and event.cash_day != day:
                remaining.append((event, quantity, value))
                continue
            for update in event.valuations:
                if update.available_session == day:
                    value = update.cash_per_share
            if event.cash_day == day:
                amount = quantity * value
                paid = paid.index_add(
                    0, torch.tensor([event.security_index]), amount.reshape(1)
                )
                if payments is not None:
                    payments.append(
                        LoanCashPayment(
                            day,
                            event.security_index,
                            float(quantity.detach()),
                            float(amount.detach()),
                        )
                    )
            else:
                remaining.append((event, quantity, value))
        self.cash_claims = remaining
        return paid

    def _keep(self, keep):
        for key in (
            "name",
            "opened",
            "return_day",
            "accrual_end",
            "root",
            "annual_rate",
            "fee_rate",
            "fee_growth",
            "fee_period",
            "imputed",
            "placeholder",
            "quantity",
            "principal",
            "rent_due",
            "fees_due",
        ):
            setattr(self, key, getattr(self, key)[keep])

    def split(self, name, shares_per_prior_share):
        """Same-security split changes deliverable shares, not loan principal."""
        self.quantity = self.quantity * _tensor(
            np.where(self.name == name, shares_per_prior_share, 1.0)
        )

    def provision_fractions(self, name, ratio, day, auction):
        """Truncate each remaining original contract, preserving its principal.

        Dommo's last spot settlement precedes conversion. Pending returns or
        already split roots require their own custody/rounding terms, not net
        position rounding. Returned fractional units are a separate signed
        auction obligation in the share/cash book, not tradable loan quantity.

        A sub-one-share contract has no deliverable quantity. Primary research
        stops its rent at the preceding conversion close and pays accrued rent
        today; the explicit alternative retains rent until fraction payment.
        Neither convention is claimed as an observed B3 invoice for that case.
        """
        ids = np.flatnonzero(self.name == name)
        if not len(ids):
            return _tensor(0.0)
        if np.any(self.return_day[ids] >= 0) or len(np.unique(self.root[ids])) != len(
            ids
        ):
            raise ValueError(
                "loan fractions require settled returns and unsplit original contracts"
            )
        converted = self.quantity[ids] * ratio
        whole = converted.floor()
        fraction = (converted - whole).sum()
        quantity = self.quantity.clone()
        quantity[ids] = whole / ratio
        self.quantity = quantity
        zero = ids[whole.detach().numpy() == 0]
        self.return_day[zero] = (
            auction.payment_session
            if auction.zero_quantity_rent_through_payment
            else day
        )
        if not auction.zero_quantity_rent_through_payment:
            self.accrual_end[zero] = day - 1
        return fraction

    def deliver(self, source, destination, ratio, allocation, *, final):
        """Transfer a contractual portion without repricing its principal/rate.

        Allocation is the fraction of remaining loan principal assigned to the
        delivered leg, distinct from the number of deliverable successor shares.
        """
        ids = np.flatnonzero(self.name == source)
        if not len(ids):
            return
        for key in ("quantity", "principal", "rent_due", "fees_due"):
            value = getattr(self, key)
            incoming = value[ids] * (ratio if key == "quantity" else allocation)
            remaining = np.ones(len(self.name))
            remaining[ids] = (
                0 if final else (1 if key == "quantity" else 1 - allocation)
            )
            share = _tensor(remaining[:, None] if value.ndim == 2 else remaining)
            setattr(self, key, torch.cat((value * share, incoming)))
        self.name = np.r_[self.name, np.full(len(ids), destination)]
        for key in (
            "opened",
            "return_day",
            "accrual_end",
            "root",
            "annual_rate",
            "fee_rate",
            "fee_growth",
            "fee_period",
            "imputed",
            "placeholder",
        ):
            value = getattr(self, key)
            setattr(self, key, np.concatenate((value, value[ids])))
        self._keep(
            (self.quantity.detach().numpy() > 0)
            | (self.principal.detach().numpy() != 0)
        )

    def detach(self):
        for key in ("quantity", "principal", "rent_due", "fees_due", "root_fees"):
            setattr(self, key, getattr(self, key).detach())
        self.cash_claims = [
            (event, quantity.detach(), value)
            for event, quantity, value in self.cash_claims
        ]

    def detached_copy(self):
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, torch.Tensor):
                value = value.detach().clone()
            elif isinstance(value, np.ndarray):
                value = value.copy()
            elif item.name == "cash_claims":
                value = [
                    (event, quantity.detach().clone(), mark)
                    for event, quantity, mark in value
                ]
            values[item.name] = value
        return LoanContracts(**values)
