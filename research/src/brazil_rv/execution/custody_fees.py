"""Monthly custody charges on ordinary settled stock, including loan pipelines.

Assessment/payment dates are explicit calendar contracts, never inferred from a
replay window's last row. Client debit timing and close-price valuation are
hypotheses. Undelivered corporate rights have a separately reported base.
"""

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass(frozen=True)
class CustodyAssessment:
    date: str
    payment_date: str

    def __post_init__(self):
        date, pay = np.datetime64(self.date), np.datetime64(self.payment_date)
        if not np.datetime64("2016-07-18") <= date <= np.datetime64("2024-12-30"):
            raise ValueError(
                "custody tariff admitted only for 2016-07-18 through 2024-12-30"
            )
        if pay < date:
            raise ValueError("custody payment cannot precede assessment")


def monthly_custody_fee(value, date):
    """Progressive value charge, excluding the separate active-account maintenance.

    082/2009 and 101/2018 supply the old brackets. Their literal exemption
    boundaries differ at exactly R$300,000; retain each dated wording. 177/2020
    starts on February 2, 2021; 014/2022 and 221/2023 update its exemption.
    """
    CustodyAssessment(str(date), str(date))
    value = torch.as_tensor(value, dtype=torch.float64)
    date = str(date)
    if date < "2021-02-02":
        limits = (0, 1e6, 10e6, 100e6, 1e9, 10e9, float("inf"))
        rates = (0.00013, 0.000072, 0.000032, 0.000025, 0.000015, 0.000005)
        fee = sum(
            (value - lo).clamp(min=0, max=hi - lo) * rate / 12
            for lo, hi, rate in zip(limits[:-1], limits[1:], rates)
        )
        exempt = value <= 300000 if date < "2019-01-01" else value < 300000
        return torch.where(exempt, 0.0, fee)
    threshold = (
        20000
        if date < "2023-01-01"
        else (23084.39 if date < "2024-01-01" else 24164.73)
    )
    limits = (0, 1e5, 2e5, 3e5, 1.7e6, 17e6, 170e6, 1.7e9, 17e9, float("inf"))
    rates = (
        0.0005,
        0.0004,
        0.0002,
        0.00013,
        0.000072,
        0.000032,
        0.000025,
        0.000015,
        0.000005,
    )
    fee = sum(
        (value - lo).clamp(min=0, max=hi - lo) * rate / 12
        for lo, hi, rate in zip(limits[:-1], limits[1:], rates)
    )
    return torch.where(value < threshold, 0.0, fee)


def monthly_custody_maintenance(value, date):
    """One active resident CNPJ/custodian account, even when month-end stock is zero.

    Explicit open/active-account hypothesis; no dormant-account or fund discount.
    Values are printed in 144/2015, 120/2016, 011/2017 and 101/2018. Active
    resident accounts become exempt under 177/2020 on February 2, 2021.
    """
    CustodyAssessment(str(date), str(date))
    value = torch.as_tensor(value, dtype=torch.float64)
    date = str(date)
    if date >= "2021-02-02":
        return value * 0
    low, high = (
        (7.59, 8.02)
        if date < "2017-01-01"
        else (8.18, 8.65)
        if date < "2018-01-01"
        else (8.40, 8.88)
        if date < "2019-01-01"
        else (8.78, 9.28)
    )
    return torch.where(
        value <= 5000, torch.full_like(value, low), torch.full_like(value, high)
    )


def custody_schedule(calendar, start, stop, payment_lag=0):
    """Known full calendar, inclusive date bounds; never assume its tail is month-end."""
    dates = np.asarray(calendar, dtype="datetime64[D]")
    month_end = np.flatnonzero(
        dates[:-1].astype("datetime64[M]") != dates[1:].astype("datetime64[M]")
    )
    selected = month_end[
        (dates[month_end] >= np.datetime64(start))
        & (dates[month_end] <= np.datetime64(stop))
    ]
    if payment_lag < 0 or (len(selected) and selected[-1] + payment_lag >= len(dates)):
        raise ValueError("custody payment requires the supplied future calendar")
    return tuple(
        CustodyAssessment(str(dates[i]), str(dates[i + payment_lag])) for i in selected
    )


@dataclass
class CustodyFees:
    names: int
    # Signed spot obligations and undelivered *new* loans. Renewal is not delivery.
    pending: list = field(default_factory=list)
    invoices: list = field(default_factory=list)
    # Extra economic units whose physical receipt is later: (from, until, units).
    corporate_pending: list = field(default_factory=list)

    @property
    def liability(self):
        return sum(
            (fee for _, fee in self.invoices), torch.tensor(0.0, dtype=torch.float64)
        )

    def fill(self, day, spot_day, signed, opened, loan_lag):
        signed = torch.as_tensor(signed, dtype=torch.float64)
        opened = torch.as_tensor(opened, dtype=torch.float64)
        if bool((signed.detach() != 0).any()):
            self.pending.append((spot_day, signed.clone()))
        if loan_lag and bool((opened.detach() != 0).any()):
            self.pending.append((day + loan_lag, opened.clone()))

    def split(self, name, ratio, *, day=None, delivery=None, shares=None, loans=None):
        if delivery is not None and delivery > day:
            borrowed = loans._by_name(
                loans.quantity * torch.as_tensor(loans.accrual_end < 0)
            )
            settled = (
                torch.as_tensor(shares, dtype=torch.float64)[name]
                + borrowed[name]
                - sum(
                    (q[name] for due, q in self.pending if due > day),
                    borrowed[name] * 0,
                )
            )
            extra = torch.zeros(self.names, dtype=torch.float64)
            extra[name] = settled.clamp_min(0) * (ratio - 1)
            self.corporate_pending.append((day, delivery, extra))
            for due, quantity in self.pending:
                if day < due < delivery:
                    extra = torch.zeros(self.names, dtype=torch.float64)
                    extra[name] = quantity[name] * (ratio - 1)
                    self.corporate_pending.append((due, delivery, extra))
        changed = []
        for due, quantity in self.pending:
            q = quantity.clone()
            q[name] = q[name] * ratio
            changed.append((due, q))
        self.pending = changed

    def deliver(
        self, source, destination, ratio, incoming, day, *, final, receipt_day=None
    ):
        adjusted = []
        for due, quantity in self.pending:
            changed = quantity.clone()
            if receipt_day is None:
                changed[destination] = quantity[destination] + quantity[source] * ratio
            if final:
                changed[source] = 0
            adjusted.append((due, changed))
        self.pending = adjusted
        adjusted = []
        for start, due, quantity in self.corporate_pending:
            changed = quantity.clone()
            changed[destination] = quantity[destination] + quantity[source] * ratio
            if final:
                changed[source] = 0
            adjusted.append((start, due, changed))
        self.corporate_pending = adjusted
        if receipt_day is not None and receipt_day > day:
            receipt = torch.zeros(self.names, dtype=torch.float64)
            receipt[destination] = torch.as_tensor(
                incoming, dtype=torch.float64
            ).clamp_min(0)
            self.corporate_pending.append((day, receipt_day, receipt))

    def cash_cancel(self, name, loans):
        ids = (loans.name == name) & (loans.accrual_end < 0)
        if bool((loans.quantity[ids].detach() > 0).any()):
            raise ValueError(
                "cash cancellation with live share loans requires explicit loan terms"
            )
        self.split(name, 0)
        adjusted = []
        for start, due, quantity in self.corporate_pending:
            changed = quantity.clone()
            changed[name] = 0
            adjusted.append((start, due, changed))
        self.corporate_pending = adjusted

    def close(
        self,
        day,
        date,
        shares,
        loans,
        marks,
        assessments,
        *,
        economic=False,
        claim_names=(),
        claim_fraction=1.0,
    ):
        self.pending = [(d, q) for d, q in self.pending if d > day]
        self.corporate_pending = [
            (s, d, q) for s, d, q in self.corporate_pending if d > day
        ]
        shares = torch.as_tensor(shares, dtype=torch.float64)
        marks = torch.as_tensor(marks, dtype=torch.float64)
        borrowed = loans._by_name(
            loans.quantity * torch.as_tensor(loans.accrual_end < 0)
        )
        rights = sum(
            (q for s, _, q in self.corporate_pending if s <= day),
            torch.zeros_like(shares),
        )
        physical = (
            shares
            + borrowed
            - sum((q for _, q in self.pending), torch.zeros_like(shares))
            - rights
        )
        rights = rights.clamp_min(0)
        if claim_names:
            indices = list(claim_names)
            rights = rights.clone()
            rights[indices] = physical[indices].clamp_min(0)
            physical = physical.clone()
            physical[indices] = 0
        tolerance = 1e-10 * (shares.abs() + borrowed + 1)
        if bool((physical.detach() < -tolerance.detach()).any()):
            raise ValueError("negative physical custody requires settlement admission")
        physical = physical.clamp_min(0)
        fee = torch.tensor(0.0, dtype=torch.float64)
        maintenance = fee * 0
        base = torch.tensor(0.0, dtype=torch.float64)
        physical_base, claim_base = base * 0, base * 0
        for assessment in assessments:
            if assessment.date == str(date):
                quantity = shares.clamp_min(0) if economic else physical + rights
                if bool(
                    (
                        (quantity.detach() > 0)
                        & (~torch.isfinite(marks) | (marks <= 0))
                    ).any()
                ):
                    raise ValueError("custody holdings require an admitted valuation")
                physical_base = (physical * torch.nan_to_num(marks)).sum()
                claim_base = (rights * torch.nan_to_num(marks)).sum()
                base = (
                    (quantity * torch.nan_to_num(marks)).sum()
                    if economic
                    else physical_base + claim_base * claim_fraction
                )
                maintenance = monthly_custody_maintenance(base, date)
                fee = monthly_custody_fee(base, date) + maintenance
                self.invoices.append((assessment.payment_date, fee))
        paid = sum((f for d, f in self.invoices if d == str(date)), fee * 0)
        self.invoices = [(d, f) for d, f in self.invoices if d != str(date)]
        return base, fee, paid, physical, maintenance, physical_base, claim_base

    def detach(self):
        self.pending = [(d, q.detach()) for d, q in self.pending]
        self.invoices = [(d, q.detach()) for d, q in self.invoices]
        self.corporate_pending = [
            (s, d, q.detach()) for s, d, q in self.corporate_pending
        ]

    def detached_copy(self):
        return CustodyFees(
            self.names,
            [(d, q.detach().clone()) for d, q in self.pending],
            [(d, q.detach().clone()) for d, q in self.invoices],
            [(s, d, q.detach().clone()) for s, d, q in self.corporate_pending],
        )
