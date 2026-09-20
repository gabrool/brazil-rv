"""Monthly custody charges on ordinary settled stock, including loan pipelines.

Assessment/payment dates are explicit calendar contracts, never inferred from a
replay window's last row. Client debit timing and close-price valuation are
hypotheses. Corporate physical transitions require separate admission.
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
        if not np.datetime64("2023-01-01") <= date <= np.datetime64("2024-12-30"):
            raise ValueError("custody tariff admitted only for 2023-2024")
        if pay < date:
            raise ValueError("custody payment cannot precede assessment")


def monthly_custody_fee(value, date):
    CustodyAssessment(str(date), str(date))
    value = torch.as_tensor(value, dtype=torch.float64)
    threshold = 23084.39 if str(date)[:4] == "2023" else 24164.73
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

    def split(self, name, ratio):
        changed = []
        for due, quantity in self.pending:
            q = quantity.clone()
            q[name] = q[name] * ratio
            changed.append((due, q))
        self.pending = changed

    def require_ordinary(self, name, shares, loans):
        pending = any(float(q[name].detach()) != 0 for _, q in self.pending)
        if float(shares) != 0 or pending or bool((loans.name == name).any()):
            raise ValueError("corporate physical custody requires separate admission")

    def close(self, day, date, shares, loans, marks, assessments, *, economic=False):
        self.pending = [(d, q) for d, q in self.pending if d > day]
        shares = torch.as_tensor(shares, dtype=torch.float64)
        marks = torch.as_tensor(marks, dtype=torch.float64)
        borrowed = loans._by_name(
            loans.quantity * torch.as_tensor(loans.accrual_end < 0)
        )
        physical = (
            shares
            + borrowed
            - sum((q for _, q in self.pending), torch.zeros_like(shares))
        )
        tolerance = 1e-10 * (shares.abs() + borrowed + 1)
        if bool((physical.detach() < -tolerance.detach()).any()):
            raise ValueError("negative physical custody requires settlement admission")
        physical = physical.clamp_min(0)
        fee = torch.tensor(0.0, dtype=torch.float64)
        base = torch.tensor(0.0, dtype=torch.float64)
        for assessment in assessments:
            if assessment.date == str(date):
                quantity = shares.clamp_min(0) if economic else physical
                if bool(
                    (
                        (quantity.detach() > 0)
                        & (~torch.isfinite(marks) | (marks <= 0))
                    ).any()
                ):
                    raise ValueError("custody holdings require an admitted valuation")
                base = (quantity * torch.nan_to_num(marks)).sum()
                fee = monthly_custody_fee(base, date)
                self.invoices.append((assessment.payment_date, fee))
        paid = sum((f for d, f in self.invoices if d == str(date)), fee * 0)
        self.invoices = [(d, f) for d, f in self.invoices if d != str(date)]
        return base, fee, paid, physical

    def detach(self):
        self.pending = [(d, q.detach()) for d, q in self.pending]
        self.invoices = [(d, q.detach()) for d, q in self.invoices]

    def detached_copy(self):
        return CustodyFees(
            self.names,
            [(d, q.detach().clone()) for d, q in self.pending],
            [(d, q.detach().clone()) for d, q in self.invoices],
        )
