"""Dated money-market accrual between equity closes, without changing features."""

from dataclasses import replace
import numpy as np


def close_interval_returns(source_dates, source_returns, equity_dates):
    """Compound money dates in [previous equity close, current equity close).

    Missing coverage stays NaN. A weekend or holiday with no monetary observation
    adds no fabricated observation. The first equity row has no prior interval.
    """
    money = np.asarray(source_dates, dtype="datetime64[D]")
    rates = np.asarray(source_returns, dtype=np.float64)
    dates = np.asarray(equity_dates, dtype="datetime64[D]")
    if (
        len(money) != len(rates)
        or not len(money)
        or np.any(money[1:] <= money[:-1])
        or np.any(dates[1:] <= dates[:-1])
        or not np.all(np.isfinite(rates) & (rates > -1))
    ):
        raise ValueError("cash source requires ordered unique dates and valid returns")
    result = np.full(len(dates), np.nan)
    # Small independent products avoid subtraction of decades of cumulative logs.
    for day in range(1, len(dates)):
        if money[0] <= dates[day - 1] and dates[day] <= money[-1]:
            first, last = np.searchsorted(money, [dates[day - 1], dates[day]])
            result[day] = np.prod(1 + rates[first:last]) - 1
    return result


def apply_cash_calendar(inputs, calendar, returns, manifest_sha256):
    """Explicit accounting/benchmark amendment after loading frozen PolicyData."""
    indices = np.asarray(inputs.session_indices)
    if not np.array_equal(
        np.asarray(inputs.dates, dtype="datetime64[D]"), calendar[indices]
    ):
        raise ValueError("cash calendar differs from evaluation axes")
    values = np.asarray(returns)[indices]
    if not np.isfinite(values).all():
        raise ValueError("cash source does not cover the requested evaluation interval")
    provenance = dict(inputs.source_artifact_hashes or {})
    if "cash_calendar" in provenance:
        raise ValueError("cash calendar already applied")
    provenance["cash_calendar"] = manifest_sha256
    return replace(inputs, cdi_returns=values, source_artifact_hashes=provenance)
