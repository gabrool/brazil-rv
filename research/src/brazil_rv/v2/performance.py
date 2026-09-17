"""Currency-consistent account statistics and dated cash/FX benchmark alignment."""

from __future__ import annotations

import numpy as np


def align_benchmarks(dates, previous_date, fx_dates, fx, rate_dates, annual_us_percent):
    """Valuation-only PTAX and EFFR ACT/360; no future fill, calendar days included.

    EFFR's effective-date rate earns interest over that overnight period. Its next
    morning publication is irrelevant to ex-post benchmarking; these arrays must
    not be used as decision features. Non-fixing dates use the last dated PTAX,
    with age retained. This is reference-fix USD valuation, not an executable FX fill.
    """
    endpoints = np.r_[
        np.datetime64(previous_date, "D"), np.asarray(dates, dtype="datetime64[D]")
    ]
    fx_dates, rate_dates = (
        np.asarray(x, dtype="datetime64[D]") for x in (fx_dates, rate_dates)
    )
    if (np.diff(endpoints).astype(int) <= 0).any():
        raise ValueError("account endpoints must be strictly chronological")
    index = np.searchsorted(fx_dates, endpoints, side="right") - 1
    calendar = np.arange(endpoints[0], endpoints[-1], dtype="datetime64[D]")
    ri = np.searchsorted(rate_dates, calendar, side="right") - 1
    if (index < 0).any() or (ri < 0).any():
        raise ValueError("cash/FX benchmark lacks a prior observation")
    fixes = np.asarray(fx)[index]
    rates = np.asarray(annual_us_percent)[ri] / 100 / 360
    if (
        not np.isfinite(fixes).all()
        or (fixes <= 0).any()
        or not np.isfinite(rates).all()
    ):
        raise ValueError("invalid cash/FX benchmark")
    accumulated = np.r_[0.0, np.cumsum(np.log1p(rates))]
    positions = (endpoints - endpoints[0]).astype(int)
    cash = np.expm1(np.diff(accumulated[positions]))
    return {
        "fx_ratio": fixes[:-1] / fixes[1:],
        "us_cash": cash,
        "fx_age_days": (endpoints - fx_dates[index]).astype(int),
        "us_rate_max_age_days": int((calendar - rate_dates[ri]).astype(int).max()),
    }


def sharpe(returns):
    values = np.asarray(returns, dtype=np.float64)
    sigma = values.std(ddof=1) if len(values) > 1 else 0.0
    return float(np.sqrt(252) * values.mean() / sigma) if sigma > 1e-12 else None


def maximum_drawdown(returns):
    wealth = np.r_[1.0, np.cumprod(1 + np.asarray(returns))]
    return float((wealth / np.maximum.accumulate(wealth) - 1).min())


def performance(absolute_brl, cdi, benchmarks=None):
    r, cash = (np.asarray(x, dtype=np.float64) for x in (absolute_brl, cdi))
    if r.shape != cash.shape or not np.isfinite(r).all() or not np.isfinite(cash).all():
        raise ValueError("performance requires aligned finite account and CDI returns")
    excess = r - cash
    result = {
        "sessions": len(r),
        "sharpe_brl_minus_cdi": sharpe(excess),
        "sharpe_brl_minus_zero": sharpe(r),
        "absolute_compounded_return_brl": float(np.prod(1 + r) - 1),
        "relative_to_cdi_compounded_return": float(np.prod((1 + r) / (1 + cash)) - 1),
        "annualized_volatility_brl": float(r.std(ddof=1) * np.sqrt(252)),
        "maximum_drawdown_brl": maximum_drawdown(r),
    }
    for label, values in (("absolute_brl", r), ("cdi_excess", excess)):
        result[label + "_winning_fraction"] = float((values > 1e-12).mean())
        result[label + "_losing_fraction"] = float((values < -1e-12).mean())
        result[label + "_flat_fraction"] = float((np.abs(values) <= 1e-12).mean())
    if benchmarks is not None:
        ratio, us_cash = benchmarks["fx_ratio"], benchmarks["us_cash"]
        if np.shape(ratio) != r.shape or np.shape(us_cash) != r.shape:
            raise ValueError("USD benchmark dates differ from account dates")
        usd = (1 + r) * ratio - 1
        result.update(
            sharpe_usd_minus_us_cash=sharpe(usd - us_cash),
            absolute_compounded_return_usd=float(np.prod(1 + usd) - 1),
            maximum_drawdown_usd=maximum_drawdown(usd),
            fx_carried_endpoint_count=int((benchmarks["fx_age_days"] > 0).sum()),
            fx_max_age_days=int(benchmarks["fx_age_days"].max()),
            us_rate_max_age_days=benchmarks["us_rate_max_age_days"],
            us_cash_proxy="EFFR, ACT/360 over calendar days",
            fx_valuation="BCB PTAX BRL per USD; last dated fix on holidays",
        )
    return result
