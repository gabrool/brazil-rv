"""Causal risk-scaling overlays evaluated on saved daily book returns.

A scale ``m_t`` in ``(0, 1]`` multiplies the book's CDI-excess return on day
``t``; the unused fraction of capital is cash earning CDI, so excess over CDI
scales linearly. ``m_t`` may use only information strictly before day ``t``.
Scaling down never binds a planned cap, so the ex-post approximation is exact for
the linear components of the account (stock and hedge P&L, proportional costs and
borrow, fully remunerated proceeds). It ignores fixed minimum fees and the
allocator's path dependence; an exact test reruns the ledger with a daily gross
cap of ``m_t`` times the planned cap. Rescaling trades are charged explicitly.
"""

from __future__ import annotations

import math

import numpy as np

from .performance import maximum_drawdown, sharpe

SESSIONS_PER_YEAR = 252


def trailing_volatility(returns, window, *, min_periods=None):
    """Annualised sample volatility of the ``window`` returns strictly before ``t``."""
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("returns must be a one-dimensional daily series")
    if window < 2:
        raise ValueError("volatility window must cover at least two sessions")
    minimum = window if min_periods is None else max(2, int(min_periods))
    result = np.full(values.shape, np.nan)
    for day in range(values.size):
        sample = values[max(0, day - window) : day]
        sample = sample[np.isfinite(sample)]
        if sample.size >= minimum:
            result[day] = float(sample.std(ddof=1) * math.sqrt(SESSIONS_PER_YEAR))
    return result


def volatility_target_scale(volatility, target_annual, *, floor=0.0, cap=1.0):
    """``clip(target / volatility, floor, cap)``; unknown volatility keeps the cap."""
    if not 0.0 <= floor <= cap:
        raise ValueError("scale floor must lie in [0, cap]")
    if target_annual <= 0:
        raise ValueError("target volatility must be positive")
    vol = np.asarray(volatility, dtype=np.float64)
    known = np.isfinite(vol) & (vol > 0)
    ratio = np.where(known, target_annual / np.where(known, vol, 1.0), cap)
    return np.clip(ratio, floor, cap)


def combine_scales(*scales):
    """Elementwise minimum of several causal scales (the most defensive one binds)."""
    if not scales:
        raise ValueError("at least one scale is required")
    return np.minimum.reduce([np.asarray(s, dtype=np.float64) for s in scales])


def apply_scale(
    excess_bps, scale, *, previous_gross=2.0, cost_bps=4.0, initial_scale=1.0
):
    """Scale daily excess returns and charge the turnover created by scale changes.

    Changing the multiplier from ``m_{t-1}`` to ``m_t`` trades ``|m_t - m_{t-1}|``
    times the previous day's gross exposure (fraction of NAV), charged once at
    ``cost_bps`` per unit notional. The book's own trading costs are inside its
    excess return and scale with ``m_t`` like every other proportional component.
    """
    excess = np.asarray(excess_bps, dtype=np.float64)
    multiplier = np.asarray(scale, dtype=np.float64)
    if excess.shape != multiplier.shape:
        raise ValueError("scale must align with the daily excess returns")
    if np.any(multiplier < 0) or np.any(multiplier > 1.0 + 1e-12):
        raise ValueError(
            "overlay scales must lie in [0, 1]; this is a de-risking overlay"
        )
    gross = np.broadcast_to(np.asarray(previous_gross, dtype=np.float64), excess.shape)
    previous = np.concatenate(([initial_scale], multiplier[:-1]))
    rescale_turnover = np.abs(multiplier - previous) * gross
    scaled = multiplier * excess - rescale_turnover * cost_bps
    return scaled, rescale_turnover


def summarize(excess_bps, cdi_bps):
    """Currency-consistent statistics matching the repository performance helpers."""
    excess = np.asarray(excess_bps, dtype=np.float64) / 1e4
    cash = np.asarray(cdi_bps, dtype=np.float64) / 1e4
    if excess.shape != cash.shape or not np.isfinite(excess).all():
        raise ValueError("excess and CDI series must align and be finite")
    absolute = excess + cash
    return {
        "sessions": int(excess.size),
        "mean_excess_bps": float(excess.mean() * 1e4),
        "annualized_volatility": float(
            absolute.std(ddof=1) * math.sqrt(SESSIONS_PER_YEAR)
        ),
        "sharpe_zero_rate": sharpe(absolute),
        "sharpe_cdi_excess": sharpe(excess),
        "maximum_drawdown": maximum_drawdown(absolute),
        "compounded_excess_over_cdi": float(np.prod((1 + absolute) / (1 + cash)) - 1),
    }


def circular_block_indices(length, *, block, draws, seed):
    """Index draws for a circular moving-block bootstrap of one daily series."""
    if length < 2 or block < 1 or draws < 1:
        raise ValueError(
            "bootstrap requires at least two sessions, a positive block and draws"
        )
    rng = np.random.default_rng(seed)
    blocks = math.ceil(length / block)
    starts = rng.integers(0, length, size=(draws, blocks))
    offsets = np.arange(block)
    indices = (starts[:, :, None] + offsets[None, None, :]) % length
    return indices.reshape(draws, blocks * block)[:, :length]


def paired_bootstrap(
    base_excess_bps,
    overlay_excess_bps,
    cdi_bps,
    *,
    block=40,
    draws=10000,
    seed=20260921,
):
    """Paired circular block-bootstrap intervals for mean-excess and Sharpe differences."""
    base = np.asarray(base_excess_bps, dtype=np.float64)
    overlay = np.asarray(overlay_excess_bps, dtype=np.float64)
    cash = np.asarray(cdi_bps, dtype=np.float64)
    indices = circular_block_indices(base.size, block=block, draws=draws, seed=seed)
    delta = overlay - base
    mean_delta = delta[indices].mean(axis=1)

    def sharpe_rows(series):
        values = series[indices]
        sigma = values.std(axis=1, ddof=1)
        return np.where(
            sigma > 1e-12,
            math.sqrt(SESSIONS_PER_YEAR) * values.mean(axis=1) / sigma,
            np.nan,
        )

    sharpe_delta = sharpe_rows(overlay + cash) - sharpe_rows(base + cash)
    finite = np.isfinite(sharpe_delta)
    return {
        "block_sessions": int(block),
        "draws": int(draws),
        "circular": True,
        "mean_excess_delta_bps": float(delta.mean()),
        "mean_excess_delta_interval": [
            float(np.percentile(mean_delta, 2.5)),
            float(np.percentile(mean_delta, 97.5)),
        ],
        "sharpe_zero_rate_delta_interval": [
            float(np.percentile(sharpe_delta[finite], 2.5)),
            float(np.percentile(sharpe_delta[finite], 97.5)),
        ]
        if finite.any()
        else None,
    }


def evaluate_overlay(
    daily, scale, *, cost_bps=4.0, block=40, draws=10000, seed=20260921
):
    """Compare a saved book with its causally rescaled counterpart."""
    excess = np.asarray(daily["net_excess_bps"], dtype=np.float64)
    cash = np.asarray(daily["cdi_bps"], dtype=np.float64)
    gross = daily.get("gross")
    if gross is None:
        previous_gross = np.full(excess.shape, 2.0)
    else:
        gross = np.asarray(gross, dtype=np.float64)
        previous_gross = np.concatenate(([gross[0]], gross[:-1]))
    scaled, turnover = apply_scale(
        excess, scale, previous_gross=previous_gross, cost_bps=cost_bps
    )
    base, overlay = summarize(excess, cash), summarize(scaled, cash)
    multiplier = np.asarray(scale, dtype=np.float64)
    return {
        "base": base,
        "overlay": overlay,
        "sharpe_zero_rate_delta": (
            None
            if base["sharpe_zero_rate"] is None or overlay["sharpe_zero_rate"] is None
            else overlay["sharpe_zero_rate"] - base["sharpe_zero_rate"]
        ),
        "mean_scale": float(multiplier.mean()),
        "fraction_scaled_below_one": float((multiplier < 1.0 - 1e-12).mean()),
        "mean_rescale_turnover": float(turnover.mean()),
        "mean_rescale_cost_bps": float((turnover * cost_bps).mean()),
        "paired": paired_bootstrap(
            excess, scaled, cash, block=block, draws=draws, seed=seed
        ),
        "approximation": (
            "ex-post linear rescaling with explicit rescale turnover; exact for "
            "proportional account components, ignores fixed minimum fees and allocator "
            "path dependence; confirm with a ledger rerun at a daily gross cap of "
            "scale times the planned cap"
        ),
    }


def trailing_external_volatility(
    book_dates, series_dates, series_returns, window, *, min_periods=None
):
    """Annualised volatility of an external daily series using observations before each book date."""
    book = np.asarray(book_dates, dtype="datetime64[D]")
    dates = np.asarray(series_dates, dtype="datetime64[D]")
    returns = np.asarray(series_returns, dtype=np.float64)
    if dates.shape != returns.shape or dates.ndim != 1:
        raise ValueError("external series dates and returns must align")
    order = np.argsort(dates, kind="stable")
    dates, returns = dates[order], returns[order]
    minimum = window if min_periods is None else max(2, int(min_periods))
    result = np.full(book.shape, np.nan)
    for i, day in enumerate(book):
        stop = int(np.searchsorted(dates, day, side="left"))
        sample = returns[max(0, stop - window) : stop]
        sample = sample[np.isfinite(sample)]
        if sample.size >= minimum:
            result[i] = float(sample.std(ddof=1) * math.sqrt(SESSIONS_PER_YEAR))
    return result
