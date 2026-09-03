from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .contract import DECISION_MINUTE_INDEX, INTRADAY_DAILY_FEATURES


@dataclass(frozen=True)
class IntradayDailyResult:
    values: NDArray[np.float32]
    valid: NDArray[np.bool_]
    entry_open: NDArray[np.float64]
    entry_open_valid: NDArray[np.bool_]
    session_close: NDArray[np.float64]
    session_close_valid: NDArray[np.bool_]
    realized_daily_vol: NDArray[np.float64]
    fast_present: NDArray[np.bool_]
    feature_names: tuple[str, ...] = INTRADAY_DAILY_FEATURES


def _validate_minutes(
    open_price: NDArray[np.floating],
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    close: NDArray[np.floating],
    volume: NDArray[np.floating],
    observed: NDArray[np.bool_],
    cutoff: int,
) -> tuple[NDArray[np.float64], ...]:
    arrays = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (open_price, high, low, close, volume)
    )
    seen = np.asarray(observed, dtype=np.bool_)
    if arrays[0].ndim != 3 or any(value.shape != arrays[0].shape for value in arrays) or seen.shape != arrays[0].shape:
        raise ValueError("M1 arrays must be aligned [date, name, minute]")
    if cutoff <= 0 or cutoff >= arrays[0].shape[2] or cutoff % 5:
        raise ValueError("decision cutoff must be an in-session five-minute boundary")
    return (*arrays, seen)


def _safe_log_ratio(numerator: NDArray[np.floating], denominator: NDArray[np.floating]) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    top = np.asarray(numerator, dtype=np.float64)
    bottom = np.asarray(denominator, dtype=np.float64)
    valid = np.isfinite(top) & np.isfinite(bottom) & (top > 0) & (bottom > 0)
    output = np.full(top.shape, np.nan, dtype=np.float64)
    output[valid] = np.log(top[valid] / bottom[valid])
    return output, valid


def _rolling_sum(
    values: NDArray[np.float64], valid: NDArray[np.bool_], window: int
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    output = np.full(values.shape, np.nan, dtype=np.float64)
    mask = np.zeros(values.shape, dtype=np.bool_)
    for end in range(window - 1, values.shape[0]):
        complete = valid[end - window + 1 : end + 1].all(axis=0)
        output[end, complete] = values[end - window + 1 : end + 1, complete].sum(axis=0)
        mask[end, complete] = True
    return output, mask


def _rolling_mean(
    values: NDArray[np.float64], valid: NDArray[np.bool_], window: int
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    output = np.full(values.shape, np.nan, dtype=np.float64)
    mask = np.zeros(values.shape, dtype=np.bool_)
    for end in range(window - 1, values.shape[0]):
        complete = valid[end - window + 1 : end + 1].all(axis=0)
        output[end, complete] = values[end - window + 1 : end + 1, complete].mean(axis=0)
        mask[end, complete] = True
    return output, mask


def five_minute_returns(
    open_price: NDArray[np.floating],
    close: NDArray[np.floating],
    observed: NDArray[np.bool_],
    *,
    cutoff: int = DECISION_MINUTE_INDEX,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Build completed five-minute returns from minutes strictly before cutoff."""

    open_ = np.asarray(open_price, dtype=np.float64)
    close_ = np.asarray(close, dtype=np.float64)
    seen = np.asarray(observed, dtype=np.bool_)
    if open_.shape != close_.shape or open_.shape != seen.shape or open_.ndim != 3:
        raise ValueError("five-minute inputs are misaligned")
    if cutoff <= 0 or cutoff > open_.shape[2] or cutoff % 5:
        raise ValueError("invalid five-minute cutoff")
    block_count = cutoff // 5
    starts = open_[..., :cutoff:5]
    ends = close_[..., 4:cutoff:5]
    block_seen = seen[..., :cutoff].reshape(*seen.shape[:2], block_count, 5).all(axis=-1)
    returns, valid = _safe_log_ratio(ends, starts)
    valid &= block_seen
    returns[~valid] = np.nan
    return returns, valid


def _daily_realized_vol(
    returns: NDArray[np.float64], valid: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    minimum = int(np.ceil(0.8 * returns.shape[2]))
    count = valid.sum(axis=2)
    mask = count >= minimum
    values = np.sqrt(np.where(valid, returns**2, 0.0).sum(axis=2))
    values[~mask] = np.nan
    return values, mask


def _rolling_skew_5m(
    returns: NDArray[np.float64], valid: NDArray[np.bool_], window: int
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    def rolling_sum(values: NDArray[np.floating]) -> NDArray[np.float64]:
        cumulative = np.concatenate(
            (
                np.zeros((1, values.shape[1]), dtype=np.float64),
                np.cumsum(values, axis=0, dtype=np.float64),
            ),
            axis=0,
        )
        output = np.zeros(values.shape, dtype=np.float64)
        output[window - 1 :] = cumulative[window:] - cumulative[:-window]
        return output

    clean = np.where(valid, returns, 0.0)
    count = rolling_sum(valid.sum(axis=2, dtype=np.int32))
    first = rolling_sum(clean.sum(axis=2))
    second = rolling_sum((clean**2).sum(axis=2))
    third = rolling_sum((clean**3).sum(axis=2))
    minimum = int(np.ceil(0.8 * window * returns.shape[2]))
    safe_count = np.maximum(count, 1.0)
    mean = first / safe_count
    variance = np.maximum(second / safe_count - mean**2, 0.0)
    third_central = third / safe_count - 3.0 * mean * second / safe_count + 2.0 * mean**3
    mask = (count >= minimum) & (variance > 0)
    output = np.full(returns.shape[:2], np.nan, dtype=np.float64)
    output[mask] = third_central[mask] / np.power(variance[mask], 1.5)
    return output, mask


def _rolling_roll_spread(
    returns: NDArray[np.float64], valid: NDArray[np.bool_], window: int
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    pairs = valid[..., 1:] & valid[..., :-1]
    left = np.where(pairs, returns[..., :-1], 0.0)
    right = np.where(pairs, returns[..., 1:], 0.0)

    def rolling_sum(values: NDArray[np.floating]) -> NDArray[np.float64]:
        cumulative = np.concatenate(
            (
                np.zeros((1, values.shape[1]), dtype=np.float64),
                np.cumsum(values, axis=0, dtype=np.float64),
            ),
            axis=0,
        )
        output = np.zeros(values.shape, dtype=np.float64)
        output[window - 1 :] = cumulative[window:] - cumulative[:-window]
        return output

    count = rolling_sum(pairs.sum(axis=2, dtype=np.int32))
    sum_left = rolling_sum(left.sum(axis=2))
    sum_right = rolling_sum(right.sum(axis=2))
    cross = rolling_sum((left * right).sum(axis=2))
    safe_count = np.maximum(count, 1.0)
    # Roll's estimator is defined from the sample serial covariance.  The
    # spread is undefined when covariance is non-negative; zero is not a
    # measured zero spread and must remain masked.
    covariance_numerator = cross - sum_left * sum_right / safe_count
    covariance = covariance_numerator / np.maximum(count - 1.0, 1.0)
    minimum = int(np.ceil(0.8 * window * (returns.shape[2] - 1)))
    mask = (count >= max(minimum, 2)) & (covariance < 0.0)
    output = np.full(returns.shape[:2], np.nan, dtype=np.float64)
    output[mask] = 2.0 * np.sqrt(-covariance[mask])
    return output, mask


def _corwin_schultz(
    daily_high: NDArray[np.float64],
    daily_low: NDArray[np.float64],
    daily_valid: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    output = np.full(daily_high.shape, np.nan, dtype=np.float64)
    valid = np.zeros(daily_high.shape, dtype=np.bool_)
    denominator = 3.0 - 2.0 * np.sqrt(2.0)
    for day in range(1, daily_high.shape[0]):
        mask = daily_valid[day] & daily_valid[day - 1]
        if not mask.any():
            continue
        log_range_today = np.log(daily_high[day, mask] / daily_low[day, mask])
        log_range_prior = np.log(daily_high[day - 1, mask] / daily_low[day - 1, mask])
        beta = log_range_today**2 + log_range_prior**2
        high_two = np.maximum(daily_high[day, mask], daily_high[day - 1, mask])
        low_two = np.minimum(daily_low[day, mask], daily_low[day - 1, mask])
        gamma = np.log(high_two / low_two) ** 2
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denominator - np.sqrt(
            gamma / denominator
        )
        alpha = np.maximum(alpha, 0.0)
        spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
        output[day, mask] = spread
        valid[day, mask] = True
    return output, valid


def build_intraday_daily_features(
    open_price: NDArray[np.floating],
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    close: NDArray[np.floating],
    volume: NDArray[np.floating],
    observed: NDArray[np.bool_],
    *,
    cutoff: int = DECISION_MINUTE_INDEX,
) -> IntradayDailyResult:
    """Build all features visible immediately before the 15:45 entry bar."""

    open_, high_, low_, close_, volume_, seen = _validate_minutes(
        open_price, high, low, close, volume, observed, cutoff
    )
    shape = open_.shape[:2]
    values = np.zeros((*shape, len(INTRADAY_DAILY_FEATURES)), dtype=np.float32)
    masks = np.zeros(values.shape, dtype=np.bool_)

    def assign(index: int, column: NDArray[np.floating], valid: NDArray[np.bool_]) -> None:
        usable = np.asarray(valid, dtype=np.bool_) & np.isfinite(column)
        values[..., index][usable] = np.asarray(column)[usable].astype(np.float32)
        masks[..., index] = usable

    entry = open_[..., cutoff]
    entry_valid = seen[..., cutoff] & np.isfinite(entry) & (entry > 0)
    day_open = open_[..., 0]
    open_valid = seen[..., 0] & np.isfinite(day_open) & (day_open > 0)
    final_close = close_[..., -1]
    final_valid = seen[..., -1] & np.isfinite(final_close) & (final_close > 0)

    overnight = np.full(shape, np.nan, dtype=np.float64)
    overnight_valid = np.zeros(shape, dtype=np.bool_)
    if shape[0] > 1:
        overnight[1:], overnight_valid[1:] = _safe_log_ratio(day_open[1:], final_close[:-1])
        overnight_valid[1:] &= open_valid[1:] & final_valid[:-1]
        overnight[~overnight_valid] = np.nan
    intraday, intraday_valid = _safe_log_ratio(entry, day_open)
    intraday_valid &= entry_valid & open_valid
    intraday[~intraday_valid] = np.nan
    assign(0, overnight, overnight_valid)
    assign(1, intraday, intraday_valid)
    for index, base, base_valid, window in (
        (2, overnight, overnight_valid, 5),
        (3, overnight, overnight_valid, 20),
        (4, intraday, intraday_valid, 5),
        (5, intraday, intraday_valid, 20),
    ):
        assign(index, *_rolling_sum(base, base_valid, window))
    differential = overnight - intraday
    differential_valid = overnight_valid & intraday_valid
    assign(6, differential, differential_valid)
    assign(7, *_rolling_mean(differential, differential_valid, 20))

    last30, last30_valid = _safe_log_ratio(final_close, open_[..., -30])
    full_return, full_valid = _safe_log_ratio(final_close, day_open)
    last30_valid &= final_valid & seen[..., -30]
    full_valid &= final_valid & open_valid
    with np.errstate(divide="ignore", invalid="ignore"):
        last30_share = last30 / full_return
    last30_share_valid = last30_valid & full_valid & (np.abs(full_return) > 1e-12)
    prior_last30 = np.full(shape, np.nan, dtype=np.float64)
    prior_last30_valid = np.zeros(shape, dtype=np.bool_)
    prior_last30[1:] = last30_share[:-1]
    prior_last30_valid[1:] = last30_share_valid[:-1]
    assign(8, prior_last30, prior_last30_valid)

    full_volume = np.where(seen & np.isfinite(volume_), np.maximum(volume_, 0.0), 0.0).sum(axis=2)
    last_hour_volume = np.where(
        seen[..., -60:] & np.isfinite(volume_[..., -60:]),
        np.maximum(volume_[..., -60:], 0.0),
        0.0,
    ).sum(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        last_hour_share = last_hour_volume / full_volume
    full_volume_valid = seen.sum(axis=2) >= int(np.ceil(0.8 * seen.shape[2]))
    last_hour_valid = (
        (full_volume > 0)
        & full_volume_valid
        & (seen[..., -60:].sum(axis=2) >= 48)
    )
    lag_last_hour = np.full(shape, np.nan, dtype=np.float64)
    lag_last_hour_valid = np.zeros(shape, dtype=np.bool_)
    lag_last_hour[1:] = last_hour_share[:-1]
    lag_last_hour_valid[1:] = last_hour_valid[:-1]
    assign(9, lag_last_hour, lag_last_hour_valid)

    safe_full_volume = np.where(seen & np.isfinite(volume_), np.maximum(volume_, 0.0), 0.0)
    full_vwap = (np.where(seen, close_, 0.0) * safe_full_volume).sum(axis=2) / np.maximum(
        safe_full_volume.sum(axis=2), 1.0
    )
    close_vwap, close_vwap_valid = _safe_log_ratio(final_close, full_vwap)
    close_vwap_valid &= (
        final_valid & full_volume_valid & (safe_full_volume.sum(axis=2) > 0)
    )
    lag_close_vwap = np.full(shape, np.nan, dtype=np.float64)
    lag_close_vwap_valid = np.zeros(shape, dtype=np.bool_)
    lag_close_vwap[1:] = close_vwap[:-1]
    lag_close_vwap_valid[1:] = close_vwap_valid[:-1]
    assign(10, lag_close_vwap, lag_close_vwap_valid)

    prefix_seen = seen[..., :cutoff]
    prefix_volume = np.where(
        prefix_seen & np.isfinite(volume_[..., :cutoff]),
        np.maximum(volume_[..., :cutoff], 0.0),
        0.0,
    )
    prefix_vwap = (
        np.where(prefix_seen, close_[..., :cutoff], 0.0) * prefix_volume
    ).sum(axis=2) / np.maximum(prefix_volume.sum(axis=2), 1.0)
    vwap_deviation, vwap_valid = _safe_log_ratio(entry, prefix_vwap)
    prefix_valid = prefix_seen.sum(axis=2) >= int(np.ceil(0.8 * cutoff))
    vwap_valid &= entry_valid & prefix_valid & (prefix_volume.sum(axis=2) > 0)
    assign(11, vwap_deviation, vwap_valid)

    block_returns, block_valid = five_minute_returns(open_, close_, seen, cutoff=cutoff)
    realized, realized_valid = _daily_realized_vol(block_returns, block_valid)
    assign(12, realized, realized_valid)
    assign(13, *_rolling_mean(realized, realized_valid, 5))
    realized20, realized20_valid = _rolling_mean(realized, realized_valid, 20)
    assign(14, realized20, realized20_valid)
    assign(15, *_rolling_skew_5m(block_returns, block_valid, 20))
    assign(16, *_rolling_roll_spread(block_returns, block_valid, 20))

    prefix_high = np.where(prefix_seen, high_[..., :cutoff], np.nan)
    prefix_low = np.where(prefix_seen, low_[..., :cutoff], np.nan)
    with np.errstate(all="ignore"):
        day_high = np.nanmax(prefix_high, axis=2)
        day_low = np.nanmin(prefix_low, axis=2)
    day_range_valid = (
        (prefix_seen.sum(axis=2) >= int(0.8 * cutoff))
        & np.isfinite(day_high)
        & np.isfinite(day_low)
        & (day_high >= day_low)
        & (day_low > 0)
    )
    spread, spread_valid = _corwin_schultz(day_high, day_low, day_range_valid)
    assign(17, *_rolling_mean(spread, spread_valid, 20))
    range_value, range_valid = _safe_log_ratio(day_high, day_low)
    range_valid &= day_range_valid
    assign(18, range_value, range_valid)

    prefix_total = prefix_volume.sum(axis=2)
    relative = np.full(shape, np.nan, dtype=np.float64)
    relative_valid = np.zeros(shape, dtype=np.bool_)
    for day in range(20, shape[0]):
        history_valid = prefix_valid[day - 20 : day].all(axis=0)
        median = np.median(prefix_total[day - 20 : day], axis=0)
        usable = (
            prefix_valid[day]
            & history_valid
            & np.isfinite(median)
            & (median > 0)
        )
        relative[day, usable] = prefix_total[day, usable] / median[usable]
        relative_valid[day, usable] = True
    assign(19, relative, relative_valid)
    return IntradayDailyResult(
        values=values,
        valid=masks,
        entry_open=entry.copy(),
        entry_open_valid=entry_valid,
        session_close=np.where(final_valid, final_close, np.nan),
        session_close_valid=final_valid,
        realized_daily_vol=realized,
        fast_present=entry_valid & realized_valid,
    )
