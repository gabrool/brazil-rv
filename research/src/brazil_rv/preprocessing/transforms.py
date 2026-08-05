from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
from numpy.typing import NDArray

from .contract import (
    BETA_CLIP,
    BETA_EWMA_ALPHA,
    BETA_MIN_PAIRED_SESSIONS,
    BETA_VARIANCE_FLOOR,
    DECISION_EQUITY_INDICES,
    DOLLAR_VOLUME_LOG_CENTER,
    DOLLAR_VOLUME_LOG_SCALE,
    HORIZONS,
    MAD_NORMALIZATION,
    LIQUIDITY_SELECTED_RATE_ZERO_SLOW_CHANNEL_INDICES,
    MIN_ACTIVE_EQUITIES,
    MIN_ADJACENT_RETURNS_PER_DAY,
    OBSERVED_FRACTION_WINDOW,
    PRICE_FEATURE_CLIP,
    PRICE_VOL_FLOOR,
    PRICE_VOL_REFERENCE,
    RATE_VOL_FLOOR_BP,
    RATE_VOL_REFERENCE_BP,
    REALIZED_VOL_LOG_CLIP,
    REALIZED_VOL_LOG_FLOOR,
    REALIZED_VOL_MIN_FRACTION,
    REAL_VOLUME_LOG_CENTER,
    REAL_VOLUME_LOG_SCALE,
    RETURN_WINDOWS,
    SLOW_LONG_MIN_VALID,
    SLOW_LONG_WINDOW,
    SLOW_SHORT_MIN_VALID,
    SLOW_SHORT_WINDOW,
    VOL_EWMA_ALPHA,
    VOL_REGIME_CLIP,
    VOL_WARMUP_VALID_DAYS,
    VOLUME_FEATURE_CLIP,
    VOLUME_LOOKBACK_SESSIONS,
    VOLUME_MAD_FLOOR,
    VOLUME_MIN_OBSERVATIONS,
)


@dataclass(frozen=True)
class InstrumentFeatures:
    dynamic: NDArray[np.float32]
    dynamic_valid: NDArray[np.bool_]
    slow: NDArray[np.float32]
    slow_rank_valid: NDArray[np.bool_]
    sigma: NDArray[np.float64]
    data_ready: NDArray[np.bool_]
    daily_change: NDArray[np.float64]
    daily_change_valid: NDArray[np.bool_]


def build_equity_features(
    raw_grid: NDArray[np.float64],
    observed: NDArray[np.bool_],
    valid_day: NDArray[np.bool_],
    *,
    market_dates: tuple[date, ...] | None = None,
) -> InstrumentFeatures:
    """Build equity features with first-observed opening-price semantics."""
    return build_causal_features(
        raw_grid,
        observed,
        valid_day,
        is_rate=False,
        market_dates=market_dates,
        early_open_cutoff=DECISION_EQUITY_INDICES[0],
    )


def build_causal_features(
    raw_grid: NDArray[np.float64],
    observed: NDArray[np.bool_],
    valid_day: NDArray[np.bool_],
    *,
    is_rate: bool,
    extra_ready: NDArray[np.bool_] | None = None,
    market_dates: tuple[date, ...] | None = None,
    include_dollar_volume: bool = True,
    early_open_cutoff: int | None = None,
) -> InstrumentFeatures:
    """Build instrument features with state available strictly before each date."""
    date_count, minute_count, field_count = raw_grid.shape
    if field_count != 5 or observed.shape != (date_count, minute_count):
        raise ValueError("raw_grid must be [date, minute, OHLCV]")
    if valid_day.shape != (date_count,):
        raise ValueError("valid_day must align to the date axis")
    if extra_ready is None:
        extra_ready = np.ones(date_count, dtype=bool)
    if market_dates is None:
        market_dates = tuple(date(2000, 1, 3) for _ in range(date_count))
    if len(market_dates) != date_count:
        raise ValueError("market_dates must align to the date axis")
    if early_open_cutoff is not None and not 0 < early_open_cutoff <= minute_count:
        raise ValueError("early_open_cutoff must fall inside the minute grid")

    slow = np.zeros((date_count, 32), dtype=np.float32)
    # Overnight gap, trailing dollar volume, and trailing 20-day RV validity.
    slow_rank_valid = np.zeros((date_count, 3), dtype=bool)
    sigma_by_day = np.zeros(date_count, dtype=np.float64)
    data_ready = np.zeros(date_count, dtype=bool)

    summaries = _daily_summaries(
        raw_grid,
        observed,
        valid_day,
        is_rate=is_rate,
        early_open_cutoff=early_open_cutoff,
    )
    warmup_variances: list[float] = []
    ewma_variance: float | None = None
    volatility_floor = RATE_VOL_FLOOR_BP if is_rate else PRICE_VOL_FLOOR
    volatility_reference = RATE_VOL_REFERENCE_BP if is_rate else PRICE_VOL_REFERENCE

    for date_idx in range(date_count):
        ready = (
            ewma_variance is not None and valid_day[date_idx] and extra_ready[date_idx]
        )
        if ready:
            sigma = np.sqrt(max(ewma_variance, volatility_floor**2))
            sigma_by_day[date_idx] = sigma
            data_ready[date_idx] = True
            slow[date_idx, 0] = np.float32(
                np.clip(
                    np.log(sigma / volatility_reference),
                    -VOL_REGIME_CLIP,
                    VOL_REGIME_CLIP,
                )
            )
            _build_slow_day(
                slow[date_idx],
                slow_rank_valid[date_idx],
                summaries,
                sigma_by_day,
                valid_day,
                date_idx,
                sigma,
                minute_count,
                market_dates[date_idx],
                is_rate=is_rate,
                include_dollar_volume=include_dollar_volume,
            )

        daily_variance = _daily_variance(
            raw_grid[date_idx, :, 3], observed[date_idx], is_rate=is_rate
        )
        if valid_day[date_idx] and daily_variance is not None:
            if ewma_variance is None:
                warmup_variances.append(daily_variance)
                if len(warmup_variances) == VOL_WARMUP_VALID_DAYS:
                    ewma_variance = float(np.median(warmup_variances))
            else:
                ewma_variance = (
                    1.0 - VOL_EWMA_ALPHA
                ) * ewma_variance + VOL_EWMA_ALPHA * daily_variance

    dynamic, dynamic_valid = build_dynamic_features(
        raw_grid,
        observed,
        data_ready,
        sigma_by_day,
        is_rate=is_rate,
        first_observed_open=early_open_cutoff is not None,
    )
    return InstrumentFeatures(
        dynamic=dynamic,
        dynamic_valid=dynamic_valid,
        slow=slow,
        slow_rank_valid=slow_rank_valid,
        sigma=sigma_by_day,
        data_ready=data_ready,
        daily_change=summaries["change"],
        daily_change_valid=summaries["change_valid"],
    )


def build_liquidity_selected_rate_features(
    raw_grid: NDArray[np.float64],
    observed: NDArray[np.bool_],
    valid_day: NDArray[np.bool_],
    *,
    market_dates: tuple[date, ...] | None = None,
) -> InstrumentFeatures:
    """Build DI1$N features from independent B3-session price paths.

    Each session is translated to a first-observed-open level of 10 before shared
    causal transforms run. Within-session differences are unchanged, while no
    absolute level or prior-session close can enter a later session.
    """
    session_local_grid = raw_grid.copy()
    for date_idx in range(raw_grid.shape[0]):
        positions = np.flatnonzero(observed[date_idx])
        if positions.size == 0:
            continue
        level_offset = 10.0 - raw_grid[date_idx, positions[0], 0]
        session_local_grid[date_idx, positions, :4] += level_offset

    base = build_causal_features(
        session_local_grid,
        observed,
        valid_day,
        is_rate=True,
        market_dates=market_dates,
        include_dollar_volume=False,
    )
    slow = base.slow.copy()
    slow[:, LIQUIDITY_SELECTED_RATE_ZERO_SLOW_CHANNEL_INDICES] = 0.0
    slow[:, 7:11] = 0.0
    summaries = _daily_summaries(session_local_grid, observed, valid_day, is_rate=True)
    for date_idx in np.flatnonzero(base.data_ready):
        _populate_session_local_trailing_slow(
            slow[int(date_idx)],
            summaries,
            float(base.sigma[date_idx]),
            int(date_idx),
            raw_grid.shape[1],
        )
    return InstrumentFeatures(
        dynamic=base.dynamic,
        dynamic_valid=base.dynamic_valid,
        slow=slow,
        slow_rank_valid=np.zeros_like(base.slow_rank_valid),
        sigma=base.sigma,
        data_ready=base.data_ready,
        daily_change=np.zeros_like(base.daily_change),
        daily_change_valid=np.zeros_like(base.daily_change_valid),
    )


def build_dynamic_features(
    raw_grid: NDArray[np.float64],
    observed: NDArray[np.bool_],
    valid_day: NDArray[np.bool_],
    sigma_by_day: NDArray[np.float64],
    *,
    is_rate: bool,
    mapping_changed: NDArray[np.bool_] | None = None,
    first_observed_open: bool = False,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Build causal minute features from volatility state fixed before each day."""
    date_count, minute_count, field_count = raw_grid.shape
    if (
        field_count != 5
        or observed.shape != (date_count, minute_count)
        or valid_day.shape != (date_count,)
        or sigma_by_day.shape != (date_count,)
    ):
        raise ValueError(
            "Dynamic feature inputs do not share the fixed date/minute axes"
        )
    if mapping_changed is None:
        mapping_changed = np.zeros_like(observed)
    if mapping_changed.shape != observed.shape:
        raise ValueError("mapping_changed must align to the minute grid")
    dynamic = np.zeros((date_count, minute_count, 26), dtype=np.float32)
    validity = np.zeros((date_count, minute_count, 4), dtype=bool)
    for date_idx in np.flatnonzero(valid_day & (sigma_by_day > 0)):
        _build_dynamic_day(
            dynamic[date_idx],
            validity[date_idx],
            raw_grid,
            observed,
            valid_day,
            mapping_changed,
            int(date_idx),
            float(sigma_by_day[date_idx]),
            is_rate=is_rate,
            first_observed_open=first_observed_open,
        )
    return dynamic, validity


def _build_dynamic_day(
    output: NDArray[np.float32],
    validity: NDArray[np.bool_],
    raw_grid: NDArray[np.float64],
    observed: NDArray[np.bool_],
    valid_day: NDArray[np.bool_],
    mapping_changed: NDArray[np.bool_],
    date_idx: int,
    sigma: float,
    *,
    is_rate: bool,
    first_observed_open: bool,
) -> None:
    current_raw = raw_grid[date_idx]
    current_observed = observed[date_idx]
    positions = np.flatnonzero(current_observed)
    if positions.size == 0:
        return

    current_mapping_changed = mapping_changed[date_idx]
    prices = current_raw[positions, :4]
    anchors = np.empty(positions.size, dtype=np.float64)
    anchors[0] = prices[0, 0]
    anchors[1:] = current_raw[positions[:-1], 3]
    moves = _price_change(prices, anchors[:, None], is_rate=is_rate)
    move_valid = ~current_mapping_changed[positions]
    output[positions[move_valid], :4] = np.clip(
        moves[move_valid] / sigma,
        -PRICE_FEATURE_CLIP,
        PRICE_FEATURE_CLIP,
    ).astype(np.float32)

    first_history_date = max(0, date_idx - VOLUME_LOOKBACK_SESSIONS)
    prior_observed = observed[first_history_date:date_idx, positions]
    counts = prior_observed.sum(axis=0)
    baseline_ready = counts >= VOLUME_MIN_OBSERVATIONS
    if baseline_ready.any():
        prior_volume = raw_grid[first_history_date:date_idx, positions, 4]
        prior_logs = np.full(prior_volume.shape, np.nan, dtype=np.float64)
        np.log(prior_volume, out=prior_logs, where=prior_observed)
        usable = prior_logs[:, baseline_ready]
        median = np.nanmedian(usable, axis=0)
        mad = np.nanmedian(np.abs(usable - median), axis=0)
        scale = np.maximum(MAD_NORMALIZATION * mad, VOLUME_MAD_FLOOR)
        current_log_volume = np.log(current_raw[positions[baseline_ready], 4])
        surprise = np.clip(
            (current_log_volume - median) / scale,
            -VOLUME_FEATURE_CLIP,
            VOLUME_FEATURE_CLIP,
        )
        ready_positions = positions[baseline_ready]
        output[ready_positions, 4] = surprise.astype(np.float32)
        validity[ready_positions, 2] = True
    output[positions, 5] = 1.0

    roll_prefix = np.concatenate(([0], np.cumsum(current_mapping_changed)))
    if first_observed_open:
        open_position = int(positions[0])
        elapsed = positions - open_position + 1
    else:
        open_position = 0
        elapsed = positions + 1
    if current_observed[open_position]:
        open_price = current_raw[open_position, 0]
        same_contract = roll_prefix[positions + 1] == roll_prefix[open_position + 1]
        usable_positions = positions[same_contract]
        since_open = _price_change(
            current_raw[usable_positions, 3], open_price, is_rate=is_rate
        )
        output[usable_positions, 6] = np.clip(
            since_open / (sigma * np.sqrt(elapsed[same_contract])),
            -PRICE_FEATURE_CLIP,
            PRICE_FEATURE_CLIP,
        ).astype(np.float32)

    for channel, window in zip((7, 8, 9), RETURN_WINDOWS, strict=True):
        endpoints = positions - window
        exact = endpoints >= 0
        exact[exact] &= current_observed[endpoints[exact]]
        exact[exact] &= (
            roll_prefix[positions[exact] + 1] == roll_prefix[endpoints[exact] + 1]
        )
        if exact.any():
            usable_positions = positions[exact]
            returns = _price_change(
                current_raw[usable_positions, 3],
                current_raw[endpoints[exact], 3],
                is_rate=is_rate,
            )
            output[usable_positions, channel] = np.clip(
                returns / (sigma * np.sqrt(window)),
                -PRICE_FEATURE_CLIP,
                PRICE_FEATURE_CLIP,
            ).astype(np.float32)
            if window == 15:
                validity[usable_positions, 0] = True
            elif window == 60:
                validity[usable_positions, 1] = True

    adjacent = (
        current_observed[1:] & current_observed[:-1] & ~current_mapping_changed[1:]
    )
    one_minute = np.zeros(current_raw.shape[0] - 1, dtype=np.float64)
    if adjacent.any():
        one_minute[adjacent] = _price_change(
            current_raw[1:, 3][adjacent],
            current_raw[:-1, 3][adjacent],
            is_rate=is_rate,
        )
    for channel, window in zip((10, 11, 12), RETURN_WINDOWS, strict=True):
        minimum = int(np.ceil(REALIZED_VOL_MIN_FRACTION * window))
        for minute_idx in positions[positions >= window]:
            window_valid = adjacent[minute_idx - window : minute_idx]
            if int(window_valid.sum()) < minimum:
                continue
            values = one_minute[minute_idx - window : minute_idx][window_valid]
            rms = np.sqrt(np.mean(values**2))
            output[minute_idx, channel] = np.float32(
                np.clip(
                    np.log(max(rms / sigma, REALIZED_VOL_LOG_FLOOR)),
                    -REALIZED_VOL_LOG_CLIP,
                    REALIZED_VOL_LOG_CLIP,
                )
            )
            if window == 30:
                validity[minute_idx, 3] = True

    cumulative = np.cumsum(np.where(current_observed, current_raw[:, 4], 0.0))
    history_indices = np.flatnonzero(valid_day[:date_idx])[-VOLUME_LOOKBACK_SESSIONS:]
    if history_indices.size >= VOLUME_MIN_OBSERVATIONS:
        history_cumulative = np.cumsum(
            np.where(
                observed[history_indices],
                raw_grid[history_indices, :, 4],
                0.0,
            ),
            axis=1,
        )
        history_logs = np.log1p(history_cumulative)
        for minute_idx in positions:
            usable = history_logs[:, minute_idx]
            if usable.size < VOLUME_MIN_OBSERVATIONS:
                continue
            median = np.median(usable)
            scale = max(
                MAD_NORMALIZATION * np.median(np.abs(usable - median)),
                VOLUME_MAD_FLOOR,
            )
            output[minute_idx, 13] = np.float32(
                np.clip(
                    (np.log1p(cumulative[minute_idx]) - median) / scale,
                    -VOLUME_FEATURE_CLIP,
                    VOLUME_FEATURE_CLIP,
                )
            )

    running_high = np.full(current_raw.shape[0], -np.inf)
    running_low = np.full(current_raw.shape[0], np.inf)
    high = -np.inf
    low = np.inf
    for minute_idx in positions:
        if current_mapping_changed[minute_idx]:
            high = -np.inf
            low = np.inf
        high = max(high, current_raw[minute_idx, 1])
        low = min(low, current_raw[minute_idx, 2])
        running_high[minute_idx] = high
        running_low[minute_idx] = low
    ranges = running_high[positions] - running_low[positions]
    usable_range = ranges > 0
    usable_positions = positions[usable_range]
    output[usable_positions, 14] = np.clip(
        2.0
        * (current_raw[usable_positions, 3] - running_low[usable_positions])
        / ranges[usable_range]
        - 1.0,
        -1.0,
        1.0,
    ).astype(np.float32)

    cumulative_observed = np.concatenate(([0], np.cumsum(current_observed)))
    for minute_idx in range(current_raw.shape[0]):
        start = max(0, minute_idx + 1 - OBSERVED_FRACTION_WINDOW)
        denominator = minute_idx + 1 - start
        output[minute_idx, 15] = np.float32(
            (cumulative_observed[minute_idx + 1] - cumulative_observed[start])
            / denominator
        )


def _daily_summaries(
    raw_grid: NDArray[np.float64],
    observed: NDArray[np.bool_],
    valid_day: NDArray[np.bool_],
    *,
    is_rate: bool,
    early_open_cutoff: int | None = None,
) -> dict[str, NDArray[np.generic]]:
    date_count, minute_count, _ = raw_grid.shape
    output: dict[str, NDArray[np.generic]] = {
        "open": np.zeros(date_count, dtype=np.float64),
        "open_valid": np.zeros(date_count, dtype=bool),
        "early_open": np.zeros(date_count, dtype=np.float64),
        "early_open_valid": np.zeros(date_count, dtype=bool),
        "close": np.zeros(date_count, dtype=np.float64),
        "close_valid": np.zeros(date_count, dtype=bool),
        "change": np.zeros(date_count, dtype=np.float64),
        "change_valid": np.zeros(date_count, dtype=bool),
        "open_close": np.zeros(date_count, dtype=np.float64),
        "open_close_valid": np.zeros(date_count, dtype=bool),
        "last_60": np.zeros(date_count, dtype=np.float64),
        "last_60_valid": np.zeros(date_count, dtype=bool),
        "rv": np.zeros(date_count, dtype=np.float64),
        "rv_valid": np.zeros(date_count, dtype=bool),
        "volume": np.zeros(date_count, dtype=np.float64),
        "volume_valid": np.zeros(date_count, dtype=bool),
        "dollar_volume": np.zeros(date_count, dtype=np.float64),
        "dollar_volume_valid": np.zeros(date_count, dtype=bool),
        "observed_fraction": observed.mean(axis=1, dtype=np.float64),
    }
    previous_close: float | None = None
    for date_idx in range(date_count):
        if not valid_day[date_idx]:
            continue
        day_observed = observed[date_idx]
        positions = np.flatnonzero(day_observed)
        if positions.size == 0:
            continue
        day = raw_grid[date_idx]
        if early_open_cutoff is None:
            if day_observed[0]:
                output["open"][date_idx] = day[0, 0]
                output["open_valid"][date_idx] = True
                output["early_open"][date_idx] = day[0, 0]
                output["early_open_valid"][date_idx] = True
        else:
            session_open, session_open_valid = _completed_session_open(
                day, day_observed
            )
            output["open"][date_idx] = session_open
            output["open_valid"][date_idx] = session_open_valid
            early_open, early_open_valid = _early_open_before_cutoff(
                day, day_observed, early_open_cutoff
            )
            output["early_open"][date_idx] = early_open
            output["early_open_valid"][date_idx] = early_open_valid
        final_idx = int(positions[-1])
        final_close = float(day[final_idx, 3])
        output["close"][date_idx] = final_close
        output["close_valid"][date_idx] = True
        if previous_close is not None:
            output["change"][date_idx] = _scalar_price_change(
                final_close, previous_close, is_rate=is_rate
            )
            output["change_valid"][date_idx] = True
        previous_close = final_close

        if output["open_valid"][date_idx]:
            output["open_close"][date_idx] = _scalar_price_change(
                final_close, float(output["open"][date_idx]), is_rate=is_rate
            )
            output["open_close_valid"][date_idx] = True
        boundary = final_idx - 60
        if boundary >= 0 and day_observed[boundary]:
            output["last_60"][date_idx] = _scalar_price_change(
                final_close, float(day[boundary, 3]), is_rate=is_rate
            )
            output["last_60_valid"][date_idx] = True
        variance = _daily_variance(day[:, 3], day_observed, is_rate=is_rate)
        if variance is not None:
            output["rv"][date_idx] = np.sqrt(variance)
            output["rv_valid"][date_idx] = True
        volume = float(day[positions, 4].sum())
        if np.isfinite(volume) and volume > 0:
            output["volume"][date_idx] = volume
            output["volume_valid"][date_idx] = True
        dollar_volume = float(np.sum(day[positions, 3] * day[positions, 4]))
        if np.isfinite(dollar_volume) and dollar_volume > 0:
            output["dollar_volume"][date_idx] = dollar_volume
            output["dollar_volume_valid"][date_idx] = True
    return output


def _completed_session_open(
    day: NDArray[np.float64], day_observed: NDArray[np.bool_]
) -> tuple[float, bool]:
    positions = np.flatnonzero(day_observed)
    if positions.size == 0:
        return 0.0, False
    return float(day[int(positions[0]), 0]), True


def _early_open_before_cutoff(
    day: NDArray[np.float64],
    day_observed: NDArray[np.bool_],
    cutoff: int,
) -> tuple[float, bool]:
    positions = np.flatnonzero(day_observed[:cutoff])
    if positions.size == 0:
        return 0.0, False
    return float(day[int(positions[0]), 0]), True


def _populate_session_local_trailing_slow(
    output: NDArray[np.float32],
    summaries: dict[str, NDArray[np.generic]],
    sigma: float,
    date_idx: int,
    minute_count: int,
) -> None:
    daily_scale = sigma * np.sqrt(minute_count)
    direction_indices = np.flatnonzero(
        summaries["open_close_valid"][:date_idx].astype(bool)
    )
    rv_indices = np.flatnonzero(summaries["rv_valid"][:date_idx].astype(bool))
    for channel_return, channel_rv, window, minimum in (
        (7, 9, SLOW_SHORT_WINDOW, SLOW_SHORT_MIN_VALID),
        (8, 10, SLOW_LONG_WINDOW, SLOW_LONG_MIN_VALID),
    ):
        indices = direction_indices[-window:]
        if indices.size >= minimum:
            changes = summaries["open_close"][indices].astype(np.float64)
            output[channel_return] = np.float32(
                np.clip(
                    changes.sum() / (daily_scale * np.sqrt(indices.size)),
                    -PRICE_FEATURE_CLIP,
                    PRICE_FEATURE_CLIP,
                )
            )
        indices = rv_indices[-window:]
        if indices.size < minimum:
            continue
        intraday_rv = summaries["rv"][indices].astype(np.float64)
        daily_rv_ratio = np.sqrt(np.mean(intraday_rv**2)) / sigma
        output[channel_rv] = np.float32(
            np.clip(
                np.log(max(daily_rv_ratio, REALIZED_VOL_LOG_FLOOR)),
                -REALIZED_VOL_LOG_CLIP,
                REALIZED_VOL_LOG_CLIP,
            )
        )


def _build_slow_day(
    output: NDArray[np.float32],
    rank_valid: NDArray[np.bool_],
    summaries: dict[str, NDArray[np.generic]],
    sigma_by_day: NDArray[np.float64],
    valid_day: NDArray[np.bool_],
    date_idx: int,
    sigma: float,
    minute_count: int,
    market_date: date,
    *,
    is_rate: bool,
    include_dollar_volume: bool,
) -> None:
    completed = np.flatnonzero(
        valid_day[:date_idx] & summaries["close_valid"][:date_idx].astype(bool)
    )
    daily_scale = sigma * np.sqrt(minute_count)
    previous_idx = int(completed[-1]) if completed.size else None

    if previous_idx is not None and bool(summaries["early_open_valid"][date_idx]):
        gap = _scalar_price_change(
            float(summaries["early_open"][date_idx]),
            float(summaries["close"][previous_idx]),
            is_rate=is_rate,
        )
        output[1] = np.float32(
            np.clip(gap / daily_scale, -PRICE_FEATURE_CLIP, PRICE_FEATURE_CLIP)
        )
        rank_valid[0] = True

    if previous_idx is not None:
        if bool(summaries["change_valid"][previous_idx]):
            output[2] = np.float32(
                np.clip(
                    float(summaries["change"][previous_idx]) / daily_scale,
                    -PRICE_FEATURE_CLIP,
                    PRICE_FEATURE_CLIP,
                )
            )
        if bool(summaries["open_close_valid"][previous_idx]):
            output[3] = np.float32(
                np.clip(
                    float(summaries["open_close"][previous_idx]) / daily_scale,
                    -PRICE_FEATURE_CLIP,
                    PRICE_FEATURE_CLIP,
                )
            )
        if bool(summaries["last_60_valid"][previous_idx]):
            output[4] = np.float32(
                np.clip(
                    float(summaries["last_60"][previous_idx]) / (sigma * np.sqrt(60)),
                    -PRICE_FEATURE_CLIP,
                    PRICE_FEATURE_CLIP,
                )
            )
        prior_sigma = sigma_by_day[previous_idx]
        if bool(summaries["rv_valid"][previous_idx]) and prior_sigma > 0:
            output[5] = np.float32(
                np.clip(
                    np.log(
                        max(
                            float(summaries["rv"][previous_idx]) / prior_sigma,
                            REALIZED_VOL_LOG_FLOOR,
                        )
                    ),
                    -REALIZED_VOL_LOG_CLIP,
                    REALIZED_VOL_LOG_CLIP,
                )
            )
        output[6] = np.float32(
            _prior_session_robust_ratio(
                summaries["volume"],
                summaries["volume_valid"].astype(bool),
                previous_idx,
            )
        )

    change_indices = np.flatnonzero(summaries["change_valid"][:date_idx].astype(bool))
    for channel_return, channel_rv, window, minimum in (
        (7, 9, SLOW_SHORT_WINDOW, SLOW_SHORT_MIN_VALID),
        (8, 10, SLOW_LONG_WINDOW, SLOW_LONG_MIN_VALID),
    ):
        indices = change_indices[-window:]
        if indices.size < minimum:
            continue
        changes = summaries["change"][indices].astype(np.float64)
        output[channel_return] = np.float32(
            np.clip(
                changes.sum() / (daily_scale * np.sqrt(indices.size)),
                -PRICE_FEATURE_CLIP,
                PRICE_FEATURE_CLIP,
            )
        )
        daily_rv_ratio = np.sqrt(np.mean(changes**2)) / daily_scale
        output[channel_rv] = np.float32(
            np.clip(
                np.log(max(daily_rv_ratio, REALIZED_VOL_LOG_FLOOR)),
                -REALIZED_VOL_LOG_CLIP,
                REALIZED_VOL_LOG_CLIP,
            )
        )
        if window == SLOW_LONG_WINDOW:
            rank_valid[2] = True

    rv_ratio_values: list[float] = []
    for index in np.flatnonzero(summaries["rv_valid"][:date_idx].astype(bool))[-20:]:
        historical_sigma = sigma_by_day[index]
        if historical_sigma > 0:
            rv_ratio_values.append(
                np.log(
                    max(
                        float(summaries["rv"][index]) / historical_sigma,
                        REALIZED_VOL_LOG_FLOOR,
                    )
                )
            )
    if len(rv_ratio_values) >= SLOW_LONG_MIN_VALID:
        output[11] = np.float32(
            np.clip(
                np.std(rv_ratio_values),
                0.0,
                REALIZED_VOL_LOG_CLIP,
            )
        )

    volume_indices = np.flatnonzero(summaries["volume_valid"][:date_idx].astype(bool))[
        -VOLUME_LOOKBACK_SESSIONS:
    ]
    if volume_indices.size >= VOLUME_MIN_OBSERVATIONS:
        median_volume = np.median(summaries["volume"][volume_indices])
        output[12] = np.float32(
            (np.log1p(median_volume) - REAL_VOLUME_LOG_CENTER) / REAL_VOLUME_LOG_SCALE
        )

    if include_dollar_volume:
        dollar_indices = np.flatnonzero(
            summaries["dollar_volume_valid"][:date_idx].astype(bool)
        )[-VOLUME_LOOKBACK_SESSIONS:]
        if dollar_indices.size >= VOLUME_MIN_OBSERVATIONS:
            median_dollar = np.median(summaries["dollar_volume"][dollar_indices])
            output[13] = np.float32(
                (np.log1p(median_dollar) - DOLLAR_VOLUME_LOG_CENTER)
                / DOLLAR_VOLUME_LOG_SCALE
            )
            rank_valid[1] = True
        if previous_idx is not None:
            output[14] = np.float32(
                _prior_session_robust_ratio(
                    summaries["dollar_volume"],
                    summaries["dollar_volume_valid"].astype(bool),
                    previous_idx,
                )
            )

    source_sessions = np.flatnonzero(valid_day[:date_idx])
    for channel, window, minimum in (
        (15, SLOW_SHORT_WINDOW, SLOW_SHORT_MIN_VALID),
        (16, SLOW_LONG_WINDOW, SLOW_LONG_MIN_VALID),
    ):
        indices = source_sessions[-window:]
        if indices.size >= minimum:
            output[channel] = np.float32(
                np.mean(summaries["observed_fraction"][indices])
            )

    weekday = market_date.weekday()
    output[26] = np.float32(np.sin(2.0 * np.pi * weekday / 5.0))
    output[27] = np.float32(np.cos(2.0 * np.pi * weekday / 5.0))
    month_end = date(
        market_date.year + (market_date.month == 12),
        market_date.month % 12 + 1,
        1,
    )
    days_to_month_end = (month_end - market_date).days - 1
    output[28] = np.float32(max(0.0, 1.0 - days_to_month_end / 7.0))
    quarter_month = 3 * ((market_date.month - 1) // 3 + 1)
    if quarter_month == 12:
        next_quarter = date(market_date.year + 1, 1, 1)
    else:
        next_quarter = date(market_date.year, quarter_month + 1, 1)
    days_to_quarter_end = (next_quarter - market_date).days - 1
    output[29] = np.float32(max(0.0, 1.0 - days_to_quarter_end / 14.0))


def _prior_session_robust_ratio(
    values: NDArray[np.generic],
    valid: NDArray[np.bool_],
    previous_idx: int,
) -> float:
    if not valid[previous_idx]:
        return 0.0
    history = np.flatnonzero(valid[:previous_idx])[-VOLUME_LOOKBACK_SESSIONS:]
    if history.size < VOLUME_MIN_OBSERVATIONS:
        return 0.0
    logs = np.log1p(values[history].astype(np.float64))
    median = np.median(logs)
    scale = max(
        MAD_NORMALIZATION * np.median(np.abs(logs - median)),
        VOLUME_MAD_FLOOR,
    )
    return float(
        np.clip(
            (np.log1p(float(values[previous_idx])) - median) / scale,
            -VOLUME_FEATURE_CLIP,
            VOLUME_FEATURE_CLIP,
        )
    )


def add_equity_cross_sectional_dynamic(
    dynamic: NDArray[np.float32],
    dynamic_valid: NDArray[np.bool_],
    active: NDArray[np.bool_],
) -> None:
    """Populate channels 16-25 in-place for one equity date."""
    if dynamic.ndim != 3 or dynamic.shape[2] != 26:
        raise ValueError("dynamic must have shape [equity, minute, 26]")
    if dynamic_valid.shape != (*dynamic.shape[:2], 4):
        raise ValueError("dynamic_valid has the wrong shape")
    active_slots = np.flatnonzero(active)
    if active_slots.size < MIN_ACTIVE_EQUITIES:
        return

    measurement_specs = (
        (7, 0, (16, 18, 20, 22)),
        (9, 1, (17, 19, 21, 23)),
        (4, 2, (None, None, None, 24)),
        (11, 3, (None, None, None, 25)),
    )
    for source_channel, valid_channel, destinations in measurement_specs:
        for minute_idx in range(dynamic.shape[1]):
            valid = active & dynamic_valid[:, minute_idx, valid_channel]
            valid_slots = np.flatnonzero(valid)
            if valid_slots.size < MIN_ACTIVE_EQUITIES:
                continue
            values = dynamic[valid_slots, minute_idx, source_channel].astype(np.float64)
            rank_channel = destinations[3]
            if rank_channel is not None:
                dynamic[valid_slots, minute_idx, rank_channel] = centered_midranks(
                    values
                )
            if destinations[0] is None:
                continue

            leave_one_out_median = _leave_one_out_medians(values)
            leave_one_out_mad = np.empty(values.size, dtype=np.float64)
            for median in np.unique(leave_one_out_median):
                group = leave_one_out_median == median
                distance_medians = _leave_one_out_medians(np.abs(values - median))
                leave_one_out_mad[group] = distance_medians[group]
            positions = np.full(dynamic.shape[0], -1, dtype=np.int32)
            positions[valid_slots] = np.arange(valid_slots.size, dtype=np.int32)
            focal_positions = positions[active_slots]
            focal_valid = focal_positions >= 0
            aggregate_median = np.full(active_slots.size, np.median(values))
            aggregate_breadth = np.full(
                active_slots.size, 2.0 * np.mean(values > 0.0) - 1.0
            )
            aggregate_dispersion = np.full(
                active_slots.size,
                MAD_NORMALIZATION * np.median(np.abs(values - np.median(values))),
            )
            if focal_valid.any():
                selected = focal_positions[focal_valid]
                aggregate_median[focal_valid] = leave_one_out_median[selected]
                positive_count = int(np.count_nonzero(values > 0.0))
                aggregate_breadth[focal_valid] = (
                    2.0
                    * ((positive_count - (values[selected] > 0.0)) / (values.size - 1))
                    - 1.0
                )
                aggregate_dispersion[focal_valid] = (
                    MAD_NORMALIZATION * leave_one_out_mad[selected]
                )
            dynamic[active_slots, minute_idx, destinations[0]] = aggregate_median
            dynamic[active_slots, minute_idx, destinations[1]] = aggregate_breadth
            dynamic[active_slots, minute_idx, destinations[2]] = np.clip(
                aggregate_dispersion, 0.0, PRICE_FEATURE_CLIP
            )


def _leave_one_out_medians(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the median after removing each corresponding value."""
    size = values.size
    if size < 2:
        raise ValueError("Leave-one-out medians require at least two values")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    rank = np.empty(size, dtype=np.int32)
    rank[order] = np.arange(size, dtype=np.int32)
    output = np.empty(size, dtype=np.float64)
    if size % 2 == 0:
        upper = size // 2
        output[rank < upper] = sorted_values[upper]
        output[rank >= upper] = sorted_values[upper - 1]
        return output
    middle = size // 2
    below = rank < middle
    at = rank == middle
    above = rank > middle
    output[below] = 0.5 * (sorted_values[middle] + sorted_values[middle + 1])
    output[at] = 0.5 * (sorted_values[middle - 1] + sorted_values[middle + 1])
    output[above] = 0.5 * (sorted_values[middle - 1] + sorted_values[middle])
    return output


def add_slow_cross_sectional_ranks(
    slow: NDArray[np.float32],
    slow_rank_valid: NDArray[np.bool_],
    active: NDArray[np.bool_],
) -> None:
    """Populate equity slow channels 17-19 in-place for one date."""
    for source_channel, valid_channel, destination in (
        (1, 0, 17),
        (13, 1, 18),
        (10, 2, 19),
    ):
        valid = active & slow_rank_valid[:, valid_channel]
        slots = np.flatnonzero(valid)
        if slots.size >= MIN_ACTIVE_EQUITIES:
            slow[slots, destination] = centered_midranks(
                slow[slots, source_channel].astype(np.float64)
            )


def causal_exposure_betas(
    equity_change: NDArray[np.float64],
    equity_valid: NDArray[np.bool_],
    context_change: NDArray[np.float64],
    context_valid: NDArray[np.bool_],
) -> NDArray[np.float32]:
    """Return betas whose row D is based only on paired sessions before D."""
    date_count, equity_count = equity_change.shape
    if equity_valid.shape != equity_change.shape:
        raise ValueError("equity validity must align to changes")
    if (
        context_change.shape[0] != date_count
        or context_valid.shape != context_change.shape
    ):
        raise ValueError("context changes and validity must align to dates")
    context_count = context_change.shape[1]
    output = np.zeros((date_count, equity_count, context_count), dtype=np.float32)
    count = np.zeros((equity_count, context_count), dtype=np.int32)
    mean_x = np.zeros((equity_count, context_count), dtype=np.float64)
    mean_y = np.zeros((equity_count, context_count), dtype=np.float64)
    mean_xy = np.zeros((equity_count, context_count), dtype=np.float64)
    mean_y2 = np.zeros((equity_count, context_count), dtype=np.float64)

    for date_idx in range(date_count):
        ready = count >= BETA_MIN_PAIRED_SESSIONS
        variance = np.maximum(mean_y2 - mean_y**2, BETA_VARIANCE_FLOOR)
        covariance = mean_xy - mean_x * mean_y
        output[date_idx, ready] = np.clip(
            covariance[ready] / variance[ready], -BETA_CLIP, BETA_CLIP
        ).astype(np.float32)

        paired = equity_valid[date_idx, :, None] & context_valid[date_idx, None, :]
        if not paired.any():
            continue
        x = np.broadcast_to(equity_change[date_idx, :, None], paired.shape)
        y = np.broadcast_to(context_change[date_idx, None, :], paired.shape)
        first = paired & (count == 0)
        continuing = paired & ~first
        mean_x[first] = x[first]
        mean_y[first] = y[first]
        mean_xy[first] = x[first] * y[first]
        mean_y2[first] = y[first] ** 2
        mean_x[continuing] = (1.0 - BETA_EWMA_ALPHA) * mean_x[
            continuing
        ] + BETA_EWMA_ALPHA * x[continuing]
        mean_y[continuing] = (1.0 - BETA_EWMA_ALPHA) * mean_y[
            continuing
        ] + BETA_EWMA_ALPHA * y[continuing]
        mean_xy[continuing] = (1.0 - BETA_EWMA_ALPHA) * mean_xy[
            continuing
        ] + BETA_EWMA_ALPHA * x[continuing] * y[continuing]
        mean_y2[continuing] = (1.0 - BETA_EWMA_ALPHA) * mean_y2[
            continuing
        ] + BETA_EWMA_ALPHA * y[continuing] ** 2
        count[paired] += 1
    return output


def centered_midranks(values: NDArray[np.floating]) -> NDArray[np.float32]:
    """Map ascending average midranks to the open interval (-1, 1)."""
    values = np.asarray(values)
    size = values.size
    if size == 0:
        return np.empty(0, dtype=np.float32)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(size, dtype=np.float64)
    start = 0
    while start < size:
        end = start + 1
        while end < size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    return (2.0 * ((ranks - 0.5) / size) - 1.0).astype(np.float32)


def _daily_variance(
    close: NDArray[np.float64], observed: NDArray[np.bool_], *, is_rate: bool
) -> float | None:
    adjacent = observed[1:] & observed[:-1]
    if int(adjacent.sum()) < MIN_ADJACENT_RETURNS_PER_DAY:
        return None
    current = close[1:][adjacent]
    previous = close[:-1][adjacent]
    returns = _price_change(current, previous, is_rate=is_rate)
    return float(np.mean(returns**2))


def rate_change_basis_points(
    current: NDArray[np.float64] | float,
    previous: NDArray[np.float64] | float,
) -> NDArray[np.float64]:
    """Convert annual percentage-rate level changes to basis points."""
    return 100.0 * (np.asarray(current, dtype=np.float64) - previous)


def _price_change(
    current: NDArray[np.float64],
    previous: NDArray[np.float64] | float,
    *,
    is_rate: bool,
) -> NDArray[np.float64]:
    return (
        rate_change_basis_points(current, previous)
        if is_rate
        else np.log(current / previous)
    )


def _scalar_price_change(current: float, previous: float, *, is_rate: bool) -> float:
    if is_rate:
        return float(rate_change_basis_points(current, previous))
    return float(np.log(current / previous))


def build_daily_changes(
    daily_close: NDArray[np.float64],
    observed: NDArray[np.bool_],
    *,
    is_rate: bool,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Build completed-session changes using the previous observed source close."""
    changes = np.zeros(daily_close.shape, dtype=np.float64)
    valid = np.zeros(daily_close.shape, dtype=bool)
    previous_close: float | None = None
    for date_idx in range(daily_close.shape[0]):
        if not observed[date_idx]:
            continue
        current_close = float(daily_close[date_idx])
        if previous_close is not None:
            changes[date_idx] = _scalar_price_change(
                current_close,
                previous_close,
                is_rate=is_rate,
            )
            valid[date_idx] = True
        previous_close = current_close
    return changes, valid


def build_prior_rate_level(
    daily_close: NDArray[np.float64], observed: NDArray[np.bool_]
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    date_count = daily_close.shape[0]
    scaled = np.zeros(date_count, dtype=np.float32)
    ready = np.zeros(date_count, dtype=bool)
    prior_close: float | None = None
    for date_idx in range(date_count):
        if prior_close is not None:
            scaled[date_idx] = np.float32(np.clip(prior_close / 10.0, -1.0, 3.0))
            ready[date_idx] = True
        if observed[date_idx]:
            prior_close = float(daily_close[date_idx])
    return scaled, ready


def time_to_expiry_scaled(
    market_dates: tuple[date, ...], expiry_date: date
) -> NDArray[np.float32]:
    days = np.array(
        [max((expiry_date - market_date).days, 0) for market_date in market_dates],
        dtype=np.float64,
    )
    return np.clip(np.log1p(days / 365.25) / np.log(11.0), 0.0, 1.0).astype(np.float32)


def build_raw_returns(
    raw_grid: NDArray[np.float64], observed: NDArray[np.bool_]
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    date_count = raw_grid.shape[0]
    raw_returns = np.zeros(
        (date_count, len(DECISION_EQUITY_INDICES), len(HORIZONS)),
        dtype=np.float32,
    )
    endpoint_mask = np.zeros(raw_returns.shape, dtype=bool)
    for decision_idx, entry_index in enumerate(DECISION_EQUITY_INDICES):
        entry_observed = observed[:, entry_index]
        entry_open = raw_grid[:, entry_index, 0]
        for horizon_idx, horizon in enumerate(HORIZONS):
            exit_index = entry_index + horizon - 1
            valid = entry_observed & observed[:, exit_index]
            endpoint_mask[:, decision_idx, horizon_idx] = valid
            raw_returns[valid, decision_idx, horizon_idx] = np.log(
                raw_grid[valid, exit_index, 3] / entry_open[valid]
            ).astype(np.float32)
    return raw_returns, endpoint_mask


def center_cross_section(
    raw_returns: NDArray[np.float32],
    candidate_mask: NDArray[np.bool_],
    sigma: NDArray[np.float64],
) -> tuple[
    NDArray[np.float32],
    NDArray[np.bool_],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.bool_],
]:
    """Build one date of masked raw returns and centered-rank targets."""
    masked_raw = np.where(candidate_mask, raw_returns, 0.0).astype(np.float32)
    targets = np.zeros(masked_raw.shape, dtype=np.float32)
    label_mask = np.zeros(masked_raw.shape, dtype=bool)
    medians = np.zeros(masked_raw.shape[1:], dtype=np.float32)
    horizon_mask = np.zeros(masked_raw.shape[1:], dtype=bool)

    for decision_idx in range(masked_raw.shape[1]):
        for horizon_idx, horizon in enumerate(HORIZONS):
            valid = candidate_mask[:, decision_idx, horizon_idx]
            if int(valid.sum()) < MIN_ACTIVE_EQUITIES:
                masked_raw[:, decision_idx, horizon_idx] = 0.0
                continue
            values = masked_raw[valid, decision_idx, horizon_idx].astype(np.float64)
            median = np.float32(np.median(values))
            standardized = (values - float(median)) / (sigma[valid] * np.sqrt(horizon))
            medians[decision_idx, horizon_idx] = median
            horizon_mask[decision_idx, horizon_idx] = True
            label_mask[valid, decision_idx, horizon_idx] = True
            targets[valid, decision_idx, horizon_idx] = centered_midranks(standardized)

    return masked_raw, label_mask, targets, medians, horizon_mask
