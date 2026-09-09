from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, time
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .contract import INTRADAY_DAILY_FEATURES
from .corporate_actions import AlignedActionTerms, VerifiedActionTerm
from .decision_clock import SessionDefinition


NATIVE_FAST_FEATURES = (
    "adjacent_endpoint_log_return_over_s5",
    "block_log_high_low_over_s5",
    "signed_close_location",
    "relative_same_clock_volume",
    "observed_fraction",
    "elapsed_fraction",
    "last_price_age_fraction",
)


@dataclass(frozen=True)
class NativeFastResult:
    """Native five-minute values and their independent availability masks."""

    values: NDArray[np.float32]
    valid: NDArray[np.bool_]
    patch_mask: NDArray[np.bool_]
    last_price_age_minutes: NDArray[np.float32]
    last_price_age_valid: NDArray[np.bool_]
    feature_names: tuple[str, ...] = NATIVE_FAST_FEATURES


def _clock_minutes(start: time, end: time, session: SessionDefinition) -> int:
    delta = datetime.combine(session.trade_date, end) - datetime.combine(
        session.trade_date, start
    )
    seconds = delta.total_seconds()
    if seconds <= 0.0 or seconds % 60.0:
        raise ValueError(
            f"session clocks must define positive whole minutes on {session.trade_date}"
        )
    return int(seconds // 60.0)


def _minute_of_day(value: time, session: SessionDefinition) -> int:
    if value.second or value.microsecond:
        raise ValueError(
            f"session clocks must use minute precision on {session.trade_date}"
        )
    return value.hour * 60 + value.minute


def _native_session_layout(
    sessions: Sequence[SessionDefinition],
    *,
    date_count: int,
    minute_count: int,
    block_minutes: int,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    if block_minutes != 5:
        raise ValueError("the native fast contract requires five-minute blocks")
    if date_count <= 0 or len(sessions) != date_count:
        raise ValueError("one SessionDefinition is required for each date row")
    prefix_minutes = np.empty(date_count, dtype=np.int64)
    continuous_minutes = np.empty(date_count, dtype=np.int64)
    open_minutes = np.empty(date_count, dtype=np.int64)
    patch_counts = np.empty(date_count, dtype=np.int64)
    for day, session in enumerate(sessions):
        open_minutes[day] = _minute_of_day(session.continuous_open, session)
        prefix_minutes[day] = _clock_minutes(
            session.continuous_open, session.decision_time, session
        )
        continuous_minutes[day] = _clock_minutes(
            session.continuous_open, session.continuous_close, session
        )
        if prefix_minutes[day] >= continuous_minutes[day]:
            raise ValueError(f"decision must precede the close on {session.trade_date}")
        patch_counts[day] = prefix_minutes[day] // block_minutes
    if np.any(patch_counts == 0):
        raise ValueError("every session prefix must contain a completed block")
    if np.any(prefix_minutes > minute_count):
        raise ValueError("M1 inputs do not reach every session decision")
    return prefix_minutes, continuous_minutes, open_minutes, patch_counts


def build_native_fast_features(
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    close: NDArray[np.floating],
    volume: NDArray[np.floating],
    observed: NDArray[np.bool_],
    *,
    volume_valid: NDArray[np.bool_],
    session_valid: NDArray[np.bool_],
    sigma_asof: NDArray[np.floating],
    sessions: Sequence[SessionDefinition],
    max_patches: int | None = None,
    block_minutes: int = 5,
    minimum_sigma: float = 1e-8,
) -> NativeFastResult:
    """Allocate and build the causal native within-day representation."""

    input_shape = np.shape(high)
    if len(input_shape) != 3:
        raise ValueError("native fast M1 arrays must align [date, name, minute]")
    _, _, _, patch_counts = _native_session_layout(
        sessions,
        date_count=input_shape[0],
        minute_count=input_shape[2],
        block_minutes=block_minutes,
    )
    required_patches = int(patch_counts.max())
    if max_patches is None:
        max_patches = required_patches
    if not isinstance(max_patches, (int, np.integer)) or isinstance(max_patches, bool):
        raise TypeError("max_patches must be an integer")
    max_patches = int(max_patches)
    if max_patches < required_patches:
        raise ValueError("max_patches cannot truncate a session prefix")

    dates, names, _ = input_shape
    patch_shape = (dates, names, max_patches)
    return build_native_fast_features_into(
        high,
        low,
        close,
        volume,
        observed,
        volume_valid=volume_valid,
        session_valid=session_valid,
        sigma_asof=sigma_asof,
        sessions=sessions,
        values_out=np.zeros(
            (*patch_shape, len(NATIVE_FAST_FEATURES)), dtype=np.float32
        ),
        valid_out=np.zeros((*patch_shape, len(NATIVE_FAST_FEATURES)), dtype=np.bool_),
        patch_mask_out=np.zeros(patch_shape, dtype=np.bool_),
        last_price_age_minutes_out=np.zeros(patch_shape, dtype=np.float32),
        last_price_age_valid_out=np.zeros(patch_shape, dtype=np.bool_),
        block_minutes=block_minutes,
        minimum_sigma=minimum_sigma,
    )


def build_native_fast_features_into(
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    close: NDArray[np.floating],
    volume: NDArray[np.floating],
    observed: NDArray[np.bool_],
    *,
    volume_valid: NDArray[np.bool_],
    session_valid: NDArray[np.bool_],
    sigma_asof: NDArray[np.floating],
    sessions: Sequence[SessionDefinition],
    values_out: NDArray[np.float32],
    valid_out: NDArray[np.bool_],
    patch_mask_out: NDArray[np.bool_],
    last_price_age_minutes_out: NDArray[np.float32],
    last_price_age_valid_out: NDArray[np.bool_],
    block_minutes: int = 5,
    minimum_sigma: float = 1e-8,
) -> NativeFastResult:
    """Stream native features into caller-owned arrays, including memmaps.

    M1 inputs are ``[date, name, minute]`` and minute zero is each date's
    declared continuous open. ``volume_valid`` certifies meaningful activity,
    including documented no-trade zeros; ``observed`` means an actual price
    bar exists. Only slices before the declared decision are accessed.
    """

    high_, low_, close_, volume_ = (
        np.asarray(value) for value in (high, low, close, volume)
    )
    seen = np.asarray(observed)
    activity_seen = np.asarray(volume_valid)
    supported = np.asarray(session_valid)
    sigma = np.asarray(sigma_asof)
    numeric_inputs = (high_, low_, close_, volume_)
    if high_.ndim != 3 or any(value.shape != high_.shape for value in numeric_inputs):
        raise ValueError("native fast M1 arrays must align [date, name, minute]")
    if any(not np.issubdtype(value.dtype, np.number) for value in numeric_inputs):
        raise TypeError("native fast market inputs must be numeric")
    if seen.dtype != np.bool_ or activity_seen.dtype != np.bool_:
        raise TypeError("observed and volume_valid must be boolean arrays")
    if seen.shape != high_.shape or activity_seen.shape != high_.shape:
        raise ValueError("native fast M1 masks must align [date, name, minute]")
    daily_shape = high_.shape[:2]
    if supported.dtype != np.bool_ or supported.shape != daily_shape:
        raise ValueError("session_valid must be boolean and align [date, name]")
    if np.any(seen & ~supported[..., None]) or np.any(
        activity_seen & ~supported[..., None]
    ):
        raise ValueError("native M1 observations cannot exist outside source support")
    if np.any(activity_seen & (~np.isfinite(volume_) | (volume_ < 0.0))):
        raise ValueError("valid native M1 activity must be finite and non-negative")
    if sigma.shape != daily_shape or not np.issubdtype(sigma.dtype, np.number):
        raise ValueError("sigma_asof must be numeric and align [date, name]")
    if not np.isfinite(minimum_sigma) or minimum_sigma <= 0.0:
        raise ValueError("minimum_sigma must be positive and finite")

    prefix_minutes, continuous_minutes, open_minutes, patch_counts = (
        _native_session_layout(
            sessions,
            date_count=high_.shape[0],
            minute_count=high_.shape[2],
            block_minutes=block_minutes,
        )
    )
    values = np.asarray(values_out)
    valid = np.asarray(valid_out)
    patch_mask = np.asarray(patch_mask_out)
    age_minutes = np.asarray(last_price_age_minutes_out)
    age_valid = np.asarray(last_price_age_valid_out)
    if values.ndim != 4 or values.shape[:2] != daily_shape:
        raise ValueError("values_out must align [date, name, patch, channel]")
    if values.shape[3] != len(NATIVE_FAST_FEATURES):
        raise ValueError("values_out has the wrong native channel count")
    patch_shape = values.shape[:3]
    if patch_shape[2] < int(patch_counts.max()):
        raise ValueError("output patch axis cannot truncate a session prefix")
    outputs = (
        (values, values.shape, np.dtype(np.float32), "values_out"),
        (valid, values.shape, np.dtype(np.bool_), "valid_out"),
        (patch_mask, patch_shape, np.dtype(np.bool_), "patch_mask_out"),
        (
            age_minutes,
            patch_shape,
            np.dtype(np.float32),
            "last_price_age_minutes_out",
        ),
        (
            age_valid,
            patch_shape,
            np.dtype(np.bool_),
            "last_price_age_valid_out",
        ),
    )
    for output, expected_shape, expected_dtype, name in outputs:
        if output.shape != expected_shape or output.dtype != expected_dtype:
            raise ValueError(
                f"{name} must have shape {expected_shape} and dtype {expected_dtype}"
            )
        if not output.flags.writeable:
            raise ValueError(f"{name} must be writable")
        output.fill(0)

    dates, names = daily_shape
    activity_history: deque[
        dict[int, tuple[NDArray[np.float64], NDArray[np.bool_]]]
    ] = deque(maxlen=20)
    log_two = np.log(2.0)
    for day in range(dates):
        session_supported = supported[day]
        sigma_day = np.asarray(sigma[day], dtype=np.float64)
        risk_valid = np.isfinite(sigma_day) & (sigma_day > minimum_sigma)
        s5 = sigma_day * np.sqrt(block_minutes / continuous_minutes[day])
        last_price_minute = np.full(names, -1, dtype=np.int64)
        previous_endpoint = np.zeros(names, dtype=np.float64)
        previous_endpoint_valid = np.zeros(names, dtype=np.bool_)
        current_activity: dict[int, tuple[NDArray[np.float64], NDArray[np.bool_]]] = {}
        for patch in range(int(patch_counts[day])):
            start = patch * block_minutes
            stop = start + block_minutes
            clock_minute = int(open_minutes[day] + start)
            patch_mask[day, :, patch] = session_supported

            high_slice = np.asarray(high_[day, :, start:stop], dtype=np.float64)
            low_slice = np.asarray(low_[day, :, start:stop], dtype=np.float64)
            close_slice = np.asarray(close_[day, :, start:stop], dtype=np.float64)
            seen_slice = seen[day, :, start:stop]
            price_bar_valid = (
                seen_slice
                & np.isfinite(high_slice)
                & np.isfinite(low_slice)
                & np.isfinite(close_slice)
                & (low_slice > 0.0)
                & (high_slice >= low_slice)
                & (close_slice >= low_slice)
                & (close_slice <= high_slice)
            )
            complete_price = session_supported & price_bar_valid.all(axis=1)
            block_high = high_slice.max(axis=1)
            block_low = low_slice.min(axis=1)
            block_close = close_slice[:, -1]

            range_valid = complete_price & risk_valid
            valid[day, range_valid, patch, 1] = True
            values[day, range_valid, patch, 1] = (
                np.log(block_high[range_valid] / block_low[range_valid])
                / s5[range_valid]
            ).astype(np.float32)

            valid[day, complete_price, patch, 2] = True
            positive_range = complete_price & (block_high > block_low)
            values[day, positive_range, patch, 2] = (
                2.0
                * (block_close[positive_range] - block_low[positive_range])
                / (block_high[positive_range] - block_low[positive_range])
                - 1.0
            ).astype(np.float32)

            endpoint_valid = (
                session_supported
                & seen_slice[:, -1]
                & np.isfinite(block_close)
                & (block_close > 0.0)
            )
            if patch:
                return_valid = endpoint_valid & previous_endpoint_valid & risk_valid
                valid[day, return_valid, patch, 0] = True
                values[day, return_valid, patch, 0] = (
                    np.log(block_close[return_valid] / previous_endpoint[return_valid])
                    / s5[return_valid]
                ).astype(np.float32)
            previous_endpoint = block_close.copy()
            previous_endpoint_valid = endpoint_valid

            valid[day, session_supported, patch, 4] = True
            values[day, session_supported, patch, 4] = (
                seen_slice[session_supported].sum(axis=1) / block_minutes
            ).astype(np.float32)
            valid[day, session_supported, patch, 5] = True
            values[day, session_supported, patch, 5] = np.float32(
                stop / prefix_minutes[day]
            )

            close_observed = seen_slice & np.isfinite(close_slice) & (close_slice > 0.0)
            for minute in range(block_minutes):
                current = session_supported & close_observed[:, minute]
                last_price_minute[current] = start + minute
            current_age_valid = session_supported & (last_price_minute >= 0)
            raw_age = stop - 1 - last_price_minute
            age_valid[day, current_age_valid, patch] = True
            age_minutes[day, current_age_valid, patch] = raw_age[
                current_age_valid
            ].astype(np.float32)
            valid[day, current_age_valid, patch, 6] = True
            values[day, current_age_valid, patch, 6] = np.clip(
                raw_age[current_age_valid] / continuous_minutes[day], 0.0, 1.0
            ).astype(np.float32)

            volume_slice = np.asarray(volume_[day, :, start:stop], dtype=np.float64)
            activity_slice_valid = (
                activity_seen[day, :, start:stop]
                & np.isfinite(volume_slice)
                & (volume_slice >= 0.0)
            )
            block_volume_valid = session_supported & activity_slice_valid.all(axis=1)
            block_volume = np.where(activity_slice_valid, volume_slice, 0.0).sum(axis=1)
            if len(activity_history) == 20 and block_volume_valid.any():
                history_values = np.zeros((20, names), dtype=np.float64)
                history_valid = np.zeros((20, names), dtype=np.bool_)
                for history_index, history_day in enumerate(activity_history):
                    prior = history_day.get(clock_minute)
                    if prior is not None:
                        history_values[history_index] = prior[0]
                        history_valid[history_index] = prior[1]
                candidates = block_volume_valid & (history_valid.sum(axis=0) >= 16)
                candidate_names = np.flatnonzero(candidates)
                if candidate_names.size:
                    candidate_history = np.where(
                        history_valid[:, candidate_names],
                        history_values[:, candidate_names],
                        np.nan,
                    )
                    baseline = np.nanmedian(candidate_history, axis=0)
                    usable = np.isfinite(baseline) & (baseline > 0.0)
                    usable_names = candidate_names[usable]
                    relative = np.log1p(block_volume[usable_names] / baseline[usable])
                    valid[day, usable_names, patch, 3] = True
                    values[day, usable_names, patch, 3] = np.clip(
                        relative - log_two, -5.0, 5.0
                    ).astype(np.float32)
            current_activity[clock_minute] = (
                block_volume.copy(),
                block_volume_valid.copy(),
            )
        activity_history.append(current_activity)

    return NativeFastResult(
        values=values,
        valid=valid,
        patch_mask=patch_mask,
        last_price_age_minutes=age_minutes,
        last_price_age_valid=age_valid,
    )


@dataclass(frozen=True)
class IntradayDailyResult:
    values: NDArray[np.float32]
    valid: NDArray[np.bool_]
    decision_mark: NDArray[np.float64]
    decision_mark_valid: NDArray[np.bool_]
    session_close: NDArray[np.float64]
    session_close_valid: NDArray[np.bool_]
    realized_daily_vol: NDArray[np.float64]
    fast_present: NDArray[np.bool_]
    return_consistent: NDArray[np.bool_]
    support_fraction: NDArray[np.float32]
    source_age_sessions: NDArray[np.float32]
    feature_names: tuple[str, ...] = INTRADAY_DAILY_FEATURES


def detect_open_gap_boundaries(
    raw_open: NDArray[np.floating],
    raw_close: NDArray[np.floating],
    observed: NDArray[np.bool_],
    *,
    maximum_absolute_log_gap: float = 0.30,
) -> NDArray[np.bool_]:
    """Flag large open gaps using only information available at that open.

    The comparison uses the last observed close strictly before each row.  It
    never consults the current close, quantity, distribution code or any
    later classification, so mutating post-decision data cannot change the
    decision row's boundary mask.
    """

    open_ = np.asarray(raw_open, dtype=np.float64)
    close = np.asarray(raw_close, dtype=np.float64)
    seen = np.asarray(observed, dtype=np.bool_)
    if open_.ndim != 2 or open_.shape != close.shape or open_.shape != seen.shape:
        raise ValueError("daily open/close/observed arrays must align [date, name]")
    if not np.isfinite(maximum_absolute_log_gap) or maximum_absolute_log_gap <= 0:
        raise ValueError("maximum_absolute_log_gap must be finite and positive")
    output = np.zeros(open_.shape, dtype=np.bool_)
    last_close = np.full(open_.shape[1], np.nan, dtype=np.float64)
    for row in range(open_.shape[0]):
        usable = (
            np.isfinite(open_[row])
            & (open_[row] > 0.0)
            & np.isfinite(last_close)
            & (last_close > 0.0)
        )
        output[row, usable] = (
            np.abs(np.log(open_[row, usable] / last_close[usable]))
            > maximum_absolute_log_gap
        )
        close_valid = seen[row] & np.isfinite(close[row]) & (close[row] > 0.0)
        last_close[close_valid] = close[row, close_valid]
    return output


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
    if (
        arrays[0].ndim != 3
        or any(value.shape != arrays[0].shape for value in arrays)
        or seen.shape != arrays[0].shape
    ):
        raise ValueError("M1 arrays must be aligned [date, name, minute]")
    if cutoff <= 0 or cutoff >= arrays[0].shape[2] or cutoff % 5:
        raise ValueError("decision cutoff must be an in-session five-minute boundary")
    return (*arrays, seen)


def _safe_log_ratio(
    numerator: NDArray[np.floating], denominator: NDArray[np.floating]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    top = np.asarray(numerator, dtype=np.float64)
    bottom = np.asarray(denominator, dtype=np.float64)
    valid = np.isfinite(top) & np.isfinite(bottom) & (top > 0) & (bottom > 0)
    output = np.full(top.shape, np.nan, dtype=np.float64)
    output[valid] = np.log(top[valid] / bottom[valid])
    return output, valid


def _rolling_summary(
    values: NDArray[np.floating],
    valid: NDArray[np.bool_],
    window: int,
    *,
    total: bool = False,
    completed_valid: NDArray[np.bool_] | None = None,
) -> tuple[
    NDArray[np.float64], NDArray[np.bool_], NDArray[np.float32], NDArray[np.float32]
]:
    """Use >=80% actual observations in a fixed window; never fill missing rows.

    Sums contain only observed returns (no extrapolation); means divide by actual
    support. Historical quality may include that historical session's close,
    whereas the final row uses only its decision-time validity. Age is the number
    of sessions since the newest observation consumed, not the estimate's age.
    """
    output = np.full(values.shape, np.nan, dtype=np.float64)
    mask = np.zeros(values.shape, dtype=np.bool_)
    support = np.zeros(values.shape, dtype=np.float32)
    age = np.full(values.shape, -1.0, dtype=np.float32)
    for end in range(window - 1, len(values)):
        start = end - window + 1
        usable = valid[start : end + 1].copy()
        if completed_valid is not None:
            usable[:-1] &= completed_valid[start:end]
        count = usable.sum(axis=0)
        support[end] = count / window
        accepted = count >= int(np.ceil(0.8 * window))
        summed = np.where(usable, values[start : end + 1], 0.0).sum(axis=0)
        output[end, accepted] = (
            summed[accepted] if total else summed[accepted] / count[accepted]
        )
        mask[end] = accepted
        latest = window - 1 - np.argmax(usable[::-1], axis=0)
        age[end, accepted] = window - 1 - latest[accepted]
    return output, mask, support, age


def decision_action_boundaries(
    open_price: NDArray[np.floating],
    close: NDArray[np.floating],
    observed: NDArray[np.bool_],
    prior_daily_resolved: NDArray[np.bool_],
    terms: Sequence[VerifiedActionTerm],
    sessions: Sequence[SessionDefinition],
    isins: Sequence[str],
) -> NDArray[np.bool_]:
    """Current open gaps, prior unresolved history and already announced events.

    Daily resolution row u is assessed at the following decision. A term that
    was unavailable at today's cutoff contributes no event or uncertainty bit.
    """
    boundary = detect_open_gap_boundaries(open_price, close, observed)
    boundary[1:] |= ~prior_daily_resolved[:-1]
    day_by_date = {session.trade_date: i for i, session in enumerate(sessions)}
    name_by_isin = {isin: i for i, isin in enumerate(isins)}
    for term in terms:
        name = name_by_isin.get(term.isin)
        if name is None:
            continue
        for event_date in {term.ex_date, term.effective_date}:
            day = day_by_date.get(event_date)
            if day is not None and term.available_at <= sessions[day].decision_at:
                boundary[day, name] = True
    return boundary


def shareholder_reference_returns(
    close: NDArray[np.floating],
    observed: NDArray[np.bool_],
    actions: AlignedActionTerms,
) -> NDArray[np.float64]:
    """Exact adjacent COTAHIST wealth returns, known only after each close."""
    terminal_price = np.take_along_axis(close, actions.successor_index, axis=1)
    terminal_observed = np.take_along_axis(observed, actions.successor_index, axis=1)
    ending_wealth = (
        actions.shares_per_prior_share * terminal_price + actions.cash_per_prior_share
    )
    output = np.full(close.shape, np.nan, dtype=np.float64)
    move, usable = _safe_log_ratio(ending_wealth[1:], close[:-1])
    usable &= observed[:-1] & terminal_observed[1:] & actions.session_resolved[1:]
    output[1:] = np.where(usable, move, np.nan)
    return output


def return_consistency(
    m1_close: NDArray[np.floating],
    m1_valid: NDArray[np.bool_],
    official_log_return: NDArray[np.floating],
) -> NDArray[np.bool_]:
    """Compare adjacent observed M1 endpoints with shareholder-wealth returns.

    The immutable 0.005 tolerance is in log-return units. The first row, missing
    exact endpoints and unavailable official returns are unsupported, not matches.
    The result is available only after each session closes.
    """
    accepted = np.zeros(m1_close.shape, dtype=np.bool_)
    move, usable = _safe_log_ratio(m1_close[1:], m1_close[:-1])
    accepted[1:] = (
        usable
        & m1_valid[1:]
        & m1_valid[:-1]
        & np.isfinite(official_log_return[1:])
        & (np.abs(move - official_log_return[1:]) <= 0.005)
    )
    return accepted


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


def _rolling_total(
    values: NDArray[np.floating] | NDArray[np.integer], window: int
) -> NDArray[np.float64]:
    source = np.asarray(values, dtype=np.float64)
    cumulative = np.concatenate(
        (
            np.zeros((1, *source.shape[1:]), dtype=np.float64),
            np.cumsum(source, axis=0, dtype=np.float64),
        ),
        axis=0,
    )
    output = np.zeros(source.shape, dtype=np.float64)
    output[window - 1 :] = cumulative[window:] - cumulative[:-window]
    return output


def _rolling_skew_from_moments(
    count: NDArray[np.integer],
    first: NDArray[np.floating],
    second: NDArray[np.floating],
    third: NDArray[np.floating],
    expected_per_day: NDArray[np.integer],
    window: int,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    rolling_count = _rolling_total(count, window)
    rolling_first = _rolling_total(first, window)
    rolling_second = _rolling_total(second, window)
    rolling_third = _rolling_total(third, window)
    expected = _rolling_total(expected_per_day, window)
    safe_count = np.maximum(rolling_count, 1.0)
    mean = rolling_first / safe_count
    variance = np.maximum(rolling_second / safe_count - mean**2, 0.0)
    third_central = (
        rolling_third / safe_count
        - 3.0 * mean * rolling_second / safe_count
        + 2.0 * mean**3
    )
    minimum = np.ceil(0.8 * expected)[:, None]
    valid = (rolling_count >= minimum) & (variance > 0.0)
    valid[: window - 1] = False
    output = np.full(count.shape, np.nan, dtype=np.float64)
    output[valid] = third_central[valid] / np.power(variance[valid], 1.5)
    return output, valid


def _rolling_spread_from_moments(
    count: NDArray[np.integer],
    sum_left: NDArray[np.floating],
    sum_right: NDArray[np.floating],
    cross: NDArray[np.floating],
    expected_per_day: NDArray[np.integer],
    window: int,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    rolling_count = _rolling_total(count, window)
    rolling_left = _rolling_total(sum_left, window)
    rolling_right = _rolling_total(sum_right, window)
    rolling_cross = _rolling_total(cross, window)
    safe_count = np.maximum(rolling_count, 1.0)
    covariance = (
        rolling_cross - rolling_left * rolling_right / safe_count
    ) / np.maximum(rolling_count - 1.0, 1.0)
    minimum = np.maximum(
        np.ceil(0.8 * _rolling_total(expected_per_day, window))[:, None], 2.0
    )
    valid = (rolling_count >= minimum) & (covariance < 0.0)
    valid[: window - 1] = False
    output = np.full(count.shape, np.nan, dtype=np.float64)
    output[valid] = 2.0 * np.sqrt(-covariance[valid])
    return output, valid


def build_intraday_daily_features(
    open_price: NDArray[np.floating],
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    close: NDArray[np.floating],
    volume: NDArray[np.floating],
    observed: NDArray[np.bool_],
    *,
    volume_valid: NDArray[np.bool_],
    session_valid: NDArray[np.bool_],
    sessions: Sequence[SessionDefinition],
    official_log_return: NDArray[np.floating] | None = None,
    completed_action_boundary: NDArray[np.bool_] | None = None,
    same_day_boundary: NDArray[np.bool_] | None = None,
) -> IntradayDailyResult:
    market = tuple(
        np.asarray(value) for value in (open_price, high, low, close, volume)
    )
    open_, high_, low_, close_, volume_ = market
    seen = np.asarray(observed)
    activity_seen = np.asarray(volume_valid)
    supported = np.asarray(session_valid)
    if open_.ndim != 3 or any(value.shape != open_.shape for value in market):
        raise ValueError("M1 arrays must be aligned [date, name, minute]")
    if any(not np.issubdtype(value.dtype, np.number) for value in market):
        raise TypeError("M1 market arrays must be numeric")
    if seen.dtype != np.bool_ or activity_seen.dtype != np.bool_:
        raise TypeError("observed and volume_valid must be boolean")
    if seen.shape != open_.shape or activity_seen.shape != open_.shape:
        raise ValueError("M1 price/activity masks must align with market arrays")
    if supported.dtype != np.bool_ or supported.shape != open_.shape[:2]:
        raise ValueError("session_valid must be boolean and align [date, name]")
    if np.any(seen & ~supported[..., None]) or np.any(
        activity_seen & ~supported[..., None]
    ):
        raise ValueError("M1 observations cannot exist outside source support")
    if np.any(activity_seen & (~np.isfinite(volume_) | (volume_ < 0.0))):
        raise ValueError("valid M1 activity must be finite and non-negative")

    prefix_minutes, continuous_minutes, _, patch_counts = _native_session_layout(
        sessions,
        date_count=open_.shape[0],
        minute_count=open_.shape[2],
        block_minutes=5,
    )
    if np.any(continuous_minutes > open_.shape[2]):
        raise ValueError("M1 inputs do not cover every continuous session")

    dates, names = open_.shape[:2]
    shape = (dates, names)

    def floats() -> NDArray[np.float64]:
        return np.full(shape, np.nan, dtype=np.float64)

    day_open = floats()
    open_valid = np.zeros(shape, dtype=np.bool_)
    decision_mark = floats()
    decision_mark_valid = np.zeros(shape, dtype=np.bool_)
    final_close = floats()
    final_valid = np.zeros(shape, dtype=np.bool_)
    last30_open = floats()
    last30_open_valid = np.zeros(shape, dtype=np.bool_)
    full_volume = np.zeros(shape, dtype=np.float64)
    full_volume_valid = np.zeros(shape, dtype=np.bool_)
    last_hour_volume = np.zeros(shape, dtype=np.float64)
    last_hour_valid = np.zeros(shape, dtype=np.bool_)
    full_vwap = floats()
    full_vwap_valid = np.zeros(shape, dtype=np.bool_)
    prefix_total = np.zeros(shape, dtype=np.float64)
    prefix_vwap = floats()
    prefix_price_valid = np.zeros(shape, dtype=np.bool_)
    prefix_activity_valid = np.zeros(shape, dtype=np.bool_)
    prefix_vwap_valid = np.zeros(shape, dtype=np.bool_)
    day_high = floats()
    day_low = floats()
    day_range_valid = np.zeros(shape, dtype=np.bool_)
    realized = floats()
    realized_valid = np.zeros(shape, dtype=np.bool_)

    return_count = np.zeros(shape, dtype=np.int32)
    return_first = np.zeros(shape, dtype=np.float64)
    return_second = np.zeros(shape, dtype=np.float64)
    return_third = np.zeros(shape, dtype=np.float64)
    pair_count = np.zeros(shape, dtype=np.int32)
    pair_left = np.zeros(shape, dtype=np.float64)
    pair_right = np.zeros(shape, dtype=np.float64)
    pair_cross = np.zeros(shape, dtype=np.float64)
    expected_returns = np.maximum(patch_counts - 1, 0).astype(np.int32)
    expected_pairs = np.maximum(patch_counts - 2, 0).astype(np.int32)

    for day in range(dates):
        prefix = int(prefix_minutes[day])
        continuous = int(continuous_minutes[day])
        open_day = np.asarray(open_[day, :, :continuous], dtype=np.float64)
        high_day = np.asarray(high_[day, :, :continuous], dtype=np.float64)
        low_day = np.asarray(low_[day, :, :continuous], dtype=np.float64)
        close_day = np.asarray(close_[day, :, :continuous], dtype=np.float64)
        volume_day = np.asarray(volume_[day, :, :continuous], dtype=np.float64)
        seen_day = seen[day, :, :continuous]
        activity_day = activity_seen[day, :, :continuous]

        day_open[day] = open_day[:, 0]
        open_valid[day] = (
            seen_day[:, 0] & np.isfinite(day_open[day]) & (day_open[day] > 0.0)
        )
        final_close[day] = close_day[:, -1]
        final_valid[day] = (
            seen_day[:, -1] & np.isfinite(final_close[day]) & (final_close[day] > 0.0)
        )
        if continuous >= 30:
            last30_open[day] = open_day[:, continuous - 30]
            last30_open_valid[day] = (
                seen_day[:, continuous - 30]
                & np.isfinite(last30_open[day])
                & (last30_open[day] > 0.0)
            )

        prefix_seen = seen_day[:, :prefix]
        prefix_close = close_day[:, :prefix]
        completed_price = prefix_seen & np.isfinite(prefix_close) & (prefix_close > 0.0)
        has_completed_price = completed_price.any(axis=1)
        if has_completed_price.any():
            last_offset = (
                prefix
                - 1
                - np.argmax(completed_price[has_completed_price, ::-1], axis=1)
            )
            decision_mark[day, has_completed_price] = prefix_close[
                has_completed_price, last_offset
            ]
            decision_mark_valid[day, has_completed_price] = True

        safe_volume = np.where(activity_day, volume_day, 0.0)
        full_volume[day] = safe_volume.sum(axis=1)
        full_volume_valid[day] = activity_day.all(axis=1)
        finite_close = seen_day & np.isfinite(close_day) & (close_day > 0.0)
        priced_activity = activity_day & ((volume_day == 0.0) | finite_close)
        full_vwap_valid[day] = (
            full_volume_valid[day]
            & priced_activity.all(axis=1)
            & (full_volume[day] > 0.0)
        )
        weighted_full = (np.where(finite_close, close_day, 0.0) * safe_volume).sum(
            axis=1
        )
        full_vwap[day, full_vwap_valid[day]] = (
            weighted_full[full_vwap_valid[day]] / full_volume[day, full_vwap_valid[day]]
        )
        if continuous >= 60:
            last_hour_volume[day] = safe_volume[:, -60:].sum(axis=1)
            last_hour_valid[day] = (full_volume[day] > 0.0) & full_volume_valid[day]

        prefix_volume = safe_volume[:, :prefix]
        prefix_total[day] = prefix_volume.sum(axis=1)
        prefix_activity = activity_day[:, :prefix]
        prefix_activity_valid[day] = prefix_activity.all(axis=1)
        finite_prefix_close = (
            prefix_seen & np.isfinite(prefix_close) & (prefix_close > 0.0)
        )
        priced_prefix_activity = prefix_activity & (
            (volume_day[:, :prefix] == 0.0) | finite_prefix_close
        )
        prefix_vwap_valid[day] = (
            prefix_activity_valid[day]
            & priced_prefix_activity.all(axis=1)
            & (prefix_total[day] > 0.0)
        )
        weighted_prefix = (
            np.where(finite_prefix_close, prefix_close, 0.0) * prefix_volume
        ).sum(axis=1)
        prefix_vwap[day, prefix_vwap_valid[day]] = (
            weighted_prefix[prefix_vwap_valid[day]]
            / prefix_total[day, prefix_vwap_valid[day]]
        )
        prefix_count = prefix_seen.sum(axis=1)
        prefix_price_valid[day] = prefix_count >= int(np.ceil(0.8 * prefix))
        day_high[day] = np.max(
            np.where(prefix_seen, high_day[:, :prefix], -np.inf), axis=1
        )
        day_low[day] = np.min(
            np.where(prefix_seen, low_day[:, :prefix], np.inf), axis=1
        )
        day_range_valid[day] = (
            (prefix_count >= int(0.8 * prefix))
            & np.isfinite(day_high[day])
            & np.isfinite(day_low[day])
            & (day_high[day] >= day_low[day])
            & (day_low[day] > 0.0)
        )

        block_closes = prefix_close[:, 4:prefix:5]
        block_seen = prefix_seen[:, 4:prefix:5]
        if block_closes.shape[1] < 2:
            continue
        block_returns = np.zeros((names, block_closes.shape[1] - 1), dtype=np.float64)
        block_valid = (
            block_seen[:, 1:]
            & block_seen[:, :-1]
            & np.isfinite(block_closes[:, 1:])
            & np.isfinite(block_closes[:, :-1])
            & (block_closes[:, 1:] > 0.0)
            & (block_closes[:, :-1] > 0.0)
        )
        block_returns[block_valid] = np.log(
            block_closes[:, 1:][block_valid] / block_closes[:, :-1][block_valid]
        )
        return_count[day] = block_valid.sum(axis=1)
        return_first[day] = np.where(block_valid, block_returns, 0.0).sum(axis=1)
        return_second[day] = np.where(block_valid, block_returns**2, 0.0).sum(axis=1)
        return_third[day] = np.where(block_valid, block_returns**3, 0.0).sum(axis=1)
        minimum_returns = int(np.ceil(0.8 * expected_returns[day]))
        realized_valid[day] = return_count[day] >= minimum_returns
        realized[day, realized_valid[day]] = np.sqrt(
            return_second[day, realized_valid[day]]
        )

        if block_returns.shape[1] < 2:
            continue
        adjacent_valid = block_valid[:, 1:] & block_valid[:, :-1]
        left = block_returns[:, :-1]
        right = block_returns[:, 1:]
        pair_count[day] = adjacent_valid.sum(axis=1)
        pair_left[day] = np.where(adjacent_valid, left, 0.0).sum(axis=1)
        pair_right[day] = np.where(adjacent_valid, right, 0.0).sum(axis=1)
        pair_cross[day] = np.where(adjacent_valid, left * right, 0.0).sum(axis=1)

    consistent = (
        final_valid.copy()
        if official_log_return is None
        else return_consistency(final_close, final_valid, official_log_return)
    )
    completed_accepted = consistent.copy()
    if completed_action_boundary is not None:
        completed_accepted &= ~completed_action_boundary
    prior_accepted = np.zeros(shape, dtype=np.bool_)
    prior_accepted[1:] = completed_accepted[:-1]
    known_clear = (
        np.ones(shape, dtype=np.bool_)
        if same_day_boundary is None
        else ~same_day_boundary
    )
    values = np.zeros((*shape, len(INTRADAY_DAILY_FEATURES)), dtype=np.float32)
    masks = np.zeros(values.shape, dtype=np.bool_)
    support = np.zeros(values.shape, dtype=np.float32)
    source_age = np.full(values.shape, -1.0, dtype=np.float32)

    def assign(
        index: int,
        column: NDArray[np.floating],
        valid: NDArray[np.bool_],
        fraction: NDArray[np.floating] | None = None,
        age: NDArray[np.floating] | float = 0.0,
    ) -> None:
        usable = np.asarray(valid, dtype=np.bool_) & np.isfinite(column)
        values[..., index][usable] = np.asarray(column)[usable].astype(np.float32)
        masks[..., index] = usable
        support[..., index] = usable if fraction is None else fraction
        source_age[..., index] = np.where(usable, age, -1.0)

    overnight = floats()
    overnight_valid = np.zeros(shape, dtype=np.bool_)
    if dates > 1:
        overnight[1:], overnight_valid[1:] = _safe_log_ratio(
            day_open[1:], final_close[:-1]
        )
        overnight_valid[1:] &= open_valid[1:] & final_valid[:-1]
        overnight_valid &= prior_accepted & known_clear
        overnight[~overnight_valid] = np.nan
    intraday, intraday_valid = _safe_log_ratio(decision_mark, day_open)
    intraday_valid &= decision_mark_valid & open_valid
    intraday[~intraday_valid] = np.nan
    assign(0, overnight, overnight_valid)
    assign(1, intraday, intraday_valid)
    for index, base, base_valid, window in (
        (2, overnight, overnight_valid, 5),
        (3, overnight, overnight_valid, 20),
        (4, intraday, intraday_valid, 5),
        (5, intraday, intraday_valid, 20),
    ):
        assign(
            index,
            *_rolling_summary(
                base,
                base_valid,
                window,
                total=True,
                completed_valid=completed_accepted if index in (2, 3) else None,
            ),
        )
    differential = overnight - intraday
    differential_valid = overnight_valid & intraday_valid
    assign(6, differential, differential_valid)
    assign(
        7,
        *_rolling_summary(
            differential,
            differential_valid,
            20,
            completed_valid=completed_accepted,
        ),
    )

    last30, last30_valid = _safe_log_ratio(final_close, last30_open)
    full_return, full_return_valid = _safe_log_ratio(final_close, day_open)
    last30_valid &= final_valid & last30_open_valid
    full_return_valid &= final_valid & open_valid
    with np.errstate(divide="ignore", invalid="ignore"):
        last30_share = last30 / full_return
    last30_share_valid = (
        last30_valid & full_return_valid & (np.abs(full_return) > 1e-12)
    )
    prior_last30 = floats()
    prior_last30_valid = np.zeros(shape, dtype=np.bool_)
    prior_last30[1:] = last30_share[:-1]
    prior_last30_valid[1:] = last30_share_valid[:-1]
    assign(8, prior_last30, prior_last30_valid & prior_accepted, age=1.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        last_hour_share = last_hour_volume / full_volume
    lag_last_hour = floats()
    lag_last_hour_valid = np.zeros(shape, dtype=np.bool_)
    lag_last_hour[1:] = last_hour_share[:-1]
    lag_last_hour_valid[1:] = last_hour_valid[:-1]
    assign(9, lag_last_hour, lag_last_hour_valid & prior_accepted, age=1.0)

    close_vwap, close_vwap_valid = _safe_log_ratio(final_close, full_vwap)
    close_vwap_valid &= final_valid & full_vwap_valid
    lag_close_vwap = floats()
    lag_close_vwap_valid = np.zeros(shape, dtype=np.bool_)
    lag_close_vwap[1:] = close_vwap[:-1]
    lag_close_vwap_valid[1:] = close_vwap_valid[:-1]
    assign(10, lag_close_vwap, lag_close_vwap_valid & prior_accepted, age=1.0)

    vwap_deviation, vwap_valid = _safe_log_ratio(decision_mark, prefix_vwap)
    vwap_valid &= decision_mark_valid & prefix_vwap_valid
    assign(11, vwap_deviation, vwap_valid)
    assign(12, realized, realized_valid)
    assign(13, *_rolling_summary(realized, realized_valid, 5))
    assign(14, *_rolling_summary(realized, realized_valid, 20))
    assign(
        15,
        *_rolling_skew_from_moments(
            return_count,
            return_first,
            return_second,
            return_third,
            expected_returns,
            20,
        ),
    )
    assign(
        16,
        *_rolling_spread_from_moments(
            pair_count,
            pair_left,
            pair_right,
            pair_cross,
            expected_pairs,
            20,
        ),
    )
    for feature, count, expected in (
        (15, return_count, expected_returns),
        (16, pair_count, expected_pairs),
    ):
        expected_total = _rolling_total(expected, 20)[:, None]
        support[..., feature] = np.divide(
            _rolling_total(count, 20),
            expected_total,
            out=np.zeros(shape),
            where=expected_total > 0,
        )
        for day in range(19, dates):
            usable = masks[day, :, feature]
            source_age[day, usable, feature] = np.argmax(
                (count[day - 19 : day + 1, usable] > 0)[::-1],
                axis=0,
            )
    spread, spread_valid = _corwin_schultz(day_high, day_low, day_range_valid)
    assign(
        17,
        *_rolling_summary(
            spread,
            spread_valid & known_clear,
            20,
            completed_valid=completed_accepted,
        ),
    )
    range_value, range_valid = _safe_log_ratio(day_high, day_low)
    range_valid &= day_range_valid
    assign(18, range_value, range_valid)

    relative = floats()
    relative_valid = np.zeros(shape, dtype=np.bool_)
    for day in range(20, dates):
        history = (
            prefix_activity_valid[day - 20 : day] & completed_accepted[day - 20 : day]
        )
        history_valid = history.sum(axis=0) >= 16
        median = np.full(names, np.nan, dtype=np.float64)
        if history_valid.any():
            median[history_valid] = np.nanmedian(
                np.where(
                    history[:, history_valid],
                    prefix_total[day - 20 : day, history_valid],
                    np.nan,
                ),
                axis=0,
            )
        support[day, :, 19] = history.sum(axis=0) / 20
        usable = (
            prefix_activity_valid[day]
            & history_valid
            & np.isfinite(median)
            & (median > 0.0)
        )
        relative[day, usable] = prefix_total[day, usable] / median[usable]
        relative_valid[day, usable] = True
    assign(19, relative, relative_valid, support[..., 19].copy())

    return IntradayDailyResult(
        values=values,
        valid=masks,
        # Completed prefix mark; execution entry is independently sourced.
        decision_mark=decision_mark,
        decision_mark_valid=decision_mark_valid,
        session_close=np.where(final_valid, final_close, np.nan),
        session_close_valid=final_valid,
        realized_daily_vol=realized,
        fast_present=prefix_price_valid & realized_valid,
        return_consistent=consistent,
        support_fraction=support,
        source_age_sessions=source_age,
    )
