from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from .contract import SLOW_FEATURES


@dataclass(frozen=True)
class SlowFeatureResult:
    values: NDArray[np.float32]
    valid: NDArray[np.bool_]
    feature_names: tuple[str, ...] = SLOW_FEATURES
    cluster_labels: NDArray[np.int16] | None = None


def _aligned_daily(*arrays: NDArray[np.generic]) -> tuple[NDArray[np.float64], ...]:
    converted = tuple(np.asarray(value, dtype=np.float64) for value in arrays)
    if (
        not converted
        or converted[0].ndim != 2
        or any(value.shape != converted[0].shape for value in converted)
    ):
        raise ValueError("daily arrays must be aligned [date, name]")
    return converted


def _ambiguous_interval_clear(
    ambiguous_action: NDArray[np.bool_], horizon: int
) -> NDArray[np.bool_]:
    """Mark trailing ``(t-horizon, t]`` intervals without ambiguous actions."""

    ambiguous = np.asarray(ambiguous_action, dtype=np.bool_)
    if ambiguous.ndim != 2 or horizon < 0:
        raise ValueError(
            "ambiguous_action must be [date, name] and horizon non-negative"
        )
    if horizon == 0:
        return np.ones(ambiguous.shape, dtype=np.bool_)
    clear = np.zeros(ambiguous.shape, dtype=np.bool_)
    cumulative = np.cumsum(ambiguous, axis=0, dtype=np.int32)
    clear[horizon:] = cumulative[horizon:] - cumulative[:-horizon] == 0
    return clear


def exact_log_return(
    close: NDArray[np.floating],
    horizon: int,
    ambiguous_action: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    values = np.asarray(close, dtype=np.float64)
    if values.ndim != 2 or horizon <= 0:
        raise ValueError("close must be [date, name] and horizon positive")
    output = np.full(values.shape, np.nan, dtype=np.float64)
    valid = np.zeros(values.shape, dtype=np.bool_)
    with np.errstate(divide="ignore", invalid="ignore"):
        candidate = np.log(values[horizon:] / values[:-horizon])
    mask = (
        np.isfinite(candidate)
        & np.isfinite(values[horizon:])
        & np.isfinite(values[:-horizon])
        & (values[horizon:] > 0)
        & (values[:-horizon] > 0)
    )
    if ambiguous_action is not None:
        ambiguous = np.asarray(ambiguous_action, dtype=np.bool_)
        if ambiguous.shape != values.shape:
            raise ValueError("ambiguous_action must align with close")
        # For a return ending at t, exclude events in (t-horizon, t].
        mask &= _ambiguous_interval_clear(ambiguous, horizon)[horizon:]
    output[horizon:] = np.where(mask, candidate, np.nan)
    valid[horizon:] = mask
    return output, valid


def yang_zhang_volatility(
    open_price: NDArray[np.floating],
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    close: NDArray[np.floating],
    window: int,
    ambiguous_action: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Yang-Zhang daily volatility with sample variances and exact windows."""

    open_, high_, low_, close_ = _aligned_daily(open_price, high, low, close)
    if window < 2:
        raise ValueError("Yang-Zhang window must be at least two")
    shape = close_.shape
    ambiguous = (
        np.zeros(shape, dtype=np.bool_)
        if ambiguous_action is None
        else np.asarray(ambiguous_action, dtype=np.bool_)
    )
    if ambiguous.shape != shape:
        raise ValueError("ambiguous_action must align with daily prices")
    interval_clear = _ambiguous_interval_clear(ambiguous, window)
    output = np.full(shape, np.nan, dtype=np.float64)
    valid = np.zeros(shape, dtype=np.bool_)
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        overnight = np.log(open_[1:] / close_[:-1])
        intraday = np.log(close_[1:] / open_[1:])
        rogers_satchell = np.log(high_[1:] / close_[1:]) * np.log(
            high_[1:] / open_[1:]
        ) + np.log(low_[1:] / close_[1:]) * np.log(low_[1:] / open_[1:])
    component_valid = (
        np.isfinite(overnight)
        & np.isfinite(intraday)
        & np.isfinite(rogers_satchell)
        & (open_[1:] > 0)
        & (high_[1:] > 0)
        & (low_[1:] > 0)
        & (close_[1:] > 0)
        & (close_[:-1] > 0)
    )
    minimum = math.ceil(0.8 * window)
    for end in range(window, shape[0]):
        start = end - window
        component_window = component_valid[start:end]
        complete = (component_window.sum(axis=0) >= minimum) & interval_clear[end]
        if not complete.any():
            continue
        usable = component_window[:, complete]
        sigma_open = np.nanvar(
            np.where(usable, overnight[start:end, complete], np.nan),
            axis=0,
            ddof=1,
        )
        sigma_close = np.nanvar(
            np.where(usable, intraday[start:end, complete], np.nan),
            axis=0,
            ddof=1,
        )
        sigma_rs = np.nanmean(
            np.where(usable, rogers_satchell[start:end, complete], np.nan),
            axis=0,
        )
        variance = np.maximum(sigma_open + k * sigma_close + (1.0 - k) * sigma_rs, 0.0)
        output[end, complete] = np.sqrt(variance)
        valid[end, complete] = True
    return output, valid


def _rolling_stat(
    values: NDArray[np.float64],
    window: int,
    statistic: str,
    *,
    minimum: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    output = np.full(values.shape, np.nan, dtype=np.float64)
    valid = np.zeros(values.shape, dtype=np.bool_)
    needed = math.ceil(0.8 * window) if minimum is None else minimum
    for end in range(window - 1, values.shape[0]):
        sample = values[end - window + 1 : end + 1]
        finite = np.isfinite(sample)
        enough = finite.sum(axis=0) >= needed
        if not enough.any():
            continue
        safe = np.where(finite, sample, np.nan)
        with np.errstate(all="ignore"):
            if statistic == "mean":
                result = np.nanmean(safe, axis=0)
            elif statistic == "std":
                result = np.nanstd(safe, axis=0)
            elif statistic == "max":
                result = np.nanmax(safe, axis=0)
            elif statistic == "median":
                result = np.nanmedian(safe, axis=0)
            else:
                raise ValueError(statistic)
        output[end, enough] = result[enough]
        valid[end, enough] = np.isfinite(result[enough])
    return output, valid


def _rolling_moments(
    values: NDArray[np.float64], window: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    skew = np.full(values.shape, np.nan, dtype=np.float64)
    kurtosis = np.full(values.shape, np.nan, dtype=np.float64)
    valid = np.zeros(values.shape, dtype=np.bool_)
    for end in range(window - 1, values.shape[0]):
        sample = values[end - window + 1 : end + 1]
        finite = np.isfinite(sample)
        complete = finite.sum(axis=0) >= math.ceil(0.8 * window)
        if not complete.any():
            continue
        selected = np.where(finite[:, complete], sample[:, complete], np.nan)
        centered = selected - np.nanmean(selected, axis=0)
        scale = np.sqrt(np.nanmean(centered**2, axis=0))
        nonzero = scale > 0
        slots = np.flatnonzero(complete)[nonzero]
        if slots.size:
            standardized = centered[:, nonzero] / scale[nonzero]
            skew[end, slots] = np.nanmean(standardized**3, axis=0)
            kurtosis[end, slots] = np.nanmean(standardized**4, axis=0) - 3.0
            valid[end, slots] = True
    return skew, kurtosis, valid


def _rolling_high_low_range(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    observed: NDArray[np.bool_],
    window: int,
    ambiguous_action: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Return ``log(max(H)/min(L))`` over an exact session window."""

    output = np.full(high.shape, np.nan, dtype=np.float64)
    valid = np.zeros(high.shape, dtype=np.bool_)
    ambiguous = (
        np.zeros(high.shape, dtype=np.bool_)
        if ambiguous_action is None
        else np.asarray(ambiguous_action, dtype=np.bool_)
    )
    if ambiguous.shape != high.shape:
        raise ValueError("ambiguous_action must align with daily ranges")
    interval_clear = _ambiguous_interval_clear(ambiguous, window - 1)
    for end in range(window - 1, high.shape[0]):
        start = end - window + 1
        usable_rows = (
            observed[start : end + 1]
            & np.isfinite(high[start : end + 1])
            & np.isfinite(low[start : end + 1])
            & (high[start : end + 1] > 0)
            & (low[start : end + 1] > 0)
        )
        complete = (
            usable_rows.sum(axis=0) >= math.ceil(0.8 * window)
        ) & interval_clear[end]
        if not complete.any():
            continue
        maximum = np.nanmax(
            np.where(usable_rows[:, complete], high[start : end + 1, complete], np.nan),
            axis=0,
        )
        minimum = np.nanmin(
            np.where(usable_rows[:, complete], low[start : end + 1, complete], np.nan),
            axis=0,
        )
        usable = maximum >= minimum
        slots = np.flatnonzero(complete)[usable]
        output[end, slots] = np.log(maximum[usable] / minimum[usable])
        valid[end, slots] = True
    return output, valid


def monthly_cluster_labels(
    dates: Sequence[object],
    median_removed_returns: NDArray[np.floating],
    valid: NDArray[np.bool_],
    active: NDArray[np.bool_],
    *,
    lookback: int = 126,
    minimum_observed: int = 101,
    cluster_count: int = 12,
) -> NDArray[np.int16]:
    """Freeze each month's clusters from information through the prior session."""

    returns = np.asarray(median_removed_returns, dtype=np.float64)
    available = np.asarray(valid, dtype=np.bool_)
    membership = np.asarray(active, dtype=np.bool_)
    if returns.shape != available.shape or returns.shape != membership.shape:
        raise ValueError("cluster inputs must be aligned [date, name]")
    date_values = np.asarray(dates, dtype="datetime64[D]")
    if date_values.shape != (returns.shape[0],):
        raise ValueError("cluster dates are misaligned")
    labels = np.full(returns.shape, -1, dtype=np.int16)
    current = np.full(returns.shape[1], -1, dtype=np.int16)
    prior_month: str | None = None
    for day in range(returns.shape[0]):
        month = str(date_values[day].astype("datetime64[M]"))
        if month != prior_month:
            prior_month = month
            current = np.full(returns.shape[1], -1, dtype=np.int16)
            start = max(0, day - lookback)
            sample = returns[start:day]
            sample_valid = available[start:day]
            eligible = (sample_valid.sum(axis=0) >= minimum_observed) & (
                membership[day - 1] if day else False
            )
            slots = np.flatnonzero(eligible)
            if slots.size >= cluster_count:
                correlations = pairwise_masked_correlation(
                    sample[:, slots],
                    sample_valid[:, slots],
                    minimum_observed=minimum_observed,
                )
                clustered = deterministic_average_linkage(
                    correlations,
                    np.ones(slots.size, dtype=np.bool_),
                    cluster_count=cluster_count,
                )
                current[slots] = clustered
        labels[day] = current
    return labels


def pairwise_masked_correlation(
    values: NDArray[np.floating],
    valid: NDArray[np.bool_],
    *,
    minimum_observed: int,
) -> NDArray[np.float64]:
    """Exact pairwise Spearman correlations with one sort per name.

    For each name, ranks against every other name's common-observation mask
    are built together.  This preserves Experiment 46's pairwise-deletion
    definition without sorting each of the O(n²) pairs independently.
    """

    array = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=np.bool_) & np.isfinite(array)
    if array.ndim != 2 or mask.shape != array.shape or minimum_observed < 2:
        raise ValueError("invalid pairwise-correlation inputs")
    row_count, name_count = array.shape
    common_ranks = np.zeros((name_count, row_count, name_count), dtype=np.float32)
    for subject in range(name_count):
        observed_rows = np.flatnonzero(mask[:, subject])
        order = observed_rows[
            np.argsort(array[observed_rows, subject], kind="mergesort")
        ]
        if not order.size:
            continue
        partner_observed = mask[order]
        cumulative = np.cumsum(partner_observed, axis=0, dtype=np.int32)
        start = 0
        while start < order.size:
            end = start + 1
            while (
                end < order.size
                and array[order[end], subject] == array[order[start], subject]
            ):
                end += 1
            before = cumulative[start - 1] if start else 0
            tied_count = cumulative[end - 1] - before
            average_rank = before + 0.5 * (tied_count + 1)
            common_ranks[subject, order[start:end]] = np.where(
                partner_observed[start:end], average_rank, 0.0
            )
            start = end

    counts = mask.astype(np.int32).T @ mask.astype(np.int32)
    correlation = np.full((name_count, name_count), np.nan, dtype=np.float64)
    for left in range(name_count - 1):
        left_ranks = common_ranks[left, :, left + 1 :].astype(np.float64)
        right_ranks = common_ranks[left + 1 :, :, left].T.astype(np.float64)
        count = counts[left, left + 1 :].astype(np.float64)
        left_sum = left_ranks.sum(axis=0)
        right_sum = right_ranks.sum(axis=0)
        cross = (left_ranks * right_ranks).sum(axis=0)
        left_square = (left_ranks**2).sum(axis=0)
        right_square = (right_ranks**2).sum(axis=0)
        safe_count = np.maximum(count, 1.0)
        covariance = cross - left_sum * right_sum / safe_count
        left_variance = left_square - left_sum**2 / safe_count
        right_variance = right_square - right_sum**2 / safe_count
        denominator = np.sqrt(np.maximum(left_variance * right_variance, 0.0))
        usable = (count >= minimum_observed) & (denominator > 0)
        row = np.full(name_count - left - 1, np.nan, dtype=np.float64)
        row[usable] = covariance[usable] / denominator[usable]
        correlation[left, left + 1 :] = row
        correlation[left + 1 :, left] = row
    np.fill_diagonal(correlation, 1.0)
    return np.clip(correlation, -1.0, 1.0)


def deterministic_average_linkage(
    correlations: NDArray[np.floating],
    eligible: NDArray[np.bool_],
    *,
    cluster_count: int,
    minimum_size: int = 3,
) -> NDArray[np.int16]:
    """O(n² log n) deterministic average-linkage via Lance-Williams updates."""

    import heapq

    matrix = np.asarray(correlations, dtype=np.float64)
    allowed = np.asarray(eligible, dtype=np.bool_)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or allowed.shape != (matrix.shape[0],)
    ):
        raise ValueError("average-linkage inputs are misaligned")
    slots = np.flatnonzero(allowed)
    labels = np.full(matrix.shape[0], -1, dtype=np.int16)
    if slots.size < cluster_count or cluster_count <= 0 or minimum_size <= 0:
        return labels
    active: dict[int, tuple[int, ...]] = {int(slot): (int(slot),) for slot in slots}
    distances: dict[tuple[int, int], float] = {}
    heap: list[tuple[float, int, int, int, int]] = []
    for left_position, left_slot in enumerate(slots[:-1]):
        left = int(left_slot)
        for right_slot in slots[left_position + 1 :]:
            right = int(right_slot)
            correlation = matrix[left, right]
            distance = (
                2.0
                if not np.isfinite(correlation)
                else float(np.clip(1.0 - correlation, 0.0, 2.0))
            )
            distances[(left, right)] = distance
            heapq.heappush(heap, (distance, left, right, left, right))
    next_id = matrix.shape[0]
    while len(active) > cluster_count:
        while heap:
            distance, _, _, left, right = heapq.heappop(heap)
            key = (min(left, right), max(left, right))
            if left in active and right in active and distances.get(key) == distance:
                break
        else:
            raise RuntimeError("average-linkage heap was exhausted")
        left_members = active.pop(left)
        right_members = active.pop(right)
        new_members = tuple(sorted((*left_members, *right_members)))
        new_id = next_id
        next_id += 1
        for other, other_members in tuple(active.items()):
            left_key = (min(left, other), max(left, other))
            right_key = (min(right, other), max(right, other))
            left_distance = distances[left_key]
            right_distance = distances[right_key]
            updated = (
                len(left_members) * left_distance + len(right_members) * right_distance
            ) / len(new_members)
            key = (min(new_id, other), max(new_id, other))
            distances[key] = updated
            heapq.heappush(
                heap,
                (
                    updated,
                    min(min(new_members), min(other_members)),
                    max(min(new_members), min(other_members)),
                    new_id,
                    other,
                ),
            )
        active[new_id] = new_members
    ordered = sorted(active.values(), key=lambda members: members[0])
    distance = np.clip(1.0 - matrix, 0.0, 2.0)
    distance[~np.isfinite(distance)] = 2.0
    while len(ordered) > 1 and min(map(len, ordered)) < minimum_size:
        source = min(
            range(len(ordered)), key=lambda index: (len(ordered[index]), index)
        )
        destination = min(
            (index for index in range(len(ordered)) if index != source),
            key=lambda index: (
                float(np.mean(distance[np.ix_(ordered[source], ordered[index])])),
                index,
            ),
        )
        ordered[destination] = tuple(sorted((*ordered[destination], *ordered[source])))
        del ordered[source]
    ordered.sort(key=lambda members: members[0])
    for label, members in enumerate(ordered):
        labels[np.asarray(members, dtype=np.int64)] = label
    return labels


def _peer_features(
    return_5: NDArray[np.float64],
    valid_5: NDArray[np.bool_],
    return_21: NDArray[np.float64],
    valid_21: NDArray[np.bool_],
    labels: NDArray[np.int16],
    active: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    output = np.full((*return_5.shape, 5), np.nan, dtype=np.float64)
    valid = np.zeros(output.shape, dtype=np.bool_)
    for day in range(return_5.shape[0]):
        for name in range(return_5.shape[1]):
            label = labels[day, name]
            if label < 0 or not active[day, name]:
                continue
            cluster = (labels[day] == label) & active[day]
            cluster[name] = False
            peer5 = cluster & valid_5[day]
            peer21 = cluster & valid_21[day]
            if valid_5[day, name] and int(peer5.sum()) >= 3:
                mean5 = float(np.mean(return_5[day, peer5]))
                output[day, name, 0] = mean5
                output[day, name, 2] = return_5[day, name] - mean5
                valid[day, name, 0] = True
                valid[day, name, 2] = True
            if valid_21[day, name] and int(peer21.sum()) >= 3:
                values21 = return_21[day, peer21]
                mean21 = float(np.mean(values21))
                output[day, name, 1] = mean21
                output[day, name, 3] = return_21[day, name] - mean21
                output[day, name, 4] = float(np.std(values21))
                valid[day, name, 1] = True
                valid[day, name, 3] = True
                valid[day, name, 4] = True
    return output, valid


def build_slow_features_into(
    shareholder_wealth_open: NDArray[np.floating],
    shareholder_wealth_high: NDArray[np.floating],
    shareholder_wealth_low: NDArray[np.floating],
    shareholder_wealth_close: NDArray[np.floating],
    volume_brl: NDArray[np.floating],
    trades: NDArray[np.floating],
    shareholder_wealth_valid: NDArray[np.bool_],
    active: NDArray[np.bool_],
    dates: Sequence[object],
    *,
    raw_high: NDArray[np.floating],
    raw_low: NDArray[np.floating],
    raw_close: NDArray[np.floating],
    price_observed: NDArray[np.bool_],
    history_observed: NDArray[np.bool_],
    activity_valid: NDArray[np.bool_],
    consume: Callable[[int, NDArray[np.floating], NDArray[np.bool_]], None],
    cluster_labels: NDArray[np.integer] | None = None,
    ambiguous_action: NDArray[np.bool_] | None = None,
) -> NDArray[np.int16]:
    """Compute one slow field at a time and immediately hand it to ``consume``.

    The callback is responsible for writing/normalizing the field.  Raw
    32-field cubes are never materialized by this path; only the small set of
    dependency panels needed by later formulas remains live.  Cross-session
    price features consume the shareholder-wealth coordinate and its validity;
    same-session shape, activity, and linked observation age retain independent
    raw-source masks.
    """

    open_, high, low, close, volume, trade_count = tuple(
        np.asarray(value)
        for value in (
            shareholder_wealth_open,
            shareholder_wealth_high,
            shareholder_wealth_low,
            shareholder_wealth_close,
            volume_brl,
            trades,
        )
    )
    if close.ndim != 2 or any(
        value.shape != close.shape for value in (open_, high, low, volume, trade_count)
    ):
        raise ValueError("daily arrays must be aligned [date, name]")
    wealth_seen = np.asarray(shareholder_wealth_valid, dtype=np.bool_)
    raw_high_, raw_low_, raw_close_ = tuple(
        np.asarray(value) for value in (raw_high, raw_low, raw_close)
    )
    price_seen = np.asarray(price_observed, dtype=np.bool_)
    history_seen = np.asarray(history_observed, dtype=np.bool_)
    activity_seen = np.asarray(activity_valid, dtype=np.bool_)
    membership = np.asarray(active, dtype=np.bool_)
    if (
        wealth_seen.shape != close.shape
        or any(
            value.shape != close.shape for value in (raw_high_, raw_low_, raw_close_)
        )
        or price_seen.shape != close.shape
        or history_seen.shape != close.shape
        or activity_seen.shape != close.shape
        or membership.shape != close.shape
    ):
        raise ValueError(
            "shareholder-wealth, raw-price, history, activity, and active "
            "inputs are misaligned"
        )
    if np.any(
        activity_seen
        & (
            ~np.isfinite(volume)
            | (volume < 0.0)
            | ~np.isfinite(trade_count)
            | (trade_count < 0.0)
        )
    ):
        raise ValueError("valid slow activity must be finite and non-negative")
    ambiguous = (
        np.zeros(close.shape, dtype=np.bool_)
        if ambiguous_action is None
        else np.asarray(ambiguous_action, dtype=np.bool_)
    )
    if ambiguous.shape != close.shape:
        raise ValueError("ambiguous_action is misaligned")

    def assign(
        index: int, column: NDArray[np.floating], mask: NDArray[np.bool_]
    ) -> None:
        usable = np.asarray(mask, dtype=np.bool_) & np.isfinite(column)
        consume(index, np.asarray(column), usable)

    retained_returns: dict[int, tuple[NDArray[np.float64], NDArray[np.bool_]]] = {}
    for index, horizon in enumerate((1, 5, 21, 63, 126, 252)):
        result = exact_log_return(close, horizon, ambiguous)
        assign(index, *result)
        if horizon in {1, 5, 21, 252}:
            retained_returns[horizon] = result
        else:
            del result
    momentum = retained_returns[252][0] - retained_returns[21][0]
    assign(
        6,
        momentum,
        retained_returns[252][1] & retained_returns[21][1],
    )
    del momentum, retained_returns[252]

    for index, window in zip((7, 8, 9), (5, 20, 60), strict=True):
        yz_result = yang_zhang_volatility(open_, high, low, close, window, ambiguous)
        assign(index, *yz_result)
        if window == 5:
            yz_5 = yz_result
    vol_of_vol, vol_of_vol_valid = _rolling_stat(yz_5[0], 60, "std")
    assign(
        10,
        vol_of_vol,
        vol_of_vol_valid & _ambiguous_interval_clear(ambiguous, 64),
    )
    del vol_of_vol, vol_of_vol_valid, yz_5, yz_result
    skew, kurtosis, moments_valid = _rolling_moments(retained_returns[1][0], 60)
    moments_valid &= _ambiguous_interval_clear(ambiguous, 60)
    assign(11, skew, moments_valid)
    assign(12, kurtosis, moments_valid)
    maximum, maximum_valid = _rolling_stat(retained_returns[1][0], 21, "max")
    assign(
        13,
        maximum,
        maximum_valid & _ambiguous_interval_clear(ambiguous, 21),
    )
    rolling_high, rolling_high_valid = _rolling_stat(high, 252, "max")
    with np.errstate(divide="ignore", invalid="ignore"):
        high_distance = np.log(close / rolling_high)
    assign(
        14,
        high_distance,
        rolling_high_valid
        & wealth_seen
        & (close > 0)
        & _ambiguous_interval_clear(ambiguous, 251),
    )

    market = np.full(close.shape[0], np.nan, dtype=np.float64)
    for day in range(close.shape[0]):
        mask = retained_returns[1][1][day] & membership[day]
        if mask.any():
            market[day] = float(np.median(retained_returns[1][0][day, mask]))
    beta = np.full(close.shape, np.nan, dtype=np.float64)
    idio = np.full(close.shape, np.nan, dtype=np.float64)
    beta_valid = np.zeros(close.shape, dtype=np.bool_)
    for day in range(59, close.shape[0]):
        market_window = market[day - 59 : day + 1]
        for name in range(close.shape[1]):
            name_window = retained_returns[1][0][day - 59 : day + 1, name]
            mask = np.isfinite(market_window) & np.isfinite(name_window)
            if (
                int(mask.sum()) < math.ceil(0.8 * 60)
                or np.var(market_window[mask]) <= 0
            ):
                continue
            coefficient = float(
                np.cov(name_window[mask], market_window[mask], ddof=0)[0, 1]
                / np.var(market_window[mask])
            )
            residual = name_window[mask] - coefficient * market_window[mask]
            beta[day, name] = coefficient
            idio[day, name] = float(np.std(residual))
            beta_valid[day, name] = True
    beta_valid &= _ambiguous_interval_clear(ambiguous, 60)
    assign(15, beta, beta_valid)
    assign(16, idio, beta_valid)

    volume_for_window = np.where(activity_seen, volume, np.nan)
    volume_mean, volume_window_valid = _rolling_stat(
        volume_for_window, 20, "mean", minimum=20
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        log_volume = np.log(volume_mean)
    assign(17, log_volume, volume_window_valid & (volume_mean > 0))
    volume_std, _ = _rolling_stat(volume_for_window, 20, "std", minimum=20)
    with np.errstate(divide="ignore", invalid="ignore"):
        volume_z = (volume - volume_mean) / volume_std
    assign(
        18,
        volume_z,
        activity_seen & volume_window_valid & (volume_std > 0),
    )
    amihud_daily = np.full(close.shape, np.nan, dtype=np.float64)
    amihud_source_valid = (
        activity_seen & retained_returns[1][1] & np.isfinite(volume) & (volume > 0.0)
    )
    amihud_daily[amihud_source_valid] = (
        np.abs(retained_returns[1][0][amihud_source_valid])
        / volume[amihud_source_valid]
    )
    amihud, amihud_valid = _rolling_stat(
        amihud_daily, 20, "mean", minimum=20
    )
    assign(
        19,
        amihud,
        amihud_valid & _ambiguous_interval_clear(ambiguous, 20),
    )
    trades_for_window = np.where(activity_seen, trade_count, np.nan)
    trades_mean, trades_valid = _rolling_stat(trades_for_window, 20, "mean", minimum=20)
    trades_std, _ = _rolling_stat(trades_for_window, 20, "std", minimum=20)
    with np.errstate(divide="ignore", invalid="ignore"):
        trades_z = (trade_count - trades_mean) / trades_std
    assign(20, trades_z, activity_seen & trades_valid & (trades_std > 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        turnover = volume / volume_mean
    assign(
        21,
        turnover,
        activity_seen & volume_window_valid & (volume_mean > 0),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        range_1 = np.log(raw_high_ / raw_low_)
    range_valid = price_seen & np.isfinite(range_1) & (raw_high_ > 0) & (raw_low_ > 0)
    assign(22, range_1, range_valid)
    range_5, range_5_valid = _rolling_high_low_range(
        high, low, wealth_seen, 5, ambiguous
    )
    assign(23, range_5, range_5_valid)
    close_location = np.full(close.shape, 0.5, dtype=np.float64)
    nonzero_range = price_seen & (raw_high_ > raw_low_)
    close_location[nonzero_range] = (
        raw_close_[nonzero_range] - raw_low_[nonzero_range]
    ) / (raw_high_[nonzero_range] - raw_low_[nonzero_range])
    assign(
        24,
        close_location,
        price_seen & (raw_high_ >= raw_low_) & np.isfinite(close_location),
    )
    history_age = np.zeros(close.shape, dtype=np.float64)
    history_left_censored = np.zeros(close.shape, dtype=np.float64)
    history_valid = np.zeros(close.shape, dtype=np.bool_)
    for name in range(close.shape[1]):
        slots = np.flatnonzero(history_seen[:, name])
        if slots.size:
            indices = np.arange(slots[0], close.shape[0])
            history_age[indices, name] = indices - slots[0]
            history_left_censored[indices, name] = float(slots[0] == 0)
            history_valid[indices, name] = True
    assign(25, history_age, history_valid)
    assign(26, history_left_censored, history_valid)

    daily_residual = retained_returns[1][0].copy()
    residual_valid = retained_returns[1][1] & membership
    for day in range(close.shape[0]):
        mask = residual_valid[day] & membership[day]
        if mask.any():
            daily_residual[day, mask] -= np.median(daily_residual[day, mask])
    labels = (
        monthly_cluster_labels(
            dates, daily_residual, residual_valid, membership, cluster_count=12
        )
        if cluster_labels is None
        else np.asarray(cluster_labels, dtype=np.int16)
    )
    if labels.shape != close.shape:
        raise ValueError("cluster_labels must have shape [date, name]")
    peer_values, peer_valid = _peer_features(
        retained_returns[5][0],
        retained_returns[5][1],
        retained_returns[21][0],
        retained_returns[21][1],
        labels,
        membership,
    )
    for offset in range(5):
        assign(27 + offset, peer_values[..., offset], peer_valid[..., offset])
    return labels


def build_slow_features(
    shareholder_wealth_open: NDArray[np.floating],
    shareholder_wealth_high: NDArray[np.floating],
    shareholder_wealth_low: NDArray[np.floating],
    shareholder_wealth_close: NDArray[np.floating],
    volume_brl: NDArray[np.floating],
    trades: NDArray[np.floating],
    shareholder_wealth_valid: NDArray[np.bool_],
    active: NDArray[np.bool_],
    dates: Sequence[object],
    *,
    raw_high: NDArray[np.floating],
    raw_low: NDArray[np.floating],
    raw_close: NDArray[np.floating],
    price_observed: NDArray[np.bool_],
    history_observed: NDArray[np.bool_],
    activity_valid: NDArray[np.bool_],
    cluster_labels: NDArray[np.integer] | None = None,
    ambiguous_action: NDArray[np.bool_] | None = None,
) -> SlowFeatureResult:
    """Convenience in-memory wrapper around the streamed slow producer."""

    shape = np.asarray(shareholder_wealth_close).shape
    if len(shape) != 2:
        raise ValueError("shareholder_wealth_close must be [date, name]")
    values = np.zeros((*shape, len(SLOW_FEATURES)), dtype=np.float32)
    valid = np.zeros(values.shape, dtype=np.bool_)

    def consume(
        index: int, column: NDArray[np.floating], mask: NDArray[np.bool_]
    ) -> None:
        values[..., index][mask] = np.asarray(column)[mask].astype(np.float32)
        valid[..., index] = mask

    labels = build_slow_features_into(
        shareholder_wealth_open,
        shareholder_wealth_high,
        shareholder_wealth_low,
        shareholder_wealth_close,
        volume_brl,
        trades,
        shareholder_wealth_valid,
        active,
        dates,
        raw_high=raw_high,
        raw_low=raw_low,
        raw_close=raw_close,
        price_observed=price_observed,
        history_observed=history_observed,
        activity_valid=activity_valid,
        consume=consume,
        cluster_labels=cluster_labels,
        ambiguous_action=ambiguous_action,
    )
    return SlowFeatureResult(values=values, valid=valid, cluster_labels=labels)
