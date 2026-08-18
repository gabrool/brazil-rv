from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from brazil_rv.modeling.contract import (
    TRAIN_END,
    VALIDATION_END,
    VALIDATION_START,
)

from .contract import (
    EQUITY_SESSION_MINUTES,
    PRICE_FEATURE_CLIP,
    REALIZED_VOL_LOG_CLIP,
    REALIZED_VOL_LOG_FLOOR,
    REALIZED_VOL_MIN_FRACTION,
    RETURN_WINDOWS,
)
from .transforms import build_dynamic_features

PROFILE_SCHEMA = "EQUITY_CAUSAL_TOD_PROFILE"
PROFILE_BIN_MINUTES = 30
PROFILE_BIN_COUNT = math.ceil(EQUITY_SESSION_MINUTES / PROFILE_BIN_MINUTES)


@dataclass(frozen=True)
class ProfileConfig:
    bin_minutes: int = PROFILE_BIN_MINUTES
    prior_session_equivalents: float = 20.0
    relative_variance_lower_bound: float = 0.25
    relative_variance_upper_bound: float = 4.0


@dataclass(frozen=True)
class CausalProfile:
    relative_variance: NDArray[np.float64]
    historical_profile_days: NDArray[np.int32]
    shrinkage_weight: NDArray[np.float64]
    historical_observation_count: NDArray[np.int64]
    daily_variance: NDArray[np.float64]
    daily_observation_count: NDArray[np.int64]


def _bounded_unit_mean(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    lower: float,
    upper: float,
) -> NDArray[np.float64]:
    usable = weights > 0.0
    if not usable.any():
        return np.ones_like(values)

    def mean(scale: float) -> float:
        return float(
            np.average(
                np.clip(scale * values, lower, upper)[usable],
                weights=weights[usable],
            )
        )

    low_scale, high_scale = 0.0, 1.0
    while mean(high_scale) < 1.0:
        high_scale *= 2.0
    for _ in range(80):
        middle = 0.5 * (low_scale + high_scale)
        if mean(middle) < 1.0:
            low_scale = middle
        else:
            high_scale = middle
    projected = np.clip(high_scale * values, lower, upper)
    if not np.isclose(
        np.average(projected[usable], weights=weights[usable]),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("Bounded seasonal profile is not unit normalized")
    return projected


def estimate_causal_profile(
    daily_variance: NDArray[np.float64],
    daily_observation_count: NDArray[np.int64],
    trade_dates: tuple[date, ...],
    config: ProfileConfig = ProfileConfig(),
) -> CausalProfile:
    """Emit the profile before each training-session update and freeze after train."""
    variances = np.asarray(daily_variance, dtype=np.float64)
    observations = np.asarray(daily_observation_count, dtype=np.int64)
    expected = (len(trade_dates), PROFILE_BIN_COUNT)
    if variances.shape != expected or observations.shape != expected:
        raise ValueError("Daily profile inputs do not match the date/bin contract")
    valid_daily = (observations >= 2) & np.isfinite(variances) & (variances > 0.0)
    q = np.ones(expected, dtype=np.float64)
    profile_days = np.zeros(expected, dtype=np.int32)
    shrinkage = np.zeros(expected, dtype=np.float64)
    historical_observations = np.zeros(expected, dtype=np.int64)
    variance_total = np.zeros(PROFILE_BIN_COUNT, dtype=np.float64)
    day_count = np.zeros(PROFILE_BIN_COUNT, dtype=np.int32)
    observation_total = np.zeros(PROFILE_BIN_COUNT, dtype=np.int64)

    def current() -> NDArray[np.float64]:
        means = np.divide(
            variance_total,
            day_count,
            out=np.ones(PROFILE_BIN_COUNT, dtype=np.float64),
            where=day_count > 0,
        )
        known = day_count > 0
        if known.any():
            reference_weights = observation_total.astype(np.float64)
            if not (reference_weights[known] > 0.0).any():
                reference_weights = day_count.astype(np.float64)
            reference = float(
                np.average(means[known], weights=reference_weights[known])
            )
            relative = np.where(known, means / reference, 1.0)
        else:
            relative = np.ones(PROFILE_BIN_COUNT, dtype=np.float64)
        prior = config.prior_session_equivalents
        shrunk = (day_count * relative + prior) / (day_count + prior)
        weights = np.where(observation_total > 0, observation_total, day_count).astype(
            np.float64
        )
        return _bounded_unit_mean(
            shrunk,
            weights,
            config.relative_variance_lower_bound,
            config.relative_variance_upper_bound,
        )

    frozen: NDArray[np.float64] | None = None
    frozen_days: NDArray[np.int32] | None = None
    frozen_observations: NDArray[np.int64] | None = None
    for date_idx, trade_date in enumerate(trade_dates):
        if trade_date <= TRAIN_END:
            emitted = current()
            emitted_days = day_count.copy()
            emitted_observations = observation_total.copy()
        else:
            if frozen is None:
                frozen = current()
                frozen_days = day_count.copy()
                frozen_observations = observation_total.copy()
            emitted = frozen
            emitted_days = frozen_days
            emitted_observations = frozen_observations
        assert emitted_days is not None and emitted_observations is not None
        q[date_idx] = emitted
        profile_days[date_idx] = emitted_days
        shrinkage[date_idx] = emitted_days / (
            emitted_days + config.prior_session_equivalents
        )
        historical_observations[date_idx] = emitted_observations
        if trade_date <= TRAIN_END:
            valid = valid_daily[date_idx]
            variance_total[valid] += variances[date_idx, valid]
            day_count[valid] += 1
            observation_total[valid] += observations[date_idx, valid]

    validation = np.asarray(
        [VALIDATION_START <= value <= VALIDATION_END for value in trade_dates]
    )
    if not np.isfinite(q).all() or np.any(q <= 0.0):
        raise RuntimeError("Seasonal profile is not finite and positive")
    if validation.any() and not np.all(q[validation] == q[validation][0]):
        raise RuntimeError("Validation profile is not frozen at training end")
    return CausalProfile(
        q,
        profile_days,
        shrinkage,
        historical_observations,
        variances,
        observations,
    )


def daily_close_move_statistics(
    raw_grid: NDArray[np.float64],
    observed: NDArray[np.bool_],
    membership: NDArray[np.bool_],
    data_ready: NDArray[np.bool_],
    sigma: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Return date/bin moments for unclipped daily-volatility-normalized moves."""
    date_count, minute_count, field_count = raw_grid.shape
    if (
        field_count != 5
        or minute_count != EQUITY_SESSION_MINUTES
        or observed.shape != (date_count, minute_count)
        or membership.shape != (date_count,)
        or data_ready.shape != (date_count,)
        or sigma.shape != (date_count,)
    ):
        raise ValueError("Close-move statistic inputs are misaligned")
    total = np.zeros((date_count, PROFILE_BIN_COUNT), dtype=np.float64)
    total_sq = np.zeros_like(total)
    count = np.zeros(total.shape, dtype=np.int64)
    for date_idx in np.flatnonzero(membership & data_ready & (sigma > 0.0)):
        positions = np.flatnonzero(observed[date_idx])
        if positions.size == 0:
            continue
        anchors = np.empty(positions.size, dtype=np.float64)
        anchors[0] = raw_grid[date_idx, positions[0], 0]
        anchors[1:] = raw_grid[date_idx, positions[:-1], 3]
        moves = np.log(raw_grid[date_idx, positions, 3] / anchors) / sigma[date_idx]
        bins = positions // PROFILE_BIN_MINUTES
        np.add.at(total[date_idx], bins, moves)
        np.add.at(total_sq[date_idx], bins, moves**2)
        np.add.at(count[date_idx], bins, 1)
    return total, total_sq, count


def variance_from_sufficient_statistics(
    total: NDArray[np.float64],
    total_sq: NDArray[np.float64],
    count: NDArray[np.int64],
) -> NDArray[np.float64]:
    if total.shape != total_sq.shape or total.shape != count.shape:
        raise ValueError("Variance sufficient statistics are misaligned")
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    variance = np.divide(total_sq, count, out=np.zeros_like(total), where=count > 0)
    return np.maximum(variance - mean**2, 0.0)


def build_full_tod_dynamic_features(
    raw_grid: NDArray[np.float64],
    observed: NDArray[np.bool_],
    data_ready: NDArray[np.bool_],
    sigma: NDArray[np.float64],
    q_by_day: NDArray[np.float64],
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Build the canonical full-strength time-of-day-normalized equity features."""
    legacy, validity = build_dynamic_features(
        raw_grid,
        observed,
        data_ready,
        sigma,
        is_rate=False,
        first_observed_open=True,
    )
    if q_by_day.shape != (raw_grid.shape[0], PROFILE_BIN_COUNT):
        raise ValueError("Seasonal profile does not match the raw date axis")
    output = legacy.copy()
    output[..., :4] = 0.0
    output[..., 6:13] = 0.0
    bins = np.arange(raw_grid.shape[1], dtype=np.int64) // PROFILE_BIN_MINUTES
    minute_variance = q_by_day[:, bins]
    minute_scale = np.sqrt(minute_variance)

    for date_idx in np.flatnonzero(data_ready & (sigma > 0.0)):
        positions = np.flatnonzero(observed[date_idx])
        if positions.size == 0:
            continue
        day = raw_grid[date_idx]
        prices = day[positions, :4]
        anchors = np.empty(positions.size, dtype=np.float64)
        anchors[0] = prices[0, 0]
        anchors[1:] = day[positions[:-1], 3]
        moves = np.log(prices / anchors[:, None])
        output[date_idx, positions, :4] = np.clip(
            moves / (sigma[date_idx] * minute_scale[date_idx, positions, None]),
            -PRICE_FEATURE_CLIP,
            PRICE_FEATURE_CLIP,
        ).astype(np.float32)

        open_position = int(positions[0])
        cumulative_variance = np.concatenate(
            ([0.0], np.cumsum(minute_variance[date_idx], dtype=np.float64))
        )
        denominator = np.sqrt(
            cumulative_variance[positions + 1] - cumulative_variance[open_position]
        )
        output[date_idx, positions, 6] = np.clip(
            np.log(day[positions, 3] / day[open_position, 0])
            / (sigma[date_idx] * denominator),
            -PRICE_FEATURE_CLIP,
            PRICE_FEATURE_CLIP,
        ).astype(np.float32)

        for channel, window in zip((7, 8, 9), RETURN_WINDOWS, strict=True):
            endpoints = positions - window
            exact = endpoints >= 0
            exact[exact] &= observed[date_idx, endpoints[exact]]
            if not exact.any():
                continue
            current = positions[exact]
            prior = endpoints[exact]
            integrated = (
                cumulative_variance[current + 1] - cumulative_variance[prior + 1]
            )
            returns = np.log(day[current, 3] / day[prior, 3])
            output[date_idx, current, channel] = np.clip(
                returns / (sigma[date_idx] * np.sqrt(integrated)),
                -PRICE_FEATURE_CLIP,
                PRICE_FEATURE_CLIP,
            ).astype(np.float32)

        adjacent = observed[date_idx, 1:] & observed[date_idx, :-1]
        corrected = np.zeros(raw_grid.shape[1] - 1, dtype=np.float64)
        if adjacent.any():
            raw = np.log(day[1:, 3][adjacent] / day[:-1, 3][adjacent])
            corrected[adjacent] = raw / minute_scale[date_idx, 1:][adjacent]
        for channel, window in zip((10, 11, 12), RETURN_WINDOWS, strict=True):
            minimum = int(math.ceil(REALIZED_VOL_MIN_FRACTION * window))
            for minute_idx in positions[positions >= window]:
                window_valid = adjacent[minute_idx - window : minute_idx]
                if int(window_valid.sum()) < minimum:
                    continue
                values = corrected[minute_idx - window : minute_idx][window_valid]
                rms = math.sqrt(float(np.mean(values**2)))
                output[date_idx, minute_idx, channel] = np.float32(
                    np.clip(
                        math.log(max(rms / sigma[date_idx], REALIZED_VOL_LOG_FLOOR)),
                        -REALIZED_VOL_LOG_CLIP,
                        REALIZED_VOL_LOG_CLIP,
                    )
                )
    return output, validity


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_profile_artifact(
    output_dir: Path,
    profile: CausalProfile,
    trade_dates: tuple[date, ...],
    config: ProfileConfig = ProfileConfig(),
) -> dict[str, object]:
    profile_path = output_dir / "equity_tod_profile.npy"
    np.save(profile_path, profile.relative_variance, allow_pickle=False)
    manifest = {
        "schema": PROFILE_SCHEMA,
        "configuration": asdict(config),
        "profile_input": "unclipped_daily_volatility_normalized_equity_close_moves",
        "training_profile_freeze_date": str(TRAIN_END),
        "post_training_update_rule": "frozen_training_end_profile",
        "date_count": len(trade_dates),
        "first_date": str(trade_dates[0]),
        "last_date": str(trade_dates[-1]),
        "array": {
            "file": profile_path.name,
            "dtype": profile.relative_variance.dtype.name,
            "shape": list(profile.relative_variance.shape),
            "sha256": _sha256(profile_path),
        },
    }
    temporary = output_dir / "equity_tod_profile.json.tmp"
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, output_dir / "equity_tod_profile.json")
    return manifest


def validate_profile_artifact(
    features_dir: Path, trade_dates: tuple[date, ...]
) -> dict[str, object]:
    manifest = json.loads(
        (features_dir / "equity_tod_profile.json").read_text(encoding="utf-8")
    )
    path = features_dir / str(manifest["array"]["file"])
    q = np.load(path, allow_pickle=False)
    expected = (len(trade_dates), PROFILE_BIN_COUNT)
    if (
        manifest.get("schema") != PROFILE_SCHEMA
        or q.shape != expected
        or q.dtype != np.float64
        or manifest["array"]["sha256"] != _sha256(path)
    ):
        raise ValueError("Invalid causal TOD profile artifact")
    if not np.isfinite(q).all() or np.any(q < 0.25) or np.any(q > 4.0):
        raise ValueError("Causal TOD profile is outside its finite safety bounds")
    post_train = np.asarray([value > TRAIN_END for value in trade_dates])
    if post_train.any() and not np.all(q[post_train] == q[post_train][0]):
        raise ValueError("Causal TOD profile changes after the training cutoff")
    return manifest
