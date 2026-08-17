from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import polars as pl
from numpy.typing import NDArray

from brazil_rv.modeling.contract import (
    PROJECT_ROOT,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
)

from .contract import (
    DECISION_EQUITY_INDICES,
    DYNAMIC_CHANNELS,
    EQUITY_SESSION_MINUTES,
    EQUITY_SESSION_START_MINUTE,
    EXPECTED_EQUITIES,
    PRICE_FEATURE_CLIP,
    REALIZED_VOL_LOG_CLIP,
    REALIZED_VOL_LOG_FLOOR,
    REALIZED_VOL_MIN_FRACTION,
    RETURN_WINDOWS,
)
from .io import (
    SOURCE_COLUMNS,
    cotahist_files,
    dense_grid,
    load_assignments,
    load_market_dates_and_security_dates,
    prepare_session_bars,
    read_research_interval,
    validate_physical_source_identity,
    validate_source_date_isolation,
)
from .transforms import build_dynamic_features, build_equity_features

PROFILE_SCHEMA = "EQUITY_INTRADAY_VARIANCE_PROFILE_V1"
VARIANT_SCHEMA = "EQUITY_INTRADAY_NORMALIZATION_OVERLAY_V1"
PROFILE_BIN_MINUTES = 30
PROFILE_BIN_COUNT = math.ceil(EQUITY_SESSION_MINUTES / PROFILE_BIN_MINUTES)
VISIBLE_EQUITY_MINUTES = max(DECISION_EQUITY_INDICES)
DECISION_FEATURE_MINUTES = tuple(value - 1 for value in DECISION_EQUITY_INDICES)

ARMS = {
    "legacy_daily_vol": 0.0,
    "equity_tod_half": 0.5,
    "equity_tod_full": 1.0,
}

# Return breadth/ranks remain parent-bound because the seasonal factor is positive
# and common to every equity at a date/minute. The realized-volatility rank is
# rebuilt because missing paths can produce different effective corrected windows.
AFFECTED_DYNAMIC_CHANNELS = (
    0,
    1,
    2,
    3,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    16,
    17,
    20,
    21,
    25,
)
AFFECTED_PEER_CHANNELS = (0, 1, 4, 5)
INVARIANT_DYNAMIC_CHANNELS = tuple(
    channel
    for channel in range(len(DYNAMIC_CHANNELS))
    if channel not in AFFECTED_DYNAMIC_CHANNELS
)


@dataclass(frozen=True)
class ProfileConfig:
    bin_minutes: int = PROFILE_BIN_MINUTES
    prior_session_equivalents: float = 20.0
    relative_variance_lower_bound: float = 0.25
    relative_variance_upper_bound: float = 4.0
    opening_bin: int = 0
    midday_bin: int = 4

    def __post_init__(self) -> None:
        if self.bin_minutes != PROFILE_BIN_MINUTES:
            raise ValueError("The first experiment requires 30-minute bins")
        if self.prior_session_equivalents <= 0.0:
            raise ValueError("Profile prior must be positive")
        if not (
            0.0
            < self.relative_variance_lower_bound
            <= 1.0
            <= self.relative_variance_upper_bound
        ):
            raise ValueError("Relative-variance safety bounds must bracket one")
        if not 0 <= self.opening_bin < PROFILE_BIN_COUNT:
            raise ValueError("Opening bin is outside the equity session")
        if not 0 <= self.midday_bin < PROFILE_BIN_COUNT:
            raise ValueError("Midday bin is outside the equity session")


@dataclass(frozen=True)
class CausalProfile:
    relative_variance: NDArray[np.float64]
    historical_profile_days: NDArray[np.int32]
    shrinkage_weight: NDArray[np.float64]
    historical_observation_count: NDArray[np.int64]
    daily_variance: NDArray[np.float64]
    daily_observation_count: NDArray[np.int64]


@dataclass(frozen=True)
class EquitySourceContext:
    parent: Path
    manifest: dict[str, object]
    assignments: pl.DataFrame
    market_dates: tuple[date, ...]
    accepted_dates: dict[str, frozenset[date]]
    slot_by_security: dict[str, int]
    allowed_date_count: int


@dataclass(frozen=True)
class ReconstructedEquity:
    slot: int
    security_id: str
    source_path: Path
    raw_grid: NDArray[np.float64]
    observed: NDArray[np.bool_]
    dynamic: NDArray[np.float32]
    dynamic_valid: NDArray[np.bool_]
    sigma: NDArray[np.float64]
    data_ready: NDArray[np.bool_]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_commit() -> str:
    repository = Path(__file__).resolve().parents[4]
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def write_canonical_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_directory(output_dir: Path, writer: Callable[[Path], None]) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    partial = output_dir.with_name(f"{output_dir.name}.{uuid4().hex}.partial")
    partial.mkdir(parents=True)
    try:
        writer(partial)
        os.replace(partial, output_dir)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return output_dir


def workspace_path(value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path.resolve()
    normalized = str(value).replace("\\", "/")
    marker = "/quant-data/"
    position = normalized.lower().find(marker)
    if position >= 0:
        rebased = PROJECT_ROOT / normalized[position + 1 :]
        if rebased.exists():
            return rebased.resolve()
    raise FileNotFoundError(path)


def _bounded_unit_mean(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    lower: float,
    upper: float,
) -> NDArray[np.float64]:
    """Project positive values into a finite box with weighted mean exactly one."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    usable = weights > 0.0
    if not usable.any():
        return np.ones_like(values)

    def mean(scale: float) -> float:
        bounded = np.clip(scale * values, lower, upper)
        return float(np.average(bounded[usable], weights=weights[usable]))

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
    """Emit each training profile before its session update; freeze at train end."""
    variances = np.asarray(daily_variance, dtype=np.float64)
    observations = np.asarray(daily_observation_count, dtype=np.int64)
    expected = (len(trade_dates), PROFILE_BIN_COUNT)
    if variances.shape != observations.shape or variances.shape != expected:
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
            reference_weights = np.where(known, observation_total, 0).astype(np.float64)
            if not (reference_weights > 0.0).any():
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

    if not np.isfinite(q).all() or np.any(q <= 0.0):
        raise RuntimeError("Seasonal profile is not finite and positive")
    validation = np.asarray(
        [VALIDATION_START <= value <= VALIDATION_END for value in trade_dates]
    )
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
    """Return date/bin moments for unclipped legacy-normalized close moves."""
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


def _seasonal_minute_variance(
    q_by_day: NDArray[np.float64], minute_count: int, gamma: float
) -> NDArray[np.float64]:
    bins = np.arange(minute_count, dtype=np.int64) // PROFILE_BIN_MINUTES
    return np.power(q_by_day[:, bins], gamma)


def build_seasonal_dynamic_features(
    raw_grid: NDArray[np.float64],
    observed: NDArray[np.bool_],
    data_ready: NDArray[np.bool_],
    sigma: NDArray[np.float64],
    q_by_day: NDArray[np.float64],
    gamma: float,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Rebuild every equity intraday price feature from the underlying path."""
    if gamma not in ARMS.values():
        raise ValueError(f"Unsupported seasonal normalization strength: {gamma}")
    legacy, validity = build_dynamic_features(
        raw_grid,
        observed,
        data_ready,
        sigma,
        is_rate=False,
        first_observed_open=True,
    )
    if gamma == 0.0:
        return legacy, validity
    if q_by_day.shape != (raw_grid.shape[0], PROFILE_BIN_COUNT):
        raise ValueError("Seasonal profile does not match the raw date axis")
    output = legacy.copy()
    output[..., :4] = 0.0
    output[..., 6:13] = 0.0
    minute_variance = _seasonal_minute_variance(q_by_day, raw_grid.shape[1], gamma)
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
            # Element j is the return ending in scheduled minute j + 1.
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


def dynamic_validity_from_observed(
    observed: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    equity_count, minute_count = observed.shape
    validity = np.zeros((equity_count, minute_count, 4), dtype=bool)
    minutes = np.arange(minute_count)
    for destination, window in ((0, 15), (1, 60)):
        usable = minutes >= window
        current = minutes[usable]
        validity[:, current, destination] = (
            observed[:, current] & observed[:, current - window]
        )
    for minute_idx in range(30, minute_count):
        adjacent = (
            observed[:, minute_idx - 29 : minute_idx + 1]
            & observed[:, minute_idx - 30 : minute_idx]
        )
        validity[:, minute_idx, 3] = observed[:, minute_idx] & (
            adjacent.sum(axis=1) >= 24
        )
    return validity


def load_source_context(parent: Path) -> EquitySourceContext:
    parent = parent.resolve()
    manifest = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))
    date_index = pl.read_parquet(parent / "date_index.parquet").sort("date_idx")
    market_dates = tuple(date_index.get_column("trade_date").to_list())
    allowed_date_count = sum(value <= VALIDATION_END for value in market_dates)
    if not allowed_date_count or market_dates[allowed_date_count - 1] != VALIDATION_END:
        raise ValueError("Parent date axis does not end validation exactly")
    canonical = manifest["canonical_inputs"]
    assignments_dir = workspace_path(
        canonical["accepted_xp_assignments"]["resolved_path"]
    )
    cotahist_dir = workspace_path(canonical["parsed_cotahist"]["resolved_path"])
    universe_dir = workspace_path(canonical["point_in_time_universe"]["resolved_path"])
    assignments = load_assignments(assignments_dir)
    security_ids = tuple(assignments.get_column("security_id").to_list())
    research_start, _ = read_research_interval(universe_dir)
    reconstructed_dates, accepted_dates = load_market_dates_and_security_dates(
        cotahist_files(cotahist_dir),
        security_ids,
        research_start,
        VALIDATION_END,
    )
    if reconstructed_dates != market_dates[:allowed_date_count]:
        raise ValueError("Parent date axis differs from canonical COTAHIST")
    validate_source_date_isolation(assignments, accepted_dates)
    equity_index = pl.read_parquet(parent / "equity_index.parquet").sort("equity_slot")
    if tuple(equity_index["security_id"]) != security_ids:
        raise ValueError("Assignment order differs from the parent equity axis")
    return EquitySourceContext(
        parent,
        manifest,
        assignments,
        market_dates,
        accepted_dates,
        {security_id: slot for slot, security_id in enumerate(security_ids)},
        allowed_date_count,
    )


def _load_source_through_validation(path: Path, first_date: date) -> pl.DataFrame:
    return (
        pl.scan_parquet(path)
        .select(SOURCE_COLUMNS)
        .filter(pl.col("ts_exchange").dt.date().is_between(first_date, VALIDATION_END))
        .collect()
    )


def iter_reconstructed_equities(
    context: EquitySourceContext,
) -> Iterator[ReconstructedEquity]:
    market_dates = context.market_dates[: context.allowed_date_count]
    all_dates = frozenset(market_dates)
    for group in context.assignments.partition_by("source_file", maintain_order=True):
        source_path = workspace_path(group.item(0, "source_file"))
        source = _load_source_through_validation(source_path, market_dates[0])
        validate_physical_source_identity(group, source, source_path)
        group_ids = tuple(group.get_column("security_id").to_list())
        allowed_dates = frozenset().union(
            *(context.accepted_dates[security_id] for security_id in group_ids)
        )
        if not allowed_dates <= all_dates:
            raise ValueError("Accepted equity dates enter the held-out period")
        session_bars = prepare_session_bars(
            source,
            source_path,
            allowed_dates,
            market_dates,
            EQUITY_SESSION_START_MINUTE,
            EQUITY_SESSION_MINUTES,
        )
        for assignment in group.iter_rows(named=True):
            security_id = str(assignment["security_id"])
            bars = session_bars.filter(
                pl.col("trade_date").is_in(tuple(context.accepted_dates[security_id]))
            )
            if bars.is_empty():
                raise ValueError(f"No accepted bars for {security_id}")
            raw_grid, observed = dense_grid(
                bars,
                context.allowed_date_count,
                EQUITY_SESSION_MINUTES,
            )
            identity_day = np.fromiter(
                (
                    assignment["first_overlap_date"]
                    <= trade_date
                    <= assignment["last_overlap_date"]
                    for trade_date in market_dates
                ),
                dtype=bool,
                count=context.allowed_date_count,
            )
            result = build_equity_features(
                raw_grid,
                observed,
                identity_day,
                market_dates=market_dates,
            )
            yield ReconstructedEquity(
                context.slot_by_security[security_id],
                security_id,
                source_path,
                raw_grid,
                observed,
                result.dynamic,
                result.dynamic_valid,
                result.sigma,
                result.data_ready,
            )


def parent_identity(context: EquitySourceContext) -> dict[str, object]:
    digest = hashlib.sha256()
    for filename in ("manifest.json", "feature_schema.json", "sample_index.parquet"):
        with (context.parent / filename).open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    return {
        "path": str(context.parent),
        "contract_version": context.manifest["contract_version"],
        "metadata_sha256": digest.hexdigest(),
    }


def parent_artifact_hashes(context: EquitySourceContext) -> dict[str, str]:
    filenames = sorted(
        {
            *context.manifest["outputs"],
            "manifest.json",
            "feature_schema.json",
            "date_index.parquet",
            "equity_index.parquet",
            "sample_index.parquet",
            "peer_feature_build_audit.json",
        }
    )
    return {filename: sha256_file(context.parent / filename) for filename in filenames}


def equity_source_hashes(context: EquitySourceContext) -> dict[str, str]:
    paths = sorted(
        {workspace_path(value) for value in context.assignments["source_file"]},
        key=str,
    )
    return {str(path): sha256_file(path) for path in paths}


def build_equity_tod_profile(
    parent: Path,
    output_dir: Path,
    *,
    config: ProfileConfig = ProfileConfig(),
) -> Path:
    """Build the shared causal profile without reading held-out observations."""
    context = load_source_context(parent)
    membership = np.load(
        context.parent / "equity_membership.npy", mmap_mode="r", allow_pickle=False
    )[: context.allowed_date_count]
    parent_ready = np.load(
        context.parent / "equity_data_ready.npy", mmap_mode="r", allow_pickle=False
    )[: context.allowed_date_count]
    parent_dynamic = np.load(
        context.parent / "equity_features.npy", mmap_mode="r", allow_pickle=False
    )
    total = np.zeros((context.allowed_date_count, PROFILE_BIN_COUNT), dtype=np.float64)
    total_sq = np.zeros_like(total)
    count = np.zeros(total.shape, dtype=np.int64)
    seen = np.zeros(EXPECTED_EQUITIES, dtype=bool)
    for equity in iter_reconstructed_equities(context):
        if seen[equity.slot]:
            raise ValueError(f"Equity slot {equity.slot} was assigned twice")
        seen[equity.slot] = True
        if not np.array_equal(equity.data_ready, parent_ready[:, equity.slot]):
            raise ValueError(f"Parent readiness drift for slot {equity.slot}")
        if not np.array_equal(
            equity.dynamic[..., :16],
            parent_dynamic[: context.allowed_date_count, equity.slot, :, :16],
        ):
            raise ValueError(f"Parent base-feature drift for slot {equity.slot}")
        values = daily_close_move_statistics(
            equity.raw_grid,
            equity.observed,
            np.asarray(membership[:, equity.slot], dtype=bool),
            equity.data_ready,
            equity.sigma,
        )
        total += values[0]
        total_sq += values[1]
        count += values[2]
    if not seen.all():
        raise ValueError("Not every parent equity slot was reconstructed")

    profile = estimate_causal_profile(
        variance_from_sufficient_statistics(total, total_sq, count),
        count,
        context.market_dates[: context.allowed_date_count],
        config,
    )
    parent_hashes = parent_artifact_hashes(context)
    source_hashes = equity_source_hashes(context)

    def write(partial: Path) -> None:
        q_path = partial / "equity_tod_profile.npy"
        np.save(q_path, profile.relative_variance, allow_pickle=False)
        rows: list[dict[str, object]] = []
        for date_idx, trade_date in enumerate(
            context.market_dates[: context.allowed_date_count]
        ):
            split = (
                "train"
                if TRAIN_START <= trade_date <= TRAIN_END
                else (
                    "validation"
                    if VALIDATION_START <= trade_date <= VALIDATION_END
                    else "warmup_or_embargo"
                )
            )
            for bin_idx in range(PROFILE_BIN_COUNT):
                start = bin_idx * PROFILE_BIN_MINUTES
                q = profile.relative_variance[date_idx, bin_idx]
                rows.append(
                    {
                        "date_idx": date_idx,
                        "trade_date": trade_date,
                        "split": split,
                        "bin_idx": bin_idx,
                        "session_minute_start": start,
                        "session_minute_end_exclusive": min(
                            start + PROFILE_BIN_MINUTES, EQUITY_SESSION_MINUTES
                        ),
                        "relative_variance": q,
                        "standard_deviation_multiplier": math.sqrt(q),
                        "effective_historical_profile_days": int(
                            profile.historical_profile_days[date_idx, bin_idx]
                        ),
                        "shrinkage_weight": profile.shrinkage_weight[date_idx, bin_idx],
                        "historical_observation_count": int(
                            profile.historical_observation_count[date_idx, bin_idx]
                        ),
                        "current_daily_variance_estimate": profile.daily_variance[
                            date_idx, bin_idx
                        ],
                        "current_daily_observation_count": int(
                            profile.daily_observation_count[date_idx, bin_idx]
                        ),
                    }
                )
        csv_path = partial / "equity_tod_profile.csv"
        pl.DataFrame(rows).write_csv(csv_path)
        manifest = {
            "schema": PROFILE_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository_commit": repository_commit(),
            "parent_feature_store": parent_identity(context),
            "parent_artifact_sha256": parent_hashes,
            "equity_source_sha256": source_hashes,
            "configuration": asdict(config),
            "training_window": [str(TRAIN_START), str(TRAIN_END)],
            "validation_window": [str(VALIDATION_START), str(VALIDATION_END)],
            "training_profile_freeze_date": str(TRAIN_END),
            "profile_input": "unclipped_legacy_normalized_equity_close_moves",
            "historical_count_unit": "valid_session_bin_estimates",
            "current_date_update_rule": "emit_then_update",
            "validation_update_rule": "frozen_training_end_profile",
            "test_accessed": False,
            "date_count": context.allowed_date_count,
            "bin_count": PROFILE_BIN_COUNT,
            "artifacts": {
                q_path.name: sha256_file(q_path),
                csv_path.name: sha256_file(csv_path),
            },
        }
        write_canonical_json(partial / "equity_tod_profile.json", manifest)

    return atomic_directory(output_dir, write)


def load_equity_tod_profile(
    profile_dir: Path,
) -> tuple[dict[str, object], NDArray[np.float64]]:
    manifest_path = profile_dir / "equity_tod_profile.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != PROFILE_SCHEMA
        or manifest.get("test_accessed") is not False
    ):
        raise ValueError("Invalid equity time-of-day profile manifest")
    q_path = profile_dir / "equity_tod_profile.npy"
    csv_path = profile_dir / "equity_tod_profile.csv"
    for path in (q_path, csv_path):
        if sha256_file(path) != manifest["artifacts"][path.name]:
            raise ValueError(f"Profile artifact hash mismatch: {path}")
    q = np.load(q_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (int(manifest["date_count"]), int(manifest["bin_count"]))
    if q.shape != expected_shape or q.dtype != np.dtype(np.float64):
        raise ValueError("Seasonal profile array has the wrong shape or dtype")
    if not np.isfinite(q).all() or np.any(q <= 0.0):
        raise ValueError("Seasonal profile contains an invalid value")
    return manifest, q
