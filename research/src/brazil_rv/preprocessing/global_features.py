from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
from numpy.typing import NDArray

from .contract import (
    DECISION_GLOBAL_INDICES,
    DECISION_TIMES,
    GLOBAL_CONTEXT_FAMILIES,
    GLOBAL_CONTEXT_SYMBOLS,
    GLOBAL_QUOTE_DIRECTIONS,
    GLOBAL_SESSION_MINUTES,
    GLOBAL_SESSION_START_MINUTE,
    MAD_NORMALIZATION,
    PRICE_FEATURE_CLIP,
    PRICE_VOL_FLOOR,
    PRICE_VOL_REFERENCE,
    REALIZED_VOL_LOG_CLIP,
    REALIZED_VOL_LOG_FLOOR,
    REAL_VOLUME_LOG_CENTER,
    REAL_VOLUME_LOG_SCALE,
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
from .global_source import load_global_symbol
from .transforms import build_dynamic_features

B3_TIMEZONE = ZoneInfo("America/Sao_Paulo")
GLOBEX_TIMEZONE = ZoneInfo("America/Chicago")
GLOBEX_CLOSE_HOUR = 16
GLOBEX_SESSION_MINUTES = 23 * 60


@dataclass(frozen=True)
class GlobalInstrumentFeatures:
    dynamic: NDArray[np.float32]
    slow: NDArray[np.float32]
    data_ready: NDArray[np.bool_]
    context_index: pl.DataFrame
    coverage: pl.DataFrame


def _b3_timestamp(trade_date: date, value: time) -> datetime:
    return datetime.combine(trade_date, value, B3_TIMEZONE).astimezone(UTC)


def _calendar_features(trade_date: date) -> tuple[float, float, float, float]:
    weekday = trade_date.weekday()
    month_end = date(
        trade_date.year + (trade_date.month == 12),
        trade_date.month % 12 + 1,
        1,
    )
    quarter_month = 3 * ((trade_date.month - 1) // 3 + 1)
    next_quarter = (
        date(trade_date.year + 1, 1, 1)
        if quarter_month == 12
        else date(trade_date.year, quarter_month + 1, 1)
    )
    return (
        float(np.sin(2.0 * np.pi * weekday / 5.0)),
        float(np.cos(2.0 * np.pi * weekday / 5.0)),
        max(0.0, 1.0 - ((month_end - trade_date).days - 1) / 7.0),
        max(0.0, 1.0 - ((next_quarter - trade_date).days - 1) / 14.0),
    )


def build_global_grid(
    frame: pl.DataFrame, market_dates: tuple[date, ...]
) -> tuple[
    NDArray[np.float64],
    NDArray[np.bool_],
    NDArray[np.bool_],
    pl.DataFrame,
]:
    date_index = pl.DataFrame(
        {
            "trade_date": pl.Series(market_dates, dtype=pl.Date),
            "date_idx": pl.Series(range(len(market_dates)), dtype=pl.Int32),
        }
    )
    rows = (
        frame.with_columns(
            pl.col("ts_event_utc")
            .dt.convert_time_zone("America/Sao_Paulo")
            .alias("b3_ts")
        )
        .with_columns(
            pl.col("b3_ts").dt.date().alias("trade_date"),
            (
                pl.col("b3_ts").dt.hour().cast(pl.Int16) * 60
                + pl.col("b3_ts").dt.minute().cast(pl.Int16)
                - GLOBAL_SESSION_START_MINUTE
            ).alias("minute_idx"),
        )
        .filter(
            pl.col("trade_date").is_in(market_dates),
            pl.col("minute_idx").is_between(0, GLOBAL_SESSION_MINUTES - 1),
        )
        .join(date_index, on="trade_date", how="inner")
        .sort("ts_event_utc")
    )
    if rows.select("date_idx", "minute_idx").is_duplicated().any():
        raise ValueError("Duplicate normalized global row on the B3-aligned grid")
    raw = np.zeros((len(market_dates), GLOBAL_SESSION_MINUTES, 5), dtype=np.float64)
    observed = np.zeros(raw.shape[:2], dtype=bool)
    mapping_changed = np.zeros_like(observed)
    if rows.height:
        date_idx = rows["date_idx"].to_numpy()
        minute_idx = rows["minute_idx"].to_numpy()
        raw[date_idx, minute_idx] = rows.select(
            "open", "high", "low", "close", "volume"
        ).to_numpy()
        observed[date_idx, minute_idx] = True
        mapping_changed[date_idx, minute_idx] = rows["mapping_changed"].to_numpy()
    index = rows.select(
        "global_slot",
        "continuous_symbol",
        "family",
        "quote_direction",
        "date_idx",
        "trade_date",
        "minute_idx",
        "ts_event_utc",
        "bar_end_utc",
        "b3_ts",
        "instrument_id",
        "raw_symbol",
        "expiration_utc",
        "mapping_changed",
    )
    return raw, observed, mapping_changed, index


def _session_summaries(frame: pl.DataFrame) -> pl.DataFrame:
    rows = (
        frame.with_columns(
            pl.col("ts_event_utc")
            .dt.convert_time_zone("America/Chicago")
            .alias("globex_ts")
        )
        .with_columns(
            (pl.col("globex_ts") + pl.duration(hours=7)).dt.date().alias("session_date")
        )
        .sort("ts_event_utc")
        .with_columns(
            pl.col("close").shift(1).over("session_date").alias("previous_close"),
            pl.col("raw_symbol").shift(1).over("session_date").alias("previous_symbol"),
            pl.col("ts_event_utc").shift(1).over("session_date").alias("previous_ts"),
        )
        .with_columns(
            (
                (
                    pl.col("ts_event_utc") - pl.col("previous_ts")
                    == pl.duration(minutes=1)
                )
                & (pl.col("raw_symbol") == pl.col("previous_symbol"))
            ).alias("adjacent"),
            pl.when(
                (
                    pl.col("ts_event_utc") - pl.col("previous_ts")
                    == pl.duration(minutes=1)
                )
                & (pl.col("raw_symbol") == pl.col("previous_symbol"))
            )
            .then((pl.col("close") / pl.col("previous_close")).log())
            .otherwise(None)
            .alias("one_minute_return"),
        )
    )
    summaries = rows.group_by("session_date", maintain_order=True).agg(
        pl.col("ts_event_utc").first().alias("first_ts"),
        pl.col("bar_end_utc").last().alias("last_bar_end"),
        pl.col("open").first().alias("open"),
        pl.col("close").last().alias("close"),
        pl.col("raw_symbol").first().alias("open_symbol"),
        pl.col("raw_symbol").last().alias("close_symbol"),
        pl.col("mapping_changed").sum().alias("roll_count"),
        pl.col("volume").sum().alias("volume"),
        pl.len().alias("observed_minutes"),
        pl.col("one_minute_return").count().alias("return_count"),
        pl.col("one_minute_return").pow(2).mean().sqrt().alias("realized_vol"),
    )
    endpoint = rows.select(
        "session_date",
        pl.col("ts_event_utc").alias("endpoint_ts"),
        pl.col("close").alias("endpoint_close"),
        pl.col("raw_symbol").alias("endpoint_symbol"),
    )
    summaries = (
        summaries.with_columns(
            (pl.col("last_bar_end") - pl.duration(minutes=60))
            .cast(pl.Datetime("ns", "UTC"))
            .alias("endpoint_ts")
        )
        .join(endpoint, on=["session_date", "endpoint_ts"], how="left")
        .with_columns(
            (
                (pl.col("open_symbol") == pl.col("close_symbol"))
                & (pl.col("roll_count") == 0)
            ).alias("open_close_valid"),
            (
                pl.col("endpoint_close").is_not_null()
                & (pl.col("endpoint_symbol") == pl.col("close_symbol"))
                & (pl.col("roll_count") == 0)
            ).alias("last_60_valid"),
            (pl.col("observed_minutes") / GLOBEX_SESSION_MINUTES)
            .clip(0.0, 1.0)
            .alias("observed_fraction"),
        )
        .with_columns(
            pl.when(pl.col("open_close_valid"))
            .then((pl.col("close") / pl.col("open")).log())
            .otherwise(0.0)
            .alias("open_close"),
            pl.when(pl.col("last_60_valid"))
            .then((pl.col("close") / pl.col("endpoint_close")).log())
            .otherwise(0.0)
            .alias("last_60"),
        )
    )
    session_ends = [
        datetime.combine(value, time(GLOBEX_CLOSE_HOUR), GLOBEX_TIMEZONE).astimezone(
            UTC
        )
        for value in summaries["session_date"].to_list()
    ]
    summaries = summaries.with_columns(
        pl.Series("session_end_utc", session_ends, dtype=pl.Datetime("us", "UTC")),
        pl.col("close").shift(1).alias("previous_session_close"),
        pl.col("close_symbol").shift(1).alias("previous_session_symbol"),
    ).with_columns(
        (
            pl.col("previous_session_close").is_not_null()
            & (pl.col("close_symbol") == pl.col("previous_session_symbol"))
        ).alias("change_valid"),
        pl.when(
            pl.col("previous_session_close").is_not_null()
            & (pl.col("close_symbol") == pl.col("previous_session_symbol"))
        )
        .then((pl.col("close") / pl.col("previous_session_close")).log())
        .otherwise(0.0)
        .alias("change"),
    )
    return summaries.sort("session_end_utc")


def _volatility_state(
    summaries: pl.DataFrame, market_dates: tuple[date, ...]
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    ends = summaries["session_end_utc"].cast(pl.Int64).to_numpy()
    variance = summaries["realized_vol"].fill_null(0.0).to_numpy() ** 2
    valid = summaries["return_count"].to_numpy() >= 30
    sigma_before_session = np.zeros(summaries.height, dtype=np.float64)
    sigma_after_session = np.zeros(summaries.height, dtype=np.float64)
    warmup: list[float] = []
    ewma: float | None = None
    for index in range(summaries.height):
        if ewma is not None:
            sigma_before_session[index] = np.sqrt(max(ewma, PRICE_VOL_FLOOR**2))
        if valid[index]:
            if ewma is None:
                warmup.append(float(variance[index]))
                if len(warmup) == VOL_WARMUP_VALID_DAYS:
                    ewma = float(np.median(warmup))
            else:
                ewma = (1.0 - VOL_EWMA_ALPHA) * ewma + VOL_EWMA_ALPHA * float(
                    variance[index]
                )
        if ewma is not None:
            sigma_after_session[index] = np.sqrt(max(ewma, PRICE_VOL_FLOOR**2))
    cutoffs = np.asarray(
        [
            int(_b3_timestamp(value, time(4, 30)).timestamp() * 1_000_000)
            for value in market_dates
        ],
        dtype=np.int64,
    )
    completed = np.searchsorted(ends, cutoffs, side="right") - 1
    sigma = np.zeros(len(market_dates), dtype=np.float64)
    ready = completed >= 0
    sigma[ready] = sigma_after_session[completed[ready]]
    return sigma, completed, sigma_before_session


def _robust_volume_ratio(values: NDArray[np.float64], previous: int) -> float:
    history = values[max(0, previous - VOLUME_LOOKBACK_SESSIONS) : previous]
    history = history[history > 0]
    if history.size < VOLUME_MIN_OBSERVATIONS or values[previous] <= 0:
        return 0.0
    logs = np.log1p(history)
    median = np.median(logs)
    scale = max(MAD_NORMALIZATION * np.median(np.abs(logs - median)), VOLUME_MAD_FLOOR)
    return float(
        np.clip(
            (np.log1p(values[previous]) - median) / scale,
            -VOLUME_FEATURE_CLIP,
            VOLUME_FEATURE_CLIP,
        )
    )


def _expiry_scale(expiry: datetime | None, decision: datetime) -> float:
    if expiry is None:
        return 0.0
    days = max((expiry.date() - decision.date()).days, 0)
    return float(np.clip(np.log1p(days / 365.25) / np.log(11.0), 0.0, 1.0))


def _base_slow(
    summaries: pl.DataFrame,
    previous: int,
    sigma_before_session: NDArray[np.float64],
    sigma: float,
    trade_date: date,
) -> NDArray[np.float32]:
    output = np.zeros(32, dtype=np.float32)
    output[0] = np.float32(
        np.clip(np.log(sigma / PRICE_VOL_REFERENCE), -VOL_REGIME_CLIP, VOL_REGIME_CLIP)
    )
    daily_scale = sigma * np.sqrt(GLOBEX_SESSION_MINUTES)
    changes = summaries["change"].to_numpy()
    change_valid = summaries["change_valid"].to_numpy()
    if change_valid[previous]:
        output[2] = np.float32(
            np.clip(
                changes[previous] / daily_scale, -PRICE_FEATURE_CLIP, PRICE_FEATURE_CLIP
            )
        )
    if summaries.item(previous, "open_close_valid"):
        output[3] = np.float32(
            np.clip(
                summaries.item(previous, "open_close") / daily_scale,
                -PRICE_FEATURE_CLIP,
                PRICE_FEATURE_CLIP,
            )
        )
    if summaries.item(previous, "last_60_valid"):
        output[4] = np.float32(
            np.clip(
                summaries.item(previous, "last_60") / (sigma * np.sqrt(60)),
                -PRICE_FEATURE_CLIP,
                PRICE_FEATURE_CLIP,
            )
        )
    historical_sigma = sigma_before_session[previous]
    realized_vol = summaries.item(previous, "realized_vol")
    if realized_vol is not None and historical_sigma > 0:
        output[5] = np.float32(
            np.clip(
                np.log(
                    max(float(realized_vol) / historical_sigma, REALIZED_VOL_LOG_FLOOR)
                ),
                -REALIZED_VOL_LOG_CLIP,
                REALIZED_VOL_LOG_CLIP,
            )
        )
    volumes = summaries["volume"].to_numpy()
    output[6] = np.float32(_robust_volume_ratio(volumes, previous))
    valid_indices = np.flatnonzero(change_valid[: previous + 1])
    for return_channel, vol_channel, window, minimum in (
        (7, 9, SLOW_SHORT_WINDOW, SLOW_SHORT_MIN_VALID),
        (8, 10, SLOW_LONG_WINDOW, SLOW_LONG_MIN_VALID),
    ):
        indices = valid_indices[-window:]
        if indices.size < minimum:
            continue
        values = changes[indices]
        output[return_channel] = np.float32(
            np.clip(
                values.sum() / (daily_scale * np.sqrt(indices.size)),
                -PRICE_FEATURE_CLIP,
                PRICE_FEATURE_CLIP,
            )
        )
        output[vol_channel] = np.float32(
            np.clip(
                np.log(
                    max(
                        np.sqrt(np.mean(values**2)) / daily_scale,
                        REALIZED_VOL_LOG_FLOOR,
                    )
                ),
                -REALIZED_VOL_LOG_CLIP,
                REALIZED_VOL_LOG_CLIP,
            )
        )
    rv_ratios = [
        np.log(
            max(
                float(summaries.item(index, "realized_vol"))
                / sigma_before_session[index],
                REALIZED_VOL_LOG_FLOOR,
            )
        )
        for index in range(max(0, previous - 19), previous + 1)
        if summaries.item(index, "realized_vol") is not None
        and sigma_before_session[index] > 0
    ]
    if len(rv_ratios) >= SLOW_LONG_MIN_VALID:
        output[11] = np.float32(np.clip(np.std(rv_ratios), 0.0, REALIZED_VOL_LOG_CLIP))
    volume_history = volumes[max(0, previous - 19) : previous + 1]
    volume_history = volume_history[volume_history > 0]
    if volume_history.size >= VOLUME_MIN_OBSERVATIONS:
        output[12] = np.float32(
            (np.log1p(np.median(volume_history)) - REAL_VOLUME_LOG_CENTER)
            / REAL_VOLUME_LOG_SCALE
        )
    observed = summaries["observed_fraction"].to_numpy()
    recent_observed = observed[max(0, previous - 4) : previous + 1]
    if recent_observed.size >= SLOW_SHORT_MIN_VALID:
        output[16] = np.float32(recent_observed.mean())
    output[26:30] = _calendar_features(trade_date)
    return output


def build_global_instrument_features(
    source_dir: Path,
    symbol: str,
    market_dates: tuple[date, ...],
) -> GlobalInstrumentFeatures:
    frame = load_global_symbol(source_dir, symbol)
    raw, observed, mapping_changed, context_index = build_global_grid(
        frame, market_dates
    )
    summaries = _session_summaries(frame)
    sigma, completed, sigma_before_session = _volatility_state(summaries, market_dates)
    dynamic, _ = build_dynamic_features(
        raw,
        observed,
        sigma > 0,
        sigma,
        is_rate=False,
        mapping_changed=mapping_changed,
    )
    slow = np.zeros(
        (len(market_dates), len(DECISION_GLOBAL_INDICES), 32), dtype=np.float32
    )
    ready = np.zeros(slow.shape[:2], dtype=bool)
    coverage_rows: list[dict[str, object]] = []
    event_ns = frame["ts_event_utc"].cast(pl.Int64).to_numpy()
    end_ns = frame["bar_end_utc"].cast(pl.Int64).to_numpy()
    closes = frame["close"].to_numpy()
    raw_symbols = frame["raw_symbol"].to_list()
    expiries = frame["expiration_utc"].to_list()
    slot = GLOBAL_CONTEXT_SYMBOLS.index(symbol)
    for date_idx, trade_date in enumerate(market_dates):
        previous = int(completed[date_idx])
        base = (
            _base_slow(
                summaries, previous, sigma_before_session, sigma[date_idx], trade_date
            )
            if previous >= 0 and sigma[date_idx] > 0
            else np.zeros(32, dtype=np.float32)
        )
        previous_close_time = (
            _b3_timestamp(market_dates[date_idx - 1], time(16, 45))
            if date_idx > 0
            else None
        )
        previous_position = (
            int(
                np.searchsorted(
                    end_ns,
                    int(previous_close_time.timestamp() * 1_000_000_000),
                    side="right",
                )
                - 1
            )
            if previous_close_time is not None
            else -1
        )
        for decision_idx, decision_time in enumerate(DECISION_TIMES):
            cutoff = DECISION_GLOBAL_INDICES[decision_idx]
            start = cutoff - 345
            window_observed = observed[date_idx, start:cutoff]
            decision = _b3_timestamp(trade_date, decision_time)
            decision_ns = int(decision.timestamp() * 1_000_000_000)
            current_position = int(
                np.searchsorted(end_ns, decision_ns, side="right") - 1
            )
            available = sigma[date_idx] > 0 and bool(window_observed.any())
            ready[date_idx, decision_idx] = available
            if available:
                values = base.copy()
                if (
                    previous_position >= 0
                    and current_position >= 0
                    and raw_symbols[previous_position] == raw_symbols[current_position]
                ):
                    elapsed = max(
                        (event_ns[current_position] - event_ns[previous_position])
                        / 60_000_000_000,
                        1.0,
                    )
                    values[1] = np.float32(
                        np.clip(
                            np.log(closes[current_position] / closes[previous_position])
                            / (sigma[date_idx] * np.sqrt(elapsed)),
                            -PRICE_FEATURE_CLIP,
                            PRICE_FEATURE_CLIP,
                        )
                    )
                expiry = expiries[current_position] if current_position >= 0 else None
                values[31] = np.float32(_expiry_scale(expiry, decision))
                slow[date_idx, decision_idx] = values
            observed_positions = np.flatnonzero(window_observed)
            last_minute = (
                int(observed_positions[-1] + start) if observed_positions.size else -1
            )
            last_end = (
                _b3_timestamp(
                    trade_date,
                    time(
                        (GLOBAL_SESSION_START_MINUTE + last_minute + 1) // 60,
                        (GLOBAL_SESSION_START_MINUTE + last_minute + 1) % 60,
                    ),
                )
                if last_minute >= 0
                else None
            )
            coverage_rows.append(
                {
                    "global_slot": slot,
                    "continuous_symbol": symbol,
                    "family": GLOBAL_CONTEXT_FAMILIES[slot],
                    "quote_direction": GLOBAL_QUOTE_DIRECTIONS[slot],
                    "date_idx": date_idx,
                    "trade_date": trade_date,
                    "decision_idx": decision_idx,
                    "decision_time_utc": decision,
                    "observed_minutes": int(window_observed.sum()),
                    "observed_fraction": float(window_observed.mean()),
                    "last_observed_bar_end_utc": last_end,
                    "staleness_minutes": (
                        float((decision - last_end).total_seconds() / 60.0)
                        if last_end is not None
                        else None
                    ),
                    "roll_count": int(mapping_changed[date_idx, start:cutoff].sum()),
                    "expiry_available": bool(
                        current_position >= 0 and expiries[current_position] is not None
                    ),
                    "ready": available,
                }
            )
    return GlobalInstrumentFeatures(
        dynamic=dynamic,
        slow=slow,
        data_ready=ready,
        context_index=context_index,
        coverage=pl.DataFrame(coverage_rows),
    )
