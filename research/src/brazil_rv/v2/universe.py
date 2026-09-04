from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl
from numpy.typing import NDArray

from .contract import (
    FINETUNE_START,
    UNIVERSE_MIN_HISTORY,
    UNIVERSE_MIN_MEDIAN_VOLUME_BRL,
    UNIVERSE_MIN_PRIOR_CLOSE_BRL,
    UNIVERSE_MIN_TRADED,
    UNIVERSE_PRIOR_SESSIONS,
)


@dataclass(frozen=True)
class UniverseResult:
    active: NDArray[np.bool_]
    prior_traded_sessions: NDArray[np.int16]
    prior_median_volume_brl: NDArray[np.float64]
    prior_close_brl: NDArray[np.float64]
    history_sessions: NDArray[np.int32]


def session_calendar(
    daily: pl.DataFrame, *, minimum_traded_names: int = 50
) -> tuple[object, ...]:
    """Return COTAHIST dates having at least the required distinct ISINs."""

    if minimum_traded_names <= 0:
        raise ValueError("minimum_traded_names must be positive")
    isin_column = "isin" if "isin" in daily.columns else "security_id"
    if not {"trade_date", isin_column}.issubset(daily.columns):
        raise ValueError("daily data must contain trade_date and ISIN identity")
    rows = (
        daily.filter(pl.col(isin_column).is_not_null())
        .group_by("trade_date")
        .agg(pl.col(isin_column).n_unique().alias("traded_names"))
        .filter(pl.col("traded_names") >= minimum_traded_names)
        .sort("trade_date")
    )
    return tuple(rows.get_column("trade_date").to_list())


def build_daily_universe(
    close_brl: NDArray[np.floating],
    volume_brl: NDArray[np.floating],
    observed: NDArray[np.bool_],
    *,
    prior_sessions: int = UNIVERSE_PRIOR_SESSIONS,
    minimum_traded: int = UNIVERSE_MIN_TRADED,
    minimum_median_volume_brl: float = UNIVERSE_MIN_MEDIAN_VOLUME_BRL,
    minimum_prior_close_brl: float = UNIVERSE_MIN_PRIOR_CLOSE_BRL,
    minimum_history_sessions: int = UNIVERSE_MIN_HISTORY,
) -> UniverseResult:
    """Build a strictly prior-session daily universe.

    Missing security observations count as zero volume and as not traded. History
    is calendar-session age since the first observed security-day; no value from
    the candidate date is consulted.
    """

    close = np.asarray(close_brl, dtype=np.float64)
    volume = np.asarray(volume_brl, dtype=np.float64)
    seen = np.asarray(observed, dtype=np.bool_)
    if close.ndim != 2 or close.shape != volume.shape or close.shape != seen.shape:
        raise ValueError("close, volume, and observed must be aligned [date, name]")
    if prior_sessions <= 0 or not 0 <= minimum_traded <= prior_sessions:
        raise ValueError("invalid prior-session/traded thresholds")
    if minimum_history_sessions <= 0 or minimum_median_volume_brl < 0:
        raise ValueError("invalid universe thresholds")

    date_count, name_count = close.shape
    active = np.zeros_like(seen)
    traded_count = np.zeros(close.shape, dtype=np.int16)
    median_volume = np.full(close.shape, np.nan, dtype=np.float64)
    prior_close = np.full(close.shape, np.nan, dtype=np.float64)
    history = np.zeros(close.shape, dtype=np.int32)
    first_seen = np.full(name_count, -1, dtype=np.int64)
    for date_index in range(date_count):
        newly_seen = (first_seen < 0) & seen[date_index]
        first_seen[newly_seen] = date_index
        if date_index == 0:
            continue
        age = np.where(first_seen >= 0, date_index - first_seen, 0)
        history[date_index] = age.astype(np.int32)
        prior_start = max(0, date_index - prior_sessions)
        for name in range(name_count):
            prior_rows = np.flatnonzero(seen[prior_start:date_index, name])
            if prior_rows.size:
                prior_close[date_index, name] = close[
                    prior_start + prior_rows[-1], name
                ]
        if date_index < prior_sessions:
            continue
        start = date_index - prior_sessions
        window_seen = seen[start:date_index]
        window_volume = np.where(
            window_seen & np.isfinite(volume[start:date_index]),
            volume[start:date_index],
            0.0,
        )
        traded_count[date_index] = window_seen.sum(axis=0).astype(np.int16)
        median_volume[date_index] = np.median(window_volume, axis=0)
        active[date_index] = (
            (traded_count[date_index] >= minimum_traded)
            & (median_volume[date_index] >= minimum_median_volume_brl)
            & np.isfinite(prior_close[date_index])
            & (prior_close[date_index] >= minimum_prior_close_brl)
            & (history[date_index] >= minimum_history_sessions)
        )
    return UniverseResult(
        active=active,
        prior_traded_sessions=traded_count,
        prior_median_volume_brl=median_volume,
        prior_close_brl=prior_close,
        history_sessions=history,
    )


def assert_security_subset(all_isins: tuple[str, ...], required_isins: tuple[str, ...]) -> None:
    missing = sorted(set(required_isins) - set(all_isins))
    if missing:
        raise ValueError(f"Required v1 ISINs are absent from the v2 axis: {missing}")


def v1_pit_coverage_table(
    dates: NDArray[np.datetime64],
    all_isins: tuple[str, ...],
    active: NDArray[np.bool_],
    required_isins: tuple[str, ...],
    *,
    start: date = FINETUNE_START,
) -> pl.DataFrame:
    """Audit the dynamic PIT-active subset of the mapped v1 identities.

    The invariant is axis inclusion plus at least one eligible post-start
    session per mapped identity.  It intentionally does not claim that all 158
    identities are simultaneously active on every date.
    """

    calendar = np.asarray(dates, dtype="datetime64[D]")
    membership = np.asarray(active, dtype=np.bool_)
    if membership.shape != (calendar.size, len(all_isins)):
        raise ValueError("v1 PIT coverage axes are misaligned")
    assert_security_subset(all_isins, required_isins)
    if len(set(required_isins)) != len(required_isins):
        raise ValueError("required v1 ISINs must be unique")
    selected = calendar >= np.datetime64(start)
    if not selected.any():
        return pl.DataFrame(
            schema={
                "trade_date": pl.Date,
                "active_v1_count": pl.Int32,
                "mapped_v1_count": pl.Int32,
            }
        )
    lookup = {isin: index for index, isin in enumerate(all_isins)}
    slots = np.asarray([lookup[isin] for isin in required_isins], dtype=np.int64)
    required_active = membership[selected][:, slots]
    never_active = [
        isin
        for isin, seen in zip(required_isins, required_active.any(axis=0), strict=True)
        if not seen
    ]
    if never_active:
        raise ValueError(
            "mapped v1 ISINs never PIT-active from the v2 fine-tune start: "
            f"{never_active}"
        )
    return pl.DataFrame(
        {
            "trade_date": calendar[selected],
            "active_v1_count": required_active.sum(axis=1).astype(np.int32),
            "mapped_v1_count": np.full(
                int(selected.sum()), len(required_isins), dtype=np.int32
            ),
        }
    )


def v1_pit_inactive_exceptions_table(
    dates: NDArray[np.datetime64],
    all_isins: tuple[str, ...],
    active: NDArray[np.bool_],
    required_isins: tuple[str, ...],
    *,
    start: date = FINETUNE_START,
) -> pl.DataFrame:
    """Document every mapped v1 name not PIT-eligible on every post-start date."""

    calendar = np.asarray(dates, dtype="datetime64[D]")
    membership = np.asarray(active, dtype=np.bool_)
    if membership.shape != (calendar.size, len(all_isins)):
        raise ValueError("v1 PIT exception axes are misaligned")
    assert_security_subset(all_isins, required_isins)
    selected = calendar >= np.datetime64(start)
    lookup = {isin: index for index, isin in enumerate(all_isins)}
    rows: list[dict[str, object]] = []
    for isin in required_isins:
        values = membership[selected, lookup[isin]]
        active_rows = np.flatnonzero(values)
        inactive_count = int((~values).sum())
        if not inactive_count:
            continue
        selected_dates = calendar[selected]
        rows.append(
            {
                "isin": isin,
                "inactive_session_count": inactive_count,
                "post_start_session_count": int(values.size),
                "first_active_date": (
                    selected_dates[active_rows[0]].astype(object)
                    if active_rows.size
                    else None
                ),
                "last_active_date": (
                    selected_dates[active_rows[-1]].astype(object)
                    if active_rows.size
                    else None
                ),
                "reason": (
                    "dynamic point-in-time liquidity, price, history, or delisting screen"
                ),
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "isin": pl.String,
            "inactive_session_count": pl.Int32,
            "post_start_session_count": pl.Int32,
            "first_active_date": pl.Date,
            "last_active_date": pl.Date,
            "reason": pl.String,
        },
    ).sort("isin")
