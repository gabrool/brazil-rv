from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import polars as pl
from numpy.typing import NDArray

from .contract import (
    ACCUMULATED_TEST_AFTER,
    COTAHIST_YEARS,
    INTRADAY_DAILY_FEATURES,
    SLOW_FEATURES,
    STORE_START,
    V1_STORE_V2_ZERO_SLOW_FIELDS,
)
from .corporate_actions import (
    action_presence_array,
    action_coverage_table,
    adjust_daily_ohlc,
    align_action_arrays,
    audit_m1_adjustment_status,
    cash_reinvestment_review_table,
    cash_reinvestment_unavailable_mask,
    detect_distribution_changes,
    detect_split_candidates,
    distribution_review_table,
    provider_failure_mask,
    split_review_table,
    validate_action_table,
)
from .data_foundation import (
    build_security_master,
    filter_cash_equities,
    load_cotahist,
    panel_from_daily,
    source_records,
    verify_v1_mapping,
)
from .features import build_slow_features
from .intraday_features import (
    IntradayDailyResult,
    build_intraday_daily_features,
    mask_action_boundaries,
)
from .normalization import rank_gauss_panel
from .sidecars import (
    SidecarResult,
    available_archive_mapping,
    derive_known_archive_features,
    materialize_known_archive,
)
from .store import write_store
from .targets import build_multi_day_targets, build_to_close_target
from .universe import build_daily_universe, session_calendar, v1_pit_coverage_table

EXPECTED_V1_DATES = 1_248
V1_STORE_START = date(2021, 7, 19)


def _validate_v1_calendar(values: Sequence[date]) -> None:
    """Validate the complete physical v1 store axis, including warm-up rows."""

    dates = tuple(values)
    if (
        len(dates) != EXPECTED_V1_DATES
        or dates[0] != V1_STORE_START
        or dates[-1] != ACCUMULATED_TEST_AFTER
        or any(left >= right for left, right in zip(dates, dates[1:], strict=False))
    ):
        raise ValueError("canonical v1 calendar has the wrong fixed axis")


@dataclass(frozen=True)
class StreamedIntraday:
    result: IntradayDailyResult
    audit: pl.DataFrame
    source_paths: tuple[Path, ...]


def build_v1_fast_mappings(
    v1_store: Path,
    assignments: pl.DataFrame,
    dates: NDArray[np.datetime64],
    isins: Sequence[str],
) -> tuple[pl.DataFrame, pl.DataFrame, list[Path]]:
    """Bind the unchanged dense v1 archive to sparse v2 date/ISIN slots."""

    root = Path(v1_store).resolve()
    required = [
        root / "date_index.parquet",
        root / "equity_index.parquet",
        root / "equity_features.npy",
        root / "equity_slow.npy",
        root / "equity_data_ready.npy",
        root / "manifest.json",
        root / "feature_schema.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"v1 fast-store files missing: {missing}")
    source_dates = pl.read_parquet(required[0]).sort("date_idx")
    source_equities = pl.read_parquet(required[1]).sort("equity_slot")
    from brazil_rv.preprocessing.contract import (
        DYNAMIC_CHANNELS,
        EQUITY_SESSION_MINUTES,
        EXPECTED_EQUITIES,
    )

    if not {"date_idx", "trade_date"}.issubset(source_dates.columns):
        raise ValueError("v1 date index has the wrong schema")
    if not {"equity_slot", "security_id"}.issubset(source_equities.columns):
        raise ValueError("v1 equity index has the wrong schema")
    if source_dates.get_column("date_idx").to_list() != list(range(source_dates.height)):
        raise ValueError("v1 date index is not contiguous")
    if source_equities.get_column("equity_slot").to_list() != list(
        range(source_equities.height)
    ):
        raise ValueError("v1 equity index is not contiguous")
    if source_equities.height != EXPECTED_EQUITIES:
        raise ValueError(
            f"canonical v1 equity axis must contain {EXPECTED_EQUITIES} names"
        )
    v1_dates = source_dates.get_column("trade_date").to_list()
    _validate_v1_calendar(v1_dates)
    features = np.load(root / "equity_features.npy", mmap_mode="r", allow_pickle=False)
    slow = np.load(root / "equity_slow.npy", mmap_mode="r", allow_pickle=False)
    ready = np.load(root / "equity_data_ready.npy", mmap_mode="r", allow_pickle=False)
    if (
        features.shape
        != (
            EXPECTED_V1_DATES,
            EXPECTED_EQUITIES,
            EQUITY_SESSION_MINUTES,
            len(DYNAMIC_CHANNELS),
        )
        or slow.shape != (EXPECTED_V1_DATES, EXPECTED_EQUITIES, 32)
        or slow.dtype != np.float32
        or ready.shape != features.shape[:2]
        or ready.dtype != np.bool_
    ):
        raise ValueError("canonical v1 fast arrays have the wrong fixed axes")
    for value in (features, slow, ready):
        mmap = getattr(value, "_mmap", None)
        if mmap is not None:
            mmap.close()
    date_lookup = {value: index for index, value in enumerate(dates.astype(object))}
    date_rows = []
    for row in source_dates.iter_rows(named=True):
        target = date_lookup.get(row["trade_date"])
        if target is None:
            raise ValueError(f"v1 fast date absent from daily calendar: {row['trade_date']}")
        date_rows.append(
            {
                "trade_date": row["trade_date"],
                "v2_date_index": target,
                "v1_date_index": int(row["date_idx"]),
            }
        )
    mapping = assignments.select("security_id", "isin").unique()
    if mapping.get_column("security_id").n_unique() != mapping.height:
        raise ValueError("v1 assignments are not one-to-one")
    bound = source_equities.join(mapping, on="security_id", how="left", validate="1:1")
    if bound.get_column("isin").null_count() or bound.get_column("isin").n_unique() != bound.height:
        raise ValueError("v1 equity slots do not map one-to-one onto ISIN")
    isin_lookup = {value: index for index, value in enumerate(isins)}
    isin_rows = []
    for row in bound.iter_rows(named=True):
        target = isin_lookup.get(row["isin"])
        if target is None:
            raise ValueError(f"v1 ISIN absent from daily axis: {row['isin']}")
        isin_rows.append(
            {
                "isin": row["isin"],
                "security_id": row["security_id"],
                "v2_isin_index": target,
                "v1_equity_slot": int(row["equity_slot"]),
            }
        )
    return pl.DataFrame(date_rows), pl.DataFrame(isin_rows), required


@dataclass(frozen=True)
class MinutePanel:
    """M1 arrays on their native (usually v1-only) date and ISIN axes."""

    dates: NDArray[np.datetime64]
    isins: tuple[str, ...]
    open_brl: NDArray[np.floating]
    high_brl: NDArray[np.floating]
    low_brl: NDArray[np.floating]
    close_brl: NDArray[np.floating]
    volume: NDArray[np.floating]
    observed: NDArray[np.bool_]

    def __post_init__(self) -> None:
        shape = self.open_brl.shape
        arrays = (
            self.high_brl,
            self.low_brl,
            self.close_brl,
            self.volume,
            self.observed,
        )
        if (
            len(shape) != 3
            or shape[:2] != (len(self.dates), len(self.isins))
            or any(value.shape != shape for value in arrays)
        ):
            raise ValueError("minute panel arrays are misaligned")
        if len(set(self.isins)) != len(self.isins):
            raise ValueError("minute panel ISIN axis must be unique")


def stream_intraday_from_assignments(
    assignments: pl.DataFrame,
    daily: pl.DataFrame,
    dates: Sequence[date],
    isins: Sequence[str],
) -> StreamedIntraday:
    """Build daily M1 derivatives one physical source at a time.

    A physical XP file is loaded once, but every accepted identity segment is
    filtered to its exact COTAHIST ISIN dates before gridding.  Only daily
    derivatives are retained, so the dense 405-minute archive is never expanded
    across the broad v2 ISIN axis.
    """

    from brazil_rv.modeling.contract import workspace_path
    from brazil_rv.preprocessing.contract import (
        EQUITY_SESSION_MINUTES,
        EQUITY_SESSION_START_MINUTE,
    )
    from brazil_rv.preprocessing.io import (
        dense_grid,
        load_source_file,
        prepare_session_bars,
        validate_physical_source_identity,
    )

    required = {"isin", "source_file"}
    if not required.issubset(assignments.columns):
        raise ValueError("streaming assignments need ISIN and source_file")
    calendar = tuple(dates)
    date_lookup = {value: index for index, value in enumerate(calendar)}
    isin_lookup = {value: index for index, value in enumerate(isins)}
    shape = (len(calendar), len(isins))
    feature_values = np.zeros(
        (*shape, len(INTRADAY_DAILY_FEATURES)), dtype=np.float32
    )
    feature_valid = np.zeros(feature_values.shape, dtype=np.bool_)
    entry = np.full(shape, np.nan, dtype=np.float64)
    entry_valid = np.zeros(shape, dtype=np.bool_)
    realized = np.full(shape, np.nan, dtype=np.float64)
    present = np.zeros(shape, dtype=np.bool_)
    session_close = np.full(shape, np.nan, dtype=np.float64)
    session_close_valid = np.zeros(shape, dtype=np.bool_)
    audit_rows: list[dict[str, object]] = []
    source_paths: list[Path] = []
    dates_by_isin = {
        key[0] if isinstance(key, tuple) else key: frozenset(
            group.get_column("trade_date").to_list()
        )
        for key, group in daily.select("isin", "trade_date").group_by("isin")
    }
    for group in assignments.partition_by("source_file"):
        raw_path = Path(str(group[0, "source_file"]))
        source_path = raw_path if raw_path.is_file() else workspace_path(raw_path)
        source_path = source_path.resolve()
        source_paths.append(source_path)
        source = load_source_file(source_path)
        source_sha256 = source_records([source_path])[0]["sha256"]
        if "xp_symbol" in group.columns:
            validate_physical_source_identity(group, source, source_path)
        for row in group.iter_rows(named=True):
            isin = str(row["isin"])
            target = isin_lookup.get(isin)
            if target is None:
                raise ValueError(f"accepted M1 ISIN absent from daily axis: {isin}")
            # dense_grid uses zero for an unobserved entry price.  Retain that
            # masked payload outside the compact working slice as well.
            entry[:, target] = 0.0
            allowed = dates_by_isin.get(isin, frozenset())
            first = row.get("first_overlap_date")
            last = row.get("last_overlap_date")
            if isinstance(first, str):
                first = date.fromisoformat(first)
            if isinstance(last, str):
                last = date.fromisoformat(last)
            if first is not None:
                allowed = frozenset(value for value in allowed if value >= first)
            if last is not None:
                allowed = frozenset(value for value in allowed if value <= last)
            allowed = allowed.intersection(date_lookup)
            if not allowed:
                audit_rows.append(
                    {
                        "isin": isin,
                        "security_id": row.get("security_id"),
                        "source_file": str(source_path),
                        "source_sha256": source_sha256,
                        "allowed_date_count": 0,
                        "observed_session_count": 0,
                        "exact_session_close_count": 0,
                        "fast_present_count": 0,
                    }
                )
                continue
            # Preserve global-calendar semantics for the longest 20-session
            # reducers and one-session lags while avoiding the years of empty
            # M1 history outside this accepted identity segment.
            start = max(date_lookup[min(allowed)] - 20, 0)
            stop = min(date_lookup[max(allowed)] + 21, len(calendar))
            local_calendar = calendar[start:stop]
            bars = prepare_session_bars(
                source,
                source_path,
                allowed,
                local_calendar,
                EQUITY_SESSION_START_MINUTE,
                EQUITY_SESSION_MINUTES,
            )
            grid, observed = dense_grid(
                bars, len(local_calendar), EQUITY_SESSION_MINUTES
            )
            native = build_intraday_daily_features(
                grid[:, None, :, 0],
                grid[:, None, :, 1],
                grid[:, None, :, 2],
                grid[:, None, :, 3],
                grid[:, None, :, 4],
                observed[:, None, :],
            )
            feature_values[start:stop, target] = native.values[:, 0]
            feature_valid[start:stop, target] = native.valid[:, 0]
            entry[start:stop, target] = native.entry_open[:, 0]
            entry_valid[start:stop, target] = native.entry_open_valid[:, 0]
            realized[start:stop, target] = native.realized_daily_vol[:, 0]
            present[start:stop, target] = native.fast_present[:, 0]
            session_close[start:stop, target] = native.session_close[:, 0]
            session_close_valid[start:stop, target] = native.session_close_valid[:, 0]
            has_bar = observed.any(axis=1)
            exact_close = observed[:, -1]
            audit_rows.append(
                {
                    "isin": isin,
                    "security_id": row.get("security_id"),
                    "source_file": str(source_path),
                    "source_sha256": source_sha256,
                    "allowed_date_count": len(allowed),
                    "observed_session_count": int(has_bar.sum()),
                    "exact_session_close_count": int(exact_close.sum()),
                    "fast_present_count": int(native.fast_present.sum()),
                }
            )
            del bars, grid, observed, native
        del source
    return StreamedIntraday(
        result=IntradayDailyResult(
            values=feature_values,
            valid=feature_valid,
            entry_open=entry,
            entry_open_valid=entry_valid,
            session_close=session_close,
            session_close_valid=session_close_valid,
            realized_daily_vol=realized,
            fast_present=present,
        ),
        audit=pl.DataFrame(audit_rows),
        source_paths=tuple(sorted(set(source_paths))),
    )


def _coverage_table(
    dates: NDArray[np.datetime64],
    valid: NDArray[np.bool_],
    names: Sequence[str],
    *,
    family: str,
) -> pl.DataFrame:
    mask = np.asarray(valid, dtype=np.bool_)
    if mask.ndim != 3 or mask.shape[0] != len(dates) or mask.shape[2] != len(names):
        raise ValueError("feature coverage axes are misaligned")
    years = dates.astype("datetime64[Y]").astype(np.int64) + 1970
    rows: list[dict[str, object]] = []
    for year in sorted(set(years.tolist())):
        selected = years == year
        denominator = int(selected.sum()) * mask.shape[1]
        for feature_index, name in enumerate(names):
            count = int(mask[selected, :, feature_index].sum())
            rows.append(
                {
                    "family": family,
                    "year": int(year),
                    "feature": name,
                    "valid_count": count,
                    "possible_count": denominator,
                    "coverage": count / denominator if denominator else 0.0,
                }
            )
    return pl.DataFrame(rows)


def _sidecar_coverage_table(
    dates: NDArray[np.datetime64],
    groups: Mapping[
        str, tuple[Sequence[str], NDArray[np.bool_], Sequence[str]]
    ],
    active: NDArray[np.bool_],
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    years = dates.astype("datetime64[Y]").astype(np.int64) + 1970
    membership = np.asarray(active, dtype=np.bool_)
    if membership.shape[0] != len(dates):
        raise ValueError("sidecar coverage membership axis is misaligned")
    for group, (names, raw_mask, available_features) in sorted(groups.items()):
        mask = np.asarray(raw_mask, dtype=np.bool_)
        if (
            mask.shape != (len(dates), mask.shape[1], len(names))
            or membership.shape != mask.shape[:2]
        ):
            raise ValueError(f"sidecar coverage axes are misaligned for {group}")
        exact = set(available_features)
        for year in sorted(set(years.tolist())):
            selected = years == year
            denominator = int(selected.sum()) * mask.shape[1]
            active_denominator = int(membership[selected].sum())
            for feature_index, name in enumerate(names):
                count = int(mask[selected, :, feature_index].sum())
                active_count = int(
                    (mask[selected, :, feature_index] & membership[selected]).sum()
                )
                rows.append(
                    {
                        "group": group,
                        "year": int(year),
                        "feature": name,
                        "valid_count": count,
                        "possible_count": denominator,
                        "coverage": count / denominator if denominator else 0.0,
                        "active_valid_count": active_count,
                        "active_possible_count": active_denominator,
                        "active_coverage": (
                            active_count / active_denominator
                            if active_denominator
                            else 0.0
                        ),
                        "archive_semantics_available": name in exact,
                    }
                )
    return pl.DataFrame(
        rows,
        schema={
            "group": pl.String,
            "year": pl.Int64,
            "feature": pl.String,
            "valid_count": pl.Int64,
            "possible_count": pl.Int64,
            "coverage": pl.Float64,
            "active_valid_count": pl.Int64,
            "active_possible_count": pl.Int64,
            "active_coverage": pl.Float64,
            "archive_semantics_available": pl.Boolean,
        },
    )


def _align_intraday_result(
    result: IntradayDailyResult,
    minute_dates: NDArray[np.datetime64],
    minute_isins: Sequence[str],
    dates: NDArray[np.datetime64],
    isins: Sequence[str],
) -> IntradayDailyResult:
    date_lookup = {value: index for index, value in enumerate(dates.tolist())}
    isin_lookup = {value: index for index, value in enumerate(isins)}
    output_shape = (len(dates), len(isins))
    values = np.zeros((*output_shape, result.values.shape[2]), dtype=np.float32)
    valid = np.zeros(values.shape, dtype=np.bool_)
    entry = np.full(output_shape, np.nan, dtype=np.float64)
    entry_valid = np.zeros(output_shape, dtype=np.bool_)
    realized = np.full(output_shape, np.nan, dtype=np.float64)
    session_close = np.full(output_shape, np.nan, dtype=np.float64)
    session_close_valid = np.zeros(output_shape, dtype=np.bool_)
    present = np.zeros(output_shape, dtype=np.bool_)
    for source_date, day in enumerate(minute_dates.astype("datetime64[D]").tolist()):
        target_date = date_lookup.get(day)
        if target_date is None:
            continue
        for source_isin, isin in enumerate(minute_isins):
            target_isin = isin_lookup.get(isin)
            if target_isin is None:
                continue
            values[target_date, target_isin] = result.values[source_date, source_isin]
            valid[target_date, target_isin] = result.valid[source_date, source_isin]
            entry[target_date, target_isin] = result.entry_open[source_date, source_isin]
            entry_valid[target_date, target_isin] = result.entry_open_valid[
                source_date, source_isin
            ]
            session_close[target_date, target_isin] = result.session_close[
                source_date, source_isin
            ]
            session_close_valid[target_date, target_isin] = (
                result.session_close_valid[source_date, source_isin]
            )
            realized[target_date, target_isin] = result.realized_daily_vol[
                source_date, source_isin
            ]
            present[target_date, target_isin] = result.fast_present[
                source_date, source_isin
            ]
    return IntradayDailyResult(
        values=values,
        valid=valid,
        entry_open=entry,
        entry_open_valid=entry_valid,
        session_close=session_close,
        session_close_valid=session_close_valid,
        realized_daily_vol=realized,
        fast_present=present,
    )


def build_daily_store(
    daily: pl.DataFrame,
    actions: pl.DataFrame,
    output_dir: Path,
    *,
    minute_panel: MinutePanel | None = None,
    streamed_intraday: StreamedIntraday | None = None,
    sidecars: Mapping[str, SidecarResult] | None = None,
    action_acquisition_audit: pl.DataFrame | None = None,
    v1_assignments: pl.DataFrame | None = None,
    v1_calendar: Sequence[date] | None = None,
    source_paths: Sequence[Path] = (),
    implementation_commit: str | None = None,
    cotahist_raw_sources: Sequence[Path] = (),
    cotahist_parse_audit: Path | None = None,
    v1_fast_store: Path | None = None,
    minimum_calendar_names: int = 50,
    store_start: date | None = STORE_START,
) -> Path:
    """Build the immutable aligned daily store from already-acquired sources."""

    if minute_panel is not None and streamed_intraday is not None:
        raise ValueError("provide either a minute panel or streamed intraday data")

    v1_isins: tuple[str, ...] = ()
    if v1_assignments is not None:
        v1_isins = tuple(v1_assignments.get_column("isin").cast(pl.String).to_list())
    cash = filter_cash_equities(daily, v1_isins=v1_isins)
    calendar = session_calendar(cash, minimum_traded_names=minimum_calendar_names)
    if not calendar:
        raise ValueError("COTAHIST produced no qualifying sessions")
    panel = panel_from_daily(cash, dates=calendar)
    if v1_assignments is not None:
        verify_v1_mapping(v1_assignments, panel.isins)
    if v1_calendar is not None:
        if not v1_calendar:
            raise ValueError("v1 calendar cannot be empty")
        matching_slice = tuple(
            day
            for day in calendar
            if v1_calendar[0] <= day <= v1_calendar[-1]
        )
        if matching_slice != tuple(v1_calendar):
            raise ValueError("v1 date axis differs from the matching COTAHIST calendar slice")

    checked_actions = validate_action_table(actions)
    split, cash_distribution, unresolved = align_action_arrays(
        checked_actions, panel.dates, panel.isins
    )
    price_adjustment_unresolved = unresolved.copy()
    recorded_action = action_presence_array(
        checked_actions, panel.dates, panel.isins
    )
    distribution_changed = detect_distribution_changes(
        panel.distribution_number, panel.observed
    )
    distribution_unresolved = distribution_changed & ~recorded_action
    unresolved |= distribution_unresolved
    price_adjustment_unresolved |= distribution_unresolved
    provider_failed = np.zeros(panel.observed.shape, dtype=np.bool_)
    if action_acquisition_audit is not None:
        provider_failed = provider_failure_mask(
            action_acquisition_audit,
            panel.dates,
            panel.isins,
            panel.observed,
        )
        unresolved |= provider_failed
        price_adjustment_unresolved |= provider_failed
    split_detected = detect_split_candidates(
        panel.close_brl, panel.quantity, panel.observed
    )
    split_disagreement = split_detected != (split != 1.0)
    unresolved |= split_disagreement
    price_adjustment_unresolved |= split_disagreement
    cash_reinvestment_unavailable = cash_reinvestment_unavailable_mask(
        panel.close_brl, cash_distribution
    )
    unresolved |= cash_reinvestment_unavailable
    intraday_action_boundary = recorded_action | unresolved
    adjusted = adjust_daily_ohlc(
        panel.open_brl,
        panel.high_brl,
        panel.low_brl,
        panel.close_brl,
        split,
        cash_distribution,
        unresolved,
    )
    universe = build_daily_universe(panel.close_brl, panel.volume_brl, panel.observed)
    slow_raw = build_slow_features(
        adjusted.adjusted_open,
        adjusted.adjusted_high,
        adjusted.adjusted_low,
        adjusted.adjusted_close,
        adjusted.total_return_close,
        panel.volume_brl,
        panel.trades,
        panel.observed,
        universe.active,
        panel.dates,
        unresolved_action=unresolved,
        price_adjustment_unresolved=price_adjustment_unresolved,
    )
    slow_values, slow_valid = rank_gauss_panel(
        slow_raw.values, slow_raw.valid, universe.active
    )

    shape = panel.observed.shape
    intraday_values = np.zeros((*shape, len(INTRADAY_DAILY_FEATURES)), dtype=np.float32)
    intraday_valid = np.zeros(intraday_values.shape, dtype=np.bool_)
    fast_sigma = np.full(shape, np.nan, dtype=np.float64)
    fast_present = np.zeros(shape, dtype=np.bool_)
    entry = np.full(shape, np.nan, dtype=np.float64)
    entry_valid = np.zeros(shape, dtype=np.bool_)
    realized_daily = np.full(shape, np.nan, dtype=np.float64)
    m1_session_close = np.full(shape, np.nan, dtype=np.float64)
    m1_session_close_valid = np.zeros(shape, dtype=np.bool_)
    if minute_panel is not None:
        native = build_intraday_daily_features(
            minute_panel.open_brl,
            minute_panel.high_brl,
            minute_panel.low_brl,
            minute_panel.close_brl,
            minute_panel.volume,
            minute_panel.observed,
        )
        aligned = _align_intraday_result(
            native, minute_panel.dates, minute_panel.isins, panel.dates, panel.isins
        )
        aligned = mask_action_boundaries(aligned, intraday_action_boundary)
        intraday_values, intraday_valid = rank_gauss_panel(
            aligned.values, aligned.valid, universe.active
        )
        fast_sigma = np.where(aligned.valid[..., 14], aligned.values[..., 14], np.nan)
        fast_present = aligned.fast_present
        entry = aligned.entry_open
        entry_valid = aligned.entry_open_valid
        realized_daily = aligned.realized_daily_vol
        m1_session_close = aligned.session_close
        m1_session_close_valid = aligned.session_close_valid
    elif streamed_intraday is not None:
        aligned = streamed_intraday.result
        if aligned.values.shape[:2] != shape:
            raise ValueError("streamed intraday derivatives are misaligned")
        aligned = mask_action_boundaries(aligned, intraday_action_boundary)
        intraday_values, intraday_valid = rank_gauss_panel(
            aligned.values, aligned.valid, universe.active
        )
        fast_sigma = np.where(aligned.valid[..., 14], aligned.values[..., 14], np.nan)
        fast_present = aligned.fast_present
        entry = aligned.entry_open
        entry_valid = aligned.entry_open_valid
        realized_daily = aligned.realized_daily_vol
        m1_session_close = aligned.session_close
        m1_session_close_valid = aligned.session_close_valid

    slow_sigma = np.where(slow_raw.valid[..., 8], slow_raw.values[..., 8], np.nan)
    targets = build_multi_day_targets(
        adjusted.total_return_close,
        universe.active,
        fast_sigma,
        slow_sigma,
        fast_present,
        unresolved,
    )
    arrays: dict[str, NDArray[np.generic]] = {
        "active": universe.active,
        "observed": panel.observed,
        "raw_open": panel.open_brl,
        "raw_high": panel.high_brl,
        "raw_low": panel.low_brl,
        "raw_close": panel.close_brl,
        "adjusted_open": adjusted.adjusted_open,
        "adjusted_high": adjusted.adjusted_high,
        "adjusted_low": adjusted.adjusted_low,
        "adjusted_close": adjusted.adjusted_close,
        "total_return_close": adjusted.total_return_close,
        "price_adjustment_factor": adjusted.price_factor,
        "total_return_adjustment_factor": adjusted.total_return_factor,
        "volume_brl": panel.volume_brl,
        "trade_count": panel.trades,
        "quantity": panel.quantity,
        "distribution_number": panel.distribution_number,
        "recorded_action_mask": recorded_action,
        "distribution_change_mask": distribution_changed,
        "provider_action_failure_mask": provider_failed,
        "split_disagreement_mask": split_disagreement,
        "cash_reinvestment_unavailable_mask": cash_reinvestment_unavailable,
        "intraday_action_boundary_mask": intraday_action_boundary,
        "price_adjustment_unresolved": price_adjustment_unresolved,
        "adjusted_price_level_valid": ~np.maximum.accumulate(
            price_adjustment_unresolved, axis=0
        ),
        "unresolved_action": unresolved,
        "slow_values": slow_values,
        "slow_valid": slow_valid,
        "intraday_values": intraday_values,
        "intraday_valid": intraday_valid,
        "fast_present": fast_present,
        "target_primary": targets.primary,
        "target_valid": targets.primary_valid,
        "target_normalized_residual": targets.normalized_residual,
        "target_raw_midrank": targets.raw_midrank,
        "target_raw_valid": targets.raw_valid,
        "target_raw_log_return": targets.raw_log_return,
    }
    if minute_panel is not None or streamed_intraday is not None:
        to_close = build_to_close_target(
            entry,
            np.where(m1_session_close_valid, m1_session_close, np.nan),
            realized_daily,
            universe.active,
            fast_present & entry_valid,
        )
        arrays.update(
            {
                "target_to_close": to_close.target,
                "target_to_close_valid": to_close.valid,
                "target_to_close_normalized_residual": to_close.normalized_residual,
                "target_to_close_raw_log_return": to_close.raw_log_return,
            }
        )
    feature_names: dict[str, Sequence[str]] = {
        "slow": SLOW_FEATURES,
        "intraday": INTRADAY_DAILY_FEATURES,
        "horizons": tuple(str(value) for value in targets.horizons),
    }
    coverage_tables = [
        _coverage_table(panel.dates, slow_valid, SLOW_FEATURES, family="slow"),
        _coverage_table(
            panel.dates,
            intraday_valid,
            INTRADAY_DAILY_FEATURES,
            family="intraday",
        ),
    ]
    sidecar_masks: dict[
        str, tuple[Sequence[str], NDArray[np.bool_], Sequence[str]]
    ] = {}
    for group, result in sorted((sidecars or {}).items()):
        if result.values.shape[:2] != shape:
            raise ValueError(f"sidecar {group} does not match the store axes")
        values, valid = rank_gauss_panel(result.values, result.valid, universe.active)
        arrays[f"sidecar_{group}_values"] = values
        arrays[f"sidecar_{group}_valid"] = valid
        feature_names[f"sidecar_{group}"] = result.feature_names
        sidecar_masks[group] = (
            result.feature_names,
            result.valid,
            result.archive_semantics_available,
        )

    tables = {
        "security_master": build_security_master(cash),
        "feature_coverage": pl.concat(coverage_tables),
        "sidecar_coverage": _sidecar_coverage_table(
            panel.dates, sidecar_masks, universe.active
        ),
        "universe_size": pl.DataFrame(
            {
                "trade_date": panel.dates,
                "member_count": universe.active.sum(axis=1).astype(np.int32),
            }
        ),
        "corporate_action_split_review": split_review_table(
            panel.dates.astype(object), panel.isins, split_detected, split
        ),
        "corporate_action_distribution_review": distribution_review_table(
            panel.dates,
            panel.isins,
            distribution_changed,
            recorded_action,
        ),
        "corporate_action_cash_reinvestment_review": cash_reinvestment_review_table(
            panel.dates,
            panel.isins,
            cash_distribution,
            panel.close_brl,
            panel.observed,
        ),
        "corporate_action_coverage": action_coverage_table(
            checked_actions,
            panel.dates,
            panel.isins,
            action_acquisition_audit,
        ),
        "corporate_actions": checked_actions,
    }
    if v1_assignments is not None:
        tables["v1_pit_active_coverage"] = v1_pit_coverage_table(
            panel.dates,
            panel.isins,
            universe.active,
            tuple(v1_assignments.get_column("isin").cast(pl.String).to_list()),
        )
    if action_acquisition_audit is not None:
        tables["corporate_action_acquisition_audit"] = action_acquisition_audit
    if minute_panel is not None or streamed_intraday is not None:
        tables["m1_adjustment_audit"] = audit_m1_adjustment_status(
            panel.dates,
            panel.isins,
            m1_session_close,
            panel.close_brl,
            adjusted.adjusted_close,
            split,
            cash_distribution,
        )
    if streamed_intraday is not None:
        tables["m1_source_audit"] = streamed_intraday.audit
    keep = np.ones(len(panel.dates), dtype=np.bool_)
    if store_start is not None:
        keep &= panel.dates >= np.datetime64(store_start)
    if not keep.any():
        raise ValueError("store_start removes the complete calendar")
    kept_dates = panel.dates[keep]
    tables["feature_coverage"] = pl.concat(
        [
            _coverage_table(
                kept_dates,
                slow_valid[keep],
                SLOW_FEATURES,
                family="slow",
            ),
            _coverage_table(
                kept_dates,
                intraday_valid[keep],
                INTRADAY_DAILY_FEATURES,
                family="intraday",
            ),
        ]
    )
    tables["sidecar_coverage"] = _sidecar_coverage_table(
        kept_dates,
        {
            group: (names, np.asarray(mask)[keep], available)
            for group, (names, mask, available) in sidecar_masks.items()
        },
        universe.active[keep],
    )
    tables["universe_size"] = tables["universe_size"].filter(
        pl.col("trade_date") >= kept_dates[0].astype(object)
    )
    v1_fast_files: list[dict[str, object]] = []
    if v1_fast_store is not None:
        if v1_assignments is None:
            raise ValueError("v1 fast store requires one-to-one v1 assignments")
        date_mapping, isin_mapping, fast_paths = build_v1_fast_mappings(
            v1_fast_store, v1_assignments, kept_dates, panel.isins
        )
        tables["v1_fast_date_mapping"] = date_mapping
        tables["v1_fast_isin_mapping"] = isin_mapping
        v1_fast_files = source_records(fast_paths)
    metadata = {
        "store_start": str(kept_dates[0]),
        "store_end": str(kept_dates[-1]),
        "lookback_rows_materialized": False,
        "slow_entry_alignment": {
            "pretrain": "through_t",
            "finetune_evaluation": "through_t_minus_1",
        },
        "v1_fast_store": str(v1_fast_store.resolve()) if v1_fast_store else None,
        "v1_fast_files": v1_fast_files,
        "v1_store_v2_zero_dynamic_channels": [9, 11, 14, 22, 24, 25],
        "v1_store_v2_zero_slow_fields": list(V1_STORE_V2_ZERO_SLOW_FIELDS),
        "v1_isin_subset_verified": v1_assignments is not None,
        "v1_calendar_verified": v1_calendar is not None,
        "implementation_git_commit": implementation_commit,
        "cash_reinvestment_unavailable_count_foundation": int(
            cash_reinvestment_unavailable.sum()
        ),
        "cash_reinvestment_unavailable_count_store_rows": int(
            cash_reinvestment_unavailable[keep].sum()
        ),
        "cotahist_provenance": {
            "raw_archives": source_records(cotahist_raw_sources),
            "parse_audit": (
                source_records((cotahist_parse_audit,))[0]
                if cotahist_parse_audit is not None
                else None
            ),
        },
    }
    if "v1_pit_active_coverage" in tables:
        active_counts = tables["v1_pit_active_coverage"].get_column(
            "active_v1_count"
        )
        metadata["v1_pit_active_count"] = {
            "semantics": (
                "dynamic PIT-active subset of the exact mapped v1 identities; "
                "all mapped identities are active at least once, not necessarily "
                "simultaneously"
            ),
            "minimum": int(active_counts.min()),
            "median": float(active_counts.median()),
            "maximum": int(active_counts.max()),
        }
    return write_store(
        output_dir,
        dates=kept_dates,
        isins=panel.isins,
        arrays=arrays,
        row_indices=np.flatnonzero(keep),
        feature_names=feature_names,
        sources=source_records(
            (
                *source_paths,
                *(streamed_intraday.source_paths if streamed_intraday else ()),
            )
        ),
        metadata=metadata,
        tables=tables,
    )


def load_minute_npz(path: Path) -> MinutePanel:
    archive = np.load(path, allow_pickle=False)
    required = {"dates", "isins", "open", "high", "low", "close", "volume", "observed"}
    if not required.issubset(archive.files):
        raise ValueError(f"minute archive keys missing: {sorted(required - set(archive.files))}")
    return MinutePanel(
        dates=np.asarray(archive["dates"], dtype="datetime64[D]"),
        isins=tuple(str(value) for value in archive["isins"].tolist()),
        open_brl=archive["open"],
        high_brl=archive["high"],
        low_brl=archive["low"],
        close_brl=archive["close"],
        volume=archive["volume"],
        observed=archive["observed"],
    )


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the immutable Brazil-RV v2 daily store")
    parser.add_argument("--cotahist-root", required=True, type=Path)
    parser.add_argument("--cotahist-raw-root", required=True, type=Path)
    parser.add_argument("--cotahist-parse-audit", required=True, type=Path)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--actions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--minute-npz", type=Path)
    parser.add_argument("--v1-assignments", required=True, type=Path)
    parser.add_argument("--v1-store", required=True, type=Path)
    parser.add_argument(
        "--sidecar",
        action="append",
        default=[],
        metavar="GROUP=PARQUET",
        help="Known v1 sidecar archive; may be repeated",
    )
    return parser.parse_args(arguments)


def _load_action_bundle(
    actions_path: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, tuple[Path, ...]]:
    """Load and hash-verify the exact immutable acquisition bundle."""

    actions_path = actions_path.resolve()
    manifest_path = actions_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"corporate-action acquisition manifest is missing: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "V2_CORPORATE_ACTIONS_V1":
        raise ValueError("corporate-action acquisition manifest has wrong schema")
    paths: dict[str, Path] = {}
    for key in ("security_master", "actions", "acquisition_audit"):
        record = manifest.get(key)
        if not isinstance(record, dict) or not {"path", "sha256"}.issubset(record):
            raise ValueError(f"corporate-action manifest lacks {key}")
        path = (manifest_path.parent / str(record["path"])).resolve()
        if (
            not path.is_file()
            or source_records([path])[0]["sha256"] != record["sha256"]
        ):
            raise ValueError(f"corporate-action bundle hash mismatch: {key}")
        paths[key] = path
    if paths["actions"] != actions_path:
        raise ValueError("--actions is not the action file bound by its manifest")
    return (
        pl.read_parquet(paths["actions"]),
        pl.read_parquet(paths["acquisition_audit"]),
        pl.read_parquet(paths["security_master"]),
        (manifest_path, *paths.values()),
    )


def _parse_sidecars(
    arguments: Sequence[str],
    dates: Sequence[object],
    isins: Sequence[str],
    assignments: pl.DataFrame | None,
    daily_volume_brl: NDArray[np.floating] | None = None,
) -> dict[str, SidecarResult]:
    normalized_dates = tuple(
        value.astype("datetime64[D]").astype(object)
        if isinstance(value, np.datetime64)
        else value
        for value in dates
    )
    if not normalized_dates:
        raise ValueError("sidecar materialization needs a nonempty calendar")
    assignment_mapping = None
    if assignments is not None:
        assignment_mapping = assignments.select("security_id", "isin").unique()
        if (
            assignment_mapping.get_column("security_id").n_unique()
            != assignment_mapping.height
        ):
            raise ValueError("sidecar assignments are not one-to-one")

    grouped: dict[str, list[pl.DataFrame]] = {}
    for argument in arguments:
        if "=" not in argument:
            raise ValueError("--sidecar must be GROUP=PARQUET")
        group, raw_path = argument.split("=", 1)
        path = Path(raw_path)
        schema = pl.read_parquet_schema(path)
        feature_mapping = available_archive_mapping(group, tuple(schema))
        value_columns = sorted(
            {column for column in feature_mapping.values() if column is not None}
        )
        derivation_columns = {
            "lending": (
                "source_position_date",
                "source_trade_date",
                "lending_balance_brl",
            ),
            "oddlot": (
                "source_trade_date",
                "regular_volume_brl",
                "odd_lot_volume_brl",
            ),
        }.get(group, ())
        value_columns = sorted(
            {*value_columns, *(column for column in derivation_columns if column in schema)}
        )
        if not value_columns:
            # The caller still hash-binds the archive and the store reports zero
            # coverage. Avoid reading millions of rows when it has no compatible
            # frozen feature columns.
            grouped.setdefault(group, [])
            continue
        if "available_date" not in schema:
            raise ValueError(f"sidecar archive lacks available_date: {path}")
        identity = "isin" if "isin" in schema else "security_id"
        if identity not in schema:
            raise ValueError(f"sidecar archive lacks a usable identity: {path}")
        optional = [
            column
            for column in (
                "decision_idx",
                "available_timestamp",
                "delivery_timestamp",
            )
            if column in schema
        ]
        masks = [
            f"{column}_mask"
            for column in value_columns
            if f"{column}_mask" in schema
        ]
        projected = ["available_date", identity, *optional, *value_columns, *masks]
        lazy = (
            pl.scan_parquet(path)
            .select(projected)
            .filter(
                pl.col("available_date").is_between(
                    normalized_dates[0], normalized_dates[-1]
                )
            )
        )
        if identity == "isin":
            lazy = lazy.filter(
                pl.col("isin").is_in(pl.Series("isin", isins).implode())
            )
        else:
            if assignment_mapping is None:
                raise ValueError("security_id sidecars require v1 assignments")
            lazy = (
                lazy.filter(
                    pl.col("security_id").is_in(
                        assignment_mapping.get_column("security_id").implode()
                    )
                )
                .join(assignment_mapping.lazy(), on="security_id", how="left")
                .drop("security_id")
            )

        # Intraday v1 state archives repeat a complete state for all 55
        # decision coordinates. The v2 decision is later than coordinate 54,
        # so reduce each large file to its final same-day snapshot before it is
        # collected. Peak memory is then proportional to date/ISIN output rows,
        # rather than to the 5M-row physical archive.
        if "decision_idx" in optional and not {
            "available_timestamp",
            "delivery_timestamp",
        }.intersection(optional):
            bounds = lazy.select(
                pl.col("decision_idx").min().alias("minimum"),
                pl.col("decision_idx").max().alias("maximum"),
            ).collect(engine="streaming")
            if bounds.height and bounds.item(0, "minimum") is not None:
                if bounds.item(0, "minimum") < 0 or bounds.item(0, "maximum") > 54:
                    raise ValueError(
                        f"sidecar decision_idx is outside canonical 0..54: {path}"
                    )
            row_columns = [
                column
                for column in projected
                if column not in {"available_date", identity, "decision_idx"}
            ]
            lazy = lazy.group_by("available_date", "isin").agg(
                pl.col("decision_idx").max(),
                *(
                    pl.col(column).sort_by("decision_idx").last().alias(column)
                    for column in row_columns
                ),
            )
        source = lazy.collect(engine="streaming")
        if source.get_column("isin").null_count():
            raise ValueError(f"sidecar archive contains unmapped identities: {path}")
        grouped.setdefault(group, []).append(source)
    output: dict[str, SidecarResult] = {}
    for group, sources in grouped.items():
        if sources:
            source = pl.concat(sources, how="diagonal_relaxed")
        else:
            source = pl.DataFrame(
                schema={"available_date": pl.Date, "isin": pl.String}
            )
        identity = "isin" if "isin" in source.columns else "security_id"
        keys = ["available_date", identity]
        for optional in ("decision_idx", "available_timestamp", "delivery_timestamp"):
            if optional in source.columns:
                keys.append(optional)
        value_columns = [column for column in source.columns if column not in keys]
        source = source.group_by(keys, maintain_order=True).agg(
            pl.col(column).drop_nulls().last().alias(column)
            for column in value_columns
        )
        source = derive_known_archive_features(
            source,
            dates,
            isins,
            group=group,
            daily_volume_brl=daily_volume_brl,
        )
        output[group] = materialize_known_archive(source, dates, isins, group=group)
    return output


def _require_clean_implementation_commit(
    repository: Path, expected_commit: str
) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError(
            "v2 store construction requires a clean tracked and untracked worktree"
        )
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current_commit != expected_commit:
        raise ValueError(
            "--implementation-commit does not match the checked-out repository"
        )


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_args(arguments)
    if not re.fullmatch(r"[0-9a-f]{40}", args.implementation_commit):
        raise ValueError("--implementation-commit must be a full lowercase Git SHA")
    repository = Path(__file__).resolve().parents[4]
    _require_clean_implementation_commit(repository, args.implementation_commit)
    raw_sources = tuple(
        (args.cotahist_raw_root / f"COTAHIST_A{year}.ZIP").resolve()
        for year in COTAHIST_YEARS
    )
    missing_raw = [str(path) for path in raw_sources if not path.is_file()]
    if missing_raw:
        raise FileNotFoundError(f"COTAHIST raw archives missing: {missing_raw}")
    if not args.cotahist_parse_audit.is_file():
        raise FileNotFoundError(args.cotahist_parse_audit)
    from brazil_rv.preprocessing.contract import EXPECTED_EQUITIES

    if args.v1_assignments.is_dir():
        from brazil_rv.preprocessing.io import load_assignments

        assignments = load_assignments(args.v1_assignments)
        assignment_path = (
            args.v1_assignments / "xp_accepted_source_assignments_v1.parquet"
        )
    else:
        assignments = pl.read_parquet(args.v1_assignments)
        assignment_path = args.v1_assignments
    if (
        assignments.height != EXPECTED_EQUITIES
        or assignments.get_column("security_id").n_unique() != EXPECTED_EQUITIES
        or assignments.get_column("isin").n_unique() != EXPECTED_EQUITIES
    ):
        raise ValueError(
            f"canonical v1 assignments must bind exactly {EXPECTED_EQUITIES} identities"
        )
    v1_isins = (
        tuple(assignments.get_column("isin").cast(pl.String).to_list())
        if assignments is not None
        else ()
    )
    paths = sorted(args.cotahist_root.glob("year=*/equities_daily_*.parquet"))
    daily = load_cotahist(paths, v1_isins=v1_isins).filter(
        pl.col("trade_date").dt.year().is_in(COTAHIST_YEARS)
    )
    foundation = filter_cash_equities(daily, v1_isins=v1_isins)
    available_years = set(
        foundation.get_column("trade_date").dt.year().unique().to_list()
    )
    if available_years != set(COTAHIST_YEARS):
        raise ValueError(
            "canonical COTAHIST foundation must contain exactly years "
            f"{COTAHIST_YEARS}; got {sorted(available_years)}"
        )
    calendar = session_calendar(foundation)
    isins = tuple(sorted(foundation.get_column("isin").unique().to_list()))
    daily_panel = panel_from_daily(foundation, dates=calendar, isins=isins)
    v1_calendar = tuple(
        pl.read_parquet(args.v1_store / "date_index.parquet")
        .sort("date_idx")
        .get_column("trade_date")
        .to_list()
    )
    _validate_v1_calendar(v1_calendar)
    actions, acquisition_audit, action_master, action_sources = _load_action_bundle(
        args.actions
    )
    expected_master = build_security_master(foundation)
    master_columns = ["isin", "ticker", "first_date", "last_date"]
    if not action_master.select(master_columns).sort(master_columns).equals(
        expected_master.select(master_columns).sort(master_columns)
    ):
        raise ValueError(
            "corporate-action security master differs from the COTAHIST foundation"
        )
    streamed = None
    minute = load_minute_npz(args.minute_npz) if args.minute_npz else None
    if minute is None and assignments is not None:
        streamed = stream_intraday_from_assignments(
            assignments, foundation, calendar, isins
        )
    sidecars = _parse_sidecars(
        args.sidecar,
        calendar,
        isins,
        assignments,
        daily_panel.volume_brl,
    )
    # build_daily_store reconstructs the canonical panel from ``daily``.  Drop
    # the acquisition-only copies before that second materialization so local
    # validation does not retain two broad daily panels at its memory peak.
    del daily_panel, foundation, action_master, expected_master
    output = build_daily_store(
        daily,
        actions,
        args.output_dir,
        minute_panel=minute,
        streamed_intraday=streamed,
        sidecars=sidecars,
        action_acquisition_audit=acquisition_audit,
        v1_assignments=assignments,
        v1_calendar=v1_calendar,
        source_paths=(
            *paths,
            assignment_path,
            *action_sources,
            *((args.minute_npz,) if args.minute_npz else ()),
            *(
                Path(value.split("=", 1)[1])
                for value in args.sidecar
            ),
        ),
        implementation_commit=args.implementation_commit,
        cotahist_raw_sources=raw_sources,
        cotahist_parse_audit=args.cotahist_parse_audit,
        v1_fast_store=args.v1_store,
    )
    print(json.dumps({"store": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
