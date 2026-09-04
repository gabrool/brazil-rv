from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import polars as pl
from numpy.typing import NDArray

from .contract import (
    ACCUMULATED_TEST_AFTER,
    COTAHIST_YEARS,
    HORIZONS,
    INTRADAY_DAILY_FEATURES,
    SLOW_FEATURES,
    STORE_START,
    V1_STORE_V2_ZERO_SLOW_FIELDS,
)
from .corporate_actions import (
    action_calendar_alignment_table,
    action_coverage_table,
    adjust_daily_ohlc,
    align_action_arrays,
    audit_m1_adjustment_status,
    cash_unit_adjustment_audit,
    cotahist_action_classification_table,
    dividend_close_drop_audit,
    detect_cotahist_actions,
    detect_distribution_changes,
    m1_cotahist_mismatch_by_year,
    provider_split_detection_audit,
    split_review_table,
    validate_action_table,
)
from .data_foundation import (
    build_security_master,
    continuation_identity_axis,
    detect_isin_successions,
    filter_cash_equities,
    inherit_linked_history,
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
    replace_daily_close_anchors,
)
from .normalization import rank_gauss_panel_into
from .sidecars import (
    SidecarResult,
    available_archive_mapping,
    derive_known_archive_features,
    materialize_known_archive,
    rebuild_publication_lag_validity,
)
from .store import close_memmap, peak_rss_bytes, write_store
from .targets import build_multi_day_targets_into, build_to_close_target
from .universe import (
    build_daily_universe,
    session_calendar,
    v1_pit_coverage_table,
    v1_pit_inactive_exceptions_table,
)

EXPECTED_V1_DATES = 1_248
V1_STORE_START = date(2021, 7, 19)
EXTERNAL_VALIDITY_BOOTSTRAP_REPLICATIONS = 1_000
EXTERNAL_VALIDITY_BOOTSTRAP_CONFIDENCE = 0.95
EXTERNAL_VALIDITY_MIN_NAMES = 20
EXTERNAL_VALIDITY_MIN_NAME_DAYS = 2_000


def _workspace_array(
    directory: Path,
    name: str,
    shape: Sequence[int],
    dtype: np.dtype[np.generic] | type[np.generic],
    *,
    fill: float | bool | None = None,
) -> np.memmap:
    """Create one disk-backed build output without an in-memory duplicate."""

    output = np.lib.format.open_memmap(
        directory / f"{name}.npy",
        mode="w+",
        dtype=np.dtype(dtype),
        shape=tuple(int(value) for value in shape),
    )
    if fill is not None:
        for start in range(0, output.shape[0], 64):
            output[start : start + 64] = fill
    return output


def _copy_workspace_array(
    directory: Path,
    name: str,
    values: NDArray[np.generic],
    *,
    dtype: np.dtype[np.generic] | type[np.generic] | None = None,
) -> np.memmap:
    """Write one family to disk in bounded date chunks and reopen it read-only."""

    source = np.asarray(values)
    destination = _workspace_array(
        directory,
        name,
        source.shape,
        source.dtype if dtype is None else dtype,
    )
    for start in range(0, source.shape[0], 64):
        destination[start : start + 64] = source[start : start + 64]
    close_memmap(destination)
    return np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)


def _copy_selected_workspace_array(
    directory: Path,
    name: str,
    values: NDArray[np.generic],
    rows: NDArray[np.integer],
    *,
    dtype: np.dtype[np.generic] | type[np.generic] | None = None,
) -> np.memmap:
    """Copy selected increasing date rows without advanced-index panel copies."""

    source = np.asarray(values)
    selected = np.asarray(rows, dtype=np.int64)
    destination = _workspace_array(
        directory,
        name,
        (selected.size, *source.shape[1:]),
        source.dtype if dtype is None else dtype,
    )
    for start in range(0, selected.size, 64):
        destination[start : start + 64] = source[selected[start : start + 64]]
    close_memmap(destination)
    return np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)


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
    if source_dates.get_column("date_idx").to_list() != list(
        range(source_dates.height)
    ):
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
            raise ValueError(
                f"v1 fast date absent from daily calendar: {row['trade_date']}"
            )
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
    if (
        bound.get_column("isin").null_count()
        or bound.get_column("isin").n_unique() != bound.height
    ):
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
    feature_values = np.zeros((*shape, len(INTRADAY_DAILY_FEATURES)), dtype=np.float32)
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
            close_anchor_consistent=session_close_valid.copy(),
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


def _target_validity_tables(
    dates: NDArray[np.datetime64],
    valid: NDArray[np.bool_],
    active: NDArray[np.bool_],
    observed: NDArray[np.bool_],
    survival_identities: Sequence[str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Audit target coverage by year and by eventual panel survival."""

    mask = np.asarray(valid, dtype=np.bool_)
    membership = np.asarray(active, dtype=np.bool_)
    seen = np.asarray(observed, dtype=np.bool_)
    if (
        mask.shape != (*membership.shape, len(HORIZONS))
        or seen.shape != membership.shape
    ):
        raise ValueError("target-validity audit axes are misaligned")
    years = dates.astype("datetime64[Y]").astype(np.int64) + 1970
    yearly: list[dict[str, object]] = []
    for year in sorted(set(years.tolist())):
        year_rows = years == year
        for horizon_index, horizon in enumerate(HORIZONS):
            eligible_rows = year_rows.copy()
            eligible_rows[max(0, len(dates) - horizon) :] = False
            denominator_mask = membership[eligible_rows] & seen[eligible_rows]
            denominator = int(denominator_mask.sum())
            numerator = int(
                (mask[eligible_rows, :, horizon_index] & denominator_mask).sum()
            )
            yearly.append(
                {
                    "year": int(year),
                    "horizon_sessions": horizon,
                    "valid_target_name_days": numerator,
                    "observed_member_name_days": denominator,
                    "validity_ratio": numerator / denominator if denominator else 0.0,
                }
            )

    groups = _eventual_survival_groups(dates, seen, survival_identities)
    survival_rows: list[dict[str, object]] = []
    ratios: dict[tuple[str, int], float] = {}
    denominators: dict[tuple[str, int], int] = {}
    for label, names in groups.items():
        for horizon_index, horizon in enumerate(HORIZONS):
            eligible_dates = np.arange(len(dates)) < len(dates) - horizon
            denominator_mask = (
                membership & seen & eligible_dates[:, None] & names[None, :]
            )
            denominator = int(denominator_mask.sum())
            numerator = int((mask[..., horizon_index] & denominator_mask).sum())
            ratio = numerator / denominator if denominator else 0.0
            ratios[(label, horizon)] = ratio
            denominators[(label, horizon)] = denominator
            survival_rows.append(
                {
                    "group": label,
                    "horizon_sessions": horizon,
                    "name_count": int(names.sum()),
                    "valid_target_name_days": numerator,
                    "observed_member_name_days": denominator,
                    "validity_ratio": ratio,
                }
            )
    for horizon in HORIZONS:
        if not all(denominators[(label, horizon)] > 0 for label in groups):
            continue
        gap = abs(
            ratios[("delisted_within_panel", horizon)]
            - ratios[("survives_to_final_year", horizon)]
        )
        if gap > 0.10:
            raise ValueError(
                "target validity is survivor-skewed by more than 10 percentage "
                f"points at horizon {horizon}: {gap:.6f}"
            )
    return pl.DataFrame(yearly), pl.DataFrame(survival_rows)


def _feature_validity_by_survival(
    dates: NDArray[np.datetime64],
    active: NDArray[np.bool_],
    observed: NDArray[np.bool_],
    families: Mapping[str, tuple[NDArray[np.bool_], NDArray[np.bool_]]],
    survival_identities: Sequence[str] | None = None,
    maximum_gap: float | None = 0.05,
) -> pl.DataFrame:
    """Gate feature-mask rates against eventual panel survival.

    The denominator is a family's observable population, supplied separately
    from its feature masks.  This distinguishes a missing optional archive
    from feature invalidation inside rows where that archive is present.
    """

    membership = np.asarray(active, dtype=np.bool_)
    seen = np.asarray(observed, dtype=np.bool_)
    if membership.shape != seen.shape or membership.shape[0] != len(dates):
        raise ValueError("feature-survival axes are misaligned")
    groups = _eventual_survival_groups(dates, seen, survival_identities)
    rows: list[dict[str, object]] = []
    ratios: dict[tuple[str, str], float] = {}
    denominators: dict[tuple[str, str], int] = {}
    for family, (raw_valid, raw_present) in sorted(families.items()):
        valid = np.asarray(raw_valid, dtype=np.bool_)
        present = np.asarray(raw_present, dtype=np.bool_)
        if (
            valid.ndim != 3
            or valid.shape[:2] != seen.shape
            or present.shape != seen.shape
        ):
            raise ValueError(f"feature family {family} is misaligned")
        for label, names in groups.items():
            eligible = membership & seen & present & names[None, :]
            denominator = int(eligible.sum()) * valid.shape[2]
            numerator = int((valid & eligible[..., None]).sum())
            ratio = numerator / denominator if denominator else 0.0
            ratios[(family, label)] = ratio
            denominators[(family, label)] = denominator
            rows.append(
                {
                    "family": family,
                    "group": label,
                    "name_count": int(names.sum()),
                    "valid_feature_cells": numerator,
                    "possible_feature_cells": denominator,
                    "validity_ratio": ratio,
                    "mask_rate": 1.0 - ratio if denominator else None,
                }
            )
        if maximum_gap is not None and all(
            denominators[(family, label)] > 0 for label in groups
        ):
            gap = abs(
                ratios[(family, "delisted_within_panel")]
                - ratios[(family, "survives_to_final_year")]
            )
            if gap > maximum_gap:
                raise ValueError(
                    f"feature validity for {family} is survivor-skewed by more "
                    f"than {100.0 * maximum_gap:g} percentage points: {gap:.6f}"
                )
    return pl.DataFrame(rows)


def _eventual_survival_groups(
    dates: NDArray[np.datetime64],
    observed: NDArray[np.bool_],
    survival_identities: Sequence[str] | None = None,
) -> dict[str, NDArray[np.bool_]]:
    """Classify names by the final observation of their continuation identity."""

    calendar = np.asarray(dates, dtype="datetime64[D]")
    seen = np.asarray(observed, dtype=np.bool_)
    if seen.ndim != 2 or seen.shape[0] != calendar.size:
        raise ValueError("survival-group axes are misaligned")
    identities = (
        tuple(str(index) for index in range(seen.shape[1]))
        if survival_identities is None
        else tuple(str(value) for value in survival_identities)
    )
    if len(identities) != seen.shape[1]:
        raise ValueError("survival identity axis is misaligned")
    years = calendar.astype("datetime64[Y]").astype(np.int64) + 1970
    identity_last: dict[str, int] = {}
    for name, identity in enumerate(identities):
        rows = np.flatnonzero(seen[:, name])
        if rows.size:
            identity_last[identity] = max(identity_last.get(identity, -1), int(rows[-1]))
    final_year = int(years[-1])
    survives = np.asarray(
        [
            identity in identity_last and years[identity_last[identity]] == final_year
            for identity in identities
        ],
        dtype=np.bool_,
    )
    return {
        "delisted_within_panel": ~survives,
        "survives_to_final_year": survives,
    }


def _prior_adv20(
    volume_brl: NDArray[np.floating], observed: NDArray[np.bool_]
) -> NDArray[np.float32]:
    """Return the causal mean BRL volume over the 20 sessions before each row."""

    volume = np.asarray(volume_brl, dtype=np.float64)
    seen = np.asarray(observed, dtype=np.bool_)
    if volume.ndim != 2 or seen.shape != volume.shape:
        raise ValueError("prior-ADV20 axes are misaligned")
    output = np.full(volume.shape, np.nan, dtype=np.float32)
    clean = np.where(seen & np.isfinite(volume) & (volume >= 0.0), volume, 0.0)
    cumulative = np.vstack(
        (np.zeros((1, volume.shape[1]), dtype=np.float64), np.cumsum(clean, axis=0))
    )
    for day in range(20, volume.shape[0]):
        output[day] = ((cumulative[day] - cumulative[day - 20]) / 20.0).astype(
            np.float32
        )
    return output


def _external_feature_validity_by_survival_liquidity(
    dates: NDArray[np.datetime64],
    active: NDArray[np.bool_],
    observed: NDArray[np.bool_],
    prior_adv20: NDArray[np.floating],
    family: str,
    valid: NDArray[np.bool_],
    present: NDArray[np.bool_],
    survival_identities: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Gate survivor-favoring external coverage with name-clustered intervals."""

    membership = np.asarray(active, dtype=np.bool_)
    seen = np.asarray(observed, dtype=np.bool_)
    mask = np.asarray(valid, dtype=np.bool_)
    availability = np.asarray(present, dtype=np.bool_)
    adv = np.asarray(prior_adv20, dtype=np.float64)
    if (
        membership.shape != seen.shape
        or adv.shape != seen.shape
        or availability.shape != seen.shape
        or mask.ndim != 3
        or mask.shape[:2] != seen.shape
        or seen.shape[0] != len(dates)
    ):
        raise ValueError(f"external feature family {family} is misaligned")
    groups = _eventual_survival_groups(dates, seen, survival_identities)
    eligible = membership & seen & availability & np.isfinite(adv)
    pooled_adv = adv[eligible]
    if pooled_adv.size == 0:
        raise ValueError(f"external feature family {family} has no observable population")
    edges = np.quantile(pooled_adv, (0.25, 0.50, 0.75))
    quartile = np.full(seen.shape, -1, dtype=np.int8)
    quartile[eligible] = np.searchsorted(edges, adv[eligible], side="right")
    identities = (
        tuple(str(index) for index in range(seen.shape[1]))
        if survival_identities is None
        else tuple(str(value) for value in survival_identities)
    )
    if len(identities) != seen.shape[1]:
        raise ValueError("external feature survival identity axis is misaligned")
    rows: list[dict[str, object]] = []
    strata = [(0, eligible)] + [
        (quartile_index + 1, quartile == quartile_index)
        for quartile_index in range(4)
    ]
    for quartile_number, stratum in strata:
        ratios: dict[str, float] = {}
        denominators: dict[str, int] = {}
        numerators_by_identity: dict[str, NDArray[np.int64]] = {}
        denominators_by_identity: dict[str, NDArray[np.int64]] = {}
        supported_names: dict[str, int] = {}
        present_name_days: dict[str, int] = {}
        for label, names in groups.items():
            selected = eligible & stratum & names[None, :]
            per_column_days = selected.sum(axis=0, dtype=np.int64)
            per_column_valid = (mask & selected[..., None]).sum(
                axis=(0, 2), dtype=np.int64
            )
            clustered: dict[str, list[int]] = {}
            for column in np.flatnonzero(names):
                identity = identities[int(column)]
                counts = clustered.setdefault(identity, [0, 0])
                counts[0] += int(per_column_valid[column])
                counts[1] += int(per_column_days[column]) * mask.shape[2]
            supported = [counts for counts in clustered.values() if counts[1] > 0]
            cluster_numerators = np.asarray(
                [counts[0] for counts in supported], dtype=np.int64
            )
            cluster_denominators = np.asarray(
                [counts[1] for counts in supported], dtype=np.int64
            )
            present_days = int(per_column_days.sum())
            denominator = int(cluster_denominators.sum())
            numerator = int(cluster_numerators.sum())
            ratio = numerator / denominator if denominator else 0.0
            ratios[label] = ratio
            denominators[label] = denominator
            numerators_by_identity[label] = cluster_numerators
            denominators_by_identity[label] = cluster_denominators
            supported_names[label] = int(cluster_denominators.size)
            present_name_days[label] = present_days
            rows.append(
                {
                    "family": family,
                    "prior_adv20_quartile": quartile_number,
                    "prior_adv20_lower_brl": (
                        None
                        if quartile_number in {0, 1}
                        else float(edges[quartile_number - 2])
                    ),
                    "prior_adv20_upper_brl": (
                        None
                        if quartile_number in {0, 4}
                        else float(edges[quartile_number - 1])
                    ),
                    "group": label,
                    "name_count": int(names.sum()),
                    "supported_continuation_name_count": supported_names[label],
                    "family_present_name_days": present_days,
                    "valid_feature_cells": numerator,
                    "possible_feature_cells": denominator,
                    "validity_ratio": ratio,
                    "mask_rate": 1.0 - ratio if denominator else None,
                    "survivor_minus_delisted_gap": None,
                    "bootstrap_lower_95": None,
                    "bootstrap_upper_95": None,
                    "bootstrap_replications": EXTERNAL_VALIDITY_BOOTSTRAP_REPLICATIONS,
                    "bootstrap_cluster_unit": "continuation_name",
                    "gap_direction": None,
                    "stratified_gate_passed": None,
                    "stratum_is_binding": False,
                    "support_threshold_met": None,
                    "gate_decision": None,
                    "coverage_note": (
                        "Delisted mid-liquidity names have thin lending records: "
                        "5,212 present name-days across 526 names; lending value "
                        "will be coverage-limited for exactly that segment."
                        if family == "sidecar_lending"
                        else ""
                    ),
                }
            )
        if all(denominators[label] > 0 for label in groups):
            gap = (
                ratios["survives_to_final_year"]
                - ratios["delisted_within_panel"]
            )
            direction = (
                "survivor_above_delisted"
                if gap > 0.0
                else "delisted_above_survivor"
            )
            sampled_ratios: dict[str, NDArray[np.float64]] = {}
            for label in groups:
                seed_payload = (
                    "v2-external-validity-name-bootstrap|"
                    f"{family}|{quartile_number}|{label}"
                ).encode("utf-8")
                seed = int.from_bytes(
                    hashlib.sha256(seed_payload).digest()[:8], "little"
                )
                rng = np.random.default_rng(seed)
                cluster_numerators = numerators_by_identity[label]
                cluster_denominators = denominators_by_identity[label]
                draw = rng.integers(
                    0,
                    cluster_numerators.size,
                    size=(
                        EXTERNAL_VALIDITY_BOOTSTRAP_REPLICATIONS,
                        cluster_numerators.size,
                    ),
                )
                sampled_ratios[label] = cluster_numerators[draw].sum(axis=1) / (
                    cluster_denominators[draw].sum(axis=1)
                )
            samples = (
                sampled_ratios["survives_to_final_year"]
                - sampled_ratios["delisted_within_panel"]
            )
            tail = 0.5 * (1.0 - EXTERNAL_VALIDITY_BOOTSTRAP_CONFIDENCE)
            lower, upper = np.quantile(samples, (tail, 1.0 - tail))
            support_met = all(
                supported_names[label] >= EXTERNAL_VALIDITY_MIN_NAMES
                and present_name_days[label] >= EXTERNAL_VALIDITY_MIN_NAME_DAYS
                for label in groups
            )
            binding = quartile_number > 0 and support_met
            passed = not (binding and float(lower) > 0.05)
            decision = (
                "diagnostic_unstratified"
                if quartile_number == 0
                else (
                    "pass"
                    if binding and passed
                    else (
                        "fail_survivor_favoring_interval"
                        if binding
                        else "reported_not_gated_insufficient_support"
                    )
                )
            )
            for row in rows[-len(groups) :]:
                row["survivor_minus_delisted_gap"] = gap
                row["bootstrap_lower_95"] = float(lower)
                row["bootstrap_upper_95"] = float(upper)
                row["gap_direction"] = direction
                row["stratified_gate_passed"] = passed if binding else None
                row["stratum_is_binding"] = binding
                row["support_threshold_met"] = support_met
                row["gate_decision"] = decision
            if binding and float(lower) > 0.05:
                raise ValueError(
                    f"external feature validity for {family} has a supported "
                    "survivor-minus-delisted name-bootstrap lower bound above "
                    "5 percentage points within prior-ADV20 quartile "
                    f"{quartile_number}: {float(lower):.6f}"
                )
        else:
            for row in rows[-len(groups) :]:
                row["stratum_is_binding"] = False
                row["support_threshold_met"] = False
                row["gate_decision"] = (
                    "diagnostic_unstratified_no_two_group_support"
                    if quartile_number == 0
                    else "reported_not_gated_insufficient_support"
                )
    return pl.DataFrame(
        rows,
        schema_overrides={
            "prior_adv20_lower_brl": pl.Float64,
            "prior_adv20_upper_brl": pl.Float64,
            "mask_rate": pl.Float64,
            "survivor_minus_delisted_gap": pl.Float64,
            "bootstrap_lower_95": pl.Float64,
            "bootstrap_upper_95": pl.Float64,
            "gap_direction": pl.String,
            "stratified_gate_passed": pl.Boolean,
            "support_threshold_met": pl.Boolean,
            "gate_decision": pl.String,
            "coverage_note": pl.String,
        },
    )


def _fast_sigma_ratio_table(
    dates: NDArray[np.datetime64],
    fast_sigma: NDArray[np.floating],
    daily_sigma: NDArray[np.floating],
    fast_present: NDArray[np.bool_],
) -> pl.DataFrame:
    fast = np.asarray(fast_sigma, dtype=np.float64)
    daily = np.asarray(daily_sigma, dtype=np.float64)
    present = np.asarray(fast_present, dtype=np.bool_)
    if fast.shape != daily.shape or present.shape != fast.shape:
        raise ValueError("fast/daily sigma audit axes are misaligned")
    years = dates.astype("datetime64[Y]").astype(np.int64) + 1970
    rows: list[dict[str, object]] = []
    for year in sorted(set(years.tolist())):
        mask = (
            (years == year)[:, None]
            & present
            & np.isfinite(fast)
            & np.isfinite(daily)
            & (daily > 0)
        )
        ratio = fast[mask] / daily[mask]
        rows.append(
            {
                "year": int(year),
                "comparable_name_days": int(ratio.size),
                "ratio_p05": float(np.quantile(ratio, 0.05)) if ratio.size else None,
                "ratio_median": float(np.median(ratio)) if ratio.size else None,
                "ratio_p95": float(np.quantile(ratio, 0.95)) if ratio.size else None,
            }
        )
    return pl.DataFrame(rows)


def _sidecar_coverage_table(
    dates: NDArray[np.datetime64],
    groups: Mapping[str, tuple[Sequence[str], NDArray[np.bool_], Sequence[str]]],
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
    anchor_consistent = np.zeros(output_shape, dtype=np.bool_)
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
            entry[target_date, target_isin] = result.entry_open[
                source_date, source_isin
            ]
            entry_valid[target_date, target_isin] = result.entry_open_valid[
                source_date, source_isin
            ]
            session_close[target_date, target_isin] = result.session_close[
                source_date, source_isin
            ]
            session_close_valid[target_date, target_isin] = result.session_close_valid[
                source_date, source_isin
            ]
            realized[target_date, target_isin] = result.realized_daily_vol[
                source_date, source_isin
            ]
            present[target_date, target_isin] = result.fast_present[
                source_date, source_isin
            ]
            anchor_consistent[target_date, target_isin] = (
                result.close_anchor_consistent[source_date, source_isin]
            )
    return IntradayDailyResult(
        values=values,
        valid=valid,
        entry_open=entry,
        entry_open_valid=entry_valid,
        session_close=session_close,
        session_close_valid=session_close_valid,
        realized_daily_vol=realized,
        fast_present=present,
        close_anchor_consistent=anchor_consistent,
    )


def build_daily_store(
    daily: pl.DataFrame,
    actions: pl.DataFrame,
    output_dir: Path,
    *,
    minute_panel: MinutePanel | None = None,
    streamed_intraday: StreamedIntraday | None = None,
    sidecars: Mapping[str, SidecarResult] | None = None,
    stream_intraday: bool = False,
    sidecar_arguments: Sequence[str] = (),
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

    if (
        sum(
            value is not None and value is not False
            for value in (minute_panel, streamed_intraday, stream_intraday)
        )
        > 1
    ):
        raise ValueError("provide exactly one intraday source mode")
    if sidecars is not None and sidecar_arguments:
        raise ValueError("provide materialized or streamed sidecars, not both")
    has_intraday = (
        minute_panel is not None or streamed_intraday is not None or stream_intraday
    )

    v1_isins: tuple[str, ...] = ()
    if v1_assignments is not None:
        v1_isins = tuple(v1_assignments.get_column("isin").cast(pl.String).to_list())
    cash = filter_cash_equities(daily, v1_isins=v1_isins)
    calendar = session_calendar(cash, minimum_traded_names=minimum_calendar_names)
    if not calendar:
        raise ValueError("COTAHIST produced no qualifying sessions")
    panel = panel_from_daily(cash, dates=calendar)
    isin_successions = detect_isin_successions(cash)
    continuation_isins = continuation_identity_axis(panel.isins, isin_successions)
    keep = np.ones(len(panel.dates), dtype=np.bool_)
    if store_start is not None:
        keep &= panel.dates >= np.datetime64(store_start)
    if not keep.any():
        raise ValueError("store_start removes the complete calendar")
    kept_rows = np.flatnonzero(keep)
    kept_dates = panel.dates[kept_rows]
    output_parent = Path(output_dir).resolve().parent
    output_parent.mkdir(parents=True, exist_ok=True)
    workspace_handle = tempfile.TemporaryDirectory(
        prefix=f".{Path(output_dir).name}.arrays-", dir=output_parent
    )
    workspace = Path(workspace_handle.name)
    if v1_assignments is not None:
        verify_v1_mapping(v1_assignments, panel.isins)
    if v1_calendar is not None:
        if not v1_calendar:
            raise ValueError("v1 calendar cannot be empty")
        matching_slice = tuple(
            day for day in calendar if v1_calendar[0] <= day <= v1_calendar[-1]
        )
        if matching_slice != tuple(v1_calendar):
            raise ValueError(
                "v1 date axis differs from the matching COTAHIST calendar slice"
            )

    checked_actions = validate_action_table(actions)
    provider_split, provider_cash_distribution, _ = align_action_arrays(
        checked_actions, panel.dates, panel.isins
    )
    distribution_changed = detect_distribution_changes(
        panel.distribution_number, panel.observed
    )
    detected_actions = detect_cotahist_actions(
        panel.close_brl,
        panel.quantity,
        panel.distribution_number,
        panel.observed,
    )
    target_exclusion_event = (
        detected_actions.cash_event | detected_actions.ambiguous_event
    )
    intraday_action_boundary = detected_actions.split_event
    adjusted = adjust_daily_ohlc(
        panel.open_brl,
        panel.high_brl,
        panel.low_brl,
        panel.close_brl,
        detected_actions.price_ratio,
        detected_actions.split_event,
    )
    adjusted_paths: dict[str, Path] = {}
    for name, values in (
        ("price_adjustment_factor", adjusted.price_factor),
        ("adjusted_open", adjusted.adjusted_open),
        ("adjusted_high", adjusted.adjusted_high),
        ("adjusted_low", adjusted.adjusted_low),
        ("adjusted_close", adjusted.adjusted_close),
    ):
        materialized = _copy_workspace_array(workspace, name, values)
        adjusted_paths[name] = workspace / f"{name}.npy"
        close_memmap(materialized)
    del adjusted
    gc.collect()
    price_adjustment_factor = np.load(
        adjusted_paths["price_adjustment_factor"], mmap_mode="r", allow_pickle=False
    )
    adjusted_open = np.load(
        adjusted_paths["adjusted_open"], mmap_mode="r", allow_pickle=False
    )
    adjusted_high = np.load(
        adjusted_paths["adjusted_high"], mmap_mode="r", allow_pickle=False
    )
    adjusted_low = np.load(
        adjusted_paths["adjusted_low"], mmap_mode="r", allow_pickle=False
    )
    adjusted_close = np.load(
        adjusted_paths["adjusted_close"], mmap_mode="r", allow_pickle=False
    )
    linked_universe_inputs = tuple(
        inherit_linked_history(values, panel.dates, panel.isins, isin_successions)
        for values in (panel.close_brl, panel.volume_brl, panel.observed)
    )
    universe = build_daily_universe(*linked_universe_inputs)
    date_lookup = {value: index for index, value in enumerate(panel.dates)}
    isin_lookup = {value: index for index, value in enumerate(panel.isins)}
    for row in isin_successions.iter_rows(named=True):
        boundary = date_lookup[np.datetime64(row["successor_first_date"], "D")]
        predecessor = isin_lookup[str(row["predecessor_isin"])]
        successor = isin_lookup[str(row["successor_isin"])]
        universe.active[boundary:, predecessor] = False
        universe.active[:boundary, successor] = False
    linked_slow_inputs = tuple(
        inherit_linked_history(values, panel.dates, panel.isins, isin_successions)
        for values in (
            adjusted_open,
            adjusted_high,
            adjusted_low,
            adjusted_close,
            linked_universe_inputs[1],
            panel.trades,
            linked_universe_inputs[2],
            detected_actions.ambiguous_event,
        )
    )
    slow_raw = build_slow_features(
        *linked_slow_inputs[:6],
        linked_slow_inputs[6],
        universe.active,
        panel.dates,
        ambiguous_action=linked_slow_inputs[7],
    )
    slow_sigma = _copy_workspace_array(
        workspace,
        "slow_sigma",
        np.where(slow_raw.valid[..., 8], slow_raw.values[..., 8], np.nan),
        dtype=np.float32,
    )
    slow_values = _workspace_array(
        workspace,
        "slow_values",
        (kept_rows.size, len(panel.isins), len(SLOW_FEATURES)),
        np.float32,
    )
    slow_valid = _workspace_array(
        workspace,
        "slow_valid",
        slow_values.shape,
        np.bool_,
    )
    rank_gauss_panel_into(
        slow_raw.values,
        slow_raw.valid,
        universe.active,
        slow_values,
        slow_valid,
        source_rows=kept_rows,
    )
    inherit_linked_history(
        slow_values,
        kept_dates,
        panel.isins,
        isin_successions,
        copy=False,
    )
    inherit_linked_history(
        slow_valid,
        kept_dates,
        panel.isins,
        isin_successions,
        copy=False,
    )
    del slow_raw
    del linked_slow_inputs
    gc.collect()

    shape = panel.observed.shape
    intraday_values = _workspace_array(
        workspace,
        "intraday_values",
        (kept_rows.size, len(panel.isins), len(INTRADAY_DAILY_FEATURES)),
        np.float32,
        fill=0.0,
    )
    intraday_valid = _workspace_array(
        workspace,
        "intraday_valid",
        intraday_values.shape,
        np.bool_,
        fill=False,
    )
    fast_sigma = np.full(shape, np.nan, dtype=np.float64)
    fast_present = np.zeros(shape, dtype=np.bool_)
    entry = np.full(shape, np.nan, dtype=np.float64)
    entry_valid = np.zeros(shape, dtype=np.bool_)
    realized_daily = np.full(shape, np.nan, dtype=np.float64)
    m1_session_close = np.full(shape, np.nan, dtype=np.float64)
    m1_session_close_valid = np.zeros(shape, dtype=np.bool_)
    close_anchor_consistent = np.zeros(shape, dtype=np.bool_)
    intraday_audit: pl.DataFrame | None = None
    intraday_source_paths: tuple[Path, ...] = ()
    if stream_intraday:
        if v1_assignments is None:
            raise ValueError("streamed intraday construction requires v1 assignments")
        streamed_intraday = stream_intraday_from_assignments(
            v1_assignments, cash, calendar, panel.isins
        )
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
        del native
        m1_session_close = aligned.session_close.copy()
        m1_session_close_valid = aligned.session_close_valid.copy()
        aligned = replace_daily_close_anchors(
            aligned, panel.close_brl, panel.observed, copy_buffers=False
        )
        aligned = mask_action_boundaries(aligned, intraday_action_boundary)
        rank_gauss_panel_into(
            aligned.values,
            aligned.valid,
            universe.active,
            intraday_values,
            intraday_valid,
            source_rows=kept_rows,
        )
        fast_sigma = np.where(aligned.valid[..., 14], aligned.values[..., 14], np.nan)
        fast_present = aligned.fast_present
        entry = aligned.entry_open
        entry_valid = aligned.entry_open_valid
        realized_daily = aligned.realized_daily_vol
        close_anchor_consistent = aligned.close_anchor_consistent
    elif streamed_intraday is not None:
        intraday_audit = streamed_intraday.audit
        intraday_source_paths = streamed_intraday.source_paths
        aligned = streamed_intraday.result
        if aligned.values.shape[:2] != shape:
            raise ValueError("streamed intraday derivatives are misaligned")
        m1_session_close = aligned.session_close.copy()
        m1_session_close_valid = aligned.session_close_valid.copy()
        aligned = replace_daily_close_anchors(
            aligned, panel.close_brl, panel.observed, copy_buffers=False
        )
        aligned = mask_action_boundaries(aligned, intraday_action_boundary)
        rank_gauss_panel_into(
            aligned.values,
            aligned.valid,
            universe.active,
            intraday_values,
            intraday_valid,
            source_rows=kept_rows,
        )
        fast_sigma = np.where(aligned.valid[..., 14], aligned.values[..., 14], np.nan)
        fast_present = aligned.fast_present
        entry = aligned.entry_open
        entry_valid = aligned.entry_open_valid
        realized_daily = aligned.realized_daily_vol
        close_anchor_consistent = aligned.close_anchor_consistent

    to_close_arrays: dict[str, NDArray[np.generic]] = {}
    if has_intraday:
        to_close = build_to_close_target(
            entry,
            np.where(panel.observed, panel.close_brl, np.nan),
            realized_daily,
            universe.active,
            fast_present & entry_valid & close_anchor_consistent,
        )
        for name, values in (
            ("target_to_close", to_close.target),
            ("target_to_close_valid", to_close.valid),
            (
                "target_to_close_normalized_residual",
                to_close.normalized_residual,
            ),
            ("target_to_close_raw_log_return", to_close.raw_log_return),
            ("m1_cotahist_close_consistent_mask", close_anchor_consistent),
        ):
            to_close_arrays[name] = _copy_selected_workspace_array(
                workspace, f"store_{name}", values, kept_rows
            )
        del to_close
        if "aligned" in locals():
            del aligned
        streamed_intraday = None
        gc.collect()

    inherit_linked_history(
        intraday_values,
        kept_dates,
        panel.isins,
        isin_successions,
        copy=False,
    )
    inherit_linked_history(
        intraday_valid,
        kept_dates,
        panel.isins,
        isin_successions,
        copy=False,
    )

    target_shape = (kept_rows.size, len(panel.isins), len(HORIZONS))
    target_primary = _workspace_array(
        workspace, "target_primary", target_shape, np.float32
    )
    target_valid = _workspace_array(workspace, "target_valid", target_shape, np.bool_)
    target_normalized_residual = _workspace_array(
        workspace, "target_normalized_residual", target_shape, np.float32
    )
    target_raw_midrank = _workspace_array(
        workspace, "target_raw_midrank", target_shape, np.float32
    )
    target_raw_valid = _workspace_array(
        workspace, "target_raw_valid", target_shape, np.bool_
    )
    target_raw_log_return = _workspace_array(
        workspace, "target_raw_log_return", target_shape, np.float32
    )
    build_multi_day_targets_into(
        adjusted_close,
        universe.active,
        slow_sigma,
        target_exclusion_event,
        primary=target_primary,
        primary_valid=target_valid,
        normalized_residual=target_normalized_residual,
        raw_midrank=target_raw_midrank,
        raw_valid=target_raw_valid,
        raw_log_return=target_raw_log_return,
        source_rows=kept_rows,
    )
    full_store_arrays: dict[str, NDArray[np.generic]] = {
        "active": universe.active,
        "observed": panel.observed,
        "raw_open": panel.open_brl,
        "raw_high": panel.high_brl,
        "raw_low": panel.low_brl,
        "raw_close": panel.close_brl,
        "adjusted_open": adjusted_open,
        "adjusted_high": adjusted_high,
        "adjusted_low": adjusted_low,
        "adjusted_close": adjusted_close,
        "price_adjustment_factor": price_adjustment_factor,
        "volume_brl": panel.volume_brl,
        "trade_count": panel.trades,
        "quantity": panel.quantity,
        "distribution_number": panel.distribution_number,
        "distribution_change_mask": distribution_changed,
        "detected_event_mask": detected_actions.event_candidate,
        "detected_split_mask": detected_actions.split_event,
        "detected_cash_event_mask": detected_actions.cash_event,
        "ambiguous_action_mask": detected_actions.ambiguous_event,
        "target_exclusion_event_mask": target_exclusion_event,
        "intraday_action_boundary_mask": intraday_action_boundary,
    }
    arrays: dict[str, NDArray[np.generic]] = {
        name: _copy_selected_workspace_array(
            workspace, f"store_{name}", values, kept_rows
        )
        for name, values in full_store_arrays.items()
    }
    arrays.update(
        {
            "slow_values": slow_values,
            "slow_valid": slow_valid,
            "intraday_values": intraday_values,
            "intraday_valid": intraday_valid,
            "fast_present": _copy_selected_workspace_array(
                workspace, "store_fast_present", fast_present, kept_rows
            ),
            "target_primary": target_primary,
            "target_valid": target_valid,
            "target_normalized_residual": target_normalized_residual,
            "target_raw_midrank": target_raw_midrank,
            "target_raw_valid": target_raw_valid,
            "target_raw_log_return": target_raw_log_return,
        }
    )
    arrays.update(to_close_arrays)
    feature_names: dict[str, Sequence[str]] = {
        "slow": SLOW_FEATURES,
        "intraday": INTRADAY_DAILY_FEATURES,
        "horizons": tuple(str(value) for value in HORIZONS),
    }
    coverage_tables = [
        _coverage_table(kept_dates, slow_valid, SLOW_FEATURES, family="slow"),
        _coverage_table(
            kept_dates,
            intraday_valid,
            INTRADAY_DAILY_FEATURES,
            family="intraday",
        ),
    ]

    def iter_sidecars() -> Iterator[tuple[str, SidecarResult]]:
        if sidecars:
            yield from sorted(sidecars.items())
            return
        groups = sorted({argument.split("=", 1)[0] for argument in sidecar_arguments})
        for group in groups:
            arguments = [
                argument
                for argument in sidecar_arguments
                if argument.split("=", 1)[0] == group
            ]
            parsed = _parse_sidecars(
                arguments,
                panel.dates,
                panel.isins,
                v1_assignments,
                panel.volume_brl,
            )
            yield group, parsed.pop(group)

    sidecar_coverage_frames: list[pl.DataFrame] = []
    sidecar_survival_frames: list[pl.DataFrame] = []
    sidecar_liquidity_frames: list[pl.DataFrame] = []
    sidecar_contemporaneity_rows: list[dict[str, object]] = []
    prior_adv20 = _prior_adv20(
        linked_universe_inputs[1][keep], linked_universe_inputs[2][keep]
    )
    for group, result in iter_sidecars():
        if result.values.shape[:2] != shape:
            raise ValueError(f"sidecar {group} does not match the store axes")
        if not result.publication_lag_reproduced:
            raise ValueError(
                f"sidecar {group} lacks an independent publication-lag validity proof"
            )
        sidecar_contemporaneity_rows.append(
            {
                "family": f"sidecar_{group}",
                "publication_lag_validity_reproduced": True,
                "publication_lag_valid_cells": result.publication_lag_valid_cells,
                "publication_lag_source_rows": result.publication_lag_source_rows,
                "d_plus_one_rows_checked": result.d_plus_one_rows_checked,
                "d_plus_one_violations": result.d_plus_one_violations,
                "availability_contract": (
                    "raw archive available_date/decision coordinate; daily source "
                    "archives are D+1-lagged upstream"
                ),
            }
        )
        values = _workspace_array(
            workspace,
            f"sidecar_{group}_values",
            (kept_rows.size, len(panel.isins), len(result.feature_names)),
            np.float32,
        )
        valid = _workspace_array(
            workspace,
            f"sidecar_{group}_valid",
            values.shape,
            np.bool_,
        )
        rank_gauss_panel_into(
            result.values,
            result.valid,
            universe.active,
            values,
            valid,
            source_rows=kept_rows,
        )
        inherit_linked_history(
            values,
            kept_dates,
            panel.isins,
            isin_successions,
            copy=False,
        )
        inherit_linked_history(
            valid,
            kept_dates,
            panel.isins,
            isin_successions,
            copy=False,
        )
        arrays[f"sidecar_{group}_values"] = values
        arrays[f"sidecar_{group}_valid"] = valid
        feature_names[f"sidecar_{group}"] = result.feature_names
        raw_valid = result.valid[kept_rows]
        sidecar_coverage_frames.append(
            _sidecar_coverage_table(
                kept_dates,
                {
                    group: (
                        result.feature_names,
                        raw_valid,
                        result.archive_semantics_available,
                    )
                },
                universe.active[keep],
            )
        )
        sidecar_survival_frames.append(
            _feature_validity_by_survival(
                kept_dates,
                universe.active[keep],
                panel.observed[keep],
                {
                    f"sidecar_{group}": (
                        raw_valid,
                        raw_valid.any(axis=2),
                    )
                },
                continuation_isins,
                maximum_gap=None,
            )
        )
        sidecar_liquidity_frames.append(
            _external_feature_validity_by_survival_liquidity(
                kept_dates,
                universe.active[keep],
                panel.observed[keep],
                prior_adv20,
                f"sidecar_{group}",
                raw_valid,
                raw_valid.any(axis=2),
                continuation_isins,
            )
        )
        del raw_valid, result
        gc.collect()
    del prior_adv20, linked_universe_inputs
    gc.collect()

    tables = {
        "security_master": build_security_master(
            cash, succession_links=isin_successions
        ),
        "isin_succession_links": isin_successions,
        "feature_coverage": pl.concat(coverage_tables),
        "sidecar_coverage": (
            pl.concat(sidecar_coverage_frames)
            if sidecar_coverage_frames
            else _sidecar_coverage_table(kept_dates, {}, universe.active[keep])
        ),
        "universe_size": pl.DataFrame(
            {
                "trade_date": panel.dates,
                "member_count": universe.active.sum(axis=1).astype(np.int32),
            }
        ),
        "corporate_action_split_review": split_review_table(
            panel.dates.astype(object),
            panel.isins,
            detected_actions.split_event,
            provider_split,
        ),
        "cotahist_action_classification": cotahist_action_classification_table(
            panel.dates,
            panel.isins,
            distribution_changed,
            detected_actions,
        ),
        "corporate_action_calendar_alignment": action_calendar_alignment_table(
            checked_actions, panel.dates, panel.isins
        ),
        "corporate_action_cash_unit_adjustment": cash_unit_adjustment_audit(
            checked_actions
        ),
        "corporate_action_dividend_close_drop": dividend_close_drop_audit(
            panel.dates,
            panel.isins,
            provider_cash_distribution,
            panel.close_brl,
            panel.observed,
        ),
        "provider_split_detection_audit": provider_split_detection_audit(
            panel.dates,
            panel.isins,
            detected_actions.split_event,
            checked_actions,
            action_acquisition_audit,
        ),
        "fast_rv_to_yang_zhang_ratio": _fast_sigma_ratio_table(
            panel.dates, fast_sigma, slow_sigma, fast_present
        ),
        "corporate_action_coverage": action_coverage_table(
            checked_actions,
            panel.dates,
            panel.isins,
            action_acquisition_audit,
        ),
        "corporate_actions": checked_actions,
        "sidecar_contemporaneity": pl.DataFrame(
            sidecar_contemporaneity_rows,
            schema={
                "family": pl.String,
                "publication_lag_validity_reproduced": pl.Boolean,
                "publication_lag_valid_cells": pl.Int64,
                "publication_lag_source_rows": pl.Int64,
                "d_plus_one_rows_checked": pl.Int64,
                "d_plus_one_violations": pl.Int64,
                "availability_contract": pl.String,
            },
        ),
        "external_feature_validity_by_survival_adv20_quartile": (
            pl.concat(sidecar_liquidity_frames)
            if sidecar_liquidity_frames
            else pl.DataFrame()
        ),
    }
    if v1_assignments is not None:
        v1_isins = tuple(v1_assignments.get_column("isin").cast(pl.String).to_list())
        tables["v1_pit_active_coverage"] = v1_pit_coverage_table(
            panel.dates,
            panel.isins,
            universe.active,
            v1_isins,
        )
        tables["v1_pit_inactive_exceptions"] = v1_pit_inactive_exceptions_table(
            panel.dates,
            panel.isins,
            universe.active,
            v1_isins,
        )
    if action_acquisition_audit is not None:
        tables["corporate_action_acquisition_audit"] = action_acquisition_audit
    if has_intraday:
        detected_split_factor = np.ones(shape, dtype=np.float64)
        detected_split_factor[detected_actions.split_event] = (
            1.0 / detected_actions.price_ratio[detected_actions.split_event]
        )
        m1_adjustment = audit_m1_adjustment_status(
            panel.dates,
            panel.isins,
            m1_session_close,
            panel.close_brl,
            adjusted_close,
            detected_split_factor,
            np.zeros(shape, dtype=np.float64),
        )
        tables["m1_adjustment_audit"] = m1_adjustment
        tables["m1_price_adjusted_segments"] = m1_adjustment.filter(
            pl.col("status") == "price_adjusted"
        )
        tables["m1_cotahist_mismatch_by_year"] = m1_cotahist_mismatch_by_year(
            panel.dates,
            m1_session_close,
            m1_session_close_valid,
            panel.close_brl,
            panel.observed,
        )
    if intraday_audit is not None:
        tables["m1_source_audit"] = intraday_audit
    tables["feature_coverage"] = pl.concat(
        [
            _coverage_table(
                kept_dates,
                slow_valid,
                SLOW_FEATURES,
                family="slow",
            ),
            _coverage_table(
                kept_dates,
                intraday_valid,
                INTRADAY_DAILY_FEATURES,
                family="intraday",
            ),
        ]
    )
    tables["universe_size"] = tables["universe_size"].filter(
        pl.col("trade_date") >= kept_dates[0].astype(object)
    )
    (
        tables["target_validity_by_year"],
        tables["target_validity_by_survival"],
    ) = _target_validity_tables(
        kept_dates,
        target_valid,
        universe.active[keep],
        panel.observed[keep],
        continuation_isins,
    )
    feature_gate_families: dict[str, tuple[NDArray[np.bool_], NDArray[np.bool_]]] = {
        "slow_return": (slow_valid[..., :7], panel.observed[keep]),
        "slow_volatility": (slow_valid[..., 7:17], panel.observed[keep]),
        "slow_liquidity_and_state": (
            slow_valid[..., 17:27],
            panel.observed[keep],
        ),
        "slow_peer": (slow_valid[..., 27:], panel.observed[keep]),
        "classification_masks": (
            np.broadcast_to(
                panel.observed[keep, :, None],
                (*panel.observed[keep].shape, 7),
            ),
            panel.observed[keep],
        ),
    }
    if has_intraday:
        feature_gate_families["intraday"] = (
            intraday_valid,
            fast_present[keep],
        )
    base_feature_survival = _feature_validity_by_survival(
        kept_dates,
        universe.active[keep],
        panel.observed[keep],
        feature_gate_families,
        continuation_isins,
    )
    tables["feature_validity_by_survival"] = pl.concat(
        [base_feature_survival, *sidecar_survival_frames]
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
    build_peak_rss = peak_rss_bytes()
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
        "isin_succession_link_count": isin_successions.height,
        "survival_identity": (
            "root ISIN after exact same-ticker consecutive-session COTAHIST succession"
        ),
        "feature_history_identity": (
            "successor ISIN inherits predecessor rows strictly before its first session"
        ),
        "survivorship_gates": {
            "internally_derived_feature_family_max_gap": 0.05,
            "target_family_max_gap": 0.10,
            "external_sidecar_validity": (
                "independent publication-lag mask identity plus a 1000-rep "
                "name-clustered interval on survivor-minus-delisted validity "
                "within pooled causal prior-ADV20 quartiles; binding only with "
                "at least 20 names and 2000 present name-days in both groups; "
                "fail only when the 95% lower bound exceeds +0.05"
            ),
            "external_sidecar_name_bootstrap_replications": (
                EXTERNAL_VALIDITY_BOOTSTRAP_REPLICATIONS
            ),
            "external_sidecar_name_bootstrap_confidence": (
                EXTERNAL_VALIDITY_BOOTSTRAP_CONFIDENCE
            ),
            "external_sidecar_min_names_per_group": EXTERNAL_VALIDITY_MIN_NAMES,
            "external_sidecar_min_name_days_per_group": (
                EXTERNAL_VALIDITY_MIN_NAME_DAYS
            ),
        },
        "return_definition": (
            "COTAHIST price return adjusted only for detected split/bonus "
            "boundaries; provider actions are audit-only"
        ),
        "future_total_return_variant": (
            "registered but not implemented: survivor-subset sensitivity only"
        ),
        "cotahist_provenance": {
            "raw_archives": source_records(cotahist_raw_sources),
            "parse_audit": (
                source_records((cotahist_parse_audit,))[0]
                if cotahist_parse_audit is not None
                else None
            ),
        },
        "build_peak_rss_bytes": build_peak_rss,
        "build_peak_rss_gib": build_peak_rss / (1024**3),
    }
    options_composition = tables[
        "external_feature_validity_by_survival_adv20_quartile"
    ]
    if options_composition.width:
        options_composition = options_composition.filter(
            pl.col("family") == "sidecar_options"
        )
        if options_composition.height:
            unstratified = options_composition.filter(
                pl.col("prior_adv20_quartile") == 0
            ).row(0, named=True)
            binding = options_composition.filter(
                pl.col("stratum_is_binding")
            )
            reported = options_composition.filter(
                (pl.col("prior_adv20_quartile") > 0)
                & (~pl.col("stratum_is_binding"))
            )
            metadata["options_validity_composition_signature"] = {
                "unstratified_survivor_minus_delisted_gap": unstratified[
                    "survivor_minus_delisted_gap"
                ],
                "unstratified_absolute_gap": abs(
                    unstratified["survivor_minus_delisted_gap"]
                ),
                "unstratified_direction": unstratified["gap_direction"],
                "binding_quartile_count": (
                    binding.get_column("prior_adv20_quartile").n_unique()
                    if binding.height
                    else 0
                ),
                "reported_nonbinding_quartile_count": (
                    reported.get_column("prior_adv20_quartile").n_unique()
                    if reported.height
                    else 0
                ),
                "stratified_prior_adv20_gate_passed": (
                    bool(binding.get_column("stratified_gate_passed").all())
                    if binding.height
                    else True
                ),
                "interpretation": (
                    "liquidity-composition signature, not a leakage signature"
                ),
            }
    if "v1_pit_active_coverage" in tables:
        active_counts = tables["v1_pit_active_coverage"].get_column("active_v1_count")
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
    result = write_store(
        output_dir,
        dates=kept_dates,
        isins=panel.isins,
        arrays=arrays,
        feature_names=feature_names,
        sources=source_records(
            (
                *source_paths,
                *intraday_source_paths,
            )
        ),
        metadata=metadata,
        tables=tables,
    )
    for value in arrays.values():
        close_memmap(value)
    for value in (
        price_adjustment_factor,
        adjusted_open,
        adjusted_high,
        adjusted_low,
        adjusted_close,
        slow_sigma,
    ):
        close_memmap(value)
    workspace_handle.cleanup()
    return result


def load_minute_npz(path: Path) -> MinutePanel:
    archive = np.load(path, allow_pickle=False)
    required = {"dates", "isins", "open", "high", "low", "close", "volume", "observed"}
    if not required.issubset(archive.files):
        raise ValueError(
            f"minute archive keys missing: {sorted(required - set(archive.files))}"
        )
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
    parser = argparse.ArgumentParser(
        description="Build the immutable Brazil-RV v2 daily store"
    )
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
    if manifest.get("schema") != "V2_CORPORATE_ACTIONS_V2":
        raise ValueError("corporate-action acquisition manifest has wrong schema")
    paths: dict[str, Path] = {}
    for key in (
        "security_master",
        "actions",
        "acquisition_audit",
        "cash_unit_adjustment_audit",
    ):
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
            "events": ("event_itr_dfp_recent_5s",),
            "options": ("source_trade_date",),
            "fundamentals": ("source_receipt_date",),
        }.get(group, ())
        value_columns = sorted(
            {
                *value_columns,
                *(column for column in derivation_columns if column in schema),
            }
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
            f"{column}_mask" for column in value_columns if f"{column}_mask" in schema
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
            lazy = lazy.filter(pl.col("isin").is_in(pl.Series("isin", isins).implode()))
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
            source = pl.DataFrame(schema={"available_date": pl.Date, "isin": pl.String})
        identity = "isin" if "isin" in source.columns else "security_id"
        keys = ["available_date", identity]
        for optional in ("decision_idx", "available_timestamp", "delivery_timestamp"):
            if optional in source.columns:
                keys.append(optional)
        value_columns = [column for column in source.columns if column not in keys]
        source = source.group_by(keys, maintain_order=True).agg(
            pl.col(column).drop_nulls().last().alias(column) for column in value_columns
        )
        source = derive_known_archive_features(
            source,
            dates,
            isins,
            group=group,
            daily_volume_brl=daily_volume_brl,
        )
        materialized = materialize_known_archive(source, dates, isins, group=group)
        rebuilt = rebuild_publication_lag_validity(
            source,
            dates,
            isins,
            group=group,
            feature_columns=available_archive_mapping(group, source.columns),
            date_only_available_before_decision=True,
        )
        if not np.array_equal(materialized.valid, rebuilt):
            differing = int(np.count_nonzero(materialized.valid != rebuilt))
            raise ValueError(
                f"sidecar {group} validity is not reproducible from its "
                f"publication-lagged archive: {differing} cells differ"
            )
        lag_columns = [
            column
            for column in ("source_trade_date", "source_position_date")
            if column in source.columns
        ]
        lag_rows = 0
        lag_violations = 0
        for column in lag_columns:
            comparable = source.filter(
                pl.col(column).is_not_null() & pl.col("available_date").is_not_null()
            )
            lag_rows += comparable.height
            lag_violations += comparable.filter(
                pl.col("available_date") <= pl.col(column)
            ).height
        if lag_violations:
            raise ValueError(
                f"sidecar {group} has {lag_violations} daily rows not delayed to D+1"
            )
        output[group] = replace(
            materialized,
            publication_lag_reproduced=True,
            publication_lag_valid_cells=int(rebuilt.sum()),
            publication_lag_source_rows=source.height,
            d_plus_one_rows_checked=lag_rows,
            d_plus_one_violations=lag_violations,
        )
        del rebuilt
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
    if (
        not action_master.select(master_columns)
        .sort(master_columns)
        .equals(expected_master.select(master_columns).sort(master_columns))
    ):
        raise ValueError(
            "corporate-action security master differs from the COTAHIST foundation"
        )
    minute = load_minute_npz(args.minute_npz) if args.minute_npz else None
    # The builder materializes M1 and sidecar families only when their turn is
    # reached, then releases each raw family after writing its normalized
    # memmap. Avoid retaining a second daily panel in this acquisition scope.
    del foundation, action_master, expected_master
    output = build_daily_store(
        daily,
        actions,
        args.output_dir,
        minute_panel=minute,
        stream_intraday=minute is None,
        sidecar_arguments=args.sidecar,
        action_acquisition_audit=acquisition_audit,
        v1_assignments=assignments,
        v1_calendar=v1_calendar,
        source_paths=(
            *paths,
            assignment_path,
            *action_sources,
            *((args.minute_npz,) if args.minute_npz else ()),
            *(Path(value.split("=", 1)[1]) for value in args.sidecar),
        ),
        implementation_commit=args.implementation_commit,
        cotahist_raw_sources=raw_sources,
        cotahist_parse_audit=args.cotahist_parse_audit,
        v1_fast_store=args.v1_store,
    )
    print(json.dumps({"store": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
