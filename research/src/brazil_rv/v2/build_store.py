from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from pathlib import Path, PureWindowsPath
from typing import Mapping, Sequence

import numpy as np
import polars as pl
from numpy.typing import NDArray

from .contract import (
    DEVELOPMENT_END,
    DECISION_FEATURE_CONTRACT,
    FEATURE_AGE_CONTRACT,
    HORIZONS,
    INTRADAY_DAILY_FEATURES,
    SIDECAR_FEATURES,
    SLOW_FEATURES,
    STORE_START,
)
from .corporate_actions import (
    DetectedActionResult,
    InferredActionResult,
    VerifiedActionTerm,
    action_coverage_resolved_mask,
    action_calendar_alignment_table,
    action_coverage_table,
    align_action_payment_sessions,
    align_action_arrays,
    align_decision_known_action_terms,
    align_verified_action_terms,
    build_shareholder_wealth_ohlc_into,
    cash_unit_adjustment_audit,
    cotahist_action_classification_table,
    dividend_close_drop_audit,
    detect_cotahist_actions,
    detect_distribution_changes,
    infer_cotahist_action_terms,
    m1_cotahist_mismatch_by_year,
    provider_actions_to_verified_terms,
    provider_split_detection_audit,
    split_review_table,
    validate_action_table,
    validate_verified_action_terms,
    verified_conversion_terms_from_links,
    verified_action_terms_to_table,
)
from .data_foundation import (
    build_security_master,
    continuation_identity_axis,
    detect_isin_successions,
    load_isin_link_allowlist,
    load_cotahist,
    panel_from_daily,
    prepare_cash_equities,
    source_records,
)
from .decision_clock import (
    SessionDefinition,
    assert_calendar_complete,
    calendar_completeness_table,
    load_session_schedule,
    next_session_decision_cutoffs,
    schedule_source_label,
    schedule_frame,
)
from .features import build_slow_features_into, wealth_chain_restart_mask
from .feature_spec import (
    FeatureSpec,
    feature_schema_sha256,
    feature_specs,
    native_fast_feature_specs,
    observation_age_sessions_into,
    transform_feature_panel_into,
)
from .intraday_features import (
    IntradayDailyResult,
    NATIVE_FAST_FEATURES,
    build_intraday_daily_features,
    build_native_fast_features_into,
    detect_open_gap_boundaries,
    decision_action_boundaries,
    shareholder_reference_returns,
)
from .sidecars import (
    SidecarResult,
    available_archive_mapping,
    derive_known_archive_features,
    materialize_known_archive,
    rebuild_publication_lag_validity,
)
from .store import (
    available_memory_status_bytes,
    close_memmap,
    peak_rss_bytes,
    write_store,
)
from .targets import (
    build_economic_multi_day_targets_into,
    build_to_close_target,
)
from .universe import (
    build_daily_universe,
)

EXTERNAL_VALIDITY_BOOTSTRAP_REPLICATIONS = 1_000
EXTERNAL_VALIDITY_BOOTSTRAP_CONFIDENCE = 0.95
EXTERNAL_VALIDITY_MIN_NAMES = 20
EXTERNAL_VALIDITY_MIN_NAME_DAYS = 2_000
MINIMUM_BUILD_FREE_MEMORY_BYTES = 10 * 1024**3
COMMON_STATE_DIAGNOSTICS = (
    "recent_market_log_return",
    "median_raw_daily_volatility",
    "raw_cross_sectional_return_dispersion",
)


@dataclass(frozen=True)
class _DecisionContinuationInputs:
    close_brl: NDArray[np.float64]
    volume_brl: NDArray[np.float64]
    trades: NDArray[np.float64]
    observed: NDArray[np.bool_]
    trade_observed: NDArray[np.bool_]
    activity_valid: NDArray[np.bool_]
    ambiguous_action: NDArray[np.bool_]
    claim_owner: NDArray[np.bool_]


def _action_alignment_role_table(
    terms: Sequence[VerifiedActionTerm],
    dates: NDArray[np.datetime64],
    decision_timestamps: Sequence[datetime | None],
) -> pl.DataFrame:
    """Audit retrospective settlement versus historical feature availability."""

    calendar = tuple(np.asarray(dates, dtype="datetime64[D]").astype(object))
    decisions = dict(zip(calendar, decision_timestamps, strict=True))
    rows: list[dict[str, object]] = []
    for term in validate_verified_action_terms(terms):
        event_date = (
            term.effective_date
            if term.shares_per_prior_share != 1.0
            or term.action_type == "simple_conversion"
            else term.ex_date
        )
        cutoff = decisions.get(event_date)
        rows.append(
            {
                "action_type": term.action_type,
                "isin": term.isin,
                "resulting_isin": term.resulting_isin,
                "event_date": event_date,
                "available_at": term.available_at,
                "source": term.source,
                "evidence": term.evidence,
                "resolved_term": term.resolved,
                "following_decision_in_store": cutoff is not None,
                "known_by_following_decision": (
                    cutoff is not None and term.available_at <= cutoff
                ),
                "retrospective_role": "outcome_and_accounting",
                "decision_time_role": "daily_row_input_at_following_decision_if_known",
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "action_type": pl.String,
            "isin": pl.String,
            "resulting_isin": pl.String,
            "event_date": pl.Date,
            "available_at": pl.Datetime(time_zone="UTC"),
            "source": pl.String,
            "evidence": pl.String,
            "resolved_term": pl.Boolean,
            "following_decision_in_store": pl.Boolean,
            "known_by_following_decision": pl.Boolean,
            "retrospective_role": pl.String,
            "decision_time_role": pl.String,
        },
    )


def _route_decision_known_continuations(
    *,
    dates: NDArray[np.datetime64],
    isins: Sequence[str],
    links: pl.DataFrame,
    decision_timestamps: Sequence[datetime],
    raw_close: NDArray[np.floating],
    volume_brl: NDArray[np.floating],
    trades: NDArray[np.floating],
    observed: NDArray[np.bool_],
    trade_observed: NDArray[np.bool_],
    activity_valid: NDArray[np.bool_],
    ambiguous_action: NDArray[np.bool_],
    shareholder_wealth_arrays: Sequence[NDArray[np.generic]],
) -> _DecisionContinuationInputs:
    """Route explicitly verified conversions without rewriting unknown history.

    Price/wealth, monetary liquidity, trade count, and validity have different
    units, so this is intentionally not a generic history copier.  A link
    known by its effective-session decision carries the predecessor's causal
    lookback into the successor.  A link learned later may retire the
    predecessor from that later decision onward, but cannot backfill model
    inputs that were unavailable historically.
    """

    calendar = np.asarray(dates, dtype="datetime64[D]")
    shape = (calendar.size, len(isins))
    cutoffs = tuple(decision_timestamps)
    raw_arrays = tuple(
        np.asarray(value)
        for value in (
            raw_close,
            volume_brl,
            trades,
            observed,
            trade_observed,
            activity_valid,
            ambiguous_action,
        )
    )
    if len(cutoffs) != calendar.size or any(
        value.shape != shape for value in raw_arrays
    ):
        raise ValueError("continuation routing inputs are misaligned")
    wealth_arrays = tuple(np.asarray(value) for value in shareholder_wealth_arrays)
    if any(value.shape != shape for value in wealth_arrays):
        raise ValueError("shareholder-wealth continuation inputs are misaligned")
    if links.is_empty():
        return _DecisionContinuationInputs(
            close_brl=raw_arrays[0],
            volume_brl=raw_arrays[1],
            trades=raw_arrays[2],
            observed=raw_arrays[3],
            trade_observed=raw_arrays[4],
            activity_valid=raw_arrays[5],
            ambiguous_action=raw_arrays[6],
            claim_owner=np.ones(shape, dtype=np.bool_),
        )
    if any(not value.flags.writeable for value in wealth_arrays):
        raise ValueError("shareholder-wealth continuation destinations are read-only")

    linked_close = np.asarray(raw_close, dtype=np.float64).copy()
    linked_volume = np.asarray(volume_brl, dtype=np.float64).copy()
    linked_trades = np.asarray(trades, dtype=np.float64).copy()
    linked_observed = np.asarray(observed, dtype=np.bool_).copy()
    linked_trade_observed = np.asarray(trade_observed, dtype=np.bool_).copy()
    linked_activity_valid = np.asarray(activity_valid, dtype=np.bool_).copy()
    linked_ambiguous = np.asarray(ambiguous_action, dtype=np.bool_).copy()
    claim_owner = np.ones(shape, dtype=np.bool_)
    date_lookup = {value: index for index, value in enumerate(calendar)}
    isin_lookup = {str(value): index for index, value in enumerate(isins)}

    for row in links.sort("successor_first_date").iter_rows(named=True):
        predecessor = isin_lookup.get(str(row["predecessor_isin"]))
        successor = isin_lookup.get(str(row["successor_isin"]))
        boundary = date_lookup.get(np.datetime64(row["successor_first_date"], "D"))
        first_known = row["first_known_at"]
        if predecessor is None or successor is None or boundary is None:
            raise ValueError("verified continuation is outside the store axes")
        if not isinstance(first_known, datetime) or first_known.tzinfo is None:
            raise ValueError(
                "verified continuation first_known_at must be timezone-aware"
            )
        known_slots = [
            index
            for index in range(boundary, calendar.size)
            if cutoffs[index] >= first_known
        ]
        if known_slots:
            claim_owner[known_slots[0] :, predecessor] = False
        if first_known > cutoffs[boundary]:
            continue

        q = float(row["shares_received_per_prior_share"])
        d = float(row["cash_entitlement_per_prior_share"])
        if not np.isfinite(q) or q <= 0.0 or not np.isfinite(d) or d < 0.0:
            raise ValueError("verified continuation q/d terms are invalid")
        claim_owner[:boundary, successor] = False
        for wealth in wealth_arrays:
            wealth[:, successor] = wealth[:, predecessor]
        linked_volume[:boundary, successor] = linked_volume[:boundary, predecessor]
        linked_trades[:boundary, successor] = linked_trades[:boundary, predecessor]
        linked_observed[:boundary, successor] = linked_observed[:boundary, predecessor]
        linked_trade_observed[:boundary, successor] = linked_trade_observed[
            :boundary, predecessor
        ]
        linked_activity_valid[:boundary, successor] = linked_activity_valid[
            :boundary, predecessor
        ]
        linked_ambiguous[:boundary, successor] = linked_ambiguous[
            :boundary, predecessor
        ]
        linked_close[:boundary, successor] = linked_close[:boundary, predecessor] / q
        prior_prints = np.flatnonzero(linked_observed[:boundary, predecessor])
        if prior_prints.size:
            prior = int(prior_prints[-1])
            # Keep the successor reference in a price-only coordinate.  The
            # cash entitlement is represented separately in shareholder
            # wealth and the claim ledger; subtracting it here would silently
            # reinvest the cash and erase the corresponding price return.
            successor_equivalent = linked_close[prior, predecessor] / q
            if not np.isfinite(successor_equivalent) or successor_equivalent <= 0.0:
                raise ValueError(
                    "verified conversion implies a non-positive prior reference price"
                )
            linked_close[prior, successor] = successor_equivalent

    return _DecisionContinuationInputs(
        close_brl=linked_close,
        volume_brl=linked_volume,
        trades=linked_trades,
        observed=linked_observed,
        trade_observed=linked_trade_observed,
        activity_valid=linked_activity_valid,
        ambiguous_action=linked_ambiguous,
        claim_owner=claim_owner,
    )


def _directory_size_bytes(root: Path) -> int:
    source = root.resolve(strict=True)
    if not source.is_dir():
        raise NotADirectoryError(source)
    return sum(path.stat().st_size for path in source.rglob("*") if path.is_file())


def _is_windows_system_drive(path: Path) -> bool:
    raw = str(path)
    if PureWindowsPath(raw).drive.casefold() == "c:":
        return True
    return re.search(r"(?i)(?:^|[\\/])c:[\\/]", raw) is not None


def _build_resource_preflight(
    *, output_dir: Path, previous_store: Path
) -> dict[str, object]:
    status = available_memory_status_bytes()
    available = int(status["available_build_memory_bytes"])
    output = output_dir.resolve()
    output_parent = output.parent
    if not output_parent.is_dir():
        raise FileNotFoundError(
            f"store output parent must exist before preflight: {output_parent}"
        )
    previous = previous_store.resolve(strict=True)
    previous_size = _directory_size_bytes(previous)
    if previous_size <= 0:
        raise ValueError("previous immutable store is empty")
    output_free = int(shutil.disk_usage(output_parent).free)
    required_output_free = 3 * previous_size
    process_temp = Path(tempfile.gettempdir()).resolve()
    test_scratch_raw = os.environ.get("BRAZIL_RV_TEST_SCRATCH")
    staging_roots = {
        "process_temp_workspace": str(process_temp),
        "array_workspace_parent": str(output_parent),
        "atomic_store_staging_parent": str(output_parent),
    }
    if test_scratch_raw:
        staging_roots["test_scratch_workspace"] = str(Path(test_scratch_raw).resolve())
    system_drive_roots = [
        name
        for name, path in staging_roots.items()
        if _is_windows_system_drive(Path(path))
    ]
    violations: list[str] = []
    if available < MINIMUM_BUILD_FREE_MEMORY_BYTES:
        violations.append("available_build_memory_below_10_gib")
    if output_free < required_output_free:
        violations.append("output_drive_free_below_three_times_previous_store")
    if system_drive_roots:
        violations.append("workspace_or_staging_root_on_windows_system_drive")
    preflight: dict[str, object] = {
        **status,
        "minimum_required_bytes": MINIMUM_BUILD_FREE_MEMORY_BYTES,
        "output_directory": str(output),
        "output_drive_free_bytes": output_free,
        "previous_store": str(previous),
        "previous_store_size_bytes": previous_size,
        "minimum_output_drive_free_bytes": required_output_free,
        "minimum_output_drive_free_multiple": 3,
        "resolved_temporary_workspace": str(process_temp),
        "staging_roots": staging_roots,
        "windows_system_drive_staging_roots": system_drive_roots,
        "violations": violations,
        "passed": not violations,
    }
    return preflight


def _require_build_resource_preflight(
    preflight: Mapping[str, object], *, allow_low_memory: bool = False
) -> None:
    violations = list(preflight["violations"])
    effective_violations = [
        violation
        for violation in violations
        if not (allow_low_memory and violation == "available_build_memory_below_10_gib")
    ]
    if not effective_violations:
        return

    physical = int(preflight["available_physical_memory_bytes"])
    details = f"physical {physical / 1024**3:.2f} GiB"
    available_commit = preflight["available_commit_memory_bytes"]
    commit_limit = preflight["commit_limit_bytes"]
    if available_commit is not None and commit_limit is not None:
        details += (
            f", available commit {int(available_commit) / 1024**3:.2f} GiB"
            f", commit limit {int(commit_limit) / 1024**3:.2f} GiB"
        )
    message = (
        "v2 store build refused before source loading: "
        f"violations={effective_violations}; conservative available build memory "
        f"is {int(preflight['available_build_memory_bytes']) / 1024**3:.2f} GiB "
        f"({details}); output free is "
        f"{int(preflight['output_drive_free_bytes']) / 1024**3:.2f} GiB; required "
        f"output free is {int(preflight['minimum_output_drive_free_bytes']) / 1024**3:.2f} GiB"
    )
    if "available_build_memory_below_10_gib" in effective_violations:
        raise MemoryError(message)
    raise OSError(message)


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


@dataclass(frozen=True)
class StreamedIntraday:
    result: IntradayDailyResult
    audit: pl.DataFrame
    source_paths: tuple[Path, ...]
    native_arrays: Mapping[str, NDArray[np.generic]]
    native_mapping: pl.DataFrame
    to_close_entry: NDArray[np.float32]
    to_close_entry_valid: NDArray[np.bool_]


@dataclass(frozen=True)
class MinutePanel:
    """M1 price/activity arrays with independent source-session support."""

    dates: NDArray[np.datetime64]
    isins: tuple[str, ...]
    open_brl: NDArray[np.floating]
    high_brl: NDArray[np.floating]
    low_brl: NDArray[np.floating]
    close_brl: NDArray[np.floating]
    volume: NDArray[np.floating]
    observed: NDArray[np.bool_]
    volume_valid: NDArray[np.bool_]
    session_valid: NDArray[np.bool_]

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
        if self.observed.dtype != np.bool_:
            raise TypeError("minute observed mask must be boolean")
        if self.volume_valid.shape != shape or self.volume_valid.dtype != np.bool_:
            raise ValueError("minute volume_valid must be boolean and aligned")
        if (
            self.session_valid.shape != shape[:2]
            or self.session_valid.dtype != np.bool_
        ):
            raise ValueError("minute session_valid must be boolean and aligned")
        if np.any(self.observed & ~self.session_valid[..., None]):
            raise ValueError("minute prices cannot be observed outside source support")
        if np.any(self.volume_valid & ~self.session_valid[..., None]):
            raise ValueError("minute activity cannot be valid outside source support")
        activity = np.asarray(self.volume)
        if np.any(self.volume_valid & (~np.isfinite(activity) | (activity < 0.0))):
            raise ValueError("valid minute activity must be finite and non-negative")
        if len(set(self.isins)) != len(self.isins):
            raise ValueError("minute panel ISIN axis must be unique")


def stream_intraday_from_assignments(
    assignments: pl.DataFrame,
    daily: pl.DataFrame,
    sessions: Sequence[SessionDefinition],
    isins: Sequence[str],
    *,
    sigma_asof: NDArray[np.floating],
    kept_rows: NDArray[np.integer],
    workspace: Path,
    official_log_return: NDArray[np.floating] | None = None,
    completed_action_boundary: NDArray[np.bool_] | None = None,
    same_day_boundary: NDArray[np.bool_] | None = None,
) -> StreamedIntraday:
    """Build sparse native M1 and broad daily summaries source-by-source.

    A physical XP file is loaded once, but every accepted identity segment is
    filtered to its exact COTAHIST ISIN dates before date-specific gridding.
    Native patches stay on the compact accepted-M1 identity axis.  Daily scalar
    summaries alone are scattered to the broad store identity axis.
    """

    from brazil_rv.modeling.contract import workspace_path
    from brazil_rv.preprocessing.io import (
        dense_grid,
        SOURCE_COLUMNS,
        validate_session_bars,
        validate_physical_source_identity,
    )

    required = {"security_id", "isin", "source_file"}
    if not required.issubset(assignments.columns):
        raise ValueError(
            f"streaming assignments columns missing: {sorted(required - set(assignments.columns))}"
        )
    if (
        assignments.is_empty()
        or assignments.get_column("security_id").null_count()
        or assignments.get_column("isin").null_count()
        or assignments.get_column("source_file").null_count()
        or assignments.get_column("security_id").n_unique() != assignments.height
        or assignments.get_column("isin").n_unique() != assignments.height
    ):
        raise ValueError("M1 assignments must bind unique non-null security/ISIN rows")
    if "manual_decision" in assignments.columns and set(
        assignments.get_column("manual_decision").cast(pl.String).to_list()
    ) != {"ACCEPTED"}:
        raise ValueError("every M1 assignment must be explicitly accepted")
    if "normalization_rule" in assignments.columns and set(
        assignments.get_column("normalization_rule").cast(pl.String).to_list()
    ) != {"FILTER_TO_COTAHIST_SECURITY_DATES"}:
        raise ValueError("M1 assignments have an unsupported identity filter")

    calendar = tuple(row.trade_date for row in sessions)
    if not calendar:
        raise ValueError("M1 streaming requires a nonempty session schedule")
    sigma = np.asarray(sigma_asof)
    if sigma.shape != (len(calendar), len(isins)):
        raise ValueError("M1 sigma_asof is misaligned with the daily store axes")
    selected = np.asarray(kept_rows)
    if (
        selected.ndim != 1
        or not np.issubdtype(selected.dtype, np.integer)
        or np.any(selected < 0)
        or np.any(selected >= len(calendar))
        or (selected.size > 1 and np.any(np.diff(selected) <= 0))
    ):
        raise ValueError("M1 kept_rows must be increasing schedule positions")
    selected = selected.astype(np.int64, copy=False)
    date_lookup = {value: index for index, value in enumerate(calendar)}
    isin_lookup = {value: index for index, value in enumerate(isins)}
    shape = (len(calendar), len(isins))
    feature_values = _workspace_array(
        workspace,
        "full_intraday_values",
        (*shape, len(INTRADAY_DAILY_FEATURES)),
        np.float32,
        fill=0.0,
    )
    feature_valid = _workspace_array(
        workspace,
        "full_intraday_valid",
        feature_values.shape,
        np.bool_,
        fill=False,
    )
    support = _workspace_array(
        workspace, "full_intraday_support", feature_values.shape, np.float32, fill=0.0
    )
    source_age = _workspace_array(
        workspace,
        "full_intraday_source_age",
        feature_values.shape,
        np.float32,
        fill=-1.0,
    )
    return_consistent = _workspace_array(
        workspace, "full_m1_return_consistent", shape, np.bool_, fill=False
    )
    entry = _workspace_array(
        workspace, "full_intraday_decision_mark", shape, np.float32, fill=np.nan
    )
    entry_valid = _workspace_array(
        workspace, "full_intraday_entry_valid", shape, np.bool_, fill=False
    )
    realized = _workspace_array(
        workspace, "full_intraday_realized", shape, np.float32, fill=np.nan
    )
    present = _workspace_array(
        workspace, "full_fast_present", shape, np.bool_, fill=False
    )
    session_close = _workspace_array(
        workspace, "full_m1_session_close", shape, np.float32, fill=np.nan
    )
    session_close_valid = _workspace_array(
        workspace, "full_m1_session_close_valid", shape, np.bool_, fill=False
    )
    to_close_entry = _workspace_array(
        workspace, "full_to_close_entry", shape, np.float32, fill=np.nan
    )
    to_close_entry_valid = _workspace_array(
        workspace, "full_to_close_entry_valid", shape, np.bool_, fill=False
    )

    ordered_assignments = assignments.sort("security_id")
    fast_mapping_rows: list[dict[str, object]] = []
    for fast_index, row in enumerate(ordered_assignments.iter_rows(named=True)):
        isin = str(row["isin"])
        store_index = isin_lookup.get(isin)
        if store_index is None:
            raise ValueError(f"accepted M1 ISIN absent from daily axis: {isin}")
        fast_mapping_rows.append(
            {
                "fast_index": fast_index,
                "store_name_index": store_index,
                "isin": isin,
                "security_id": str(row["security_id"]),
            }
        )
    native_mapping = pl.DataFrame(fast_mapping_rows)
    fast_by_isin = {
        str(row["isin"]): int(row["fast_index"]) for row in fast_mapping_rows
    }
    patch_count = max(
        (
            (row.decision_time.hour * 60 + row.decision_time.minute)
            - (row.continuous_open.hour * 60 + row.continuous_open.minute)
        )
        // 5
        for row in sessions
    )
    native_shape = (selected.size, ordered_assignments.height, patch_count)
    native_arrays: dict[str, NDArray[np.generic]] = {
        "fast_patch_values": _workspace_array(
            workspace,
            "store_fast_patch_values",
            (*native_shape, len(NATIVE_FAST_FEATURES)),
            np.float32,
            fill=0.0,
        ),
        "fast_patch_valid": _workspace_array(
            workspace,
            "store_fast_patch_valid",
            (*native_shape, len(NATIVE_FAST_FEATURES)),
            np.bool_,
            fill=False,
        ),
        "fast_patch_mask": _workspace_array(
            workspace,
            "store_fast_patch_mask",
            native_shape,
            np.bool_,
            fill=False,
        ),
        "fast_last_price_age_minutes": _workspace_array(
            workspace,
            "store_fast_last_price_age_minutes",
            native_shape,
            np.float32,
            fill=0.0,
        ),
        "fast_last_price_age_valid": _workspace_array(
            workspace,
            "store_fast_last_price_age_valid",
            native_shape,
            np.bool_,
            fill=False,
        ),
    }
    store_row_by_global = np.full(len(calendar), -1, dtype=np.int64)
    store_row_by_global[selected] = np.arange(selected.size, dtype=np.int64)
    audit_rows: list[dict[str, object]] = []
    source_paths: list[Path] = []
    dates_by_isin = {
        key[0] if isinstance(key, tuple) else key: frozenset(
            group.get_column("trade_date").to_list()
        )
        for key, group in daily.select("isin", "trade_date").group_by("isin")
    }
    for group in ordered_assignments.partition_by("source_file"):
        raw_path = Path(str(group[0, "source_file"]))
        source_path = raw_path if raw_path.is_file() else workspace_path(raw_path)
        source_path = source_path.resolve()
        source_paths.append(source_path)
        source = (
            pl.scan_parquet(source_path)
            .filter(
                pl.col("ts_exchange").dt.date().is_between(calendar[0], calendar[-1])
            )
            .select(list(SOURCE_COLUMNS))
            .collect()
        )
        source_sha256 = source_records([source_path])[0]["sha256"]
        if not source.is_empty() and "xp_symbol" in group.columns:
            validate_physical_source_identity(group, source, source_path)
        claimed_dates: set[date] = set()
        for row in group.iter_rows(named=True):
            isin = str(row["isin"])
            target = isin_lookup.get(isin)
            assert target is not None
            fast_index = fast_by_isin[isin]
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
            overlap = claimed_dates.intersection(allowed)
            if overlap:
                raise ValueError(
                    "one physical M1 source assigns the same session to multiple "
                    f"identities: {source_path}, {min(overlap)}"
                )
            claimed_dates.update(allowed)
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
            stop = date_lookup[max(allowed)] + 1
            local_calendar = calendar[start:stop]
            local_sessions = tuple(sessions[start:stop])
            schedule_index = pl.DataFrame(
                {
                    "trade_date": pl.Series(local_calendar, dtype=pl.Date),
                    "date_idx": pl.Series(range(len(local_calendar)), dtype=pl.Int32),
                    "continuous_open_minute": pl.Series(
                        [
                            value.continuous_open.hour * 60
                            + value.continuous_open.minute
                            for value in local_sessions
                        ],
                        dtype=pl.Int16,
                    ),
                    "continuous_minute_count": pl.Series(
                        [
                            value.continuous_close.hour * 60
                            + value.continuous_close.minute
                            - value.continuous_open.hour * 60
                            - value.continuous_open.minute
                            for value in local_sessions
                        ],
                        dtype=pl.Int16,
                    ),
                }
            )
            bars = (
                source.with_columns(
                    pl.col("ts_exchange").dt.date().alias("trade_date"),
                    (
                        pl.col("ts_exchange").dt.hour().cast(pl.Int16) * 60
                        + pl.col("ts_exchange").dt.minute().cast(pl.Int16)
                    ).alias("clock_minute"),
                )
                .filter(pl.col("trade_date").is_in(tuple(allowed)))
                .join(schedule_index, on="trade_date", how="inner")
                .with_columns(
                    (pl.col("clock_minute") - pl.col("continuous_open_minute"))
                    .cast(pl.Int16)
                    .alias("minute_idx")
                )
                .filter(
                    pl.col("minute_idx") >= 0,
                    pl.col("minute_idx") < pl.col("continuous_minute_count"),
                )
                .sort("ts_exchange")
            )
            validate_session_bars(bars, source_path)
            max_minutes = max(
                value.continuous_close.hour * 60
                + value.continuous_close.minute
                - value.continuous_open.hour * 60
                - value.continuous_open.minute
                for value in local_sessions
            )
            grid, observed = dense_grid(bars, len(local_calendar), max_minutes)
            session_valid = (
                np.asarray(
                    [value in allowed for value in local_calendar], dtype=np.bool_
                )
                & observed.any(axis=1)
            )[:, None]
            # The accepted sparse MT5 archive does not certify that an absent
            # minute is a zero-trade minute.  Only physical rows therefore
            # carry valid activity; gaps remain unknown even on a supported
            # source session.
            volume_valid = observed[:, None, :].copy()
            native = build_intraday_daily_features(
                grid[:, None, :, 0],
                grid[:, None, :, 1],
                grid[:, None, :, 2],
                grid[:, None, :, 3],
                grid[:, None, :, 4],
                observed[:, None, :],
                volume_valid=volume_valid,
                session_valid=session_valid,
                sessions=local_sessions,
                official_log_return=(
                    None
                    if official_log_return is None
                    else official_log_return[start:stop, target, None]
                ),
                completed_action_boundary=(
                    None
                    if completed_action_boundary is None
                    else completed_action_boundary[start:stop, target, None]
                ),
                same_day_boundary=(
                    None
                    if same_day_boundary is None
                    else same_day_boundary[start:stop, target, None]
                ),
            )
            prefix_indices = np.asarray(
                [
                    value.decision_time.hour * 60
                    + value.decision_time.minute
                    - value.continuous_open.hour * 60
                    - value.continuous_open.minute
                    for value in local_sessions
                ],
                dtype=np.int64,
            )
            local_entry = grid[np.arange(len(local_calendar)), prefix_indices, 0]
            local_entry_valid = (
                observed[np.arange(len(local_calendar)), prefix_indices]
                & np.isfinite(local_entry)
                & (local_entry > 0.0)
                & session_valid[:, 0]
            )
            to_close_entry[start:stop, target] = np.where(
                local_entry_valid, local_entry, np.nan
            ).astype(np.float32)
            to_close_entry_valid[start:stop, target] = local_entry_valid
            feature_values[start:stop, target] = native.values[:, 0]
            feature_valid[start:stop, target] = native.valid[:, 0]
            support[start:stop, target] = native.support_fraction[:, 0]
            source_age[start:stop, target] = native.source_age_sessions[:, 0]
            return_consistent[start:stop, target] = native.return_consistent[:, 0]
            entry[start:stop, target] = native.decision_mark[:, 0]
            entry_valid[start:stop, target] = native.decision_mark_valid[:, 0]
            realized[start:stop, target] = native.realized_daily_vol[:, 0]
            present[start:stop, target] = native.fast_present[:, 0]
            session_close[start:stop, target] = native.session_close[:, 0]
            session_close_valid[start:stop, target] = native.session_close_valid[:, 0]
            local_native_shape = (len(local_calendar), 1, patch_count)
            local_native_values = np.empty(
                (*local_native_shape, len(NATIVE_FAST_FEATURES)), dtype=np.float32
            )
            local_native_valid = np.empty_like(local_native_values, dtype=np.bool_)
            local_native_patch_mask = np.empty(local_native_shape, dtype=np.bool_)
            local_native_age = np.empty(local_native_shape, dtype=np.float32)
            local_native_age_valid = np.empty(local_native_shape, dtype=np.bool_)
            build_native_fast_features_into(
                grid[:, None, :, 1],
                grid[:, None, :, 2],
                grid[:, None, :, 3],
                grid[:, None, :, 4],
                observed[:, None, :],
                volume_valid=volume_valid,
                session_valid=session_valid,
                sigma_asof=sigma[start:stop, target, None],
                sessions=local_sessions,
                values_out=local_native_values,
                valid_out=local_native_valid,
                patch_mask_out=local_native_patch_mask,
                last_price_age_minutes_out=local_native_age,
                last_price_age_valid_out=local_native_age_valid,
            )
            local_store_rows = store_row_by_global[start:stop]
            retained_local = np.flatnonzero(local_store_rows >= 0)
            retained_store = local_store_rows[retained_local]
            native_arrays["fast_patch_values"][retained_store, fast_index] = (
                local_native_values[retained_local, 0]
            )
            native_arrays["fast_patch_valid"][retained_store, fast_index] = (
                local_native_valid[retained_local, 0]
            )
            native_arrays["fast_patch_mask"][retained_store, fast_index] = (
                local_native_patch_mask[retained_local, 0]
            )
            native_arrays["fast_last_price_age_minutes"][retained_store, fast_index] = (
                local_native_age[retained_local, 0]
            )
            native_arrays["fast_last_price_age_valid"][retained_store, fast_index] = (
                local_native_age_valid[retained_local, 0]
            )
            has_bar = observed.any(axis=1)
            exact_close = np.asarray(
                [
                    observed[day_index, minutes - 1]
                    for day_index, minutes in enumerate(
                        schedule_index.get_column("continuous_minute_count").to_list()
                    )
                ],
                dtype=np.bool_,
            )
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
            del (
                bars,
                grid,
                observed,
                native,
                volume_valid,
                local_native_values,
                local_native_valid,
                local_native_patch_mask,
                local_native_age,
                local_native_age_valid,
            )
        del source
    return StreamedIntraday(
        result=IntradayDailyResult(
            values=feature_values,
            valid=feature_valid,
            decision_mark=entry,
            decision_mark_valid=entry_valid,
            session_close=session_close,
            session_close_valid=session_close_valid,
            realized_daily_vol=realized,
            fast_present=present,
            return_consistent=return_consistent,
            support_fraction=support,
            source_age_sessions=source_age,
        ),
        audit=pl.DataFrame(audit_rows),
        source_paths=tuple(sorted(set(source_paths))),
        native_arrays=native_arrays,
        native_mapping=native_mapping,
        to_close_entry=to_close_entry,
        to_close_entry_valid=to_close_entry_valid,
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


def _common_state_diagnostic_panel(
    wealth_close: NDArray[np.floating],
    wealth_valid: NDArray[np.bool_],
    unresolved_action: NDArray[np.bool_],
    active: NDArray[np.bool_],
    sigma_asof: NDArray[np.floating],
    decision_rows: NDArray[np.integer],
    dates: NDArray[np.datetime64],
    *,
    minimum_names: int,
) -> tuple[NDArray[np.float32], NDArray[np.bool_], pl.DataFrame]:
    """Build audit-only common market state on the canonical decision clock.

    For decision row ``t``, the return statistics use the exact shareholder-
    wealth move from raw daily rows ``t-2`` to ``t-1``.  The risk statistic
    uses the already-lagged raw ``sigma_asof[t]``.  No value is exposed as a
    model feature in this refactor; the panel documents information removed by
    cross-sectional ranks and provides a clean input for a later registered
    two-field context comparison.
    """

    close = np.asarray(wealth_close, dtype=np.float64)
    valid = np.asarray(wealth_valid, dtype=np.bool_)
    unresolved = np.asarray(unresolved_action, dtype=np.bool_)
    membership = np.asarray(active, dtype=np.bool_)
    sigma = np.asarray(sigma_asof, dtype=np.float64)
    rows = np.asarray(decision_rows, dtype=np.int64)
    if any(
        value.shape != close.shape for value in (valid, unresolved, membership, sigma)
    ):
        raise ValueError("common-state inputs must align [date, name]")
    if minimum_names < 1:
        raise ValueError("common-state minimum_names must be positive")

    values = np.zeros((rows.size, len(COMMON_STATE_DIAGNOSTICS)), dtype=np.float32)
    masks = np.zeros_like(values, dtype=np.bool_)
    return_support = np.zeros(rows.size, dtype=np.int32)
    volatility_support = np.zeros(rows.size, dtype=np.int32)
    for output_row, decision_row in enumerate(rows):
        source_row = int(decision_row) - 1
        prior_row = source_row - 1
        if prior_row >= 0:
            usable = (
                membership[decision_row]
                & valid[source_row]
                & valid[prior_row]
                & ~unresolved[source_row]
                & np.isfinite(close[source_row])
                & np.isfinite(close[prior_row])
                & (close[source_row] > 0.0)
                & (close[prior_row] > 0.0)
            )
            return_support[output_row] = int(usable.sum())
            if return_support[output_row] >= minimum_names:
                returns = np.log(close[source_row, usable] / close[prior_row, usable])
                values[output_row, 0] = np.float32(np.median(returns))
                values[output_row, 2] = np.float32(np.std(returns, ddof=0))
                masks[output_row, (0, 2)] = True

        usable_sigma = (
            membership[decision_row]
            & np.isfinite(sigma[decision_row])
            & (sigma[decision_row] > 1e-8)
        )
        volatility_support[output_row] = int(usable_sigma.sum())
        if volatility_support[output_row] >= minimum_names:
            values[output_row, 1] = np.float32(
                np.median(sigma[decision_row, usable_sigma])
            )
            masks[output_row, 1] = True

    table_rows = []
    for index, decision_row in enumerate(rows):
        row: dict[str, object] = {
            "trade_date": np.asarray(dates)[decision_row].astype(object),
            "return_support_names": int(return_support[index]),
            "volatility_support_names": int(volatility_support[index]),
        }
        for feature_index, name in enumerate(COMMON_STATE_DIAGNOSTICS):
            row[name] = (
                float(values[index, feature_index])
                if masks[index, feature_index]
                else None
            )
            row[f"{name}_valid"] = bool(masks[index, feature_index])
        table_rows.append(row)
    return values, masks, pl.DataFrame(table_rows)


def _target_validity_tables(
    dates: NDArray[np.datetime64],
    valid: NDArray[np.bool_],
    active: NDArray[np.bool_],
    observed: NDArray[np.bool_],
    survival_identities: Sequence[str] | None = None,
    *,
    family: str,
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
                    "family": family,
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
                    "family": family,
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


def _cotahist_action_counts_by_year(
    dates: NDArray[np.datetime64],
    distribution_changed: NDArray[np.bool_],
    detected: DetectedActionResult,
) -> pl.DataFrame:
    change = np.asarray(distribution_changed, dtype=np.bool_)
    shape = (len(dates), detected.split_event.shape[1])
    arrays = (
        change,
        detected.split_event,
        detected.cash_event,
        detected.ambiguous_event,
        detected.price_jump_anomaly_mask,
    )
    if any(np.asarray(value).shape != shape for value in arrays):
        raise ValueError("COTAHIST action-count axes are misaligned")
    years = dates.astype("datetime64[Y]").astype(np.int64) + 1970
    rows: list[dict[str, object]] = []
    for year in sorted(set(years.tolist())):
        selected = years == year
        rows.append(
            {
                "year": int(year),
                "distribution_change_count": int(change[selected].sum()),
                "split_count": int(detected.split_event[selected].sum()),
                "cash_count": int(detected.cash_event[selected].sum()),
                "ambiguous_count": int(detected.ambiguous_event[selected].sum()),
                "jump_only_anomaly_count": int(
                    detected.price_jump_anomaly_mask[selected].sum()
                ),
            }
        )
    return pl.DataFrame(rows)


def _inferred_action_counts_by_year(
    dates: NDArray[np.datetime64], result: InferredActionResult | None
) -> pl.DataFrame:
    schema = {
        "year": pl.Int32,
        "u1_count": pl.Int64,
        "c1_count": pl.Int64,
        "u2_count": pl.Int64,
        "u2_per_trade_count": pl.Int64,
        "u2_total_quantity_count": pl.Int64,
        "insufficient_market_support_session_count": pl.Int64,
    }
    if result is None:
        return pl.DataFrame(schema=schema)
    years = dates.astype("datetime64[Y]").astype(np.int64) + 1970
    rows = []
    for year in sorted(set(years.tolist())):
        selected = years == year
        rows.append(
            {
                "year": int(year),
                "u1_count": int(result.u1_event[selected].sum()),
                "c1_count": int(result.c1_event[selected].sum()),
                "u2_count": int(result.u2_event[selected].sum()),
                "u2_per_trade_count": int(result.u2_per_trade_event[selected].sum()),
                "u2_total_quantity_count": int(
                    result.u2_total_quantity_event[selected].sum()
                ),
                "insufficient_market_support_session_count": int(
                    result.insufficient_market_support[selected].sum()
                ),
            }
        )
    return pl.DataFrame(rows, schema=schema)


def _large_move_no_action_by_year(
    dates: NDArray[np.datetime64], result: InferredActionResult | None
) -> pl.DataFrame:
    schema = {"year": pl.Int32, "large_move_no_action_count": pl.Int64}
    if result is None:
        return pl.DataFrame(schema=schema)
    years = dates.astype("datetime64[Y]").astype(np.int64) + 1970
    return pl.DataFrame(
        [
            {
                "year": int(year),
                "large_move_no_action_count": int(
                    result.large_move_no_action[years == year].sum()
                ),
            }
            for year in sorted(set(years.tolist()))
        ],
        schema=schema,
    )


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
            identity_last[identity] = max(
                identity_last.get(identity, -1), int(rows[-1])
            )
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
    volume_brl: NDArray[np.floating], activity_valid: NDArray[np.bool_]
) -> NDArray[np.float32]:
    """Return exact causal ADV20, invalidated by unknown source activity."""

    volume = np.asarray(volume_brl, dtype=np.float64)
    valid = np.asarray(activity_valid, dtype=np.bool_)
    if volume.ndim != 2 or valid.shape != volume.shape:
        raise ValueError("prior-ADV20 axes are misaligned")
    if np.any(valid & (~np.isfinite(volume) | (volume < 0.0))):
        raise ValueError("prior-ADV20 valid activity must be finite and non-negative")
    output = np.full(volume.shape, np.nan, dtype=np.float32)
    clean = np.where(valid, volume, 0.0)
    cumulative = np.vstack(
        (np.zeros((1, volume.shape[1]), dtype=np.float64), np.cumsum(clean, axis=0))
    )
    counts = np.vstack(
        (
            np.zeros((1, volume.shape[1]), dtype=np.int32),
            np.cumsum(valid, axis=0, dtype=np.int32),
        )
    )
    for day in range(20, volume.shape[0]):
        exact = counts[day] - counts[day - 20] == 20
        output[day, exact] = (
            (cumulative[day, exact] - cumulative[day - 20, exact]) / 20.0
        ).astype(np.float32)
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
        raise ValueError(
            f"external feature family {family} has no observable population"
        )
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
        (quartile_index + 1, quartile == quartile_index) for quartile_index in range(4)
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
            gap = ratios["survives_to_final_year"] - ratios["delisted_within_panel"]
            direction = (
                "survivor_above_delisted" if gap > 0.0 else "delisted_above_survivor"
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
    return_consistent = np.zeros(output_shape, dtype=np.bool_)
    support = np.zeros(values.shape, dtype=np.float32)
    source_age = np.full(values.shape, -1.0, dtype=np.float32)
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
            entry[target_date, target_isin] = result.decision_mark[
                source_date, source_isin
            ]
            entry_valid[target_date, target_isin] = result.decision_mark_valid[
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
            return_consistent[target_date, target_isin] = result.return_consistent[
                source_date, source_isin
            ]
            support[target_date, target_isin] = result.support_fraction[
                source_date, source_isin
            ]
            source_age[target_date, target_isin] = result.source_age_sessions[
                source_date, source_isin
            ]
    return IntradayDailyResult(
        values=values,
        valid=valid,
        decision_mark=entry,
        decision_mark_valid=entry_valid,
        session_close=session_close,
        session_close_valid=session_close_valid,
        realized_daily_vol=realized,
        fast_present=present,
        return_consistent=return_consistent,
        support_fraction=support,
        source_age_sessions=source_age,
    )


def build_daily_store(
    daily: pl.DataFrame,
    actions: pl.DataFrame,
    output_dir: Path,
    *,
    minute_panel: MinutePanel | None = None,
    sidecars: Mapping[str, SidecarResult] | None = None,
    stream_intraday: bool = False,
    sidecar_arguments: Sequence[str] = (),
    action_acquisition_audit: pl.DataFrame | None = None,
    m1_assignments: pl.DataFrame | None = None,
    source_paths: Sequence[Path] = (),
    implementation_commit: str | None = None,
    cotahist_raw_sources: Sequence[Path] = (),
    cotahist_parse_audit: Path | None = None,
    session_schedule: Sequence[SessionDefinition] | None = None,
    following_decision_at: datetime | None = None,
    isin_link_allowlist: Path | None = None,
    minimum_rank_names: int = 20,
    store_start: date | None = STORE_START,
    security_axis: Sequence[str] | None = None,
    resource_preflight: Mapping[str, object] | None = None,
    action_terms_source: str = "verified_contractual_terms",
    schedule_source: str | None = None,
    schedule_reconstruction_audit: Mapping[str, object] | None = None,
) -> Path:
    """Build the immutable aligned daily store from already-acquired sources."""

    if (
        sum(
            value is not None and value is not False
            for value in (minute_panel, stream_intraday)
        )
        > 1
    ):
        raise ValueError("provide exactly one intraday source mode")
    if sidecars is not None and sidecar_arguments:
        raise ValueError("provide materialized or streamed sidecars, not both")
    if minimum_rank_names < 1:
        raise ValueError("minimum_rank_names must be positive")
    if action_terms_source not in {
        "verified_contractual_terms",
        "inferred_cotahist_dismes_v1",
    }:
        raise ValueError("unsupported corporate-action source tier")
    has_intraday = minute_panel is not None or stream_intraday

    m1_isins: tuple[str, ...] = ()
    if m1_assignments is not None:
        m1_isins = tuple(m1_assignments.get_column("isin").cast(pl.String).to_list())
    validation = prepare_cash_equities(
        daily,
        v1_isins=m1_isins,
        require_units=True,
        maximum_rejection_fraction=0.005,
    )
    cash = validation.accepted
    if session_schedule is None:
        raise ValueError("an authoritative B3 session schedule is required")
    archive_dates = list(validation.source_session_dates)
    assert_calendar_complete(session_schedule, archive_dates)
    calendar = tuple(row.trade_date for row in session_schedule)
    if not calendar:
        raise ValueError("B3 session schedule is empty")
    archive_date_set = frozenset(archive_dates)
    source_session_complete = np.asarray(
        [value in archive_date_set for value in calendar], dtype=np.bool_
    )
    panel = panel_from_daily(
        cash,
        dates=calendar,
        isins=security_axis,
        source_session_complete=source_session_complete,
        invalid_observations=validation.rejected,
    )
    proposed_isin_successions = detect_isin_successions(cash)
    if isin_link_allowlist is None:
        isin_successions = pl.DataFrame(
            schema={
                "predecessor_isin": pl.String,
                "successor_isin": pl.String,
            }
        )
    else:
        isin_successions = load_isin_link_allowlist(
            isin_link_allowlist, proposed_isin_successions
        )
    continuation_isins = continuation_identity_axis(panel.isins, isin_successions)
    decision_timestamps = tuple(row.decision_at for row in session_schedule)
    daily_action_cutoffs = next_session_decision_cutoffs(
        session_schedule, following_decision_at=following_decision_at
    )
    resolved_schedule_source = schedule_source or schedule_source_label(
        session_schedule
    )
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
    checked_actions = validate_action_table(actions)
    if action_acquisition_audit is None:
        raise ValueError(
            "economic store construction requires an action acquisition audit; "
            "an empty action table does not prove zero actions"
        )
    provider_split, provider_cash_distribution, _ = align_action_arrays(
        checked_actions, panel.dates, panel.isins
    )
    conversion_verified_action_terms = verified_conversion_terms_from_links(
        isin_successions
    )
    inferred_actions: InferredActionResult | None = None
    if action_terms_source == "inferred_cotahist_dismes_v1":
        inference_universe = build_daily_universe(
            panel.close_brl,
            panel.volume_brl,
            panel.observed,
            trade_observed=panel.trade_observed,
            activity_valid=panel.activity_valid,
            source_session_complete=panel.source_session_complete,
        )
        inferred_actions = infer_cotahist_action_terms(
            panel.dates,
            panel.isins,
            panel.close_brl,
            panel.quantity,
            panel.trades,
            panel.distribution_number,
            panel.observed,
            inference_universe.active,
        )
        provider_verified_action_terms: tuple[VerifiedActionTerm, ...] = ()
        source_action_terms = inferred_actions.terms
        action_coverage_resolved = inferred_actions.coverage_resolved
    else:
        provider_verified_action_terms = provider_actions_to_verified_terms(
            checked_actions
        )
        source_action_terms = provider_verified_action_terms
        action_coverage_resolved = action_coverage_resolved_mask(
            action_acquisition_audit, panel.dates, panel.isins
        )
    verified_action_terms = validate_verified_action_terms(
        (*source_action_terms, *conversion_verified_action_terms)
    )
    retrospective_actions = align_verified_action_terms(
        verified_action_terms,
        panel.dates,
        panel.isins,
        coverage_resolved=action_coverage_resolved,
    )
    decision_actions = align_decision_known_action_terms(
        verified_action_terms,
        panel.dates,
        panel.isins,
        coverage_resolved=action_coverage_resolved,
        decision_timestamps=daily_action_cutoffs,
    )
    action_payment_session = align_action_payment_sessions(
        verified_action_terms, panel.dates, panel.isins
    )
    first_kept_row = int(kept_rows[0])
    action_payment_session = np.where(
        action_payment_session < 0,
        -1,
        action_payment_session - first_kept_row,
    ).astype(np.int64)
    if (action_payment_session[kept_rows] < -1).any():
        raise ValueError("an in-store action payment predates the store date axis")
    distribution_changed = detect_distribution_changes(
        panel.distribution_number, panel.observed
    )
    detected_actions = detect_cotahist_actions(
        panel.close_brl,
        panel.quantity,
        panel.distribution_number,
        panel.observed,
    )
    # Price/quantity/DISMES classifications are retained for diagnostics only.
    # Accepted cross-session features are governed solely by contractual action
    # terms that were known at that historical decision.  An unresolved cell
    # must break the unit/history chain rather than silently assuming q=1,d=0.
    diagnostic_intraday_boundary_lagged = detected_actions.split_event
    diagnostic_intraday_boundary_sameday = detect_open_gap_boundaries(
        panel.open_brl, panel.close_brl, panel.observed
    )
    identity_successor = np.broadcast_to(
        np.arange(len(panel.isins), dtype=np.int64), panel.observed.shape
    )
    intraday_unit_or_unresolved_boundary = decision_action_boundaries(
        panel.open_brl,
        panel.close_brl,
        panel.observed,
        decision_actions.session_resolved,
        verified_action_terms,
        session_schedule,
        panel.isins,
    )
    # Completed reference returns use retrospective terms solely for historical
    # M1 validation. The scalar builder shifts their use past the closing time.
    intraday_official_return = shareholder_reference_returns(
        panel.close_brl, panel.observed, retrospective_actions
    )
    intraday_completed_boundary = (
        retrospective_actions.has_action
        | ~retrospective_actions.session_resolved
        | (retrospective_actions.successor_index != identity_successor)
    )
    intraday_same_day_boundary = intraday_unit_or_unresolved_boundary
    wealth_paths = {
        name: workspace / f"{name}.npy"
        for name in (
            "shareholder_wealth_open",
            "shareholder_wealth_high",
            "shareholder_wealth_low",
            "shareholder_wealth_close",
            "shareholder_wealth_valid",
        )
    }
    shareholder_wealth_open = _workspace_array(
        workspace, "shareholder_wealth_open", panel.close_brl.shape, np.float32
    )
    shareholder_wealth_high = _workspace_array(
        workspace, "shareholder_wealth_high", panel.close_brl.shape, np.float32
    )
    shareholder_wealth_low = _workspace_array(
        workspace, "shareholder_wealth_low", panel.close_brl.shape, np.float32
    )
    shareholder_wealth_close = _workspace_array(
        workspace, "shareholder_wealth_close", panel.close_brl.shape, np.float32
    )
    shareholder_wealth_valid = _workspace_array(
        workspace, "shareholder_wealth_valid", panel.close_brl.shape, np.bool_
    )
    build_shareholder_wealth_ohlc_into(
        panel.open_brl,
        panel.high_brl,
        panel.low_brl,
        panel.close_brl,
        panel.observed,
        decision_actions,
        wealth_open=shareholder_wealth_open,
        wealth_high=shareholder_wealth_high,
        wealth_low=shareholder_wealth_low,
        wealth_close=shareholder_wealth_close,
        wealth_valid=shareholder_wealth_valid,
    )
    decision_continuation = _route_decision_known_continuations(
        dates=panel.dates,
        isins=panel.isins,
        links=isin_successions,
        decision_timestamps=decision_timestamps,
        raw_close=panel.close_brl,
        volume_brl=panel.volume_brl,
        trades=panel.trades,
        observed=panel.observed,
        trade_observed=panel.trade_observed,
        activity_valid=panel.activity_valid,
        ambiguous_action=~decision_actions.session_resolved,
        shareholder_wealth_arrays=(
            shareholder_wealth_open,
            shareholder_wealth_high,
            shareholder_wealth_low,
            shareholder_wealth_close,
            shareholder_wealth_valid,
        ),
    )
    for materialized in (
        shareholder_wealth_open,
        shareholder_wealth_high,
        shareholder_wealth_low,
        shareholder_wealth_close,
        shareholder_wealth_valid,
    ):
        close_memmap(materialized)
    gc.collect()
    shareholder_wealth_open = np.load(
        wealth_paths["shareholder_wealth_open"], mmap_mode="r", allow_pickle=False
    )
    shareholder_wealth_high = np.load(
        wealth_paths["shareholder_wealth_high"], mmap_mode="r", allow_pickle=False
    )
    shareholder_wealth_low = np.load(
        wealth_paths["shareholder_wealth_low"], mmap_mode="r", allow_pickle=False
    )
    shareholder_wealth_close = np.load(
        wealth_paths["shareholder_wealth_close"], mmap_mode="r", allow_pickle=False
    )
    shareholder_wealth_valid = np.load(
        wealth_paths["shareholder_wealth_valid"], mmap_mode="r", allow_pickle=False
    )
    universe = build_daily_universe(
        decision_continuation.close_brl,
        decision_continuation.volume_brl,
        decision_continuation.observed,
        trade_observed=decision_continuation.trade_observed,
        activity_valid=decision_continuation.activity_valid,
        source_session_complete=panel.source_session_complete,
    )
    universe = replace(
        universe, active=universe.active & decision_continuation.claim_owner
    )
    # A recurrent calendar step exists from an identity's first causal history
    # row onward.  A decision on t consumes daily market state only through
    # t-1; accepted, timely conversion links carry that history without a
    # second consumer-side shift.
    slow_timestep_valid = np.zeros_like(panel.observed, dtype=np.bool_)
    if slow_timestep_valid.shape[0] > 1:
        slow_timestep_valid[1:] = np.maximum.accumulate(
            decision_continuation.observed[:-1], axis=0
        )
    linked_slow_inputs = (
        shareholder_wealth_open,
        shareholder_wealth_high,
        shareholder_wealth_low,
        shareholder_wealth_close,
        decision_continuation.volume_brl,
        decision_continuation.trades,
        shareholder_wealth_valid,
        decision_continuation.ambiguous_action,
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
    slow_specs = feature_specs(
        "slow", SLOW_FEATURES, minimum_rank_names=minimum_rank_names
    )
    slow_age_sessions = _workspace_array(
        workspace,
        "slow_age_sessions",
        slow_values.shape,
        np.float32,
    )
    slow_sigma = _workspace_array(
        workspace,
        "slow_sigma",
        panel.observed.shape,
        np.float32,
        fill=np.nan,
    )

    def consume_slow_feature(
        feature_index: int,
        raw_values: NDArray[np.floating],
        raw_valid: NDArray[np.bool_],
    ) -> None:
        values_3d = np.asarray(raw_values)[..., None]
        valid_3d = np.asarray(raw_valid, dtype=np.bool_)[..., None]
        transform_feature_panel_into(
            values_3d,
            valid_3d,
            universe.active,
            slow_specs[feature_index : feature_index + 1],
            slow_values[..., feature_index : feature_index + 1],
            slow_valid[..., feature_index : feature_index + 1],
            source_rows=kept_rows - 1,
            membership_rows=kept_rows,
            minimum_rank_names=minimum_rank_names,
        )
        observation_age_sessions_into(
            valid_3d,
            universe.active,
            slow_age_sessions[..., feature_index : feature_index + 1],
            source_rows=kept_rows - 1,
            decision_rows=kept_rows,
        )
        if feature_index == 8:
            slow_sigma[...] = np.where(raw_valid, raw_values, np.nan).astype(np.float32)

    build_slow_features_into(
        *linked_slow_inputs[:6],
        linked_slow_inputs[6],
        universe.active,
        panel.dates,
        raw_high=panel.high_brl,
        raw_low=panel.low_brl,
        raw_close=panel.close_brl,
        price_observed=panel.observed,
        history_observed=decision_continuation.observed,
        activity_valid=decision_continuation.activity_valid,
        consume=consume_slow_feature,
        ambiguous_action=linked_slow_inputs[7],
    )
    target_scale_sigma = _workspace_array(
        workspace,
        "target_scale_sigma",
        slow_sigma.shape,
        np.float32,
        fill=np.nan,
    )
    target_scale_sigma[1:] = slow_sigma[:-1]
    (
        common_state_values,
        common_state_valid,
        common_state_table,
    ) = _common_state_diagnostic_panel(
        shareholder_wealth_close,
        shareholder_wealth_valid,
        decision_continuation.ambiguous_action,
        universe.active,
        target_scale_sigma,
        kept_rows,
        panel.dates,
        minimum_names=minimum_rank_names,
    )
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
    intraday_age_sessions = _workspace_array(
        workspace,
        "intraday_age_sessions",
        intraday_values.shape,
        np.float32,
        fill=-1.0,
    )
    intraday_support_fraction = _workspace_array(
        workspace,
        "intraday_support_fraction",
        intraday_values.shape,
        np.float32,
        fill=0.0,
    )
    fast_sigma = np.full(shape, np.nan, dtype=np.float64)
    fast_present = np.zeros(shape, dtype=np.bool_)
    entry = np.full(shape, np.nan, dtype=np.float64)
    entry_valid = np.zeros(shape, dtype=np.bool_)
    realized_daily = np.full(shape, np.nan, dtype=np.float64)
    m1_session_close = np.full(shape, np.nan, dtype=np.float64)
    m1_session_close_valid = np.zeros(shape, dtype=np.bool_)
    return_consistent = np.zeros(shape, dtype=np.bool_)
    intraday_audit: pl.DataFrame | None = None
    intraday_source_paths: tuple[Path, ...] = ()
    native_fast_arrays: dict[str, NDArray[np.generic]] = {}
    native_fast_mapping: pl.DataFrame | None = None
    streamed_workspace_arrays: tuple[NDArray[np.generic], ...] = ()
    streamed_intraday: StreamedIntraday | None = None
    if stream_intraday:
        if m1_assignments is None:
            raise ValueError("streamed intraday construction requires M1 assignments")
        streamed_intraday = stream_intraday_from_assignments(
            m1_assignments,
            cash,
            session_schedule,
            panel.isins,
            sigma_asof=target_scale_sigma,
            kept_rows=kept_rows,
            workspace=workspace,
            official_log_return=intraday_official_return,
            completed_action_boundary=intraday_completed_boundary,
            same_day_boundary=intraday_same_day_boundary,
        )
    if minute_panel is not None:
        if not np.array_equal(minute_panel.dates, panel.dates):
            raise ValueError(
                "native M1 panel dates must exactly match the authoritative schedule"
            )
        minute_store_indices = np.asarray(
            [panel.isins.index(value) for value in minute_panel.isins],
            dtype=np.int64,
        )
        native = build_intraday_daily_features(
            minute_panel.open_brl,
            minute_panel.high_brl,
            minute_panel.low_brl,
            minute_panel.close_brl,
            minute_panel.volume,
            minute_panel.observed,
            volume_valid=minute_panel.volume_valid,
            session_valid=minute_panel.session_valid,
            sessions=session_schedule,
            official_log_return=intraday_official_return[:, minute_store_indices],
            completed_action_boundary=intraday_completed_boundary[
                :, minute_store_indices
            ],
            same_day_boundary=intraday_same_day_boundary[:, minute_store_indices],
        )
        aligned = _align_intraday_result(
            native, minute_panel.dates, minute_panel.isins, panel.dates, panel.isins
        )
        del native
        m1_session_close = aligned.session_close.copy()
        m1_session_close_valid = aligned.session_close_valid.copy()
        intraday_specs = feature_specs(
            "intraday",
            INTRADAY_DAILY_FEATURES,
            minimum_rank_names=minimum_rank_names,
        )
        transform_feature_panel_into(
            aligned.values,
            aligned.valid,
            universe.active,
            intraday_specs,
            intraday_values,
            intraday_valid,
            source_rows=kept_rows,
            minimum_rank_names=minimum_rank_names,
        )
        observation_age_sessions_into(
            aligned.valid,
            universe.active,
            intraday_age_sessions,
            source_rows=kept_rows,
            decision_rows=kept_rows,
            source_age_sessions=aligned.source_age_sessions,
        )
        intraday_support_fraction[:] = aligned.support_fraction[kept_rows]
        fast_sigma = np.where(aligned.valid[..., 14], aligned.values[..., 14], np.nan)
        fast_present = aligned.fast_present
        prefix_indices = np.asarray(
            [
                row.decision_time.hour * 60
                + row.decision_time.minute
                - row.continuous_open.hour * 60
                - row.continuous_open.minute
                for row in session_schedule
            ],
            dtype=np.int64,
        )
        if np.any(prefix_indices >= minute_panel.open_brl.shape[2]):
            raise ValueError("native M1 panel does not contain the decision entry bar")
        minute_entry = minute_panel.open_brl[
            np.arange(len(session_schedule)), :, prefix_indices
        ]
        minute_entry_valid = minute_panel.observed[
            np.arange(len(session_schedule)), :, prefix_indices
        ]
        minute_entry_valid &= np.isfinite(minute_entry) & (minute_entry > 0.0)
        entry = np.full(shape, np.nan, dtype=np.float32)
        entry_valid = np.zeros(shape, dtype=np.bool_)
        entry[:, minute_store_indices] = np.where(
            minute_entry_valid, minute_entry, np.nan
        ).astype(np.float32)
        entry_valid[:, minute_store_indices] = minute_entry_valid
        realized_daily = aligned.realized_daily_vol
        return_consistent = aligned.return_consistent
        patch_count = max(
            (
                (row.decision_time.hour * 60 + row.decision_time.minute)
                - (row.continuous_open.hour * 60 + row.continuous_open.minute)
            )
            // 5
            for row in session_schedule
        )
        native_full_shape = (
            len(session_schedule),
            len(minute_panel.isins),
            patch_count,
        )
        native_full = {
            "fast_patch_values": np.empty(
                (*native_full_shape, len(NATIVE_FAST_FEATURES)), dtype=np.float32
            ),
            "fast_patch_valid": np.empty(
                (*native_full_shape, len(NATIVE_FAST_FEATURES)), dtype=np.bool_
            ),
            "fast_patch_mask": np.empty(native_full_shape, dtype=np.bool_),
            "fast_last_price_age_minutes": np.empty(
                native_full_shape, dtype=np.float32
            ),
            "fast_last_price_age_valid": np.empty(native_full_shape, dtype=np.bool_),
        }
        build_native_fast_features_into(
            minute_panel.high_brl,
            minute_panel.low_brl,
            minute_panel.close_brl,
            minute_panel.volume,
            minute_panel.observed,
            volume_valid=(
                minute_panel.observed
                if minute_panel.volume_valid is None
                else minute_panel.volume_valid
            ),
            session_valid=(
                minute_panel.observed.any(axis=2)
                if minute_panel.session_valid is None
                else minute_panel.session_valid
            ),
            sigma_asof=target_scale_sigma[:, minute_store_indices],
            sessions=session_schedule,
            values_out=native_full["fast_patch_values"],
            valid_out=native_full["fast_patch_valid"],
            patch_mask_out=native_full["fast_patch_mask"],
            last_price_age_minutes_out=native_full["fast_last_price_age_minutes"],
            last_price_age_valid_out=native_full["fast_last_price_age_valid"],
        )
        native_fast_arrays = {
            name: _copy_selected_workspace_array(
                workspace, f"store_{name}", values, kept_rows
            )
            for name, values in native_full.items()
        }
        native_fast_mapping = pl.DataFrame(
            {
                "fast_index": np.arange(len(minute_panel.isins), dtype=np.int32),
                "store_name_index": minute_store_indices.astype(np.int32),
                "isin": minute_panel.isins,
                "security_id": [f"ISIN:{value}" for value in minute_panel.isins],
            }
        )
        del native_full
    elif streamed_intraday is not None:
        intraday_audit = streamed_intraday.audit
        intraday_source_paths = streamed_intraday.source_paths
        aligned = streamed_intraday.result
        streamed_workspace_arrays = (
            aligned.values,
            aligned.valid,
            aligned.decision_mark,
            aligned.decision_mark_valid,
            aligned.session_close,
            aligned.session_close_valid,
            aligned.realized_daily_vol,
            aligned.fast_present,
            aligned.return_consistent,
            aligned.support_fraction,
            aligned.source_age_sessions,
            streamed_intraday.to_close_entry,
            streamed_intraday.to_close_entry_valid,
        )
        if aligned.values.shape[:2] != shape:
            raise ValueError("streamed intraday derivatives are misaligned")
        m1_session_close = aligned.session_close.copy()
        m1_session_close_valid = aligned.session_close_valid.copy()
        intraday_specs = feature_specs(
            "intraday",
            INTRADAY_DAILY_FEATURES,
            minimum_rank_names=minimum_rank_names,
        )
        transform_feature_panel_into(
            aligned.values,
            aligned.valid,
            universe.active,
            intraday_specs,
            intraday_values,
            intraday_valid,
            source_rows=kept_rows,
            minimum_rank_names=minimum_rank_names,
        )
        observation_age_sessions_into(
            aligned.valid,
            universe.active,
            intraday_age_sessions,
            source_rows=kept_rows,
            decision_rows=kept_rows,
            source_age_sessions=aligned.source_age_sessions,
        )
        intraday_support_fraction[:] = aligned.support_fraction[kept_rows]
        fast_sigma = np.where(aligned.valid[..., 14], aligned.values[..., 14], np.nan)
        fast_present = aligned.fast_present
        entry = streamed_intraday.to_close_entry
        entry_valid = streamed_intraday.to_close_entry_valid
        realized_daily = aligned.realized_daily_vol
        return_consistent = aligned.return_consistent
        native_fast_arrays = dict(streamed_intraday.native_arrays)
        native_fast_mapping = streamed_intraday.native_mapping

    to_close_arrays: dict[str, NDArray[np.generic]] = {}
    if has_intraday:
        to_close = build_to_close_target(
            entry,
            np.where(m1_session_close_valid, m1_session_close, np.nan),
            realized_daily,
            universe.active,
            fast_present & entry_valid & return_consistent,
        )
        for name, values in (
            ("target_to_close", to_close.target),
            ("target_to_close_valid", to_close.valid),
            (
                "target_to_close_normalized_residual",
                to_close.normalized_residual,
            ),
            ("target_to_close_raw_log_return", to_close.raw_log_return),
            ("m1_cotahist_return_consistent_mask", return_consistent),
        ):
            to_close_arrays[name] = _copy_selected_workspace_array(
                workspace, f"store_{name}", values, kept_rows
            )
        del to_close
        if "aligned" in locals():
            del aligned
        streamed_intraday = None
        gc.collect()

    target_shape = (kept_rows.size, len(panel.isins), len(HORIZONS))
    target_arrays: dict[str, NDArray[np.generic]] = {
        "target_primary": _workspace_array(
            workspace, "store_target_primary", target_shape, np.float32
        ),
        "target_valid": _workspace_array(
            workspace, "store_target_valid", target_shape, np.bool_
        ),
        "target_normalized_residual": _workspace_array(
            workspace,
            "store_target_normalized_residual",
            target_shape,
            np.float32,
        ),
        "target_normalized_cross_section_valid": _workspace_array(
            workspace,
            "store_target_normalized_cross_section_valid",
            (kept_rows.size, len(HORIZONS)),
            np.bool_,
        ),
        "target_shareholder_midrank": _workspace_array(
            workspace, "store_target_shareholder_midrank", target_shape, np.float32
        ),
        "target_shareholder_valid": _workspace_array(
            workspace, "store_target_shareholder_valid", target_shape, np.bool_
        ),
        "target_shareholder_simple_return": _workspace_array(
            workspace,
            "store_target_shareholder_simple_return",
            target_shape,
            np.float32,
        ),
        "target_terminal_wealth": _workspace_array(
            workspace, "store_target_terminal_wealth", target_shape, np.float32
        ),
        "target_terminal_loss": _workspace_array(
            workspace, "store_target_terminal_loss", target_shape, np.bool_
        ),
        "target_price_midrank": _workspace_array(
            workspace, "store_target_price_midrank", target_shape, np.float32
        ),
        "target_price_valid": _workspace_array(
            workspace, "store_target_price_valid", target_shape, np.bool_
        ),
        "target_price_simple_return": _workspace_array(
            workspace,
            "store_target_price_simple_return",
            target_shape,
            np.float32,
        ),
    }
    build_economic_multi_day_targets_into(
        panel.close_brl,
        panel.observed,
        universe.active,
        target_scale_sigma,
        retrospective_actions,
        primary=target_arrays["target_primary"],
        primary_valid=target_arrays["target_valid"],
        normalized_residual=target_arrays["target_normalized_residual"],
        normalized_cross_section_valid=target_arrays[
            "target_normalized_cross_section_valid"
        ],
        shareholder_midrank=target_arrays["target_shareholder_midrank"],
        shareholder_valid=target_arrays["target_shareholder_valid"],
        shareholder_simple_return=target_arrays["target_shareholder_simple_return"],
        terminal_wealth=target_arrays["target_terminal_wealth"],
        terminal_loss=target_arrays["target_terminal_loss"],
        price_midrank=target_arrays["target_price_midrank"],
        price_valid=target_arrays["target_price_valid"],
        price_simple_return=target_arrays["target_price_simple_return"],
        source_rows=kept_rows,
        minimum_rank_names=minimum_rank_names,
    )
    eventual_survival_groups = _eventual_survival_groups(
        panel.dates, panel.observed, continuation_isins
    )
    eventual_survival = np.broadcast_to(
        eventual_survival_groups["survives_to_final_year"][None, :],
        panel.observed.shape,
    )
    full_store_arrays: dict[str, NDArray[np.generic]] = {
        "active": universe.active,
        "observed": panel.observed,
        "trade_observed": panel.trade_observed,
        "activity_valid": panel.activity_valid,
        "source_session_complete": np.broadcast_to(
            panel.source_session_complete[:, None], panel.observed.shape
        ),
        "slow_timestep_valid": slow_timestep_valid,
        "raw_open": panel.open_brl,
        "raw_high": panel.high_brl,
        "raw_low": panel.low_brl,
        "raw_close": panel.close_brl,
        "shareholder_wealth_open": shareholder_wealth_open,
        "shareholder_wealth_high": shareholder_wealth_high,
        "shareholder_wealth_low": shareholder_wealth_low,
        "shareholder_wealth_close": shareholder_wealth_close,
        "shareholder_wealth_valid": shareholder_wealth_valid,
        "action_shares_per_prior_share": retrospective_actions.shares_per_prior_share,
        "action_cash_per_prior_share": retrospective_actions.cash_per_prior_share,
        "action_session_resolved": retrospective_actions.session_resolved,
        "action_has_action": retrospective_actions.has_action,
        "action_successor_index": retrospective_actions.successor_index,
        "action_payment_session": action_payment_session,
        "prior_reference_close": universe.prior_close_brl,
        "audit_eventual_survives_to_final_year": eventual_survival,
        "volume_brl": panel.volume_brl,
        "trade_count": panel.trades,
        "quantity": panel.quantity,
        "distribution_number": panel.distribution_number,
        "distribution_change_mask": distribution_changed,
        "detected_event_mask": detected_actions.event_candidate,
        "detected_split_mask": detected_actions.split_event,
        "detected_cash_event_mask": detected_actions.cash_event,
        "ambiguous_action_mask": detected_actions.ambiguous_event,
        "price_jump_anomaly_mask": detected_actions.price_jump_anomaly_mask,
        "intraday_boundary_lagged_mask": diagnostic_intraday_boundary_lagged,
        "intraday_boundary_sameday_mask": diagnostic_intraday_boundary_sameday,
        "intraday_unit_or_unresolved_boundary_mask": (
            intraday_unit_or_unresolved_boundary
        ),
    }
    if inferred_actions is not None:
        full_store_arrays.update(
            {
                "inferred_action_u1_mask": inferred_actions.u1_event,
                "inferred_action_c1_mask": inferred_actions.c1_event,
                "inferred_action_u2_mask": inferred_actions.u2_event,
                "inferred_action_large_move_no_action_mask": (
                    inferred_actions.large_move_no_action
                ),
            }
        )
    arrays: dict[str, NDArray[np.generic]] = {
        name: _copy_selected_workspace_array(
            workspace,
            f"store_{name}",
            values,
            kept_rows,
            dtype=(np.float32 if np.issubdtype(values.dtype, np.floating) else None),
        )
        for name, values in full_store_arrays.items()
    }
    arrays.update(
        {
            "slow_values": slow_values,
            "slow_valid": slow_valid,
            "slow_age_sessions": slow_age_sessions,
            "intraday_values": intraday_values,
            "intraday_valid": intraday_valid,
            "intraday_age_sessions": intraday_age_sessions,
            "intraday_support_fraction": intraday_support_fraction,
            "fast_present": _copy_selected_workspace_array(
                workspace, "store_fast_present", fast_present, kept_rows
            ),
            "target_scale_sigma": _copy_selected_workspace_array(
                workspace,
                "store_target_scale_sigma",
                target_scale_sigma,
                kept_rows,
            ),
            "common_state_diagnostic_values": common_state_values,
            "common_state_diagnostic_valid": common_state_valid,
        }
    )
    arrays.update(target_arrays)
    del target_arrays
    gc.collect()
    arrays.update(to_close_arrays)
    arrays.update(native_fast_arrays)
    feature_names: dict[str, Sequence[str]] = {
        "slow": SLOW_FEATURES,
        "intraday": INTRADAY_DAILY_FEATURES,
        "native_fast": NATIVE_FAST_FEATURES,
        "common_state_diagnostic": COMMON_STATE_DIAGNOSTICS,
        "horizons": tuple(str(value) for value in HORIZONS),
    }
    all_feature_specs: list[FeatureSpec] = [
        *slow_specs,
        *feature_specs(
            "intraday",
            INTRADAY_DAILY_FEATURES,
            minimum_rank_names=minimum_rank_names,
        ),
        *native_fast_feature_specs(),
    ]
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
                m1_assignments,
                panel.volume_brl,
            )
            yield group, parsed.pop(group)

    sidecar_coverage_frames: list[pl.DataFrame] = []
    sidecar_survival_frames: list[pl.DataFrame] = []
    sidecar_liquidity_frames: list[pl.DataFrame] = []
    sidecar_contemporaneity_rows: list[dict[str, object]] = []
    sidecar_capability_rows: list[dict[str, object]] = []
    sidecar_capability_summary: dict[str, dict[str, list[str]]] = {}
    prior_adv20 = _prior_adv20(
        decision_continuation.volume_brl[keep],
        decision_continuation.activity_valid[keep],
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
        enabled = list(result.feature_names)
        missing = list(result.source_missing_candidates)
        sidecar_capability_summary[group] = {
            "enabled": enabled,
            "source_missing": missing,
        }
        sidecar_capability_rows.extend(
            {
                "group": group,
                "candidate_feature": candidate,
                "enabled": candidate in result.feature_names,
                "status": (
                    "enabled_exact_source_semantics"
                    if candidate in result.feature_names
                    else "disabled_source_semantics_unavailable"
                ),
            }
            for candidate in SIDECAR_FEATURES[group]
        )
        if not result.feature_names:
            del result
            gc.collect()
            continue
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
        age_sessions = _workspace_array(
            workspace,
            f"sidecar_{group}_age_sessions",
            values.shape,
            np.float32,
            fill=-1.0,
        )
        sidecar_specs = feature_specs(
            f"sidecar_{group}",
            result.feature_names,
            minimum_rank_names=minimum_rank_names,
        )
        transform_feature_panel_into(
            result.values,
            result.valid,
            universe.active,
            sidecar_specs,
            values,
            valid,
            source_rows=kept_rows,
            minimum_rank_names=minimum_rank_names,
        )
        source_ages = np.asarray(result.age_sessions[kept_rows], dtype=np.float32)
        if source_ages.shape != age_sessions.shape:
            raise ValueError(f"sidecar {group} source ages are misaligned")
        age_sessions[...] = np.where(universe.active[keep, :, None], source_ages, -1.0)
        arrays[f"sidecar_{group}_values"] = values
        arrays[f"sidecar_{group}_valid"] = valid
        arrays[f"sidecar_{group}_age_sessions"] = age_sessions
        feature_names[f"sidecar_{group}"] = result.feature_names
        all_feature_specs.extend(sidecar_specs)
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
    del prior_adv20
    gc.collect()

    tables = {
        "security_master": build_security_master(
            cash, succession_links=isin_successions
        ),
        "isin_succession_candidates": proposed_isin_successions,
        "isin_succession_links": isin_successions,
        "b3_session_schedule": schedule_frame(session_schedule),
        "calendar_completeness": calendar_completeness_table(
            session_schedule, archive_dates
        ),
        "raw_validation_by_year": validation.audit_by_year,
        "raw_validation_rejections": validation.rejected,
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
        "cotahist_action_counts_by_year": _cotahist_action_counts_by_year(
            panel.dates,
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
        "corporate_actions_provider_observations": checked_actions,
        "corporate_actions_verified_terms": verified_action_terms_to_table(
            verified_action_terms
        ),
        "corporate_action_alignment_roles": _action_alignment_role_table(
            verified_action_terms, panel.dates, daily_action_cutoffs
        ),
        "inferred_action_counts_by_year": _inferred_action_counts_by_year(
            panel.dates, inferred_actions
        ),
        "large_move_no_action_by_year": _large_move_no_action_by_year(
            panel.dates, inferred_actions
        ),
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
        "sidecar_capabilities": pl.DataFrame(
            sidecar_capability_rows,
            schema={
                "group": pl.String,
                "candidate_feature": pl.String,
                "enabled": pl.Boolean,
                "status": pl.String,
            },
        ),
        "external_feature_validity_by_survival_adv20_quartile": (
            pl.concat(sidecar_liquidity_frames)
            if sidecar_liquidity_frames
            else pl.DataFrame()
        ),
    }
    if native_fast_mapping is not None:
        tables["native_fast_security_mapping"] = native_fast_mapping
    if action_acquisition_audit is not None:
        tables["corporate_action_acquisition_audit"] = action_acquisition_audit
    if has_intraday:
        tables["m1_cotahist_mismatch_by_year"] = m1_cotahist_mismatch_by_year(
            panel.dates,
            m1_session_close,
            m1_session_close_valid,
            panel.close_brl,
            panel.observed,
        )
        ratio_valid = (
            m1_session_close_valid
            & panel.observed
            & np.isfinite(m1_session_close)
            & (m1_session_close > 0)
            & np.isfinite(panel.close_brl)
            & (panel.close_brl > 0)
        )
        ratio_rows, ratio_names = np.nonzero(ratio_valid & keep[:, None])
        tables["m1_cotahist_level_ratio"] = pl.DataFrame(
            {
                "trade_date": panel.dates[ratio_rows].astype("datetime64[ms]"),
                "isin": np.asarray(panel.isins)[ratio_names],
                "m1_to_cotahist_close": m1_session_close[ratio_rows, ratio_names]
                / panel.close_brl[ratio_rows, ratio_names],
                "return_consistent": return_consistent[ratio_rows, ratio_names],
                "completed_action_boundary": intraday_completed_boundary[
                    ratio_rows, ratio_names
                ],
            }
        ).with_columns(pl.col("trade_date").cast(pl.Date))
    if intraday_audit is not None:
        tables["m1_source_audit"] = intraday_audit
    tables["common_state_diagnostics"] = common_state_table
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
    target_audits = [
        _target_validity_tables(
            kept_dates,
            np.asarray(arrays[mask_name], dtype=np.bool_),
            universe.active[keep],
            panel.observed[keep],
            continuation_isins,
            family=family,
        )
        for family, mask_name in (
            ("scaled_median_adjusted", "target_valid"),
            ("shareholder", "target_shareholder_valid"),
            ("price", "target_price_valid"),
        )
    ]
    tables["target_validity_by_year"] = pl.concat(
        [yearly for yearly, _ in target_audits]
    )
    tables["target_validity_by_survival"] = pl.concat(
        [survival for _, survival in target_audits]
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
    wealth_restarts = wealth_chain_restart_mask(shareholder_wealth_valid)[keep]
    active_wealth_restarts = wealth_restarts & universe.active[keep]
    calendar_year = kept_dates.astype("datetime64[Y]").astype(np.int64) + 1970
    wealth_restarts_by_year = {
        str(year): int(active_wealth_restarts[calendar_year == year].sum())
        for year in sorted(set(calendar_year.tolist()))
    }
    build_peak_rss = peak_rss_bytes()
    metadata = {
        "store_start": str(kept_dates[0]),
        "store_end": str(kept_dates[-1]),
        "final_daily_action_cutoff": (
            daily_action_cutoffs[-1].isoformat() if daily_action_cutoffs[-1] else None
        ),
        "lookback_rows_materialized": False,
        "slow_entry_alignment": dict(DECISION_FEATURE_CONTRACT),
        "native_fast": {
            "enabled": native_fast_mapping is not None,
            "security_count": (
                native_fast_mapping.height if native_fast_mapping is not None else 0
            ),
            "channels": list(NATIVE_FAST_FEATURES),
            "time_axis": (
                "date-specific continuous-open to 15:45 decision prefix; "
                "five-minute completed blocks"
            ),
            "legacy_v1_artifact_required": False,
        },
        "implementation_git_commit": implementation_commit,
        "action_terms_source": action_terms_source,
        "schedule_source": resolved_schedule_source,
        "schedule_reconstruction_audit": (
            dict(schedule_reconstruction_audit)
            if schedule_reconstruction_audit is not None
            else None
        ),
        "isin_succession_candidate_count": proposed_isin_successions.height,
        "isin_succession_link_count": isin_successions.height,
        "cotahist_action_detection_role": (
            "development_grade_inferred_terms"
            if inferred_actions is not None
            else "diagnostic_only"
        ),
        "survival_identity": "permanent ISIN; only source-verified conversions may link",
        "eventual_survival_audit": (
            "future panel observation through the final store year; audit-only and "
            "never exposed to features, eligibility, normalization, or orders"
        ),
        "terminal_status": (
            "verified source unavailable; the evaluation ledger may apply an "
            "explicitly registered and labelled terminal-settlement convention"
        ),
        "common_state_diagnostics": {
            "role": "audit_only_not_model_input",
            "feature_names": list(COMMON_STATE_DIAGNOSTICS),
            "decision_timing": (
                "return statistics use shareholder wealth t-2 to t-1; median "
                "raw volatility uses already-lagged sigma_asof[t]"
            ),
            "minimum_support_names": minimum_rank_names,
        },
        "feature_history_identity": (
            "permanent ISIN; verified conversions known by the effective-session "
            "decision route contractual shareholder wealth and separately rebase "
            "price/liquidity/history inputs by their units"
        ),
        "calendar_contract": {
            "schema": "BRAZIL_RV_B3_EQUITY_SESSION_SCHEDULE_V1",
            "schedule_source": resolved_schedule_source,
            "authority": (
                "reconstructed schedule cross-checked against committed holidays "
                "with zero unexplained exceptions"
                if resolved_schedule_source == "reconstructed_v1"
                else "explicit versioned schedule"
            ),
            "minimum_name_count_is_diagnostic_only": True,
        },
        "market_observation_masks": {
            "observed": "raw daily price row observed",
            "trade_observed": "raw daily activity row observed",
            "activity_valid": (
                "activity row observed or exact no-trade zero certified by a "
                "complete source session after causal history begins"
            ),
            "source_session_complete": (
                "date-level raw-source coverage, broadcast on the stored name axis"
            ),
            "incomplete_source_policy": "price and activity remain invalid; never zero",
            "m1_activity_policy": (
                "volume validity is independent of price observation; absent minutes "
                "remain invalid because the accepted M1 archive does not certify zeros"
            ),
        },
        "raw_validation": {
            "maximum_rejection_fraction": 0.005,
            "rejection_fraction": validation.rejection_fraction,
            "rejected_rows": validation.rejected.height,
            "exact_duplicate_rows_collapsed": validation.exact_duplicate_rows_collapsed,
        },
        "feature_schema": {
            "specifications": [asdict(spec) for spec in all_feature_specs],
            "sha256": feature_schema_sha256(all_feature_specs),
            "minimum_rank_names": minimum_rank_names,
        },
        "feature_age_contract": dict(FEATURE_AGE_CONTRACT),
        "intraday_consistency": {
            "comparison": "adjacent exact M1 close log return versus COTAHIST shareholder wealth",
            "maximum_absolute_log_return_difference": 0.005,
            "clock": "completed session validation only; never gate its own decision prefix",
            "action_rows": "exclude completed inferred-action/identity rows; current open-known boundaries only",
            "rolling_minimum_fraction": 0.8,
            "rolling_sums": "sum actual observations; no extrapolation or filling",
            "activity": "only physical rows certify activity; absent sparse-archive minutes remain unknown",
            "source_age": "sessions since newest observation consumed; independent of feature validity; unknown -1",
            "to_close_target": "exact entry and continuous-close prices both in M1 units",
        },
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
        "sidecar_capabilities": sidecar_capability_summary,
        "return_definition": {
            "primary": (
                "log shareholder wealth over (t,t+H], cross-sectionally "
                "median-adjusted, divided name-by-name by the strictly lagged "
                "Yang-Zhang scale at t times sqrt(H), clipped, then tie-aware "
                "mid-ranked"
            ),
            "shareholder_family": (
                "labelled action-tier quantity multipliers plus cash entitlements "
                "held as cash; unresolved action coverage invalidates every crossing "
                "interval"
            ),
            "price_family": (
                "economic-share price return following action-tier q and successor "
                "claims while excluding cash entitlements, with distinct masks; "
                "never substituted for gross shareholder wealth"
            ),
            "entry_exit_price": "daily last-trade close proxy",
            "synthetic_cash_distribution": "disabled",
        },
        "corporate_action_contract": {
            "action_terms_source": action_terms_source,
            "coverage_status": (
                "inferred" if inferred_actions is not None else "verified"
            ),
            "provider_rows_enter_model_arrays": inferred_actions is None,
            "provider_rows_are_audit_only": inferred_actions is not None,
            "inferred_terms_are_verified": False,
            "legacy_realized_ratio_classifier_is_diagnostic_only": True,
            "stored_action_arrays": "retrospective outcome/accounting terms",
            "feature_action_alignment": (
                "daily row e uses terms known by the next session's 15:45 decision; "
                "same-day intraday boundaries use only the open-gap diagnostic"
            ),
            "currency": "BRL",
            "retrospective_coverage_resolved_fraction": float(
                retrospective_actions.session_resolved.mean()
            ),
            "decision_known_coverage_resolved_fraction": float(
                decision_actions.session_resolved.mean()
            ),
            "payment_date_modeling": (
                "inferred ex-session payment under the development-grade tier"
                if inferred_actions is not None
                else "unsupported unless explicitly sourced"
            ),
            "complex_or_unmapped_terms": "unresolved",
        },
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
        "wealth_chain_restarts_active_name_days": wealth_restarts_by_year,
        "resource_preflight": dict(resource_preflight or {}),
    }
    options_composition = tables["external_feature_validity_by_survival_adv20_quartile"]
    if options_composition.width:
        options_composition = options_composition.filter(
            pl.col("family") == "sidecar_options"
        )
        if options_composition.height:
            unstratified = options_composition.filter(
                pl.col("prior_adv20_quartile") == 0
            ).row(0, named=True)
            binding = options_composition.filter(pl.col("stratum_is_binding"))
            reported = options_composition.filter(
                (pl.col("prior_adv20_quartile") > 0) & (~pl.col("stratum_is_binding"))
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
    try:
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
    finally:
        for value in arrays.values():
            close_memmap(value)
        for value in (
            *streamed_workspace_arrays,
            shareholder_wealth_open,
            shareholder_wealth_high,
            shareholder_wealth_low,
            shareholder_wealth_close,
            shareholder_wealth_valid,
            slow_sigma,
            target_scale_sigma,
        ):
            close_memmap(value)
        workspace_handle.cleanup()
    return result


def load_minute_npz(path: Path) -> MinutePanel:
    archive = np.load(path, allow_pickle=False)
    required = {
        "dates",
        "isins",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "observed",
        "volume_valid",
        "session_valid",
    }
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
        volume_valid=archive["volume_valid"],
        session_valid=archive["session_valid"],
    )


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the immutable Brazil-RV v2 daily store"
    )
    parser.add_argument("--cotahist-root", required=True, type=Path)
    parser.add_argument("--cotahist-raw-root", required=True, type=Path)
    parser.add_argument("--cotahist-parse-audit", required=True, type=Path)
    parser.add_argument("--session-schedule", required=True, type=Path)
    parser.add_argument("--session-schedule-audit", type=Path)
    parser.add_argument(
        "--action-terms-source",
        choices=("verified_contractual_terms", "inferred_cotahist_dismes_v1"),
        default="verified_contractual_terms",
    )
    parser.add_argument(
        "--isin-links-allowlist",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "configs"
        / "v2"
        / "isin_links_allowlist.csv",
    )
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--actions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--store-end", type=date.fromisoformat, default=DEVELOPMENT_END)
    parser.add_argument(
        "--previous-store",
        required=True,
        type=Path,
        help="Prior immutable v2 store used to enforce three-times disk headroom",
    )
    parser.add_argument(
        "--i-understand-low-memory-risk",
        action="store_true",
        help=(
            "Explicitly admit a build below the 10-GiB memory gate; disk and "
            "staging-location gates remain mandatory"
        ),
    )
    parser.add_argument("--minute-npz", type=Path)
    parser.add_argument("--m1-assignments", required=True, type=Path)
    parser.add_argument(
        "--sidecar",
        action="append",
        default=[],
        metavar="GROUP=PARQUET",
        help="Known publication-lagged sidecar archive; may be repeated",
    )
    return parser.parse_args(arguments)


def _load_action_bundle(
    actions_path: Path,
    *,
    end_date: date | None = None,
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
        pl.scan_parquet(paths["actions"])
        .filter(pl.lit(True) if end_date is None else pl.col("ex_date") <= end_date)
        .collect(),
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
        grouped.setdefault(group, [])
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
                "annual_taker_rate",
                "loan_rate_annual_decimal",
                "lending_taker_fee_annual_decimal",
            ),
            "oddlot": (
                "source_trade_date",
                "regular_volume_brl",
                "odd_lot_volume_brl",
            ),
            "events": (
                "filing_receipt_timestamp",
                "event_type",
            ),
            "options": (
                "source_trade_date",
                "put_oi",
                "call_oi",
                "put_open_interest",
                "call_open_interest",
                "oi_snapshot_complete",
                "open_interest_snapshot_complete",
            ),
            "fundamentals": (
                "source_receipt_date",
                "receipt_timestamp",
                "total_liabilities_brl",
                "total_assets_brl",
                "liabilities_brl",
                "assets_brl",
            ),
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
                "public_available_at",
                "available_timestamp",
                "delivery_timestamp",
                "filing_receipt_timestamp",
                "receipt_timestamp",
                "state_asof_timestamp",
                "timestamp_precision",
                "processing_latency_seconds",
                "revision",
                "version",
                "fetched_at",
                "event_time",
                "effective_time",
            )
            if column in schema
        ]
        for timestamp_column in (
            "public_available_at",
            "available_timestamp",
            "delivery_timestamp",
            "filing_receipt_timestamp",
            "receipt_timestamp",
            "state_asof_timestamp",
        ):
            optional.extend(
                column
                for column in (
                    f"{timestamp_column}_precision",
                    f"{timestamp_column}_processing_latency_seconds",
                )
                if column in schema
            )
        masks = [
            f"{column}_mask" for column in value_columns if f"{column}_mask" in schema
        ]
        projected = list(
            dict.fromkeys(
                ["available_date", identity, *optional, *value_columns, *masks]
            )
        )
        lazy = (
            pl.scan_parquet(path)
            .select(projected)
            .filter(pl.col("available_date") <= normalized_dates[-1])
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
            "public_available_at",
            "available_timestamp",
            "delivery_timestamp",
            "filing_receipt_timestamp",
            "receipt_timestamp",
            "state_asof_timestamp",
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
        source = derive_known_archive_features(
            source,
            dates,
            isins,
            group=group,
            daily_volume_brl=daily_volume_brl,
        )
        provided = available_archive_mapping(group, source.columns)
        partitions: tuple[tuple[str, tuple[str, ...], str | None], ...]
        if group == "lending":
            partitions = (
                (
                    "lending_position",
                    SIDECAR_FEATURES["lending"][:3],
                    "source_position_date",
                ),
                (
                    "lending_rate",
                    SIDECAR_FEATURES["lending"][3:],
                    "source_trade_date",
                ),
            )
        else:
            partitions = ((group, SIDECAR_FEATURES[group], None),)
        for record_family, partition_features, source_date_column in partitions:
            partition_columns = sorted(
                {
                    provided[name]
                    for name in partition_features
                    if provided.get(name) is not None
                }
            )
            if not partition_columns:
                continue
            partition = source
            if source_date_column is not None and source_date_column in source.columns:
                partition = partition.filter(pl.col(source_date_column).is_not_null())
            partition = partition.with_columns(
                pl.lit(record_family).alias("__record_family"),
                *(
                    pl.lit(True).alias(f"__provided__{column}")
                    for column in partition_columns
                ),
            )
            grouped.setdefault(group, []).append(partition)
    output: dict[str, SidecarResult] = {}
    for group, sources in grouped.items():
        if sources:
            source = pl.concat(sources, how="diagonal_relaxed")
        else:
            source = pl.DataFrame(schema={"available_date": pl.Date, "isin": pl.String})
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


def _validate_cotahist_parse_audit(
    audit_path: Path, raw_sources: Sequence[Path]
) -> None:
    """Require a successful parser audit bound to every exact raw archive."""

    payload = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    rows = payload.get("audits") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("COTAHIST parse audit has the wrong schema")
    expected = {
        int(re.search(r"A(\d{4})\.ZIP$", path.name, re.IGNORECASE).group(1)): (
            path.resolve(),
            str(source_records((path,))[0]["sha256"]),
        )
        for path in raw_sources
    }
    if not set(expected).issubset(int(row.get("year", -1)) for row in rows):
        raise ValueError("COTAHIST parse audit does not cover the requested raw inputs")
    for raw in rows:
        year = int(raw["year"])
        if year not in expected:
            continue
        source_path, source_sha = expected[year]
        if raw.get("error"):
            raise ValueError(f"COTAHIST parser failed for {year}: {raw['error']}")
        if raw.get("record_count_valid") is not True:
            raise ValueError(f"COTAHIST record-count audit failed for {year}")
        if (
            int(raw.get("header_records", 0)) != 1
            or int(raw.get("trailer_records", 0)) != 1
        ):
            raise ValueError(f"COTAHIST structural audit failed for {year}")
        if (
            int(raw.get("malformed_length_records", 0)) != 0
            or int(raw.get("other_records", 0)) != 0
        ):
            raise ValueError(f"COTAHIST malformed-record audit failed for {year}")
        if Path(str(raw.get("source_zip", ""))).resolve() != source_path:
            raise ValueError(f"COTAHIST raw path mismatch in parse audit for {year}")
        if raw.get("source_sha256") != source_sha:
            raise ValueError(f"COTAHIST raw hash mismatch in parse audit for {year}")


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_args(arguments)
    if not re.fullmatch(r"[0-9a-f]{40}", args.implementation_commit):
        raise ValueError("--implementation-commit must be a full lowercase Git SHA")
    repository = Path(__file__).resolve().parents[4]
    _require_clean_implementation_commit(repository, args.implementation_commit)
    resource_preflight = _build_resource_preflight(
        output_dir=args.output_dir,
        previous_store=args.previous_store,
    )
    resource_preflight["low_memory_override_authorized"] = bool(
        args.i_understand_low_memory_risk
    )
    resource_preflight["effective_violations"] = [
        violation
        for violation in resource_preflight["violations"]
        if not (
            args.i_understand_low_memory_risk
            and violation == "available_build_memory_below_10_gib"
        )
    ]
    resource_preflight["effective_passed"] = not resource_preflight[
        "effective_violations"
    ]
    print(
        json.dumps({"resource_preflight": resource_preflight}, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )
    _require_build_resource_preflight(
        resource_preflight,
        allow_low_memory=args.i_understand_low_memory_risk,
    )
    years = tuple(range(2009, args.store_end.year + 1))
    raw_sources = tuple(
        (args.cotahist_raw_root / f"COTAHIST_A{year}.ZIP").resolve() for year in years
    )
    missing_raw = [str(path) for path in raw_sources if not path.is_file()]
    if missing_raw:
        raise FileNotFoundError(f"COTAHIST raw archives missing: {missing_raw}")
    if not args.cotahist_parse_audit.is_file():
        raise FileNotFoundError(args.cotahist_parse_audit)
    _validate_cotahist_parse_audit(args.cotahist_parse_audit, raw_sources)
    full_schedule = load_session_schedule(args.session_schedule)
    schedule = tuple(
        row
        for row in full_schedule
        if row.trade_date <= args.store_end
    )
    # Calendar metadata preserves the last completed row's original information
    # cutoff. No market data or consumer row for that following session is read.
    following_decision_at = next(
        (row.decision_at for row in full_schedule if row.trade_date > args.store_end),
        None,
    )
    resolved_schedule_source = schedule_source_label(schedule)
    schedule_reconstruction_audit: dict[str, object] | None = None
    schedule_audit_path: Path | None = None
    if resolved_schedule_source == "reconstructed_v1":
        if args.session_schedule_audit is None:
            raise ValueError("reconstructed_v1 requires --session-schedule-audit")
        schedule_audit_path = args.session_schedule_audit.resolve()
        schedule_reconstruction_audit = json.loads(
            schedule_audit_path.read_text(encoding="utf-8")
        )
        exception_record = schedule_reconstruction_audit.get(
            "exception_explanations", {}
        )
        if (
            schedule_reconstruction_audit.get("schedule_source") != "reconstructed_v1"
            or schedule_reconstruction_audit.get("schedule_sha256")
            != source_records((args.session_schedule,))[0]["sha256"]
            or not isinstance(exception_record, dict)
            or exception_record.get("unexplained_count") != 0
        ):
            raise ValueError(
                "reconstructed schedule audit does not bind a closed schedule"
            )
    if args.m1_assignments.is_dir():
        assignment_path = (
            args.m1_assignments / "xp_accepted_source_assignments_v1.parquet"
        )
    else:
        assignment_path = args.m1_assignments
    assignments = pl.read_parquet(assignment_path)
    required_assignment_columns = {"security_id", "isin", "source_file"}
    if (
        not required_assignment_columns.issubset(assignments.columns)
        or assignments.is_empty()
        or assignments.get_column("security_id").n_unique() != assignments.height
        or assignments.get_column("isin").n_unique() != assignments.height
    ):
        raise ValueError(
            "canonical M1 assignments must bind unique security_id and ISIN rows"
        )
    m1_isins = tuple(assignments.get_column("isin").cast(pl.String).to_list())
    paths = [
        args.cotahist_root / f"year={year}" / f"equities_daily_{year}.parquet"
        for year in years
    ]
    daily = load_cotahist(paths, v1_isins=m1_isins, end_date=args.store_end)
    foundation = daily
    available_years = set(
        foundation.get_column("trade_date").dt.year().unique().to_list()
    )
    if available_years != set(years):
        raise ValueError(
            "canonical COTAHIST foundation must contain exactly years "
            f"{years}; got {sorted(available_years)}"
        )
    actions, acquisition_audit, action_master, action_sources = _load_action_bundle(
        args.actions,
        end_date=args.store_end,
    )
    expected_master = build_security_master(foundation)
    # Provider master may extend beyond this build. Only its identity coverage
    # is needed; its future last-observation dates never define the model axis.
    if not expected_master.join(
        action_master, on=["isin", "ticker"], how="anti"
    ).is_empty():
        raise ValueError("corporate-action security master misses a COTAHIST identity")
    prior_manifest = json.loads((args.previous_store / "manifest.json").read_text())
    axis_path = args.previous_store / "isin_index.npy"
    if (
        source_records([axis_path])[0]["sha256"]
        != prior_manifest["indices"]["isin_index.npy"]["sha256"]
    ):
        raise ValueError("previous-store security axis hash mismatch")
    security_axis = tuple(np.load(axis_path, allow_pickle=False).tolist())
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
        m1_assignments=assignments,
        source_paths=(
            *paths,
            assignment_path,
            args.session_schedule,
            *((schedule_audit_path,) if schedule_audit_path is not None else ()),
            args.isin_links_allowlist,
            *action_sources,
            *((args.minute_npz,) if args.minute_npz else ()),
            *(Path(value.split("=", 1)[1]) for value in args.sidecar),
        ),
        implementation_commit=args.implementation_commit,
        security_axis=security_axis,
        cotahist_raw_sources=raw_sources,
        cotahist_parse_audit=args.cotahist_parse_audit,
        session_schedule=schedule,
        following_decision_at=following_decision_at,
        isin_link_allowlist=args.isin_links_allowlist,
        resource_preflight=resource_preflight,
        action_terms_source=args.action_terms_source,
        schedule_source=resolved_schedule_source,
        schedule_reconstruction_audit=schedule_reconstruction_audit,
    )
    print(json.dumps({"store": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
