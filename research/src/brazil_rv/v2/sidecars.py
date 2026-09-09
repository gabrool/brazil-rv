from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
from numpy.typing import NDArray

from .contract import SIDECAR_FEATURES


B3_TIMEZONE = ZoneInfo("America/Sao_Paulo")
_PUBLICATION_ORDER_COLUMN = "__publication_order_us"
_ROW_INDEX_COLUMN = "__row_index"
_TIMESTAMP_COLUMNS = (
    "public_available_at",
    "available_timestamp",
    "delivery_timestamp",
    "filing_receipt_timestamp",
    "receipt_timestamp",
    "state_asof_timestamp",
)
_PRECISION_UPPER_BOUND = {
    "instant": timedelta(0),
    "exact": timedelta(0),
    "microsecond": timedelta(microseconds=1),
    "millisecond": timedelta(milliseconds=1),
    "second": timedelta(seconds=1),
    "minute": timedelta(minutes=1),
}


ARCHIVE_COLUMN_MAP: dict[str, dict[str, str | None]] = {
    "lending": {
        # These three columns are derived below from the raw BRL balance and
        # raw COTAHIST BRL volume.  The v1 tanh/log transforms are not relabelled
        # as raw v2 quantities.
        "loan_balance_to_volume_20": "loan_balance_to_volume_20",
        "loan_balance_change_1": "loan_balance_change_1",
        "loan_balance_change_5": "loan_balance_change_5",
        # These columns are emitted only by the raw annual-decimal adapter.
        # Saturated legacy tanh fields are deliberately not inverted.
        "loan_rate": "loan_rate",
        "loan_rate_change_5": "loan_rate_change_5",
    },
    "events": {
        "sessions_since_financial_filing": "sessions_since_financial_filing",
        "standardized_unexpected_earnings": None,
    },
    "options": {
        "put_call_log_oi_ratio": "put_call_log_oi_ratio",
        "delta_oi_to_volume_1": None,
        "atm_iv_to_median_20": None,
        "put_skew": None,
    },
    "oddlot": {
        "oddlot_volume_share": "oddlot_volume_share",
        "oddlot_volume_share_change_5": "oddlot_volume_share_change_5",
    },
    "rebalance": {name: name for name in SIDECAR_FEATURES["rebalance"]},
    "fundamentals": {
        "log_market_cap": None,
        "book_to_market": None,
        "gross_profitability": None,
        "liabilities_to_assets": "liabilities_to_assets",
    },
}


@dataclass(frozen=True)
class SidecarResult:
    group: str
    feature_names: tuple[str, ...]
    values: NDArray[np.float32]
    valid: NDArray[np.bool_]
    age_sessions: NDArray[np.float32]
    coverage_by_year: tuple[dict[str, object], ...]
    archive_semantics_available: tuple[str, ...] = ()
    source_missing_candidates: tuple[str, ...] = ()
    publication_lag_reproduced: bool = False
    publication_lag_valid_cells: int = 0
    publication_lag_source_rows: int = 0
    d_plus_one_rows_checked: int = 0
    d_plus_one_violations: int = 0


def _as_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[D]").astype(object)
    return date.fromisoformat(str(value)[:10])


def _available_before_decision(
    row: Mapping[str, object],
    decision_date: date,
    decision_time: time,
    *,
    date_only_available_before_decision: bool,
) -> bool:
    effective = _effective_public_availability(row)
    if effective is not None:
        candidate = datetime.combine(
            decision_date, decision_time, tzinfo=B3_TIMEZONE
        )
        return effective <= candidate.astimezone(UTC)
    available_date = _as_date(row.get("available_date"))
    if available_date < decision_date:
        return True
    if available_date > decision_date:
        return False
    decision_idx = row.get("decision_idx")
    if decision_idx is None:
        # A same-date publication with no timestamp is not safely usable at the
        # decision.  Callers may opt in only when ``available_date`` is an
        # authoritative, already-lagged research availability date (the v1
        # archives apply their source-specific D+1 rules upstream).
        return date_only_available_before_decision
    source_time = datetime.combine(available_date, time(10, 15)) + timedelta(
        minutes=5 * int(decision_idx)
    )
    return source_time.time() <= decision_time


def _timestamp_column(row: Mapping[str, object]) -> str | None:
    selected = [column for column in _TIMESTAMP_COLUMNS if row.get(column) is not None]
    if len(selected) > 1:
        instants = {str(row[column]) for column in selected}
        if len(instants) > 1:
            raise ValueError(
                f"sidecar row has conflicting public timestamps: {selected}"
            )
    return selected[0] if selected else None


def _effective_public_availability(
    row: Mapping[str, object],
) -> datetime | None:
    """Return the conservative, latency-adjusted public instant in UTC."""

    column = _timestamp_column(row)
    if column is None:
        return None
    timestamp = row[column]
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if not isinstance(timestamp, datetime):
        raise ValueError(f"{column} has an unsupported timestamp type")
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=B3_TIMEZONE)
    precision = row.get(f"{column}_precision", row.get("timestamp_precision"))
    if precision is None:
        raise ValueError(f"{column} requires declared timestamp precision")
    precision_name = str(precision).strip().casefold()
    try:
        precision_delay = _PRECISION_UPPER_BOUND[precision_name]
    except KeyError as error:
        raise ValueError(f"unsupported timestamp precision: {precision}") from error
    latency = row.get(
        f"{column}_processing_latency_seconds",
        row.get("processing_latency_seconds"),
    )
    if latency is None:
        raise ValueError(f"{column} requires declared processing latency")
    latency_seconds = float(latency)
    if not np.isfinite(latency_seconds) or latency_seconds < 0.0:
        raise ValueError("processing latency must be finite and nonnegative")
    return (
        timestamp.astimezone(UTC)
        + precision_delay
        + timedelta(seconds=latency_seconds)
    )


def _availability_order(row: Mapping[str, object]) -> tuple[datetime, int]:
    timestamp = _effective_public_availability(row)
    if timestamp is not None:
        return timestamp, int(row.get("decision_idx") or -1)
    available_date = _as_date(row["available_date"])
    return datetime.combine(available_date, time.min, tzinfo=B3_TIMEZONE).astimezone(
        UTC
    ), int(row.get("decision_idx") if row.get("decision_idx") is not None else -1)


def _source_age_date(
    row: Mapping[str, object], group: str, feature: str
) -> date:
    if feature.startswith("sessions_since_"):
        return _as_date(row["available_date"])
    preferred: tuple[str, ...]
    if group == "lending" and feature.startswith("loan_balance"):
        preferred = ("source_position_date",)
    elif group in {"lending", "oddlot", "options"}:
        preferred = (
            "source_trade_date",
            "source_position_date",
            "snapshot_date",
            "reference_date",
        )
    elif group == "rebalance":
        preferred = ("announcement_date", "source_date")
    else:
        preferred = (
            "filing_receipt_date",
            "source_date",
            "reference_date",
        )
    for column in preferred:
        value = row.get(column)
        if value is not None:
            return _as_date(value)
    timestamp = _effective_public_availability(row)
    if timestamp is not None:
        return timestamp.astimezone(B3_TIMEZONE).date()
    return _as_date(row["available_date"])


def _exchange_session_age(
    calendar: NDArray[np.datetime64], decision_index: int, source_date: date
) -> float:
    source = np.datetime64(source_date, "D")
    insertion = int(np.searchsorted(calendar, source, side="left"))
    if insertion > decision_index:
        raise ValueError("sidecar source date is after its decision session")
    return float(decision_index - insertion)


def _decision_index_for_row(
    row: Mapping[str, object],
    normalized_dates: tuple[date, ...],
    decision_time: time,
    *,
    date_only_available_before_decision: bool,
    allow_pre_window_state: bool,
) -> int | None:
    effective = _effective_public_availability(row)
    if effective is not None:
        index = bisect_left(
            normalized_dates,
            effective,
            key=lambda day: datetime.combine(
                day, decision_time, tzinfo=B3_TIMEZONE
            ).astimezone(UTC),
        )
        return index if index < len(normalized_dates) else None
    available_date = _as_date(row["available_date"])
    index = bisect_left(normalized_dates, available_date)
    if index >= len(normalized_dates):
        return None
    if normalized_dates[index] != available_date:
        return 0 if index == 0 and allow_pre_window_state else None
    if not _available_before_decision(
        row,
        available_date,
        decision_time,
        date_only_available_before_decision=date_only_available_before_decision,
    ):
        return None
    return index


def _same_logical_record(
    left: Mapping[str, object],
    right: Mapping[str, object],
    columns: Sequence[str],
) -> bool:
    for column in columns:
        left_value = left.get(column)
        right_value = right.get(column)
        left_mask = left.get(f"{column}_mask", True)
        right_mask = right.get(f"{column}_mask", True)
        if bool(left_mask) != bool(right_mask):
            return False
        if left_value is None or right_value is None:
            if left_value is not None or right_value is not None:
                return False
            continue
        try:
            equal = bool(
                np.isclose(
                    float(left_value),
                    float(right_value),
                    rtol=0.0,
                    atol=0.0,
                    equal_nan=True,
                )
            )
        except (TypeError, ValueError):
            equal = left_value == right_value
        if not equal:
            return False
    return True


def materialize_sidecar(
    source: pl.DataFrame,
    dates: Sequence[date | np.datetime64],
    isins: Sequence[str],
    *,
    group: str,
    feature_columns: Mapping[str, str | None] | None = None,
    decision_time: time = time(15, 45),
    date_only_available_before_decision: bool = False,
) -> SidecarResult:
    """Exact no-fill date/ISIN join with decision-time availability enforcement."""

    if group not in SIDECAR_FEATURES:
        raise ValueError(f"unknown sidecar group: {group}")
    identity = "isin" if "isin" in source.columns else "security_id"
    if identity not in source.columns or "available_date" not in source.columns:
        raise ValueError("sidecar source needs identity and available_date")
    candidates = SIDECAR_FEATURES[group]
    columns = dict(
        feature_columns
        if feature_columns is not None
        else available_archive_mapping(group, source.columns)
    )
    unknown = set(columns) - set(candidates)
    if unknown:
        raise ValueError(f"unknown sidecar features for {group}: {sorted(unknown)}")
    names = tuple(
        name for name in candidates if columns.get(name) is not None
    )
    columns = {name: columns[name] for name in names}
    absent = sorted(
        column
        for column in set(columns.values()) - {None}
        if column not in source.columns
    )
    if absent:
        raise ValueError(f"sidecar source columns missing: {absent}")
    normalized_dates = tuple(_as_date(value) for value in dates)
    if any(
        left >= right
        for left, right in zip(normalized_dates, normalized_dates[1:], strict=False)
    ):
        raise ValueError("sidecar decision dates must be strictly increasing")
    isin_lookup = {value: index for index, value in enumerate(isins)}
    values = np.zeros((len(dates), len(isins), len(names)), dtype=np.float32)
    valid = np.zeros(values.shape, dtype=np.bool_)
    age_sessions = np.full(values.shape, -1.0, dtype=np.float32)
    chosen: dict[tuple[int, int, str], Mapping[str, object]] = {}
    stateful = group in {"events", "rebalance", "fundamentals"}
    for row in source.iter_rows(named=True):
        date_index = _decision_index_for_row(
            row,
            normalized_dates,
            decision_time,
            date_only_available_before_decision=date_only_available_before_decision,
            allow_pre_window_state=stateful,
        )
        isin_index = isin_lookup.get(row[identity])
        if date_index is None or isin_index is None:
            continue
        record_family = str(row.get("__record_family") or group)
        key = (date_index, isin_index, record_family)
        previous = chosen.get(key)
        provided_columns = tuple(
            column
            for column in columns.values()
            if column is not None
            and (
                f"__provided__{column}" not in source.columns
                or bool(row[f"__provided__{column}"])
            )
        )
        if not provided_columns:
            continue
        if previous is not None and _availability_order(previous) == _availability_order(row):
            if not _same_logical_record(previous, row, provided_columns):
                raise ValueError(
                    "sidecar has conflicting logical records at one publication "
                    f"coordinate: {group}/{record_family}"
                )
            continue
        if previous is None or _availability_order(row) > _availability_order(previous):
            chosen[key] = row
    updated = np.zeros(values.shape, dtype=np.bool_)
    calendar = np.asarray(normalized_dates, dtype="datetime64[D]")
    for (date_index, isin_index, record_family), row in chosen.items():
        for feature_index, name in enumerate(names):
            column = columns[name]
            if column is None:
                continue
            marker = f"__provided__{column}"
            if marker in source.columns and not bool(row[marker]):
                continue
            if updated[date_index, isin_index, feature_index]:
                raise ValueError(
                    f"multiple logical records provide {group}/{name} at one decision"
                )
            value = row[column]
            mask_column = f"{column}_mask"
            is_valid = value is not None and np.isfinite(float(value))
            if mask_column in source.columns:
                is_valid &= bool(row[mask_column])
            if is_valid:
                values[date_index, isin_index, feature_index] = float(value)
                valid[date_index, isin_index, feature_index] = True
            updated[date_index, isin_index, feature_index] = True
            age_sessions[date_index, isin_index, feature_index] = (
                float(value)
                if name.startswith("sessions_since_")
                and value is not None
                and np.isfinite(float(value))
                and float(value) >= 0.0
                else _exchange_session_age(
                    calendar,
                    date_index,
                    _source_age_date(row, group, name),
                )
            )
    if stateful:
        for date_index in range(1, len(normalized_dates)):
            carry = ~updated[date_index] & (age_sessions[date_index - 1] >= 0.0)
            values[date_index][carry] = values[date_index - 1][carry]
            valid[date_index][carry] = valid[date_index - 1][carry]
            age_sessions[date_index][carry] = (
                age_sessions[date_index - 1][carry] + 1.0
            )
    coverage: list[dict[str, object]] = []
    years = np.asarray([value.year for value in normalized_dates], dtype=np.int16)
    for year in sorted(set(years.tolist())):
        rows = years == year
        denominator = int(rows.sum()) * len(isins)
        for feature_index, name in enumerate(names):
            count = int(valid[rows, :, feature_index].sum())
            coverage.append(
                {
                    "group": group,
                    "year": int(year),
                    "feature": name,
                    "valid_count": count,
                    "possible_count": denominator,
                    "coverage": count / denominator if denominator else 0.0,
                }
            )
    return SidecarResult(
        group=group,
        feature_names=names,
        values=values,
        valid=valid,
        age_sessions=age_sessions,
        coverage_by_year=tuple(coverage),
        archive_semantics_available=tuple(
            name for name in names if columns[name] is not None
        ),
        source_missing_candidates=tuple(
            name for name in candidates if name not in names
        ),
    )


def rebuild_publication_lag_validity(
    source: pl.DataFrame,
    dates: Sequence[date | np.datetime64],
    isins: Sequence[str],
    *,
    group: str,
    feature_columns: Mapping[str, str | None] | None = None,
    decision_time: time = time(15, 45),
    date_only_available_before_decision: bool = False,
) -> NDArray[np.bool_]:
    """Independently rebuild validity from publication-lagged archive rows.

    This intentionally does not consume a materialized ``SidecarResult``.  It
    replays the raw archive's availability coordinate, latest-known snapshot,
    finite-value rule, and archived source masks so the store can assert that
    no survival- or end-of-panel condition entered external feature validity.
    """

    if group not in SIDECAR_FEATURES:
        raise ValueError(f"unknown sidecar group: {group}")
    identity = "isin" if "isin" in source.columns else "security_id"
    if identity not in source.columns or "available_date" not in source.columns:
        raise ValueError("sidecar source needs identity and available_date")
    candidates = SIDECAR_FEATURES[group]
    columns = dict(
        feature_columns
        if feature_columns is not None
        else available_archive_mapping(group, source.columns)
    )
    unknown = set(columns) - set(candidates)
    if unknown:
        raise ValueError(f"unknown sidecar features for {group}: {sorted(unknown)}")
    names = tuple(name for name in candidates if columns.get(name) is not None)
    columns = {name: columns[name] for name in names}
    normalized_dates = tuple(_as_date(value) for value in dates)
    isin_lookup = {str(value): index for index, value in enumerate(isins)}
    stateful = group in {"events", "rebalance", "fundamentals"}
    selected: dict[tuple[int, int, str], Mapping[str, object]] = {}
    for row in source.iter_rows(named=True):
        date_index = _decision_index_for_row(
            row,
            normalized_dates,
            decision_time,
            date_only_available_before_decision=date_only_available_before_decision,
            allow_pre_window_state=stateful,
        )
        isin_index = isin_lookup.get(str(row[identity]))
        if date_index is None or isin_index is None:
            continue
        record_family = str(row.get("__record_family") or group)
        key = (int(date_index), int(isin_index), record_family)
        provided_columns = tuple(
            column
            for column in columns.values()
            if column is not None
            and (
                f"__provided__{column}" not in source.columns
                or bool(row[f"__provided__{column}"])
            )
        )
        if not provided_columns:
            continue
        previous = selected.get(key)
        if previous is not None and _availability_order(previous) == _availability_order(row):
            if not _same_logical_record(previous, row, provided_columns):
                raise ValueError(
                    "sidecar has conflicting logical records at one publication "
                    f"coordinate: {group}/{record_family}"
                )
            continue
        if previous is None or _availability_order(row) > _availability_order(previous):
            selected[key] = row
    rebuilt = np.zeros((len(dates), len(isins), len(names)), dtype=np.bool_)
    updated = np.zeros_like(rebuilt)
    source_columns = set(source.columns)
    for (date_index, isin_index, _record_family), row in selected.items():
        for feature_index, name in enumerate(names):
            column = columns[name]
            if column is None:
                continue
            marker = f"__provided__{column}"
            if marker in source_columns and not bool(row[marker]):
                continue
            if updated[date_index, isin_index, feature_index]:
                raise ValueError(
                    f"multiple logical records provide {group}/{name} at one decision"
                )
            value = row[column]
            valid = value is not None and np.isfinite(float(value))
            mask_column = f"{column}_mask"
            if mask_column in source_columns:
                valid &= bool(row[mask_column])
            rebuilt[date_index, isin_index, feature_index] = valid
            updated[date_index, isin_index, feature_index] = True
    if stateful:
        known = updated[0].copy() if len(dates) else np.zeros(rebuilt.shape[1:])
        for date_index in range(1, len(dates)):
            carry = ~updated[date_index] & known
            rebuilt[date_index][carry] = rebuilt[date_index - 1][carry]
            known |= updated[date_index]
    return rebuilt


def _timestamp_upper_expression(source: pl.DataFrame, column: str) -> pl.Expr:
    """Conservative upper precision bound plus declared processing latency."""

    dtype = source.schema[column]
    if not isinstance(dtype, pl.Datetime):
        raise ValueError(f"{column} must have a Datetime dtype")
    precision_column = (
        f"{column}_precision"
        if f"{column}_precision" in source.columns
        else "timestamp_precision"
    )
    latency_column = (
        f"{column}_processing_latency_seconds"
        if f"{column}_processing_latency_seconds" in source.columns
        else "processing_latency_seconds"
    )
    present = source.get_column(column).is_not_null()
    if present.any() and precision_column not in source.columns:
        raise ValueError(f"{column} requires declared timestamp precision")
    if present.any() and latency_column not in source.columns:
        raise ValueError(f"{column} requires declared processing latency")
    if not present.any():
        return pl.lit(None, dtype=pl.Int64)
    precision = pl.col(precision_column).cast(pl.String).str.to_lowercase()
    invalid_precision = source.filter(
        pl.col(column).is_not_null()
        & ~precision.is_in(tuple(_PRECISION_UPPER_BOUND))
    )
    if not invalid_precision.is_empty():
        raise ValueError(f"{column} has missing or unsupported timestamp precision")
    latency = pl.col(latency_column).cast(pl.Float64)
    invalid_latency = source.filter(
        pl.col(column).is_not_null()
        & (~latency.is_finite() | (latency < 0.0))
    )
    if not invalid_latency.is_empty():
        raise ValueError(f"{column} has invalid declared processing latency")
    value = pl.col(column)
    if dtype.time_zone is None:
        value = value.dt.replace_time_zone(str(B3_TIMEZONE))
    value_us = value.dt.convert_time_zone("UTC").dt.epoch("us")
    resolution_us = (
        pl.when(precision.is_in(("instant", "exact")))
        .then(0)
        .when(precision == "microsecond")
        .then(1)
        .when(precision == "millisecond")
        .then(1_000)
        .when(precision == "second")
        .then(1_000_000)
        .when(precision == "minute")
        .then(60_000_000)
        .otherwise(None)
    )
    return (value_us + resolution_us + latency * 1_000_000).cast(pl.Int64)


def _publication_order_expression(source: pl.DataFrame) -> pl.Expr:
    """Return each row's publication instant as UTC microseconds.

    Daily archives usually expose only ``available_date`` (and occasionally a
    v1 ``decision_idx``).  Timestamped archives may use either a timezone-aware
    instant or the historical naive B3-local convention.  One integer ordering
    lets the as-of joins below remain deterministic across input order/chunks.
    """

    timestamp_expressions: list[pl.Expr] = []
    for column in _TIMESTAMP_COLUMNS:
        if column not in source.columns:
            continue
        timestamp_expressions.append(_timestamp_upper_expression(source, column))

    available_midnight = (
        pl.col("available_date")
        .cast(pl.Date)
        .cast(pl.Datetime("us"))
        .dt.replace_time_zone(str(B3_TIMEZONE))
        .dt.convert_time_zone("UTC")
        .dt.epoch("us")
    )
    if "decision_idx" in source.columns:
        # v1 intraday coordinates start at 10:15 and advance in five-minute
        # increments. Null retains the date-only midnight ordering.
        intraday_offset = (
            (pl.lit(10 * 60 + 15) + 5 * pl.col("decision_idx").cast(pl.Int64))
            * 60
            * 1_000_000
        )
        date_order = pl.when(pl.col("decision_idx").is_not_null()).then(
            available_midnight + intraday_offset
        ).otherwise(available_midnight)
    else:
        date_order = available_midnight
    return pl.coalesce(*timestamp_expressions, date_order).cast(pl.Int64)


def _canonical_publication_frame(source: pl.DataFrame) -> pl.DataFrame:
    """Deduplicate exact rows and impose a publication-stable row order."""

    output = source.unique(maintain_order=False).with_columns(
        _publication_order_expression(source).alias(_PUBLICATION_ORDER_COLUMN)
    )
    sort_columns = ["isin", _PUBLICATION_ORDER_COLUMN, "available_date"]
    sort_columns.extend(
        column
        for column in ("source_position_date", "source_trade_date", "decision_idx")
        if column in output.columns
    )
    return output.sort(sort_columns, nulls_last=True).with_row_index(_ROW_INDEX_COLUMN)


def _reject_ambiguous_vintages(
    source: pl.DataFrame, source_date_column: str
) -> None:
    """Reject conflicting records at one exact publication coordinate."""

    relevant = source.filter(pl.col(source_date_column).is_not_null())
    if relevant.is_empty():
        return
    keys = ["isin", source_date_column, _PUBLICATION_ORDER_COLUMN]
    ambiguous = relevant.group_by(keys).len().filter(pl.col("len") > 1)
    if not ambiguous.is_empty():
        preview = ambiguous.head(10).select(keys).to_dicts()
        raise ValueError(
            f"ambiguous {source_date_column} revisions at one publication "
            f"coordinate: {preview}"
        )


def _vintage_lag_values(
    source: pl.DataFrame,
    calendar: tuple[date, ...],
    *,
    source_date_column: str,
    value_column: str,
    valid_column: str,
    lag: int,
) -> pl.DataFrame:
    """Select an exact-session lag at the vintage known by each source row."""

    prior_column = f"__prior_source_date_{lag}"
    prior_value = f"__prior_{value_column}_{lag}"
    prior_valid = f"__prior_{valid_column}_{lag}"
    prior_dates = [
        calendar[index - lag] if index >= lag else None
        for index in range(len(calendar))
    ]
    calendar_lookup = pl.DataFrame(
        {
            source_date_column: calendar,
            prior_column: prior_dates,
        },
        schema_overrides={source_date_column: pl.Date, prior_column: pl.Date},
    )
    left = source.join(calendar_lookup, on=source_date_column, how="left")
    right = source.select(
        "isin",
        pl.col(source_date_column).alias("__candidate_source_date"),
        pl.col(_PUBLICATION_ORDER_COLUMN).alias("__candidate_publication_order"),
        pl.col(value_column).alias(prior_value),
        pl.col(valid_column).alias(prior_valid),
    )
    selected = left.sort(_PUBLICATION_ORDER_COLUMN).join_asof(
        right.sort("__candidate_publication_order"),
        left_on=_PUBLICATION_ORDER_COLUMN,
        right_on="__candidate_publication_order",
        by_left=["isin", prior_column],
        by_right=["isin", "__candidate_source_date"],
        strategy="backward",
        check_sortedness=False,
    )
    return selected.select(
        _ROW_INDEX_COLUMN,
        pl.col(prior_value).fill_null(0.0),
        pl.col(prior_valid).fill_null(False),
    )


def _raw_lending_features(
    source: pl.DataFrame,
    dates: Sequence[date | np.datetime64],
    isins: Sequence[str],
    daily_volume_brl: NDArray[np.floating],
) -> pl.DataFrame:
    """Derive exact v2 lending fields from raw, D+1-available quantities."""

    required = {"available_date", "isin"}
    if not required.issubset(source.columns):
        return source
    has_balance = {
        "source_position_date",
        "lending_balance_brl",
    }.issubset(source.columns)
    raw_rate_columns = tuple(
        column
        for column in (
            "annual_taker_rate",
            "loan_rate_annual_decimal",
            "lending_taker_fee_annual_decimal",
        )
        if column in source.columns
    )
    if len(raw_rate_columns) > 1:
        raise ValueError("lending archive has multiple raw annual-rate columns")
    has_rate = "source_trade_date" in source.columns and bool(raw_rate_columns)
    if not has_balance and not has_rate:
        return source
    calendar = tuple(_as_date(value) for value in dates)
    volume = np.asarray(daily_volume_brl, dtype=np.float64)
    if volume.shape != (len(calendar), len(isins)):
        raise ValueError("daily BRL volume is misaligned with lending axes")

    balance_derived_columns = {
        "loan_balance_to_volume_20",
        "loan_balance_to_volume_20_mask",
        "loan_balance_change_1",
        "loan_balance_change_1_mask",
        "loan_balance_change_5",
        "loan_balance_change_5_mask",
    }
    rate_derived_columns = {
        "loan_rate",
        "loan_rate_mask",
        "loan_rate_change_5",
        "loan_rate_change_5_mask",
    }
    derived_columns = (
        (balance_derived_columns if has_balance else set())
        | (rate_derived_columns if has_rate else set())
    )
    output = _canonical_publication_frame(
        source.drop(
            [column for column in derived_columns if column in source.columns]
        )
    )
    calendar_positions = pl.DataFrame(
        {
            "__calendar_date": calendar,
            "__day_index": np.arange(len(calendar), dtype=np.int32),
        },
        schema_overrides={"__calendar_date": pl.Date},
    )
    name_positions = pl.DataFrame(
        {
            "isin": list(isins),
            "__name_index": np.arange(len(isins), dtype=np.int32),
        }
    )

    if has_balance:
        _reject_ambiguous_vintages(output, "source_position_date")
        balance = (
            output.filter(pl.col("source_position_date").is_not_null())
            .join(
                calendar_positions,
                left_on="source_position_date",
                right_on="__calendar_date",
                how="left",
            )
            .join(name_positions, on="isin", how="left")
        )
        rolling_volume = np.zeros_like(volume, dtype=np.float64)
        rolling_support = np.zeros(volume.shape, dtype=np.int16)
        rolling_negative = np.zeros(volume.shape, dtype=np.bool_)
        if len(calendar) >= 20:
            rolling_sum = np.zeros(len(isins), dtype=np.float64)
            support_count = np.zeros(len(isins), dtype=np.int16)
            negative_count = np.zeros(len(isins), dtype=np.int16)
            for day in range(len(calendar)):
                current = volume[day]
                current_finite = np.isfinite(current)
                rolling_sum += np.where(current_finite, current, 0.0)
                support_count += current_finite.astype(np.int16)
                negative_count += (current_finite & (current < 0.0)).astype(
                    np.int16
                )
                if day >= 20:
                    expired = volume[day - 20]
                    expired_finite = np.isfinite(expired)
                    rolling_sum -= np.where(expired_finite, expired, 0.0)
                    support_count -= expired_finite.astype(np.int16)
                    negative_count -= (
                        expired_finite & (expired < 0.0)
                    ).astype(np.int16)
                if day >= 19:
                    rolling_volume[day] = rolling_sum / 20.0
                    rolling_support[day] = support_count
                    rolling_negative[day] = negative_count > 0

        day_index = (
            balance.get_column("__day_index").fill_null(-1).to_numpy().astype(np.int64)
        )
        name_index = (
            balance.get_column("__name_index")
            .fill_null(-1)
            .to_numpy()
            .astype(np.int64)
        )
        safe_day = np.clip(day_index, 0, max(len(calendar) - 1, 0))
        safe_name = np.clip(name_index, 0, max(len(isins) - 1, 0))
        mean_volume = rolling_volume[safe_day, safe_name]
        support = rolling_support[safe_day, safe_name]
        has_negative = rolling_negative[safe_day, safe_name]
        balance_value = (
            balance.get_column("lending_balance_brl")
            .cast(pl.Float64)
            .fill_null(np.nan)
            .to_numpy()
        )
        level_valid = (
            (day_index >= 19)
            & (name_index >= 0)
            & (support == 20)
            & ~has_negative
            & np.isfinite(balance_value)
            & (balance_value >= 0.0)
            & np.isfinite(mean_volume)
            & (mean_volume > 0.0)
        )
        level = np.zeros(balance.height, dtype=np.float64)
        level[level_valid] = balance_value[level_valid] / mean_volume[level_valid]
        balance = balance.with_columns(
            pl.Series("loan_balance_to_volume_20", level),
            pl.Series("loan_balance_to_volume_20_mask", level_valid),
        )
        for lag in (1, 5):
            prior = _vintage_lag_values(
                balance,
                calendar,
                source_date_column="source_position_date",
                value_column="loan_balance_to_volume_20",
                valid_column="loan_balance_to_volume_20_mask",
                lag=lag,
            )
            balance = balance.join(prior, on=_ROW_INDEX_COLUMN, how="left")
            feature = f"loan_balance_change_{lag}"
            prior_value = f"__prior_loan_balance_to_volume_20_{lag}"
            prior_valid = f"__prior_loan_balance_to_volume_20_mask_{lag}"
            valid = pl.col("loan_balance_to_volume_20_mask") & pl.col(prior_valid)
            balance = balance.with_columns(
                pl.when(valid)
                .then(pl.col("loan_balance_to_volume_20") - pl.col(prior_value))
                .otherwise(0.0)
                .alias(feature),
                valid.alias(f"{feature}_mask"),
            ).drop(prior_value, prior_valid)
        output = output.join(
            balance.select(
                _ROW_INDEX_COLUMN,
                "loan_balance_to_volume_20",
                "loan_balance_to_volume_20_mask",
                "loan_balance_change_1",
                "loan_balance_change_1_mask",
                "loan_balance_change_5",
                "loan_balance_change_5_mask",
            ),
            on=_ROW_INDEX_COLUMN,
            how="left",
        )

    if has_rate:
        _reject_ambiguous_vintages(output, "source_trade_date")
        rate = output.filter(pl.col("source_trade_date").is_not_null())
        raw_column = raw_rate_columns[0]
        raw_rate = (
            rate.get_column(raw_column)
            .cast(pl.Float64)
            .fill_null(np.nan)
            .to_numpy()
            .copy()
        )
        raw_mask_column = f"{raw_column}_mask"
        if raw_mask_column in rate.columns:
            rate_mask = (
                rate.get_column(raw_mask_column)
                .fill_null(False)
                .to_numpy()
            )
        else:
            rate_mask = np.ones(rate.height, dtype=np.bool_)
        rate_valid = rate_mask & np.isfinite(raw_rate) & (raw_rate >= 0.0)
        raw_rate[~rate_valid] = 0.0
        rate = rate.with_columns(
            pl.Series("loan_rate", raw_rate),
            pl.Series("loan_rate_mask", rate_valid),
        )
        prior = _vintage_lag_values(
            rate,
            calendar,
            source_date_column="source_trade_date",
            value_column="loan_rate",
            valid_column="loan_rate_mask",
            lag=5,
        )
        rate = rate.join(prior, on=_ROW_INDEX_COLUMN, how="left")
        valid = pl.col("loan_rate_mask") & pl.col("__prior_loan_rate_mask_5")
        rate = rate.with_columns(
            pl.when(valid)
            .then(pl.col("loan_rate") - pl.col("__prior_loan_rate_5"))
            .otherwise(0.0)
            .alias("loan_rate_change_5"),
            valid.alias("loan_rate_change_5_mask"),
        )
        output = output.join(
            rate.select(
                _ROW_INDEX_COLUMN,
                "loan_rate",
                "loan_rate_mask",
                "loan_rate_change_5",
                "loan_rate_change_5_mask",
            ),
            on=_ROW_INDEX_COLUMN,
            how="left",
        )

    value_columns = sorted(
        column for column in derived_columns if not column.endswith("_mask")
    )
    mask_columns = sorted(column for column in derived_columns if column.endswith("_mask"))
    for column in value_columns:
        if column not in output.columns:
            output = output.with_columns(pl.lit(0.0).alias(column))
    for column in mask_columns:
        if column not in output.columns:
            output = output.with_columns(pl.lit(False).alias(column))
    return (
        output.with_columns(
            *(pl.col(column).fill_null(0.0).cast(pl.Float64) for column in value_columns),
            *(pl.col(column).fill_null(False).cast(pl.Boolean) for column in mask_columns),
        )
        .sort(_ROW_INDEX_COLUMN)
        .drop(_ROW_INDEX_COLUMN, _PUBLICATION_ORDER_COLUMN)
    )


def _raw_options_features(source: pl.DataFrame) -> pl.DataFrame:
    """Derive the canonical put/call signal only from complete raw OI counts."""

    count_pairs = (
        ("put_open_interest", "call_open_interest"),
        ("put_oi", "call_oi"),
    )
    pairs = tuple(pair for pair in count_pairs if set(pair).issubset(source.columns))
    if not pairs:
        return source
    if len(pairs) > 1:
        raise ValueError("options archive has multiple raw put/call OI column pairs")
    completeness_columns = tuple(
        column
        for column in ("oi_snapshot_complete", "open_interest_snapshot_complete")
        if column in source.columns
    )
    if len(completeness_columns) != 1:
        # Absence is not evidence that a missing side count means a known zero.
        return source
    put_column, call_column = pairs[0]
    put = source.get_column(put_column).cast(pl.Float64).fill_null(np.nan).to_numpy()
    call = source.get_column(call_column).cast(pl.Float64).fill_null(np.nan).to_numpy()
    complete = (
        source.get_column(completeness_columns[0])
        .cast(pl.Boolean)
        .fill_null(False)
        .to_numpy()
    )
    valid = (
        complete
        & np.isfinite(put)
        & np.isfinite(call)
        & (put >= 0.0)
        & (call >= 0.0)
        & (np.floor(put) == put)
        & (np.floor(call) == call)
        & ((put + call) > 0.0)
    )
    ratio = np.zeros(source.height, dtype=np.float64)
    ratio[valid] = np.log((put[valid] + 1.0) / (call[valid] + 1.0))
    valid &= np.isfinite(ratio)
    ratio[~valid] = 0.0
    return source.with_columns(
        pl.Series("put_call_log_oi_ratio", ratio),
        pl.Series("put_call_log_oi_ratio_mask", valid),
    )


def _raw_events_features(
    source: pl.DataFrame, dates: Sequence[date | np.datetime64]
) -> pl.DataFrame:
    """Derive exact filing age from timestamped financial-filing events."""

    timestamp_columns = (
        ("filing_receipt_timestamp",)
        if "filing_receipt_timestamp" in source.columns
        else (
            ("available_timestamp",)
            if {"available_timestamp", "event_type"}.issubset(source.columns)
            else ()
        )
    )
    if "isin" not in source.columns or not timestamp_columns:
        return source
    if len(timestamp_columns) > 1:
        raise ValueError("event archive has multiple filing timestamp columns")
    timestamp_column = timestamp_columns[0]
    dtype = source.schema[timestamp_column]
    if not isinstance(dtype, pl.Datetime):
        raise ValueError(f"{timestamp_column} must have a Datetime dtype")

    events = source.filter(pl.col(timestamp_column).is_not_null())
    if timestamp_column == "available_timestamp":
        events = events.filter(
            pl.col("event_type")
            .cast(pl.String)
            .str.to_uppercase()
            .is_in(["ITR", "DFP", "FINANCIAL_FILING"])
        )
    publication_us = _timestamp_upper_expression(events, timestamp_column)
    events = events.with_columns(publication_us.alias("__filing_publication_us"))

    calendar = tuple(_as_date(value) for value in dates)
    decision_instants = pl.DataFrame(
        {
            "available_date": calendar,
            "__session_position": np.arange(len(calendar), dtype=np.int32),
        },
        schema_overrides={"available_date": pl.Date},
    ).with_columns(
        pl.col("available_date")
        .cast(pl.Datetime("us"))
        .dt.replace_time_zone(str(B3_TIMEZONE))
        .dt.offset_by("15h45m")
        .dt.convert_time_zone("UTC")
        .dt.epoch("us")
        .alias("__decision_us")
    )
    effective = (
        events.sort("__filing_publication_us")
        .join_asof(
            decision_instants.sort("__decision_us"),
            left_on="__filing_publication_us",
            right_on="__decision_us",
            strategy="forward",
        )
        .filter(pl.col("__session_position").is_not_null())
        .sort("isin", "__session_position", "__filing_publication_us")
        .group_by("isin", "__session_position", maintain_order=True)
        .agg(
            pl.col("__filing_publication_us").last(),
        )
        .rename({"__session_position": "__event_session_position"})
    )
    if effective.is_empty():
        return pl.DataFrame(schema={"available_date": pl.Date, "isin": pl.String})
    names = effective.select("isin").unique()
    states = (
        names.join(decision_instants, how="cross")
        .sort("isin", "__session_position")
        .join_asof(
            effective.sort("isin", "__event_session_position"),
            left_on="__session_position",
            right_on="__event_session_position",
            by="isin",
            strategy="backward",
            check_sortedness=False,
        )
    )
    valid = pl.col("__filing_publication_us").is_not_null()
    return states.select(
        "available_date",
        "isin",
        pl.col("__decision_us")
        .cast(pl.Datetime("us", time_zone="UTC"))
        .alias("public_available_at"),
        pl.lit("exact").alias("public_available_at_precision"),
        pl.lit(0.0).alias("public_available_at_processing_latency_seconds"),
        pl.col("__filing_publication_us")
        .cast(pl.Datetime("us", time_zone="UTC"))
        .alias("event_time"),
        pl.when(valid)
        .then(
            pl.col("__session_position")
            - pl.col("__event_session_position")
        )
        .otherwise(0.0)
        .cast(pl.Float64)
        .alias("sessions_since_financial_filing"),
        valid.alias("sessions_since_financial_filing_mask"),
    )


def _raw_fundamental_features(source: pl.DataFrame) -> pl.DataFrame:
    """Derive liabilities/assets from compatible raw values on the same filing row."""

    pairs = tuple(
        pair
        for pair in (
            ("total_liabilities_brl", "total_assets_brl"),
            ("liabilities_brl", "assets_brl"),
        )
        if set(pair).issubset(source.columns)
    )
    if not pairs:
        return source
    if len(pairs) > 1:
        raise ValueError("fundamental archive has multiple raw liability/asset pairs")
    liabilities_column, assets_column = pairs[0]
    liabilities = (
        source.get_column(liabilities_column)
        .cast(pl.Float64)
        .fill_null(np.nan)
        .to_numpy()
    )
    assets = (
        source.get_column(assets_column)
        .cast(pl.Float64)
        .fill_null(np.nan)
        .to_numpy()
    )
    valid = np.isfinite(liabilities) & np.isfinite(assets) & (assets > 0.0)
    for column in (liabilities_column, assets_column):
        mask_column = f"{column}_mask"
        if mask_column in source.columns:
            valid &= source.get_column(mask_column).fill_null(False).to_numpy()
    ratio = np.zeros(source.height, dtype=np.float64)
    ratio[valid] = liabilities[valid] / assets[valid]
    valid &= np.isfinite(ratio)
    ratio[~valid] = 0.0
    return source.with_columns(
        pl.Series("liabilities_to_assets", ratio),
        pl.Series("liabilities_to_assets_mask", valid),
    )


def _raw_oddlot_features(
    source: pl.DataFrame,
    dates: Sequence[date | np.datetime64],
) -> pl.DataFrame:
    """Derive the untransformed odd-lot BRL-volume share and exact lag-5 change."""

    required = {
        "available_date",
        "isin",
        "source_trade_date",
        "regular_volume_brl",
        "odd_lot_volume_brl",
    }
    if not required.issubset(source.columns):
        return pl.DataFrame(schema={"available_date": pl.Date, "isin": pl.String})
    calendar = tuple(_as_date(value) for value in dates)
    output = _canonical_publication_frame(source)
    _reject_ambiguous_vintages(output, "source_trade_date")
    calendar_dates = pl.DataFrame(
        {"source_trade_date": calendar},
        schema_overrides={"source_trade_date": pl.Date},
    ).with_columns(pl.lit(True).alias("__is_session"))
    output = output.join(calendar_dates, on="source_trade_date", how="left")
    regular = pl.col("regular_volume_brl").cast(pl.Float64)
    odd = pl.col("odd_lot_volume_brl").cast(pl.Float64)
    total = regular + odd
    valid = (
        pl.col("__is_session").fill_null(False)
        & regular.is_finite()
        & odd.is_finite()
        & (regular >= 0.0)
        & (odd >= 0.0)
        & (total > 0.0)
    ).fill_null(False)
    output = output.with_columns(
        pl.when(valid)
        .then(odd / total)
        .otherwise(0.0)
        .alias("oddlot_volume_share"),
        valid.alias("oddlot_volume_share_mask"),
    )
    prior = _vintage_lag_values(
        output,
        calendar,
        source_date_column="source_trade_date",
        value_column="oddlot_volume_share",
        valid_column="oddlot_volume_share_mask",
        lag=5,
    )
    output = output.join(prior, on=_ROW_INDEX_COLUMN, how="left")
    change_valid = pl.col("oddlot_volume_share_mask") & pl.col(
        "__prior_oddlot_volume_share_mask_5"
    )
    output_columns = ["available_date", "isin", "source_trade_date"]
    output_columns.extend(
        column
        for column in ("decision_idx", "available_timestamp", "delivery_timestamp")
        if column in output.columns
    )
    output_columns.extend(
        (
            "oddlot_volume_share",
            "oddlot_volume_share_mask",
            "oddlot_volume_share_change_5",
            "oddlot_volume_share_change_5_mask",
        )
    )
    return (
        output.with_columns(
            pl.when(change_valid)
            .then(
                pl.col("oddlot_volume_share")
                - pl.col("__prior_oddlot_volume_share_5")
            )
            .otherwise(0.0)
            .alias("oddlot_volume_share_change_5"),
            change_valid.alias("oddlot_volume_share_change_5_mask"),
        )
        .sort(_ROW_INDEX_COLUMN)
        .select(output_columns)
    )


def derive_known_archive_features(
    source: pl.DataFrame,
    dates: Sequence[date | np.datetime64],
    isins: Sequence[str],
    *,
    group: str,
    daily_volume_brl: NDArray[np.floating] | None = None,
) -> pl.DataFrame:
    """Expose only fields whose archive semantics exactly match the v2 contract."""

    if group == "lending":
        if daily_volume_brl is None:
            raise ValueError("lending sidecar needs aligned daily BRL volume")
        return _raw_lending_features(source, dates, isins, daily_volume_brl)
    if group == "oddlot":
        return _raw_oddlot_features(source, dates)
    if group == "options":
        return _raw_options_features(source)
    if group == "events":
        # A legacy decision-grid age is not an exact event identity/timestamp.
        # Rebuild the current 15:45 state only from genuine public records.
        derived = _raw_events_features(source, dates)
        if derived is source:
            return pl.DataFrame(schema={"available_date": pl.Date, "isin": pl.String})
        return derived
    if group == "fundamentals":
        return _raw_fundamental_features(source)
    if group == "rebalance":
        # Experiment-33's 55-coordinate state ends at 14:45 and cannot be
        # promoted to a 15:45 snapshot.  Timestamped current-state records are
        # accepted; unsupported legacy grids are disabled rather than delayed.
        if not any(column in source.columns for column in _TIMESTAMP_COLUMNS):
            return pl.DataFrame(schema={"available_date": pl.Date, "isin": pl.String})
        return source
    return source


def bind_sidecar_isins(source: pl.DataFrame, assignments: pl.DataFrame) -> pl.DataFrame:
    """Replace a v1 ``security_id`` key with its audited one-to-one ISIN."""

    if "isin" in source.columns:
        return source
    if "security_id" not in source.columns:
        raise ValueError("sidecar has neither ISIN nor security_id")
    if not {"security_id", "isin"}.issubset(assignments.columns):
        raise ValueError("assignments need security_id and ISIN")
    mapping = assignments.select("security_id", "isin").unique()
    if mapping.get_column("security_id").n_unique() != mapping.height:
        raise ValueError("sidecar assignment maps one security_id to multiple ISINs")
    bound = source.join(mapping, on="security_id", how="left", validate="m:1")
    if bound.get_column("isin").null_count():
        raise ValueError("sidecar contains unmapped security_id values")
    return bound.drop("security_id")


def available_archive_mapping(
    group: str, columns: Sequence[str]
) -> dict[str, str | None]:
    """Map the ordered candidate subset supported by exact source semantics."""

    if group not in ARCHIVE_COLUMN_MAP:
        raise ValueError(f"unknown sidecar group: {group}")
    available = set(columns)
    return {
        feature: feature if feature in available else source
        for feature, source in ARCHIVE_COLUMN_MAP[group].items()
        if feature in available or (source is not None and source in available)
    }


def materialize_known_archive(
    source: pl.DataFrame,
    dates: Sequence[date | np.datetime64],
    isins: Sequence[str],
    *,
    group: str,
    decision_time: time = time(15, 45),
) -> SidecarResult:
    """Materialize only fields whose source semantics match the canonical contract.

    The v1 archive ``available_date`` is already source-lagged (including D+1
    where required), so a matching date is authoritative before the decision.
    Missing candidates are reported in ``source_missing_candidates`` and do not
    consume enabled feature-array columns.
    """

    return materialize_sidecar(
        source,
        dates,
        isins,
        group=group,
        feature_columns=available_archive_mapping(group, source.columns),
        decision_time=decision_time,
        date_only_available_before_decision=True,
    )


def validate_sidecar(
    result: SidecarResult, active: NDArray[np.bool_] | None = None
) -> None:
    if result.values.shape != result.valid.shape or result.values.ndim != 3:
        raise ValueError("sidecar arrays must be aligned [date, name, feature]")
    if result.age_sessions.shape != result.values.shape:
        raise ValueError("sidecar source ages must align with feature arrays")
    if result.values.shape[2] != len(result.feature_names):
        raise ValueError("sidecar feature axis is misaligned")
    if not np.isfinite(result.values).all() or np.any(
        result.values[~result.valid] != 0
    ):
        raise ValueError("sidecar values must be finite and invalid cells exactly zero")
    if (
        not np.isfinite(result.age_sessions).all()
        or np.any(result.valid & (result.age_sessions < 0.0))
        or np.any(result.age_sessions < -1.0)
    ):
        raise ValueError(
            "sidecar source ages must be finite last-observation ages or -1, and "
            "every valid feature must have a known age"
        )
    if active is not None:
        membership = np.asarray(active, dtype=np.bool_)
        if membership.shape != result.values.shape[:2]:
            raise ValueError("sidecar membership mask is misaligned")
