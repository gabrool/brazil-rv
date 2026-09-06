from __future__ import annotations

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


ARCHIVE_COLUMN_MAP: dict[str, dict[str, str | None]] = {
    "lending": {
        # These three columns are derived below from the raw BRL balance and
        # raw COTAHIST BRL volume.  The v1 tanh/log transforms are not relabelled
        # as raw v2 quantities.
        "loan_balance_to_volume_20": "loan_balance_to_volume_20",
        "loan_balance_change_1": "loan_balance_change_1",
        "loan_balance_change_5": "loan_balance_change_5",
        # The archived transforms are one-to-one.  They are inverted below
        # before the canonical v2 fields are rank-normalized.
        "loan_rate": "lending_taker_fee_level_log_tanh",
        "loan_rate_change_5": "lending_taker_fee_change_5_tanh",
    },
    "events": {
        "sessions_until_announced_earnings": None,
        "sessions_since_earnings": None,
        "standardized_unexpected_earnings": None,
    },
    "options": {
        "put_call_oi_ratio": "options_put_call_oi_log_ratio_tanh",
        # This is the archived one-session OI change divided by trailing
        # stock ADV20, behind a reversible signed-log/tanh transform.
        "delta_oi_to_volume_1": "options_oi_change_to_stock_adv20_tanh",
        "atm_iv_to_median_20": None,
        "put_skew": "options_put_skew_tanh",
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
        "leverage": "fund_leverage",
    },
}


@dataclass(frozen=True)
class SidecarResult:
    group: str
    feature_names: tuple[str, ...]
    values: NDArray[np.float32]
    valid: NDArray[np.bool_]
    coverage_by_year: tuple[dict[str, object], ...]
    archive_semantics_available: tuple[str, ...] = ()
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
    timestamp = row.get("available_timestamp") or row.get("delivery_timestamp")
    if timestamp is not None:
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if not isinstance(timestamp, datetime):
            raise ValueError("availability timestamp has an unsupported type")
        # Source archives historically stored naive CVM timestamps in B3 local
        # time.  Aware timestamps may use any honest zone/offset; compare both
        # instants in UTC instead of attaching the source offset to 15:45.
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=B3_TIMEZONE)
        candidate = datetime.combine(
            decision_date, decision_time, tzinfo=B3_TIMEZONE
        )
        return timestamp.astimezone(UTC) <= candidate.astimezone(UTC)
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


def _availability_order(row: Mapping[str, object]) -> tuple[datetime, int]:
    timestamp = row.get("available_timestamp") or row.get("delivery_timestamp")
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if isinstance(timestamp, datetime):
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=B3_TIMEZONE)
        return timestamp.astimezone(UTC), int(row.get("decision_idx") or -1)
    available_date = _as_date(row["available_date"])
    return datetime.combine(available_date, time.min, tzinfo=B3_TIMEZONE).astimezone(
        UTC
    ), int(row.get("decision_idx") if row.get("decision_idx") is not None else -1)


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
    names = SIDECAR_FEATURES[group]
    columns = dict(feature_columns or {name: name for name in names})
    if set(columns) != set(names):
        raise ValueError("feature mapping must cover the exact frozen group")
    absent = sorted(
        column
        for column in set(columns.values()) - {None}
        if column not in source.columns
    )
    if absent:
        raise ValueError(f"sidecar source columns missing: {absent}")
    normalized_dates = tuple(_as_date(value) for value in dates)
    date_lookup = {value: index for index, value in enumerate(normalized_dates)}
    isin_lookup = {value: index for index, value in enumerate(isins)}
    values = np.zeros((len(dates), len(isins), len(names)), dtype=np.float32)
    valid = np.zeros(values.shape, dtype=np.bool_)
    chosen: dict[tuple[int, int], Mapping[str, object]] = {}
    for row in source.iter_rows(named=True):
        available_date = _as_date(row["available_date"])
        date_index = date_lookup.get(available_date)
        isin_index = isin_lookup.get(row[identity])
        if date_index is None or isin_index is None:
            continue
        if not _available_before_decision(
            row,
            available_date,
            decision_time,
            date_only_available_before_decision=date_only_available_before_decision,
        ):
            continue
        key = (date_index, isin_index)
        previous = chosen.get(key)
        if previous is not None and _availability_order(
            previous
        ) == _availability_order(row):
            raise ValueError(
                "sidecar has ambiguous rows at one availability coordinate"
            )
        if previous is None or _availability_order(row) > _availability_order(previous):
            chosen[key] = row
    for (date_index, isin_index), row in chosen.items():
        for feature_index, name in enumerate(names):
            column = columns[name]
            if column is None:
                continue
            value = row[column]
            mask_column = f"{column}_mask"
            is_valid = value is not None and np.isfinite(float(value))
            if mask_column in source.columns:
                is_valid &= bool(row[mask_column])
            if is_valid:
                values[date_index, isin_index, feature_index] = float(value)
                valid[date_index, isin_index, feature_index] = True
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
        coverage_by_year=tuple(coverage),
        archive_semantics_available=tuple(
            name for name in names if columns[name] is not None
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
    names = SIDECAR_FEATURES[group]
    columns = dict(feature_columns or {name: name for name in names})
    if set(columns) != set(names):
        raise ValueError("feature mapping must cover the exact frozen group")
    normalized_dates = tuple(_as_date(value) for value in dates)
    date_lookup = {value: index for index, value in enumerate(normalized_dates)}
    isin_lookup = {str(value): index for index, value in enumerate(isins)}
    selected: dict[tuple[int, int], Mapping[str, object]] = {}
    for row in source.iter_rows(named=True):
        available_date = _as_date(row["available_date"])
        coordinate = (date_lookup.get(available_date), isin_lookup.get(str(row[identity])))
        if coordinate[0] is None or coordinate[1] is None:
            continue
        if not _available_before_decision(
            row,
            available_date,
            decision_time,
            date_only_available_before_decision=date_only_available_before_decision,
        ):
            continue
        key = (int(coordinate[0]), int(coordinate[1]))
        previous = selected.get(key)
        if previous is not None and _availability_order(previous) == _availability_order(row):
            raise ValueError("sidecar has ambiguous rows at one availability coordinate")
        if previous is None or _availability_order(row) > _availability_order(previous):
            selected[key] = row
    rebuilt = np.zeros((len(dates), len(isins), len(names)), dtype=np.bool_)
    source_columns = set(source.columns)
    for (date_index, isin_index), row in selected.items():
        for feature_index, name in enumerate(names):
            column = columns[name]
            if column is None:
                continue
            value = row[column]
            valid = value is not None and np.isfinite(float(value))
            mask_column = f"{column}_mask"
            if mask_column in source_columns:
                valid &= bool(row[mask_column])
            rebuilt[date_index, isin_index, feature_index] = valid
    return rebuilt


def _publication_order_expression(source: pl.DataFrame) -> pl.Expr:
    """Return each row's publication instant as UTC microseconds.

    Daily archives usually expose only ``available_date`` (and occasionally a
    v1 ``decision_idx``).  Timestamped archives may use either a timezone-aware
    instant or the historical naive B3-local convention.  One integer ordering
    lets the as-of joins below remain deterministic across input order/chunks.
    """

    timestamp_expressions: list[pl.Expr] = []
    for column in ("available_timestamp", "delivery_timestamp"):
        if column not in source.columns:
            continue
        dtype = source.schema[column]
        if not isinstance(dtype, pl.Datetime):
            raise ValueError(f"{column} must have a Datetime dtype")
        value = pl.col(column)
        if dtype.time_zone is None:
            value = value.dt.replace_time_zone(str(B3_TIMEZONE))
        else:
            value = value.dt.convert_time_zone("UTC")
        timestamp_expressions.append(value.dt.convert_time_zone("UTC").dt.epoch("us"))

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
    has_rate = {
        "source_trade_date",
        "lending_taker_fee_level_log_tanh",
    }.issubset(source.columns)
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

    volume_observed = np.isfinite(volume)
    first_observed = np.full(len(isins), len(calendar), dtype=np.int64)
    observed_names = volume_observed.any(axis=0)
    first_observed[observed_names] = np.argmax(
        volume_observed[:, observed_names], axis=0
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
        zero_volume = np.where(volume_observed, volume, 0.0)
        rolling_volume = np.zeros_like(volume, dtype=np.float64)
        if len(calendar) >= 20:
            volume_windows = np.lib.stride_tricks.sliding_window_view(
                zero_volume, 20, axis=0
            )
            rolling_volume[19:] = np.mean(
                volume_windows, axis=-1, dtype=np.float64
            )
        negative = np.isfinite(volume) & (volume < 0.0)
        rolling_negative = np.zeros(volume.shape, dtype=np.bool_)
        if len(calendar) >= 20:
            negative_windows = np.lib.stride_tricks.sliding_window_view(
                negative, 20, axis=0
            )
            rolling_negative[19:] = np.any(negative_windows, axis=-1)

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
            & ((day_index - 19) >= first_observed[safe_name])
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
        transformed = (
            rate.get_column("lending_taker_fee_level_log_tanh")
            .cast(pl.Float64)
            .fill_null(np.nan)
            .to_numpy()
        )
        if "lending_taker_fee_level_log_tanh_mask" in rate.columns:
            rate_mask = (
                rate.get_column("lending_taker_fee_level_log_tanh_mask")
                .fill_null(True)
                .to_numpy()
            )
        else:
            rate_mask = np.ones(rate.height, dtype=np.bool_)
        rate_valid = rate_mask & np.isfinite(transformed) & (np.abs(transformed) < 1.0)
        raw_rate = np.zeros(rate.height, dtype=np.float64)
        raw_rate[rate_valid] = np.expm1(2.0 * np.arctanh(transformed[rate_valid]))
        rate_valid &= np.isfinite(raw_rate) & (raw_rate >= 0.0)
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
    """Invert archive transforms whose raw v2 quantity is recoverable."""

    output = source
    transforms = (
        (
            "options_put_call_oi_log_ratio_tanh",
            "put_call_oi_ratio",
            3.0,
            False,
        ),
        (
            "options_oi_change_to_stock_adv20_tanh",
            "delta_oi_to_volume_1",
            3.0,
            True,
        ),
        ("options_put_skew_tanh", "put_skew", 0.25, False),
    )
    for archived, feature, scale, signed_log in transforms:
        if archived not in output.columns:
            continue
        archived_mask = f"{archived}_mask"
        values = output.get_column(archived).cast(pl.Float64).fill_null(0.0).to_numpy()
        valid = np.isfinite(values) & (np.abs(values) < 1.0)
        if archived_mask in output.columns:
            valid &= output.get_column(archived_mask).fill_null(False).to_numpy()
        inverse = np.zeros(len(values), dtype=np.float64)
        latent = scale * np.arctanh(np.where(valid, values, 0.0))
        if signed_log:
            inverse[valid] = np.sign(latent[valid]) * np.expm1(np.abs(latent[valid]))
        elif feature == "put_call_oi_ratio":
            inverse[valid] = np.exp(latent[valid])
        else:
            inverse[valid] = latent[valid]
        valid &= np.isfinite(inverse)
        inverse[~valid] = 0.0
        output = output.with_columns(
            pl.Series(feature, inverse),
            pl.Series(f"{feature}_mask", valid),
        )
    return output


def _raw_events_features(
    source: pl.DataFrame, dates: Sequence[date | np.datetime64]
) -> pl.DataFrame:
    """Derive causal sessions since the latest observable RAD earnings event."""

    required = {"available_date", "isin", "event_itr_dfp_recent_5s"}
    if not required.issubset(source.columns):
        return source
    output = _canonical_publication_frame(source)
    ambiguous = (
        output.group_by("isin", _PUBLICATION_ORDER_COLUMN)
        .len()
        .filter(pl.col("len") > 1)
    )
    if not ambiguous.is_empty():
        raise ValueError(
            "ambiguous event states at one publication coordinate: "
            f"{ambiguous.head(10).to_dicts()}"
        )
    recent = (
        pl.col("event_itr_dfp_recent_5s").cast(pl.Float64).fill_null(0.0) > 0.5
    )
    if "event_itr_dfp_recent_5s_mask" in output.columns:
        # Historical event-state archives treated a null mask as an update;
        # only explicit false freezes the prior state.
        mask_ok = pl.col("event_itr_dfp_recent_5s_mask").fill_null(True)
    else:
        mask_ok = pl.lit(True)
    output = (
        output.with_columns(
            recent.alias("__recent"),
            mask_ok.alias("__mask_ok"),
        )
        .with_columns(
            pl.when(pl.col("__mask_ok"))
            .then(pl.col("__recent"))
            .otherwise(None)
            .alias("__masked_recent")
        )
        .with_columns(
            pl.col("__masked_recent")
            .forward_fill()
            .shift(1)
            .over("isin")
            .alias("__prior_recent")
        )
        .with_columns(
            (
                pl.col("__mask_ok")
                & pl.col("__recent")
                & ~pl.col("__prior_recent").fill_null(False)
            ).alias("__event")
        )
        .with_columns(
            pl.when(pl.col("__event"))
            .then(pl.col("available_date").cast(pl.Date))
            .otherwise(None)
            .forward_fill()
            .over("isin")
            .alias("__last_event_date")
        )
    )
    calendar = tuple(_as_date(value) for value in dates)
    current_positions = pl.DataFrame(
        {
            "available_date": calendar,
            "__current_session_position": np.arange(len(calendar), dtype=np.int32),
        },
        schema_overrides={"available_date": pl.Date},
    )
    event_positions = current_positions.rename(
        {
            "available_date": "__last_event_date",
            "__current_session_position": "__event_session_position",
        }
    )
    output = output.join(current_positions, on="available_date", how="left").join(
        event_positions, on="__last_event_date", how="left"
    )
    valid = pl.col("__current_session_position").is_not_null() & pl.col(
        "__event_session_position"
    ).is_not_null()
    return (
        output.with_columns(
            pl.when(valid)
            .then(
                pl.col("__current_session_position")
                - pl.col("__event_session_position")
            )
            .otherwise(0.0)
            .cast(pl.Float64)
            .alias("sessions_since_earnings"),
            valid.alias("sessions_since_earnings_mask"),
        )
        .sort(_ROW_INDEX_COLUMN)
        .drop(
            _ROW_INDEX_COLUMN,
            _PUBLICATION_ORDER_COLUMN,
            "__recent",
            "__mask_ok",
            "__masked_recent",
            "__prior_recent",
            "__event",
            "__last_event_date",
            "__current_session_position",
            "__event_session_position",
        )
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
        return _raw_events_features(source, dates)
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
    """Map every frozen feature, using ``None`` for unavailable archive fields."""

    if group not in ARCHIVE_COLUMN_MAP:
        raise ValueError(f"unknown sidecar group: {group}")
    available = set(columns)
    return {
        feature: (
            feature if feature in available else source if source in available else None
        )
        for feature, source in ARCHIVE_COLUMN_MAP[group].items()
    }


def materialize_known_archive(
    source: pl.DataFrame,
    dates: Sequence[date | np.datetime64],
    isins: Sequence[str],
    *,
    group: str,
    decision_time: time = time(15, 45),
) -> SidecarResult:
    """Materialize all fields available in a known v1 archive.

    The v1 archive ``available_date`` is already source-lagged (including D+1
    where required), so a matching date is authoritative before the decision.
    Missing frozen fields remain exactly zero with a false validity mask and are
    reported as zero coverage rather than rejected.
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
    if result.values.shape[2] != len(result.feature_names):
        raise ValueError("sidecar feature axis is misaligned")
    if not np.isfinite(result.values).all() or np.any(
        result.values[~result.valid] != 0
    ):
        raise ValueError("sidecar values must be finite and invalid cells exactly zero")
    if active is not None:
        membership = np.asarray(active, dtype=np.bool_)
        if membership.shape != result.values.shape[:2]:
            raise ValueError("sidecar membership mask is misaligned")
