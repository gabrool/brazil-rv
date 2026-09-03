from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Mapping, Sequence

import numpy as np
import polars as pl
from numpy.typing import NDArray

from .contract import SIDECAR_FEATURES


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
        "ex_distribution_next_1": None,
        "ex_distribution_next_2": None,
        "ex_distribution_next_3": None,
        "standardized_unexpected_earnings": None,
    },
    "options": {
        "put_call_oi_ratio": None,
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
        candidate = datetime.combine(decision_date, decision_time)
        if timestamp.tzinfo is not None:
            candidate = candidate.replace(tzinfo=timestamp.tzinfo)
        return timestamp <= candidate
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
        if timestamp.tzinfo is not None:
            timestamp = timestamp.replace(tzinfo=None)
        return timestamp, int(row.get("decision_idx") or -1)
    available_date = _as_date(row["available_date"])
    return datetime.combine(available_date, time.min), int(
        row.get("decision_idx") if row.get("decision_idx") is not None else -1
    )


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
    date_lookup = {value: index for index, value in enumerate(calendar)}
    isin_lookup = {value: index for index, value in enumerate(isins)}
    volume = np.asarray(daily_volume_brl, dtype=np.float64)
    if volume.shape != (len(calendar), len(isins)):
        raise ValueError("daily BRL volume is misaligned with lending axes")
    levels: dict[tuple[str, date], tuple[float, bool]] = {}
    rates: dict[tuple[str, date], tuple[float, bool]] = {}
    output_rows: list[dict[str, object]] = []
    for row in source.iter_rows(named=True):
        isin = str(row["isin"])
        output = dict(row)
        if has_balance:
            output.update(
                {
                    "loan_balance_to_volume_20": 0.0,
                    "loan_balance_to_volume_20_mask": False,
                    "loan_balance_change_1": 0.0,
                    "loan_balance_change_1_mask": False,
                    "loan_balance_change_5": 0.0,
                    "loan_balance_change_5_mask": False,
                }
            )
        if has_rate:
            output.update(
                {
                    "loan_rate": 0.0,
                    "loan_rate_mask": False,
                    "loan_rate_change_5": 0.0,
                    "loan_rate_change_5_mask": False,
                }
            )
        source_position = row.get("source_position_date")
        if has_balance and source_position is not None:
            source_date = _as_date(source_position)
            day = date_lookup.get(source_date)
            name = isin_lookup.get(isin)
            balance = row.get("lending_balance_brl")
            valid = day is not None and name is not None and day >= 19
            mean_volume = np.nan
            if valid:
                history = volume[day - 19 : day + 1, name]
                # A non-trading name-day is an economic zero in the exact
                # calendar window. Negative recorded volume is invalid.
                valid = bool(np.all(~np.isfinite(history) | (history >= 0.0)))
                mean_volume = (
                    float(np.mean(np.where(np.isfinite(history), history, 0.0)))
                    if valid
                    else np.nan
                )
                valid &= (
                    balance is not None
                    and np.isfinite(float(balance))
                    and float(balance) >= 0.0
                    and np.isfinite(mean_volume)
                    and mean_volume > 0.0
                )
            level = float(balance) / mean_volume if valid else 0.0
            levels[(isin, source_date)] = (level, valid)
            output["loan_balance_to_volume_20"] = level
            output["loan_balance_to_volume_20_mask"] = valid
        source_trade = row.get("source_trade_date")
        transformed = row.get("lending_taker_fee_level_log_tanh")
        transformed_mask = row.get("lending_taker_fee_level_log_tanh_mask")
        if has_rate and source_trade is not None:
            source_date = _as_date(source_trade)
            valid = (
                transformed is not None
                and transformed_mask is not False
                and np.isfinite(float(transformed))
                and abs(float(transformed)) < 1.0
            )
            rate = (
                float(np.expm1(2.0 * np.arctanh(float(transformed)))) if valid else 0.0
            )
            valid &= np.isfinite(rate) and rate >= 0.0
            rates[(isin, source_date)] = (rate, valid)
            output["loan_rate"] = rate if valid else 0.0
            output["loan_rate_mask"] = valid
        output_rows.append(output)
    for row in output_rows:
        source_position = row.get("source_position_date")
        if source_position is not None:
            source_date = _as_date(source_position)
            day = date_lookup.get(source_date)
            current = levels.get((row["isin"], source_date), (0.0, False))
            for lag in (1, 5):
                prior_date = (
                    calendar[day - lag] if day is not None and day >= lag else None
                )
                prior = levels.get((row["isin"], prior_date), (0.0, False))
                valid = current[1] and prior[1]
                feature = f"loan_balance_change_{lag}"
                row[feature] = current[0] - prior[0] if valid else 0.0
                row[f"{feature}_mask"] = valid
        source_trade = row.get("source_trade_date")
        if source_trade is not None:
            source_date = _as_date(source_trade)
            day = date_lookup.get(source_date)
            prior_date = calendar[day - 5] if day is not None and day >= 5 else None
            current = rates.get((row["isin"], source_date), (0.0, False))
            prior = rates.get((row["isin"], prior_date), (0.0, False))
            valid = current[1] and prior[1]
            row["loan_rate_change_5"] = current[0] - prior[0] if valid else 0.0
            row["loan_rate_change_5_mask"] = valid
    return (
        pl.DataFrame(output_rows)
        if output_rows
        else pl.DataFrame(schema={"available_date": pl.Date, "isin": pl.String})
    )


def _raw_options_features(source: pl.DataFrame) -> pl.DataFrame:
    """Invert archive transforms whose raw v2 quantity is recoverable."""

    output = source
    transforms = (
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
        else:
            inverse[valid] = latent[valid]
        valid &= np.isfinite(inverse)
        inverse[~valid] = 0.0
        output = output.with_columns(
            pl.Series(feature, inverse),
            pl.Series(f"{feature}_mask", valid),
        )
    return output


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
    date_lookup = {value: index for index, value in enumerate(calendar)}
    shares: dict[tuple[str, date], tuple[float, bool]] = {}
    rows: list[dict[str, object]] = []
    for row in source.iter_rows(named=True):
        isin = str(row["isin"])
        source_date = _as_date(row["source_trade_date"])
        regular = row["regular_volume_brl"]
        odd = row["odd_lot_volume_brl"]
        valid = (
            date_lookup.get(source_date) is not None
            and regular is not None
            and odd is not None
            and np.isfinite(float(regular))
            and np.isfinite(float(odd))
            and float(regular) >= 0.0
            and float(odd) >= 0.0
            and float(regular) + float(odd) > 0.0
        )
        share = float(odd) / (float(regular) + float(odd)) if valid else 0.0
        shares[(isin, source_date)] = (share, valid)
        rows.append(
            {
                "available_date": _as_date(row["available_date"]),
                "isin": isin,
                "source_trade_date": source_date,
                "oddlot_volume_share": share,
                "oddlot_volume_share_mask": valid,
            }
        )
    for row in rows:
        day = date_lookup.get(row["source_trade_date"])
        prior_date = calendar[day - 5] if day is not None and day >= 5 else None
        current = shares[(row["isin"], row["source_trade_date"])]
        prior = shares.get((row["isin"], prior_date), (0.0, False))
        valid = current[1] and prior[1]
        row["oddlot_volume_share_change_5"] = current[0] - prior[0] if valid else 0.0
        row["oddlot_volume_share_change_5_mask"] = valid
    return (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(schema={"available_date": pl.Date, "isin": pl.String})
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
