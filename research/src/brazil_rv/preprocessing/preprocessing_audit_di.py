from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from brazil_rv.modeling.contract import TRAIN_END

from .analyze_preprocessing import (
    PERSISTENCE_LEVELS,
    AuditArrays,
    AuditDates,
    DistributionAccumulator,
    beta_readiness,
    causal_factor_betas,
    fit_di_curve,
    maturity_hull_intersection,
    spearman_correlation,
    validate_audit_indices,
)
from .contract import (
    BETA_CLIP,
    CONTEXT_SESSION_MINUTES,
    CONTEXT_SESSION_START_MINUTE,
    DECISION_CONTEXT_INDICES,
    DECISION_EQUITY_INDICES,
    EQUITY_SESSION_MINUTES,
    EQUITY_SESSION_START_MINUTE,
    FIXED_RATE_CONTEXT_SYMBOLS,
    PRICE_VOL_FLOOR,
    VOL_EWMA_ALPHA,
    VOL_WARMUP_VALID_DAYS,
)
from .io import (
    SOURCE_COLUMNS,
    cotahist_files,
    dense_grid,
    discover_context_files,
    load_assignments,
    load_context_expiries,
    load_market_dates_and_security_dates,
    prepare_session_bars,
    validate_physical_source_identity,
    validate_source_date_isolation,
)
from .transforms import _daily_summaries, _daily_variance


DI_FIT_AVAILABILITY_THRESHOLD = 0.95
DI_BETA_READINESS_THRESHOLD = 0.80
BIVARIATE_ALIGNMENT_CORRELATION_THRESHOLD = 0.80
EXTRA_DI_BROAD_DECISION_COVERAGE_THRESHOLD = 0.80


def _context_minute_index() -> pl.Expr:
    return (
        pl.col("ts_exchange").dt.hour().cast(pl.Int16) * 60
        + pl.col("ts_exchange").dt.minute().cast(pl.Int16)
        - CONTEXT_SESSION_START_MINUTE
    ).alias("minute_idx")


@dataclass(frozen=True)
class DIInputs:
    context_dir: Path
    catalogue_path: Path
    assignments_dir: Path
    cotahist_dir: Path


@dataclass(frozen=True)
class EquityCausalState:
    sigma: NDArray[np.float64]
    change: NDArray[np.float64]
    change_valid: NDArray[np.bool_]
    source_files: tuple[Path, ...]


@dataclass(frozen=True)
class EquityCausalScope:
    output_security_ids: tuple[str, ...]
    in_scope_assignments: pl.DataFrame
    accepted_dates: dict[str, frozenset[date]]


@dataclass(frozen=True)
class RateGrids:
    close: NDArray[np.float64]
    observed: NDArray[np.bool_]
    expiries: tuple[date, ...]
    source_rows: tuple[dict[str, object], ...]
    files: tuple[Path, ...]


@dataclass(frozen=True)
class ExtraFixedContractCandidate:
    symbol: str
    source_paths: tuple[Path, ...]


def _discover_extra_fixed_contracts(
    context_dir: Path,
) -> tuple[ExtraFixedContractCandidate, ...]:
    found: dict[str, list[Path]] = {}
    for path in sorted(context_dir.glob("*.parquet")):
        schema = pl.read_parquet_schema(path)
        if "symbol" not in schema:
            continue
        frame = pl.read_parquet(path, columns=["symbol"], n_rows=1)
        if frame.is_empty():
            continue
        symbol = str(frame.item())
        if symbol.startswith("DI1F") and symbol[4:].isdigit():
            found.setdefault(symbol, []).append(path)
    return tuple(
        ExtraFixedContractCandidate(symbol, tuple(found[symbol]))
        for symbol in sorted(found.keys() - set(FIXED_RATE_CONTEXT_SYMBOLS))
    )


def _extra_contract_expiry(
    catalogue: pl.DataFrame, symbol: str
) -> tuple[date | None, str | None]:
    rows = catalogue.filter(pl.col("name") == symbol)
    if rows.is_empty():
        return None, "missing_catalogue_expiry"
    expiries: set[date] = set()
    invalid = False
    for value in rows.get_column("expiration_time"):
        try:
            if value is None or float(value) <= 0.0:
                invalid = True
                continue
            expiries.add(datetime.fromtimestamp(float(value), tz=timezone.utc).date())
        except (OSError, OverflowError, TypeError, ValueError):
            invalid = True
    if invalid or not expiries:
        return None, "invalid_catalogue_expiry"
    if len(expiries) != 1:
        return None, "non_unique_catalogue_expiry"
    return next(iter(expiries)), None


def audit_extra_fixed_contracts(
    context_dir: Path,
    catalogue_path: Path,
    dates: AuditDates,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    tuple[Path, ...],
]:
    """Audit raw-only fixed-DI candidates without treating them as store features."""
    candidates = _discover_extra_fixed_contracts(context_dir)
    if not candidates:
        return [], [], ()
    catalogue = pl.read_parquet(catalogue_path, columns=["name", "expiration_time"])
    catalogue_identity = {
        "catalogue_path": str(catalogue_path),
        "catalogue_size_bytes": catalogue_path.stat().st_size,
        "catalogue_mtime_ns": catalogue_path.stat().st_mtime_ns,
    }
    date_lookup = {value: index for index, value in enumerate(dates.trade_dates)}
    allowed_end = dates.trade_dates[int(dates.validation[-1])]
    scopes = (
        (
            "overall",
            "train_validation",
            np.concatenate((dates.train, dates.validation)),
        ),
        ("training", "training", dates.train),
        ("validation", "validation", dates.validation),
    )
    coverage_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    source_files: list[Path] = []
    for candidate in candidates:
        source_files.extend(candidate.source_paths)
        issues: list[str] = []
        expiry, expiry_issue = _extra_contract_expiry(catalogue, candidate.symbol)
        if expiry_issue is not None:
            issues.append(expiry_issue)
        if len(candidate.source_paths) != 1:
            issues.append("non_unique_raw_source")
        source = candidate.source_paths[0] if len(candidate.source_paths) == 1 else None
        source_identities = [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in candidate.source_paths
        ]
        scan = pl.DataFrame(
            schema={
                "symbol": pl.String,
                "ts_exchange": pl.Datetime(time_zone="UTC"),
                "close": pl.Float64,
                "trade_date": pl.Date,
                "minute_idx": pl.Int16,
            }
        )
        minimum: float | None = None
        maximum: float | None = None
        if source is not None:
            schema = pl.read_parquet_schema(source)
            required = {"symbol", "ts_exchange", "close"}
            if missing := sorted(required - set(schema)):
                issues.append(f"missing_raw_columns:{','.join(missing)}")
            else:
                scan = (
                    pl.scan_parquet(source)
                    .select("symbol", "ts_exchange", "close")
                    .with_columns(
                        pl.col("ts_exchange").dt.date().alias("trade_date"),
                        _context_minute_index(),
                    )
                    .filter(
                        pl.col("trade_date").is_between(
                            dates.trade_dates[0], allowed_end
                        ),
                        pl.col("minute_idx").is_between(0, CONTEXT_SESSION_MINUTES - 1),
                        pl.col("close").is_finite(),
                    )
                    .sort("ts_exchange")
                    .collect()
                )
                if scan.is_empty():
                    issues.append("no_usable_audited_period_rows")
                else:
                    symbols = scan.get_column("symbol").unique().to_list()
                    if symbols != [candidate.symbol]:
                        issues.append("raw_source_symbol_identity_mismatch")
                    minimum = float(scan.get_column("close").min())
                    maximum = float(scan.get_column("close").max())
                    if minimum < 1.0 or maximum > 50.0:
                        issues.append("invalid_annual_percentage_rate_scale")
                    duplicate = scan.select("trade_date", "minute_idx").is_duplicated()
                    if duplicate.any():
                        issues.append("duplicate_session_minute")
        observed = np.zeros(
            (len(dates.trade_dates), CONTEXT_SESSION_MINUTES), dtype=bool
        )
        for row in scan.select("trade_date", "minute_idx").iter_rows(named=True):
            date_idx = date_lookup.get(row["trade_date"])
            if date_idx is not None:
                observed[date_idx, int(row["minute_idx"])] = True
        raw_dates = scan.get_column("trade_date").to_list()
        metrics: dict[str, dict[str, object]] = {}
        for scope_kind, scope_value, indices in scopes:
            allowed_dates = {dates.trade_dates[int(index)] for index in indices}
            raw_row_count = sum(value in allowed_dates for value in raw_dates)
            observed_minutes = int(observed[indices].sum())
            possible_minutes = int(indices.size) * CONTEXT_SESSION_MINUTES
            decision_observed = np.asarray(
                [
                    observed[int(index), cutoff - 1]
                    for index in indices
                    for cutoff in DECISION_CONTEXT_INDICES
                ],
                dtype=bool,
            )
            decision_fraction = (
                float(decision_observed.mean()) if decision_observed.size else None
            )
            maturity = (
                [
                    max((expiry - dates.trade_dates[int(index)]).days, 0) / 365.25
                    for index in indices
                ]
                if expiry is not None
                else []
            )
            metrics[scope_kind] = {
                "raw_m1_row_count": raw_row_count,
                "observed_session_minute_count": observed_minutes,
                "possible_session_minute_count": possible_minutes,
                "raw_row_coverage_fraction": (
                    observed_minutes / possible_minutes if possible_minutes else None
                ),
                "observed_date_count": int(observed[indices].any(axis=1).sum()),
                "observed_decision_count": int(decision_observed.sum()),
                "possible_decision_count": int(decision_observed.size),
                "observed_fraction_at_decisions": decision_fraction,
                "missing_fraction_at_decisions": (
                    None if decision_fraction is None else 1.0 - decision_fraction
                ),
                "minimum_time_to_expiry_years": min(maturity) if maturity else None,
                "maximum_time_to_expiry_years": max(maturity) if maturity else None,
            }
        training_fraction = metrics["training"]["observed_fraction_at_decisions"]
        validation_fraction = metrics["validation"]["observed_fraction_at_decisions"]
        broadly_covered = bool(
            not issues
            and training_fraction is not None
            and validation_fraction is not None
            and float(training_fraction) >= EXTRA_DI_BROAD_DECISION_COVERAGE_THRESHOLD
            and float(validation_fraction) >= EXTRA_DI_BROAD_DECISION_COVERAGE_THRESHOLD
        )
        if issues:
            status = "discovered_but_unusable"
        elif broadly_covered:
            status = "broadly_covered_raw_candidate"
        else:
            status = "partially_covered"
        common: dict[str, object] = {
            "symbol": candidate.symbol,
            "contract_role": "raw_archive_extra_candidate",
            "feature_store_slot": None,
            "feature_ready_date_count": None,
            "feature_store_integrated": False,
            "candidate_status": status,
            "audit_issues": issues,
            "expiry_date": expiry,
            "source_path": None if source is None else str(source),
            "source_paths": [str(path) for path in candidate.source_paths],
            "source_identities": source_identities,
            "raw_row_scope": "research_start_through_validation_end",
            "quote_semantics": "annual_percentage_rate",
            "observed_rate_minimum": minimum,
            "observed_rate_maximum": maximum,
            "broad_raw_decision_coverage_threshold": EXTRA_DI_BROAD_DECISION_COVERAGE_THRESHOLD,
            **catalogue_identity,
        }
        for scope_kind, scope_value, indices in scopes:
            coverage_rows.append(
                {
                    **common,
                    "scope_kind": scope_kind,
                    "scope_value": scope_value,
                    "date_count": int(indices.size),
                    **metrics[scope_kind],
                }
            )
        summaries.append(
            {
                **common,
                "training_raw_row_coverage_fraction": metrics["training"][
                    "raw_row_coverage_fraction"
                ],
                "validation_raw_row_coverage_fraction": metrics["validation"][
                    "raw_row_coverage_fraction"
                ],
                "training_decision_observation_coverage": training_fraction,
                "validation_decision_observation_coverage": validation_fraction,
            }
        )
    return coverage_rows, summaries, tuple(source_files)


def load_rate_grids(inputs: DIInputs, dates: AuditDates) -> RateGrids:
    files_by_symbol = discover_context_files(inputs.context_dir)
    expiries_by_symbol = load_context_expiries(inputs.catalogue_path)
    date_lookup = {value: index for index, value in enumerate(dates.trade_dates)}
    close = np.zeros(
        (
            len(dates.trade_dates),
            len(FIXED_RATE_CONTEXT_SYMBOLS),
            CONTEXT_SESSION_MINUTES,
        ),
        dtype=np.float64,
    )
    observed = np.zeros(close.shape, dtype=bool)
    source_rows: list[dict[str, object]] = []
    files: list[Path] = []
    for slot, symbol in enumerate(FIXED_RATE_CONTEXT_SYMBOLS):
        path = files_by_symbol[symbol]
        files.append(path)
        scan = (
            pl.scan_parquet(path)
            .select("ts_exchange", "close")
            .with_columns(
                pl.col("ts_exchange").dt.date().alias("trade_date"),
                _context_minute_index(),
            )
            .filter(
                pl.col("trade_date").is_between(
                    dates.trade_dates[0], dates.trade_dates[int(dates.validation[-1])]
                ),
                pl.col("minute_idx").is_between(0, CONTEXT_SESSION_MINUTES - 1),
                pl.col("close").is_finite(),
            )
            .sort("ts_exchange")
            .collect()
        )
        if scan.is_empty():
            raise ValueError(f"No audit-permitted raw DI rows for {symbol}")
        minimum = float(scan.get_column("close").min())
        maximum = float(scan.get_column("close").max())
        if minimum < 1.0 or maximum > 50.0:
            raise ValueError(f"{symbol} is not in annual percentage-rate units")
        duplicate = scan.select("trade_date", "minute_idx").is_duplicated()
        if duplicate.any():
            raise ValueError(f"Duplicate DI minute in {path}")
        for row in scan.select("trade_date", "minute_idx", "close").iter_rows(
            named=True
        ):
            date_idx = date_lookup.get(row["trade_date"])
            if date_idx is None:
                continue
            minute = int(row["minute_idx"])
            close[date_idx, slot, minute] = float(row["close"])
            observed[date_idx, slot, minute] = True
        source_rows.append(
            {
                "symbol": symbol,
                "raw_m1_row_count": scan.height,
                "raw_row_scope": "research_start_through_validation_end",
                "first_timestamp": str(scan.get_column("ts_exchange").min()),
                "last_timestamp": str(scan.get_column("ts_exchange").max()),
                "source_path": str(path),
                "source_size_bytes": path.stat().st_size,
                "source_mtime_ns": path.stat().st_mtime_ns,
                "quote_semantics": "annual_percentage_rate",
                "five_minute_change_unit": "basis_points",
                "observed_rate_minimum": minimum,
                "observed_rate_maximum": maximum,
            }
        )
    return RateGrids(
        close,
        observed,
        tuple(expiries_by_symbol[symbol] for symbol in FIXED_RATE_CONTEXT_SYMBOLS),
        tuple(source_rows),
        tuple(files),
    )


def _maturity_years(trade_date: date, expiries: tuple[date, ...]) -> np.ndarray:
    return np.asarray(
        [max((expiry - trade_date).days, 0) / 365.25 for expiry in expiries],
        dtype=np.float64,
    )


def _scope(date_value: date) -> tuple[str, str]:
    if date_value <= TRAIN_END:
        return "training_year", str(date_value.year)
    return "validation", "validation"


def _aggregate_factor_rows(
    rows: list[dict[str, object]], model: str
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    scopes: list[tuple[str, str, list[dict[str, object]]]] = [
        ("overall", "train_validation", rows)
    ]
    values = sorted({(str(row["scope_kind"]), str(row["scope_value"])) for row in rows})
    scopes.extend(
        (
            kind,
            value,
            [
                row
                for row in rows
                if row["scope_kind"] == kind and row["scope_value"] == value
            ],
        )
        for kind, value in values
    )
    for scope_kind, scope_value, selected in scopes:
        eligible = len(selected)
        fits = [row for row in selected if row["fit_available"]]
        four = [row for row in selected if int(row["contract_count"]) == 4]
        three_or_more = [row for row in selected if int(row["contract_count"]) >= 3]
        curvature = [row for row in fits if row["curvature"] is not None]

        def summary(field: str, statistic: str) -> float | None:
            array = np.asarray(
                [float(row[field]) for row in fits if row[field] is not None],
                dtype=np.float64,
            )
            if not array.size:
                return None
            return {
                "mean": float(array.mean()),
                "std": float(array.std()),
                "median": float(np.median(array)),
                "p05": float(np.quantile(array, 0.05)),
                "p95": float(np.quantile(array, 0.95)),
                "p99": float(np.quantile(array, 0.99)),
            }[statistic]

        output.append(
            {
                "model": model,
                "scope_kind": scope_kind,
                "scope_value": scope_value,
                "eligible_decision_count": eligible,
                "fit_count": len(fits),
                "fit_availability_fraction": len(fits) / eligible if eligible else None,
                "four_ready_fraction": len(four) / eligible if eligible else None,
                "at_least_three_ready_fraction": len(three_or_more) / eligible
                if eligible
                else None,
                "fewer_than_three_fraction": 1.0 - len(three_or_more) / eligible
                if eligible
                else None,
                "raw_maturity_design_condition_number_median": summary(
                    "raw_maturity_design_condition_number", "median"
                ),
                "raw_maturity_design_condition_number_p99": summary(
                    "raw_maturity_design_condition_number", "p99"
                ),
                "maturity_span_years_median": summary("maturity_span_years", "median"),
                "maturity_span_years_p05": summary("maturity_span_years", "p05"),
                "minimum_distinct_maturity_separation_years_median": summary(
                    "minimum_distinct_maturity_separation_years", "median"
                ),
                "minimum_distinct_maturity_separation_years_p05": summary(
                    "minimum_distinct_maturity_separation_years", "p05"
                ),
                "residual_rmse_mean": summary("residual_rmse", "mean"),
                "residual_rmse_p95": summary("residual_rmse", "p95"),
                "explained_variance_mean": summary("explained_variance", "mean"),
                "explained_variance_p05": (
                    float(
                        np.quantile(
                            [float(row["explained_variance"]) for row in fits], 0.05
                        )
                    )
                    if fits
                    else None
                ),
                "level_mean": summary("level", "mean"),
                "level_std": summary("level", "std"),
                "tilt_mean": summary("tilt", "mean"),
                "tilt_std": summary("tilt", "std"),
                "curvature_fit_count": len(curvature),
                "curvature_fit_fraction": len(curvature) / eligible
                if eligible
                else None,
                "curvature_incremental_residual_reduction_mean": (
                    float(
                        np.mean(
                            [
                                float(row["curvature_incremental_reduction"])
                                for row in curvature
                            ]
                        )
                    )
                    if curvature
                    else None
                ),
            }
        )
    return output


def _daily_rate_factors(
    grids: RateGrids, dates: AuditDates
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    date_count = int(dates.validation[-1]) + 1
    closes = np.zeros((date_count, len(FIXED_RATE_CONTEXT_SYMBOLS)), dtype=np.float64)
    close_valid = np.zeros_like(closes, dtype=bool)
    for date_idx in range(date_count):
        for slot in range(len(FIXED_RATE_CONTEXT_SYMBOLS)):
            positions = np.flatnonzero(grids.observed[date_idx, slot])
            if positions.size:
                closes[date_idx, slot] = grids.close[date_idx, slot, positions[-1]]
                close_valid[date_idx, slot] = True
    changes = np.zeros_like(closes)
    change_valid = np.zeros_like(close_valid)
    prior = np.zeros(closes.shape[1], dtype=np.float64)
    has_prior = np.zeros(closes.shape[1], dtype=bool)
    for date_idx in range(date_count):
        current = close_valid[date_idx]
        paired = current & has_prior
        changes[date_idx, paired] = 100.0 * (closes[date_idx, paired] - prior[paired])
        change_valid[date_idx, paired] = True
        prior[current] = closes[date_idx, current]
        has_prior[current] = True
    factors = np.zeros((date_count, 2), dtype=np.float64)
    valid = np.zeros_like(factors, dtype=bool)
    for date_idx, trade_date in enumerate(dates.trade_dates[:date_count]):
        fit = fit_di_curve(
            changes[date_idx],
            _maturity_years(trade_date, grids.expiries),
            change_valid[date_idx],
        )
        if fit is not None:
            factors[date_idx] = (fit.level, fit.tilt)
            valid[date_idx] = True
    return factors, valid, change_valid


def _causal_sigma(
    raw_grid: NDArray[np.float64],
    observed: NDArray[np.bool_],
    valid_day: NDArray[np.bool_],
) -> NDArray[np.float64]:
    sigma = np.zeros(valid_day.size, dtype=np.float64)
    warmup: list[float] = []
    ewma_variance: float | None = None
    for date_idx in range(valid_day.size):
        if ewma_variance is not None and valid_day[date_idx]:
            sigma[date_idx] = np.sqrt(max(ewma_variance, PRICE_VOL_FLOOR**2))
        variance = _daily_variance(
            raw_grid[date_idx, :, 3], observed[date_idx], is_rate=False
        )
        if not valid_day[date_idx] or variance is None:
            continue
        if ewma_variance is None:
            warmup.append(variance)
            if len(warmup) == VOL_WARMUP_VALID_DAYS:
                ewma_variance = float(np.median(warmup))
        else:
            ewma_variance = (
                1.0 - VOL_EWMA_ALPHA
            ) * ewma_variance + VOL_EWMA_ALPHA * variance
    return sigma


def prepare_equity_causal_scope(
    inputs: DIInputs,
    dates: AuditDates,
    equity_index: pl.DataFrame,
    arrays: AuditArrays,
) -> EquityCausalScope:
    """Validate the permitted identity scope without opening raw MT5 sources."""
    audit_start = dates.trade_dates[0]
    audit_end = dates.trade_dates[int(dates.validation[-1])]
    output_security_ids = tuple(
        equity_index.sort("equity_slot").get_column("security_id").to_list()
    )
    assignments = load_assignments(inputs.assignments_dir)
    in_scope_assignments = assignments.filter(
        pl.col("first_overlap_date") <= audit_end,
        pl.col("last_overlap_date") >= audit_start,
    )
    in_scope_ids = frozenset(in_scope_assignments.get_column("security_id"))

    allowed_indices = np.concatenate((dates.train, dates.validation))
    validate_audit_indices(allowed_indices, dates.trade_dates, allow_validation=True)
    membership = arrays.array("equity_membership.npy")
    readiness = arrays.array("equity_data_ready.npy")
    active = np.asarray(
        membership[allowed_indices] & readiness[allowed_indices], dtype=bool
    ).any(axis=0)
    excluded_active = [
        output_security_ids[int(slot)]
        for slot in np.flatnonzero(active)
        if output_security_ids[int(slot)] not in in_scope_ids
    ]
    if excluded_active:
        raise ValueError(
            "Train/validation feature-store membership/readiness uses securities "
            f"outside the permitted identity reconstruction: {excluded_active}"
        )

    requested_ids = tuple(
        security_id
        for security_id in output_security_ids
        if security_id in in_scope_ids
    )
    _, accepted_dates = load_market_dates_and_security_dates(
        cotahist_files(inputs.cotahist_dir),
        requested_ids,
        audit_start,
        audit_end,
    )
    validate_source_date_isolation(in_scope_assignments, accepted_dates)
    return EquityCausalScope(
        output_security_ids,
        in_scope_assignments,
        accepted_dates,
    )


def load_equity_causal_state(
    inputs: DIInputs,
    dates: AuditDates,
    scope: EquityCausalScope,
) -> EquityCausalState:
    """Rebuild only causal daily state from accepted raw identity segments."""
    assignments = scope.in_scope_assignments
    security_ids = scope.output_security_ids
    accepted_dates = scope.accepted_dates
    audit_dates = dates.trade_dates[: int(dates.validation[-1]) + 1]
    slot_lookup = {security_id: slot for slot, security_id in enumerate(security_ids)}
    sigma = np.zeros((len(audit_dates), len(security_ids)), dtype=np.float64)
    changes = np.zeros((len(audit_dates), len(security_ids)), dtype=np.float64)
    valid = np.zeros_like(changes, dtype=bool)
    source_files: list[Path] = []
    for group in assignments.partition_by("source_file", maintain_order=True):
        source_path = Path(str(group.item(0, "source_file")))
        if not source_path.is_absolute():
            source_path = inputs.assignments_dir / source_path
        source_files.append(source_path)
        source = (
            pl.scan_parquet(source_path)
            .select(SOURCE_COLUMNS)
            .filter(
                pl.col("ts_exchange")
                .dt.date()
                .is_between(
                    dates.trade_dates[0],
                    dates.trade_dates[int(dates.validation[-1])],
                )
            )
            .collect()
        )
        validate_physical_source_identity(group, source, source_path)
        group_security_ids = tuple(group.get_column("security_id").to_list())
        allowed_dates = frozenset().union(
            *(accepted_dates[security_id] for security_id in group_security_ids)
        )
        session_bars = prepare_session_bars(
            source,
            source_path,
            allowed_dates,
            audit_dates,
            EQUITY_SESSION_START_MINUTE,
            EQUITY_SESSION_MINUTES,
        )
        for assignment in group.iter_rows(named=True):
            security_id = assignment["security_id"]
            bars = session_bars.filter(
                pl.col("trade_date").is_in(tuple(accepted_dates[security_id]))
            )
            if bars.is_empty():
                raise ValueError(
                    f"Accepted assignment produced no audit bars: {security_id}"
                )
            raw_grid, observed = dense_grid(
                bars, len(audit_dates), EQUITY_SESSION_MINUTES
            )
            identity_day = np.fromiter(
                (
                    assignment["first_overlap_date"]
                    <= trade_date
                    <= assignment["last_overlap_date"]
                    for trade_date in audit_dates
                ),
                dtype=bool,
                count=len(audit_dates),
            )
            daily = _daily_summaries(
                raw_grid,
                observed,
                identity_day,
                is_rate=False,
                early_open_cutoff=DECISION_EQUITY_INDICES[0],
            )
            slot = slot_lookup[security_id]
            sigma[:, slot] = _causal_sigma(raw_grid, observed, identity_day)
            changes[:, slot] = daily["change"]
            valid[:, slot] = daily["change_valid"]
    return EquityCausalState(sigma, changes, valid, tuple(source_files))


def _coverage_rows(
    grids: RateGrids,
    arrays: AuditArrays,
    dates: AuditDates,
) -> list[dict[str, object]]:
    context_ready = arrays.array("context_data_ready.npy")
    fixed_slots = [
        LOCAL_CONTEXT_SYMBOLS.index(symbol) for symbol in FIXED_RATE_CONTEXT_SYMBOLS
    ]
    source_by_symbol = {str(row["symbol"]): row for row in grids.source_rows}
    rows: list[dict[str, object]] = []
    for slot, (symbol, expiry) in enumerate(
        zip(FIXED_RATE_CONTEXT_SYMBOLS, grids.expiries, strict=True)
    ):
        scopes: list[tuple[str, str, np.ndarray]] = [
            (
                "overall",
                "train_validation",
                np.concatenate((dates.train, dates.validation)),
            ),
            ("validation", "validation", dates.validation),
        ]
        for year in sorted(
            {dates.trade_dates[int(index)].year for index in dates.train}
        ):
            indices = np.asarray(
                [
                    index
                    for index in dates.train
                    if dates.trade_dates[int(index)].year == year
                ],
                dtype=np.int64,
            )
            scopes.append(("training_year", str(year), indices))
        for scope_kind, scope_value, indices in scopes:
            decision_observed = []
            for date_idx in indices:
                for cutoff in DECISION_CONTEXT_INDICES:
                    decision_observed.append(
                        bool(grids.observed[int(date_idx), slot, cutoff - 1])
                    )
            maturity = [
                max((expiry - dates.trade_dates[int(index)]).days, 0) / 365.25
                for index in indices
            ]
            source = source_by_symbol[symbol] if scope_kind == "overall" else {}
            rows.append(
                {
                    "symbol": symbol,
                    "contract_role": "feature_store_baseline",
                    "feature_store_integrated": True,
                    "candidate_status": "feature_store_baseline",
                    "feature_store_slot": fixed_slots[slot],
                    "scope_kind": scope_kind,
                    "scope_value": scope_value,
                    **source,
                    "expiry_date": expiry,
                    "date_count": int(indices.size),
                    "feature_ready_date_count": int(
                        np.asarray(
                            context_ready[indices, fixed_slots[slot]], dtype=bool
                        ).sum()
                    ),
                    "observed_decision_count": int(sum(decision_observed)),
                    "possible_decision_count": len(decision_observed),
                    "observed_fraction_at_decisions": (
                        float(np.mean(decision_observed)) if decision_observed else None
                    ),
                    "missing_fraction_at_decisions": (
                        1.0 - float(np.mean(decision_observed))
                        if decision_observed
                        else None
                    ),
                    "minimum_time_to_expiry_years": min(maturity) if maturity else None,
                    "maximum_time_to_expiry_years": max(maturity) if maturity else None,
                }
            )
    return rows


def di_computability_assessment(
    fit_availability_fraction: float,
    beta_readiness_fraction_by_factor: dict[str, float],
) -> dict[str, object]:
    factor_candidate_computable = (
        fit_availability_fraction >= DI_FIT_AVAILABILITY_THRESHOLD
    )
    exposures_computable = all(
        beta_readiness_fraction_by_factor.get(factor, 0.0)
        >= DI_BETA_READINESS_THRESHOLD
        for factor in ("level", "tilt")
    )
    if factor_candidate_computable and exposures_computable:
        conclusion = (
            "The existing archive is sufficient to construct a causal level/tilt "
            "candidate and stock-specific exposures."
        )
    elif fit_availability_fraction > 0.0 and any(
        value > 0.0 for value in beta_readiness_fraction_by_factor.values()
    ):
        conclusion = (
            "The existing archive supports only partial causal construction of a "
            "level/tilt candidate and stock-specific exposures; the coverage gaps "
            "are quantified in the canonical outputs."
        )
    else:
        conclusion = (
            "The existing archive is insufficient to construct the causal factor "
            "candidate and stock-specific exposures."
        )
    return {
        "conclusion": conclusion,
        "factor_fit_availability_threshold": DI_FIT_AVAILABILITY_THRESHOLD,
        "factor_beta_readiness_threshold": DI_BETA_READINESS_THRESHOLD,
        "fit_availability_fraction": fit_availability_fraction,
        "factor_beta_readiness_fraction_by_factor": beta_readiness_fraction_by_factor,
        "causal_level_tilt_candidate_computable": factor_candidate_computable,
        "causal_stock_specific_exposures_computable": exposures_computable,
    }


def bivariate_contract_beta_alignment(
    beta_rows: list[dict[str, object]],
    threshold: float = BIVARIATE_ALIGNMENT_CORRELATION_THRESHOLD,
) -> dict[str, object]:
    factors: dict[str, dict[str, object]] = {}
    for factor in ("level", "tilt"):
        candidates = [
            row
            for row in beta_rows
            if row.get("row_type") == "existing_contract_beta_correlation"
            and row.get("factor") == factor
            and row.get("spearman_correlation") is not None
        ]
        if not candidates:
            factors[factor] = {
                "contract_symbol": None,
                "paired_count": 0,
                "spearman_correlation": None,
                "maximum_absolute_contract_beta_correlation": None,
                "meets_threshold": False,
            }
            continue
        strongest = max(
            candidates, key=lambda row: abs(float(row["spearman_correlation"]))
        )
        correlation = float(strongest["spearman_correlation"])
        factors[factor] = {
            "contract_symbol": strongest["contract_symbol"],
            "paired_count": int(strongest["paired_count"]),
            "spearman_correlation": correlation,
            "maximum_absolute_contract_beta_correlation": abs(correlation),
            "meets_threshold": abs(correlation) >= threshold,
        }
    return {
        "correlation_threshold": threshold,
        "level": factors["level"],
        "tilt": factors["tilt"],
        "both_factors_individually_meet_threshold": all(
            bool(factors[factor]["meets_threshold"]) for factor in ("level", "tilt")
        ),
        "interpretation": (
            "Limited bivariate factor-to-contract alignment diagnostic; it does "
            "not establish rotation or subspace equivalence."
        ),
    }


def _fit_quality_snapshot(row: dict[str, object]) -> dict[str, object]:
    return {
        key: row[key]
        for key in (
            "fit_availability_fraction",
            "explained_variance_mean",
            "explained_variance_p05",
            "residual_rmse_mean",
            "residual_rmse_p95",
        )
    }


def run_di_audit(
    arrays: AuditArrays,
    dates: AuditDates,
    equity_index: pl.DataFrame,
    inputs: DIInputs,
    *,
    equity_state: EquityCausalState | None = None,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
]:
    grids = load_rate_grids(inputs, dates)
    extra_coverage, extra_summaries, extra_source_files = audit_extra_fixed_contracts(
        inputs.context_dir,
        inputs.catalogue_path,
        dates,
    )
    coverage_rows = _coverage_rows(grids, arrays, dates)
    coverage_rows.extend(extra_coverage)
    context_ready = arrays.array("context_data_ready.npy")
    fixed_slots = np.asarray(
        [LOCAL_CONTEXT_SYMBOLS.index(symbol) for symbol in FIXED_RATE_CONTEXT_SYMBOLS],
        dtype=np.int64,
    )
    audit_dates = np.concatenate((dates.train, dates.validation))
    factor_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    maturity_rows: list[np.ndarray] = []
    maturity_ready: list[np.ndarray] = []
    for date_idx in audit_dates:
        index = int(date_idx)
        trade_date = dates.trade_dates[index]
        scope_kind, scope_value = _scope(trade_date)
        maturity = _maturity_years(trade_date, grids.expiries)
        for decision_idx, cutoff in enumerate(DECISION_CONTEXT_INDICES):
            end = cutoff - 1
            start = cutoff - 6
            endpoint = grids.observed[index, :, end] & grids.observed[index, :, start]
            feature_ready = np.asarray(context_ready[index, fixed_slots], dtype=bool)
            ready = endpoint & feature_ready
            changes = 100.0 * (
                grids.close[index, :, end] - grids.close[index, :, start]
            )
            fit = fit_di_curve(changes, maturity, ready)
            row: dict[str, object] = {
                "scope_kind": scope_kind,
                "scope_value": scope_value,
                "date_idx": index,
                "trade_date": trade_date,
                "decision_idx": decision_idx,
                "cutoff_minute_idx": end,
                "ready_contracts": list(np.asarray(FIXED_RATE_CONTEXT_SYMBOLS)[ready]),
                "maturity_years": maturity.tolist(),
                "contract_count": int(ready.sum()),
                "fit_available": fit is not None,
                "level": None,
                "tilt": None,
                "raw_maturity_design_condition_number": None,
                "maturity_span_years": None,
                "minimum_distinct_maturity_separation_years": None,
                "residual_rmse": None,
                "explained_variance": None,
                "curvature": None,
                "curvature_incremental_reduction": None,
            }
            if fit is not None:
                row.update(
                    {
                        "level": fit.level,
                        "tilt": fit.tilt,
                        "raw_maturity_design_condition_number": fit.raw_maturity_design_condition_number,
                        "maturity_span_years": fit.maturity_span_years,
                        "minimum_distinct_maturity_separation_years": (
                            fit.minimum_distinct_maturity_separation_years
                        ),
                        "residual_rmse": fit.residual_rmse,
                        "explained_variance": fit.explained_variance_fraction,
                        "curvature": fit.curvature,
                        "curvature_incremental_reduction": fit.curvature_incremental_residual_reduction,
                    }
                )
            factor_rows.append(row)
            no_f28 = ready.copy()
            no_f28[FIXED_RATE_CONTEXT_SYMBOLS.index("DI1F28")] = False
            sensitivity = fit_di_curve(changes, maturity, no_f28)
            sensitivity_rows.append(
                {
                    **{
                        key: row[key]
                        for key in (
                            "scope_kind",
                            "scope_value",
                            "date_idx",
                            "trade_date",
                            "decision_idx",
                            "cutoff_minute_idx",
                            "ready_contracts",
                            "maturity_years",
                        )
                    },
                    "contract_count": int(no_f28.sum()),
                    "fit_available": sensitivity is not None,
                    "level": None if sensitivity is None else sensitivity.level,
                    "tilt": None if sensitivity is None else sensitivity.tilt,
                    "raw_maturity_design_condition_number": (
                        None
                        if sensitivity is None
                        else sensitivity.raw_maturity_design_condition_number
                    ),
                    "maturity_span_years": None
                    if sensitivity is None
                    else sensitivity.maturity_span_years,
                    "minimum_distinct_maturity_separation_years": None
                    if sensitivity is None
                    else sensitivity.minimum_distinct_maturity_separation_years,
                    "residual_rmse": None
                    if sensitivity is None
                    else sensitivity.residual_rmse,
                    "explained_variance": None
                    if sensitivity is None
                    else sensitivity.explained_variance_fraction,
                    "curvature": None,
                    "curvature_incremental_reduction": None,
                }
            )
            maturity_rows.append(maturity)
            maturity_ready.append(ready)
    fit_summary = _aggregate_factor_rows(factor_rows, "level_tilt")
    fit_summary.extend(
        _aggregate_factor_rows(sensitivity_rows, "level_tilt_without_DI1F28")
    )
    fit_output = [dict(row_type="summary", **row) for row in fit_summary]
    fit_output.extend(
        dict(row_type="decision", model="level_tilt", **row) for row in factor_rows
    )
    fit_output.extend(
        dict(row_type="decision", model="level_tilt_without_DI1F28", **row)
        for row in sensitivity_rows
    )
    hull = maturity_hull_intersection(
        np.asarray(maturity_rows), np.asarray(maturity_ready)
    )

    daily_factors, daily_factor_valid, daily_contract_valid = _daily_rate_factors(
        grids, dates
    )
    if equity_state is None:
        scope = prepare_equity_causal_scope(inputs, dates, equity_index, arrays)
        equity_state = load_equity_causal_state(inputs, dates, scope)
    betas, beta_ready = causal_factor_betas(
        equity_state.change,
        equity_state.change_valid,
        daily_factors,
        daily_factor_valid,
    )
    membership = arrays.array("equity_membership.npy")
    existing_beta_ready = beta_readiness(
        equity_state.change_valid, daily_contract_valid
    )
    equity_ready = arrays.array("equity_data_ready.npy")
    slow = arrays.array("equity_slow.npy")
    beta_rows: list[dict[str, object]] = []
    all_allowed = np.concatenate((dates.train, dates.validation))
    active_allowed = np.asarray(
        membership[all_allowed] & equity_ready[all_allowed], dtype=bool
    )
    total_active = int(active_allowed.sum())
    for factor_idx, factor_name in enumerate(("level", "tilt")):
        scopes: list[tuple[str, str, np.ndarray]] = [
            ("overall", "train_validation", all_allowed),
            ("validation", "validation", dates.validation),
        ]
        for year in sorted(
            {dates.trade_dates[int(index)].year for index in dates.train}
        ):
            scopes.append(
                (
                    "training_year",
                    str(year),
                    np.asarray(
                        [
                            index
                            for index in dates.train
                            if dates.trade_dates[int(index)].year == year
                        ],
                        dtype=np.int64,
                    ),
                )
            )
        for scope_kind, scope_value, indices in scopes:
            active = np.asarray(membership[indices] & equity_ready[indices], dtype=bool)
            use = active & beta_ready[indices, :, factor_idx]
            values = betas[indices, :, factor_idx][use]
            stats = DistributionAccumulator(
                f"di_beta:{factor_name}:{scope_kind}:{scope_value}",
                -BETA_CLIP,
                BETA_CLIP,
            )
            stats.update(values, possible_count=int(active.sum()))
            beta_rows.append(
                {
                    "row_type": "factor_beta_distribution",
                    "factor": factor_name,
                    "scope_kind": scope_kind,
                    "scope_value": scope_value,
                    **stats.row(),
                }
            )
        ready_fraction = np.divide(
            (beta_ready[all_allowed, :, factor_idx] & active_allowed).sum(axis=0),
            active_allowed.sum(axis=0),
            out=np.zeros(active_allowed.shape[1], dtype=np.float64),
            where=active_allowed.sum(axis=0) > 0,
        )
        for threshold in PERSISTENCE_LEVELS:
            beta_rows.append(
                {
                    "row_type": "persistent_readiness",
                    "factor": factor_name,
                    "scope_kind": "overall",
                    "scope_value": "train_validation",
                    "readiness_threshold": threshold,
                    "equity_fraction_meeting_threshold": float(
                        np.mean(ready_fraction >= threshold)
                    ),
                }
            )
        factor_values = betas[all_allowed, :, factor_idx]
        for offset, symbol in enumerate(FIXED_RATE_CONTEXT_SYMBOLS):
            existing = np.asarray(slow[all_allowed, :, 22 + offset], dtype=np.float64)
            use = (
                active_allowed
                & beta_ready[all_allowed, :, factor_idx]
                & existing_beta_ready[all_allowed, :, offset]
            )
            beta_rows.append(
                {
                    "row_type": "existing_contract_beta_correlation",
                    "factor": factor_name,
                    "scope_kind": "overall",
                    "scope_value": "train_validation",
                    "contract_symbol": symbol,
                    "paired_count": int(use.sum()),
                    "spearman_correlation": spearman_correlation(
                        factor_values[use], existing[use]
                    ),
                }
            )
    overall_fit = next(
        row
        for row in fit_summary
        if row["model"] == "level_tilt" and row["scope_kind"] == "overall"
    )
    overall_sensitivity = next(
        row
        for row in fit_summary
        if row["model"] == "level_tilt_without_DI1F28"
        and row["scope_kind"] == "overall"
    )
    fit_fraction = float(overall_fit["fit_availability_fraction"] or 0.0)
    beta_readiness_by_factor = {
        factor: float(
            next(
                row
                for row in beta_rows
                if row["row_type"] == "factor_beta_distribution"
                and row["factor"] == factor
                and row["scope_kind"] == "overall"
            )["observed_fraction"]
            or 0.0
        )
        for factor in ("level", "tilt")
    }
    computability = di_computability_assessment(fit_fraction, beta_readiness_by_factor)
    alignment = bivariate_contract_beta_alignment(beta_rows)
    broad_extra = [
        row["symbol"]
        for row in extra_summaries
        if row["candidate_status"] == "broadly_covered_raw_candidate"
    ]
    feasibility: dict[str, object] = {
        "verdict": computability["conclusion"],
        "audit_interval_end": str(dates.trade_dates[int(dates.validation[-1])]),
        "held_out_test_accessed": False,
        "baseline_contracts": list(FIXED_RATE_CONTEXT_SYMBOLS),
        "maturity_hull": hull,
        "computability": computability,
        "fit_quality_diagnostics": {
            "status": (
                "continuous_diagnostics_require_review_of_di_factor_fit_summary.csv"
            ),
            "baseline_level_tilt": _fit_quality_snapshot(overall_fit),
            "missing_DI1F28_sensitivity": _fit_quality_snapshot(overall_sensitivity),
        },
        "empirical_usefulness": {
            "established_by_this_audit": False,
            "required_next_evidence": "chronological_ablation",
        },
        "bivariate_contract_beta_alignment": alignment,
        "extra_fixed_contract_audit": {
            "broad_raw_decision_coverage_threshold": (
                EXTRA_DI_BROAD_DECISION_COVERAGE_THRESHOLD
            ),
            "contracts": extra_summaries,
            "broadly_covered_raw_candidates": broad_extra,
            "integration_status": (
                "diagnostic_only; extra contracts are not integrated into the "
                "current feature store"
            ),
        },
        "construction_scope": {
            "level_tilt_uses_only_baseline_contracts": list(FIXED_RATE_CONTEXT_SYMBOLS),
            "curvature": "diagnostic_only_when_all_four_contracts_are_ready",
            "constant_maturity_interpolation_over_full_audited_sample": hull[
                "constant_maturity_without_extrapolation_full_interval"
            ],
            "exact_historical_rolled_curve_or_tradable_contract_reconstruction": False,
        },
        "limitations": [
            "Raw row counts and timestamps are restricted through validation end to preserve the held-out split.",
            "Continuous DI1 series are not used as exact maturity points.",
            "Curvature is never proposed as a production feature.",
            "Availability and readiness establish computability, not empirical usefulness.",
        ],
    }
    access = {
        "rate_source_files": [str(path) for path in grids.files],
        "rate_source_identities": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in grids.files
        ],
        "extra_rate_source_files": [str(path) for path in extra_source_files],
        "extra_rate_source_identities": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in extra_source_files
        ],
        "equity_source_files": [str(path) for path in equity_state.source_files],
        "equity_source_identities": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in equity_state.source_files
        ],
        "factor_decision_count": len(factor_rows),
        "equity_daily_change_valid_count": int(equity_state.change_valid.sum()),
        "factor_daily_valid_count": int(daily_factor_valid[:, 0].sum()),
        "active_equity_date_count": total_active,
    }
    return coverage_rows, fit_output, beta_rows, feasibility, access


# Imported late to keep the contract list above visually focused.
from .contract import LOCAL_CONTEXT_SYMBOLS  # noqa: E402
