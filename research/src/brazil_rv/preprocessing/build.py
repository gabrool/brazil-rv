from __future__ import annotations

import json
import time as clock
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from numpy.lib.format import open_memmap

from .audit import audit_feature_store

from .contract import (
    CANONICAL_OUTPUT_POINTER,
    CONTEXT_FAMILIES,
    CONTEXT_SESSION_MINUTES,
    CONTEXT_SESSION_START_MINUTE,
    CONTEXT_SYMBOLS,
    CONTRACT_VERSION,
    DECISION_CONTEXT_INDICES,
    DECISION_EQUITY_INDICES,
    DECISION_TIMES,
    DYNAMIC_CHANNELS,
    EQUITY_SESSION_MINUTES,
    EQUITY_SESSION_START_MINUTE,
    EQUITY_SLOW_CHANNELS,
    EXPECTED_DATE_COUNT,
    EXPECTED_EQUITIES,
    EXPECTED_SAMPLE_COUNT,
    HORIZONS,
    MIN_ACTIVE_EQUITIES,
    OUTPUT_BASE,
    RATE_CONTEXT_SYMBOLS,
    manifest_constants,
    output_array_specs,
)
from .io import (
    cotahist_files,
    create_output_memmaps,
    dense_grid,
    discover_context_files,
    expand_membership,
    full_session_final_closes,
    load_assignments,
    load_context_expiries,
    load_market_dates_and_security_dates,
    load_source_file,
    prepare_session_bars,
    read_research_interval,
    resolve_inputs,
    validate_physical_source_identity,
    validate_source_date_isolation,
)
from .transforms import (
    add_equity_cross_sectional_dynamic,
    add_slow_cross_sectional_ranks,
    build_causal_features,
    build_daily_changes,
    build_prior_rate_level,
    build_raw_returns,
    causal_exposure_betas,
    center_cross_section,
    time_to_expiry_scaled,
)


def main() -> None:
    started = clock.perf_counter()
    created_at = datetime.now(timezone.utc)
    inputs = resolve_inputs()
    research_start, research_end = read_research_interval(inputs.universe_dir)
    assignments = load_assignments(inputs.assignments_dir)
    security_ids = tuple(assignments.get_column("security_id").to_list())
    slot_by_security = {
        security_id: slot for slot, security_id in enumerate(security_ids)
    }
    market_dates, bar_assignment_dates = load_market_dates_and_security_dates(
        cotahist_files(inputs.cotahist_dir),
        security_ids,
        research_start,
        research_end,
    )
    validate_source_date_isolation(assignments, bar_assignment_dates)
    context_files = discover_context_files(inputs.context_dir)
    context_expiries = load_context_expiries(inputs.catalogue_path)

    if len(market_dates) != EXPECTED_DATE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_DATE_COUNT} market dates, found {len(market_dates)}"
        )
    output_dir = OUTPUT_BASE / f"m1_features_v1_{created_at:%Y%m%dT%H%M%S%fZ}"
    arrays = create_output_memmaps(output_dir, len(market_dates))

    membership_rows = pl.read_parquet(
        inputs.universe_dir / "universe_membership_monthly.parquet"
    ).select(
        "security_id",
        "effective_from",
        "effective_to_exclusive",
        "is_member",
    )
    membership = expand_membership(membership_rows, market_dates, security_ids)
    arrays["equity_membership.npy"][...] = membership

    equity_sigma = np.zeros((len(market_dates), EXPECTED_EQUITIES), dtype=np.float64)
    equity_change = np.zeros_like(equity_sigma)
    equity_change_valid = np.zeros_like(equity_sigma, dtype=bool)
    dynamic_valid_path = output_dir / ".equity_dynamic_valid.npy"
    dynamic_valid = open_memmap(
        dynamic_valid_path,
        mode="w+",
        dtype=bool,
        shape=(len(market_dates), EXPECTED_EQUITIES, EQUITY_SESSION_MINUTES, 4),
    )
    dynamic_valid[...] = False
    slow_rank_valid_path = output_dir / ".equity_slow_rank_valid.npy"
    slow_rank_valid = open_memmap(
        slow_rank_valid_path,
        mode="w+",
        dtype=bool,
        shape=(len(market_dates), EXPECTED_EQUITIES, 3),
    )
    slow_rank_valid[...] = False
    security_audits: list[dict[str, object]] = []
    source_groups = assignments.partition_by("source_file", maintain_order=True)
    for source_number, group in enumerate(source_groups, start=1):
        source_path = Path(group.item(0, "source_file"))
        source = load_source_file(source_path)
        validate_physical_source_identity(group, source, source_path)
        group_security_ids = tuple(group.get_column("security_id").to_list())
        allowed_dates = frozenset().union(
            *(bar_assignment_dates[security_id] for security_id in group_security_ids)
        )
        session_bars = prepare_session_bars(
            source,
            source_path,
            allowed_dates,
            market_dates,
            EQUITY_SESSION_START_MINUTE,
            EQUITY_SESSION_MINUTES,
        )
        for assignment in group.iter_rows(named=True):
            security_id = assignment["security_id"]
            slot = slot_by_security[security_id]
            assignment_dates = bar_assignment_dates[security_id]
            bars = session_bars.filter(
                pl.col("trade_date").is_in(tuple(assignment_dates))
            )
            if bars.is_empty():
                raise ValueError(
                    "Accepted assignment produced no used bars: "
                    f"security_id={security_id}, xp_symbol={assignment['xp_symbol']}, "
                    f"source={source_path}"
                )
            raw_grid, observed = dense_grid(
                bars, len(market_dates), EQUITY_SESSION_MINUTES
            )
            identity_day = np.fromiter(
                (
                    assignment["first_overlap_date"]
                    <= trade_date
                    <= assignment["last_overlap_date"]
                    for trade_date in market_dates
                ),
                dtype=bool,
                count=len(market_dates),
            )
            result = build_causal_features(
                raw_grid,
                observed,
                identity_day,
                is_rate=False,
                market_dates=market_dates,
            )
            arrays["equity_features.npy"][:, slot] = result.dynamic
            arrays["equity_slow.npy"][:, slot] = result.slow
            arrays["equity_data_ready.npy"][:, slot] = result.data_ready
            dynamic_valid[:, slot] = result.dynamic_valid
            slow_rank_valid[:, slot] = result.slow_rank_valid
            equity_sigma[:, slot] = result.sigma
            equity_change[:, slot] = result.daily_change
            equity_change_valid[:, slot] = result.daily_change_valid

            raw_returns, _ = build_raw_returns(raw_grid, observed)
            arrays["raw_returns.npy"][:, slot] = raw_returns
            member = membership[:, slot]
            security_audits.append(
                {
                    "equity_slot": slot,
                    "security_id": security_id,
                    "latest_ticker": assignment["latest_ticker"],
                    "source_file": str(source_path),
                    "valid_source_dates": len(assignment_dates),
                    "feature_ready_dates": int(result.data_ready.sum()),
                    "member_dates": int(member.sum()),
                    "active_dates": int((member & result.data_ready).sum()),
                    "observed_bars": int(observed.sum()),
                    "possible_session_bars": len(assignment_dates)
                    * EQUITY_SESSION_MINUTES,
                }
            )
        if source_number % 20 == 0 or source_number == len(source_groups):
            print(f"Processed equity source {source_number}/{len(source_groups)}")

    all_market_dates = frozenset(market_dates)
    context_change = np.zeros(
        (len(market_dates), len(CONTEXT_SYMBOLS)), dtype=np.float64
    )
    context_change_valid = np.zeros_like(context_change, dtype=bool)
    for context_slot, symbol in enumerate(CONTEXT_SYMBOLS):
        source_path = context_files[symbol]
        source = load_source_file(source_path)
        bars = prepare_session_bars(
            source,
            source_path,
            all_market_dates,
            market_dates,
            CONTEXT_SESSION_START_MINUTE,
            CONTEXT_SESSION_MINUTES,
        )
        raw_grid, observed = dense_grid(
            bars, len(market_dates), CONTEXT_SESSION_MINUTES
        )
        valid_day = np.ones(len(market_dates), dtype=bool)
        if symbol in RATE_CONTEXT_SYMBOLS:
            daily_close, daily_close_observed = full_session_final_closes(
                source, market_dates
            )
            prior_rate, prior_ready = build_prior_rate_level(
                daily_close, daily_close_observed
            )
            expiry_scaled = time_to_expiry_scaled(
                market_dates, context_expiries[symbol]
            )
            result = build_causal_features(
                raw_grid,
                observed,
                valid_day,
                is_rate=True,
                extra_ready=prior_ready,
                market_dates=market_dates,
                include_dollar_volume=False,
            )
            daily_change, daily_change_valid = build_daily_changes(
                daily_close,
                daily_close_observed,
                is_rate=True,
            )
        else:
            result = build_causal_features(
                raw_grid,
                observed,
                valid_day,
                is_rate=False,
                market_dates=market_dates,
                include_dollar_volume=False,
            )
            daily_change = result.daily_change
            daily_change_valid = result.daily_change_valid
        arrays["context_features.npy"][:, context_slot] = result.dynamic
        arrays["context_slow.npy"][:, context_slot] = result.slow
        if symbol in RATE_CONTEXT_SYMBOLS:
            arrays["context_slow.npy"][:, context_slot, 30] = prior_rate
            arrays["context_slow.npy"][:, context_slot, 31] = expiry_scaled
        arrays["context_data_ready.npy"][:, context_slot] = result.data_ready
        context_change[:, context_slot] = daily_change
        context_change_valid[:, context_slot] = daily_change_valid
        print(f"Processed context {symbol}")

    exposure_betas = causal_exposure_betas(
        equity_change,
        equity_change_valid,
        context_change,
        context_change_valid,
    )
    exposure_betas *= arrays["equity_data_ready.npy"][..., None]
    arrays["equity_slow.npy"][:, :, 20:26] = exposure_betas

    sample_rows: list[dict[str, object]] = []
    daily_audits: list[dict[str, object]] = []
    security_label_counts = np.zeros((EXPECTED_EQUITIES, len(HORIZONS)), dtype=np.int64)
    sample_id = 0
    for date_idx, trade_date in enumerate(market_dates):
        active = membership[date_idx] & arrays["equity_data_ready.npy"][date_idx]
        add_equity_cross_sectional_dynamic(
            arrays["equity_features.npy"][date_idx],
            dynamic_valid[date_idx],
            active,
        )
        add_slow_cross_sectional_ranks(
            arrays["equity_slow.npy"][date_idx], slow_rank_valid[date_idx], active
        )
        observed = arrays["equity_features.npy"][date_idx, :, :, 5].astype(bool)
        entry_observed = observed[:, DECISION_EQUITY_INDICES]
        exit_observed = np.stack(
            [
                observed[:, np.asarray(DECISION_EQUITY_INDICES) + horizon - 1]
                for horizon in HORIZONS
            ],
            axis=2,
        )
        candidate = (
            membership[date_idx, :, None, None]
            & arrays["equity_data_ready.npy"][date_idx, :, None, None]
            & entry_observed[:, :, None]
            & exit_observed
        )
        (
            masked_raw,
            label_mask,
            targets,
            medians,
            horizon_mask,
        ) = center_cross_section(
            arrays["raw_returns.npy"][date_idx],
            candidate,
            equity_sigma[date_idx],
        )
        arrays["raw_returns.npy"][date_idx] = masked_raw
        arrays["label_mask.npy"][date_idx] = label_mask
        arrays["targets.npy"][date_idx] = targets
        arrays["cross_section_median.npy"][date_idx] = medians
        arrays["horizon_mask.npy"][date_idx] = horizon_mask
        security_label_counts += label_mask.sum(axis=1)

        active_count = int(active.sum())
        context_ready_count = int(arrays["context_data_ready.npy"][date_idx].sum())
        feature_eligible = (
            context_ready_count == len(CONTEXT_SYMBOLS)
            and active_count >= MIN_ACTIVE_EQUITIES
        )
        valid_label_counts = label_mask.sum(axis=0)
        if feature_eligible:
            for decision_idx, decision_time in enumerate(DECISION_TIMES):
                sample_rows.append(
                    {
                        "sample_id": sample_id,
                        "date_idx": date_idx,
                        "trade_date": trade_date,
                        "decision_idx": decision_idx,
                        "decision_time": decision_time,
                        "equity_cutoff_index": DECISION_EQUITY_INDICES[decision_idx],
                        "context_cutoff_index": DECISION_CONTEXT_INDICES[decision_idx],
                        "active_equity_count": active_count,
                        "valid_label_count_30": int(
                            valid_label_counts[decision_idx, 0]
                        ),
                        "valid_label_count_60": int(
                            valid_label_counts[decision_idx, 1]
                        ),
                        "valid_label_count_120": int(
                            valid_label_counts[decision_idx, 2]
                        ),
                        "horizon_mask_30": bool(horizon_mask[decision_idx, 0]),
                        "horizon_mask_60": bool(horizon_mask[decision_idx, 1]),
                        "horizon_mask_120": bool(horizon_mask[decision_idx, 2]),
                    }
                )
                sample_id += 1
        daily_audits.append(
            {
                "trade_date": trade_date,
                "member_count_accepted_axis": int(membership[date_idx].sum()),
                "feature_ready_count": int(
                    arrays["equity_data_ready.npy"][date_idx].sum()
                ),
                "active_equity_count": active_count,
                "context_ready_count": context_ready_count,
                "sample_count": len(DECISION_TIMES) if feature_eligible else 0,
                "valid_labels_30": int(label_mask[:, :, 0].sum()),
                "valid_labels_60": int(label_mask[:, :, 1].sum()),
                "valid_labels_120": int(label_mask[:, :, 2].sum()),
            }
        )

    dynamic_valid.flush()
    slow_rank_valid.flush()
    del dynamic_valid, slow_rank_valid
    dynamic_valid_path.unlink()
    slow_rank_valid_path.unlink()

    for array in arrays.values():
        array.flush()

    date_index = _date_index_frame(market_dates)
    equity_index = _equity_index_frame(assignments)
    context_index = _context_index_frame(context_files, context_expiries)
    sample_index = _sample_index_frame(sample_rows)
    if sample_index.height != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLE_COUNT} samples, found {sample_index.height}"
        )
    _validate_output(
        arrays,
        assignments,
        context_index,
        context_expiries,
        sample_index,
    )

    for audit in security_audits:
        slot = int(audit["equity_slot"])
        possible = int(audit["possible_session_bars"])
        audit["observed_bar_fraction"] = (
            float(audit["observed_bars"]) / possible if possible else 0.0
        )
        audit["valid_labels_30"] = int(security_label_counts[slot, 0])
        audit["valid_labels_60"] = int(security_label_counts[slot, 1])
        audit["valid_labels_120"] = int(security_label_counts[slot, 2])

    date_index.write_parquet(output_dir / "date_index.parquet")
    equity_index.write_parquet(output_dir / "equity_index.parquet")
    context_index.write_parquet(output_dir / "context_index.parquet")
    sample_index.write_parquet(output_dir / "sample_index.parquet")
    pl.DataFrame(daily_audits).write_parquet(output_dir / "daily_audit.parquet")
    pl.DataFrame(security_audits).sort("equity_slot").write_parquet(
        output_dir / "security_audit.parquet"
    )

    first_feature_date = sample_index.item(0, "trade_date")
    duration = clock.perf_counter() - started
    _write_feature_schema(output_dir)
    _write_manifest(
        output_dir,
        inputs.manifest_entries(),
        assignments,
        context_files,
        market_dates,
        sample_index.height,
        first_feature_date,
        research_start,
        research_end,
        created_at,
        duration,
    )
    audit_dir = audit_feature_store(output_dir)
    _promote(output_dir)
    print(f"Canonical output: {output_dir}")
    print(f"Statistical audit: {audit_dir}")


def _date_index_frame(market_dates: tuple[object, ...]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date_idx": pl.Series(range(len(market_dates)), dtype=pl.Int32),
            "trade_date": pl.Series(market_dates, dtype=pl.Date),
        }
    )


def _equity_index_frame(assignments: pl.DataFrame) -> pl.DataFrame:
    return (
        assignments.with_row_index("equity_slot")
        .with_columns(pl.col("equity_slot").cast(pl.Int16))
        .select(
            "equity_slot",
            "security_id",
            "isin",
            "latest_ticker",
            "xp_symbol",
            "source_file",
            "source_assignment_type",
            "first_overlap_date",
            "last_overlap_date",
        )
    )


def _context_index_frame(
    context_files: dict[str, Path], context_expiries: dict[str, object]
) -> pl.DataFrame:
    expiry_dates = [context_expiries.get(symbol) for symbol in CONTEXT_SYMBOLS]
    return pl.DataFrame(
        {
            "context_slot": pl.Series(range(len(CONTEXT_SYMBOLS)), dtype=pl.Int8),
            "symbol": CONTEXT_SYMBOLS,
            "family": CONTEXT_FAMILIES,
            "source_file": [str(context_files[symbol]) for symbol in CONTEXT_SYMBOLS],
            "expiry_date": pl.Series(expiry_dates, dtype=pl.Date),
        }
    )


def _sample_index_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(
        pl.col("sample_id").cast(pl.Int64),
        pl.col("date_idx").cast(pl.Int32),
        pl.col("trade_date").cast(pl.Date),
        pl.col("decision_idx").cast(pl.Int8),
        pl.col("decision_time").cast(pl.Time),
        pl.col("equity_cutoff_index").cast(pl.Int16),
        pl.col("context_cutoff_index").cast(pl.Int16),
        pl.col("active_equity_count").cast(pl.Int16),
        pl.col("valid_label_count_30").cast(pl.Int16),
        pl.col("valid_label_count_60").cast(pl.Int16),
        pl.col("valid_label_count_120").cast(pl.Int16),
    )


def _validate_output(
    arrays: dict[str, np.memmap],
    assignments: pl.DataFrame,
    context_index: pl.DataFrame,
    context_expiries: dict[str, object],
    sample_index: pl.DataFrame,
) -> None:
    date_count = arrays["equity_features.npy"].shape[0]
    if assignments.height != EXPECTED_EQUITIES:
        raise ValueError(f"Expected {EXPECTED_EQUITIES} equity slots")
    if tuple(context_index.get_column("symbol")) != CONTEXT_SYMBOLS:
        raise ValueError("Context axis does not match the required order")
    if len(DECISION_EQUITY_INDICES) != 55 or len(HORIZONS) != 3:
        raise ValueError("Decision or horizon axis has the wrong cardinality")
    for filename, spec in output_array_specs(date_count).items():
        array = arrays[filename]
        if array.shape != spec.shape or array.dtype != spec.dtype:
            raise ValueError(
                f"Output contract mismatch for {filename}: {array.shape} {array.dtype}"
            )

    for filename in (
        "equity_features.npy",
        "equity_slow.npy",
        "context_features.npy",
        "context_slow.npy",
        "raw_returns.npy",
        "targets.npy",
        "cross_section_median.npy",
    ):
        for date_idx in range(date_count):
            if not np.isfinite(arrays[filename][date_idx]).all():
                raise ValueError(f"Non-finite value in {filename} at date {date_idx}")

    if np.any(arrays["context_features.npy"][..., 16:26] != 0):
        raise ValueError("Context cross-sectional dynamic channels must be zero")
    if np.any(arrays["equity_slow.npy"][..., 30:32] != 0):
        raise ValueError("Equity DI-only slow channels must be zero")
    if np.any(arrays["context_slow.npy"][..., 13:15] != 0):
        raise ValueError("Context equity-dollar-volume channels must be zero")
    if np.any(arrays["context_slow.npy"][..., 17:26] != 0):
        raise ValueError("Context rank and beta channels must be zero")
    if np.any(arrays["context_slow.npy"][:, :2, 30:32] != 0):
        raise ValueError("WIN/WDO DI-only slow channels must be zero")

    for date_idx in range(date_count):
        label_mask = arrays["label_mask.npy"][date_idx]
        targets = arrays["targets.npy"][date_idx]
        if np.any(targets[~label_mask] != 0):
            raise ValueError(f"Nonzero masked target at date {date_idx}")
        if np.any(arrays["raw_returns.npy"][date_idx][~label_mask] != 0):
            raise ValueError(f"Nonzero masked raw return at date {date_idx}")
        valid_targets = targets[label_mask]
        if valid_targets.size and (
            valid_targets.min() <= -1.0 or valid_targets.max() >= 1.0
        ):
            raise ValueError(f"Rank target outside (-1, 1) at date {date_idx}")
        counts = label_mask.sum(axis=0)
        means = np.divide(
            (targets * label_mask).sum(axis=0, dtype=np.float64),
            counts,
            out=np.zeros_like(counts, dtype=np.float64),
            where=counts > 0,
        )
        if np.any(np.abs(means[counts > 0]) > 2e-6):
            raise ValueError(
                f"Rank target cross-section is not centered at date {date_idx}"
            )
        if not np.array_equal(
            counts >= MIN_ACTIVE_EQUITIES, arrays["horizon_mask.npy"][date_idx]
        ):
            raise ValueError(f"Horizon mask is inconsistent at date {date_idx}")
        observed = arrays["equity_features.npy"][date_idx, :, :, 5].astype(bool)
        entry = observed[:, DECISION_EQUITY_INDICES]
        exits = np.stack(
            [
                observed[:, np.asarray(DECISION_EQUITY_INDICES) + horizon - 1]
                for horizon in HORIZONS
            ],
            axis=2,
        )
        required = (
            arrays["equity_membership.npy"][date_idx, :, None, None]
            & arrays["equity_data_ready.npy"][date_idx, :, None, None]
            & entry[:, :, None]
            & exits
            & arrays["horizon_mask.npy"][date_idx, None, :, :]
        )
        if np.any(label_mask & ~required):
            raise ValueError(f"Inconsistent true label mask at date {date_idx}")

    sample_dates = sample_index.get_column("date_idx").to_numpy()
    if not arrays["context_data_ready.npy"][sample_dates].all():
        raise ValueError("Sample index includes a date with unavailable context")
    active_counts = (
        arrays["equity_membership.npy"][sample_dates]
        & arrays["equity_data_ready.npy"][sample_dates]
    ).sum(axis=1)
    if np.any(active_counts < MIN_ACTIVE_EQUITIES):
        raise ValueError("Sample index includes fewer than 30 active equities")
    if np.any(
        active_counts != sample_index.get_column("active_equity_count").to_numpy()
    ):
        raise ValueError("Sample-index active equity count is inconsistent")
    if len(context_expiries) != len(RATE_CONTEXT_SYMBOLS):
        raise ValueError("Not every fixed-DI expiry was resolved")
    if sample_index.is_empty():
        raise ValueError("No feature-eligible sample exists")


def _write_feature_schema(output_dir: Path) -> None:
    schema = {
        "contract_version": CONTRACT_VERSION,
        "dynamic_channels": [
            {"index": index, "name": name}
            for index, name in enumerate(DYNAMIC_CHANNELS)
        ],
        "slow_channels": [
            {"index": index, "name": name}
            for index, name in enumerate(EQUITY_SLOW_CHANNELS)
        ],
        "dynamic_semantics": {
            "0:5": "Existing causal normalized OHLC moves, robust volume surprise, and observed flag.",
            "6:16": "Causal instrument returns, realized volatility, cumulative volume, range, and source-quality state through the current minute.",
            "16:26": "Point-in-time equity leave-one-out market summaries and centered cross-sectional midranks; exactly zero for context instruments.",
        },
        "slow_semantics": {
            "0:17": "Current-open-only gap plus completed-session volatility, return, liquidity, and quality state.",
            "17:20": "Point-in-time equity centered midranks; exactly zero for context instruments.",
            "20:26": "Causal equity betas to the six existing contexts; exactly zero for context instruments.",
            "26:30": "Deterministic current-date calendar encodings.",
            "30:32": "Existing DI prior-rate and expiry fields; zero for equities, WIN, and WDO.",
        },
        "family_inapplicable_zero_fields": {
            "equity": [30, 31],
            "all_context": list(range(13, 15)) + list(range(17, 26)),
            "WIN_WDO": [30, 31],
            "DI": list(range(20, 26)),
            "context_dynamic": list(range(16, 26)),
        },
        "grids": {
            "equity": {
                "timezone": "America/Sao_Paulo",
                "interval": "[10:00,16:45)",
                "minutes": EQUITY_SESSION_MINUTES,
            },
            "context": {
                "timezone": "America/Sao_Paulo",
                "interval": "[09:00,16:45)",
                "minutes": CONTEXT_SESSION_MINUTES,
            },
        },
        "decisions": {
            "times": [value.isoformat() for value in DECISION_TIMES],
            "equity_cutoff_indices": list(DECISION_EQUITY_INDICES),
            "context_cutoff_indices": list(DECISION_CONTEXT_INDICES),
            "input_rule": "indices strictly below the cutoff",
        },
        "horizons_minutes": list(HORIZONS),
        "base_target": "(log(exit_close / entry_open) - contemporaneous_cross_section_median) / (causal_equity_sigma * sqrt(horizon_minutes))",
        "stored_target": "2 * ((average_one_based_midrank - 0.5) / valid_cross_section_size) - 1",
        "target_grouping": "independently by date, decision, and horizon",
        "masks": {
            "equity_membership": "monthly point-in-time universe membership",
            "equity_data_ready": "accepted identity interval and prior volatility state; optional features never tighten readiness",
            "context_data_ready": "prior volatility state; DI also requires its existing prior-rate state",
            "label_mask": "membership, readiness, exact entry/exit endpoints, and a valid horizon cross-section",
            "horizon_mask": "at least 30 valid equity labels for the date, decision, and horizon",
        },
        "missingness": "Invalid continuous features are zero; observed, readiness, membership, patch masks, and quality channels carry availability.",
    }
    (output_dir / "feature_schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )


def _write_manifest(
    output_dir: Path,
    canonical_inputs: dict[str, dict[str, str]],
    assignments: pl.DataFrame,
    context_files: dict[str, Path],
    market_dates: tuple[object, ...],
    sample_count: int,
    first_feature_date: object,
    research_start: object,
    research_end: object,
    created_at: datetime,
    duration: float,
) -> None:
    outputs = {
        filename: {"shape": list(spec.shape), "dtype": spec.dtype.name}
        for filename, spec in output_array_specs(len(market_dates)).items()
    }
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "created_at_utc": created_at.isoformat(),
        "research_start": str(research_start),
        "research_end": str(research_end),
        "canonical_inputs": canonical_inputs,
        "accepted_assignment_count": assignments.height,
        "accepted_decision_package": canonical_inputs["accepted_xp_assignments"][
            "resolved_path"
        ],
        "equity_source_files": sorted(
            assignments.get_column("source_file").unique().to_list()
        ),
        "context_source_files": {
            symbol: str(context_files[symbol]) for symbol in CONTEXT_SYMBOLS
        },
        "context_symbols": list(CONTEXT_SYMBOLS),
        "constants": manifest_constants(),
        "outputs": outputs,
        "metadata_files": [
            "manifest.json",
            "feature_schema.json",
            "date_index.parquet",
            "equity_index.parquet",
            "context_index.parquet",
            "sample_index.parquet",
            "daily_audit.parquet",
            "security_audit.parquet",
        ],
        "date_count": len(market_dates),
        "sample_count": sample_count,
        "first_feature_eligible_date": str(first_feature_date),
        "last_date": str(market_dates[-1]),
        "build_duration_seconds": duration,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _promote(output_dir: Path) -> None:
    temporary_pointer = CANONICAL_OUTPUT_POINTER.with_name(
        f"{CANONICAL_OUTPUT_POINTER.name}.tmp"
    )
    temporary_pointer.write_text(str(output_dir), encoding="utf-8")
    temporary_pointer.replace(CANONICAL_OUTPUT_POINTER)


if __name__ == "__main__":
    main()
