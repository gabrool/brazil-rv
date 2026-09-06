from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl
from numpy.typing import NDArray

from .artifacts import sha256_file, write_json_atomic
from .corporate_actions import (
    DetectedActionResult,
    InferredActionResult,
    detect_cotahist_actions,
    infer_cotahist_action_terms,
    provider_split_detection_audit,
)

AUDIT_SCHEMA = "BRAZIL_RV_V2_CORPORATE_ACTION_RECLASSIFICATION_AUDIT_V3"
BREAKDOWN_SCHEMA = "BRAZIL_RV_V2_CORPORATE_ACTION_RECLASSIFICATION_BREAKDOWN_V2"


def _finite(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _classification_counts(
    dates: NDArray[np.datetime64], result: DetectedActionResult
) -> list[dict[str, object]]:
    years = dates.astype("datetime64[Y]").astype(np.int64) + 1970
    rows: list[dict[str, object]] = []
    for year in sorted(set(years.tolist())):
        selected = years == year
        rows.append(
            {
                "year": int(year),
                "split_count": int(result.split_event[selected].sum()),
                "cash_count": int(result.cash_event[selected].sum()),
                "ambiguous_count": int(result.ambiguous_event[selected].sum()),
                "jump_only_anomaly_count": int(
                    result.price_jump_anomaly_mask[selected].sum()
                ),
            }
        )
    return rows


def _audit_rows(table: pl.DataFrame) -> list[dict[str, object]]:
    return [
        {
            **row,
            "precision": _finite(row["precision"]),
            "recall": _finite(row["recall"]),
        }
        for row in table.to_dicts()
    ]


def _nearest_offset(
    mask: NDArray[np.bool_], date_index: int, name_index: int, radius: int
) -> int | None:
    """Return the nearest session offset, preferring the earlier row on ties."""

    start = max(0, date_index - radius)
    stop = min(mask.shape[0], date_index + radius + 1)
    candidates = np.flatnonzero(mask[start:stop, name_index]) + start - date_index
    if not candidates.size:
        return None
    return int(min(candidates.tolist(), key=lambda value: (abs(value), value)))


def _provider_coverage_mask(
    dates: NDArray[np.datetime64],
    isins: Sequence[str],
    provider_actions: pl.DataFrame,
    acquisition_audit: pl.DataFrame,
) -> NDArray[np.bool_]:
    covered = np.zeros((dates.size, len(isins)), dtype=np.bool_)
    lookup = {str(isin): index for index, isin in enumerate(isins)}
    required = {"isin", "first_date", "last_date", "status"}
    if not required.issubset(acquisition_audit.columns):
        raise ValueError("corporate-action acquisition audit has wrong schema")
    for row in acquisition_audit.filter(pl.col("status") != "failed").iter_rows(
        named=True
    ):
        name_index = lookup.get(str(row["isin"]))
        if name_index is None:
            continue
        first = np.datetime64(row["first_date"], "D")
        last = np.datetime64(row["last_date"], "D")
        covered[:, name_index] |= (dates >= first) & (dates <= last)
    if acquisition_audit.is_empty():
        for isin in provider_actions.get_column("isin").unique().to_list():
            name_index = lookup.get(str(isin))
            if name_index is not None:
                covered[:, name_index] = True
    return covered


def _matched_precision_recall(
    actual: NDArray[np.bool_],
    predicted: NDArray[np.bool_],
    covered: NDArray[np.bool_],
    *,
    radius: int,
) -> dict[str, object]:
    actual_rows = np.argwhere(actual & covered)
    predicted_rows = np.argwhere(predicted & covered)
    matched_actual = sum(
        _nearest_offset(predicted, int(day), int(name), radius) is not None
        for day, name in actual_rows
    )
    matched_predicted = sum(
        _nearest_offset(actual, int(day), int(name), radius) is not None
        for day, name in predicted_rows
    )
    return {
        "matching_tolerance_sessions": radius,
        "provider_split_count": int(actual_rows.shape[0]),
        "detected_split_count": int(predicted_rows.shape[0]),
        "matched_provider_split_count": int(matched_actual),
        "matched_detected_split_count": int(matched_predicted),
        "recall": (
            matched_actual / int(actual_rows.shape[0]) if actual_rows.size else None
        ),
        "precision": (
            matched_predicted / int(predicted_rows.shape[0])
            if predicted_rows.size
            else None
        ),
    }


def _ratio_bucket(log_factor: float) -> str:
    magnitude = abs(log_factor)
    if magnitude <= 0.04:
        return "abs_log_factor_le_0p04"
    if magnitude <= 0.08:
        return "abs_log_factor_0p04_to_0p08"
    if magnitude <= 0.30:
        return "abs_log_factor_0p08_to_0p30"
    return "abs_log_factor_gt_0p30"


def _offset_counts(
    rows: Sequence[dict[str, object]], field: str
) -> list[dict[str, object]]:
    counts: dict[int | None, int] = {}
    for row in rows:
        value = row[field]
        offset = None if value is None else int(value)
        counts[offset] = counts.get(offset, 0) + 1
    return [
        {"offset_sessions": offset, "count": count}
        for offset, count in sorted(
            counts.items(), key=lambda item: (item[0] is None, item[0] or 0)
        )
    ]


def _build_breakdown(
    *,
    dates: NDArray[np.datetime64],
    isins: Sequence[str],
    provider_actions: pl.DataFrame,
    acquisition_audit: pl.DataFrame,
    distribution_change: NDArray[np.bool_],
    without_fallback: DetectedActionResult,
    with_fallback: DetectedActionResult,
    inferred: InferredActionResult,
) -> dict[str, Any]:
    from .corporate_actions import align_action_arrays

    provider_factor, _, _ = align_action_arrays(provider_actions, dates, isins)
    provider_split = provider_factor != 1.0
    covered = _provider_coverage_mask(dates, isins, provider_actions, acquisition_audit)
    immediate_jump = np.isfinite(without_fallback.price_ratio) & (
        np.abs(np.log(without_fallback.price_ratio)) >= 0.04
    )
    immediate_large_jump = np.isfinite(without_fallback.price_ratio) & (
        np.abs(np.log(without_fallback.price_ratio)) >= 0.30
    )
    provider_rows: list[dict[str, object]] = []
    for day, name in np.argwhere(provider_split & covered):
        factor = float(provider_factor[day, name])
        log_factor = float(np.log(factor))
        price_ratio = float(without_fallback.price_ratio[day, name])
        quantity_ratio = float(without_fallback.quantity_ratio[day, name])
        log_price_ratio = (
            float(np.log(price_ratio))
            if np.isfinite(price_ratio) and price_ratio > 0.0
            else None
        )
        log_quantity_ratio = (
            float(np.log(quantity_ratio))
            if np.isfinite(quantity_ratio) and quantity_ratio > 0.0
            else None
        )
        dismes_offset = _nearest_offset(distribution_change, int(day), int(name), 3)
        jump_offset = _nearest_offset(immediate_jump, int(day), int(name), 3)
        dismes_day = None if dismes_offset is None else int(day + dismes_offset)
        if dismes_day is None:
            dismes_log_price = dismes_log_quantity = None
        else:
            dismes_price_ratio = float(without_fallback.price_ratio[dismes_day, name])
            dismes_quantity_ratio = float(
                without_fallback.quantity_ratio[dismes_day, name]
            )
            dismes_log_price = (
                float(np.log(dismes_price_ratio))
                if np.isfinite(dismes_price_ratio) and dismes_price_ratio > 0.0
                else None
            )
            dismes_log_quantity = (
                float(np.log(dismes_quantity_ratio))
                if np.isfinite(dismes_quantity_ratio) and dismes_quantity_ratio > 0.0
                else None
            )
        provider_rows.append(
            {
                "isin": str(isins[name]),
                "provider_date": str(dates[day]),
                "provider_factor": factor,
                "log_provider_factor": log_factor,
                "ratio_bucket": _ratio_bucket(log_factor),
                "nearest_dismes_offset_sessions": dismes_offset,
                "nearest_abs_log_return_ge_0p04_offset_sessions": jump_offset,
                "provider_date_log_price_ratio": log_price_ratio,
                "provider_date_log_quantity_ratio": log_quantity_ratio,
                "provider_date_log_ratio_sum": _finite(
                    log_price_ratio + log_quantity_ratio
                    if log_price_ratio is not None and log_quantity_ratio is not None
                    else None
                ),
                "dismes_date": (
                    str(dates[dismes_day]) if dismes_day is not None else None
                ),
                "dismes_date_log_price_ratio": dismes_log_price,
                "dismes_date_log_quantity_ratio": dismes_log_quantity,
                "dismes_date_log_ratio_sum": (
                    dismes_log_price + dismes_log_quantity
                    if dismes_log_price is not None and dismes_log_quantity is not None
                    else None
                ),
                "dismes_only_split_within_2_sessions": _nearest_offset(
                    without_fallback.split_event, int(day), int(name), 2
                )
                is not None,
                "fallback_split_within_2_sessions": _nearest_offset(
                    with_fallback.split_event, int(day), int(name), 2
                )
                is not None,
                "price_jump_precedes_dismes_by_1_or_2_sessions": (
                    dismes_offset is not None
                    and jump_offset is not None
                    and jump_offset - dismes_offset in (-2, -1)
                ),
                "price_corroborated_abs_log_ge_0p30_within_2": _nearest_offset(
                    immediate_large_jump, int(day), int(name), 2
                )
                is not None,
                "dismes_within_2_of_provider_date": _nearest_offset(
                    distribution_change, int(day), int(name), 2
                )
                is not None,
                "u2_per_trade_hit_within_2": _nearest_offset(
                    inferred.u2_per_trade_candidate, int(day), int(name), 2
                )
                is not None,
                "u2_total_quantity_hit_within_2": _nearest_offset(
                    inferred.u2_total_quantity_candidate, int(day), int(name), 2
                )
                is not None,
            }
        )

    detected_rows: list[dict[str, object]] = []
    for day, name in np.argwhere(without_fallback.split_event & covered):
        provider_offset = _nearest_offset(provider_split, int(day), int(name), 2)
        price_ratio = float(without_fallback.price_ratio[day, name])
        quantity_ratio = float(without_fallback.quantity_ratio[day, name])
        log_price_ratio = (
            float(np.log(price_ratio))
            if np.isfinite(price_ratio) and price_ratio > 0.0
            else None
        )
        log_quantity_ratio = (
            float(np.log(quantity_ratio))
            if np.isfinite(quantity_ratio) and quantity_ratio > 0.0
            else None
        )
        detected_rows.append(
            {
                "isin": str(isins[name]),
                "detected_date": str(dates[day]),
                "nearest_provider_split_offset_sessions": provider_offset,
                "log_price_ratio": log_price_ratio,
                "log_quantity_ratio": log_quantity_ratio,
                "abs_log_price_ratio": (
                    abs(log_price_ratio) if log_price_ratio is not None else None
                ),
            }
        )
    unmatched = [
        row
        for row in detected_rows
        if row["nearest_provider_split_offset_sessions"] is None
    ]
    unmatched.sort(
        key=lambda row: float(row["abs_log_price_ratio"] or 0.0), reverse=True
    )

    bucket_rows: list[dict[str, object]] = []
    for bucket in (
        "abs_log_factor_le_0p04",
        "abs_log_factor_0p04_to_0p08",
        "abs_log_factor_0p08_to_0p30",
        "abs_log_factor_gt_0p30",
    ):
        selected = [row for row in provider_rows if row["ratio_bucket"] == bucket]
        bucket_rows.append(
            {
                "ratio_bucket": bucket,
                "provider_split_count": len(selected),
                "dismes_within_2_fraction": (
                    sum(
                        row["nearest_dismes_offset_sessions"] is not None
                        and abs(int(row["nearest_dismes_offset_sessions"])) <= 2
                        for row in selected
                    )
                    / len(selected)
                    if selected
                    else None
                ),
                "dismes_only_detection_within_2_fraction": (
                    sum(
                        bool(row["dismes_only_split_within_2_sessions"])
                        for row in selected
                    )
                    / len(selected)
                    if selected
                    else None
                ),
                "fallback_detection_within_2_fraction": (
                    sum(
                        bool(row["fallback_split_within_2_sessions"])
                        for row in selected
                    )
                    / len(selected)
                    if selected
                    else None
                ),
            }
        )

    low_rows = [
        row for row in provider_rows if abs(float(row["log_provider_factor"])) <= 0.08
    ]
    high_rows = [
        row for row in provider_rows if abs(float(row["log_provider_factor"])) > 0.08
    ]
    very_large = [
        row for row in provider_rows if abs(float(row["log_provider_factor"])) > 0.30
    ]
    corroborated_very_large = [
        row
        for row in very_large
        if bool(row["price_corroborated_abs_log_ge_0p30_within_2"])
    ]
    low_dismes = (
        sum(
            row["nearest_dismes_offset_sessions"] is not None
            and abs(int(row["nearest_dismes_offset_sessions"])) <= 2
            for row in low_rows
        )
        / len(low_rows)
        if low_rows
        else None
    )
    high_detected = (
        sum(bool(row["dismes_only_split_within_2_sessions"]) for row in high_rows)
        / len(high_rows)
        if high_rows
        else None
    )
    lagged_large = (
        sum(
            bool(row["price_jump_precedes_dismes_by_1_or_2_sessions"])
            for row in very_large
        )
        / len(very_large)
        if very_large
        else None
    )
    dimmes_metrics = _matched_precision_recall(
        provider_split, without_fallback.split_event, covered, radius=2
    )
    fallback_metrics = _matched_precision_recall(
        provider_split, with_fallback.split_event, covered, radius=2
    )
    stop_reasons: list[str] = []
    if low_dismes is not None and low_dismes < 0.90:
        stop_reasons.append("low_factor_provider_rows_lack_dismes_coverage")
    legacy_enable_fallback = bool(lagged_large is not None and lagged_large >= 0.10)
    selected_metrics = fallback_metrics if legacy_enable_fallback else dimmes_metrics
    if (
        selected_metrics["precision"] is None
        or float(selected_metrics["precision"]) < 0.70
    ):
        stop_reasons.append("plus_or_minus_2_precision_below_0p70")
    return {
        "schema": BREAKDOWN_SCHEMA,
        "purpose": (
            "coverage and timing audit only; realized-price ratios never authorize "
            "canonical action units or economic adjustments"
        ),
        "provider_splits": provider_rows,
        "dismes_only_detections": detected_rows,
        "largest_unmatched_dismes_only_detections": unmatched[:30],
        "ratio_bucket_summary": bucket_rows,
        "dismes_offset_distribution": _offset_counts(
            provider_rows, "nearest_dismes_offset_sessions"
        ),
        "price_jump_offset_distribution": _offset_counts(
            provider_rows, "nearest_abs_log_return_ge_0p04_offset_sessions"
        ),
        "plus_or_minus_2_metrics": {
            "dismes_only": dimmes_metrics,
            "dismes_plus_strict_fallback": fallback_metrics,
        },
        "decision_inputs": {
            "low_factor_count": len(low_rows),
            "low_factor_dismes_within_2_fraction": low_dismes,
            "above_0p08_count": len(high_rows),
            "above_0p08_dismes_only_detection_within_2_fraction": high_detected,
            "above_0p30_count": len(very_large),
            "above_0p30_jump_precedes_dismes_fraction": lagged_large,
        },
        "development_inference_quality": {
            "audit_only_not_a_gate": True,
            "provider_abs_log_factor_gt_0p30_count": len(very_large),
            "price_corroborated_count": len(corroborated_very_large),
            "price_corroborated_fraction": (
                len(corroborated_very_large) / len(very_large) if very_large else None
            ),
            "provider_noise_excluded_from_recall_count": (
                len(very_large) - len(corroborated_very_large)
            ),
            "corroborated_dismes_within_2_fraction": (
                sum(
                    bool(row["dismes_within_2_of_provider_date"])
                    for row in corroborated_very_large
                )
                / len(corroborated_very_large)
                if corroborated_very_large
                else None
            ),
            "corroborated_u2_per_trade_hit_fraction": (
                sum(
                    bool(row["u2_per_trade_hit_within_2"])
                    for row in corroborated_very_large
                )
                / len(corroborated_very_large)
                if corroborated_very_large
                else None
            ),
            "corroborated_u2_total_quantity_hit_fraction": (
                sum(
                    bool(row["u2_total_quantity_hit_within_2"])
                    for row in corroborated_very_large
                )
                / len(corroborated_very_large)
                if corroborated_very_large
                else None
            ),
        },
        "decision": {
            "legacy_strict_fallback_would_be_enabled": legacy_enable_fallback,
            "legacy_classifier_stop_reasons": stop_reasons,
            "legacy_classifier_gate_passed": not stop_reasons,
            "canonical_price_ratio_adjustment_authorized": False,
            "u2_development_inference_enabled": True,
            "development_inference_quality_is_gate": False,
            "canonical_requirement": (
                "verified contractual share and cash terms; otherwise affected "
                "cross-boundary outcomes remain unresolved"
            ),
        },
    }


def build_reclassification_audit(
    *,
    dates: NDArray[np.datetime64],
    isins: Sequence[str],
    raw_close: NDArray[np.floating],
    quantity: NDArray[np.floating],
    trades: NDArray[np.floating],
    distribution_number: NDArray[np.floating],
    observed: NDArray[np.bool_],
    active: NDArray[np.bool_],
    old_cash_event: NDArray[np.bool_],
    old_distribution_change: NDArray[np.bool_],
    provider_actions: pl.DataFrame,
    acquisition_audit: pl.DataFrame,
    old_target_validity_by_year: pl.DataFrame,
) -> dict[str, Any]:
    """Compare both causal classifiers and apply the frozen fallback rule."""

    without_fallback = detect_cotahist_actions(
        raw_close,
        quantity,
        distribution_number,
        observed,
        undocumented_split_fallback=False,
    )
    with_fallback = detect_cotahist_actions(
        raw_close,
        quantity,
        distribution_number,
        observed,
        undocumented_split_fallback=True,
    )
    inferred = infer_cotahist_action_terms(
        dates,
        isins,
        raw_close,
        quantity,
        trades,
        distribution_number,
        observed,
        active,
    )
    audits = {
        "dismes_only": provider_split_detection_audit(
            dates,
            isins,
            without_fallback.split_event,
            provider_actions,
            acquisition_audit,
        ),
        "dismes_plus_strict_fallback": provider_split_detection_audit(
            dates,
            isins,
            with_fallback.split_event,
            provider_actions,
            acquisition_audit,
        ),
    }
    headline: dict[str, dict[str, object]] = {}
    for name, table in audits.items():
        row = table.filter(pl.col("period") == "all").to_dicts()[0]
        headline[name] = {
            "covered_name_days": int(row["covered_name_days"]),
            "detected_split_count": int(row["detected_split_count"]),
            "provider_split_count": int(row["provider_split_count"]),
            "true_positive_count": int(row["true_positive_count"]),
            "precision": _finite(row["precision"]),
            "recall": _finite(row["recall"]),
        }
    base_precision = headline["dismes_only"]["precision"]
    fallback_precision = headline["dismes_plus_strict_fallback"]["precision"]
    base_recall = headline["dismes_only"]["recall"]
    fallback_recall = headline["dismes_plus_strict_fallback"]["recall"]
    if any(
        value is None
        for value in (
            base_precision,
            fallback_precision,
            base_recall,
            fallback_recall,
        )
    ):
        adopt_fallback = False
        recall_gain = None
        precision_loss = None
    else:
        recall_gain = float(fallback_recall) - float(base_recall)
        precision_loss = float(base_precision) - float(fallback_precision)
        adopt_fallback = recall_gain >= 0.01 and precision_loss <= 0.01

    old_cash = np.asarray(old_cash_event, dtype=np.bool_)
    old_distribution = np.asarray(old_distribution_change, dtype=np.bool_)
    if old_cash.shape != old_distribution.shape:
        raise ValueError("old action masks are misaligned")
    old_cash_count = int(old_cash.sum())
    undocumented_old_cash = int((old_cash & ~old_distribution).sum())
    target_totals = (
        old_target_validity_by_year.group_by("horizon_sessions")
        .agg(
            pl.col("valid_target_name_days").sum(),
            pl.col("observed_member_name_days").sum(),
        )
        .with_columns(
            (
                pl.col("valid_target_name_days") / pl.col("observed_member_name_days")
            ).alias("validity_ratio")
        )
        .sort("horizon_sessions")
    )
    breakdown = _build_breakdown(
        dates=np.asarray(dates, dtype="datetime64[D]"),
        isins=isins,
        provider_actions=provider_actions,
        acquisition_audit=acquisition_audit,
        distribution_change=old_distribution,
        without_fallback=without_fallback,
        with_fallback=with_fallback,
        inferred=inferred,
    )
    years = np.asarray(dates, dtype="datetime64[Y]").astype(np.int64) + 1970
    inferred_counts = [
        {
            "year": int(year),
            "u1_count": int(inferred.u1_event[years == year].sum()),
            "c1_count": int(inferred.c1_event[years == year].sum()),
            "u2_count": int(inferred.u2_event[years == year].sum()),
            "large_move_no_action_count": int(
                inferred.large_move_no_action[years == year].sum()
            ),
        }
        for year in sorted(set(years.tolist()))
    ]
    return {
        "schema": AUDIT_SCHEMA,
        "research_claim": False,
        "source_store_immutable": True,
        "decision_rule": {
            "adopt_strict_fallback_if_recall_gain_at_least": 0.01,
            "and_precision_loss_at_most": 0.01,
        },
        "provider_split_comparison": {
            "headline": headline,
            "by_period": {name: _audit_rows(table) for name, table in audits.items()},
            "recall_gain": recall_gain,
            "precision_loss": precision_loss,
        },
        "decision": {
            "legacy_undocumented_split_fallback": adopt_fallback,
            "legacy_classifier": (
                "dismes_plus_strict_fallback" if adopt_fallback else "dismes_only"
            ),
            "canonical_price_ratio_adjustment_authorized": False,
            "canonical_action_contract": (
                "verified contractual share/cash terms; unresolved otherwise"
            ),
        },
        "classification_counts_by_year": {
            "dismes_only": _classification_counts(dates, without_fallback),
            "dismes_plus_strict_fallback": _classification_counts(dates, with_fallback),
        },
        "development_inferred_action_counts_by_year": inferred_counts,
        "defect_record": {
            "old_detected_cash_event_count": old_cash_count,
            "old_cash_events_without_dismes_change": undocumented_old_cash,
            "old_cash_events_without_dismes_change_fraction": (
                undocumented_old_cash / old_cash_count if old_cash_count else None
            ),
            "old_target_validity_by_horizon": target_totals.to_dicts(),
        },
        "corporate_action_reclassification_breakdown": breakdown,
    }


def audit_store(
    source_store: Path,
    output: Path,
    *,
    breakdown_output: Path | None = None,
    implementation_commit: str | None = None,
) -> str:
    root = source_store.resolve()
    required = (
        "date_index.npy",
        "isin_index.npy",
        "raw_close.npy",
        "quantity.npy",
        "trade_count.npy",
        "distribution_number.npy",
        "observed.npy",
        "active.npy",
        "detected_cash_event_mask.npy",
        "distribution_change_mask.npy",
        "corporate_actions.parquet",
        "corporate_action_acquisition_audit.parquet",
        "target_validity_by_year.parquet",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"accepted store audit inputs missing: {missing}")
    manifest_path = root / "manifest.json"
    manifest_sha_path = root / "manifest.sha256"
    if not manifest_path.is_file() or not manifest_sha_path.is_file():
        raise FileNotFoundError("accepted store lacks its immutable manifest binding")
    manifest_sha = sha256_file(manifest_path)
    recorded_manifest_sha = manifest_sha_path.read_text(encoding="ascii").split()[0]
    if manifest_sha != recorded_manifest_sha:
        raise ValueError("accepted store manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: dict[str, dict[str, object]] = {}
    for section in ("indices", "arrays", "tables"):
        for name, record in manifest.get(section, {}).items():
            filename = str(record.get("path", name))
            records[filename] = record
    verified: dict[str, str] = {}
    for filename in required:
        record = records.get(filename)
        path = root / filename
        if record is None:
            raise ValueError(f"accepted store manifest omits audit input: {filename}")
        actual_sha = sha256_file(path)
        if (
            int(record.get("bytes", -1)) != path.stat().st_size
            or str(record.get("sha256", "")) != actual_sha
        ):
            raise ValueError(f"accepted store audit input hash mismatch: {filename}")
        verified[filename] = actual_sha
    payload = build_reclassification_audit(
        dates=np.load(root / "date_index.npy", mmap_mode="r"),
        isins=tuple(str(value) for value in np.load(root / "isin_index.npy")),
        raw_close=np.load(root / "raw_close.npy", mmap_mode="r"),
        quantity=np.load(root / "quantity.npy", mmap_mode="r"),
        trades=np.load(root / "trade_count.npy", mmap_mode="r"),
        distribution_number=np.load(root / "distribution_number.npy", mmap_mode="r"),
        observed=np.load(root / "observed.npy", mmap_mode="r"),
        active=np.load(root / "active.npy", mmap_mode="r"),
        old_cash_event=np.load(root / "detected_cash_event_mask.npy", mmap_mode="r"),
        old_distribution_change=np.load(
            root / "distribution_change_mask.npy", mmap_mode="r"
        ),
        provider_actions=pl.read_parquet(root / "corporate_actions.parquet"),
        acquisition_audit=pl.read_parquet(
            root / "corporate_action_acquisition_audit.parquet"
        ),
        old_target_validity_by_year=pl.read_parquet(
            root / "target_validity_by_year.parquet"
        ),
    )
    payload["source_store"] = {
        "root": str(root),
        "manifest_sha256": manifest_sha,
        "verified_input_sha256s": verified,
    }
    payload["implementation_git_commit"] = implementation_commit
    breakdown = payload["corporate_action_reclassification_breakdown"]
    if not isinstance(breakdown, dict):
        raise TypeError("corporate-action breakdown was not materialized")
    breakdown["source_store"] = payload["source_store"]
    breakdown["implementation_git_commit"] = implementation_commit
    breakdown_path = (
        output.with_name("corporate_action_reclassification_breakdown.json")
        if breakdown_output is None
        else breakdown_output
    )
    breakdown_sha256 = write_json_atomic(breakdown_path, breakdown)
    payload["breakdown_artifact"] = {
        "path": str(breakdown_path.resolve()),
        "sha256": breakdown_sha256,
    }
    return write_json_atomic(output, payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the fix-pass-3 COTAHIST classifier on an accepted store"
    )
    parser.add_argument("--source-store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--breakdown-output", type=Path)
    parser.add_argument("--implementation-commit", required=True)
    arguments = parser.parse_args()
    audit_store(
        arguments.source_store,
        arguments.output,
        breakdown_output=arguments.breakdown_output,
        implementation_commit=arguments.implementation_commit,
    )


if __name__ == "__main__":
    main()
