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
    detect_cotahist_actions,
    provider_split_detection_audit,
)

AUDIT_SCHEMA = "BRAZIL_RV_V2_CORPORATE_ACTION_RECLASSIFICATION_AUDIT_V1"


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


def build_reclassification_audit(
    *,
    dates: NDArray[np.datetime64],
    isins: Sequence[str],
    raw_close: NDArray[np.floating],
    quantity: NDArray[np.floating],
    distribution_number: NDArray[np.floating],
    observed: NDArray[np.bool_],
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
                pl.col("valid_target_name_days")
                / pl.col("observed_member_name_days")
            ).alias("validity_ratio")
        )
        .sort("horizon_sessions")
    )
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
            "undocumented_split_fallback": adopt_fallback,
            "classifier": (
                "dismes_plus_strict_fallback" if adopt_fallback else "dismes_only"
            ),
        },
        "classification_counts_by_year": {
            "dismes_only": _classification_counts(dates, without_fallback),
            "dismes_plus_strict_fallback": _classification_counts(
                dates, with_fallback
            ),
        },
        "defect_record": {
            "old_detected_cash_event_count": old_cash_count,
            "old_cash_events_without_dismes_change": undocumented_old_cash,
            "old_cash_events_without_dismes_change_fraction": (
                undocumented_old_cash / old_cash_count if old_cash_count else None
            ),
            "old_target_validity_by_horizon": target_totals.to_dicts(),
        },
    }


def audit_store(source_store: Path, output: Path) -> str:
    root = source_store.resolve()
    required = (
        "date_index.npy",
        "isin_index.npy",
        "raw_close.npy",
        "quantity.npy",
        "distribution_number.npy",
        "observed.npy",
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
    recorded_manifest_sha = manifest_sha_path.read_text(
        encoding="ascii"
    ).split()[0]
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
        distribution_number=np.load(
            root / "distribution_number.npy", mmap_mode="r"
        ),
        observed=np.load(root / "observed.npy", mmap_mode="r"),
        old_cash_event=np.load(
            root / "detected_cash_event_mask.npy", mmap_mode="r"
        ),
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
    return write_json_atomic(output, payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the fix-pass-3 COTAHIST classifier on an accepted store"
    )
    parser.add_argument("--source-store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    audit_store(arguments.source_store, arguments.output)


if __name__ == "__main__":
    main()
