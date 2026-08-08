from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from brazil_rv.preprocessing.contract import GLOBAL_SLOW_CHANNELS, SLOW_CHANNELS

from .context_ablation import resolve_context_ablation_for_store
from .contract import (
    GLOBAL_CONTEXT_SYMBOLS,
    LOCAL_CONTEXT_SYMBOLS,
    TRAIN_END,
    TRAIN_START,
)
from .data import resolve_feature_store, select_sample_split, validate_feature_store
from .feature_ablation import resolve_feature_ablation_for_store
from .stage2_context_ablation import _feature_store_identity

AUDIT_NAME = "stage4_training_slow_feature_redundancy"
AUDIT_VERSION = 1
FROZEN_CONTEXT_ABLATION = "drop_win_and_global_non_rates"
EXPECTED_RETAINED_CONTEXTS = (
    "WDO$",
    "DI1F27",
    "DI1F28",
    "DI1F29",
    "DI1F31",
    "DI1$N",
    "ZT.v.0",
    "ZN.v.0",
)
AUDIT_JSON = "slow_feature_audit.json"
STATS_CSV = "slow_feature_stats.csv"
PEARSON_CSV = "slow_feature_pearson.csv"
SPEARMAN_CSV = "slow_feature_spearman.csv"
FOCUSED_CSV = "slow_feature_focused_correlations.csv"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_csv(path: Path, frame: pl.DataFrame) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        frame.write_csv(temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _nullable_matrix(matrix: np.ndarray) -> list[list[float | None]]:
    return [
        [float(value) if np.isfinite(value) else None for value in row]
        for row in matrix
    ]


def compute_slow_statistics(
    values: np.ndarray,
    *,
    slow_names: tuple[str, ...] = SLOW_CHANNELS,
    removed_indices: tuple[int, ...],
) -> dict[str, object]:
    """Summarize applicable observations; inapplicable cells must be NaN."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(slow_names):
        raise ValueError("Slow-audit values have the wrong shape")
    if len(slow_names) != len(set(slow_names)) or slow_names != SLOW_CHANNELS:
        raise ValueError("Slow-audit channel axis is not canonical")
    if not set(removed_indices) <= set(range(len(slow_names))):
        raise ValueError("Slow-audit removal index is outside the canonical axis")

    finite = np.isfinite(array)
    statistics: list[dict[str, object]] = []
    for index, name in enumerate(slow_names):
        valid = array[finite[:, index], index]
        statistics.append(
            {
                "index": index,
                "name": name,
                "valid_observation_count": int(valid.size),
                "finite_fraction": float(valid.size / array.shape[0]),
                "exact_zero_fraction": (
                    float(np.mean(valid == 0.0)) if valid.size else None
                ),
                "mean": float(np.mean(valid)) if valid.size else None,
                "standard_deviation": (
                    float(np.std(valid, ddof=0)) if valid.size else None
                ),
            }
        )

    frame = pd.DataFrame(array, columns=slow_names)
    pearson = frame.corr(method="pearson", min_periods=2).to_numpy(dtype=np.float64)
    spearman = frame.corr(method="spearman", min_periods=2).to_numpy(dtype=np.float64)
    pair_counts = finite.astype(np.int64).T @ finite.astype(np.int64)
    retained = tuple(
        index for index in range(len(slow_names)) if index not in removed_indices
    )
    focused: list[dict[str, object]] = []
    for removed in removed_indices:
        for method, matrix in (("pearson", pearson), ("spearman", spearman)):
            candidates = [
                (retained_index, float(matrix[removed, retained_index]))
                for retained_index in retained
                if np.isfinite(matrix[removed, retained_index])
            ]
            candidates.sort(key=lambda item: (-abs(item[1]), item[0]))
            for rank, (retained_index, correlation) in enumerate(
                candidates[:5], start=1
            ):
                focused.append(
                    {
                        "removed_index": removed,
                        "removed_name": slow_names[removed],
                        "method": method,
                        "rank": rank,
                        "retained_index": retained_index,
                        "retained_name": slow_names[retained_index],
                        "correlation": correlation,
                        "absolute_correlation": abs(correlation),
                        "paired_valid_observation_count": int(
                            pair_counts[removed, retained_index]
                        ),
                    }
                )
    return {
        "statistics": statistics,
        "pearson": _nullable_matrix(pearson),
        "spearman": _nullable_matrix(spearman),
        "paired_valid_observation_counts": pair_counts.tolist(),
        "focused": focused,
    }


def _matrix_frame(
    matrix: list[list[float | None]], names: tuple[str, ...]
) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"index": index, "name": names[index], **dict(zip(names, row, strict=True))}
            for index, row in enumerate(matrix)
        ]
    )


def _indices(value: object, field: str) -> set[int]:
    if not isinstance(value, list) or any(
        not isinstance(index, int) or isinstance(index, bool) for index in value
    ):
        raise ValueError(f"Feature schema has invalid applicability field: {field}")
    return set(value)


def _applicable(width: int, inapplicable: set[int]) -> np.ndarray:
    if not inapplicable <= set(range(width)):
        raise ValueError("Feature schema has an out-of-range inapplicable channel")
    result = np.ones(width, dtype=bool)
    result[list(sorted(inapplicable))] = False
    return result


def _training_values(
    store: Path,
) -> tuple[np.ndarray, dict[str, object]]:
    sample_index = validate_feature_store(store)
    training = select_sample_split(sample_index, "train").sort("sample_id")
    if (
        training["trade_date"].min() != TRAIN_START
        or training["trade_date"].max() != TRAIN_END
    ):
        raise ValueError("Slow audit did not resolve the exact training period")
    date_indices = np.unique(training["date_idx"].to_numpy()).astype(np.int64)
    sample_mask = np.column_stack(
        [
            training[column].to_numpy().astype("<i8", copy=False)
            for column in ("sample_id", "date_idx", "decision_idx")
        ]
    )

    context = resolve_context_ablation_for_store(store, FROZEN_CONTEXT_ABLATION)
    retained_local = tuple(
        symbol
        for slot, symbol in enumerate(LOCAL_CONTEXT_SYMBOLS)
        if slot not in context.local_slots
    )
    retained_global = tuple(
        symbol
        for slot, symbol in enumerate(GLOBAL_CONTEXT_SYMBOLS)
        if slot not in context.global_slots
    )
    if (*retained_local, *retained_global) != EXPECTED_RETAINED_CONTEXTS:
        raise ValueError(
            "Frozen context ablation does not resolve to the expected core"
        )

    schema = json.loads((store / "feature_schema.json").read_text(encoding="utf-8"))
    family = schema.get("family_inapplicable_zero_fields")
    if not isinstance(family, dict):
        raise ValueError("Feature schema is missing family applicability")
    all_context = _indices(family.get("all_context"), "all_context")
    applicability: dict[str, np.ndarray] = {
        "equity": _applicable(32, _indices(family.get("equity"), "equity")),
        "global": _applicable(32, _indices(schema.get("global_slow"), "global_slow")),
    }
    for symbol in retained_local:
        inapplicable = set(all_context)
        if symbol in {"WIN$", "WDO$"}:
            inapplicable |= _indices(family.get("WIN_WDO"), "WIN_WDO")
        elif symbol in {"DI1F27", "DI1F28", "DI1F29", "DI1F31"}:
            inapplicable |= _indices(
                family.get("fixed_maturity_DI"), "fixed_maturity_DI"
            )
        elif symbol == "DI1$N":
            inapplicable |= _indices(family.get("DI1$N"), "DI1$N")
        applicability[symbol] = _applicable(32, inapplicable)

    equity_slow = np.load(store / "equity_slow.npy", mmap_mode="r")
    membership = np.load(store / "equity_membership.npy", mmap_mode="r")
    equity_ready = np.load(store / "equity_data_ready.npy", mmap_mode="r")
    equity_values = np.asarray(equity_slow[date_indices], dtype=np.float32)
    active_equities = np.asarray(
        membership[date_indices] & equity_ready[date_indices], dtype=bool
    )
    equity_rows = equity_values[active_equities].copy()
    equity_rows[:, ~applicability["equity"]] = np.nan

    local_slow = np.load(store / "context_slow.npy", mmap_mode="r")
    local_ready = np.load(store / "context_data_ready.npy", mmap_mode="r")
    row_groups = [equity_rows]
    group_counts: dict[str, int] = {"equity": int(equity_rows.shape[0])}
    for slot, symbol in enumerate(LOCAL_CONTEXT_SYMBOLS):
        if symbol not in retained_local:
            continue
        ready = np.asarray(local_ready[date_indices, slot], dtype=bool)
        rows = np.asarray(local_slow[date_indices, slot], dtype=np.float32)[
            ready
        ].copy()
        rows[:, ~applicability[symbol]] = np.nan
        row_groups.append(rows)
        group_counts[symbol] = int(rows.shape[0])

    global_slow = np.load(store / "global_slow.npy", mmap_mode="r")
    global_ready = np.load(store / "global_data_ready.npy", mmap_mode="r")
    train_dates = training["date_idx"].to_numpy().astype(np.int64, copy=False)
    decisions = training["decision_idx"].to_numpy().astype(np.int64, copy=False)
    for slot, symbol in enumerate(GLOBAL_CONTEXT_SYMBOLS):
        if symbol not in retained_global:
            continue
        ready = np.asarray(global_ready[train_dates, slot, decisions], dtype=bool)
        rows = np.asarray(global_slow[train_dates, slot, decisions], dtype=np.float32)[
            ready
        ].copy()
        rows[:, ~applicability["global"]] = np.nan
        row_groups.append(rows)
        group_counts[symbol] = int(rows.shape[0])

    values = np.concatenate(row_groups, axis=0)
    if not np.isfinite(values).any(axis=1).all():
        raise ValueError("Slow audit retained a row with no applicable observation")
    return values, {
        "training_start": str(TRAIN_START),
        "training_end": str(TRAIN_END),
        "training_date_count": int(training["trade_date"].n_unique()),
        "training_sample_count": training.height,
        "training_sample_mask_sha256": hashlib.sha256(
            sample_mask.tobytes()
        ).hexdigest(),
        "training_sample_mask_fields": ["sample_id", "date_idx", "decision_idx"],
        "observation_rule": (
            "One ready membership-and-data-ready equity state per training date; "
            "one ready retained-local state per training date; and one ready "
            "retained-global state per training sample decision. Family-inapplicable "
            "cells from feature_schema.json are excluded as NaN before statistics."
        ),
        "row_group_counts": group_counts,
        "pooled_observation_count": int(values.shape[0]),
        "excluded_splits": ["embargo_1", "validation", "embargo_2", "test"],
        "sampling": {"used": False, "seed": None},
        "retained_context_symbols": list(EXPECTED_RETAINED_CONTEXTS),
    }


def run_training_slow_audit(store: Path, output_dir: Path) -> Path:
    feature_identity = _feature_store_identity(store)
    feature_ablation = resolve_feature_ablation_for_store(store, "drop_slow_low_prior")
    values, sample_contract = _training_values(store)
    analysis = compute_slow_statistics(
        values,
        removed_indices=feature_ablation.slow_indices,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    stats_path = output_dir / STATS_CSV
    pearson_path = output_dir / PEARSON_CSV
    spearman_path = output_dir / SPEARMAN_CSV
    focused_path = output_dir / FOCUSED_CSV
    _atomic_write_csv(stats_path, pl.DataFrame(analysis["statistics"]))
    _atomic_write_csv(pearson_path, _matrix_frame(analysis["pearson"], SLOW_CHANNELS))
    _atomic_write_csv(spearman_path, _matrix_frame(analysis["spearman"], SLOW_CHANNELS))
    _atomic_write_csv(focused_path, pl.DataFrame(analysis["focused"]))

    audit = {
        "audit_name": AUDIT_NAME,
        "audit_version": AUDIT_VERSION,
        "validation_only_experiment": True,
        "test_metrics_accessed": False,
        "feature_store": feature_identity,
        "feature_manifest_sha256": feature_identity["manifest_sha256"],
        "context_ablation": FROZEN_CONTEXT_ABLATION,
        "feature_ablation": feature_ablation.metadata(),
        "canonical_slow_channels": [
            {"index": index, "name": name} for index, name in enumerate(SLOW_CHANNELS)
        ],
        "canonical_global_slow_channels": [
            {"index": index, "name": name}
            for index, name in enumerate(GLOBAL_SLOW_CHANNELS)
        ],
        "sample_contract": sample_contract,
        "statistics": analysis["statistics"],
        "pearson_correlation_matrix": analysis["pearson"],
        "spearman_correlation_matrix": analysis["spearman"],
        "paired_valid_observation_counts": analysis["paired_valid_observation_counts"],
        "focused_correlations": analysis["focused"],
        "output_sha256": {
            STATS_CSV: _sha256(stats_path),
            PEARSON_CSV: _sha256(pearson_path),
            SPEARMAN_CSV: _sha256(spearman_path),
            FOCUSED_CSV: _sha256(focused_path),
        },
    }
    audit_path = output_dir / AUDIT_JSON
    _atomic_write_json(audit_path, audit)
    return audit_path


def validate_training_slow_audit(
    audit_path: Path, feature_identity: dict[str, object]
) -> dict[str, object]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_ablation = resolve_feature_ablation_for_store(
        Path(str(feature_identity["resolved_path"])), "drop_slow_low_prior"
    ).metadata()
    if (
        audit.get("audit_name") != AUDIT_NAME
        or audit.get("audit_version") != AUDIT_VERSION
        or audit.get("test_metrics_accessed") is not False
        or audit.get("validation_only_experiment") is not True
        or audit.get("feature_manifest_sha256")
        != feature_identity.get("manifest_sha256")
        or audit.get("context_ablation") != FROZEN_CONTEXT_ABLATION
        or audit.get("feature_ablation") != expected_ablation
    ):
        raise ValueError("Training-only slow-feature audit identity is incompatible")
    sample = audit.get("sample_contract")
    if (
        not isinstance(sample, dict)
        or sample.get("training_start") != str(TRAIN_START)
        or sample.get("training_end") != str(TRAIN_END)
        or sample.get("excluded_splits")
        != ["embargo_1", "validation", "embargo_2", "test"]
        or sample.get("sampling") != {"used": False, "seed": None}
    ):
        raise ValueError("Training-only slow-feature audit used the wrong sample")
    output_hashes = audit.get("output_sha256")
    if not isinstance(output_hashes, dict):
        raise ValueError("Training-only slow-feature audit lacks output hashes")
    for name in (STATS_CSV, PEARSON_CSV, SPEARMAN_CSV, FOCUSED_CSV):
        path = audit_path.parent / name
        if output_hashes.get(name) != _sha256(path):
            raise ValueError(f"Training-only slow-feature audit output changed: {name}")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = resolve_feature_store().resolve()
    path = run_training_slow_audit(store, args.output_dir.resolve())
    print(f"Wrote training-only slow-feature audit: {path}")


if __name__ == "__main__":
    main()
