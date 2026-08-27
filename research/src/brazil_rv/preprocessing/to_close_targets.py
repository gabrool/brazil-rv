from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import polars as pl
from numpy.lib.format import open_memmap
from numpy.typing import NDArray

from ..execution.inputs import iter_discovery_equity_grids
from ..modeling.contract import TRAIN_END
from ..modeling.data import TO_CLOSE_TARGET_SCHEMA, feature_store_identity
from .contract import (
    DECISION_EQUITY_INDICES,
    EQUITY_SESSION_MINUTES,
    MIN_ACTIVE_EQUITIES,
)
from .transforms import centered_midranks

TO_CLOSE_TARGET_FILES = (
    "leg_raw_returns.npy",
    "leg_targets.npy",
    "leg_label_mask.npy",
    "leg_cross_section_median.npy",
)
FINAL_MARK_MINUTE = EQUITY_SESSION_MINUTES - 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def exact_to_close_returns(
    raw_grid: NDArray[np.float64], observed: NDArray[np.bool_]
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Return exact open-at-entry to final-close labels for one security."""
    if raw_grid.ndim != 3 or raw_grid.shape[1:] != (EQUITY_SESSION_MINUTES, 5):
        raise ValueError("Raw equity grid must be [date,405,OHLCV]")
    if observed.shape != raw_grid.shape[:2]:
        raise ValueError("Observation mask does not align to the raw equity grid")
    output = np.zeros((raw_grid.shape[0], len(DECISION_EQUITY_INDICES)), np.float32)
    mask = np.zeros(output.shape, dtype=bool)
    final_valid = observed[:, FINAL_MARK_MINUTE]
    final_close = raw_grid[:, FINAL_MARK_MINUTE, 3]
    for decision, entry in enumerate(DECISION_EQUITY_INDICES):
        valid = observed[:, entry] & final_valid
        mask[:, decision] = valid
        output[valid, decision] = np.log(
            final_close[valid] / raw_grid[valid, entry, 0]
        ).astype(np.float32)
    return output, mask


def center_to_close_cross_section(
    raw_returns: NDArray[np.float32],
    candidate_mask: NDArray[np.bool_],
    sigma: NDArray[np.float64],
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.bool_],
    NDArray[np.float32],
]:
    if raw_returns.shape != candidate_mask.shape or raw_returns.ndim != 2:
        raise ValueError("To-close returns and masks must be [equity,decision]")
    if sigma.shape != (raw_returns.shape[0],):
        raise ValueError("Target scale does not align to the equity axis")
    masked_raw = np.where(candidate_mask, raw_returns, 0.0).astype(np.float32)
    targets = np.zeros_like(masked_raw)
    label_mask = np.zeros_like(candidate_mask)
    medians = np.zeros(masked_raw.shape[1], dtype=np.float32)
    horizons = FINAL_MARK_MINUTE + 1 - np.asarray(DECISION_EQUITY_INDICES)
    for decision, horizon in enumerate(horizons):
        valid = candidate_mask[:, decision]
        if np.any(valid & (~np.isfinite(sigma) | (sigma <= 0.0))):
            raise ValueError("Valid to-close labels require finite positive scales")
        if int(valid.sum()) < MIN_ACTIVE_EQUITIES:
            masked_raw[:, decision] = 0.0
            continue
        values = masked_raw[valid, decision].astype(np.float64)
        median = float(np.median(values))
        standardized = (values - median) / (sigma[valid] * np.sqrt(horizon))
        medians[decision] = median
        label_mask[valid, decision] = True
        targets[valid, decision] = centered_midranks(standardized)
    return masked_raw, targets, label_mask, medians


def mutation_audit() -> dict[str, bool]:
    raw = np.ones((1, EQUITY_SESSION_MINUTES, 5), dtype=np.float64) * 100.0
    observed = np.ones(raw.shape[:2], dtype=bool)
    raw[:, :, 3] = 101.0
    baseline, baseline_mask = exact_to_close_returns(raw, observed)
    middle = raw.copy()
    middle[0, 200, 3] = 50_000.0
    unchanged, _ = exact_to_close_returns(middle, observed)
    final = raw.copy()
    final[0, FINAL_MARK_MINUTE, 3] = 102.0
    changed, _ = exact_to_close_returns(final, observed)
    missing = observed.copy()
    missing[0, FINAL_MARK_MINUTE] = False
    _, missing_mask = exact_to_close_returns(raw, missing)
    checks = {
        "intermediate_price_mutation_invariant": bool(
            np.array_equal(baseline, unchanged)
        ),
        "final_mark_mutation_sensitive": bool(not np.array_equal(baseline, changed)),
        "missing_final_mark_masks_all_decisions": bool(
            baseline_mask.all() and not missing_mask.any()
        ),
    }
    if not all(checks.values()):
        raise ValueError("Experiment 55 to-close target audit failed")
    return checks


def _target_scale_identity(
    directory: Path, store_identity: dict[str, object], required_dates: int
) -> dict[str, object]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = manifest.get("array_sha256")
    scale_path = directory / "target_scale.npy"
    actual = _sha256(scale_path)
    scale = np.load(scale_path, mmap_mode="r", allow_pickle=False)
    if (
        not isinstance(recorded, dict)
        or recorded.get("target_scale.npy") != actual
        or manifest.get("source_feature_store") != store_identity
        or manifest.get("official_validation_evaluated") is not False
        or manifest.get("test_accessed") is not False
        or scale.ndim != 2
        or scale.shape[0] < required_dates
    ):
        raise ValueError("Target-scale source differs from its sealed contract")
    return {
        "path": str(directory.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "target_scale_sha256": actual,
        "recorded_through": manifest.get("through"),
        "rows_consumed": required_dates,
    }


def to_close_target_identity(
    directory: Path, store_identity: dict[str, object]
) -> dict[str, object]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = {name: _sha256(directory / name) for name in TO_CLOSE_TARGET_FILES}
    if (
        manifest.get("schema") != TO_CLOSE_TARGET_SCHEMA
        or manifest.get("source_feature_store") != store_identity
        or manifest.get("through") != TRAIN_END.isoformat()
        or manifest.get("array_sha256") != hashes
        or manifest.get("official_validation_accessed") is not False
        or manifest.get("test_accessed") is not False
    ):
        raise ValueError("Experiment 55 target sidecar identity differs")
    return {
        "schema": TO_CLOSE_TARGET_SCHEMA,
        "path": str(directory.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "array_sha256": hashes,
        "through": TRAIN_END.isoformat(),
    }


def _close_memmaps(arrays: dict[str, np.memmap]) -> None:
    for array in arrays.values():
        array.flush()
        mapping = getattr(array, "_mmap", None)
        if mapping is not None:
            mapping.close()


def build_to_close_targets(
    store: Path, target_scale_source: Path, output_dir: Path
) -> Path:
    store = store.resolve()
    store_identity = feature_store_identity(store)
    if output_dir.exists():
        to_close_target_identity(output_dir, store_identity)
        return output_dir
    date_index = (
        pl.read_parquet(store / "date_index.parquet")
        .filter(pl.col("trade_date") <= TRAIN_END)
        .sort("date_idx")
    )
    dates = tuple(date_index["trade_date"])
    if not dates or dates[-1] != TRAIN_END:
        raise ValueError("Canonical TRAIN date axis is incomplete")
    date_count = len(dates)
    membership = np.load(
        store / "equity_membership.npy", mmap_mode="r", allow_pickle=False
    )[:date_count]
    ready = np.load(
        store / "equity_data_ready.npy", mmap_mode="r", allow_pickle=False
    )[:date_count]
    equity_count = membership.shape[1]
    scale_identity = _target_scale_identity(
        target_scale_source.resolve(), store_identity, date_count
    )
    target_scale = np.load(
        target_scale_source / "target_scale.npy", mmap_mode="r", allow_pickle=False
    )[:date_count]

    partial = output_dir.with_name(f".{output_dir.name}.tmp-{uuid4().hex}")
    partial.mkdir(parents=True)
    shape = (date_count, equity_count, len(DECISION_EQUITY_INDICES), 1)
    specs = {
        "leg_raw_returns.npy": (np.float32, shape),
        "leg_targets.npy": (np.float32, shape),
        "leg_label_mask.npy": (bool, shape),
        "leg_cross_section_median.npy": (
            np.float32,
            (date_count, len(DECISION_EQUITY_INDICES), 1),
        ),
    }
    arrays = {
        name: open_memmap(partial / name, mode="w+", dtype=dtype, shape=array_shape)
        for name, (dtype, array_shape) in specs.items()
    }
    for array in arrays.values():
        array[...] = 0
    seen = np.zeros(equity_count, dtype=bool)
    try:
        for grid in iter_discovery_equity_grids(store):
            if grid.trade_dates != dates:
                raise ValueError("Raw and canonical TRAIN date axes differ")
            returns, endpoint_mask = exact_to_close_returns(
                np.stack(
                    (
                        grid.open_price,
                        grid.high,
                        grid.low,
                        grid.close,
                        grid.real_volume,
                    ),
                    axis=-1,
                ),
                grid.observed,
            )
            arrays["leg_raw_returns.npy"][:, grid.equity_slot, :, 0] = returns
            arrays["leg_label_mask.npy"][:, grid.equity_slot, :, 0] = endpoint_mask
            seen[grid.equity_slot] = True
        if not seen.all():
            raise ValueError("To-close target builder did not emit every security")

        label_count = 0
        cross_section_count = 0
        for date_idx in range(date_count):
            endpoint = np.asarray(
                arrays["leg_label_mask.npy"][date_idx, :, :, 0], dtype=bool
            )
            candidate = membership[date_idx, :, None] & ready[date_idx, :, None] & endpoint
            raw, targets, label_mask, medians = center_to_close_cross_section(
                np.asarray(arrays["leg_raw_returns.npy"][date_idx, :, :, 0]),
                candidate,
                np.asarray(target_scale[date_idx], dtype=np.float64),
            )
            arrays["leg_raw_returns.npy"][date_idx, :, :, 0] = raw
            arrays["leg_targets.npy"][date_idx, :, :, 0] = targets
            arrays["leg_label_mask.npy"][date_idx, :, :, 0] = label_mask
            arrays["leg_cross_section_median.npy"][date_idx, :, 0] = medians
            label_count += int(label_mask.sum())
            cross_section_count += int(label_mask.any(axis=0).sum())

        mask = arrays["leg_label_mask.npy"]
        for name in ("leg_raw_returns.npy", "leg_targets.npy"):
            values = arrays[name]
            if not np.isfinite(values).all() or np.any(values[~mask] != 0):
                raise ValueError(f"Invalid values in {name}")
        audit = {
            "schema": "EXPERIMENT55_TO_CLOSE_TARGET_AUDIT_V1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "through": TRAIN_END.isoformat(),
            "label_count": label_count,
            "valid_cross_section_count": cross_section_count,
            "mutation": mutation_audit(),
            "official_validation_accessed": False,
            "test_accessed": False,
        }
        (partial / "audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _close_memmaps(arrays)
        arrays.clear()
        hashes = {name: _sha256(partial / name) for name in TO_CLOSE_TARGET_FILES}
        horizons = (FINAL_MARK_MINUTE + 1 - np.asarray(DECISION_EQUITY_INDICES)).tolist()
        manifest = {
            "schema": TO_CLOSE_TARGET_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_feature_store": store_identity,
            "target_scale_source": scale_identity,
            "through": TRAIN_END.isoformat(),
            "leg_names": ["to_close"],
            "horizon_minutes_by_decision": horizons,
            "shape": list(shape),
            "construction": (
                "exact open[T] to final close[404], cross-sectional median removed, "
                "divided by causal per-name sigma*sqrt(405-T), then midranked"
            ),
            "array_sha256": hashes,
            "audit_sha256": _sha256(partial / "audit.json"),
            "official_validation_accessed": False,
            "test_accessed": False,
        }
        (partial / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, output_dir)
    except BaseException:
        if arrays:
            _close_memmaps(arrays)
        shutil.rmtree(partial, ignore_errors=True)
        raise
    to_close_target_identity(output_dir, store_identity)
    return output_dir
