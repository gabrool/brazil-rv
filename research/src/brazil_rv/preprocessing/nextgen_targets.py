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

from ..modeling.contract import VALIDATION_END, workspace_path
from ..modeling.data import feature_store_identity
from .contract import (
    DECISION_EQUITY_INDICES,
    EQUITY_SESSION_MINUTES,
    EQUITY_SESSION_START_MINUTE,
    MIN_ACTIVE_EQUITIES,
)
from .io import (
    cotahist_files,
    dense_grid,
    load_assignments,
    load_market_dates_and_security_dates,
    load_source_file,
    prepare_session_bars,
    read_research_interval,
    validate_physical_source_identity,
    validate_source_date_isolation,
)
from .transforms import centered_midranks

NEXTGEN_TARGET_SCHEMA = "EXPERIMENT48_15M_LEG_TARGETS_V1"
NEXTGEN_TARGET_FILES = (
    "leg_raw_returns.npy",
    "leg_targets.npy",
    "leg_label_mask.npy",
    "leg_cross_section_median.npy",
)
LEG_MINUTES = 15
LEG_NAMES = ("0_to_15", "15_to_30")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _recorded_input(store_manifest: dict[str, object], name: str) -> Path:
    inputs = store_manifest.get("canonical_inputs")
    if not isinstance(inputs, dict) or not isinstance(inputs.get(name), dict):
        raise ValueError(f"Feature store does not record canonical input {name}")
    value = inputs[name].get("resolved_path")
    if not isinstance(value, str):
        raise ValueError(f"Feature store canonical input {name} has no resolved path")
    return workspace_path(value)


def exact_leg_returns(
    raw_grid: NDArray[np.float64], observed: NDArray[np.bool_]
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    if raw_grid.ndim != 3 or raw_grid.shape[2] != 5:
        raise ValueError("Raw equity grid must be [date, minute, OHLCV]")
    if observed.shape != raw_grid.shape[:2]:
        raise ValueError("Observation mask does not align to raw equity grid")
    output = np.zeros(
        (raw_grid.shape[0], len(DECISION_EQUITY_INDICES), len(LEG_NAMES)),
        dtype=np.float32,
    )
    mask = np.zeros(output.shape, dtype=bool)
    for decision, entry in enumerate(DECISION_EQUITY_INDICES):
        endpoints = (
            (entry, entry + LEG_MINUTES - 1),
            (entry + LEG_MINUTES, entry + 2 * LEG_MINUTES - 1),
        )
        for leg, (leg_entry, leg_exit) in enumerate(endpoints):
            valid = observed[:, leg_entry] & observed[:, leg_exit]
            mask[:, decision, leg] = valid
            output[valid, decision, leg] = np.log(
                raw_grid[valid, leg_exit, 3] / raw_grid[valid, leg_entry, 0]
            ).astype(np.float32)
    return output, mask


def center_leg_cross_section(
    raw_returns: NDArray[np.float32],
    candidate_mask: NDArray[np.bool_],
    sigma: NDArray[np.float64],
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.bool_],
    NDArray[np.float32],
]:
    if raw_returns.shape != candidate_mask.shape or raw_returns.ndim != 3:
        raise ValueError("Leg returns and masks must be [equity, decision, leg]")
    if sigma.shape != (raw_returns.shape[0],):
        raise ValueError("Target scale does not align to the equity axis")
    masked_raw = np.where(candidate_mask, raw_returns, 0.0).astype(np.float32)
    targets = np.zeros_like(masked_raw)
    label_mask = np.zeros_like(candidate_mask)
    medians = np.zeros(masked_raw.shape[1:], dtype=np.float32)
    for decision in range(masked_raw.shape[1]):
        for leg in range(masked_raw.shape[2]):
            valid = candidate_mask[:, decision, leg]
            if np.any(valid & (~np.isfinite(sigma) | (sigma <= 0.0))):
                raise ValueError(
                    "Valid leg labels require finite positive target scales"
                )
            if int(valid.sum()) < MIN_ACTIVE_EQUITIES:
                masked_raw[:, decision, leg] = 0.0
                continue
            values = masked_raw[valid, decision, leg].astype(np.float64)
            median = float(np.median(values))
            standardized = (values - median) / (sigma[valid] * np.sqrt(LEG_MINUTES))
            medians[decision, leg] = median
            label_mask[valid, decision, leg] = True
            targets[valid, decision, leg] = centered_midranks(standardized)
    return masked_raw, targets, label_mask, medians


def mutation_causality_audit() -> dict[str, bool]:
    raw = np.zeros((1, EQUITY_SESSION_MINUTES, 5), dtype=np.float64)
    observed = np.zeros(raw.shape[:2], dtype=bool)
    entry = DECISION_EQUITY_INDICES[0]
    endpoints = (entry, entry + 14, entry + 15, entry + 29)
    raw[0, entry, 0] = 100.0
    raw[0, entry + 14, 3] = 101.0
    raw[0, entry + 15, 0] = 101.5
    raw[0, entry + 29, 3] = 102.0
    observed[0, list(endpoints)] = True
    baseline, baseline_mask = exact_leg_returns(raw, observed)

    post = raw.copy()
    post_observed = observed.copy()
    post[0, entry + 30, 3] = 50_000.0
    post_observed[0, entry + 30] = True
    post_values, _ = exact_leg_returns(post, post_observed)
    at_exit = raw.copy()
    at_exit[0, entry + 29, 3] = 103.0
    changed, _ = exact_leg_returns(at_exit, observed)
    missing = observed.copy()
    missing[0, entry + 15] = False
    _, missing_mask = exact_leg_returns(raw, missing)
    checks = {
        "post_window_mutation_invariant": bool(np.array_equal(baseline, post_values)),
        "exact_leg2_exit_mutation_sensitive": bool(
            baseline[0, 0, 1] != changed[0, 0, 1]
        ),
        "missing_leg2_entry_masks_label": bool(
            baseline_mask[0, 0, 1] and not missing_mask[0, 0, 1]
        ),
        "leg2_starts_at_minute_15": bool(
            np.isclose(baseline[0, 0, 1], np.log(102.0 / 101.5))
        ),
    }
    if not all(checks.values()):
        raise ValueError("Experiment 48 target mutation audit failed")
    return checks


def target_scale_source_identity(
    directory: Path, store_identity: dict[str, object]
) -> dict[str, object]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = manifest.get("array_sha256")
    if not isinstance(hashes, dict) or "target_scale.npy" not in hashes:
        raise ValueError("Target-scale source does not record its array hash")
    actual = _sha256(directory / "target_scale.npy")
    if (
        manifest.get("source_feature_store") != store_identity
        or manifest.get("through") != VALIDATION_END.isoformat()
        or manifest.get("official_validation_evaluated") is not False
        or manifest.get("test_accessed") is not False
        or hashes["target_scale.npy"] != actual
    ):
        raise ValueError(
            "Target-scale source differs from the frozen development contract"
        )
    return {
        "path": str(directory.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "target_scale_sha256": actual,
        "through": manifest["through"],
    }


def nextgen_target_identity(
    directory: Path, store_identity: dict[str, object]
) -> dict[str, object]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = {name: _sha256(directory / name) for name in NEXTGEN_TARGET_FILES}
    if (
        manifest.get("schema") != NEXTGEN_TARGET_SCHEMA
        or manifest.get("source_feature_store") != store_identity
        or manifest.get("through") != VALIDATION_END.isoformat()
        or manifest.get("official_validation_accessed") is not False
        or manifest.get("test_accessed") is not False
        or manifest.get("array_sha256") != hashes
    ):
        raise ValueError("Experiment 48 target sidecar identity differs")
    return {
        "schema": NEXTGEN_TARGET_SCHEMA,
        "path": str(directory.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "array_sha256": hashes,
        "through": manifest["through"],
    }


def _close_memmaps(arrays: dict[str, np.memmap]) -> None:
    for array in arrays.values():
        array.flush()
        mapping = getattr(array, "_mmap", None)
        if mapping is not None:
            mapping.close()


def build_nextgen_targets(
    store: Path, target_scale_source: Path, output_dir: Path
) -> Path:
    store = store.resolve()
    store_identity = feature_store_identity(store)
    if output_dir.exists():
        nextgen_target_identity(output_dir, store_identity)
        return output_dir
    scale_identity = target_scale_source_identity(
        target_scale_source.resolve(), store_identity
    )
    store_manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
    assignments_dir = _recorded_input(store_manifest, "accepted_xp_assignments")
    cotahist_dir = _recorded_input(store_manifest, "parsed_cotahist")
    universe_dir = _recorded_input(store_manifest, "point_in_time_universe")
    research_start, research_end = read_research_interval(universe_dir)
    through = min(research_end, VALIDATION_END)
    assignments = load_assignments(assignments_dir)
    security_ids = tuple(assignments.get_column("security_id").to_list())
    market_dates, assignment_dates = load_market_dates_and_security_dates(
        cotahist_files(cotahist_dir),
        security_ids,
        research_start,
        through,
        allow_empty_security_dates=True,
    )
    validate_source_date_isolation(assignments, assignment_dates)
    date_index = (
        pl.read_parquet(store / "date_index.parquet")
        .filter(pl.col("trade_date") <= through)
        .sort("date_idx")
    )
    equity_index = pl.read_parquet(store / "equity_index.parquet").sort("equity_slot")
    if (
        tuple(date_index["trade_date"]) != market_dates
        or not np.array_equal(
            date_index["date_idx"].to_numpy(), np.arange(len(market_dates))
        )
        or tuple(equity_index["security_id"]) != security_ids
    ):
        raise ValueError("Experiment 48 target axes differ from the feature store")

    date_count = len(market_dates)
    equity_count = len(security_ids)
    decision_count = len(DECISION_EQUITY_INDICES)
    target_scale = np.load(
        target_scale_source / "target_scale.npy", mmap_mode="r", allow_pickle=False
    )
    if target_scale.shape != (date_count, equity_count):
        raise ValueError("Target scale does not cover the development-only axes")

    partial = output_dir.with_name(f".{output_dir.name}.tmp-{uuid4().hex}")
    partial.mkdir(parents=True)
    specs = {
        "leg_raw_returns.npy": (
            np.float32,
            (date_count, equity_count, decision_count, len(LEG_NAMES)),
        ),
        "leg_targets.npy": (
            np.float32,
            (date_count, equity_count, decision_count, len(LEG_NAMES)),
        ),
        "leg_label_mask.npy": (
            bool,
            (date_count, equity_count, decision_count, len(LEG_NAMES)),
        ),
        "leg_cross_section_median.npy": (
            np.float32,
            (date_count, decision_count, len(LEG_NAMES)),
        ),
    }
    arrays = {
        name: open_memmap(partial / name, mode="w+", dtype=dtype, shape=shape)
        for name, (dtype, shape) in specs.items()
    }
    for array in arrays.values():
        array[...] = 0
    slot_by_security = {
        security_id: slot for slot, security_id in enumerate(security_ids)
    }
    try:
        groups = assignments.partition_by("source_file", maintain_order=True)
        for source_number, group in enumerate(groups, start=1):
            source_path = Path(group.item(0, "source_file"))
            source = load_source_file(source_path)
            validate_physical_source_identity(group, source, source_path)
            allowed_dates = frozenset().union(
                *(assignment_dates[value] for value in group["security_id"])
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
                bars = session_bars.filter(
                    pl.col("trade_date").is_in(tuple(assignment_dates[security_id]))
                )
                raw_grid, observed = dense_grid(
                    bars, date_count, EQUITY_SESSION_MINUTES
                )
                returns, endpoint_mask = exact_leg_returns(raw_grid, observed)
                slot = slot_by_security[security_id]
                arrays["leg_raw_returns.npy"][:, slot] = returns
                arrays["leg_label_mask.npy"][:, slot] = endpoint_mask
            if source_number % 20 == 0 or source_number == len(groups):
                print(f"Built Experiment 48 exact legs {source_number}/{len(groups)}")

        membership = np.load(
            store / "equity_membership.npy", mmap_mode="r", allow_pickle=False
        )[:date_count]
        data_ready = np.load(
            store / "equity_data_ready.npy", mmap_mode="r", allow_pickle=False
        )[:date_count]
        counts = np.zeros((len(LEG_NAMES),), dtype=np.int64)
        cross_sections = np.zeros((len(LEG_NAMES),), dtype=np.int64)
        for date_idx in range(date_count):
            endpoint_mask = np.asarray(
                arrays["leg_label_mask.npy"][date_idx], dtype=bool
            )
            candidate = (
                membership[date_idx, :, None, None]
                & data_ready[date_idx, :, None, None]
                & endpoint_mask
            )
            raw, targets, label_mask, medians = center_leg_cross_section(
                np.asarray(arrays["leg_raw_returns.npy"][date_idx]),
                candidate,
                np.asarray(target_scale[date_idx], dtype=np.float64),
            )
            arrays["leg_raw_returns.npy"][date_idx] = raw
            arrays["leg_targets.npy"][date_idx] = targets
            arrays["leg_label_mask.npy"][date_idx] = label_mask
            arrays["leg_cross_section_median.npy"][date_idx] = medians
            counts += label_mask.sum(axis=(0, 1))
            cross_sections += label_mask.any(axis=0).sum(axis=0)

        for name in ("leg_raw_returns.npy", "leg_targets.npy"):
            values = arrays[name]
            mask = arrays["leg_label_mask.npy"]
            if not np.isfinite(values).all() or np.any(values[~mask] != 0):
                raise ValueError(f"Invalid values in {name}")
        audit = {
            "schema": "EXPERIMENT48_15M_LEG_TARGET_AUDIT_V1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "through": through.isoformat(),
            "date_count": date_count,
            "equity_count": equity_count,
            "decision_count": decision_count,
            "leg_names": list(LEG_NAMES),
            "label_count": {
                name: int(counts[index]) for index, name in enumerate(LEG_NAMES)
            },
            "valid_cross_section_count": {
                name: int(cross_sections[index]) for index, name in enumerate(LEG_NAMES)
            },
            "mutation_causality": mutation_causality_audit(),
            "official_validation_accessed": False,
            "test_accessed": False,
        }
        (partial / "audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _close_memmaps(arrays)
        arrays.clear()
        hashes = {name: _sha256(partial / name) for name in NEXTGEN_TARGET_FILES}
        manifest = {
            "schema": NEXTGEN_TARGET_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_feature_store": store_identity,
            "target_scale_source": scale_identity,
            "through": through.isoformat(),
            "horizon_minutes": LEG_MINUTES,
            "leg_names": list(LEG_NAMES),
            "shape": [date_count, equity_count, decision_count, len(LEG_NAMES)],
            "construction": (
                "exact open[T] to close[T+14] and open[T+15] to close[T+29]; "
                "each leg independently raw-median-centered, divided by causal "
                "per-name sigma*sqrt(15), then cross-sectional midranked"
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
    nextgen_target_identity(output_dir, store_identity)
    return output_dir
