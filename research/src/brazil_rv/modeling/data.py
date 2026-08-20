from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .contract import (
    ABSOLUTE_PATCH_COUNT,
    CANONICAL_DROPPED_LOCAL_SLOTS,
    CANONICAL_NEUTRALIZED_EQUITY_SLOW_INDICES,
    CANONICAL_RETAINED_GLOBAL_SLOTS,
    CONTEXT_COUNT,
    DECISION_GLOBAL_INDICES,
    DISCOVERY_FIT_DATE_COUNTS,
    DISCOVERY_SELECTION_DATE_COUNT,
    EQUITY_ABSOLUTE_START_PATCH,
    EQUITY_COUNT,
    EXPECTED_DECISIONS_PER_DATE,
    EXPECTED_SPLIT_DATE_COUNTS,
    FEATURE_STORE_POINTER,
    GH200_RUNTIME,
    GLOBAL_CONTEXT_COUNT,
    GLOBAL_WINDOW_MINUTES,
    HORIZON_COUNT,
    LOCAL_CONTEXT_COUNT,
    PATCH_INPUT_WIDTH,
    PATCH_MINUTES,
    RuntimeSettings,
    SLOW_FEATURE_COUNT,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
    workspace_path,
)

FEATURE_ARRAY_FILES = (
    "equity_features.npy",
    "equity_slow.npy",
    "equity_membership.npy",
    "equity_data_ready.npy",
    "context_features.npy",
    "context_slow.npy",
    "context_data_ready.npy",
    "targets.npy",
    "global_features.npy",
    "global_slow.npy",
    "global_data_ready.npy",
    "label_mask.npy",
    "raw_returns.npy",
)
FEATURE_STORE_CONTRACT = "M1_FEATURES_PIT_CAUSAL_TOD"
DI_TILT_SIDECAR_SCHEMA = "DI_TILT_EXPOSURE_SIDECAR_V1"
MARKET_STATE_SPECS = (
    (0, 8),  # ES return_30m_normalized
    (0, 11),  # ES realized_vol_30m_log_ratio
    (5, 8),  # HG return_30m_normalized
    (7, 8),  # 6M return_30m_normalized
)
SCREENING_FOLD_NOTE = (
    "Causal stored features are reused, but the time-of-day profile adapted inside "
    "these historical training dates. These are screening folds, not exact replicas "
    "of the officially frozen preprocessing regime."
)


@dataclass(frozen=True)
class BatchRequest:
    indices: tuple[int, ...]
    valid_count: int


@dataclass(frozen=True)
class DiscoveryFold:
    name: str
    fit_rows: pl.DataFrame
    selection_rows: pl.DataFrame


def resolve_feature_store(pointer: Path = FEATURE_STORE_POINTER) -> Path:
    store = workspace_path(pointer.read_text(encoding="utf-8").strip())
    if not store.is_dir():
        raise FileNotFoundError(store)
    return store


def feature_store_identity(store: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    for name in ("manifest.json", "feature_schema.json", "sample_index.parquet"):
        with (store / name).open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    schema = json.loads((store / "feature_schema.json").read_text(encoding="utf-8"))
    if schema.get("contract_version") != FEATURE_STORE_CONTRACT:
        raise ValueError(
            f"Expected {FEATURE_STORE_CONTRACT}, got {schema.get('contract_version')}"
        )
    return {
        "path": str(store.resolve()),
        "contract_version": schema["contract_version"],
        "metadata_sha256": digest.hexdigest(),
    }


FeatureStoreIdentityCache = dict[str, dict[str, object]]


def di_tilt_sidecar_identity(
    sidecar: Path,
    source_feature_store: dict[str, object],
    *,
    require_residual: bool = False,
) -> dict[str, object]:
    manifest_path = sidecar / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != DI_TILT_SIDECAR_SCHEMA:
        raise ValueError("Unexpected DI tilt sidecar schema")
    if manifest.get("source_feature_store") != source_feature_store:
        raise ValueError("DI tilt sidecar feature store differs from training store")
    if manifest.get("test_accessed") is not False:
        raise ValueError("DI tilt sidecar must not access the held-out test")
    hashes = manifest.get("files_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("DI tilt sidecar has no immutable file hashes")
    required = ["tilt_exposure.npy", "tilt_ready.npy", "audit.json"]
    if require_residual:
        required.extend(("residual_targets.npy", "residual_mask.npy"))
    for name in required:
        path = sidecar / name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        if hashes.get(name) != digest.hexdigest():
            raise ValueError(
                f"DI tilt sidecar file differs from recorded contract: {name}"
            )
    return manifest


def validate_feature_store_identity(
    store: Path,
    recorded_identity: object,
    *,
    identity_cache: FeatureStoreIdentityCache | None = None,
) -> dict[str, object]:
    if not isinstance(recorded_identity, dict):
        raise ValueError("Checkpoint feature-store identity must be a dictionary")
    key = str(store.resolve())
    actual = None if identity_cache is None else identity_cache.get(key)
    if actual is None:
        actual = feature_store_identity(store)
    if actual != recorded_identity:
        raise ValueError("Checkpoint feature store differs from the resolved store")
    if identity_cache is not None:
        identity_cache[key] = actual
    return actual


def int64_identity_sha256(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<i8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def sample_window_metadata(rows: pl.DataFrame, name: str) -> dict[str, object]:
    date_indices = rows.get_column("date_idx").unique().sort().cast(pl.Int64).to_numpy()
    sample_ids = rows.get_column("sample_id").sort().cast(pl.Int64).to_numpy()
    return {
        "name": name,
        "start": str(rows.get_column("trade_date").min()),
        "end": str(rows.get_column("trade_date").max()),
        "date_count": rows.get_column("trade_date").n_unique(),
        "sample_count": rows.height,
        "date_identity_sha256": int64_identity_sha256(date_indices),
        "sample_identity_sha256": int64_identity_sha256(sample_ids),
    }


def _validate_sample_index(sample_index: pl.DataFrame) -> None:
    sample_ids = sample_index.get_column("sample_id").to_numpy()
    if not np.array_equal(sample_ids, np.arange(sample_index.height)):
        raise ValueError("sample_id must be sorted, unique, and contiguous")
    expected = list(range(EXPECTED_DECISIONS_PER_DATE))
    per_date = sample_index.group_by("trade_date").agg(pl.col("decision_idx").sort())
    if any(values.to_list() != expected for values in per_date["decision_idx"]):
        raise ValueError("Each eligible date must contain decisions 0..54 exactly once")
    decision = pl.col("decision_idx").cast(pl.Int16)
    invalid = sample_index.filter(
        (pl.col("equity_cutoff_index") != 15 + PATCH_MINUTES * decision)
        | (pl.col("context_cutoff_index") != 75 + PATCH_MINUTES * decision)
    )
    if invalid.height:
        raise ValueError("Sample cutoffs violate the causal five-minute grid")


def load_sample_index(store: Path, *, through: date | None = None) -> pl.DataFrame:
    rows = pl.scan_parquet(store / "sample_index.parquet")
    if through is not None:
        rows = rows.filter(pl.col("trade_date") <= through)
    collected = rows.collect().sort("sample_id")
    _validate_sample_index(collected)
    return collected


def select_sample_split(sample_index: pl.DataFrame, split: str) -> pl.DataFrame:
    trade_date = pl.col("trade_date")
    filters = {
        "train": trade_date.is_between(TRAIN_START, TRAIN_END),
        "embargo_1": (trade_date > TRAIN_END) & (trade_date < VALIDATION_START),
        "validation": trade_date.is_between(VALIDATION_START, VALIDATION_END),
        "embargo_2": (trade_date > VALIDATION_END) & (trade_date < TEST_START),
        "test": trade_date.is_between(TEST_START, TEST_END),
    }
    try:
        return sample_index.filter(filters[split])
    except KeyError as error:
        raise ValueError(f"Unknown split: {split}") from error


def discovery_folds(training_rows: pl.DataFrame) -> tuple[DiscoveryFold, ...]:
    dates = training_rows.select("date_idx", "trade_date").unique().sort("date_idx")
    expected = EXPECTED_SPLIT_DATE_COUNTS["train"]
    if dates.height != expected:
        raise ValueError(f"Discovery folds require exactly {expected} training dates")
    folds: list[DiscoveryFold] = []
    for name, fit_count in DISCOVERY_FIT_DATE_COUNTS.items():
        selection_dates = dates.slice(fit_count, DISCOVERY_SELECTION_DATE_COUNT)
        fit_dates = dates.head(fit_count)
        folds.append(
            DiscoveryFold(
                name=name,
                fit_rows=training_rows.join(
                    fit_dates.select("date_idx"), on="date_idx", how="semi"
                ),
                selection_rows=training_rows.join(
                    selection_dates.select("date_idx"), on="date_idx", how="semi"
                ),
            )
        )
    left, right = folds
    overlap = set(left.selection_rows["date_idx"]) & set(
        right.selection_rows["date_idx"]
    )
    if overlap:
        raise ValueError("Discovery selection periods overlap")
    return tuple(folds)


def select_training_window(
    sample_index: pl.DataFrame, window: str
) -> tuple[pl.DataFrame, pl.DataFrame, str]:
    training_rows = select_sample_split(sample_index, "train")
    if window == "official":
        return (
            training_rows,
            select_sample_split(sample_index, "validation"),
            (
                "The official validation split is consumed and is reserved for sparse "
                "confirmation of stage winners. The held-out test is untouched."
            ),
        )
    folds = {fold.name: fold for fold in discovery_folds(training_rows)}
    try:
        fold = folds[window]
    except KeyError as error:
        raise ValueError(f"Unknown training window: {window}") from error
    return fold.fit_rows, fold.selection_rows, SCREENING_FOLD_NOTE


def _common_batch(
    arrays: dict[str, np.ndarray],
    rows: dict[str, np.ndarray],
    positions: np.ndarray,
    valid_count: int,
) -> tuple[
    dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    dates = rows["date_idx"][positions].astype(np.int64, copy=False)
    decisions = rows["decision_idx"][positions].astype(np.int64, copy=False)
    sample_id = rows["sample_id"][positions].astype(np.int64, copy=True)
    date_idx = dates.copy()
    decision_idx = decisions.copy()
    equity_cutoffs = rows["equity_cutoff_index"][positions]
    context_cutoffs = rows["context_cutoff_index"][positions]
    batch_size = positions.size
    sample_valid_mask = np.arange(batch_size) < valid_count
    active = np.asarray(
        arrays["equity_membership.npy"][dates] & arrays["equity_data_ready.npy"][dates],
        dtype=bool,
    )
    targets = np.zeros((batch_size, EQUITY_COUNT, HORIZON_COUNT), dtype=np.float32)
    label_mask = np.zeros_like(targets, dtype=bool)
    raw_returns = np.zeros_like(targets)
    for decision in np.unique(decisions):
        group = np.flatnonzero(decisions == decision)
        targets[group] = arrays["targets.npy"][dates[group], :, int(decision), :]
        label_mask[group] = arrays["label_mask.npy"][dates[group], :, int(decision), :]
        raw_returns[group] = arrays["raw_returns.npy"][
            dates[group], :, int(decision), :
        ]
    padded = ~sample_valid_mask
    targets[padded] = 0
    label_mask[padded] = False
    raw_returns[padded] = 0
    sample_id[padded] = date_idx[padded] = decision_idx[padded] = -1
    return (
        {
            "targets": targets,
            "label_mask": label_mask,
            "raw_returns": raw_returns,
            "sample_valid_mask": sample_valid_mask,
            "sample_id": sample_id,
            "date_idx": date_idx,
            "decision_idx": decision_idx,
        },
        active,
        equity_cutoffs,
        context_cutoffs,
        dates,
        decisions,
    )


def _context_readiness(
    arrays: dict[str, np.ndarray],
    date_idx: np.ndarray,
    decision_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_ready = np.asarray(
        arrays["context_data_ready.npy"][date_idx], dtype=bool
    ).copy()
    local_ready[:, CANONICAL_DROPPED_LOCAL_SLOTS] = False
    all_global = np.asarray(arrays["global_data_ready.npy"][date_idx], dtype=bool)
    global_ready = all_global[
        np.arange(date_idx.size)[:, None],
        np.arange(GLOBAL_CONTEXT_COUNT)[None, :],
        decision_idx[:, None],
    ].copy()
    keep = np.zeros(GLOBAL_CONTEXT_COUNT, dtype=bool)
    keep[list(CANONICAL_RETAINED_GLOBAL_SLOTS)] = True
    global_ready &= keep
    return local_ready, global_ready, all_global


def _build_patch_batch(
    arrays: dict[str, np.ndarray],
    date_idx: np.ndarray,
    equity_cutoffs: np.ndarray,
    decision_idx: np.ndarray,
    context_cutoffs: np.ndarray,
    active: np.ndarray,
) -> dict[str, np.ndarray]:
    batch_size = date_idx.size
    local_ready, global_ready, all_global_ready = _context_readiness(
        arrays, date_idx, decision_idx
    )
    patches = np.zeros(
        (
            batch_size,
            EQUITY_COUNT + CONTEXT_COUNT,
            ABSOLUTE_PATCH_COUNT,
            PATCH_INPUT_WIDTH,
        ),
        dtype=np.float32,
    )
    history_mask = np.zeros(
        (batch_size, EQUITY_COUNT + CONTEXT_COUNT, ABSOLUTE_PATCH_COUNT), dtype=bool
    )
    instrument_mask = np.zeros((batch_size, EQUITY_COUNT + CONTEXT_COUNT), dtype=bool)
    instrument_mask[:, :EQUITY_COUNT] = active
    slow = np.zeros(
        (batch_size, EQUITY_COUNT + CONTEXT_COUNT, SLOW_FEATURE_COUNT),
        dtype=np.float32,
    )
    equity_slow = np.asarray(
        arrays["equity_slow.npy"][date_idx], dtype=np.float32
    ).copy()
    equity_slow[..., CANONICAL_NEUTRALIZED_EQUITY_SLOW_INDICES] = 0.0
    slow[:, :EQUITY_COUNT] = equity_slow * active[..., None]
    state_position = np.empty(batch_size, dtype=np.int64)
    for cutoff in np.unique(equity_cutoffs):
        group = np.flatnonzero(equity_cutoffs == cutoff)
        context_cutoff = int(context_cutoffs[group[0]])
        state = context_cutoff // PATCH_MINUTES
        equity_patches = int(cutoff) // PATCH_MINUTES
        if EQUITY_ABSOLUTE_START_PATCH + equity_patches != state:
            raise ValueError("Equity and context patch clocks are misaligned")
        state_position[group] = state
        prefix = np.asarray(
            arrays["equity_features.npy"][date_idx[group], :, : int(cutoff), :],
            dtype=np.float32,
        ).reshape(group.size, EQUITY_COUNT, equity_patches, PATCH_INPUT_WIDTH)
        patches[group, :EQUITY_COUNT, EQUITY_ABSOLUTE_START_PATCH:state] = (
            prefix * active[group, :, None, None]
        )
        history_mask[group, :EQUITY_COUNT, EQUITY_ABSOLUTE_START_PATCH:state] = active[
            group, :, None
        ]
        ready = local_ready[group]
        local = np.asarray(
            arrays["context_features.npy"][date_idx[group], :, :context_cutoff, :],
            dtype=np.float32,
        ).reshape(group.size, LOCAL_CONTEXT_COUNT, state, PATCH_INPUT_WIDTH)
        start = EQUITY_COUNT
        patches[group, start : start + LOCAL_CONTEXT_COUNT, :state] = (
            local * ready[..., None, None]
        )
        history_mask[group, start : start + LOCAL_CONTEXT_COUNT, :state] = ready[
            ..., None
        ]
    global_start = EQUITY_COUNT + LOCAL_CONTEXT_COUNT
    instrument_mask[:, EQUITY_COUNT:global_start] = local_ready
    slow[:, EQUITY_COUNT:global_start] = (
        np.asarray(arrays["context_slow.npy"][date_idx], dtype=np.float32)
        * local_ready[..., None]
    )
    cutoffs = np.asarray(DECISION_GLOBAL_INDICES)[decision_idx]
    minutes = (
        cutoffs[:, None]
        - GLOBAL_WINDOW_MINUTES
        + np.arange(GLOBAL_WINDOW_MINUTES)[None, :]
    )
    global_grid = np.asarray(arrays["global_features.npy"][date_idx], dtype=np.float32)
    global_values = global_grid[
        np.arange(batch_size)[:, None, None],
        np.arange(GLOBAL_CONTEXT_COUNT)[None, :, None],
        minutes[:, None, :],
    ].reshape(batch_size, GLOBAL_CONTEXT_COUNT, ABSOLUTE_PATCH_COUNT, PATCH_INPUT_WIDTH)
    patches[:, global_start:] = global_values * global_ready[..., None, None]
    history_mask[:, global_start:] = global_ready[..., None]
    instrument_mask[:, global_start:] = global_ready
    global_slow = np.asarray(arrays["global_slow.npy"][date_idx], dtype=np.float32)
    decision_slow = global_slow[
        np.arange(batch_size)[:, None],
        np.arange(GLOBAL_CONTEXT_COUNT)[None, :],
        decision_idx[:, None],
    ]
    slow[:, global_start:] = decision_slow * global_ready[..., None]
    market_state = np.zeros((batch_size, len(MARKET_STATE_SPECS)), dtype=np.float32)
    endpoints = np.asarray(DECISION_GLOBAL_INDICES, dtype=np.int64)[decision_idx] - 1
    batch_positions = np.arange(batch_size)
    for output_idx, (slot, channel) in enumerate(MARKET_STATE_SPECS):
        ready = all_global_ready[batch_positions, slot, decision_idx]
        values = global_grid[batch_positions, slot, endpoints, channel]
        market_state[:, output_idx] = np.where(ready, values, 0.0)
    return {
        "patches": patches,
        "history_patch_mask": history_mask,
        "instrument_mask": instrument_mask,
        "slow_features": slow,
        "state_position": state_position,
        "market_state": market_state,
    }


class VectorizedFeatureDataset(Dataset[dict[str, np.ndarray]]):
    def __init__(
        self,
        store: Path,
        sample_index: pl.DataFrame,
        tilt_sidecar: Path | None = None,
    ) -> None:
        self.store = store
        self.tilt_sidecar = tilt_sidecar
        self.rows = {
            name: sample_index.get_column(name).to_numpy()
            for name in (
                "sample_id",
                "date_idx",
                "decision_idx",
                "equity_cutoff_index",
                "context_cutoff_index",
            )
        }
        self._arrays: dict[str, np.ndarray] | None = None
        self._tilt_arrays: tuple[np.ndarray, np.ndarray] | None = None
        self._residual_arrays: tuple[np.ndarray, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.rows["sample_id"])

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_arrays"] = None
        state["_tilt_arrays"] = None
        state["_residual_arrays"] = None
        return state

    def _open_arrays(self) -> dict[str, np.ndarray]:
        if self._arrays is None:
            self._arrays = {
                name: np.load(self.store / name, mmap_mode="r", allow_pickle=False)
                for name in FEATURE_ARRAY_FILES
            }
        return self._arrays

    def _open_tilt_arrays(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self.tilt_sidecar is None:
            return None
        if self._tilt_arrays is None:
            self._tilt_arrays = (
                np.load(
                    self.tilt_sidecar / "tilt_exposure.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                ),
                np.load(
                    self.tilt_sidecar / "tilt_ready.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                ),
            )
        return self._tilt_arrays

    def _open_residual_arrays(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self.tilt_sidecar is None:
            return None
        if not (self.tilt_sidecar / "residual_targets.npy").is_file():
            return None
        if self._residual_arrays is None:
            self._residual_arrays = (
                np.load(
                    self.tilt_sidecar / "residual_targets.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                ),
                np.load(
                    self.tilt_sidecar / "residual_mask.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                ),
            )
        return self._residual_arrays

    def __getitem__(self, request: BatchRequest) -> dict[str, np.ndarray]:
        arrays = self._open_arrays()
        positions = np.asarray(request.indices, dtype=np.int64)
        common, active, equity_cutoffs, context_cutoffs, dates, decisions = (
            _common_batch(arrays, self.rows, positions, request.valid_count)
        )
        inputs = _build_patch_batch(
            arrays,
            dates,
            equity_cutoffs,
            decisions,
            context_cutoffs,
            active,
        )
        tilt = self._open_tilt_arrays()
        if tilt is None:
            tilt_exposure = np.zeros(active.shape, dtype=np.float32)
        else:
            exposure, ready = tilt
            tilt_exposure = np.asarray(exposure[dates], dtype=np.float32) * np.asarray(
                ready[dates] & active, dtype=np.float32
            )
        output = {**inputs, **common, "tilt_exposure": tilt_exposure}
        residual = self._open_residual_arrays()
        if residual is not None:
            source_targets, source_mask = residual
            targets = np.zeros_like(common["targets"])
            mask = np.zeros_like(common["label_mask"])
            for decision in np.unique(decisions):
                group = np.flatnonzero(decisions == decision)
                targets[group] = source_targets[dates[group], :, int(decision), :]
                mask[group] = source_mask[dates[group], :, int(decision), :]
            padded = np.arange(dates.size) >= request.valid_count
            targets[padded] = 0.0
            mask[padded] = False
            output.update(residual_targets=targets, residual_mask=mask)
        return output


class DateStratifiedBatchSampler(Sampler[BatchRequest]):
    def __init__(
        self, sample_index: pl.DataFrame, runtime: RuntimeSettings, seed: int
    ) -> None:
        self.runtime = runtime
        self.seed = seed
        self.epoch = 0
        self.sample_count = sample_index.height
        self.decision_indices = sample_index["decision_idx"].to_numpy()
        positions: dict[object, list[int]] = {}
        for position, trade_date in enumerate(sample_index["trade_date"]):
            positions.setdefault(trade_date, []).append(position)
        self.dates = tuple(positions)
        self.positions_by_date = {
            value: np.asarray(items) for value, items in positions.items()
        }
        self.replace_dates = len(self.dates) < runtime.effective_batch_size
        self.epoch_sample_count = (
            math.ceil(self.sample_count / runtime.effective_batch_size)
            * runtime.effective_batch_size
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.epoch_sample_count // self.runtime.loader_batch_size

    def __iter__(self) -> Iterator[BatchRequest]:
        generator = np.random.default_rng(self.seed + self.epoch)
        effective = self.runtime.effective_batch_size
        for _ in range(self.epoch_sample_count // effective):
            chosen = generator.choice(
                len(self.dates), effective, replace=self.replace_dates
            )
            indices = [
                int(
                    (choices := self.positions_by_date[self.dates[int(date)]])[
                        generator.integers(len(choices))
                    ]
                )
                for date in chosen
            ]
            indices.sort(key=lambda position: int(self.decision_indices[position]))
            for start in range(0, effective, self.runtime.loader_batch_size):
                values = tuple(indices[start : start + self.runtime.loader_batch_size])
                yield BatchRequest(values, len(values))


class DecisionGroupedBatchSampler(Sampler[BatchRequest]):
    def __init__(self, sample_index: pl.DataFrame, batch_size: int) -> None:
        self.batch_size = batch_size
        order = np.arange(sample_index.height)
        sample_ids = sample_index["sample_id"].to_numpy()
        decisions = sample_index["decision_idx"].to_numpy()
        order = order[np.argsort(sample_ids[order], kind="stable")]
        self.positions = order[np.argsort(decisions[order], kind="stable")]

    def __len__(self) -> int:
        return math.ceil(self.positions.size / self.batch_size)

    def __iter__(self) -> Iterator[BatchRequest]:
        for start in range(0, self.positions.size, self.batch_size):
            values = self.positions[start : start + self.batch_size].tolist()
            valid_count = len(values)
            values.extend([values[-1]] * (self.batch_size - valid_count))
            yield BatchRequest(tuple(values), valid_count)


def tensorize_vectorized_batch(batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {key: torch.from_numpy(value) for key, value in batch.items()}


def seed_worker(_: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _create_loader(
    dataset: VectorizedFeatureDataset,
    sampler: Sampler[BatchRequest],
    runtime: RuntimeSettings,
    seed: int,
) -> DataLoader[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    kwargs: dict[str, object] = {}
    if runtime.num_workers:
        kwargs.update(
            persistent_workers=True,
            prefetch_factor=runtime.prefetch_factor,
            multiprocessing_context="spawn",
        )
    return DataLoader(
        dataset,
        batch_size=None,
        sampler=sampler,
        num_workers=runtime.num_workers,
        pin_memory=True,
        collate_fn=tensorize_vectorized_batch,
        worker_init_fn=seed_worker,
        generator=generator,
        **kwargs,
    )


def create_training_loaders(
    store: Path,
    train_rows: pl.DataFrame,
    validation_rows: pl.DataFrame,
    runtime: RuntimeSettings = GH200_RUNTIME,
    seed: int = 29,
    tilt_sidecar: Path | None = None,
) -> tuple[
    DataLoader[dict[str, torch.Tensor]],
    DataLoader[dict[str, torch.Tensor]],
    DateStratifiedBatchSampler,
]:
    sampler = DateStratifiedBatchSampler(train_rows, runtime, seed)
    train = _create_loader(
        VectorizedFeatureDataset(store, train_rows, tilt_sidecar),
        sampler,
        runtime,
        seed,
    )
    validation = _create_loader(
        VectorizedFeatureDataset(store, validation_rows, tilt_sidecar),
        DecisionGroupedBatchSampler(validation_rows, runtime.evaluation_batch_size),
        runtime,
        seed,
    )
    return train, validation, sampler


def create_evaluation_loader(
    store: Path,
    rows: pl.DataFrame,
    runtime: RuntimeSettings = GH200_RUNTIME,
    seed: int = 29,
    tilt_sidecar: Path | None = None,
) -> DataLoader[dict[str, torch.Tensor]]:
    return _create_loader(
        VectorizedFeatureDataset(store, rows, tilt_sidecar),
        DecisionGroupedBatchSampler(rows, runtime.evaluation_batch_size),
        runtime,
        seed,
    )
