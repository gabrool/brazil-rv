from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .contract import (
    ABSOLUTE_PATCH_COUNT,
    CONTEXT_COUNT,
    CONTEXT_SYMBOLS,
    EFFECTIVE_BATCH_SIZE,
    EQUITY_ABSOLUTE_START_PATCH,
    EQUITY_COUNT,
    EXPECTED_ARRAY_SHAPES,
    EXPECTED_DECISIONS_PER_DATE,
    EXPECTED_SAMPLE_COUNT,
    FEATURE_CONTRACT_VERSION,
    FEATURE_STORE_POINTER,
    INSTRUMENT_COUNT,
    PATCH_INPUT_WIDTH,
    PATCH_MINUTES,
    RuntimeProfile,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
)

FEATURE_ARRAY_FILES = (
    "equity_features.npy",
    "equity_slow.npy",
    "equity_membership.npy",
    "equity_data_ready.npy",
    "context_features.npy",
    "context_slow.npy",
    "targets.npy",
    "label_mask.npy",
    "raw_returns.npy",
)
CACHE_BUFFER_BYTES = 64 * 1024**2


@dataclass(frozen=True)
class BatchRequest:
    indices: tuple[int, ...]
    valid_count: int


@dataclass(frozen=True)
class CacheWarmupReport:
    bytes_read: int
    files_read: int
    seconds: float


def resolve_feature_store(pointer: Path = FEATURE_STORE_POINTER) -> Path:
    store = Path(pointer.read_text(encoding="utf-8").strip())
    if not store.is_dir():
        raise FileNotFoundError(f"Canonical feature pointer resolves to {store}")
    return store


def load_sample_index(store: Path) -> pl.DataFrame:
    return pl.read_parquet(store / "sample_index.parquet").sort("sample_id")


def _validate_sample_index(sample_index: pl.DataFrame) -> None:
    if sample_index.get_column("sample_id").n_unique() != sample_index.height:
        raise ValueError("sample_id must be unique")
    expected_decisions = list(range(EXPECTED_DECISIONS_PER_DATE))
    decisions_by_date = sample_index.group_by("trade_date").agg(
        pl.col("decision_idx").sort()
    )
    if any(
        decisions != expected_decisions
        for decisions in decisions_by_date.get_column("decision_idx").to_list()
    ):
        raise ValueError("Every eligible date must contain decision_idx exactly 0..54")
    decision = pl.col("decision_idx").cast(pl.Int16)
    equity_cutoff = pl.col("equity_cutoff_index")
    context_cutoff = pl.col("context_cutoff_index")
    invalid_cutoffs = sample_index.filter(
        (equity_cutoff != 15 + PATCH_MINUTES * decision)
        | (context_cutoff != 75 + PATCH_MINUTES * decision)
        | (equity_cutoff % PATCH_MINUTES != 0)
        | (context_cutoff % PATCH_MINUTES != 0)
    )
    if invalid_cutoffs.height:
        raise ValueError("Sample-index cutoffs violate the fixed patch grid")


def validate_feature_store(store: Path) -> pl.DataFrame:
    manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
    if manifest["contract_version"] != FEATURE_CONTRACT_VERSION:
        raise ValueError(
            f"Expected feature contract {FEATURE_CONTRACT_VERSION}, "
            f"found {manifest['contract_version']}"
        )
    sample_index = load_sample_index(store)
    if sample_index.height != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLE_COUNT} samples, found {sample_index.height}"
        )
    _validate_sample_index(sample_index)
    for filename, expected_shape in EXPECTED_ARRAY_SHAPES.items():
        array = np.load(store / filename, mmap_mode="r")
        if array.shape != expected_shape:
            raise ValueError(
                f"Expected {filename} shape {expected_shape}, found {array.shape}"
            )
    context_symbols = tuple(
        pl.read_parquet(store / "context_index.parquet")
        .sort("context_slot")
        .get_column("symbol")
    )
    if context_symbols != CONTEXT_SYMBOLS:
        raise ValueError("Context axis does not match the fixed contract")
    splits = split_sample_index(sample_index)
    split_ids = [
        set(splits[name].get_column("sample_id").to_list())
        for name in ("train", "validation", "test")
    ]
    if any(
        split_ids[left] & split_ids[right]
        for left in range(len(split_ids))
        for right in range(left + 1, len(split_ids))
    ):
        raise ValueError("Training, validation, and test rows must be disjoint")
    if splits["train"].get_column("trade_date").n_unique() < EFFECTIVE_BATCH_SIZE:
        raise ValueError("Training requires at least 32 distinct dates")
    return sample_index


def select_sample_split(sample_index: pl.DataFrame, split: str) -> pl.DataFrame:
    trade_date = pl.col("trade_date")
    filters = {
        "train": trade_date.is_between(TRAIN_START, TRAIN_END),
        "embargo_1": (trade_date > TRAIN_END) & (trade_date < VALIDATION_START),
        "validation": trade_date.is_between(VALIDATION_START, VALIDATION_END),
        "embargo_2": (trade_date > VALIDATION_END) & (trade_date < TEST_START),
        "test": trade_date.is_between(TEST_START, TEST_END),
    }
    return sample_index.filter(filters[split])


def split_sample_index(sample_index: pl.DataFrame) -> dict[str, pl.DataFrame]:
    return {
        split: select_sample_split(sample_index, split)
        for split in (
            "train",
            "embargo_1",
            "validation",
            "embargo_2",
            "test",
        )
    }


def warm_feature_store_cache(store: Path) -> CacheWarmupReport:
    started = time.perf_counter()
    buffer = bytearray(CACHE_BUFFER_BYTES)
    bytes_read = 0
    for filename in FEATURE_ARRAY_FILES:
        with (store / filename).open("rb", buffering=0) as source:
            while read_count := source.readinto(buffer):
                bytes_read += read_count
    return CacheWarmupReport(
        bytes_read=bytes_read,
        files_read=len(FEATURE_ARRAY_FILES),
        seconds=time.perf_counter() - started,
    )


class VectorizedFeatureDataset(Dataset[dict[str, np.ndarray]]):
    def __init__(self, store: Path, sample_index: pl.DataFrame) -> None:
        self.store = store
        self.rows = {
            column: np.asarray(sample_index.get_column(column).to_list())
            for column in (
                "sample_id",
                "date_idx",
                "decision_idx",
                "equity_cutoff_index",
                "context_cutoff_index",
            )
        }
        self._arrays: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.rows["sample_id"])

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_arrays"] = None
        return state

    def _open_arrays(self) -> dict[str, np.ndarray]:
        if self._arrays is None:
            self._arrays = {
                filename: np.load(self.store / filename, mmap_mode="r")
                for filename in FEATURE_ARRAY_FILES
            }
        return self._arrays

    def __getitem__(self, request: BatchRequest) -> dict[str, np.ndarray]:
        arrays = self._open_arrays()
        positions = np.asarray(request.indices, dtype=np.int64)
        sample_id = self.rows["sample_id"][positions].astype(np.int64, copy=True)
        date_idx = self.rows["date_idx"][positions].astype(np.int64, copy=True)
        decision_idx = self.rows["decision_idx"][positions].astype(np.int64, copy=True)
        equity_cutoffs = self.rows["equity_cutoff_index"][positions]
        context_cutoffs = self.rows["context_cutoff_index"][positions]
        batch_size = positions.size
        sample_valid_mask = np.arange(batch_size) < request.valid_count

        active_equities = np.asarray(
            arrays["equity_membership.npy"][date_idx]
            & arrays["equity_data_ready.npy"][date_idx],
            dtype=bool,
        )
        instrument_mask = np.ones((batch_size, INSTRUMENT_COUNT), dtype=bool)
        instrument_mask[:, :EQUITY_COUNT] = active_equities
        patches = np.zeros(
            (
                batch_size,
                INSTRUMENT_COUNT,
                ABSOLUTE_PATCH_COUNT,
                PATCH_INPUT_WIDTH,
            ),
            dtype=np.float32,
        )
        history_patch_mask = np.zeros(
            (batch_size, INSTRUMENT_COUNT, ABSOLUTE_PATCH_COUNT), dtype=bool
        )
        targets = np.zeros((batch_size, EQUITY_COUNT, 3), dtype=np.float32)
        label_mask = np.zeros((batch_size, EQUITY_COUNT, 3), dtype=bool)
        raw_returns = np.zeros((batch_size, EQUITY_COUNT, 3), dtype=np.float32)
        state_position = np.empty(batch_size, dtype=np.int64)

        for decision in np.unique(decision_idx):
            group = np.flatnonzero(decision_idx == decision)
            equity_cutoff = int(equity_cutoffs[group[0]])
            context_cutoff = int(context_cutoffs[group[0]])
            group_state_position = context_cutoff // PATCH_MINUTES
            equity_patch_count = equity_cutoff // PATCH_MINUTES
            if EQUITY_ABSOLUTE_START_PATCH + equity_patch_count != group_state_position:
                raise ValueError("Equity and context patch clocks are misaligned")
            state_position[group] = group_state_position
            equity_prefix = np.asarray(
                arrays["equity_features.npy"][date_idx[group], :, :equity_cutoff, :],
                dtype=np.float32,
            ).reshape(
                group.size,
                EQUITY_COUNT,
                equity_patch_count,
                PATCH_INPUT_WIDTH,
            )
            patches[
                group,
                :EQUITY_COUNT,
                EQUITY_ABSOLUTE_START_PATCH:group_state_position,
            ] = equity_prefix
            history_patch_mask[
                group,
                :EQUITY_COUNT,
                EQUITY_ABSOLUTE_START_PATCH:group_state_position,
            ] = active_equities[group, :, None]
            context_prefix = np.asarray(
                arrays["context_features.npy"][date_idx[group], :, :context_cutoff, :],
                dtype=np.float32,
            ).reshape(
                group.size,
                CONTEXT_COUNT,
                group_state_position,
                PATCH_INPUT_WIDTH,
            )
            patches[group, EQUITY_COUNT:, :group_state_position] = context_prefix
            history_patch_mask[group, EQUITY_COUNT:, :group_state_position] = True
            targets[group] = arrays["targets.npy"][date_idx[group], :, int(decision), :]
            label_mask[group] = arrays["label_mask.npy"][
                date_idx[group], :, int(decision), :
            ]
            raw_returns[group] = arrays["raw_returns.npy"][
                date_idx[group], :, int(decision), :
            ]

        patches[:, :EQUITY_COUNT] *= active_equities[:, :, None, None]
        slow_features = np.zeros((batch_size, INSTRUMENT_COUNT, 3), dtype=np.float32)
        slow_features[:, :EQUITY_COUNT, 0] = arrays["equity_slow.npy"][date_idx, :, 0]
        slow_features[:, EQUITY_COUNT:] = arrays["context_slow.npy"][date_idx]

        padded = ~sample_valid_mask
        targets[padded] = 0.0
        label_mask[padded] = False
        raw_returns[padded] = 0.0
        sample_id[padded] = -1
        date_idx[padded] = -1
        decision_idx[padded] = -1
        return {
            "patches": patches,
            "history_patch_mask": history_patch_mask,
            "instrument_mask": instrument_mask,
            "slow_features": slow_features,
            "state_position": state_position,
            "targets": targets,
            "label_mask": label_mask,
            "raw_returns": raw_returns,
            "sample_valid_mask": sample_valid_mask,
            "sample_id": sample_id,
            "date_idx": date_idx,
            "decision_idx": decision_idx,
        }


class DateStratifiedMicrobatchSampler(Sampler[BatchRequest]):
    def __init__(
        self, sample_index: pl.DataFrame, profile: RuntimeProfile, seed: int
    ) -> None:
        self.profile = profile
        self.seed = seed
        self.epoch = 0
        self.sample_count = sample_index.height
        positions_by_date: dict[object, list[int]] = {}
        for position, trade_date in enumerate(
            sample_index.get_column("trade_date").to_list()
        ):
            positions_by_date.setdefault(trade_date, []).append(position)
        self.dates = tuple(positions_by_date)
        self.positions_by_date = {
            trade_date: np.asarray(positions, dtype=np.int64)
            for trade_date, positions in positions_by_date.items()
        }
        if len(self.dates) < EFFECTIVE_BATCH_SIZE:
            raise ValueError("Date-stratified sampling requires 32 distinct dates")
        self.epoch_sample_count = (
            math.ceil(self.sample_count / EFFECTIVE_BATCH_SIZE) * EFFECTIVE_BATCH_SIZE
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.epoch_sample_count // self.profile.microbatch_size

    def __iter__(self) -> Iterator[BatchRequest]:
        generator = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.epoch_sample_count // EFFECTIVE_BATCH_SIZE):
            selected_dates = generator.choice(
                len(self.dates), size=EFFECTIVE_BATCH_SIZE, replace=False
            )
            effective_indices = []
            for date_position in selected_dates:
                trade_date = self.dates[int(date_position)]
                choices = self.positions_by_date[trade_date]
                effective_indices.append(int(choices[generator.integers(len(choices))]))
            for start in range(0, EFFECTIVE_BATCH_SIZE, self.profile.microbatch_size):
                indices = tuple(
                    effective_indices[start : start + self.profile.microbatch_size]
                )
                yield BatchRequest(indices=indices, valid_count=len(indices))


class SequentialPaddedBatchSampler(Sampler[BatchRequest]):
    def __init__(self, row_count: int, batch_size: int) -> None:
        self.row_count = row_count
        self.batch_size = batch_size

    def __len__(self) -> int:
        return math.ceil(self.row_count / self.batch_size)

    def __iter__(self) -> Iterator[BatchRequest]:
        for start in range(0, self.row_count, self.batch_size):
            real_indices = list(
                range(start, min(start + self.batch_size, self.row_count))
            )
            valid_count = len(real_indices)
            real_indices.extend([real_indices[-1]] * (self.batch_size - valid_count))
            yield BatchRequest(tuple(real_indices), valid_count)


def tensorize_vectorized_batch(
    batch: dict[str, np.ndarray],
) -> dict[str, torch.Tensor]:
    return {key: torch.from_numpy(value) for key, value in batch.items()}


def seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _create_loader(
    dataset: VectorizedFeatureDataset,
    sampler: Sampler[BatchRequest],
    profile: RuntimeProfile,
    seed: int,
) -> DataLoader[dict[str, torch.Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=None,
        sampler=sampler,
        num_workers=profile.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=profile.prefetch_factor,
        in_order=True,
        multiprocessing_context="spawn",
        collate_fn=tensorize_vectorized_batch,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def create_training_loaders(
    store: Path,
    train_rows: pl.DataFrame,
    validation_rows: pl.DataFrame,
    profile: RuntimeProfile,
    seed: int,
) -> tuple[
    DataLoader[dict[str, torch.Tensor]],
    DataLoader[dict[str, torch.Tensor]],
    DateStratifiedMicrobatchSampler,
]:
    sampler = DateStratifiedMicrobatchSampler(train_rows, profile, seed)
    train_loader = _create_loader(
        VectorizedFeatureDataset(store, train_rows), sampler, profile, seed
    )
    validation_loader = _create_loader(
        VectorizedFeatureDataset(store, validation_rows),
        SequentialPaddedBatchSampler(
            validation_rows.height, profile.evaluation_batch_size
        ),
        profile,
        seed,
    )
    return train_loader, validation_loader, sampler


def create_evaluation_loader(
    store: Path,
    rows: pl.DataFrame,
    profile: RuntimeProfile,
    seed: int,
) -> DataLoader[dict[str, torch.Tensor]]:
    return _create_loader(
        VectorizedFeatureDataset(store, rows),
        SequentialPaddedBatchSampler(rows.height, profile.evaluation_batch_size),
        profile,
        seed,
    )
