from __future__ import annotations

import math
import random
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
    CANONICAL_DROPPED_LOCAL_SLOTS,
    CANONICAL_NEUTRALIZED_EQUITY_SLOW_INDICES,
    CANONICAL_RETAINED_GLOBAL_SLOTS,
    CONTEXT_COUNT,
    CONTEXT_GENERIC_DYNAMIC_COUNT,
    DECISION_GLOBAL_INDICES,
    EFFECTIVE_BATCH_SIZE,
    EQUITY_ABSOLUTE_START_PATCH,
    EQUITY_COUNT,
    EXPECTED_DECISIONS_PER_DATE,
    FEATURE_STORE_POINTER,
    GLOBAL_CONTEXT_COUNT,
    GLOBAL_CONTEXT_SETTINGS,
    GLOBAL_WINDOW_MINUTES,
    HORIZON_COUNT,
    LOCAL_CONTEXT_COUNT,
    NEURAL_MODELS,
    PATCH_INPUT_WIDTH,
    PATCH_MINUTES,
    PEER_STATE_WIDTH,
    RuntimeSettings,
    SLOW_FEATURE_COUNT,
    TABULAR_FEATURE_COUNT,
    TABULAR_OFFSETS,
    TCNArchitecture,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
    validate_peer_feature_mode,
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
PEER_ARRAY_FILES = ("equity_peer_features.npy", "equity_peer_valid.npy")


@dataclass(frozen=True)
class BatchRequest:
    indices: tuple[int, ...]
    valid_count: int


def resolve_feature_store(pointer: Path = FEATURE_STORE_POINTER) -> Path:
    store = Path(pointer.read_text(encoding="utf-8").strip())
    if not store.is_dir():
        raise FileNotFoundError(store)
    return store


def _validate_sample_index(sample_index: pl.DataFrame) -> None:
    sample_ids = sample_index.get_column("sample_id").to_numpy()
    if not np.array_equal(sample_ids, np.arange(sample_index.height)):
        raise ValueError("sample_id must be sorted, unique, and contiguous")
    expected = list(range(EXPECTED_DECISIONS_PER_DATE))
    per_date = sample_index.group_by("trade_date").agg(pl.col("decision_idx").sort())
    if any(
        values.to_list() != expected for values in per_date.get_column("decision_idx")
    ):
        raise ValueError("Each eligible date must contain decisions 0..54 exactly once")
    decision = pl.col("decision_idx").cast(pl.Int16)
    invalid = sample_index.filter(
        (pl.col("equity_cutoff_index") != 15 + PATCH_MINUTES * decision)
        | (pl.col("context_cutoff_index") != 75 + PATCH_MINUTES * decision)
    )
    if invalid.height:
        raise ValueError("Sample cutoffs violate the causal five-minute grid")


def load_sample_index(store: Path) -> pl.DataFrame:
    rows = pl.read_parquet(store / "sample_index.parquet").sort("sample_id")
    _validate_sample_index(rows)
    return rows


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


def _common_batch(
    arrays: dict[str, np.ndarray],
    rows: dict[str, np.ndarray],
    positions: np.ndarray,
    valid_count: int,
) -> tuple[
    dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    source_dates = rows["date_idx"][positions].astype(np.int64, copy=False)
    source_decisions = rows["decision_idx"][positions].astype(np.int64, copy=False)
    sample_id = rows["sample_id"][positions].astype(np.int64, copy=True)
    date_idx = source_dates.copy()
    decision_idx = source_decisions.copy()
    equity_cutoffs = rows["equity_cutoff_index"][positions]
    context_cutoffs = rows["context_cutoff_index"][positions]
    batch_size = positions.size
    sample_valid_mask = np.arange(batch_size) < valid_count
    active = np.asarray(
        arrays["equity_membership.npy"][source_dates]
        & arrays["equity_data_ready.npy"][source_dates],
        dtype=bool,
    )
    targets = np.zeros((batch_size, EQUITY_COUNT, HORIZON_COUNT), dtype=np.float32)
    label_mask = np.zeros_like(targets, dtype=bool)
    raw_returns = np.zeros_like(targets)
    for decision in np.unique(source_decisions):
        group = np.flatnonzero(source_decisions == decision)
        targets[group] = arrays["targets.npy"][source_dates[group], :, int(decision), :]
        label_mask[group] = arrays["label_mask.npy"][
            source_dates[group], :, int(decision), :
        ]
        raw_returns[group] = arrays["raw_returns.npy"][
            source_dates[group], :, int(decision), :
        ]
    padded = ~sample_valid_mask
    targets[padded] = 0
    label_mask[padded] = False
    raw_returns[padded] = 0
    sample_id[padded] = date_idx[padded] = decision_idx[padded] = -1
    common = {
        "targets": targets,
        "label_mask": label_mask,
        "raw_returns": raw_returns,
        "sample_valid_mask": sample_valid_mask,
        "sample_id": sample_id,
        "date_idx": date_idx,
        "decision_idx": decision_idx,
    }
    return (
        common,
        active,
        equity_cutoffs,
        context_cutoffs,
        source_dates,
        source_decisions,
    )


def _build_peer_state(
    arrays: dict[str, np.ndarray],
    date_idx: np.ndarray,
    equity_cutoffs: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    state = np.zeros((date_idx.size, EQUITY_COUNT, PEER_STATE_WIDTH), dtype=np.float32)
    for cutoff in np.unique(equity_cutoffs):
        group = np.flatnonzero(equity_cutoffs == cutoff)
        minute = int(cutoff) - 1
        numeric = np.asarray(
            arrays["equity_peer_features.npy"][date_idx[group], :, minute, :4],
            dtype=np.float32,
        )
        valid = np.asarray(
            arrays["equity_peer_valid.npy"][date_idx[group], :, minute, :2], dtype=bool
        )
        state[group, :, :4] = np.where(
            np.tile(valid, 2),
            numeric,
            0.0,
        )
        state[group, :, 4:] = valid
    return state * active[..., None]


def _context_readiness(
    arrays: dict[str, np.ndarray],
    date_idx: np.ndarray,
    decision_idx: np.ndarray,
    global_context: str,
) -> tuple[np.ndarray, np.ndarray]:
    if global_context not in GLOBAL_CONTEXT_SETTINGS:
        raise ValueError(f"Invalid global context: {global_context}")
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
    if global_context == "masked":
        global_ready[:] = False
    return local_ready, global_ready


def _build_patch_batch(
    arrays: dict[str, np.ndarray],
    date_idx: np.ndarray,
    equity_cutoffs: np.ndarray,
    decision_idx: np.ndarray,
    context_cutoffs: np.ndarray,
    active: np.ndarray,
    global_context: str | None,
) -> dict[str, np.ndarray]:
    needs_context = global_context is not None
    batch_size = date_idx.size
    instrument_count = EQUITY_COUNT + (CONTEXT_COUNT if needs_context else 0)
    local_ready = global_ready = None
    if needs_context:
        local_ready, global_ready = _context_readiness(
            arrays, date_idx, decision_idx, global_context
        )
    patches = np.zeros(
        (batch_size, instrument_count, ABSOLUTE_PATCH_COUNT, PATCH_INPUT_WIDTH),
        dtype=np.float32,
    )
    history_mask = np.zeros(
        (batch_size, instrument_count, ABSOLUTE_PATCH_COUNT), dtype=bool
    )
    instrument_mask = np.zeros((batch_size, instrument_count), dtype=bool)
    instrument_mask[:, :EQUITY_COUNT] = active
    slow = np.zeros(
        (batch_size, instrument_count, SLOW_FEATURE_COUNT), dtype=np.float32
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
        )
        prefix = prefix.reshape(
            group.size, EQUITY_COUNT, equity_patches, PATCH_INPUT_WIDTH
        )
        patches[group, :EQUITY_COUNT, EQUITY_ABSOLUTE_START_PATCH:state] = (
            prefix * active[group, :, None, None]
        )
        history_mask[group, :EQUITY_COUNT, EQUITY_ABSOLUTE_START_PATCH:state] = active[
            group, :, None
        ]
        if needs_context:
            assert local_ready is not None
            ready = local_ready[group]
            local = np.asarray(
                arrays["context_features.npy"][date_idx[group], :, :context_cutoff, :],
                dtype=np.float32,
            )
            local = local.reshape(
                group.size, LOCAL_CONTEXT_COUNT, state, PATCH_INPUT_WIDTH
            )
            start = EQUITY_COUNT
            patches[group, start : start + LOCAL_CONTEXT_COUNT, :state] = (
                local * ready[..., None, None]
            )
            history_mask[group, start : start + LOCAL_CONTEXT_COUNT, :state] = ready[
                ..., None
            ]
    if needs_context:
        assert local_ready is not None and global_ready is not None
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
        global_grid = np.asarray(
            arrays["global_features.npy"][date_idx], dtype=np.float32
        )
        global_values = global_grid[
            np.arange(batch_size)[:, None, None],
            np.arange(GLOBAL_CONTEXT_COUNT)[None, :, None],
            minutes[:, None, :],
        ].reshape(
            batch_size, GLOBAL_CONTEXT_COUNT, ABSOLUTE_PATCH_COUNT, PATCH_INPUT_WIDTH
        )
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
    return {
        "patches": patches,
        "history_patch_mask": history_mask,
        "instrument_mask": instrument_mask,
        "slow_features": slow,
        "state_position": state_position,
    }


def build_tabular_batch(
    arrays: dict[str, np.ndarray],
    date_idx: np.ndarray,
    decision_idx: np.ndarray,
    equity_cutoffs: np.ndarray,
    context_cutoffs: np.ndarray,
    active: np.ndarray,
    global_context: str,
) -> dict[str, np.ndarray]:
    batch_size = date_idx.size
    local_ready, global_ready = _context_readiness(
        arrays, date_idx, decision_idx, global_context
    )
    global_cutoffs = np.asarray(DECISION_GLOBAL_INDICES)[decision_idx]
    global_grid = np.asarray(arrays["global_features.npy"][date_idx], dtype=np.float32)
    output = np.zeros(
        (batch_size, EQUITY_COUNT, TABULAR_FEATURE_COUNT), dtype=np.float32
    )
    cursor = 0
    equity_slow = np.asarray(
        arrays["equity_slow.npy"][date_idx], dtype=np.float32
    ).copy()
    equity_slow[..., CANONICAL_NEUTRALIZED_EQUITY_SLOW_INDICES] = 0.0
    output[:, :, :SLOW_FEATURE_COUNT] = equity_slow * active[..., None]
    cursor += SLOW_FEATURE_COUNT
    equity_validity: list[np.ndarray] = []
    for offset in TABULAR_OFFSETS:
        block = np.zeros(
            (batch_size, EQUITY_COUNT, PATCH_INPUT_WIDTH // PATCH_MINUTES),
            dtype=np.float32,
        )
        valid = np.zeros((batch_size, EQUITY_COUNT), dtype=bool)
        for cutoff in np.unique(equity_cutoffs):
            group = np.flatnonzero(equity_cutoffs == cutoff)
            minute = int(cutoff) - 1 - offset
            if minute >= 0:
                values = np.asarray(
                    arrays["equity_features.npy"][date_idx[group], :, minute, :],
                    dtype=np.float32,
                )
                group_valid = (values[..., 5] > 0.5) & active[group]
                block[group] = values * group_valid[..., None]
                valid[group] = group_valid
        output[:, :, cursor : cursor + block.shape[-1]] = block
        cursor += block.shape[-1]
        equity_validity.append(valid)
    context_validity: list[np.ndarray] = []
    for slot in range(LOCAL_CONTEXT_COUNT):
        for offset in TABULAR_OFFSETS:
            block = np.zeros(
                (batch_size, CONTEXT_GENERIC_DYNAMIC_COUNT), dtype=np.float32
            )
            valid = np.zeros(batch_size, dtype=bool)
            for cutoff in np.unique(context_cutoffs):
                group = np.flatnonzero(context_cutoffs == cutoff)
                minute = int(cutoff) - 1 - offset
                if minute >= 0:
                    values = np.asarray(
                        arrays["context_features.npy"][
                            date_idx[group], slot, minute, :
                        ],
                        dtype=np.float32,
                    )
                    group_valid = (values[:, 5] > 0.5) & local_ready[group, slot]
                    block[group] = (
                        values[:, :CONTEXT_GENERIC_DYNAMIC_COUNT] * group_valid[:, None]
                    )
                    valid[group] = group_valid
            output[:, :, cursor : cursor + CONTEXT_GENERIC_DYNAMIC_COUNT] = block[
                :, None
            ]
            cursor += CONTEXT_GENERIC_DYNAMIC_COUNT
            context_validity.append(valid)
    global_minutes = global_cutoffs[:, None] - 1 - np.asarray(TABULAR_OFFSETS)[None, :]
    global_values = global_grid[
        np.arange(batch_size)[:, None, None],
        np.arange(GLOBAL_CONTEXT_COUNT)[None, :, None],
        global_minutes[:, None, :],
    ]
    global_validity = (global_values[..., 5] > 0.5) & global_ready[..., None]
    block = (
        global_values[..., :CONTEXT_GENERIC_DYNAMIC_COUNT] * global_validity[..., None]
    ).reshape(batch_size, -1)
    output[:, :, cursor : cursor + block.shape[-1]] = block[:, None]
    cursor += block.shape[-1]
    local_slow = (
        np.asarray(arrays["context_slow.npy"][date_idx], dtype=np.float32)
        * local_ready[..., None]
    ).reshape(batch_size, -1)
    output[:, :, cursor : cursor + local_slow.shape[-1]] = local_slow[:, None]
    cursor += local_slow.shape[-1]
    global_slow = np.asarray(arrays["global_slow.npy"][date_idx], dtype=np.float32)
    decision_slow = global_slow[
        np.arange(batch_size)[:, None],
        np.arange(GLOBAL_CONTEXT_COUNT)[None, :],
        decision_idx[:, None],
    ]
    decision_slow = (decision_slow * global_ready[..., None]).reshape(batch_size, -1)
    output[:, :, cursor : cursor + decision_slow.shape[-1]] = decision_slow[:, None]
    cursor += decision_slow.shape[-1]
    position = decision_idx.astype(np.float32) / (EXPECTED_DECISIONS_PER_DATE - 1)
    output[:, :, cursor] = np.sin(2 * np.pi * position)[:, None]
    output[:, :, cursor + 1] = np.cos(2 * np.pi * position)[:, None]
    cursor += 2
    for valid in equity_validity:
        output[:, :, cursor] = valid
        cursor += 1
    for valid in context_validity:
        output[:, :, cursor] = valid[:, None]
        cursor += 1
    for valid in global_validity.reshape(batch_size, -1).T:
        output[:, :, cursor] = valid[:, None]
        cursor += 1
    for ready in local_ready.T:
        output[:, :, cursor] = ready[:, None]
        cursor += 1
    for ready in global_ready.T:
        output[:, :, cursor] = ready[:, None]
        cursor += 1
    if cursor != TABULAR_FEATURE_COUNT:
        raise RuntimeError(
            f"Tabular construction ended at {cursor}, expected {TABULAR_FEATURE_COUNT}"
        )
    return {
        "tabular_features": output * active[..., None],
        "equity_mask": active.copy(),
    }


class VectorizedFeatureDataset(Dataset[dict[str, np.ndarray]]):
    def __init__(
        self,
        store: Path,
        sample_index: pl.DataFrame,
        model_name: str,
        global_context: str | None,
        tcn_architecture: TCNArchitecture | None = None,
        peer_features: str = "none",
    ) -> None:
        if model_name not in NEURAL_MODELS:
            raise ValueError(f"Neural dataset required, found {model_name}")
        if model_name == "tcn":
            if tcn_architecture is None:
                raise ValueError("TCN architecture is required")
            needs_context = tcn_architecture.fusion_mode in (
                "context_only",
                "context_pooled",
            )
        else:
            if tcn_architecture is not None:
                raise ValueError("TCN architecture is only valid for TCN")
            needs_context = model_name in ("context_only", "context_pooled", "mlp")
        if needs_context != (global_context is not None):
            raise ValueError("Global-context setting does not match model inputs")
        self.store = store
        self.model_name = model_name
        self.needs_context = needs_context
        self.global_context = global_context
        self.peer_features = validate_peer_feature_mode(model_name, peer_features)
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

    def __len__(self) -> int:
        return len(self.rows["sample_id"])

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_arrays"] = None
        return state

    def _open_arrays(self) -> dict[str, np.ndarray]:
        if self._arrays is None:
            filenames = FEATURE_ARRAY_FILES
            if self.model_name != "mlp" and not self.needs_context:
                filenames = tuple(
                    name
                    for name in filenames
                    if not name.startswith(("context_", "global_"))
                )
            if self.peer_features == "selected":
                filenames = (*filenames, *PEER_ARRAY_FILES)
            self._arrays = {
                name: np.load(self.store / name, mmap_mode="r", allow_pickle=False)
                for name in filenames
            }
        return self._arrays

    def __getitem__(self, request: BatchRequest) -> dict[str, np.ndarray]:
        arrays = self._open_arrays()
        positions = np.asarray(request.indices, dtype=np.int64)
        common, active, equity_cutoffs, context_cutoffs, dates, decisions = (
            _common_batch(arrays, self.rows, positions, request.valid_count)
        )
        if self.model_name == "mlp":
            inputs = build_tabular_batch(
                arrays,
                dates,
                decisions,
                equity_cutoffs,
                context_cutoffs,
                active,
                self.global_context,
            )
        else:
            inputs = _build_patch_batch(
                arrays,
                dates,
                equity_cutoffs,
                decisions,
                context_cutoffs,
                active,
                self.global_context,
            )
            if self.peer_features == "selected":
                inputs["peer_state"] = _build_peer_state(
                    arrays, dates, equity_cutoffs, active
                )
        padded = ~common["sample_valid_mask"]
        for value in inputs.values():
            value[padded] = 0
        return {**inputs, **common}


class DateStratifiedMicrobatchSampler(Sampler[BatchRequest]):
    def __init__(
        self, sample_index: pl.DataFrame, runtime: RuntimeSettings, seed: int
    ) -> None:
        self.runtime = runtime
        self.seed = seed
        self.epoch = 0
        self.sample_count = sample_index.height
        self.decision_indices = sample_index.get_column("decision_idx").to_numpy()
        positions: dict[object, list[int]] = {}
        for position, trade_date in enumerate(sample_index.get_column("trade_date")):
            positions.setdefault(trade_date, []).append(position)
        self.dates = tuple(positions)
        self.positions_by_date = {
            date: np.asarray(values) for date, values in positions.items()
        }
        if len(self.dates) < EFFECTIVE_BATCH_SIZE:
            raise ValueError(
                f"Training requires at least {EFFECTIVE_BATCH_SIZE} distinct dates"
            )
        self.epoch_sample_count = (
            math.ceil(self.sample_count / EFFECTIVE_BATCH_SIZE) * EFFECTIVE_BATCH_SIZE
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.epoch_sample_count // self.runtime.microbatch_size

    def __iter__(self) -> Iterator[BatchRequest]:
        generator = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.epoch_sample_count // EFFECTIVE_BATCH_SIZE):
            chosen = generator.choice(
                len(self.dates), EFFECTIVE_BATCH_SIZE, replace=False
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
            for start in range(0, EFFECTIVE_BATCH_SIZE, self.runtime.microbatch_size):
                values = tuple(indices[start : start + self.runtime.microbatch_size])
                yield BatchRequest(values, len(values))


class SequentialPaddedBatchSampler(Sampler[BatchRequest]):
    def __init__(self, row_count: int, batch_size: int) -> None:
        self.row_count = row_count
        self.batch_size = batch_size

    def __len__(self) -> int:
        return math.ceil(self.row_count / self.batch_size)

    def __iter__(self) -> Iterator[BatchRequest]:
        for start in range(0, self.row_count, self.batch_size):
            values = list(range(start, min(start + self.batch_size, self.row_count)))
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
    model_name: str,
    global_context: str | None,
    runtime: RuntimeSettings,
    seed: int,
    tcn_architecture: TCNArchitecture | None = None,
    peer_features: str = "none",
) -> tuple[
    DataLoader[dict[str, torch.Tensor]],
    DataLoader[dict[str, torch.Tensor]],
    DateStratifiedMicrobatchSampler,
]:
    sampler = DateStratifiedMicrobatchSampler(train_rows, runtime, seed)
    train = _create_loader(
        VectorizedFeatureDataset(
            store,
            train_rows,
            model_name,
            global_context,
            tcn_architecture,
            peer_features,
        ),
        sampler,
        runtime,
        seed,
    )
    validation = _create_loader(
        VectorizedFeatureDataset(
            store,
            validation_rows,
            model_name,
            global_context,
            tcn_architecture,
            peer_features,
        ),
        SequentialPaddedBatchSampler(
            validation_rows.height, runtime.evaluation_batch_size
        ),
        runtime,
        seed,
    )
    return train, validation, sampler


def create_evaluation_loader(
    store: Path,
    rows: pl.DataFrame,
    model_name: str,
    global_context: str | None,
    runtime: RuntimeSettings,
    seed: int,
    tcn_architecture: TCNArchitecture | None = None,
    peer_features: str = "none",
) -> DataLoader[dict[str, torch.Tensor]]:
    return _create_loader(
        VectorizedFeatureDataset(
            store, rows, model_name, global_context, tcn_architecture, peer_features
        ),
        SequentialPaddedBatchSampler(rows.height, runtime.evaluation_batch_size),
        runtime,
        seed,
    )
