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

from ..preprocessing.peer_features import validate_peer_arrays
from .context_ablation import NO_CONTEXT_ABLATION, ResolvedContextAblation
from .contract import (
    ABSOLUTE_PATCH_COUNT,
    CONTEXT_GENERIC_DYNAMIC_COUNT,
    DECISION_GLOBAL_INDICES,
    GLOBAL_CONTEXT_COUNT,
    GLOBAL_CONTEXT_SETTINGS,
    GLOBAL_CONTEXT_SYMBOLS,
    GLOBAL_WINDOW_MINUTES,
    EFFECTIVE_BATCH_SIZE,
    EQUITY_ABSOLUTE_START_PATCH,
    EQUITY_COUNT,
    EXPECTED_ARRAY_SHAPES,
    EXPECTED_DECISIONS_PER_DATE,
    EXPECTED_ELIGIBLE_DATE_COUNT,
    EXPECTED_FIRST_ELIGIBLE_DATE,
    EXPECTED_LAST_ELIGIBLE_DATE,
    EXPECTED_SAMPLE_COUNT,
    EXPECTED_SPLIT_DATE_COUNTS,
    EXPECTED_SPLIT_SAMPLE_COUNTS,
    FEATURE_CONTRACT_VERSION,
    FEATURE_STORE_POINTER,
    HORIZON_COUNT,
    INSTRUMENT_COUNT,
    LOCAL_CONTEXT_AVAILABILITY_RULE,
    LOCAL_CONTEXT_COUNT,
    LOCAL_CONTEXT_SYMBOLS,
    MIN_ACTIVE_EQUITIES,
    NEURAL_MODELS,
    PATCH_INPUT_WIDTH,
    PATCH_MINUTES,
    PEER_FEATURE_MODES,
    PEER_STATE_WIDTH,
    RuntimeSettings,
    SLOW_FEATURE_COUNT,
    TABULAR_FEATURE_COUNT,
    TABULAR_OFFSETS,
    TEST_END,
    TEST_START,
    TCNArchitecture,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
    validate_peer_feature_mode,
)
from .feature_ablation import (
    NO_FEATURE_ABLATION,
    ResolvedFeatureAblation,
    apply_feature_ablation_to_slow_features,
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
PEER_ARRAY_FILES = (
    "equity_peer_features.npy",
    "equity_peer_valid.npy",
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


@dataclass(frozen=True)
class TabularRowBatch:
    features: np.ndarray | torch.Tensor
    labels: np.ndarray | torch.Tensor
    weights: np.ndarray | torch.Tensor
    sample_id: np.ndarray
    date_idx: np.ndarray
    decision_idx: np.ndarray
    equity_slot: np.ndarray


def resolve_feature_store(pointer: Path = FEATURE_STORE_POINTER) -> Path:
    store = Path(pointer.read_text(encoding="utf-8").strip())
    if not store.is_dir():
        raise FileNotFoundError(f"Canonical feature pointer resolves to {store}")
    return store


def load_sample_index(store: Path) -> pl.DataFrame:
    return pl.read_parquet(store / "sample_index.parquet").sort("sample_id")


def _validate_sample_index(sample_index: pl.DataFrame) -> None:
    sample_ids = sample_index.get_column("sample_id").to_numpy()
    if np.unique(sample_ids).size != sample_index.height or not np.array_equal(
        np.sort(sample_ids), np.arange(sample_index.height)
    ):
        raise ValueError("sample_id must be unique and contiguous from zero")
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
    if (
        manifest.get("local_context_availability_rule")
        != LOCAL_CONTEXT_AVAILABILITY_RULE
    ):
        raise ValueError(
            "Feature manifest has the wrong local-context availability rule"
        )
    sample_index = load_sample_index(store)
    if sample_index.height != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLE_COUNT} samples, found {sample_index.height}"
        )
    _validate_sample_index(sample_index)
    for filename, expected_shape in EXPECTED_ARRAY_SHAPES.items():
        array = np.load(store / filename, mmap_mode="r", allow_pickle=False)
        if array.shape != expected_shape:
            raise ValueError(
                f"Expected {filename} shape {expected_shape}, found {array.shape}"
            )
    validate_peer_arrays(
        np.load(store / "equity_peer_features.npy", mmap_mode="r", allow_pickle=False),
        np.load(store / "equity_peer_valid.npy", mmap_mode="r", allow_pickle=False),
    )
    context_ready = np.load(
        store / "context_data_ready.npy", mmap_mode="r", allow_pickle=False
    )
    if context_ready.dtype != np.dtype(bool):
        raise ValueError("context_data_ready.npy must have boolean dtype")
    membership = np.load(
        store / "equity_membership.npy", mmap_mode="r", allow_pickle=False
    )
    equity_ready = np.load(
        store / "equity_data_ready.npy", mmap_mode="r", allow_pickle=False
    )
    active_by_date = (membership & equity_ready).sum(axis=1)
    expected_eligible_dates = np.flatnonzero(active_by_date >= MIN_ACTIVE_EQUITIES)
    sample_dates = np.sort(
        sample_index.get_column("date_idx").unique().to_numpy().astype(np.int64)
    )
    if not np.array_equal(sample_dates, expected_eligible_dates):
        raise ValueError(
            "Sample dates must equal the dates meeting only the active-equity rule"
        )
    if sample_dates.size != EXPECTED_ELIGIBLE_DATE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ELIGIBLE_DATE_COUNT} eligible dates, "
            f"found {sample_dates.size}"
        )
    if (
        sample_index.get_column("trade_date").min() != EXPECTED_FIRST_ELIGIBLE_DATE
        or sample_index.get_column("trade_date").max() != EXPECTED_LAST_ELIGIBLE_DATE
    ):
        raise ValueError("Feature-eligible date boundaries do not match the contract")
    repeated_active = active_by_date[
        sample_index.get_column("date_idx").to_numpy().astype(np.int64)
    ]
    if np.any(repeated_active < MIN_ACTIVE_EQUITIES) or not np.array_equal(
        repeated_active,
        sample_index.get_column("active_equity_count").to_numpy(),
    ):
        raise ValueError("Sample-index active-equity counts are inconsistent")
    local_symbols = tuple(
        pl.read_parquet(store / "context_index.parquet")
        .sort("context_slot")
        .get_column("symbol")
    )
    if local_symbols != LOCAL_CONTEXT_SYMBOLS:
        raise ValueError("Local context axis does not match the fixed contract")
    global_symbols = tuple(
        pl.scan_parquet(store / "global_context_index.parquet")
        .select("global_slot", "continuous_symbol")
        .unique()
        .sort("global_slot")
        .collect()
        .get_column("continuous_symbol")
    )
    if global_symbols != GLOBAL_CONTEXT_SYMBOLS:
        raise ValueError("Global context axis does not match the fixed contract")
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
    for split, expected_dates in EXPECTED_SPLIT_DATE_COUNTS.items():
        actual_dates = splits[split].get_column("trade_date").n_unique()
        actual_samples = splits[split].height
        if (
            actual_dates != expected_dates
            or actual_samples != EXPECTED_SPLIT_SAMPLE_COUNTS[split]
        ):
            raise ValueError(f"{split} split counts do not match the contract")
    if splits["train"].get_column("trade_date").n_unique() < EFFECTIVE_BATCH_SIZE:
        raise ValueError(
            f"Training requires at least {EFFECTIVE_BATCH_SIZE} distinct dates"
        )
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
        for split in ("train", "embargo_1", "validation", "embargo_2", "test")
    }


def warm_feature_store_cache(
    store: Path, peer_features: str = "none"
) -> CacheWarmupReport:
    if peer_features not in PEER_FEATURE_MODES:
        raise ValueError(f"Invalid peer-feature mode: {peer_features}")
    filenames = FEATURE_ARRAY_FILES + (
        PEER_ARRAY_FILES if peer_features != "none" else ()
    )
    started = time.perf_counter()
    buffer = bytearray(CACHE_BUFFER_BYTES)
    bytes_read = 0
    for filename in filenames:
        with (store / filename).open("rb", buffering=0) as source:
            while read_count := source.readinto(buffer):
                bytes_read += read_count
    return CacheWarmupReport(
        bytes_read=bytes_read,
        files_read=len(filenames),
        seconds=time.perf_counter() - started,
    )


def _common_batch(
    arrays: dict[str, np.ndarray],
    rows: dict[str, np.ndarray],
    positions: np.ndarray,
    valid_count: int,
) -> tuple[
    dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    sample_id = rows["sample_id"][positions].astype(np.int64, copy=True)
    date_idx = rows["date_idx"][positions].astype(np.int64, copy=True)
    decision_idx = rows["decision_idx"][positions].astype(np.int64, copy=True)
    equity_cutoffs = rows["equity_cutoff_index"][positions]
    context_cutoffs = rows["context_cutoff_index"][positions]
    batch_size = positions.size
    sample_valid_mask = np.arange(batch_size) < valid_count
    active_equities = np.asarray(
        arrays["equity_membership.npy"][date_idx]
        & arrays["equity_data_ready.npy"][date_idx],
        dtype=bool,
    )
    targets = np.zeros((batch_size, EQUITY_COUNT, HORIZON_COUNT), dtype=np.float32)
    label_mask = np.zeros_like(targets, dtype=bool)
    raw_returns = np.zeros_like(targets)
    for decision in np.unique(decision_idx):
        group = np.flatnonzero(decision_idx == decision)
        targets[group] = arrays["targets.npy"][date_idx[group], :, int(decision), :]
        label_mask[group] = arrays["label_mask.npy"][
            date_idx[group], :, int(decision), :
        ]
        raw_returns[group] = arrays["raw_returns.npy"][
            date_idx[group], :, int(decision), :
        ]
    padded = ~sample_valid_mask
    targets[padded] = 0.0
    label_mask[padded] = False
    raw_returns[padded] = 0.0
    sample_id[padded] = -1
    date_idx[padded] = -1
    decision_idx[padded] = -1
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
        active_equities,
        equity_cutoffs,
        context_cutoffs,
        rows["date_idx"][positions].astype(np.int64, copy=False),
        rows["decision_idx"][positions].astype(np.int64, copy=False),
    )


def _build_peer_state(
    arrays: dict[str, np.ndarray],
    date_idx: np.ndarray,
    equity_cutoffs: np.ndarray,
    active_equities: np.ndarray,
    peer_features: str,
) -> np.ndarray:
    batch_size = date_idx.size
    state = np.zeros((batch_size, EQUITY_COUNT, PEER_STATE_WIDTH), dtype=np.float32)
    if peer_features == "masked_control":
        return state
    if peer_features not in ("selected", "selected_plus_issuer"):
        raise ValueError(f"Invalid peer-feature batch mode: {peer_features}")

    for equity_cutoff in np.unique(equity_cutoffs):
        group = np.flatnonzero(equity_cutoffs == equity_cutoff)
        feature_minute_idx = int(equity_cutoff) - 1
        numeric = np.asarray(
            arrays["equity_peer_features.npy"][
                date_idx[group], :, feature_minute_idx, :
            ],
            dtype=np.float32,
        )
        valid = np.asarray(
            arrays["equity_peer_valid.npy"][date_idx[group], :, feature_minute_idx, :],
            dtype=bool,
        )
        selected_15 = valid[..., 0]
        selected_60 = valid[..., 1]
        state[group, :, 0] = np.where(selected_15, numeric[..., 0], 0.0)
        state[group, :, 1] = np.where(selected_60, numeric[..., 1], 0.0)
        state[group, :, 2] = np.where(selected_15, numeric[..., 2], 0.0)
        state[group, :, 3] = np.where(selected_60, numeric[..., 3], 0.0)
        state[group, :, 4] = selected_15
        state[group, :, 5] = selected_60
        if peer_features == "selected_plus_issuer":
            issuer_15 = valid[..., 2]
            issuer_60 = valid[..., 3]
            state[group, :, 6] = np.where(issuer_15, numeric[..., 4], 0.0)
            state[group, :, 7] = np.where(issuer_60, numeric[..., 5], 0.0)
            state[group, :, 8] = issuer_15
            state[group, :, 9] = issuer_60

    state *= active_equities[..., None]
    return state


def _build_patch_batch(
    arrays: dict[str, np.ndarray],
    date_idx: np.ndarray,
    equity_cutoffs: np.ndarray,
    decision_idx: np.ndarray,
    context_cutoffs: np.ndarray,
    active_equities: np.ndarray,
    global_context: str | None,
    context_ablation: ResolvedContextAblation = NO_CONTEXT_ABLATION,
    feature_ablation: ResolvedFeatureAblation = NO_FEATURE_ABLATION,
) -> dict[str, np.ndarray]:
    batch_size = date_idx.size
    needs_context = global_context is not None
    if needs_context and global_context not in GLOBAL_CONTEXT_SETTINGS:
        raise ValueError(f"Invalid global context setting: {global_context}")
    if context_ablation.key != "none" and not needs_context:
        raise ValueError("Context-free batches do not accept a context ablation")
    if context_ablation.key != "none" and global_context != "enabled":
        raise ValueError("Context ablations require enabled global context")
    local_ready = (
        np.asarray(arrays["context_data_ready.npy"][date_idx], dtype=bool).copy()
        if needs_context
        else None
    )
    if local_ready is not None and context_ablation.local_slots:
        local_ready[:, context_ablation.local_slots] = False
    global_ready = None
    if needs_context:
        readiness = np.asarray(arrays["global_data_ready.npy"][date_idx], dtype=bool)
        global_ready = readiness[
            np.arange(batch_size)[:, None],
            np.arange(GLOBAL_CONTEXT_COUNT)[None, :],
            decision_idx[:, None],
        ].copy()
        if global_context == "masked":
            global_ready[:] = False
        elif context_ablation.global_slots:
            global_ready[:, context_ablation.global_slots] = False

    patches = np.zeros(
        (batch_size, INSTRUMENT_COUNT, ABSOLUTE_PATCH_COUNT, PATCH_INPUT_WIDTH),
        dtype=np.float32,
    )
    history_patch_mask = np.zeros(
        (batch_size, INSTRUMENT_COUNT, ABSOLUTE_PATCH_COUNT), dtype=bool
    )
    instrument_mask = np.zeros((batch_size, INSTRUMENT_COUNT), dtype=bool)
    instrument_mask[:, :EQUITY_COUNT] = active_equities
    slow_features = np.zeros(
        (batch_size, INSTRUMENT_COUNT, SLOW_FEATURE_COUNT), dtype=np.float32
    )
    equity_slow = np.asarray(
        arrays["equity_slow.npy"][date_idx], dtype=np.float32
    ).copy()
    if context_ablation.equity_slow_indices:
        equity_slow[..., context_ablation.equity_slow_indices] = 0.0
    slow_features[:, :EQUITY_COUNT] = equity_slow * active_equities[..., None]
    state_position = np.empty(batch_size, dtype=np.int64)

    for equity_cutoff in np.unique(equity_cutoffs):
        group = np.flatnonzero(equity_cutoffs == equity_cutoff)
        context_cutoff = int(context_cutoffs[group[0]])
        state = context_cutoff // PATCH_MINUTES
        equity_patch_count = int(equity_cutoff) // PATCH_MINUTES
        if EQUITY_ABSOLUTE_START_PATCH + equity_patch_count != state:
            raise ValueError("Equity and context patch clocks are misaligned")
        state_position[group] = state
        equity_prefix = np.asarray(
            arrays["equity_features.npy"][date_idx[group], :, : int(equity_cutoff), :],
            dtype=np.float32,
        ).reshape(group.size, EQUITY_COUNT, equity_patch_count, PATCH_INPUT_WIDTH)
        patches[
            group,
            :EQUITY_COUNT,
            EQUITY_ABSOLUTE_START_PATCH:state,
        ] = equity_prefix * active_equities[group, :, None, None]
        history_patch_mask[
            group,
            :EQUITY_COUNT,
            EQUITY_ABSOLUTE_START_PATCH:state,
        ] = active_equities[group, :, None]
        if needs_context:
            ready = local_ready[group]
            context_prefix = np.asarray(
                arrays["context_features.npy"][date_idx[group], :, :context_cutoff, :],
                dtype=np.float32,
            ).reshape(group.size, LOCAL_CONTEXT_COUNT, state, PATCH_INPUT_WIDTH)
            patches[
                group, EQUITY_COUNT : EQUITY_COUNT + LOCAL_CONTEXT_COUNT, :state
            ] = context_prefix * ready[..., None, None]
            history_patch_mask[
                group, EQUITY_COUNT : EQUITY_COUNT + LOCAL_CONTEXT_COUNT, :state
            ] = ready[..., None]

    if needs_context:
        global_start = EQUITY_COUNT + LOCAL_CONTEXT_COUNT
        instrument_mask[:, EQUITY_COUNT:global_start] = local_ready
        slow_features[:, EQUITY_COUNT:global_start] = (
            np.asarray(arrays["context_slow.npy"][date_idx], dtype=np.float32)
            * local_ready[..., None]
        )
        global_cutoffs = np.asarray(DECISION_GLOBAL_INDICES)[decision_idx]
        minute_indices = (
            global_cutoffs[:, None]
            - GLOBAL_WINDOW_MINUTES
            + np.arange(GLOBAL_WINDOW_MINUTES)[None, :]
        )
        global_grid = np.asarray(
            arrays["global_features.npy"][date_idx], dtype=np.float32
        )
        global_prefix = global_grid[
            np.arange(batch_size)[:, None, None],
            np.arange(GLOBAL_CONTEXT_COUNT)[None, :, None],
            minute_indices[:, None, :],
        ].reshape(
            batch_size,
            GLOBAL_CONTEXT_COUNT,
            ABSOLUTE_PATCH_COUNT,
            PATCH_INPUT_WIDTH,
        )
        patches[:, global_start:] = global_prefix * global_ready[..., None, None]
        history_patch_mask[:, global_start:] = global_ready[..., None]
        instrument_mask[:, global_start:] = global_ready
        global_slow = np.asarray(arrays["global_slow.npy"][date_idx], dtype=np.float32)
        decision_slow = global_slow[
            np.arange(batch_size)[:, None],
            np.arange(GLOBAL_CONTEXT_COUNT)[None, :],
            decision_idx[:, None],
        ]
        slow_features[:, global_start:] = decision_slow * global_ready[..., None]
    slow_features = apply_feature_ablation_to_slow_features(
        slow_features, feature_ablation
    )
    return {
        "patches": patches,
        "history_patch_mask": history_patch_mask,
        "instrument_mask": instrument_mask,
        "slow_features": slow_features,
        "state_position": state_position,
    }


def build_tabular_batch(
    arrays: dict[str, np.ndarray],
    date_idx: np.ndarray,
    decision_idx: np.ndarray,
    equity_cutoffs: np.ndarray,
    context_cutoffs: np.ndarray,
    active_equities: np.ndarray,
    global_context: str,
    context_ablation: ResolvedContextAblation = NO_CONTEXT_ABLATION,
    feature_ablation: ResolvedFeatureAblation = NO_FEATURE_ABLATION,
) -> dict[str, np.ndarray]:
    """Construct the exact shared MLP/XGBoost representation."""
    batch_size = date_idx.size
    if global_context not in GLOBAL_CONTEXT_SETTINGS:
        raise ValueError(f"Invalid global context setting: {global_context}")
    if context_ablation.key != "none" and global_context != "enabled":
        raise ValueError("Context ablations require enabled global context")
    local_ready = np.asarray(
        arrays["context_data_ready.npy"][date_idx], dtype=bool
    ).copy()
    if context_ablation.local_slots:
        local_ready[:, context_ablation.local_slots] = False
    readiness = np.asarray(arrays["global_data_ready.npy"][date_idx], dtype=bool)
    global_ready = readiness[
        np.arange(batch_size)[:, None],
        np.arange(GLOBAL_CONTEXT_COUNT)[None, :],
        decision_idx[:, None],
    ].copy()
    if global_context == "masked":
        global_ready[:] = False
    elif context_ablation.global_slots:
        global_ready[:, context_ablation.global_slots] = False
    global_cutoffs = np.asarray(DECISION_GLOBAL_INDICES)[decision_idx]
    global_grid = np.asarray(arrays["global_features.npy"][date_idx], dtype=np.float32)
    output = np.zeros(
        (batch_size, EQUITY_COUNT, TABULAR_FEATURE_COUNT), dtype=np.float32
    )
    cursor = 0
    equity_slow = np.asarray(
        arrays["equity_slow.npy"][date_idx], dtype=np.float32
    ).copy()
    if context_ablation.equity_slow_indices:
        equity_slow[..., context_ablation.equity_slow_indices] = 0.0
    equity_slow = apply_feature_ablation_to_slow_features(equity_slow, feature_ablation)
    output[:, :, cursor : cursor + SLOW_FEATURE_COUNT] = (
        equity_slow * active_equities[..., None]
    )
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
            if minute < 0:
                continue
            values = np.asarray(
                arrays["equity_features.npy"][date_idx[group], :, minute, :],
                dtype=np.float32,
            )
            group_valid = (values[..., 5] > 0.5) & active_equities[group]
            block[group] = values * group_valid[..., None]
            valid[group] = group_valid
        output[:, :, cursor : cursor + block.shape[-1]] = block
        cursor += block.shape[-1]
        equity_validity.append(valid)

    context_validity: list[np.ndarray] = []
    for context_slot in range(LOCAL_CONTEXT_COUNT):
        for offset in TABULAR_OFFSETS:
            block = np.zeros(
                (batch_size, CONTEXT_GENERIC_DYNAMIC_COUNT), dtype=np.float32
            )
            valid = np.zeros(batch_size, dtype=bool)
            for cutoff in np.unique(context_cutoffs):
                group = np.flatnonzero(context_cutoffs == cutoff)
                minute = int(cutoff) - 1 - offset
                if minute < 0:
                    continue
                values = np.asarray(
                    arrays["context_features.npy"][
                        date_idx[group], context_slot, minute, :
                    ],
                    dtype=np.float32,
                )
                group_valid = (values[:, 5] > 0.5) & local_ready[group, context_slot]
                block[group] = (
                    values[:, :CONTEXT_GENERIC_DYNAMIC_COUNT] * group_valid[:, None]
                )
                valid[group] = group_valid
            output[:, :, cursor : cursor + CONTEXT_GENERIC_DYNAMIC_COUNT] = block[
                :, None, :
            ]
            cursor += CONTEXT_GENERIC_DYNAMIC_COUNT
            context_validity.append(valid)

    global_minutes = (
        global_cutoffs[:, None]
        - 1
        - np.asarray(TABULAR_OFFSETS, dtype=np.int64)[None, :]
    )
    global_values = global_grid[
        np.arange(batch_size)[:, None, None],
        np.arange(GLOBAL_CONTEXT_COUNT)[None, :, None],
        global_minutes[:, None, :],
    ]
    global_validity = (global_values[..., 5] > 0.5) & global_ready[..., None]
    global_dynamic = (
        global_values[..., :CONTEXT_GENERIC_DYNAMIC_COUNT] * global_validity[..., None]
    ).reshape(batch_size, -1)
    output[:, :, cursor : cursor + global_dynamic.shape[-1]] = global_dynamic[:, None]
    cursor += global_dynamic.shape[-1]
    context_slow = apply_feature_ablation_to_slow_features(
        np.asarray(arrays["context_slow.npy"][date_idx], dtype=np.float32)
        * local_ready[..., None],
        feature_ablation,
    ).reshape(batch_size, LOCAL_CONTEXT_COUNT * SLOW_FEATURE_COUNT)
    output[:, :, cursor : cursor + context_slow.shape[-1]] = context_slow[:, None]
    cursor += context_slow.shape[-1]
    global_slow = np.asarray(arrays["global_slow.npy"][date_idx], dtype=np.float32)
    decision_slow = global_slow[
        np.arange(batch_size)[:, None],
        np.arange(GLOBAL_CONTEXT_COUNT)[None, :],
        decision_idx[:, None],
    ]
    decision_slow = apply_feature_ablation_to_slow_features(
        decision_slow * global_ready[..., None], feature_ablation
    ).reshape(batch_size, -1)
    output[:, :, cursor : cursor + decision_slow.shape[-1]] = decision_slow[:, None]
    cursor += decision_slow.shape[-1]

    normalized_position = decision_idx.astype(np.float32) / (
        EXPECTED_DECISIONS_PER_DATE - 1
    )
    output[:, :, cursor] = np.sin(2.0 * np.pi * normalized_position)[:, None]
    output[:, :, cursor + 1] = np.cos(2.0 * np.pi * normalized_position)[:, None]
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
        raise RuntimeError(f"Tabular feature construction ended at width {cursor}")
    output *= active_equities[..., None]
    return {"tabular_features": output, "equity_mask": active_equities.copy()}


class VectorizedFeatureDataset(Dataset[dict[str, np.ndarray]]):
    def __init__(
        self,
        store: Path,
        sample_index: pl.DataFrame,
        model_name: str,
        global_context: str | None,
        tcn_architecture: TCNArchitecture | None = None,
        context_ablation: ResolvedContextAblation = NO_CONTEXT_ABLATION,
        feature_ablation: ResolvedFeatureAblation = NO_FEATURE_ABLATION,
        peer_features: str = "none",
    ) -> None:
        if model_name not in NEURAL_MODELS:
            raise ValueError(
                f"Vectorized dataset requires a neural model: {model_name}"
            )
        if model_name == "tcn":
            if tcn_architecture is None:
                raise ValueError("TCN architecture is required for model tcn")
            needs_context = tcn_architecture.fusion_mode in (
                "context_only",
                "context_pooled",
            )
        else:
            if tcn_architecture is not None:
                raise ValueError(
                    f"TCN architecture is forbidden for model {model_name}"
                )
            needs_context = model_name in ("context_only", "context_pooled", "mlp")
        if needs_context and global_context not in GLOBAL_CONTEXT_SETTINGS:
            raise ValueError("Context-consuming models require global context setting")
        if not needs_context and global_context is not None:
            raise ValueError("Context-free models do not accept global context setting")
        if not needs_context and context_ablation.key != "none":
            raise ValueError("Context-free models do not accept context ablations")
        if context_ablation.key != "none" and global_context != "enabled":
            raise ValueError("Context ablations require enabled global context")
        peer_features = validate_peer_feature_mode(model_name, peer_features)
        self.store = store
        self.model_name = model_name
        self.needs_context = needs_context
        self.global_context = global_context
        self.context_ablation = context_ablation
        self.feature_ablation = feature_ablation
        self.peer_features = peer_features
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
            filenames = (
                FEATURE_ARRAY_FILES
                if self.model_name == "mlp" or self.needs_context
                else tuple(
                    name
                    for name in FEATURE_ARRAY_FILES
                    if not name.startswith(("context_", "global_"))
                )
            )
            if self.peer_features != "none":
                filenames = (*filenames, *PEER_ARRAY_FILES)
            self._arrays = {
                filename: np.load(
                    self.store / filename, mmap_mode="r", allow_pickle=False
                )
                for filename in filenames
            }
        return self._arrays

    def __getitem__(self, request: BatchRequest) -> dict[str, np.ndarray]:
        arrays = self._open_arrays()
        positions = np.asarray(request.indices, dtype=np.int64)
        (
            common,
            active_equities,
            equity_cutoffs,
            context_cutoffs,
            source_date_idx,
            source_decision_idx,
        ) = _common_batch(arrays, self.rows, positions, request.valid_count)
        if self.model_name == "mlp":
            inputs = build_tabular_batch(
                arrays,
                source_date_idx,
                source_decision_idx,
                equity_cutoffs,
                context_cutoffs,
                active_equities,
                self.global_context,
                self.context_ablation,
                self.feature_ablation,
            )
            inputs["tabular_features"][~common["sample_valid_mask"]] = 0.0
            inputs["equity_mask"][~common["sample_valid_mask"]] = False
        else:
            inputs = _build_patch_batch(
                arrays,
                source_date_idx,
                equity_cutoffs,
                source_decision_idx,
                context_cutoffs,
                active_equities,
                self.global_context,
                self.context_ablation,
                self.feature_ablation,
            )
            if self.peer_features != "none":
                inputs["peer_state"] = _build_peer_state(
                    arrays,
                    source_date_idx,
                    equity_cutoffs,
                    active_equities,
                    self.peer_features,
                )
                inputs["peer_state"][~common["sample_valid_mask"]] = 0.0
        return {**inputs, **common}


class TabularRowIterator:
    """Re-iterable compact valid-row source for one horizon booster."""

    def __init__(
        self,
        store: Path,
        sample_index: pl.DataFrame,
        horizon_index: int,
        *,
        device: str,
        global_context: str,
        batch_size: int = 64,
        context_ablation: ResolvedContextAblation = NO_CONTEXT_ABLATION,
        feature_ablation: ResolvedFeatureAblation = NO_FEATURE_ABLATION,
    ) -> None:
        if horizon_index not in range(HORIZON_COUNT):
            raise ValueError("horizon_index is outside the three-horizon contract")
        if device not in ("cpu", "cuda"):
            raise ValueError("TabularRowIterator device must be 'cpu' or 'cuda'")
        self.dataset = VectorizedFeatureDataset(
            store,
            sample_index,
            "mlp",
            global_context,
            context_ablation=context_ablation,
            feature_ablation=feature_ablation,
        )
        self.horizon_index = horizon_index
        self.device = device
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[TabularRowBatch]:
        equity_slots = np.arange(EQUITY_COUNT, dtype=np.int16)
        for start in range(0, len(self.dataset), self.batch_size):
            stop = min(start + self.batch_size, len(self.dataset))
            request = BatchRequest(tuple(range(start, stop)), stop - start)
            batch = self.dataset[request]
            valid = batch["label_mask"][:, :, self.horizon_index]
            counts = valid.sum(axis=1)
            row_sample, row_equity = np.nonzero(valid)
            if row_sample.size == 0:
                continue
            features = batch["tabular_features"][row_sample, row_equity]
            labels = batch["targets"][row_sample, row_equity, self.horizon_index]
            weights = (1.0 / counts[row_sample]).astype(np.float32)
            if self.device == "cuda":
                features = torch.from_numpy(features).to("cuda")
                labels = torch.from_numpy(labels).to("cuda")
                weights = torch.from_numpy(weights).to("cuda")
            yield TabularRowBatch(
                features=features,
                labels=labels,
                weights=weights,
                sample_id=batch["sample_id"][row_sample],
                date_idx=batch["date_idx"][row_sample],
                decision_idx=batch["decision_idx"][row_sample],
                equity_slot=equity_slots[row_equity],
            )


class DateStratifiedMicrobatchSampler(Sampler[BatchRequest]):
    def __init__(
        self, sample_index: pl.DataFrame, runtime: RuntimeSettings, seed: int
    ) -> None:
        self.runtime = runtime
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
            raise ValueError(
                f"Date-stratified sampling requires {EFFECTIVE_BATCH_SIZE} distinct dates"
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
            selected_dates = generator.choice(
                len(self.dates), size=EFFECTIVE_BATCH_SIZE, replace=False
            )
            effective_indices = []
            for date_position in selected_dates:
                trade_date = self.dates[int(date_position)]
                choices = self.positions_by_date[trade_date]
                effective_indices.append(int(choices[generator.integers(len(choices))]))
            for start in range(0, EFFECTIVE_BATCH_SIZE, self.runtime.microbatch_size):
                indices = tuple(
                    effective_indices[start : start + self.runtime.microbatch_size]
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
    runtime: RuntimeSettings,
    seed: int,
) -> DataLoader[dict[str, torch.Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=None,
        sampler=sampler,
        num_workers=runtime.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=runtime.prefetch_factor,
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
    model_name: str,
    global_context: str | None,
    runtime: RuntimeSettings,
    seed: int,
    tcn_architecture: TCNArchitecture | None = None,
    context_ablation: ResolvedContextAblation = NO_CONTEXT_ABLATION,
    feature_ablation: ResolvedFeatureAblation = NO_FEATURE_ABLATION,
    peer_features: str = "none",
) -> tuple[
    DataLoader[dict[str, torch.Tensor]],
    DataLoader[dict[str, torch.Tensor]],
    DateStratifiedMicrobatchSampler,
]:
    sampler = DateStratifiedMicrobatchSampler(train_rows, runtime, seed)
    train_loader = _create_loader(
        VectorizedFeatureDataset(
            store,
            train_rows,
            model_name,
            global_context,
            tcn_architecture,
            context_ablation,
            feature_ablation,
            peer_features,
        ),
        sampler,
        runtime,
        seed,
    )
    validation_loader = _create_loader(
        VectorizedFeatureDataset(
            store,
            validation_rows,
            model_name,
            global_context,
            tcn_architecture,
            context_ablation,
            feature_ablation,
            peer_features,
        ),
        SequentialPaddedBatchSampler(
            validation_rows.height, runtime.evaluation_batch_size
        ),
        runtime,
        seed,
    )
    return train_loader, validation_loader, sampler


def create_evaluation_loader(
    store: Path,
    rows: pl.DataFrame,
    model_name: str,
    global_context: str | None,
    runtime: RuntimeSettings,
    seed: int,
    tcn_architecture: TCNArchitecture | None = None,
    context_ablation: ResolvedContextAblation = NO_CONTEXT_ABLATION,
    feature_ablation: ResolvedFeatureAblation = NO_FEATURE_ABLATION,
    peer_features: str = "none",
) -> DataLoader[dict[str, torch.Tensor]]:
    return _create_loader(
        VectorizedFeatureDataset(
            store,
            rows,
            model_name,
            global_context,
            tcn_architecture,
            context_ablation,
            feature_ablation,
            peer_features,
        ),
        SequentialPaddedBatchSampler(rows.height, runtime.evaluation_batch_size),
        runtime,
        seed,
    )
