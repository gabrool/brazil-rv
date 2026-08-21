from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
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
EXTERNAL_SIDECAR_SCHEMA = "PIT_EXTERNAL_FEATURE_SIDECAR"
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


@dataclass(frozen=True)
class ExternalFeatureSidecar:
    path: Path
    cadence: str
    feature_names: tuple[str, ...]
    identity: dict[str, object]

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def input_width(self) -> int:
        return 2 * self.feature_count


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _axis_sha256(indices: np.ndarray, labels: list[str]) -> str:
    digest = hashlib.sha256()
    for index, label in zip(indices.tolist(), labels, strict=True):
        digest.update(str(int(index)).encode("ascii"))
        digest.update(b"\0")
        digest.update(label.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def feature_store_axis_identity(store: Path) -> dict[str, object]:
    dates = (
        pl.read_parquet(store / "date_index.parquet")
        .select("date_idx", "trade_date")
        .sort("date_idx")
    )
    equities = (
        pl.read_parquet(store / "equity_index.parquet")
        .select("equity_slot", "security_id")
        .sort("equity_slot")
    )
    date_indices = dates.get_column("date_idx").cast(pl.Int64).to_numpy()
    equity_slots = equities.get_column("equity_slot").cast(pl.Int64).to_numpy()
    if not np.array_equal(date_indices, np.arange(dates.height)):
        raise ValueError("Canonical date axis must be contiguous and sorted")
    if equities.height != EQUITY_COUNT or not np.array_equal(
        equity_slots, np.arange(EQUITY_COUNT)
    ):
        raise ValueError(
            f"Canonical equity axis must contain slots 0..{EQUITY_COUNT - 1}"
        )
    security_ids = equities.get_column("security_id").cast(pl.String).to_list()
    if len(set(security_ids)) != EQUITY_COUNT:
        raise ValueError("Canonical equity axis security_id values must be unique")
    return {
        "date_count": dates.height,
        "date_axis_sha256": _axis_sha256(
            date_indices,
            dates.get_column("trade_date").cast(pl.String).to_list(),
        ),
        "equity_count": equities.height,
        "equity_axis_sha256": _axis_sha256(equity_slots, security_ids),
        "decision_count": EXPECTED_DECISIONS_PER_DATE,
    }


@lru_cache(maxsize=16)
def load_external_sidecar(sidecar_dir: Path, store: Path) -> ExternalFeatureSidecar:
    path = sidecar_dir.resolve()
    manifest_path = path / "manifest.json"
    values_path = path / "values.npy"
    mask_path = path / "mask.npy"
    if not path.is_dir():
        raise FileNotFoundError(path)
    for required in (manifest_path, values_path, mask_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != EXTERNAL_SIDECAR_SCHEMA:
        raise ValueError("External sidecar manifest has an unknown schema")
    cadence = manifest.get("cadence")
    if cadence not in ("daily", "intraday"):
        raise ValueError("External sidecar cadence must be daily or intraday")
    feature_names_value = manifest.get("feature_names")
    if (
        not isinstance(feature_names_value, list)
        or not feature_names_value
        or any(not isinstance(name, str) or not name for name in feature_names_value)
    ):
        raise ValueError("External sidecar feature_names must be nonempty strings")
    feature_names = tuple(feature_names_value)
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("External sidecar feature_names must be unique")
    store_identity = feature_store_identity(store)
    if manifest.get("feature_store_identity") != store_identity:
        raise ValueError("External sidecar feature store identity is misaligned")
    axes = feature_store_axis_identity(store)
    if manifest.get("axes") != axes:
        raise ValueError("External sidecar date/equity axes are misaligned")
    shape = [int(axes["date_count"]), EQUITY_COUNT]
    if cadence == "intraday":
        shape.append(EXPECTED_DECISIONS_PER_DATE)
    shape.append(len(feature_names))
    arrays_value = manifest.get("arrays")
    if not isinstance(arrays_value, dict):
        raise ValueError("External sidecar arrays metadata is missing")
    normalized_arrays: dict[str, dict[str, object]] = {}
    arrays: dict[str, np.ndarray] = {}
    for name, array_path, dtype in (
        ("values.npy", values_path, "float32"),
        ("mask.npy", mask_path, "bool"),
    ):
        metadata = arrays_value.get(name)
        if not isinstance(metadata, dict):
            raise ValueError(f"External sidecar metadata is missing for {name}")
        actual_sha256 = _file_sha256(array_path)
        expected = {
            "shape": shape,
            "dtype": dtype,
            "sha256": actual_sha256,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(f"External sidecar {name} metadata differs from its file")
        array = np.load(array_path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != shape or array.dtype.name != dtype:
            raise ValueError(f"External sidecar {name} has the wrong array contract")
        arrays[name] = array
        normalized_arrays[name] = expected
    values = arrays["values.npy"]
    mask = arrays["mask.npy"]
    for date_idx in range(int(axes["date_count"])):
        date_values = np.asarray(values[date_idx])
        date_mask = np.asarray(mask[date_idx])
        if np.any(date_values[~date_mask] != 0):
            raise ValueError("External sidecar invalid values must be exactly zero")
        if not np.isfinite(date_values[date_mask]).all():
            raise ValueError("External sidecar valid values must be finite")
    identity = {
        "path": str(path),
        "schema": EXTERNAL_SIDECAR_SCHEMA,
        "manifest_sha256": _file_sha256(manifest_path),
        "cadence": cadence,
        "feature_names": list(feature_names),
        "feature_count": len(feature_names),
        "feature_store_identity": store_identity,
        "axes": axes,
        "arrays": normalized_arrays,
    }
    return ExternalFeatureSidecar(path, cadence, feature_names, identity)


def load_recorded_external_sidecar(
    recorded_identity: object,
    store: Path,
) -> ExternalFeatureSidecar | None:
    if recorded_identity is None:
        return None
    if not isinstance(recorded_identity, dict) or not isinstance(
        recorded_identity.get("path"), str
    ):
        raise ValueError("Recorded external sidecar identity is malformed")
    sidecar = load_external_sidecar(Path(recorded_identity["path"]), store)
    if sidecar.identity != recorded_identity:
        raise ValueError("Recorded external sidecar differs from its files")
    return sidecar


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
) -> tuple[np.ndarray, np.ndarray]:
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
    return local_ready, global_ready


def _build_patch_batch(
    arrays: dict[str, np.ndarray],
    date_idx: np.ndarray,
    equity_cutoffs: np.ndarray,
    decision_idx: np.ndarray,
    context_cutoffs: np.ndarray,
    active: np.ndarray,
) -> dict[str, np.ndarray]:
    batch_size = date_idx.size
    local_ready, global_ready = _context_readiness(arrays, date_idx, decision_idx)
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
    return {
        "patches": patches,
        "history_patch_mask": history_mask,
        "instrument_mask": instrument_mask,
        "slow_features": slow,
        "state_position": state_position,
    }


def _build_sidecar_batch(
    arrays: dict[str, np.ndarray],
    cadence: str,
    date_idx: np.ndarray,
    decision_idx: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    values_array = arrays["values.npy"]
    mask_array = arrays["mask.npy"]
    if cadence == "daily":
        values = np.asarray(values_array[date_idx], dtype=np.float32).copy()
        mask = np.asarray(mask_array[date_idx], dtype=bool).copy()
    else:
        feature_count = values_array.shape[-1]
        values = np.zeros(
            (date_idx.size, EQUITY_COUNT, feature_count), dtype=np.float32
        )
        mask = np.zeros_like(values, dtype=bool)
        for decision in np.unique(decision_idx):
            group = np.flatnonzero(decision_idx == decision)
            values[group] = values_array[date_idx[group], :, int(decision), :]
            mask[group] = mask_array[date_idx[group], :, int(decision), :]
    mask &= active[..., None]
    values *= mask
    return np.concatenate((values, mask.astype(np.float32)), axis=-1)


class VectorizedFeatureDataset(Dataset[dict[str, np.ndarray]]):
    def __init__(
        self,
        store: Path,
        sample_index: pl.DataFrame,
        sidecar: ExternalFeatureSidecar | None = None,
    ) -> None:
        self.store = store
        self.sidecar = sidecar
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
        self._sidecar_arrays: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.rows["sample_id"])

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_arrays"] = None
        state["_sidecar_arrays"] = None
        return state

    def _open_arrays(self) -> dict[str, np.ndarray]:
        if self._arrays is None:
            self._arrays = {
                name: np.load(self.store / name, mmap_mode="r", allow_pickle=False)
                for name in FEATURE_ARRAY_FILES
            }
        return self._arrays

    def _open_sidecar_arrays(self) -> dict[str, np.ndarray]:
        if self.sidecar is None:
            raise RuntimeError("Dataset has no external sidecar")
        if self._sidecar_arrays is None:
            self._sidecar_arrays = {
                name: np.load(
                    self.sidecar.path / name, mmap_mode="r", allow_pickle=False
                )
                for name in ("values.npy", "mask.npy")
            }
        return self._sidecar_arrays

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
        if self.sidecar is not None:
            inputs["sidecar_features"] = _build_sidecar_batch(
                self._open_sidecar_arrays(),
                self.sidecar.cadence,
                dates,
                decisions,
                active,
            )
        return {**inputs, **common}


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
    sidecar: ExternalFeatureSidecar | None = None,
) -> tuple[
    DataLoader[dict[str, torch.Tensor]],
    DataLoader[dict[str, torch.Tensor]],
    DateStratifiedBatchSampler,
]:
    sampler = DateStratifiedBatchSampler(train_rows, runtime, seed)
    train = _create_loader(
        VectorizedFeatureDataset(store, train_rows, sidecar), sampler, runtime, seed
    )
    validation = _create_loader(
        VectorizedFeatureDataset(store, validation_rows, sidecar),
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
    sidecar: ExternalFeatureSidecar | None = None,
) -> DataLoader[dict[str, torch.Tensor]]:
    return _create_loader(
        VectorizedFeatureDataset(store, rows, sidecar),
        DecisionGroupedBatchSampler(rows, runtime.evaluation_batch_size),
        runtime,
        seed,
    )
