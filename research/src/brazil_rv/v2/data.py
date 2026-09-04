from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset

from .contract import (
    ALLOWED_LOOKBACKS,
    DECISION_MINUTE_INDEX,
    FAST_PATCH_MINUTES,
    FINETUNE_START,
    HORIZONS,
    PRETRAIN_END,
    STORE_START,
    V1_STORE_V2_ZERO_SLOW_FIELDS,
)
from .data_roots import ExternalFileResolution, resolve_external_files
from .store import V2Store, open_store_for_samples
from .splits import AccessPurpose, PREREGISTRATION_ROOT, authorize_dates

Stage = Literal["pretrain", "finetune", "evaluation", "joint"]
V1_STORE_V2_ZERO_DYNAMIC_CHANNELS = (9, 11, 14, 22, 24, 25)


def _validate_stage_dates(
    selected_dates: NDArray[np.datetime64], stage: Stage
) -> None:
    dates = np.asarray(selected_dates, dtype="datetime64[D]")
    pretrain = (dates >= np.datetime64(STORE_START)) & (
        dates <= np.datetime64(PRETRAIN_END)
    )
    fine_or_evaluation = dates >= np.datetime64(FINETUNE_START)
    if stage == "pretrain":
        valid = pretrain
    elif stage in {"finetune", "evaluation"}:
        valid = fine_or_evaluation
    elif stage == "joint":
        valid = pretrain | fine_or_evaluation
    else:
        raise ValueError(f"unknown v2 stage: {stage}")
    if not valid.all():
        invalid = str(dates[np.flatnonzero(~valid)[0]])
        raise ValueError(f"{stage} stage cannot consume session {invalid}")


def pack_fast_patches(
    minute_values: NDArray[np.floating],
    minute_valid: NDArray[np.bool_],
    *,
    cutoff: int = DECISION_MINUTE_INDEX,
    patch_minutes: int = FAST_PATCH_MINUTES,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Pack only completed pre-entry minutes into non-overlapping patches."""

    values = np.asarray(minute_values, dtype=np.float32)
    valid = np.asarray(minute_valid, dtype=np.bool_)
    if values.ndim < 3 or values.shape[:-1] != valid.shape:
        raise ValueError("minute values must end in fields and align with minute_valid")
    if cutoff > values.shape[-2] or cutoff <= 0 or cutoff % patch_minutes:
        raise ValueError("fast cutoff must form complete patches")
    prefix = values[..., :cutoff, :]
    prefix_valid = valid[..., :cutoff]
    patch_count = cutoff // patch_minutes
    patches = prefix.reshape(
        *prefix.shape[:-2], patch_count, patch_minutes * prefix.shape[-1]
    )
    patch_valid = prefix_valid.reshape(
        *prefix_valid.shape[:-1], patch_count, patch_minutes
    ).all(axis=-1)
    patches = np.where(patch_valid[..., None], patches, 0.0).astype(np.float32)
    return patches, patch_valid


def slow_row_index(
    sample_date_index: int, stage: Stage, *, joint_pretrain: bool | None = None
) -> int:
    if sample_date_index < 0:
        raise ValueError("sample date index must be non-negative")
    if stage == "pretrain" or (stage == "joint" and bool(joint_pretrain)):
        return sample_date_index
    if stage in {"finetune", "evaluation"} or (
        stage == "joint" and joint_pretrain is not None and not bool(joint_pretrain)
    ):
        return sample_date_index - 1
    if stage == "joint":
        raise ValueError("joint slow alignment requires the sample window")
    raise ValueError(f"unknown v2 stage: {stage}")


def required_store_date_indices(
    date_indices: Sequence[int],
    dates: NDArray[np.datetime64],
    *,
    stage: Stage,
    lookback: int,
) -> NDArray[np.int64]:
    """Return the finite union of sample rows and their causal slow histories."""

    indices = np.asarray(date_indices, dtype=np.int64)
    date_axis = np.asarray(dates, dtype="datetime64[D]")
    if lookback not in ALLOWED_LOOKBACKS:
        raise ValueError("lookback must be 20, 60, or 120")
    if (
        indices.ndim != 1
        or not indices.size
        or np.any(indices < 0)
        or np.any(indices >= date_axis.size)
    ):
        raise ValueError("date indices are outside the store")
    required = set(int(value) for value in indices)
    for date_index in indices:
        index = int(date_index)
        joint_pretrain = bool(
            stage == "joint"
            and date_axis[index] <= np.datetime64(PRETRAIN_END)
        )
        end = slow_row_index(
            index,
            stage,
            joint_pretrain=joint_pretrain if stage == "joint" else None,
        )
        if end >= 0:
            required.update(range(max(0, end - lookback + 1), end + 1))
    return np.asarray(sorted(required), dtype=np.int64)


def causal_history_end_offsets(
    date_indices: Sequence[int],
    dates: NDArray[np.datetime64],
    *,
    stage: Stage,
) -> NDArray[np.int64]:
    """Return each sample's frozen slow-history endpoint relative to t."""

    indices = np.asarray(date_indices, dtype=np.int64)
    date_axis = np.asarray(dates, dtype="datetime64[D]")
    offsets = np.empty(len(indices), dtype=np.int64)
    for row, date_index in enumerate(indices):
        index = int(date_index)
        joint_pretrain = bool(
            stage == "joint"
            and date_axis[index] <= np.datetime64(PRETRAIN_END)
        )
        offsets[row] = (
            slow_row_index(
                index,
                stage,
                joint_pretrain=joint_pretrain if stage == "joint" else None,
            )
            - index
        )
    return offsets


def lazy_slow_window(
    values: NDArray[np.floating],
    valid: NDArray[np.bool_],
    *,
    end_index: int,
    lookback: int,
) -> tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.bool_]]:
    """Create a left-padded `[name, lookback, feature]` view on demand."""

    source = np.asarray(values)
    mask = np.asarray(valid, dtype=np.bool_)
    if source.ndim != 3 or source.shape != mask.shape:
        raise ValueError("slow values and validity must align [date, name, feature]")
    if lookback not in ALLOWED_LOOKBACKS:
        raise ValueError("lookback must be 20, 60, or 120")
    output = np.zeros((source.shape[1], lookback, source.shape[2]), dtype=np.float32)
    output_valid = np.zeros(output.shape, dtype=np.bool_)
    history_mask = np.zeros((source.shape[1], lookback), dtype=np.bool_)
    if end_index < 0:
        return output, output_valid, history_mask
    if end_index >= source.shape[0]:
        raise IndexError("slow end index exceeds store")
    start = max(0, end_index - lookback + 1)
    sample = np.asarray(source[start : end_index + 1], dtype=np.float32).transpose(1, 0, 2)
    sample_valid = mask[start : end_index + 1].transpose(1, 0, 2)
    offset = lookback - sample.shape[1]
    output[:, offset:] = np.where(sample_valid, sample, 0.0)
    output_valid[:, offset:] = sample_valid
    history_mask[:, offset:] = sample_valid.any(axis=-1)
    return output, output_valid, history_mask


def _zero_invalid_values(
    values: NDArray[np.generic],
    valid: NDArray[np.bool_],
    *,
    name: str,
) -> NDArray[np.generic]:
    """Zero unavailable cells and reject non-finite values marked available."""

    array = np.asarray(values)
    mask = np.asarray(valid, dtype=np.bool_)
    while mask.ndim < array.ndim:
        mask = mask[..., None]
    try:
        aligned = np.broadcast_to(mask, array.shape)
    except ValueError as error:
        raise ValueError(f"{name} and its validity mask are misaligned") from error
    clean = np.where(aligned, array, 0).astype(array.dtype, copy=False)
    if np.issubdtype(clean.dtype, np.floating) and not np.isfinite(clean).all():
        raise ValueError(f"{name} contains non-finite values marked available")
    return clean


class V2DailyDataset(Dataset[dict[str, object]]):
    """One full cross-section per session, with lazy stage-correct slow history."""

    def __init__(
        self,
        store: V2Store | str | Path,
        date_indices: Sequence[int],
        *,
        stage: Stage,
        lookback: int = 60,
        enabled_sidecars: Sequence[str] = (),
        fast_store: str | Path | None = None,
        verify_fast_hashes: bool = True,
        purpose: AccessPurpose | None = None,
        registration_path: Path | None = None,
        preregistration_root: Path = PREREGISTRATION_ROOT,
    ) -> None:
        self.date_indices = np.asarray(date_indices, dtype=np.int64)
        self.stage = stage
        self.lookback = lookback
        self.enabled_sidecars = tuple(enabled_sidecars)
        self._sample_date_indices = frozenset(
            int(value) for value in self.date_indices
        )
        access_purpose: AccessPurpose = purpose or (
            "evaluation" if stage == "evaluation" else "training"
        )
        if self.date_indices.ndim != 1 or not self.date_indices.size:
            raise ValueError("date indices must be a nonempty vector")
        if lookback not in ALLOWED_LOOKBACKS:
            raise ValueError("lookback must be 20, 60, or 120")
        if np.unique(self.date_indices).size != self.date_indices.size:
            raise ValueError("v2 sample builder emits exactly one row per session")
        if isinstance(store, V2Store):
            self.store = store
            if np.any(
                (self.date_indices < 0)
                | (self.date_indices >= self.store.dates.size)
            ):
                raise ValueError("date indices are outside the store")
            required_indices = required_store_date_indices(
                self.date_indices,
                self.store.dates,
                stage=stage,
                lookback=lookback,
            )
            if not self.store.authorized_for(required_indices):
                raise PermissionError(
                    "the open store was not authorized for every sample and "
                    "causal-history date"
                )
            selected_dates = self.store.dates[self.date_indices]
            _validate_stage_dates(selected_dates, stage)
            requested = tuple(
                sorted(selected_dates.astype(object).tolist())
            )
            token_path = registration_path
            if (
                token_path is None
                and self.store.access_ledger is not None
                and self.store.access_ledger.registration is not None
            ):
                token_path = self.store.access_ledger.registration.path
            access_ledger = authorize_dates(
                requested,
                purpose=access_purpose,
                registration_path=token_path,
                preregistration_root=preregistration_root,
            )
        else:
            store_path = Path(store)
            date_axis = np.load(
                store_path / "date_index.npy", allow_pickle=False
            ).astype("datetime64[D]", copy=False)
            if np.any(
                (self.date_indices < 0) | (self.date_indices >= date_axis.size)
            ):
                raise ValueError("date indices are outside the store")
            _validate_stage_dates(date_axis[self.date_indices], stage)
            history_offsets = causal_history_end_offsets(
                self.date_indices,
                date_axis,
                stage=stage,
            )
            self.store, access_ledger = open_store_for_samples(
                store_path,
                self.date_indices,
                purpose=access_purpose,
                history_lookbacks=lookback,
                history_end_offsets=history_offsets,
                registration_path=registration_path,
                preregistration_root=preregistration_root,
            )
        self.access_ledger = access_ledger
        self._external_fast_features: NDArray[np.generic] | None = None
        self._external_fast_ready: NDArray[np.generic] | None = None
        self._external_fast_slow: NDArray[np.generic] | None = None
        self.external_artifact_resolutions: tuple[ExternalFileResolution, ...] = ()
        self._fast_date_mapping = np.full(self.store.dates.size, -1, dtype=np.int64)
        self._fast_v2_slots = np.empty(0, dtype=np.int64)
        self._fast_v1_slots = np.empty(0, dtype=np.int64)
        if np.any(
            (self.date_indices < 0) | (self.date_indices >= self.store.dates.size)
        ):
            raise ValueError("date indices are outside the store")
        if stage in {"finetune", "evaluation"} and np.any(self.date_indices == 0):
            raise ValueError("fine/evaluation samples need a prior slow session")
        self.store.array_shape("slow_values")
        self.store.array_shape("slow_valid")
        self.store.array_shape("active")
        for group in self.enabled_sidecars:
            self.store.array_shape(f"sidecar_{group}_values")
            self.store.array_shape(f"sidecar_{group}_valid")
        configured_fast = fast_store or self.store.manifest.get("metadata", {}).get(
            "v1_fast_store"
        )
        if configured_fast:
            self._open_external_fast(
                None if fast_store is None else Path(configured_fast),
                verify_fast_hashes,
            )

    def _open_external_fast(self, root: Path | None, verify_hashes: bool) -> None:
        if not verify_hashes:
            raise ValueError("external v1 fast artifacts must be hash-verified")
        records = self.store.manifest.get("metadata", {}).get("v1_fast_files", [])
        if not isinstance(records, list) or any(
            not isinstance(record, Mapping) for record in records
        ):
            raise ValueError("external v1 fast artifact records are malformed")
        required = (
            "equity_features.npy",
            "equity_slow.npy",
            "equity_data_ready.npy",
        )
        try:
            if root is None:
                paths, resolutions = resolve_external_files(records, required)
            else:
                resolved_root = root.resolve(strict=True)
                paths, resolutions = resolve_external_files(
                    records, required, local_root=resolved_root
                )
        except ValueError as error:
            if "mismatch" in str(error).casefold():
                raise ValueError("external v1 fast hash mismatch") from error
            raise
        self.external_artifact_resolutions = resolutions
        features = np.load(paths[required[0]], mmap_mode="r", allow_pickle=False)
        slow = np.load(paths[required[1]], mmap_mode="r", allow_pickle=False)
        ready = np.load(paths[required[2]], mmap_mode="r", allow_pickle=False)
        if (
            features.ndim != 4
            or features.shape[2] < DECISION_MINUTE_INDEX
            or features.shape[3] != 26
            or slow.shape != (*features.shape[:2], 32)
            or slow.dtype != np.float32
            or ready.shape != features.shape[:2]
            or ready.dtype != np.bool_
        ):
            raise ValueError("external v1 fast arrays have the wrong contract")
        date_mapping = self.store.read_table(
            "v1_fast_date_mapping", self.date_indices
        )
        isin_mapping = self.store.read_table("v1_fast_isin_mapping")
        for target, source in date_mapping.select(
            "v2_date_index", "v1_date_index"
        ).iter_rows():
            if not 0 <= target < self.store.dates.size or not 0 <= source < features.shape[0]:
                raise ValueError("external v1 date mapping is outside its axes")
            if self._fast_date_mapping[target] >= 0:
                raise ValueError("external v1 date mapping is not one-to-one")
            self._fast_date_mapping[target] = source
        self._fast_v2_slots = isin_mapping.get_column(
            "v2_isin_index"
        ).to_numpy().astype(np.int64)
        self._fast_v1_slots = isin_mapping.get_column(
            "v1_equity_slot"
        ).to_numpy().astype(np.int64)
        if (
            len(set(self._fast_v2_slots.tolist())) != self._fast_v2_slots.size
            or len(set(self._fast_v1_slots.tolist())) != self._fast_v1_slots.size
            or np.any(self._fast_v2_slots < 0)
            or np.any(self._fast_v1_slots < 0)
            or np.any(self._fast_v2_slots >= len(self.store.isins))
            or np.any(self._fast_v1_slots >= features.shape[1])
        ):
            raise ValueError("external v1 ISIN mapping is not one-to-one")
        self._external_fast_features = features
        self._external_fast_slow = slow
        self._external_fast_ready = ready

    def __len__(self) -> int:
        return int(self.date_indices.size)

    def _sample_is_pretrain(self, date_index: int) -> bool:
        return self.stage == "pretrain" or (
            self.stage == "joint"
            and self.store.dates[date_index] <= np.datetime64(PRETRAIN_END)
        )

    def _slow_window(
        self, end_index: int
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.bool_]]:
        """Read and concatenate only the requested windows from mmap sources."""

        windows: list[NDArray[np.float32]] = []
        masks: list[NDArray[np.bool_]] = []
        history_masks: list[NDArray[np.bool_]] = []
        sources = [("slow_values", "slow_valid")]
        sources.extend(
            (f"sidecar_{group}_values", f"sidecar_{group}_valid")
            for group in self.enabled_sidecars
        )
        for value_name, valid_name in sources:
            start = max(0, end_index - self.lookback + 1)
            selector = slice(start, end_index + 1)
            values = self.store.read(value_name, selector)
            valid = self.store.read(valid_name, selector)
            window, mask, history = lazy_slow_window(
                values,
                valid,
                end_index=len(values) - 1,
                lookback=self.lookback,
            )
            windows.append(window)
            masks.append(mask)
            history_masks.append(history)
        return (
            np.concatenate(windows, axis=2),
            np.concatenate(masks, axis=2),
            np.logical_or.reduce(history_masks),
        )

    def _fast(self, date_index: int) -> tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.bool_]]:
        name_count = len(self.store.isins)
        if self._sample_is_pretrain(date_index):
            width = int(self.store.manifest.get("metadata", {}).get("fast_minute_feature_count", 26))
            patches = np.zeros(
                (name_count, DECISION_MINUTE_INDEX // FAST_PATCH_MINUTES, width * FAST_PATCH_MINUTES),
                dtype=np.float32,
            )
            patch_mask = np.zeros(patches.shape[:2], dtype=np.bool_)
            return patches, patch_mask, np.zeros(name_count, dtype=np.bool_)
        if self.store.has_array("fast_minute_values"):
            minute_values = self.store.read("fast_minute_values", date_index)
            minute_valid = self.store.read("fast_minute_valid", date_index)
            patches, patch_mask = pack_fast_patches(minute_values, minute_valid)
            present = patch_mask.any(axis=1)
            if self.store.has_array("fast_present"):
                present &= np.asarray(
                    self.store.read("fast_present", date_index), dtype=np.bool_
                )
            return patches, patch_mask, present
        source_date = int(self._fast_date_mapping[date_index])
        if self._external_fast_features is None or source_date < 0:
            patches = np.zeros(
                (
                    name_count,
                    DECISION_MINUTE_INDEX // FAST_PATCH_MINUTES,
                    26 * FAST_PATCH_MINUTES,
                ),
                dtype=np.float32,
            )
            return (
                patches,
                np.zeros(patches.shape[:2], dtype=np.bool_),
                np.zeros(name_count, dtype=np.bool_),
            )
        source = np.asarray(
            self._external_fast_features[
                source_date, self._fast_v1_slots, :DECISION_MINUTE_INDEX
            ],
            dtype=np.float32,
        ).copy()
        source[..., V1_STORE_V2_ZERO_DYNAMIC_CHANNELS] = 0.0
        assert self._external_fast_ready is not None
        source_ready = np.asarray(
            self._external_fast_ready[source_date, self._fast_v1_slots],
            dtype=np.bool_,
        )
        if self.store.has_array("fast_present"):
            stored_present = np.asarray(
                self.store.read("fast_present", date_index)[self._fast_v2_slots],
                dtype=np.bool_,
            )
            source_ready &= stored_present
        minute_valid = np.broadcast_to(
            source_ready[:, None], source.shape[:2]
        )
        source_patches, source_patch_mask = pack_fast_patches(source, minute_valid)
        patches = np.zeros((name_count, *source_patches.shape[1:]), dtype=np.float32)
        patch_mask = np.zeros((name_count, source_patch_mask.shape[1]), dtype=np.bool_)
        present = np.zeros(name_count, dtype=np.bool_)
        patches[self._fast_v2_slots] = source_patches
        patch_mask[self._fast_v2_slots] = source_patch_mask
        present[self._fast_v2_slots] = source_ready
        return patches, patch_mask, present

    def _v1_equity_slow(
        self,
        date_index: int,
        fast_present: NDArray[np.bool_],
    ) -> NDArray[np.float32]:
        """Map the exact masked v1 slow row onto the sparse v2 ISIN axis."""

        output = np.zeros((len(self.store.isins), 32), dtype=np.float32)
        if self._sample_is_pretrain(date_index) or self._external_fast_slow is None:
            return output
        source_date = int(self._fast_date_mapping[date_index])
        if source_date < 0:
            return output
        mapped_present = np.asarray(
            fast_present[self._fast_v2_slots], dtype=np.bool_
        )
        if not mapped_present.any():
            return output
        source = np.asarray(
            self._external_fast_slow[source_date, self._fast_v1_slots],
            dtype=np.float32,
        ).copy()
        source[..., V1_STORE_V2_ZERO_SLOW_FIELDS] = 0.0
        if not np.isfinite(source[mapped_present]).all():
            raise ValueError("present external v1 slow rows must be finite")
        source[~mapped_present] = 0.0
        output[self._fast_v2_slots] = source
        return output

    def __getitem__(self, item: int) -> dict[str, object]:
        date_index = int(self.date_indices[item])
        is_pretrain = self._sample_is_pretrain(date_index)
        slow_end = slow_row_index(
            date_index,
            self.stage,
            joint_pretrain=is_pretrain if self.stage == "joint" else None,
        )
        history, feature_mask, history_mask = self._slow_window(slow_end)
        patches, patch_mask, fast_present = self._fast(date_index)
        v1_equity_slow = self._v1_equity_slow(date_index, fast_present)
        active = np.asarray(
            self.store.read("active", date_index), dtype=np.bool_
        )
        sample: dict[str, object] = {
            "date_index": np.int64(date_index),
            "trade_date": str(self.store.dates[date_index]),
            "slow_features": history,
            "slow_feature_mask": feature_mask,
            "slow_history_mask": history_mask,
            "fast_patches": patches,
            "fast_patch_mask": patch_mask,
            "fast_present": fast_present,
            "v1_equity_slow": v1_equity_slow,
            "days_since_last_slow_row": np.full(
                len(self.store.isins),
                0.0 if is_pretrain else 1.0,
                dtype=np.float32,
            ),
            "active_mask": active,
        }
        for source, destination in (
            ("target_primary", "targets"),
            ("target_valid", "target_mask"),
            ("target_raw_midrank", "raw_targets"),
            ("target_raw_valid", "raw_target_mask"),
            ("target_raw_log_return", "raw_log_returns"),
            ("target_to_close", "to_close_target"),
            ("target_to_close_valid", "to_close_mask"),
            ("intraday_values", "intraday_features"),
            ("intraday_valid", "intraday_feature_mask"),
        ):
            if self.store.has_array(source):
                sample[destination] = self.store.read(source, date_index)
        for key in ("target_mask", "raw_target_mask"):
            value = sample.get(key)
            if not isinstance(value, np.ndarray):
                continue
            clipped = np.asarray(value, dtype=np.bool_).copy()
            for horizon_index, horizon in enumerate(HORIZONS):
                if any(
                    endpoint not in self._sample_date_indices
                    for endpoint in range(date_index, date_index + horizon + 1)
                ):
                    clipped[:, horizon_index] = False
            sample[key] = clipped
        if isinstance(sample.get("to_close_mask"), np.ndarray):
            sample["to_close_mask"] = np.asarray(
                sample["to_close_mask"], dtype=np.bool_
            ) & np.asarray(fast_present, dtype=np.bool_)
        for value_key, mask_key in (
            ("slow_features", "slow_feature_mask"),
            ("fast_patches", "fast_patch_mask"),
            ("v1_equity_slow", "fast_present"),
            ("intraday_features", "intraday_feature_mask"),
            ("targets", "target_mask"),
            ("raw_targets", "raw_target_mask"),
            ("raw_log_returns", "raw_target_mask"),
            ("to_close_target", "to_close_mask"),
        ):
            value = sample.get(value_key)
            if value is None:
                continue
            mask = sample.get(mask_key)
            if not isinstance(value, np.ndarray) or not isinstance(mask, np.ndarray):
                raise ValueError(f"{value_key} requires its aligned validity mask")
            sample[value_key] = _zero_invalid_values(
                value,
                mask,
                name=value_key,
            )
        for key, value in sample.items():
            if (
                isinstance(value, np.ndarray)
                and np.issubdtype(value.dtype, np.floating)
                and not np.isfinite(value).all()
            ):
                raise ValueError(f"dataset boundary produced non-finite {key}")
        return sample


def store_date_lookup(store: V2Store) -> Mapping[str, int]:
    return {str(value): index for index, value in enumerate(store.dates)}
