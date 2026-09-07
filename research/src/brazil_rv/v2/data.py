from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset, default_collate

from .contract import (
    ALLOWED_LOOKBACKS,
    DECISION_SAMPLE_SCHEMA,
    DECISION_MINUTE_INDEX,
    FAST_PATCH_MINUTES,
    FINETUNE_START,
    HORIZONS,
    PRETRAIN_END,
    REGISTERED_PRIMARY_TARGET,
    REGISTERED_PRIMARY_TARGET_MASK,
    STORE_START,
    V1_STORE_V2_ZERO_SLOW_FIELDS,
)
from .data_roots import ExternalFileResolution, resolve_external_files
from .store import V2Store, open_store_for_samples
from .splits import AccessPurpose, PREREGISTRATION_ROOT, authorize_dates

Stage = Literal["pretrain", "finetune", "evaluation", "joint"]
V1_STORE_V2_ZERO_DYNAMIC_CHANNELS = (9, 11, 14, 22, 24, 25)
_NATIVE_FAST_CHANNELS = 7
_LEGACY_FAST_PREFIX_PATCHES = 12
_COMPACT_FAST_KEYS = frozenset(
    {
        "fast_patch_values",
        "fast_patch_valid",
        "fast_patch_mask",
        "fast_name_index",
        "fast_state_position",
        "v1_equity_slow",
    }
)


@dataclass(frozen=True)
class ScalarFeatureView:
    """One ordered store-backed decision-axis scalar feature view."""

    date_indices: NDArray[np.int64]
    dates: NDArray[np.datetime64]
    isins: tuple[str, ...]
    active: NDArray[np.bool_]
    names: tuple[str, ...]
    values: NDArray[np.float32]
    valid: NDArray[np.bool_]
    age_sessions: NDArray[np.float32]


def scalar_feature_names(store: V2Store, families: Sequence[str]) -> tuple[str, ...]:
    """Resolve requested scalar families in canonical FeatureSpec order."""

    requested = set(families)
    if len(requested) != len(families):
        raise ValueError("scalar feature families must be unique")
    allowed = {"slow", "intraday"} | {
        key for key in requested if isinstance(key, str) and key.startswith("sidecar_")
    }
    if requested - allowed:
        raise ValueError(
            f"unsupported scalar feature families: {sorted(requested - allowed)}"
        )
    ordered = tuple(
        family for family in ("slow", "intraday") if family in requested
    ) + tuple(sorted(family for family in requested if family.startswith("sidecar_")))
    manifest_names = store.manifest.get("feature_names")
    if not isinstance(manifest_names, Mapping):
        raise ValueError("store manifest lacks ordered feature names")
    names: list[str] = []
    for family in ordered:
        family_names = manifest_names.get(family)
        if (
            not isinstance(family_names, list)
            or not family_names
            or not all(isinstance(name, str) and name for name in family_names)
        ):
            raise ValueError(f"store manifest lacks ordered {family} feature names")
        names.extend(family_names)
    return tuple(names)


def read_scalar_feature_view(
    store: V2Store,
    date_indices: Sequence[int] | NDArray[np.integer],
    families: Sequence[str],
) -> ScalarFeatureView:
    """Read one canonical date/security scalar view for any model consumer."""

    indices = np.asarray(date_indices, dtype=np.int64)
    if indices.ndim != 1 or not indices.size or np.any(indices < 0):
        raise ValueError("scalar feature rows must be a nonempty index vector")
    requested = set(families)
    ordered = tuple(
        family for family in ("slow", "intraday") if family in requested
    ) + tuple(sorted(family for family in requested if family.startswith("sidecar_")))
    names = scalar_feature_names(store, families)
    active = np.asarray(store.read("active", indices), dtype=np.bool_)
    expected_axis = (indices.size, len(store.isins))
    if active.shape != expected_axis:
        raise ValueError("scalar feature activity is misaligned with the store axes")
    values: list[NDArray[np.float32]] = []
    validity: list[NDArray[np.bool_]] = []
    ages: list[NDArray[np.float32]] = []
    for family in ordered:
        family_values = np.asarray(
            store.read(f"{family}_values", indices), dtype=np.float32
        )
        family_valid = np.asarray(
            store.read(f"{family}_valid", indices), dtype=np.bool_
        )
        family_age = np.asarray(
            store.read(f"{family}_age_sessions", indices), dtype=np.float32
        )
        if (
            family_values.ndim != 3
            or family_valid.shape != family_values.shape
            or family_age.shape != family_values.shape
        ):
            raise ValueError(f"{family} scalar feature arrays are misaligned")
        if (
            np.isinf(family_values).any()
            or not np.isfinite(family_values[family_valid]).all()
        ):
            raise ValueError(f"{family} has non-finite values marked available")
        if (
            not np.isfinite(family_age).all()
            or np.any(family_age < -1.0)
            or np.any(family_valid & (family_age < 0.0))
        ):
            raise ValueError(f"{family} feature ages violate the scalar contract")
        values.append(family_values)
        validity.append(family_valid)
        ages.append(family_age)
    if values:
        combined_values = np.concatenate(values, axis=-1, dtype=np.float32)
        combined_valid = np.concatenate(validity, axis=-1, dtype=np.bool_)
        combined_age = np.concatenate(ages, axis=-1, dtype=np.float32)
    else:
        combined_values = np.empty((*expected_axis, 0), dtype=np.float32)
        combined_valid = np.empty((*expected_axis, 0), dtype=np.bool_)
        combined_age = np.empty((*expected_axis, 0), dtype=np.float32)
    if combined_values.shape[-1] != len(names):
        raise ValueError("scalar feature names do not match the stored feature width")
    return ScalarFeatureView(
        date_indices=indices,
        dates=np.asarray(store.dates[indices], dtype="datetime64[D]"),
        isins=tuple(store.isins),
        active=active,
        names=names,
        values=combined_values,
        valid=combined_valid,
        age_sessions=combined_age,
    )


def collate_v2_daily(
    samples: Sequence[Mapping[str, object]],
    *,
    fixed_fast_name_count: int | None = None,
) -> dict[str, object]:
    """Collate daily panels on one stage-fixed sparse fast-name axis."""

    if not samples:
        raise ValueError("cannot collate an empty v2 batch")
    compact: list[dict[str, NDArray[np.generic]]] = []
    patch_shape: tuple[int, int] | None = None
    max_fast_names = 0
    for sample in samples:
        fields: dict[str, NDArray[np.generic]] = {}
        missing = _COMPACT_FAST_KEYS - sample.keys()
        if missing:
            raise ValueError(f"v2 sample lacks compact fast fields: {sorted(missing)}")
        for key in _COMPACT_FAST_KEYS:
            value = sample[key]
            if not isinstance(value, np.ndarray):
                raise TypeError(f"{key} must be a NumPy array before collation")
            fields[key] = value
        values = fields["fast_patch_values"]
        valid = fields["fast_patch_valid"]
        mask = fields["fast_patch_mask"]
        names = fields["fast_name_index"]
        positions = fields["fast_state_position"]
        legacy_slow = fields["v1_equity_slow"]
        if values.ndim != 3:
            raise ValueError("fast_patch_values must have shape [fast, patch, channel]")
        fast_count, patch_count, channel_count = values.shape
        if (
            valid.shape != values.shape
            or mask.shape != (fast_count, patch_count)
            or names.shape != (fast_count,)
            or positions.shape != (fast_count,)
            or legacy_slow.shape != (fast_count, 32)
        ):
            raise ValueError("compact fast fields are misaligned")
        current_patch_shape = (patch_count, channel_count)
        if patch_shape is None:
            patch_shape = current_patch_shape
        elif patch_shape != current_patch_shape:
            raise ValueError("one v2 batch cannot mix fast patch layouts")
        compact.append(fields)
        max_fast_names = max(max_fast_names, fast_count)

    if fixed_fast_name_count is not None:
        if fixed_fast_name_count <= 0 or fixed_fast_name_count % 16:
            raise ValueError("fixed fast-name count must be a positive multiple of 16")
        if max_fast_names > fixed_fast_name_count:
            raise ValueError(
                "batch fast-name count exceeds the registered stage padding width"
            )
        max_fast_names = fixed_fast_name_count

    assert patch_shape is not None
    patch_count, channel_count = patch_shape
    padded: dict[str, list[torch.Tensor]] = {key: [] for key in _COMPACT_FAST_KEYS}
    for fields in compact:
        fast_count = fields["fast_name_index"].shape[0]
        values = np.zeros(
            (max_fast_names, patch_count, channel_count), dtype=np.float32
        )
        valid = np.zeros(values.shape, dtype=np.bool_)
        mask = np.zeros((max_fast_names, patch_count), dtype=np.bool_)
        names = np.full(max_fast_names, -1, dtype=np.int64)
        positions = np.zeros(max_fast_names, dtype=np.int64)
        legacy_slow = np.zeros((max_fast_names, 32), dtype=np.float32)
        if fast_count:
            values[:fast_count] = fields["fast_patch_values"]
            valid[:fast_count] = fields["fast_patch_valid"]
            mask[:fast_count] = fields["fast_patch_mask"]
            names[:fast_count] = fields["fast_name_index"]
            positions[:fast_count] = fields["fast_state_position"]
            legacy_slow[:fast_count] = fields["v1_equity_slow"]
        for key, value in (
            ("fast_patch_values", values),
            ("fast_patch_valid", valid),
            ("fast_patch_mask", mask),
            ("fast_name_index", names),
            ("fast_state_position", positions),
            ("v1_equity_slow", legacy_slow),
        ):
            padded[key].append(torch.from_numpy(value))

    ordinary = [
        {key: value for key, value in sample.items() if key not in _COMPACT_FAST_KEYS}
        for sample in samples
    ]
    result = dict(default_collate(ordinary))
    result.update({key: torch.stack(values) for key, values in padded.items()})
    return result


def stage_fast_name_count(*datasets: "V2DailyDataset") -> int:
    """Return the fixed stage name width: maximum active names rounded to 16."""

    if not datasets:
        raise ValueError("at least one dataset is required")
    maximum = 0
    for dataset in datasets:
        active = np.asarray(
            dataset.store.read("active", dataset.date_indices), dtype=np.bool_
        )
        maximum = max(maximum, int(active.sum(axis=1).max(initial=0)))
    if maximum <= 0:
        raise ValueError("stage contains no active names")
    return ((maximum + 15) // 16) * 16


def _validate_stage_dates(selected_dates: NDArray[np.datetime64], stage: Stage) -> None:
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


def slow_row_index(sample_date_index: int, stage: Stage) -> int:
    if sample_date_index < 0:
        raise ValueError("sample date index must be non-negative")
    if stage not in {"pretrain", "finetune", "evaluation", "joint"}:
        raise ValueError(f"unknown v2 stage: {stage}")
    return sample_date_index


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
        end = slow_row_index(index, stage)
        if end >= 0:
            required.update(range(max(0, end - lookback + 1), end + 1))
    return np.asarray(sorted(required), dtype=np.int64)


def causal_history_end_offsets(
    date_indices: Sequence[int],
    *,
    stage: Stage,
) -> NDArray[np.int64]:
    """Return each sample's frozen slow-history endpoint relative to t."""

    indices = np.asarray(date_indices, dtype=np.int64)
    offsets = np.empty(len(indices), dtype=np.int64)
    for row, date_index in enumerate(indices):
        index = int(date_index)
        offsets[row] = slow_row_index(index, stage) - index
    return offsets


def lazy_slow_window(
    values: NDArray[np.floating],
    valid: NDArray[np.bool_],
    timestep_valid: NDArray[np.bool_],
    age_sessions: NDArray[np.floating],
    *,
    end_index: int,
    lookback: int,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.bool_],
    NDArray[np.bool_],
    NDArray[np.float32],
]:
    """Create a left-padded `[name, lookback, feature]` view on demand."""

    source = np.asarray(values)
    mask = np.asarray(valid, dtype=np.bool_)
    timesteps = np.asarray(timestep_valid, dtype=np.bool_)
    ages = np.asarray(age_sessions, dtype=np.float32)
    if source.ndim != 3 or source.shape != mask.shape:
        raise ValueError("slow values and validity must align [date, name, feature]")
    if ages.shape != source.shape:
        raise ValueError("slow feature ages must align [date, name, feature]")
    if timesteps.shape != source.shape[:2]:
        raise ValueError("slow timestep validity must align [date, name]")
    if np.any(mask & ~timesteps[..., None]):
        raise ValueError("slow features cannot be valid outside real timesteps")
    if (
        not np.isfinite(ages).all()
        or np.any(mask & (ages < 0.0))
        or np.any(ages < -1.0)
        or np.any((ages >= 0.0) & ~timesteps[..., None])
    ):
        raise ValueError(
            "slow feature ages must be finite last-observation ages or -1, known "
            "only on real timesteps, with every valid feature age known"
        )
    if lookback not in ALLOWED_LOOKBACKS:
        raise ValueError("lookback must be 20, 60, or 120")
    output = np.zeros((source.shape[1], lookback, source.shape[2]), dtype=np.float32)
    output_valid = np.zeros(output.shape, dtype=np.bool_)
    history_mask = np.zeros((source.shape[1], lookback), dtype=np.bool_)
    output_age = np.full(output.shape, -1.0, dtype=np.float32)
    if end_index < 0:
        return output, output_valid, history_mask, output_age
    if end_index >= source.shape[0]:
        raise IndexError("slow end index exceeds store")
    start = max(0, end_index - lookback + 1)
    sample = np.asarray(source[start : end_index + 1], dtype=np.float32).transpose(
        1, 0, 2
    )
    sample_valid = mask[start : end_index + 1].transpose(1, 0, 2)
    sample_timesteps = timesteps[start : end_index + 1].transpose(1, 0)
    sample_age = ages[start : end_index + 1].transpose(1, 0, 2)
    offset = lookback - sample.shape[1]
    output[:, offset:] = np.where(sample_valid, sample, 0.0)
    output_valid[:, offset:] = sample_valid
    history_mask[:, offset:] = sample_timesteps
    output_age[:, offset:] = np.where(sample_timesteps[..., None], sample_age, -1.0)
    return output, output_valid, history_mask, output_age


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
        target_window_indices: Sequence[int] | None = None,
        fast_store: str | Path | None = None,
        verify_fast_hashes: bool = True,
        purpose: AccessPurpose | None = None,
        registration_path: Path | None = None,
        preregistration_root: Path = PREREGISTRATION_ROOT,
    ) -> None:
        self.date_indices = np.asarray(date_indices, dtype=np.int64)
        self.stage = stage
        self.lookback = lookback
        self.primary_target_name = REGISTERED_PRIMARY_TARGET
        self.primary_target_mask_name = REGISTERED_PRIMARY_TARGET_MASK
        if len(set(enabled_sidecars)) != len(enabled_sidecars):
            raise ValueError("enabled sidecar groups must be unique")
        self.enabled_sidecars = tuple(sorted(enabled_sidecars))
        target_indices = (
            self.date_indices
            if target_window_indices is None
            else np.asarray(target_window_indices, dtype=np.int64)
        )
        if target_indices.ndim != 1 or not target_indices.size:
            raise ValueError("target window indices must be a nonempty vector")
        if np.unique(target_indices).size != target_indices.size or np.any(
            np.diff(target_indices) <= 0
        ):
            raise ValueError(
                "target window indices must be strictly ordered and unique"
            )
        self.target_window_indices = target_indices
        self._target_date_indices = frozenset(int(value) for value in target_indices)
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
                (self.date_indices < 0) | (self.date_indices >= self.store.dates.size)
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
            if not self.store.authorized_for(self.target_window_indices):
                raise PermissionError(
                    "the open store was not authorized for the declared target window"
                )
            selected_dates = self.store.dates[self.date_indices]
            _validate_stage_dates(selected_dates, stage)
            requested = tuple(sorted(selected_dates.astype(object).tolist()))
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
            if np.any((self.date_indices < 0) | (self.date_indices >= date_axis.size)):
                raise ValueError("date indices are outside the store")
            _validate_stage_dates(date_axis[self.date_indices], stage)
            history_offsets = causal_history_end_offsets(
                self.date_indices,
                stage=stage,
            )
            self.store, access_ledger = open_store_for_samples(
                store_path,
                self.date_indices,
                purpose=access_purpose,
                history_lookbacks=lookback,
                history_end_offsets=history_offsets,
                target_window_indices=self.target_window_indices,
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
        self._native_fast_store_indices = np.empty(0, dtype=np.int64)
        self._fast_patch_count = DECISION_MINUTE_INDEX // FAST_PATCH_MINUTES
        self._fast_channel_count = _NATIVE_FAST_CHANNELS
        if np.any(
            (self.date_indices < 0) | (self.date_indices >= self.store.dates.size)
        ):
            raise ValueError("date indices are outside the store")
        if np.any(
            (self.target_window_indices < 0)
            | (self.target_window_indices >= self.store.dates.size)
        ):
            raise ValueError("target window indices are outside the store")
        if not set(self.date_indices.tolist()).issubset(self._target_date_indices):
            raise ValueError("every sample date must be inside its target window")
        self.store.array_shape("slow_values")
        self.store.array_shape("slow_valid")
        if self.store.array_shape("slow_age_sessions") != self.store.array_shape(
            "slow_values"
        ):
            raise ValueError("slow feature ages are misaligned with the store")
        if self.store.array_shape("slow_timestep_valid") != (
            self.store.dates.size,
            len(self.store.isins),
        ):
            raise ValueError("slow_timestep_valid is misaligned with the store axes")
        self.store.array_shape("active")
        if not self.store.has_array("intraday_values") or not self.store.has_array(
            "intraday_valid"
        ):
            raise ValueError("canonical v2 samples require current intraday features")
        if self.store.array_shape("intraday_age_sessions") != self.store.array_shape(
            "intraday_values"
        ):
            raise ValueError("current feature ages are misaligned with the store")
        for group in self.enabled_sidecars:
            self.store.array_shape(f"sidecar_{group}_values")
            self.store.array_shape(f"sidecar_{group}_valid")
            if self.store.array_shape(
                f"sidecar_{group}_age_sessions"
            ) != self.store.array_shape(f"sidecar_{group}_values"):
                raise ValueError(f"{group} source ages are misaligned with the store")
        native_arrays = {
            "fast_patch_values",
            "fast_patch_valid",
            "fast_patch_mask",
        }
        native_present = native_arrays.intersection(self.store.array_names)
        if native_present and native_present != native_arrays:
            raise ValueError("the native fast arrays must be stored together")
        if native_present:
            if not self.store.has_array("fast_present"):
                raise ValueError(
                    "native fast arrays require their stored fast_present source mask"
                )
            value_shape = self.store.array_shape("fast_patch_values")
            if (
                len(value_shape) != 4
                or value_shape[0] != self.store.dates.size
                or value_shape[-1] != _NATIVE_FAST_CHANNELS
                or self.store.array_shape("fast_patch_valid") != value_shape
                or self.store.array_shape("fast_patch_mask") != value_shape[:-1]
            ):
                raise ValueError("native fast arrays have the wrong contract")
            mapping = self.store.read_table("native_fast_security_mapping").sort(
                "fast_index"
            )
            self._native_fast_store_indices = (
                mapping.get_column("store_name_index").to_numpy().astype(np.int64)
            )
            if not np.array_equal(
                mapping.get_column("fast_index").to_numpy(),
                np.arange(value_shape[1], dtype=np.int64),
            ):
                raise ValueError("native fast indices must be contiguous")
            if self._native_fast_store_indices.shape != (value_shape[1],):
                raise ValueError("native fast mapping does not cover its array axis")
            self._fast_patch_count = int(value_shape[2])
            self._fast_channel_count = int(value_shape[3])
        configured_fast = fast_store
        if configured_fast is None and not native_present:
            configured_fast = self.store.manifest.get("metadata", {}).get(
                "v1_fast_store"
            )
        if configured_fast and native_present:
            raise ValueError("native and legacy fast streams cannot be mixed")
        if configured_fast:
            self._open_external_fast(
                None if fast_store is None else Path(configured_fast),
                verify_fast_hashes,
            )
            self._fast_channel_count = 26 * FAST_PATCH_MINUTES

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
        date_mapping = self.store.read_table("v1_fast_date_mapping", self.date_indices)
        isin_mapping = self.store.read_table("v1_fast_isin_mapping")
        for target, source in date_mapping.select(
            "v2_date_index", "v1_date_index"
        ).iter_rows():
            if (
                not 0 <= target < self.store.dates.size
                or not 0 <= source < features.shape[0]
            ):
                raise ValueError("external v1 date mapping is outside its axes")
            if self._fast_date_mapping[target] >= 0:
                raise ValueError("external v1 date mapping is not one-to-one")
            self._fast_date_mapping[target] = source
        self._fast_v2_slots = (
            isin_mapping.get_column("v2_isin_index").to_numpy().astype(np.int64)
        )
        self._fast_v1_slots = (
            isin_mapping.get_column("v1_equity_slot").to_numpy().astype(np.int64)
        )
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
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.bool_],
        NDArray[np.bool_],
        NDArray[np.float32],
    ]:
        """Read and concatenate only the requested windows from mmap sources."""

        start = max(0, end_index - self.lookback + 1)
        indices = np.arange(start, end_index + 1, dtype=np.int64)
        view = read_scalar_feature_view(
            self.store,
            indices,
            ("slow", *(f"sidecar_{group}" for group in self.enabled_sidecars)),
        )
        return lazy_slow_window(
            view.values,
            view.valid,
            self.store.read("slow_timestep_valid", indices),
            view.age_sessions,
            end_index=len(indices) - 1,
            lookback=self.lookback,
        )

    def _empty_fast(
        self,
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.bool_],
        NDArray[np.bool_],
        NDArray[np.int64],
        NDArray[np.int64],
        NDArray[np.bool_],
    ]:
        return (
            np.zeros(
                (0, self._fast_patch_count, self._fast_channel_count),
                dtype=np.float32,
            ),
            np.zeros(
                (0, self._fast_patch_count, self._fast_channel_count),
                dtype=np.bool_,
            ),
            np.zeros((0, self._fast_patch_count), dtype=np.bool_),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.zeros(len(self.store.isins), dtype=np.bool_),
        )

    def _fast(
        self, date_index: int
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.bool_],
        NDArray[np.bool_],
        NDArray[np.int64],
        NDArray[np.int64],
        NDArray[np.bool_],
    ]:
        name_count = len(self.store.isins)
        if self._sample_is_pretrain(date_index):
            return self._empty_fast()
        if self.store.has_array("fast_patch_values"):
            values = np.asarray(
                self.store.read("fast_patch_values", date_index), dtype=np.float32
            )
            valid = np.asarray(
                self.store.read("fast_patch_valid", date_index), dtype=np.bool_
            )
            patch_mask = np.asarray(
                self.store.read("fast_patch_mask", date_index), dtype=np.bool_
            )
            if (
                values.shape
                != (
                    self._native_fast_store_indices.size,
                    self._fast_patch_count,
                    self._fast_channel_count,
                )
                or valid.shape != values.shape
                or patch_mask.shape != values.shape[:-1]
            ):
                raise ValueError("native fast row differs from its frozen axes")
            if np.any(valid & ~patch_mask[..., None]):
                raise ValueError("native fast validity escapes its patch mask")
            state_position = patch_mask.sum(axis=1, dtype=np.int64)
            expected_mask = (
                np.arange(self._fast_patch_count, dtype=np.int64)[None, :]
                < state_position[:, None]
            )
            if not np.array_equal(patch_mask, expected_mask):
                raise ValueError("native fast patch masks must be contiguous prefixes")
            stored_present = np.asarray(
                self.store.read("fast_present", date_index), dtype=np.bool_
            )
            if stored_present.shape != (name_count,):
                raise ValueError("stored fast_present is misaligned with name axis")
            native_present = stored_present[self._native_fast_store_indices]
            if np.any(native_present & (state_position == 0)):
                raise ValueError(
                    "stored fast_present cannot identify an empty native patch stream"
                )
            selected = native_present & (state_position > 0)
            if not selected.any():
                return self._empty_fast()
            compact_valid = valid[selected]
            compact_values = _zero_invalid_values(
                values[selected], compact_valid, name="fast_patch_values"
            ).astype(np.float32, copy=False)
            name_index = self._native_fast_store_indices[selected]
            present = np.zeros(name_count, dtype=np.bool_)
            present[name_index] = True
            return (
                compact_values,
                compact_valid,
                patch_mask[selected],
                name_index,
                state_position[selected],
                present,
            )
        source_date = int(self._fast_date_mapping[date_index])
        if self._external_fast_features is None or source_date < 0:
            return self._empty_fast()
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
        minute_valid = np.broadcast_to(source_ready[:, None], source.shape[:2])
        source_patches, source_patch_mask = pack_fast_patches(source, minute_valid)
        source_valid = np.broadcast_to(
            source_patch_mask[..., None], source_patches.shape
        ).copy()
        selected = source_ready & source_patch_mask.any(axis=1)
        if not selected.any():
            return self._empty_fast()
        name_index = self._fast_v2_slots[selected]
        present = np.zeros(name_count, dtype=np.bool_)
        present[name_index] = True
        return (
            source_patches[selected],
            source_valid[selected],
            source_patch_mask[selected],
            name_index,
            np.full(
                selected.sum(),
                _LEGACY_FAST_PREFIX_PATCHES + self._fast_patch_count,
                dtype=np.int64,
            ),
            present,
        )

    def _v1_equity_slow(
        self,
        date_index: int,
        fast_name_index: NDArray[np.int64],
    ) -> NDArray[np.float32]:
        """Return the legacy slow context on the same compact fast-name axis."""

        output = np.zeros((fast_name_index.size, 32), dtype=np.float32)
        if self._sample_is_pretrain(date_index) or self._external_fast_slow is None:
            return output
        source_date = int(self._fast_date_mapping[date_index])
        if source_date < 0 or not fast_name_index.size:
            return output
        v1_by_v2 = dict(zip(self._fast_v2_slots.tolist(), self._fast_v1_slots.tolist()))
        try:
            source_slots = np.asarray(
                [v1_by_v2[int(index)] for index in fast_name_index], dtype=np.int64
            )
        except KeyError as error:
            raise ValueError("compact legacy fast name lacks its v1 mapping") from error
        source = np.asarray(
            self._external_fast_slow[source_date, source_slots],
            dtype=np.float32,
        ).copy()
        source[..., V1_STORE_V2_ZERO_SLOW_FIELDS] = 0.0
        if not np.isfinite(source).all():
            raise ValueError("present external v1 slow rows must be finite")
        return source

    def __getitem__(self, item: int) -> dict[str, object]:
        date_index = int(self.date_indices[item])
        slow_end = slow_row_index(date_index, self.stage)
        history, feature_mask, history_mask, feature_age = self._slow_window(slow_end)
        (
            fast_values,
            fast_valid,
            patch_mask,
            fast_name_index,
            fast_state_position,
            fast_present,
        ) = self._fast(date_index)
        v1_equity_slow = self._v1_equity_slow(date_index, fast_name_index)
        active = np.asarray(self.store.read("active", date_index), dtype=np.bool_)
        current_view = read_scalar_feature_view(
            self.store, np.asarray([date_index], dtype=np.int64), ("intraday",)
        )
        sample: dict[str, object] = {
            "schema": DECISION_SAMPLE_SCHEMA,
            "date_index": np.int64(date_index),
            "trade_date": str(self.store.dates[date_index]),
            "slow_features": history,
            "slow_feature_mask": feature_mask,
            "slow_history_mask": history_mask,
            "slow_feature_age_sessions": feature_age,
            "fast_patch_values": fast_values,
            "fast_patch_valid": fast_valid,
            "fast_patch_mask": patch_mask,
            "fast_name_index": fast_name_index,
            "fast_state_position": fast_state_position,
            "fast_present": fast_present,
            "v1_equity_slow": v1_equity_slow,
            "active_mask": active,
            "current_features": current_view.values[0],
            "current_feature_mask": current_view.valid[0],
            "current_feature_age_sessions": current_view.age_sessions[0],
        }
        target_pairs = (
            (
                REGISTERED_PRIMARY_TARGET,
                REGISTERED_PRIMARY_TARGET_MASK,
                "targets",
                "target_mask",
            ),
            (
                "target_shareholder_midrank",
                "target_shareholder_valid",
                "shareholder_targets",
                "shareholder_target_mask",
            ),
            (
                "target_shareholder_simple_return",
                "target_shareholder_valid",
                "shareholder_simple_returns",
                "shareholder_target_mask",
            ),
            (
                "target_terminal_wealth",
                "target_shareholder_valid",
                "terminal_wealth",
                "shareholder_target_mask",
            ),
            (
                "target_terminal_loss",
                "target_shareholder_valid",
                "terminal_loss",
                "shareholder_target_mask",
            ),
            (
                "target_price_midrank",
                "target_price_valid",
                "price_targets",
                "price_target_mask",
            ),
            (
                "target_price_simple_return",
                "target_price_valid",
                "price_simple_returns",
                "price_target_mask",
            ),
        )
        target_masks: dict[str, NDArray[np.bool_]] = {}
        for _, mask_source, _, mask_destination in target_pairs:
            if mask_destination in target_masks or not self.store.has_array(
                mask_source
            ):
                continue
            clipped = np.asarray(
                self.store.read(mask_source, date_index), dtype=np.bool_
            ).copy()
            for horizon_index, horizon in enumerate(HORIZONS):
                if any(
                    endpoint not in self._target_date_indices
                    for endpoint in range(date_index, date_index + horizon + 1)
                ):
                    clipped[:, horizon_index] = False
            target_masks[mask_destination] = clipped
            sample[mask_destination] = clipped
        # Authorize and clip each endpoint mask before requesting its numeric
        # payload.  V2Store applies the same capability check at the mmap
        # boundary; this ordering also prevents a caller from observing a
        # value before its exact evaluation window has admitted the endpoint.
        for value_source, _, value_destination, mask_destination in target_pairs:
            mask = target_masks.get(mask_destination)
            if mask is None or not self.store.has_array(value_source):
                continue
            sample[value_destination] = self.store.read_target(
                value_source,
                date_index,
                valid_mask=mask,
            )
        for source, destination in (
            ("target_to_close_valid", "to_close_mask"),
            ("target_to_close", "to_close_target"),
        ):
            if self.store.has_array(source):
                sample[destination] = self.store.read(source, date_index)
        if isinstance(sample.get("to_close_mask"), np.ndarray):
            sample["to_close_mask"] = np.asarray(
                sample["to_close_mask"], dtype=np.bool_
            ) & np.asarray(fast_present, dtype=np.bool_)
        for value_key, mask_key in (
            ("slow_features", "slow_feature_mask"),
            ("fast_patch_values", "fast_patch_valid"),
            ("current_features", "current_feature_mask"),
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
