from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from brazil_rv.preprocessing.contract import output_array_specs
from brazil_rv.preprocessing.intraday_normalization import (
    AFFECTED_DYNAMIC_CHANNELS,
    AFFECTED_PEER_CHANNELS,
    ARMS,
    DECISION_FEATURE_MINUTES,
    DEVELOPMENT_IDENTITY_SCHEMA,
    INVARIANT_DYNAMIC_CHANNELS,
    VARIANT_SCHEMA,
    VISIBLE_EQUITY_MINUTES,
    canonical_frame_identity,
    canonical_json_identity,
    development_parent_identity,
)

from .contract import TRAIN_END, TRAIN_START, VALIDATION_END, VALIDATION_START

VARIANT_MANIFEST = "intraday_normalization_variant.json"
PARENT_METADATA_FILES = (
    "date_index.parquet",
    "sample_index.parquet",
    "equity_index.parquet",
    "context_index.parquet",
    "global_context_index.parquet",
    "feature_schema.json",
    "manifest_contract",
)


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"Feature variant has invalid {name}")
    return value


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_variant_manifest(store: Path) -> dict[str, object] | None:
    path = store / VARIANT_MANIFEST
    if not path.is_file():
        return None
    manifest = _object(json.loads(path.read_text(encoding="utf-8")), "manifest")
    if manifest.get("schema") != VARIANT_SCHEMA:
        raise ValueError("Wrong feature-variant schema")
    if (
        manifest.get("test_accessed") is not False
        or manifest.get("test_rows_present") is not False
    ):
        raise ValueError("Feature variant is not development-only")
    arm = manifest.get("arm")
    if (
        not isinstance(arm, str)
        or ARMS.get(arm, 0.0) <= 0.0
        or manifest.get("gamma") != ARMS[arm]
    ):
        raise ValueError("Feature variant has an invalid arm/gamma contract")
    if manifest.get("allowed_date_end") != str(VALIDATION_END):
        raise ValueError("Feature variant has an invalid validation boundary")
    date_count = manifest.get("allowed_date_count")
    if (
        isinstance(date_count, bool)
        or not isinstance(date_count, int)
        or date_count <= 0
    ):
        raise ValueError("Feature variant has an invalid development date count")
    if manifest.get("split_boundaries") != {
        "training": [str(TRAIN_START), str(TRAIN_END)],
        "validation": [str(VALIDATION_START), str(VALIDATION_END)],
    }:
        raise ValueError("Feature variant has invalid split boundaries")
    if (
        manifest.get("profile_freeze_date") != str(TRAIN_END)
        or manifest.get("validation_update_rule") != "frozen_training_end_profile"
    ):
        raise ValueError("Feature variant has an invalid frozen-profile contract")
    return manifest


def _recorded_parent_identity(manifest: dict[str, object]) -> dict[str, object]:
    identity = _object(
        manifest.get("canonical_parent_feature_store"), "parent identity"
    )
    if set(identity) != {
        "path",
        "contract_version",
        "metadata_sha256",
        "hash_scope",
    }:
        raise ValueError("Feature variant has an ambiguous parent identity")
    if not isinstance(identity.get("path"), str) or not isinstance(
        identity.get("contract_version"), str
    ):
        raise ValueError("Feature variant has an invalid parent identity")
    if not _valid_sha256(identity.get("metadata_sha256")):
        raise ValueError("Feature variant has an invalid parent development hash")
    return identity


def variant_parent(store: Path, manifest: dict[str, object]) -> Path:
    identity = _recorded_parent_identity(manifest)
    recorded = Path(str(identity["path"]))
    if not recorded.is_absolute() or str(recorded.resolve()) != str(recorded):
        raise ValueError("Feature variant parent path must be absolute and canonical")
    parent = recorded.resolve()
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    if parent.resolve() == store.resolve():
        raise ValueError("Feature variant cannot parent itself")
    return parent


def _development_scope(date_count: int) -> dict[str, object]:
    return {
        "kind": "development_only",
        "end_date": str(VALIDATION_END),
        "date_count": date_count,
        "date_array_scope": "prefix_only",
    }


def _development_rows(parent: Path, filename: str) -> pl.DataFrame:
    return (
        pl.scan_parquet(parent / filename)
        .filter(pl.col("trade_date") <= VALIDATION_END)
        .collect()
    )


def _validate_parent_artifacts(
    parent: Path,
    parent_manifest: dict[str, object],
    artifacts_record: dict[str, object],
    date_count: int,
) -> None:
    scope = _development_scope(date_count)
    if artifacts_record.get("schema") != DEVELOPMENT_IDENTITY_SCHEMA or (
        artifacts_record.get("hash_scope") != scope
    ):
        raise ValueError("Feature variant has an invalid parent hash scope")
    artifacts = _object(artifacts_record.get("artifacts"), "parent artifact identities")
    specs = output_array_specs(date_count)
    outputs = _object(parent_manifest.get("outputs"), "parent output inventory")
    if set(outputs) != set(specs) or set(artifacts) != {
        *specs,
        *PARENT_METADATA_FILES,
    }:
        raise ValueError("Feature variant parent artifact inventory mismatch")
    for filename, spec in specs.items():
        path = parent / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        record = _object(artifacts[filename], f"parent artifact {filename}")
        expected = {
            "scope": "date_prefix",
            "end_date": str(VALIDATION_END),
            "filename": filename,
            "dtype": spec.dtype.name,
            "development_shape": list(spec.shape),
        }
        if (
            set(record) != {*expected, "sha256"}
            or any(record.get(key) != value for key, value in expected.items())
            or not _valid_sha256(record.get("sha256"))
        ):
            raise ValueError(f"Feature variant parent artifact mismatch: {filename}")

    date_index = _development_rows(parent, "date_index.parquet")
    if (
        date_index.height != date_count
        or date_index.is_empty()
        or (date_index.get_column("trade_date").max() != VALIDATION_END)
        or not np.array_equal(
            date_index.sort("date_idx").get_column("date_idx").to_numpy(),
            np.arange(date_count),
        )
    ):
        raise ValueError("Feature variant parent date boundary mismatch")
    sample_index = _development_rows(parent, "sample_index.parquet")
    expected_metadata = {
        "date_index.parquet": {
            "scope": "rows_through_validation_end",
            "end_date": str(VALIDATION_END),
            **canonical_frame_identity(date_index, sort_by=("date_idx",)),
        },
        "sample_index.parquet": {
            "scope": "rows_through_validation_end",
            "end_date": str(VALIDATION_END),
            **canonical_frame_identity(sample_index, sort_by=("sample_id",)),
        },
    }
    for filename, sort_by in (
        ("equity_index.parquet", ("equity_slot",)),
        ("context_index.parquet", ("context_slot",)),
        ("global_context_index.parquet", ("global_slot",)),
    ):
        expected_metadata[filename] = {
            "scope": "complete_non_date_axis",
            **canonical_frame_identity(
                pl.read_parquet(parent / filename), sort_by=sort_by
            ),
        }
    schema = _object(
        json.loads((parent / "feature_schema.json").read_text(encoding="utf-8")),
        "parent feature schema",
    )
    expected_metadata["feature_schema.json"] = {
        "scope": "complete_non_observation_metadata",
        **canonical_json_identity(schema),
    }
    manifest_contract = {
        key: parent_manifest[key]
        for key in (
            "contract_version",
            "build_git_commit",
            "canonical_inputs",
            "constants",
        )
    }
    expected_metadata["manifest_contract"] = {
        "scope": "stage_relevant_non_observation_metadata",
        **canonical_json_identity(manifest_contract),
    }
    for filename, expected in expected_metadata.items():
        if artifacts.get(filename) != expected:
            raise ValueError(f"Feature variant parent metadata mismatch: {filename}")


def _validate_overlay(
    store: Path,
    manifest: dict[str, object],
    key: str,
    expected: dict[str, object],
    *,
    verify_hash: bool,
) -> None:
    entry = _object(manifest.get(key), key)
    if (
        set(entry) != {*expected, "sha256"}
        or any(entry.get(name) != value for name, value in expected.items())
        or not _valid_sha256(entry.get("sha256"))
    ):
        raise ValueError(f"Feature-variant metadata mismatch: {key}")
    path = store / str(entry["file"])
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(values.shape) != expected["shape"] or values.dtype != np.dtype(
        expected["dtype"]
    ):
        raise ValueError(f"Feature-variant array contract mismatch: {path.name}")
    if verify_hash:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"Feature-variant array hash mismatch: {path.name}")


def validate_variant_binding(
    store: Path,
    manifest: dict[str, object],
    *,
    verify_overlay_hashes: bool,
) -> tuple[Path, dict[str, object]]:
    parent_identity = _recorded_parent_identity(manifest)
    parent = variant_parent(store, manifest)
    parent_manifest = _object(
        json.loads((parent / "manifest.json").read_text(encoding="utf-8")),
        "parent manifest",
    )
    schema = _object(
        json.loads((parent / "feature_schema.json").read_text(encoding="utf-8")),
        "parent feature schema",
    )
    contract_version = manifest.get("contract_version")
    if not isinstance(contract_version, str) or any(
        value != contract_version
        for value in (
            parent_identity["contract_version"],
            parent_manifest.get("contract_version"),
            schema.get("contract_version"),
        )
    ):
        raise ValueError("Feature variant contract version mismatch")
    date_count = int(manifest["allowed_date_count"])
    artifacts_record = _object(
        manifest.get("parent_artifact_sha256"), "parent artifact record"
    )
    _validate_parent_artifacts(parent, parent_manifest, artifacts_record, date_count)
    expected_parent = development_parent_identity(
        parent, contract_version, artifacts_record
    )
    if parent_identity != expected_parent:
        raise ValueError("Feature variant parent identity mismatch")

    specs = output_array_specs(date_count)
    expected_affected = {
        "equity_features.npy": list(AFFECTED_DYNAMIC_CHANNELS),
        "equity_peer_features.npy": list(AFFECTED_PEER_CHANNELS),
    }
    if (
        manifest.get("affected_arrays") != expected_affected
        or manifest.get("parent_bound_arrays")
        != sorted(set(specs) - set(expected_affected))
        or manifest.get("parent_bound_dynamic_channels")
        != list(INVARIANT_DYNAMIC_CHANNELS)
        or manifest.get("parent_bound_peer_channels") != [2, 3]
    ):
        raise ValueError("Feature variant has an invalid parent-bound array contract")
    _validate_overlay(
        store,
        manifest,
        "dynamic_overlay",
        {
            "file": "equity_features_overlay.npy",
            "shape": [
                date_count,
                specs["equity_features.npy"].shape[1],
                VISIBLE_EQUITY_MINUTES,
                len(AFFECTED_DYNAMIC_CHANNELS),
            ],
            "dtype": "float32",
            "channels": list(AFFECTED_DYNAMIC_CHANNELS),
        },
        verify_hash=verify_overlay_hashes,
    )
    _validate_overlay(
        store,
        manifest,
        "peer_overlay",
        {
            "file": "equity_peer_features_overlay.npy",
            "shape": [
                date_count,
                specs["equity_peer_features.npy"].shape[1],
                len(DECISION_FEATURE_MINUTES),
                len(AFFECTED_PEER_CHANNELS),
            ],
            "dtype": "float32",
            "minutes": list(DECISION_FEATURE_MINUTES),
            "channels": list(AFFECTED_PEER_CHANNELS),
        },
        verify_hash=verify_overlay_hashes,
    )
    return parent, parent_identity


def feature_variant_identity(
    store: Path,
    manifest: dict[str, object] | None = None,
    *,
    verify_overlay_hashes: bool = False,
) -> dict[str, object]:
    store = store.resolve()
    if manifest is None:
        manifest = load_variant_manifest(store)
    if manifest is None:
        raise ValueError("Feature store is not a development-only variant")
    _, parent_identity = validate_variant_binding(
        store, manifest, verify_overlay_hashes=verify_overlay_hashes
    )
    digest = hashlib.sha256()
    digest.update((store / VARIANT_MANIFEST).read_bytes())
    digest.update(str(parent_identity["metadata_sha256"]).encode("ascii"))
    return {
        "path": str(store),
        "contract_version": parent_identity["contract_version"],
        "metadata_sha256": digest.hexdigest(),
        "hash_scope": parent_identity["hash_scope"],
    }


class OverlayArray:
    """Expose a sparse last-channel overlay with the parent's ndarray interface."""

    def __init__(
        self,
        parent: np.ndarray,
        overlay: np.ndarray,
        channels: tuple[int, ...],
        *,
        minute_positions: tuple[int, ...] | None = None,
    ) -> None:
        if parent.ndim != 4 or overlay.ndim != 4:
            raise ValueError("Feature overlays require four-dimensional arrays")
        self.parent = parent
        self.overlay = overlay
        self.channels = channels
        self.channel_position = {
            channel: position for position, channel in enumerate(channels)
        }
        self.minute_position = (
            None
            if minute_positions is None
            else {minute: position for position, minute in enumerate(minute_positions)}
        )
        self.shape = parent.shape
        self.dtype = parent.dtype
        self.ndim = parent.ndim

    @staticmethod
    def _expanded_key(key: Any) -> tuple[Any, Any, Any, Any]:
        if not isinstance(key, tuple):
            key = (key,)
        if any(value is Ellipsis for value in key):
            position = next(i for i, value in enumerate(key) if value is Ellipsis)
            fill = 4 - (len(key) - 1)
            key = (*key[:position], *(slice(None),) * fill, *key[position + 1 :])
        if len(key) > 4:
            raise IndexError("Too many indices for feature overlay")
        return (*key, *(slice(None),) * (4 - len(key)))

    def _overlay_base_key(self, key: tuple[Any, Any, Any, Any]) -> tuple[Any, Any, Any]:
        date_key, equity_key, minute_key, _ = key
        dates = np.asarray(np.arange(self.parent.shape[0])[date_key])
        if dates.size and (
            int(dates.min()) < 0 or int(dates.max()) >= self.overlay.shape[0]
        ):
            raise ValueError("Feature variant cannot serve held-out dates")
        minutes = np.asarray(np.arange(self.parent.shape[2])[minute_key])
        if self.minute_position is None:
            if minutes.size and (
                int(minutes.min()) < 0 or int(minutes.max()) >= self.overlay.shape[2]
            ):
                raise ValueError("Feature variant does not contain this minute")
            mapped_minute = minute_key
        else:
            flat = minutes.reshape(-1).tolist()
            if any(int(value) not in self.minute_position for value in flat):
                raise ValueError("Peer overlay only contains decision minutes")
            mapped = np.asarray(
                [self.minute_position[int(value)] for value in flat], dtype=np.int64
            ).reshape(minutes.shape)
            mapped_minute = int(mapped) if mapped.ndim == 0 else mapped
        return date_key, equity_key, mapped_minute

    def __getitem__(self, key: Any) -> np.ndarray:
        expanded = self._expanded_key(key)
        channel_key = expanded[3]
        selected = np.asarray(np.arange(self.parent.shape[3])[channel_key])
        if selected.ndim == 0:
            channel = int(selected)
            position = self.channel_position.get(channel)
            if position is None:
                return self.parent[expanded]
            base = self._overlay_base_key(expanded)
            return self.overlay[(*base, position)]

        output = np.asarray(self.parent[expanded]).copy()
        affected = [
            (output_position, self.channel_position[int(channel)])
            for output_position, channel in enumerate(selected.tolist())
            if int(channel) in self.channel_position
        ]
        if not affected:
            return output
        base = self._overlay_base_key(expanded)
        for output_position, overlay_position in affected:
            output[..., output_position] = self.overlay[(*base, overlay_position)]
        return output


def open_variant_arrays(
    store: Path,
    parent: Path,
    manifest: dict[str, object],
    filenames: tuple[str, ...],
) -> dict[str, np.ndarray | OverlayArray]:
    arrays: dict[str, np.ndarray | OverlayArray] = {
        name: np.load(parent / name, mmap_mode="r", allow_pickle=False)
        for name in filenames
    }
    if "equity_features.npy" in arrays:
        dynamic = manifest["dynamic_overlay"]
        arrays["equity_features.npy"] = OverlayArray(
            arrays["equity_features.npy"],
            np.load(store / dynamic["file"], mmap_mode="r", allow_pickle=False),
            tuple(dynamic["channels"]),
        )
    if "equity_peer_features.npy" in arrays:
        peer = manifest["peer_overlay"]
        arrays["equity_peer_features.npy"] = OverlayArray(
            arrays["equity_peer_features.npy"],
            np.load(store / peer["file"], mmap_mode="r", allow_pickle=False),
            tuple(peer["channels"]),
            minute_positions=tuple(peer["minutes"]),
        )
    return arrays
