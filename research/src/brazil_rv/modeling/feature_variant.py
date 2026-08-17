from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .contract import VALIDATION_END

VARIANT_SCHEMA = "EQUITY_INTRADAY_NORMALIZATION_OVERLAY_V1"
VARIANT_MANIFEST = "intraday_normalization_variant.json"


def load_variant_manifest(store: Path) -> dict[str, object] | None:
    path = store / VARIANT_MANIFEST
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != VARIANT_SCHEMA:
        raise ValueError("Wrong feature-variant schema")
    if (
        manifest.get("test_accessed") is not False
        or manifest.get("test_rows_present") is not False
    ):
        raise ValueError("Feature variant is not development-only")
    expected_gamma = {"equity_tod_half": 0.5, "equity_tod_full": 1.0}
    if manifest.get("gamma") != expected_gamma.get(str(manifest.get("arm"))):
        raise ValueError("Feature variant has an invalid arm/gamma contract")
    if manifest.get("allowed_date_end") != str(VALIDATION_END):
        raise ValueError("Feature variant has an invalid validation boundary")
    return manifest


def variant_parent(store: Path, manifest: dict[str, object]) -> Path:
    parent = Path(manifest["canonical_parent_feature_store"]["path"])
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    if parent.resolve() == store.resolve():
        raise ValueError("Feature variant cannot parent itself")
    return parent


def validate_variant_binding(
    store: Path,
    manifest: dict[str, object],
    parent_identity: dict[str, object],
    *,
    verify_overlay_hashes: bool,
) -> None:
    if manifest["canonical_parent_feature_store"] != parent_identity:
        raise ValueError("Feature variant parent identity mismatch")
    for key in ("dynamic_overlay", "peer_overlay"):
        entry = manifest[key]
        path = store / entry["file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(values.shape) != entry["shape"] or values.dtype != np.dtype(
            entry["dtype"]
        ):
            raise ValueError(f"Feature-variant array contract mismatch: {path.name}")
        if verify_overlay_hashes:
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"Feature-variant array hash mismatch: {path.name}")


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
