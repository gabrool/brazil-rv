from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from .contract import ALLOWED_LOOKBACKS, DECISION_MINUTE_INDEX, HORIZONS
from .splits import (
    PREREGISTRATION_ROOT,
    AccessLedger,
    AccessPurpose,
    authorize_dates,
)

STORE_SCHEMA = "V2_DAILY_STORE_V1"
_SAFE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_WRITE_VERIFICATION = object()
_MAX_CAUSAL_HISTORY_ROWS = max(*ALLOWED_LOOKBACKS, 253)
_TARGET_VALUE_MASKS = {
    "target_primary": "target_valid",
    "target_normalized_residual": "target_valid",
    "target_raw_midrank": "target_raw_valid",
    "target_raw_log_return": "target_raw_valid",
    "target_to_close": "target_to_close_valid",
    "target_to_close_normalized_residual": "target_to_close_valid",
    "target_to_close_raw_log_return": "target_to_close_valid",
}
_MULTI_HORIZON_TARGET_MASKS = frozenset(("target_valid", "target_raw_valid"))
_VERIFIED_HASHES: set[tuple[str, int, int, str]] = set()


@dataclass(frozen=True)
class _StoreAccessGrant:
    """Internal capability created only after an exact date authorization."""

    root: Path
    date_indices: frozenset[int]
    ledger: AccessLedger


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matches_verified_file(path: Path, *, size: int, sha256: str) -> bool:
    """Verify immutable store content once per process and stable file stat."""

    stat = path.stat()
    if stat.st_size != size:
        return False
    identity = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, sha256)
    if identity in _VERIFIED_HASHES:
        return True
    if sha256_file(path) != sha256:
        return False
    _VERIFIED_HASHES.add(identity)
    return True


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _axis_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(values), ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _validate_axes(
    dates: NDArray[np.datetime64], isins: tuple[str, ...]
) -> None:
    if dates.ndim != 1 or dates.size == 0:
        raise ValueError("store needs a nonempty one-dimensional date axis")
    day_values = dates.astype("datetime64[D]").astype(np.int64)
    if np.any(np.diff(day_values) <= 0):
        raise ValueError("store dates must be strictly increasing and unique")
    if not isins or len(set(isins)) != len(isins) or any(not value for value in isins):
        raise ValueError("store ISINs must be nonempty and unique")


def _validate_array_shapes(
    arrays: Mapping[str, NDArray[np.generic]], date_count: int, isin_count: int
) -> None:
    for name, raw in arrays.items():
        if not _SAFE_NAME.fullmatch(name):
            raise ValueError(f"unsafe array name: {name}")
        value = np.asarray(raw)
        if value.ndim < 2 or value.shape[:2] != (date_count, isin_count):
            raise ValueError(
                f"{name} must begin with the [date, ISIN] axes; got {value.shape}"
            )
        if value.dtype == object:
            raise ValueError(f"object array is forbidden: {name}")
        if name.endswith("_valid") or name.endswith("_mask") or name in {
            "active",
            "observed",
            "fast_present",
            "unresolved_action",
        }:
            if value.dtype != np.bool_:
                raise ValueError(f"mask array must have boolean dtype: {name}")
    paired = (
        ("slow_values", "slow_valid"),
        ("intraday_values", "intraday_valid"),
        ("target_primary", "target_valid"),
        ("target_raw_midrank", "target_raw_valid"),
        ("target_to_close", "target_to_close_valid"),
    )
    for values_name, mask_name in paired:
        if (values_name in arrays) != (mask_name in arrays):
            raise ValueError(f"{values_name} and {mask_name} must be stored together")
        if values_name in arrays and arrays[values_name].shape != arrays[mask_name].shape:
            raise ValueError(f"{values_name} and {mask_name} are misaligned")


def write_store(
    output_dir: Path,
    *,
    dates: Sequence[object],
    isins: Sequence[str],
    arrays: Mapping[str, NDArray[np.generic]],
    row_indices: Sequence[int] | NDArray[np.integer] | None = None,
    feature_names: Mapping[str, Sequence[str]] | None = None,
    sources: Sequence[Mapping[str, object]] = (),
    metadata: Mapping[str, object] | None = None,
    tables: Mapping[str, pl.DataFrame] | None = None,
) -> Path:
    """Atomically create one immutable, manifest-hashed v2 store."""

    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    date_axis = np.asarray(dates, dtype="datetime64[D]")
    isin_axis = tuple(str(value) for value in isins)
    _validate_axes(date_axis, isin_axis)
    materialized = {name: np.asarray(value) for name, value in arrays.items()}
    if any(value.ndim < 2 for value in materialized.values()):
        raise ValueError("store arrays must begin with the [date, ISIN] axes")
    source_date_counts = {int(value.shape[0]) for value in materialized.values()}
    if len(source_date_counts) > 1:
        raise ValueError("store source arrays do not share one date axis")
    source_date_count = (
        date_axis.size if not source_date_counts else source_date_counts.pop()
    )
    _validate_array_shapes(materialized, source_date_count, len(isin_axis))
    selected_rows: NDArray[np.int64] | slice | None = None
    if row_indices is not None:
        raw_rows = np.asarray(row_indices)
        if (
            raw_rows.ndim != 1
            or not np.issubdtype(raw_rows.dtype, np.integer)
            or np.issubdtype(raw_rows.dtype, np.bool_)
        ):
            raise TypeError("row_indices must be a one-dimensional integer sequence")
        selected_rows = raw_rows.astype(np.int64, copy=False)
        if selected_rows.size != date_axis.size:
            raise ValueError("row_indices must select exactly one row per output date")
        if (
            np.any(selected_rows < 0)
            or np.any(selected_rows >= source_date_count)
            or (selected_rows.size > 1 and np.any(np.diff(selected_rows) <= 0))
        ):
            raise ValueError(
                "row_indices must be unique, increasing, and inside the source axis"
            )
        if selected_rows.size == 1 or np.all(np.diff(selected_rows) == 1):
            # A first-axis slice is a view; advanced indexing would allocate a
            # full copy of every selected tensor during the atomic write.
            selected_rows = slice(
                int(selected_rows[0]), int(selected_rows[-1]) + 1
            )
    elif source_date_count != date_axis.size:
        raise ValueError("store arrays do not match the output date axis")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    try:
        np.save(staging / "date_index.npy", date_axis, allow_pickle=False)
        np.save(staging / "isin_index.npy", np.asarray(isin_axis, dtype=np.str_), allow_pickle=False)
        inventory: dict[str, dict[str, object]] = {}
        for name, source_value in sorted(materialized.items()):
            value = (
                source_value
                if selected_rows is None
                else np.asarray(source_value[selected_rows])
            )
            path = staging / f"{name}.npy"
            np.save(path, value, allow_pickle=False)
            inventory[name] = {
                "path": path.name,
                "shape": list(value.shape),
                "dtype": value.dtype.str,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            del value
        index_inventory = {}
        for name in ("date_index.npy", "isin_index.npy"):
            path = staging / name
            index_inventory[name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        table_inventory: dict[str, dict[str, object]] = {}
        for name, frame in sorted((tables or {}).items()):
            if not _SAFE_NAME.fullmatch(name):
                raise ValueError(f"unsafe table name: {name}")
            path = staging / f"{name}.parquet"
            frame.write_parquet(path)
            table_inventory[name] = {
                "path": path.name,
                "rows": frame.height,
                "columns": frame.columns,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        manifest: dict[str, Any] = {
            "schema": STORE_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "axes": {
                "date_count": int(date_axis.size),
                "date_start": str(date_axis[0]),
                "date_end": str(date_axis[-1]),
                "date_identity_sha256": _axis_sha256([str(value) for value in date_axis]),
                "isin_count": len(isin_axis),
                "isin_identity_sha256": _axis_sha256(isin_axis),
                "horizons": list(HORIZONS),
                "decision_minute_index": DECISION_MINUTE_INDEX,
            },
            "indices": index_inventory,
            "arrays": inventory,
            "tables": table_inventory,
            "feature_names": {
                key: list(value) for key, value in (feature_names or {}).items()
            },
            "sources": [dict(value) for value in sources],
            "metadata": dict(metadata or {}),
            "official_validation_accessed": False,
            "test_accessed": False,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_bytes(_json_bytes(manifest))
        (staging / "manifest.sha256").write_text(
            f"{sha256_file(manifest_path)}  manifest.json\n", encoding="ascii"
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    opened = V2Store._open_internal(
        output, verify_hashes=True, grant=_WRITE_VERIFICATION
    )
    opened.close()
    return output


@dataclass
class V2Store:
    root: Path
    manifest: dict[str, Any]
    dates: NDArray[np.datetime64] = field(repr=False)
    isins: tuple[str, ...]
    _arrays: dict[str, NDArray[np.generic]] = field(repr=False)
    access_ledger: AccessLedger | None = None
    _authorized_date_indices: frozenset[int] = frozenset()

    @classmethod
    def open(
        cls,
        root: Path,
        *,
        verify_hashes: bool = True,
    ) -> "V2Store":
        del root, verify_hashes
        raise PermissionError(
            "v2 feature and target arrays must be opened through "
            "open_store_for_dates"
        )

    @classmethod
    def _open_internal(
        cls,
        root: Path,
        *,
        verify_hashes: bool,
        grant: _StoreAccessGrant | object,
    ) -> "V2Store":
        path = Path(root).resolve()
        write_verification = grant is _WRITE_VERIFICATION
        if not write_verification and (
            not isinstance(grant, _StoreAccessGrant) or grant.root != path
        ):
            raise PermissionError("invalid v2 store access capability")
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != STORE_SCHEMA:
            raise ValueError("not a v2 daily store")
        sha_record = (path / "manifest.sha256").read_text(encoding="ascii").split()[0]
        if verify_hashes and not _matches_verified_file(
            manifest_path,
            size=manifest_path.stat().st_size,
            sha256=sha_record,
        ):
            raise ValueError("store manifest hash mismatch")
        for name, record in manifest["indices"].items():
            item = path / name
            if verify_hashes and not _matches_verified_file(
                item, size=int(record["bytes"]), sha256=str(record["sha256"])
            ):
                raise ValueError(f"store index hash mismatch: {name}")
        dates = np.load(path / "date_index.npy", mmap_mode="r", allow_pickle=False)
        isin_array = np.load(path / "isin_index.npy", mmap_mode="r", allow_pickle=False)
        isins = tuple(str(value) for value in isin_array.tolist())
        _validate_axes(np.asarray(dates), isins)
        arrays: dict[str, NDArray[np.generic]] = {}
        for name, record in manifest["arrays"].items():
            item = path / record["path"]
            if verify_hashes and not _matches_verified_file(
                item, size=int(record["bytes"]), sha256=str(record["sha256"])
            ):
                raise ValueError(f"store array hash mismatch: {name}")
            value = np.load(item, mmap_mode="r", allow_pickle=False)
            if list(value.shape) != record["shape"] or value.dtype.str != record["dtype"]:
                raise ValueError(f"store array schema mismatch: {name}")
            arrays[name] = value
        for name, record in manifest.get("tables", {}).items():
            item = path / record["path"]
            if verify_hashes and not _matches_verified_file(
                item, size=int(record["bytes"]), sha256=str(record["sha256"])
            ):
                raise ValueError(f"store table hash mismatch: {name}")
        _validate_array_shapes(arrays, dates.size, len(isins))
        return cls(
            root=path,
            manifest=manifest,
            dates=dates,
            isins=isins,
            _arrays=arrays,
            access_ledger=None if write_verification else grant.ledger,
            _authorized_date_indices=(
                frozenset(range(dates.size))
                if write_verification
                else grant.date_indices
            ),
        )

    def authorized_for(self, date_indices: Sequence[int]) -> bool:
        """Return whether every requested row belongs to this store capability."""

        return set(int(value) for value in date_indices).issubset(
            self._authorized_date_indices
        )

    @property
    def array_names(self) -> frozenset[str]:
        """Names present in the immutable store, without exposing array handles."""

        return frozenset(self._arrays)

    def has_array(self, name: str) -> bool:
        return name in self._arrays

    def array_shape(self, name: str) -> tuple[int, ...]:
        """Return immutable shape metadata without returning the backing mmap."""

        try:
            shape = self.manifest["arrays"][name]["shape"]
        except KeyError as error:
            raise KeyError(f"v2 store does not contain {name}") from error
        return tuple(int(value) for value in shape)

    def array_dtype(self, name: str) -> np.dtype[np.generic]:
        """Return immutable dtype metadata without returning the backing mmap."""

        try:
            dtype = self.manifest["arrays"][name]["dtype"]
        except KeyError as error:
            raise KeyError(f"v2 store does not contain {name}") from error
        return np.dtype(dtype)

    def _checked_date_selector(
        self,
        selector: int | np.integer | slice | range | Sequence[int] | NDArray[np.integer],
    ) -> int | NDArray[np.int64]:
        date_count = int(self.dates.size)
        if isinstance(selector, (int, np.integer)) and not isinstance(
            selector, (bool, np.bool_)
        ):
            index = int(selector)
            if index < 0:
                raise IndexError("negative v2 store date indices are forbidden")
            if not 0 <= index < date_count:
                raise IndexError("v2 store date index is outside its axis")
            indices = np.asarray([index], dtype=np.int64)
            normalized: int | NDArray[np.int64] = index
        elif isinstance(selector, slice):
            if selector.step is not None and selector.step <= 0:
                raise ValueError("v2 store date slices require a positive step")
            if selector.start is not None and selector.start < 0:
                raise IndexError("negative v2 store date indices are forbidden")
            if selector.stop is not None and selector.stop < 0:
                raise IndexError("negative v2 store date indices are forbidden")
            indices = np.arange(date_count, dtype=np.int64)[selector]
            normalized = indices
        else:
            if isinstance(selector, range):
                indices = np.fromiter(selector, dtype=np.int64, count=len(selector))
            else:
                raw = np.asarray(selector)
                if (
                    raw.ndim != 1
                    or not np.issubdtype(raw.dtype, np.integer)
                    or np.issubdtype(raw.dtype, np.bool_)
                ):
                    raise TypeError(
                        "v2 store date selector must be a one-dimensional integer sequence"
                    )
                indices = raw.astype(np.int64, copy=False)
            if np.any(indices < 0):
                raise IndexError("negative v2 store date indices are forbidden")
            if np.any(indices >= date_count):
                raise IndexError("v2 store date index is outside its axis")
            normalized = indices
        if not self.authorized_for(indices):
            raise PermissionError(
                "v2 store read contains a date row outside the authorization grant"
            )
        return normalized

    def read(
        self,
        name: str,
        date_selector: (
            int
            | np.integer
            | slice
            | range
            | Sequence[int]
            | NDArray[np.integer]
        ),
    ) -> NDArray[np.generic]:
        """Copy authorized date rows without exposing the backing whole-store mmap."""

        try:
            array = self._arrays[name]
        except KeyError as error:
            raise KeyError(f"v2 store does not contain {name}") from error
        selector = self._checked_date_selector(date_selector)
        result = np.asarray(array[selector]).copy()
        mask_name = _TARGET_VALUE_MASKS.get(name)
        if name in _MULTI_HORIZON_TARGET_MASKS:
            return self._target_mask(name, selector)
        if mask_name is None:
            return result
        if mask_name in _MULTI_HORIZON_TARGET_MASKS:
            valid = self._target_mask(mask_name, selector)
        else:
            valid = np.asarray(self._arrays[mask_name][selector], dtype=np.bool_)
        if result.shape != valid.shape:
            raise ValueError(f"{name} and {mask_name} are misaligned")
        return np.where(valid, result, 0).astype(result.dtype, copy=False)

    def _target_mask(
        self, name: str, selector: int | slice | NDArray[np.int64]
    ) -> NDArray[np.bool_]:
        """Clip multi-horizon targets to endpoints inside this exact grant."""

        raw = np.asarray(self._arrays[name][selector], dtype=np.bool_).copy()
        scalar = raw.ndim == 2
        masks = raw[None, ...] if scalar else raw
        selected = np.atleast_1d(np.arange(self.dates.size, dtype=np.int64)[selector])
        if masks.shape[0] != selected.size or masks.shape[-1] != len(HORIZONS):
            raise ValueError(f"{name} has the wrong target axes")
        granted = self._authorized_date_indices
        for row, date_index in enumerate(selected):
            for horizon_index, horizon in enumerate(HORIZONS):
                if int(date_index) + horizon not in granted:
                    masks[row, :, horizon_index] = False
        return masks[0] if scalar else masks

    def read_table(
        self,
        name: str,
        date_selector: (
            int
            | np.integer
            | slice
            | range
            | Sequence[int]
            | NDArray[np.integer]
            | None
        ) = None,
    ) -> pl.DataFrame:
        """Read only the two runtime mapping tables within this capability.

        Audit and coverage tables remain immutable artifacts, but are not
        exposed through a date-bounded training/evaluation store handle.
        """

        try:
            record = self.manifest.get("tables", {})[name]
        except KeyError as error:
            raise KeyError(f"v2 store does not contain table {name}") from error
        if name == "v1_fast_isin_mapping":
            if date_selector is not None:
                raise ValueError("the static ISIN mapping has no date selector")
            return pl.read_parquet(self.root / record["path"])
        if name != "v1_fast_date_mapping":
            raise PermissionError(
                "audit tables are not exposed through a sealed store capability"
            )
        if date_selector is None:
            raise PermissionError("the fast date mapping requires authorized dates")
        indices = np.atleast_1d(self._checked_date_selector(date_selector))
        allowed = pl.Series(
            "trade_date",
            self.dates[indices].astype("datetime64[D]").astype(object).tolist(),
            dtype=pl.Date,
        )
        return pl.read_parquet(self.root / record["path"]).filter(
            pl.col("trade_date").is_in(allowed.implode())
        )

    def close(self) -> None:
        """Release mmap handles eagerly where NumPy exposes them."""

        for value in (*self._arrays.values(), self.dates):
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self._arrays.clear()


def open_store(
    root: Path,
    *,
    verify_hashes: bool = True,
) -> V2Store:
    return V2Store.open(root, verify_hashes=verify_hashes)


def _validated_store_indices(
    dates: NDArray[np.datetime64], date_indices: Sequence[int]
) -> NDArray[np.int64]:
    indices = np.asarray(date_indices)
    if (
        indices.ndim != 1
        or not indices.size
        or not np.issubdtype(indices.dtype, np.integer)
        or np.issubdtype(indices.dtype, np.bool_)
        or np.any(indices < 0)
        or np.any(indices >= dates.size)
    ):
        raise ValueError("requested store date indices are invalid")
    return indices.astype(np.int64, copy=False)


def open_store_for_dates(
    root: Path,
    date_indices: Sequence[int],
    *,
    purpose: AccessPurpose,
    registration_path: Path | None = None,
    preregistration_root: Path = PREREGISTRATION_ROOT,
    verify_hashes: bool = True,
) -> tuple[V2Store, AccessLedger]:
    """Authorize an exact date request before any store array is memory-mapped."""

    path = Path(root).resolve()
    dates = np.load(path / "date_index.npy", allow_pickle=False)
    indices = _validated_store_indices(dates, date_indices)
    requested = tuple(sorted(dates[indices].astype(object).tolist()))
    ledger = authorize_dates(
        requested,
        purpose=purpose,
        registration_path=registration_path,
        preregistration_root=preregistration_root,
    )
    grant = _StoreAccessGrant(
        root=path,
        date_indices=frozenset(int(value) for value in indices.tolist()),
        ledger=ledger,
    )
    return (
        V2Store._open_internal(path, verify_hashes=verify_hashes, grant=grant),
        ledger,
    )


def _per_sample_integer_parameter(
    value: int | Sequence[int],
    count: int,
    *,
    name: str,
) -> NDArray[np.int64]:
    if isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    ):
        return np.full(count, int(value), dtype=np.int64)
    raw = np.asarray(value)
    if (
        raw.shape != (count,)
        or not np.issubdtype(raw.dtype, np.integer)
        or np.issubdtype(raw.dtype, np.bool_)
    ):
        raise TypeError(f"{name} must be one integer per sample row")
    return raw.astype(np.int64, copy=False)


def open_store_for_samples(
    root: Path,
    date_indices: Sequence[int],
    *,
    purpose: AccessPurpose,
    history_lookbacks: int | Sequence[int],
    history_end_offsets: int | Sequence[int],
    registration_path: Path | None = None,
    preregistration_root: Path = PREREGISTRATION_ROOT,
    verify_hashes: bool = True,
) -> tuple[V2Store, AccessLedger]:
    """Authorize samples plus only their explicitly bounded causal histories.

    The returned ledger describes the sample rows alone. History rows are a
    derived capability: each must end at the sample row or its immediate prior
    session, and each lookback is one of the frozen slow/baseline spans. This
    permits deliberately unsampleable embargo dates to provide past context
    without making them training, selection, or evaluation samples.
    """

    path = Path(root).resolve()
    dates = np.load(path / "date_index.npy", allow_pickle=False)
    indices = _validated_store_indices(dates, date_indices)
    requested = tuple(sorted(dates[indices].astype(object).tolist()))
    ledger = authorize_dates(
        requested,
        purpose=purpose,
        registration_path=registration_path,
        preregistration_root=preregistration_root,
    )
    lookbacks = _per_sample_integer_parameter(
        history_lookbacks, len(indices), name="history_lookbacks"
    )
    offsets = _per_sample_integer_parameter(
        history_end_offsets, len(indices), name="history_end_offsets"
    )
    if not np.isin(lookbacks, (*ALLOWED_LOOKBACKS, _MAX_CAUSAL_HISTORY_ROWS)).all():
        raise ValueError(
            "causal history lookbacks must be a frozen slow or baseline span"
        )
    if not np.isin(offsets, (-1, 0)).all():
        raise ValueError("causal history may end only at t or t-1")
    ends = indices + offsets
    if np.any(ends < 0) or np.any(ends >= dates.size):
        raise ValueError("causal history endpoint is outside the store")
    authorized = set(int(value) for value in indices)
    for end, lookback in zip(ends.tolist(), lookbacks.tolist(), strict=True):
        authorized.update(
            range(max(0, int(end) - int(lookback) + 1), int(end) + 1)
        )
    grant = _StoreAccessGrant(
        root=path,
        date_indices=frozenset(authorized),
        ledger=ledger,
    )
    return (
        V2Store._open_internal(path, verify_hashes=verify_hashes, grant=grant),
        ledger,
    )
