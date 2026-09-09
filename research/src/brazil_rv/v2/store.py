from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from .contract import (
    ALLOWED_LOOKBACKS,
    DECISION_MINUTE_INDEX,
    HORIZONS,
    REGISTERED_PRIMARY_TARGET,
    REGISTERED_PRIMARY_TARGET_MASK,
    TARGET_NEUTRALIZATION_BETA_GROUPS,
    TARGET_NEUTRALIZATION_FEATURES,
    TARGET_NEUTRALIZATION_MIN_NONLINEAR_NAMES,
    TARGET_NEUTRALIZATION_VOL_GROUPS,
)
from .feature_spec import FeatureSpec, feature_schema_sha256
from .normalization import midrank_unit_interval
from .splits import (
    PREREGISTRATION_ROOT,
    AccessLedger,
    AccessPurpose,
    authorize_dates,
)

STORE_SCHEMA = "BRAZIL_RV_V2_DAILY_STORE_V3"
_SAFE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_WRITE_VERIFICATION = object()
_MAX_CAUSAL_HISTORY_ROWS = max(*ALLOWED_LOOKBACKS, 253)
_TARGET_VALUE_MASKS = {
    "target_primary": "target_valid",
    "target_normalized_residual": "target_valid",
    "target_shareholder_midrank": "target_shareholder_valid",
    "target_shareholder_simple_return": "target_shareholder_valid",
    "target_terminal_wealth": "target_shareholder_valid",
    "target_terminal_loss": "target_shareholder_valid",
    "target_price_midrank": "target_price_valid",
    "target_price_simple_return": "target_price_valid",
    "target_to_close": "target_to_close_valid",
    "target_to_close_normalized_residual": "target_to_close_valid",
    "target_to_close_raw_log_return": "target_to_close_valid",
}
_VIRTUAL_TARGET_VALUE_MASKS = {
    REGISTERED_PRIMARY_TARGET: REGISTERED_PRIMARY_TARGET_MASK,
}
_VIRTUAL_TARGETS = frozenset(
    (*_VIRTUAL_TARGET_VALUE_MASKS, *_VIRTUAL_TARGET_VALUE_MASKS.values())
)
_MULTI_HORIZON_TARGET_MASKS = frozenset(
    ("target_valid", "target_shareholder_valid", "target_price_valid")
)
_DATE_ONLY_ARRAYS: frozenset[str] = frozenset()
_DATE_HORIZON_ARRAYS = frozenset(("target_normalized_cross_section_valid",))
_DATE_COMMON_STATE_ARRAYS = frozenset(
    ("common_state_diagnostic_values", "common_state_diagnostic_valid")
)
_SPARSE_FAST_ARRAYS = frozenset(
    {
        "fast_patch_values",
        "fast_patch_valid",
        "fast_patch_mask",
        "fast_last_price_age_minutes",
        "fast_last_price_age_valid",
    }
)
_FEATURE_SPEC_KEYS = frozenset(item.name for item in dataclass_fields(FeatureSpec))
_NON_MODEL_FEATURE_NAME_GROUPS = frozenset(("common_state_diagnostic", "horizons"))
_PRIMARY_MODEL_FEATURE_GROUPS = ("slow", "intraday", "native_fast")


def characteristic_neutral_targets(
    simple_returns: NDArray[np.floating],
    scaled_target_valid: NDArray[np.bool_],
    sigma_asof: NDArray[np.floating],
    characteristics: NDArray[np.floating],
    characteristic_valid: NDArray[np.bool_],
    *,
    horizons: Sequence[int] = HORIZONS,
    clip: float = 5.0,
    minimum_names: int = 20,
    nonlinear_minimum_names: int = TARGET_NEUTRALIZATION_MIN_NONLINEAR_NAMES,
    fallback_flags: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Residualize the scaled return target under the registered rev-4 view.

    ``characteristics`` are the store's already rank-Gaussianized decision-row
    values for lagged volatility, beta, and log ADV.  At sufficient support the
    design uses ten volatility groups, five beta groups, and linear log ADV.
    Smaller supported cross-sections retain the registered rev-3 linear
    fallback.  Work is deliberately row-wise so the virtual view never
    materializes another full-store panel.
    """

    returns = np.asarray(simple_returns, dtype=np.float64)
    target_valid = np.asarray(scaled_target_valid, dtype=np.bool_)
    sigma = np.asarray(sigma_asof, dtype=np.float64)
    z = np.asarray(characteristics, dtype=np.float64)
    z_valid = np.asarray(characteristic_valid, dtype=np.bool_)
    if returns.ndim != 3 or target_valid.shape != returns.shape:
        raise ValueError("neutral-target returns and validity must align")
    if sigma.shape != returns.shape[:2] or z.shape != (*returns.shape[:2], 3):
        raise ValueError("neutral-target risk characteristics are misaligned")
    if z_valid.shape != returns.shape[:2]:
        raise ValueError("neutral-target characteristic validity is misaligned")
    if returns.shape[-1] != len(horizons):
        raise ValueError("neutral-target horizon axis is misaligned")
    if (
        clip <= 0.0
        or not np.isfinite(clip)
        or minimum_names < 1
        or nonlinear_minimum_names < minimum_names
    ):
        raise ValueError("neutral-target clip and minimum support must be positive")
    if fallback_flags is not None:
        fallback = np.asarray(fallback_flags)
        if fallback.dtype != np.bool_ or fallback.shape != (
            returns.shape[0],
            returns.shape[-1],
        ):
            raise ValueError("neutral-target fallback flags are misaligned")

    def equal_count_groups(values: NDArray[np.float64], count: int) -> NDArray[np.int64]:
        order = np.argsort(values, kind="stable")
        groups = np.empty(values.size, dtype=np.int64)
        groups[order] = np.minimum(
            np.arange(values.size, dtype=np.int64) * count // values.size,
            count - 1,
        )
        return groups

    output = np.zeros(returns.shape, dtype=np.float32)
    output_valid = np.zeros(returns.shape, dtype=np.bool_)
    for day in range(returns.shape[0]):
        for horizon_index, horizon in enumerate(horizons):
            population = (
                target_valid[day, :, horizon_index]
                & z_valid[day]
                & np.isfinite(returns[day, :, horizon_index])
                & np.isfinite(sigma[day])
                & (sigma[day] > 1e-8)
            )
            if int(population.sum()) < minimum_names:
                continue
            names = np.flatnonzero(population)
            realized = returns[day, names, horizon_index]
            y = np.clip(
                (realized - np.median(realized))
                / (sigma[day, names] * np.sqrt(float(horizon))),
                -clip,
                clip,
            )
            if names.size < nonlinear_minimum_names:
                design = np.column_stack((np.ones(names.size), z[day, names]))
                if fallback_flags is not None:
                    fallback_flags[day, horizon_index] = True
            else:
                vol_group = equal_count_groups(
                    z[day, names, 0], TARGET_NEUTRALIZATION_VOL_GROUPS
                )
                beta_group = equal_count_groups(
                    z[day, names, 1], TARGET_NEUTRALIZATION_BETA_GROUPS
                )
                design = np.column_stack(
                    (
                        np.eye(TARGET_NEUTRALIZATION_VOL_GROUPS, dtype=np.float64)[
                            vol_group
                        ],
                        np.eye(TARGET_NEUTRALIZATION_BETA_GROUPS, dtype=np.float64)[
                            beta_group
                        ],
                        z[day, names, 2],
                    )
                )
            coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
            residual = y - design @ coefficients
            tolerance = (
                128.0
                * np.finfo(np.float64).eps
                * max(1.0, float(np.max(np.abs(y), initial=0.0)))
            )
            if float(np.max(np.abs(residual), initial=0.0)) <= tolerance:
                residual.fill(0.0)
            else:
                # LAPACK implementations may leave different machine-epsilon
                # noise among mathematically tied projected residuals.  Exact
                # ranking would amplify that irrelevant noise into a material
                # characteristic exposure.  Canonicalize only ties within a
                # bound far below the float32 target's stored precision.
                tie_tolerance = 512.0 * np.finfo(np.float64).eps * max(
                    1.0, float(np.max(np.abs(y), initial=0.0))
                )
                order = np.argsort(residual, kind="stable")
                sorted_residual = residual[order]
                start = 0
                for end in range(1, names.size + 1):
                    if (
                        end == names.size
                        or sorted_residual[end] - sorted_residual[start]
                        > tie_tolerance
                    ):
                        residual[order[start:end]] = sorted_residual[start]
                        start = end
            output[day, names, horizon_index] = midrank_unit_interval(residual)
            output_valid[day, names, horizon_index] = True
    return output, output_valid


def _validated_feature_schema_sha256(
    metadata: Mapping[str, object],
    feature_names: Mapping[str, Sequence[str]],
) -> str:
    """Validate and hash the complete ordered FeatureSpec contract."""

    if not isinstance(metadata, Mapping) or not isinstance(feature_names, Mapping):
        raise ValueError("current v2 stores require feature schema mappings")
    schema = metadata.get("feature_schema")
    if not isinstance(schema, Mapping):
        raise ValueError("current v2 stores require metadata.feature_schema")
    raw_specs = schema.get("specifications")
    if not isinstance(raw_specs, Sequence) or isinstance(raw_specs, (str, bytes)):
        raise ValueError("current v2 stores require FeatureSpec specifications")
    specs: list[FeatureSpec] = []
    for raw_spec in raw_specs:
        if not isinstance(raw_spec, Mapping) or set(raw_spec) != _FEATURE_SPEC_KEYS:
            raise ValueError("FeatureSpec specification fields are incomplete")
        try:
            specs.append(FeatureSpec(**dict(raw_spec)))
        except (TypeError, ValueError) as error:
            raise ValueError("invalid FeatureSpec specification") from error

    declared_model_groups = set(feature_names) - _NON_MODEL_FEATURE_NAME_GROUPS
    unknown_groups = (
        declared_model_groups
        - set(_PRIMARY_MODEL_FEATURE_GROUPS)
        - {key for key in declared_model_groups if key.startswith("sidecar_")}
    )
    if unknown_groups:
        raise ValueError(f"unregistered feature groups: {sorted(unknown_groups)}")
    model_groups = tuple(
        key for key in _PRIMARY_MODEL_FEATURE_GROUPS if key in declared_model_groups
    ) + tuple(
        sorted(key for key in declared_model_groups if key.startswith("sidecar_"))
    )
    ordered_names = tuple(
        str(name) for key in model_groups for name in feature_names[key]
    )
    if tuple(spec.name for spec in specs) != ordered_names:
        raise ValueError(
            "FeatureSpec order does not match ordered feature names: "
            f"specifications={tuple(spec.name for spec in specs)!r}; "
            f"feature_names={ordered_names!r}"
        )
    if tuple(dict.fromkeys(spec.family for spec in specs)) != model_groups:
        raise ValueError("FeatureSpec families do not match ordered feature groups")

    declared_sha = schema.get("sha256")
    if (
        not isinstance(declared_sha, str)
        or len(declared_sha) != 64
        or any(character not in "0123456789abcdef" for character in declared_sha)
    ):
        raise ValueError("metadata feature-schema SHA-256 is malformed")
    calculated_sha = feature_schema_sha256(specs)
    if declared_sha != calculated_sha:
        raise ValueError("metadata feature-schema SHA-256 mismatch")
    return calculated_sha


def peak_rss_bytes() -> int:
    """Return this process's peak resident set size in bytes."""

    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.argtypes = []
        get_current_process.restype = wintypes.HANDLE
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        get_process_memory_info.restype = wintypes.BOOL
        if not get_process_memory_info(
            get_current_process(), ctypes.byref(counters), counters.cb
        ):
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)

    import resource

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _raw_memory_status_bytes() -> dict[str, int | None]:
    """Read one system memory snapshot without a third-party dependency."""

    if sys.platform == "win32":
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return {
            "total_physical_memory_bytes": int(status.ullTotalPhys),
            "available_physical_memory_bytes": int(status.ullAvailPhys),
            "commit_limit_bytes": int(status.ullTotalPageFile),
            "available_commit_memory_bytes": int(status.ullAvailPageFile),
        }

    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    total_pages = int(os.sysconf("SC_PHYS_PAGES"))
    return {
        "total_physical_memory_bytes": page_size * total_pages,
        "available_physical_memory_bytes": page_size * available_pages,
        "commit_limit_bytes": None,
        "available_commit_memory_bytes": None,
    }


def available_memory_status_bytes() -> dict[str, int | str | None]:
    """Return memory headroom used to admit a store build.

    Windows allocations consume system commit as well as resident memory.  A
    build is therefore admitted against the smaller of available physical and
    available commit memory.  Other supported platforms retain the physical
    headroom rule because they do not expose an equivalent hard commit budget
    through the standard library.
    """

    status: dict[str, int | str | None] = dict(_raw_memory_status_bytes())
    available_physical = int(status["available_physical_memory_bytes"])
    available_commit = status["available_commit_memory_bytes"]
    if available_commit is None:
        status["available_build_memory_bytes"] = available_physical
        status["admission_basis"] = "available_physical_memory"
    else:
        status["available_build_memory_bytes"] = min(
            available_physical, int(available_commit)
        )
        status["admission_basis"] = (
            "minimum_of_available_physical_and_available_commit_memory"
        )
    return status


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
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _axis_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(values), ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _validate_axes(dates: NDArray[np.datetime64], isins: tuple[str, ...]) -> None:
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
        if name in _DATE_ONLY_ARRAYS:
            if value.shape != (date_count,):
                raise ValueError(f"{name} must have the [date] axis; got {value.shape}")
            continue
        if name in _DATE_HORIZON_ARRAYS:
            if value.shape != (date_count, len(HORIZONS)):
                raise ValueError(
                    f"{name} must have the [date, horizon] axes; got {value.shape}"
                )
            if value.dtype != np.bool_:
                raise ValueError(f"mask array must have boolean dtype: {name}")
            continue
        if name in _DATE_COMMON_STATE_ARRAYS:
            if value.ndim != 2 or value.shape[0] != date_count:
                raise ValueError(
                    f"{name} must have the [date, common-state] axes; got {value.shape}"
                )
            expected_dtype = (
                np.bool_ if name.endswith("_valid") else np.dtype(np.float32)
            )
            if value.dtype != expected_dtype:
                raise ValueError(f"unexpected common-state dtype: {name}")
            continue
        if name in _SPARSE_FAST_ARRAYS:
            if value.ndim < 3 or value.shape[0] != date_count:
                raise ValueError(
                    f"{name} must begin with the [date, fast-security, patch] "
                    f"axes; got {value.shape}"
                )
            if value.dtype == object:
                raise ValueError(f"object array is forbidden: {name}")
            if name.endswith("_valid") or name.endswith("_mask"):
                if value.dtype != np.bool_:
                    raise ValueError(f"mask array must have boolean dtype: {name}")
            continue
        if value.ndim < 2 or value.shape[:2] != (date_count, isin_count):
            raise ValueError(
                f"{name} must begin with the [date, ISIN] axes; got {value.shape}"
            )
        if value.dtype == object:
            raise ValueError(f"object array is forbidden: {name}")
        if (
            name.endswith("_valid")
            or name.endswith("_mask")
            or name
            in {
                "active",
                "observed",
                "fast_present",
                "action_session_resolved",
                "action_has_action",
                "target_terminal_loss",
            }
        ):
            if value.dtype != np.bool_:
                raise ValueError(f"mask array must have boolean dtype: {name}")
    feature_families = (
        ("slow_values", "slow_valid", "slow_age_sessions"),
        ("intraday_values", "intraday_valid", "intraday_age_sessions"),
        *tuple(
            (
                name,
                f"{name.removesuffix('_values')}_valid",
                f"{name.removesuffix('_values')}_age_sessions",
            )
            for name in arrays
            if name.startswith("sidecar_") and name.endswith("_values")
        ),
    )
    common_state_present = tuple(name in arrays for name in _DATE_COMMON_STATE_ARRAYS)
    if any(common_state_present) and not all(common_state_present):
        raise ValueError(
            "common-state diagnostic values and validity must be stored together"
        )
    if all(common_state_present) and (
        arrays["common_state_diagnostic_values"].shape
        != arrays["common_state_diagnostic_valid"].shape
    ):
        raise ValueError("common-state diagnostic values and validity are misaligned")
    declared_feature_arrays = {
        array_name for family in feature_families for array_name in family
    }
    orphan_feature_arrays = {
        name
        for name in arrays
        if (
            name
            in {
                "slow_valid",
                "slow_age_sessions",
                "intraday_valid",
                "intraday_age_sessions",
            }
            or (
                name.startswith("sidecar_")
                and (name.endswith("_valid") or name.endswith("_age_sessions"))
            )
        )
        and name not in declared_feature_arrays
    }
    if orphan_feature_arrays:
        raise ValueError(
            "feature validity/age arrays lack their value array: "
            f"{sorted(orphan_feature_arrays)}"
        )
    for values_name, mask_name, age_name in feature_families:
        present = tuple(name in arrays for name in (values_name, mask_name, age_name))
        if any(present) and not all(present):
            raise ValueError(
                f"{values_name}, {mask_name}, and {age_name} must be stored together"
            )
        if not all(present):
            continue
        shape = arrays[values_name].shape
        if arrays[mask_name].shape != shape or arrays[age_name].shape != shape:
            raise ValueError(
                f"{values_name}, {mask_name}, and {age_name} are misaligned"
            )
        ages = np.asarray(arrays[age_name])
        if ages.dtype != np.float32:
            raise ValueError(f"feature age array must have float32 dtype: {age_name}")
        for start in range(0, date_count, 32):
            block = ages[start : start + 32]
            valid_block = np.asarray(arrays[mask_name][start : start + 32])
            if (
                not np.isfinite(block).all()
                or np.any(block < -1.0)
                or np.any(valid_block & (block < 0.0))
            ):
                raise ValueError(
                    f"{age_name} must contain finite last-observation ages or -1, "
                    "and every valid feature must have a known age"
                )
    paired = (
        ("fast_patch_values", "fast_patch_valid"),
        ("target_primary", "target_valid"),
        ("target_shareholder_midrank", "target_shareholder_valid"),
        ("target_price_midrank", "target_price_valid"),
        ("target_to_close", "target_to_close_valid"),
    )
    for values_name, mask_name in paired:
        if (values_name in arrays) != (mask_name in arrays):
            raise ValueError(f"{values_name} and {mask_name} must be stored together")
        if (
            values_name in arrays
            and arrays[values_name].shape != arrays[mask_name].shape
        ):
            raise ValueError(f"{values_name} and {mask_name} are misaligned")
    if "slow_values" in arrays:
        timestep_valid = arrays.get("slow_timestep_valid")
        if timestep_valid is None:
            raise ValueError("slow values require slow_timestep_valid")
        if (
            timestep_valid.shape != arrays["slow_values"].shape[:2]
            or timestep_valid.dtype != np.bool_
        ):
            raise ValueError("slow_timestep_valid must be boolean [date, ISIN]")
        if np.any(arrays["slow_valid"] & ~timestep_valid[..., None]):
            raise ValueError("slow features cannot be valid outside real timesteps")
    if "fast_patch_values" in arrays:
        fast_shape = arrays["fast_patch_values"].shape
        if len(fast_shape) != 4 or fast_shape[-1] != 7:
            raise ValueError(
                "native fast values must have shape [date, fast, patch, 7]"
            )
        if "fast_patch_mask" not in arrays:
            raise ValueError("native fast values require fast_patch_mask")
        if arrays["fast_patch_mask"].shape != fast_shape[:-1]:
            raise ValueError("native fast patch mask is misaligned")
        for values_name, mask_name in (
            ("fast_last_price_age_minutes", "fast_last_price_age_valid"),
        ):
            if (values_name in arrays) != (mask_name in arrays):
                raise ValueError(
                    f"{values_name} and {mask_name} must be stored together"
                )
            if values_name in arrays and arrays[values_name].shape != fast_shape[:-1]:
                raise ValueError(
                    f"{values_name} is misaligned with native fast patches"
                )
    for values_name, mask_name in (
        ("target_shareholder_simple_return", "target_shareholder_valid"),
        ("target_terminal_wealth", "target_shareholder_valid"),
        ("target_terminal_loss", "target_shareholder_valid"),
        ("target_price_simple_return", "target_price_valid"),
    ):
        if values_name not in arrays:
            continue
        if mask_name not in arrays:
            raise ValueError(f"{values_name} requires {mask_name}")
        if arrays[values_name].shape != arrays[mask_name].shape:
            raise ValueError(f"{values_name} and {mask_name} are misaligned")
    action_names = {
        "action_shares_per_prior_share",
        "action_cash_per_prior_share",
        "action_session_resolved",
        "action_has_action",
        "action_successor_index",
        "action_payment_session",
    }
    present_actions = action_names.intersection(arrays)
    if present_actions and present_actions != action_names:
        raise ValueError(
            "the canonical action arrays must be stored as one complete contract"
        )
    if present_actions:
        q = np.asarray(arrays["action_shares_per_prior_share"])
        d = np.asarray(arrays["action_cash_per_prior_share"])
        resolved = np.asarray(arrays["action_session_resolved"], dtype=np.bool_)
        has_action = np.asarray(arrays["action_has_action"], dtype=np.bool_)
        successor = np.asarray(arrays["action_successor_index"])
        payment = np.asarray(arrays["action_payment_session"])
        if (
            not np.issubdtype(q.dtype, np.floating)
            or not np.issubdtype(d.dtype, np.floating)
            or not np.isfinite(q).all()
            or not np.isfinite(d).all()
            or (q < 0).any()
            or (d < 0).any()
        ):
            raise ValueError("canonical action q/d arrays are invalid")
        if not np.issubdtype(successor.dtype, np.integer) or (
            (successor < 0).any() or (successor >= isin_count).any()
        ):
            raise ValueError("canonical action successor indices are invalid")
        if (
            not np.issubdtype(payment.dtype, np.integer)
            or (payment < -1).any()
            or (payment > date_count).any()
        ):
            raise ValueError("canonical action payment sessions are invalid")
        identity_successor = np.broadcast_to(
            np.arange(isin_count, dtype=np.int64), (date_count, isin_count)
        )
        silent_terms = ~has_action & (
            (q != 1.0) | (d != 0.0) | (successor != identity_successor)
        )
        if silent_terms.any():
            raise ValueError("economic action terms require has_action=true")
        event_session = np.broadcast_to(
            np.arange(date_count, dtype=np.int64)[:, None],
            (date_count, isin_count),
        )
        if ((payment >= 0) & (payment < event_session)).any():
            raise ValueError(
                "action payment sessions cannot precede their action session"
            )
        invalid_payment_binding = (payment >= 0) & (
            ~has_action | ~resolved | (d == 0.0)
        )
        if invalid_payment_binding.any():
            raise ValueError(
                "payment sessions require a resolved cash-bearing action on the same cell"
            )
    if "prior_reference_close" in arrays:
        prior_reference = np.asarray(arrays["prior_reference_close"])
        if (
            not np.issubdtype(prior_reference.dtype, np.floating)
            or np.isinf(prior_reference).any()
            or np.any(np.isfinite(prior_reference) & (prior_reference <= 0.0))
        ):
            raise ValueError(
                "prior_reference_close must contain positive prices or NaN"
            )
    if "audit_eventual_survives_to_final_year" in arrays:
        eventual_survival = np.asarray(arrays["audit_eventual_survives_to_final_year"])
        if eventual_survival.dtype != np.bool_:
            raise ValueError("audit_eventual_survives_to_final_year must be boolean")


def _validate_native_fast_mapping(
    frame: pl.DataFrame,
    *,
    fast_count: int,
    isins: Sequence[str],
) -> None:
    required = {"fast_index", "store_name_index", "isin"}
    if not required.issubset(frame.columns):
        raise ValueError(
            "native fast security mapping columns missing: "
            f"{sorted(required - set(frame.columns))}"
        )
    ordered = frame.select(required).sort("fast_index")
    fast_indices = ordered.get_column("fast_index").cast(pl.Int64).to_list()
    store_indices = ordered.get_column("store_name_index").cast(pl.Int64).to_list()
    mapped_isins = ordered.get_column("isin").cast(pl.String).to_list()
    if fast_indices != list(range(fast_count)):
        raise ValueError("native fast indices must be contiguous and complete")
    if len(set(store_indices)) != fast_count or any(
        index < 0 or index >= len(isins) for index in store_indices
    ):
        raise ValueError("native fast store-name mapping is not one-to-one")
    if any(isins[index] != isin for index, isin in zip(store_indices, mapped_isins)):
        raise ValueError("native fast ISIN identities disagree with the store axis")


def close_memmap(array: NDArray[np.generic]) -> None:
    """Flush and close a NumPy memmap, including through ndarray views."""

    candidate: object = array
    seen: set[int] = set()
    while isinstance(candidate, np.ndarray) and id(candidate) not in seen:
        seen.add(id(candidate))
        if isinstance(candidate, np.memmap):
            mapping = getattr(candidate, "_mmap", None)
            if mapping is not None and not mapping.closed:
                candidate.flush()
                mapping.close()
            return
        candidate = candidate.base


class StoreStaging:
    """Build final store arrays directly inside one atomic staging directory."""

    def __init__(
        self,
        output_dir: Path,
        *,
        dates: Sequence[object],
        isins: Sequence[str],
    ) -> None:
        self.output = Path(output_dir).resolve()
        if self.output.exists():
            raise FileExistsError(self.output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.dates = np.asarray(dates, dtype="datetime64[D]")
        self.isins = tuple(str(value) for value in isins)
        _validate_axes(self.dates, self.isins)
        self.staging = Path(
            tempfile.mkdtemp(
                prefix=f".{self.output.name}.building-", dir=self.output.parent
            )
        )
        np.save(self.staging / "date_index.npy", self.dates, allow_pickle=False)
        np.save(
            self.staging / "isin_index.npy",
            np.asarray(self.isins, dtype=np.str_),
            allow_pickle=False,
        )
        self._arrays: dict[str, tuple[tuple[int, ...], np.dtype[np.generic]]] = {}
        self._sealed = False

    def __enter__(self) -> StoreStaging:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        if not self._sealed:
            shutil.rmtree(self.staging, ignore_errors=True)

    def array_path(self, name: str) -> Path:
        if not _SAFE_NAME.fullmatch(name):
            raise ValueError(f"unsafe array name: {name}")
        return self.staging / f"{name}.npy"

    def create_array(
        self,
        name: str,
        shape: Sequence[int],
        dtype: np.dtype[np.generic] | type[np.generic],
        *,
        fill: float | bool | None = None,
        chunk_rows: int = 64,
    ) -> np.memmap:
        normalized_shape = tuple(int(value) for value in shape)
        valid_shape = (
            normalized_shape == (self.dates.size,)
            if name in _DATE_ONLY_ARRAYS
            else normalized_shape == (self.dates.size, len(HORIZONS))
            if name in _DATE_HORIZON_ARRAYS
            else len(normalized_shape) == 2
            and normalized_shape[0] == self.dates.size
            and normalized_shape[1] > 0
            if name in _DATE_COMMON_STATE_ARRAYS
            else len(normalized_shape) >= 3 and normalized_shape[0] == self.dates.size
            if name in _SPARSE_FAST_ARRAYS
            else len(normalized_shape) >= 2
            and normalized_shape[:2] == (self.dates.size, len(self.isins))
        )
        if not valid_shape:
            raise ValueError(
                "store arrays must begin with the output [date, ISIN] axes"
            )
        if name in self._arrays or self.array_path(name).exists():
            raise ValueError(f"duplicate store array: {name}")
        normalized_dtype = np.dtype(dtype)
        output = np.lib.format.open_memmap(
            self.array_path(name),
            mode="w+",
            dtype=normalized_dtype,
            shape=normalized_shape,
        )
        if fill is not None:
            if chunk_rows <= 0:
                raise ValueError("chunk_rows must be positive")
            for start in range(0, normalized_shape[0], chunk_rows):
                output[start : start + chunk_rows] = fill
        self._arrays[name] = (normalized_shape, normalized_dtype)
        return output

    def write_array(
        self,
        name: str,
        values: NDArray[np.generic],
        *,
        row_indices: Sequence[int] | NDArray[np.integer] | None = None,
        dtype: np.dtype[np.generic] | type[np.generic] | None = None,
        chunk_rows: int = 64,
    ) -> None:
        source = np.asarray(values)
        valid_source = (
            source.ndim == 1
            if name in _DATE_ONLY_ARRAYS
            else source.ndim == 2 and source.shape[1] == len(HORIZONS)
            if name in _DATE_HORIZON_ARRAYS
            else source.ndim == 2 and source.shape[1] > 0
            if name in _DATE_COMMON_STATE_ARRAYS
            else source.ndim >= 3
            if name in _SPARSE_FAST_ARRAYS
            else source.ndim >= 2 and source.shape[1] == len(self.isins)
        )
        if not valid_source:
            raise ValueError(
                "store source arrays must begin with the [date, ISIN] axes"
            )
        if chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")
        if row_indices is None:
            if source.shape[0] != self.dates.size:
                raise ValueError(
                    "store source array does not match the output date axis"
                )
            rows: NDArray[np.int64] | None = None
        else:
            raw_rows = np.asarray(row_indices)
            if (
                raw_rows.ndim != 1
                or not np.issubdtype(raw_rows.dtype, np.integer)
                or np.issubdtype(raw_rows.dtype, np.bool_)
            ):
                raise TypeError(
                    "row_indices must be a one-dimensional integer sequence"
                )
            rows = raw_rows.astype(np.int64, copy=False)
            if (
                rows.size != self.dates.size
                or np.any(rows < 0)
                or np.any(rows >= source.shape[0])
                or (rows.size > 1 and np.any(np.diff(rows) <= 0))
            ):
                raise ValueError(
                    "row_indices must select one unique increasing source row per output date"
                )
        destination = self.create_array(
            name,
            (self.dates.size, *source.shape[1:]),
            source.dtype if dtype is None else dtype,
        )
        try:
            for start in range(0, self.dates.size, chunk_rows):
                stop = min(start + chunk_rows, self.dates.size)
                source_rows: slice | NDArray[np.int64] = (
                    slice(start, stop) if rows is None else rows[start:stop]
                )
                destination[start:stop] = source[source_rows]
        finally:
            close_memmap(destination)

    def open_array(self, name: str, *, mode: str = "r") -> np.memmap:
        if name not in self._arrays:
            raise KeyError(name)
        return np.load(self.array_path(name), mmap_mode=mode, allow_pickle=False)

    def create_scratch_array(
        self,
        name: str,
        shape: Sequence[int],
        dtype: np.dtype[np.generic] | type[np.generic],
    ) -> np.memmap:
        if not _SAFE_NAME.fullmatch(name):
            raise ValueError(f"unsafe scratch-array name: {name}")
        path = self.staging / f".scratch_{name}.npy"
        if path.exists():
            raise ValueError(f"duplicate scratch array: {name}")
        return np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.dtype(dtype),
            shape=tuple(int(value) for value in shape),
        )

    def remove_scratch_array(self, name: str) -> None:
        path = self.staging / f".scratch_{name}.npy"
        path.unlink()

    def seal(
        self,
        *,
        feature_names: Mapping[str, Sequence[str]] | None = None,
        sources: Sequence[Mapping[str, object]] = (),
        metadata: Mapping[str, object] | None = None,
        tables: Mapping[str, pl.DataFrame] | None = None,
        maximum_peak_rss_bytes: int | None = None,
    ) -> Path:
        arrays = {name: self.open_array(name) for name in sorted(self._arrays)}
        try:
            _validate_array_shapes(arrays, self.dates.size, len(self.isins))
            if "fast_patch_values" in arrays:
                mapping = (tables or {}).get("native_fast_security_mapping")
                if mapping is None:
                    raise ValueError(
                        "native fast arrays require native_fast_security_mapping"
                    )
                _validate_native_fast_mapping(
                    mapping,
                    fast_count=arrays["fast_patch_values"].shape[1],
                    isins=self.isins,
                )
        finally:
            for value in arrays.values():
                close_memmap(value)
        inventory: dict[str, dict[str, object]] = {}
        for name, (shape, dtype) in sorted(self._arrays.items()):
            path = self.array_path(name)
            inventory[name] = {
                "path": path.name,
                "shape": list(shape),
                "dtype": dtype.str,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        index_inventory = {}
        for name in ("date_index.npy", "isin_index.npy"):
            path = self.staging / name
            index_inventory[name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        table_inventory: dict[str, dict[str, object]] = {}
        for name, frame in sorted((tables or {}).items()):
            if not _SAFE_NAME.fullmatch(name):
                raise ValueError(f"unsafe table name: {name}")
            path = self.staging / f"{name}.parquet"
            frame.write_parquet(path)
            table_inventory[name] = {
                "path": path.name,
                "rows": frame.height,
                "columns": frame.columns,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        ordered_feature_names = {
            key: list(value) for key, value in (feature_names or {}).items()
        }
        metadata_payload = dict(metadata or {})
        if maximum_peak_rss_bytes is not None:
            measured_peak = peak_rss_bytes()
            if measured_peak > maximum_peak_rss_bytes:
                raise MemoryError(
                    f"store publication exceeded peak RSS bound: {measured_peak} > {maximum_peak_rss_bytes}"
                )
            metadata_payload["build_peak_rss_bytes"] = measured_peak
            metadata_payload["build_peak_rss_gib"] = measured_peak / 1024**3
        feature_schema_identity = _validated_feature_schema_sha256(
            metadata_payload, ordered_feature_names
        )
        manifest: dict[str, Any] = {
            "schema": STORE_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "axes": {
                "date_count": int(self.dates.size),
                "date_start": str(self.dates[0]),
                "date_end": str(self.dates[-1]),
                "date_identity_sha256": _axis_sha256(
                    [str(value) for value in self.dates]
                ),
                "isin_count": len(self.isins),
                "isin_identity_sha256": _axis_sha256(self.isins),
                "horizons": list(HORIZONS),
                "decision_minute_index": DECISION_MINUTE_INDEX,
            },
            "indices": index_inventory,
            "arrays": inventory,
            "tables": table_inventory,
            "feature_names": ordered_feature_names,
            "feature_schema_sha256": feature_schema_identity,
            "feature_schema_source": "metadata_feature_specifications",
            "sources": [dict(value) for value in sources],
            "metadata": metadata_payload,
            "official_validation_accessed": False,
            "test_accessed": False,
        }
        manifest_path = self.staging / "manifest.json"
        manifest_path.write_bytes(_json_bytes(manifest))
        (self.staging / "manifest.sha256").write_text(
            f"{sha256_file(manifest_path)}  manifest.json\n", encoding="ascii"
        )
        os.replace(self.staging, self.output)
        self._sealed = True
        opened = V2Store._open_internal(
            self.output, verify_hashes=True, grant=_WRITE_VERIFICATION
        )
        opened.close()
        return self.output


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

    date_axis = np.asarray(dates, dtype="datetime64[D]")
    isin_axis = tuple(str(value) for value in isins)
    _validate_axes(date_axis, isin_axis)
    materialized = {name: np.asarray(value) for name, value in arrays.items()}
    if any(
        value.ndim < (1 if name in _DATE_ONLY_ARRAYS else 2)
        for name, value in materialized.items()
    ):
        raise ValueError("store arrays must begin with their registered axes")
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
            selected_rows = slice(int(selected_rows[0]), int(selected_rows[-1]) + 1)
    elif source_date_count != date_axis.size:
        raise ValueError("store arrays do not match the output date axis")
    with StoreStaging(output_dir, dates=date_axis, isins=isin_axis) as staging:
        for name, source_value in sorted(materialized.items()):
            staging.write_array(
                name,
                source_value,
                row_indices=(
                    None
                    if selected_rows is None
                    else np.arange(source_date_count, dtype=np.int64)[selected_rows]
                ),
            )
        return staging.seal(
            feature_names=feature_names,
            sources=sources,
            metadata=metadata,
            tables=tables,
        )


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
            "v2 feature and target arrays must be opened through open_store_for_dates"
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
        schema = manifest.get("schema")
        if schema != STORE_SCHEMA:
            raise ValueError("not a current v2 daily store")
        feature_schema_identity = _validated_feature_schema_sha256(
            manifest.get("metadata", {}), manifest.get("feature_names", {})
        )
        if (
            manifest.get("feature_schema_sha256") != feature_schema_identity
            or manifest.get("feature_schema_source")
            != "metadata_feature_specifications"
        ):
            raise ValueError("store FeatureSpec identity mismatch")
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
            if (
                list(value.shape) != record["shape"]
                or value.dtype.str != record["dtype"]
            ):
                raise ValueError(f"store array schema mismatch: {name}")
            arrays[name] = value
        for name, record in manifest.get("tables", {}).items():
            item = path / record["path"]
            if verify_hashes and not _matches_verified_file(
                item, size=int(record["bytes"]), sha256=str(record["sha256"])
            ):
                raise ValueError(f"store table hash mismatch: {name}")
        # The immutable writer validated payload values before sealing them.
        # Hash and header checks above preserve that contract without decoding
        # ungranted rows (including held-out dates) during a reader open.
        if "fast_patch_values" in arrays:
            try:
                mapping_record = manifest["tables"]["native_fast_security_mapping"]
            except KeyError as error:
                raise ValueError(
                    "native fast arrays lack their security mapping"
                ) from error
            _validate_native_fast_mapping(
                pl.read_parquet(path / mapping_record["path"]),
                fast_count=arrays["fast_patch_values"].shape[1],
                isins=isins,
            )
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

        virtual = _VIRTUAL_TARGETS if self._neutral_target_available() else ()
        return frozenset((*self._arrays, *virtual))

    def has_array(self, name: str) -> bool:
        return name in self._arrays or (
            name in _VIRTUAL_TARGETS and self._neutral_target_available()
        )

    def _neutral_target_available(self) -> bool:
        required = {
            "target_valid",
            "target_shareholder_valid",
            "target_shareholder_simple_return",
            "target_scale_sigma",
            "slow_values",
            "slow_valid",
        }
        return required.issubset(self._arrays)

    def array_shape(self, name: str) -> tuple[int, ...]:
        """Return immutable shape metadata without returning the backing mmap."""

        if name in _VIRTUAL_TARGETS:
            name = "target_valid"
        try:
            shape = self.manifest["arrays"][name]["shape"]
        except KeyError as error:
            raise KeyError(f"v2 store does not contain {name}") from error
        return tuple(int(value) for value in shape)

    def array_dtype(self, name: str) -> np.dtype[np.generic]:
        """Return immutable dtype metadata without returning the backing mmap."""

        if name == REGISTERED_PRIMARY_TARGET:
            return np.dtype(np.float32)
        if name == REGISTERED_PRIMARY_TARGET_MASK:
            return np.dtype(np.bool_)
        try:
            dtype = self.manifest["arrays"][name]["dtype"]
        except KeyError as error:
            raise KeyError(f"v2 store does not contain {name}") from error
        return np.dtype(dtype)

    def _checked_date_selector(
        self,
        selector: int
        | np.integer
        | slice
        | range
        | Sequence[int]
        | NDArray[np.integer],
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
            int | np.integer | slice | range | Sequence[int] | NDArray[np.integer]
        ),
    ) -> NDArray[np.generic]:
        """Copy authorized date rows without exposing the backing whole-store mmap."""

        selector = self._checked_date_selector(date_selector)
        if name in _VIRTUAL_TARGETS:
            values, valid = self._neutral_primary_view(selector)
            return valid if name == REGISTERED_PRIMARY_TARGET_MASK else values
        try:
            array = self._arrays[name]
        except KeyError as error:
            raise KeyError(f"v2 store does not contain {name}") from error
        mask_name = _TARGET_VALUE_MASKS.get(name)
        if name in _MULTI_HORIZON_TARGET_MASKS:
            return self._target_mask(name, selector)
        if mask_name is None:
            return np.asarray(array[selector]).copy()
        if mask_name in _MULTI_HORIZON_TARGET_MASKS:
            valid = self._target_mask(mask_name, selector)
        else:
            valid = np.asarray(self._arrays[mask_name][selector], dtype=np.bool_).copy()
        return self._read_target_payload(name, array, selector, valid)

    def read_target(
        self,
        name: str,
        date_selector: (
            int | np.integer | slice | range | Sequence[int] | NDArray[np.integer]
        ),
        *,
        valid_mask: NDArray[np.bool_],
    ) -> NDArray[np.generic]:
        """Read only explicitly authorized target cells inside a local window.

        A store capability can cover a union of fit, selection, and evaluation
        dates.  A caller evaluating one window must therefore narrow the store's
        endpoint-valid mask before any target payload is decoded.  The supplied
        mask may only remove cells from the canonical capability mask.
        """

        selector = self._checked_date_selector(date_selector)
        if name == REGISTERED_PRIMARY_TARGET:
            values, canonical = self._neutral_primary_view(selector)
            requested = np.asarray(valid_mask)
            if requested.dtype != np.bool_ or requested.shape != canonical.shape:
                raise ValueError(
                    "local target validity mask must be boolean and match the "
                    "selected target payload"
                )
            if np.any(requested & ~canonical):
                raise PermissionError(
                    "local target validity mask exceeds the store capability"
                )
            return np.where(requested, values, 0.0).astype(np.float32, copy=False)
        mask_name = _TARGET_VALUE_MASKS.get(name)
        if mask_name is None:
            raise ValueError(f"{name} is not a masked target payload")
        try:
            array = self._arrays[name]
        except KeyError as error:
            raise KeyError(f"v2 store does not contain {name}") from error
        if mask_name in _MULTI_HORIZON_TARGET_MASKS:
            canonical = self._target_mask(mask_name, selector)
        else:
            canonical = np.asarray(
                self._arrays[mask_name][selector], dtype=np.bool_
            ).copy()
        requested = np.asarray(valid_mask)
        if requested.dtype != np.bool_ or requested.shape != canonical.shape:
            raise ValueError(
                "local target validity mask must be boolean and match the "
                "selected target payload"
            )
        if np.any(requested & ~canonical):
            raise PermissionError(
                "local target validity mask exceeds the store capability"
            )
        return self._read_target_payload(name, array, selector, requested)

    def _neutral_primary_view(
        self, selector: int | NDArray[np.int64]
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        """Build the rev-4 characteristic-neutral target on authorized rows."""

        scalar = isinstance(selector, int)
        indices = (
            np.asarray([selector], dtype=np.int64)
            if scalar
            else np.asarray(selector, dtype=np.int64)
        )
        canonical = self._target_mask("target_valid", indices)
        shareholder_valid = self._target_mask("target_shareholder_valid", indices)
        if np.any(canonical & ~shareholder_valid):
            raise ValueError(
                "scaled-target validity exceeds shareholder-return validity"
            )
        returns = self._read_target_payload(
            "target_shareholder_simple_return",
            self._arrays["target_shareholder_simple_return"],
            indices,
            canonical,
        )
        sigma = np.asarray(
            self._arrays["target_scale_sigma"][indices], dtype=np.float64
        )
        slow = np.asarray(self._arrays["slow_values"][indices], dtype=np.float64)
        slow_valid = np.asarray(self._arrays["slow_valid"][indices], dtype=np.bool_)
        feature_names = self.manifest.get("feature_names", {}).get("slow")
        if not isinstance(feature_names, list):
            raise ValueError("store lacks ordered slow-feature names")
        try:
            feature_indices = [
                feature_names.index(name) for name in TARGET_NEUTRALIZATION_FEATURES
            ]
        except ValueError as error:
            raise ValueError("store lacks a neutral-target characteristic") from error
        characteristics = slow[..., feature_indices]
        characteristic_valid = slow_valid[..., feature_indices].all(axis=-1)
        characteristic_valid &= np.isfinite(characteristics).all(axis=-1)
        output, valid = characteristic_neutral_targets(
            returns,
            canonical,
            sigma,
            characteristics,
            characteristic_valid,
            horizons=HORIZONS,
        )
        return (output[0], valid[0]) if scalar else (output, valid)

    def neutral_target_fallback_flags(
        self, selector: int | NDArray[np.int64]
    ) -> NDArray[np.bool_]:
        """Report the authorized date/horizon cells using the rev-3 fallback."""

        scalar = isinstance(selector, int)
        indices = (
            np.asarray([selector], dtype=np.int64)
            if scalar
            else np.asarray(selector, dtype=np.int64)
        )
        if not self.authorized_for(indices):
            raise PermissionError("neutral-target fallback audit exceeds store access")
        canonical = self._target_mask("target_valid", indices)
        returns = self._read_target_payload(
            "target_shareholder_simple_return",
            self._arrays["target_shareholder_simple_return"],
            indices,
            canonical,
        )
        sigma = np.asarray(
            self._arrays["target_scale_sigma"][indices], dtype=np.float64
        )
        slow = np.asarray(self._arrays["slow_values"][indices], dtype=np.float64)
        slow_valid = np.asarray(self._arrays["slow_valid"][indices], dtype=np.bool_)
        feature_names = self.manifest.get("feature_names", {}).get("slow")
        if not isinstance(feature_names, list):
            raise ValueError("store lacks ordered slow-feature names")
        feature_indices = [
            feature_names.index(name) for name in TARGET_NEUTRALIZATION_FEATURES
        ]
        characteristics = slow[..., feature_indices]
        characteristic_valid = slow_valid[..., feature_indices].all(axis=-1)
        characteristic_valid &= np.isfinite(characteristics).all(axis=-1)
        flags = np.zeros((len(indices), len(HORIZONS)), dtype=np.bool_)
        characteristic_neutral_targets(
            returns,
            canonical,
            sigma,
            characteristics,
            characteristic_valid,
            horizons=HORIZONS,
            fallback_flags=flags,
        )
        return flags[0] if scalar else flags

    def _read_target_payload(
        self,
        name: str,
        array: NDArray[np.generic],
        selector: int | NDArray[np.int64],
        valid: NDArray[np.bool_],
    ) -> NDArray[np.generic]:
        expected_shape = (
            array.shape[1:]
            if isinstance(selector, int)
            else (
                selector.size,
                *array.shape[1:],
            )
        )
        if expected_shape != valid.shape:
            raise ValueError(f"{name} and its validity mask are misaligned")
        # Read the authorization/validity mask first, then index only permitted
        # payload cells.  Reading a whole target row and zeroing it afterwards
        # still decodes outcomes whose endpoint is outside this capability.
        # That is an information-boundary violation even if the values are not
        # returned to the caller.
        result = np.zeros(valid.shape, dtype=array.dtype)
        coordinates = np.nonzero(valid)
        if not coordinates[0].size:
            return result
        if isinstance(selector, int):
            source_coordinates = (selector, *coordinates)
        else:
            source_coordinates = (selector[coordinates[0]], *coordinates[1:])
        result[coordinates] = array[source_coordinates]
        return result

    def _target_mask(
        self, name: str, selector: int | slice | NDArray[np.int64]
    ) -> NDArray[np.bool_]:
        """Clip targets unless every session in (t, t+H] is in this grant."""

        raw = np.asarray(self._arrays[name][selector], dtype=np.bool_).copy()
        scalar = raw.ndim == 2
        masks = raw[None, ...] if scalar else raw
        selected = np.atleast_1d(np.arange(self.dates.size, dtype=np.int64)[selector])
        if masks.shape[0] != selected.size or masks.shape[-1] != len(HORIZONS):
            raise ValueError(f"{name} has the wrong target axes")
        granted = self._authorized_date_indices
        for row, date_index in enumerate(selected):
            for horizon_index, horizon in enumerate(HORIZONS):
                if any(
                    endpoint not in granted
                    for endpoint in range(
                        int(date_index), int(date_index) + horizon + 1
                    )
                ):
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
        if name in {"v1_fast_isin_mapping", "native_fast_security_mapping"}:
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
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
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
    target_window_indices: Sequence[int] | None = None,
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
    if target_window_indices is None:
        target_indices = indices
    else:
        target_indices = _validated_store_indices(dates, target_window_indices)
    requested_indices = np.unique(np.concatenate((indices, target_indices)))
    requested = tuple(sorted(dates[requested_indices].astype(object).tolist()))
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
    authorized = set(int(value) for value in requested_indices)
    for end, lookback in zip(ends.tolist(), lookbacks.tolist(), strict=True):
        authorized.update(range(max(0, int(end) - int(lookback) + 1), int(end) + 1))
    grant = _StoreAccessGrant(
        root=path,
        date_indices=frozenset(authorized),
        ledger=ledger,
    )
    return (
        V2Store._open_internal(path, verify_hashes=verify_hashes, grant=grant),
        ledger,
    )
