from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from .contract import (
    ACCUMULATED_TEST_AFTER,
    DEVELOPMENT_END,
    FALLBACK_TEST_START,
    FINETUNE_START,
    HORIZONS,
    OFFICIAL_END,
    OFFICIAL_START,
    PRETRAIN_END,
    SELECTION_EMBARGO_SESSIONS,
    STORE_START,
)

PRETRAIN_START = STORE_START
DEVELOPMENT_START = FINETUNE_START
OFFICIAL_VALIDATION_START = OFFICIAL_START
OFFICIAL_VALIDATION_END = OFFICIAL_END
TEST_FALLBACK_START = FALLBACK_TEST_START
TEST_FALLBACK_END = ACCUMULATED_TEST_AFTER
FIT_EMBARGO_SESSIONS = SELECTION_EMBARGO_SESSIONS
BLOCK_PARITY_SESSIONS = 5
FIT_TO_SELECTION_PURGE_SESSIONS = 10
SELECTION_SESSIONS = 55
SELECTION_TO_EVALUATION_PURGE_SESSIONS = 10

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PREREGISTRATION_ROOT = PROJECT_ROOT / "research" / "preregistrations"

AccessPurpose = Literal["training", "selection", "evaluation"]

_SELECTION_WINDOWS = {
    "F1": (date(2023, 7, 3), date(2023, 12, 29)),
    "F2": (date(2024, 1, 2), date(2024, 6, 28)),
    "F3": (date(2024, 7, 1), date(2024, 12, 30)),
}


def development_evaluation_windows() -> dict[str, tuple[date, date]]:
    """Return the registered development evaluation date bounds."""

    return dict(_SELECTION_WINDOWS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _date_identity(dates: Sequence[date]) -> str:
    payload = json.dumps(
        [value.isoformat() for value in dates], separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class RegistrationToken:
    path: Path
    sha256: str

    def payload(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class AccessLedger:
    purpose: AccessPurpose
    official_validation_accessed: bool
    test_accessed: bool
    registration: RegistrationToken | None

    def payload(self) -> dict[str, object]:
        return {
            "purpose": self.purpose,
            "official_validation_accessed": self.official_validation_accessed,
            "test_accessed": self.test_accessed,
            "registration": (
                None if self.registration is None else self.registration.payload()
            ),
        }


@dataclass(frozen=True)
class DevelopmentFold:
    name: str
    fit_dates: tuple[date, ...]
    purge_before_dates: tuple[date, ...]
    selection_dates: tuple[date, ...]
    purge_after_dates: tuple[date, ...]
    evaluation_dates: tuple[date, ...]

    def payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "fit_dates": [value.isoformat() for value in self.fit_dates],
            "purge_before_dates": [
                value.isoformat() for value in self.purge_before_dates
            ],
            "selection_dates": [value.isoformat() for value in self.selection_dates],
            "purge_after_dates": [
                value.isoformat() for value in self.purge_after_dates
            ],
            "evaluation_dates": [value.isoformat() for value in self.evaluation_dates],
            "fit_date_identity_sha256": _date_identity(self.fit_dates),
            "purge_before_date_identity_sha256": _date_identity(
                self.purge_before_dates
            ),
            "selection_date_identity_sha256": _date_identity(self.selection_dates),
            "purge_after_date_identity_sha256": _date_identity(self.purge_after_dates),
            "evaluation_date_identity_sha256": _date_identity(self.evaluation_dates),
            "label_intervals_sha256": _label_intervals_sha256(self),
            "no_label_overlap_assertion": assert_no_label_overlap(self),
        }


@dataclass(frozen=True)
class BlockParityDirection:
    name: str
    selection_dates: tuple[date, ...]
    evaluation_dates: tuple[date, ...]


def _ordered_unique(dates: Sequence[date]) -> tuple[date, ...]:
    values = tuple(dates)
    if not values or values != tuple(sorted(values)) or len(set(values)) != len(values):
        raise ValueError("dates must be nonempty, unique, and chronological")
    return values


def validate_contiguous_session_axis(
    dates: Sequence[date],
    session_indices: NDArray[np.integer],
) -> NDArray[np.int64]:
    """Bind dates to consecutive positions on a caller-identified calendar."""
    values = _ordered_unique(dates)
    indices = np.asarray(session_indices)
    if (
        indices.ndim != 1
        or indices.shape != (len(values),)
        or not np.issubdtype(indices.dtype, np.integer)
        or np.issubdtype(indices.dtype, np.bool_)
    ):
        raise TypeError("session_indices must be a one-dimensional integer array")
    normalized = indices.astype(np.int64, copy=False)
    if (normalized < 0).any():
        raise ValueError("session_indices must be non-negative")
    if len(normalized) > 1 and not np.equal(np.diff(normalized), 1).all():
        raise ValueError(
            "evaluation dates must be contiguous canonical sessions; "
            "gappy or parity-only axes are not valid for row-lag metrics"
        )
    return normalized


def _registration_token(path: Path, *, preregistration_root: Path) -> RegistrationToken:
    root = preregistration_root.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("registration must be inside research/preregistrations")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return RegistrationToken(resolved, _sha256(resolved))


def authorize_dates(
    dates: Sequence[date],
    *,
    purpose: AccessPurpose,
    registration_path: Path | None = None,
    preregistration_root: Path = PREREGISTRATION_ROOT,
) -> AccessLedger:
    """Authorize a model-data access before any array is opened.

    The current codebase refuses every potential v2 test date. Official
    validation requires a repository preregistration whose exact bytes are
    recorded in the returned ledger.
    """
    values = _ordered_unique(dates)
    if purpose not in ("training", "selection", "evaluation"):
        raise ValueError(f"Unknown access purpose: {purpose}")
    test_dates = [value for value in values if value >= TEST_FALLBACK_START]
    if test_dates:
        raise ValueError(
            "v2 held-out test dates are unconditionally sealed in this codebase"
        )
    official = [
        value
        for value in values
        if OFFICIAL_VALIDATION_START <= value <= OFFICIAL_VALIDATION_END
    ]
    if official and purpose != "evaluation":
        raise PermissionError(
            "v2 official validation may be evaluated after registration, "
            "but it may never be used for training or selection"
        )
    allowed = [
        value
        for value in values
        if PRETRAIN_START <= value <= PRETRAIN_END
        or DEVELOPMENT_START <= value <= DEVELOPMENT_END
        or OFFICIAL_VALIDATION_START <= value <= OFFICIAL_VALIDATION_END
    ]
    if len(allowed) != len(values):
        allowed_set = set(allowed)
        invalid = next(value for value in values if value not in allowed_set)
        raise ValueError(f"date is outside an authorized v2 window: {invalid}")
    if official and registration_path is None:
        raise PermissionError("v2 official validation requires a preregistration token")
    if not official and registration_path is not None:
        raise ValueError(
            "a preregistration token may only authorize official validation"
        )
    token = (
        None
        if registration_path is None
        else _registration_token(
            registration_path, preregistration_root=preregistration_root
        )
    )
    return AccessLedger(purpose, bool(official), False, token)


def development_folds(calendar_dates: Sequence[date]) -> tuple[DevelopmentFold, ...]:
    dates = _ordered_unique(calendar_dates)
    development = tuple(
        value for value in dates if DEVELOPMENT_START <= value <= DEVELOPMENT_END
    )
    if not development:
        raise ValueError("calendar contains no v2 development sessions")
    position = {value: index for index, value in enumerate(dates)}
    folds: list[DevelopmentFold] = []
    for name, (start, end) in _SELECTION_WINDOWS.items():
        evaluation = tuple(value for value in development if start <= value <= end)
        before = tuple(value for value in development if value < start)
        if dates[0] > start or dates[-1] < end or not evaluation:
            raise ValueError(f"{name} evaluation window is outside the calendar")
        if len(before) <= FIT_EMBARGO_SESSIONS:
            raise ValueError(f"{name} has too few pre-evaluation sessions")
        fit = before[:-FIT_EMBARGO_SESSIONS]
        heldout = before[-FIT_EMBARGO_SESSIONS:]
        selection_start = FIT_TO_SELECTION_PURGE_SESSIONS
        selection_end = selection_start + SELECTION_SESSIONS
        purge_before = heldout[:selection_start]
        selection = heldout[selection_start:selection_end]
        purge_after = heldout[selection_end:]
        if (
            len(selection) != SELECTION_SESSIONS
            or len(purge_before) != FIT_TO_SELECTION_PURGE_SESSIONS
            or len(purge_after) != SELECTION_TO_EVALUATION_PURGE_SESSIONS
        ):
            raise ValueError(f"{name} chronological holdout has the wrong size")
        if position[fit[-1]] + max(HORIZONS) >= position[selection[0]]:
            raise ValueError(f"{name} fit target interval overlaps selection")
        if position[selection[-1]] + max(HORIZONS) >= position[evaluation[0]]:
            raise ValueError(f"{name} selection target interval overlaps evaluation")
        fold = DevelopmentFold(
            name,
            fit,
            purge_before,
            selection,
            purge_after,
            evaluation,
        )
        assert_no_label_overlap(fold)
        folds.append(fold)
    return tuple(folds)


def label_intervals(
    fold: DevelopmentFold,
) -> dict[str, dict[str, list[tuple[int, int]]]]:
    """Return usable target entry/exit session pairs for each fold segment."""

    ordered = (
        *fold.fit_dates,
        *fold.purge_before_dates,
        *fold.selection_dates,
        *fold.purge_after_dates,
        *fold.evaluation_dates,
    )
    if tuple(sorted(ordered)) != ordered or len(set(ordered)) != len(ordered):
        raise ValueError("fold segments must be unique and chronological")
    position = {value: index for index, value in enumerate(ordered)}
    result: dict[str, dict[str, list[tuple[int, int]]]] = {}
    for segment, entries, exits in (
        (
            "fit",
            fold.fit_dates,
            (*fold.fit_dates, *fold.purge_before_dates),
        ),
        (
            "selection",
            fold.selection_dates,
            fold.selection_dates,
        ),
        ("evaluation", fold.evaluation_dates, fold.evaluation_dates),
    ):
        allowed_exits = set(exits)
        by_horizon: dict[str, list[tuple[int, int]]] = {}
        for horizon in HORIZONS:
            pairs = []
            for entry in entries:
                start = position[entry]
                stop = start + horizon
                if stop < len(ordered) and ordered[stop] in allowed_exits:
                    pairs.append((start, stop))
            by_horizon[str(horizon)] = pairs
        result[segment] = by_horizon
    return result


def assert_no_label_overlap(fold: DevelopmentFold) -> bool:
    """Assert fit, selection, and evaluation labels never enter another segment."""

    intervals = label_intervals(fold)
    ordered = (
        *fold.fit_dates,
        *fold.purge_before_dates,
        *fold.selection_dates,
        *fold.purge_after_dates,
        *fold.evaluation_dates,
    )
    position = {value: index for index, value in enumerate(ordered)}
    if (
        position[fold.fit_dates[-1]] + max(HORIZONS)
        >= position[fold.selection_dates[0]]
    ):
        raise ValueError("fit label reaches selection")
    if (
        position[fold.selection_dates[-1]] + max(HORIZONS)
        >= position[fold.evaluation_dates[0]]
    ):
        raise ValueError("selection label reaches evaluation")
    allowed_entries = {
        "fit": set(fold.fit_dates),
        "selection": set(fold.selection_dates),
        "evaluation": set(fold.evaluation_dates),
    }
    allowed_exits = {
        "fit": set((*fold.fit_dates, *fold.purge_before_dates)),
        "selection": set(fold.selection_dates),
        "evaluation": set(fold.evaluation_dates),
    }
    for segment, horizons in intervals.items():
        for pairs in horizons.values():
            if any(
                ordered[start] not in allowed_entries[segment]
                or ordered[stop] not in allowed_exits[segment]
                for start, stop in pairs
            ):
                raise ValueError(f"{segment} label crosses its registered window")
    evaluation_d10 = intervals["evaluation"][str(max(HORIZONS))]
    if evaluation_d10 and any(
        ordered[stop] > DEVELOPMENT_END for _, stop in evaluation_d10
    ):
        raise ValueError("evaluation D10 label crosses the development boundary")
    return True


def _label_intervals_sha256(fold: DevelopmentFold) -> str:
    payload = json.dumps(
        label_intervals(fold), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def block_parity_directions(
    selection_dates: Sequence[date],
) -> tuple[BlockParityDirection, BlockParityDirection]:
    dates = _ordered_unique(selection_dates)
    block = np.arange(len(dates), dtype=np.int64) // BLOCK_PARITY_SESSIONS
    even = tuple(
        value for value, index in zip(dates, block, strict=True) if index % 2 == 0
    )
    odd = tuple(
        value for value, index in zip(dates, block, strict=True) if index % 2 == 1
    )
    if not even or not odd or set(even) & set(odd):
        raise ValueError("selection window cannot form both block parities")
    return (
        BlockParityDirection("even_select_odd_evaluate", even, odd),
        BlockParityDirection("odd_select_even_evaluate", odd, even),
    )


def mask_targets_to_window(
    target_mask: NDArray[np.bool_],
    *,
    calendar_dates: Sequence[date],
    window_dates: Sequence[date],
    horizons: Sequence[int] = HORIZONS,
) -> NDArray[np.bool_]:
    """Restrict targets so every session from entry through exit stays in-window."""
    endpoint_mask = target_window_endpoint_mask(
        calendar_dates=calendar_dates,
        window_dates=window_dates,
        horizons=horizons,
    )
    mask = np.asarray(target_mask)
    if mask.dtype != np.bool_:
        raise TypeError("target mask must be a Boolean array")
    if (
        mask.ndim != 3
        or mask.shape[0] != endpoint_mask.shape[0]
        or mask.shape[2] != endpoint_mask.shape[1]
    ):
        raise ValueError("target mask shape differs from date/horizon axes")
    return mask & endpoint_mask[:, None, :]


def target_window_endpoint_mask(
    *,
    calendar_dates: Sequence[date],
    window_dates: Sequence[date],
    horizons: Sequence[int] = HORIZONS,
) -> NDArray[np.bool_]:
    """Authorize entry/horizon coordinates without touching target payloads.

    Consumers must apply this date-only mask to target validity before reading
    numeric target arrays.  In particular, F3 tail labels whose endpoints enter
    the later sealed window remain unreadable rather than being decoded and then
    zeroed.
    """

    dates = _ordered_unique(calendar_dates)
    window = _ordered_unique(window_dates)
    horizon_values = tuple(int(value) for value in horizons)
    if (
        not horizon_values
        or any(value <= 0 for value in horizon_values)
        or len(set(horizon_values)) != len(horizon_values)
    ):
        raise ValueError("target horizons must be unique and positive")
    window_set = set(window)
    if not window_set.issubset(dates):
        raise ValueError("window dates are absent from the calendar")
    in_window = np.asarray([value in window_set for value in dates], dtype=bool)
    result = np.zeros((len(dates), len(horizon_values)), dtype=np.bool_)
    for day in range(len(dates)):
        for horizon_index, horizon in enumerate(horizon_values):
            stop = day + horizon
            contained = stop < len(dates) and bool(in_window[day : stop + 1].all())
            result[day, horizon_index] = contained
    return result
