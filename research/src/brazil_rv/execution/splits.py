from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

from ..modeling.contract import (
    DISCOVERY_FIT_DATE_COUNTS,
    DISCOVERY_SELECTION_DATE_COUNT,
    EXPECTED_SPLIT_DATE_COUNTS,
    THIRD_DISCOVERY_SELECTION_END,
    THIRD_DISCOVERY_SELECTION_START,
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class PurgedFold:
    name: str
    fit_dates: tuple[date, ...]
    heldout_dates: tuple[date, ...]
    embargo_dates: tuple[date, ...]

    def payload(self) -> dict[str, object]:
        fit_dates = [item.isoformat() for item in self.fit_dates]
        heldout_dates = [item.isoformat() for item in self.heldout_dates]
        embargo_dates = [item.isoformat() for item in self.embargo_dates]
        value: dict[str, object] = {
            "name": self.name,
            "fit_dates": fit_dates,
            "heldout_dates": heldout_dates,
            "embargo_dates": embargo_dates,
            "fit_date_identity_sha256": _canonical_sha256(fit_dates),
            "heldout_date_identity_sha256": _canonical_sha256(heldout_dates),
            "embargo_date_identity_sha256": _canonical_sha256(embargo_dates),
        }
        value["sha256"] = _canonical_sha256(value)
        return value


@dataclass(frozen=True)
class PurgedKFoldManifest:
    folds: tuple[PurgedFold, ...]
    date_count: int
    embargo_sessions: int
    schema: str = "BRAZIL_RV_PURGED_KFOLD_V1"

    def payload(self) -> dict[str, object]:
        training_dates = sorted(
            value.isoformat() for fold in self.folds for value in fold.heldout_dates
        )
        value: dict[str, object] = {
            "schema": self.schema,
            "date_count": self.date_count,
            "fold_count": len(self.folds),
            "embargo_sessions": self.embargo_sessions,
            "training_date_identity_sha256": _canonical_sha256(training_dates),
            "folds": [fold.payload() for fold in self.folds],
        }
        value["sha256"] = _canonical_sha256(value)
        return value

    @property
    def sha256(self) -> str:
        return str(self.payload()["sha256"])

    def write(self, path: Path) -> None:
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(self.payload(), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


def purged_training_folds(
    training_dates: Sequence[date],
    *,
    fold_count: int = 5,
    embargo_sessions: int = 5,
) -> PurgedKFoldManifest:
    """Partition TRAIN dates and purge each fold's adjacent fit sessions.

    The held-out blocks collectively cover every date exactly once. For a
    particular fold, its five neighboring sessions on either side are neither
    fit nor held out; those dates remain held out in their own fold.
    """
    dates = tuple(training_dates)
    expected = EXPECTED_SPLIT_DATE_COUNTS["train"]
    if len(dates) != expected or len(set(dates)) != expected:
        raise ValueError(f"Purged folds require exactly {expected} unique dates")
    if dates != tuple(sorted(dates)):
        raise ValueError("Training dates must be strictly chronological")
    if fold_count != 5 or embargo_sessions != 5:
        raise ValueError(
            "The registered purged TRAIN contract is exactly 5 folds / 5 sessions"
        )

    quotient, remainder = divmod(len(dates), fold_count)
    sizes = [quotient + (index < remainder) for index in range(fold_count)]
    folds: list[PurgedFold] = []
    start = 0
    for index, size in enumerate(sizes):
        stop = start + size
        purge_start = max(0, start - embargo_sessions)
        purge_stop = min(len(dates), stop + embargo_sessions)
        heldout = dates[start:stop]
        embargo = dates[purge_start:start] + dates[stop:purge_stop]
        fit = dates[:purge_start] + dates[purge_stop:]
        folds.append(PurgedFold(f"fold_{index}", fit, heldout, embargo))
        start = stop
    return PurgedKFoldManifest(tuple(folds), len(dates), embargo_sessions)


def load_purged_training_folds(
    path: Path, expected_training_dates: Sequence[date]
) -> PurgedKFoldManifest:
    """Load a frozen manifest and bind it to the canonical TRAIN date axis."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "BRAZIL_RV_PURGED_KFOLD_V1"
    ):
        raise ValueError("Purged-fold manifest schema differs")
    raw_folds = payload.get("folds")
    if not isinstance(raw_folds, list) or len(raw_folds) != 5:
        raise ValueError("Purged-fold manifest must contain five folds")
    expected = purged_training_folds(expected_training_dates)
    if payload != expected.payload():
        raise ValueError("Purged-fold manifest differs from the canonical TRAIN axis")
    return expected


@dataclass(frozen=True)
class PolicyEvaluationSlice:
    name: str
    dates: tuple[date, ...]


def policy_evaluation_slices(
    oof_dates: Sequence[date],
    expected_training_dates: Sequence[date],
) -> tuple[PolicyEvaluationSlice, ...]:
    """Return the canonical C/A/B evaluation windows over OOF TRAIN dates."""
    dates = tuple(oof_dates)
    if dates != tuple(expected_training_dates):
        raise ValueError("OOF dates differ from the canonical TRAIN axis")
    expected = EXPECTED_SPLIT_DATE_COUNTS["train"]
    if len(dates) != expected or len(set(dates)) != expected:
        raise ValueError(f"Policy slices require exactly {expected} unique OOF dates")
    if dates != tuple(sorted(dates)):
        raise ValueError("OOF dates must be strictly chronological")
    fold_c = tuple(
        value
        for value in dates
        if THIRD_DISCOVERY_SELECTION_START <= value <= THIRD_DISCOVERY_SELECTION_END
    )
    fold_a_start = int(DISCOVERY_FIT_DATE_COUNTS["fold_a"])
    fold_b_start = int(DISCOVERY_FIT_DATE_COUNTS["fold_b"])
    fold_a = dates[fold_a_start : fold_a_start + DISCOVERY_SELECTION_DATE_COUNT]
    fold_b = dates[fold_b_start : fold_b_start + DISCOVERY_SELECTION_DATE_COUNT]
    if (
        len(fold_c) != 105
        or len(fold_a) != 102
        or len(fold_b) != 102
        or fold_c != dates[fold_a_start - 105 : fold_a_start]
        or not set(fold_c).isdisjoint(fold_a)
        or not set(fold_c).isdisjoint(fold_b)
        or not set(fold_a).isdisjoint(fold_b)
    ):
        raise ValueError("OOF dates do not reproduce the canonical C/A/B windows")
    return (
        PolicyEvaluationSlice("fold_c", fold_c),
        PolicyEvaluationSlice("fold_a", fold_a),
        PolicyEvaluationSlice("fold_b", fold_b),
    )
