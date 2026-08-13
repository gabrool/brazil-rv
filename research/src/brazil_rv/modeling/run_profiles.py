from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import polars as pl

from .contract import (
    EQUITY_COUNT,
    EXPECTED_DECISIONS_PER_DATE,
    FEATURE_STORE_POINTER,
    MAX_EPOCHS,
    MIN_ACTIVE_EQUITIES,
    PROJECT_ROOT,
    TRAIN_END,
    TRAIN_START,
)

RUN_PROFILE_SCHEMA_VERSION = "B3_MODEL_RUN_PROFILE_V2"
RUN_PROFILE_NAMES = ("production", "experiment")
EXPERIMENT_EQUITY_COUNT = 48
EXPERIMENT_DECISION_INDICES = tuple(range(0, EXPECTED_DECISIONS_PER_DATE, 3))
EXPERIMENT_MAXIMUM_EPOCHS = 3
MINIMUM_TRAINING_DATES = 512
LIQUIDITY_MEASURE = "median_daily_turnover_brl"
LIQUIDITY_AGGREGATION = (
    "median of canonical lagged monthly values whose effective interval overlaps "
    "the training interval and for which is_member=true"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _identity_sha256(metadata: dict[str, object]) -> str:
    return hashlib.sha256(_canonical_json(metadata).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RunProfile:
    name: str
    equity_slots: tuple[int, ...]
    security_ids: tuple[str, ...]
    symbols: tuple[str, ...]
    decision_indices: tuple[int, ...]
    maximum_epochs: int
    minimum_active_equities: int
    minimum_training_dates: int
    decision_grouped_batches: bool
    provenance: dict[str, object]
    selection: tuple[dict[str, object], ...]
    identity_sha256: str

    @property
    def equity_count(self) -> int:
        return len(self.equity_slots)

    @property
    def instrument_count(self) -> int:
        from .contract import CONTEXT_COUNT

        return self.equity_count + CONTEXT_COUNT

    def metadata(self) -> dict[str, object]:
        return {
            "schema_version": RUN_PROFILE_SCHEMA_VERSION,
            "name": self.name,
            "equity_count": self.equity_count,
            "equity_slots": list(self.equity_slots),
            "security_ids": list(self.security_ids),
            "symbols": list(self.symbols),
            "decision_indices": list(self.decision_indices),
            "maximum_epochs": self.maximum_epochs,
            "minimum_active_equities": self.minimum_active_equities,
            "minimum_training_dates": self.minimum_training_dates,
            "decision_grouped_batches": self.decision_grouped_batches,
            "provenance": self.provenance,
            "selection": list(self.selection),
            "identity_sha256": self.identity_sha256,
        }


def _profile_payload(
    *,
    name: str,
    equity_slots: tuple[int, ...],
    security_ids: tuple[str, ...],
    symbols: tuple[str, ...],
    decision_indices: tuple[int, ...],
    maximum_epochs: int,
    decision_grouped_batches: bool,
    provenance: dict[str, object],
    selection: tuple[dict[str, object], ...],
) -> dict[str, object]:
    return {
        "schema_version": RUN_PROFILE_SCHEMA_VERSION,
        "name": name,
        "equity_count": len(equity_slots),
        "equity_slots": list(equity_slots),
        "security_ids": list(security_ids),
        "symbols": list(symbols),
        "decision_indices": list(decision_indices),
        "maximum_epochs": maximum_epochs,
        "minimum_active_equities": MIN_ACTIVE_EQUITIES,
        "minimum_training_dates": MINIMUM_TRAINING_DATES,
        "decision_grouped_batches": decision_grouped_batches,
        "provenance": provenance,
        "selection": list(selection),
    }


def _portable_path_candidates(value: str) -> tuple[Path, ...]:
    candidates = [Path(value).expanduser()]
    windows_parts = PureWindowsPath(value).parts
    if "quant-data" in windows_parts:
        offset = windows_parts.index("quant-data")
        candidates.append(PROJECT_ROOT.joinpath(*windows_parts[offset:]))
    return tuple(dict.fromkeys(candidates))


def _exact_directory_matches(
    candidates: tuple[Path, ...],
    expected_name: str,
    checked: list[str],
) -> tuple[Path, ...]:
    matches: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        checked.append(str(resolved))
        if candidate.is_dir() and candidate.name == expected_name:
            matches.append(resolved)
    return tuple(dict.fromkeys(matches))


def _single_portable_match(
    candidates: tuple[Path, ...],
    expected_name: str,
    checked: list[str],
) -> Path | None:
    matches = _exact_directory_matches(candidates, expected_name, checked)
    if len(matches) > 1:
        raise ValueError(
            f"PIT-universe identity is ambiguous; exact checked candidates: {' | '.join(checked)}"
        )
    return matches[0] if matches else None


def _resolved_universe(feature_manifest: dict[str, Any]) -> Path:
    canonical = feature_manifest.get("canonical_inputs")
    if not isinstance(canonical, dict):
        raise ValueError("Feature manifest is missing canonical_inputs")
    universe = canonical.get("point_in_time_universe")
    if not isinstance(universe, dict) or not isinstance(
        universe.get("resolved_path"), str
    ):
        raise ValueError("Feature manifest is missing its PIT-universe identity")
    recorded = str(universe["resolved_path"])
    expected_name = PureWindowsPath(recorded).name
    checked: list[str] = []
    recorded_candidates = _portable_path_candidates(recorded)
    direct = _single_portable_match(recorded_candidates[:1], expected_name, checked)
    if direct is not None:
        return direct
    mapped = _single_portable_match(recorded_candidates[1:], expected_name, checked)
    if mapped is not None:
        return mapped
    pointer = (
        PROJECT_ROOT
        / "quant-data"
        / "b3"
        / "interim"
        / "universe"
        / "pit_v1_canonical_path.txt"
    )
    if not pointer.is_file():
        raise FileNotFoundError(
            "Resolved PIT universe does not exist; "
            f"recorded={recorded}; exact checked candidates={' | '.join(checked)}"
        )
    pointer_value = pointer.read_text(encoding="utf-8").strip()
    pointer_match = _single_portable_match(
        _portable_path_candidates(pointer_value), expected_name, checked
    )
    if pointer_match is None:
        raise FileNotFoundError(
            "Canonical PIT universe cannot be portably resolved to the feature "
            f"manifest identity; recorded={recorded}; "
            f"exact checked candidates={' | '.join(checked)}"
        )
    return pointer_match


def _equity_axis(store: Path) -> pl.DataFrame:
    axis = pl.read_parquet(store / "equity_index.parquet").sort("equity_slot")
    expected = np.arange(EQUITY_COUNT, dtype=np.int16)
    slots = axis.get_column("equity_slot").to_numpy()
    if axis.height != EQUITY_COUNT or not np.array_equal(slots, expected):
        raise ValueError("Feature-store equity axis is not the canonical 158-slot axis")
    return axis


def _production_profile(store: Path, manifest: dict[str, Any]) -> RunProfile:
    axis = _equity_axis(store)
    universe = _resolved_universe(manifest)
    slots = tuple(int(value) for value in axis.get_column("equity_slot"))
    security_ids = tuple(str(value) for value in axis.get_column("security_id"))
    symbols = tuple(str(value) for value in axis.get_column("xp_symbol"))
    provenance: dict[str, object] = {
        "feature_store_pointer": str(FEATURE_STORE_POINTER),
        "resolved_feature_store": str(store),
        "feature_manifest_sha256": _sha256(store / "manifest.json"),
        "equity_index_sha256": _sha256(store / "equity_index.parquet"),
        "resolved_pit_universe": str(universe),
        "selection_policy": "canonical feature-store equity axis in slot order",
    }
    selection = tuple(
        {
            "packed_slot": packed,
            "canonical_equity_slot": slot,
            "security_id": security_id,
            "symbol": symbol,
        }
        for packed, (slot, security_id, symbol) in enumerate(
            zip(slots, security_ids, symbols, strict=True)
        )
    )
    payload = _profile_payload(
        name="production",
        equity_slots=slots,
        security_ids=security_ids,
        symbols=symbols,
        decision_indices=tuple(range(EXPECTED_DECISIONS_PER_DATE)),
        maximum_epochs=MAX_EPOCHS,
        decision_grouped_batches=True,
        provenance=provenance,
        selection=selection,
    )
    return RunProfile(
        name="production",
        equity_slots=slots,
        security_ids=security_ids,
        symbols=symbols,
        decision_indices=tuple(range(EXPECTED_DECISIONS_PER_DATE)),
        maximum_epochs=MAX_EPOCHS,
        minimum_active_equities=MIN_ACTIVE_EQUITIES,
        minimum_training_dates=MINIMUM_TRAINING_DATES,
        decision_grouped_batches=True,
        provenance=provenance,
        selection=selection,
        identity_sha256=_identity_sha256(payload),
    )


def _experiment_profile(store: Path, manifest: dict[str, Any]) -> RunProfile:
    axis = _equity_axis(store)
    universe = _resolved_universe(manifest)
    metrics_path = universe / "universe_metrics_monthly.parquet"
    universe_manifest_path = universe / "manifest.json"
    metrics = (
        pl.scan_parquet(metrics_path)
        .filter(
            (pl.col("effective_from") <= pl.lit(TRAIN_END))
            & (pl.col("effective_to_exclusive") > pl.lit(TRAIN_START))
            & pl.col("accepted_identity")
            & pl.col("is_member")
            & pl.col(LIQUIDITY_MEASURE).is_finite()
        )
        .group_by("security_id")
        .agg(
            pl.col(LIQUIDITY_MEASURE).median().alias("training_liquidity_score_brl"),
            pl.len().alias("eligible_month_count"),
            pl.col("effective_from").min().alias("first_effective_from"),
            pl.col("effective_to_exclusive").max().alias("last_effective_to_exclusive"),
        )
        .collect()
    )
    ranked = (
        axis.join(metrics, on="security_id", how="inner", validate="1:1")
        .sort(
            ["training_liquidity_score_brl", "security_id"],
            descending=[True, False],
        )
        .head(EXPERIMENT_EQUITY_COUNT)
    )
    if ranked.height != EXPERIMENT_EQUITY_COUNT:
        raise ValueError(
            f"Experiment profile requires {EXPERIMENT_EQUITY_COUNT} eligible equities; "
            f"found {ranked.height}"
        )
    slots = tuple(int(value) for value in ranked.get_column("equity_slot"))
    security_ids = tuple(str(value) for value in ranked.get_column("security_id"))
    symbols = tuple(str(value) for value in ranked.get_column("xp_symbol"))
    provenance: dict[str, object] = {
        "feature_store_pointer": str(FEATURE_STORE_POINTER),
        "resolved_feature_store": str(store),
        "feature_manifest_sha256": _sha256(store / "manifest.json"),
        "equity_index_sha256": _sha256(store / "equity_index.parquet"),
        "resolved_pit_universe": str(universe),
        "pit_universe_manifest_sha256": _sha256(universe_manifest_path),
        "liquidity_source": str(metrics_path),
        "liquidity_source_sha256": _sha256(metrics_path),
        "liquidity_measure": LIQUIDITY_MEASURE,
        "liquidity_aggregation": LIQUIDITY_AGGREGATION,
        "selection_interval": {
            "start": TRAIN_START.isoformat(),
            "end": TRAIN_END.isoformat(),
            "validation_or_test_rows_used": False,
        },
        "ranking_tie_break": "security_id ascending",
    }
    selection = tuple(
        {
            "packed_slot": packed,
            "canonical_equity_slot": int(row["equity_slot"]),
            "security_id": str(row["security_id"]),
            "symbol": str(row["xp_symbol"]),
            "training_liquidity_score_brl": float(row["training_liquidity_score_brl"]),
            "eligible_month_count": int(row["eligible_month_count"]),
            "first_effective_from": row["first_effective_from"].isoformat(),
            "last_effective_to_exclusive": row[
                "last_effective_to_exclusive"
            ].isoformat(),
        }
        for packed, row in enumerate(ranked.iter_rows(named=True))
    )
    payload = _profile_payload(
        name="experiment",
        equity_slots=slots,
        security_ids=security_ids,
        symbols=symbols,
        decision_indices=EXPERIMENT_DECISION_INDICES,
        maximum_epochs=EXPERIMENT_MAXIMUM_EPOCHS,
        decision_grouped_batches=True,
        provenance=provenance,
        selection=selection,
    )
    return RunProfile(
        name="experiment",
        equity_slots=slots,
        security_ids=security_ids,
        symbols=symbols,
        decision_indices=EXPERIMENT_DECISION_INDICES,
        maximum_epochs=EXPERIMENT_MAXIMUM_EPOCHS,
        minimum_active_equities=MIN_ACTIVE_EQUITIES,
        minimum_training_dates=MINIMUM_TRAINING_DATES,
        decision_grouped_batches=True,
        provenance=provenance,
        selection=selection,
        identity_sha256=_identity_sha256(payload),
    )


def resolve_run_profile(name: str, store: Path) -> RunProfile:
    if name not in RUN_PROFILE_NAMES:
        raise ValueError(f"Invalid run profile: {name}")
    resolved_store = store.expanduser().resolve()
    manifest = json.loads(
        (resolved_store / "manifest.json").read_text(encoding="utf-8")
    )
    return (
        _production_profile(resolved_store, manifest)
        if name == "production"
        else _experiment_profile(resolved_store, manifest)
    )


def write_run_profile(path: Path, profile: RunProfile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(profile.metadata(), output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def validate_run_profile_artifact(path: Path, expected: RunProfile) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Run-profile artifact is unreadable: {path}") from error
    if payload != expected.metadata():
        raise ValueError("Run-profile artifact does not exactly match resolved policy")


def filter_profile_rows(
    sample_index: pl.DataFrame,
    store: Path,
    profile: RunProfile,
    *,
    require_training_dates: bool = True,
) -> pl.DataFrame:
    if profile.name == "production":
        return sample_index
    filtered = sample_index.filter(
        pl.col("decision_idx").is_in(profile.decision_indices)
    )
    membership = np.load(
        store / "equity_membership.npy", mmap_mode="r", allow_pickle=False
    )
    readiness = np.load(
        store / "equity_data_ready.npy", mmap_mode="r", allow_pickle=False
    )
    date_indices = filtered.get_column("date_idx").to_numpy().astype(np.int64)
    active_counts = (
        membership[date_indices][:, profile.equity_slots]
        & readiness[date_indices][:, profile.equity_slots]
    ).sum(axis=1)
    filtered = filtered.with_columns(
        pl.Series("profile_active_equity_count", active_counts.astype(np.int16))
    ).filter(pl.col("profile_active_equity_count") >= profile.minimum_active_equities)
    training_dates = (
        filtered.filter(pl.col("trade_date").is_between(TRAIN_START, TRAIN_END))
        .get_column("trade_date")
        .n_unique()
    )
    if require_training_dates and training_dates < profile.minimum_training_dates:
        raise ValueError(
            f"Experiment profile requires at least {profile.minimum_training_dates} "
            f"training dates, found {training_dates}"
        )
    decisions = set(filtered.get_column("decision_idx").unique().to_list())
    if decisions != set(profile.decision_indices):
        raise ValueError(
            "Experiment profile lost one or more required decision indices"
        )
    return filtered
