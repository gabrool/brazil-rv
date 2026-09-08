from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, replace
from datetime import date
from functools import partial
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import polars as pl
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from brazil_rv.execution.inputs import load_daily_cdi_rates

from .artifacts import inventory, sha256_file, write_json_atomic
from .baselines import BaselinePanel, build_store_baselines
from .bova11 import load_bova11_series
from .hedge_beta import HEDGE_BETA_MAX_AGE, load_hedge_beta_sidecar
from .lending_archive import LendingBorrowPanels, load_lending_borrow_panels
from .config import (
    PROJECT_ROOT,
    PROTOCOL_CONFIG_ROOT,
    ModelConfig,
    protocol_preset,
)
from .contract import (
    ALLOWED_LOOKBACKS,
    DEVELOPMENT_END,
    FINETUNE_START,
    GBDT_SEEDS,
    HORIZONS,
    OFFICIAL_START,
    PRETRAIN_END,
    REGISTERED_PRIMARY_TARGET,
    REGISTERED_PRIMARY_TARGET_MASK,
    SCORE_ARTIFACT_SCHEMA,
    SIDECAR_FEATURES,
    STORE_START,
)
from .data import (
    V2DailyDataset,
    collate_v2_daily,
    read_scalar_feature_view,
    scalar_feature_names,
    stage_fast_name_count,
)
from .data_roots import resolve_external_files, resolve_external_root
from .evaluate import (
    EVALUATION_SCHEMA,
    EvaluationInputs,
    EvaluationResult,
    evaluate_scores,
)
from .gbdt import (
    GBDTConfig,
    MultiHorizonGBDT,
    assemble_gbdt_scalar_view,
    gbdt_scalar_feature_names,
)
from .score import ScoreArtifact, score_checkpoint_artifact
from .splits import AccessPurpose, development_folds
from .store import STORE_SCHEMA, V2Store, open_store_for_samples
from .train import (
    DatePairBatchSampler,
    StageTrainingResult,
    pretrain_internal_split,
    train_stage,
)

PIPELINE_SCHEMA = "BRAZIL_RV_V2_PIPELINE_VALIDATION_V13"
_PRIOR_PIPELINE_SCHEMA = "BRAZIL_RV_V2_PIPELINE_VALIDATION_V12"
_ACCEPTANCE_ANCESTOR_SCHEMAS = frozenset(
    {
        "BRAZIL_RV_V2_PIPELINE_VALIDATION_V5",
        "BRAZIL_RV_V2_PIPELINE_VALIDATION_V6",
    }
)
PIPELINE_NETWORK_RESUME_SCHEMA = "BRAZIL_RV_V2_PIPELINE_NETWORK_RESUME_V2"
PIPELINE_FLAGS: dict[str, bool] = {
    "pipeline_validation": True,
    "research_claim": False,
    "official_validation_accessed": False,
    "test_accessed": False,
    "transfer_chronology_clean": True,
}
_MIN_WINDOW_SESSIONS = max(HORIZONS) + 2
_BASELINE_SIGNAL_SIGNS = {
    "reversal_5": -1.0,
    "reversal_21": -1.0,
    "momentum_12_1": 1.0,
    "reversal_5_momentum_12_1_blend": None,
    "inverse_volatility_20": -1.0,
}
_NAIVE_SIGNAL_BASELINES = tuple(
    name for name in _BASELINE_SIGNAL_SIGNS if name != "inverse_volatility_20"
)
_LEDGER_MASK_COVERAGE_FIELDS = frozenset(
    {
        "stale_mark_name_days",
        "unresolved_action_name_days",
        "valuation_scenario_count",
        "actual_risk_breach_dates",
    }
)
_REQUIRED_ARRAYS = frozenset(
    {
        "active",
        "intraday_unit_or_unresolved_boundary_mask",
        "observed",
        "shareholder_wealth_close",
        "shareholder_wealth_valid",
        "slow_values",
        "slow_valid",
        "intraday_values",
        "intraday_valid",
        "fast_present",
        "target_primary",
        "target_valid",
        "target_shareholder_midrank",
        "target_shareholder_simple_return",
        "target_shareholder_valid",
        "target_price_midrank",
        "target_price_valid",
        "raw_close",
        "action_shares_per_prior_share",
        "action_cash_per_prior_share",
        "action_session_resolved",
        "action_has_action",
        "action_successor_index",
        "action_payment_session",
        "prior_reference_close",
        "audit_eventual_survives_to_final_year",
        "target_scale_sigma",
    }
)
_V1_FAST_FILES = (
    "equity_features.npy",
    "equity_slow.npy",
    "equity_data_ready.npy",
)
_LEGACY_ROUND1_ROOT = Path(
    r"D:\quant-data\b3\processed\model_runs\v2_round1_81fe0cb_20260907T023339Z"
)
_LEGACY_ROUND1_RESULT_SHA256 = (
    "ca39f11340f13956dca11da7fb6b3a8fa0a38f8a79cb558387d81aff26bf57a0"
)
_LEGACY_ROUND1_INVENTORY_SHA256 = (
    "b5084c817fc478c2759d962af12e282c292cade6f8e715aebed290ba5500ce08"
)


@dataclass(frozen=True)
class ValidationRuntime:
    fine_epochs: int = 3
    handoff_epochs: int = 1
    gbdt_maximum_rounds: int = 3000
    gbdt_early_stopping_rounds: int = 100
    gbdt_num_threads: int = 0
    max_fit_sessions: int | None = None
    max_selection_sessions: int | None = None
    max_pretrain_fit_sessions: int | None = None
    max_pretrain_selection_sessions: int | None = None
    slow_lookback: int = 60
    pairs_per_batch: int = 8
    evaluation_batch_size: int = 1
    compile_forward: bool = True
    device: str | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.fine_epochs <= 3:
            raise ValueError("fine_epochs must be between one and three")
        if not 1 <= self.handoff_epochs <= 3:
            raise ValueError("handoff_epochs must be between one and three")
        if not 1 <= self.gbdt_maximum_rounds <= 3000:
            raise ValueError("GBDT rounds must be between one and 3000")
        if not 1 <= self.gbdt_early_stopping_rounds <= 100:
            raise ValueError("GBDT early stopping must be between one and 100 rounds")
        if self.gbdt_num_threads < 0:
            raise ValueError("GBDT thread count must be non-negative")
        for value in (
            self.max_fit_sessions,
            self.max_selection_sessions,
            self.max_pretrain_fit_sessions,
            self.max_pretrain_selection_sessions,
        ):
            if value is not None and value < _MIN_WINDOW_SESSIONS:
                raise ValueError(
                    f"bounded validation windows need at least {_MIN_WINDOW_SESSIONS} sessions"
                )
        if self.slow_lookback not in ALLOWED_LOOKBACKS:
            raise ValueError("slow_lookback must be 20, 60, or 120")
        if self.pairs_per_batch != 8:
            raise ValueError("pipeline validation requires exactly 8 date pairs")
        if self.evaluation_batch_size <= 0:
            raise ValueError("validation evaluation_batch_size must be positive")


@dataclass(frozen=True)
class PipelineValidationResult:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    inventory_path: Path
    inventory_sha256: str


def _read_store_header(root: Path) -> tuple[dict[str, object], NDArray[np.datetime64]]:
    store_root = root.resolve()
    manifest_path = store_root / "manifest.json"
    hash_path = store_root / "manifest.sha256"
    if not manifest_path.is_file() or not hash_path.is_file():
        raise FileNotFoundError("validation requires an immutable v2 store manifest")
    expected_manifest_sha = hash_path.read_text(encoding="ascii").split()[0]
    if sha256_file(manifest_path) != expected_manifest_sha:
        raise ValueError("store manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != STORE_SCHEMA:
        raise ValueError("validation requires a real v2 daily store")
    metadata = manifest.get("metadata")
    calendar_contract = (
        metadata.get("calendar_contract") if isinstance(metadata, Mapping) else None
    )
    if (
        not isinstance(calendar_contract, Mapping)
        or calendar_contract.get("schema") != "BRAZIL_RV_B3_EQUITY_SESSION_SCHEDULE_V1"
        or calendar_contract.get("schedule_source") != metadata.get("schedule_source")
    ):
        raise ValueError("store lacks its explicit session-calendar contract")
    tables = manifest.get("tables")
    calendar_record = (
        tables.get("calendar_completeness") if isinstance(tables, Mapping) else None
    )
    calendar_path = store_root / "calendar_completeness.parquet"
    if not isinstance(calendar_record, Mapping) or (
        int(calendar_record.get("rows", -1)) != 0
        or int(calendar_record.get("bytes", -1)) != calendar_path.stat().st_size
        or calendar_record.get("sha256") != sha256_file(calendar_path)
    ):
        raise ValueError("store calendar has missing or unexplained sessions")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("store has no immutable source identities")
    arrays = manifest.get("arrays")
    if not isinstance(arrays, Mapping) or not _REQUIRED_ARRAYS.issubset(arrays):
        missing = sorted(_REQUIRED_ARRAYS - set(arrays or ()))
        raise ValueError(f"store lacks validation arrays: {missing}")
    if (
        manifest.get("official_validation_accessed") is not False
        or manifest.get("test_accessed") is not False
    ):
        raise ValueError("source store records sealed-window access")
    indices = manifest.get("indices")
    if not isinstance(indices, Mapping):
        raise ValueError("store manifest lacks its immutable index inventory")
    date_record = indices.get("date_index.npy")
    isin_record = indices.get("isin_index.npy")
    date_path = store_root / "date_index.npy"
    isin_path = store_root / "isin_index.npy"
    if (
        not isinstance(date_record, Mapping)
        or not isinstance(isin_record, Mapping)
        or (
            int(date_record.get("bytes", -1)) != date_path.stat().st_size
            or date_record.get("sha256") != sha256_file(date_path)
            or int(isin_record.get("bytes", -1)) != isin_path.stat().st_size
            or isin_record.get("sha256") != sha256_file(isin_path)
        )
    ):
        raise ValueError("store axes differ from their immutable manifest")
    dates = np.load(date_path, allow_pickle=False)
    isins = np.load(isin_path, allow_pickle=False)
    if dates.dtype.kind != "M" or dates.ndim != 1:
        raise ValueError("store date index has the wrong contract")
    if isins.dtype.kind not in "US" or isins.ndim != 1:
        raise ValueError("store ISIN index has the wrong contract")
    axes = manifest.get("axes")
    date_identity = hashlib.sha256(
        json.dumps(
            [str(value) for value in dates],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    isin_identity = hashlib.sha256(
        json.dumps(
            [str(value) for value in isins],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if not isinstance(axes, Mapping) or (
        int(axes.get("date_count", -1)) != dates.size
        or int(axes.get("isin_count", -1)) != isins.size
        or axes.get("date_identity_sha256") != date_identity
        or axes.get("isin_identity_sha256") != isin_identity
    ):
        raise ValueError("store logical axis identities do not match the manifest")
    python_dates = tuple(dates.astype("datetime64[D]").astype(object).tolist())
    if (
        not python_dates
        or python_dates[0] != STORE_START
        or python_dates[-1] < DEVELOPMENT_END
    ):
        raise ValueError("store does not span the frozen v2 development foundation")
    return manifest, dates


def _external_artifact_resolutions(
    store_manifest: Mapping[str, object],
) -> list[dict[str, object]]:
    metadata = store_manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("store manifest metadata is malformed")
    records = metadata.get("v1_fast_files", [])
    if not records:
        return []
    if not isinstance(records, list) or any(
        not isinstance(record, Mapping) for record in records
    ):
        raise ValueError("external v1 fast artifact records are malformed")
    _, resolutions = resolve_external_files(records, _V1_FAST_FILES)
    return [resolution.payload() for resolution in resolutions]


def _assert_overrides_outside_store(
    store_root: Path, resolutions: Sequence[Mapping[str, object]]
) -> None:
    for resolution in resolutions:
        configured = resolution.get("override_file")
        if configured is None:
            continue
        override = Path(str(configured)).resolve(strict=True)
        if override == store_root or override.is_relative_to(store_root):
            raise ValueError(
                "data-root override must remain outside the immutable store"
            )


def _bounded(
    indices: NDArray[np.int64], maximum: int | None, *, tail: bool
) -> NDArray[np.int64]:
    if maximum is None or len(indices) <= maximum:
        return indices.copy()
    return indices[-maximum:].copy() if tail else indices[:maximum].copy()


def _dates_for_indices(
    dates: NDArray[np.datetime64], indices: Sequence[int]
) -> tuple[date, ...]:
    return tuple(
        np.asarray(dates[np.asarray(indices, dtype=np.int64)])
        .astype("datetime64[D]")
        .astype(object)
        .tolist()
    )


def _date_indices(
    dates: NDArray[np.datetime64], requested: Sequence[date]
) -> NDArray[np.int64]:
    by_date = {
        value: index
        for index, value in enumerate(
            dates.astype("datetime64[D]").astype(object).tolist()
        )
    }
    try:
        result = np.asarray([by_date[value] for value in requested], dtype=np.int64)
    except KeyError as error:
        raise ValueError(
            f"registered split date is absent from the store: {error}"
        ) from error
    if result.size < _MIN_WINDOW_SESSIONS or np.any(np.diff(result) != 1):
        raise ValueError("validation split must be a contiguous full-session axis")
    return result


def _dataset(
    store: V2Store,
    indices: NDArray[np.int64],
    *,
    stage: str,
    purpose: AccessPurpose,
    lookback: int,
    sidecars: Sequence[str],
    target_window_indices: NDArray[np.int64] | None = None,
) -> V2DailyDataset:
    requested_dates = _dates_for_indices(store.dates, indices)
    if any(value >= OFFICIAL_START for value in requested_dates):
        raise PermissionError("pipeline validation refuses every 2025/2026 session")
    return V2DailyDataset(
        store,
        indices,
        stage=stage,
        lookback=lookback,
        enabled_sidecars=sidecars,
        purpose=purpose,
        target_window_indices=target_window_indices,
    )


def _training_loaders(
    store: V2Store,
    fit_indices: NDArray[np.int64],
    selection_indices: NDArray[np.int64],
    *,
    stage: str,
    runtime: ValidationRuntime,
    sidecars: Sequence[str],
    seed: int,
    time_decay_half_life: float | None = None,
    fit_target_window_indices: NDArray[np.int64] | None = None,
) -> tuple[DataLoader[dict[str, object]], DataLoader[dict[str, object]]]:
    fit = _dataset(
        store,
        fit_indices,
        stage=stage,
        purpose="training",
        lookback=runtime.slow_lookback,
        sidecars=sidecars,
        target_window_indices=(
            fit_indices
            if fit_target_window_indices is None
            else fit_target_window_indices
        ),
    )
    selection = _dataset(
        store,
        selection_indices,
        stage=stage,
        purpose="selection",
        lookback=runtime.slow_lookback,
        sidecars=sidecars,
        target_window_indices=selection_indices,
    )
    sampler = DatePairBatchSampler(
        fit.date_indices,
        pairs_per_batch=runtime.pairs_per_batch,
        seed=seed,
        time_decay_half_life=time_decay_half_life,
        drop_last=True,
    )
    fixed_fast_name_count = stage_fast_name_count(fit, selection)
    stage_collate = partial(
        collate_v2_daily, fixed_fast_name_count=fixed_fast_name_count
    )
    return (
        DataLoader(
            fit,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=stage_collate,
        ),
        DataLoader(
            selection,
            batch_size=runtime.evaluation_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=stage_collate,
        ),
    )


def _score_loader(
    store: V2Store,
    indices: NDArray[np.int64],
    *,
    runtime: ValidationRuntime,
    sidecars: Sequence[str],
) -> DataLoader[dict[str, object]]:
    dataset = _dataset(
        store,
        indices,
        stage="evaluation",
        purpose="evaluation",
        lookback=runtime.slow_lookback,
        sidecars=sidecars,
    )
    fixed_fast_name_count = stage_fast_name_count(dataset)
    return DataLoader(
        dataset,
        batch_size=runtime.evaluation_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(
            collate_v2_daily, fixed_fast_name_count=fixed_fast_name_count
        ),
    )


def _array_record(path: Path, values: NDArray[np.generic]) -> dict[str, object]:
    return {
        "path": path.name,
        "shape": list(values.shape),
        "dtype": values.dtype.str,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _persist_score_panel(
    root: Path,
    arrays: Mapping[str, NDArray[np.generic]],
    metadata: Mapping[str, object],
) -> tuple[Path, str]:
    root.mkdir(parents=True, exist_ok=False)
    records: dict[str, dict[str, object]] = {}
    for name, raw in sorted(arrays.items()):
        values = np.asarray(raw)
        path = root / f"{name}.npy"
        np.save(path, values, allow_pickle=False)
        records[path.name] = _array_record(path, values)
    manifest_path = root / "validation_manifest.json"
    manifest_sha = write_json_atomic(
        manifest_path,
        {
            "schema": PIPELINE_SCHEMA,
            "status": "completed",
            **PIPELINE_FLAGS,
            "metadata": dict(metadata),
            "artifacts": records,
        },
    )
    return manifest_path, manifest_sha


def _mark_pipeline_manifest(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"pipeline artifact manifest must be an object: {path}")
    if (
        payload.get("official_validation_accessed") is not False
        or payload.get("test_accessed") is not False
    ):
        raise PermissionError(f"pipeline artifact touched sealed data: {path}")
    if payload.get("transfer_chronology_clean") is not True:
        raise PermissionError(
            f"pipeline artifact has contaminated or unknown transfer chronology: {path}"
        )
    payload.update(PIPELINE_FLAGS)
    return write_json_atomic(path, payload)


def _load_score_arrays(
    artifact: ScoreArtifact,
    *,
    expected_dates: NDArray[np.datetime64] | None = None,
    expected_isins: Sequence[str] | None = None,
    expected_feature_schema_sha256: str | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Verify chronology/access provenance before opening score payload arrays."""

    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("score manifest must be an object")
    if (
        manifest.get("schema") != SCORE_ARTIFACT_SCHEMA
        or manifest.get("status") != "completed"
    ):
        raise ValueError("score artifact is stale or incomplete")
    if (
        manifest.get("official_validation_accessed") is not False
        or manifest.get("test_accessed") is not False
        or manifest.get("transfer_chronology_clean") is not True
    ):
        raise PermissionError(
            "score artifact has sealed-window access or contaminated chronology"
        )
    if (
        expected_feature_schema_sha256 is not None
        and manifest.get("feature_schema_sha256") != expected_feature_schema_sha256
    ):
        raise ValueError("score feature schema differs from the canonical store")
    records = manifest.get("artifacts")
    if not isinstance(records, Mapping):
        raise ValueError("score manifest lacks its artifact inventory")
    if sha256_file(artifact.manifest_path) != artifact.manifest_sha256:
        raise ValueError("score artifact manifest identity is stale")
    for path in (
        artifact.scores_path,
        artifact.score_mask_path,
        artifact.date_index_path,
        artifact.isin_index_path,
    ):
        record = records.get(path.name)
        if not isinstance(record, Mapping) or (
            path.stat().st_size != int(record.get("bytes", -1))
            or sha256_file(path) != record.get("sha256")
        ):
            raise ValueError(f"score artifact hash mismatch: {path}")
    dates = np.asarray(
        np.load(artifact.date_index_path, allow_pickle=False), dtype="datetime64[D]"
    )
    isins = tuple(
        str(value)
        for value in np.load(artifact.isin_index_path, allow_pickle=False).tolist()
    )
    if expected_dates is not None and not np.array_equal(
        dates, np.asarray(expected_dates, dtype="datetime64[D]")
    ):
        raise ValueError("score date axis differs from the requested evaluation window")
    if expected_isins is not None and isins != tuple(
        str(value) for value in expected_isins
    ):
        raise ValueError("score security axis differs from the canonical store")
    scores = np.asarray(
        np.load(artifact.scores_path, allow_pickle=False), dtype=np.float32
    )
    score_mask = np.asarray(
        np.load(artifact.score_mask_path, allow_pickle=False), dtype=np.bool_
    )
    if scores.shape != score_mask.shape or scores.shape[:2] != (
        len(dates),
        len(isins),
    ):
        raise ValueError("score payload and immutable axes are misaligned")
    return scores, score_mask


def _window_target_mask(
    values: NDArray[np.bool_],
    indices: NDArray[np.int64],
    *,
    target_window_indices: NDArray[np.int64] | None = None,
) -> NDArray[np.bool_]:
    mask = np.asarray(values, dtype=np.bool_).copy()
    if (
        mask.ndim != 3
        or mask.shape[0] != len(indices)
        or mask.shape[-1] != len(HORIZONS)
    ):
        raise ValueError("window target mask has the wrong shape")
    window = indices if target_window_indices is None else target_window_indices
    index_set = frozenset(int(value) for value in window)
    for row, date_index in enumerate(indices):
        for horizon_index, horizon in enumerate(HORIZONS):
            if int(date_index) + horizon not in index_set:
                mask[row, :, horizon_index] = False
    return mask


def _evaluation_inputs(
    store: V2Store,
    indices: NDArray[np.int64],
    scores: NDArray[np.floating],
    score_mask: NDArray[np.bool_],
    cdi_by_index: NDArray[np.float64],
    bova11_close_by_index: NDArray[np.float64],
    bova11_binding: Mapping[str, str],
    lending_borrow: LendingBorrowPanels,
    source_hashes: Mapping[str, str],
    *,
    transfer_chronology_clean: bool,
) -> EvaluationInputs:
    if transfer_chronology_clean is not True:
        raise PermissionError(
            "pipeline validation refuses contaminated or unknown transfer chronology"
        )
    indices = np.asarray(indices, dtype=np.int64)
    if (
        indices.ndim != 1
        or not indices.size
        or (indices.size > 1 and np.any(np.diff(indices) != 1))
    ):
        raise ValueError("evaluation indices must be one nonempty contiguous window")
    scaled_target_mask = _window_target_mask(
        store.read("target_valid", indices), indices
    )
    neutral_target_mask = _window_target_mask(
        store.read(REGISTERED_PRIMARY_TARGET_MASK, indices), indices
    )
    shareholder_target_mask = _window_target_mask(
        store.read("target_shareholder_valid", indices), indices
    )
    price_target_mask = _window_target_mask(
        store.read("target_price_valid", indices), indices
    )
    scaled_targets = store.read_target(
        "target_primary", indices, valid_mask=scaled_target_mask
    )
    neutral_targets = store.read_target(
        REGISTERED_PRIMARY_TARGET, indices, valid_mask=neutral_target_mask
    )
    shareholder_targets = store.read_target(
        "target_shareholder_midrank", indices, valid_mask=shareholder_target_mask
    )
    shareholder_returns = store.read_target(
        "target_shareholder_simple_return",
        indices,
        valid_mask=shareholder_target_mask,
    )
    price_targets = store.read_target(
        "target_price_midrank", indices, valid_mask=price_target_mask
    )
    axes = store.manifest.get("axes")
    if not isinstance(axes, Mapping):
        raise ValueError("store manifest lacks its canonical axes")
    calendar_sha = axes.get("date_identity_sha256")
    if not isinstance(calendar_sha, str):
        raise ValueError("store manifest lacks its canonical calendar identity")
    cdi = np.asarray(cdi_by_index[indices], dtype=np.float64)
    if not np.isfinite(cdi).all():
        raise ValueError("development CDI is incomplete for the evaluation window")
    feature_names = store.manifest.get("feature_names")
    if not isinstance(feature_names, Mapping) or not isinstance(
        feature_names.get("slow"), list
    ):
        raise ValueError("store manifest lacks ordered slow-feature names")
    metadata = store.manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("store manifest metadata is malformed")
    action_terms_source = metadata.get("action_terms_source")
    schedule_source = metadata.get("schedule_source")
    if not isinstance(action_terms_source, str) or not isinstance(schedule_source, str):
        raise ValueError("store lacks action and schedule source-tier labels")
    action_contract = metadata.get("corporate_action_contract")
    if not isinstance(action_contract, Mapping) or (
        action_contract.get("stored_action_arrays")
        != "retrospective outcome/accounting terms"
    ):
        raise ValueError(
            "evaluator requires retrospective outcome/accounting action arrays"
        )
    slow_names = tuple(str(value) for value in feature_names["slow"])
    diagnostic_names = (
        "yang_zhang_vol_20",
        "beta_60",
        "log_volume_mean_20",
        "momentum_12_1",
        "log_return_5",
    )
    # Store row t is the canonical decision snapshot. Its internally derived
    # daily fields were sourced from t-1 during construction, so consumers
    # must not apply another shift here.
    slow_prior = np.asarray(store.read("slow_values", indices))
    slow_valid = np.asarray(store.read("slow_valid", indices), dtype=np.bool_)
    prior_feature_values = {
        name: np.where(
            slow_valid[..., slow_names.index(name)],
            slow_prior[..., slow_names.index(name)],
            np.nan,
        )
        for name in diagnostic_names
    }
    history_index = slow_names.index("observed_history_age_sessions")
    history_valid = slow_valid[..., history_index]
    transformed_history_age = np.asarray(
        slow_prior[..., history_index], dtype=np.float64
    )
    history_age_sessions = np.where(
        history_valid,
        np.expm1(np.clip(transformed_history_age, 0.0, 1.0) * np.log1p(252.0)),
        np.nan,
    )
    source_archive_present: dict[str, NDArray[np.bool_]] = {}
    source_feature_valid: dict[str, NDArray[np.bool_]] = {}
    for group in SIDECAR_FEATURES:
        names = feature_names.get(f"sidecar_{group}")
        if not isinstance(names, list) or not names:
            continue
        valid = np.asarray(
            store.read(f"sidecar_{group}_valid", indices), dtype=np.bool_
        )
        ages = np.asarray(
            store.read(f"sidecar_{group}_age_sessions", indices), dtype=np.float64
        )
        source_archive_present[group] = np.any(ages >= 0.0, axis=-1)
        source_feature_valid[group] = np.any(valid, axis=-1)
    initial_reference_price = np.asarray(
        store.read("prior_reference_close", np.asarray([indices[0]], dtype=np.int64))[
            0
        ],
        dtype=np.float64,
    )
    action_payment_session = np.asarray(
        store.read("action_payment_session", indices), dtype=np.int64
    )
    known_payment = action_payment_session >= 0
    action_payment_session[known_payment] -= int(indices[0])
    beta_start = max(0, int(indices[0]) - HEDGE_BETA_MAX_AGE)
    beta_dates = store.dates[beta_start : int(indices[-1]) + 1]
    beta_panel = load_hedge_beta_sidecar(
        Path(bova11_binding["hedge_beta_root"]),
        expected_manifest_sha256=bova11_binding["hedge_beta_manifest_sha256"],
        expected_store_manifest_sha256=sha256_file(store.root / "manifest.json"),
        expected_bova11_manifest_sha256=bova11_binding["manifest_sha256"],
        canonical_dates=beta_dates.astype(object).tolist(),
        isins=store.isins,
    )
    history_count = int(indices[0]) - beta_start
    return EvaluationInputs(
        dates=_dates_for_indices(store.dates, indices),
        session_indices=indices.copy(),
        calendar_identity_sha256=calendar_sha,
        scores=np.asarray(scores),
        score_mask=np.asarray(score_mask, dtype=np.bool_),
        scaled_midrank_targets=scaled_targets,
        scaled_target_mask=scaled_target_mask,
        neutral_midrank_targets=neutral_targets,
        neutral_target_mask=neutral_target_mask,
        shareholder_midrank_targets=shareholder_targets,
        shareholder_simple_returns=shareholder_returns,
        shareholder_target_mask=shareholder_target_mask,
        price_midrank_targets=price_targets,
        price_target_mask=price_target_mask,
        active=np.asarray(store.read("active", indices), dtype=np.bool_),
        raw_close=store.read("raw_close", indices),
        action_shares_per_prior_share=store.read(
            "action_shares_per_prior_share", indices
        ),
        action_cash_per_prior_share=store.read("action_cash_per_prior_share", indices),
        action_session_resolved=np.asarray(
            store.read("action_session_resolved", indices), dtype=np.bool_
        ),
        action_has_action=np.asarray(
            store.read("action_has_action", indices), dtype=np.bool_
        ),
        action_successor_index=np.asarray(
            store.read("action_successor_index", indices), dtype=np.int64
        ),
        action_payment_session=action_payment_session,
        security_ids=store.isins,
        target_scale_sigma=store.read("target_scale_sigma", indices),
        prior_feature_values=prior_feature_values,
        cdi_returns=cdi,
        transfer_chronology_clean=transfer_chronology_clean,
        action_terms_source=action_terms_source,
        schedule_source=schedule_source,
        source_artifact_hashes={
            **source_hashes,
            "lending_archive_manifest": lending_borrow.manifest_sha256,
            "lending_archive_balances": lending_borrow.balance_sha256,
            "lending_archive_rates": lending_borrow.rate_sha256,
        },
        history_age_sessions=history_age_sessions,
        source_archive_present=source_archive_present or None,
        source_feature_valid=source_feature_valid or None,
        initial_reference_price=initial_reference_price,
        initial_unresolved_action=(
            ~np.asarray(
                store.read(
                    "action_session_resolved", np.asarray([int(indices[0]) - 1])
                ),
                dtype=np.bool_,
            )[0]
            if int(indices[0]) > 0
            else np.zeros(len(store.isins), dtype=np.bool_)
        ),
        eventual_survives_to_final_year=np.asarray(
            store.read("audit_eventual_survives_to_final_year", indices),
            dtype=np.bool_,
        ),
        action_alignment="retrospective",
        annual_borrow_rate_by_name=np.asarray(
            lending_borrow.annual_taker_rate[indices], dtype=np.float64
        ),
        borrow_rate_imputed=np.asarray(
            lending_borrow.rate_imputed[indices], dtype=np.bool_
        ),
        borrow_rate_placeholder=np.asarray(
            lending_borrow.rate_placeholder[indices], dtype=np.bool_
        ),
        shortable_by_borrow_source={
            name: np.asarray(values[indices], dtype=np.bool_)
            for name, values in lending_borrow.availability.items()
        },
        borrow_source_label=lending_borrow.source_label,
        bova11_close=np.asarray(bova11_close_by_index[indices], dtype=np.float64),
        bova11_manifest_sha256=bova11_binding["manifest_sha256"],
        bova11_data_sha256=bova11_binding["data_sha256"],
        hedge_beta=beta_panel.values[history_count:],
        hedge_beta_valid=beta_panel.valid[history_count:],
        hedge_beta_history=(
            beta_panel.values[:history_count],
            beta_panel.valid[:history_count],
        ),
        hedge_beta_manifest_sha256=beta_panel.manifest_sha256,
        initial_hedge_reference_price=(
            float(bova11_close_by_index[int(indices[0]) - 1])
            if indices[0] > 0
            else np.nan
        ),
        neutral_target_fallback_flags=store.neutral_target_fallback_flags(indices),
    )


def _evaluate_and_write(
    *,
    store: V2Store,
    indices: NDArray[np.int64],
    scores: NDArray[np.floating],
    score_mask: NDArray[np.bool_],
    cdi_by_index: NDArray[np.float64],
    bova11_close_by_index: NDArray[np.float64],
    bova11_binding: Mapping[str, str],
    lending_borrow: LendingBorrowPanels,
    source_hashes: Mapping[str, str],
    window_name: str,
    path: Path,
    transfer_chronology_clean: bool = True,
) -> tuple[EvaluationResult, str]:
    result = evaluate_scores(
        _evaluation_inputs(
            store,
            indices,
            scores,
            score_mask,
            cdi_by_index,
            bova11_close_by_index,
            bova11_binding,
            lending_borrow,
            source_hashes,
            transfer_chronology_clean=transfer_chronology_clean,
        ),
        window_name=window_name,
    )
    result.report.update(PIPELINE_FLAGS)
    return result, write_json_atomic(path, result.report)


def _evaluation_summary(
    result: EvaluationResult, report_path: Path, report_sha256: str
) -> dict[str, object]:
    economics = result.report["economics"]
    assert isinstance(economics, Mapping)
    summaries = economics["summaries"]
    assert isinstance(summaries, list)
    headline = next(
        row
        for row in summaries
        if isinstance(row, Mapping)
        and row.get("cost_bps_per_side") == 4.0
        and row.get("annual_borrow_rate") == 0.02
        and row.get("borrow_source") == "borrow_balance"
        and row.get("volatility_balanced_entries") is True
        and row.get("beta_hedge") is True
    )
    headline_report = economics["headline"]
    assert isinstance(headline_report, Mapping)
    diagnostics = result.report["diagnostics"]
    assert isinstance(diagnostics, Mapping)
    realized_beta = diagnostics["realized_beta"]
    assert isinstance(realized_beta, Mapping)
    return {
        "report": str(report_path),
        "report_sha256": report_sha256,
        "mean_daily_primary_neutral_target_ic": result.report[
            "mean_daily_primary_neutral_target_ic"
        ],
        "daily_primary_neutral_target_ic": [
            _finite_or_none(value) for value in result.daily_primary_ic
        ],
        "legacy_scaled_target_ic": {
            f"D{row['horizon_sessions']}": row["legacy_scaled_target_ic"]
            for row in result.report["horizon_readouts"]
        },
        "headline_economics": {**dict(headline), **dict(headline_report)},
        "borrow_cell_economics": [
            dict(row)
            for row in summaries
            if isinstance(row, Mapping)
            and row.get("scenario")
            in {"borrow_strict", "borrow_balance", "borrow_open"}
        ],
        "realized_beta_after_hedge": dict(realized_beta),
        "lending_coverage": diagnostics["lending_coverage"],
    }


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _legacy_round1_baseline_identity(
    baseline_records: Sequence[Mapping[str, object]],
    *,
    prior_root: Path,
    prior_result_sha256: str,
    prior_inventory_sha256: str,
) -> dict[str, object]:
    """Compare every legacy naive IC with the hash-bound rev-2 Round-1 artifact."""

    root, root_resolution = resolve_external_root(prior_root)
    result_path = root / "round1_result.json"
    inventory_path = root / "artifact_inventory.json"
    if sha256_file(result_path) != prior_result_sha256.casefold():
        raise ValueError("legacy Round-1 result SHA-256 mismatch")
    if sha256_file(inventory_path) != prior_inventory_sha256.casefold():
        raise ValueError("legacy Round-1 inventory SHA-256 mismatch")
    mismatches: list[str] = []
    compared = 0
    for record in baseline_records:
        name = str(record.get("name"))
        fold = str(record.get("fold"))
        evaluation = record.get("evaluation")
        current = (
            evaluation.get("legacy_scaled_target_ic")
            if isinstance(evaluation, Mapping)
            else None
        )
        path = root / "baselines" / name / fold / "evaluation.json"
        if not path.is_file() or not isinstance(current, Mapping):
            mismatches.append(f"{name}:{fold}:missing")
            continue
        prior = json.loads(path.read_text(encoding="utf-8"))
        if (
            prior.get("official_validation_accessed") is not False
            or prior.get("test_accessed") is not False
        ):
            raise PermissionError("legacy Round-1 baseline has protected access")
        rows = prior.get("horizon_readouts")
        if not isinstance(rows, list):
            mismatches.append(f"{name}:{fold}:horizon_rows_missing")
            continue
        expected = {
            f"D{row['horizon_sessions']}": row.get("mean_scaled_target_spearman_ic")
            for row in rows
            if isinstance(row, Mapping) and "horizon_sessions" in row
        }
        compared += len(expected)
        if dict(current) != expected:
            mismatches.append(f"{name}:{fold}:legacy_ic_differs")
    return {
        "passed": not mismatches and compared == 15 * len(HORIZONS),
        "comparison_count": compared,
        "expected_comparison_count": 15 * len(HORIZONS),
        "mismatches": mismatches,
        "prior_root": str(root),
        "prior_result_sha256": prior_result_sha256.casefold(),
        "prior_inventory_sha256": prior_inventory_sha256.casefold(),
        "root_resolution": root_resolution.payload(),
    }


def _development_acceptance(
    *,
    baseline_records: Sequence[Mapping[str, object]],
    gbdt_records: Sequence[Mapping[str, object]],
    action_terms_source: object,
    schedule_source: object,
    native_fast_audit_passed: bool = True,
    legacy_round1_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    expected = {
        (fold, name) for fold in ("F1", "F2", "F3") for name in _BASELINE_SIGNAL_SIGNS
    }
    received = {
        (str(record.get("fold")), str(record.get("name")))
        for record in baseline_records
    }
    violations: list[str] = []
    if received != expected:
        violations.append("baseline_roster_or_fold_coverage_mismatch")

    pooled_ic: dict[str, float | None] = {}
    for name in _NAIVE_SIGNAL_BASELINES:
        values: list[float] = []
        for record in baseline_records:
            if record.get("name") != name:
                continue
            evaluation = record.get("evaluation")
            if not isinstance(evaluation, Mapping):
                continue
            daily = evaluation.get("daily_primary_neutral_target_ic")
            if not isinstance(daily, list):
                continue
            values.extend(float(value) for value in daily if value is not None)
        point = float(np.mean(values)) if values else None
        pooled_ic[name] = point
        if point is None:
            violations.append(f"{name}_pooled_ic_undefined")
        elif abs(point) >= 0.10:
            violations.append(f"{name}_absolute_pooled_ic_not_below_0_10")

    inverse_volatility_by_fold: dict[str, float | None] = {}
    for fold in ("F1", "F2", "F3"):
        values: list[float] = []
        for record in baseline_records:
            if (
                record.get("name") != "inverse_volatility_20"
                or record.get("fold") != fold
            ):
                continue
            evaluation = record.get("evaluation")
            daily = (
                evaluation.get("daily_primary_neutral_target_ic")
                if isinstance(evaluation, Mapping)
                else None
            )
            if isinstance(daily, list):
                values.extend(float(value) for value in daily if value is not None)
        point = float(np.mean(values)) if values else None
        inverse_volatility_by_fold[fold] = point
        if point is None or abs(point) >= 0.02:
            violations.append(
                f"inverse_volatility_20_{fold}_absolute_neutral_ic_not_below_0_02"
            )

    reversal_records = [
        record for record in baseline_records if record.get("name") == "reversal_5"
    ]
    reversal_definition_ok = bool(reversal_records) and all(
        record.get("signal_definition_sign") == -1.0 for record in reversal_records
    )
    if not reversal_definition_ok:
        violations.append("reversal_5_definition_is_not_negative_five_session_return")

    economics_rows: list[dict[str, object]] = []
    unresolved_stale_fractions: list[float] = []
    for record in (*baseline_records, *gbdt_records):
        evaluation = record.get("evaluation")
        if not isinstance(evaluation, Mapping):
            violations.append("evaluation_summary_missing")
            continue
        headline = evaluation.get("headline_economics")
        if not isinstance(headline, Mapping):
            violations.append("headline_economics_missing")
            continue
        label = f"{record.get('engine')}:{record.get('fold')}:{record.get('name', 'ensemble')}"
        gross_value = headline.get("mean_gross_fraction_nav")
        unresolved_stale_value = headline.get(
            "mean_unresolved_stale_inventory_fraction_nav"
        )
        gross = None if gross_value is None else float(gross_value)
        within_gross_band = gross is not None and 1.8 <= gross <= 2.2
        gross_deployment_label = (
            None
            if gross is None
            else "within_band"
            if within_gross_band
            else "gross_underdeployed"
            if gross < 1.8
            else "gross_overdeployed"
        )
        unresolved_stale = (
            None
            if unresolved_stale_value is None
            else abs(float(unresolved_stale_value))
        )
        economics_rows.append(
            {
                "evaluation": label,
                "mean_deployed_gross_fraction_nav": gross,
                "minimum_deployed_gross_fraction_nav": headline.get(
                    "minimum_gross_fraction_nav"
                ),
                "maximum_deployed_gross_fraction_nav": headline.get(
                    "maximum_gross_fraction_nav"
                ),
                "mean_whole_book_gross_fraction_nav": headline.get(
                    "mean_gross_fraction_nav_including_hedge"
                ),
                "mean_absolute_whole_book_net_fraction_nav": headline.get(
                    "mean_absolute_net_fraction_nav_including_hedge"
                ),
                "terminal_whole_book_net_notional": headline.get(
                    "terminal_net_notional_including_hedge"
                ),
                "gross_target": 2.0,
                "within_ten_percent_of_gross_target": within_gross_band,
                "gross_deployment_label": gross_deployment_label,
                "gross_shortfall_decomposition": headline.get(
                    "gross_shortfall_decomposition"
                ),
                "entry_defect_signatures": headline.get("entry_defect_signatures"),
                "unresolved_inventory_count": headline.get(
                    "unresolved_inventory_count"
                ),
                "unresolved_inventory_notional": headline.get(
                    "unresolved_inventory_notional"
                ),
                "terminal_nav": headline.get("terminal_nav"),
                "mean_unresolved_stale_inventory_fraction_nav": unresolved_stale,
                "terminal_unresolved_inventory_fraction_nav": headline.get(
                    "terminal_unresolved_inventory_fraction_nav"
                ),
                "terminal_unresolved_reason_breakdown": headline.get(
                    "terminal_unresolved_reason_breakdown"
                ),
                "zero_entry_days_by_cause": headline.get("zero_entry_days_by_cause"),
                "blocked_entry_candidates_by_cause": headline.get(
                    "blocked_entry_candidates_by_cause"
                ),
                "risk_trim_days_by_type": headline.get("risk_trim_days_by_type"),
                "risk_trim_notional_by_type": headline.get(
                    "risk_trim_notional_by_type"
                ),
                "intended_entry_count": headline.get("intended_entry_count"),
                "intended_entry_fill_rate": headline.get("intended_entry_fill_rate"),
                "mean_pending_exit_age_sessions": headline.get(
                    "mean_pending_exit_age_sessions"
                ),
                "cancellations_by_reason": headline.get("cancellations_by_reason"),
                "terminal_settlement_convention": headline.get(
                    "terminal_settlement_convention"
                ),
                "settlement_grace_sessions": headline.get("settlement_grace_sessions"),
                "ineligible_hold_sessions": headline.get("ineligible_hold_sessions"),
                "settlement_haircut": headline.get("settlement_haircut"),
                "terminal_settlement_count": headline.get("terminal_settlement_count"),
                "terminal_settlement_notional": headline.get(
                    "terminal_settlement_notional"
                ),
                "terminal_settlement_notional_fraction_nav": headline.get(
                    "terminal_settlement_notional_fraction_nav"
                ),
                "settled_then_printed_count": headline.get(
                    "settled_then_printed_count"
                ),
                "terminal_settlement_economics_unresolved": headline.get(
                    "terminal_settlement_economics_unresolved"
                ),
                "exit_instructions_by_cause": headline.get(
                    "exit_instructions_by_cause"
                ),
                "ineligible_held_sessions_by_cause": headline.get(
                    "ineligible_held_sessions_by_cause"
                ),
                "ineligible_exit_reeligible_within_10_sessions_share": headline.get(
                    "ineligible_exit_reeligible_within_10_sessions_share"
                ),
                "realized_beta_after_hedge": evaluation.get(
                    "realized_beta_after_hedge"
                ),
                "mean_volatility_quota_by_quintile": headline.get(
                    "mean_volatility_quota_by_quintile"
                ),
                "mean_entry_eligible_name_count": headline.get(
                    "mean_entry_eligible_name_count"
                ),
                "mean_volatility_group_size_by_quintile": headline.get(
                    "mean_volatility_group_size_by_quintile"
                ),
                "minimum_volatility_group_size_by_quintile": headline.get(
                    "minimum_volatility_group_size_by_quintile"
                ),
                "mean_volatility_occupancy_long_by_quintile": headline.get(
                    "mean_volatility_occupancy_long_by_quintile"
                ),
                "mean_volatility_occupancy_short_by_quintile": headline.get(
                    "mean_volatility_occupancy_short_by_quintile"
                ),
                "mean_absolute_volatility_occupancy_deviation_long": headline.get(
                    "mean_absolute_volatility_occupancy_deviation_long"
                ),
                "mean_absolute_volatility_occupancy_deviation_short": headline.get(
                    "mean_absolute_volatility_occupancy_deviation_short"
                ),
                "spilled_entries_long_by_quintile": headline.get(
                    "spilled_entries_long_by_quintile"
                ),
                "spilled_entries_short_by_quintile": headline.get(
                    "spilled_entries_short_by_quintile"
                ),
                "imputed_rate_share_of_short_notional": headline.get(
                    "imputed_rate_share_of_short_notional"
                ),
                "placeholder_rate_share_of_short_notional": headline.get(
                    "placeholder_rate_share_of_short_notional"
                ),
                "placeholder_priced_session_count": headline.get(
                    "placeholder_priced_session_count"
                ),
                "hedge_notional_cap_nav": headline.get("hedge_notional_cap_nav"),
                "hedge_capped_session_count": headline.get(
                    "hedge_capped_session_count"
                ),
                "maximum_absolute_hedge_fraction_nav": headline.get(
                    "maximum_absolute_hedge_fraction_nav"
                ),
                "lending_coverage": evaluation.get("lending_coverage"),
                "borrow_cell_economics": evaluation.get("borrow_cell_economics"),
            }
        )
        if (
            headline.get("terminal_settlement_convention")
            != "last_mark_after_10_sessions"
        ):
            violations.append(f"{label}_terminal_settlement_convention_missing")
        if headline.get("ineligible_hold_sessions") != 5:
            violations.append(f"{label}_ineligible_hold_sessions_not_five")
        signatures = headline.get("entry_defect_signatures")
        decomposition = headline.get("gross_shortfall_decomposition")
        signature_limits = {
            "D1_entry_pending_printed_unblocked_unfilled": 0,
            "D2_entry_fill_quantity_short": 0,
            "D3_blocked_open_slots": 0,
            "D4_cap_block_defects": 0,
            "D5_ineligible_exit_within_hold_window": 0,
        }
        signatures_pass = isinstance(signatures, Mapping) and all(
            signatures.get(name) == limit for name, limit in signature_limits.items()
        )
        if (
            gross is None
            or not 1.5 <= gross <= 2.25
            or not signatures_pass
            or not isinstance(decomposition, Mapping)
        ):
            violations.append(f"{label}_deployed_gross_defect_or_outside_hard_bounds")
        if unresolved_stale is None:
            violations.append(f"{label}_unresolved_stale_fraction_missing")
        elif unresolved_stale >= 0.02:
            violations.append(f"{label}_mean_unresolved_stale_fraction_not_below_0_02")
        else:
            unresolved_stale_fractions.append(unresolved_stale)

        quota = headline.get("mean_volatility_quota_by_quintile")
        long_occupancy = headline.get("mean_volatility_occupancy_long_by_quintile")
        short_occupancy = headline.get("mean_volatility_occupancy_short_by_quintile")
        if not all(
            isinstance(values, list) and len(values) == 5
            for values in (quota, long_occupancy, short_occupancy)
        ):
            violations.append(f"{label}_volatility_occupancy_missing")
        else:
            for side in ("long", "short"):
                raw_deviation = headline.get(
                    f"mean_absolute_volatility_occupancy_deviation_{side}"
                )
                if raw_deviation is None or float(raw_deviation) > 2.0:
                    violations.append(
                        f"{label}_{side}_mean_volatility_quintile_occupancy_outside_2"
                    )

    realized_beta_by_control: dict[str, dict[str, float | None]] = {
        name: {} for name in _BASELINE_SIGNAL_SIGNS
    }
    for record in baseline_records:
        name = str(record.get("name"))
        fold = str(record.get("fold"))
        evaluation = record.get("evaluation")
        diagnostic = (
            evaluation.get("realized_beta_after_hedge")
            if isinstance(evaluation, Mapping)
            else None
        )
        raw_beta = (
            diagnostic.get("slope_beta") if isinstance(diagnostic, Mapping) else None
        )
        realized_beta_by_control[name][fold] = (
            None if raw_beta is None else float(raw_beta)
        )
    for name, folds in realized_beta_by_control.items():
        passing = sum(
            value is not None and abs(value) <= 0.30 for value in folds.values()
        )
        if passing < 2:
            violations.append(f"{name}_realized_beta_after_hedge_fewer_than_two_folds")

    mean_unresolved_stale = (
        float(np.mean(unresolved_stale_fractions))
        if unresolved_stale_fractions
        else None
    )
    if action_terms_source != "inferred_cotahist_dismes_v1":
        violations.append("action_terms_source_is_not_development_inference_tier")
    if schedule_source != "reconstructed_v1":
        violations.append("schedule_source_is_not_reconstructed_v1")
    if not native_fast_audit_passed:
        violations.append("independent_native_fast_20x20_audit_missing_or_failed")
    if (
        legacy_round1_identity is not None
        and legacy_round1_identity.get("passed") is not True
    ):
        violations.append("legacy_scaled_target_ic_differs_from_round1")

    return {
        "status": (
            "development_grade_inferred_actions" if not violations else "unsupported"
        ),
        "reasons": violations,
        "labels": {
            "action_terms_source": action_terms_source,
            "schedule_source": schedule_source,
            "economics_tier": "development_grade_close_proxy",
            "terminal_settlement_convention": "last_mark_after_10_sessions",
        },
        "sanity_bounds": {
            "naive_absolute_pooled_ic_strictly_below": 0.10,
            "reversal_5_definition_sign": -1.0,
            "gross_target": 2.0,
            "gross_relative_tolerance": 0.10,
            "gross_hard_floor": 1.5,
            "gross_hard_ceiling": 2.25,
            "realized_beta_after_hedge_absolute_bound_in_at_least_two_folds": 0.30,
            "mean_volatility_quintile_occupancy_absolute_slot_tolerance": 2.0,
            "entry_defect_signature_limits": {
                "D1_entry_pending_printed_unblocked_unfilled": 0,
                "D2_entry_fill_quantity_short": 0,
                "D3_blocked_open_slots": 0,
                "D4_cap_block_defects": 0,
                "D5_ineligible_exit_within_hold_window": 0,
            },
            "per_evaluation_mean_unresolved_stale_inventory_fraction_strictly_below": 0.02,
            "terminal_settlement_economics_unresolved_fraction_nav": 0.15,
        },
        "naive_pooled_primary_neutral_target_ic": pooled_ic,
        "inverse_volatility_neutral_ic_by_fold": inverse_volatility_by_fold,
        "inverse_volatility_absolute_neutral_ic_strictly_below": 0.02,
        "realized_beta_after_hedge_by_control": realized_beta_by_control,
        "reversal_5_definition_negative_signed": reversal_definition_ok,
        "economics_by_evaluation": economics_rows,
        "mean_unresolved_stale_inventory_fraction_nav_across_evaluations": (
            mean_unresolved_stale
        ),
        "unsupported_for_research_claims": [
            "verified_contractual_action_terms",
            "auction_execution_marks",
            "historically_executable_borrow",
        ],
        "independent_native_fast_20x20_audit_passed": native_fast_audit_passed,
        "legacy_round1_baseline_identity": (
            None if legacy_round1_identity is None else dict(legacy_round1_identity)
        ),
    }


def _verify_native_fast_audit(
    path: Path | None,
    *,
    expected_sha256: str | None,
    store_manifest_sha256: str,
) -> dict[str, object] | None:
    if path is None and expected_sha256 is None:
        return None
    if path is None or expected_sha256 is None:
        raise ValueError("native-fast audit requires both path and SHA-256")
    source = path.resolve(strict=True)
    if sha256_file(source) != expected_sha256.casefold():
        raise ValueError("native-fast audit SHA-256 mismatch")
    payload = json.loads(source.read_text(encoding="utf-8"))
    store = payload.get("store") if isinstance(payload, Mapping) else None
    comparison = payload.get("comparison") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != "BRAZIL_RV_V2_NATIVE_FAST_RAW_AUDIT_V1"
        or payload.get("status") != "passed"
        or payload.get("research_claim") is not False
        or payload.get("official_validation_accessed") is not False
        or payload.get("test_accessed") is not False
        or payload.get("action_terms_source") != "inferred_cotahist_dismes_v1"
        or payload.get("schedule_source") != "reconstructed_v1"
        or not isinstance(store, Mapping)
        or store.get("manifest_sha256") != store_manifest_sha256
        or not isinstance(comparison, Mapping)
        or comparison.get("feature_mask_exact") is not True
        or comparison.get("patch_mask_exact") is not True
        or comparison.get("age_valid_exact") is not True
        or comparison.get("independent_formula_implementation") is not True
    ):
        raise ValueError("native-fast audit is not a passing exact-store 20x20 audit")
    selection = payload.get("selection")
    if (
        not isinstance(selection, Mapping)
        or selection.get("name_count") != 20
        or selection.get("session_count") != 20
    ):
        raise ValueError(
            "native-fast audit does not cover exactly 20 names x 20 sessions"
        )
    return {"path": str(source), "sha256": expected_sha256.casefold()}


def _slow_feature_count(store: V2Store, sidecars: Sequence[str]) -> int:
    width = int(store.array_shape("slow_values")[-1])
    for group in sidecars:
        width += int(store.array_shape(f"sidecar_{group}_values")[-1])
        store.array_shape(f"sidecar_{group}_valid")
    return width


def _gbdt_features(
    store: V2Store,
    indices: NDArray[np.int64],
    sidecars: Sequence[str],
) -> NDArray[np.float32]:
    if np.any(indices < 0):
        raise ValueError("GBDT rows are outside the canonical store")
    slow = read_scalar_feature_view(
        store,
        indices,
        ("slow", *(f"sidecar_{group}" for group in sidecars)),
    )
    current = read_scalar_feature_view(store, indices, ("intraday",))
    fast_present = np.asarray(store.read("fast_present", indices), dtype=np.bool_)
    return np.concatenate(
        (
            assemble_gbdt_scalar_view(slow, label="slow"),
            assemble_gbdt_scalar_view(current, label="intraday"),
            fast_present[..., None],
        ),
        axis=-1,
        dtype=np.float32,
    )


def _gbdt_feature_names(store: V2Store, sidecars: Sequence[str]) -> tuple[str, ...]:
    slow_names = scalar_feature_names(
        store, ("slow", *(f"sidecar_{group}" for group in sidecars))
    )
    intraday_names = scalar_feature_names(store, ("intraday",))
    result = [
        *gbdt_scalar_feature_names(slow_names),
        *gbdt_scalar_feature_names(intraday_names),
    ]
    result.append("fast_present")
    if not all(isinstance(value, str) and value for value in result):
        raise ValueError("store feature names are malformed")
    return tuple(result)


def _train_once(
    *,
    store: V2Store,
    fit_indices: NDArray[np.int64],
    selection_indices: NDArray[np.int64],
    stage: str,
    seed: int,
    fold: str,
    output_dir: Path,
    model_config: ModelConfig,
    maximum_epochs: int,
    runtime: ValidationRuntime,
    sidecars: Sequence[str],
    fit_target_window_indices: NDArray[np.int64] | None = None,
    pretrain_checkpoint: Path | None = None,
    expected_pretrain_sha256: str | None = None,
) -> StageTrainingResult:
    train_loader, selection_loader = _training_loaders(
        store,
        fit_indices,
        selection_indices,
        stage={"P": "pretrain", "F": "finetune", "J": "joint"}[stage],
        runtime=runtime,
        sidecars=sidecars,
        seed=seed,
        time_decay_half_life=model_config.time_decay_half_life_sessions,
        fit_target_window_indices=fit_target_window_indices,
    )
    result = train_stage(
        stage=stage,
        seed=seed,
        fold=fold,
        train_loader=train_loader,
        selection_loader=selection_loader,
        output_dir=output_dir,
        model_config=model_config,
        pretrain_checkpoint=pretrain_checkpoint,
        expected_pretrain_sha256=expected_pretrain_sha256,
        maximum_epochs=maximum_epochs,
        device=None if runtime.device is None else torch.device(runtime.device),
    )
    _mark_pipeline_manifest(result.manifest_path)
    return result


def _score_once(
    *,
    store: V2Store,
    indices: NDArray[np.int64],
    checkpoint: Path,
    model_config: ModelConfig,
    output_dir: Path,
    runtime: ValidationRuntime,
    sidecars: Sequence[str],
) -> ScoreArtifact:
    result = score_checkpoint_artifact(
        checkpoint=checkpoint,
        model_config=model_config,
        loader=_score_loader(store, indices, runtime=runtime, sidecars=sidecars),
        output_dir=output_dir,
        expected_checkpoint_sha256=sha256_file(checkpoint),
        device=None if runtime.device is None else torch.device(runtime.device),
    )
    manifest_sha256 = _mark_pipeline_manifest(result.manifest_path)
    return replace(result, manifest_sha256=manifest_sha256)


def _run_baselines(
    *,
    store: V2Store,
    fold_indices: Mapping[str, NDArray[np.int64]],
    cdi_by_index: NDArray[np.float64],
    bova11_close_by_index: NDArray[np.float64],
    bova11_binding: Mapping[str, str],
    lending_borrow: LendingBorrowPanels,
    root: Path,
    source_hashes: Mapping[str, str],
) -> list[dict[str, object]]:
    first_index = min(int(indices[0]) for indices in fold_indices.values())
    last_index = max(int(indices[-1]) for indices in fold_indices.values())
    baseline_start = max(0, first_index - 253)
    baseline_indices = np.arange(baseline_start, last_index + 1, dtype=np.int64)
    panels = build_store_baselines(store, baseline_indices)
    records: list[dict[str, object]] = []
    for fold, indices in fold_indices.items():
        for name, panel in sorted(panels.items()):
            assert isinstance(panel, BaselinePanel)
            artifact_root = root / fold / name
            local_indices = indices - baseline_start
            manifest_path, manifest_sha = _persist_score_panel(
                artifact_root,
                {
                    "scores": panel.scores[local_indices],
                    "score_mask": panel.score_mask[local_indices],
                },
                {
                    "engine": "naive_baseline",
                    "fold": fold,
                    "baseline": name,
                    "signal_definition_sign": _BASELINE_SIGNAL_SIGNS[name],
                    "decision_source_max_session_offset": -1,
                    "date_indices": indices.tolist(),
                },
            )
            result, report_sha = _evaluate_and_write(
                store=store,
                indices=indices,
                scores=panel.scores[local_indices],
                score_mask=panel.score_mask[local_indices],
                cdi_by_index=cdi_by_index,
                bova11_close_by_index=bova11_close_by_index,
                bova11_binding=bova11_binding,
                lending_borrow=lending_borrow,
                source_hashes={
                    **source_hashes,
                    "score_manifest": manifest_sha,
                },
                window_name=fold,
                path=artifact_root / "evaluation.json",
            )
            records.append(
                {
                    "engine": "baseline",
                    "fold": fold,
                    "name": name,
                    "signal_definition_sign": _BASELINE_SIGNAL_SIGNS[name],
                    "score_manifest": str(manifest_path),
                    "score_manifest_sha256": manifest_sha,
                    "evaluation": _evaluation_summary(
                        result, artifact_root / "evaluation.json", report_sha
                    ),
                }
            )
    return records


def _run_gbdt(
    *,
    store: V2Store,
    fit_indices: Mapping[str, NDArray[np.int64]],
    fit_target_window_indices: Mapping[str, NDArray[np.int64]],
    selection_indices: Mapping[str, NDArray[np.int64]],
    evaluation_indices: Mapping[str, NDArray[np.int64]],
    cdi_by_index: NDArray[np.float64],
    bova11_close_by_index: NDArray[np.float64],
    bova11_binding: Mapping[str, str],
    lending_borrow: LendingBorrowPanels,
    root: Path,
    source_hashes: Mapping[str, str],
    runtime: ValidationRuntime,
    sidecars: Sequence[str],
) -> list[dict[str, object]]:
    config = GBDTConfig(
        maximum_rounds=runtime.gbdt_maximum_rounds,
        early_stopping_rounds=runtime.gbdt_early_stopping_rounds,
        seeds=GBDT_SEEDS,
        num_threads=runtime.gbdt_num_threads,
    )
    feature_names = _gbdt_feature_names(store, sidecars)
    records: list[dict[str, object]] = []
    for fold, train_indices in fit_indices.items():
        selection_rows = selection_indices[fold]
        evaluation_rows = evaluation_indices[fold]
        train_features = _gbdt_features(store, train_indices, sidecars)
        selection_features = _gbdt_features(store, selection_rows, sidecars)
        evaluation_features = _gbdt_features(store, evaluation_rows, sidecars)
        if train_features.shape[-1] != len(feature_names):
            raise ValueError("GBDT feature names differ from the assembled width")
        train_mask = _window_target_mask(
            store.read(REGISTERED_PRIMARY_TARGET_MASK, train_indices),
            train_indices,
            target_window_indices=fit_target_window_indices[fold],
        )
        train_targets = store.read_target(
            REGISTERED_PRIMARY_TARGET, train_indices, valid_mask=train_mask
        )
        selection_mask = _window_target_mask(
            store.read(REGISTERED_PRIMARY_TARGET_MASK, selection_rows),
            selection_rows,
        )
        selection_targets = store.read_target(
            REGISTERED_PRIMARY_TARGET, selection_rows, valid_mask=selection_mask
        )
        active = np.asarray(store.read("active", evaluation_rows), dtype=np.bool_)
        score_mask = np.repeat(active[..., None], len(HORIZONS), axis=-1)
        model = MultiHorizonGBDT(config, feature_names=feature_names)
        model.fit(
            train_features,
            train_targets,
            train_mask,
            selection_features,
            selection_targets,
            selection_mask,
            train_dates=train_indices,
            validation_dates=selection_rows,
        )
        predictions = model.predict_ranks(evaluation_features, score_mask).astype(
            np.float32, copy=False
        )
        importance = {
            name: values.tolist()
            for name, values in model.feature_importance(evaluation_features).items()
        }
        model_artifact = _persist_gbdt_models(
            model,
            root / "models" / fold,
            verification_features=evaluation_features,
            verification_mask=score_mask,
        )
        artifact_root = root / fold
        manifest_path, manifest_sha = _persist_score_panel(
            artifact_root,
            {
                "scores": predictions,
                "score_mask": score_mask,
            },
            {
                "engine": "lightgbm",
                "fold": fold,
                "seeds": list(config.seeds),
                "config": asdict(config),
                "feature_names": list(feature_names),
                "feature_importance": importance,
                "model_artifact": model_artifact,
                "protocol": "fit then purge then selection then purge then evaluation",
                "fit_date_indices": train_indices.tolist(),
                "selection_date_indices": selection_rows.tolist(),
                "evaluation_date_indices": evaluation_rows.tolist(),
                "target_value_array": REGISTERED_PRIMARY_TARGET,
                "target_validity_array": REGISTERED_PRIMARY_TARGET_MASK,
            },
        )
        result, report_sha = _evaluate_and_write(
            store=store,
            indices=evaluation_rows,
            scores=predictions,
            score_mask=score_mask,
            cdi_by_index=cdi_by_index,
            bova11_close_by_index=bova11_close_by_index,
            bova11_binding=bova11_binding,
            lending_borrow=lending_borrow,
            source_hashes={**source_hashes, "score_manifest": manifest_sha},
            window_name=fold,
            path=artifact_root / "evaluation.json",
        )
        records.append(
            {
                "engine": "gbdt",
                "fold": fold,
                "seeds": list(config.seeds),
                "score_manifest": str(manifest_path),
                "score_manifest_sha256": manifest_sha,
                "evaluation": _evaluation_summary(
                    result, artifact_root / "evaluation.json", report_sha
                ),
            }
        )
    return records


def _persist_gbdt_models(
    model: MultiHorizonGBDT,
    root: Path,
    *,
    verification_features: NDArray[np.floating],
    verification_mask: NDArray[np.bool_],
) -> dict[str, object]:
    """Hash-save, reload, and prove exact prediction equality for every member."""

    manifest_path, manifest_sha = model.save(
        root,
        metadata={"status": "completed", **PIPELINE_FLAGS},
    )
    reloaded = type(model).load(root, expected_manifest_sha256=manifest_sha)
    before_raw = model.predict_raw(verification_features)
    after_raw = reloaded.predict_raw(verification_features)
    before_rank = model.predict_ranks(verification_features, verification_mask)
    after_rank = reloaded.predict_ranks(verification_features, verification_mask)
    if not np.array_equal(before_raw, after_raw) or not np.array_equal(
        before_rank, after_rank
    ):
        raise ValueError("reloaded GBDT predictions differ from the fitted ensemble")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "model_count": len(HORIZONS) * len(model.config.seeds),
        "roundtrip_prediction_identity": True,
    }


def _run_network_smokes(
    *,
    store: V2Store,
    fit_indices: NDArray[np.int64],
    fit_target_window_indices: NDArray[np.int64],
    selection_indices: NDArray[np.int64],
    evaluation_indices: NDArray[np.int64],
    pretrain_fit_indices: NDArray[np.int64],
    pretrain_selection_indices: NDArray[np.int64],
    cdi_by_index: NDArray[np.float64],
    bova11_close_by_index: NDArray[np.float64],
    bova11_binding: Mapping[str, str],
    lending_borrow: LendingBorrowPanels,
    root: Path,
    source_hashes: Mapping[str, str],
    runtime: ValidationRuntime,
    sidecars: Sequence[str],
    seed: int,
) -> dict[str, object]:
    base_config = ModelConfig(
        slow_feature_count=_slow_feature_count(store, sidecars),
        slow_lookback=runtime.slow_lookback,
        lambda_persistence=0.0,
        compile_forward=runtime.compile_forward,
    )
    scratch = _train_once(
        store=store,
        fit_indices=fit_indices,
        selection_indices=selection_indices,
        stage="F",
        seed=seed,
        fold="F1",
        output_dir=root / "from_scratch" / "training",
        model_config=base_config,
        maximum_epochs=runtime.fine_epochs,
        runtime=runtime,
        sidecars=sidecars,
        fit_target_window_indices=fit_target_window_indices,
    )
    scratch_score = _score_once(
        store=store,
        indices=evaluation_indices,
        checkpoint=scratch.raw_patience_checkpoint,
        model_config=base_config,
        output_dir=root / "from_scratch" / "scores",
        runtime=runtime,
        sidecars=sidecars,
    )
    scratch_values, scratch_mask = _load_score_arrays(
        scratch_score,
        expected_dates=store.dates[evaluation_indices],
        expected_isins=store.isins,
        expected_feature_schema_sha256=str(store.manifest["feature_schema_sha256"]),
    )
    evaluated, report_sha = _evaluate_and_write(
        store=store,
        indices=evaluation_indices,
        scores=scratch_values,
        score_mask=scratch_mask,
        cdi_by_index=cdi_by_index,
        bova11_close_by_index=bova11_close_by_index,
        bova11_binding=bova11_binding,
        lending_borrow=lending_borrow,
        source_hashes={
            **source_hashes,
            "score_manifest": sha256_file(scratch_score.manifest_path),
        },
        window_name="F1",
        path=root / "from_scratch" / "evaluation.json",
    )

    persistence_config = replace(base_config, lambda_persistence=0.1)
    persistence = _train_once(
        store=store,
        fit_indices=fit_indices,
        selection_indices=selection_indices,
        stage="F",
        seed=seed,
        fold="F1_lambda_persistence_0_1",
        output_dir=root / "persistence_lambda_0_1" / "training",
        model_config=persistence_config,
        maximum_epochs=1,
        runtime=runtime,
        sidecars=sidecars,
        fit_target_window_indices=fit_target_window_indices,
    )
    persistence_score = _score_once(
        store=store,
        indices=evaluation_indices,
        checkpoint=persistence.raw_patience_checkpoint,
        model_config=persistence_config,
        output_dir=root / "persistence_lambda_0_1" / "scores",
        runtime=runtime,
        sidecars=sidecars,
    )
    persistence_values, persistence_mask = _load_score_arrays(
        persistence_score,
        expected_dates=store.dates[evaluation_indices],
        expected_isins=store.isins,
        expected_feature_schema_sha256=str(store.manifest["feature_schema_sha256"]),
    )
    persistence_evaluated, persistence_report_sha = _evaluate_and_write(
        store=store,
        indices=evaluation_indices,
        scores=persistence_values,
        score_mask=persistence_mask,
        cdi_by_index=cdi_by_index,
        bova11_close_by_index=bova11_close_by_index,
        bova11_binding=bova11_binding,
        lending_borrow=lending_borrow,
        source_hashes={
            **source_hashes,
            "score_manifest": sha256_file(persistence_score.manifest_path),
        },
        window_name="F1_lambda_persistence_0_1",
        path=root / "persistence_lambda_0_1" / "evaluation.json",
    )

    pretrain = _train_once(
        store=store,
        fit_indices=pretrain_fit_indices,
        selection_indices=pretrain_selection_indices,
        stage="P",
        seed=seed,
        fold="pretrain_internal",
        output_dir=root / "pretrain_handoff" / "stage_p",
        model_config=base_config,
        maximum_epochs=1,
        runtime=runtime,
        sidecars=sidecars,
    )
    pretrain_sha = sha256_file(pretrain.raw_patience_checkpoint)
    handoff = _train_once(
        store=store,
        fit_indices=fit_indices,
        selection_indices=selection_indices,
        stage="F",
        seed=seed,
        fold="F1_pretrain_handoff",
        output_dir=root / "pretrain_handoff" / "stage_f",
        model_config=base_config,
        maximum_epochs=runtime.handoff_epochs,
        runtime=runtime,
        sidecars=sidecars,
        pretrain_checkpoint=pretrain.raw_patience_checkpoint,
        expected_pretrain_sha256=pretrain_sha,
        fit_target_window_indices=fit_target_window_indices,
    )
    handoff_manifest = json.loads(handoff.manifest_path.read_text(encoding="utf-8"))
    if handoff_manifest.get("pretrain_checkpoint_sha256") != pretrain_sha:
        raise ValueError("stage-P checkpoint handoff is not hash-bound in stage F")
    return {
        "from_scratch": {
            "fold": "F1",
            "seed": seed,
            "epochs_cap": runtime.fine_epochs,
            "training_manifest": str(scratch.manifest_path),
            "score_manifest": str(scratch_score.manifest_path),
            "score_manifest_sha256": sha256_file(scratch_score.manifest_path),
            "evaluation": _evaluation_summary(
                evaluated, root / "from_scratch" / "evaluation.json", report_sha
            ),
        },
        "persistence_probe": {
            "lambda_persistence": 0.1,
            "epochs_cap": 1,
            "training_manifest": str(persistence.manifest_path),
            "score_manifest": str(persistence_score.manifest_path),
            "score_manifest_sha256": sha256_file(persistence_score.manifest_path),
            "evaluation": _evaluation_summary(
                persistence_evaluated,
                root / "persistence_lambda_0_1" / "evaluation.json",
                persistence_report_sha,
            ),
        },
        "pretrain_handoff": {
            "pretrain_epochs_cap": 1,
            "handoff_f_epochs_cap": runtime.handoff_epochs,
            "pretrain_manifest": str(pretrain.manifest_path),
            "pretrain_checkpoint": str(pretrain.raw_patience_checkpoint),
            "pretrain_checkpoint_sha256": pretrain_sha,
            "handoff_manifest": str(handoff.manifest_path),
            "handoff_manifest_sha256": sha256_file(handoff.manifest_path),
        },
    }


def _git_identity() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(
            "pipeline validation requires a clean, commit-bound implementation"
        )
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise RuntimeError("git returned an invalid commit identity")
    return {"commit": commit, "tracked_worktree_clean": True}


def _store_build_commit(store_manifest: Mapping[str, object]) -> str:
    """Return the immutable store's own build provenance.

    A sealed store and the code that later evaluates it are independently
    versioned artifacts.  Requiring their commits to be equal would make every
    registered evaluator change require an otherwise forbidden store rebuild.
    The manifest hash binds the store; this field records which code built it.
    """

    metadata = store_manifest.get("metadata")
    commit = (
        metadata.get("implementation_git_commit")
        if isinstance(metadata, Mapping)
        else None
    )
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("v2 store lacks a valid build implementation commit")
    return commit


def _validate_sidecars(
    store_manifest: Mapping[str, object], sidecars: Sequence[str]
) -> tuple[str, ...]:
    normalized = tuple(sorted(sidecars))
    if len(set(normalized)) != len(normalized):
        raise ValueError("enabled sidecar groups must be unique")
    arrays = store_manifest.get("arrays")
    feature_names = store_manifest.get("feature_names")
    if not isinstance(arrays, Mapping) or not isinstance(feature_names, Mapping):
        raise ValueError("store manifest lacks arrays or feature names")
    for group in normalized:
        if not group or not group.replace("_", "a").isalnum():
            raise ValueError(f"invalid sidecar group name: {group!r}")
        values = f"sidecar_{group}_values"
        valid = f"sidecar_{group}_valid"
        if (
            values not in arrays
            or valid not in arrays
            or f"sidecar_{group}" not in feature_names
        ):
            raise ValueError(
                f"store does not contain the requested sidecar group: {group}"
            )
    return normalized


def _development_indices(
    dates: NDArray[np.datetime64],
    *,
    runtime: ValidationRuntime,
) -> tuple[
    dict[str, NDArray[np.int64]],
    dict[str, NDArray[np.int64]],
    dict[str, NDArray[np.int64]],
    dict[str, NDArray[np.int64]],
    dict[str, object],
]:
    python_dates = tuple(dates.astype("datetime64[D]").astype(object).tolist())
    folds = {fold.name: fold for fold in development_folds(python_dates)}
    fit: dict[str, NDArray[np.int64]] = {}
    fit_target_window: dict[str, NDArray[np.int64]] = {}
    selection: dict[str, NDArray[np.int64]] = {}
    evaluation: dict[str, NDArray[np.int64]] = {}
    payload: dict[str, object] = {}
    for name, fold in folds.items():
        fit[name] = _bounded(
            _date_indices(dates, fold.fit_dates), runtime.max_fit_sessions, tail=True
        )
        selection[name] = _bounded(
            _date_indices(dates, fold.selection_dates),
            runtime.max_selection_sessions,
            tail=False,
        )
        fit_target_window[name] = _date_indices(
            dates, (*fold.fit_dates, *fold.purge_before_dates)
        )
        evaluation[name] = _date_indices(dates, fold.evaluation_dates)
        payload[name] = {
            **fold.payload(),
            "validation_fit_date_indices": fit[name].tolist(),
            "validation_selection_date_indices": selection[name].tolist(),
            "validation_evaluation_date_indices": evaluation[name].tolist(),
        }
    return fit, fit_target_window, selection, evaluation, payload


def _pretrain_indices(
    dates: NDArray[np.datetime64], runtime: ValidationRuntime
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    python_dates = np.asarray(dates.astype("datetime64[D]").astype(object))
    full = np.flatnonzero(
        (python_dates >= STORE_START) & (python_dates <= PRETRAIN_END)
    ).astype(np.int64)
    fit, embargo, selection = pretrain_internal_split(full)
    return (
        _bounded(fit, runtime.max_pretrain_fit_sessions, tail=True),
        embargo,
        _bounded(selection, runtime.max_pretrain_selection_sessions, tail=False),
    )


def _load_development_cdi(
    *,
    dates: NDArray[np.datetime64],
    cdi_path: Path,
    expected_sha256: str,
    experiment52_cdi_path: Path,
    experiment52_expected_sha256: str,
) -> tuple[NDArray[np.float64], dict[str, object]]:
    resolved = cdi_path.resolve(strict=True)
    actual_sha256 = sha256_file(resolved)
    if actual_sha256.casefold() != expected_sha256.casefold():
        raise ValueError(f"development CDI series SHA256 mismatch: {actual_sha256}")
    reference_resolved = experiment52_cdi_path.resolve(strict=True)
    reference_sha256 = sha256_file(reference_resolved)
    if reference_sha256.casefold() != experiment52_expected_sha256.casefold():
        raise ValueError(
            f"Experiment-52 CDI series SHA256 mismatch: {reference_sha256}"
        )

    columns = ["trade_date", "daily_cdi_rate"]
    extension = pl.read_parquet(resolved).select(columns).sort("trade_date")
    reference = pl.read_parquet(reference_resolved).select(columns).sort("trade_date")
    for label, rows in (
        ("development extension", extension),
        ("Experiment-52", reference),
    ):
        if rows.is_empty():
            raise ValueError(f"{label} CDI series is empty")
        if rows["trade_date"].dtype != pl.Date:
            raise ValueError(f"{label} CDI trade_date must be a Parquet date")
        if rows["trade_date"].n_unique() != rows.height:
            raise ValueError(f"{label} CDI series contains duplicate dates")
        rates = rows["daily_cdi_rate"].to_numpy()
        if (
            not np.issubdtype(rates.dtype, np.floating)
            or not np.isfinite(rates).all()
            or np.any(rates <= -1.0)
        ):
            raise ValueError(f"{label} CDI series contains an invalid daily rate")

    reference_start = reference.item(0, "trade_date")
    reference_end = reference.item(-1, "trade_date")
    extension_start = extension.item(0, "trade_date")
    extension_end = extension.item(-1, "trade_date")
    if extension_start > reference_start or extension_end < reference_end:
        raise ValueError(
            "Experiment-52 CDI reference span is not fully contained in the "
            "development extension"
        )
    overlap = (
        reference.select("trade_date")
        .join(extension, on="trade_date", how="inner", validate="1:1")
        .sort("trade_date")
    )
    if overlap.height != reference.height:
        raise ValueError(
            "development CDI extension does not contain every Experiment-52 date"
        )
    exact_byte_match = reference.schema == overlap.schema and all(
        reference[column].to_numpy().tobytes() == overlap[column].to_numpy().tobytes()
        for column in columns
    )
    if not exact_byte_match:
        raise ValueError(
            "development CDI extension differs from the Experiment-52 reference"
        )
    rate_difference = np.abs(
        overlap["daily_cdi_rate"].to_numpy() - reference["daily_cdi_rate"].to_numpy()
    )
    maximum_absolute_difference = float(rate_difference.max())
    if maximum_absolute_difference != 0.0:
        raise ValueError(
            "development CDI extension differs from the Experiment-52 reference"
        )

    python_dates = np.asarray(dates.astype("datetime64[D]").astype(object))
    development_indices = np.flatnonzero(
        (python_dates >= FINETUNE_START) & (python_dates <= DEVELOPMENT_END)
    ).astype(np.int64)
    if not development_indices.size:
        raise ValueError("store has no development sessions for CDI alignment")
    requested_dates = tuple(python_dates[development_indices].tolist())
    values = load_daily_cdi_rates(resolved, requested_dates, actual_sha256)
    result = np.full(len(dates), np.nan, dtype=np.float64)
    result[development_indices] = values
    return result, {
        "development_extension": {
            "path": str(resolved),
            "sha256": actual_sha256,
        },
        "experiment52_reference": {
            "path": str(reference_resolved),
            "sha256": reference_sha256,
        },
        "equality_proof": {
            "comparison_columns": columns,
            "reference_fully_contained": True,
            "overlap_count": reference.height,
            "overlap_date_range": {
                "start": reference_start.isoformat(),
                "end": reference_end.isoformat(),
            },
            "max_abs_daily_cdi_rate": maximum_absolute_difference,
            "exact_byte_match": True,
        },
    }


def _verify_inventory_rows(
    root: Path,
    rows: list[dict[str, object]],
    *,
    excluded: set[str],
) -> None:
    if inventory(root, exclude=excluded) != rows:
        raise RuntimeError("pipeline validation inventory changed while sealing")


def _verify_bound_inventory_files(root: Path, rows: object) -> None:
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("prior classical inventory file list is malformed")
    for row in rows:
        relative = row.get("path")
        expected_bytes = row.get("bytes")
        expected_sha = row.get("sha256")
        if (
            not isinstance(relative, str)
            or not isinstance(expected_bytes, int)
            or not isinstance(expected_sha, str)
        ):
            raise ValueError("prior classical inventory row is malformed")
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("prior classical inventory escapes its immutable root")
        if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha:
            raise ValueError(f"prior classical artifact hash mismatch: {relative}")


def _verified_prior_acceptance(
    *,
    root: Path,
    expected_manifest_sha256: str,
    expected_inventory_sha256: str,
    store_manifest_sha256: str,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    source = Path(root).resolve(strict=True)
    manifest_path = source / "pipeline_validation_manifest.json"
    inventory_path = source / "inventory.json"
    if sha256_file(manifest_path).casefold() != expected_manifest_sha256.casefold():
        raise ValueError("prior acceptance manifest SHA-256 mismatch")
    if sha256_file(inventory_path).casefold() != expected_inventory_sha256.casefold():
        raise ValueError("prior acceptance inventory SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inventory_payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(inventory_payload, dict):
        raise ValueError("prior acceptance audit payload is malformed")
    if (
        manifest.get("schema") != _PRIOR_PIPELINE_SCHEMA
        or manifest.get("status") != "completed"
        or manifest.get("pipeline_validation") is not True
        or manifest.get("research_claim") is not False
        or manifest.get("official_validation_accessed") is not False
        or manifest.get("test_accessed") is not False
        or manifest.get("transfer_chronology_clean") is not True
    ):
        raise ValueError("prior acceptance manifest is stale or has invalid access")
    if (
        inventory_payload.get("schema") != f"{_PRIOR_PIPELINE_SCHEMA}_INVENTORY"
        or inventory_payload.get("status") != "completed"
    ):
        raise ValueError("prior acceptance inventory schema is stale")
    _verify_bound_inventory_files(source, inventory_payload.get("files"))
    sources = manifest.get("sources")
    store_source = sources.get("store") if isinstance(sources, Mapping) else None
    if not isinstance(store_source, Mapping) or (
        str(store_source.get("manifest_sha256", "")).casefold()
        != store_manifest_sha256.casefold()
    ):
        raise ValueError("prior acceptance used a different immutable store")
    results = manifest.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("prior acceptance lacks its result roster")
    baselines = results.get("baselines")
    gbdt = results.get("gbdt_triage")
    if (
        not isinstance(baselines, list)
        or len(baselines) != 15
        or not isinstance(gbdt, list)
        or len(gbdt) != 1
        or any(not isinstance(row, dict) for row in (*baselines, *gbdt))
    ):
        raise ValueError("prior acceptance classical roster is incomplete")
    return source, manifest, inventory_payload


def _assert_prior_store_build_identity(
    prior: Mapping[str, object], *, expected_store_build_commit: str
) -> None:
    if prior.get("store_build_implementation_commit") != expected_store_build_commit:
        raise ValueError("prior acceptance used a different sealed store build")


def _verified_bound_acceptance_root(
    binding: Mapping[str, object], *, store_manifest_sha256: str
) -> tuple[Path, str, Mapping[str, object]]:
    root_raw = binding.get("root")
    expected_manifest_sha = binding.get("manifest_sha256")
    expected_inventory_sha = binding.get("inventory_sha256")
    expected_commit = binding.get("implementation_commit")
    if not all(
        isinstance(value, str)
        for value in (
            root_raw,
            expected_manifest_sha,
            expected_inventory_sha,
            expected_commit,
        )
    ):
        raise ValueError("bound acceptance source identity is malformed")
    source = Path(root_raw).resolve(strict=True)
    manifest_path = source / "pipeline_validation_manifest.json"
    inventory_path = source / "inventory.json"
    if sha256_file(manifest_path) != expected_manifest_sha:
        raise ValueError("bound acceptance manifest SHA-256 mismatch")
    if sha256_file(inventory_path) != expected_inventory_sha:
        raise ValueError("bound acceptance inventory SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inventory_payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    schema = manifest.get("schema") if isinstance(manifest, Mapping) else None
    if (
        not isinstance(manifest, Mapping)
        or not isinstance(inventory_payload, Mapping)
        or schema not in _ACCEPTANCE_ANCESTOR_SCHEMAS
        or manifest.get("status") != "completed"
        or manifest.get("pipeline_validation") is not True
        or manifest.get("research_claim") is not False
        or manifest.get("official_validation_accessed") is not False
        or manifest.get("test_accessed") is not False
        or manifest.get("transfer_chronology_clean") is not True
        or inventory_payload.get("schema") != f"{schema}_INVENTORY"
        or inventory_payload.get("status") != "completed"
    ):
        raise ValueError("bound acceptance source is stale or has invalid access")
    code = manifest.get("code")
    if not isinstance(code, Mapping) or code.get("commit") != expected_commit:
        raise ValueError("bound acceptance implementation identity mismatch")
    sources = manifest.get("sources")
    store_source = sources.get("store") if isinstance(sources, Mapping) else None
    if not isinstance(store_source, Mapping) or (
        str(store_source.get("manifest_sha256", "")).casefold()
        != store_manifest_sha256.casefold()
    ):
        raise ValueError("bound acceptance used a different immutable store")
    _verify_bound_inventory_files(source, inventory_payload.get("files"))
    return source, str(schema), manifest


def _verified_acceptance_ancestors(
    prior: Mapping[str, object], *, store_manifest_sha256: str
) -> dict[Path, str]:
    roots: dict[Path, str] = {}
    sources = prior.get("sources")
    binding = (
        sources.get("prior_classical_acceptance")
        if isinstance(sources, Mapping)
        else None
    )
    while isinstance(binding, Mapping):
        root, schema, manifest = _verified_bound_acceptance_root(
            binding, store_manifest_sha256=store_manifest_sha256
        )
        if root in roots:
            raise ValueError("prior acceptance ancestry contains a cycle")
        roots[root] = schema
        sources = manifest.get("sources")
        binding = (
            sources.get("prior_classical_acceptance")
            if isinstance(sources, Mapping)
            else None
        )
    return roots


def _load_prior_score_panel(
    *,
    source_roots: Mapping[Path, str],
    record: Mapping[str, object],
    expected_indices: NDArray[np.int64],
    expected_name_count: int,
) -> tuple[NDArray[np.float32], NDArray[np.bool_], str]:
    manifest_raw = record.get("score_manifest")
    expected_manifest_sha = record.get("score_manifest_sha256")
    if not isinstance(manifest_raw, str) or not isinstance(expected_manifest_sha, str):
        raise ValueError("prior classical record lacks its score binding")
    manifest_path = Path(manifest_raw).resolve(strict=True)
    matching_roots = [
        (root, schema)
        for root, schema in source_roots.items()
        if manifest_path.is_relative_to(root)
    ]
    if not matching_roots:
        raise ValueError("prior score panel is outside every verified acceptance root")
    _, expected_schema = max(matching_roots, key=lambda row: len(row[0].parts))
    if sha256_file(manifest_path) != expected_manifest_sha:
        raise ValueError("prior score-panel manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or (
        manifest.get("schema") != expected_schema
        or manifest.get("status") != "completed"
        or manifest.get("pipeline_validation") is not True
        or manifest.get("research_claim") is not False
        or manifest.get("official_validation_accessed") is not False
        or manifest.get("test_accessed") is not False
        or manifest.get("transfer_chronology_clean") is not True
    ):
        raise ValueError("prior score panel is stale or has invalid access")
    metadata = manifest.get("metadata")
    artifacts = manifest.get("artifacts")
    if not isinstance(metadata, Mapping) or not isinstance(artifacts, Mapping):
        raise ValueError("prior score panel manifest is malformed")
    index_values = metadata.get("date_indices", metadata.get("evaluation_date_indices"))
    if not isinstance(index_values, list) or not np.array_equal(
        np.asarray(index_values, dtype=np.int64), expected_indices
    ):
        raise ValueError("prior score panel has a different evaluation window")
    arrays: list[NDArray[np.generic]] = []
    for filename in ("scores.npy", "score_mask.npy"):
        path = manifest_path.parent / filename
        row = artifacts.get(filename)
        if not isinstance(row, Mapping) or (
            path.stat().st_size != int(row.get("bytes", -1))
            or sha256_file(path) != row.get("sha256")
        ):
            raise ValueError(f"prior score-panel artifact hash mismatch: {path}")
        arrays.append(np.load(path, allow_pickle=False))
    scores = np.asarray(arrays[0], dtype=np.float32)
    score_mask = np.asarray(arrays[1], dtype=np.bool_)
    if scores.shape != score_mask.shape or scores.shape[:2] != (
        len(expected_indices),
        expected_name_count,
    ):
        raise ValueError("prior score panel arrays are misaligned")
    return scores, score_mask, expected_manifest_sha


def _non_ledger_report(report: Mapping[str, object]) -> dict[str, object]:
    payload = {
        key: value
        for key, value in report.items()
        if key not in {"economics", "schema"}
    }
    coverage = payload.get("mask_coverage")
    if isinstance(coverage, Mapping):
        payload["mask_coverage"] = {
            key: value
            for key, value in coverage.items()
            if key not in _LEDGER_MASK_COVERAGE_FIELDS
        }
    return payload


def _canonical_payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _changed_field_paths(
    left: object, right: object, prefix: str = "economics"
) -> set[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        paths: set[str] = set()
        for key in set(left) | set(right):
            path = f"{prefix}.{key}"
            if key not in left or key not in right:
                paths.add(path)
            else:
                paths.update(_changed_field_paths(left[key], right[key], path))
        return paths
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return {f"{prefix}[]"}
        paths: set[str] = set()
        for old, new in zip(left, right, strict=True):
            paths.update(_changed_field_paths(old, new, f"{prefix}[]"))
        return paths
    return set() if left == right else {prefix}


def _verified_classical_source(
    *,
    root: Path,
    expected_inventory_sha256: str,
    expected_failure_sha256: str,
) -> dict[str, object]:
    source = Path(root).resolve(strict=True)
    inventory_path = source / "artifact_inventory.json"
    failure_path = source / "failure_record.json"
    if sha256_file(inventory_path).casefold() != expected_inventory_sha256.casefold():
        raise ValueError("completed classical inventory SHA-256 mismatch")
    if sha256_file(failure_path).casefold() != expected_failure_sha256.casefold():
        raise ValueError("completed classical failure-record SHA-256 mismatch")
    inventory_payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    if not isinstance(inventory_payload, Mapping) or not isinstance(failure, Mapping):
        raise ValueError("completed classical audit payload is malformed")
    if (
        inventory_payload.get("schema")
        != "BRAZIL_RV_V2_PIPELINE_VALIDATION_FAILURE_INVENTORY_V1"
        or inventory_payload.get("status") != "failed"
        or failure.get("schema") != "BRAZIL_RV_V2_PIPELINE_VALIDATION_FAILURE_V1"
        or failure.get("status") != "failed"
    ):
        raise ValueError("completed classical failure schemas are not recognized")
    access = failure.get("access_audit")
    if not isinstance(access, Mapping) or (
        access.get("pipeline_validation") is not True
        or access.get("research_claim") is not False
        or access.get("official_validation_accessed") is not False
        or access.get("test_accessed") is not False
        or access.get("all_registrations_null") is not True
        or access.get("json_sidecars_verified") is not True
        or access.get("transfer_chronology_clean") is not True
    ):
        raise ValueError("completed classical source records invalid access state")
    completed = failure.get("completed_before_failure")
    not_started = failure.get("not_started")
    if (
        not isinstance(completed, Mapping)
        or not isinstance(not_started, Mapping)
        or (
            completed.get("baseline_evaluations") != 12
            or completed.get("gbdt_evaluations") != 2
            or completed.get("gbdt_head_models") != 100
            or completed.get("gbdt_model_manifests") != 4
            or not_started.get("checkpoint_count") != 0
            or not_started.get("network_artifact_count") != 0
            or not_started.get("neural_history_count") != 0
        )
    ):
        raise ValueError("completed classical failure boundary is not score-free")
    excluded_raw = inventory_payload.get("excluded_self")
    rows = inventory_payload.get("files")
    if (
        not isinstance(excluded_raw, list)
        or not all(isinstance(value, str) for value in excluded_raw)
        or not isinstance(rows, list)
        or any(not isinstance(row, dict) for row in rows)
    ):
        raise ValueError("completed classical inventory is malformed")
    _verify_inventory_rows(source, rows, excluded=set(excluded_raw))
    if (source / "network_smokes").exists():
        raise ValueError(
            "completed classical source unexpectedly contains neural output"
        )
    baseline_evaluations = list((source / "baselines").rglob("evaluation.json"))
    gbdt_evaluations = list((source / "gbdt_triage").rglob("evaluation.json"))
    gbdt_models = list((source / "gbdt_triage").rglob("*.txt"))
    gbdt_manifests = list((source / "gbdt_triage").rglob("model_manifest.json"))
    if (
        len(baseline_evaluations) != 12
        or len(gbdt_evaluations) != 2
        or len(gbdt_models) != 100
        or len(gbdt_manifests) != 4
    ):
        raise ValueError("completed classical source has the wrong artifact grid")
    for path in (*baseline_evaluations, *gbdt_evaluations, *gbdt_manifests):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or (
            payload.get("official_validation_accessed") is not False
            or payload.get("test_accessed") is not False
            or payload.get("transfer_chronology_clean") is not True
        ):
            raise PermissionError(
                f"completed classical artifact has invalid chronology/access: {path}"
            )
    return {
        "root": str(source),
        "inventory": str(inventory_path),
        "inventory_sha256": expected_inventory_sha256.casefold(),
        "failure_record": str(failure_path),
        "failure_record_sha256": expected_failure_sha256.casefold(),
        "implementation_commit": failure.get("implementation_commit"),
        "baseline_evaluation_count": len(baseline_evaluations),
        "gbdt_evaluation_count": len(gbdt_evaluations),
        "gbdt_model_count": len(gbdt_models),
        "gbdt_model_manifest_count": len(gbdt_manifests),
        "neural_artifacts_present": False,
    }


def run_pipeline_validation(
    *,
    store_root: Path,
    cdi_path: Path,
    cdi_sha256: str,
    experiment52_cdi_path: Path,
    experiment52_cdi_sha256: str,
    bova11_root: Path,
    bova11_manifest_sha256: str,
    hedge_beta_root: Path,
    hedge_beta_manifest_sha256: str,
    lending_archive_root: Path,
    lending_archive_manifest_sha256: str,
    output_root: Path,
    runtime: ValidationRuntime = ValidationRuntime(),
    enabled_sidecars: Sequence[str] = (),
    native_fast_audit_path: Path | None = None,
    native_fast_audit_sha256: str | None = None,
) -> PipelineValidationResult:
    """Run only the development-fold integration checks required by v2 section 11.

    This command deliberately produces integration diagnostics, never research
    evidence. It refuses a dirty implementation and every session in 2025/2026.
    The caller must supply the exact immutable store, the hash-pinned development
    CDI extension, and the hash-pinned Experiment-52 CDI reference it extends.
    """

    store_path = Path(store_root).resolve(strict=True)
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(output)
    if any(
        output == source
        or output.is_relative_to(source)
        or source.is_relative_to(output)
        for source in (store_path,)
    ):
        raise ValueError("validation output and immutable input store must be disjoint")
    code = _git_identity()
    store_manifest, dates = _read_store_header(store_path)
    store_metadata = store_manifest.get("metadata")
    if not isinstance(store_metadata, Mapping):
        raise ValueError("v2 store metadata is malformed")
    store_build_commit = _store_build_commit(store_manifest)
    sidecars = _validate_sidecars(store_manifest, enabled_sidecars)
    external_resolutions = _external_artifact_resolutions(store_manifest)
    _assert_overrides_outside_store(store_path, external_resolutions)
    (
        fit_indices,
        fit_target_window_indices,
        selection_indices,
        evaluation_indices,
        fold_payload,
    ) = _development_indices(dates, runtime=runtime)
    pretrain_fit, pretrain_embargo, pretrain_selection = _pretrain_indices(
        dates, runtime
    )
    triage = protocol_preset("triage")
    full = protocol_preset("full")
    if triage.folds != ("F1", "F2") or triage.seeds != (11,):
        raise ValueError("triage protocol differs from the validation contract")
    if full.folds != ("F1", "F2", "F3"):
        raise ValueError("full protocol differs from the validation contract")
    missing_folds = set(full.folds) - set(evaluation_indices)
    if missing_folds:
        raise ValueError(
            f"store calendar is missing development folds: {sorted(missing_folds)}"
        )
    cdi_by_index, cdi_provenance = _load_development_cdi(
        dates=dates,
        cdi_path=Path(cdi_path),
        expected_sha256=cdi_sha256,
        experiment52_cdi_path=Path(experiment52_cdi_path),
        experiment52_expected_sha256=experiment52_cdi_sha256,
    )
    bova11 = load_bova11_series(
        bova11_root,
        expected_manifest_sha256=bova11_manifest_sha256,
        canonical_dates=dates.astype("datetime64[D]").astype(object).tolist(),
    )
    bova11_binding = {
        "root": str(Path(bova11_root).resolve(strict=True)),
        "hedge_beta_root": str(hedge_beta_root.resolve(strict=True)),
        "hedge_beta_manifest_sha256": hedge_beta_manifest_sha256,
        "manifest_sha256": bova11.manifest_sha256,
        "data_sha256": bova11.data_sha256,
    }
    lending_borrow = load_lending_borrow_panels(
        lending_archive_root,
        expected_manifest_sha256=lending_archive_manifest_sha256,
        canonical_dates=dates.astype("datetime64[D]").astype(object).tolist(),
        canonical_isins=[
            str(value)
            for value in np.load(store_path / "isin_index.npy", allow_pickle=False)
        ],
    )
    store_manifest_sha = sha256_file(store_path / "manifest.json")
    native_fast_audit = _verify_native_fast_audit(
        native_fast_audit_path,
        expected_sha256=native_fast_audit_sha256,
        store_manifest_sha256=store_manifest_sha,
    )
    requested_groups = (
        pretrain_fit,
        pretrain_selection,
        *[fit_indices[name] for name in full.folds],
        *[fit_target_window_indices[name] for name in full.folds],
        *[selection_indices[name] for name in full.folds],
        *[evaluation_indices[name] for name in full.folds],
    )
    requested_indices = np.unique(np.concatenate(requested_groups)).astype(
        np.int64, copy=False
    )
    requested_dates = _dates_for_indices(dates, requested_indices)
    if any(value >= OFFICIAL_START for value in requested_dates):
        raise PermissionError("pipeline validation refuses every 2025/2026 session")
    history_lookbacks = np.full(
        len(requested_indices), runtime.slow_lookback, dtype=np.int64
    )
    baseline_samples = np.concatenate([evaluation_indices[name] for name in full.folds])
    history_lookbacks[np.isin(requested_indices, baseline_samples)] = 253
    history_end_offsets = np.where(
        dates[requested_indices] <= np.datetime64(PRETRAIN_END), 0, -1
    ).astype(np.int64)
    store, source_access = open_store_for_samples(
        store_path,
        requested_indices,
        purpose="evaluation",
        history_lookbacks=history_lookbacks,
        history_end_offsets=history_end_offsets,
    )
    source_hashes = {
        "v2_store_manifest": store_manifest_sha,
        "cdi_development_extension": str(
            cdi_provenance["development_extension"]["sha256"]
        ),
        "cdi_experiment52_reference": str(
            cdi_provenance["experiment52_reference"]["sha256"]
        ),
        "bova11_manifest": bova11.manifest_sha256,
        "bova11_data": bova11.data_sha256,
    }
    output.mkdir(parents=True, exist_ok=False)
    try:
        baseline_records = _run_baselines(
            store=store,
            fold_indices={name: evaluation_indices[name] for name in full.folds},
            cdi_by_index=cdi_by_index,
            bova11_close_by_index=bova11.close_by_session,
            bova11_binding=bova11_binding,
            lending_borrow=lending_borrow,
            root=output / "baselines",
            source_hashes=source_hashes,
        )
        gbdt_records = _run_gbdt(
            store=store,
            fit_indices={"F1": fit_indices["F1"]},
            fit_target_window_indices={"F1": fit_target_window_indices["F1"]},
            selection_indices={"F1": selection_indices["F1"]},
            evaluation_indices={"F1": evaluation_indices["F1"]},
            cdi_by_index=cdi_by_index,
            bova11_close_by_index=bova11.close_by_session,
            bova11_binding=bova11_binding,
            lending_borrow=lending_borrow,
            root=output / "gbdt_triage",
            source_hashes=source_hashes,
            runtime=runtime,
            sidecars=sidecars,
        )
        legacy_identity = _legacy_round1_baseline_identity(
            baseline_records,
            prior_root=_LEGACY_ROUND1_ROOT,
            prior_result_sha256=_LEGACY_ROUND1_RESULT_SHA256,
            prior_inventory_sha256=_LEGACY_ROUND1_INVENTORY_SHA256,
        )
        protocol_hashes = {
            name: {
                "path": str(PROTOCOL_CONFIG_ROOT / f"{name}.json"),
                "sha256": sha256_file(PROTOCOL_CONFIG_ROOT / f"{name}.json"),
            }
            for name in ("triage", "full")
        }
        acceptance = _development_acceptance(
            baseline_records=baseline_records,
            gbdt_records=gbdt_records,
            action_terms_source=store_metadata.get("action_terms_source"),
            schedule_source=store_metadata.get("schedule_source"),
            native_fast_audit_passed=native_fast_audit is not None,
            legacy_round1_identity=legacy_identity,
        )
        manifest_path = output / "pipeline_validation_manifest.json"
        manifest_sha = write_json_atomic(
            manifest_path,
            {
                "schema": PIPELINE_SCHEMA,
                "status": "completed",
                "engineering_acceptance_status": acceptance["status"],
                "engineering_acceptance_reasons": acceptance["reasons"],
                **PIPELINE_FLAGS,
                "scope": (
                    "development-only integration validation; numbers are not "
                    "research claims"
                ),
                "code": code,
                "store_build_implementation_commit": store_build_commit,
                "runtime": asdict(runtime),
                "protocols": protocol_hashes,
                "enabled_sidecars": list(sidecars),
                "sources": {
                    "store": {
                        "root": str(store_path),
                        "manifest_sha256": store_manifest_sha,
                        "access_ledger": source_access.payload(),
                        "external_artifact_resolutions": external_resolutions,
                    },
                    "cdi": cdi_provenance,
                    "bova11": bova11_binding,
                    "native_fast_raw_audit": native_fast_audit,
                    "lending_archive": {
                        "root": str(Path(lending_archive_root).resolve(strict=True)),
                        "manifest_sha256": lending_borrow.manifest_sha256,
                        "balances_sha256": lending_borrow.balance_sha256,
                        "rates_sha256": lending_borrow.rate_sha256,
                        "source_label": lending_borrow.source_label,
                        "source_unavailable_dates": [
                            value.isoformat()
                            for value in lending_borrow.source_unavailable_dates
                        ],
                        "source_placeholder_dates": [
                            value.isoformat()
                            for value in lending_borrow.source_placeholder_dates
                        ],
                    },
                    "legacy_round1_baseline_reference": legacy_identity,
                },
                "date_contract": {
                    "minimum_date": min(requested_dates).isoformat(),
                    "maximum_date": max(requested_dates).isoformat(),
                    "official_validation_accessed": False,
                    "test_accessed": False,
                    "pretrain_fit_date_indices": pretrain_fit.tolist(),
                    "pretrain_embargo_date_indices": pretrain_embargo.tolist(),
                    "pretrain_selection_date_indices": pretrain_selection.tolist(),
                    "development_folds": fold_payload,
                },
                "results": {
                    "development_acceptance": acceptance,
                    "baselines": baseline_records,
                    "gbdt_triage": gbdt_records,
                    "network_smokes": {
                        "status": "not_run",
                        "reason": (
                            "not required for development-grade classical data acceptance; "
                            "neural validation belongs to the frozen registered round"
                        ),
                    },
                },
            },
        )
        excluded = {"inventory.json", "inventory.json.sha256"}
        rows = inventory(output, exclude=excluded)
        inventory_path = output / "inventory.json"
        inventory_sha = write_json_atomic(
            inventory_path,
            {
                "schema": f"{PIPELINE_SCHEMA}_INVENTORY",
                "status": "completed",
                **PIPELINE_FLAGS,
                "excluded_self": sorted(excluded),
                "files": rows,
            },
        )
        _verify_inventory_rows(output, rows, excluded=excluded)
    finally:
        store.close()
    return PipelineValidationResult(
        root=output,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        inventory_path=inventory_path,
        inventory_sha256=inventory_sha,
    )


def replay_classical_economics(
    *,
    store_root: Path,
    store_manifest_sha256: str,
    cdi_path: Path,
    cdi_sha256: str,
    experiment52_cdi_path: Path,
    experiment52_cdi_sha256: str,
    bova11_root: Path,
    bova11_manifest_sha256: str,
    hedge_beta_root: Path,
    hedge_beta_manifest_sha256: str,
    lending_archive_root: Path,
    lending_archive_manifest_sha256: str,
    prior_classical_root: Path,
    prior_classical_manifest_sha256: str,
    prior_classical_inventory_sha256: str,
    output_root: Path,
) -> PipelineValidationResult:
    """Re-evaluate sealed classical scores after a ledger-only repair.

    Models and score panels are never recomputed. Every prior input is verified
    against the sealed inventory, and the complete evaluation report outside
    ``economics`` (apart from the intentional schema version) must be exactly
    equal as a decoded JSON value before the replay can be accepted.
    """

    store_path = Path(store_root).resolve(strict=True)
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(output)
    code = _git_identity()
    store_manifest, dates = _read_store_header(store_path)
    actual_store_sha = sha256_file(store_path / "manifest.json")
    if actual_store_sha.casefold() != store_manifest_sha256.casefold():
        raise ValueError("sealed store manifest SHA-256 mismatch")
    store_metadata = store_manifest.get("metadata")
    if not isinstance(store_metadata, Mapping):
        raise ValueError("store metadata is malformed")
    store_build_commit = store_metadata.get("implementation_git_commit")
    if not isinstance(store_build_commit, str):
        raise ValueError("store lacks its build implementation commit")
    bova11 = load_bova11_series(
        bova11_root,
        expected_manifest_sha256=bova11_manifest_sha256,
        canonical_dates=dates.astype("datetime64[D]").astype(object).tolist(),
    )
    bova11_binding = {
        "root": str(Path(bova11_root).resolve(strict=True)),
        "hedge_beta_root": str(hedge_beta_root.resolve(strict=True)),
        "hedge_beta_manifest_sha256": hedge_beta_manifest_sha256,
        "manifest_sha256": bova11.manifest_sha256,
        "data_sha256": bova11.data_sha256,
    }
    lending_borrow = load_lending_borrow_panels(
        lending_archive_root,
        expected_manifest_sha256=lending_archive_manifest_sha256,
        canonical_dates=dates.astype("datetime64[D]").astype(object).tolist(),
        canonical_isins=[
            str(value)
            for value in np.load(store_path / "isin_index.npy", allow_pickle=False)
        ],
    )

    source_root, prior, _ = _verified_prior_acceptance(
        root=prior_classical_root,
        expected_manifest_sha256=prior_classical_manifest_sha256,
        expected_inventory_sha256=prior_classical_inventory_sha256,
        store_manifest_sha256=actual_store_sha,
    )
    prior_code = prior.get("code")
    if not isinstance(prior_code, Mapping) or not isinstance(
        prior_code.get("commit"), str
    ):
        raise ValueError("prior acceptance code identity is malformed")
    _assert_prior_store_build_identity(
        prior, expected_store_build_commit=store_build_commit
    )
    if output == source_root or output.is_relative_to(source_root):
        raise ValueError(
            "ledger replay output must be outside the prior immutable root"
        )

    runtime_payload = prior.get("runtime")
    if not isinstance(runtime_payload, Mapping):
        raise ValueError("prior acceptance runtime is malformed")
    runtime = ValidationRuntime(**dict(runtime_payload))
    _, _, _, evaluation_indices, fold_payload = _development_indices(
        dates, runtime=runtime
    )
    expected_folds = {name: evaluation_indices[name] for name in ("F1", "F2", "F3")}
    sidecars_raw = prior.get("enabled_sidecars")
    if not isinstance(sidecars_raw, list):
        raise ValueError("prior acceptance sidecar roster is malformed")
    sidecars = _validate_sidecars(
        store_manifest, [str(value) for value in sidecars_raw]
    )
    external_resolutions = _external_artifact_resolutions(store_manifest)
    _assert_overrides_outside_store(store_path, external_resolutions)

    cdi_by_index, cdi_provenance = _load_development_cdi(
        dates=dates,
        cdi_path=Path(cdi_path),
        expected_sha256=cdi_sha256,
        experiment52_cdi_path=Path(experiment52_cdi_path),
        experiment52_expected_sha256=experiment52_cdi_sha256,
    )
    prior_sources = prior.get("sources")
    prior_cdi = prior_sources.get("cdi") if isinstance(prior_sources, Mapping) else None
    if not isinstance(prior_cdi, Mapping):
        raise ValueError("prior acceptance CDI provenance is missing")
    score_source_roots = {
        source_root: _PRIOR_PIPELINE_SCHEMA,
        **_verified_acceptance_ancestors(prior, store_manifest_sha256=actual_store_sha),
    }
    for key in ("development_extension", "experiment52_reference"):
        old = prior_cdi.get(key)
        new = cdi_provenance.get(key)
        if (
            not isinstance(old, Mapping)
            or not isinstance(new, Mapping)
            or (old.get("sha256") != new.get("sha256"))
        ):
            raise ValueError(f"ledger replay CDI identity differs for {key}")
    prior_fast = (
        prior_sources.get("native_fast_raw_audit")
        if isinstance(prior_sources, Mapping)
        else None
    )
    if not isinstance(prior_fast, Mapping):
        raise ValueError("prior acceptance lacks the native-fast audit binding")
    fast_path = prior_fast.get("path")
    fast_sha = prior_fast.get("sha256")
    if not isinstance(fast_path, str) or not isinstance(fast_sha, str):
        raise ValueError("prior native-fast audit binding is malformed")
    native_fast_audit = _verify_native_fast_audit(
        Path(fast_path),
        expected_sha256=fast_sha,
        store_manifest_sha256=actual_store_sha,
    )

    results = prior["results"]
    assert isinstance(results, Mapping)
    baseline_source = results["baselines"]
    gbdt_source = results["gbdt_triage"]
    assert isinstance(baseline_source, list) and isinstance(gbdt_source, list)
    baseline_roster = {
        (str(record.get("fold")), str(record.get("name")))
        for record in baseline_source
        if isinstance(record, Mapping)
    }
    expected_baselines = {
        (fold, name) for fold in expected_folds for name in _BASELINE_SIGNAL_SIGNS
    }
    if baseline_roster != expected_baselines:
        raise ValueError("prior acceptance baseline roster differs from the contract")
    if {str(record.get("fold")) for record in gbdt_source} != {"F1"}:
        raise ValueError("prior acceptance GBDT roster differs from the contract")

    requested_indices = np.unique(np.concatenate(list(expected_folds.values()))).astype(
        np.int64, copy=False
    )
    requested_dates = _dates_for_indices(dates, requested_indices)
    if any(value >= OFFICIAL_START for value in requested_dates):
        raise PermissionError("ledger replay refuses every 2025/2026 session")
    store, source_access = open_store_for_samples(
        store_path,
        requested_indices,
        purpose="evaluation",
        history_lookbacks=np.full(len(requested_indices), 253, dtype=np.int64),
        history_end_offsets=np.full(len(requested_indices), -1, dtype=np.int64),
    )
    source_hashes = {
        "v2_store_manifest": actual_store_sha,
        "cdi_development_extension": str(
            cdi_provenance["development_extension"]["sha256"]
        ),
        "cdi_experiment52_reference": str(
            cdi_provenance["experiment52_reference"]["sha256"]
        ),
        "bova11_manifest": bova11.manifest_sha256,
        "bova11_data": bova11.data_sha256,
    }
    output.mkdir(parents=True, exist_ok=False)
    comparison_rows: list[dict[str, object]] = []
    changed_paths: set[str] = set()
    prior_evaluation_schemas: set[str] = set()

    def replay_record(
        record: Mapping[str, object], *, destination: Path
    ) -> dict[str, object]:
        fold = str(record.get("fold"))
        indices = expected_folds.get(fold)
        if indices is None:
            raise ValueError(f"prior score uses an unexpected fold: {fold}")
        scores, score_mask, score_manifest_sha = _load_prior_score_panel(
            source_roots=score_source_roots,
            record=record,
            expected_indices=indices,
            expected_name_count=len(store.isins),
        )
        old_summary = record.get("evaluation")
        if not isinstance(old_summary, Mapping):
            raise ValueError("prior classical record lacks its evaluation summary")
        old_path_raw = old_summary.get("report")
        old_sha = old_summary.get("report_sha256")
        if not isinstance(old_path_raw, str) or not isinstance(old_sha, str):
            raise ValueError("prior evaluation binding is malformed")
        old_path = Path(old_path_raw).resolve(strict=True)
        if not old_path.is_relative_to(source_root) or sha256_file(old_path) != old_sha:
            raise ValueError("prior evaluation report identity mismatch")
        old_report = json.loads(old_path.read_text(encoding="utf-8"))
        if not isinstance(old_report, Mapping):
            raise ValueError("prior evaluation report is malformed")
        old_schema = old_report.get("schema")
        if not isinstance(old_schema, str):
            raise ValueError("prior evaluation report schema is malformed")
        prior_evaluation_schemas.add(old_schema)
        destination.mkdir(parents=True, exist_ok=False)
        new_result, new_sha = _evaluate_and_write(
            store=store,
            indices=indices,
            scores=scores,
            score_mask=score_mask,
            cdi_by_index=cdi_by_index,
            bova11_close_by_index=bova11.close_by_session,
            bova11_binding=bova11_binding,
            lending_borrow=lending_borrow,
            source_hashes={**source_hashes, "score_manifest": score_manifest_sha},
            window_name=fold,
            path=destination / "evaluation.json",
        )
        old_non_ledger = _non_ledger_report(old_report)
        new_non_ledger = _non_ledger_report(new_result.report)
        if old_non_ledger != new_non_ledger:
            raise RuntimeError(
                f"non-ledger evaluation fields changed during replay: {fold}"
            )
        old_economics = old_report.get("economics")
        new_economics = new_result.report.get("economics")
        if not isinstance(old_economics, Mapping) or not isinstance(
            new_economics, Mapping
        ):
            raise ValueError("evaluation economics payload is malformed")
        local_changed = _changed_field_paths(old_economics, new_economics)
        old_coverage = old_report.get("mask_coverage")
        new_coverage = new_result.report.get("mask_coverage")
        if isinstance(old_coverage, Mapping) and isinstance(new_coverage, Mapping):
            for key in _LEDGER_MASK_COVERAGE_FIELDS:
                if old_coverage.get(key) != new_coverage.get(key):
                    local_changed.add(f"mask_coverage.{key}")
        changed_paths.update(local_changed)
        label = (
            f"baseline:{fold}:{record.get('name')}"
            if record.get("engine") == "baseline"
            else f"gbdt:{fold}"
        )
        comparison_rows.append(
            {
                "evaluation": label,
                "prior_report": str(old_path),
                "prior_report_sha256": old_sha,
                "replayed_report": str(destination / "evaluation.json"),
                "replayed_report_sha256": new_sha,
                "non_ledger_payload_sha256": _canonical_payload_sha256(new_non_ledger),
                "non_ledger_fields_bit_identical": True,
                "changed_ledger_field_paths": sorted(local_changed),
            }
        )
        return {
            **{key: value for key, value in record.items() if key != "evaluation"},
            "evaluation": _evaluation_summary(
                new_result, destination / "evaluation.json", new_sha
            ),
            "score_reused_without_recomputation": True,
        }

    try:
        baseline_records = [
            replay_record(
                record,
                destination=(
                    output / "baselines" / str(record["fold"]) / str(record["name"])
                ),
            )
            for record in baseline_source
        ]
        gbdt_records = [
            replay_record(
                record,
                destination=output / "gbdt_triage" / str(record["fold"]),
            )
            for record in gbdt_source
        ]
        acceptance = _development_acceptance(
            baseline_records=baseline_records,
            gbdt_records=gbdt_records,
            action_terms_source=store_metadata.get("action_terms_source"),
            schedule_source=store_metadata.get("schedule_source"),
            native_fast_audit_passed=native_fast_audit is not None,
        )
        if len(prior_evaluation_schemas) != 1:
            raise ValueError("prior evaluation reports do not share one schema")
        prior_evaluation_schema = next(iter(prior_evaluation_schemas))
        protocol_hashes = {
            name: {
                "path": str(PROTOCOL_CONFIG_ROOT / f"{name}.json"),
                "sha256": sha256_file(PROTOCOL_CONFIG_ROOT / f"{name}.json"),
            }
            for name in ("triage", "full")
        }
        manifest_path = output / "pipeline_validation_manifest.json"
        manifest_sha = write_json_atomic(
            manifest_path,
            {
                "schema": PIPELINE_SCHEMA,
                "status": "completed",
                "engineering_acceptance_status": acceptance["status"],
                "engineering_acceptance_reasons": acceptance["reasons"],
                **PIPELINE_FLAGS,
                "scope": (
                    "development-only ledger replay over hash-reused classical "
                    "scores; numbers are not research claims"
                ),
                "code": code,
                "store_build_implementation_commit": store_build_commit,
                "runtime": asdict(runtime),
                "protocols": protocol_hashes,
                "enabled_sidecars": list(sidecars),
                "sources": {
                    "store": {
                        "root": str(store_path),
                        "manifest_sha256": actual_store_sha,
                        "access_ledger": source_access.payload(),
                        "external_artifact_resolutions": external_resolutions,
                    },
                    "cdi": cdi_provenance,
                    "bova11": bova11_binding,
                    "native_fast_raw_audit": native_fast_audit,
                    "prior_classical_acceptance": {
                        "root": str(source_root),
                        "manifest_sha256": prior_classical_manifest_sha256,
                        "inventory_sha256": prior_classical_inventory_sha256,
                        "implementation_commit": prior_code.get("commit"),
                    },
                },
                "date_contract": {
                    "minimum_date": min(requested_dates).isoformat(),
                    "maximum_date": max(requested_dates).isoformat(),
                    "official_validation_accessed": False,
                    "test_accessed": False,
                    "development_folds": fold_payload,
                },
                "ledger_replay_proof": {
                    "score_or_model_recomputation": False,
                    "prior_evaluation_schema": prior_evaluation_schema,
                    "replayed_evaluation_schema": EVALUATION_SCHEMA,
                    "schema_field_is_the_only_non_economics_exception": True,
                    "all_non_ledger_fields_bit_identical": all(
                        row["non_ledger_fields_bit_identical"] is True
                        for row in comparison_rows
                    ),
                    "comparison_count": len(comparison_rows),
                    "changed_ledger_field_paths": sorted(changed_paths),
                    "comparisons": comparison_rows,
                },
                "results": {
                    "development_acceptance": acceptance,
                    "baselines": baseline_records,
                    "gbdt_triage": gbdt_records,
                    "network_smokes": {
                        "status": "not_run",
                        "reason": "neural validation belongs to the registered round",
                    },
                },
            },
        )
        excluded = {"inventory.json", "inventory.json.sha256"}
        rows = inventory(output, exclude=excluded)
        inventory_path = output / "inventory.json"
        inventory_sha = write_json_atomic(
            inventory_path,
            {
                "schema": f"{PIPELINE_SCHEMA}_INVENTORY",
                "status": "completed",
                **PIPELINE_FLAGS,
                "excluded_self": sorted(excluded),
                "files": rows,
            },
        )
        _verify_inventory_rows(output, rows, excluded=excluded)
    finally:
        store.close()
    return PipelineValidationResult(
        root=output,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        inventory_path=inventory_path,
        inventory_sha256=inventory_sha,
    )


def resume_network_validation(
    *,
    store_root: Path,
    store_manifest_sha256: str,
    cdi_path: Path,
    cdi_sha256: str,
    experiment52_cdi_path: Path,
    experiment52_cdi_sha256: str,
    bova11_root: Path,
    bova11_manifest_sha256: str,
    hedge_beta_root: Path,
    hedge_beta_manifest_sha256: str,
    lending_archive_root: Path,
    lending_archive_manifest_sha256: str,
    completed_classical_root: Path,
    completed_classical_inventory_sha256: str,
    completed_classical_failure_sha256: str,
    output_root: Path,
    runtime: ValidationRuntime = ValidationRuntime(device="cuda"),
    enabled_sidecars: Sequence[str] = (),
) -> PipelineValidationResult:
    """Continue only the F1 neural legs after a sealed classical-only run.

    This path intentionally permits the immutable store's build commit to differ
    from the current loader commit. It compensates by requiring the exact store,
    classical inventory, and score-free failure-record hashes and records both
    commits in the continuation manifest.
    """

    if (
        runtime.fine_epochs != 3
        or runtime.handoff_epochs != 1
        or runtime.slow_lookback != 60
        or runtime.max_fit_sessions is not None
        or runtime.max_selection_sessions is not None
        or runtime.max_pretrain_fit_sessions is not None
        or runtime.max_pretrain_selection_sessions is not None
        or runtime.device != "cuda"
    ):
        raise ValueError("network continuation requires the exact full-F1 GPU runtime")
    store_path = Path(store_root).resolve(strict=True)
    output = Path(output_root).resolve()
    classical_path = Path(completed_classical_root).resolve(strict=True)
    if output.exists():
        raise FileExistsError(output)
    if any(
        output == source
        or output.is_relative_to(source)
        or source.is_relative_to(output)
        for source in (store_path, classical_path)
    ):
        raise ValueError("network output must be disjoint from its immutable inputs")
    code = _git_identity()
    store_manifest, dates = _read_store_header(store_path)
    actual_store_manifest_sha = sha256_file(store_path / "manifest.json")
    if actual_store_manifest_sha.casefold() != store_manifest_sha256.casefold():
        raise ValueError("sealed store manifest SHA-256 mismatch")
    store_metadata = store_manifest.get("metadata")
    if not isinstance(store_metadata, Mapping):
        raise ValueError("store manifest metadata is malformed")
    store_build_commit = store_metadata.get("implementation_git_commit")
    if not isinstance(store_build_commit, str) or len(store_build_commit) != 40:
        raise ValueError("store manifest lacks its build commit")
    classical_source = _verified_classical_source(
        root=classical_path,
        expected_inventory_sha256=completed_classical_inventory_sha256,
        expected_failure_sha256=completed_classical_failure_sha256,
    )
    if classical_source["implementation_commit"] != store_build_commit:
        raise ValueError("completed classical run differs from the store build commit")
    sidecars = _validate_sidecars(store_manifest, enabled_sidecars)
    external_resolutions = _external_artifact_resolutions(store_manifest)
    _assert_overrides_outside_store(store_path, external_resolutions)
    (
        fit_indices,
        fit_target_window_indices,
        selection_indices,
        evaluation_indices,
        fold_payload,
    ) = _development_indices(dates, runtime=runtime)
    pretrain_fit, pretrain_embargo, pretrain_selection = _pretrain_indices(
        dates, runtime
    )
    triage = protocol_preset("triage")
    if (
        triage.seeds != (11,)
        or "F1" not in fit_indices
        or "F1" not in selection_indices
    ):
        raise ValueError("F1 seed protocol differs from the continuation contract")
    cdi_by_index, cdi_provenance = _load_development_cdi(
        dates=dates,
        cdi_path=Path(cdi_path),
        expected_sha256=cdi_sha256,
        experiment52_cdi_path=Path(experiment52_cdi_path),
        experiment52_expected_sha256=experiment52_cdi_sha256,
    )
    bova11 = load_bova11_series(
        bova11_root,
        expected_manifest_sha256=bova11_manifest_sha256,
        canonical_dates=dates.astype("datetime64[D]").astype(object).tolist(),
    )
    bova11_binding = {
        "root": str(Path(bova11_root).resolve(strict=True)),
        "hedge_beta_root": str(hedge_beta_root.resolve(strict=True)),
        "hedge_beta_manifest_sha256": hedge_beta_manifest_sha256,
        "manifest_sha256": bova11.manifest_sha256,
        "data_sha256": bova11.data_sha256,
    }
    lending_borrow = load_lending_borrow_panels(
        lending_archive_root,
        expected_manifest_sha256=lending_archive_manifest_sha256,
        canonical_dates=dates.astype("datetime64[D]").astype(object).tolist(),
        canonical_isins=[
            str(value)
            for value in np.load(store_path / "isin_index.npy", allow_pickle=False)
        ],
    )
    requested_indices = np.unique(
        np.concatenate(
            (
                pretrain_fit,
                pretrain_selection,
                fit_indices["F1"],
                fit_target_window_indices["F1"],
                selection_indices["F1"],
                evaluation_indices["F1"],
            )
        )
    ).astype(np.int64, copy=False)
    requested_dates = _dates_for_indices(dates, requested_indices)
    if any(value >= OFFICIAL_START for value in requested_dates):
        raise PermissionError("network continuation refuses every 2025/2026 session")
    history_lookbacks = np.full(
        len(requested_indices), runtime.slow_lookback, dtype=np.int64
    )
    history_end_offsets = np.where(
        dates[requested_indices] <= np.datetime64(PRETRAIN_END), 0, -1
    ).astype(np.int64)
    store, source_access = open_store_for_samples(
        store_path,
        requested_indices,
        purpose="evaluation",
        history_lookbacks=history_lookbacks,
        history_end_offsets=history_end_offsets,
    )
    source_hashes = {
        "v2_store_manifest": actual_store_manifest_sha,
        "cdi_development_extension": str(
            cdi_provenance["development_extension"]["sha256"]
        ),
        "cdi_experiment52_reference": str(
            cdi_provenance["experiment52_reference"]["sha256"]
        ),
        "bova11_manifest": bova11.manifest_sha256,
        "bova11_data": bova11.data_sha256,
        "completed_classical_inventory": completed_classical_inventory_sha256,
        "completed_classical_failure_record": completed_classical_failure_sha256,
    }
    output.mkdir(parents=True, exist_ok=False)
    try:
        network = _run_network_smokes(
            store=store,
            fit_indices=fit_indices["F1"],
            fit_target_window_indices=fit_target_window_indices["F1"],
            selection_indices=selection_indices["F1"],
            evaluation_indices=evaluation_indices["F1"],
            pretrain_fit_indices=pretrain_fit,
            pretrain_selection_indices=pretrain_selection,
            cdi_by_index=cdi_by_index,
            bova11_close_by_index=bova11.close_by_session,
            bova11_binding=bova11_binding,
            lending_borrow=lending_borrow,
            root=output / "network_smokes",
            source_hashes=source_hashes,
            runtime=runtime,
            sidecars=sidecars,
            seed=11,
        )
        protocol_hashes = {
            name: {
                "path": str(PROTOCOL_CONFIG_ROOT / f"{name}.json"),
                "sha256": sha256_file(PROTOCOL_CONFIG_ROOT / f"{name}.json"),
            }
            for name in ("triage", "full")
        }
        manifest_path = output / "pipeline_network_resume_manifest.json"
        manifest_sha = write_json_atomic(
            manifest_path,
            {
                "schema": PIPELINE_NETWORK_RESUME_SCHEMA,
                "status": "completed",
                **PIPELINE_FLAGS,
                "scope": (
                    "development-only F1 neural integration continuation; "
                    "numbers are not research claims"
                ),
                "code": code,
                "store_build_commit": store_build_commit,
                "runtime": asdict(runtime),
                "protocols": protocol_hashes,
                "enabled_sidecars": list(sidecars),
                "sources": {
                    "store": {
                        "root": str(store_path),
                        "manifest_sha256": actual_store_manifest_sha,
                        "access_ledger": source_access.payload(),
                        "external_artifact_resolutions": external_resolutions,
                    },
                    "cdi": cdi_provenance,
                    "bova11": bova11_binding,
                    "completed_classical_validation": classical_source,
                },
                "date_contract": {
                    "minimum_date": min(requested_dates).isoformat(),
                    "maximum_date": max(requested_dates).isoformat(),
                    "official_validation_accessed": False,
                    "test_accessed": False,
                    "pretrain_fit_date_indices": pretrain_fit.tolist(),
                    "pretrain_embargo_date_indices": pretrain_embargo.tolist(),
                    "pretrain_selection_date_indices": pretrain_selection.tolist(),
                    "F1": fold_payload["F1"],
                },
                "results": {"network_smokes": network},
            },
        )
        excluded = {"inventory.json", "inventory.json.sha256"}
        rows = inventory(output, exclude=excluded)
        inventory_path = output / "inventory.json"
        inventory_sha = write_json_atomic(
            inventory_path,
            {
                "schema": f"{PIPELINE_NETWORK_RESUME_SCHEMA}_INVENTORY",
                "status": "completed",
                **PIPELINE_FLAGS,
                "excluded_self": sorted(excluded),
                "files": rows,
            },
        )
        _verify_inventory_rows(output, rows, excluded=excluded)
    finally:
        store.close()
    return PipelineValidationResult(
        root=output,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        inventory_path=inventory_path,
        inventory_sha256=inventory_sha,
    )


def _optional_positive(value: str) -> int:
    parsed = int(value)
    if parsed < _MIN_WINDOW_SESSIONS:
        raise argparse.ArgumentTypeError(
            f"bounded session windows must be at least {_MIN_WINDOW_SESSIONS}"
        )
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the non-research v2 development pipeline validation."
    )
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument(
        "--cdi-path",
        type=Path,
        required=True,
        help="Development-extension daily_cdi.parquet used for validation scoring.",
    )
    parser.add_argument(
        "--cdi-sha256",
        required=True,
        help="Expected SHA-256 of the development-extension CDI file.",
    )
    parser.add_argument(
        "--experiment52-cdi-path",
        type=Path,
        required=True,
        help="Exact Experiment-52 reference daily_cdi.parquet.",
    )
    parser.add_argument(
        "--experiment52-cdi-sha256",
        required=True,
        help="Expected SHA-256 of the Experiment-52 CDI reference.",
    )
    parser.add_argument(
        "--bova11-root",
        type=Path,
        required=True,
        help="Immutable development-only exact-identity BOVA11 close artifact.",
    )
    parser.add_argument("--bova11-manifest-sha256", required=True)
    parser.add_argument("--hedge-beta-root", type=Path, required=True)
    parser.add_argument("--hedge-beta-manifest-sha256", required=True)
    parser.add_argument(
        "--lending-archive-root",
        type=Path,
        required=True,
        help="Immutable rev4b direct lending archive.",
    )
    parser.add_argument("--lending-archive-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--native-fast-audit", type=Path)
    parser.add_argument("--native-fast-audit-sha256")
    parser.add_argument(
        "--store-manifest-sha256",
        help="Exact sealed-store manifest hash required for a network continuation.",
    )
    parser.add_argument(
        "--completed-classical-root",
        type=Path,
        help="Sealed partial validation root containing completed classical legs.",
    )
    parser.add_argument("--completed-classical-inventory-sha256")
    parser.add_argument("--completed-classical-failure-sha256")
    parser.add_argument(
        "--prior-classical-root",
        type=Path,
        help=(
            "Sealed completed classical acceptance whose score panels are reused "
            "for a ledger-only replay."
        ),
    )
    parser.add_argument("--prior-classical-manifest-sha256")
    parser.add_argument("--prior-classical-inventory-sha256")
    parser.add_argument("--sidecar", action="append", default=[])
    parser.add_argument("--fine-epochs", type=int, default=3)
    parser.add_argument("--handoff-epochs", type=int, default=1)
    parser.add_argument("--gbdt-maximum-rounds", type=int, default=3000)
    parser.add_argument("--gbdt-early-stopping-rounds", type=int, default=100)
    parser.add_argument("--gbdt-num-threads", type=int, default=0)
    parser.add_argument("--max-fit-sessions", type=_optional_positive)
    parser.add_argument("--max-selection-sessions", type=_optional_positive)
    parser.add_argument("--max-pretrain-fit-sessions", type=_optional_positive)
    parser.add_argument("--max-pretrain-selection-sessions", type=_optional_positive)
    parser.add_argument(
        "--slow-lookback", type=int, choices=ALLOWED_LOOKBACKS, default=60
    )
    parser.add_argument("--pairs-per-batch", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument(
        "--compile-forward", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    runtime = ValidationRuntime(
        fine_epochs=arguments.fine_epochs,
        handoff_epochs=arguments.handoff_epochs,
        gbdt_maximum_rounds=arguments.gbdt_maximum_rounds,
        gbdt_early_stopping_rounds=arguments.gbdt_early_stopping_rounds,
        gbdt_num_threads=arguments.gbdt_num_threads,
        max_fit_sessions=arguments.max_fit_sessions,
        max_selection_sessions=arguments.max_selection_sessions,
        max_pretrain_fit_sessions=arguments.max_pretrain_fit_sessions,
        max_pretrain_selection_sessions=arguments.max_pretrain_selection_sessions,
        slow_lookback=arguments.slow_lookback,
        pairs_per_batch=arguments.pairs_per_batch,
        evaluation_batch_size=arguments.evaluation_batch_size,
        compile_forward=arguments.compile_forward,
        device=arguments.device,
    )
    continuation = (
        arguments.store_manifest_sha256,
        arguments.completed_classical_root,
        arguments.completed_classical_inventory_sha256,
        arguments.completed_classical_failure_sha256,
    )
    ledger_replay = (
        arguments.store_manifest_sha256,
        arguments.prior_classical_root,
        arguments.prior_classical_manifest_sha256,
        arguments.prior_classical_inventory_sha256,
    )
    if any(value is not None for value in ledger_replay[1:]):
        if any(value is None for value in ledger_replay):
            raise ValueError("ledger replay requires all four source bindings")
        if any(value is not None for value in continuation[1:]):
            raise ValueError("ledger replay and network continuation are exclusive")
        result = replay_classical_economics(
            store_root=arguments.store_root,
            store_manifest_sha256=arguments.store_manifest_sha256,
            cdi_path=arguments.cdi_path,
            cdi_sha256=arguments.cdi_sha256,
            experiment52_cdi_path=arguments.experiment52_cdi_path,
            experiment52_cdi_sha256=arguments.experiment52_cdi_sha256,
            bova11_root=arguments.bova11_root,
            bova11_manifest_sha256=arguments.bova11_manifest_sha256,
            hedge_beta_root=arguments.hedge_beta_root,
            hedge_beta_manifest_sha256=arguments.hedge_beta_manifest_sha256,
            lending_archive_root=arguments.lending_archive_root,
            lending_archive_manifest_sha256=(arguments.lending_archive_manifest_sha256),
            prior_classical_root=arguments.prior_classical_root,
            prior_classical_manifest_sha256=(arguments.prior_classical_manifest_sha256),
            prior_classical_inventory_sha256=(
                arguments.prior_classical_inventory_sha256
            ),
            output_root=arguments.output_root,
        )
    elif any(value is not None for value in continuation):
        if any(value is None for value in continuation):
            raise ValueError("network continuation requires all four source bindings")
        result = resume_network_validation(
            store_root=arguments.store_root,
            store_manifest_sha256=arguments.store_manifest_sha256,
            cdi_path=arguments.cdi_path,
            cdi_sha256=arguments.cdi_sha256,
            experiment52_cdi_path=arguments.experiment52_cdi_path,
            experiment52_cdi_sha256=arguments.experiment52_cdi_sha256,
            bova11_root=arguments.bova11_root,
            bova11_manifest_sha256=arguments.bova11_manifest_sha256,
            hedge_beta_root=arguments.hedge_beta_root,
            hedge_beta_manifest_sha256=arguments.hedge_beta_manifest_sha256,
            lending_archive_root=arguments.lending_archive_root,
            lending_archive_manifest_sha256=(arguments.lending_archive_manifest_sha256),
            completed_classical_root=arguments.completed_classical_root,
            completed_classical_inventory_sha256=(
                arguments.completed_classical_inventory_sha256
            ),
            completed_classical_failure_sha256=(
                arguments.completed_classical_failure_sha256
            ),
            output_root=arguments.output_root,
            runtime=runtime,
            enabled_sidecars=arguments.sidecar,
        )
    else:
        result = run_pipeline_validation(
            store_root=arguments.store_root,
            cdi_path=arguments.cdi_path,
            cdi_sha256=arguments.cdi_sha256,
            experiment52_cdi_path=arguments.experiment52_cdi_path,
            experiment52_cdi_sha256=arguments.experiment52_cdi_sha256,
            bova11_root=arguments.bova11_root,
            bova11_manifest_sha256=arguments.bova11_manifest_sha256,
            hedge_beta_root=arguments.hedge_beta_root,
            hedge_beta_manifest_sha256=arguments.hedge_beta_manifest_sha256,
            lending_archive_root=arguments.lending_archive_root,
            lending_archive_manifest_sha256=(arguments.lending_archive_manifest_sha256),
            output_root=arguments.output_root,
            runtime=runtime,
            enabled_sidecars=arguments.sidecar,
            native_fast_audit_path=arguments.native_fast_audit,
            native_fast_audit_sha256=arguments.native_fast_audit_sha256,
        )
    print(
        json.dumps(
            {
                "root": str(result.root),
                "manifest": str(result.manifest_path),
                "manifest_sha256": result.manifest_sha256,
                "inventory": str(result.inventory_path),
                "inventory_sha256": result.inventory_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
