from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import polars as pl
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from brazil_rv.execution.inputs import load_daily_cdi_rates

from .artifacts import inventory, sha256_file, write_json_atomic
from .baselines import BaselinePanel, build_baselines
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
    HORIZONS,
    OFFICIAL_START,
    PRETRAIN_END,
    STORE_START,
)
from .data import V2DailyDataset
from .evaluate import EvaluationInputs, EvaluationResult, evaluate_scores
from .gbdt import GBDTConfig, MultiHorizonGBDT, assemble_gbdt_features
from .score import ScoreArtifact, score_checkpoint_artifact
from .splits import AccessPurpose, development_folds
from .store import STORE_SCHEMA, V2Store, open_store_for_samples
from .train import (
    DatePairBatchSampler,
    StageTrainingResult,
    block_parity_mask,
    pretrain_internal_split,
    stitch_block_parity_predictions,
    train_stage,
)

PIPELINE_SCHEMA = "BRAZIL_RV_V2_PIPELINE_VALIDATION_V1"
PIPELINE_FLAGS: dict[str, bool] = {
    "pipeline_validation": True,
    "research_claim": False,
    "official_validation_accessed": False,
    "test_accessed": False,
}
_MIN_WINDOW_SESSIONS = max(HORIZONS) + 2
_REQUIRED_ARRAYS = frozenset(
    {
        "active",
        "ambiguous_action_mask",
        "observed",
        "slow_values",
        "slow_valid",
        "intraday_values",
        "intraday_valid",
        "fast_present",
        "target_primary",
        "target_valid",
        "target_raw_midrank",
        "target_raw_valid",
        "target_raw_log_return",
        "adjusted_close",
        "target_exclusion_event_mask",
    }
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
    if not isinstance(metadata, Mapping) or (
        metadata.get("v1_isin_subset_verified") is not True
        or metadata.get("v1_calendar_verified") is not True
    ):
        raise ValueError("store lacks the canonical v1 identity/calendar assertions")
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
    date_path = store_root / "date_index.npy"
    if not isinstance(date_record, Mapping) or (
        int(date_record.get("bytes", -1)) != date_path.stat().st_size
        or date_record.get("sha256") != sha256_file(date_path)
    ):
        raise ValueError("store date index differs from its immutable manifest")
    dates = np.load(date_path, allow_pickle=False)
    if dates.dtype.kind != "M" or dates.ndim != 1:
        raise ValueError("store date index has the wrong contract")
    python_dates = tuple(dates.astype("datetime64[D]").astype(object).tolist())
    if (
        not python_dates
        or python_dates[0] != STORE_START
        or python_dates[-1] < DEVELOPMENT_END
    ):
        raise ValueError("store does not span the frozen v2 development foundation")
    return manifest, dates


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
) -> tuple[DataLoader[dict[str, object]], DataLoader[dict[str, object]]]:
    fit = _dataset(
        store,
        fit_indices,
        stage=stage,
        purpose="training",
        lookback=runtime.slow_lookback,
        sidecars=sidecars,
    )
    selection = _dataset(
        store,
        selection_indices,
        stage=stage,
        purpose="selection",
        lookback=runtime.slow_lookback,
        sidecars=sidecars,
    )
    sampler = DatePairBatchSampler(
        fit.date_indices,
        pairs_per_batch=runtime.pairs_per_batch,
        seed=seed,
        time_decay_half_life=time_decay_half_life,
        drop_last=True,
    )
    return (
        DataLoader(fit, batch_sampler=sampler, num_workers=0),
        DataLoader(
            selection,
            batch_size=runtime.evaluation_batch_size,
            shuffle=False,
            num_workers=0,
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
    return DataLoader(
        dataset,
        batch_size=runtime.evaluation_batch_size,
        shuffle=False,
        num_workers=0,
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
    payload.update(PIPELINE_FLAGS)
    return write_json_atomic(path, payload)


def _window_target_mask(
    values: NDArray[np.bool_], indices: NDArray[np.int64]
) -> NDArray[np.bool_]:
    mask = np.asarray(values, dtype=np.bool_).copy()
    if (
        mask.ndim != 3
        or mask.shape[0] != len(indices)
        or mask.shape[-1] != len(HORIZONS)
    ):
        raise ValueError("window target mask has the wrong shape")
    index_set = frozenset(int(value) for value in indices)
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
    source_hashes: Mapping[str, str],
    pathwise_scores: tuple[NDArray[np.floating], ...] = (),
    pathwise_score_masks: tuple[NDArray[np.bool_], ...] = (),
) -> EvaluationInputs:
    targets = store.read("target_primary", indices)
    raw_targets = store.read("target_raw_midrank", indices)
    raw_returns = store.read("target_raw_log_return", indices)
    target_mask = _window_target_mask(
        store.read("target_valid", indices), indices
    )
    raw_target_mask = _window_target_mask(
        store.read("target_raw_valid", indices), indices
    )
    targets = np.where(target_mask, targets, 0.0)
    raw_targets = np.where(raw_target_mask, raw_targets, 0.0)
    raw_returns = np.where(raw_target_mask, raw_returns, 0.0)
    axes = store.manifest.get("axes")
    if not isinstance(axes, Mapping):
        raise ValueError("store manifest lacks its canonical axes")
    calendar_sha = axes.get("date_identity_sha256")
    if not isinstance(calendar_sha, str):
        raise ValueError("store manifest lacks its canonical calendar identity")
    cdi = np.asarray(cdi_by_index[indices], dtype=np.float64)
    if not np.isfinite(cdi).all():
        raise ValueError("development CDI is incomplete for the evaluation window")
    return EvaluationInputs(
        dates=_dates_for_indices(store.dates, indices),
        session_indices=indices.copy(),
        calendar_identity_sha256=calendar_sha,
        scores=np.asarray(scores),
        score_mask=np.asarray(score_mask, dtype=np.bool_),
        residual_midrank_targets=targets,
        raw_midrank_targets=raw_targets,
        raw_log_returns=raw_returns,
        target_mask=target_mask,
        raw_target_mask=raw_target_mask,
        active=np.asarray(store.read("active", indices), dtype=np.bool_),
        adjusted_close=store.read("adjusted_close", indices),
        target_exclusion_event=np.asarray(
            store.read("target_exclusion_event_mask", indices), dtype=np.bool_
        ),
        cdi_returns=cdi,
        source_artifact_hashes=dict(source_hashes),
        pathwise_scores=pathwise_scores,
        pathwise_score_masks=pathwise_score_masks,
    )


def _evaluate_and_write(
    *,
    store: V2Store,
    indices: NDArray[np.int64],
    scores: NDArray[np.floating],
    score_mask: NDArray[np.bool_],
    cdi_by_index: NDArray[np.float64],
    source_hashes: Mapping[str, str],
    window_name: str,
    path: Path,
    pathwise_scores: tuple[NDArray[np.floating], ...] = (),
    pathwise_score_masks: tuple[NDArray[np.bool_], ...] = (),
) -> tuple[EvaluationResult, str]:
    result = evaluate_scores(
        _evaluation_inputs(
            store,
            indices,
            scores,
            score_mask,
            cdi_by_index,
            source_hashes,
            pathwise_scores,
            pathwise_score_masks,
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
    )
    return {
        "report": str(report_path),
        "report_sha256": report_sha256,
        "pooled_primary_ic": result.report["pooled_primary_ic"],
        "headline_economics": dict(headline),
    }


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
    if np.any(indices <= 0):
        raise ValueError("fine-tune GBDT rows require a prior slow session")
    slow_parts = [store.read("slow_values", indices - 1)]
    slow_parts.extend(
        store.read(f"sidecar_{group}_values", indices - 1)
        for group in sidecars
    )
    slow = np.concatenate(slow_parts, axis=-1)[:, :, None, :]
    intraday = store.read("intraday_values", indices)
    fast_present = np.asarray(store.read("fast_present", indices), dtype=np.bool_)
    days = np.ones(fast_present.shape, dtype=np.float32)
    return assemble_gbdt_features(slow, intraday, fast_present, days)


def _gbdt_feature_names(store: V2Store, sidecars: Sequence[str]) -> tuple[str, ...]:
    names = store.manifest.get("feature_names")
    if not isinstance(names, Mapping):
        raise ValueError("store manifest lacks feature names")
    result = list(names.get("slow", ()))
    for group in sidecars:
        result.extend(names.get(f"sidecar_{group}", ()))
    result.extend(names.get("intraday", ()))
    result.extend(("fast_present", "days_since_last_slow_row"))
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
    parity: int | None,
    output_dir: Path,
    model_config: ModelConfig,
    maximum_epochs: int,
    runtime: ValidationRuntime,
    sidecars: Sequence[str],
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
        selection_parity=parity,
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
    _mark_pipeline_manifest(result.manifest_path)
    return result


def _run_baselines(
    *,
    store: V2Store,
    fold_indices: Mapping[str, NDArray[np.int64]],
    cdi_by_index: NDArray[np.float64],
    root: Path,
    source_hashes: Mapping[str, str],
) -> list[dict[str, object]]:
    first_index = min(int(indices[0]) for indices in fold_indices.values())
    last_index = max(int(indices[-1]) for indices in fold_indices.values())
    baseline_start = max(0, first_index - 253)
    baseline_indices = np.arange(
        baseline_start, last_index + 1, dtype=np.int64
    )
    close = store.read("adjusted_close", baseline_indices)
    observed = np.asarray(
        store.read("observed", baseline_indices), dtype=np.bool_
    )
    active = np.asarray(
        store.read("active", baseline_indices), dtype=np.bool_
    )
    ambiguous = np.asarray(
        store.read("ambiguous_action_mask", baseline_indices), dtype=np.bool_
    )
    panels = build_baselines(
        close, observed, active, ambiguous, slow_lag=1
    )
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
                    "slow_lag_sessions": 1,
                    "date_indices": indices.tolist(),
                },
            )
            result, report_sha = _evaluate_and_write(
                store=store,
                indices=indices,
                scores=panel.scores[local_indices],
                score_mask=panel.score_mask[local_indices],
                cdi_by_index=cdi_by_index,
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
    selection_indices: Mapping[str, NDArray[np.int64]],
    cdi_by_index: NDArray[np.float64],
    root: Path,
    source_hashes: Mapping[str, str],
    runtime: ValidationRuntime,
    sidecars: Sequence[str],
    seed: int,
) -> list[dict[str, object]]:
    config = GBDTConfig(
        maximum_rounds=runtime.gbdt_maximum_rounds,
        early_stopping_rounds=runtime.gbdt_early_stopping_rounds,
        seeds=(seed,),
        num_threads=runtime.gbdt_num_threads,
    )
    feature_names = _gbdt_feature_names(store, sidecars)
    records: list[dict[str, object]] = []
    for fold, train_indices in fit_indices.items():
        evaluation_indices = selection_indices[fold]
        train_features = _gbdt_features(store, train_indices, sidecars)
        evaluation_features = _gbdt_features(store, evaluation_indices, sidecars)
        if train_features.shape[-1] != len(feature_names):
            raise ValueError("GBDT feature names differ from the assembled width")
        train_targets = store.read("target_primary", train_indices)
        train_mask = _window_target_mask(
            store.read("target_valid", train_indices), train_indices
        )
        selection_targets = store.read("target_primary", evaluation_indices)
        selection_mask = _window_target_mask(
            store.read("target_valid", evaluation_indices),
            evaluation_indices,
        )
        active = np.asarray(
            store.read("active", evaluation_indices), dtype=np.bool_
        )
        score_mask = np.repeat(active[..., None], len(HORIZONS), axis=-1)
        predictions: dict[int, NDArray[np.float32]] = {}
        importance: dict[str, object] = {}
        model_artifacts: dict[str, object] = {}
        for parity in (0, 1):
            selected = block_parity_mask(len(evaluation_indices), parity)
            model = MultiHorizonGBDT(config, feature_names=feature_names)
            model.fit(
                train_features,
                train_targets,
                train_mask,
                evaluation_features[selected],
                selection_targets[selected],
                selection_mask[selected],
                train_dates=train_indices,
                validation_dates=evaluation_indices[selected],
            )
            predictions[parity] = model.predict_ranks(evaluation_features, score_mask)
            label = "even" if parity == 0 else "odd"
            importance[f"selected_on_{label}"] = {
                name: values.tolist()
                for name, values in model.feature_importance(
                    evaluation_features
                ).items()
            }
            model_artifacts[f"selected_on_{label}"] = _persist_gbdt_models(
                model,
                root / "models" / fold / f"selected_on_{label}",
                verification_features=evaluation_features,
                verification_mask=score_mask,
            )
        stitched = stitch_block_parity_predictions(
            predictions[0], predictions[1]
        ).astype(np.float32, copy=False)
        artifact_root = root / fold
        manifest_path, manifest_sha = _persist_score_panel(
            artifact_root,
            {
                "scores": stitched,
                "score_mask": score_mask,
                "selected_on_even_scores": predictions[0],
                "selected_on_odd_scores": predictions[1],
            },
            {
                "engine": "lightgbm",
                "fold": fold,
                "seed": seed,
                "config": asdict(config),
                "feature_names": list(feature_names),
                "feature_importance": importance,
                "model_artifacts": model_artifacts,
                "cross_fit": (
                    "5-session block parity; each parity-selected model scores "
                    "only the opposite parity"
                ),
                "fit_date_indices": train_indices.tolist(),
                "selection_date_indices": evaluation_indices.tolist(),
            },
        )
        result, report_sha = _evaluate_and_write(
            store=store,
            indices=evaluation_indices,
            scores=stitched,
            score_mask=score_mask,
            cdi_by_index=cdi_by_index,
            source_hashes={**source_hashes, "score_manifest": manifest_sha},
            window_name=fold,
            path=artifact_root / "evaluation.json",
            pathwise_scores=(predictions[0], predictions[1]),
            pathwise_score_masks=(score_mask, score_mask),
        )
        records.append(
            {
                "engine": "gbdt",
                "fold": fold,
                "seed": seed,
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
    reloaded = type(model).load(
        root, expected_manifest_sha256=manifest_sha
    )
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
    selection_indices: NDArray[np.int64],
    pretrain_fit_indices: NDArray[np.int64],
    pretrain_selection_indices: NDArray[np.int64],
    cdi_by_index: NDArray[np.float64],
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
    scratch_results: dict[int, StageTrainingResult] = {}
    scratch_scores: dict[int, ScoreArtifact] = {}
    for parity in (0, 1):
        label = "even" if parity == 0 else "odd"
        trained = _train_once(
            store=store,
            fit_indices=fit_indices,
            selection_indices=selection_indices,
            stage="F",
            seed=seed,
            fold=f"F1_select_{label}",
            parity=parity,
            output_dir=root / "from_scratch" / f"select_{label}_training",
            model_config=base_config,
            maximum_epochs=runtime.fine_epochs,
            runtime=runtime,
            sidecars=sidecars,
        )
        scratch_results[parity] = trained
        scratch_scores[parity] = _score_once(
            store=store,
            indices=selection_indices,
            checkpoint=trained.raw_patience_checkpoint,
            model_config=base_config,
            output_dir=root / "from_scratch" / f"select_{label}_scores",
            runtime=runtime,
            sidecars=sidecars,
        )
    selected_on_even = np.load(scratch_scores[0].scores_path, allow_pickle=False)
    selected_on_odd = np.load(scratch_scores[1].scores_path, allow_pickle=False)
    even_mask = np.load(scratch_scores[0].score_mask_path, allow_pickle=False)
    odd_mask = np.load(scratch_scores[1].score_mask_path, allow_pickle=False)
    if not np.array_equal(even_mask, odd_mask):
        raise ValueError("block-parity model score masks differ")
    stitched = stitch_block_parity_predictions(
        selected_on_even, selected_on_odd
    ).astype(np.float32, copy=False)
    crossfit_root = root / "from_scratch" / "crossfit"
    score_manifest, score_manifest_sha = _persist_score_panel(
        crossfit_root,
        {"scores": stitched, "score_mask": even_mask},
        {
            "engine": "daily_multi_horizon_network",
            "fold": "F1",
            "seed": seed,
            "lambda_persistence": 0.0,
            "epochs_cap": runtime.fine_epochs,
            "cross_fit": (
                "5-session block parity; each parity-selected model scores only "
                "the opposite parity"
            ),
            "selected_on_even_score_manifest_sha256": sha256_file(
                scratch_scores[0].manifest_path
            ),
            "selected_on_odd_score_manifest_sha256": sha256_file(
                scratch_scores[1].manifest_path
            ),
        },
    )
    network_sources = {
        **source_hashes,
        "score_manifest": score_manifest_sha,
        "selected_on_even_score_manifest": sha256_file(scratch_scores[0].manifest_path),
        "selected_on_odd_score_manifest": sha256_file(scratch_scores[1].manifest_path),
    }
    evaluated, report_sha = _evaluate_and_write(
        store=store,
        indices=selection_indices,
        scores=stitched,
        score_mask=even_mask,
        cdi_by_index=cdi_by_index,
        source_hashes=network_sources,
        window_name="F1",
        path=crossfit_root / "evaluation.json",
        pathwise_scores=(selected_on_even, selected_on_odd),
        pathwise_score_masks=(even_mask, odd_mask),
    )

    persistence_config = replace(base_config, lambda_persistence=0.1)
    persistence_results: dict[int, StageTrainingResult] = {}
    persistence_scores: dict[int, ScoreArtifact] = {}
    for parity in (0, 1):
        label = "even" if parity == 0 else "odd"
        trained = _train_once(
            store=store,
            fit_indices=fit_indices,
            selection_indices=selection_indices,
            stage="F",
            seed=seed,
            fold=f"F1_lambda_persistence_0_1_select_{label}",
            parity=parity,
            output_dir=(root / "persistence_lambda_0_1" / f"select_{label}_training"),
            model_config=persistence_config,
            maximum_epochs=1,
            runtime=runtime,
            sidecars=sidecars,
        )
        persistence_results[parity] = trained
        persistence_scores[parity] = _score_once(
            store=store,
            indices=selection_indices,
            checkpoint=trained.raw_patience_checkpoint,
            model_config=persistence_config,
            output_dir=(root / "persistence_lambda_0_1" / f"select_{label}_scores"),
            runtime=runtime,
            sidecars=sidecars,
        )
    persistence_even = np.load(persistence_scores[0].scores_path, allow_pickle=False)
    persistence_odd = np.load(persistence_scores[1].scores_path, allow_pickle=False)
    persistence_even_mask = np.load(
        persistence_scores[0].score_mask_path, allow_pickle=False
    )
    persistence_odd_mask = np.load(
        persistence_scores[1].score_mask_path, allow_pickle=False
    )
    if not np.array_equal(persistence_even_mask, persistence_odd_mask):
        raise ValueError("persistence-probe block-parity score masks differ")
    persistence_stitched = stitch_block_parity_predictions(
        persistence_even, persistence_odd
    ).astype(np.float32, copy=False)
    persistence_crossfit = root / "persistence_lambda_0_1" / "crossfit"
    persistence_manifest, persistence_manifest_sha = _persist_score_panel(
        persistence_crossfit,
        {
            "scores": persistence_stitched,
            "score_mask": persistence_even_mask,
        },
        {
            "engine": "daily_multi_horizon_network",
            "fold": "F1",
            "seed": seed,
            "lambda_persistence": 0.1,
            "epochs_cap": 1,
            "cross_fit": (
                "5-session block parity; each parity-selected model scores only "
                "the opposite parity"
            ),
            "selected_on_even_score_manifest_sha256": sha256_file(
                persistence_scores[0].manifest_path
            ),
            "selected_on_odd_score_manifest_sha256": sha256_file(
                persistence_scores[1].manifest_path
            ),
        },
    )
    persistence_evaluated, persistence_report_sha = _evaluate_and_write(
        store=store,
        indices=selection_indices,
        scores=persistence_stitched,
        score_mask=persistence_even_mask,
        cdi_by_index=cdi_by_index,
        source_hashes={
            **source_hashes,
            "score_manifest": persistence_manifest_sha,
            "selected_on_even_score_manifest": sha256_file(
                persistence_scores[0].manifest_path
            ),
            "selected_on_odd_score_manifest": sha256_file(
                persistence_scores[1].manifest_path
            ),
        },
        window_name="F1_lambda_persistence_0_1",
        path=persistence_crossfit / "evaluation.json",
        pathwise_scores=(persistence_even, persistence_odd),
        pathwise_score_masks=(persistence_even_mask, persistence_odd_mask),
    )

    pretrain = _train_once(
        store=store,
        fit_indices=pretrain_fit_indices,
        selection_indices=pretrain_selection_indices,
        stage="P",
        seed=seed,
        fold="pretrain_internal",
        parity=None,
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
        fold="F1_pretrain_handoff_select_even",
        parity=0,
        output_dir=root / "pretrain_handoff" / "stage_f_select_even",
        model_config=base_config,
        maximum_epochs=runtime.handoff_epochs,
        runtime=runtime,
        sidecars=sidecars,
        pretrain_checkpoint=pretrain.raw_patience_checkpoint,
        expected_pretrain_sha256=pretrain_sha,
    )
    handoff_manifest = json.loads(handoff.manifest_path.read_text(encoding="utf-8"))
    if handoff_manifest.get("pretrain_checkpoint_sha256") != pretrain_sha:
        raise ValueError("stage-P checkpoint handoff is not hash-bound in stage F")
    return {
        "from_scratch": {
            "fold": "F1",
            "seed": seed,
            "epochs_cap": runtime.fine_epochs,
            "training_manifests": {
                "selected_on_even": str(scratch_results[0].manifest_path),
                "selected_on_odd": str(scratch_results[1].manifest_path),
            },
            "score_manifest": str(score_manifest),
            "score_manifest_sha256": score_manifest_sha,
            "evaluation": _evaluation_summary(
                evaluated, crossfit_root / "evaluation.json", report_sha
            ),
        },
        "persistence_probe": {
            "lambda_persistence": 0.1,
            "epochs_cap": 1,
            "training_manifests": {
                "selected_on_even": str(persistence_results[0].manifest_path),
                "selected_on_odd": str(persistence_results[1].manifest_path),
            },
            "score_manifest": str(persistence_manifest),
            "score_manifest_sha256": persistence_manifest_sha,
            "evaluation": _evaluation_summary(
                persistence_evaluated,
                persistence_crossfit / "evaluation.json",
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


def _validate_sidecars(
    store_manifest: Mapping[str, object], sidecars: Sequence[str]
) -> tuple[str, ...]:
    normalized = tuple(sidecars)
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
        if values not in arrays or valid not in arrays or group not in feature_names:
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
    dict[str, object],
]:
    python_dates = tuple(dates.astype("datetime64[D]").astype(object).tolist())
    folds = {fold.name: fold for fold in development_folds(python_dates)}
    fit: dict[str, NDArray[np.int64]] = {}
    selection: dict[str, NDArray[np.int64]] = {}
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
        payload[name] = {
            **fold.payload(),
            "validation_fit_date_indices": fit[name].tolist(),
            "validation_selection_date_indices": selection[name].tolist(),
        }
    return fit, selection, payload


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
    for label, rows in (("development extension", extension), ("Experiment-52", reference)):
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
        reference[column].to_numpy().tobytes()
        == overlap[column].to_numpy().tobytes()
        for column in columns
    )
    if not exact_byte_match:
        raise ValueError(
            "development CDI extension differs from the Experiment-52 reference"
        )
    rate_difference = np.abs(
        overlap["daily_cdi_rate"].to_numpy()
        - reference["daily_cdi_rate"].to_numpy()
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


def run_pipeline_validation(
    *,
    store_root: Path,
    cdi_path: Path,
    cdi_sha256: str,
    experiment52_cdi_path: Path,
    experiment52_cdi_sha256: str,
    output_root: Path,
    runtime: ValidationRuntime = ValidationRuntime(),
    enabled_sidecars: Sequence[str] = (),
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
    if (
        output == store_path
        or output.is_relative_to(store_path)
        or store_path.is_relative_to(output)
    ):
        raise ValueError("validation output and immutable input store must be disjoint")
    code = _git_identity()
    store_manifest, dates = _read_store_header(store_path)
    store_metadata = store_manifest.get("metadata")
    if (
        not isinstance(store_metadata, Mapping)
        or store_metadata.get("implementation_git_commit") != code["commit"]
    ):
        raise ValueError(
            "v2 store implementation commit differs from the validation code"
        )
    sidecars = _validate_sidecars(store_manifest, enabled_sidecars)
    fit_indices, selection_indices, fold_payload = _development_indices(
        dates, runtime=runtime
    )
    pretrain_fit, pretrain_embargo, pretrain_selection = _pretrain_indices(
        dates, runtime
    )
    triage = protocol_preset("triage")
    full = protocol_preset("full")
    if triage.folds != ("F1", "F2") or triage.seeds != (11,):
        raise ValueError("triage protocol differs from the validation contract")
    if full.folds != ("F1", "F2", "F3"):
        raise ValueError("full protocol differs from the validation contract")
    missing_folds = set(full.folds) - set(selection_indices)
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
    store_manifest_sha = sha256_file(store_path / "manifest.json")
    requested_groups = (
        pretrain_fit,
        pretrain_selection,
        *[fit_indices[name] for name in full.folds],
        *[selection_indices[name] for name in full.folds],
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
    baseline_samples = np.concatenate(
        [selection_indices[name] for name in full.folds]
    )
    history_lookbacks[
        np.isin(requested_indices, baseline_samples)
    ] = 253
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
    }
    output.mkdir(parents=True, exist_ok=False)
    try:
        baseline_records = _run_baselines(
            store=store,
            fold_indices={name: selection_indices[name] for name in full.folds},
            cdi_by_index=cdi_by_index,
            root=output / "baselines",
            source_hashes=source_hashes,
        )
        gbdt_records = _run_gbdt(
            store=store,
            fit_indices={name: fit_indices[name] for name in triage.folds},
            selection_indices={name: selection_indices[name] for name in triage.folds},
            cdi_by_index=cdi_by_index,
            root=output / "gbdt_triage",
            source_hashes=source_hashes,
            runtime=runtime,
            sidecars=sidecars,
            seed=triage.seeds[0],
        )
        network = _run_network_smokes(
            store=store,
            fit_indices=fit_indices["F1"],
            selection_indices=selection_indices["F1"],
            pretrain_fit_indices=pretrain_fit,
            pretrain_selection_indices=pretrain_selection,
            cdi_by_index=cdi_by_index,
            root=output / "network_smokes",
            source_hashes=source_hashes,
            runtime=runtime,
            sidecars=sidecars,
            seed=triage.seeds[0],
        )
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
                **PIPELINE_FLAGS,
                "scope": (
                    "development-only integration validation; numbers are not "
                    "research claims"
                ),
                "code": code,
                "runtime": asdict(runtime),
                "protocols": protocol_hashes,
                "enabled_sidecars": list(sidecars),
                "sources": {
                    "store": {
                        "root": str(store_path),
                        "manifest_sha256": store_manifest_sha,
                        "access_ledger": source_access.payload(),
                    },
                    "cdi": cdi_provenance,
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
                    "baselines": baseline_records,
                    "gbdt_triage": gbdt_records,
                    "network_smokes": network,
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
    parser.add_argument("--output-root", type=Path, required=True)
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
    result = run_pipeline_validation(
        store_root=arguments.store_root,
        cdi_path=arguments.cdi_path,
        cdi_sha256=arguments.cdi_sha256,
        experiment52_cdi_path=arguments.experiment52_cdi_path,
        experiment52_cdi_sha256=arguments.experiment52_cdi_sha256,
        output_root=arguments.output_root,
        runtime=runtime,
        enabled_sidecars=arguments.sidecar,
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
