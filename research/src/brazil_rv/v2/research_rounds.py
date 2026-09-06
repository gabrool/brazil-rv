from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .artifacts import inventory, sha256_file, verify_inventory, write_json_atomic
from .baselines import BaselinePanel, build_store_baselines
from .config import PROJECT_ROOT
from .contract import (
    GBDT_SEEDS,
    HORIZONS,
    PRETRAIN_END,
    PRIMARY_HORIZONS,
    RUN_MANY_PLAN_SCHEMA,
    SCORE_ARTIFACT_SCHEMA,
    STORE_START,
    TRAINING_STAGE_SCHEMA,
)
from .evaluate import (
    EVALUATION_SCHEMA,
    EvaluationInputs,
    EvaluationResult,
    _input_hashes,
    _paired_primary_daily,
    _primary_daily_metrics,
    _primary_population_components,
    _spearman,
    _validate_paired_identity,
    evaluate_scores,
    headline_ledger_protocol,
    primary_population_protocol,
)
from .data import ScalarFeatureView, read_scalar_feature_view, scalar_feature_names
from .gbdt import (
    GBDTConfig,
    MultiHorizonGBDT,
    assemble_gbdt_scalar_view,
    gbdt_scalar_feature_names,
)
from .splits import (
    FIT_TO_SELECTION_PURGE_SESSIONS,
    SELECTION_SESSIONS,
    SELECTION_TO_EVALUATION_PURGE_SESSIONS,
    development_evaluation_windows,
    development_folds,
)
from .store import V2Store, open_store_for_samples
from .train import rank_average_ensemble
from .validate_pipeline import (
    _date_indices,
    _evaluation_inputs,
    _load_development_cdi,
    _read_store_header,
    _window_target_mask,
)

ROUND1_SCHEMA = "BRAZIL_RV_V2_RESEARCH_ROUND1_CANONICAL_V3"
ROUND2_SCHEMA = "BRAZIL_RV_V2_RESEARCH_ROUND2_CANONICAL_V3"
RESEARCH_SCORE_SCHEMA = "BRAZIL_RV_V2_RESEARCH_SCORE_V3"
PREREGISTRATION = (
    PROJECT_ROOT / "research" / "preregistrations" / "v2_round1_round2_rev2.md"
)
BOOTSTRAP_REPLICATIONS = 10_000
BOOTSTRAP_BLOCK = 20
BOOTSTRAP_SEED = 20260815
NETWORK_SEEDS = (11, 29, 47)
RUNG_GROUPS: dict[str, tuple[str, ...]] = {
    "a_slow": (),
    "b_intraday": (),
    "c_lending": ("lending",),
    "d_all_sidecars": (
        "lending",
        "oddlot",
        "options",
        "rebalance",
        "events",
        "fundamentals",
    ),
}
RESEARCH_FLAGS = {
    "research_claim": True,
    "official_validation_accessed": False,
    "test_accessed": False,
    "deployment_changed": False,
    "transfer_chronology_clean": True,
}
DEVELOPMENT_SOURCE_TIER_LABELS = {
    "action_terms_source": "inferred_cotahist_dismes_v1",
    "schedule_source": "reconstructed_v1",
}
REGISTRATION_PROTOCOL_BEGIN = "<!-- BRAZIL_RV_V2_PROTOCOL_JSON_BEGIN -->"
REGISTRATION_PROTOCOL_END = "<!-- BRAZIL_RV_V2_PROTOCOL_JSON_END -->"


@dataclass(frozen=True)
class _ResearchEvaluation:
    result: EvaluationResult
    inputs: EvaluationInputs


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
        raise RuntimeError("v2 research requires a clean commit-bound worktree")
    if len(commit) != 40:
        raise RuntimeError("invalid git commit identity")
    return {"commit": commit, "tracked_worktree_clean": True}


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def load_registration_protocol(path: Path = PREREGISTRATION) -> dict[str, object]:
    """Parse the sole machine-readable protocol block from a registration."""

    text = path.read_text(encoding="utf-8")
    if (
        text.count(REGISTRATION_PROTOCOL_BEGIN) != 1
        or text.count(REGISTRATION_PROTOCOL_END) != 1
    ):
        raise ValueError("registration must contain exactly one protocol JSON block")
    raw = (
        text.split(REGISTRATION_PROTOCOL_BEGIN, 1)[1]
        .split(REGISTRATION_PROTOCOL_END, 1)[0]
        .strip()
    )
    if not raw.startswith("```json") or not raw.endswith("```"):
        raise ValueError("registration protocol block must be a fenced JSON object")
    payload = json.loads(raw[len("```json") : -len("```")].strip())
    if not isinstance(payload, dict):
        raise ValueError("registration protocol JSON must be an object")
    return payload


def registration_protocol_from_code() -> dict[str, object]:
    """Build protocol facts that registration prose may not override."""

    return {
        "schema": "BRAZIL_RV_V2_REGISTRATION_PROTOCOL_V1",
        "purge_sessions": {
            "fit_to_selection": FIT_TO_SELECTION_PURGE_SESSIONS,
            "selection_to_evaluation": SELECTION_TO_EVALUATION_PURGE_SESSIONS,
        },
        "selection_sessions": SELECTION_SESSIONS,
        "evaluation_window_per_fold": {
            name: {"start": start.isoformat(), "end": end.isoformat()}
            for name, (start, end) in development_evaluation_windows().items()
        },
        "cross_fit": "none",
        "models_per_fold_seed": 1,
        "primary_population_rule": primary_population_protocol(),
        "headline_cell": headline_ledger_protocol(),
        "source_tier_labels": dict(DEVELOPMENT_SOURCE_TIER_LABELS),
    }


def verify_registration_protocol(path: Path = PREREGISTRATION) -> dict[str, object]:
    """Fail before a freeze when registration and executable policy differ."""

    recorded = load_registration_protocol(path)
    expected = registration_protocol_from_code()
    if recorded != expected:
        raise ValueError(
            "machine-readable registration protocol differs from the executable "
            "development-fold, evaluator, ledger, or source-tier contract"
        )
    return recorded


def _source_tier_labels(store_manifest: Mapping[str, object]) -> dict[str, str]:
    metadata = store_manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("store manifest lacks source-tier metadata")
    action = metadata.get("action_terms_source")
    schedule = metadata.get("schedule_source")
    labels = {"action_terms_source": action, "schedule_source": schedule}
    if labels != DEVELOPMENT_SOURCE_TIER_LABELS:
        raise ValueError(
            "registered rev-2 research requires the labelled development-grade "
            "action and reconstructed schedule tiers"
        )
    return dict(DEVELOPMENT_SOURCE_TIER_LABELS)


def _assert_source_tier_labels(
    payload: Mapping[str, object], *, expected: Mapping[str, str], path: Path
) -> None:
    metadata = payload.get("metadata")
    nested = metadata if isinstance(metadata, Mapping) else {}
    for key, value in expected.items():
        actual = payload.get(key, nested.get(key))
        if actual != value:
            raise ValueError(f"{path} has a different {key}: {actual!r}")


def _verify_development_acceptance(
    path: Path,
    *,
    expected_sha256: str,
    store_manifest_sha256: str,
    implementation: Mapping[str, object],
    source_tiers: Mapping[str, str],
) -> dict[str, object]:
    source = Path(path).resolve(strict=True)
    if sha256_file(source) != expected_sha256.casefold():
        raise ValueError("development acceptance report SHA-256 mismatch")
    report = _read_json(source)
    if (
        report.get("schema") != "BRAZIL_RV_V2_PIPELINE_VALIDATION_V7"
        or report.get("status") != "completed"
        or report.get("engineering_acceptance_status")
        != "development_grade_inferred_actions"
        or report.get("research_claim") is not False
    ):
        raise ValueError("development acceptance report is not an accepted rev-2 gate")
    _assert_false_access(report, path=source)
    if report.get("code") != implementation:
        raise ValueError("development acceptance implementation differs from rev-2")
    sources = report.get("sources")
    store = sources.get("store") if isinstance(sources, Mapping) else None
    if (
        not isinstance(store, Mapping)
        or store.get("manifest_sha256") != store_manifest_sha256
    ):
        raise ValueError("development acceptance report binds a different store")
    results = report.get("results")
    acceptance = (
        results.get("development_acceptance") if isinstance(results, Mapping) else None
    )
    labels = acceptance.get("labels") if isinstance(acceptance, Mapping) else None
    if not isinstance(labels, Mapping) or any(
        labels.get(key) != value for key, value in source_tiers.items()
    ):
        raise ValueError("development acceptance source tiers differ from the store")
    if acceptance.get("reasons") != []:
        raise ValueError("development acceptance retains failed sanity bounds")
    return report


def _assert_false_access(payload: Mapping[str, object], *, path: Path) -> None:
    if (
        payload.get("official_validation_accessed") is not False
        or payload.get("test_accessed") is not False
    ):
        raise PermissionError(f"sealed-window access recorded in {path}")
    if not isinstance(payload.get("transfer_chronology_clean"), bool):
        raise ValueError(f"transfer chronology status is absent from {path}")


def _assert_current_clean_training(
    payload: Mapping[str, object], *, path: Path
) -> None:
    _assert_false_access(payload, path=path)
    if (
        payload.get("schema") != TRAINING_STAGE_SCHEMA
        or payload.get("status") != "completed"
    ):
        raise ValueError(f"trajectory uses a stale or incomplete run schema: {path}")
    if payload.get("transfer_chronology_clean") is not True:
        raise PermissionError(
            f"trajectory has contaminated transfer chronology: {path}"
        )
    feature_schema = payload.get("feature_schema_sha256")
    if not isinstance(feature_schema, str) or len(feature_schema) != 64:
        raise ValueError(f"trajectory lacks a canonical feature schema: {path}")
    _assert_source_tier_labels(
        payload,
        expected={
            "action_terms_source": "inferred_cotahist_dismes_v1",
            "schedule_source": "reconstructed_v1",
        },
        path=path,
    )


def _array_record(path: Path, values: NDArray[np.generic]) -> dict[str, object]:
    return {
        "path": path.name,
        "shape": list(values.shape),
        "dtype": values.dtype.str,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _persist_scores(
    root: Path,
    arrays: Mapping[str, NDArray[np.generic]],
    metadata: Mapping[str, object],
) -> tuple[Path, str]:
    action_terms_source = metadata.get("action_terms_source")
    schedule_source = metadata.get("schedule_source")
    if not isinstance(action_terms_source, str) or not isinstance(schedule_source, str):
        raise ValueError("research score metadata lacks source-tier labels")
    root.mkdir(parents=True, exist_ok=False)
    records: dict[str, dict[str, object]] = {}
    for label, raw in sorted(arrays.items()):
        values = np.asarray(raw)
        path = root / f"{label}.npy"
        np.save(path, values, allow_pickle=False)
        records[path.name] = _array_record(path, values)
    manifest = root / "score_manifest.json"
    digest = write_json_atomic(
        manifest,
        {
            "schema": RESEARCH_SCORE_SCHEMA,
            "status": "completed",
            **RESEARCH_FLAGS,
            "action_terms_source": action_terms_source,
            "schedule_source": schedule_source,
            "metadata": dict(metadata),
            "artifacts": records,
        },
    )
    return manifest, digest


def _fold_indices(
    dates: NDArray[np.datetime64],
) -> tuple[
    dict[str, NDArray[np.int64]],
    dict[str, NDArray[np.int64]],
    dict[str, NDArray[np.int64]],
    dict[str, NDArray[np.int64]],
    dict[str, object],
]:
    python_dates = tuple(dates.astype("datetime64[D]").astype(object).tolist())
    folds = development_folds(python_dates)
    fit: dict[str, NDArray[np.int64]] = {}
    selection: dict[str, NDArray[np.int64]] = {}
    evaluation: dict[str, NDArray[np.int64]] = {}
    fit_target_window: dict[str, NDArray[np.int64]] = {}
    payload: dict[str, object] = {}
    for fold in folds:
        fit[fold.name] = _date_indices(dates, fold.fit_dates)
        selection[fold.name] = _date_indices(dates, fold.selection_dates)
        evaluation[fold.name] = _date_indices(dates, fold.evaluation_dates)
        fit_target_window[fold.name] = _date_indices(
            dates, (*fold.fit_dates, *fold.purge_before_dates)
        )
        payload[fold.name] = fold.payload()
    if tuple(fit) != ("F1", "F2", "F3"):
        raise ValueError("development fold roster differs from the registration")
    return fit, selection, evaluation, fit_target_window, payload


def _pretrain_indices(dates: NDArray[np.datetime64]) -> NDArray[np.int64]:
    full = np.flatnonzero(
        (dates >= np.datetime64(STORE_START)) & (dates <= np.datetime64(PRETRAIN_END))
    ).astype(np.int64)
    if not full.size or np.any(np.diff(full) != 1):
        raise ValueError("pretrain axis is incomplete")
    # Each stored row is already the canonical decision snapshot. The store
    # builder consumes warm-up observations when constructing its first row;
    # consumers must not discard or shift that row a second time.
    return full


def _open_round_store(
    store_root: Path,
    fit: Mapping[str, NDArray[np.int64]],
    selection: Mapping[str, NDArray[np.int64]],
    evaluation: Mapping[str, NDArray[np.int64]],
    fit_target_window: Mapping[str, NDArray[np.int64]],
    pretrain: NDArray[np.int64],
) -> tuple[V2Store, dict[str, object]]:
    requested = np.unique(
        np.concatenate(
            (
                pretrain,
                *(fit[name] for name in ("F1", "F2", "F3")),
                *(selection[name] for name in ("F1", "F2", "F3")),
                *(evaluation[name] for name in ("F1", "F2", "F3")),
            )
        )
    ).astype(np.int64)
    store, ledger = open_store_for_samples(
        store_root,
        requested,
        purpose="evaluation",
        history_lookbacks=253,
        history_end_offsets=np.where(requested == 0, 0, -1),
        target_window_indices=np.unique(
            np.concatenate(
                (
                    pretrain,
                    *(fit_target_window[name] for name in ("F1", "F2", "F3")),
                    *(selection[name] for name in ("F1", "F2", "F3")),
                    *(evaluation[name] for name in ("F1", "F2", "F3")),
                )
            )
        ),
    )
    return store, ledger.payload()


def _evaluate(
    *,
    store: V2Store,
    indices: NDArray[np.int64],
    scores: NDArray[np.floating],
    score_mask: NDArray[np.bool_],
    cdi: NDArray[np.float64],
    source_hashes: Mapping[str, str],
    fold: str,
    output: Path,
) -> _ResearchEvaluation:
    inputs = _evaluation_inputs(
        store,
        indices,
        scores,
        score_mask,
        cdi,
        source_hashes,
        transfer_chronology_clean=True,
    )
    result = evaluate_scores(inputs, window_name=fold)
    result.report.update(RESEARCH_FLAGS)
    write_json_atomic(output, result.report)
    return _ResearchEvaluation(result=result, inputs=inputs)


def _single_family_ic_series(
    inputs: EvaluationInputs,
    *,
    targets: str,
    target_mask: str,
) -> NDArray[np.float64]:
    indexes = np.asarray(
        [HORIZONS.index(horizon) for horizon in PRIMARY_HORIZONS], dtype=np.int64
    )
    scores = np.asarray(inputs.scores, dtype=np.float64)[..., indexes]
    outcomes = np.asarray(getattr(inputs, targets), dtype=np.float64)[..., indexes]
    population = (
        np.asarray(inputs.active, dtype=np.bool_)
        & np.asarray(getattr(inputs, target_mask), dtype=np.bool_)[..., indexes].all(
            axis=-1
        )
        & np.isfinite(outcomes).all(axis=-1)
        & np.asarray(inputs.score_mask, dtype=np.bool_)[..., indexes].all(axis=-1)
        & np.isfinite(scores).all(axis=-1)
    )
    result = np.full(len(inputs.dates), np.nan, dtype=np.float64)
    for day in range(len(inputs.dates)):
        if int(population[day].sum()) < 20:
            continue
        values = np.asarray(
            [
                _spearman(scores[day, :, head], outcomes[day, :, head], population[day])
                for head in range(len(PRIMARY_HORIZONS))
            ]
        )
        if np.isfinite(values).all():
            result[day] = float(values.mean())
    return result


def _single_spread_series(inputs: EvaluationInputs) -> NDArray[np.float64]:
    indexes = np.asarray(
        [HORIZONS.index(horizon) for horizon in PRIMARY_HORIZONS], dtype=np.int64
    )
    scores = np.asarray(inputs.scores, dtype=np.float64)[..., indexes]
    returns = np.asarray(inputs.shareholder_simple_returns, dtype=np.float64)[
        ..., indexes
    ]
    population = (
        np.asarray(inputs.active, dtype=np.bool_)
        & np.asarray(inputs.shareholder_target_mask, dtype=np.bool_)[..., indexes].all(
            axis=-1
        )
        & np.isfinite(returns).all(axis=-1)
        & np.asarray(inputs.score_mask, dtype=np.bool_)[..., indexes].all(axis=-1)
        & np.isfinite(scores).all(axis=-1)
    )
    result = np.full(len(inputs.dates), np.nan, dtype=np.float64)
    for day in range(len(inputs.dates)):
        names = np.flatnonzero(population[day])
        if names.size < 20:
            continue
        decile_count = max(1, names.size // 10)
        values = []
        for head, horizon in enumerate(PRIMARY_HORIZONS):
            order = names[np.argsort(scores[day, names, head], kind="stable")]
            spread = (
                returns[day, order[-decile_count:], head].mean()
                - returns[day, order[:decile_count], head].mean()
            )
            values.append(float(spread) * 10_000.0 / horizon)
        result[day] = float(np.mean(values))
    return result


def _single_persistence_series(
    inputs: EvaluationInputs, *, lag: int
) -> NDArray[np.float64]:
    indexes = np.asarray(
        [HORIZONS.index(horizon) for horizon in PRIMARY_HORIZONS], dtype=np.int64
    )
    scores = np.asarray(inputs.scores, dtype=np.float64)[..., indexes]
    score_mask = np.asarray(inputs.score_mask, dtype=np.bool_)[..., indexes]
    active = np.asarray(inputs.active, dtype=np.bool_)
    result = np.full(len(inputs.dates), np.nan, dtype=np.float64)
    for day in range(lag, len(inputs.dates)):
        population = (
            active[day]
            & active[day - lag]
            & score_mask[day].all(axis=-1)
            & score_mask[day - lag].all(axis=-1)
            & np.isfinite(scores[day]).all(axis=-1)
            & np.isfinite(scores[day - lag]).all(axis=-1)
        )
        if int(population.sum()) < 20:
            continue
        values = np.asarray(
            [
                _spearman(scores[day, :, head], scores[day - lag, :, head], population)
                for head in range(len(PRIMARY_HORIZONS))
            ]
        )
        if np.isfinite(values).all():
            result[day] = float(values.mean())
    return result


def _daily_series(
    evaluation: _ResearchEvaluation,
) -> dict[str, NDArray[np.float64]]:
    report = evaluation.result.report
    daily_primary = report.get("daily_primary_ic")
    economics = report.get("economics")
    if (
        not isinstance(daily_primary, list)
        or not isinstance(economics, Mapping)
        or not isinstance(economics.get("daily_table"), list)
    ):
        raise ValueError("evaluation report lacks registered daily readouts")

    ordered_dates = tuple(value.isoformat() for value in evaluation.result.dates)
    if (
        tuple(str(row.get("date")) for row in daily_primary if isinstance(row, Mapping))
        != ordered_dates
    ):
        raise ValueError("evaluation primary table differs from its retained date axis")

    primary_values = np.asarray(evaluation.result.daily_primary_ic, dtype=np.float64)
    if primary_values.shape != (len(ordered_dates),):
        raise ValueError("retained primary IC series differs from its date axis")
    headline = [
        row
        for row in economics["daily_table"]
        if isinstance(row, Mapping)
        and float(row.get("cost_bps_per_side", -1.0)) == 4.0
        and float(row.get("annual_borrow_rate", -1.0)) == 0.02
    ]
    headline_by_date: dict[str, float] = {}
    for row in headline:
        day = str(row.get("date"))
        if day in headline_by_date:
            raise ValueError("headline economics contains duplicate dates")
        value = row.get("net_excess_all_cash_bps")
        headline_by_date[day] = np.nan if value is None else float(value)
    headline_summary = economics.get("headline")
    if not isinstance(headline_summary, Mapping) or not isinstance(
        headline_summary.get("economics_unresolved"), bool
    ):
        raise ValueError("evaluation report lacks economics resolution status")
    economics_values = np.asarray(
        [headline_by_date.get(day, np.nan) for day in ordered_dates],
        dtype=np.float64,
    )
    if headline_summary["economics_unresolved"]:
        economics_values.fill(np.nan)
    return {
        "primary_scaled_target_ic": primary_values,
        "shareholder_rank_ic": _single_family_ic_series(
            evaluation.inputs,
            targets="shareholder_midrank_targets",
            target_mask="shareholder_target_mask",
        ),
        "price_return_rank_ic": _single_family_ic_series(
            evaluation.inputs,
            targets="price_midrank_targets",
            target_mask="price_target_mask",
        ),
        "persistence_1": _single_persistence_series(evaluation.inputs, lag=1),
        "persistence_5": _single_persistence_series(evaluation.inputs, lag=5),
        "shareholder_return_spread_bps_per_holding_session": _single_spread_series(
            evaluation.inputs
        ),
        "headline_net_excess_bps": economics_values,
    }


def _folded_bootstrap(
    values: Sequence[NDArray[np.floating]],
    *,
    replications: int = BOOTSTRAP_REPLICATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in values)
    if not arrays or any(
        value.ndim != 1 or len(value) < BOOTSTRAP_BLOCK for value in arrays
    ):
        raise ValueError("each fold needs at least one bootstrap block")
    finite = np.concatenate(arrays)
    possible_observations = int(sum(len(value) for value in arrays))
    finite_observations = int(np.isfinite(finite).sum())
    if finite_observations == 0:
        return {
            "estimate": None,
            "lower_95": None,
            "upper_95": None,
            "possible_observations": possible_observations,
            "finite_observations": 0,
            "undefined_reason": "no_defined_daily_values",
            "replications": replications,
            "block_length_sessions": BOOTSTRAP_BLOCK,
            "fold_boundary_preserved": True,
        }
    estimate = float(np.nanmean(finite))
    generator = np.random.default_rng(seed)
    sums = np.zeros(replications, dtype=np.float64)
    counts = np.zeros(replications, dtype=np.int64)
    for array in arrays:
        blocks = math.ceil(len(array) / BOOTSTRAP_BLOCK)
        starts = generator.integers(
            0,
            len(array) - BOOTSTRAP_BLOCK + 1,
            size=(replications, blocks),
        )
        sampled = (
            starts[..., None] + np.arange(BOOTSTRAP_BLOCK, dtype=np.int64)
        ).reshape(replications, -1)[:, : len(array)]
        selected = array[sampled]
        sums += np.nansum(selected, axis=1)
        counts += np.isfinite(selected).sum(axis=1)
    draws = np.divide(sums, counts, out=np.full(replications, np.nan), where=counts > 0)
    return {
        "estimate": estimate,
        "lower_95": float(np.nanquantile(draws, 0.025)),
        "upper_95": float(np.nanquantile(draws, 0.975)),
        "possible_observations": possible_observations,
        "finite_observations": finite_observations,
        "undefined_reason": None,
        "replications": replications,
        "block_length_sessions": BOOTSTRAP_BLOCK,
        "fold_boundary_preserved": True,
    }


def _readout_point(readout: Mapping[str, object]) -> float | None:
    value = readout.get("estimate")
    return None if value is None else float(value)


def _ranking_point(readout: Mapping[str, object]) -> float:
    value = _readout_point(readout)
    return -math.inf if value is None else value


def _point_is_negative(readout: Mapping[str, object]) -> bool:
    value = _readout_point(readout)
    return value is not None and value < 0.0


def _economics_not_worse(
    candidate: Mapping[str, object], baseline: Mapping[str, object]
) -> bool:
    candidate_value = _readout_point(candidate)
    baseline_value = _readout_point(baseline)
    return (
        candidate_value is not None
        and baseline_value is not None
        and candidate_value >= baseline_value
    )


def _small_interval_spanning_zero(readout: Mapping[str, object]) -> bool:
    estimate = _readout_point(readout)
    lower = readout.get("lower_95")
    upper = readout.get("upper_95")
    return (
        estimate is not None
        and lower is not None
        and upper is not None
        and estimate <= 0.002
        and float(lower) <= 0.0 <= float(upper)
    )


def _pooled_readouts(
    evaluations: Mapping[str, _ResearchEvaluation],
) -> dict[str, object]:
    if tuple(evaluations) != ("F1", "F2", "F3"):
        raise ValueError("pooled report roster must be F1/F2/F3")
    series = {
        fold: _daily_series(evaluation) for fold, evaluation in evaluations.items()
    }
    labels = tuple(series["F1"])
    return {
        "folds": {
            fold: {label: _folded_bootstrap((values[label],)) for label in labels}
            for fold, values in series.items()
        },
        "pooled": {
            label: _folded_bootstrap(
                tuple(series[fold][label] for fold in ("F1", "F2", "F3"))
            )
            for label in labels
        },
        "horizons": {
            fold: evaluation.result.report["horizon_readouts"]
            for fold, evaluation in evaluations.items()
        },
        "economics_grid": {
            fold: evaluation.result.report["economics"]["summaries"]
            for fold, evaluation in evaluations.items()
        },
        "coverage": {
            fold: {
                "primary": evaluation.result.report["primary_support"],
                "economics": evaluation.result.report["economics"]["coverage"],
            }
            for fold, evaluation in evaluations.items()
        },
    }


def _common_family_ic_delta(
    candidate: EvaluationInputs,
    baseline: EvaluationInputs,
    *,
    targets: str,
    target_mask: str,
) -> tuple[NDArray[np.float64], list[dict[str, object]]]:
    indexes = np.asarray(
        [HORIZONS.index(horizon) for horizon in PRIMARY_HORIZONS], dtype=np.int64
    )
    candidate_scores = np.asarray(candidate.scores, dtype=np.float64)[..., indexes]
    baseline_scores = np.asarray(baseline.scores, dtype=np.float64)[..., indexes]
    outcomes = np.asarray(getattr(candidate, targets), dtype=np.float64)[..., indexes]
    outcome_mask = np.asarray(getattr(candidate, target_mask), dtype=np.bool_)[
        ..., indexes
    ]
    population = (
        np.asarray(candidate.active, dtype=np.bool_)
        & outcome_mask.all(axis=-1)
        & np.isfinite(outcomes).all(axis=-1)
        & np.asarray(candidate.score_mask, dtype=np.bool_)[..., indexes].all(axis=-1)
        & np.asarray(baseline.score_mask, dtype=np.bool_)[..., indexes].all(axis=-1)
        & np.isfinite(candidate_scores).all(axis=-1)
        & np.isfinite(baseline_scores).all(axis=-1)
    )
    result = np.full(len(candidate.dates), np.nan, dtype=np.float64)
    for day in range(len(candidate.dates)):
        if int(population[day].sum()) < 20:
            continue
        left = np.asarray(
            [
                _spearman(
                    candidate_scores[day, :, head],
                    outcomes[day, :, head],
                    population[day],
                )
                for head in range(len(PRIMARY_HORIZONS))
            ]
        )
        right = np.asarray(
            [
                _spearman(
                    baseline_scores[day, :, head],
                    outcomes[day, :, head],
                    population[day],
                )
                for head in range(len(PRIMARY_HORIZONS))
            ]
        )
        if np.isfinite(left).all() and np.isfinite(right).all():
            result[day] = float(left.mean() - right.mean())
    return result, _paired_population_rows(candidate.dates, population, result)


def _common_spread_delta(
    candidate: EvaluationInputs,
    baseline: EvaluationInputs,
) -> tuple[NDArray[np.float64], list[dict[str, object]]]:
    indexes = np.asarray(
        [HORIZONS.index(horizon) for horizon in PRIMARY_HORIZONS], dtype=np.int64
    )
    candidate_scores = np.asarray(candidate.scores, dtype=np.float64)[..., indexes]
    baseline_scores = np.asarray(baseline.scores, dtype=np.float64)[..., indexes]
    returns = np.asarray(candidate.shareholder_simple_returns, dtype=np.float64)[
        ..., indexes
    ]
    outcome_mask = np.asarray(candidate.shareholder_target_mask, dtype=np.bool_)[
        ..., indexes
    ]
    population = (
        np.asarray(candidate.active, dtype=np.bool_)
        & outcome_mask.all(axis=-1)
        & np.isfinite(returns).all(axis=-1)
        & np.asarray(candidate.score_mask, dtype=np.bool_)[..., indexes].all(axis=-1)
        & np.asarray(baseline.score_mask, dtype=np.bool_)[..., indexes].all(axis=-1)
        & np.isfinite(candidate_scores).all(axis=-1)
        & np.isfinite(baseline_scores).all(axis=-1)
    )
    result = np.full(len(candidate.dates), np.nan, dtype=np.float64)
    for day in range(len(candidate.dates)):
        names = np.flatnonzero(population[day])
        if names.size < 20:
            continue
        decile_count = max(1, names.size // 10)
        deltas = []
        for head, horizon in enumerate(PRIMARY_HORIZONS):
            spreads = []
            for scores in (candidate_scores, baseline_scores):
                order = names[np.argsort(scores[day, names, head], kind="stable")]
                spread = (
                    returns[day, order[-decile_count:], head].mean()
                    - returns[day, order[:decile_count], head].mean()
                )
                spreads.append(float(spread) * 10_000.0 / horizon)
            deltas.append(spreads[0] - spreads[1])
        result[day] = float(np.mean(deltas))
    return result, _paired_population_rows(candidate.dates, population, result)


def _common_persistence_delta(
    candidate: EvaluationInputs,
    baseline: EvaluationInputs,
    *,
    lag: int,
) -> tuple[NDArray[np.float64], list[dict[str, object]]]:
    indexes = np.asarray(
        [HORIZONS.index(horizon) for horizon in PRIMARY_HORIZONS], dtype=np.int64
    )
    candidate_scores = np.asarray(candidate.scores, dtype=np.float64)[..., indexes]
    baseline_scores = np.asarray(baseline.scores, dtype=np.float64)[..., indexes]
    candidate_mask = np.asarray(candidate.score_mask, dtype=np.bool_)[..., indexes]
    baseline_mask = np.asarray(baseline.score_mask, dtype=np.bool_)[..., indexes]
    active = np.asarray(candidate.active, dtype=np.bool_)
    result = np.full(len(candidate.dates), np.nan, dtype=np.float64)
    populations = np.zeros_like(active)
    for day in range(lag, len(candidate.dates)):
        population = (
            active[day]
            & active[day - lag]
            & candidate_mask[day].all(axis=-1)
            & candidate_mask[day - lag].all(axis=-1)
            & baseline_mask[day].all(axis=-1)
            & baseline_mask[day - lag].all(axis=-1)
            & np.isfinite(candidate_scores[day]).all(axis=-1)
            & np.isfinite(candidate_scores[day - lag]).all(axis=-1)
            & np.isfinite(baseline_scores[day]).all(axis=-1)
            & np.isfinite(baseline_scores[day - lag]).all(axis=-1)
        )
        populations[day] = population
        if int(population.sum()) < 20:
            continue
        left = np.asarray(
            [
                _spearman(
                    candidate_scores[day, :, head],
                    candidate_scores[day - lag, :, head],
                    population,
                )
                for head in range(len(PRIMARY_HORIZONS))
            ]
        )
        right = np.asarray(
            [
                _spearman(
                    baseline_scores[day, :, head],
                    baseline_scores[day - lag, :, head],
                    population,
                )
                for head in range(len(PRIMARY_HORIZONS))
            ]
        )
        if np.isfinite(left).all() and np.isfinite(right).all():
            result[day] = float(left.mean() - right.mean())
    return result, _paired_population_rows(
        candidate.dates, populations, result, unavailable_before=lag
    )


def _paired_population_rows(
    dates: Sequence[object],
    population: NDArray[np.bool_],
    values: NDArray[np.float64],
    *,
    unavailable_before: int = 0,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for day, day_value in enumerate(dates):
        count = int(population[day].sum())
        value = values[day]
        reason = None
        if day < unavailable_before:
            reason = "lag_precedes_window"
        elif count < 20:
            reason = "fewer_than_20_common_names"
        elif not np.isfinite(value):
            reason = "undefined_on_common_population"
        rows.append(
            {
                "date": day_value.isoformat(),
                "common_candidate_baseline_name_count": count,
                "delta": None if not np.isfinite(value) else float(value),
                "undefined_reason": reason,
            }
        )
    return rows


def _paired_readouts(
    candidate: Mapping[str, _ResearchEvaluation],
    baseline: Mapping[str, _ResearchEvaluation],
) -> dict[str, object]:
    if tuple(candidate) != ("F1", "F2", "F3") or tuple(baseline) != (
        "F1",
        "F2",
        "F3",
    ):
        raise ValueError("paired report roster must be F1/F2/F3")
    deltas: dict[str, dict[str, NDArray[np.float64]]] = {}
    population_audit: dict[str, dict[str, list[dict[str, object]]]] = {}
    for fold in ("F1", "F2", "F3"):
        left = candidate[fold]
        right = baseline[fold]
        if left.result.dates != right.result.dates:
            raise ValueError(f"paired date axes differ for {fold}")
        _validate_paired_identity(left.result.report, right.result.report)
        primary_delta, primary_rows = _paired_primary_daily(left.result, right.result)
        shareholder_delta, shareholder_rows = _common_family_ic_delta(
            left.inputs,
            right.inputs,
            targets="shareholder_midrank_targets",
            target_mask="shareholder_target_mask",
        )
        price_delta, price_rows = _common_family_ic_delta(
            left.inputs,
            right.inputs,
            targets="price_midrank_targets",
            target_mask="price_target_mask",
        )
        persistence_1, persistence_1_rows = _common_persistence_delta(
            left.inputs, right.inputs, lag=1
        )
        persistence_5, persistence_5_rows = _common_persistence_delta(
            left.inputs, right.inputs, lag=5
        )
        spread_delta, spread_rows = _common_spread_delta(left.inputs, right.inputs)
        candidate_series = _daily_series(left)
        baseline_series = _daily_series(right)
        if (
            left.result.report["economics"]["headline"]["economics_unresolved"]
            or right.result.report["economics"]["headline"]["economics_unresolved"]
        ):
            economics_delta = np.full(len(left.result.dates), np.nan)
        else:
            economics_delta = (
                candidate_series["headline_net_excess_bps"]
                - baseline_series["headline_net_excess_bps"]
            )
        economics_rows = [
            {
                "date": day.isoformat(),
                "delta": None if not np.isfinite(value) else float(value),
                "undefined_reason": (
                    "economics_unresolved"
                    if (
                        left.result.report["economics"]["headline"][
                            "economics_unresolved"
                        ]
                        or right.result.report["economics"]["headline"][
                            "economics_unresolved"
                        ]
                    )
                    else (None if np.isfinite(value) else "missing_daily_economics")
                ),
            }
            for day, value in zip(left.result.dates, economics_delta, strict=True)
        ]
        deltas[fold] = {
            "primary_scaled_target_ic": primary_delta,
            "shareholder_rank_ic": shareholder_delta,
            "price_return_rank_ic": price_delta,
            "persistence_1": persistence_1,
            "persistence_5": persistence_5,
            "shareholder_return_spread_bps_per_holding_session": spread_delta,
            "headline_net_excess_bps": economics_delta,
        }
        population_audit[fold] = {
            "primary_scaled_target_ic": primary_rows,
            "shareholder_rank_ic": shareholder_rows,
            "price_return_rank_ic": price_rows,
            "persistence_1": persistence_1_rows,
            "persistence_5": persistence_5_rows,
            "shareholder_return_spread_bps_per_holding_session": spread_rows,
            "headline_net_excess_bps": economics_rows,
        }
    labels = tuple(deltas["F1"])
    return {
        "schema": "BRAZIL_RV_V2_POOLED_PAIRED_READOUTS_V2",
        "folds": {
            fold: {label: _folded_bootstrap((deltas[fold][label],)) for label in labels}
            for fold in deltas
        },
        "pooled": {
            label: _folded_bootstrap(
                tuple(deltas[fold][label] for fold in ("F1", "F2", "F3"))
            )
            for label in labels
        },
        "population_audit": population_audit,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def _feature_names(store: V2Store, rung: str) -> tuple[str, ...]:
    slow_names = scalar_feature_names(store, ("slow",))
    output = list(gbdt_scalar_feature_names(slow_names))
    if rung != "a_slow":
        intraday_names = scalar_feature_names(store, ("intraday",))
        output.extend(gbdt_scalar_feature_names(intraday_names))
        output.append("fast_present")
    for group in RUNG_GROUPS[rung]:
        group_names = scalar_feature_names(store, (f"sidecar_{group}",))
        output.extend(gbdt_scalar_feature_names(group_names))
    if len(output) != len(set(output)):
        raise ValueError("GBDT feature names are not unique")
    return tuple(str(value) for value in output)


def _gbdt_features(
    store: V2Store,
    indices: NDArray[np.int64],
    rung: str,
    *,
    pretrain_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float32]:
    pretrain = (
        np.zeros(len(indices), dtype=np.bool_)
        if pretrain_mask is None
        else np.asarray(pretrain_mask, dtype=np.bool_)
    )
    if pretrain.shape != (len(indices),):
        raise ValueError("pretrain row mask is misaligned")
    if np.any(indices < 0):
        raise ValueError("GBDT rows are outside the canonical store")
    slow = read_scalar_feature_view(store, indices, ("slow",))
    parts = [assemble_gbdt_scalar_view(slow, label="slow")]
    if rung != "a_slow":
        intraday = read_scalar_feature_view(store, indices, ("intraday",))
        intraday_valid = intraday.valid.copy()
        intraday_age = intraday.age_sessions.copy()
        intraday_valid[pretrain] = False
        intraday_age[pretrain] = -1.0
        masked_intraday = ScalarFeatureView(
            date_indices=intraday.date_indices,
            dates=intraday.dates,
            isins=intraday.isins,
            active=intraday.active,
            names=intraday.names,
            values=intraday.values,
            valid=intraday_valid,
            age_sessions=intraday_age,
        )
        present = np.asarray(store.read("fast_present", indices), dtype=np.float32)
        present[pretrain] = 0.0
        parts.extend(
            (
                assemble_gbdt_scalar_view(masked_intraday, label="intraday"),
                present[..., None],
            )
        )
    for group in RUNG_GROUPS[rung]:
        sidecar = read_scalar_feature_view(store, indices, (f"sidecar_{group}",))
        parts.append(assemble_gbdt_scalar_view(sidecar, label=group))
    result = np.concatenate(parts, axis=-1, dtype=np.float32)
    if result.shape[-1] != len(_feature_names(store, rung)) or np.isinf(result).any():
        raise ValueError("GBDT feature panel violates its frozen contract")
    return result


def _persist_model(
    model: MultiHorizonGBDT,
    root: Path,
    verification_features: NDArray[np.floating],
    verification_mask: NDArray[np.bool_],
    *,
    source_tiers: Mapping[str, str],
) -> dict[str, object]:
    manifest, digest = model.save(
        root,
        metadata={"status": "completed", **RESEARCH_FLAGS, **source_tiers},
    )
    restored = MultiHorizonGBDT.load(root, expected_manifest_sha256=digest)
    if not np.array_equal(
        model.predict_ranks(verification_features, verification_mask),
        restored.predict_ranks(verification_features, verification_mask),
    ):
        raise ValueError("GBDT model round-trip changed registered ranks")
    return {"manifest": str(manifest), "manifest_sha256": digest}


def _run_gbdt_candidate(
    *,
    store: V2Store,
    rung: str,
    fit: Mapping[str, NDArray[np.int64]],
    fit_target_window: Mapping[str, NDArray[np.int64]],
    selection: Mapping[str, NDArray[np.int64]],
    evaluation: Mapping[str, NDArray[np.int64]],
    pretrain: NDArray[np.int64] | None,
    decay_half_life: float | None,
    cdi: NDArray[np.float64],
    source_hashes: Mapping[str, str],
    root: Path,
    num_threads: int,
) -> tuple[dict[str, _ResearchEvaluation], dict[str, object]]:
    config = GBDTConfig(seeds=GBDT_SEEDS, num_threads=num_threads)
    feature_names = _feature_names(store, rung)
    reports: dict[str, _ResearchEvaluation] = {}
    records: dict[str, object] = {}
    for fold in ("F1", "F2", "F3"):
        fine = fit[fold]
        train_indices = fine if pretrain is None else np.concatenate((pretrain, fine))
        train_pretrain = np.zeros(len(train_indices), dtype=np.bool_)
        if pretrain is not None:
            train_pretrain[: len(pretrain)] = True
        selection_indices = selection[fold]
        evaluation_indices = evaluation[fold]
        train_x = _gbdt_features(
            store, train_indices, rung, pretrain_mask=train_pretrain
        )
        selection_x = _gbdt_features(store, selection_indices, rung)
        evaluation_x = _gbdt_features(store, evaluation_indices, rung)
        train_window = (
            fit_target_window[fold]
            if pretrain is None
            else np.concatenate((pretrain, fit_target_window[fold]))
        )
        train_mask = _window_target_mask(
            store.read("target_valid", train_indices),
            train_indices,
            target_window_indices=train_window,
        )
        train_y = np.asarray(
            store.read_target("target_primary", train_indices, valid_mask=train_mask)
        )
        selection_mask = _window_target_mask(
            store.read("target_valid", selection_indices), selection_indices
        )
        selection_y = np.asarray(
            store.read_target(
                "target_primary", selection_indices, valid_mask=selection_mask
            )
        )
        active = np.asarray(store.read("active", evaluation_indices), dtype=np.bool_)
        score_mask = np.repeat(active[..., None], len(HORIZONS), axis=-1)
        sample_weights = None
        if decay_half_life is not None:
            ages = train_indices[-1] - train_indices
            sample_weights = np.power(0.5, ages / decay_half_life)
        model = MultiHorizonGBDT(config, feature_names=feature_names)
        model.fit(
            train_x,
            train_y,
            train_mask,
            selection_x,
            selection_y,
            selection_mask,
            train_dates=train_indices,
            validation_dates=selection_indices,
            sample_weights=sample_weights,
        )
        predictions = model.predict_ranks(evaluation_x, score_mask)
        raw_importance = model.feature_importance(evaluation_x)
        importance = {name: values.tolist() for name, values in raw_importance.items()}
        model_record = _persist_model(
            model,
            root / "models" / fold,
            evaluation_x,
            score_mask,
            source_tiers=_source_tier_labels(store.manifest),
        )
        fold_root = root / fold
        manifest, manifest_sha = _persist_scores(
            fold_root,
            {
                "scores": predictions,
                "score_mask": score_mask,
            },
            {
                **_source_tier_labels(store.manifest),
                "engine": "lightgbm",
                "rung": rung,
                "fold": fold,
                "feature_names": list(feature_names),
                "config": asdict(config),
                "pretrain_included": pretrain is not None,
                "time_decay_half_life_sessions": decay_half_life,
                "fit_date_indices": train_indices.tolist(),
                "selection_date_indices": selection_indices.tolist(),
                "evaluation_date_indices": evaluation_indices.tolist(),
                "model": model_record,
                "importance": importance,
            },
        )
        evaluated = _evaluate(
            store=store,
            indices=evaluation_indices,
            scores=predictions,
            score_mask=score_mask,
            cdi=cdi,
            source_hashes=source_hashes,
            fold=fold,
            output=fold_root / "evaluation.json",
        )
        reports[fold] = evaluated
        records[fold] = {
            "score_manifest": str(manifest),
            "score_manifest_sha256": manifest_sha,
            "evaluation": str(fold_root / "evaluation.json"),
            "evaluation_sha256": sha256_file(fold_root / "evaluation.json"),
        }
        del model, train_x, selection_x, evaluation_x, train_y, selection_y
    return reports, records


def freeze_round1(
    *,
    store_root: Path,
    cdi_path: Path,
    cdi_sha256: str,
    experiment52_cdi_path: Path,
    experiment52_cdi_sha256: str,
    acceptance_path: Path,
    acceptance_sha256: str,
    output_root: Path,
    num_threads: int,
) -> str:
    protocol = verify_registration_protocol()
    code = _git_identity()
    output = output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    store = store_root.resolve(strict=True)
    store_manifest, dates = _read_store_header(store)
    store_manifest_sha256 = sha256_file(store / "manifest.json")
    store_metadata = store_manifest.get("metadata")
    if (
        not isinstance(store_metadata, Mapping)
        or store_metadata.get("implementation_git_commit") != code["commit"]
    ):
        raise ValueError("Round-1 store was not built by the frozen implementation")
    source_tiers = _source_tier_labels(store_manifest)
    _verify_development_acceptance(
        acceptance_path,
        expected_sha256=acceptance_sha256,
        store_manifest_sha256=store_manifest_sha256,
        implementation=code,
        source_tiers=source_tiers,
    )
    fit, selection, evaluation, _, folds = _fold_indices(dates)
    _load_development_cdi(
        dates=dates,
        cdi_path=cdi_path,
        expected_sha256=cdi_sha256,
        experiment52_cdi_path=experiment52_cdi_path,
        experiment52_expected_sha256=experiment52_cdi_sha256,
    )
    if num_threads < 0:
        raise ValueError("num_threads must be non-negative")
    output.mkdir(parents=True, exist_ok=False)
    design = {
        "schema": ROUND1_SCHEMA,
        "status": "frozen_before_score",
        **RESEARCH_FLAGS,
        "frozen_at_utc": _utc_now(),
        "implementation": code,
        **source_tiers,
        "preregistration": {
            "path": str(PREREGISTRATION.resolve(strict=True)),
            "sha256": sha256_file(PREREGISTRATION),
            "protocol": protocol,
        },
        "store": {
            "root": str(store),
            "manifest_sha256": store_manifest_sha256,
            "schema": store_manifest["schema"],
        },
        "development_acceptance": {
            "path": str(acceptance_path.resolve(strict=True)),
            "sha256": acceptance_sha256.casefold(),
            "status": "development_grade_inferred_actions",
        },
        "cdi": {
            "development_extension": {
                "path": str(cdi_path.resolve(strict=True)),
                "sha256": cdi_sha256,
            },
            "experiment52_reference": {
                "path": str(experiment52_cdi_path.resolve(strict=True)),
                "sha256": experiment52_cdi_sha256,
            },
        },
        "folds": folds,
        "baseline_roster": [
            "reversal_5",
            "reversal_21",
            "momentum_12_1",
            "reversal_5_momentum_12_1_blend",
            "inverse_volatility_20",
        ],
        "gbdt_rungs": {name: list(groups) for name, groups in RUNG_GROUPS.items()},
        "gbdt_seeds": list(GBDT_SEEDS),
        "data_span_arms": ["fine_only", "pretrain_uniform", "pretrain_decay_756"],
        "gbdt_num_threads": num_threads,
        "bootstrap": {
            "replications": BOOTSTRAP_REPLICATIONS,
            "block_length_sessions": BOOTSTRAP_BLOCK,
            "seed": BOOTSTRAP_SEED,
            "fold_boundary_preserved": True,
        },
        "economics_tier": {
            "price_source": "close_proxy",
            "terminal_settlement_convention": "last_mark_after_10_sessions",
            "settlement_grace_sessions": 10,
            "settlement_haircut": 0.30,
            "settlement_economics_unresolved_fraction_nav": 0.15,
            "executable_borrow": False,
        },
    }
    return write_json_atomic(output / "frozen_design.json", design)


def run_round1(
    *,
    output_root: Path,
    num_threads: int,
) -> str:
    output = output_root.resolve(strict=True)
    design_path = output / "frozen_design.json"
    design = _read_json(design_path)
    if (
        design.get("schema") != ROUND1_SCHEMA
        or design.get("status") != "frozen_before_score"
    ):
        raise ValueError("Round-1 root is not a frozen registered design")
    code = _git_identity()
    if design.get("implementation") != code:
        raise ValueError("Round-1 implementation differs from the frozen commit")
    if num_threads != design.get("gbdt_num_threads"):
        raise ValueError("Round-1 thread setting differs from the frozen design")
    if (output / "round1_result.json").exists():
        raise FileExistsError(output / "round1_result.json")
    store_root = Path(str(design["store"]["root"]))
    store_manifest, dates = _read_store_header(store_root)
    if sha256_file(store_root / "manifest.json") != design["store"]["manifest_sha256"]:
        raise ValueError("Round-1 store manifest hash mismatch")
    source_tiers = _source_tier_labels(store_manifest)
    if any(design.get(key) != value for key, value in source_tiers.items()):
        raise ValueError("Round-1 frozen source tiers differ from the store")
    acceptance = design.get("development_acceptance")
    if not isinstance(acceptance, Mapping):
        raise ValueError("Round-1 frozen design lacks development acceptance")
    _verify_development_acceptance(
        Path(str(acceptance["path"])),
        expected_sha256=str(acceptance["sha256"]),
        store_manifest_sha256=str(design["store"]["manifest_sha256"]),
        implementation=code,
        source_tiers=source_tiers,
    )
    fit, selection, evaluation, fit_target_window, _ = _fold_indices(dates)
    pretrain = _pretrain_indices(dates)
    cdi_design = design["cdi"]
    cdi, cdi_provenance = _load_development_cdi(
        dates=dates,
        cdi_path=Path(str(cdi_design["development_extension"]["path"])),
        expected_sha256=str(cdi_design["development_extension"]["sha256"]),
        experiment52_cdi_path=Path(str(cdi_design["experiment52_reference"]["path"])),
        experiment52_expected_sha256=str(
            cdi_design["experiment52_reference"]["sha256"]
        ),
    )
    store, access = _open_round_store(
        store_root, fit, selection, evaluation, fit_target_window, pretrain
    )
    source_hashes = {
        "v2_store_manifest": str(design["store"]["manifest_sha256"]),
        "cdi_development_extension": str(
            cdi_provenance["development_extension"]["sha256"]
        ),
        "cdi_experiment52_reference": str(
            cdi_provenance["experiment52_reference"]["sha256"]
        ),
        "preregistration": str(design["preregistration"]["sha256"]),
    }
    events: list[dict[str, object]] = [
        {"event": "round1_started", "at_utc": _utc_now()}
    ]
    try:
        baseline_start = max(0, min(int(x[0]) for x in evaluation.values()) - 253)
        baseline_end = max(int(x[-1]) for x in evaluation.values())
        baseline_axis = np.arange(baseline_start, baseline_end + 1, dtype=np.int64)
        panels = build_store_baselines(store, baseline_axis)
        baseline_reports: dict[str, dict[str, _ResearchEvaluation]] = {
            name: {} for name in panels
        }
        baseline_records: dict[str, dict[str, object]] = {name: {} for name in panels}
        for fold in ("F1", "F2", "F3"):
            indices = evaluation[fold]
            local = indices - baseline_start
            for name, panel in panels.items():
                assert isinstance(panel, BaselinePanel)
                root = output / "baselines" / name / fold
                manifest, manifest_sha = _persist_scores(
                    root,
                    {
                        "scores": panel.scores[local],
                        "score_mask": panel.score_mask[local],
                    },
                    {
                        **_source_tier_labels(store.manifest),
                        "engine": "naive_baseline",
                        "name": name,
                        "fold": fold,
                        "evaluation_date_indices": indices.tolist(),
                    },
                )
                result = _evaluate(
                    store=store,
                    indices=indices,
                    scores=panel.scores[local],
                    score_mask=panel.score_mask[local],
                    cdi=cdi,
                    source_hashes=source_hashes,
                    fold=fold,
                    output=root / "evaluation.json",
                )
                baseline_reports[name][fold] = result
                baseline_records[name][fold] = {
                    "score_manifest": str(manifest),
                    "score_manifest_sha256": manifest_sha,
                    "evaluation": str(root / "evaluation.json"),
                    "evaluation_sha256": sha256_file(root / "evaluation.json"),
                }
        baseline_summary = {
            name: _pooled_readouts(reports)
            for name, reports in baseline_reports.items()
        }
        events.append({"event": "baselines_completed", "at_utc": _utc_now()})

        rung_reports: dict[str, dict[str, _ResearchEvaluation]] = {}
        rung_records: dict[str, object] = {}
        rung_summaries: dict[str, object] = {}
        rung_comparisons: dict[str, object] = {}
        kept: list[str] = []
        previous: str | None = None
        for rung in RUNG_GROUPS:
            reports, records = _run_gbdt_candidate(
                store=store,
                rung=rung,
                fit=fit,
                fit_target_window=fit_target_window,
                selection=selection,
                evaluation=evaluation,
                pretrain=None,
                decay_half_life=None,
                cdi=cdi,
                source_hashes=source_hashes,
                root=output / "gbdt_ladder" / rung,
                num_threads=num_threads,
            )
            rung_reports[rung] = reports
            rung_records[rung] = records
            rung_summaries[rung] = _pooled_readouts(reports)
            if previous is None:
                kept.append(rung)
            else:
                paired = _paired_readouts(reports, rung_reports[previous])
                rung_comparisons[f"{rung}_minus_{previous}"] = paired
                pooled = paired["pooled"]
                if not (
                    _point_is_negative(pooled["primary_scaled_target_ic"])
                    and _point_is_negative(pooled["headline_net_excess_bps"])
                ):
                    kept.append(rung)
            previous = rung
            events.append({"event": f"{rung}_completed", "at_utc": _utc_now()})
        parent = max(
            kept,
            key=lambda rung: (
                _ranking_point(
                    rung_summaries[rung]["pooled"]["primary_scaled_target_ic"]
                ),
                _ranking_point(
                    rung_summaries[rung]["pooled"]["headline_net_excess_bps"]
                ),
                -list(RUNG_GROUPS).index(rung),
            ),
        )

        span_reports: dict[str, dict[str, _ResearchEvaluation]] = {
            "fine_only": rung_reports[parent]
        }
        span_records: dict[str, object] = {"fine_only": rung_records[parent]}
        for arm, decay in (("pretrain_uniform", None), ("pretrain_decay_756", 756.0)):
            reports, records = _run_gbdt_candidate(
                store=store,
                rung=parent,
                fit=fit,
                fit_target_window=fit_target_window,
                selection=selection,
                evaluation=evaluation,
                pretrain=pretrain,
                decay_half_life=decay,
                cdi=cdi,
                source_hashes=source_hashes,
                root=output / "gbdt_data_span" / arm,
                num_threads=num_threads,
            )
            span_reports[arm] = reports
            span_records[arm] = records
            events.append({"event": f"{arm}_completed", "at_utc": _utc_now()})
        span_summaries = {
            arm: _pooled_readouts(reports) for arm, reports in span_reports.items()
        }
        span_comparisons = {
            f"{arm}_minus_fine_only": _paired_readouts(
                reports, span_reports["fine_only"]
            )
            for arm, reports in span_reports.items()
            if arm != "fine_only"
        }
        events.append({"event": "round1_completed", "at_utc": _utc_now()})
        result = {
            "schema": ROUND1_SCHEMA,
            "status": "completed",
            **RESEARCH_FLAGS,
            **source_tiers,
            "completed_at_utc": _utc_now(),
            "frozen_design": {
                "path": str(design_path),
                "sha256": sha256_file(design_path),
            },
            "implementation": code,
            "store_access": access,
            "sources": source_hashes,
            "baselines": {"artifacts": baseline_records, "readouts": baseline_summary},
            "gbdt_ladder": {
                "artifacts": rung_records,
                "readouts": rung_summaries,
                "paired_deltas": rung_comparisons,
                "kept_rungs": kept,
                "parent_rung": parent,
                "preference_rule": (
                    "keep a rung if pooled-IC delta is positive with an interval mostly "
                    "above zero OR headline net excess improves; drop a rung that worsens "
                    "both; keep ambiguous rungs; parent is best kept pooled IC with "
                    "economics as tie-break"
                ),
            },
            "gbdt_data_span_preview": {
                "artifacts": span_records,
                "readouts": span_summaries,
                "paired_deltas": span_comparisons,
                "decision_weight": "informational_only",
            },
            "operational_events": events,
        }
        return write_json_atomic(output / "round1_result.json", result)
    finally:
        store.close()


def _existing_score_and_evaluation_record(root: Path) -> dict[str, object]:
    manifest_path = root / "score_manifest.json"
    evaluation_path = root / "evaluation.json"
    manifest = _read_json(manifest_path)
    _assert_false_access(manifest, path=manifest_path)
    if (
        manifest.get("schema") != RESEARCH_SCORE_SCHEMA
        or manifest.get("status") != "completed"
    ):
        raise ValueError(f"incomplete registered score artifact: {root}")
    if manifest.get("transfer_chronology_clean") is not True:
        raise PermissionError(
            f"registered score has contaminated transfer chronology: {root}"
        )
    _assert_source_tier_labels(
        manifest,
        expected={
            "action_terms_source": "inferred_cotahist_dismes_v1",
            "schedule_source": "reconstructed_v1",
        },
        path=manifest_path,
    )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"score manifest lacks artifacts: {manifest_path}")
    for name, raw_record in artifacts.items():
        if not isinstance(name, str) or not isinstance(raw_record, Mapping):
            raise ValueError(f"malformed score artifact record: {manifest_path}")
        path = root / name
        if (
            not path.is_file()
            or path.stat().st_size != int(raw_record["bytes"])
            or sha256_file(path) != raw_record["sha256"]
        ):
            raise ValueError(f"registered score artifact hash mismatch: {path}")
    metadata = manifest.get("metadata")
    if isinstance(metadata, Mapping) and isinstance(metadata.get("model"), Mapping):
        raw_model = metadata["model"]
        model_manifest = Path(str(raw_model["manifest"])).resolve(strict=True)
        expected = str(raw_model["manifest_sha256"])
        restored = MultiHorizonGBDT.load(
            model_manifest.parent,
            expected_manifest_sha256=expected,
        )
        del restored
    _evaluation_from_path(evaluation_path)
    return {
        "score_manifest": str(manifest_path),
        "score_manifest_sha256": sha256_file(manifest_path),
        "evaluation": str(evaluation_path),
        "evaluation_sha256": sha256_file(evaluation_path),
    }


def resume_round1(*, output_root: Path, num_threads: int) -> str:
    """Reuse complete candidates and score only registered candidates still absent."""
    output = output_root.resolve(strict=True)
    result_path = output / "round1_result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    design_path = output / "frozen_design.json"
    design = _read_json(design_path)
    if (
        design.get("schema") != ROUND1_SCHEMA
        or design.get("status") != "frozen_before_score"
    ):
        raise ValueError("Round-1 root is not a frozen registered design")
    recovery_code = _git_identity()
    if num_threads != design.get("gbdt_num_threads"):
        raise ValueError("Round-1 thread setting differs from the frozen design")

    store_root = Path(str(design["store"]["root"]))
    if sha256_file(store_root / "manifest.json") != design["store"]["manifest_sha256"]:
        raise ValueError("Round-1 store manifest hash mismatch")
    store_manifest, dates = _read_store_header(store_root)
    source_tiers = _source_tier_labels(store_manifest)
    if any(design.get(key) != value for key, value in source_tiers.items()):
        raise ValueError("Round-1 frozen source tiers differ from the store")
    acceptance = design.get("development_acceptance")
    if not isinstance(acceptance, Mapping):
        raise ValueError("Round-1 frozen design lacks development acceptance")
    _verify_development_acceptance(
        Path(str(acceptance["path"])),
        expected_sha256=str(acceptance["sha256"]),
        store_manifest_sha256=str(design["store"]["manifest_sha256"]),
        implementation=design["implementation"],
        source_tiers=source_tiers,
    )
    fit, selection, evaluation, fit_target_window, _ = _fold_indices(dates)
    pretrain = _pretrain_indices(dates)
    cdi_design = design["cdi"]
    cdi, cdi_provenance = _load_development_cdi(
        dates=dates,
        cdi_path=Path(str(cdi_design["development_extension"]["path"])),
        expected_sha256=str(cdi_design["development_extension"]["sha256"]),
        experiment52_cdi_path=Path(str(cdi_design["experiment52_reference"]["path"])),
        experiment52_expected_sha256=str(
            cdi_design["experiment52_reference"]["sha256"]
        ),
    )
    store, access = _open_round_store(
        store_root, fit, selection, evaluation, fit_target_window, pretrain
    )
    source_hashes = {
        "v2_store_manifest": str(design["store"]["manifest_sha256"]),
        "cdi_development_extension": str(
            cdi_provenance["development_extension"]["sha256"]
        ),
        "cdi_experiment52_reference": str(
            cdi_provenance["experiment52_reference"]["sha256"]
        ),
        "preregistration": str(design["preregistration"]["sha256"]),
    }
    reused_candidates = ["all_naive_baselines"]
    scored_candidates: list[str] = []

    baseline_reports: dict[str, dict[str, _ResearchEvaluation]] = {}
    baseline_records: dict[str, dict[str, object]] = {}
    for raw_name in design["baseline_roster"]:
        name = str(raw_name)
        baseline_reports[name] = {}
        baseline_records[name] = {}
        for fold in ("F1", "F2", "F3"):
            root = output / "baselines" / name / fold
            baseline_reports[name][fold] = _evaluation_from_artifacts(
                root / "evaluation.json",
                store=store,
                indices=evaluation[fold],
                cdi=cdi,
            )
            baseline_records[name][fold] = _existing_score_and_evaluation_record(root)
    baseline_summary = {
        name: _pooled_readouts(reports) for name, reports in baseline_reports.items()
    }

    rung_reports: dict[str, dict[str, _ResearchEvaluation]] = {}
    rung_records: dict[str, object] = {}
    rung_summaries: dict[str, object] = {}
    rung_comparisons: dict[str, object] = {}
    kept: list[str] = []
    previous: str | None = None
    for rung in RUNG_GROUPS:
        candidate_root = output / "gbdt_ladder" / rung
        completed = tuple(
            (candidate_root / fold / "evaluation.json").is_file()
            for fold in ("F1", "F2", "F3")
        )
        if all(completed):
            reports = {}
            records = {}
            for fold in ("F1", "F2", "F3"):
                root = candidate_root / fold
                reports[fold] = _evaluation_from_artifacts(
                    root / "evaluation.json",
                    store=store,
                    indices=evaluation[fold],
                    cdi=cdi,
                )
                records[fold] = _existing_score_and_evaluation_record(root)
            reused_candidates.append(f"gbdt_ladder/{rung}")
        elif any(completed) or candidate_root.exists():
            raise ValueError(f"refusing to resume partial registered candidate: {rung}")
        else:
            reports, records = _run_gbdt_candidate(
                store=store,
                rung=rung,
                fit=fit,
                fit_target_window=fit_target_window,
                selection=selection,
                evaluation=evaluation,
                pretrain=None,
                decay_half_life=None,
                cdi=cdi,
                source_hashes=source_hashes,
                root=candidate_root,
                num_threads=num_threads,
            )
            scored_candidates.append(f"gbdt_ladder/{rung}")
        rung_reports[rung] = reports
        rung_records[rung] = records
        rung_summaries[rung] = _pooled_readouts(reports)
        if previous is None:
            kept.append(rung)
        else:
            paired = _paired_readouts(reports, rung_reports[previous])
            rung_comparisons[f"{rung}_minus_{previous}"] = paired
            pooled = paired["pooled"]
            if not (
                _point_is_negative(pooled["primary_scaled_target_ic"])
                and _point_is_negative(pooled["headline_net_excess_bps"])
            ):
                kept.append(rung)
        previous = rung
    parent = max(
        kept,
        key=lambda rung: (
            _ranking_point(rung_summaries[rung]["pooled"]["primary_scaled_target_ic"]),
            _ranking_point(rung_summaries[rung]["pooled"]["headline_net_excess_bps"]),
            -list(RUNG_GROUPS).index(rung),
        ),
    )

    span_reports: dict[str, dict[str, _ResearchEvaluation]] = {
        "fine_only": rung_reports[parent]
    }
    span_records: dict[str, object] = {"fine_only": rung_records[parent]}
    for arm in ("pretrain_uniform", "pretrain_decay_756"):
        candidate_root = output / "gbdt_data_span" / arm
        completed = tuple(
            (candidate_root / fold / "evaluation.json").is_file()
            for fold in ("F1", "F2", "F3")
        )
        if all(completed):
            reports = {}
            records = {}
            for fold in ("F1", "F2", "F3"):
                root = candidate_root / fold
                reports[fold] = _evaluation_from_artifacts(
                    root / "evaluation.json",
                    store=store,
                    indices=evaluation[fold],
                    cdi=cdi,
                )
                records[fold] = _existing_score_and_evaluation_record(root)
            reused_candidates.append(f"gbdt_data_span/{arm}")
        elif any(completed) or candidate_root.exists():
            raise ValueError(
                f"refusing to resume partial registered data-span candidate: {arm}"
            )
        else:
            decay = None if arm == "pretrain_uniform" else 756.0
            reports, records = _run_gbdt_candidate(
                store=store,
                rung=parent,
                fit=fit,
                fit_target_window=fit_target_window,
                selection=selection,
                evaluation=evaluation,
                pretrain=pretrain,
                decay_half_life=decay,
                cdi=cdi,
                source_hashes=source_hashes,
                root=candidate_root,
                num_threads=num_threads,
            )
            scored_candidates.append(f"gbdt_data_span/{arm}")
        span_reports[arm] = reports
        span_records[arm] = records
    span_summaries = {
        arm: _pooled_readouts(reports) for arm, reports in span_reports.items()
    }
    span_comparisons = {
        f"{arm}_minus_fine_only": _paired_readouts(reports, span_reports["fine_only"])
        for arm, reports in span_reports.items()
        if arm != "fine_only"
    }

    store.close()
    result = {
        "schema": ROUND1_SCHEMA,
        "status": "completed",
        **RESEARCH_FLAGS,
        **source_tiers,
        "completed_at_utc": _utc_now(),
        "frozen_design": {
            "path": str(design_path),
            "sha256": sha256_file(design_path),
        },
        "implementation": recovery_code,
        "score_implementation": design["implementation"],
        "reporting_recovery": {
            "scope": (
                "ignore prediction-specific hashes in paired-input identity, serialize "
                "undefined zero-support readouts as JSON null, hash-verify and reuse "
                "complete candidates, and score only registered candidates still absent"
            ),
            "implementation": recovery_code,
            "reused_candidates": reused_candidates,
            "scored_candidates": scored_candidates,
            "result_changing_retry": False,
        },
        "store_access": access,
        "sources": source_hashes,
        "baselines": {"artifacts": baseline_records, "readouts": baseline_summary},
        "gbdt_ladder": {
            "artifacts": rung_records,
            "readouts": rung_summaries,
            "paired_deltas": rung_comparisons,
            "kept_rungs": kept,
            "parent_rung": parent,
            "preference_rule": (
                "keep a rung if pooled-IC delta is positive with an interval mostly "
                "above zero OR headline net excess improves; drop a rung that worsens "
                "both; keep ambiguous rungs; parent is best kept pooled IC with "
                "economics as tie-break"
            ),
        },
        "gbdt_data_span_preview": {
            "artifacts": span_records,
            "readouts": span_summaries,
            "paired_deltas": span_comparisons,
            "decision_weight": "informational_only",
        },
        "operational_events": [
            {
                "event": "round1_reporting_recovered_from_completed_artifacts",
                "at_utc": _utc_now(),
            }
        ],
    }
    return write_json_atomic(result_path, result)


def _verify_sealed_root(root: Path, *, expected_schema: str) -> dict[str, object]:
    source = root.resolve(strict=True)
    inventory_path = source / "artifact_inventory.json"
    inventory_payload = _read_json(inventory_path)
    if (
        inventory_payload.get("schema") != "BRAZIL_RV_V2_RESEARCH_INVENTORY_V1"
        or inventory_payload.get("status") != "passed"
    ):
        raise ValueError("source research root lacks its passing inventory")
    rows = inventory_payload.get("files")
    excluded = inventory_payload.get("excluded_self")
    if (
        not isinstance(rows, list)
        or not isinstance(excluded, list)
        or any(not isinstance(row, dict) for row in rows)
        or any(not isinstance(value, str) for value in excluded)
    ):
        raise ValueError("source research inventory is malformed")
    if inventory(source, exclude=set(excluded)) != rows:
        raise ValueError("source research inventory no longer matches its files")
    result_name = (
        "round1_result.json"
        if expected_schema == ROUND1_SCHEMA
        else "round2_result.json"
    )
    result = _read_json(source / result_name)
    if result.get("schema") != expected_schema or result.get("status") != "completed":
        raise ValueError("source research result is incomplete")
    _assert_false_access(result, path=source / result_name)
    if result.get("transfer_chronology_clean") is not True:
        raise PermissionError(
            "canonical research rounds refuse contaminated transfer chronology"
        )
    return result


def freeze_round2(
    *,
    round1_root: Path,
    store_root: Path,
    cdi_path: Path,
    cdi_sha256: str,
    experiment52_cdi_path: Path,
    experiment52_cdi_sha256: str,
    output_root: Path,
    fast_checkpoint: Path | None,
    fast_checkpoint_sha256: str | None,
    max_parallel: int,
) -> str:
    code = _git_identity()
    if not 4 <= max_parallel <= 6:
        raise ValueError("Round 2 requires four to six concurrent trajectories")
    round1_path = round1_root.resolve(strict=True)
    round1 = _verify_sealed_root(round1_path, expected_schema=ROUND1_SCHEMA)
    if round1.get("implementation") != code:
        raise ValueError("Round 1 and Round 2 must use the same frozen implementation")
    parent = round1["gbdt_ladder"]["parent_rung"]
    if parent not in RUNG_GROUPS:
        raise ValueError("Round-1 GBDT parent is not a registered rung")
    output = output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    store = store_root.resolve(strict=True)
    store_manifest, dates = _read_store_header(store)
    store_manifest_sha256 = sha256_file(store / "manifest.json")
    source_tiers = _source_tier_labels(store_manifest)
    if any(round1.get(key) != value for key, value in source_tiers.items()):
        raise ValueError("Round 1 and Round 2 source tiers differ")
    if store_manifest_sha256 != round1.get("sources", {}).get("v2_store_manifest"):
        raise ValueError("Round 2 must use the exact Round-1 development store")
    _fold_indices(dates)
    _load_development_cdi(
        dates=dates,
        cdi_path=cdi_path,
        expected_sha256=cdi_sha256,
        experiment52_cdi_path=experiment52_cdi_path,
        experiment52_expected_sha256=experiment52_cdi_sha256,
    )
    if fast_checkpoint is not None or fast_checkpoint_sha256 is not None:
        raise ValueError(
            "canonical Round 2 requires the native fresh fast encoder; legacy v1 "
            "checkpoint transfer has contaminated chronology and belongs in a "
            "separately labelled development-only ablation"
        )
    fast: dict[str, object] = {
        "mode": "native_fresh",
        "transfer_chronology_clean": True,
    }
    output.mkdir(parents=True, exist_ok=False)
    design = {
        "schema": ROUND2_SCHEMA,
        "status": "frozen_before_score",
        **RESEARCH_FLAGS,
        "frozen_at_utc": _utc_now(),
        "implementation": code,
        **source_tiers,
        "preregistration": {
            "path": str(PREREGISTRATION.resolve(strict=True)),
            "sha256": sha256_file(PREREGISTRATION),
        },
        "round1": {
            "root": str(round1_path),
            "result_sha256": sha256_file(round1_path / "round1_result.json"),
            "inventory_sha256": sha256_file(round1_path / "artifact_inventory.json"),
            "gbdt_parent_rung": parent,
        },
        "store": {
            "root": str(store),
            "manifest_sha256": store_manifest_sha256,
            "schema": store_manifest["schema"],
        },
        "cdi": {
            "development_extension": {
                "path": str(cdi_path.resolve(strict=True)),
                "sha256": cdi_sha256,
            },
            "experiment52_reference": {
                "path": str(experiment52_cdi_path.resolve(strict=True)),
                "sha256": experiment52_cdi_sha256,
            },
        },
        "enabled_sidecars": list(RUNG_GROUPS[parent]),
        "fast_initialization": fast,
        "network": {
            "seeds": list(NETWORK_SEEDS),
            "folds": ["F1", "F2", "F3"],
            "arms": ["A_fine_only", "B_pretrain_finetune", "C_joint_decay_756"],
            "maximum_epochs": 20,
            "patience": 3,
            "lambda_persistence": 0.0,
            "lookback_sessions": 60,
            "pairs_per_batch": 8,
            "pretrained_learning_rate_multiplier": 0.3,
            "joint_time_decay_half_life_sessions": 756.0,
        },
        "first_failure_stop": {
            "required_smoke": "arm_A_F1_seed_11_one_epoch_no_scores",
            "smoke_reused_by_registered_runs": False,
        },
        "max_parallel_trajectories": max_parallel,
        "bootstrap": {
            "replications": BOOTSTRAP_REPLICATIONS,
            "block_length_sessions": BOOTSTRAP_BLOCK,
            "seed": BOOTSTRAP_SEED,
            "fold_boundary_preserved": True,
        },
    }
    return write_json_atomic(output / "frozen_design.json", design)


def _training_command(
    *,
    design: Mapping[str, object],
    output_dir: Path,
    stage: str,
    seed: int,
    fold: str | None = None,
    pretrain_checkpoint: Path | None = None,
    pretrain_sha256: str | None = None,
    maximum_epochs: int = 20,
    score_output: bool = True,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "brazil_rv.v2.train",
        "--store",
        str(design["store"]["root"]),
        "--output-dir",
        str(output_dir),
        "--stage",
        stage,
        "--seed",
        str(seed),
        "--maximum-epochs",
        str(maximum_epochs),
        "--patience",
        "3",
        "--lookback",
        "60",
        "--pairs-per-batch",
        "8",
        "--selection-batch-size",
        "1",
        "--num-workers",
        "0",
        "--lambda-persistence",
        "0.0",
        "--device",
        "cuda",
    ]
    if score_output:
        command.extend(("--score-output-dir", str(output_dir / "scores")))
    if fold is not None:
        command.extend(("--fold", fold))
    for group in design["enabled_sidecars"]:
        command.extend(("--sidecar", str(group)))
    fast = design.get("fast_initialization")
    if not isinstance(fast, Mapping) or fast != {
        "mode": "native_fresh",
        "transfer_chronology_clean": True,
    }:
        raise ValueError(
            "Round-2 design lacks the canonical native fast initialization"
        )
    if pretrain_checkpoint is not None:
        if pretrain_sha256 is None:
            raise ValueError("stage-P handoff hash is required")
        command.extend(
            (
                "--pretrain-checkpoint",
                str(pretrain_checkpoint),
                "--pretrain-sha256",
                pretrain_sha256,
            )
        )
    return command


def _plan_job(
    *,
    name: str,
    seed: int,
    fold: str,
    run_dir: Path,
    command: Sequence[str],
    stage: str,
    source_tiers: Mapping[str, str],
) -> dict[str, object]:
    return {
        "name": name,
        "seed": seed,
        "fold": fold,
        "run_dir": str(run_dir),
        "cwd": str(PROJECT_ROOT),
        "command": list(command),
        "expected_manifest": {
            "schema": TRAINING_STAGE_SCHEMA,
            "status": "completed",
            "stage": stage,
            "official_validation_accessed": False,
            "test_accessed": False,
            "transfer_chronology_clean": True,
            **source_tiers,
        },
    }


def write_round2_plan_smoke(*, output_root: Path) -> str:
    root = output_root.resolve(strict=True)
    design = _read_json(root / "frozen_design.json")
    if (
        design.get("schema") != ROUND2_SCHEMA
        or design.get("status") != "frozen_before_score"
    ):
        raise ValueError("Round-2 root is not frozen")
    run_dir = root / "smoke" / "arm_A_F1_seed_11"
    job = _plan_job(
        name="smoke_arm_A_F1_seed_11",
        seed=11,
        fold="F1",
        run_dir=run_dir,
        command=_training_command(
            design=design,
            output_dir=run_dir,
            stage="F",
            seed=11,
            fold="F1",
            maximum_epochs=1,
            score_output=False,
        ),
        stage="F",
        source_tiers={
            "action_terms_source": str(design["action_terms_source"]),
            "schedule_source": str(design["schedule_source"]),
        },
    )
    return write_json_atomic(
        root / "round2_plan_smoke.json",
        {
            "schema": RUN_MANY_PLAN_SCHEMA,
            "phase": "rev2_smoke",
            "max_parallel": 1,
            "first_failure_stop": True,
            "research_candidate_score": False,
            "reused_by_registered_runs": False,
            "jobs": [job],
        },
    )


def write_round2_plan_p(*, output_root: Path) -> str:
    root = output_root.resolve(strict=True)
    design = _read_json(root / "frozen_design.json")
    if (
        design.get("schema") != ROUND2_SCHEMA
        or design.get("status") != "frozen_before_score"
    ):
        raise ValueError("Round-2 root is not frozen")
    if not (root / "round2_plan_smoke.json").is_file():
        raise FileNotFoundError("Round-2 smoke plan is absent")
    smoke_root = root / "smoke" / "arm_A_F1_seed_11"
    smoke_manifest_path = smoke_root / "run_manifest.json"
    smoke = _read_json(smoke_manifest_path)
    _assert_current_clean_training(smoke, path=smoke_manifest_path)
    if (
        smoke.get("stage") != "F"
        or smoke.get("seed") != 11
        or smoke.get("fold") != "F1"
        or smoke.get("epochs_completed") != 1
        or (smoke_root / "scores").exists()
    ):
        raise ValueError("Round-2 first-smoke contract did not pass exactly")
    jobs = []
    for seed in NETWORK_SEEDS:
        run_dir = root / "trajectories" / "arm_B" / "stage_P" / f"seed_{seed}"
        jobs.append(
            _plan_job(
                name=f"arm_B_stage_P_seed_{seed}",
                seed=seed,
                fold="pretrain_internal",
                run_dir=run_dir,
                command=_training_command(
                    design=design, output_dir=run_dir, stage="P", seed=seed
                ),
                stage="P",
                source_tiers={
                    "action_terms_source": str(design["action_terms_source"]),
                    "schedule_source": str(design["schedule_source"]),
                },
            )
        )
    return write_json_atomic(
        root / "round2_plan_p.json",
        {
            "schema": RUN_MANY_PLAN_SCHEMA,
            "phase": "rev2_stage_P",
            "max_parallel": int(design["max_parallel_trajectories"]),
            "jobs": jobs,
        },
    )


def write_round2_plan_main(*, output_root: Path) -> str:
    root = output_root.resolve(strict=True)
    design = _read_json(root / "frozen_design.json")
    if (
        design.get("schema") != ROUND2_SCHEMA
        or design.get("status") != "frozen_before_score"
    ):
        raise ValueError("Round-2 root is not frozen under the current schema")
    if not (root / "round2_plan_p.json").is_file():
        raise FileNotFoundError("Stage-P plan is absent")
    handoffs: dict[int, tuple[Path, str]] = {}
    for seed in NETWORK_SEEDS:
        p_root = root / "trajectories" / "arm_B" / "stage_P" / f"seed_{seed}"
        manifest = _read_json(p_root / "run_manifest.json")
        _assert_current_clean_training(manifest, path=p_root / "run_manifest.json")
        if manifest.get("stage") != "P" or manifest.get("seed") != seed:
            raise ValueError(f"Stage-P trajectory is incomplete for seed {seed}")
        checkpoint = p_root / "raw_patience.pt"
        digest = sha256_file(checkpoint)
        if manifest.get("artifacts", {}).get("raw_patience.pt") != digest:
            raise ValueError(f"Stage-P checkpoint manifest mismatch for seed {seed}")
        handoffs[seed] = (checkpoint, digest)
    jobs = []
    arms = (
        ("arm_A", "F", False),
        ("arm_B", "F", True),
        ("arm_C", "J", False),
    )
    for arm, stage, uses_handoff in arms:
        for fold in ("F1", "F2", "F3"):
            for seed in NETWORK_SEEDS:
                run_dir = root / "trajectories" / arm / f"{fold}_seed_{seed}"
                handoff, handoff_sha = handoffs[seed] if uses_handoff else (None, None)
                jobs.append(
                    _plan_job(
                        name=f"{arm}_{fold}_seed_{seed}",
                        seed=seed,
                        fold=fold,
                        run_dir=run_dir,
                        command=_training_command(
                            design=design,
                            output_dir=run_dir,
                            stage=stage,
                            seed=seed,
                            fold=fold,
                            pretrain_checkpoint=handoff,
                            pretrain_sha256=handoff_sha,
                        ),
                        stage=stage,
                        source_tiers={
                            "action_terms_source": str(design["action_terms_source"]),
                            "schedule_source": str(design["schedule_source"]),
                        },
                    )
                )
    return write_json_atomic(
        root / "round2_plan_main.json",
        {
            "schema": RUN_MANY_PLAN_SCHEMA,
            "phase": "rev2_registered_arms",
            "max_parallel": int(design["max_parallel_trajectories"]),
            "stage_p_handoffs": {
                str(seed): {"path": str(path), "sha256": digest}
                for seed, (path, digest) in handoffs.items()
            },
            "jobs": jobs,
        },
    )


def _score_artifact(
    root: Path,
    *,
    require_clean_transfer: bool = False,
    expected_dates: NDArray[np.datetime64] | None = None,
    expected_isins: Sequence[str] | None = None,
    expected_feature_schema_sha256: str | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    manifest_path = root / "score_manifest.json"
    manifest = _read_json(manifest_path)
    _assert_false_access(manifest, path=manifest_path)
    if (
        manifest.get("schema")
        not in {
            SCORE_ARTIFACT_SCHEMA,
            RESEARCH_SCORE_SCHEMA,
        }
        or manifest.get("status") != "completed"
    ):
        raise ValueError(f"incomplete score artifact: {root}")
    _assert_source_tier_labels(
        manifest,
        expected={
            "action_terms_source": "inferred_cotahist_dismes_v1",
            "schedule_source": "reconstructed_v1",
        },
        path=manifest_path,
    )
    if require_clean_transfer and manifest.get("transfer_chronology_clean") is not True:
        raise PermissionError(
            f"score artifact has contaminated transfer chronology: {root}"
        )
    scores_path = root / "scores.npy"
    mask_path = root / "score_mask.npy"
    paths = [scores_path, mask_path]
    if manifest["schema"] == SCORE_ARTIFACT_SCHEMA:
        paths.extend((root / "date_index.npy", root / "isin_index.npy"))
    records = manifest.get("artifacts")
    if not isinstance(records, Mapping):
        raise ValueError(f"score artifact lacks its inventory: {root}")
    for path in paths:
        record = records.get(path.name)
        if not isinstance(record, Mapping):
            raise ValueError(f"score artifact omits {path.name}: {root}")
        if (
            path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"score artifact hash mismatch: {path}")
    if manifest["schema"] == SCORE_ARTIFACT_SCHEMA:
        if (
            expected_feature_schema_sha256 is None
            or manifest.get("feature_schema_sha256") != expected_feature_schema_sha256
        ):
            raise ValueError("network score feature schema differs from the store")
        dates = np.asarray(
            np.load(root / "date_index.npy", allow_pickle=False),
            dtype="datetime64[D]",
        )
        isins = tuple(
            str(value)
            for value in np.load(root / "isin_index.npy", allow_pickle=False).tolist()
        )
        if expected_dates is None or expected_isins is None:
            raise ValueError("canonical network scores require expected immutable axes")
        if not np.array_equal(dates, np.asarray(expected_dates, dtype="datetime64[D]")):
            raise ValueError("network score date axis differs from its registered fold")
        if isins != tuple(str(value) for value in expected_isins):
            raise ValueError("network score security axis differs from the store")
    return (
        np.load(scores_path, allow_pickle=False),
        np.load(mask_path, allow_pickle=False),
    )


def _aggregate_network_fold(
    root: Path,
    arm: str,
    fold: str,
    *,
    expected_dates: NDArray[np.datetime64],
    expected_isins: Sequence[str],
    expected_feature_schema_sha256: str,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    members = []
    reference_mask: NDArray[np.bool_] | None = None
    for seed in NETWORK_SEEDS:
        run = root / "trajectories" / arm / f"{fold}_seed_{seed}"
        manifest = _read_json(run / "run_manifest.json")
        _assert_current_clean_training(manifest, path=run / "run_manifest.json")
        if manifest.get("seed") != seed or manifest.get("fold") != fold:
            raise ValueError(f"trajectory is incomplete: {run}")
        scores, mask = _score_artifact(
            run / "scores",
            require_clean_transfer=True,
            expected_dates=expected_dates,
            expected_isins=expected_isins,
            expected_feature_schema_sha256=expected_feature_schema_sha256,
        )
        if reference_mask is None:
            reference_mask = mask
        elif not np.array_equal(reference_mask, mask):
            raise ValueError("network member score masks differ")
        members.append(scores)
    assert reference_mask is not None
    ensemble = rank_average_ensemble(members, reference_mask)
    return ensemble, reference_mask


def _load_round1_parent_fold(
    round1_root: Path,
    parent: str,
    fold: str,
    *,
    expected_indices: NDArray[np.int64],
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    root = round1_root / "gbdt_ladder" / parent / fold
    manifest = _read_json(root / "score_manifest.json")
    _assert_false_access(manifest, path=root / "score_manifest.json")
    if manifest.get("schema") != RESEARCH_SCORE_SCHEMA:
        raise ValueError("Round-1 parent score schema is stale")
    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping) or (
        metadata.get("fold") != fold
        or metadata.get("evaluation_date_indices") != expected_indices.tolist()
    ):
        raise ValueError("Round-1 parent score axis differs from its registered fold")
    return _score_artifact(root, require_clean_transfer=True)


def _evaluation_from_path(path: Path) -> dict[str, object]:
    payload = _read_json(path)
    _assert_false_access(payload, path=path)
    if payload.get("schema") != EVALUATION_SCHEMA:
        raise ValueError(f"not a v2 evaluation: {path}")
    return payload


def _evaluation_from_artifacts(
    path: Path,
    *,
    store: V2Store,
    indices: NDArray[np.int64],
    cdi: NDArray[np.float64],
) -> _ResearchEvaluation:
    """Rebuild retained comparison inputs from hash-bound score/store artifacts."""

    report = _evaluation_from_path(path)
    transfer_chronology_clean = report.get("transfer_chronology_clean")
    if transfer_chronology_clean is not True:
        raise PermissionError(
            f"evaluation has contaminated or unknown transfer chronology: {path}"
        )
    scores, score_mask = _score_artifact(path.parent, require_clean_transfer=True)
    score_manifest = _read_json(path.parent / "score_manifest.json")
    if score_manifest.get("transfer_chronology_clean") is not True:
        raise ValueError(f"score/evaluation transfer chronology differs: {path}")
    metadata = score_manifest.get("metadata")
    if not isinstance(metadata, Mapping) or (
        metadata.get("evaluation_date_indices") != indices.tolist()
    ):
        raise ValueError(f"score artifact axis differs from the evaluation: {path}")
    source_hashes = report.get("source_artifact_hashes")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ValueError(f"evaluation lacks source artifact hashes: {path}")
    inputs = _evaluation_inputs(
        store,
        indices,
        scores,
        score_mask,
        cdi,
        {str(key): str(value) for key, value in source_hashes.items()},
        transfer_chronology_clean=True,
    )
    recorded_hashes = report.get("input_hashes")
    rebuilt_hashes = _input_hashes(inputs)
    if recorded_hashes != rebuilt_hashes:
        recorded = recorded_hashes if isinstance(recorded_hashes, Mapping) else {}
        differing = sorted(
            key
            for key in set(recorded) | set(rebuilt_hashes)
            if recorded.get(key) != rebuilt_hashes.get(key)
        )
        raise ValueError(
            "evaluation inputs no longer match score/store artifacts: "
            f"{path}; differing identities: {differing}"
        )
    (
        primary_scores,
        primary_targets,
        primary_outcome_mask,
        primary_score_mask,
    ) = _primary_population_components(inputs)
    _, daily_primary, _ = _primary_daily_metrics(
        primary_scores,
        primary_targets,
        primary_outcome_mask,
        primary_score_mask,
        inputs.dates,
    )
    economics = report.get("economics")
    if not isinstance(economics, Mapping) or not isinstance(
        economics.get("daily_table"), list
    ):
        raise ValueError(f"evaluation lacks headline economics: {path}")
    rows = [
        row
        for row in economics["daily_table"]
        if isinstance(row, Mapping)
        and float(row.get("cost_bps_per_side", -1.0)) == 4.0
        and float(row.get("annual_borrow_rate", -1.0)) == 0.02
    ]
    by_date: dict[str, float] = {}
    for row in rows:
        key = str(row.get("date"))
        if key in by_date:
            raise ValueError(
                f"evaluation has duplicate headline economics dates: {path}"
            )
        value = row.get("net_excess_all_cash_bps")
        by_date[key] = np.nan if value is None else float(value)
    headline = np.asarray(
        [by_date.get(value.isoformat(), np.nan) for value in inputs.dates],
        dtype=np.float64,
    )
    retained = EvaluationResult(
        report=report,
        dates=inputs.dates,
        daily_primary_ic=daily_primary,
        headline_economics_dates=inputs.dates,
        headline_net_excess_bps=headline,
        primary_scores=primary_scores,
        primary_targets=primary_targets,
        primary_outcome_mask=primary_outcome_mask,
        primary_score_mask=primary_score_mask,
    )
    return _ResearchEvaluation(result=retained, inputs=inputs)


def finalize_round2(*, output_root: Path) -> str:
    root = output_root.resolve(strict=True)
    design = _read_json(root / "frozen_design.json")
    if (
        design.get("schema") != ROUND2_SCHEMA
        or design.get("implementation") != _git_identity()
    ):
        raise ValueError("Round-2 design differs from the clean implementation")
    if (root / "round2_result.json").exists():
        raise FileExistsError(root / "round2_result.json")
    round1_root = Path(str(design["round1"]["root"]))
    _verify_sealed_root(round1_root, expected_schema=ROUND1_SCHEMA)
    parent = str(design["round1"]["gbdt_parent_rung"])
    store_root = Path(str(design["store"]["root"]))
    store_manifest, dates = _read_store_header(store_root)
    fit, selection, evaluation, fit_target_window, _ = _fold_indices(dates)
    pretrain = _pretrain_indices(dates)
    store, access = _open_round_store(
        store_root, fit, selection, evaluation, fit_target_window, pretrain
    )
    cdi_design = design["cdi"]
    cdi, provenance = _load_development_cdi(
        dates=dates,
        cdi_path=Path(str(cdi_design["development_extension"]["path"])),
        expected_sha256=str(cdi_design["development_extension"]["sha256"]),
        experiment52_cdi_path=Path(str(cdi_design["experiment52_reference"]["path"])),
        experiment52_expected_sha256=str(
            cdi_design["experiment52_reference"]["sha256"]
        ),
    )
    source_hashes = {
        "v2_store_manifest": str(design["store"]["manifest_sha256"]),
        "cdi_development_extension": str(provenance["development_extension"]["sha256"]),
        "cdi_experiment52_reference": str(
            provenance["experiment52_reference"]["sha256"]
        ),
        "preregistration": str(design["preregistration"]["sha256"]),
        "round1_result": str(design["round1"]["result_sha256"]),
    }
    arm_reports: dict[str, dict[str, _ResearchEvaluation]] = {}
    arm_artifacts: dict[str, object] = {}
    try:
        for arm in ("arm_A", "arm_B", "arm_C"):
            arm_reports[arm] = {}
            arm_artifacts[arm] = {}
            for fold in ("F1", "F2", "F3"):
                scores, mask = _aggregate_network_fold(
                    root,
                    arm,
                    fold,
                    expected_dates=store.dates[evaluation[fold]],
                    expected_isins=store.isins,
                    expected_feature_schema_sha256=str(
                        store.manifest["feature_schema_sha256"]
                    ),
                )
                aggregate = root / "aggregates" / arm / fold
                manifest, digest = _persist_scores(
                    aggregate,
                    {"scores": scores, "score_mask": mask},
                    {
                        **_source_tier_labels(store.manifest),
                        "engine": "starter_network",
                        "arm": arm,
                        "fold": fold,
                        "seeds": list(NETWORK_SEEDS),
                        "seed_aggregation": "tie-aware rank average",
                    },
                )
                evaluated = _evaluate(
                    store=store,
                    indices=evaluation[fold],
                    scores=scores,
                    score_mask=mask,
                    cdi=cdi,
                    source_hashes=source_hashes,
                    fold=fold,
                    output=aggregate / "evaluation.json",
                )
                arm_reports[arm][fold] = evaluated
                arm_artifacts[arm][fold] = {
                    "score_manifest": str(manifest),
                    "score_manifest_sha256": digest,
                    "evaluation": str(aggregate / "evaluation.json"),
                    "evaluation_sha256": sha256_file(aggregate / "evaluation.json"),
                }
        arm_readouts = {
            arm: _pooled_readouts(reports) for arm, reports in arm_reports.items()
        }
        arm_deltas = {
            "B_minus_A": _paired_readouts(arm_reports["arm_B"], arm_reports["arm_A"]),
            "C_minus_A": _paired_readouts(arm_reports["arm_C"], arm_reports["arm_A"]),
        }
        eligible = ["arm_A"]
        a_economics = arm_readouts["arm_A"]["pooled"]["headline_net_excess_bps"]
        for arm in ("arm_B", "arm_C"):
            if _economics_not_worse(
                arm_readouts[arm]["pooled"]["headline_net_excess_bps"],
                a_economics,
            ):
                eligible.append(arm)
        long_small_and_uncertain = all(
            _small_interval_spanning_zero(
                arm_deltas[label]["pooled"]["primary_scaled_target_ic"]
            )
            for label in ("B_minus_A", "C_minus_A")
        )
        arm_order = {"arm_A": 2, "arm_B": 1, "arm_C": 0}
        chosen_arm = (
            "arm_A"
            if long_small_and_uncertain
            else max(
                eligible,
                key=lambda arm: (
                    _ranking_point(
                        arm_readouts[arm]["pooled"]["primary_scaled_target_ic"]
                    ),
                    _ranking_point(
                        arm_readouts[arm]["pooled"]["headline_net_excess_bps"]
                    ),
                    arm_order[arm],
                ),
            )
        )

        comparator_reports: dict[str, dict[str, _ResearchEvaluation]] = {
            "network": arm_reports[chosen_arm],
            "gbdt": {},
            "ensemble": {},
        }
        comparator_artifacts: dict[str, object] = {
            "network": arm_artifacts[chosen_arm],
            "gbdt": {},
            "ensemble": {},
        }
        for fold in ("F1", "F2", "F3"):
            network_scores, network_mask = _score_artifact(
                root / "aggregates" / chosen_arm / fold,
                require_clean_transfer=True,
            )
            gbdt_scores, gbdt_mask = _load_round1_parent_fold(
                round1_root,
                parent,
                fold,
                expected_indices=evaluation[fold],
            )
            if not np.array_equal(network_mask, gbdt_mask):
                raise ValueError("network and GBDT score masks differ")
            gbdt_root = root / "comparators" / "gbdt" / fold
            g_manifest, g_digest = _persist_scores(
                gbdt_root,
                {"scores": gbdt_scores, "score_mask": gbdt_mask},
                {
                    **_source_tier_labels(store.manifest),
                    "engine": "round1_gbdt_parent",
                    "parent_rung": parent,
                    "fold": fold,
                },
            )
            g_eval = _evaluate(
                store=store,
                indices=evaluation[fold],
                scores=gbdt_scores,
                score_mask=gbdt_mask,
                cdi=cdi,
                source_hashes=source_hashes,
                fold=fold,
                output=gbdt_root / "evaluation.json",
            )
            comparator_reports["gbdt"][fold] = g_eval
            comparator_artifacts["gbdt"][fold] = {
                "score_manifest": str(g_manifest),
                "score_manifest_sha256": g_digest,
                "evaluation": str(gbdt_root / "evaluation.json"),
                "evaluation_sha256": sha256_file(gbdt_root / "evaluation.json"),
            }

            ensemble_scores = rank_average_ensemble(
                (network_scores, gbdt_scores), network_mask
            )
            ensemble_root = root / "comparators" / "ensemble" / fold
            e_manifest, e_digest = _persist_scores(
                ensemble_root,
                {"scores": ensemble_scores, "score_mask": network_mask},
                {
                    **_source_tier_labels(store.manifest),
                    "engine": "fixed_equal_weight_rank_average",
                    "members": ["network", "gbdt"],
                    "weights": [0.5, 0.5],
                    "fold": fold,
                },
            )
            e_eval = _evaluate(
                store=store,
                indices=evaluation[fold],
                scores=ensemble_scores,
                score_mask=network_mask,
                cdi=cdi,
                source_hashes=source_hashes,
                fold=fold,
                output=ensemble_root / "evaluation.json",
            )
            comparator_reports["ensemble"][fold] = e_eval
            comparator_artifacts["ensemble"][fold] = {
                "score_manifest": str(e_manifest),
                "score_manifest_sha256": e_digest,
                "evaluation": str(ensemble_root / "evaluation.json"),
                "evaluation_sha256": sha256_file(ensemble_root / "evaluation.json"),
            }
        comparator_readouts = {
            name: _pooled_readouts(reports)
            for name, reports in comparator_reports.items()
        }
        comparator_deltas = {
            "network_minus_gbdt": _paired_readouts(
                comparator_reports["network"], comparator_reports["gbdt"]
            ),
            "ensemble_minus_gbdt": _paired_readouts(
                comparator_reports["ensemble"], comparator_reports["gbdt"]
            ),
            "ensemble_minus_network": _paired_readouts(
                comparator_reports["ensemble"], comparator_reports["network"]
            ),
        }
        parent_order = {"network": 2, "gbdt": 1, "ensemble": 0}
        v2_parent = max(
            comparator_readouts,
            key=lambda name: (
                _ranking_point(
                    comparator_readouts[name]["pooled"]["primary_scaled_target_ic"]
                ),
                _ranking_point(
                    comparator_readouts[name]["pooled"]["headline_net_excess_bps"]
                ),
                parent_order[name],
            ),
        )
        stage_p = {}
        for seed in NETWORK_SEEDS:
            p_root = root / "trajectories" / "arm_B" / "stage_P" / f"seed_{seed}"
            manifest = _read_json(p_root / "run_manifest.json")
            _assert_current_clean_training(manifest, path=p_root / "run_manifest.json")
            if manifest.get("stage") != "P" or manifest.get("seed") != seed:
                raise ValueError(f"Stage-P trajectory identity differs for seed {seed}")
            history = _read_json(p_root / "history.json")
            stage_p[str(seed)] = {
                "history": str(p_root / "history.json"),
                "history_sha256": sha256_file(p_root / "history.json"),
                "raw_patience_checkpoint_sha256": sha256_file(
                    p_root / "raw_patience.pt"
                ),
                "final_ema_checkpoint_sha256": sha256_file(p_root / "final_ema.pt"),
                "selected_epoch": manifest.get("selected_epoch"),
                "stopped_epoch": manifest.get("stopped_epoch"),
                "holdout_history": history,
            }
        result = {
            "schema": ROUND2_SCHEMA,
            "status": "completed",
            **RESEARCH_FLAGS,
            **_source_tier_labels(store.manifest),
            "completed_at_utc": _utc_now(),
            "frozen_design": {
                "path": str(root / "frozen_design.json"),
                "sha256": sha256_file(root / "frozen_design.json"),
            },
            "implementation": _git_identity(),
            "store_access": access,
            "sources": source_hashes,
            "stage_p_holdout": stage_p,
            "data_span_arms": {
                "artifacts": arm_artifacts,
                "readouts": arm_readouts,
                "paired_deltas": arm_deltas,
                "eligible_by_economics": eligible,
                "long_history_small_and_uncertain": long_small_and_uncertain,
                "chosen_arm": chosen_arm,
                "preference_rule": (
                    "highest mean pooled IC unless headline economics are worse than Arm A; "
                    "if neither long-history arm beats A by more than 0.002 with intervals "
                    "spanning zero, prefer A"
                ),
            },
            "parent_comparison": {
                "artifacts": comparator_artifacts,
                "readouts": comparator_readouts,
                "paired_deltas": comparator_deltas,
                "v2_parent": v2_parent,
                "preference_rule": "best pooled IC with economics as tie-break",
            },
        }
        return write_json_atomic(root / "round2_result.json", result)
    finally:
        store.close()


def seal_root(
    *,
    root: Path,
    stdout_log: Path | None = None,
    stderr_log: Path | None = None,
    research_claim: bool = True,
) -> str:
    output = root.resolve(strict=True)
    for source, label in ((stdout_log, "stdout.log"), (stderr_log, "stderr.log")):
        if source is not None:
            destination = output / "operational_logs" / label
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(destination)
            shutil.copyfile(source.resolve(strict=True), destination)
    json_paths = [
        path
        for path in output.rglob("*.json")
        if path.name not in {"artifact_inventory.json", "access_audit.json"}
    ]
    flagged = []
    transfer_flags: list[bool] = []
    for path in json_paths:
        payload = _read_json(path)
        if "official_validation_accessed" in payload or "test_accessed" in payload:
            _assert_false_access(payload, path=path)
            flagged.append(path.relative_to(output).as_posix())
            transfer = payload["transfer_chronology_clean"]
            assert isinstance(transfer, bool)
            transfer_flags.append(transfer)
    access_flags = {
        **RESEARCH_FLAGS,
        "research_claim": research_claim,
        "transfer_chronology_clean": all(transfer_flags),
    }
    audit = {
        "schema": "BRAZIL_RV_V2_RESEARCH_ACCESS_AUDIT_V1",
        "status": "passed",
        **access_flags,
        "json_artifacts_with_access_flags": flagged,
        "json_artifact_count": len(json_paths),
        "audited_at_utc": _utc_now(),
    }
    write_json_atomic(output / "access_audit.json", audit)
    excluded = {"artifact_inventory.json", "artifact_inventory.json.sha256"}
    rows = inventory(output, exclude=excluded)
    inventory_payload = {
        "schema": "BRAZIL_RV_V2_RESEARCH_INVENTORY_V1",
        "status": "passed",
        **access_flags,
        "excluded_self": sorted(excluded),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "files": rows,
    }
    digest = write_json_atomic(output / "artifact_inventory.json", inventory_payload)
    if inventory(output, exclude=excluded) != rows:
        raise RuntimeError("research root changed while its inventory was sealed")
    verify_inventory(output, inventory(output))
    return digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run registered v2 research rounds")
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-round1")
    freeze.add_argument("--store", type=Path, required=True)
    freeze.add_argument("--cdi", type=Path, required=True)
    freeze.add_argument("--cdi-sha256", required=True)
    freeze.add_argument("--experiment52-cdi", type=Path, required=True)
    freeze.add_argument("--experiment52-cdi-sha256", required=True)
    freeze.add_argument("--development-acceptance", type=Path, required=True)
    freeze.add_argument("--development-acceptance-sha256", required=True)
    freeze.add_argument("--output-root", type=Path, required=True)
    freeze.add_argument("--num-threads", type=int, default=0)
    run = commands.add_parser("run-round1")
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--num-threads", type=int, default=0)
    resume = commands.add_parser("resume-round1")
    resume.add_argument("--output-root", type=Path, required=True)
    resume.add_argument("--num-threads", type=int, default=0)
    seal = commands.add_parser("seal-root")
    seal.add_argument("--root", type=Path, required=True)
    seal.add_argument("--stdout-log", type=Path)
    seal.add_argument("--stderr-log", type=Path)
    seal.add_argument(
        "--research-claim", action=argparse.BooleanOptionalAction, default=True
    )
    freeze2 = commands.add_parser("freeze-round2")
    freeze2.add_argument("--round1-root", type=Path, required=True)
    freeze2.add_argument("--store", type=Path, required=True)
    freeze2.add_argument("--cdi", type=Path, required=True)
    freeze2.add_argument("--cdi-sha256", required=True)
    freeze2.add_argument("--experiment52-cdi", type=Path, required=True)
    freeze2.add_argument("--experiment52-cdi-sha256", required=True)
    freeze2.add_argument("--output-root", type=Path, required=True)
    freeze2.add_argument("--fast-checkpoint", type=Path)
    freeze2.add_argument("--fast-checkpoint-sha256")
    freeze2.add_argument("--max-parallel", type=int, default=4)
    plan_p = commands.add_parser("write-round2-plan-p")
    plan_p.add_argument("--output-root", type=Path, required=True)
    plan_smoke = commands.add_parser("write-round2-plan-smoke")
    plan_smoke.add_argument("--output-root", type=Path, required=True)
    plan_main = commands.add_parser("write-round2-plan-main")
    plan_main.add_argument("--output-root", type=Path, required=True)
    finalize = commands.add_parser("finalize-round2")
    finalize.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "freeze-round1":
        digest = freeze_round1(
            store_root=arguments.store,
            cdi_path=arguments.cdi,
            cdi_sha256=arguments.cdi_sha256,
            experiment52_cdi_path=arguments.experiment52_cdi,
            experiment52_cdi_sha256=arguments.experiment52_cdi_sha256,
            acceptance_path=arguments.development_acceptance,
            acceptance_sha256=arguments.development_acceptance_sha256,
            output_root=arguments.output_root,
            num_threads=arguments.num_threads,
        )
    elif arguments.command == "run-round1":
        digest = run_round1(
            output_root=arguments.output_root,
            num_threads=arguments.num_threads,
        )
    elif arguments.command == "resume-round1":
        digest = resume_round1(
            output_root=arguments.output_root,
            num_threads=arguments.num_threads,
        )
    elif arguments.command == "freeze-round2":
        digest = freeze_round2(
            round1_root=arguments.round1_root,
            store_root=arguments.store,
            cdi_path=arguments.cdi,
            cdi_sha256=arguments.cdi_sha256,
            experiment52_cdi_path=arguments.experiment52_cdi,
            experiment52_cdi_sha256=arguments.experiment52_cdi_sha256,
            output_root=arguments.output_root,
            fast_checkpoint=arguments.fast_checkpoint,
            fast_checkpoint_sha256=arguments.fast_checkpoint_sha256,
            max_parallel=arguments.max_parallel,
        )
    elif arguments.command == "write-round2-plan-p":
        digest = write_round2_plan_p(output_root=arguments.output_root)
    elif arguments.command == "write-round2-plan-smoke":
        digest = write_round2_plan_smoke(output_root=arguments.output_root)
    elif arguments.command == "write-round2-plan-main":
        digest = write_round2_plan_main(output_root=arguments.output_root)
    elif arguments.command == "finalize-round2":
        digest = finalize_round2(output_root=arguments.output_root)
    else:
        digest = seal_root(
            root=arguments.root,
            stdout_log=arguments.stdout_log,
            stderr_log=arguments.stderr_log,
            research_claim=arguments.research_claim,
        )
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
