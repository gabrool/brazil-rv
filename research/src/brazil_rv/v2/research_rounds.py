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

from brazil_rv.execution.stateful_ledger import TERMINAL_SETTLEMENT_CONVENTION

from .artifacts import inventory, sha256_file, verify_inventory, write_json_atomic
from .baselines import BaselinePanel, build_store_baselines
from .bova11 import load_bova11_series
from .config import PROJECT_ROOT
from .contract import (
    GBDT_SEEDS,
    HORIZONS,
    PRETRAIN_END,
    PRIMARY_HORIZONS,
    REGISTERED_PRIMARY_TARGET,
    REGISTERED_PRIMARY_TARGET_MASK,
    RUN_MANY_PLAN_SCHEMA,
    SCORE_ARTIFACT_SCHEMA,
    STORE_START,
    TRAINING_STAGE_SCHEMA,
)
from .evaluate import (
    EVALUATION_SCHEMA,
    PRIOR_EVALUATION_SCHEMA,
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
from .lending_archive import LendingBorrowPanels, load_lending_borrow_panels
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
    _LEGACY_ROUND1_INVENTORY_SHA256,
    _LEGACY_ROUND1_RESULT_SHA256,
    _LEGACY_ROUND1_ROOT,
    _date_indices,
    _evaluation_inputs,
    _load_development_cdi,
    _non_ledger_report,
    _read_store_header,
    _window_target_mask,
)

PRIOR_ROUND1_SCHEMA = "BRAZIL_RV_V2_RESEARCH_ROUND1_CANONICAL_V4E"
ROUND1_SCHEMA = "BRAZIL_RV_V2_RESEARCH_ROUND1_CANONICAL_V4F"
ROUND2_SCHEMA = "BRAZIL_RV_V2_RESEARCH_ROUND2_CANONICAL_V4F"
RESEARCH_SCORE_SCHEMA = "BRAZIL_RV_V2_RESEARCH_SCORE_V4D"
PREREGISTRATION = (
    PROJECT_ROOT / "research" / "preregistrations" / "v2_round1_round2_rev4f.md"
)
BOOTSTRAP_REPLICATIONS = 10_000
BOOTSTRAP_BLOCK = 20
BOOTSTRAP_SEED = 20260815
NETWORK_SEEDS = (11, 29, 47)
_BASELINE_SIGNAL_NAMES = (
    "reversal_5",
    "reversal_21",
    "momentum_12_1",
    "reversal_5_momentum_12_1_blend",
    "inverse_volatility_20",
)
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
REGISTRATION_PROTOCOL_BEGIN = "<!-- BRAZIL_RV_V2_REV4F_PROTOCOL_JSON_BEGIN -->"
REGISTRATION_PROTOCOL_END = "<!-- BRAZIL_RV_V2_REV4F_PROTOCOL_JSON_END -->"


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
        "schema": "BRAZIL_RV_V2_REGISTRATION_PROTOCOL_V4F",
        "hedge_beta": {
            "lookback": 60,
            "minimum_pairs": 40,
            "last_endpoint": "t-1",
            "blume": [0.67, 0.33],
            "clip": [-1.0, 3.0],
            "fallback_max_age": 20,
            "fallback_default": 1.0,
        },
        "hedge_decision": "15:45; fixed quantity from prior BOVA11 close and prior NAV",
        "borrow_daily_accrual": "expm1(log1p(annual_rate)/252); fee separately",
        "cost_grid": {
            "cost_bps": [2, 4, 7],
            "borrow": ["balance", "strict", "open"],
            "construction": "headline",
        },
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
        "target_neutralization": {
            "target_value_array": REGISTERED_PRIMARY_TARGET,
            "target_validity_array": REGISTERED_PRIMARY_TARGET_MASK,
            "method": "ols_10_vol_dummies_5_beta_dummies_linear_rank_gauss_log_adv",
            "intercept": "absorbed_by_full_dummy_blocks",
            "clip": 5.0,
            "minimum_names": 20,
            "nonlinear_minimum_names": 40,
            "fallback_below_40_names": "rev3_intercept_plus_three_linear_risks",
        },
        "headline_cell": headline_ledger_protocol(),
        "headline_cell_name": "borrow_balance",
        "comparators": [
            "borrow_strict",
            "borrow_open",
            "comparator_sterile_proceeds",
            "comparator_uniform_borrow",
        ],
        "candidate_decision_rule": {
            "primary": "pooled_primary_neutral_target_ic",
            "ineligible": (
                "constructed_headline_net_excess_negative_with_95_interval_below_zero"
            ),
            "economics_override": (
                "paired_constructed_economics_interval_above_zero_and_paired_ic_"
                "interval_includes_zero"
            ),
        },
        "borrow_cells": [
            "borrow_strict",
            "borrow_balance",
            "borrow_open",
            "uniform",
        ],
        "borrow_rate": {
            "observed_rate_lookback_sessions": 60,
            "missing_rate_imputation": (
                "same_day_cross_sectional_observed_rate_75th_percentile"
            ),
            "registration_fee": {
                "fraction_of_contract_rate": 0.20,
                "annual_floor": 0.00025,
                "annual_cap": 0.0070,
            },
            "equity_rate_floor": None,
            "hedge_short_rate_floor": 0.02,
        },
        "borrow_availability": {
            "borrow_strict": "lending_trade_observed_in_prior_20_sessions",
            "borrow_balance": (
                "positive_published_open_balance_or_lending_trade_observed_in_"
                "prior_60_sessions"
            ),
            "borrow_open": "all_names_when_a_causal_cross_sectional_rate_exists",
            "pre_first_rate_borrow": "placeholder_0.02",
        },
        "lending_archive": {
            "source_label": "lending_archive_v2_2009_202412",
            "last_source_session": "2024-12-30",
            "availability_lag_sessions": 1,
            "store_lending_feature_rebuilt": False,
        },
        "acceptance_legacy_identity": {
            "root": str(_LEGACY_ROUND1_ROOT),
            "result_sha256": _LEGACY_ROUND1_RESULT_SHA256,
            "inventory_sha256": _LEGACY_ROUND1_INVENTORY_SHA256,
        },
        "construction": {
            "caps_scope": "equity_only",
            "volatility_strata": "five_equal_count_yang_zhang_vol_20_quintiles",
            "rank_within_stratum": True,
            "quota_remainder_order": [3, 2, 4, 1, 5],
            "small_stratum_scaling_threshold_multiple": 2,
            "fill_order": "within_quintile_then_global_band_spill",
            "retention_uses_current_quintile": True,
            "occupancy_mean_absolute_deviation_limit_slots": 2.0,
        },
        "realized_beta_label_threshold": 0.30,
        "inverse_volatility_neutral_ic_absolute_bound": 0.02,
        "round1": {
            "controls": list(_BASELINE_SIGNAL_NAMES),
            "gbdt_rungs": ["b_intraday"],
            "data_span_preview": False,
            "cpu_only": True,
        },
        "round2": {
            "requires_explicit_go_after_round1": True,
            "arms": ["A_fine_only", "B_pretrain_finetune"],
            "dropped_arm": "C_joint_decay_756",
            "parent_comparison_network_arm": "B_pretrain_finetune",
        },
        "bova11": {
            "isin": "BRBOVACTF003",
            "ticker": "BOVA11",
            "security_spec": "CI",
            "market_type": 10,
            "bdi_code": "14",
            "supplied_bdi_02_corrected_from_raw_cotahist": True,
            "rebalance_threshold_fraction_nav": 0.05,
            "hedge_notional_cap_nav": 0.60,
            "cost_bps_per_side": 4.0,
            "short_borrow_floor": 0.02,
        },
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
            "registered rev-3 research requires the labelled development-grade "
            "action and reconstructed schedule tiers"
        )
    return dict(DEVELOPMENT_SOURCE_TIER_LABELS)


def _store_build_implementation_commit(
    store_manifest: Mapping[str, object],
) -> str:
    metadata = store_manifest.get("metadata")
    commit = (
        metadata.get("implementation_git_commit")
        if isinstance(metadata, Mapping)
        else None
    )
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("store manifest lacks its build implementation commit")
    return commit


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
    expected_implementation: Mapping[str, object] | None,
    store_build_implementation_commit: str,
    source_tiers: Mapping[str, str],
) -> dict[str, object]:
    source = Path(path).resolve(strict=True)
    if sha256_file(source) != expected_sha256.casefold():
        raise ValueError("development acceptance report SHA-256 mismatch")
    report = _read_json(source)
    if (
        report.get("schema") != "BRAZIL_RV_V2_PIPELINE_VALIDATION_V13"
        or report.get("status") != "completed"
        or report.get("engineering_acceptance_status")
        != "development_grade_inferred_actions"
        or report.get("research_claim") is not False
    ):
        raise ValueError("development acceptance report is not an accepted rev-4c gate")
    _assert_false_access(report, path=source)
    implementation = report.get("code")
    if (
        not isinstance(implementation, Mapping)
        or not isinstance(implementation.get("commit"), str)
        or len(str(implementation["commit"])) != 40
        or implementation.get("tracked_worktree_clean") is not True
    ):
        raise ValueError("development acceptance lacks a clean implementation identity")
    if (
        expected_implementation is not None
        and implementation != expected_implementation
    ):
        raise ValueError("development acceptance implementation differs from rev-4")
    if (
        report.get("store_build_implementation_commit")
        != store_build_implementation_commit
    ):
        raise ValueError("development acceptance store-build provenance mismatch")
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


def _load_frozen_lending(
    design: Mapping[str, object],
    *,
    store_root: Path,
    dates: NDArray[np.datetime64],
) -> LendingBorrowPanels:
    record = design.get("lending_archive")
    if not isinstance(record, Mapping):
        raise ValueError("frozen design lacks the direct lending archive")
    panels = load_lending_borrow_panels(
        Path(str(record["root"])),
        expected_manifest_sha256=str(record["manifest_sha256"]),
        canonical_dates=dates.astype("datetime64[D]").astype(object).tolist(),
        canonical_isins=[
            str(value)
            for value in np.load(store_root / "isin_index.npy", allow_pickle=False)
        ],
    )
    if (
        panels.balance_sha256 != record.get("balances_sha256")
        or panels.rate_sha256 != record.get("rates_sha256")
        or panels.source_label != record.get("source_label")
    ):
        raise ValueError("lending artifacts differ from their frozen identities")
    return panels


def _evaluate(
    *,
    store: V2Store,
    indices: NDArray[np.int64],
    scores: NDArray[np.floating],
    score_mask: NDArray[np.bool_],
    cdi: NDArray[np.float64],
    bova11_close_by_index: NDArray[np.float64],
    bova11_binding: Mapping[str, str],
    lending_borrow: LendingBorrowPanels,
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
        bova11_close_by_index,
        bova11_binding,
        lending_borrow,
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
        if isinstance(row, Mapping) and row.get("scenario") == "borrow_balance"
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
    legacy_by_date: dict[str, list[float]] = {day: [] for day in ordered_dates}
    metric_rows = report.get("daily_metric_table")
    if not isinstance(metric_rows, list):
        raise ValueError("evaluation report lacks the daily metric table")
    for row in metric_rows:
        if (
            isinstance(row, Mapping)
            and row.get("horizon_sessions") in PRIMARY_HORIZONS
            and row.get("legacy_scaled_target_ic") is not None
        ):
            legacy_by_date[str(row["date"])].append(
                float(row["legacy_scaled_target_ic"])
            )
    legacy_values = np.asarray(
        [
            float(np.mean(legacy_by_date[day]))
            if len(legacy_by_date[day]) == len(PRIMARY_HORIZONS)
            else np.nan
            for day in ordered_dates
        ],
        dtype=np.float64,
    )
    return {
        "primary_neutral_target_ic": primary_values,
        "legacy_scaled_target_ic": legacy_values,
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
            "finite_bootstrap_replications": 0,
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
    finite_draws = draws[np.isfinite(draws)]
    if not finite_draws.size:
        lower: float | None = None
        upper: float | None = None
        undefined_reason: str | None = "no_finite_bootstrap_draws"
    else:
        lower = float(np.quantile(finite_draws, 0.025))
        upper = float(np.quantile(finite_draws, 0.975))
        undefined_reason = None
    return {
        "estimate": estimate,
        "lower_95": lower,
        "upper_95": upper,
        "possible_observations": possible_observations,
        "finite_observations": finite_observations,
        "finite_bootstrap_replications": int(finite_draws.size),
        "undefined_reason": undefined_reason,
        "replications": replications,
        "block_length_sessions": BOOTSTRAP_BLOCK,
        "fold_boundary_preserved": True,
    }


def _readout_point(readout: Mapping[str, object]) -> float | None:
    value = readout.get("estimate")
    if value is None:
        return None
    point = float(value)
    return point if math.isfinite(point) else None


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


def _interval_includes_zero(readout: Mapping[str, object]) -> bool:
    lower = readout.get("lower_95")
    upper = readout.get("upper_95")
    return (
        lower is not None and upper is not None and float(lower) <= 0.0 <= float(upper)
    )


def _interval_is_positive(readout: Mapping[str, object]) -> bool:
    lower = readout.get("lower_95")
    return lower is not None and float(lower) > 0.0


def _significantly_negative(readout: Mapping[str, object]) -> bool:
    upper = readout.get("upper_95")
    return upper is not None and float(upper) < 0.0


def _weighted_candidate_designation(
    reports: Mapping[str, Mapping[str, _ResearchEvaluation]],
    *,
    exact_tie_priority: Sequence[str],
) -> tuple[str | None, dict[str, object]]:
    """Apply the registered IC-first rule with a supported economics override."""

    readouts = {name: _pooled_readouts(rows) for name, rows in reports.items()}
    eligible = [
        name
        for name in reports
        if not _significantly_negative(
            readouts[name]["pooled"]["headline_net_excess_bps"]
        )
    ]
    if not eligible:
        return None, {
            "eligible_candidates": [],
            "ic_best_candidate": None,
            "economics_override_candidates": [],
            "economics_override_applied": False,
            "chosen_candidate": None,
            "rule": (
                "no candidate is designated when every constructed-book economics "
                "interval is entirely below zero"
            ),
        }
    priority = {
        name: len(exact_tie_priority) - index
        for index, name in enumerate(exact_tie_priority)
    }
    ic_best = max(
        eligible,
        key=lambda name: (
            _ranking_point(readouts[name]["pooled"]["primary_neutral_target_ic"]),
            priority.get(name, 0),
        ),
    )
    overrides: list[tuple[str, dict[str, object]]] = []
    for name in eligible:
        if name == ic_best:
            continue
        paired = _paired_readouts(reports[name], reports[ic_best])
        if _interval_is_positive(
            paired["pooled"]["headline_net_excess_bps"]
        ) and _interval_includes_zero(paired["pooled"]["primary_neutral_target_ic"]):
            overrides.append((name, paired))
    chosen = (
        ic_best
        if not overrides
        else max(
            overrides,
            key=lambda item: (
                _ranking_point(item[1]["pooled"]["headline_net_excess_bps"]),
                _ranking_point(
                    readouts[item[0]]["pooled"]["primary_neutral_target_ic"]
                ),
                priority.get(item[0], 0),
            ),
        )[0]
    )
    return chosen, {
        "eligible_candidates": eligible,
        "ic_best_candidate": ic_best,
        "economics_override_candidates": [name for name, _ in overrides],
        "economics_override_applied": chosen != ic_best,
        "chosen_candidate": chosen,
        "rule": (
            "designate the eligible pooled-neutral-IC leader unless another candidate "
            "has a positive paired constructed-book economics interval and its paired "
            "IC interval versus the leader includes zero"
        ),
    }


def _pooled_readouts(
    evaluations: Mapping[str, _ResearchEvaluation],
) -> dict[str, object]:
    folds = tuple(evaluations)
    if not folds or any(fold not in ("F1", "F2", "F3") for fold in folds):
        raise ValueError("pooled report roster must be an ordered fold subset")
    series = {
        fold: _daily_series(evaluation) for fold, evaluation in evaluations.items()
    }
    labels = tuple(series[folds[0]])
    return {
        "folds": {
            fold: {label: _folded_bootstrap((values[label],)) for label in labels}
            for fold, values in series.items()
        },
        "pooled": {
            label: _folded_bootstrap(tuple(series[fold][label] for fold in folds))
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


_ECONOMICS_DECOMPOSITION_FIELDS = (
    "gross_long_short_spread_pnl_bps",
    "hedge_gross_pnl_bps",
    "free_cash_interest_bps",
    "short_proceeds_interest_bps",
    "equity_trading_cost_bps",
    "hedge_cost_bps",
    "equity_borrow_observed_rate_bps",
    "equity_borrow_registration_fee_bps",
    "hedge_borrow_bps",
    "cdi_benchmark_bps",
)


def _scenario_daily_rows(
    evaluation: _ResearchEvaluation, scenario: str
) -> list[Mapping[str, object]]:
    economics = evaluation.result.report.get("economics")
    rows = economics.get("daily_table") if isinstance(economics, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError("evaluation lacks its economics daily table")
    selected = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("scenario") == scenario
    ]
    if len(selected) != len(evaluation.result.dates):
        raise ValueError(f"economics scenario {scenario} has an incomplete date axis")
    return selected


def _economics_row_components(row: Mapping[str, object]) -> dict[str, float]:
    equity_cost = (
        float(row["turnover_cost_bps"]) - float(row["hedge_cost_bps"])
        if row.get("turnover_cost_bps") is not None
        and row.get("hedge_cost_bps") is not None
        else math.nan
    )
    return {
        name: (
            equity_cost
            if name == "equity_trading_cost_bps"
            else math.nan
            if row.get(name) is None
            else float(row[name])
        )
        for name in _ECONOMICS_DECOMPOSITION_FIELDS
    }


def _economics_scenario_detail(
    evaluations: Mapping[str, _ResearchEvaluation], scenario: str
) -> dict[str, object]:
    by_fold: dict[str, object] = {}
    pooled_daily: list[NDArray[np.float64]] = []
    pooled_components: dict[str, list[float]] = {
        name: [] for name in _ECONOMICS_DECOMPOSITION_FIELDS
    }
    for fold, evaluation in evaluations.items():
        rows = _scenario_daily_rows(evaluation, scenario)
        net = np.asarray(
            [
                np.nan
                if row.get("net_excess_all_cash_bps") is None
                else float(row["net_excess_all_cash_bps"])
                for row in rows
            ],
            dtype=np.float64,
        )
        pooled_daily.append(net)
        component_rows = [_economics_row_components(row) for row in rows]
        for values in component_rows:
            for name, value in values.items():
                if math.isfinite(value):
                    pooled_components[name].append(value)
        components = {
            name: (float(np.mean(finite)) if finite else None)
            for name in _ECONOMICS_DECOMPOSITION_FIELDS
            for finite in [
                [
                    values[name]
                    for values in component_rows
                    if math.isfinite(values[name])
                ]
            ]
        }
        reconciliation = []
        for row, values in zip(rows, component_rows, strict=True):
            if row.get("net_excess_all_cash_bps") is None or not all(
                math.isfinite(value) for value in values.values()
            ):
                continue
            recomposed = (
                values["gross_long_short_spread_pnl_bps"]
                + values["hedge_gross_pnl_bps"]
                + values["free_cash_interest_bps"]
                + values["short_proceeds_interest_bps"]
                - values["equity_trading_cost_bps"]
                - values["hedge_cost_bps"]
                - values["equity_borrow_observed_rate_bps"]
                - values["equity_borrow_registration_fee_bps"]
                - values["hedge_borrow_bps"]
                - values["cdi_benchmark_bps"]
            )
            reconciliation.append(recomposed - float(row["net_excess_all_cash_bps"]))
        economics = evaluation.result.report["economics"]
        if not isinstance(economics, Mapping):
            raise ValueError("evaluation economics payload is malformed")
        summaries = economics.get("summaries")
        if not isinstance(summaries, list):
            raise ValueError("evaluation economics summaries are malformed")
        summary = next(
            row
            for row in summaries
            if isinstance(row, Mapping) and row.get("scenario") == scenario
        )
        readouts = _daily_series(evaluation)
        by_fold[fold] = {
            "net_excess_bps_per_day": _folded_bootstrap((net,)),
            "annualized_net_excess_sharpe": summary.get("annualized_net_excess_sharpe"),
            "mean_gross_fraction_nav": summary.get("mean_gross_fraction_nav"),
            "mean_turnover_fraction_nav": summary.get("mean_turnover_fraction_nav"),
            "average_holding_sessions_approximation": summary.get(
                "average_holding_sessions_approximation"
            ),
            "persistence_1": _folded_bootstrap((readouts["persistence_1"],)),
            "persistence_5": _folded_bootstrap((readouts["persistence_5"],)),
            "gross_label": "equity gross excludes BOVA11 hedge notional",
            "settlement_label": TERMINAL_SETTLEMENT_CONVENTION,
            "mean_component_bps_per_day": components,
            "maximum_absolute_daily_reconciliation_error_bps": max(
                (abs(value) for value in reconciliation), default=0.0
            ),
        }
    all_net = np.concatenate(pooled_daily)
    finite_net = all_net[np.isfinite(all_net)]
    standard_deviation = (
        float(np.std(finite_net, ddof=1)) if finite_net.size > 1 else 0.0
    )
    return {
        "folds": by_fold,
        "pooled": {
            "net_excess_bps_per_day": _folded_bootstrap(tuple(pooled_daily)),
            "annualized_net_excess_sharpe": (
                float(np.mean(finite_net) / standard_deviation * np.sqrt(252.0))
                if standard_deviation > 0.0
                else 0.0
            ),
            "mean_component_bps_per_day": {
                name: float(np.mean(values)) if values else None
                for name, values in pooled_components.items()
            },
            "gross_label": "equity gross excludes BOVA11 hedge notional",
            "settlement_label": TERMINAL_SETTLEMENT_CONVENTION,
        },
    }


def _round1_economics_detail(
    candidates: Mapping[str, Mapping[str, _ResearchEvaluation]],
) -> dict[str, object]:
    return {
        "schema": "BRAZIL_RV_V2_ROUND1_REV4E_ECONOMICS_DETAIL_V1",
        **RESEARCH_FLAGS,
        "headline_scenario": "borrow_balance",
        "comparator_scenario": "comparator_sterile_proceeds",
        "books": {
            name: {
                "headline": _economics_scenario_detail(evaluations, "borrow_balance"),
                "sterile_comparator": _economics_scenario_detail(
                    evaluations, "comparator_sterile_proceeds"
                ),
            }
            for name, evaluations in candidates.items()
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
    folds = tuple(candidate)
    if not folds or any(fold not in baseline for fold in folds):
        raise ValueError("paired reports require a nonempty common fold roster")
    deltas: dict[str, dict[str, NDArray[np.float64]]] = {}
    population_audit: dict[str, dict[str, list[dict[str, object]]]] = {}
    for fold in folds:
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
            "primary_neutral_target_ic": primary_delta,
            "shareholder_rank_ic": shareholder_delta,
            "price_return_rank_ic": price_delta,
            "persistence_1": persistence_1,
            "persistence_5": persistence_5,
            "shareholder_return_spread_bps_per_holding_session": spread_delta,
            "headline_net_excess_bps": economics_delta,
        }
        population_audit[fold] = {
            "primary_neutral_target_ic": primary_rows,
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
            label: _folded_bootstrap(tuple(deltas[fold][label] for fold in folds))
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
    for group in _resolved_sidecar_groups(store, RUNG_GROUPS[rung]):
        group_names = scalar_feature_names(store, (f"sidecar_{group}",))
        output.extend(gbdt_scalar_feature_names(group_names))
    if len(output) != len(set(output)):
        raise ValueError("GBDT feature names are not unique")
    return tuple(str(value) for value in output)


def _resolved_sidecar_groups(store: V2Store, groups: Sequence[str]) -> tuple[str, ...]:
    manifest_names = store.manifest.get("feature_names")
    metadata = store.manifest.get("metadata")
    capabilities = (
        metadata.get("sidecar_capabilities") if isinstance(metadata, Mapping) else None
    )
    if not isinstance(manifest_names, Mapping):
        raise ValueError("store manifest lacks ordered feature names")
    resolved: list[str] = []
    for group in groups:
        names = manifest_names.get(f"sidecar_{group}")
        if isinstance(names, list) and names:
            resolved.append(group)
            continue
        capability = (
            capabilities.get(group) if isinstance(capabilities, Mapping) else None
        )
        enabled = capability.get("enabled") if isinstance(capability, Mapping) else None
        missing = (
            capability.get("source_missing")
            if isinstance(capability, Mapping)
            else None
        )
        if enabled != [] or not isinstance(missing, list) or not missing:
            raise ValueError(
                "store lacks both materialized and explicitly source-missing "
                f"sidecar capability: {group}"
            )
    return tuple(resolved)


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
    for group in _resolved_sidecar_groups(store, RUNG_GROUPS[rung]):
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
    bova11_close_by_index: NDArray[np.float64],
    bova11_binding: Mapping[str, str],
    lending_borrow: LendingBorrowPanels,
    source_hashes: Mapping[str, str],
    root: Path,
    num_threads: int,
) -> tuple[dict[str, _ResearchEvaluation], dict[str, object]]:
    config = GBDTConfig(seeds=GBDT_SEEDS, num_threads=num_threads)
    feature_names = _feature_names(store, rung)
    reports: dict[str, _ResearchEvaluation] = {}
    records: dict[str, object] = {}
    folds = ("F1", "F2") if rung == "d_all_sidecars" else ("F1", "F2", "F3")
    if rung == "d_all_sidecars":
        records["F3"] = {
            "status": "source_unsupported",
            "source": "oddlot",
            "reason": "oddlot_archive_ends_before_F3",
        }
    for fold in folds:
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
            store.read(REGISTERED_PRIMARY_TARGET_MASK, train_indices),
            train_indices,
            target_window_indices=train_window,
        )
        train_y = np.asarray(
            store.read_target(
                REGISTERED_PRIMARY_TARGET, train_indices, valid_mask=train_mask
            )
        )
        selection_mask = _window_target_mask(
            store.read(REGISTERED_PRIMARY_TARGET_MASK, selection_indices),
            selection_indices,
        )
        selection_y = np.asarray(
            store.read_target(
                REGISTERED_PRIMARY_TARGET,
                selection_indices,
                valid_mask=selection_mask,
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
                "target_value_array": REGISTERED_PRIMARY_TARGET,
                "target_validity_array": REGISTERED_PRIMARY_TARGET_MASK,
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
            bova11_close_by_index=bova11_close_by_index,
            bova11_binding=bova11_binding,
            lending_borrow=lending_borrow,
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
    bova11_root: Path,
    bova11_manifest_sha256: str,
    hedge_beta_root: Path,
    hedge_beta_manifest_sha256: str,
    lending_archive_root: Path,
    lending_archive_manifest_sha256: str,
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
    store_build_commit = _store_build_implementation_commit(store_manifest)
    source_tiers = _source_tier_labels(store_manifest)
    acceptance_report = _verify_development_acceptance(
        acceptance_path,
        expected_sha256=acceptance_sha256,
        store_manifest_sha256=store_manifest_sha256,
        expected_implementation=None,
        store_build_implementation_commit=store_build_commit,
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

    bova11 = load_bova11_series(
        bova11_root,
        expected_manifest_sha256=bova11_manifest_sha256,
        canonical_dates=dates.astype("datetime64[D]").astype(object).tolist(),
    )
    lending_borrow = load_lending_borrow_panels(
        lending_archive_root,
        expected_manifest_sha256=lending_archive_manifest_sha256,
        canonical_dates=dates.astype("datetime64[D]").astype(object).tolist(),
        canonical_isins=[
            str(value)
            for value in np.load(store / "isin_index.npy", allow_pickle=False)
        ],
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
            "build_implementation_commit": store_build_commit,
        },
        "development_acceptance": {
            "path": str(acceptance_path.resolve(strict=True)),
            "sha256": acceptance_sha256.casefold(),
            "status": "development_grade_inferred_actions",
            "implementation": dict(acceptance_report["code"]),
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
        "bova11": {
            "root": str(Path(bova11_root).resolve(strict=True)),
            "hedge_beta_root": str(hedge_beta_root.resolve(strict=True)),
            "hedge_beta_manifest_sha256": hedge_beta_manifest_sha256,
            "manifest_sha256": bova11.manifest_sha256,
            "data_sha256": bova11.data_sha256,
        },
        "lending_archive": {
            "root": str(Path(lending_archive_root).resolve(strict=True)),
            "manifest_sha256": lending_borrow.manifest_sha256,
            "balances_sha256": lending_borrow.balance_sha256,
            "rates_sha256": lending_borrow.rate_sha256,
            "source_label": lending_borrow.source_label,
            "source_unavailable_dates": [
                value.isoformat() for value in lending_borrow.source_unavailable_dates
            ],
            "source_placeholder_dates": [
                value.isoformat() for value in lending_borrow.source_placeholder_dates
            ],
        },
        "folds": folds,
        "baseline_roster": list(_BASELINE_SIGNAL_NAMES),
        "gbdt_rungs": {"b_intraday": []},
        "gbdt_seeds": list(GBDT_SEEDS),
        "data_span_arms": [],
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
            "ineligible_hold_sessions": 5,
            "settlement_grace_sessions": 10,
            "settlement_haircut": 0.30,
            "settlement_economics_unresolved_fraction_nav": 0.15,
            "headline_uses_executable_borrow": True,
            "direct_lending_archive_readout_present": True,
            "volatility_balanced_entries": True,
            "beta_hedge": True,
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
    store_build_commit = _store_build_implementation_commit(store_manifest)
    if design["store"].get("build_implementation_commit") != store_build_commit:
        raise ValueError("Round-1 frozen store-build provenance mismatch")
    source_tiers = _source_tier_labels(store_manifest)
    if any(design.get(key) != value for key, value in source_tiers.items()):
        raise ValueError("Round-1 frozen source tiers differ from the store")
    acceptance = design.get("development_acceptance")
    if not isinstance(acceptance, Mapping):
        raise ValueError("Round-1 frozen design lacks development acceptance")
    acceptance_implementation = acceptance.get("implementation")
    if not isinstance(acceptance_implementation, Mapping):
        raise ValueError("Round-1 frozen design lacks acceptance implementation")
    _verify_development_acceptance(
        Path(str(acceptance["path"])),
        expected_sha256=str(acceptance["sha256"]),
        store_manifest_sha256=str(design["store"]["manifest_sha256"]),
        expected_implementation=acceptance_implementation,
        store_build_implementation_commit=store_build_commit,
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
    bova_design = design["bova11"]
    bova11 = load_bova11_series(
        Path(str(bova_design["root"])),
        expected_manifest_sha256=str(bova_design["manifest_sha256"]),
        canonical_dates=dates.astype("datetime64[D]").astype(object).tolist(),
    )
    if bova11.data_sha256 != bova_design["data_sha256"]:
        raise ValueError("Round-1 BOVA11 data hash differs from the frozen design")
    bova11_binding = {
        **bova_design,
        "manifest_sha256": bova11.manifest_sha256,
        "data_sha256": bova11.data_sha256,
    }

    lending_design = design.get("lending_archive")
    if not isinstance(lending_design, Mapping):
        raise ValueError("Round-1 frozen design lacks the lending archive")
    lending_borrow = load_lending_borrow_panels(
        Path(str(lending_design["root"])),
        expected_manifest_sha256=str(lending_design["manifest_sha256"]),
        canonical_dates=dates.astype("datetime64[D]").astype(object).tolist(),
        canonical_isins=[
            str(value)
            for value in np.load(store_root / "isin_index.npy", allow_pickle=False)
        ],
    )
    if (
        lending_borrow.balance_sha256 != lending_design.get("balances_sha256")
        or lending_borrow.rate_sha256 != lending_design.get("rates_sha256")
        or lending_borrow.source_label != lending_design.get("source_label")
    ):
        raise ValueError("Round-1 lending artifacts differ from the frozen design")
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
        "bova11_manifest": bova11.manifest_sha256,
        "bova11_data": bova11.data_sha256,
        "lending_archive_manifest": lending_borrow.manifest_sha256,
        "lending_archive_balances": lending_borrow.balance_sha256,
        "lending_archive_rates": lending_borrow.rate_sha256,
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
                    bova11_close_by_index=bova11.close_by_session,
                    bova11_binding=bova11_binding,
                    lending_borrow=lending_borrow,
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
        for rung in ("b_intraday",):
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
                bova11_close_by_index=bova11.close_by_session,
                bova11_binding=bova11_binding,
                lending_borrow=lending_borrow,
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
                    _point_is_negative(pooled["primary_neutral_target_ic"])
                    and _point_is_negative(pooled["headline_net_excess_bps"])
                ):
                    kept.append(rung)
            previous = rung
            events.append({"event": f"{rung}_completed", "at_utc": _utc_now()})
        parent = max(
            kept,
            key=lambda rung: (
                _ranking_point(
                    rung_summaries[rung]["pooled"]["primary_neutral_target_ic"]
                ),
                _ranking_point(
                    rung_summaries[rung]["pooled"]["headline_net_excess_bps"]
                ),
                -list(RUNG_GROUPS).index(rung),
            ),
        )

        designated_rung, rung_designation = _weighted_candidate_designation(
            {parent: rung_reports[parent]},
            exact_tie_priority=(parent,),
        )

        span_records: dict[str, object] = {}
        span_summaries: dict[str, object] = {}
        span_comparisons: dict[str, object] = {}
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
                "designated_rung": designated_rung,
                "designation": rung_designation,
                "preference_rule": rung_designation["rule"],
            },
            "gbdt_data_span_preview": {
                "status": "not_rerun_by_rev4_reduced_roster",
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
    store_build_commit = _store_build_implementation_commit(store_manifest)
    if design["store"].get("build_implementation_commit") != store_build_commit:
        raise ValueError("Round-1 frozen store-build provenance mismatch")
    source_tiers = _source_tier_labels(store_manifest)
    if any(design.get(key) != value for key, value in source_tiers.items()):
        raise ValueError("Round-1 frozen source tiers differ from the store")
    acceptance = design.get("development_acceptance")
    if not isinstance(acceptance, Mapping):
        raise ValueError("Round-1 frozen design lacks development acceptance")
    acceptance_implementation = acceptance.get("implementation")
    if not isinstance(acceptance_implementation, Mapping):
        raise ValueError("Round-1 frozen design lacks acceptance implementation")
    _verify_development_acceptance(
        Path(str(acceptance["path"])),
        expected_sha256=str(acceptance["sha256"]),
        store_manifest_sha256=str(design["store"]["manifest_sha256"]),
        expected_implementation=acceptance_implementation,
        store_build_implementation_commit=store_build_commit,
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
    bova_design = design["bova11"]
    bova11 = load_bova11_series(
        Path(str(bova_design["root"])),
        expected_manifest_sha256=str(bova_design["manifest_sha256"]),
        canonical_dates=dates.astype("datetime64[D]").astype(object).tolist(),
    )
    if bova11.data_sha256 != bova_design["data_sha256"]:
        raise ValueError("Round-1 BOVA11 data hash differs from frozen design")
    bova11_binding = {
        **bova_design,
        "manifest_sha256": bova11.manifest_sha256,
        "data_sha256": bova11.data_sha256,
    }
    lending_borrow = _load_frozen_lending(design, store_root=store_root, dates=dates)
    source_hashes["bova11_manifest"] = bova11.manifest_sha256
    source_hashes["bova11_data"] = bova11.data_sha256
    source_hashes["lending_archive_manifest"] = lending_borrow.manifest_sha256
    source_hashes["lending_archive_balances"] = lending_borrow.balance_sha256
    source_hashes["lending_archive_rates"] = lending_borrow.rate_sha256
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
                bova11_close_by_index=bova11.close_by_session,
                bova11_binding=bova11_binding,
                lending_borrow=lending_borrow,
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
    for rung in design["gbdt_rungs"]:
        candidate_root = output / "gbdt_ladder" / rung
        candidate_folds = (
            ("F1", "F2") if rung == "d_all_sidecars" else ("F1", "F2", "F3")
        )
        completed = tuple(
            (candidate_root / fold / "evaluation.json").is_file()
            for fold in candidate_folds
        )
        if all(completed):
            reports = {}
            records = {}
            for fold in candidate_folds:
                root = candidate_root / fold
                reports[fold] = _evaluation_from_artifacts(
                    root / "evaluation.json",
                    store=store,
                    indices=evaluation[fold],
                    cdi=cdi,
                    bova11_close_by_index=bova11.close_by_session,
                    bova11_binding=bova11_binding,
                    lending_borrow=lending_borrow,
                )
                records[fold] = _existing_score_and_evaluation_record(root)
            if rung == "d_all_sidecars":
                records["F3"] = {
                    "status": "source_unsupported",
                    "source": "oddlot",
                    "reason": "oddlot_archive_ends_before_F3",
                }
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
                bova11_close_by_index=bova11.close_by_session,
                bova11_binding=bova11_binding,
                lending_borrow=lending_borrow,
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
                _point_is_negative(pooled["primary_neutral_target_ic"])
                and _point_is_negative(pooled["headline_net_excess_bps"])
            ):
                kept.append(rung)
        previous = rung
    parent = max(
        kept,
        key=lambda rung: (
            _ranking_point(rung_summaries[rung]["pooled"]["primary_neutral_target_ic"]),
            _ranking_point(rung_summaries[rung]["pooled"]["headline_net_excess_bps"]),
            -list(RUNG_GROUPS).index(rung),
        ),
    )
    designated_rung, rung_designation = _weighted_candidate_designation(
        {parent: rung_reports[parent]},
        exact_tie_priority=(parent,),
    )

    span_reports: dict[str, dict[str, _ResearchEvaluation]] = {}
    span_records: dict[str, object] = {}
    for arm in design["data_span_arms"]:
        candidate_root = output / "gbdt_data_span" / arm
        candidate_folds = (
            ("F1", "F2") if parent == "d_all_sidecars" else ("F1", "F2", "F3")
        )
        completed = tuple(
            (candidate_root / fold / "evaluation.json").is_file()
            for fold in candidate_folds
        )
        if all(completed):
            reports = {}
            records = {}
            for fold in candidate_folds:
                root = candidate_root / fold
                reports[fold] = _evaluation_from_artifacts(
                    root / "evaluation.json",
                    store=store,
                    indices=evaluation[fold],
                    cdi=cdi,
                    bova11_close_by_index=bova11.close_by_session,
                    bova11_binding=bova11_binding,
                    lending_borrow=lending_borrow,
                )
                records[fold] = _existing_score_and_evaluation_record(root)
            if parent == "d_all_sidecars":
                records["F3"] = {
                    "status": "source_unsupported",
                    "source": "oddlot",
                    "reason": "oddlot_archive_ends_before_F3",
                }
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
                bova11_close_by_index=bova11.close_by_session,
                bova11_binding=bova11_binding,
                lending_borrow=lending_borrow,
                source_hashes=source_hashes,
                root=candidate_root,
                num_threads=num_threads,
            )
            scored_candidates.append(f"gbdt_data_span/{arm}")
        span_reports[arm] = reports
        span_records[arm] = records
    span_summaries: dict[str, object] = {
        arm: _pooled_readouts(reports) for arm, reports in span_reports.items()
    }
    span_comparisons = {
        f"{arm}_minus_fine_only": _paired_readouts(reports, rung_reports[parent])
        for arm, reports in span_reports.items()
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
            "designated_rung": designated_rung,
            "designation": rung_designation,
            "preference_rule": rung_designation["rule"],
        },
        "gbdt_data_span_preview": {
            "status": "not_rerun_by_rev4_reduced_roster",
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


def _ledger_replay_non_ledger_projection(
    report: Mapping[str, object],
) -> dict[str, object]:
    projection = _non_ledger_report(report)
    diagnostics = projection.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("evaluation lacks its diagnostics payload")
    projection["diagnostics"] = {
        key: value for key, value in diagnostics.items() if key != "realized_beta"
    }
    return projection


def freeze_round1_ledger_replay(
    *,
    source_round1_root: Path,
    output_root: Path,
    hedge_beta_root: Path,
    hedge_beta_manifest_sha256: str,
) -> str:
    """Bind a new economic-beta sidecar to sealed rev4e scores before replay."""

    protocol = verify_registration_protocol()
    code = _git_identity()
    source = source_round1_root.resolve(strict=True)
    source_result = _verify_sealed_root(source, expected_schema=PRIOR_ROUND1_SCHEMA)
    if source_result.get("gbdt_ladder", {}).get("parent_rung") != "b_intraday":
        raise ValueError("rev4f replay requires the sealed b_intraday parent")
    source_design_path = source / "frozen_design.json"
    source_design = _read_json(source_design_path)
    if source_design.get("schema") != PRIOR_ROUND1_SCHEMA:
        raise ValueError("rev4f replay source is not the sealed rev4e design")
    output = output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    design = {
        "schema": ROUND1_SCHEMA,
        "status": "frozen_before_score",
        **RESEARCH_FLAGS,
        "frozen_at_utc": _utc_now(),
        "scope": "ledger_only_replay_of_hash_bound_rev4e_score_panels",
        "implementation": code,
        "preregistration": {
            "path": str(PREREGISTRATION.resolve(strict=True)),
            "sha256": sha256_file(PREREGISTRATION),
            "protocol": protocol,
        },
        "prior_round1": {
            "root": str(source),
            "schema": PRIOR_ROUND1_SCHEMA,
            "result_sha256": sha256_file(source / "round1_result.json"),
            "inventory_sha256": sha256_file(source / "artifact_inventory.json"),
            "frozen_design_sha256": sha256_file(source_design_path),
        },
        "store": source_design["store"],
        "development_acceptance": source_design["development_acceptance"],
        "cdi": source_design["cdi"],
        "bova11": {
            **source_design["bova11"],
            "hedge_beta_root": str(hedge_beta_root.resolve(strict=True)),
            "hedge_beta_manifest_sha256": hedge_beta_manifest_sha256,
        },
        "lending_archive": source_design["lending_archive"],
        "folds": source_design["folds"],
        "baseline_roster": list(_BASELINE_SIGNAL_NAMES),
        "gbdt_rungs": {"b_intraday": []},
        "gbdt_seeds": list(GBDT_SEEDS),
        "bootstrap": source_design["bootstrap"],
        "ledger_change": {
            "economic_beta": protocol["hedge_beta"],
            "hedge_decision": protocol["hedge_decision"],
            "borrow_daily_accrual": protocol["borrow_daily_accrual"],
            "cost_grid": protocol["cost_grid"],
            "pending_entry_retention": True,
        },
    }
    return write_json_atomic(output / "frozen_design.json", design)


def run_round1_ledger_replay(*, output_root: Path) -> str:
    """Re-evaluate only the ledger over the sealed rev4e Round-1 score panels."""

    output = output_root.resolve(strict=True)
    design_path = output / "frozen_design.json"
    design = _read_json(design_path)
    if (
        design.get("schema") != ROUND1_SCHEMA
        or design.get("status") != "frozen_before_score"
        or design.get("scope") != "ledger_only_replay_of_hash_bound_rev4e_score_panels"
    ):
        raise ValueError("Round-1 rev4f root is not its frozen ledger replay")
    code = _git_identity()
    if design.get("implementation") != code:
        raise ValueError("Round-1 rev4f implementation differs from the freeze")
    result_path = output / "round1_result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    source_binding = design.get("prior_round1")
    if not isinstance(source_binding, Mapping):
        raise ValueError("Round-1 rev4f lacks its prior-root binding")
    source = Path(str(source_binding["root"])).resolve(strict=True)
    source_result = _verify_sealed_root(source, expected_schema=PRIOR_ROUND1_SCHEMA)
    for name, path in (
        ("result_sha256", source / "round1_result.json"),
        ("inventory_sha256", source / "artifact_inventory.json"),
        ("frozen_design_sha256", source / "frozen_design.json"),
    ):
        if sha256_file(path) != source_binding.get(name):
            raise ValueError(f"sealed rev4e {name} differs from the frozen binding")
    store_root = Path(str(design["store"]["root"])).resolve(strict=True)
    store_manifest, dates = _read_store_header(store_root)
    if sha256_file(store_root / "manifest.json") != design["store"]["manifest_sha256"]:
        raise ValueError("Round-1 rev4f store manifest hash mismatch")
    source_tiers = _source_tier_labels(store_manifest)
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
    bova_design = design["bova11"]
    bova11 = load_bova11_series(
        Path(str(bova_design["root"])),
        expected_manifest_sha256=str(bova_design["manifest_sha256"]),
        canonical_dates=dates.astype("datetime64[D]").astype(object).tolist(),
    )
    if bova11.data_sha256 != bova_design["data_sha256"]:
        raise ValueError("Round-1 rev4f BOVA11 data hash mismatch")
    bova11_binding = {
        **bova_design,
        "manifest_sha256": bova11.manifest_sha256,
        "data_sha256": bova11.data_sha256,
    }
    lending_borrow = _load_frozen_lending(design, store_root=store_root, dates=dates)
    store, access = _open_round_store(
        store_root, fit, selection, evaluation, fit_target_window, pretrain
    )
    shutil.copytree(source / "baselines", output / "baselines")
    shutil.copytree(source / "gbdt_ladder", output / "gbdt_ladder")
    comparison_rows: list[dict[str, object]] = []

    def replay_one(
        source_path: Path, destination_path: Path, fold: str
    ) -> _ResearchEvaluation:
        retained = _evaluation_from_artifacts(
            source_path,
            store=store,
            indices=evaluation[fold],
            cdi=cdi,
            bova11_close_by_index=bova11.close_by_session,
            bova11_binding=bova11_binding,
            lending_borrow=lending_borrow,
            expected_fold=fold,
            expected_evaluation_schema=PRIOR_EVALUATION_SCHEMA,
        )
        replayed = evaluate_scores(retained.inputs, window_name=fold)
        for key in ("research_claim", "deployment_changed"):
            if key in retained.result.report:
                replayed.report[key] = retained.result.report[key]
        old_projection = _ledger_replay_non_ledger_projection(retained.result.report)
        new_projection = _ledger_replay_non_ledger_projection(replayed.report)
        if old_projection != new_projection:
            raise RuntimeError(
                f"non-ledger evaluation fields changed in rev4f replay: {source_path}"
            )
        new_sha = write_json_atomic(destination_path, replayed.report)
        comparison_rows.append(
            {
                "source": str(source_path),
                "source_sha256": sha256_file(source_path),
                "replayed": str(destination_path),
                "replayed_sha256": new_sha,
                "score_or_model_recomputed": False,
                "non_ledger_fields_bit_identical": True,
            }
        )
        return _ResearchEvaluation(result=replayed, inputs=retained.inputs)

    baseline_reports: dict[str, dict[str, _ResearchEvaluation]] = {}
    baseline_records: dict[str, dict[str, object]] = {}
    rung_reports: dict[str, dict[str, _ResearchEvaluation]] = {"b_intraday": {}}
    rung_records: dict[str, dict[str, object]] = {"b_intraday": {}}
    try:
        for name in _BASELINE_SIGNAL_NAMES:
            baseline_reports[name] = {}
            baseline_records[name] = {}
            for fold in ("F1", "F2", "F3"):
                source_fold = source / "baselines" / name / fold
                destination_fold = output / "baselines" / name / fold
                replayed = replay_one(
                    source_fold / "evaluation.json",
                    destination_fold / "evaluation.json",
                    fold,
                )
                baseline_reports[name][fold] = replayed
                baseline_records[name][fold] = {
                    "score_manifest": str(destination_fold / "score_manifest.json"),
                    "score_manifest_sha256": sha256_file(
                        destination_fold / "score_manifest.json"
                    ),
                    "evaluation": str(destination_fold / "evaluation.json"),
                    "evaluation_sha256": sha256_file(
                        destination_fold / "evaluation.json"
                    ),
                    "score_reused_without_recomputation": True,
                }
        for fold in ("F1", "F2", "F3"):
            source_fold = source / "gbdt_ladder" / "b_intraday" / fold
            destination_fold = output / "gbdt_ladder" / "b_intraday" / fold
            replayed = replay_one(
                source_fold / "evaluation.json",
                destination_fold / "evaluation.json",
                fold,
            )
            rung_reports["b_intraday"][fold] = replayed
            rung_records["b_intraday"][fold] = {
                "score_manifest": str(destination_fold / "score_manifest.json"),
                "score_manifest_sha256": sha256_file(
                    destination_fold / "score_manifest.json"
                ),
                "evaluation": str(destination_fold / "evaluation.json"),
                "evaluation_sha256": sha256_file(destination_fold / "evaluation.json"),
                "score_reused_without_recomputation": True,
            }
        baseline_summary = {
            name: _pooled_readouts(reports)
            for name, reports in baseline_reports.items()
        }
        rung_summary = {"b_intraday": _pooled_readouts(rung_reports["b_intraday"])}
        designated, designation = _weighted_candidate_designation(
            rung_reports, exact_tie_priority=("b_intraday",)
        )
        if designated not in {"b_intraday", None}:
            raise RuntimeError("rev4f produced an impossible rung designation")
        economics_detail = _round1_economics_detail(
            {**baseline_reports, "b_intraday": rung_reports["b_intraday"]}
        )
        detail_path = output / "round1_rev4f_economics_detail.json"
        detail_sha = write_json_atomic(detail_path, economics_detail)
        source_hashes = {
            "v2_store_manifest": str(design["store"]["manifest_sha256"]),
            "cdi_development_extension": str(
                cdi_provenance["development_extension"]["sha256"]
            ),
            "cdi_experiment52_reference": str(
                cdi_provenance["experiment52_reference"]["sha256"]
            ),
            "bova11_manifest": bova11.manifest_sha256,
            "bova11_data": bova11.data_sha256,
            "lending_archive_manifest": lending_borrow.manifest_sha256,
            "lending_archive_balances": lending_borrow.balance_sha256,
            "lending_archive_rates": lending_borrow.rate_sha256,
            "preregistration": str(design["preregistration"]["sha256"]),
            "prior_round1_result": str(source_binding["result_sha256"]),
            "prior_round1_inventory": str(source_binding["inventory_sha256"]),
        }
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
            "score_implementation": source_result["implementation"],
            "store_access": access,
            "sources": source_hashes,
            "ledger_replay": {
                "source_root": str(source),
                "score_or_model_recomputation": False,
                "evaluation_count": len(comparison_rows),
                "all_non_ledger_fields_bit_identical": all(
                    row["non_ledger_fields_bit_identical"] is True
                    for row in comparison_rows
                ),
                "comparisons": comparison_rows,
            },
            "economics_detail": {
                "path": str(detail_path),
                "sha256": detail_sha,
            },
            "baselines": {
                "artifacts": baseline_records,
                "readouts": baseline_summary,
            },
            "gbdt_ladder": {
                "artifacts": rung_records,
                "readouts": rung_summary,
                "paired_deltas": {},
                "kept_rungs": ["b_intraday"],
                "parent_rung": "b_intraday",
                "designated_rung": designated,
                "designation": designation,
                "preference_rule": designation["rule"],
            },
            "gbdt_data_span_preview": {
                "status": "not_rerun_by_rev4f_ledger_only_replay",
                "artifacts": {},
                "readouts": {},
                "paired_deltas": {},
                "decision_weight": "informational_only",
            },
            "operational_events": [
                {
                    "event": "rev4f_ledger_only_replay_completed",
                    "at_utc": _utc_now(),
                }
            ],
        }
        return write_json_atomic(result_path, result)
    finally:
        store.close()


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
        if expected_schema in {ROUND1_SCHEMA, PRIOR_ROUND1_SCHEMA}
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
    bova11_root: Path,
    bova11_manifest_sha256: str,
    hedge_beta_root: Path,
    hedge_beta_manifest_sha256: str,
    lending_archive_root: Path,
    lending_archive_manifest_sha256: str,
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
    round1_implementation = round1.get("implementation")
    if (
        not isinstance(round1_implementation, Mapping)
        or not isinstance(round1_implementation.get("commit"), str)
        or len(round1_implementation["commit"]) != 40
        or round1_implementation.get("tracked_worktree_clean") is not True
    ):
        raise ValueError("sealed Round 1 lacks a clean commit-bound implementation")
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
    bova11 = load_bova11_series(
        bova11_root,
        expected_manifest_sha256=bova11_manifest_sha256,
        canonical_dates=dates.astype("datetime64[D]").astype(object).tolist(),
    )
    lending_borrow = load_lending_borrow_panels(
        lending_archive_root,
        expected_manifest_sha256=lending_archive_manifest_sha256,
        canonical_dates=dates.astype("datetime64[D]").astype(object).tolist(),
        canonical_isins=[
            str(value)
            for value in np.load(store / "isin_index.npy", allow_pickle=False)
        ],
    )
    round1_sources = round1.get("sources")
    if not isinstance(round1_sources, Mapping) or (
        round1_sources.get("lending_archive_manifest") != lending_borrow.manifest_sha256
    ):
        raise ValueError("Round 2 must use the exact Round-1 lending archive")
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
            "implementation": dict(round1_implementation),
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
        "bova11": {
            "root": str(Path(bova11_root).resolve(strict=True)),
            "hedge_beta_root": str(hedge_beta_root.resolve(strict=True)),
            "hedge_beta_manifest_sha256": hedge_beta_manifest_sha256,
            "manifest_sha256": bova11.manifest_sha256,
            "data_sha256": bova11.data_sha256,
        },
        "lending_archive": {
            "root": str(Path(lending_archive_root).resolve(strict=True)),
            "manifest_sha256": lending_borrow.manifest_sha256,
            "balances_sha256": lending_borrow.balance_sha256,
            "rates_sha256": lending_borrow.rate_sha256,
            "source_label": lending_borrow.source_label,
        },
        "enabled_sidecars": list(RUNG_GROUPS[parent]),
        "fast_initialization": fast,
        "network": {
            "seeds": list(NETWORK_SEEDS),
            "folds": ["F1", "F2", "F3"],
            "arms": ["A_fine_only", "B_pretrain_finetune"],
            "dropped_arm": "C_joint_decay_756",
            "parent_comparison_network_arm": "B_pretrain_finetune",
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
    graph_count = 3 if stage == "J" else 2
    compiled_graphs = {
        "training": 2 if stage == "J" else 1,
        "selection": 1,
        "total": graph_count,
    }
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
            "compiled_graph_count": graph_count,
            "compiled_graphs": compiled_graphs,
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
            "phase": "rev4_smoke",
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
        or smoke.get("compiled_graph_count") != 2
        or smoke.get("compiled_graphs") != {"training": 1, "selection": 1, "total": 2}
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
            "phase": "rev4_stage_P",
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
            "phase": "rev4_registered_arms_A_and_B",
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


def _evaluation_from_path(
    path: Path, *, expected_schema: str = EVALUATION_SCHEMA
) -> dict[str, object]:
    payload = _read_json(path)
    _assert_false_access(payload, path=path)
    if payload.get("schema") != expected_schema:
        raise ValueError(f"not a v2 evaluation: {path}")
    return payload


def _evaluation_from_artifacts(
    path: Path,
    *,
    store: V2Store,
    indices: NDArray[np.int64],
    cdi: NDArray[np.float64],
    bova11_close_by_index: NDArray[np.float64],
    bova11_binding: Mapping[str, str],
    lending_borrow: LendingBorrowPanels,
    allow_legacy_missing_indices: bool = False,
    expected_fold: str | None = None,
    expected_evaluation_schema: str = EVALUATION_SCHEMA,
) -> _ResearchEvaluation:
    """Rebuild retained comparison inputs from hash-bound score/store artifacts."""

    report = _evaluation_from_path(path, expected_schema=expected_evaluation_schema)
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
    if not isinstance(metadata, Mapping):
        raise ValueError(f"score artifact lacks evaluation metadata: {path}")
    recorded_indices = metadata.get("evaluation_date_indices")
    if recorded_indices is None and not allow_legacy_missing_indices:
        raise ValueError(f"score artifact axis differs from the evaluation: {path}")
    if recorded_indices is not None and recorded_indices != indices.tolist():
        raise ValueError(f"score artifact axis differs from the evaluation: {path}")
    if expected_fold is not None and metadata.get("fold") != expected_fold:
        raise ValueError(f"score artifact fold differs from the evaluation: {path}")
    source_hashes = report.get("source_artifact_hashes")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ValueError(f"evaluation lacks source artifact hashes: {path}")
    inputs = _evaluation_inputs(
        store,
        indices,
        scores,
        score_mask,
        cdi,
        bova11_close_by_index,
        bova11_binding,
        lending_borrow,
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
        if isinstance(row, Mapping) and row.get("scenario") == "borrow_balance"
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


def _round2_arm_decision(
    arm_reports: Mapping[str, Mapping[str, _ResearchEvaluation]],
) -> tuple[dict[str, object], dict[str, object], list[str], bool, str | None]:
    if tuple(arm_reports) != ("arm_A", "arm_B"):
        raise ValueError("rev-4 Round 2 requires exactly Arms A and B")
    readouts = {arm: _pooled_readouts(reports) for arm, reports in arm_reports.items()}
    deltas = {
        "B_minus_A": _paired_readouts(arm_reports["arm_B"], arm_reports["arm_A"]),
    }
    chosen, designation = _weighted_candidate_designation(
        arm_reports, exact_tie_priority=("arm_A", "arm_B")
    )
    return (
        readouts,
        deltas,
        list(designation["eligible_candidates"]),
        bool(designation["economics_override_applied"]),
        chosen,
    )


def _round2_stage_p(root: Path) -> dict[str, object]:
    stage_p: dict[str, object] = {}
    for seed in NETWORK_SEEDS:
        p_root = root / "trajectories" / "arm_B" / "stage_P" / f"seed_{seed}"
        manifest = _read_json(p_root / "run_manifest.json")
        _assert_current_clean_training(manifest, path=p_root / "run_manifest.json")
        if manifest.get("stage") != "P" or manifest.get("seed") != seed:
            raise ValueError(f"Stage-P trajectory identity differs for seed {seed}")
        history_path = p_root / "history.json"
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if not isinstance(history, list):
            raise ValueError(f"Stage-P history must be a JSON array: {history_path}")
        stage_p[str(seed)] = {
            "history": str(history_path),
            "history_sha256": sha256_file(history_path),
            "raw_patience_checkpoint_sha256": sha256_file(p_root / "raw_patience.pt"),
            "final_ema_checkpoint_sha256": sha256_file(p_root / "final_ema.pt"),
            "selected_epoch": manifest.get("selected_epoch"),
            "stopped_epoch": manifest.get("stopped_epoch"),
            "holdout_history": history,
        }
    return stage_p


def _round2_result(
    *,
    root: Path,
    design: Mapping[str, object],
    store: V2Store,
    access: Mapping[str, object],
    source_hashes: Mapping[str, str],
    arm_reports: Mapping[str, Mapping[str, _ResearchEvaluation]],
    arm_artifacts: Mapping[str, object],
    comparator_reports: Mapping[str, Mapping[str, _ResearchEvaluation]],
    comparator_artifacts: Mapping[str, object],
    reporting_recovery: Mapping[str, object] | None = None,
) -> str:
    arm_readouts, arm_deltas, eligible, uncertain, chosen_arm = _round2_arm_decision(
        arm_reports
    )
    if comparator_reports["network"] is not arm_reports["arm_B"]:
        raise ValueError("rev-4 parent comparison must use Arm B")
    comparator_readouts = {
        name: _pooled_readouts(reports) for name, reports in comparator_reports.items()
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
    v2_parent, parent_designation = _weighted_candidate_designation(
        comparator_reports,
        exact_tie_priority=("network", "gbdt", "ensemble"),
    )
    implementation = _git_identity()
    result: dict[str, object] = {
        "schema": ROUND2_SCHEMA,
        "status": "completed",
        **RESEARCH_FLAGS,
        **_source_tier_labels(store.manifest),
        "completed_at_utc": _utc_now(),
        "frozen_design": {
            "path": str(root / "frozen_design.json"),
            "sha256": sha256_file(root / "frozen_design.json"),
        },
        "implementation": implementation,
        "store_access": dict(access),
        "sources": dict(source_hashes),
        "stage_p_holdout": _round2_stage_p(root),
        "data_span_arms": {
            "artifacts": dict(arm_artifacts),
            "readouts": arm_readouts,
            "paired_deltas": arm_deltas,
            "eligible_by_economics": eligible,
            "economics_override_applied": uncertain,
            "chosen_arm": chosen_arm,
            "dropped_arm": "arm_C",
            "preference_rule": _weighted_candidate_designation.__doc__,
        },
        "parent_comparison": {
            "artifacts": dict(comparator_artifacts),
            "readouts": comparator_readouts,
            "paired_deltas": comparator_deltas,
            "v2_parent": v2_parent,
            "designation": parent_designation,
        },
    }
    if reporting_recovery is not None:
        result["score_implementation"] = design["implementation"]
        result["reporting_recovery"] = dict(reporting_recovery)
        result["operational_events"] = [
            {
                "event": "round2_reporting_recovered_from_completed_artifacts",
                "at_utc": _utc_now(),
            }
        ]
    return write_json_atomic(root / "round2_result.json", result)


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
    bova_design = design["bova11"]
    bova11 = load_bova11_series(
        Path(str(bova_design["root"])),
        expected_manifest_sha256=str(bova_design["manifest_sha256"]),
        canonical_dates=dates.astype("datetime64[D]").astype(object).tolist(),
    )
    if bova11.data_sha256 != bova_design["data_sha256"]:
        raise ValueError("Round-2 BOVA11 data hash differs from the frozen design")
    bova11_binding = {
        **bova_design,
        "manifest_sha256": bova11.manifest_sha256,
        "data_sha256": bova11.data_sha256,
    }
    lending_borrow = _load_frozen_lending(design, store_root=store_root, dates=dates)
    source_hashes = {
        "v2_store_manifest": str(design["store"]["manifest_sha256"]),
        "cdi_development_extension": str(provenance["development_extension"]["sha256"]),
        "cdi_experiment52_reference": str(
            provenance["experiment52_reference"]["sha256"]
        ),
        "bova11_manifest": bova11.manifest_sha256,
        "bova11_data": bova11.data_sha256,
        "lending_archive_manifest": lending_borrow.manifest_sha256,
        "lending_archive_balances": lending_borrow.balance_sha256,
        "lending_archive_rates": lending_borrow.rate_sha256,
        "preregistration": str(design["preregistration"]["sha256"]),
        "round1_result": str(design["round1"]["result_sha256"]),
    }
    arm_reports: dict[str, dict[str, _ResearchEvaluation]] = {}
    arm_artifacts: dict[str, object] = {}
    try:
        for arm in ("arm_A", "arm_B"):
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
                        "evaluation_date_indices": evaluation[fold].tolist(),
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
                    bova11_close_by_index=bova11.close_by_session,
                    bova11_binding=bova11_binding,
                    lending_borrow=lending_borrow,
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
        _round2_arm_decision(arm_reports)

        comparator_reports: dict[str, dict[str, _ResearchEvaluation]] = {
            "network": arm_reports["arm_B"],
            "gbdt": {},
            "ensemble": {},
        }
        comparator_artifacts: dict[str, object] = {
            "network": arm_artifacts["arm_B"],
            "gbdt": {},
            "ensemble": {},
        }
        for fold in ("F1", "F2", "F3"):
            network_scores, network_mask = _score_artifact(
                root / "aggregates" / "arm_B" / fold,
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
                    "evaluation_date_indices": evaluation[fold].tolist(),
                },
            )
            g_eval = _evaluate(
                store=store,
                indices=evaluation[fold],
                scores=gbdt_scores,
                score_mask=gbdt_mask,
                cdi=cdi,
                bova11_close_by_index=bova11.close_by_session,
                bova11_binding=bova11_binding,
                lending_borrow=lending_borrow,
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
                    "evaluation_date_indices": evaluation[fold].tolist(),
                },
            )
            e_eval = _evaluate(
                store=store,
                indices=evaluation[fold],
                scores=ensemble_scores,
                score_mask=network_mask,
                cdi=cdi,
                bova11_close_by_index=bova11.close_by_session,
                bova11_binding=bova11_binding,
                lending_borrow=lending_borrow,
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
        return _round2_result(
            root=root,
            design=design,
            store=store,
            access=access,
            source_hashes=source_hashes,
            arm_reports=arm_reports,
            arm_artifacts=arm_artifacts,
            comparator_reports=comparator_reports,
            comparator_artifacts=comparator_artifacts,
        )
    finally:
        store.close()


def recover_round2_result(*, output_root: Path) -> str:
    """Write the missing Round-2 report from complete hash-bound artifacts only."""
    root = output_root.resolve(strict=True)
    result_path = root / "round2_result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    design_path = root / "frozen_design.json"
    design = _read_json(design_path)
    score_implementation = design.get("implementation")
    if (
        design.get("schema") != ROUND2_SCHEMA
        or design.get("status") != "frozen_before_score"
        or not isinstance(score_implementation, Mapping)
        or score_implementation.get("tracked_worktree_clean") is not True
        or not isinstance(score_implementation.get("commit"), str)
        or len(str(score_implementation["commit"])) != 40
    ):
        raise ValueError("Round-2 root is not a clean commit-bound frozen design")
    recovery_implementation = _git_identity()

    round1_root = Path(str(design["round1"]["root"]))
    _verify_sealed_root(round1_root, expected_schema=ROUND1_SCHEMA)
    store_root = Path(str(design["store"]["root"]))
    if sha256_file(store_root / "manifest.json") != design["store"]["manifest_sha256"]:
        raise ValueError("Round-2 store manifest hash mismatch")
    store_manifest, dates = _read_store_header(store_root)
    source_tiers = _source_tier_labels(store_manifest)
    if any(design.get(key) != value for key, value in source_tiers.items()):
        raise ValueError("Round-2 frozen source tiers differ from the store")
    fit, selection, evaluation, fit_target_window, _ = _fold_indices(dates)
    pretrain = _pretrain_indices(dates)
    store, access = _open_round_store(
        store_root, fit, selection, evaluation, fit_target_window, pretrain
    )
    try:
        cdi_design = design["cdi"]
        cdi, provenance = _load_development_cdi(
            dates=dates,
            cdi_path=Path(str(cdi_design["development_extension"]["path"])),
            expected_sha256=str(cdi_design["development_extension"]["sha256"]),
            experiment52_cdi_path=Path(
                str(cdi_design["experiment52_reference"]["path"])
            ),
            experiment52_expected_sha256=str(
                cdi_design["experiment52_reference"]["sha256"]
            ),
        )
        bova_design = design["bova11"]
        bova11 = load_bova11_series(
            Path(str(bova_design["root"])),
            expected_manifest_sha256=str(bova_design["manifest_sha256"]),
            canonical_dates=dates.astype("datetime64[D]").astype(object).tolist(),
        )
        if bova11.data_sha256 != bova_design["data_sha256"]:
            raise ValueError("Round-2 BOVA11 data hash differs from frozen design")
        bova11_binding = {
            **bova_design,
            "manifest_sha256": bova11.manifest_sha256,
            "data_sha256": bova11.data_sha256,
        }
        lending_borrow = _load_frozen_lending(
            design, store_root=store_root, dates=dates
        )
        source_hashes = {
            "v2_store_manifest": str(design["store"]["manifest_sha256"]),
            "cdi_development_extension": str(
                provenance["development_extension"]["sha256"]
            ),
            "cdi_experiment52_reference": str(
                provenance["experiment52_reference"]["sha256"]
            ),
            "bova11_manifest": bova11.manifest_sha256,
            "bova11_data": bova11.data_sha256,
            "lending_archive_manifest": lending_borrow.manifest_sha256,
            "lending_archive_balances": lending_borrow.balance_sha256,
            "lending_archive_rates": lending_borrow.rate_sha256,
            "preregistration": str(design["preregistration"]["sha256"]),
            "round1_result": str(design["round1"]["result_sha256"]),
        }

        arm_reports: dict[str, dict[str, _ResearchEvaluation]] = {}
        arm_artifacts: dict[str, object] = {}
        for arm in ("arm_A", "arm_B"):
            arm_reports[arm] = {}
            arm_artifacts[arm] = {}
            for fold in ("F1", "F2", "F3"):
                aggregate = root / "aggregates" / arm / fold
                stored_scores, stored_mask = _score_artifact(
                    aggregate, require_clean_transfer=True
                )
                expected_scores, expected_mask = _aggregate_network_fold(
                    root,
                    arm,
                    fold,
                    expected_dates=store.dates[evaluation[fold]],
                    expected_isins=store.isins,
                    expected_feature_schema_sha256=str(
                        store.manifest["feature_schema_sha256"]
                    ),
                )
                if not np.array_equal(stored_mask, expected_mask) or not np.array_equal(
                    stored_scores, expected_scores
                ):
                    raise ValueError(
                        f"stored Round-2 aggregate differs from its seed members: {arm}/{fold}"
                    )
                arm_reports[arm][fold] = _evaluation_from_artifacts(
                    aggregate / "evaluation.json",
                    store=store,
                    indices=evaluation[fold],
                    cdi=cdi,
                    bova11_close_by_index=bova11.close_by_session,
                    bova11_binding=bova11_binding,
                    lending_borrow=lending_borrow,
                    allow_legacy_missing_indices=True,
                    expected_fold=fold,
                )
                arm_artifacts[arm][fold] = _existing_score_and_evaluation_record(
                    aggregate
                )

        _round2_arm_decision(arm_reports)
        comparator_reports: dict[str, dict[str, _ResearchEvaluation]] = {
            "network": arm_reports["arm_B"],
            "gbdt": {},
            "ensemble": {},
        }
        comparator_artifacts: dict[str, object] = {
            "network": arm_artifacts["arm_B"],
            "gbdt": {},
            "ensemble": {},
        }
        for name in ("gbdt", "ensemble"):
            for fold in ("F1", "F2", "F3"):
                candidate = root / "comparators" / name / fold
                comparator_reports[name][fold] = _evaluation_from_artifacts(
                    candidate / "evaluation.json",
                    store=store,
                    indices=evaluation[fold],
                    cdi=cdi,
                    bova11_close_by_index=bova11.close_by_session,
                    bova11_binding=bova11_binding,
                    lending_borrow=lending_borrow,
                    allow_legacy_missing_indices=True,
                    expected_fold=fold,
                )
                comparator_artifacts[name][fold] = (
                    _existing_score_and_evaluation_record(candidate)
                )
        return _round2_result(
            root=root,
            design=design,
            store=store,
            access=access,
            source_hashes=source_hashes,
            arm_reports=arm_reports,
            arm_artifacts=arm_artifacts,
            comparator_reports=comparator_reports,
            comparator_artifacts=comparator_artifacts,
            reporting_recovery={
                "scope": (
                    "hash-verify and reuse all completed trajectory, aggregate, comparator, "
                    "and evaluation artifacts; tolerate only the missing evaluation-index "
                    "metadata in the original reporting outputs after rebuilding and "
                    "hash-verifying their full evaluation inputs; serialize unsupported "
                    "bootstrap intervals as JSON null"
                ),
                "implementation": recovery_implementation,
                "score_implementation": score_implementation,
                "reused_aggregate_evaluations": 6,
                "reused_comparator_evaluations": 6,
                "scores_recomputed": 0,
                "evaluations_recomputed": 0,
                "result_changing_retry": False,
            },
        )
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
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping) and (
            "official_validation_accessed" in payload or "test_accessed" in payload
        ):
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
    freeze.add_argument("--bova11-root", type=Path, required=True)
    freeze.add_argument("--bova11-manifest-sha256", required=True)
    freeze.add_argument("--hedge-beta-root", type=Path, required=True)
    freeze.add_argument("--hedge-beta-manifest-sha256", required=True)
    freeze.add_argument("--lending-archive-root", type=Path, required=True)
    freeze.add_argument("--lending-archive-manifest-sha256", required=True)
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
    freeze_replay = commands.add_parser("freeze-round1-ledger-replay")
    freeze_replay.add_argument("--source-round1-root", type=Path, required=True)
    freeze_replay.add_argument("--output-root", type=Path, required=True)
    freeze_replay.add_argument("--hedge-beta-root", type=Path, required=True)
    freeze_replay.add_argument("--hedge-beta-manifest-sha256", required=True)
    run_replay = commands.add_parser("run-round1-ledger-replay")
    run_replay.add_argument("--output-root", type=Path, required=True)
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
    freeze2.add_argument("--bova11-root", type=Path, required=True)
    freeze2.add_argument("--bova11-manifest-sha256", required=True)
    freeze2.add_argument("--hedge-beta-root", type=Path, required=True)
    freeze2.add_argument("--hedge-beta-manifest-sha256", required=True)
    freeze2.add_argument("--lending-archive-root", type=Path, required=True)
    freeze2.add_argument("--lending-archive-manifest-sha256", required=True)
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
    recover2 = commands.add_parser("recover-round2-result")
    recover2.add_argument("--output-root", type=Path, required=True)
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
            bova11_root=arguments.bova11_root,
            bova11_manifest_sha256=arguments.bova11_manifest_sha256,
            hedge_beta_root=arguments.hedge_beta_root,
            hedge_beta_manifest_sha256=arguments.hedge_beta_manifest_sha256,
            lending_archive_root=arguments.lending_archive_root,
            lending_archive_manifest_sha256=(arguments.lending_archive_manifest_sha256),
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
    elif arguments.command == "freeze-round1-ledger-replay":
        digest = freeze_round1_ledger_replay(
            source_round1_root=arguments.source_round1_root,
            output_root=arguments.output_root,
            hedge_beta_root=arguments.hedge_beta_root,
            hedge_beta_manifest_sha256=arguments.hedge_beta_manifest_sha256,
        )
    elif arguments.command == "run-round1-ledger-replay":
        digest = run_round1_ledger_replay(output_root=arguments.output_root)
    elif arguments.command == "freeze-round2":
        digest = freeze_round2(
            round1_root=arguments.round1_root,
            store_root=arguments.store,
            cdi_path=arguments.cdi,
            cdi_sha256=arguments.cdi_sha256,
            experiment52_cdi_path=arguments.experiment52_cdi,
            experiment52_cdi_sha256=arguments.experiment52_cdi_sha256,
            bova11_root=arguments.bova11_root,
            bova11_manifest_sha256=arguments.bova11_manifest_sha256,
            hedge_beta_root=arguments.hedge_beta_root,
            hedge_beta_manifest_sha256=arguments.hedge_beta_manifest_sha256,
            lending_archive_root=arguments.lending_archive_root,
            lending_archive_manifest_sha256=(arguments.lending_archive_manifest_sha256),
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
    elif arguments.command == "recover-round2-result":
        digest = recover_round2_result(output_root=arguments.output_root)
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
