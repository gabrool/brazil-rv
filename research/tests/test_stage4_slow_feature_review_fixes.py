from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import torch

from brazil_rv.modeling import (
    analyze_stage4_slow_feature_ablation,
    audit_slow_features,
    stage4_slow_feature_ablation,
)
from brazil_rv.modeling.analyze_context_ablation import (
    BOOTSTRAP_BLOCK_TRADING_DAYS,
    BOOTSTRAP_REPLICATIONS,
)
from brazil_rv.modeling.audit_slow_features import (
    AUDIT_JSON,
    AUDIT_NAME,
    AUDIT_VERSION,
    EQUITY_LOCAL_AXIS,
    FOCUSED_CSV,
    GLOBAL_AXIS,
    PEARSON_CSV,
    SPEARMAN_CSV,
    STATS_CSV,
    SlowAuditGroupInput,
    _analyze_group,
    _select_training_samples,
    validate_training_slow_audit,
)
from brazil_rv.modeling.context_ablation import get_context_ablation
from brazil_rv.modeling.contract import (
    FEATURE_CONTRACT_VERSION,
    GLOBAL_CONTEXT_SYMBOLS,
    HORIZONS,
    LOCAL_CONTEXT_SYMBOLS,
    SplitBoundaries,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
)
from brazil_rv.modeling.feature_ablation import (
    get_feature_ablation,
    resolve_feature_ablation,
)
from brazil_rv.modeling.process_lock import exclusive_process_lock
from brazil_rv.modeling.stage2_context_ablation import (
    _feature_store_identity,
    _training_semantics,
)
from brazil_rv.modeling.stage3_context_addition import (
    PACKAGED_FEATURE_MANIFEST_SHA256,
    STAGE2_PRODUCING_COMMIT,
    STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION,
    STAGE3_LOGICAL_CONFIGURATION_ORDER,
    STAGE3_SEEDS,
    stage3_jobs,
)
from brazil_rv.preprocessing.contract import (
    GLOBAL_SLOW_CHANNELS,
    GLOBAL_UNUSED_SLOW_CHANNEL_INDICES,
    SLOW_CHANNELS,
)

REMOVED_INDICES = (0, 10, 11, 12, 14, 15, 16, 26, 27, 28, 29)
STAGE3_COMMIT = "3" * 40
STAGE4_COMMIT = "4" * 40


def _resolved_feature(key: str):
    return resolve_feature_ablation(
        get_feature_ablation(key), slow_features=SLOW_CHANNELS
    )


def _schema(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "contract_version": FEATURE_CONTRACT_VERSION,
                "slow_channels": [
                    {"index": index, "name": name}
                    for index, name in enumerate(SLOW_CHANNELS)
                ],
                "global_slow_channels": [
                    {"index": index, "name": name}
                    for index, name in enumerate(GLOBAL_SLOW_CHANNELS)
                ],
                "global_slow": list(GLOBAL_UNUSED_SLOW_CHANNEL_INDICES),
            }
        ),
        encoding="utf-8",
    )


def test_axis_partitioned_groups_preserve_names_active_sets_and_structural_noops() -> (
    None
):
    equity_values = np.zeros((6, 32), dtype=np.float64)
    equity_values[:, 0] = np.arange(6)
    equity_values[:, 1] = np.arange(6) * 2
    equity_values[:, 20] = np.arange(6) * 3
    equity_applicable = np.ones(32, dtype=bool)
    equity_applicable[[30, 31]] = False
    equity = SlowAuditGroupInput(
        "equity",
        "equity",
        None,
        EQUITY_LOCAL_AXIS,
        SLOW_CHANNELS,
        equity_values,
        equity_applicable,
        (20,),
    )

    global_values = np.zeros((6, 32), dtype=np.float64)
    global_values[:, 0] = np.arange(6)
    global_values[:, 1] = -np.arange(6)
    global_values[:, 14] = 999.0
    global_applicable = np.ones(32, dtype=bool)
    global_applicable[list(GLOBAL_UNUSED_SLOW_CHANNEL_INDICES)] = False
    global_group = SlowAuditGroupInput(
        "global:ZT.v.0",
        "global_context",
        "ZT.v.0",
        GLOBAL_AXIS,
        GLOBAL_SLOW_CHANNELS,
        global_values,
        global_applicable,
    )

    ablation = _resolved_feature("drop_slow_low_prior")
    equity_result = _analyze_group(equity, ablation)
    global_result = _analyze_group(global_group, ablation)
    assert _analyze_group(equity, ablation) == equity_result
    assert _analyze_group(global_group, ablation) == global_result
    assert [row["name"] for row in equity_result["channels"]] == list(SLOW_CHANNELS)
    assert [row["name"] for row in global_result["channels"]] == list(
        GLOBAL_SLOW_CHANNELS
    )
    assert (
        equity_result["channels"][10]["name"] != global_result["channels"][10]["name"]
    )
    assert equity_result["channels"][20]["status"] == "context_disabled"
    assert equity_result["channels"][20]["valid_observation_count"] == 0
    assert all(
        row["retained_index"] != 20 for row in equity_result["focused_correlations"]
    )
    assert global_result["channels"][14]["status"] == "inapplicable"
    assert global_result["channels"][14]["valid_observation_count"] == 0
    mapping_14 = next(
        row for row in global_result["removed_position_mapping"] if row["index"] == 14
    )
    assert mapping_14 == {
        "index": 14,
        "equity_local_name": "daily_dollar_volume_regime_20d",
        "global_name": "unused_equity_liquidity_14",
        "global_structurally_unused": True,
        "group_axis_identity": GLOBAL_AXIS,
        "group_axis_name": "unused_equity_liquidity_14",
        "group_structurally_inapplicable": True,
    }
    assert equity_result["pearson_correlation_matrix"][0][1] == pytest.approx(1.0)
    assert global_result["pearson_correlation_matrix"][0][1] == pytest.approx(-1.0)


def test_training_sample_selection_excludes_embargo_validation_and_test() -> None:
    rows = pl.DataFrame(
        {
            "sample_id": [70, 10, 20, 30, 40, 50],
            "date_idx": [5, 0, 1, 2, 3, 4],
            "decision_idx": [0, 1, 2, 3, 4, 5],
            "trade_date": [
                TEST_START,
                TRAIN_START,
                TRAIN_END,
                TRAIN_END + timedelta(days=1),
                VALIDATION_START,
                VALIDATION_END + timedelta(days=1),
            ],
        }
    )
    training, contract = _select_training_samples(rows)
    assert training["sample_id"].to_list() == [10, 20]
    assert contract["training_sample_count"] == 2
    assert contract["training_start"] == str(TRAIN_START)
    assert contract["training_end"] == str(TRAIN_END)
    changed_nontraining = rows.with_columns(
        pl.when(pl.col("trade_date") > TRAIN_END)
        .then(pl.col("sample_id") + 10_000)
        .otherwise(pl.col("sample_id"))
        .alias("sample_id")
    )
    _, changed_contract = _select_training_samples(changed_nontraining)
    assert (
        changed_contract["training_sample_mask_sha256"]
        == contract["training_sample_mask_sha256"]
    )


class _ContextIdentity:
    def metadata(self) -> dict[str, object]:
        return get_context_ablation("drop_win_and_global_non_rates").metadata()


def _audit_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, object], dict[str, object]]:
    store = tmp_path / "store"
    store.mkdir()
    _schema(store / "feature_schema.json")
    feature_identity = {
        "resolved_path": str(store),
        "manifest_sha256": "a" * 64,
        "contract_version": FEATURE_CONTRACT_VERSION,
        "global_context_source_hashes": {"source": "x"},
        "global_context_normalized_store_hashes": {"store": "x"},
        "canonical_inputs": {"input": "x"},
    }
    values = np.arange(12 * 32, dtype=np.float64).reshape(12, 32)
    equity_applicable = np.ones(32, dtype=bool)
    equity_applicable[[30, 31]] = False
    global_applicable = np.ones(32, dtype=bool)
    global_applicable[list(GLOBAL_UNUSED_SLOW_CHANNEL_INDICES)] = False
    inputs = (
        SlowAuditGroupInput(
            "equity",
            "equity",
            None,
            EQUITY_LOCAL_AXIS,
            SLOW_CHANNELS,
            values,
            equity_applicable,
            (20,),
        ),
        SlowAuditGroupInput(
            "global:ZT.v.0",
            "global_context",
            "ZT.v.0",
            GLOBAL_AXIS,
            GLOBAL_SLOW_CHANNELS,
            values,
            global_applicable,
        ),
    )
    sample_contract = {
        "split": "train",
        "training_start": str(TRAIN_START),
        "training_end": str(TRAIN_END),
        "training_date_count": 2,
        "training_sample_count": 2,
        "training_sample_mask_sha256": "b" * 64,
        "training_sample_mask_fields": ["sample_id", "date_idx", "decision_idx"],
        "excluded_splits": ["embargo_1", "validation", "embargo_2", "test"],
        "sampling": {"used": False, "seed": None},
        "observation_rule": "synthetic partitioned groups",
        "group_order": [group.group_id for group in inputs],
        "row_group_counts": {
            group.group_id: int(group.values.shape[0]) for group in inputs
        },
        "total_observation_count": sum(group.values.shape[0] for group in inputs),
        "retained_context_symbols": list(
            audit_slow_features.EXPECTED_RETAINED_CONTEXTS
        ),
    }
    monkeypatch.setattr(
        audit_slow_features, "_training_groups", lambda path: (inputs, sample_contract)
    )
    monkeypatch.setattr(
        audit_slow_features,
        "resolve_context_ablation_for_store",
        lambda *args: _ContextIdentity(),
    )
    ablation = _resolved_feature("drop_slow_low_prior")
    groups = [_analyze_group(group, ablation) for group in inputs]
    output_dir = tmp_path / "audit"
    output_dir.mkdir()
    audit_slow_features._stats_frame(groups).write_csv(output_dir / STATS_CSV)
    audit_slow_features._correlation_frame(groups, "pearson").write_csv(
        output_dir / PEARSON_CSV
    )
    audit_slow_features._correlation_frame(groups, "spearman").write_csv(
        output_dir / SPEARMAN_CSV
    )
    audit_slow_features._focused_frame(groups).write_csv(output_dir / FOCUSED_CSV)
    audit = {
        "audit_name": AUDIT_NAME,
        "audit_version": AUDIT_VERSION,
        "validation_only_experiment": True,
        "test_metrics_accessed": False,
        "feature_store": feature_identity,
        "feature_manifest_sha256": feature_identity["manifest_sha256"],
        "context_ablation": "drop_win_and_global_non_rates",
        "context_ablation_metadata": _ContextIdentity().metadata(),
        "feature_ablation": ablation.metadata(),
        "canonical_axes": {
            EQUITY_LOCAL_AXIS: [
                {"index": index, "name": name}
                for index, name in enumerate(SLOW_CHANNELS)
            ],
            GLOBAL_AXIS: [
                {"index": index, "name": name}
                for index, name in enumerate(GLOBAL_SLOW_CHANNELS)
            ],
        },
        "group_order": [group["group_id"] for group in groups],
        "sample_contract": copy.deepcopy(sample_contract),
        "groups": groups,
        "output_sha256": {
            name: audit_slow_features._sha256(output_dir / name)
            for name in (STATS_CSV, PEARSON_CSV, SPEARMAN_CSV, FOCUSED_CSV)
        },
    }
    audit_path = output_dir / AUDIT_JSON
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    return audit_path, feature_identity, audit


@pytest.mark.parametrize(
    "corruption", ("mapping", "active_set", "sample_hash", "output_hash")
)
def test_audit_validation_rejects_semantic_or_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    audit_path, feature_identity, audit = _audit_fixture(tmp_path, monkeypatch)
    assert validate_training_slow_audit(audit_path, feature_identity) == audit
    if corruption == "mapping":
        audit["groups"][0]["removed_position_mapping"][0]["global_name"] = "wrong"
    elif corruption == "active_set":
        audit["groups"][0]["channels"][20]["status"] = "active"
    elif corruption == "sample_hash":
        audit["sample_contract"]["training_sample_mask_sha256"] = "0" * 64
    else:
        (audit_path.parent / STATS_CSV).write_text("mutated", encoding="utf-8")
    if corruption != "output_hash":
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_training_slow_audit(audit_path, feature_identity)


def _validation_dates() -> list[date]:
    weekdays: list[date] = []
    current = VALIDATION_START
    while current <= VALIDATION_END:
        if current.weekday() < 5:
            weekdays.append(current)
        current += timedelta(days=1)
    indices = np.linspace(0, len(weekdays) - 1, 244, dtype=int)
    dates = [weekdays[index] for index in indices]
    assert len(set(dates)) == 244
    return dates


def _feature_store(path: Path) -> tuple[Path, dict[str, object]]:
    path.mkdir()
    manifest = {
        "contract_version": FEATURE_CONTRACT_VERSION,
        "global_context": {
            "source_hashes": {"source": "canonical"},
            "normalized_store_hashes": {"store": "canonical"},
        },
        "canonical_inputs": {"universe": "canonical"},
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _schema(path / "feature_schema.json")
    identity = _feature_store_identity(path)
    identity["manifest_sha256"] = PACKAGED_FEATURE_MANIFEST_SHA256
    return path, identity


def _write_completed_run(
    run_dir: Path,
    configuration: dict[str, object],
    feature_key: str,
    seed: int,
    commit: str,
    score: float,
    *,
    legacy_feature_identity: bool,
    recorded_store_path: str,
) -> dict[str, str]:
    run_dir.mkdir(parents=True)
    identity = configuration["feature_store"]
    manifest = {
        **copy.deepcopy(configuration["training_semantics"]),
        "status": "completed",
        "seed": seed,
        "git_commit_sha": commit,
        "resolved_feature_store_path": recorded_store_path,
        "feature_manifest_contract_version": configuration["feature_contract"],
        "split_boundaries": copy.deepcopy(configuration["split_boundaries"]),
        "context_ablation": get_context_ablation(
            "drop_win_and_global_non_rates"
        ).metadata(),
        "global_context_source_hashes": identity["global_context_source_hashes"],
        "global_context_normalized_store_hashes": identity[
            "global_context_normalized_store_hashes"
        ],
        "resolved_source_paths": identity["canonical_inputs"],
        "best_validation_primary_score": score,
        "best_epoch": 2,
        "stopped_epoch": 3,
        "successful_optimizer_updates": 231,
        "training_duration_seconds": 30.0,
        "scheduler_steps": {"steps_per_epoch": 77},
    }
    feature_metadata = stage4_slow_feature_ablation._feature_ablation_metadata(
        feature_key
    )
    checkpoint: dict[str, object] = {}
    if not legacy_feature_identity:
        manifest["feature_ablation"] = feature_metadata
        checkpoint["feature_ablation"] = feature_metadata
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    torch.save(checkpoint, run_dir / "best.pt")
    torch.save(checkpoint, run_dir / "final.pt")
    rows = [
        {
            "trade_date": trade_date,
            "date_idx": date_index,
            "horizon_minutes": horizon,
            "spearman_ic": score,
            "rank_target_pearson_ic": score / 2,
            "top_return": 0.02,
            "bottom_return": -0.01,
            "top_minus_bottom": 0.03,
            "long_only_top": 0.02,
            "one_way_turnover": 0.40,
        }
        for date_index, trade_date in enumerate(_validation_dates())
        for horizon in HORIZONS
    ]
    pl.DataFrame(rows).write_parquet(run_dir / "validation_daily_metrics.parquet")
    (run_dir / "validation_metrics.json").write_text(
        json.dumps(
            {
                "primary_score": score,
                "horizons": [
                    {
                        "horizon_minutes": horizon,
                        "mean_daily_spearman_ic": score,
                        "mean_top_minus_bottom": 0.03,
                        "mean_one_way_turnover": 0.40,
                    }
                    for horizon in HORIZONS
                ],
            }
        ),
        encoding="utf-8",
    )
    pl.DataFrame(
        {
            "epoch": [1, 2, 3],
            "optimizer_steps": [77, 77, 77],
            "epoch_seconds": [9.0, 10.0, 11.0],
        }
    ).write_csv(run_dir / "history.csv")
    return stage4_slow_feature_ablation._artifact_hashes(run_dir)


def _strict_adoption_fixture(
    tmp_path: Path, *, portable_source_path: str | None = None
) -> tuple[
    Path,
    dict[str, object],
    dict[int, tuple[dict[str, object], Path, float, str, dict[str, str], str]],
]:
    store, current_identity = _feature_store(tmp_path / "store")
    source_path = portable_source_path or str(store)
    source_identity = copy.deepcopy(current_identity)
    source_identity["resolved_path"] = source_path
    source_stage2 = tmp_path / "stage2.json"
    source_stage2.write_text("{}", encoding="utf-8")
    split = {key: str(value) for key, value in asdict(SplitBoundaries()).items()}
    source_configuration = {
        "orchestrator_git_commit_sha": STAGE3_COMMIT,
        "feature_store": source_identity,
        "feature_contract": FEATURE_CONTRACT_VERSION,
        "local_context_symbols": list(LOCAL_CONTEXT_SYMBOLS),
        "global_context_symbols": list(GLOBAL_CONTEXT_SYMBOLS),
        "training_semantics": _training_semantics(),
        "split_boundaries": split,
        "logical_configuration_order": list(STAGE3_LOGICAL_CONFIGURATION_ORDER),
        "context_ablation_by_logical_configuration": dict(
            STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION
        ),
        "context_ablation_metadata_by_logical_configuration": {
            logical: get_context_ablation(key).metadata()
            for logical, key in STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION.items()
        },
        "seeds": list(STAGE3_SEEDS),
        "logical_job_count": 24,
        "adopted_stage2_job_count": 3,
        "new_training_job_count": 21,
        "source_stage2_state": str(source_stage2),
        "source_stage2_state_sha256": hashlib.sha256(
            source_stage2.read_bytes()
        ).hexdigest(),
        "source_stage2_feature_store_resolved_path": source_path,
        "required_stage2_producing_commit": STAGE2_PRODUCING_COMMIT,
        "required_feature_manifest_sha256": PACKAGED_FEATURE_MANIFEST_SHA256,
    }
    jobs = []
    for base in stage3_jobs():
        logical = str(base["logical_configuration"])
        seed = int(base["seed"])
        run_dir = tmp_path / "stage3_runs" / f"{logical}_{seed}"
        score = 0.01 + seed / 1_000_000.0
        manifest_sha = hashlib.sha256(f"{logical}/{seed}".encode()).hexdigest()
        if logical == "core":
            hashes = _write_completed_run(
                run_dir,
                source_configuration,
                "none",
                seed,
                STAGE3_COMMIT,
                score,
                legacy_feature_identity=True,
                recorded_store_path=source_path,
            )
            manifest_sha = hashes["run_manifest.json"]
        jobs.append(
            {
                **base,
                "status": "completed",
                "result_origin": "trained_stage3",
                "run_dir": str(run_dir),
                "run_manifest_sha256": manifest_sha,
                "producing_git_commit_sha": STAGE3_COMMIT,
                "source_stage2_state": None,
                "source_stage2_state_sha256": None,
                "source_stage2_job": None,
                "started_at_utc": "2026-08-06T00:00:00+00:00",
                "completed_at_utc": "2026-08-06T01:00:00+00:00",
                "primary_validation_ic": score,
                "recovery_count": 0,
                "last_recovery_at_utc": None,
                "error": None,
            }
        )
    source_state = {
        "state_version": 1,
        "sweep_name": "stage3_context_addition_matched_seeds",
        "status": "completed",
        "configuration": source_configuration,
        "jobs": jobs,
    }
    source_path_file = tmp_path / "stage3_state.json"
    source_path_file.write_text(json.dumps(source_state), encoding="utf-8")
    audit_path = tmp_path / "audit.json"
    audit_path.write_text("{}", encoding="utf-8")
    feature_metadata = {
        key: stage4_slow_feature_ablation._feature_ablation_metadata(key)
        for key in ("none", "drop_slow_low_prior")
    }
    configuration = {
        "orchestrator_git_commit_sha": STAGE4_COMMIT,
        "feature_store": current_identity,
        "feature_contract": FEATURE_CONTRACT_VERSION,
        "local_context_symbols": list(LOCAL_CONTEXT_SYMBOLS),
        "global_context_symbols": list(GLOBAL_CONTEXT_SYMBOLS),
        "retained_context_symbols": list(
            stage4_slow_feature_ablation.EXPECTED_RETAINED_CONTEXTS
        ),
        "training_semantics": _training_semantics(),
        "split_boundaries": split,
        "logical_configuration_order": ["full_slow", "drop_slow_low_prior"],
        "feature_ablation_by_logical_configuration": {
            "full_slow": "none",
            "drop_slow_low_prior": "drop_slow_low_prior",
        },
        "feature_ablation_metadata_by_key": feature_metadata,
        "context_ablation": "drop_win_and_global_non_rates",
        "context_ablation_metadata": get_context_ablation(
            "drop_win_and_global_non_rates"
        ).metadata(),
        "seeds": [11, 29, 47],
        "logical_job_count": 6,
        "adopted_stage3_job_count": 3,
        "new_training_job_count": 3,
        "source_stage3_state": str(source_path_file),
        "source_stage3_state_sha256": hashlib.sha256(
            source_path_file.read_bytes()
        ).hexdigest(),
        "source_stage3_feature_store_resolved_path": source_path,
        "source_stage3_producing_commit": STAGE3_COMMIT,
        "required_feature_manifest_sha256": PACKAGED_FEATURE_MANIFEST_SHA256,
        "training_slow_audit": {
            "path": str(audit_path),
            "sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            "audit_name": AUDIT_NAME,
            "audit_version": AUDIT_VERSION,
        },
    }
    adopted = stage4_slow_feature_ablation._validated_stage3_adoptions(
        source_path_file, configuration
    )
    return source_path_file, configuration, adopted


def test_portable_stage3_store_identity_passes_and_different_identity_fails(
    tmp_path: Path,
) -> None:
    source_state, configuration, adopted = _strict_adoption_fixture(
        tmp_path, portable_source_path="/lambda/nfs/b3/feature-store"
    )
    assert tuple(adopted) == (11, 29, 47)
    assert all(values[-1] == "legacy_implicit_none" for values in adopted.values())
    state = json.loads(source_state.read_text(encoding="utf-8"))
    state["configuration"]["feature_store"]["canonical_inputs"] = {
        "universe": "different"
    }
    source_state.write_text(json.dumps(state), encoding="utf-8")
    configuration["source_stage3_state_sha256"] = hashlib.sha256(
        source_state.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="different feature store"):
        stage4_slow_feature_ablation._validated_stage3_adoptions(
            source_state, configuration
        )


@pytest.mark.parametrize(
    "corruption",
    (
        "core_plus_win",
        "seed",
        "context",
        "hyperparameter",
        "feature_manifest",
        "split",
        "source_state_hash",
        "producing_commit",
        "run_manifest_hash",
    ),
)
def test_stage3_adoption_fails_closed_on_provenance_drift(
    tmp_path: Path, corruption: str
) -> None:
    source_state, configuration, _ = _strict_adoption_fixture(tmp_path)
    state = json.loads(source_state.read_text(encoding="utf-8"))
    core = next(
        job
        for job in state["jobs"]
        if job["logical_configuration"] == "core" and job["seed"] == 11
    )
    if corruption == "core_plus_win":
        core["logical_configuration"] = "core_plus_win"
    elif corruption == "seed":
        core["seed"] = 7
    elif corruption == "context":
        core["context_ablation"] = "drop_global_non_rates"
    elif corruption == "hyperparameter":
        state["configuration"]["training_semantics"]["sam"]["rho"] = 0.1
    elif corruption == "feature_manifest":
        state["configuration"]["feature_store"]["manifest_sha256"] = "0" * 64
    elif corruption == "split":
        state["configuration"]["split_boundaries"]["validation_end"] = "2025-07-01"
    elif corruption == "source_state_hash":
        state["nonsemantic_marker"] = "changed"
    elif corruption == "producing_commit":
        core["producing_git_commit_sha"] = "0" * 40
    elif corruption == "run_manifest_hash":
        core["run_manifest_sha256"] = "0" * 64
    source_state.write_text(json.dumps(state), encoding="utf-8")
    if corruption != "source_state_hash":
        configuration["source_stage3_state_sha256"] = hashlib.sha256(
            source_state.read_bytes()
        ).hexdigest()
    with pytest.raises(ValueError):
        stage4_slow_feature_ablation._validated_stage3_adoptions(
            source_state, configuration
        )


@pytest.mark.parametrize(
    "metadata_case", ("partial", "unknown", "treatment", "contradictory")
)
def test_legacy_feature_identity_accepts_only_complete_implicit_none(
    tmp_path: Path, metadata_case: str
) -> None:
    _, configuration, adopted = _strict_adoption_fixture(tmp_path)
    run_dir = adopted[11][1]
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    final = torch.load(run_dir / "final.pt", map_location="cpu", weights_only=False)
    none = configuration["feature_ablation_metadata_by_key"]["none"]
    treatment = configuration["feature_ablation_metadata_by_key"]["drop_slow_low_prior"]
    if metadata_case == "partial":
        manifest["feature_ablation"] = none
    elif metadata_case == "unknown":
        manifest["feature_ablation"] = {"key": "unknown"}
        best["feature_ablation"] = {"key": "unknown"}
        final["feature_ablation"] = {"key": "unknown"}
    elif metadata_case == "treatment":
        manifest["feature_ablation"] = treatment
        best["feature_ablation"] = treatment
        final["feature_ablation"] = treatment
    else:
        manifest["feature_ablation"] = none
        best["feature_ablation"] = treatment
        final["feature_ablation"] = none
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    torch.save(best, run_dir / "best.pt")
    torch.save(final, run_dir / "final.pt")
    with pytest.raises(ValueError):
        stage4_slow_feature_ablation.validate_stage4_completed_run(
            run_dir,
            configuration,
            "none",
            11,
            STAGE3_COMMIT,
            allow_legacy_none=True,
        )


def _new_stage4_state(
    source_state: Path,
    configuration: dict[str, object],
    adopted: dict[int, tuple],
) -> dict[str, object]:
    return stage4_slow_feature_ablation._new_state(configuration, source_state, adopted)


def _add_treatment_run(
    tmp_path: Path,
    configuration: dict[str, object],
    job: dict[str, object],
    suffix: str = "",
) -> Path:
    seed = int(job["seed"])
    run_dir = tmp_path / f"treatment_{seed}{suffix}"
    _write_completed_run(
        run_dir,
        configuration,
        "drop_slow_low_prior",
        seed,
        STAGE4_COMMIT,
        0.02 + seed / 1_000_000.0,
        legacy_feature_identity=False,
        recorded_store_path=str(configuration["feature_store"]["resolved_path"]),
    )
    return run_dir


def test_interrupted_treatment_recovers_one_artifact_once_and_rejects_multiple(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_state, configuration, adopted = _strict_adoption_fixture(tmp_path)
    state = _new_stage4_state(source_state, configuration, adopted)
    job = state["jobs"][3]
    job["status"] = "running"
    first = _add_treatment_run(tmp_path, configuration, job)
    monkeypatch.setattr(
        stage4_slow_feature_ablation,
        "_candidate_treatment_runs",
        lambda *args: (first,),
    )
    assert stage4_slow_feature_ablation._complete_or_recover_job(job, configuration)
    assert job["status"] == "completed"
    assert job["recovery_count"] == 1
    assert job["last_recovery_at_utc"]
    assert stage4_slow_feature_ablation._complete_or_recover_job(job, configuration)
    assert job["recovery_count"] == 1

    second_job = _new_stage4_state(source_state, configuration, adopted)["jobs"][3]
    second_job["status"] = "running"
    second = _add_treatment_run(tmp_path, configuration, second_job, "_duplicate")
    monkeypatch.setattr(
        stage4_slow_feature_ablation,
        "_candidate_treatment_runs",
        lambda *args: (first, second),
    )
    with pytest.raises(ValueError, match="Multiple completed runs"):
        stage4_slow_feature_ablation._complete_or_recover_job(second_job, configuration)


def test_state_rejects_mutated_job_mapping_and_required_output_hash(
    tmp_path: Path,
) -> None:
    source_state, configuration, adopted = _strict_adoption_fixture(tmp_path)
    state = _new_stage4_state(source_state, configuration, adopted)
    state["jobs"][3]["feature_ablation_metadata"][
        "resolved_position_mapping_sha256"
    ] = "0" * 64
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="job metadata is malformed"):
        stage4_slow_feature_ablation._load_state(
            state_path, configuration, source_state, adopted
        )

    state = _new_stage4_state(source_state, configuration, adopted)
    state["jobs"][0]["output_sha256"]["final.pt"] = "0" * 64
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="adopted provenance"):
        stage4_slow_feature_ablation._load_state(
            state_path, configuration, source_state, adopted
        )


@pytest.mark.parametrize("failure", ("launch", "nonzero"))
def test_subprocess_failure_records_failed_state_and_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    source_state, configuration, adopted = _strict_adoption_fixture(tmp_path)
    store = Path(str(configuration["feature_store"]["resolved_path"]))
    state_dir = tmp_path / "runner"
    state_dir.mkdir()
    stage4_slow_feature_ablation._atomic_write_json(
        state_dir / "state.json",
        _new_stage4_state(source_state, configuration, adopted),
    )
    monkeypatch.setattr(
        stage4_slow_feature_ablation,
        "_prepare",
        lambda **kwargs: (STAGE4_COMMIT, True, store, configuration),
    )
    monkeypatch.setattr(
        stage4_slow_feature_ablation,
        "_validated_stage3_adoptions",
        lambda *args: adopted,
    )
    monkeypatch.setattr(
        stage4_slow_feature_ablation, "active_lock_owner", lambda p: None
    )
    monkeypatch.setattr(
        stage4_slow_feature_ablation,
        "PRODUCTION_TRAINING_LOCK",
        tmp_path / "production.lock",
    )
    monkeypatch.setattr(
        stage4_slow_feature_ablation, "_assert_invocation_identity", lambda *args: None
    )
    monkeypatch.setattr(
        stage4_slow_feature_ablation, "_production_run_directories", lambda: set()
    )
    if failure == "launch":

        def fail_launch(*args, **kwargs):
            raise OSError("synthetic launch failure")

        monkeypatch.setattr(stage4_slow_feature_ablation.subprocess, "run", fail_launch)
    else:
        monkeypatch.setattr(
            stage4_slow_feature_ablation.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(returncode=9),
        )
    with pytest.raises(RuntimeError):
        stage4_slow_feature_ablation.run_sweep(
            state_dir, source_state, tmp_path / "audit.json"
        )
    failed = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
    assert failed["jobs"][3]["status"] == "failed"
    assert failed["jobs"][3]["error"]


def _complete_treatments(
    tmp_path: Path,
    state: dict[str, object],
    configuration: dict[str, object],
) -> None:
    for job in state["jobs"][3:]:
        run_dir = _add_treatment_run(tmp_path, configuration, job)
        score, hashes, source = (
            stage4_slow_feature_ablation.validate_stage4_completed_run(
                run_dir,
                configuration,
                "drop_slow_low_prior",
                int(job["seed"]),
                STAGE4_COMMIT,
            )
        )
        job.update(
            {
                "status": "completed",
                "result_origin": "trained_stage4",
                "run_dir": str(run_dir),
                "run_manifest_sha256": hashes["run_manifest.json"],
                "output_sha256": hashes,
                "feature_ablation_identity_source": source,
                "producing_git_commit_sha": STAGE4_COMMIT,
                "completed_at_utc": "2026-08-08T00:00:00+00:00",
                "primary_validation_ic": score,
                "error": None,
            }
        )
    state["status"] = "completed"
    state["completed_at_utc"] = "2026-08-08T00:00:00+00:00"


def test_completed_treatments_resume_without_duplicate_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_state, configuration, adopted = _strict_adoption_fixture(tmp_path)
    state = _new_stage4_state(source_state, configuration, adopted)
    _complete_treatments(tmp_path, state, configuration)
    state_dir = tmp_path / "resume"
    state_dir.mkdir()
    (state_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    store = Path(str(configuration["feature_store"]["resolved_path"]))
    monkeypatch.setattr(
        stage4_slow_feature_ablation,
        "_prepare",
        lambda **kwargs: (STAGE4_COMMIT, True, store, configuration),
    )
    monkeypatch.setattr(
        stage4_slow_feature_ablation,
        "_validated_stage3_adoptions",
        lambda *args: adopted,
    )
    monkeypatch.setattr(
        stage4_slow_feature_ablation, "active_lock_owner", lambda p: None
    )
    monkeypatch.setattr(
        stage4_slow_feature_ablation,
        "PRODUCTION_TRAINING_LOCK",
        tmp_path / "production.lock",
    )
    monkeypatch.setattr(
        stage4_slow_feature_ablation.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("completed treatment was retrained"),
    )
    result = stage4_slow_feature_ablation.run_sweep(
        state_dir, source_state, tmp_path / "audit.json"
    )
    completed = json.loads(result.read_text(encoding="utf-8"))
    assert (
        sum(job["result_origin"] == "trained_stage4" for job in completed["jobs"]) == 3
    )


def test_sweep_and_production_locks_fail_before_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_state, configuration, adopted = _strict_adoption_fixture(tmp_path)
    store = Path(str(configuration["feature_store"]["resolved_path"]))
    state_dir = tmp_path / "locks"
    state_dir.mkdir()
    monkeypatch.setattr(
        stage4_slow_feature_ablation,
        "_prepare",
        lambda **kwargs: (STAGE4_COMMIT, True, store, configuration),
    )
    monkeypatch.setattr(
        stage4_slow_feature_ablation,
        "_validated_stage3_adoptions",
        lambda *args: adopted,
    )
    monkeypatch.setattr(
        stage4_slow_feature_ablation.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("training subprocess started"),
    )
    with exclusive_process_lock(state_dir / "sweep.lock", "synthetic holder"):
        with pytest.raises(RuntimeError):
            stage4_slow_feature_ablation.run_sweep(
                state_dir, source_state, tmp_path / "audit.json"
            )
    monkeypatch.setattr(
        stage4_slow_feature_ablation,
        "active_lock_owner",
        lambda path: {"purpose": "other production run"},
    )
    with pytest.raises(RuntimeError, match="Another production training run"):
        stage4_slow_feature_ablation.run_sweep(
            state_dir, source_state, tmp_path / "audit.json"
        )


def test_analyzer_requires_exact_observations_keeps_bootstrap_and_ignores_test_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_state, configuration, adopted = _strict_adoption_fixture(tmp_path)
    state = _new_stage4_state(source_state, configuration, adopted)
    _complete_treatments(tmp_path, state, configuration)
    state_path = tmp_path / "stage4_state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    audit_path = Path(str(configuration["training_slow_audit"]["path"]))
    monkeypatch.setattr(
        analyze_stage4_slow_feature_ablation,
        "validate_training_slow_audit",
        lambda *args: {"audit_version": AUDIT_VERSION},
    )
    first_run = Path(str(state["jobs"][0]["run_dir"]))
    (first_run / "test_metrics.json").write_text("not valid json", encoding="utf-8")
    (first_run / "test_predictions.parquet").write_bytes(b"not parquet")
    original_read_text = Path.read_text
    original_read_parquet = analyze_stage4_slow_feature_ablation.pl.read_parquet

    def guarded_read_text(path: Path, *args, **kwargs):
        if path.name.startswith("test_"):
            raise AssertionError(f"final-test artifact accessed: {path}")
        return original_read_text(path, *args, **kwargs)

    def guarded_read_parquet(path, *args, **kwargs):
        if Path(path).name.startswith("test_"):
            raise AssertionError(f"final-test artifact accessed: {path}")
        return original_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(
        analyze_stage4_slow_feature_ablation.pl, "read_parquet", guarded_read_parquet
    )
    output = tmp_path / "analysis"
    paths = analyze_stage4_slow_feature_ablation.analyze_sweep(
        state_path,
        output,
        stage3_state_path=source_state,
        slow_audit_path=audit_path,
    )
    summary = json.loads(original_read_text(paths[0], encoding="utf-8"))
    bootstrap = summary["paired_moving_block_bootstrap"]
    assert bootstrap["block_trading_days"] == BOOTSTRAP_BLOCK_TRADING_DAYS == 5
    assert bootstrap["replications"] == BOOTSTRAP_REPLICATIONS == 10_000
    assert summary["test_metrics_accessed"] is False
    assert summary["selection"] is None

    target_job = state["jobs"][3]
    daily_path = Path(str(target_job["run_dir"])) / "validation_daily_metrics.parquet"
    daily = original_read_parquet(daily_path)
    daily = daily.with_columns(
        pl.when(pl.arange(0, pl.len()) == 0)
        .then(pl.col("date_idx") + 1)
        .otherwise(pl.col("date_idx"))
        .alias("date_idx")
    )
    daily.write_parquet(daily_path)
    target_job["output_sha256"]["validation_daily_metrics.parquet"] = hashlib.sha256(
        daily_path.read_bytes()
    ).hexdigest()
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="identical validation observations"):
        analyze_stage4_slow_feature_ablation.analyze_sweep(
            state_path,
            output,
            stage3_state_path=source_state,
            slow_audit_path=audit_path,
        )


def test_final_test_derived_state_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="forbidden test-derived field"):
        stage4_slow_feature_ablation._reject_test_derived_metadata(
            {"final_test_metric": 0.1}, "Stage-4 synthetic state"
        )


def test_job_specifications_bind_exact_dual_axis_mapping() -> None:
    for job in stage4_slow_feature_ablation.stage4_jobs():
        expected = stage4_slow_feature_ablation._feature_ablation_metadata(
            str(job["feature_ablation"])
        )
        assert job["feature_ablation_metadata"] == expected
        specification = json.loads(str(job["serialized_job_specification"]))
        assert specification["feature_ablation"] == expected
        assert len(str(job["job_specification_sha256"])) == 64


def test_stage4_atomic_state_write_replaces_and_cleans_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    path.write_text('{"old": true}', encoding="utf-8")
    stage4_slow_feature_ablation._atomic_write_json(path, {"value": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1}

    def fail_replace(source, destination):
        raise OSError("synthetic atomic replace failure")

    monkeypatch.setattr(stage4_slow_feature_ablation.os, "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic atomic replace failure"):
        stage4_slow_feature_ablation._atomic_write_json(path, {"value": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1}
    assert not path.with_name("state.json.tmp").exists()
