from __future__ import annotations

import copy
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from brazil_rv.modeling import (
    analyze_stage3_context_addition,
    stage3_context_addition,
)
from brazil_rv.modeling.context_ablation import get_context_ablation
from brazil_rv.modeling.contract import (
    FEATURE_CONTRACT_VERSION,
    GLOBAL_CONTEXT_SYMBOLS,
    HORIZONS,
    LOCAL_CONTEXT_SYMBOLS,
    VALIDATION_END,
    VALIDATION_START,
)
from brazil_rv.modeling.stage2_context_ablation import (
    STAGE2_CONTEXT_ABLATION_ORDER,
    STAGE2_SEEDS,
    _configuration as stage2_configuration,
)
from brazil_rv.modeling.stage3_context_addition import (
    ADOPTED_STAGE2_CONTEXT_ABLATION,
    ADOPTED_STAGE2_LOGICAL_CONFIGURATION,
    PACKAGED_FEATURE_MANIFEST_SHA256,
    STAGE2_PRODUCING_COMMIT,
    STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION,
    STAGE3_LOGICAL_CONFIGURATION_ORDER,
    STAGE3_SEEDS,
    _atomic_write_json,
    _configuration,
    _load_state,
    _new_state,
    _validated_stage2_adoptions,
    build_stage3_command,
    stage3_jobs,
    validate_stage3_completed_run,
)

NEW_COMMIT = "d" * 40
NON_RATE_GLOBALS = (
    "ES.v.0",
    "NQ.v.0",
    "CL.v.0",
    "HG.v.0",
    "6E.v.0",
    "6M.v.0",
)


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


def _feature_store(path: Path, identity: str = "canonical") -> Path:
    path.mkdir(parents=True)
    manifest = {
        "contract_version": FEATURE_CONTRACT_VERSION,
        "store_identity": identity,
        "global_context": {
            "source_hashes": {"source": f"source-{identity}"},
            "normalized_store_hashes": {"store": f"store-{identity}"},
        },
        "canonical_inputs": {"universe": f"universe-{identity}"},
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return path


def _core_score(seed: int) -> float:
    return 0.01 + seed * 0.00001


def _stage3_score(logical: str, seed: int) -> float:
    return _core_score(seed) + 0.001 * STAGE3_LOGICAL_CONFIGURATION_ORDER.index(logical)


def _completed_run(
    run_dir: Path,
    configuration: dict[str, object],
    key: str,
    seed: int,
    commit: str,
    score: float,
    *,
    recorded_feature_store: str | None = None,
) -> dict[str, object]:
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = configuration["feature_store"]
    assert isinstance(identity, dict)
    manifest = {
        **copy.deepcopy(configuration["training_semantics"]),
        "status": "completed",
        "seed": seed,
        "git_commit_sha": commit,
        "resolved_feature_store_path": (
            recorded_feature_store
            if recorded_feature_store is not None
            else str(identity["resolved_path"])
        ),
        "feature_manifest_contract_version": configuration["feature_contract"],
        "split_boundaries": copy.deepcopy(configuration["split_boundaries"]),
        "context_ablation": get_context_ablation(key).metadata(),
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
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    horizon_values = {
        30: score - 0.0001,
        60: score,
        120: score + 0.0001,
    }
    rows = [
        {
            "trade_date": trade_date,
            "date_idx": date_index,
            "horizon_minutes": horizon,
            "spearman_ic": horizon_values[horizon],
            "rank_target_pearson_ic": horizon_values[horizon] / 2,
            "top_return": 0.002 + score,
            "bottom_return": -0.001,
            "top_minus_bottom": 0.003 + score,
            "long_only_top": 0.002 + score,
            "one_way_turnover": 0.4 + score,
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
                        "mean_daily_spearman_ic": horizon_values[horizon],
                        "mean_top_minus_bottom": 0.003 + score,
                        "mean_one_way_turnover": 0.4 + score,
                    }
                    for horizon in HORIZONS
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "best.pt").write_bytes(b"fixture")
    pl.DataFrame(
        {
            "epoch": [1, 2, 3],
            "optimizer_steps": [77, 77, 77],
            "epoch_seconds": [9.0, 10.0, 11.0],
        }
    ).write_csv(run_dir / "history.csv")
    return manifest


def _source_stage2_state(
    tmp_path: Path,
    store: Path,
    *,
    recorded_feature_store: str | None = None,
) -> Path:
    source_configuration = stage2_configuration(
        STAGE2_PRODUCING_COMMIT,
        store,
        tmp_path / "stage1" / "state.json",
    )
    source_identity = source_configuration["feature_store"]
    assert isinstance(source_identity, dict)
    source_identity["manifest_sha256"] = PACKAGED_FEATURE_MANIFEST_SHA256
    if recorded_feature_store is not None:
        source_identity["resolved_path"] = recorded_feature_store
    jobs: list[dict[str, object]] = []
    for position, key in enumerate(STAGE2_CONTEXT_ABLATION_ORDER):
        for seed in STAGE2_SEEDS:
            run_dir = tmp_path / "stage2_runs" / f"{key}_{seed}"
            score = 0.0
            if key == ADOPTED_STAGE2_CONTEXT_ABLATION:
                score = _stage3_score(ADOPTED_STAGE2_LOGICAL_CONFIGURATION, seed)
                _completed_run(
                    run_dir,
                    source_configuration,
                    key,
                    seed,
                    STAGE2_PRODUCING_COMMIT,
                    score,
                    recorded_feature_store=recorded_feature_store,
                )
            jobs.append(
                {
                    "context_ablation": key,
                    "seed": seed,
                    "command": ["source", key, str(seed)],
                    "status": "completed",
                    "result_origin": "trained_stage2",
                    "run_dir": str(run_dir),
                    "producing_git_commit_sha": STAGE2_PRODUCING_COMMIT,
                    "source_stage1_state": None,
                    "source_stage1_job": None,
                    "started_at_utc": "2026-08-05T00:00:00+00:00",
                    "completed_at_utc": "2026-08-05T01:00:00+00:00",
                    "primary_validation_ic": score,
                    "error": None,
                    "position": position * len(STAGE2_SEEDS) + STAGE2_SEEDS.index(seed),
                }
            )
    state_path = tmp_path / "stage2" / "state.json"
    state_path.parent.mkdir()
    state_path.write_text(
        json.dumps(
            {
                "state_version": 1,
                "sweep_name": "stage2_context_ablation_matched_seeds",
                "status": "completed",
                "configuration": source_configuration,
                "created_at_utc": "2026-08-05T00:00:00+00:00",
                "completed_at_utc": "2026-08-05T02:00:00+00:00",
                "jobs": jobs,
            }
        ),
        encoding="utf-8",
    )
    return state_path


def _fixture(
    tmp_path: Path,
    *,
    recorded_feature_store: str | None = None,
) -> tuple[dict[str, object], Path, Path, dict[int, tuple]]:
    store = _feature_store(tmp_path / "feature_store")
    source_state = _source_stage2_state(
        tmp_path,
        store,
        recorded_feature_store=recorded_feature_store,
    )
    configuration = _configuration(NEW_COMMIT, store, source_state)
    identity = configuration["feature_store"]
    assert isinstance(identity, dict)
    identity["manifest_sha256"] = PACKAGED_FEATURE_MANIFEST_SHA256
    adopted = _validated_stage2_adoptions(source_state, configuration)
    return configuration, store, source_state, adopted


def _completed_stage3_state(
    tmp_path: Path,
) -> tuple[Path, dict[str, object], Path]:
    configuration, _, source_state, adopted = _fixture(tmp_path)
    state = _new_state(configuration, source_state, adopted)
    for job in state["jobs"]:
        if job["status"] == "completed":
            continue
        logical = str(job["logical_configuration"])
        key = str(job["context_ablation"])
        seed = int(job["seed"])
        run_dir = tmp_path / "stage3_runs" / f"{logical}_{seed}"
        score = _stage3_score(logical, seed)
        _completed_run(run_dir, configuration, key, seed, NEW_COMMIT, score)
        job.update(
            {
                "status": "completed",
                "result_origin": "trained_stage3",
                "run_dir": str(run_dir),
                "run_manifest_sha256": hashlib.sha256(
                    (run_dir / "run_manifest.json").read_bytes()
                ).hexdigest(),
                "producing_git_commit_sha": NEW_COMMIT,
                "source_stage2_state": None,
                "source_stage2_state_sha256": None,
                "source_stage2_job": None,
                "started_at_utc": "2026-08-06T00:00:00+00:00",
                "completed_at_utc": "2026-08-06T01:00:00+00:00",
                "primary_validation_ic": score,
                "error": None,
            }
        )
    state["status"] = "completed"
    state["completed_at_utc"] = "2026-08-06T02:00:00+00:00"
    state_path = tmp_path / "stage3" / "state.json"
    state_path.parent.mkdir()
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return state_path, state, source_state


def _rewrite_source_state(
    source_state: Path,
    configuration: dict[str, object],
    mutate,
) -> None:
    state = json.loads(source_state.read_text(encoding="utf-8"))
    mutate(state)
    source_state.write_text(json.dumps(state), encoding="utf-8")
    configuration["source_stage2_state_sha256"] = hashlib.sha256(
        source_state.read_bytes()
    ).hexdigest()


def test_stage3_matrix_order_commands_and_counts_are_exact() -> None:
    jobs = stage3_jobs()
    assert STAGE3_LOGICAL_CONFIGURATION_ORDER == (
        "core",
        "core_plus_win",
        "core_plus_es",
        "core_plus_nq",
        "core_plus_cl",
        "core_plus_hg",
        "core_plus_6e",
        "core_plus_6m",
    )
    assert STAGE3_SEEDS == (11, 29, 47)
    assert len(jobs) == 24
    assert tuple(
        (
            job["logical_configuration"],
            job["context_ablation"],
            job["seed"],
        )
        for job in jobs
    ) == tuple(
        (
            logical,
            STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION[logical],
            seed,
        )
        for logical in STAGE3_LOGICAL_CONFIGURATION_ORDER
        for seed in STAGE3_SEEDS
    )
    for job in jobs:
        command = job["command"]
        assert "--context-ablation" in command
        assert (
            command[command.index("--context-ablation") + 1] == job["context_ablation"]
        )
        assert not any("test" in value.lower() for value in command)
    with pytest.raises(ValueError):
        build_stage3_command("unknown", 11)
    with pytest.raises(ValueError):
        build_stage3_command("core", 7)


def test_stage3_ablation_semantics_keep_fixed_core_and_add_one_source() -> None:
    core = get_context_ablation(STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION["core"])
    assert core.removed_local_symbols == ("WIN$",)
    assert core.removed_global_symbols == NON_RATE_GLOBALS
    assert core.neutralized_equity_slow_features == ("beta_to_WIN",)
    assert set(LOCAL_CONTEXT_SYMBOLS) - set(core.removed_local_symbols) == {
        "WDO$",
        "DI1F27",
        "DI1F28",
        "DI1F29",
        "DI1F31",
        "DI1$N",
    }
    assert set(GLOBAL_CONTEXT_SYMBOLS) - set(core.removed_global_symbols) == {
        "ZT.v.0",
        "ZN.v.0",
    }
    win = get_context_ablation(
        STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION["core_plus_win"]
    )
    assert win.key == "drop_global_non_rates"
    assert win.removed_local_symbols == ()
    assert win.removed_global_symbols == NON_RATE_GLOBALS
    for logical, symbol in (
        ("core_plus_es", "ES.v.0"),
        ("core_plus_nq", "NQ.v.0"),
        ("core_plus_cl", "CL.v.0"),
        ("core_plus_hg", "HG.v.0"),
        ("core_plus_6e", "6E.v.0"),
        ("core_plus_6m", "6M.v.0"),
    ):
        specification = get_context_ablation(
            STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION[logical]
        )
        assert specification.removed_local_symbols == ("WIN$",)
        assert specification.removed_global_symbols == tuple(
            candidate for candidate in NON_RATE_GLOBALS if candidate != symbol
        )
        assert specification.neutralized_equity_slow_features == ("beta_to_WIN",)


def test_fresh_state_and_dry_run_adopt_three_leave_twenty_one_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration, store, source_state, adopted = _fixture(tmp_path)
    state = _new_state(configuration, source_state, adopted)
    assert len(state["jobs"]) == 24
    assert sum(job["result_origin"] == "adopted_stage2" for job in state["jobs"]) == 3
    assert sum(job["status"] == "pending" for job in state["jobs"]) == 21
    for job in state["jobs"]:
        if job["result_origin"] == "adopted_stage2":
            assert job["logical_configuration"] == "core_plus_win"
            assert job["context_ablation"] == "drop_global_non_rates"
            assert job["run_manifest_sha256"]
            assert job["source_stage2_job"]["producing_git_commit_sha"] == (
                STAGE2_PRODUCING_COMMIT
            )
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    monkeypatch.setattr(
        stage3_context_addition,
        "_prepare",
        lambda **kwargs: (NEW_COMMIT, False, store, configuration),
    )
    monkeypatch.setattr(
        stage3_context_addition.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("dry run attempted training"),
    )
    payload = stage3_context_addition.dry_run_payload(source_state)
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert before == after
    assert payload["logical_job_count"] == 24
    assert payload["adopted_completed_job_count"] == 3
    assert payload["pending_training_job_count"] == 21


@pytest.mark.parametrize(
    "corruption",
    (
        "version",
        "sweep",
        "status",
        "matrix",
        "producing_commit",
        "feature_identity",
        "training_semantics",
        "split",
        "score",
        "duplicate_run_dir",
        "test_selection",
    ),
)
def test_stage2_adoption_rejects_incompatible_source_state(
    tmp_path: Path, corruption: str
) -> None:
    configuration, _, source_state, _ = _fixture(tmp_path)

    def mutate(state):
        target = next(
            job
            for job in state["jobs"]
            if job["context_ablation"] == ADOPTED_STAGE2_CONTEXT_ABLATION
            and job["seed"] == 11
        )
        if corruption == "version":
            state["state_version"] = 2
        elif corruption == "sweep":
            state["sweep_name"] = "wrong"
        elif corruption == "status":
            state["status"] = "running"
        elif corruption == "matrix":
            target["seed"] = 7
        elif corruption == "producing_commit":
            target["producing_git_commit_sha"] = "e" * 40
        elif corruption == "feature_identity":
            state["configuration"]["feature_store"]["manifest_sha256"] = "0" * 64
        elif corruption == "training_semantics":
            state["configuration"]["training_semantics"]["sam"]["rho"] = 0.1
        elif corruption == "split":
            state["configuration"]["split_boundaries"]["validation_end"] = "2025-07-01"
        elif corruption == "score":
            target["primary_validation_ic"] += 1.0
        elif corruption == "duplicate_run_dir":
            state["jobs"][1]["run_dir"] = state["jobs"][0]["run_dir"]
        else:
            target["test_primary_score"] = 1.0

    _rewrite_source_state(source_state, configuration, mutate)
    with pytest.raises(ValueError):
        _validated_stage2_adoptions(source_state, configuration)


def test_missing_or_ambiguous_stage2_reuse_fails_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration, _, source_state, _ = _fixture(tmp_path)
    target = next(
        job
        for job in json.loads(source_state.read_text(encoding="utf-8"))["jobs"]
        if job["context_ablation"] == ADOPTED_STAGE2_CONTEXT_ABLATION
        and job["seed"] == 11
    )
    run_dir = Path(target["run_dir"])
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["git_commit_sha"] = "e" * 40
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("training attempted")

    monkeypatch.setattr(stage3_context_addition.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="exactly one compatible"):
        _validated_stage2_adoptions(source_state, configuration)
    assert not called

    _completed_run(
        run_dir,
        configuration,
        ADOPTED_STAGE2_CONTEXT_ABLATION,
        11,
        STAGE2_PRODUCING_COMMIT,
        _stage3_score("core_plus_win", 11),
    )
    duplicate = tmp_path / "duplicate"
    _completed_run(
        duplicate,
        configuration,
        ADOPTED_STAGE2_CONTEXT_ABLATION,
        11,
        STAGE2_PRODUCING_COMMIT,
        _stage3_score("core_plus_win", 11),
    )
    original = stage3_context_addition._source_run_candidates

    def candidates(source_job, seed):
        found = original(source_job, seed)
        return (*found, duplicate) if seed == 11 else found

    monkeypatch.setattr(stage3_context_addition, "_source_run_candidates", candidates)
    with pytest.raises(ValueError, match="found 2"):
        _validated_stage2_adoptions(source_state, configuration)


def test_windows_and_linux_feature_path_representations_are_equivalent(
    tmp_path: Path,
) -> None:
    recorded = r"Z:\lambda\quant-data\b3\processed\feature_store"
    configuration, _, source_state, adopted = _fixture(
        tmp_path, recorded_feature_store=recorded
    )
    assert set(adopted) == set(STAGE3_SEEDS)
    state = _new_state(configuration, source_state, adopted)
    assert sum(job["result_origin"] == "adopted_stage2" for job in state["jobs"]) == 3


def test_atomic_state_resume_and_adopted_manifest_hash_validation(
    tmp_path: Path,
) -> None:
    configuration, _, source_state, adopted = _fixture(tmp_path)
    state = _new_state(configuration, source_state, adopted)
    state_path = tmp_path / "orchestrator" / "state.json"
    state_path.parent.mkdir()
    _atomic_write_json(state_path, state)
    assert _load_state(state_path, configuration, source_state, adopted) == state
    adopted_job = next(
        job for job in state["jobs"] if job["result_origin"] == "adopted_stage2"
    )
    manifest_path = Path(adopted_job["run_dir"]) / "run_manifest.json"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    refreshed = _validated_stage2_adoptions(source_state, configuration)
    with pytest.raises(ValueError, match="adopted provenance"):
        _load_state(state_path, configuration, source_state, refreshed)


@pytest.mark.parametrize(
    "corruption",
    (
        "configuration",
        "matrix",
        "metadata",
        "command",
        "state_status",
        "recovery",
        "source_position",
        "unexpected_adoption",
    ),
)
def test_resume_rejects_incompatible_or_malformed_state(
    tmp_path: Path, corruption: str
) -> None:
    configuration, _, source_state, adopted = _fixture(tmp_path)
    state = copy.deepcopy(_new_state(configuration, source_state, adopted))
    if corruption == "configuration":
        state["configuration"]["seeds"] = [29]
    elif corruption == "matrix":
        state["jobs"].pop()
    elif corruption == "metadata":
        state["jobs"][0]["context_ablation_metadata"]["key"] = "wrong"
    elif corruption == "command":
        state["jobs"][0]["command"].append("--wrong")
    elif corruption == "state_status":
        state["status"] = "failed"
    elif corruption == "recovery":
        state["jobs"][0]["recovery_count"] = -1
    elif corruption == "source_position":
        adopted_job = next(
            job for job in state["jobs"] if job["result_origin"] == "adopted_stage2"
        )
        adopted_job["source_stage2_job"]["position"] = -1
    else:
        state["jobs"][0]["result_origin"] = "adopted_stage2"
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError):
        _load_state(state_path, configuration, source_state, adopted)


def test_interrupted_and_completed_jobs_recover_explicitly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration, _, source_state, adopted = _fixture(tmp_path)
    state = _new_state(configuration, source_state, adopted)
    job = state["jobs"][0]
    job["status"] = "running"
    monkeypatch.setattr(
        stage3_context_addition, "_candidate_run_dirs", lambda *args: ()
    )
    assert not stage3_context_addition._complete_or_recover_job(job, configuration)
    assert job["recovery_count"] == 1
    assert job["last_recovery_at_utc"]

    run_dir = tmp_path / "recovered"
    _completed_run(
        run_dir,
        configuration,
        str(job["context_ablation"]),
        int(job["seed"]),
        NEW_COMMIT,
        _stage3_score("core", 11),
    )
    monkeypatch.setattr(
        stage3_context_addition,
        "_candidate_run_dirs",
        lambda *args: (run_dir,),
    )
    assert stage3_context_addition._complete_or_recover_job(job, configuration)
    assert job["status"] == "completed"
    assert job["result_origin"] == "trained_stage3"
    assert job["run_manifest_sha256"]


def test_ambiguous_new_stage3_runs_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration, _, source_state, adopted = _fixture(tmp_path)
    job = _new_state(configuration, source_state, adopted)["jobs"][0]
    first = tmp_path / "first"
    second = tmp_path / "second"
    for run_dir in (first, second):
        _completed_run(
            run_dir,
            configuration,
            str(job["context_ablation"]),
            int(job["seed"]),
            NEW_COMMIT,
            _stage3_score("core", 11),
        )
    monkeypatch.setattr(
        stage3_context_addition,
        "_candidate_run_dirs",
        lambda *args: (first, second),
    )
    with pytest.raises(ValueError, match="Multiple completed runs"):
        stage3_context_addition._complete_or_recover_job(job, configuration)


def test_failed_training_preserves_three_adopted_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration, store, source_state, adopted = _fixture(tmp_path)
    state_dir = tmp_path / "orchestrator"
    state_dir.mkdir()
    _atomic_write_json(
        state_dir / "state.json",
        _new_state(configuration, source_state, adopted),
    )
    monkeypatch.setattr(
        stage3_context_addition,
        "_prepare",
        lambda **kwargs: (NEW_COMMIT, True, store, configuration),
    )
    monkeypatch.setattr(stage3_context_addition, "active_lock_owner", lambda path: None)
    monkeypatch.setattr(
        stage3_context_addition,
        "PRODUCTION_TRAINING_LOCK",
        tmp_path / "production.lock",
    )
    monkeypatch.setattr(
        stage3_context_addition,
        "_assert_invocation_identity",
        lambda *args: None,
    )
    monkeypatch.setattr(
        stage3_context_addition,
        "_production_run_directories",
        lambda: set(),
    )
    monkeypatch.setattr(
        stage3_context_addition.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=9),
    )
    with pytest.raises(RuntimeError, match="training failed"):
        stage3_context_addition.run_sweep(state_dir, source_state)
    failed = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
    assert sum(job["result_origin"] == "adopted_stage2" for job in failed["jobs"]) == 3
    assert failed["jobs"][0]["status"] == "failed"
    assert failed["jobs"][0]["error"] == "training exited with code 9"


def test_completed_jobs_are_never_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_path, state, source_state = _completed_stage3_state(tmp_path)
    state_dir = tmp_path / "resume"
    state_dir.mkdir()
    (state_dir / "state.json").write_text(
        fixture_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    configuration = state["configuration"]
    store = Path(str(configuration["feature_store"]["resolved_path"]))
    monkeypatch.setattr(
        stage3_context_addition,
        "_prepare",
        lambda **kwargs: (NEW_COMMIT, True, store, configuration),
    )
    monkeypatch.setattr(stage3_context_addition, "active_lock_owner", lambda path: None)
    monkeypatch.setattr(
        stage3_context_addition,
        "PRODUCTION_TRAINING_LOCK",
        tmp_path / "production.lock",
    )
    monkeypatch.setattr(
        stage3_context_addition.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("completed job was rerun"),
    )
    result = stage3_context_addition.run_sweep(state_dir, source_state)
    completed = json.loads(result.read_text(encoding="utf-8"))
    assert completed["status"] == "completed"
    assert (
        sum(job["result_origin"] == "adopted_stage2" for job in completed["jobs"]) == 3
    )
    assert (
        sum(job["result_origin"] == "trained_stage3" for job in completed["jobs"]) == 21
    )


@pytest.mark.parametrize(
    "corruption",
    (
        "partial",
        "duplicate",
        "origin",
        "configuration",
        "score",
        "manifest_hash",
        "command",
    ),
)
def test_analyzer_rejects_incomplete_or_ambiguous_state(
    tmp_path: Path, corruption: str
) -> None:
    state_path, state, source_state = _completed_stage3_state(tmp_path)
    if corruption == "partial":
        state["status"] = "running"
    elif corruption == "duplicate":
        state["jobs"][1]["run_dir"] = state["jobs"][0]["run_dir"]
    elif corruption == "origin":
        state["jobs"][0]["result_origin"] = "adopted_stage2"
    elif corruption == "configuration":
        state["configuration"]["seeds"] = [29]
    elif corruption == "score":
        state["jobs"][0]["primary_validation_ic"] += 1.0
    elif corruption == "manifest_hash":
        state["jobs"][0]["run_manifest_sha256"] = "0" * 64
    else:
        state["jobs"][0]["command"].append("--wrong")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    output = tmp_path / "analysis"
    with pytest.raises(ValueError):
        analyze_stage3_context_addition.analyze_sweep(state_path, output, source_state)
    assert not output.exists()


def test_analyzer_uses_same_seed_core_and_never_reads_test_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_path, state, source_state = _completed_stage3_state(tmp_path)
    first_run = Path(str(state["jobs"][0]["run_dir"]))
    (first_run / "test_metrics.json").write_text("not valid json", encoding="utf-8")
    (first_run / "test_predictions.parquet").write_bytes(b"not parquet")
    original_read_text = Path.read_text

    def guarded_read_text(path: Path, *args, **kwargs):
        if path.name.startswith("test_"):
            raise AssertionError(f"test artifact accessed: {path}")
        return original_read_text(path, *args, **kwargs)

    original_read_parquet = analyze_stage3_context_addition.pl.read_parquet

    def guarded_read_parquet(path, *args, **kwargs):
        if Path(path).name.startswith("test_"):
            raise AssertionError(f"test artifact accessed: {path}")
        return original_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(
        analyze_stage3_context_addition.pl,
        "read_parquet",
        guarded_read_parquet,
    )
    output = tmp_path / "analysis"
    json_path, csv_path = analyze_stage3_context_addition.analyze_sweep(
        state_path, output, source_state
    )
    first_json = original_read_text(json_path, encoding="utf-8")
    first_csv = original_read_text(csv_path, encoding="utf-8")
    second_json, second_csv = analyze_stage3_context_addition.analyze_sweep(
        state_path, output, source_state
    )
    assert original_read_text(second_json, encoding="utf-8") == first_json
    assert original_read_text(second_csv, encoding="utf-8") == first_csv
    summary = json.loads(first_json)
    table = pl.read_csv(csv_path)
    assert summary["logical_job_count"] == 24
    assert summary["configuration_count"] == 8
    assert summary["same_seed_reference_configuration"] == "core"
    assert summary["selection"] is None
    assert len(summary["runs"]) == 24
    assert table.height == 24
    candidate = next(
        row
        for row in summary["configurations"]
        if row["logical_configuration"] == "core_plus_es"
    )
    deltas = candidate["primary_delta_across_training_seeds"]
    assert tuple(
        deltas["paired_delta_by_seed"][str(seed)] for seed in STAGE3_SEEDS
    ) == pytest.approx((0.002, 0.002, 0.002))
    assert deltas["positive_seed_count"] == 3
    run = next(
        row
        for row in summary["runs"]
        if row["logical_configuration"] == "core_plus_es" and row["seed"] == 29
    )
    assert run["periods"]["first_half"][
        "primary_delta_vs_same_seed_core"
    ] == pytest.approx(0.002)
    assert run["periods"]["latest_half"][
        "primary_delta_vs_same_seed_core"
    ] == pytest.approx(0.002)
    assert run["training"]["optimizer_updates"] == 231
    assert run["within_trained_model_daily_bootstrap"]["replications"] == 10_000
    assert summary["result_origin_counts"] == {
        "adopted_stage2": 3,
        "trained_stage3": 21,
    }


def test_completed_run_rejects_test_selection_and_semantic_mismatch(
    tmp_path: Path,
) -> None:
    configuration, _, _, _ = _fixture(tmp_path)
    run_dir = tmp_path / "run"
    manifest = _completed_run(
        run_dir,
        configuration,
        STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION["core"],
        11,
        NEW_COMMIT,
        _core_score(11),
    )
    assert validate_stage3_completed_run(
        run_dir,
        configuration,
        STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION["core"],
        11,
        NEW_COMMIT,
    ) == pytest.approx(_core_score(11))
    original_path = manifest["resolved_feature_store_path"]
    manifest["resolved_feature_store_path"] = (
        r"Z:\\different\\quant-data\\b3\\processed\\feature_store"
    )
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="different feature store"):
        validate_stage3_completed_run(
            run_dir,
            configuration,
            STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION["core"],
            11,
            NEW_COMMIT,
        )
    manifest["resolved_feature_store_path"] = original_path
    manifest["test_primary_score"] = 1.0
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="test-derived"):
        validate_stage3_completed_run(
            run_dir,
            configuration,
            STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION["core"],
            11,
            NEW_COMMIT,
        )


def test_stage3_sources_name_no_test_artifacts_or_split_selector() -> None:
    runner = Path(stage3_context_addition.__file__).read_text(encoding="utf-8")
    analyzer = Path(analyze_stage3_context_addition.__file__).read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "test_metrics",
        "test_predictions",
        "select_sample_split",
    ):
        assert forbidden not in runner
        assert forbidden not in analyzer
