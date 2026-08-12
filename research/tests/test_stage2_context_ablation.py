from __future__ import annotations

import copy
import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from brazil_rv.modeling import (
    analyze_stage2_context_ablation,
    stage2_context_ablation,
)
from brazil_rv.modeling.context_ablation import (
    STAGE1_CONTEXT_ABLATION_ORDER,
    get_context_ablation,
    resolve_context_ablation,
)
from brazil_rv.modeling.contract import (
    FEATURE_CONTRACT_VERSION,
    GLOBAL_CONTEXT_SYMBOLS,
    HORIZONS,
    LOCAL_CONTEXT_SYMBOLS,
    VALIDATION_END,
    VALIDATION_START,
)
from brazil_rv.modeling.stage1_context_ablation import stage1_jobs
from brazil_rv.modeling.stage2_context_ablation import (
    ADOPTED_STAGE1_KEYS,
    STAGE1_PRODUCING_COMMIT,
    STAGE2_CONTEXT_ABLATION_ORDER,
    STAGE2_SEEDS,
    _atomic_write_json,
    _configuration,
    _load_state,
    _new_state,
    feature_stores_equivalent,
    stage2_jobs,
    validate_stage2_completed_run,
)
from brazil_rv.preprocessing.contract import SLOW_CHANNELS


ORIGINAL_STAGE1_ROWS = (
    (
        "none",
        (),
        (),
        (),
        "Use every canonical local and global context source.",
    ),
    (
        "drop_fixed_di",
        ("DI1F27", "DI1F28", "DI1F29", "DI1F31"),
        (),
        (
            "beta_to_DI1F27",
            "beta_to_DI1F28",
            "beta_to_DI1F29",
            "beta_to_DI1F31",
        ),
        "Remove all four fixed-maturity DI futures and their equity betas.",
    ),
    (
        "drop_all_di",
        ("DI1F27", "DI1F28", "DI1F29", "DI1F31", "DI1$N"),
        (),
        (
            "beta_to_DI1F27",
            "beta_to_DI1F28",
            "beta_to_DI1F29",
            "beta_to_DI1F31",
        ),
        "Remove every DI future and all fixed-maturity DI equity betas.",
    ),
    (
        "drop_us_equities",
        (),
        ("ES.v.0", "NQ.v.0"),
        (),
        "Remove the ES and NQ global equity futures.",
    ),
    (
        "drop_us_rates",
        (),
        ("ZT.v.0", "ZN.v.0"),
        (),
        "Remove the ZT and ZN global rate futures.",
    ),
    (
        "drop_commodities",
        (),
        ("CL.v.0", "HG.v.0"),
        (),
        "Remove the CL and HG global commodity futures.",
    ),
    (
        "drop_global_fx",
        (),
        ("6E.v.0", "6M.v.0"),
        (),
        "Remove the 6E and 6M global FX futures.",
    ),
    (
        "drop_all_local",
        LOCAL_CONTEXT_SYMBOLS,
        (),
        (
            "beta_to_WIN",
            "beta_to_WDO",
            "beta_to_DI1F27",
            "beta_to_DI1F28",
            "beta_to_DI1F29",
            "beta_to_DI1F31",
        ),
        "Remove all seven local contexts and every local-source equity beta.",
    ),
    (
        "drop_all_global",
        (),
        GLOBAL_CONTEXT_SYMBOLS,
        (),
        "Remove all eight global context futures.",
    ),
    (
        "drop_all_context",
        LOCAL_CONTEXT_SYMBOLS,
        GLOBAL_CONTEXT_SYMBOLS,
        (
            "beta_to_WIN",
            "beta_to_WDO",
            "beta_to_DI1F27",
            "beta_to_DI1F28",
            "beta_to_DI1F29",
            "beta_to_DI1F31",
        ),
        "Remove all local and global contexts and every local-source equity beta.",
    ),
    (
        "drop_win",
        ("WIN$",),
        (),
        ("beta_to_WIN",),
        "Remove the WIN$ local future and its equity beta.",
    ),
    (
        "drop_wdo",
        ("WDO$",),
        (),
        ("beta_to_WDO",),
        "Remove the WDO$ local future and its equity beta.",
    ),
    (
        "drop_di1f27",
        ("DI1F27",),
        (),
        ("beta_to_DI1F27",),
        "Remove DI1F27 and its equity beta.",
    ),
    (
        "drop_di1f28",
        ("DI1F28",),
        (),
        ("beta_to_DI1F28",),
        "Remove DI1F28 and its equity beta.",
    ),
    (
        "drop_di1f29",
        ("DI1F29",),
        (),
        ("beta_to_DI1F29",),
        "Remove DI1F29 and its equity beta.",
    ),
    (
        "drop_di1f31",
        ("DI1F31",),
        (),
        ("beta_to_DI1F31",),
        "Remove DI1F31 and its equity beta.",
    ),
    (
        "drop_di1n",
        ("DI1$N",),
        (),
        (),
        "Remove the liquidity-selected DI1$N local rate future.",
    ),
    (
        "drop_es",
        (),
        ("ES.v.0",),
        (),
        "Remove the ES global equity future.",
    ),
    (
        "drop_nq",
        (),
        ("NQ.v.0",),
        (),
        "Remove the NQ global equity future.",
    ),
    (
        "drop_zt",
        (),
        ("ZT.v.0",),
        (),
        "Remove the ZT global rate future.",
    ),
    (
        "drop_zn",
        (),
        ("ZN.v.0",),
        (),
        "Remove the ZN global rate future.",
    ),
    (
        "drop_cl",
        (),
        ("CL.v.0",),
        (),
        "Remove the CL global commodity future.",
    ),
    (
        "drop_hg",
        (),
        ("HG.v.0",),
        (),
        "Remove the HG global commodity future.",
    ),
    (
        "drop_6e",
        (),
        ("6E.v.0",),
        (),
        "Remove the 6E global FX future.",
    ),
    (
        "drop_6m",
        (),
        ("6M.v.0",),
        (),
        "Remove the 6M global FX future.",
    ),
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


def _score(key: str, seed: int) -> float:
    return 0.01 + seed * 0.00001 + STAGE2_CONTEXT_ABLATION_ORDER.index(key) * 0.001


def _completed_run(
    run_dir: Path,
    configuration: dict[str, object],
    key: str,
    seed: int,
    commit: str,
    *,
    score: float | None = None,
    feature_store: Path | None = None,
) -> dict[str, object]:
    run_dir.mkdir(parents=True)
    score = _score(key, seed) if score is None else score
    identity = configuration["feature_store"]
    assert isinstance(identity, dict)
    manifest = {
        **copy.deepcopy(configuration["training_semantics"]),
        "status": "completed",
        "seed": seed,
        "git_commit_sha": commit,
        "resolved_feature_store_path": str(
            feature_store or Path(str(identity["resolved_path"]))
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


def _stage1_state(
    path: Path,
    configuration: dict[str, object],
    *,
    source_store: Path | None = None,
) -> Path:
    semantics = configuration["training_semantics"]
    assert isinstance(semantics, dict)
    identity = configuration["feature_store"]
    assert isinstance(identity, dict)
    source_store = source_store or Path(str(identity["resolved_path"]))
    source_configuration = {
        "git_commit_sha": STAGE1_PRODUCING_COMMIT,
        "resolved_feature_store_path": str(source_store),
        "feature_contract": configuration["feature_contract"],
        "local_context_symbols": configuration["local_context_symbols"],
        "global_context_symbols": configuration["global_context_symbols"],
        "model_name": semantics["model_name"],
        "tcn_settings": semantics["tcn_settings"],
        "parameter_count": semantics["parameter_count"],
        "peer_features": semantics["peer_features"],
        "optimizer_variant": semantics["optimizer_variant"],
        "objective": semantics["objective"],
        "sam": semantics["sam"],
        "global_context": semantics["global_context"],
        "seed": 29,
        "split_boundaries": configuration["split_boundaries"],
        "ablation_order": list(STAGE1_CONTEXT_ABLATION_ORDER),
    }
    jobs: list[dict[str, object]] = []
    for key in STAGE1_CONTEXT_ABLATION_ORDER:
        run_dir = path / "runs" / key
        score = _score(key, 29) if key in ADOPTED_STAGE1_KEYS else 0.0
        if key in ADOPTED_STAGE1_KEYS:
            _completed_run(
                run_dir,
                configuration,
                key,
                29,
                STAGE1_PRODUCING_COMMIT,
                score=score,
                feature_store=source_store,
            )
        jobs.append(
            {
                "context_ablation": key,
                "seed": 29,
                "status": "completed",
                "run_dir": str(run_dir),
                "primary_validation_ic": score,
            }
        )
    path.mkdir(parents=True, exist_ok=True)
    state_path = path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "state_version": 1,
                "sweep_name": "stage1_context_ablation_seed29",
                "status": "completed",
                "configuration": source_configuration,
                "jobs": jobs,
            }
        ),
        encoding="utf-8",
    )
    return state_path


def _fixture_configuration(tmp_path: Path) -> tuple[dict[str, object], Path, Path]:
    store = _feature_store(tmp_path / "feature_store")
    stage1_placeholder = tmp_path / "stage1" / "state.json"
    configuration = _configuration("c" * 40, store, stage1_placeholder)
    stage1_state = _stage1_state(
        stage1_placeholder.parent, configuration, source_store=store
    )
    assert configuration["source_stage1_state"] == str(stage1_state)
    return configuration, store, stage1_state


def _completed_stage2_state(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    configuration, _, stage1_state = _fixture_configuration(tmp_path)
    state = _new_state(configuration, stage1_state)
    for job in state["jobs"]:
        if job["status"] == "completed":
            continue
        key = str(job["context_ablation"])
        seed = int(job["seed"])
        run_dir = tmp_path / "stage2_runs" / f"{key}_{seed}"
        score = _score(key, seed)
        _completed_run(run_dir, configuration, key, seed, "c" * 40, score=score)
        job.update(
            {
                "status": "completed",
                "result_origin": "trained_stage2",
                "run_dir": str(run_dir),
                "producing_git_commit_sha": "c" * 40,
                "completed_at_utc": "2026-08-06T00:00:00+00:00",
                "primary_validation_ic": score,
                "error": None,
            }
        )
    state["status"] = "completed"
    state["completed_at_utc"] = "2026-08-06T00:00:00+00:00"
    state_path = tmp_path / "stage2_state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return state_path, state


def test_drop_global_non_rates_removes_only_six_globals() -> None:
    resolved = resolve_context_ablation(
        get_context_ablation("drop_global_non_rates"),
        local_symbols=LOCAL_CONTEXT_SYMBOLS,
        global_symbols=GLOBAL_CONTEXT_SYMBOLS,
        equity_slow_features=SLOW_CHANNELS,
    )
    expected_removed = (
        "ES.v.0",
        "NQ.v.0",
        "CL.v.0",
        "HG.v.0",
        "6E.v.0",
        "6M.v.0",
    )
    assert resolved.specification.removed_global_symbols == expected_removed
    assert resolved.global_slots == tuple(
        GLOBAL_CONTEXT_SYMBOLS.index(symbol) for symbol in expected_removed
    )
    assert set(GLOBAL_CONTEXT_SYMBOLS) - set(expected_removed) == {"ZT.v.0", "ZN.v.0"}
    assert resolved.local_slots == ()
    assert resolved.equity_slow_indices == ()


def test_original_stage1_order_and_specifications_are_field_exact() -> None:
    assert STAGE1_CONTEXT_ABLATION_ORDER == tuple(
        row[0] for row in ORIGINAL_STAGE1_ROWS
    )
    actual = tuple(
        (
            key,
            get_context_ablation(key).removed_local_symbols,
            get_context_ablation(key).removed_global_symbols,
            get_context_ablation(key).neutralized_equity_slow_features,
            get_context_ablation(key).description,
        )
        for key in STAGE1_CONTEXT_ABLATION_ORDER
    )
    assert actual == ORIGINAL_STAGE1_ROWS


def test_stage1_and_stage2_matrices_are_exact() -> None:
    first = stage1_jobs()
    assert len(first) == 25
    assert tuple(job["context_ablation"] for job in first) == (
        STAGE1_CONTEXT_ABLATION_ORDER
    )
    assert {job["seed"] for job in first} == {29}
    second = stage2_jobs()
    assert len(second) == 18
    assert tuple((job["context_ablation"], job["seed"]) for job in second) == tuple(
        (key, seed) for key in STAGE2_CONTEXT_ABLATION_ORDER for seed in STAGE2_SEEDS
    )


def test_fresh_state_and_dry_run_adopt_five_leave_thirteen_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration, store, stage1_state = _fixture_configuration(tmp_path)
    state = _new_state(configuration, stage1_state)
    assert len(state["jobs"]) == 18
    assert sum(job["result_origin"] == "adopted_stage1" for job in state["jobs"]) == 5
    assert sum(job["status"] == "pending" for job in state["jobs"]) == 13
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    monkeypatch.setattr(
        stage2_context_ablation,
        "_prepare",
        lambda **kwargs: ("c" * 40, False, store, configuration),
    )
    monkeypatch.setattr(
        stage2_context_ablation.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("dry run attempted training"),
    )
    payload = stage2_context_ablation.dry_run_payload(stage1_state)
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert before == after
    assert payload["logical_job_count"] == 18
    assert payload["adopted_completed_job_count"] == 5
    assert payload["pending_training_job_count"] == 13


def test_feature_store_equivalence_uses_manifest_not_basename(tmp_path: Path) -> None:
    home = _feature_store(tmp_path / "home" / "quant-data" / "store", "same")
    nfs = _feature_store(tmp_path / "lambda" / "quant-data" / "store", "same")
    different = _feature_store(tmp_path / "other" / "quant-data" / "store", "different")
    assert feature_stores_equivalent(home, nfs)
    assert not feature_stores_equivalent(home, different)


@pytest.mark.parametrize(
    "corruption",
    ("seed", "ablation", "commit", "feature_fingerprint", "duplicate_run_dir"),
)
def test_stage1_adoption_rejects_wrong_identity(
    tmp_path: Path, corruption: str
) -> None:
    configuration, _, stage1_state = _fixture_configuration(tmp_path)
    state = json.loads(stage1_state.read_text(encoding="utf-8"))
    if corruption == "seed":
        state["jobs"][0]["seed"] = 11
    elif corruption == "ablation":
        state["jobs"][0]["context_ablation"] = "drop_win"
    elif corruption == "commit":
        state["configuration"]["git_commit_sha"] = "d" * 40
    elif corruption == "feature_fingerprint":
        other = _feature_store(tmp_path / "different_store", "different")
        state["configuration"]["resolved_feature_store_path"] = str(other)
    else:
        state["jobs"][1]["run_dir"] = state["jobs"][0]["run_dir"]
    stage1_state.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError):
        _new_state(configuration, stage1_state)


@pytest.mark.parametrize(
    "corruption",
    (
        "seed",
        "ablation",
        "model",
        "rho",
        "temperature",
        "split",
        "feature_contract",
        "producing_commit",
        "feature_fingerprint",
    ),
)
def test_completed_run_rejects_semantic_mismatch(
    tmp_path: Path, corruption: str
) -> None:
    configuration, _, _ = _fixture_configuration(tmp_path)
    run_dir = tmp_path / "candidate"
    manifest = _completed_run(
        run_dir, configuration, "none", 29, STAGE1_PRODUCING_COMMIT
    )
    if corruption == "seed":
        manifest["seed"] = 11
    elif corruption == "ablation":
        manifest["context_ablation"] = get_context_ablation("drop_di1n").metadata()
    elif corruption == "model":
        manifest["tcn_settings"]["width"] = 128
    elif corruption == "rho":
        manifest["sam"]["rho"] = 0.1
    elif corruption == "temperature":
        manifest["objective"]["temperature"] = 0.2
    elif corruption == "split":
        manifest["split_boundaries"]["validation_end"] = "2025-07-01"
    elif corruption == "feature_contract":
        manifest["feature_manifest_contract_version"] = "wrong"
    elif corruption == "producing_commit":
        manifest["git_commit_sha"] = "d" * 40
    else:
        other = _feature_store(tmp_path / "different_store", "different")
        manifest["resolved_feature_store_path"] = str(other)
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_stage2_completed_run(
            run_dir,
            configuration,
            "none",
            29,
            STAGE1_PRODUCING_COMMIT,
        )


def test_atomic_state_write_and_load_resume(tmp_path: Path) -> None:
    configuration, _, stage1_state = _fixture_configuration(tmp_path)
    state = _new_state(configuration, stage1_state)
    state_path = tmp_path / "stage2" / "state.json"
    state_path.parent.mkdir()
    _atomic_write_json(state_path, state)
    assert not state_path.with_name("state.json.tmp").exists()
    loaded = _load_state(state_path, configuration, stage1_state)
    assert loaded == state


def test_resume_revalidates_adopted_stage1_provenance(tmp_path: Path) -> None:
    configuration, _, stage1_state = _fixture_configuration(tmp_path)
    state = _new_state(configuration, stage1_state)
    adopted = next(
        job for job in state["jobs"] if job["result_origin"] == "adopted_stage1"
    )
    adopted["primary_validation_ic"] = float(adopted["primary_validation_ic"]) + 1.0
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="adopted provenance"):
        _load_state(state_path, configuration, stage1_state)


def test_completed_jobs_are_not_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_path, state = _completed_stage2_state(tmp_path)
    state_dir = tmp_path / "orchestrator"
    state_dir.mkdir()
    (state_dir / "state.json").write_text(
        fixture_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    configuration = state["configuration"]
    store = Path(str(configuration["feature_store"]["resolved_path"]))
    stage1_state = Path(str(configuration["source_stage1_state"]))
    monkeypatch.setattr(
        stage2_context_ablation,
        "_prepare",
        lambda **kwargs: ("c" * 40, True, store, configuration),
    )
    monkeypatch.setattr(stage2_context_ablation, "active_lock_owner", lambda path: None)
    monkeypatch.setattr(
        stage2_context_ablation,
        "PRODUCTION_TRAINING_LOCK",
        tmp_path / "production.lock",
    )
    monkeypatch.setattr(
        stage2_context_ablation.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("completed job was rerun"),
    )
    result = stage2_context_ablation.run_sweep(state_dir, stage1_state)
    completed = json.loads(result.read_text(encoding="utf-8"))
    assert completed["status"] == "completed"
    assert len(completed["jobs"]) == 18


def test_failed_job_preserves_adopted_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration, store, stage1_state = _fixture_configuration(tmp_path)
    state_dir = tmp_path / "orchestrator"
    state_dir.mkdir()
    _atomic_write_json(
        state_dir / "state.json", _new_state(configuration, stage1_state)
    )
    monkeypatch.setattr(
        stage2_context_ablation,
        "_prepare",
        lambda **kwargs: ("c" * 40, True, store, configuration),
    )
    monkeypatch.setattr(stage2_context_ablation, "active_lock_owner", lambda path: None)
    monkeypatch.setattr(
        stage2_context_ablation,
        "PRODUCTION_TRAINING_LOCK",
        tmp_path / "production.lock",
    )
    monkeypatch.setattr(
        stage2_context_ablation, "_assert_invocation_identity", lambda *args: None
    )
    monkeypatch.setattr(
        stage2_context_ablation, "_production_run_directories", lambda: set()
    )
    monkeypatch.setattr(
        stage2_context_ablation.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=9),
    )
    with pytest.raises(RuntimeError, match="training failed"):
        stage2_context_ablation.run_sweep(state_dir, stage1_state)
    failed = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
    assert sum(job["result_origin"] == "adopted_stage1" for job in failed["jobs"]) == 5
    assert failed["jobs"][0]["status"] == "failed"
    assert failed["jobs"][0]["error"] == "training exited with code 9"


def test_ambiguous_completed_stage2_runs_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration, _, _ = _fixture_configuration(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    _completed_run(first, configuration, "none", 11, "c" * 40)
    _completed_run(second, configuration, "none", 11, "c" * 40)
    job = next(
        job
        for job in _new_state(
            configuration, Path(str(configuration["source_stage1_state"]))
        )["jobs"]
        if job["context_ablation"] == "none" and job["seed"] == 11
    )
    monkeypatch.setattr(
        stage2_context_ablation,
        "_candidate_run_dirs",
        lambda *args: (first, second),
    )
    with pytest.raises(ValueError, match="Multiple completed runs"):
        stage2_context_ablation._complete_or_recover_job(job, configuration)


def test_matched_seed_analyzer_outputs_are_deterministic_and_test_embargoed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_path, state = _completed_stage2_state(tmp_path)
    first_run = Path(str(state["jobs"][0]["run_dir"]))
    (first_run / "test_metrics.json").write_text("not valid json", encoding="utf-8")
    (first_run / "test_predictions.parquet").write_bytes(b"not parquet")
    original_read_text = Path.read_text

    def guarded_read_text(path: Path, *args, **kwargs):
        if path.name.startswith("test_"):
            raise AssertionError(f"test artifact accessed: {path}")
        return original_read_text(path, *args, **kwargs)

    original_read_parquet = analyze_stage2_context_ablation.pl.read_parquet

    def guarded_read_parquet(path, *args, **kwargs):
        if Path(path).name.startswith("test_"):
            raise AssertionError(f"test artifact accessed: {path}")
        return original_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(
        analyze_stage2_context_ablation.pl,
        "read_parquet",
        guarded_read_parquet,
    )
    json_path, csv_path = analyze_stage2_context_ablation.analyze_sweep(
        state_path, tmp_path / "analysis"
    )
    summary = json.loads(original_read_text(json_path, encoding="utf-8"))
    table = pl.read_csv(csv_path)
    assert json_path.name == "stage2_context_ablation_summary.json"
    assert csv_path.name == "stage2_context_ablation_summary.csv"
    assert summary["logical_job_count"] == 18
    assert len(summary["runs"]) == 18
    assert table.height == 18
    fixed = next(
        row
        for row in summary["configurations"]
        if row["context_ablation"] == "drop_fixed_di"
    )
    deltas = fixed["primary_delta_across_training_seeds"]
    assert tuple(
        float(deltas["paired_delta_by_seed"][str(seed)]) for seed in STAGE2_SEEDS
    ) == pytest.approx((0.001, 0.001, 0.001))
    assert deltas["mean"] == pytest.approx(0.001)
    assert deltas["minimum"] == pytest.approx(0.001)
    assert deltas["maximum"] == pytest.approx(0.001)
    assert deltas["positive_seed_count"] == 3
    assert deltas["zero_seed_count"] == 0
    assert deltas["negative_seed_count"] == 0
    run = next(
        row
        for row in summary["runs"]
        if row["context_ablation"] == "drop_fixed_di" and row["seed"] == 29
    )
    assert run["periods"]["first_half"][
        "primary_delta_vs_same_seed_baseline"
    ] == pytest.approx(0.001)
    assert run["periods"]["latest_half"][
        "primary_delta_vs_same_seed_baseline"
    ] == pytest.approx(0.001)
    assert run["training"]["optimizer_updates"] == 231
    assert run["within_trained_model_daily_bootstrap"]["replications"] == 10_000
    assert (
        "conclusive" in summary["uncertainty_interpretation"]["across_training_seeds"]
    )


@pytest.mark.parametrize(
    "corruption", ("partial", "duplicate", "origin", "configuration", "score")
)
def test_analyzer_rejects_incomplete_or_ambiguous_state(
    tmp_path: Path, corruption: str
) -> None:
    state_path, state = _completed_stage2_state(tmp_path)
    if corruption == "partial":
        state["status"] = "running"
    elif corruption == "duplicate":
        state["jobs"][1]["run_dir"] = state["jobs"][0]["run_dir"]
    elif corruption == "origin":
        state["jobs"][0]["result_origin"] = "adopted_stage1"
    elif corruption == "configuration":
        state["configuration"]["seeds"] = [29]
    else:
        state["jobs"][0]["primary_validation_ic"] += 1.0
    state_path.write_text(json.dumps(state), encoding="utf-8")
    output = tmp_path / "analysis"
    with pytest.raises(ValueError):
        analyze_stage2_context_ablation.analyze_sweep(state_path, output)
    assert not output.exists()


def test_stage2_source_names_only_validation_artifacts() -> None:
    runner = Path(stage2_context_ablation.__file__).read_text(encoding="utf-8")
    analyzer = Path(analyze_stage2_context_ablation.__file__).read_text(
        encoding="utf-8"
    )
    for forbidden in ("test_metrics", "test_predictions", "select_sample_split"):
        assert forbidden not in runner
        assert forbidden not in analyzer
