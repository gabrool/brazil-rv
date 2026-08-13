from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path

import pytest

from brazil_rv.modeling import stage5_context_routing as stage5
from brazil_rv.modeling import train
from brazil_rv.modeling.audit_realized_distributions import AUDIT_JSON
from brazil_rv.modeling.context_routing_adaptive import (
    CONTROL_RUN_COUNT_MAXIMUM,
    CONTROL_RUN_COUNT_MINIMUM,
    ROUTING_RUN_COUNT_MAXIMUM,
    ROUTING_RUN_COUNT_MINIMUM,
    TOTAL_TRAINING_RUN_COUNT_MAXIMUM,
    TOTAL_TRAINING_RUN_COUNT_MINIMUM,
    joint_synthesis_gate,
    seed29_candidate_gate,
    select_candidate,
    three_seed_candidate_gate,
    within_source_combination_gate,
)
from brazil_rv.modeling.contract import ALLOWED_SEEDS, HORIZONS, HardwareInfo
from brazil_rv.modeling.stage5_context_routing import (
    CompletedRun,
    _ensure_job,
    _new_job,
    _new_state,
    _record_decision,
    _recover_job,
    build_training_command,
    control_job,
    issuer_jobs,
    issuer_seed29_gate,
    issuer_three_seed_gate,
    routing_job,
    routing_jobs,
    routing_stage,
)


def _metrics(
    primary: float,
    horizon_ic: tuple[float, float, float] | None = None,
    spreads: tuple[float, float, float] | None = None,
    turnover: tuple[float, float, float] | None = None,
) -> dict[str, object]:
    horizon_ic = horizon_ic or (primary, primary, primary)
    spreads = spreads or (primary, primary, primary)
    turnover = turnover or (primary, primary, primary)
    return {
        "primary_ic": primary,
        "mean_gross_top_minus_bottom": sum(spreads) / 3,
        "mean_one_way_turnover": sum(turnover) / 3,
        "horizons": {
            f"{horizon}m": {
                "spearman_ic": horizon_ic[index],
                "gross_top_minus_bottom": spreads[index],
                "one_way_turnover": turnover[index],
            }
            for index, horizon in enumerate(HORIZONS)
        },
    }


def _run(
    seed: int,
    metrics: dict[str, object],
    *,
    peer: str = "selected",
    slow: str = "late_only",
    macro: str = "late_only",
    experiment: str = "legacy",
) -> CompletedRun:
    return CompletedRun(
        run_dir=Path(f"run-{seed}-{peer}-{slow}-{macro}"),
        seed=seed,
        peer_features=peer,
        slow_routing=slow,
        macro_temporal_routing=macro,
        context_routing_experiment=experiment,
        producing_git_commit_sha="a" * 40,
        primary_ic=float(metrics["primary_ic"]),
        metrics=metrics,
        output_sha256={"run_manifest.json": "manifest-hash"},
    )


def _gate(
    primary: float,
    *,
    horizons: tuple[float, float, float] | None = None,
    spread: float | None = None,
    turnover: float | None = None,
) -> dict[str, object]:
    return seed29_candidate_gate(
        _metrics(0.0, spreads=(0.0, 0.0, 0.0), turnover=(0.0, 0.0, 0.0)),
        _metrics(
            primary,
            horizon_ic=horizons,
            spreads=(primary if spread is None else spread,) * 3,
            turnover=(primary if turnover is None else turnover,) * 3,
        ),
    )


def _candidate(
    identity: str, slow: str, macro: str, gate: dict[str, object]
) -> dict[str, object]:
    return {
        "identity": identity,
        "slow_routing": slow,
        "macro_temporal_routing": macro,
        "gate": gate,
    }


def _valid_performance_artifacts(
    run_dir: Path, identity: str = "profile-hash"
) -> tuple[dict[str, object], dict[str, str]]:
    run_dir.mkdir(exist_ok=True)
    trace_path = run_dir / stage5.PROFILER_TRACE_FILENAME
    trace_path.write_text(
        json.dumps({"traceEvents": [{"name": "bounded-update", "ph": "X"}]}),
        encoding="utf-8",
    )
    training_phases = {
        name: 0.01 for name in stage5.SAM_BOUNDED_CUDA_PHASES
    }
    validation_phases = {
        name: 0.02 for name in stage5.VALIDATION_BOUNDED_CUDA_PHASES
    }
    profile: dict[str, object] = {
        "version": stage5.PERFORMANCE_PROFILE_VERSION,
        "run_profile": "experiment",
        "run_profile_identity_sha256": identity,
        "measurement_contract": {
            "aggregate_wall_clock_clock": "time.perf_counter",
            "bounded_sampling_enabled": True,
            "bounded_cuda_clock": "torch.cuda.Event",
            "bounded_training_scope": (
                "first_completed_effective_training_update_of_epoch_1"
            ),
            "bounded_validation_scope": "first_validation_batch_of_epoch_1",
            "cuda_synchronization_policy": (
                "one synchronization at each bounded CUDA profiling boundary only"
            ),
            "sampled_cuda_timings_are_not_extrapolated": True,
            "worker_construction_is_sum_across_workers_and_may_overlap": True,
        },
        "epochs": [
            {
                "epoch": 1,
                "aggregate_wall_clock": {
                    "training_seconds": 1.0,
                    "validation_seconds": 2.0,
                    "artifact_io_seconds": 3.0,
                    "epoch_seconds": 6.0,
                },
                "training": {
                    "main_process_dataloader_wait_seconds": 0.1,
                    "worker_batch_construction_seconds_sum": 0.2,
                    "worker_timing_scope": (
                        "sum_across_workers_may_overlap_wall_time"
                    ),
                    "h2d_bytes": 100,
                    "h2d_enqueue_wall_seconds": 0.01,
                    "effective_update_wall_seconds": 0.6,
                    "total_epoch_training_wall_seconds": 1.0,
                    "profiler_trace_artifact_io_wall_seconds": 0.5,
                    "physical_microbatch_count": 8,
                    "effective_update_count": 1,
                },
                "validation": {
                    "main_process_dataloader_wait_seconds": 0.2,
                    "worker_batch_construction_seconds_sum": 0.3,
                    "worker_timing_scope": (
                        "sum_across_workers_may_overlap_wall_time"
                    ),
                    "h2d_bytes": 200,
                    "h2d_enqueue_wall_seconds": 0.02,
                    "device_to_host_and_result_collection_wall_seconds": 0.4,
                    "metric_construction_wall_seconds": 0.5,
                    "total_validation_wall_seconds": 2.0,
                    "batch_count": 2,
                },
                "training_decision_grouping": {
                    "physical_microbatch_unique_decision_counts": [1] * 8,
                    "effective_batch_unique_decision_counts": [3],
                },
                "peak_cuda_memory": {
                    "allocated_bytes": 700,
                    "reserved_bytes": 900,
                },
            }
        ],
        "bounded_training_update": {
            "scope": "first_completed_effective_training_update_of_epoch_1",
            "profiled_epoch": 1,
            "effective_update_index": 0,
            "h2d_bytes": 100,
            "h2d_enqueue_wall_seconds": 0.01,
            "total_effective_update_wall_seconds": 0.6,
            "cuda_phase_seconds": training_phases,
        },
        "bounded_validation_batch": {
            "scope": "first_validation_batch_of_epoch_1",
            "batch_index": 0,
            "h2d_bytes": 100,
            "h2d_enqueue_wall_seconds": 0.01,
            "cuda_phase_seconds": validation_phases,
        },
        "profiler_trace": {
            "filename": stage5.PROFILER_TRACE_FILENAME,
            "scope": "first_completed_effective_training_update_of_epoch_1",
            "sha256": stage5._sha256(trace_path),
            "profiled_epoch": 1,
            "effective_update_index": 0,
            "trace_artifact_io_wall_seconds": 0.5,
        },
        "final_artifact_io_wall_seconds": 4.0,
        "whole_run": {
            "training_wall_seconds": 1.0,
            "validation_wall_seconds": 2.0,
            "artifact_io_wall_seconds": 7.0,
            "run_wall_seconds": 11.0,
            "h2d_bytes": 300,
        },
        "peak_cuda_memory": {
            "allocated_bytes": 700,
            "reserved_bytes": 900,
        },
    }
    first_epoch = profile["epochs"][0]
    for epoch_number in (2, 3):
        epoch = json.loads(json.dumps(first_epoch))
        epoch["epoch"] = epoch_number
        epoch["training"]["profiler_trace_artifact_io_wall_seconds"] = 0.0
        profile["epochs"].append(epoch)
    profile["whole_run"] = {
        "training_wall_seconds": 3.0,
        "validation_wall_seconds": 6.0,
        "artifact_io_wall_seconds": 13.0,
        "run_wall_seconds": 23.0,
        "h2d_bytes": 900,
    }
    profile_path = run_dir / "performance_profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    return profile, {
        "performance_profile.json": stage5._sha256(profile_path),
        stage5.PROFILER_TRACE_FILENAME: stage5._sha256(trace_path),
    }


def test_adaptive_jobs_are_mandatory_only_and_frozen() -> None:
    routing = routing_jobs()
    assert len(routing) == 4
    assert {
        (job["slow_routing"], job["macro_temporal_routing"], job["seed"])
        for job in routing
    } == {
        ("early_concat", "late_only", 29),
        ("film", "late_only", 29),
        ("late_only", "early_concat", 29),
        ("late_only", "film", 29),
    }
    assert stage5.PEER_PRIMARY_STATE_POINTER.name == (
        "peer_primary_matrix_current_path.txt"
    )
    assert (
        stage5.OUTPUT_POINTER.name
        == "context_routing_sequence_outputs_current_path.txt"
    )
    assert all(job["stage"] == "mandatory_seed29_screen" for job in routing)
    assert all("job_id" in job for job in routing)
    assert [job["seed"] for job in issuer_jobs()] == [29, 11, 47]
    assert ROUTING_RUN_COUNT_MINIMUM == 4
    assert ROUTING_RUN_COUNT_MAXIMUM == 9
    assert CONTROL_RUN_COUNT_MINIMUM == 1
    assert CONTROL_RUN_COUNT_MAXIMUM == 3
    assert TOTAL_TRAINING_RUN_COUNT_MINIMUM == 6
    assert TOTAL_TRAINING_RUN_COUNT_MAXIMUM == 15
    assert control_job(29)["command"] == list(
        build_training_command(seed=29, peer_features="selected")
    )
    assert routing_stage("early_concat", "late_only") == "slow_only"
    assert routing_stage("late_only", "film") == "macro_temporal_only"
    assert routing_stage("film", "film") == "joint_synthesis"
    with pytest.raises(ValueError, match="at least one enabled route"):
        routing_job("late_only", "late_only", 29, "forbidden_control")

    with pytest.raises(ValueError, match="never trains"):
        routing_stage("late_only", "late_only")
    parsed = train.parse_args(
        build_training_command(
            seed=29,
            peer_features="selected",
            slow_routing="early_concat",
            macro_temporal_routing="film",
            context_routing_experiment="factorial_v1",
        )[3:]
    )
    assert parsed.run_profile == "experiment"
    assert parsed.context_routing_experiment == "factorial_v1"
    assert parsed.slow_routing == "early_concat"
    assert parsed.macro_temporal_routing == "film"
    assert parsed.peer_features == "selected"
    assert parsed.context_ablation == "drop_win_and_global_non_rates"


def test_factorial_cli_rejects_drift_from_the_frozen_incumbent() -> None:
    command = list(
        build_training_command(
            seed=11,
            peer_features="selected",
            context_routing_experiment="factorial_v1",
        )[3:]
    )
    command[command.index("64")] = "128"
    with pytest.raises(SystemExit):
        train.parse_args(command)


def test_issuer_sequential_gates_use_paired_validation_contract() -> None:
    incumbent29 = _run(29, _metrics(0.0))
    passing29 = _run(
        29,
        _metrics(
            0.10,
            horizon_ic=(0.10, 0.05, -0.01),
            spreads=(0.02, -0.01, 0.03),
            turnover=(0.01, 0.02, 0.03),
        ),
        peer="selected_plus_issuer",
    )
    screen = issuer_seed29_gate(incumbent29, passing29)
    assert screen["passed"] is True
    assert screen["criteria"]["positive_horizon_ic_delta_count"] == 2
    assert screen["criteria"]["gross_spread_deterioration_count"] == 1

    failing29 = _run(
        29,
        _metrics(
            0.10,
            horizon_ic=(0.10, 0.05, -0.01),
            spreads=(-0.02, -0.01, 0.03),
        ),
        peer="selected_plus_issuer",
    )
    assert issuer_seed29_gate(incumbent29, failing29)["passed"] is False

    incumbents = {seed: _run(seed, _metrics(0.0)) for seed in ALLOWED_SEEDS}
    effects = {11: 0.10, 29: 0.10, 47: -0.05}
    issuers = {
        seed: _run(
            seed,
            _metrics(effect, horizon_ic=(effect, effect / 2, -effect / 4)),
            peer="selected_plus_issuer",
        )
        for seed, effect in effects.items()
    }
    confirmation = issuer_three_seed_gate(incumbents, issuers)
    assert confirmation["passed"] is True
    assert confirmation["criteria"]["positive_paired_primary_seed_count"] == 2
    assert confirmation["criteria"]["positive_mean_horizon_effect_count"] == 2


@pytest.mark.parametrize(
    ("treatment", "failed_criterion"),
    [
        (_metrics(0.0), "positive_paired_primary_ic_delta"),
        (
            _metrics(0.1, horizon_ic=(0.1, -0.1, -0.1)),
            "positive_horizon_ic_delta_count",
        ),
        (
            _metrics(0.1, spreads=(-0.1, -0.1, 0.1)),
            "gross_spread_deterioration_count",
        ),
    ],
)
def test_seed29_candidate_gate_fails_each_precommitted_criterion(
    treatment: dict[str, object], failed_criterion: str
) -> None:
    gate = seed29_candidate_gate(_metrics(0.0), treatment)
    assert gate["passed"] is False
    criterion = gate["criteria"][failed_criterion]
    assert criterion is False or criterion < 2 or criterion > 1


def test_three_seed_candidate_gate_fails_seed_and_horizon_replication() -> None:
    controls = {seed: _metrics(0.0) for seed in ALLOWED_SEEDS}
    one_positive_seed = {
        11: _metrics(0.3),
        29: _metrics(-0.1),
        47: _metrics(-0.1),
    }
    seed_failure = three_seed_candidate_gate(controls, one_positive_seed)
    assert seed_failure["criteria"]["mean_paired_primary_effect"] > 0.0
    assert seed_failure["criteria"]["positive_paired_primary_seed_count"] == 1
    assert seed_failure["passed"] is False

    only_one_positive_horizon = {
        11: _metrics(0.1, horizon_ic=(0.1, -0.01, -0.02)),
        29: _metrics(0.1, horizon_ic=(0.1, -0.01, -0.02)),
        47: _metrics(-0.05, horizon_ic=(-0.05, -0.01, -0.02)),
    }
    horizon_failure = three_seed_candidate_gate(controls, only_one_positive_horizon)
    assert horizon_failure["criteria"]["mean_paired_primary_effect"] > 0.0
    assert horizon_failure["criteria"]["positive_paired_primary_seed_count"] == 2
    assert horizon_failure["criteria"]["positive_mean_horizon_effect_count"] == 1
    assert horizon_failure["passed"] is False


def test_new_state_does_not_precreate_conditional_or_mandatory_jobs(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "audit.json"
    preflight = tmp_path / "preflight.json"
    audit.write_text("audit", encoding="utf-8")
    preflight.write_text("preflight", encoding="utf-8")
    incumbents = {
        seed: (_run(seed, _metrics(0.0)), {"source": seed}) for seed in ALLOWED_SEEDS
    }
    state = _new_state({}, incumbents, audit, preflight)
    assert state["issuer_jobs"] == []
    assert state["routing_jobs"] == []
    assert state["decisions"] == []
    specification = routing_jobs()[0]
    job = _ensure_job(state, specification)
    assert state["routing_jobs"] == [job]
    assert job["job_id"] == specification["job_id"]
    assert _ensure_job(state, specification) is job


@pytest.mark.parametrize(
    ("early_passed", "film_passed", "expected"),
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
    ],
)
def test_within_source_gate_covers_every_pass_fail_skip_branch(
    early_passed: bool, film_passed: bool, expected: bool
) -> None:
    early = _gate(0.1 if early_passed else -0.1)
    film = _gate(0.1 if film_passed else -0.1)
    decision = within_source_combination_gate(early, film)
    assert decision["should_run"] is expected
    assert decision["decision"] == ("run" if expected else "skip")
    assert "reason" in decision
    assert decision["inputs"]["early_concat_paired_metrics"] is not None


def test_joint_gate_requires_both_selected_sources() -> None:
    selected = {"selected": {"identity": "candidate"}}
    missing = {"selected": None}
    assert joint_synthesis_gate(selected, selected)["should_run"] is True
    assert joint_synthesis_gate(selected, missing)["should_run"] is False
    assert joint_synthesis_gate(missing, selected)["should_run"] is False
    assert joint_synthesis_gate(missing, missing)["should_run"] is False


def test_selection_persists_metrics_and_deterministic_tiebreak() -> None:
    first = _candidate(
        "slow=early_concat|macro=late_only",
        "early_concat",
        "late_only",
        _gate(0.10, spread=0.02, turnover=0.03),
    )
    better_primary = _candidate(
        "slow=film|macro=late_only",
        "film",
        "late_only",
        _gate(0.11, spread=0.01, turnover=0.10),
    )
    rejected = _candidate(
        "slow=late_only|macro=film",
        "late_only",
        "film",
        _gate(-0.01),
    )
    selection = select_candidate([first, better_primary, rejected])
    assert selection["selected"]["identity"] == better_primary["identity"]
    assert selection["eligible_candidate_count"] == 2
    assert selection["ineligible_candidates"][0]["gate"] == rejected["gate"]
    assert selection["deterministic_tiebreak"][-1] == "ascending_lexical_identity"
    assert selection["ranking"][0]["tiebreak_values"][
        "delta_primary_ic"
    ] == pytest.approx(0.11)

    simple = _candidate("a", "early_concat", "late_only", _gate(0.1))
    complex_candidate = _candidate("b", "early_concat_film", "late_only", _gate(0.1))
    tied = select_candidate([complex_candidate, simple])
    assert tied["selected"]["identity"] == "a"
    assert select_candidate([rejected])["selected"] is None


def test_decision_restart_is_idempotent_and_fails_on_drift() -> None:
    state: dict[str, object] = {"decisions": []}
    payload = {"decision": "skip", "reason": "gate_failed", "metrics": {"x": 1}}
    first = _record_decision(state, "conditional_arm", payload)
    assert _record_decision(state, "conditional_arm", payload) is first
    assert len(state["decisions"]) == 1
    with pytest.raises(ValueError, match="changed on restart"):
        _record_decision(
            state,
            "conditional_arm",
            {"decision": "run", "reason": "drifted"},
        )


def test_preflight_failure_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_identity: dict[str, object] = {}

    def fail(path: Path, identity: dict[str, object]) -> None:
        assert identity is expected_identity
        path.write_text(
            json.dumps(
                {"version": stage5.PREFLIGHT_VERSION, "status": "failed", "steps": 3}
            ),
            encoding="utf-8",
        )
        raise RuntimeError("identity mismatch")

    monkeypatch.setattr(stage5, "run_routing_identity_preflight", fail)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        stage5._ensure_preflight(tmp_path, expected_identity)
    monkeypatch.setattr(
        stage5,
        "validate_runtime",
        lambda: HardwareInfo(
            "GH200",
            (9, 0),
            100 * 1024**3,
            "aarch64",
            "Linux",
            "2.13.0",
            "12.6",
            90100,
        ),
    )
    with pytest.raises(ValueError, match="top-level artifact"):
        stage5._ensure_preflight(tmp_path, expected_identity)


def test_runbook_is_atomic_and_available_before_preflight_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature_store = tmp_path / "feature-store"
    feature_store.mkdir()
    peer_state = tmp_path / "peer-primary.json"
    peer_state.write_text("{}", encoding="utf-8")
    state_dir = tmp_path / "state"
    monkeypatch.setattr(stage5, "_git_identity", lambda **_: ("a" * 40, True))
    monkeypatch.setattr(stage5, "resolve_feature_store", lambda: feature_store)
    profile = stage5.RunProfile(
        name="experiment",
        equity_slots=(0,),
        security_ids=("SEC",),
        symbols=("EQ",),
        decision_indices=(0,),
        maximum_epochs=3,
        minimum_active_equities=1,
        minimum_training_dates=1,
        decision_grouped_batches=True,
        provenance={},
        selection=(),
        identity_sha256="profile",
    )
    monkeypatch.setattr(stage5, "resolve_run_profile", lambda *_: profile)
    rows = type("Rows", (), {"height": 19})()
    monkeypatch.setattr(stage5, "load_sample_index", lambda *_: rows)
    monkeypatch.setattr(
        stage5,
        "prepare_feature_store_session",
        lambda path, *_: path.write_text("{}", encoding="utf-8"),
    )
    monkeypatch.setattr(
        stage5,
        "validate_session_preparation",
        lambda *_: (object(), object()),
    )
    monkeypatch.setattr(stage5, "filter_profile_rows", lambda rows, *_: rows)
    monkeypatch.setattr(
        stage5, "select_sample_split", lambda *_: type("Rows", (), {"height": 19})()
    )
    monkeypatch.setattr(stage5, "build_routing_preflight_identity", lambda *_: {})
    monkeypatch.setattr(stage5, "_feature_store_identity", lambda _: {})
    monkeypatch.setattr(stage5, "_configuration", lambda *_: {})
    monkeypatch.setattr(stage5, "_source_incumbents", lambda *_: {})
    monkeypatch.setattr(stage5, "exclusive_process_lock", lambda *_: nullcontext())

    def fail_preflight(path: Path, identity: dict[str, object]) -> Path:
        assert identity == {}
        runbook = path / stage5.RUNBOOK
        assert runbook.is_file()
        text = runbook.read_text(encoding="utf-8")
        assert "## Ubuntu/bash" in text
        assert "# Launch." in text
        assert "# Resume" in text
        assert "--status" in text
        assert "OUTPUT_POINTER=" in text
        assert "sha256sum --check" in text
        assert "tar -tzf" in text
        assert stage5.ARCHIVE_NAME.endswith(".tar.gz")
        raise RuntimeError("preflight stopped")

    monkeypatch.setattr(stage5, "_ensure_preflight", fail_preflight)
    with pytest.raises(RuntimeError, match="preflight stopped"):
        stage5.run_experiment(state_dir, peer_state)
    assert (state_dir / stage5.RUNBOOK).is_file()
    assert not (state_dir / f"{stage5.RUNBOOK}.tmp").exists()
    assert not (state_dir / stage5.SESSION_PREPARATION_FILENAME).exists()
    assert not (state_dir / "run_profile.json").exists()
    assert not (state_dir / "realized_distribution_audit").exists()
    assert not (state_dir / "state.json").exists()


def test_stage5_call_order_is_preflight_then_session_audit_and_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    feature_store = tmp_path / "feature-store"
    feature_store.mkdir()
    peer_state = tmp_path / "peer-primary.json"
    peer_state.write_text("{}", encoding="utf-8")
    state_dir = tmp_path / "state"

    class Column:
        def to_list(self) -> list[int]:
            return list(range(19))

    class Rows:
        height = 19

        def get_column(self, _: str) -> Column:
            return Column()

        def is_empty(self) -> bool:
            return False

    rows = Rows()
    profile = stage5.RunProfile(
        name="experiment",
        equity_slots=(0,),
        security_ids=("SEC",),
        symbols=("EQ",),
        decision_indices=(0,),
        maximum_epochs=3,
        minimum_active_equities=1,
        minimum_training_dates=1,
        decision_grouped_batches=True,
        provenance={},
        selection=(),
        identity_sha256="profile",
    )
    monkeypatch.setattr(stage5, "_git_identity", lambda **_: ("a" * 40, True))
    monkeypatch.setattr(stage5, "resolve_feature_store", lambda: feature_store)
    monkeypatch.setattr(stage5, "resolve_run_profile", lambda *_: profile)
    monkeypatch.setattr(stage5, "load_sample_index", lambda *_: rows)
    monkeypatch.setattr(stage5, "select_sample_split", lambda *_: rows)
    filter_calls = 0

    def filter_rows(*_: object, **__: object) -> Rows:
        nonlocal filter_calls
        filter_calls += 1
        if filter_calls == 2:
            calls.append("inputs")
        return rows

    monkeypatch.setattr(stage5, "filter_profile_rows", filter_rows)
    monkeypatch.setattr(stage5, "build_routing_preflight_identity", lambda *_: {})
    monkeypatch.setattr(stage5, "exclusive_process_lock", lambda *_: nullcontext())

    def preflight(path: Path, _: dict[str, object]) -> Path:
        assert (path / stage5.RUNBOOK).is_file()
        calls.extend(("runbook", "preflight"))
        artifact = path / stage5.PREFLIGHT_JSON
        artifact.write_text("{}", encoding="utf-8")
        return artifact

    monkeypatch.setattr(stage5, "_ensure_preflight", preflight)

    def prepare(path: Path, *_: object) -> None:
        calls.append("session")
        path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(stage5, "prepare_feature_store_session", prepare)
    monkeypatch.setattr(
        stage5, "validate_session_preparation", lambda *_: (rows, object())
    )
    monkeypatch.setattr(stage5, "_feature_store_identity", lambda _: {})

    def audit(path: Path, *_: object) -> Path:
        calls.append("audit")
        artifact = path / "audit.json"
        artifact.write_text("{}", encoding="utf-8")
        return artifact

    monkeypatch.setattr(stage5, "_ensure_audit", audit)
    monkeypatch.setattr(stage5, "_configuration", lambda *_: {})
    monkeypatch.setattr(stage5, "_source_incumbents", lambda *_: {})
    monkeypatch.setattr(
        stage5, "_load_state", lambda *_: {"status": "running"}
    )
    monkeypatch.setattr(stage5, "_persist_state", lambda *_: None)
    monkeypatch.setattr(stage5, "_log", lambda *_: None)
    monkeypatch.setattr(stage5, "_finalize_outputs", lambda *_: None)
    monkeypatch.setattr(
        stage5,
        "_run_adaptive_sequence",
        lambda *_: calls.append("training"),
    )

    stage5.run_experiment(state_dir, peer_state)

    assert calls == ["runbook", "preflight", "session", "inputs", "audit", "training"]


def test_dry_run_uses_only_minimal_preflight_identity_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature_store = tmp_path / "feature-store"
    peer_state = tmp_path / "peer-primary.json"
    feature_store.mkdir()
    peer_state.write_text("{}", encoding="utf-8")
    profile = stage5.RunProfile(
        name="experiment",
        equity_slots=(0,),
        security_ids=("SEC",),
        symbols=("EQ",),
        decision_indices=(0,),
        maximum_epochs=3,
        minimum_active_equities=1,
        minimum_training_dates=1,
        decision_grouped_batches=True,
        provenance={},
        selection=(),
        identity_sha256="profile",
    )
    rows = type("Rows", (), {"height": 19})()
    monkeypatch.setattr(stage5, "_git_identity", lambda **_: ("a" * 40, True))
    monkeypatch.setattr(stage5, "resolve_feature_store", lambda: feature_store)
    monkeypatch.setattr(stage5, "resolve_run_profile", lambda *_: profile)
    monkeypatch.setattr(stage5, "load_sample_index", lambda *_: rows)
    monkeypatch.setattr(stage5, "select_sample_split", lambda *_: rows)
    monkeypatch.setattr(stage5, "filter_profile_rows", lambda *_: rows)
    monkeypatch.setattr(
        stage5, "build_routing_preflight_identity", lambda *_: {"identity": "ok"}
    )
    monkeypatch.setattr(
        stage5,
        "_source_incumbents",
        lambda *_: (_ for _ in ()).throw(AssertionError("incumbents resolved")),
    )
    monkeypatch.setattr(
        stage5,
        "_feature_store_identity",
        lambda *_: (_ for _ in ()).throw(AssertionError("full identity resolved")),
    )

    payload = stage5.dry_run_payload(peer_state)

    assert payload["routing_identity_preflight"] == {"identity": "ok"}
    assert payload["execution_order"][3] == "routing_identity_preflight"
    assert payload["execution_order"][4] == "session_validation_and_cache_warmup"


def test_dry_run_reports_adaptive_minimum_and_maximum() -> None:
    payload = {
        "execution_order": [
            "runbook",
            "experiment_lock",
            "minimal_profile_and_preflight_identity",
            "routing_identity_preflight",
            "session_validation_and_cache_warmup",
            "train_and_validation_input_filtering",
            "realized_distribution_audit",
            "incumbent_resolution",
            "adaptive_training_sequence",
        ],
        "control_runs": {"minimum": 1, "maximum": 3},
        "issuer_runs": {"mandatory": 1, "conditional": 2, "minimum": 1, "maximum": 3},
        "routing_runs": {
            "mandatory": 4,
            "conditional_maximum": 5,
            "minimum": 4,
            "maximum": 9,
        },
        "total_training_runs": {"minimum": 6, "maximum": 15},
    }
    output = stage5.format_dry_run(payload)
    assert (
        "runbook -> experiment_lock -> minimal_profile_and_preflight_identity "
        "-> routing_identity_preflight -> session_validation_and_cache_warmup "
        "-> train_and_validation_input_filtering -> realized_distribution_audit "
        "-> incumbent_resolution -> adaptive_training_sequence"
    ) in output
    assert "routing runs: mandatory=4 conditional_max=5 min=4 max=9" in output
    assert "matched control runs: min=1 max=3" in output
    assert "total training runs: min=6 max=15" in output
    assert "all-off scaffold control training: no" in output
    assert "held-out test accessed: no" in output


def test_running_job_recovers_only_one_validated_completed_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specification = routing_jobs()[0]
    job = _new_job(specification)
    job["status"] = "running"
    completed = _run(
        int(specification["seed"]),
        _metrics(0.12),
        slow=str(specification["slow_routing"]),
        macro=str(specification["macro_temporal_routing"]),
        experiment="factorial_v1",
    )
    monkeypatch.setattr(stage5, "_candidate_runs", lambda *_: (completed,))
    recovered = _recover_job(job, tmp_path, "a" * 40)
    assert recovered is completed
    assert job["status"] == "completed"
    assert job["recovery_count"] == 1
    assert job["run_manifest_sha256"] == "manifest-hash"
    assert _recover_job(job, tmp_path, "a" * 40) is completed

    pending = _new_job(specification)
    with pytest.raises(ValueError, match="contaminates a pending"):
        _recover_job(pending, tmp_path, "a" * 40)

    ambiguous = _new_job(specification)
    ambiguous["status"] = "running"
    monkeypatch.setattr(stage5, "_candidate_runs", lambda *_: (completed, completed))
    with pytest.raises(ValueError, match="Multiple completed runs"):
        _recover_job(ambiguous, tmp_path, "a" * 40)


def test_audit_directory_is_promoted_once_and_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    calls = []

    def fake_run(store: Path, output: Path) -> Path:
        calls.append((store, output))
        output.mkdir()
        audit_path = output / AUDIT_JSON
        audit_path.write_text("{}", encoding="utf-8")
        return audit_path

    validated = []
    monkeypatch.setattr(stage5, "run_realized_distribution_audit", fake_run)
    monkeypatch.setattr(
        stage5,
        "validate_realized_distribution_audit",
        lambda path, identity: validated.append((path, identity)),
    )
    feature_store = tmp_path / "store"
    identity = {"manifest_sha256": "synthetic"}
    first = stage5._ensure_audit(state_dir, feature_store, identity)
    second = stage5._ensure_audit(state_dir, feature_store, identity)
    assert first == second == state_dir / "realized_distribution_audit" / AUDIT_JSON
    assert first.is_file()
    assert len(calls) == 1
    assert len(validated) == 2


def test_session_performance_summary_includes_all_artifact_io(tmp_path: Path) -> None:
    session = tmp_path / "session_preparation.json"
    session.write_text(
        json.dumps(
            {
                "cache_warmup": {"mode": "selected", "seconds": 0.25},
                "performance": {"total_preparation_seconds": 0.5},
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    _, output_hashes = _valid_performance_artifacts(run_dir)
    state = {
        "control_jobs": [
            {
                "status": "completed",
                "job_id": "control-seed-29",
                "run_dir": str(run_dir),
                "output_sha256": output_hashes,
            }
        ],
        "issuer_jobs": [],
        "routing_jobs": [],
        "configuration": {
            "run_profile_identity_sha256": "profile-hash",
            "session_preparation": {"path": str(session)},
        },
        "session_phase_performance": {"routing_identity_preflight_seconds": 5.0},
    }
    payload = stage5._session_performance_payload(state)
    assert payload["version"] == "B3_SESSION_PERFORMANCE_SUMMARY_V2"
    assert payload["additive_totals"] == {
        "training_wall_seconds": 3.0,
        "validation_wall_seconds": 6.0,
        "artifact_io_wall_seconds": 13.0,
        "run_wall_seconds": 23.0,
        "h2d_bytes": 900,
    }
    assert payload["run_count"] == 1
    assert payload["runs"][0]["bounded_training_update"]["scope"] == (
        "first_completed_effective_training_update_of_epoch_1"
    )
    assert "sampled CUDA timings are not aggregated" in payload[
        "bounded_sample_policy"
    ]


@pytest.mark.parametrize(
    "drift",
    (
        "missing_field",
        "negative",
        "nonfinite",
        "identity",
        "trace_hash",
        "trace_structure",
    ),
)
def test_performance_and_trace_validation_fail_closed(
    tmp_path: Path, drift: str
) -> None:
    run_dir = tmp_path / drift
    profile, _ = _valid_performance_artifacts(run_dir)
    if drift == "missing_field":
        del profile["epochs"][0]["validation"]["metric_construction_wall_seconds"]
    elif drift == "negative":
        profile["epochs"][0]["training"]["h2d_enqueue_wall_seconds"] = -0.1
    elif drift == "nonfinite":
        profile["epochs"][0]["validation"]["total_validation_wall_seconds"] = (
            float("nan")
        )
    elif drift == "identity":
        profile["run_profile_identity_sha256"] = "drifted"
    elif drift == "trace_hash":
        profile["profiler_trace"]["sha256"] = "0" * 64
    else:
        trace_path = run_dir / stage5.PROFILER_TRACE_FILENAME
        trace_path.write_text("[]", encoding="utf-8")
        profile["profiler_trace"]["sha256"] = stage5._sha256(trace_path)
    (run_dir / "performance_profile.json").write_text(
        json.dumps(profile), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        stage5._validate_performance_profile(run_dir, "profile-hash")


def test_experiment_outputs_require_hashed_bounded_profiler_trace() -> None:
    assert stage5.PROFILER_TRACE_FILENAME in stage5._EXPERIMENT_REQUIRED_OUTPUTS
    assert stage5.PROFILER_TRACE_FILENAME not in stage5._archive_members()
