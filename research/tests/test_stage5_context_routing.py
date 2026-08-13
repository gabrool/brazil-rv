from __future__ import annotations

import json
from pathlib import Path

import pytest

from brazil_rv.modeling import stage5_context_routing as stage5
from brazil_rv.modeling import train
from brazil_rv.modeling.audit_realized_distributions import AUDIT_JSON
from brazil_rv.modeling.context_routing_adaptive import (
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
from brazil_rv.modeling.contract import ALLOWED_SEEDS, HORIZONS
from brazil_rv.modeling.stage5_context_routing import (
    CompletedRun,
    _ensure_job,
    _new_job,
    _new_state,
    _record_decision,
    _recover_job,
    build_training_command,
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
    assert TOTAL_TRAINING_RUN_COUNT_MINIMUM == 5
    assert TOTAL_TRAINING_RUN_COUNT_MAXIMUM == 12
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
    def fail(path: Path) -> None:
        path.write_text(
            json.dumps(
                {"version": stage5.PREFLIGHT_VERSION, "status": "failed", "steps": 3}
            ),
            encoding="utf-8",
        )
        raise RuntimeError("identity mismatch")

    monkeypatch.setattr(stage5, "run_routing_identity_preflight", fail)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        stage5._ensure_preflight(tmp_path)
    with pytest.raises(RuntimeError, match="did not pass"):
        stage5._ensure_preflight(tmp_path)


def test_dry_run_reports_adaptive_minimum_and_maximum() -> None:
    payload = {
        "incumbent_runs": {"11": "a", "29": "b", "47": "c"},
        "issuer_runs": {"mandatory": 1, "conditional": 2, "minimum": 1, "maximum": 3},
        "routing_runs": {
            "mandatory": 4,
            "conditional_maximum": 5,
            "minimum": 4,
            "maximum": 9,
        },
        "total_training_runs": {"minimum": 5, "maximum": 12},
    }
    output = stage5.format_dry_run(payload)
    assert "routing runs: mandatory=4 conditional_max=5 min=4 max=9" in output
    assert "total training runs: min=5 max=12" in output
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
