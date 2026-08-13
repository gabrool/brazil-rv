from __future__ import annotations

from pathlib import Path

import pytest

from brazil_rv.modeling import stage5_context_routing as stage5
from brazil_rv.modeling import train
from brazil_rv.modeling.audit_realized_distributions import AUDIT_JSON
from brazil_rv.modeling.contract import ALLOWED_SEEDS, CONTEXT_ROUTING_MODES, HORIZONS
from brazil_rv.modeling.stage5_context_routing import (
    CompletedRun,
    build_training_command,
    issuer_jobs,
    issuer_seed29_gate,
    issuer_three_seed_gate,
    _new_job,
    _recover_job,
    routing_jobs,
    routing_stage,
    routing_summary,
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


def test_job_matrices_are_complete_staged_and_frozen() -> None:
    routing = routing_jobs()
    assert len(routing) == 48
    identities = {
        (job["slow_routing"], job["macro_temporal_routing"], job["seed"])
        for job in routing
    }
    assert identities == {
        (slow, macro, seed)
        for slow in CONTEXT_ROUTING_MODES
        for macro in CONTEXT_ROUTING_MODES
        for seed in ALLOWED_SEEDS
    }
    assert [job["stage"] for job in routing[:3]] == ["scaffold_control"] * 3
    assert routing_stage("early_concat", "late_only") == "slow_only"
    assert routing_stage("late_only", "film") == "macro_temporal_only"
    assert routing_stage("film", "film") == "joint_factorial"
    assert [job["seed"] for job in issuer_jobs()] == [29, 11, 47]

    factorial_command = build_training_command(
        seed=29,
        peer_features="selected",
        slow_routing="early_concat",
        macro_temporal_routing="film",
        context_routing_experiment="factorial_v1",
    )
    parsed = train.parse_args(factorial_command[3:])
    assert parsed.context_routing_experiment == "factorial_v1"
    assert parsed.slow_routing == "early_concat"
    assert parsed.macro_temporal_routing == "film"
    assert parsed.peer_features == "selected"
    assert parsed.context_ablation == "drop_win_and_global_non_rates"

    legacy_command = build_training_command(
        seed=29, peer_features="selected_plus_issuer"
    )
    legacy = train.parse_args(legacy_command[3:])
    assert legacy.context_routing_experiment == "legacy"
    assert legacy.slow_routing == legacy.macro_temporal_routing == "late_only"


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
    assert screen["transaction_cost_modeling"] is False

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
            _metrics(
                effect,
                horizon_ic=(effect, effect / 2, -effect / 4),
            ),
            peer="selected_plus_issuer",
        )
        for seed, effect in effects.items()
    }
    confirmation = issuer_three_seed_gate(incumbents, issuers)
    assert confirmation["passed"] is True
    assert confirmation["criteria"]["positive_paired_primary_seed_count"] == 2
    assert confirmation["criteria"]["positive_mean_horizon_effect_count"] == 2


def _route_flags(route: str) -> tuple[int, int]:
    return (
        int(route in {"early_concat", "early_concat_film"}),
        int(route in {"film", "early_concat_film"}),
    )


def test_routing_summary_reports_independent_factorial_effects() -> None:
    coefficients = (0.01, 0.02, 0.03, 0.04)
    incumbents = {seed: _run(seed, _metrics(seed / 1000)) for seed in ALLOWED_SEEDS}
    runs = {}
    for slow in CONTEXT_ROUTING_MODES:
        slow_early, slow_film = _route_flags(slow)
        for macro in CONTEXT_ROUTING_MODES:
            macro_early, macro_film = _route_flags(macro)
            effect = sum(
                value * flag
                for value, flag in zip(
                    coefficients,
                    (slow_early, slow_film, macro_early, macro_film),
                    strict=True,
                )
            )
            for seed in ALLOWED_SEEDS:
                score = seed / 1000 + effect
                runs[(slow, macro, seed)] = _run(
                    seed,
                    _metrics(score),
                    slow=slow,
                    macro=macro,
                    experiment="factorial_v1",
                )

    summary = routing_summary(incumbents, runs)
    assert summary["run_count"] == 48
    assert summary["winner_selected"] is False
    expected = {
        "slow_early_concat": 0.01,
        "slow_film": 0.02,
        "macro_temporal_early_concat": 0.03,
        "macro_temporal_film": 0.04,
    }
    for name, value in expected.items():
        effect = summary["independent_main_effects"][name]
        assert effect["matched_comparison_count"] == 24
        assert effect["mean_delta_primary_ic"] == pytest.approx(value)
        assert effect["positive_primary_comparison_count"] == 24


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
    assert job["output_sha256"] == completed.output_sha256
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
