from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from .analyze_stage2_context_ablation import (
    _period_metrics,
    _read_validation_daily_metrics,
    _validate_metrics_json,
)
from .audit_realized_distributions import (
    AUDIT_JSON,
    AUDIT_NAME,
    AUDIT_VERSION,
    run_realized_distribution_audit,
    validate_realized_distribution_audit,
)
from .context_ablation import resolve_context_ablation_for_store
from .contract import (
    ALLOWED_SEEDS,
    CONTEXT_ROUTING_MODES,
    FEATURE_CONTRACT_VERSION,
    HORIZONS,
    PROJECT_ROOT,
    RUN_OUTPUT_BASE,
    VALIDATION_END,
    VALIDATION_START,
    TCNArchitecture,
    TCNSettings,
    architecture_for_model,
    context_routing_metadata,
    expected_trainable_parameter_count,
    peer_feature_metadata,
)
from .data import resolve_feature_store, validate_feature_store
from .engine import objective_metadata, sam_metadata
from .evaluate import _validate_run_checkpoint_identity
from .feature_ablation import resolve_feature_ablation_for_store
from .process_lock import (
    PRODUCTION_TRAINING_LOCK,
    active_lock_owner,
    exclusive_process_lock,
)
from .stage2_context_ablation import _feature_store_identity
from .stage3_context_addition import (
    _feature_identities_equivalent,
    _reject_test_derived_metadata,
)

EXPERIMENT_NAME = "stage5_context_routing_factorial_v1"
STATE_VERSION = 1
PEER_PRIMARY_STATE_POINTER = (
    RUN_OUTPUT_BASE / "_ops" / "peer_primary_matrix_canonical_path.txt"
)
FROZEN_CONTEXT_ABLATION = "drop_win_and_global_non_rates"
FROZEN_FEATURE_ABLATION = "none"
FROZEN_TCN_BASE = {
    "fusion": "context_pooled",
    "width": 64,
    "receptive_field": "full",
    "block": "swiglu",
}
ROUTING_MODE_ORDER = CONTEXT_ROUTING_MODES
ISSUER_SEED_ORDER = (29, 11, 47)
ROUTING_SEEDS = ALLOWED_SEEDS
_REQUIRED_OUTPUTS = (
    "run_manifest.json",
    "best.pt",
    "final.pt",
    "history.csv",
    "validation_metrics.json",
    "validation_daily_metrics.parquet",
)
_REPOSITORY = PROJECT_ROOT / "quant" / "b3-quant"
_RESEARCH = _REPOSITORY / "research"
SUMMARY_JSON = "stage5_context_routing_summary.json"


@dataclass(frozen=True)
class CompletedRun:
    run_dir: Path
    seed: int
    peer_features: str
    slow_routing: str
    macro_temporal_routing: str
    context_routing_experiment: str
    producing_git_commit_sha: str
    primary_ic: float
    metrics: dict[str, object]
    output_sha256: dict[str, str]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _git_identity(*, require_clean: bool) -> tuple[str, bool]:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=_REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    clean = not bool(status)
    if require_clean and not clean:
        raise RuntimeError("Stage-5 execution requires a clean Git worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, clean


def resolve_peer_primary_state(explicit: Path | None = None) -> Path:
    source = PEER_PRIMARY_STATE_POINTER if explicit is None else explicit
    source = source.expanduser().resolve()
    if source.suffix.lower() == ".txt":
        if not source.is_file():
            raise FileNotFoundError(
                f"Peer-primary state pointer does not exist: {source}"
            )
        target = Path(source.read_text(encoding="utf-8").strip()).expanduser()
        if not target.is_absolute():
            target = source.parent / target
        source = target.resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"Authoritative peer-primary state does not exist: {source}"
        )
    return source


def _settings(
    slow_routing: str = "late_only",
    macro_temporal_routing: str = "late_only",
    experiment: str = "legacy",
) -> TCNSettings:
    return TCNSettings(
        **FROZEN_TCN_BASE,
        slow_routing=slow_routing,
        macro_temporal_routing=macro_temporal_routing,
        context_routing_experiment=experiment,
    )


def build_training_command(
    *,
    seed: int,
    peer_features: str,
    slow_routing: str = "late_only",
    macro_temporal_routing: str = "late_only",
    context_routing_experiment: str = "legacy",
) -> tuple[str, ...]:
    if seed not in ALLOWED_SEEDS:
        raise ValueError("Training seed is outside the frozen experiment contract")
    settings = _settings(
        slow_routing, macro_temporal_routing, context_routing_experiment
    )
    architecture_for_model("tcn", settings)
    if peer_features not in {"selected", "selected_plus_issuer"}:
        raise ValueError("Stage-5 supports only selected peer representations")
    if context_routing_experiment == "factorial_v1" and peer_features != "selected":
        raise ValueError("Routing experiments must keep peer_features=selected")
    command = [
        sys.executable,
        "-m",
        "brazil_rv.modeling.train",
        "--model",
        "tcn",
        "--tcn-fusion",
        settings.fusion,
        "--tcn-width",
        str(settings.width),
        "--tcn-receptive-field",
        settings.receptive_field,
        "--tcn-block",
        settings.block,
        "--peer-features",
        peer_features,
        "--optimizer",
        "sam_adamw",
        "--objective",
        "soft_spearman",
        "--soft-rank-temperature",
        "0.50",
        "--sam-rho",
        "0.125",
        "--global-context",
        "enabled",
        "--context-ablation",
        FROZEN_CONTEXT_ABLATION,
        "--feature-ablation",
        FROZEN_FEATURE_ABLATION,
        "--seed",
        str(seed),
    ]
    if context_routing_experiment == "factorial_v1":
        command.extend(
            (
                "--context-routing-experiment",
                context_routing_experiment,
                "--slow-routing",
                slow_routing,
                "--macro-temporal-routing",
                macro_temporal_routing,
            )
        )
    return tuple(command)


def routing_stage(slow_routing: str, macro_temporal_routing: str) -> str:
    if slow_routing == "late_only" and macro_temporal_routing == "late_only":
        return "scaffold_control"
    if slow_routing != "late_only" and macro_temporal_routing == "late_only":
        return "slow_only"
    if slow_routing == "late_only" and macro_temporal_routing != "late_only":
        return "macro_temporal_only"
    return "joint_factorial"


def routing_jobs() -> tuple[dict[str, object], ...]:
    jobs: list[dict[str, object]] = []
    combinations = [
        ("late_only", "late_only"),
        *((slow, "late_only") for slow in ROUTING_MODE_ORDER[1:]),
        *(("late_only", macro) for macro in ROUTING_MODE_ORDER[1:]),
        *(
            (slow, macro)
            for slow in ROUTING_MODE_ORDER[1:]
            for macro in ROUTING_MODE_ORDER[1:]
        ),
    ]
    if len(combinations) != 16 or len(set(combinations)) != 16:
        raise RuntimeError("Routing matrix must contain the complete 4x4 factorial")
    for slow, macro in combinations:
        for seed in ROUTING_SEEDS:
            specification = {
                "job_kind": "routing",
                "stage": routing_stage(slow, macro),
                "seed": seed,
                "peer_features": "selected",
                "slow_routing": slow,
                "macro_temporal_routing": macro,
                "context_routing_experiment": "factorial_v1",
                "command": list(
                    build_training_command(
                        seed=seed,
                        peer_features="selected",
                        slow_routing=slow,
                        macro_temporal_routing=macro,
                        context_routing_experiment="factorial_v1",
                    )
                ),
            }
            serialized = json.dumps(
                specification, sort_keys=True, separators=(",", ":")
            )
            jobs.append(
                {
                    **specification,
                    "serialized_job_specification": serialized,
                    "job_specification_sha256": hashlib.sha256(
                        serialized.encode()
                    ).hexdigest(),
                }
            )
    return tuple(jobs)


def issuer_jobs() -> tuple[dict[str, object], ...]:
    jobs = []
    for seed in ISSUER_SEED_ORDER:
        specification = {
            "job_kind": "issuer",
            "stage": "seed29_screen" if seed == 29 else "matched_seed_confirmation",
            "seed": seed,
            "peer_features": "selected_plus_issuer",
            "slow_routing": "late_only",
            "macro_temporal_routing": "late_only",
            "context_routing_experiment": "legacy",
            "command": list(
                build_training_command(seed=seed, peer_features="selected_plus_issuer")
            ),
        }
        serialized = json.dumps(specification, sort_keys=True, separators=(",", ":"))
        jobs.append(
            {
                **specification,
                "serialized_job_specification": serialized,
                "job_specification_sha256": hashlib.sha256(
                    serialized.encode()
                ).hexdigest(),
            }
        )
    return tuple(jobs)


def _artifact_hashes(run_dir: Path) -> dict[str, str]:
    hashes = {}
    for name in _REQUIRED_OUTPUTS:
        path = run_dir / name
        if not path.is_file():
            raise ValueError(f"Completed run is missing required artifact: {path}")
        hashes[name] = _sha256(path)
    return hashes


def _manifest_peer_mode(manifest: dict[str, object]) -> str | None:
    peer = manifest.get("peer_features")
    return str(peer.get("mode")) if isinstance(peer, dict) else None


def validate_completed_run(
    run_dir: Path,
    feature_store: Path,
    *,
    seed: int,
    peer_features: str,
    slow_routing: str,
    macro_temporal_routing: str,
    context_routing_experiment: str,
    producing_commit: str | None = None,
) -> CompletedRun:
    run_dir = run_dir.resolve()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    settings = _settings(
        slow_routing, macro_temporal_routing, context_routing_experiment
    )
    architecture = architecture_for_model("tcn", settings)
    if not isinstance(architecture, TCNArchitecture):
        raise AssertionError("Frozen TCN settings did not resolve a TCN architecture")
    expected_context = resolve_context_ablation_for_store(
        feature_store, FROZEN_CONTEXT_ABLATION
    ).metadata()
    expected_feature = resolve_feature_ablation_for_store(
        feature_store, FROZEN_FEATURE_ABLATION
    ).metadata()
    recorded_settings = manifest.get("tcn_settings")
    try:
        normalized_settings = (
            TCNSettings(**recorded_settings)
            if isinstance(recorded_settings, dict)
            else None
        )
    except TypeError as error:
        raise ValueError("Completed run has invalid TCN settings") from error
    expected_commit = manifest.get("git_commit_sha")
    if (
        manifest.get("status") != "completed"
        or manifest.get("model_name") != "tcn"
        or manifest.get("model_family") != "tcn"
        or manifest.get("seed") != seed
        or normalized_settings != settings
        or manifest.get("global_context") != "enabled"
        or manifest.get("context_ablation") != expected_context
        or manifest.get("feature_ablation") != expected_feature
        or _manifest_peer_mode(manifest) != peer_features
        or manifest.get("peer_features")
        != peer_feature_metadata("tcn", architecture, peer_features)
        or manifest.get("objective") != objective_metadata("soft_spearman", 0.50)
        or manifest.get("optimizer_variant") != "sam_adamw"
        or manifest.get("sam") != sam_metadata("sam_adamw", 0.125)
        or manifest.get("parameter_count")
        != expected_trainable_parameter_count("tcn", architecture, peer_features)
        or not isinstance(expected_commit, str)
        or (producing_commit is not None and expected_commit != producing_commit)
    ):
        raise ValueError(
            f"Completed run violates the frozen Stage-5 identity: {run_dir}"
        )
    if context_routing_experiment == "factorial_v1":
        if manifest.get("context_routing") != context_routing_metadata(architecture):
            raise ValueError("Factorial run has invalid routing metadata")
    else:
        recorded_routing = manifest.get("context_routing")
        if (
            recorded_routing is not None
            and recorded_routing != context_routing_metadata(architecture)
        ):
            raise ValueError("Legacy run has invalid routing metadata")
    recorded_store = (
        Path(str(manifest["resolved_feature_store_path"])).expanduser().resolve()
    )
    try:
        same_store = recorded_store.samefile(feature_store)
    except OSError:
        same_store = recorded_store == feature_store
    if not same_store:
        raise ValueError("Completed run uses another resolved feature store")

    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    final = torch.load(run_dir / "final.pt", map_location="cpu", weights_only=False)
    if not isinstance(best, dict) or not isinstance(final, dict):
        raise ValueError("Completed run checkpoints are malformed")
    _validate_run_checkpoint_identity(manifest, best, feature_store, run_dir)
    _validate_run_checkpoint_identity(manifest, final, feature_store, run_dir)
    daily = _read_validation_daily_metrics(run_dir)
    metrics = _period_metrics(daily, VALIDATION_START, VALIDATION_END)
    _validate_metrics_json(run_dir, metrics)
    primary = float(metrics["primary_ic"])
    if not math.isclose(
        primary,
        float(manifest["best_validation_primary_score"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Completed run manifest and validation metrics disagree")
    return CompletedRun(
        run_dir=run_dir,
        seed=seed,
        peer_features=peer_features,
        slow_routing=slow_routing,
        macro_temporal_routing=macro_temporal_routing,
        context_routing_experiment=context_routing_experiment,
        producing_git_commit_sha=str(expected_commit),
        primary_ic=primary,
        metrics=metrics,
        output_sha256=_artifact_hashes(run_dir),
    )


def _source_incumbents(
    state_path: Path,
    feature_store: Path,
) -> dict[int, tuple[CompletedRun, dict[str, object]]]:
    raw = state_path.read_bytes()
    state = json.loads(raw)
    _reject_test_derived_metadata(state, "peer-primary source state")
    if state.get("status") != "completed" or not isinstance(state.get("jobs"), list):
        raise ValueError(
            "Peer-primary adoption requires a completed authoritative state"
        )
    candidates: dict[int, list[tuple[CompletedRun, dict[str, object]]]] = {
        seed: [] for seed in ALLOWED_SEEDS
    }
    for position, job in enumerate(state["jobs"]):
        if not isinstance(job, dict) or job.get("status") != "completed":
            continue
        run_value = job.get("run_dir")
        if not isinstance(run_value, str):
            continue
        run_dir = Path(run_value).expanduser().resolve()
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        seed = manifest.get("seed")
        if seed not in candidates or _manifest_peer_mode(manifest) != "selected":
            continue
        try:
            completed = validate_completed_run(
                run_dir,
                feature_store,
                seed=int(seed),
                peer_features="selected",
                slow_routing="late_only",
                macro_temporal_routing="late_only",
                context_routing_experiment="legacy",
            )
        except (KeyError, OSError, TypeError, ValueError):
            continue
        manifest_hash = completed.output_sha256["run_manifest.json"]
        if (
            job.get("run_manifest_sha256") is not None
            and job.get("run_manifest_sha256") != manifest_hash
        ):
            raise ValueError("Authoritative peer-primary job manifest hash disagrees")
        candidates[int(seed)].append(
            (
                completed,
                {
                    "position": position,
                    "run_dir": str(run_dir),
                    "run_manifest_sha256": manifest_hash,
                    "source_job_identity": {
                        key: job.get(key)
                        for key in (
                            "logical_configuration",
                            "configuration",
                            "seed",
                            "result_origin",
                            "producing_git_commit_sha",
                        )
                        if key in job
                    },
                },
            )
        )
    if any(len(values) != 1 for values in candidates.values()):
        counts = {seed: len(values) for seed, values in candidates.items()}
        raise ValueError(
            "Authoritative peer-primary state must identify exactly one frozen "
            f"selected incumbent per seed: {counts}"
        )
    return {seed: values[0] for seed, values in candidates.items()}


def _metric_delta(
    treatment: dict[str, object], control: dict[str, object]
) -> dict[str, object]:
    treatment_horizons = treatment["horizons"]
    control_horizons = control["horizons"]
    if not isinstance(treatment_horizons, dict) or not isinstance(
        control_horizons, dict
    ):
        raise ValueError("Validation horizon metrics are malformed")
    horizons = {}
    for horizon in (f"{value}m" for value in HORIZONS):
        current = treatment_horizons[horizon]
        baseline = control_horizons[horizon]
        horizons[horizon] = {
            "control_ic": baseline["spearman_ic"],
            "treatment_ic": current["spearman_ic"],
            "delta_ic": float(current["spearman_ic"]) - float(baseline["spearman_ic"]),
            "control_gross_spread": baseline["gross_top_minus_bottom"],
            "treatment_gross_spread": current["gross_top_minus_bottom"],
            "delta_gross_spread": float(current["gross_top_minus_bottom"])
            - float(baseline["gross_top_minus_bottom"]),
            "control_turnover": baseline["one_way_turnover"],
            "treatment_turnover": current["one_way_turnover"],
            "delta_turnover": float(current["one_way_turnover"])
            - float(baseline["one_way_turnover"]),
        }
    return {
        "control_primary_ic": control["primary_ic"],
        "treatment_primary_ic": treatment["primary_ic"],
        "delta_primary_ic": float(treatment["primary_ic"])
        - float(control["primary_ic"]),
        "control_mean_gross_spread": control["mean_gross_top_minus_bottom"],
        "treatment_mean_gross_spread": treatment["mean_gross_top_minus_bottom"],
        "delta_mean_gross_spread": float(treatment["mean_gross_top_minus_bottom"])
        - float(control["mean_gross_top_minus_bottom"]),
        "control_mean_turnover": control["mean_one_way_turnover"],
        "treatment_mean_turnover": treatment["mean_one_way_turnover"],
        "delta_mean_turnover": float(treatment["mean_one_way_turnover"])
        - float(control["mean_one_way_turnover"]),
        "horizons": horizons,
    }


def issuer_seed29_gate(
    incumbent: CompletedRun,
    issuer: CompletedRun,
) -> dict[str, object]:
    if incumbent.seed != 29 or issuer.seed != 29:
        raise ValueError("Issuer first-stage gate requires the matched seed 29 pair")
    delta = _metric_delta(issuer.metrics, incumbent.metrics)
    horizon_values = list(delta["horizons"].values())
    positive_horizons = sum(float(row["delta_ic"]) > 0.0 for row in horizon_values)
    deteriorated_spreads = sum(
        float(row["delta_gross_spread"]) < 0.0 for row in horizon_values
    )
    passed = (
        float(delta["delta_primary_ic"]) > 0.0
        and positive_horizons >= 2
        and deteriorated_spreads <= 1
    )
    return {
        "stage": "seed29_screen",
        "passed": passed,
        "criteria": {
            "positive_paired_primary_ic_delta": float(delta["delta_primary_ic"]) > 0.0,
            "positive_horizon_ic_delta_count": positive_horizons,
            "minimum_positive_horizon_count": 2,
            "gross_spread_deterioration_count": deteriorated_spreads,
            "maximum_gross_spread_deterioration_count": 1,
        },
        "paired_metrics": delta,
        "transaction_cost_modeling": False,
    }


def issuer_three_seed_gate(
    incumbents: dict[int, CompletedRun],
    issuers: dict[int, CompletedRun],
) -> dict[str, object]:
    if (
        tuple(sorted(incumbents)) != ALLOWED_SEEDS
        or tuple(sorted(issuers)) != ALLOWED_SEEDS
    ):
        raise ValueError("Issuer confirmation requires matched seeds 11, 29, and 47")
    paired = {
        seed: _metric_delta(issuers[seed].metrics, incumbents[seed].metrics)
        for seed in ALLOWED_SEEDS
    }
    primary = np.asarray(
        [paired[seed]["delta_primary_ic"] for seed in ALLOWED_SEEDS],
        dtype=np.float64,
    )
    horizon_means = {
        f"{horizon}m": float(
            np.mean(
                [
                    paired[seed]["horizons"][f"{horizon}m"]["delta_ic"]
                    for seed in ALLOWED_SEEDS
                ]
            )
        )
        for horizon in HORIZONS
    }
    passed = (
        float(primary.mean()) > 0.0
        and int((primary > 0.0).sum()) >= 2
        and sum(value > 0.0 for value in horizon_means.values()) >= 2
    )
    return {
        "stage": "three_seed_confirmation",
        "passed": passed,
        "criteria": {
            "mean_paired_primary_effect": float(primary.mean()),
            "positive_paired_primary_seed_count": int((primary > 0.0).sum()),
            "minimum_positive_seed_count": 2,
            "mean_horizon_effects": horizon_means,
            "positive_mean_horizon_effect_count": sum(
                value > 0.0 for value in horizon_means.values()
            ),
            "minimum_positive_mean_horizon_count": 2,
        },
        "paired_by_seed": {str(seed): paired[seed] for seed in ALLOWED_SEEDS},
        "transaction_cost_modeling": False,
    }


def routing_summary(
    incumbents: dict[int, CompletedRun],
    runs: dict[tuple[str, str, int], CompletedRun],
) -> dict[str, object]:
    expected = {
        (slow, macro, seed)
        for slow in ROUTING_MODE_ORDER
        for macro in ROUTING_MODE_ORDER
        for seed in ROUTING_SEEDS
    }
    if set(runs) != expected:
        raise ValueError(
            "Routing summary requires the complete 4x4 matched-seed matrix"
        )
    control = {seed: runs[("late_only", "late_only", seed)] for seed in ROUTING_SEEDS}
    scaffold = {
        str(seed): _metric_delta(control[seed].metrics, incumbents[seed].metrics)
        for seed in ROUTING_SEEDS
    }
    arms = {}
    for slow in ROUTING_MODE_ORDER:
        for macro in ROUTING_MODE_ORDER:
            identity = f"slow={slow}|macro={macro}"
            arms[identity] = {
                "stage": routing_stage(slow, macro),
                "paired_vs_factorial_control": {
                    str(seed): _metric_delta(
                        runs[(slow, macro, seed)].metrics,
                        control[seed].metrics,
                    )
                    for seed in ROUTING_SEEDS
                },
            }

    effects: dict[str, list[dict[str, object]]] = {
        "slow_early_concat": [],
        "slow_film": [],
        "macro_temporal_early_concat": [],
        "macro_temporal_film": [],
    }
    for seed in ROUTING_SEEDS:
        for macro in ROUTING_MODE_ORDER:
            for film in (False, True):
                off = "film" if film else "late_only"
                on = "early_concat_film" if film else "early_concat"
                effects["slow_early_concat"].append(
                    _metric_delta(
                        runs[(on, macro, seed)].metrics,
                        runs[(off, macro, seed)].metrics,
                    )
                )
            for early in (False, True):
                off = "early_concat" if early else "late_only"
                on = "early_concat_film" if early else "film"
                effects["slow_film"].append(
                    _metric_delta(
                        runs[(on, macro, seed)].metrics,
                        runs[(off, macro, seed)].metrics,
                    )
                )
        for slow in ROUTING_MODE_ORDER:
            for film in (False, True):
                off = "film" if film else "late_only"
                on = "early_concat_film" if film else "early_concat"
                effects["macro_temporal_early_concat"].append(
                    _metric_delta(
                        runs[(slow, on, seed)].metrics,
                        runs[(slow, off, seed)].metrics,
                    )
                )
            for early in (False, True):
                off = "early_concat" if early else "late_only"
                on = "early_concat_film" if early else "film"
                effects["macro_temporal_film"].append(
                    _metric_delta(
                        runs[(slow, on, seed)].metrics,
                        runs[(slow, off, seed)].metrics,
                    )
                )

    main_effects = {}
    for name, comparisons in effects.items():
        main_effects[name] = {
            "matched_comparison_count": len(comparisons),
            "mean_delta_primary_ic": float(
                np.mean([row["delta_primary_ic"] for row in comparisons])
            ),
            "positive_primary_comparison_count": sum(
                float(row["delta_primary_ic"]) > 0.0 for row in comparisons
            ),
            "mean_delta_gross_spread": float(
                np.mean([row["delta_mean_gross_spread"] for row in comparisons])
            ),
            "mean_delta_turnover": float(
                np.mean([row["delta_mean_turnover"] for row in comparisons])
            ),
            "mean_horizon_ic_delta": {
                f"{horizon}m": float(
                    np.mean(
                        [
                            row["horizons"][f"{horizon}m"]["delta_ic"]
                            for row in comparisons
                        ]
                    )
                )
                for horizon in HORIZONS
            },
        }
    return {
        "matrix": "complete_4x4_factorial_across_seeds_11_29_47",
        "run_count": len(runs),
        "factorial_scaffold_vs_exact_legacy_incumbent": scaffold,
        "arms": arms,
        "independent_main_effects": main_effects,
        "winner_selected": False,
        "final_holdout_status": "sealed_not_accessed",
        "transaction_cost_modeling": False,
    }


def _configuration(
    commit: str,
    feature_store: Path,
    peer_primary_state: Path,
) -> dict[str, object]:
    legacy = architecture_for_model("tcn", _settings())
    factorial = architecture_for_model(
        "tcn", _settings("late_only", "late_only", "factorial_v1")
    )
    return {
        "orchestrator_git_commit_sha": commit,
        "feature_store": _feature_store_identity(feature_store),
        "feature_contract": FEATURE_CONTRACT_VERSION,
        "source_peer_primary_state": str(peer_primary_state),
        "source_peer_primary_state_sha256": _sha256(peer_primary_state),
        "context_ablation": resolve_context_ablation_for_store(
            feature_store, FROZEN_CONTEXT_ABLATION
        ).metadata(),
        "feature_ablation": resolve_feature_ablation_for_store(
            feature_store, FROZEN_FEATURE_ABLATION
        ).metadata(),
        "legacy_incumbent": {
            "tcn_settings": asdict(_settings()),
            "architecture": asdict(legacy),
            "peer_features": peer_feature_metadata("tcn", legacy, "selected"),
            "parameter_count": expected_trainable_parameter_count(
                "tcn", legacy, "selected"
            ),
        },
        "factorial_scaffold": {
            "schema": context_routing_metadata(factorial),
            "parameter_count": expected_trainable_parameter_count(
                "tcn", factorial, "selected"
            ),
            "state_dict_structure_equal_across_arms": True,
        },
        "seeds": list(ALLOWED_SEEDS),
        "issuer_job_count_maximum": 3,
        "routing_job_count": 48,
        "final_holdout_status": "sealed_not_accessed",
    }


def _new_job(base: dict[str, object]) -> dict[str, object]:
    return {
        **base,
        "status": "pending",
        "run_dir": None,
        "run_manifest_sha256": None,
        "output_sha256": None,
        "producing_git_commit_sha": None,
        "started_at_utc": None,
        "completed_at_utc": None,
        "primary_validation_ic": None,
        "recovery_count": 0,
        "last_recovery_at_utc": None,
        "skip_reason": None,
        "error": None,
    }


def _new_state(
    configuration: dict[str, object],
    incumbents: dict[int, tuple[CompletedRun, dict[str, object]]],
    audit_path: Path,
) -> dict[str, object]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "state_version": STATE_VERSION,
        "experiment_name": EXPERIMENT_NAME,
        "status": "running",
        "configuration": configuration,
        "created_at_utc": now,
        "completed_at_utc": None,
        "realized_distribution_audit": {
            "path": str(audit_path),
            "sha256": _sha256(audit_path),
            "audit_name": AUDIT_NAME,
            "audit_version": AUDIT_VERSION,
        },
        "incumbent_runs": {
            str(seed): {
                "run_dir": str(run.run_dir),
                "primary_validation_ic": run.primary_ic,
                "producing_git_commit_sha": run.producing_git_commit_sha,
                "output_sha256": run.output_sha256,
                "source_provenance": provenance,
            }
            for seed, (run, provenance) in incumbents.items()
        },
        "issuer_jobs": [_new_job(base) for base in issuer_jobs()],
        "routing_jobs": [_new_job(base) for base in routing_jobs()],
        "issuer_screen": None,
        "routing_summary": None,
        "summary_artifact": None,
    }


def _configurations_match(
    stored: dict[str, object], current: dict[str, object]
) -> bool:
    left = dict(stored)
    right = dict(current)
    left_feature = left.pop("feature_store", None)
    right_feature = right.pop("feature_store", None)
    left.pop("source_peer_primary_state", None)
    right.pop("source_peer_primary_state", None)
    return (
        left == right
        and isinstance(left_feature, dict)
        and isinstance(right_feature, dict)
        and _feature_identities_equivalent(left_feature, right_feature)
    )


def _validate_job_shape(
    recorded: list[object],
    expected: tuple[dict[str, object], ...],
    *,
    allow_skipped: bool,
) -> list[dict[str, object]]:
    if not isinstance(recorded, list) or len(recorded) != len(expected):
        raise ValueError("Stage-5 state has the wrong job count")
    result = []
    immutable = (
        "job_kind",
        "stage",
        "seed",
        "peer_features",
        "slow_routing",
        "macro_temporal_routing",
        "context_routing_experiment",
        "command",
        "serialized_job_specification",
        "job_specification_sha256",
    )
    statuses = {"pending", "running", "failed", "completed"}
    if allow_skipped:
        statuses.add("skipped")
    for job, base in zip(recorded, expected, strict=True):
        if not isinstance(job, dict):
            raise ValueError("Stage-5 job is malformed")
        recovery = job.get("recovery_count")
        if (
            job.get("status") not in statuses
            or any(job.get(field) != base[field] for field in immutable)
            or not isinstance(recovery, int)
            or isinstance(recovery, bool)
            or recovery < 0
        ):
            raise ValueError("Stage-5 job identity drifted")
        result.append(job)
    return result


def _load_state(
    state_path: Path,
    configuration: dict[str, object],
    incumbents: dict[int, tuple[CompletedRun, dict[str, object]]],
    audit_path: Path,
) -> dict[str, object]:
    if not state_path.exists():
        return _new_state(configuration, incumbents, audit_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    _reject_test_derived_metadata(state, "Stage-5 state")
    if (
        state.get("state_version") != STATE_VERSION
        or state.get("experiment_name") != EXPERIMENT_NAME
        or state.get("status") not in {"running", "completed"}
        or not isinstance(state.get("configuration"), dict)
        or not _configurations_match(state["configuration"], configuration)
    ):
        raise ValueError("Stage-5 state has an incompatible identity")
    _validate_job_shape(state.get("issuer_jobs"), issuer_jobs(), allow_skipped=True)
    _validate_job_shape(state.get("routing_jobs"), routing_jobs(), allow_skipped=False)
    audit = state.get("realized_distribution_audit")
    if (
        not isinstance(audit, dict)
        or audit.get("sha256") != _sha256(audit_path)
        or audit.get("audit_name") != AUDIT_NAME
        or audit.get("audit_version") != AUDIT_VERSION
    ):
        raise ValueError("Stage-5 audit provenance changed")
    for seed, (run, provenance) in incumbents.items():
        recorded = state.get("incumbent_runs", {}).get(str(seed))
        expected = {
            "run_dir": str(run.run_dir),
            "primary_validation_ic": run.primary_ic,
            "producing_git_commit_sha": run.producing_git_commit_sha,
            "output_sha256": run.output_sha256,
            "source_provenance": provenance,
        }
        if recorded != expected:
            raise ValueError(f"Stage-5 incumbent provenance changed for seed {seed}")
    return state


def _production_run_directories() -> set[Path]:
    if not RUN_OUTPUT_BASE.is_dir():
        return set()
    return {
        path.resolve()
        for path in RUN_OUTPUT_BASE.iterdir()
        if path.is_dir() and path.name != "_ops"
    }


def _candidate_runs(
    job: dict[str, object],
    feature_store: Path,
    commit: str,
) -> tuple[CompletedRun, ...]:
    candidates = set()
    if isinstance(job.get("run_dir"), str):
        recorded = Path(str(job["run_dir"]))
        if recorded.is_dir():
            candidates.add(recorded.resolve())
    for run_dir in _production_run_directories():
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        settings = manifest.get("tcn_settings")
        if (
            manifest.get("git_commit_sha") == commit
            and manifest.get("seed") == job["seed"]
            and _manifest_peer_mode(manifest) == job["peer_features"]
            and isinstance(settings, dict)
            and settings.get("slow_routing", "late_only") == job["slow_routing"]
            and settings.get("macro_temporal_routing", "late_only")
            == job["macro_temporal_routing"]
            and settings.get("context_routing_experiment", "legacy")
            == job["context_routing_experiment"]
        ):
            candidates.add(run_dir)
    completed = []
    for run_dir in sorted(candidates):
        try:
            completed.append(
                validate_completed_run(
                    run_dir,
                    feature_store,
                    seed=int(job["seed"]),
                    peer_features=str(job["peer_features"]),
                    slow_routing=str(job["slow_routing"]),
                    macro_temporal_routing=str(job["macro_temporal_routing"]),
                    context_routing_experiment=str(job["context_routing_experiment"]),
                    producing_commit=commit,
                )
            )
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
            continue
    return tuple(completed)


def _bind_completed_job(
    job: dict[str, object],
    run: CompletedRun,
    commit: str,
) -> None:
    job.update(
        {
            "status": "completed",
            "run_dir": str(run.run_dir),
            "run_manifest_sha256": run.output_sha256["run_manifest.json"],
            "output_sha256": run.output_sha256,
            "producing_git_commit_sha": commit,
            "completed_at_utc": job.get("completed_at_utc")
            or datetime.now(timezone.utc).isoformat(),
            "primary_validation_ic": run.primary_ic,
            "skip_reason": None,
            "error": None,
        }
    )


def _recover_job(
    job: dict[str, object],
    feature_store: Path,
    commit: str,
) -> CompletedRun | None:
    candidates = _candidate_runs(job, feature_store, commit)
    status = job["status"]
    if status == "completed":
        if len(candidates) != 1:
            raise ValueError("Completed Stage-5 job has missing or ambiguous artifacts")
        run = candidates[0]
        if (
            job.get("run_manifest_sha256") != run.output_sha256["run_manifest.json"]
            or job.get("output_sha256") != run.output_sha256
            or job.get("producing_git_commit_sha") != commit
            or not math.isclose(
                float(job["primary_validation_ic"]),
                run.primary_ic,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("Completed Stage-5 job state disagrees with artifacts")
        return run
    if status == "skipped":
        return None
    if status == "pending" and candidates:
        raise ValueError("Unbound completed run contaminates a pending Stage-5 job")
    if status == "failed" and candidates:
        raise ValueError("Failed Stage-5 job has an unbound completed artifact")
    if status == "running":
        if len(candidates) > 1:
            raise ValueError("Multiple completed runs match a recovering Stage-5 job")
        job["recovery_count"] = int(job["recovery_count"]) + 1
        job["last_recovery_at_utc"] = datetime.now(timezone.utc).isoformat()
        if candidates:
            _bind_completed_job(job, candidates[0], commit)
            return candidates[0]
    return None


def _launch_job(
    job: dict[str, object],
    state: dict[str, object],
    state_path: Path,
    feature_store: Path,
    commit: str,
) -> CompletedRun:
    recovered = _recover_job(job, feature_store, commit)
    if recovered is not None:
        _atomic_write_json(state_path, state)
        return recovered
    if owner := active_lock_owner(PRODUCTION_TRAINING_LOCK):
        raise RuntimeError(f"Another production training run is active: {owner}")
    before = _production_run_directories()
    job.update(
        {
            "status": "running",
            "run_dir": None,
            "run_manifest_sha256": None,
            "output_sha256": None,
            "producing_git_commit_sha": None,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_at_utc": None,
            "primary_validation_ic": None,
            "skip_reason": None,
            "error": None,
        }
    )
    _atomic_write_json(state_path, state)
    try:
        result = subprocess.run(job["command"], cwd=_RESEARCH, check=False)
    except OSError as error:
        job.update({"status": "failed", "error": str(error)})
        _atomic_write_json(state_path, state)
        raise RuntimeError("Could not start Stage-5 training") from error
    created = tuple(
        sorted(path for path in _production_run_directories() if path not in before)
    )
    if len(created) == 1:
        job["run_dir"] = str(created[0])
    if result.returncode != 0 or len(created) != 1:
        job.update(
            {
                "status": "failed",
                "error": (
                    f"training exited with code {result.returncode}; "
                    f"new run directory count={len(created)}"
                ),
            }
        )
        _atomic_write_json(state_path, state)
        raise RuntimeError(str(job["error"]))
    run = validate_completed_run(
        created[0],
        feature_store,
        seed=int(job["seed"]),
        peer_features=str(job["peer_features"]),
        slow_routing=str(job["slow_routing"]),
        macro_temporal_routing=str(job["macro_temporal_routing"]),
        context_routing_experiment=str(job["context_routing_experiment"]),
        producing_commit=commit,
    )
    _bind_completed_job(job, run, commit)
    _atomic_write_json(state_path, state)
    return run


def _ensure_audit(
    state_dir: Path,
    feature_store: Path,
    feature_identity: dict[str, object],
) -> Path:
    final_dir = state_dir / "realized_distribution_audit"
    final = final_dir / AUDIT_JSON
    if final.is_file():
        validate_realized_distribution_audit(final, feature_identity)
        return final
    if final_dir.exists():
        raise ValueError(
            "Incomplete realized-distribution audit directory requires operator inspection"
        )
    partial = state_dir / f"realized_distribution_audit.partial.{uuid4().hex}"
    audit_path = run_realized_distribution_audit(feature_store, partial)
    validate_realized_distribution_audit(audit_path, feature_identity)
    os.replace(partial, final_dir)
    return final


def _assert_invocation_identity(
    commit: str,
    feature_store: Path,
    configuration: dict[str, object],
    peer_primary_state: Path,
    audit_path: Path,
) -> None:
    current_commit, _ = _git_identity(require_clean=True)
    current_store = resolve_feature_store().resolve()
    current_identity = _feature_store_identity(current_store)
    if (
        current_commit != commit
        or not _feature_identities_equivalent(
            current_identity, configuration["feature_store"]
        )
        or not feature_store.samefile(current_store)
        or _sha256(peer_primary_state)
        != configuration["source_peer_primary_state_sha256"]
        or _sha256(audit_path)
        != json.loads(
            (audit_path.parent.parent / "state.json").read_text(encoding="utf-8")
        )["realized_distribution_audit"]["sha256"]
    ):
        raise RuntimeError(
            "Git, authoritative source, audit, or canonical feature store changed mid-run"
        )


def _summary_payload(
    state: dict[str, object],
    incumbents: dict[int, CompletedRun],
    issuer_runs: dict[int, CompletedRun],
    routing_runs: dict[tuple[str, str, int], CompletedRun],
) -> dict[str, object]:
    issuer = state["issuer_screen"]
    routing = routing_summary(incumbents, routing_runs)
    return {
        "experiment_name": EXPERIMENT_NAME,
        "state_version": STATE_VERSION,
        "status": "completed",
        "created_at_utc": state["created_at_utc"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": state["configuration"],
        "realized_distribution_audit": state["realized_distribution_audit"],
        "issuer_screen": issuer,
        "issuer_completed_seed_count": len(issuer_runs),
        "routing": routing,
        "final_holdout_status": "sealed_not_accessed",
        "winner_selected": False,
        "transaction_cost_modeling": False,
    }


def dry_run_payload(peer_primary_state: Path) -> dict[str, object]:
    commit, clean = _git_identity(require_clean=False)
    feature_store = resolve_feature_store().resolve()
    validate_feature_store(feature_store)
    configuration = _configuration(commit, feature_store, peer_primary_state)
    incumbents = _source_incumbents(peer_primary_state, feature_store)
    return {
        "experiment_name": EXPERIMENT_NAME,
        "dry_run": True,
        "worktree_clean": clean,
        "orchestrator_git_commit_sha": commit,
        "resolved_feature_store_path": str(feature_store),
        "source_peer_primary_state": str(peer_primary_state),
        "source_peer_primary_state_sha256": _sha256(peer_primary_state),
        "incumbent_runs": {
            str(seed): str(values[0].run_dir) for seed, values in incumbents.items()
        },
        "realized_distribution_audit_will_run_first": True,
        "issuer_seed29_job_count": 1,
        "issuer_confirmation_job_count_if_seed29_passes": 2,
        "routing_job_count": len(routing_jobs()),
        "routing_matrix": "complete_4x4_factorial_across_three_seeds",
        "final_holdout_status": "sealed_not_accessed",
        "configuration": configuration,
    }


def format_dry_run(payload: dict[str, object]) -> str:
    return "\n".join(
        (
            f"incumbent runs: {len(payload['incumbent_runs'])}",
            "realized-distribution audit runs first: yes",
            "issuer screen: seed 29, then seeds 11 and 47 only if it passes",
            f"routing training jobs: {payload['routing_job_count']}",
            "routing matrix: complete 4x4 factorial across seeds 11,29,47",
            "held-out test accessed: no",
        )
    )


def run_experiment(state_dir: Path, peer_primary_state: Path) -> Path:
    commit, _ = _git_identity(require_clean=True)
    feature_store = resolve_feature_store().resolve()
    validate_feature_store(feature_store)
    feature_identity = _feature_store_identity(feature_store)
    configuration = _configuration(commit, feature_store, peer_primary_state)
    state_dir = state_dir.resolve()
    if state_dir == feature_store or state_dir.is_relative_to(feature_store):
        raise ValueError(
            "Stage-5 state and audit artifacts must be outside the feature store"
        )
    incumbents_with_provenance = _source_incumbents(peer_primary_state, feature_store)
    incumbents = {
        seed: values[0] for seed, values in incumbents_with_provenance.items()
    }
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "state.json"
    with exclusive_process_lock(state_dir / "experiment.lock", EXPERIMENT_NAME):
        audit_path = _ensure_audit(state_dir, feature_store, feature_identity)
        if owner := active_lock_owner(PRODUCTION_TRAINING_LOCK):
            raise RuntimeError(f"Another production training run is active: {owner}")
        state = _load_state(
            state_path,
            configuration,
            incumbents_with_provenance,
            audit_path,
        )
        _atomic_write_json(state_path, state)

        issuer_runs: dict[int, CompletedRun] = {}
        seed29_job = state["issuer_jobs"][0]
        _assert_invocation_identity(
            commit, feature_store, configuration, peer_primary_state, audit_path
        )
        issuer_runs[29] = _launch_job(
            seed29_job, state, state_path, feature_store, commit
        )
        seed29_result = issuer_seed29_gate(incumbents[29], issuer_runs[29])
        state["issuer_screen"] = {"seed29": seed29_result, "three_seed": None}
        _atomic_write_json(state_path, state)

        if seed29_result["passed"]:
            for job in state["issuer_jobs"][1:]:
                _assert_invocation_identity(
                    commit,
                    feature_store,
                    configuration,
                    peer_primary_state,
                    audit_path,
                )
                run = _launch_job(job, state, state_path, feature_store, commit)
                issuer_runs[int(job["seed"])] = run
            state["issuer_screen"]["three_seed"] = issuer_three_seed_gate(
                incumbents, issuer_runs
            )
        else:
            for job in state["issuer_jobs"][1:]:
                if job["status"] == "completed":
                    raise ValueError(
                        "Issuer confirmation run exists although the seed-29 gate failed"
                    )
                job.update(
                    {
                        "status": "skipped",
                        "skip_reason": "seed29_issuer_gate_failed",
                        "error": None,
                    }
                )
        _atomic_write_json(state_path, state)

        routing_runs: dict[tuple[str, str, int], CompletedRun] = {}
        total = len(state["routing_jobs"])
        for position, job in enumerate(state["routing_jobs"], start=1):
            _assert_invocation_identity(
                commit,
                feature_store,
                configuration,
                peer_primary_state,
                audit_path,
            )
            run = _launch_job(job, state, state_path, feature_store, commit)
            identity = (
                str(job["slow_routing"]),
                str(job["macro_temporal_routing"]),
                int(job["seed"]),
            )
            routing_runs[identity] = run
            print(
                f"[routing {position}/{total}] {identity[0]} / {identity[1]} "
                f"seed={identity[2]} IC={run.primary_ic:.8f}",
                flush=True,
            )

        summary = _summary_payload(state, incumbents, issuer_runs, routing_runs)
        summary_path = state_dir / SUMMARY_JSON
        _atomic_write_json(summary_path, summary)
        state["routing_summary"] = summary["routing"]
        state["summary_artifact"] = {
            "path": str(summary_path),
            "sha256": _sha256(summary_path),
        }
        state["status"] = "completed"
        state["completed_at_utc"] = summary["completed_at_utc"]
        _atomic_write_json(state_path, state)
    return state_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--peer-primary-state", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    peer_primary_state = resolve_peer_primary_state(args.peer_primary_state)
    if args.dry_run:
        print(format_dry_run(dry_run_payload(peer_primary_state)), flush=True)
        return
    state_path = run_experiment(args.state_dir.resolve(), peer_primary_state)
    print(f"Stage-5 context-routing experiment completed: {state_path}", flush=True)


if __name__ == "__main__":
    main()
