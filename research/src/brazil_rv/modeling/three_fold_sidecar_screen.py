from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from .analyze import compare_observation_ensembles, load_run_observations
from .contract import ALLOWED_SEEDS, MAX_EPOCHS
from .data import feature_store_identity, load_external_sidecar
from .designated_challenger import (
    DESIGNATED_CHALLENGER_NAME,
    compare_discovery_screen,
    load_designated_challenger_members,
)
from .engine import EvaluationObservations, objective_metadata
from .metrics import primary_validation_score
from .metrics import moving_block_bootstrap
from .provenance import repository_commit
from .train import run_training
from .trajectory import predictions_for_rule, simulate_patience3

FOLDS = ("fold_c", "fold_a", "fold_b")
PRIMARY_RULE = "bidirectional_odd_even_crossfit_patience3_raw"
SECONDARY_RULE = "final_ema_0995"
GATE_MEAN = 0.001
MAXIMUM_DIVERSITY_MEMBER_FOLD_LOSS = 0.001
MAX_PARALLEL = 2


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def crossfit_patience_observations(
    run_dir: Path,
    frozen_replays: list[dict[str, object]] | None = None,
) -> tuple[EvaluationObservations, list[dict[str, object]]]:
    reference = load_run_observations(run_dir, "final_raw")
    dates = np.unique(reference.date_idx)
    if dates.size < 6:
        raise ValueError("Cross-fit Patience requires at least six selection dates")
    parities = {"odd": dates[0::2], "even": dates[1::2]}
    predictions = np.empty_like(reference.predictions)
    replay_rows = []
    if frozen_replays is not None:
        if len(frozen_replays) != 2:
            raise ValueError("Frozen cross-fit replay must contain both directions")
        seen_evaluation_parities = set()
        for replay in frozen_replays:
            selection_parity = str(replay["selection_parity"])
            evaluation_parity = str(replay["evaluation_parity"])
            if {selection_parity, evaluation_parity} != {"odd", "even"}:
                raise ValueError("Frozen cross-fit replay has invalid parities")
            if evaluation_parity in seen_evaluation_parities:
                raise ValueError("Frozen cross-fit replay repeats an evaluation parity")
            seen_evaluation_parities.add(evaluation_parity)
            selected_epoch = int(replay["selected_epoch"])
            evaluation = np.isin(reference.date_idx, parities[evaluation_parity])
            with np.load(
                run_dir / "validation_predictions" / f"epoch_{selected_epoch:02d}.npz",
                allow_pickle=False,
            ) as values:
                predictions[evaluation] = values["raw"][evaluation]
            replay_rows.append(dict(replay))
        return replace(reference, predictions=predictions), replay_rows

    epoch_predictions = []
    for epoch in range(1, MAX_EPOCHS + 1):
        with np.load(
            run_dir / "validation_predictions" / f"epoch_{epoch:02d}.npz",
            allow_pickle=False,
        ) as values:
            epoch_predictions.append(values["raw"].copy())
    for selection_parity, evaluation_parity in (("odd", "even"), ("even", "odd")):
        selection = np.isin(reference.date_idx, parities[selection_parity])
        evaluation = np.isin(reference.date_idx, parities[evaluation_parity])
        scores = [
            primary_validation_score(
                values[selection],
                reference.targets[selection],
                reference.label_mask[selection],
                reference.date_idx[selection],
            )
            for values in epoch_predictions
        ]
        replay = simulate_patience3(scores)
        selected_epoch = int(replay["selected_epoch"])
        predictions[evaluation] = epoch_predictions[selected_epoch - 1][evaluation]
        replay_rows.append(
            {
                "selection_parity": selection_parity,
                "evaluation_parity": evaluation_parity,
                "selected_epoch": selected_epoch,
                "stopped_epoch": int(replay["stopped_epoch"]),
                "selection_half_ic": float(replay["selected_score"]),
            }
        )
    return replace(reference, predictions=predictions), replay_rows


def _run_job(
    store: Path,
    run_dir: Path,
    seed: int,
    fold: str,
    sidecar_dir: Path,
    zero_dynamic_channels: tuple[int, ...],
    zero_slow_fields: tuple[int, ...],
) -> str:
    run_training(
        store=store,
        seed=seed,
        selection_window=fold,
        run_dir=run_dir,
        sidecar_dir=sidecar_dir,
        zero_dynamic_channels=zero_dynamic_channels,
        zero_slow_fields=zero_slow_fields,
    )
    return str(run_dir)


def _candidate_run(output_dir: Path, fold: str, seed: int) -> Path:
    return output_dir / "runs" / fold / f"seed_{seed}"


def _parent_run(
    parent_campaign: Path, fold_c_parent: Path, fold: str, seed: int
) -> Path:
    if fold == "fold_c":
        return fold_c_parent / "fold_c" / f"seed_{seed}"
    return parent_campaign / fold / f"seed_{seed}"


def _members(
    paths: Mapping[str, Path],
    readout: str,
    frozen_replays: Mapping[str, list[dict[str, object]]] | None = None,
) -> tuple[dict[str, EvaluationObservations], dict[str, object]]:
    members = {}
    replays = {}
    for name, path in paths.items():
        if readout == PRIMARY_RULE:
            observations, replay = crossfit_patience_observations(
                path,
                None if frozen_replays is None else frozen_replays[name],
            )
        elif readout == SECONDARY_RULE:
            reference = load_run_observations(path, "final_raw")
            observations = replace(
                reference, predictions=predictions_for_rule(path, readout)
            )
            replay = []
        else:
            raise ValueError(f"Unknown readout: {readout}")
        members[name] = observations
        replays[name] = replay
    return members, replays


def _extract_analysis(path: Path) -> dict[str, object]:
    value = _read_json(path / "analysis.json")
    return {
        "candidate_ensemble_ic": value["candidate"]["ensemble_ic"],
        "parent_ensemble_ic": value["parent"]["ensemble_ic"],
        "candidate_minus_parent_ic": value["candidate_minus_parent_primary_ic"],
        "candidate": value["candidate"],
        "parent": value["parent"],
        "per_date_delta_bootstrap": value["per_date_delta_bootstrap"],
        "horizon_guardrails": value["horizon_guardrails"],
        "time_of_day_guardrails": value["time_of_day_guardrails"],
        "analysis": str((path / "analysis.json").resolve()),
    }


def _compare_primary(
    *,
    fold: str,
    candidate: Mapping[str, EvaluationObservations],
    parent: Mapping[str, EvaluationObservations],
    output_dir: Path,
    run_root: Path,
) -> dict[str, object]:
    if fold in ("fold_a", "fold_b"):
        report = compare_discovery_screen(
            candidate,
            parent,
            fold=fold,
            candidate_rule=PRIMARY_RULE,
            parent_rule=PRIMARY_RULE,
            output_dir=output_dir,
            run_root=run_root,
            require_designated_parent_match=False,
        )
        summary = _read_json(report / "screen_summary.json")
        return {
            "candidate_minus_canonical_ic": summary["candidate_minus_canonical_ic"],
            "candidate_minus_challenger_ic": summary["candidate_minus_challenger_ic"],
            "vs_canonical": _extract_analysis(report / "vs_canonical"),
            "vs_challenger": _extract_analysis(report / "vs_designated_challenger"),
        }
    report = compare_observation_ensembles(
        candidate,
        parent,
        candidate_rule=PRIMARY_RULE,
        parent_rule=PRIMARY_RULE,
        output_dir=output_dir / "vs_canonical",
        comparison_metadata={
            "fold": fold,
            "retention_comparator": True,
            "designated_challenger_available": False,
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    canonical = _extract_analysis(report)
    return {
        "candidate_minus_canonical_ic": canonical["candidate_minus_parent_ic"],
        "candidate_minus_challenger_ic": None,
        "vs_canonical": canonical,
        "vs_challenger": None,
    }


def _gate(rows: Mapping[str, Mapping[str, object]], key: str) -> bool:
    deltas = [float(rows[fold][key]) for fold in FOLDS]
    return float(np.mean(deltas)) >= GATE_MEAN and all(value >= 0 for value in deltas)


def run_three_fold_sidecar_screen(
    *,
    store: Path,
    sidecar_dir: Path,
    candidate_name: str,
    parent_campaign: Path,
    fold_c_parent: Path,
    fold_c_parent_replay_report: Path,
    run_root: Path,
    output_dir: Path,
    experiment_role: str,
    parallel_processes: int = 2,
    zero_dynamic_channels: tuple[int, ...] = (),
    zero_slow_fields: tuple[int, ...] = (),
    analysis_only: bool = False,
) -> Path:
    if output_dir.exists() and not analysis_only:
        raise FileExistsError(output_dir)
    if not output_dir.is_dir() and analysis_only:
        raise FileNotFoundError(output_dir)
    if not candidate_name or not experiment_role:
        raise ValueError("Candidate name and experiment role must be nonempty")
    if not 1 <= parallel_processes <= MAX_PARALLEL:
        raise ValueError("A screen allows one or two isolated training processes")
    if len(set(zero_dynamic_channels)) != len(zero_dynamic_channels) or any(
        not 0 <= value < 26 for value in zero_dynamic_channels
    ):
        raise ValueError("Dynamic zeroing indices are invalid")
    if len(set(zero_slow_fields)) != len(zero_slow_fields) or any(
        not 0 <= value < 32 for value in zero_slow_fields
    ):
        raise ValueError("Slow-field zeroing indices are invalid")
    sidecar = load_external_sidecar(sidecar_dir, store)
    fold_c_report = _read_json(fold_c_parent_replay_report)
    replay_metadata = fold_c_report.get("comparison_metadata", {})
    parent_replays_by_fold = replay_metadata.get("parent_patience_replays_by_fold")
    if parent_replays_by_fold is None:
        parent_replays_by_fold = {
            "fold_c": replay_metadata.get("parent_patience_replays")
        }
    expected_seeds = {f"seed_{seed}" for seed in ALLOWED_SEEDS}
    if set(parent_replays_by_fold) not in ({"fold_c"}, set(FOLDS)) or any(
        not isinstance(replays, dict) or set(replays) != expected_seeds
        for replays in parent_replays_by_fold.values()
    ):
        raise ValueError("Parent replay report has malformed frozen seed replays")
    commit = repository_commit()
    output_dir.mkdir(parents=True, exist_ok=analysis_only)
    manifest_path = output_dir / "campaign_manifest.json"
    if analysis_only:
        manifest = _read_json(manifest_path)
        if (
            manifest.get("status") != "running"
            or manifest.get("candidate_name") != candidate_name
            or manifest.get("experiment_role") != experiment_role
            or manifest.get("feature_store") != str(store.resolve())
            or manifest.get("official_validation_accessed") is not False
            or manifest.get("test_accessed") is not False
        ):
            raise ValueError("Existing campaign does not match the analysis-only request")
    else:
        manifest = {
            "schema": "THREE_FOLD_EXTERNAL_SIDECAR_SCREEN_V1",
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "candidate_name": candidate_name,
            "experiment_role": experiment_role,
            "repository_commit": commit,
            "feature_store": str(store.resolve()),
            "feature_store_identity": feature_store_identity(store),
            "external_sidecar": sidecar.identity,
            "folds": list(FOLDS),
            "fold_c": {
                "fit": "2021-08-16..2023-03-31 (407 dates)",
                "selection": "2023-04-03..2023-08-31 (105 dates)",
                "date_sampling": "512 draws with replacement per effective batch",
                "parent_replay_report": str(fold_c_parent_replay_report.resolve()),
            },
            "seeds": list(ALLOWED_SEEDS),
            "objective": objective_metadata(),
            "base_feature_pruning": {
                "dynamic_channels": list(zero_dynamic_channels),
                "slow_fields": list(zero_slow_fields),
                "applied_from_epoch_zero": True,
            },
            "primary_gate": "mean delta >= +0.001 and every fold delta >= 0",
            "diversity_guardrail": "standalone delta >= -0.001 on every fold",
            "retention_comparator": "canonical_parent_only",
            "designated_challenger_role": "informational_only_on_folds_a_b",
            "secondary_readout": SECONDARY_RULE,
            "official_validation_accessed": False,
            "test_accessed": False,
        }
        _atomic_json(manifest_path, manifest)

    jobs = [
        (
            store,
            _candidate_run(output_dir, fold, seed),
            seed,
            fold,
            sidecar_dir,
            zero_dynamic_channels,
            zero_slow_fields,
        )
        for fold in FOLDS
        for seed in ALLOWED_SEEDS
    ]
    if analysis_only:
        for fold in FOLDS:
            for seed in ALLOWED_SEEDS:
                run_manifest = _read_json(
                    _candidate_run(output_dir, fold, seed) / "run_manifest.json"
                )
                if (
                    run_manifest.get("status") != "completed"
                    or run_manifest.get("seed") != seed
                    or run_manifest.get("split", {}).get("training") != fold
                    or run_manifest.get("split", {}).get("test_accessed") is not False
                ):
                    raise ValueError(f"Candidate run is not complete: {fold}/seed_{seed}")
    elif parallel_processes == 1:
        for job in jobs:
            print(_run_job(*job), flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=parallel_processes, mp_context=mp.get_context("spawn")
        ) as executor:
            futures = [executor.submit(_run_job, *job) for job in jobs]
            for future in as_completed(futures):
                print(future.result(), flush=True)

    analyses: dict[str, object] = {}
    for fold in FOLDS:
        candidate_paths = {
            f"seed_{seed}": _candidate_run(output_dir, fold, seed)
            for seed in ALLOWED_SEEDS
        }
        parent_paths = {
            f"seed_{seed}": _parent_run(parent_campaign, fold_c_parent, fold, seed)
            for seed in ALLOWED_SEEDS
        }
        candidate_primary, candidate_replays = _members(candidate_paths, PRIMARY_RULE)
        parent_primary, parent_replays = _members(
            parent_paths,
            PRIMARY_RULE,
            parent_replays_by_fold.get(fold),
        )
        standalone = _compare_primary(
            fold=fold,
            candidate=candidate_primary,
            parent=parent_primary,
            output_dir=output_dir / "analysis" / fold / "primary_standalone",
            run_root=run_root,
        )
        stack_members = {
            **{f"parent_{key}": value for key, value in parent_primary.items()},
            **{f"candidate_{key}": value for key, value in candidate_primary.items()},
        }
        stack = _compare_primary(
            fold=fold,
            candidate=stack_members,
            parent=parent_primary,
            output_dir=output_dir / "analysis" / fold / "primary_parent_plus_candidate",
            run_root=run_root,
        )

        candidate_ema, _ = _members(candidate_paths, SECONDARY_RULE)
        parent_ema, _ = _members(parent_paths, SECONDARY_RULE)
        ema = compare_observation_ensembles(
            candidate_ema,
            parent_ema,
            candidate_rule=SECONDARY_RULE,
            parent_rule=SECONDARY_RULE,
            output_dir=output_dir / "analysis" / fold / "secondary_ema",
            comparison_metadata={
                "fold": fold,
                "retention_eligible": False,
                "official_validation_accessed": False,
                "test_accessed": False,
            },
        )
        ema_stack = compare_observation_ensembles(
            {
                **{f"parent_{key}": value for key, value in parent_ema.items()},
                **{f"candidate_{key}": value for key, value in candidate_ema.items()},
            },
            parent_ema,
            candidate_rule="parent_plus_candidate_final_ema_0995_uniform_6",
            parent_rule=SECONDARY_RULE,
            output_dir=output_dir / "analysis" / fold / "secondary_ema_stack",
            comparison_metadata={
                "fold": fold,
                "retention_eligible": False,
                "official_validation_accessed": False,
                "test_accessed": False,
            },
        )
        mixed = None
        if fold in ("fold_a", "fold_b"):
            challenger = load_designated_challenger_members(fold, run_root=run_root)
            mixed_dir = compare_observation_ensembles(
                {**challenger, **candidate_ema},
                challenger,
                candidate_rule=(
                    f"{DESIGNATED_CHALLENGER_NAME}_plus_candidate_final_ema_0995"
                ),
                parent_rule=DESIGNATED_CHALLENGER_NAME,
                output_dir=output_dir
                / "analysis"
                / fold
                / "secondary_challenger_plus_candidate_ema",
                comparison_metadata={
                    "fold": fold,
                    "informational_only": True,
                    "retention_comparator": False,
                    "beats_either_allowed": False,
                    "official_validation_accessed": False,
                    "test_accessed": False,
                },
            )
            mixed = _extract_analysis(mixed_dir)
        analyses[fold] = {
            "primary_standalone": standalone,
            "primary_parent_plus_candidate": stack,
            "secondary_ema": _extract_analysis(ema),
            "secondary_ema_parent_plus_candidate": _extract_analysis(ema_stack),
            "secondary_challenger_plus_candidate_ema": mixed,
            "candidate_patience_replays": candidate_replays,
            "parent_patience_replays": parent_replays,
        }

    standalone_rows = {
        fold: {
            "delta": analyses[fold]["primary_standalone"][
                "candidate_minus_canonical_ic"
            ]
        }
        for fold in FOLDS
    }
    stack_rows = {
        fold: {
            "delta": analyses[fold]["primary_parent_plus_candidate"][
                "candidate_minus_canonical_ic"
            ]
        }
        for fold in FOLDS
    }
    standalone_pass = _gate(standalone_rows, "delta")
    diversity_pass = _gate(stack_rows, "delta") and all(
        float(standalone_rows[fold]["delta"]) >= -MAXIMUM_DIVERSITY_MEMBER_FOLD_LOSS
        for fold in FOLDS
    )
    pooled_daily = np.concatenate(
        [
            np.asarray(
                pl.read_parquet(
                    output_dir
                    / "analysis"
                    / fold
                    / "primary_standalone"
                    / "vs_canonical"
                    / "daily_delta.parquet"
                )["candidate_minus_parent_ic"]
            )
            for fold in FOLDS
        ]
    )
    pooled_intervals = {
        str(block): {
            key: np.asarray(value).tolist()
            for key, value in moving_block_bootstrap(
                pooled_daily,
                replications=10_000,
                block_length=block,
                seed=46 + block,
            ).items()
        }
        for block in (5, 10)
    }
    intervals_support = all(
        float(pooled_intervals[str(block)]["lower_95"][0]) > 0.0 for block in (5, 10)
    )
    standalone_pass = standalone_pass and intervals_support
    summary = {
        "schema": "THREE_FOLD_EXTERNAL_SIDECAR_SCREEN_SUMMARY_V1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_name": candidate_name,
        "experiment_role": experiment_role,
        "folds": analyses,
        "primary_paths": {
            "standalone": {
                "fold_deltas": {fold: standalone_rows[fold]["delta"] for fold in FOLDS},
                "mean_delta": float(
                    np.mean([standalone_rows[fold]["delta"] for fold in FOLDS])
                ),
                "gate_passed": standalone_pass,
                "pooled_daily_delta_bootstrap": pooled_intervals,
                "intervals_support_superiority": intervals_support,
            },
            "parent_plus_candidate": {
                "fold_deltas": {fold: stack_rows[fold]["delta"] for fold in FOLDS},
                "mean_delta": float(
                    np.mean([stack_rows[fold]["delta"] for fold in FOLDS])
                ),
                "gate_passed": diversity_pass,
                "extra_guardrail": "standalone delta >= -0.001 on every fold",
            },
        },
        "candidate_retained": standalone_pass or diversity_pass,
        "selection_contract": {
            "primary_readout": PRIMARY_RULE,
            "secondary_readout": SECONDARY_RULE,
            "retention_comparator": "canonical_parent_only",
            "designated_challenger_role": "informational_only",
            "beats_either_allowed": False,
            "base_feature_pruning": {
                "dynamic_channels": list(zero_dynamic_channels),
                "slow_fields": list(zero_slow_fields),
                "applied_from_epoch_zero": True,
            },
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    }
    summary_path = output_dir / "screen_summary.json"
    _atomic_json(summary_path, summary)
    _atomic_json(
        manifest_path,
        {
            **manifest,
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "analysis_repository_commit": commit,
            "analysis_only_resume": analysis_only,
            "screen_summary": str(summary_path.resolve()),
            "candidate_retained": summary["candidate_retained"],
        },
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one bias-free sidecar candidate on all three discovery folds"
    )
    for name in (
        "store",
        "sidecar_dir",
        "parent_campaign",
        "fold_c_parent",
        "fold_c_parent_replay_report",
        "run_root",
        "output_dir",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--experiment-role", required=True)
    parser.add_argument("--parallel-processes", type=int, default=2)
    parser.add_argument("--zero-dynamic-channels", type=int, nargs="*", default=[])
    parser.add_argument("--zero-slow-fields", type=int, nargs="*", default=[])
    parser.add_argument("--analysis-only", action="store_true")
    values = vars(parser.parse_args())
    values["zero_dynamic_channels"] = tuple(values["zero_dynamic_channels"])
    values["zero_slow_fields"] = tuple(values["zero_slow_fields"])
    print(run_three_fold_sidecar_screen(**values))


if __name__ == "__main__":
    main()
