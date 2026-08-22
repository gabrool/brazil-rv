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

from .analyze import compare_observation_ensembles, load_run_observations
from .contract import ALLOWED_SEEDS, MAX_EPOCHS
from .designated_challenger import load_designated_challenger_members
from .engine import EvaluationObservations
from .metrics import primary_validation_score
from .mixed_state_stack import load_mixed_state_variant_members
from .train import run_training
from .trajectory import predictions_for_rule, simulate_patience3

FOLDS = ("fold_c", "fold_a", "fold_b")
GATE_MEAN = 0.001
MAX_PARALLEL = 2


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def crossfit_patience_observations(
    run_dir: Path,
) -> tuple[EvaluationObservations, list[dict[str, object]]]:
    reference = load_run_observations(run_dir, "final_raw")
    epoch_predictions = []
    for epoch in range(1, MAX_EPOCHS + 1):
        with np.load(
            run_dir / "validation_predictions" / f"epoch_{epoch:02d}.npz",
            allow_pickle=False,
        ) as values:
            epoch_predictions.append(values["raw"].copy())
    dates = np.unique(reference.date_idx)
    if dates.size < 6:
        raise ValueError("Cross-fit Patience requires at least six selection dates")
    parities = {"odd": dates[0::2], "even": dates[1::2]}
    predictions = np.empty_like(reference.predictions)
    replay_rows = []
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
    sidecar: Path | None,
    dynamic: tuple[int, ...],
    slow: tuple[int, ...],
) -> str:
    run_training(
        store=store,
        seed=seed,
        selection_window=fold,
        run_dir=run_dir,
        sidecar_dir=sidecar,
        zero_dynamic_channels=dynamic,
        zero_slow_fields=slow,
    )
    return str(run_dir)


def _run_path(output_dir: Path, candidate: bool, fold: str, seed: int) -> Path:
    root = output_dir / ("f3_candidate" if candidate else "fold_c_parent")
    return root / fold / f"seed_{seed}"


def _parent_path(output_dir: Path, parent_campaign: Path, fold: str, seed: int) -> Path:
    return (
        _run_path(output_dir, False, fold, seed)
        if fold == "fold_c"
        else parent_campaign / fold / f"seed_{seed}"
    )


def _members(
    paths: Mapping[str, Path], readout: str
) -> tuple[dict[str, EvaluationObservations], dict[str, object]]:
    members = {}
    replays = {}
    for name, path in paths.items():
        if readout == "patience3_raw":
            observations, replay = crossfit_patience_observations(path)
        elif readout == "final_ema_0995":
            reference = load_run_observations(path, "final_raw")
            observations = replace(
                reference, predictions=predictions_for_rule(path, readout)
            )
            replay = []
        else:
            raise ValueError(f"Unknown P1 readout: {readout}")
        members[name] = observations
        replays[name] = replay
    return members, replays


def _extract(report: Path) -> dict[str, object]:
    value = json.loads((report / "analysis.json").read_text(encoding="utf-8"))
    return {
        "candidate_ensemble_ic": value["candidate"]["ensemble_ic"],
        "parent_ensemble_ic": value["parent"]["ensemble_ic"],
        "candidate_minus_parent_ic": value["candidate_minus_parent_primary_ic"],
        "per_date_delta_bootstrap": value["per_date_delta_bootstrap"],
        "horizon_guardrails": value["horizon_guardrails"],
        "time_of_day_guardrails": value["time_of_day_guardrails"],
        "analysis": str(report / "analysis.json"),
    }


def _gate(fold_rows: Mapping[str, Mapping[str, object]]) -> bool:
    deltas = [float(fold_rows[fold]["candidate_minus_parent_ic"]) for fold in FOLDS]
    return float(np.mean(deltas)) >= GATE_MEAN and all(value >= 0 for value in deltas)


def run_p1_campaign(
    *,
    store: Path,
    parent_campaign: Path,
    selected_sidecar: Path,
    f2_selection: Path,
    attribution_report: Path,
    p0_summary: Path,
    run_root: Path,
    phase_b_campaign: Path,
    phase_c_campaign: Path,
    external_program: Path,
    output_dir: Path,
    parallel_processes: int = 2,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not 1 <= parallel_processes <= MAX_PARALLEL:
        raise ValueError("P1 allows one or two isolated training processes")
    f2 = json.loads(f2_selection.read_text(encoding="utf-8"))
    attribution = json.loads(attribution_report.read_text(encoding="utf-8"))
    p0 = json.loads(p0_summary.read_text(encoding="utf-8"))
    if not f2.get("f3_allowed"):
        raise ValueError("F2 did not select the preregistered minimum six features")
    dynamic = tuple(int(value) for value in attribution["dead_dynamic_indices"])
    slow = tuple(int(value) for value in attribution["dead_slow_indices"])
    output_dir.mkdir(parents=True)
    manifest_path = output_dir / "campaign_manifest.json"
    manifest = {
        "schema": "P1_FEATURE_PROGRAM_V1",
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "folds": list(FOLDS),
        "fold_c": {
            "fit": "2021-08-16..2023-03-31 (407 dates)",
            "selection": "2023-04-03..2023-08-31 (105 dates)",
            "date_sampling": "512 draws with replacement per effective batch",
        },
        "seeds": list(ALLOWED_SEEDS),
        "selected_sidecar": str(selected_sidecar.resolve()),
        "selected_features": f2["selected_features"],
        "dead_dynamic_indices": list(dynamic),
        "dead_slow_indices": list(slow),
        "primary_gate": "mean delta >= +0.001 and every fold delta >= 0",
        "retention_comparator": "canonical parent only",
        "mixed_state_role": "secondary_informational_only",
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(manifest_path, manifest)

    jobs = []
    for seed in ALLOWED_SEEDS:
        jobs.append(
            (
                store,
                _run_path(output_dir, False, "fold_c", seed),
                seed,
                "fold_c",
                None,
                (),
                (),
            )
        )
    for fold in FOLDS:
        for seed in ALLOWED_SEEDS:
            jobs.append(
                (
                    store,
                    _run_path(output_dir, True, fold, seed),
                    seed,
                    fold,
                    selected_sidecar,
                    dynamic,
                    slow,
                )
            )
    if parallel_processes == 1:
        for job in jobs:
            print(_run_job(*job), flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=parallel_processes, mp_context=mp.get_context("spawn")
        ) as executor:
            futures = [executor.submit(_run_job, *job) for job in jobs]
            for future in as_completed(futures):
                print(future.result(), flush=True)

    analyses: dict[str, object] = {"patience3_raw": {}, "final_ema_0995": {}}
    for readout in analyses:
        for fold in FOLDS:
            candidate, candidate_replays = _members(
                {
                    f"seed_{seed}": _run_path(output_dir, True, fold, seed)
                    for seed in ALLOWED_SEEDS
                },
                readout,
            )
            parent, parent_replays = _members(
                {
                    f"seed_{seed}": _parent_path(
                        output_dir, parent_campaign, fold, seed
                    )
                    for seed in ALLOWED_SEEDS
                },
                readout,
            )
            metadata = {
                "fold": fold,
                "readout": readout,
                "candidate_patience_replays": candidate_replays,
                "parent_patience_replays": parent_replays,
                "retention_comparator": readout == "patience3_raw",
                "official_validation_accessed": False,
                "test_accessed": False,
            }
            standalone = compare_observation_ensembles(
                candidate,
                parent,
                candidate_rule=readout,
                parent_rule=readout,
                output_dir=output_dir / "analysis" / readout / fold / "standalone",
                comparison_metadata={**metadata, "composition": "candidate_three"},
            )
            diversity = compare_observation_ensembles(
                {
                    **{f"parent_{key}": value for key, value in parent.items()},
                    **{f"candidate_{key}": value for key, value in candidate.items()},
                },
                parent,
                candidate_rule=readout,
                parent_rule=readout,
                output_dir=output_dir
                / "analysis"
                / readout
                / fold
                / "parent_plus_candidate",
                comparison_metadata={
                    **metadata,
                    "composition": "parent_three_plus_candidate_three",
                },
            )
            analyses[readout][fold] = {
                "standalone": _extract(standalone),
                "parent_plus_candidate": _extract(diversity),
            }

    primary = analyses["patience3_raw"]
    standalone_folds = {fold: primary[fold]["standalone"] for fold in FOLDS}
    diversity_folds = {fold: primary[fold]["parent_plus_candidate"] for fold in FOLDS}
    standalone_pass = _gate(standalone_folds)
    diversity_pass = _gate(diversity_folds) and all(
        float(standalone_folds[fold]["candidate_minus_parent_ic"]) >= -0.001
        for fold in FOLDS
    )

    mixed_variant = p0.get("selected_discovery_variant")
    mixed_secondary: dict[str, object] = {}
    for fold in ("fold_a", "fold_b"):
        candidate, _ = _members(
            {
                f"f3_seed_{seed}": _run_path(output_dir, True, fold, seed)
                for seed in ALLOWED_SEEDS
            },
            "final_ema_0995",
        )
        if mixed_variant is None:
            base = load_designated_challenger_members(fold, run_root=run_root)
            base_name = "designated_challenger"
        else:
            base = load_mixed_state_variant_members(
                str(mixed_variant),
                fold,
                run_root=run_root,
                phase_b_campaign=phase_b_campaign,
                phase_c_campaign=phase_c_campaign,
                external_program=external_program,
            )
            base_name = str(mixed_variant)
        comparison = compare_observation_ensembles(
            {**base, **candidate},
            base,
            candidate_rule="mixed_state_plus_f3_final_ema_0995",
            parent_rule=base_name,
            output_dir=output_dir / "analysis" / "mixed_state_secondary" / fold,
            comparison_metadata={
                "informational_only": True,
                "retention_comparator": False,
                "official_validation_accessed": False,
                "test_accessed": False,
            },
        )
        mixed_secondary[fold] = _extract(comparison)

    summary = {
        "schema": "P1_F3_SUMMARY_V1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "readouts": analyses,
        "primary_paths": {
            "standalone": {
                "folds": standalone_folds,
                "gate_passed": standalone_pass,
            },
            "parent_plus_candidate": {
                "folds": diversity_folds,
                "gate_passed": diversity_pass,
                "extra_guardrail": "standalone delta >= -0.001 on every fold",
            },
        },
        "f3_passed": standalone_pass or diversity_pass,
        "f4_required": standalone_pass or diversity_pass,
        "mixed_state_secondary": {
            "base": mixed_variant or "designated_challenger",
            "folds": mixed_secondary,
            "informational_only": True,
        },
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(output_dir / "f3_summary.json", summary)
    _atomic_json(
        manifest_path,
        {
            **manifest,
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "f3_summary": str((output_dir / "f3_summary.json").resolve()),
            "f3_passed": summary["f3_passed"],
        },
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the three-fold P1 F3 screen")
    for name in (
        "store",
        "parent_campaign",
        "selected_sidecar",
        "f2_selection",
        "attribution_report",
        "p0_summary",
        "run_root",
        "phase_b_campaign",
        "phase_c_campaign",
        "external_program",
        "output_dir",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--parallel-processes", type=int, default=2)
    print(run_p1_campaign(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
