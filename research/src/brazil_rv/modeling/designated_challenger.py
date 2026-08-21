from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np

from .analyze import compare_observation_ensembles, load_run_observations
from .contract import ALLOWED_SEEDS, MAX_EPOCHS, RUN_OUTPUT_BASE
from .engine import EvaluationObservations, assert_observations_aligned
from .metrics import primary_validation_score
from .trajectory import simulate_patience3

DESIGNATED_CHALLENGER_NAME = "parent_patience_plus_residual_auxiliary_final_ema_0995"
PARENT_DISCOVERY_ARTIFACT = "trajectory_discovery_e22dd67_20260819T134332Z"
RESIDUAL_DISCOVERY_ARTIFACT = "next_stage_3b60ac9_20260820T233000Z"
RESIDUAL_OFFICIAL_ARTIFACT = "next_stage_official_921dd3a_20260821T085500Z"
PARENT_COMMIT = "e22dd67"
RESIDUAL_COMMIT = "3b60ac9ea6601bf88c0c3157248f6efa8ed374f0"
OFFICIAL_RESIDUAL_COMMIT = "921dd3a3494e7855d97afbdfc4d10b414efafa59"
DISCOVERY_FOLDS = ("fold_a", "fold_b")


def challenger_contract() -> dict[str, object]:
    return {
        "schema": "DESIGNATED_CHALLENGER_V1",
        "name": DESIGNATED_CHALLENGER_NAME,
        "role": "standing_informational_comparator",
        "retention_comparator": "canonical_parent_only",
        "beats_either_allowed": False,
        "members": {
            "parent": {
                "count": 3,
                "seeds": list(ALLOWED_SEEDS),
                "rule": "bidirectional_odd_even_crossfit_patience3_raw",
                "patience": 3,
                "minimum_ic_improvement": 0.0001,
            },
            "residual_auxiliary": {
                "count": 3,
                "seeds": list(ALLOWED_SEEDS),
                "target": "win_wdo_di_level_residual_rank",
                "loss": "soft_spearman",
                "weight": 0.5,
                "head_initialization": "zero_weight_and_bias",
                "trajectory_epochs": 20,
                "sam_rho": 0.125,
                "learning_rate": 0.0003,
                "readout": "final_ema_0995",
            },
        },
        "ensemble": (
            "uniform within-sample/horizon average of tie-aware ranks across all "
            "six members"
        ),
        "learned_weights": False,
        "hyperparameters": {
            "policy": "frozen_to_source_run_manifests_without_overrides",
            "parent_repository_commit": PARENT_COMMIT,
            "residual_repository_commit": RESIDUAL_COMMIT,
            "official_residual_repository_commit": OFFICIAL_RESIDUAL_COMMIT,
        },
        "discovery_artifacts": {
            "parent": PARENT_DISCOVERY_ARTIFACT,
            "residual_auxiliary": RESIDUAL_DISCOVERY_ARTIFACT,
        },
        "official_artifact_reserved_for_next_official_read": (
            RESIDUAL_OFFICIAL_ARTIFACT
        ),
    }


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_run(
    run_dir: Path,
    *,
    fold: str,
    seed: int,
    residual: bool,
) -> None:
    manifest = _read_json(run_dir / "run_manifest.json")
    variant = manifest.get("model", {}).get("variant", {})
    expected_commit = (
        OFFICIAL_RESIDUAL_COMMIT
        if residual and fold == "official"
        else RESIDUAL_COMMIT
        if residual
        else PARENT_COMMIT
    )
    commit = str(manifest.get("repository_commit", ""))
    if (
        manifest.get("status") != "completed"
        or int(manifest.get("seed", -1)) != seed
        or manifest.get("split", {}).get("training") != fold
        or manifest.get("split", {}).get("test_accessed") is not False
        or not commit.startswith(expected_commit)
    ):
        raise ValueError(f"Run differs from the designated contract: {run_dir}")
    if residual and (
        variant.get("name") != "residual_auxiliary"
        or variant.get("auxiliary_target") != "win_wdo_di_level_residual_rank"
        or float(variant.get("auxiliary_weight", -1.0)) != 0.5
        or variant.get("head_initialization") != "zero_weight_and_bias"
        or manifest.get("objective", {}).get("main", {}).get("name") != "soft_spearman"
        or manifest.get("training", {}).get("epochs") != 20
        or 0.995 not in manifest.get("training", {}).get("ema_decays", [])
    ):
        raise ValueError(f"Residual run hyperparameters differ: {run_dir}")


def _crossfit_parent_observations(
    run_dir: Path,
) -> tuple[EvaluationObservations, list[dict[str, object]]]:
    reference = load_run_observations(run_dir, "final_raw")
    epochs = []
    for epoch in range(1, MAX_EPOCHS + 1):
        with np.load(
            run_dir / "validation_predictions" / f"epoch_{epoch:02d}.npz",
            allow_pickle=False,
        ) as values:
            epochs.append(values["raw"].copy())
    dates = np.unique(reference.date_idx)
    if dates.size != 102:
        raise ValueError(f"Expected 102 discovery dates, found {dates.size}")
    parities = {"odd": dates[0::2], "even": dates[1::2]}
    predictions = np.empty_like(reference.predictions)
    directions = []
    for selection_parity, evaluation_parity in (("odd", "even"), ("even", "odd")):
        selection_mask = np.isin(reference.date_idx, parities[selection_parity])
        evaluation_mask = np.isin(reference.date_idx, parities[evaluation_parity])
        scores = [
            primary_validation_score(
                values[selection_mask],
                reference.targets[selection_mask],
                reference.label_mask[selection_mask],
                reference.date_idx[selection_mask],
            )
            for values in epochs
        ]
        replay = simulate_patience3(scores)
        selected_epoch = int(replay["selected_epoch"])
        predictions[evaluation_mask] = epochs[selected_epoch - 1][evaluation_mask]
        directions.append(
            {
                "selection_parity": selection_parity,
                "evaluation_parity": evaluation_parity,
                "selected_epoch": selected_epoch,
                "stopped_epoch": int(replay["stopped_epoch"]),
            }
        )
    return replace(reference, predictions=predictions), directions


def load_designated_challenger_members(
    fold: str,
    *,
    run_root: Path = RUN_OUTPUT_BASE,
) -> dict[str, EvaluationObservations]:
    if fold not in DISCOVERY_FOLDS:
        raise ValueError(f"Unknown discovery fold: {fold}")
    members = {}
    for seed in ALLOWED_SEEDS:
        parent_run = run_root / PARENT_DISCOVERY_ARTIFACT / fold / f"seed_{seed}"
        _validate_run(parent_run, fold=fold, seed=seed, residual=False)
        parent, _ = _crossfit_parent_observations(parent_run)
        residual_run = (
            run_root
            / RESIDUAL_DISCOVERY_ARTIFACT
            / "phase_c"
            / "runs"
            / "residual_auxiliary"
            / fold
            / f"seed_{seed}"
        )
        _validate_run(residual_run, fold=fold, seed=seed, residual=True)
        residual = load_run_observations(residual_run, "final_ema_0995")
        assert_observations_aligned(parent, residual)
        members[f"parent_seed_{seed}"] = parent
        members[f"residual_seed_{seed}"] = residual
    return members


def load_official_challenger_members(
    parent_members: Mapping[str, EvaluationObservations],
    *,
    run_root: Path = RUN_OUTPUT_BASE,
) -> dict[str, EvaluationObservations]:
    expected = {f"seed_{seed}" for seed in ALLOWED_SEEDS}
    if set(parent_members) != expected:
        raise ValueError("Official parent members must be seeds 11, 29, and 47")
    members = {}
    for seed in ALLOWED_SEEDS:
        key = f"seed_{seed}"
        run = (
            run_root
            / RESIDUAL_OFFICIAL_ARTIFACT
            / "official_runs"
            / "residual_auxiliary"
            / key
        )
        _validate_run(run, fold="official", seed=seed, residual=True)
        residual = load_run_observations(run, "final_ema_0995")
        assert_observations_aligned(parent_members[key], residual)
        members[f"parent_{key}"] = parent_members[key]
        members[f"residual_{key}"] = residual
    return members


def compare_discovery_screen(
    candidate_members: Mapping[str, EvaluationObservations],
    parent_members: Mapping[str, EvaluationObservations],
    *,
    fold: str,
    candidate_rule: str,
    parent_rule: str,
    output_dir: Path,
    run_root: Path = RUN_OUTPUT_BASE,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    challenger_members = load_designated_challenger_members(fold, run_root=run_root)
    canonical_parent = {
        f"seed_{seed}": challenger_members[f"parent_seed_{seed}"]
        for seed in ALLOWED_SEEDS
    }
    if set(parent_members) != set(canonical_parent):
        raise ValueError("Canonical parent members must be seeds 11, 29, and 47")
    for name, expected in canonical_parent.items():
        actual = parent_members[name]
        assert_observations_aligned(expected, actual)
        if not np.array_equal(expected.predictions, actual.predictions):
            raise ValueError(f"Canonical parent predictions differ for {name}")
    canonical_dir = compare_observation_ensembles(
        candidate_members,
        parent_members,
        candidate_rule=candidate_rule,
        parent_rule=parent_rule,
        output_dir=output_dir / "vs_canonical",
        comparison_metadata={"fold": fold, "retention_comparator": True},
    )
    challenger_dir = compare_observation_ensembles(
        candidate_members,
        challenger_members,
        candidate_rule=candidate_rule,
        parent_rule=DESIGNATED_CHALLENGER_NAME,
        output_dir=output_dir / "vs_designated_challenger",
        comparison_metadata={"fold": fold, "informational_only": True},
    )
    canonical = _read_json(canonical_dir / "analysis.json")
    challenger = _read_json(challenger_dir / "analysis.json")
    _atomic_json(
        output_dir / "screen_summary.json",
        {
            "schema": "DISCOVERY_SCREEN_WITH_DESIGNATED_CHALLENGER_V1",
            "fold": fold,
            "candidate_rule": candidate_rule,
            "canonical_rule": parent_rule,
            "designated_challenger": challenger_contract(),
            "candidate_minus_canonical_ic": canonical[
                "candidate_minus_parent_primary_ic"
            ],
            "candidate_minus_challenger_ic": challenger[
                "candidate_minus_parent_primary_ic"
            ],
            "selection_contract": {
                "retention_comparator": "canonical_parent_only",
                "challenger_role": "informational_only",
                "beats_either_allowed": False,
            },
            "canonical_analysis": str(canonical_dir / "analysis.json"),
            "challenger_analysis": str(challenger_dir / "analysis.json"),
        },
    )
    return output_dir
