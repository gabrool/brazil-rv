from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .analyze import compare_observation_ensembles, load_run_observations
from .contract import ALLOWED_SEEDS
from .engine import EvaluationObservations

RULE = "patience3_raw"
VALIDATION_DATE_COUNT = 244


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _manifest(run_dir: Path) -> dict[str, object]:
    value = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if value.get("status") != "completed" or value.get("split", {}).get(
        "test_accessed"
    ) is not False:
        raise ValueError(f"Run is not completed and test-clean: {run_dir}")
    return value


def _seeded_runs(run_dirs: Sequence[Path], *, candidate: bool) -> dict[int, Path]:
    runs: dict[int, Path] = {}
    for run_dir in run_dirs:
        manifest = _manifest(run_dir)
        seed = int(manifest["seed"])
        if seed in runs:
            raise ValueError(f"Duplicate seed {seed}")
        split = manifest["split"]
        if candidate:
            frozen = manifest.get("frozen_selection")
            variant = manifest.get("model", {}).get("variant", {})
            if (
                split.get("training") != "official"
                or not isinstance(frozen, dict)
                or frozen.get("selected_rule") != RULE
                or variant.get("name") != "multi_depth_stats"
            ):
                raise ValueError(f"Not a frozen official multi-depth run: {run_dir}")
        elif (
            split.get("training") != "train"
            or split.get("selection") != "validation"
        ):
            raise ValueError(f"Not a matched official parent run: {run_dir}")
        runs[seed] = run_dir
    if set(runs) != set(ALLOWED_SEEDS):
        raise ValueError(f"Expected seeds {ALLOWED_SEEDS}, found {sorted(runs)}")
    return runs


def _saved_parent_observations(run_dir: Path) -> EvaluationObservations:
    path = run_dir / "validation_observations.npz"
    with np.load(path, allow_pickle=False) as values:
        expected = set(EvaluationObservations.__dataclass_fields__)
        if set(values.files) != expected:
            raise ValueError(f"Unexpected observation fields: {path}")
        observations = EvaluationObservations(
            **{name: values[name].copy() for name in expected}
        )
    if np.unique(observations.date_idx).size != VALIDATION_DATE_COUNT:
        raise ValueError(f"Parent observations are not official validation: {path}")
    return observations


def run_confirmation(
    parent_run_dirs: Sequence[Path],
    multi_depth_run_dirs: Sequence[Path],
    output_dir: Path,
) -> Path:
    parents = _seeded_runs(parent_run_dirs, candidate=False)
    multi_depth = _seeded_runs(multi_depth_run_dirs, candidate=True)
    parent_identity = {
        json.dumps(_manifest(path)["feature_store_identity"], sort_keys=True)
        for path in parents.values()
    }
    candidate_identity = {
        json.dumps(_manifest(path)["feature_store_identity"], sort_keys=True)
        for path in multi_depth.values()
    }
    if len(parent_identity) != 1 or parent_identity != candidate_identity:
        raise ValueError("Parent and multi-depth feature-store identities differ")

    parent_members = {
        f"parent_seed_{seed}": _saved_parent_observations(path)
        for seed, path in sorted(parents.items())
    }
    multi_members = {
        f"multi_depth_seed_{seed}": load_run_observations(path, RULE)
        for seed, path in sorted(multi_depth.items())
    }
    if any(
        np.unique(observations.date_idx).size != VALIDATION_DATE_COUNT
        for observations in multi_members.values()
    ):
        raise ValueError("Multi-depth observations are not official validation")

    compare_observation_ensembles(
        {**parent_members, **multi_members},
        parent_members,
        candidate_rule="uniform_parent_three_plus_multi_depth_three",
        parent_rule="uniform_parent_three",
        output_dir=output_dir,
        comparison_metadata={
            "purpose": "single sparse Phase A/Phase B stage-winner confirmation",
            "checkpoint_rule": RULE,
            "ensemble_weights_learned": False,
            "official_validation_accessed": True,
            "test_accessed": False,
            "campaign_driver": False,
        },
    )
    _atomic_json(
        output_dir / "confirmation_manifest.json",
        {
            "schema": "OFFICIAL_PHASE_A_DIVERSITY_CONFIRMATION",
            "parent_runs": [str(path.resolve()) for path in parents.values()],
            "multi_depth_runs": [
                str(path.resolve()) for path in multi_depth.values()
            ],
            "checkpoint_rule": RULE,
            "composition": "parent_three_plus_multi_depth_three",
            "ensemble_weights_learned": False,
            "official_validation_accessed": True,
            "test_accessed": False,
            "campaign_driver": False,
        },
    )
    return output_dir


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Confirm the frozen six-member recipe on official validation"
    )
    parser.add_argument("--parent-run", action="append", type=Path, required=True)
    parser.add_argument(
        "--multi-depth-run", action="append", type=Path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(arguments)


def main() -> None:
    args = parse_args()
    print(run_confirmation(args.parent_run, args.multi_depth_run, args.output_dir))


if __name__ == "__main__":
    main()
