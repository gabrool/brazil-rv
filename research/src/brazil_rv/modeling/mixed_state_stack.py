from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .analyze import compare_observation_ensembles, load_run_observations
from .contract import ALLOWED_SEEDS
from .designated_challenger import (
    DESIGNATED_CHALLENGER_NAME,
    PARENT_DISCOVERY_ARTIFACT,
    load_designated_challenger_members,
)
from .engine import EvaluationObservations

FOLDS = ("fold_a", "fold_b")
VARIANTS = {
    "all_listed_members": (
        "residual",
        "combined_aux",
        "options",
        "lending",
        "regular_activity",
        "adr",
        "market_gate",
    ),
    "residual_options_adr": ("residual", "options", "adr"),
}
GATE_MEAN = 0.001


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _validate_run(run: Path, fold: str, seed: int) -> None:
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "completed"
        or int(manifest.get("seed", -1)) != seed
        or manifest.get("split", {}).get("training") != fold
        or manifest.get("split", {}).get("test_accessed") is not False
        or not (run / "validation_predictions" / "epoch_20.npz").is_file()
        or not (run / "validation_reference.npz").is_file()
    ):
        raise ValueError(f"Mixed-state source run differs: {run}")


def _ema_members(
    path: Path, fold: str, family: str
) -> dict[str, EvaluationObservations]:
    members = {}
    for seed in ALLOWED_SEEDS:
        run = path / fold / f"seed_{seed}"
        _validate_run(run, fold, seed)
        members[f"{family}_seed_{seed}"] = load_run_observations(run, "final_ema_0995")
    return members


def load_mixed_state_variant_members(
    variant: str,
    fold: str,
    *,
    run_root: Path,
    phase_b_campaign: Path,
    phase_c_campaign: Path,
    external_program: Path,
) -> dict[str, EvaluationObservations]:
    if variant not in VARIANTS or fold not in FOLDS:
        raise ValueError("Unknown mixed-state variant or fold")
    challenger = load_designated_challenger_members(fold, run_root=run_root)
    parent = {
        f"parent_seed_{seed}": challenger[f"parent_seed_{seed}"]
        for seed in ALLOWED_SEEDS
    }
    paths = {
        "combined_aux": phase_b_campaign / "runs" / "combined",
        "market_gate": phase_c_campaign / "runs" / "competitive_feature_gate",
        "lending": external_program / "campaigns" / "d1_bdi_lending",
        "options": external_program / "campaigns" / "d3_options",
        "regular_activity": external_program / "campaigns" / "d9_regular_activity",
        "adr": external_program / "campaigns" / "d10_adr",
    }
    families = {
        "residual": {
            name: member
            for name, member in challenger.items()
            if name.startswith("residual_")
        },
        **{family: _ema_members(path, fold, family) for family, path in paths.items()},
    }
    members = dict(parent)
    for family in VARIANTS[variant]:
        members.update(families[family])
    return members


def run_mixed_state_stack(
    *,
    run_root: Path,
    phase_b_campaign: Path,
    phase_c_campaign: Path,
    external_program: Path,
    output_dir: Path,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    summaries: dict[str, object] = {}
    for fold in FOLDS:
        challenger = load_designated_challenger_members(fold, run_root=run_root)
        parent = {
            f"parent_seed_{seed}": challenger[f"parent_seed_{seed}"]
            for seed in ALLOWED_SEEDS
        }
        summaries[fold] = {}
        for variant, family_names in VARIANTS.items():
            candidate = load_mixed_state_variant_members(
                variant,
                fold,
                run_root=run_root,
                phase_b_campaign=phase_b_campaign,
                phase_c_campaign=phase_c_campaign,
                external_program=external_program,
            )
            base = output_dir / variant / fold
            canonical_dir = compare_observation_ensembles(
                candidate,
                parent,
                candidate_rule="uniform_mixed_state_rank_average",
                parent_rule="crossfit_patience3_raw",
                output_dir=base / "vs_canonical",
                comparison_metadata={
                    "variant": variant,
                    "composition_families": ["parent", *family_names],
                    "criterion_selected_on_reporting_folds": True,
                    "confirmation_status": "discovery_only",
                    "retention_comparator": True,
                    "ensemble_weights_learned": False,
                    "official_validation_accessed": False,
                    "test_accessed": False,
                },
            )
            challenger_dir = compare_observation_ensembles(
                candidate,
                challenger,
                candidate_rule="uniform_mixed_state_rank_average",
                parent_rule=DESIGNATED_CHALLENGER_NAME,
                output_dir=base / "vs_designated_challenger",
                comparison_metadata={
                    "variant": variant,
                    "informational_only": True,
                    "beats_either_allowed": False,
                    "ensemble_weights_learned": False,
                    "official_validation_accessed": False,
                    "test_accessed": False,
                },
            )
            canonical = json.loads(
                (canonical_dir / "analysis.json").read_text(encoding="utf-8")
            )
            designated = json.loads(
                (challenger_dir / "analysis.json").read_text(encoding="utf-8")
            )
            summaries[fold][variant] = {
                "member_count": len(candidate),
                "candidate_minus_canonical_ic": canonical[
                    "candidate_minus_parent_primary_ic"
                ],
                "candidate_minus_designated_challenger_ic": designated[
                    "candidate_minus_parent_primary_ic"
                ],
                "vs_canonical_analysis": str(canonical_dir / "analysis.json"),
                "vs_challenger_analysis": str(challenger_dir / "analysis.json"),
            }
    variants = {}
    for variant in VARIANTS:
        deltas = [
            float(summaries[fold][variant]["candidate_minus_canonical_ic"])
            for fold in FOLDS
        ]
        variants[variant] = {
            "folds": {fold: summaries[fold][variant] for fold in FOLDS},
            "mean_candidate_minus_canonical_ic": float(np.mean(deltas)),
            "canonical_gate_passed": bool(
                np.mean(deltas) >= GATE_MEAN and all(value >= 0 for value in deltas)
            ),
        }
    passed = [
        name for name, value in variants.items() if value["canonical_gate_passed"]
    ]
    selected = (
        min(
            passed,
            key=lambda name: (
                -float(variants[name]["mean_candidate_minus_canonical_ic"]),
                name,
            ),
        )
        if passed
        else None
    )
    report = {
        "schema": "P0_GRAND_MIXED_STATE_STACK_V1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_artifact": PARENT_DISCOVERY_ARTIFACT,
        "variants": variants,
        "selected_discovery_variant": selected,
        "selection_warning": (
            "The family-inclusion criterion was derived from these same folds; a "
            "pass is discovery evidence only and requires the later bundled official read."
        ),
        "retention_rule": (
            "mean candidate-minus-canonical IC >= +0.001 and every fold >= 0; "
            "designated challenger is informational only"
        ),
        "ensemble_weights_learned": False,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(output_dir / "mixed_state_summary.json", report)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the P0.1 mixed-state analysis")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--phase-b-campaign", type=Path, required=True)
    parser.add_argument("--phase-c-campaign", type=Path, required=True)
    parser.add_argument("--external-program", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(run_mixed_state_stack(**vars(args)))


if __name__ == "__main__":
    main()
