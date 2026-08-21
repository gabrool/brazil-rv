from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .analyze import compare_observation_ensembles, load_run_observations
from .contract import ALLOWED_SEEDS, RUN_OUTPUT_BASE
from .designated_challenger import (
    DESIGNATED_CHALLENGER_NAME,
    PARENT_DISCOVERY_ARTIFACT,
    compare_discovery_screen,
    crossfit_patience3_observations,
    load_designated_challenger_members,
)
from .run_discovery_campaign import EXTERNAL_DATA_READOUT_CONTRACT

DISCOVERY_FOLDS = ("fold_a", "fold_b")
PRIMARY_RULE = "crossfit_patience3_raw"
SECONDARY_RULE = "final_ema_0995"
MINIMUM_MEAN_GAIN = 0.001
MAXIMUM_DIVERSITY_MEMBER_FOLD_LOSS = 0.001


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _analysis_delta(path: Path) -> float:
    return float(_read_json(path)["candidate_minus_parent_primary_ic"])


def compare_external_data_campaign(
    campaign_dir: Path,
    output_dir: Path,
    *,
    run_root: Path = RUN_OUTPUT_BASE,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    campaign = _read_json(campaign_dir / "campaign_manifest.json")
    if (
        campaign.get("status") != "completed"
        or campaign.get("external_sidecar") is None
        or campaign.get("folds") != list(DISCOVERY_FOLDS)
        or campaign.get("seeds") != list(ALLOWED_SEEDS)
        or campaign.get("official_validation_accessed") is not False
        or campaign.get("test_accessed") is not False
        or campaign.get("external_data_readout_contract")
        != EXTERNAL_DATA_READOUT_CONTRACT
        or campaign.get("trajectory_selection") is not None
    ):
        raise ValueError("Campaign is not a completed external-data discovery screen")
    output_dir.mkdir(parents=True)
    fold_rows: dict[str, dict[str, object]] = {}
    patience_replays: dict[str, dict[str, object]] = {}

    for fold in DISCOVERY_FOLDS:
        challenger = load_designated_challenger_members(fold, run_root=run_root)
        parent_patience = {
            f"seed_{seed}": challenger[f"parent_seed_{seed}"] for seed in ALLOWED_SEEDS
        }
        parent_ema = {
            f"seed_{seed}": load_run_observations(
                run_root / PARENT_DISCOVERY_ARTIFACT / fold / f"seed_{seed}",
                SECONDARY_RULE,
            )
            for seed in ALLOWED_SEEDS
        }
        candidate_patience = {}
        candidate_ema = {}
        patience_replays[fold] = {}
        for seed in ALLOWED_SEEDS:
            name = f"seed_{seed}"
            run = campaign_dir / fold / name
            observations, directions = crossfit_patience3_observations(run)
            candidate_patience[name] = observations
            candidate_ema[name] = load_run_observations(run, SECONDARY_RULE)
            patience_replays[fold][name] = directions

        primary_dir = compare_discovery_screen(
            candidate_patience,
            parent_patience,
            fold=fold,
            candidate_rule=PRIMARY_RULE,
            parent_rule=PRIMARY_RULE,
            output_dir=output_dir / fold / "primary_patience3_raw",
            run_root=run_root,
        )
        primary_stack = {
            **{f"parent_{name}": value for name, value in parent_patience.items()},
            **{
                f"candidate_{name}": value for name, value in candidate_patience.items()
            },
        }
        primary_stack_dir = compare_discovery_screen(
            primary_stack,
            parent_patience,
            fold=fold,
            candidate_rule="parent_plus_candidate_crossfit_patience3_raw_uniform_6",
            parent_rule=PRIMARY_RULE,
            output_dir=output_dir / fold / "primary_parent_plus_candidate_stack",
            run_root=run_root,
        )
        matched_ema_dir = compare_observation_ensembles(
            candidate_ema,
            parent_ema,
            candidate_rule=SECONDARY_RULE,
            parent_rule=SECONDARY_RULE,
            output_dir=output_dir / fold / "secondary_ema_0995_vs_matched_parent",
            comparison_metadata={
                "fold": fold,
                "secondary_readout": True,
                "retention_eligible": False,
            },
        )
        challenger_ema_dir = compare_observation_ensembles(
            candidate_ema,
            challenger,
            candidate_rule=SECONDARY_RULE,
            parent_rule=DESIGNATED_CHALLENGER_NAME,
            output_dir=output_dir / fold / "secondary_ema_0995_vs_challenger",
            comparison_metadata={
                "fold": fold,
                "secondary_readout": True,
                "informational_only": True,
            },
        )
        secondary_stack = {
            **{f"parent_{name}": value for name, value in parent_ema.items()},
            **{f"candidate_{name}": value for name, value in candidate_ema.items()},
        }
        matched_ema_stack_dir = compare_observation_ensembles(
            secondary_stack,
            parent_ema,
            candidate_rule="parent_plus_candidate_final_ema_0995_uniform_6",
            parent_rule=SECONDARY_RULE,
            output_dir=output_dir / fold / "secondary_ema_parent_plus_candidate_stack",
            comparison_metadata={
                "fold": fold,
                "secondary_readout": True,
                "additive_information_readout": True,
                "retention_eligible": False,
            },
        )
        primary = _read_json(primary_dir / "screen_summary.json")
        primary_stack_summary = _read_json(primary_stack_dir / "screen_summary.json")
        fold_rows[fold] = {
            "primary_candidate_minus_canonical": float(
                primary["candidate_minus_canonical_ic"]
            ),
            "primary_candidate_minus_challenger": float(
                primary["candidate_minus_challenger_ic"]
            ),
            "primary_stack_minus_canonical": float(
                primary_stack_summary["candidate_minus_canonical_ic"]
            ),
            "primary_stack_minus_challenger": float(
                primary_stack_summary["candidate_minus_challenger_ic"]
            ),
            "secondary_ema_candidate_minus_matched_parent": _analysis_delta(
                matched_ema_dir / "analysis.json"
            ),
            "secondary_ema_candidate_minus_challenger": _analysis_delta(
                challenger_ema_dir / "analysis.json"
            ),
            "secondary_ema_stack_minus_matched_parent": _analysis_delta(
                matched_ema_stack_dir / "analysis.json"
            ),
            "primary_analysis": str(primary_dir / "screen_summary.json"),
            "primary_stack_analysis": str(primary_stack_dir / "screen_summary.json"),
            "secondary_matched_analysis": str(matched_ema_dir / "analysis.json"),
            "secondary_challenger_analysis": str(challenger_ema_dir / "analysis.json"),
        }

    means = {
        key: float(np.mean([float(row[key]) for row in fold_rows.values()]))
        for key in next(iter(fold_rows.values()))
        if key.endswith(("canonical", "challenger", "parent"))
    }
    standalone_passes = means[
        "primary_candidate_minus_canonical"
    ] >= MINIMUM_MEAN_GAIN and all(
        float(row["primary_candidate_minus_canonical"]) >= 0.0
        for row in fold_rows.values()
    )
    diversity_passes = means[
        "primary_stack_minus_canonical"
    ] >= MINIMUM_MEAN_GAIN and all(
        float(row["primary_stack_minus_canonical"]) >= 0.0
        and float(row["primary_candidate_minus_canonical"])
        >= -MAXIMUM_DIVERSITY_MEMBER_FOLD_LOSS
        for row in fold_rows.values()
    )
    _atomic_json(
        output_dir / "screen_summary.json",
        {
            "schema": "EXTERNAL_DATA_DISCOVERY_SCREEN_V1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "campaign": str(campaign_dir.resolve()),
            "external_sidecar": campaign["external_sidecar"],
            "folds": fold_rows,
            "mean_fold_deltas": means,
            "retention_outcomes": {
                "standalone_model_improvement": standalone_passes,
                "parent_plus_candidate_diversity_recipe": diversity_passes,
                "dataset_retained": standalone_passes or diversity_passes,
            },
            "patience_replays": patience_replays,
            "selection_contract": {
                "primary_readout": PRIMARY_RULE,
                "secondary_readout": SECONDARY_RULE,
                "retention_comparator": "canonical_parent_patience3_raw_only",
                "standalone_role": "model_improvement_readout",
                "parent_plus_candidate_stack_role": (
                    "predeclared_diversity_recipe_readout"
                ),
                "parent_plus_candidate_stack_retention_eligible": True,
                "retention_comparator_for_both_paths": "canonical_parent_only",
                "standalone_gate": {
                    "minimum_mean_fold_ic_gain": MINIMUM_MEAN_GAIN,
                    "minimum_each_fold_ic_gain": 0.0,
                },
                "diversity_recipe_gate": {
                    "minimum_mean_fold_ic_gain": MINIMUM_MEAN_GAIN,
                    "minimum_each_fold_ic_gain": 0.0,
                    "maximum_standalone_candidate_loss_each_fold": (
                        MAXIMUM_DIVERSITY_MEMBER_FOLD_LOSS
                    ),
                },
                "secondary_retention_eligible": False,
                "designated_challenger_role": "informational_only",
                "beats_either_allowed": False,
                "official_validation_accessed": False,
                "test_accessed": False,
            },
        },
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare an external-data campaign to the frozen parent and challenger"
    )
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=RUN_OUTPUT_BASE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        compare_external_data_campaign(
            args.campaign_dir,
            args.output_dir,
            run_root=args.run_root,
        )
    )


if __name__ == "__main__":
    main()
