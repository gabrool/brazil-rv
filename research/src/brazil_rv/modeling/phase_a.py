from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .analyze import compare_observation_ensembles, load_run_observations
from .contract import ALLOWED_SEEDS, MAX_EPOCHS
from .data import feature_store_identity, resolve_feature_store
from .engine import EvaluationObservations
from .metrics import primary_validation_score
from .model import PHASE_A_MODEL_VARIANTS
from .provenance import repository_commit
from .train import run_training
from .trajectory import predictions_for_rule, simulate_patience3

DISCOVERY_FOLDS = ("fold_a", "fold_b")
PHASE_A_READOUTS = ("patience3_raw", "final_ema_0995")
STAGE_ONE_SEEDS = (29,)
THREE_SEED_COMPLETION = ALLOWED_SEEDS
PROMOTION_RULE = (
    "stop after seed 29 only when patience3_raw candidate-minus-parent IC is "
    "<= -0.003 on both folds and final_ema_0995 candidate-minus-parent IC is "
    "<= -0.002 on both folds; otherwise complete seeds 11/29/47"
)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _run_manifest(run_dir: Path) -> dict[str, object]:
    return json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))


def _variant_from_manifest(manifest: Mapping[str, object]) -> str:
    model = manifest.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("Run manifest has no model metadata")
    variant = model.get("variant")
    if not isinstance(variant, Mapping) or not isinstance(variant.get("name"), str):
        raise ValueError("Run manifest has no model variant")
    return str(variant["name"])


def _parent_run(parent_campaign: Path, fold: str, seed: int) -> Path:
    return parent_campaign / fold / f"seed_{seed}"


def _candidate_run(output_dir: Path, variant: str, fold: str, seed: int) -> Path:
    return output_dir / "runs" / variant / fold / f"seed_{seed}"


def _validate_parent_campaign(
    parent_campaign: Path,
    *,
    store: Path,
    identity: Mapping[str, object],
) -> None:
    campaign = json.loads(
        (parent_campaign / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    if (
        campaign.get("status") != "completed"
        or Path(str(campaign.get("feature_store"))).resolve() != store.resolve()
        or campaign.get("feature_store_identity") != identity
        or campaign.get("official_validation_accessed") is not False
        or campaign.get("test_accessed") is not False
    ):
        raise ValueError("Trajectory parent campaign does not match Phase A")
    for fold in DISCOVERY_FOLDS:
        for seed in ALLOWED_SEEDS:
            run_dir = _parent_run(parent_campaign, fold, seed)
            manifest = _run_manifest(run_dir)
            if (
                manifest.get("status") != "completed"
                or manifest.get("seed") != seed
                or manifest.get("split", {}).get("training") != fold
                or manifest.get("split", {}).get("test_accessed") is not False
            ):
                raise ValueError(f"Trajectory parent run does not match: {run_dir}")


def _completed_candidate_matches(
    run_dir: Path,
    *,
    store: Path,
    identity: Mapping[str, object],
    commit: str,
    variant: str,
    fold: str,
    seed: int,
) -> bool:
    if not (run_dir / "run_manifest.json").is_file():
        return False
    manifest = _run_manifest(run_dir)
    return bool(
        manifest.get("status") == "completed"
        and manifest.get("repository_commit") == commit
        and Path(str(manifest.get("feature_store"))).resolve() == store.resolve()
        and manifest.get("feature_store_identity") == identity
        and manifest.get("seed") == seed
        and manifest.get("split", {}).get("training") == fold
        and manifest.get("split", {}).get("test_accessed") is False
        and _variant_from_manifest(manifest) == variant
    )


def _date_parities(date_idx: np.ndarray) -> dict[str, np.ndarray]:
    dates = np.unique(date_idx)
    if dates.size != 102:
        raise ValueError(f"Phase A expected 102 selection dates, found {dates.size}")
    return {"odd": dates[0::2], "even": dates[1::2]}


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
    predictions = np.empty_like(reference.predictions)
    directions = []
    parities = _date_parities(reference.date_idx)
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
            for values in epoch_predictions
        ]
        replay = simulate_patience3(scores)
        selected_epoch = int(replay["selected_epoch"])
        predictions[evaluation_mask] = epoch_predictions[selected_epoch - 1][
            evaluation_mask
        ]
        directions.append(
            {
                "selection_parity": selection_parity,
                "evaluation_parity": evaluation_parity,
                "selected_epoch": selected_epoch,
                "stopped_epoch": int(replay["stopped_epoch"]),
                "selection_half_ic": float(replay["selected_score"]),
            }
        )
    return replace(reference, predictions=predictions), directions


def _readout_observations(
    run_dir: Path, readout: str
) -> tuple[EvaluationObservations, list[dict[str, object]]]:
    if readout == "patience3_raw":
        return crossfit_patience_observations(run_dir)
    if readout != "final_ema_0995":
        raise ValueError(f"Unsupported Phase A readout: {readout}")
    return (
        replace(
            load_run_observations(run_dir, "final_raw"),
            predictions=predictions_for_rule(run_dir, readout),
        ),
        [],
    )


def _analyze_variant(
    *,
    variant: str,
    seeds: Sequence[int],
    stage: str,
    output_dir: Path,
    parent_campaign: Path,
) -> dict[str, object]:
    summary_path = output_dir / "analysis" / stage / variant / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("variant") != variant
            or summary.get("stage") != stage
            or summary.get("seeds") != list(seeds)
        ):
            raise ValueError(f"Existing Phase A analysis differs: {summary_path}")
        return summary
    readout_summaries: dict[str, object] = {}
    for readout in PHASE_A_READOUTS:
        fold_summaries = {}
        for fold in DISCOVERY_FOLDS:
            candidate_members = {}
            parent_members = {}
            candidate_replays = {}
            parent_replays = {}
            for seed in seeds:
                member = f"seed_{seed}"
                candidate, candidate_replay = _readout_observations(
                    _candidate_run(output_dir, variant, fold, seed), readout
                )
                parent, parent_replay = _readout_observations(
                    _parent_run(parent_campaign, fold, seed), readout
                )
                candidate_members[member] = candidate
                parent_members[member] = parent
                candidate_replays[member] = candidate_replay
                parent_replays[member] = parent_replay
            comparison_dir = output_dir / "analysis" / stage / variant / fold / readout
            compare_observation_ensembles(
                candidate_members,
                parent_members,
                candidate_rule=readout,
                parent_rule=readout,
                output_dir=comparison_dir,
                comparison_metadata={
                    "variant": variant,
                    "stage": stage,
                    "fold": fold,
                    "seeds": list(seeds),
                    "adaptive_checkpoint_crossfit": readout == "patience3_raw",
                    "candidate_patience_replays": candidate_replays,
                    "parent_patience_replays": parent_replays,
                    "official_validation_accessed": False,
                    "test_accessed": False,
                },
            )
            report = json.loads(
                (comparison_dir / "analysis.json").read_text(encoding="utf-8")
            )
            fold_summaries[fold] = {
                "candidate_ensemble_ic": report["candidate"]["ensemble_ic"],
                "parent_ensemble_ic": report["parent"]["ensemble_ic"],
                "candidate_minus_parent_primary_ic": report[
                    "candidate_minus_parent_primary_ic"
                ],
                "per_date_delta_bootstrap": report["per_date_delta_bootstrap"],
                "horizon_guardrails": report["horizon_guardrails"],
                "time_of_day_guardrails": report["time_of_day_guardrails"],
                "analysis": str((comparison_dir / "analysis.json").resolve()),
            }
        readout_summaries[readout] = {
            "folds": fold_summaries,
            "mean_fold_candidate_ic": float(
                np.mean(
                    [
                        fold_summaries[fold]["candidate_ensemble_ic"]
                        for fold in DISCOVERY_FOLDS
                    ]
                )
            ),
            "mean_fold_parent_ic": float(
                np.mean(
                    [
                        fold_summaries[fold]["parent_ensemble_ic"]
                        for fold in DISCOVERY_FOLDS
                    ]
                )
            ),
            "mean_fold_candidate_minus_parent_ic": float(
                np.mean(
                    [
                        fold_summaries[fold]["candidate_minus_parent_primary_ic"]
                        for fold in DISCOVERY_FOLDS
                    ]
                )
            ),
        }
    summary = {
        "schema": "PHASE_A_VARIANT_ANALYSIS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "variant": variant,
        "stage": stage,
        "seeds": list(seeds),
        "readouts": readout_summaries,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(summary_path, summary)
    return summary


def promote_after_stage_one(summary: Mapping[str, object]) -> bool:
    readouts = summary["readouts"]
    patience = readouts["patience3_raw"]["folds"]
    ema = readouts["final_ema_0995"]["folds"]
    clear_patience_loss = all(
        patience[fold]["candidate_minus_parent_primary_ic"] <= -0.003
        for fold in DISCOVERY_FOLDS
    )
    clear_ema_loss = all(
        ema[fold]["candidate_minus_parent_primary_ic"] <= -0.002
        for fold in DISCOVERY_FOLDS
    )
    return not (clear_patience_loss and clear_ema_loss)


def _run_candidate(
    *,
    store: Path,
    identity: Mapping[str, object],
    commit: str,
    output_dir: Path,
    variant: str,
    fold: str,
    seed: int,
) -> Path:
    run_dir = _candidate_run(output_dir, variant, fold, seed)
    if run_dir.exists():
        if not _completed_candidate_matches(
            run_dir,
            store=store,
            identity=identity,
            commit=commit,
            variant=variant,
            fold=fold,
            seed=seed,
        ):
            raise ValueError(f"Existing Phase A run differs: {run_dir}")
        return run_dir
    return run_training(
        store=store,
        seed=seed,
        selection_window=fold,
        run_dir=run_dir,
        variant=variant,
    )


def run_phase_a_campaign(
    store: Path,
    parent_campaign: Path,
    output_dir: Path,
) -> Path:
    commit = repository_commit()
    identity = feature_store_identity(store)
    _validate_parent_campaign(parent_campaign, store=store, identity=identity)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "campaign_manifest.json"
    immutable = {
        "schema": "PHASE_A_CAMPAIGN",
        "repository_commit": commit,
        "feature_store": str(store.resolve()),
        "feature_store_identity": identity,
        "trajectory_parent": str(parent_campaign.resolve()),
        "variants": list(PHASE_A_MODEL_VARIANTS),
        "stage_one_seeds": list(STAGE_ONE_SEEDS),
        "three_seed_completion": list(THREE_SEED_COMPLETION),
        "promotion_rule": PROMOTION_RULE,
        "readouts": list(PHASE_A_READOUTS),
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    created_at = datetime.now(timezone.utc).isoformat()
    existing_results = {}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(existing.get(key) != value for key, value in immutable.items()):
            raise ValueError("Existing Phase A campaign has a different contract")
        created_at = str(existing["created_at"])
        existing_results = dict(existing.get("results", {}))
    _atomic_json(
        manifest_path,
        {
            **immutable,
            "status": "running",
            "created_at": created_at,
            "results": existing_results,
        },
    )

    results = existing_results
    for variant in PHASE_A_MODEL_VARIANTS:
        for fold in DISCOVERY_FOLDS:
            _run_candidate(
                store=store,
                identity=identity,
                commit=commit,
                output_dir=output_dir,
                variant=variant,
                fold=fold,
                seed=STAGE_ONE_SEEDS[0],
            )
        stage_one = _analyze_variant(
            variant=variant,
            seeds=STAGE_ONE_SEEDS,
            stage="stage_one",
            output_dir=output_dir,
            parent_campaign=parent_campaign,
        )
        promoted = promote_after_stage_one(stage_one)
        final_analysis = stage_one
        if promoted:
            for fold in DISCOVERY_FOLDS:
                for seed in THREE_SEED_COMPLETION:
                    _run_candidate(
                        store=store,
                        identity=identity,
                        commit=commit,
                        output_dir=output_dir,
                        variant=variant,
                        fold=fold,
                        seed=seed,
                    )
            final_analysis = _analyze_variant(
                variant=variant,
                seeds=THREE_SEED_COMPLETION,
                stage="three_seed",
                output_dir=output_dir,
                parent_campaign=parent_campaign,
            )
        results[variant] = {
            "promoted_after_stage_one": promoted,
            "stage_one_analysis": str(
                (
                    output_dir / "analysis" / "stage_one" / variant / "summary.json"
                ).resolve()
            ),
            "final_stage": final_analysis["stage"],
            "final_analysis": str(
                (
                    output_dir
                    / "analysis"
                    / str(final_analysis["stage"])
                    / variant
                    / "summary.json"
                ).resolve()
            ),
        }
        _atomic_json(
            manifest_path,
            {
                **immutable,
                "status": "running",
                "created_at": created_at,
                "results": results,
            },
        )

    _atomic_json(
        manifest_path,
        {
            **immutable,
            "status": "completed",
            "created_at": created_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "results": results,
        },
    )
    return output_dir


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the staged six-candidate Phase A representation screen"
    )
    parser.add_argument("--parent-campaign", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(arguments)


def main() -> None:
    args = parse_args()
    store = resolve_feature_store()
    print(
        run_phase_a_campaign(
            store,
            args.parent_campaign,
            args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
