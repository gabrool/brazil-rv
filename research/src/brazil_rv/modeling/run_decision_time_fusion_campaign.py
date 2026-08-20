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
from .model import DECISION_TIME_FUSION_VARIANT
from .provenance import repository_commit
from .train import run_training
from .trajectory import predictions_for_rule, simulate_patience3

FOLDS = ("fold_a", "fold_b")
READOUTS = ("patience3_raw", "final_ema_0995")


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _parent_run(parent_campaign: Path, fold: str, seed: int) -> Path:
    return parent_campaign / fold / f"seed_{seed}"


def _candidate_run(output_dir: Path, fold: str, seed: int) -> Path:
    return output_dir / "runs" / fold / f"seed_{seed}"


def _date_parities(date_idx: np.ndarray) -> dict[str, np.ndarray]:
    dates = np.unique(date_idx)
    if dates.size != 102:
        raise ValueError(f"Expected 102 selection dates, found {dates.size}")
    return {"odd": dates[0::2], "even": dates[1::2]}


def _crossfit_patience(run_dir: Path) -> EvaluationObservations:
    reference = load_run_observations(run_dir, "final_raw")
    epochs = []
    for epoch in range(1, MAX_EPOCHS + 1):
        with np.load(
            run_dir / "validation_predictions" / f"epoch_{epoch:02d}.npz",
            allow_pickle=False,
        ) as values:
            epochs.append(values["raw"].copy())
    predictions = np.empty_like(reference.predictions)
    parities = _date_parities(reference.date_idx)
    for selection, evaluation in (("odd", "even"), ("even", "odd")):
        selection_mask = np.isin(reference.date_idx, parities[selection])
        evaluation_mask = np.isin(reference.date_idx, parities[evaluation])
        scores = [
            primary_validation_score(
                values[selection_mask],
                reference.targets[selection_mask],
                reference.label_mask[selection_mask],
                reference.date_idx[selection_mask],
            )
            for values in epochs
        ]
        selected_epoch = int(simulate_patience3(scores)["selected_epoch"])
        predictions[evaluation_mask] = epochs[selected_epoch - 1][evaluation_mask]
    return replace(reference, predictions=predictions)


def _readout(run_dir: Path, rule: str) -> EvaluationObservations:
    if rule == "patience3_raw":
        return _crossfit_patience(run_dir)
    reference = load_run_observations(run_dir, "final_raw")
    return replace(reference, predictions=predictions_for_rule(run_dir, rule))


def _validate_parent(
    parent_campaign: Path,
    *,
    store: Path,
    identity: Mapping[str, object],
) -> None:
    manifest = json.loads(
        (parent_campaign / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("status") != "completed"
        or Path(str(manifest.get("feature_store"))).resolve() != store.resolve()
        or manifest.get("feature_store_identity") != identity
        or manifest.get("official_validation_accessed") is not False
        or manifest.get("test_accessed") is not False
    ):
        raise ValueError("Trajectory parent campaign does not match the experiment")


def _completed_candidate_matches(
    run_dir: Path,
    *,
    commit: str,
    fold: str,
    seed: int,
) -> bool:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return bool(
        manifest.get("status") == "completed"
        and manifest.get("repository_commit") == commit
        and manifest.get("seed") == seed
        and manifest.get("split", {}).get("training") == fold
        and manifest.get("split", {}).get("test_accessed") is False
        and manifest.get("model", {}).get("variant", {}).get("name")
        == DECISION_TIME_FUSION_VARIANT
    )


def _analyze(output_dir: Path, parent_campaign: Path) -> dict[str, object]:
    summaries = {}
    for readout in READOUTS:
        folds = {}
        for fold in FOLDS:
            candidate_members = {
                f"seed_{seed}": _readout(
                    _candidate_run(output_dir, fold, seed), readout
                )
                for seed in ALLOWED_SEEDS
            }
            parent_members = {
                f"seed_{seed}": _readout(
                    _parent_run(parent_campaign, fold, seed), readout
                )
                for seed in ALLOWED_SEEDS
            }
            analysis_dir = output_dir / "analysis" / readout / fold
            compare_observation_ensembles(
                candidate_members,
                parent_members,
                candidate_rule=readout,
                parent_rule=readout,
                output_dir=analysis_dir,
                comparison_metadata={
                    "variant": DECISION_TIME_FUSION_VARIANT,
                    "fold": fold,
                    "seeds": list(ALLOWED_SEEDS),
                    "parent_rng_stream_preserved": True,
                    "adaptive_checkpoint_crossfit": readout == "patience3_raw",
                    "official_validation_accessed": False,
                    "test_accessed": False,
                },
            )
            report = json.loads(
                (analysis_dir / "analysis.json").read_text(encoding="utf-8")
            )
            folds[fold] = {
                "candidate_ensemble_ic": report["candidate"]["ensemble_ic"],
                "parent_ensemble_ic": report["parent"]["ensemble_ic"],
                "candidate_minus_parent_primary_ic": report[
                    "candidate_minus_parent_primary_ic"
                ],
                "per_date_delta_bootstrap": report["per_date_delta_bootstrap"],
                "horizon_guardrails": report["horizon_guardrails"],
                "time_of_day_guardrails": report["time_of_day_guardrails"],
                "analysis": str((analysis_dir / "analysis.json").resolve()),
            }
        summaries[readout] = {
            "folds": folds,
            "mean_fold_candidate_minus_parent_ic": float(
                np.mean(
                    [folds[fold]["candidate_minus_parent_primary_ic"] for fold in FOLDS]
                )
            ),
        }
    return summaries


def run_campaign(store: Path, parent_campaign: Path, output_dir: Path) -> Path:
    commit = repository_commit()
    identity = feature_store_identity(store)
    _validate_parent(parent_campaign, store=store, identity=identity)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "campaign_manifest.json"
    immutable = {
        "schema": "DECISION_TIME_FUSION_CAMPAIGN",
        "repository_commit": commit,
        "feature_store": str(store.resolve()),
        "feature_store_identity": identity,
        "trajectory_parent": str(parent_campaign.resolve()),
        "variant": DECISION_TIME_FUSION_VARIANT,
        "folds": list(FOLDS),
        "seeds": list(ALLOWED_SEEDS),
        "readouts": list(READOUTS),
        "parent_rng_stream_preserved": True,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    created_at = datetime.now(timezone.utc).isoformat()
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(existing.get(key) != value for key, value in immutable.items()):
            raise ValueError("Existing campaign has a different contract")
        created_at = str(existing["created_at"])
    _atomic_json(
        manifest_path,
        {**immutable, "status": "running", "created_at": created_at},
    )
    for fold in FOLDS:
        for seed in ALLOWED_SEEDS:
            run_dir = _candidate_run(output_dir, fold, seed)
            if run_dir.exists():
                if not _completed_candidate_matches(
                    run_dir, commit=commit, fold=fold, seed=seed
                ):
                    raise ValueError(f"Existing candidate run differs: {run_dir}")
                continue
            run_training(
                store=store,
                seed=seed,
                selection_window=fold,
                run_dir=run_dir,
                variant=DECISION_TIME_FUSION_VARIANT,
            )
    summaries = _analyze(output_dir, parent_campaign)
    completed_at = datetime.now(timezone.utc).isoformat()
    _atomic_json(
        output_dir / "summary.json",
        {
            "schema": "DECISION_TIME_FUSION_ANALYSIS",
            "created_at": completed_at,
            "readouts": summaries,
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    _atomic_json(
        manifest_path,
        {
            **immutable,
            "status": "completed",
            "created_at": created_at,
            "completed_at": completed_at,
            "summary": str((output_dir / "summary.json").resolve()),
        },
    )
    return output_dir


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the corrected decision-time shared-fusion experiment"
    )
    parser.add_argument("--parent-campaign", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(arguments)


def main() -> None:
    args = parse_args()
    print(
        run_campaign(
            resolve_feature_store(),
            args.parent_campaign,
            args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
