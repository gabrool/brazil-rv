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
from .engine import EvaluationObservations, assert_observations_aligned
from .metrics import primary_validation_score
from .provenance import repository_commit
from .trajectory import simulate_patience3

DISCOVERY_FOLDS = ("fold_a", "fold_b")
DISCOVERY_MARGIN = 0.001


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def discovery_gate_passed(fold_deltas: Mapping[str, float]) -> bool:
    if set(fold_deltas) != set(DISCOVERY_FOLDS):
        raise ValueError("EMA stack gate requires exactly fold_a and fold_b")
    return all(float(fold_deltas[fold]) >= DISCOVERY_MARGIN for fold in DISCOVERY_FOLDS)


def _date_parities(date_idx: np.ndarray) -> dict[str, np.ndarray]:
    dates = np.unique(date_idx)
    if dates.size != 102:
        raise ValueError(f"Expected 102 discovery selection dates, found {dates.size}")
    return {"odd": dates[0::2], "even": dates[1::2]}


def crossfit_patience_observations(
    run_dir: Path,
) -> tuple[EvaluationObservations, list[dict[str, object]]]:
    reference = load_run_observations(run_dir, "final_raw")
    raw_epochs = []
    for epoch in range(1, MAX_EPOCHS + 1):
        with np.load(
            run_dir / "validation_predictions" / f"epoch_{epoch:02d}.npz",
            allow_pickle=False,
        ) as values:
            raw_epochs.append(values["raw"].copy())
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
            for values in raw_epochs
        ]
        replay = simulate_patience3(scores)
        selected_epoch = int(replay["selected_epoch"])
        predictions[evaluation_mask] = raw_epochs[selected_epoch - 1][evaluation_mask]
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


def _discovery_run(root: Path, fold: str, seed: int) -> Path:
    return root / fold / f"seed_{seed}"


def _residual_discovery_run(root: Path, fold: str, seed: int) -> Path:
    return root / "runs" / "residual_auxiliary" / fold / f"seed_{seed}"


def _validate_discovery_run(
    run: Path, *, fold: str, seed: int, residual: bool
) -> None:
    manifest = _read_json(run / "run_manifest.json")
    variant = manifest.get("model", {}).get("variant", {}).get("name")
    if (
        manifest.get("status") != "completed"
        or int(manifest.get("seed", -1)) != seed
        or manifest.get("split", {}).get("training") != fold
        or manifest.get("split", {}).get("test_accessed") is not False
        or (residual and variant != "residual_auxiliary")
    ):
        raise ValueError(f"Invalid discovery run: {run}")


def _comparison_row(path: Path) -> dict[str, object]:
    report = _read_json(path / "analysis.json")
    return {
        "ema_stack_ic": report["candidate"]["ensemble_ic"],
        "patience_stack_ic": report["parent"]["ensemble_ic"],
        "ema_minus_patience_stack_ic": report["candidate_minus_parent_primary_ic"],
        "per_date_delta_bootstrap": report["per_date_delta_bootstrap"],
        "horizon_guardrails": report["horizon_guardrails"],
        "time_of_day_guardrails": report["time_of_day_guardrails"],
        "analysis": str((path / "analysis.json").resolve()),
    }


def _run_discovery(
    parent_campaign: Path,
    residual_campaign: Path,
    output_dir: Path,
) -> dict[str, object]:
    folds = {}
    for fold in DISCOVERY_FOLDS:
        ema_stack = {}
        patience_stack = {}
        parent_replays = {}
        residual_replays = {}
        for seed in ALLOWED_SEEDS:
            key = f"seed_{seed}"
            parent_run = _discovery_run(parent_campaign, fold, seed)
            _validate_discovery_run(
                parent_run, fold=fold, seed=seed, residual=False
            )
            parent, parent_replay = crossfit_patience_observations(parent_run)
            residual_run = _residual_discovery_run(residual_campaign, fold, seed)
            _validate_discovery_run(
                residual_run, fold=fold, seed=seed, residual=True
            )
            residual_patience, residual_replay = crossfit_patience_observations(
                residual_run
            )
            residual_ema = load_run_observations(residual_run, "final_ema_0995")
            assert_observations_aligned(residual_patience, residual_ema)
            ema_stack[f"parent_{key}"] = parent
            ema_stack[f"residual_{key}"] = residual_ema
            patience_stack[f"parent_{key}"] = parent
            patience_stack[f"residual_{key}"] = residual_patience
            parent_replays[key] = parent_replay
            residual_replays[key] = residual_replay
        comparison = output_dir / "discovery" / fold
        compare_observation_ensembles(
            ema_stack,
            patience_stack,
            candidate_rule="parent_patience_plus_residual_final_ema_0995",
            parent_rule="parent_patience_plus_residual_patience",
            output_dir=comparison,
            comparison_metadata={
                "zero_training": True,
                "fold": fold,
                "seeds": list(ALLOWED_SEEDS),
                "parent_patience_replays": parent_replays,
                "residual_patience_replays": residual_replays,
                "official_validation_accessed": False,
                "test_accessed": False,
            },
        )
        folds[fold] = _comparison_row(comparison)
    deltas = {
        fold: float(row["ema_minus_patience_stack_ic"])
        for fold, row in folds.items()
    }
    return {
        "folds": folds,
        "gate": {
            "threshold_each_fold": DISCOVERY_MARGIN,
            "fold_deltas": deltas,
            "passed": discovery_gate_passed(deltas),
        },
    }


def _load_legacy_observations(path: Path) -> EvaluationObservations:
    with np.load(path, allow_pickle=False) as values:
        return EvaluationObservations(
            **{
                name: values[name].copy()
                for name in EvaluationObservations.__dataclass_fields__
            }
        )


def _parent_official_members(
    parent_reproduction: Path,
) -> dict[str, EvaluationObservations]:
    members = {}
    for path in parent_reproduction.rglob("validation_observations.npz"):
        manifest_path = path.parent / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "completed":
            continue
        seed = int(manifest["seed"])
        if seed not in ALLOWED_SEEDS:
            continue
        key = f"seed_{seed}"
        if key in members:
            raise ValueError(f"Duplicate parent official observations for {key}")
        members[key] = _load_legacy_observations(path)
    expected = {f"seed_{seed}" for seed in ALLOWED_SEEDS}
    if set(members) != expected:
        raise ValueError(f"Parent official members differ: {sorted(members)}")
    reference = members["seed_11"]
    for member in members.values():
        assert_observations_aligned(reference, member)
    if np.unique(reference.date_idx).size != 244:
        raise ValueError("Official comparison requires exactly 244 validation dates")
    return members


def _official_residual_run(official_campaign: Path, seed: int) -> Path:
    run = official_campaign / "official_runs" / "residual_auxiliary" / f"seed_{seed}"
    manifest = _read_json(run / "run_manifest.json")
    if (
        manifest.get("status") != "completed"
        or int(manifest.get("seed", -1)) != seed
        or manifest.get("split", {}).get("training") != "official"
        or manifest.get("split", {}).get("test_accessed") is not False
        or manifest.get("model", {}).get("variant", {}).get("name")
        != "residual_auxiliary"
    ):
        raise ValueError(f"Invalid official residual run: {run}")
    return run


def _run_official(
    parent_reproduction: Path,
    official_campaign: Path,
    output_dir: Path,
) -> dict[str, object]:
    stage3_manifest = _read_json(official_campaign / "stage3_manifest.json")
    if (
        stage3_manifest.get("status") != "completed"
        or stage3_manifest.get("stack_variants") != ["residual_auxiliary"]
        or stage3_manifest.get("test_accessed") is not False
    ):
        raise ValueError("Official campaign is incomplete, different, or crossed test")
    parent = _parent_official_members(parent_reproduction)
    ema_stack = {}
    patience_stack = {}
    for seed in ALLOWED_SEEDS:
        key = f"seed_{seed}"
        run = _official_residual_run(official_campaign, seed)
        ema_stack[f"parent_{key}"] = parent[key]
        ema_stack[f"residual_{key}"] = load_run_observations(
            run, "final_ema_0995"
        )
        patience_stack[f"parent_{key}"] = parent[key]
        patience_stack[f"residual_{key}"] = load_run_observations(
            run, "patience3_raw"
        )
    comparison = output_dir / "official"
    compare_observation_ensembles(
        ema_stack,
        patience_stack,
        candidate_rule="parent_patience_plus_residual_final_ema_0995",
        parent_rule="parent_patience_plus_residual_patience",
        output_dir=comparison,
        comparison_metadata={
            "zero_training": True,
            "seeds": list(ALLOWED_SEEDS),
            "discovery_gate_threshold_each_fold": DISCOVERY_MARGIN,
            "official_validation_accessed": True,
            "test_accessed": False,
        },
    )
    return _comparison_row(comparison)


def run_ema_residual_stack_reanalysis(
    parent_campaign: Path,
    residual_campaign: Path,
    parent_reproduction: Path,
    official_campaign: Path,
    output_dir: Path,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    immutable = {
        "schema": "EMA_RESIDUAL_STACK_REANALYSIS",
        "repository_commit": repository_commit(),
        "parent_discovery_campaign": str(parent_campaign.resolve()),
        "residual_discovery_campaign": str(residual_campaign.resolve()),
        "parent_reproduction": str(parent_reproduction.resolve()),
        "official_campaign": str(official_campaign.resolve()),
        "seeds": list(ALLOWED_SEEDS),
        "discovery_gate_threshold_each_fold": DISCOVERY_MARGIN,
        "test_accessed": False,
    }
    manifest_path = output_dir / "manifest.json"
    created_at = datetime.now(timezone.utc).isoformat()
    _atomic_json(
        manifest_path,
        {
            **immutable,
            "created_at": created_at,
            "status": "discovery_running",
            "official_validation_accessed": False,
        },
    )
    discovery = _run_discovery(parent_campaign, residual_campaign, output_dir)
    if not discovery["gate"]["passed"]:
        _atomic_json(
            manifest_path,
            {
                **immutable,
                "created_at": created_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "status": "completed",
                "discovery": discovery,
                "official_validation_accessed": False,
                "official": None,
            },
        )
        return output_dir
    _atomic_json(
        manifest_path,
        {
            **immutable,
            "created_at": created_at,
            "status": "official_running",
            "discovery": discovery,
            "official_validation_accessed": True,
        },
    )
    official = _run_official(parent_reproduction, official_campaign, output_dir)
    _atomic_json(
        manifest_path,
        {
            **immutable,
            "created_at": created_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
            "discovery": discovery,
            "official_validation_accessed": True,
            "official": official,
        },
    )
    return output_dir


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare saved EMA and Patience residual members behind a fold gate"
    )
    parser.add_argument("--parent-campaign", required=True, type=Path)
    parser.add_argument("--residual-campaign", required=True, type=Path)
    parser.add_argument("--parent-reproduction", required=True, type=Path)
    parser.add_argument("--official-campaign", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(arguments)


def main() -> None:
    args = parse_args()
    print(
        run_ema_residual_stack_reanalysis(
            args.parent_campaign,
            args.residual_campaign,
            args.parent_reproduction,
            args.official_campaign,
            args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
