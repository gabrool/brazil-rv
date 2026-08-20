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
from .data import (
    auxiliary_target_identity,
    feature_store_identity,
    resolve_feature_store,
)
from .engine import EvaluationObservations, objective_metadata
from .metrics import primary_validation_score
from .model import PHASE_B_AUXILIARY_VARIANTS
from .provenance import repository_commit
from .train import run_training
from .trajectory import predictions_for_rule, simulate_patience3

DISCOVERY_FOLDS = ("fold_a", "fold_b")
PHASE_B_READOUTS = ("patience3_raw", "final_ema_0995")
PHASE_A_DIVERSITY_VARIANT = "multi_depth_stats"
MAX_PARALLEL_PROCESSES = 2


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _run_manifest(run_dir: Path) -> dict[str, object]:
    return json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))


def _parent_run(parent_campaign: Path, fold: str, seed: int) -> Path:
    return parent_campaign / fold / f"seed_{seed}"


def _phase_a_run(phase_a_campaign: Path, fold: str, seed: int) -> Path:
    return phase_a_campaign / "runs" / PHASE_A_DIVERSITY_VARIANT / fold / f"seed_{seed}"


def _candidate_run(output_dir: Path, variant: str, fold: str, seed: int) -> Path:
    return output_dir / "runs" / variant / fold / f"seed_{seed}"


def _run_candidate_job(
    store: Path,
    auxiliary_target_dir: Path,
    output_dir: Path,
    sidecar_identity: dict[str, object],
    variant: str,
    fold: str,
    seed: int,
) -> str:
    run_dir = _candidate_run(output_dir, variant, fold, seed)
    run_training(
        store=store,
        seed=seed,
        selection_window=fold,
        run_dir=run_dir,
        auxiliary_variant=variant,
        auxiliary_target_dir=auxiliary_target_dir,
        auxiliary_identity_value=sidecar_identity,
    )
    return str(run_dir)


def _validate_source_campaigns(
    parent_campaign: Path,
    phase_a_campaign: Path,
    *,
    store: Path,
    identity: Mapping[str, object],
) -> None:
    parent = json.loads(
        (parent_campaign / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    phase_a = json.loads(
        (phase_a_campaign / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    for name, manifest in (("parent", parent), ("Phase A", phase_a)):
        if (
            manifest.get("status") != "completed"
            or Path(str(manifest.get("feature_store"))).resolve() != store.resolve()
            or manifest.get("feature_store_identity") != identity
            or manifest.get("official_validation_accessed") is not False
            or manifest.get("test_accessed") is not False
        ):
            raise ValueError(f"{name} campaign does not match Phase B")
    for fold in DISCOVERY_FOLDS:
        for seed in ALLOWED_SEEDS:
            for name, run_dir in (
                ("parent", _parent_run(parent_campaign, fold, seed)),
                ("Phase A", _phase_a_run(phase_a_campaign, fold, seed)),
            ):
                manifest = _run_manifest(run_dir)
                if (
                    manifest.get("status") != "completed"
                    or manifest.get("seed") != seed
                    or manifest.get("split", {}).get("training") != fold
                    or manifest.get("split", {}).get("test_accessed") is not False
                ):
                    raise ValueError(f"{name} source run does not match: {run_dir}")


def _completed_candidate_matches(
    run_dir: Path,
    *,
    store: Path,
    store_identity: Mapping[str, object],
    sidecar_identity: Mapping[str, object],
    commit: str,
    variant: str,
    fold: str,
    seed: int,
) -> bool:
    if not (run_dir / "run_manifest.json").is_file():
        return False
    manifest = _run_manifest(run_dir)
    model = manifest.get("model")
    auxiliary = model.get("auxiliary") if isinstance(model, Mapping) else None
    return bool(
        manifest.get("status") == "completed"
        and manifest.get("repository_commit") == commit
        and Path(str(manifest.get("feature_store"))).resolve() == store.resolve()
        and manifest.get("feature_store_identity") == store_identity
        and manifest.get("auxiliary_target_identity") == sidecar_identity
        and manifest.get("objective") == objective_metadata(variant)
        and manifest.get("seed") == seed
        and manifest.get("split", {}).get("training") == fold
        and manifest.get("split", {}).get("test_accessed") is False
        and isinstance(auxiliary, Mapping)
        and auxiliary.get("name") == variant
    )


def _date_parities(date_idx: np.ndarray) -> dict[str, np.ndarray]:
    dates = np.unique(date_idx)
    if dates.size != 102:
        raise ValueError(f"Phase B expected 102 selection dates, found {dates.size}")
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
    for selection_parity, evaluation_parity in (("odd", "even"), ("even", "odd")):
        parities = _date_parities(reference.date_idx)
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
        raise ValueError(f"Unsupported Phase B readout: {readout}")
    return (
        replace(
            load_run_observations(run_dir, "final_raw"),
            predictions=predictions_for_rule(run_dir, readout),
        ),
        [],
    )


def _named_members(
    runs: list[tuple[str, Path]], readout: str
) -> tuple[dict[str, EvaluationObservations], dict[str, object]]:
    members = {}
    replays = {}
    for name, run_dir in runs:
        observations, replay = _readout_observations(run_dir, readout)
        members[name] = observations
        replays[name] = replay
    return members, replays


def _comparison(
    candidate: dict[str, EvaluationObservations],
    parent: dict[str, EvaluationObservations],
    *,
    output_dir: Path,
    readout: str,
    metadata: dict[str, object],
) -> dict[str, object]:
    compare_observation_ensembles(
        candidate,
        parent,
        candidate_rule=readout,
        parent_rule=readout,
        output_dir=output_dir,
        comparison_metadata=metadata,
    )
    return json.loads((output_dir / "analysis.json").read_text(encoding="utf-8"))


def _analyze_variant(
    *,
    variant: str,
    output_dir: Path,
    parent_campaign: Path,
    phase_a_campaign: Path,
) -> dict[str, object]:
    summary_path = output_dir / "analysis" / variant / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("variant") != variant:
            raise ValueError(f"Existing Phase B analysis differs: {summary_path}")
        return summary

    readout_summaries: dict[str, object] = {}
    for readout in PHASE_B_READOUTS:
        folds = {}
        for fold in DISCOVERY_FOLDS:
            parent_runs = [
                (f"parent_seed_{seed}", _parent_run(parent_campaign, fold, seed))
                for seed in ALLOWED_SEEDS
            ]
            candidate_runs = [
                (
                    f"{variant}_seed_{seed}",
                    _candidate_run(output_dir, variant, fold, seed),
                )
                for seed in ALLOWED_SEEDS
            ]
            phase_a_runs = [
                (
                    f"{PHASE_A_DIVERSITY_VARIANT}_seed_{seed}",
                    _phase_a_run(phase_a_campaign, fold, seed),
                )
                for seed in ALLOWED_SEEDS
            ]
            parent, parent_replays = _named_members(parent_runs, readout)
            candidate, candidate_replays = _named_members(candidate_runs, readout)
            phase_a, phase_a_replays = _named_members(phase_a_runs, readout)
            metadata = {
                "variant": variant,
                "fold": fold,
                "readout": readout,
                "adaptive_checkpoint_crossfit": readout == "patience3_raw",
                "parent_patience_replays": parent_replays,
                "candidate_patience_replays": candidate_replays,
                "phase_a_patience_replays": phase_a_replays,
                "ensemble_weights_learned": False,
                "official_validation_accessed": False,
                "test_accessed": False,
            }
            base = output_dir / "analysis" / variant / fold / readout
            standalone = _comparison(
                candidate,
                parent,
                output_dir=base / "standalone",
                readout=readout,
                metadata={**metadata, "composition": "phase_b_three_vs_parent_three"},
            )
            pooled = _comparison(
                {**parent, **candidate},
                parent,
                output_dir=base / "parent_plus_phase_b",
                readout=readout,
                metadata={**metadata, "composition": "parent_three_plus_phase_b_three"},
            )
            stacked = _comparison(
                {**parent, **phase_a, **candidate},
                parent,
                output_dir=base / "parent_plus_phase_a_plus_phase_b",
                readout=readout,
                metadata={
                    **metadata,
                    "composition": (
                        "parent_three_plus_multi_depth_three_plus_phase_b_three"
                    ),
                },
            )
            folds[fold] = {
                "standalone": standalone,
                "parent_plus_phase_b": pooled,
                "parent_plus_phase_a_plus_phase_b": stacked,
            }
        readout_summaries[readout] = {
            "folds": folds,
            "standalone_mean_delta": float(
                np.mean(
                    [
                        folds[fold]["standalone"]["candidate_minus_parent_primary_ic"]
                        for fold in DISCOVERY_FOLDS
                    ]
                )
            ),
            "parent_plus_phase_b_mean_delta": float(
                np.mean(
                    [
                        folds[fold]["parent_plus_phase_b"][
                            "candidate_minus_parent_primary_ic"
                        ]
                        for fold in DISCOVERY_FOLDS
                    ]
                )
            ),
            "stacked_mean_delta": float(
                np.mean(
                    [
                        folds[fold]["parent_plus_phase_a_plus_phase_b"][
                            "candidate_minus_parent_primary_ic"
                        ]
                        for fold in DISCOVERY_FOLDS
                    ]
                )
            ),
        }
    summary = {
        "schema": "PHASE_B_VARIANT_ANALYSIS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "variant": variant,
        "seeds": list(ALLOWED_SEEDS),
        "readouts": readout_summaries,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(summary_path, summary)
    return summary


def _analyze_phase_a_baseline(
    output_dir: Path, parent_campaign: Path, phase_a_campaign: Path
) -> dict[str, object]:
    path = output_dir / "analysis" / "phase_a_baseline.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    readouts = {}
    for readout in PHASE_B_READOUTS:
        folds = {}
        for fold in DISCOVERY_FOLDS:
            parent, parent_replays = _named_members(
                [
                    (f"parent_seed_{seed}", _parent_run(parent_campaign, fold, seed))
                    for seed in ALLOWED_SEEDS
                ],
                readout,
            )
            phase_a, phase_a_replays = _named_members(
                [
                    (
                        f"{PHASE_A_DIVERSITY_VARIANT}_seed_{seed}",
                        _phase_a_run(phase_a_campaign, fold, seed),
                    )
                    for seed in ALLOWED_SEEDS
                ],
                readout,
            )
            report = _comparison(
                {**parent, **phase_a},
                parent,
                output_dir=output_dir
                / "analysis"
                / "phase_a_baseline"
                / fold
                / readout,
                readout=readout,
                metadata={
                    "composition": "parent_three_plus_multi_depth_three",
                    "fold": fold,
                    "readout": readout,
                    "parent_patience_replays": parent_replays,
                    "phase_a_patience_replays": phase_a_replays,
                    "ensemble_weights_learned": False,
                    "official_validation_accessed": False,
                    "test_accessed": False,
                },
            )
            folds[fold] = report
        readouts[readout] = {
            "folds": folds,
            "mean_delta": float(
                np.mean(
                    [
                        folds[fold]["candidate_minus_parent_primary_ic"]
                        for fold in DISCOVERY_FOLDS
                    ]
                )
            ),
        }
    value = {
        "schema": "PHASE_A_DIVERSITY_BASELINE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "readouts": readouts,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(path, value)
    return value


def _select_discovery_winners(
    analyses: Mapping[str, Mapping[str, object]],
    phase_a_baseline: Mapping[str, object],
    output_dir: Path,
) -> Path:
    raw = {
        variant: summary["readouts"]["patience3_raw"]
        for variant, summary in analyses.items()
    }
    standalone_winner = min(
        raw,
        key=lambda variant: (-raw[variant]["standalone_mean_delta"], variant),
    )
    standalone_delta = float(raw[standalone_winner]["standalone_mean_delta"])
    standalone_folds = raw[standalone_winner]["folds"]
    standalone_positive_both = all(
        standalone_folds[fold]["standalone"]["candidate_minus_parent_primary_ic"] > 0.0
        for fold in DISCOVERY_FOLDS
    )
    recency_source = (
        standalone_winner
        if standalone_delta > 0.0 and standalone_positive_both
        else "parent"
    )

    phase_a_delta = float(phase_a_baseline["readouts"]["patience3_raw"]["mean_delta"])
    stack_winner = min(
        raw,
        key=lambda variant: (-raw[variant]["stacked_mean_delta"], variant),
    )
    stack_delta = float(raw[stack_winner]["stacked_mean_delta"])
    stack_folds = raw[stack_winner]["folds"]
    phase_a_folds = phase_a_baseline["readouts"]["patience3_raw"]["folds"]
    stack_adds_to_phase_a_both = all(
        stack_folds[fold]["parent_plus_phase_a_plus_phase_b"][
            "candidate_minus_parent_primary_ic"
        ]
        > phase_a_folds[fold]["candidate_minus_parent_primary_ic"]
        for fold in DISCOVERY_FOLDS
    )
    final_phase_b_member = (
        stack_winner
        if stack_delta > phase_a_delta and stack_adds_to_phase_a_both
        else None
    )
    path = output_dir / "discovery_selection.json"
    _atomic_json(
        path,
        {
            "schema": "PHASE_B_DISCOVERY_SELECTION",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "primary_readout": "cross_fitted_patience3_raw",
            "secondary_readout": "final_ema_0995",
            "recency_full_history_source": recency_source,
            "recency_source_mean_delta_vs_parent": (
                standalone_delta if recency_source != "parent" else 0.0
            ),
            "phase_a_baseline_mean_delta_vs_parent": phase_a_delta,
            "recency_source_positive_on_both_folds": standalone_positive_both,
            "phase_b_stack_candidate": stack_winner,
            "phase_b_stack_mean_delta_vs_parent": stack_delta,
            "phase_b_stack_adds_to_phase_a_on_both_folds": stack_adds_to_phase_a_both,
            "final_diversity_phase_b_member": final_phase_b_member,
            "ensemble_weights_learned": False,
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    return path


def run_phase_b_campaign(
    store: Path,
    auxiliary_target_dir: Path,
    parent_campaign: Path,
    phase_a_campaign: Path,
    output_dir: Path,
    *,
    parallel_processes: int = 1,
) -> Path:
    commit = repository_commit()
    if not 1 <= parallel_processes <= MAX_PARALLEL_PROCESSES:
        raise ValueError("Phase B supports one or two isolated trajectory processes")
    store_identity = feature_store_identity(store)
    sidecar_identity = auxiliary_target_identity(auxiliary_target_dir, store_identity)
    _validate_source_campaigns(
        parent_campaign,
        phase_a_campaign,
        store=store,
        identity=store_identity,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "campaign_manifest.json"
    immutable = {
        "schema": "PHASE_B_CAMPAIGN",
        "repository_commit": commit,
        "feature_store": str(store.resolve()),
        "feature_store_identity": store_identity,
        "auxiliary_target_identity": sidecar_identity,
        "trajectory_parent": str(parent_campaign.resolve()),
        "phase_a_diversity_source": str(phase_a_campaign.resolve()),
        "variants": list(PHASE_B_AUXILIARY_VARIANTS),
        "folds": list(DISCOVERY_FOLDS),
        "seeds": list(ALLOWED_SEEDS),
        "readouts": list(PHASE_B_READOUTS),
        "ensemble_weights_learned": False,
        "official_validation_accessed": False,
        "parallel_processes": parallel_processes,
        "test_accessed": False,
    }
    created_at = datetime.now(timezone.utc).isoformat()
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(existing.get(key) != value for key, value in immutable.items()):
            raise ValueError("Existing Phase B campaign has a different contract")
        created_at = str(existing["created_at"])
    _atomic_json(
        manifest_path,
        {**immutable, "status": "running", "created_at": created_at},
    )

    pending = []
    for variant in PHASE_B_AUXILIARY_VARIANTS:
        for fold in DISCOVERY_FOLDS:
            for seed in ALLOWED_SEEDS:
                run_dir = _candidate_run(output_dir, variant, fold, seed)
                if run_dir.exists():
                    if not _completed_candidate_matches(
                        run_dir,
                        store=store,
                        store_identity=store_identity,
                        sidecar_identity=sidecar_identity,
                        commit=commit,
                        variant=variant,
                        fold=fold,
                        seed=seed,
                    ):
                        raise ValueError(f"Existing Phase B run differs: {run_dir}")
                else:
                    pending.append(
                        (
                            store,
                            auxiliary_target_dir,
                            output_dir,
                            dict(sidecar_identity),
                            variant,
                            fold,
                            seed,
                        )
                    )
    if parallel_processes == 1:
        for job in pending:
            print(_run_candidate_job(*job), flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=parallel_processes,
            mp_context=mp.get_context("spawn"),
        ) as executor:
            futures = [executor.submit(_run_candidate_job, *job) for job in pending]
            for future in as_completed(futures):
                print(future.result(), flush=True)

    phase_a_baseline = _analyze_phase_a_baseline(
        output_dir, parent_campaign, phase_a_campaign
    )
    analyses = {
        variant: _analyze_variant(
            variant=variant,
            output_dir=output_dir,
            parent_campaign=parent_campaign,
            phase_a_campaign=phase_a_campaign,
        )
        for variant in PHASE_B_AUXILIARY_VARIANTS
    }
    selection = _select_discovery_winners(analyses, phase_a_baseline, output_dir)
    _atomic_json(
        manifest_path,
        {
            **immutable,
            "status": "completed",
            "created_at": created_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "discovery_selection": str(selection.resolve()),
        },
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the two-fold, three-seed Phase B auxiliary-target campaign"
    )
    parser.add_argument("--auxiliary-target-dir", required=True, type=Path)
    parser.add_argument("--parent-campaign", required=True, type=Path)
    parser.add_argument("--phase-a-campaign", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--parallel-processes", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        run_phase_b_campaign(
            resolve_feature_store(),
            args.auxiliary_target_dir,
            args.parent_campaign,
            args.phase_a_campaign,
            args.output_dir,
            parallel_processes=args.parallel_processes,
        )
    )


if __name__ == "__main__":
    main()
