from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from itertools import combinations
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .analyze import compare_observation_ensembles
from .contract import RUN_OUTPUT_BASE
from .data import resolve_feature_store
from .designated_challenger import load_official_challenger_members
from .engine import EvaluationObservations, assert_observations_aligned
from .feature_removal import _definition
from .metrics import primary_validation_score, rank_average_predictions
from .provenance import repository_commit
from .train import run_training
from .trajectory import predictions_for_rule

BASE_SEEDS = (11, 29, 47)
EXPANSION_SEEDS = (61, 79, 97, 113, 131, 149, 167)
ALL_SEEDS = (*BASE_SEEDS, *EXPANSION_SEEDS)
PATIENCE_RULE = "patience3_raw"
RESIDUAL_RULE = "final_ema_0995"
VALIDATION_DATE_COUNT = 244
DEPLOYMENT_TOLERANCE = 0.0005
HISTORICAL_RESIDUAL_COMMIT = "921dd3a3494e7855d97afbdfc4d10b414efafa59"
EXPECTED_DYNAMIC_ZEROING = (9, 11, 14, 22, 24, 25)
EXPECTED_SLOW_ZEROING = (1, 2, 3, 12, 13, 14, 15, 16, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29)
EXPECTED_REMOVED_FIELDS = (
    "dynamic_9",
    "dynamic_11",
    "dynamic_14",
    "dynamic_22",
    "dynamic_24",
    "dynamic_25",
    "slow_1",
    "slow_2",
    "slow_3",
    "slow_12",
    "slow_13",
    "slow_14",
    "slow_15",
    "slow_16",
    "slow_18",
    "slow_20",
    "slow_22",
    "slow_23",
    "slow_24",
    "slow_25",
    "slow_26",
    "slow_27",
    "slow_28",
    "slow_29",
)

PARENT_ARTIFACT = "parent_reproduction_4067962_e22dd67_20260819T131142Z"
RESIDUAL_ARTIFACT = "next_stage_official_921dd3a_20260821T085500Z"
FEATURE_REMOVAL_ARTIFACT = "feature_removal_stage_c_repair_d5b5e1f_20260823T232938Z"
SELECTION_ARTIFACT = "trajectory_crossfit_3054228_20260819T161200Z"
STALENESS_ARTIFACT = "next_stage_c0d0598_20260820T225000Z"
RESIDUAL_SIDECAR = "next_stage_3b60ac9_20260820T233000Z"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_legacy_observations(path: Path) -> EvaluationObservations:
    with np.load(path, allow_pickle=False) as values:
        expected = set(EvaluationObservations.__dataclass_fields__)
        if set(values.files) != expected:
            raise ValueError(f"Unexpected parent observation fields: {path}")
        return EvaluationObservations(
            **{name: values[name].copy() for name in expected}
        )


def _parent_members(parent_root: Path) -> dict[str, EvaluationObservations]:
    members: dict[str, EvaluationObservations] = {}
    for path in parent_root.rglob("validation_observations.npz"):
        manifest_path = path.parent / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _read_json(manifest_path)
        seed = int(manifest.get("seed", -1))
        split = manifest.get("split", {})
        if seed not in BASE_SEEDS:
            continue
        if (
            manifest.get("status") != "completed"
            or split.get("training") != "train"
            or split.get("selection") != "validation"
            or split.get("test_accessed") is not False
        ):
            raise ValueError(f"Parent run differs from the canonical comparator: {path.parent}")
        key = f"seed_{seed}"
        if key in members:
            raise ValueError(f"Duplicate canonical parent member: {key}")
        members[key] = _load_legacy_observations(path)
    expected = {f"seed_{seed}" for seed in BASE_SEEDS}
    if set(members) != expected:
        raise ValueError(f"Canonical parent members differ: {sorted(members)}")
    reference = members["seed_11"]
    for member in members.values():
        assert_observations_aligned(reference, member)
    if np.unique(reference.date_idx).size != VALIDATION_DATE_COUNT:
        raise ValueError("Canonical parent observations are not official validation")
    return members


def _run_path(output_dir: Path, recipe: str, seed: int) -> Path:
    return output_dir / "runs" / recipe / f"seed_{seed}"


def _completed_current_run(
    run_dir: Path,
    *,
    seed: int,
    dynamic: tuple[int, ...],
    slow: tuple[int, ...],
    commit: str,
) -> bool:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = _read_json(manifest_path)
    zeroing = manifest.get("equity_input_zeroing", {})
    return bool(
        manifest.get("status") == "completed"
        and manifest.get("repository_commit") == commit
        and int(manifest.get("seed", -1)) == seed
        and manifest.get("split", {}).get("training") == "official"
        and manifest.get("split", {}).get("test_accessed") is False
        and zeroing.get("dynamic_channels") == list(dynamic)
        and zeroing.get("slow_fields") == list(slow)
        and manifest.get("frozen_selection", {}).get("selected_rule")
        == PATIENCE_RULE
    )


def _train_current_job(
    store: Path,
    run_dir: Path,
    seed: int,
    dynamic: tuple[int, ...],
    slow: tuple[int, ...],
    selection_file: Path,
    commit: str,
) -> str:
    if run_dir.exists():
        if not _completed_current_run(
            run_dir,
            seed=seed,
            dynamic=dynamic,
            slow=slow,
            commit=commit,
        ):
            raise ValueError(f"Existing official run differs: {run_dir}")
        return str(run_dir)
    run_training(
        store=store,
        seed=seed,
        selection_window="official",
        selection_rule_file=selection_file,
        run_dir=run_dir,
        zero_dynamic_channels=dynamic,
        zero_slow_fields=slow,
    )
    return str(run_dir)


def _run_current_jobs(
    *,
    store: Path,
    output_dir: Path,
    recipe: str,
    seeds: Sequence[int],
    dynamic: tuple[int, ...],
    slow: tuple[int, ...],
    selection_file: Path,
    commit: str,
) -> None:
    jobs = [
        (
            store,
            _run_path(output_dir, recipe, seed),
            seed,
            dynamic,
            slow,
            selection_file,
            commit,
        )
        for seed in seeds
    ]
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(_train_current_job, *job) for job in jobs]
        for future in as_completed(futures):
            print(future.result(), flush=True)


def _current_members(
    output_dir: Path, recipe: str, seeds: Sequence[int]
) -> dict[str, EvaluationObservations]:
    return {
        f"seed_{seed}": replace(
            _load_reference(_run_path(output_dir, recipe, seed)),
            predictions=predictions_for_rule(
                _run_path(output_dir, recipe, seed), PATIENCE_RULE
            ),
        )
        for seed in seeds
    }


def _load_reference(run_dir: Path) -> EvaluationObservations:
    with np.load(run_dir / "validation_reference.npz", allow_pickle=False) as values:
        fields = {
            name: values[name].copy()
            for name in EvaluationObservations.__dataclass_fields__
            if name != "predictions"
        }
    return EvaluationObservations(predictions=np.empty((0,), dtype=np.float32), **fields)


def _analysis(
    *,
    candidate: Mapping[str, EvaluationObservations],
    parent: Mapping[str, EvaluationObservations],
    candidate_rule: str,
    parent_rule: str,
    output_dir: Path,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    if not output_dir.exists():
        compare_observation_ensembles(
            candidate,
            parent,
            candidate_rule=candidate_rule,
            parent_rule=parent_rule,
            output_dir=output_dir,
            comparison_metadata={
                **metadata,
                "official_validation_accessed": True,
                "test_accessed": False,
            },
        )
    return _read_json(output_dir / "analysis.json")


def arm_supported(report: Mapping[str, object]) -> bool:
    interval = report["per_date_delta_bootstrap"]["10"]
    return float(interval["lower_95"][0]) > 0.0


def promotion_decision(
    challenger: Mapping[str, object], store_v2: Mapping[str, object]
) -> dict[str, object]:
    reports = {"challenger": challenger, "store_v2": store_v2}
    supported = [name for name, report in reports.items() if arm_supported(report)]
    if len(supported) == 2:
        promoted = max(
            supported,
            key=lambda name: float(reports[name]["candidate"]["ensemble_ic"]),
        )
    elif len(supported) == 1:
        promoted = supported[0]
    else:
        promoted = None
    return {
        "supported": supported,
        "promoted": promoted,
        "canonical_remains_deployed": promoted is None,
        "rule": (
            "An arm is supported iff its paired block-10 95% interval versus "
            "canonical excludes zero from above. If both are supported, promote "
            "the higher-official-IC arm; if one is supported, promote it; if "
            "neither is supported, canonical remains deployed."
        ),
    }


def deploy_expansion(expanded_ic: float, promoted_three_seed_ic: float) -> bool:
    return expanded_ic >= promoted_three_seed_ic - DEPLOYMENT_TOLERANCE


def _seed_diagnostics(
    members_by_seed: Mapping[int, Sequence[EvaluationObservations]],
) -> dict[str, object]:
    if tuple(members_by_seed) != ALL_SEEDS:
        raise ValueError("Seed diagnostics require the frozen 3-to-10 seed order")
    flat = {
        f"seed_{seed}_member_{member_index}": member
        for seed, members in members_by_seed.items()
        for member_index, member in enumerate(members)
    }
    reference = next(iter(flat.values()))
    for member in flat.values():
        assert_observations_aligned(reference, member)
    seed_predictions = {
        seed: rank_average_predictions(
            [member.predictions for member in members], reference.label_mask
        )
        for seed, members in members_by_seed.items()
    }
    correlations = []
    matrix = np.eye(len(ALL_SEEDS), dtype=np.float64)
    for left, right in combinations(ALL_SEEDS, 2):
        value = primary_validation_score(
            seed_predictions[left],
            seed_predictions[right],
            reference.label_mask,
            reference.date_idx,
        )
        left_index = ALL_SEEDS.index(left)
        right_index = ALL_SEEDS.index(right)
        matrix[left_index, right_index] = value
        matrix[right_index, left_index] = value
        correlations.append(
            {
                "left_seed": left,
                "right_seed": right,
                "prediction_spearman": value,
            }
        )
    curve = []
    for count in range(len(BASE_SEEDS), len(ALL_SEEDS) + 1):
        seeds = ALL_SEEDS[:count]
        predictions = rank_average_predictions(
            [
                member.predictions
                for seed in seeds
                for member in members_by_seed[seed]
            ],
            reference.label_mask,
        )
        curve.append(
            {
                "seed_count": count,
                "seeds": list(seeds),
                "member_count": sum(len(members_by_seed[seed]) for seed in seeds),
                "ensemble_ic": primary_validation_score(
                    predictions,
                    reference.targets,
                    reference.label_mask,
                    reference.date_idx,
                ),
            }
        )
    return {
        "seed_correlation_matrix": {
            "seed_order": list(ALL_SEEDS),
            "values": matrix.tolist(),
        },
        "seed_correlation_pairs": correlations,
        "ensemble_gain_curve": curve,
    }


def _ensure_historical_worktree(output_dir: Path, repository: Path) -> Path:
    worktree = output_dir / "source" / f"residual_{HISTORICAL_RESIDUAL_COMMIT[:7]}"
    if worktree.exists():
        actual = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=worktree,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if actual != HISTORICAL_RESIDUAL_COMMIT:
            raise ValueError("Historical residual worktree commit differs")
        return worktree
    worktree.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        (
            "git",
            "worktree",
            "add",
            "--detach",
            str(worktree),
            HISTORICAL_RESIDUAL_COMMIT,
        ),
        cwd=repository,
        check=True,
    )
    return worktree


def _residual_job(
    worker: Path,
    worktree: Path,
    store: Path,
    sidecar: Path,
    run_dir: Path,
    seed: int,
) -> str:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(worktree / "research" / "src")
    subprocess.run(
        (
            sys.executable,
            str(worker),
            "--store",
            str(store),
            "--sidecar",
            str(sidecar),
            "--run-dir",
            str(run_dir),
            "--seed",
            str(seed),
        ),
        cwd=worktree,
        env=environment,
        check=True,
    )
    return str(run_dir)


def _run_residual_jobs(
    *,
    worker: Path,
    worktree: Path,
    store: Path,
    sidecar: Path,
    output_dir: Path,
) -> None:
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        futures = [
            pool.submit(
                _residual_job,
                worker,
                worktree,
                store,
                sidecar,
                _run_path(output_dir, "residual_fixed_final_ema", seed),
                seed,
            )
            for seed in EXPANSION_SEEDS
        ]
        for future in as_completed(futures):
            print(future.result(), flush=True)


def _fixed_residual_members(
    output_dir: Path,
    parent_reference: EvaluationObservations,
    worker_sha256: str,
) -> dict[str, EvaluationObservations]:
    members = {}
    for seed in EXPANSION_SEEDS:
        run = _run_path(output_dir, "residual_fixed_final_ema", seed)
        manifest = _read_json(run / "run_manifest.json")
        if (
            manifest.get("status") != "completed"
            or manifest.get("seed") != seed
            or manifest.get("repository_commit") != HISTORICAL_RESIDUAL_COMMIT
            or manifest.get("fixed_fit_worker_sha256") != worker_sha256
            or manifest.get("model", {}).get("variant", {}).get("name")
            != "residual_auxiliary"
            or manifest.get("training", {}).get("official_monitoring") is not False
            or manifest.get("split", {}).get("test_accessed") is not False
        ):
            raise ValueError(f"Supplementary residual run differs: {run}")
        with np.load(run / "validation_reference.npz", allow_pickle=False) as values:
            fields = {
                name: values[name].copy()
                for name in EvaluationObservations.__dataclass_fields__
                if name != "predictions"
            }
        with np.load(
            run / "validation_predictions" / "epoch_20.npz", allow_pickle=False
        ) as values:
            observations = EvaluationObservations(
                predictions=values["ema_0995"].copy(), **fields
            )
        assert_observations_aligned(parent_reference, observations)
        if np.unique(observations.date_idx).size != VALIDATION_DATE_COUNT:
            raise ValueError("Supplementary residual observations are not official")
        members[f"seed_{seed}"] = observations
    return members


def _artifact_inventory(paths: Sequence[Path]) -> list[dict[str, object]]:
    inventory = []
    for path in paths:
        inventory.append(
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return inventory


def run_official_read(*, output_dir: Path, preregistration: Path) -> Path:
    commit = repository_commit()
    repository = Path(__file__).resolve().parents[4]
    store = resolve_feature_store()
    parent_root = RUN_OUTPUT_BASE / PARENT_ARTIFACT
    residual_root = RUN_OUTPUT_BASE / RESIDUAL_ARTIFACT
    feature_root = RUN_OUTPUT_BASE / FEATURE_REMOVAL_ARTIFACT
    feature_specification = (
        feature_root / "stage_c" / "store_v2_feature_specification.json"
    )
    selection_file = RUN_OUTPUT_BASE / SELECTION_ARTIFACT / "trajectory_selection.json"
    staleness_report = RUN_OUTPUT_BASE / STALENESS_ARTIFACT / "d1_staleness" / "staleness_report.json"
    sidecar = RUN_OUTPUT_BASE.parent / "auxiliary_targets" / RESIDUAL_SIDECAR
    required = (
        preregistration,
        parent_root,
        residual_root,
        feature_root,
        selection_file,
        staleness_report,
        sidecar,
    )
    if any(not path.exists() for path in required):
        missing = [str(path) for path in required if not path.exists()]
        raise FileNotFoundError(f"Official-read preflight inputs missing: {missing}")
    spec = _read_json(feature_specification)
    removed = tuple(sorted(str(value) for value in spec["removed_fields"]))
    if removed != tuple(sorted(EXPECTED_REMOVED_FIELDS)):
        raise ValueError("Experiment-41 store-v2 removal fields differ")
    dynamic, slow = _definition(removed)
    if dynamic != EXPECTED_DYNAMIC_ZEROING or slow != EXPECTED_SLOW_ZEROING:
        raise ValueError("Experiment-41 store-v2 removal indices differ")
    selection = _read_json(selection_file)
    if selection.get("selected_rule") != PATIENCE_RULE:
        raise ValueError("Frozen official selection rule is not Raw Patience-3")
    output_dir.mkdir(parents=True, exist_ok=True)
    prereg_copy = output_dir / "preregistration.md"
    if prereg_copy.exists():
        if prereg_copy.read_bytes() != preregistration.read_bytes():
            raise ValueError("Existing official-read preregistration differs")
    else:
        prereg_copy.write_bytes(preregistration.read_bytes())
    manifest_path = output_dir / "official_read_manifest.json"
    existing_manifest = _read_json(manifest_path) if manifest_path.exists() else None
    if existing_manifest is not None and (
        existing_manifest.get("repository_commit") != commit
        or existing_manifest.get("preregistration_sha256") != _sha256(prereg_copy)
        or existing_manifest.get("test_accessed") is not False
    ):
        raise ValueError("Existing official-read root has a different contract")
    manifest = {
        "schema": "OFFICIAL_READ_EXPERIMENT_43_V1",
        "status": "running",
        "created_at": (
            str(existing_manifest["created_at"])
            if existing_manifest is not None
            else _now()
        ),
        "repository_commit": commit,
        "preregistration": str(prereg_copy.resolve()),
        "preregistration_sha256": _sha256(prereg_copy),
        "arms": ["canonical", "challenger", "store_v2"],
        "base_seeds": list(BASE_SEEDS),
        "expansion_seeds": list(EXPANSION_SEEDS),
        "official_validation_accessed": True,
        "test_accessed": False,
    }
    _atomic_json(manifest_path, manifest)
    parent_manifests = tuple(parent_root.rglob("run_manifest.json"))
    parent_predictions = tuple(parent_root.rglob("validation_observations.npz"))
    residual_runs = residual_root / "official_runs" / "residual_auxiliary"
    residual_manifests = tuple(residual_runs.rglob("run_manifest.json"))
    residual_predictions = tuple(residual_runs.rglob("epoch_20.npz"))
    if (
        len(parent_manifests),
        len(parent_predictions),
        len(residual_manifests),
        len(residual_predictions),
    ) != (3, 3, 3, 3):
        raise ValueError("Reused official source inventory is not exactly 3+3 members")
    reused_source_paths = sorted(
        (
            *parent_manifests,
            *parent_predictions,
            *residual_manifests,
            *residual_predictions,
            feature_specification,
            selection_file,
            staleness_report,
            sidecar / "manifest.json",
        ),
        key=str,
    )
    if any(not path.is_file() for path in reused_source_paths):
        missing = [str(path) for path in reused_source_paths if not path.is_file()]
        raise FileNotFoundError(f"Official-read source inventory incomplete: {missing}")
    source_inventory_path = output_dir / "reused_source_inventory.json"
    if not source_inventory_path.exists():
        _atomic_json(
            source_inventory_path,
            {
                "schema": "OFFICIAL_READ_REUSED_SOURCE_INVENTORY_V1",
                "created_at": _now(),
                "files": _artifact_inventory(reused_source_paths),
                "official_validation_accessed": True,
                "test_accessed": False,
            },
        )

    _run_current_jobs(
        store=store,
        output_dir=output_dir,
        recipe="store_v2",
        seeds=BASE_SEEDS,
        dynamic=dynamic,
        slow=slow,
        selection_file=selection_file,
        commit=commit,
    )
    canonical = _parent_members(parent_root)
    challenger = load_official_challenger_members(canonical, run_root=RUN_OUTPUT_BASE)
    store_v2 = _current_members(output_dir, "store_v2", BASE_SEEDS)
    arm1 = _analysis(
        candidate=challenger,
        parent=canonical,
        candidate_rule="parent3_patience_plus_residual3_final_ema_0995",
        parent_rule="canonical_parent3_patience3_raw",
        output_dir=output_dir / "analysis" / "challenger_vs_canonical",
        metadata={"arm": "challenger", "member_count": 6},
    )
    arm2 = _analysis(
        candidate=store_v2,
        parent=canonical,
        candidate_rule="store_v2_parent3_patience3_raw",
        parent_rule="canonical_parent3_patience3_raw",
        output_dir=output_dir / "analysis" / "store_v2_vs_canonical",
        metadata={"arm": "store_v2", "member_count": 3},
    )
    decision = promotion_decision(arm1, arm2)
    decision.update(
        {
            "created_at": _now(),
            "challenger_analysis": str(
                (output_dir / "analysis" / "challenger_vs_canonical" / "analysis.json").resolve()
            ),
            "store_v2_analysis": str(
                (output_dir / "analysis" / "store_v2_vs_canonical" / "analysis.json").resolve()
            ),
            "official_validation_accessed": True,
            "test_accessed": False,
        }
    )
    _atomic_json(output_dir / "promotion_decision.json", decision)

    promoted = decision["promoted"]
    supplementary: dict[str, object] | None = None
    deployed_member_paths: list[str] = []
    if promoted is not None:
        _run_current_jobs(
            store=store,
            output_dir=output_dir,
            recipe="parent" if promoted == "challenger" else "store_v2",
            seeds=EXPANSION_SEEDS,
            dynamic=() if promoted == "challenger" else dynamic,
            slow=() if promoted == "challenger" else slow,
            selection_file=selection_file,
            commit=commit,
        )
        if promoted == "challenger":
            worker = repository / "research" / "src" / "brazil_rv" / "modeling" / "official_residual_worker.py"
            worktree = _ensure_historical_worktree(output_dir, repository)
            _run_residual_jobs(
                worker=worker,
                worktree=worktree,
                store=store,
                sidecar=sidecar,
                output_dir=output_dir,
            )
            new_parent = _current_members(output_dir, "parent", EXPANSION_SEEDS)
            reference = canonical["seed_11"]
            new_residual = _fixed_residual_members(
                output_dir, reference, _sha256(worker)
            )
            expanded = {
                **challenger,
                **{f"parent_seed_{seed}": new_parent[f"seed_{seed}"] for seed in EXPANSION_SEEDS},
                **{f"residual_seed_{seed}": new_residual[f"seed_{seed}"] for seed in EXPANSION_SEEDS},
            }
            three_seed = challenger
            members_by_seed = {
                seed: (
                    expanded[f"parent_seed_{seed}"],
                    expanded[f"residual_seed_{seed}"],
                )
                for seed in ALL_SEEDS
            }
        else:
            new_store = _current_members(output_dir, "store_v2", EXPANSION_SEEDS)
            expanded = {**store_v2, **new_store}
            three_seed = store_v2
            members_by_seed = {seed: (expanded[f"seed_{seed}"],) for seed in ALL_SEEDS}
        supplement_report = _analysis(
            candidate=expanded,
            parent=three_seed,
            candidate_rule=f"{promoted}_10_seed_uniform_rank",
            parent_rule=f"{promoted}_3_seed_uniform_rank",
            output_dir=output_dir / "analysis" / "supplementary_10_vs_3",
            metadata={"promoted_arm": promoted, "measurement": "seed_expansion"},
        )
        diagnostics = _seed_diagnostics(members_by_seed)
        expanded_ic = float(supplement_report["candidate"]["ensemble_ic"])
        three_seed_ic = float(supplement_report["parent"]["ensemble_ic"])
        deploy_ten = deploy_expansion(expanded_ic, three_seed_ic)
        supplementary = {
            "analysis": str(
                (output_dir / "analysis" / "supplementary_10_vs_3" / "analysis.json").resolve()
            ),
            "expanded_ic": expanded_ic,
            "three_seed_ic": three_seed_ic,
            "deployment_tolerance": DEPLOYMENT_TOLERANCE,
            "deploy_10_seed": deploy_ten,
            **diagnostics,
        }
        _atomic_json(output_dir / "supplementary_measurement.json", supplementary)
        deployed_seeds = ALL_SEEDS if deploy_ten else BASE_SEEDS
        recipe = "parent" if promoted == "challenger" else "store_v2"
        deployed_member_paths = [
            str(_run_path(output_dir, recipe, seed).resolve())
            for seed in deployed_seeds
            if promoted == "store_v2" or seed in EXPANSION_SEEDS
        ]
        if promoted == "challenger" and deploy_ten:
            deployed_member_paths.extend(
                str(_run_path(output_dir, "residual_fixed_final_ema", seed).resolve())
                for seed in EXPANSION_SEEDS
            )
    else:
        deployed_seeds = BASE_SEEDS
        deploy_ten = False
    if promoted == "challenger":
        deployed_member_paths = [
            str(parent_root.resolve()),
            str(residual_root.resolve()),
            *deployed_member_paths,
        ]
    elif promoted is None:
        deployed_member_paths = [str(parent_root.resolve())]

    deployed = {
        "schema": "OFFICIAL_DEPLOYED_RECIPE_V1",
        "created_at": _now(),
        "recipe": "canonical" if promoted is None else str(promoted),
        "seed_count": len(deployed_seeds),
        "seeds": list(deployed_seeds),
        "parent_state": PATIENCE_RULE,
        "residual_state": RESIDUAL_RULE if promoted == "challenger" else None,
        "equity_input_zeroing": (
            {"dynamic_channels": list(dynamic), "slow_fields": list(slow)}
            if promoted == "store_v2"
            else {"dynamic_channels": [], "slow_fields": []}
        ),
        "ensemble": "uniform within-sample/horizon tie-aware rank average",
        "learned_weights": False,
        "supplementary_10_seed_deployed": deploy_ten,
        "new_member_run_paths": deployed_member_paths,
        "official_validation_accessed": True,
        "test_accessed": False,
    }
    _atomic_json(output_dir / "deployed_recipe.json", deployed)
    ledger = {
        "schema": "VALIDATION_ACCESS_LEDGER_EVENT_V1",
        "event": 3,
        "experiment": 43,
        "arms": ["canonical_reused", "challenger_reused", "store_v2_trained"],
        "official_monitor_runs": [
            str(_run_path(output_dir, "store_v2", seed).resolve())
            for seed in BASE_SEEDS
        ]
        + [
            str(_run_path(output_dir, "parent" if promoted == "challenger" else "store_v2", seed).resolve())
            for seed in EXPANSION_SEEDS
            if promoted is not None
        ],
        "fixed_no_monitor_runs": [
            str(_run_path(output_dir, "residual_fixed_final_ema", seed).resolve())
            for seed in EXPANSION_SEEDS
            if promoted == "challenger"
        ],
        "official_validation_accessed": True,
        "test_accessed": False,
    }
    _atomic_json(output_dir / "validation_access_ledger.json", ledger)
    required_outputs = sorted(
        (
            path
            for path in output_dir.rglob("*")
            if path.is_file()
            and path != manifest_path
            and path.suffix in {".json", ".md", ".npz", ".parquet", ".csv"}
        ),
        key=str,
    )
    final_manifest = {
        **manifest,
        "status": "completed",
        "completed_at": _now(),
        "promotion": promoted,
        "deployed_recipe": deployed,
        "supplementary_performed": supplementary is not None,
        "required_output_inventory": _artifact_inventory(required_outputs),
        "official_validation_accessed": True,
        "test_accessed": False,
    }
    _atomic_json(manifest_path, final_manifest)
    return output_dir


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen Experiment-43 official read")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    return parser.parse_args(arguments)


def main() -> None:
    args = parse_args()
    print(run_official_read(output_dir=args.output_dir, preregistration=args.preregistration))


if __name__ == "__main__":
    main()
