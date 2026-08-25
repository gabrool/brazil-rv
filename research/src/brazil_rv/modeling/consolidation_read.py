from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import date, datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl
import torch

from .analyze import compare_observation_ensembles
from .contract import GH200_RUNTIME, MAX_EPOCHS, RUN_OUTPUT_BASE, VALIDATION_END
from .data import (
    create_training_loaders,
    feature_store_identity,
    load_external_sidecar,
    load_sample_index,
    resolve_feature_store,
    sample_window_metadata,
    select_training_window,
)
from .engine import (
    EvaluationObservations,
    assert_observations_aligned,
    checkpoint_payload,
    collect_validation_observations,
    compile_model,
    compile_training_objective,
    objective_metadata,
    sam_metadata,
    summarize_evaluation_observations,
    train_one_epoch,
    validation_primary_metric,
)
from .ensemble_science import (
    STORE_V2_DYNAMIC,
    STORE_V2_SLOW,
    _bagged_dates,
    _derived_seed,
)
from .metrics import (
    combine_rank_predictions,
    primary_validation_score,
    rank_prediction_similarity,
    rank_transform_predictions,
)
from .model import build_model, count_trainable_parameters
from .official_read import (
    ALL_SEEDS,
    BASE_SEEDS,
    PATIENCE_RULE,
    PARENT_ARTIFACT,
    SELECTION_ARTIFACT,
    _current_members,
    _load_reference,
    _parent_members,
    _seed_diagnostics,
)
from .optim import build_optimizer, build_scheduler
from .provenance import build_run_provenance, repository_commit
from .train import run_training, set_seeds
from .trajectory import ModelEMA, predictions_for_rule, temporarily_load_state

PREREGISTRATION = Path("research/preregistrations/experiment45_consolidation_read.md")
OFFICIAL_READ_ARTIFACT = "official_read_c04ea91_20260824T140900Z"
ENSEMBLE_SCIENCE_ARTIFACT = "ensemble_science_8dff0be_20260824T174700Z"
FEATURE_REMOVAL_ARTIFACT = "feature_removal_stage_c_repair_d5b5e1f_20260823T232938Z"
ARCHIVED_STORE_V2_IC = 0.043235373
SANITY_BAND = 0.0015
DEPLOYMENT_TOLERANCE = 0.0005
MAX_ARM2_NEW_TRAJECTORIES = 20
EXPECTED_EXPERIMENT43_MANIFEST_SHA256 = (
    "6dbe8314262dd61c0dfda835055282644aed83053149be57398b86673da9ff85"
)
EXPECTED_EXPERIMENT43_ANALYSIS_SHA256 = (
    "53f9082775a90ac5d28b5573b827c518b7e4c409aecbde2273e02c18d9343652"
)
EXPECTED_EXPERIMENT44_ANALYSIS_SHA256 = (
    "b933e740034cf6499d4569d98ebdb4242c697ee6664c44cafcbc9c606aa913b7"
)
EXPECTED_EXPERIMENT44_CATALOGUE_SHA256 = (
    "77d2e9567094fb26efb90f9f599c37eaa92cb0a898feb6081bb8d55817e06584"
)
EXPECTED_EXPERIMENT44_DESIGN_SHA256 = (
    "e0e991079e88f146534a24f3fd94de545f8d773de4547fc49d395d984523d8d9"
)
COMPARATOR_IDENTITIES = tuple(
    f"prune_r2|seed_{seed}|patience3_raw" for seed in BASE_SEEDS
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        np.savez(output, **values)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def derive_consensus(analysis: Mapping[str, object]) -> dict[str, object]:
    named = analysis.get("named_read_arm")
    if not isinstance(named, Mapping) or named.get("label") != "e2_plus_archive":
        raise ValueError("Experiment 44 did not freeze the expected read arm")
    paths = named.get("paths")
    if not isinstance(paths, Mapping) or set(paths) != {"fold_c", "fold_a", "fold_b"}:
        raise ValueError("Experiment 44 read arm does not have three frozen paths")
    counts: Counter[str] = Counter()
    gains: defaultdict[str, list[float]] = defaultdict(list)
    for fold in ("fold_c", "fold_a", "fold_b"):
        path = paths[fold]
        if not isinstance(path, Mapping):
            raise ValueError(f"Malformed Experiment 44 path: {fold}")
        heldout_members = [str(value) for value in path["heldout_members"]]
        if len(heldout_members) != len(set(heldout_members)):
            raise ValueError(f"Experiment 44 path repeats a member: {fold}")
        counts.update(heldout_members)
        for step in path["steps"]:
            gains[str(step["addition"])].append(float(step["marginal_ic"]))
    candidates = []
    for identity, total in counts.items():
        if identity in COMPARATOR_IDENTITIES or total < 2:
            continue
        values = gains[identity]
        if not values:
            raise ValueError(
                f"Repeated non-comparator lacks a recorded marginal gain: {identity}"
            )
        candidates.append(
            {
                "identity": identity,
                "total_repeat_count": total,
                "mean_recorded_marginal_gain": float(np.mean(values)),
            }
        )
    candidates.sort(
        key=lambda row: (
            -int(row["total_repeat_count"]),
            -float(row["mean_recorded_marginal_gain"]),
            str(row["identity"]),
        )
    )
    selected = candidates[:16]
    if not selected:
        return {
            "label": "e2_plus_archive",
            "withdrawn": True,
            "reason": "No non-comparator member reached total repeat count >= 2",
            "members": [],
        }
    members = [
        {
            "identity": identity,
            "total_repeat_count": counts[identity],
            "mean_recorded_marginal_gain": 0.0,
            "comparator": True,
        }
        for identity in COMPARATOR_IDENTITIES
    ] + [{**row, "comparator": False} for row in selected]
    for row in members:
        row["raw_weight"] = max(int(row["total_repeat_count"]), 1)
    weight_total = sum(int(row["raw_weight"]) for row in members)
    for row in members:
        row["normalized_weight"] = row["raw_weight"] / weight_total
    return {
        "label": "e2_plus_archive",
        "withdrawn": False,
        "rule": (
            "Include non-comparator identities with total repeat count >= 2; "
            "cap at 16 by total, mean recorded marginal gain, then lexical identity; "
            "always include each comparator with weight max(total, 1)."
        ),
        "non_comparator_candidates_before_cap": len(candidates),
        "members": members,
        "raw_weight_total": weight_total,
    }


def sanity_band_passed(fresh_three_seed_ic: float) -> bool:
    delta = abs(fresh_three_seed_ic - ARCHIVED_STORE_V2_IC)
    return delta <= SANITY_BAND or math.isclose(
        delta, SANITY_BAND, rel_tol=0.0, abs_tol=1e-12
    )


def deploy_arm1_ten_seed(fresh_ten_seed_ic: float, fresh_three_seed_ic: float) -> bool:
    return fresh_ten_seed_ic >= fresh_three_seed_ic - DEPLOYMENT_TOLERANCE


def arm2_supported(report: Mapping[str, object]) -> bool:
    return float(report["per_date_delta_bootstrap"]["10"]["lower_95"][0]) > 0.0


def deployment_choice(
    *,
    fresh_three_seed_ic: float,
    fresh_ten_seed_ic: float,
    arm2_report: Mapping[str, object],
) -> dict[str, object]:
    sanity = sanity_band_passed(fresh_three_seed_ic)
    support = arm2_supported(arm2_report)
    deploy_ten = deploy_arm1_ten_seed(fresh_ten_seed_ic, fresh_three_seed_ic)
    if not sanity:
        recipe = None
    elif support:
        recipe = "e2_plus_archive"
    else:
        recipe = "store_v2_10_seed" if deploy_ten else "store_v2_3_seed"
    return {
        "sanity_band_passed": sanity,
        "sanity_rule": (
            "Fresh store-v2 3-seed official IC must be within +/-0.0015 of "
            "the archived 0.043235373; otherwise every deployment declaration halts."
        ),
        "arm1_deploy_10_seed": deploy_ten,
        "arm1_rule": (
            "Deploy 10 seeds iff its official IC is at least fresh 3-seed IC "
            "minus 0.0005; otherwise deploy fresh 3 seeds."
        ),
        "arm2_supported": support,
        "arm2_rule": (
            "Supported iff the paired block-10 95% interval versus the archived "
            "store-v2 comparator excludes zero from above."
        ),
        "deployed_recipe": recipe,
        "deployment_halted_for_review": not sanity,
    }


def _source_run(record: Mapping[str, object]) -> Path:
    if "source_run" in record:
        return Path(str(record["source_run"]))
    source = record.get("source")
    if isinstance(source, Mapping) and "run_dir" in source:
        return Path(str(source["run_dir"]))
    raise ValueError(f"Roster record is not backed by a run: {record['identity']}")


def _static_run_contract(manifest: Mapping[str, object]) -> dict[str, object]:
    zeroing = manifest.get("equity_input_zeroing")
    if not isinstance(zeroing, Mapping):
        raise ValueError("Source run does not record equity-input zeroing")
    variation = manifest.get("training_variation")
    if not isinstance(variation, Mapping):
        variation = {}
    sidecar = manifest.get("external_sidecar")
    return {
        "zero_dynamic_channels": list(zeroing["dynamic_channels"]),
        "zero_slow_fields": list(zeroing["slow_fields"]),
        "external_sidecar": sidecar,
        "training_horizon_indices": variation.get("training_horizon_indices"),
    }


def _identity_parts(identity: str) -> tuple[str, int, str]:
    family, seed_value, state = identity.split("|", 2)
    if not seed_value.startswith("seed_"):
        raise ValueError(f"Malformed member identity: {identity}")
    return family, int(seed_value.removeprefix("seed_")), state


def _job_name(family: str, seed: int) -> str:
    return f"{family}__seed_{seed}"


def _source_hash(path: Path, expected: str) -> dict[str, object]:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"Frozen source hash differs: {path}: {actual} != {expected}")
    return _artifact(path)


def freeze_design(*, output_dir: Path, preregistration: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    store = resolve_feature_store()
    experiment43 = RUN_OUTPUT_BASE / OFFICIAL_READ_ARTIFACT
    experiment44 = RUN_OUTPUT_BASE / ENSEMBLE_SCIENCE_ARTIFACT
    experiment43_manifest = experiment43 / "official_read_manifest.json"
    experiment43_analysis = (
        experiment43 / "analysis" / "store_v2_vs_canonical" / "analysis.json"
    )
    experiment44_analysis = experiment44 / "analysis" / "experiment44_analysis.json"
    experiment44_catalogue = experiment44 / "freeze" / "combined_catalogue.json"
    experiment44_design = experiment44 / "freeze" / "frozen_design.json"
    selection_file = RUN_OUTPUT_BASE / SELECTION_ARTIFACT / "trajectory_selection.json"
    feature_spec = (
        RUN_OUTPUT_BASE
        / FEATURE_REMOVAL_ARTIFACT
        / "stage_c"
        / "store_v2_feature_specification.json"
    )
    required = (
        preregistration,
        experiment43_manifest,
        experiment43_analysis,
        experiment44_analysis,
        experiment44_catalogue,
        experiment44_design,
        selection_file,
        feature_spec,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Experiment 45 preflight inputs missing: {missing}")
    core_sources = [
        _source_hash(experiment43_manifest, EXPECTED_EXPERIMENT43_MANIFEST_SHA256),
        _source_hash(experiment43_analysis, EXPECTED_EXPERIMENT43_ANALYSIS_SHA256),
        _source_hash(experiment44_analysis, EXPECTED_EXPERIMENT44_ANALYSIS_SHA256),
        _source_hash(experiment44_catalogue, EXPECTED_EXPERIMENT44_CATALOGUE_SHA256),
        _source_hash(experiment44_design, EXPECTED_EXPERIMENT44_DESIGN_SHA256),
        _artifact(selection_file),
        _artifact(feature_spec),
        _artifact(preregistration),
    ]
    selection = _read_json(selection_file)
    if selection.get("selected_rule") != PATIENCE_RULE:
        raise ValueError("Experiment 45 requires the frozen Raw Patience-3 rule")
    archived_analysis = _read_json(experiment43_analysis)
    if (
        abs(float(archived_analysis["candidate"]["ensemble_ic"]) - ARCHIVED_STORE_V2_IC)
        > 5e-10
    ):
        raise ValueError("Archived Experiment-43 store-v2 IC differs")
    analysis = _read_json(experiment44_analysis)
    consensus = derive_consensus(analysis)
    if consensus["withdrawn"]:
        raise ValueError("The frozen consensus arm was withdrawn")
    catalogue = _read_json(experiment44_catalogue)
    records = list(catalogue["members"])
    records_by_identity: defaultdict[str, list[Mapping[str, object]]] = defaultdict(
        list
    )
    for record in records:
        records_by_identity[str(record["identity"])].append(record)

    sample_index = load_sample_index(store, through=VALIDATION_END)
    fit_rows, _, _ = select_training_window(sample_index, "official")
    fit_dates = tuple(fit_rows.get_column("trade_date").unique().sort().to_list())
    if len(fit_dates) != 716:
        raise ValueError(
            f"Official fit window must have 716 dates, found {len(fit_dates)}"
        )

    jobs: list[dict[str, object]] = [
        {
            "job_name": _job_name("arm1_store_v2", seed),
            "family": "arm1_store_v2",
            "seed": seed,
            "states": [PATIENCE_RULE],
            "official_monitoring": True,
            "zero_dynamic_channels": list(STORE_V2_DYNAMIC),
            "zero_slow_fields": list(STORE_V2_SLOW),
            "external_sidecar": None,
            "date_multiset": None,
            "training_horizon_indices": None,
            "arm": "arm1",
            "source_manifest_inventory": [],
        }
        for seed in ALL_SEEDS
    ]
    source_manifests: list[dict[str, object]] = []
    realization_by_key: dict[tuple[str, int], dict[str, object]] = {}
    member_realizations: dict[str, dict[str, object]] = {}
    for row in consensus["members"]:
        identity = str(row["identity"])
        family, seed, state = _identity_parts(identity)
        if identity in COMPARATOR_IDENTITIES:
            member_realizations[identity] = {
                "job_name": _job_name("arm1_store_v2", seed),
                "state": state,
                "horizon_coverage": [0, 1, 2],
            }
            continue
        member_records = records_by_identity[identity]
        if len(member_records) != 3:
            raise ValueError(f"Consensus member lacks three frozen folds: {identity}")
        coverages = {
            tuple(int(value) for value in record["horizon_coverage"])
            for record in member_records
        }
        if len(coverages) != 1:
            raise ValueError(f"Consensus member horizon coverage differs: {identity}")
        run_contracts = []
        manifest_inventory = []
        for record in member_records:
            run_dir = _source_run(record)
            manifest_path = run_dir / "run_manifest.json"
            expected_hash = (
                record.get("source_run_manifest_sha256")
                if "source_run" in record
                else record["source"]["run_manifest_sha256"]
            )
            if _sha256(manifest_path) != expected_hash:
                raise ValueError(f"Frozen member source manifest differs: {identity}")
            manifest = _read_json(manifest_path)
            run_contracts.append(_static_run_contract(manifest))
            item = _artifact(manifest_path)
            manifest_inventory.append(item)
            source_manifests.append(item)
        canonical_contract = run_contracts[0]
        comparable_keys = (
            "zero_dynamic_channels",
            "zero_slow_fields",
            "external_sidecar",
            "training_horizon_indices",
        )
        if any(
            any(contract[key] != canonical_contract[key] for key in comparable_keys)
            for contract in run_contracts[1:]
        ):
            raise ValueError(f"Static source contracts differ across folds: {identity}")
        sidecar = canonical_contract["external_sidecar"]
        if sidecar is not None:
            sidecar_path = Path(str(sidecar["path"]))
            sidecar_manifest = sidecar_path / "manifest.json"
            if not sidecar_manifest.is_file():
                raise FileNotFoundError(sidecar_manifest)
            source_manifests.append(_artifact(sidecar_manifest))
        key = (family, seed)
        existing = realization_by_key.get(key)
        if existing is None:
            horizon_indices = canonical_contract["training_horizon_indices"]
            date_multiset = None
            bag_seed = None
            if family == "e2a_bagged_dates":
                bag_seed = _derived_seed("experiment44", "e2a", "official", seed)
                date_multiset = [
                    str(value) for value in _bagged_dates(fit_dates, bag_seed)
                ]
            existing = {
                "job_name": _job_name(family, seed),
                "family": family,
                "seed": seed,
                "states": [],
                "official_monitoring": False,
                "zero_dynamic_channels": canonical_contract["zero_dynamic_channels"],
                "zero_slow_fields": canonical_contract["zero_slow_fields"],
                "external_sidecar": sidecar,
                "date_multiset": date_multiset,
                "bag_seed": bag_seed,
                "training_horizon_indices": horizon_indices,
                "arm": "arm2",
                "source_manifest_inventory": manifest_inventory,
            }
            realization_by_key[key] = existing
        if state not in existing["states"]:
            existing["states"].append(state)
        if state == PATIENCE_RULE:
            existing["official_monitoring"] = True
        member_realizations[identity] = {
            "job_name": existing["job_name"],
            "state": state,
            "horizon_coverage": list(next(iter(coverages))),
        }
    arm2_jobs = list(realization_by_key.values())
    if len(arm2_jobs) > MAX_ARM2_NEW_TRAJECTORIES:
        raise ValueError(
            f"Consensus requires {len(arm2_jobs)} new Arm-2 trajectories; trim required"
        )
    jobs.extend(sorted(arm2_jobs, key=lambda row: str(row["job_name"])))
    output_dir.mkdir(parents=True)
    preregistration_copy = output_dir / "preregistration.md"
    preregistration_copy.write_bytes(preregistration.read_bytes())
    design = {
        "schema": "EXPERIMENT45_FROZEN_DESIGN_V1",
        "created_at": _now(),
        "repository_commit": repository_commit(),
        "preregistration": str(preregistration_copy.resolve()),
        "preregistration_sha256": _sha256(preregistration_copy),
        "feature_store": str(store.resolve()),
        "feature_store_identity": feature_store_identity(store),
        "official_fit_date_count": len(fit_dates),
        "selection_file": str(selection_file.resolve()),
        "selection_file_sha256": _sha256(selection_file),
        "source_roots": {
            "experiment43": str(experiment43.resolve()),
            "experiment44": str(experiment44.resolve()),
        },
        "core_source_inventory": core_sources,
        "selected_source_manifest_inventory": sorted(
            {item["path"]: item for item in source_manifests}.values(),
            key=lambda item: str(item["path"]),
        ),
        "consensus": consensus,
        "member_realizations": member_realizations,
        "jobs": jobs,
        "arm1_trajectory_count": len(ALL_SEEDS),
        "arm2_new_trajectory_count": len(arm2_jobs),
        "maximum_parallel_training_processes": 2,
        "sanity_band": SANITY_BAND,
        "deployment_tolerance": DEPLOYMENT_TOLERANCE,
        "arm2_block10_lower_bound_must_exceed_zero": True,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    path = output_dir / "frozen_design.json"
    _atomic_json(path, design)
    _atomic_json(
        output_dir / "freeze_manifest.json",
        {
            "schema": "EXPERIMENT45_FREEZE_MANIFEST_V1",
            "created_at": _now(),
            "design": str(path.resolve()),
            "design_sha256": _sha256(path),
            "new_member_training_started": False,
            "official_prediction_opened": False,
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    return path


def _write_history(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_reference(path: Path, observations: EvaluationObservations) -> None:
    _atomic_npz(
        path,
        {
            name: getattr(observations, name)
            for name in EvaluationObservations.__dataclass_fields__
            if name != "predictions"
        },
    )


def _run_fixed_final_ema(
    *, store: Path, run_dir: Path, job: Mapping[str, object]
) -> Path:
    seed = int(job["seed"])
    sidecar_identity = job["external_sidecar"]
    sidecar = (
        None
        if sidecar_identity is None
        else load_external_sidecar(Path(str(sidecar_identity["path"])), store)
    )
    if sidecar is not None and sidecar.identity != sidecar_identity:
        raise ValueError("Recorded external sidecar identity differs")
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "validation_predictions").mkdir()
    set_seeds(seed)
    torch.set_float32_matmul_precision("high")
    sample_index = load_sample_index(store, through=VALIDATION_END)
    train_rows, validation_rows, selection_note = select_training_window(
        sample_index, "official"
    )
    date_multiset = (
        None
        if job["date_multiset"] is None
        else tuple(date.fromisoformat(str(value)) for value in job["date_multiset"])
    )
    horizons = (
        None
        if job["training_horizon_indices"] is None
        else tuple(int(value) for value in job["training_horizon_indices"])
    )
    dynamic = tuple(int(value) for value in job["zero_dynamic_channels"])
    slow = tuple(int(value) for value in job["zero_slow_fields"])
    train_loader, validation_loader, sampler = create_training_loaders(
        store,
        train_rows,
        validation_rows,
        GH200_RUNTIME,
        seed,
        sidecar,
        dynamic,
        slow,
        date_multiset,
        horizons,
    )
    single_horizon = (
        horizons[0] if horizons is not None and len(horizons) == 1 else None
    )
    model = build_model(
        None if sidecar is None else sidecar.feature_count,
        single_horizon_index=single_horizon,
    ).cuda()
    ema = ModelEMA(model, 0.995)
    optimizer, _ = build_optimizer(model)
    scheduler, steps_per_epoch, warmup_steps = build_scheduler(
        optimizer, train_rows.height, MAX_EPOCHS
    )
    identity = feature_store_identity(store)
    provenance = build_run_provenance(
        repository_commit_value=repository_commit(),
        feature_store=store,
        feature_store_metadata=identity,
        seed=seed,
        fit_window=sample_window_metadata(train_rows, "official_fit"),
        selection_window=sample_window_metadata(validation_rows, "official_read"),
        selection_note=selection_note,
        parameter_count=count_trainable_parameters(model),
        training_sample_count=train_rows.height,
        date_replacement=sampler.replace_dates,
        external_sidecar=None if sidecar is None else sidecar.identity,
    )
    provenance["equity_input_zeroing"] = {
        "scope": "158_equity_inputs_only",
        "dynamic_channels": list(dynamic),
        "slow_fields": list(slow),
        "context_and_global_inputs_unchanged": True,
        "history_masks_unchanged": True,
        "applied_from_epoch_zero": True,
    }
    serialized_dates = (
        None if date_multiset is None else [str(value) for value in date_multiset]
    )
    provenance["training_variation"] = {
        "date_multiset": serialized_dates,
        "date_multiset_sha256": (
            None
            if serialized_dates is None
            else hashlib.sha256(
                ("\n".join(serialized_dates) + "\n").encode("utf-8")
            ).hexdigest()
        ),
        "training_horizon_indices": None if horizons is None else list(horizons),
        "single_horizon_head": single_horizon,
        "selection_window_unchanged": True,
    }
    if (
        provenance["training"]["steps_per_epoch"],
        provenance["training"]["warmup_steps"],
    ) != (steps_per_epoch, warmup_steps):
        raise RuntimeError("Scheduler and fixed-fit provenance differ")
    manifest = {
        "status": "running",
        "created_at": _now(),
        "repository_commit": provenance["repository_commit"],
        "run_provenance": provenance,
        "feature_store": str(store.resolve()),
        "feature_store_identity": identity,
        "external_sidecar": None if sidecar is None else sidecar.identity,
        "equity_input_zeroing": provenance["equity_input_zeroing"],
        "training_variation": provenance["training_variation"],
        "split": {
            "training": "official",
            "selection": "final_only_official_read",
            "test_accessed": False,
        },
        "seed": seed,
        "model": provenance["model"],
        "parameter_count": count_trainable_parameters(model),
        "objective": objective_metadata(),
        "optimizer": "sam_adamw",
        "sam": sam_metadata(),
        "training": {
            **provenance["training"],
            "official_monitoring": False,
            "validation_evaluations": 1,
            "readout": "final_epoch_ema_0995",
        },
        "official_validation_accessed": True,
        "test_accessed": False,
    }
    _atomic_json(run_dir / "run_manifest.json", manifest)
    compiled_model = compile_model(model)
    compiled_objective = compile_training_objective()
    history = []
    started = time.perf_counter()
    try:
        for epoch in range(1, MAX_EPOCHS + 1):
            epoch_started = time.perf_counter()
            sampler.set_epoch(epoch)
            result = train_one_epoch(
                compiled_model,
                train_loader,
                optimizer,
                scheduler,
                GH200_RUNTIME,
                compiled_objective,
                after_update=lambda: ema.update(model),
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_objective_loss": result["objective_loss"],
                    "optimizer_steps": result["optimizer_steps"],
                    "epoch_seconds": time.perf_counter() - epoch_started,
                }
            )
            _write_history(run_dir / "history.csv", history)
        with temporarily_load_state(model, ema.shadow):
            observations, loss = collect_validation_observations(
                model, validation_loader
            )
        score = validation_primary_metric(observations)
        _write_reference(run_dir / "validation_reference.npz", observations)
        _atomic_npz(
            run_dir / "validation_predictions" / "epoch_20.npz",
            {"ema_0995": observations.predictions},
        )
        temporary = run_dir / "checkpoints" / "epoch_20.pt.tmp"
        torch.save(
            checkpoint_payload(
                model,
                {"ema_0995": ema.cpu_state_dict()},
                seed=seed,
                epoch=MAX_EPOCHS,
                validation_scores={"ema_0995": score},
                feature_store=store,
                run_provenance=provenance,
            ),
            temporary,
        )
        os.replace(temporary, run_dir / "checkpoints" / "epoch_20.pt")
        summary, daily = summarize_evaluation_observations(observations, loss)
        _atomic_json(run_dir / "validation_metrics.json", summary)
        pl.DataFrame(daily).write_parquet(run_dir / "validation_daily_metrics.parquet")
        _atomic_json(
            run_dir / "run_manifest.json",
            {
                **manifest,
                "status": "completed",
                "completed_at": _now(),
                "epochs_completed": MAX_EPOCHS,
                "final_ema_0995_validation_primary_ic": score,
                "final_validation_objective_loss": loss,
                "total_run_seconds": time.perf_counter() - started,
            },
        )
    except BaseException:
        _atomic_json(
            run_dir / "run_manifest.json",
            {
                **manifest,
                "status": "failed",
                "failed_at": _now(),
                "epochs_completed": len(history),
                "total_run_seconds": time.perf_counter() - started,
            },
        )
        raise
    return run_dir


def _job_run_dir(program_root: Path, job: Mapping[str, object]) -> Path:
    return program_root / "runs" / str(job["job_name"])


def _completed_job(
    run_dir: Path, job: Mapping[str, object], design_sha256: str
) -> bool:
    manifest_path = run_dir / "run_manifest.json"
    contract_path = run_dir / "experiment45_member_contract.json"
    if not manifest_path.is_file() or not contract_path.is_file():
        return False
    manifest = _read_json(manifest_path)
    contract = _read_json(contract_path)
    return bool(
        manifest.get("status") == "completed"
        and manifest.get("repository_commit") == repository_commit()
        and manifest.get("split", {}).get("test_accessed") is False
        and contract.get("design_sha256") == design_sha256
        and contract.get("job") == job
        and contract.get("test_accessed") is False
    )


def _train_job(
    store: Path,
    program_root: Path,
    job: Mapping[str, object],
    selection_file: Path,
    design_sha256: str,
) -> str:
    run_dir = _job_run_dir(program_root, job)
    if run_dir.exists():
        if not _completed_job(run_dir, job, design_sha256):
            raise ValueError(f"Existing Experiment-45 job differs: {run_dir}")
        return str(run_dir)
    sidecar_identity = job["external_sidecar"]
    sidecar_dir = (
        None if sidecar_identity is None else Path(str(sidecar_identity["path"]))
    )
    if job["official_monitoring"]:
        date_multiset = (
            None
            if job["date_multiset"] is None
            else tuple(date.fromisoformat(str(value)) for value in job["date_multiset"])
        )
        horizons = (
            None
            if job["training_horizon_indices"] is None
            else tuple(int(value) for value in job["training_horizon_indices"])
        )
        run_training(
            store=store,
            seed=int(job["seed"]),
            selection_window="official",
            selection_rule_file=selection_file,
            run_dir=run_dir,
            sidecar_dir=sidecar_dir,
            zero_dynamic_channels=tuple(
                int(value) for value in job["zero_dynamic_channels"]
            ),
            zero_slow_fields=tuple(int(value) for value in job["zero_slow_fields"]),
            date_multiset=date_multiset,
            training_horizon_indices=horizons,
        )
    else:
        _run_fixed_final_ema(store=store, run_dir=run_dir, job=job)
    _atomic_json(
        run_dir / "experiment45_member_contract.json",
        {
            "schema": "EXPERIMENT45_MEMBER_REALIZATION_V1",
            "created_at": _now(),
            "design_sha256": design_sha256,
            "job": dict(job),
            "official_validation_accessed": True,
            "test_accessed": False,
        },
    )
    return str(run_dir)


def _run_jobs(
    *, program_root: Path, design: Mapping[str, object], design_sha256: str
) -> None:
    store = Path(str(design["feature_store"]))
    selection_file = Path(str(design["selection_file"]))
    jobs = list(design["jobs"])
    with ProcessPoolExecutor(
        max_workers=2, mp_context=mp.get_context("spawn")
    ) as executor:
        futures = [
            executor.submit(
                _train_job,
                store,
                program_root,
                job,
                selection_file,
                design_sha256,
            )
            for job in jobs
        ]
        for future in as_completed(futures):
            print(future.result(), flush=True)


def _member_observation(run_dir: Path, state: str) -> EvaluationObservations:
    return replace(
        _load_reference(run_dir), predictions=predictions_for_rule(run_dir, state)
    )


def _comparison(
    *,
    candidate: Mapping[str, EvaluationObservations],
    parent: Mapping[str, EvaluationObservations],
    candidate_rule: str,
    parent_rule: str,
    output_dir: Path,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    analysis_path = output_dir / "analysis.json"
    if analysis_path.is_file():
        report = _read_json(analysis_path)
        if (
            report.get("comparison_metadata")
            != {
                **metadata,
                "official_validation_accessed": True,
                "test_accessed": False,
            }
            or report.get("candidate_rule") != candidate_rule
            or report.get("parent_rule") != parent_rule
        ):
            raise ValueError(f"Existing comparison contract differs: {output_dir}")
        return report
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
    return _read_json(analysis_path)


def _composition_diagnostics(
    members: Mapping[str, EvaluationObservations],
) -> dict[str, object]:
    reference = next(iter(members.values()))
    ranks = {}
    member_ic = {}
    for identity, member in members.items():
        assert_observations_aligned(reference, member)
        member_ic[identity] = primary_validation_score(
            member.predictions,
            reference.targets,
            reference.label_mask,
            reference.date_idx,
        )
        ranks[identity] = rank_transform_predictions(
            member.predictions, reference.label_mask
        )
    pairs = []
    for left, right in combinations(members, 2):
        pairs.append(
            {
                "left": left,
                "right": right,
                "prediction_spearman": rank_prediction_similarity(
                    ranks[left], ranks[right], reference.label_mask, reference.date_idx
                ),
            }
        )
    return {"member_ic": member_ic, "prediction_spearman_pairs": pairs}


def _deployed_checkpoint_inventory(
    *,
    program_root: Path,
    design: Mapping[str, object],
    deployed_recipe: str | None,
    deploy_ten: bool,
) -> tuple[list[str], list[dict[str, object]]]:
    if deployed_recipe is None:
        return [], []
    jobs = {str(job["job_name"]): job for job in design["jobs"]}
    if deployed_recipe == "e2_plus_archive":
        job_names = sorted(
            {
                str(realization["job_name"])
                for realization in design["member_realizations"].values()
            }
        )
    else:
        seeds = ALL_SEEDS if deploy_ten else BASE_SEEDS
        job_names = [_job_name("arm1_store_v2", seed) for seed in seeds]
    inventory = []
    for job_name in job_names:
        job = jobs[job_name]
        run_dir = _job_run_dir(program_root, job)
        paths = {run_dir / "checkpoints" / "epoch_20.pt"}
        if job["official_monitoring"]:
            diagnostics = _read_json(run_dir / "trajectory_diagnostics.json")
            selected_epoch = int(diagnostics["patience3"]["selected_epoch"])
            paths.add(run_dir / "checkpoints" / f"epoch_{selected_epoch:02d}.pt")
        inventory.extend(_artifact(path) for path in sorted(paths))
    return job_names, inventory


def run_read(*, program_root: Path, design_path: Path) -> Path:
    freeze_manifest = _read_json(design_path.parent / "freeze_manifest.json")
    design_sha256 = _sha256(design_path)
    if freeze_manifest.get("design_sha256") != design_sha256:
        raise ValueError("Experiment-45 frozen design hash differs")
    design = _read_json(design_path)
    if (
        design.get("repository_commit") != repository_commit()
        or design.get("official_validation_accessed") is not False
        or design.get("test_accessed") is not False
    ):
        raise ValueError("Experiment-45 frozen design contract differs")
    jobs = list(design["jobs"])
    official_monitor_jobs = [
        str(_job_run_dir(program_root, job).resolve())
        for job in jobs
        if job["official_monitoring"]
    ]
    fixed_jobs = [
        str(_job_run_dir(program_root, job).resolve())
        for job in jobs
        if not job["official_monitoring"]
    ]
    ledger_path = program_root / "validation_access_ledger.json"
    _atomic_json(
        ledger_path,
        {
            "schema": "VALIDATION_ACCESS_LEDGER_EVENT_V1",
            "status": "running",
            "created_at": _now(),
            "event": 4,
            "experiment": 45,
            "arms": ["store_v2_3_to_10", "e2_plus_archive_consensus"],
            "official_monitor_runs": official_monitor_jobs,
            "fixed_no_monitor_runs": fixed_jobs,
            "official_validation_accessed": True,
            "test_accessed": False,
        },
    )
    manifest_path = program_root / "consolidation_read_manifest.json"
    _atomic_json(
        manifest_path,
        {
            "schema": "CONSOLIDATION_READ_EXPERIMENT_45_V1",
            "status": "running",
            "created_at": _now(),
            "repository_commit": repository_commit(),
            "design": str(design_path.resolve()),
            "design_sha256": design_sha256,
            "official_validation_accessed": True,
            "test_accessed": False,
        },
    )
    _run_jobs(program_root=program_root, design=design, design_sha256=design_sha256)

    source43 = Path(str(design["source_roots"]["experiment43"]))
    archived = _current_members(source43, "store_v2", BASE_SEEDS)
    archived_reference = next(iter(archived.values()))
    archived_ic = primary_validation_score(
        combine_rank_predictions(
            [member.predictions for member in archived.values()],
            archived_reference.label_mask,
        ),
        archived_reference.targets,
        archived_reference.label_mask,
        archived_reference.date_idx,
    )
    if abs(archived_ic - ARCHIVED_STORE_V2_IC) > 5e-10:
        raise ValueError(f"Archived comparator IC differs at read: {archived_ic}")
    fresh = {
        seed: _member_observation(
            program_root / "runs" / _job_name("arm1_store_v2", seed),
            PATIENCE_RULE,
        )
        for seed in ALL_SEEDS
    }
    fresh_three = {f"seed_{seed}": fresh[seed] for seed in BASE_SEEDS}
    fresh_ten = {f"seed_{seed}": fresh[seed] for seed in ALL_SEEDS}
    arm1_three = _comparison(
        candidate=fresh_three,
        parent=archived,
        candidate_rule="fresh_store_v2_3_seed_patience3_raw",
        parent_rule="archived_experiment43_store_v2_3_seed",
        output_dir=program_root / "analysis" / "arm1_fresh3_vs_archived3",
        metadata={"arm": "arm1", "measurement": "reproduction_sanity"},
    )
    arm1_ten = _comparison(
        candidate=fresh_ten,
        parent=archived,
        candidate_rule="fresh_store_v2_10_seed_patience3_raw",
        parent_rule="archived_experiment43_store_v2_3_seed",
        output_dir=program_root / "analysis" / "arm1_fresh10_vs_archived3",
        metadata={"arm": "arm1", "measurement": "deployment_form"},
    )
    arm1_gain = _comparison(
        candidate=fresh_ten,
        parent=fresh_three,
        candidate_rule="fresh_store_v2_10_seed_patience3_raw",
        parent_rule="fresh_store_v2_3_seed_patience3_raw",
        output_dir=program_root / "analysis" / "arm1_fresh10_vs_fresh3",
        metadata={"arm": "arm1", "measurement": "seed_expansion"},
    )
    diagnostics = _seed_diagnostics({seed: (fresh[seed],) for seed in ALL_SEEDS})
    fresh_three_ic = float(arm1_three["candidate"]["ensemble_ic"])
    fresh_ten_ic = float(arm1_ten["candidate"]["ensemble_ic"])
    _atomic_json(
        program_root / "analysis" / "arm1_seed_diagnostics.json",
        {
            "schema": "EXPERIMENT45_ARM1_SEED_DIAGNOSTICS_V1",
            "created_at": _now(),
            "fresh_three_seed_ic": fresh_three_ic,
            "fresh_ten_seed_ic": fresh_ten_ic,
            "archived_comparator_ic": archived_ic,
            "sanity_band": SANITY_BAND,
            "sanity_band_passed": sanity_band_passed(fresh_three_ic),
            **diagnostics,
            "official_validation_accessed": True,
            "test_accessed": False,
        },
    )

    composition_members: dict[str, EvaluationObservations] = {}
    weights = []
    coverages = []
    for member in design["consensus"]["members"]:
        identity = str(member["identity"])
        realization = design["member_realizations"][identity]
        run_dir = program_root / "runs" / str(realization["job_name"])
        observation = _member_observation(run_dir, str(realization["state"]))
        assert_observations_aligned(archived_reference, observation)
        composition_members[identity] = observation
        weights.append(float(member["raw_weight"]))
        coverages.append(tuple(int(value) for value in realization["horizon_coverage"]))
    composition_predictions = combine_rank_predictions(
        [member.predictions for member in composition_members.values()],
        archived_reference.label_mask,
        weights=weights,
        horizon_coverage=coverages,
    )
    composition = replace(archived_reference, predictions=composition_predictions)
    arm2 = _comparison(
        candidate={"e2_plus_archive": composition},
        parent=archived,
        candidate_rule="e2_plus_archive_consensus_weighted_rank",
        parent_rule="archived_experiment43_store_v2_3_seed",
        output_dir=program_root / "analysis" / "arm2_vs_archived3",
        metadata={
            "arm": "arm2",
            "composition": design["consensus"],
            "member_realizations": design["member_realizations"],
        },
    )
    _atomic_json(
        program_root / "analysis" / "arm2_member_diagnostics.json",
        {
            "schema": "EXPERIMENT45_ARM2_MEMBER_DIAGNOSTICS_V1",
            "created_at": _now(),
            **_composition_diagnostics(composition_members),
            "official_validation_accessed": True,
            "test_accessed": False,
        },
    )
    decision = deployment_choice(
        fresh_three_seed_ic=fresh_three_ic,
        fresh_ten_seed_ic=fresh_ten_ic,
        arm2_report=arm2,
    )
    decision.update(
        {
            "schema": "EXPERIMENT45_PROMOTION_DECISION_V1",
            "created_at": _now(),
            "archived_comparator_ic": archived_ic,
            "fresh_three_seed_ic": fresh_three_ic,
            "fresh_ten_seed_ic": fresh_ten_ic,
            "arm2_ic": float(arm2["candidate"]["ensemble_ic"]),
            "arm2_block10_interval": arm2["per_date_delta_bootstrap"]["10"],
            "official_validation_accessed": True,
            "test_accessed": False,
        }
    )
    _atomic_json(program_root / "promotion_decision.json", decision)

    deploy_ten = bool(decision["arm1_deploy_10_seed"])
    deployed_recipe = decision["deployed_recipe"]
    deployed_jobs, checkpoint_inventory = _deployed_checkpoint_inventory(
        program_root=program_root,
        design=design,
        deployed_recipe=deployed_recipe,
        deploy_ten=deploy_ten,
    )
    if deployed_recipe == "e2_plus_archive":
        deployed_members = design["consensus"]["members"]
        ensemble = "weighted tie-aware rank average with per-horizon membership"
    elif deployed_recipe is not None:
        deployed_seeds = ALL_SEEDS if deploy_ten else BASE_SEEDS
        deployed_members = [
            {
                "identity": f"store_v2|seed_{seed}|patience3_raw",
                "raw_weight": 1,
                "normalized_weight": 1 / len(deployed_seeds),
            }
            for seed in deployed_seeds
        ]
        ensemble = "uniform tie-aware rank average"
    else:
        deployed_members = []
        ensemble = None
    deployed = {
        "schema": "EXPERIMENT45_DEPLOYED_RECIPE_V1",
        "created_at": _now(),
        "recipe": deployed_recipe,
        "deployment_halted_for_review": decision["deployment_halted_for_review"],
        "members": deployed_members,
        "ensemble": ensemble,
        "measured_member_job_names": deployed_jobs,
        "retained_checkpoint_inventory": checkpoint_inventory,
        "root_cause_retention_rule": (
            "Retain every deployed measured member's selected and final-state "
            "checkpoint until superseded by a future deployment."
        ),
        "official_validation_accessed": True,
        "test_accessed": False,
    }
    _atomic_json(program_root / "deployed_recipe.json", deployed)
    canonical = _parent_members(RUN_OUTPUT_BASE / PARENT_ARTIFACT)
    canonical_reference = next(iter(canonical.values()))
    canonical_ic = primary_validation_score(
        combine_rank_predictions(
            [member.predictions for member in canonical.values()],
            canonical_reference.label_mask,
        ),
        canonical_reference.targets,
        canonical_reference.label_mask,
        canonical_reference.date_idx,
    )
    _atomic_json(
        program_root / "reference_baselines.json",
        {
            "schema": "EXPERIMENT45_REFERENCE_BASELINES_V1",
            "created_at": _now(),
            "archived_store_v2_3_seed_ic": archived_ic,
            "canonical_parent58_3_seed_ic": canonical_ic,
            "decision_comparator": "archived_store_v2_3_seed",
            "canonical_is_reference_only": True,
            "official_validation_accessed": True,
            "test_accessed": False,
        },
    )
    _atomic_json(
        ledger_path,
        {
            **_read_json(ledger_path),
            "status": "completed",
            "completed_at": _now(),
            "deployment_declaration": str(
                (program_root / "deployed_recipe.json").resolve()
            ),
        },
    )
    outputs = sorted(
        path
        for path in program_root.rglob("*")
        if path.is_file() and path != manifest_path and not path.name.endswith(".tmp")
    )
    _atomic_json(
        manifest_path,
        {
            **_read_json(manifest_path),
            "status": "completed",
            "completed_at": _now(),
            "decision": decision,
            "deployed_recipe": deployed,
            "arm1_analyses": {
                "fresh3_vs_archived3": arm1_three,
                "fresh10_vs_archived3": arm1_ten,
                "fresh10_vs_fresh3": arm1_gain,
            },
            "arm2_analysis": arm2,
            "required_output_inventory": [_artifact(path) for path in outputs],
            "official_validation_accessed": True,
            "test_accessed": False,
        },
    )
    return program_root


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen Experiment-45 consolidation read"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--output-dir", type=Path, required=True)
    freeze.add_argument("--preregistration", type=Path, default=PREREGISTRATION)
    run = subparsers.add_parser("run")
    run.add_argument("--program-root", type=Path, required=True)
    run.add_argument("--design", type=Path, required=True)
    return parser.parse_args(arguments)


def main() -> None:
    args = parse_args()
    if args.command == "freeze":
        result = freeze_design(
            output_dir=args.output_dir, preregistration=args.preregistration
        )
    else:
        result = run_read(program_root=args.program_root, design_path=args.design)
    print(result)


if __name__ == "__main__":
    main()
