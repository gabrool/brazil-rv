from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from .contract import ALLOWED_SEEDS, HORIZONS, VALIDATION_END
from .data import (
    feature_store_identity,
    load_sample_index,
    resolve_feature_store,
    select_training_window,
)
from .engine import EvaluationObservations, assert_observations_aligned
from .feature_removal import FEATURES, _definition
from .metrics import (
    combine_rank_predictions,
    daily_horizon_ic,
    finite_mean,
    moving_block_bootstrap,
    per_date_primary_ic,
    primary_validation_score,
    sample_level_spearman_ic,
)
from .provenance import repository_commit
from .three_fold_sidecar_screen import crossfit_patience_observations
from .train import run_training
from .trajectory import predictions_for_rule

FOLDS = ("fold_c", "fold_a", "fold_b")
E2_FAMILIES = ("e2a_bagged_dates", "e2b_feature_subspace")
E2_HORIZON_FAMILIES = tuple(f"e2c_horizon_{minutes}" for minutes in HORIZONS)
E2_ALL_FAMILIES = (*E2_FAMILIES, *E2_HORIZON_FAMILIES)
STATES = ("patience3_raw", "final_ema_0995")
WEIGHT_GRID = (0.5, 0.6, 0.7, 0.8, 0.9)
BOOTSTRAP_BLOCKS = (5, 10)
BOOTSTRAP_REPLICATIONS = 10_000
MAX_GREEDY_ADDITIONS = 12
STORE_V2_DYNAMIC = (9, 11, 14, 22, 24, 25)
STORE_V2_SLOW = (1, 2, 3, 12, 13, 14, 15, 16, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29)
SUPERIORITY_MEAN = 0.001
PRIMARY_DATE_COUNTS = {"fold_c": 105, "fold_a": 102, "fold_b": 102}
PREREGISTRATION = Path("research/preregistrations/experiment44_ensemble_science.md")
AMENDMENT = Path("research/preregistrations/experiment43_amendment_a1.md")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _derived_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _npz_content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as values:
        for key in sorted(values.files):
            array = np.asarray(values[key])
            digest.update(key.encode("utf-8") + b"\0")
            digest.update(array.dtype.str.encode("ascii") + b"\0")
            digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _bagged_dates(dates: Sequence[object], seed: int) -> tuple[object, ...]:
    if not dates:
        raise ValueError("Cannot bag an empty fit window")
    generator = np.random.default_rng(seed)
    values: list[object] = []
    while len(values) < len(dates):
        start = int(generator.integers(len(dates)))
        values.extend(dates[(start + offset) % len(dates)] for offset in range(20))
    return tuple(values[: len(dates)])


def _kept_feature_keys() -> tuple[str, ...]:
    removed = {
        *(f"dynamic_{index}" for index in STORE_V2_DYNAMIC),
        *(f"slow_{index}" for index in STORE_V2_SLOW),
    }
    return tuple(str(row["key"]) for row in FEATURES if str(row["key"]) not in removed)


def freeze_design(*, output_dir: Path, store: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    sample_index = load_sample_index(store, through=VALIDATION_END)
    kept = _kept_feature_keys()
    if len(kept) != 34:
        raise ValueError(f"Store-v2 must retain exactly 34 fields, found {len(kept)}")
    jobs = []
    for fold in FOLDS:
        fit, selection, _ = select_training_window(sample_index, fold)
        fit_dates = tuple(fit.get_column("trade_date").unique().sort().to_list())
        if selection.get_column("trade_date").n_unique() != PRIMARY_DATE_COUNTS[fold]:
            raise ValueError(f"Unexpected selection date count for {fold}")
        for seed in ALLOWED_SEEDS:
            bag_seed = _derived_seed("experiment44", "e2a", fold, seed)
            bagged = _bagged_dates(fit_dates, bag_seed)
            subspace_seed = _derived_seed("experiment44", "e2b", fold, seed)
            generator = np.random.default_rng(subspace_seed)
            subspace = tuple(sorted(generator.choice(kept, 8, replace=False).tolist()))
            jobs.extend(
                (
                    {
                        "family": "e2a_bagged_dates",
                        "fold": fold,
                        "seed": seed,
                        "bag_seed": bag_seed,
                        "date_multiset": [str(value) for value in bagged],
                        "date_multiset_sha256": _canonical_json_sha256(
                            [str(value) for value in bagged]
                        ),
                        "additional_zero_fields": [],
                        "horizon_index": None,
                    },
                    {
                        "family": "e2b_feature_subspace",
                        "fold": fold,
                        "seed": seed,
                        "subspace_seed": subspace_seed,
                        "date_multiset": None,
                        "additional_zero_fields": list(subspace),
                        "additional_zero_fields_sha256": _canonical_json_sha256(
                            subspace
                        ),
                        "horizon_index": None,
                    },
                )
            )
            for horizon_index, minutes in enumerate(HORIZONS):
                jobs.append(
                    {
                        "family": f"e2c_horizon_{minutes}",
                        "fold": fold,
                        "seed": seed,
                        "date_multiset": None,
                        "additional_zero_fields": [],
                        "horizon_index": horizon_index,
                    }
                )
    if len(jobs) != 45:
        raise AssertionError("Experiment 44 must freeze exactly 45 trajectories")
    rule_grid = {
        "greedy": {
            "selection_with_replacement": True,
            "maximum_additions": MAX_GREEDY_ADDITIONS,
            "stop_at_nonpositive_marginal": True,
            "rotations": {
                "fold_c": ["fold_a", "fold_b"],
                "fold_a": ["fold_c", "fold_b"],
                "fold_b": ["fold_c", "fold_a"],
            },
        },
        "e1_fixed_sets": [
            "residual_ema3",
            "residual_ema3_plus_options_oi",
            "best_five_adapter_ema",
            "parent58_3",
            "full_primary_roster",
        ],
        "e2_fixed_sets": ["e2a_9", "e2b_9", "e2c_27", "all_e2"],
        "weights": list(WEIGHT_GRID),
        "hygiene": ["honest_shrunken_skill", "median_rank", "trimmed_rank_20pct"],
        "bootstrap_blocks": list(BOOTSTRAP_BLOCKS),
        "bootstrap_replications": BOOTSTRAP_REPLICATIONS,
        "superiority_mean_delta": SUPERIORITY_MEAN,
        "every_heldout_fold_nonnegative": True,
        "maximum_advanced_compositions": 1,
    }
    prereg = PREREGISTRATION.resolve()
    amendment = AMENDMENT.resolve()
    design = {
        "schema": "EXPERIMENT44_FROZEN_DESIGN_V1",
        "created_at": _now(),
        "repository_commit": repository_commit(),
        "feature_store": str(store.resolve()),
        "feature_store_identity": feature_store_identity(store),
        "preregistration": str(prereg),
        "preregistration_sha256": _sha256(prereg),
        "amendment_a1": str(amendment),
        "amendment_a1_sha256": _sha256(amendment),
        "store_v2_zeroing": {
            "dynamic": list(STORE_V2_DYNAMIC),
            "slow": list(STORE_V2_SLOW),
            "retained_fields": list(kept),
        },
        "jobs": jobs,
        "rule_grid": rule_grid,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    path = output_dir / "frozen_design.json"
    _atomic_json(path, design)
    _atomic_json(
        output_dir / "freeze_manifest.json",
        {
            "schema": "EXPERIMENT44_FREEZE_MANIFEST_V1",
            "created_at": _now(),
            "design": str(path.resolve()),
            "design_sha256": _sha256(path),
            "scores_computed": False,
            "trajectories_started": False,
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    return path


def _job_run_dir(program_root: Path, job: Mapping[str, object]) -> Path:
    return (
        program_root
        / "e2_runs"
        / str(job["family"])
        / str(job["fold"])
        / f"seed_{job['seed']}"
    )


def _train_e2_job(store: Path, program_root: Path, job: Mapping[str, object]) -> str:
    run_dir = _job_run_dir(program_root, job)
    if run_dir.exists():
        manifest = _read_json(run_dir / "run_manifest.json")
        if manifest.get("status") != "completed":
            raise ValueError(f"Existing E2 run is incomplete: {run_dir}")
        return str(run_dir)
    added = tuple(str(value) for value in job["additional_zero_fields"])
    added_dynamic, added_slow = _definition(added)
    date_multiset = None
    if job["date_multiset"] is not None:
        date_multiset = tuple(
            datetime.strptime(str(value), "%Y-%m-%d").date()
            for value in job["date_multiset"]
        )
    horizon_index = job["horizon_index"]
    run_training(
        store=store,
        seed=int(job["seed"]),
        selection_window=str(job["fold"]),
        run_dir=run_dir,
        zero_dynamic_channels=tuple(sorted((*STORE_V2_DYNAMIC, *added_dynamic))),
        zero_slow_fields=tuple(sorted((*STORE_V2_SLOW, *added_slow))),
        date_multiset=date_multiset,
        training_horizon_indices=(int(horizon_index),)
        if horizon_index is not None
        else None,
    )
    return str(run_dir)


def run_e2(
    *, program_root: Path, design_path: Path, parallel_processes: int = 2
) -> Path:
    if parallel_processes not in (1, 2):
        raise ValueError("Experiment 44 permits at most two training processes")
    design = _read_json(design_path)
    if (
        _sha256(design_path)
        != _read_json(design_path.parent / "freeze_manifest.json")["design_sha256"]
    ):
        raise ValueError("Frozen Experiment-44 design hash differs")
    if (
        design.get("official_validation_accessed") is not False
        or design.get("test_accessed") is not False
    ):
        raise ValueError("Experiment-44 design does not seal official/test data")
    store = Path(str(design["feature_store"]))
    jobs = list(design["jobs"])
    if len(jobs) != 45:
        raise ValueError("Frozen E2 design does not contain 45 jobs")
    if parallel_processes == 1:
        for job in jobs:
            print(_train_e2_job(store, program_root, job), flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=2, mp_context=mp.get_context("spawn")
        ) as executor:
            futures = [
                executor.submit(_train_e2_job, store, program_root, job) for job in jobs
            ]
            for future in as_completed(futures):
                print(future.result(), flush=True)
    manifest = {
        "schema": "EXPERIMENT44_E2_TRAINING_V1",
        "status": "completed",
        "completed_at": _now(),
        "design": str(design_path.resolve()),
        "design_sha256": _sha256(design_path),
        "trajectory_count": len(jobs),
        "maximum_parallel_processes": parallel_processes,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    path = program_root / "e2_training_manifest.json"
    _atomic_json(path, manifest)
    return path


def _run_a1_job(store: Path, run_dir: Path, seed: int, selection_file: Path) -> str:
    if run_dir.exists():
        manifest = _read_json(run_dir / "run_manifest.json")
        if manifest.get("status") != "completed":
            raise ValueError(f"Existing Amendment-A1 run is incomplete: {run_dir}")
        return str(run_dir)
    run_training(
        store=store,
        seed=seed,
        selection_window="official",
        run_dir=run_dir,
        selection_rule_file=selection_file,
        zero_dynamic_channels=STORE_V2_DYNAMIC,
        zero_slow_fields=STORE_V2_SLOW,
    )
    return str(run_dir)


def run_amendment_a1(
    *,
    output_dir: Path,
    source_official_root: Path,
    selection_file: Path,
    parallel_processes: int = 2,
) -> Path:
    if parallel_processes not in (1, 2):
        raise ValueError("Amendment A1 permits at most two training processes")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    store = resolve_feature_store()
    jobs = [
        (store, output_dir / "runs" / "store_v2" / f"seed_{seed}", seed, selection_file)
        for seed in ALLOWED_SEEDS
    ]
    if parallel_processes == 1:
        for job in jobs:
            print(_run_a1_job(*job), flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=2, mp_context=mp.get_context("spawn")
        ) as executor:
            futures = [executor.submit(_run_a1_job, *job) for job in jobs]
            for future in as_completed(futures):
                print(future.result(), flush=True)
    comparisons = []
    for seed in ALLOWED_SEEDS:
        source = source_official_root / "runs" / "store_v2" / f"seed_{seed}"
        reproduced = output_dir / "runs" / "store_v2" / f"seed_{seed}"
        source_manifest = _read_json(source / "run_manifest.json")
        reproduced_manifest = _read_json(reproduced / "run_manifest.json")
        if (
            source_manifest.get("split", {}).get("test_accessed") is not False
            or reproduced_manifest.get("split", {}).get("test_accessed") is not False
        ):
            raise ValueError("Amendment A1 encountered held-out-test access")
        files = [f"epoch_{epoch:02d}.npz" for epoch in range(1, 21)] + [
            "tail_candidates.npz"
        ]
        file_rows = []
        for name in files:
            source_hash = _sha256(source / "validation_predictions" / name)
            reproduced_hash = _sha256(reproduced / "validation_predictions" / name)
            source_content = _npz_content_sha256(
                source / "validation_predictions" / name
            )
            reproduced_content = _npz_content_sha256(
                reproduced / "validation_predictions" / name
            )
            file_rows.append(
                {
                    "file": name,
                    "source_sha256": source_hash,
                    "reproduced_sha256": reproduced_hash,
                    "source_array_content_sha256": source_content,
                    "reproduced_array_content_sha256": reproduced_content,
                    "exact_match": source_content == reproduced_content,
                }
            )
        if not all(row["exact_match"] for row in file_rows):
            raise ValueError(
                f"Amendment-A1 predictions failed exact match for seed {seed}"
            )
        comparisons.append(
            {
                "seed": seed,
                "source_run": str(source.resolve()),
                "reproduced_run": str(reproduced.resolve()),
                "prediction_archives": file_rows,
            }
        )
    result = {
        "schema": "EXPERIMENT43_AMENDMENT_A1_REPRODUCTION_V1",
        "status": "completed",
        "completed_at": _now(),
        "amendment": str(AMENDMENT.resolve()),
        "amendment_sha256": _sha256(AMENDMENT.resolve()),
        "source_official_root": str(source_official_root.resolve()),
        "selection_file": str(selection_file.resolve()),
        "selection_file_sha256": _sha256(selection_file),
        "comparisons": comparisons,
        "all_prediction_archives_exact_match": True,
        "store_v2_comparator_activated": True,
        "new_candidate_or_selection": False,
        "official_validation_reaccessed_only_for_exact_reproduction": True,
        "test_accessed": False,
    }
    path = output_dir / "amendment_a1_reproduction.json"
    _atomic_json(path, result)
    return path


def _write_reference(path: Path, observations: EvaluationObservations) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{
            name: getattr(observations, name)
            for name in EvaluationObservations.__dataclass_fields__
            if name != "predictions"
        },
    )


def _load_reference(path: Path) -> EvaluationObservations:
    with np.load(path, allow_pickle=False) as values:
        fields = {name: values[name].copy() for name in values.files}
    return EvaluationObservations(
        predictions=np.zeros_like(fields["targets"]), **fields
    )


def _write_prediction(path: Path, predictions: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, predictions=predictions)


def _load_prediction(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as values:
        return values["predictions"].copy()


def materialize_e2_states(*, program_root: Path, design_path: Path) -> Path:
    design = _read_json(design_path)
    records = []
    references: dict[str, Path] = {}
    for job in design["jobs"]:
        run_dir = _job_run_dir(program_root, job)
        patience, replay = crossfit_patience_observations(run_dir)
        ema = replace(
            patience,
            predictions=predictions_for_rule(run_dir, "final_ema_0995"),
        )
        fold = str(job["fold"])
        reference_path = program_root / "member_states" / "references" / f"{fold}.npz"
        if fold not in references:
            _write_reference(reference_path, patience)
            references[fold] = reference_path
        else:
            assert_observations_aligned(_load_reference(reference_path), patience)
        for state, observations in (
            ("patience3_raw", patience),
            ("final_ema_0995", ema),
        ):
            path = (
                program_root
                / "member_states"
                / str(job["family"])
                / fold
                / f"seed_{job['seed']}_{state}.npz"
            )
            _write_prediction(path, observations.predictions)
            horizon = job["horizon_index"]
            records.append(
                {
                    "family": str(job["family"]),
                    "seed": int(job["seed"]),
                    "state": state,
                    "fold": fold,
                    "prediction": str(path.resolve()),
                    "prediction_sha256": _sha256(path),
                    "reference": str(reference_path.resolve()),
                    "reference_sha256": _sha256(reference_path),
                    "horizon_coverage": list(range(3))
                    if horizon is None
                    else [int(horizon)],
                    "source_run": str(run_dir.resolve()),
                    "source_run_manifest_sha256": _sha256(
                        run_dir / "run_manifest.json"
                    ),
                    "crossfit_replay": replay if state == "patience3_raw" else [],
                    "tags": ["e2", str(job["family"]), state],
                }
            )
    catalogue = {
        "schema": "EXPERIMENT44_E2_MEMBER_STATES_V1",
        "created_at": _now(),
        "records": records,
        "record_count": len(records),
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    path = program_root / "e2_member_catalogue.json"
    _atomic_json(path, catalogue)
    return path


def freeze_catalogue(*, provisional_path: Path, output_path: Path) -> Path:
    if output_path.exists() or output_path.with_suffix(".manifest.json").exists():
        raise FileExistsError(output_path)
    provisional = _read_json(provisional_path)
    records = list(provisional["members"])
    references = {
        str(fold): Path(str(path))
        for fold, path in dict(provisional["references"]).items()
    }
    if set(references) != set(FOLDS):
        raise ValueError("Roster must provide one reference for every discovery fold")
    loaded_references = {}
    for fold, path in references.items():
        reference = _load_reference(path)
        if np.unique(reference.date_idx).size != PRIMARY_DATE_COUNTS[fold]:
            raise ValueError(
                f"Reference does not cover the complete {fold} selection window"
            )
        loaded_references[fold] = reference
    seen = set()
    for record in records:
        identity = str(record["identity"])
        fold = str(record["fold"])
        key = (identity, fold)
        if key in seen:
            raise ValueError(f"Duplicate roster record: {key}")
        seen.add(key)
        path = Path(str(record["prediction"]))
        if _sha256(path) != record["prediction_sha256"]:
            raise ValueError(f"Roster prediction hash differs: {path}")
        predictions = _load_prediction(path)
        if predictions.shape != loaded_references[fold].targets.shape:
            raise ValueError(f"Roster prediction does not align with {fold}: {path}")
        if record.get("official_validation_accessed") not in (
            None,
            False,
        ) or record.get("test_accessed") not in (None, False):
            raise ValueError(f"Roster member accesses sealed data: {identity}")
    grouped: dict[str, set[str]] = {}
    for record in records:
        grouped.setdefault(str(record["identity"]), set()).add(str(record["fold"]))
    for record in records:
        folds = grouped[str(record["identity"])]
        expected_tier = (
            "primary"
            if set(FOLDS).issubset(folds)
            else "secondary"
            if {"fold_a", "fold_b"}.issubset(folds)
            else "ineligible"
        )
        if record["tier"] != expected_tier:
            raise ValueError(
                f"Roster tier differs for {record['identity']}: {record['tier']} vs {expected_tier}"
            )
    comparator = sorted(
        identity
        for identity, folds in grouped.items()
        if set(FOLDS).issubset(folds)
        and "comparator_store_v2"
        in next(record["tags"] for record in records if record["identity"] == identity)
    )
    if len(comparator) != 3:
        raise ValueError(
            f"Roster must freeze exactly three store-v2 comparator members, found {len(comparator)}"
        )
    catalogue = {
        "schema": "EXPERIMENT44_FROZEN_ROSTER_V1",
        "created_at": _now(),
        "source_provisional": str(provisional_path.resolve()),
        "source_provisional_sha256": _sha256(provisional_path),
        "references": {fold: str(path.resolve()) for fold, path in references.items()},
        "reference_sha256": {fold: _sha256(path) for fold, path in references.items()},
        "members": records,
        "primary_identities": sorted(
            identity
            for identity, folds in grouped.items()
            if set(FOLDS).issubset(folds)
        ),
        "secondary_identities": sorted(
            identity
            for identity, folds in grouped.items()
            if not set(FOLDS).issubset(folds) and {"fold_a", "fold_b"}.issubset(folds)
        ),
        "ineligible_identities": sorted(
            identity
            for identity, folds in grouped.items()
            if not {"fold_a", "fold_b"}.issubset(folds)
        ),
        "comparator_identities": comparator,
        "scores_computed": False,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(output_path, catalogue)
    _atomic_json(
        output_path.with_suffix(".manifest.json"),
        {
            "schema": "EXPERIMENT44_ROSTER_FREEZE_MANIFEST_V1",
            "created_at": _now(),
            "catalogue": str(output_path.resolve()),
            "catalogue_sha256": _sha256(output_path),
            "scores_computed": False,
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    return output_path


def merge_e2_catalogue(
    *, base_catalogue_path: Path, e2_catalogue_path: Path, output_path: Path
) -> Path:
    base = _read_json(base_catalogue_path)
    e2 = _read_json(e2_catalogue_path)
    records = list(base["members"])
    for row in e2["records"]:
        family = str(row["family"])
        state = str(row["state"])
        seed = int(row["seed"])
        records.append(
            {
                "identity": f"{family}|seed_{seed}|{state}",
                "family": family,
                "seed": seed,
                "state": state,
                "fold": str(row["fold"]),
                "prediction": str(row["prediction"]),
                "prediction_sha256": str(row["prediction_sha256"]),
                "horizon_coverage": list(row["horizon_coverage"]),
                "tags": list(row["tags"]),
                "tier": "primary",
                "source_run": str(row["source_run"]),
                "source_run_manifest_sha256": str(row["source_run_manifest_sha256"]),
                "official_validation_accessed": False,
                "test_accessed": False,
            }
        )
    provisional = {
        "schema": "EXPERIMENT44_COMBINED_PROVISIONAL_ROSTER_V1",
        "created_at": _now(),
        "references": dict(base["references"]),
        "members": records,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    provisional_path = output_path.with_suffix(".provisional.json")
    _atomic_json(provisional_path, provisional)
    return freeze_catalogue(provisional_path=provisional_path, output_path=output_path)


def _observation(
    reference: EvaluationObservations, prediction_path: Path
) -> EvaluationObservations:
    predictions = _load_prediction(prediction_path)
    if predictions.shape != reference.targets.shape:
        raise ValueError(f"Prediction archive is misaligned: {prediction_path}")
    return replace(reference, predictions=predictions)


def _bootstrap(values: np.ndarray, seed: int) -> dict[str, object]:
    return {
        str(block): {
            key: np.asarray(result).tolist()
            for key, result in moving_block_bootstrap(
                values,
                replications=BOOTSTRAP_REPLICATIONS,
                block_length=block,
                seed=seed + block,
            ).items()
        }
        for block in BOOTSTRAP_BLOCKS
    }


def _daily_comparison(
    candidate: EvaluationObservations, comparator: EvaluationObservations, seed: int
) -> dict[str, object]:
    assert_observations_aligned(comparator, candidate)
    candidate_ic = sample_level_spearman_ic(
        candidate.predictions, candidate.targets, candidate.label_mask
    )
    comparator_ic = sample_level_spearman_ic(
        comparator.predictions, comparator.targets, comparator.label_mask
    )
    dates, candidate_daily = per_date_primary_ic(candidate_ic, candidate.date_idx)
    other_dates, comparator_daily = per_date_primary_ic(
        comparator_ic, comparator.date_idx
    )
    if not np.array_equal(dates, other_dates):
        raise ValueError("Candidate and comparator dates differ")
    daily_delta = candidate_daily - comparator_daily
    _, candidate_horizon = daily_horizon_ic(candidate_ic, candidate.date_idx)
    _, comparator_horizon = daily_horizon_ic(comparator_ic, comparator.date_idx)
    return {
        "candidate_ic": primary_validation_score(
            candidate.predictions,
            candidate.targets,
            candidate.label_mask,
            candidate.date_idx,
        ),
        "comparator_ic": primary_validation_score(
            comparator.predictions,
            comparator.targets,
            comparator.label_mask,
            comparator.date_idx,
        ),
        "candidate_minus_comparator_ic": finite_mean(daily_delta),
        "per_horizon_delta": {
            str(minutes): finite_mean(
                candidate_horizon[:, index] - comparator_horizon[:, index]
            )
            for index, minutes in enumerate(HORIZONS)
        },
        "date_count": int(dates.size),
        "bootstrap": _bootstrap(daily_delta, seed),
    }


class MemberPool:
    def __init__(self, catalogue: Mapping[str, object]) -> None:
        self.records = list(catalogue["members"])
        self.references = {
            fold: _load_reference(Path(str(path)))
            for fold, path in dict(catalogue["references"]).items()
        }
        self.by_identity: dict[str, dict[str, Mapping[str, object]]] = {}
        for record in self.records:
            identity = str(record["identity"])
            fold = str(record["fold"])
            self.by_identity.setdefault(identity, {})[fold] = record
        self._prediction_cache: dict[tuple[str, str], np.ndarray] = {}

    def identities(self, required_folds: Sequence[str] = FOLDS) -> list[str]:
        required = set(required_folds)
        return sorted(
            identity
            for identity, records in self.by_identity.items()
            if required.issubset(records)
        )

    def record(self, identity: str, fold: str) -> Mapping[str, object]:
        return self.by_identity[identity][fold]

    def prediction(self, identity: str, fold: str) -> np.ndarray:
        key = (identity, fold)
        if key not in self._prediction_cache:
            record = self.record(identity, fold)
            path = Path(str(record["prediction"]))
            if _sha256(path) != record["prediction_sha256"]:
                raise ValueError(f"Frozen prediction hash differs: {path}")
            self._prediction_cache[key] = _load_prediction(path)
        return self._prediction_cache[key]

    def observations(self, identity: str, fold: str) -> EvaluationObservations:
        return _observation(
            self.references[fold], Path(str(self.record(identity, fold)["prediction"]))
        )

    def coverage(self, identity: str, fold: str) -> tuple[int, ...]:
        return tuple(
            int(value) for value in self.record(identity, fold)["horizon_coverage"]
        )

    def tagged(self, tag: str, required_folds: Sequence[str] = FOLDS) -> list[str]:
        return [
            identity
            for identity in self.identities(required_folds)
            if tag in self.record(identity, required_folds[0])["tags"]
        ]


def _combine(
    pool: MemberPool,
    fold: str,
    identities: Sequence[str],
    weights: Sequence[float] | None = None,
    reduction: str = "mean",
) -> EvaluationObservations:
    reference = pool.references[fold]
    predictions = combine_rank_predictions(
        [pool.prediction(identity, fold) for identity in identities],
        reference.label_mask,
        weights=weights,
        horizon_coverage=[pool.coverage(identity, fold) for identity in identities],
        reduction=reduction,
    )
    return replace(reference, predictions=predictions)


def _score(
    pool: MemberPool,
    fold: str,
    identities: Sequence[str],
    weights: Sequence[float] | None = None,
    reduction: str = "mean",
) -> float:
    observations = _combine(pool, fold, identities, weights, reduction)
    return primary_validation_score(
        observations.predictions,
        observations.targets,
        observations.label_mask,
        observations.date_idx,
    )


def _gate(fold_reports: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    deltas = {
        fold: float(report["candidate_minus_comparator_ic"])
        for fold, report in fold_reports.items()
    }
    mean = float(np.mean(tuple(deltas.values())))
    passed = mean >= SUPERIORITY_MEAN and all(value >= 0.0 for value in deltas.values())
    return {
        "gate": "complexity-adding superiority: mean delta >= +0.001; every held-out fold >= 0; paired block-5/10 intervals reported",
        "fold_deltas": deltas,
        "mean_delta": mean,
        "point_estimate_requirements_passed": passed,
        "bootstrap_support_reported": all(
            set(report["bootstrap"]) == {"5", "10"} for report in fold_reports.values()
        ),
        "passed": passed,
    }


def _greedy(
    pool: MemberPool, comparator: Sequence[str], candidates: Sequence[str], label: str
) -> dict[str, object]:
    heldout_reports = {}
    paths = {}
    comparator_by_fold = {fold: _combine(pool, fold, comparator) for fold in FOLDS}
    for heldout in FOLDS:
        selection_folds = [fold for fold in FOLDS if fold != heldout]
        chosen = list(comparator)
        current = float(
            np.mean([_score(pool, fold, chosen) for fold in selection_folds])
        )
        steps = []
        for _ in range(MAX_GREEDY_ADDITIONS):
            scored = []
            for identity in candidates:
                candidate_score = float(
                    np.mean(
                        [
                            _score(pool, fold, [*chosen, identity])
                            for fold in selection_folds
                        ]
                    )
                )
                scored.append((candidate_score, identity))
            best_score, best_identity = min(
                scored, key=lambda value: (-value[0], value[1])
            )
            marginal = best_score - current
            if marginal <= 0.0:
                break
            chosen.append(best_identity)
            steps.append(
                {
                    "addition": best_identity,
                    "selection_mean_ic": best_score,
                    "marginal_ic": marginal,
                    "count_after": Counter(chosen)[best_identity],
                }
            )
            current = best_score
        candidate_observations = _combine(pool, heldout, chosen)
        heldout_reports[heldout] = _daily_comparison(
            candidate_observations,
            comparator_by_fold[heldout],
            _derived_seed(label, heldout),
        )
        paths[heldout] = {
            "selection_folds": selection_folds,
            "heldout_fold": heldout,
            "start_members": list(comparator),
            "steps": steps,
            "heldout_members": chosen,
            "stopped_after_additions": len(steps),
        }
    return {
        "label": label,
        "paths": paths,
        "heldout": heldout_reports,
        "gate": _gate(heldout_reports),
    }


def _fixed_grid(
    pool: MemberPool,
    comparator: Sequence[str],
    sets: Mapping[str, Sequence[str]],
    label: str,
) -> list[dict[str, object]]:
    comparator_observations = {fold: _combine(pool, fold, comparator) for fold in FOLDS}
    rows = []
    for set_name, members in sets.items():
        if not members:
            rows.append({"set": set_name, "status": "empty_predeclared_set"})
            continue
        for weight in WEIGHT_GRID:
            identities = [*comparator, *members]
            weights = [weight / len(comparator)] * len(comparator) + [
                (1.0 - weight) / len(members)
            ] * len(members)
            reports = {
                fold: _daily_comparison(
                    _combine(pool, fold, identities, weights),
                    comparator_observations[fold],
                    _derived_seed(label, set_name, weight, fold),
                )
                for fold in FOLDS
            }
            rows.append(
                {
                    "set": set_name,
                    "comparator_weight": weight,
                    "members": identities,
                    "weights": weights,
                    "folds": reports,
                    "gate": _gate(reports),
                }
            )
    return rows


def _standalone_and_diversity(
    pool: MemberPool, identities: Sequence[str]
) -> dict[str, object]:
    result = {}
    for fold in FOLDS:
        reference = pool.references[fold]
        scores = {}
        for identity in identities:
            mask = reference.label_mask.copy()
            coverage = pool.coverage(identity, fold)
            mask[..., [index for index in range(3) if index not in coverage]] = False
            scores[identity] = primary_validation_score(
                pool.prediction(identity, fold),
                reference.targets,
                mask,
                reference.date_idx,
            )
        correlations = []
        for left_index, left in enumerate(identities):
            for right in identities[left_index + 1 :]:
                shared = sorted(
                    set(pool.coverage(left, fold)).intersection(
                        pool.coverage(right, fold)
                    )
                )
                if not shared:
                    continue
                mask = reference.label_mask.copy()
                mask[..., [index for index in range(3) if index not in shared]] = False
                correlations.append(
                    {
                        "left": left,
                        "right": right,
                        "shared_horizon_indices": shared,
                        "prediction_spearman": primary_validation_score(
                            pool.prediction(left, fold),
                            pool.prediction(right, fold),
                            mask,
                            reference.date_idx,
                        ),
                    }
                )
        result[fold] = {"standalone_ic": scores, "prediction_spearman": correlations}
    return result


def _supplementary_ab(
    pool: MemberPool, comparator: Sequence[str], candidates: Sequence[str]
) -> dict[str, object]:
    rows = {}
    for selection_fold, evaluation_fold in (("fold_a", "fold_b"), ("fold_b", "fold_a")):
        chosen = list(comparator)
        current = _score(pool, selection_fold, chosen)
        steps = []
        for _ in range(MAX_GREEDY_ADDITIONS):
            eligible = [
                identity
                for identity in candidates
                if selection_fold in pool.by_identity[identity]
                and evaluation_fold in pool.by_identity[identity]
            ]
            scored = [
                (_score(pool, selection_fold, [*chosen, identity]), identity)
                for identity in eligible
            ]
            best_score, best_identity = min(
                scored, key=lambda value: (-value[0], value[1])
            )
            marginal = best_score - current
            if marginal <= 0:
                break
            chosen.append(best_identity)
            steps.append(
                {
                    "addition": best_identity,
                    "selection_ic": best_score,
                    "marginal_ic": marginal,
                }
            )
            current = best_score
        rows[f"{selection_fold}_to_{evaluation_fold}"] = {
            "selection_fold": selection_fold,
            "evaluation_fold": evaluation_fold,
            "steps": steps,
            "evaluation": _daily_comparison(
                _combine(pool, evaluation_fold, chosen),
                _combine(pool, evaluation_fold, comparator),
                _derived_seed("supplementary", selection_fold, evaluation_fold),
            ),
            "gate_eligible": False,
        }
    return {"rotations": rows, "supplementary_only": True, "can_open_gate": False}


def _hygiene(
    pool: MemberPool, comparator: Sequence[str], primary_roster: Sequence[str]
) -> list[dict[str, object]]:
    identities = sorted(set((*comparator, *primary_roster)))
    comparator_observations = {fold: _combine(pool, fold, comparator) for fold in FOLDS}
    rows = []
    for reduction in ("median", "trimmed_mean"):
        reports = {
            fold: _daily_comparison(
                _combine(pool, fold, identities, reduction=reduction),
                comparator_observations[fold],
                _derived_seed("hygiene", reduction, fold),
            )
            for fold in FOLDS
        }
        rows.append(
            {
                "rule": reduction,
                "members": identities,
                "folds": reports,
                "gate": _gate(reports),
            }
        )
    reports = {}
    weights_by_fold = {}
    for heldout in FOLDS:
        selection_folds = [fold for fold in FOLDS if fold != heldout]
        skills = np.asarray(
            [
                float(
                    np.mean(
                        [_score(pool, fold, [identity]) for fold in selection_folds]
                    )
                )
                for identity in identities
            ]
        )
        center = float(np.mean(skills))
        weights = np.maximum(0.0, center + 0.5 * (skills - center))
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        weights /= weights.sum()
        weights_by_fold[heldout] = {
            identity: float(weight)
            for identity, weight in zip(identities, weights, strict=True)
        }
        reports[heldout] = _daily_comparison(
            _combine(pool, heldout, identities, weights.tolist()),
            comparator_observations[heldout],
            _derived_seed("hygiene", "shrunken_skill", heldout),
        )
    rows.append(
        {
            "rule": "honest_shrunken_skill",
            "weights_selected_on_other_two_folds": weights_by_fold,
            "folds": reports,
            "gate": _gate(reports),
        }
    )
    return rows


def analyze_program(*, program_root: Path, catalogue_path: Path) -> Path:
    catalogue = _read_json(catalogue_path)
    freeze_manifest = _read_json(catalogue_path.with_suffix(".manifest.json"))
    if _sha256(catalogue_path) != freeze_manifest["catalogue_sha256"]:
        raise ValueError("Frozen roster catalogue hash differs")
    if (
        catalogue.get("official_validation_accessed") is not False
        or catalogue.get("test_accessed") is not False
    ):
        raise ValueError("Frozen roster does not seal official/test data")
    pool = MemberPool(catalogue)
    comparator = pool.tagged("comparator_store_v2")
    if len(comparator) != 3:
        raise ValueError(
            f"Comparator must contain three store-v2 members, found {len(comparator)}"
        )
    primary = [identity for identity in pool.identities() if identity not in comparator]
    archive_primary = [
        identity
        for identity in primary
        if "e2" not in pool.record(identity, "fold_c")["tags"]
    ]
    e2_primary = [
        identity
        for identity in primary
        if "e2" in pool.record(identity, "fold_c")["tags"]
    ]
    diversity = _standalone_and_diversity(pool, [*comparator, *primary])
    e1_greedy = _greedy(pool, comparator, archive_primary, "e1_archive_primary")
    secondary = [
        identity
        for identity in pool.identities(("fold_a", "fold_b"))
        if identity not in comparator and identity not in archive_primary
    ]
    supplementary = _supplementary_ab(pool, comparator, [*archive_primary, *secondary])
    e1_sets = {
        "residual_ema3": pool.tagged("residual_ema", FOLDS),
        "residual_ema3_plus_options_oi": sorted(
            set(pool.tagged("residual_ema", FOLDS) + pool.tagged("options_oi", FOLDS))
        ),
        "best_five_adapter_ema": pool.tagged("best_five_adapter_ema", FOLDS),
        "parent58_3": pool.tagged("parent58", FOLDS),
        "full_primary_roster": archive_primary,
    }
    e1_grid = _fixed_grid(pool, comparator, e1_sets, "e1_fixed")
    hygiene = _hygiene(pool, comparator, archive_primary)
    e2_greedy = _greedy(pool, comparator, e2_primary, "e2_only")
    full_greedy = _greedy(pool, comparator, primary, "e2_plus_archive")
    e2_sets = {
        "e2a_9": pool.tagged("e2a_bagged_dates", FOLDS),
        "e2b_9": pool.tagged("e2b_feature_subspace", FOLDS),
        "e2c_27": sorted(
            set().union(
                *(set(pool.tagged(family, FOLDS)) for family in E2_HORIZON_FAMILIES)
            )
        ),
        "all_e2": e2_primary,
    }
    e2_grid = _fixed_grid(pool, comparator, e2_sets, "e2_fixed")
    gate_rows = [
        e1_greedy,
        e2_greedy,
        full_greedy,
        *hygiene,
        *(row for row in e1_grid if "gate" in row),
        *(row for row in e2_grid if "gate" in row),
    ]
    passing = [row for row in gate_rows if row["gate"]["passed"]]
    winner = None
    if passing:
        winner = min(
            passing,
            key=lambda row: (
                -float(row["gate"]["mean_delta"]),
                0 if "paths" in row else 1,
                str(row.get("label", row.get("set"))),
            ),
        )
    result = {
        "schema": "EXPERIMENT44_ANALYSIS_V1",
        "created_at": _now(),
        "catalogue": str(catalogue_path.resolve()),
        "comparator": comparator,
        "primary_archive_member_count": len(archive_primary),
        "e2_member_count": len(e2_primary),
        "diversity": diversity,
        "e1": {
            "greedy": e1_greedy,
            "supplementary_ab": supplementary,
            "fixed_grid": e1_grid,
            "hygiene": hygiene,
        },
        "e2": {
            "greedy_e2": e2_greedy,
            "greedy_full": full_greedy,
            "fixed_grid": e2_grid,
        },
        "named_read_arm": winner,
        "named_read_arm_count": 0 if winner is None else 1,
        "null_conclusion": None
        if winner is not None
        else "uniform weighting was not the binding constraint at the current pool",
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    path = program_root / "analysis" / "experiment44_analysis.json"
    _atomic_json(path, result)
    return path


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen Experiment-44 ensemble-science program"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-design")
    freeze.add_argument("--output-dir", type=Path, required=True)
    train = subparsers.add_parser("run-e2")
    train.add_argument("--program-root", type=Path, required=True)
    train.add_argument("--design", type=Path, required=True)
    train.add_argument("--parallel-processes", type=int, default=2)
    states = subparsers.add_parser("materialize-e2")
    states.add_argument("--program-root", type=Path, required=True)
    states.add_argument("--design", type=Path, required=True)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--program-root", type=Path, required=True)
    analyze.add_argument("--catalogue", type=Path, required=True)
    catalogue = subparsers.add_parser("freeze-catalogue")
    catalogue.add_argument("--provisional", type=Path, required=True)
    catalogue.add_argument("--output", type=Path, required=True)
    merge = subparsers.add_parser("merge-e2-catalogue")
    merge.add_argument("--base", type=Path, required=True)
    merge.add_argument("--e2", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    a1 = subparsers.add_parser("amendment-a1")
    a1.add_argument("--output-dir", type=Path, required=True)
    a1.add_argument("--source-official-root", type=Path, required=True)
    a1.add_argument("--selection-file", type=Path, required=True)
    a1.add_argument("--parallel-processes", type=int, default=2)
    return parser.parse_args(arguments)


def main() -> None:
    args = parse_args()
    if args.command == "freeze-design":
        print(freeze_design(output_dir=args.output_dir, store=resolve_feature_store()))
    elif args.command == "run-e2":
        print(
            run_e2(
                program_root=args.program_root,
                design_path=args.design,
                parallel_processes=args.parallel_processes,
            )
        )
    elif args.command == "materialize-e2":
        print(
            materialize_e2_states(
                program_root=args.program_root, design_path=args.design
            )
        )
    elif args.command == "freeze-catalogue":
        print(
            freeze_catalogue(provisional_path=args.provisional, output_path=args.output)
        )
    elif args.command == "merge-e2-catalogue":
        print(
            merge_e2_catalogue(
                base_catalogue_path=args.base,
                e2_catalogue_path=args.e2,
                output_path=args.output,
            )
        )
    elif args.command == "amendment-a1":
        print(
            run_amendment_a1(
                output_dir=args.output_dir,
                source_official_root=args.source_official_root,
                selection_file=args.selection_file,
                parallel_processes=args.parallel_processes,
            )
        )
    else:
        print(
            analyze_program(
                program_root=args.program_root, catalogue_path=args.catalogue
            )
        )


if __name__ == "__main__":
    main()
