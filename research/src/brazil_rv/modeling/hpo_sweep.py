from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from .analyze import compare_observation_ensembles, load_run_observations
from .contract import (
    ALLOWED_SEEDS,
    DYNAMIC_CHANNEL_COUNT,
    TCN_ARCHITECTURE,
    TrainingSpecification,
)
from .data import feature_store_identity
from .metrics import moving_block_bootstrap
from .provenance import repository_commit
from .three_fold_sidecar_screen import crossfit_patience_observations
from .train import run_training

STORE_V2_DYNAMIC_ZERO = (9, 11, 14, 22, 24, 25)
STORE_V2_SLOW_ZERO = (
    1,
    2,
    3,
    12,
    13,
    14,
    15,
    16,
    18,
    20,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
)
EXPECTED_STORE_IDENTITY_SHA256 = (
    "c90103b0f99e0017dc1303284a1ab61eca99106094227f5823ba718756d28a6b"
)
EXPECTED_PARENT_REPLAY_SHA256 = (
    "15d9f54a3a90900e643472b1b301578fc343ddfef566518cc0b5985963d39f2b"
)
STAGE1_FOLDS = ("fold_b", "fold_c")
STAGE2_FOLDS = ("fold_c", "fold_a", "fold_b")
PRIMARY_READOUT = "bidirectional_odd_even_crossfit_patience3_raw"
SECONDARY_READOUT = "final_ema_0995"
MAX_PARALLEL = 2


@dataclass(frozen=True)
class SweepConfig:
    config_id: str
    track: str | None
    specification: TrainingSpecification

    @property
    def receptive_field(self) -> int:
        architecture = self.specification.architecture
        return 1 + (architecture.kernel_size - 1) * sum(architecture.dilations)

    @property
    def retained_component_count(self) -> int:
        architecture = self.specification.architecture
        return (
            architecture.residual_blocks
            + int(architecture.dropout > 0)
            + int(self.specification.weight_decay > 0)
        )


def sweep_configurations() -> tuple[SweepConfig, ...]:
    incumbent = TrainingSpecification()

    def config(
        config_id: str,
        track: str | None,
        *,
        architecture=None,
        patch_minutes: int = incumbent.patch_minutes,
        learning_rate: float = incumbent.learning_rate,
        weight_decay: float = incumbent.weight_decay,
        temperature: float = incumbent.soft_rank_temperature,
        sam_rho: float = incumbent.sam_rho,
    ) -> SweepConfig:
        return SweepConfig(
            config_id,
            track,
            TrainingSpecification(
                architecture=architecture or incumbent.architecture,
                patch_minutes=patch_minutes,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                soft_rank_temperature=temperature,
                sam_rho=sam_rho,
            ),
        )

    dropout_zero = replace(TCN_ARCHITECTURE, dropout=0.0)
    return (
        config("C0", None),
        config("S1", "SIMP", architecture=dropout_zero, weight_decay=0.0),
        config(
            "S2",
            "SIMP",
            architecture=dropout_zero,
            weight_decay=0.0,
            sam_rho=0.25,
        ),
        config(
            "S3",
            "SIMP",
            architecture=dropout_zero,
            weight_decay=0.0,
            sam_rho=0.50,
        ),
        config("P1", "SIMP", architecture=dropout_zero),
        config("P2", "SIMP", weight_decay=0.0, sam_rho=0.25),
        config(
            "R1",
            "SIMP",
            architecture=replace(
                TCN_ARCHITECTURE,
                residual_blocks=4,
                dilations=(1, 4, 16, 32),
            ),
        ),
        config(
            "R2",
            "VAL",
            architecture=replace(
                TCN_ARCHITECTURE,
                residual_blocks=8,
                dilations=(1, 2, 4, 8, 16, 32, 1, 1),
            ),
        ),
        config(
            "R3",
            "VAL",
            architecture=replace(
                TCN_ARCHITECTURE,
                kernel_size=7,
                dilations=(1, 1, 2, 2, 4, 4),
            ),
        ),
        config(
            "R4",
            "VAL",
            architecture=replace(
                TCN_ARCHITECTURE,
                patch_input_width=10 * DYNAMIC_CHANNEL_COUNT,
            ),
            patch_minutes=10,
        ),
        config("R5", "VAL", learning_rate=5e-4),
        config("R6", "VAL", learning_rate=2e-4),
        config("R7", "VAL", temperature=1.0),
        config("R8", "VAL", learning_rate=5e-4, temperature=1.0, sam_rho=0.25),
        config(
            "R9",
            "VAL",
            architecture=replace(
                TCN_ARCHITECTURE,
                residual_blocks=4,
                kernel_size=7,
                dilations=(1, 2, 4, 8),
            ),
            learning_rate=5e-4,
        ),
        config(
            "R10",
            "VAL",
            architecture=replace(
                TCN_ARCHITECTURE,
                patch_input_width=10 * DYNAMIC_CHANNEL_COUNT,
                residual_blocks=4,
                dilations=(1, 4, 8, 16),
            ),
            patch_minutes=10,
            weight_decay=0.02,
            temperature=1.0,
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _config_record(config: SweepConfig) -> dict[str, object]:
    return {
        "config_id": config.config_id,
        "track": config.track,
        "specification": asdict(config.specification),
        "receptive_field": config.receptive_field,
        "retained_component_count": config.retained_component_count,
    }


def _jsonable(value: object) -> object:
    return json.loads(json.dumps(value))


def _validate_sources(
    store: Path, parent_root: Path, parent_replay_report: Path
) -> dict[str, object]:
    identity = feature_store_identity(store)
    if identity["metadata_sha256"] != EXPECTED_STORE_IDENTITY_SHA256:
        raise ValueError("Feature store identity is not the frozen store-v2 source")
    if _sha256(parent_replay_report) != EXPECTED_PARENT_REPLAY_SHA256:
        raise ValueError("Parent replay report differs from the frozen source")
    replay = _read_json(parent_replay_report)
    replays = replay.get("comparison_metadata", {}).get(
        "parent_patience_replays_by_fold"
    )
    expected_seed_keys = {f"seed_{seed}" for seed in ALLOWED_SEEDS}
    if not isinstance(replays, dict) or set(replays) != set(STAGE2_FOLDS):
        raise ValueError("Parent replay report does not cover folds C/A/B")
    if any(set(replays[fold]) != expected_seed_keys for fold in STAGE2_FOLDS):
        raise ValueError("Parent replay report does not cover all three seeds")
    manifests = {}
    for fold in STAGE2_FOLDS:
        for seed in ALLOWED_SEEDS:
            path = parent_root / fold / f"seed_{seed}" / "run_manifest.json"
            value = _read_json(path)
            if (
                value.get("status") != "completed"
                or value.get("seed") != seed
                or value.get("split", {}).get("training") != fold
                or value.get("split", {}).get("test_accessed") is not False
                or value.get("feature_store_identity") != identity
                or value.get("equity_input_zeroing", {}).get("dynamic_channels")
                != list(STORE_V2_DYNAMIC_ZERO)
                or value.get("equity_input_zeroing", {}).get("slow_fields")
                != list(STORE_V2_SLOW_ZERO)
            ):
                raise ValueError(
                    f"Invalid store-v2 parent manifest: {fold}/seed_{seed}"
                )
            manifests[f"{fold}/seed_{seed}"] = {
                "path": str(path),
                "sha256": _sha256(path),
            }
    return {
        "feature_store": identity,
        "parent_root": str(parent_root),
        "parent_manifests": manifests,
        "parent_replay_report": str(parent_replay_report),
        "parent_replay_report_sha256": _sha256(parent_replay_report),
    }


def _run_path(root: Path, stage: str, config_id: str, fold: str, seed: int) -> Path:
    return root / stage / "runs" / config_id / fold / f"seed_{seed}"


def _run_job(store: Path, run_dir: Path, config_id: str, fold: str, seed: int) -> str:
    configs = {item.config_id: item for item in sweep_configurations()}
    run_training(
        store=store,
        seed=seed,
        selection_window=fold,
        run_dir=run_dir,
        zero_dynamic_channels=STORE_V2_DYNAMIC_ZERO,
        zero_slow_fields=STORE_V2_SLOW_ZERO,
        training_specification=configs[config_id].specification,
    )
    return str(run_dir)


def _completed_run(path: Path, config: SweepConfig, fold: str, seed: int) -> bool:
    manifest_path = path / "run_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = _read_json(manifest_path)
    return (
        manifest.get("status") == "completed"
        and manifest.get("seed") == seed
        and manifest.get("split", {}).get("training") == fold
        and manifest.get("split", {}).get("test_accessed") is False
        and manifest.get("training")
        == manifest.get("run_provenance", {}).get("training")
        and manifest.get("model", {}).get("architecture")
        == _jsonable(asdict(config.specification.architecture))
        and manifest.get("training", {}).get("learning_rate")
        == config.specification.learning_rate
        and manifest.get("training", {}).get("adamw_weight_decay")
        == config.specification.weight_decay
        and manifest.get("training", {}).get("objective", {}).get("temperature")
        == config.specification.soft_rank_temperature
        and manifest.get("sam", {}).get("rho") == config.specification.sam_rho
    )


def _execute_jobs(
    jobs: Sequence[tuple[Path, Path, str, str, int]],
    configs: Mapping[str, SweepConfig],
    parallel_processes: int,
) -> None:
    pending = []
    for job in jobs:
        _, path, config_id, fold, seed = job
        if _completed_run(path, configs[config_id], fold, seed):
            continue
        if path.exists():
            raise RuntimeError(f"Incomplete run requires reviewed repair: {path}")
        pending.append(job)
    if parallel_processes == 1:
        for job in pending:
            print(_run_job(*job), flush=True)
        return
    with ProcessPoolExecutor(
        max_workers=parallel_processes, mp_context=mp.get_context("spawn")
    ) as executor:
        futures = [executor.submit(_run_job, *job) for job in pending]
        for future in as_completed(futures):
            print(future.result(), flush=True)


def _primary(path: Path, frozen_replays=None):
    return crossfit_patience_observations(path, frozen_replays)


def _comparison(
    candidate: Mapping[str, object],
    parent: Mapping[str, object],
    *,
    output_dir: Path,
    candidate_rule: str,
    parent_rule: str,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    analysis_path = output_dir / "analysis.json"
    if not analysis_path.is_file():
        compare_observation_ensembles(
            candidate,
            parent,
            candidate_rule=candidate_rule,
            parent_rule=parent_rule,
            output_dir=output_dir,
            comparison_metadata=metadata,
        )
    value = _read_json(analysis_path)
    return {
        "delta": value["candidate_minus_parent_primary_ic"],
        "analysis": str(analysis_path),
        "daily_delta": str(output_dir / "daily_delta.parquet"),
    }


def _stage1_analysis(root: Path) -> dict[str, object]:
    configs = sweep_configurations()
    control = configs[0]
    rows = []
    for config in configs[1:]:
        fold_rows = {}
        for fold in STAGE1_FOLDS:
            candidate_path = _run_path(root, "stage1", config.config_id, fold, 29)
            control_path = _run_path(root, "stage1", control.config_id, fold, 29)
            candidate_primary, candidate_replay = _primary(candidate_path)
            control_primary, control_replay = _primary(control_path)
            base = root / "stage1" / "analysis" / config.config_id / fold
            primary = _comparison(
                {"seed_29": candidate_primary},
                {"seed_29": control_primary},
                output_dir=base / "primary",
                candidate_rule=PRIMARY_READOUT,
                parent_rule=f"C0_{PRIMARY_READOUT}",
                metadata={
                    "stage": 1,
                    "fold": fold,
                    "matched_seed": 29,
                    "candidate_patience_replay": candidate_replay,
                    "control_patience_replay": control_replay,
                    "official_validation_accessed": False,
                    "test_accessed": False,
                },
            )
            secondary = _comparison(
                {"seed_29": load_run_observations(candidate_path, SECONDARY_READOUT)},
                {"seed_29": load_run_observations(control_path, SECONDARY_READOUT)},
                output_dir=base / "secondary_ema_0995",
                candidate_rule=SECONDARY_READOUT,
                parent_rule=f"C0_{SECONDARY_READOUT}",
                metadata={
                    "stage": 1,
                    "fold": fold,
                    "matched_seed": 29,
                    "decision_eligible": False,
                    "official_validation_accessed": False,
                    "test_accessed": False,
                },
            )
            fold_rows[fold] = {"primary": primary, "secondary": secondary}
            rows.append(
                {
                    "config_id": config.config_id,
                    "track": config.track,
                    "fold": fold,
                    "primary_delta": primary["delta"],
                    "secondary_ema_0995_delta": secondary["delta"],
                }
            )
        mean = float(
            np.mean([fold_rows[fold]["primary"]["delta"] for fold in STAGE1_FOLDS])
        )
        threshold = 0.0 if config.track == "VAL" else -0.0005
        qualifies = all(
            float(fold_rows[fold]["primary"]["delta"]) >= threshold
            for fold in STAGE1_FOLDS
        )
        for row in rows[-2:]:
            row["two_fold_mean"] = mean
            row["qualifies"] = qualifies

    qualifiers = {
        track: sorted(
            {
                row["config_id"]: float(row["two_fold_mean"])
                for row in rows
                if row["track"] == track and row["qualifies"]
            }.items(),
            key=lambda item: (-item[1], item[0]),
        )
        for track in ("VAL", "SIMP")
    }
    advanced = []
    for track in ("VAL", "SIMP"):
        if qualifiers[track]:
            advanced.append(qualifiers[track][0][0])
    remaining = sorted(
        [
            item
            for values in qualifiers.values()
            for item in values
            if item[0] not in advanced
        ],
        key=lambda item: (-item[1], item[0]),
    )
    if remaining:
        advanced.append(remaining[0][0])
    table_path = root / "stage1" / "stage1_screen.parquet"
    pl.DataFrame(rows).write_parquet(table_path)
    result = {
        "schema": "EXPERIMENT47_STAGE1_SCREEN_V1",
        "rows": rows,
        "table": str(table_path),
        "qualifiers": qualifiers,
        "advanced_config_ids": advanced,
        "advancement_rule": {
            "VAL": "positive delta on both folds",
            "SIMP": "delta >= -0.0005 on both folds",
            "cap": 3,
            "priority": "best VAL, best SIMP, then next-best remaining qualifier",
        },
        "multiplicity_acknowledged": True,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(root / "stage1" / "stage1_screen.json", result)
    return result


def _pooled_intervals(paths: Sequence[Path]) -> dict[str, object]:
    values = np.concatenate(
        [
            np.asarray(pl.read_parquet(path)["candidate_minus_parent_ic"])
            for path in paths
        ]
    )
    return {
        str(block): {
            key: np.asarray(value).tolist()
            for key, value in moving_block_bootstrap(
                values,
                replications=10_000,
                block_length=block,
                seed=47 + block,
            ).items()
        }
        for block in (5, 10)
    }


def _stage2_analysis(
    root: Path,
    parent_root: Path,
    parent_replay_report: Path,
    advanced: Sequence[str],
) -> dict[str, object]:
    configs = {item.config_id: item for item in sweep_configurations()}
    parent_replays = _read_json(parent_replay_report)["comparison_metadata"][
        "parent_patience_replays_by_fold"
    ]
    rows = []
    for config_id in advanced:
        config = configs[config_id]
        folds = {}
        daily_paths = []
        for fold in STAGE2_FOLDS:
            candidate_primary = {}
            parent_primary = {}
            candidate_replays = {}
            frozen_parent_replays = {}
            candidate_ema = {}
            parent_ema = {}
            for seed in ALLOWED_SEEDS:
                key = f"seed_{seed}"
                candidate_path = _run_path(root, "stage2", config_id, fold, seed)
                parent_path = parent_root / fold / key
                candidate_primary[key], candidate_replays[key] = _primary(
                    candidate_path
                )
                parent_primary[key], frozen_parent_replays[key] = _primary(
                    parent_path, parent_replays[fold][key]
                )
                candidate_ema[key] = load_run_observations(
                    candidate_path, SECONDARY_READOUT
                )
                parent_ema[key] = load_run_observations(parent_path, SECONDARY_READOUT)
            base = root / "stage2" / "analysis" / config_id / fold
            primary = _comparison(
                candidate_primary,
                parent_primary,
                output_dir=base / "primary",
                candidate_rule=PRIMARY_READOUT,
                parent_rule="store_v2_parent3_crossfit_patience3_raw",
                metadata={
                    "stage": 2,
                    "fold": fold,
                    "candidate_patience_replays": candidate_replays,
                    "parent_patience_replays": frozen_parent_replays,
                    "official_validation_accessed": False,
                    "test_accessed": False,
                },
            )
            secondary = _comparison(
                candidate_ema,
                parent_ema,
                output_dir=base / "secondary_ema_0995",
                candidate_rule=SECONDARY_READOUT,
                parent_rule=f"store_v2_parent3_{SECONDARY_READOUT}",
                metadata={
                    "stage": 2,
                    "fold": fold,
                    "decision_eligible": False,
                    "official_validation_accessed": False,
                    "test_accessed": False,
                },
            )
            folds[fold] = {"primary": primary, "secondary": secondary}
            daily_paths.append(Path(primary["daily_delta"]))
        deltas = {fold: float(folds[fold]["primary"]["delta"]) for fold in STAGE2_FOLDS}
        mean = float(np.mean(tuple(deltas.values())))
        intervals = _pooled_intervals(daily_paths)
        intervals_support = all(
            float(intervals[str(block)]["lower_95"][0]) > 0 for block in (5, 10)
        )
        if config.track == "VAL":
            passed = (
                mean >= 0.001
                and all(value >= 0 for value in deltas.values())
                and intervals_support
            )
        else:
            passed = mean >= 0 and all(value >= -0.0005 for value in deltas.values())
        rows.append(
            {
                "config_id": config_id,
                "track": config.track,
                "fold_deltas": deltas,
                "three_fold_mean": mean,
                "pooled_daily_delta_bootstrap": intervals,
                "intervals_support_superiority": intervals_support,
                "retained_component_count": config.retained_component_count,
                "gate_passed": passed,
                "folds": folds,
            }
        )
    simp_passers = [
        row for row in rows if row["track"] == "SIMP" and row["gate_passed"]
    ]
    simp_winner = None
    if simp_passers:
        simp_winner = min(
            simp_passers,
            key=lambda row: (
                row["retained_component_count"],
                -row["three_fold_mean"],
                row["config_id"],
            ),
        )["config_id"]
    result = {
        "schema": "EXPERIMENT47_STAGE2_CONFIRMATION_V1",
        "rows": rows,
        "future_official_read_arms": [
            row["config_id"]
            for row in rows
            if row["track"] == "VAL" and row["gate_passed"]
        ],
        "future_training_recipe_specification": simp_winner,
        "gates": {
            "VAL": (
                "three-fold mean >= +0.001, every fold >= 0, and pooled block-5 "
                "and block-10 lower 95% bounds > 0"
            ),
            "SIMP": "three-fold mean >= 0 and no fold < -0.0005",
            "simplest": (
                "fewest residual blocks plus nonzero weight-decay/dropout components; "
                "ties by higher mean"
            ),
        },
        "deployment_changed": False,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(root / "stage2" / "stage2_confirmation.json", result)
    return result


def run_hpo_sweep(
    *,
    store: Path,
    parent_root: Path,
    parent_replay_report: Path,
    output_dir: Path,
    parallel_processes: int = 2,
    resume: bool = False,
) -> Path:
    if not 1 <= parallel_processes <= MAX_PARALLEL:
        raise ValueError("Sweep allows one or two training processes")
    store = store.resolve()
    parent_root = parent_root.resolve()
    parent_replay_report = parent_replay_report.resolve()
    output_dir = output_dir.resolve()
    configs = sweep_configurations()
    config_map = {item.config_id: item for item in configs}
    if len(config_map) != 16 or tuple(config_map) != tuple(
        ["C0", "S1", "S2", "S3", "P1", "P2"] + [f"R{index}" for index in range(1, 11)]
    ):
        raise RuntimeError("Frozen sweep configuration roster differs")
    if output_dir.exists() and not resume:
        raise FileExistsError(output_dir)
    if not output_dir.exists():
        sources = _validate_sources(store, parent_root, parent_replay_report)
        output_dir.mkdir(parents=True)
        design = {
            "schema": "EXPERIMENT47_FROZEN_DESIGN_V1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repository_commit": repository_commit(),
            "sources": sources,
            "configs": [_config_record(item) for item in configs],
            "stage1": {
                "seed": 29,
                "folds": list(STAGE1_FOLDS),
                "trajectory_count": 32,
            },
            "stage2": {
                "maximum_config_count": 3,
                "seeds": list(ALLOWED_SEEDS),
                "folds": list(STAGE2_FOLDS),
                "maximum_trajectory_count": 27,
            },
            "patch10_causal_resolution": (
                "left-pad only the oldest edge of each available prefix to a "
                "10-minute multiple; never consume the next interval; 35 tokens"
            ),
            "supporting_intervals_resolution": (
                "both pooled block-5 and block-10 lower 95% bounds exceed zero"
            ),
            "official_validation_accessed": False,
            "test_accessed": False,
        }
        design_path = output_dir / "frozen_design.json"
        _atomic_json(design_path, design)
        manifest = {
            "schema": "EXPERIMENT47_PROGRAM_MANIFEST_V1",
            "status": "frozen",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repository_commit": repository_commit(),
            "frozen_design": str(design_path),
            "frozen_design_sha256": _sha256(design_path),
            "official_validation_accessed": False,
            "test_accessed": False,
        }
        _atomic_json(output_dir / "program_manifest.json", manifest)
    else:
        manifest = _read_json(output_dir / "program_manifest.json")
        if (
            manifest.get("status") == "completed"
            or manifest.get("repository_commit") != repository_commit()
            or manifest.get("official_validation_accessed") is not False
            or manifest.get("test_accessed") is not False
        ):
            raise ValueError("Program root is not eligible for resume")

    stage1_jobs = [
        (
            store,
            _run_path(output_dir, "stage1", config.config_id, fold, 29),
            config.config_id,
            fold,
            29,
        )
        for config in configs
        for fold in STAGE1_FOLDS
    ]
    _atomic_json(
        output_dir / "program_manifest.json",
        {**manifest, "status": "stage1_running", "stage1_job_count": len(stage1_jobs)},
    )
    _execute_jobs(stage1_jobs, config_map, parallel_processes)
    stage1 = _stage1_analysis(output_dir)
    advanced = stage1["advanced_config_ids"]
    stage2 = None
    if advanced:
        stage2_jobs = [
            (
                store,
                _run_path(output_dir, "stage2", config_id, fold, seed),
                config_id,
                fold,
                seed,
            )
            for config_id in advanced
            for fold in STAGE2_FOLDS
            for seed in ALLOWED_SEEDS
        ]
        _atomic_json(
            output_dir / "program_manifest.json",
            {
                **manifest,
                "status": "stage2_running",
                "stage1_screen": str(output_dir / "stage1" / "stage1_screen.json"),
                "advanced_config_ids": advanced,
                "stage2_job_count": len(stage2_jobs),
            },
        )
        _execute_jobs(stage2_jobs, config_map, parallel_processes)
        stage2 = _stage2_analysis(
            output_dir, parent_root, parent_replay_report, advanced
        )
    all_runs = sorted(
        str(path.parent) for path in output_dir.rglob("run_manifest.json")
    )
    result = {
        "schema": "EXPERIMENT47_RESULT_V1",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "stage1": stage1,
        "stage2": stage2,
        "stage2_skipped": not advanced,
        "trajectory_count": len(all_runs),
        "pool_eligibility": {
            "family": "hyperparameter_jittered",
            "runs": all_runs,
            "eligible_states": [PRIMARY_READOUT, SECONDARY_READOUT],
            "pool_scored": False,
        },
        "hpo_architecture_axes_closed_permanently_for_generation": True,
        "deployment_changed": False,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    result_path = output_dir / "experiment47_result.json"
    _atomic_json(result_path, result)
    _atomic_json(
        output_dir / "program_manifest.json",
        {
            **manifest,
            "status": "completed",
            "completed_at": result["completed_at"],
            "stage1_screen": str(output_dir / "stage1" / "stage1_screen.json"),
            "stage2_confirmation": (
                None
                if stage2 is None
                else str(output_dir / "stage2" / "stage2_confirmation.json")
            ),
            "result": str(result_path),
            "trajectory_count": len(all_runs),
        },
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Experiment-47 HPO sweep")
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--parent-replay-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parallel-processes", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    print(run_hpo_sweep(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
