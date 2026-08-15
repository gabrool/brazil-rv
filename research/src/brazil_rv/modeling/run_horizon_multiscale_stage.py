from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
import torch

from .contract import (
    BASELINE_TCN_SETTINGS,
    HORIZONS,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
    TCN_READOUTS,
    architecture_for_model,
    tcn_tap_receptive_field_minutes,
)
from .data import load_sample_index, resolve_feature_store, select_sample_split
from .evaluate import load_current_neural_run
from .horizon_diagnostics import (
    BOOTSTRAP_SEED,
    build_oof_plan,
    assert_analysis_rows,
    atomic_csv,
    atomic_json,
    atomic_parquet,
    feature_store_identity,
    run_target_basis_audit,
)
from .metrics import moving_block_bootstrap
from .model import build_neural_model, count_trainable_parameters
from .stage_conflict_oof import run_gradient_audit, run_oof_residual_probes
from .stage_representation_probes import (
    run_context_inference_probes,
    run_frozen_block_probes,
)
from .train import _run_neural, parse_args as parse_training_args


STAGE_SCHEMA = "HORIZON_MULTISCALE_STAGE_V1"
TRAINING_RUN_COUNT = 16
REMOTE_COMMAND = (
    "cd /home/ubuntu/Brazil-RV/quant/b3-quant && "
    "uv run --project research python -m "
    "brazil_rv.modeling.run_horizon_multiscale_stage "
    "--output-dir /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/"
    "model_runs/horizon_multiscale_stage_20260815"
)


@dataclass(frozen=True)
class Arm:
    name: str
    readout: str = "final"
    seed: int = 29
    training_horizon: str = "all"
    context_ablation: str = "none"


ARMS = (
    Arm("final_seed29"),
    Arm("shared_multiscale_seed29", "shared_multiscale"),
    Arm("horizon_multiscale_seed29", "horizon_multiscale"),
    Arm("final_score_mlp_seed29", "final_score_mlp"),
    Arm("final_seed11", seed=11),
    Arm("final_seed47", seed=47),
    Arm("horizon_multiscale_seed11", "horizon_multiscale", 11),
    Arm("horizon_multiscale_seed47", "horizon_multiscale", 47),
    Arm("single_horizon_30", training_horizon="30"),
    Arm("single_horizon_60", training_horizon="60"),
    Arm("single_horizon_120", training_horizon="120"),
    Arm("without_wdo", context_ablation="wdo"),
    Arm("without_br_rates", context_ablation="br_rates"),
    Arm("without_us_rates", context_ablation="us_rates"),
    Arm("oof_fold_1"),
    Arm("oof_fold_2"),
)
ARM_BY_NAME = {arm.name: arm for arm in ARMS}


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the train/validation-only horizon multiscale stage"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(arguments)


def _git_commit() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _fingerprint(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


class Stage:
    def __init__(self, output_dir: Path, store: Path) -> None:
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "stage_manifest.json"
        self.commit = _git_commit()
        self.store_identity = feature_store_identity(store)
        self.logger = logging.getLogger(f"horizon-stage-{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        for handler in (
            logging.StreamHandler(),
            logging.FileHandler(self.output_dir / "stage.log", encoding="utf-8"),
        ):
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if (
                self.manifest.get("schema") != STAGE_SCHEMA
                or self.manifest.get("repository_commit") != self.commit
                or self.manifest.get("feature_store") != self.store_identity
            ):
                raise ValueError(
                    "Existing stage state has incompatible commit or feature store"
                )
        else:
            self.manifest = {
                "schema": STAGE_SCHEMA,
                "status": "running",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "repository_commit": self.commit,
                "feature_store": self.store_identity,
                "date_boundaries": {
                    "train": [str(TRAIN_START), str(TRAIN_END)],
                    "validation": [str(VALIDATION_START), str(VALIDATION_END)],
                    "test_start_not_accessed": str(TEST_START),
                },
                "training_run_count": TRAINING_RUN_COUNT,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "command": REMOTE_COMMAND,
                "steps": {},
            }
            self.write_manifest()

    def write_manifest(self) -> None:
        atomic_json(self.manifest_path, self.manifest)

    def step(
        self,
        name: str,
        config: dict[str, object],
        artifacts: tuple[Path, ...],
        action: Callable[[], None],
        validator: Callable[[], bool] | None = None,
    ) -> None:
        fingerprint = _fingerprint(
            {
                "commit": self.commit,
                "feature_store": self.store_identity,
                "config": config,
            }
        )
        existing = self.manifest["steps"].get(name)
        if existing and existing.get("status") == "completed":
            if existing.get("fingerprint") != fingerprint:
                raise ValueError(f"Completed step {name} has incompatible metadata")
            missing = [str(path) for path in artifacts if not path.exists()]
            if missing:
                raise ValueError(
                    f"Completed step {name} is missing artifacts: {missing}"
                )
            if validator is not None and not validator():
                raise ValueError(f"Completed step {name} failed artifact validation")
            self.logger.info("reuse completed step %s", name)
            return
        self.logger.info("start step %s", name)
        started = time.perf_counter()
        self.manifest["steps"][name] = {
            "status": "running",
            "fingerprint": fingerprint,
            "config": config,
            "artifacts": [str(path) for path in artifacts],
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        self.write_manifest()
        try:
            action()
            missing = [str(path) for path in artifacts if not path.exists()]
            if missing:
                raise RuntimeError(f"Step {name} did not produce {missing}")
            if validator is not None and not validator():
                raise RuntimeError(f"Step {name} failed artifact validation")
        except BaseException as error:
            self.manifest["steps"][name].update(
                {
                    "status": "failed",
                    "error": repr(error),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            self.write_manifest()
            raise
        self.manifest["steps"][name].update(
            {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        self.write_manifest()
        self.logger.info("completed step %s", name)


def _training_args(arm: Arm) -> argparse.Namespace:
    arguments = [
        "--seed",
        str(arm.seed),
        "--tcn-readout",
        arm.readout,
        "--training-horizon",
        arm.training_horizon,
        "--context-family-ablation",
        arm.context_ablation,
    ]
    return parse_training_args(arguments)


def _validate_completed_run(run_dir: Path, arm: Arm, store: Path) -> bool:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = manifest.get("model", {})
    settings = model.get("tcn_settings", {}) if isinstance(model, dict) else {}
    return (
        manifest.get("status") == "completed"
        and Path(str(manifest.get("feature_store"))).resolve() == store.resolve()
        and manifest.get("seed") == arm.seed
        and settings.get("readout") == arm.readout
        and manifest.get("training_horizon", "all") == arm.training_horizon
        and manifest.get("context_family_ablation", "none") == arm.context_ablation
        and (run_dir / "best_checkpoint.pt").exists()
        and (run_dir / "validation_metrics.json").exists()
        and (run_dir / "validation_daily_metrics.parquet").exists()
    )


def _run_arm(
    output_dir: Path,
    arm: Arm,
    store: Path,
    fit_rows: pl.DataFrame,
    selection_rows: pl.DataFrame,
    *,
    fit_name: str = "train",
    selection_name: str = "validation",
    allow_date_replacement: bool = False,
) -> None:
    assert_analysis_rows(fit_rows)
    assert_analysis_rows(
        selection_rows, allow_validation=selection_name == "validation"
    )
    if output_dir.exists():
        if _validate_completed_run(output_dir, arm, store):
            return
        moved = output_dir.with_name(
            f"{output_dir.name}.incomplete."
            f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
        )
        output_dir.rename(moved)
    output_dir.mkdir(parents=True)
    _run_neural(
        _training_args(arm),
        store,
        fit_rows,
        selection_rows,
        output_dir,
        fit_name=fit_name,
        selection_name=selection_name,
        allow_date_replacement=allow_date_replacement,
    )


def _preflight(
    store: Path, train_rows: pl.DataFrame, validation_rows: pl.DataFrame, path: Path
) -> None:
    assert_analysis_rows(train_rows)
    assert_analysis_rows(validation_rows, allow_validation=True)
    if train_rows.get_column("trade_date").min() != TRAIN_START:
        raise ValueError("Training start differs from the canonical contract")
    if validation_rows.get_column("trade_date").min() != VALIDATION_START:
        raise ValueError("Validation start differs from the canonical contract")
    torch.manual_seed(29)
    counts: dict[str, int] = {}
    state_keys: dict[str, list[str]] = {}
    for readout in TCN_READOUTS:
        architecture = architecture_for_model(
            "tcn", replace(BASELINE_TCN_SETTINGS, readout=readout)
        )
        model = build_neural_model("tcn", architecture, "selected")
        counts[readout] = count_trainable_parameters(model)
        state_keys[readout] = sorted(model.state_dict())
    final = counts["final"]
    if counts["horizon_multiscale"] - final != 18:
        raise AssertionError("Horizon multiscale must add exactly 18 parameters")
    if counts["final_score_mlp"] - final != 17:
        raise AssertionError("Score MLP must add exactly 17 parameters")
    summary = {
        "feature_store": feature_store_identity(store),
        "test_accessed": False,
        "parameter_counts": counts,
        "added_parameter_counts": {
            name: count - final for name, count in counts.items()
        },
        "final_state_dict_key_count": len(state_keys["final"]),
        "tap_receptive_field_minutes": tcn_tap_receptive_field_minutes(
            architecture_for_model("tcn", BASELINE_TCN_SETTINGS)
        ),
        "readout_modes": TCN_READOUTS,
    }
    atomic_json(path, summary)


def _run_artifacts(run_dir: Path) -> tuple[Path, ...]:
    return (
        run_dir / "run_manifest.json",
        run_dir / "best_checkpoint.pt",
        run_dir / "validation_metrics.json",
        run_dir / "validation_daily_metrics.parquet",
    )


def _metrics(run_dir: Path) -> dict[str, object]:
    return json.loads((run_dir / "validation_metrics.json").read_text(encoding="utf-8"))


def _arm_record(arm: Arm, run_dir: Path) -> dict[str, object]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    metrics = _metrics(run_dir)
    horizon_values = {
        int(row["horizon_minutes"]): float(row["mean_daily_spearman_ic"])
        for row in metrics["horizons"]
    }
    return {
        "arm": arm.name,
        "seed": arm.seed,
        "readout": arm.readout,
        "training_horizon": arm.training_horizon,
        "context_family_ablation": arm.context_ablation,
        "validation_ic": metrics["primary_score"],
        **{f"ic_{minutes}": horizon_values[minutes] for minutes in HORIZONS},
        "best_epoch": manifest["best_epoch"],
        "epochs_completed": manifest["epochs_completed"],
        "parameter_count": manifest["parameter_count"],
        "training_duration_seconds": manifest["total_run_seconds"],
        "run_dir": str(run_dir),
    }


def _daily_map(run_dir: Path) -> dict[tuple[int, int], float]:
    frame = pl.read_parquet(run_dir / "validation_daily_metrics.parquet")
    return {
        (int(date_idx), int(horizon)): float(value)
        for date_idx, horizon, value in frame.select(
            "date_idx", "horizon_minutes", "spearman_ic"
        ).iter_rows()
    }


def _comparison(
    name: str,
    candidate: Path,
    control: Path,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    candidate_values = _daily_map(candidate)
    control_values = _daily_map(control)
    paired: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for minutes in HORIZONS:
        keys = sorted(key for key in control_values if key[1] == minutes)
        deltas = np.asarray(
            [candidate_values[key] - control_values[key] for key in keys]
        )
        interval = moving_block_bootstrap(deltas, seed=BOOTSTRAP_SEED)
        rows.append(
            {
                "comparison": name,
                "seed": seed,
                "horizon_minutes": minutes,
                "control_ic": float(np.mean([control_values[key] for key in keys])),
                "candidate_ic": float(np.mean([candidate_values[key] for key in keys])),
                "delta_ic": float(interval["estimate"][0]),
                "delta_lower_95": float(interval["lower_95"][0]),
                "delta_upper_95": float(interval["upper_95"][0]),
            }
        )
        paired.extend(
            {
                "comparison": name,
                "seed": seed,
                "date_idx": key[0],
                "horizon_minutes": minutes,
                "delta_ic": candidate_values[key] - control_values[key],
            }
            for key in keys
        )
    aggregate = np.asarray(
        [
            np.mean(
                [
                    candidate_values[date_idx, minutes]
                    - control_values[date_idx, minutes]
                    for minutes in HORIZONS
                ]
            )
            for date_idx in sorted({key[0] for key in control_values})
        ]
    )
    interval = moving_block_bootstrap(aggregate, seed=BOOTSTRAP_SEED)
    rows.append(
        {
            "comparison": name,
            "seed": seed,
            "horizon_minutes": 0,
            "control_ic": _metrics(control)["primary_score"],
            "candidate_ic": _metrics(candidate)["primary_score"],
            "delta_ic": float(interval["estimate"][0]),
            "delta_lower_95": float(interval["lower_95"][0]),
            "delta_upper_95": float(interval["upper_95"][0]),
        }
    )
    return rows, paired


def _consolidate(output_dir: Path, run_dirs: dict[str, Path]) -> None:
    records = [
        _arm_record(ARM_BY_NAME[name], run_dir) for name, run_dir in run_dirs.items()
    ]
    atomic_csv(pl.DataFrame(records), output_dir / "run_matrix.csv")
    comparisons: list[dict[str, object]] = []
    paired: list[dict[str, object]] = []
    for seed in (11, 29, 47):
        name = f"horizon_multiscale_vs_final_seed{seed}"
        rows, daily = _comparison(
            name,
            run_dirs[f"horizon_multiscale_seed{seed}"],
            run_dirs[f"final_seed{seed}"],
            seed,
        )
        comparisons.extend(rows)
        paired.extend(daily)
    for name, candidate in (
        ("shared_multiscale_vs_final_seed29", "shared_multiscale_seed29"),
        ("final_score_mlp_vs_final_seed29", "final_score_mlp_seed29"),
        (
            "horizon_multiscale_vs_shared_seed29",
            "horizon_multiscale_seed29",
        ),
    ):
        control = "shared_multiscale_seed29" if "vs_shared" in name else "final_seed29"
        rows, daily = _comparison(name, run_dirs[candidate], run_dirs[control], 29)
        comparisons.extend(rows)
        paired.extend(daily)
    paired_frame = pl.DataFrame(paired)
    three_seed = paired_frame.filter(
        pl.col("comparison").str.starts_with("horizon_multiscale_vs_final")
    )
    for minutes in HORIZONS:
        values = (
            three_seed.filter(pl.col("horizon_minutes") == minutes)
            .group_by("date_idx")
            .agg(pl.col("delta_ic").mean())
            .sort("date_idx")
            .get_column("delta_ic")
            .to_numpy()
        )
        interval = moving_block_bootstrap(values, seed=BOOTSTRAP_SEED)
        seed_deltas = [
            float(
                next(
                    row["delta_ic"]
                    for row in comparisons
                    if row["comparison"] == f"horizon_multiscale_vs_final_seed{seed}"
                    and row["horizon_minutes"] == minutes
                )
            )
            for seed in (11, 29, 47)
        ]
        comparisons.append(
            {
                "comparison": "horizon_multiscale_vs_final_three_seed",
                "seed": 0,
                "horizon_minutes": minutes,
                "control_ic": None,
                "candidate_ic": None,
                "delta_ic": float(interval["estimate"][0]),
                "delta_lower_95": float(interval["lower_95"][0]),
                "delta_upper_95": float(interval["upper_95"][0]),
                "seed_mean_delta": float(np.mean(seed_deltas)),
                "seed_std_delta": float(np.std(seed_deltas, ddof=1)),
            }
        )
    per_date = (
        three_seed.group_by("date_idx", "horizon_minutes")
        .agg(pl.col("delta_ic").mean())
        .group_by("date_idx")
        .agg(pl.col("delta_ic").mean())
        .sort("date_idx")
        .get_column("delta_ic")
        .to_numpy()
    )
    interval = moving_block_bootstrap(per_date, seed=BOOTSTRAP_SEED)
    aggregate_seed_deltas = [
        float(
            next(
                row["delta_ic"]
                for row in comparisons
                if row["comparison"] == f"horizon_multiscale_vs_final_seed{seed}"
                and row["horizon_minutes"] == 0
            )
        )
        for seed in (11, 29, 47)
    ]
    comparisons.append(
        {
            "comparison": "horizon_multiscale_vs_final_three_seed",
            "seed": 0,
            "horizon_minutes": 0,
            "control_ic": None,
            "candidate_ic": None,
            "delta_ic": float(interval["estimate"][0]),
            "delta_lower_95": float(interval["lower_95"][0]),
            "delta_upper_95": float(interval["upper_95"][0]),
            "seed_mean_delta": float(np.mean(aggregate_seed_deltas)),
            "seed_std_delta": float(np.std(aggregate_seed_deltas, ddof=1)),
        }
    )
    atomic_csv(
        pl.DataFrame(comparisons, infer_schema_length=None),
        output_dir / "multiscale_comparison.csv",
    )
    averaged_daily = (
        three_seed.group_by("date_idx", "horizon_minutes")
        .agg(pl.col("delta_ic").mean())
        .with_columns(
            pl.lit("horizon_multiscale_vs_final_three_seed").alias("comparison"),
            pl.lit(0).alias("seed"),
        )
        .select(paired_frame.columns)
    )
    atomic_parquet(
        pl.concat((paired_frame, averaged_daily), how="vertical_relaxed"),
        output_dir / "multiscale_paired_daily.parquet",
    )
    gate_rows: list[dict[str, object]] = []
    for name, run_dir in run_dirs.items():
        arm = ARM_BY_NAME[name]
        if arm.readout not in ("shared_multiscale", "horizon_multiscale"):
            continue
        model, _, _ = load_current_neural_run(run_dir)
        weights = model.scale_weights().detach().numpy()
        if weights.ndim == 1:
            weights = np.tile(weights, (len(HORIZONS), 1))
        fields = tcn_tap_receptive_field_minutes(model.architecture)
        for horizon, minutes in enumerate(HORIZONS):
            entropy = float(-(weights[horizon] * np.log(weights[horizon])).sum())
            for block, (field, weight) in enumerate(
                zip(fields, weights[horizon], strict=True), start=1
            ):
                gate_rows.append(
                    {
                        "arm": name,
                        "seed": arm.seed,
                        "horizon_minutes": minutes,
                        "block": block,
                        "receptive_field_minutes": field,
                        "weight": float(weight),
                        "weight_std": None,
                        "entropy": entropy,
                        "entropy_std": None,
                    }
                )
    gate_frame = pl.DataFrame(gate_rows)
    gate_summary = (
        gate_frame.filter(pl.col("arm").str.starts_with("horizon_multiscale"))
        .group_by("horizon_minutes", "block", "receptive_field_minutes")
        .agg(
            pl.col("weight").mean(),
            pl.col("weight").std().alias("weight_std"),
            pl.col("entropy").mean(),
            pl.col("entropy").std().alias("entropy_std"),
        )
        .with_columns(
            pl.lit("horizon_multiscale_three_seed").alias("arm"),
            pl.lit(0).alias("seed"),
        )
        .select(gate_frame.columns)
    )
    atomic_csv(
        pl.concat((gate_frame, gate_summary), how="vertical_relaxed"),
        output_dir / "multiscale_gate_weights.csv",
    )
    control = next(row for row in records if row["arm"] == "final_seed29")
    single_rows = []
    for minutes in HORIZONS:
        row = next(
            value for value in records if value["arm"] == f"single_horizon_{minutes}"
        )
        single_rows.append(
            {
                **row,
                "paired_control_ic": control[f"ic_{minutes}"],
                "selected_horizon_ic": row[f"ic_{minutes}"],
                "delta_from_control": (row[f"ic_{minutes}"] - control[f"ic_{minutes}"]),
            }
        )
    gradient_dir = output_dir / "audits" / "gradient"
    atomic_csv(
        pl.DataFrame(single_rows),
        gradient_dir / "single_horizon_controls.csv",
    )
    gradient_summary_path = gradient_dir / "horizon_gradient_summary.json"
    gradient_summary = json.loads(gradient_summary_path.read_text(encoding="utf-8"))
    gradient_summary["single_horizon_controls"] = single_rows
    atomic_json(gradient_summary_path, gradient_summary)
    context_rows = []
    for family in ("wdo", "br_rates", "us_rates"):
        row = next(value for value in records if value["arm"] == f"without_{family}")
        context_rows.append(
            {
                **row,
                "paired_control_ic": control["validation_ic"],
                "delta_from_control": (row["validation_ic"] - control["validation_ic"]),
            }
        )
    context_dir = output_dir / "audits" / "context"
    atomic_csv(
        pl.DataFrame(context_rows),
        context_dir / "context_training_ablations.csv",
    )
    context_summary_path = context_dir / "context_family_summary.json"
    context_summary = json.loads(context_summary_path.read_text(encoding="utf-8"))
    context_summary["training_ablations"] = context_rows
    atomic_json(context_summary_path, context_summary)
    three_seed_rows = [
        row
        for row in comparisons
        if row["comparison"] == "horizon_multiscale_vs_final_three_seed"
    ]
    worst = min(
        float(row["delta_ic"]) for row in three_seed_rows if row["horizon_minutes"] != 0
    )

    def aggregate_delta(name: str) -> float:
        return float(
            next(
                row["delta_ic"]
                for row in comparisons
                if row["comparison"] == name and row["horizon_minutes"] == 0
            )
        )

    shared_delta = aggregate_delta("shared_multiscale_vs_final_seed29")
    specialized_delta = aggregate_delta("horizon_multiscale_vs_shared_seed29")
    score_delta = aggregate_delta("final_score_mlp_vs_final_seed29")
    three_seed_aggregate = next(
        row for row in three_seed_rows if row["horizon_minutes"] == 0
    )
    frozen_summary = json.loads(
        (
            output_dir / "audits" / "frozen_block" / "frozen_block_probe_summary.json"
        ).read_text(encoding="utf-8")
    )
    consistent_gain = (
        float(three_seed_aggregate["delta_lower_95"]) > 0.0 and worst > 0.0
    )
    hypothesis_evidence = {
        "shared_scale_aggregation_delta_seed29": shared_delta,
        "horizon_specialization_over_shared_delta_seed29": specialized_delta,
        "score_capacity_control_delta_seed29": score_delta,
        "score_vs_horizon_gain_difference_seed29": (
            score_delta - aggregate_delta("horizon_multiscale_vs_final_seed29")
        ),
        "single_horizon_deltas": {
            str(row["training_horizon"]): row["delta_from_control"]
            for row in single_rows
        },
        "probe_best_tap_by_horizon": frozen_summary["best_tap_by_horizon"],
        "probe_information_loss_by_horizon": frozen_summary[
            "earlier_tap_beats_final_post_fusion_by_horizon"
        ],
        "consistent_three_seed_gain": consistent_gain,
    }
    summary = {
        "training_run_count": TRAINING_RUN_COUNT,
        "test_accessed": False,
        "bootstrap": {"block_days": 5, "seed": BOOTSTRAP_SEED},
        "three_seed_horizon_multiscale": three_seed_rows,
        "worst_horizon_delta": worst,
        "hypothesis_evidence": hypothesis_evidence,
        "conclusion": (
            "The paired three-seed result is consistent across horizons and its "
            "aggregate interval excludes zero."
            if consistent_gain
            else "No consistent paired three-seed gain was established; the "
            "final-state bottleneck hypothesis is not supported strongly enough."
        ),
        "promotion": "none",
    }
    atomic_json(output_dir / "stage_summary.json", summary)
    lines = [
        "# Horizon multiscale stage summary",
        "",
        f"- Training runs: {TRAINING_RUN_COUNT}",
        "- Held-out test accessed: no",
        f"- Shared aggregation delta (seed 29): {shared_delta:.6f}",
        f"- Horizon specialization over shared (seed 29): {specialized_delta:.6f}",
        f"- Score-capacity control delta (seed 29): {score_delta:.6f}",
        f"- Worst-horizon paired delta: {worst:.6f}",
        f"- Conclusion: {summary['conclusion']}",
        "- Architecture promotion: none.",
        "",
        "See the CSV, Parquet, and JSON artifacts for the paired and diagnostic evidence.",
    ]
    _atomic_text(output_dir / "stage_summary.md", "\n".join(lines) + "\n")


def run_stage(output_dir: Path) -> Path:
    store = resolve_feature_store()
    sample_index = load_sample_index(store)
    train_rows = select_sample_split(sample_index, "train")
    validation_rows = select_sample_split(sample_index, "validation")
    stage = Stage(output_dir, store)
    preflight = stage.output_dir / "preflight.json"
    stage.step(
        "preflight",
        {"schema": STAGE_SCHEMA},
        (preflight,),
        lambda: _preflight(store, train_rows, validation_rows, preflight),
    )
    target_dir = stage.output_dir / "audits" / "target_basis"
    stage.step(
        "target_basis",
        {"window": "train"},
        (
            target_dir / "target_basis_summary.json",
            target_dir / "target_pairwise.csv",
            target_dir / "target_basis_by_date.parquet",
            target_dir / "target_basis_by_decision.csv",
        ),
        lambda: run_target_basis_audit(store, train_rows, target_dir),
    )
    run_dirs = {arm.name: stage.output_dir / "runs" / arm.name for arm in ARMS}

    def training_step(
        name: str,
        fit_rows: pl.DataFrame = train_rows,
        selection_rows: pl.DataFrame = validation_rows,
        fit_name: str = "train",
        selection_name: str = "validation",
        allow_date_replacement: bool = False,
    ) -> None:
        arm = ARM_BY_NAME[name]
        stage.step(
            f"train_{name}",
            {
                **asdict(arm),
                "fit_name": fit_name,
                "selection_name": selection_name,
                "fit_start": str(fit_rows.get_column("trade_date").min()),
                "fit_end": str(fit_rows.get_column("trade_date").max()),
                "selection_start": str(selection_rows.get_column("trade_date").min()),
                "selection_end": str(selection_rows.get_column("trade_date").max()),
            },
            _run_artifacts(run_dirs[name]),
            lambda: _run_arm(
                run_dirs[name],
                arm,
                store,
                fit_rows,
                selection_rows,
                fit_name=fit_name,
                selection_name=selection_name,
                allow_date_replacement=allow_date_replacement,
            ),
            lambda: _validate_completed_run(run_dirs[name], arm, store),
        )

    training_step("final_seed29")
    frozen_dir = stage.output_dir / "audits" / "frozen_block"
    stage.step(
        "frozen_block_probes",
        {"checkpoint": "final_seed29"},
        (
            frozen_dir / "frozen_block_probes.csv",
            frozen_dir / "frozen_block_probe_summary.json",
        ),
        lambda: run_frozen_block_probes(
            run_dirs["final_seed29"],
            store,
            train_rows,
            validation_rows,
            frozen_dir,
        ),
    )
    context_dir = stage.output_dir / "audits" / "context"
    stage.step(
        "context_inference_probes",
        {"checkpoint": "final_seed29", "window": "validation"},
        (
            context_dir / "context_inference_probes.csv",
            context_dir / "context_permutation_manifest.json",
            context_dir / "context_family_summary.json",
        ),
        lambda: run_context_inference_probes(
            run_dirs["final_seed29"], store, validation_rows, context_dir
        ),
    )
    gradient_dir = stage.output_dir / "audits" / "gradient"
    stage.step(
        "gradient_audit",
        {"checkpoint": "final_seed29", "window": "train"},
        (
            gradient_dir / "horizon_gradient_audit.parquet",
            gradient_dir / "horizon_gradient_summary.json",
        ),
        lambda: run_gradient_audit(
            run_dirs["final_seed29"], store, train_rows, gradient_dir
        ),
    )
    for name in (
        "shared_multiscale_seed29",
        "horizon_multiscale_seed29",
        "final_score_mlp_seed29",
        "final_seed11",
        "final_seed47",
        "horizon_multiscale_seed11",
        "horizon_multiscale_seed47",
        "single_horizon_30",
        "single_horizon_60",
        "single_horizon_120",
        "without_wdo",
        "without_br_rates",
        "without_us_rates",
    ):
        training_step(name)
    oof_dir = stage.output_dir / "audits" / "oof"
    windows, plan = build_oof_plan(train_rows)
    stage.step(
        "oof_plan",
        {"window": "train"},
        (oof_dir / "oof_plan.json",),
        lambda: (
            oof_dir.mkdir(parents=True, exist_ok=True),
            atomic_json(oof_dir / "oof_plan.json", plan),
        ),
    )
    training_step(
        "oof_fold_1",
        windows["B0"],
        windows["B1"],
        "B0",
        "B1",
        True,
    )
    training_step(
        "oof_fold_2",
        pl.concat((windows["B0"], windows["B1"])).sort("sample_id"),
        windows["B2"],
        "B0+B1",
        "B2",
        True,
    )
    stage.step(
        "oof_residual_probes",
        {"fit": "B2", "evaluation": "B3"},
        (
            oof_dir / "oof_residual_probes.csv",
            oof_dir / "oof_residual_probe_summary.json",
        ),
        lambda: run_oof_residual_probes(
            run_dirs["oof_fold_1"],
            run_dirs["oof_fold_2"],
            store,
            windows["B2"],
            windows["B3"],
            oof_dir,
        ),
    )
    consolidated = (
        stage.output_dir / "run_matrix.csv",
        stage.output_dir / "multiscale_comparison.csv",
        stage.output_dir / "multiscale_paired_daily.parquet",
        stage.output_dir / "multiscale_gate_weights.csv",
        stage.output_dir / "stage_summary.json",
        stage.output_dir / "stage_summary.md",
        gradient_dir / "single_horizon_controls.csv",
        context_dir / "context_training_ablations.csv",
    )
    stage.step(
        "consolidate",
        {"training_runs": TRAINING_RUN_COUNT},
        consolidated,
        lambda: _consolidate(stage.output_dir, run_dirs),
    )
    stage.manifest["status"] = "completed"
    stage.manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    stage.write_manifest()
    return stage.output_dir


def main() -> None:
    args = parse_args()
    print(run_stage(args.output_dir))


if __name__ == "__main__":
    main()
