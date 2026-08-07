from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl

from .analyze_context_ablation import (
    BOOTSTRAP_BLOCK_TRADING_DAYS,
    BOOTSTRAP_REPLICATIONS,
    BOOTSTRAP_SEED,
    paired_moving_block_bootstrap,
)
from .analyze_stage2_context_ablation import (
    FIRST_HALF_END,
    LATEST_HALF_START,
    _daily_primary,
    _manifest_sha256,
    _period_metrics,
    _read_validation_daily_metrics,
    _training_diagnostics,
    _validate_metrics_json,
)
from .context_ablation import get_context_ablation
from .contract import (
    FEATURE_CONTRACT_VERSION,
    HORIZONS,
    VALIDATION_END,
    VALIDATION_START,
    SplitBoundaries,
)
from .stage2_context_ablation import _training_semantics
from .stage3_context_addition import (
    ADOPTED_STAGE2_CONTEXT_ABLATION,
    ADOPTED_STAGE2_LOGICAL_CONFIGURATION,
    PACKAGED_FEATURE_MANIFEST_SHA256,
    STAGE2_PRODUCING_COMMIT,
    STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION,
    STAGE3_LOGICAL_CONFIGURATION_ORDER,
    STAGE3_SEEDS,
    STATE_VERSION,
    SWEEP_NAME,
    _completed_job_artifacts,
    _reject_test_derived_metadata,
    _validated_stage2_adoptions,
    build_stage3_command,
    validate_stage3_completed_run,
)

_SUMMARY_JSON = "stage3_context_addition_summary.json"
_SUMMARY_CSV = "stage3_context_addition_summary.csv"
_PERIODS = ("full_validation", "first_half", "latest_half")


def _with_core_delta(
    current: dict[str, object], core: dict[str, object]
) -> dict[str, object]:
    result = dict(current)
    result["primary_delta_vs_same_seed_core"] = float(current["primary_ic"]) - float(
        core["primary_ic"]
    )
    result["gross_top_minus_bottom_delta_vs_same_seed_core"] = float(
        current["mean_gross_top_minus_bottom"]
    ) - float(core["mean_gross_top_minus_bottom"])
    result["one_way_turnover_delta_vs_same_seed_core"] = float(
        current["mean_one_way_turnover"]
    ) - float(core["mean_one_way_turnover"])
    current_horizons = current["horizons"]
    core_horizons = core["horizons"]
    if not isinstance(current_horizons, dict) or not isinstance(core_horizons, dict):
        raise ValueError("Period horizon metrics are malformed")
    result["horizons"] = {
        horizon: {
            **values,
            "spearman_delta_vs_same_seed_core": float(values["spearman_ic"])
            - float(core_horizons[horizon]["spearman_ic"]),
            "gross_top_minus_bottom_delta_vs_same_seed_core": float(
                values["gross_top_minus_bottom"]
            )
            - float(core_horizons[horizon]["gross_top_minus_bottom"]),
            "one_way_turnover_delta_vs_same_seed_core": float(
                values["one_way_turnover"]
            )
            - float(core_horizons[horizon]["one_way_turnover"]),
        }
        for horizon, values in current_horizons.items()
    }
    return result


def _delta_summary(values_by_seed: dict[int, float]) -> dict[str, object]:
    if tuple(sorted(values_by_seed)) != STAGE3_SEEDS:
        raise ValueError("Across-seed summary requires exactly seeds 11, 29, and 47")
    values = np.asarray(
        [values_by_seed[seed] for seed in STAGE3_SEEDS], dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError("Across-seed deltas must be finite")
    tolerance = 1e-15
    return {
        "paired_delta_by_seed": {
            str(seed): float(values_by_seed[seed]) for seed in STAGE3_SEEDS
        },
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "positive_seed_count": int(np.sum(values > tolerance)),
        "zero_seed_count": int(np.sum(np.abs(values) <= tolerance)),
        "negative_seed_count": int(np.sum(values < -tolerance)),
    }


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        pl.DataFrame(rows).write_csv(temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_configuration(configuration: dict[str, object]) -> None:
    expected = {
        "feature_contract": FEATURE_CONTRACT_VERSION,
        "training_semantics": _training_semantics(),
        "split_boundaries": {
            key: str(value) for key, value in asdict(SplitBoundaries()).items()
        },
        "logical_configuration_order": list(STAGE3_LOGICAL_CONFIGURATION_ORDER),
        "context_ablation_by_logical_configuration": dict(
            STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION
        ),
        "context_ablation_metadata_by_logical_configuration": {
            logical: get_context_ablation(key).metadata()
            for logical, key in STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION.items()
        },
        "seeds": list(STAGE3_SEEDS),
        "logical_job_count": 24,
        "adopted_stage2_job_count": 3,
        "new_training_job_count": 21,
        "required_stage2_producing_commit": STAGE2_PRODUCING_COMMIT,
        "required_feature_manifest_sha256": PACKAGED_FEATURE_MANIFEST_SHA256,
    }
    for field, value in expected.items():
        if configuration.get(field) != value:
            raise ValueError(f"Stage-3 configuration is incompatible: {field}")
    feature_identity = configuration.get("feature_store")
    if (
        not isinstance(feature_identity, dict)
        or feature_identity.get("manifest_sha256") != PACKAGED_FEATURE_MANIFEST_SHA256
        or not isinstance(configuration.get("orchestrator_git_commit_sha"), str)
        or not isinstance(configuration.get("source_stage2_state"), str)
        or not isinstance(configuration.get("source_stage2_state_sha256"), str)
        or not isinstance(
            configuration.get("source_stage2_feature_store_resolved_path"), str
        )
    ):
        raise ValueError("Stage-3 configuration is missing strict provenance")


def _raw_periods(daily: pl.DataFrame) -> dict[str, dict[str, object]]:
    return {
        "full_validation": _period_metrics(daily, VALIDATION_START, VALIDATION_END),
        "first_half": _period_metrics(daily, VALIDATION_START, FIRST_HALF_END),
        "latest_half": _period_metrics(daily, LATEST_HALF_START, VALIDATION_END),
    }


def analyze_sweep(
    state_path: Path,
    output_dir: Path,
    stage2_state_path: Path | None = None,
) -> tuple[Path, Path]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    _reject_test_derived_metadata(state, "Stage-3 state")
    if (
        state.get("state_version") != STATE_VERSION
        or state.get("sweep_name") != SWEEP_NAME
        or state.get("status") != "completed"
    ):
        raise ValueError("Analyzer requires a completed Stage-3 state")
    configuration = state.get("configuration")
    jobs = state.get("jobs")
    if not isinstance(configuration, dict) or not isinstance(jobs, list):
        raise ValueError("Stage-3 state is missing configuration or jobs")
    _validate_configuration(configuration)
    source_state = (
        stage2_state_path
        if stage2_state_path is not None
        else Path(str(configuration["source_stage2_state"]))
    )
    adopted_source = _validated_stage2_adoptions(source_state, configuration)

    expected_order = tuple(
        (
            logical,
            STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION[logical],
            seed,
        )
        for logical in STAGE3_LOGICAL_CONFIGURATION_ORDER
        for seed in STAGE3_SEEDS
    )
    actual_order = tuple(
        (
            job.get("logical_configuration"),
            job.get("context_ablation"),
            job.get("seed"),
        )
        for job in jobs
        if isinstance(job, dict)
    )
    if actual_order != expected_order:
        raise ValueError("Analyzer requires the exact canonical 24-job matrix")
    if any(job.get("status") != "completed" for job in jobs):
        raise ValueError("Analyzer refuses a partially completed matrix")
    resolved: list[Path] = []
    for job in jobs:
        run_dir, _, _ = _completed_job_artifacts(job, configuration)
        job["run_dir"] = str(run_dir)
        resolved.append(run_dir.resolve())
    run_dirs = tuple(resolved)
    if len(set(run_dirs)) != 24:
        raise ValueError("Analyzer requires 24 distinct run directories")

    validated: dict[
        tuple[str, int],
        tuple[dict[str, object], Path, dict[str, object], pl.DataFrame],
    ] = {}
    for job, run_dir in zip(jobs, run_dirs, strict=True):
        logical = str(job["logical_configuration"])
        key = str(job["context_ablation"])
        seed = int(job["seed"])
        should_be_adopted = logical == ADOPTED_STAGE2_LOGICAL_CONFIGURATION
        expected_origin = "adopted_stage2" if should_be_adopted else "trained_stage3"
        if job.get("result_origin") != expected_origin:
            raise ValueError(
                f"Stage-3 job has the wrong result origin: {logical}/{seed}"
            )
        producing_commit = (
            STAGE2_PRODUCING_COMMIT
            if should_be_adopted
            else str(configuration["orchestrator_git_commit_sha"])
        )
        if job.get("producing_git_commit_sha") != producing_commit:
            raise ValueError(
                f"Stage-3 job has the wrong producing commit: {logical}/{seed}"
            )
        if job.get("context_ablation_metadata") != get_context_ablation(key).metadata():
            raise ValueError(
                f"Stage-3 job has wrong ablation metadata: {logical}/{seed}"
            )
        if job.get("command") != list(build_stage3_command(logical, seed)):
            raise ValueError(
                f"Stage-3 job has wrong training command: {logical}/{seed}"
            )
        if should_be_adopted:
            source_job, source_run_dir, source_score, source_manifest_sha = (
                adopted_source[seed]
            )
            provenance = job.get("source_stage2_job")
            if (
                job.get("source_stage2_state_sha256")
                != configuration["source_stage2_state_sha256"]
                or not isinstance(provenance, dict)
                or provenance.get("logical_configuration")
                != ADOPTED_STAGE2_CONTEXT_ABLATION
                or provenance.get("context_ablation") != ADOPTED_STAGE2_CONTEXT_ABLATION
                or provenance.get("seed") != seed
                or provenance.get("run_dir") != source_job.get("run_dir")
                or job.get("run_manifest_sha256") != source_manifest_sha
                or _manifest_sha256(source_run_dir / "run_manifest.json")
                != source_manifest_sha
                or not math.isclose(
                    float(job.get("primary_validation_ic")),
                    source_score,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    f"Adopted Stage-3 provenance is invalid: {logical}/{seed}"
                )
        elif (
            job.get("source_stage2_state") is not None
            or job.get("source_stage2_state_sha256") is not None
            or job.get("source_stage2_job") is not None
        ):
            raise ValueError(
                f"New Stage-3 run has Stage-2 provenance: {logical}/{seed}"
            )
        score = validate_stage3_completed_run(
            run_dir, configuration, key, seed, producing_commit
        )
        manifest_path = run_dir / "run_manifest.json"
        manifest_sha = _manifest_sha256(manifest_path)
        if manifest_sha != job.get("run_manifest_sha256") or not math.isclose(
            score,
            float(job.get("primary_validation_ic")),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"Stage-3 state disagrees with run artifacts: {logical}/{seed}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        daily = _read_validation_daily_metrics(run_dir)
        full = _period_metrics(daily, VALIDATION_START, VALIDATION_END)
        _validate_metrics_json(run_dir, full)
        validated[(logical, seed)] = (job, run_dir, manifest, daily)

    core_periods = {
        seed: _raw_periods(validated[("core", seed)][3]) for seed in STAGE3_SEEDS
    }
    core_daily = {
        seed: _daily_primary(validated[("core", seed)][3]) for seed in STAGE3_SEEDS
    }

    run_results: list[dict[str, object]] = []
    results_by_identity: dict[tuple[str, int], dict[str, object]] = {}
    for logical in STAGE3_LOGICAL_CONFIGURATION_ORDER:
        key = STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION[logical]
        for seed in STAGE3_SEEDS:
            job, run_dir, manifest, daily = validated[(logical, seed)]
            raw_periods = _raw_periods(daily)
            periods = {
                name: _with_core_delta(period, core_periods[seed][name])
                for name, period in raw_periods.items()
            }
            result = {
                "logical_configuration": logical,
                "context_ablation": key,
                "context_ablation_specification": manifest["context_ablation"],
                "ablation_specification_sha256": manifest["context_ablation"][
                    "specification_sha256"
                ],
                "seed": seed,
                "result_origin": job["result_origin"],
                "run_dir": str(run_dir),
                "run_manifest_path": str((run_dir / "run_manifest.json").resolve()),
                "run_manifest_sha256": job["run_manifest_sha256"],
                "producing_git_commit_sha": manifest["git_commit_sha"],
                "source_stage2_state": job["source_stage2_state"],
                "source_stage2_state_sha256": job["source_stage2_state_sha256"],
                "source_stage2_job": job["source_stage2_job"],
                "feature_store_identity": configuration["feature_store"],
                "split_boundaries": manifest["split_boundaries"],
                "periods": periods,
                "training": _training_diagnostics(run_dir, manifest),
                "within_trained_model_daily_bootstrap": (
                    paired_moving_block_bootstrap(
                        core_daily[seed], _daily_primary(daily)
                    )
                ),
            }
            run_results.append(result)
            results_by_identity[(logical, seed)] = result

    configuration_results: list[dict[str, object]] = []
    summaries_by_logical: dict[str, dict[str, object]] = {}
    for logical in STAGE3_LOGICAL_CONFIGURATION_ORDER:
        key = STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION[logical]
        seed_results = [results_by_identity[(logical, seed)] for seed in STAGE3_SEEDS]
        primary_deltas = {
            seed: float(
                results_by_identity[(logical, seed)]["periods"]["full_validation"][
                    "primary_delta_vs_same_seed_core"
                ]
            )
            for seed in STAGE3_SEEDS
        }
        summary = {
            "logical_configuration": logical,
            "context_ablation": key,
            "absolute_validation_primary_ic_by_seed": {
                str(seed): float(
                    results_by_identity[(logical, seed)]["periods"]["full_validation"][
                        "primary_ic"
                    ]
                )
                for seed in STAGE3_SEEDS
            },
            "primary_delta_across_training_seeds": _delta_summary(primary_deltas),
            "horizon_delta_across_training_seeds": {
                f"{horizon}m": _delta_summary(
                    {
                        seed: float(
                            results_by_identity[(logical, seed)]["periods"][
                                "full_validation"
                            ]["horizons"][f"{horizon}m"][
                                "spearman_delta_vs_same_seed_core"
                            ]
                        )
                        for seed in STAGE3_SEEDS
                    }
                )
                for horizon in HORIZONS
            },
            "period_delta_across_training_seeds": {
                period: _delta_summary(
                    {
                        seed: float(
                            results_by_identity[(logical, seed)]["periods"][period][
                                "primary_delta_vs_same_seed_core"
                            ]
                        )
                        for seed in STAGE3_SEEDS
                    }
                )
                for period in ("first_half", "latest_half")
            },
            "diagnostic_delta_across_training_seeds": {
                "gross_top_minus_bottom": _delta_summary(
                    {
                        seed: float(
                            results_by_identity[(logical, seed)]["periods"][
                                "full_validation"
                            ]["gross_top_minus_bottom_delta_vs_same_seed_core"]
                        )
                        for seed in STAGE3_SEEDS
                    }
                ),
                "one_way_turnover": _delta_summary(
                    {
                        seed: float(
                            results_by_identity[(logical, seed)]["periods"][
                                "full_validation"
                            ]["one_way_turnover_delta_vs_same_seed_core"]
                        )
                        for seed in STAGE3_SEEDS
                    }
                ),
            },
            "seed_results": seed_results,
        }
        configuration_results.append(summary)
        summaries_by_logical[logical] = summary

    csv_rows: list[dict[str, object]] = []
    for result in run_results:
        logical = str(result["logical_configuration"])
        full = result["periods"]["full_validation"]
        first = result["periods"]["first_half"]
        latest = result["periods"]["latest_half"]
        bootstrap = result["within_trained_model_daily_bootstrap"]
        aggregate = summaries_by_logical[logical]["primary_delta_across_training_seeds"]
        row: dict[str, object] = {
            "logical_configuration": logical,
            "context_ablation": result["context_ablation"],
            "seed": result["seed"],
            "result_origin": result["result_origin"],
            "primary_validation_ic": full["primary_ic"],
            "primary_delta_vs_same_seed_core": full["primary_delta_vs_same_seed_core"],
            "first_half_primary_ic": first["primary_ic"],
            "first_half_delta_vs_same_seed_core": first[
                "primary_delta_vs_same_seed_core"
            ],
            "latest_half_primary_ic": latest["primary_ic"],
            "latest_half_delta_vs_same_seed_core": latest[
                "primary_delta_vs_same_seed_core"
            ],
            "mean_gross_top_minus_bottom": full["mean_gross_top_minus_bottom"],
            "gross_top_minus_bottom_delta_vs_same_seed_core": full[
                "gross_top_minus_bottom_delta_vs_same_seed_core"
            ],
            "mean_one_way_turnover": full["mean_one_way_turnover"],
            "one_way_turnover_delta_vs_same_seed_core": full[
                "one_way_turnover_delta_vs_same_seed_core"
            ],
            "bootstrap_paired_mean_delta": bootstrap["paired_mean_delta"],
            "bootstrap_interval_lower_95": bootstrap["interval_lower_95"],
            "bootstrap_interval_upper_95": bootstrap["interval_upper_95"],
            "across_seed_delta_mean": aggregate["mean"],
            "across_seed_delta_median": aggregate["median"],
            "across_seed_delta_minimum": aggregate["minimum"],
            "across_seed_delta_maximum": aggregate["maximum"],
            "positive_seed_count": aggregate["positive_seed_count"],
            "zero_seed_count": aggregate["zero_seed_count"],
            "negative_seed_count": aggregate["negative_seed_count"],
            **result["training"],
            "run_dir": result["run_dir"],
            "producing_git_commit_sha": result["producing_git_commit_sha"],
            "run_manifest_sha256": result["run_manifest_sha256"],
        }
        for horizon in HORIZONS:
            horizon_metrics = full["horizons"][f"{horizon}m"]
            row.update(
                {
                    f"ic_{horizon}m": horizon_metrics["spearman_ic"],
                    f"delta_ic_{horizon}m_vs_same_seed_core": (
                        horizon_metrics["spearman_delta_vs_same_seed_core"]
                    ),
                    f"gross_top_minus_bottom_{horizon}m": horizon_metrics[
                        "gross_top_minus_bottom"
                    ],
                    f"one_way_turnover_{horizon}m": horizon_metrics["one_way_turnover"],
                }
            )
        csv_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / _SUMMARY_JSON
    csv_path = output_dir / _SUMMARY_CSV
    summary = {
        "state_file": str(state_path.resolve()),
        "logical_job_count": 24,
        "configuration_count": 8,
        "seeds": list(STAGE3_SEEDS),
        "logical_configuration_order": list(STAGE3_LOGICAL_CONFIGURATION_ORDER),
        "context_ablation_by_logical_configuration": dict(
            STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION
        ),
        "same_seed_reference_configuration": "core",
        "feature_store_identity": configuration["feature_store"],
        "split_boundaries": configuration["split_boundaries"],
        "orchestrator_git_commit_sha": configuration["orchestrator_git_commit_sha"],
        "source_stage2_state": str(source_state),
        "source_stage2_state_sha256": configuration["source_stage2_state_sha256"],
        "result_origin_counts": {
            "adopted_stage2": 3,
            "trained_stage3": 21,
        },
        "uncertainty_interpretation": {
            "within_trained_model_daily_time_series": {
                "method": (
                    "paired moving-block bootstrap of daily primary IC "
                    "differences versus the same-seed core run"
                ),
                "block_trading_days": BOOTSTRAP_BLOCK_TRADING_DAYS,
                "replications": BOOTSTRAP_REPLICATIONS,
                "seed": BOOTSTRAP_SEED,
            },
            "across_training_seeds": (
                "The three matched-seed deltas, their sign consistency, "
                "and their range describe training-seed sensitivity. "
                "Three seeds do not support a conclusive conventional "
                "seed-level significance test."
            ),
            "adopted_result_reuse": (
                "The three core_plus_win results are strictly validated "
                "Stage-2 artifacts, not newly trained Stage-3 models."
            ),
        },
        "selection": None,
        "runs": run_results,
        "configurations": configuration_results,
    }
    _atomic_write_json(json_path, summary)
    _atomic_write_csv(csv_path, csv_rows)
    return json_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--stage2-state", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_file = args.state_file.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else state_file.parent / "analysis"
    )
    source = args.stage2_state.resolve() if args.stage2_state is not None else None
    json_path, csv_path = analyze_sweep(
        state_file, output_dir, stage2_state_path=source
    )
    print(f"Wrote Stage-3 JSON summary: {json_path}")
    print(f"Wrote Stage-3 CSV summary: {csv_path}")


if __name__ == "__main__":
    main()
