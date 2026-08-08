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
    _period_metrics,
    _read_validation_daily_metrics,
    _training_diagnostics,
    _validate_metrics_json,
)
from .audit_slow_features import validate_training_slow_audit
from .context_ablation import get_context_ablation
from .contract import (
    FEATURE_CONTRACT_VERSION,
    HORIZONS,
    VALIDATION_END,
    VALIDATION_START,
    SplitBoundaries,
)
from .stage2_context_ablation import _training_semantics
from .stage4_slow_feature_ablation import (
    EXPECTED_RETAINED_CONTEXTS,
    FROZEN_CONTEXT_ABLATION,
    PACKAGED_FEATURE_MANIFEST_SHA256,
    STAGE4_FEATURE_ABLATION_BY_LOGICAL_CONFIGURATION,
    STAGE4_LOGICAL_CONFIGURATION_ORDER,
    STAGE4_SEEDS,
    STATE_VERSION,
    SWEEP_NAME,
    _completed_job_artifacts,
    _reject_test_derived_metadata,
    _validated_stage3_adoptions,
    stage4_jobs,
    validate_stage4_completed_run,
)

SUMMARY_JSON = "stage4_slow_low_prior_summary.json"
RUNS_CSV = "stage4_slow_low_prior_runs.csv"
MATCHED_CSV = "stage4_slow_low_prior_matched_seeds.csv"


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        "retained_context_symbols": list(EXPECTED_RETAINED_CONTEXTS),
        "training_semantics": _training_semantics(),
        "split_boundaries": {
            key: str(value) for key, value in asdict(SplitBoundaries()).items()
        },
        "logical_configuration_order": list(STAGE4_LOGICAL_CONFIGURATION_ORDER),
        "feature_ablation_by_logical_configuration": dict(
            STAGE4_FEATURE_ABLATION_BY_LOGICAL_CONFIGURATION
        ),
        "context_ablation": FROZEN_CONTEXT_ABLATION,
        "context_ablation_metadata": get_context_ablation(
            FROZEN_CONTEXT_ABLATION
        ).metadata(),
        "seeds": list(STAGE4_SEEDS),
        "logical_job_count": 6,
        "adopted_stage3_job_count": 3,
        "new_training_job_count": 3,
        "required_feature_manifest_sha256": PACKAGED_FEATURE_MANIFEST_SHA256,
    }
    for field, value in expected.items():
        if configuration.get(field) != value:
            raise ValueError(f"Stage-4 configuration is incompatible: {field}")
    feature_identity = configuration.get("feature_store")
    metadata = configuration.get("feature_ablation_metadata_by_key")
    audit = configuration.get("training_slow_audit")
    if (
        not isinstance(feature_identity, dict)
        or feature_identity.get("manifest_sha256") != PACKAGED_FEATURE_MANIFEST_SHA256
        or not isinstance(metadata, dict)
        or tuple(metadata) != ("none", "drop_slow_low_prior")
        or any(not isinstance(metadata[key], dict) for key in metadata)
        or not isinstance(configuration.get("orchestrator_git_commit_sha"), str)
        or not isinstance(configuration.get("source_stage3_state"), str)
        or not isinstance(configuration.get("source_stage3_state_sha256"), str)
        or not isinstance(configuration.get("source_stage3_producing_commit"), str)
        or not isinstance(audit, dict)
        or not isinstance(audit.get("sha256"), str)
    ):
        raise ValueError("Stage-4 configuration is missing strict provenance")


def _periods(daily: pl.DataFrame) -> dict[str, dict[str, object]]:
    return {
        "full_validation": _period_metrics(daily, VALIDATION_START, VALIDATION_END),
        "first_half": _period_metrics(daily, VALIDATION_START, FIRST_HALF_END),
        "latest_half": _period_metrics(daily, LATEST_HALF_START, VALIDATION_END),
    }


def _period_delta(
    treatment: dict[str, object], control: dict[str, object]
) -> dict[str, object]:
    treatment_horizons = treatment["horizons"]
    control_horizons = control["horizons"]
    if not isinstance(treatment_horizons, dict) or not isinstance(
        control_horizons, dict
    ):
        raise ValueError("Stage-4 period horizon metrics are malformed")
    return {
        "start": treatment["start"],
        "end": treatment["end"],
        "date_count": treatment["date_count"],
        "control_primary_ic": control["primary_ic"],
        "treatment_primary_ic": treatment["primary_ic"],
        "delta_ic": float(treatment["primary_ic"]) - float(control["primary_ic"]),
        "control_mean_gross_top_minus_bottom": control["mean_gross_top_minus_bottom"],
        "treatment_mean_gross_top_minus_bottom": treatment[
            "mean_gross_top_minus_bottom"
        ],
        "delta_mean_gross_top_minus_bottom": float(
            treatment["mean_gross_top_minus_bottom"]
        )
        - float(control["mean_gross_top_minus_bottom"]),
        "control_mean_one_way_turnover": control["mean_one_way_turnover"],
        "treatment_mean_one_way_turnover": treatment["mean_one_way_turnover"],
        "delta_mean_one_way_turnover": float(treatment["mean_one_way_turnover"])
        - float(control["mean_one_way_turnover"]),
        "horizons": {
            horizon: {
                "control_spearman_ic": control_horizons[horizon]["spearman_ic"],
                "treatment_spearman_ic": treatment_horizons[horizon]["spearman_ic"],
                "delta_ic": float(treatment_horizons[horizon]["spearman_ic"])
                - float(control_horizons[horizon]["spearman_ic"]),
                "control_gross_top_minus_bottom": control_horizons[horizon][
                    "gross_top_minus_bottom"
                ],
                "treatment_gross_top_minus_bottom": treatment_horizons[horizon][
                    "gross_top_minus_bottom"
                ],
                "delta_gross_top_minus_bottom": float(
                    treatment_horizons[horizon]["gross_top_minus_bottom"]
                )
                - float(control_horizons[horizon]["gross_top_minus_bottom"]),
                "control_one_way_turnover": control_horizons[horizon][
                    "one_way_turnover"
                ],
                "treatment_one_way_turnover": treatment_horizons[horizon][
                    "one_way_turnover"
                ],
                "delta_one_way_turnover": float(
                    treatment_horizons[horizon]["one_way_turnover"]
                )
                - float(control_horizons[horizon]["one_way_turnover"]),
            }
            for horizon in treatment_horizons
        },
    }


def _three_seed_summary(
    controls: dict[int, float], treatments: dict[int, float]
) -> dict[str, object]:
    if (
        tuple(sorted(controls)) != STAGE4_SEEDS
        or tuple(sorted(treatments)) != STAGE4_SEEDS
    ):
        raise ValueError("Stage-4 summary requires exact matched seeds 11, 29, and 47")
    control = np.asarray([controls[seed] for seed in STAGE4_SEEDS], dtype=np.float64)
    treatment = np.asarray(
        [treatments[seed] for seed in STAGE4_SEEDS], dtype=np.float64
    )
    delta = treatment - control
    if not np.isfinite(control).all() or not np.isfinite(treatment).all():
        raise ValueError("Stage-4 seed metrics must be finite")
    return {
        "control_by_seed": {str(seed): float(controls[seed]) for seed in STAGE4_SEEDS},
        "treatment_by_seed": {
            str(seed): float(treatments[seed]) for seed in STAGE4_SEEDS
        },
        "delta_by_seed": {
            str(seed): float(treatments[seed] - controls[seed]) for seed in STAGE4_SEEDS
        },
        "control_mean": float(np.mean(control)),
        "treatment_mean": float(np.mean(treatment)),
        "mean_delta": float(np.mean(delta)),
        "median_delta": float(np.median(delta)),
        "delta_standard_deviation": float(np.std(delta, ddof=1)),
        "nonnegative_seed_delta_count": int(np.sum(delta >= -1e-15)),
        "negative_seed_delta_count": int(np.sum(delta < -1e-15)),
    }


def analyze_sweep(
    state_path: Path,
    output_dir: Path,
    *,
    stage3_state_path: Path | None = None,
    slow_audit_path: Path | None = None,
) -> tuple[Path, Path, Path]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    _reject_test_derived_metadata(state, "Stage-4 state")
    if (
        state.get("state_version") != STATE_VERSION
        or state.get("sweep_name") != SWEEP_NAME
        or state.get("status") != "completed"
    ):
        raise ValueError("Analyzer requires a completed Stage-4 state")
    configuration = state.get("configuration")
    jobs = state.get("jobs")
    if not isinstance(configuration, dict) or not isinstance(jobs, list):
        raise ValueError("Stage-4 state is missing configuration or jobs")
    _validate_configuration(configuration)
    source_state = (
        stage3_state_path
        if stage3_state_path is not None
        else Path(str(configuration["source_stage3_state"]))
    )
    audit_config = configuration["training_slow_audit"]
    audit_path = (
        slow_audit_path
        if slow_audit_path is not None
        else Path(str(audit_config["path"]))
    )
    if _sha256(source_state) != configuration["source_stage3_state_sha256"]:
        raise ValueError("Analyzer Stage-3 source state hash disagrees")
    if _sha256(audit_path) != audit_config["sha256"]:
        raise ValueError("Analyzer slow-audit hash disagrees")
    validate_training_slow_audit(audit_path, configuration["feature_store"])
    _validated_stage3_adoptions(source_state, configuration)

    expected_order = tuple(
        (job["logical_configuration"], job["feature_ablation"], job["seed"])
        for job in stage4_jobs()
    )
    actual_order = tuple(
        (
            job.get("logical_configuration"),
            job.get("feature_ablation"),
            job.get("seed"),
        )
        for job in jobs
        if isinstance(job, dict)
    )
    if actual_order != expected_order or any(
        job.get("status") != "completed" for job in jobs
    ):
        raise ValueError("Analyzer requires the exact completed six-job matrix")

    validated: dict[
        tuple[str, int],
        tuple[
            dict[str, object], Path, dict[str, object], pl.DataFrame, dict[str, object]
        ],
    ] = {}
    resolved_dirs: list[Path] = []
    reference_keys: pl.DataFrame | None = None
    for job, base in zip(jobs, stage4_jobs(), strict=True):
        if any(
            job.get(field) != base[field]
            for field in (
                "logical_configuration",
                "context_ablation",
                "context_ablation_metadata",
                "feature_ablation",
                "seed",
                "command",
                "serialized_job_specification",
                "job_specification_sha256",
            )
        ):
            raise ValueError("Analyzer found mutable Stage-4 job specification")
        logical = str(job["logical_configuration"])
        feature_key = str(job["feature_ablation"])
        seed = int(job["seed"])
        expected_origin = (
            "adopted_stage3" if logical == "full_slow" else "trained_stage4"
        )
        if job.get("result_origin") != expected_origin:
            raise ValueError(f"Stage-4 job has wrong origin: {logical}/{seed}")
        run_dir, score, manifest_sha, hashes, identity_source = (
            _completed_job_artifacts(job, configuration)
        )
        expected_commit = (
            configuration["source_stage3_producing_commit"]
            if logical == "full_slow"
            else configuration["orchestrator_git_commit_sha"]
        )
        if job.get("producing_git_commit_sha") != expected_commit:
            raise ValueError(
                f"Stage-4 job has wrong producing commit: {logical}/{seed}"
            )
        checked_score, checked_hashes, checked_source = validate_stage4_completed_run(
            run_dir,
            configuration,
            feature_key,
            seed,
            str(expected_commit),
            allow_legacy_none=logical == "full_slow",
        )
        if (
            not math.isclose(score, checked_score, rel_tol=0.0, abs_tol=1e-12)
            or manifest_sha != hashes["run_manifest.json"]
            or hashes != checked_hashes
            or identity_source != checked_source
        ):
            raise ValueError(f"Stage-4 artifact validation drifted: {logical}/{seed}")
        manifest = json.loads(
            (run_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        daily = _read_validation_daily_metrics(run_dir)
        keys = daily.select("trade_date", "date_idx", "horizon_minutes")
        if reference_keys is None:
            reference_keys = keys
        elif not keys.equals(reference_keys):
            raise ValueError(
                "Stage-4 runs do not use identical validation observations"
            )
        periods = _periods(daily)
        _validate_metrics_json(run_dir, periods["full_validation"])
        validated[(logical, seed)] = (job, run_dir, manifest, daily, periods)
        resolved_dirs.append(run_dir.resolve())
    if len(set(resolved_dirs)) != 6:
        raise ValueError("Analyzer requires six distinct run directories")

    run_results: list[dict[str, object]] = []
    run_csv: list[dict[str, object]] = []
    for logical in STAGE4_LOGICAL_CONFIGURATION_ORDER:
        for seed in STAGE4_SEEDS:
            job, run_dir, manifest, daily, periods = validated[(logical, seed)]
            training = _training_diagnostics(run_dir, manifest)
            feature_metadata = manifest.get("feature_ablation")
            if logical == "full_slow" and feature_metadata is None:
                feature_metadata = configuration["feature_ablation_metadata_by_key"][
                    "none"
                ]
            result = {
                "logical_configuration": logical,
                "seed": seed,
                "result_origin": job["result_origin"],
                "run_dir": str(run_dir),
                "context_ablation": FROZEN_CONTEXT_ABLATION,
                "context_ablation_specification": manifest["context_ablation"],
                "feature_ablation": job["feature_ablation"],
                "feature_ablation_specification": feature_metadata,
                "feature_ablation_identity_source": job[
                    "feature_ablation_identity_source"
                ],
                "primary_validation_ic": periods["full_validation"]["primary_ic"],
                "periods": periods,
                "training": training,
                "validation_coverage": {
                    "date_count": daily["trade_date"].n_unique(),
                    "horizon_count": daily["horizon_minutes"].n_unique(),
                    "daily_horizon_row_count": daily.height,
                },
                "producing_git_commit_sha": manifest["git_commit_sha"],
                "run_manifest_sha256": job["run_manifest_sha256"],
                "output_sha256": job["output_sha256"],
                "job_specification_sha256": job["job_specification_sha256"],
            }
            run_results.append(result)
            row = {
                "logical_configuration": logical,
                "seed": seed,
                "result_origin": job["result_origin"],
                "context_ablation": FROZEN_CONTEXT_ABLATION,
                "feature_ablation": job["feature_ablation"],
                "primary_validation_ic": periods["full_validation"]["primary_ic"],
                "first_half_primary_ic": periods["first_half"]["primary_ic"],
                "latest_half_primary_ic": periods["latest_half"]["primary_ic"],
                "mean_gross_top_minus_bottom": periods["full_validation"][
                    "mean_gross_top_minus_bottom"
                ],
                "mean_one_way_turnover": periods["full_validation"][
                    "mean_one_way_turnover"
                ],
                "validation_date_count": daily["trade_date"].n_unique(),
                "validation_daily_horizon_row_count": daily.height,
                **training,
                "run_dir": str(run_dir),
                "producing_git_commit_sha": manifest["git_commit_sha"],
                "run_manifest_sha256": job["run_manifest_sha256"],
                "job_specification_sha256": job["job_specification_sha256"],
            }
            for horizon in HORIZONS:
                values = periods["full_validation"]["horizons"][f"{horizon}m"]
                row[f"ic_{horizon}m"] = values["spearman_ic"]
                row[f"gross_top_minus_bottom_{horizon}m"] = values[
                    "gross_top_minus_bottom"
                ]
                row[f"one_way_turnover_{horizon}m"] = values["one_way_turnover"]
            run_csv.append(row)

    matched: list[dict[str, object]] = []
    matched_csv: list[dict[str, object]] = []
    for seed in STAGE4_SEEDS:
        control = validated[("full_slow", seed)][4]
        treatment = validated[("drop_slow_low_prior", seed)][4]
        comparisons = {
            period: _period_delta(treatment[period], control[period])
            for period in ("full_validation", "first_half", "latest_half")
        }
        row = {
            "seed": seed,
            "delta_definition": "treatment_minus_control",
            "control_run_dir": str(validated[("full_slow", seed)][1]),
            "treatment_run_dir": str(validated[("drop_slow_low_prior", seed)][1]),
            "periods": comparisons,
        }
        matched.append(row)
        full = comparisons["full_validation"]
        first = comparisons["first_half"]
        latest = comparisons["latest_half"]
        csv_row = {
            "seed": seed,
            "delta_ic": full["delta_ic"],
            "control_ic": full["control_primary_ic"],
            "treatment_ic": full["treatment_primary_ic"],
            "first_half_delta_ic": first["delta_ic"],
            "latest_half_delta_ic": latest["delta_ic"],
            "delta_gross_top_minus_bottom": full["delta_mean_gross_top_minus_bottom"],
            "delta_one_way_turnover": full["delta_mean_one_way_turnover"],
            "validation_date_count": full["date_count"],
            "control_run_dir": row["control_run_dir"],
            "treatment_run_dir": row["treatment_run_dir"],
        }
        for horizon in HORIZONS:
            csv_row[f"delta_ic_{horizon}m"] = full["horizons"][f"{horizon}m"][
                "delta_ic"
            ]
        matched_csv.append(csv_row)

    full_summary = _three_seed_summary(
        {
            seed: float(
                validated[("full_slow", seed)][4]["full_validation"]["primary_ic"]
            )
            for seed in STAGE4_SEEDS
        },
        {
            seed: float(
                validated[("drop_slow_low_prior", seed)][4]["full_validation"][
                    "primary_ic"
                ]
            )
            for seed in STAGE4_SEEDS
        },
    )
    period_summaries = {
        period: _three_seed_summary(
            {
                seed: float(validated[("full_slow", seed)][4][period]["primary_ic"])
                for seed in STAGE4_SEEDS
            },
            {
                seed: float(
                    validated[("drop_slow_low_prior", seed)][4][period]["primary_ic"]
                )
                for seed in STAGE4_SEEDS
            },
        )
        for period in ("first_half", "latest_half")
    }
    horizon_summaries = {
        f"{horizon}m": _three_seed_summary(
            {
                seed: float(
                    validated[("full_slow", seed)][4]["full_validation"]["horizons"][
                        f"{horizon}m"
                    ]["spearman_ic"]
                )
                for seed in STAGE4_SEEDS
            },
            {
                seed: float(
                    validated[("drop_slow_low_prior", seed)][4]["full_validation"][
                        "horizons"
                    ][f"{horizon}m"]["spearman_ic"]
                )
                for seed in STAGE4_SEEDS
            },
        )
        for horizon in HORIZONS
    }
    operational_summaries = {
        metric: _three_seed_summary(
            {
                seed: float(validated[("full_slow", seed)][4]["full_validation"][field])
                for seed in STAGE4_SEEDS
            },
            {
                seed: float(
                    validated[("drop_slow_low_prior", seed)][4]["full_validation"][
                        field
                    ]
                )
                for seed in STAGE4_SEEDS
            },
        )
        for metric, field in (
            ("gross_top_minus_bottom", "mean_gross_top_minus_bottom"),
            ("one_way_turnover", "mean_one_way_turnover"),
        )
    }

    control_daily = np.mean(
        np.stack(
            [_daily_primary(validated[("full_slow", seed)][3]) for seed in STAGE4_SEEDS]
        ),
        axis=0,
    )
    treatment_daily = np.mean(
        np.stack(
            [
                _daily_primary(validated[("drop_slow_low_prior", seed)][3])
                for seed in STAGE4_SEEDS
            ]
        ),
        axis=0,
    )
    bootstrap = paired_moving_block_bootstrap(control_daily, treatment_daily)
    horizon_bootstraps = {}
    for horizon in HORIZONS:
        control = np.mean(
            np.stack(
                [
                    validated[("full_slow", seed)][3]
                    .filter(pl.col("horizon_minutes") == horizon)
                    .sort("trade_date")["spearman_ic"]
                    .to_numpy()
                    for seed in STAGE4_SEEDS
                ]
            ),
            axis=0,
        )
        treatment = np.mean(
            np.stack(
                [
                    validated[("drop_slow_low_prior", seed)][3]
                    .filter(pl.col("horizon_minutes") == horizon)
                    .sort("trade_date")["spearman_ic"]
                    .to_numpy()
                    for seed in STAGE4_SEEDS
                ]
            ),
            axis=0,
        )
        horizon_bootstraps[f"{horizon}m"] = paired_moving_block_bootstrap(
            control, treatment
        )

    latest_deltas = np.asarray(
        [float(row["periods"]["latest_half"]["delta_ic"]) for row in matched]
    )
    horizon_consistent_regression = {
        f"{horizon}m": all(
            float(
                row["periods"]["full_validation"]["horizons"][f"{horizon}m"]["delta_ic"]
            )
            < -1e-15
            for row in matched
        )
        for horizon in HORIZONS
    }
    spread_deltas = np.asarray(
        [
            float(
                row["periods"]["full_validation"]["delta_mean_gross_top_minus_bottom"]
            )
            for row in matched
        ]
    )
    turnover_deltas = np.asarray(
        [
            float(row["periods"]["full_validation"]["delta_mean_one_way_turnover"])
            for row in matched
        ]
    )
    evidence = {
        "mean_matched_seed_delta_nonnegative": full_summary["mean_delta"] >= -1e-15,
        "nonnegative_seed_delta_count": full_summary["nonnegative_seed_delta_count"],
        "latest_half_consistent_regression": bool(np.all(latest_deltas < -1e-15)),
        "horizon_consistent_regression": horizon_consistent_regression,
        "operational_metrics": {
            "gross_spread_consistent_regression": bool(np.all(spread_deltas < -1e-15)),
            "turnover_consistent_increase": bool(np.all(turnover_deltas > 1e-15)),
            "coverage_changed": False,
        },
    }
    evidence["operational_metrics_deteriorated"] = bool(
        evidence["operational_metrics"]["gross_spread_consistent_regression"]
        or evidence["operational_metrics"]["turnover_consistent_increase"]
        or evidence["operational_metrics"]["coverage_changed"]
    )

    summary = {
        "state_file": str(state_path.resolve()),
        "state_file_sha256": _sha256(state_path),
        "logical_job_count": 6,
        "seeds": list(STAGE4_SEEDS),
        "control": "full_slow",
        "treatment": "drop_slow_low_prior",
        "delta_definition": "treatment_minus_control",
        "context_ablation": FROZEN_CONTEXT_ABLATION,
        "feature_ablation_registry": configuration["feature_ablation_metadata_by_key"],
        "feature_store_identity": configuration["feature_store"],
        "split_boundaries": configuration["split_boundaries"],
        "source_stage3_state": str(source_state),
        "source_stage3_state_sha256": configuration["source_stage3_state_sha256"],
        "training_slow_audit": {
            **configuration["training_slow_audit"],
            "resolved_path": str(audit_path),
        },
        "orchestrator_git_commit_sha": configuration["orchestrator_git_commit_sha"],
        "result_origin_counts": {"adopted_stage3": 3, "trained_stage4": 3},
        "test_metrics_accessed": False,
        "validation_coverage": {
            "date_count": 244,
            "horizons": list(HORIZONS),
            "daily_horizon_row_count_per_run": 244 * len(HORIZONS),
            "identical_observations_across_all_runs": True,
        },
        "runs": run_results,
        "matched_seeds": matched,
        "three_seed_summary": full_summary,
        "period_summaries": period_summaries,
        "horizon_summaries": horizon_summaries,
        "operational_summaries": operational_summaries,
        "paired_moving_block_bootstrap": {
            "primary_ic": bootstrap,
            "horizons": horizon_bootstraps,
            "aggregation": (
                "Daily treatment-minus-control effects are computed within each "
                "matched seed and averaged across the three seeds before resampling."
            ),
            "block_trading_days": BOOTSTRAP_BLOCK_TRADING_DAYS,
            "replications": BOOTSTRAP_REPLICATIONS,
            "seed": BOOTSTRAP_SEED,
        },
        "decision_evidence": evidence,
        "uncertainty_interpretation": (
            "The moving-block interval describes the validation-date time series "
            "after matched-seed averaging. The three seed deltas separately show "
            "optimization sensitivity; they are not treated as a large sample."
        ),
        "selection": None,
        "automatic_winner_selection": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / SUMMARY_JSON
    runs_path = output_dir / RUNS_CSV
    matched_path = output_dir / MATCHED_CSV
    _atomic_write_json(json_path, summary)
    _atomic_write_csv(runs_path, run_csv)
    _atomic_write_csv(matched_path, matched_csv)
    return json_path, runs_path, matched_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--stage3-state", type=Path)
    parser.add_argument("--slow-audit", type=Path)
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
    paths = analyze_sweep(
        state_file,
        output_dir,
        stage3_state_path=(
            args.stage3_state.resolve() if args.stage3_state is not None else None
        ),
        slow_audit_path=(
            args.slow_audit.resolve() if args.slow_audit is not None else None
        ),
    )
    print(f"Wrote Stage-4 JSON summary: {paths[0]}")
    print(f"Wrote Stage-4 run CSV: {paths[1]}")
    print(f"Wrote Stage-4 matched-seed CSV: {paths[2]}")


if __name__ == "__main__":
    main()
