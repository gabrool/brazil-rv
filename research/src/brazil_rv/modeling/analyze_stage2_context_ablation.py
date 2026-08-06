from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from .analyze_context_ablation import (
    BOOTSTRAP_BLOCK_TRADING_DAYS,
    BOOTSTRAP_REPLICATIONS,
    BOOTSTRAP_SEED,
    paired_moving_block_bootstrap,
)
from .contract import (
    FEATURE_CONTRACT_VERSION,
    HORIZONS,
    VALIDATION_END,
    VALIDATION_START,
    SplitBoundaries,
)
from .stage2_context_ablation import (
    ADOPTED_STAGE1_KEYS,
    STATE_VERSION,
    STAGE1_PRODUCING_COMMIT,
    STAGE2_CONTEXT_ABLATION_ORDER,
    STAGE2_SEEDS,
    SWEEP_NAME,
    _training_semantics,
    validate_stage2_completed_run,
)

FIRST_HALF_END = date(2024, 12, 31)
LATEST_HALF_START = date(2025, 1, 1)
_SUMMARY_JSON = "stage2_context_ablation_summary.json"
_SUMMARY_CSV = "stage2_context_ablation_summary.csv"
_DAILY_COLUMNS = (
    "trade_date",
    "date_idx",
    "horizon_minutes",
    "spearman_ic",
    "top_return",
    "bottom_return",
    "top_minus_bottom",
    "long_only_top",
    "one_way_turnover",
)


def _read_validation_daily_metrics(run_dir: Path) -> pl.DataFrame:
    frame = pl.read_parquet(run_dir / "validation_daily_metrics.parquet")
    if not set(_DAILY_COLUMNS) <= set(frame.columns):
        raise ValueError(f"Validation daily metrics are incomplete: {run_dir}")
    frame = frame.select(_DAILY_COLUMNS).sort("trade_date", "horizon_minutes")
    if frame.select(
        pl.struct("trade_date", "horizon_minutes").is_duplicated().any()
    ).item():
        raise ValueError(f"Validation daily metrics contain duplicates: {run_dir}")
    if frame.height != 244 * len(HORIZONS):
        raise ValueError(f"Validation daily metrics have the wrong size: {run_dir}")
    dates = frame.get_column("trade_date")
    if (
        dates.n_unique() != 244
        or dates.min() != VALIDATION_START
        or dates.max() != VALIDATION_END
    ):
        raise ValueError(f"Validation daily metrics have the wrong dates: {run_dir}")
    horizon_rows = frame.group_by("trade_date").agg(
        pl.col("horizon_minutes").sort().alias("horizons")
    )
    if any(tuple(values) != HORIZONS for values in horizon_rows["horizons"]):
        raise ValueError(f"Validation daily metrics have missing horizons: {run_dir}")
    for column in _DAILY_COLUMNS[3:]:
        if not np.isfinite(frame[column].to_numpy()).all():
            raise ValueError(
                f"Validation daily metrics contain nonfinite {column}: {run_dir}"
            )
    return frame


def _mean(frame: pl.DataFrame, column: str) -> float:
    value = float(frame[column].mean())
    if not math.isfinite(value):
        raise ValueError(f"Cannot summarize nonfinite validation column: {column}")
    return value


def _period_metrics(frame: pl.DataFrame, start: date, end: date) -> dict[str, object]:
    period = frame.filter(pl.col("trade_date").is_between(start, end))
    if period.is_empty():
        raise ValueError(f"No validation rows exist between {start} and {end}")
    horizons: dict[str, dict[str, float]] = {}
    for horizon in HORIZONS:
        selected = period.filter(pl.col("horizon_minutes") == horizon)
        horizons[f"{horizon}m"] = {
            "spearman_ic": _mean(selected, "spearman_ic"),
            "gross_top_minus_bottom": _mean(selected, "top_minus_bottom"),
            "one_way_turnover": _mean(selected, "one_way_turnover"),
        }
    return {
        "start": str(start),
        "end": str(end),
        "date_count": period["trade_date"].n_unique(),
        "primary_ic": float(
            np.mean([values["spearman_ic"] for values in horizons.values()])
        ),
        "mean_gross_top_minus_bottom": float(
            np.mean([values["gross_top_minus_bottom"] for values in horizons.values()])
        ),
        "mean_one_way_turnover": float(
            np.mean([values["one_way_turnover"] for values in horizons.values()])
        ),
        "horizons": horizons,
    }


def _with_baseline_delta(
    current: dict[str, object], baseline: dict[str, object]
) -> dict[str, object]:
    result = dict(current)
    result["primary_delta_vs_same_seed_baseline"] = float(
        current["primary_ic"]
    ) - float(baseline["primary_ic"])
    result["gross_top_minus_bottom_delta_vs_same_seed_baseline"] = float(
        current["mean_gross_top_minus_bottom"]
    ) - float(baseline["mean_gross_top_minus_bottom"])
    result["one_way_turnover_delta_vs_same_seed_baseline"] = float(
        current["mean_one_way_turnover"]
    ) - float(baseline["mean_one_way_turnover"])
    current_horizons = current["horizons"]
    baseline_horizons = baseline["horizons"]
    if not isinstance(current_horizons, dict) or not isinstance(
        baseline_horizons, dict
    ):
        raise ValueError("Period horizon metrics are malformed")
    result["horizons"] = {
        horizon: {
            **values,
            "spearman_delta_vs_same_seed_baseline": float(values["spearman_ic"])
            - float(baseline_horizons[horizon]["spearman_ic"]),
            "gross_top_minus_bottom_delta_vs_same_seed_baseline": float(
                values["gross_top_minus_bottom"]
            )
            - float(baseline_horizons[horizon]["gross_top_minus_bottom"]),
            "one_way_turnover_delta_vs_same_seed_baseline": float(
                values["one_way_turnover"]
            )
            - float(baseline_horizons[horizon]["one_way_turnover"]),
        }
        for horizon, values in current_horizons.items()
    }
    return result


def _daily_primary(frame: pl.DataFrame) -> np.ndarray:
    return (
        frame.group_by("trade_date")
        .agg(pl.col("spearman_ic").mean().alias("primary_ic"))
        .sort("trade_date")["primary_ic"]
        .to_numpy()
    )


def _validate_metrics_json(run_dir: Path, full: dict[str, object]) -> None:
    metrics = json.loads(
        (run_dir / "validation_metrics.json").read_text(encoding="utf-8")
    )
    if not math.isclose(
        float(metrics["primary_score"]),
        float(full["primary_ic"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"Validation JSON and daily primary IC disagree: {run_dir}")
    expected = {f"{int(row['horizon_minutes'])}m": row for row in metrics["horizons"]}
    horizons = full["horizons"]
    if not isinstance(horizons, dict) or expected.keys() != horizons.keys():
        raise ValueError(f"Validation JSON has the wrong horizons: {run_dir}")
    for horizon, row in expected.items():
        actual = horizons[horizon]
        fields = {
            "mean_daily_spearman_ic": "spearman_ic",
            "mean_top_minus_bottom": "gross_top_minus_bottom",
            "mean_one_way_turnover": "one_way_turnover",
        }
        if any(
            not math.isclose(
                float(row[source]),
                float(actual[target]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for source, target in fields.items()
        ):
            raise ValueError(
                f"Validation JSON and daily horizon metrics disagree: {run_dir}"
            )


def _training_diagnostics(
    run_dir: Path, manifest: dict[str, object]
) -> dict[str, object]:
    history = pl.read_csv(run_dir / "history.csv")
    required = {"epoch", "optimizer_steps", "epoch_seconds"}
    if not required <= set(history.columns) or history.is_empty():
        raise ValueError(f"Training history is incomplete: {run_dir}")
    if not np.isfinite(history["epoch_seconds"].to_numpy()).all():
        raise ValueError(f"Training history has nonfinite durations: {run_dir}")
    optimizer_updates = int(history["optimizer_steps"].sum())
    if optimizer_updates != int(manifest["successful_optimizer_updates"]):
        raise ValueError(f"Training history update count disagrees: {run_dir}")
    return {
        "best_epoch": int(manifest["best_epoch"]),
        "stopped_epoch": int(manifest["stopped_epoch"]),
        "optimizer_updates": optimizer_updates,
        "training_duration_seconds": float(manifest["training_duration_seconds"]),
        "history_epoch_count": history.height,
        "history_epoch_seconds_total": float(history["epoch_seconds"].sum()),
        "history_epoch_seconds_mean": float(history["epoch_seconds"].mean()),
        "history_epoch_seconds_median": float(history["epoch_seconds"].median()),
        "history_epoch_seconds_minimum": float(history["epoch_seconds"].min()),
        "history_epoch_seconds_maximum": float(history["epoch_seconds"].max()),
    }


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _delta_summary(values_by_seed: dict[int, float]) -> dict[str, object]:
    if tuple(sorted(values_by_seed)) != STAGE2_SEEDS:
        raise ValueError("Across-seed summary requires exactly seeds 11, 29, and 47")
    values = np.asarray(
        [values_by_seed[seed] for seed in STAGE2_SEEDS], dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError("Across-seed deltas must be finite")
    tolerance = 1e-15
    return {
        "paired_delta_by_seed": {
            str(seed): float(values_by_seed[seed]) for seed in STAGE2_SEEDS
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
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
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


def analyze_sweep(state_path: Path, output_dir: Path) -> tuple[Path, Path]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (
        state.get("state_version") != STATE_VERSION
        or state.get("sweep_name") != SWEEP_NAME
        or state.get("status") != "completed"
    ):
        raise ValueError("Analyzer requires a completed Stage-2 state")
    configuration = state.get("configuration")
    jobs = state.get("jobs")
    if not isinstance(configuration, dict) or not isinstance(jobs, list):
        raise ValueError("Stage-2 state is missing configuration or jobs")
    expected_configuration = {
        "feature_contract": FEATURE_CONTRACT_VERSION,
        "training_semantics": _training_semantics(),
        "split_boundaries": {
            key: str(value) for key, value in asdict(SplitBoundaries()).items()
        },
        "configuration_order": list(STAGE2_CONTEXT_ABLATION_ORDER),
        "seeds": list(STAGE2_SEEDS),
        "required_stage1_producing_commit": STAGE1_PRODUCING_COMMIT,
    }
    for field, value in expected_configuration.items():
        if configuration.get(field) != value:
            raise ValueError(f"Stage-2 configuration is incompatible: {field}")
    if (
        not isinstance(configuration.get("feature_store"), dict)
        or not isinstance(configuration.get("orchestrator_git_commit_sha"), str)
        or not isinstance(configuration.get("source_stage1_state"), str)
    ):
        raise ValueError("Stage-2 configuration is missing provenance")
    expected_order = tuple(
        (key, seed) for key in STAGE2_CONTEXT_ABLATION_ORDER for seed in STAGE2_SEEDS
    )
    actual_order = tuple(
        (job.get("context_ablation"), job.get("seed"))
        for job in jobs
        if isinstance(job, dict)
    )
    if actual_order != expected_order:
        raise ValueError("Analyzer requires the exact canonical 18-job matrix")
    if any(job.get("status") != "completed" for job in jobs):
        raise ValueError("Analyzer refuses a partially completed matrix")
    run_dirs = tuple(Path(str(job.get("run_dir"))).resolve() for job in jobs)
    if len(set(run_dirs)) != 18:
        raise ValueError("Analyzer requires 18 distinct run directories")

    validated: dict[
        tuple[str, int], tuple[dict[str, object], Path, dict[str, object], pl.DataFrame]
    ] = {}
    for job, run_dir in zip(jobs, run_dirs, strict=True):
        key = str(job["context_ablation"])
        seed = int(job["seed"])
        should_be_adopted = seed == 29 and key in ADOPTED_STAGE1_KEYS
        expected_origin = "adopted_stage1" if should_be_adopted else "trained_stage2"
        if job.get("result_origin") != expected_origin:
            raise ValueError(f"Stage-2 job has the wrong result origin: {key}/{seed}")
        producing_commit = (
            STAGE1_PRODUCING_COMMIT
            if should_be_adopted
            else str(configuration["orchestrator_git_commit_sha"])
        )
        if job.get("producing_git_commit_sha") != producing_commit:
            raise ValueError(
                f"Stage-2 job has the wrong producing commit: {key}/{seed}"
            )
        if should_be_adopted:
            source = job.get("source_stage1_job")
            if (
                job.get("source_stage1_state")
                != configuration.get("source_stage1_state")
                or not isinstance(source, dict)
                or source.get("context_ablation") != key
                or source.get("seed") != 29
            ):
                raise ValueError(f"Adopted Stage-2 provenance is invalid: {key}/{seed}")
        elif (
            job.get("source_stage1_state") is not None
            or job.get("source_stage1_job") is not None
        ):
            raise ValueError(f"New Stage-2 run has Stage-1 provenance: {key}/{seed}")
        score = validate_stage2_completed_run(
            run_dir, configuration, key, seed, producing_commit
        )
        if not math.isclose(
            score,
            float(job.get("primary_validation_ic")),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Stage-2 state score disagrees for {key}/{seed}")
        manifest_path = run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        daily = _read_validation_daily_metrics(run_dir)
        full = _period_metrics(daily, VALIDATION_START, VALIDATION_END)
        _validate_metrics_json(run_dir, full)
        validated[(key, seed)] = (job, run_dir, manifest, daily)

    baseline_periods: dict[int, dict[str, dict[str, object]]] = {}
    baseline_daily: dict[int, np.ndarray] = {}
    for seed in STAGE2_SEEDS:
        baseline = validated[("none", seed)][3]
        baseline_periods[seed] = {
            "full_validation": _period_metrics(
                baseline, VALIDATION_START, VALIDATION_END
            ),
            "first_half": _period_metrics(baseline, VALIDATION_START, FIRST_HALF_END),
            "latest_half": _period_metrics(baseline, LATEST_HALF_START, VALIDATION_END),
        }
        baseline_daily[seed] = _daily_primary(baseline)

    run_results: list[dict[str, object]] = []
    csv_rows: list[dict[str, object]] = []
    results_by_identity: dict[tuple[str, int], dict[str, object]] = {}
    for key in STAGE2_CONTEXT_ABLATION_ORDER:
        for seed in STAGE2_SEEDS:
            job, run_dir, manifest, daily = validated[(key, seed)]
            raw_periods = {
                "full_validation": _period_metrics(
                    daily, VALIDATION_START, VALIDATION_END
                ),
                "first_half": _period_metrics(daily, VALIDATION_START, FIRST_HALF_END),
                "latest_half": _period_metrics(
                    daily, LATEST_HALF_START, VALIDATION_END
                ),
            }
            periods = {
                name: _with_baseline_delta(period, baseline_periods[seed][name])
                for name, period in raw_periods.items()
            }
            bootstrap = paired_moving_block_bootstrap(
                baseline_daily[seed], _daily_primary(daily)
            )
            training = _training_diagnostics(run_dir, manifest)
            result = {
                "context_ablation": key,
                "seed": seed,
                "result_origin": job["result_origin"],
                "run_dir": str(run_dir),
                "run_manifest_path": str((run_dir / "run_manifest.json").resolve()),
                "run_manifest_sha256": _manifest_sha256(run_dir / "run_manifest.json"),
                "producing_git_commit_sha": manifest["git_commit_sha"],
                "source_stage1_state": job["source_stage1_state"],
                "source_stage1_job": job["source_stage1_job"],
                "feature_store_identity": configuration["feature_store"],
                "split_boundaries": manifest["split_boundaries"],
                "periods": periods,
                "training": training,
                "within_trained_model_daily_bootstrap": bootstrap,
            }
            run_results.append(result)
            results_by_identity[(key, seed)] = result

    configuration_results: list[dict[str, object]] = []
    configuration_summaries: dict[str, dict[str, object]] = {}
    for key in STAGE2_CONTEXT_ABLATION_ORDER:
        seed_results = [results_by_identity[(key, seed)] for seed in STAGE2_SEEDS]
        primary_deltas = {
            seed: float(
                results_by_identity[(key, seed)]["periods"]["full_validation"][
                    "primary_delta_vs_same_seed_baseline"
                ]
            )
            for seed in STAGE2_SEEDS
        }
        horizon_delta_summaries = {
            f"{horizon}m": _delta_summary(
                {
                    seed: float(
                        results_by_identity[(key, seed)]["periods"]["full_validation"][
                            "horizons"
                        ][f"{horizon}m"]["spearman_delta_vs_same_seed_baseline"]
                    )
                    for seed in STAGE2_SEEDS
                }
            )
            for horizon in HORIZONS
        }
        period_delta_summaries = {
            period: _delta_summary(
                {
                    seed: float(
                        results_by_identity[(key, seed)]["periods"][period][
                            "primary_delta_vs_same_seed_baseline"
                        ]
                    )
                    for seed in STAGE2_SEEDS
                }
            )
            for period in ("first_half", "latest_half")
        }
        diagnostic_delta_summaries = {
            "gross_top_minus_bottom": _delta_summary(
                {
                    seed: float(
                        results_by_identity[(key, seed)]["periods"]["full_validation"][
                            "gross_top_minus_bottom_delta_vs_same_seed_baseline"
                        ]
                    )
                    for seed in STAGE2_SEEDS
                }
            ),
            "one_way_turnover": _delta_summary(
                {
                    seed: float(
                        results_by_identity[(key, seed)]["periods"]["full_validation"][
                            "one_way_turnover_delta_vs_same_seed_baseline"
                        ]
                    )
                    for seed in STAGE2_SEEDS
                }
            ),
        }
        summary = {
            "context_ablation": key,
            "absolute_validation_primary_ic_by_seed": {
                str(seed): float(
                    results_by_identity[(key, seed)]["periods"]["full_validation"][
                        "primary_ic"
                    ]
                )
                for seed in STAGE2_SEEDS
            },
            "primary_delta_across_training_seeds": _delta_summary(primary_deltas),
            "horizon_delta_across_training_seeds": horizon_delta_summaries,
            "period_delta_across_training_seeds": period_delta_summaries,
            "diagnostic_delta_across_training_seeds": diagnostic_delta_summaries,
            "seed_results": seed_results,
        }
        configuration_results.append(summary)
        configuration_summaries[key] = summary

    for result in run_results:
        key = str(result["context_ablation"])
        full = result["periods"]["full_validation"]
        first = result["periods"]["first_half"]
        latest = result["periods"]["latest_half"]
        training = result["training"]
        bootstrap = result["within_trained_model_daily_bootstrap"]
        aggregate = configuration_summaries[key]["primary_delta_across_training_seeds"]
        row: dict[str, object] = {
            "context_ablation": key,
            "seed": result["seed"],
            "result_origin": result["result_origin"],
            "primary_validation_ic": full["primary_ic"],
            "primary_delta_vs_same_seed_baseline": full[
                "primary_delta_vs_same_seed_baseline"
            ],
            "first_half_primary_ic": first["primary_ic"],
            "first_half_delta_vs_same_seed_baseline": first[
                "primary_delta_vs_same_seed_baseline"
            ],
            "latest_half_primary_ic": latest["primary_ic"],
            "latest_half_delta_vs_same_seed_baseline": latest[
                "primary_delta_vs_same_seed_baseline"
            ],
            "mean_gross_top_minus_bottom": full["mean_gross_top_minus_bottom"],
            "gross_top_minus_bottom_delta_vs_same_seed_baseline": full[
                "gross_top_minus_bottom_delta_vs_same_seed_baseline"
            ],
            "mean_one_way_turnover": full["mean_one_way_turnover"],
            "one_way_turnover_delta_vs_same_seed_baseline": full[
                "one_way_turnover_delta_vs_same_seed_baseline"
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
            **training,
            "run_dir": result["run_dir"],
            "producing_git_commit_sha": result["producing_git_commit_sha"],
            "run_manifest_sha256": result["run_manifest_sha256"],
        }
        for horizon in HORIZONS:
            horizon_metrics = full["horizons"][f"{horizon}m"]
            row.update(
                {
                    f"ic_{horizon}m": horizon_metrics["spearman_ic"],
                    f"delta_ic_{horizon}m_vs_same_seed_baseline": horizon_metrics[
                        "spearman_delta_vs_same_seed_baseline"
                    ],
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
        "logical_job_count": 18,
        "configuration_count": 6,
        "seeds": list(STAGE2_SEEDS),
        "configuration_order": list(STAGE2_CONTEXT_ABLATION_ORDER),
        "feature_store_identity": configuration["feature_store"],
        "split_boundaries": configuration["split_boundaries"],
        "orchestrator_git_commit_sha": configuration["orchestrator_git_commit_sha"],
        "source_stage1_state": configuration["source_stage1_state"],
        "uncertainty_interpretation": {
            "within_trained_model_daily_time_series": {
                "method": "paired moving-block bootstrap of daily primary IC differences",
                "block_trading_days": BOOTSTRAP_BLOCK_TRADING_DAYS,
                "replications": BOOTSTRAP_REPLICATIONS,
                "seed": BOOTSTRAP_SEED,
            },
            "across_training_seeds": (
                "The three paired seed deltas, their sign consistency, and their range "
                "are primary evidence. Three seeds do not support a conclusive "
                "conventional seed-level significance test."
            ),
        },
        "runs": run_results,
        "configurations": configuration_results,
    }
    _atomic_write_json(json_path, summary)
    _atomic_write_csv(csv_path, csv_rows)
    return json_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-file", type=Path, required=True)
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
    json_path, csv_path = analyze_sweep(state_file, output_dir)
    print(f"Wrote Stage-2 JSON summary: {json_path}")
    print(f"Wrote Stage-2 CSV summary: {csv_path}")


if __name__ == "__main__":
    main()
