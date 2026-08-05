from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from .context_ablation import STAGE1_CONTEXT_ABLATION_ORDER, get_context_ablation
from .contract import HORIZONS, VALIDATION_END, VALIDATION_START
from .stage1_context_ablation import DEFAULT_STATE_DIR, validate_completed_run

BOOTSTRAP_BLOCK_TRADING_DAYS = 5
BOOTSTRAP_REPLICATIONS = 10_000
BOOTSTRAP_SEED = 20260805
LATEST_HALF_START = date(2025, 1, 1)
FIRST_HALF_END = date(2024, 12, 31)
_SUMMARY_JSON = "stage1_context_ablation_summary.json"
_SUMMARY_CSV = "stage1_context_ablation_summary.csv"
_COMMON_MANIFEST_FIELDS = (
    "git_commit_sha",
    "resolved_feature_store_path",
    "feature_manifest_contract_version",
    "model_name",
    "model_family",
    "tcn_settings",
    "architecture_constants",
    "parameter_count",
    "optimizer_variant",
    "objective",
    "sam",
    "global_context",
    "global_context_source_hashes",
    "global_context_normalized_store_hashes",
    "split_boundaries",
    "training_constants",
    "optimizer_constants",
    "scheduler_constants",
    "scheduler_steps",
    "physical_microbatch_size",
    "accumulation_steps",
    "effective_batch_size",
    "evaluation_batch_size",
    "num_workers",
    "prefetch_factor",
    "precision",
    "bf16",
    "grad_scaler_used",
    "pytorch_version",
    "cuda_version",
    "hardware",
)


def paired_moving_block_bootstrap(
    baseline_daily_primary: np.ndarray,
    ablation_daily_primary: np.ndarray,
    *,
    replications: int = BOOTSTRAP_REPLICATIONS,
    seed: int = BOOTSTRAP_SEED,
    block_length: int = BOOTSTRAP_BLOCK_TRADING_DAYS,
) -> dict[str, float | int]:
    baseline = np.asarray(baseline_daily_primary, dtype=np.float64)
    ablation = np.asarray(ablation_daily_primary, dtype=np.float64)
    if baseline.ndim != 1 or not np.array_equal(baseline.shape, ablation.shape):
        raise ValueError(
            "Paired bootstrap inputs must be aligned one-dimensional arrays"
        )
    if baseline.size < block_length or replications <= 0:
        raise ValueError("Paired bootstrap fixture is too short or has no replications")
    if not np.isfinite(baseline).all() or not np.isfinite(ablation).all():
        raise ValueError("Paired bootstrap inputs must be finite")
    differences = ablation - baseline
    blocks_per_replication = math.ceil(differences.size / block_length)
    generator = np.random.default_rng(seed)
    starts = generator.integers(
        0,
        differences.size - block_length + 1,
        size=(replications, blocks_per_replication),
    )
    offsets = np.arange(block_length, dtype=np.int64)
    indices = (starts[..., None] + offsets).reshape(replications, -1)
    replicated_means = differences[indices[:, : differences.size]].mean(axis=1)
    lower, upper = np.quantile(replicated_means, (0.025, 0.975))
    return {
        "paired_mean_delta": float(differences.mean()),
        "interval_lower_95": float(lower),
        "interval_upper_95": float(upper),
        "block_trading_days": block_length,
        "replications": replications,
        "bootstrap_seed": seed,
    }


def _manifest_configuration(manifest: dict[str, object]) -> dict[str, object]:
    missing = [field for field in _COMMON_MANIFEST_FIELDS if field not in manifest]
    if missing:
        raise ValueError(f"Run manifest is missing configuration fields: {missing}")
    return {field: manifest[field] for field in _COMMON_MANIFEST_FIELDS}


def _read_daily_metrics(run_dir: Path) -> pl.DataFrame:
    frame = pl.read_parquet(run_dir / "validation_daily_metrics.parquet")
    required = {"trade_date", "date_idx", "horizon_minutes", "spearman_ic"}
    if not required <= set(frame.columns):
        raise ValueError(f"Daily metrics are missing columns for run {run_dir}")
    frame = frame.select(
        "trade_date", "date_idx", "horizon_minutes", "spearman_ic"
    ).sort("trade_date", "horizon_minutes")
    if frame.select(
        pl.struct("trade_date", "horizon_minutes").is_duplicated().any()
    ).item():
        raise ValueError(
            f"Daily metrics contain duplicate date/horizon rows: {run_dir}"
        )
    if frame.height != 244 * len(HORIZONS):
        raise ValueError(
            f"Daily metrics do not contain the full validation matrix: {run_dir}"
        )
    dates = frame.get_column("trade_date")
    if (
        dates.n_unique() != 244
        or dates.min() != VALIDATION_START
        or dates.max() != VALIDATION_END
    ):
        raise ValueError(f"Daily metrics have the wrong validation dates: {run_dir}")
    counts = frame.group_by("trade_date").agg(
        pl.col("horizon_minutes").sort().alias("horizons")
    )
    if any(tuple(values) != HORIZONS for values in counts["horizons"].to_list()):
        raise ValueError(f"Daily metrics have missing or extra horizons: {run_dir}")
    if not np.isfinite(frame["spearman_ic"].to_numpy()).all():
        raise ValueError(f"Daily metrics contain nonfinite IC values: {run_dir}")
    return frame


def _period_metrics(frame: pl.DataFrame, start: date, end: date) -> dict[str, object]:
    period = frame.filter(pl.col("trade_date").is_between(start, end))
    if period.is_empty():
        raise ValueError(f"No validation daily rows exist between {start} and {end}")
    horizons = {
        int(horizon): float(
            period.filter(pl.col("horizon_minutes") == horizon)["spearman_ic"].mean()
        )
        for horizon in HORIZONS
    }
    return {
        "start": str(start),
        "end": str(end),
        "date_count": period["trade_date"].n_unique(),
        "primary_ic": float(np.mean(tuple(horizons.values()))),
        "horizon_ic": {f"{horizon}m": value for horizon, value in horizons.items()},
    }


def _daily_primary(frame: pl.DataFrame) -> np.ndarray:
    return (
        frame.group_by("trade_date")
        .agg(pl.col("spearman_ic").mean().alias("primary_ic"))
        .sort("trade_date")["primary_ic"]
        .to_numpy()
    )


def _validate_metrics_summary(run_dir: Path, full: dict[str, object]) -> None:
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
    expected_horizons = {
        int(row["horizon_minutes"]): float(row["mean_daily_spearman_ic"])
        for row in metrics["horizons"]
    }
    actual_horizons = {
        int(key.removesuffix("m")): float(value)
        for key, value in full["horizon_ic"].items()
    }
    if expected_horizons.keys() != actual_horizons.keys() or any(
        not math.isclose(
            expected_horizons[horizon],
            actual_horizons[horizon],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for horizon in expected_horizons
    ):
        raise ValueError(f"Validation JSON and daily horizon ICs disagree: {run_dir}")


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_write_csv(path: Path, frame: pl.DataFrame) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    frame.write_csv(temporary)
    os.replace(temporary, path)


def analyze_sweep(state_path: Path, output_dir: Path) -> tuple[Path, Path]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "completed":
        raise ValueError("Analyzer requires a completed Stage-1 state")
    configuration = state.get("configuration")
    jobs = state.get("jobs")
    if not isinstance(configuration, dict) or not isinstance(jobs, list):
        raise ValueError("Stage-1 state is missing configuration or jobs")
    if (
        len(jobs) != 25
        or tuple(job.get("context_ablation") for job in jobs if isinstance(job, dict))
        != STAGE1_CONTEXT_ABLATION_ORDER
    ):
        raise ValueError("Analyzer requires the exact canonical 25-run matrix")
    if any(job.get("status") != "completed" for job in jobs):
        raise ValueError("Analyzer refuses a partially completed matrix")
    run_dirs = [Path(str(job.get("run_dir"))).resolve() for job in jobs]
    if len(set(run_dirs)) != 25:
        raise ValueError("Analyzer requires 25 distinct run directories")

    manifests: dict[str, dict[str, object]] = {}
    daily_frames: dict[str, pl.DataFrame] = {}
    baseline_configuration: dict[str, object] | None = None
    baseline_keys: pl.DataFrame | None = None
    for job, run_dir in zip(jobs, run_dirs, strict=True):
        key = str(job["context_ablation"])
        validate_completed_run(run_dir, configuration, key)
        manifest = json.loads(
            (run_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        common = _manifest_configuration(manifest)
        if baseline_configuration is None:
            baseline_configuration = common
        elif common != baseline_configuration:
            raise ValueError(f"Non-ablation run settings differ for {key}")
        frame = _read_daily_metrics(run_dir)
        keys = frame.select("trade_date", "horizon_minutes")
        if baseline_keys is None:
            baseline_keys = keys
        elif not keys.equals(baseline_keys):
            raise ValueError(f"Daily date/horizon alignment differs for {key}")
        manifests[key] = manifest
        daily_frames[key] = frame

    baseline = daily_frames["none"]
    baseline_periods = {
        "full_validation": _period_metrics(baseline, VALIDATION_START, VALIDATION_END),
        "first_half": _period_metrics(baseline, VALIDATION_START, FIRST_HALF_END),
        "latest_half": _period_metrics(baseline, LATEST_HALF_START, VALIDATION_END),
    }
    baseline_daily_primary = _daily_primary(baseline)
    results: list[dict[str, object]] = []
    csv_rows: list[dict[str, object]] = []
    for key, run_dir in zip(STAGE1_CONTEXT_ABLATION_ORDER, run_dirs, strict=True):
        frame = daily_frames[key]
        periods = {
            "full_validation": _period_metrics(frame, VALIDATION_START, VALIDATION_END),
            "first_half": _period_metrics(frame, VALIDATION_START, FIRST_HALF_END),
            "latest_half": _period_metrics(frame, LATEST_HALF_START, VALIDATION_END),
        }
        _validate_metrics_summary(run_dir, periods["full_validation"])
        for period_name, period in periods.items():
            baseline_period = baseline_periods[period_name]
            period["delta_vs_baseline"] = float(period["primary_ic"]) - float(
                baseline_period["primary_ic"]
            )
            period["horizon_delta_vs_baseline"] = {
                horizon: float(period["horizon_ic"][horizon])
                - float(baseline_period["horizon_ic"][horizon])
                for horizon in ("30m", "60m", "120m")
            }
        bootstrap = paired_moving_block_bootstrap(
            baseline_daily_primary, _daily_primary(frame)
        )
        manifest = manifests[key]
        result = {
            "context_ablation": key,
            "description": get_context_ablation(key).description,
            "run_dir": str(run_dir),
            "run_manifest_path": str((run_dir / "run_manifest.json").resolve()),
            "run_manifest_sha256": _manifest_sha256(run_dir / "run_manifest.json"),
            "context_ablation_specification": manifest["context_ablation"],
            "ablation_specification_sha256": manifest["context_ablation"][
                "specification_sha256"
            ],
            "best_epoch": int(manifest["best_epoch"]),
            "training_duration_seconds": float(manifest["training_duration_seconds"]),
            "periods": periods,
            "first_half_vs_second_half_primary_delta": float(
                periods["latest_half"]["primary_ic"]
            )
            - float(periods["first_half"]["primary_ic"]),
            "paired_moving_block_bootstrap": bootstrap,
        }
        results.append(result)
        full = periods["full_validation"]
        first = periods["first_half"]
        latest = periods["latest_half"]
        csv_rows.append(
            {
                "context_ablation": key,
                "description": result["description"],
                "primary_ic": full["primary_ic"],
                "primary_delta_vs_baseline": full["delta_vs_baseline"],
                "ic_30m": full["horizon_ic"]["30m"],
                "delta_30m_vs_baseline": full["horizon_delta_vs_baseline"]["30m"],
                "ic_60m": full["horizon_ic"]["60m"],
                "delta_60m_vs_baseline": full["horizon_delta_vs_baseline"]["60m"],
                "ic_120m": full["horizon_ic"]["120m"],
                "delta_120m_vs_baseline": full["horizon_delta_vs_baseline"]["120m"],
                "first_half_primary_ic": first["primary_ic"],
                "latest_half_primary_ic": latest["primary_ic"],
                "latest_half_delta_vs_baseline": latest["delta_vs_baseline"],
                "second_minus_first_half_primary_ic": result[
                    "first_half_vs_second_half_primary_delta"
                ],
                "bootstrap_paired_mean_delta": bootstrap["paired_mean_delta"],
                "bootstrap_lower_95": bootstrap["interval_lower_95"],
                "bootstrap_upper_95": bootstrap["interval_upper_95"],
                "best_epoch": result["best_epoch"],
                "training_duration_seconds": result["training_duration_seconds"],
                "run_dir": result["run_dir"],
                "run_manifest_sha256": result["run_manifest_sha256"],
                "ablation_specification_sha256": result[
                    "ablation_specification_sha256"
                ],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / _SUMMARY_JSON
    csv_path = output_dir / _SUMMARY_CSV
    summary = {
        "state_file": str(state_path.resolve()),
        "configuration": configuration,
        "run_count": 25,
        "baseline_primary_validation_ic": baseline_periods["full_validation"][
            "primary_ic"
        ],
        "bootstrap_method": {
            "method": "paired moving-block bootstrap of daily primary IC differences",
            "daily_primary_statistic": "equal mean of 30m, 60m, and 120m daily Spearman IC",
            "block_trading_days": BOOTSTRAP_BLOCK_TRADING_DAYS,
            "replications": BOOTSTRAP_REPLICATIONS,
            "seed": BOOTSTRAP_SEED,
            "interpretation": (
                "Time-series uncertainty conditional on the seed-29 trained models; "
                "this is not across-seed training uncertainty."
            ),
        },
        "results": results,
    }
    _atomic_write_json(json_path, summary)
    _atomic_write_csv(csv_path, pl.DataFrame(csv_rows))
    return json_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state-file", type=Path, default=DEFAULT_STATE_DIR / "state.json"
    )
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
    print(f"Wrote Stage-1 JSON summary: {json_path}")
    print(f"Wrote Stage-1 CSV summary: {csv_path}")


if __name__ == "__main__":
    main()
