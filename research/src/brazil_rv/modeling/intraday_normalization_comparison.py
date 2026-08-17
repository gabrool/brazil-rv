from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

from brazil_rv.preprocessing.intraday_normalization import ARMS, write_canonical_json

from .analyze_stock_time_attribution import (
    load_attribution_inputs,
    primary_time_bins,
    rank_decomposition,
)
from .contract import HORIZONS
from .metrics import moving_block_bootstrap

SEEDS = (11, 29, 47)
BOOTSTRAP_REPLICATIONS = 10_000
BOOTSTRAP_BLOCK_DAYS = 5
BOOTSTRAP_SEED = 20260815
COMPARISON_SCHEMA = "EQUITY_INTRADAY_NORMALIZATION_COMPARISON_V1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric_record(arm: str, seed: int, run_dir: Path) -> dict[str, object]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads(
        (run_dir / "validation_metrics.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("status") != "completed"
        or manifest["split"].get("test_accessed") is not False
    ):
        raise ValueError(f"Run is not a completed validation-only run: {run_dir}")
    horizon = {int(row["horizon_minutes"]): row for row in metrics["horizons"]}
    daily = pl.read_parquet(run_dir / "validation_daily_metrics.parquet")
    daily_aggregate = (
        daily.group_by("date_idx")
        .agg(pl.col("spearman_ic").mean().alias("aggregate_ic"))
        .sort("date_idx")
    )
    row: dict[str, object] = {
        "arm": arm,
        "gamma": ARMS[arm],
        "seed": seed,
        "selected_epoch": int(manifest["best_epoch"]),
        "selection_metric": "validation_primary_ic",
        "selection_value": float(manifest["best_validation_score"]),
        "aggregate_validation_ic": float(metrics["primary_score"]),
        "worst_horizon_ic": float(
            min(float(horizon[value]["mean_daily_spearman_ic"]) for value in HORIZONS)
        ),
        "daily_ic_mean": float(daily_aggregate["aggregate_ic"].mean()),
        "daily_ic_standard_deviation": float(daily_aggregate["aggregate_ic"].std()),
        "parameter_count": int(manifest["parameter_count"]),
        "duration_seconds": float(manifest["total_run_seconds"]),
        "run_path": str(run_dir.resolve()),
        "feature_store_metadata_sha256": manifest["feature_store_identity"][
            "metadata_sha256"
        ],
    }
    for minutes in HORIZONS:
        values = horizon[minutes]
        row[f"ic_{minutes}m"] = float(values["mean_daily_spearman_ic"])
        row[f"gross_spread_{minutes}m"] = float(values["mean_top_minus_bottom"])
        row[f"turnover_{minutes}m"] = float(values["mean_one_way_turnover"])
    return row


def _daily_matrix(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pl.read_parquet(run_dir / "validation_daily_metrics.parquet").sort(
        "date_idx", "horizon_minutes"
    )
    dates = frame["date_idx"].unique(maintain_order=True).to_numpy()
    matrix = np.empty((dates.size, len(HORIZONS)), dtype=np.float64)
    for horizon_index, horizon in enumerate(HORIZONS):
        selected = frame.filter(pl.col("horizon_minutes") == horizon).sort("date_idx")
        if not np.array_equal(selected["date_idx"].to_numpy(), dates):
            raise ValueError("Daily validation metric date axes differ by horizon")
        matrix[:, horizon_index] = selected["spearman_ic"].to_numpy()
    return dates, matrix


def _paired_worst_bootstrap(
    candidate: np.ndarray, legacy: np.ndarray
) -> tuple[float, float, float]:
    if candidate.shape != legacy.shape or candidate.ndim != 2:
        raise ValueError("Worst-horizon bootstrap inputs are misaligned")
    date_count = candidate.shape[0]
    block_count = math.ceil(date_count / BOOTSTRAP_BLOCK_DAYS)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    starts = generator.integers(
        0,
        date_count - BOOTSTRAP_BLOCK_DAYS + 1,
        size=(BOOTSTRAP_REPLICATIONS, block_count),
    )
    indices = (
        starts[..., None] + np.arange(BOOTSTRAP_BLOCK_DAYS, dtype=np.int64)
    ).reshape(BOOTSTRAP_REPLICATIONS, -1)[:, :date_count]
    candidate_means = np.nanmean(candidate[indices], axis=1)
    legacy_means = np.nanmean(legacy[indices], axis=1)
    replicated = np.min(candidate_means, axis=1) - np.min(legacy_means, axis=1)
    estimate = float(np.min(candidate.mean(axis=0)) - np.min(legacy.mean(axis=0)))
    lower, upper = np.quantile(replicated, (0.025, 0.975))
    return estimate, float(lower), float(upper)


def _bootstrap_row(
    arm: str,
    seed: int,
    endpoint: str,
    estimate: float,
    lower: float,
    upper: float,
    *,
    horizon_minutes: int | None = None,
    time_bin_30m: int | None = None,
) -> dict[str, object]:
    return {
        "arm": arm,
        "seed": seed,
        "endpoint": endpoint,
        "horizon_minutes": horizon_minutes,
        "time_bin_30m": time_bin_30m,
        "estimate": estimate,
        "lower_95": lower,
        "upper_95": upper,
        "block_trading_days": BOOTSTRAP_BLOCK_DAYS,
        "replications": BOOTSTRAP_REPLICATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def _time_bin_daily(run_dir: Path, cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    inputs = load_attribution_inputs(run_dir, cache_dir)
    sample_ic = rank_decomposition(
        inputs.predictions, inputs.targets, inputs.label_mask
    ).sample_ic
    dates = np.unique(inputs.date_idx)
    bins = primary_time_bins()
    output = np.full((dates.size, len(bins)), np.nan, dtype=np.float64)
    for date_position, date_idx in enumerate(dates):
        on_date = inputs.date_idx == date_idx
        for bin_idx, decisions in enumerate(bins):
            selected = on_date & np.isin(inputs.decision_idx, decisions)
            output[date_position, bin_idx] = float(np.nanmean(sample_ic[selected]))
    return dates, output


def _paired_bootstrap_rows(
    arm: str,
    seed: int,
    candidate_run: Path,
    legacy_run: Path,
    cache_dir: Path,
) -> list[dict[str, object]]:
    candidate_dates, candidate = _daily_matrix(candidate_run)
    legacy_dates, legacy = _daily_matrix(legacy_run)
    if not np.array_equal(candidate_dates, legacy_dates):
        raise ValueError("Matched runs have different validation dates")
    rows: list[dict[str, object]] = []
    horizon_result = moving_block_bootstrap(
        candidate - legacy,
        replications=BOOTSTRAP_REPLICATIONS,
        block_length=BOOTSTRAP_BLOCK_DAYS,
        seed=BOOTSTRAP_SEED,
    )
    for horizon_index, minutes in enumerate(HORIZONS):
        rows.append(
            _bootstrap_row(
                arm,
                seed,
                f"ic_{minutes}m_delta",
                float(horizon_result["estimate"][horizon_index]),
                float(horizon_result["lower_95"][horizon_index]),
                float(horizon_result["upper_95"][horizon_index]),
                horizon_minutes=minutes,
            )
        )
    aggregate = moving_block_bootstrap(
        np.mean(candidate - legacy, axis=1),
        replications=BOOTSTRAP_REPLICATIONS,
        block_length=BOOTSTRAP_BLOCK_DAYS,
        seed=BOOTSTRAP_SEED,
    )
    rows.append(
        _bootstrap_row(
            arm,
            seed,
            "aggregate_ic_delta",
            float(aggregate["estimate"][0]),
            float(aggregate["lower_95"][0]),
            float(aggregate["upper_95"][0]),
        )
    )
    estimate, lower, upper = _paired_worst_bootstrap(candidate, legacy)
    rows.append(
        _bootstrap_row(
            arm,
            seed,
            "worst_horizon_ic_delta",
            estimate,
            lower,
            upper,
        )
    )
    candidate_dates, candidate_bins = _time_bin_daily(candidate_run, cache_dir)
    legacy_dates, legacy_bins = _time_bin_daily(legacy_run, cache_dir)
    if not np.array_equal(candidate_dates, legacy_dates):
        raise ValueError("Matched time-bin runs have different validation dates")
    time_result = moving_block_bootstrap(
        candidate_bins - legacy_bins,
        replications=BOOTSTRAP_REPLICATIONS,
        block_length=BOOTSTRAP_BLOCK_DAYS,
        seed=BOOTSTRAP_SEED,
    )
    for bin_idx in range(candidate_bins.shape[1]):
        rows.append(
            _bootstrap_row(
                arm,
                seed,
                f"time_bin_30m_{bin_idx + 1:02d}_delta",
                float(time_result["estimate"][bin_idx]),
                float(time_result["lower_95"][bin_idx]),
                float(time_result["upper_95"][bin_idx]),
                time_bin_30m=bin_idx,
            )
        )
    return rows


def consolidate_intraday_normalization_stage(
    run_dirs: dict[tuple[str, int], Path],
    attribution_dir: Path,
    diagnostics_dir: Path,
    output_dir: Path,
    cache_dir: Path,
) -> Path:
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if set(run_dirs) != expected:
        raise ValueError("Comparison requires the exact nine-run matrix")
    output_dir.mkdir(parents=True, exist_ok=False)
    records = [
        _metric_record(arm, seed, run_dirs[(arm, seed)])
        for arm in ARMS
        for seed in SEEDS
    ]
    run_metrics = pl.DataFrame(records)
    if run_metrics["parameter_count"].n_unique() != 1:
        raise ValueError("Model parameter counts differ across normalization arms")
    deltas: list[dict[str, object]] = []
    metric_names = [
        "aggregate_validation_ic",
        *(f"ic_{minutes}m" for minutes in HORIZONS),
        "worst_horizon_ic",
        "daily_ic_mean",
    ]
    by_key = {(row["arm"], int(row["seed"])): row for row in records}
    for arm in tuple(ARMS)[1:]:
        for seed in SEEDS:
            candidate = by_key[(arm, seed)]
            legacy = by_key[("legacy_daily_vol", seed)]
            row: dict[str, object] = {
                "arm": arm,
                "gamma": ARMS[arm],
                "seed": seed,
                "candidate_win": float(candidate["aggregate_validation_ic"])
                > float(legacy["aggregate_validation_ic"]),
            }
            for name in metric_names:
                row[f"{name}_delta"] = float(candidate[name]) - float(legacy[name])
            deltas.append(row)
    matched = pl.DataFrame(deltas)

    bootstrap_rows: list[dict[str, object]] = []
    for arm in tuple(ARMS)[1:]:
        for seed in SEEDS:
            bootstrap_rows.extend(
                _paired_bootstrap_rows(
                    arm,
                    seed,
                    run_dirs[(arm, seed)],
                    run_dirs[("legacy_daily_vol", seed)],
                    cache_dir,
                )
            )
    bootstrap = pl.DataFrame(bootstrap_rows)

    attribution = pl.read_csv(attribution_dir / "time_of_day_30m.csv")
    run_lookup = {
        path.name: {"arm": arm, "seed": seed} for (arm, seed), path in run_dirs.items()
    }
    time_rows = [
        {**run_lookup[row["run"]], **row} for row in attribution.iter_rows(named=True)
    ]
    time_of_day = pl.DataFrame(time_rows)
    if time_of_day["arm"].null_count() or time_of_day["seed"].null_count():
        raise ValueError("Time-of-day attribution contains an unknown run")

    arm_summary: list[dict[str, object]] = []
    for arm in ARMS:
        selected = run_metrics.filter(pl.col("arm") == arm)
        candidate_delta = matched.filter(pl.col("arm") == arm)
        row = {
            "arm": arm,
            "gamma": ARMS[arm],
            "aggregate_ic_mean": float(selected["aggregate_validation_ic"].mean()),
            "aggregate_ic_standard_deviation": float(
                selected["aggregate_validation_ic"].std()
            ),
            "worst_horizon_ic_mean": float(selected["worst_horizon_ic"].mean()),
            "candidate_win_count": (
                0
                if arm == "legacy_daily_vol"
                else int(candidate_delta["candidate_win"].sum())
            ),
            "mean_aggregate_ic_delta_vs_legacy": (
                0.0
                if arm == "legacy_daily_vol"
                else float(candidate_delta["aggregate_validation_ic_delta"].mean())
            ),
            "mean_worst_horizon_delta_vs_legacy": (
                0.0
                if arm == "legacy_daily_vol"
                else float(candidate_delta["worst_horizon_ic_delta"].mean())
            ),
        }
        for minutes in HORIZONS:
            row[f"ic_{minutes}m_mean"] = float(selected[f"ic_{minutes}m"].mean())
            row[f"ic_{minutes}m_mean_delta_vs_legacy"] = (
                0.0
                if arm == "legacy_daily_vol"
                else float(candidate_delta[f"ic_{minutes}m_delta"].mean())
            )
        arm_summary.append(row)

    diagnostics = json.loads(
        (diagnostics_dir / "heteroskedasticity_summary.json").read_text(
            encoding="utf-8"
        )
    )
    hetero = {row["arm"]: row for row in diagnostics["primary_results"]}
    interpretation: list[dict[str, object]] = []
    legacy_rmse = float(hetero["legacy_daily_vol"]["weighted_log_std_rmse"])
    for row in arm_summary[1:]:
        arm = str(row["arm"])
        rmse = float(hetero[arm]["weighted_log_std_rmse"])
        ic_delta = float(row["mean_aggregate_ic_delta_vs_legacy"])
        time_deltas = (
            bootstrap.filter(
                (pl.col("arm") == arm) & pl.col("time_bin_30m").is_not_null()
            )
            .group_by("time_bin_30m")
            .agg(pl.col("estimate").mean())
            .sort("time_bin_30m")
        )
        bin_deltas = time_deltas["estimate"].to_numpy()
        largest_bin = int(time_deltas["time_bin_30m"][int(np.argmax(bin_deltas))])
        positive_horizons = sum(
            float(row[f"ic_{minutes}m_mean_delta_vs_legacy"]) > 0.0
            for minutes in HORIZONS
        )
        interpretation.append(
            {
                "arm": arm,
                "heteroskedasticity_reduced": rmse < legacy_rmse,
                "ic_improved": ic_delta > 0.0,
                "heteroskedasticity_reduced_and_ic_improved": (
                    rmse < legacy_rmse and ic_delta > 0.0
                ),
                "heteroskedasticity_reduced_but_ic_not_improved": (
                    rmse < legacy_rmse and ic_delta <= 0.0
                ),
                "validation_weighted_log_std_rmse": rmse,
                "mean_aggregate_ic_delta": ic_delta,
                "positive_horizon_count": positive_horizons,
                "improvement_is_broad_across_horizons": positive_horizons
                == len(HORIZONS),
                "mean_time_bin_30m_deltas": bin_deltas.tolist(),
                "positive_time_bin_count": int(np.count_nonzero(bin_deltas > 0.0)),
                "largest_time_bin_gain": largest_bin,
                "largest_gain_is_opening_bin": largest_bin == 0,
                "gains_concentrated_in_opening_bins": (
                    largest_bin == 0 and int(np.count_nonzero(bin_deltas > 0.0)) <= 2
                ),
            }
        )
    full_opening_midday = float(
        hetero["equity_tod_full"]["opening_to_midday_std_ratio"]
    )
    summary = {
        "schema": COMPARISON_SCHEMA,
        "test_accessed": False,
        "automatic_promotion": False,
        "arms": arm_summary,
        "interpretation": interpretation,
        "half_outperformed_full": (
            arm_summary[1]["aggregate_ic_mean"] > arm_summary[2]["aggregate_ic_mean"]
        ),
        "full_strength_remained_under_corrected": full_opening_midday > 1.0,
        "full_strength_overcorrected_or_inverted": full_opening_midday < 1.0,
        "bootstrap": {
            "block_trading_days": BOOTSTRAP_BLOCK_DAYS,
            "replications": BOOTSTRAP_REPLICATIONS,
            "seed": BOOTSTRAP_SEED,
        },
        "parameter_count": int(run_metrics["parameter_count"][0]),
        "run_count": run_metrics.height,
    }
    run_metrics.write_csv(output_dir / "run_metrics.csv")
    matched.write_csv(output_dir / "matched_seed_deltas.csv")
    bootstrap.write_csv(output_dir / "bootstrap_intervals.csv")
    time_of_day.write_csv(output_dir / "time_of_day_ic.csv")
    write_canonical_json(output_dir / "stage_summary.json", summary)
    markdown = [
        "# Intraday normalization stage",
        "",
        "No arm is promoted automatically. Results use validation only.",
        "",
        "| Arm | Aggregate IC mean | SD | Mean delta | Wins |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in arm_summary:
        markdown.append(
            f"| {row['arm']} | {row['aggregate_ic_mean']:.6f} | "
            f"{row['aggregate_ic_standard_deviation']:.6f} | "
            f"{row['mean_aggregate_ic_delta_vs_legacy']:.6f} | "
            f"{row['candidate_win_count']} |"
        )
    markdown.extend(("", "## Interpretation", ""))
    for row in interpretation:
        markdown.append(
            f"- `{row['arm']}`: heteroskedasticity_reduced="
            f"{row['heteroskedasticity_reduced']}, ic_improved={row['ic_improved']}, "
            f"positive_horizons={row['positive_horizon_count']}/{len(HORIZONS)}, "
            f"positive_time_bins={row['positive_time_bin_count']}."
        )
    (output_dir / "stage_summary.md").write_bytes(
        ("\n".join(markdown) + "\n").encode("utf-8")
    )
    source_hashes = {
        f"{arm}_seed{seed}": {
            filename: _sha256_file(run_dirs[(arm, seed)] / filename)
            for filename in (
                "run_manifest.json",
                "validation_metrics.json",
                "validation_daily_metrics.parquet",
                "history.csv",
                "best_checkpoint.pt",
            )
        }
        for arm in ARMS
        for seed in SEEDS
    }
    source_hashes["validation_attribution"] = {
        filename: _sha256_file(attribution_dir / filename)
        for filename in ("summary.json", "time_of_day_5m.csv", "time_of_day_30m.csv")
    }
    source_hashes["heteroskedasticity_diagnostics"] = {
        "heteroskedasticity_summary.json": _sha256_file(
            diagnostics_dir / "heteroskedasticity_summary.json"
        )
    }
    source_hashes["validation_prediction_cache"] = {
        "cache_manifest.json": _sha256_file(cache_dir / "cache_manifest.json")
    }
    output_names = (
        "run_metrics.csv",
        "matched_seed_deltas.csv",
        "bootstrap_intervals.csv",
        "time_of_day_ic.csv",
        "stage_summary.json",
        "stage_summary.md",
    )
    write_canonical_json(
        output_dir / "comparison_manifest.json",
        {
            "schema": COMPARISON_SCHEMA,
            "test_accessed": False,
            "source_sha256": source_hashes,
            "source_paths": {
                "validation_attribution": str(attribution_dir.resolve()),
                "heteroskedasticity_diagnostics": str(diagnostics_dir.resolve()),
                "validation_prediction_cache": str(cache_dir.resolve()),
                "runs": {
                    f"{arm}_seed{seed}": str(run_dirs[(arm, seed)].resolve())
                    for arm in ARMS
                    for seed in SEEDS
                },
            },
            "output_sha256": {
                name: _sha256_file(output_dir / name) for name in output_names
            },
        },
    )
    validate_intraday_normalization_comparison(output_dir)
    return output_dir


def _assert_comparison_frame(
    actual: pl.DataFrame,
    expected: pl.DataFrame,
    sort_by: tuple[str, ...],
    label: str,
) -> None:
    if actual.columns != expected.columns or actual.height != expected.height:
        raise ValueError(f"{label} schema or row count does not reconstruct")
    actual = actual.sort(list(sort_by))
    expected = expected.sort(list(sort_by))
    for name in actual.columns:
        actual_null = actual[name].is_null().to_numpy()
        expected_null = expected[name].is_null().to_numpy()
        if not np.array_equal(actual_null, expected_null):
            raise ValueError(f"{label} null policy differs: {name}")
        left = actual[name].drop_nulls()
        right = expected[name].drop_nulls()
        if actual.schema[name].is_numeric() and expected.schema[name].is_numeric():
            equal = np.allclose(
                left.to_numpy(),
                right.to_numpy(),
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
            )
        else:
            equal = left.to_list() == right.to_list()
        if not equal:
            raise ValueError(f"{label} does not reconstruct: {name}")


def _require_comparison_value(actual: object, expected: object, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{label} keys do not reconstruct")
        for key, value in expected.items():
            _require_comparison_value(actual[key], value, f"{label}/{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{label} list does not reconstruct")
        for index, value in enumerate(expected):
            _require_comparison_value(actual[index], value, f"{label}/{index}")
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or not np.isclose(
            actual, expected, rtol=1e-10, atol=1e-12
        ):
            raise ValueError(f"{label} does not reconstruct")
        return
    if actual != expected:
        raise ValueError(f"{label} does not reconstruct")


def _reconstruct_stage_summary(
    run_metrics: pl.DataFrame,
    matched: pl.DataFrame,
    bootstrap: pl.DataFrame,
    diagnostics: dict[str, object],
) -> dict[str, object]:
    arm_summary: list[dict[str, object]] = []
    for arm in ARMS:
        selected = run_metrics.filter(pl.col("arm") == arm)
        candidate_delta = matched.filter(pl.col("arm") == arm)
        row = {
            "arm": arm,
            "gamma": ARMS[arm],
            "aggregate_ic_mean": float(selected["aggregate_validation_ic"].mean()),
            "aggregate_ic_standard_deviation": float(
                selected["aggregate_validation_ic"].std()
            ),
            "worst_horizon_ic_mean": float(selected["worst_horizon_ic"].mean()),
            "candidate_win_count": (
                0
                if arm == "legacy_daily_vol"
                else int(candidate_delta["candidate_win"].sum())
            ),
            "mean_aggregate_ic_delta_vs_legacy": (
                0.0
                if arm == "legacy_daily_vol"
                else float(candidate_delta["aggregate_validation_ic_delta"].mean())
            ),
            "mean_worst_horizon_delta_vs_legacy": (
                0.0
                if arm == "legacy_daily_vol"
                else float(candidate_delta["worst_horizon_ic_delta"].mean())
            ),
        }
        for minutes in HORIZONS:
            row[f"ic_{minutes}m_mean"] = float(selected[f"ic_{minutes}m"].mean())
            row[f"ic_{minutes}m_mean_delta_vs_legacy"] = (
                0.0
                if arm == "legacy_daily_vol"
                else float(candidate_delta[f"ic_{minutes}m_delta"].mean())
            )
        arm_summary.append(row)

    hetero = {row["arm"]: row for row in diagnostics["primary_results"]}
    if set(hetero) != set(ARMS):
        raise ValueError("Comparison diagnostics omit a normalization arm")
    interpretation: list[dict[str, object]] = []
    legacy_rmse = float(hetero["legacy_daily_vol"]["weighted_log_std_rmse"])
    for row in arm_summary[1:]:
        arm = str(row["arm"])
        rmse = float(hetero[arm]["weighted_log_std_rmse"])
        ic_delta = float(row["mean_aggregate_ic_delta_vs_legacy"])
        time_deltas = (
            bootstrap.filter(
                (pl.col("arm") == arm) & pl.col("time_bin_30m").is_not_null()
            )
            .group_by("time_bin_30m")
            .agg(pl.col("estimate").mean())
            .sort("time_bin_30m")
        )
        if time_deltas.height != len(primary_time_bins()):
            raise ValueError("Comparison time-bin endpoint lattice is incomplete")
        bin_deltas = time_deltas["estimate"].to_numpy()
        largest_bin = int(time_deltas["time_bin_30m"][int(np.argmax(bin_deltas))])
        positive_horizons = sum(
            float(row[f"ic_{minutes}m_mean_delta_vs_legacy"]) > 0.0
            for minutes in HORIZONS
        )
        interpretation.append(
            {
                "arm": arm,
                "heteroskedasticity_reduced": rmse < legacy_rmse,
                "ic_improved": ic_delta > 0.0,
                "heteroskedasticity_reduced_and_ic_improved": (
                    rmse < legacy_rmse and ic_delta > 0.0
                ),
                "heteroskedasticity_reduced_but_ic_not_improved": (
                    rmse < legacy_rmse and ic_delta <= 0.0
                ),
                "validation_weighted_log_std_rmse": rmse,
                "mean_aggregate_ic_delta": ic_delta,
                "positive_horizon_count": positive_horizons,
                "improvement_is_broad_across_horizons": positive_horizons
                == len(HORIZONS),
                "mean_time_bin_30m_deltas": bin_deltas.tolist(),
                "positive_time_bin_count": int(np.count_nonzero(bin_deltas > 0.0)),
                "largest_time_bin_gain": largest_bin,
                "largest_gain_is_opening_bin": largest_bin == 0,
                "gains_concentrated_in_opening_bins": (
                    largest_bin == 0 and int(np.count_nonzero(bin_deltas > 0.0)) <= 2
                ),
            }
        )
    full_opening_midday = float(
        hetero["equity_tod_full"]["opening_to_midday_std_ratio"]
    )
    return {
        "schema": COMPARISON_SCHEMA,
        "test_accessed": False,
        "automatic_promotion": False,
        "arms": arm_summary,
        "interpretation": interpretation,
        "half_outperformed_full": (
            arm_summary[1]["aggregate_ic_mean"] > arm_summary[2]["aggregate_ic_mean"]
        ),
        "full_strength_remained_under_corrected": full_opening_midday > 1.0,
        "full_strength_overcorrected_or_inverted": full_opening_midday < 1.0,
        "bootstrap": {
            "block_trading_days": BOOTSTRAP_BLOCK_DAYS,
            "replications": BOOTSTRAP_REPLICATIONS,
            "seed": BOOTSTRAP_SEED,
        },
        "parameter_count": int(run_metrics["parameter_count"][0]),
        "run_count": run_metrics.height,
    }


def _comparison_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Intraday normalization stage",
        "",
        "No arm is promoted automatically. Results use validation only.",
        "",
        "| Arm | Aggregate IC mean | SD | Mean delta | Wins |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["arms"]:
        lines.append(
            f"| {row['arm']} | {row['aggregate_ic_mean']:.6f} | "
            f"{row['aggregate_ic_standard_deviation']:.6f} | "
            f"{row['mean_aggregate_ic_delta_vs_legacy']:.6f} | "
            f"{row['candidate_win_count']} |"
        )
    lines.extend(("", "## Interpretation", ""))
    for row in summary["interpretation"]:
        lines.append(
            f"- `{row['arm']}`: heteroskedasticity_reduced="
            f"{row['heteroskedasticity_reduced']}, ic_improved={row['ic_improved']}, "
            f"positive_horizons={row['positive_horizon_count']}/{len(HORIZONS)}, "
            f"positive_time_bins={row['positive_time_bin_count']}."
        )
    return "\n".join(lines) + "\n"


def _validate_comparison_semantics(
    output_dir: Path, lineage: dict[str, object]
) -> None:
    expected_keys = {f"{arm}_seed{seed}" for arm in ARMS for seed in SEEDS}
    run_paths = lineage["source_paths"].get("runs", {})
    if set(run_paths) != expected_keys:
        raise ValueError("Comparison source run-path matrix is incomplete")
    run_dirs = {
        (arm, seed): Path(run_paths[f"{arm}_seed{seed}"])
        for arm in ARMS
        for seed in SEEDS
    }
    for (arm, seed), path in run_dirs.items():
        if path.name != f"{arm}_seed{seed}":
            raise ValueError("Comparison source run path has the wrong identity")
    expected_run_metrics = pl.DataFrame(
        [
            _metric_record(arm, seed, run_dirs[(arm, seed)])
            for arm in ARMS
            for seed in SEEDS
        ]
    )
    actual_run_metrics = pl.read_csv(output_dir / "run_metrics.csv")
    _assert_comparison_frame(
        actual_run_metrics,
        expected_run_metrics,
        ("arm", "seed"),
        "run metrics",
    )

    by_key = {
        (row["arm"], int(row["seed"])): row
        for row in expected_run_metrics.iter_rows(named=True)
    }
    metric_names = [
        "aggregate_validation_ic",
        *(f"ic_{minutes}m" for minutes in HORIZONS),
        "worst_horizon_ic",
        "daily_ic_mean",
    ]
    delta_rows: list[dict[str, object]] = []
    for arm in tuple(ARMS)[1:]:
        for seed in SEEDS:
            candidate = by_key[(arm, seed)]
            legacy = by_key[("legacy_daily_vol", seed)]
            row: dict[str, object] = {
                "arm": arm,
                "gamma": ARMS[arm],
                "seed": seed,
                "candidate_win": float(candidate["aggregate_validation_ic"])
                > float(legacy["aggregate_validation_ic"]),
            }
            for name in metric_names:
                row[f"{name}_delta"] = float(candidate[name]) - float(legacy[name])
            delta_rows.append(row)
    expected_matched = pl.DataFrame(delta_rows)
    _assert_comparison_frame(
        pl.read_csv(output_dir / "matched_seed_deltas.csv"),
        expected_matched,
        ("arm", "seed"),
        "matched-seed deltas",
    )

    cache_dir = Path(lineage["source_paths"]["validation_prediction_cache"])
    bootstrap_rows: list[dict[str, object]] = []
    for arm in tuple(ARMS)[1:]:
        for seed in SEEDS:
            bootstrap_rows.extend(
                _paired_bootstrap_rows(
                    arm,
                    seed,
                    run_dirs[(arm, seed)],
                    run_dirs[("legacy_daily_vol", seed)],
                    cache_dir,
                )
            )
    expected_bootstrap = pl.DataFrame(bootstrap_rows)
    actual_bootstrap = pl.read_csv(output_dir / "bootstrap_intervals.csv")
    _assert_comparison_frame(
        actual_bootstrap,
        expected_bootstrap,
        ("arm", "seed", "endpoint"),
        "bootstrap intervals",
    )
    if actual_bootstrap.select("arm", "seed", "endpoint").n_unique() != (
        actual_bootstrap.height
    ):
        raise ValueError("Comparison bootstrap endpoint lattice has duplicates")

    attribution_dir = Path(lineage["source_paths"]["validation_attribution"])
    attribution = pl.read_csv(attribution_dir / "time_of_day_30m.csv")
    run_lookup = {
        path.name: {"arm": arm, "seed": seed} for (arm, seed), path in run_dirs.items()
    }
    expected_time = pl.DataFrame(
        [{**run_lookup[row["run"]], **row} for row in attribution.iter_rows(named=True)]
    )
    _assert_comparison_frame(
        pl.read_csv(output_dir / "time_of_day_ic.csv"),
        expected_time,
        ("arm", "seed", "time_bin_30m", "horizon_minutes"),
        "time-of-day IC",
    )
    expected_time_keys = {
        (arm, seed, bin_idx, horizon)
        for arm in ARMS
        for seed in SEEDS
        for bin_idx in range(len(primary_time_bins()))
        for horizon in HORIZONS
    }
    time_key_columns = ("arm", "seed", "time_bin_30m", "horizon_minutes")
    actual_time_keys = set(expected_time.select(time_key_columns).iter_rows())
    if (
        expected_time.height != len(expected_time_keys)
        or expected_time.select(time_key_columns).n_unique() != expected_time.height
        or actual_time_keys != expected_time_keys
    ):
        raise ValueError("Comparison time-of-day endpoint lattice is incomplete")

    diagnostics = json.loads(
        (
            Path(lineage["source_paths"]["heteroskedasticity_diagnostics"])
            / "heteroskedasticity_summary.json"
        ).read_text(encoding="utf-8")
    )
    expected_summary = _reconstruct_stage_summary(
        expected_run_metrics, expected_matched, expected_bootstrap, diagnostics
    )
    actual_summary = json.loads(
        (output_dir / "stage_summary.json").read_text(encoding="utf-8")
    )
    _require_comparison_value(actual_summary, expected_summary, "stage summary")
    if (output_dir / "stage_summary.md").read_bytes() != _comparison_markdown(
        expected_summary
    ).encode("utf-8"):
        raise ValueError("Comparison Markdown does not reconstruct")


def validate_intraday_normalization_comparison(output_dir: Path) -> None:
    summary = json.loads(
        (output_dir / "stage_summary.json").read_text(encoding="utf-8")
    )
    if (
        summary.get("schema") != COMPARISON_SCHEMA
        or summary.get("test_accessed") is not False
    ):
        raise ValueError("Invalid normalization-stage comparison summary")
    run_metrics = pl.read_csv(output_dir / "run_metrics.csv")
    deltas = pl.read_csv(output_dir / "matched_seed_deltas.csv")
    bootstrap = pl.read_csv(output_dir / "bootstrap_intervals.csv")
    time = pl.read_csv(output_dir / "time_of_day_ic.csv")
    lineage = json.loads(
        (output_dir / "comparison_manifest.json").read_text(encoding="utf-8")
    )
    if (
        lineage.get("schema") != COMPARISON_SCHEMA
        or lineage.get("test_accessed") is not False
    ):
        raise ValueError("Invalid comparison lineage manifest")
    for name, expected_hash in lineage["output_sha256"].items():
        if _sha256_file(output_dir / name) != expected_hash:
            raise ValueError(f"Consolidated output hash mismatch: {name}")
    for source_name, artifacts in lineage["source_sha256"].items():
        if source_name in (
            "validation_attribution",
            "heteroskedasticity_diagnostics",
            "validation_prediction_cache",
        ):
            source_path = Path(lineage["source_paths"][source_name])
            for name, expected_hash in artifacts.items():
                if _sha256_file(source_path / name) != expected_hash:
                    raise ValueError(
                        f"Comparison source hash mismatch: {source_name}/{name}"
                    )
            continue
        source_path = Path(
            run_metrics.filter(
                (pl.col("arm") + "_seed" + pl.col("seed").cast(pl.String))
                == source_name
            )["run_path"][0]
        )
        for name, expected_hash in artifacts.items():
            if _sha256_file(source_path / name) != expected_hash:
                raise ValueError(
                    f"Comparison source hash mismatch: {source_name}/{name}"
                )
    if run_metrics.height != 9 or set(run_metrics["arm"]) != set(ARMS):
        raise ValueError("Comparison run matrix is incomplete")
    if deltas.height != 6 or set(deltas["seed"]) != set(SEEDS):
        raise ValueError("Matched-seed comparison matrix is incomplete")
    if bootstrap.is_empty() or time.is_empty():
        raise ValueError("Comparison diagnostics are empty")
    if run_metrics["parameter_count"].n_unique() != 1:
        raise ValueError("Comparison parameter counts differ")
    for row in deltas.iter_rows(named=True):
        candidate = run_metrics.filter(
            (pl.col("arm") == row["arm"]) & (pl.col("seed") == row["seed"])
        ).row(0, named=True)
        legacy = run_metrics.filter(
            (pl.col("arm") == "legacy_daily_vol") & (pl.col("seed") == row["seed"])
        ).row(0, named=True)
        for metric in (
            "aggregate_validation_ic",
            *(f"ic_{minutes}m" for minutes in HORIZONS),
            "worst_horizon_ic",
            "daily_ic_mean",
        ):
            expected = float(candidate[metric]) - float(legacy[metric])
            if not np.isclose(row[f"{metric}_delta"], expected, rtol=1e-10, atol=1e-12):
                raise ValueError(f"Matched delta does not reconstruct: {metric}")
    for arm_row in summary["arms"]:
        selected = run_metrics.filter(pl.col("arm") == arm_row["arm"])
        expected_mean = float(selected["aggregate_validation_ic"].mean())
        expected_std = float(selected["aggregate_validation_ic"].std())
        if not np.isclose(
            arm_row["aggregate_ic_mean"], expected_mean, rtol=1e-10, atol=1e-12
        ) or not np.isclose(
            arm_row["aggregate_ic_standard_deviation"],
            expected_std,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(f"Arm summary does not reconstruct: {arm_row['arm']}")
    for frame in (run_metrics, deltas, bootstrap, time):
        for name, dtype in frame.schema.items():
            if (
                dtype.is_numeric()
                and not np.isfinite(frame[name].drop_nulls().to_numpy()).all()
            ):
                raise ValueError(f"Non-finite comparison column: {name}")
    _validate_comparison_semantics(output_dir, lineage)
