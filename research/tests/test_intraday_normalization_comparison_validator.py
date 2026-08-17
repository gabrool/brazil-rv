from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

import brazil_rv.modeling.intraday_normalization_comparison as comparison
from brazil_rv.modeling.contract import HORIZONS
from brazil_rv.preprocessing.intraday_normalization import ARMS


def _bootstrap_rows(
    arm: str,
    seed: int,
    _candidate: Path,
    _legacy: Path,
    _cache: Path,
) -> list[dict[str, object]]:
    base = 0.001 * (1 + seed) + (0.01 if arm == "equity_tod_full" else 0.005)
    rows = [
        comparison._bootstrap_row(
            arm,
            seed,
            f"ic_{horizon}m_delta",
            base + horizon / 100000.0,
            base - 0.001,
            base + 0.001,
            horizon_minutes=horizon,
        )
        for horizon in HORIZONS
    ]
    rows.extend(
        (
            comparison._bootstrap_row(
                arm,
                seed,
                "aggregate_ic_delta",
                base,
                base - 0.001,
                base + 0.001,
            ),
            comparison._bootstrap_row(
                arm,
                seed,
                "worst_horizon_ic_delta",
                base - 0.002,
                base - 0.003,
                base - 0.001,
            ),
            comparison._bootstrap_row(
                arm,
                seed,
                "time_bin_30m_01_delta",
                base + 0.002,
                base + 0.001,
                base + 0.003,
                time_bin_30m=0,
            ),
        )
    )
    return rows


def _comparison_sources(
    tmp_path: Path,
) -> tuple[dict[tuple[str, int], Path], Path, Path, Path]:
    runs: dict[tuple[str, int], Path] = {}
    attribution = tmp_path / "attribution"
    diagnostics = tmp_path / "diagnostics"
    cache = tmp_path / "cache"
    attribution.mkdir()
    diagnostics.mkdir()
    cache.mkdir()
    time_rows: list[dict[str, object]] = []
    for arm_index, arm in enumerate(ARMS):
        for seed in comparison.SEEDS:
            run = tmp_path / "runs" / f"{arm}_seed{seed}"
            run.mkdir(parents=True)
            runs[(arm, seed)] = run
            score = 0.02 + arm_index * 0.005 + seed / 100000.0
            (run / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "split": {"test_accessed": False},
                        "best_epoch": 3,
                        "best_validation_score": score,
                        "parameter_count": 1234,
                        "total_run_seconds": 10.0 + seed,
                        "feature_store_identity": {"metadata_sha256": f"store-{arm}"},
                    }
                ),
                encoding="utf-8",
            )
            (run / "validation_metrics.json").write_text(
                json.dumps(
                    {
                        "primary_score": score,
                        "horizons": [
                            {
                                "horizon_minutes": horizon,
                                "mean_daily_spearman_ic": score + horizon / 100000.0,
                                "mean_top_minus_bottom": 0.01,
                                "mean_one_way_turnover": 0.2,
                            }
                            for horizon in HORIZONS
                        ],
                    }
                ),
                encoding="utf-8",
            )
            pl.DataFrame(
                [
                    {
                        "date_idx": date_idx,
                        "horizon_minutes": horizon,
                        "spearman_ic": score + date_idx / 10000.0 + horizon / 100000.0,
                    }
                    for date_idx in range(5)
                    for horizon in HORIZONS
                ]
            ).write_parquet(run / "validation_daily_metrics.parquet")
            (run / "history.csv").write_text("epoch,score\n1,0.1\n", encoding="utf-8")
            (run / "best_checkpoint.pt").write_bytes(b"fixture-checkpoint")
            for horizon in HORIZONS:
                time_rows.append(
                    {
                        "run": run.name,
                        "time_bin_30m": 0,
                        "horizon_minutes": horizon,
                        "ic": score,
                    }
                )
    (attribution / "summary.json").write_text("{}", encoding="utf-8")
    pl.DataFrame({"run": ["fixture"], "ic": [0.1]}).write_csv(
        attribution / "time_of_day_5m.csv"
    )
    pl.DataFrame(time_rows).write_csv(attribution / "time_of_day_30m.csv")
    (diagnostics / "heteroskedasticity_summary.json").write_text(
        json.dumps(
            {
                "primary_results": [
                    {
                        "arm": arm,
                        "weighted_log_std_rmse": 1.0 - arm_index * 0.1,
                        "opening_to_midday_std_ratio": 1.0,
                    }
                    for arm_index, arm in enumerate(ARMS)
                ]
            }
        ),
        encoding="utf-8",
    )
    (cache / "cache_manifest.json").write_text(
        json.dumps({"test_accessed": False}), encoding="utf-8"
    )
    return runs, attribution, diagnostics, cache


def test_comparison_validator_reconstructs_every_consolidated_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs, attribution, diagnostics, cache = _comparison_sources(tmp_path)
    output = tmp_path / "comparison"
    monkeypatch.setattr(comparison, "_paired_bootstrap_rows", _bootstrap_rows)
    monkeypatch.setattr(comparison, "primary_time_bins", lambda: ((0, 1),))
    comparison.consolidate_intraday_normalization_stage(
        runs, attribution, diagnostics, output, cache
    )

    manifest_path = output / "comparison_manifest.json"
    baseline_manifest = manifest_path.read_bytes()
    cases = (
        ("bootstrap", "bootstrap_intervals.csv", "bootstrap intervals"),
        ("interpretation", "stage_summary.json", "stage summary"),
        ("horizon_mean", "stage_summary.json", "stage summary"),
        ("time_bin", "time_of_day_ic.csv", "time-of-day IC"),
        ("markdown", "stage_summary.md", "Markdown"),
    )
    for case, filename, expected_error in cases:
        path = output / filename
        baseline = path.read_bytes()
        if filename == "bootstrap_intervals.csv":
            frame = pl.read_csv(path).with_columns(
                pl.when(pl.int_range(pl.len()) == 0)
                .then(pl.col("lower_95") - 0.1)
                .otherwise(pl.col("lower_95"))
                .alias("lower_95")
            )
            frame.write_csv(path)
        elif filename == "time_of_day_ic.csv":
            frame = pl.read_csv(path).with_columns(
                pl.when(pl.int_range(pl.len()) == 0)
                .then(pl.col("ic") + 0.1)
                .otherwise(pl.col("ic"))
                .alias("ic")
            )
            frame.write_csv(path)
        elif filename == "stage_summary.json":
            summary = json.loads(path.read_text(encoding="utf-8"))
            if case == "horizon_mean":
                summary["arms"][1]["ic_30m_mean"] += 0.1
            else:
                summary["interpretation"][0]["ic_improved"] = not summary[
                    "interpretation"
                ][0]["ic_improved"]
            path.write_text(json.dumps(summary), encoding="utf-8")
        else:
            path.write_bytes(b"scientifically incorrect\n")
        manifest = json.loads(baseline_manifest.decode("utf-8"))
        manifest["output_sha256"][filename] = comparison._sha256_file(path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match=expected_error):
            comparison.validate_intraday_normalization_comparison(output)
        path.write_bytes(baseline)
        manifest_path.write_bytes(baseline_manifest)
