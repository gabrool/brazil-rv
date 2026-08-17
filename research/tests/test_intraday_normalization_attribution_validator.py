from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

import brazil_rv.modeling.run_intraday_normalization_stage as stage
from brazil_rv.modeling.contract import HORIZONS
from brazil_rv.preprocessing.intraday_normalization import ARMS, sha256_file


def _metric_table(
    predictions: np.ndarray,
    _targets: np.ndarray,
    _raw_returns: np.ndarray,
    _label_mask: np.ndarray,
    _date_idx: np.ndarray,
    _decision_idx: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    value = float(predictions.mean())
    metrics = {
        "primary_score": value,
        "mean_valid_sample_spearman_ic": value,
        "horizons": [
            {
                "horizon_minutes": horizon,
                "mean_sample_spearman_ic": value + horizon / 10000.0,
            }
            for horizon in HORIZONS
        ],
    }
    daily = [
        {"date_idx": 0, "horizon_minutes": horizon, "daily_ic": value}
        for horizon in HORIZONS
    ]
    return metrics, daily


def _time_5m(inputs: SimpleNamespace) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "run": inputs.run,
                "decision_idx": decision,
                "horizon_minutes": horizon,
                "ic": 0.1,
            }
            for decision in range(2)
            for horizon in HORIZONS
        ]
    )


def _time_30m(inputs: SimpleNamespace) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "run": inputs.run,
                "time_bin_30m": "bin-0",
                "horizon_minutes": horizon,
                "ic": 0.1,
            }
            for horizon in HORIZONS
        ]
    )


def _horizon(inputs: SimpleNamespace) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"run": inputs.run, "horizon_minutes": horizon, "ic": 0.1}
            for horizon in HORIZONS
        ]
    )


def test_finite_rehashed_prediction_cache_must_reconstruct_validation_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    attribution_dir = tmp_path / "attribution"
    stores = {arm: tmp_path / "stores" / arm for arm in ARMS}
    cache_dir.mkdir()
    attribution_dir.mkdir()
    for store in stores.values():
        store.mkdir(parents=True)
    sample_index = pl.DataFrame(
        {
            "sample_id": [0, 1],
            "date_idx": [0, 0],
            "decision_idx": [0, 1],
        }
    )
    monkeypatch.setattr(stage, "EQUITY_COUNT", 2)
    monkeypatch.setattr(stage, "EXPECTED_DECISIONS_PER_DATE", 2)
    monkeypatch.setattr(
        stage, "load_sample_index", lambda _store, end_date: sample_index
    )
    monkeypatch.setattr(stage, "select_sample_split", lambda frame, _split: frame)
    monkeypatch.setattr(
        stage,
        "load_variant_manifest",
        lambda store: (
            None
            if Path(store).name == "legacy_daily_vol"
            else {"arm": Path(store).name}
        ),
    )
    monkeypatch.setattr(stage, "create_metric_table", _metric_table)
    monkeypatch.setattr(
        stage,
        "load_attribution_inputs",
        lambda run_dir, _cache: SimpleNamespace(run=Path(run_dir).name),
    )
    monkeypatch.setattr(stage, "time_of_day_attribution", _time_5m)
    monkeypatch.setattr(stage, "time_of_day_30m_attribution", _time_30m)
    monkeypatch.setattr(stage, "horizon_attribution", _horizon)
    monkeypatch.setattr(stage, "primary_time_bins", lambda: ("bin-0",))

    run_dirs: dict[tuple[str, int], Path] = {}
    expected_5m: list[pl.DataFrame] = []
    expected_30m: list[pl.DataFrame] = []
    expected_horizon: list[pl.DataFrame] = []
    for arm in ARMS:
        for seed in stage.SEEDS:
            run_dir = tmp_path / "runs" / f"{arm}_seed{seed}"
            run_dir.mkdir(parents=True)
            run_dirs[(arm, seed)] = run_dir
            (run_dir / "run_manifest.json").write_text(
                json.dumps({"seed": seed, "feature_store": str(stores[arm])}),
                encoding="utf-8",
            )
            predictions = np.full((2, 2, len(HORIZONS)), seed / 100.0, dtype=np.float32)
            targets = np.ones_like(predictions)
            raw_returns = np.ones_like(predictions)
            label_mask = np.ones_like(predictions, dtype=bool)
            cache_path = stage._cache_path(cache_dir, run_dir)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache_path,
                predictions=predictions,
                targets=targets,
                raw_returns=raw_returns,
                label_mask=label_mask,
                date_idx=np.asarray([0, 0], dtype=np.int64),
                decision_idx=np.asarray([0, 1], dtype=np.int64),
            )
            metrics, daily = _metric_table(
                predictions,
                targets,
                raw_returns,
                label_mask,
                np.asarray([0, 0]),
                np.asarray([0, 1]),
            )
            (run_dir / "validation_metrics.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )
            pl.DataFrame(daily).write_parquet(
                run_dir / "validation_daily_metrics.parquet"
            )
            inputs = SimpleNamespace(run=run_dir.name)
            expected_5m.append(_time_5m(inputs))
            expected_30m.append(_time_30m(inputs))
            expected_horizon.append(_horizon(inputs))
    stage._write_cache_manifest(cache_dir, run_dirs)

    outputs = {
        "stock_attribution": pl.DataFrame({"run": ["all"], "value": [1.0]}),
        "time_of_day_5m": pl.concat(expected_5m),
        "time_of_day_30m": pl.concat(expected_30m),
        "horizon_attribution": pl.concat(expected_horizon),
        "opening_regimes": pl.DataFrame({"run": ["all"], "value": [1.0]}),
        "opening_context": pl.DataFrame({"run": ["all"], "value": [1.0]}),
        "bootstrap_summary": pl.DataFrame({"run": ["all"], "value": [1.0]}),
    }
    for stem, frame in outputs.items():
        frame.write_csv(attribution_dir / f"{stem}.csv")
        frame.write_parquet(attribution_dir / f"{stem}.parquet")
    (attribution_dir / "summary.json").write_text(
        json.dumps(
            {
                "split": "validation",
                "test_accessed": False,
                "runs": [str(path.resolve()) for path in run_dirs.values()],
                "outputs": {stem: frame.height for stem, frame in outputs.items()},
            }
        ),
        encoding="utf-8",
    )
    stage._validate_attribution(attribution_dir, cache_dir, run_dirs)

    corrupt_run = run_dirs[("equity_tod_half", stage.SEEDS[0])]
    corrupt_path = stage._cache_path(cache_dir, corrupt_run)
    with np.load(corrupt_path, allow_pickle=False) as archive:
        cached = {name: archive[name] for name in archive.files}
    cached["predictions"] = cached["predictions"] + np.float32(0.25)
    np.savez(corrupt_path, **cached)
    cache_manifest_path = cache_dir / "cache_manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    cache_manifest["entries"][corrupt_run.name]["sha256"] = sha256_file(corrupt_path)
    cache_manifest_path.write_text(json.dumps(cache_manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="metric does not reconstruct"):
        stage._validate_attribution(attribution_dir, cache_dir, run_dirs)
