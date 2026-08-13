from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch

from .contract import HORIZONS, MIN_IC_EQUITIES, TRAIN_END, TRAIN_START
from .evaluate import collect_neural_evaluation
from .metrics import average_ranks, ranking_diagnostics, sample_level_ic


@dataclass(frozen=True)
class AttributionInputs:
    run_name: str
    predictions: np.ndarray
    targets: np.ndarray
    raw_returns: np.ndarray
    label_mask: np.ndarray
    date_idx: np.ndarray
    decision_idx: np.ndarray
    security_ids: tuple[str, ...]
    market_overnight_gap: np.ndarray
    opening_thresholds: tuple[float, float]


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation-only stock/time attribution"
    )
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    return parser.parse_args(arguments)


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _cache_path(cache_dir: Path, run_dir: Path) -> Path:
    resolved = run_dir.resolve()
    return cache_dir.joinpath(*resolved.parts[1:], "predictions.npz")


def _load_cached(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as values:
        return {name: values[name] for name in values.files}


def _write_cache(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def _security_ids(store: Path) -> tuple[str, ...]:
    frame = pl.read_parquet(store / "equity_index.parquet").sort("equity_slot")
    return tuple(str(value) for value in frame.get_column("security_id"))


def _opening_context(
    store: Path,
    validation_dates: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float]]:
    slow = np.load(store / "equity_slow.npy", mmap_mode="r", allow_pickle=False)
    membership = np.load(
        store / "equity_membership.npy", mmap_mode="r", allow_pickle=False
    )
    ready = np.load(store / "equity_data_ready.npy", mmap_mode="r", allow_pickle=False)

    def market_gap(date_indices: np.ndarray) -> np.ndarray:
        output = np.zeros(date_indices.size, dtype=np.float64)
        for position, date_index in enumerate(date_indices):
            active = membership[date_index] & ready[date_index]
            output[position] = (
                float(np.median(slow[date_index, active, 1])) if active.any() else 0.0
            )
        return output

    sample_index = pl.read_parquet(store / "sample_index.parquet")
    training_dates = (
        sample_index.filter(pl.col("trade_date").is_between(TRAIN_START, TRAIN_END))
        .get_column("date_idx")
        .unique()
        .sort()
        .to_numpy()
    )
    training_gaps = market_gap(training_dates)
    thresholds = tuple(float(value) for value in np.quantile(training_gaps, (0.2, 0.8)))
    unique_validation, inverse = np.unique(validation_dates, return_inverse=True)
    return market_gap(unique_validation)[inverse], thresholds


def load_attribution_inputs(
    run_dir: Path, cache_dir: Path | None = None
) -> AttributionInputs:
    cached = (
        _load_cached(_cache_path(cache_dir, run_dir)) if cache_dir is not None else None
    )
    if cached is None:
        observations, _, _, _, store = collect_neural_evaluation(run_dir, "validation")
        values = {
            "predictions": observations.predictions,
            "targets": observations.targets,
            "raw_returns": observations.raw_returns,
            "label_mask": observations.label_mask,
            "date_idx": observations.date_idx,
            "decision_idx": observations.decision_idx,
        }
        if cache_dir is not None:
            _write_cache(_cache_path(cache_dir, run_dir), values)
    else:
        values = cached
        checkpoint = torch.load(
            run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False
        )
        store = Path(str(checkpoint["feature_store"]))
    market_gap, thresholds = _opening_context(store, values["date_idx"])
    return AttributionInputs(
        run_dir.name,
        values["predictions"],
        values["targets"],
        values["raw_returns"],
        values["label_mask"].astype(bool),
        values["date_idx"].astype(np.int64),
        values["decision_idx"].astype(np.int64),
        _security_ids(store),
        market_gap,
        thresholds,
    )


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def stock_attribution(inputs: AttributionInputs) -> pl.DataFrame:
    sample_count, equity_count, horizon_count = inputs.predictions.shape
    contributions = np.full_like(inputs.predictions, np.nan, dtype=np.float64)
    for sample in range(sample_count):
        for horizon in range(horizon_count):
            valid = inputs.label_mask[sample, :, horizon]
            count = int(valid.sum())
            if count < MIN_IC_EQUITIES:
                continue
            predicted = average_ranks(inputs.predictions[sample, valid, horizon])
            actual = average_ranks(inputs.targets[sample, valid, horizon])
            predicted = (predicted - predicted.mean()) / predicted.std(ddof=1)
            actual = (actual - actual.mean()) / actual.std(ddof=1)
            contributions[sample, valid, horizon] = predicted * actual / count
    rows: list[dict[str, object]] = []
    for equity in range(equity_count):
        for horizon, minutes in enumerate(HORIZONS):
            valid = inputs.label_mask[:, equity, horizon]
            rows.append(
                {
                    "run": inputs.run_name,
                    "security_id": inputs.security_ids[equity],
                    "equity_slot": equity,
                    "horizon_minutes": minutes,
                    "observations": int(valid.sum()),
                    "coverage": float(valid.mean()),
                    "mean_spearman_contribution": _finite_mean(
                        contributions[:, equity, horizon]
                    ),
                    "time_series_rank_skill": _correlation(
                        inputs.predictions[valid, equity, horizon],
                        inputs.targets[valid, equity, horizon],
                    ),
                    "mean_raw_return": float(
                        np.mean(inputs.raw_returns[valid, equity, horizon])
                    )
                    if valid.any()
                    else float("nan"),
                }
            )
    return pl.DataFrame(rows)


def _group_metrics(
    inputs: AttributionInputs, group_name: str, group_values: np.ndarray
) -> pl.DataFrame:
    spearman, _ = sample_level_ic(inputs.predictions, inputs.targets, inputs.label_mask)
    diagnostics = ranking_diagnostics(
        inputs.predictions,
        inputs.raw_returns,
        inputs.label_mask,
        inputs.date_idx,
        inputs.decision_idx,
    )
    rows: list[dict[str, object]] = []
    for group in np.unique(group_values):
        selected = group_values == group
        for horizon, minutes in enumerate(HORIZONS):
            rows.append(
                {
                    "run": inputs.run_name,
                    group_name: group.item() if hasattr(group, "item") else group,
                    "horizon_minutes": minutes,
                    "samples": int(selected.sum()),
                    "mean_spearman_ic": _finite_mean(spearman[selected, horizon]),
                    "mean_top_minus_bottom": _finite_mean(
                        diagnostics["top_minus_bottom"][selected, horizon]
                    ),
                    "mean_one_way_turnover": _finite_mean(
                        diagnostics["one_way_turnover"][selected, horizon]
                    ),
                    "label_coverage": float(
                        inputs.label_mask[selected, :, horizon].mean()
                    ),
                }
            )
    return pl.DataFrame(rows)


def time_of_day_attribution(inputs: AttributionInputs) -> pl.DataFrame:
    return _group_metrics(inputs, "decision_idx", inputs.decision_idx)


def horizon_attribution(inputs: AttributionInputs) -> pl.DataFrame:
    return _group_metrics(
        inputs, "all_samples", np.zeros(inputs.date_idx.size, dtype=np.int8)
    ).drop("all_samples")


def opening_attribution(inputs: AttributionInputs) -> pl.DataFrame:
    low, high = inputs.opening_thresholds
    regimes = np.where(
        inputs.market_overnight_gap <= low,
        "negative_tail",
        np.where(inputs.market_overnight_gap >= high, "positive_tail", "middle"),
    )
    return _group_metrics(inputs, "opening_regime", regimes)


def analyze_runs(
    run_dirs: list[Path], output_dir: Path, cache_dir: Path | None = None
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    inputs = [
        load_attribution_inputs(run_dir.resolve(), cache_dir) for run_dir in run_dirs
    ]
    outputs = {
        "stock_attribution": pl.concat([stock_attribution(value) for value in inputs]),
        "time_of_day_5m": pl.concat(
            [time_of_day_attribution(value) for value in inputs]
        ),
        "horizon_attribution": pl.concat(
            [horizon_attribution(value) for value in inputs]
        ),
        "opening_regimes": pl.concat([opening_attribution(value) for value in inputs]),
    }
    for name, frame in outputs.items():
        frame.write_parquet(output_dir / f"{name}.parquet")
        frame.write_csv(output_dir / f"{name}.csv")
    _atomic_json(
        output_dir / "summary.json",
        {
            "split": "validation",
            "test_accessed": False,
            "runs": [str(path.resolve()) for path in run_dirs],
            "outputs": {name: frame.height for name, frame in outputs.items()},
        },
    )
    return output_dir


def main() -> None:
    args = parse_args()
    print(analyze_runs(args.run_dir, args.output_dir, args.cache_dir))


if __name__ == "__main__":
    main()
