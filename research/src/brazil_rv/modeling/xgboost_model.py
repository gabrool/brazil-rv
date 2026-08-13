from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
import xgboost as xgb

from .contract import (
    EQUITY_COUNT,
    HORIZONS,
    TABULAR_FEATURE_COUNT,
    XGBOOST_CANDIDATES,
    XGBOOST_DEVICE,
    XGBOOST_EARLY_STOPPING_ROUNDS,
    XGBOOST_FIXED_PARAMETERS,
    XGBOOST_INNER_EMBARGO_DATES,
    XGBOOST_INNER_VALIDATION_FRACTION,
    XGBOOST_MAX_BOOSTING_ROUNDS,
    XGBoostCandidate,
)
from .data import TabularRowBatch, TabularRowIterator
from .metrics import create_metric_table


@dataclass(frozen=True)
class InnerDateSplit:
    training_rows: pl.DataFrame
    validation_rows: pl.DataFrame


@dataclass(frozen=True)
class MatrixBundle:
    training: xgb.DMatrix
    validation: xgb.DMatrix
    validation_source: TabularRowIterator


def inner_date_split(rows: pl.DataFrame) -> InnerDateSplit:
    dates = tuple(rows.get_column("trade_date").unique().sort())
    validation_count = max(
        1, int(np.ceil(len(dates) * XGBOOST_INNER_VALIDATION_FRACTION))
    )
    validation_start = len(dates) - validation_count
    training_stop = validation_start - XGBOOST_INNER_EMBARGO_DATES
    if training_stop <= 0:
        raise ValueError("Training interval is too short for the inner embargo")
    training_dates = dates[:training_stop]
    validation_dates = dates[validation_start:]
    return InnerDateSplit(
        rows.filter(pl.col("trade_date").is_in(training_dates)),
        rows.filter(pl.col("trade_date").is_in(validation_dates)),
    )


def candidate_parameters(candidate: XGBoostCandidate, seed: int) -> dict[str, object]:
    return {
        **dict(XGBOOST_FIXED_PARAMETERS),
        **asdict(candidate),
        "seed": seed,
        "verbosity": 0,
        "validate_parameters": True,
    }


class QuantileBatchDataIter(xgb.DataIter):
    def __init__(self, source: Iterable[TabularRowBatch], cache_prefix: Path) -> None:
        super().__init__(
            cache_prefix=str(cache_prefix), release_data=True, on_host=True
        )
        self.source = source
        self._iterator = iter(source)

    def reset(self) -> None:
        self._iterator = iter(self.source)

    def next(self, input_data: Callable[..., None]) -> bool:
        try:
            batch = next(self._iterator)
        except StopIteration:
            return False
        input_data(data=batch.features, label=batch.labels, weight=batch.weights)
        return True


def _matrix(
    source: TabularRowIterator, cache_prefix: Path, reference: xgb.DMatrix | None = None
) -> xgb.DMatrix:
    matrix = xgb.ExtMemQuantileDMatrix(
        QuantileBatchDataIter(source, cache_prefix),
        max_bin=int(XGBOOST_FIXED_PARAMETERS["max_bin"]),
        ref=reference,
    )
    if matrix.num_col() != TABULAR_FEATURE_COUNT:
        raise ValueError(
            f"Expected {TABULAR_FEATURE_COUNT} XGBoost features, found {matrix.num_col()}"
        )
    return matrix


def _matrix_bundles(
    store: Path,
    training_rows: pl.DataFrame,
    validation_rows: pl.DataFrame,
    global_context: str,
    cache_dir: Path,
) -> dict[int, MatrixBundle]:
    bundles: dict[int, MatrixBundle] = {}
    for horizon_index, horizon in enumerate(HORIZONS):
        training_source = TabularRowIterator(
            store,
            training_rows,
            horizon_index,
            device=XGBOOST_DEVICE,
            global_context=global_context,
        )
        validation_source = TabularRowIterator(
            store,
            validation_rows,
            horizon_index,
            device=XGBOOST_DEVICE,
            global_context=global_context,
        )
        training = _matrix(training_source, cache_dir / f"train_{horizon}m")
        validation = _matrix(
            validation_source, cache_dir / f"validation_{horizon}m", training
        )
        bundles[horizon_index] = MatrixBundle(training, validation, validation_source)
    return bundles


def _prediction_arrays(
    store: Path, rows: pl.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dates = rows.get_column("date_idx").to_numpy().astype(np.int64)
    decisions = rows.get_column("decision_idx").to_numpy().astype(np.int64)
    count = rows.height
    targets = np.zeros((count, EQUITY_COUNT, len(HORIZONS)), dtype=np.float32)
    raw_returns = np.zeros_like(targets)
    label_mask = np.zeros_like(targets, dtype=bool)
    target_store = np.load(store / "targets.npy", mmap_mode="r", allow_pickle=False)
    return_store = np.load(store / "raw_returns.npy", mmap_mode="r", allow_pickle=False)
    mask_store = np.load(store / "label_mask.npy", mmap_mode="r", allow_pickle=False)
    for decision in np.unique(decisions):
        group = np.flatnonzero(decisions == decision)
        targets[group] = target_store[dates[group], :, int(decision), :]
        raw_returns[group] = return_store[dates[group], :, int(decision), :]
        label_mask[group] = mask_store[dates[group], :, int(decision), :]
    return targets, raw_returns, label_mask


def _predict(
    boosters: dict[int, xgb.Booster],
    bundles: dict[int, MatrixBundle],
    rows: pl.DataFrame,
) -> np.ndarray:
    predictions = np.zeros((rows.height, EQUITY_COUNT, len(HORIZONS)), dtype=np.float32)
    sample_positions = {
        int(sample_id): position
        for position, sample_id in enumerate(rows.get_column("sample_id"))
    }
    for horizon_index in range(len(HORIZONS)):
        bundle = bundles[horizon_index]
        flat = (
            boosters[horizon_index]
            .predict(bundle.validation, strict_shape=True)
            .reshape(-1)
        )
        cursor = 0
        for batch in bundle.validation_source:
            count = len(batch.sample_id)
            positions = np.fromiter(
                (sample_positions[int(value)] for value in batch.sample_id),
                dtype=np.int64,
                count=count,
            )
            predictions[positions, batch.equity_slot, horizon_index] = flat[
                cursor : cursor + count
            ]
            cursor += count
        if cursor != flat.size:
            raise RuntimeError(
                "XGBoost prediction rows do not align with valid equity rows"
            )
    return predictions


def _metrics(
    store: Path, rows: pl.DataFrame, predictions: np.ndarray
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    targets, raw_returns, label_mask = _prediction_arrays(store, rows)
    summary, daily = create_metric_table(
        predictions,
        targets,
        raw_returns,
        label_mask,
        rows.get_column("date_idx").to_numpy(),
        rows.get_column("decision_idx").to_numpy(),
    )
    return summary, daily, (targets, raw_returns, label_mask)


def _best_rounds(booster: xgb.Booster) -> int:
    return int(getattr(booster, "best_iteration", 0)) + 1


def _tune(
    store: Path,
    training_rows: pl.DataFrame,
    global_context: str,
    cache_dir: Path,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    split = inner_date_split(training_rows)
    bundles = _matrix_bundles(
        store,
        split.training_rows,
        split.validation_rows,
        global_context,
        cache_dir / "inner",
    )
    results: list[dict[str, object]] = []
    for candidate_index, candidate in enumerate(XGBOOST_CANDIDATES):
        boosters: dict[int, xgb.Booster] = {}
        rounds: dict[int, int] = {}
        for horizon_index in range(len(HORIZONS)):
            bundle = bundles[horizon_index]
            booster = xgb.train(
                candidate_parameters(candidate, seed),
                bundle.training,
                num_boost_round=XGBOOST_MAX_BOOSTING_ROUNDS,
                evals=[(bundle.validation, "validation")],
                early_stopping_rounds=XGBOOST_EARLY_STOPPING_ROUNDS,
                verbose_eval=False,
            )
            boosters[horizon_index] = booster
            rounds[horizon_index] = _best_rounds(booster)
        predictions = _predict(boosters, bundles, split.validation_rows)
        summary, _, _ = _metrics(store, split.validation_rows, predictions)
        results.append(
            {
                "candidate_index": candidate_index,
                **asdict(candidate),
                "primary_score": float(summary["primary_score"]),
                **{
                    f"horizon_{HORIZONS[index]}m_boosting_rounds": value
                    for index, value in rounds.items()
                },
            }
        )
    finite = [row for row in results if np.isfinite(float(row["primary_score"]))]
    if not finite:
        raise ValueError(
            "Every XGBoost candidate produced a non-finite validation score"
        )
    selected = max(finite, key=lambda row: float(row["primary_score"]))
    return selected, results


def _refit(
    store: Path,
    training_rows: pl.DataFrame,
    validation_rows: pl.DataFrame,
    global_context: str,
    cache_dir: Path,
    selected: dict[str, object],
    seed: int,
) -> tuple[dict[int, xgb.Booster], dict[int, MatrixBundle]]:
    candidate = XGBoostCandidate(
        int(selected["max_depth"]),
        float(selected["learning_rate"]),
        int(selected["min_child_weight"]),
    )
    bundles = _matrix_bundles(
        store, training_rows, validation_rows, global_context, cache_dir / "final"
    )
    boosters = {
        index: xgb.train(
            candidate_parameters(candidate, seed),
            bundles[index].training,
            num_boost_round=int(selected[f"horizon_{horizon}m_boosting_rounds"]),
            verbose_eval=False,
        )
        for index, horizon in enumerate(HORIZONS)
    }
    return boosters, bundles


def booster_path(run_dir: Path, horizon: int) -> Path:
    return run_dir / f"booster_{horizon}m.ubj"


def _save_boosters(run_dir: Path, boosters: dict[int, xgb.Booster]) -> None:
    for index, horizon in enumerate(HORIZONS):
        destination = booster_path(run_dir, horizon)
        temporary = destination.with_suffix(".tmp.ubj")
        boosters[index].save_model(temporary)
        os.replace(temporary, destination)


def _load_boosters(run_dir: Path) -> dict[int, xgb.Booster]:
    boosters: dict[int, xgb.Booster] = {}
    for index, horizon in enumerate(HORIZONS):
        booster = xgb.Booster()
        booster.load_model(booster_path(run_dir, horizon))
        boosters[index] = booster
    return boosters


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _prediction_frame(
    rows: pl.DataFrame, predictions: np.ndarray, label_mask: np.ndarray
) -> pl.DataFrame:
    records: list[dict[str, object]] = []
    for sample, sample_id in enumerate(rows.get_column("sample_id")):
        for equity in range(EQUITY_COUNT):
            for horizon_index, horizon in enumerate(HORIZONS):
                if label_mask[sample, equity, horizon_index]:
                    records.append(
                        {
                            "sample_id": int(sample_id),
                            "equity_slot": equity,
                            "horizon_minutes": horizon,
                            "prediction": float(
                                predictions[sample, equity, horizon_index]
                            ),
                        }
                    )
    return pl.DataFrame(records)


def train_xgboost_run(
    store: Path,
    training_rows: pl.DataFrame,
    validation_rows: pl.DataFrame,
    global_context: str,
    run_dir: Path,
    seed: int,
) -> None:
    with xgb.config_context(use_cuda_async_pool=True):
        with tempfile.TemporaryDirectory(prefix="xgboost_", dir=run_dir) as temporary:
            cache_dir = Path(temporary)
            selected, tuning = _tune(
                store, training_rows, global_context, cache_dir, seed
            )
            boosters, bundles = _refit(
                store,
                training_rows,
                validation_rows,
                global_context,
                cache_dir,
                selected,
                seed,
            )
            predictions = _predict(boosters, bundles, validation_rows)
            summary, daily, arrays = _metrics(store, validation_rows, predictions)
            _save_boosters(run_dir, boosters)
    pl.DataFrame(tuning).write_parquet(run_dir / "tuning_results.parquet")
    pl.DataFrame(daily).write_parquet(run_dir / "validation_daily_metrics.parquet")
    _prediction_frame(validation_rows, predictions, arrays[2]).write_parquet(
        run_dir / "validation_predictions.parquet"
    )
    _atomic_json(run_dir / "selected_xgboost_settings.json", selected)
    _atomic_json(run_dir / "validation_metrics.json", summary)
    _atomic_json(
        run_dir / "run_manifest.json",
        {
            "status": "completed",
            "feature_store": str(store),
            "split": {
                "training": "train",
                "selection": "validation",
                "test_accessed": False,
            },
            "seed": seed,
            "global_context": global_context,
            "model": {"model_name": "xgboost", "device": XGBOOST_DEVICE},
            "selected_settings": selected,
        },
    )


def evaluate_saved_xgboost(
    store: Path,
    training_rows: pl.DataFrame,
    rows: pl.DataFrame,
    global_context: str,
    run_dir: Path,
    work_dir: Path,
) -> tuple[np.ndarray, dict[str, object], list[dict[str, object]], pl.DataFrame]:
    with xgb.config_context(use_cuda_async_pool=True):
        with tempfile.TemporaryDirectory(
            prefix="xgboost_evaluate_", dir=work_dir
        ) as temporary:
            bundles = _matrix_bundles(
                store, training_rows, rows, global_context, Path(temporary)
            )
            predictions = _predict(_load_boosters(run_dir), bundles, rows)
    summary, daily, arrays = _metrics(store, rows, predictions)
    return predictions, summary, daily, _prediction_frame(rows, predictions, arrays[2])
