from __future__ import annotations

import gc
import hashlib
import hmac
import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
import torch
import xgboost as xgb

from .context_ablation import ResolvedContextAblation
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
    XGBOOST_VERSION,
    XGBoostCandidate,
)
from .data import TabularRowBatch, TabularRowIterator
from .metrics import create_metric_table


@dataclass(frozen=True)
class InnerDateSplit:
    training_dates: tuple[object, ...]
    embargo_dates: tuple[object, ...]
    validation_dates: tuple[object, ...]
    training_rows: pl.DataFrame
    validation_rows: pl.DataFrame


@dataclass(frozen=True)
class MatrixBundle:
    training: xgb.DMatrix
    validation: xgb.DMatrix
    validation_source: TabularRowIterator
    device: str


@dataclass(frozen=True)
class XGBoostRunResult:
    selected_settings: dict[str, object]
    tuning_rows: list[dict[str, object]]
    validation_summary: dict[str, object]
    validation_daily_rows: list[dict[str, object]]
    validation_predictions: pl.DataFrame
    matrix_dimensions: dict[str, object]
    boosting_rounds: dict[str, int]
    booster_sha256: dict[str, str]
    native_cuda_qualification: dict[str, object]


def _validate_requested_device(device: str) -> None:
    if device not in ("cpu", "cuda"):
        raise ValueError("XGBoost device must be 'cpu' or 'cuda'")


def _validate_batch_device(batch: TabularRowBatch, device: str) -> None:
    _validate_requested_device(device)
    arrays = (batch.features, batch.labels, batch.weights)
    if device == "cuda":
        if not all(
            isinstance(value, torch.Tensor) and value.device.type == "cuda"
            for value in arrays
        ):
            raise TypeError(
                "CUDA XGBoost batches require CUDA torch.Tensor features, labels, "
                "and weights"
            )
    elif not all(isinstance(value, np.ndarray) for value in arrays):
        raise TypeError(
            "CPU XGBoost test batches require NumPy features, labels, and weights"
        )


class QuantileBatchDataIter(xgb.DataIter):
    """Device-checked XGBoost iterator over compact valid equity rows."""

    def __init__(
        self,
        source: Iterable[TabularRowBatch],
        cache_prefix: Path | None = None,
        *,
        device: str,
    ) -> None:
        _validate_requested_device(device)
        super().__init__(
            cache_prefix=None if cache_prefix is None else str(cache_prefix),
            release_data=True,
            on_host=device == "cuda",
        )
        self.source = source
        self.device = device
        self._iterator = iter(source)

    def reset(self) -> None:
        self._iterator = iter(self.source)

    def next(self, input_data: Callable[..., None]) -> bool:
        try:
            batch = next(self._iterator)
        except StopIteration:
            return False
        _validate_batch_device(batch, self.device)
        input_data(data=batch.features, label=batch.labels, weight=batch.weights)
        return True


def validate_xgboost_runtime() -> dict[str, object]:
    if xgb.__version__ != XGBOOST_VERSION:
        raise RuntimeError(
            f"XGBoost production runs require {XGBOOST_VERSION}, found {xgb.__version__}"
        )
    build_info = xgb.build_info()
    if not bool(build_info.get("USE_CUDA")):
        raise RuntimeError("XGBoost production runs require a CUDA-enabled build")
    return {"version": xgb.__version__, "build_info": build_info}


def inner_date_split(training_rows: pl.DataFrame) -> InnerDateSplit:
    dates = tuple(training_rows.get_column("trade_date").unique().sort().to_list())
    validation_count = max(
        1, int(np.ceil(len(dates) * XGBOOST_INNER_VALIDATION_FRACTION))
    )
    validation_start = len(dates) - validation_count
    embargo_start = validation_start - XGBOOST_INNER_EMBARGO_DATES
    if embargo_start <= 0:
        raise ValueError(
            "Training interval is too short for inner validation and embargo"
        )
    training_dates = dates[:embargo_start]
    embargo_dates = dates[embargo_start:validation_start]
    validation_dates = dates[validation_start:]
    return InnerDateSplit(
        training_dates=training_dates,
        embargo_dates=embargo_dates,
        validation_dates=validation_dates,
        training_rows=training_rows.filter(pl.col("trade_date").is_in(training_dates)),
        validation_rows=training_rows.filter(
            pl.col("trade_date").is_in(validation_dates)
        ),
    )


def candidate_parameters(candidate: XGBoostCandidate, seed: int) -> dict[str, object]:
    return {
        **dict(XGBOOST_FIXED_PARAMETERS),
        **asdict(candidate),
        "seed": seed,
        "validate_parameters": True,
    }


def booster_device(booster: xgb.Booster) -> str:
    config = json.loads(booster.save_config())
    try:
        device = config["learner"]["generic_param"]["device"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "XGBoost booster configuration is missing its device"
        ) from error
    if not isinstance(device, str):
        raise ValueError("XGBoost booster configuration has an invalid device")
    return device


def validate_booster_device(booster: xgb.Booster, requested_device: str) -> None:
    _validate_requested_device(requested_device)
    configured_device = booster_device(booster)
    if requested_device == "cuda":
        matches = configured_device.startswith("cuda")
    else:
        matches = configured_device == "cpu"
    if not matches:
        raise ValueError(
            f"XGBoost booster device {configured_device!r} does not match "
            f"requested device {requested_device!r}"
        )


def _validate_horizon_indices(mapping: dict[int, object], name: str) -> None:
    expected = set(range(len(HORIZONS)))
    if set(mapping) != expected:
        raise ValueError(f"{name} must contain exactly the three horizon indices")


def build_quantile_matrix(
    source: Iterable[TabularRowBatch],
    cache_prefix: Path,
    *,
    device: str,
    reference: xgb.DMatrix | None = None,
) -> xgb.DMatrix:
    _validate_requested_device(device)
    iterator = QuantileBatchDataIter(source, cache_prefix, device=device)
    matrix = xgb.ExtMemQuantileDMatrix(
        iterator,
        max_bin=int(XGBOOST_FIXED_PARAMETERS["max_bin"]),
        ref=reference,
    )
    if matrix.num_col() != TABULAR_FEATURE_COUNT:
        raise ValueError(
            f"XGBoost matrix has {matrix.num_col()} columns, expected {TABULAR_FEATURE_COUNT}"
        )
    return matrix


def _build_matrix_bundles(
    store: Path,
    training_rows: pl.DataFrame,
    validation_rows: pl.DataFrame,
    global_context: str,
    context_ablation: ResolvedContextAblation,
    cache_dir: Path,
    *,
    device: str,
) -> dict[int, MatrixBundle]:
    _validate_requested_device(device)
    cache_dir.mkdir(parents=True, exist_ok=True)
    bundles: dict[int, MatrixBundle] = {}
    for horizon_index, horizon in enumerate(HORIZONS):
        training_source = TabularRowIterator(
            store,
            training_rows,
            horizon_index,
            global_context=global_context,
            device=device,
            context_ablation=context_ablation,
        )
        validation_source = TabularRowIterator(
            store,
            validation_rows,
            horizon_index,
            global_context=global_context,
            device=device,
            context_ablation=context_ablation,
        )
        training = build_quantile_matrix(
            training_source, cache_dir / f"train_{horizon}m", device=device
        )
        validation = build_quantile_matrix(
            validation_source,
            cache_dir / f"validation_{horizon}m",
            device=device,
            reference=training,
        )
        bundles[horizon_index] = MatrixBundle(
            training=training,
            validation=validation,
            validation_source=validation_source,
            device=device,
        )
    return bundles


def _best_rounds(booster: xgb.Booster) -> int:
    try:
        return int(booster.best_iteration) + 1
    except AttributeError:
        return int(booster.num_boosted_rounds())


def _common_evaluation_arrays(store: Path, rows: pl.DataFrame) -> dict[str, np.ndarray]:
    date_idx = rows.get_column("date_idx").to_numpy().astype(np.int64)
    decision_idx = rows.get_column("decision_idx").to_numpy().astype(np.int64)
    sample_id = rows.get_column("sample_id").to_numpy().astype(np.int64)
    targets_source = np.load(store / "targets.npy", mmap_mode="r", allow_pickle=False)
    raw_source = np.load(store / "raw_returns.npy", mmap_mode="r", allow_pickle=False)
    mask_source = np.load(store / "label_mask.npy", mmap_mode="r", allow_pickle=False)
    targets = np.zeros((rows.height, EQUITY_COUNT, len(HORIZONS)), dtype=np.float32)
    raw_returns = np.zeros_like(targets)
    label_mask = np.zeros_like(targets, dtype=bool)
    for decision in np.unique(decision_idx):
        group = np.flatnonzero(decision_idx == decision)
        targets[group] = targets_source[date_idx[group], :, int(decision), :]
        raw_returns[group] = raw_source[date_idx[group], :, int(decision), :]
        label_mask[group] = mask_source[date_idx[group], :, int(decision), :]
    return {
        "sample_id": sample_id,
        "date_idx": date_idx,
        "decision_idx": decision_idx,
        "targets": targets,
        "raw_returns": raw_returns,
        "label_mask": label_mask,
    }


def _fill_horizon_predictions(
    output: np.ndarray,
    horizon_index: int,
    values: np.ndarray,
    source: TabularRowIterator,
    position_by_sample: dict[int, int],
) -> None:
    cursor = 0
    for batch in source:
        count = batch.features.shape[0]
        batch_values = values[cursor : cursor + count]
        for row, prediction in enumerate(batch_values):
            sample_position = position_by_sample[int(batch.sample_id[row])]
            output[sample_position, int(batch.equity_slot[row]), horizon_index] = (
                prediction
            )
        cursor += count
    if cursor != values.size:
        raise ValueError("Prediction row count does not match the tabular iterator")


def predict_from_bundles(
    boosters: dict[int, xgb.Booster],
    bundles: dict[int, MatrixBundle],
    rows: pl.DataFrame,
    boosting_rounds: dict[int, int] | None = None,
    *,
    requested_device: str,
) -> np.ndarray:
    _validate_requested_device(requested_device)
    _validate_horizon_indices(boosters, "boosters")
    _validate_horizon_indices(bundles, "matrix bundles")
    predictions = np.zeros((rows.height, EQUITY_COUNT, len(HORIZONS)), dtype=np.float32)
    position_by_sample = {
        int(sample_id): position
        for position, sample_id in enumerate(rows.get_column("sample_id"))
    }
    for horizon_index in range(len(HORIZONS)):
        bundle = bundles[horizon_index]
        if bundle.device != requested_device:
            raise ValueError(
                f"XGBoost matrix device {bundle.device!r} does not match "
                f"requested device {requested_device!r}"
            )
        booster = boosters[horizon_index]
        validate_booster_device(booster, requested_device)
        prediction_options = (
            {}
            if boosting_rounds is None
            else {"iteration_range": (0, boosting_rounds[horizon_index])}
        )
        values = booster.predict(bundle.validation, **prediction_options)
        _fill_horizon_predictions(
            predictions,
            horizon_index,
            np.asarray(values, dtype=np.float32),
            bundle.validation_source,
            position_by_sample,
        )
    return predictions


def evaluate_predictions(
    store: Path,
    rows: pl.DataFrame,
    predictions: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, np.ndarray]]:
    arrays = _common_evaluation_arrays(store, rows)
    summary, daily_rows = create_metric_table(
        predictions,
        arrays["targets"],
        arrays["raw_returns"],
        arrays["label_mask"],
        arrays["date_idx"],
        arrays["decision_idx"],
    )
    return summary, daily_rows, arrays


def select_candidate(tuning_rows: list[dict[str, object]]) -> dict[str, object]:
    if len(tuning_rows) != len(XGBOOST_CANDIDATES):
        raise ValueError(
            f"Expected {len(XGBOOST_CANDIDATES)} tuning results, found {len(tuning_rows)}"
        )
    best: dict[str, object] | None = None
    best_score = -float("inf")
    for expected_index, row in enumerate(tuning_rows):
        if int(row["candidate_index"]) != expected_index:
            raise ValueError("XGBoost tuning results are not in candidate-grid order")
        score = float(row["primary_score"])
        if np.isfinite(score) and score > best_score:
            best = row
            best_score = score
    if best is None:
        raise ValueError("Every XGBoost tuning candidate produced a nonfinite score")
    return best


def tune_xgboost(
    store: Path,
    training_rows: pl.DataFrame,
    global_context: str,
    context_ablation: ResolvedContextAblation,
    cache_dir: Path,
    seed: int,
    *,
    device: str,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    if device != XGBOOST_DEVICE:
        raise ValueError("Production XGBoost tuning requires the CUDA device")
    split = inner_date_split(training_rows)
    bundles = _build_matrix_bundles(
        store,
        split.training_rows,
        split.validation_rows,
        global_context,
        context_ablation,
        cache_dir / "inner",
        device=device,
    )
    tuning_rows: list[dict[str, object]] = []
    for candidate_index, candidate in enumerate(XGBOOST_CANDIDATES):
        parameters = candidate_parameters(candidate, seed)
        if parameters["device"] != device:
            raise ValueError("XGBoost parameters and matrix device disagree")
        boosters: dict[int, xgb.Booster] = {}
        rounds: dict[int, int] = {}
        for horizon_index, horizon in enumerate(HORIZONS):
            bundle = bundles[horizon_index]
            booster = xgb.train(
                parameters,
                bundle.training,
                num_boost_round=XGBOOST_MAX_BOOSTING_ROUNDS,
                evals=[(bundle.validation, "inner_validation")],
                early_stopping_rounds=XGBOOST_EARLY_STOPPING_ROUNDS,
                verbose_eval=False,
            )
            validate_booster_device(booster, device)
            boosters[horizon_index] = booster
            rounds[horizon_index] = _best_rounds(booster)
        predictions = predict_from_bundles(
            boosters,
            bundles,
            split.validation_rows,
            rounds,
            requested_device=device,
        )
        summary, _, _ = evaluate_predictions(store, split.validation_rows, predictions)
        horizon_scores = {
            int(row["horizon_minutes"]): float(row["mean_daily_spearman_ic"])
            for row in summary["horizons"]
        }
        tuning_rows.append(
            {
                "candidate_index": candidate_index,
                **asdict(candidate),
                "primary_score": float(summary["primary_score"]),
                **{
                    f"horizon_{horizon}m_mean_daily_spearman_ic": horizon_scores[
                        horizon
                    ]
                    for horizon in HORIZONS
                },
                **{
                    f"horizon_{horizon}m_boosting_rounds": rounds[horizon_index]
                    for horizon_index, horizon in enumerate(HORIZONS)
                },
            }
        )
    selected = select_candidate(tuning_rows)
    split_metadata = {
        "training_date_count": len(split.training_dates),
        "embargo_date_count": len(split.embargo_dates),
        "validation_date_count": len(split.validation_dates),
        "training_first_date": str(split.training_dates[0]),
        "training_last_date": str(split.training_dates[-1]),
        "embargo_first_date": str(split.embargo_dates[0]),
        "embargo_last_date": str(split.embargo_dates[-1]),
        "validation_first_date": str(split.validation_dates[0]),
        "validation_last_date": str(split.validation_dates[-1]),
    }
    del bundles, boosters, bundle, booster
    gc.collect()
    return selected, tuning_rows, split_metadata


def _matrix_dimensions(bundles: dict[int, MatrixBundle]) -> dict[str, object]:
    return {
        f"{horizon}m": {
            "training_rows": int(bundles[horizon_index].training.num_row()),
            "validation_rows": int(bundles[horizon_index].validation.num_row()),
            "columns": int(bundles[horizon_index].training.num_col()),
        }
        for horizon_index, horizon in enumerate(HORIZONS)
    }


def refit_xgboost(
    store: Path,
    training_rows: pl.DataFrame,
    validation_rows: pl.DataFrame,
    global_context: str,
    context_ablation: ResolvedContextAblation,
    cache_dir: Path,
    selected: dict[str, object],
    seed: int,
    *,
    device: str,
) -> tuple[
    dict[int, xgb.Booster],
    dict[int, MatrixBundle],
    dict[int, int],
]:
    candidate = XGBoostCandidate(
        max_depth=int(selected["max_depth"]),
        learning_rate=float(selected["learning_rate"]),
        min_child_weight=int(selected["min_child_weight"]),
    )
    if device != XGBOOST_DEVICE:
        raise ValueError("Production XGBoost refit requires the CUDA device")
    parameters = candidate_parameters(candidate, seed)
    if parameters["device"] != device:
        raise ValueError("XGBoost parameters and matrix device disagree")
    bundles = _build_matrix_bundles(
        store,
        training_rows,
        validation_rows,
        global_context,
        context_ablation,
        cache_dir / "final",
        device=device,
    )
    boosters: dict[int, xgb.Booster] = {}
    boosting_rounds: dict[int, int] = {}
    for horizon_index, horizon in enumerate(HORIZONS):
        rounds = int(selected[f"horizon_{horizon}m_boosting_rounds"])
        booster = xgb.train(
            parameters,
            bundles[horizon_index].training,
            num_boost_round=rounds,
            verbose_eval=False,
        )
        validate_booster_device(booster, device)
        boosters[horizon_index] = booster
        boosting_rounds[horizon_index] = rounds
    return boosters, bundles, boosting_rounds


def booster_path(run_dir: Path, horizon: int) -> Path:
    return run_dir / f"booster_{horizon}m.ubj"


def _horizon_hash_key(horizon: int) -> str:
    return f"{horizon}m"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_boosters(run_dir: Path, boosters: dict[int, xgb.Booster]) -> dict[str, str]:
    _validate_horizon_indices(boosters, "boosters")
    hashes: dict[str, str] = {}
    for horizon_index, horizon in enumerate(HORIZONS):
        destination = booster_path(run_dir, horizon)
        temporary = destination.with_name(f"{destination.stem}.tmp.ubj")
        temporary.unlink(missing_ok=True)
        boosters[horizon_index].save_model(temporary)
        os.replace(temporary, destination)
        hashes[_horizon_hash_key(horizon)] = _sha256(destination)
    return hashes


def validate_booster_hashes(run_dir: Path, expected_sha256: object) -> dict[str, str]:
    expected_keys = {_horizon_hash_key(horizon) for horizon in HORIZONS}
    if not isinstance(expected_sha256, dict) or set(expected_sha256) != expected_keys:
        raise ValueError(
            "XGBoost booster SHA256 map must contain exactly 30m, 60m, and 120m"
        )
    expected_files = {booster_path(run_dir, horizon).name for horizon in HORIZONS}
    actual_files = {
        path.name for path in run_dir.glob("booster_*m.ubj") if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("XGBoost run must contain exactly three horizon booster files")

    validated: dict[str, str] = {}
    for horizon in HORIZONS:
        key = _horizon_hash_key(horizon)
        expected = expected_sha256[key]
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ValueError(f"Malformed XGBoost booster SHA256 for horizon {key}")
        actual = _sha256(booster_path(run_dir, horizon))
        if not hmac.compare_digest(actual, expected):
            raise ValueError(f"XGBoost booster SHA256 mismatch for horizon {key}")
        validated[key] = expected
    return validated


def _load_booster(path: Path, requested_device: str) -> xgb.Booster:
    _validate_requested_device(requested_device)
    booster = xgb.Booster()
    booster.load_model(path)
    booster.set_param({"device": requested_device})
    validate_booster_device(booster, requested_device)
    return booster


def load_boosters(
    run_dir: Path,
    *,
    requested_device: str,
    expected_sha256: object,
) -> dict[int, xgb.Booster]:
    validate_booster_hashes(run_dir, expected_sha256)
    return {
        horizon_index: _load_booster(booster_path(run_dir, horizon), requested_device)
        for horizon_index, horizon in enumerate(HORIZONS)
    }


def qualify_native_cuda_xgboost(work_dir: Path) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Native CUDA XGBoost qualification requires an available GPU"
        )

    qualification_dir: Path | None = None
    with xgb.config_context(use_cuda_async_pool=True):
        with tempfile.TemporaryDirectory(
            prefix="xgboost_cuda_qualification_", dir=work_dir
        ) as temporary:
            qualification_dir = Path(temporary)
            row_count = 16
            source = [
                TabularRowBatch(
                    features=(
                        torch.arange(
                            row_count * TABULAR_FEATURE_COUNT,
                            dtype=torch.float32,
                            device="cuda",
                        ).reshape(row_count, TABULAR_FEATURE_COUNT)
                        / TABULAR_FEATURE_COUNT
                    ),
                    labels=torch.linspace(
                        -1.0, 1.0, row_count, dtype=torch.float32, device="cuda"
                    ),
                    weights=torch.full(
                        (row_count,),
                        1.0 / row_count,
                        dtype=torch.float32,
                        device="cuda",
                    ),
                    sample_id=np.arange(row_count, dtype=np.int64),
                    date_idx=np.zeros(row_count, dtype=np.int64),
                    decision_idx=np.zeros(row_count, dtype=np.int64),
                    equity_slot=np.arange(row_count, dtype=np.int16),
                )
            ]
            matrix = build_quantile_matrix(
                source,
                qualification_dir / "matrix",
                device=XGBOOST_DEVICE,
            )
            parameters = {
                **dict(XGBOOST_FIXED_PARAMETERS),
                "max_depth": 2,
                "learning_rate": 0.2,
                "min_child_weight": 1,
                "seed": 0,
                "validate_parameters": True,
            }
            booster = xgb.train(parameters, matrix, num_boost_round=2)
            validate_booster_device(booster, XGBOOST_DEVICE)
            original = np.asarray(booster.predict(matrix), dtype=np.float32)
            if not np.isfinite(original).all():
                raise ValueError(
                    "Native CUDA XGBoost qualification produced nonfinite output"
                )

            destination = qualification_dir / "qualification.ubj"
            temporary_model = qualification_dir / "qualification.tmp.ubj"
            booster.save_model(temporary_model)
            os.replace(temporary_model, destination)
            reloaded = _load_booster(destination, XGBOOST_DEVICE)
            restored = np.asarray(reloaded.predict(matrix), dtype=np.float32)
            if not np.isfinite(restored).all():
                raise ValueError(
                    "Reloaded native CUDA XGBoost qualification output is nonfinite"
                )
            if not np.array_equal(restored, original):
                raise ValueError(
                    "Native CUDA XGBoost qualification save/load changed predictions"
                )
            report = {
                "passed": True,
                "device": booster_device(reloaded),
                "rows": row_count,
                "columns": TABULAR_FEATURE_COUNT,
                "boosting_rounds": 2,
                "exact_reload_prediction_equality": True,
            }
            del reloaded, booster, matrix, source
            gc.collect()
    if qualification_dir is None or qualification_dir.exists():
        raise RuntimeError("Native CUDA XGBoost qualification cache was not removed")
    return report


def prediction_long_frame(
    rows: pl.DataFrame,
    predictions: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> pl.DataFrame:
    sample, equity_slot, horizon_index = np.nonzero(arrays["label_mask"])
    sample_ids = arrays["sample_id"]
    date_idx = arrays["date_idx"]
    decision_idx = arrays["decision_idx"]
    return pl.DataFrame(
        {
            "sample_id": sample_ids[sample],
            "date_idx": date_idx[sample],
            "decision_idx": decision_idx[sample],
            "equity_slot": equity_slot.astype(np.int16),
            "horizon_minutes": np.asarray(HORIZONS, dtype=np.int16)[horizon_index],
            "prediction": predictions[sample, equity_slot, horizon_index],
            "target": arrays["targets"][sample, equity_slot, horizon_index],
            "raw_return": arrays["raw_returns"][sample, equity_slot, horizon_index],
        }
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(f".tmp{path.suffix}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_parquet(path: Path, frame: pl.DataFrame) -> None:
    temporary = path.with_suffix(f".tmp{path.suffix}")
    frame.write_parquet(temporary)
    os.replace(temporary, path)


def _save_and_qualify_boosters(
    run_dir: Path,
    boosters: dict[int, xgb.Booster],
    bundles: dict[int, MatrixBundle],
    validation_rows: pl.DataFrame,
    *,
    device: str,
) -> dict[str, str]:
    original_predictions = predict_from_bundles(
        boosters, bundles, validation_rows, requested_device=device
    )
    booster_sha256 = save_boosters(run_dir, boosters)
    reloaded = load_boosters(
        run_dir,
        requested_device=device,
        expected_sha256=booster_sha256,
    )
    reloaded_predictions = predict_from_bundles(
        reloaded, bundles, validation_rows, requested_device=device
    )
    if reloaded_predictions.shape != (
        validation_rows.height,
        EQUITY_COUNT,
        len(HORIZONS),
    ):
        raise ValueError("Reloaded XGBoost prediction shape is invalid")
    if not np.array_equal(reloaded_predictions, original_predictions):
        raise ValueError("XGBoost save/load changed validation predictions")
    del reloaded
    gc.collect()
    return booster_sha256


def write_xgboost_artifacts(run_dir: Path, result: XGBoostRunResult) -> None:
    _write_parquet(run_dir / "tuning_results.parquet", pl.DataFrame(result.tuning_rows))
    _write_json(run_dir / "selected_xgboost_settings.json", result.selected_settings)
    _write_json(run_dir / "validation_summary.json", result.validation_summary)
    _write_parquet(
        run_dir / "validation_daily_metrics.parquet",
        pl.DataFrame(result.validation_daily_rows),
    )
    _write_parquet(
        run_dir / "validation_predictions.parquet",
        result.validation_predictions,
    )


def train_xgboost_run(
    store: Path,
    training_rows: pl.DataFrame,
    validation_rows: pl.DataFrame,
    global_context: str,
    context_ablation: ResolvedContextAblation,
    run_dir: Path,
    seed: int,
) -> XGBoostRunResult:
    with xgb.config_context(use_cuda_async_pool=True):
        with tempfile.TemporaryDirectory(prefix="xgboost_", dir=run_dir) as temporary:
            cache_dir = Path(temporary)
            native_cuda_qualification = qualify_native_cuda_xgboost(cache_dir)
            selected, tuning_rows, inner_split = tune_xgboost(
                store,
                training_rows,
                global_context,
                context_ablation,
                cache_dir,
                seed,
                device=XGBOOST_DEVICE,
            )
            boosters, bundles, boosting_rounds = refit_xgboost(
                store,
                training_rows,
                validation_rows,
                global_context,
                context_ablation,
                cache_dir,
                selected,
                seed,
                device=XGBOOST_DEVICE,
            )
            predictions = predict_from_bundles(
                boosters,
                bundles,
                validation_rows,
                requested_device=XGBOOST_DEVICE,
            )
            validation_summary, validation_daily_rows, arrays = evaluate_predictions(
                store, validation_rows, predictions
            )
            dates = dict(
                pl.read_parquet(store / "date_index.parquet")
                .select("date_idx", "trade_date")
                .iter_rows()
            )
            validation_daily_rows = [
                {"trade_date": dates[int(row["date_idx"])], **row}
                for row in validation_daily_rows
            ]
            selected_settings = {
                "feature_store": str(store.resolve()),
                "candidate_index": int(selected["candidate_index"]),
                "max_depth": int(selected["max_depth"]),
                "learning_rate": float(selected["learning_rate"]),
                "min_child_weight": int(selected["min_child_weight"]),
                "fixed_parameters": dict(XGBOOST_FIXED_PARAMETERS),
                "boosting_rounds": {
                    f"{HORIZONS[index]}m": rounds
                    for index, rounds in boosting_rounds.items()
                },
                "inner_split": inner_split,
            }
            booster_sha256 = _save_and_qualify_boosters(
                run_dir,
                boosters,
                bundles,
                validation_rows,
                device=XGBOOST_DEVICE,
            )
            result = XGBoostRunResult(
                selected_settings=selected_settings,
                tuning_rows=tuning_rows,
                validation_summary=validation_summary,
                validation_daily_rows=validation_daily_rows,
                validation_predictions=prediction_long_frame(
                    validation_rows, predictions, arrays
                ),
                matrix_dimensions=_matrix_dimensions(bundles),
                boosting_rounds={
                    f"{HORIZONS[index]}m": rounds
                    for index, rounds in boosting_rounds.items()
                },
                booster_sha256=booster_sha256,
                native_cuda_qualification=native_cuda_qualification,
            )
            write_xgboost_artifacts(run_dir, result)
            del bundles, boosters
            gc.collect()
    return result


def evaluate_saved_xgboost(
    store: Path,
    training_rows: pl.DataFrame,
    rows: pl.DataFrame,
    global_context: str,
    context_ablation: ResolvedContextAblation,
    run_dir: Path,
    work_dir: Path,
    expected_booster_sha256: object,
) -> tuple[
    np.ndarray,
    dict[str, object],
    list[dict[str, object]],
    pl.DataFrame,
]:
    with xgb.config_context(use_cuda_async_pool=True):
        validated_sha256 = validate_booster_hashes(run_dir, expected_booster_sha256)
        with tempfile.TemporaryDirectory(
            prefix="xgboost_evaluate_", dir=work_dir
        ) as temporary:
            cache_dir = Path(temporary)
            bundles = _build_matrix_bundles(
                store,
                training_rows,
                rows,
                global_context,
                context_ablation,
                cache_dir,
                device=XGBOOST_DEVICE,
            )
            boosters = load_boosters(
                run_dir,
                requested_device=XGBOOST_DEVICE,
                expected_sha256=validated_sha256,
            )
            predictions = predict_from_bundles(
                boosters,
                bundles,
                rows,
                requested_device=XGBOOST_DEVICE,
            )
            del bundles, boosters
            gc.collect()
    summary, daily_rows, arrays = evaluate_predictions(store, rows, predictions)
    return (
        predictions,
        summary,
        daily_rows,
        prediction_long_frame(rows, predictions, arrays),
    )
