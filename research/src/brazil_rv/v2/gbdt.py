from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .artifacts import sha256_file, write_json_atomic
from .baselines import rank_gaussianize
from .contract import GBDT_SEEDS
from .normalization import average_ranks

try:
    import lightgbm as lgb
except ImportError:  # The research extra is deliberately optional at import time.
    lgb = None


class LightGBMUnavailable(RuntimeError):
    pass


def require_lightgbm() -> Any:
    if lgb is None:
        raise LightGBMUnavailable(
            "The v2 GBDT engine requires LightGBM; run `uv add lightgbm` in research"
        )
    return lgb


@dataclass(frozen=True)
class GBDTConfig:
    num_leaves: int = 31
    learning_rate: float = 0.03
    feature_fraction: float = 0.7
    bagging_fraction: float = 0.7
    bagging_freq: int = 1
    min_data_in_leaf: int = 200
    lambda_l2: float = 1.0
    maximum_rounds: int = 3000
    early_stopping_rounds: int = 100
    seeds: tuple[int, ...] = GBDT_SEEDS
    num_threads: int = 0

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("at least one GBDT seed is required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("GBDT seeds must be unique")
        if self.maximum_rounds <= 0 or self.early_stopping_rounds <= 0:
            raise ValueError("boosting and early-stopping rounds must be positive")


def assemble_gbdt_features(
    slow_features: NDArray[np.floating],
    intraday_features: NDArray[np.floating],
    fast_present: NDArray[np.bool_],
    days_since_last_slow_row: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Combine the normalized last slow/sidecar step, intraday fields, and flags."""

    slow = np.asarray(slow_features, dtype=np.float32)
    intraday = np.asarray(intraday_features, dtype=np.float32)
    present = np.asarray(fast_present, dtype=np.float32)
    days = np.asarray(days_since_last_slow_row, dtype=np.float32)
    if slow.ndim != 4 or intraday.ndim != 3:
        raise ValueError("slow and intraday features must be daily panels")
    if slow.shape[:2] != intraday.shape[:2]:
        raise ValueError("slow and intraday panels are misaligned")
    if present.shape != slow.shape[:2] or days.shape != slow.shape[:2]:
        raise ValueError("GBDT flags are misaligned with the feature panels")
    return np.concatenate(
        (slow[:, :, -1], intraday, present[..., None], days[..., None]),
        axis=-1,
        dtype=np.float32,
    )


def _validate_panel(
    features: NDArray[np.floating],
    targets: NDArray[np.floating],
    mask: NDArray[np.bool_],
) -> tuple[int, int, int]:
    if features.ndim != 3:
        raise ValueError("features must have shape [date, name, feature]")
    if targets.shape != mask.shape or targets.ndim != 3 or targets.shape[-1] != 5:
        raise ValueError("targets and mask must have shape [date, name, 5]")
    if features.shape[:2] != targets.shape[:2]:
        raise ValueError("feature and target panels are misaligned")
    return features.shape


def _date_ids(
    date_count: int, name_count: int, dates: Sequence[int] | None
) -> NDArray[np.int64]:
    values = (
        np.arange(date_count, dtype=np.int64)
        if dates is None
        else np.asarray(dates, dtype=np.int64)
    )
    if values.shape != (date_count,):
        raise ValueError("dates must contain one identity per panel date")
    return np.repeat(values, name_count)


def _daily_spearman(
    predictions: NDArray[np.floating],
    targets: NDArray[np.floating],
    dates: NDArray[np.int64],
) -> float:
    values: list[float] = []
    for date_id in np.unique(dates):
        selected = dates == date_id
        if selected.sum() < 2:
            continue
        left = average_ranks(np.asarray(predictions[selected], dtype=np.float64))
        right = average_ranks(np.asarray(targets[selected], dtype=np.float64))
        left -= left.mean()
        right -= right.mean()
        denominator = np.sqrt(np.sum(left**2) * np.sum(right**2))
        if denominator > 0:
            values.append(float(np.sum(left * right) / denominator))
    return float(np.mean(values)) if values else 0.0


def _metric_for_dates(dates: NDArray[np.int64]):
    def metric(
        predictions: NDArray[np.floating], dataset: Any
    ) -> tuple[str, float, bool]:
        return (
            "mean_daily_spearman",
            _daily_spearman(predictions, dataset.get_label(), dates),
            True,
        )

    return metric


def _sample_weights(
    weights: NDArray[np.floating] | None,
    date_count: int,
    name_count: int,
) -> NDArray[np.float64] | None:
    if weights is None:
        return None
    values = np.asarray(weights, dtype=np.float64)
    if values.shape == (date_count,):
        values = np.repeat(values, name_count)
    elif values.shape == (date_count, name_count):
        values = values.reshape(-1)
    else:
        raise ValueError("sample weights must be per-date or per-date/name")
    return values


class MultiHorizonGBDT:
    """Five independent LightGBM regressors per seed, ensembled by mean."""

    def __init__(
        self,
        config: GBDTConfig = GBDTConfig(),
        *,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        self.config = config
        self.feature_names = None if feature_names is None else tuple(feature_names)
        self.models: dict[int, list[Any]] = {}

    def fit(
        self,
        train_features: NDArray[np.floating],
        train_targets: NDArray[np.floating],
        train_mask: NDArray[np.bool_],
        validation_features: NDArray[np.floating],
        validation_targets: NDArray[np.floating],
        validation_mask: NDArray[np.bool_],
        *,
        train_dates: Sequence[int] | None = None,
        validation_dates: Sequence[int] | None = None,
        sample_weights: NDArray[np.floating] | None = None,
    ) -> MultiHorizonGBDT:
        library = require_lightgbm()
        train_date_count, train_name_count, feature_count = _validate_panel(
            train_features, train_targets, train_mask
        )
        valid_date_count, valid_name_count, valid_feature_count = _validate_panel(
            validation_features, validation_targets, validation_mask
        )
        if feature_count != valid_feature_count:
            raise ValueError("training and validation feature widths differ")
        if self.feature_names is not None and len(self.feature_names) != feature_count:
            raise ValueError("feature_names differs from the feature width")
        train_x = np.asarray(train_features, dtype=np.float32).reshape(
            -1, feature_count
        )
        valid_x = np.asarray(validation_features, dtype=np.float32).reshape(
            -1, feature_count
        )
        _date_ids(train_date_count, train_name_count, train_dates)
        valid_date_rows = _date_ids(
            valid_date_count, valid_name_count, validation_dates
        )
        weights = _sample_weights(sample_weights, train_date_count, train_name_count)
        transformed_train = np.stack(
            [
                rank_gaussianize(train_targets[..., head], train_mask[..., head])
                for head in range(5)
            ],
            axis=-1,
        )
        transformed_validation = np.stack(
            [
                rank_gaussianize(
                    validation_targets[..., head], validation_mask[..., head]
                )
                for head in range(5)
            ],
            axis=-1,
        )
        self.models = {head: [] for head in range(5)}
        for head in range(5):
            train_valid = np.asarray(train_mask[..., head], dtype=bool).reshape(-1)
            validation_valid = np.asarray(
                validation_mask[..., head], dtype=bool
            ).reshape(-1)
            if train_valid.sum() < 2 or validation_valid.sum() < 2:
                raise ValueError(f"horizon {head} has too few valid rows")
            train_y = transformed_train[..., head].reshape(-1)
            valid_y = transformed_validation[..., head].reshape(-1)
            for seed in self.config.seeds:
                params = {
                    "objective": "regression",
                    "metric": "None",
                    "num_leaves": self.config.num_leaves,
                    "learning_rate": self.config.learning_rate,
                    "feature_fraction": self.config.feature_fraction,
                    "bagging_fraction": self.config.bagging_fraction,
                    "bagging_freq": self.config.bagging_freq,
                    "min_data_in_leaf": self.config.min_data_in_leaf,
                    "lambda_l2": self.config.lambda_l2,
                    "seed": seed,
                    "feature_fraction_seed": seed,
                    "bagging_seed": seed,
                    "data_random_seed": seed,
                    "deterministic": True,
                    "force_col_wise": True,
                    "verbosity": -1,
                    "num_threads": self.config.num_threads,
                }
                training = library.Dataset(
                    train_x[train_valid],
                    label=train_y[train_valid],
                    weight=None if weights is None else weights[train_valid],
                    feature_name="auto"
                    if self.feature_names is None
                    else list(self.feature_names),
                    free_raw_data=False,
                )
                validation = library.Dataset(
                    valid_x[validation_valid],
                    label=valid_y[validation_valid],
                    feature_name="auto"
                    if self.feature_names is None
                    else list(self.feature_names),
                    reference=training,
                    free_raw_data=False,
                )
                booster = library.train(
                    params,
                    training,
                    num_boost_round=self.config.maximum_rounds,
                    valid_sets=[validation],
                    valid_names=["selection"],
                    feval=_metric_for_dates(valid_date_rows[validation_valid]),
                    callbacks=[
                        library.early_stopping(
                            self.config.early_stopping_rounds,
                            first_metric_only=True,
                            verbose=False,
                        )
                    ],
                )
                self.models[head].append(booster)
        return self

    def _require_fit(self) -> None:
        if set(self.models) != set(range(5)) or any(
            len(models) != len(self.config.seeds) for models in self.models.values()
        ):
            raise RuntimeError("the five-horizon GBDT ensemble has not been fit")

    def predict_raw(self, features: NDArray[np.floating]) -> NDArray[np.float32]:
        self._require_fit()
        if features.ndim != 3:
            raise ValueError("features must have shape [date, name, feature]")
        date_count, name_count, feature_count = features.shape
        flat = np.asarray(features, dtype=np.float32).reshape(-1, feature_count)
        result = np.empty((date_count, name_count, 5), dtype=np.float32)
        for head, models in self.models.items():
            members = [
                model.predict(flat, num_iteration=model.best_iteration or None)
                for model in models
            ]
            result[..., head] = np.mean(np.stack(members), axis=0).reshape(
                date_count, name_count
            )
        return result

    def predict_ranks(
        self,
        features: NDArray[np.floating],
        score_mask: NDArray[np.bool_],
    ) -> NDArray[np.float32]:
        self._require_fit()
        if features.ndim != 3:
            raise ValueError("features must have shape [date, name, feature]")
        date_count, name_count, _ = features.shape
        mask = np.asarray(score_mask, dtype=bool)
        if mask.ndim == 2:
            mask = np.repeat(mask[..., None], 5, axis=-1)
        if mask.shape != (date_count, name_count, 5):
            raise ValueError("score_mask is misaligned with GBDT predictions")
        flat = np.asarray(features, dtype=np.float32).reshape(-1, features.shape[-1])
        ranked = np.zeros((date_count, name_count, 5), dtype=np.float32)
        for head, models in self.models.items():
            member_ranks = []
            for model in models:
                member = model.predict(
                    flat, num_iteration=model.best_iteration or None
                ).reshape(date_count, name_count)
                ranks = np.zeros((date_count, name_count), dtype=np.float32)
                for date in range(date_count):
                    valid = mask[date, :, head]
                    ranks[date, valid] = average_ranks(member[date, valid])
                member_ranks.append(ranks)
            ranked[..., head] = np.mean(np.stack(member_ranks), axis=0)
        return ranked

    def feature_importance(
        self,
        features: NDArray[np.floating] | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        self._require_fit()
        gain = np.mean(
            np.stack(
                [
                    model.feature_importance(importance_type="gain")
                    for models in self.models.values()
                    for model in models
                ]
            ),
            axis=0,
        )
        result = {"gain": np.asarray(gain, dtype=np.float64)}
        if features is not None:
            flat = np.asarray(features, dtype=np.float32).reshape(
                -1, features.shape[-1]
            )
            contributions = []
            for models in self.models.values():
                for model in models:
                    values = model.predict(
                        flat,
                        num_iteration=model.best_iteration or None,
                        pred_contrib=True,
                    )
                    contributions.append(np.mean(np.abs(values[:, :-1]), axis=0))
            result["mean_abs_tree_shap"] = np.mean(np.stack(contributions), axis=0)
        return result

    def save(
        self,
        root: Path,
        *,
        metadata: dict[str, object] | None = None,
    ) -> tuple[Path, str]:
        """Atomically retain every member and return its hash-bound manifest."""

        self._require_fit()
        output = Path(root).resolve()
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.saving-", dir=output.parent)
        )
        try:
            records: list[dict[str, object]] = []
            for head in range(5):
                members = self.models[head]
                for seed, booster in zip(self.config.seeds, members, strict=True):
                    filename = f"head_{head}_seed_{seed}.txt"
                    path = staging / filename
                    booster.save_model(str(path))
                    records.append(
                        {
                            "head": head,
                            "seed": seed,
                            "path": filename,
                            "bytes": path.stat().st_size,
                            "sha256": sha256_file(path),
                        }
                    )
            extra = dict(metadata or {})
            reserved = {"schema", "config", "feature_names", "models"}
            if reserved & extra.keys():
                raise ValueError("GBDT metadata collides with manifest identity")
            manifest_path = staging / "model_manifest.json"
            manifest_sha256 = write_json_atomic(
                manifest_path,
                {
                    "schema": "BRAZIL_RV_V2_GBDT_MODELS_V1",
                    "config": asdict(self.config),
                    "feature_names": (
                        None
                        if self.feature_names is None
                        else list(self.feature_names)
                    ),
                    "models": records,
                    **extra,
                },
            )
            os.replace(staging, output)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return output / "model_manifest.json", manifest_sha256

    @classmethod
    def load(
        cls,
        root: Path,
        *,
        expected_manifest_sha256: str,
    ) -> MultiHorizonGBDT:
        """Load only an externally hash-identified complete five-head ensemble."""

        path = Path(root).resolve()
        manifest_path = path / "model_manifest.json"
        actual_manifest_sha256 = sha256_file(manifest_path)
        if actual_manifest_sha256 != expected_manifest_sha256:
            raise ValueError("GBDT model manifest SHA-256 mismatch")
        sha_record = (path / "model_manifest.json.sha256").read_text(
            encoding="ascii"
        ).split()[0]
        if sha_record != expected_manifest_sha256:
            raise ValueError("GBDT manifest sidecar SHA-256 mismatch")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema") != "BRAZIL_RV_V2_GBDT_MODELS_V1":
            raise ValueError("GBDT model manifest schema is not recognized")
        config_payload = payload.get("config")
        if not isinstance(config_payload, dict):
            raise ValueError("GBDT model manifest lacks its configuration")
        seeds = config_payload.get("seeds")
        if not isinstance(seeds, list):
            raise ValueError("GBDT model seed roster is malformed")
        config_payload["seeds"] = tuple(int(value) for value in seeds)
        config = GBDTConfig(**config_payload)
        names = payload.get("feature_names")
        if names is not None and (
            not isinstance(names, list)
            or not all(isinstance(value, str) and value for value in names)
        ):
            raise ValueError("GBDT ordered feature names are malformed")
        model = cls(config, feature_names=names)
        records = payload.get("models")
        if not isinstance(records, list) or len(records) != 5 * len(config.seeds):
            raise ValueError("GBDT model manifest has an incomplete member roster")
        by_key: dict[tuple[int, int], Any] = {}
        library = require_lightgbm()
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("GBDT model record is malformed")
            head = record.get("head")
            seed = record.get("seed")
            filename = record.get("path")
            if (
                type(head) is not int
                or head not in range(5)
                or type(seed) is not int
                or seed not in config.seeds
                or not isinstance(filename, str)
                or Path(filename).name != filename
            ):
                raise ValueError("GBDT model identity is malformed")
            key = (head, seed)
            if key in by_key:
                raise ValueError("GBDT model manifest repeats a member")
            artifact = path / filename
            if (
                artifact.stat().st_size != int(record.get("bytes", -1))
                or sha256_file(artifact) != record.get("sha256")
            ):
                raise ValueError("GBDT member hash or size mismatch")
            by_key[key] = library.Booster(model_file=str(artifact))
        expected = {
            (head, seed) for head in range(5) for seed in config.seeds
        }
        if set(by_key) != expected:
            raise ValueError("GBDT model roster differs from its configuration")
        model.models = {
            head: [by_key[(head, seed)] for seed in config.seeds]
            for head in range(5)
        }
        return model
