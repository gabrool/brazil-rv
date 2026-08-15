from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import polars as pl
import torch
from torch import nn

from .baselines import SharedCausalTCN
from .contract import (
    GH200_RUNTIME,
    HORIZONS,
    TCNArchitecture,
    TCNSettings,
    architecture_for_model,
    tcn_tap_receptive_field_minutes,
)
from .data import create_analysis_loader
from .engine import EvaluationObservations, _autocast, _predict, _to_device
from .evaluate import load_current_neural_run
from .horizon_diagnostics import (
    BOOTSTRAP_SEED,
    RIDGE_PENALTIES,
    RidgeSufficientStatistics,
    assert_analysis_rows,
    atomic_csv,
    atomic_json,
    context_permutation,
    mask_context_family_batch,
    permute_context_family_batch,
)
from .metrics import (
    create_metric_table,
    moving_block_bootstrap,
    sample_level_spearman_ic,
)


class DailyICAccumulator:
    def __init__(self) -> None:
        self.values: dict[tuple[int, int], list[float]] = defaultdict(list)

    def add(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        mask: np.ndarray,
        date_idx: np.ndarray,
    ) -> None:
        values = sample_level_spearman_ic(
            predictions.astype(np.float32, copy=False),
            targets.astype(np.float32, copy=False),
            mask.astype(bool, copy=False),
        )
        for sample, date_value in enumerate(date_idx):
            for horizon, value in enumerate(values[sample]):
                if math.isfinite(float(value)):
                    self.values[int(date_value), horizon].append(float(value))

    def add_single(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        mask: np.ndarray,
        date_idx: np.ndarray,
        horizon: int,
    ) -> None:
        values = sample_level_spearman_ic(
            predictions[..., None].astype(np.float32, copy=False),
            targets[..., None].astype(np.float32, copy=False),
            mask[..., None].astype(bool, copy=False),
        )[:, 0]
        for date_value, value in zip(date_idx, values, strict=True):
            if math.isfinite(float(value)):
                self.values[int(date_value), horizon].append(float(value))

    def horizon_score(self, horizon: int) -> float:
        daily = [
            float(np.mean(values))
            for (_, current), values in self.values.items()
            if current == horizon and values
        ]
        return float(np.mean(daily)) if daily else float("nan")

    def aggregate_score(self) -> float:
        scores = np.asarray(
            [self.horizon_score(index) for index in range(len(HORIZONS))]
        )
        return float(np.nanmean(scores))


def analysis_loader(
    store: Path,
    rows: pl.DataFrame,
    checkpoint: dict[str, object],
    batch_size: int,
):
    settings_value = checkpoint["tcn_settings"]
    settings = TCNSettings(**settings_value) if settings_value is not None else None
    architecture = architecture_for_model(str(checkpoint["model_name"]), settings)
    peer = checkpoint["peer_features"]
    return create_analysis_loader(
        store,
        rows,
        str(checkpoint["model_name"]),
        checkpoint["global_context"],
        GH200_RUNTIME,
        int(checkpoint["seed"]),
        architecture if isinstance(architecture, TCNArchitecture) else None,
        str(peer["mode"]),
        str(checkpoint.get("context_family_ablation", "none")),
        batch_size,
    )


def model_diagnostics(
    model: SharedCausalTCN, batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    transferred = _to_device(batch, device)
    arguments = [
        transferred[name]
        for name in (
            "patches",
            "history_patch_mask",
            "instrument_mask",
            "slow_features",
            "state_position",
        )
    ]
    if "peer_state" in transferred:
        arguments.append(transferred["peer_state"])
    with torch.inference_mode(), _autocast(device):
        return model.extract_diagnostics(*arguments)


def candidate_tensors(
    diagnostics: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    taps = [diagnostics[f"block_{index}"] for index in range(1, 7)]
    return {
        **{f"block_{index}": tap for index, tap in enumerate(taps, start=1)},
        "uniform_mean": torch.stack(taps, dim=0).mean(dim=0),
        "concatenated": torch.cat(taps, dim=-1),
        "final_post_fusion": diagnostics["final_pre_head"],
    }


def update_ridge_statistics(
    statistics: dict[str, list[RidgeSufficientStatistics]],
    candidates: dict[str, np.ndarray],
    targets: np.ndarray,
    label_mask: np.ndarray,
    row_mask: np.ndarray,
) -> None:
    for name, features in candidates.items():
        for horizon in range(len(HORIZONS)):
            statistics[name][horizon].update(
                features,
                targets[..., horizon],
                label_mask[..., horizon] & row_mask[:, None],
            )


def run_frozen_block_probes(
    run_dir: Path,
    store: Path,
    train_rows: pl.DataFrame,
    validation_rows: pl.DataFrame,
    output_dir: Path,
) -> dict[str, object]:
    assert_analysis_rows(train_rows)
    assert_analysis_rows(validation_rows, allow_validation=True)
    model, checkpoint, restored_store = load_current_neural_run(run_dir)
    if restored_store.resolve() != store.resolve():
        raise ValueError("Probe checkpoint feature store differs from the stage store")
    if int(checkpoint["seed"]) != 29:
        raise ValueError("Frozen probes require the seed-29 control")
    if checkpoint["tcn_settings"]["readout"] != "final":
        raise ValueError("Frozen probes require final readout")
    if not isinstance(model, SharedCausalTCN):
        raise TypeError("Frozen block probes require the TCN")
    model = model.cuda().eval()
    unique_dates = train_rows.get_column("trade_date").unique().sort().to_list()
    split = math.floor(0.8 * len(unique_dates))
    boundary = unique_dates[split]
    early_dates = set(unique_dates[:split])
    dimensions = {
        **{f"block_{index}": model.architecture.width for index in range(1, 7)},
        "uniform_mean": model.architecture.width,
        "concatenated": 6 * model.architecture.width,
        "final_post_fusion": model.architecture.width,
    }
    early_stats = {
        name: [RidgeSufficientStatistics(width) for _ in HORIZONS]
        for name, width in dimensions.items()
    }
    all_stats = {
        name: [RidgeSufficientStatistics(width) for _ in HORIZONS]
        for name, width in dimensions.items()
    }
    trade_by_date = dict(
        train_rows.select("date_idx", "trade_date").unique().iter_rows()
    )
    loader = analysis_loader(
        store, train_rows, checkpoint, GH200_RUNTIME.evaluation_batch_size
    )
    for batch in loader:
        valid_count = int(batch["sample_valid_mask"].sum())
        candidates = {
            name: tensor[:valid_count].float().cpu().numpy()
            for name, tensor in candidate_tensors(
                model_diagnostics(model, batch)
            ).items()
        }
        targets = batch["targets"][:valid_count].numpy()
        masks = batch["label_mask"][:valid_count].numpy()
        dates = batch["date_idx"][:valid_count].numpy()
        early = np.asarray(
            [trade_by_date[int(value)] in early_dates for value in dates],
            dtype=bool,
        )
        update_ridge_statistics(early_stats, candidates, targets, masks, early)
        update_ridge_statistics(
            all_stats,
            candidates,
            targets,
            masks,
            np.ones(valid_count, dtype=bool),
        )
    candidate_models = {
        name: [
            [stats[horizon].solve(penalty) for penalty in RIDGE_PENALTIES]
            for horizon in range(len(HORIZONS))
        ]
        for name, stats in early_stats.items()
    }
    calibration = {
        (name, horizon, penalty): DailyICAccumulator()
        for name in dimensions
        for horizon in range(len(HORIZONS))
        for penalty in range(len(RIDGE_PENALTIES))
    }
    calibration_rows = train_rows.filter(pl.col("trade_date") >= boundary)
    loader = analysis_loader(
        store, calibration_rows, checkpoint, GH200_RUNTIME.evaluation_batch_size
    )
    for batch in loader:
        valid_count = int(batch["sample_valid_mask"].sum())
        candidates = {
            name: tensor[:valid_count].float().cpu().numpy()
            for name, tensor in candidate_tensors(
                model_diagnostics(model, batch)
            ).items()
        }
        targets = batch["targets"][:valid_count].numpy()
        masks = batch["label_mask"][:valid_count].numpy()
        dates = batch["date_idx"][:valid_count].numpy()
        for name, features in candidates.items():
            for horizon in range(len(HORIZONS)):
                for penalty_index, (coefficient, intercept) in enumerate(
                    candidate_models[name][horizon]
                ):
                    calibration[name, horizon, penalty_index].add_single(
                        features @ coefficient + intercept,
                        targets[..., horizon],
                        masks[..., horizon],
                        dates,
                        horizon,
                    )
    selected = {
        (name, horizon): max(
            range(len(RIDGE_PENALTIES)),
            key=lambda penalty: calibration[name, horizon, penalty].horizon_score(
                horizon
            ),
        )
        for name in dimensions
        for horizon in range(len(HORIZONS))
    }
    fitted = {
        (name, horizon): all_stats[name][horizon].solve(
            RIDGE_PENALTIES[selected[name, horizon]]
        )
        for name in dimensions
        for horizon in range(len(HORIZONS))
    }
    scores = {
        name: DailyICAccumulator() for name in (*dimensions, "incumbent_predictions")
    }
    loader = analysis_loader(
        store, validation_rows, checkpoint, GH200_RUNTIME.evaluation_batch_size
    )
    for batch in loader:
        valid_count = int(batch["sample_valid_mask"].sum())
        diagnostics = model_diagnostics(model, batch)
        candidates = {
            name: tensor[:valid_count].float().cpu().numpy()
            for name, tensor in candidate_tensors(diagnostics).items()
        }
        targets = batch["targets"][:valid_count].numpy()
        masks = batch["label_mask"][:valid_count].numpy()
        dates = batch["date_idx"][:valid_count].numpy()
        for name, features in candidates.items():
            predictions = np.zeros_like(targets)
            for horizon in range(len(HORIZONS)):
                coefficient, intercept = fitted[name, horizon]
                predictions[..., horizon] = features @ coefficient + intercept
            scores[name].add(predictions, targets, masks, dates)
        scores["incumbent_predictions"].add(
            diagnostics["predictions"][:valid_count].float().cpu().numpy(),
            targets,
            masks,
            dates,
        )
    rows: list[dict[str, object]] = []
    for name, accumulator in scores.items():
        for horizon, minutes in enumerate(HORIZONS):
            coefficient_norm = None
            penalty = None
            feature_dim = 3 if name == "incumbent_predictions" else dimensions[name]
            valid_count = None
            if name != "incumbent_predictions":
                coefficient, _ = fitted[name, horizon]
                coefficient_norm = float(np.linalg.norm(coefficient))
                penalty = RIDGE_PENALTIES[selected[name, horizon]]
                valid_count = all_stats[name][horizon].count
            rows.append(
                {
                    "candidate": name,
                    "horizon_minutes": minutes,
                    "validation_ic": accumulator.horizon_score(horizon),
                    "selected_penalty": penalty,
                    "feature_dim": feature_dim,
                    "valid_count": valid_count,
                    "coefficient_norm": coefficient_norm,
                }
            )
        rows.append(
            {
                "candidate": name,
                "horizon_minutes": 0,
                "validation_ic": accumulator.aggregate_score(),
                "selected_penalty": None,
                "feature_dim": (
                    3 if name == "incumbent_predictions" else dimensions[name]
                ),
                "valid_count": None,
                "coefficient_norm": None,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(pl.DataFrame(rows), output_dir / "frozen_block_probes.csv")
    best_taps = {
        str(minutes): max(
            (f"block_{index}" for index in range(1, 7)),
            key=lambda name: scores[name].horizon_score(horizon),
        )
        for horizon, minutes in enumerate(HORIZONS)
    }
    summary = {
        "probe_calibration_start": str(boundary),
        "ridge_penalties": RIDGE_PENALTIES,
        "receptive_field_minutes": tcn_tap_receptive_field_minutes(model.architecture),
        "best_tap_by_horizon": best_taps,
        "shallower_30_deeper_120": (
            int(best_taps["30"].split("_")[1]) < int(best_taps["120"].split("_")[1])
        ),
        "concatenation_beats_every_individual_tap_all_horizons": all(
            scores["concatenated"].horizon_score(horizon)
            > max(
                scores[f"block_{index}"].horizon_score(horizon) for index in range(1, 7)
            )
            for horizon in range(len(HORIZONS))
        ),
        "concatenated_beats_final_post_fusion_by_horizon": {
            str(minutes): scores["concatenated"].horizon_score(horizon)
            > scores["final_post_fusion"].horizon_score(horizon)
            for horizon, minutes in enumerate(HORIZONS)
        },
        "earlier_tap_beats_final_post_fusion_by_horizon": {
            str(minutes): max(
                scores[f"block_{index}"].horizon_score(horizon) for index in range(1, 6)
            )
            > scores["final_post_fusion"].horizon_score(horizon)
            for horizon, minutes in enumerate(HORIZONS)
        },
    }
    atomic_json(output_dir / "frozen_block_probe_summary.json", summary)
    return summary


def collect_observations(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    transform: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]
    | None = None,
) -> EvaluationObservations:
    device = next(model.parameters()).device
    collected: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "sample_id",
            "targets",
            "raw_returns",
            "label_mask",
            "date_idx",
            "decision_idx",
        )
    }
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for original in loader:
            batch = original if transform is None else transform(original)
            valid_count = int(batch["sample_valid_mask"].sum())
            transferred = _to_device(batch, device)
            with _autocast(device):
                prediction = _predict(model, transferred)
            predictions.append(prediction[:valid_count].float().cpu().numpy())
            for name in collected:
                collected[name].append(batch[name][:valid_count].numpy())
    arrays = {
        "predictions": np.concatenate(predictions),
        **{name: np.concatenate(parts) for name, parts in collected.items()},
    }
    order = np.argsort(arrays["sample_id"], kind="stable")
    return EvaluationObservations(
        **{
            name: arrays[name][order]
            for name in EvaluationObservations.__dataclass_fields__
        }
    )


def paired_daily_rows(
    baseline: list[dict[str, object]],
    candidate: list[dict[str, object]],
    mode: str,
) -> list[dict[str, object]]:
    base = {
        (int(row["date_idx"]), int(row["horizon_minutes"])): float(row["spearman_ic"])
        for row in baseline
    }
    compared = {
        (int(row["date_idx"]), int(row["horizon_minutes"])): float(row["spearman_ic"])
        for row in candidate
    }
    rows: list[dict[str, object]] = []
    for minutes in (*HORIZONS, 0):
        if minutes:
            keys = sorted(key for key in base if key[1] == minutes)
            baseline_values = np.asarray([base[key] for key in keys])
            candidate_values = np.asarray([compared[key] for key in keys])
        else:
            dates = sorted({key[0] for key in base})
            baseline_values = np.asarray(
                [np.mean([base[date_idx, h] for h in HORIZONS]) for date_idx in dates]
            )
            candidate_values = np.asarray(
                [
                    np.mean([compared[date_idx, h] for h in HORIZONS])
                    for date_idx in dates
                ]
            )
        interval = moving_block_bootstrap(
            candidate_values - baseline_values, seed=BOOTSTRAP_SEED
        )
        rows.append(
            {
                "mode": mode,
                "horizon_minutes": minutes,
                "baseline_ic": float(np.mean(baseline_values)),
                "candidate_ic": float(np.mean(candidate_values)),
                "delta_ic": float(interval["estimate"][0]),
                "delta_lower_95": float(interval["lower_95"][0]),
                "delta_upper_95": float(interval["upper_95"][0]),
            }
        )
    return rows


def run_context_inference_probes(
    run_dir: Path,
    store: Path,
    validation_rows: pl.DataFrame,
    output_dir: Path,
) -> dict[str, object]:
    assert_analysis_rows(validation_rows, allow_validation=True)
    model, checkpoint, _ = load_current_neural_run(run_dir)
    if (
        int(checkpoint["seed"]) != 29
        or checkpoint["tcn_settings"]["readout"] != "final"
    ):
        raise ValueError("Context probes require the seed-29 final control")
    model = model.cuda().eval()
    mapping, permutation_manifest = context_permutation(validation_rows)
    batch_size = validation_rows.get_column("trade_date").n_unique()
    modes: dict[str, Callable | None] = {"baseline": None}
    for family in ("wdo", "br_rates", "us_rates"):
        modes[f"{family}_masked"] = lambda batch, family=family: (
            mask_context_family_batch(batch, family)
        )
        modes[f"{family}_permuted"] = lambda batch, family=family: (
            permute_context_family_batch(batch, family, mapping)
        )
    daily_by_mode: dict[str, list[dict[str, object]]] = {}
    summary_by_mode: dict[str, dict[str, object]] = {}
    for name, transform in modes.items():
        observations = collect_observations(
            model,
            analysis_loader(store, validation_rows, checkpoint, batch_size),
            transform,
        )
        summary, daily = create_metric_table(
            observations.predictions,
            observations.targets,
            observations.raw_returns,
            observations.label_mask,
            observations.date_idx,
            observations.decision_idx,
        )
        summary_by_mode[name] = summary
        daily_by_mode[name] = daily
    baseline_summary = summary_by_mode["baseline"]
    rows = [
        {
            "mode": "baseline",
            "horizon_minutes": row["horizon_minutes"],
            "baseline_ic": row["mean_daily_spearman_ic"],
            "candidate_ic": row["mean_daily_spearman_ic"],
            "delta_ic": 0.0,
            "delta_lower_95": 0.0,
            "delta_upper_95": 0.0,
        }
        for row in baseline_summary["horizons"]
    ]
    rows.append(
        {
            "mode": "baseline",
            "horizon_minutes": 0,
            "baseline_ic": baseline_summary["primary_score"],
            "candidate_ic": baseline_summary["primary_score"],
            "delta_ic": 0.0,
            "delta_lower_95": 0.0,
            "delta_upper_95": 0.0,
        }
    )
    for name in modes:
        if name != "baseline":
            rows.extend(
                paired_daily_rows(daily_by_mode["baseline"], daily_by_mode[name], name)
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(pl.DataFrame(rows), output_dir / "context_inference_probes.csv")
    atomic_json(output_dir / "context_permutation_manifest.json", permutation_manifest)
    summary = {
        "interpretation": (
            "Inference corruption measures checkpoint reliance/alignment; "
            "retraining ablation separately measures replaceable usefulness."
        ),
        "inference": summary_by_mode,
    }
    atomic_json(output_dir / "context_family_summary.json", summary)
    return summary
