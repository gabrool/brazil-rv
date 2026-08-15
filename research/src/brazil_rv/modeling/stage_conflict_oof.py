from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn

from .baselines import SharedCausalTCN
from .contract import GH200_RUNTIME, HORIZONS, TRAIN_END
from .engine import (
    _autocast,
    _predict,
    _to_device,
    objective_loss,
    select_training_label_mask,
)
from .evaluate import load_current_neural_run
from .horizon_diagnostics import (
    RIDGE_PENALTIES,
    RidgeSufficientStatistics,
    assert_analysis_rows,
    atomic_csv,
    atomic_json,
    atomic_parquet,
    gradient_cosine,
    rank_standardize_predictions,
    residual_source_designs,
)
from .stage_representation_probes import (
    DailyICAccumulator,
    analysis_loader,
    update_ridge_statistics,
)


def gradient_parameter_groups(
    model: SharedCausalTCN,
) -> dict[str, tuple[nn.Parameter, ...]]:
    return {
        "input_projection": tuple(model.input_projection.parameters()),
        **{
            f"block_{index}": tuple(block.parameters())
            for index, block in enumerate(model.blocks, start=1)
        },
        "slow_projection": tuple(model.slow_projection.parameters()),
        "peer_adapter": (
            () if model.peer_adapter is None else tuple(model.peer_adapter.parameters())
        ),
        "shared_fusion": tuple(
            itertools.chain.from_iterable(
                module.parameters()
                for module in (
                    getattr(model, "fusion_input", nn.Identity()),
                    getattr(model, "fusion_output", nn.Identity()),
                    getattr(model, "fusion_gate", nn.Identity()),
                    getattr(model, "fusion_norm", nn.Identity()),
                )
            )
        ),
        "prediction_head": tuple(model.prediction_head.parameters()),
    }


def run_gradient_audit(
    run_dir: Path,
    store: Path,
    train_rows: pl.DataFrame,
    output_dir: Path,
) -> dict[str, object]:
    assert_analysis_rows(train_rows)
    model, checkpoint, _ = load_current_neural_run(run_dir)
    if not isinstance(model, SharedCausalTCN):
        raise TypeError("Gradient audit requires TCN")
    if (
        int(checkpoint["seed"]) != 29
        or checkpoint["tcn_settings"]["readout"] != "final"
    ):
        raise ValueError("Gradient audit requires the seed-29 final control")
    model = model.cuda().float().eval()
    dates = train_rows.get_column("trade_date").unique().sort().to_list()
    selected_dates = sorted(
        {dates[int(round(value))] for value in np.linspace(0, len(dates) - 1, 5)}
    )
    grid = train_rows.filter(
        pl.col("trade_date").is_in(selected_dates)
        & pl.col("decision_idx").is_in((0, 18, 36, 54))
    ).sort("trade_date", "decision_idx")
    assert_analysis_rows(grid)
    loader = analysis_loader(store, grid, checkpoint, 1)
    groups = gradient_parameter_groups(model)
    parameters = tuple(model.parameters())
    positions = {id(parameter): index for index, parameter in enumerate(parameters)}
    objective = checkpoint["objective"]
    rows: list[dict[str, object]] = []
    for batch in loader:
        transferred = _to_device(batch, next(model.parameters()).device)
        predictions = _predict(model, transferred).float()
        vectors: dict[int, dict[str, torch.Tensor | None]] = {}
        norms: dict[int, dict[str, float | None]] = {}
        for horizon, minutes in enumerate(HORIZONS):
            loss = objective_loss(
                predictions,
                transferred["targets"],
                select_training_label_mask(transferred["label_mask"], str(minutes)),
                str(objective["name"]),
                objective.get("temperature"),
            )
            gradients = torch.autograd.grad(
                loss,
                parameters,
                retain_graph=horizon < len(HORIZONS) - 1,
                allow_unused=True,
            )
            vectors[horizon] = {}
            norms[horizon] = {}
            for name, grouped in groups.items():
                parts = [
                    gradients[positions[id(parameter)]].reshape(-1)
                    for parameter in grouped
                    if gradients[positions[id(parameter)]] is not None
                ]
                vector = torch.cat(parts).detach() if parts else None
                vectors[horizon][name] = vector
                norms[horizon][name] = (
                    None if vector is None else float(torch.linalg.vector_norm(vector))
                )
        for left, right in itertools.combinations(range(len(HORIZONS)), 2):
            for group in groups:
                cosine, undefined = gradient_cosine(
                    vectors[left][group], vectors[right][group]
                )
                rows.append(
                    {
                        "date_idx": int(batch["date_idx"][0]),
                        "decision_idx": int(batch["decision_idx"][0]),
                        "group": group,
                        "left_horizon": HORIZONS[left],
                        "right_horizon": HORIZONS[right],
                        "cosine": cosine,
                        "undefined_reason": undefined,
                        "left_gradient_norm": norms[left][group],
                        "right_gradient_norm": norms[right][group],
                    }
                )
    frame = pl.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_parquet(frame, output_dir / "horizon_gradient_audit.parquet")
    grouped = (
        frame.group_by("group", "left_horizon", "right_horizon")
        .agg(
            pl.col("cosine").mean().alias("mean_cosine"),
            pl.col("cosine").median().alias("median_cosine"),
            (pl.col("cosine") < 0).mean().alias("fraction_negative"),
            pl.col("left_gradient_norm").mean().alias("mean_left_gradient_norm"),
            pl.col("right_gradient_norm").mean().alias("mean_right_gradient_norm"),
            pl.col("cosine").count().alias("valid_samples"),
        )
        .sort("group", "left_horizon", "right_horizon")
    )
    summary = {
        "train_only": True,
        "train_end": str(TRAIN_END),
        "sample_count": grid.height,
        "by_group_and_horizon_pair": grouped.to_dicts(),
        "single_horizon_controls": None,
    }
    atomic_json(output_dir / "horizon_gradient_summary.json", summary)
    return summary


def probe_pass(
    model: nn.Module,
    checkpoint: dict[str, object],
    store: Path,
    rows: pl.DataFrame,
):
    loader = analysis_loader(
        store, rows, checkpoint, GH200_RUNTIME.evaluation_batch_size
    )
    device = next(model.parameters()).device
    for batch in loader:
        valid_count = int(batch["sample_valid_mask"].sum())
        designs = {
            name: value[:valid_count].numpy()
            for name, value in residual_source_designs(batch).items()
        }
        transferred = _to_device(batch, device)
        with torch.inference_mode(), _autocast(device):
            predictions = _predict(model, transferred)
        targets = batch["targets"][:valid_count].numpy()
        masks = batch["label_mask"][:valid_count].numpy()
        base = rank_standardize_predictions(
            predictions[:valid_count].float().cpu().numpy(), masks
        )
        yield (
            batch,
            valid_count,
            designs,
            targets,
            masks,
            base,
            targets - base,
        )


def run_oof_residual_probes(
    fold_1_run: Path,
    fold_2_run: Path,
    store: Path,
    b2_rows: pl.DataFrame,
    b3_rows: pl.DataFrame,
    output_dir: Path,
) -> dict[str, object]:
    assert_analysis_rows(b2_rows)
    assert_analysis_rows(b3_rows)
    model_1, checkpoint_1, _ = load_current_neural_run(fold_1_run)
    model_2, checkpoint_2, _ = load_current_neural_run(fold_2_run)
    model_1 = model_1.cuda().eval()
    model_2 = model_2.cuda().eval()
    b2_dates = b2_rows.get_column("trade_date").unique().sort().to_list()
    split = len(b2_dates) // 2
    calibration_start = b2_dates[split]
    early_dates = set(b2_dates[:split])
    trade_by_date = dict(b2_rows.select("date_idx", "trade_date").unique().iter_rows())
    dimensions: dict[str, int] | None = None
    early_stats: dict[str, list[RidgeSufficientStatistics]] = {}
    all_stats: dict[str, list[RidgeSufficientStatistics]] = {}
    for batch, valid_count, designs, _, masks, _, residual in probe_pass(
        model_1, checkpoint_1, store, b2_rows
    ):
        if dimensions is None:
            dimensions = {name: value.shape[-1] for name, value in designs.items()}
            early_stats = {
                name: [RidgeSufficientStatistics(width) for _ in HORIZONS]
                for name, width in dimensions.items()
            }
            all_stats = {
                name: [RidgeSufficientStatistics(width) for _ in HORIZONS]
                for name, width in dimensions.items()
            }
        dates = batch["date_idx"][:valid_count].numpy()
        early = np.asarray(
            [trade_by_date[int(value)] in early_dates for value in dates],
            dtype=bool,
        )
        update_ridge_statistics(early_stats, designs, residual, masks, early)
        update_ridge_statistics(
            all_stats,
            designs,
            residual,
            masks,
            np.ones(valid_count, dtype=bool),
        )
    if dimensions is None:
        raise ValueError("OOF probe fit window produced no rows")
    candidates = {
        name: [
            [early_stats[name][horizon].solve(penalty) for penalty in RIDGE_PENALTIES]
            for horizon in range(len(HORIZONS))
        ]
        for name in dimensions
    }
    calibration = {
        (name, horizon, penalty): DailyICAccumulator()
        for name in dimensions
        for horizon in range(len(HORIZONS))
        for penalty in range(len(RIDGE_PENALTIES))
    }
    late_rows = b2_rows.filter(pl.col("trade_date") >= calibration_start)
    for batch, valid_count, designs, _, masks, _, residual in probe_pass(
        model_1, checkpoint_1, store, late_rows
    ):
        dates = batch["date_idx"][:valid_count].numpy()
        for name, features in designs.items():
            for horizon in range(len(HORIZONS)):
                for penalty_index, (coefficient, intercept) in enumerate(
                    candidates[name][horizon]
                ):
                    calibration[name, horizon, penalty_index].add_single(
                        features @ coefficient + intercept,
                        residual[..., horizon],
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
    correction_scores = {name: DailyICAccumulator() for name in dimensions}
    combined_scores = {name: DailyICAccumulator() for name in dimensions}
    base_score = DailyICAccumulator()
    for batch, valid_count, designs, targets, masks, base, residual in probe_pass(
        model_2, checkpoint_2, store, b3_rows
    ):
        dates = batch["date_idx"][:valid_count].numpy()
        base_score.add(base, targets, masks, dates)
        for name, features in designs.items():
            correction = np.zeros_like(targets)
            for horizon in range(len(HORIZONS)):
                coefficient, intercept = fitted[name, horizon]
                correction[..., horizon] = features @ coefficient + intercept
            correction_scores[name].add(correction, residual, masks, dates)
            combined_scores[name].add(base + correction, targets, masks, dates)
    rows: list[dict[str, object]] = []
    for name in dimensions:
        for horizon, minutes in enumerate(HORIZONS):
            base_ic = base_score.horizon_score(horizon)
            combined_ic = combined_scores[name].horizon_score(horizon)
            rows.append(
                {
                    "probe": name,
                    "horizon_minutes": minutes,
                    "residual_prediction_ic": correction_scores[name].horizon_score(
                        horizon
                    ),
                    "base_ic": base_ic,
                    "base_plus_correction_ic": combined_ic,
                    "delta_from_base": combined_ic - base_ic,
                    "selected_penalty": RIDGE_PENALTIES[selected[name, horizon]],
                    "valid_count": all_stats[name][horizon].count,
                    "feature_dim": dimensions[name],
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(pl.DataFrame(rows), output_dir / "oof_residual_probes.csv")
    summary = {
        "fit_block": "B2",
        "evaluation_block": "B3",
        "calibration_start": str(calibration_start),
        "sector_subsector_note": (
            "The selected-peer sidecar is the available audited sector/subsector "
            "diagnostic; no additional PIT sector array was synthesized."
        ),
        "results": rows,
    }
    atomic_json(output_dir / "oof_residual_probe_summary.json", summary)
    return summary
