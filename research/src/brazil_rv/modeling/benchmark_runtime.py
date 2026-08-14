from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import replace

import numpy as np
import torch

from .contract import (
    BASELINE_TCN_SETTINGS,
    EFFECTIVE_BATCH_SIZE,
    GH200_RUNTIME,
    MAX_EPOCHS,
    RuntimeSettings,
    TCNArchitecture,
    architecture_for_model,
)
from .data import (
    create_training_loaders,
    load_sample_index,
    resolve_feature_store,
    select_sample_split,
)
from .engine import (
    EvaluationObservations,
    TrainingObjective,
    collect_validation_observations,
    compile_model,
    compile_training_objective,
    eager_training_objective,
    run_effective_batch_update,
    validation_primary_metric,
)
from .model import build_neural_model
from .optim import build_optimizer, build_scheduler
from .train import set_seeds

COMPILE_MODES = (
    "default",
    "max-autotune-no-cudagraphs",
)
MICROBATCH_SIZES = (64, 128, 256)
EVALUATION_BATCH_SIZES = (256, 512, 1024)
STEADY_UPDATE_COUNT = 10
PARITY_ABSOLUTE_TOLERANCE = 2e-5
SEED = 29


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temporarily benchmark the real incumbent on GH200"
    )
    parser.add_argument("--compile-mode", choices=COMPILE_MODES, default="default")
    parser.add_argument(
        "--microbatch-size", type=int, choices=MICROBATCH_SIZES, default=64
    )
    parser.add_argument(
        "--evaluation-batch-size",
        type=int,
        choices=EVALUATION_BATCH_SIZES,
        default=256,
    )
    args = parser.parse_args(arguments)
    if EFFECTIVE_BATCH_SIZE % args.microbatch_size:
        parser.error("--microbatch-size must divide 512")
    return args


def _timed_update(
    model: torch.nn.Module,
    iterator: Iterator[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    runtime: RuntimeSettings,
    objective: TrainingObjective,
) -> float:
    batches = [next(iterator) for _ in range(runtime.accumulation_steps)]
    torch.cuda.synchronize()
    started = time.perf_counter()
    run_effective_batch_update(
        model,
        batches,
        optimizer,
        scheduler,
        runtime,
        "sam_adamw",
        "soft_spearman",
        0.50,
        0.125,
        training_objective=objective,
    )
    torch.cuda.synchronize()
    return time.perf_counter() - started


def _compiled_soft_spearman_parity(
    compiled_objective: TrainingObjective,
) -> dict[str, float]:
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    predictions = (
        torch.randn((2, 158, 3), generator=generator, device="cuda") * 4
    ).round() / 4
    targets = (
        torch.randn((2, 158, 3), generator=generator, device="cuda") * 4
    ).round() / 4
    mask = torch.rand((2, 158, 3), generator=generator, device="cuda") > 0.1
    eager_predictions = predictions.clone().requires_grad_(True)
    compiled_predictions = predictions.clone().requires_grad_(True)
    eager_objective = eager_training_objective("soft_spearman", 0.50)
    eager_loss = eager_objective(eager_predictions, targets, mask)
    compiled_loss = compiled_objective(compiled_predictions, targets, mask)
    eager_loss.backward()
    compiled_loss.backward()
    torch.cuda.synchronize()
    eager_gradient = eager_predictions.grad
    compiled_gradient = compiled_predictions.grad
    if eager_gradient is None or compiled_gradient is None:
        raise RuntimeError("Compiled soft-Spearman parity produced no gradient")
    finite = (
        torch.isfinite(eager_loss)
        & torch.isfinite(compiled_loss)
        & torch.isfinite(eager_gradient).all()
        & torch.isfinite(compiled_gradient).all()
    )
    if not bool(finite):
        raise FloatingPointError("Compiled soft-Spearman parity is non-finite")
    loss_difference = float((compiled_loss - eager_loss).abs().detach())
    gradient_difference = float(
        (compiled_gradient - eager_gradient).abs().max().detach()
    )
    if (
        loss_difference > PARITY_ABSOLUTE_TOLERANCE
        or gradient_difference > PARITY_ABSOLUTE_TOLERANCE
    ):
        raise FloatingPointError(
            "Compiled soft-Spearman parity exceeded "
            f"{PARITY_ABSOLUTE_TOLERANCE}: "
            f"loss={loss_difference}, gradient={gradient_difference}"
        )
    return {
        "maximum_absolute_loss_difference": loss_difference,
        "maximum_absolute_gradient_difference": gradient_difference,
        "absolute_tolerance": PARITY_ABSOLUTE_TOLERANCE,
    }


def _timed_validation(
    model: torch.nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
) -> tuple[EvaluationObservations, float, float, float, float]:
    torch.cuda.synchronize()
    validation_started = time.perf_counter()
    collection_started = time.perf_counter()
    observations, _ = collect_validation_observations(
        model, loader, "soft_spearman", 0.50
    )
    torch.cuda.synchronize()
    collection_seconds = time.perf_counter() - collection_started
    metric_started = time.perf_counter()
    primary_metric = validation_primary_metric(observations)
    primary_metric_seconds = time.perf_counter() - metric_started
    complete_seconds = time.perf_counter() - validation_started
    return (
        observations,
        primary_metric,
        complete_seconds,
        collection_seconds,
        primary_metric_seconds,
    )


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    torch.set_float32_matmul_precision("high")
    set_seeds(SEED)
    runtime = replace(
        GH200_RUNTIME,
        microbatch_size=args.microbatch_size,
        accumulation_steps=EFFECTIVE_BATCH_SIZE // args.microbatch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        compile_mode=args.compile_mode,
    )
    store = resolve_feature_store()
    sample_index = load_sample_index(store)
    train_rows = select_sample_split(sample_index, "train")
    validation_rows = select_sample_split(sample_index, "validation")
    architecture = architecture_for_model("tcn", BASELINE_TCN_SETTINGS)
    assert isinstance(architecture, TCNArchitecture)
    train_loader, validation_loader, sampler = create_training_loaders(
        store,
        train_rows,
        validation_rows,
        "tcn",
        "enabled",
        runtime,
        SEED,
        architecture,
        "selected",
    )
    model = build_neural_model("tcn", architecture, "selected").cuda()
    optimizer, _ = build_optimizer(model)
    scheduler, _, _ = build_scheduler(optimizer, train_rows.height, MAX_EPOCHS)
    compiled_model = compile_model(model, runtime)
    compiled_objective = compile_training_objective("soft_spearman", 0.50, runtime)
    parity = _compiled_soft_spearman_parity(compiled_objective)
    sampler.set_epoch(1)
    iterator = iter(train_loader)
    torch.cuda.reset_peak_memory_stats()
    cold_seconds = _timed_update(
        compiled_model,
        iterator,
        optimizer,
        scheduler,
        runtime,
        compiled_objective,
    )
    steady_seconds = [
        _timed_update(
            compiled_model,
            iterator,
            optimizer,
            scheduler,
            runtime,
            compiled_objective,
        )
        for _ in range(STEADY_UPDATE_COUNT)
    ]
    (
        cold_observations,
        cold_primary_metric,
        cold_validation_seconds,
        _,
        _,
    ) = _timed_validation(model, validation_loader)
    (
        steady_observations,
        steady_primary_metric,
        steady_validation_seconds,
        steady_collection_seconds,
        steady_primary_metric_seconds,
    ) = _timed_validation(model, validation_loader)
    observations_match = all(
        np.array_equal(
            getattr(cold_observations, name),
            getattr(steady_observations, name),
            equal_nan=True,
        )
        for name in EvaluationObservations.__dataclass_fields__
    )
    if not observations_match:
        raise RuntimeError("Cold and steady validation observations differ")
    if cold_primary_metric != steady_primary_metric:
        raise RuntimeError("Cold and steady validation primary metrics differ")
    return {
        "settings": {
            "compile_mode": runtime.compile_mode,
            "microbatch_size": runtime.microbatch_size,
            "accumulation_steps": runtime.accumulation_steps,
            "evaluation_batch_size": runtime.evaluation_batch_size,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "steady_update_count": STEADY_UPDATE_COUNT,
            "seed": SEED,
        },
        "compiled_soft_spearman_parity": parity,
        "cold_first_update_seconds": cold_seconds,
        "steady_update_seconds": steady_seconds,
        "steady_update_median_seconds": statistics.median(steady_seconds),
        "cold_complete_validation_seconds": cold_validation_seconds,
        "steady_complete_validation_seconds": steady_validation_seconds,
        "steady_validation_collection_seconds": steady_collection_seconds,
        "steady_validation_primary_metric_seconds": steady_primary_metric_seconds,
        "validation_observations_exact_match": True,
        "validation_primary_metric_exact_match": True,
        "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_cuda_bytes": torch.cuda.max_memory_reserved(),
        "completed": True,
    }


def main() -> None:
    print(json.dumps(run_benchmark(parse_args()), separators=(",", ":")))


if __name__ == "__main__":
    main()
