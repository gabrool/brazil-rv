from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Iterator, Sequence
from dataclasses import replace

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
    TrainingObjective,
    collect_validation_observations,
    compile_model,
    compile_training_objective,
    run_effective_batch_update,
    validation_primary_metric,
)
from .model import build_neural_model
from .optim import build_optimizer, build_scheduler
from .train import set_seeds

COMPILE_MODES = (
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)
MICROBATCH_SIZES = (64, 128, 256)
EVALUATION_BATCH_SIZES = (256, 512, 1024)
STEADY_UPDATE_COUNT = 3
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
    validation_started = time.perf_counter()
    collection_started = time.perf_counter()
    observations, _ = collect_validation_observations(
        compiled_model, validation_loader, "soft_spearman", 0.50
    )
    torch.cuda.synchronize()
    collection_seconds = time.perf_counter() - collection_started
    metric_started = time.perf_counter()
    validation_primary_metric(observations)
    primary_metric_seconds = time.perf_counter() - metric_started
    validation_seconds = time.perf_counter() - validation_started
    return {
        "settings": {
            "compile_mode": runtime.compile_mode,
            "microbatch_size": runtime.microbatch_size,
            "accumulation_steps": runtime.accumulation_steps,
            "evaluation_batch_size": runtime.evaluation_batch_size,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "seed": SEED,
        },
        "cold_first_update_seconds": cold_seconds,
        "steady_update_seconds": steady_seconds,
        "steady_update_median_seconds": statistics.median(steady_seconds),
        "complete_validation_seconds": validation_seconds,
        "validation_collection_seconds": collection_seconds,
        "validation_primary_metric_seconds": primary_metric_seconds,
        "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_cuda_bytes": torch.cuda.max_memory_reserved(),
        "completed": True,
    }


def main() -> None:
    print(json.dumps(run_benchmark(parse_args()), separators=(",", ":")))


if __name__ == "__main__":
    main()
