from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

import numpy as np
import torch

from .contract import (
    COMPILE_PARITY_PREDICTION_ATOL,
    COMPILE_PARITY_PREDICTION_RTOL,
    GH200_RUNTIME,
    PROJECT_ROOT,
    RUN_OUTPUT_BASE,
    SANITY_DECISION_INDEX,
    SANITY_MAX_LOSS,
    SANITY_MAX_STEPS,
    SANITY_MEMORIZATION_SAMPLE_COUNT,
    SANITY_MIN_SPEARMAN,
    SANITY_SMOKE_SAMPLE_COUNT,
)
from .data import (
    BatchRequest,
    VectorizedFeatureDataset,
    create_training_loaders,
    resolve_feature_store,
    select_sample_split,
    tensorize_vectorized_batch,
    validate_feature_store,
    warm_feature_store_cache,
)
from .engine import (
    _optimizer_update,
    _predict,
    _to_cuda,
    build_compile_metadata,
    checkpoint_payload,
    clone_eager_reference_model,
    compile_model,
    qualify_eager_compiled_model,
    require_compile_parity,
    run_effective_batch_update,
    soft_spearman_loss,
    validate_runtime,
    warmup_compiled_model,
)
from .metrics import sample_level_ic
from .model import build_neural_model, count_trainable_parameters
from .optim import build_optimizer, build_scheduler

SANITY_TEMPERATURE = 0.10
SANITY_SAM_RHO = 0.020
SANITY_MODEL = "tcn"


def _evaluate_fixed_batch(
    model: torch.nn.Module,
    cpu_batch: dict[str, torch.Tensor],
) -> tuple[float, float, bool]:
    batch = _to_cuda(cpu_batch)
    valid_count = int(cpu_batch["sample_valid_mask"].sum())
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = _predict(model, batch)
        loss = soft_spearman_loss(
            output[:valid_count],
            batch["targets"][:valid_count],
            batch["label_mask"][:valid_count],
            SANITY_TEMPERATURE,
        )
    predictions = output[:valid_count].float().cpu().numpy()
    targets = cpu_batch["targets"][:valid_count].numpy()
    masks = cpu_batch["label_mask"][:valid_count].numpy()
    spearman, _ = sample_level_ic(predictions, targets, masks)
    return (
        float(loss),
        float(np.nanmean(spearman)),
        bool(np.isfinite(predictions).all()),
    )


def _checkpoint_compatibility(
    model: torch.nn.Module,
    evaluation_batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    training_sample_count: int,
    feature_store: Path,
) -> tuple[bool, float]:
    model.eval()
    cuda_batch = _to_cuda(evaluation_batch)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        compiled_predictions = _predict(model, cuda_batch).float()
    git_commit_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT / "quant" / "b3-quant",
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    checkpoint = checkpoint_payload(
        model,
        optimizer,
        scheduler,
        SANITY_MODEL,
        "adamw",
        SANITY_TEMPERATURE,
        None,
        11,
        0,
        0.0,
        feature_store,
        git_commit_sha,
    )
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temporary:
        checkpoint_path = Path(temporary.name)
    try:
        torch.save(checkpoint, checkpoint_path)
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        eager_model = build_neural_model(SANITY_MODEL)
        eager_optimizer, _ = build_optimizer(eager_model)
        eager_scheduler, _, _ = build_scheduler(eager_optimizer, training_sample_count)
        eager_model.load_state_dict(loaded["model_state_dict"])
        eager_optimizer.load_state_dict(loaded["optimizer_state_dict"])
        eager_scheduler.load_state_dict(loaded["scheduler_state_dict"])
        eager_model.to("cuda").eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            eager_predictions = _predict(eager_model, cuda_batch).float()
    finally:
        checkpoint_path.unlink(missing_ok=True)
    maximum_difference = float(
        (compiled_predictions - eager_predictions).abs().max().cpu()
    )
    compatible = bool(
        torch.isfinite(eager_predictions).all()
        and torch.allclose(
            compiled_predictions,
            eager_predictions,
            atol=COMPILE_PARITY_PREDICTION_ATOL,
            rtol=COMPILE_PARITY_PREDICTION_RTOL,
        )
    )
    return compatible, maximum_difference


def main() -> None:
    runtime = GH200_RUNTIME
    hardware = validate_runtime()
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(11)
    torch.cuda.manual_seed_all(11)
    np.random.seed(11)

    feature_store = resolve_feature_store()
    sample_index = validate_feature_store(feature_store)
    training = select_sample_split(sample_index, "train")
    memorization_rows = (
        training.filter(training.get_column("decision_idx") == SANITY_DECISION_INDEX)
        .sort("trade_date")
        .head(SANITY_MEMORIZATION_SAMPLE_COUNT)
    )
    if (
        memorization_rows.height != SANITY_MEMORIZATION_SAMPLE_COUNT
        or memorization_rows.get_column("trade_date").n_unique()
        != SANITY_MEMORIZATION_SAMPLE_COUNT
    ):
        raise ValueError("Memorization samples must cover distinct training dates")

    cache_report = warm_feature_store_cache(feature_store)
    train_loader, evaluation_loader, sampler = create_training_loaders(
        feature_store, training, memorization_rows, SANITY_MODEL, runtime, 11
    )
    sampler.set_epoch(0)
    smoke_batches = list(islice(train_loader, runtime.accumulation_steps))
    evaluation_batch = next(iter(evaluation_loader))
    memorization_dataset = VectorizedFeatureDataset(
        feature_store, memorization_rows, SANITY_MODEL
    )
    memorization_batch = tensorize_vectorized_batch(
        memorization_dataset[
            BatchRequest(
                tuple(range(SANITY_MEMORIZATION_SAMPLE_COUNT)),
                SANITY_MEMORIZATION_SAMPLE_COUNT,
            )
        ]
    )
    smoke_sample_count = sum(
        int(batch["sample_valid_mask"].sum()) for batch in smoke_batches
    )
    smoke_distinct_dates = int(
        torch.cat([batch["date_idx"] for batch in smoke_batches]).unique().numel()
    )

    model = build_neural_model(SANITY_MODEL).to("cuda")
    initial_state = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    eager_reference = clone_eager_reference_model(model)
    compile_setup = compile_model(model, runtime)
    compile_parity = qualify_eager_compiled_model(
        eager_reference,
        model,
        smoke_batches[0],
        include_backward=True,
        temperature=SANITY_TEMPERATURE,
    )
    require_compile_parity(compile_parity)
    del eager_reference
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    compile_report = warmup_compiled_model(
        model, smoke_batches[0], evaluation_batch, SANITY_TEMPERATURE
    )
    compile_metadata = build_compile_metadata(
        compile_setup, compile_parity, compile_report
    )

    torch.cuda.reset_peak_memory_stats()
    model.train()
    adamw_optimizer, _ = build_optimizer(model)
    adamw_scheduler, _, _ = build_scheduler(adamw_optimizer, training.height)
    adamw_scheduler_before = adamw_scheduler.last_epoch
    adamw_smoke = run_effective_batch_update(
        model,
        smoke_batches,
        adamw_optimizer,
        adamw_scheduler,
        runtime,
        "adamw",
        SANITY_TEMPERATURE,
        None,
        check_predictions_finite=True,
    )
    adamw_smoke["scheduler_steps"] = adamw_scheduler.last_epoch - adamw_scheduler_before
    adamw_smoke["parameters_finite"] = all(
        bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
    )
    adamw_smoke["passed"] = bool(
        adamw_smoke["backward_passes"] == 8
        and adamw_smoke["scheduler_steps"] == 1
        and adamw_smoke["predictions_finite"]
        and adamw_smoke["parameters_finite"]
    )

    model.load_state_dict(initial_state)
    sam_optimizer, _ = build_optimizer(model)
    sam_scheduler, _, _ = build_scheduler(sam_optimizer, training.height)
    sam_scheduler_before = sam_scheduler.last_epoch
    sam_smoke = run_effective_batch_update(
        model,
        smoke_batches,
        sam_optimizer,
        sam_scheduler,
        runtime,
        "sam_adamw",
        SANITY_TEMPERATURE,
        SANITY_SAM_RHO,
        check_predictions_finite=True,
    )
    sam_smoke["scheduler_steps"] = sam_scheduler.last_epoch - sam_scheduler_before
    sam_smoke["parameters_finite"] = all(
        bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
    )
    sam_smoke["passed"] = bool(
        sam_smoke["backward_passes"] == 16
        and sam_smoke["scheduler_steps"] == 1
        and sam_smoke["predictions_finite"]
        and sam_smoke["parameters_finite"]
    )

    model.load_state_dict(initial_state)
    checkpoint_optimizer, _ = build_optimizer(model)
    checkpoint_scheduler, _, _ = build_scheduler(checkpoint_optimizer, training.height)
    checkpoint_compatible, checkpoint_max_difference = _checkpoint_compatibility(
        model,
        evaluation_batch,
        checkpoint_optimizer,
        checkpoint_scheduler,
        training.height,
        feature_store,
    )

    memorization_model = build_neural_model(SANITY_MODEL).to("cuda").eval()
    memorization_optimizer, _ = build_optimizer(memorization_model)
    memorization_scheduler, _, _ = build_scheduler(
        memorization_optimizer, memorization_rows.height
    )
    initial_loss, _, initial_predictions_finite = _evaluate_fixed_batch(
        memorization_model, memorization_batch
    )
    final_loss = initial_loss
    final_spearman = float("-inf")
    predictions_finite = initial_predictions_finite
    gradients_finite = True
    completed_steps = 0
    for step in range(1, SANITY_MAX_STEPS + 1):
        cuda_batch = _to_cuda(memorization_batch)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predictions = _predict(memorization_model, cuda_batch)
        loss = soft_spearman_loss(
            predictions,
            cuda_batch["targets"],
            cuda_batch["label_mask"],
            SANITY_TEMPERATURE,
        )
        if not bool(torch.isfinite(loss)):
            gradients_finite = False
            break
        loss.backward()
        predictions_finite &= bool(torch.isfinite(predictions).all())
        try:
            _optimizer_update(
                memorization_model,
                memorization_optimizer,
                memorization_scheduler,
            )
        except FloatingPointError:
            gradients_finite = False
            break
        completed_steps = step
        if step == 1 or step % 10 == 0:
            final_loss, final_spearman, finite = _evaluate_fixed_batch(
                memorization_model, memorization_batch
            )
            predictions_finite &= finite
            if (
                final_loss <= SANITY_MAX_LOSS
                and final_loss <= 0.25 * initial_loss
                and final_spearman >= SANITY_MIN_SPEARMAN
            ):
                break
    if completed_steps % 10:
        final_loss, final_spearman, finite = _evaluate_fixed_batch(
            memorization_model, memorization_batch
        )
        predictions_finite &= finite
    memorization_passed = bool(
        final_loss <= SANITY_MAX_LOSS
        and final_loss <= 0.25 * initial_loss
        and final_spearman >= SANITY_MIN_SPEARMAN
        and predictions_finite
        and gradients_finite
    )

    peak_allocated = max(
        torch.cuda.max_memory_allocated(),
        compile_report.peak_allocated_cuda_memory_bytes,
    )
    peak_reserved = max(
        torch.cuda.max_memory_reserved(),
        compile_report.peak_reserved_cuda_memory_bytes,
    )
    memory_passed = bool(
        peak_allocated < 0.8 * hardware.total_vram_bytes
        and peak_reserved < 0.9 * hardware.total_vram_bytes
    )
    passed = bool(
        compile_parity.passed
        and adamw_smoke["passed"]
        and sam_smoke["passed"]
        and memorization_passed
        and checkpoint_compatible
        and memory_passed
        and smoke_sample_count == SANITY_SMOKE_SAMPLE_COUNT
        and smoke_distinct_dates == SANITY_SMOKE_SAMPLE_COUNT
    )
    report = {
        "model_name": SANITY_MODEL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "resolved_feature_store_path": str(feature_store),
        "hardware": asdict(hardware),
        "feature_cache_warmup": asdict(cache_report),
        "compile_parity": {
            "metadata": compile_metadata,
            "passed": compile_parity.passed,
        },
        "adamw_smoke": {
            "sample_count": smoke_sample_count,
            "distinct_date_count": smoke_distinct_dates,
            **adamw_smoke,
        },
        "sam_smoke": {
            "rho": SANITY_SAM_RHO,
            "sample_count": smoke_sample_count,
            "distinct_date_count": smoke_distinct_dates,
            **sam_smoke,
        },
        "memorization": {
            "sample_count": SANITY_MEMORIZATION_SAMPLE_COUNT,
            "decision_index": SANITY_DECISION_INDEX,
            "optimizer_steps": completed_steps,
            "initial_soft_spearman_loss": initial_loss,
            "final_soft_spearman_loss": final_loss,
            "final_mean_hard_spearman": final_spearman,
            "all_gradients_finite": gradients_finite,
            "all_predictions_finite": predictions_finite,
            "passed": memorization_passed,
        },
        "checkpoint_compatibility": {
            "executed": True,
            "maximum_absolute_prediction_difference": checkpoint_max_difference,
            "passed": checkpoint_compatible,
        },
        "peak_allocated_cuda_memory_bytes": peak_allocated,
        "peak_reserved_cuda_memory_bytes": peak_reserved,
        "memory_passed": memory_passed,
        "parameter_count": count_trainable_parameters(memorization_model),
        "passed": passed,
    }
    created_at = datetime.now(timezone.utc)
    output_dir = RUN_OUTPUT_BASE / f"sanity_tcn_{created_at:%Y%m%dT%H%M%S%fZ}"
    if output_dir.exists():
        raise FileExistsError(f"Sanity output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "sanity_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    if not passed:
        raise RuntimeError(f"Cloud sanity criteria failed: {json.dumps(report)}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
