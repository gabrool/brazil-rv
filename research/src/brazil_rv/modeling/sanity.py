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
    _run_accumulated_update,
    _to_cuda,
    build_compile_metadata,
    checkpoint_payload,
    clone_eager_reference_model,
    compile_model,
    rank_target_huber_loss,
    qualify_eager_compiled_model,
    require_compile_parity,
    validate_runtime,
    warmup_compiled_model,
)
from .metrics import sample_level_ic
from .model import build_neural_model, count_trainable_parameters
from .muon import PYTORCH_MUON_REFERENCE
from .optim import build_optimizers


def _evaluate_fixed_batch(
    model: torch.nn.Module,
    cpu_batch: dict[str, torch.Tensor],
) -> tuple[float, float, bool]:
    batch = _to_cuda(cpu_batch)
    valid_count = int(cpu_batch["sample_valid_mask"].sum())
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = _predict(model, batch)
        loss = rank_target_huber_loss(
            output[:valid_count],
            batch["targets"][:valid_count],
            batch["label_mask"][:valid_count],
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


def _validate_compiled_checkpoint_compatibility(
    model: torch.nn.Module,
    evaluation_batch: dict[str, torch.Tensor],
    muon_backend: str,
    completed_steps: int,
    validation_score: float,
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
        "context_pooled",
        "hybrid",
        muon_backend,
        11,
        completed_steps,
        validation_score,
        feature_store,
        git_commit_sha,
    )
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temporary:
        checkpoint_path = Path(temporary.name)
    try:
        torch.save(checkpoint, checkpoint_path)
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        eager_model = build_neural_model("context_pooled")
        eager_model.load_state_dict(loaded["model_state_dict"])
        eager_model.to("cuda").eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            eager_predictions = _predict(eager_model, cuda_batch).float()
    finally:
        checkpoint_path.unlink(missing_ok=True)
    if not bool(torch.isfinite(eager_predictions).all()):
        raise ValueError("Eager checkpoint restoration produced nonfinite predictions")
    maximum_difference = float(
        (compiled_predictions - eager_predictions).abs().max().cpu()
    )
    compatible = bool(
        torch.allclose(
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
        raise ValueError(
            "Memorization samples must cover "
            f"{SANITY_MEMORIZATION_SAMPLE_COUNT} distinct training dates"
        )
    cache_report = warm_feature_store_cache(feature_store)
    train_loader, evaluation_loader, sampler = create_training_loaders(
        feature_store, training, memorization_rows, "context_pooled", runtime, 11
    )
    sampler.set_epoch(0)
    smoke_batches = list(islice(train_loader, runtime.accumulation_steps))
    evaluation_batch = next(iter(evaluation_loader))
    memorization_dataset = VectorizedFeatureDataset(
        feature_store, memorization_rows, "context_pooled"
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
    smoke_distinct_date_count = int(
        torch.cat([batch["date_idx"] for batch in smoke_batches]).unique().numel()
    )
    smoke_batch_sizes = [
        int(batch["sample_valid_mask"].sum()) for batch in smoke_batches
    ]

    model = build_neural_model("context_pooled").to("cuda")
    model.eval()
    initial_model_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    smoke_optimizers, _, muon_backend = build_optimizers(model, "hybrid")
    eager_reference = clone_eager_reference_model(model)
    compile_setup = compile_model(model, runtime)
    compile_parity = qualify_eager_compiled_model(
        eager_reference,
        model,
        smoke_batches[0],
        include_backward=True,
    )
    require_compile_parity(compile_parity)
    del eager_reference
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    compile_report = warmup_compiled_model(
        model,
        smoke_batches[0],
        evaluation_batch,
    )
    compile_metadata = build_compile_metadata(
        compile_setup, compile_parity, compile_report
    )

    torch.cuda.reset_peak_memory_stats()
    model.train()
    for optimizer in smoke_optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    (
        smoke_loss_sum,
        smoke_valid_samples,
        smoke_update_succeeded,
        smoke_gradient_norm,
        smoke_predictions_finite,
    ) = _run_accumulated_update(
        model,
        smoke_batches,
        smoke_optimizers,
        {},
        check_predictions_finite=True,
    )
    smoke_peak_allocated = torch.cuda.max_memory_allocated()
    smoke_peak_reserved = torch.cuda.max_memory_reserved()
    smoke_normalized_loss = float(smoke_loss_sum) / max(smoke_valid_samples, 1)
    smoke_gradients_finite = smoke_update_succeeded and np.isfinite(smoke_gradient_norm)
    smoke_passed = (
        len(smoke_batches) == runtime.accumulation_steps
        and smoke_batch_sizes == [runtime.microbatch_size] * runtime.accumulation_steps
        and smoke_sample_count == SANITY_SMOKE_SAMPLE_COUNT
        and smoke_distinct_date_count == SANITY_SMOKE_SAMPLE_COUNT
        and smoke_valid_samples > 0
        and np.isfinite(smoke_normalized_loss)
        and bool(smoke_predictions_finite)
        and smoke_gradients_finite
        and smoke_peak_allocated < 0.8 * hardware.total_vram_bytes
        and smoke_peak_reserved < 0.9 * hardware.total_vram_bytes
    )

    model.load_state_dict(initial_model_state)
    del initial_model_state, smoke_optimizers
    torch.cuda.empty_cache()
    optimizers, _, muon_backend = build_optimizers(model, "hybrid")
    model.eval()
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    all_gradients_finite = True
    all_predictions_finite = True
    final_loss = float("inf")
    final_spearman = float("-inf")
    completed_steps = 0

    for step in range(1, SANITY_MAX_STEPS + 1):
        batch = _to_cuda(memorization_batch)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predictions = _predict(model, batch)
            sample_loss = rank_target_huber_loss(
                predictions, batch["targets"], batch["label_mask"]
            )
        sample_loss.backward()
        all_predictions_finite &= bool(torch.isfinite(predictions).all())
        gradients_finite, _ = _optimizer_update(model, optimizers, {})
        all_gradients_finite &= gradients_finite
        if not gradients_finite:
            break
        completed_steps = step
        if step == 1 or step % 10 == 0:
            final_loss, final_spearman, predictions_finite = _evaluate_fixed_batch(
                model, memorization_batch
            )
            all_predictions_finite &= predictions_finite
            if (
                final_loss < SANITY_MAX_LOSS
                and final_spearman > SANITY_MIN_SPEARMAN
                and all_predictions_finite
            ):
                break

    if completed_steps % 10:
        final_loss, final_spearman, predictions_finite = _evaluate_fixed_batch(
            model, memorization_batch
        )
        all_predictions_finite &= predictions_finite
    memorization_succeeded = (
        final_loss < SANITY_MAX_LOSS
        and final_spearman > SANITY_MIN_SPEARMAN
        and all_gradients_finite
        and all_predictions_finite
    )
    compiled_checkpoint_eager_compatible = False
    compiled_checkpoint_max_absolute_difference: float | None = None
    if memorization_succeeded:
        (
            compiled_checkpoint_eager_compatible,
            compiled_checkpoint_max_absolute_difference,
        ) = _validate_compiled_checkpoint_compatibility(
            model,
            memorization_batch,
            muon_backend,
            completed_steps,
            final_spearman,
            feature_store,
        )

    memorization_peak_allocated = torch.cuda.max_memory_allocated()
    memorization_peak_reserved = torch.cuda.max_memory_reserved()
    memorization_passed = (
        memorization_succeeded
        and compiled_checkpoint_eager_compatible
        and memorization_peak_allocated < 0.8 * hardware.total_vram_bytes
        and memorization_peak_reserved < 0.9 * hardware.total_vram_bytes
    )
    peak_allocated = max(
        smoke_peak_allocated,
        memorization_peak_allocated,
        compile_report.peak_allocated_cuda_memory_bytes,
    )
    peak_reserved = max(
        smoke_peak_reserved,
        memorization_peak_reserved,
        compile_report.peak_reserved_cuda_memory_bytes,
    )
    passed = (
        compile_parity.passed
        and smoke_passed
        and memorization_passed
        and peak_allocated < 0.8 * hardware.total_vram_bytes
        and peak_reserved < 0.9 * hardware.total_vram_bytes
    )
    created_at = datetime.now(timezone.utc)
    output_dir = RUN_OUTPUT_BASE / (
        f"sanity_context_pooled_{created_at:%Y%m%dT%H%M%S%fZ}"
    )
    if output_dir.exists():
        raise FileExistsError(f"Sanity output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    report = {
        "model_name": "context_pooled",
        "model_family": "transformer",
        "muon_backend": muon_backend,
        "muon_reference": dict(PYTORCH_MUON_REFERENCE),
        "created_at_utc": created_at.isoformat(),
        "resolved_feature_store_path": str(feature_store),
        "hardware": asdict(hardware),
        "compile": compile_metadata,
        "feature_cache_warmup": asdict(cache_report),
        "smoke": {
            "sample_count": smoke_sample_count,
            "distinct_date_count": smoke_distinct_date_count,
            "physical_batch_size": runtime.microbatch_size,
            "physical_batch_count": len(smoke_batches),
            "accumulation_steps": runtime.accumulation_steps,
            "optimizer_steps": int(smoke_update_succeeded),
            "normalized_loss": smoke_normalized_loss,
            "gradient_norm": smoke_gradient_norm,
            "all_gradients_finite": bool(smoke_gradients_finite),
            "all_predictions_finite": bool(smoke_predictions_finite),
            "peak_allocated_cuda_memory_bytes": smoke_peak_allocated,
            "peak_reserved_cuda_memory_bytes": smoke_peak_reserved,
            "passed": bool(smoke_passed),
        },
        "memorization": {
            "sample_count": SANITY_MEMORIZATION_SAMPLE_COUNT,
            "distinct_date_count": SANITY_MEMORIZATION_SAMPLE_COUNT,
            "physical_batch_size": runtime.microbatch_size,
            "accumulation_steps": 1,
            "decision_index": SANITY_DECISION_INDEX,
            "optimizer_steps": completed_steps,
            "final_rank_target_huber_loss": final_loss,
            "mean_valid_sample_spearman_ic": final_spearman,
            "all_gradients_finite": all_gradients_finite,
            "all_predictions_finite": all_predictions_finite,
            "compiled_checkpoint_eager_compatible": (
                compiled_checkpoint_eager_compatible
            ),
            "compiled_checkpoint_max_absolute_difference": (
                compiled_checkpoint_max_absolute_difference
            ),
            "peak_allocated_cuda_memory_bytes": memorization_peak_allocated,
            "peak_reserved_cuda_memory_bytes": memorization_peak_reserved,
            "passed": bool(memorization_passed),
        },
        "peak_allocated_cuda_memory_bytes": peak_allocated,
        "peak_reserved_cuda_memory_bytes": peak_reserved,
        "parameter_count": count_trainable_parameters(model),
        "passed": passed,
    }
    (output_dir / "sanity_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    if not passed:
        raise RuntimeError(
            "Cloud sanity criteria failed: "
            f"smoke_passed={smoke_passed}, "
            f"memorization_passed={memorization_passed}, "
            f"loss={final_loss}, spearman={final_spearman}, "
            f"finite_gradients={all_gradients_finite}, "
            f"finite_predictions={all_predictions_finite}, "
            f"compiled_checkpoint_eager_compatible="
            f"{compiled_checkpoint_eager_compatible}, "
            f"peak_allocated={peak_allocated}, peak_reserved={peak_reserved}"
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
