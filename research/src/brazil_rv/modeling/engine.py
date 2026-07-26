from __future__ import annotations

import copy
import json
import math
import platform as system_platform
import statistics
import time
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch._functorch.config as functorch_config
from torch import nn

from .contract import (
    architecture_for_model,
    COMPILE_PARITY_GRADIENT_COSINE_MIN,
    COMPILE_PARITY_GRADIENT_MAX_ABSOLUTE_ATOL,
    COMPILE_PARITY_GRADIENT_MAX_ABSOLUTE_RTOL,
    COMPILE_PARITY_GRADIENT_RELATIVE_L2_MAX,
    COMPILE_PARITY_LOSS_ATOL,
    COMPILE_PARITY_LOSS_RTOL,
    COMPILE_PARITY_PREDICTION_ATOL,
    COMPILE_PARITY_PREDICTION_RTOL,
    COMPILE_STEADY_STATE_PASS_COUNT,
    COMPILE_WARMUP_PASS_COUNT,
    CompileEvaluationWarmupReport,
    CompileParityReport,
    CompileParityThresholds,
    CompileSetupReport,
    CompileWarmupReport,
    GH200_RUNTIME,
    GRADIENT_CLIP,
    HUBER_DELTA,
    HardwareInfo,
    RuntimeSettings,
)
from .metrics import create_metric_table
from .muon import PYTORCH_MUON_REFERENCE


def validate_runtime() -> HardwareInfo:
    runtime = GH200_RUNTIME
    if not torch.cuda.is_available():
        raise RuntimeError("Cloud runtime requires CUDA")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Cloud runtime requires exactly one visible CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected CUDA device does not support BF16")
    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    cpu_architecture = system_platform.machine()
    if total_memory < runtime.minimum_vram_bytes:
        raise RuntimeError(
            f"GH200 runtime requires at least {runtime.minimum_vram_bytes} "
            f"VRAM bytes, found {total_memory}"
        )
    if capability != runtime.expected_compute_capability:
        raise RuntimeError(
            f"GH200 runtime requires compute capability "
            f"{runtime.expected_compute_capability}, found {capability}"
        )
    if cpu_architecture != runtime.required_cpu_architecture:
        raise RuntimeError(
            f"GH200 runtime requires CPU architecture "
            f"{runtime.required_cpu_architecture}, found {cpu_architecture}"
        )
    return HardwareInfo(
        device_name=device_name,
        compute_capability=capability,
        total_vram_bytes=total_memory,
        cpu_architecture=cpu_architecture,
        platform=system_platform.platform(),
        pytorch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
    )


def compile_model(
    model: nn.Module,
    runtime: RuntimeSettings,
) -> CompileSetupReport:
    compile_api = getattr(model, "compile", None)
    if not callable(compile_api):
        raise RuntimeError("Installed PyTorch lacks callable nn.Module.compile")
    control_available = hasattr(functorch_config, "backward_pass_autocast")
    if control_available:
        functorch_config.backward_pass_autocast = "off"
        if functorch_config.backward_pass_autocast != "off":
            raise RuntimeError("Failed to set compiler backward autocast policy to off")
        policy = "explicit_off"
    else:
        policy = "legacy_implicit"
    compile_api(
        backend=runtime.compile_backend,
        mode=runtime.compile_mode,
        fullgraph=runtime.compile_fullgraph,
        dynamic=runtime.compile_dynamic,
    )
    return CompileSetupReport(
        api="nn.Module.compile",
        backend=runtime.compile_backend,
        mode=runtime.compile_mode,
        fullgraph=runtime.compile_fullgraph,
        dynamic=runtime.compile_dynamic,
        backward_pass_autocast_control_available=control_available,
        backward_pass_autocast_policy=policy,
    )


def clone_eager_reference_model(model: nn.Module) -> nn.Module:
    reference = copy.deepcopy(model)
    source_parameters = tuple(model.named_parameters())
    reference_parameters = tuple(reference.named_parameters())
    if tuple(name for name, _ in source_parameters) != tuple(
        name for name, _ in reference_parameters
    ):
        raise RuntimeError("Eager reference parameter names do not match")
    for (name, source), (_, cloned) in zip(
        source_parameters, reference_parameters, strict=True
    ):
        if not torch.equal(source, cloned):
            raise RuntimeError(f"Eager reference parameter values differ: {name}")
        if source is cloned or source.data_ptr() == cloned.data_ptr():
            raise RuntimeError(f"Eager reference parameter storage is shared: {name}")
    return reference


def _relative_l2_error(difference_norm: float, reference_norm: float) -> float:
    if reference_norm == 0.0:
        return 0.0 if difference_norm == 0.0 else float("inf")
    return difference_norm / reference_norm


def _compile_parity_report(
    eager_predictions: torch.Tensor,
    compiled_predictions: torch.Tensor,
    eager_loss_tensor: torch.Tensor,
    compiled_loss_tensor: torch.Tensor,
    eager_gradients: tuple[tuple[str, torch.Tensor | None], ...] | None = None,
    compiled_gradients: tuple[tuple[str, torch.Tensor | None], ...] | None = None,
) -> CompileParityReport:
    if (eager_gradients is None) != (compiled_gradients is None):
        raise ValueError("Both gradient snapshots must be provided together")
    if eager_predictions.shape != compiled_predictions.shape:
        raise ValueError("Eager and compiled prediction shapes differ")

    eager_predictions = eager_predictions.float()
    compiled_predictions = compiled_predictions.float()
    prediction_difference = compiled_predictions - eager_predictions
    prediction_difference_norm = float(torch.linalg.vector_norm(prediction_difference))
    eager_prediction_norm = float(torch.linalg.vector_norm(eager_predictions))
    eager_predictions_finite = bool(torch.isfinite(eager_predictions).all())
    compiled_predictions_finite = bool(torch.isfinite(compiled_predictions).all())
    prediction_allclose = bool(
        torch.allclose(
            eager_predictions,
            compiled_predictions,
            atol=COMPILE_PARITY_PREDICTION_ATOL,
            rtol=COMPILE_PARITY_PREDICTION_RTOL,
        )
    )
    prediction_max_absolute_difference = float(prediction_difference.abs().max())
    prediction_relative_l2_error = _relative_l2_error(
        prediction_difference_norm, eager_prediction_norm
    )

    eager_loss = float(eager_loss_tensor)
    compiled_loss = float(compiled_loss_tensor)
    losses_finite = math.isfinite(eager_loss) and math.isfinite(compiled_loss)
    loss_absolute_difference = abs(compiled_loss - eager_loss)
    loss_tolerance = COMPILE_PARITY_LOSS_ATOL + COMPILE_PARITY_LOSS_RTOL * abs(
        eager_loss
    )
    forward_passed = (
        eager_predictions_finite
        and compiled_predictions_finite
        and prediction_allclose
        and losses_finite
        and loss_absolute_difference <= loss_tolerance
    )

    if eager_gradients is None or compiled_gradients is None:
        return CompileParityReport(
            mode="forward_only",
            dropout_enabled=False,
            batch_size=eager_predictions.shape[0],
            passed=forward_passed,
            eager_predictions_finite=eager_predictions_finite,
            compiled_predictions_finite=compiled_predictions_finite,
            prediction_allclose=prediction_allclose,
            prediction_max_absolute_difference=prediction_max_absolute_difference,
            prediction_relative_l2_error=prediction_relative_l2_error,
            eager_loss=eager_loss,
            compiled_loss=compiled_loss,
            losses_finite=losses_finite,
            loss_absolute_difference=loss_absolute_difference,
            loss_tolerance=loss_tolerance,
            gradient_presence_match=None,
            eager_gradients_finite=None,
            compiled_gradients_finite=None,
            gradient_parameter_count=None,
            eager_gradient_l2_norm=None,
            compiled_gradient_l2_norm=None,
            eager_gradient_max_absolute=None,
            gradient_relative_l2_error=None,
            gradient_cosine_similarity=None,
            gradient_max_absolute_difference=None,
            gradient_max_absolute_tolerance=None,
        )

    eager_names = tuple(name for name, _ in eager_gradients)
    compiled_names = tuple(name for name, _ in compiled_gradients)
    if eager_names != compiled_names:
        raise ValueError("Eager and compiled gradient parameter names differ")

    gradient_presence_match = all(
        (eager_gradient is None) == (compiled_gradient is None)
        for (_, eager_gradient), (_, compiled_gradient) in zip(
            eager_gradients, compiled_gradients, strict=True
        )
    )
    eager_gradients_finite = all(
        gradient is None or bool(torch.isfinite(gradient).all())
        for _, gradient in eager_gradients
    )
    compiled_gradients_finite = all(
        gradient is None or bool(torch.isfinite(gradient).all())
        for _, gradient in compiled_gradients
    )
    gradient_parameter_count = 0
    eager_squared_norm = 0.0
    compiled_squared_norm = 0.0
    difference_squared_norm = 0.0
    dot_product = 0.0
    eager_gradient_max_absolute = 0.0
    gradient_max_absolute_difference = 0.0
    for (_, eager_gradient), (_, compiled_gradient) in zip(
        eager_gradients, compiled_gradients, strict=True
    ):
        if eager_gradient is None and compiled_gradient is None:
            continue
        gradient_parameter_count += 1
        eager_float = None if eager_gradient is None else eager_gradient.float()
        compiled_float = (
            None if compiled_gradient is None else compiled_gradient.float()
        )
        if (
            eager_float is not None
            and compiled_float is not None
            and eager_float.shape != compiled_float.shape
        ):
            raise ValueError("Eager and compiled gradient shapes differ")
        if eager_float is not None:
            eager_squared_norm += float(eager_float.square().sum())
            eager_gradient_max_absolute = max(
                eager_gradient_max_absolute,
                float(eager_float.abs().max()),
            )
        if compiled_float is not None:
            compiled_squared_norm += float(compiled_float.square().sum())
        if eager_float is None:
            difference = compiled_float
        elif compiled_float is None:
            difference = -eager_float
        else:
            difference = compiled_float - eager_float
            dot_product += float((eager_float * compiled_float).sum())
        if difference is not None:
            difference_squared_norm += float(difference.square().sum())
            gradient_max_absolute_difference = max(
                gradient_max_absolute_difference,
                float(difference.abs().max()),
            )

    eager_gradient_l2_norm = math.sqrt(eager_squared_norm)
    compiled_gradient_l2_norm = math.sqrt(compiled_squared_norm)
    difference_l2_norm = math.sqrt(difference_squared_norm)
    gradient_relative_l2_error = _relative_l2_error(
        difference_l2_norm, eager_gradient_l2_norm
    )
    if eager_gradient_l2_norm == 0.0:
        gradient_cosine_similarity = 1.0 if compiled_gradient_l2_norm == 0.0 else -1.0
    elif compiled_gradient_l2_norm == 0.0:
        gradient_cosine_similarity = -1.0
    else:
        gradient_cosine_similarity = dot_product / (
            eager_gradient_l2_norm * compiled_gradient_l2_norm
        )
    gradient_max_absolute_tolerance = (
        COMPILE_PARITY_GRADIENT_MAX_ABSOLUTE_ATOL
        + COMPILE_PARITY_GRADIENT_MAX_ABSOLUTE_RTOL * eager_gradient_max_absolute
    )
    backward_passed = (
        gradient_presence_match
        and eager_gradients_finite
        and compiled_gradients_finite
        and gradient_relative_l2_error <= COMPILE_PARITY_GRADIENT_RELATIVE_L2_MAX
        and gradient_cosine_similarity >= COMPILE_PARITY_GRADIENT_COSINE_MIN
        and gradient_max_absolute_difference <= gradient_max_absolute_tolerance
    )
    return CompileParityReport(
        mode="forward_backward",
        dropout_enabled=False,
        batch_size=eager_predictions.shape[0],
        passed=forward_passed and backward_passed,
        eager_predictions_finite=eager_predictions_finite,
        compiled_predictions_finite=compiled_predictions_finite,
        prediction_allclose=prediction_allclose,
        prediction_max_absolute_difference=prediction_max_absolute_difference,
        prediction_relative_l2_error=prediction_relative_l2_error,
        eager_loss=eager_loss,
        compiled_loss=compiled_loss,
        losses_finite=losses_finite,
        loss_absolute_difference=loss_absolute_difference,
        loss_tolerance=loss_tolerance,
        gradient_presence_match=gradient_presence_match,
        eager_gradients_finite=eager_gradients_finite,
        compiled_gradients_finite=compiled_gradients_finite,
        gradient_parameter_count=gradient_parameter_count,
        eager_gradient_l2_norm=eager_gradient_l2_norm,
        compiled_gradient_l2_norm=compiled_gradient_l2_norm,
        eager_gradient_max_absolute=eager_gradient_max_absolute,
        gradient_relative_l2_error=gradient_relative_l2_error,
        gradient_cosine_similarity=gradient_cosine_similarity,
        gradient_max_absolute_difference=gradient_max_absolute_difference,
        gradient_max_absolute_tolerance=gradient_max_absolute_tolerance,
    )


def _rank_target_huber_sample_losses(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    label_mask: torch.Tensor,
    delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    difference = predictions.float() - targets.float()
    absolute = difference.abs()
    elementwise = torch.where(
        absolute <= delta,
        0.5 * difference.square(),
        delta * (absolute - 0.5 * delta),
    )
    mask = label_mask.bool()
    label_counts = mask.sum(dim=1)
    valid_horizons = label_counts > 0
    horizon_loss = (elementwise * mask).sum(dim=1) / label_counts.clamp_min(1)
    valid_horizon_counts = valid_horizons.sum(dim=1)
    valid_samples = valid_horizon_counts > 0
    sample_loss = (horizon_loss * valid_horizons).sum(
        dim=1
    ) / valid_horizon_counts.clamp_min(1)
    return sample_loss, valid_samples


def rank_target_huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    label_mask: torch.Tensor,
    delta: float = HUBER_DELTA,
) -> torch.Tensor:
    sample_loss, valid_samples = _rank_target_huber_sample_losses(
        predictions, targets, label_mask, delta
    )
    if bool(valid_samples.any()):
        return sample_loss[valid_samples].mean()
    return predictions.float().sum() * 0.0


def _loss_sum(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    label_mask: torch.Tensor,
) -> torch.Tensor:
    sample_loss, valid_samples = _rank_target_huber_sample_losses(
        predictions, targets, label_mask, HUBER_DELTA
    )
    return sample_loss[valid_samples].sum()


def _to_cuda(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    model_keys = (
        ("tabular_features", "equity_mask")
        if "tabular_features" in batch
        else (
            "patches",
            "history_patch_mask",
            "instrument_mask",
            "slow_features",
            "state_position",
        )
    )
    return {
        key: batch[key].to("cuda", non_blocking=True)
        for key in (*model_keys, "targets", "label_mask")
    }


def _predict(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if "tabular_features" in batch:
        return model(batch["tabular_features"], batch["equity_mask"])
    return model(
        batch["patches"],
        batch["history_patch_mask"],
        batch["instrument_mask"],
        batch["slow_features"],
        batch["state_position"],
    )


def _gradient_snapshot(
    model: nn.Module,
) -> tuple[tuple[str, torch.Tensor | None], ...]:
    return tuple(
        (name, None if parameter.grad is None else parameter.grad.detach().clone())
        for name, parameter in model.named_parameters()
    )


def qualify_eager_compiled_model(
    eager_model: nn.Module,
    compiled_model: nn.Module,
    cpu_batch: dict[str, torch.Tensor],
    *,
    include_backward: bool,
) -> CompileParityReport:
    eager_parameters = tuple(eager_model.named_parameters())
    compiled_parameters = tuple(compiled_model.named_parameters())
    if tuple(name for name, _ in eager_parameters) != tuple(
        name for name, _ in compiled_parameters
    ):
        raise RuntimeError("Eager and compiled parameter names differ")
    for (name, eager_parameter), (_, compiled_parameter) in zip(
        eager_parameters, compiled_parameters, strict=True
    ):
        if not torch.equal(eager_parameter, compiled_parameter):
            raise RuntimeError(f"Eager and compiled initial parameters differ: {name}")

    eager_training = eager_model.training
    compiled_training = compiled_model.training
    batch = _to_cuda(cpu_batch)
    try:
        eager_model.eval()
        compiled_model.eval()
        eager_model.zero_grad(set_to_none=True)
        compiled_model.zero_grad(set_to_none=True)

        with (
            torch.enable_grad() if include_backward else torch.no_grad(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        ):
            eager_predictions = _predict(eager_model, batch)
            eager_loss = rank_target_huber_loss(
                eager_predictions, batch["targets"], batch["label_mask"]
            )
        if include_backward:
            eager_loss.backward()
            eager_gradients = _gradient_snapshot(eager_model)
        else:
            eager_gradients = None
        eager_predictions = eager_predictions.detach().clone()
        eager_loss = eager_loss.detach().clone()

        with (
            torch.enable_grad() if include_backward else torch.no_grad(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        ):
            compiled_predictions = _predict(compiled_model, batch)
            compiled_loss = rank_target_huber_loss(
                compiled_predictions, batch["targets"], batch["label_mask"]
            )
        if include_backward:
            compiled_loss.backward()
            compiled_gradients = _gradient_snapshot(compiled_model)
        else:
            compiled_gradients = None
        compiled_predictions = compiled_predictions.detach().clone()
        compiled_loss = compiled_loss.detach().clone()
        torch.cuda.synchronize()
        return _compile_parity_report(
            eager_predictions,
            compiled_predictions,
            eager_loss,
            compiled_loss,
            eager_gradients,
            compiled_gradients,
        )
    finally:
        eager_model.zero_grad(set_to_none=True)
        compiled_model.zero_grad(set_to_none=True)
        eager_model.train(eager_training)
        compiled_model.train(compiled_training)


def require_compile_parity(report: CompileParityReport) -> None:
    if not report.passed:
        details = json.dumps(asdict(report), indent=2, sort_keys=True)
        raise RuntimeError(f"Eager/compiled qualification failed:\n{details}")


def build_compile_metadata(
    setup: CompileSetupReport,
    parity: CompileParityReport,
    warmup: CompileWarmupReport | CompileEvaluationWarmupReport,
) -> dict[str, object]:
    return {
        "enabled": True,
        "eager_fallback_allowed": False,
        "setup": asdict(setup),
        "parity_thresholds": asdict(CompileParityThresholds()),
        "parity": asdict(parity),
        "warmup": asdict(warmup),
    }


def _timed_training_warmup_pass(
    model: nn.Module, batch: dict[str, torch.Tensor]
) -> float:
    model.zero_grad(set_to_none=True)
    try:
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predictions = _predict(model, batch)
            loss = rank_target_huber_loss(
                predictions, batch["targets"], batch["label_mask"]
            )
        loss.backward()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        if not bool(torch.isfinite(predictions).all()):
            raise ValueError("Compiled training warmup produced nonfinite predictions")
        if any(
            parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ):
            raise ValueError("Compiled training warmup produced nonfinite gradients")
        return seconds
    finally:
        model.zero_grad(set_to_none=True)


def _timed_evaluation_warmup_pass(
    model: nn.Module, batch: dict[str, torch.Tensor]
) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        predictions = _predict(model, batch)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    if not bool(torch.isfinite(predictions).all()):
        raise ValueError("Compiled evaluation warmup produced nonfinite predictions")
    return seconds


def warmup_compiled_model(
    model: nn.Module,
    training_batch: dict[str, torch.Tensor],
    evaluation_batch: dict[str, torch.Tensor],
) -> CompileWarmupReport:
    torch.cuda.reset_peak_memory_stats()
    training_cuda = _to_cuda(training_batch)
    evaluation_cuda = _to_cuda(evaluation_batch)
    model.train()
    training_pass_seconds = tuple(
        _timed_training_warmup_pass(model, training_cuda)
        for _ in range(COMPILE_WARMUP_PASS_COUNT)
    )
    model.eval()
    evaluation_pass_seconds = tuple(
        _timed_evaluation_warmup_pass(model, evaluation_cuda)
        for _ in range(COMPILE_WARMUP_PASS_COUNT)
    )
    return CompileWarmupReport(
        training_pass_seconds=training_pass_seconds,
        training_steady_state_median_seconds=statistics.median(
            training_pass_seconds[-COMPILE_STEADY_STATE_PASS_COUNT:]
        ),
        evaluation_pass_seconds=evaluation_pass_seconds,
        evaluation_steady_state_median_seconds=statistics.median(
            evaluation_pass_seconds[-COMPILE_STEADY_STATE_PASS_COUNT:]
        ),
        peak_allocated_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_cuda_memory_bytes=torch.cuda.max_memory_reserved(),
    )


def warmup_compiled_evaluation(
    model: nn.Module, evaluation_batch: dict[str, torch.Tensor]
) -> CompileEvaluationWarmupReport:
    torch.cuda.reset_peak_memory_stats()
    cuda_batch = _to_cuda(evaluation_batch)
    model.eval()
    pass_seconds = tuple(
        _timed_evaluation_warmup_pass(model, cuda_batch)
        for _ in range(COMPILE_WARMUP_PASS_COUNT)
    )
    return CompileEvaluationWarmupReport(
        evaluation_pass_seconds=pass_seconds,
        evaluation_steady_state_median_seconds=statistics.median(
            pass_seconds[-COMPILE_STEADY_STATE_PASS_COUNT:]
        ),
        peak_allocated_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_cuda_memory_bytes=torch.cuda.max_memory_reserved(),
    )


def _optimizer_update(
    model: nn.Module,
    optimizers: dict[str, torch.optim.Optimizer],
    schedulers: dict[str, torch.optim.lr_scheduler.LambdaLR],
) -> tuple[bool, float]:
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
    update_succeeded = bool(torch.isfinite(gradient_norm))
    if update_succeeded:
        for optimizer in optimizers.values():
            optimizer.step()
        for scheduler in schedulers.values():
            scheduler.step()
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    return update_succeeded, float(gradient_norm)


def _run_accumulated_update(
    model: nn.Module,
    effective_batch: list[dict[str, torch.Tensor]],
    optimizers: dict[str, torch.optim.Optimizer],
    schedulers: dict[str, torch.optim.lr_scheduler.LambdaLR],
    *,
    check_predictions_finite: bool = False,
) -> tuple[torch.Tensor, int, bool, float, bool | None]:
    if not effective_batch:
        raise ValueError("Effective batch must contain at least one physical batch")
    effective_valid_samples = sum(
        int(batch["label_mask"].any(dim=(1, 2)).sum()) for batch in effective_batch
    )
    denominator = max(effective_valid_samples, 1)
    effective_loss_sum: torch.Tensor | None = None
    predictions_finite: bool | None = True if check_predictions_finite else None
    for buffered_batch in effective_batch:
        batch = _to_cuda(buffered_batch)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predictions = _predict(model, batch)
            microbatch_loss_sum = _loss_sum(
                predictions,
                batch["targets"],
                batch["label_mask"],
            )
        if check_predictions_finite:
            predictions_finite = bool(
                predictions_finite and bool(torch.isfinite(predictions).all())
            )
        (microbatch_loss_sum / denominator).backward()
        detached_loss = microbatch_loss_sum.detach()
        effective_loss_sum = (
            detached_loss
            if effective_loss_sum is None
            else effective_loss_sum + detached_loss
        )
    update_succeeded, gradient_norm = _optimizer_update(model, optimizers, schedulers)
    if effective_loss_sum is None:
        raise RuntimeError("Effective batch contains no physical batches")
    return (
        effective_loss_sum,
        effective_valid_samples,
        update_succeeded,
        gradient_norm,
        predictions_finite,
    )


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    optimizers: dict[str, torch.optim.Optimizer],
    schedulers: dict[str, torch.optim.lr_scheduler.LambdaLR],
    runtime: RuntimeSettings,
) -> dict[str, float | int]:
    model.train()
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    valid_sample_count = 0
    optimizer_steps = 0
    gradient_norms: list[float] = []
    effective_batch: list[dict[str, torch.Tensor]] = []

    for cpu_batch in loader:
        effective_batch.append(cpu_batch)
        if len(effective_batch) != runtime.accumulation_steps:
            continue
        (
            effective_loss_sum,
            effective_valid_samples,
            update_succeeded,
            gradient_norm,
            _,
        ) = _run_accumulated_update(
            model,
            effective_batch,
            optimizers,
            schedulers,
        )
        loss_sum += float(effective_loss_sum)
        valid_sample_count += effective_valid_samples
        if update_succeeded:
            optimizer_steps += 1
        gradient_norms.append(gradient_norm)
        effective_batch.clear()

    if effective_batch:
        raise ValueError("Training epoch ended inside an effective batch")
    if valid_sample_count == 0:
        raise ValueError("Training epoch contains no valid labeled sample")
    return {
        "optimizer_steps": optimizer_steps,
        "train_loss": loss_sum / valid_sample_count,
        "mean_gradient_norm": float(np.mean(gradient_norms)),
        "maximum_gradient_norm": float(np.max(gradient_norms)),
        "muon_lr": (
            optimizers["muon"].param_groups[0]["lr"] if "muon" in optimizers else 0.0
        ),
        "adamw_lr": optimizers["adamw"].param_groups[0]["lr"],
    }


def _filter_evaluation_rows(
    predictions: torch.Tensor,
    cpu_batch: dict[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    valid = cpu_batch["sample_valid_mask"].numpy().astype(bool, copy=False)
    valid_count = int(valid.sum())
    return {
        "predictions": predictions[:valid_count].float().cpu().numpy(),
        **{
            key: cpu_batch[key].numpy()[valid]
            for key in (
                "targets",
                "raw_returns",
                "label_mask",
                "date_idx",
                "decision_idx",
            )
        },
    }


def evaluate_model(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    model.eval()
    total_loss = 0.0
    valid_sample_count = 0
    collected: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "predictions",
            "targets",
            "raw_returns",
            "label_mask",
            "date_idx",
            "decision_idx",
        )
    }
    with torch.no_grad():
        for cpu_batch in loader:
            batch = _to_cuda(cpu_batch)
            valid_count = int(cpu_batch["sample_valid_mask"].sum())
            loss_count = int(
                cpu_batch["label_mask"][:valid_count].any(dim=(1, 2)).sum()
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictions = _predict(model, batch)
                loss_sum = _loss_sum(
                    predictions[:valid_count],
                    batch["targets"][:valid_count],
                    batch["label_mask"][:valid_count],
                )
            total_loss += float(loss_sum)
            valid_sample_count += loss_count
            valid_arrays = _filter_evaluation_rows(predictions, cpu_batch)
            for key, values in valid_arrays.items():
                collected[key].append(values)

    arrays = {key: np.concatenate(parts, axis=0) for key, parts in collected.items()}
    summary, daily_rows = create_metric_table(
        arrays["predictions"],
        arrays["targets"],
        arrays["raw_returns"],
        arrays["label_mask"],
        arrays["date_idx"],
        arrays["decision_idx"],
    )
    if valid_sample_count == 0:
        raise ValueError("Evaluation split contains no valid labeled sample")
    summary["rank_target_huber_loss"] = total_loss / valid_sample_count
    return summary, daily_rows


def checkpoint_payload(
    model: nn.Module,
    model_name: str,
    optimizer_variant: str,
    muon_backend: str | None,
    seed: int,
    epoch: int,
    validation_score: float,
    feature_store: Path,
    git_commit_sha: str,
) -> dict[str, object]:
    if getattr(model, "model_name", None) != model_name:
        raise ValueError("Checkpoint model name does not match the model")
    architecture = architecture_for_model(model_name)

    return {
        "muon_backend": muon_backend,
        "muon_reference": dict(PYTORCH_MUON_REFERENCE),
        "model_name": model_name,
        "optimizer_variant": optimizer_variant,
        "seed": seed,
        "epoch": epoch,
        "validation_score": validation_score,
        "model_state_dict": model.state_dict(),
        "architecture_constants": asdict(architecture),
        "resolved_feature_store_path": str(feature_store),
        "git_commit_sha": git_commit_sha,
    }
