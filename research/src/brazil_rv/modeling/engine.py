from __future__ import annotations

import copy
import json
import math
import platform as system_platform
import statistics
import time
from collections.abc import Iterable, Sized
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch._functorch.config as functorch_config
from torch import nn

from .context_ablation import NO_CONTEXT_ABLATION, ResolvedContextAblation
from .contract import (
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
    NEURAL_OBJECTIVES,
    NeuralArchitecture,
    RuntimeSettings,
    SAM_NORM_EPS,
    SAM_RHOS,
    SOFT_RANK_STANDARDIZATION_EPS,
    SOFT_RANK_TEMPERATURES,
    SOFT_SPEARMAN_CORRELATION_EPS,
    TCNSettings,
    expected_trainable_parameter_count,
)
from .metrics import create_metric_table


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


def validate_soft_rank_temperature(temperature: float) -> float:
    if temperature not in SOFT_RANK_TEMPERATURES:
        raise ValueError(
            f"Soft-rank temperature must be one of {SOFT_RANK_TEMPERATURES}"
        )
    return temperature


def validate_sam_rho(rho: float) -> float:
    if rho not in SAM_RHOS:
        raise ValueError(f"SAM rho must be one of {SAM_RHOS}")
    return rho


def experiment_decimal(value: float, minimum_fraction_digits: int) -> str:
    text = f"{value:.10f}".rstrip("0")
    whole, fraction = text.split(".")
    return f"{whole}p{fraction.ljust(minimum_fraction_digits, '0')}"


def validate_neural_objective(
    objective: str, temperature: float | None
) -> tuple[str, float | None]:
    if objective not in NEURAL_OBJECTIVES:
        raise ValueError(f"Neural objective must be one of {NEURAL_OBJECTIVES}")
    if objective == "soft_spearman":
        if temperature is None:
            raise ValueError("soft_spearman requires a soft-rank temperature")
        return objective, validate_soft_rank_temperature(temperature)
    if temperature is not None:
        raise ValueError("rank_huber does not accept a soft-rank temperature")
    return objective, None


def objective_metadata(objective: str, temperature: float | None) -> dict[str, object]:
    objective, temperature = validate_neural_objective(objective, temperature)
    if objective == "soft_spearman":
        return {
            "name": objective,
            "temperature": temperature,
            "score_standardization": "masked_cross_sectional",
            "soft_rank": "pairwise_sigmoid",
            "aggregation": "equal_valid_cross_section_horizon",
            "reported_validation_metric": "hard_spearman",
        }
    return {
        "name": objective,
        "temperature": None,
        "delta": HUBER_DELTA,
        "target": "centered_cross_sectional_midrank",
        "aggregation": "equal_valid_sample_then_horizon_then_equity",
        "reported_validation_metric": "hard_spearman",
    }


def sam_metadata(
    optimizer_variant: str, sam_rho: float | None
) -> dict[str, object] | None:
    if optimizer_variant == "adamw":
        if sam_rho is not None:
            raise ValueError("Ordinary AdamW does not accept a SAM rho")
        return None
    if optimizer_variant != "sam_adamw" or sam_rho is None:
        raise ValueError("SAM-AdamW requires a supported positive rho")
    return {
        "rho": validate_sam_rho(sam_rho),
        "norm": "l2",
        "adaptive": False,
        "base_optimizer": "adamw",
        "same_batch_replay": True,
        "same_rng_replay": True,
        "backward_passes_per_update": 16,
    }


def _soft_spearman_loss_sum(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    label_mask: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    validate_soft_rank_temperature(temperature)
    with torch.autocast(device_type=predictions.device.type, enabled=False):
        scores = predictions.float().transpose(1, 2)
        target_ranks = targets.float().transpose(1, 2)
        mask = label_mask.bool().transpose(1, 2)
        counts = mask.sum(dim=-1)
        valid_groups = counts >= 2
        safe_counts = counts.clamp_min(1)

        score_mean = (scores * mask).sum(dim=-1) / safe_counts
        score_centered = (scores - score_mean.unsqueeze(-1)) * mask
        score_variance = score_centered.square().sum(dim=-1) / safe_counts
        standardized = score_centered / torch.sqrt(
            score_variance.unsqueeze(-1) + SOFT_RANK_STANDARDIZATION_EPS
        )

        equity_count = scores.shape[-1]
        not_self = ~torch.eye(
            equity_count, dtype=torch.bool, device=scores.device
        ).reshape(1, 1, equity_count, equity_count)
        pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2) & not_self
        pairwise = (
            standardized.unsqueeze(-1) - standardized.unsqueeze(-2)
        ) / temperature
        soft_ranks = (1.0 + (torch.sigmoid(pairwise) * pair_mask).sum(dim=-1)) * mask

        soft_mean = soft_ranks.sum(dim=-1) / safe_counts
        soft_centered = (soft_ranks - soft_mean.unsqueeze(-1)) * mask
        target_mean = (target_ranks * mask).sum(dim=-1) / safe_counts
        target_centered = (target_ranks - target_mean.unsqueeze(-1)) * mask
        covariance = (soft_centered * target_centered).sum(dim=-1)
        denominator = (
            (soft_centered.square().sum(dim=-1) * target_centered.square().sum(dim=-1))
            .clamp_min(SOFT_SPEARMAN_CORRELATION_EPS)
            .sqrt()
        )
        correlation = covariance / denominator
        group_losses = (1.0 - correlation) * valid_groups
        return group_losses.sum(), valid_groups.sum()


def soft_spearman_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    label_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    loss_sum, group_count = _soft_spearman_loss_sum(
        predictions, targets, label_mask, temperature
    )
    return loss_sum / group_count.clamp_min(1)


def _rank_huber_sample_losses(
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


def _rank_huber_loss_sum(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    label_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sample_loss, valid_samples = _rank_huber_sample_losses(
        predictions, targets, label_mask, HUBER_DELTA
    )
    return sample_loss[valid_samples].sum(), valid_samples.sum()


def rank_huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    label_mask: torch.Tensor,
) -> torch.Tensor:
    loss_sum, sample_count = _rank_huber_loss_sum(predictions, targets, label_mask)
    return loss_sum / sample_count.clamp_min(1)


def _objective_loss_sum(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    label_mask: torch.Tensor,
    objective: str,
    temperature: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    objective, temperature = validate_neural_objective(objective, temperature)
    if objective == "soft_spearman":
        assert temperature is not None
        return _soft_spearman_loss_sum(predictions, targets, label_mask, temperature)
    return _rank_huber_loss_sum(predictions, targets, label_mask)


def objective_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    label_mask: torch.Tensor,
    objective: str,
    temperature: float | None,
) -> torch.Tensor:
    loss_sum, loss_count = _objective_loss_sum(
        predictions, targets, label_mask, objective, temperature
    )
    return loss_sum / loss_count.clamp_min(1)


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
    objective: str,
    temperature: float | None,
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

        with torch.enable_grad() if include_backward else torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                eager_predictions = _predict(eager_model, batch)
            eager_loss = objective_loss(
                eager_predictions,
                batch["targets"],
                batch["label_mask"],
                objective,
                temperature,
            )
        if include_backward:
            eager_loss.backward()
            eager_gradients = _gradient_snapshot(eager_model)
        else:
            eager_gradients = None
        eager_predictions = eager_predictions.detach().clone()
        eager_loss = eager_loss.detach().clone()

        with torch.enable_grad() if include_backward else torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                compiled_predictions = _predict(compiled_model, batch)
            compiled_loss = objective_loss(
                compiled_predictions,
                batch["targets"],
                batch["label_mask"],
                objective,
                temperature,
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
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    objective: str,
    temperature: float | None,
) -> float:
    model.zero_grad(set_to_none=True)
    try:
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predictions = _predict(model, batch)
        loss = objective_loss(
            predictions,
            batch["targets"],
            batch["label_mask"],
            objective,
            temperature,
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
    objective: str,
    temperature: float | None,
) -> CompileWarmupReport:
    torch.cuda.reset_peak_memory_stats()
    training_cuda = _to_cuda(training_batch)
    evaluation_cuda = _to_cuda(evaluation_batch)
    model.train()
    training_pass_seconds = tuple(
        _timed_training_warmup_pass(model, training_cuda, objective, temperature)
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
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> torch.Tensor:
    parameters = tuple(model.parameters())
    reference = parameters[0]
    try:
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP)
        if not _host_flags(torch.isfinite(gradient_norm))[0]:
            raise FloatingPointError("Gradient clipping produced a non-finite norm")
        optimizer.step()
        if not _host_flags(_tensors_finite(parameters, reference))[0]:
            raise FloatingPointError("Optimizer update produced non-finite parameters")
        if scheduler is not None:
            scheduler.step()
        return gradient_norm.detach()
    finally:
        optimizer.zero_grad(set_to_none=True)


def _objective_loss_count(
    effective_batch: list[dict[str, torch.Tensor]], objective: str
) -> int:
    if objective == "soft_spearman":
        return sum(
            int((batch["label_mask"].sum(dim=1) >= 2).sum())
            for batch in effective_batch
        )
    return sum(
        int(batch["label_mask"].any(dim=(1, 2)).sum()) for batch in effective_batch
    )


def _gradient_l2_norm(model: nn.Module) -> torch.Tensor:
    squared = [
        parameter.grad.detach().float().square().sum()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not squared:
        return next(model.parameters()).new_zeros((), dtype=torch.float32)
    return torch.sqrt(torch.stack(squared).sum())


def _tensors_finite(
    tensors: Iterable[torch.Tensor], reference: torch.Tensor
) -> torch.Tensor:
    predicates = [torch.isfinite(tensor).all() for tensor in tensors]
    if not predicates:
        return torch.ones((), dtype=torch.bool, device=reference.device)
    return torch.stack(predicates).all()


def _tensor_pairs_equal(
    pairs: Iterable[tuple[torch.Tensor, torch.Tensor]], reference: torch.Tensor
) -> torch.Tensor:
    predicates = [torch.eq(left, right).all() for left, right in pairs]
    if not predicates:
        return torch.ones((), dtype=torch.bool, device=reference.device)
    return torch.stack(predicates).all()


def _host_flags(*flags: torch.Tensor) -> tuple[bool, ...]:
    """Synchronize one stacked collection of device predicates."""
    return tuple(bool(value) for value in torch.stack(flags).tolist())


def _host_floats(*values: torch.Tensor) -> tuple[float, ...]:
    """Synchronize one stacked collection of device diagnostics."""
    scalars = torch.stack([value.detach().float() for value in values]).tolist()
    return tuple(float(value) for value in scalars)


def _accumulate_objective_gradients(
    model: nn.Module,
    effective_batch: list[dict[str, torch.Tensor]],
    objective: str,
    temperature: float | None,
    loss_count: int,
    *,
    check_predictions_finite: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    reference = next(model.parameters())
    loss_sum = reference.new_zeros((), dtype=torch.float32)
    losses_finite = torch.ones((), dtype=torch.bool, device=reference.device)
    predictions_finite = (
        torch.ones((), dtype=torch.bool, device=reference.device)
        if check_predictions_finite
        else None
    )
    for buffered_batch in effective_batch:
        batch = _to_cuda(buffered_batch)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predictions = _predict(model, batch)
        microbatch_loss_sum, _ = _objective_loss_sum(
            predictions,
            batch["targets"],
            batch["label_mask"],
            objective,
            temperature,
        )
        detached_loss = microbatch_loss_sum.detach()
        loss_sum = loss_sum + detached_loss
        losses_finite = losses_finite & torch.isfinite(detached_loss)
        if predictions_finite is not None:
            predictions_finite = predictions_finite & torch.isfinite(predictions).all()
        (microbatch_loss_sum / loss_count).backward()
    gradients_finite = _tensors_finite(
        (
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ),
        reference,
    )
    loss_ok, gradients_ok = _host_flags(losses_finite, gradients_finite)
    if not loss_ok:
        raise FloatingPointError("Training objective loss is non-finite")
    if not gradients_ok:
        raise FloatingPointError("Training objective gradients are non-finite")
    return loss_sum, predictions_finite


def _rng_state(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    parameter = next(model.parameters())
    cuda_state = (
        torch.cuda.get_rng_state(parameter.device)
        if parameter.device.type == "cuda"
        else None
    )
    return torch.get_rng_state(), cuda_state


def _restore_rng_state(
    model: nn.Module, state: tuple[torch.Tensor, torch.Tensor | None]
) -> None:
    cpu_state, cuda_state = state
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, next(model.parameters()).device)


def _rng_states_equal(
    left: tuple[torch.Tensor, torch.Tensor | None],
    right: tuple[torch.Tensor, torch.Tensor | None],
) -> bool:
    left_cpu, left_cuda = left
    right_cpu, right_cuda = right
    if not torch.equal(left_cpu, right_cpu):
        return False
    if left_cuda is None or right_cuda is None:
        return left_cuda is right_cuda
    return torch.equal(left_cuda, right_cuda)


def _apply_sam_perturbation(
    parameters: tuple[nn.Parameter, ...], scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    reference = parameters[0]
    squared_norm = reference.new_zeros((), dtype=torch.float32)
    perturbations_finite = torch.ones((), dtype=torch.bool, device=reference.device)
    with torch.no_grad():
        for parameter in parameters:
            if parameter.grad is None:
                continue
            perturbation = parameter.grad.detach().float() * scale
            squared_norm = squared_norm + perturbation.square().sum()
            perturbations_finite = (
                perturbations_finite & torch.isfinite(perturbation).all()
            )
            parameter.add_(perturbation)
    return torch.sqrt(squared_norm), perturbations_finite


def _restore_parameters(
    parameters: tuple[nn.Parameter, ...], snapshot: list[torch.Tensor]
) -> None:
    with torch.no_grad():
        for parameter, original in zip(parameters, snapshot, strict=True):
            parameter.copy_(original)


def _run_adamw_update(
    model: nn.Module,
    effective_batch: list[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    objective: str,
    temperature: float | None,
    loss_count: int,
    check_predictions_finite: bool,
) -> dict[str, float | int | bool | None]:
    optimizer.zero_grad(set_to_none=True)
    loss_sum, predictions_finite = _accumulate_objective_gradients(
        model,
        effective_batch,
        objective,
        temperature,
        loss_count,
        check_predictions_finite=check_predictions_finite,
    )
    gradient_norm = _optimizer_update(model, optimizer, scheduler)
    diagnostic_tensors = [loss_sum, gradient_norm]
    if predictions_finite is not None:
        diagnostic_tensors.append(predictions_finite.float())
    diagnostics = _host_floats(*diagnostic_tensors)
    return {
        "loss_sum": diagnostics[0],
        "loss_count": loss_count,
        "gradient_norm": diagnostics[1],
        "first_pass_gradient_norm": None,
        "perturbation_norm": None,
        "second_pass_gradient_norm": None,
        "predictions_finite": (
            None if predictions_finite is None else bool(diagnostics[2])
        ),
        "backward_passes": len(effective_batch),
        "rng_replay_exact": None,
        "all_finite": True,
    }


def _run_sam_update(
    model: nn.Module,
    effective_batch: list[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    objective: str,
    temperature: float | None,
    rho: float,
    loss_count: int,
    check_predictions_finite: bool,
) -> dict[str, float | int | bool | None]:
    rho = validate_sam_rho(rho)
    parameters = tuple(model.parameters())
    reference = parameters[0]
    initial_parameters = [parameter.detach().clone() for parameter in parameters]
    rng_state = _rng_state(model)
    optimizer.zero_grad(set_to_none=True)
    try:
        first_loss_sum, first_predictions_finite = _accumulate_objective_gradients(
            model,
            effective_batch,
            objective,
            temperature,
            loss_count,
            check_predictions_finite=check_predictions_finite,
        )
        first_pass_end_rng = _rng_state(model)
        first_gradient_norm = _gradient_l2_norm(model)
        if not _host_flags(torch.isfinite(first_gradient_norm))[0]:
            raise FloatingPointError("First-pass SAM gradient norm is non-finite")

        scale = rho / (first_gradient_norm + SAM_NORM_EPS)
        perturbation_norm, perturbations_finite = _apply_sam_perturbation(
            parameters, scale
        )
        perturbations_ok, perturbation_norm_ok = _host_flags(
            perturbations_finite, torch.isfinite(perturbation_norm)
        )
        if not perturbations_ok:
            raise FloatingPointError("SAM perturbation is non-finite")
        if not perturbation_norm_ok:
            raise FloatingPointError("SAM perturbation norm is non-finite")

        optimizer.zero_grad(set_to_none=True)
        _restore_rng_state(model, rng_state)
        try:
            second_loss_sum, second_predictions_finite = (
                _accumulate_objective_gradients(
                    model,
                    effective_batch,
                    objective,
                    temperature,
                    loss_count,
                    check_predictions_finite=check_predictions_finite,
                )
            )
            second_pass_end_rng = _rng_state(model)
        finally:
            _restore_parameters(parameters, initial_parameters)

        restoration_exact, restored_finite = _host_flags(
            _tensor_pairs_equal(
                zip(parameters, initial_parameters, strict=True), reference
            ),
            _tensors_finite(parameters, reference),
        )
        if not restoration_exact:
            raise RuntimeError("SAM parameter restoration was not exact")
        if not restored_finite:
            raise FloatingPointError("Restored SAM parameters are non-finite")
        if not _rng_states_equal(first_pass_end_rng, second_pass_end_rng):
            raise RuntimeError("SAM RNG replay diverged")
        second_gradient_norm = _gradient_l2_norm(model)
        if not _host_flags(torch.isfinite(second_gradient_norm))[0]:
            raise FloatingPointError("Second-pass SAM gradient norm is non-finite")
        gradient_norm = _optimizer_update(model, optimizer, scheduler)

        predictions_finite = (
            None
            if first_predictions_finite is None
            else first_predictions_finite & second_predictions_finite
        )
        diagnostic_tensors = [
            first_loss_sum,
            second_loss_sum,
            gradient_norm,
            first_gradient_norm,
            perturbation_norm,
            second_gradient_norm,
        ]
        if predictions_finite is not None:
            diagnostic_tensors.append(predictions_finite.float())
        diagnostics = _host_floats(*diagnostic_tensors)
        return {
            "loss_sum": diagnostics[0],
            "second_loss_sum": diagnostics[1],
            "loss_count": loss_count,
            "gradient_norm": diagnostics[2],
            "first_pass_gradient_norm": diagnostics[3],
            "perturbation_norm": diagnostics[4],
            "second_pass_gradient_norm": diagnostics[5],
            "predictions_finite": (
                None if predictions_finite is None else bool(diagnostics[6])
            ),
            "backward_passes": 2 * len(effective_batch),
            "rng_replay_exact": True,
            "all_finite": True,
        }
    except BaseException:
        _restore_parameters(parameters, initial_parameters)
        optimizer.zero_grad(set_to_none=True)
        raise


def run_effective_batch_update(
    model: nn.Module,
    effective_batch: list[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    runtime: RuntimeSettings,
    optimizer_variant: str,
    objective: str,
    temperature: float | None,
    sam_rho: float | None,
    *,
    check_predictions_finite: bool = False,
) -> dict[str, float | int | bool | None]:
    if len(effective_batch) != runtime.accumulation_steps:
        raise ValueError(
            f"Effective batch requires exactly {runtime.accumulation_steps} "
            "physical microbatches"
        )
    objective_metadata(objective, temperature)
    sam_metadata(optimizer_variant, sam_rho)
    loss_count = _objective_loss_count(effective_batch, objective)
    if loss_count == 0:
        raise ValueError("Effective batch contains no valid objective unit")
    if optimizer_variant == "adamw":
        return _run_adamw_update(
            model,
            effective_batch,
            optimizer,
            scheduler,
            objective,
            temperature,
            loss_count,
            check_predictions_finite,
        )
    assert sam_rho is not None
    return _run_sam_update(
        model,
        effective_batch,
        optimizer,
        scheduler,
        objective,
        temperature,
        sam_rho,
        loss_count,
        check_predictions_finite,
    )


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    runtime: RuntimeSettings,
    optimizer_variant: str,
    objective: str,
    temperature: float | None,
    sam_rho: float | None,
) -> dict[str, float | int | bool | None]:
    if not isinstance(loader, Sized):
        raise TypeError("Training loader must expose its physical microbatch count")
    if len(loader) % runtime.accumulation_steps:
        raise ValueError("Training epoch would end inside an effective batch")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    loss_count = 0
    optimizer_steps = 0
    gradient_norms: list[float] = []
    first_gradient_norms: list[float] = []
    perturbation_norms: list[float] = []
    second_gradient_norms: list[float] = []
    backward_passes = 0
    effective_batch: list[dict[str, torch.Tensor]] = []

    for cpu_batch in loader:
        effective_batch.append(cpu_batch)
        if len(effective_batch) != runtime.accumulation_steps:
            continue
        update = run_effective_batch_update(
            model,
            effective_batch,
            optimizer,
            scheduler,
            runtime,
            optimizer_variant,
            objective,
            temperature,
            sam_rho,
        )
        loss_sum += float(update["loss_sum"])
        loss_count += int(update["loss_count"])
        optimizer_steps += 1
        backward_passes += int(update["backward_passes"])
        gradient_norms.append(float(update["gradient_norm"]))
        if update["first_pass_gradient_norm"] is not None:
            first_gradient_norms.append(float(update["first_pass_gradient_norm"]))
            perturbation_norms.append(float(update["perturbation_norm"]))
            second_gradient_norms.append(float(update["second_pass_gradient_norm"]))
        effective_batch.clear()

    if effective_batch:
        raise ValueError("Training epoch ended inside an effective batch")
    if loss_count == 0:
        raise ValueError("Training epoch contains no valid objective unit")
    return {
        "optimizer_steps": optimizer_steps,
        "backward_passes": backward_passes,
        "train_objective_loss": loss_sum / loss_count,
        "mean_gradient_norm": float(np.mean(gradient_norms)),
        "maximum_gradient_norm": float(np.max(gradient_norms)),
        "mean_first_pass_sam_gradient_norm": (
            float(np.mean(first_gradient_norms)) if first_gradient_norms else None
        ),
        "mean_sam_perturbation_norm": (
            float(np.mean(perturbation_norms)) if perturbation_norms else None
        ),
        "mean_second_pass_sam_gradient_norm": (
            float(np.mean(second_gradient_norms)) if second_gradient_norms else None
        ),
        "all_finite": True,
        "adamw_lr": optimizer.param_groups[0]["lr"],
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
    objective: str,
    temperature: float | None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    objective_metadata(objective, temperature)
    model.eval()
    total_loss = 0.0
    total_loss_count = 0
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
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictions = _predict(model, batch)
            loss_sum, loss_count = _objective_loss_sum(
                predictions[:valid_count],
                batch["targets"][:valid_count],
                batch["label_mask"][:valid_count],
                objective,
                temperature,
            )
            total_loss += float(loss_sum)
            total_loss_count += int(loss_count)
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
    if total_loss_count == 0:
        raise ValueError("Evaluation split contains no valid objective unit")
    summary["objective"] = objective_metadata(objective, temperature)
    summary["objective_loss"] = total_loss / total_loss_count
    return summary, daily_rows


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    model_name: str,
    architecture: NeuralArchitecture,
    tcn_settings: TCNSettings | None,
    optimizer_variant: str,
    objective: str,
    temperature: float | None,
    sam_rho: float | None,
    seed: int,
    epoch: int,
    validation_score: float,
    feature_store: Path,
    global_context: str | None,
    feature_manifest: dict[str, object],
    git_commit_sha: str,
    context_ablation: ResolvedContextAblation = NO_CONTEXT_ABLATION,
) -> dict[str, object]:
    if getattr(model, "model_name", None) != model_name:
        raise ValueError("Checkpoint model name does not match the model")

    return {
        "model_name": model_name,
        "optimizer_variant": optimizer_variant,
        "objective": objective_metadata(objective, temperature),
        "sam": sam_metadata(optimizer_variant, sam_rho),
        "seed": seed,
        "epoch": epoch,
        "validation_score": validation_score,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "tcn_settings": None if tcn_settings is None else asdict(tcn_settings),
        "architecture_constants": asdict(architecture),
        "parameter_count": expected_trainable_parameter_count(model_name, architecture),
        "resolved_feature_store_path": str(feature_store),
        "feature_manifest_contract_version": feature_manifest["contract_version"],
        "global_context": global_context,
        "context_ablation": context_ablation.metadata(),
        "global_context_source_hashes": feature_manifest["global_context"][
            "source_hashes"
        ],
        "global_context_normalized_store_hashes": feature_manifest["global_context"][
            "normalized_store_hashes"
        ],
        "git_commit_sha": git_commit_sha,
    }
