from __future__ import annotations

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
    ArchitectureConstants,
    CLOUD_RUNTIME_CONTRACT_VERSION,
    COMPILE_STEADY_STATE_PASS_COUNT,
    COMPILE_WARMUP_PASS_COUNT,
    CompileWarmupReport,
    CONTRACT_VERSION,
    GRADIENT_CLIP,
    HUBER_DELTA,
    HardwareInfo,
    MUON_COMPATIBILITY_CONTRACT_VERSION,
    RuntimeProfile,
)
from .metrics import create_metric_table
from .muon import PYTORCH_MUON_REFERENCE


def validate_runtime_profile(profile: RuntimeProfile) -> HardwareInfo:
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
    if total_memory < profile.minimum_vram_bytes:
        raise RuntimeError(
            f"Profile {profile.name} requires at least "
            f"{profile.minimum_vram_bytes} VRAM bytes, found {total_memory}"
        )
    if capability != profile.expected_compute_capability:
        raise RuntimeError(
            f"Profile {profile.name} requires compute capability "
            f"{profile.expected_compute_capability}, found {capability}"
        )
    if (
        profile.required_device_name_fragment is not None
        and profile.required_device_name_fragment not in device_name
    ):
        raise RuntimeError(
            f"Profile {profile.name} requires a device name containing "
            f"{profile.required_device_name_fragment!r}, found {device_name!r}"
        )
    if (
        profile.required_cpu_architecture is not None
        and cpu_architecture != profile.required_cpu_architecture
    ):
        raise RuntimeError(
            f"Profile {profile.name} requires CPU architecture "
            f"{profile.required_cpu_architecture}, found {cpu_architecture}"
        )
    return HardwareInfo(
        profile=profile.name,
        device_name=device_name,
        compute_capability=capability,
        total_vram_bytes=total_memory,
        cpu_architecture=cpu_architecture,
        platform=system_platform.platform(),
        pytorch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
    )


def compile_model(model: nn.Module, profile: RuntimeProfile) -> None:
    if not hasattr(functorch_config, "backward_pass_autocast"):
        raise RuntimeError(
            "Installed PyTorch lacks torch._functorch.config.backward_pass_autocast"
        )
    functorch_config.backward_pass_autocast = "off"
    model.compile(
        backend=profile.compile_backend,
        mode=profile.compile_mode,
        fullgraph=profile.compile_fullgraph,
        dynamic=profile.compile_dynamic,
    )


def _masked_huber_sample_losses(
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


def masked_huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    label_mask: torch.Tensor,
    delta: float = HUBER_DELTA,
) -> torch.Tensor:
    sample_loss, valid_samples = _masked_huber_sample_losses(
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
    sample_loss, valid_samples = _masked_huber_sample_losses(
        predictions, targets, label_mask, HUBER_DELTA
    )
    return sample_loss[valid_samples].sum()


def _to_cuda(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].to("cuda", non_blocking=True)
        for key in (
            "patches",
            "history_patch_mask",
            "instrument_mask",
            "slow_features",
            "state_position",
            "targets",
            "label_mask",
        )
    }


def _predict(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        batch["patches"],
        batch["history_patch_mask"],
        batch["instrument_mask"],
        batch["slow_features"],
        batch["state_position"],
    )


def _timed_training_warmup_pass(
    model: nn.Module, batch: dict[str, torch.Tensor]
) -> float:
    model.zero_grad(set_to_none=True)
    try:
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predictions = _predict(model, batch)
            loss = masked_huber_loss(predictions, batch["targets"], batch["label_mask"])
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
    *,
    training_mode: bool = True,
) -> CompileWarmupReport:
    torch.cuda.reset_peak_memory_stats()
    training_cuda = _to_cuda(training_batch)
    evaluation_cuda = _to_cuda(evaluation_batch)
    model.train(training_mode)
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
) -> tuple[tuple[float, float, float, float, float], float, int, int]:
    torch.cuda.reset_peak_memory_stats()
    cuda_batch = _to_cuda(evaluation_batch)
    model.eval()
    pass_seconds = tuple(
        _timed_evaluation_warmup_pass(model, cuda_batch)
        for _ in range(COMPILE_WARMUP_PASS_COUNT)
    )
    return (
        pass_seconds,
        statistics.median(pass_seconds[-COMPILE_STEADY_STATE_PASS_COUNT:]),
        torch.cuda.max_memory_allocated(),
        torch.cuda.max_memory_reserved(),
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


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    optimizers: dict[str, torch.optim.Optimizer],
    schedulers: dict[str, torch.optim.lr_scheduler.LambdaLR],
    profile: RuntimeProfile,
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
        if len(effective_batch) != profile.accumulation_steps:
            continue
        effective_valid_samples = sum(
            int(batch["label_mask"].any(dim=(1, 2)).sum()) for batch in effective_batch
        )
        denominator = max(effective_valid_samples, 1)
        effective_loss_sum: torch.Tensor | None = None
        for buffered_batch in effective_batch:
            batch = _to_cuda(buffered_batch)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                microbatch_loss_sum = _loss_sum(
                    _predict(model, batch),
                    batch["targets"],
                    batch["label_mask"],
                )
            (microbatch_loss_sum / denominator).backward()
            detached_loss = microbatch_loss_sum.detach()
            effective_loss_sum = (
                detached_loss
                if effective_loss_sum is None
                else effective_loss_sum + detached_loss
            )
        update_succeeded, gradient_norm = _optimizer_update(
            model, optimizers, schedulers
        )
        if effective_loss_sum is None:
            raise RuntimeError("Effective batch contains no microbatches")
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
    summary["masked_huber_loss"] = total_loss / valid_sample_count
    return summary, daily_rows


def checkpoint_payload(
    model: nn.Module,
    model_variant: str,
    optimizer_variant: str,
    muon_backend: str | None,
    runtime_profile: str,
    seed: int,
    epoch: int,
    validation_score: float,
    feature_store: Path,
    git_commit_sha: str,
) -> dict[str, object]:
    return {
        "contract_version": CONTRACT_VERSION,
        "cloud_runtime_contract_version": CLOUD_RUNTIME_CONTRACT_VERSION,
        "muon_compatibility_contract_version": (MUON_COMPATIBILITY_CONTRACT_VERSION),
        "muon_backend": muon_backend,
        "muon_reference": dict(PYTORCH_MUON_REFERENCE),
        "model_variant": model_variant,
        "optimizer_variant": optimizer_variant,
        "runtime_profile": runtime_profile,
        "seed": seed,
        "epoch": epoch,
        "validation_score": validation_score,
        "model_state_dict": model.state_dict(),
        "architecture_constants": asdict(ArchitectureConstants()),
        "resolved_feature_store_path": str(feature_store),
        "git_commit_sha": git_commit_sha,
    }
