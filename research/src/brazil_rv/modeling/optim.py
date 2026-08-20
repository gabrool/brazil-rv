from __future__ import annotations

import math

import torch
from torch import nn

from .contract import (
    ADAMW_BETAS,
    ADAMW_EPS,
    ADAMW_LR,
    ADAMW_WEIGHT_DECAY,
    EFFECTIVE_BATCH_SIZE,
    FINAL_LR_FACTOR,
    MAX_EPOCHS,
    WARMUP_FRACTION,
)


def partition_parameters(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    owners: dict[int, tuple[nn.Module, str]] = {}
    for module in model.modules():
        for attribute, parameter in module.named_parameters(recurse=False):
            if id(parameter) in owners:
                raise ValueError("A parameter has multiple owning modules")
            owners[id(parameter)] = (module, attribute)

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        module, attribute = owners[id(parameter)]
        if (
            isinstance(module, (nn.RMSNorm, nn.Embedding))
            or attribute == "bias"
            or (module is model and attribute == "state_token")
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)

    routed = [*decay, *no_decay]
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if {id(parameter) for parameter in routed} != {
        id(parameter) for parameter in trainable
    }:
        raise ValueError("Optimizer parameter routing is incomplete")
    return {"decay": decay, "no_decay": no_decay}


def build_optimizer(
    model: nn.Module,
    weight_decay: float = ADAMW_WEIGHT_DECAY,
) -> tuple[torch.optim.AdamW, dict[str, list[nn.Parameter]]]:
    groups = partition_parameters(model)
    optimizer = torch.optim.AdamW(
        (
            {"params": groups["decay"], "weight_decay": weight_decay},
            {"params": groups["no_decay"], "weight_decay": 0.0},
        ),
        lr=ADAMW_LR,
        betas=ADAMW_BETAS,
        eps=ADAMW_EPS,
        fused=True,
    )
    return optimizer, groups


def learning_rate_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    if step <= warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return FINAL_LR_FACTOR + (1.0 - FINAL_LR_FACTOR) * cosine


def scheduler_step_contract(
    training_sample_count: int,
    maximum_epochs: int = MAX_EPOCHS,
    effective_batch_size: int = EFFECTIVE_BATCH_SIZE,
) -> tuple[int, int]:
    if training_sample_count <= 0:
        raise ValueError("training_sample_count must be positive")
    if maximum_epochs <= 0:
        raise ValueError("maximum_epochs must be positive")
    if effective_batch_size <= 0:
        raise ValueError("effective_batch_size must be positive")
    steps_per_epoch = math.ceil(training_sample_count / effective_batch_size)
    total_steps = steps_per_epoch * maximum_epochs
    return steps_per_epoch, max(1, math.floor(WARMUP_FRACTION * total_steps))


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_sample_count: int,
    maximum_epochs: int = MAX_EPOCHS,
) -> tuple[torch.optim.lr_scheduler.LambdaLR, int, int]:
    steps_per_epoch, warmup_steps = scheduler_step_contract(
        training_sample_count, maximum_epochs
    )
    total_steps = steps_per_epoch * maximum_epochs

    def schedule(last_epoch: int) -> float:
        update_number = min(last_epoch + 1, total_steps)
        return learning_rate_factor(update_number, total_steps, warmup_steps)

    return (
        torch.optim.lr_scheduler.LambdaLR(optimizer, schedule),
        steps_per_epoch,
        warmup_steps,
    )
