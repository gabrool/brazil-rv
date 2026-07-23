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
    MUON_ADJUST_LR_FN,
    MUON_EPS,
    MUON_LR,
    MUON_MOMENTUM,
    MUON_NESTEROV,
    MUON_NS_COEFFICIENTS,
    MUON_NS_STEPS,
    MUON_WEIGHT_DECAY,
    OPTIMIZER_VARIANTS,
    WARMUP_FRACTION,
)
from .layers import MuonLinear
from .muon import PyTorch213Muon

OFFICIAL_MUON_BACKEND = "torch.optim.Muon"
REFERENCE_MUON_BACKEND = "brazil_rv.modeling.muon.PyTorch213Muon"


def partition_parameters(
    model: nn.Module, optimizer_variant: str
) -> dict[str, list[nn.Parameter]]:
    if optimizer_variant not in OPTIMIZER_VARIANTS:
        raise ValueError(f"Unknown optimizer variant: {optimizer_variant}")
    owners: dict[int, tuple[nn.Module, str]] = {}
    for module in model.modules():
        for attribute, parameter in module.named_parameters(recurse=False):
            if id(parameter) in owners:
                raise ValueError("A parameter has multiple owning modules")
            owners[id(parameter)] = (module, attribute)

    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    muon_ids = {
        parameter_id
        for parameter_id, (module, attribute) in owners.items()
        if isinstance(module, MuonLinear) and attribute == "weight"
    }
    muon: list[nn.Parameter] = []
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []

    for parameter in trainable:
        module, attribute = owners[id(parameter)]
        if optimizer_variant == "hybrid" and id(parameter) in muon_ids:
            if parameter.ndim != 2:
                raise ValueError("Every Muon parameter must be two-dimensional")
            muon.append(parameter)
        elif (
            isinstance(module, (nn.RMSNorm, nn.Embedding))
            or attribute == "bias"
            or (module is model and attribute == "state_token")
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)

    groups = {"muon": muon, "decay": decay, "no_decay": no_decay}
    routed = [parameter for group in groups.values() for parameter in group]
    routed_ids = [id(parameter) for parameter in routed]
    trainable_ids = {id(parameter) for parameter in trainable}
    if len(routed_ids) != len(set(routed_ids)):
        raise ValueError("An optimizer parameter was assigned more than once")
    if set(routed_ids) != trainable_ids:
        raise ValueError("Optimizer parameter routing is incomplete")
    if (
        optimizer_variant == "hybrid"
        and {id(parameter) for parameter in muon} != muon_ids
    ):
        raise ValueError("Muon routing must contain only MuonLinear.weight")
    return groups


def build_optimizers(
    model: nn.Module, optimizer_variant: str
) -> tuple[
    dict[str, torch.optim.Optimizer],
    dict[str, list[nn.Parameter]],
    str | None,
]:
    groups = partition_parameters(model, optimizer_variant)
    optimizers: dict[str, torch.optim.Optimizer] = {}
    muon_backend: str | None = None
    if optimizer_variant == "hybrid":
        if hasattr(torch.optim, "Muon"):
            muon_class = torch.optim.Muon
            muon_backend = OFFICIAL_MUON_BACKEND
        else:
            muon_class = PyTorch213Muon
            muon_backend = REFERENCE_MUON_BACKEND
        optimizers["muon"] = muon_class(
            groups["muon"],
            lr=MUON_LR,
            momentum=MUON_MOMENTUM,
            nesterov=MUON_NESTEROV,
            ns_coefficients=MUON_NS_COEFFICIENTS,
            eps=MUON_EPS,
            ns_steps=MUON_NS_STEPS,
            weight_decay=MUON_WEIGHT_DECAY,
            adjust_lr_fn=MUON_ADJUST_LR_FN,
        )
    adamw_parameters = [
        {
            "params": groups["decay"],
            "weight_decay": ADAMW_WEIGHT_DECAY,
        },
        {"params": groups["no_decay"], "weight_decay": 0.0},
    ]
    optimizers["adamw"] = torch.optim.AdamW(
        adamw_parameters,
        lr=ADAMW_LR,
        betas=ADAMW_BETAS,
        eps=ADAMW_EPS,
        fused=True,
    )
    return optimizers, groups, muon_backend


def learning_rate_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    if step <= warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return FINAL_LR_FACTOR + (1.0 - FINAL_LR_FACTOR) * cosine


def build_schedulers(
    optimizers: dict[str, torch.optim.Optimizer],
    training_sample_count: int,
) -> tuple[dict[str, torch.optim.lr_scheduler.LambdaLR], int, int]:
    steps_per_epoch = math.ceil(training_sample_count / EFFECTIVE_BATCH_SIZE)
    total_steps = steps_per_epoch * MAX_EPOCHS
    warmup_steps = max(1, math.floor(WARMUP_FRACTION * total_steps))

    def schedule(last_epoch: int) -> float:
        update_number = min(last_epoch + 1, total_steps)
        return learning_rate_factor(update_number, total_steps, warmup_steps)

    schedulers = {
        name: torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        for name, optimizer in optimizers.items()
    }
    return schedulers, steps_per_epoch, warmup_steps
