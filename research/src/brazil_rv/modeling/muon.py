# Derived from pytorch/pytorch v2.13.0:
# torch/optim/_muon.py
# Git blob: 2e45e07c4a596fb93f435130c344bb634ee0541c
#
# Compatibility-only changes:
# - uses public torch.optim.Optimizer
# - replaces PyTorch-private typing aliases
# - replaces the PyTorch-private scalar helper with a local equivalent
# - omits the PyTorch-private Dynamo decorator
# - renames Muon to PyTorch213Muon
# - shortens documentation; computation is unchanged
#
# Copyright notices and license terms:
# THIRD_PARTY_NOTICES.md
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Iterable, MutableMapping
from typing import Any

import torch

PYTORCH_MUON_UPSTREAM_TAG = "v2.13.0"
PYTORCH_MUON_UPSTREAM_PATH = "torch/optim/_muon.py"
PYTORCH_MUON_UPSTREAM_BLOB_SHA = "2e45e07c4a596fb93f435130c344bb634ee0541c"
PYTORCH_MUON_BACKEND_NAME = "brazil_rv.modeling.muon.PyTorch213Muon"
PYTORCH_MUON_REFERENCE = {
    "upstream_tag": PYTORCH_MUON_UPSTREAM_TAG,
    "upstream_path": PYTORCH_MUON_UPSTREAM_PATH,
    "upstream_blob_sha": PYTORCH_MUON_UPSTREAM_BLOB_SHA,
}

EPS = 1e-7
DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315
DEFAULT_NS_STEPS = 5


def _zeropower_via_newtonschulz(
    grad: torch.Tensor,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> torch.Tensor:
    """Apply PyTorch 2.13's quintic Newton-Schulz orthogonalization."""
    if ns_steps >= 100:
        raise ValueError(
            "Number of steps must be less than 100 for computational efficiency"
        )
    if len(grad.shape) != 2:
        raise ValueError("Input tensor gradient must be a 2D matrix")
    if len(ns_coefficients) != 3:
        raise ValueError("Coefficients must be a tuple of exactly 3 values")
    a, b, c = ns_coefficients
    ortho_grad = grad.bfloat16()
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    ortho_grad.div_(ortho_grad.norm().clamp(min=eps))
    for _ in range(ns_steps):
        gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(
            gram_matrix,
            gram_matrix,
            gram_matrix,
            beta=b,
            alpha=c,
        )
        ortho_grad = torch.addmm(
            ortho_grad,
            gram_update,
            ortho_grad,
            beta=a,
        )
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    return ortho_grad


def _adjust_lr(
    lr: float | torch.Tensor,
    adjust_lr_fn: str | None,
    param_shape: torch.Size,
) -> float | torch.Tensor:
    """Apply PyTorch 2.13's Muon learning-rate shape adjustment."""
    rows, columns = param_shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        adjusted_ratio = math.sqrt(max(1, rows / columns))
    elif adjust_lr_fn == "match_rms_adamw":
        adjusted_ratio = 0.2 * math.sqrt(max(rows, columns))
    else:
        adjusted_ratio = 1.0
    return lr * adjusted_ratio


def _to_scalar(value: float | torch.Tensor) -> float | torch.Tensor:
    if isinstance(value, torch.Tensor) and value.dim() != 0:
        return value.squeeze()
    return value


class PyTorch213Muon(torch.optim.Optimizer):
    """Public-API compatibility implementation of PyTorch v2.13 Muon."""

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float | torch.Tensor = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (
            DEFAULT_A,
            DEFAULT_B,
            DEFAULT_C,
        ),
        eps: float = EPS,
        ns_steps: int = DEFAULT_NS_STEPS,
        adjust_lr_fn: str | None = None,
    ) -> None:
        if isinstance(lr, torch.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"momentum should be >= 0 but is: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"weight decay should be >= 0 but is: {weight_decay}")
        if adjust_lr_fn is not None and adjust_lr_fn not in [
            "original",
            "match_rms_adamw",
        ]:
            raise ValueError(
                f"Adjust learning rate function {adjust_lr_fn} is not supported"
            )

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.ndim != 2:
                    raise ValueError(
                        "Muon only supports 2D parameters whereas we found a "
                        f"parameter with size: {parameter.size()}"
                    )

    def _init_group(
        self,
        group: MutableMapping[str, Any],
        params_with_grad: list[torch.Tensor],
        grads: list[torch.Tensor],
        muon_momentum_bufs: list[torch.Tensor],
    ) -> bool:
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            if torch.is_complex(parameter):
                raise RuntimeError("Muon does not support complex parameters")
            if parameter.grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")

            params_with_grad.append(parameter)
            grads.append(parameter.grad)
            state = self.state[parameter]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(
                    parameter.grad,
                    memory_format=torch.preserve_format,
                )
            muon_momentum_bufs.append(state["momentum_buffer"])
        return False

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one Muon optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: list[torch.Tensor] = []
            grads: list[torch.Tensor] = []
            muon_momentum_bufs: list[torch.Tensor] = []
            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                muon_momentum_bufs,
            )
            muon(
                params_with_grad,
                grads,
                muon_momentum_bufs,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                nesterov=group["nesterov"],
                ns_coefficients=group["ns_coefficients"],
                eps=group["eps"],
                ns_steps=group["ns_steps"],
                adjust_lr_fn=group["adjust_lr_fn"],
                has_complex=has_complex,
            )
        return loss


def _single_tensor_muon(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    muon_momentum_bufs: list[torch.Tensor],
    *,
    lr: float | torch.Tensor,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: str | None,
    has_complex: bool,
) -> None:
    lr = _to_scalar(lr)
    if has_complex:
        raise ValueError("Complex parameters are not supported")

    for index, param in enumerate(params):
        grad = grads[index]
        if grad.ndim != 2:
            raise ValueError("Param gradient must be a 2D matrix")

        buf = muon_momentum_bufs[index]
        buf.lerp_(grad, 1 - momentum)
        update = grad.lerp(buf, momentum) if nesterov else buf
        update = _zeropower_via_newtonschulz(
            update,
            ns_coefficients,
            ns_steps,
            eps,
        )
        adjusted_lr = _adjust_lr(lr, adjust_lr_fn, param.shape)
        param.mul_(1 - lr * weight_decay)
        param.add_(update, alpha=-adjusted_lr)


def muon(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    muon_momentum_bufs: list[torch.Tensor],
    *,
    foreach: bool | None = None,
    lr: float | torch.Tensor,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: str | None,
    has_complex: bool,
) -> None:
    """Functional Muon update matching PyTorch v2.13."""
    if foreach is not None and foreach:
        raise RuntimeError("Foreach is not supported for Muon yet")

    _single_tensor_muon(
        params,
        grads,
        muon_momentum_bufs,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
        ns_coefficients=ns_coefficients,
        ns_steps=ns_steps,
        eps=eps,
        adjust_lr_fn=adjust_lr_fn,
        has_complex=has_complex,
    )
