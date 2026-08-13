from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from brazil_rv.modeling.contract import (
    ABSOLUTE_PATCH_COUNT,
    ADAMW_BETAS,
    ADAMW_EPS,
    ADAMW_LR,
    ADAMW_WEIGHT_DECAY,
    CONTEXT_COUNT,
    EQUITY_COUNT,
    EXPECTED_SPLIT_SAMPLE_COUNTS,
    FINAL_LR_FACTOR,
    GH200_RUNTIME,
    HORIZON_COUNT,
    LOCAL_CONTEXT_COUNT,
    PATCH_INPUT_WIDTH,
    PEER_STATE_WIDTH,
    SLOW_FEATURE_COUNT,
    TCNSettings,
    WARMUP_FRACTION,
    context_routing_metadata,
    resolve_tcn_architecture,
)
from brazil_rv.modeling.engine import (
    _predict,
    _rng_state,
    _rng_states_equal,
    _to_cuda,
    compile_model,
    objective_metadata,
    objective_loss,
    run_effective_batch_update,
    sam_metadata,
    validate_runtime,
)
from brazil_rv.modeling.model import build_neural_model
from brazil_rv.modeling.optim import build_optimizer, build_scheduler


PREFLIGHT_VERSION = "ROUTING_IDENTITY_PREFLIGHT_V1"
PREFLIGHT_STEPS = 3
PREFLIGHT_SEED = 29
PREFLIGHT_TOLERANCES = {
    "base_initial_max_absolute": 0.0,
    "prediction_atol": 1e-6,
    "prediction_rtol": 1e-6,
    "loss_atol": 1e-7,
    "loss_rtol": 1e-7,
    "gradient_atol": 1e-7,
    "gradient_rtol": 1e-6,
    "perturbation_atol": 1e-7,
    "perturbation_rtol": 1e-6,
    "update_atol": 1e-7,
    "update_rtol": 1e-6,
}
PREFLIGHT_EXECUTIONS = ("eager", "compiled")
_STEP_QUANTITIES = (
    "predictions",
    "loss",
    "prediction_rng_state",
    "first_pass_base_gradients",
    "sam_base_perturbations",
    "second_pass_base_gradients",
    "base_parameter_updates",
    "update_rng_state",
    "sam_scalar_diagnostics",
)
_TENSOR_QUANTITIES = {
    "initial_base_state",
    "predictions",
    "loss",
    "first_pass_base_gradients",
    "sam_base_perturbations",
    "second_pass_base_gradients",
    "base_parameter_updates",
}
_ZERO_GRADIENT_EXCEPTIONS = {
    "fusion_norm.bias": (
        "The LayerNorm bias immediately before the shared prediction head adds the "
        "same hidden shift to every equity, producing only a constant per-horizon "
        "prediction shift; cross-sectional soft-Spearman is shift-invariant, and "
        "this bias is excluded from weight decay."
    ),
    "prediction_head.bias": (
        "Adds a constant per-horizon shift across equities; cross-sectional "
        "soft-Spearman is shift-invariant, and this bias is excluded from weight decay."
    ),
}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _mapping_hash(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in values.items():
        digest.update(name.encode())
        digest.update(_tensor_hash(value).encode())
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_normalize(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _parameter_identity(model: nn.Module, *, routing: bool) -> list[dict[str, Any]]:
    prefix = "routing."
    return [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "numel": parameter.numel(),
            "requires_grad": parameter.requires_grad,
        }
        for name, parameter in model.named_parameters()
        if name.startswith(prefix) is routing
    ]


def _parameter_snapshot(model: nn.Module, *, routing: bool) -> dict[str, torch.Tensor]:
    prefix = "routing."
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name.startswith(prefix) is routing
    }


def _gradient_snapshot(model: nn.Module, *, routing: bool) -> dict[str, torch.Tensor]:
    prefix = "routing."
    return {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name.startswith(prefix) is routing and parameter.grad is not None
    }


def _difference_snapshot(
    current: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    if current.keys() != reference.keys():
        raise RuntimeError("Parameter identities changed during identity preflight")
    return {name: current[name] - reference[name] for name in current}


def compare_tensor_mappings(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if left.keys() != right.keys():
        return {
            "passed": False,
            "reason": "tensor_identity_mismatch",
            "left_names": list(left),
            "right_names": list(right),
        }
    maximum_absolute = 0.0
    difference_squared = 0.0
    reference_squared = 0.0
    passed = True
    for name in left:
        lhs = left[name].float()
        rhs = right[name].float()
        if lhs.shape != rhs.shape:
            return {
                "passed": False,
                "reason": f"tensor_shape_mismatch:{name}",
                "left_shape": list(lhs.shape),
                "right_shape": list(rhs.shape),
            }
        difference = lhs - rhs
        maximum_absolute = max(maximum_absolute, float(difference.abs().max().item()))
        difference_squared += float(difference.double().square().sum().item())
        reference_squared += float(rhs.double().square().sum().item())
        passed = passed and bool(torch.allclose(lhs, rhs, atol=atol, rtol=rtol))
    relative_l2 = (
        difference_squared**0.5 / reference_squared**0.5
        if reference_squared > 0.0
        else difference_squared**0.5
    )
    return {
        "passed": passed,
        "maximum_absolute_error": maximum_absolute,
        "relative_l2_error": relative_l2,
        "left_sha256": _mapping_hash(left),
        "right_sha256": _mapping_hash(right),
        "tensor_count": len(left),
    }


def _compare_tensors(
    left: torch.Tensor, right: torch.Tensor, *, atol: float, rtol: float
) -> dict[str, Any]:
    return compare_tensor_mappings(
        {"value": left.detach().cpu()},
        {"value": right.detach().cpu()},
        atol=atol,
        rtol=rtol,
    )


def _rng_hash(state: tuple[torch.Tensor, torch.Tensor | None]) -> str:
    values = {"cpu": state[0]}
    if state[1] is not None:
        values["cuda"] = state[1]
    return _mapping_hash(values)


class _SamCapture:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.base_reference: dict[str, torch.Tensor] = {}
        self.routing_reference: dict[str, torch.Tensor] = {}
        self.events: dict[str, dict[str, torch.Tensor]] = {}
        self.base_coverage = {
            name: {
                "first_pass_gradient": False,
                "second_pass_gradient": False,
                "parameter_update": False,
            }
            for name in _parameter_snapshot(model, routing=False)
        }
        self.routing_gradient_names: set[str] = set()
        self.routing_changed_names: set[str] = set()

    def begin_step(self) -> None:
        self.base_reference = _parameter_snapshot(self.model, routing=False)
        self.routing_reference = _parameter_snapshot(self.model, routing=True)
        self.events = {}

    def __call__(self, event: str, model: nn.Module) -> None:
        if model is not self.model:
            raise RuntimeError("SAM observer received an unexpected model")
        if event in ("first_gradients", "second_gradients"):
            self.events[event] = _gradient_snapshot(model, routing=False)
            coverage_key = (
                "first_pass_gradient"
                if event == "first_gradients"
                else "second_pass_gradient"
            )
            for name, parameter in model.named_parameters():
                if not name.startswith("routing.") and parameter.grad is not None:
                    exercised = bool(torch.count_nonzero(parameter.grad).item())
                    self.base_coverage[name][coverage_key] |= exercised
            self.routing_gradient_names.update(_gradient_snapshot(model, routing=True))
            return
        base = _parameter_snapshot(model, routing=False)
        routing = _parameter_snapshot(model, routing=True)
        differences = _difference_snapshot(base, self.base_reference)
        self.events[event] = differences
        if event == "updated_parameters":
            for name, difference in differences.items():
                self.base_coverage[name]["parameter_update"] |= bool(
                    torch.count_nonzero(difference).item()
                )
        self.routing_changed_names.update(
            name
            for name, value in routing.items()
            if not torch.equal(value, self.routing_reference[name])
        )

    def coverage_report(self) -> dict[str, Any]:
        exception_names = set(_ZERO_GRADIENT_EXCEPTIONS)
        unexercised = [
            name
            for name, coverage in self.base_coverage.items()
            if name not in exception_names and not all(coverage.values())
        ]
        return {
            "per_parameter": self.base_coverage,
            "legitimate_zero_gradient_exceptions": [
                {"name": name, "justification": justification}
                for name, justification in sorted(_ZERO_GRADIENT_EXCEPTIONS.items())
            ],
            "exception_policy": (
                "No base parameter is expected to have zero gradients or updates."
                if not _ZERO_GRADIENT_EXCEPTIONS
                else "Only the explicitly justified parameters may remain unexercised."
            ),
            "unexercised_expected_parameters": unexercised,
            "passed": not unexercised,
        }


def _synthetic_batch(equity_count: int = EQUITY_COUNT) -> dict[str, torch.Tensor]:
    batch_size = 1
    active_equities = min(32, equity_count)
    instrument_count = equity_count + CONTEXT_COUNT
    state_position = 15
    patches = torch.zeros(
        batch_size,
        instrument_count,
        ABSOLUTE_PATCH_COUNT,
        PATCH_INPUT_WIDTH,
    )
    history = torch.zeros(
        batch_size, instrument_count, ABSOLUTE_PATCH_COUNT, dtype=torch.bool
    )
    history[:, :active_equities, 12:state_position] = True
    wdo_di_start = equity_count + 1
    history[:, wdo_di_start : equity_count + LOCAL_CONTEXT_COUNT, :state_position] = (
        True
    )
    global_start = equity_count + LOCAL_CONTEXT_COUNT
    zt_zn = slice(global_start + 2, global_start + 4)
    history[:, zt_zn] = True
    rows = history.nonzero(as_tuple=False)
    channels = torch.arange(PATCH_INPUT_WIDTH)
    patches[history] = (
        (rows[:, 1, None] * 17 + rows[:, 2, None] * 13 + channels[None, :] * 7)
        .remainder(101)
        .add(1)
        .float()
        .div(1000.0)
    )
    instrument = torch.zeros(batch_size, instrument_count, dtype=torch.bool)
    instrument[:, :active_equities] = True
    instrument[:, wdo_di_start : equity_count + LOCAL_CONTEXT_COUNT] = True
    instrument[:, zt_zn] = True
    slow = torch.zeros(batch_size, instrument_count, SLOW_FEATURE_COUNT)
    slow_rows = instrument.nonzero(as_tuple=False)
    slow_channels = torch.arange(SLOW_FEATURE_COUNT)
    slow[instrument] = (
        (slow_rows[:, 1, None] * 11 + slow_channels[None, :] * 5)
        .remainder(67)
        .add(1)
        .float()
        .div(1000.0)
    )
    peer = torch.zeros(batch_size, equity_count, PEER_STATE_WIDTH)
    equity_rows = torch.arange(active_equities)[:, None]
    peer_channels = torch.arange(PEER_STATE_WIDTH)[None, :]
    peer[:, :active_equities] = (
        (equity_rows * 3 + peer_channels * 7).remainder(43).add(1).float().div(1000.0)
    )
    targets = torch.zeros(batch_size, equity_count, HORIZON_COUNT)
    ranks = torch.linspace(-0.95, 0.95, active_equities)
    targets[:, :active_equities] = ranks[None, :, None]
    label_mask = torch.zeros(batch_size, equity_count, HORIZON_COUNT, dtype=torch.bool)
    label_mask[:, :active_equities] = True
    return {
        "patches": patches,
        "history_patch_mask": history,
        "instrument_mask": instrument,
        "slow_features": slow,
        "state_position": torch.full((batch_size,), state_position),
        "peer_state": peer,
        "targets": targets,
        "label_mask": label_mask,
    }


def _fixture_metadata(batch: dict[str, torch.Tensor]) -> dict[str, Any]:
    history = batch["history_patch_mask"]
    patches = batch["patches"]
    equity_count = patches.shape[1] - CONTEXT_COUNT
    global_start = equity_count + LOCAL_CONTEXT_COUNT
    source_slices = {
        "equity": slice(0, equity_count),
        "wdo": slice(equity_count + 1, equity_count + 2),
        "di": slice(equity_count + 2, equity_count + LOCAL_CONTEXT_COUNT),
        "zt": slice(global_start + 2, global_start + 3),
        "zn": slice(global_start + 3, global_start + 4),
    }
    return {
        "physical_microbatch_size": 1,
        "packed_equity_count": equity_count,
        "packed_instrument_count": patches.shape[1],
        "accumulation_steps": GH200_RUNTIME.accumulation_steps,
        "active_equities": 32,
        "state_position": 15,
        "patch_tensor_sha256": _tensor_hash(patches),
        "history_mask_sha256": _tensor_hash(history),
        "valid_patch_positions": int(history.sum().item()),
        "valid_nonzero_values": int(torch.count_nonzero(patches[history]).item()),
        "unavailable_nonzero_values": int(
            torch.count_nonzero(patches[~history]).item()
        ),
        "finite": bool(torch.isfinite(patches).all()),
        "source_valid_patch_positions": {
            name: int(history[:, instrument_slice].sum().item())
            for name, instrument_slice in source_slices.items()
        },
    }


def _settings(experiment: str) -> TCNSettings:
    return TCNSettings(
        "context_pooled",
        64,
        "full",
        "swiglu",
        "late_only",
        "late_only",
        experiment,
    )


def _seeded_models(
    device: str = "cuda", equity_count: int = EQUITY_COUNT
) -> tuple[nn.Module, nn.Module]:
    with torch.random.fork_rng():
        torch.manual_seed(PREFLIGHT_SEED)
        legacy = build_neural_model(
            "tcn",
            resolve_tcn_architecture(_settings("legacy")),
            "selected",
            equity_count,
        ).to(device)
        torch.manual_seed(PREFLIGHT_SEED)
        scaffold = build_neural_model(
            "tcn",
            resolve_tcn_architecture(_settings("factorial_v1")),
            "selected",
            equity_count,
        ).to(device)
    return legacy, scaffold


def build_routing_preflight_identity(
    git_commit: str,
    run_profile: dict[str, Any] | None = None,
    train_sample_count: int | None = None,
) -> dict[str, Any]:
    if len(git_commit) != 40 or any(
        character not in "0123456789abcdef" for character in git_commit
    ):
        raise ValueError("Preflight identity requires a full lowercase Git commit SHA")
    profile = run_profile or {
        "name": "production",
        "equity_count": EQUITY_COUNT,
        "maximum_epochs": 20,
    }
    equity_count = int(profile.get("equity_count", -1))
    maximum_epochs = int(profile.get("maximum_epochs", -1))
    if equity_count <= 0 or maximum_epochs <= 0:
        raise ValueError("Preflight run profile has invalid packed dimensions")
    if train_sample_count is None:
        train_sample_count = EXPECTED_SPLIT_SAMPLE_COUNTS["train"]
    if train_sample_count <= 0:
        raise ValueError("Preflight training sample count must be positive")
    legacy_architecture = resolve_tcn_architecture(_settings("legacy"))
    factorial_architecture = resolve_tcn_architecture(_settings("factorial_v1"))
    legacy, scaffold = _seeded_models("cpu", equity_count)
    legacy_base = _parameter_identity(legacy, routing=False)
    scaffold_base = _parameter_identity(scaffold, routing=False)
    if legacy_base != scaffold_base or _parameter_identity(legacy, routing=True):
        raise RuntimeError("Legacy and scaffold base parameter identities diverged")
    required_comparisons = [
        {"step": 0, "quantity": "initial_base_state"},
        *[
            {"step": step, "quantity": quantity}
            for step in range(1, PREFLIGHT_STEPS + 1)
            for quantity in _STEP_QUANTITIES
        ],
        {"step": PREFLIGHT_STEPS, "quantity": "base_parameter_coverage"},
        {"step": PREFLIGHT_STEPS, "quantity": "inactive_routing_parameters"},
    ]
    identity = {
        "git": {"commit": git_commit, "clean_worktree": True},
        "run_profile": profile,
        "packed_shape": {
            "equity_count": equity_count,
            "instrument_count": equity_count + CONTEXT_COUNT,
        },
        "legacy_architecture": {
            "settings": asdict(_settings("legacy")),
            "resolved": asdict(legacy_architecture),
        },
        "factorial_architecture": {
            "settings": asdict(_settings("factorial_v1")),
            "resolved": asdict(factorial_architecture),
            "routing_schema": context_routing_metadata(factorial_architecture),
        },
        "parameter_identities": {
            "base": legacy_base,
            "routing": _parameter_identity(scaffold, routing=True),
        },
        "objective": objective_metadata("soft_spearman", 0.50),
        "optimizer": {
            "name": "adamw",
            "lr": ADAMW_LR,
            "betas": list(ADAMW_BETAS),
            "eps": ADAMW_EPS,
            "decayed_weight_decay": ADAMW_WEIGHT_DECAY,
            "zero_decay_weight_decay": 0.0,
            "fused": True,
            "scheduler": "linear_warmup_cosine_decay",
            "warmup_fraction": WARMUP_FRACTION,
            "final_lr_factor": FINAL_LR_FACTOR,
            "train_sample_count": train_sample_count,
            "maximum_epochs": maximum_epochs,
        },
        "sam": sam_metadata("sam_adamw", 0.125),
        "runtime": asdict(GH200_RUNTIME),
        "execution_contract": {
            "steps": PREFLIGHT_STEPS,
            "seed": PREFLIGHT_SEED,
            "executions": list(PREFLIGHT_EXECUTIONS),
            "required_comparisons": required_comparisons,
            "inactive_routing_gradients_and_updates_required_absent": True,
        },
        "synthetic_fixture": _fixture_metadata(_synthetic_batch(equity_count)),
    }
    return _json_normalize(identity)


def _record_comparison(
    comparisons: list[dict[str, Any]],
    *,
    execution: str,
    step: int,
    quantity: str,
    comparison: dict[str, Any],
) -> None:
    comparisons.append(
        {
            "execution": execution,
            "step": step,
            "quantity": quantity,
            **comparison,
        }
    )


def _run_execution(
    execution: str,
    batch: dict[str, torch.Tensor],
    equity_count: int,
    train_sample_count: int,
    maximum_epochs: int,
) -> dict[str, Any]:
    legacy, scaffold = _seeded_models(equity_count=equity_count)
    compile_setups: dict[str, Any] = {}
    if execution == "compiled":
        compile_setups = {
            "legacy": asdict(compile_model(legacy, GH200_RUNTIME)),
            "scaffold": asdict(compile_model(scaffold, GH200_RUNTIME)),
        }
    elif execution != "eager":
        raise ValueError(f"Unknown preflight execution mode: {execution}")

    comparisons: list[dict[str, Any]] = []
    _record_comparison(
        comparisons,
        execution=execution,
        step=0,
        quantity="initial_base_state",
        comparison=compare_tensor_mappings(
            _parameter_snapshot(legacy, routing=False),
            _parameter_snapshot(scaffold, routing=False),
            atol=PREFLIGHT_TOLERANCES["base_initial_max_absolute"],
            rtol=0.0,
        ),
    )
    legacy_optimizer, _ = build_optimizer(legacy)
    scaffold_optimizer, _ = build_optimizer(scaffold)
    legacy_scheduler, _, _ = build_scheduler(
        legacy_optimizer, train_sample_count, maximum_epochs
    )
    scaffold_scheduler, _, _ = build_scheduler(
        scaffold_optimizer, train_sample_count, maximum_epochs
    )
    cuda_batch = _to_cuda(batch)
    effective_batch = [batch] * GH200_RUNTIME.accumulation_steps
    legacy_capture = _SamCapture(legacy)
    scaffold_capture = _SamCapture(scaffold)
    legacy.train()
    scaffold.train()

    for step in range(1, PREFLIGHT_STEPS + 1):
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            legacy_predictions = _predict(legacy, cuda_batch)
        legacy_loss = objective_loss(
            legacy_predictions,
            cuda_batch["targets"],
            cuda_batch["label_mask"],
            "soft_spearman",
            0.50,
        )
        legacy_prediction_rng = _rng_state(legacy)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            scaffold_predictions = _predict(scaffold, cuda_batch)
        scaffold_loss = objective_loss(
            scaffold_predictions,
            cuda_batch["targets"],
            cuda_batch["label_mask"],
            "soft_spearman",
            0.50,
        )
        scaffold_prediction_rng = _rng_state(scaffold)
        _record_comparison(
            comparisons,
            execution=execution,
            step=step,
            quantity="predictions",
            comparison=_compare_tensors(
                legacy_predictions,
                scaffold_predictions,
                atol=PREFLIGHT_TOLERANCES["prediction_atol"],
                rtol=PREFLIGHT_TOLERANCES["prediction_rtol"],
            ),
        )
        _record_comparison(
            comparisons,
            execution=execution,
            step=step,
            quantity="loss",
            comparison=_compare_tensors(
                legacy_loss,
                scaffold_loss,
                atol=PREFLIGHT_TOLERANCES["loss_atol"],
                rtol=PREFLIGHT_TOLERANCES["loss_rtol"],
            ),
        )
        _record_comparison(
            comparisons,
            execution=execution,
            step=step,
            quantity="prediction_rng_state",
            comparison={
                "passed": _rng_states_equal(
                    legacy_prediction_rng, scaffold_prediction_rng
                ),
                "left_sha256": _rng_hash(legacy_prediction_rng),
                "right_sha256": _rng_hash(scaffold_prediction_rng),
            },
        )

        update_cpu_rng = torch.get_rng_state().clone()
        update_cuda_rng = torch.cuda.get_rng_state().clone()
        legacy_capture.begin_step()
        legacy_diagnostics = run_effective_batch_update(
            legacy,
            effective_batch,
            legacy_optimizer,
            legacy_scheduler,
            GH200_RUNTIME,
            "sam_adamw",
            "soft_spearman",
            0.50,
            0.125,
            check_predictions_finite=True,
            sam_observer=legacy_capture,
        )
        legacy_update_rng = _rng_state(legacy)
        torch.set_rng_state(update_cpu_rng)
        torch.cuda.set_rng_state(update_cuda_rng)
        scaffold_capture.begin_step()
        scaffold_diagnostics = run_effective_batch_update(
            scaffold,
            effective_batch,
            scaffold_optimizer,
            scaffold_scheduler,
            GH200_RUNTIME,
            "sam_adamw",
            "soft_spearman",
            0.50,
            0.125,
            check_predictions_finite=True,
            sam_observer=scaffold_capture,
        )
        scaffold_update_rng = _rng_state(scaffold)
        for event, quantity, tolerance_prefix in (
            ("first_gradients", "first_pass_base_gradients", "gradient"),
            ("perturbed_parameters", "sam_base_perturbations", "perturbation"),
            ("second_gradients", "second_pass_base_gradients", "gradient"),
            ("updated_parameters", "base_parameter_updates", "update"),
        ):
            _record_comparison(
                comparisons,
                execution=execution,
                step=step,
                quantity=quantity,
                comparison=compare_tensor_mappings(
                    legacy_capture.events[event],
                    scaffold_capture.events[event],
                    atol=PREFLIGHT_TOLERANCES[f"{tolerance_prefix}_atol"],
                    rtol=PREFLIGHT_TOLERANCES[f"{tolerance_prefix}_rtol"],
                ),
            )
        _record_comparison(
            comparisons,
            execution=execution,
            step=step,
            quantity="update_rng_state",
            comparison={
                "passed": _rng_states_equal(legacy_update_rng, scaffold_update_rng),
                "left_sha256": _rng_hash(legacy_update_rng),
                "right_sha256": _rng_hash(scaffold_update_rng),
            },
        )
        _record_comparison(
            comparisons,
            execution=execution,
            step=step,
            quantity="sam_scalar_diagnostics",
            comparison={
                "passed": legacy_diagnostics == scaffold_diagnostics,
                "legacy": legacy_diagnostics,
                "scaffold": scaffold_diagnostics,
            },
        )

    legacy_coverage = legacy_capture.coverage_report()
    scaffold_coverage = scaffold_capture.coverage_report()
    expected_base_names = [
        row["name"] for row in _parameter_identity(legacy, routing=False)
    ]
    comparisons.append(
        {
            "execution": execution,
            "step": PREFLIGHT_STEPS,
            "quantity": "base_parameter_coverage",
            "passed": (
                legacy_coverage["passed"] is True
                and scaffold_coverage["passed"] is True
                and legacy_coverage == scaffold_coverage
            ),
            "expected_parameter_names": expected_base_names,
            "legacy": legacy_coverage,
            "scaffold": scaffold_coverage,
        }
    )
    expected_routing_names = [
        row["name"] for row in _parameter_identity(scaffold, routing=True)
    ]
    gradient_names = sorted(scaffold_capture.routing_gradient_names)
    changed_names = sorted(scaffold_capture.routing_changed_names)
    comparisons.append(
        {
            "execution": execution,
            "step": PREFLIGHT_STEPS,
            "quantity": "inactive_routing_parameters",
            "passed": not gradient_names and not changed_names,
            "expected_parameter_names": expected_routing_names,
            "gradient_parameter_names": gradient_names,
            "changed_parameter_names": changed_names,
            "sha256": _mapping_hash(_parameter_snapshot(scaffold, routing=True)),
        }
    )
    return {
        "execution": execution,
        "compile_setups": compile_setups,
        "comparisons": comparisons,
        "passed": all(bool(item["passed"]) for item in comparisons),
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Routing identity preflight artifact is invalid: {message}")


def _exact_keys(value: Any, expected: set[str], context: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{context} must be an object")
    _require(set(value) == expected, f"{context} fields do not match the contract")
    return value


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _validate_coverage(comparison: dict[str, Any], expected_names: list[str]) -> None:
    _exact_keys(
        comparison,
        {
            "execution",
            "step",
            "quantity",
            "passed",
            "expected_parameter_names",
            "legacy",
            "scaffold",
        },
        "base parameter coverage comparison",
    )
    _require(
        comparison["expected_parameter_names"] == expected_names,
        "base parameter coverage identities drifted",
    )
    expected_exceptions = [
        {"name": name, "justification": justification}
        for name, justification in sorted(_ZERO_GRADIENT_EXCEPTIONS.items())
    ]
    for model_name in ("legacy", "scaffold"):
        report = _exact_keys(
            comparison[model_name],
            {
                "per_parameter",
                "legitimate_zero_gradient_exceptions",
                "exception_policy",
                "unexercised_expected_parameters",
                "passed",
            },
            f"{model_name} coverage report",
        )
        _require(
            report["legitimate_zero_gradient_exceptions"] == expected_exceptions,
            f"{model_name} zero-gradient exceptions drifted",
        )
        per_parameter = report["per_parameter"]
        _require(
            isinstance(per_parameter, dict) and list(per_parameter) == expected_names,
            f"{model_name} per-parameter coverage identities drifted",
        )
        unexercised: list[str] = []
        for name, coverage in per_parameter.items():
            _exact_keys(
                coverage,
                {"first_pass_gradient", "second_pass_gradient", "parameter_update"},
                f"coverage for {name}",
            )
            _require(
                all(type(value) is bool for value in coverage.values()),
                f"coverage for {name} must contain booleans",
            )
            if name not in _ZERO_GRADIENT_EXCEPTIONS and not all(coverage.values()):
                unexercised.append(name)
        _require(
            report["unexercised_expected_parameters"] == unexercised,
            f"{model_name} unexercised parameter summary is inconsistent",
        )
        _require(
            report["passed"] is (not unexercised),
            f"{model_name} stored coverage status is inconsistent",
        )
        _require(
            not unexercised,
            f"{model_name} contains unexercised expected base parameters: "
            f"{unexercised}",
        )


def _validate_comparison(
    comparison: Any, expected_base_names: list[str], expected_routing_names: list[str]
) -> None:
    _require(isinstance(comparison, dict), "comparison must be an object")
    _require(comparison.get("passed") is True, "every comparison must pass")
    quantity = comparison.get("quantity")
    if quantity in _TENSOR_QUANTITIES:
        _exact_keys(
            comparison,
            {
                "execution",
                "step",
                "quantity",
                "passed",
                "maximum_absolute_error",
                "relative_l2_error",
                "left_sha256",
                "right_sha256",
                "tensor_count",
            },
            f"{quantity} comparison",
        )
        _require(
            _finite_number(comparison["maximum_absolute_error"])
            and float(comparison["maximum_absolute_error"]) >= 0.0
            and _finite_number(comparison["relative_l2_error"])
            and float(comparison["relative_l2_error"]) >= 0.0,
            f"{quantity} errors must be finite and nonnegative",
        )
        _require(
            _valid_sha256(comparison["left_sha256"])
            and _valid_sha256(comparison["right_sha256"])
            and type(comparison["tensor_count"]) is int
            and comparison["tensor_count"] > 0,
            f"{quantity} hashes or tensor count are malformed",
        )
        return
    if quantity in ("prediction_rng_state", "update_rng_state"):
        _exact_keys(
            comparison,
            {"execution", "step", "quantity", "passed", "left_sha256", "right_sha256"},
            f"{quantity} comparison",
        )
        _require(
            comparison["left_sha256"] == comparison["right_sha256"]
            and _valid_sha256(comparison["left_sha256"]),
            f"{quantity} states differ or are malformed",
        )
        return
    if quantity == "sam_scalar_diagnostics":
        _exact_keys(
            comparison,
            {"execution", "step", "quantity", "passed", "legacy", "scaffold"},
            "SAM diagnostics comparison",
        )
        _require(
            isinstance(comparison["legacy"], dict)
            and comparison["legacy"] == comparison["scaffold"],
            "SAM diagnostics differ or are malformed",
        )
        return
    if quantity == "base_parameter_coverage":
        _validate_coverage(comparison, expected_base_names)
        return
    if quantity == "inactive_routing_parameters":
        _exact_keys(
            comparison,
            {
                "execution",
                "step",
                "quantity",
                "passed",
                "expected_parameter_names",
                "gradient_parameter_names",
                "changed_parameter_names",
                "sha256",
            },
            "inactive routing comparison",
        )
        _require(
            comparison["expected_parameter_names"] == expected_routing_names
            and comparison["gradient_parameter_names"] == []
            and comparison["changed_parameter_names"] == []
            and _valid_sha256(comparison["sha256"]),
            "inactive routing gradients, updates, or identities are invalid",
        )
        return
    raise ValueError(
        f"Routing identity preflight artifact is invalid: unknown comparison {quantity!r}"
    )


def validate_routing_identity_preflight(
    payload: dict[str, Any],
    expected_identity: dict[str, Any],
    current_environment: dict[str, Any],
) -> dict[str, Any]:
    _exact_keys(
        payload,
        {
            "version",
            "status",
            "steps",
            "seed",
            "identity",
            "identity_sha256",
            "objective",
            "optimizer",
            "sam",
            "runtime",
            "tolerances",
            "synthetic_fixture",
            "environment",
            "executions",
            "maximum_observed_errors",
        },
        "top-level artifact",
    )
    _require(payload["version"] == PREFLIGHT_VERSION, "version mismatch")
    _require(payload["status"] == "passed", "status is not passed")
    _require(
        type(payload["steps"]) is int and payload["steps"] == 3,
        "step count must be exactly three",
    )
    _require(payload["seed"] == PREFLIGHT_SEED, "seed mismatch")
    _require(payload["identity"] == expected_identity, "expected identity mismatch")
    _require(
        payload["identity_sha256"] == _json_hash(expected_identity),
        "identity hash mismatch",
    )
    for key in ("objective", "optimizer", "sam", "runtime", "synthetic_fixture"):
        _require(payload[key] == expected_identity[key], f"{key} drifted")
    _require(payload["tolerances"] == PREFLIGHT_TOLERANCES, "tolerances drifted")
    environment = _exact_keys(
        payload["environment"],
        {
            "device_name",
            "compute_capability",
            "total_vram_bytes",
            "cpu_architecture",
            "platform",
            "pytorch_version",
            "cuda_version",
            "cudnn_version",
        },
        "environment",
    )
    _require(
        isinstance(environment["device_name"], str)
        and environment["device_name"]
        and environment["compute_capability"]
        == expected_identity["runtime"]["expected_compute_capability"]
        and type(environment["total_vram_bytes"]) is int
        and environment["total_vram_bytes"]
        >= expected_identity["runtime"]["minimum_vram_bytes"]
        and environment["cpu_architecture"]
        == expected_identity["runtime"]["required_cpu_architecture"]
        and all(
            isinstance(environment[key], str) and environment[key]
            for key in ("platform", "pytorch_version", "cuda_version")
        )
        and type(environment["cudnn_version"]) is int,
        "PyTorch/CUDA/GH200 environment is structurally invalid",
    )
    _require(
        payload["environment"] == _json_normalize(current_environment),
        "PyTorch/CUDA/GH200 environment drifted",
    )
    executions = payload["executions"]
    _require(
        isinstance(executions, list) and len(executions) == 2,
        "exactly two executions are required",
    )
    _require(
        [row.get("execution") for row in executions] == list(PREFLIGHT_EXECUTIONS),
        "eager and compiled executions are required in order",
    )
    expected_pairs = [
        (row["step"], row["quantity"])
        for row in expected_identity["execution_contract"]["required_comparisons"]
    ]
    expected_base_names = [
        row["name"] for row in expected_identity["parameter_identities"]["base"]
    ]
    expected_routing_names = [
        row["name"] for row in expected_identity["parameter_identities"]["routing"]
    ]
    all_comparisons: list[dict[str, Any]] = []
    for execution in executions:
        _exact_keys(
            execution,
            {"execution", "compile_setups", "comparisons", "passed"},
            "execution",
        )
        _require(execution["passed"] is True, "execution did not pass")
        mode = execution["execution"]
        if mode == "eager":
            _require(
                execution["compile_setups"] == {},
                "eager execution has compile metadata",
            )
        else:
            setups = _exact_keys(
                execution["compile_setups"], {"legacy", "scaffold"}, "compiled setups"
            )
            _require(
                setups["legacy"] == setups["scaffold"],
                "compiled setup differs between models",
            )
            for setup in setups.values():
                _exact_keys(
                    setup,
                    {
                        "api",
                        "backend",
                        "mode",
                        "fullgraph",
                        "dynamic",
                        "backward_pass_autocast_control_available",
                        "backward_pass_autocast_policy",
                    },
                    "compiled setup",
                )
                _require(
                    setup["api"] == "nn.Module.compile"
                    and setup["backend"]
                    == expected_identity["runtime"]["compile_backend"]
                    and setup["mode"] == expected_identity["runtime"]["compile_mode"]
                    and setup["fullgraph"]
                    is expected_identity["runtime"]["compile_fullgraph"]
                    and setup["dynamic"]
                    is expected_identity["runtime"]["compile_dynamic"],
                    "compiled runtime settings drifted",
                )
        comparisons = execution["comparisons"]
        _require(isinstance(comparisons, list), "comparisons must be a list")
        _require(
            [(row.get("step"), row.get("quantity")) for row in comparisons]
            == expected_pairs,
            "required comparisons are missing, duplicated, or reordered",
        )
        for comparison in comparisons:
            _require(
                comparison.get("execution") == mode,
                "comparison execution identity mismatch",
            )
            _validate_comparison(
                comparison, expected_base_names, expected_routing_names
            )
        all_comparisons.extend(comparisons)
    observed = {
        "absolute": max(
            float(row.get("maximum_absolute_error", 0.0)) for row in all_comparisons
        ),
        "relative_l2": max(
            float(row.get("relative_l2_error", 0.0)) for row in all_comparisons
        ),
    }
    _require(
        payload["maximum_observed_errors"] == observed,
        "maximum observed errors are inconsistent",
    )
    return payload


def run_routing_identity_preflight(
    output_path: Path, expected_identity: dict[str, Any]
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": PREFLIGHT_VERSION,
        "status": "failed",
        "steps": PREFLIGHT_STEPS,
        "seed": PREFLIGHT_SEED,
        "identity": expected_identity,
        "identity_sha256": _json_hash(expected_identity),
        "objective": expected_identity["objective"],
        "optimizer": expected_identity["optimizer"],
        "sam": expected_identity["sam"],
        "runtime": expected_identity["runtime"],
        "tolerances": PREFLIGHT_TOLERANCES,
        "synthetic_fixture": expected_identity["synthetic_fixture"],
    }
    try:
        hardware = validate_runtime()
        current_environment = _json_normalize(asdict(hardware))
        payload["environment"] = current_environment
        equity_count = int(expected_identity["packed_shape"]["equity_count"])
        train_sample_count = int(expected_identity["optimizer"]["train_sample_count"])
        maximum_epochs = int(expected_identity["optimizer"]["maximum_epochs"])
        batch = _synthetic_batch(equity_count)
        _require(
            _fixture_metadata(batch) == expected_identity["synthetic_fixture"],
            "generated fixture drifted from expected identity",
        )
        executions = [
            _run_execution(
                mode,
                batch,
                equity_count,
                train_sample_count,
                maximum_epochs,
            )
            for mode in PREFLIGHT_EXECUTIONS
        ]
        comparisons = [
            comparison
            for execution in executions
            for comparison in execution["comparisons"]
        ]
        payload["executions"] = executions
        payload["maximum_observed_errors"] = {
            "absolute": max(
                float(item.get("maximum_absolute_error", 0.0)) for item in comparisons
            ),
            "relative_l2": max(
                float(item.get("relative_l2_error", 0.0)) for item in comparisons
            ),
        }
        payload["status"] = "passed"
        validate_routing_identity_preflight(
            payload, expected_identity, current_environment
        )
    except BaseException as exc:
        payload["status"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        _atomic_write_json(output_path, payload)
        raise RuntimeError(
            f"Routing identity preflight failed; see {output_path}"
        ) from exc
    _atomic_write_json(output_path, payload)
    if payload["status"] != "passed":
        raise RuntimeError(f"Routing identity preflight failed; see {output_path}")
    return payload
