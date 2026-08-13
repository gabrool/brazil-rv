from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from brazil_rv.modeling.contract import (
    ABSOLUTE_PATCH_COUNT,
    EQUITY_COUNT,
    EXPECTED_SPLIT_SAMPLE_COUNTS,
    GH200_RUNTIME,
    HORIZON_COUNT,
    INSTRUMENT_COUNT,
    LOCAL_CONTEXT_COUNT,
    PATCH_INPUT_WIDTH,
    PEER_STATE_WIDTH,
    SLOW_FEATURE_COUNT,
    TCNSettings,
    resolve_tcn_architecture,
)
from brazil_rv.modeling.engine import (
    _predict,
    _rng_state,
    _rng_states_equal,
    _to_cuda,
    compile_model,
    objective_loss,
    run_effective_batch_update,
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
        self.routing_gradients_seen = False
        self.routing_changed = False

    def begin_step(self) -> None:
        self.base_reference = _parameter_snapshot(self.model, routing=False)
        self.routing_reference = _parameter_snapshot(self.model, routing=True)
        self.events = {}

    def __call__(self, event: str, model: nn.Module) -> None:
        if model is not self.model:
            raise RuntimeError("SAM observer received an unexpected model")
        if event in ("first_gradients", "second_gradients"):
            self.events[event] = _gradient_snapshot(model, routing=False)
            self.routing_gradients_seen = self.routing_gradients_seen or bool(
                _gradient_snapshot(model, routing=True)
            )
            return
        base = _parameter_snapshot(model, routing=False)
        routing = _parameter_snapshot(model, routing=True)
        self.events[event] = _difference_snapshot(base, self.base_reference)
        self.routing_changed = self.routing_changed or any(
            not torch.equal(value, self.routing_reference[name])
            for name, value in routing.items()
        )


def _synthetic_batch() -> dict[str, torch.Tensor]:
    batch_size = 1
    active_equities = 32
    state_position = 15
    patches = torch.zeros(
        batch_size,
        INSTRUMENT_COUNT,
        ABSOLUTE_PATCH_COUNT,
        PATCH_INPUT_WIDTH,
    )
    history = torch.zeros(
        batch_size, INSTRUMENT_COUNT, ABSOLUTE_PATCH_COUNT, dtype=torch.bool
    )
    history[:, :active_equities, 12:state_position] = True
    history[:, EQUITY_COUNT : EQUITY_COUNT + LOCAL_CONTEXT_COUNT, :state_position] = (
        True
    )
    history[:, EQUITY_COUNT + LOCAL_CONTEXT_COUNT :] = True
    instrument = torch.zeros(batch_size, INSTRUMENT_COUNT, dtype=torch.bool)
    instrument[:, :active_equities] = True
    instrument[:, EQUITY_COUNT:] = True
    generator = torch.Generator().manual_seed(PREFLIGHT_SEED)
    slow = 0.01 * torch.randn(
        batch_size,
        INSTRUMENT_COUNT,
        SLOW_FEATURE_COUNT,
        generator=generator,
    )
    peer = 0.01 * torch.randn(
        batch_size, EQUITY_COUNT, PEER_STATE_WIDTH, generator=generator
    )
    targets = torch.zeros(batch_size, EQUITY_COUNT, HORIZON_COUNT)
    ranks = torch.linspace(-0.95, 0.95, active_equities)
    targets[:, :active_equities] = ranks[None, :, None]
    label_mask = torch.zeros(batch_size, EQUITY_COUNT, HORIZON_COUNT, dtype=torch.bool)
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


def _seeded_models() -> tuple[nn.Module, nn.Module]:
    torch.manual_seed(PREFLIGHT_SEED)
    legacy = build_neural_model(
        "tcn", resolve_tcn_architecture(_settings("legacy")), "selected"
    ).to("cuda")
    torch.manual_seed(PREFLIGHT_SEED)
    scaffold = build_neural_model(
        "tcn", resolve_tcn_architecture(_settings("factorial_v1")), "selected"
    ).to("cuda")
    return legacy, scaffold


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


def _run_execution(execution: str, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
    legacy, scaffold = _seeded_models()
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
        legacy_optimizer, EXPECTED_SPLIT_SAMPLE_COUNTS["train"]
    )
    scaffold_scheduler, _, _ = build_scheduler(
        scaffold_optimizer, EXPECTED_SPLIT_SAMPLE_COUNTS["train"]
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

    inactive_passed = (
        not scaffold_capture.routing_gradients_seen
        and not scaffold_capture.routing_changed
    )
    comparisons.append(
        {
            "execution": execution,
            "step": PREFLIGHT_STEPS,
            "quantity": "inactive_routing_parameters",
            "passed": inactive_passed,
            "gradients_seen": scaffold_capture.routing_gradients_seen,
            "parameters_changed": scaffold_capture.routing_changed,
            "sha256": _mapping_hash(_parameter_snapshot(scaffold, routing=True)),
        }
    )
    return {
        "execution": execution,
        "compile_setups": compile_setups,
        "comparisons": comparisons,
        "passed": all(bool(item["passed"]) for item in comparisons),
    }


def run_routing_identity_preflight(output_path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": PREFLIGHT_VERSION,
        "status": "failed",
        "steps": PREFLIGHT_STEPS,
        "seed": PREFLIGHT_SEED,
        "objective": {"name": "soft_spearman", "temperature": 0.50},
        "optimizer": {"name": "sam_adamw", "rho": 0.125},
        "runtime": asdict(GH200_RUNTIME),
        "tolerances": PREFLIGHT_TOLERANCES,
        "synthetic_fixture": {
            "physical_microbatch_size": 1,
            "accumulation_steps": GH200_RUNTIME.accumulation_steps,
            "active_equities": 32,
        },
    }
    try:
        hardware = validate_runtime()
        payload["environment"] = asdict(hardware)
        batch = _synthetic_batch()
        executions = [_run_execution(mode, batch) for mode in ("eager", "compiled")]
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
        payload["status"] = (
            "passed" if all(bool(item["passed"]) for item in comparisons) else "failed"
        )
    except BaseException as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        _atomic_write_json(output_path, payload)
        raise RuntimeError(
            f"Routing identity preflight failed; see {output_path}"
        ) from exc
    _atomic_write_json(output_path, payload)
    if payload["status"] != "passed":
        raise RuntimeError(f"Routing identity preflight failed; see {output_path}")
    return payload
