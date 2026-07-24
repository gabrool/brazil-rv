from __future__ import annotations

import copy
import inspect
import json
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import brazil_rv.modeling.engine as engine_module
from brazil_rv.modeling.contract import (
    ABSOLUTE_PATCH_COUNT,
    ADAMW_BETAS,
    ADAMW_EPS,
    ADAMW_LR,
    ADAMW_WEIGHT_DECAY,
    COMPILE_PARITY_GRADIENT_COSINE_MIN,
    COMPILE_PARITY_GRADIENT_MAX_ABSOLUTE_ATOL,
    COMPILE_PARITY_GRADIENT_MAX_ABSOLUTE_RTOL,
    COMPILE_PARITY_GRADIENT_RELATIVE_L2_MAX,
    COMPILE_PARITY_LOSS_ATOL,
    COMPILE_PARITY_LOSS_RTOL,
    COMPILE_PARITY_PREDICTION_ATOL,
    COMPILE_PARITY_PREDICTION_RTOL,
    CompileEvaluationWarmupReport,
    CompileParityThresholds,
    CompileSetupReport,
    EFFECTIVE_BATCH_SIZE,
    EQUITY_COUNT,
    FINAL_LR_FACTOR,
    INSTRUMENT_COUNT,
    MAX_EPOCHS,
    MUON_ADJUST_LR_FN,
    MUON_COMPATIBILITY_CONTRACT_VERSION,
    MUON_EPS,
    MUON_LR,
    MUON_MOMENTUM,
    MUON_NESTEROV,
    MUON_NS_COEFFICIENTS,
    MUON_NS_STEPS,
    MUON_WEIGHT_DECAY,
    PATCH_INPUT_WIDTH,
    RUNTIME_PROFILES,
    TORCH_COMPILE_COMPATIBILITY_CONTRACT_VERSION,
)
from brazil_rv.modeling.evaluate import _validate_run_checkpoint_identity
from brazil_rv.modeling.engine import (
    _compile_parity_report,
    _filter_evaluation_rows,
    _optimizer_update,
    build_compile_metadata,
    checkpoint_payload,
    clone_eager_reference_model,
    compile_model,
    masked_huber_loss,
    train_one_epoch,
    validate_runtime_profile,
)
from brazil_rv.modeling.layers import MuonLinear
from brazil_rv.modeling.metrics import create_metric_table
from brazil_rv.modeling.train import _atomic_write_json
from brazil_rv.modeling.model import CrossAssetPatchITransformerV1
from brazil_rv.modeling.muon import PYTORCH_MUON_REFERENCE, PyTorch213Muon
from brazil_rv.modeling.optim import (
    OFFICIAL_MUON_BACKEND,
    REFERENCE_MUON_BACKEND,
    build_schedulers,
    learning_rate_factor,
    partition_parameters,
)


class _TrackingOptimizer:
    def __init__(self, parameters: list[nn.Parameter]) -> None:
        self.parameters = parameters
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1

    def zero_grad(self, *, set_to_none: bool) -> None:
        assert set_to_none
        for parameter in self.parameters:
            parameter.grad = None


class _TrackingScheduler:
    def __init__(self) -> None:
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1


def _assert_nested_exact(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_exact(left_item, right_item)
    else:
        assert left == right


class _AtomicFallbackModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.muon_weight = nn.Parameter(
            torch.arange(6, dtype=torch.float32).reshape(2, 3) / 10
        )
        self.adamw_bias = nn.Parameter(torch.tensor([0.5, -0.25]))


def test_real_fallback_participates_in_atomic_joint_updates() -> None:
    model = _AtomicFallbackModel()
    optimizers: dict[str, torch.optim.Optimizer] = {
        "muon": PyTorch213Muon(
            [model.muon_weight],
            lr=MUON_LR,
            momentum=MUON_MOMENTUM,
            nesterov=MUON_NESTEROV,
            ns_coefficients=MUON_NS_COEFFICIENTS,
            eps=MUON_EPS,
            ns_steps=MUON_NS_STEPS,
            weight_decay=MUON_WEIGHT_DECAY,
            adjust_lr_fn=MUON_ADJUST_LR_FN,
        ),
        "adamw": torch.optim.AdamW(
            [model.adamw_bias],
            lr=ADAMW_LR,
            betas=ADAMW_BETAS,
            eps=ADAMW_EPS,
            weight_decay=ADAMW_WEIGHT_DECAY,
            fused=False,
        ),
    }
    schedulers = {
        name: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        for name, optimizer in optimizers.items()
    }

    model.muon_weight.grad = torch.full_like(model.muon_weight, 0.25)
    model.adamw_bias.grad = torch.tensor([0.5, -0.75])
    parameter_snapshots = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    scheduler_epochs = {
        name: scheduler.last_epoch for name, scheduler in schedulers.items()
    }
    succeeded, gradient_norm = _optimizer_update(model, optimizers, schedulers)
    assert succeeded
    assert np.isfinite(gradient_norm)
    assert not torch.equal(model.muon_weight, parameter_snapshots["muon_weight"])
    assert not torch.equal(model.adamw_bias, parameter_snapshots["adamw_bias"])
    assert all(
        schedulers[name].last_epoch == scheduler_epochs[name] + 1 for name in schedulers
    )
    assert all(parameter.grad is None for parameter in model.parameters())
    momentum_buffer = optimizers["muon"].state[model.muon_weight]["momentum_buffer"]
    assert torch.isfinite(momentum_buffer).all()

    for nonfinite_partition in ("muon", "adamw"):
        model.muon_weight.grad = torch.full_like(model.muon_weight, 0.125)
        model.adamw_bias.grad = torch.tensor([0.25, -0.5])
        nonfinite_parameter = (
            model.muon_weight if nonfinite_partition == "muon" else model.adamw_bias
        )
        nonfinite_parameter.grad.flatten()[0] = float("inf")
        parameter_snapshots = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        optimizer_states = {
            name: copy.deepcopy(optimizer.state_dict())
            for name, optimizer in optimizers.items()
        }
        scheduler_states = {
            name: copy.deepcopy(scheduler.state_dict())
            for name, scheduler in schedulers.items()
        }

        succeeded, gradient_norm = _optimizer_update(model, optimizers, schedulers)
        assert not succeeded
        assert not np.isfinite(gradient_norm)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(
                parameter,
                parameter_snapshots[name],
                atol=0,
                rtol=0,
            )
        for name, optimizer in optimizers.items():
            _assert_nested_exact(optimizer.state_dict(), optimizer_states[name])
        for name, scheduler in schedulers.items():
            _assert_nested_exact(scheduler.state_dict(), scheduler_states[name])
        assert all(parameter.grad is None for parameter in model.parameters())


def test_bf16_nonfinite_gradient_skips_joint_optimizer_update() -> None:
    for nonfinite_partition in ("muon", "adamw"):
        model = nn.Linear(2, 1)
        model.weight.grad = torch.ones_like(model.weight)
        model.bias.grad = torch.ones_like(model.bias)
        parameter_groups = {
            "muon": [model.weight],
            "adamw": [model.bias],
        }
        parameter_groups[nonfinite_partition][0].grad.fill_(float("inf"))
        optimizers = {
            name: _TrackingOptimizer(parameters)
            for name, parameters in parameter_groups.items()
        }
        schedulers = {name: _TrackingScheduler() for name in optimizers}
        succeeded, gradient_norm = _optimizer_update(model, optimizers, schedulers)
        assert not succeeded
        assert not np.isfinite(gradient_norm)
        assert all(optimizer.step_count == 0 for optimizer in optimizers.values())
        assert all(scheduler.step_count == 0 for scheduler in schedulers.values())
        assert all(parameter.grad is None for parameter in model.parameters())


def test_cloud_engine_has_no_scaler_dependency() -> None:
    assert "scaler" not in inspect.signature(_optimizer_update).parameters
    assert "scaler" not in inspect.signature(train_one_epoch).parameters


def test_muon_partition_is_complete_disjoint_and_exact() -> None:
    model = CrossAssetPatchITransformerV1("full")
    groups = partition_parameters(model, "hybrid")
    routed = [parameter for group in groups.values() for parameter in group]
    trainable = list(model.parameters())
    assert len(routed) == len({id(parameter) for parameter in routed})
    assert {id(parameter) for parameter in routed} == {
        id(parameter) for parameter in trainable
    }
    expected_muon = {
        id(module.weight)
        for module in model.modules()
        if isinstance(module, MuonLinear)
    }
    assert {id(parameter) for parameter in groups["muon"]} == expected_muon
    assert all(parameter.ndim == 2 for parameter in groups["muon"])


def test_adamw_decay_partition() -> None:
    model = CrossAssetPatchITransformerV1("full")
    groups = partition_parameters(model, "adamw")
    decay_ids = {id(parameter) for parameter in groups["decay"]}
    no_decay_ids = {id(parameter) for parameter in groups["no_decay"]}
    for module in model.modules():
        if isinstance(module, nn.Linear):
            assert id(module.weight) in decay_ids
            if module.bias is not None:
                assert id(module.bias) in no_decay_ids
        elif isinstance(module, (nn.Embedding, nn.RMSNorm)):
            assert id(module.weight) in no_decay_ids
    assert id(model.state_token) in no_decay_ids


def test_masked_huber_reduction() -> None:
    predictions = torch.zeros(2, 3, 2)
    targets = torch.tensor(
        [
            [[0.0, 1.0], [2.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.5], [0.0, 1.5], [0.0, 0.0]],
        ]
    )
    mask = torch.tensor(
        [
            [[True, True], [True, False], [False, False]],
            [[False, True], [False, True], [False, False]],
        ]
    )
    expected = torch.tensor((0.625 + 0.5625) / 2.0)
    torch.testing.assert_close(masked_huber_loss(predictions, targets, mask), expected)
    all_invalid_predictions = torch.ones(1, 3, 2, requires_grad=True)
    all_invalid_loss = masked_huber_loss(
        all_invalid_predictions,
        torch.zeros_like(all_invalid_predictions),
        torch.zeros(1, 3, 2, dtype=torch.bool),
    )
    assert all_invalid_loss == 0.0
    all_invalid_loss.backward()
    assert not all_invalid_predictions.grad.any()


def test_scheduler_endpoints_and_actual_update_numbering() -> None:
    total_steps = 1_000
    warmup_steps = 50
    assert learning_rate_factor(0, total_steps, warmup_steps) == 0.0
    assert learning_rate_factor(warmup_steps, total_steps, warmup_steps) == 1.0
    assert learning_rate_factor(total_steps, total_steps, warmup_steps) == 0.1

    parameter = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    schedulers, steps_per_epoch, actual_warmup = build_schedulers(
        {"adamw": optimizer}, EFFECTIVE_BATCH_SIZE * 10
    )
    actual_total = steps_per_epoch * MAX_EPOCHS
    actual_factors = []
    for _ in range(actual_total):
        actual_factors.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        schedulers["adamw"].step()
    assert actual_factors[0] == learning_rate_factor(1, actual_total, actual_warmup)
    assert actual_factors[0] > 0.0
    assert actual_factors[-1] == FINAL_LR_FACTOR


def test_compile_setup_modern_explicit_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_compile(module: nn.Module, **kwargs: object) -> None:
        calls.append(kwargs)

    config = SimpleNamespace(backward_pass_autocast="same_as_forward")
    monkeypatch.setattr(engine_module, "functorch_config", config)
    monkeypatch.setattr(nn.Module, "compile", fake_compile)
    profile = RUNTIME_PROFILES["a10"]
    report = compile_model(nn.Linear(2, 1), profile)
    assert calls == [
        {
            "backend": profile.compile_backend,
            "mode": profile.compile_mode,
            "fullgraph": profile.compile_fullgraph,
            "dynamic": profile.compile_dynamic,
        }
    ]
    assert config.backward_pass_autocast == "off"
    assert report == CompileSetupReport(
        api="nn.Module.compile",
        backend="inductor",
        mode="reduce-overhead",
        fullgraph=True,
        dynamic=False,
        backward_pass_autocast_control_available=True,
        backward_pass_autocast_policy="explicit_off",
    )


def test_compile_setup_legacy_implicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_compile(module: nn.Module, **kwargs: object) -> None:
        calls.append(kwargs)

    config = SimpleNamespace()
    monkeypatch.setattr(engine_module, "functorch_config", config)
    monkeypatch.setattr(nn.Module, "compile", fake_compile)
    profile = RUNTIME_PROFILES["gh200"]
    report = compile_model(nn.Linear(2, 1), profile)
    assert calls == [
        {
            "backend": profile.compile_backend,
            "mode": profile.compile_mode,
            "fullgraph": profile.compile_fullgraph,
            "dynamic": profile.compile_dynamic,
        }
    ]
    assert not hasattr(config, "backward_pass_autocast")
    assert not report.backward_pass_autocast_control_available
    assert report.backward_pass_autocast_policy == "legacy_implicit"


def test_compile_setup_requires_callable_module_api() -> None:
    model = SimpleNamespace(compile=None)
    with pytest.raises(RuntimeError, match="nn.Module.compile"):
        compile_model(model, RUNTIME_PROFILES["a10"])


def _synthetic_compile_parity(
    *,
    prediction_pair: tuple[torch.Tensor, torch.Tensor] | None = None,
    loss_pair: tuple[float, float] = (1.0, 1.0),
    gradient_pair: tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
    include_backward: bool = True,
):
    if prediction_pair is None:
        predictions = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        prediction_pair = (predictions, predictions.clone())
    if gradient_pair is None:
        gradient = torch.tensor([1.0, 2.0], dtype=torch.float32)
        gradient_pair = (gradient, gradient.clone())
    eager_gradients = (("weight", gradient_pair[0]),) if include_backward else None
    compiled_gradients = (("weight", gradient_pair[1]),) if include_backward else None
    return _compile_parity_report(
        prediction_pair[0],
        prediction_pair[1],
        torch.tensor(loss_pair[0]),
        torch.tensor(loss_pair[1]),
        eager_gradients,
        compiled_gradients,
    )


def test_compile_parity_exact_and_within_threshold_pass() -> None:
    assert _synthetic_compile_parity().passed
    eager_predictions = torch.tensor([[1.0, 2.0]])
    compiled_predictions = eager_predictions + 1e-3
    report = _synthetic_compile_parity(
        prediction_pair=(eager_predictions, compiled_predictions),
        loss_pair=(1.0, 1.0001),
        gradient_pair=(
            torch.tensor([1.0, 2.0]),
            torch.tensor([1.00001, 2.0]),
        ),
    )
    assert report.passed


def test_compile_parity_prediction_and_loss_divergence_fail() -> None:
    eager_predictions = torch.tensor([[1.0, 2.0]])
    prediction_failure = _synthetic_compile_parity(
        prediction_pair=(eager_predictions, eager_predictions + 0.1)
    )
    loss_failure = _synthetic_compile_parity(loss_pair=(1.0, 1.1))
    assert not prediction_failure.passed
    assert not prediction_failure.prediction_allclose
    assert not loss_failure.passed
    assert loss_failure.loss_absolute_difference > loss_failure.loss_tolerance


def test_compile_parity_gradient_presence_and_finiteness_failures() -> None:
    presence_failure = _synthetic_compile_parity(gradient_pair=(torch.ones(2), None))
    eager_nonfinite = _synthetic_compile_parity(
        gradient_pair=(torch.tensor([float("inf")]), torch.ones(1))
    )
    compiled_nonfinite = _synthetic_compile_parity(
        gradient_pair=(torch.ones(1), torch.tensor([float("nan")]))
    )
    assert not presence_failure.passed
    assert not presence_failure.gradient_presence_match
    assert not eager_nonfinite.passed
    assert not eager_nonfinite.eager_gradients_finite
    assert not compiled_nonfinite.passed
    assert not compiled_nonfinite.compiled_gradients_finite


def test_compile_parity_nonfinite_predictions_fail() -> None:
    finite = torch.ones(1, 1)
    eager_nonfinite = _synthetic_compile_parity(
        prediction_pair=(torch.full((1, 1), float("inf")), finite)
    )
    compiled_nonfinite = _synthetic_compile_parity(
        prediction_pair=(finite, torch.full((1, 1), float("nan")))
    )
    assert not eager_nonfinite.passed
    assert not eager_nonfinite.eager_predictions_finite
    assert not compiled_nonfinite.passed
    assert not compiled_nonfinite.compiled_predictions_finite


def test_compile_parity_gradient_relative_l2_threshold_failure() -> None:
    report = _synthetic_compile_parity(
        gradient_pair=(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.011]))
    )
    assert not report.passed
    assert report.gradient_relative_l2_error is not None
    assert report.gradient_relative_l2_error > COMPILE_PARITY_GRADIENT_RELATIVE_L2_MAX


def test_compile_parity_gradient_cosine_threshold_failure() -> None:
    report = _synthetic_compile_parity(
        gradient_pair=(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.02]))
    )
    assert not report.passed
    assert report.gradient_cosine_similarity is not None
    assert report.gradient_cosine_similarity < COMPILE_PARITY_GRADIENT_COSINE_MIN


def test_compile_parity_gradient_max_absolute_threshold_failure() -> None:
    eager_gradient = torch.ones(100)
    compiled_gradient = eager_gradient.clone()
    compiled_gradient[0] += 0.012
    report = _synthetic_compile_parity(
        gradient_pair=(eager_gradient, compiled_gradient)
    )
    assert not report.passed
    assert report.gradient_relative_l2_error is not None
    assert report.gradient_relative_l2_error <= COMPILE_PARITY_GRADIENT_RELATIVE_L2_MAX
    assert report.gradient_max_absolute_difference is not None
    assert report.gradient_max_absolute_tolerance is not None
    assert (
        report.gradient_max_absolute_difference > report.gradient_max_absolute_tolerance
    )


def test_compile_parity_zero_gradient_rules() -> None:
    both_zero = _synthetic_compile_parity(
        gradient_pair=(torch.zeros(2), torch.zeros(2))
    )
    compiled_nonzero = _synthetic_compile_parity(
        gradient_pair=(torch.zeros(2), torch.tensor([1e-4, 0.0]))
    )
    assert both_zero.passed
    assert both_zero.gradient_relative_l2_error == 0.0
    assert both_zero.gradient_cosine_similarity == 1.0
    assert not compiled_nonzero.passed
    assert compiled_nonzero.gradient_relative_l2_error == float("inf")
    assert compiled_nonzero.gradient_cosine_similarity == -1.0


def test_compile_parity_forward_only_has_null_gradient_fields() -> None:
    report = _synthetic_compile_parity(include_backward=False)
    assert report.passed
    assert report.mode == "forward_only"
    for field in (
        "gradient_presence_match",
        "eager_gradients_finite",
        "compiled_gradients_finite",
        "gradient_parameter_count",
        "eager_gradient_l2_norm",
        "compiled_gradient_l2_norm",
        "eager_gradient_max_absolute",
        "gradient_relative_l2_error",
        "gradient_cosine_similarity",
        "gradient_max_absolute_difference",
        "gradient_max_absolute_tolerance",
    ):
        assert getattr(report, field) is None


def test_compile_qualification_uses_requested_grad_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eager_model = nn.Linear(1, 1, bias=False)
    compiled_model = copy.deepcopy(eager_model)
    batch = {
        "targets": torch.zeros(1, 1, 1),
        "label_mask": torch.ones(1, 1, 1, dtype=torch.bool),
    }
    grad_modes: list[bool] = []

    def record_grad_mode(
        model: nn.Module, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        grad_modes.append(torch.is_grad_enabled())
        return next(model.parameters()).sum().reshape(1, 1, 1)

    monkeypatch.setattr(engine_module, "_to_cuda", lambda batch: batch)
    monkeypatch.setattr(engine_module, "_predict", record_grad_mode)
    monkeypatch.setattr(torch, "autocast", lambda **_: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    with torch.no_grad():
        forward_backward = engine_module.qualify_eager_compiled_model(
            eager_model,
            compiled_model,
            batch,
            include_backward=True,
        )
    assert forward_backward.mode == "forward_backward"
    assert grad_modes == [True, True]

    grad_modes.clear()
    with torch.no_grad():
        forward_only = engine_module.qualify_eager_compiled_model(
            eager_model,
            compiled_model,
            batch,
            include_backward=False,
        )
    assert grad_modes == [False, False]
    assert forward_only.mode == "forward_only"
    for field in (
        "gradient_presence_match",
        "eager_gradients_finite",
        "compiled_gradients_finite",
        "gradient_parameter_count",
        "eager_gradient_l2_norm",
        "compiled_gradient_l2_norm",
        "eager_gradient_max_absolute",
        "gradient_relative_l2_error",
        "gradient_cosine_similarity",
        "gradient_max_absolute_difference",
        "gradient_max_absolute_tolerance",
    ):
        assert getattr(forward_only, field) is None


def test_compile_metadata_schema_is_exact() -> None:
    setup = CompileSetupReport(
        api="nn.Module.compile",
        backend="inductor",
        mode="reduce-overhead",
        fullgraph=True,
        dynamic=False,
        backward_pass_autocast_control_available=False,
        backward_pass_autocast_policy="legacy_implicit",
    )
    parity = _synthetic_compile_parity(include_backward=False)
    warmup = CompileEvaluationWarmupReport(
        evaluation_pass_seconds=(1.0, 2.0, 3.0, 4.0, 5.0),
        evaluation_steady_state_median_seconds=4.0,
        peak_allocated_cuda_memory_bytes=101,
        peak_reserved_cuda_memory_bytes=202,
    )
    metadata = build_compile_metadata(setup, parity, warmup)
    assert metadata == {
        "enabled": True,
        "eager_fallback_allowed": False,
        "setup": asdict(setup),
        "parity_thresholds": asdict(CompileParityThresholds()),
        "parity": asdict(parity),
        "warmup": asdict(warmup),
    }
    assert "backward_pass_autocast" not in metadata
    assert "backward_pass_autocast" not in metadata["setup"]
    assert TORCH_COMPILE_COMPATIBILITY_CONTRACT_VERSION == (
        "TORCH_COMPILE_COMPATIBILITY_V1"
    )
    assert asdict(CompileParityThresholds()) == {
        "prediction_atol": COMPILE_PARITY_PREDICTION_ATOL,
        "prediction_rtol": COMPILE_PARITY_PREDICTION_RTOL,
        "loss_atol": COMPILE_PARITY_LOSS_ATOL,
        "loss_rtol": COMPILE_PARITY_LOSS_RTOL,
        "gradient_relative_l2_max": COMPILE_PARITY_GRADIENT_RELATIVE_L2_MAX,
        "gradient_cosine_min": COMPILE_PARITY_GRADIENT_COSINE_MIN,
        "gradient_max_absolute_atol": COMPILE_PARITY_GRADIENT_MAX_ABSOLUTE_ATOL,
        "gradient_max_absolute_rtol": COMPILE_PARITY_GRADIENT_MAX_ABSOLUTE_RTOL,
    }


def test_clone_eager_reference_has_distinct_exact_parameters_and_rng() -> None:
    torch.manual_seed(17)
    model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4))
    rng_state = torch.random.get_rng_state().clone()
    reference = clone_eager_reference_model(model)
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    for (source_name, source), (reference_name, cloned) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert source_name == reference_name
        torch.testing.assert_close(source, cloned, atol=0, rtol=0)
        assert source is not cloned
        assert source.data_ptr() != cloned.data_ptr()
    with torch.no_grad():
        next(reference.parameters()).add_(1.0)
    assert not torch.equal(next(model.parameters()), next(reference.parameters()))


@pytest.mark.parametrize("profile_name", tuple(RUNTIME_PROFILES))
def test_runtime_profile_hardware_validation(
    profile_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = RUNTIME_PROFILES[profile_name]
    device_name = {
        "a10": "NVIDIA A10",
        "a100": "NVIDIA A100-SXM4-40GB",
        "gh200": "NVIDIA GH200 480GB",
    }[profile_name]
    cpu_architecture = profile.required_cpu_architecture or "x86_64"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: device_name)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _: profile.expected_compute_capability,
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=profile.minimum_vram_bytes),
    )
    monkeypatch.setattr(
        engine_module.system_platform, "machine", lambda: cpu_architecture
    )
    monkeypatch.setattr(
        engine_module.system_platform, "platform", lambda: "test-platform"
    )
    hardware = validate_runtime_profile(profile)
    assert hardware.profile == profile.name
    assert hardware.compute_capability == profile.expected_compute_capability
    if profile.required_device_name_fragment is not None:
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "wrong-device")
    else:
        monkeypatch.setattr(engine_module.system_platform, "machine", lambda: "x86_64")
    with pytest.raises(RuntimeError):
        validate_runtime_profile(profile)


def test_padded_evaluation_matches_unpadded_reference() -> None:
    generator = np.random.default_rng(9)
    real_predictions = generator.normal(size=(2, 30, 3)).astype(np.float32)
    real_targets = generator.normal(size=(2, 30, 3)).astype(np.float32)
    real_returns = generator.normal(size=(2, 30, 3)).astype(np.float32)
    real_mask = np.ones((2, 30, 3), dtype=bool)
    padded_predictions = np.concatenate(
        (real_predictions, np.repeat(real_predictions[-1:], 2, axis=0))
    )
    cpu_batch = {
        "targets": torch.from_numpy(
            np.concatenate((real_targets, np.zeros((2, 30, 3), np.float32)))
        ),
        "raw_returns": torch.from_numpy(
            np.concatenate((real_returns, np.zeros((2, 30, 3), np.float32)))
        ),
        "label_mask": torch.from_numpy(
            np.concatenate((real_mask, np.zeros((2, 30, 3), bool)))
        ),
        "sample_valid_mask": torch.tensor([True, True, False, False]),
        "date_idx": torch.tensor([5, 5, -1, -1]),
        "decision_idx": torch.tensor([0, 1, -1, -1]),
    }
    filtered = _filter_evaluation_rows(torch.from_numpy(padded_predictions), cpu_batch)
    np.testing.assert_array_equal(filtered["predictions"], real_predictions)
    np.testing.assert_array_equal(filtered["targets"], real_targets)
    reference, reference_daily = create_metric_table(
        real_predictions,
        real_targets,
        real_returns,
        real_mask,
        np.asarray([5, 5], dtype=np.int64),
        np.asarray([0, 1], dtype=np.int64),
    )
    padded, padded_daily = create_metric_table(
        filtered["predictions"],
        filtered["targets"],
        filtered["raw_returns"],
        filtered["label_mask"],
        filtered["date_idx"],
        filtered["decision_idx"],
    )
    np.testing.assert_allclose(
        padded["primary_score"], reference["primary_score"], equal_nan=True
    )
    np.testing.assert_allclose(
        padded["mean_valid_sample_spearman_ic"],
        reference["mean_valid_sample_spearman_ic"],
        equal_nan=True,
    )
    for padded_horizon, reference_horizon in zip(
        padded["horizons"], reference["horizons"], strict=True
    ):
        assert padded_horizon.keys() == reference_horizon.keys()
        np.testing.assert_allclose(
            list(padded_horizon.values()),
            list(reference_horizon.values()),
            equal_nan=True,
        )
    for padded_row, reference_row in zip(padded_daily, reference_daily, strict=True):
        assert padded_row.keys() == reference_row.keys()
        np.testing.assert_allclose(
            list(padded_row.values()),
            list(reference_row.values()),
            equal_nan=True,
        )


def test_daily_ic_aggregation_with_ties() -> None:
    tied = np.repeat(np.arange(10, dtype=np.float32), 3)
    predictions = np.empty((2, 30, 3), dtype=np.float32)
    targets = np.empty_like(predictions)
    for horizon in range(3):
        predictions[0, :, horizon] = tied
        predictions[1, :, horizon] = -tied
        targets[:, :, horizon] = tied
    mask = np.ones_like(predictions, dtype=bool)
    summary, daily_rows = create_metric_table(
        predictions,
        targets,
        np.zeros_like(predictions),
        mask,
        np.asarray([5, 5], dtype=np.int64),
        np.asarray([0, 1], dtype=np.int64),
    )
    np.testing.assert_allclose(summary["primary_score"], 0.0, atol=1e-15)
    np.testing.assert_allclose(
        [row["spearman_ic"] for row in daily_rows], 0.0, atol=1e-15
    )


def _matching_run_identity(
    feature_store: Path,
    *,
    optimizer_variant: str = "hybrid",
    muon_backend: str | None = OFFICIAL_MUON_BACKEND,
) -> dict[str, object]:
    return {
        "contract_version": "CROSS_ASSET_ITRANSFORMER_V1",
        "cloud_runtime_contract_version": ("CROSS_ASSET_ITRANSFORMER_CLOUD_RUNTIME_V1"),
        "muon_compatibility_contract_version": (MUON_COMPATIBILITY_CONTRACT_VERSION),
        "muon_backend": muon_backend,
        "muon_reference": dict(PYTORCH_MUON_REFERENCE),
        "model_variant": "full",
        "optimizer_variant": optimizer_variant,
        "seed": 11,
        "runtime_profile": "a10",
        "resolved_feature_store_path": str(feature_store),
        "git_commit_sha": "test-sha",
        "architecture_constants": {"d_model": 256},
    }


@pytest.mark.parametrize(
    "muon_backend",
    (OFFICIAL_MUON_BACKEND, REFERENCE_MUON_BACKEND),
)
def test_evaluation_identity_accepts_matching_hybrid_backends(
    muon_backend: str,
    tmp_path: Path,
) -> None:
    feature_store = tmp_path.resolve()
    manifest = _matching_run_identity(
        feature_store,
        muon_backend=muon_backend,
    )
    checkpoint = dict(manifest)
    evaluation_profile = "a100"
    assert evaluation_profile != manifest["runtime_profile"]
    manifest["evaluation_runtime_profile"] = evaluation_profile
    _validate_run_checkpoint_identity(manifest, checkpoint, feature_store)


def test_evaluation_identity_accepts_matching_adamw_without_muon(
    tmp_path: Path,
) -> None:
    feature_store = tmp_path.resolve()
    manifest = _matching_run_identity(
        feature_store,
        optimizer_variant="adamw",
        muon_backend=None,
    )
    _validate_run_checkpoint_identity(manifest, dict(manifest), feature_store)


def test_evaluation_identity_rejects_matching_unknown_optimizer_variant(
    tmp_path: Path,
) -> None:
    feature_store = tmp_path.resolve()
    manifest = _matching_run_identity(
        feature_store,
        optimizer_variant="unknown",
        muon_backend=None,
    )
    with pytest.raises(ValueError, match="optimizer_variant"):
        _validate_run_checkpoint_identity(manifest, dict(manifest), feature_store)


@pytest.mark.parametrize(
    ("field", "stale_value"),
    (
        ("muon_compatibility_contract_version", "STALE_MUON_CONTRACT"),
        (
            "muon_reference",
            {
                "upstream_tag": "v2.12.0",
                "upstream_path": "torch/optim/_muon.py",
                "upstream_blob_sha": "stale",
            },
        ),
    ),
)
def test_evaluation_identity_rejects_stale_but_matching_muon_values(
    field: str,
    stale_value: object,
    tmp_path: Path,
) -> None:
    feature_store = tmp_path.resolve()
    manifest = _matching_run_identity(feature_store)
    checkpoint = dict(manifest)
    manifest[field] = stale_value
    checkpoint[field] = copy.deepcopy(stale_value)
    with pytest.raises(ValueError, match=field):
        _validate_run_checkpoint_identity(manifest, checkpoint, feature_store)


@pytest.mark.parametrize(
    ("optimizer_variant", "muon_backend"),
    (
        ("hybrid", None),
        ("hybrid", "unknown.Muon"),
        ("adamw", OFFICIAL_MUON_BACKEND),
    ),
)
def test_evaluation_identity_rejects_invalid_matching_backend_semantics(
    optimizer_variant: str,
    muon_backend: str | None,
    tmp_path: Path,
) -> None:
    feature_store = tmp_path.resolve()
    manifest = _matching_run_identity(
        feature_store,
        optimizer_variant=optimizer_variant,
        muon_backend=muon_backend,
    )
    with pytest.raises(ValueError, match="muon_backend"):
        _validate_run_checkpoint_identity(manifest, dict(manifest), feature_store)


@pytest.mark.parametrize(
    "field",
    (
        "contract_version",
        "cloud_runtime_contract_version",
        "muon_compatibility_contract_version",
        "muon_backend",
        "muon_reference",
        "model_variant",
        "optimizer_variant",
        "seed",
        "runtime_profile",
        "resolved_feature_store_path",
        "git_commit_sha",
        "architecture_constants",
    ),
)
def test_evaluation_identity_rejects_each_mismatch(field: str, tmp_path: Path) -> None:
    feature_store = tmp_path.resolve()
    manifest = _matching_run_identity(feature_store)
    checkpoint = dict(manifest)
    checkpoint[field] = {"mismatch": field}
    with pytest.raises(ValueError, match=field):
        _validate_run_checkpoint_identity(manifest, checkpoint, feature_store)


def test_atomic_json_write_replaces_final_without_temporary_file(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run_manifest.json"
    _atomic_write_json(output, {"status": "running"})
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "running"}
    assert not (tmp_path / "run_manifest.json.tmp").exists()


def test_checkpoint_round_trip_eager(tmp_path: Path) -> None:
    torch.manual_seed(13)
    model = CrossAssetPatchITransformerV1("temporal_only")
    model.train()
    patches = torch.zeros(1, INSTRUMENT_COUNT, ABSOLUTE_PATCH_COUNT, PATCH_INPUT_WIDTH)
    history = torch.zeros(1, INSTRUMENT_COUNT, ABSOLUTE_PATCH_COUNT, dtype=torch.bool)
    history[:, 0, 12:15] = True
    instrument = torch.zeros(1, INSTRUMENT_COUNT, dtype=torch.bool)
    instrument[:, 0] = True
    instrument[:, EQUITY_COUNT:] = True
    slow = torch.zeros(1, INSTRUMENT_COUNT, 3)
    state_position = torch.tensor([15])
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    prediction = model(patches, history, instrument, slow, state_position)
    prediction.square().mean().backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        expected = model(patches, history, instrument, slow, state_position)
    payload = checkpoint_payload(
        model,
        "temporal_only",
        "adamw",
        None,
        "a10",
        11,
        1,
        0.0,
        tmp_path,
        "test-sha",
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(payload, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored = CrossAssetPatchITransformerV1("temporal_only").eval()
    restored.load_state_dict(loaded["model_state_dict"])
    with torch.no_grad():
        actual = restored(patches, history, instrument, slow, state_position)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
