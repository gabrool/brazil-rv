from __future__ import annotations

import ast
import copy
import importlib
import inspect
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import brazil_rv.modeling.muon as compatibility_module
import brazil_rv.modeling.optim as optim_module
from brazil_rv.modeling.contract import (
    ADAMW_BETAS,
    ADAMW_EPS,
    ADAMW_LR,
    ADAMW_WEIGHT_DECAY,
    EFFECTIVE_BATCH_SIZE,
    MUON_ADJUST_LR_FN,
    MUON_EPS,
    MUON_LR,
    MUON_MOMENTUM,
    MUON_NESTEROV,
    MUON_NS_COEFFICIENTS,
    MUON_NS_STEPS,
    MUON_WEIGHT_DECAY,
)
from brazil_rv.modeling.layers import MuonLinear
from brazil_rv.modeling.model import build_neural_model
from brazil_rv.modeling.muon import (
    DEFAULT_A,
    DEFAULT_B,
    DEFAULT_C,
    DEFAULT_NS_STEPS,
    EPS,
    PYTORCH_MUON_BACKEND_NAME,
    PYTORCH_MUON_UPSTREAM_BLOB_SHA,
    PYTORCH_MUON_UPSTREAM_PATH,
    PYTORCH_MUON_UPSTREAM_TAG,
    PyTorch213Muon,
    _adjust_lr,
    _zeropower_via_newtonschulz,
    muon,
)
from brazil_rv.modeling.optim import (
    OFFICIAL_MUON_BACKEND,
    REFERENCE_MUON_BACKEND,
    build_optimizers,
    build_schedulers,
    partition_parameters,
)

REFERENCE_TORCH_VERSION = "2.13.0"
_SHAPES = ((3, 5), (5, 3), (8, 8))


def _require_official_reference() -> Any:
    version = torch.__version__.split("+", 1)[0]
    if version != REFERENCE_TORCH_VERSION:
        pytest.skip(
            "official Muon differential reference requires "
            f"PyTorch {REFERENCE_TORCH_VERSION}; found {version}"
        )
    if getattr(torch.optim, "Muon", None) is None:
        pytest.skip("official torch.optim.Muon is unavailable")
    return importlib.import_module("torch.optim._muon")


def _require_native_bf16_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        pytest.skip("native CUDA BF16 unavailable")


def _production_kwargs() -> dict[str, Any]:
    return {
        "lr": MUON_LR,
        "momentum": MUON_MOMENTUM,
        "nesterov": MUON_NESTEROV,
        "ns_coefficients": MUON_NS_COEFFICIENTS,
        "eps": MUON_EPS,
        "ns_steps": MUON_NS_STEPS,
        "weight_decay": MUON_WEIGHT_DECAY,
        "adjust_lr_fn": MUON_ADJUST_LR_FN,
    }


def _clone_kwargs(values: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in values.items()
    }


def _assert_exact(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_exact(left_item, right_item)
    else:
        assert left == right


def _parameters(
    values: list[torch.Tensor],
) -> list[nn.Parameter]:
    return [nn.Parameter(value.clone()) for value in values]


def _gradient_sequence(
    shapes: tuple[tuple[int, int], ...] = _SHAPES,
    steps: int = 10,
) -> list[list[torch.Tensor]]:
    generator = torch.Generator().manual_seed(20260723)
    return [
        [
            torch.randn(shape, generator=generator) * ((step + 1) / 10)
            for shape in shapes
        ]
        for step in range(steps)
    ]


def _assert_optimizer_step_equal(
    official_optimizer: torch.optim.Optimizer,
    compatibility_optimizer: torch.optim.Optimizer,
    official_parameters: list[nn.Parameter],
    compatibility_parameters: list[nn.Parameter],
    gradients: list[list[torch.Tensor]],
) -> None:
    for step_gradients in gradients:
        for official_parameter, compatibility_parameter, gradient in zip(
            official_parameters,
            compatibility_parameters,
            step_gradients,
            strict=True,
        ):
            official_parameter.grad = gradient.clone()
            compatibility_parameter.grad = gradient.clone()
        official_optimizer.step()
        compatibility_optimizer.step()
        for official_parameter, compatibility_parameter in zip(
            official_parameters,
            compatibility_parameters,
            strict=True,
        ):
            torch.testing.assert_close(
                official_parameter,
                compatibility_parameter,
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                official_optimizer.state[official_parameter]["momentum_buffer"],
                compatibility_optimizer.state[compatibility_parameter][
                    "momentum_buffer"
                ],
                atol=0,
                rtol=0,
            )
        for official_group, compatibility_group in zip(
            official_optimizer.param_groups,
            compatibility_optimizer.param_groups,
            strict=True,
        ):
            _assert_exact(official_group["lr"], compatibility_group["lr"])


def test_upstream_identity_and_official_defaults() -> None:
    assert PYTORCH_MUON_UPSTREAM_TAG == "v2.13.0"
    assert PYTORCH_MUON_UPSTREAM_PATH == "torch/optim/_muon.py"
    assert PYTORCH_MUON_UPSTREAM_BLOB_SHA == "2e45e07c4a596fb93f435130c344bb634ee0541c"
    assert PYTORCH_MUON_BACKEND_NAME == REFERENCE_MUON_BACKEND
    assert (EPS, DEFAULT_A, DEFAULT_B, DEFAULT_C, DEFAULT_NS_STEPS) == (
        1e-7,
        3.4445,
        -4.7750,
        2.0315,
        5,
    )


def test_newton_schulz_helper_exact_cpu_parity() -> None:
    official = _require_official_reference()
    generator = torch.Generator().manual_seed(9182)
    for shape in ((1, 1), (3, 5), (5, 3), (8, 8), (32, 8), (8, 32)):
        random_gradient = torch.randn(shape, generator=generator)
        cases = (
            random_gradient,
            torch.zeros_like(random_gradient),
            random_gradient * 1e-12,
            random_gradient * 1e12,
        )
        for gradient in cases:
            for ns_steps in (0, 1, 5):
                expected = official._zeropower_via_newtonschulz(
                    gradient.clone(),
                    MUON_NS_COEFFICIENTS,
                    ns_steps,
                    MUON_EPS,
                )
                actual = _zeropower_via_newtonschulz(
                    gradient.clone(),
                    MUON_NS_COEFFICIENTS,
                    ns_steps,
                    MUON_EPS,
                )
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_newton_schulz_helper_exact_cuda_parity_when_native_bf16() -> None:
    official = _require_official_reference()
    _require_native_bf16_cuda()
    generator = torch.Generator().manual_seed(331)
    for shape in ((8, 8), (32, 8), (8, 32)):
        gradient = torch.randn(shape, generator=generator, device="cpu").cuda()
        for ns_steps in (0, 1, 5):
            expected = official._zeropower_via_newtonschulz(
                gradient.clone(),
                MUON_NS_COEFFICIENTS,
                ns_steps,
                MUON_EPS,
            )
            actual = _zeropower_via_newtonschulz(
                gradient.clone(),
                MUON_NS_COEFFICIENTS,
                ns_steps,
                MUON_EPS,
            )
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_learning_rate_adjustment_exact_parity() -> None:
    official = _require_official_reference()
    for shape in (torch.Size((8, 8)), torch.Size((12, 4)), torch.Size((4, 12))):
        for adjustment in (None, "original", "match_rms_adamw", "unknown"):
            assert _adjust_lr(0.02, adjustment, shape) == official._adjust_lr(
                0.02,
                adjustment,
                shape,
            )


@pytest.mark.parametrize(
    ("name", "overrides"),
    (
        ("production", {}),
        ("no_nesterov", {"nesterov": False}),
        ("no_weight_decay", {"weight_decay": 0.0}),
        ("zero_ns_steps", {"ns_steps": 0}),
        ("one_ns_step", {"ns_steps": 1}),
        ("no_lr_adjustment", {"adjust_lr_fn": None}),
        ("match_rms_adamw", {"adjust_lr_fn": "match_rms_adamw"}),
        ("tensor_lr", {"lr": torch.tensor([MUON_LR])}),
    ),
)
def test_optimizer_ten_step_exact_parity(
    name: str,
    overrides: dict[str, Any],
) -> None:
    del name
    _require_official_reference()
    official_class = torch.optim.Muon
    generator = torch.Generator().manual_seed(87)
    initial = [torch.randn(shape, generator=generator) for shape in _SHAPES]
    official_parameters = _parameters(initial)
    compatibility_parameters = _parameters(initial)
    kwargs = {**_production_kwargs(), **overrides}
    official_optimizer = official_class(
        official_parameters,
        **_clone_kwargs(kwargs),
    )
    compatibility_optimizer = PyTorch213Muon(
        compatibility_parameters,
        **_clone_kwargs(kwargs),
    )
    _assert_optimizer_step_equal(
        official_optimizer,
        compatibility_optimizer,
        official_parameters,
        compatibility_parameters,
        _gradient_sequence(),
    )


def test_optimizer_multiple_parameter_groups_exact_parity() -> None:
    _require_official_reference()
    official_class = torch.optim.Muon
    generator = torch.Generator().manual_seed(106)
    initial = [torch.randn(shape, generator=generator) for shape in _SHAPES]
    official_parameters = _parameters(initial)
    compatibility_parameters = _parameters(initial)

    def groups(parameters: list[nn.Parameter]) -> list[dict[str, Any]]:
        return [
            {
                "params": [parameters[0]],
                "lr": 0.02,
                "momentum": 0.95,
                "nesterov": True,
                "ns_coefficients": MUON_NS_COEFFICIENTS,
                "eps": MUON_EPS,
                "ns_steps": 5,
                "weight_decay": 0.01,
                "adjust_lr_fn": "original",
            },
            {
                "params": parameters[1:],
                "lr": 0.007,
                "momentum": 0.7,
                "nesterov": False,
                "ns_coefficients": MUON_NS_COEFFICIENTS,
                "eps": MUON_EPS,
                "ns_steps": 1,
                "weight_decay": 0.0,
                "adjust_lr_fn": "match_rms_adamw",
            },
        ]

    official_optimizer = official_class(
        groups(official_parameters),
        **_production_kwargs(),
    )
    compatibility_optimizer = PyTorch213Muon(
        groups(compatibility_parameters),
        **_production_kwargs(),
    )
    _assert_optimizer_step_equal(
        official_optimizer,
        compatibility_optimizer,
        official_parameters,
        compatibility_parameters,
        _gradient_sequence(),
    )


def test_real_model_muon_shapes_and_one_update_exact_parity() -> None:
    _require_official_reference()
    official_class = torch.optim.Muon
    model = build_neural_model("context_pooled")
    routed = partition_parameters(model, "hybrid")["muon"]
    shapes = sorted({tuple(parameter.shape) for parameter in routed})
    assert shapes == [(256, 256), (256, 704), (704, 256)]
    generator = torch.Generator().manual_seed(511)
    initial = [torch.randn(shape, generator=generator) for shape in shapes]
    gradients = [[torch.randn(shape, generator=generator) for shape in shapes]]
    official_parameters = _parameters(initial)
    compatibility_parameters = _parameters(initial)
    official_optimizer = official_class(
        official_parameters,
        **_production_kwargs(),
    )
    compatibility_optimizer = PyTorch213Muon(
        compatibility_parameters,
        **_production_kwargs(),
    )
    _assert_optimizer_step_equal(
        official_optimizer,
        compatibility_optimizer,
        official_parameters,
        compatibility_parameters,
        gradients,
    )


def test_grad_none_exact_parity_and_lazy_state() -> None:
    _require_official_reference()
    official_class = torch.optim.Muon
    initial = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    official_parameter = nn.Parameter(initial.clone())
    compatibility_parameter = nn.Parameter(initial.clone())
    official_optimizer = official_class(
        [official_parameter],
        **_production_kwargs(),
    )
    compatibility_optimizer = PyTorch213Muon(
        [compatibility_parameter],
        **_production_kwargs(),
    )
    official_optimizer.step()
    compatibility_optimizer.step()
    torch.testing.assert_close(official_parameter, initial, atol=0, rtol=0)
    torch.testing.assert_close(compatibility_parameter, initial, atol=0, rtol=0)
    assert official_parameter not in official_optimizer.state
    assert compatibility_parameter not in compatibility_optimizer.state
    _assert_exact(
        official_optimizer.state_dict(),
        compatibility_optimizer.state_dict(),
    )


def test_closure_runs_with_grad_and_returns_value_exactly() -> None:
    _require_official_reference()
    official_class = torch.optim.Muon
    initial = torch.arange(15, dtype=torch.float32).reshape(3, 5) / 10
    official_parameter = nn.Parameter(initial.clone())
    compatibility_parameter = nn.Parameter(initial.clone())
    official_optimizer = official_class(
        [official_parameter],
        **_production_kwargs(),
    )
    compatibility_optimizer = PyTorch213Muon(
        [compatibility_parameter],
        **_production_kwargs(),
    )
    official_grad_flags: list[bool] = []
    compatibility_grad_flags: list[bool] = []

    def closure(
        parameter: nn.Parameter,
        grad_flags: list[bool],
    ) -> Callable[[], torch.Tensor]:
        def run() -> torch.Tensor:
            grad_flags.append(torch.is_grad_enabled())
            loss = parameter.square().sum()
            loss.backward()
            return loss

        return run

    official_loss = official_optimizer.step(
        closure(official_parameter, official_grad_flags)
    )
    compatibility_loss = compatibility_optimizer.step(
        closure(compatibility_parameter, compatibility_grad_flags)
    )
    assert official_grad_flags == compatibility_grad_flags == [True]
    torch.testing.assert_close(official_loss, compatibility_loss, atol=0, rtol=0)
    torch.testing.assert_close(
        official_parameter,
        compatibility_parameter,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        official_optimizer.state[official_parameter]["momentum_buffer"],
        compatibility_optimizer.state[compatibility_parameter]["momentum_buffer"],
        atol=0,
        rtol=0,
    )


def _exception_type(call: Callable[[], object]) -> type[BaseException]:
    try:
        call()
    except BaseException as error:
        return type(error)
    raise AssertionError("Expected an exception")


def _step_error(
    optimizer_class: type[torch.optim.Optimizer],
    *,
    dtype: torch.dtype = torch.float32,
    sparse: bool = False,
    **kwargs: Any,
) -> None:
    parameter = nn.Parameter(torch.ones(3, 5, dtype=dtype))
    if sparse:
        indices = torch.tensor([[0, 1], [1, 3]])
        values = torch.tensor([1.0, -2.0])
        parameter.grad = torch.sparse_coo_tensor(
            indices,
            values,
            parameter.shape,
            check_invariants=True,
        )
    else:
        parameter.grad = torch.ones_like(parameter)
    optimizer_class([parameter], **kwargs).step()


def test_error_types_match_official() -> None:
    official = _require_official_reference()
    official_class = torch.optim.Muon
    constructor_cases: tuple[tuple[Callable[[], object], Callable[[], object]], ...] = (
        (
            lambda: official_class([nn.Parameter(torch.ones(3))]),
            lambda: PyTorch213Muon([nn.Parameter(torch.ones(3))]),
        ),
        (
            lambda: official_class([nn.Parameter(torch.ones(3, 5))], lr=-1.0),
            lambda: PyTorch213Muon([nn.Parameter(torch.ones(3, 5))], lr=-1.0),
        ),
        (
            lambda: official_class(
                [nn.Parameter(torch.ones(3, 5))],
                lr=torch.ones(2),
            ),
            lambda: PyTorch213Muon(
                [nn.Parameter(torch.ones(3, 5))],
                lr=torch.ones(2),
            ),
        ),
        (
            lambda: official_class(
                [nn.Parameter(torch.ones(3, 5))],
                momentum=-0.1,
            ),
            lambda: PyTorch213Muon(
                [nn.Parameter(torch.ones(3, 5))],
                momentum=-0.1,
            ),
        ),
        (
            lambda: official_class(
                [nn.Parameter(torch.ones(3, 5))],
                weight_decay=-0.1,
            ),
            lambda: PyTorch213Muon(
                [nn.Parameter(torch.ones(3, 5))],
                weight_decay=-0.1,
            ),
        ),
        (
            lambda: official_class(
                [nn.Parameter(torch.ones(3, 5))],
                adjust_lr_fn="invalid",
            ),
            lambda: PyTorch213Muon(
                [nn.Parameter(torch.ones(3, 5))],
                adjust_lr_fn="invalid",
            ),
        ),
    )
    for official_call, compatibility_call in constructor_cases:
        assert _exception_type(official_call) is _exception_type(compatibility_call)

    step_cases = (
        (
            lambda: _step_error(official_class, dtype=torch.complex64),
            lambda: _step_error(PyTorch213Muon, dtype=torch.complex64),
        ),
        (
            lambda: _step_error(official_class, sparse=True),
            lambda: _step_error(PyTorch213Muon, sparse=True),
        ),
        (
            lambda: _step_error(official_class, ns_steps=100),
            lambda: _step_error(PyTorch213Muon, ns_steps=100),
        ),
        (
            lambda: _step_error(
                official_class,
                ns_coefficients=(1.0, 2.0),
            ),
            lambda: _step_error(
                PyTorch213Muon,
                ns_coefficients=(1.0, 2.0),
            ),
        ),
        (
            lambda: official.muon(
                [],
                [],
                [],
                foreach=True,
                **_production_kwargs(),
                has_complex=False,
            ),
            lambda: muon(
                [],
                [],
                [],
                foreach=True,
                **_production_kwargs(),
                has_complex=False,
            ),
        ),
    )
    for official_call, compatibility_call in step_cases:
        assert _exception_type(official_call) is _exception_type(compatibility_call)


def _run_steps(
    optimizer: torch.optim.Optimizer,
    parameter: nn.Parameter,
    gradients: list[torch.Tensor],
) -> None:
    for gradient in gradients:
        parameter.grad = gradient.clone()
        optimizer.step()


@pytest.mark.parametrize(
    "official_to_compatibility",
    (True, False),
    ids=("official_to_compatibility", "compatibility_to_official"),
)
def test_cross_backend_state_loading_continues_exactly(
    official_to_compatibility: bool,
) -> None:
    _require_official_reference()
    official_class = torch.optim.Muon
    generator = torch.Generator().manual_seed(774)
    initial = torch.randn(5, 3, generator=generator)
    warmup_gradients = [torch.randn(5, 3, generator=generator) for _ in range(4)]
    continuation_gradients = [torch.randn(5, 3, generator=generator) for _ in range(6)]
    donor_class, receiver_class = (
        (official_class, PyTorch213Muon)
        if official_to_compatibility
        else (PyTorch213Muon, official_class)
    )
    donor_parameter = nn.Parameter(initial.clone())
    donor = donor_class([donor_parameter], **_production_kwargs())
    _run_steps(donor, donor_parameter, warmup_gradients)
    receiver_parameter = nn.Parameter(donor_parameter.detach().clone())
    receiver = receiver_class([receiver_parameter], **_production_kwargs())
    receiver.load_state_dict(copy.deepcopy(donor.state_dict()))

    for gradient in continuation_gradients:
        donor_parameter.grad = gradient.clone()
        receiver_parameter.grad = gradient.clone()
        donor.step()
        receiver.step()
        torch.testing.assert_close(
            donor_parameter,
            receiver_parameter,
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            donor.state[donor_parameter]["momentum_buffer"],
            receiver.state[receiver_parameter]["momentum_buffer"],
            atol=0,
            rtol=0,
        )


def test_state_dict_round_trip_is_exact() -> None:
    _require_official_reference()
    official_class = torch.optim.Muon
    generator = torch.Generator().manual_seed(91)
    initial = torch.randn(3, 5, generator=generator)
    official_parameter = nn.Parameter(initial.clone())
    compatibility_parameter = nn.Parameter(initial.clone())
    official_optimizer = official_class(
        [official_parameter],
        **_production_kwargs(),
    )
    compatibility_optimizer = PyTorch213Muon(
        [compatibility_parameter],
        **_production_kwargs(),
    )
    gradients = [torch.randn(3, 5, generator=generator) for _ in range(3)]
    _run_steps(official_optimizer, official_parameter, gradients)
    _run_steps(compatibility_optimizer, compatibility_parameter, gradients)
    _assert_exact(
        official_optimizer.state_dict(),
        compatibility_optimizer.state_dict(),
    )

    restored_parameter = nn.Parameter(compatibility_parameter.detach().clone())
    restored = PyTorch213Muon([restored_parameter], **_production_kwargs())
    restored.load_state_dict(compatibility_optimizer.state_dict())
    _assert_exact(restored.state_dict(), compatibility_optimizer.state_dict())


class _HybridParityModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(11, 4)
        self.norm = nn.RMSNorm(4)
        self.square = MuonLinear(4, 4, bias=False)
        self.tall = MuonLinear(4, 6, bias=False)
        self.wide = MuonLinear(6, 4, bias=False)
        self.bias = nn.Parameter(torch.zeros(4))
        self.output = nn.Linear(4, 2, bias=True)

    def forward(self, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(self.embedding(indices) + values)
        hidden = torch.tanh(self.square(hidden))
        hidden = torch.tanh(self.wide(torch.tanh(self.tall(hidden))))
        return self.output(hidden + self.bias)


def _hybrid_optimizers(
    model: nn.Module,
    muon_class: type[torch.optim.Optimizer],
) -> tuple[
    dict[str, torch.optim.Optimizer],
    dict[str, torch.optim.lr_scheduler.LambdaLR],
]:
    groups = partition_parameters(model, "hybrid")
    optimizers: dict[str, torch.optim.Optimizer] = {
        "muon": muon_class(groups["muon"], **_production_kwargs()),
        "adamw": torch.optim.AdamW(
            [
                {
                    "params": groups["decay"],
                    "weight_decay": ADAMW_WEIGHT_DECAY,
                },
                {
                    "params": groups["no_decay"],
                    "weight_decay": 0.0,
                },
            ],
            lr=ADAMW_LR,
            betas=ADAMW_BETAS,
            eps=ADAMW_EPS,
            fused=True,
        ),
    }
    schedulers, _, _ = build_schedulers(optimizers, EFFECTIVE_BATCH_SIZE)
    return optimizers, schedulers


def test_deterministic_hybrid_trajectory_exact_for_twenty_updates() -> None:
    _require_official_reference()
    official_class = torch.optim.Muon
    original_threads = torch.get_num_threads()
    original_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(244)
        official_model = _HybridParityModel()
        compatibility_model = _HybridParityModel()
        compatibility_model.load_state_dict(official_model.state_dict())
        official_optimizers, official_schedulers = _hybrid_optimizers(
            official_model,
            official_class,
        )
        compatibility_optimizers, compatibility_schedulers = _hybrid_optimizers(
            compatibility_model,
            PyTorch213Muon,
        )
        generator = torch.Generator().manual_seed(7781)
        batches = [
            (
                torch.randint(0, 11, (7,), generator=generator),
                torch.randn(7, 4, generator=generator),
                torch.randn(7, 2, generator=generator),
            )
            for _ in range(20)
        ]
        for indices, values, targets in batches:
            for optimizer in official_optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            for optimizer in compatibility_optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            official_loss = F.mse_loss(
                official_model(indices, values),
                targets,
            )
            compatibility_loss = F.mse_loss(
                compatibility_model(indices, values),
                targets,
            )
            torch.testing.assert_close(
                official_loss,
                compatibility_loss,
                atol=0,
                rtol=0,
            )
            official_loss.backward()
            compatibility_loss.backward()
            official_norm = torch.nn.utils.clip_grad_norm_(
                official_model.parameters(),
                1.0,
            )
            compatibility_norm = torch.nn.utils.clip_grad_norm_(
                compatibility_model.parameters(),
                1.0,
            )
            torch.testing.assert_close(
                official_norm,
                compatibility_norm,
                atol=0,
                rtol=0,
            )
            for optimizer in official_optimizers.values():
                optimizer.step()
            for optimizer in compatibility_optimizers.values():
                optimizer.step()
            for scheduler in official_schedulers.values():
                scheduler.step()
            for scheduler in compatibility_schedulers.values():
                scheduler.step()

            for (_, official_parameter), (_, compatibility_parameter) in zip(
                official_model.named_parameters(),
                compatibility_model.named_parameters(),
                strict=True,
            ):
                torch.testing.assert_close(
                    official_parameter,
                    compatibility_parameter,
                    atol=0,
                    rtol=0,
                )
            _assert_exact(
                official_optimizers["muon"].state_dict(),
                compatibility_optimizers["muon"].state_dict(),
            )
            _assert_exact(
                official_optimizers["adamw"].state_dict(),
                compatibility_optimizers["adamw"].state_dict(),
            )
            for name in official_optimizers:
                for official_group, compatibility_group in zip(
                    official_optimizers[name].param_groups,
                    compatibility_optimizers[name].param_groups,
                    strict=True,
                ):
                    _assert_exact(official_group["lr"], compatibility_group["lr"])
                assert (
                    official_schedulers[name].last_epoch
                    == compatibility_schedulers[name].last_epoch
                )
    finally:
        torch.use_deterministic_algorithms(original_deterministic)
        torch.set_num_threads(original_threads)


class _RoutingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.muon = MuonLinear(3, 4, bias=False)
        self.embedding = nn.Embedding(5, 4)
        self.output = nn.Linear(4, 2, bias=True)


def test_backend_resolution_prefers_official_when_present() -> None:
    _require_official_reference()
    official_class = torch.optim.Muon
    optimizers, _, backend = build_optimizers(_RoutingModel(), "hybrid")
    assert type(optimizers["muon"]) is official_class
    assert backend == OFFICIAL_MUON_BACKEND


def test_backend_resolution_uses_fallback_and_explicit_frozen_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class RecordingMuon(PyTorch213Muon):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured.update(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.delattr(torch.optim, "Muon", raising=False)
    monkeypatch.setattr(optim_module, "PyTorch213Muon", RecordingMuon)
    optimizers, _, backend = build_optimizers(_RoutingModel(), "hybrid")
    assert isinstance(optimizers["muon"], RecordingMuon)
    assert backend == REFERENCE_MUON_BACKEND
    assert captured == _production_kwargs()


def test_official_constructor_failure_propagates_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SentinelError(RuntimeError):
        pass

    class FailingOfficialMuon:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise SentinelError("official constructor failed")

    class ForbiddenFallback:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise AssertionError("fallback must not be constructed")

    monkeypatch.setattr(torch.optim, "Muon", FailingOfficialMuon, raising=False)
    monkeypatch.setattr(optim_module, "PyTorch213Muon", ForbiddenFallback)
    with pytest.raises(SentinelError, match="official constructor failed"):
        build_optimizers(_RoutingModel(), "hybrid")


def test_adamw_build_has_no_muon_backend() -> None:
    optimizers, groups, backend = build_optimizers(_RoutingModel(), "adamw")
    assert optimizers.keys() == {"adamw"}
    assert groups["muon"] == []
    assert backend is None


def test_fallback_subclasses_public_optimizer_and_accepts_production_settings() -> None:
    assert issubclass(PyTorch213Muon, torch.optim.Optimizer)
    parameter = nn.Parameter(torch.ones(3, 5))
    optimizer = PyTorch213Muon([parameter], **_production_kwargs())
    assert optimizer.param_groups[0]["lr"] == MUON_LR


def test_fallback_cpu_step_is_finite_and_scheduler_integrates() -> None:
    parameter = nn.Parameter(torch.arange(15, dtype=torch.float32).reshape(3, 5))
    optimizer = PyTorch213Muon([parameter], **_production_kwargs())
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 1.0 / (step + 1),
    )
    parameter.grad = torch.full_like(parameter, 0.25)
    optimizer.step()
    scheduler.step()
    assert torch.isfinite(parameter).all()
    assert torch.isfinite(optimizer.state[parameter]["momentum_buffer"]).all()
    assert optimizer.param_groups[0]["lr"] == MUON_LR / 2


def test_fallback_state_round_trip_and_grad_none() -> None:
    parameter = nn.Parameter(torch.ones(5, 3))
    optimizer = PyTorch213Muon([parameter], **_production_kwargs())
    untouched = nn.Parameter(torch.full((3, 5), 2.0))
    untouched_optimizer = PyTorch213Muon([untouched], **_production_kwargs())
    untouched_optimizer.step()
    assert untouched not in untouched_optimizer.state

    parameter.grad = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    optimizer.step()
    restored_parameter = nn.Parameter(parameter.detach().clone())
    restored = PyTorch213Muon([restored_parameter], **_production_kwargs())
    restored.load_state_dict(optimizer.state_dict())
    _assert_exact(restored.state_dict(), optimizer.state_dict())


def test_fallback_enforces_errors_without_official_reference() -> None:
    with pytest.raises(ValueError):
        PyTorch213Muon([nn.Parameter(torch.ones(3))])
    with pytest.raises(ValueError):
        PyTorch213Muon([nn.Parameter(torch.ones(3, 5))], lr=-1.0)
    with pytest.raises(ValueError):
        PyTorch213Muon([nn.Parameter(torch.ones(3, 5))], momentum=-1.0)
    with pytest.raises(ValueError):
        PyTorch213Muon([nn.Parameter(torch.ones(3, 5))], weight_decay=-1.0)
    with pytest.raises(ValueError):
        PyTorch213Muon(
            [nn.Parameter(torch.ones(3, 5))],
            adjust_lr_fn="invalid",
        )
    with pytest.raises(ValueError):
        PyTorch213Muon(
            [nn.Parameter(torch.ones(3, 5))],
            lr=torch.ones(2),
        )
    with pytest.raises(RuntimeError):
        muon(
            [],
            [],
            [],
            foreach=True,
            **_production_kwargs(),
            has_complex=False,
        )


def test_fallback_step_rejects_complex_parameter() -> None:
    with pytest.raises(RuntimeError):
        _step_error(PyTorch213Muon, dtype=torch.complex64)


def test_fallback_step_rejects_sparse_gradient() -> None:
    with pytest.raises(RuntimeError):
        _step_error(PyTorch213Muon, sparse=True)


def test_fallback_step_rejects_excessive_newton_schulz_steps() -> None:
    with pytest.raises(ValueError):
        _step_error(PyTorch213Muon, ns_steps=100)


def test_fallback_step_rejects_invalid_coefficient_count() -> None:
    with pytest.raises(ValueError):
        _step_error(PyTorch213Muon, ns_coefficients=(1.0, 2.0))


def test_fallback_source_imports_no_private_pytorch_helpers() -> None:
    source = inspect.getsource(compatibility_module)
    tree = ast.parse(source)
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)
    assert all(
        not module.startswith(("torch._", "torch.optim._"))
        for module in imported_modules
    )
    assert "from torch.optim.optimizer" not in source
    assert "_disable_dynamo_if_unsupported" not in source


def test_local_scalar_helper_matches_float_and_one_element_cpu_tensor() -> None:
    official = _require_official_reference()
    for learning_rate in (0.02, torch.tensor(0.02), torch.tensor([0.02])):
        expected = official._to_scalar(learning_rate)
        actual = compatibility_module._to_scalar(learning_rate)
        _assert_exact(expected, actual)


def test_local_scalar_helper_matches_one_element_cuda_tensor_when_available() -> None:
    official = _require_official_reference()
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    learning_rate = torch.tensor([0.02], device="cuda")
    expected = official._to_scalar(learning_rate)
    actual = compatibility_module._to_scalar(learning_rate)
    _assert_exact(expected, actual)
