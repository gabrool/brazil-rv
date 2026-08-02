from __future__ import annotations

import copy
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch
from torch import nn

import brazil_rv.modeling.engine as engine
import brazil_rv.modeling.train as train
from brazil_rv.modeling.contract import (
    ADAMW_BETAS,
    ADAMW_EPS,
    ADAMW_LR,
    FEATURE_CONTRACT_VERSION,
    ADAMW_WEIGHT_DECAY,
    BASELINE_TCN_SETTINGS,
    GH200_RUNTIME,
    HUBER_DELTA,
    NEURAL_OBJECTIVES,
    SAM_RHOS,
    SOFT_RANK_TEMPERATURES,
    architecture_for_model,
)
from brazil_rv.modeling.engine import (
    checkpoint_payload,
    objective_loss,
    objective_metadata,
    rank_huber_loss,
    run_effective_batch_update,
    sam_metadata,
    soft_spearman_loss,
    train_one_epoch,
)
from brazil_rv.modeling.model import build_neural_model
from brazil_rv.modeling.optim import build_optimizer


BASELINE_TCN_ARCHITECTURE = architecture_for_model("tcn", BASELINE_TCN_SETTINGS)
BASELINE_TCN_CLI = (
    "--tcn-fusion",
    "context_pooled",
    "--tcn-width",
    "128",
    "--tcn-receptive-field",
    "full",
    "--tcn-block",
    "gelu",
)


def _rank_targets(equity_count: int) -> torch.Tensor:
    ranks = torch.arange(equity_count, dtype=torch.float32) + 1
    return (2.0 * ((ranks - 0.5) / equity_count) - 1.0).reshape(1, equity_count, 1)


def test_soft_spearman_direction_constants_and_affine_invariance() -> None:
    targets = _rank_targets(6)
    mask = torch.ones_like(targets, dtype=torch.bool)
    ordered = targets.clone().requires_grad_()
    perfect = soft_spearman_loss(ordered, targets, mask, 0.05)
    reversed_loss = soft_spearman_loss(-ordered, targets, mask, 0.05)
    shifted = soft_spearman_loss(ordered + 37.0, targets, mask, 0.05)
    scaled = soft_spearman_loss(ordered * 9.0, targets, mask, 0.05)
    constant_scores = torch.zeros_like(ordered, requires_grad=True)
    constant = soft_spearman_loss(constant_scores, targets, mask, 0.05)
    constant.backward()

    assert perfect.item() < 1e-5
    assert reversed_loss.item() > 2.0 - 1e-5
    torch.testing.assert_close(shifted, perfect, atol=1e-6, rtol=0)
    torch.testing.assert_close(scaled, perfect, atol=1e-6, rtol=0)
    assert constant.item() == pytest.approx(1.0)
    assert torch.isfinite(constant_scores.grad).all()


def test_soft_spearman_masks_padding_and_excludes_singletons() -> None:
    predictions = torch.tensor(
        [[[0.0], [1.0], [2.0], [1000.0]], [[3.0], [8.0], [0.0], [0.0]]]
    )
    targets = torch.tensor(
        [[[-0.8], [0.0], [0.8], [-999.0]], [[-0.5], [0.5], [0.0], [0.0]]]
    )
    mask = torch.tensor(
        [[[True], [True], [True], [False]], [[True], [False], [False], [False]]]
    )
    active_only = soft_spearman_loss(
        predictions[:1, :3], targets[:1, :3], mask[:1, :3], 0.1
    )
    padded = soft_spearman_loss(predictions, targets, mask, 0.1)
    torch.testing.assert_close(padded, active_only, atol=0, rtol=0)


def test_groups_are_independent_and_equally_weighted() -> None:
    predictions = torch.zeros(2, 6, 3)
    targets = torch.zeros_like(predictions)
    mask = torch.zeros_like(predictions, dtype=torch.bool)
    targets[0, :2, 0] = _rank_targets(2)[0, :, 0]
    predictions[0, :2, 0] = targets[0, :2, 0]
    mask[0, :2, 0] = True
    targets[1, :, 2] = _rank_targets(6)[0, :, 0]
    predictions[1, :, 2] = -targets[1, :, 2]
    mask[1, :, 2] = True

    loss = soft_spearman_loss(predictions, targets, mask, 0.05)
    assert loss.item() == pytest.approx(1.0, abs=1e-5)

    predictions[0, :2, 1] = -targets[0, :2, 0]
    targets[0, :2, 1] = targets[0, :2, 0]
    mask[0, :2, 1] = True
    independent_horizons = soft_spearman_loss(predictions, targets, mask, 0.05)
    assert independent_horizons.item() == pytest.approx(4.0 / 3.0, abs=1e-5)


def test_temperature_metadata_and_fp32_under_bf16_autocast() -> None:
    assert tuple(SOFT_RANK_TEMPERATURES) == (0.05, 0.10, 0.20, 0.50)
    assert objective_metadata("soft_spearman", 0.1) == {
        "name": "soft_spearman",
        "temperature": 0.1,
        "score_standardization": "masked_cross_sectional",
        "soft_rank": "pairwise_sigmoid",
        "aggregation": "equal_valid_cross_section_horizon",
        "reported_validation_metric": "hard_spearman",
    }
    with pytest.raises(ValueError):
        objective_metadata("soft_spearman", 0.3)
    targets = _rank_targets(4).bfloat16()
    predictions = targets.clone().requires_grad_()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = soft_spearman_loss(
            predictions, targets, torch.ones_like(targets, dtype=torch.bool), 0.1
        )
    loss.backward()
    assert loss.dtype == torch.float32
    assert torch.isfinite(predictions.grad).all()


def _historical_rank_huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    label_mask: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
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
    if bool(valid_samples.any()):
        return sample_loss[valid_samples].mean()
    return predictions.float().sum() * 0.0


def test_rank_huber_exact_historical_numerics_masking_and_weighting() -> None:
    assert HUBER_DELTA == 1.0
    predictions = torch.tensor(
        [
            [[-2.0, 0.2, 9.0], [0.4, -0.8, -7.0], [3.0, 0.0, 5.0]],
            [[0.1, 1.5, -2.0], [0.9, -1.5, 3.0], [-0.3, 0.5, 4.0]],
        ],
        requires_grad=True,
    )
    targets = torch.tensor(
        [
            [[-0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 0.0]],
            [[-2.0 / 3.0, -0.5, -0.5], [0.0, 0.5, 0.5], [2.0 / 3.0, 0.0, 0.0]],
        ]
    )
    mask = torch.tensor(
        [
            [[True, True, False], [True, True, False], [False, False, False]],
            [[True, True, True], [True, True, True], [True, False, False]],
        ]
    )
    expected = _historical_rank_huber_loss(predictions, targets, mask)
    actual = rank_huber_loss(predictions, targets, mask)
    expected_gradient = torch.autograd.grad(expected, predictions, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(actual, predictions)[0]

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=0, rtol=0)
    masked_changed = predictions.detach().clone()
    masked_changed[~mask] = 1_000_000.0
    torch.testing.assert_close(
        rank_huber_loss(masked_changed, targets, mask), actual, atol=0, rtol=0
    )


def test_objective_dispatch_and_rank_huber_metadata() -> None:
    assert tuple(NEURAL_OBJECTIVES) == ("soft_spearman", "rank_huber")
    assert objective_metadata("rank_huber", None) == {
        "name": "rank_huber",
        "temperature": None,
        "delta": 1.0,
        "target": "centered_cross_sectional_midrank",
        "aggregation": "equal_valid_sample_then_horizon_then_equity",
        "reported_validation_metric": "hard_spearman",
    }
    with pytest.raises(ValueError, match="does not accept"):
        objective_metadata("rank_huber", 0.1)

    predictions = torch.tensor([[[-0.5], [0.5]]])
    targets = torch.tensor([[[-0.75], [0.75]]])
    mask = torch.ones_like(targets, dtype=torch.bool)
    torch.testing.assert_close(
        objective_loss(predictions, targets, mask, "rank_huber", None),
        rank_huber_loss(predictions, targets, mask),
        atol=0,
        rtol=0,
    )


def test_soft_spearman_eager_compiled_predictions_loss_and_gradients() -> None:
    predictions = torch.randn(2, 7, 3, requires_grad=True)
    targets = torch.stack([_rank_targets(7)[0, :, 0]] * 6).reshape(2, 3, 7)
    targets = targets.transpose(1, 2)
    mask = torch.ones_like(targets, dtype=torch.bool)
    compiled_loss = torch.compile(
        soft_spearman_loss, backend="eager", fullgraph=True, dynamic=False
    )

    eager = soft_spearman_loss(predictions, targets, mask, 0.2)
    eager_gradient = torch.autograd.grad(eager, predictions, retain_graph=True)[0]
    compiled = compiled_loss(predictions, targets, mask, 0.2)
    compiled_gradient = torch.autograd.grad(compiled, predictions)[0]
    torch.testing.assert_close(compiled, eager, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(compiled_gradient, eager_gradient, atol=1e-6, rtol=1e-6)


class _CrossSectionModel(nn.Module):
    def __init__(self, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(2, 3, bias=False)
        self.dropout_masks: list[torch.Tensor] = []

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        dropped = self.dropout(features)
        self.dropout_masks.append(dropped.eq(0).detach().clone())
        return self.linear(dropped)


class _PureCrossSectionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 3, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


class _ManyParameterModel(nn.Module):
    def __init__(self, parameter_count: int = 12) -> None:
        super().__init__()
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.randn(2, 3) * 0.1) for _ in range(parameter_count)]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return sum(features @ weight for weight in self.weights)


class _CompiledCudaDropoutModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(2, 3, bias=False)
        self.register_buffer(
            "dropout_trace",
            torch.zeros(16, 7, 2, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "trace_index", torch.zeros((), dtype=torch.long), persistent=False
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        dropped = self.dropout(features)
        mask = dropped.eq(0).detach()
        indices = self.trace_index.reshape(1, 1, 1).expand_as(mask)
        self.dropout_trace.scatter_(0, indices, mask)
        self.trace_index.add_(1)
        return self.linear(dropped)


class _FiniteForwardNanBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, inputs: torch.Tensor) -> torch.Tensor:
        del ctx
        return inputs.clone()

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> torch.Tensor:
        del ctx
        return torch.full_like(gradient, float("nan"))


class _CountingScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1

    def state_dict(self) -> dict[str, int]:
        return {"steps": self.steps}


def _microbatches() -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(19)
    batches = []
    for index in range(8):
        equity_count = 3 + index % 4
        features = torch.randn(1, 7, 2, generator=generator)
        targets = torch.zeros(1, 7, 3)
        mask = torch.zeros_like(targets, dtype=torch.bool)
        for horizon in range(1 + index % 3):
            targets[0, :equity_count, horizon] = _rank_targets(equity_count)[0, :, 0]
            mask[0, :equity_count, horizon] = True
        batches.append({"features": features, "targets": targets, "label_mask": mask})
    return batches


class _CountingAdamW(torch.optim.AdamW):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.step_calls = 0

    def step(self, closure: object = None) -> object:
        self.step_calls += 1
        return super().step(closure)


def _adamw(model: nn.Module) -> _CountingAdamW:
    return _CountingAdamW(
        model.parameters(),
        lr=ADAMW_LR,
        betas=ADAMW_BETAS,
        eps=ADAMW_EPS,
        weight_decay=ADAMW_WEIGHT_DECAY,
        fused=False,
    )


@pytest.fixture
def cpu_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine, "_to_cuda", lambda batch: batch)
    monkeypatch.setattr(
        engine, "_predict", lambda model, batch: model(batch["features"])
    )
    monkeypatch.setattr(engine.torch, "autocast", lambda **_: nullcontext())


@pytest.mark.parametrize(
    ("objective", "temperature", "optimizer_variant", "sam_rho"),
    [
        ("soft_spearman", 0.1, "adamw", None),
        ("soft_spearman", 0.1, "sam_adamw", 0.025),
        ("rank_huber", None, "adamw", None),
        ("rank_huber", None, "sam_adamw", 0.025),
    ],
)
def test_eager_and_compiled_updates_match(
    cpu_engine: None,
    objective: str,
    temperature: float | None,
    optimizer_variant: str,
    sam_rho: float | None,
) -> None:
    eager_model = _PureCrossSectionModel()
    compiled_base = copy.deepcopy(eager_model)
    compiled_model = torch.compile(
        compiled_base, backend="eager", fullgraph=True, dynamic=False
    )
    eager_optimizer = _adamw(eager_model)
    compiled_optimizer = _adamw(compiled_base)

    eager_update = run_effective_batch_update(
        eager_model,
        _microbatches(),
        eager_optimizer,
        None,
        GH200_RUNTIME,
        optimizer_variant,
        objective,
        temperature,
        sam_rho,
    )
    compiled_update = run_effective_batch_update(
        compiled_model,
        _microbatches(),
        compiled_optimizer,
        None,
        GH200_RUNTIME,
        optimizer_variant,
        objective,
        temperature,
        sam_rho,
    )
    assert compiled_update["backward_passes"] == eager_update["backward_passes"]
    for eager, compiled in zip(
        eager_model.parameters(), compiled_base.parameters(), strict=True
    ):
        torch.testing.assert_close(compiled, eager, atol=2e-7, rtol=2e-7)


@pytest.mark.parametrize("rho", SAM_RHOS)
@pytest.mark.parametrize(
    ("objective", "temperature"),
    [("soft_spearman", 0.1), ("rank_huber", None)],
)
def test_all_supported_sam_rhos_work_with_either_objective(
    cpu_engine: None,
    objective: str,
    temperature: float | None,
    rho: float,
) -> None:
    model = _CrossSectionModel()
    optimizer = _adamw(model)
    update = run_effective_batch_update(
        model,
        _microbatches(),
        optimizer,
        None,
        GH200_RUNTIME,
        "sam_adamw",
        objective,
        temperature,
        rho,
    )
    assert update["backward_passes"] == 16
    assert optimizer.step_calls == 1


@pytest.mark.parametrize(
    ("optimizer_variant", "sam_rho", "expected_flag_arities", "float_arity"),
    [
        ("adamw", None, [2, 1, 1], 2),
        ("sam_adamw", 0.025, [2, 1, 2, 2, 2, 1, 1, 1], 6),
    ],
)
def test_update_synchronization_boundaries_do_not_scale_with_parameters(
    cpu_engine: None,
    monkeypatch: pytest.MonkeyPatch,
    optimizer_variant: str,
    sam_rho: float | None,
    expected_flag_arities: list[int],
    float_arity: int,
) -> None:
    model = _ManyParameterModel()
    optimizer = _adamw(model)
    flag_arities: list[int] = []
    float_arities: list[int] = []
    original_host_flags = engine._host_flags
    original_host_floats = engine._host_floats

    def observed_flags(*flags: torch.Tensor) -> tuple[bool, ...]:
        assert all(flag.shape == torch.Size([]) for flag in flags)
        flag_arities.append(len(flags))
        return original_host_flags(*flags)

    def observed_floats(*values: torch.Tensor) -> tuple[float, ...]:
        assert all(value.shape == torch.Size([]) for value in values)
        float_arities.append(len(values))
        return original_host_floats(*values)

    monkeypatch.setattr(engine, "_host_flags", observed_flags)
    monkeypatch.setattr(engine, "_host_floats", observed_floats)
    run_effective_batch_update(
        model,
        _microbatches(),
        optimizer,
        None,
        GH200_RUNTIME,
        optimizer_variant,
        "soft_spearman",
        0.1,
        sam_rho,
    )

    assert flag_arities == expected_flag_arities
    assert float_arities == [float_arity]
    reference = next(model.parameters())
    assert engine._tensors_finite(model.parameters(), reference).shape == torch.Size([])
    snapshots = [parameter.detach().clone() for parameter in model.parameters()]
    assert engine._tensor_pairs_equal(
        zip(model.parameters(), snapshots, strict=True), reference
    ).shape == torch.Size([])


def test_eight_microbatches_match_concatenated_group_normalization(
    cpu_engine: None,
) -> None:
    batches = _microbatches()
    accumulated_model = _CrossSectionModel()
    reference_model = copy.deepcopy(accumulated_model)
    accumulated_optimizer = _adamw(accumulated_model)
    reference_optimizer = _adamw(reference_model)
    scheduler = _CountingScheduler()

    update = run_effective_batch_update(
        accumulated_model,
        batches,
        accumulated_optimizer,
        scheduler,
        GH200_RUNTIME,
        "adamw",
        "soft_spearman",
        0.1,
        None,
    )
    features = torch.cat([batch["features"] for batch in batches])
    targets = torch.cat([batch["targets"] for batch in batches])
    mask = torch.cat([batch["label_mask"] for batch in batches])
    reference_loss = soft_spearman_loss(reference_model(features), targets, mask, 0.1)
    reference_loss.backward()
    torch.nn.utils.clip_grad_norm_(reference_model.parameters(), 1.0)
    reference_optimizer.step()

    assert update["backward_passes"] == 8
    assert accumulated_optimizer.step_calls == 1
    assert scheduler.steps == 1
    for actual, expected in zip(
        accumulated_model.parameters(), reference_model.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-7)


def test_rank_huber_accumulated_batch_matches_concatenated_batch(
    cpu_engine: None,
) -> None:
    batches = _microbatches()
    accumulated_model = _CrossSectionModel()
    reference_model = copy.deepcopy(accumulated_model)
    accumulated_optimizer = _adamw(accumulated_model)
    reference_optimizer = _adamw(reference_model)
    scheduler = _CountingScheduler()

    update = run_effective_batch_update(
        accumulated_model,
        batches,
        accumulated_optimizer,
        scheduler,
        GH200_RUNTIME,
        "adamw",
        "rank_huber",
        None,
        None,
    )
    features = torch.cat([batch["features"] for batch in batches])
    targets = torch.cat([batch["targets"] for batch in batches])
    mask = torch.cat([batch["label_mask"] for batch in batches])
    reference_loss = rank_huber_loss(reference_model(features), targets, mask)
    reference_loss.backward()
    torch.nn.utils.clip_grad_norm_(reference_model.parameters(), 1.0)
    reference_optimizer.step()

    assert update["loss_count"] == len(batches)
    assert update["backward_passes"] == 8
    assert accumulated_optimizer.step_calls == 1
    assert scheduler.steps == 1
    for actual, expected in zip(
        accumulated_model.parameters(), reference_model.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-7)


def test_rank_huber_selected_for_both_sam_passes(
    cpu_engine: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected: list[tuple[str, float | None]] = []
    original = engine._objective_loss_sum

    def observed(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        label_mask: torch.Tensor,
        objective: str,
        temperature: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected.append((objective, temperature))
        return original(predictions, targets, label_mask, objective, temperature)

    monkeypatch.setattr(engine, "_objective_loss_sum", observed)
    model = _CrossSectionModel()
    optimizer = _adamw(model)
    scheduler = _CountingScheduler()
    update = run_effective_batch_update(
        model,
        _microbatches(),
        optimizer,
        scheduler,
        GH200_RUNTIME,
        "sam_adamw",
        "rank_huber",
        None,
        0.025,
    )

    assert selected == [("rank_huber", None)] * 16
    assert update["backward_passes"] == 16
    assert optimizer.step_calls == 1
    assert scheduler.steps == 1


@pytest.mark.parametrize(
    ("objective", "temperature"),
    [("soft_spearman", 0.1), ("rank_huber", None)],
)
def test_incomplete_group_fails_before_forward(
    cpu_engine: None, objective: str, temperature: float | None
) -> None:
    model = _CrossSectionModel()
    optimizer = _adamw(model)
    with pytest.raises(ValueError, match="exactly 8"):
        run_effective_batch_update(
            model,
            _microbatches()[:-1],
            optimizer,
            None,
            GH200_RUNTIME,
            "adamw",
            objective,
            temperature,
            None,
        )
    assert not model.dropout_masks
    assert optimizer.step_calls == 0


def test_incomplete_training_epoch_fails_before_forward_or_update(
    cpu_engine: None,
) -> None:
    model = _CrossSectionModel()
    optimizer = _adamw(model)
    scheduler = _CountingScheduler()
    with pytest.raises(ValueError, match="end inside an effective batch"):
        train_one_epoch(
            model,
            _microbatches()[:-1],
            optimizer,
            scheduler,
            GH200_RUNTIME,
            "sam_adamw",
            "rank_huber",
            None,
            0.025,
        )
    assert not model.dropout_masks
    assert optimizer.step_calls == 0
    assert scheduler.steps == 0


def test_training_epoch_updates_once_per_complete_accumulated_group(
    cpu_engine: None,
) -> None:
    model = _CrossSectionModel()
    optimizer = _adamw(model)
    scheduler = _CountingScheduler()
    result = train_one_epoch(
        model,
        [*_microbatches(), *_microbatches()],
        optimizer,
        scheduler,
        GH200_RUNTIME,
        "sam_adamw",
        "rank_huber",
        None,
        0.025,
    )
    assert result["optimizer_steps"] == 2
    assert result["backward_passes"] == 32
    assert optimizer.step_calls == 2
    assert scheduler.steps == 2


def test_sam_replays_same_batches_rng_and_counts(cpu_engine: None) -> None:
    torch.manual_seed(29)
    model = _CrossSectionModel(dropout=0.5).train()
    optimizer = _adamw(model)
    scheduler = _CountingScheduler()
    backward_calls = 0

    def count_backward(gradient: torch.Tensor) -> torch.Tensor:
        nonlocal backward_calls
        backward_calls += 1
        return gradient

    model.linear.weight.register_hook(count_backward)
    update = run_effective_batch_update(
        model,
        _microbatches(),
        optimizer,
        scheduler,
        GH200_RUNTIME,
        "sam_adamw",
        "soft_spearman",
        0.1,
        0.025,
    )
    assert update["backward_passes"] == 16
    assert update["rng_replay_exact"] is True
    assert optimizer.step_calls == 1
    assert scheduler.steps == 1
    assert len(model.dropout_masks) == 16
    assert backward_calls == 16
    for first, second in zip(
        model.dropout_masks[:8], model.dropout_masks[8:], strict=True
    ):
        torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_compiled_cuda_sam_replays_identical_dropout_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("native CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("native CUDA BF16 is unavailable")
    capability = torch.cuda.get_device_capability()
    if capability[0] < 8:
        pytest.skip(
            f"native CUDA BF16 compilation is unavailable at capability {capability}"
        )

    monkeypatch.setattr(
        engine,
        "_to_cuda",
        lambda batch: {
            key: value.to("cuda", non_blocking=False) for key, value in batch.items()
        },
    )
    monkeypatch.setattr(
        engine, "_predict", lambda model, batch: model(batch["features"])
    )
    model = _CompiledCudaDropoutModel().to("cuda").train()
    engine.compile_model(model, GH200_RUNTIME)
    warmup = torch.randn(1, 7, 2, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        model(warmup)
    torch.cuda.synchronize()
    model.dropout_trace.zero_()
    model.trace_index.zero_()
    torch.manual_seed(29)
    torch.cuda.manual_seed_all(29)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ADAMW_LR,
        betas=ADAMW_BETAS,
        eps=ADAMW_EPS,
        weight_decay=ADAMW_WEIGHT_DECAY,
        fused=True,
    )

    update = run_effective_batch_update(
        model,
        _microbatches(),
        optimizer,
        None,
        GH200_RUNTIME,
        "sam_adamw",
        "soft_spearman",
        0.1,
        0.025,
    )

    assert update["rng_replay_exact"] is True
    assert int(model.trace_index) == 16
    torch.testing.assert_close(
        model.dropout_trace[:8], model.dropout_trace[8:], atol=0, rtol=0
    )


def test_sam_matches_reference_l2_adamw_update(cpu_engine: None) -> None:
    batches = _microbatches()
    actual_model = _CrossSectionModel()
    reference_model = copy.deepcopy(actual_model)
    actual_optimizer = _adamw(actual_model)
    reference_optimizer = _adamw(reference_model)
    rho = 0.025

    run_effective_batch_update(
        actual_model,
        batches,
        actual_optimizer,
        None,
        GH200_RUNTIME,
        "sam_adamw",
        "soft_spearman",
        0.1,
        rho,
    )

    features = torch.cat([batch["features"] for batch in batches])
    targets = torch.cat([batch["targets"] for batch in batches])
    mask = torch.cat([batch["label_mask"] for batch in batches])
    first = soft_spearman_loss(reference_model(features), targets, mask, 0.1)
    first.backward()
    norm = torch.sqrt(
        torch.stack(
            [
                parameter.grad.float().square().sum()
                for parameter in reference_model.parameters()
            ]
        ).sum()
    )
    originals = [
        parameter.detach().clone() for parameter in reference_model.parameters()
    ]
    with torch.no_grad():
        for parameter in reference_model.parameters():
            parameter.add_(parameter.grad.float() * (rho / (norm + 1e-12)))
    reference_optimizer.zero_grad(set_to_none=True)
    second = soft_spearman_loss(reference_model(features), targets, mask, 0.1)
    second.backward()
    with torch.no_grad():
        for parameter, original in zip(
            reference_model.parameters(), originals, strict=True
        ):
            parameter.copy_(original)
    torch.nn.utils.clip_grad_norm_(reference_model.parameters(), 1.0)
    reference_optimizer.step()

    for actual, expected in zip(
        actual_model.parameters(), reference_model.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-7)


def test_sam_restores_parameters_after_second_pass_exception(
    cpu_engine: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _CrossSectionModel()
    optimizer = _adamw(model)
    scheduler = _CountingScheduler()
    initial = [parameter.detach().clone() for parameter in model.parameters()]
    calls = 0

    def failing_predict(
        module: nn.Module, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == 9:
            raise RuntimeError("injected")
        return module(batch["features"])

    monkeypatch.setattr(engine, "_predict", failing_predict)
    with pytest.raises(RuntimeError, match="injected"):
        run_effective_batch_update(
            model,
            _microbatches(),
            optimizer,
            scheduler,
            GH200_RUNTIME,
            "sam_adamw",
            "soft_spearman",
            0.1,
            0.025,
        )
    for parameter, expected in zip(model.parameters(), initial, strict=True):
        torch.testing.assert_close(parameter, expected, atol=0, rtol=0)
    assert optimizer.step_calls == 0
    assert scheduler.steps == 0


def test_sam_restores_parameters_after_first_pass_exception(
    cpu_engine: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _CrossSectionModel()
    optimizer = _adamw(model)
    scheduler = _CountingScheduler()
    initial = [parameter.detach().clone() for parameter in model.parameters()]

    def failing_predict(
        module: nn.Module, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        module(batch["features"])
        raise RuntimeError("first-pass injected")

    monkeypatch.setattr(engine, "_predict", failing_predict)
    with pytest.raises(RuntimeError, match="first-pass injected"):
        run_effective_batch_update(
            model,
            _microbatches(),
            optimizer,
            scheduler,
            GH200_RUNTIME,
            "sam_adamw",
            "soft_spearman",
            0.1,
            0.025,
        )
    for parameter, expected in zip(model.parameters(), initial, strict=True):
        torch.testing.assert_close(parameter, expected, atol=0, rtol=0)
    assert optimizer.step_calls == 0
    assert scheduler.steps == 0


def test_sam_restores_parameters_when_perturbation_construction_fails(
    cpu_engine: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _CrossSectionModel()
    optimizer = _adamw(model)
    scheduler = _CountingScheduler()
    initial = [parameter.detach().clone() for parameter in model.parameters()]

    def failing_perturbation(
        parameters: tuple[nn.Parameter, ...], scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del scale
        with torch.no_grad():
            parameters[0].add_(1.0)
        raise RuntimeError("perturbation injected")

    monkeypatch.setattr(engine, "_apply_sam_perturbation", failing_perturbation)
    with pytest.raises(RuntimeError, match="perturbation injected"):
        run_effective_batch_update(
            model,
            _microbatches(),
            optimizer,
            scheduler,
            GH200_RUNTIME,
            "sam_adamw",
            "soft_spearman",
            0.1,
            0.025,
        )
    for parameter, expected in zip(model.parameters(), initial, strict=True):
        torch.testing.assert_close(parameter, expected, atol=0, rtol=0)
    assert optimizer.step_calls == 0
    assert scheduler.steps == 0


def test_sam_rejects_nonfinite_perturbation_and_restores_parameters(
    cpu_engine: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _CrossSectionModel()
    optimizer = _adamw(model)
    scheduler = _CountingScheduler()
    initial = [parameter.detach().clone() for parameter in model.parameters()]

    def nonfinite_perturbation(
        parameters: tuple[nn.Parameter, ...], scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del scale
        with torch.no_grad():
            parameters[0].add_(1.0)
        reference = parameters[0]
        return (
            reference.new_zeros((), dtype=torch.float32),
            torch.zeros((), dtype=torch.bool, device=reference.device),
        )

    monkeypatch.setattr(engine, "_apply_sam_perturbation", nonfinite_perturbation)
    with pytest.raises(FloatingPointError, match="perturbation is non-finite"):
        run_effective_batch_update(
            model,
            _microbatches(),
            optimizer,
            scheduler,
            GH200_RUNTIME,
            "sam_adamw",
            "soft_spearman",
            0.1,
            0.025,
        )
    for parameter, expected in zip(model.parameters(), initial, strict=True):
        torch.testing.assert_close(parameter, expected, atol=0, rtol=0)
    assert optimizer.step_calls == 0
    assert scheduler.steps == 0


def test_sam_rejects_nonfinite_loss_and_gradients_without_update(
    cpu_engine: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _CrossSectionModel()
    optimizer = _adamw(model)
    scheduler = _CountingScheduler()
    initial = [parameter.detach().clone() for parameter in model.parameters()]
    monkeypatch.setattr(
        engine,
        "_predict",
        lambda module, batch: module(batch["features"]) * float("nan"),
    )

    with pytest.raises(FloatingPointError, match="loss"):
        run_effective_batch_update(
            model,
            _microbatches(),
            optimizer,
            scheduler,
            GH200_RUNTIME,
            "sam_adamw",
            "soft_spearman",
            0.1,
            0.025,
        )
    for parameter, expected in zip(model.parameters(), initial, strict=True):
        torch.testing.assert_close(parameter, expected, atol=0, rtol=0)
    assert optimizer.step_calls == 0
    assert scheduler.steps == 0


def test_sam_rejects_nonfinite_gradients_with_finite_loss_without_update(
    cpu_engine: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _CrossSectionModel()
    optimizer = _adamw(model)
    scheduler = _CountingScheduler()
    initial = [parameter.detach().clone() for parameter in model.parameters()]

    monkeypatch.setattr(
        engine,
        "_predict",
        lambda module, batch: _FiniteForwardNanBackward.apply(
            module(batch["features"])
        ),
    )
    with pytest.raises(FloatingPointError, match="gradients"):
        run_effective_batch_update(
            model,
            _microbatches(),
            optimizer,
            scheduler,
            GH200_RUNTIME,
            "sam_adamw",
            "soft_spearman",
            0.1,
            0.025,
        )
    for parameter, expected in zip(model.parameters(), initial, strict=True):
        torch.testing.assert_close(parameter, expected, atol=0, rtol=0)
    assert optimizer.step_calls == 0
    assert scheduler.steps == 0


def test_sam_rejects_nonfinite_post_update_parameters_and_restores_snapshot(
    cpu_engine: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _CrossSectionModel()
    optimizer = _adamw(model)
    scheduler = _CountingScheduler()
    initial = [parameter.detach().clone() for parameter in model.parameters()]

    def nonfinite_step(closure: object = None) -> None:
        del closure
        optimizer.step_calls += 1
        with torch.no_grad():
            next(model.parameters()).fill_(float("inf"))

    monkeypatch.setattr(optimizer, "step", nonfinite_step)
    with pytest.raises(FloatingPointError, match="non-finite parameters"):
        run_effective_batch_update(
            model,
            _microbatches(),
            optimizer,
            scheduler,
            GH200_RUNTIME,
            "sam_adamw",
            "soft_spearman",
            0.1,
            0.025,
        )
    for parameter, expected in zip(model.parameters(), initial, strict=True):
        torch.testing.assert_close(parameter, expected, atol=0, rtol=0)
    assert optimizer.step_calls == 1
    assert scheduler.steps == 0


def test_rho_grid_metadata_cli_and_run_names() -> None:
    assert tuple(SAM_RHOS) == (0.025, 0.050, 0.075, 0.100, 0.125)
    assert sam_metadata("adamw", None) is None
    for rho in SAM_RHOS:
        assert sam_metadata("sam_adamw", rho)["rho"] == rho
    for invalid in (0.0, -0.01, 0.020, 0.126):
        with pytest.raises(ValueError):
            sam_metadata("sam_adamw", invalid)
    with pytest.raises(ValueError):
        sam_metadata("adamw", 0.025)

    adamw = train.parse_args(
        [
            "--model",
            "tcn",
            *BASELINE_TCN_CLI,
            "--optimizer",
            "adamw",
            "--soft-rank-temperature",
            "0.10",
            "--seed",
            "11",
        ]
    )
    assert adamw.objective == "soft_spearman"
    assert adamw.sam_rho is None
    soft_anchor = train.parse_args(
        [
            "--model",
            "tcn",
            *BASELINE_TCN_CLI,
            "--optimizer",
            "sam_adamw",
            "--objective",
            "soft_spearman",
            "--soft-rank-temperature",
            "0.50",
            "--sam-rho",
            "0.125",
            "--seed",
            "11",
        ]
    )
    assert soft_anchor.objective == "soft_spearman"
    assert soft_anchor.temperature == 0.50
    assert soft_anchor.sam_rho == 0.125
    rank_huber = train.parse_args(
        [
            "--model",
            "tcn",
            *BASELINE_TCN_CLI,
            "--optimizer",
            "sam_adamw",
            "--objective",
            "rank_huber",
            "--sam-rho",
            "0.025",
            "--seed",
            "11",
        ]
    )
    assert rank_huber.objective == "rank_huber"
    assert rank_huber.temperature is None
    assert rank_huber.sam_rho == 0.025
    with pytest.raises(SystemExit):
        train.parse_args(
            [
                "--model",
                "tcn",
                *BASELINE_TCN_CLI,
                "--optimizer",
                "adamw",
                "--objective",
                "rank_huber",
                "--soft-rank-temperature",
                "0.10",
                "--seed",
                "11",
            ]
        )

    created = datetime(2026, 1, 2, 3, 4, 5, 6789, tzinfo=timezone.utc)
    assert (
        train._run_directory_name(
            "tcn",
            BASELINE_TCN_SETTINGS,
            "adamw",
            "soft_spearman",
            0.1,
            None,
            "enabled",
            11,
            created,
        )
        == "tcn_context_pooled_w128_rffull_bgelu_soft_spearman_adamw_tau0p10_"
        "global-enabled_seed11_20260102T030405006789Z"
    )
    assert (
        train._run_directory_name(
            "tcn",
            BASELINE_TCN_SETTINGS,
            "sam_adamw",
            "soft_spearman",
            0.5,
            0.125,
            "enabled",
            11,
            created,
        )
        == "tcn_context_pooled_w128_rffull_bgelu_soft_spearman_sam_adamw_"
        "rho0p125_tau0p50_global-enabled_seed11_20260102T030405006789Z"
    )
    assert (
        train._run_directory_name(
            "tcn",
            BASELINE_TCN_SETTINGS,
            "sam_adamw",
            "rank_huber",
            None,
            0.025,
            "enabled",
            11,
            created,
        )
        == "tcn_context_pooled_w128_rffull_bgelu_rank_huber_sam_adamw_"
        "rho0p025_global-enabled_seed11_20260102T030405006789Z"
    )


@pytest.mark.parametrize(
    ("objective", "temperature"),
    [("soft_spearman", 0.1), ("rank_huber", None)],
)
def test_checkpoint_round_trip_contains_resume_boundary_state(
    tmp_path: Path, objective: str, temperature: float | None
) -> None:
    model = build_neural_model("tcn", BASELINE_TCN_ARCHITECTURE)
    optimizer, _ = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 1e-3)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    feature_manifest = {
        "contract_version": FEATURE_CONTRACT_VERSION,
        "global_context": {
            "source_hashes": {"ES": "source-sha256"},
            "normalized_store_hashes": {"ES": "store-sha256"},
        },
    }
    payload = checkpoint_payload(
        model,
        optimizer,
        scheduler,
        "tcn",
        BASELINE_TCN_ARCHITECTURE,
        BASELINE_TCN_SETTINGS,
        "sam_adamw",
        objective,
        temperature,
        0.025,
        11,
        3,
        0.12,
        tmp_path,
        "enabled",
        feature_manifest,
        "test-sha",
    )
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    restored = build_neural_model("tcn", BASELINE_TCN_ARCHITECTURE)
    restored_optimizer, _ = build_optimizer(restored)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer, lambda _: 1.0
    )
    restored.load_state_dict(loaded["model_state_dict"])
    restored_optimizer.load_state_dict(loaded["optimizer_state_dict"])
    restored_scheduler.load_state_dict(loaded["scheduler_state_dict"])
    assert loaded["objective"] == objective_metadata(objective, temperature)
    assert loaded["sam"] == sam_metadata("sam_adamw", 0.025)
    expected_optimizer = loaded["optimizer_state_dict"]
    actual_optimizer = restored_optimizer.state_dict()
    assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
    assert actual_optimizer["state"].keys() == expected_optimizer["state"].keys()
    for parameter_id, expected_state in expected_optimizer["state"].items():
        actual_state = actual_optimizer["state"][parameter_id]
        assert actual_state.keys() == expected_state.keys()
        for field, expected in expected_state.items():
            actual = actual_state[field]
            if isinstance(expected, torch.Tensor):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            else:
                assert actual == expected
    assert restored_scheduler.state_dict() == loaded["scheduler_state_dict"]
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
