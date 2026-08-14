from __future__ import annotations

import warnings
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch._inductor.exc import InductorError

from brazil_rv.modeling import benchmark_runtime, train as train_module
from brazil_rv.modeling.benchmark_runtime import parse_args as parse_benchmark_args
from brazil_rv.modeling.contract import (
    ABSOLUTE_PATCH_COUNT,
    BASELINE_TCN_SETTINGS,
    EQUITY_COUNT,
    PATCH_INPUT_WIDTH,
    RuntimeSettings,
    SLOW_FEATURE_COUNT,
    TABULAR_FEATURE_COUNT,
    architecture_for_model,
)
from brazil_rv.modeling.engine import (
    EvaluationObservations,
    _filter_evaluation_rows,
    checkpoint_payload,
    collect_validation_observations,
    compile_model,
    compile_training_objective,
    eager_training_objective,
    objective_loss,
    rank_huber_loss,
    run_effective_batch_update,
    soft_spearman_loss,
    train_one_epoch,
)
from brazil_rv.modeling.evaluate import load_current_neural_run
from brazil_rv.modeling.metrics import create_metric_table, primary_validation_score
from brazil_rv.modeling.model import build_neural_model
from brazil_rv.modeling.train import parse_args


class TinyRanker(nn.Module):
    model_name = "mlp"

    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(2, 3, bias=False)
        self.dropout_outputs: list[torch.Tensor] = []
        self.inference_flags: list[bool] = []

    def forward(
        self, features: torch.Tensor, equity_mask: torch.Tensor
    ) -> torch.Tensor:
        dropped = self.dropout(features)
        self.dropout_outputs.append(dropped.detach().clone())
        self.inference_flags.append(torch.is_inference_mode_enabled())
        return self.linear(dropped) * equity_mask[..., None]


class TinyModeRanker(nn.Module):
    model_name = "mlp"

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 3, bias=False)

    def forward(
        self, features: torch.Tensor, equity_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.linear(features) * equity_mask[..., None]

    def cuda(self, device: object = None) -> TinyModeRanker:
        del device
        return self


def _microbatch(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "tabular_features": torch.randn(1, 4, 2, generator=generator),
        "equity_mask": torch.ones(1, 4, dtype=torch.bool),
        "targets": torch.tensor(
            [
                [
                    [-0.75, -0.75, -0.75],
                    [-0.25, -0.25, -0.25],
                    [0.25, 0.25, 0.25],
                    [0.75, 0.75, 0.75],
                ]
            ]
        ),
        "label_mask": torch.ones(1, 4, 3, dtype=torch.bool),
    }


def _validation_observations() -> EvaluationObservations:
    shape = (2, 4, 3)
    return EvaluationObservations(
        sample_id=np.array([0, 1]),
        predictions=np.zeros(shape, dtype=np.float32),
        targets=np.zeros(shape, dtype=np.float32),
        raw_returns=np.zeros(shape, dtype=np.float32),
        label_mask=np.ones(shape, dtype=bool),
        date_idx=np.array([0, 0]),
        decision_idx=np.array([0, 1]),
    )


def test_objectives_match_current_group_aggregation() -> None:
    predictions = torch.tensor([[[-1.0, -0.5, 0.0], [0.0, 0.5, 1.0], [1.0, 0.0, -1.0]]])
    targets = predictions / 2
    mask = torch.ones_like(predictions, dtype=torch.bool)
    assert torch.isfinite(soft_spearman_loss(predictions, targets, mask, 0.50))
    difference = predictions - targets
    expected = (0.5 * difference.square()).mean()
    torch.testing.assert_close(rank_huber_loss(predictions, targets, mask), expected)
    torch.testing.assert_close(
        objective_loss(predictions, targets, mask, "rank_huber", None), expected
    )


def _compiled_runtime(backend: str) -> RuntimeSettings:
    return RuntimeSettings(
        microbatch_size=1,
        accumulation_steps=2,
        evaluation_batch_size=2,
        num_workers=0,
        compile_backend=backend,
    )


def _soft_spearman_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predictions = torch.tensor(
        [
            [
                [-1.0, 0.0, 0.5],
                [-1.0, 1.0, 0.0],
                [0.0, 2.0, -0.5],
                [1.0, 3.0, -0.5],
                [2.0, 4.0, 1.0],
            ],
            [
                [2.0, -1.0, 0.0],
                [1.0, -1.0, 1.0],
                [0.0, 0.0, 1.0],
                [-1.0, 1.0, 2.0],
                [-2.0, 2.0, 3.0],
            ],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [
            [
                [-2.0, 4.0, 0.0],
                [-1.0, 3.0, 1.0],
                [0.0, 2.0, 2.0],
                [1.0, 1.0, 3.0],
                [2.0, 0.0, 4.0],
            ],
            [
                [2.0, 0.0, 4.0],
                [1.0, 1.0, 3.0],
                [0.0, 2.0, 2.0],
                [-1.0, 3.0, 1.0],
                [-2.0, 4.0, 0.0],
            ],
        ],
        dtype=torch.float32,
    )
    mask = torch.ones_like(predictions, dtype=torch.bool)
    mask[0, 1, 1] = False
    mask[0, 4, 2] = False
    mask[1, 0, 0] = False
    mask[1, 3, 2] = False
    return predictions, targets, mask


def _assert_compiled_soft_spearman_parity(backend: str) -> None:
    predictions, targets, mask = _soft_spearman_inputs()
    eager_predictions = predictions.clone().requires_grad_(True)
    compiled_predictions = predictions.clone().requires_grad_(True)
    eager = eager_training_objective("soft_spearman", 0.50)
    compiled = compile_training_objective(
        "soft_spearman", 0.50, _compiled_runtime(backend)
    )
    eager_loss = eager(eager_predictions, targets, mask)
    compiled_loss = compiled(compiled_predictions, targets, mask)
    eager_loss.backward()
    compiled_loss.backward()
    assert torch.isfinite(eager_loss)
    assert torch.isfinite(compiled_loss)
    assert torch.isfinite(eager_predictions.grad).all()
    assert torch.isfinite(compiled_predictions.grad).all()
    torch.testing.assert_close(compiled_loss, eager_loss)
    torch.testing.assert_close(compiled_predictions.grad, eager_predictions.grad)


def test_compiled_soft_spearman_matches_eager_loss_and_gradients() -> None:
    _assert_compiled_soft_spearman_parity("aot_eager")


@pytest.mark.skipif(
    "inductor" not in torch._dynamo.list_backends(),
    reason="TorchInductor is unavailable",
)
def test_inductor_default_soft_spearman_matches_eager_loss_and_gradients() -> None:
    try:
        _assert_compiled_soft_spearman_parity("inductor")
    except InductorError as error:
        if "InvalidCxxCompiler" not in str(error):
            raise
        pytest.skip(f"TorchInductor compiler is unavailable: {error}")


def test_compiled_soft_spearman_matches_eager_sam_update() -> None:
    runtime = _compiled_runtime("aot_eager")
    torch.manual_seed(41)
    eager_model = TinyRanker()
    compiled_model = deepcopy(eager_model)
    eager_optimizer = torch.optim.AdamW(eager_model.parameters(), lr=1e-3)
    compiled_optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=1e-3)
    eager_objective = eager_training_objective("soft_spearman", 0.50)
    compiled_objective = compile_training_objective("soft_spearman", 0.50, runtime)
    warm_predictions = torch.zeros(1, 4, 3, requires_grad=True)
    compiled_objective(
        warm_predictions,
        _microbatch(1)["targets"],
        _microbatch(1)["label_mask"],
    ).backward()
    batches = [_microbatch(1), _microbatch(2)]
    rng_state = torch.get_rng_state()
    torch.set_rng_state(rng_state)
    eager_result = run_effective_batch_update(
        eager_model,
        batches,
        eager_optimizer,
        None,
        runtime,
        "sam_adamw",
        "soft_spearman",
        0.50,
        0.125,
        training_objective=eager_objective,
    )
    torch.set_rng_state(rng_state)
    compiled_result = run_effective_batch_update(
        compiled_model,
        batches,
        compiled_optimizer,
        None,
        runtime,
        "sam_adamw",
        "soft_spearman",
        0.50,
        0.125,
        training_objective=compiled_objective,
    )
    for eager_parameter, compiled_parameter in zip(
        eager_model.parameters(), compiled_model.parameters(), strict=True
    ):
        torch.testing.assert_close(compiled_parameter, eager_parameter)
    torch.testing.assert_close(compiled_result["loss_sum"], eager_result["loss_sum"])
    torch.testing.assert_close(
        compiled_result["gradient_norm"], eager_result["gradient_norm"]
    )
    assert compiled_result["loss_count"] == eager_result["loss_count"]
    assert compiled_result["backward_passes"] == eager_result["backward_passes"]


def test_compiled_training_updates_eager_validation_and_restores_training_mode() -> (
    None
):
    runtime = _compiled_runtime("aot_eager")
    torch.manual_seed(53)
    model = TinyModeRanker()
    compiled_model = compile_model(model, runtime)
    compiled_objective = compile_training_objective("soft_spearman", 0.50, runtime)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    training_batches = [_microbatch(5), _microbatch(6)]
    evaluation_batch = {
        name: torch.cat([batch[name] for batch in training_batches])
        for name in ("tabular_features", "equity_mask", "targets", "label_mask")
    }
    evaluation_batch.update(
        {
            "sample_valid_mask": torch.ones(2, dtype=torch.bool),
            "sample_id": torch.tensor([1, 0]),
            "raw_returns": torch.zeros_like(evaluation_batch["targets"]),
            "date_idx": torch.tensor([0, 0]),
            "decision_idx": torch.tensor([1, 0]),
        }
    )
    model.eval()
    with torch.inference_mode():
        before_update = model(
            evaluation_batch["tabular_features"],
            evaluation_batch["equity_mask"],
        )[[1, 0]].numpy()
    originals = [parameter.detach().clone() for parameter in model.parameters()]
    train_one_epoch(
        compiled_model,
        training_batches,
        optimizer,
        scheduler,
        runtime,
        "sam_adamw",
        "soft_spearman",
        0.50,
        0.125,
        compiled_objective,
    )
    assert any(
        not torch.equal(parameter, original)
        for parameter, original in zip(model.parameters(), originals, strict=True)
    )
    observations, _ = collect_validation_observations(
        model, [evaluation_batch], "soft_spearman", 0.50
    )
    with torch.inference_mode():
        current_predictions = model(
            evaluation_batch["tabular_features"],
            evaluation_batch["equity_mask"],
        )[[1, 0]].numpy()
    np.testing.assert_allclose(observations.predictions, current_predictions)
    assert not np.allclose(observations.predictions, before_update)
    assert not model.training
    train_one_epoch(
        compiled_model,
        training_batches,
        optimizer,
        scheduler,
        runtime,
        "sam_adamw",
        "soft_spearman",
        0.50,
        0.125,
        compiled_objective,
    )
    assert model.training


def test_production_routes_compiled_training_and_eager_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = TinyModeRanker()
    compiled_model = object()
    compiled_objective = object()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    training_models: list[object] = []
    validation_models: list[object] = []
    validation_parameters: list[torch.Tensor] = []
    checkpoint_models: list[object] = []

    class Sampler:
        def set_epoch(self, epoch: int) -> None:
            assert epoch == 1

    def fake_compile(current: nn.Module) -> object:
        assert current is model
        return compiled_model

    def fake_train(current: object, *_: object) -> dict[str, object]:
        training_models.append(current)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.0)
        return {
            "objective_loss": 0.1,
            "optimizer_steps": 1,
            "backward_passes": 2,
            "mean_gradient_norm": 0.2,
        }

    def fake_validation(
        current: nn.Module, *_: object
    ) -> tuple[EvaluationObservations, float]:
        validation_models.append(current)
        validation_parameters.append(next(current.parameters()).detach().clone())
        return _validation_observations(), 0.1

    def fake_checkpoint(
        current: nn.Module, *_: object, **__: object
    ) -> dict[str, object]:
        checkpoint_models.append(current)
        return {}

    class OutputFrame:
        def write_parquet(self, _: Path) -> None:
            return None

    monkeypatch.setattr(train_module, "MAX_EPOCHS", 1)
    monkeypatch.setattr(
        train_module,
        "create_training_loaders",
        lambda *_: ([object()], [object()], Sampler()),
    )
    monkeypatch.setattr(train_module, "build_neural_model", lambda *_: model)
    monkeypatch.setattr(
        train_module, "build_optimizer", lambda _: (optimizer, object())
    )
    monkeypatch.setattr(train_module, "build_scheduler", lambda *_: (scheduler, 1, 0))
    monkeypatch.setattr(train_module, "compile_model", fake_compile)
    monkeypatch.setattr(
        train_module, "compile_training_objective", lambda *_: compiled_objective
    )
    monkeypatch.setattr(train_module, "train_one_epoch", fake_train)
    monkeypatch.setattr(
        train_module, "collect_validation_observations", fake_validation
    )
    monkeypatch.setattr(train_module, "validation_primary_metric", lambda _: 0.25)
    monkeypatch.setattr(train_module, "checkpoint_payload", fake_checkpoint)
    monkeypatch.setattr(
        train_module,
        "summarize_evaluation_observations",
        lambda *_: ({}, [{"metric": 0.25}]),
    )
    monkeypatch.setattr(train_module, "_atomic_json", lambda *_: None)
    monkeypatch.setattr(train_module, "_atomic_torch_save", lambda *_: None)
    monkeypatch.setattr(train_module, "_write_history", lambda *_: None)
    monkeypatch.setattr(train_module.pl, "DataFrame", lambda *_: OutputFrame())
    rows = SimpleNamespace(height=512)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    original_parameter = next(model.parameters()).detach().clone()
    train_module._run_neural(
        train_module.parse_args([]),
        tmp_path,
        rows,
        rows,
        run_dir,
    )
    assert training_models == [compiled_model]
    assert validation_models == [model]
    torch.testing.assert_close(validation_parameters[0], original_parameter + 1.0)
    assert checkpoint_models == [model]


def test_benchmark_routes_compiled_updates_and_eager_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = TinyModeRanker()
    compiled_model = object()
    compiled_objective = object()
    update_models: list[object] = []
    validation_models: list[object] = []

    class Sampler:
        def set_epoch(self, epoch: int) -> None:
            assert epoch == 1

    def fake_update(current: object, *_: object) -> float:
        update_models.append(current)
        return 1.0

    def fake_validation(
        current: object, _: object
    ) -> tuple[EvaluationObservations, float, float, float, float]:
        validation_models.append(current)
        return _validation_observations(), 0.25, 3.0, 2.0, 1.0

    rows = SimpleNamespace(height=512)
    monkeypatch.setattr(benchmark_runtime, "set_seeds", lambda _: None)
    monkeypatch.setattr(benchmark_runtime, "resolve_feature_store", lambda: Path("."))
    monkeypatch.setattr(benchmark_runtime, "load_sample_index", lambda _: object())
    monkeypatch.setattr(benchmark_runtime, "select_sample_split", lambda *_: rows)
    monkeypatch.setattr(
        benchmark_runtime,
        "create_training_loaders",
        lambda *_: ([object()], object(), Sampler()),
    )
    monkeypatch.setattr(benchmark_runtime, "build_neural_model", lambda *_: model)
    monkeypatch.setattr(
        benchmark_runtime, "build_optimizer", lambda _: (object(), object())
    )
    monkeypatch.setattr(
        benchmark_runtime, "build_scheduler", lambda *_: (object(), 1, 0)
    )
    monkeypatch.setattr(benchmark_runtime, "compile_model", lambda *_: compiled_model)
    monkeypatch.setattr(
        benchmark_runtime,
        "compile_training_objective",
        lambda *_: compiled_objective,
    )
    monkeypatch.setattr(
        benchmark_runtime,
        "_compiled_soft_spearman_parity",
        lambda _: {
            "maximum_absolute_loss_difference": 0.0,
            "maximum_absolute_gradient_difference": 0.0,
            "absolute_tolerance": benchmark_runtime.PARITY_ABSOLUTE_TOLERANCE,
        },
    )
    monkeypatch.setattr(benchmark_runtime, "_timed_update", fake_update)
    monkeypatch.setattr(benchmark_runtime, "_timed_validation", fake_validation)
    monkeypatch.setattr(
        benchmark_runtime.torch.cuda, "reset_peak_memory_stats", lambda: None
    )
    monkeypatch.setattr(benchmark_runtime.torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(benchmark_runtime.torch.cuda, "max_memory_reserved", lambda: 0)
    result = benchmark_runtime.run_benchmark(parse_benchmark_args([]))
    assert update_models == [compiled_model] * (
        1 + benchmark_runtime.STEADY_UPDATE_COUNT
    )
    assert validation_models == [model, model]
    assert result["validation_observations_exact_match"] is True
    assert result["validation_primary_metric_exact_match"] is True


def test_benchmark_parity_detaches_before_scalar_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator_type = torch.Generator
    randn = torch.randn
    rand = torch.rand
    monkeypatch.setattr(
        benchmark_runtime.torch,
        "Generator",
        lambda **_: generator_type(),
    )

    def cpu_randn(*shape: object, **keywords: object) -> torch.Tensor:
        keywords.pop("device", None)
        return randn(*shape, **keywords)

    def cpu_rand(*shape: object, **keywords: object) -> torch.Tensor:
        keywords.pop("device", None)
        return rand(*shape, **keywords)

    monkeypatch.setattr(benchmark_runtime.torch, "randn", cpu_randn)
    monkeypatch.setattr(benchmark_runtime.torch, "rand", cpu_rand)
    monkeypatch.setattr(benchmark_runtime.torch.cuda, "synchronize", lambda: None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parity = benchmark_runtime._compiled_soft_spearman_parity(
            eager_training_objective("soft_spearman", 0.50)
        )
    assert parity["maximum_absolute_loss_difference"] == 0.0
    assert parity["maximum_absolute_gradient_difference"] == 0.0
    assert not any(
        "requires_grad" in str(warning.message) and "scalar" in str(warning.message)
        for warning in caught
    )


def test_benchmark_cli_accepts_only_retained_compile_modes() -> None:
    assert benchmark_runtime.COMPILE_MODES == (
        "default",
        "max-autotune-no-cudagraphs",
    )
    for mode in benchmark_runtime.COMPILE_MODES:
        assert parse_benchmark_args(["--compile-mode", mode]).compile_mode == mode
    for mode in ("reduce" + "-overhead", "max" + "-autotune"):
        with pytest.raises(SystemExit):
            parse_benchmark_args(["--compile-mode", mode])


@pytest.mark.parametrize(
    ("optimizer_variant", "fail_after", "rho", "message"),
    (
        ("adamw", 0, None, "Ordinary update"),
        ("sam_adamw", 0, 0.125, "SAM pass 1"),
        ("sam_adamw", 2, 0.125, "SAM pass 2"),
    ),
)
def test_nonfinite_accumulated_loss_is_rejected_per_effective_pass(
    optimizer_variant: str,
    fail_after: int,
    rho: float | None,
    message: str,
) -> None:
    model = TinyRanker()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    originals = [parameter.detach().clone() for parameter in model.parameters()]
    calls = 0

    def detached_nonfinite_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        label_mask: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal calls
        del targets, label_mask
        calls += 1
        finite = predictions.square().sum()
        if calls > fail_after:
            return finite + predictions.new_tensor(float("inf"))
        return finite

    with pytest.raises(
        FloatingPointError, match=f"{message} accumulated loss is non-finite"
    ):
        run_effective_batch_update(
            model,
            [_microbatch(1), _microbatch(2)],
            optimizer,
            None,
            _compiled_runtime("eager"),
            optimizer_variant,
            "rank_huber",
            None,
            rho,
            training_objective=detached_nonfinite_loss,
        )
    for parameter, original in zip(model.parameters(), originals, strict=True):
        torch.testing.assert_close(parameter, original, atol=0, rtol=0)
        assert parameter.grad is None


def test_finite_loss_check_preserves_ordinary_update() -> None:
    torch.manual_seed(43)
    model = TinyRanker()
    originals = [parameter.detach().clone() for parameter in model.parameters()]
    result = run_effective_batch_update(
        model,
        [_microbatch(1), _microbatch(2)],
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        None,
        _compiled_runtime("eager"),
        "adamw",
        "soft_spearman",
        0.50,
        None,
    )
    assert torch.isfinite(result["loss_sum"])
    assert torch.isfinite(result["gradient_norm"])
    assert any(
        not torch.equal(parameter, original)
        for parameter, original in zip(model.parameters(), originals, strict=True)
    )


def test_sam_restores_exactly_before_second_gradient_update_and_replays_rng() -> None:
    torch.manual_seed(11)
    model = TinyRanker()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    runtime = RuntimeSettings(
        microbatch_size=1, accumulation_steps=2, evaluation_batch_size=2, num_workers=0
    )
    original = {
        name: value.detach().clone() for name, value in model.named_parameters()
    }
    observed: dict[str, dict[str, torch.Tensor]] = {}

    def observer(stage: str, current: nn.Module) -> None:
        observed[stage] = {
            name: value.detach().clone() for name, value in current.named_parameters()
        }

    result = run_effective_batch_update(
        model,
        [_microbatch(1), _microbatch(2)],
        optimizer,
        None,
        runtime,
        "sam_adamw",
        "soft_spearman",
        0.50,
        0.125,
        sam_observer=observer,
    )
    assert set(result) == {
        "loss_sum",
        "loss_count",
        "gradient_norm",
        "backward_passes",
    }
    assert len(model.dropout_outputs) == 4
    torch.testing.assert_close(model.dropout_outputs[0], model.dropout_outputs[2])
    torch.testing.assert_close(model.dropout_outputs[1], model.dropout_outputs[3])
    assert any(
        not torch.equal(observed["perturbed_parameters"][name], original[name])
        for name in original
    )
    for name in original:
        torch.testing.assert_close(
            observed["second_gradients"][name], original[name], atol=0, rtol=0
        )
    assert any(
        not torch.equal(value, original[name])
        for name, value in model.named_parameters()
    )


def test_metric_ordering_preserves_turnover_and_daily_weighting() -> None:
    rng = np.random.default_rng(3)
    predictions = np.round(rng.normal(size=(4, 35, 3)), 1).astype(np.float32)
    targets = np.round(rng.normal(size=(4, 35, 3)), 1).astype(np.float32)
    returns = rng.normal(scale=0.01, size=(4, 35, 3)).astype(np.float32)
    mask = np.ones((4, 35, 3), dtype=bool)
    mask[0, :6, 0] = False
    mask[1, :5, 1] = False
    mask[2, :3, 2] = False
    dates = np.array([2, 1, 2, 1])
    decisions = np.array([1, 1, 0, 0])
    first, daily_first = create_metric_table(
        predictions, targets, returns, mask, dates, decisions
    )
    order = np.array([1, 3, 0, 2])
    second, daily_second = create_metric_table(
        predictions[order],
        targets[order],
        returns[order],
        mask[order],
        dates[order],
        decisions[order],
    )
    assert np.isclose(first["primary_score"], second["primary_score"], atol=1e-15)
    assert (
        abs(
            primary_validation_score(predictions, targets, mask, dates)
            - float(first["primary_score"])
        )
        <= 1e-12
    )
    assert np.isclose(
        first["mean_valid_sample_spearman_ic"],
        second["mean_valid_sample_spearman_ic"],
        atol=1e-15,
    )
    for left, right in zip(daily_first, daily_second, strict=True):
        assert left.keys() == right.keys()
        for key in left:
            if isinstance(left[key], float):
                assert np.isclose(left[key], right[key], atol=1e-15, equal_nan=True)
            else:
                assert left[key] == right[key]


def test_evaluation_padding_filters_only_real_rows() -> None:
    predictions = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
    batch = {
        "tabular_features": torch.ones(4, 2, 2),
        "equity_mask": torch.ones(4, 2, dtype=torch.bool),
        "sample_valid_mask": torch.tensor([True, True, False, False]),
        "sample_id": torch.tensor([5, 4, -1, -1]),
        "targets": torch.zeros(4, 2, 3),
        "raw_returns": torch.zeros(4, 2, 3),
        "label_mask": torch.ones(4, 2, 3, dtype=torch.bool),
        "date_idx": torch.tensor([1, 1, -1, -1]),
        "decision_idx": torch.tensor([1, 0, -1, -1]),
    }
    filtered = _filter_evaluation_rows(predictions, batch)
    assert filtered["predictions"].shape[0] == 2
    np.testing.assert_array_equal(filtered["sample_id"], [5, 4])
    model = TinyRanker()
    observations, loss = collect_validation_observations(
        model, [batch], "rank_huber", None
    )
    np.testing.assert_array_equal(observations.sample_id, [4, 5])
    assert model.inference_flags == [True]
    assert np.isfinite(loss)


def test_current_checkpoint_round_trip_uses_one_schema(tmp_path: Path) -> None:
    settings = BASELINE_TCN_SETTINGS
    architecture = architecture_for_model("tcn", settings)
    torch.manual_seed(29)
    model = build_neural_model("tcn", architecture, "selected")
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    store = tmp_path / "store"
    store.mkdir()
    payload = checkpoint_payload(
        model,
        optimizer,
        scheduler,
        "tcn",
        architecture,
        settings,
        "sam_adamw",
        "soft_spearman",
        0.50,
        0.125,
        29,
        3,
        0.01,
        store,
        "enabled",
        "selected",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.save(payload, run_dir / "best_checkpoint.pt")
    restored, checkpoint, restored_store = load_current_neural_run(run_dir)
    assert restored_store == store
    assert checkpoint["peer_features"]["mode"] == "selected"
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0)


def test_direct_cli_defaults_to_full_incumbent_and_removed_surfaces_are_rejected() -> (
    None
):
    args = parse_args([])
    assert (args.model, args.tcn_width, args.tcn_block) == ("tcn", 64, "swiglu")
    assert (args.peer_features, args.slow_routing, args.macro_temporal_routing) == (
        "selected",
        "late_only",
        "late_only",
    )
    assert (args.optimizer, args.temperature, args.sam_rho) == (
        "sam_adamw",
        0.50,
        0.125,
    )
    assert parse_benchmark_args([]).microbatch_size == 64


def test_each_retained_neural_family_has_a_finite_forward_pass() -> None:
    transformer = build_neural_model("temporal_only").eval()
    transformer_output = transformer(
        torch.zeros(1, EQUITY_COUNT, ABSOLUTE_PATCH_COUNT, PATCH_INPUT_WIDTH),
        torch.ones(1, EQUITY_COUNT, ABSOLUTE_PATCH_COUNT, dtype=torch.bool),
        torch.ones(1, EQUITY_COUNT, dtype=torch.bool),
        torch.zeros(1, EQUITY_COUNT, SLOW_FEATURE_COUNT),
        torch.tensor([32]),
    )
    tcn_architecture = architecture_for_model("tcn", BASELINE_TCN_SETTINGS)
    tcn = build_neural_model("tcn", tcn_architecture, "selected", equity_count=4).eval()
    instrument_count = 4 + 15
    tcn_output = tcn(
        torch.zeros(1, instrument_count, ABSOLUTE_PATCH_COUNT, PATCH_INPUT_WIDTH),
        torch.ones(1, instrument_count, ABSOLUTE_PATCH_COUNT, dtype=torch.bool),
        torch.ones(1, instrument_count, dtype=torch.bool),
        torch.zeros(1, instrument_count, SLOW_FEATURE_COUNT),
        torch.tensor([32]),
        torch.zeros(1, 4, 6),
    )
    mlp = build_neural_model("mlp").eval()
    mlp_output = mlp(
        torch.zeros(1, EQUITY_COUNT, TABULAR_FEATURE_COUNT),
        torch.ones(1, EQUITY_COUNT, dtype=torch.bool),
    )
    assert transformer_output.shape == (1, EQUITY_COUNT, 3)
    assert tcn_output.shape == (1, 4, 3)
    assert mlp_output.shape == (1, EQUITY_COUNT, 3)
    assert all(
        torch.isfinite(output).all()
        for output in (transformer_output, tcn_output, mlp_output)
    )
